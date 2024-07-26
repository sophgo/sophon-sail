# 本例程适用于单芯片设备实现多模型的推理
import sophon.sail as sail
import numpy as np
import threading
import time
import os
import cv2
import queue
from multiprocessing import Process

class MultiDecoderThread(object):
    def __init__(self, tpu_id, video_list, bmodel_name_list, postprocess_atrr_list, resize_type:sail.sail_resize_type, max_que_size:int, loop_count:int):
        self.channel_list = {}
        self.tpu_id = tpu_id
        self.break_flag = False
        self.resize_type = resize_type
        self.multiDecoder = sail.MultiDecoder(15, tpu_id)
        self.multiDecoder.set_local_flag(True)
        self.bmodel_num = len(bmodel_name_list)
        if(self.bmodel_num!=postprocess_atrr_list):
            print("Error: the length of bmodel_name_list and postprocess_atrr_list is not same!")
            exit
        self.yolov5_post_list = [sail.algo_yolov5_post_1output(*postprocess_atrr) for postprocess_atrr in postprocess_atrr_list]
        self.loop_count = loop_count

        self.post_que_list = [queue.Queue(max_que_size) for i in range(self.bmodel_num)]
        self.image_que_list = [queue.Queue(max_que_size) for i in range(self.bmodel_num)]

        self.exit_flag = False
        self.flag_lock = threading.Lock()
        

        for video_name in video_list:
            channel_index = self.multiDecoder.add_channel(video_name,1)
            print("Add Channel[{}]: {}".format(channel_index,video_name))
            self.channel_list[channel_index] = video_name

        self.alpha_beta = (1.0/255,0),(1.0/255,0),(1.0/255,0)
    
    def get_exit_flag(self):
        self.flag_lock.acquire()
        flag_temp = self.exit_flag
        self.flag_lock.release()
        return flag_temp

    def EngineImagePreProcess(self, bmodel_name, tpu_id, resize_type, alpha_beta):
        engine_image_pre_process = sail.EngineImagePreProcess(bmodel_name, tpu_id, 0)
        engine_image_pre_process.InitImagePreProcess(resize_type, True, 10, 10)
        engine_image_pre_process.SetPaddingAtrr()
        engine_image_pre_process.SetConvertAtrr(alpha_beta)
        return engine_image_pre_process

    def InitProcess(self, bmodel_name_list, process_id):
        self.process_id = process_id
        self.engine_image_pre_process_list = []
        for bmodel_name in bmodel_name_list:
            self.engine_image_pre_process_list.append(self.EngineImagePreProcess(bmodel_name, self.tpu_id, self.resize_type, self.alpha_beta))

        thread_preprocess = threading.Thread(target=self.decoder_and_pushdata, args=(self.channel_list, self.multiDecoder, self.engine_image_pre_process_list))
        thread_inference_list = [threading.Thread(target=self.Inferences_thread, args=(idx, self.resize_type, self.tpu_id, self.post_que_list[idx], self.image_que_list[idx])) for idx in range(self.bmodel_num)]
        thread_postprocess_list = [threading.Thread(target=self.post_process, args=(idx, self.post_que_list[idx], 0.2, 0.5)) for idx in range(self.bmodel_num)]
        thread_drawresult_list = [threading.Thread(target=self.draw_result_thread, args=(idx, self.image_que_list[idx],)) for idx in range(self.bmodel_num)]
        for i in range(self.bmodel_num):
            thread_drawresult_list[i].start()
            thread_postprocess_list[i].start()
            thread_inference_list[i].start()
        thread_preprocess.start()
       
    
    def decoder_and_pushdata(self,channel_list, multi_decoder, PreProcessAndInferenceList):
        handle = sail.Handle(self.tpu_id)
        bmcv = sail.Bmcv(handle)
        image_index = 0
        time_start = time.time()
        total_count = 0
        while True:
            if self.get_exit_flag():
                    break
            for key in channel_list:
                if self.get_exit_flag():
                    break
                bmimg = sail.BMImage()
                ret = multi_decoder.read(int(key),bmimg)
                if ret == 0:
                    # 在此处进行数据的分发
                    for i in range(self.bmodel_num):
                        if i >=1 :
                            bmimg_temp = sail.BMImage(handle,bmimg.height(),bmimg.width(),bmimg.format(),bmimg.dtype())
                            bmcv.image_copy_to(bmimg,bmimg_temp,0,0)
                            image_index += 1
                            PreProcessAndInferenceList[i].PushImage(int(key),image_index, bmimg_temp)
                    
                    image_index += 1
                    PreProcessAndInferenceList[0].PushImage(int(key),image_index, bmimg)
                    
                    total_count += 1
                    if total_count == 2000:
                        total_time = (time.time()-time_start)*1000
                        avg_time = total_time/total_count
                        print("########################avg time: {:.2f}".format(avg_time))
                        total_count = 0

                else:
                    time.sleep(0.01)

        print("decoder_and_pushdata thread exit!")

    def Inferences_thread(self, thread_idx, resize_type:sail.sail_resize_type, device_id:int, post_queue:queue.Queue, img_queue:queue.Queue):
        while True:
            if self.get_exit_flag():
                break
            start_time = time.time()
            output_tensor_map, ost_images, channel_list ,imageidx_list, padding_atrr = self.engine_image_pre_process_list[thread_idx].GetBatchData(True)
            tensor_with_name = output_tensor_map[0]
            width_list = []
            height_list= []
            for index, channel in enumerate(channel_list):
                width_list.append(ost_images[index].width())
                height_list.append(ost_images[index].height())

            while post_queue.full():
                time.sleep(0.01)
                if self.get_exit_flag():
                    break
                continue
            post_queue.put([tensor_with_name,
                            channel_list,
                            imageidx_list,
                            width_list, 
                            height_list, 
                            padding_atrr],False)
            
            for index, channel in enumerate(channel_list):
                if self.get_exit_flag():
                    break
                while img_queue.full():
                    time.sleep(0.01)
                    if self.get_exit_flag():
                        break
                    continue 
                img_queue.put(ost_images[index])

            end_time = time.time()
            print("GetBatchData time use: {:.2f} ms".format((end_time-start_time)*1000))
        
        print("Inferences_thread thread exit!")


    def post_process(self, thread_idx, post_quque:queue.Queue, dete_threshold:float, nms_threshold:float):
        while (True):
            if self.get_exit_flag():
                break
            if post_quque.empty():
                time.sleep(0.01)
                continue
            tensor_with_name, channels ,imageidxs, ost_ws, ost_hs, padding_atrrs = post_quque.get(True)
            dete_thresholds = np.ones(len(channels),dtype=np.float32)
            nms_thresholds = np.ones(len(channels),dtype=np.float32)
            dete_thresholds = dete_threshold*dete_thresholds
            nms_thresholds = nms_threshold*nms_thresholds
            while True:
                if self.get_exit_flag():
                    break
                ret = self.yolov5_post_list[thread_idx].push_data(channels, imageidxs, tensor_with_name, dete_thresholds, nms_thresholds, ost_ws, ost_hs, padding_atrrs)
                if ret == 0:
                    break
                else:
                    print("push_data failed, ret: {}".format(ret))
                    time.sleep(0.01)
                break
        print("post_process thread exit!")
    
    def draw_result_thread(self, thread_idx, img_queue:queue.Queue):
        handle = sail.Handle(self.tpu_id)
        bmcv = sail.Bmcv(handle)
        total_count = 0
        color_list = [
            [0,0,255],
            [0,255,0],
            [255,0,0],
        ]
        start_time = time.time()
        while (True):
            if img_queue.empty():
                time.sleep(0.01)
                continue
            ocv_image = img_queue.get(True)
            objs, channel, image_idx = self.yolov5_post_list[thread_idx].get_result_npy()
            for obj in objs:
                bmcv.rectangle(ocv_image, obj[0], obj[1], obj[2]-obj[0], obj[3]-obj[1],color_list[thread_idx%len(color_list)],2)
            image = sail.BMImage(handle,ocv_image.height(),ocv_image.width(),sail.Format.FORMAT_YUV420P,sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE)
            bmcv.convert_format(ocv_image,image)
            for obj in objs:
                txt_d = "{}".format(obj[4])
                bmcv.putText(image, txt_d , obj[0], obj[1], color_list[thread_idx%len(color_list)], 1.4, 2)
            bmcv.imwrite("./images/thread{}_channel{}_idx{}.jpg".format(thread_idx,channel,image_idx),image)

            total_count += 1
            if self.loop_count <=  total_count:
                break
        end_time = time.time()
        time_use = (end_time-start_time)*1000
        avg_time = time_use/self.loop_count

        print("Total images: {}".format(self.loop_count))
        print("Total time use: {}".format(time_use))
        print("Avg time use: {}".format(avg_time))
        print("{}: {} FPS".format(self.process_id, 1000/avg_time))
        print("Result thread exit!")
        
        self.flag_lock.acquire()
        self.exit_flag = True
        self.flag_lock.release()

def process_demo(tpu_id, max_que_size, video_name_list, bmodel_name_list, postprocess_atrr_list, loop_count, process_id):
    process =  MultiDecoderThread(tpu_id, video_name_list,bmodel_name_list, postprocess_atrr_list, sail.sail_resize_type.BM_RESIZE_TPU_NEAREST, max_que_size, loop_count)
    process.InitProcess(bmodel_name_list,process_id)


if __name__ == '__main__':
    tpu_id = 0
    decoder_count = 4
    max_que_size = 10
    loop_count = 1000

    video_path = "./video"
    video_name_list = [ "001.mp4",
                        "002.mp4"]
    
    bmodel_name_list = ["yolov5s_int8_4b_one.bmodel",
                        "yolov5s_int8_4b_two.bmodel"]
    
    postprocess_atrr_list = [([4, 25200, 85],640,640,10), 
                             ([4, 25200, 85],640,640,10)]
    
    video_list = []
    for i in range(0,decoder_count):
        video_list.append(os.path.join(video_path,video_name_list[i%len(video_name_list)]))


    
    p0 = Process(target=process_demo, args=(tpu_id, max_que_size, video_list, bmodel_name_list, postprocess_atrr_list, loop_count, 1001))
    # p1 = Process(target=process_demo, args=(tpu_id, max_que_size, video_list, bmodel_name, loop_count, 1002))
    # p2 = Process(target=process_demo, args=(tpu_id, max_que_size, video_list, bmodel_name, loop_count, 1003))
    p0.start()
    # p1.start()
    # p2.start()

