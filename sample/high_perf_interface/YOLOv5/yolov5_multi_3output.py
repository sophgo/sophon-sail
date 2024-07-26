import sophon.sail as sail
import numpy as np
import threading
import time
import os
import cv2
import queue
from multiprocessing import Process

class MultiDecoderThread(object):
    def __init__(self, tpu_id, video_list, resize_type:sail.sail_resize_type, max_que_size:int, loop_count:int):
        self.channel_list = {}
        self.tpu_id = tpu_id
        self.break_flag = False
        self.resize_type = resize_type
        self.multiDecoder = sail.MultiDecoder(15, tpu_id)
        self.multiDecoder.set_local_flag(True)
        self.yolov5_post = sail.algo_yolov5_post_3output([[4, 3, 80, 80, 85],[4, 3, 40, 40, 85],[4, 3, 20, 20, 85]],640,640,10)

        self.loop_count = loop_count

        self.post_que = queue.Queue(max_que_size)
        self.image_que = queue.Queue(max_que_size)

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

    def InitProcess(self, bmodel_name, process_id):
        self.process_id = process_id
        self.engine_image_pre_process = sail.EngineImagePreProcess(bmodel_name, self.tpu_id, 0)
        self.engine_image_pre_process.InitImagePreProcess(self.resize_type, True, 10, 10)
        self.engine_image_pre_process.SetPaddingAtrr()
        self.engine_image_pre_process.SetConvertAtrr(self.alpha_beta)
        self.net_w = self.engine_image_pre_process.get_input_width()
        self.net_h = self.engine_image_pre_process.get_input_height()

        output_name = self.engine_image_pre_process.get_output_names()[0]
        self.batch_size = self.engine_image_pre_process.get_output_shape(output_name)[0]
        thread_preprocess = threading.Thread(target=self.decoder_and_pushdata, args=(self.channel_list, self.multiDecoder, self.engine_image_pre_process))
        thread_inference = threading.Thread(target=self.Inferences_thread, args=(self.resize_type, self.tpu_id, self.post_que, self.image_que))
        thread_postprocess = threading.Thread(target=self.post_process, args=(self.post_que, 0.2, 0.5))
        thread_drawresult = threading.Thread(target=self.draw_result_thread, args=(self.image_que,))
        thread_drawresult.start()
        thread_postprocess.start()
        thread_preprocess.start()
        thread_inference.start()
       
    
    def decoder_and_pushdata(self,channel_list, multi_decoder, PreProcessAndInference):
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
                    image_index += 1
                    PreProcessAndInference.PushImage(int(key),image_index, bmimg)
                    total_count += 1
                    if total_count == 2000:
                        total_time = (time.time()-time_start)*1000
                        avg_time = total_time/total_count
                        print("########################avg time: {:.2f}".format(avg_time))
                        total_count = 0

                else:
                    time.sleep(0.01)

        print("decoder_and_pushdata thread exit!")

    def Inferences_thread(self, resize_type:sail.sail_resize_type, device_id:int, post_queue:queue.Queue, img_queue:queue.Queue):
        while True:
            if self.get_exit_flag():
                break
            start_time = time.time()
            output_tensor_map, ost_images, channel_list ,imageidx_list, padding_atrr = self.engine_image_pre_process.GetBatchData(True)
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
            post_queue.put([output_tensor_map,
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


    def post_process(self, post_quque:queue.Queue, dete_threshold:float, nms_threshold:float):
        while (True):
            if self.get_exit_flag():
                break
            if post_quque.empty():
                time.sleep(0.01)
                continue
            output_tensor_map, channels ,imageidxs, ost_ws, ost_hs, padding_atrrs = post_quque.get(True)

            dete_thresholds = np.ones(len(channels),dtype=np.float32)
            nms_thresholds = np.ones(len(channels),dtype=np.float32)
            dete_thresholds = dete_threshold*dete_thresholds
            nms_thresholds = nms_threshold*nms_thresholds
            while True:
                if self.get_exit_flag():
                    break
                ret = self.yolov5_post.push_data(channels, imageidxs, output_tensor_map, dete_thresholds, nms_thresholds, ost_ws, ost_hs, padding_atrrs)
                if ret == 0:
                    break
                else:
                    print("push_data failed, ret: {}".format(ret))
                    time.sleep(0.01)
                break
        print("post_process thread exit!")
    
    def draw_result_thread(self, img_queue:queue.Queue):
        handle = sail.Handle(self.tpu_id)
        bmcv = sail.Bmcv(handle)
        total_count = 0
        start_time = time.time()
        while (True):
            if img_queue.empty():
                time.sleep(0.01)
                continue
            ocv_image = img_queue.get(True)
            objs, channel, image_idx = self.yolov5_post.get_result_npy()
            # for obj in objs:
            #     bmcv.rectangle(ocv_image, obj[0], obj[1], obj[2]-obj[0], obj[3]-obj[1],(0,0,255),2)
            # image = sail.BMImage(handle,ocv_image.height(),ocv_image.width(),sail.Format.FORMAT_YUV420P,sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE)
            # bmcv.convert_format(ocv_image,image)
            # for obj in objs:
            #     txt_d = "{}".format(obj[4])
            #     bmcv.putText(image, txt_d , obj[0], obj[1], [0,0,255], 1.4, 2)
            # bmcv.imwrite("{}_{}.jpg".format(channel,image_idx),image)

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

def process_demo(tpu_id, max_que_size, video_name_list, bmodel_name, loop_count, process_id):
    process =  MultiDecoderThread(tpu_id, video_name_list, sail.sail_resize_type.BM_RESIZE_TPU_NEAREST, max_que_size, loop_count)
    process.InitProcess(bmodel_name,process_id)


if __name__ == '__main__':
    tpu_id = 0
    decoder_count = 4           #每个进程解码的路试
    max_que_size = 10           #缓存的大小
    loop_count = 1000           #每个进程处理图片的数量，处理完毕之后会退出。

    video_path = "/data/test_yolov5_multi/"
    video_name_list = [ "N-vnp-1min-2.mp4",
                        "N-vnp-1min-2.mp4",
                        "N-vnp-1min-2.mp4",
                        "N-vnp-1min-2.mp4"]
    
    bmodel_name = "../yolov5s_v6.1_3output_int8_4b.bmodel"
    video_list = []
    for i in range(0,decoder_count):
        video_list.append(os.path.join(video_path,video_name_list[i%len(video_name_list)]))


    p0 = Process(target=process_demo, args=(tpu_id, max_que_size, video_list, bmodel_name, loop_count, 1001))
    p1 = Process(target=process_demo, args=(tpu_id, max_que_size, video_list, bmodel_name, loop_count, 1002))
    p2 = Process(target=process_demo, args=(tpu_id, max_que_size, video_list, bmodel_name, loop_count, 1003))
    p0.start()
    p1.start()
    p2.start()

