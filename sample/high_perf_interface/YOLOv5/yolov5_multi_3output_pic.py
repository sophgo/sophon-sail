import sophon.sail as sail
import numpy as np
import threading
import time
import os
import cv2
import queue
from multiprocessing import Process

def get_imagenames(image_path):
    file_list = os.listdir(image_path)
    imagenames = []
    for file_name in file_list:
        ext_name = os.path.splitext(file_name)[-1]
        if ext_name in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
            imagenames.append(os.path.join(image_path,file_name))
    return imagenames

class ImgDecoderThread(object):
    def __init__(self, tpu_id, image_name_list, resize_type:sail.sail_resize_type, max_que_size:int):
        self.resize_type = resize_type
        self.tpu_id = tpu_id
        self.image_name_list = image_name_list
        self.resize_type = resize_type
        self.yolov5_post = sail.algo_yolov5_post_3output([[4, 3, 80, 80, 85],[4, 3, 40, 40, 85],[4, 3, 20, 20, 85]],640,640,10)

        self.post_que = queue.Queue(max_que_size)
        self.image_que = queue.Queue(max_que_size)

        self.alpha_beta = (1.0/255,0),(1.0/255,0),(1.0/255,0)

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
        if(len(self.image_name_list)%self.batch_size != 0):
            sub_num = self.batch_size - len(self.image_name_list)%self.batch_size
            for i in range(sub_num):
                self.image_name_list.append(self.image_name_list[0])

        self.run_count = int(len(self.image_name_list)/self.batch_size)
        self.loop_count = len(self.image_name_list)
        
        thread_decoder = threading.Thread(target=self.decoder_and_pushdata, 
            args=(process_id,self.tpu_id, self.image_name_list, self.engine_image_pre_process))
        
        thread_inference = threading.Thread(target=self.Inferences_thread, 
            args=(self.run_count, self.resize_type, self.tpu_id, self.post_que, self.image_que))
        
        thread_postprocess = threading.Thread(target=self.post_process, 
            args=(self.loop_count, self.post_que, 0.2, 0.5))

        thread_drawresult = threading.Thread(target=self.draw_result_thread, 
            args=(self.image_que,))

        thread_decoder.start()
        thread_inference.start()
        thread_postprocess.start()
        thread_drawresult.start()

    
    def decoder_and_pushdata(self, process_id, tpu_id, image_name_list, PreProcessAndInference):
        time_start = time.time()
        handle = sail.Handle(tpu_id)
        for image_index, image_name in enumerate(image_name_list):
            # print(image_name)
            decoder = sail.Decoder(image_name,True,tpu_id)
            bmimg = decoder.read(handle)
            while(PreProcessAndInference.PushImage(process_id,image_index, bmimg) != 0):
                print("Porcess[{}]:[{}/{}]PreProcessAndInference Thread Full, sleep: 10ms!".format(
                    process_id,image_index,len(image_name_list)))
                time.sleep(0.01)
        using_time = time.time()-time_start
        print("decoder_and_pushdata thread exit, time use: {:.2f}s,avg: {:.2f}ms".format(
            using_time,using_time/len(image_name_list)*1000))


    def Inferences_thread(self, loop_count, resize_type:sail.sail_resize_type, device_id:int, post_queue:queue.Queue, img_queue:queue.Queue):
        for i in range(loop_count):
            start_time = time.time()
            output_tensor_map, ost_images, channel_list ,imageidx_list, padding_atrr = self.engine_image_pre_process.GetBatchData(True)
            width_list = []
            height_list= []
            for index, channel in enumerate(channel_list):
                width_list.append(ost_images[index].width())
                height_list.append(ost_images[index].height())

            while post_queue.full():
                time.sleep(0.01)
                continue
            post_queue.put([output_tensor_map,
                            channel_list,
                            imageidx_list,
                            width_list, 
                            height_list, 
                            padding_atrr],False)
            
            for index, channel in enumerate(channel_list):
                while img_queue.full():
                    time.sleep(0.01)
                    continue 
                img_queue.put(ost_images[index])

            end_time = time.time()
            # print("GetBatchData time use: {:.2f} ms".format((end_time-start_time)*1000))
        
        print("Inferences_thread thread exit!")
    
    def post_process(self,loop_count, post_quque:queue.Queue, dete_threshold:float, nms_threshold:float):
        cout = 0
        while(cout < loop_count/self.batch_size):
            if post_quque.empty():
                time.sleep(0.01)
                continue
            output_tensor_map, channels ,imageidxs, ost_ws, ost_hs, padding_atrrs = post_quque.get(True)

            dete_thresholds = np.ones(len(channels),dtype=np.float32)
            nms_thresholds = np.ones(len(channels),dtype=np.float32)
            dete_thresholds = dete_threshold*dete_thresholds
            nms_thresholds = nms_threshold*nms_thresholds
            while True:
                ret = self.yolov5_post.push_data(channels, imageidxs, output_tensor_map, dete_thresholds, nms_thresholds, ost_ws, ost_hs, padding_atrrs)
                if ret == 0:
                    cout +=1
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
        print("Total time use: {:.2f}ms".format(time_use))
        print("Avg time use: {:.2f}ms".format(avg_time))
        print("Process {}: {:.2f} FPS".format(self.process_id, 1000/avg_time))
        print("Result thread exit!")
        
def process_demo(tpu_id, image_name_list, bmodel_name, process_id, max_que_size):
    process =  ImgDecoderThread(tpu_id, image_name_list, sail.sail_resize_type.BM_RESIZE_TPU_NEAREST, max_que_size)
    process.InitProcess(bmodel_name,process_id)

if __name__ == '__main__':
    tpu_id = 0
    image_path = "/home/sophgo/workpath/sophon-demo/sample/YOLOv5/datasets/coco/val2017"
    image_names = get_imagenames(image_path)
    process_count = 2           #进程数
    max_que_size = 10           #缓存的大小


    image_count = len(image_names)
    each_count = int(image_count/process_count)

    image_name_list = []
    for i in range(process_count-1):
        image_name_list.append(image_names[i*each_count:(i+1)*each_count])
    image_name_list.append(image_names[(process_count-1)*each_count:])

    for image_name_l in image_name_list:
        print(len(image_name_l))
        print(image_name_l[0],image_name_l[-1])

    bmodel_name = "/home/sophgo/workpath/data/bmodel/1684/yolov5s_v6.1_3output_int8_4b.bmodel"

    process_list = []
    for i in range(0,process_count):
        p = Process(target=process_demo, args=(tpu_id, image_name_list[i], bmodel_name, i, max_que_size))
        process_list.append(p)
    
    for p in process_list:
        p.start()

