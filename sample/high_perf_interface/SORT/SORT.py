import sophon.sail as sail
import numpy as np
import threading
import time
import os
import cv2
import queue
import argparse
from multiprocessing import Process,Value

# batch_size =1 时，效果最佳
# 更多详细信息，请参考 readme.ms


# 文件夹名称
def get_imagenames(image_path):
    file_list = os.listdir(image_path)
    imagenames = []
    for file_name in file_list:
        ext_name = os.path.splitext(file_name)[-1]
        if ext_name in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
            imagenames.append(os.path.join(image_path,file_name))
    return imagenames

# 绘制跟踪框
def plot_bboxes(image, bboxes, line_thickness=None):
    image = image.asmat()
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        color = (0, 255, 0)
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(str(cls_id), 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return image

class SortThread(object):
    def __init__(self, tpu_id, image_name_list, resize_type:sail.sail_resize_type, max_que_size:int, conf_thres:float, nms_thres:float):

        # 前处理resize方法
        self.resize_type = resize_type

        # tpu_id 
        self.tpu_id = tpu_id

        # 按顺序排序的图片文件路径
        self.image_name_list = image_name_list
        
        # YOLOv5检测阈值
        self.conf_thres = conf_thres

        # NMS阈值
        self.nms_thres = nms_thres

        # 追踪检测结果
        self.results_list = []

        # yolov5 后处理检测队列
        self.post_que = queue.Queue(max_que_size)

        # 输入图像队列，用于绘制图像
        self.image_que = queue.Queue(max_que_size)

        # 检测结果队列
        self.objs_que = queue.Queue(max_que_size)

        # 预处理参数
        self.alpha_beta = (1.0/255,0),(1.0/255,0),(1.0/255,0)

        # 主进程推理计数
        self.loop_count = len(image_name_list)


    def InitProcess(self, detector_path, process_id):

        # 多路输入时使用，进程id
        self.process_id = process_id

        # 初始化前处理+推理接口
        self.engine_image_pre_process = sail.EngineImagePreProcess(detector_path, self.tpu_id, 0)

        # 设置前处理resize方式
        self.engine_image_pre_process.InitImagePreProcess(self.resize_type, True, 10, 10)
        # 设置填充方法
        self.engine_image_pre_process.SetPaddingAtrr()
        # 设置前处理方法
        self.engine_image_pre_process.SetConvertAtrr(self.alpha_beta)
        # 获取网络输入的size
        self.net_w = self.engine_image_pre_process.get_input_width()
        self.net_h = self.engine_image_pre_process.get_input_height()

        # 获取batch_size
        output_name = self.engine_image_pre_process.get_output_names()[0]
        self.batch_size = self.engine_image_pre_process.get_output_shape(output_name)[0]

        # 基于cpu优化的YOLOv5后处理接口
        self.yolov5_post = sail.algo_yolov5_post_cpu_opt_async([[self.batch_size, 3, 80, 80, 85],[self.batch_size, 3, 40, 40, 85],[self.batch_size, 3, 20, 20, 85]],640,640,10)
        
        # 追踪算法SORT post
        self.sort_post = sail.sort_tracker_controller_async(max_iou_distance=0.7, max_age=70, n_init=3)

        # 记录结果数据
        mot_saver = open("results/mot_eval/process_{}.txt".format(self.process_id), "w") 

        # 追踪的总目标数
        self.total_target_num = 0
        
        # 耗时记录，开始记录点
        start_time = time.time()


        # 解码线程，获取数据后，直接push到前处理接口的缓冲队列
        # 前处理+推理接口底层使用C++实现
        thread_decoder = threading.Thread(target=self.decoder_and_pushdata, 
            args=(process_id,self.tpu_id, self.image_name_list, self.engine_image_pre_process))
        thread_inference = threading.Thread(target=self.Inferences_thread, 
            args=(self.loop_count, self.post_que, self.image_que))
        thread_postprocess = threading.Thread(target=self.post_process, 
            args=(self.loop_count, self.post_que, 0.5, 0.5))
        thread_sort_process = threading.Thread(target=self.sort_process,args=())
        thread_draw_result = threading.Thread(target=self.draw_result, args=(self.image_que, mot_saver))

        thread_decoder.start()
        thread_inference.start()
        thread_postprocess.start()
        thread_sort_process.start()
        thread_draw_result.start()

        thread_decoder.join()
        thread_inference.join()
        thread_postprocess.join()
        thread_sort_process.join()
        thread_draw_result.join()

        mot_saver.close()

        end_time = time.time()
        total_time = end_time-start_time

        # 输出日志
        print("-"*30)
        print("Process {} total images: {}".format(self.process_id, self.loop_count))
        print("Process {} total time use: {:.2f}ms".format(self.process_id, total_time*1000))
        print("Process {} total tracking target: {}".format(self.process_id, self.total_target_num))
        print("Process {} avg single target for tracking time use: {:.2f}ms".format(self.process_id, total_time/self.total_target_num*1000))
        print("Process {} {:.2f} FPS".format(self.process_id, self.loop_count/total_time))
        print("-"*30)

        with total_fps.get_lock():
            total_fps.value += self.loop_count/total_time
            
    def decoder_and_pushdata(self, process_id, tpu_id, image_name_list, PreProcessAndInference):
        time_start = time.time()
        handle = sail.Handle(tpu_id)
        for image_index, image_name in enumerate(image_name_list):

            decoder = sail.Decoder(image_name,True,tpu_id)
            bmimg = sail.BMImage()
            ret = decoder.read(handle, bmimg)  
            while(PreProcessAndInference.PushImage(process_id,image_index, bmimg) != 0):
                print("Process[{}]:[{}/{}]PreProcessAndInference Thread Full, sleep: 10ms!".format(
                    process_id,image_index,len(image_name_list)))
                time.sleep(0.01)
            print(f"push_data {image_index}:{image_name}")
        using_time = time.time()-time_start
        print("decoder_and_pushdata thread exit, time use: {:.2f}s,avg: {:.2f}ms".format(using_time,using_time/len(image_name_list)*1000))

    def Inferences_thread(self, loop_count, post_queue:queue.Queue, img_queue:queue.Queue):
        for i in range(loop_count):
            output_tensor_map, ost_images, channel_list ,imageidx_list, padding_atrr = self.engine_image_pre_process.GetBatchData(True)
            width_list = []
            height_list= []
            for index, channel in enumerate(channel_list):
                width_list.append(ost_images[index].width())
                height_list.append(ost_images[index].height())

            while post_queue.full():
                print("-- Inferences_thread Thread Full, sleep: 10ms!")
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
                    print()
                    time.sleep(0.01)
                    continue 
                img_queue.put(ost_images[index])
  
    # YOLOv5 后处理
    def post_process(self,loop_count, post_queue:queue.Queue, dete_threshold:float, nms_threshold:float):
        cout = 0
        while(cout < loop_count/self.batch_size):
            output_tensor_map, channels ,imageidxs, ost_ws, ost_hs, padding_atrrs = post_queue.get(True)

            dete_thresholds = np.ones(len(channels),dtype=np.float32)
            nms_thresholds = np.ones(len(channels),dtype=np.float32)
            dete_thresholds = dete_threshold*dete_thresholds
            nms_thresholds = nms_threshold*nms_thresholds

            while True:
                ret = self.yolov5_post.push_data(channels, imageidxs, output_tensor_map, dete_thresholds, nms_thresholds, ost_ws, ost_hs, padding_atrrs)
                if ret == 0:
                    break
                else:
                    print("YOLOv5 post_process push_data failed, ret: {}".format(ret))
                    time.sleep(0.01)
            cout +=1

    # 追踪算法SORT
    def sort_process(self):
        # 循环计数
        for i in range(self.loop_count):
            # 从内部队列中获取YOLOv5检测结果
            objs, channel, image_idx = self.yolov5_post.get_result_npy()

            # 异步追踪算法SORT接口
            while True:
                ret = self.sort_post.push_data(objs)
                if ret == 0:
                    break
                else:
                    print("== SORT process push_data failed, ret: {}".format(ret))
                    time.sleep(0.01)

    # 存图、存结果接口
    def draw_result(self, org_img: queue.Queue, mot_saver):
        # 记录计数
        for i in range(self.loop_count):
            # 从异步处理接口SORT获取追踪信息
            result_with_id = self.sort_post.get_result_npy()
            self.total_target_num += len(result_with_id) 
            
            # 获取原始的图片
            org_bmimg = org_img.get(True)

            bboxes2draw = []
          
            # 存txt结果
            for value in list(result_with_id):
                x1, y1, x2, y2, cls_, score, track_id = value
                if draw_flag:
                    bboxes2draw.append((x1, y1, x2, y2, cls_, track_id))
                if dump_result:
                    save_str = "{},{},{},{},{},{},1,-1,-1,-1\n".format(i+1, track_id, x1, y1, x2 - x1, y2 - y1)
                    ret = mot_saver.write(save_str)

            # 存图片结果
            if draw_flag:
                save_img = plot_bboxes(org_bmimg, bboxes2draw)
                cv2.imwrite("results/images/"+self.image_name_list[i].split("/")[-1], save_img)

def process_demo(tpu_id, image_name_list, detector_path, process_id, max_que_size, conf_thres, nms_thres):
    process =  SortThread(tpu_id, image_name_list, sail.sail_resize_type.BM_PADDING_VPP_NEAREST, max_que_size, conf_thres, nms_thres)
    process.InitProcess(detector_path, process_id)

if __name__ == '__main__':

    # argparse是python用于解析命令行参数和选项的标准模块，
    # argparse模块的作用是用于解析命令行参数。
    parse = argparse.ArgumentParser(description="Demo for yolov5")
    parse.add_argument('--img_dir', default="./datasets/mot15_trainset/ADL-Rundle-6/img1", type=str, help="image path directory")#文件夹所在目录
    parse.add_argument('--detector_path', default="./models/bmodel/yolov5s_v6.1_3output_int8_1b.bmodel", type=str)
    parse.add_argument('--conf_thres', default=0.5, type=float) 
    parse.add_argument('--nms_thres', default=0.5, type=float) 
    parse.add_argument('--process_num', default=1, type=int) 
    parse.add_argument('--device_id', default=0, type=int)   
    parse.add_argument('--max_que_size', default=16, type=int)
    parse.add_argument("--draw", action='store_true', default=False)
    parse.add_argument("--dump_result", action='store_true', default=False)

    # 创建结果位置
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists("./results/images"):
        os.mkdir("./results/images")
    if not os.path.exists("./results/mot_eval"):
        os.mkdir("./results/mot_eval")

    # 解析参数
    opt = parse.parse_args()
    tpu_id = opt.device_id
    image_path = opt.img_dir
    image_name_list = sorted(get_imagenames(image_path))  # 追踪算法需要排序
    process_count = opt.process_num            # 进程数
    max_que_size = opt.max_que_size           # 缓存的大小
    detector_path = opt.detector_path         # YOLOv5检测模型结果
    conf_thres = opt.conf_thres               # 检测置信度阈值
    nms_thres = opt.nms_thres                 # nms检测


    total_fps =  Value('f', 0.0)

    # 配置记录功能
    draw_flag = opt.draw
    dump_result = opt.dump_result

    # debug 参数
    # sail.set_print_flag(True)

    # 图片计数，用于主进程检测次数
    image_count = len(image_name_list)

    # 进程路数配置
    process_list = []
    for i in range(0,process_count):
        p = Process(target=process_demo, args=(tpu_id, image_name_list, detector_path, i, max_que_size, conf_thres, nms_thres))
        process_list.append(p)
    
    for p in process_list:
        p.start()

    for p in process_list:
        p.join()

    # 输出最终fps
    print("*"*20)
    print(f"total fps:  {total_fps.value:.2f}")
    print("*"*20)