import sophon.sail as sail
import numpy as np
import threading
import time
import os
import cv2
from multiprocessing import Process, Queue

def yolox_postprocess(outputs, input_w, input_h, p6=False):
        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [input_h // stride for stride in strides]
        wsizes = [input_w // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)

def get_detectresult(predictions,dete_threshold,nms_threshold):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_threshold, score_thr=dete_threshold)
        return dets

class MultiDecoderThread(object):
    def __init__(self, tpu_id, video_list):
        self.channel_list = {}
        self.tpu_id = tpu_id
        self.break_flag = False
        self.multiDecoder = sail.MultiDecoder(15, tpu_id)
        self.multiDecoder.set_local_flag(True)

        for video_name in video_list:
            channel_index = self.multiDecoder.add_channel(video_name,0)
            print("Add Channel[{}]: {}".format(channel_index,video_name))
            self.channel_list[channel_index] = video_name

        self.alpha_beta = (1.0,0),(1.0,0),(1.0,0)

    def InitImagePreProcess(self, bmodel_name:str,  resize_type:sail.sail_resize_type):
        self.engine_image_pre_process = sail.EngineImagePreProcess(bmodel_name, self.tpu_id, 1)
        self.engine_image_pre_process.InitImagePreProcess(resize_type, False, 10, 10)
        self.engine_image_pre_process.SetPaddingAtrr()
        self.engine_image_pre_process.SetConvertAtrr(self.alpha_beta)
        self.net_w = self.engine_image_pre_process.get_input_width()
        self.net_h = self.engine_image_pre_process.get_input_height()
        thread_preprocess = threading.Thread(target=self.decoder_and_pushdata, args=(self.channel_list, self.multiDecoder, self.engine_image_pre_process))
        thread_preprocess.start()

    
    def decoder_and_pushdata(self,channel_list, multi_decoder, PreProcessAndInference):
        image_index = 0
        break_flag_preprocess = False
        total_count = 0
        time_start = time.time()
        while True:
            for key in channel_list:
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

                if self.break_flag :
                    break_flag_preprocess = True
                    break
            if break_flag_preprocess:
                break

def InferencesProcess(bmodel_name:str, video_list:list, resize_type:sail.sail_resize_type, device_id:int, max_count:int, que_list:Queue):
    process = MultiDecoderThread(device_id,video_list)
    process.InitImagePreProcess(bmodel_name,resize_type)
    post_process_count = len(que_list)
    for i in range(max_count):
        idx_que = i%post_process_count
        if(not que_list[idx_que].full()):
            start_time = time.time()
            output_tensor_map, ost_cv_images, channel_list ,imageidx_list, padding_atrr = process.engine_image_pre_process.GetBatchData_Npy2()
            end_time = time.time()
            que_list[idx_que].put([output_tensor_map,ost_cv_images,channel_list ,imageidx_list, padding_atrr],False)
            print("Queue: {},[{},{}]GetBatchData_Npy time use: {:.2f} ms".format(idx_que, i,max_count,(end_time-start_time)*1000))
        else:
            print("Queue: {} full, not push to Queue!".format(i%post_process_count))


def GetData(que:Queue, max_count):
    for i in range(max_count):
        output_tensor_map, ost_images, channel_list ,imageidx_list, padding_atrr = que.get(True)
        start_time = time.time()
        for np_resu in output_tensor_map.values():
            predictions = yolox_postprocess(np_resu, 640, 640, False)
            for idx,image in enumerate(ost_images): 
                dete_boxs = get_detectresult(predictions[idx],0.5, 0.5)
                if dete_boxs is None:
                    continue
                ost_w = image.shape[1]
                ost_h = image.shape[0]
                start_x,start_y, w_temp, h_temp = padding_atrr[idx]
                ratio_w = ost_w/w_temp
                ratio_h = ost_h/h_temp
                dete_boxs[:,0] *= ratio_w
                dete_boxs[:,1] *= ratio_h
                dete_boxs[:,2] *= ratio_w
                dete_boxs[:,3] *= ratio_h

                # for dete_box in dete_boxs:
                #     cv2.rectangle(image, (int(dete_box[0]), int(dete_box[1])), 
                #         (int(dete_box[2]), int(dete_box[3])), (0, 0, 255), 4) 
            
                # cv2.imwrite("{}_{}_{}.jpg".format(os.getpid(),channel_list[idx],imageidx_list[idx]),image)

        end_time = time.time()
        print("PID:{}, Post Use:{} ms".format(os.getpid(),(end_time-start_time)*1000))


if __name__ == '__main__':
    tpu_id = 0
    max_count = 1000
    decoder_count = 8
    post_count = 4

    video_path = "/data/video/"
    video_name_list = [ "001.mp4",
                        "002.mp4",
                        "003.mp4",
                        "004.mp4"]
    
    bmodel_name = "/data/models/yolox_s_int8_bs4.bmodel"
    video_list = []
    for i in range(0,decoder_count):
        video_list.append(os.path.join(video_path,video_name_list[i%len(video_name_list)]))
    
    que_list_0 = []
    que_list_1 = []
    for i in range(post_count):
        que_list_0.append(Queue(5))
        que_list_1.append(Queue(5))
    p0 = Process(target=InferencesProcess, args=(bmodel_name,video_list, sail.sail_resize_type.BM_RESIZE_VPP_NEAREST, tpu_id, max_count, que_list_0))
    p1 = Process(target=InferencesProcess, args=(bmodel_name,video_list, sail.sail_resize_type.BM_RESIZE_VPP_NEAREST, tpu_id, max_count, que_list_1))
    post_list = []
    for i in range(post_count):
        post_list.append(Process(target=GetData, args=(que_list_0[i],1000)))
        post_list.append(Process(target=GetData, args=(que_list_1[i],1000)))
    p0.start()
    p1.start()
    for i in range(len(post_list)):
        post_list[i].start()
    
    p0.join()
    p1.join()
    for i in range(len(post_list)):
        post_list[i].join()
  


