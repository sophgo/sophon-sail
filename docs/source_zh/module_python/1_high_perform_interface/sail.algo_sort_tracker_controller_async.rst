sort_tracker_controller_async
____________________________________________

SORT算法异步处理接口,内部用线程实现

\_\_init\_\_
>>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: python
          
        def __init__(max_iou_distance:float = 0.7, max_age:int = 30, n_init:int = 3, input_queue_size:int = 10, result_queue_size:int = 10)


**参数说明:**

* max_iou_distance: float

输入参数。模用于跟踪器中的最大交并比（IoU）距离阈值。

* max_age: int

输入参数。跟踪目标在跟踪器中存在的最大帧数。

* n_init: int

输入参数。跟踪器中的初始化帧数阈值。

* input_queue_size: int

输入参数。输入缓冲队列的大小

* result_queue_size: int

输入参数。结果缓冲队列的大小


push_data
>>>>>>>>>>>>>>>

异步处理接口,将数据推送到内部的任务队列中,与get_result_npy配合使用

**接口形式1:**
    .. code-block:: python
          
        def push_data(detected_objects:list[tuple[int, int, int, int, int, float]]) -> int



**参数说明:**

* detected_objects: list(tuple(left, top, right, bottom, class_id, score))

输入参数。检测出的物体框。

* feature:sail.Tensor

输入参数。检测出的物体的特征。


**返回值说明:**

* int

成功返回0，失败返回其他。


get_result_npy
>>>>>>>>>>>>>>>

异步处理接口,获取追踪目标的信息,与push_data配合使用

**接口形式:**
    .. code-block:: python
          
        def get_result_npy() -> tracked_objects:list[tuple[int, int, int, int, int, float, int]]


**返回值说明:**

* tracked_objects: list(tuple(left, top, right, bottom, class_id, score, track_id))

输出参数。被跟踪的物体。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import cv2
        import numpy as np
        from python.yolov5_opencv import YOLOv5
        from feature_extractor import Extractor
        class yolov5_arg:
            def __init__(self, bmodel, dev_id, conf_thresh, nms_thresh):
                self.bmodel = bmodel
                self.dev_id = dev_id
                self.conf_thresh = conf_thresh
                self.nms_thresh = nms_thresh
        if __name__ == '__main__':
            input = "data/test_car_person_1080P.mp4"
            bmodel_detector = "models/BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel"

            dev_id = 0
            conf = 0.4
            nms = 0.7

            yolov5_args = yolov5_arg(bmodel_detector, dev_id, conf, nms)
            yolov5 = YOLOv5(yolov5_args)


            cap = cv2.VideoCapture(input)
            img_batch = []

            dstc = sail.sort_tracker_controller_async(max_iou_distance=0.7, max_age=70, n_init=3)

            track_res_all_numpy = np.array([])

            for i in range(15):
                _, im = cap.read()
                if im is None:
                    break
                img_batch.append(im)
                results = yolov5(img_batch)
                det = results[0]


                # The order of this API and the demo is inconsistent, and the class_id and score are reversed 
                det[:, [4,5]] = det[:,[5,4]]
                img_batch.clear()

                det_tuple = [tuple(row) for row in det]

                # -------------------v numpy------------------------
                # left, top, right, bottom, class_id, score, track_id
                ret = dstc.push_data(det_tuple)
                track_res_numpy = np.array(dstc.get_result_npy())

                if i == 0:
                    track_res_all_numpy = track_res_numpy
                else:
                    track_res_all_numpy = np.concatenate((track_res_all_numpy, track_res_numpy), axis=0)

            cap.release() 