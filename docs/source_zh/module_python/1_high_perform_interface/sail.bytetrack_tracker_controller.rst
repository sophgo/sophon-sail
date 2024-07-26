bytetrack_tracker_controller
____________________________________________

针对ByteTrack算法，通过处理检测的结果，实现对目标的跟踪。

\_\_init\_\_
>>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: python
          
        def __init__(frame_rate:int = 30, 
                track_buffer:int = 30)


**参数说明:**

* frame_rate: int

输入参数。用于控制被追踪物体允许消失的最大帧数，数值越大则被追踪物体允许消失的最大帧数越大。

* track_buffer: int

输入参数。用于控制被追踪物体允许消失的最大帧数，数值越大则被追踪物体允许消失的最大帧数越大。



process
>>>>>>>>>>>>>

处理接口。

**接口形式1:**
    .. code-block:: python

        def process(detected_objects:list[list[int, float, float, float, float, float, float, float]], 
                tracked_objects:list[list[int, float, float, float, float, float, float, float, int]]) 
                    -> int

**参数说明1:**

* detected_objects: list[list[int, float, float, float, float, float, float, float]]

输入参数。检测出的物体框。

* tracked_objects: list[list[int, float, float, float, float, float, float, float, int]]

输出参数。被跟踪的物体。

**返回值说明:**

int

成功返回0，失败返回其他。

**示例代码:**
    .. code-block:: python

        # The example code relies on sophon-demo/sample/YOLOv5/python/yolov5_opencv.py
        import sophon.sail as sail
        import cv2
        import numpy as np
        from python.yolov5_opencv import YOLOv5
        class yolov5_arg:
            def __init__(self, bmodel, dev_id, conf_thresh, nms_thresh):
                self.bmodel = bmodel
                self.dev_id = dev_id
                self.conf_thresh = conf_thresh
                self.nms_thresh = nms_thresh
        if __name__ == '__main__':
            input = "datasets/test_car_person_1080P.mp4"
            bmodel = "models/BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel"
            dev_id = 0
            conf = 0.4
            nms = 0.7
            yolov5_args = yolov5_arg(bmodel, dev_id, conf, nms)
            yolov5 = YOLOv5(yolov5_args)

            cap = cv2.VideoCapture(input)
            img_batch = []
            btt = sail.bytetrack_tracker_controller()
            track_res_all = np.array([])

            for i in range(50):
                _, im = cap.read()
                if im is None:
                    break
                img_batch.append(im)
                results = yolov5(img_batch)
                det = results[0]

                # The order of this API and the demo is inconsistent, and the class_id and score are reversed 
                det[:, [4,5]] = det[:,[5,4]]
                det = tuple(det)
                img_batch.clear()

                det_tuple = [tuple(row) for row in det]

                # tuple(left, top, right, bottom, class_id, score, track_id)
                track_res = btt.process(det_tuple)
                track_res = np.array(track_res)
                if i == 0:
                    track_res_all = track_res
                else:
                    track_res_all = np.concatenate((track_res_all, track_res), axis=0)
            
            cap.release()