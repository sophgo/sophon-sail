sort_tracker_controller
____________________________________________

基于SORT算法对追踪目标进行匹配

构造函数
>>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: c
          
        deepsort_tracker_controller(float max_iou_distance = 0.7, int max_age = 30, int n_init = 3);

**参数说明:**

* max_iou_distance: float

输入参数。模用于跟踪器中的最大交并比（IoU）距离阈值。

* max_age: int

输入参数。跟踪目标在跟踪器中存在的最大帧数。

* n_init: int

输入参数。跟踪器中的初始化帧数阈值。


process
>>>>>>>>>>>>>

处理接口。

**接口形式:**
    .. code-block:: c

        std::vector<std::tuple<int, int, int, int, int, float, int>> process(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short);

**参数说明:**

* detected_objects_short:std::vector<std::tuple<left, top, right, bottom, class_id, score>>

输入参数。检测出的物体框。



**返回值说明:**

* tracked_objects: std::vector<std::tuple<left, top, right, bottom, class_id, score, track_id>>

输出参数。被跟踪的物体。


