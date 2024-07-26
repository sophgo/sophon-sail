deepsort_tracker_controller_async
____________________________________________

DeepSORT算法异步处理接口

构造函数
>>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: c
          
        deepsort_tracker_controller(float max_iou_distance = 0.7, int max_age = 30, int n_init = 3,int input_queue_size = 10,int result_queue_size = 10);

**参数说明:**

* max_iou_distance: float

输入参数。模用于跟踪器中的最大交并比（IoU）距离阈值。

* max_age: int

输入参数。跟踪目标在跟踪器中存在的最大帧数。

* n_init: int

输入参数。跟踪器中的初始化帧数阈值。

* input_queue_size: int

输入参数。输入数据缓冲队列的长度。

* result_queue_size: int

输入参数。结果缓冲队列的长度。

push_data
>>>>>>>>>>>>>

异步处理接口,将输入参数推送到内部的任务队列，与get_result配合使用

**接口形式1:**
    .. code-block:: c

        int push_data(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short);

**参数说明1:**

* detected_objects: 

输入参数。检测出的物体框。

**返回值说明1:**

int

成功返回0,失败返回其他。


**返回值说明1:**

int

成功返回0,失败返回其他。

get_result
>>>>>>>>>>>>>

异步处理接口，获取待追踪目标的信息，与push_data配合使用

**接口形式:**
    .. code-block:: c

        std::vector<std::tuple<int, int, int, int, int, float, int>> get_result_npy(); 

**返回值说明:**

* std::vector<std::tuple<int, int, int, int, int, float, int>>

输出参数。被跟踪的物体。
