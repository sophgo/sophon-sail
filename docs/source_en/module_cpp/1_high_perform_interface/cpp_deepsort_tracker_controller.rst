deepsort_tracker_controller
____________________________________________

针对DeepSORT算法，通过处理检测的结果和提取的特征，实现对目标的跟踪。

构造函数
>>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: c
          
        deepsort_tracker_controller(float max_cosine_distance, 
                                int nn_budget, 
                                int k_feature_dim, 
                                float max_iou_distance = 0.7, 
                                int max_age = 30, 
                                int n_init = 3);

**参数说明:**

* max_cosine_distance: float

输入参数。用于相似度计算的最大余弦距离阈值。

* nn_budget: int

输入参数。用于最近邻搜索的最大数量限制。

* k_feature_dim: int

输入参数。被检测的目标的特征维度。

* max_iou_distance: float

输入参数。模用于跟踪器中的最大交并比（IoU）距离阈值。

* max_age: int

输入参数。跟踪目标在跟踪器中存在的最大帧数。

* n_init: int

输入参数。跟踪器中的初始化帧数阈值。


process
>>>>>>>>>>>>>

处理接口。

**接口形式1:**
    .. code-block:: c

        int process(const vector<DeteObjRect>& detected_objects, 
                vector<Tensor>& feature, 
                vector<TrackObjRect>& tracked_objects);

**参数说明1:**

* detected_objects: vector<DeteObjRect>

输入参数。检测出的物体框。

* feature: vector<Tensor>

输入参数。检测出的物体的特征。

* tracked_objects: vector<TrackObjRect>

输出参数。被跟踪的物体。

**返回值说明:**

int

成功返回0，失败返回其他。