algo_yolov5_post_cpu_opt_async
_________________________________

在处理器上，针对YOLOv5模型被加速的后处理接口，内部使用线程池的方式实现。

构造函数
>>>>>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: c
          
        algo_yolov5_post_cpu_opt_async(const std::vector<std::vector<int>>& shape, 
                                int network_w=640, 
                                int network_h=640, 
                                int max_queue_size=20,
                                bool use_multiclass_nms=true);

**参数说明:**

* shape: std::vector<std::vector<int>>

输入参数。输入数据的shape。

* network_w: int

输入参数。模型的输入宽度，默认为640。

* network_h: int

输入参数。模型的输入宽度，默认为640。

* max_queue_size: int

输入参数。缓存数据的最大长度。

* use_multiclass_nms: bool

输入参数。是否使用多类NMS，默认为使用。


push_data
>>>>>>>>>>>>>

输入数据，支持任意batchsize的输入。

**接口形式1:**
    .. code-block:: c

        int push_data(
            std::vector<int> channel_idx, 
            std::vector<int> image_idx, 
            std::vector<TensorPTRWithName> input_data, 
            std::vector<float> dete_threshold,
            std::vector<float> nms_threshold,
            std::vector<int> ost_w,
            std::vector<int> ost_h,
            std::vector<std::vector<int>> padding_attr);

**参数说明1:**

* channel_idx: std::vector<int>

输入参数。输入图像序列的通道号。

* image_idx: std::vector<int>

输入参数。输入图像序列的编号。

* input_data: std::vector<TensorPTRWithName>

输入参数。输入数据，包含三个输出。

* dete_threshold: std::vector<float>

输入参数。检测阈值序列。

* nms_threshold: std::vector<float>

输入参数。nms阈值序列。

* ost_w: std::vector<int>

输入参数。原始图片序列的宽。

* ost_h: std::vector<int>

输入参数。 原始图片序列的高。

* padding_attrs: std::vector<std::vector<int>>

输入参数。填充图像序列的属性列表，填充的起始点坐标x、起始点坐标y、尺度变换之后的宽度、尺度变换之后的高度。

**返回值说明:**

成功返回0，其他值表示失败。

**接口形式2:**
    .. code-block:: c

        int push_data(
            std::vector<int> channel_idx, 
            std::vector<int> image_idx, 
            std::vector<TensorPTRWithName> input_data, 
            std::vector<std::vector<float>> dete_threshold,
            std::vector<float> nms_threshold,
            std::vector<int> ost_w,
            std::vector<int> ost_h,
            std::vector<std::vector<int>> padding_attr);

**参数说明2:**

* channel_idx: std::vector<int>

输入参数。输入图像序列的通道号。

* image_idx: std::vector<int>

输入参数。输入图像序列的编号。

* input_data: std::vector<TensorPTRWithName>

输入参数。输入数据，包含三个输出。

* dete_threshold: std::vector<std::vector<float>>

输入参数。检测阈值序列。

* nms_threshold: std::vector<float>

输入参数。nms阈值序列。

* ost_w: std::vector<int>

输入参数。原始图片序列的宽。

* ost_h: std::vector<int>

输入参数。 原始图片序列的高。

* padding_attrs: std::vector<std::vector<int>>

输入参数。填充图像序列的属性列表，填充的起始点坐标x、起始点坐标y、尺度变换之后的宽度、尺度变换之后的高度。

**返回值说明:**

成功返回0，其他值表示失败。

get_result_npy
>>>>>>>>>>>>>>>>>

获取最终的检测结果

**接口形式:**
    .. code-block:: c

        std::tuple<std::vector<std::tuple<int, int, int, int ,int, float>>,int,int> get_result_npy();

**返回值说明:**
tuple[tuple[left, top, right, bottom, class_id, score],channel_idx, image_idx]

* left: int 

检测结果最左x坐标。

* top: int

检测结果最上y坐标。

* right: int

检测结果最右x坐标。

* bottom: int

检测结果最下y坐标。

* class_id: int

检测结果的类别编号。

* score: float

检测结果的分数。

* channel_idx: int

原始图像的通道号。

* image_idx: int

原始图像的编号。

reset_anchors
>>>>>>>>>>>>>

更新anchor尺寸.

**接口形式:**
    .. code-block:: c

        int reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new);

**参数说明:**

* anchors_new: std::vector<std::vector<std::vector<int> > >

要更新的anchor尺寸列表.

**返回值说明:**

成功返回0，其他值表示失败。