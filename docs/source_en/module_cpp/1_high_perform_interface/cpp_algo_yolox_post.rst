algo_yolox_post
_________________________________

For post-processing interfaces with yolox model, \
internally implemented using thread pools.

Constructor
>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: c
          
        algo_yolox_post(const std::vector<int>& shape, 
                                int network_w=640, 
                                int network_h=640, 
                                int max_queue_size=20);

**Parameters:**

* shape: std::vector<int>

Input parameters. The shape of the input data.

* network_w: int

Input parameters. The input width of the model, which defaults to 640.

* network_h: int

Input parameters. The input height of the model, which defaults to 640.

* max_queue_size: int

Input parameters. Maximum length of cached data.




push_data
>>>>>>>>>>>>>

Input data. The value of batchsize other than 1 is supported.

**Interface:**
    .. code-block:: c

        int push_data(
            std::vector<int> channel_idx, 
            std::vector<int> image_idx, 
            TensorPTRWithName input_data, 
            std::vector<float> dete_threshold,
            std::vector<float> nms_threshold,
            std::vector<int> ost_w,
            std::vector<int> ost_h,
            std::vector<std::vector<int>> padding_attr);

**Parameters:**

* channel_idx: std::vector<int>

Input parameters. The channel number of the input image.

* image_idx: std::vector<int>

Input parameters. The sequence number of the input image.

* input_data: TensorPTRWithName

Input parameters. The input data.

* dete_threshold: std::vector<float>

Input parameters. Detection threshold sequence.

* nms_threshold: std::vector<float>

Input parameters. nms threshold.

* ost_w: std::vector<int>

Input parameters. The width of original image.

* ost_h: std::vector<int>

Input parameters. The height of original image.

* padding_attrs: std::vector<std::vector<int> >

Input parameters. The attribute list of the fill image, starting point coordinate x, starting point coordinate y, \
width after scaling, height after scaling.

**Returns:**

Return 0 if successful, otherwise failed.

get_result_npy
>>>>>>>>>>>>>>>>>

Get the final detection result.

**Interface:**
    .. code-block:: c

        std::tuple<std::vector<std::tuple<int, int, int, int ,int, float>>,int,int> get_result_npy();

**Returns:**
tuple[tuple[left, top, right, bottom, class_id, score],channel_idx, image_idx]

* left: int 

The left x coordinate of the detection result.

* top: int

The top y coordinate of the detection result.

* right: int

The right x coordinate of the detection result.

* bottom: int

The bottom y coordinate of the detection result.

* class_id: int

Category number of detection result. 

* score: float

Score of detection result.

* channel_idx: int

The channel index of original image.

* image_idx: int

The image index of original image.
