algo_yolov5_post_cpu_opt
____________________________________________

The post-processing is accelerated for the 3-output or 1-output yolov5 model.

\_\_init\_\_
>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: c
          
        algo_yolov5_post_cpu_opt(const std::vector<std::vector<int>>& shape, 
                                    int network_w=640, int network_h=640);

**Parameters:**

* shape: std::vector<std::vector<int> >

Input parameter. Shape of input data.

* network_w: int

Input parameter. Input width of the model, default is 640.

* network_h: int

Input parameter. Input height of the model, default is 640.


process
>>>>>>>>>>>>>

Processing interface.

**Interface 1:**
    .. code-block:: c

        std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> process(std::vector<TensorPTRWithName> &input_data, 
                std::vector<int> &ost_w,
                std::vector<int> &ost_h,
                std::vector<float> &dete_threshold,
                std::vector<float> &nms_threshold,
                bool input_keep_aspect_ratio,
                bool input_use_multiclass_nms);

**Parameters 1:**

* input_data: std::vector<TensorPTRWithName>

Input parameter. Input data with three outputs or one output.

* ost_w: std::vector<int>

Input parameter. Width of original images.

* ost_h: std::vector<int>

Input parameter. Height of original images.

* dete_threshold: std::vector<float>

Input parameter. Detection threshold.

* nms_threshold: std::vector<float>

Input parameter. NMS threshold.

* input_keep_aspect_ratio: bool

Input parameter. Whether to keep aspect ratio in images.

* input_use_multiclass_nms: bool

Input parameter. Whether to use multiclass nms.

**Interface 2:**
    .. code-block:: c

        std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> process(std::map<std::string, Tensor&>& input_data,
                std::vector<int> &ost_w,
                std::vector<int> &ost_h,
                std::vector<float> &dete_threshold,
                std::vector<float> &nms_threshold,
                bool input_keep_aspect_ratio,
                bool input_use_multiclass_nms);

**Parameters 2:**

* input_data: std::map<std::string, Tensor&>

Input parameter. Input data with three outputs or one output.

* ost_w: std::vector<int>

Input parameter. Width of original images.

* ost_h: std::vector<int>

Input parameter. Height of original images.

* dete_threshold: std::vector<float>

Input parameter. Detection threshold.

* nms_threshold: std::vector<float>

Input parameter. NMS threshold.

* input_keep_aspect_ratio: bool

Input parameter. Whether to keep aspect ratio for boxes.

* input_use_multiclass_nms: bool

Input parameter. Whether to multiclass nms.

**Returns:**

std::vector<std::vector<std::tuple<left, top, right, bottom, class_id, score> > >

* left: int 

The leftmost x-coordinate of the detection result.

* top: int

The topmost y-coordinate of the detection result.

* right: int

The rightmost x-coordinate of the detection result.

* bottom: int

The bottommost y-coordinate of the detection result.

* class_id: int

The class label of the detection result.

* score: float

The score of the detection result.


reset_anchors
>>>>>>>>>>>>>>>>

Update the size of the anchor.

**Interface:**
    .. code-block:: c

        int reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new);

**Parameters:**

* anchors_new: std::vector<std::vector<std::vector<int> > >

List of anchor sizes to be updated.

**Returns:**

A return value of 0 indicates success, while other values indicate failure.