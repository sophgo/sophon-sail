sail.algo_yolov5_post_cpu_opt
____________________________________________

The post-processing is accelerated for the 3-output or 1-output yolov5 model.

\_\_init\_\_
>>>>>>>>>>>>

**Interface:**
    .. code-block:: python
          
        def __init__(
                    self,
                    shapes: list[list[int]], 
                    network_w: int = 640, 
                    network_h: int = 640)

**Parameters:**

* shapes: list[list[int]]

Input. Shape of input data.

* network_w: int

Input. Input width of the model, default is 640.

* network_h: int

Input. Input height of the model, default is 640.


process
>>>>>>>>>>>>>

Processing interface.

**Interface 1:**
    .. code-block:: python

        def process(self, 
            input_data: list[TensorPTRWithName], 
            ost_w: list[int], 
            ost_h: list[int], 
            dete_threshold: list[float], 
            nms_threshold: list[float], 
            input_keep_aspect_ratio: bool, 
            input_use_multiclass_nms: bool) 
                -> list[list[tuple[int, int, int, int, int, float]]]

**Parameters 1:**

* input_data: list[TensorPTRWithName]

Input. Input data with three outputs or one output.

* ost_w: list[int]

Input. Width of original images.

* ost_h: list[int]

Input. Height of original images.

* dete_threshold: list[float]

Input. Detection threshold.

* nms_threshold: list[float]

Input. NMS threshold

* input_keep_aspect_ratio: bool

Input. Whether to keep aspect ratio in images.

* input_use_multiclass_nms: bool

Input. Whether to use multiclass nms.

**Interface 2:**
    .. code-block:: python

        def process(self, 
            input_data: dict[str, Tensor], 
            ost_w: list[int], 
            ost_h: list[int], 
            dete_threshold: list[float], 
            nms_threshold: list[float],
            input_keep_aspect_ratio: bool, 
            input_use_multiclass_nms: bool) 
                -> list[list[tuple[int, int, int, int, int, float]]]

**Parameters 2:**

* input_data: dict[str, Tensor]

Input. Input data with three outputs or one output.

* ost_w: list[int]

Input. Width of original images.

* ost_h: list[int]

Input. Height of original images.

* dete_threshold: list[float]

Input. Detection threshold.

* nms_threshold: list[float]

Input. NMS threshold

* input_keep_aspect_ratio: bool

Input. Whether to keep aspect ratio for boxes.

* input_use_multiclass_nms: bool

Input. Whether to multiclass nms.

**Returns:**

list[list[tuple[left, top, right, bottom, class_id, score]]]

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

**Interface 3:**
    .. code-block:: python

        def process(self, 
            input_data: list[TensorPTRWithName], 
            ost_w: list[int], 
            ost_h: list[int], 
            dete_threshold: list[list[float]], 
            nms_threshold: list[float], 
            input_keep_aspect_ratio: bool, 
            input_use_multiclass_nms: bool) 
                -> list[list[tuple[int, int, int, int, int, float]]]

**Parameters 3:**

* input_data: list[TensorPTRWithName]

Input. Input data with three outputs or one output.

* ost_w: list[int]

Input. Width of original images.

* ost_h: list[int]

Input. Height of original images.

* dete_threshold: list[list[float]]

Input. Detection threshold.

* nms_threshold: list[float]

Input. NMS threshold

* input_keep_aspect_ratio: bool

Input. Whether to keep aspect ratio in images.

* input_use_multiclass_nms: bool

Input. Whether to use multiclass nms.

**Interface 4:**
    .. code-block:: python

        def process(self, 
            input_data: dict[str, Tensor], 
            ost_w: list[int], 
            ost_h: list[int], 
            dete_threshold: list[list[float]], 
            nms_threshold: list[float],
            input_keep_aspect_ratio: bool, 
            input_use_multiclass_nms: bool) 
                -> list[list[tuple[int, int, int, int, int, float]]]

**Parameters 4:**

* input_data: dict[str, Tensor]

Input. Input data with three outputs or one output.

* ost_w: list[int]

Input. Width of original images.

* ost_h: list[int]

Input. Height of original images.

* dete_threshold: list[list[float]]

Input. Detection threshold.

* nms_threshold: list[float]

Input. NMS threshold

* input_keep_aspect_ratio: bool

Input. Whether to keep aspect ratio for boxes.

* input_use_multiclass_nms: bool

Input. Whether to multiclass nms.

**Returns:**

list[list[tuple[left, top, right, bottom, class_id, score]]]

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
>>>>>>>>>>>>>

Update the size of the anchor.

**Interface:**
    .. code-block:: python

        def reset_anchors(self, anchors_new: list[list[list[int]]]) -> int

**Parameters:**

* anchors_new: list[list[list[int]]]

List of anchor sizes to be updated.

**Returns:**

A return value of 0 indicates success, while other values indicate failure.