sail.algo_yolov8_seg_post_tpu_opt
____________________________________________

For the YOLOv8 segmentation model, post-processing has been accelerated using TPU.

\_\_init\_\_
>>>>>>>>>>>>

**Interface:**
    .. code-block:: python
          
        def __init__(
                    self,
                    bmodel_file: str,
                    dev_id: int,
                    detection_shape: list[int],
                    segmentation_shape: list[int],
                    network_h: int,
                    network_w: int)

**Parameters:**

* bmodel_file: str

Input. The TPU getmask bmodel path.

* dev_id: int

Input. device id.

* detection_shape: list[int]

Input. The shapes of detection head.

* segmentation_shape: list[int]

Input. The shapes of segmentation head, that is the shapes of Prototype Mask.

* network_h: int

Input. The input height of yolov8_seg network.

* network_w: int

Input. The input width of yolov8_seg network.


process
>>>>>>>>>>>>>

Processing interface.

**Interface 1:**
    .. code-block:: python

        def process(self, 
            detection_input: TensorPTRWithName, 
            segmentation_input: TensorPTRWithName, 
            ost_h: int, 
            ost_w: int,
            dete_threshold: float,
            nms_threshold: float,
            input_keep_aspect_ratio: bool,
            input_use_multiclass_nms: bool) 
                -> list[tuple[left, top, right, bottom, score, class_id, contour, mask]]

**Parameters 1:**

* detection_input: TensorPTRWithName

Input. The input data of detection head.

* segmentation_input: TensorPTRWithName

Input. The input data of segmentation head, that is Prototype Mask.

* ost_h: int

Input. Original image height.

* ost_w: int

Input. Original image width.

* dete_threshold: float

Input. Detection threshold.

* nms_threshold: float

Input. NMS threshold.

* input_keep_aspect_ratio: bool

Input. Whether to keep aspect ratio in images.

* input_use_multiclass_nms: bool

Input. Whether to use multiclass nms.

**Interface 2:**
    .. code-block:: python

        def process(self, 
            detection_input: dict[str, Tensor], 
            segmentation_input: dict[str, Tensor], 
            ost_h: int, 
            ost_w: int,
            dete_threshold: float,
            nms_threshold: float,
            input_keep_aspect_ratio: bool,
            input_use_multiclass_nms: bool) 
                -> list[tuple[left, top, right, bottom, score, class_id, contour, mask]]

**Parameters 2:**

* detection_input: dict[str, Tensor]

Input. The input data of detection head.

* segmentation_input: dict[str, Tensor]

Input. The input data of segmentation head, that is Prototype Mask.

* ost_h: int

Input. Original image height.

* ost_w: int

Input. Original image width.

* dete_threshold: float

Input. Detection threshold.

* nms_threshold: float

Input. NMS threshold.

* input_keep_aspect_ratio: bool

Input. Whether to keep aspect ratio in images.

* input_use_multiclass_nms: bool

Input. Whether to use multiclass nms.

**Returns:**

list[tuple[left, top, right, bottom, score, class_id, contour, mask]]

* left: int 

The leftmost x-coordinate of the detection box.

* top: int

The topmost y-coordinate of the detection box.

* right: int

The rightmost x-coordinate of the detection box.

* bottom: int

The bottommost y-coordinate of the detection box.

* class_id: int

The class ID of the object within the detection box..

* score: float

The score of the object within the detection box..

* contour: list[float]

The contour of the object within the detection box..

* mask: numpy.ndarray

The segmentation mask of the object within the detection box.