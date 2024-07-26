sail.tpu_kernel_api_yolov5_detect_out
____________________________________________

The post-processing is accelerated using the Tensor Computing Processor Kernel for the 3-output yolov5 model, which currently only supports BM1684x, and the version of libsophon must be no less than 0.4.6 (v23.03.01).

\_\_init\_\_
>>>>>>>>>>>>

**Interface:**
    .. code-block:: python
          
        def __init__(
                    self,
                    device_id: int,
                    shape: list[list[int]], 
                    network_w: int = 640, 
                    network_h: int = 640, 
                    module_file: str="/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so")

**Parameters:**

* device_id: int

Input. Device ID used.

* shape: list[list[int]]

Input. Shape of input data.

* network_w: int

Input. Input width of the model, default is 640.

* network_h: int

Input. Input height of the model, default is 640.

* module_file: str

Input. File path of the kernel module, default is "/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so". 


process
>>>>>>>>>>>>>

Processing interface.

**Interface 1:**
    .. code-block:: python

        def process(self, 
            input_data: list[TensorPTRWithName], 
            dete_threshold: float,
            nms_threshold: float,
            release_input: bool = False) 
                -> list[list[tuple[int, int, int, int, int, float]]]

**Parameters 1:**

* input_data: list[TensorPTRWithName]

Input. Input data with three outputs.

* dete_threshold: float

Input. Detection threshold.

* nms_threshold: float

Input. NMS threshold

* release_input: bool

Input. Release input memory, default is False.

**Interface 2:**
    .. code-block:: python

        def process(self, 
            input_data: dict[str, Tensor], 
            dete_threshold: float,
            nms_threshold: float,
            release_input: bool = False) 
                -> list[list[tuple[int, int, int, int, int, float]]]

**Parameters 2:**

* input_data: dict[str, Tensor]

Input. Input data with three outputs.

* dete_threshold: float

Input. Detection threshold.

* nms_threshold: float

Input. NMS threshold.

* release_input: bool

Input parameters. Release input memory, default is False.

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