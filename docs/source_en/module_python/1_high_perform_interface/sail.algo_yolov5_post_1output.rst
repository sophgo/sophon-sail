sail.algo_yolov5_post_1output
_________________________________

For post-processing interfaces with a single output YOLOv5 model, \
internally implemented using thread pools.

\_\_init\_\_
>>>>>>>>>>>>

**Interface:**
    .. code-block:: python
          
        def __init__(
                    self,
                    shape: list[int], 
                    network_w:int = 640, 
                    network_h:int = 640, 
                    max_queue_size: int=20,
                    input_use_multiclass_nms: bool=True,
                    agnostic: bool=False)

**Parameters:**

* shape: list[int]

The shape of the input data.

* network_w: int

The input width of the model, which defaults to 640.

* network_h: int

The input height of the model, which defaults to 640.

* max_queue_size: int

Maximum length of cached data.

* input_use_multiclass_nms: bool

Each detection box has multiple categories.

* agnostic: bool

Algorithm without class-specific NMS


push_npy
>>>>>>>>>>

Input numpy. Only input with a batchsize of 1 is supported, or data is split before input and then sent to the interface.

**Interface:**
    .. code-block:: python

        def push_npy(self, 
                channel_idx: int, 
                image_idx: int, 
                data: numpy.ndarray[Any, numpy.dtype[numpy.float_]], 
                dete_threshold: float, 
                nms_threshold: float,
                ost_w: int, 
                ost_h: int,
                padding_left: int,
                padding_top: int,
                padding_width: int,
                padding_height: int) -> int

**Parameters:**

* channel_idx: int

The channel number of the input image.

* image_idx: int

The sequence number of the input image.

* data: numpy.ndarray[Any, numpy.dtype[numpy.float\_]]

The input data.

* dete_threshold: float

Detection threshold sequence.

* nms_threshold: float

nms threshold

* ost_w: int

The width of original image.

* ost_h: int

The height of original image.

* padding_left: int

The starting point coordinate x of the fill image. Parameters can be obtained through \
the interface of general preprocessing or the inference interface with preprocessing, \
or can be calculated by yourselves.

* padding_top: int

The starting point coordinate y of the fill image. Parameters can be obtained through \
the interface of general preprocessing or the inference interface with preprocessing, \
or can be calculated by yourselves.

* padding_width: int

Fill the width of the image,

The width of the fill image. Parameters can be obtained through the interface of general \
preprocessing or the inference interface with preprocessing, or can be calculated by yourselves.

* padding_height: int

The height of the fill image. Parameters can be obtained through the interface of general \
preprocessing or the inference interface with preprocessing, or can be calculated by yourselves.

**Returns:**

Return 0 if successful, otherwise failed.


push_data
>>>>>>>>>>>>>

Input data. The value of batchsize other than 1 is supported.

**Interface1:**
    .. code-block:: python

        def push_data(self, 
            channel_idx: list[int], 
            image_idx: list[int], 
            input_data: TensorPTRWithName, 
            dete_threshold: list[float],
            nms_threshold: list[float],
            ost_w: list[int],
            ost_h: list[int],
            padding_attrs: list[list[int]]) -> int

**Parameters1:**

* channel_idx: int

The channel number of the input image.

* image_idx: int

The sequence number of the input image.

* data: numpy.ndarray[Any, numpy.dtype[numpy.float\_]],

The input data.

* dete_threshold: float

Detection threshold sequence.

* nms_threshold: float

nms threshold.

* ost_w: int

The width of original image.

* ost_h: int

The height of original image.

* padding_attrs: list[list[int]]

The attribute list of the fill image, starting point coordinate x, starting point coordinate y, \
width after scaling, height after scaling.

**Returns:**

Return 0 if successful, otherwise failed.

**Interface2:**
    .. code-block:: python

        def push_data(self, 
            channel_idx: list[int], 
            image_idx: list[int], 
            input_data: TensorPTRWithName, 
            dete_threshold: list[list[float]],
            nms_threshold: list[float],
            ost_w: list[int],
            ost_h: list[int],
            padding_attrs: list[list[int]]) -> int

**Parameters2:**

* channel_idx: int

The channel number of the input image.

* image_idx: int

The sequence number of the input image.

* data: numpy.ndarray[Any, numpy.dtype[numpy.float\_]],

The input data.

* dete_threshold: float

Detection threshold sequence.

* nms_threshold: float

nms threshold.

* ost_w: int

The width of original image.

* ost_h: int

The height of original image.

* padding_attrs: list[list[int]]

The attribute list of the fill image, starting point coordinate x, starting point coordinate y, \
width after scaling, height after scaling.

**Returns:**

Return 0 if successful, otherwise failed.

get_result_npy
>>>>>>>>>>>>>>>>>

Get the final detection result.

**Interface:**
    .. code-block:: python

        def get_result_npy(self) 
                -> tuple[tuple[int, int, int, int, int, float],int, int]

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
