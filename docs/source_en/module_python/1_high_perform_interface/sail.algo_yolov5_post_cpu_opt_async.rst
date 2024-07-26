sail.algo_yolov5_post_cpu_opt_async
______________________________________

For accelerated post-processing interfaces in cpu, \
internally implemented using thread pools.

\_\_init\_\_
>>>>>>>>>>>>

**Interface:**
    .. code-block:: python
          
        def __init__(
                    self,
                    shape: list[list[int]], 
                    network_w:int = 640, 
                    network_h:int = 640, 
                    max_queue_size: int=20,
                    use_multiclass_nms: bool=True)

**Parameters:**

* shape: list[list[int]]

The shape of the input data.

* network_w: int

The input width of the model, which defaults to 640.

* network_h: int

The input height of the model, which defaults to 640.

* max_queue_size: int

Maximum length of cached data.

* use_multiclass_nms: bool

Whether to use multi-class NMS, which defaults to true.


push_data
>>>>>>>>>>>>>

Support input with arbitrary batchsize.

**Interface:**
    .. code-block:: python

        def push_data(self, 
            channel_idx: list[int], 
            image_idx: list[int], 
            input_data: list[TensorPTRWithName], 
            dete_threshold: list[float],
            nms_threshold: list[float],
            ost_w: list[int],
            ost_h: list[int],
            padding_attrs: list[list[int]]) -> int

**Parameters:**

* channel_idx: list[int]

The channel number of the input image.

* image_idx: list[int]

The sequence number of the input image.

* input_data: list[TensorPTRWithName],

The input data, including three outputs.

* dete_threshold: list[float]

Detection threshold sequence.

* nms_threshold: list[float]

nms threshold.

* ost_w: list[int]

The width of original image.

* ost_h: list[int]

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

reset_anchors
>>>>>>>>>>>>>

Reset anchors.

**Interface:**
    .. code-block:: python

        def reset_anchors(self, anchors_new: list[list[list[int]]]) -> int

**Parameters:**

* anchors_new: list[list[list[int]]]

new anchors.

**Returns:**

Return 0 if successful, otherwise failed.