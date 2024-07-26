sail.EngineImagePreProcess
___________________________

Image inference interface with preprocessing function, internal use of thread pool way, \
higher efficiency while using Python.

__init__
>>>>>>>>>

**Interface:**
    .. code-block:: python

        def __init__(self,
                    bmodel_path: str, 
                    tpu_id: int,
                    use_mat_output: bool = False,
                    core_list:list = [])

**Parameters:**

* bmodel_path: str 

The path of the input model.

* tpu_id: int

The Tensor Computing Processor id that used.

* use_mat_output: bool

Whether to use OpenCV's Mat as the output of the picture. The default value is False, \
which means not used.

* use_mat_output: bool

When using the processor and BModel that support multi-core inference, you can choose multiple cores to use for inference. 
By default, N cores starting from core0 are used for inference, and N is determined by the current bmodel.
For processors and bmodel models that only support single-core inference, only a single core used for inference is supported, 
and the input list length of the parameter must be 1, if the length of the incoming list is greater than 1, the inference will be automatically on the core0.
Default is empty. If this parameter is not specified, the inference is performed by N cores starting from core 0.

InitImagePreProcess
>>>>>>>>>>>>>>>>>>>>>>>>>>

Initialize the image preprocessing module.

**Interface:**
    .. code-block:: python

        def InitImagePreProcess(self,
                    resize_mode: sail_resize_type, 
                    bgr2rgb: bool = False,
                    queue_in_size: int = 20, 
                    queue_out_size: int = 20) -> int

                    
**Parameters:**

* resize_mode: sail_resize_type

Methods of internal scaling.

* bgr2rgb: bool

Whether to convert an image from BGR to RGB.

* queue_in_size: int

The maximum length of the input image queue cache, which defaults to 20.
Must not be less than the batch_size of bmodel, if not, the value will be set to batch_size.

* queue_out_size: int

The maximum length of Tensor queue cache of preprocess result, which is 20 by default.
Must not be less than the batch_size of bmodel, if not, the value will be set to batch_size.

**Returns:**

Return 0 on success and other values on failure.
           

SetPaddingAtrr
>>>>>>>>>>>>>>>>>>>

Sets the Padding properties, only works when the resize_mode is among \
BM_PADDING_VPP_NEAREST, BM_PADDING_TPU_NEAREST, BM_PADDING_TPU_LINEAR, BM_PADDING_TPU_BICUBIC.

**Interface:**
    .. code-block:: python

        def SetPaddingAtrr(self,
                    padding_b:int=114,
                    padding_g:int=114,
                    padding_r:int=114,
                    align:int=0) -> int 

**Parameters:**
* padding_b: int

The padding pixel value of b channel, which defaults to 114.

* padding_g: int

The padding pixel value of g channel, which defaults to 114.
                
* padding_r: int

The padding pixel value of r channel, which defaults to 114.

* align: int

Image fill position, 0 indicates fill from the top left corner, \
1 indicates center fill, default is 0.
          
**Returns:**

Return 0 on success and other values on failure.


SetConvertAtrr
>>>>>>>>>>>>>>>>>>>

Sets the properties of the linear transformation.

**Interface:**
    .. code-block:: python

        def SetConvertAtrr(self, alpha_beta) -> int:

**Parameters:**

* alpha_beta: (a0, b0), (a1, b1), (a2, b2)。

    a0 is the coefficient of linear transformation for the 0th channel;

    b0 is the offset of linear transformation for the 0th channel;
   
    a1 is the coefficient of linear transformation for the 1th channel;

    b1 is the offset of linear transformation for the 1th channel;

    a2 is the coefficient of linear transformation for the 2th channel;

    b2 is the offset of linear transformation for the 2th channel;

**Returns:**

Return 0 on success and other values on failure.


PushImage
>>>>>>>>>>>>>>

push image data

**Interface:**
    .. code-block:: python

        def PushImage(self,
                    channel_idx: int, 
                    image_idx: int, 
                    image: BMImage) -> int

**Parameters:**
* channel_idx: int

The channel index of the input image

* image_idx: int
                
The image index of the input image

* image: BMImage
                
The input image

**Returns:**

Return 0 on success and other values on failure.


GetBatchData_Npy
>>>>>>>>>>>>>>>>>>>

Get a batch of inference results. When using this interface, \
use_mat_output must be False because the result type is BMImage.

**Interface:**
    .. code-block:: python

        def GetBatchData_Npy(self) 
        -> tuple[[dict[str, ndarray], list[BMImage],list[int],list[int],list[list[int]]]]

**Returns:**

tuple[output_array, ost_images, channels, image_idxs, padding_attrs]

* output_array: dict[str, ndarray]

The inference result

* ost_images: list[BMImage]

Original image queue

* channels: list[int]

The result corresponds to the channel sequence of the original picture.

* image_idxs: list[int]

The result corresponds to the index sequence of the original picture.

* padding_attrs: list[list[int]]

The attribute list of the filling image. The starting point coordinate x, \
starting point coordinate y, the width after scaling, and the height after scaling.



GetBatchData_Npy2
>>>>>>>>>>>>>>>>>>>>>>>

Get a batch of inference results. When using this interface, \
use_mat_output must be True because the result type is numpy.ndarray[numpy.uint8].

**Interface:**
    .. code-block:: python

        def GetBatchData_Npy2(self) 
            -> tuple[dict[str, ndarray], list[numpy.ndarray[numpy.uint8]],list[int],list[int],list[list[int]]]

**Returns:**

tuple[output_array, ost_images, channels, image_idxs, padding_attrs]

* output_array: dict[str, ndarray]

The inference result

* ost_images: list[numpy.ndarray[numpy.uint8]]

Original image queue

* channels: list[int]

The result corresponds to the channel sequence of the original picture.

* image_idxs: list[int]

The result corresponds to the index sequence of the original picture.

* padding_attrs: list[list[int]]

The attribute list of the filling image. The starting point coordinate x, \
starting point coordinate y, the width after scaling, and the height after scaling.

GetBatchData
>>>>>>>>>>>>>>>>

Get a batch of inference results. When using this interface, \
use_mat_output must be False because the result type is BMImage. It is worth noting that output tensors must be manually released.

**Interface:**
    .. code-block:: python
        
        def GetBatchData(self,
                    need_d2s: bool = True) 
                    -> tuple[list[TensorPTRWithName], list[BMImage],list[int],list[int],list[list[int]]]

**Parameters:**

* need_d2s: bool

Whether to move data to the system memory. The default value is True, which means Yes.

**Returns:**

tuple[output_array, ost_images, channels, image_idxs, padding_attrs]

* output_array: list[TensorPTRWithName]

The inference result

* ost_images: list[BMImage]

Original image queue

* channels: list[int]

The result corresponds to the channel sequence of the original picture.

* image_idxs: list[int]

The result corresponds to the index sequence of the original picture.

* padding_attrs: list[list[int]]

The attribute list of the filling image. The starting point coordinate x, \
starting point coordinate y, the width after scaling, and the height after scaling.


get_graph_name
>>>>>>>>>>>>>>>>

Get the name of model computation graph

**Interface:**
    .. code-block:: python

        def get_graph_name(self) -> str

**Returns:**

Return the first name of model computation graph

            
get_input_width
>>>>>>>>>>>>>>>>

Get the width of model input.

**Interface:**
    .. code-block:: python

        def get_input_width(self) -> int

**Returns:**

Return the width of model input.

            
get_input_height
>>>>>>>>>>>>>>>>>>>

Get the height of model input.

**Interface:**
    .. code-block:: python

        def get_input_height(self) -> int

**Returns:**

Return the height of model input.

            
get_output_names
>>>>>>>>>>>>>>>>>>>

Get tensor names of model output.

**Interface:**
    .. code-block:: python

        def get_output_names(self) -> list[str]

**Returns:**

Return all tensor names of model output.
   
            
get_output_shape
>>>>>>>>>>>>>>>>>>>

Get the shape of the specified output Tensor


**Interface:**
    .. code-block:: python
        
        def get_output_shape(self, tensor_name: str) -> list[int]

**Parameters:**

* tensor_name: str

The name of output tensor

**Returns:**

Return the shape of the specified output Tensor