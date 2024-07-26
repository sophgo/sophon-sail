sail.ImagePreProcess
______________________

General preprocessing interface, internal use of thread pool way.

__init__
>>>>>>>>>

**Interface:**
    .. code-block:: python

        def __init__(self,
                    batch_size: int, 
                    resize_mode:sail_resize_type,
                    tpu_id: int=0, 
                    queue_in_size: int=20, 
                    queue_out_size: int=20,
                    use_mat_output: bool = False)


**Parameters:**

* batch_size: int

The batch size of output

* resize_mode: sail_resize_type

Methods of internal scaling.

* tpu_id: int

The Tensor Computing Processor id that used, which defaults to 0

* queue_in_size: int

The maximum length of the input image queue cache, which defaults to 20.

* queue_out_size: int

The maximum length of Tensor queue cache of preprocess result , which is 20 by default.

* use_mat_output: bool

Whether to use OpenCV Mat as the output of the picture. The default value is False.

SetResizeImageAtrr
>>>>>>>>>>>>>>>>>>>>

Sets the properties of the image scaling.

**Interface:**
    .. code-block:: python

        def SetResizeImageAtrr(self,
                    output_width: int, 
                    output_height: int,
                    bgr2rgb: bool, 
                    dtype: ImgDtype) -> None

**Parameters:**
            
* output_width: int

The image width after scaling

* output_height: int

The image height after scaling

* bgr2rgb: bool

Whether to convert an image from BGR to GRB.

* dtype: ImgDtype  

The data type after image scaling, the current version only supports BM_FLOAT32,BM_INT8,BM_UINT8. \
It can be set according to the input data type of the model.

SetPaddingAtrr
>>>>>>>>>>>>>>>>>>>>

Sets the Padding properties, only works when the resize_mode is among \
BM_PADDING_VPP_NEAREST, BM_PADDING_TPU_NEAREST, BM_PADDING_TPU_LINEAR, BM_PADDING_TPU_BICUBIC.

**Interface:**
    .. code-block:: python

        def SetPaddingAtrr(self,
                    padding_b: int=114,
                    padding_g: int=114,
                    padding_r: int=114,
                    align: int=0) -> None

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


SetConvertAtrr
>>>>>>>>>>>>>>>>>>>>

Sets the properties of the linear transformation.

**Interface:**
    .. code-block:: python

        def SetConvertAtrr(self, alpha_beta) -> int 

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
>>>>>>>>>>>>>>>

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
            
GetBatchData
>>>>>>>>>>>>>>>

Get process result.

**Interface:**
    .. code-block:: python
        
        def GetBatchData(self) 
            -> tuple[Tensor, list[BMImage],list[int],list[int],list[list[int]]]
        
**Returns:**
tuple[data, images, channels, image_idxs, padding_attrs]

* data: Tensor

The inference result

* images: list[BMImage]

Original image queue

* channels: list[int]

The result corresponds to the channel sequence of the original picture.

* image_idxs: list[int]

The result corresponds to the index sequence of the original picture.

* padding_attrs: list[list[int]]

The attribute list of the filling image. The starting point coordinate x, \
starting point coordinate y, the width after scaling, and the height after scaling.


set_print_flag
>>>>>>>>>>>>>>>

Set the flag bit for printing logs. Logs are not printed when this interface is not used.

**Interface:**
    .. code-block:: python

        def set_print_flag(self, flag: bool) -> None:
        
**Returns:**

* flag: bool

Flag bit for printing. False means no printing, True indicates printing.
