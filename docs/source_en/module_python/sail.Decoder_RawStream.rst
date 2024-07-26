sail.Decoder_RawStream
______________________

Original stream decoder for H264/H265 decoding。

__init__
>>>>>>>>>

**Interface:**
    .. code-block:: python

        def __init__(self, tpu_id: int, decformat: str)


**Parameters:**

* tpu_id: int

The Tensor Computing Processor id that used, which defaults to 0.

* decformat: str

Input image format, supports h264 and h265.


read
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Read a frame of image from Decoder.

**Interface:**
    .. code-block:: python

        def read(self, data: bytes, image: BMImage, continue_frame: bool = False) -> int
        
**Parameters:**

* data: bytes

Input parameter. The binary data of the original stream.

* image: sail.BMImage

Output parameter. Read data into the BMImage.

* continue_frame: bool

Input parameter. Whether to read frames continuously, the default is False.

**Returns:**

* judge_ret: int

Returns 0 if the read is successful and other values if failed.


read\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Read a frame of image from Decoder.

**Interface:**
    .. code-block:: python

        read_(self, data_bytes: bytes, image: bm_image, continue_frame: bool = False)

**Parameters:**

* data: bytes

Input parameter. The binary data of the original stream.

* image: sail.bm_image

Output parameter. Read data into the BMImage.

* continue_frame: bool

Input parameter. Whether to read frames continuously, the default is False.

**Returns:**

* judge_ret: int

Returns 0 if the read is successful and other values if failed.


release
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Release the Decoder.

**Interface:**
    .. code-block:: python
    
        def release(self)
