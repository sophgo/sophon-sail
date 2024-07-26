Decoder_RawStream
____________________

Original stream decoder for H264/H265 decoding。

Constructor
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Initialize Decoder。

**Interface:**
    .. code-block:: c

        Decoder_RawStream(int tpu_id, string  decformt);

**Parameters:**

* tpu_id: int

The Tensor Computing Processor id that used, which defaults to 0.

* decformat: string

Input image format, supports h264 and h265.


read
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Read a frame of image from Decoder.

**Interface:**
    .. code-block:: c

        int read(uint8_t* data, int data_size, sail::BMImage &image,bool continueFrame = false);
        
**Parameters:**

* data: uint8_t*

Input parameter. The binary data of the original stream.

* image: BMImage

Output parameter. Read data into the BMImage.

* continueFrame: bool

Input parameter. Whether to read frames continuously, the default is False.

**Returns:**

* judge_ret: int

Returns 0 if the read is successful and other values if failed.


read\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Read a frame of image from Decoder.

**Interface:**
    .. code-block:: c

        int read_(uint8_t* data, int data_size, bm_image &image,bool continueFrame = false);

        
**Parameters:**

* data: uint8_t*

Input parameter. The binary data of the original stream.

* image: bm_image

Output parameter. Read data into the bm_image.

* continueFrame: bool

Input parameter. Whether to read frames continuously, the default is False.

**Returns:**

* judge_ret: int

Returns 0 if the read is successful and other values if failed.


release
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Release Decoder resources.

**Interface:**
    .. code-block:: c
    
        void release();