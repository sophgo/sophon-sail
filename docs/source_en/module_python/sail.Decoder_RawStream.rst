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

**Example code:**
    .. code-block:: python

        import sophon.sail as sail

        filepath = 'car.264'
        with open(filepath, 'rb') as f:
            raw264 = f.read()

        decoder = sail.Decoder_RawStream(0, 'h264')
        encoder = sail.Encoder('output.mp4', 0, 'h264_bm', 'I420', 
                            'width=1920:height=1080:bitrate=3000')
        for i in range(500):
            bmi = sail.BMImage()
            ret = decoder.read(raw264, bmi, True)
            encoder.video_write(bmi)
        decoder.release()


read_single_frame
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Read a frame of image from Decoder. Need to ensure that the input data only contains complete single frame data.

**Interface:**
    .. code-block:: python

        def read_single_frame(self, data: bytes, image: BMImage, continue_frame: bool = False, need_flush: bool = False) -> int

**Parameters:**

* data: bytes

Input parameter. The binary data of the original stream.

* image: sail.BMImage

Output parameter. Read data into the BMImage.

* continue_frame: bool

Input parameter. Whether to read frames continuously, the default is False.

* need_flush: bool

Input parameter. Whether to flush the decoder buffer, the default is False.

**Returns:**

* judge_ret: int

Returns 0 if the read is successful and 1 if the read needs to continue, and other values if failed.

Returns 1 is normal for first frame, please continue to input the next frame data.

**Example code:**
    .. code-block:: python

        import sophon.sail as sail

        # Assume these files contain a complete H264 data for one frame respectively, and they are continuous frames.
        filelist = ['frame0.264', 'frame1.264', 'frame2.264', 'frame3.264']

        decoder = sail.Decoder_RawStream(0, 'h264')
        encoder = sail.Encoder('output.mp4', 0, 'h264_bm', 'I420', 
                            'width=1920:height=1080:bitrate=3000')
        for filename in filelist:
            with open(filename, 'rb') as f:
                raw264 = f.read()
            bmi = sail.BMImage()
            ret = decoder.read_single_frame(raw264, bmi, True, False)
            if ret == 0:
                encoder.video_write(bmi)
        # flush the last frame
        bmi = sail.BMImage()
        ret = decoder.read_single_frame(b'', bmi, True, True)
        encoder.video_write(bmi)
        decoder.release()
