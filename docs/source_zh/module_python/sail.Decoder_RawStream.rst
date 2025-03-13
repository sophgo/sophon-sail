sail.Decoder_RawStream
______________________

裸流解码器，可实现H264/H265的解码。

__init__
>>>>>>>>>

**接口形式:**
    .. code-block:: python

        def __init__(self, tpu_id: int, decformat: str)


**参数说明:**

* tpu_id: int

输入参数。使用的智能视觉深度学习处理器 id，默认为0。

* decformat: str

输入参数。输入图像的格式，支持h264和h265


read
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

从Decoder中读取一帧图像。

**接口形式1:**
    .. code-block:: python

        def read(self, data: bytes, image: BMImage, continue_frame: bool = False) -> int
        
**参数说明1:**

* data: bytes

输入参数。裸流的二进制数据。

* image: sail.BMImage

输出参数。将数据读取到image中。

* continue_frame: bool

输入参数。是否连续读帧,默认为False。

**返回值说明1:**

* judge_ret: int

读取成功返回0，失败返回其他值。


read\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

从Decoder中读取一帧图像。

**接口形式1:**
    .. code-block:: python

        read_(self, data_bytes: bytes, image: bm_image, continue_frame: bool = False)

**参数说明1:**

* data: bytes

输入参数。裸流的二进制数据。

* image: sail.bm_image

输出参数。将数据读取到image中。

* continue_frame: bool

输入参数。是否连续读帧,默认为False。

**返回值说明1:**

* judge_ret: int

读取成功返回0，失败返回其他值。


release
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

释放Decoder资源。

**接口形式:**
    .. code-block:: python
    
        def release(self) -> None

**示例代码:**
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