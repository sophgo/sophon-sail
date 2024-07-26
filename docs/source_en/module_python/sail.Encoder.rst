sail.Encoder
____________

Encoder, supporting video and image encoding.

\_\_init\_\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Encoder init

piture encoder init:

**Interface:**
    .. code-block:: python

        def __init__(self)

video encoder init:

**Interface:**
    .. code-block:: python

        def __init__(self, output_path: str, handle: sail.Handle, enc_fmt: str, pix_fmt: str, enc_params: str, cache_buffer_length: int=5, abort_policy: int=0)

        def __init__(self, output_path: str, device_id: int, enc_fmt: str, pix_fmt: str, enc_params: str, cache_buffer_length: int=5, abort_policy: int=0)

**Parameters:**

* output_path: str

output path, support local video file and rtsp/rtmp stream.

* handle: sail.Hnadle

Handle instance. (either handle instance or device_id)

* device_id: int

Encoder device_id. (either handle instance or device_id, when specify device_id, the encoder will create a Handle internally)

* enc_fmt: str

encoder format, support h264_bm and h265_bm/hevc_bm.

* pix_fmt: str

output pixel format, support NV12 and I420. I420 is recommended.

* enc_params: str

encoder params, ``"width=1902:height=1080:gop=32:gop_preset=3:framerate=25:bitrate=2000"`` , width and height are necessary. By default, Bitrate is used to control quality, and Bitrate becomes invalid when qp is specified in the parameter.

* cache_buffer_length: int

The internal cache queue length defaults to 5. sail.Encoder internally maintains a cache queue to improve flow control fault tolerance during pushing stream.

* abort_policy: int

The reject policy for video_write. 0 for returns -1 immediately. 1 for pop queue header. 2 for clear the queue. 3 for blocking.

is_opened
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Determine if the encoder is turned on.

**Interface:**
    .. code-block:: python

        def is_opened(self) -> bool

**return:**

* judge_ret: bool

return True when opened, and False when failed.

**Sample:**
    .. code-block:: python
        
        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            handle = sail.Handle(dev_id)
            out_path = "path/to/your/output/file"            
            enc_fmt = "h264_bm"                           
            pix_fmt = "I420"                              
            enc_params = "width=1920:height=1080:bitrate=2000:gop=32:gop_preset=2:framerate=25"  
            cache_buffer_length = 5                      
            abort_policy = 0                             
            encoder = sail.Encoder(out_path, handle, enc_fmt, pix_fmt, enc_params, cache_buffer_length, abort_policy)
            print(encoder.is_opened())

pic_encode
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Encode an image and return the encoded data.

**Interface1:**
    .. code-block:: python

        def pic_encode(self, ext: str, image: BMImage)->numpy.array

**Interface2:**
    .. code-block:: python

        def pic_encode(self, ext: str, image: bm_image)->numpy.array
        
**Parameters:**

* ext: str

Image encoding format. such as ``".jpg"`` , ``".png"``

* image: sail.BMImage

Input image, only supports picture of FORMAT_BGR_PACKED and DATA_TYPE_EXT_1N_BYTE.

**return:**

* data: numpy.array

Encoded data stored in system memory.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            handle = sail.Handle(dev_id)
            img_path = "path/to/your/output/file"
            decoder = sail.Decoder(img_path,False,dev_id)
            img = decoder.read(handle)   
            # img = decoder.read(handle).data()   //bm_image
            encoder = sail.Encoder()
            data = encoder.pic_encode(".jpg", img)
            print(data)

video_write
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Send a frame of image to the video encoder. Asynchronous interface, after format conversion, is placed in the internal cache queue.

**Interface1:**
    .. code-block:: python

        def video_write(self, image: sail.BMImage)->int

**Interface2:**
    .. code-block:: python

        def video_write(self, image: sail.bm_image)->int

**Parameters:**

* image: sail.BMImage

On the BM1684, 
when the pixel format (pix_fmt) of the encoder is set to I420, the shape of the image to be encoded can differ from the encoder's width and height. 
However, when the pixel format is NV12, the image shape must match the encoder's dimensions. In this case, a format conversion is performed internally using ``bmcv_image_storage_convert``, which may utilize NPU resources.

On the BM1684X, 
the shape of the image to be encoded can differ from the encoder's width and height. The internal resizing and format conversion are handled by ``bmcv_image_vpp_convert``.

**Return:**

* judge_ret: int

Successfully returned 0, internal cache queue full returned -1. encode failed returns -2. push stream failed returns -3. unknown abort policy returns -4.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            handle = sail.Handle(dev_id)
            img_path = "your_img_path"
            decoder = sail.Decoder(img_path,False,dev_id)
            img = decoder.read(handle)     
            out_path = "path/to/your/output/file"            
            enc_fmt = "h264_bm"                           
            pix_fmt = "I420"                              
            enc_params = "width=1920:height=1080:bitrate=2000:gop=32:gop_preset=2:framerate=25"  
            cache_buffer_length = 5                      
            abort_policy = 0                             
            encoder = sail.Encoder(out_path, handle, enc_fmt, pix_fmt, enc_params, cache_buffer_length, abort_policy)            
            ret = encoder.video_write(img)
            # ret = encoder.video_write(img.data())  #  sail.bm_image
            print(ret)

release
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

release encoder

**Interface:**
    .. code-block:: python

        def release(self)->None

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            handle = sail.Handle(dev_id)
            out_path = "path/to/your/output/file"            
            enc_fmt = "h264_bm"                           
            pix_fmt = "I420"                              
            enc_params = "width=1920:height=1080:bitrate=2000:gop=32:gop_preset=2:framerate=25"  
            cache_buffer_length = 5                      
            abort_policy = 0                             
            encoder = sail.Encoder(out_path, handle, enc_fmt, pix_fmt, enc_params, cache_buffer_length, abort_policy)
            encoder.release()
