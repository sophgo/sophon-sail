sail.Decoder
____________

解码器，可实现图像或视频的解码。

**图像解码像素格式支持说明:**

* 硬解支持jpeg baseline，不是所有的jpeg
* 视频支持硬解h264,h265。输出的像素格式为YUV-nv12、YUVJ420P或者YUV420P；

\_\_init\_\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

初始化Decoder。

**接口形式:**
    .. code-block:: python

        def __init__(self, file_path: str, compressed: bool=True, tpu_id: int=0)

**参数说明:**

* file_path: str

图像或视频文件的Path或RTSP的URL。

* compressed: bool

是否将解码的输出压缩为NV12，default：True。
开启之后可以节省内存、节省带宽，但是输入视频必须要满足宽能被16整除才行，且输入必须为视频时才能生效。

* tpu_id: int

设置使用的智能视觉深度学习处理器 id号。

is_opened
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

判断源文件是否打开。

**接口形式:**
    .. code-block:: python

        def is_opened(self) -> bool

**返回值说明:**

* judge_ret: bool

打开成功返回True，失败返回False。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            file_path = 'your_video_file_path.mp4'  # 请替换为您的文件路径
            dev_id = 0
            decoder = sail.Decoder(file_path, True, dev_id)
            ret = decoder.is_opened()
            print("Decoder opened:", ret)


read
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

从Decoder中读取一帧图像。

**接口形式1:**
    .. code-block:: python

        def read(self, handle: sail.Handle, image: sail.BMImage)->int
        
**参数说明1:**

* handle: sail.Handle

输入参数。Decoder使用的智能视觉深度学习处理器的Handle。

* image: sail.BMImage

输出参数。将数据读取到image中。

**返回值说明1:**

* judge_ret: int

读取成功返回0，失败返回其他值。


**接口形式2:**
    .. code-block:: python

        def read(self, handle: sail.Handle)->sail.BMImage

**参数说明2:**

* handle: sail.Handle

输入参数。Decoder使用的智能视觉深度学习处理器的Handle。

**返回值说明2:**

* image: sail.BMImage

将数据读取到image中。

**示例代码1:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            file_path = 'your_video_file_path.mp4'  # 请替换为您的文件路径
            dev_id = 0
            handle = sail.Handle(dev_id)
            decoder = sail.Decoder(file_path, True, dev_id)
            image = sail.BMImage()
            ret = decoder.read(handle, image)
            if ret == 0:
                print("Frame read successfully")
            else:
                print("Failed to read frame")

**示例代码2:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            file_path = 'your_video_file_path.mp4'  # 请替换为您的文件路径
            dev_id = 0
            handle = sail.Handle(dev_id)
            decoder = sail.Decoder(file_path, True, dev_id)
            BMimg = decoder.read(handle)

read\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

从Decoder中读取一帧图像。

**接口形式:**
    .. code-block:: python

        def read_(self, handle: sail.Handle, image: sail.bm_image)->int

**参数说明:**

* handle: sail.Handle

输入参数。Decoder使用的智能视觉深度学习处理器的Handle。

* image: sail.bm_image

输出参数。将数据读取到image中。

**返回值说明:**

* judge_ret: int

读取成功返回0，失败返回其他值。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            file_path = 'your_video_file_path.mp4'  # 请替换为您的文件路径
            dev_id = 0
            handle = sail.Handle(dev_id)
            decoder = sail.Decoder(file_path, True, dev_id)
            image = sail.BMImage()
            bm_img = image.data()
            ret = decoder.read_(handle, bm_img)  
            if ret == 0:
                print("Frame read successfully into bm_image")
            else:
                print("Failed to read frame into bm_image")


get_frame_shape
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取Decoder中frame中的shape。

**接口形式:**
    .. code-block:: python

        def get_frame_shape(self)->list

**返回值说明:**

* frame_shape: list

返回当前frame的shape。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            file_path = 'your_video_file_path.mp4'  # 请替换为您的文件路径
            dev_id = 0
            handle = sail.Handle(dev_id)
            decoder = sail.Decoder(file_path,True,dev_id)
            print(decoder.get_frame_shape())

release
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

释放Decoder资源。

**接口形式:**
    .. code-block:: python
    
        def release(self) -> None

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            file_path = 'your_video_file_path.mp4'  # 请替换为您的文件路径
            handle = sail.Handle(dev_id)
            decoder = sail.Decoder(file_path,True,dev_id)
            decoder.release()


reconnect
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Decoder再次连接。

**接口形式:**
    .. code-block:: python

        def reconnect(self) -> None
        
**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            file_path = "your_video_file_path.mp4"
            decoder = sail.Decoder(file_path, True, dev_id)
            # 重新连接解码器
            decoder.reconnect()


enable_dump
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

开启解码器的dump输入视频功能（不经编码），并缓存最多1000帧未解码的视频。

**接口形式:**
    .. code-block:: python
    
        def enable_dump(dump_max_seconds: int):

**参数说明:**

* dump_max_seconds: int

输入参数。dump视频的最大时长，也是内部AVpacket缓存队列的最大长度。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            file_path = "your_video_file_path.mp4"
            decoder = sail.Decoder(file_path, True, dev_id)
            dump_max_seconds = 100
            decoder.enable_dump(dump_max_seconds)

disable_dump
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

关闭解码器的dump输入视频功能，并清空开启此功能时缓存的视频帧

**接口形式:**
    .. code-block:: python
    
        def disable_dump():
            """ Disable  input video dump without encode.
            """
**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            file_path = "your_video_file_path.mp4"
            decoder = sail.Decoder(file_path, True, dev_id)
            decoder.enable_dump(100)
            decoder.disable_dump()

dump
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

在调用此函数的时刻，dump下前后数秒的输入视频。由于未经编码，必须dump下前后数秒内所有帧所依赖的关键帧。因而接口的dump实现以gop为单位，实际dump下的视频时长将高于输入参数时长。误差取决于输入视频的gop_size，gop越大，误差越大。

**接口形式:**
    .. code-block:: python
    
        def dump(dump_pre_seconds: int, dump_post_seconds: int, file_path: str)->int

* dump_pre_seconds: int

输入参数。保存调用此接口时刻之前的数秒视频。

* dump_post_seconds: int

输入参数。保存调用此接口时刻之后的数秒视频。

* file_path: str

输入参数。视频路径。

**返回值说明:**

* judge_ret: int

成功返回0,失败返回其他值。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            input_file_path = "your_rtsp_url"
            handle = sail.Handle(dev_id)
            decoder = sail.Decoder(input_file_path, True, dev_id)
            decoder.enable_dump(30)
            dump_pre_seconds = 10
            dump_post_seconds = 10
            output_file_path = "output_video_path.mp4"

            # start decode
            t_decode = time.time()
            while(True):
                if time.time() - t_decode > dump_pre_seconds:
                        break
                _ = decoder.read(handle)

            # start dump
            ret = decoder.dump(dump_pre_seconds, dump_post_seconds, output_file_path)
            if ret == 0:
                print("Decoder dump start!")
            else:
                print("Decoder dump fail!")
                exit(-1)

            # continue decode
            t_dump = time.time()
            while(True):
                if time.time() - t_dump > dump_post_seconds:
                    print("Decoder dump finish!")
                    break
                _ = decoder.read(handle)

            time.sleep(1)
            print("exit")


get_pts_dts
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取pts和dts

**接口形式:**
    .. code-block:: python

        def get_pts_dts() -> list
    
**返回值说明:**

* result: list

输出结果。输出具体的pts和dts值。


**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            input_file_path = 'your_rtsp_url'  
            dev_id = 0
            handle = sail.Handle(dev_id)
            decoder = sail.Decoder(input_file_path, True, dev_id)
            image = sail.BMImage()
            ret = decoder.read(handle, image)
            if ret == 0:
                print("Frame read successfully into bm_image")
                pts,dts=decoder.get_pts_dts()
                print("pts:",pts)
                print("dts:",dts)
            else:
                print("Failed to read frame into bm_image")
            