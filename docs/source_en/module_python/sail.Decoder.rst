sail.Decoder
____________

\_\_init\_\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: python

        def __init__(self, 
                    file_path: str, 
                    compressed: bool = True, 
                    tpu_id: int = 0)

**Parameters**

* file_path : str

Path or rtsp url to the video/image file

* compressed : bool, default: True

Whether the format of decoded output is compressed NV12.

* tpu_id: int, default: 0

ID of Tensor Computing Processor, there may be more than one Tensor Computing Processor for PCIE mode.


is_opened
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Judge if the source is opened successfully.

**Interface:**
    .. code-block:: python

        def is_opened(self)-> bool

**Returns**

* judge_ret : bool

True for success and False for failure

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            file_path = 'your_video_file_path.mp4'  
            dev_id = 0
            decoder = sail.Decoder(file_path, True, dev_id)
            ret = decoder.is_opened()
            print("Decoder opened:", ret)

read
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Read an image from the Decoder.

**Interface:**
    .. code-block:: python

        def read(self, handle: sail.Handle, image: sail.BMImage)-> int

**Parameters**

* handle : sail.Handle

Handle instance

* image : sail.BMImage

BMImage instance

**Returns**

* judge_ret : int

0 for success and others for failure


**Interface:**
    .. code-block:: python

        def read(self, handle: sail.Handle)-> sail.BMImage

**Parameters**

* handle : sail.Handle

Handle instance

**Returns**

* image : sail.BMImage

BMImage instance

**Sample1:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            file_path = 'your_video_file_path.mp4'  
            dev_id = 0
            handle = sail.Handle(dev_id)
            decoder = sail.Decoder(file_path, True, dev_id)
            image = sail.BMImage()
            ret = decoder.read(handle, image)
            if ret == 0:
                print("Frame read successfully")
            else:
                print("Failed to read frame")

**Sample2:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            file_path = 'your_video_file_path.mp4'  
            dev_id = 0
            handle = sail.Handle(dev_id)
            decoder = sail.Decoder(file_path, True, dev_id)
            BMimg = decoder.read(handle)

read\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Read an image from the Decoder.

**Interface:**
    .. code-block:: python

        def read_(self, handle: sail.Handle, image: sail.bm_image)-> int

**Parameters**

* handle : sail.Handle

Handle instance

* image : sail.bm_image

bm_image instance

**Returns**

* judge_ret : int

0 for success and others for failure

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            file_path = 'your_video_file_path.mp4'  
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

Get frame shape in the Decoder.

**Interface:**
    .. code-block:: python

        def get_frame_shape(self)-> list

**Returns**

* frame_shape : list

The shape of the frame

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            file_path = 'your_video_file_path.mp4'  
            dev_id = 0
            handle = sail.Handle(dev_id)
            decoder = sail.Decoder(file_path,True,dev_id)
            print(decoder.get_frame_shape())



release
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Release the Decoder.

**Interface:**
    .. code-block:: python
    
        def release(self)

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            file_path = 'your_video_file_path.mp4'  
            handle = sail.Handle(dev_id)
            decoder = sail.Decoder(file_path,True,dev_id)
            decoder.release()


reconnect
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Reconnect the Decoder.

**Interface:**
    .. code-block:: python

        def reconnect(self)

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            file_path = "your_video_file_path.mp4"
            decoder = sail.Decoder(file_path, True, dev_id)
            decoder.reconnect()


enable_dump
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Enable the dump input video ability of the decoder (without encoding) and cache up to 1000 frames of undecoded video.

**Interface:**
    .. code-block:: python
    
        def enable_dump(dump_max_seconds: int):
            """ enable input video dump without encode.
            """

**Parameters**

* dump_max_seconds : int

dump video max length.

**Sample:**
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

Disable the dump input video ability of the decoder and clear the cache queue.

**Interface:**
    .. code-block:: python
    
        def disable_dump():
            """ Disable  input video dump without encode.
            """
**Sample:**
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

At the time of calling this function, dump the input video for several seconds before and after. Due to the lack of encoding, it is necessary to dump the keyframes that all frames depend on within a few seconds before and after. Therefore, the dump implementation of the interface is based on gop, and the actual video duration under dump will be higher than the input parameter duration. The error depends on the gop of the input video. The larger the size and gop, the larger the error.

**Interface:**
    .. code-block:: python
    
        def dump(dump_pre_seconds, dump_post_seconds, file_path)->int
            """ dump input video without encode.
        
            Parameters:
            ----------
            dump_pre_seconds : int
                dump video length(seconds) before dump moment
            dump_post_seconds : int
                dump video length(seconds) after dump moment
            file_path : str
                output path
                
            Returns
            -------
            int, 0 for success
            """
**Sample:**
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

Get pts or dts.

**Interface:**
    .. code-block:: python

        def get_pts_dts() -> list


**Returns**

* result : list

the value of pts and dts.

**Sample:**
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