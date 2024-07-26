sail.MultiDecoder
____________________

Multichannel decoding interface, supporting simultaneous decoding of multichannel video.

\_\_init\_\_
>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: python

        def __init__(self,
                    queue_size: int = 10, 
                    tpu_id: int = 0, 
                    discard_mode: int = 0)

**Parameters:**

* queue_size: int

For each video, the length of the decoded cached image queue.

* tpu_id: int

Tpu id that used, which defaults to 0.

* discard_mode: int

Data discard method when the cache reaches the maximum value. \
0 indicates that data is not put into the cache; \
1 indicates that the image of the queue header is poped , \
and then push into the decoded image. The default value is 0.

set_read_timeout
>>>>>>>>>>>>>>>>>>>>

Set the timeout period for reading images. This takes effect on the read and read\_ interfaces.\
If the image is not obtained after the timeout, the result will be returned.

**Interface:**
    .. code-block:: python

        def set_read_timeout(self, timeout: int) -> None

**Parameters:**

* timeout: int

Timeout period, in seconds.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            queue_size=16
            dev_id=0
            discard_mode=0
            multiDecoder = sail.MultiDecoder(queue_size,dev_id,discard_mode) 
            multiDecoder.set_read_timeout(100)

add_channel
>>>>>>>>>>>>>>>>

Add a channel

**Interface:**
    .. code-block:: python

        def add_channel(self,
                    file_path: str, 
                    frame_skip_num: int = 0) -> int
            
**Parameters:**

* file_path: str

The path or link to the video.

* frame_skip_num: int

Number of active frame loss in decoded cache. The default value is 0, which means no active frame loss.

**Returns**

Returns the unique channel number corresponding to the video. The type is an integer.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            queue_size=16
            dev_id=0
            discard_mode=0
            multiDecoder = sail.MultiDecoder(queue_size,dev_id,discard_mode)
            channel_list = []
            file_path = "your_video_path"
            for i in range(4):
                idx = multiDecoder.add_channel(file_path)
                if(idx<0):
                    exit(-1)
                channel_list.append(idx) 

del_channel
>>>>>>>>>>>>>>>

Delete a video channel that has been added.

**Interface:**
    .. code-block:: python

        def del_channel(self, channel_idx: int) -> int 

**Parameters:**

* channel_idx: int

The channel number of the video to be deleted.

**Returns**

Return 0 on success and other values on failure.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            queue_size=16
            dev_id=0
            discard_mode=0
            multiDecoder = sail.MultiDecoder(queue_size,dev_id,discard_mode)
            channel_list = []
            
            file_path = "your_video_path"
            for i in range(4):
                idx = multiDecoder.add_channel(file_path)
                if(idx<0):
                    exit(-1)
                channel_list.append(idx) 
            ret = multiDecoder.del_channel(0)
            if(ret!=0):
                print("delete channel error!")

clear_queue
>>>>>>>>>>>>>>>

Clears the image cache for the specified channel.

**Interface:**
    .. code-block:: python

        def clear_queue(self, channel_idx: int) -> int 


**Parameters:**

* channel_idx: int

The channel number of the video to be deleted.

**Returns:**

Return 0 on success and other values on failure. 
            
**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            queue_size=16
            dev_id=0
            discard_mode=0
            multiDecoder = sail.MultiDecoder(queue_size,dev_id,discard_mode)
            channel_list = []
            file_path = "your_video_path"
            for i in range(4):
                idx = multiDecoder.add_channel(file_path)
                if(idx<0):
                    exit(-1)
                channel_list.append(idx) 
            ret = multiDecoder.clear_queue(0)
            if(ret!=0):
                print(" Clear failure!")
       
read
>>>>>>>>

Gets an image from the specified video channel.

**Interface1:**
    .. code-block:: python

        def read(self,
                channel_idx: int, 
                image: BMImage, 
                read_mode: int = 0) -> int 

**Parameters1:**

* channel_idx: int

The specified video channel number.

* image: BMImage

The decoded image.

* read_mode: int

Mode of obtaining images. 0 indicates that one image is read directly from the cache \
without waiting, and will be returned whether it is read or not. \
Others representations wait until the image is retrieved or timeout, and then return.


**Returns1:**

Return 0 on success and other values on failure. 
  
**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            queue_size=16
            dev_id=0
            discard_mode=0
            multiDecoder = sail.MultiDecoder(queue_size,dev_id,discard_mode)
            channel_list = []
            frame_list = []
            file_path = "your_video_path"
            for i in range(4):
                idx = multiDecoder.add_channel(file_path)
                if(idx<0):
                    exit(-1)
                channel_list.append(idx)
                frame_list.append([])
            count = 0
            while True:
                count += 1
                for idx in channel_list:
                    bmimg = sail.BMImage()
                    ret = multiDecoder.read(idx,bmimg,1)
                    frame_list[idx].append(bmimg)
                if count == 20:
                    break

**Interface2:**
    .. code-block:: python

        def read(self, channel_idx: int) -> BMImage 

**Parameters2:**

* channel_idx: int

The specified video channel number.

**Returns2:**

Returns the decoded image of type BMImage.
            
**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            queue_size=16
            dev_id=0
            discard_mode=0
            multiDecoder = sail.MultiDecoder(queue_size,dev_id,discard_mode)
            channel_list = []
            frame_list = []
            file_path = "your_video_path"
            for i in range(4):
                idx = multiDecoder.add_channel(file_path)
                if(idx<0):
                    exit(-1)
                channel_list.append(idx)
                frame_list.append([])
            count = 0
            while True:
                count += 1
                for idx in channel_list:
                    bmimg = multiDecoder.read(idx)
                    frame_list[idx].append(bmimg)
                if count == 20:
                    break 

read\_
>>>>>>>>

Gets an image from the specified video channel, usually used with BMImageArray.

**Interface1:**
    .. code-block:: python

        def read_(self,
                channel_idx: int, 
                image: bm_image, 
                read_mode: int=0) -> int 

**Parameters1:**

* channel_idx: int

The specified video channel number.

* image: bm_image

The decoded image.

* read_mode: int

Mode of obtaining images. 0 indicates that one image is read directly from the cache \
without waiting, and will be returned whether it is read or not. \
Others representations wait until the image is retrieved or timeout, and then return.

**Returns1:**

Return 0 on success and other values on failure. 

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            queue_size=16
            dev_id=0
            discard_mode=0
            multiDecoder = sail.MultiDecoder(queue_size,dev_id,discard_mode)
            channel_list = []
            frame_list = []
            file_path = "your_video_path"
            for i in range(4):
                idx = multiDecoder.add_channel(file_path)
                if(idx<0):
                    exit(-1)
                channel_list.append(idx)
                frame_list.append([])
            count = 0
            while True:
                count += 1
                for idx in channel_list:
                    img = sail.BMImage()
                    bmimg = img.data()
                    ret = multiDecoder.read_(idx,bmimg,1)
                    frame_list[idx].append(bmimg)
                if count == 20:
                    break 

**Interface2:**
    .. code-block:: python

        def read_(self, channel_idx: int) -> bm_image:

**Parameters2:**

* channel_idx: int

The specified video channel number.

**Returns2:**

Returns the decoded image of type bm_image.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            queue_size=16
            dev_id=0
            discard_mode=0
            multiDecoder = sail.MultiDecoder(queue_size,dev_id,discard_mode)
            channel_list = []
            frame_list = []
            file_path = "your_video_path"
            for i in range(4):
                idx = multiDecoder.add_channel(file_path)
                if(idx<0):
                    exit(-1)
                channel_list.append(idx)
                frame_list.append([])
            count = 0
            while True:
                count += 1
                for idx in channel_list:
                    bmimg = multiDecoder.read_(idx)
                    frame_list[idx].append(bmimg)
                if count == 20:
                    break 

reconnect
>>>>>>>>>>>>>>

Reconnect the video of the corresponding channel.

**Interface:**
    .. code-block:: python
        
        def reconnect(self, channel_idx: int) -> int 

**Parameters:**

* channel_idx: int

The channel index of the input image.

**Returns**

Return 0 on success and other values on failure. 

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            queue_size=16
            dev_id=0
            discard_mode=0
            multiDecoder = sail.MultiDecoder(queue_size,dev_id,discard_mode)
            channel_list = []
            file_path = "your_video_path"
            for i in range(4):
                idx = multiDecoder.add_channel(file_path)
                if(idx<0):
                    exit(-1)
                channel_list.append(idx)
            ret = multiDecoder.reconnect(0)
            print(ret)
            
get_frame_shape
>>>>>>>>>>>>>>>>>>

Get the image shape of the corresponding channel.

**Interface:**
    .. code-block:: python

        def get_frame_shape(self, channel_idx: int) -> list[int]
            
**Parameters:**

* channel_idx: int

The channel index of the input image.
        
**Returns**

Returns a list:[1, number of channels, image height, image width].

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            queue_size=16
            dev_id=0
            discard_mode=0
            multiDecoder = sail.MultiDecoder(queue_size,dev_id,discard_mode)
            channel_list = []
            file_path = "your_video_path"
            for i in range(4):
                idx = multiDecoder.add_channel(file_path)
                if(idx<0):
                    exit(-1)
                channel_list.append(idx)
                print(multiDecoder.get_frame_shape(idx))
           

set_local_flag
>>>>>>>>>>>>>>>>>>

Set whether the video is a local video. If it is not called, the video is represented \
as a network video stream.

**Interface:**
    .. code-block:: python

        def set_local_flag(self, flag: bool) -> None:
 
**Parameters:**

* flag: bool

Standard bit, if True, fixed decoding of 25 frames per second per video channel

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            queue_size=16
            dev_id=0
            discard_mode=0
            multiDecoder = sail.MultiDecoder(queue_size,dev_id,discard_mode)
            multiDecoder.set_local_flag(True)

get_channel_fps
>>>>>>>>>>>>>>>>>>

Get the video fps of the specified channel

**Interface**
    .. code-block:: python

        def get_channel_fps(self, channel_idx: int) -> float:
 
**Parameters**

* channel_idx: int

The specified channel index

**Returns**

Returns the video fps of the specified channel

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            queue_size=16
            dev_id=0
            discard_mode=0
            multiDecoder = sail.MultiDecoder(queue_size,dev_id,discard_mode)
            channel_list = []
            file_path = "your_video_path"
            for i in range(4):
                idx = multiDecoder.add_channel(file_path)
                if(idx<0):
                    exit(-1)
                channel_list.append(idx)
                print(multiDecoder.get_channel_fps(idx))

get_drop_num
>>>>>>>>>>>>>>>>>>

Obtain the number of dropped frames.

**Interface:**
    .. code-block:: python

        def get_drop_num(self, channel_idx: int) -> int
 
**Parameters:**

* channel_idx: int

The channel index of the input image.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            queue_size=16
            dev_id=0
            discard_mode=0
            multiDecoder = sail.MultiDecoder(queue_size,dev_id,discard_mode)
            channel_list = []
            file_path = "your_video_path"
            for i in range(4):
                idx = multiDecoder.add_channel(file_path)
                if(idx<0):
                    exit(-1)
                channel_list.append(idx)
                print(multiDecoder.get_drop_num(idx))

reset_drop_num
>>>>>>>>>>>>>>>>>>

Set the number of dropped frames to zeros.

**Interface:**
    .. code-block:: python

        def reset_drop_num(self,  channel_idx: int) -> None:
 
**Parameters:**

* channel_idx: int

The channel index of the input image.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            queue_size=16
            dev_id=0
            discard_mode=0
            multiDecoder = sail.MultiDecoder(queue_size,dev_id,discard_mode)
            channel_list = []
            file_path = "your_video_path"
            for i in range(4):
                idx = multiDecoder.add_channel(file_path)
                if(idx<0):
                    exit(-1)
                channel_list.append(idx)
                multiDecoder.reset_drop_num(idx)