sail.MultiDecoder
____________________

多路解码接口，支持同时解码多路视频。

\_\_init\_\_
>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: python

        def __init__(self,
                    queue_size: int = 10, 
                    tpu_id: int = 0, 
                    discard_mode: int = 0)

**参数说明:**

* queue_size: int

输入参数。每路视频，解码缓存图像队列的长度。

* tpu_id: int

输入参数。使用的智能视觉深度学习处理器 id，默认为0。

* discard_mode: int

输入参数。缓存达到最大值之后，数据的丢弃策略。0表示不再放数据进缓存；1表示先从队列中取出队列头的图片，丢弃之后再将解码出来的图片缓存进去。默认为0。


set_read_timeout
>>>>>>>>>>>>>>>>>>>>

设置读取图片的超时时间，对read和read_接口生效，超时之后仍然没有获取到图像，结果就会返回。

**接口形式:**
    .. code-block:: python

        def set_read_timeout(self, timeout: int) -> None

**参数说明:**

* timeout: int

输入参数。超时时间，单位是秒。

**示例代码:**
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

添加一个通道。

**接口形式1:**

通过该接口添加的通道，不会自动停止解码，会自动循环，直到该MultiDecoder析构，或者调用del_channel。

    .. code-block:: python

        def add_channel(self,
                    file_path: str, 
                    frame_skip_num: int = 0) -> int
            
**参数说明1:**

* file_path: str

输入参数。视频的路径或者链接。

* frame_skip_num: int

输入参数。解码缓存的主动丢帧数，默认是0，不主动丢帧。

**返回值说明1:**

返回视频对应的唯一的通道号。类型为整形。

**示例代码1:**
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

**接口形式2:**

该接口添加通道时，支持设置循环次数。仅适用于解码本地视频文件的场景。

    .. code-block:: python

        def add_channel(self,
                    file_path: str, 
                    frame_skip_num
                    loopnum) -> int
            
**参数说明2:**

* file_path: str

输入参数。视频的路径或者链接。

* frame_skip_num: int

输入参数。解码缓存的主动丢帧数。设置为0表示不主动丢帧。

* loopnum: int

输入参数。解码循环次数。设置为0表示不循环，解码一遍后停止。

**返回值说明2:**

返回视频对应的唯一的通道号。类型为整形。

**示例代码2:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            queue_size=16
            dev_id=0
            discard_mode=0
            multiDecoder = sail.MultiDecoder(queue_size,dev_id,discard_mode)
            channel_list = []
            file_path = "your_video_path"
            frame_skip_num: int = 0
            loopnum: int = 0 # no loop
            for i in range(4):
                idx = multiDecoder.add_channel(file_path, frame_skip_num, loopnum)
                if(idx<0):
                    exit(-1)
                channel_list.append(idx) 

del_channel
>>>>>>>>>>>>>>>

删除一个已经添加的视频通道。

**接口形式:**
    .. code-block:: python

        def del_channel(self, channel_idx: int) -> int 

**参数说明:**

* channel_idx: int

输入参数。将要删除视频的通道号。

**返回值说明**

成功返回0，其他值时表示失败。

**示例代码:**
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

清除指定通道的图片缓存。

**接口形式:**
    .. code-block:: python

        def clear_queue(self, channel_idx: int) -> int 


**参数说明:**

* channel_idx: int

输入参数。将要删除视频的通道号。

**返回值说明:**

成功返回0，其他值时表示失败。 

**示例代码:**
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

从指定的视频通道中获取一张图片。

**接口形式1:**
    .. code-block:: python

        def read(self,
                channel_idx: int, 
                image: BMImage, 
                read_mode: int = 0) -> int 

**参数说明1:**

* channel_idx: int

输入参数。指定的视频通道号。

* image: BMImage

输出参数。解码出来的图片。

* read_mode: int

输入参数。获取图片的模式，
0表示不等待，直接从缓存中读取一张，无论有没有读取到都会返回。
其他的表示等到获取到图片之后或等待时间超时再返回。

**返回值说明1:**

成功返回0，其他值时表示失败。 

**示例代码:**
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

**接口形式2:**
    .. code-block:: python

        def read(self, channel_idx: int) -> BMImage 

**参数说明2:**

* channel_idx: int

输入参数。指定的视频通道号。

**返回值说明2:**

返回解码出来的图片，类型为BMImage。  
            
**示例代码:**
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

从指定的视频通道中获取一张图片，通常是要和BMImageArray一起使用。

**接口形式1:**
    .. code-block:: python

        def read_(self,
                channel_idx: int, 
                image: bm_image, 
                read_mode: int=0) -> int 

**参数说明1:**

* channel_idx: int

输入参数。指定的视频通道号。

* image: bm_image

输出参数。解码出来的图片。

* read_mode: int

输入参数。获取图片的模式，0表示不等待，直接从缓存中读取一张，无论有没有读取到都会返回。其他的表示等到获取到图片之后或等待时间超时再返回。

**返回值说明1:**

成功返回0，其他值时表示失败。

**示例代码:**
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

**接口形式2:**
    .. code-block:: python

        def read_(self, channel_idx: int) -> int:
            """ Read a bm_image from the MultiDecoder with a given channel.

**参数说明2:**

* channel_idx: int

输入参数。指定的视频通道号。

**返回值说明2:**

返回解码出来的图片，类型为bm_image。

**示例代码:**
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

重连相应的通道的视频。

**接口形式:**
    .. code-block:: python
        
        def reconnect(self, channel_idx: int) -> int 

**参数说明:**

* channel_idx: int

输入参数。输入图像的通道号。

**返回值说明**

成功返回0，其他值时表示失败。

**示例代码:**
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

获取相应通道的图像shape。

**接口形式:**
    .. code-block:: python

        def get_frame_shape(self, channel_idx: int) -> list[int]
            
**参数说明:**

输入参数。输入图像的通道号。
        
**返回值说明**

返回一个由1，通道数，图像高度，图像宽度组成的list。

**示例代码:**
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

设置视频是否为本地视频。如果不调用则表示为视频为网络视频流。

**接口形式:**
    .. code-block:: python

        def set_local_flag(self, flag: bool) -> None:
 
**参数说明:**

* flag: bool

标准位，如果为True，每路视频每秒固定解码25帧

**示例代码:**
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

获取指定视频通道的视频帧数

**接口形式:**
    .. code-block:: python

        def get_channel_fps(self, channel_idx: int) -> float:
 
**参数说明:**

* channel_idx: int

指定需要获取视频帧数的视频通道号

**返回值说明**

返回指定视频通道的视频帧数

**示例代码:**
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

获取丢帧数。

**接口形式:**
    .. code-block:: python
        
        def get_drop_num(self, channel_idx: int) -> int:
            
**参数说明:**

输入参数。输入图像的通道号。
        
**返回值说明**

返回一个数代表丢帧数

**示例代码:**
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

设置丢帧数为0。

**接口形式:**
    .. code-block:: python

        def reset_drop_num(self, channel_idx: int) -> None:
 
**参数说明:**

输入参数。输入图像的通道号。

**示例代码:**
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

get_channel_status
>>>>>>>>>>>>>>>>>>

获取指定通道的解码器状态。

**接口形式:**
    .. code-block:: python

        def get_channel_status(self, channel_idx: int) -> sail.DecoderStatus

**参数说明:**

    - ``channel_idx`` (int): 要查询状态的通道索引。

**返回值说明:**

    返回一个 ``sail.DecoderStatus`` 枚举值，表示指定通道的解码器状态。
    比如 ``sail.DecoderStatus.OPENED`` 或 ``sail.DecoderStatus.CLOSED``。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            queue_size = 10
            dev_id = 0
            discard_mode = 0
            multiDecoder = sail.MultiDecoder(queue_size, dev_id, discard_mode)
            channel_list = []
            file_path_list = ["your_video_path" for i in range(4)]
            for i,file_path in enumerate(file_path_list):
                idx = multiDecoder.add_channel(file_path)
                if idx < 0:
                    print(f"Failed to add channel {i}. Error: {idx}")
                    continue
                channel_list.append(idx)
                status = multiDecoder.get_channel_status(idx)
                print(f"Channel {i} status: {status}")

is_channel_eof
>>>>>>>>>>>>>>>>>>

查询某个通道的解码器是否已经到达文件结尾。

**接口形式:**
    .. code-block:: python

        def is_channel_eof(self, channel_idx: int) -> bool

**参数说明:**

    - ``channel_idx`` (int): 需要查询的通道索引。

**返回值说明:**

    如果该通道已经到达文件结尾（EOF, end of file），则返回True，否则返回False。

    如果不存在索引对应的通道，则抛出异常。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            filepath = "./jellyfish_200frames.mkv"
            FRAME_NUM = 200
            md = sail.MultiDecoder()
            md.set_local_flag(True)
            frame_skip_num: int = 0
            loopnum: int = 0 # no loop
            idx = md.add_channel(filepath, frame_skip_num, loopnum)
            assert idx == 0
            cnt = 0
            while True:
                img = sail.BMImage()
                read_mode = 1 # wait block
                ret = md.read(idx, img, read_mode)
                if ret != 0:
                    if (md.get_channel_status(idx) == sail.DecoderStatus.CLOSED and md.is_channel_eof(idx)):
                        print(f"Channel {idx} reached EOF, total read {cnt} images, decode thread will stop")
                        assert cnt == FRAME_NUM, "total frame number mismatch!"
                        break
                    else:
                        print(f"Channel {idx} read meet an error, ret = {ret}, total read {cnt} images")
                        break
                else:
                    cnt += 1
                    print(f"Channel {idx} read {cnt} images")