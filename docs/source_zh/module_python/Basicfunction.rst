Basic function
_________________


主要用于获取或配置设备信息与属性。


get_available_tpu_num
>>>>>>>>>>>>>>>>>>>>>>>>>>

获取当前设备中可用智能视觉深度学习处理器的数量。

**接口形式:**
    .. code-block:: python

        def get_available_tpu_num() -> int 


**返回值说明:**

返回当前设备中可用智能视觉深度学习处理器的数量。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            tpu_len = sail.get_available_tpu_num()
            print('available tpu:',tpu_len)

set_print_flag
>>>>>>>>>>>>>>>>>>>>>>>>>>

设置是否打印程序的计算耗时信息。

**接口形式:**
    .. code-block:: python

        def set_print_flag(print_flag: bool) -> None



**参数说明:**

* print_flag: bool

print_flag为True时，打印程序的计算主要的耗时信息，否则不打印。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            sail.set_print_flag(True)

set_dump_io_flag
>>>>>>>>>>>>>>>>>>>>>>>>>>

设置是否存储输入数据和输出数据。
每个输入或输出Tensor数据将被分别保存为一个独立的二进制文件。文件名的前缀为时间戳。

**接口形式:**
    .. code-block:: python
     
        def set_dump_io_flag(dump_io_flag: bool) -> None

**参数说明:**

* dump_io_flag: bool

dump_io_flag为True时，存储输入数据和输出数据，否则不存储。


**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            sail.set_dump_io_flag(True)
            

set_loglevel
>>>>>>>>>>>>>>>>>>>>>>>>>>

设置运行过程中的日志级别为指定级别。
较低的日志等级通常用于生产环境，以减少性能开销和日志数据量，而较高的日志等级则适用于开发和调试，以便能够记录更详细的信息。

**接口形式:**
    .. code-block:: python

        def set_loglevel(sail.LogLevel loglevel) -> int 


**参数说明:**

* loglevel: LogLevel

期望的日志级别，为 ``sail.LogLevel`` 枚举值。可选的级别包括 ``TRACE``、``DEBUG``、``INFO``、``WARN``、``ERR``、``CRITICAL``、``OFF``，默认级别为 ``INFO``。

**返回值说明:**

返回类型：int

0：日志级别设置成功。
-1：传入了未知的日志级别，设置失败。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            ret = sail.set_loglevel(sail.LogLevel.TRACE)
            if (ret == 0):
                print("Set log level successfully")
            else:
                print("Unknown log level, set failed.")


set_decoder_env
>>>>>>>>>>>>>>>>>>>>>>>>>>

通过环境变量设置Decoder（也包括MultiDecoder）的参数。
必须在Decoder构造前设置，否则使用默认值。主要适用于视频解码。

**接口形式:**
    .. code-block:: python

        def set_decoder_env(env_name: str, env_value: str) -> None
            

**参数说明:**

* env_name: str

选择设置Decoder的属性名称，可选的属性名称有：

        - *'rtsp_transport'* 设置RTSP采用的传输协议。默认为TCP。
        - *'extra_frame_buffer_num'* 设置Decoder的最大缓存帧数。默认为5。
        - *'stimeout'* 设置阻塞超时时间，单位为ms。默认为20000000，即20s。
        - *'skip_non_idr'* 解码跳帧模式。0，关闭跳帧；1，跳过Non-RAP帧；2，跳过非参考帧。默认为0。
        - *'fflags'* 格式相关的flag。比如"nobuffer"。详细信息请参考ffmpeg官方文档。
        - *'rtsp_flags'* 设置RTSP是否自定义IO。默认为prefer_tcp。
        - *'refcounted_frames'* 是否使用引用计数机制。设置为1时，解码出来的图像需要程序手动释放，为0时由Decoder自动释放。
        - *'probesize'* 解析视频流时读取的最大字节数。默认为5000000。
        - *'analyzeduration'* 解析文件时读取的最大时长，单位为ms。默认为5000000。
        - *'buffer_size'* 设置缓存大小。
        - *'max_delay'* 设置最大时延。

* env_value: str

该属性的配置值

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            sail.set_decoder_env("extra_frame_buffer_num", "3") # 减小buffer以降低内存占用
            sail.set_decoder_env("probesize", "1024") # 减小probesize以降低拉流延迟
            sail.set_decoder_env("skip_non_idr", "2") # 跳过非参考帧
            dev_id = 0
            handle = sail.Handle(dev_id)
            video_path = "input_video.mp4"
            decoder = sail.Decoder(video_path, True, dev_id)            
            bmimg = decoder.read(handle)

base64_encode
>>>>>>>>>>>>>>>>>>>>>>>>>>

将字节数据进行base64编码，返回bytes类型的编码数据。不支持BM1688和CV186AH PCIE模式。

**接口形式：**
    .. code-block:: python

        def base64_encode(handle: Handle, input_bytes: bytes) -> bytes:

**参数说明:**

* handle: sail.Handle
  
设备的handle句柄，使用sail.Handle(dev_id)创建

* input_bytes: bytes

待编码的字节数据

**返回值说明**

返回base64编码的字节数据

**示例代码**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np

        if __name__ == "__main__":
            # 示例 NumPy 数组
            arr = np.array([[1, 2, 3], [4, 5, 6]])
            # 转为字节数据
            arr_bytes = arr.tobytes()
            # 创建handle，soc设备默认为dev_id为0
            handle = sail.Handle(0)
            # base字节编码
            base64_encoded_arr = sail.base64_encode(handle,arr_bytes)

base64_decode
>>>>>>>>>>>>>>>>>>>>>>>>>>>

将base64的字节编码数据进行解码，返回解码后的字节数据。不支持BM1688和CV186AH PCIE模式。

**接口形式：**
    .. code-block:: python
        
        def base64_decode(handle: Handle, encode_bytes: bytes) -> bytes:

**参数说明:**

* handle: sail.Handle
  
设备的handle句柄，使用sail.Handle(dev_id)创建

* encode_bytes: bytes

base64的字节编码数据

**返回值说明**

返回base64解码的字节数据

**示例代码**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np

        if __name__ == "__main__":
            # 示例 NumPy 数组
            arr = np.array([[1, 2, 3], [4, 5, 6]])
            shape = arr.shape
            # 转为字节数据
            arr_bytes = arr.tobytes()
            # 创建handle，soc设备默认为dev_id为0
            handle = sail.Handle(0)
            # base字节编码
            base64_encoded_arr = sail.base64_encode(handle,arr_bytes)

            # 解码数据
            base64_decode_arr = sail.base64_decode(handle,base64_encoded_arr)
            # 将生成byte数据转换为numpy数据
            res_arr = np.frombuffer(arr_bytes, dtype=np.int64).reshape(shape)

base64_encode_array
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

对numpy.array进行base64编码，生成字节编码数据。不支持BM1688和CV186AH PCIE模式。

示例代码请参考 **base64_decode_asarray** 接口提供的示例代码

**接口形式：**
    .. code-block:: python

        def base64_encode_array(handle: Handle, input_arr: numpy.ndarray) -> bytes:


**参数说明:**

* handle: sail.Handle
  
设备的handle句柄，使用sail.Handle(dev_id)创建

* input_arr: numpy.ndarray

待编码的numpy.ndarray数据

**返回值说明**

返回base64解码的字节数据

base64_decode_asarray
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

base64解码，生成numpy.array数据

**接口形式：**
    .. code-block:: python

        def base64_decode_asarray(handle: Handle, encode_arr_bytes: bytes, array_type:str = "uint8") -> numpy.ndarray:

**参数说明:**

* handle: sail.Handle

设备的handle句柄，使用sail.Handle(dev_id)创建

* encode_arr_bytes: bytes

base64编码后的numpy.ndarray的字节数据

* dtype: str

numpy.ndarray的数据类型，默认uint8，支持float、uint8、int8、int16、int32、int64

**返回值说明**

返回base64解码的一维numpay.array数组

**示例代码**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np

        if __name__ == "__main__":
            # 示例 NumPy 数组
            arr = np.array([[1,2,3],[4,5,6]],dtype=np.uint8)
            # base64编码
            base64_encoded = sail.base64_encode_array(handle,arr)
            # base64解码
            res_array = sail.base64_decode_asarray(handle,base64_encoded).reshape(shape)

get_tpu_util
>>>>>>>>>>>>>>>>>>>>>>>>>>

获取对应设备的处理器使用率

**接口形式:**
    .. code-block:: python

        def get_tpu_util(dev_id: int) -> int

**参数说明:**

* dev_id: int

需要获取处理器使用率的设备的ID。

**返回值说明:**

返回对应设备的处理器使用率百分比。

**示例代码**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            print("dev {} tpu-util is {} %".format(dev_id,sail.get_tpu_util(dev_id)))

get_vpu_util
>>>>>>>>>>>>>>>>>>>>>>>>>>

获取对应设备的VPU使用率

**接口形式:**
    .. code-block:: python

        def get_vpu_util(dev_id: int) -> list

**参数说明:**

* dev_id: int

需要获取VPU使用率的设备的ID。

**返回值说明:**

bm1684为5核vpu，返回值为长度为5的List，bm1684x为3核vpu， 返回值为长度为3的List。
List中的每项数据为对应核心的使用率百分比。

**示例代码**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            print("get_vpu_util",sail.get_vpu_util(dev_id))

get_vpp_util
>>>>>>>>>>>>>>>>>>>>>>>>>>

获取对应设备的VPP使用率

**接口形式:**
    .. code-block:: python
        
        def get_vpp_util(dev_id: int) -> list

**参数说明:**

* dev_id: int

需要获取VPP使用率的设备的ID。

**返回值说明:**

bm1684与bm1684x均为2核vpp，返回值为长度为2的List。
List中的每项数据为对应核心的使用率百分比。

**示例代码**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            print("get_vpp_util",sail.get_vpp_util(dev_id))


get_board_temp
>>>>>>>>>>>>>>>>>>>>>>>>>>

获取对应设备的板级温度。只支持PCIe模式。

**接口形式:**
    .. code-block:: python

        def get_board_temp(dev_id: int) -> int

**参数说明:**

* dev_id: int

需要获取对应板卡所在设备的ID。

**返回值说明:**

返回对应板卡的板级温度，默认单位摄氏度（℃）。

**示例代码**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            print("get_board_temp",sail.get_board_temp(dev_id))


get_chip_temp
>>>>>>>>>>>>>>>>>>>>>>>>>>

获取对应设备的处理器温度。只支持PCIe模式。

**接口形式:**
    .. code-block:: python

        def get_chip_temp(dev_id: int) -> int

**参数说明:**

* dev_id: int

需要获取处理器温度的设备的ID。

**返回值说明:**

返回对应设备的处理器的温度，默认单位摄氏度（℃）。

**示例代码**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            print("get_chip_temp",sail.get_chip_temp(dev_id))


get_dev_stat
>>>>>>>>>>>>>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: python

        def get_dev_stat(dev_id: int) -> list

**参数说明:**

* dev_id: int

需要获取内存信息的设备的ID。

**返回值说明:**

返回对应设备的内存信息列表:[mem_total,mem_used,tpu_util]。

**示例代码**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            print("get_dev_stat",sail.get_dev_stat(dev_id))
