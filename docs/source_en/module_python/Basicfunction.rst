Basic function
_________________


get_available_tpu_num
>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the number of available Tensor Computing Processors.

**Interface:**
    .. code-block:: python

        def get_available_tpu_num()->int
            

**Returns:**

* tpu_num : int

Number of available Tensor Computing Processors

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            tpu_len = sail.get_available_tpu_num()
            print('available tpu:',tpu_len)


set_print_flag
>>>>>>>>>>>>>>>>>>>>>>>>>>

Print main process time use.

**Interface:**
    .. code-block:: python

        def set_print_flag(print_flag: bool)

**Parameters:**

* print_flag : bool

If print_flag is True, print main process time use, Otherwise not print.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            set_print_flag(True)


set_dump_io_flag
>>>>>>>>>>>>>>>>>>>>>>>>>>

Dump input date and output date.

**Interface:**
    .. code-block:: python
     
        def set_dump_io_flag(dump_io_flag: bool)
           
**Parameters:**

* dump_io_flag : bool

If dump_io_flag is True, dump input data and output data, Otherwise not dump.


**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            set_dump_io_flag(True)

set_loglevel
>>>>>>>>>>>>>>>>>>>>>>>>>>

Set the logging level to the specified level.
Lower log levels are typically used in production environments to reduce performance overhead and the volume of log data, 
while higher log levels are suitable for development and debugging in order to capture more detailed information.

**Interface:**
    .. code-block:: python

        def set_loglevel(sail.LogLevel loglevel) -> int 


**Parameters:**

* loglevel: LogLevel

The Target log level as a sail.LogLevel enum value. 
The optional values include ``TRACE``, ``DEBUG``, ``INFO``, ``WARN``, ``ERR``, ``CRITICAL``, ``OFF``, and the default level is ``INFO``.

**Returns:**

return: int

Returning 0 indicates the log level was set successfully, 
whereas returning -1 indicates a failure due to an unknown log level.

**Sample:**
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

Set the parameters for the Decoder (including MutlDecoder) through environment variables. 
These must be set before the Decoder is constructed; otherwise, the default values will be used. 
This is mainly applicable to video decoding.

**Interface:**
    .. code-block:: python

        def set_decoder_env(env_name: str, env_value: str)
            
**Parameters:**

* env_name: str

The property name to set for the Decoder. The available property names are as follows:

        - *'rtsp_transport'*: The transport protocol used for RTSP. The default is TCP.
        - *'extra_frame_buffer_num'*: The maximum number of cached frames for the Decoder. The default is 5.
        - *'stimeout'*: Raise error timeout, in milliseconds. The default is 20000000, i.e., 20 seconds.
        - *'skip_non_idr'*: Decoding frame skip mode. 0, no skip; 1, skip Non-RAP frames; 2, skip non-reference frames. The default is 0.
        - *'fflags'*: format flags, like "nobuffer". Read ffmpeg official docs for more details.
        - *'rtsp_flags'*: Set RTSP flags. The default is prefer_tcp.
        - *'refcounted_frames'*: When set to 1, the decoded images need to be manually released by the program; when set to 0, they are automatically released by the Decoder.
        - *'probesize'*: the max size of the data to analyze to get stream information. 5000000 by default.
        - *'analyzeduration'*: How many microseconds are analyzed to probe the input. 5000000 microseconds by default.
        - *'buffer_size'*: The maximum socket buffer size in bytes.
        - *'max_delay'*: Maximum demuxing delay in microseconds.

* env_value: str

Environment value.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            sail.set_decoder_env("extra_frame_buffer_num", "3") # Decrease buffer num for lower memory usage
            sail.set_decoder_env("probesize", "1024") # Decrease probesize for lower latency
            sail.set_decoder_env("skip_non_idr", "2") # skip non-reference frames
            dev_id = 0
            handle = sail.Handle(dev_id)
            image_name = "your_video.mp4"
            decoder = sail.Decoder(image_name, True, dev_id)            
            bmimg = decoder.read(handle)

            
base64_encode
>>>>>>>>>>>>>>>>>>>>>>>>>>

Encode byte data into base64 and return the encoded data as bytes. BM1688 and CV186AH PCIE mode are not supported.

**Interface:**
    .. code-block:: python

        def base64_encode(handle: Handle, input_bytes: bytes) -> bytes:

**Parameters:**

* handle: Handle

The handle of the device, created using sail.Handle(dev_id).

* input_bytes: bytes

The byte data to be encoded.


**Returns:**

* bytes
 
The byte data encoded in base64.

**Sample**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np

        if __name__ == "__main__":
            arr = np.array([[1, 2, 3], [4, 5, 6]])
            # to bytes
            arr_bytes = arr.tobytes()
            # get handle
            handle = sail.Handle(0)
            # base64 encode
            base64_encoded_arr = sail.base64_encode(handle,arr_bytes)


base64_decode
>>>>>>>>>>>>>>>>>>>>>>>>

Decode the byte-encoded data in base64 and return the decoded byte data. BM1688 and CV186AH PCIE mode are not supported.

**Interface:**
    .. code-block:: python

        def base64_decode(handle: Handle, encode_bytes: bytes) -> bytes:

**Parameters:**

* handle: Handle

The handle of the device, created using sail.Handle(dev_id).

* encode_bytes: bytes

The byte-encoded data in base64.

**Returns:** 

* bytes 
 
The byte data decoded from base64. 

**Sample**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np

        if __name__ == "__main__":
            arr = np.array([[1, 2, 3], [4, 5, 6]])
            shape = arr.shape
            # tobytes
            arr_bytes = arr.tobytes()
            # get handle
            handle = sail.Handle(0)
            # base64 encode
            base64_encoded_arr = sail.base64_encode(handle,arr_bytes)

            # decode
            base64_decode_arr = sail.base64_decode(handle,base64_encoded_arr)
            # byte to numpy
            res_arr = np.frombuffer(arr_bytes, dtype=np.int64).reshape(shape)


base64_encode_array
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Encode a numpy.array into base64, generating byte-encoded data. BM1688 and CV186AH PCIE mode are not supported.

**Interface:**
    .. code-block:: python

        def base64_encode_array(handle: Handle, input_arr: numpy.ndarray) -> bytes:

**Parameters:**

* handle: Handle
 
The handle of the device, created using sail.Handle(dev_id).

* input_arr: numpy.ndarray

The numpy.ndarray data to be encoded.

**Returns**

* bytes

The byte data encoded in base64.


base64_decode_asarray
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Decode base64 to generate numpy.array data. 

**Interface:**
    .. code-block:: python

        def base64_decode_asarray(handle: Handle, encode_arr_bytes: bytes, array_type:str = "uint8") -> numpy.ndarray:

**Parameters:**
               
* handle: Handle

The handle of the device, created using sail.Handle(dev_id).

* encode_arr_bytes: bytes

The byte data of the numpy.ndarray encoded in base64.
    
* array_type: str

The data type of numpy.ndarray, defaulting to uint8, supports float, uint8, int8, int16, int32, int64.
    
**Returns** 

* numpy.array 

The one-dimensional numpy.array array decoded from base64. 

**Sample**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np

        if __name__ == "__main__":
            arr = np.array([[1,2,3],[4,5,6]],dtype=np.uint8)
            # base64 encode
            base64_encoded = sail.base64_encode_array(handle,arr)
            # base64 decode
            res_array = sail.base64_decode_asarray(handle,base64_encoded).reshape(shape)


get_tpu_util
>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the processor utilization of the specified device

**Interface:**
    .. code-block:: python

        def get_tpu_util(dev_id: int) -> int

**Parameter:**

* dev_id: int

Device ID.

**Return:**

Returns the processor percent utilization of the device corresponding to the ID.

**Sample**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            print("dev {} tpu-util is {} %".format(dev_id,sail.get_tpu_util(dev_id)))


get_vpu_util
>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the VPU utilization of the specified device

**Interface:**
    .. code-block:: python

        def get_vpu_util(dev_id: int) -> list

**Parameter:**

* dev_id: int

Device ID.

**Return:**

The vpu of bm1684 is 5-core, and the return value is a list of length 5. The vpu of bm1684x is 3-core, and the return value is a list of length 3.
Each integer in the List is the percent utilization of the corresponding core.

**Sample**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            print("get_vpu_util",sail.get_vpu_util(dev_id))


get_vpp_util
>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the VPP utilization of the specified device

**Interface:**
    .. code-block:: python
        
        def get_vpp_util(dev_id: int) -> list

**Parameter:**

* dev_id: int

Device ID.

**Return:**

The vpp of bm1684 and bm1684x are both 2-core, and the return value is a list of length 2.
Each integer in the List is the percent utilization of the corresponding core.

**Sample**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            print("get_vpp_util",sail.get_vpp_util(dev_id))


get_board_temp
>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the temperature of the board. Only supported in PCIe mode.

**Interface:**
    .. code-block:: python
        
        def get_board_temp(dev_id: int) -> int

**Parameter:**

* dev_id: int

Device ID.

**Return:**

The board temperature for the corresponding card, with the default unit in Celsius (°C)

**Sample**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            print("get_board_temp",sail.get_board_temp(dev_id))

get_chip_temp
>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the temperature of the processor. Only supported in PCIe mode.

**Interface:**
    .. code-block:: python
        
        def get_chip_temp(dev_id: int) -> int

**Parameter:**

* dev_id: int

Device ID.

**Return:**

The processor temperature for the corresponding card, with the default unit in Celsius (°C)

**Sample**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            print("get_chip_temp",sail.get_chip_temp(dev_id))

get_dev_stat
>>>>>>>>>>>>>>>>>>>>>>>>>>

Get device memory information.

**Interface:**
    .. code-block:: python
        
        def get_dev_stat(dev_id: int) -> list

**Parameter:**

* dev_id: int

Device ID.

**Return:**

A list of memory information for the corresponding device: [mem_total, mem_used, tpu_util].

**Sample**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            dev_id = 0
            print("get_dev_stat",sail.get_dev_stat(dev_id))
