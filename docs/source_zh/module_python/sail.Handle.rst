sail.Handle
_____________


Handle是设备句柄的包装类，在程序中用于设备的标识。


\_\_init\_\_
>>>>>>>>>>>>>>>

初始化Handle

**接口形式:**
    .. code-block:: python

        def __init__(self, tpu_id: int)

**参数说明:**

* tpu_id: int

创建Handle使用的智能视觉深度学习处理器的id号


get_device_id
>>>>>>>>>>>>>>>

获取Handle中智能视觉深度学习处理器的id

**接口形式:**
    .. code-block:: python

        def get_device_id(self) -> int

**返回值说明:**

* tpu_id: int

Handle中的智能视觉深度学习处理器的id号


get_sn
>>>>>>>>>>>>>>>

获取Handle中标识设备的序列码

**接口形式:**
    .. code-block:: python

        def get_sn(self) -> str

**返回值说明:**

* serial_number: str

返回Handle中设备的序列码

get_target
>>>>>>>>>>>>>>>

获取设备的智能视觉深度学习处理器型号

**接口形式:**
    .. code-block:: python

        def get_target(self) -> str

**返回值说明:**

* Tensor Computing Processor type: str

返回设备智能视觉深度学习处理器的型号

**示例代码:**
    .. code-block:: python
        
        import sophon.sail as sail
        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            print(handle.get_device_id(), handle.get_sn(), handle.get_target())