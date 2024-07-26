Handle
_____________


Handle是设备句柄的包装类，在程序中用于设备的标识。


构造函数
>>>>>>>>>>>>>>>>>>>>>>>>>>>

初始化Handle

**接口形式:**
    .. code-block:: c

        Handle(int tpu_id);

**参数说明:**

* tpu_id: int

创建Handle使用的智能视觉深度学习处理器的id号


get_device_id
>>>>>>>>>>>>>>>

获取Handle中智能视觉深度学习处理器的id

**接口形式:**
    .. code-block:: c

        int get_device_id();

**返回值说明:**

* tpu_id: int

Handle中的智能视觉深度学习处理器的id号


get_sn
>>>>>>>>>>>>>>>

获取Handle中标识设备的序列码

**接口形式:**
    .. code-block:: c

        std::string get_sn();

**返回值说明:**

* serial_number: string

返回Handle中设备的序列码

get_target
>>>>>>>>>>>>>>>

获取设备的智能视觉深度学习处理器型号

**接口形式:**
    .. code-block:: c

        std::string get_target();

**返回值说明:**

* Tensor Computing Processor type: str

返回设备智能视觉深度学习处理器的型号

**示例代码:**
    .. code-block:: c

        #include <stdio.h>
        #include <sail/cvwrapper.h>
        #include <iostream>

        using namespace std;
        
        int main() {  
            int tpu_id = 0;  
            sail::Handle handle(tpu_id);  
        
            std::cout << "Device ID: " << handle.get_device_id() << std::endl;  
            std::cout << "SN: " << handle.get_sn() << std::endl;  
            std::cout << "Target: " << handle.get_target() << std::endl;  
        
            return 0;  
        }