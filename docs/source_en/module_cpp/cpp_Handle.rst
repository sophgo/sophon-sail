Handle
_____________


Handle is a wrapper class for device handles and is used to identify devices in programs.


Constructor
>>>>>>>>>>>>>>>>>>>>>>>>>>>

Initialize Handle

**Interface:**
    .. code-block:: c

        Handle(int tpu_id);

**Parameter:**

* tpu_id: int

Create the ID number of the Tensor Computing Processor used by Handle


get_device_id
>>>>>>>>>>>>>>>

Get the id of Tensor Computing Processor in Handle

**Interface:**
    .. code-block:: c

        int get_device_id();

**Parameter:**

* tpu_id: int

The ID of the Tensor Computing Processor in the Handle.


get_sn
>>>>>>>>>>>>>>>

Get the SN(serial number) identifying the device in Handle

**Interface:**
    .. code-block:: c

        std::string get_sn();

**Returns:**

* serial_number: string

Returns the serial number of the device in Handle

get_target
>>>>>>>>>>>>>>>

Get the Tensor Computing Processor model of the device

**Interface:**
    .. code-block:: c

        std::string get_target();

**Returns:**

* Tensor Computing Processor type: str

Returns the model of the device Tensor Computing Processor


get_status
>>>>>>>>>>>>>>>

Get the status of the Tensor Computing Processor.

**Interface:**
    .. code-block:: c

        bool get_status();

**Returns:**

* Status: bool

Returns the status of the Tensor Computing Processor.

true means Active, false means Fault.

**Example Code:**
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
            std::cout << "Status: " << handle.get_status() << std::endl;  

            return 0;  
        }