sail.Handle
_____________

__init__
>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: python

        def __init__(self, tpu_id: int)
 

**Parameters**

* tpu_id : int

create handle with Tensor Computing Processor Id



get_device_id
>>>>>>>>>>>>>>>

Get Tensor Computing Processor id of this handle. 

**Interface:**
    .. code-block:: python

        def get_device_id(self)-> int
 
            
**Returns**

* tpu_id : int

Tensor Computing Processor id of this handle.



get_sn
>>>>>>>>>>>>>>>

Get serial number of this handle.

**Interface:**
    .. code-block:: python

        def get_sn(self)-> str
 

**Returns**

* serial_number : str

serial number of this handle.



get_target
>>>>>>>>>>>>>>>

Get Tensor Computing Processor type of this handle.

**Interface:**
    .. code-block:: python

        def get_target(self)-> str
 

**Returns**

* tpu_chip_type : str

Tensor Computing Processor type of this handle.


get_status
>>>>>>>>>>>>>>>

Get status of Tensor Computing Processor.

**Interface:**
    .. code-block:: python

        def get_status(self)-> bool
 

**Returns**

* status : bool

Returns the status of the Tensor Computing Processor.

True means Active, False means Fault.

**Example Code:**
    .. code-block:: python
        
        import sophon.sail as sail
        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            print(f"Device ID: {handle.get_device_id()}")
            print(f"SN: {handle.get_sn()}")
            print(f"Target: {handle.get_target()}")
            print(f"Status: {handle.get_status()}")