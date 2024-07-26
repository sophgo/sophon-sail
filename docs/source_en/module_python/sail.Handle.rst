sail.Handle
_____________

__init__
>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: python

        def __init__(self, tpu_id: int)
 

**Parameters**

* tpu_id : int

create handle with tpu Id



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

