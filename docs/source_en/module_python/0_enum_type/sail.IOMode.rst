sail.IOMode
___________


**接口形式:**
    .. code-block:: python

        # Input tensors are in system memory while output tensors are in device memory
        sail.IOMode.SYSI
        # Input tensors are in device memory while output tensors are in system memory.
        sail.IOMode.SYSO
        # Both input and output tensors are in system memory.
        sail.IOMode.SYSIO
        # Both input and output tensors are in device memory.
        sail.IOMode.DEVIO