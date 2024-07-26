IOMode
___________


IOMode is used to define the memory location information (device memory or system memory) of the input Tensor and output Tensor.


**Interface:**
    .. code-block:: c

        enum IOMode {
            SYSI,
            SYSO,
            SYSIO,
            DEVIO
        };

**Parameters:**

* SYSI

The input Tensor is in system memory, and the output Tensor is in device memory.

* SYSO

The input Tensor is in device memory, and the output Tensor is in system memory.

* SYSIO

The input Tensor is in system memory, and the output Tensor is in system memory.

* DEVIO

The input Tensor is in device memory, and the output Tensor is in device memory.
