sail.IOMode
___________


IOMode用于定义输入Tensor和输出Tensor的内存位置信息（device memory或system memory）。


**接口形式:**
    .. code-block:: python

        sail.IOMode.SYSI
        sail.IOMode.SYSO
        sail.IOMode.SYSIO
        sail.IOMode.DEVIO

**参数说明:**

* sail.IOMode.SYSI

输入Tensor在system memory，输出Tensor在device memory

* sail.IOMode.SYSO

输入Tensor在device memory，输出Tensor在system memory

* sail.IOMode.SYSIO

输入Tensor在system memory，输出Tensor在system memory

* sail.IOMode.DEVIO

输入Tensor在device memory，输出Tensor在device memory
