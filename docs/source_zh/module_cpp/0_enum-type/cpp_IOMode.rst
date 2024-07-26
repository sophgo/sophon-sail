IOMode
___________


IOMode用于定义输入Tensor和输出Tensor的内存位置信息（device memory或system memory）。


**接口形式:**
    .. code-block:: c

        enum IOMode {
            SYSI,
            SYSO,
            SYSIO,
            DEVIO
        };

**参数说明:**

* SYSI

输入Tensor在system memory，输出Tensor在device memory

* SYSO

输入Tensor在device memory，输出Tensor在system memory

* SYSIO

输入Tensor在system memory，输出Tensor在system memory

* DEVIO

输入Tensor在device memory，输出Tensor在device memory
