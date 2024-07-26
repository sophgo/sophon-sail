TensorPTRWithName
_________________________

带有名称的Tensor

.. code-block:: c
    
    struct TensorPTRWithName
    {
        TensorPTRWithName(): name(""),data(NULL) { } 
        std::string name;
        sail::Tensor* data;
    };

**参数说明**

* name: string

tensor的名字

* data: Tensor*

tensor的数据

