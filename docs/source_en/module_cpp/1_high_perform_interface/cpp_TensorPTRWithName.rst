TensorPTRWithName
_________________________

Tensor with name

.. code-block:: c
    
    struct TensorPTRWithName
    {
        TensorPTRWithName(): name(""),data(NULL) { } 
        std::string name;
        sail::Tensor* data;
    };

**Parameters**

* name: string

The name of the tensor.

* data: Tensor*

The data of the tensor.

