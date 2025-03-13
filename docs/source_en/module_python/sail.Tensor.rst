sail.Tensor
______________

\_\_init\_\_
>>>>>>>>>>>>>>>>>>>>>

Constructor allocates memory of the tensor. If synchronization between system memory and device memory is required, sync_d2s or sync_s2d needs to be executed.

**Interface:**
    .. code-block:: python

        def __init__(self, 
                    handle: sail.Handle, 
                    data: numpy.array, 
                    own_sys_data: bool=True)
 

**Parameters**
            
* handle : sail.Handle

Handle instance

* array_data : numpy.array

Tensor ndarray data, dtype can be np.float32, np.int8 or np.uint8

* own_sys_data : bool, default: True

Indicator of whether own system memory, If false, the memory will be copied to device directly  


**Interface:**
    .. code-block:: python

        def __init__(self, 
                    handle: sail.Handle, 
                    shape: list[int], 
                    dtype: sail.Dtype, 
                    own_sys_data: bool, 
                    own_dev_data: bool)
 

**Parameters**

* handle : sail.Handle

Handle instance

* shape : tuple

Tensor shape

* dytpe : sail.Dtype

Data type

* own_sys_data : bool

Indicator of whether own system memory

* own_dev_data : bool

Indicator of whether own device memory

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input = np.array([1, 2, 3]) 
            
            input_tensor1 = sail.Tensor(handle,input)
            input_tensor2 = sail.Tensor(handle,[1,2],sail.Dtype.BM_FLOAT32,true,true)

**Interface:**

This initialization method creates a new Tensor based on an existing source Tensor and reuses a portion of the source Tensor's device memory without copying device memory. 
It is suitable for scenarios such as LLM inference where memory reuse is required.

During the use of this Tensor, it is necessary to ensure that the source Tensor is not released.

    .. code-block:: python

        def __init__(self, src: Tensor, shape: list[int], offset: int)

**Parameters:**

* src: sail.Tensor

The source Tensor used to create the new Tensor.

* shape: list[int]

The shape of the new Tensor, a sequence of integers. 

The number of elements corresponding to the new shape must not exceed the number of elements in the source Tensor.

* offset: int

The offset of the Tensor's device memory relative to the source Tensor's device memory, in bytes of the dtype.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            height = 1080
            width = 1920
            data_type = sail.Dtype.BM_INT32
            src_shape = [1, 3, height, width]
            src_tensor = sail.Tensor(handle, src_shape, data_type, False, True)

            dst_shape = [1, 1, height, width]
            offset = height * width
            dst_tensor = sail.Tensor(src_tensor, dst_shape, offset)

shape
>>>>>>>>>>>>>>>>>>>>>

Get shape of the tensor.

**Interface:**
    .. code-block:: python

        def shape(self)-> list
 

**Returns**

* tensor_shape : list

Shape of the tensor

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input = np.array([1, 2, 3]) 
            
            input_tensor1 = sail.Tensor(handle,input)
            print(input_tensor1.shape())

dtype
>>>>>>>>>>>>>>>>>>>>>

Get data_type of the tensor.

**Interface:**
    .. code-block:: python

        def dtype(self)-> sail.Dtype
 

**Returns**

* data_type : sail.Dtype

return data_type of the tensor.


**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input = np.array([1, 2, 3]) 
            
            input_tensor1 = sail.Tensor(handle,input)
            print(input_tensor1.dtype())

asnumpy
>>>>>>>>>>>>>>>>>>>>>

Get system data of the tensor. If synchronization between system memory and device memory is required, sync_d2s needs to be executed.

**Interface:**
    .. code-block:: python

        def asnumpy(self)-> numpy.array
 

**Returns**

* data : numpy.array

System data of the tensor, dtype can be np.float32, np.int8
or np.uint8 with respective to the dtype of the tensor.


**Interface:**
    .. code-block:: python

        def asnumpy(self, shape: tuple)-> numpy.array


**Parameters**

* shape : tuple

Tensor shape want to get

**Returns**

* data : numpy.array

System data of the tensor, dtype can be np.float32, np.int8
or np.uint8 with respective to the dtype of the tensor.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input = np.array([1, 2, 3]) 
            
            input_tensor1 = sail.Tensor(handle,input)
            input_ = input_tensor1.asnumpy()
            input__ = input_tensor1.asnumpy((3,1))


update_data
>>>>>>>>>>>>>>>>>>>>>
Update system data of the tensor, if there is no system memory assigned, update the device memory.
    
**Interface:**
    .. code-block:: python

        def update_data(self, data: numpy.array)
 

**Parameters**

data : numpy.array

Data to update. The data type of the updated data should be the same as the tensor, The data size should not exceed \
the tensor size, and the tensor shape will not be changed.

Note: If the data is of the numpy.float16 type, you should use numpy.view(numpy.uint16) and pass it to this API.

**example:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np

        if __name__ == '__main__':
            dev_id = 0
            handle = sail.Handle(dev_id)
            
            tensor_fp32 = sail.Tensor(handle, [1,3,640,640], sail.BM_FLOAT32, True, True)
            np_fp32 = np.ones(tensor_fp32.shape(),dtype=np.float32)
            tensor_fp32.update_data(np_fp32)

            tensor_fp16 = sail.Tensor(handle, [1,3,640,640], sail.BM_FLOAT16, True, True)
            np_fp16 = np.ones(tensor_fp16.shape(),dtype=np.float16)
            tensor_fp16.update_data(np_fp16.view(np.uint16))

scale_from
>>>>>>>>>>>>>>>>>>>>>

Scale data to tensor in system memory.
    
**Interface:**
    .. code-block:: python

        def scale_from(self, data: numpy.array, scale: float32)
 

**Parameters**

* data : numpy.array with dtype of float32

Data.

* scale : float32

Scale value.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input = np.array([1, 2, 3]) 
            
            input_tensor1 = sail.Tensor(handle,input)
            input_tensor1.scale_from(input,0.1)

scale_to
>>>>>>>>>>>>>>>>>>>>>

Scale tensor to data in system memory.
    
**Interface:**
    .. code-block:: python

        def scale_to(self, scale: float32)-> numpy.array
 

**Parameters**

* scale : float32

Scale value.

**Returns**

* data : numpy.array with dtype of float32

Data.


**Interface:**
    .. code-block:: python

        def scale_to(self, scale: float32, shape: tuple)-> numpy.array


**Parameters**

* scale : float32

Scale value.

* shape : tuple

Tensor shape wanted to get

**Returns**

* data : numpy.array with dtype of float32

Data.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input = np.array([1, 2, 3])

            input_tensor1 = sail.Tensor(handle,input)
            input_tensor1_ = input_tensor1.scale_to(0.1)
            input_tensor1__ = input_tensor1.scale_to(0.1,(3,1))

reshape
>>>>>>>>>>>>>>>>>>>>>

Reset shape of the tensor.
    
**Interface:**
    .. code-block:: python

        def reshape(self, shape: list)
 

**Parameters**

* shape : list

New shape of the tensor

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input = np.array([1, 2, 3]) 
            
            input_tensor1 = sail.Tensor(handle,input)
            input_tensor1_ = input_tensor1.reshape([3,1])

own_sys_dat
>>>>>>>>>>>>>>>>>>>>>

Judge if the tensor owns data pointer in system memory.

**Interface:**
    .. code-block:: python

        def own_sys_data(self)-> bool
 

**Returns**

* judge_ret : bool

True for owns data pointer in system memory.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input = np.array([1, 2, 3]) 
            
            input_tensor1 = sail.Tensor(handle,input)
            print(input_tensor1.own_sys_data())

own_dev_data
>>>>>>>>>>>>>>>>>>>>>

Judge if the tensor owns data in device memory.

**Interface:**
    .. code-block:: python

        def own_dev_data(self)-> bool
 

**Returns**

* judge_ret : bool

True for owns data in device memory.


**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input = np.array([1, 2, 3]) 
            
            input_tensor1 = sail.Tensor(handle,input)
            print(input_tensor1.own_dev_data())

sync_s2d
>>>>>>>>>>>>>>>>>>>>>

Copy data from system memory to device memory without or with specified size.

**Interface:**
    .. code-block:: python

        def sync_s2d(self)
 


**Interface:**
    .. code-block:: python

        def sync_s2d(self, size: int)


**Parameters**

* size : int

Byte size to be copied


**Interface:**
    .. code-block:: python

        def sync_s2d(self, src: sail.Tensor, offset_src: int, offset_dst: int, len: int)->None


**Parameters**

* src: sail.Tensor

Specifies the Tensor to be copied from.

* offset_src: int

Specifies the number of elements to offset in the source Tensor from where to start copying.

* offset_dst: int

Specifies the number of elements to offset in the destination Tensor from where to start copying.

* len: int

Specifies the length of the copy, i.e., the number of elements to copy.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input = np.array([1, 2, 3]) 
            
            input_tensor1 = sail.Tensor(handle,input)
            input_tensor2 = sail.Tensor(handle,[1,2], sail.Dtype.BM_FLOAT32, True, True)
            input_tensor2.sync_s2d()
            input_tensor2.sync_s2d(1)
            input_tensor2.sync_s2d(input_tensor1,0,0,2)

sync_d2s
>>>>>>>>>>>>>>>>>>>>>

Copy data from device memory to system memory without or with specified size.

**Interface:**
    .. code-block:: python

        def sync_d2s(self)



**Interface:**
    .. code-block:: python

        def sync_d2s(self, size: int)
 

**Parameters**

* size : int

Byte size to be copied

**Interface:**
    .. code-block:: python

        def sync_d2s(self, src: sail.Tensor, offset_src: int, offset_dst: int, len: int)->None


**Parameters**

* src: sail.Tensor

Specifies the Tensor to be copied from.

* offset_src: int

Specifies the number of elements to offset in the source Tensor from where to start copying.

* offset_dst: int

Specifies the number of elements to offset in the destination Tensor from where to start copying.

* len: int

Specifies the length of the copy, i.e., the number of elements to copy.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            
            input_tensor1 = sail.Tensor(handle,[1,3],sail.Dtype.BM_FLOAT32,False,True)
            input_tensor2 = sail.Tensor(handle,[1,3],sail.Dtype.BM_FLOAT32,True,True)

            input_tensor1.ones()
            input_tensor2.sync_d2s()
            input_tensor2.sync_d2s(2)
            input_tensor2.sync_d2s(input_tensor1,0,0,2)

sync_d2d
>>>>>>>>>>>>>>>>>>>>>

Copies the data from another Tensor's device memory to this Tensor's device memory.

**Interface:**
    .. code-block:: python

        def sync_d2d(self, src: sail.Tensor, offset_src: int, offset_dst: int, len: int)->None

 

**Parameters**

* src: sail.Tensor

Specifies the Tensor to be copied from.

* offset_src: int

Specifies the number of elements to offset in the source Tensor from where to start copying.

* offset_dst: int

Specifies the number of elements to offset in the destination Tensor from where to start copying.

* len: int

Specifies the length of the copy, i.e., the number of elements to copy.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            handle_ = sail.Handle(1)
            input_tensor1 = sail.Tensor(handle,[1,3],sail.Dtype.BM_FLOAT32,False,True)
            input_tensor2 = sail.Tensor(handle_,[1,3],sail.Dtype.BM_FLOAT32,True,True)

            input_tensor1.ones()
            input_tensor2.sync_d2d(input_tensor1,0,0,2)

sync_d2d_stride
>>>>>>>>>>>>>>>>>>>>>

Copies the data from another Tensor's device memory to this Tensor's device memory in stride.

**Interface:**
    .. code-block:: python

        def sync_d2d_stride(self, src: sail.Tensor, stride_src: int, count: int)->None


**Parameters:**

* src: sail.Tensor

Specifies the Tensor to be copied from.

* stride_src: int

Specifies the stride of the source Tensor.

* stride_dst: int

Specifies the stride of the destination Tensor.stride_dst must be 1, EXCEPT: stride_dst == 4 && stride_src == 1 && Tensor_type_size == 1

* count: int

Specifies the count of elements to copy.Ensure count * stride_src <= tensor_src_size, count * stride_dst <= tensor_dst_size.


dump_data
>>>>>>>>>>>>>>>>>>>>>

Dump Tensor data to file. If synchronization between system memory and device memory is required, sync_d2s needs to be executed.

**Interface:**
    .. code-block:: python

        def sync_d2s(self, file_name:str, bin:bool = False)
 

**Parameters**

* file_name : str

file path to dump tensor

* bin : bool

binary format, default False.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np

        if __name__ == '__main__':
            dev_id = 0
            handle = sail.Handle(dev_id)
            data = np.ones([1,20], dtype=int)
            ts = sail.Tensor(handle, data)
            ts.scale_from(data, 0.1)
            ts.dump_data("./temp.txt")
            ret_data = np.loadtxt("./temp.txt")
            print(ts.asnumpy(), ret_data)


memory_set
>>>>>>>>>>>>>>>>>>>>>

Fill memory with a scalar, it will be automatically converted to tensor's dtype.

**Interface:**
    .. code-block:: python

        def memory_set(self, c: any)->None


**Parameters:**

* c: any

the value to fill.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input = 1
            input_tensor1 = sail.Tensor(handle,[1],sail.Dtype.BM_FLOAT32,True,True)

            input_tensor1.memory_set(input)

zeros
>>>>>>>>>>>>>>>>>>>>>

fill memory with zeros.

**Interface:**
    .. code-block:: python

        def zeros(self)->None

    
**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input_tensor1 = sail.Tensor(handle,(1,3),sail.Dtype.BM_FLOAT32,False,True)

            input_tensor1.zeros()

ones
>>>>>>>>>>>>>>>>>>>>>

fill memory with ones.

**Interface:**
    .. code-block:: python

        def ones(self)->None

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input_tensor1 = sail.Tensor(handle,(1,3),sail.Dtype.BM_FLOAT32,False,True)

            input_tensor1.ones()

size
>>>>>>>>>>>>>>>>>>>>>

Return the number of elements contained in the Tensor.

**Interface:**
    .. code-block:: python

        def size(self)->int

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input_tensor1 = sail.Tensor(handle,(1,3),sail.Dtype.BM_FLOAT32,False,True)

            print(input_tensor1.size())


element_size
>>>>>>>>>>>>>>>>>>>>>

Returns the size in bytes of an individual element.

**Interface:**
    .. code-block:: python

        def element_size(self)->int

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input_tensor1 = sail.Tensor(handle,(1,3),sail.Dtype.BM_FLOAT32,False,True)

            print(input_tensor1.element_size())

nbytes
>>>>>>>>>>>>>>>>>>>>>

Return the total number of bytes occupied by all elements of Tensor.

**Interface:**
    .. code-block:: python

        def nbytes(self)->int

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input_tensor1 = sail.Tensor(handle,(1,3),sail.Dtype.BM_FLOAT32,False,True)

            print(input_tensor1.nbytes())