sail.Tensor
______________


Tensor是模型推理的输入输出类型，包含了数据信息，实现内存管理。


\_\_init\_\_
>>>>>>>>>>>>>>>>>>>>>

初始化Tensor,并为Tensor分配内存,如果需要进行系统内存和设备内存的同步,需要执行sync_d2s或sync_s2d

**接口形式1:**
    .. code-block:: python

        def __init__(self, handle: Handle, data: np.array, own_sys_data=True)

**参数说明1:**

* handle: sail.Handle

设备标识Handle

* array_data: numpy.array

利用numpy.array类型初始化Tensor，其数据类型可以是np.float32,np.int8,np.uint8

* own_sys_data: bool

指示该Tensor是否拥有system memory，如果为False，则直接将数据复制到device memory

**接口形式2**
    .. code-block:: python

        def __init__(self, handle: Handle, shape: list[int], dtype: Dtype, own_sys_data: bool, own_dev_data: bool)

**参数说明2:**

* handle: sail.Handle

设备标识Handle

* shape: tuple

设置Tensor的shape

* dtype: sail.Dtype

Tensor的数据类型

* own_sys_data: bool

指示Tensor是否拥有system memory

* own_dev_data: bool

指示Tensor是否拥有device memory

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input = np.array([1, 2, 3]) 
            
            input_tensor1 = sail.Tensor(handle,input)
            input_tensor2 = sail.Tensor(handle,[1,2],sail.Dtype.BM_FLOAT32,true,true)

            
shape
>>>>>>>>>>>>>>>>>>>>>

获取Tensor的shape

**接口形式:**
    .. code-block:: python

        def shape(self) -> list :

**返回值说明:**

* tensor_shape : list

返回Tensor的shape的列表。

**示例代码:**
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

获取Tensor的数据类型

**接口形式:**
    .. code-block:: python

        def dtype(self) -> sail.Dtype :

**返回值说明:**

* data_type : sail.Dtype

返回Tensor的数据类型。


**示例代码:**
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

获取Tensor中系统内存的数据，返回numpy.array类型。如果需要进行系统内存和设备内存的同步，需要执行sync_d2s。

**接口形式:**
    .. code-block:: python

        def asnumpy(self) -> numpy.array 

        def asnumpy(self, shape: tuple) -> numpy.array

**参数说明:**

* shape: tuple

可对Tensor中的数据reshape，返回形状为shape的numpy.array

**返回值说明**

返回Tensor中系统内存的数据，返回类型为numpy.array。


**示例代码:**
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

更新Tensor中系统内存的数据，如果没有分配系统内存，则更新设备内存中的数据。
    
**接口形式:**
    .. code-block:: python

        def update_data(self, data: numpy.array) -> None

**参数说明:**

* data: numpy.array

更新的数据，数据类型应和Tensor一致，数据size不能超过Tensor的size，Tensor的shape将保持不变。

注：如果是numpy.float16类型的数据，应使用numpy.view(numpy.uint16)再传递给本接口。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np

        if __name__ == '__main__':
            dev_id = 0
            handle = sail.Handle(dev_id)
            
            tensor_fp32 = sail.Tensor(handle, [1,3,640,640], sail.BM_FLOAT32)
            np_fp32 = np.ones(tensor_fp32.shape(),dtype=np.float32)
            tensor_fp32.update_data(np_fp32)

            tensor_fp16 = sail.Tensor(handle, [1,3,640,640], sail.BM_FLOAT16)
            np_fp16 = np.ones(tensor_fp16.shape(),dtype=np.float16)
            tensor_fp16.update_data(np_fp16.view(np.uint16))

scale_from
>>>>>>>>>>>>>>>>>>>>>

先对data按比例缩放，再将数据更新到Tensor的系统内存。
    
**接口形式:**
    .. code-block:: python

        def scale_from(self, data: numpy.array, scale: float32)->None

**参数说明:**

* data: numpy.array

对data进行scale，再将数据更新到Tensor的系统内存。

* scale: float32

等比例缩放时的尺度。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input = np.array([1, 2, 3]) 
            
            input_tensor1 = sail.Tensor(handle,input)
            input_tensor1_ = input_tensor1.scale_from(input,0.1)


scale_to
>>>>>>>>>>>>>>>>>>>>>

先对Tensor进行等比例缩放，再将数据返回到系统内存。
    
**接口形式:**
    .. code-block:: python

        def scale_to(self, scale: float32)->numpy.array

        def scale_to(self, scale: float32, shape: tuple)->numpy.array

**参数说明:**

* scale: float32

等比例缩放时的尺度。

* shape: tuple

数据返回前可进行reshape，返回shape形状的数据。

**返回值说明:**

* data: numpy.array

将处理后的数据返回至系统内存，返回numpy.array

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input = np.array([1, 2, 3]) 
            
            input_tensor1 = sail.Tensor(handle,input)
            input_tensor1_ = input_tensor1.scale_to(input,0.1)
            input_tensor1__ = input_tensor1.scale_to(input,0.1,(3,1))

reshape
>>>>>>>>>>>>>>>>>>>>>

对Tensor进行reshape
    
**接口形式:**
    .. code-block:: python

        def reshape(self, shape: list)->None

**参数说明:**

* shape: list

设置期望得到的新shape。


**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input = np.array([1, 2, 3]) 
            
            input_tensor1 = sail.Tensor(handle,input)
            input_tensor1_ = input_tensor1.reshape([3,1])


own_sys_data
>>>>>>>>>>>>>>>>>>>>>

查询该Tensor是否拥有系统内存的数据指针。

**接口形式:**
    .. code-block:: python

        def own_sys_data(self)->bool

**返回值说明:**

* judge_ret: bool

如果拥有系统内存的数据指针则返回True，否则False。

**示例代码:**
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

查询该Tensor是否拥有设备内存的数据

**接口形式:**
    .. code-block:: python

        def own_dev_data(self)->bool

**返回值说明:**

* judge_ret : bool

如果拥有设备内存中的数据则返回True，否则False。

**示例代码:**
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

将Tensor中的数据从系统内存拷贝到设备内存。

**接口形式:**
    .. code-block:: python

        def sync_s2d(self)->None

        def sync_s2d(self, size)->None

**参数说明:**

* size: int

将特定size字节的数据从系统内存拷贝到设备内存。


**接口形式:**
    .. code-block:: python

        def sync_s2d(self, src: sail.Tensor, offset_src: int, offset_dst: int, len: int)->None


**参数说明:**

* src: sail.Tensor

指定被拷贝的Tensor。

* offset_src: int

指定被拷贝Tensor上的数据偏移几个元素后开始拷贝。

* offset_dst: int

指定拷贝目标Tensor上的数据偏移几个元素后开始拷贝。

* len: int

指定拷贝长度，既拷贝的元素个数。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input = np.array([1, 2, 3]) 
            
            input_tensor1 = sail.Tensor(handle,input)
            input_tensor2 = sail.Tensor(handle,[1,2],sail.Dtype.BM_FLOAT32,true,true)
            input_tensor2.sync_s2d()
            input_tensor2.sync_s2d(1)
            input_tensor2.sync_s2d(input_tensor1,0,0,2)

sync_d2s
>>>>>>>>>>>>>>>>>>>>>

将Tensor中的数据从设备内存拷贝到系统内存。

**接口形式:**
    .. code-block:: python

        def sync_d2s(self)->None
          
        def sync_d2s(self, size: int)->None

**参数说明:**

* size: int

将特定size字节的数据从设备内存拷贝到系统内存。

**接口形式:**
    .. code-block:: python

        def sync_d2s(self, src: sail.Tensor, offset_src: int, offset_dst: int, len: int)->None


**参数说明:**

* src: sail.Tensor

指定被拷贝的Tensor。

* offset_src: int

指定被拷贝Tensor上的数据偏移几个元素后开始拷贝。

* offset_dst: int

指定拷贝目标Tensor上的数据偏移几个元素后开始拷贝。

* len: int

指定拷贝长度，既拷贝的元素个数。

**示例代码:**
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

将另外一个Tensor设备内存上的数据拷贝到本Tensor的设备内存中。

**接口形式:**
    .. code-block:: python

        def sync_d2d(self, src: sail.Tensor, offset_src: int, offset_dst: int, len: int)->None


**参数说明:**

* src: sail.Tensor

指定被拷贝的Tensor。

* offset_src: int

指定被拷贝Tensor上的数据偏移几个元素后开始拷贝。

* offset_dst: int

指定拷贝目标Tensor上的数据偏移几个元素后开始拷贝。

* len: int

指定拷贝长度，既拷贝的元素个数。

**示例代码:**
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

以stride的方式将另外一个Tensor设备内存上的数据拷贝到本Tensor的设备内存中。

**接口形式:**
    .. code-block:: python

        def sync_d2d_stride(self, src: sail.Tensor, stride_src: int, stride_dst: int, count: int)->None


**参数说明:**

* src: sail.Tensor

指定被拷贝的Tensor。

* stride_src: int

指定被拷贝Tensor上数据的stride。

* stride_dst: int

指定拷贝目标Tensor上数据的stride。stride_dst必须是1，除了stride_dst为4且stride_src为1且tensor数据类型大小为1字节的情况。

* count: int

指定拷贝长度，既拷贝的元素个数。需要保证count * stride_src <= tensor_src_size, count * stride_dst <= tensor_dst_size。

dump_data
>>>>>>>>>>>>>>>>>>>>>

将Tensor中的数据写入到指定文件中,如果需要进行系统内存和设备内存的同步,需要执行sync_d2s

**接口形式:**
    .. code-block:: python

        def dump_data(file_name: str, bin: bool = False)

**参数说明:**

* file_name: str

写入文件的路径

* bin: bool

是否采用二进制的形式存储Tensor,默认false.

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np

        if __name__ == '__main__':
            dev_id = 0
            handle = sail.Handle(dev_id)
            data = np.ones([1,20],dtype=int)
            ts = sail.Tensor(handle,data)
            ts.scale_from(data,[0.01,0.1])
            ts.dump_data("./temp.txt")
            ret_data = np.loadtxt("./temp.txt")
            print(ts.asnumpy(),ret_data)

memory_set
>>>>>>>>>>>>>>>>>>>>>

将本Tensor的数据全部置为c，在接口内部根据本Tensor的dtype对c做相应的类型转换。

**接口形式:**
    .. code-block:: python

        def memory_set(self, c: any)->None


**参数说明:**

* c: any

需要填充的值。

**示例代码:**
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

将本Tensor的数据全部置为0。

**接口形式:**
    .. code-block:: python

        def zeros(self)->None
    
**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input_tensor1 = sail.Tensor(handle,(1,3),sail.Dtype.BM_FLOAT32,False,True)

            input_tensor1.zeros()


ones
>>>>>>>>>>>>>>>>>>>>>

将本Tensor的数据全部置为1。

**接口形式:**
    .. code-block:: python

        def ones(self)->None

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            handle = sail.Handle(0)
            input_tensor1 = sail.Tensor(handle,(1,3),sail.Dtype.BM_FLOAT32,False,True)

            input_tensor1.ones()
