sail.Engine
___________

Engine可以实现bmodel的加载与管理，是实现模型推理的主要模块。

\_\_init\_\_
>>>>>>>>>>>>>>>>>>>>>

初始化Engine

**接口形式1:**

创建Engine实例，并不加载bmodel

    .. code-block:: python

        def __init__(tpu_id: int)
            
        def __init__(self, handle: sail.Handle)    

**参数说明1:**

* tpu_id: int

指定Engine实例使用的智能视觉深度学习处理器的id

* handle: sail.Handle

指定Engine实例使用的设备标识Handle


**接口形式2:**

创建Engine实例并加载bmodel，需指定bmodel路径或内存中的位置。

    .. code-block:: python

        def __init__(self, bmodel_path: str, tpu_id: int, mode: sail.IOMode)

        def __init__(self, bmodel_bytes: bytes, bmodel_size: int, tpu_id: int, mode: sail.IOMode)

**参数说明2:**

* bmodel_path: str

指定bmodel文件的路径

* tpu_id: int

指定Engine实例使用的智能视觉深度学习处理器的id

* mode: sail.IOMode

指定输入/输出Tensor所在的内存位置：系统内存或设备内存。

* bmodel_bytes: bytes

bmodel在系统内存中的bytes。

* bmodel_size: int

bmodel在内存中的字节数

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine1 = sail.Engine(0)

            handle = sail.Handle(0)
            engine2 = sail.Engine(bmodel_path,handle)

            engine3 = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)

            file = open(bmodel_path,"rb")
            datas = file.read()
            file_size = os.path.getsize(bmodel_path)
            engine4 = sail.Engine(datas,file_size,0,sail.IOMode.SYSI)

get_handle
>>>>>>>>>>>>>>>>>>>>>

获取Engine中使用的设备句柄sail.Handle

**接口形式:**
    .. code-block:: python

        def get_handle(self)->sail.Handle

**返回值说明:**

* handle: sail.Handle

返回Engine中的设备句柄。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            engine1 = sail.Engine(0)
            handle = engine1.get_handle()


load
>>>>>>>>>>>>>>>>>>>>>

将bmodel载入Engine中。

**接口形式1:**

指定bmodel路径，从文件中载入bmodel。

    .. code-block:: python

        def load(self, bmodel_path: str)->bool

**参数说明1:**

* bmodel_path: str

bmodel的文件路径

**接口形式2:**

从系统内存中载入bmodel。

    .. code-block:: python

        def load(self, bmodel_bytes: bytes, bmodel_size: int)->bool

**参数说明2:**

* bmodel_bytes: bytes

bmodel在系统内存中的bytes。

* bmodel_size: int

bmodel在内存中的字节数。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine1 = sail.Engine(0)
            engine1.load(bmodel_path)


get_graph_names
>>>>>>>>>>>>>>>>>>>>>

获取Engine中所有载入的计算图的名称。

**接口形式:**
    .. code-block:: python

        def get_graph_names(self)->list

**返回值说明:**

* graph_names: list

Engine中所有计算图的name的列表。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine1 = sail.Engine(0)
            engine1.load(bmodel_path)
            graph_names = engine1.get_graph_names()


set_io_mode
>>>>>>>>>>>>>>>>>>>>>

设置Engine的输入/输出Tensor所在的内存位置：系统内存或设备内存。

**接口形式:**
    .. code-block:: python

        def set_io_mode(self, graph_name: str, mode: sail.IOMode)->None

**参数说明:**

* graph_name: str

需要配置的计算图的name。

* mode: sail.IOMode

设置Engine的输入/输出Tensor所在的内存位置：系统内存或设备内存。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            engine.set_io_mode(graph_name,sail.IOMode.SYSI)

graph_is_dynamic
>>>>>>>>>>>>>>>>>>>>>

判断选定计算图是否为动态。

**接口形式:**
    .. code-block:: python

        def graph_is_dynamic(self, graph_name: str)->bool

**参数说明:**

* graph_name: str

设定需要查询的计算图的name。

**返回值说明:**

* is_dynamic: bool

返回选定计算图是否为动态的判断结果。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            is_dynamic = engine.graph_is_dynamic(graph_name)


get_input_names
>>>>>>>>>>>>>>>>>>>>>

获取选定计算图中所有输入Tensor的name

**接口形式:**
    .. code-block:: python

        def get_input_names(self, graph_name: str)->list

**参数说明:**

* graph_name: str

设定需要查询的计算图的name。

**返回值说明:**

* input_names: list

返回选定计算图中所有输入Tensor的name的列表。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            input_names = engine.get_input_names(graph_name)


get_output_names
>>>>>>>>>>>>>>>>>>>>>

获取选定计算图中所有输出Tensor的name。

**接口形式:**
    .. code-block:: python

        def get_output_names(self, graph_name: str)->list

**参数说明:**

* graph_name: str

设定需要查询的计算图的name。

**返回值说明:**

* output_names: list

返回选定计算图中所有输出Tensor的name的列表。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            output_names = engine.get_output_names(graph_name)


get_max_input_shapes
>>>>>>>>>>>>>>>>>>>>>

查询选定计算图中所有输入Tensor对应的最大shape。

在静态模型中，输入Tensor的shape是固定的，应等于最大shape。

在动态模型中，输入Tensor的shape应小于等于最大shape。

**接口形式:**
    .. code-block:: python

        def get_max_input_shapes(self, graph_name: str)->dict {str : list}

**参数说明:**

* graph_name: str

设定需要查询的计算图的name。

**返回值说明:**

* max_shapes: dict{str : list}

返回输入Tensor中的最大shape。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            max_input_shapes = engine.get_max_input_shapes(graph_name)


get_input_shape
>>>>>>>>>>>>>>>>>>>>>

查询选定计算图中特定输入Tensor的shape。

**接口形式:**
    .. code-block:: python

        def get_input_shape(self, graph_name: str, tensor_name: str)->list
            
**参数说明:**

* graph_name: str

设定需要查询的计算图的name。

* tensor_name: str

需要查询的Tensor的name。

**返回值说明:**

* tensor_shape: list

该name下的输入Tensor中的最大维度的shape。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            input_name = engine.get_input_names(graph_name)[0]
            input_shape = engine.get_input_shape(graph_name,input_name)


get_max_output_shapes
>>>>>>>>>>>>>>>>>>>>>>>

查询选定计算图中所有输出Tensor对应的最大shape。

在静态模型中，输出Tensor的shape是固定的，应等于最大shape。

在动态模型中，输出Tensor的shape应小于等于最大shape。

**接口形式:**
    .. code-block:: python

        def get_max_output_shapes(self, graph_name: str)->dict {str : list}

**参数说明:**

* graph_name: str

设定需要查询的计算图的name。

**返回值说明:**

* max_shapes: dict{str : list}

返回输出Tensor中的最大shape。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            max_output_shapes = engine.get_max_output_shapes(graph_name)


get_output_shape
>>>>>>>>>>>>>>>>>>>>>

查询选定计算图中特定输出Tensor的shape。

**接口形式:**
    .. code-block:: python

        def get_output_shape(self, graph_name: str, tensor_name: str)->list

**参数说明:**

* graph_name: str

设定需要查询的计算图的name。

* tensor_name: str

需要查询的Tensor的name。

**返回值说明:**

* tensor_shape: list

该name下的输出Tensor的shape。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            output_name = engine.get_output_names(graph_name)[0]
            output_shape = engine.get_output_shape(graph_name,output_name)


get_input_dtype
>>>>>>>>>>>>>>>>>>>>>

获取特定计算图的特定输入Tensor的数据类型。

**接口形式:**
    .. code-block:: python

        def get_input_dtype(self, graph_name: str, tensor_name: str)->sail.Dtype

**参数说明:**

* graph_name: str

设定需要查询的计算图的name。

* tensor_name: str

需要查询的Tensor的name。

**返回值说明:**

* datatype: sail.Dtype

返回Tensor中数据的数据类型。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            input_name = engine.get_input_names(graph_name)[0]
            input_dtype = engine.get_input_dtype(graph_name,input_name)

get_output_dtype
>>>>>>>>>>>>>>>>>>>>>

获取特定计算图的特定输出Tensor的数据类型。

**接口形式:**
    .. code-block:: python

        def get_output_dtype(self, graph_name: str, tensor_name: str)->sail.Dtype

**参数说明:**

* graph_name: str

设定需要查询的计算图的name。

* tensor_name: str

需要查询的Tensor的name。

**返回值说明:**

* datatype: sail.Dtype

返回Tensor中数据的数据类型。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            output_name = engine.get_output_names(graph_name)[0]
            output_dtype = engine.get_output_dtype(graph_name,output_name)

get_input_scale
>>>>>>>>>>>>>>>>>>>>>

获取特定计算图的特定输入Tensor的scale，只在int8模型中有效。

**接口形式:**
    .. code-block:: python

        def get_input_scale(self, graph_name: str, tensor_name: str)->float32

**参数说明:**

* graph_name: str

设定需要查询的计算图的name。

* tensor_name: str

需要查询的Tensor的name。

**返回值说明:**

* scale: float32

返回Tensor数据的scale。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            input_name = engine.get_input_names(graph_name)[0]
            input_scale = engine.get_input_scale(graph_name,input_name)

get_output_scale
>>>>>>>>>>>>>>>>>>>>>

获取特定计算图的特定输出Tensor的scale，只在int8模型中有效。

**接口形式:**
    .. code-block:: python

        def get_output_scale(self, graph_name: str, tensor_name: str)->float32

**参数说明:**

* graph_name: str

设定需要查询的计算图的name。

* tensor_name: str

需要查询的Tensor的name。

**返回值说明:**

* scale: float32

返回Tensor数据的scale。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            output_name = engine.get_output_names(graph_name)[0]
            output_scale = engine.get_output_scale(graph_name,output_name)

process
>>>>>>>>>>>>>>>>>>>>>

在特定的计算图上进行前向推理。

**接口形式1:**
    .. code-block:: python

        def process(self, graph_name: str, input_tensors: dict {str : numpy.array}, core_list: list[int])->dict {str : numpy.array}
            """ Inference with provided system data of input tensors.

**参数说明1:**

* graph_name: str

特定的计算图name。

* input_tensors: dict{str : numpy.array}

所有的输入Tensor的数据，利用系统内存中的numpy.array传入。

* core_list: list[int]

输入参数。该参数仅对支持多核推理的处理器有效，可以选择推理时使用的core。设bmodel为对应的核数为N，此时core_list为空或者core_list的长度大于N，都会使用从core0开始的N个core来做推理。对于仅支持单核推理的处理器可忽略此参数。

**返回值说明1:**

* output_tensors: dict{str : numpy.array}

所有的输出Tensor的数据，返回类型为numpy.array的数据。


**接口形式2:**
    .. code-block:: python

        def process(self, graph_name: str, input_tensors: dict {str : sail.Tensor}, output_tensors: dict {str : sail.Tensor}, core_list: list[int])->None
        
        def process(self, graph_name: str, input_tensors: dict {str : sail.Tensor}, input_shapes: dict {str : list}, output_tensors: dict {str : sail.Tensor}, core_list: list[int])->None

**参数说明2:**

* graph_name: str

输入参数。特定的计算图name。

* input_tensors: dict{str : sail.Tensor}

输入参数。所有的输入Tensor的数据，利用sail.Tensor传入。

* input_shapes : dict {str : list}

输入参数。所有传入Tensor的shape。

* output_tensors: dict{str : sail.Tensor}

输出参数。所有的输出Tensor的数据，利用sail.Tensor返回。

* core_list: list[int]

输入参数。该参数仅对支持多核推理的处理器有效，可以选择推理时使用的core。设bmodel为对应的核数为N，若core_list为空则使用从core0开始的N个core做推理；若core_list的长度大于N，则使用core_list中对应的前N个core做推理。对于仅支持单核推理的处理器可忽略此参数。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            # prepare tensor map
            input_tensors_map = engine.create_input_tensors_map(graph_name)
            # inference type1 
            output_tensors_map = engine.process(graph_name, input_tensors_map)
            
            # inference type2 
            output_tensors_map_ = engine.create_output_tensors_map(graph_name)
            engine.process(graph_name, input_tensors_map, output_tensors_map_)


get_device_id
>>>>>>>>>>>>>>>>>>>>>

获取Engine中的设备id号

**接口形式:**
    .. code-block:: python

        def get_device_id(self)->int

**返回值说明:**

* tpu_id : int

返回Engine中的设备id号。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            dev_id = engine.get_device_id()


create_input_tensors_map
>>>>>>>>>>>>>>>>>>>>>>>>>>>

创建输入Tensor的映射，在python接口中为字典dict{str : Tensor}

**接口形式:**
    .. code-block:: python

        def create_input_tensors_map(self, graph_name: str, create_mode: int = -1)->dict{str : Tensor}

**参数说明:**

* graph_name: str

特定的计算图name。

* create_mode: int

创建Tensor分配内存的模式。为0时只分配系统内存，为1时只分配设备内存，其他时则根据Engine中IOMode的配置分配。

**返回值说明:**

output: dict{str : Tensor}

返回name:tensor的字典。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            # prepare tensor map
            input_tensors_map = engine.create_input_tensors_map(graph_name)

create_output_tensors_map
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

创建输入Tensor的映射，在python接口中为字典dict{str : Tensor}

**接口形式:**
    .. code-block:: python

        def create_output_tensors_map(self, graph_name: str, create_mode: int = -1)->dict{str : Tensor}

**参数说明:**

* graph_name: str

特定的计算图name。

* create_mode: int

创建Tensor分配内存的模式。为0时只分配系统内存，为1时只分配设备内存，其他时则根据Engine中IOMode的配置分配。

**返回值说明:**

output: dict{str : Tensor}

返回name:tensor的字典。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            # prepare tensor map
            output_tensors_map = engine.create_output_tensors_map(graph_name)
