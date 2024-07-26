sail.MultiEngine
________________

多线程的推理引擎，实现特定计算图的多线程推理。

MultiEngine
>>>>>>>>>>>>>>>

初始化MutiEngine。

**接口形式:**
    .. code-block:: python

        def __init__(self, bmodel_path: str, device_ids: list[int], sys_out: bool=True, graph_idx: int=0)

**参数说明:**

* bmodel_path: str

bmodel所在的文件路径。

* device_ids: lists[int]

该MultiEngine可见的智能视觉深度学习处理器的ID。

* sys_out: bool

表示是否将结果拷贝到系统内存，默认为True

* graph_idx : int

特定的计算图的index。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            dev_id = [0,1]
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.MultiEngine(bmodel_path,dev_id)
        
set_print_flag
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

设置是否打印调试信息。

**接口形式:**
    .. code-block:: python

        def set_print_flag(self, print_flag: bool)->None

**参数说明:**

* print_flag: bool

为True时，打印调试信息，否则不打印。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            dev_id = [0,1]
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.MultiEngine(bmodel_path,dev_id)
            engine.set_print_flag(True)

set_print_time
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

设置是否打印主要处理耗时。

**接口形式:**
    .. code-block:: python
        
        def set_print_time(self, print_flag: bool)->None

**参数说明:**

* print_flag: bool

为True时，打印主要耗时，否则不打印。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            dev_id = [0,1]
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.MultiEngine(bmodel_path,dev_id)
            engine.set_print_time(True)

get_device_ids
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取MultiEngine中所有可用的智能视觉深度学习处理器的id。

**接口形式:**
    .. code-block:: python

        def get_device_ids(self) -> list[int] 

**返回值说明:**

* device_ids: list[int]

返回可见的智能视觉深度学习处理器的ids

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            dev_id = [0,1]
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.MultiEngine(bmodel_path,dev_id)
            ids = engine.get_device_ids()


get_graph_names
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取MultiEngine中所有载入的计算图的名称。

**接口形式:**
    .. code-block:: python

        def get_graph_names(self)->list 

**返回值说明:**

* graph_names: list

MultiEngine中所有计算图的name的列表。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            dev_id = [0,1]
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.MultiEngine(bmodel_path,dev_id)
            graph_names = engine.get_graph_names()

get_input_names
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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
            dev_id = [0,1]
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.MultiEngine(bmodel_path,dev_id)
            graph_names = engine.get_graph_names()
            input_names = engine.get_input_names(graph_names[0])

get_output_names
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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
            dev_id = [0,1]
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.MultiEngine(bmodel_path,dev_id)
            graph_names = engine.get_graph_names()
            output_names = engine.get_output_names(graph_names[0])
            

get_input_shape
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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
            dev_id = [0,1]
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.MultiEngine(bmodel_path,dev_id)
            graph_names = engine.get_graph_names()
            input_names = engine.get_input_names(graph_names[0])
            input_shape = engine.get_input_shape(graph_name,input_names[0])

get_output_shape
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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
            dev_id = [0,1]
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.MultiEngine(bmodel_path,dev_id)
            graph_names = engine.get_graph_names()
            output_names = engine.get_output_names(graph_names[0])
            output_shape = engine.get_output_shape(graph_name,output_names[0])

process
>>>>>>>>>>>>>>>>>>>

在特定的计算图上进行推理，需要提供系统内存的输入数据。

**接口形式:**
    .. code-block:: python

        def process(self, input_tensors: dict {str : numpy.array})->dict {str : numpy.array}

        def process(self, input_tensors: list[dict{str: sophon.sail.Tensor}] )->dict {str : Tensor}

**参数说明:**

* input_tensors: dict{ str : numpy.array }

输入的Tensors。

**返回值说明:**

* output_tensors: dict{str : numpy.array}

返回推理之后的结果。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np

        if __name__ == '__main__':
            dev_id = [0,1]
            handle = sail.Handle(0)
            bmodel_path = 'your_bmodel.bmodel'
            engine = sail.MultiEngine(bmodel_path, dev_id)
            graph_name = engine.get_graph_names()[0]
            input_names = engine.get_input_names(graph_name)
            output_names = engine.get_output_names(graph_name)

            input_tensors_map = {}
            
            # form 1
            input_numpy_map = {}
            for input_name in input_names:
                data = np.ones(engine.get_input_shape(graph_name,input_name),dtype=np.float32)
                input_numpy_map = {input_name:data}
            output_tensors_map = engine.process(input_numpy_map)
            print(output_tensors_map)
            
            # form 2 
            for input_name in input_names:
                data = np.ones(engine.get_input_shape(graph_name,input_name),dtype=np.float32)
                tensor = sail.Tensor(handle,data)
                input_tensors_map[input_name] = tensor
            input_tensors_vector = [input_tensors_map]
            output_tensors_map = engine.process(input_tensors_vector)
            print(output_tensors_map)