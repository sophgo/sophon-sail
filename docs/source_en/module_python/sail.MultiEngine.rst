sail.MultiEngine
________________

MultiEngine
>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: python

        def __init__(self, 
                    bmodel_path: str, 
                    device_ids: list[int], 
                    sys_out: bool = True, 
                    graph_idx: int = 0)

**Parameters**

* bmodel_path : str

Path to bmodel

* device_ids : list[int]    

Tensor Computing Processor ID. You can use bm-smi to see available IDs

* sys_out : bool, default: True

The flag of copy result to system memory.

* graph_idx : int, default: 0

The specified graph index

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            dev_id = [0,1]
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.MultiEngine(bmodel_path,dev_id)


set_print_flag
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Print debug messages.

**Interface:**
    .. code-block:: python

        def set_print_flag(self, print_flag: bool)
 

**Parameters**

* print_flag : bool

if print_flag is true, print debug messages

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            dev_id = [0,1]
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.MultiEngine(bmodel_path,dev_id)
            engine.set_print_flag(true)


set_print_time
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Print main process time use.

**Interface:**
    .. code-block:: python
        
        def set_print_time(self, print_flag: bool)
 

**Parameters**

* print_flag : bool

if print_flag is true, print main process time use, Otherwise not print.

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            dev_id = [0,1]
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.MultiEngine(bmodel_path,dev_id)
            engine.set_print_time(true)


get_device_ids
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get device ids of this MultiEngine.

**Interface:**
    .. code-block:: python

        def get_device_ids(self)-> list[int] 
 
            
**Returns**

* device_ids : list[int]    

Tensor Computing Processor ids of this MultiEngine.

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

Get all graph names in the loaded bmodels.

**Interface:**
    .. code-block:: python

        def get_graph_names(self)-> list
 

**Returns**

* graph_names : list

Graph names list in loaded context

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

Get all input tensor names of the specified graph.

**Interface:**
    .. code-block:: python

        def get_input_names(self, graph_name: str)-> list
 

**Parameters**

* graph_name : str

Specified graph name

**Returns**

* input_names : list

All the input tensor names of the graph

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

Get all output tensor names of the specified graph.

**Interface:**
    .. code-block:: python

        def get_output_names(self, graph_name: str)-> list
 

**Parameters**

* graph_name : str

Specified graph name

**Returns**

* input_names : list

All the output tensor names of the graph

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

Get the maximum dimension shape of an input tensor in a graph. \
There are cases that there are multiple input shapes in one input name, \
This API only returns the maximum dimension one for the memory allocation \
in order to get the best performance.

**Interface:**
    .. code-block:: python

        def get_input_shape(self, graph_name: str, tensor_name: str)-> list
 

**Parameters**

* graph_name : str

The specified graph name

* tensor_name : str

The specified input tensor name

**Returns**

* tensor_shape : list

The maxmim dimension shape of the tensor

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

Get the shape of an output tensor in a graph.

**Interface:**
    .. code-block:: python

        def get_output_shape(self, graph_name: str, tensor_name: str)-> list
 

**Parameters**

* graph_name : str

The specified graph name

* tensor_name : str

The specified output tensor name

**Returns**

* tensor_shape : list

The shape of the tensor

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

Inference with provided system data of input tensors.

**Interface:**
    .. code-block:: python

        def process(self, input_tensors: dict {str : numpy.array})-> dict {str : numpy.array}

        def process(self, input_tensors: list[dict{str: sophon.sail.Tensor}] )->dict {str : Tensor}

**Parameters**

* input_tensors : dict {str : numpy.array}

Data of all input tensors in system memory

**Returns**

* output_tensors : dict {str : numpy.array}

Data of all output tensors in system memory


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