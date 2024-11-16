sail.Engine
___________

\_\_init\_\_
>>>>>>>>>>>>>>>>>>>>>

Constructor without or with bmodel loaded.

**Interface:**
    .. code-block:: python

        def __init__(tpu_id: int)

**Parameters**

* tpu_id : int

Tensor Computing Processor ID. You can use bm-smi to see available IDs


**Interface:**
    .. code-block:: python

        def __init__(self, handle: Handle)

**Parameters**

* hanle : Handle

A Handle instance


**Interface:**
    .. code-block:: python

        def __init__(self, 
                    bmodel_path: str, 
                    tpu_id: int, 
                    mode: sail.IOMode)

**Parameters**

* bmodel_path : str

Path to bmodel

* tpu_id : int

Tensor Computing Processor ID. You can use bm-smi to see available IDs

* mode : sail.IOMode

Specify the input/output tensors are in system memory or device memory


**Interface:**
    .. code-block:: python

        def __init__(self, 
                    bmodel_bytes:str, 
                    bmodel_size: int, 
                    tpu_id: int, 
                    mode: sail.IOMode)


**Parameters**

* bmodel_bytes : bytes

Bytes of  bmodel in system memory

* bmodel_size : int

Bmodel byte size

* tpu_id : int

Tensor Computing Processor ID. You can use bm-smi to see available IDs

* mode : sail.IOMode

Specify the input/output tensors are in system memory or device memory

**Sample:**
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

Get Handle instance.

**Interface:**
    .. code-block:: python

        def get_handle(self)->sail.Handle


**Returns**

* handle: sail.Handle

Handle instance

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            engine1 = sail.Engine(0)
            handle = engine1.get_handle()


load
>>>>>>>>>>>>>>>>>>>>>

Load bmodel from file.

**Interface:**
    .. code-block:: python

        def load(self, bmodel_path: str)->bool


**Parameters**

* bmodel_path : str

Path to bmodel

**Interface:**
    .. code-block:: python

        def load(self, bmodel_bytes: bytes, bmodel_size: int)->bool

**Parameters**

* bmodel_bytes : bytes

Bytes of  bmodel in system memory

* bmodel_size : int

Bmodel byte size


**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine1 = sail.Engine(0)
            engine1.load(bmodel_path)


get_graph_names
>>>>>>>>>>>>>>>>>>>>>

Get all graph names in the loaded bmodels.

**Interface:**
    .. code-block:: python

        def get_graph_names(self)-> list

**Returns**

* graph_names : list

Graph names list in loaded context

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine1 = sail.Engine(0)
            engine1.load(bmodel_path)
            graph_names = engine1.get_graph_names()


set_io_mode
>>>>>>>>>>>>>>>>>>>>>

Set IOMode for a graph.

**Interface:**
    .. code-block:: python

        def set_io_mode(self, graph_name: str, mode: sail.IOMode)

**Parameters**

* graph_name: str

The specified graph name

* mode : sail.IOMode

Specified io mode


**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            engine.set_io_mode(graph_name,sail.IOMode.SYSI)


graph_is_dynamic
>>>>>>>>>>>>>>>>>>>>>

Determine whether a selected computational map is dynamic.

**Interface:**
    .. code-block:: python

        def graph_is_dynamic(self, graph_name: str) -> list

**Parameters**

* graph_name : str

Specified graph name

**Returns**

* is_dynamic : bool

A boolean value indicating whether the selected computation graph is dynamic or not.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            is_dynamic = engine.graph_is_dynamic(graph_name)


get_input_names
>>>>>>>>>>>>>>>>>>>>>

Get all input tensor names of the specified graph.

**Interface:**
    .. code-block:: python

        def get_input_names(self, graph_name: str) -> list

**Parameters**

* graph_name : str

Specified graph name

**Returns**

* input_names : list

All the input tensor names of the graph

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            input_names = engine.get_input_names(graph_name)


get_output_names
>>>>>>>>>>>>>>>>>>>>>

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

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            output_names = engine.get_output_names(graph_name)


get_max_input_shapes
>>>>>>>>>>>>>>>>>>>>>

Get max shapes of input tensors in a graph. \
For static models, the max shape is fixed and it should not be changed. \
For dynamic models, the tensor shape should be smaller than or equal to \
the max shape.

**Interface:**
    .. code-block:: python

        def get_max_input_shapes(self, graph_name: str)-> dict {str : list}

**Parameters**

* graph_name : str

The specified graph name

**Returns**

* max_shapes : dict {str : list}

Max shape of the input tensors

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            max_input_shapes = engine.get_max_input_shapes(graph_name)


get_input_shape
>>>>>>>>>>>>>>>>>>>>>

Get the maximum dimension shape of an input tensor in a graph. \
There are cases that there are multiple input shapes in one input name, \
This API only returns the maximum dimension one for the memory allocation  \
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

**Sample:**
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

Get max shapes of input tensors in a graph. \
For static models, the max shape is fixed and it should not be changed. \
For dynamic models, the tensor shape should be smaller than or equal to \
the max shape.

**Interface:**
    .. code-block:: python

        def get_max_output_shapes(self, graph_name: str)-> dict {str : list}

**Parameters**

* graph_name : str

The specified graph name

**Returns**

* max_shapes : dict {str : list}

Max shape of the output tensors

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            max_output_shapes = engine.get_max_output_shapes(graph_name)


get_output_shape
>>>>>>>>>>>>>>>>>>>>>

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

tensor_shape : list

The shape of the tensor

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            output_name = engine.get_output_names(graph_name)[0]
            input_shape = engine.get_output_shape(graph_name,output_name)


get_input_dtype
>>>>>>>>>>>>>>>>>>>>>

Get scale of an input tensor. Only used for int8 models.

**Interface:**
    .. code-block:: python

        def get_input_dtype(self, graph_name: str, tensor_name: str)-> sail.Dtype

**Parameters**

* graph_name : str

The specified graph name

* tensor_name : str

The specified output tensor name

**Returns**

* scale: sail.Dtype

Data type of the input tensor

**Sample:**
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

Get scale of an output tensor. Only used for int8 models.

**Interface:**
    .. code-block:: python

        def get_output_dtype(self, graph_name: str, tensor_name: str)-> sail.Dtype

**Parameters**

* graph_name : str

The specified graph name

* tensor_name : str

The specified output tensor name

**Returns**

* scale: sail.Dtype

Data type of the output tensor

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            output_name = engine.get_output_names(graph_name)[0]
            input_shape = engine.get_output_dtype(graph_name,output_name)

get_input_scale
>>>>>>>>>>>>>>>>>>>>>

Get scale of an input tensor. Only used for int8 models.

**Interface:**
    .. code-block:: python

        def get_input_scale(self, graph_name: str, tensor_name: str)-> float32

**Parameters**
            
* graph_name : str

The specified graph name

* tensor_name : str

The specified output tensor name

**Returns**

* scale: float32

Scale of the input tensor

**Sample:**
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

Get scale of an output tensor. Only used for int8 models.

**Interface:**
    .. code-block:: python

        def get_output_scale(self, graph_name: str, tensor_name: str)-> float32

**Parameters**

* graph_name : str

The specified graph name

* tensor_name : str

The specified output tensor name

**Returns**

* scale: float32

Scale of the output tensor

**Sample:**
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

Inference with provided system data of input tensors, \
with or without input shapes and output tensors.

**Interface:**
    .. code-block:: python

        def process(self, 
                    graph_name: str,
                    input_tensors: dict {str : numpy.array},
                    core_list: list[int])-> dict {str : numpy.array}


**Parameters**

* graph_name : str

The specified graph name

* input_tensors : dict {str : numpy.array}

Data of all input tensors in system memory

* core_list : list[int]

This parameter is only valid for processors that support multi-core inference, and the core used for inference can be selected. Set bmodel as the corresponding kernel number N, and if corelist is empty, use N cores starting from core0 for inference; If the length of corelist is greater than N, use the corresponding top N cores in corelist for inference. This parameter can be ignored for processors that only support single core inference.
**Returns**

* output_tensors : dict {str : numpy.array}

Data of all output tensors in system memory


**Interface:**
    .. code-block:: python

        def process(self, 
                    graph_name: str, 
                    input_tensors: dict {str : sail.Tensor}, 
                    output_tensors: dict {str : sail.Tensor},
                    core_list: list[int])


**Parameters**

* graph_name : str

The specified graph name

* input_tensors : dict {str : sail.Tensor}

Input tensors managed by user

* output_tensors : dict {str : sail.Tensor}

Output tensors managed by user

* core_list : list[int]

This parameter is only valid for processors that support multi-core inference, and the core used for inference can be selected. Set bmodel as the corresponding kernel number N, and if corelist is empty, use N cores starting from core0 for inference; If the length of corelist is greater than N, use the corresponding top N cores in corelist for inference. This parameter can be ignored for processors that only support single core inference.

**Interface:**
    .. code-block:: python

        def process(self, 
                    graph_name: str, 
                    input_tensors: dict {str : sail.Tensor}, 
                    input_shapes: dict {str : list}, 
                    output_tensors: dict {str : sail.Tensor},
                    core_list: list[int])

**Parameters**

* graph_name : str

The specified graph name

* input_tensors : dict {str : sail.Tensor}

Input tensors managed by user

* input_shapes : dict {str : list}

Shapes of all input tensors

* output_tensors : dict {str : sail.Tensor}

Output tensors managed by user

* core_list : list[int]

This parameter is only valid for processors that support multi-core inference, and the core used for inference can be selected. Set bmodel as the corresponding kernel number N, and if corelist is empty, use N cores starting from core0 for inference; If the length of corelist is greater than N, use the corresponding top N cores in corelist for inference. This parameter can be ignored for processors that only support single core inference.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import os
        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            # prepare tensor map
            input_tensors_map = engine.create_input_tensors_map(graph_name)
            # inference type1 
            output_tensors_map = engine.process(graph_name, input_tensors_map)
            
            # inference type2 
            output_tensors_map_ = engine.create_output_tensors_map(graph_name,)
            engine.process(graph_name, input_tensors_map, output_tensors_map_)

get_device_id
>>>>>>>>>>>>>>>>>>>>>

Get device id of this engine 

**Interface:**
    .. code-block:: python

        def get_device_id(self)-> int
 
            
**Returns**

* tpu_id : int

Tensor Computing Processor id of this engine 
   
**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            dev_id = engine.get_device_id()


create_input_tensors_map
>>>>>>>>>>>>>>>>>>>>>>>>>>>

Create input tensors map, according to and bmodel.

**Interface:**
    .. code-block:: python

        def create_input_tensors_map(self, 
                                    graph_name: str, 
                                    create_mode: int)-> dict[str,Tensor]
 
**Parameters**:

* graph_name : str

The specified graph name.

* create_mode: int

Tensor Create mode, \
case 0: only allocate system memory; \
case 1: only allocate device memory; \
case other: according to engine IOMode.

**Returns**

* output: dict[str,Tensor]

Output result.

**Sample:**
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

Create output tensors map, according to and bmodel.

**Interface:**
    .. code-block:: python

        def create_output_tensors_map(self, 
                                    graph_name: str, 
                                    create_mode: int)-> dict[str,Tensor]
 

**Parameters**:

* graph_name : str

The specified graph name.

* create_mode: int

Tensor Create mode, \
case 0: only allocate system memory; \
case 1: only allocate device memory; \
case other: according to engine IOMode.

**Returns**

* output: dict[str,Tensor]

Output result.


**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            bmodel_path = "your_bmodel.bmodel"
            engine = sail.Engine(bmodel_path,0,sail.IOMode.SYSI)
            graph_name = engine.get_graph_names()[0]
            # prepare tensor map
            output_tensors_map = engine.create_output_tensors_map(graph_name)
