sail.EngineLLM
______________

EngineLLM针对大语言模型推理而设计，可以实现多芯bmodel的加载与管理。

通用参数说明
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **graph_name**: str

  需要获取信息的计算图的名称

- **tensor_name**: str

  需要获取信息的Tensor（张量）的名称

- **index**: int

  Tensor的索引

- **stage**: int

  stage的索引

构造函数 \_\_init\_\_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EngineLLM构造函数，初始化EngineLLM实例。

**接口形式1:**

创建EngineLLM实例，从bmodel文件中加载模型。

    .. code-block:: python

        def __init__(self, bmodel_path: str, dev_ids: list[int])

**参数说明1:**

* bmodel_path: str

bmodel文件的路径

* dev_ids: list[int]

EngineLLM实例使用的智能视觉深度学习处理器的id列表


**接口形式2:**

创建EngineLLM实例，从内存数据中加载模型。

    .. code-block:: python

        def __init__(self, bmodel_bytes: bytes, bmodel_size: int, dev_ids: list[int])    

**参数说明2:**

* bmodel_bytes: bytes

bmodel在系统内存中的bytes

* bmodel_size: int

bmodel在内存中的字节数

* dev_ids: list[int]

EngineLLM实例使用的智能视觉深度学习处理器的id列表

**接口形式3:**

创建EngineLLM实例，从bmodel文件中加载模型，加载时使用指定的bmruntime flag。
典型使用场景是，在BM1688设备上执行LLM模型的推理时，可以设置flag为 ``BM_RUNTIME_SHARE_MEM`` ，以节省设备内存。

    .. code-block:: python

        def __init__(self, bmodel_path: str, flags: int, dev_ids: list[int])

**参数说明1:**

* bmodel_path: str

bmodel文件的路径

* flags: int

加载bmodel使用的bmruntime flag。
推荐通过 ``BmrtFlag`` 枚举设置，例如 ``sail.BmrtFlag.BM_RUNTIME_SHARE_MEM`` 。
更多信息请参考《BMRuntime 开发参考手册》。

* dev_ids: list[int]

EngineLLM实例使用的智能视觉深度学习处理器的id列表

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import os

        if __name__ == '__main__':
            bmodel_path = "example_8dev.bmodel"
            dev_ids = [i for i in range(8)]

            engine1 = sail.EngineLLM(bmodel_path, dev_ids)

            file = open(bmodel_path,"rb")
            datas = file.read()
            file_size = os.path.getsize(bmodel_path)
            engine2 = sail.EngineLLM(datas, file_size, dev_ids)

            llm_bmodel_path = "llama.bmodel"
            flag = sail.BmrtFlag.BM_RUNTIME_SHARE_MEM
            engine3 = sail.EngineLLM(llm_bmodel_path, flag, dev_ids)

推理接口 process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用给定的输入输出Tensor，在某一个计算图上执行推理。

**接口形式:**
    .. code-block:: python

        def process(self, 
                    graph_name: str, 
                    input: dict[int, Tensor], 
                    output: dict[int, Tensor])
                    -> int

**参数说明:**

* graph_name: str

需要推理的计算图名称

* input: dict[int, Tensor]

输入Tensor

* output: dict[int, Tensor]

输出Tensor

**返回值说明:**

* return: int

返回0表示推理成功，其他值表示失败

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            bmodel_path = "example_8dev.bmodel"
            dev_ids = [i for i in range(8)]

            engine1 = sail.EngineLLM(bmodel_path, dev_ids)
            graph_name_0 = engine1.get_graph_names()[0]
            input_tensors = engine1.get_input_tensors(graph_name_0)
            output_tensors = engine1.get_output_tensors(graph_name_0)
            ret = engine1.process(graph_name_0, input_tensors, output_tensors)
            if (ret):
                print(f"{graph_name_0} inference failed!")
            else:
                print(f"{graph_name_0} inference succeeded!")


获取信息接口
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

此小节介绍的接口用于从EngineLLM实例中获取模型信息。

小节末尾提供了调用这些接口的例程。

get_device_ids
>>>>>>>>>>>>>>>>>>>>>

获取EngineLLM所使用的设备号列表。

**接口形式:**
    .. code-block:: python

        def get_device_ids(self) -> list[int]

**返回值说明:**

* dev_ids: list[int]

返回EngineLLM所使用的设备号列表


get_graph_names
>>>>>>>>>>>>>>>>>>>>>

获取EngineLLM中所有载入的计算图（graph）的名称。

**接口形式:**
    .. code-block:: python

        def get_graph_names(self) -> list[str]

**返回值说明:**

* graph_names: list[str]

获取EngineLLM中所有计算图的名称


get_addr_mode
>>>>>>>>>>>>>>>>>>>>>

获取EngineLLM中某个指定计算图的addr_mode。

**接口形式:**
    .. code-block:: python

        def get_addr_mode(self, graph_name: str) -> int


get_stage_num
>>>>>>>>>>>>>>>>>>>>>

获取EngineLLM中某个指定计算图的stage_num。
stage_num的定义请参考《BMRuntime开发参考手册》。

**接口形式:**
    .. code-block:: python

        def get_stage_num(self, graph_name: str) -> int


get_input_num
>>>>>>>>>>>>>>>>>>>>>

获取EngineLLM中某个指定计算图的输入的个数。

**接口形式:**
    .. code-block:: python

        def get_input_num(self, graph_name: str) -> int


get_output_num
>>>>>>>>>>>>>>>>>>>>>

获取EngineLLM中某个指定计算图的输出的个数。

**接口形式:**
    .. code-block:: python

        def get_output_num(self, graph_name: str) -> int


get_is_dynamic
>>>>>>>>>>>>>>>>>>>>>

获取EngineLLM中某个指定计算图是否是动态的。
动态网络的定义请参考《BMRuntime开发参考手册》。

**接口形式:**
    .. code-block:: python

        def get_is_dynamic(self, graph_name: str) -> bool


get_input_name
>>>>>>>>>>>>>>>>>>>>>

获取指定计算图中某个索引的输入Tensor名称。

**接口形式:**
    .. code-block:: python

        def get_input_name(self, graph_name: str, index: int) -> str


get_output_name
>>>>>>>>>>>>>>>>>>>>>

获取指定计算图中某个索引的输出Tensor名称。

**接口形式:**
    .. code-block:: python

        def get_output_name(self, graph_name: str, index: int) -> str


get_input_tensor_devid
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取指定计算图中某个索引的输入Tensor的设备ID。

**接口形式:**
    .. code-block:: python

        def get_input_tensor_devid(self, graph_name: str, index: int) -> int


get_output_tensor_devid
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取指定计算图中某个索引的输出Tensor的设备ID。

**接口形式:**
    .. code-block:: python

        def get_output_tensor_devid(self, graph_name: str, index: int) -> int


get_input_shape
>>>>>>>>>>>>>>>>>>>>>

获取指定计算图中某个索引的输入Tensor的形状。

**接口形式:**
    .. code-block:: python

        def get_input_shape(self, graph_name: str, index: int, stage: int = 0) -> list[int]


get_output_shape
>>>>>>>>>>>>>>>>>>>>>

获取指定计算图中某个索引的输出Tensor的形状。

**接口形式:**
    .. code-block:: python

        def get_output_shape(self, graph_name: str, index: int, stage: int = 0) -> list[int]


get_input_max_shape
>>>>>>>>>>>>>>>>>>>>>

获取指定计算图中某个索引的输入Tensor在不同stage中的最大形状。

**接口形式:**
    .. code-block:: python

        def get_input_max_shape(self, graph_name: str, index: int) -> list[int]


get_output_max_shape
>>>>>>>>>>>>>>>>>>>>>

获取指定计算图中某个索引的输出Tensor在不同stage中的最大形状。

**接口形式:**
    .. code-block:: python

        def get_output_max_shape(self, graph_name: str, index: int) -> list[int]


get_input_dtype
>>>>>>>>>>>>>>>>>>>>>

获取指定计算图中某个索引的输入Tensor的数据类型。

**接口形式:**
    .. code-block:: python

        def get_input_dtype(self, graph_name: str, index: int) -> Dtype


get_output_dtype
>>>>>>>>>>>>>>>>>>>>>

获取指定计算图中某个索引的输出Tensor的数据类型。

**接口形式:**
    .. code-block:: python

        def get_output_dtype(self, graph_name: str, index: int) -> Dtype


get_input_scale
>>>>>>>>>>>>>>>>>>>>>

获取指定计算图中某个索引的输入Tensor的缩放因子。

**接口形式:**
    .. code-block:: python

        def get_input_scale(self, graph_name: str, index: int) -> float


get_output_scale
>>>>>>>>>>>>>>>>>>>>>

获取指定计算图中某个索引的输出Tensor的缩放因子。

**接口形式:**
    .. code-block:: python

        def get_output_scale(self, graph_name: str, index: int) -> float


**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            bmodel_path = "example_8dev.bmodel"
            dev_ids: list[int] = [i for i in range(8)]
            engine1 = sail.EngineLLM(bmodel_path, dev_ids)
            dev_ids = engine1.get_device_ids()
            graph_names = engine1.get_graph_names()
            graph_name_0 = graph_names[0]

            query_index = 0
            query_stage = 0
            addr_mode = engine1.get_addr_mode(graph_name_0)
            stage_num = engine1.get_stage_num(graph_name_0)
            input_num = engine1.get_input_num(graph_name_0)
            is_dynamic = engine1.get_is_dynamic(graph_name_0)
            input_name = engine1.get_input_name(graph_name_0, query_index)
            input_tensor_devid = engine1.get_input_tensor_devid(
                                        graph_name_0, query_index)
            input_shape = engine1.get_input_shape(
                                            graph_name_0, query_index, query_stage)
            input_max_shape = engine1.get_input_max_shape(
                                    graph_name_0, query_index)
            input_dtype = engine1.get_input_dtype(graph_name_0, query_index)
            input_scale = engine1.get_input_scale(graph_name_0, query_index)
            # usage about output is omitted, which is the same as input


创建Tensor接口
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

get_input_tensors
>>>>>>>>>>>>>>>>>>>>>

获取指定计算图中符合条件的输入Tensor及其索引。

该接口使用net_info中的input_mems创建Tensor，而非额外重新申请内存。
为了内存安全，该接口仅在addr_mode模式为1时可用，否则返回的字典为空。
input_mems和addr_mode的定义请参考《BMRuntime开发参考手册》。

**接口形式1:**

根据Tensor名称和stage，获取一组对应的输入Tensor。

    .. code-block:: python

        def get_input_tensors(self, graph_name: str, tensor_name: str, stage: int = 0) -> dict[int, Tensor]

**接口形式2:**

根据stage，获取一组对应的输入Tensor。

    .. code-block:: python

        def get_input_tensors(self, graph_name: str, stage: int = 0) -> dict[int, Tensor]

**返回值说明:**

返回由索引和Tensor组成的字典dict[int, Tensor]


get_input_tensor
>>>>>>>>>>>>>>>>>>>

根据索引和stage，获取一个对应的输入Tensor。

该接口使用net_info中的input_mems创建Tensor，而非额外重新申请内存。
为了内存安全，该接口仅在addr_mode模式为1时可用，否则返回的字典为空。
input_mems和addr_mode的定义请参考《BMRuntime开发参考手册》。

**接口形式:**

根据Tensor名称和索引，获取一个对应的输入Tensor。

    .. code-block:: python

        def get_input_tensor(self, graph_name: str, index: int, stage: int = 0) -> Tensor


get_output_tensors
>>>>>>>>>>>>>>>>>>>>>

获取指定计算图中符合条件的输出Tensor及其索引。

该接口使用net_info中的output_mems创建Tensor，而非额外重新申请内存。
为了内存安全，该接口仅在addr_mode模式为1时可用，否则返回的字典为空。
output_mems和addr_mode的定义请参考《BMRuntime开发参考手册》。

**接口形式1:**

根据Tensor名称和stage，获取一组对应的输出Tensor。

    .. code-block:: python

        def get_output_tensors(self, graph_name: str, tensor_name: str, stage: int = 0) -> dict[int, Tensor]

**接口形式2:**

根据stage，获取一组对应的输出Tensor。

    .. code-block:: python

        def get_output_tensors(self, graph_name: str, stage: int = 0) -> dict[int, Tensor]

**返回值说明:**

返回由索引和Tensor组成的字典dict[int, Tensor]


get_output_tensor
>>>>>>>>>>>>>>>>>>>

根据索引和stage，获取一个对应的输出Tensor。

该接口使用net_info中的output_mems创建Tensor，而非额外重新申请内存。
为了内存安全，该接口仅在addr_mode模式为1时可用，否则返回的字典为空。
output_mems和addr_mode的定义请参考《BMRuntime开发参考手册》。

**接口形式:**

根据Tensor名称和索引，获取一个对应的输出Tensor。

    .. code-block:: python

        def get_output_tensor(self, graph_name: str, index: int, stage: int = 0) -> Tensor


get_input_tensors_addrmode0
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

根据Tensor名称和stage，获取一组对应的输入Tensor。

该接口使用net_info中的input_mems创建Tensor，而非额外重新申请内存。
该接口在addr_mode模式不为1时也可用，调用者需要确认Tensor的内存安全。
addr_mode的定义请参考《BMRuntime开发参考手册》。

**接口形式:**

获取所有输入Tensors。

    .. code-block:: python

        def get_input_tensors_addrmode0(self, graph_name: str, stage: int = 0) -> dict[int, Tensor]

**返回值说明:**

返回由索引和Tensor组成的字典dict[int, Tensor]


get_output_tensors_addrmode0
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

根据Tensor名称和stage，获取一组对应的输出Tensor。

该接口使用net_info中的output_mems创建Tensor，而非额外重新申请内存。
该接口在addr_mode模式不为1时也可用，调用者需要确认Tensor的内存安全。
addr_mode的定义请参考《BMRuntime开发参考手册》。

**接口形式:**

获取所有输出Tensors。

    .. code-block:: python

        def get_output_tensors_addrmode0(self, graph_name: str, stage: int = 0) -> dict[int, Tensor]

**返回值说明:**

返回由索引和Tensor组成的字典dict[int, Tensor]