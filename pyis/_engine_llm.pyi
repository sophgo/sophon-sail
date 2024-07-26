from typing import Any
from ._basic import *

class EngineLLM:
    def __init__(self, bmodel_path: str, dev_ids: list[int]) :
        """ Constructor, which loads bmodel from file. 
        
        Parameters:
        -----------
        bmodel_path: str
            The path of bmodel to be load.
        dev_ids : list[int]
            A list of int, representing device indexes used to inference.

        """
        pass

    def __init__(self, bmodel_bytes: bytes, bmodel_size: int, dev_ids: list[int]) : 
        """ Constructor, which loads bmodel from bytes.

        Parameters:
        -----------
        bmodel_bytes: bytes
            A bytes data representing the bmodel to be load.
        bmodel_size: int
            The size of the binary model in bytes. 
            This is typically the length of the `bmodel_bytes`.
        dev_ids : list[int]
            A list of int, representing device indexes used to inference.

        """ 
        pass

    def get_device_ids(self) -> list[int]:
        """Get device id list of this engine """
        pass

    def get_graph_names(self) -> list[str]:
        """Get all graph names in the loaded bmodels """
        pass

    def get_addr_mode(self, graph_name: str) -> int:
        """Get address assign mode of a graph
        
        Parameters:
        -----------
        graph_name: str
            The specified graph name.

        Returns:
        -----------
        addr_mode: int
            addr_mode 1 means bmodel will allocate device mem internally, and 0 means not.

        """
        pass

    def get_stage_num(self, graph_name: str) -> int:
        """Get stage num of a graph
        
        Parameters:
        -----------
        graph_name: str
            The specified graph name.

        Returns:
        -----------
        stage_num: int

        """
        pass

    def get_input_num(self, graph_name: str) -> int:
        """Get the number of a graph's input  """
        pass

    def get_output_num(self, graph_name: str) -> int:
        """Get the number of a graph's output  """
        pass

    def get_is_dynamic(self, graph_name: str) -> bool:
        """Get whether the graph is dynamic  """
        pass


    def process(self, graph_name: str, 
        input: dict[int, Tensor], 
        output: dict[int, Tensor],
        core_list: list[int] = [0]) -> int:
        # output: dict[int, Tensor],
        # core_list: list[int] = []) -> int:
        """
        Inference with provided input and output tensors

        Parameters:
        ----------
        graph_name: str
            The specified graph to inference.
        input : dict[int, Tensor]
            Input index and tensor.
        output : dict[int, Tensor]
            Output index and tensor.

        Returns:
        ----------
        return: int
            0 means success and else for failure.
        """
        pass


    def get_input_name(self, graph_name: str, index: int) -> str:
        """Get input tensor name of a specified index in a graph  """
        pass

    def get_input_tensor_devid(self, graph_name: str, index: int) -> int:
        """Get device id of a input tensor in a graph  """
        pass

    def get_output_name(self, graph_name: str, index: int) -> str:
        """Get output tensor name of a specified index in a graph  """
        pass

    def get_output_tensor_devid(self, graph_name: str, index: int) -> int:
        """Get device id of a output tensor in a graph  """
        pass
    
    def get_input_shape(self, graph_name: str, index: int, stage: int = 0) -> list[int]:
        """Get the shape of an input tensor in a graph """
        pass

    def get_input_max_shape(self, graph_name: str, index: int) -> list[int]:
        """Get the max shape of a graph's index-th input """
        pass

    def get_output_shape(self, graph_name: str, index: int, stage: int = 0) -> list[int]:
        """Get the shape of an output tensor in a graph """
        pass
    
    def get_output_max_shape(self, graph_name: str, index: int) -> list[int]:
        """Get the max shape of a graph's index-th output """
        pass

    def get_input_dtype(self, graph_name: str, index: int) -> Dtype:
        """Get data type of an input tensor"""
        pass

    def get_output_dtype(self, graph_name: str, index: int) -> Dtype:
        """Get data type of an output tensor"""
        pass

    def get_input_scale(self, graph_name: str, index: int) -> float:
        """Get scale of an input tensor"""
        pass

    def get_output_scale(self, graph_name: str, index: int) -> float:
        """Get scale of an output tensor"""
        pass

    def get_input_tensors(self, graph_name: str, tensor_name: str, stage: int = 0) -> dict[int,Tensor]: 
        """Get all input Tensors and input index, only for address assign mode is 1 """
        pass

    def get_input_tensors(self, graph_name: str, stage: int = 0) -> dict[int,Tensor]: 
        """Get all input Tensors and input index, only for address assign mode is 1 """
        pass

    def get_input_tensor(self, graph_name: str, index: int, stage: int = 0) -> Tensor: 
        """Get one input Tensor, only for address assign mode is 1 """
        pass

    def create_max_input_tensors(self, graph_name: str) -> dict[int,Tensor]: 
        """Create input Tensors according to max input shape, only for address assign mode 0 """
        pass

    def get_output_tensors(self, graph_name: str, tensor_name: str, stage: int = 0) -> dict[int,Tensor]: 
        """Get all output Tensors and output index, only for address assign mode is 1 """
        pass

    def get_output_tensors(self, graph_name: str, stage: int = 0) -> dict[int,Tensor]: 
        """Get all output Tensors and output index, only for address assign mode is 1 """
        pass

    def get_output_tensor(self, graph_name: str, index: int, stage: int = 0) -> Tensor: 
        """Get one output Tensor, only for address assign mode is 1 """
        pass

    def create_max_output_tensors(self, graph_name: str) -> dict[int,Tensor]: 
        """Create output Tensors according to max output shape, only for address assign mode 0 """
        pass

    def get_input_tensors_addrmode0(self, graph_name: str, stage: int = 0) -> dict[int,Tensor]: 
        """Get all input Tensors and input index, even when addr_mode is 0 """
        pass

    def get_output_tensors_addrmode0(self, graph_name: str, stage: int = 0) -> dict[int,Tensor]: 
        """Get all output Tensors and output index, even when addr_mode is 0 """
        pass