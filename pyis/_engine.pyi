from typing import Any
import numpy
from ._basic import *

class Engine:
    def __init__(self, dev_id: int) : pass
    def __init__(self, handle: Handle) : pass
    def __init__(self, bmodel_path: str, dev_id: int, mode:IOMode) : pass
    def __init__(self, bmodel_bytes: bytes, bmodel_size: int, dev_id: int, mode:IOMode) : pass

    def load(self, bmodel_path: str) -> bool: 
        """ Load bmodel from file """
        pass

    def load(self, bmodel_bytes: bytes, bmodel_size: int) -> bool: 
        """ load bmodel from system memory """
        pass

    def get_handle(self) -> Handle:
        """Get Handle instance """
        pass

    def get_device_id(self) -> int:
        """Get device id of this engine """
        pass

    def get_graph_names(self) -> list[str]:
        """Get all graph names in the loaded bomodels """
        pass

    def set_io_mode(self, graph_name: str, mode: IOMode) -> bool:
        """Set IOMode for a graph """
        pass

    def get_input_names(self, graph_name: str) -> list[str]:
        """Get all input tensor names of the specified graph """
        pass

    def get_output_names(self, graph_name: str) -> list[str]:
        """Get all output tensor names of the specified graph """
        pass
    
    def get_max_input_shapes(self, graph_name: str) -> dict[str, list[int]]:
        """Get max shapes of input tensors in a graph """
        pass

    def get_input_shape(self, graph_name: str, tensor_name: str) -> list[int]:
        """Get the shape of an input tensor in a graph """
        pass

    def get_max_output_shapes(self, graph_name: str) -> dict[str, list[int]]:
        """Get max shapes of output tensors in a graph """
        pass
    
    def get_output_shape(self, graph_name: str, tensor_name: str) -> list[int]:
        """Get the shape of an output tensor in a graph """
        pass
    
    def get_input_dtype(self, graph_name: str, tensor_name: str) -> Dtype:
        """Get data type of an input tensor"""
        pass
    
    def get_output_dtype(self, graph_name: str, tensor_name: str) -> Dtype:
        """Get data type of an output tensor"""
        pass

    def get_input_scale(self, graph_name: str, tensor_name: str) -> float:
        """Get scale of an input tensor"""
        pass

    def get_output_scale(self, graph_name: str, tensor_name: str) -> float:
        """Get scale of an output tensor"""
        pass

    def process(self, graph_name: str, 
        input: dict[str, Tensor], 
        output: dict[str, Tensor],
        core_list: list[int] = []) -> None:
        """
        Inference with provided input and output tensors

        Parameters:
        ----------
        graph_name: str
            The specified graph name.
        input : dict[str, Tensor]
            Input tensors.
        output : dict[str, Tensor]
            Output tensors.
        core_list : list[int]
            Cores used to inference, default is [].
        """
        pass
    
    def process(self, graph_name: str, 
        input: dict[str, Tensor], 
        input_shapes: dict[str, list[int]],
        output: dict[str, Tensor],
        core_list: list[int] = []) -> None:
        """
        Inference with provided input and output tensors and input shapes.
        
        Parameters:
        ----------
        graph_name: str
            The specified graph name.
        input : dict[str, Tensor]
            Input tensors.
        input_shapes : dict[str, list[int]]  
            Real input tensor shapes.
        output : dict[str, Tensor]
            Output tensors.
        core_list : list[int]
            Cores used to inference, default is [].
        """
        pass
    def process(self, graph_name: str,
        input_tensors: dict[str, numpy.ndarray[Any, numpy.dtype[numpy.float_]]],
        core_list: list[int] = []) -> dict[str, numpy.ndarray[Any, numpy.dtype[numpy.float_]]] :
        """
        Inference with provided input.

        Parameters:
        ----------
        graph_name : str
            The specified graph name.
        input_tensors: dict[str,ndarray]
            Input tensors.
        core_list : list[int]
            Cores used to inference, default is [].

        Returns
        -------
        dict[str,ndarray]
        """
        pass
    
    def create_input_tensors_map(self, graph_name: str, create_mode: int = -1) -> dict[str,Tensor]:
        """
        Create input tensors map, according to and bmodel.

        Parameters:
        ----------
        graph_name : str
            The specified graph name.
        create_mode: Tensor Create mode
            case 0: only allocate system memory 
            case 1: only allocate device memory
            case other: according to engine IOMode

        Returns
        -------
        dict[str,Tensor]
        """
        pass

    def create_output_tensors_map(self, graph_name: str, create_mode: int = -1) -> dict[str,Tensor]:
        """
        Create output tensors map, according to and bmodel.

        Parameters:
        ----------
        graph_name : str
            The specified graph name.
        create_mode: Tensor Create mode
            case 0: only allocate system memory 
            case 1: only allocate device memory
            case other: according to engine IOMode

        Returns
        -------
        dict[str,Tensor]
        """
        pass  


class MultiEngine:
    def __init__(self, bmodel_path: str, dev_ids: list[int], sys_out: bool = True, graph_idx: int = 0) : pass

    def set_print_flag(self, flag:bool) -> None :pass

    def set_print_time(self, flag:bool) -> None :pass
    
    def get_device_ids(self) -> list[int]:
        """Get device ids of this MultiEngine """
        pass

    def get_graph_names(self) -> list[str]:
        """Get all graph names in the loaded bomodels """
        pass  

    def get_input_names(self, graph_name: str) -> list[str]:
        """Get all input tensor names of the specified graph """
        pass

    def get_output_names(self, graph_name: str) -> list[str]:
        """Get all output tensor names of the specified graph """
        pass

    def get_input_shape(self, graph_name: str, tensor_name: str) -> list[int]:
        """Get the shape of an input tensor in a graph """
        pass

    def get_output_shape(self, graph_name: str, tensor_name: str) -> list[int]:
        """Get the shape of an output tensor in a graph """
        pass

    def process(self,
        input_tensors: dict[str, numpy.ndarray[Any, numpy.dtype[numpy.float_]]]) -> dict[str, numpy.ndarray[Any, numpy.dtype[numpy.float_]]] :
        """
        Inference with provided input.

        Parameters:
        ----------
        input_tensors: dict[str,ndarray]
            Input tensors.

        Returns
        -------
        dict[str,ndarray]
        """
        pass    
