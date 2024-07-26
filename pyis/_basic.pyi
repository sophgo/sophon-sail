"""
SAIL basic
===============

Provides
  1. Basic data types, mainly including Handle and Tensor 
  2. Some common basic interfaces that do not rely on sophon multimedia

"""

import enum
from typing import Any,Tuple
import numpy

class Dtype(enum.Enum):
    BM_FLOAT32 = 0
    BM_FLOAT16 = 1
    BM_INT8 = 2
    BM_UINT8 = 3
    BM_INT16 = 4
    BM_UINT16 = 5
    BM_INT32 = 6
    BM_UINT32 = 7
    BM_BFLOAT16 = 8

class IOMode(enum.Enum):
    SYSI = 0
    SYSO = 1
    SYSIO = 2
    DEVIO = 3

class LogLevel(enum.Enum):
    """
    The Enumeration class of log level, including TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL, OFF.
    """
    TRACE       = 0
    DEBUG       = 1
    INFO        = 2
    WARN        = 3
    ERROR       = 4
    CRITICAL    = 5
    OFF         = 6

class Handle:
    def __init__(self, dev_id: int) -> Handle:
        """ 
        Constructor with device id.

        Parameters:
        ----------
        dev_id: int
           ID of TPU.
        """
        pass

    def get_device_id(self) -> int:
        """ Get device id of this handle. """
        pass

    def get_sn(self) -> str:
        """ Get Serial Number. """
        pass

    def get_target(self) -> str:
        """ Get TPU chip type of this handle. """
        pass

class Tensor:
    def __init__(self, handle: Handle, shape: list[int], 
        dtype: Dtype = Dtype.BM_FLOAT32,
        own_sys_data: bool = False,
        own_dev_data: bool = False) -> Tensor: pass
        
    def __init__(self, handle: Handle, data: numpy.ndarray[Any,numpy.dtype[Any]], own_sys_data: bool = True, own_dev_data: bool = True) -> Tensor: 
        """
        Constructor allocates device memory of the tensor use numpy.ndarray, \n
        Input numpy must be C_CONTIGUOUS True

        Parameters:
        ----------
        handle: Handle
        data: numpy.ndarray
            dtype is float_ | uint8 | int32 | int_, C_CONTIGUOUS flag must be True.
        own_sys_data: bool, default is True.
            Indicator of whether own system memory, If false, the memory will be copied to device directly  
        own_dev_data: bool, default is True.
        """
        pass
    
    def __init__(self, other: Tensor, ranges: list[Tuple[int,int]] | list[tuple[int,int],tuple[int,int]], d2d_flag = True):
        """
        Create a tensor from specific range of another tensor.
        like the slice function in numpy.
        Only supports tensor with at most 2D.

        Parameters:
        ----------
        other: Tensor
            other 2D or 1D tensor. 
        ranges: list(tuple())
            a list of tuple
            if you want to slice a tensor like tensor[20:40, 50:60] in numpy
            you can give the para ranges = [(20,40), (50,60)]
        
        d2d_flag: bool
            true for d2d
            false for c2c

        """
        pass
  
    def shape(self) -> list[int]: pass
 
    def dtype(self) -> int: pass
 
    def reshape(self, shape: list[int]) -> None: pass

    def own_sys_data(self) -> bool: pass

    def own_dev_data(self) -> bool: pass


    def asnumpy(self, shape: list[int] = None) -> numpy.ndarray[Any,numpy.dtype[Any]] : 
        """
        Get ndarray in system memory of the tensor.

        Parameters:
        ----------
        shape: list[int], optional
            If provided, Shape of output data. 
        """
        pass

    def update_data(self, data: numpy.ndarray[Any, numpy.dtype[numpy.any]]) -> None: 
        """
        If own_sys_data, Update system data of the tensor. \n
        Else if own_dev_data Update device data of the tensor. 

        Parameters:
        ----------
        data: numpy.ndarray
            dtype is float_ | uint8 | int32 | int_.
        """
        pass

    def scale_from(self, data: numpy.ndarray[Any, numpy.dtype[numpy.any]], scale: float) -> None:
        """ 
        Scale data to tensor in system memory

        Parameters:
        ----------
        data : ndarray
            Data of type float32 to be scaled from. 
        scale : float
            Scale value. 
        """
        pass

    def scale_to(self, scale: float, shape: list[int] = None) -> numpy.ndarray[Any, numpy.dtype[Any]] :
        """ 
        Scale tensor to data in system memory.

        Parameters:
        ----------
        scale : float
            Scale value. 
        shape : list[int], optional
            If provided, Shape of output data to scale to. 

        """
        pass

    def sync_s2d(self, size: int = None) -> None:
        """ 
        move size data from system to device

        Parameters:
        ----------
        size : int, optional
            If provided, byte size to be copied.
            else, move all data from system to device.
        """
        pass

    def sync_s2d(self, src: Tensor, offset_src: int, offset_dst: int, len: int):
        """ 
        Copy data from system memory to device memory with specified size.

        Parameters:
        ----------
        src: Tensor, Specifies the Tensor to be copied from.
        offset_src: int, Specifies the number of elements to offset in the source Tensor from where to start copying.
        offset_dst: int, Specifies the number of elements to offset in the destination Tensor from where to start copying.
        len: int, Specifies the length of the copy, i.e., the number of elements to copy.
        """
        pass

    def sync_d2s(self, size: int = None) : 
        """ 
        move size data from device to system

        Parameters:
        ----------
        size : int, optional
            If provided, byte size to be copied.
            else, move size data from device to system
        """
        pass

    def sync_d2s(self, src: Tensor, offset_src: int, offset_dst: int, len: int):
        """ 
        Copy data from device memory to system memory with specified size.

        Parameters:
        ----------
        src: Tensor, Specifies the Tensor to be copied from.
        offset_src: int, Specifies the number of elements to offset in the source Tensor from where to start copying.
        offset_dst: int, Specifies the number of elements to offset in the destination Tensor from where to start copying.
        len: int, Specifies the length of the copy, i.e., the number of elements to copy.
        """
        pass

    def sync_d2d(self, src: Tensor, offset_src: int, offset_dst: int, len: int):
        """ 
        Copy data from device memory to device memory with specified size.

        Parameters:
        ----------
        src: Tensor, Specifies the Tensor to be copied from.
        offset_src: int, Specifies the number of elements to offset in the source Tensor from where to start copying.
        offset_dst: int, Specifies the number of elements to offset in the destination Tensor from where to start copying.
        len: int, Specifies the length of the copy, i.e., the number of elements to copy.
        """
        pass
    
    def pysys_data(self) ->  numpy.ndarray[Any, numpy.dtype[numpy.int32]] :   pass

    def memory_set(self, c: any) -> None : 
        """
        Fill memory with a scalar, it will be automatically converted to tensor's dtype.

        Parameters:
        ----------
        c: int
        """
        pass
    
    def zeros(self) -> None:
        """
        Fill memory with zeros
        """

    def ones(self) -> None:
        """
        Fill memory with ones
        """

    def is_dev_data_valid(self) -> bool:
        """
        Check whether device data is valid
        """

    def is_sys_data_valid(self) -> bool:
        """
        Check whether system data is valid
        """

    def device_id(self) -> int:
        """
        Get device id.
        """

    def dump_data(self, file_name: str, bin:bool = False) -> None:
        """
        Dump Tensor data to file.

        Parameters:
        ----------
        file_name: str
            file path to dump tensor
        
        bin: bool
            binary format, default False.
        """
        pass

class TensorPTRWithName:
    def get_name(self) -> str:
        """ Get the name of the Tensor.

        pass
        """

    def get_data(self) -> Tensor:
        """ Get the Tensor.
        
        pass
        """

def ReleaseTensorPtr(tensor_with_name: TensorPTRWithName) -> None:
    """ Release TensorPTRWithName data.

    pass
    """

def CreateTensorPTRWithName(name: str, handle: Handle, data: numpy.ndarray[Any,numpy.dtype[Any]], own_sys_data: bool = False, own_dev_data:bool = True) -> TensorPTRWithName:
    """ Create TensorPTRWithName use name and numpy array.

    Parameters:
    ----------
    name: input numpy array name
    handle: Handle with result
    data: input numpy array
    own_sys_data: bool, default is False.
        Indicator of whether own system memory, If false, the memory will be copied to device directly  
    own_dev_data: bool, default is True.
    Returns
    -------
    TensorPTRWithName
    """

def get_available_tpu_num() -> int:
    """
    Get the number of available TPUs.

    Returns
    -------
    Number of available TPUs.
    """
    pass

def set_print_flag(print_flag: bool) -> None:
    """ Print main process time use  """
    pass

def set_dump_io_flag(dump_io_flag: bool) -> None:
    """ Dump io data"""
    pass

def argmax(tensor:Tensor) -> int:
    """ 
    Get the indices of the maximum value

    Parameters:
    ----------
    tensor : Tensor, input tensor.

    Returns
    -------
    int, the indices of the maximum value.
    """ 
    pass
  

def argmin(tensor:Tensor) -> int:
    """ 
    Get the indices of the minimum value

    Parameters:
    ----------
    tensor : Tensor, input tensor.

    Returns
    -------
    int, the indices of the minimum value.
    """ 
    
def set_loglevel(loglevel: LogLevel) -> int:
    """
    Set the level of output log.

    Parameters:
    ----------
    loglevel: LogLevel
        The optional level is TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL, OFF. The default level is INFO. 

    Returns:
    ----------
        int, 0 for success and -1 for failure.
    """
    pass