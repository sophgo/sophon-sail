"""
SAIL Kernel API
===============

Provides
  1. Based on tpu kernel 

"""

from typing import Any
from ._basic import *

class tpu_kernel_api_yolov5_detect_out:
    
    def __init__(self, device_id: int, shapes: list[list[int]], network_w:int = 640, network_h:int = 640, 
                module_file: str='/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so'): 

        """ tpu_kernel_api_yolov5_detect_out Constructor.

        Parameters:
        ----------
        device_id: int, Device id
        shapes: list[list[int]],Input Data shape
        network_w: int, Network input width 
        network_h: int, Network input height 
        module_file: str, TPU Kernel module file,default is '/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so'
        """
        pass


    def process(self,input:list[TensorPTRWithName], dete_threshold: float, nms_threshold: float, release_input: bool = False) -> list[list[tuple[int, int, int, int, int, float]]]:
        """ Process

        Parameters:
        ----------
        input: list[TensorPTRWithName], Input Data 
        dete_threshold: float, Detection threshold
        nms_threshold: float, NMS threshold
        release_input: bool, Release input memory

        Returns
        -------
        list[list[tuple[left, top, right, bottom, class_id, score]]]
            
        """

    def process(self,input:dict[str, Tensor], dete_threshold: float, nms_threshold: float, release_input: bool = False) -> list[list[tuple[int, int, int, int, int, float]]]:
        """ Process

        Parameters:
        ----------
        input: dict[str, Tensor], Input Data 
        dete_threshold: float, Detection threshold
        nms_threshold: float, NMS threshold
        release_input: bool, Release input memory

        Returns
        -------
        list[list[tuple[left, top, right, bottom, class_id, score]]]
            
        """
                
    def reset_anchors(self, anchors_new: list[list[list[int]]]):

        """ Reset Anchors
        
        Parameters:
        ----------
        anchors_new: list[list[list[int]]]
        
        Returns
        -------
        return 0 for success and other for failure
        """
        pass

class tpu_kernel_api_openpose_part_nms:
    
    def __init__(self, device_id: int, network_c: int,  
                module_file: str='/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so'): 

        """ tpu_kernel_api_openpose_part_nms Constructor.

        Parameters:
        ----------
        device_id: int, Device id
        network_c: int, Pose input channel 
        module_file: str, TPU Kernel module file,default is '/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so'
        """
        pass


    def process(self, input_data: TensorPTRWithName, shape: list[int], threshold: list[float], max_peak_num: list[int]) -> tuple[list[list[int]], list[list[float]], list[list[int]]]:
        """ Process

        Parameters:
        ----------
        input_data: TensorPTRWithName, Input Data 
        shape: list[int], Input Data width and height
        threshold: list[float], Detection threshold
        max_peak_num: list[int], Peak Detection maxium num

        Returns
        -------
        tuple[list[list[int]], list[list[float]], list[list[int]]]
            
        """

    def process(self, input: dict[str, Tensor], shape: list[int], threshold: list[float], max_peak_num: list[int]) -> tuple[list[list[int]], list[list[float]], list[list[int]]]:
        """ Process

        Parameters:
        ----------
        input: dict[str, Tensor], Input Data 
        shape: list[int], Input Data width and height
        threshold: list[float], Detection threshold
        max_peak_num: list[int], Peak Detection maxium num

        Returns
        -------
        tuple[list[list[int]], list[list[float]], list[list[int]]]
            
        """
                
    def reset_network_c(self, network_c_new: int):

        """ Reset channel_num
        
        Parameters:
        ----------
        network_c_new: int
        
        Returns
        -------
        return 0 for success and other for failure
        """
        pass