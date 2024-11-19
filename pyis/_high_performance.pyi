from typing import Any
import numpy
from ._basic import *
from ._multimedia import *

class EngineImagePreProcess:
    
    def __init__(self, bmodel_path: str, tpu_id: int, use_mat_output: bool = False, core_list:list = []): 
        """ EngineImagePreProcess Constructor.

        Parameters
        ----------
        bmodel_path: str
        tpu_id: int
            ID of TPU, there may be more than one TPU for PCIE mode,default is 0.
        core_list: list
            CORE of choosed TPU,default is None.
        use_mat_output : bool
            Use opencv Mat for output images
        """
        pass

    def InitImagePreProcess(self, resize_mode: sail_resize_type, bgr2rgb: bool = False,
        queue_in_size: int = 20, queue_out_size: int = 20) -> int:
        """ initialize ImagePreProcess.

        Parameters
        ----------
        resize_mode: sail_resize_type
            Resize Methods
        bgr2rgb: bool
            The flag of convert BGR image to RGB, default is False.
        queue_in_size: int
            Max input image data queue size, default is 20.
        queue_out_size: int
            Max output tensor data queue size, default is 20.
        
        Returns
        -------
        0 for success and other for failure 
        """
        pass


    def SetPaddingAtrr(self,padding_b:int=114,padding_g:int=114,padding_r:int=114,align:int=0) -> int :
        """ Set the padding attribute object.

        Parameters
        ----------
        padding_b: int
            padding value of b channel, dafault 114
        padding_g: int
            padding value of g channel, dafault 114
        padding_r: int
            padding value of r channel, dafault 114
        align: int
            padding position, default 0: start left top, 1 for center
        
        Returns
        -------
        0 for success and other for failure 
        """

    def SetConvertAtrr(self,alpha_beta) -> int :
        """ Set the linear transformation attribute 

        Parameters
        ----------
        alpha_beta:like (a0, b0), (a1, b1), (a2, b2) factors

        Returns
        -------
        0 for success and other for failure 
        """
        pass

    def PushImage(self, channel_idx : int, image_idx : int, image: BMImage) -> int: 
        """ Push Image
        Parameters
        ----------
        channel_idx : int
            Channel index number of the image
        image_idx: int
            Image index number of the image
        image: BMImage
            Input image
        
        Returns
        -------
        0 for success and other for failure 
        """
        pass


    def GetBatchData_Npy(self) -> tuple:
        """ Get the Batch Data object
        
        Returns
        -------
        [dict[str, ndarray], list[BMImage],list[int],list[int],list[list[int]]]
            Output ndarray map, Original Images, Original Channel index, Original Indexm, Padding Atrr(start_x, start_y, width, height)
        """
        pass

    def GetBatchData_Npy2(self) -> tuple:
        """ Get the Batch Data object
        
        Returns
        -------
        [dict[str, ndarray], list[numpy.ndarray[numpy.uint8]],list[int],list[int],list[list[int]]]
            Output ndarray map, Original Images, Original Channel index, Original Indexm, Padding Atrr(start_x, start_y, width, height)
        """
        pass

    def GetBatchData(self, need_d2s: bool = True) -> tuple:
        """ Get the Batch Data object
        
        Parameters
        ------------
        need_d2s : bool
            Need copy data to system memory.

        Returns
        -------
        [list[TensorPTRWithName], list[BMImage],list[int],list[int],list[list[int]]]
            Output ndarray map, Original Images, Original Channel index, Original Indexm, Padding Atrr(start_x, start_y, width, height)
        """
        pass

    def get_graph_name(self) -> str:
        """ Get first graph name in the loaded bomodel
        
        Returns
        -------
        First graph name
        """
        pass

    def get_input_width(self) -> int:
        """ Get model input width

        Returns
        -------
        Model input width
        """
        pass

    def get_input_height(self) -> int:
        """ Get model input height

        Returns
        -------
        Model input height
        """
        pass

    def get_output_names(self) -> list[str]:
        """ Get all output tensor names of the first graph

        Returns
        -------
        All the output tensor names of the graph
        """
        pass
    
    def get_output_shape(self, tensor_name: str) -> list[int]:
        """ Get the shape of an output tensor in frist graph
        
        Parameters
        ----------
        tensor_name : str
             The specified tensor name

        Returns
        -------
        The shape of the tensor
        """
        pass

class ImagePreProcess:
    
    def __init__(self, batch_size: int, resize_mode:sail_resize_type, tpu_id: int=0, 
        queue_in_size: int=20, queue_out_size: int=20):  
        """ ImagePreProcess Constructor.

        Parameters
        ----------
        batch_size: int
            Output batch size.
        resize_mode: sail_resize_type
            Resize Methods
        tpu_id: int
            ID of TPU, there may be more than one TPU for PCIE mode,default is 0.
        queue_in_size: int
            Max input image data queue size, default is 20.
        queue_out_size: int
            Max output tensor data queue size, default is 20.
        """
        pass


    def SetResizeImageAtrr(self, output_width: int, output_height: int ,bgr2rgb : bool, dtype: ImgDtype) -> None : 

        """ Set the Resize Image attribute 
        
        Parameters
        ----------
        output_width: int
            The width of resized image.
        output_height: int
            The height of resized image.
        bgr2rgb: bool
            The flag of convert BGR image to RGB.
        dtype: ImgDtype  
            The data type of resized image,Only supported BM_FLOAT32,BM_INT8,BM_UINT8   
        """
        pass


    def SetPaddingAtrr(self,padding_b:int=114,padding_g:int=114,padding_r:int=114,align:int=0) -> None :
        """ Set the padding attribute object.

        Parameters
        ----------
        padding_b: int
            padding value of b channel, dafault 114
        padding_g: int
            padding value of g channel, dafault 114
        padding_r: int
            padding value of r channel, dafault 114
        align: int
            padding position, default 0: start left top, 1 for center

        """
        pass


    def SetConvertAtrr(self,alpha_beta) -> int :
        """ Set the linear transformation attribute 

        Parameters
        ----------
        alpha_beta:like (a0, b0), (a1, b1), (a2, b2) factors

        Returns
        -------
        0 for success and other for failure 
        """
        pass


    def PushImage(self, channel_idx : int, image_idx : int, image: BMImage) -> int: 
        """ Push Image
        Parameters
        ----------
        channel_idx : int
            Channel index number of the image
        image_idx: int
            Image index number of the image
        image: BMImage
            Input image
        
        Returns
        -------
        0 for success and other for failure 
        """
        pass

    def GetBatchData(self) -> tuple[Tensor, list[BMImage],list[int],list[int],list[list[int]]]:
        """ Get the Batch Data object
        
        Returns
        -------
        [Tensor, list[BMImage],list[int],list[int],list[list[int]]]
            Output Tensor, Original Images, Original Channel index, Original Index, Padding Atrr(start_x, start_y, width, height)
        """
        pass

    def set_print_flag(self, flag : bool) -> None:
        """ Set the print flag
        
        Parameters
        ----------
        flag : int
        """
        pass

class algo_yolov5_post_1output:
    
    def __init__(self, shape: list[int], network_w:int = 640, network_h:int = 640, max_queue_size: int=20, input_use_multiclass_nms: bool=True, agnostic: bool=False): 
        
        """ algo_yolov5_post_1output Constructor.

        Parameters
        ----------
        shape: list[int]
        network_w: int, network input width 
        network_h: int, network input height 
        max_queue_size: int, default 20
        input_use_multiclass_nms: bool, default True
        agnostic: bool, default False
        """
        pass

    def push_npy(self, 
                channel_idx : int, 
                image_idx : int, 
                data : numpy.ndarray[Any, numpy.dtype[numpy.float_]], 
                dete_threshold : float, 
                nms_threshold : float,
                ost_w : int, 
                ost_h : int,
                padding_left : int,
                padding_top : int,
                padding_width : int,
                padding_height : int) -> int:
        """ algo_yolov5_post_1output push data.

        Parameters
        ----------
        
        channel_idx : Channel index number of the image.
        image_idx : Image index number of the image.
        data : Input Data
        dete_threshold : Detection threshold
        nms_threshold : NMS threshold
        ost_w : Original image width
        ost_h : Original image height
        padding_left : Padding left
        padding_top : Padding top
        padding_width : Padding width
        padding_height : Padding height

        Returns
        -------
        return 0 for success and other for failure
        """
        pass

    def get_result_npy(self) -> tuple[tuple[int, int, int, int, int, float],int, int]:
        """ Get the PostProcess result
        
        Returns
        -------
        tuple[tuple[left, top, right, bottom, class_id, score],channel_idx, image_idx]
            
        """
        pass


    def push_data( self, 
        channel_idx : list[int], 
        image_idx : list[int], 
        input_data : TensorPTRWithName, 
        dete_threshold : list[float],
        nms_threshold : list[float],
        ost_w : list[int],
        ost_h : list[int],
        padding_attr : list[list[int]]) -> int:
        """ algo_yolov5_post_1output push data.

        Parameters
        ----------
        
        channel_idx : Channel index number of the images.
        image_idx : Image index number of the images.
        data : Input Data
        dete_threshold : Detection thresholds
        nms_threshold : NMS thresholds
        ost_w : Original images width
        ost_h : Original images height
        padding_attr : Padding Attribute[start_x, start_y, width, height]

        Returns
        -------
        return 0 for success and other for failure
        """
        pass

class algo_yolov8_post_1output_async:
    
    def __init__(self, shape: list[int], network_w:int = 640, network_h:int = 640, max_queue_size: int=20, input_use_multiclass_nms: bool=True, agnostic: bool=False): 
        """ algo_yolov8_post_1output_async Constructor.

        Parameters
        ----------
        shape: list[int]
        network_w: int, network input width 
        network_h: int, network input height 
        max_queue_size: int, default 20
        input_use_multiclass_nms: bool, default True
        agnostic: bool, default False
        """
        pass

    def push_npy(self, 
                channel_idx : int, 
                image_idx : int, 
                data : numpy.ndarray[Any, numpy.dtype[numpy.float_]], 
                dete_threshold : float, 
                nms_threshold : float,
                ost_w : int, 
                ost_h : int,
                padding_left : int,
                padding_top : int,
                padding_width : int,
                padding_height : int) -> int:
        """ algo_yolov8_post_1output_async push data.

        Parameters
        ----------
        
        channel_idx : Channel index number of the image.
        image_idx : Image index number of the image.
        data : Input Data
        dete_threshold : Detection threshold
        nms_threshold : NMS threshold
        ost_w : Original image width
        ost_h : Original image height
        padding_left : Padding left
        padding_top : Padding top
        padding_width : Padding width
        padding_height : Padding height

        Returns
        -------
        return 0 for success and other for failure
        """
        pass

    def get_result_npy(self) -> tuple[tuple[int, int, int, int, int, float],int, int]:
        """ Get the PostProcess result
        
        Returns
        -------
        tuple[tuple[left, top, right, bottom, class_id, score],channel_idx, image_idx]
            
        """
        pass


    def push_data( self, 
        channel_idx : list[int], 
        image_idx : list[int], 
        input_data : TensorPTRWithName, 
        dete_threshold : list[float],
        nms_threshold : list[float],
        ost_w : list[int],
        ost_h : list[int],
        padding_attr : list[list[int]]) -> int:
        """ algo_yolov8_post_1output_async push data.

        Parameters
        ----------
        
        channel_idx : Channel index number of the images.
        image_idx : Image index number of the images.
        data : Input Data
        dete_threshold : Detection thresholds
        nms_threshold : NMS thresholds
        ost_w : Original images width
        ost_h : Original images height
        padding_attr : Padding Attribute[start_x, start_y, width, height]

        Returns
        -------
        return 0 for success and other for failure
        """
        pass
    
class algo_yolov8_post_cpu_opt_1output_async:
    
    def __init__(self, shape: list[int], network_w:int = 640, network_h:int = 640, max_queue_size: int=20, input_use_multiclass_nms: bool=True, agnostic: bool=False): 
        """ algo_yolov8_post_cpu_opt_1output_async Constructor.

        Parameters
        ----------
        shape: list[int]
        network_w: int, network input width 
        network_h: int, network input height 
        max_queue_size: int, default 20
        input_use_multiclass_nms: bool, default True
        agnostic: bool, default False
        """
        pass

    def push_npy(self, 
                channel_idx : int, 
                image_idx : int, 
                data : numpy.ndarray[Any, numpy.dtype[numpy.float_]], 
                dete_threshold : float, 
                nms_threshold : float,
                ost_w : int, 
                ost_h : int,
                padding_left : int,
                padding_top : int,
                padding_width : int,
                padding_height : int) -> int:
        """ algo_yolov8_post_cpu_opt_1output_async push data.

        Parameters
        ----------
        
        channel_idx : Channel index number of the image.
        image_idx : Image index number of the image.
        data : Input Data
        dete_threshold : Detection threshold
        nms_threshold : NMS threshold
        ost_w : Original image width
        ost_h : Original image height
        padding_left : Padding left
        padding_top : Padding top
        padding_width : Padding width
        padding_height : Padding height

        Returns
        -------
        return 0 for success and other for failure
        """
        pass

    def get_result_npy(self) -> tuple[tuple[int, int, int, int, int, float],int, int]:
        """ Get the PostProcess result
        
        Returns
        -------
        tuple[tuple[left, top, right, bottom, class_id, score],channel_idx, image_idx]
            
        """
        pass


    def push_data( self, 
        channel_idx : list[int], 
        image_idx : list[int], 
        input_data : TensorPTRWithName, 
        dete_threshold : list[float],
        nms_threshold : list[float],
        ost_w : list[int],
        ost_h : list[int],
        padding_attr : list[list[int]]) -> int:
        """ algo_yolov8_post_cpu_opt_1output_async push data.

        Parameters
        ----------
        
        channel_idx : Channel index number of the images.
        image_idx : Image index number of the images.
        data : Input Data
        dete_threshold : Detection thresholds
        nms_threshold : NMS thresholds
        ost_w : Original images width
        ost_h : Original images height
        padding_attr : Padding Attribute[start_x, start_y, width, height]

        Returns
        -------
        return 0 for success and other for failure
        """
        pass
    
class algo_yolov5_post_3output:
    
    def __init__(self, shape: list[list[int]], network_w:int = 640, network_h:int = 640, max_queue_size: int=20, input_use_multiclass_nms: bool=True, agnostic: bool=False): 
        """ algo_yolov5_post_1output Constructor.

        Parameters
        ----------
        shape: list[list[int]]
        network_w: int, network input width 
        network_h: int, network input height 
        max_queue_size: int, default 20
        input_use_multiclass_nms: bool, default True
        agnostic: bool, default False
        """
        pass
    
    def reset_anchors(self, anchors_new: list[list[list[int]]]):
        """ Reset Anchors
        
        Parameters
        ----------
        anchors_new: list[list[list[int]]]
        
        Returns
        -------
        return 0 for success and other for failure
        """
        pass
    
    def get_result_npy(self) -> tuple[tuple[int, int, int, int, int, float],int, int]:
        """ Get the PostProcess result
        
        Returns
        -------
        tuple[tuple[left, top, right, bottom, class_id, score],channel_idx, image_idx]
            
        """
        pass


    def push_data( self, 
        channel_idx : list[int], 
        image_idx : list[int], 
        input_data : list[TensorPTRWithName], 
        dete_threshold : list[float],
        nms_threshold : list[float],
        ost_w : list[int],
        ost_h : list[int],
        padding_attr : list[list[int]]) -> int:
        """ algo_yolov5_post_1output push data.

        Parameters
        ----------
        
        channel_idx : Channel index number of the images.
        image_idx : Image index number of the images.
        data : Input Data
        dete_threshold : Detection thresholds
        nms_threshold : NMS thresholds
        ost_w : Original images width
        ost_h : Original images height
        padding_attr : Padding Attribute[start_x, start_y, width, height]

        Returns
        -------
        return 0 for success and other for failure
        """
        pass

class algo_yolov5_post_cpu_opt_async:
    
    def __init__(self, shape: list[list[int]], network_w:int = 640, network_h:int = 640, max_queue_size: int=20, use_multiclass_nms: bool=True): 
        """ algo_yolov5_post_cpu_opt_async Constructor.

        Parameters
        ----------
        shape: list[list[int]]
        network_w: int, network input width 
        network_h: int, network input height 
        max_queue_size: int, default 20
        use_multiclass_nms: bool, whether to use multi-class NMS, default True
        """
        pass
    
    def reset_anchors(self, anchors_new: list[list[list[int]]]):
        """ Reset Anchors
        
        Parameters
        ----------
        anchors_new: list[list[list[int]]]
        
        Returns
        -------
        return 0 for success and other for failure
        """
        pass
    
    def get_result_npy(self) -> tuple[tuple[int, int, int, int, int, float],int, int]:
        """ Get the PostProcess result
        
        Returns
        -------
        tuple[tuple[left, top, right, bottom, class_id, score],channel_idx, image_idx]
            
        """
        pass


    def push_data( self, 
        channel_idx : list[int], 
        image_idx : list[int], 
        input_data : list[TensorPTRWithName], 
        dete_threshold : list[float],
        nms_threshold : list[float],
        ost_w : list[int],
        ost_h : list[int],
        padding_attr : list[list[int]]) -> int:
        """ algo_yolov5_post_cpu_opt_async push data.

        Parameters
        ----------
        
        channel_idx : Channel index number of the images.
        image_idx : Image index number of the images.
        input_data : Input Data
        dete_threshold : Detection thresholds
        nms_threshold : NMS thresholds
        ost_w : Original images width
        ost_h : Original images height
        padding_attr : Padding Attribute[start_x, start_y, width, height]

        Returns
        -------
        return 0 for success and other for failure
        """
        pass

class algo_yolov5_post_cpu_opt:
    
    def __init__(self, shapes: list[list[int]], network_w:int = 640, network_h:int = 640): 

        """ algo_yolov5_post_cpu_opt Constructor.

        Parameters
        ----------
        shapes: list[list[int]],Input Data shape
        network_w: int, Network input width 
        network_h: int, Network input height 
        """
        pass


    def process(self,input_data:list[TensorPTRWithName], ost_w: list[int], ost_h: list[int], dete_threshold: list[float], nms_threshold: list[float], input_keep_aspect_ratio: bool, input_use_multiclass_nms: bool) -> list[list[tuple[int, int, int, int, int, float]]]:
        """ Process

        Parameters
        ----------
        input_data: list[TensorPTRWithName], Input Data 
        ost_w: list[int], Original image width
        ost_h: list[int], Original image height
        dete_threshold: list[float], Detection threshold
        nms_threshold: list[float], NMS threshold
        input_keep_aspect_ratio: bool, Input keeping aspect ratio  
        input_use_multiclass_nms: bool, Input with multiclass

        Returns
        -------
        list[list[tuple[left, top, right, bottom, class_id, score]]]
            
        """
        pass

    def process(self,input:dict[str, Tensor], ost_w: list[int], ost_h: list[int], dete_threshold: list[float], nms_threshold: list[float], input_keep_aspect_ratio: bool, input_use_multiclass_nms: bool) -> list[list[tuple[int, int, int, int, int, float]]]:
        """ Process

        Parameters
        ----------
        input: dict[str, Tensor], Input Data 
        ost_w: list[int], Original image width
        ost_h: list[int], Original image height
        dete_threshold: list[float], Detection threshold
        nms_threshold: list[float], NMS threshold
        input_keep_aspect_ratio: bool, Input keeping aspect ratio  
        input_use_multiclass_nms: bool, Input with multiclass

        Returns
        -------
        list[list[tuple[left, top, right, bottom, class_id, score]]]
            
        """
        pass
                
    def reset_anchors(self, anchors_new: list[list[list[int]]]):

        """ Reset Anchors
        
        Parameters
        ----------
        anchors_new: list[list[list[int]]]
        
        Returns
        -------
        return 0 for success and other for failure
        """
        pass

class algo_yolov8_seg_post_tpu_opt:
    
    def __init__(self, bmodel_file: str, dev_id: int, detection_shape: list[int], segmentation_shape: list[int], network_h: int, network_w: int): 

        """ algo_yolov8_seg_post_tpu_opt Constructor.

        Parameters
        ----------
        bmodel_file: str, The TPU getmask bmodel path
        dev_id: int, device id
        detection_shape: list[int], The shapes of detection head
        segmentation_shape: list[int], The shapes of segmentation head, that is Prototype Mask
        network_h: int, The input height of yolov8 network
        network_w: int, The input width of yolov8 network
        """
        pass
    
    def process(self, detection_input: TensorPTRWithName, segmentation_input: TensorPTRWithName, ost_h: int, ost_w: int, dete_threshold: float, nms_threshold: float, input_keep_aspect_ratio: bool, input_use_multiclass_nms: bool) -> list[tuple[int, int, int, int, float, int, list[float], numpy.ndarray]]:
        """ Process

        Parameters
        ----------
        detection_input: TensorPTRWithName, The input data of detection head
        segmentation_input: TensorPTRWithName, The input data of segmentation head, that is Prototype Mask
        ost_h: int, Original image height
        ost_w: int, Original image width
        dete_threshold: float, Detection threshold
        nms_threshold: float, NMS threshold
        input_keep_aspect_ratio: bool, Input keeping aspect ratio  
        input_use_multiclass_nms: bool, Input with multiclass

        Returns
        -------
        list[tuple[left, top, right, bottom, score, class_id, contour, mask]]
            
        """
        pass
    
    def process(self, detection_input: dict[str, Tensor], segmentation_input: dict[str, Tensor], ost_h: int, ost_w: int, dete_threshold: float, nms_threshold: float, input_keep_aspect_ratio: bool, input_use_multiclass_nms: bool) -> list[tuple[int, int, int, int, float, int, list[float], numpy.ndarray]]:
        """ Process

        Parameters
        ----------
        detection_input: dict[str, Tensor], The input data of detection head
        segmentation_input: dict[str, Tensor], The input data of segmentation head, that is Prototype Mask
        ost_h: int, Original image height
        ost_w: int, Original image width
        dete_threshold: float, Detection threshold
        nms_threshold: float, NMS threshold
        input_keep_aspect_ratio: bool, Input keeping aspect ratio  
        input_use_multiclass_nms: bool, Input with multiclass

        Returns
        -------
        list[tuple[left, top, right, bottom, score, class_id, contour, mask]]
            
        """
        pass

class deepsort_tracker_controller_async:
    def __init__(self, max_cosine_distance:float, nn_budget:int, k_feature_dim:int, max_iou_distance:float=0.7, max_age:int=30, n_init:int=3, queue_size:int=10):

        """asynchronous deepsort Constructor

        Parameters
        ----------
        max_cosine_distance:float, 
        nn_budget:int, 
        k_feature_dim:int, 
        max_iou_distance:float=0.7, 
        max_age:int=30, 
        n_init:int=3,
        queue_size:int=10
        """

    def push_data(self, detected_objects:list[tuple[int, int, int, int, int, float]], feature:Tensor) -> int:
        """ Asynchronous processing interface. Use with get_result_npy()!
            Track the objects based on the detected objects and their features
        
        Parameters
        ----------
        detected_objects: list(tuple(left, top, right, bottom, class_id, score))
        feature: sail.Tensor, the features of the detected objects
        
        Returns
        -------
        int:Returns 0 on success and others on failure.
        """
    
    def push_data(self, detected_objects:list[tuple[int, int, int, int, int, float]], feature:list[numpy.array]) -> int:
        """ Asynchronous processing interface. Use with get_result_npy()!
            Track the objects based on the detected objects and their features
        
        Parameters
        ----------
        detected_objects: list(tuple(left, top, right, bottom, class_id, score))
        feature: list[numpy.array], the features of the detected objects
        
        Returns
        -------
        int:Returns 0 on success and others on failure.
        """

    def get_result_npy(self) -> list[tuple[int, int, int, int, int, float, int]]:
        """ Asynchronous processing interface. Use with push_data()!
            Track the objects based on the detected objects and their features
        
        Returns
        -------
        tracked_objects: list(tuple(left, top, right, bottom, class_id, score, track_id))
        """
    def set_processing_timer(flag: bool):
        """ set the flag whether printing time cost of each processing 

        Parameters
        ----------
        flag: bool, if True, print the time cost in each processing
                    if False, nothing happens
        
        """

class sort_tracker_controller_async:
    def __init__(self, max_iou_distance:float=0.7, max_age:int=30, n_init:int=3, input_queue_size:int=10, result_queue_size:int=10):

        """asynchronous sort Constructor

        Parameters
        ----------
        max_iou_distance:float=0.7, 
        max_age:int=30, 
        n_init:int=3,
        input_queue_size:int=10,
        result_queue_size:int=10
        """

    def push_data(self, detected_objects:list[tuple[int, int, int, int, int, float]]) -> int:
        """ Asynchronous processing interface. Use with get_result_npy()!
            Track the objects based on the detected objects and their features
        
        Parameters
        ----------
        detected_objects: list(tuple(left, top, right, bottom, class_id, score))
        
        Returns
        -------
        int:Returns 0 on success and others on failure.
        """

    def get_result_npy(self) -> list[tuple[int, int, int, int, int, float, int]]:
        """ Asynchronous processing interface. Use with push_data()!
            Track the objects based on the detected objects and their features
        
        Returns
        -------
        tracked_objects: list(tuple(left, top, right, bottom, class_id, score, track_id))
        """


class Perf:
    def __init__(self, bmodel_path:str, tpu_id:list[int], max_que_size:int, mode:IOMode=IOMode.SYSO, thread_count:int=2, free_input:bool=False):

        """Perf Constructor

        Parameters
        ----------
        bmodel_path:float, Path to bmodel.\n
        tpu_id:list[int], ID of TPUs, there may be more than one TPU for PCIE mode.\n
        max_que_size:int, max queue size.\n
        mode:IOMode, Specify the input/output tensors are in system memory or device memory, default SYSO.\n
        thread_count:int, thread counts with each tpu.\n
        free_input:bool,  release memory of input, default false.\n
        """

    def PushTensor(self, tensor_index:int, input_tensors:list[TensorPTRWithName]) -> int:
        """ Push Tensors 
        
        Parameters
        ----------
        tensor_index: int, index number of the Tensors.
        input_tensors: list[TensorPTRWithName], Input tensors with name
        
        Returns
        -------
        int:Returns 0 on success and others on failure.
        """
    
    def SetEnd(self) -> int:
        """ Stop Push Tensors
        
        Returns
        -------
        int:Returns 0 on success and others on failure.
        """

    def GetResult(self) -> tuple[int, list[TensorPTRWithName]]:
        """ Get the Result Data object
        
        Returns
        -------
        tensor_index: int, Original input data index
        output_data: list(TensorPTRWithName), Output result
        """

    def get_input_dtype(self, graph_name: str, tensor_name: str) -> Dtype:
        """Get data type of an input tensor"""
        pass

    def get_input_scale(self, graph_name: str, tensor_name: str) -> float:
        """Get scale of an input tensor"""
        pass

    def get_input_shape(self, graph_name: str, tensor_name: str) -> list[int]:
        """Get the shape of an input tensor in a graph """
        pass

    def get_output_names(self, graph_name: str) -> list[str]:
        """Get all output tensor names of the specified graph """
        pass

    def get_input_names(self, graph_name: str) -> list[str]:
        """Get all input tensor names of the specified graph """
        pass

    def get_graph_names(self) -> list[str]:
        """Get all graph names in the loaded bomodels """
        pass

class DecoderImages:
    def __init__(self, image_list:list[str], tpu_id:int, queue_size:int):
        """ DecoderImages Constructor

        Parameters
        -----------
        image_list:list[str], image name list\n
        tpu_id:int,TPU ID. You can use bm-smi to see available IDs.\n
        queue_size:int, max queue size\n
        """
        pass

    def setResizeAttr(self, width:int, height:int, resize_alg:bmcv_resize_algorithm = bmcv_resize_algorithm.BMCV_INTER_LINEAR) -> int:
        """ Set the Resize Attr object
        Parameters
        -----------
        width:int,output width\n
        height:int,output height\n
        resize_alg:bmcv_resize_algorithm, Resize algorithm, defalut BMCV_INTER_LINEAR

        Returns
        -----------
        int:Returns 0 on success, other for failed.
        """
        pass

    def start(self) -> int:
        """ Start read images

        Returns
        -----------
        int:Returns 0 on success, other for failed.
        """
        pass


    def read(self, image: BMImage) -> int:
        """ read

        Parameters
        -----------
        image: Output BMImage.

        Returns
        -------
        int:Returns 0 on success, 1 for queue empty, -1 for end or thread stoped.
        """  
        pass

    def stop(self) -> None:
        """ stop 
        """
        pass

    def get_schedule(self) -> int:
        """ Get the schedule object

        Returns
        ---------
        The number of decoded images.
        """
        pass
