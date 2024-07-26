from typing import Any
import numpy
from ._basic import *

class algo_yolov5_post_1output:
    
    def __init__(self, shape: list[int], network_w:int = 640, network_h:int = 640, max_queue_size: int=20, input_use_multiclass_nms: bool=True, agnostic: bool=False): 
        """ algo_yolov5_post_1output Constructor.

        Parameters:
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

        Parameters:
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

        Parameters:
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


class bytetrack_tracker_controller:
    def __init__(self, frame_rate:int = 30, track_buffer:int = 30):

        """bytetrack Constructor

        Parameters:
        ----------
        frame_rate: int, max_lost_time = int(frame_rate / 30.0 * track_buffer)
        track_buffer: int, max_lost_time = int(frame_rate / 30.0 * track_buffer)
        
        """

    def process(detected_objects:list[tuple[int, int, int, int, int, float]])-> list[tuple[int, int, int, int, int, float, int]]:
        
        """process interface

        Parameters:
        ----------
        detected_objects: list(tuple(left, top, right, bottom, class_id, score)), detected box list
        

        Returns:
        tracked_objects: list(tuple(left, top, right, bottom, class_id, score, track_id)), tracked box list output
        

        """

class deepsort_tracker_controller:
    def __init__(self, max_cosine_distance:float, nn_budget:int, k_feature_dim:int, max_iou_distance:float=0.7, max_age:int=30, n_init:int=3):

        """deepsort Constructor

        Parameters:
        ----------
        max_cosine_distance:float, 
        nn_budget:int, 
        k_feature_dim:int, 
        max_iou_distance:float=0.7, 
        max_age:int=30, 
        n_init:int=3,
        
        
        """

    def process(self, detected_objects:list[tuple[int, int, int, int, int, float]], features:list[Tensor])-> list[tuple[int, int, int, int, int, float, int]]:
        
        """process interface

        Parameters:
        ----------
        detected_objects: list(tuple(left, top, right, bottom, class_id, score)), detected box list
        
        features: list[sail.Tensor], features extracted by extractor

        Returns:
        tracked_objects: list(tuple(left, top, right, bottom, class_id, score, track_id)), tracked box list output
        

        """
    
    def process(self, detected_objects:list[tuple[int, int, int, int, int, float]], features:numpy.array)-> list[tuple[int, int, int, int, int, float, int]]:
        
        """process interface

        Parameters:
        ----------
        detected_objects: list(tuple(left, top, right, bottom, class_id, score)), detected box list
        
        features: numpy.array, features extracted by extractor

        Returns:
        tracked_objects: list(tuple(left, top, right, bottom, class_id, score, track_id)), tracked box list output
        

        """

class sort_tracker_controller:
    def __init__(self,max_iou_distance:float=0.7, max_age:int=30, n_init:int=3):

        """sort Constructor

        Parameters:
        ----------
        max_iou_distance:float=0.7, 
        max_age:int=30, 
        n_init:int=3,
        
        
        """

    def process(self, detected_objects:list[tuple[int, int, int, int, int, float]])-> list[tuple[int, int, int, int, int, float, int]]:
        
        """process interface

        Parameters:
        ----------
        detected_objects: list(tuple(left, top, right, bottom, class_id, score)), detected box list

        Returns:
        tracked_objects: list(tuple(left, top, right, bottom, class_id, score, track_id)), tracked box list output
        

        """

def nms_rotated(boxes: numpy.ndarray[Any, numpy.dtype[numpy.float32]], 
                scores: numpy.ndarray[Any, numpy.dtype[numpy.float32]], 
                threshold: float)-> list[int]:
    
    """nms with rotated boxes
        
    Parameters:
    ----------   
    boxes: numpy array (float), [x,y,w,h,theta]
    scores: numpy array (float), score of each box
    threshold: float, IOU threshold
    
    Returns:
    keep_id: list (int), the id of original boxes that kept after filtering

    
    """