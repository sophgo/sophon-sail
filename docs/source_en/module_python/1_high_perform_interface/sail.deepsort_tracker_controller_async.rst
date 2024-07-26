deepsort_tracker_controller
____________________________________________

For the DeepSORT algorithm, tracking of targets is achieved by processing the detection results and extracting features.

\_\_init\_\_
>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: python
          
        def __init__(max_cosine_distance:float, 
                nn_budget:int, 
                k_feature_dim:int, 
                max_iou_distance:float = 0.7, 
                max_age:int = 30, 
                n_init:int = 3,
                queue_size:int = 10)


**Parameters:**

* max_cosine_distance: float

Input. Maximum threshold for cosine distance used in similarity calculation.

* nn_budget: int

Input. Maximum number for nearest neighbor search.

* k_feature_dim: int

Input. The feature dimension of the detected objects.

* max_iou_distance: float

Input. Maximum Intersection over Union (IoU) distance threshold used in the tracker.

* max_age: int

Input. The maximum number of frames that a tracked object can exist in the tracker.

* n_init: int

Input. The threshold for the number of initialization frames in the tracker.

* queue_size: int

Input. Buffer size of the result queue.


push_data
>>>>>>>>>>>>>>>

**Interface 1:**
    .. code-block:: python
          
        def push_data(detected_objects:list[tuple[int, int, int, int, int, float]], 
                      feature:sail.Tensor) -> int



**Parameters:**

* detected_objects: list(tuple(left, top, right, bottom, class_id, score))

Input. Detected objects.

* feature:sail.Tensor

Input. The features of the detected objects.


**Returns:**

* int

Returns 0 on success and others on failure.

**Interface 2:**
    .. code-block:: python
          
        def push_data(detected_objects:list[tuple[int, int, int, int, int, float, int]], 
                      feature:list[numpy.array]) -> int



**Parameters:**

* detected_objects: list(tuple(left, top, right, bottom, class_id, score))

Input. Detected objects.

* feature: list[numpy.array]

Input. The features of the detected objects.


**Returns:**

* int

Returns 0 on success and others on failure.



get_result_npy
>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: python
          
        def get_result_npy() -> tracked_objects:list[list[int, int, int, int, int, float, int]]


**Returns:**

* tracked_objects: list(tuple(left, top, right, bottom, class_id, score, track_id))

Output。Tracked objects.