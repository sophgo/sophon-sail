sort_tracker_controller_async
____________________________________________

SORT algorithm to track target

\_\_init\_\_
>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: python
          
        def __init__(max_iou_distance:float = 0.7, 
                max_age:int = 30, 
                n_init:int = 3,
                input_queue_size:int = 10,
                result_queue_size:int = 10)


**Parameters:**

* max_iou_distance: float

Input. Maximum Intersection over Union (IoU) distance threshold used in the tracker.

* max_age: int

Input. The maximum number of frames that a tracked object can exist in the tracker.

* n_init: int

Input. The threshold for the number of initialization frames in the tracker.

* input_queue_size: int

Input. Buffer size of the input queue.

* result_queue_size: int

Input. Buffer size of the result queue.


push_data
>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: python
          
        def push_data(detected_objects:list[tuple[int, int, int, int, int, float]]) -> int



**Parameters:**

* detected_objects: list(tuple(left, top, right, bottom, class_id, score))

Input. Detected objects.

**Returns:**

* int

Returns 0 on success and others on failure.


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