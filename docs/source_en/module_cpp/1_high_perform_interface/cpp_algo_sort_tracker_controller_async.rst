sort_tracker_controller_async
____________________________________________

SORT algorithm asynchronous processing interface

Constructor
>>>>>>>>>>>>>>>

**interface:**
    .. code-block:: c
          
        sort_tracker_controller(float max_iou_distance = 0.7, int max_age = 30, int n_init = 3,int input_queue_size = 10,int result_queue_size = 10);

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
>>>>>>>>>>>>>

Asynchronous processing interface, pushes input parameters to the internal task queue, used in conjunction with get_result

**interface :**
    .. code-block:: c

        int push_data(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short);

**Parameters :**

* detected_objects_short: std::vector<std::tuple<left, top, right, bottom, class_id, score>>

Input parameter. Detected object frame.

**Returns :**

int

Returns 0 on success and others on failure.


get_result
>>>>>>>>>>>>>

Asynchronous processing interface, obtains the information of the target to be tracked, and is used in conjunction with push_data.

**Interface:**
    .. code-block:: c

        std::vector<std::tuple<int, int, int, int, int, float, int>> get_result();

**Parameters:**

* vector<std::tuple<left, top, right, bottom, class_id, score, track_id>>

Output parameters. The object being tracked.