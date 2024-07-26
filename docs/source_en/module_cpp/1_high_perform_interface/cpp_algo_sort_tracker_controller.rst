sort_tracker_controller
____________________________________________

SORT algorithm to track target

sort_tracker_controller
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: c
          
        sort_tracker_controller(float max_iou_distance = 0.7, int max_age = 30, int n_init = 3);

**Parameters:**

* max_iou_distance: float

Input. Maximum Intersection over Union (IoU) distance threshold used in the tracker.

* max_age: int

Input. The maximum number of frames that a tracked object can exist in the tracker.

* n_init: int

Input. The threshold for the number of initialization frames in the tracker.



process
>>>>>>>>>>>>>

Processing interface.

**Interface:**
    .. code-block:: c

        td::vector<std::tuple<int, int, int, int, int, float, int>> process(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short);

**Parameters:**


* detected_objects: std::vector<std::tuple<left, top, right, bottom, class_id, score>>

Input. Detected objects.


**返回值说明:**

* tracked_objects:  std::vector<std::tuple<left, top, right, bottom, class_id, score, track_id>>

Output。Tracked objects.
