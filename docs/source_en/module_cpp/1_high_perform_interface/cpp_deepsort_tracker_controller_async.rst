deepsort_tracker_controller_async
____________________________________________

DeepSORT algorithm asynchronous processing interface

Constructor
>>>>>>>>>>>>>>>

**interface:**
    .. code-block:: c
          
        deepsort_tracker_controller(float max_cosine_distance, 
                                int nn_budget, 
                                int k_feature_dim, 
                                float max_iou_distance = 0.7, 
                                int max_age = 30, 
                                int n_init = 3,
                                int queue_size = 10);

**Parameters:**

* max_cosine_distance: float

Input parameter. As the maximum cosine distance threshold when calculating similarity.

* nn_budget: int

Input parameter. Maximum number limit used for nearest neighbor searches.

* k_feature_dim: int

Input parameter. Feature dimensions of the detected target.

* max_iou_distance: float

Input parameter. The IoU distance threshold used in trackers.

* max_age: int

Input parameter. The maximum number of frames that the tracking target will exist in the tracker.

* n_init: int

Input parameter. Initialization frame threshold in the tracker.

* queue_size: int

Input parameter. The length of the result buffer queue.

push_data
>>>>>>>>>>>>>

Asynchronous processing interface, pushes input parameters to the internal task queue, used in conjunction with get_result

**interface 1:**
    .. code-block:: c

        int push_data(const vector<DeteObjRect>& detected_objects, 
                      vector<Tensor>& feature);

**Parameters 1:**

* detected_objects: vector<DeteObjRect>

Input parameter. Detected object frame.

* feature: vector<Tensor>

Input parameter. The characteristics of detected objects.


**Returns 1:**

int

Returns 0 on success and others on failure.


**interface 2:**
    .. code-block:: c

        int push_data(const vector<DeteObjRect>& detected_objects, 
                      vector<vector<float>>& feature);

**Parameters 2:**

* detected_objects: vector<DeteObjRect>

Input parameter. The detected object frame.

* feature: vector<float>

Input parameter.The characteristics of detected objects.


**Return 2:**

int

Returns 0 on success and others on failure.

get_result
>>>>>>>>>>>>>

Asynchronous processing interface, obtains the information of the target to be tracked, and is used in conjunction with push_data.

**Interface:**
    .. code-block:: c

        vector<TrackObjRect> get_result();

**Parameters:**

* vector<TrackObjRect>

Output parameters. The object being tracked.