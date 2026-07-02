bytetrack_tracker_controller
____________________________________________

For the ByteTrack algorithm, tracking of targets is achieved by processing the detection results.

\_\_init\_\_
>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: python
          
        def __init__(frame_rate:int = 30, 
                track_buffer:int = 30)


**Parameters:**

* frame_rate: int

Input. Used to control the maximum number of frames allowed to disappear for tracked objects.

* track_buffer: int

Input. Used to control the maximum number of frames allowed to disappear for tracked objects.



process
>>>>>>>>>>>>>

Processing interface.

**Interface:**
    .. code-block:: python

        def process(detected_objects:list[tuple[int, int, int, int, int, float]]) 
                    -> tracked_objects:list[tuple[int, int, int, int, int, float, int]]

**Parameters:**

* detected_objects: list[tuple(left, top, right, bottom, class_id, score)]

Input. Detected objects.

* tracked_objects: list[tuple(left, top, right, bottom, class_id, score, track_id)]

Output. Tracked objects. The track_id is the ID of the tracked object.

**Returns:**

int

A return value of 0 indicates success, while other values indicate failure.