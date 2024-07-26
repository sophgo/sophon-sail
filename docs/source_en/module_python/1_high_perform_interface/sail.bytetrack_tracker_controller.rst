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

        def process(detected_objects:list[list[int, float, float, float, float, float, float, float]], 
                tracked_objects:list[list[int, float, float, float, float, float, float, float, int]]) 
                    -> int

**Parameters:**

* detected_objects: list[list[int, float, float, float, float, float, float, float]]

Input. Detected objects.

* tracked_objects: list[list[int, float, float, float, float, float, float, float, int]]

Output。Tracked objects.

**Returns:**

int

A return value of 0 indicates success, while other values indicate failure.