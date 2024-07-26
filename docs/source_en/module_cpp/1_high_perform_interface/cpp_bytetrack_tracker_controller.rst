bytetrack_tracker_controller
____________________________________________

For the ByteTrack algorithm, tracking of targets is achieved by processing the detection results.

Constructor
>>>>>>>>>>>>

**Interface:**
    .. code-block:: c
          
        bytetrack_tracker_controller(int frame_rate = 30, 
                                int track_buffer = 30);

**Parameters:**

* frame_rate: int

Input. Used to control the maximum number of frames allowed to disappear for tracked objects.

* track_buffer: int

Input. Used to control the maximum number of frames allowed to disappear for tracked objects.


process
>>>>>>>>>>>>>

Processing interface.

**Interface:**
    .. code-block:: c

        int process(const vector<DeteObjRect>& detected_objects, 
                vector<TrackObjRect>& tracked_objects);

**Parameters:**

* detected_objects: vector<DeteObjRect>

Input. Detected objects.

* tracked_objects: vector<TrackObjRect>

Output. Tracked objects.

**Returns:**

int

Returns 0 on success and others on failure.