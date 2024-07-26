sail.nms_rotated
_________________________

nms with rotating boxes

sail.nms_rotated
>>>>>>>>>>>>>>>>>>>

get the retained index of rotating boxes according to the threshold

**Interface:**
    .. code-block:: python

        def nms_rotated(boxes: numpy.ndarray[numpy.float32], scores: numpy.ndarray[numpy.float32], threshold: float)-> list[int]:


**Parameters:**

* boxes: numpy.ndarray[numpy.float32]

all rotating boxes, shape is (N,5), box is represented by [x,y,w,h,theta]

* scores: numpy.ndarray[numpy.float32]

the confidence corresponding to the rotating boxes, shape is (N,)

* threshold: float

IOU threshold

**Returns:**

the retained index of rotating boxes according to the threshold

**Sample:**
    .. code-block:: python
      
        import sophon.sail as sail
        import numpy as np

        boxes = np.load("boxes_data.npy")
        scores = np.load("scores_data.npy")
        threshold = 0.3

        indices = sail.nms_rotated(boxes, scores, threshold)

        # show all the boxes that remained
        print(boxes[indices])



