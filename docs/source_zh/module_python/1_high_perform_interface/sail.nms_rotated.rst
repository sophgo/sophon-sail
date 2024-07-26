sail.nms_rotated
_________________________

旋转nms

sail.nms_rotated
>>>>>>>>>>>>>>>>>>>

根据阈值获取旋转框的保留索引

**接口形式:**
    .. code-block:: python

        def nms_rotated(boxes: numpy.ndarray[numpy.float32], scores: numpy.ndarray[numpy.float32], threshold: float)-> list[int]:


**参数说明:**

* boxes: numpy.ndarray[numpy.float32]

所有旋转框，shape是(N,5)，每个框是[x,y,w,h,theta]

* scores: numpy.ndarray[numpy.float32]

所有旋转框对应的置信度，shape是(N,)

* threshold: float

IOU阈值

**返回值说明:**

返回保留的旋转框索引。

**示例代码:**
    .. code-block:: python
      
        import sophon.sail as sail
        import numpy as np

        boxes = np.load("boxes_data.npy")
        scores = np.load("scores_data.npy")
        threshold = 0.3

        indices = sail.nms_rotated(boxes, scores, threshold)

        # show all the boxes that remained
        print(boxes[indices])



