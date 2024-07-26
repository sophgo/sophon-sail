bmcv_resize_algorithm
___________________________


The interpolation strategies in image resize


**Interface:**
    .. code-block:: c

        enum bmcv_resize_algorithm_ {
            BMCV_INTER_NEAREST = 0,
            BMCV_INTER_LINEAR  = 1,
            BMCV_INTER_BICUBIC = 2
        } bmcv_resize_algorithm;

**Parameters**

* BMCV_INTER_NEAREST

Using nearest neighbor interpolation method while resizing.

* BMCV_INTER_LINEAR

Using linear interpolation method while resizing.

* BMCV_INTER_BICUBIC

Using double cubic interpolation method while resizing.


