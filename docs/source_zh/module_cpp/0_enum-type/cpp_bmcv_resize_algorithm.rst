bmcv_resize_algorithm
___________________________


定义图像resize中常见的插值策略


**接口形式:**
    .. code-block:: c

        enum bmcv_resize_algorithm_ {
            BMCV_INTER_NEAREST = 0,
            BMCV_INTER_LINEAR  = 1,
            BMCV_INTER_BICUBIC = 2
        } bmcv_resize_algorithm;

**参数说明**

* BMCV_INTER_NEAREST

最近邻插值算法

* BMCV_INTER_LINEAR

双线性插值算法

* BMCV_INTER_BICUBIC

双三次插值算法


