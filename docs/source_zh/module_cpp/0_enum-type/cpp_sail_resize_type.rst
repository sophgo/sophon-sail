sail_resize_type
______________________

图像预处理对应的预处理方法。

**接口形式:**

.. code-block:: c

        enum sail_resize_type {
            BM_RESIZE_VPP_NEAREST = 0,
            BM_RESIZE_TPU_NEAREST = 1,
            BM_RESIZE_TPU_LINEAR = 2,
            BM_RESIZE_TPU_BICUBIC = 3,
            BM_PADDING_VPP_NEAREST = 4,
            BM_PADDING_TPU_NEAREST = 5,
            BM_PADDING_TPU_LINEAR = 6,
            BM_PADDING_TPU_BICUBIC = 7
        };
        
**参数说明:**

* BM_RESIZE_VPP_NEAREST

使用VPP，最近邻的方法进行图像尺度变换。

* BM_RESIZE_TPU_NEAREST

使用智能视觉深度学习处理器，最近邻的方法进行图像尺度变换。

* BM_RESIZE_TPU_LINEAR

使用智能视觉深度学习处理器，线性插值的方法进行图像尺度变换。

* BM_RESIZE_TPU_BICUBIC

使用智能视觉深度学习处理器，双三次插值的方法进行图像尺度变换。

* BM_PADDING_VPP_NEAREST

使用VPP，最近邻的方法进行带padding的图像尺度变换。

* BM_PADDING_TPU_NEAREST

使用智能视觉深度学习处理器，最近邻的方法进行带padding的图像尺度变换。

* BM_PADDING_TPU_LINEAR

使用智能视觉深度学习处理器，线性插值的方法进行带padding的图像尺度变换。

* BM_PADDING_TPU_BICUBIC

使用智能视觉深度学习处理器，双三次插值的方法进行带padding的图像尺度变换。
