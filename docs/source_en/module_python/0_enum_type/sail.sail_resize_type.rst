sail.sail_resize_type
______________________

图像预处理对应的预处理方法。

**接口形式:**
.. code-block:: python

        sail_resize_type.BM_RESIZE_VPP_NEAREST
        sail_resize_type.BM_RESIZE_TPU_NEAREST
        sail_resize_type.BM_RESIZE_TPU_LINEAR
        sail_resize_type.BM_RESIZE_TPU_BICUBIC
        sail_resize_type.BM_PADDING_VPP_NEAREST
        sail_resize_type.BM_PADDING_TPU_NEAREST
        sail_resize_type.BM_PADDING_TPU_LINEAR
        sail_resize_type.BM_PADDING_TPU_BICUBIC

        
**参数说明:**

* BM_RESIZE_VPP_NEAREST

Use VPP, the nearest neighbor method for image scaling

* BM_RESIZE_TPU_NEAREST

Use Tensor Computing Processor, the nearest neighbor method for image scaling

* BM_RESIZE_TPU_LINEAR

Use Tensor Computing Processor, the linear interpolation method for image scaling.

* BM_RESIZE_TPU_BICUBIC

Use Tensor Computing Processor, double cubic interpolation method for image scaling

* BM_PADDING_VPP_NEAREST

Use VPP, the nearest neighbor method for image scaling with padding

* BM_PADDING_TPU_NEAREST

Use Tensor Computing Processor, the nearest neighbor method for image scaling with padding

* BM_PADDING_TPU_LINEAR

Use Tensor Computing Processor, the linear interpolation method for image scaling with padding

* BM_PADDING_TPU_BICUBIC

Use Tensor Computing Processor, double cubic interpolation method for image scaling with padding
