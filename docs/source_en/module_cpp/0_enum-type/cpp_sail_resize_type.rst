sail_resize_type
______________________

Preprocessing method corresponding to image preprocessing.

**Interface:**

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
        
**Paramters:**

* BM_RESIZE_VPP_NEAREST

Use VPP to perform image scale transformation using the nearest neighbor method.

* BM_RESIZE_TPU_NEAREST

Use Tensor Computing Processor to perform image scale transformation using the nearest neighbor method.

* BM_RESIZE_TPU_LINEAR

Use Tensor Computing Processor to perform image scale transformation using linear interpolation.

* BM_RESIZE_TPU_BICUBIC

Use Tensor Computing Processor to perform image scale transformation using bicubic interpolation.

* BM_PADDING_VPP_NEAREST

Use VPP to perform image scale transformation with padding using the nearest neighbor method.

* BM_PADDING_TPU_NEAREST

Use Tensor Computing Processor to perform image scale transformation with padding using the nearest neighbor method.

* BM_PADDING_TPU_LINEAR

Use Tensor Computing Processor to perform image scale transformation with padding using linear interpolation.

* BM_PADDING_TPU_BICUBIC

Use Tensor Computing Processor to perform image scale transformation with padding using bicubic interpolation method.
