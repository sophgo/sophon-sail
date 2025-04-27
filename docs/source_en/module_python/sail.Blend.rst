sail.Blend
____________

Picture stitching. Multiple pictures can be stitched together. Only BM1688 and CV186AH SoC mode support this interface.


\_\_init\_\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Initialize Blend.

**Interface:**
    .. code-block:: python

        def __init__(self, src_h: int, ovlap_attr: list[list[int]], bd_attr: list[list[int]], wgt_phy_mem: list[list[string]], wgt_mode: sail.blend_wgt_mode))

**Parameters:**

* src_h: int

Input images. The height of the input images must be consistent.

* ovlap_attr: list[list[int]]

The left and right boundaries of the images overlap area.

* bd_attr: list[list[int]]

The left and right black border properties of the input images. This option is not currently supported, so just leave it blank.

* wgt_phy_mem: list[list[string]]

Weights file for input images. Reference[:ref:`Get weight files of images`]，get image weight files。

* wgt_mode: sail.blend_wgt_mode

The stitching mode of the input image. Currently the following two stitching modes are supported, usually sail.blend_wgt_mode.BM_STITCH_WGT_YUV_SHARE is selected.

* sail.blend_wgt_mode.BM_STITCH_WGT_YUV_SHARE: YUV share alpha and beta(1 alpha + 1 beta)

* sail.blend_wgt_mode.BM_STITCH_WGT_UV_SHARE: UV share alpha and beta(2 alpha + 2 beta)

process
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Perform image stitching.

**Interface:**
    .. code-block:: python

        def process(input: list[BMImage], output: BMImage) -> int

**Returns**

* ret: int

Returns whether the image stitching is successful, 0 means success, other values ​​mean failure.

**Parameters:**

* input: list[BMImage]

Input BMImage to be stitched.

* output: BMImage

Output stitched BMImage.

**Interface:**
    .. code-block:: python

        def process(input: list[BMImage]) -> BMImage

**Returns**

* output: BMImage

Return stitched BMImage.

**Parameters:**

* input: list[BMImage]

Input BMImage to be stitched.

**Sample:**
    .. code-block:: python

        import sophon.sail  as sail
        if __name__ == '__main__':
            handle = sail.Handle(0)
            decoder = sail.Decoder("./left.jpg", False, 0)
            image_left = sail.BMImage()
            ret = decoder.read(handle, image_left)

            decoder = sail.Decoder("./right.jpg", False, 0)
            image_right = sail.BMImage()
            ret = decoder.read(handle, image_right)

            blend_obj = sail.Blend(2240, [[2112],[2239]], [], [["data/wgt/c01_alpha_444p_m2__0_2240x128.bin","data/wgt/c01_beta_444p_m2__0_2240x128.bin"]], sail.blend_wgt_mode.BM_STITCH_WGT_YUV_SHARE)
            img = blend_obj.process([image_left,image_right])

            sail.Bmcv(handle).imwrite("result.jpg",img)