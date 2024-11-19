Blend
____________

Picture stitching. Multiple pictures can be stitched together. Only BM1688 and CV186AH  support this interface.


Constructor Blend()
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Initialize Blend

**Interface:**
    .. code-block:: c

        Blend(int src_h, std::vector<std::vector<short>> ovlap_attr, std::vector<std::vector<short>> bd_attr, std::vector<std::vector<string>> wgt_phy_mem, bm_stitch_wgt_mode wgt_mode)

**Parameters:**

* src_h: int

Input images. The height of the input images must be consistent.

* ovlap_attr: std::vector<std::vector<short>>

The left and right boundaries of the images overlap area.

* bd_attr: std::vector<std::vector<short>>

The left and right black border properties of the input images. This option is not currently supported, so just leave it blank.

* wgt_phy_mem: std::vector<std::vector<string>>

Weights file for input images. Reference[:ref:`Get weight files of images`]，get image weight files。

* wgt_mode: bm_stitch_wgt_mode

The stitching mode of the input image. Currently the following two stitching modes are supported, usually BM_STITCH_WGT_YUV_SHARE is selected.

* BM_STITCH_WGT_YUV_SHARE: YUV share alpha and beta(1 alpha + 1 beta)

* BM_STITCH_WGT_UV_SHARE: UV share alpha and beta(2 alpha + 2 beta)

process
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Perform image stitching.

**Interface:**
    .. code-block:: c

        int process(std::vector<BMImage*> &input, BMImage &output)

**Returns**

* ret: int

Returns whether the image stitching is successful, 0 means success, other values ​​mean failure.

**Parameters:**

* input: std::vector<BMImage*>

Input BMImage to be stitched.

* output: BMImage

Output stitched BMImage.

**Interface:**
    .. code-block:: c

        BMImage process(std::vector<BMImage*> &input)

**Returns**

* output: BMImage

Return stitched BMImage.

**Parameters:**

* input: std::vector<BMImage*>

Input BMImage to be stitched.

**Sample:**
    .. code-block:: c

        #include "cvwrapper.h"

        int main(){
            Handle handle(0);
            Decoder decoder("./left.jpg", False, 0);
            BMImage image_left;
            decoder.read(handle, image_left);

            Decoder decoder("./right.jpg", False, 0);
            BMImage image_right;
            decoder.read(handle, image_left);

            Blend blend_tmp(2240, {{2112},{2239}}, {}, {{"data/wgt/c01_alpha_444p_m2__0_2240x128.bin","data/wgt/c01_beta_444p_m2__0_2240x128.bin"}}, BM_STITCH_WGT_YUV_SHARE);
            BMImage bmimg = blend_tmp.process({image_left, image_right});
            sail::Bmcv bmcv(handle);
            int ret = bmcv.imwrite("result.jpg", img);
        }
