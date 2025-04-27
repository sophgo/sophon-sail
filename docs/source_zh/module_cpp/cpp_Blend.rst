Blend
____________

图片拼接。可实现多张图片进行拼接。仅支持BM1688和CV186AH，SoC模式。


构造函数Blend()
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

初始化Blend。

**接口形式:**
    .. code-block:: c

        Blend(int src_h, std::vector<std::vector<short>> ovlap_attr, std::vector<std::vector<short>> bd_attr, std::vector<std::vector<string>> wgt_phy_mem, bm_stitch_wgt_mode wgt_mode)

**参数说明:**

* src_h: int

输入图片的高度。输入图片的高度需一致。

* ovlap_attr: std::vector<std::vector<short>>

输入图片重叠区域的左右边界。

* bd_attr: std::vector<std::vector<short>>

输入图片的左右黑边属性。目前不支持该选项，设为空即可。

* wgt_phy_mem: std::vector<std::vector<string>>

输入图片的权重文件。参考[:ref:`获取图片的权重文件`]，获取图片权重文件。

* wgt_mode: bm_stitch_wgt_mode

输入图片的拼接模式。目前支持以下两种拼接模式，通常选择BM_STITCH_WGT_YUV_SHARE。

* BM_STITCH_WGT_YUV_SHARE: YUV share alpha and beta(1 alpha + 1 beta)

* BM_STITCH_WGT_UV_SHARE: UV share alpha and beta(2 alpha + 2 beta)

process
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

判断源文件是否打开。

**接口形式1:**
    .. code-block:: c

        int process(std::vector<BMImage*> &input, BMImage &output)

**返回值说明**

* ret: int

返回图片是否拼接成功，0表示成功，其他表示失败。

**参数说明1:**

* input: std::vector<BMImage*>

输入待拼接的BMImage。

* output: BMImage

输出拼接后的BMImage。

**接口形式2:**
    .. code-block:: c

        BMImage process(std::vector<BMImage*> &input)

**返回值说明**

* output: BMImage

返回拼接后的BMImage。

**参数说明2:**

* input: std::vector<BMImage*>

输入待拼接的BMImage。

**示例代码:**
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


