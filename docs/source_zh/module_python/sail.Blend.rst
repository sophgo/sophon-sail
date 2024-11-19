sail.Blend
____________

图片拼接。可实现多张图片进行拼接。仅支持BM1688和CV186AH。


\_\_init\_\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

初始化Blend。

**接口形式:**
    .. code-block:: python

        def __init__(self, src_h: int, ovlap_attr: list[list[int]], bd_attr: list[list[int]], wgt_phy_mem: list[list[string]], wgt_mode: sail.blend_wgt_mode))

**参数说明:**

* src_h: int

输入图片的高度。输入图片的高度需一致。

* ovlap_attr: list[list[int]]

输入图片重叠区域的左右边界。

* bd_attr: list[list[int]]

输入图片的左右黑边属性。目前不支持该选项，设为空即可。

* wgt_phy_mem: list[list[string]]

输入图片的权重文件。参考[:ref:`获取图片的权重文件`]，获取图片权重文件。

* wgt_mode: sail.blend_wgt_mode

输入图片的拼接模式。目前支持以下两种拼接模式，通常选择sail.blend_wgt_mode.BM_STITCH_WGT_YUV_SHARE。

* sail.blend_wgt_mode.BM_STITCH_WGT_YUV_SHARE: YUV share alpha and beta(1 alpha + 1 beta)

* sail.blend_wgt_mode.BM_STITCH_WGT_UV_SHARE: UV share alpha and beta(2 alpha + 2 beta)

process
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

进行图片拼接。

**接口形式1:**
    .. code-block:: python

        def process(input: list[BMImage], output: BMImage) -> int

**返回值说明**

* ret: int

返回图片是否拼接成功，0表示成功，其他表示失败。

**参数说明1:**

* input: list[BMImage]

输入待拼接的BMImage。

* output: BMImage

输出拼接后的BMImage。

**接口形式2:**
    .. code-block:: python

        def process(input: list[BMImage]) -> BMImage

**返回值说明**

* output: BMImage

返回拼接后的BMImage。

**参数说明2:**

* input: list[BMImage]

输入待拼接的BMImage。

**示例代码:**
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
