sail.Bmcv
_________

Bmcv封装了常用的图像处理接口, 支持硬件加速。

**实现接口硬件说明**

本文档中, 在BM1684/BM1684X上实现的接口, 负责其实现的硬件单元可能有不同情况(i.e. crop_and_resize在BM1684上由VPP+智能视觉深度学习处理器实现)。影响实现的硬件单元的因素如下:

1. 输入图片/输出图片数量不大于16
2. 图片无需按照原图比例进行缩放
3. 输入图片/输出图片不为以下数据格式: DATA_TYPE_EXT_1N_BYTE_SIGNED DATA_TYPE_EXT_4N_BYTE DATA_TYPE_EXT_FLOAT32
4. 输入的图片device memory不在DDR0上
5. 输入图片格式不为FORMAT_YUV422P, 输出图片格式不为FORMAT_NV12或FORMAT_COMPRESSED, 并且输入图片格式-输出图片格式不为以下组合:

+-----------------------+-----------------------------------------------+
|  input_format         |  output_format                                |
+=======================+===============================================+
|  FORMAT_RGBP_SEPARATE |  FORMAT_ARGB_PACKED FORMAT_ABGR_PACKED        |
+-----------------------+-----------------------------------------------+
|  FORMAT_BGRP_SEPARATE |  FORMAT_ARGB_PACKED FORMAT_ABGR_PACKED        |
+-----------------------+-----------------------------------------------+
|  FORMAT_GRAY          |  FORMAT_YUV420P FORMAT_YUV422P FORMAT_YUV444P |
+-----------------------+-----------------------------------------------+
|  FORMAT_YUV420P       |  FORMAT_GRAY FORMAT_YUV422P FORMAT_YUV444P    |
+-----------------------+-----------------------------------------------+
|  FORMAT_YUV444P       |  FORMAT_GRAY                                  |
+-----------------------+-----------------------------------------------+
|  FORMAT_COMPRESSED    |  FORMAT_GRAY FORMAT_YUV422P FORMAT_YUV444P    |
+-----------------------+-----------------------------------------------+

当且仅当满足上述5个条件时, 接口实现硬件为"VPP+智能视觉深度学习处理器"的Bmcv接口会使用VPP; 否则, 将会使用智能视觉深度学习处理器，
智能视觉深度学习处理器仅支持最邻近插值（Nearest Interpolitan），如不满足上述使用VPP的条件且缩放算法为其它插值策略，将会报错。

\_\_init\_\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

初始化Bmcv

**接口形式:**
    .. code-block:: python

        def __init__(self, handle: sail.Handle)
          

**参数说明:**

* handle: sail.Handle

指定Bmcv使用的设备句柄。


bm_image_to_tensor
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

将BMImage/BMImageArray转换为Tensor。

**接口形式1:**
    .. code-block:: python

        def bm_image_to_tensor(self, image: sail.BMImage) -> sail.Tensor
           

**参数说明1:**

* image: sail.BMImage

需要转换的图像数据。

**返回值说明1:**

* tensor: sail.Tensor

返回转换后的Tensor。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            tensor = bmcv.bm_image_to_tensor(BMimg)# here is a sail.Tensor

**接口形式2:**
    .. code-block:: python

        def bm_image_to_tensor(self, 
                image: sail.BMImageArray, 
                tensor) -> None
           
            
**参数说明2:**

* image: sail.BMImageArray

输入参数。需要转换的图像数据。

* tensor: sail.Tensor

输出参数。转换后的Tensor。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            tensor = sail.Tensor(handle,(1920,1080),sail.Dtype.BM_FLOAT32,True,True)
            bmcv.bm_image_to_tensor(BMimg,tensor)
          
tensor_to_bm_image
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

将Tensor转换为BMImage/BMImageArray。

**接口形式1:**
    .. code-block:: python

        def tensor_to_bm_image(self, 
                tensor: sail.Tensor, 
                bgr2rgb: bool=False,
                layout: str = 'nchw') -> sail.BMImage


**参数说明1:**

* tensor: sail.Tensor

输入参数。待转换的Tensor。

* bgr2rgb: bool, default: False

输入参数。是否进行图像的通道变换。

* layout: str, default: 'nchw'

输入参数。输入Tensor的布局。

**返回值说明1:**

* image : sail.BMImage

返回转换后的图像。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            tensor = bmcv.bm_image_to_tensor(BMimg)# here is a sail.Tensor
            BMimg2 = bmcv.tensor_to_bm_image(tensor)

**接口形式2:**
    .. code-block:: python

        def tensor_to_bm_image(self, 
                tensor: sail.Tensor, 
                format: sail.Format) -> sail.BMImage


**参数说明2:**

* tensor: sail.Tensor

输入参数。待转换的Tensor。

* format: sail.Format

输入参数。BMImage的像素格式。

**返回值说明2:**

* image : sail.BMImage

返回转换后的图像。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            tensor = bmcv.bm_image_to_tensor(BMimg)# here is a sail.Tensor
            BMimg2 = bmcv.tensor_to_bm_image(tensor, sail.Format.FORMAT_BGR_PLANAR)

**接口形式3:**
    .. code-block:: python

        def tensor_to_bm_image(self, 
                tensor: sail.Tensor, 
                img: sail.BMImage | sail.BMImageArray, 
                bgr2rgb: bool=False,
                layout: str = 'nchw') -> None
            
**参数说明3:**

* tensor: sail.Tensor

输入参数。待转换的Tensor。

* img : sail.BMImage | sail.BMImageArray

输出参数。返回转换后的图像。

* bgr2rgb: bool, default: False

输入参数。是否进行图像的通道变换。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            tensor = bmcv.bm_image_to_tensor(BMimg)# here is a sail.Tensor
            BMimg2 = sail.BMImage()
            bmcv.tensor_to_bm_image(tensor,BMimg2)

**接口形式4:**
    .. code-block:: python

        def tensor_to_bm_image(self, 
                tensor: sail.Tensor, 
                img: sail.BMImage | sail.BMImageArray, 
                format: sail.Format) -> None
            

**参数说明4:**

* tensor: sail.Tensor

输入参数。待转换的Tensor。

* img: sail.BMImage | sail.BMImageArray

输出参数。返回转换后的图像。

* format: sail.Format

输入参数。BMImage的像素格式。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            tensor = bmcv.bm_image_to_tensor(BMimg)# here is a sail.Tensor
            BMimg2 = sail.BMImage()
            bmcv.tensor_to_bm_image(tensor,BMimg2, sail.Format.FORMAT_BGR_PLANAR)

crop_and_resize
>>>>>>>>>>>>>>>>>>>>>>

对图片进行裁剪并resize。

**实现硬件**
* BM1684: VPP+智能视觉深度学习处理器
* BM1684X: VPP

**接口形式:**
    .. code-block:: python

        def crop_and_resize(self, 
                input: sail.BMImage | sail.BMImageArray, 
                crop_x0: int, 
                crop_y0: int, 
                crop_w: int, 
                crop_h: int, 
                resize_w: int, 
                resize_h: int, 
                resize_alg: sail.bmcv_resize_algorithm=sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST) 
                    -> sail.BMImage


**参数说明:**

* input : sail.BMImage | sail.BMImageArray

待处理的图像或图像数组。

* crop_x0 : int

裁剪窗口在x轴上的起始点。

* crop_y0 : int

裁剪窗口在y轴上的起始点。

* crop_w : int 

裁剪窗口的宽。

* crop_h : int 

裁剪窗口的高。

* resize_w : int

图像resize的目标宽度。

* resize_h : int

图像resize的目标高度。

* resize_alg : sail.bmcv_resize_algorithm

图像resize的插值算法，默认为sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST

**返回值说明:**

* output : sail.BMImage | sail.BMImageArray

返回处理后的图像或图像数组。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            BMimg3 = bmcv.crop_and_resize(BMimg,0,0,BMimg.width(),BMimg.height(),640,640,sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST)

crop
>>>>>>>>>

对图像进行裁剪。

**接口形式1:**
    .. code-block:: python

        def crop(self, 
                input: sail.BMImage | sail.BMImageArray, 
                crop_x0: int, 
                crop_y0: int, 
                crop_w: int, 
                crop_h: int) -> sail.BMImage | sail.BMImageArray
            

**参数说明:**

* input : sail.BMImage | sail.BMImageArray

待处理的图像或图像数组。

* crop_x0 : int

裁剪窗口在x轴上的起始点。

* crop_y0 : int

裁剪窗口在y轴上的起始点。

* crop_w : int 

裁剪窗口的宽。

* crop_h : int 

裁剪窗口的高。

**返回值说明:**

* output : sail.BMImage | sail.BMImageArray

返回处理后的图像或图像数组。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            crop_x0, crop_y0, crop_w, crop_h = 100, 100, 200, 200
            cropped_BMimg = bmcv.crop(BMimg, crop_x0, crop_y0, crop_w, crop_h)

对图像进行裁剪,可从一张图上裁剪出多个小图

**接口形式2:**
    .. code-block:: python

        def crop(self, 
                input: sail.BMImage  
                rects: list[list[]] ) -> list[sail.BMImage]
            

**参数说明:**

* input : sail.BMImage 

待处理的图像。

* rects: list[list[]]

[[crop_x0,crop_y0,crop_w0,crop_h0],[crop_x1,crop_y1,crop_w1,crop_h1]]

* crop_xi : int

第i个裁剪窗口在x轴上的起始点。

* crop_yi : int

第i个裁剪窗口在y轴上的起始点。

* crop_wi : int 

第i个裁剪窗口的宽。

* crop_hi : int 

第i个裁剪窗口的高。

**返回值说明1:**

* output : list[sail.BMImage] 

返回处理后的图像列表。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            rects = [
                [0, 0, 40, 40],
                [40, 40, 80, 80],
                #...more
            ]
            cropped_images_list = bmcv.crop(BMimg, rects)

resize
>>>>>>>>>>>>>>>>>

对图像进行resize。

**接口形式:**
    .. code-block:: python

        def resize(self, 
                input: sail.BMImage | sail.BMImageArray, 
                resize_w: int, 
                resize_h: int, 
                resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST)
            -> sail.BMImage | sail.BMImageArray

**参数说明:**

* input : sail.BMImage | sail.BMImageArray

待处理的图像或图像数组。

* resize_w : int

图像resize的目标宽度。

* resize_h : int

图像resize的目标高度。

* resize_alg : sail.bmcv_resize_algorithm

图像resize的插值算法，默认为sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST

**返回值说明:**

* output : sail.BMImage | sail.BMImageArray

返回处理后的图像或图像数组。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            BMimg_resize = bmcv.resize(BMimg,640,640,resize_alg=sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST)
            
vpp_crop_and_resize
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

利用VPP硬件加速图片的裁剪与resize。

**实现硬件**
* BM1684: VPP
* BM1684X: VPP

**接口形式:**
    .. code-block:: python

        def vpp_crop_and_resize(self, 
                        input: sail.BMImage | sail.BMImageArray, 
                        crop_x0: int, 
                        crop_y0: int, 
                        crop_w: int, 
                        crop_h: int, 
                        resize_w: int, 
                        resize_h: int, 
                        resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST)
                        -> sail.BMImage | sail.BMImageArray

**参数说明:**

* input : sail.BMImage | sail.BMImageArray

待处理的图像或图像数组。

* crop_x0 : int

裁剪窗口在x轴上的起始点。

* crop_y0 : int

裁剪窗口在y轴上的起始点。

* crop_w : int 

裁剪窗口的宽。

* crop_h : int 

裁剪窗口的高。

* resize_w : int

图像resize的目标宽度。

* resize_h : int

图像resize的目标高度。

* resize_alg : sail.bmcv_resize_algorithm

图像resize的插值算法，默认为sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST

**返回值说明:**

* output : sail.BMImage | sail.BMImageArray

返回处理后的图像或图像数组。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            crop_x0 = 100  
            crop_y0 = 100  
            crop_w = 200   
            crop_h = 200   
            resize_w = 300  
            resize_h = 300  
            
            resized_BMimg = bmcv.vpp_crop_and_resize(
                BMimg, 
                crop_x0, 
                crop_y0, 
                crop_w, 
                crop_h, 
                resize_w, 
                resize_h, 
                sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST
            )

vpp_crop_and_resize_padding
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

利用VPP硬件加速图片的裁剪与resize，并padding到指定大小。

**实现硬件**
* BM1684: VPP
* BM1684X: VPP

**接口形式:**
    .. code-block:: python

        def vpp_crop_and_resize_padding(self, 
                            input: sail.BMImage | sail.BMImageArray, 
                            crop_x0: int, 
                            crop_y0: int, 
                            crop_w: int, 
                            crop_h: int, 
                            resize_w: int, 
                            resize_h: int, 
                            padding: sail.PaddingAtrr,
                            resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST)
                             -> sail.BMImage | sail.BMImageArray

**参数说明:**

* input : sail.BMImage | sail.BMImageArray

待处理的图像或图像数组。

* crop_x0 : int

裁剪窗口在x轴上的起始点。

* crop_y0 : int

裁剪窗口在y轴上的起始点。

* crop_w : int 

裁剪窗口的宽。

* crop_h : int 

裁剪窗口的高。

* resize_w : int

图像resize的目标宽度。

* resize_h : int

图像resize的目标高度。

* padding : sail.PaddingAtrr

padding的配置信息。

* resize_alg : sail.bmcv_resize_algorithm

图像resize的插值算法，默认为sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST

**返回值说明:**

* output : sail.BMImage | sail.BMImageArray

返回处理后的图像或图像数组。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            paddingatt = sail.PaddingAtrr()   
            paddingatt.set_stx(0)
            paddingatt.set_sty(0)
            paddingatt.set_w(640)
            paddingatt.set_h(640)
            paddingatt.set_r(114)
            paddingatt.set_g(114)
            paddingatt.set_b(114)
            BMimg4 = bmcv.vpp_crop_and_resize_padding(BMimg,0,0,BMimg.width(),BMimg.height(),640,640,paddingatt)

vpp_crop
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

利用VPP硬件加速图片的裁剪。

**接口形式:**
    .. code-block:: python

        def vpp_crop(self, 
            input: sail.BMImage | sail.BMImageArray, 
            crop_x0: int, 
            crop_y0: int, 
            crop_w: int, 
            crop_h: int)->sail.BMImage | sail.BMImageArray

**参数说明:**

* input : sail.BMImage | sail.BMImageArray

待处理的图像或图像数组。

* crop_x0 : int

裁剪窗口在x轴上的起始点。

* crop_y0 : int

裁剪窗口在y轴上的起始点。

* crop_w : int 

裁剪窗口的宽。

* crop_h : int 

裁剪窗口的高。

**返回值说明:**

* output : sail.BMImage | sail.BMImageArray

返回处理后的图像或图像数组。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            crop_x0 = 100  
            crop_y0 = 100  
            crop_w = 200   
            crop_h = 200 
            BMimg4 = bmcv.vpp_crop(BMimg,crop_x0,crop_y0,crop_w,crop_h)

vpp_resize
>>>>>>>>>>>>>>>>>

利用VPP硬件加速图片的resize，采用最近邻插值算法。 

**接口形式1:**
    .. code-block:: python

        def vpp_resize(self, 
                input: sail.BMImage | sail.BMImageArray, 
                resize_w: int, 
                resize_h: int,
                resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST)->sail.BMImage | sail.BMImageArray

**参数说明1:**

* input : sail.BMImage | sail.BMImageArray

待处理的图像或图像数组。

* resize_w : int

图像resize的目标宽度。

* resize_h : int

图像resize的目标高度。

* resize_alg : sail.bmcv_resize_algorithm

图像resize的插值算法，默认为sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST

**返回值说明1:**

* output : sail.BMImage | sail.BMImageArray

返回处理后的图像或图像数组。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            BMimg_resize = bmcv.vpp_resize(BMimg,640,640,resize_alg=sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST)

**接口形式2:**
    .. code-block:: python

         def vpp_resize(self, 
                input: sail.BMImage | sail.BMImageArray, 
                output: sail.BMImage | sail.BMImageArray, 
                resize_w: int, 
                resize_h: int,
                resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST)->None

**参数说明2:**

* input : sail.BMImage | sail.BMImageArray

输入参数。待处理的图像或图像数组。

* output : sail.BMImage | sail.BMImageArray

输出参数。处理后的图像或图像数组。

* resize_w : int

输入参数。图像resize的目标宽度。

* resize_h : int

输入参数。图像resize的目标高度。

* resize_alg : sail.bmcv_resize_algorithm

图像resize的插值算法，默认为sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            BMimg_resize = sail.BMImage()
            bmcv.vpp_resize(BMimg,BMimg_resize,640,640,resize_alg=sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST)

vpp_resize_padding
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

利用VPP硬件加速图片的resize，并padding。

**接口形式:**
    .. code-block:: python

        def vpp_resize_padding(self, 
                input: sail.BMImage | sail.BMImageArray, 
                resize_w: int, 
                resize_h: int, 
                padding: PaddingAtrr,
                resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST)->sail.BMImage | sail.BMImageArray

**参数说明:**

* input : sail.BMImage | sail.BMImageArray

待处理的图像或图像数组。

* resize_w : int

图像resize的目标宽度。

* resize_h : int

图像resize的目标高度。

* padding : sail.PaddingAtrr

padding的配置信息。

* resize_alg : sail.bmcv_resize_algorithm

图像resize的插值算法，默认为sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST

**返回值说明:**

* output : sail.BMImage | sail.BMImageArray

返回处理后的图像或图像数组。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name, True, tpu_id)
            BMimg = decoder.read(handle)  # here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            paddingatt = sail.PaddingAtrr()   
            paddingatt.set_stx(0)
            paddingatt.set_sty(0)
            paddingatt.set_w(640)
            paddingatt.set_h(640)
            paddingatt.set_r(114)
            paddingatt.set_g(114)
            paddingatt.set_b(114)
            BMimg4 = bmcv.vpp_resize_padding(BMimg,640,640,paddingatt)

warp
>>>>>>>>>>>>>>>>

对图像进行仿射变换。

**实现硬件**
* BM1684: 智能视觉深度学习处理器
* BM1684X: 智能视觉深度学习处理器

**接口形式1:**
    .. code-block:: python

        def warp(self, 
            input: sail.BMImage | sail.BMImageArray, 
            matrix: list,
            use_bilinear: int = 0,
            similar_to_opencv: bool = False)->sail.BMImage | sail.BMImageArray

**参数说明1:**

* input : sail.BMImage | sail.BMImageArray

待处理的图像或图像数组。

* matrix: 2d list

2x3的仿射变换矩阵。

* use_bilinear: int

是否使用双线性插值，默认为0使用最近邻插值，1为双线性插值

* similar_to_opencv: bool

是否使用与opencv仿射变换对齐的接口

**返回值说明1:**

* output : sail.BMImage | sail.BMImageArray

返回处理后的图像或图像数组。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            rotated_matrix = [[0.9996914396,-0.02484,0],[0.02484,0.9996914396,0]]
            BMimg6 = bmcv.warp(BMimg, rotated_matrix)

**接口形式2:**
    .. code-block:: python

        def warp(self, 
            input: sail.BMImage | sail.BMImageArray, 
            output: sail.BMImage | sail.BMImageArray, 
            matrix: list,
            use_bilinear: int = 0,
            similar_to_opencv: bool = False)->int

**参数说明2:**

* input : sail.BMImage | sail.BMImageArray

待处理的图像或图像数组。

* output : sail.BMImage | sail.BMImageArray

待返回的输出图像或图像数组。

* matrix: 2d list

2x3的仿射变换矩阵。

* use_bilinear: int

是否使用双线性插值，默认为0使用最近邻插值，1为双线性插值

* similar_to_opencv: bool

是否使用与opencv仿射变换对齐的接口

**返回值说明2:**

如果成功返回0，否则返回非0值。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            rotated_matrix = [[0.9996914396,-0.02484,0],[0.02484,0.9996914396,0]]
            output = sail.BMImage()
            ret = bmcv.warp(BMimg, output,rotated_matrix)

convert_to
>>>>>>>>>>>>>>

对图像进行线性变换。

**实现硬件**
* BM1684: 智能视觉深度学习处理器
* BM1684X: VPP+智能视觉深度学习处理器

**接口形式1:**
    .. code-block:: python

        def convert_to(self, 
            input: sail.BMImage | sail.BMImageArray, 
            alpha_beta: tuple)->sail.BMImage | sail.BMImageArray
    
**参数说明1:**

* input : sail.BMImage | sail.BMImageArray

待处理的图像或图像数组。

* alpha_beta: tuple

分别为三个通道线性变换的系数((a0, b0), (a1, b1), (a2, b2))。

**返回值说明1:**

* output : sail.BMImage | sail.BMImageArray

返回处理后的图像或图像数组。


**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            alpha_beta = (1.0/255,0),(1.0/255,0),(1.0/255,0)
            BMimg5 = bmcv.convert_to(BMimg, alpha_beta)

**接口形式2:**
    .. code-block:: python

        def convert_to(self, 
                input: sail.BMImage | sail.BMImageArray, 
                output: sail.BMImage | sail.BMImageArray, 
                alpha_beta: tuple)->None
    
**参数说明2:**

* input : sail.BMImage | sail.BMImageArray

输入参数。待处理的图像或图像数组。

* output : sail.BMImage | sail.BMImageArray

输出参数。返回处理后的图像或图像数组。

* alpha_beta: tuple

分别为三个通道线性变换的系数((a0, b0), (a1, b1), (a2, b2))。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            alpha_beta = (1.0/255,0),(1.0/255,0),(1.0/255,0)
            BMimg5 = sail.BMImage()
            bmcv.convert_to(BMimg, BMimg5,alpha_beta)

yuv2bgr
>>>>>>>>>>>>

将图像的格式从YUV转换为BGR。

**实现硬件**
* BM1684: 智能视觉深度学习处理器+VPP
* BM1684X: VPP

**接口形式:**
    .. code-block:: python

        def yuv2bgr(input: sail.BMImage | sail.BMImageArray)
                -> sail.BMImage | sail.BMImageArray

**参数说明:**

* input : sail.BMImage | sail.BMImageArray

待转换的图像。

**返回值说明:**

* output : sail.BMImage | sail.BMImageArray

返回转换后的图像。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            BMimg5 = bmcv.yuv2bgr(BMimg)

rectangle
>>>>>>>>>>>>>>>>>>

在图像上画一个矩形框。

**实现硬件**
* BM1684: 智能视觉深度学习处理器
* BM1684X: VPP

**接口形式:**
    .. code-block:: python

        def rectangle(self, 
                image: sail.BMImage, 
                x0: int, 
                y0: int, 
                w: int, 
                h: int, 
                color: tuple, 
                thickness: int=1)->int

        def rectangle(self,
                image: sail.bm_image, 
                x0: int, 
                y0: int, 
                w: int, 
                h: int, 
                color: tuple, 
                thickness: int=1)->int

**参数说明:**

* image : sail.BMImage 或 sail.bm_image

待画框的图像。

* x0 : int

矩形框在x轴上的起点。

* y0 : int

矩形框在y轴上的起点。

* w : int

矩形框的宽度。

* h : int

矩形框的高度。

* color : tuple

矩形框的颜色。

* thickness : int

矩形框线条的粗细。

**返回值说明:**

如果画框成功返回0，否则返回非0值。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            ret = bmcv.rectangle(BMimg, 20, 20, 600, 600,(0,0,255),2)
            # ret = bmcv.rectangle(BMimg.data(), 20, 20, 600, 600,(0,0,255),2)

fillRectangle
>>>>>>>>>>>>>>>>>>

在图像上填充一个矩形框。

**实现硬件**
* BM1684: 智能视觉深度学习处理器
* BM1684X: VPP

**接口形式:**
    .. code-block:: python

        def fillRectangle(self, 
                image: sail.BMImage, 
                x0: int, 
                y0: int, 
                w: int, 
                h: int, 
                color: tuple)->int

        def fillRectangle(self,
                image: sail.bm_image, 
                x0: int, 
                y0: int, 
                w: int, 
                h: int, 
                color: tuple)->int

**参数说明:**

* image : sail.BMImage 或 sail.bm_image

待画框的图像。

* x0 : int

矩形框在x轴上的起点。

* y0 : int

矩形框在y轴上的起点。

* w : int

矩形框的宽度。

* h : int

矩形框的高度。

* color : tuple

矩形框的颜色。


**返回值说明:**

如果画框成功返回0，否则返回非0值。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            ret = bmcv.fillRectangle(BMimg, 20, 20, 600, 600,(0,0,255))
            # ret = bmcv.fillRectangle(BMimg.data(), 20, 20, 600, 600,(0,0,255))

imwrite
>>>>>>>>>>>>>>>>>

将图像保存在特定文件。

**实现硬件**

* BM1684: JPU+VPP+智能视觉深度学习处理器

* BM1684X: JPU+VPP

**接口形式:**
    .. code-block:: python

        def imwrite(self, file_name: str, image: sail.BMImage) -> int

        def imwrite(self, file_name: str, image: sail.bm_image) -> int

**参数说明:**

* file_name : str

文件的名称。

* output : sail.BMImage 或 sail.bm_image

需要保存的图像。

**返回值说明:**

* process_status : int

如果保存成功返回0，否则返回非0值。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            bmcv.imwrite("{}_{}.jpg".format(BMimg.width(),BMimg.height()),BMimg)
            # bmcv.imwrite("{}_{}.jpg".format(BMimg.width(),BMimg.height()),BMimg.data())


imread
>>>>>>>>>>>>>>>>>

读取和解码图片文件，仅支持 JPEG baseline 格式的硬解码。对于其他格式，如 PNG 和 BMP，则采用软解码。
对于 JPEG baseline 图片，返回的 BMImage 将保持 YUV 色彩空间，像素格式依据图片文件本身的采样方式，例如 YUV420；
而对于其他格式，返回的 BMImage 将保持其输入的对应色彩空间。

**实现硬件**

* BM1684: JPU(JPEG baseline), CPU(其他格式)

* BM1684X: JPU(JPEG baseline), CPU(其他格式)

**接口形式:**
    .. code-block:: python

        def imread(self, filename: str) -> BMImage

**参数说明:**

* filename : str

需要读取的图片文件路径。

**返回值说明:**

* output : sail.BMImage

返回解码得到的BMImage。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            bmcv = sail.Bmcv(handle)
            filename = "your_image_path"
            BMimg = bmcv.imread(filename)


get_handle
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取Bmcv中的设备句柄Handle。

**接口形式:**
    .. code-block:: python

        def get_handle(self)->sail.Handle

**返回值说明:**

* handle: sail.Handle

Bmcv中的设备句柄Handle。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            handle1 = bmcv.get_handle()

crop_and_resize_padding
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

对图像进行裁剪并resize，然后padding。

**实现硬件**
* BM1684: VPP+智能视觉深度学习处理器
* BM1684X: VPP+智能视觉深度学习处理器

**接口形式:**
    .. code-block:: python

        def crop_and_resize_padding(self, 
                        input: sail.BMImage, 
                        crop_x0: int, 
                        crop_y0: int, 
                        crop_w: int, 
                        crop_h: int, 
                        resize_w: int, 
                        resize_h: int, 
                        padding: PaddingAtrr, 
                        resize_alg=sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST)
                    ->sail.BMImage

**参数说明:**

* input : sail.BMImage

待处理的图像。

* crop_x0 : int

裁剪窗口在x轴上的起始点。

* crop_y0 : int

裁剪窗口在y轴上的起始点。

* crop_w : int 

裁剪窗口的宽。

* crop_h : int 

裁剪窗口的高。

* resize_w : int

图像resize的目标宽度。

* resize_h : int

图像resize的目标高度。

* padding : sail.PaddingAtrr

padding的配置信息。

* resize_alg : bmcv_resize_algorithm

resize采用的插值算法。

**返回值说明:**

* output : sail.BMImage

返回处理后的图像。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            crop_x0 = 100  
            crop_y0 = 100  
            crop_w = 200   
            crop_h = 200  
            resize_w = 300  
            resize_h = 300  

            paddingatt = sail.PaddingAtrr()   
            paddingatt.set_stx(0)
            paddingatt.set_sty(0)
            paddingatt.set_w(300)
            paddingatt.set_h(300)
            paddingatt.set_r(114)
            paddingatt.set_g(114)
            paddingatt.set_b(114)
            padded_BMimg = bmcv.crop_and_resize_padding(
                BMimg,
                crop_x0,
                crop_y0,
                crop_w,
                crop_h,
                resize_w,
                resize_h,
                paddingatt,
                sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST
            )


convert_format
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

将图像的格式转换为output中的格式，并拷贝到output。

**实现硬件**
* BM1684: VPP+智能视觉深度学习处理器
* BM1684X: VPP

**接口形式1:**
    .. code-block:: python

        def convert_format(self, input: sail.BMImage, output: sail.BMImage)->None

**参数说明1:**

* input : sail.BMImage

输入参数。待转换的图像。

* output : sail.BMImage

输出参数。将input中的图像转化为output的图像格式并拷贝到output。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            output = sail.BMImage()
            bmcv.convert_format(BMimg,output)

**接口形式2:**

将一张图像转换成目标格式。

    .. code-block:: python

        def convert_format(self, input: sail.BMImage, image_format:sail.bm_image_format_ext)->sail.BMImage

**参数说明2:**

* input : sail.BMImage

待转换的图像。

* image_format : sail.bm_image_format_ext

转换的目标格式。

**返回值说明2:**

* output : sail.BMImage

返回转换后的图像。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            output = bmcv.convert_format(BMimg,sail.FORMAT_BGR_PLANAR)
            
vpp_convert_format
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

利用VPP硬件加速图片的格式转换。

**实现硬件**
* BM1684: VPP
* BM1684X: VPP

**接口形式1:**
    .. code-block:: python

        def vpp_convert_format(self, input: sail.BMImage, output: sail.BMImage)->None 

**参数说明1:**

* input : sail.BMImage

输入参数。待转换的图像。

* output : sail.BMImage

输出参数。将input中的图像转化为output的图像格式并拷贝到output。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            output = sail.BMImage()
            bmcv.vpp_convert_format(BMimg,output)

**接口形式2:**

将一张图像转换成目标格式。

    .. code-block:: python

        def vpp_convert_format(self, input: sail.BMImage, image_format:sail.bm_image_format_ext)->sail.BMImage

**参数说明2:**

* input : sail.BMImage

待转换的图像。

* image_format : sail.bm_image_format_ext

转换的目标格式。

**返回值说明2:**

* output : sail.BMImage

返回转换后的图像。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            output = bmcv.vpp_convert_format(BMimg,sail.FORMAT_BGR_PLANAR)
            
putText
>>>>>>>>>>

在图像上添加text。只支持英文文字。

输入的BMImage支持的像素格式为：
FORMAT_GRAY、FORMAT_YUV420P、FORMAT_YUV422P、FORMAT_YUV444P、FORMAT_NV12、
FORMAT_NV21、FORMAT_NV16、FORMAT_NV61。

**实现硬件**
* BM1684: 处理器
* BM1684X: 处理器

**接口形式:**
    .. code-block:: python
        
        def putText(self, 
                input: sail.BMImage, 
                text: str, 
                x: int, 
                y: int, 
                color: tuple, 
                fontScale: int, 
                thickness: int)->int

        def putText(self, 
                input: sail.bm_image, 
                text: str, 
                x: int, 
                y: int, 
                color: tuple, 
                fontScale: int, 
                thickness: int)->int

**参数说明:**

* input : sail.BMImage 或 sail.bm_image

待处理的图像。

* text: str

需要添加的文本。

* x: int

添加的起始点位置。

* y: int

添加的起始点位置。

* color : tuple

字体的颜色。

* fontScale: int

字号的大小。

* thickness : int

字体的粗细。建议设置为偶数。

**返回值说明:**

* process_status : int

如果处理成功返回0，否则返回非0值。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            bgr_img = decoder.read(handle)
            bmcv = sail.Bmcv(handle)
            yuv_img = bmcv.convert_format(bgr_img, sail.FORMAT_YUV420P)
            ret = bmcv.putText(yuv_img, "some text" , 20, 20, [0,0,255], 1.4, 2)
            assert ret == 0


image_add_weighted
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

将两张图像按不同的权重相加。

**实现硬件**
* BM1684: 智能视觉深度学习处理器
* BM1684X: 智能视觉深度学习处理器

**接口形式1:**
    .. code-block:: python
        
        def image_add_weighted(self, 
                    input0: sail.BMImage, 
                    alpha: float, 
                    input1: sail.BMImage, 
                    beta: float, 
                    gamma: float, 
                    output: sail.BMImage)->None

**参数说明1:**

* input0 : sail.BMImage

输入参数。待处理的图像0。

* alpha : float

输入参数。两张图像相加的权重alpha

* input1 : sail.BMImage

输入参数。待处理的图像1。

* beta : float

输入参数。两张图像相加的权重beta

* gamma : float

输入参数。两张图像相加的权重gamma

* output: BMImage

输出参数。相加后的图像output = input1 * alpha + input2 * beta + gamma


**示例代码1:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name1 = "your_image_path1"
            image_name2 = "your_image_path2"
            decoder1 = sail.Decoder(image_name1,True,tpu_id)
            decoder2 = sail.Decoder(image_name2,True,tpu_id)
            BMimg1 = decoder1.read(handle)# here is a sail.BMImage
            BMimg2 = decoder2.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            bmg = sail.BMImage()
            bmcv.image_add_weighted(BMimg1,0.5,BMimg2,0.5,0.5,bmg)

**接口形式2:**
    .. code-block:: python

        def image_add_weighted(self, 
                    input0: sail.BMImage, 
                    alpha: float, 
                    input1: sail.BMImage, 
                    beta: float, 
                    gamma: float)->sail.BMImage


**参数说明2:**

* input0 : sail.BMImage

输入参数。待处理的图像0。

* alpha : float

输入参数。两张图像相加的权重alpha

* input1 : sail.BMImage

输入参数。待处理的图像1。

* beta : float

输入参数。两张图像相加的权重beta

* gamma : float

输入参数。两张图像相加的权重gamma

**返回值说明2:**

* output: sail.BMImage

返回相加后的图像output = input1 * alpha + input2 * beta + gamma

**示例代码2:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name1 = "your_image_path1"
            image_name2 = "your_image_path2"
            decoder1 = sail.Decoder(image_name1,True,tpu_id)
            decoder2 = sail.Decoder(image_name2,True,tpu_id)
            BMimg1 = decoder1.read(handle)# here is a sail.BMImage
            BMimg2 = decoder2.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            bmg = bmcv.image_add_weighted(BMimg1,0.5,BMimg2,0.5,0.5)

image_copy_to
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

进行图像间的数据拷贝

**实现硬件**
* BM1684: 智能视觉深度学习处理器
* BM1684X: VPP+智能视觉深度学习处理器

**接口形式:**
    .. code-block:: python

        def image_copy_to(self, 
                    input: BMImage|BMImageArray, 
                    output: BMImage|BMImageArray, 
                    start_x: int, 
                    start_y: int)->None

**参数说明:**

* input: BMImage|BMImageArray

输入参数。待拷贝的BMImage或BMImageArray。

* output: BMImage|BMImageArray

输出参数。拷贝后的BMImage或BMImageArray

* start_x: int

输入参数。拷贝到目标图像的起始点。

* start_y: int

输入参数。拷贝到目标图像的起始点。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name1 = "your_image_path1"
            image_name2 = "your_image_path2"
            decoder1 = sail.Decoder(image_name1,True,tpu_id)
            decoder2 = sail.Decoder(image_name2,True,tpu_id)
            BMimg1 = decoder1.read(handle)# here is a sail.BMImage
            BMimg2 = decoder2.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            bmcv.image_copy_to(BMimg1,BMimg2,0,0)

image_copy_to_padding
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

进行input和output间的图像数据拷贝并padding。

**实现硬件**
* BM1684: 智能视觉深度学习处理器
* BM1684X: VPP+智能视觉深度学习处理器

**接口形式:**
    .. code-block:: python
    
        def image_copy_to_padding(self, 
                    input: BMImage|BMImageArray, 
                    output: BMImage|BMImageArray, 
                    padding_r: int, 
                    padding_g: int, 
                    padding_b: int, 
                    start_x: int, 
                    start_y: int)->None

**参数说明:**

* input: BMImage|BMImageArray

输入参数。待拷贝的BMImage或BMImageArray。

* output: BMImage|BMImageArray

输出参数。拷贝后的BMImage或BMImageArray

* padding_r: int

输入参数。R通道的padding值。

* padding_g: int

输入参数。G通道的padding值。

* padding_b: int

输入参数。B通道的padding值。

* start_x: int

输入参数。拷贝到目标图像的起始点。

* start_y: int

输入参数。拷贝到目标图像的起始点。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name1 = "your_image_path1"
            image_name2 = "your_image_path2"
            decoder1 = sail.Decoder(image_name1,True,tpu_id)
            decoder2 = sail.Decoder(image_name2,True,tpu_id)
            BMimg1 = decoder1.read(handle)# here is a sail.BMImage
            BMimg2 = decoder2.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            bmcv.image_copy_to_padding(BMimg1,BMimg2,128,128,128,0,0)
            
nms
>>>>>>>>

利用智能视觉深度学习处理器进行NMS

**注意：请查询《BMCV开发参考手册/BMCV API》确认当前算子是否适配BM1688。**

**实现硬件**
* BM1684: 智能视觉深度学习处理器
* BM1684X: 智能视觉深度学习处理器

**接口形式:**
    .. code-block:: python

        def nms(self, input: numpy.ndarray, threshold: float)->numpy.ndarray

**参数说明:**

* input: numpy.ndarray

待处理的检测框的数组，shape必须是（n,5） n<56000 [left,top,right,bottom,score]。

* threshold: float

nms的阈值。

**返回值说明:**

* result: numpy.ndarray

返回NMS后的检测框数组。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            bmcv = sail.Bmcv(handle)
            input_boxes = np.array([
            [50, 50, 100, 100, 0.9],
            [60, 60, 110, 110, 0.85],
            [200, 200, 250, 250, 0.7],
            [130, 50, 180, 100, 0.8],
            [205, 205, 255, 255, 0.75]
            ])
            nms_threshold = 0.5
            selected_boxes  = bmcv.nms(input_boxes, nms_threshold)
            print(selected_boxes)
            
drawPoint
>>>>>>>>>>>>>

在图像上画点。

**实现硬件**
* BM1684: 处理器
* BM1684X: VPP

**接口形式:**
    .. code-block:: python

        def drawPoint(self, 
                image: BMImage, 
                center: Tuple[int, int], 
                color: Tuple[int, int, int], 
                radius: int) -> int:

        def drawPoint(self, 
                image: bm_image, 
                center: Tuple[int, int], 
                color: Tuple[int, int, int], 
                radius: int) -> int:

**参数说明:**

* image: BMImage 或 bm_image

输入图像，在该BMImage上直接画点作为输出。

* center: Tuple[int, int]

点的中心坐标。

* color: Tuple[int, int, int]

点的颜色。

* radius: int

点的半径。

**返回值说明**

如果画点成功返回0，否则返回非0值。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            ret = bmcv.drawPoint(BMimg, (320, 320), (0,255,255),2)
            # ret = bmcv.drawPoint(BMimg.data(), (320, 320), (0,255,255),2)

warp_perspective
>>>>>>>>>>>>>>>>>>>>>

对图像进行透视变换。

**接口形式:**
    .. code-block:: python

        def warp_perspective(self, 
                    input: BMImage, 
                    coordinate: tuple, 
                    output_width: int,  
                    output_height: int,
                    format: bm_image_format_ext = FORMAT_BGR_PLANAR,  
                    dtype: bm_image_data_format_ext = DATA_TYPE_EXT_1N_BYTE,
                    use_bilinear: int = 0 ) -> BMImage:

**参数说明:**

* input: BMImage

待处理的图像。

* coordinate: tuple

变换区域的四顶点原始坐标。tuple(tuple(int,int))

例如((left_top.x, left_top.y), (right_top.x, right_top.y),
(left_bottom.x, left_bottom.y), (right_bottom.x, right_bottom.y))

* output_width: Output width

输出图像的宽。

* output_height: Output height

输出图像的高。

* bm_image_format_ext: sail.Format

输出图像的格式。

* dtype: sail.ImgDtype

输出图像的数据类型。

* use_bilinear: bool

是否使用双线性插值。

**返回值说明:**

* output: image

输出变换后的图像。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            img = bmcv.warp_perspective(BMimg, ((100, 100), (540, 100), (100, 380), (540, 380)),640,640)

get_bm_data_type
>>>>>>>>>>>>>>>>>>>>

将ImgDtype转换为Dtype

**接口形式:**
    .. code-block:: python

        def get_bm_data_type((self, format: sail.ImgDtype) -> sail.Dtype

**参数说明:**

* format: sail.ImgDtype

需要转换的类型。

**返回值说明:**

* ret: sail.Dtype

转换后的类型。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            type = bmcv.get_bm_data_type(sail.DATA_TYPE_EXT_FLOAT32)

get_bm_image_data_format
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

将Dtype转换为ImgDtype。

**接口形式:**
    .. code-block:: python

        def get_bm_image_data_format(self, dtype: sail.Dtype) -> sail.ImgDtype

**参数说明:**

* dtype: sail.Dtype

需要转换的sail.Dtype

**返回值说明:**

* ret: sail.ImgDtype

返回转换后的类型。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            type = bmcv.get_bm_image_data_format(sail.BM_FLOAT32)

imdecode
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

从内存中载入图像到BMImage中。

**接口形式:**
    .. code-block:: python

        def imdecode(self, data_bytes: bytes) -> sail.BMImage:
          
**参数说明:**

* data_bytes: bytes

系统内存中图像的bytes

**返回值说明:**

* ret: sail.BMImage

返回解码后的图像。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            with open(image_name, 'rb') as image_file:
                image_data_bytes = image_file.read()
            bmcv = sail.Bmcv(handle)
            src_img = bmcv.imdecode(image_data_bytes)

imencode
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

编码一张图片，并返回编码后的数据。

**接口形式:**
    .. code-block:: python

        def imencode(self, ext: str, img: BMImage) -> numpy.ndarray:
          
**参数说明:**

* ext: str

输入参数。图片编码格式。 ``".jpg"`` , ``".png"`` 等。

* img: BMImage

输入参数。输入图片，只支持FORMAT_BGR_PACKED，DATA_TYPE_EXT_1N_BYTE的图片。

**返回值说明:**

* ret: numpy.array

编码后放在系统内存中的数据。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            ret = bmcv.imencode(".jpg",BMimg)

fft
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

实现对Tensor的快速傅里叶变换。

**注意：请查询《BMCV开发参考手册/BMCV API》确认当前算子是否适配BM1688。**

**接口形式:**
    .. code-block:: python

        fft(self, forward: bool, input_real: Tensor) -> list[Tensor]

        fft(self, forward: bool, input_real: Tensor, input_imag: Tensor) -> list[Tensor]
    
**参数说明:**

* forward: bool

是否进行正向迁移。

* input_real: Tensor

输入的实数部分。

* input_imag: Tensor

输入的虚数部分。

**返回值说明:**

* ret: list[Tensor]

返回输出的实数部分和虚数部分。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            random_array1 = np.random.rand(1, 1, 512, 512).astype('float32'),
            random_array2 = np.random.rand(1, 1, 512, 512).astype('float32'),
            bmcv = sail.Bmcv(handle)
            input_real  = sail.Tensor(handle,random_array1,True)
            input_imag  = sail.Tensor(handle,random_array2,True)
            forward = True
            result_complex = bmcv.fft(forward,input_real,input_imag)

convert_yuv420p_to_gray
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

将YUV420P格式的图片转为灰度图。

**接口形式1:**
    .. code-block:: python

        def convert_yuv420p_to_gray(self, input: sail.BMImage, output: sail.BMImage)->None 

**参数说明1:**

* input : sail.BMImage

输入参数。待转换的图像。

* output : sail.BMImage

输出参数。转换后的图像。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg1 = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            BMimg2 = sail.BMImage()
            bmcv.convert_yuv420p_to_gray(BMimg1,BMimg2)

**接口形式2:**

将YUV420P格式的图片转为灰度图。

    .. code-block:: python

        def convert_yuv420p_to_gray_(self, input: sail.bm_image, output: sail.bm_image)->None 

**参数说明2:**

* input : sail.bm_image

待转换的图像。

* output : sail.bm_image

转换后的图像。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg1 = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            bmg = sail.BMImage()
            bmcv.convert_yuv420p_to_gray_(BMimg1.data(),bmg.data())

mat_to_bm_image
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
将opencv的mat转为sail的BMImage。

**接口形式1:**
    .. code-block:: python

        def mat_to_bm_image(self, mat: numpy.ndarray[numpy.uint8]) -> BMImage: 

**参数说明1:**

* mat : numpy

输入参数。待转换的opencv mat。

**返回值说明:**

* ret: sail.BMImage

返回转换后的sail.BMImage。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            bmcv = sail.Bmcv(handle)
            opencv_mat = cv2.imread(image_name)
            sail_bm_image = bmcv.mat_to_bm_image(opencv_mat)

**接口形式2:**
    .. code-block:: python

        def mat_to_bm_image(self, mat: numpy.ndarray[numpy.uint8], img: BMImage) -> int: 

**参数说明2:**

* mat : numpy

待转换的opencv mat。

* img : sail.BMImage

转换后的BMImage。

**返回值说明:**

* ret: int

成功后返回0

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            bmcv = sail.Bmcv(handle)
            opencv_mat = cv2.imread(image_name)
            BMimg2 = sail.BMImage()
            ret = bmcv.mat_to_bm_image(opencv_mat,BMimg2)

watermark_superpose
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

实现对图片添加多个水印。

**接口形式:**
    .. code-block:: python

        def watermark_superpose(self,
        image: sail.BMImage,
        water_name:string,
        bitmap_type: int,
        pitch: int,
        rects: list[list[int]],
        color: tuple
                )->int
    
**参数说明:**

* image: sail.BMImage

输入图片

* water_name:string

水印文件路径

* bitmap_type: int

输入参数。水印类型, 值0表示水印为8bit数据类型(有透明度信息), 值1表示水印为1bit数据类型(无透明度信息)。

* pitch: int

输入参数。水印文件每行的byte数, 可理解为水印的宽。

* rects: list[list[int]]

输入参数。水印位置，包含每个水印起始点和宽高。

* color: tuple

输入参数。水印的颜色。

**返回值说明:**

* ret: int

返回是否成功

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path1"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg1 = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            bmg = sail.BMImage()
            water_name = 'your_watermark_path'
            ret = bmcv.watermark_superpose(BMimg1,water_name,0,117,[[0,0,117,79],[0,90,117,79]],[128,128,128])

polylines
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

可以实现在一张图像上画一条或多条线段，从而可以实现画多边形的功能，并支持指定线的颜色和线的宽度。

**接口形式:**
    .. code-block:: c

        def polylines(self, image: BMImage, pts: list[list[tuple(int, int)]], isClosed: bool, color: tuple(int, int, int), thickness: int = 1, shift: int = 0) -> int:


**参数说明:**

* img : sail.BMImage

输入图片。

* pts : list[list[tuple(int, int)]]

线段的起始点和终点坐标，可输入多个坐标点。图像左上角为原点，向右延伸为x方向，向下延伸为y方向。

* isClosed : bool
  
图形是否闭合。

* color : tuple(int, int, int)

画线的颜色，分别为RGB三个通道的值。

* thickness : int 

画线的宽度，对于YUV格式的图像建议设置为偶数。

* shift : int

多边形缩放倍数，默认不缩放。缩放倍数为(1/2)^shift。


**返回值说明:**

* ret: int

成功后返回0

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg1 = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            bm = bmcv.vpp_convert_format(BMimg1,sail.FORMAT_YUV444P)
            ret = bmcv.polylines(bm,[[(10,20),(40,80)]],True,[128,128,128])

mosaic
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

该接口用于在图像上打一个或多个马赛克。

**接口形式:**
    .. code-block:: python

        def mosaic(self, mosaic_num: int, img: sail.BMImage, rects: list[list[int,int,int,int]], is_expand:int) -> int


**参数说明:**

* mosaic_num : int

马赛克数量，指rects中列表长度。

* img : sail.BMImage

待转换的图像。

* rects : list[list[int,int,int,int]]

多个马赛克位置，列表中每个元素中参数为[马赛克在x轴起始点,马赛克在y轴起始点,马赛克宽,马赛克高]

* is_expand : int
  
是否扩列。值为0时表示不扩列, 值为1时表示在原马赛克周围扩列一个宏块(8个像素)。


**返回值说明:**

* ret: int

成功后返回0

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg1 = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            ret = bmcv.mosaic(2, BMimg1, [[10,10,100,2000],[500,500,1000,100]], 1)
  
  

gaussian_blur
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

该接口用于对图像进行高斯滤波。
**注意：旧版本SDK并不支持BM1684X，当前SDK是否支持请查询《BMCV开发参考手册》, BMCV API页面查看。**

**接口形式:**
    .. code-block:: python
        
        def gaussian_blur(self, input: BMImage, kw: int, kh : int, sigmaX : float, sigmaY : float = 0.0) -> BMImage: 


**参数说明:**

* input : sail.BMImage

待转换的图像。

* kw : int

kernel 在width方向上的大小。

* kh : int
  
kernel 在height方向上的大小。

* sigmaX : float

X方向上的高斯核标准差。

* sigmaY : float

Y方向上的高斯核标准差。如果为0则表示与X方向上的高斯核标准差相同。默认为0。

**返回值说明:**

* output : sail.BMImage

返回经过高斯滤波的图像。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            handle = sail.Handle(0)
            bmcv = sail.Bmcv(handle)


            bmimg = sail.BMImage()
            decoder = sail.Decoder("your_img.jpg",True,0)
            bmimg = decoder.read(handle)

            print(bmimg.format())
            output = bmcv.gaussian_blur(bmimg, 3, 3, 0.1)

            bmcv.imwrite("out.jpg",output)

transpose
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

该接口可以实现图片宽和高的转置。

**接口形式1:**
    .. code-block:: python

        def transpose(self, src: sail.BMImage) -> sail.BMImage:


**参数说明1:**

* src : sail.BMImage

待转换的图像。


**返回值说明1:**

* output: sail.BMImage:

返回转换后的图像。


**接口形式2:**
    .. code-block:: python

        def transpose(self, src: sail.BMImage, dst: sail.BMImage) -> int:


**参数说明2:**

* src : sail.BMImage

待转换的图像。

* dst : sail.BMImage

输出图像的 sail.BMImage 结构体。

**返回值说明2:**

* ret : int

成功返回0，否则返回非0值。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            
            handle = sail.Handle(0)
            bmcv = sail.Bmcv(handle)
            bmimg = sail.BMImage()
            decoder = sail.Decoder("your_img.jpg",True,0)
            bmimg = decoder.read(handle)
            img = bmcv.convert_format(bmimg,sail.Format.FORMAT_GRAY)
            print("readed")
            print(img.format())
            output = bmcv.transpose(img)

            bmcv.imwrite("out.jpg",output)


Sobel
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

边缘检测Sobel算子。

**注意：请查询《BMCV开发参考手册/BMCV API》确认当前算子是否适配BM1684X、BM1688。**

**接口形式1:**
    .. code-block:: python

        def Sobel(self, input: BMImage, output: BMImage, dx: int, dy: int, ksize: int = 3, scale: float = 1, delta: float = 0) -> int:


**参数说明1:**

* input : sail.BMImage

待转换的图像。

* output : sail.BMImage

转换后的图像。

* dx : int

x方向上的差分阶数。

* dy : int
  
y方向上的差分阶数。

* ksize : int

Sobel核的大小，必须是-1,1,3,5或7。其中特殊的，如果是-1则使用3×3 Scharr滤波器，如果是1则使用3×1或者1×3的核。默认值为3。

* scale : float

对求出的差分结果乘以该系数，默认值为1。

* delta : float

在输出最终结果之前加上该偏移量，默认值为0。

**返回值说明1:**

* ret: int

成功后返回0

**接口形式2:**
    .. code-block:: python

        def Sobel(self, input: BMImage, dx: int, dy: int, ksize: int = 3, scale: float = 1, delta: float = 0) -> BMImage:


**参数说明2:**

* input : sail.BMImage

待转换的图像。

* dx : int

x方向上的差分阶数。

* dy : int
  
y方向上的差分阶数。

* ksize : int

Sobel核的大小，必须是-1,1,3,5或7。其中特殊的，如果是-1则使用3×3 Scharr滤波器，如果是1则使用3×1或者1×3的核。默认值为3。

* scale : float

对求出的差分结果乘以该系数，默认值为1。

* delta : float

在输出最终结果之前加上该偏移量，默认值为0。

**返回值说明2:**

* output: sail.BMImage

返回转换后的图像。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            handle = sail.Handle(0)
            bmcv = sail.Bmcv(handle)

            bmimg = sail.BMImage()
            decoder = sail.Decoder("your_img.jpg",True,1)
            bmimg = decoder.read(handle)
            
            print(bmimg.format())
            output = bmcv.Sobel(bmimg, 1, 1)

            bmcv.imwrite("out.jpg",output)

drawLines
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

在给定的图像上绘制多条线段。

**注意：请查询《BMCV开发参考手册/BMCV API》确认当前算子是否适配BM1684X。**

**接口形式:**
    .. code-block:: python

        def drawLines(self, image: BMImage, start_points: list[tuple[int, int]], end_points: list[tuple[int, int]], line_num: int, color: tuple[int, int, int], thickness: int) -> int:


**参数说明:**

* img : sail.BMImage

输入图片。

* start_points : list[tuple[int, int]]

线段的起始点坐标列表，图像左上角为原点，向右为x方向，向下为y方向。

* end_points : list[tuple[int, int]]

线段的结束点坐标列表，必须与起始点列表长度相同。 

* line_num : int

线段的数量，必须和线段起、终点列表长度值相同

* color : tuple[int, int, int]

线段的颜色，RGB格式。

* thickness : int 

线段的宽度。

**返回值说明:**

* ret: int

成功后返回0。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg1 = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            start_points = [(10, 10), (30, 30)]  
            end_points = [(20, 20), (40, 40)]    
            line_num = 2
            color = (255, 0, 0)  
            thickness = 2  
            bm = bmcv.vpp_convert_format(BMimg1,sail.FORMAT_YUV444P)
            ret = bmcv.drawLines(bm, start_points, end_points, line_num,color, thickness)

stft
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

实现对信号的短时傅里叶变换（STFT）。

**注意：请查询《BMCV开发参考手册/BMCV API》确认当前算子是否适配BM1684X。**

**接口形式:**
    .. code-block:: python

        stft(self, input_real: numpy.ndarray, input_imag: numpy.ndarray, real_input: bool, normalize: bool, n_fft: int, hop_len: int,
                pad_mode: int, win_mode: int) -> tuple[numpy.ndarray, numpy.ndarray]

        stft(self, input_real: Tensor, input_imag: Tensor, real_input: bool, normalize: bool, n_fft: int, hop_len: int,
                pad_mode: int, win_mode: int) -> tuple[Tensor, Tensor]
    
**参数说明:**

* input_real: numpy.ndarray 或者 Tensor
    输入信号的实数部分。

* input_imag: numpy.ndarray 或者 Tensor
    输入信号的虚数部分。

* real_input: bool
    是否仅使用实数输入的标志。

* normalize: bool
    是否对输出进行归一化的标志。

* n_fft: int
    STFT计算中使用的FFT点数。

* hop_len: int
    窗口滑动的步长。

* pad_mode: int
    输入信号的填充模式，0表示CONSTANT填充，1表示REFLECT填充。

* win_mode: int
    窗口函数的类型，0表示HANN窗，1表示HAMM窗。

**返回值说明:**

* result: tuple[numpy.ndarray, numpy.ndarray] 或者 tuple[Tensor, Tensor]
    返回输出的实数部分和虚数部分。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            random_array1 = np.random.rand(2, 4096).astype('float32')
            random_array2 = np.random.rand(2, 4096).astype('float32')
            bmcv = sail.Bmcv(handle)
            input_real = sail.Tensor(handle, random_array1, True)
            input_imag = sail.Tensor(handle, random_array2, True)
            real_input = False
            normalize = True
            n_fft = 1024
            hop_len = 256
            pad_mode = 0  # 填充模式示例
            win_mode = 1  # 窗口类型示例
            stft_R, stft_I = bmcv.stft(input_real, input_imag, real_input, normalize, n_fft, hop_len, pad_mode, win_mode)

istft
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

实现对信号的逆短时傅里叶变换（ISTFT）。

**注意：请查询《BMCV开发参考手册/BMCV API》确认当前算子是否适配BM1684X。**

**接口形式:**
    .. code-block:: python

        istft(self, input_real: numpy.ndarray, input_imag: numpy.ndarray, real_input: bool, normalize: bool, L: int, hop_len: int,
            pad_mode: int, win_mode: int) -> tuple[numpy.ndarray, numpy.ndarray]:

        istft(self, input_real: Tensor, input_imag: Tensor, real_input: bool, normalize: bool, L: int, hop_len: int,
            pad_mode: int, win_mode: int) -> tuple[Tensor, Tensor]:

**参数说明:**

* input_real: numpy.ndarray 或者 Tensor
    输入信号的实数部分。

* input_imag: numpy.ndarray 或者 Tensor
    输入信号的虚数部分。

* real_input: bool
    输出的信号是否为实数， false 为复数， true 为实数。

* normalize: bool
    是否对输出进行归一化。

* L: int
    原始时域信号的长度。

* hop_len: int
    窗口滑动的步长，必须与STFT计算时使用的值相同。

* pad_mode: int
    输入信号的填充模式，必须与STFT计算时使用的值相同。

* win_mode: int
    窗口函数的类型，必须与STFT计算时使用的值相同。

**返回值说明:**

* result: tuple[numpy.ndarray, numpy.ndarray] 或者 tuple[Tensor, Tensor]
    返回输出的实数部分和虚数部分。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            random_array1 = np.random.rand(2, 513, 17).astype('float32')
            random_array2 = np.random.rand(2, 513, 17).astype('float32')
            bmcv = sail.Bmcv(handle)
            input_real = sail.Tensor(handle, random_array1, True)
            input_imag = sail.Tensor(handle, random_array2, True)
            real_input = False
            normalize = True
            L = 4096
            hop_len = 256
            pad_mode = 0  # 填充模式示例
            win_mode = 1  # 窗口类型示例
            istft_R, istft_I = bmcv.istft(input_real, input_imag, real_input, normalize, 4096, hop_len, pad_mode, win_mode)

bmcv_overlay
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

图片上添加带透明通道的水印图。仅支持BM1688和CV186AH。

**接口形式:**
    .. code-block:: python

        def bmcv_overlay(self, image: BMImage, overlay_info: list[list[int]], overlay_image: list[BMImage]) -> int:   

        
**参数说明:**

* image : sail.BMImage

输入/输出图片。

* overlay_info: list[list[int]]

一组水印图的位置大小信息，格式是[x,y,w,h]

* overlay_image: list[BMImage]

一组水印图，目前仅支持sail.Format.FORMAT_ARGB_PACKED格式

**返回值说明:**

* ret: int

成功后返回0。

**注：**

需要自行确保overlay_info中的所有矩形位置不重叠

**示例代码1:**

直接读入RGBA图片作为水印

    .. code-block:: python

        import sophon.sail as sail
        import cv2

        if __name__ == '__main__':

            handle = sail.Handle(0)

            decoder = sail.Decoder("pics/1.jpg")
            image_org = decoder.read(handle)

            bmcv = sail.Bmcv(handle)

            buffer = cv2.imread("pics/icon.png", cv2.IMREAD_UNCHANGED)
            buffer_x = 100
            buffer_y = 50
            buffer_w = 100
            buffer_h = 30

            buffer_x1 = 500
            buffer_y1 = 200
            buffer_w1 = 60
            buffer_h1 = 70

            buffer = cv2.resize(buffer, (buffer_w, buffer_h))
            buffer1 = cv2.resize(buffer, (buffer_w1, buffer_h1))

            overlay_images = []
            overlay_info = []
            overlay_images.append(sail.BMImage(handle, buffer, buffer_h, buffer_w, sail.Format.FORMAT_ARGB_PACKED, sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE))
            overlay_info.append([buffer_x,buffer_y,buffer_w, buffer_h])
            overlay_images.append(sail.BMImage(handle, buffer1, buffer_h1, buffer_w1, sail.Format.FORMAT_ARGB_PACKED, sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE))
            overlay_info.append([buffer_x1,buffer_y1,buffer_w1, buffer_h1])

            ret = bmcv.bmcv_overlay(image_org, overlay_info, overlay_images)

            ret = bmcv.imwrite("overlayed.jpg", image_org)


**示例代码2:**

用其他库生成中文字的水印图，用overlay画在指定位置，间接实现puttext功能

    .. code-block:: python
      
        import sophon.sail as sail
        from PIL import Image, ImageDraw, ImageFont

        if __name__ == '__main__':
            # 获取原图
            handle = sail.Handle(0)
            bmcv = sail.Bmcv(handle)
            decoder = sail.Decoder("pics/1.jpg")
            org_image = decoder.read(handle)

            # 字体基本信息
            watermark_w, watermark_h = 60, 30
            text = ["人", "车", "狗"]
            text_color = (255, 0, 0, 255) 
            font = ImageFont.truetype("fonts/simhei.ttf", watermark_h, encoding="utf-8")

            # 生成标签水印图
            overlay_images = []
            for i in range(len(text)):
                blank_image = Image.new('RGBA', (watermark_w, watermark_h), (255,255,255,0))
                draw = ImageDraw.Draw(blank_image)
                draw.text((0,0), text[i], fill=text_color, font=font)
                # blank_image.save("watermark.png")
                buffer = blank_image.tobytes()
                overlay_images.append(sail.BMImage(handle, buffer, watermark_h, watermark_w, sail.Format.FORMAT_ARGB_PACKED, sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE))

            # 经过推理等其他过程
            # 模拟获得的水印图位置
            overlay_info = []
            for i in range(len(text)):
                x = int(i*org_image.width()/len(text))
                y = int(org_image.height()/2)
                overlay_info.append([x, y, watermark_w, watermark_h])

            ret = bmcv.bmcv_overlay(org_image, overlay_info, overlay_images)
            ret = bmcv.imwrite("overlayed.jpg", org_image)

