sail.Bmcv
_________

\_\_init\_\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: python

        def __init__(self, handle: sail.Handle)
            
**Parameters:**
* handle : sail.Handle

Handle instance


bm_image_to_tensor
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Convert image to tensor.

**Interface1:**
    .. code-block:: python

        def bm_image_to_tensor(self, 
                            image: sail.BMImage | sail.BMImageArray
                            ) -> sail.Tensor
            
**Parameters:**
* image : sail.BMImage | sail.BMImageArray

BMImage/BMImageArray instance

**Returns:**
Return sail.Tensor instance

**Sample:**
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

**Interface2:**
    .. code-block:: python

        def bm_image_to_tensor(self, 
                            image: sail.BMImage | sail.BMImageArray, 
                            tensor: sail.Tensor)
            
**Parameters:**
* image : sail.BMImage | sail.BMImageArray
                
BMImage/BMImageArray instance

* tensor : sail.Tensor

Tensor instance

    **Sample:**
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

Convert tensor to image.

**Interface:**
    .. code-block:: python

        def tensor_to_bm_image(self, 
                            tensor: sail.Tensor, 
                            bgr2rgb: bool = False,
                            layout: str = 'nchw') -> sail.BMImage
            
**Parameters:**
* tensor : sail.Tensor

Tensor instance

* bgr2rgb : bool 
  
Swap color channel, default: False

* layout : str

Layout of the input tensor

**Returns:**
Return BMImage instance

**Sample:**
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

**Interface:**
    .. code-block:: python

        def tensor_to_bm_image(self, 
                            tensor: sail.Tensor, 
                            format: sail.Format) -> sail.BMImage
            
**Parameters:**
* tensor : sail.Tensor

Tensor instance

* format : sail.Format 
  
Format of the BMImage

**Returns:**
Return BMImage instance

**Sample:**
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

**Interface:**
    .. code-block:: python

        def tensor_to_bm_image(self, 
                            tensor: sail.Tensor, 
                            img: sail.BMImage | sail.BMImageArray, 
                            bgr2rgb: bool = False,
                            layout: str = 'nchw')
            
**Parameters:**
* tensor : sail.Tensor

Tensor instance

* img : sail.BMImage

BMImage instance

* bgr2rgb : bool 
  
Swap color channel, default: False

**Sample:**
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

**Interface:**
    .. code-block:: python

        def tensor_to_bm_image(self, 
                            tensor: sail.Tensor, 
                            img: sail.BMImage | sail.BMImageArray, 
                            format: sail.Format)
            
**Parameters:**
* tensor : sail.Tensor

Tensor instance

* img : sail.BMImage

BMImage instance

* format : sail.Format 
  
Format of the BMImage

**Sample:**
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

Crop then resize an image or an image array.

**Interface:**
    .. code-block:: python

        def crop_and_resize(self,
                            input: sail.BMImage|sail.BMImageArray, 
                            crop_x0: int, 
                            crop_y0: int, 
                            crop_w: int, 
                            crop_h: int, 
                            resize_w: int, 
                            resize_h: int, 
                            resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST
                            )->sail.BMImage
            
**Parameters:**
* input : sail.BMImage|sail.BMImageArray, 

Input image or image array

* crop_x0 : int

Start point x of the crop window

* crop_y0 : int

Start point y of the crop window

* crop_w : int

Width of the crop window

* crop_h : int

Height of the crop window

* resize_w : int

Target width

* resize_h : int

Target height

* resize_alg : bmcv_resize_algorithm

Resize algorithm, default is BMCV_INTER_NEAREST

**Returns:**

* output : sail.BMImage

Output image or image array

**Sample:**
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
Crop an image or an image array with given window.

**Interface:**
    .. code-block:: python

        def crop(self, 
                input: sail.BMImage, 
                crop_x0: int, 
                crop_y0: int, 
                crop_w: int, 
                crop_h: int)-> sail.BMImage
            
**Parameters:**
* input : sail.BMImage 

Input image

* crop_x0 : int

Start point x of the crop window

* crop_y0 : int

Start point y of the crop window

* crop_w : int

Width of the crop window

* crop_h : int

Height of the crop window

**Returns:**

* output : sail.BMImage

Output image

**Sample:**
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

**Interface:**
    .. code-block:: python

        def crop(self, input, crop_x0, crop_y0, crop_w, crop_h):
            
**Parameters:**
* input : sail.BMImageArray 

Input image array

* crop_x0 : int

Start point x of the crop window

* crop_y0 : int

Start point y of the crop window

* crop_w : int

Width of the crop window

* crop_h : int

Height of the crop window

**Returns:**

* output : sail.BMImageArray

Output image array

**Sample:**
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

Resize an image or an image array

**Interface:**
    .. code-block:: python

        def resize(self, 
                input: sail.BMImage | sail.BMImageArray, 
                resize_w: int, 
                resize_h: int, 
                resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST)-> sail.BMImage
            
**Parameters:**
* input : sail.BMImage | sail.BMImageArray

Input image or image array

* resize_w : int

Target width

* resize_h : int

Target height

* resize_alg : bmcv_resize_algorithm

Resize algorithm, default is BMCV_INTER_NEAREST

**Returns:**

* output : sail.BMImage | sail.BMImageArray

Output image or image array

**Sample:**
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

Crop then resize an image or an image array using vpp

**Interface:**
    .. code-block:: python

        def vpp_crop_and_resize(self, 
                                input: sail.BMImage | sail.BMImageArray, 
                                crop_x0: int, 
                                crop_y0: int, 
                                crop_w: int, 
                                crop_h: int, 
                                resize_w: int, 
                                resize_h: int,
                                resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST)-> sail.BMImage
            
**Parameters:**
* input : sail.BMImage | sail.BMImageArray

Input image or image array

* crop_x0 : int

Start point x of the crop window

* crop_y0 : int

Start point y of the crop window

* crop_w : int

Width of the crop window

* crop_h : int

Height of the crop window

* resize_w : int

Target width

* resize_h : int

Target height

* resize_alg : bmcv_resize_algorithm

Resize algorithm, default is BMCV_INTER_NEAREST

**Returns:**

* output : sail.BMImage | sail.BMImageArray

Output image or image array

**Sample:**
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

Crop then resize an image or an image array using vpp.

**Interface:**
    .. code-block:: python

        def vpp_crop_and_resize_padding(self, 
                                        input: sail.BMImage | sail.BMImageArray, 
                                        crop_x0: int, 
                                        crop_y0: int, 
                                        crop_w: int, 
                                        crop_h: int, 
                                        resize_w: int, 
                                        resize_h: int, 
                                        padding: PaddingAtrr,
                                        resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST)-> sail.BMImage
            
**Parameters:**

* input : sail.BMImage | sail.BMImageArray

Input image or image array

* crop_x0 : int

Start point x of the crop window

* crop_y0 : int

Start point y of the crop window

* crop_w : int

Width of the crop window

* crop_h : int

Height of the crop window

* resize_w : int

Target width

* resize_h : int

Target height

* padding : PaddingAtrr

padding info

* resize_alg : bmcv_resize_algorithm

Resize algorithm, default is BMCV_INTER_NEAREST

**Returns:**
* output : sail.BMImage | sail.BMImageArray

Output image or image array

**Sample:**
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

Crop an image or an image array with given window using vpp.

**Interface:**
    .. code-block:: python

        def vpp_crop(self, 
                    input: sail.BMImage | sail.BMImageArray, 
                    crop_x0: int, 
                    crop_y0: int, 
                    crop_w: int, 
                    crop_h: int)-> sail.BMImage
            
**Parameters:**
* input : sail.BMImage | sail.BMImageArray

Input image or image array

* crop_x0 : int

Start point x of the crop window

* crop_y0 : int

Start point y of the crop window

* crop_w : int

Width of the crop window

* crop_h : int

Height of the crop window

**Returns:**
* output : sail.BMImage | sail.BMImageArray

Output image or image array

**Sample:**
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

Resize an image or an image array with interpolation of INTER_NEAREST using vpp.

**Interface:**
    .. code-block:: python

        def vpp_resize(self, 
                    input: sail.BMImage | sail.BMImageArray, 
                    resize_w: int, 
                    resize_h: int,
                    resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST)-> sail.BMImage | sail.BMImageArray
            
**Parameters:**

* input : sail.BMImage | sail.BMImageArray

Input image or image array

* resize_w : int

Target width

* resize_h : int

Target height

* resize_alg : bmcv_resize_algorithm

Resize algorithm, default is BMCV_INTER_NEAREST

**Returns:**

* output : sail.BMImage | sail.BMImageArray

Output image or image array

**Sample:**
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


**Interface:**
    .. code-block:: python

        def vpp_resize(self, 
                        input: sail.BMImage | sail.BMImageArray, 
                        output: sail.BMImage | sail.BMImageArray, 
                        resize_w: int, 
                        resize_h: int,
                        resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST)
           
**Parameters:**

* input : sail.BMImage | sail.BMImageArray

Input image

* output : sail.BMImage | sail.BMImageArray

Output image

* resize_w : int

Target width

* resize_h : int

Target height

* resize_alg : bmcv_resize_algorithm

Resize algorithm, default is BMCV_INTER_NEAREST

**Sample:**
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

Resize an image or an image array with interpolation of INTER_NEAREST using vpp.

**Interface:**
    .. code-block:: python

        def vpp_resize_padding(self, 
                            input: sail.BMImage | sail.BMImageArray, 
                            resize_w: int, 
                            resize_h: int, 
                            padding: PaddingAtrr,
                            resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST)-> sail.BMImage | sail.BMImageArray
   
**Parameters:**

* input : sail.BMImage | sail.BMImageArray

Input image or image array

* resize_w : int

Target width

* resize_h : int

Target height

* padding : PaddingAtrr

padding info

* resize_alg : bmcv_resize_algorithm

Resize algorithm, default is BMCV_INTER_NEAREST

**Returns:**

* output : sail.BMImage | sail.BMImageArray

Output image or image array

**Sample:**
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

Applies an affine transformation to an image or an image array.

**Interface:**
    .. code-block:: python

        def warp(self, 
                input: sail.BMImage | sail.BMImageArray, 
                matrix: 2d list,
                use_bilinear: int = 0,
                similar_to_opencv: bool = False)-> sail.BMImage | sail.BMImageArray
            
**Parameters:**

* input : sail.BMImage | sail.BMImageArray

Input image or image array

* matrix: 2d list

2x3 transformation matrix

* use_bilinear: int

Whether to use bilinear interpolation, default to 0 using nearest neighbor interpolation, 1 being bilinear interpolation

* similar_to_opencv: bool

Whether to use the interface aligning the affine transformation interface of OpenCV

**Returns:**

* output : sail.BMImage | sail.BMImageArray

Output image or image array

**Sample:**
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

**Interface:**
    .. code-block:: python

        def warp(self, 
                input: sail.BMImage | sail.BMImageArray, 
                output: sail.BMImage | sail.BMImageArray, 
                matrix: 2d list,
                use_bilinear: int = 0,
                similar_to_opencv: bool = False)-> int
            
**Parameters:**

* input : sail.BMImage | sail.BMImageArray

Input image or image array

* output : sail.BMImage | sail.BMImageArray

Output image or image array

* matrix: 2d list

2x3 transformation matrix

* use_bilinear: int

Whether to use bilinear interpolation, default to 0 using nearest neighbor interpolation, 1 being bilinear interpolation

* similar_to_opencv: bool

Whether to use the interface aligning the affine transformation interface of OpenCV

**Returns:**

0 for success and others for failure

**Sample:**
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

Applies a linear transformation to an image or an image array.

**Interface:**
    .. code-block:: python

        def convert_to(self, 
                        input: sail.BMImage | sail.BMImageArray, 
                        alpha_beta: tuple)-> sail.BMImage | sail.BMImageArray
            
**Parameters:**

* input : sail.BMImage | sail.BMImageArray
 
Input image or image array

* alpha_beta: tuple

(a0, b0), (a1, b1), (a2, b2) factors

**Returns:**

* output : sail.BMImage | sail.BMImageArray

Output image or image array

**Sample:**
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

**Interface:**
    .. code-block:: python

        def convert_to(self, 
                        input: sail.BMImage | sail.BMImageArray, 
                        output: sail.BMImage | sail.BMImageArray, 
                        alpha_beta: tuple)
           
**Parameters:**

* input : sail.BMImage | sail.BMImageArray

Input image or image array

* output : sail.BMImage | sail.BMImageArray

Output image or image array

* alpha_beta: tuple

(a0, b0), (a1, b1), (a2, b2) factors

**Sample:**
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

Convert an image or an image array from YUV to BGR.

**Interface:**
    .. code-block:: python

        def yuv2bgr(self, input: sail.BMImage | sail.BMImageArray)-> sail.BMImage | sail.BMImageArray
            
**Parameters:**

* input : sail.BMImage | sail.BMImageArray

Input image or image array

**Returns:**

* output : sail.BMImage | sail.BMImageArray

Output image or image array

**Sample:**
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

Draw a rectangle on input image.

**Interface:**
    .. code-block:: python

        def rectangle(self, 
                    image: sail.BMImage, 
                    x0: int, 
                    y0: int, 
                    w: int, 
                    h: int, 
                    color: tuple, 
                    thickness: int = 1)-> int
        
        def rectangle(self, 
                        image: sail.bm_image, 
                        x0: int, 
                        y0: int, 
                        w: int, 
                        h: int, 
                        color: tuple, 
                        thickness: int = 1)-> int
                    

**Parameters:**

* image : sail.BMImage | sail.bm_image

Input image

* x0 : int

Start point x of rectangle

* y0 : int

Start point y of rectangle

* w : int

Width of rectangle

* h : int

Height of rectangle

* color : tuple

Color of rectangle

* thickness : int

Thickness of rectangle, default: 1

**Returns:**

* process_status : int

0 for success and others for failure

**Sample:**
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

fillRectangle
>>>>>>>>>>>>>>>>>>

Fill a rectangle on input image.

**Interface:**
    .. code-block:: python

        def fillRectangle(self, 
                    image: sail.BMImage, 
                    x0: int, 
                    y0: int, 
                    w: int, 
                    h: int, 
                    color: tuple)-> int

        def fillRectangle(self,    
                    image: sail.bm_image, 
                    x0: int, 
                    y0: int, 
                    w: int, 
                    h: int, 
                    color: tuple)-> int

            
**Parameters:**

* image : sail.BMImage | sail.bm_image

Input image

* x0 : int

Start point x of rectangle

* y0 : int

Start point y of rectangle

* w : int

Width of rectangle

* h : int

Height of rectangle

* color : tuple

Color of rectangle


**Returns:**

* process_status : int

0 for success and others for failure

**Sample:**
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

imwrite
>>>>>>>>>>>>>>>>>

Save the image to the specified file.

**Interface:**
    .. code-block:: python

        def imwrite(self, file_name: str, image: sail.BMImage)-> int

        def imwrite(self, file_name: str, image: sail.bm_image)-> int
             
**Parameters:**

* file_name : str

Name of the file

* output : sail.BMImage | sail.bm_image

Image to be saved

**Returns:**

* process_status : int

0 for success and others for failure

**Sample:**
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


imread
>>>>>>>>>>>>>>>>>

Read and decode one image files and supports hard decoding only for JPEG baseline format. For other formats, such as PNG and BMP, soft decoding is used.
The returned BMImage for JPEG baseline images keeps YUV color space, and the pixel format depends on the sampling information in the file like YUV420. 
The returned BMImage for other formats will maintain the corresponding color space of their input.

**Interface:**
    .. code-block:: python

        def imread(self, filename: str) -> BMImage
             
**Parameters:**

* filename : str

Name of file to be read.

**Returns:**

* output : sail.BMImage

The decoded image.

**Sample:**
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

Get Handle instance.

**Interface:**
    .. code-block:: python

        def get_handle(self)-> sail.Handle

**Returns:**

* handle: sail.Handle

Handle instance

**Sample:**
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

Crop then resize an image.

**Interface:**
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
                                    resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST
                                    )-> sail.BMImage

**Parameters:**

* input : sail.BMImage

Input image

* crop_x0 : int

Start point x of the crop window

* crop_y0 : int

Start point y of the crop window

* crop_w : int

Width of the crop window

* crop_h : int

Height of the crop window

* resize_w : int

Target width

* resize_h : int

Target height

* padding : PaddingAtrr

padding info

* resize_alg : bmcv_resize_algorithm

Resize algorithm, default is BMCV_INTER_NEAREST

**Returns:**

* output : sail.BMImage

Output image

**Sample:**
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
Convert input to output format or convert an image to BGR PLANAR format. 

**Interface:**
    .. code-block:: python

        def convert_format(self, input: sail.BMImage, output: sail.BMImage)

**Parameters:**

* input : sail.BMImage

BMimage instance

* output : sail.BMImage

output image

**Sample:**
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

**Interface:**
    .. code-block:: python

        def convert_format(self, input: sail.BMImage)-> sail.BMImage

**Parameters:**

* input : sail.BMImage

BMimage instance

**Returns:**

* output : sail.BMImage

output image

**Sample:**
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

Convert input to output format or convert an image to BGR PLANAR format using vpp. 

**Interface:**
    .. code-block:: python

        def vpp_convert_format(self, input: sail.BMImage, output: sail.BMImage)
            
**Parameters:**

* input : sail.BMImage

BMimage instance

* output : sail.BMImage

output image

**Sample:**
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

**Interface:**
    .. code-block:: python

        def vpp_convert_format(self, input): 
            
**Parameters:**

* input : sail.BMImage

BMimage instance

**Returns:**

* output : sail.BMImage

output image

**Sample:**
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

Draws a text on the image.

Supported pixel format for input BMImage: 
FORMAT_GRAY, FORMAT_YUV420P, FORMAT_YUV422P, FORMAT_YUV444P, FORMAT_NV12, 
FORMAT_NV21, FORMAT_NV16, FORMAT_NV61

**Interface:**
    .. code-block:: python
        
        def putText(self, 
                    input: sail.BMImage, 
                    text: str, 
                    x: int, 
                    y: int, 
                    color: tuple, 
                    fontScale: float, 
                    thickness: int)-> int
        
        def putText(self, 
                    input: sail.bm_image, 
                    text: str, 
                    x: int, 
                    y: int, 
                    color: tuple, 
                    fontScale: float, 
                    thickness: int)-> int
            
**Parameters:**

* input : sail.BMImage

BMimage instance

* text: str

Text to write on an image.

* x: int

Start point x

* y: int

Start point y

* color : tuple

Color of text

* fontScale : float

Size of font

* thickness : int

Thickness of text

**Returns:**

* process_status : int

0 for success and others for failure

**Sample:**
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

Calculates the weighted sum of two images

**Interface:**
    .. code-block:: python
        
        def image_add_weighted(self, 
                            input0: sail.BMImage, 
                            alpha: float, 
                            input1: float, 
                            beta: float, 
                            gamma: float, 
                            output: BMImage)

**Parameters:**

* input0 : sail.BMImage

BMimage instance.

* alpha : float

alpha instance.

* input1 : sail.BMImage

BMImage instance.

* beta: float

beta instance.

* gamma: float

gamma instance.

* output: BMImage

result BMImage, output = input1 * alpha + input2 * beta + gamma.

**Sample1:**
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

**Interface:**
    .. code-block:: python

        def image_add_weighted(self, 
                                input0: sail.BMImage, 
                                alpha: float, 
                                input1: sail.BMImage, 
                                beta: float, 
                                gamma: float)-> BMImage

**Parameters:**

* input0 : sail.BMImage

BMimage instance.

* alpha : float

alpha instance.

* input1 : sail.BMImage

BMImage instance.

* beta: float

beta instance.

* gamma: float

gamma instance.

**Returns:**

* output: BMImage

result BMImage, output = input1 * alpha + input2 * beta + gamma.

**Sample2:**
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

Copy the input to the output.

**Interface:**
    .. code-block:: python

        def image_copy_to(self, 
                        input: BMImage|BMImageArray, 
                        output: BMImage|BMImageArray, 
                        start_x: int, 
                        start_y: int)

**Parameters:**

* input: BMImage|BMImageArray

Input image or image array.

* output: BMImage|BMImageArray

Output image or image array.

* start_x: int

Point start x.

* start_y: int

Point start y.

**Sample:**
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

Copy the input to the output width padding.

**Interface:**
    .. code-block:: python
    
        def image_copy_to_padding(self, 
                                input: BMImage|BMImageArray, 
                                output: BMImage|BMImageArray, 
                                padding_r: int, 
                                padding_g: int, 
                                padding_b: int, 
                                start_x: int, 
                                start_y: int)

**Parameters:**

* input: BMImage|BMImageArray

Input image or image array.

* output: BMImage|BMImageArray

Output image or image array.

* padding_r: int

r value for padding.

* padding_g: int

g value for padding.

* padding_b: int

b value for padding.

* start_x: int

point start x.

* start_y: int

point start y.

**Sample:**
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
Do nms use tpu.

**Note:** For details about whether this operator in current SDK supports BM1688, check the page "BMCV API" in 《Multimedia User Guide》. 

**Interface:**
    .. code-block:: python

        def nms(self, 
                input: [float, float, float, float, float], 
                threshold: float)-> numpy.ndarray[Any, numpy.dtype[numpy.float32]]

**Parameters:**

* input: [float, float, float, float, float]

input proposal array, shape must be (self, n,5) n<56000, \
proposal:[left,top,right,bottom,score].

* threshold: float

nms threshold.

**Returns:**

return nms result, numpy.ndarray[Any, numpy.dtype[numpy.float32]]

**Sample:**
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

Draw Point on input image.

**Interface:**
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

**Parameters:**

* image: BMImage | bm_image
  
Input image

* center: Tuple[int, int]

center of point, (point_x, point_y)

* color: Tuple[int, int, int], 

color of drawn, (b,g,r)

* radius: int

Radius of drawn

**Sample:**
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



warp_perspective
>>>>>>>>>>>>>>>>>>>>>

Applies a perspective transformation to an image.

**Interface:**
    .. code-block:: python

        def warp_perspective(input: BMImage, 
                            coordinate: [[int,int],[int,int],[int,int],[int,int]], 
                            output_width: int,  
                            output_height: int, 
                            format: bm_image_format_ext = FORMAT_BGR_PLANAR,  
                            dtype: bm_image_data_format_ext = DATA_TYPE_EXT_1N_BYTE, 
                            use_bilinear: int = 0 ) -> BMImage:
                  
**Parameters:**

* input: BMImage

Input image

coordinate: [[int,int],[int,int],[int,int],[int,int]]

* Original coordinate, like(left_top.x, left_top.y), (right_top.x, right_top.y), \
    (left_bottom.x, left_bottom.y), (right_bottom.x, right_bottom.y)

* output_width: int

Output width

* output_height: int

Output height

* format : bm_image_format_ext

Output image format, Only FORMAT_BGR_PLANAR,FORMAT_RGB_PLANAR 

* dtype: bm_image_data_format_ext

Output image dtype, Only DATA_TYPE_EXT_1N_BYTE,DATA_TYPE_EXT_4N_BYTE

* use_bilinear: int

Bilinear use flag.

**Returns:**

Output image
            
**Sample:**
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

Convert bm_image_data_format_ext to bm_data_type_t

**Interface:**
    .. code-block:: python

        def get_bm_data_type((self, format: sail.ImgDtype) -> sail.Dtype

**Parameters:**

* format: sail.ImgDtype

The type that needs to be converted.

**Returns:**

* ret: sail.Dtype

The converted type.

**Sample:**
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
>>>>>>>>>>>>>>>>>>>>>>>>>>>

Convert bm_data_type_t to bm_image_data_format_ext

**Interface:**
    .. code-block:: python

        def get_bm_image_data_format(self, dtype: bm_data_type_t) -> bm_image_data_format_ext:

**Parameters:**

* dtype: sail.Dtype

The sail.Dtype that needs to be converted.

**Returns:**

* ret: sail.ImgDtype

The converted type.

**Sample:**
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

Load image from system memory

**Interface:**
    .. code-block:: python

        def imdecode(self, data_bytes: bytes) -> BMImage:

**Parameters:**

* data_bytes: bytes

image data bytes in system memory

**Returns:**

return decoded image

**Sample:**
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

Encode an BMimage and return the encoded data.

**Interface:**
    .. code-block:: python

        def imencode(self, ext: str, img: BMImage) -> numpy.ndarray:
          
**Parameters:**

* ext: str

Input parameter. Image encoding format, supported formats include ``".jpg"`` , ``".png"`` , etc.

* img: BMImage

Input parameter. Input BMImage, only FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE pictures are supported.

**Return:**

* ret: numpy.array

The data that is encoded and placed in system memory.

**Sample:**
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

1d or 2d fft (only real part, or real part and imaginary part)

**Note:** For details about whether this operator in current SDK supports BM1688, check the page "BMCV API" in 《Multimedia User Guide》. 

**Interface:**
    .. code-block:: python

        def fft(self, forward: bool, input_real: Tensor)-> list[Tensor]:
            
**Parameters:**

* forward: bool

positive transfer

* input_real: Tensor

input tensor


**Returns:**

return list[Tensor], The real and imaginary part of output
 
**Interface:**
    .. code-block:: python

        def fft(self, 
                forward: bool, 
                input_real: Tensor, 
                input_imag: Tensor) -> list[Tensor]:
            
**Parameters:**

* forward: bool

positive transfer

* input_real: Tensor

input tensor real part

* input_imag: Tensor

input tensor imaginary part


Returns:

return list[Tensor], The real and imaginary part of output


mat_to_bm_image
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
transform opencv mat to sail BMImage.

**Interface:**
    .. code-block:: python

        def mat_to_bm_image(self, mat: numpy.ndarray[numpy.uint8]) -> BMImage: 

**Parameters:**

* mat : numpy

input opencv mat.

**Returns:**

* ret: sail.BMImage

return sail.BMImage.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import cv2

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            bmcv = sail.Bmcv(handle)
            opencv_mat = cv2.imread(image_name)
            sail_bm_image = bmcv.mat_to_bm_image(opencv_mat)

**Interface:**
    .. code-block:: python

        def mat_to_bm_image(self, mat: numpy.ndarray[numpy.uint8], img: BMImage) -> int: 

**Parameters:**

* mat : numpy

input opencv mat.

* img : sail.BMImage

output sail.BMImage.

**Returns:**

* ret: int

returns 0 if success.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        import cv2

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path"
            bmcv = sail.Bmcv(handle)
            opencv_mat = cv2.imread(image_name)
            BMimg2 = sail.BMImage()
            ret = bmcv.mat_to_bm_image(opencv_mat,BMimg2)
            
polylines
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

It can be realized to draw one or more line segments on an image, 
so that the function of drawing polygons can be realized, and the color and width of the line can be specified.


**Interface:**
    .. code-block:: python

        def polylines(self, image: BMImage, pts: list[list[tuple(int, int)]], isClosed: bool, color: tuple(int, int, int), thickness: int = 1, shift: int = 0) -> int:


**Parameters:**


* img : sail.BMImage

Input BMImage.

* pts : list[list[tuple(int, int)]]

The starting point and end point coordinates of the line segment, multiple coordinate points can be entered. The upper left corner of the image is the origin, 
extending to the right in the x direction and extending down in the y direction.

* isClosed : bool
  
Whether the graph is closed.

* color : tuple(int, int, int)

The color of the line is the value of the three RGB channels.

* thickness : int 

The width of the lines is recommended to be even for YUV format images.

* shift : int

Polygon scaling multiple. Default is not scaling. The scaling factor is(1/2)^shift。


**Returns:**

* ret: int

returns 0 if success.

**Sample:**
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

This interface is used to print one or more mosaics on an image.

**Interface:**
    .. code-block:: python

        def mosaic(self, mosaic_num: int, img: BMImage, rects: list[list[int,int,int,int]], is_expand:int) -> int


**Parameters:**

* mosaic_num : int

Number of mosaics, length of list in rects.

* img : sail.BMImage

Input BMImage.

* rects : list[list[int,int,int,int]]

Multiple Mosaic positions, the parameters in each element in the list are 
[Mosaic at X-axis start point, Mosaic at Y-axis start point, Mosaic width, Mosaic height].

* is_expand : int
  
Whether to expand the column. 
A value of 0 means that the column is not expanded, 
and a value of 1 means that a macro block (8 pixels) is expanded around the original Mosaic.


**Returns:**

* ret: int

returns 0 if success.

**Sample:**

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

This interface is used for image Gaussian filtering.
**Note: The previous SDK does not support BM1684X. 
For details about whether the current SDK supports BM1684X, check the page "BMCV API" in《Multimedia User Guide》. ***

**Interface:**
    .. code-block:: python
        
        def gaussian_blur(self, input: BMImage, kw: int, kh : int, sigmaX : float, sigmaY : float = 0.0) -> BMImage: 


**Parameters:**

* input : sail.BMImage

Input BMImage.

* kw : int

The size of kernel in the width direction.

* kh : int
  
The size of kernel in the height direction.

* sigmaX : float

Gaussian kernel standard deviation in the X direction.

* sigmaY : float

Gaussian kernel standard deviation in the Y direction.Default is 0, 
which means that it is the same standard deviation as the Gaussian kernel in the X direction.

**Returns:**

* output : sail.BMImage

Returns a Gaussian filtered image.

**Sample:**
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

The interface can realize the transposition of image width and height.

**Interface1:**
    .. code-block:: python

        def transpose(self, src: sail.BMImage) -> sail.BMImage:


**Parameters1:**

* src : sail.BMImage

Input BMImage.


**Returns2:**

* output: sail.BMImage:

output sail.BMImage.


**Interface2:**
    .. code-block:: python

        def transpose(self, src: sail.BMImage, dst: sail.BMImage) -> int:


**Parameters2:**

* src : sail.BMImage

Input BMImage.

* dst : sail.BMImage

output sail.BMImage.

**Returns2:**

* ret : int

returns 0 if success.


**Sample:**
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



watermark_superpose
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Implement adding multiple watermarks to images

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

* Image: save BMImage

Input image

* Watername: string

Watermark file path

* Bitmap_type: int

Input parameters. Watermark type, a value of 0 indicates that the watermark is an 8-bit data type (with transparency information), and a value of 1 indicates that the watermark is a 1-bit data type (without transparency information).

* Pitch: int

Input parameters. The number of bytes per line in a watermark file can be understood as the width of the watermark.

* Rects: list

Input parameters. Watermark position, including the starting point and width/height of each watermark.

* Color: tuple

Input parameters. The color of the watermark.

**Return value description:**

* Ret: int

Whether the return was successful

**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image_path1"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg1 = decoder.read(handle)# here is a sail.BMImage
            bmcv = sail.Bmcv(handle)
            water_name = 'your_watermark_path'
            ret = bmcv.watermark_superpose(BMimg1,water_name,0,117,[[0,0,117,79],[0,90,117,79]],[128,128,128])
            bmcv.imwrite("aafaa.jpg",BMimg1)

Sobel
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Soble operator for edge detection.

**Note:** For details about whether this operator in current SDK supports BM1684X/BM1688, check the page "BMCV API" in 《Multimedia User Guide》. 

**Interface1:**
    .. code-block:: python

        def Sobel(self, input: BMImage, output: BMImage, dx: int, dy: int, ksize: int = 3, scale: float = 1, delta: float = 0) -> int:

**Parameters1:**

* input

Input BMImage 

* output

Output BMImage

* dx

Order of the derivative x.

* dy

Order of the derivative y

* ksize

ize of the extended Sobel kernel; it must be -1, 1, 3, 5, or 7. -1 means 3x3 Scharr filter will be used.

* scale

Optional scale factor for the computed derivative values; by default, no scaling is applied

* delta

Optional delta value that is added to the results prior to storing them in dst.

**Returns1:**

* ret: int

returns 0 if success.

**Interface2:**
    .. code-block:: python

        def Sobel(self, input: BMImage, dx: int, dy: int, ksize: int = 3, scale: float = 1, delta: float = 0) -> BMImage:

**Parameters2:**

* input

Input BMImage 

* dx

Order of the derivative x.

* dy

Order of the derivative y

* ksize

ize of the extended Sobel kernel; it must be -1, 1, 3, 5, or 7. -1 means 3x3 Scharr filter will be used.

* scale

Optional scale factor for the computed derivative values; by default, no scaling is applied

* delta

Optional delta value that is added to the results prior to storing them in dst.

**Returns2:**

* output: BMImage

returns porcessed BMImage.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            bmcv = sail.Bmcv(handle)
            bmimg = sail.BMImage()
            decoder = sail.Decoder("your_img.jpg",True,tpu_id)
            bmimg = decoder.read(handle)

            print(bmimg.format())
            output = bmcv.Sobel(bmimg, 1, 1)
            bmcv.imwrite("out.jpg",output)



drawLines
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Draws multiple lines on a given image.

**Note:** For details about whether this operator in current SDK supports BM1684X, check the page "BMCV API" in 《Multimedia User Guide》. 

**Interface1:**
    .. code-block:: python

        def drawLines(self, image: BMImage, start_points: list[tuple[int, int]], end_points: list[tuple[int, int]], line_num: int, color: tuple[int, int, int], thickness: int) -> int:


**Parameters:**

* image : sail.BMImage

Input image.

* start_points : list[tuple[int, int]]

A list of starting point coordinates for the line segments, with the top-left corner of the image as the origin, extending to the right for the x-direction and down for the y-direction.

* end_points : list[tuple[int, int]]

A List of ending point coordinates for the lines, which must have the same length as the list of starting points.

* line_num : int

The number of line segments, which must be the same as the length of the starting and ending points lists.

* color : tuple[int, int, int]

The color of the line segments, in RGB format.

* thickness : int 

The thickness of the line segments.

**Returns:**

* ret: int

Returns 0 upon success.

**Sample:**
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

Short-Time Fourier Transform(STFT)

**Note:** For details about whether this operator in current SDK supports BM1684X, check the page "BMCV API" in 《Multimedia User Guide》. 

**Interface1:**
    .. code-block:: python

        stft(self, input_real: numpy.ndarray, input_imag: numpy.ndarray, real_input: bool, normalize: bool, n_fft: int, hop_len: int,
                pad_mode: int, win_mode: int) -> tuple[numpy.ndarray, numpy.ndarray]

        stft(self, input_real: Tensor, input_imag: Tensor, real_input: bool, normalize: bool, n_fft: int, hop_len: int,
                pad_mode: int, win_mode: int) -> tuple[Tensor, Tensor]

**Parameters:**

* input_real: numpy.ndarray or Tensor
    The real part of the input signal.

* input_imag: numpy.ndarray or Tensor
    The imaginary part of the input signal.

* real_input: bool
    A flag indicating whether to use only real input.

* normalize: bool
    A flag indicating whether to normalize the output.

* n_fft: int
    The number of FFT points used in the STFT computation.

* hop_len: int
    The step size for sliding the window.

* pad_mode: int
    The padding mode for the input signal. 0 indicates CONSTANT padding, and 1 indicates REFLECT padding.

* win_mode: int
    The type of window function. 0 represents the HANN window, and 1 represents the HAMM window.

**Returns:**

* result: tuple[numpy.ndarray, numpy.ndarray]
    Returns the real part and imaginary part of the output.

**Sample:**
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
            pad_mode = 0
            win_mode = 1
            stft_R, stft_I = bmcv.stft(input_real, input_imag, real_input, normalize, n_fft, hop_len, pad_mode, win_mode)

istft
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Inverse Short-Time Fourier Transform(ISTFT)

**Note:** For details about whether this operator in current SDK supports BM1684X, check the page "BMCV API" in 《Multimedia User Guide》. 

**Interface1:**
    .. code-block:: python

        istft(self, input_real: numpy.ndarray, input_imag: numpy.ndarray, real_input: bool, normalize: bool, L: int, hop_len: int,
            pad_mode: int, win_mode: int) -> tuple[numpy.ndarray, numpy.ndarray]:

        istft(self, input_real: Tensor, input_imag: Tensor, real_input: bool, normalize: bool, L: int, hop_len: int,
            pad_mode: int, win_mode: int) -> tuple[Tensor, Tensor]:

**Parameters:**

* input_real: numpy.ndarray or Tensor
    The real part of the input signal.

* input_imag: numpy.ndarray or Tensor
    The imaginary part of the input signal.

* real_input: bool
    Indicates whether the output signal is real; false for complex, true for real.

* normalize: bool
    Whether to normalize the output.

* L: int
    The length of the original time-domain signal.

* hop_len: int
    The step size for sliding the window; must match the value used during STFT computation.

* pad_mode: int
    The padding mode for the input signal; must match the value used during STFT computation.

* win_mode: int
    The type of window function; must match the value used during STFT computation.

**Returns:**

* result: tuple[numpy.ndarray, numpy.ndarray]
    Returns the real part and imaginary part of the output.

**Sample:**
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
            pad_mode = 0
            win_mode = 1
            istft_R, istft_I = bmcv.istft(input_real, input_imag, real_input, normalize, 4096, hop_len, pad_mode, win_mode)

bmcv_overlay
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Add a watermark with a transparent channel to the image. Only support BM1688, CV186AH。

**Interface:**
    .. code-block:: python

        def bmcv_overlay(self, image: BMImage, overlay_info: list[list[int]], overlay_image: list[BMImage]) -> int:

        
**Parameters:**

* image : sail.BMImage

input/output image

* overlay_info: list[list[int]]

The position and size information of a set of watermarks in the format of [x,y,w,h]

* overlay_image: list[BMImage]

A group of watermark, only support sail.Format.FORMAT_ARGB_PACKED

**Returns:**

* ret: int

0 for success

**Note：**

You need to make sure that all rectangles in overlay_info do not overlap

**Sample1:**

Read RGBA images directly as watermarks

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


**Sample2:**

Use other libraries to generate the watermark of Chinese text, and use overlay to draw in the designated position to realize the puttext function indirectly

    .. code-block:: python
      
        import sophon.sail as sail
        from PIL import Image, ImageDraw, ImageFont

        if __name__ == '__main__':
            # get original image
            handle = sail.Handle(0)
            bmcv = sail.Bmcv(handle)
            decoder = sail.Decoder("pics/1.jpg")
            org_image = decoder.read(handle)

            # the info of font
            watermark_w, watermark_h = 60, 30
            text = ["人", "车", "狗"]
            text_color = (255, 0, 0, 255) 
            font = ImageFont.truetype("fonts/simhei.ttf", watermark_h, encoding="utf-8")

            # get watermark of labels
            overlay_images = []
            for i in range(len(text)):
                blank_image = Image.new('RGBA', (watermark_w, watermark_h), (255,255,255,0))
                draw = ImageDraw.Draw(blank_image)
                draw.text((0,0), text[i], fill=text_color, font=font)
                # blank_image.save("watermark.png")
                buffer = blank_image.tobytes()
                overlay_images.append(sail.BMImage(handle, buffer, watermark_h, watermark_w, sail.Format.FORMAT_ARGB_PACKED, sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE))

            # after inference and other procedure
            # the position of the watermark map obtained by simulation
            overlay_info = []
            for i in range(len(text)):
                x = int(i*org_image.width()/len(text))
                y = int(org_image.height()/2)
                overlay_info.append([x, y, watermark_w, watermark_h])

            ret = bmcv.bmcv_overlay(org_image, overlay_info, overlay_images)
            ret = bmcv.imwrite("overlayed.jpg", org_image)

faiss_indexflatL2
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Calculate squared L2 distance between query vectors and database vectors, output the top topK L2sqr-values and the corresponding indices.

**Interface1:**
    .. code-block:: python

        faiss_indexflatL2(self, query_vecs: numpy.ndarray, query_vecs_L2norm: numpy.ndarray, database_vecs: numpy.ndarray, database_vecs_L2norm: numpy.ndarray,
                        vec_dims: int, query_vecs_nums: int, database_vecs_nums: int, topK: int) -> tuple[numpy.ndarray, numpy.ndarray]:
        
        faiss_indexflatL2(self, query_vecs: numpy.ndarray, query_vecs_L2norm: numpy.ndarray, database_vecs: Tensor, database_vecs_L2norm: Tensor,
                        vec_dims: int, query_vecs_nums: int, database_vecs_nums: int, topK: int) -> tuple[numpy.ndarray, numpy.ndarray]:

**Parameters:**

* query_vecs: numpy.ndarray
    The query vectors, the supported data type is only numpy.float32.

* query_vecs_L2norm: numpy.ndarray
    Calculates the sum of the square values of the elements in each row of the query vector, numpy.float32.

* database_vecs: numpy.ndarray or Tensor
    The database vectors, the supported data type is only numpy.float32 or sail.Dtype.BM_FLOAT32.

* database_vecs_L2norm: numpy.ndarray or Tensor
    Calculates the sum of the square values of the elements in each row of the database vector, numpy.float32 or sail.Dtype.BM_FLOAT32.

* vec_dims: int
    The dimension of the query vectors and database vectors.

* query_vecs_nums: int
    The numbers of the query vectors.

* database_vecs_nums: int
    The numbers of the database vectors.

* topK: int
    Get top topK values.

**Returns:**

* result: tuple[numpy.ndarray, numpy.ndarray]
    Returns the square value of the top topK best-matched L2 distance and its corresponding index.

**Sample1:**
    .. code-block:: python

        import numpy as np
        import sophon.sail as sail
        import time

        if __name__ == '__main__':
            # 1. database_vecs
            db_vecs = np.array([
                [-2.0, 0.0, 4.0],
                [-5.0, 3.0, -1.0],
                [1.0, 2.0, 4.0],
                [0.0, 5.0, -3.0],
                [2.0, 1.0, -4.0],
            ], dtype=np.float32)

            # 2. query_vecs
            query_vecs = np.array([
                [1.0, 2.0, 3.0]
            ], dtype=np.float16)

            # 3. test bmcv.faiss_indexflatL2
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            bmcv = sail.Bmcv(handle)

            db_vecs_square = np.square(db_vecs)
            db_vecs_l2norm = np.sum(db_vecs_square, axis=1)
            print("db_vecs_l2norm:", db_vecs_l2norm)

            query_vecs_square = np.square(query_vecs)
            query_vecs_l2norm = np.sum(query_vecs_square, axis=1)
            print("query_vecs_l2norm:", query_vecs_l2norm)

            start = time.time()
            similarity_L2, index_L2 = bmcv.faiss_indexflatL2(query_vecs, query_vecs_l2norm, db_vecs, db_vecs_l2norm, 3, 1, 5, 3)
            end = time.time()
            execution_time_milliseconds_L2 = (end - start) * 1000
            print(f"Execution time: {execution_time_milliseconds_L2:.6f} milliseconds")
            print("similarity_L2:", similarity_L2)
            print("index_L2:", index_L2)

**Sample2:**
    .. code-block:: python 
        
        import numpy as np
        import sophon.sail as sail
        import time

        if __name__ == '__main__':
            # 1. database_vecs
            db_vecs = np.array([
                [-2.0, 0.0, 4.0],
                [-5.0, 3.0, -1.0],
                [1.0, 2.0, 4.0],
                [0.0, 5.0, -3.0],
                [2.0, 1.0, -4.0],
            ], dtype=np.float32)

            # 2. query_vecs
            query_vecs = np.array([
                [1.0, 2.0, 3.0]
            ], dtype=np.float16)

            # 3. test bmcv.faiss_indexflatL2
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            bmcv = sail.Bmcv(handle)

            db_vecs_square = np.square(db_vecs)
            db_vecs_l2norm = np.sum(db_vecs_square, axis=1)
            print("db_vecs_l2norm:", db_vecs_l2norm)

            query_vecs_square = np.square(query_vecs)
            query_vecs_l2norm = np.sum(query_vecs_square, axis=1)
            print("query_vecs_l2norm:", query_vecs_l2norm)
            
            # 4. database_vecs and db_vecs_l2norm to sail::Tensor
            db_tensor = sail.Tensor(handle, db_vecs, False)
            db_tensor_l2norm = sail.Tensor(handle, db_vecs_l2norm, False)
            
            start = time.time()
            similarity_L2, index_L2 = bmcv.faiss_indexflatL2(query_vecs, query_vecs_l2norm, db_tensor, db_tensor_l2norm, 3, 1, 5, 3)
            end = time.time()
            execution_time_milliseconds_L2 = (end - start) * 1000
            print(f"Execution time: {execution_time_milliseconds_L2:.6f} milliseconds")
            print("similarity_L2:", similarity_L2)
            print("index_L2:", index_L2)

faiss_indexflatIP
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Calculate inner product distance between query vectors and database vectors, output the top K IP-values and the corresponding indices.

**Interface:**
    .. code-block:: python

        faiss_indexflatIP(self, query_vecs: numpy.ndarray, database_vecs: numpy.ndarray,
                        vec_dims: int, query_vecs_nums: int, database_vecs_nums: int, topK: int) -> tuple[numpy.ndarray, numpy.ndarray]:
        
        faiss_indexflatIP(self, query_vecs: numpy.ndarray, database_vecs: Tensor,
                        vec_dims: int, query_vecs_nums: int, database_vecs_nums: int, topK: int) -> tuple[numpy.ndarray, numpy.ndarray]:

**Parameters:**

* query_vecs: numpy.ndarray
    The query vectors, the supported data type is only numpy.float32.

* database_vecs: numpy.ndarray or Tensor
    The database vectors, the supported data type is only numpy.float32 or sail.Dtype.BM_FLOAT32.

* vec_dims: int
    The dimension of the query vectors and database vectors.

* query_vecs_nums: int
    The numbers of the query vectors.

* database_vecs_nums: int
    The numbers of the database vectors.

* topK: int
    Get top topK values.

**Returns:**

* result: tuple[numpy.ndarray, numpy.ndarray]
    Returns the top topK best-matched inner product distance values and their corresponding indexes.

**Sample:**
    .. code-block:: python

        import numpy as np
        import sophon.sail as sail
        import time

        if __name__ == '__main__':
            # 1. database_vecs
            db_vecs = np.array([
                [-2.0, 0.0, 4.0],
                [-5.0, 3.0, -1.0],
                [1.0, 2.0, 4.0],
                [0.0, 5.0, -3.0],
                [2.0, 1.0, -4.0],
            ], dtype=np.float32)

            # 2. query_vecs
            query_vecs = np.array([
                [1.0, 2.0, 3.0]
            ], dtype=np.float16)

            # 3. test bmcv.faiss_indexflatIP
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            bmcv = sail.Bmcv(handle)

            db_tensor = sail.Tensor(handle, db_vecs, False)
            start = time.time()
            similarity_IP, index_IP = bmcv.faiss_indexflatIP(query_vecs, db_tensor, 3, 1, 5, 3)
            # similarity_IP, index_IP = bmcv.faiss_indexflatIP(query_vecs, db_vecs, 3, 1, 5, 3)
            end = time.time()
            execution_time_milliseconds_L2 = (end - start) * 1000
            print(f"Execution time: {execution_time_milliseconds_L2:.6f} milliseconds")
            print("similarity_IP:", similarity_IP)
            print("index_IP:", index_IP)

faiss_indexPQ_encode
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Perform PQ quantization encoding on the input vector and output the encoded vector.

**Interface:**
    .. code-block:: python

        faiss_indexPQ_encode(self, input_vecs: Tensor, centroids_vecs: Tensor,
                        encode_vecs_num: int, vec_dims: int, slice_num: int, centroids_num: int, IP_metric: int) -> Tensor:
        
        faiss_indexPQ_encode(self, input_vecs: numpy.ndarray, centroids_vecs: numpy.ndarray,
                        encode_vecs_num: int, vec_dims: int, slice_num: int, centroids_num: int, IP_metric: int) -> numpy.ndarray:
        
        faiss_indexPQ_encode(self, input_vecs: numpy.ndarray, centroids_vecs: Tensor, encoded_vecs: Tensor 
                        encode_vecs_num: int, vec_dims: int, slice_num: int, centroids_num: int, IP_metric: int) -> int:

**Parameters:**

* input_vecs: numpy.ndarray or Tensor
    Input vector to be encoded, data types are supported only numpy.float32 or sail.Dtype.BM_FLOAT32, encode_vecs_num * vec_dims.

* centroids_vecs: numpy.ndarray or Tensor
    centroids vector,  data types are supported only numpy.float32 or sail.Dtype.BM_FLOAT32, slice_num * centroids_num * (vec_dims / slice_num).

* encode_vecs_num: int
    The number of vectors to be encoded.

* vec_dims: int
    Dimension of the vector to be encoded.

* slice_num: int
    The number of slices of the original vector dimensions, for example the original vector dimension is 512, slice_num = 8, and each subvector dimension is 64.

* centroids_num: int
    The number of centroids.

* IP_metric: int
    0 indicates that the distance is calculated using L2, and 1 indicates that the distance is calculated using IP.

**Returns:**

* result: numpy.ndarray or Tensor 
    Output the encoded vector, numpy.ndarray (uint8) or Tensor (BM_UINT8).

**Sample:**
    .. code-block:: python

        import faiss
        import numpy as np
        import sophon.sail as sail

        encode_vecs_num = 3
        vec_dims = 64
        db_vecs_num = 10000
        slice_num = 8
        centroids_num = 256
        nbits = 8

        np.random.seed(666)
        data = np.random.rand(db_vecs_num, vec_dims).astype('float32')
        pq = faiss.ProductQuantizer(vec_dims, slice_num, nbits)
        pq.train(data)

        # get centroids
        centroids = faiss.vector_float_to_array(pq.centroids)
        print('centroids.shape: ', centroids.shape)
        np.save('pq_centroids_random.npy', centroids)

        # faiss PQ encode
        input_vector = np.random.rand(encode_vecs_num, vec_dims).astype('float32')
        faiss_PQ_encode = pq.compute_codes(input_vector)
        print('faiss_PQ_encode:\n', faiss_PQ_encode)
        print('faiss_PQ_encode shape:', faiss_PQ_encode.shape)

        # test sail bmcv.faiss_indexPQ_encode
        handle = sail.Handle(0)
        bmcv = sail.Bmcv(handle)

        centroids_vecs = np.load("pq_centroids_random.npy").astype(np.float32)
        print("centroids_vecs shape:", centroids_vecs.shape)
        print("centroids_vecs dtype:", type(centroids_vecs))

        input_tensor = sail.Tensor(handle, input_vector, False)
        centroids_tensor = sail.Tensor(handle, centroids_vecs, False)

        encode_tensor = bmcv.faiss_indexPQ_encode(input_tensor, centroids_tensor, encode_vecs_num, vec_dims, slice_num, centroids_num, 0)
        print('bmcv_faiss_indexPQ_encode:\n', encode_tensor.asnumpy())

        encode = bmcv.faiss_indexPQ_encode(input_vector, centroids_vecs, encode_vecs_num, vec_dims, slice_num, centroids_num, 0)
        print('bmcv_faiss_indexPQ_encode:\n', encode)

faiss_indexPQ_ADC
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

The distance table is calculated by the query vector and the clustering center (centroid) vector, and the encoded database vector searches the table to calculate the distance and sorts it, output the topK distances and the corresponding indices.

**Interface:**
    .. code-block:: python

        faiss_indexPQ_ADC(self, nxquery_vecs: numpy.ndarray, centroids_vecs: numpy.ndarray, nycodes_vecs: numpy.ndarray,
                        vec_dims: int, slice_num: int, centroids_num: int, database_vecs_num: int, query_vecs_num: int, topK: int, IP_metric: int) -> tuple[numpy.ndarray, numpy.ndarray]:
        
        faiss_indexPQ_ADC(self, nxquery_vecs: numpy.ndarray, centroids_vecs: Tensor, nycodes_vecs: Tensor,
                        vec_dims: int, slice_num: int, centroids_num: int, database_vecs_num: int, query_vecs_num: int, topK: int, IP_metric: int) -> tuple[numpy.ndarray, numpy.ndarray]:

**Parameters:**

* nxquery_vecs: numpy.ndarray
   query vectors, data types are supported only numpy.float32, and the size is query_vecs_nums * vec_dims.

* centroids_vecs: numpy.ndarray or Tensor
    centroids vectors, data types are supported only numpy.float32 or sail.Dtype.BM_FLOAT32, and the size is slice_num * centroids_num * (vec_dims / slice_num).

* nycodes_vecs: numpy.ndarray or Tensor
    Encoded database vector, data types are supported only numpy.uint8 or sail.Dtype.BM_UINT8, and the size is database_vecs_nums * slice_num.

* vec_dims: int
    dimension of the query vector.

* slice_num: int
    The number of slices of the original vector dimensions, for example the original vector dimension is 512, slice_num = 8, and each subvector dimension is 64.  
   
* centroids_num: int
    The number of centroids.

* database_vecs_nums: int
    The number of database vectors.

* query_vecs_nums: int
    The number of query vectors.

* topK: int
    Get top topK values.

* IP_metric: int
    0 indicates that the distance is calculated using L2, and 1 indicates that the distance is calculated using IP.

**Returns:**

* result: tuple[numpy.ndarray, numpy.ndarray]
    Returns the top topK best-matched distance values and their corresponding indexes.

**Sample:**
    .. code-block:: python

        import numpy as np
        import faiss
        import sophon.sail as sail

        np.random.seed(1024)
        vec_dims = 512
        database_vecs_num = 10000
        query_vecs_num = 1

        nydb_vecs = np.random.rand(database_vecs_num, vec_dims).astype('float32')
        nxquery_vecs = np.random.rand(query_vecs_num, vec_dims).astype('float32')

        slice_num = 8
        n_bits = 8
        index = faiss.IndexPQ(vec_dims, slice_num, n_bits)
        index.train(nydb_vecs)
        index.add(nydb_vecs)

        # save centroids
        centroids_faiss = index.pq.centroids
        centroids = faiss.vector_float_to_array(centroids_faiss)
        print('centroids.shape: ', centroids.shape)
        np.save('centroids_random.npy', centroids)

        # Compute PQ codes for the database vectors and save
        db_codes = index.pq.compute_codes(nydb_vecs)
        print('db_codes.shape:', db_codes.shape)
        np.save('db_codes.npy', db_codes)

        # faiss 
        D, I = index.search(nxquery_vecs, k=5)
        print("The index of faiss:\n", I)
        print("The distance of faiss:\n", D)

        # test faiss_indexPQ_ADC of sail
        handle = sail.Handle(0)
        bmcv = sail.Bmcv(handle)

        # get centroids
        centroids_vecs = np.load("centroids_random.npy").astype(np.float32)
        print("centroids_vecs shape:", centroids_vecs.shape)
        print("centroids_vecs dtype:", type(centroids_vecs))
        # get PQ codes for the database vectors
        nycodes_vecs = np.load("db_codes.npy")
        print("nycodes_vecs shape:", nycodes_vecs.shape)
        print("nycodes_vecs dtype:", type(nycodes_vecs))

        # to tensor
        centroids_tensor = sail.Tensor(handle, centroids_vecs, False)
        nycode_tensor = sail.Tensor(handle, nycodes_vecs, False)

        # D_, I_ = bmcv.faiss_indexPQ_ADC(nxquery_vecs, centroids_tensor, nycode_tensor, vec_dims, slice_num, 256, database_vecs_num, query_vecs_num, 5, 0)
        D_, I_ = bmcv.faiss_indexPQ_ADC(nxquery_vecs, centroids_vecs, nycodes_vecs, vec_dims, slice_num, 256, database_vecs_num, query_vecs_num, 5, 0)
        print("The index of sail:\n", I_)
        print("The distance of sail:\n", D_)

faiss_indexPQ_SDC
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

The Symmetric Distance Computation (SDC) lookup table is used to speed up the distance calculation between PQ encodings, output the topK distances and the corresponding indices.

**Interface:**
    .. code-block:: python

        faiss_indexPQ_SDC(self, nxcodes_vecs: numpy.ndarray, nycodes_vecs: numpy.ndarray, sdc_table: numpy.ndarray,
                        slice_num: int, centroids_num: int, database_vecs_num: int, query_vecs_num: int, topK: int, IP_metric: int) -> tuple[numpy.ndarray, numpy.ndarray]:
        
        faiss_indexPQ_SDC(self, nxcodes_vecs: numpy.ndarray, nycodes_vecs: Tensor, sdc_table: Tensor,
                        slice_num: int, centroids_num: int, database_vecs_num: int, query_vecs_num: int, topK: int, IP_metric: int) -> tuple[numpy.ndarray, numpy.ndarray]:

**Parameters:**

* nxcodes_vecs: numpy.ndarray
   Encoded query vectors, data types are supported only numpy.uint8, and the size is query_vecs_nums * slice_num.

* nycodes_vecs: numpy.ndarray or Tensor
    Encoded database vector, data types are supported only numpy.uint8 or sail.Dtype.BM_UINT8, and the size is database_vecs_nums * slice_num.

* sdc_table: numpy.ndarray or Tensor
    The Symmetric Distance Computation (SDC) lookup table, data types are supported only numpy.float32 or sail.Dtype.BM_FLOAT32, and the size is slice_num * centroids_num * centroids_num.

* slice_num: int
    The number of slices of the original vector dimensions, for example the original vector dimension is 512, slice_num = 8, and each subvector dimension is 64.  
   
* centroids_num: int
    The number of centroids.

* database_vecs_nums: int
    The number of database vectors.

* query_vecs_nums: int
    The number of query vectors.

* topK: int
    Get top topK values.

* IP_metric: int
    0 indicates that the distance is calculated using L2, and 1 indicates that the distance is calculated using IP.

**Returns:**

* result: tuple[numpy.ndarray, numpy.ndarray]
    Returns the top topK best-matched distance values and their corresponding indexes.

**Sample:**
    .. code-block:: python

        import numpy as np
        import faiss
        import sophon.sail as sail

        np.random.seed(1024)
        vec_dims = 512             # The dimension of the vectors.
        database_vecs_num = 10000  # The number of the database vectors.
        query_vecs_num = 1         # The number of the query vectors.

        # 1. Random database and query
        database = np.random.rand(database_vecs_num, vec_dims).astype('float32')
        query = np.random.rand(query_vecs_num, vec_dims).astype('float32')

        # 2. The parameters of PQ
        slice_num = 8       # Divide the vector of the vec_dims dimension into 8 subvectors
        n_bits = 8          # centroids = 256 = 2^n_bits

        # 3. Create a PQ index
        index = faiss.IndexPQ(vec_dims, slice_num, n_bits)
        index.train(database)  # Train the PQ index
        index.add(database)

        # 4. Set to SDC (Symmetric Distance Calculation)
        index.search_type = faiss.IndexPQ.ST_SDC

        # 5. Calculate sdc_table and save it 
        index.pq.compute_sdc_table()

        sdc_table = faiss.vector_float_to_array(index.pq.sdc_table)
        print('sdc_table shape:', sdc_table.shape)
        np.save('sdc_table.npy', sdc_table)

        # 6. Query operations using faiss
        topK = 5
        D, I = index.search(query, topK)
        print("The index of faiss:\n", I)
        print("The distance of faiss:\n", D)

        # Calculate and save the PQ encoding of the database vector (for sail test)
        db_codes = index.pq.compute_codes(database)
        print('db_codes.shape:', db_codes.shape)
        np.save('db_codes.npy', db_codes)

        # Calculate and save the PQ encoding of the query vectors (for sail test)
        query_code = index.pq.compute_codes(query)
        print('query_code.shape:', query_code.shape)
        np.save('query_code.npy', query_code)
        print("________________________________________________________________________________")

        # test faiss_indexPQ_SDC
        handle = sail.Handle(0)
        bmcv = sail.Bmcv(handle)

        # 1. get PQ codes of the database vectors
        nycodes_vecs = np.load("db_codes.npy")
        print("nycodes_vecs shape:", nycodes_vecs.shape)
        print("nycodes_vecs dtype:", nycodes_vecs.dtype)

        # 2. get PQ codes of the query vectors
        nxcodes_vecs = np.load('query_code.npy')
        print("nxcodes_vecs shape:", nxcodes_vecs.shape)
        print("nxcodes_vecs dtype:", nxcodes_vecs.dtype)

        # 3. get the SDC table
        SDC_tables = np.load('sdc_table.npy')

        D_, I_ = bmcv.faiss_indexPQ_SDC(nxcodes_vecs, nycodes_vecs, SDC_tables, slice_num, 256, database_vecs_num, query_vecs_num, topK, 0)
        print("The index of sail:\n", I_)
        print("The distance of sail:\n", D_)


