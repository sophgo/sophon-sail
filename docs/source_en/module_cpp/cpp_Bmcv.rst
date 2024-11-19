Bmcv
_________

Bmcv encapsulates commonly used image processing interfaces and supports hardware acceleration.

The constructor of Bmcv()
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Init Bmcv

**Interface:**
    .. code-block:: c

        Bmcv(Handle handle);
          

**Parameters:**

* handle: Handle

Specify the device handle used by Bmcv.


bm_image_to_tensor
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Convert BMImage/BMImageArray to Tensor.

**Interface 1:**
    .. code-block:: c

        void bm_image_to_tensor(BMImage &img, Tensor &tensor);

        Tensor bm_image_to_tensor(BMImage &img);
           

**Parameters 1:**

* image: BMImage

Input parameter. Image image that needs to be converted

* tensor: Tensor

The converted Tensor.

**Return 1:**

* tensor: Tensor

Returns the converted Tensor.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        using namespace std;  
        int main() {  
            int dev_id = 0;
            sail::Handle handle(dev_id); 
            string image_path = "your_image_path";  
            sail::Decoder decoder(image_path,false,dev_id);
            sail::BMImage img = decoder.read(handle);   
            sail::Bmcv bmcv(handle);
            sail::Tensor tensor = bmcv.bm_image_to_tensor(img);
            return 0;  
        }

**Interface 2:**
    .. code-block:: c

        def bm_image_to_tensor( 
                image: BMImageArray, 
                tensor) -> Tensor
           
            
**Parameters 2:**

* image: BMImageArray

Input parameters. Image data that needs to be converted.

* tensor: Tensor

Output parameters. The converted Tensor.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        using namespace std;  
        int main() {  
            int dev_id = 0;
            sail::Handle handle(dev_id); 
            string image_path = "your_image_path";  
            sail::Decoder decoder(image_path,false,dev_id);
            sail::BMImage img = decoder.read(handle);   
            sail::Bmcv bmcv(handle);
            sail::Tensor tensor(handle,{1920,1080},BM_FLOAT32,true,true);
            bmcv.bm_image_to_tensor(img,tensor);
            return 0;  
        }

tensor_to_bm_image
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Convert Tensor to BMImage/BMImageArray.

**Interface 1:**
    .. code-block:: c

        void tensor_to_bm_image(Tensor &tensor, BMImage &img, bool bgr2rgb=false, std::string layout = std::string("nchw"));

        void tensor_to_bm_image(Tensor &tensor, BMImage &img, bm_image_format_ext format_);

        BMImage tensor_to_bm_image(Tensor &tensor, bool bgr2rgb=false, std::string layout = std::string("nchw"));

        BMImage tensor_to_bm_image (Tensor &tensor, bm_image_format_ext format_);


**Parameters 1:**

* tensor: Tensor

Input parameters. The Tensor to be converted.

* img : BMImage

The converted image.

**Returns 1:**

* image : BMImage

Returns the converted image.


**Interface 2:**
    .. code-block:: c

        template<std::size_t N> void   bm_image_to_tensor (BMImageArray<N> &imgs, Tensor &tensor);
        template<std::size_t N> Tensor bm_image_to_tensor (BMImageArray<N> &imgs);
            

**Parameters 2:**

* tensor: Tensor

Input parameters. The Tensor to be converted.

* img : BMImage | BMImageArray

Output parameters. Returns the converted image.

**Returns 2:**

* image : Tensor

Return the converted tensor.

**Sample1:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/tensor.h>
        using namespace std;
        int main() {
            int tpu_id = 0;  
            sail::Handle handle(tpu_id);  
            std::string image_name = "your_image_path";  
            sail::Decoder decoder(image_name, true, tpu_id);  
            sail::BMImage BMimg = decoder.read(handle);  
            sail::Bmcv bmcv(handle);  
            sail::Tensor tensor = bmcv.bm_image_to_tensor(BMimg);
            sail::BMImage BMimg2 = bmcv.tensor_to_bm_image(tensor);
            return 0;
            }

**Sample2:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/tensor.h>
        using namespace std;
        int main() {
            int tpu_id = 0;  
            sail::Handle handle(tpu_id);  
            std::string image_name = "your_image_path";  
            sail::Decoder decoder(image_name, true, tpu_id);  
            sail::BMImage BMimg = decoder.read(handle);  
            sail::Bmcv bmcv(handle);  
            sail::Tensor tensor = bmcv.bm_image_to_tensor(BMimg);
            sail::BMImage new_img();
            bmcv.tensor_to_bm_image(tensor,new_img);
            return 0;
            }

crop_and_resize
>>>>>>>>>>>>>>>>>>>>>>

Crop then resize an image or an image array.

**Interface:**
    .. code-block:: c

        int crop_and_resize(
           BMImage                      &input,
           BMImage                      &output,
           int                          crop_x0,
           int                          crop_y0,
           int                          crop_w,
           int                          crop_h,
           int                          resize_w,
           int                          resize_h,
           bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

        BMImage crop_and_resize(
           BMImage                      &input,
           int                          crop_x0,
           int                          crop_y0,
           int                          crop_w,
           int                          crop_h,
           int                          resize_w,
           int                          resize_h,
           bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

        template<std::size_t N>
        int crop_and_resize(
            BMImageArray<N>              &input,
            BMImageArray<N>              &output,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

        template<std::size_t N>
        BMImageArray<N> crop_and_resize(
            BMImageArray<N>              &input,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

**Parameters:**

* input : BMImage | BMImageArray

The image or array of images to be processed.

* output : BMImage | BMImageArray

Processed image or image array.

* crop_x0 : int

The starting point of the cropping window on the x-axis.

* crop_y0 : int

The starting point of the cropping window on the y-axis.

* crop_w : int 

The width of the crop window.

* crop_h : int 

The height of the crop window.

* resize_w : int

The target width for image resize.

* resize_h : int

The target height for image resize.

* resize_alg : bmcv_resize_algorithm

Interpolation algorithm for image resize, default is bmcv_resize_algorithm.BMCV_INTER_NEAREST

**Returns :**

* ret: int

Returns 0 for success, others for failure.

* output : BMImage | BMImageArray

Returns the processed image or image array.

**Sample1:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg3 = bmcv.crop_and_resize(BMimg, 0, 0, BMimg.width(), BMimg.height(), 640, 640);
            return 0;
        }

**Sample2:**
    .. code-block:: c


        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            ssail::BMImage BMimg3;
            int ret = bmcv.crop_and_resize(BMimg, BMimg3,0, 0, BMimg.width(), BMimg.height(), 640, 640);
            return 0;
        }

crop
>>>>>>>>>

Crop the image.

**Interface:**
    .. code-block:: c

        int crop(
           BMImage                      &input,
           BMImage                      &output,
           int                          crop_x0,
           int                          crop_y0,
           int                          crop_w,
           int                          crop_h);

        
        BMImage crop(
           BMImage                      &input,
           int                          crop_x0,
           int                          crop_y0,
           int                          crop_w,
           int                          crop_h);

        template<std::size_t N>
        int crop(
            BMImageArray<N>              &input,
            BMImageArray<N>              &output,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h);

        template<std::size_t N>
        BMImageArray<N> crop(
            BMImageArray<N>              &input,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h);
            

**Parameters:**

* input : BMImage | BMImageArray

Input parameter. The image or array of images to be processed.

* output : BMImage | BMImageArray

Output parameter. Processed image or image array.

* crop_x0 : int

Input parameter. Start point x of the crop window.

* crop_y0 : int

Input parameter. Start point y of the crop window.

* crop_w : int 

Input parameter. Width of the crop window.

* crop_h : int 

Input parameter. Height of the crop window.

**Returns:**

* ret: int

Returns 0 for success, others for failure.

* output : BMImage | BMImageArray

Returns the processed image or image array.

**Sample1:**
    .. code-block:: c


        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg3 = bmcv.crop(BMimg,100,100,200,200);
            return 0;
        }

**Sample2:**
    .. code-block:: c


        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg3;
            int ret = bmcv.crop(BMimg, BMimg3,100,100,200,200);
            return 0;
        }

resize
>>>>>>>>>>>>>>>>>

Resize the image.

**Interface:**
    .. code-block:: c

        int resize(
           BMImage                      &input,
           BMImage                      &output,
           int                          resize_w,
           int                          resize_h,
           bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

        BMImage resize(
           BMImage                      &input,
           int                          resize_w,
           int                          resize_h,
           bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

        template<std::size_t N>
        int resize(
            BMImageArray<N>              &input,
            BMImageArray<N>              &output,
            int                          resize_w,
            int                          resize_h,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

        template<std::size_t N>
        BMImageArray<N> resize(
            BMImageArray<N>              &input,
            int                          resize_w,
            int                          resize_h,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

**Parameters:**

* input : BMImage | BMImageArray

The image or array of images to be processed.

* output : BMImage | BMImageArray

Processed image or image array.

* resize_w : int

The target width for image resize.

* resize_h : int

The target height for image resize.

* resize_alg : bmcv_resize_algorithm

Interpolation algorithm for image resize, default is bmcv_resize_algorithm.BMCV_INTER_NEAREST

**Returns:**

* ret: int

Returns 0 for success, others for failure.

* output : BMImage | BMImageArray

Returns the processed image or image array.

**Sample1:**
    .. code-block:: c


        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg3 = bmcv.resize(BMimg,640,640);
            return 0;
        }

**Sample2:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg3;
            int ret = bmcv.resize(BMimg, BMimg3,640,640);
            return 0;
        }

vpp_crop_and_resize
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Use VPP hardware to accelerate image cropping and resizing.

**Interface:**
    .. code-block:: c

        int vpp_crop_and_resize(
            BMImage                      &input,
            BMImage                      &output,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

        BMImage vpp_crop_and_resize(
            BMImage                      &input,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

        template<std::size_t N>
        int vpp_crop_and_resize(
            BMImageArray<N>              &input,
            BMImageArray<N>              &output,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

        template<std::size_t N>
        BMImageArray<N> vpp_crop_and_resize(
            BMImageArray<N>              &input,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

**Parameters:**

* input : BMImage | BMImageArray

The image or array of images to be processed.

* output : BMImage | BMImageArray

Processed image or image array.

* crop_x0 : int

The starting point of the cropping window on the x-axis.

* crop_y0 : int

The starting point of the cropping window on the y-axis.

* crop_w : int 

The width of the crop window.

* crop_h : int 

The height of the crop window.

* resize_w : int

The target width for image resize.

* resize_h : int

The target height for image resize.

* resize_alg : bmcv_resize_algorithm

Interpolation algorithm for image resize, default is bmcv_resize_algorithm.BMCV_INTER_NEAREST

**Returns:**

* ret: int

Returns 0 for success, others for failure.

* output : BMImage | BMImageArray

Returns the processed image or image array.

**Sample1:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg3 = bmcv.vpp_crop_and_resize(BMimg,100,100,300,300,300,300);
            return 0;
        }

**Sample2:**
    .. code-block:: c


        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg3;
            int ret = bmcv.vpp_crop_and_resize(BMimg, BMimg3,100,100,300,300,300,300);
            return 0;
        }

vpp_crop_and_resize_padding
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Use VPP hardware to accelerate image cropping and resizing, and padding to the specified size.

**Interface:**
    .. code-block:: c

        int vpp_crop_and_resize_padding(
            BMImage                      &input,
            BMImage                      &output,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h,
            PaddingAtrr                  &padding_in,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);
        
        BMImage vpp_crop_and_resize_padding(
            BMImage                      &input,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h,
            PaddingAtrr                  &padding_in,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

        template<std::size_t N>
        int vpp_crop_and_resize_padding(
            BMImageArray<N>              &input,
            BMImageArray<N>              &output,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h,
            PaddingAtrr                  &padding_in,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

        template<std::size_t N>
        BMImageArray<N> vpp_crop_and_resize_padding(
            BMImageArray<N>              &input,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h,
            PaddingAtrr                  &padding_in,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

**Parameters:**

* input : BMImage | BMImageArray

The image or array of images to be processed.

* output : BMImage | BMImageArray

Processed image or image array.

* crop_x0 : int

The starting point of the cropping window on the x-axis.

* crop_y0 : int

The starting point of the cropping window on the y-axis.

* crop_w : int 

The width of the crop window.

* crop_h : int 

The height of the crop window.

* resize_w : int

The target width for image resize.

* resize_h : int

The target height for image resize.

* padding : PaddingAtrr

padding configuration information.

* resize_alg : bmcv_resize_algorithm

Interpolation algorithm for image resize, default is bmcv_resize_algorithm.BMCV_INTER_NEAREST

**Returns:**

* ret: int

Returns 0 for success, others for failure.

* output : BMImage | BMImageArray

Returns the processed image or image array.

**Sample1:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::PaddingAtrr paddingatt;
            paddingatt.set_stx(0);
            paddingatt.set_sty(0);
            paddingatt.set_w(640);
            paddingatt.set_h(640);
            paddingatt.set_r(114);
            paddingatt.set_g(114);
            paddingatt.set_b(114);
            sail::BMImage BMimg4 = bmcv.vpp_crop_and_resize_padding(BMimg, 0, 0, BMimg.width(), BMimg.height(), 640, 640, paddingatt);
            return 0;
        }

**Sample2:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg3;
            sail::PaddingAtrr paddingatt;
            paddingatt.set_stx(0);
            paddingatt.set_sty(0);
            paddingatt.set_w(640);
            paddingatt.set_h(640);
            paddingatt.set_r(114);
            paddingatt.set_g(114);
            paddingatt.set_b(114);
            int ret = bmcv.vpp_crop_and_resize_padding(BMimg,BMimg3, 0, 0, BMimg.width(), BMimg.height(), 640, 640, paddingatt);
            return 0;
        }

vpp_crop
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Use VPP hardware to accelerate image cropping.

**Interface:**
    .. code-block:: c

        int vpp_crop(
           BMImage                      &input,
           BMImage                      &output,
           int                          crop_x0,
           int                          crop_y0,
           int                          crop_w,
           int                          crop_h);
        
        BMImage vpp_crop(
           BMImage                      &input,
           int                          crop_x0,
           int                          crop_y0,
           int                          crop_w,
           int                          crop_h);

        template<std::size_t N>
        int vpp_crop(
            BMImageArray<N>              &input,
            BMImageArray<N>              &output,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h);

        template<std::size_t N>
        BMImageArray<N> vpp_crop(
            BMImageArray<N>              &input,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h);

**Parameters:**

* input : BMImage | BMImageArray

The image or array of images to be processed.

* output : BMImage | BMImageArray

Processed image or image array.

* crop_x0 : int

The starting point of the cropping window on the x-axis.

* crop_y0 : int

The starting point of the cropping window on the y-axis.

* crop_w : int 

The width of the crop window.

* crop_h : int 

The height of the crop window.

**返回值说明:**

* ret: int

Returns 0 for success, others for failure.

* output : BMImage | BMImageArray

Returns the processed image or image array.

**Sample1:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg3 = bmcv.vpp_crop(BMimg,100,100,200,200);
            return 0;
        }

**Sample2:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg3;
            int ret = bmcv.vpp_crop(BMimg, BMimg3,100,100,200,200);
            return 0;
        }

vpp_resize
>>>>>>>>>>>>>>>>>

Use VPP hardware to accelerate image resize and use nearest neighbor interpolation algorithm.

**接口形式1:**
    .. code-block:: c

        int vpp_resize(
            BMImage                      &input,
            BMImage                      &output,
            int                          resize_w,
            int                          resize_h,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);
        
        BMImage vpp_resize(
            BMImage                      &input,
            int                          resize_w,
            int                          resize_h,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);
        
        template<std::size_t N>
        int vpp_resize(
            BMImageArray<N>              &input,
            BMImageArray<N>              &output,
            int                          resize_w,
            int                          resize_h,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

        template<std::size_t N>
        BMImageArray<N> vpp_resize(
            BMImageArray<N>              &input,
            int                          resize_w,
            int                          resize_h,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

**Parameters:**

* input : BMImage | BMImageArray

The image or array of images to be processed.

* output : BMImage | BMImageArray

Processed image or image array.

* resize_w : int

The target width for image resize.

* resize_h : int

The target height for image resize.

* resize_alg : bmcv_resize_algorithm

Interpolation algorithm for image resize, default is bmcv_resize_algorithm.BMCV_INTER_NEAREST

**Returns:**

* ret: int

Returns 0 for success, others for failure.

* output : BMImage | BMImageArray

Returns the processed image or image array.

**Sample1:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg3 = bmcv.vpp_resize(BMimg,100,100,200,200);
            return 0;
        }

**Sample2:**
    .. code-block:: c


        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg3;
            int ret = bmcv.vpp_resize(BMimg, BMimg3,100,100,200,200);
            return 0;
        }

vpp_resize_padding
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Use VPP hardware to accelerate image resizing and padding.

**Interface:**
    .. code-block:: c

        int vpp_resize_padding(
            BMImage                      &input,
            BMImage                      &output,
            int                          resize_w,
            int                          resize_h,
            PaddingAtrr                  &padding_in,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

        BMImage vpp_resize_padding(
           BMImage                      &input,
           int                          resize_w,
           int                          resize_h,
           PaddingAtrr                  &padding_in,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

        template<std::size_t N>
            int vpp_resize_padding(
            BMImageArray<N>              &input,
            BMImageArray<N>              &output,
            int                          resize_w,
            int                          resize_h,
            PaddingAtrr                  &padding_in,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

        template<std::size_t N>
        BMImageArray<N> vpp_resize_padding(
            BMImageArray<N>              &input,
            int                          resize_w,
            int                          resize_h,
            PaddingAtrr                  &padding_in,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

**Parameters:**

* input : BMImage | BMImageArray

The image or array of images to be processed.

* resize_w : int

The target width for image resize.

* resize_h : int

The target height for image resize.

* padding : PaddingAtrr

The configuration information of padding.

**Returns:**

* ret: int

Returns 0 for success, others for failure.

* output : BMImage | BMImageArray

Returns the processed image or image array.

* resize_alg : bmcv_resize_algorithm

Interpolation algorithm for image resize, default is bmcv_resize_algorithm.BMCV_INTER_NEAREST

**Sample1:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::PaddingAtrr paddingatt;
            paddingatt.set_stx(0);
            paddingatt.set_sty(0);
            paddingatt.set_w(640);
            paddingatt.set_h(640);
            paddingatt.set_r(114);
            paddingatt.set_g(114);
            paddingatt.set_b(114);
            sail::BMImage BMimg4 = bmcv.vpp_resize_padding(BMimg, 0, 0, 640, 640, paddingatt);
            return 0;
        }

**Sample2:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg3;
            sail::PaddingAtrr paddingatt;
            paddingatt.set_stx(0);
            paddingatt.set_sty(0);
            paddingatt.set_w(640);
            paddingatt.set_h(640);
            paddingatt.set_r(114);
            paddingatt.set_g(114);
            paddingatt.set_b(114);
            int ret = bmcv.vpp_resize_padding(BMimg,BMimg3, 640, 640, paddingatt);
            return 0;
        }

warp
>>>>>>>>>>>>>>>>

Perform an affine transformation on the image.

**Interface:**
    .. code-block:: c

        int warp(
           BMImage                            &input,
           BMImage                            &output,
           const std::pair<
             std::tuple<float, float, float>,
             std::tuple<float, float, float>> &matrix,
           int                                use_bilinear = 0,
           bool                               similar_to_opencv = false);

        BMImage warp(
           BMImage                            &input,
           const std::pair<
             std::tuple<float, float, float>,
             std::tuple<float, float, float>> &matrix,
           int                                use_bilinear = 0,
           bool                               similar_to_opencv = false);

        template<std::size_t N>
        int warp(
            BMImageArray<N>                          &input,
            BMImageArray<N>                          &output,
            const std::array<
                std::pair<
                std::tuple<float, float, float>,
                std::tuple<float, float, float>>, N> &matrix,
            int                                      use_bilinear = 0,
            bool                                     similar_to_opencv = false);

        template<std::size_t N>
        BMImageArray<N> warp(
            BMImageArray<N>                          &input,
            const std::array<
                std::pair<
                std::tuple<float, float, float>,
                std::tuple<float, float, float>>, N> &matrix,
            int                                      use_bilinear = 0,
            bool                                     similar_to_opencv = false);

**Parameters:**

* input : BMImage | BMImageArray

The image or array of images to be processed.

* output : BMImage | BMImageArray

Processed image or image array.

* matrix: std::pair<
             std::tuple<float, float, float>,
             std::tuple<float, float, float> >

2x3 affine transformation matrix.

* use_bilinear: int

Whether to use bilinear interpolation, default to 0 using nearest neighbor interpolation, 1 being bilinear interpolation

* similar_to_opencv: bool

Whether to use the interface aligning the affine transformation interface of OpenCV

**Returns:**

* ret: int

Returns 0 for success, others for failure.

* output : BMImage | BMImageArray

Returns the processed image or image array.

**Sample1:**
    .. code-block:: c


        #include <sail/cvwrapper.h>
        using namespace std;
        using AffineMatrix = std::pair<
            std::tuple<float, float, float>,
            std::tuple<float, float, float>>;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            AffineMatrix rotated_matrix = std::make_pair(
                std::make_tuple(0.9996914396, -0.02484, 0.0f),
                std::make_tuple(0.02484, 0.9996914396, 0.0f)
            );
            sail::BMImage BMimg6 = bmcv.warp(BMimg, rotated_matrix);
            return 0;
        }

**Sample2:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        using AffineMatrix = std::pair<
            std::tuple<float, float, float>,
            std::tuple<float, float, float>>;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            AffineMatrix rotated_matrix = std::make_pair(
                std::make_tuple(0.9996914396, -0.02484, 0.0f),
                std::make_tuple(0.02484, 0.9996914396, 0.0f)
            );
            sail::BMImage BMimg6;
            int ret= bmcv.warp(BMimg,BMimg6, rotated_matrix);
            return 0;
        }

convert_to
>>>>>>>>>>>>>>

Perform a linear transformation on the image.

**Interface:**
    .. code-block:: c

        int convert_to(
           BMImage                      &input,
           BMImage                      &output,
           const std::tuple<
             std::pair<float, float>,
             std::pair<float, float>,
             std::pair<float, float>>   &alpha_beta);

        BMImage convert_to(
           BMImage                      &input,
           const std::tuple<
             std::pair<float, float>,
             std::pair<float, float>,
             std::pair<float, float>>   &alpha_beta);

        template<std::size_t N>
        int convert_to(
            BMImageArray<N>              &input,
            BMImageArray<N>              &output,
            const std::tuple<
                std::pair<float, float>,
                std::pair<float, float>,
                std::pair<float, float>>   &alpha_beta);

        template<std::size_t N>
        BMImageArray<N> convert_to(
            BMImageArray<N>              &input,
            const std::tuple<
                std::pair<float, float>,
                std::pair<float, float>,
                std::pair<float, float>>   &alpha_beta);
    
**Parameters:**

* input : BMImage | BMImageArray

The image or array of images to be processed.

* alpha_beta: std::tuple<
             std::pair<float, float>,
             std::pair<float, float>,
             std::pair<float, float> > 

The coefficients of the linear transformation of the three channels ((a0, b0), (a1, b1), (a2, b2)).

* output : BMImage | BMImageArray

Output parameters. Processed image or image array.

**Returns:**

* ret: int

Returns 0 for success, others for failure.

* output : BMImage | BMImageArray

Returns the processed image or image array.

**Sample1:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            std::tuple<std::pair<float, float>, std::pair<float, float>, std::pair<float, float>> alpha_beta = 
                std::make_tuple(std::make_pair(1.0 / 255, 0), std::make_pair(1.0 / 255, 0), std::make_pair(1.0 / 255, 0));
            sail::BMImage BMimg5 = bmcv.convert_to(BMimg, alpha_beta);
            return 0;
        }

**Sample2:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            std::tuple<std::pair<float, float>, std::pair<float, float>, std::pair<float, float>> alpha_beta = 
                std::make_tuple(std::make_pair(1.0 / 255, 0), std::make_pair(1.0 / 255, 0), std::make_pair(1.0 / 255, 0));
            sail::BMImage BMimg5; 
            int ret = bmcv.convert_to(BMimg,BMimg5,alpha_beta);
            return 0;
        }

yuv2bgr
>>>>>>>>>>>>

Convert the format of the image from YUV to BGR.

**Interface:**
    .. code-block:: c

        int yuv2bgr(
           BMImage                      &input,
           BMImage                      &output);

        BMImage yuv2bgr(BMImage  &input);

**Parameters:**

* input : BMImage | BMImageArray

The image to be converted.

**Returns:**

* ret: int

Returns 0 for success, others for failure.

* output : BMImage | BMImageArray

Returns the converted image.

**Sample1:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg5 = bmcv.yuv2bgr(BMimg);
            return 0;
        }

**Sample2:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg5; 
            int ret = bmcv.yuv2bgr(BMimg,BMimg5);
            return 0;
        }

rectangle
>>>>>>>>>>>>>>>>>>

Draw a rectangular box on the image.

**Interface:**
    .. code-block:: c

        int rectangle(
           BMImage                         &image,
           int                             x0,
           int                             y0,
           int                             w,
           int                             h,
           const std::tuple<int, int, int> &color,
           int                             thickness=1);

        int rectangle(
           const bm_image                  &image,
           int                             x0,
           int                             y0,
           int                             w,
           int                             h,
           const std::tuple<int, int, int> &color, // BGR
           int                             thickness=1);

**Parameters:**

* image : BMImage | bm_image

The image of the rectangle to be drawn.

* x0 : int

The starting point of the rectangle on the x-axis.

* y0 : int

The starting point of the rectangular box on the y-axis.

* w : int

The width of the rectangular box.

* h : int

The height of the rectangular box.

* color : tuple

The color of the rectangle.

* thickness : int

The thickness of the rectangular box lines.

**Returns:**

Returns 0 if the frame is successful, otherwise returns a non-zero value.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            int ret = bmcv.rectangle(BMimg, 20, 20, 600, 600, std::make_tuple(0, 0, 255), 2);
            return 0;
        }

fillRectangle
>>>>>>>>>>>>>>>>>>

Fill a rectangular box on the image.

**Interface:**
    .. code-block:: c

        int fillRectangle(
           BMImage                         &image,
           int                             x0,
           int                             y0,
           int                             w,
           int                             h,
           const std::tuple<int, int, int> &color);

        int fillRectangle(
           const bm_image                  &image,
           int                             x0,
           int                             y0,
           int                             w,
           int                             h,
           const std::tuple<int, int, int> &color);


**Parameters:**

* image : BMImage | bm_image

The image of the rectangle to be drawn.

* x0 : int

The starting point of the rectangle on the x-axis.

* y0 : int

The starting point of the rectangular box on the y-axis.

* w : int

The width of the rectangular box.

* h : int

The height of the rectangular box.

* color : tuple

The color of the rectangle.

**Returns:**

Returns 0 if the frame is successful, otherwise returns a non-zero value.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            int ret = bmcv.fillRectangle(BMimg, 20, 20, 600, 600, std::make_tuple(0, 0, 255));
            return 0;
        }

imwrite
>>>>>>>>>>>>>>>>>

Save the image in a specific file.

**Interface:**
    .. code-block:: c

        int imwrite(
           const std::string &filename,
           BMImage           &image);

        int imwrite(
           const std::string &filename,
           const bm_image     &image);


**Parameters:**

* file_name : string

The name of the file.

* output : BMImage | bm_image

The image needs to be saved.

**Returns:**

* process_status : int

Returns 0 if the save is successful, otherwise returns a non-zero value.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            int ret = bmcv.imwrite("new_3.jpg", BMimg);
            return 0;
        }


imread
>>>>>>>>>>>>>>>>>

Read and decode one image files and supports hard decoding only for JPEG baseline format. For other formats, such as PNG and BMP, soft decoding is used.
The returned BMImage for JPEG baseline images keeps YUV color space, and the pixel format depends on the sampling information in the file like YUV420. 
The returned BMImage for other formats will maintain the corresponding color space of their input.

**Interface:**
    .. code-block:: c

        BMImage imread(const std::string &filename);

**Parameters:**

* filename : string

Name of file to be read.

**Returns:**

* output : sail.BMImage

The decoded image.

**Sample:**
    .. code-block:: c++

        #include <sail/cvwrapper.h>
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            sail::Bmcv bmcv(handle);
            std::string filename = "your_image_path";
            sail::BMImage BMimg = bmcv.imread(filename);
            return 0;
        }


get_handle
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the device handle Handle in Bmcv.

**Interface:**
    .. code-block:: c

        Handle get_handle();

**Returns:**

* handle: Handle

The device handle Handle in Bmcv.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::Handle handle1 = bmcv.get_handle();
            return 0;
        }

crop_and_resize_padding
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Crop and resize the image, then padding it.

**Interface:**
    .. code-block:: c

        int vpp_crop_and_resize_padding(
            BMImage                      &input,
            BMImage                      &output,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h,
            PaddingAtrr                  &padding_in,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);
        
        BMImage vpp_crop_and_resize_padding(
            BMImage                      &input,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h,
            PaddingAtrr                  &padding_in,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

**Parameters:**

* input : BMImage

The image to be processed.

* output : BMImage

The processed image.

* crop_x0 : int

The starting point of the cropping window on the x-axis.

* crop_y0 : int

The starting point of the cropping window on the y-axis.

* crop_w : int 

The width of the crop window.

* crop_h : int 

The height of the crop window.

* resize_w : int

The target width for image resize.

* resize_h : int

The target height for image resize.

* padding : PaddingAtrr

The configuration information of padding.

* resize_alg : bmcv_resize_algorithm

The interpolation algorithm used by resize.

**Returns:**

* process_status : int

Returns 0 if the save is successful, otherwise returns a non-zero value.

* output : BMImage

Return the processed image.

**Sample1:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::PaddingAtrr paddingatt;
            paddingatt.set_stx(0);
            paddingatt.set_sty(0);
            paddingatt.set_w(640);
            paddingatt.set_h(640);
            paddingatt.set_r(114);
            paddingatt.set_g(114);
            paddingatt.set_b(114);
            sail::BMImage BMimg4 = bmcv.crop_and_resize_padding(BMimg, 0, 0, BMimg.width(), BMimg.height(), 640, 640, paddingatt);
            return 0;
        }

**Sample2:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg3;
            sail::PaddingAtrr paddingatt;
            paddingatt.set_stx(0);
            paddingatt.set_sty(0);
            paddingatt.set_w(640);
            paddingatt.set_h(640);
            paddingatt.set_r(114);
            paddingatt.set_g(114);
            paddingatt.set_b(114);
            bm_image bm_img = bmcv.crop_and_resize_padding(BMimg.data(), 0, 0, BMimg.width(), BMimg.height(), 640, 640, paddingatt);
            return 0;
        }


convert_format
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Convert the image format to the format in the output and copy it to the output.

**Interface 1:**
    .. code-block:: c

        int convert_format(
            BMImage          &input,
            BMImage          &output
        );

**Parameters 1:**

* input : BMImage

Input parameters. The image to be converted.

* output : BMImage

Output parameters. Convert the image in input to the image format of output and copy it to output.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg4;
            int ret = bmcv.convert_format(BMimg,BMimg4);
            return 0;
        }

**Interface 2:**

Convert an image to the target format.

    .. code-block:: c

        BMImage convert_format(
            BMImage          &input,
            bm_image_format_ext image_format = FORMAT_BGR_PLANAR
        );

**Parameters 2:**

* input : BMImage

The image to be converted.

* image_format : bm_image_format_ext

The target format for conversion.

**Returns 2:**

* output : BMImage

Returns the converted image.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg4 = bmcv.convert_format(BMimg);
            return 0;
        }

vpp_convert_format
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Use VPP hardware to accelerate image format conversion.

**Interface 1:**
    .. code-block:: c

        int vpp_convert_format(
            BMImage          &input,
            BMImage          &output
        );

**Parameters 1:**

* input : BMImage

Input parameters. The image to be converted.

* output : BMImage

Output parameters. Convert the image in input to the image format of output and copy it to output.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg4;
            int ret = bmcv.vpp_convert_format(BMimg,BMimg4);
            return 0;
        }

**Interface 2:**

Convert an image to the target format.

    .. code-block:: c

        BMImage vpp_convert_format(
            BMImage          &input,
            bm_image_format_ext image_format = FORMAT_BGR_PLANAR
        );

**Parameters 2:**

* input : BMImage

The image to be converted.

* image_format : bm_image_format_ext

The target format for conversion.

**Returns 2:**

* output : BMImage

Returns the converted image.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage BMimg4 = bmcv.vpp_convert_format(BMimg);
            return 0;
        }

putText
>>>>>>>>>>

Add text to the image.

Supported pixel format for input BMImage: 
FORMAT_GRAY, FORMAT_YUV420P, FORMAT_YUV422P, FORMAT_YUV444P, FORMAT_NV12, 
FORMAT_NV21, FORMAT_NV16, FORMAT_NV61

**Interface:**
    .. code-block:: c
        
        int putText(
            const BMImage                   &image,
            const std::string               &text,
            int                             x,
            int                             y,
            const std::tuple<int, int, int> &color, // BGR
            float                           fontScale,
            int                             thickness=1
        );

        int putText(
            const bm_image                  &image,
            const std::string               &text,
            int                             x,
            int                             y,
            const std::tuple<int, int, int> &color, // BGR
            float                           fontScale,
            int                             thickness=1
        );

**Parameters:**

* input : BMImage | bm_image

The image to be processed.

* text: string

Text that needs to be added.

* x: int

The starting point for adding the text on the x-axis.

* y: int

The starting point for adding the text on the y-axis.

* color : tuple

The color of the font.

* fontScale: int

The size of the font.

* thickness : int

The thickness of the font.

**Returns:**

* process_status : int

Returns 0 if processing is successful, otherwise returns a non-zero value.

**Sample:**
    .. code-block:: c++

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage bgr_img = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage yuv_img = bmcv.convert_format(bgr_img, FORMAT_YUV420P)
            int ret = bmcv.putText(yuv_img, "some text" , 20, 20, std::make_tuple(0, 0, 255), 1.4, 2);
            
            return 0;
        }

image_add_weighted
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Add two images with different weights.

**Interface 1:**
    .. code-block:: c
        
        int image_add_weighted(
            BMImage           &input1,
            float             alpha,
            BMImage           &input2,
            float             beta,
            float             gamma,
            BMImage           &output
        );

**Parameters 1:**

* input0 : BMImage

Input parameters. The image 0 to be processed.

* alpha : float

Input parameters. The weight alpha of the two images added together.

* input1 : BMImage

Input parameters. The image 1 to be processed.

* beta : float

Input parameters. The weight beta of the two images added together.

* gamma : float

Input parameters. The weight gamma of the two images added together.

* output: BMImage

Output parameters. The added image output = input1 * alpha + input2 * beta + gamma

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name1 = "your_image_path1";
            std::string image_name2 = "your_image_path2";
            sail::Decoder decoder1(image_name1, true, tpu_id);
            sail::Decoder decoder2(image_name2, true, tpu_id);
            sail::BMImage BMimg1 = decoder1.read(handle); 
            sail::BMImage BMimg2 = decoder2.read(handle); 
            sail::Bmcv bmcv(handle);
            float alpha=0.2,beta=0.5,gamma=0.8;
            int ret = bmcv.image_add_weighted(BMimg1,alpha,BMimg2,beta,gamma,BMimg2);
            return 0;
        }

**Interface 2:**
    .. code-block:: c

        BMImage image_add_weighted(
            BMImage           &input1,
            float             alpha,
            BMImage           &input2,
            float             beta,
            float             gamma
        );


**Parameters 2:**

* input0 : BMImage

Input parameters. The image 0 to be processed.

* alpha : float

Input parameters. The weight alpha of the two images added together.

* input1 : BMImage

Input parameters. The image 1 to be processed.

* beta : float

Input parameters. The weight beta of the two images added together.

* gamma : float

Input parameters. The weight gamma of the two images added together.

**Returns 2:**

* output: BMImage

Return the added image output = input1 * alpha + input2 * beta + gamma

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name1 = "your_image_path1";
            std::string image_name2 = "your_image_path2";
            sail::Decoder decoder1(image_name1, true, tpu_id);
            sail::Decoder decoder2(image_name2, true, tpu_id);
            sail::BMImage BMimg1 = decoder1.read(handle); 
            sail::BMImage BMimg2 = decoder2.read(handle); 
            sail::Bmcv bmcv(handle);
            float alpha=0.2,beta=0.5,gamma=0.8;
            sail::BMImage img= bmcv.image_add_weighted(BMimg1,alpha,BMimg2,beta,gamma);
            return 0;
        }

image_copy_to
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Copy data between images

**Interface:**
    .. code-block:: c

        int image_copy_to(BMImage &input, BMImage &output, int start_x = 0, int start_y = 0);

        template<std::size_t N>
        int image_copy_to(BMImageArray<N> &input, BMImageArray<N> &output, int start_x = 0, int start_y = 0);

**Parameters:**

* input: BMImage|BMImageArray

Input parameter. The BMImage or BMImageArray to be copied.

* output: BMImage|BMImageArray

Output parameter. Copied BMImage or BMImageArray

* start_x: int

Input parameter. Copy to the starting point of the target image.

* start_y: int

Input parameter. Copy to the starting point of the target image.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name1 = "your_image_path1";
            std::string image_name2 = "your_image_path2";
            sail::Decoder decoder1(image_name1, true, tpu_id);
            sail::Decoder decoder2(image_name2, true, tpu_id);
            sail::BMImage BMimg1 = decoder1.read(handle); 
            sail::BMImage BMimg2 = decoder2.read(handle); 
            sail::Bmcv bmcv(handle);
            bmcv.image_copy_to(BMimg1,BMimg2,0,0);
            return 0;
        }

image_copy_to_padding
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Copy and padding the image data between input and output.

**Interface:**
    .. code-block:: c
    
        int image_copy_to_padding(BMImage &input, 
                                BMImage &output,
                                unsigned int padding_r, 
                                unsigned int padding_g, 
                                unsigned int padding_b,
                                int start_x = 0, 
                                int start_y = 0);

        template<std::size_t N>
        int image_copy_to_padding(BMImageArray<N> &input, 
                                BMImageArray<N> &output, 
                                unsigned int padding_r, 
                                unsigned int padding_g, 
                                unsigned int padding_b,
                                int start_x = 0, 
                                int start_y = 0);

**Parameters:**

* input: BMImage|BMImageArray

Input parameter. The BMImage or BMImageArray to be copied.

* output: BMImage|BMImageArray

Output parameter. Copied BMImage or BMImageArray.

* padding_r: int

Input parameter. The padding value of the R channel.

* padding_g: int

Input parameter. The padding value of the G channel.

* padding_b: int

Input parameter. The padding value of the B channel.

* start_x: int

Input parameter. Copy to the starting point on x-axis of the target image.

* start_y: int

Input parameter. Copy to the starting point on y-axis of the target image.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name1 = "your_image_path1";
            std::string image_name2 = "your_image_path2";
            sail::Decoder decoder1(image_name1, true, tpu_id);
            sail::Decoder decoder2(image_name2, true, tpu_id);
            sail::BMImage BMimg1 = decoder1.read(handle); 
            sail::BMImage BMimg2 = decoder2.read(handle); 
            sail::Bmcv bmcv(handle);
            bmcv.image_copy_to_padding(BMimg1,BMimg2,128,128,128,0,0);
            return 0;
        }

nms
>>>>>>>>

Using Tensor Computing Processor for NMS

**Note:** For details about whether this operator in current SDK supports BM1688, check the page "BMCV API" in 《Multimedia User Guide》. 

**Interface:**
    .. code-block:: c

        nms_proposal_t* nms(
            face_rect_t *input_proposal,
            int proposal_size, 
            float threshold);

**Parameters:**

* input_proposal: face_rect_t

Data starting address.

* proposal_size: int

The size of the detection frame data to be processed.

* threshold: float

Threshold of nms.

**Returns:**

* result: nms_proposal_t

Returns the detection frame array after NMS.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main()
        {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            sail::Bmcv bmcv(handle);
            face_rect_t *input_proposal; 
            int proposal_size = 100; 
            float threshold = 0.5; 
            nms_proposal_t* result = bmcv.nms(input_proposal, proposal_size, threshold);
            return 0;
        }

drawPoint
>>>>>>>>>>>>>

Draw points on the image.

**Interface:**
    .. code-block:: c

        int drawPoint(
            const BMImage &image,
            std::pair<int,int> center,
            std::tuple<unsigned char, unsigned char, unsigned char> color,   // BGR
            int radius);
        
        int drawPoint(
            const bm_image  &image,
            std::pair<int,int> center,
            std::tuple<unsigned char, unsigned char, unsigned char> color,  // BGR
            int radius);


**Parameters:**

* image: BMImage

Input image. Draw points directly on the BMImage as output.

* center: std::pair<int,int>

The center coordinates of the point.

* color: std::tuple<unsigned char, unsigned char, unsigned char>

The color of the point.

* radius: int

The radius of the point. 

**Returns**

If the point is drawn successfully, 0 is returned, otherwise a non-zero value is returned.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path1";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            int ret = bmcv.drawPoint(BMimg, std::pair(320, 320), std::make_tuple(0, 255, 255), 2);
            return 0;
        }

warp_perspective
>>>>>>>>>>>>>>>>>>>>>

Performs perspective transformation on the image.

**Interface:**
    .. code-block:: c

        BMImage warp_perspective(
            BMImage                     &input,
            const std::tuple<
            std::pair<int,int>,
            std::pair<int,int>,
            std::pair<int,int>,
            std::pair<int,int>>       &coordinate,
            int                         output_width,
            int                         output_height,
            bm_image_format_ext         format = FORMAT_BGR_PLANAR,
            bm_image_data_format_ext    dtype = DATA_TYPE_EXT_1N_BYTE,
            int                         use_bilinear = 0);

**Parameters:**

* input: BMImage

The image to be processed.

* coordinate: std::tuple<
            std::pair<int,int>,
            std::pair<int,int>,
            std::pair<int,int>,
            std::pair<int,int> >

The original coordinates of the four vertices of the transformed area.

Such as, ((left_top.x, left_top.y), (right_top.x, right_top.y),
(left_bottom.x, left_bottom.y), (right_bottom.x, right_bottom.y))

* output_width: int

The width of the output image.

* output_height: int

The height of the output image.

* format: bm_image_format_ext

The format of the output image.

* dtype: bm_image_data_format_ext

The data type of the output image.

* use_bilinear: int

Whether to use bilinear interpolation.

**Returns:**

* output: BMImage

Output the transformed image.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;

        int main()
        {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            std::tuple<
                std::pair<int, int>,
                std::pair<int, int>,
                std::pair<int, int>,
                std::pair<int, int>
            > coordinate = std::make_tuple(
                std::make_pair(100, 100), 
                std::make_pair(200, 100), 
                std::make_pair(100, 200), 
                std::make_pair(200, 200)  
            )
            int output_width = 300;
            int output_height = 300; 
            bm_image_format_ext format = FORMAT_BGR_PLANAR; 
            bm_image_data_format_ext dtype = DATA_TYPE_EXT_1N_BYTE; 
            int use_bilinear = 1; 

            sail::BMImage output = bmcv.warp_perspective(BMimg,coordinate,output_width,output_height,format,dtype,use_bilinear
            );

            return 0;
        }

get_bm_data_type
>>>>>>>>>>>>>>>>>>>>

Convert ImgDtype to Dtype

**Interface:**
    .. code-block:: c

        bm_data_type_t get_bm_data_type(bm_image_data_format_ext fmt);

**Parameters:**

* fmt: bm_image_data_format_ext

The type to be converted.

**Returns:**

* ret: bm_data_type_t

The converted type.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            sail::Bmcv bmcv(handle);
            bm_data_type_t ret = bmcv.get_bm_data_type(bm_image_data_format_ext::DATA_TYPE_EXT_FLOAT32);
            return 0;
        }

get_bm_image_data_format
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Convert Dtype to ImgDtype.

**Interface:**
    .. code-block:: c

        bm_image_data_format_ext get_bm_image_data_format(bm_data_type_t dtype);

**Parameters:**

* dtype: bm_data_type_t

The Dtype that needs to be converted.

**Returns:**

* ret: bm_image_data_format_ext

Returns the converted type.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            sail::Bmcv bmcv(handle);
            bm_image_data_format_ext ret = bmcv.get_bm_image_data_format(bm_data_type_t::BM_FLOAT32);
            return 0;
        }

imdecode
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Load the image from memory into BMImage.

**Interface:**
    .. code-block:: c

        BMImage imdecode(const void* data_ptr, size_t data_size);
          
**Parameters:**

* data_ptr: void*

The data starting address.

* data_size: bytes

The data length.

**Returns:**

* ret: BMImage

Returns the decoded image.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            std::ifstream image_file(image_name, std::ios::binary);
            if (!image_file) {
                std::cout << "Error opening image file." << std::endl;
                return -1;
            }
            std::vector<char> image_data_bytes(
                (std::istreambuf_iterator<char>(image_file)),
                (std::istreambuf_iterator<char>())
            );
            image_file.close();
            sail::Bmcv bmcv(handle);
            sail::BMImage src_img = bmcv.imdecode(image_data_bytes.data(), image_data_bytes.size());
            return 0;
        }

imencode
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Encode an BMimage and return the encoded data.

**Interface1:**
    .. code-block:: c

        bool Bmcv::imencode(std::string& ext, bm_image &img, std::vector<u_char>& buf)

**Interface2:**
    .. code-block:: c

        bool Bmcv::imencode(std::string& ext, BMImage &img, std::vector<u_char>& buf)
          
**Parameters:**

* ext: string

Input parameter. Image encoding format, supported formats include ``".jpg"``, ``".png"``, etc.

* image: bm_image/BMImage

Input parameter. Input bm_image/BMImage, only FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE pictures are supported.

* buf: std::vector<u_char>

Output parameter. Data that is encoded and placed in system memory.

**返回值说明:**

* ret: bool

Returns 0 if encoding is successful and 1 if encoding fails.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            std::vector<u_char> encoded_data;
            std::string ext = ".jpg";
            bool success = bmcv.imencode(ext, BMimg, encoded_data);
            //bool success = bmcv.imencode(ext, BMimg.data(), encoded_data);  接口形式1:bm_image
            return 0;
        }

fft
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Implement the Fast Fourier Transform of Tensor.

**Note:** For details about whether this operator in current SDK supports BM1688, check the page "BMCV API" in 《Multimedia User Guide》. 

**Interface:**
    .. code-block:: c

        std::vector<Tensor> fft(bool forward, Tensor &input_real);

        std::vector<Tensor> fft(bool forward, Tensor &input_real, Tensor &input_imag);
    
**Parameters:**

* forward: bool

Whether to perform forward migration.

* input_real: Tensor

The real part of the input.

* input_imag: Tensor

The imaginary part of the input.

**Returns:**

* ret: std::vector<Tensor>

Returns the real and imaginary parts of the output.


**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main()
        {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            sail::Bmcv bmcv(handle);
            std::vector<int> shape = {512, 512};
            sail::Tensor input_real(shape);
            bool forward = true; 
            //std::vector<sail::Tensor> result_real = bmcv.fft(forward, input_real);  
            sail::Tensor input_imag(shape);
            std::vector<sail::Tensor> result_complex  = bmcv.fft(forward, input_real,input_imag);
            return 0;
        }

convert_yuv420p_to_gray
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Convert pictures in YUV420P format to grayscale images.

**Interface 1:**
    .. code-block:: c

        int convert_yuv420p_to_gray(BMImage& input, BMImage& output);

**Parameters 1:**

* input : BMImage

Input parameters. The image to be converted.

* output : BMImage

Output parameters. Converted image.


**Interface 2:**

Convert pictures in YUV420P format to grayscale images.

    .. code-block:: c

        int convert_yuv420p_to_gray_(bm_image& input, bm_image& output); 

**Parameters 2:**

* input : bm_image

The image to be converted.

* output : bm_image

The converted image.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main() {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path1";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            sail::BMImage img;
            int ret = bmcv.convert_yuv420p_to_gray(BMimg, img);
            return 0;
        }

polylines
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Draw one or more line segments on an image, 
so that the function of drawing polygons can be realized, and the color and width of the line can be specified.


**Interface:**
    .. code-block:: c

        int polylines(
            BMImage &img,
            std::vector<std::vector<std::pair<int,int>>> &pts,
            bool isClosed,
            std::tuple<unsigned char, unsigned char, unsigned char> color,
            int thickness = 1,
            int shift = 0);

**Parameters:**

* img : BMImage

Input BMImage.

* pts : std::vector<std::vector<std::pair<int,int>>>

The starting point and end point coordinates of the line segment, multiple coordinate points can be entered. The upper left corner of the image is the origin, 
extending to the right in the x direction and extending down in the y direction.

* isClosed : bool
  
Whether the graph is closed.

* color :  std::tuple<unsigned char, unsigned char, unsigned char>

The color of the line is the value of the three RGB channels.

* thickness : int 

The width of the lines is recommended to be even for YUV format images.

* shift : int

Polygon scaling multiple. Default is not scaling. The scaling factor is(1/2)^shift。


**Returns:**

* ret: int

returns 0 if success.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        int main()
        {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            std::vector<std::vector<std::pair<int, int>>> pts = {
                {{100, 100}, {150, 100}, {150, 150}, {100, 150}}, 
                {{200, 200}, {250, 200}, {250, 250}, {200, 250}}  

            };
            bool isClosed = true;
            int thickness = 2;
            std::tuple<unsigned char, unsigned char, unsigned char> color = std::make_tuple(255, 0, 0); 
            int shift = 0;
            int result = bmcv.polylines(BMimg, pts, isClosed, color, thickness, shift);
            if (result == 0) {
                std::cout << "Polylines drawn successfully." << std::endl;
            } else {
                std::cout << "Failed to draw polylines." << std::endl;
            }
            return 0;
        }

mosaic
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Print one or more mosaics on an image.

**Interface:**
    .. code-block:: c

         int mosaic(
            int mosaic_num,
            BMImage &img,
            vector<vector<int>> rects,
            int is_expand);

**Parameters:**

* mosaic_num : int

Number of mosaics, length of list in rects.

* img : BMImage

Input BMImage.

* rects : vector<vector<int>>

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
    .. code-block:: c

        #include <sail/cvwrapper.h>
        int main()
        {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            std::vector<std::vector<int>> rects = {
                {100, 100, 50, 50}, 
                {200, 200, 60, 60}  
            };
            int mosaic_num = rects.size(); /
            int is_expand = 0; 
            int result = bmcv.mosaic(mosaic_num, BMimg, rects, is_expand);
            if (result == 0) {
                std::cout << "Mosaic applied successfully." << std::endl;
            } else {
                std::.cout << "Failed to apply mosaic." << std::endl;
            }
            return 0;
        }


transpose
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Transpose of image width and height.

**Interface1:**
    .. code-block:: cpp

        BMImage Bmcv::transpose(BMImage &src);


**Parameters1:**

* src : BMImage

Input BMImage.


**Returns2:**

* output: BMImage:

output BMImage.


**Interface2:**
    .. code-block:: cpp

        int Bmcv::transpose(
            BMImage &src,
            BMImage &dst);


**Parameters2:**

* src : BMImage

Input BMImage.

* dst : BMImage

output BMImage.

**Returns2:**

* ret : int

returns 0 if success.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        int main() {  
            int dev_id = 0;
            sail::Handle handle(dev_id);
            std::string image_name = "your_img.jpg";
            sail::Decoder decoder(image_name, true, dev_id);
            sail::BMImage BMimg_input = decoder.read(handle);
            sail::BMImage BMimg_output;
            sail::Bmcv bmcv(handle);
            int ret = bmcv.transpose(BMimg_input,BMimg_output);
            if(ret != 0){
                std::cout << "gaussian_blur failed" << std::endl;
                return -1;
            }
            bmcv.imwrite("output.jpg",BMimg_output);

            return 0;  
        }

watermark_superpose
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Implement adding multiple watermarks to images.

**接口形式:**
    .. code-block:: c

        int Bmcv::watermark_superpose(
        BMImage &img,
        string water_name,
        int bitmap_type,
        int pitch,
        vector<vector<int>> rects,
        const std::tuple<int, int, int> &color);

**参数说明:**

* Image: BMImage

Input image

* Watername: string

Watermark file path

* Bitmap_type: int

Input parameters. Watermark type, a value of 0 indicates that the watermark is an 8-bit data type (with transparency information), and a value of 1 indicates that the watermark is a 1-bit data type (without transparency information).

* Pitch: int

Input parameters. The number of bytes per line in a watermark file can be understood as the width of the watermark.

* Rects: vector

Input parameters. Watermark position, including the starting point and width/height of each watermark.

* Color: const std:: tuple<int, int, int>

Input parameters. The color of the watermark.

**Return value description:**

* Ret: int

Whether the return was successful

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        int main()
        {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            std::string water_name = "your_watermark_path"; 
            int bitmap_type = 0; 
            int pitch =117; 
            std::vector<std::vector<int>> rects = {
                {10, 10, 117, 79}, 
                {200, 150, 117, 79} 
            };
            std:: vector<int> color = {128,128,128}; 
            int result = bmcv.watermark_superpose(BMimg, water_name, bitmap_type, pitch, rects, color);
            if (result == 0) {
                std::cout << "Watermarks added successfully." << std::endl;
            } else {
                std::cout << "Failed to add watermarks." << std::endl;
            }
            return 0; 
        }


gaussian_blur
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

This interface is used for image Gaussian filtering.
**Note: The previous SDK does not support BM1684X. 
For details about whether the current SDK supports BM1684X, check the page "BMCV API" in《Multimedia User Guide》. ***

**Interface1:**

.. code-block:: c

    int gaussian_blur(
        BMImage                      &input,
        BMImage                      &output,
        int                          kw,
        int                          kh,
        float                        sigmaX,
        float                        sigmaY = 0.0f);


**Parameters1:**

* input : BMImage

Input BMImage.

* output : BMImage

Output BMImage.

* kw : int

The size of kernel in the width direction.

* kh : int
  
The size of kernel in the height direction.

* sigmaX : float

Gaussian kernel standard deviation in the X direction.

* sigmaY : float

Gaussian kernel standard deviation in the Y direction.Default is 0, 
which means that it is the same standard deviation as the Gaussian kernel in the X direction.

**Returns1:**

* ret: int

returns 0 if success.

**Interface2:**

.. code-block:: c

    BMImage gaussian_blur(
        BMImage                      &input,
        int                          kw,
        int                          kh,
        float                        sigmaX,
        float                        sigmaY = 0.0f);


**Parameters2:**

* input : BMImage

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

**Returns2:**

* output : BMImage

Returns a Gaussian filtered image.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        int main() {  
            int dev_id = 0;
            sail::Handle handle(dev_id);
            std::string image_name = "your_img.jpg";
            sail::Decoder decoder(image_name, true, dev_id);
            sail::BMImage BMimg_input = decoder.read(handle);
            sail::BMImage BMimg_output;
            sail::Bmcv bmcv(handle);
            int ret = bmcv.gaussian_blur(BMimg_input,BMimg_output,3, 3, 0.1);
            if(ret != 0){
                std::cout << "gaussian_blur failed" << std::endl;
                return -1;
            }
            bmcv.imwrite("output.jpg",BMimg_output);

            return 0;  
        }
        
Sobel
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Sobel operator for edge detection.

**Note:** For details about whether this operator in current SDK supports BM1684X/BM1688, check the page "BMCV API" in 《Multimedia User Guide》. 

**Interface1:**
    .. code-block:: cpp

       int Sobel(
            BMImage &input,
            BMImage &output,
            int dx,
            int dy,
            int ksize = 3,
            float scale = 1,
            float delta = 0);
            
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
    .. code-block:: cpp

        BMImage Sobel(
            BMImage &input,
            int dx,
            int dy,
            int ksize = 3,
            float scale = 1,
            float delta = 0);

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
    .. code-block:: c

        #include <sail/cvwrapper.h>

        int main() {  
            int dev_id = 0;
            sail::Handle handle(dev_id);
            std::string image_name = "your_img.jpg";
            sail::Decoder decoder(image_name, true, dev_id);
            sail::BMImage BMimg_input = decoder.read(handle);
            sail::BMImage BMimg_output;
            sail::Bmcv bmcv(handle);
            int ret = bmcv.Sobel(BMimg_input,BMimg_output,1,1);
            if(ret != 0){
                std::cout << "Sobel failed" << std::endl;
                return -1;
            }
            bmcv.imwrite("output.jpg",BMimg_output);

            return 0;  
        }

drawLines
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

This function can be used to draw one or several line segments on an image, which can be used to draw polygons. It supports specifying the color and thickness of the lines.

**Note:** For details about whether this operator in current SDK supports BM1684X, check the page "BMCV API" in 《Multimedia User Guide》. 

**Interface:**
    .. code-block:: c

        int Bmcv::drawLines(
        BMImage &image,
        std::vector<std::pair<int,int>> &start_points,
        std::vector<std::pair<int,int>> &end_points,
        int line_num,
        std::tuple<unsigned char, unsigned char, unsigned char> color,
        int thickness
        );


**Parameters:**

* image : BMImage

Input image, needs to be one of the supported formats.

* start_points : std::vector<std::pair<int,int>>

List of coordinates for the starting points of the line segments.

* end_points : std::vector<std::pair<int,int>>
  
List of coordinates for the ending points of the line segments. The size of start_points and end_points must be the same and match the line_num parameter.

* line_num :  int

The number of line segments to draw.

* color : std::tuple<unsigned char, unsigned char, unsigned char> 

The color of the line segments, corresponding to the RGB channels.

* thickness : int

The thickness of the line segments.

**Returns:**

* ret: int

returns 0 if success.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        int main()
        {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            std::string image_name = "your_image_path";
            sail::Decoder decoder(image_name, true, tpu_id);
            sail::BMImage BMimg = decoder.read(handle); 
            sail::Bmcv bmcv(handle);
            std::vector<std::pair<int, int>> start_points = {{100, 100}, {200, 200}};
            std::vector<std::pair<int, int>> end_points = {{150, 150}, {250, 250}};
            int line_num = 2;
            std::tuple<unsigned char, unsigned char, unsigned char> color = std::make_tuple(255, 0, 0); 
            int thickness = 2;
            sail::BMImage BMimg2;
            BMimg2 = bmcv.vpp_convert_format(BMimg,FORMAT_YUV420P);
            int ret = bmcv.drawLines(BMimg2, start_points, end_points, line_num, color, thickness);
            
            return 0;
        }

stft
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Short-Time Fourier Transform(STFT)

**Note:** For details about whether this operator in current SDK supports BM1684X, check the page "BMCV API" in 《Multimedia User Guide》. 

**Interface1:**
    .. code-block:: c

        std::tuple<Tensor, Tensor> stft(
            Tensor &input_real,
            Tensor &input_imag,
            bool realInput,
            bool normalize,
            int n_fft,
            int hop_len,
            int pad_mode,
            int win_mode
            );

**参数说明:**

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
    The padding mode for the input signal.

* win_mode: int
    The type of window function.

**Returns:**

* result: tuple[Tensor, Tensor]
    Returns the real part and imaginary part of the output.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main()
        {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            sail::Bmcv bmcv(handle);
            std::vector<int> shape = {2, 4096};
            sail::Tensor input_real(shape);
            sail::Tensor input_imag(shape);
            bool real_input = false;
            bool normalize = true;
            int n_fft = 1024;
            int hop_len = 256;
            int pad_mode = 0;  // 填充模式示例
            int win_mode = 1;  // 窗口类型示例
            std::tuple<sail::Tensor, sail::Tensor> result = bmcv.stft(input_real, input_imag, realInput, normalize, n_fft, hop_len, pad_mode, win_mode);
            return 0;
        }

istft
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Inverse Short-Time Fourier Transform(ISTFT)

**Note:** For details about whether this operator in current SDK supports BM1684X, check the page "BMCV API" in 《Multimedia User Guide》. 

**Interface1:**
    .. code-block:: C

        std::tuple<Tensor, Tensor> istft(
            Tensor &input_real,
            Tensor &input_imag,
            bool realInput,
            bool normalize,
            int L,
            int hop_len,
            int pad_mode,
            int win_mode
            );   

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

* result: tuple[Tensor, Tensor]
    Returns the real part and imaginary part of the output.

**Sample:**
    .. code-block:: C

        #include <sail/cvwrapper.h>
        using namespace std;
        int main()
        {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            sail::Bmcv bmcv(handle);
            std::vector<int> shape = {2, 513, 17};
            sail::Tensor input_real(shape);
            sail::Tensor input_imag(shape);
            bool real_input = false;
            bool normalize = true;
            int L = 4096;
            int hop_len = 256;
            int pad_mode = 0;
            int win_mode = 1;
            std::tuple<sail::Tensor, sail::Tensor> result = bmcv.istft(input_real, input_imag, realInput, normalize, L, hop_len, pad_mode, win_mode);
            return 0;
        }