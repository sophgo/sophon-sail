Bmcv
_________

Bmcv封装了常用的图像处理接口，支持硬件加速。

构造函数
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

初始化Bmcv

**接口形式:**
    .. code-block:: c

        Bmcv(Handle handle);
          

**参数说明:**

* handle: Handle

指定Bmcv使用的设备句柄。


bm_image_to_tensor
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

将BMImage/BMImageArray转换为Tensor。

**接口形式1:**
    .. code-block:: c


        Tensor bm_image_to_tensor(BMImage &image);
           

**参数说明1:**

* image: BMImage

需要转换的图像数据。


**返回值说明1:**

* tensor: Tensor

返回转换后的Tensor。

**示例代码:**
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

**接口形式2:**
    .. code-block:: c

        void bm_image_to_tensor(BMImage &img, Tensor &tensor);
           
            
**参数说明2:**

* image: BMImageArray

输入参数。需要转换的图像数据。

* tensor: Tensor

输出参数。转换后的Tensor。

**示例代码:**
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

将Tensor转换为BMImage/BMImageArray。

**接口形式1:**
    .. code-block:: c

        void tensor_to_bm_image(Tensor &tensor, BMImage &img, bool bgr2rgb=false, std::string layout = std::string("nchw"));

        void tensor_to_bm_image(Tensor &tensor, BMImage &img, bm_image_format_ext format_);

        BMImage tensor_to_bm_image(Tensor &tensor, bool bgr2rgb=false, std::string layout = std::string("nchw"));

        BMImage tensor_to_bm_image (Tensor &tensor, bm_image_format_ext format_);

**参数说明1:**

* tensor: Tensor

输入参数。待转换的Tensor。

* img : BMImage

转换后的图像。

**返回值说明1:**

* image : BMImage

返回转换后的图像。


**接口形式2:**
    .. code-block:: c

        template<std::size_t N> void   bm_image_to_tensor (BMImageArray<N> &imgs, Tensor &tensor);
        template<std::size_t N> Tensor bm_image_to_tensor (BMImageArray<N> &imgs);
            

**参数说明2:**

* tensor: Tensor

输入参数。待转换的Tensor。

* img : BMImage | BMImageArray

输出参数。返回转换后的图像。

**返回值说明2:**

* image : Tensor

返回转换后的tensor。

**示例代码1:**
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

**示例代码2:**
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

对图片进行裁剪并resize。

**接口形式:**
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

**参数说明:**

* input : BMImage | BMImageArray

待处理的图像或图像数组。

* output : BMImage | BMImageArray

处理后的图像或图像数组。

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

* resize_alg : bmcv_resize_algorithm

图像resize的插值算法，默认为bmcv_resize_algorithm.BMCV_INTER_NEAREST

**返回值说明:**

* ret: int

返回0代表成功，其他代表失败。

* output : BMImage | BMImageArray

返回处理后的图像或图像数组。

**示例代码1:**
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

**示例代码2:**
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

对图像进行裁剪。

**接口形式:**
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
            

**参数说明:**

* input : BMImage | BMImageArray

待处理的图像或图像数组。

* output : BMImage | BMImageArray

处理后的图像或图像数组。

* crop_x0 : int

裁剪窗口在x轴上的起始点。

* crop_y0 : int

裁剪窗口在y轴上的起始点。

* crop_w : int 

裁剪窗口的宽。

* crop_h : int 

裁剪窗口的高。

**返回值说明:**

* ret: int

返回0代表成功，其他代表失败。

* output : BMImage | BMImageArray

返回处理后的图像或图像数组。

**示例代码1:**
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

**示例代码2:**
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

对图像进行resize。

**接口形式:**
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

**参数说明:**

* input : BMImage | BMImageArray

待处理的图像或图像数组。

* output : BMImage | BMImageArray

处理后的图像或图像数组。

* resize_w : int

图像resize的目标宽度。

* resize_h : int

图像resize的目标高度。

* resize_alg : bmcv_resize_algorithm

图像resize的插值算法，默认为bmcv_resize_algorithm.BMCV_INTER_NEAREST

**返回值说明:**

* ret: int

返回0代表成功，其他代表失败。

* output : BMImage | BMImageArray

返回处理后的图像或图像数组。

**示例代码1:**
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

**示例代码2:**
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

利用VPP硬件加速图片的裁剪与resize。

**接口形式:**
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

**参数说明:**

* input : BMImage | BMImageArray

待处理的图像或图像数组。

* output : BMImage | BMImageArray

处理后的图像或图像数组。

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

* resize_alg : bmcv_resize_algorithm

图像resize的插值算法，默认为bmcv_resize_algorithm.BMCV_INTER_NEAREST

**返回值说明:**

* ret: int

返回0代表成功，其他代表失败。

* output : BMImage | BMImageArray

返回处理后的图像或图像数组。

**示例代码1:**
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

**示例代码2:**
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

利用VPP硬件加速图片的裁剪与resize，并padding到指定大小。

**接口形式:**
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

**参数说明:**

* input : BMImage | BMImageArray

待处理的图像或图像数组。

* output : BMImage | BMImageArray

处理后的图像或图像数组。

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

* padding : PaddingAtrr

padding的配置信息。

* resize_alg : bmcv_resize_algorithm

图像resize的插值算法，默认为bmcv_resize_algorithm.BMCV_INTER_NEAREST

**返回值说明:**

* ret: int

返回0代表成功，其他代表失败。

* output : BMImage | BMImageArray

返回处理后的图像或图像数组。

**示例代码1:**
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

**示例代码2:**
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

利用VPP硬件加速图片的裁剪。

**接口形式:**
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

**参数说明:**

* input : BMImage | BMImageArray

待处理的图像或图像数组。

* output : BMImage | BMImageArray

处理后的图像或图像数组。

* crop_x0 : int

裁剪窗口在x轴上的起始点。

* crop_y0 : int

裁剪窗口在y轴上的起始点。

* crop_w : int 

裁剪窗口的宽。

* crop_h : int 

裁剪窗口的高。

**返回值说明:**

* ret: int

返回0代表成功，其他代表失败。

* output : BMImage | BMImageArray

返回处理后的图像或图像数组。

**示例代码1:**
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

**示例代码2:**
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

利用VPP硬件加速图片的resize，采用最近邻插值算法。 

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

**参数说明:**

* input : BMImage | BMImageArray

待处理的图像或图像数组。

* output : BMImage | BMImageArray

处理后的图像或图像数组。

* resize_w : int

图像resize的目标宽度。

* resize_h : int

图像resize的目标高度。

* resize_alg : bmcv_resize_algorithm

图像resize的插值算法，默认为bmcv_resize_algorithm.BMCV_INTER_NEAREST

**返回值说明:**

* ret: int

返回0代表成功，其他代表失败。

* output : BMImage | BMImageArray

返回处理后的图像或图像数组。

**示例代码1:**
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

**示例代码2:**
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

利用VPP硬件加速图片的resize，并padding。

**接口形式:**
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

**参数说明:**

* input : BMImage | BMImageArray

待处理的图像或图像数组。

* resize_w : int

图像resize的目标宽度。

* resize_h : int

图像resize的目标高度。

* padding : PaddingAtrr

padding的配置信息。

**返回值说明:**

* ret: int

返回0代表成功，其他代表失败。

* output : BMImage | BMImageArray

返回处理后的图像或图像数组。

* resize_alg : bmcv_resize_algorithm

图像resize的插值算法，默认为bmcv_resize_algorithm.BMCV_INTER_NEAREST

**示例代码1:**
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

**示例代码2:**
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

对图像进行仿射变换。

**接口形式:**
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

**参数说明:**

* input : BMImage | BMImageArray

待处理的图像或图像数组。

* output : BMImage | BMImageArray

处理后的图像或图像数组。

* matrix: std::pair<
             std::tuple<float, float, float>,
             std::tuple<float, float, float> >

2x3的仿射变换矩阵。

* use_bilinear: int

是否使用双线性插值，默认为0使用最近邻插值，1为双线性插值

* similar_to_opencv: bool

是否使用与opencv仿射变换对齐的接口

**返回值说明:**

* ret: int

返回0代表成功，其他代表失败。

* output : BMImage | BMImageArray

返回处理后的图像或图像数组。

**示例代码1:**
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

**示例代码2:**
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

对图像进行线性变换。

**接口形式:**
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
    
**参数说明:**

* input : BMImage | BMImageArray

待处理的图像或图像数组。

* alpha_beta: std::tuple<
             std::pair<float, float>,
             std::pair<float, float>,
             std::pair<float, float> > 

分别为三个通道线性变换的系数((a0, b0), (a1, b1), (a2, b2))。

* output : BMImage | BMImageArray

输出参数。处理后的图像或图像数组。

**返回值说明:**

* ret: int

返回0代表成功，其他代表失败。

* output : BMImage | BMImageArray

返回处理后的图像或图像数组。

**示例代码:**
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

**示例代码:**
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

将图像的格式从YUV转换为BGR。

**接口形式:**
    .. code-block:: c

        int yuv2bgr(
           BMImage                      &input,
           BMImage                      &output);

        BMImage yuv2bgr(BMImage  &input);

**参数说明:**

* input : BMImage | BMImageArray

待转换的图像。

**返回值说明:**

* ret: int

返回0代表成功，其他代表失败。

* output : BMImage | BMImageArray

返回转换后的图像。

**示例代码1:**
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

**示例代码2:**
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

在图像上画一个矩形框。

**接口形式:**
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

**参数说明:**

* image : BMImage | bm_image

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

在图像上填充一个矩形。

**接口形式:**
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

**参数说明:**

* image : BMImage | bm_image

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

将图像保存在特定文件。

**接口形式:**
    .. code-block:: c

        int imwrite(
           const std::string &filename,
           BMImage           &image);
        
        int imwrite(
           const std::string &filename,
           const bm_image     &image);

**参数说明:**

* file_name : string

文件的名称。

* output : BMImage | bm_image

需要保存的图像。

**返回值说明:**

* process_status : int

如果保存成功返回0，否则返回非0值。

**示例代码:**
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


imwrite
>>>>>>>>>>>>>>>>>
     
读取和解码图片文件，仅支持 JPEG baseline 格式的硬解码。对于其他格式，如 PNG 和 BMP，则采用软解码。
对于 JPEG baseline 图片，返回的 BMImage 将保持 YUV 色彩空间，像素格式依据图片文件本身的采样方式，例如 YUV420；
而对于其他格式，返回的 BMImage 将保持其输入的对应色彩空间。

**接口形式:**
    .. code-block:: c++

        BMImage imread(const std::string &filename);

**参数说明:**

* filename : string

需要读取的图片文件路径。

**返回值说明:**

* output : BMImage

返回解码得到的BMImage。

**示例代码:**
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

获取Bmcv中的设备句柄Handle。

**接口形式:**
    .. code-block:: c

        Handle get_handle();

**返回值说明:**

* handle: Handle

Bmcv中的设备句柄Handle。

**示例代码:**
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

对图像进行裁剪并resize，然后padding。

**接口形式:**
    .. code-block:: c

        bm_image crop_and_resize_padding(
            bm_image                      &input,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h,
            PaddingAtrr                  &padding_in,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);
        
        BMImage crop_and_resize_padding(
            BMImage                      &input,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h,
            PaddingAtrr                  &padding_in,
            bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

**参数说明:**

* input : BMImage/bm_image

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

* padding : PaddingAtrr

padding的配置信息。

* resize_alg : bmcv_resize_algorithm

resize采用的插值算法。

**返回值说明:**

* output : BMImage/bm_image

返回处理后的图像。

**示例代码1:**
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

**示例代码2:**
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

将图像的格式转换为output中的格式，并拷贝到output。

**接口形式1:**
    .. code-block:: c

        int convert_format(
            BMImage          &input,
            BMImage          &output
        );

**参数说明1:**

* input : BMImage

输入参数。待转换的图像。

* output : BMImage

输出参数。将input中的图像转化为output的图像格式并拷贝到output。

**示例代码:**
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

**接口形式2:**

将一张图像转换成目标格式。

    .. code-block:: c

        BMImage convert_format(
            BMImage          &input,
            bm_image_format_ext image_format = FORMAT_BGR_PLANAR
        );

**参数说明2:**

* input : BMImage

待转换的图像。

* image_format : bm_image_format_ext

转换的目标格式。

**返回值说明2:**

* output : BMImage

返回转换后的图像。

**示例代码:**
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

利用VPP硬件加速图片的格式转换。

**接口形式1:**
    .. code-block:: c

        int vpp_convert_format(
            BMImage          &input,
            BMImage          &output
        );

**参数说明1:**

* input : BMImage

输入参数。待转换的图像。

* output : BMImage

输出参数。将input中的图像转化为output的图像格式并拷贝到output。

**示例代码:**
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

**接口形式2:**

将一张图像转换成目标格式。

    .. code-block:: c

        BMImage vpp_convert_format(
            BMImage          &input,
            bm_image_format_ext image_format = FORMAT_BGR_PLANAR
        );

**参数说明2:**

* input : BMImage

待转换的图像。

* image_format : bm_image_format_ext

转换的目标格式。

**返回值说明2:**

* output : BMImage

返回转换后的图像。

**示例代码:**
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

在图像上添加text。只支持英文文字。

输入的BMImage支持的像素格式为：
FORMAT_GRAY、FORMAT_YUV420P、FORMAT_YUV422P、FORMAT_YUV444P、FORMAT_NV12、
FORMAT_NV21、FORMAT_NV16、FORMAT_NV61。

**接口形式:**
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

**参数说明:**

* input : BMImage | bm_image

待处理的图像。

* text: string

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

字体的粗细。

**返回值说明:**

* process_status : int

如果处理成功返回0，否则返回非0值。

**示例代码:**
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

将两张图像按不同的权重相加。

**接口形式1:**
    .. code-block:: c
        
        int image_add_weighted(
            BMImage           &input1,
            float             alpha,
            BMImage           &input2,
            float             beta,
            float             gamma,
            BMImage           &output
        );

**参数说明1:**

* input0 : BMImage

输入参数。待处理的图像0。

* alpha : float

输入参数。两张图像相加的权重alpha

* input1 : BMImage

输入参数。待处理的图像1。

* beta : float

输入参数。两张图像相加的权重beta

* gamma : float

输入参数。两张图像相加的权重gamma

* output: BMImage

输出参数。相加后的图像output = input1 * alpha + input2 * beta + gamma

**示例代码:**
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

**接口形式2:**
    .. code-block:: c

        BMImage image_add_weighted(
            BMImage           &input1,
            float             alpha,
            BMImage           &input2,
            float             beta,
            float             gamma
        );


**参数说明2:**

* input0 : BMImage

输入参数。待处理的图像0。

* alpha : float

输入参数。两张图像相加的权重alpha

* input1 : BMImage

输入参数。待处理的图像1。

* beta : float

输入参数。两张图像相加的权重beta

* gamma : float

输入参数。两张图像相加的权重gamma

**返回值说明2:**

* output: BMImage

返回相加后的图像output = input1 * alpha + input2 * beta + gamma

**示例代码:**
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

进行图像间的数据拷贝

**接口形式:**
    .. code-block:: c

        int image_copy_to(BMImage &input, BMImage &output, int start_x = 0, int start_y = 0);

        template<std::size_t N>
        int image_copy_to(BMImageArray<N> &input, BMImageArray<N> &output, int start_x = 0, int start_y = 0);

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

进行input和output间的图像数据拷贝并padding。

**接口形式:**
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

利用智能视觉深度学习处理器进行NMS

**注意：请查询《BMCV开发参考手册/BMCV API》确认当前算子是否适配BM1688。**

**接口形式:**
    .. code-block:: c

        nms_proposal_t* nms(
            face_rect_t *input_proposal,
            int proposal_size, 
            float threshold);

**参数说明:**

* input_proposal: face_rect_t

数据起始地址。

* proposal_size: int

待处理的检测框数据的大小。

* threshold: float

nms的阈值。

**返回值说明:**

* result: nms_proposal_t

返回NMS后的检测框数组。

**示例代码:**
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

在图像上画点。

**接口形式:**
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


**参数说明:**

* image: BMImage | bm_image

输入图像，在该BMImage上直接画点作为输出。

* center: std::pair<int,int>

点的中心坐标。

* color: std::tuple<unsigned char, unsigned char, unsigned char>

点的颜色。

* radius: int

点的半径。

**返回值说明**

如果画点成功返回0，否则返回非0值。

**示例代码:**
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

对图像进行透视变换。

**接口形式:**
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

**参数说明:**

* input: BMImage

待处理的图像。

* coordinate: std::tuple<
            std::pair<int,int>,
            std::pair<int,int>,
            std::pair<int,int>,
            std::pair<int,int> >

变换区域的四顶点原始坐标。

例如((left_top.x, left_top.y), (right_top.x, right_top.y),
(left_bottom.x, left_bottom.y), (right_bottom.x, right_bottom.y))

* output_width: int

输出图像的宽。

* output_height: int

输出图像的高。

* format: bm_image_format_ext

输出图像的格式。

* dtype: bm_image_data_format_ext

输出图像的数据类型。

* use_bilinear: int

是否使用双线性插值。

**返回值说明:**

* output: BMImage

输出变换后的图像。

**示例代码:**
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

将ImgDtype转换为Dtype

**接口形式:**
    .. code-block:: c

        bm_data_type_t get_bm_data_type(bm_image_data_format_ext fmt);

**参数说明:**

* fmt: bm_image_data_format_ext

需要转换的类型。

**返回值说明:**

* ret: bm_data_type_t

转换后的类型。

**示例代码:**
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

将Dtype转换为ImgDtype。

**接口形式:**
    .. code-block:: c

        bm_image_data_format_ext get_bm_image_data_format(bm_data_type_t dtype);

**参数说明:**

* dtype: bm_data_type_t

需要转换的Dtype

**返回值说明:**

* ret: bm_image_data_format_ext

返回转换后的类型。

**示例代码:**
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

从内存中载入图像到BMImage中。

**接口形式:**
    .. code-block:: c

        BMImage imdecode(const void* data_ptr, size_t data_size);
          
**参数说明:**

* data_ptr: void*

数据起始地址

* data_size: bytes

数据长度

**返回值说明:**

* ret: BMImage

返回解码后的图像。

**示例代码:**
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

编码一张图片，并返回编码后的数据。

**接口形式1:**
    .. code-block:: c

        bool Bmcv::imencode(std::string& ext, bm_image &img, std::vector<u_char>& buf)

**接口形式2:**
    .. code-block:: c

        bool Bmcv::imencode(std::string& ext, BMImage &img, std::vector<u_char>& buf)
          
**参数说明:**

* ext: string

输入参数。图片编码格式。 ``".jpg"`` , ``".png"`` 等。

* image: bm_image/BMImage

输入参数。输入图片，只支持FORMAT_BGR_PACKED，DATA_TYPE_EXT_1N_BYTE的图片。

* buf: std::vector<u_char>

输出参数。编码后放在系统内存中的数据。

**返回值说明:**

* ret: bool

编码成功时返回0，失败时返回1。

**示例代码:**
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

实现对Tensor的快速傅里叶变换。

**注意：请查询《BMCV开发参考手册/BMCV API》确认当前算子是否适配BM1688。**

**接口形式:**
    .. code-block:: c

        std::vector<Tensor> fft(bool forward, Tensor &input_real);

        std::vector<Tensor> fft(bool forward, Tensor &input_real, Tensor &input_imag);
    
**参数说明:**

* forward: bool

是否进行正向迁移。

* input_real: Tensor

输入的实数部分。

* input_imag: Tensor

输入的虚数部分。

**返回值说明:**

* ret: std::vector<Tensor>

返回输出的实数部分和虚数部分。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        int main()
        {
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            sail::Bmcv bmcv(handle);
            std::vector<int> shape = {1,1,512, 512};
            sail::Tensor input_real(shape);
            bool forward = true; 
            //std::vector<sail::Tensor> result_real = bmcv.fft(forward, input_real);  
            sail::Tensor input_imag(shape);
            std::vector<sail::Tensor> result_complex  = bmcv.fft(forward, input_real,input_imag);
            return 0;
        }

convert_yuv420p_to_gray
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

将YUV420P格式的图片转为灰度图。

**接口形式1:**
    .. code-block:: c

        int convert_yuv420p_to_gray(BMImage& input, BMImage& output);

**参数说明1:**

* input : BMImage

输入参数。待转换的图像。

* output : BMImage

输出参数。转换后的图像。

**示例代码:**
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

**接口形式2:**

将YUV420P格式的图片转为灰度图。

    .. code-block:: c

        int convert_yuv420p_to_gray_(bm_image& input, bm_image& output); 

**参数说明2:**

* input : bm_image

待转换的图像。

* output : bm_image

转换后的图像。

**示例代码:**
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
            int ret = bmcv.convert_yuv420p_to_gray_(BMimg.data(), img.data());
            return 0;
        }

watermark_superpose
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

实现对图片添加多个水印。

**接口形式:**
    .. code-block:: c

        int Bmcv::watermark_superpose(
        BMImage &img,
        string water_name,
        int bitmap_type,
        int pitch,
        vector<vector<int>> rects,
        vector<int> &color);

**参数说明:**

* image: BMImage

输入图片

* water_name:string

水印文件路径

* bitmap_type: int

输入参数。水印类型, 值0表示水印为8bit数据类型(有透明度信息), 值1表示水印为1bit数据类型(无透明度信息)。

* pitch: int

输入参数。水印文件每行的byte数, 可理解为水印的宽。

* rects: vector<vector<int>>

输入参数。水印位置，包含每个水印起始点和宽高。

* color: vector<int>

输入参数。水印的颜色。

**返回值说明:**

* ret: int

返回是否成功

**示例代码:**
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

polylines
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

可以实现在一张图像上画一条或多条线段，从而可以实现画多边形的功能，并支持指定线的颜色和线的宽度。

**接口形式:**
    .. code-block:: c

        int polylines(
            BMImage &img,
            std::vector<std::vector<std::pair<int,int>>> &pts,
            bool isClosed,
            std::tuple<unsigned char, unsigned char, unsigned char> color,
            int thickness = 1,
            int shift = 0);

**参数说明:**


* img : BMImage

输入图片。

* pts : std::vector<std::vector<std::pair<int,int>>>

线段的起始点和终点坐标，可输入多个坐标点。图像左上角为原点，向右延伸为x方向，向下延伸为y方向。

* isClosed : bool
  
图形是否闭合。

* color : std::tuple<unsigned char, unsigned char, unsigned char>

画线的颜色，分别为RGB三个通道的值。

* thickness : int 

画线的宽度，对于YUV格式的图像建议设置为偶数。

* shift : int

多边形缩放倍数，默认不缩放。缩放倍数为(1/2)^shift。

**返回值说明:**

* ret: int

成功后返回0

**示例代码:**
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

该接口用于在图像上打一个或多个马赛克。

**接口形式:**
    .. code-block:: c

         int mosaic(
            int mosaic_num,
            BMImage &img,
            vector<vector<int>> rects,
            int is_expand);

**参数说明:**

* mosaic_num : int

马赛克数量，指rects中列表长度。

* img : BMImage

待转换的图像。

* rects : vector<vector<int>>

多个马赛克位置，列表中每个元素中参数为[马赛克在x轴起始点,马赛克在y轴起始点,马赛克宽,马赛克高]

* is_expand : int
  
是否扩列。值为0时表示不扩列, 值为1时表示在原马赛克周围扩列一个宏块(8个像素)。

**返回值说明:**

* ret: int

成功后返回0

**示例代码:**
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


gaussian_blur
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

该接口用于对图像进行高斯滤波。
**注意：旧版本SDK并不支持BM1684X，当前SDK是否支持请查询《BMCV开发参考手册/BMCV API》查看。**

**接口形式1:**

.. code-block:: c
    
    int gaussian_blur(
        BMImage                      &input,
        BMImage                      &output,
        int                          kw,
        int                          kh,
        float                        sigmaX,
        float                        sigmaY = 0.0f);

**参数说明1:**

* input : BMImage

待转换的图像。

* output : BMImage

转换后输出的图像。

* kw : int

kernel 在width方向上的大小。

* kh : int
  
kernel 在height方向上的大小。

* sigmaX : float

X方向上的高斯核标准差。

* sigmaY : float

Y方向上的高斯核标准差。如果为0则表示与X方向上的高斯核标准差相同。默认为0。

**返回值说明1:**

* ret: int

成功后返回0

**接口形式2:**
.. code-block:: c

    BMImage gaussian_blur(
        BMImage                      &input,
        int                          kw,
        int                          kh,
        float                        sigmaX,
        float                        sigmaY = 0.0f);

**参数说明2:**
* input : BMImage
  
待转换的图像。

* kw : int

kernel 在width方向上的大小。

* kh : int
  
kernel 在height方向上的大小。

* sigmaX : float

X方向上的高斯核标准差。

* sigmaY : float

Y方向上的高斯核标准差。如果为0则表示与X方向上的高斯核标准差相同。默认为0。


**返回值说明2:**

* output: BMImage

返回经过高斯滤波的图像。

**示例代码:**
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
        


transpose
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

该接口可以实现图片宽和高的转置。

**接口形式1:**
    .. code-block:: cpp

        BMImage Bmcv::transpose(
            BMImage &src);
    


**参数说明1:**

* src : BMImage

待转换的图像。


**返回值说明1:**

* output: BMImage:

返回转换后的图像。


**接口形式2:**
    .. code-block:: cpp

        int Bmcv::transpose(
            BMImage &src,
            BMImage &dst);

**参数说明2:**

* src : BMImage

待转换的图像。

* dst : BMImage

输出图像的 BMImage 结构体。

**返回值说明2:**

* ret : int

成功返回0，否则返回非0值。


**示例代码:**
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


Sobel
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

边缘检测Sobel算子。

**注意：请查询《BMCV开发参考手册/BMCV API》确认当前算子是否适配BM1684X、BM1688。**

**接口形式1:**
    .. code-block:: cpp

        int Sobel(
            BMImage &input,
            BMImage &output,
            int dx,
            int dy,
            int ksize = 3,
            float scale = 1,
            float delta = 0);

**参数说明1:**

* input : BMImage

待转换的图像。

* output : BMImage

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
    .. code-block:: cpp

        BMImage Sobel(
            BMImage &input,
            int dx,
            int dy,
            int ksize = 3,
            float scale = 1,
            float delta = 0);

**参数说明2:**

* input : BMImage

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

* output: BMImage

返回转换后的图像。

**示例代码:**
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

可以实现在一张图像上画一条或多条线段，从而可以实现画多边形的功能，并支持指定线的颜色和线的宽度。

**注意：请查询《BMCV开发参考手册/BMCV API》确认当前算子是否适配BM1684X。**

**接口形式:**
    .. code-block:: c

        int Bmcv::drawLines(
            BMImage &image,
            std::vector<std::pair<int,int>> &start_points,
            std::vector<std::pair<int,int>> &end_points,
            int line_num,
            std::tuple<unsigned char, unsigned char, unsigned char> color,
            int thickness
        );


**参数说明:**

* img : BMImage

输入图片，需要是支持的格式之一。

* start_points : std::vector<std::pair<int,int>>

线段的起始点坐标列表。

* end_points : std::vector<std::pair<int,int>>
  
线段的结束点坐标列表。start_points 和 end_points 的size必须相同，并且与 line_num 参数相匹配。

* line_num  : int

要绘制的线段数量。

* color : std::tuple<unsigned char, unsigned char, unsigned char> 

线段的颜色，分别为RGB三个通道的值。

* thickness : int

线段的宽度。

**返回值说明:**

* ret: int

成功后返回0。

**示例代码:**
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

实现对信号的短时傅里叶变换（STFT）。

**注意：请查询《BMCV开发参考手册/BMCV API》确认当前算子是否适配BM1684X。**

**接口形式:**
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

* input_real: Tensor
    输入信号的实数部分。

* input_imag: Tensor
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

* result: tuple[Tensor, Tensor]
    返回输出的实数部分和虚数部分。

**示例代码:**
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

实现对信号的逆短时傅里叶变换（ISTFT）。

**注意：请查询《BMCV开发参考手册/BMCV API》确认当前算子是否适配BM1684X。**

**接口形式:**
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

* result: tuple[Tensor, Tensor]
    返回输出的实数部分和虚数部分。

**示例代码:**
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
            int pad_mode = 0;  // 填充模式示例
            int win_mode = 1;  // 窗口类型示例
            std::tuple<sail::Tensor, sail::Tensor> result = bmcv.istft(input_real, input_imag, realInput, normalize, L, hop_len, pad_mode, win_mode);
            return 0;
        }

faiss_indexflatL2
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

计算查询向量与数据库向量 L2 距离的平方, 输出前 topK 个最匹配的 L2 距离的平方值及其对应的索引。

**接口形式:**
    .. code-block:: C

        std::tuple<Tensor, Tensor> faiss_indexflatL2(
            Tensor &query_vecs,
            Tensor &query_vecs_L2norm,
            Tensor &database_vecs,
            Tensor& database_vecs_L2norm,
            int vec_dims,
            int query_vecs_nums,
            int database_vecs_nums,
            int topK
            );

**参数说明:**

* query_vecs: Tensor
    查询向量, 数据类型仅支持sail.Dtype.BM_FLOAT32。

* query_vecs_L2norm: Tensor
    计算查询向量每行各元素平方值的和, 数据类型为sail.Dtype.BM_FLOAT32。

* database_vecs: Tensor
    数据库向量, 数据类型仅支持sail.Dtype.BM_FLOAT32。

* database_vecs_L2norm: Tensor
    计算数据库向量每行各元素平方值的和, 数据类型为sail.Dtype.BM_FLOAT32。

* vec_dims: int
    查询向量和数据库向量的维度。

* query_vecs_nums: int
    查询向量的个数。

* database_vecs_nums: int
    数据库向量的个数。

* topK: int
    输出前 topK 个最匹配的 L2 距离的平方值及其对应的索引。

**返回值说明:**

* result: tuple[Tensor, Tensor]
    返回前 topK 个最匹配的 L2 距离的平方值及其对应的索引。

**示例代码:**
    .. code-block:: C

        #include <iostream>
        #include <vector>
        #include <sail/cvwrapper.h>

        int main(){
            // 1. database_vecs
            std::vector<std::vector<float>> db_vecs = {
                {-2.0f, 0.0f, 4.0f},
                {-5.0f, 3.0f, -1.0f},
                {1.0f, 2.0f, 4.0f},
                {0.0f, 5.0f, -3.0f},
                {2.0f, 1.0f, -4.0f}
            };

            // 2. query_vecs
            std::vector<std::vector<float>> query_vecs = {
                {1.0f, 2.0f, 3.0f}
            };

            // 3. test bmcv.faiss_indexflatL2
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            sail::Bmcv bmcv(handle);

            std::vector<float> db_vecs_l2norm;
            for (const auto& vec : db_vecs) {
                float sum = 0.0f;
                for (const auto& val : vec) {
                    sum += val * val;
                }
                db_vecs_l2norm.push_back(sum);
            }

            std::vector<float> query_vecs_l2norm;
            for (const auto& vec : query_vecs) {
                float sum = 0.0f;
                for (const auto& val : vec) {
                    sum += val * val;
                }
                query_vecs_l2norm.push_back(sum);
            }

            sail::Tensor database_vecs_tensor = Tensor(handle, {db_vecs.size(), db_vecs[0].size()}, BM_FLOAT32, true, true);
            database_vecs_tensor.reset_sys_data(db_vecs.data(), {db_vecs.size(), db_vecs[0].size()});
            database_vecs_tensor.sync_s2d();

            sail::Tensor database_vecs_L2norm_tensor = Tensor(handle, {1, db_vecs.size()}, BM_FLOAT32, true, true);
            database_vecs_L2norm_tensor.reset_sys_data(db_vecs_l2norm, {1, db_vecs.size()});
            database_vecs_L2norm_tensor.sync_s2d();

            sail::Tensor query_vecs_tensor = Tensor(handle, {query_vecs.size(), query_vecs[0].size()}, BM_FLOAT32, true, true);
            query_vecs_tensor.reset_sys_data(query_vecs.data(), {query_vecs.size(), query_vecs[0].size()});
            query_vecs_tensor.sync_s2d();

            sail::Tensor query_vecs_L2norm_tensor = Tensor(handle, {1, query_vecs.size()}, BM_FLOAT32, true, true);
            query_vecs_L2norm_tensor.reset_sys_data(query_vecs_l2norm.data(), {1, query_vecs.size()});
            query_vecs_L2norm_tensor.sync_s2d();

            std::tuple<sail::Tensor, sail::Tensor> result = bmcv.faiss_indexflatL2(query_vecs_tensor, query_vecs_L2norm_tensor, database_vecs_tensor, database_vecs_L2norm_tensor, 3, 1, 5, 3);

            return 0;
        }

faiss_indexflatIP
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

计算查询向量与数据库向量的内积距离, 输出前 topK 个最匹配的内积距离值及其对应的索引。

**接口形式:**
    .. code-block:: C

        std::tuple<Tensor, Tensor> faiss_indexflatIP(
            Tensor &query_vecs,
            Tensor &database_vecs,
            int vec_dims,
            int query_vecs_nums,
            int database_vecs_nums,
            int topK
            );

**参数说明:**

* query_vecs: Tensor
    查询向量, 数据类型仅支持sail.Dtype.BM_FLOAT32。

* database_vecs: Tensor
    数据库向量, 数据类型仅支持sail.Dtype.BM_FLOAT32。

* vec_dims: int
    查询向量和数据库向量的维度。

* query_vecs_nums: int
    查询向量的个数。

* database_vecs_nums: int
    数据库向量的个数。

* topK: int
    输出前 topK 个最匹配的内积距离值及其对应的索引。

**返回值说明:**

* result: tuple[Tensor, Tensor]
    输出前 topK 个最匹配的内积距离值及其对应的索引。

**示例代码:**
    .. code-block:: C

        #include <iostream>
        #include <vector>
        #include <sail/cvwrapper.h>

        int main(){
            // 1. database_vecs
            std::vector<std::vector<float>> db_vecs = {
                {-2.0f, 0.0f, 4.0f},
                {-5.0f, 3.0f, -1.0f},
                {1.0f, 2.0f, 4.0f},
                {0.0f, 5.0f, -3.0f},
                {2.0f, 1.0f, -4.0f}
            };

            // 2. query_vecs
            std::vector<std::vector<float>> query_vecs = {
                {1.0f, 2.0f, 3.0f}
            };

            // 3. test bmcv.faiss_indexflatL2
            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            sail::Bmcv bmcv(handle);

            sail::Tensor database_vecs_tensor = Tensor(handle, {db_vecs.size(), db_vecs[0].size()}, BM_FLOAT32, true, true);
            database_vecs_tensor.reset_sys_data(db_vecs.data(), {db_vecs.size(), db_vecs[0].size()});
            database_vecs_tensor.sync_s2d();

            sail::Tensor query_vecs_tensor = Tensor(handle, {query_vecs.size(), query_vecs[0].size()}, BM_FLOAT32, true, true);
            query_vecs_tensor.reset_sys_data(query_vecs.data(), {query_vecs.size(), query_vecs[0].size()});
            query_vecs_tensor.sync_s2d();

            std::tuple<sail::Tensor, sail::Tensor> result = bmcv.faiss_indexflatIP(query_vecs_tensor, database_vecs_tensor, 3, 1, 5, 3);

            return 0;
        }

faiss_indexPQ_encode
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

对输入向量进行PQ量化编码, 输出编码之后的向量。

**接口形式:**
    .. code-block:: C

        int faiss_indexPQ_encode(
            Tensor &input_vecs,
            Tensor &centroids_vecs,
            Tensor &encoded_vecs,
            int encode_vecs_num,
            int vec_dims,
            int slice_num,
            int centroids_num,
            int IP_metric
        );

**参数说明:**

* input_vecs: Tensor
    输入的待编码向量, 数据类型仅支持sail.Dtype.BM_FLOAT32, 大小应为encode_vecs_num * vec_dims。

* centroids_vecs: Tensor
    聚类中心 (质心) 向量, 数据类型仅支持sail.Dtype.BM_FLOAT32, 大小应为slice_num * centroids_num * (vec_dims / slice_num)。

* encoded_vecs: Tensor
    输出参数, 输出编码之后的向量, 数据类型为BM_UINT8, 大小为encode_vecs_num * slice_num。

* encode_vecs_num: int
    待编码向量的个数。

* vec_dims: int
    待编码向量的维度。

* slice_num: int
    原始向量维度的切分数量, 例如原始向量维度为512, slice_num = 8, 每个子向量维度为64。

* centroids_num: int
    聚类中心的数量。

* IP_metric: int
    0 表示使用L2计算距离, 1 表示使用IP计算距离。

**返回值说明:**

* result: int 
    成功返回0。

**示例代码:**
    .. code-block:: C

        #include <iostream>
        #include <vector>
        #include <sail/cvwrapper.h>

        int main()
        {   
            int encode_vecs_num = 3;
            int vec_dims = 64;
            int db_vecs_num = 10000;
            int slice_num = 8;
            int centroids_num = 256;
            int subvec_dims = vec_dims / slice_num;

            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            sail::Bmcv bmcv(handle);

            // centroids Tensor
            std::vector<int> centroids_shape = {slice_num, centroids_num, subvec_dim};
            sail::Tensor centroids_tensor(handle, centroids_shape, BM_FLOAT32, false, true);

            // input Tensor
            std::vector<int> input_shape = {encode_vecs_num, vec_dims};
            sail::Tensor input_tensor(handle, input_shape, BM_FLOAT32, false, true);

            // encoded Tensor
            std::vector<int> encoded_shape = {encode_vecs_num, slice_num};
            sail::Tensor encoded_tensor(handle, encoded_shape, BM_UINT8, true, true);

            int ret = 0;
            ret = bmcv.faiss_indexPQ_encode(input_tensor,
                                            centroids_tensor,
                                            encoded_tensor,
                                            encode_vecs_num,
                                            vec_dims,
                                            slice_num,
                                            centroids_num,
                                            0);
            
            return ret;
        }

faiss_indexPQ_ADC
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

通过查询向量和聚类中心 (质心) 向量计算出距离表, 编码的数据库向量查表计算距离并排序, 输出前 topK 个最匹配的距离值及其对应的索引。

**接口形式:**
    .. code-block:: C

        std::tuple<Tensor, Tensor> faiss_indexPQ_ADC (
            Tensor &nxquery_vecs,
            Tensor &centroids_vecs,
            Tensor &nycodes_vecs,
            int vec_dims,   
            int slice_num,  
            int centroids_num,
            int database_vecs_num,
            int query_vecs_num,
            int topK,
            int IP_metric
        );

**参数说明:**

* nxquery_vecs: Tensor
    查询向量, 数据类型仅支持sail.Dtype.BM_FLOAT32, 大小为query_vecs_nums * vec_dims。

* centroids_vecs: Tensor
    聚类中心 (质心) 向量, 数据类型仅支持sail.Dtype.BM_FLOAT32, 大小为slice_num * centroids_num * (vec_dims / slice_num)。

* nycodes_vecs: Tensor
    编码的数据库向量, 数据类型仅支持sail.Dtype.BM_UINT8, 大小为database_vecs_nums * slice_num。

* vec_dims: int
    查询向量的维度。

* slice_num: int
    原始向量维度的切分数量, 例如原始向量维度为512, slice_num = 8, 每个子向量维度为64。   
   
* centroids_num: int
    聚类中心 (质心) 向量的个数。

* database_vecs_nums: int
    数据库向量的个数。

* query_vecs_nums: int
    查询向量的个数。

* topK: int
    输出前 topK 个最匹配的距离值及其对应的索引。

* IP_metric: int
    0 表示使用L2计算距离, 1 表示使用IP计算距离。

**返回值说明:**

* result: tuple[Tensor, Tensor]
    输出前 topK 个最匹配的距离值及其对应的索引。

**示例代码:**
    .. code-block:: C

        #include <iostream>
        #include <vector>
        #include <sail/cvwrapper.h>

        int main()
        {   
            int vec_dims = 512;
            int slice_num = 8;
            int centroids_num = 256;
            int database_vecs_num = 10000;
            int query_vecs_num = 1;
            int subvec_dims = vec_dims / slice_num;
            int topK = 5;

            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            sail::Bmcv bmcv(handle);

            // nxquery Tensor
            std::vector<int> nxquery_shape = {query_vecs_num, vec_dims};
            sail::Tensor nxquery_tensor(handle, nxquery_shape, BM_FLOAT32, false, true);

            // centroids Tensor
            std::vector<int> centroids_shape = {slice_num, centroids_num, subvec_dim};
            sail::Tensor centroids_tensor(handle, centroids_shape, BM_FLOAT32, false, true);

            // nycodes Tensor
            std::vector<int> nycodes_shape = {database_vecs_nums, slice_num};
            sail::Tensor nycodes_tensor(handle, nycodes_shape, BM_UINT8, true, true);

            std::tuple<sail::Tensor, sail::Tensor> results = bmcv.faiss_indexPQ_ADC(nxquery_tensor,
                                        centroids_tensor,
                                        nycodes_tensor,
                                        vec_dims,
                                        slice_num,
                                        centroids_num,
                                        database_vecs_num,
                                        query_vecs_num,
                                        topK,
                                        0);

            return results;
        }

faiss_indexPQ_SDC
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

使用SDC (Symmetric Distance Computation, 对称距离计算)查找表加速 PQ 编码之间的距离计算, 输出与查询向量前 topK 个最匹配的距离值及其对应的索引。

**接口形式:**
    .. code-block:: C

        std::tuple<Tensor, Tensor> faiss_indexPQ_SDC (
            Tensor &nxcodes_vecs,
            Tensor &nycodes_vecs,
            Tensor &sdc_table, 
            int slice_num,  
            int centroids_num,
            int database_vecs_num,
            int query_vecs_num,
            int topK,
            int IP_metric
        );

**参数说明:**

* nxcodes_vecs: Tensor
    编码的查询向量, 数据类型仅支持sail.Dtype.BM_UINT8, 大小为query_vecs_nums * slice_num。

* nycodes_vecs: Tensor
    编码的数据库向量, 数据类型仅支持sail.Dtype.BM_UINT8, 大小为database_vecs_nums * slice_num。

 * sdc_table: Tensor
    sdc对称距离表, 数据类型仅支持sail.Dtype.BM_FLOAT32, 大小为slice_num * centroids_num * centroids_num。

* slice_num: int
    原始向量维度的切分数量, 例如原始向量维度为512, slice_num = 8, 每个子向量维度为64。   
   
* centroids_num: int
    聚类中心 (质心) 向量的个数。

* database_vecs_nums: int
    数据库向量的个数。

* query_vecs_nums: int
    查询向量的个数。

* topK: int
    输出前 topK 个最匹配的距离值及其对应的索引。

* IP_metric: int
    0 表示使用L2计算距离, 1 表示使用IP计算距离。

**返回值说明:**

* result: tuple[Tensor, Tensor]
    输出前 topK 个最匹配的距离值及其对应的索引。

**示例代码:**
    .. code-block:: C

        #include <iostream>
        #include <vector>
        #include <sail/cvwrapper.h>

        int main()
        {   
            int vec_dims = 512;
            int slice_num = 8;
            int centroids_num = 256;
            int database_vecs_num = 10000;
            int query_vecs_num = 1;
            int subvec_dims = vec_dims / slice_num;
            int topK = 5;

            int tpu_id = 0;
            sail::Handle handle(tpu_id);
            sail::Bmcv bmcv(handle);

            // nxcodes Tensor
            std::vector<int> nxcodes_shape = {query_vecs_num, slice_num};
            sail::Tensor nxcodes_tensor(handle, nxcodes_shape, BM_UINT8, false, true);

            // nycodes Tensor
            std::vector<int> nycodes_shape = {database_vecs_nums, slice_num};
            sail::Tensor nycodes_tensor(handle, nycodes_shape, BM_UINT8, false, true);

            // sdc_table Tensor
            std::vector<int> sdc_shape = {slice_num, centroids_num, centroids_num};
            sail::Tensor sdc_tensor(handle, sdc_shape, BM_FLOAT32, false, true);


            std::tuple<sail::Tensor, sail::Tensor> results = bmcv.faiss_indexPQ_SDC(nxcodes_tensor,
                                                                                    nycodes_tensor,
                                                                                    sdc_tensor,
                                                                                    slice_num,
                                                                                    centroids_num,
                                                                                    database_vecs_num,
                                                                                    query_vecs_num,
                                                                                    topK,
                                                                                    0);

            return results;
        }

