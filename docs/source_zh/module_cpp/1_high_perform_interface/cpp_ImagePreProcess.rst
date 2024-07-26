ImagePreProcess
______________________

通用预处理接口，内部使用线程池的方式实现。

构造函数
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: c

        ImagePreProcess(
            int batch_size,
            sail_resize_type resize_mode,
            int tpu_id=0, 
            int queue_in_size=20, 
            int queue_out_size=20,
            bool use_mat_flag=false);


**参数说明:**

* batch_size: int

输入参数。输出结果的batch size。

* resize_mode: sail_resize_type

输入参数。内部尺度变换的方法。

* tpu_id: int

输入参数。使用的智能视觉深度学习处理器 id，默认为0。

* queue_in_size: int

输入参数。输入图像队列缓存的最大长度，默认为20。

* queue_out_size: int

输入参数。输出Tensor队列缓存的最大长度，默认为20。

* use_mat_output: bool

输入参数。是否使用OpenCV的Mat作为图片的输出，默认为False，不使用。

SetResizeImageAtrr
>>>>>>>>>>>>>>>>>>>>

设置图像尺度变换的属性。

**接口形式:**
    .. code-block:: c

        void SetResizeImageAtrr(			    
            int output_width,				    
            int output_height,				    
            bool bgr2rgb,					    
            bm_image_data_format_ext  dtype);	

**参数说明:**
            
* output_width: int

输入参数。尺度变换之后的图像宽度。

* output_height: int

输入参数。尺度变换之后的图像高度。

* bgr2rgb: bool

输入参数。是否将图像有BGR转换为GRB。

* dtype: ImgDtype  

输入参数。图像尺度变换之后的数据类型，当前版本只支持BM_FLOAT32,BM_INT8,BM_UINT8。可根据模型的输入数据类型设置。


SetPaddingAtrr
>>>>>>>>>>>>>>>>>>>>

设置Padding的属性，只有在resize_mode为 BM_PADDING_VPP_NEAREST、BM_PADDING_TPU_NEAREST、BM_PADDING_TPU_LINEAR、BM_PADDING_TPU_BICUBIC时生效。

**接口形式:**
    .. code-block:: c

        void SetPaddingAtrr(		    
            int padding_b=114,		        
            int padding_g=114,		        
            int padding_r=114,		        
            int align=0);	

**参数说明:**
* padding_b: int

输入参数。要pdding的b通道像素值，默认为114。

* padding_g: int

输入参数。要pdding的g通道像素值，默认为114。
                
* padding_r: int

输入参数。要pdding的r通道像素值，默认为114。

* align: int

输入参数。图像填充为位置，0表示从左上角开始填充，1表示居中填充，默认为0。


SetConvertAtrr
>>>>>>>>>>>>>>>>>>>>

设置线性变换的属性。

**接口形式:**
    .. code-block:: c

         int SetConvertAtrr(
            const std::tuple<
                std::pair<float, float>,
                std::pair<float, float>,
                std::pair<float, float>> &alpha_beta);

**参数说明:**

* alpha_beta: (a0, b0), (a1, b1), (a2, b2)。输入参数。

    a0 描述了第 0 个 channel 进行线性变换的系数；

    b0 描述了第 0 个 channel 进行线性变换的偏移；

    a1 描述了第 1 个 channel 进行线性变换的系数；

    b1 描述了第 1 个 channel 进行线性变换的偏移；

    a2 描述了第 2 个 channel 进行线性变换的系数；

    b2 描述了第 2 个 channel 进行线性变换的偏移；

**返回值说明:**

设置成功返回0，其他值时设置失败。


PushImage
>>>>>>>>>>>>>>>

送入数据。

**接口形式:**
    .. code-block:: c

        int PushImage(
            int channel_idx, 
            int image_idx, 
            BMImage &image);

**参数说明:**

* channel_idx: int

输入参数。输入图像的通道号。
                
* image_idx: int

输入参数。输入图像的编号。

* image: BMImage

输入参数。输入图像。

**返回值说明:**

设置成功返回0，其他值时表示失败。
            
GetBatchData
>>>>>>>>>>>>>>>

获取处理的结果。

**接口形式:**
    .. code-block:: c
        
        std::tuple<sail::Tensor, 
            std::vector<BMImage>,
            std::vector<int>,
            std::vector<int>,
            std::vector<std::vector<int>>> GetBatchData();
        
**返回值说明:**
tuple[data, images, channels, image_idxs, padding_attrs]

* data: Tensor

    处理后的结果Tensor。

* images: std::vector<BMImage>

    原始图像序列。

* channels: std::vector<int>

    原始图像的通道序列。

* image_idxs: std::vector<int>

    原始图像的编号序列。

* padding_attrs: std::vector<std::vector<int> >

    填充图像的属性列表，填充的起始点坐标x、起始点坐标y、尺度变换之后的宽度、尺度变换之后的高度

set_print_flag
>>>>>>>>>>>>>>>

设置打印日志的标志位，不调用此接口时不打印日志。

**接口形式:**
    .. code-block:: c

        void set_print_flag(bool print_flag);
        
**返回值说明:**

* flag: bool

打印的标志位，False时表示不打印，True时表示打印。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h> 
        #include <opencv2/opencv.hpp>  
        #include <vector>  
        
        int main() {  
            int tpu_id = 0;  
            int batch_size = 1;  
            std::string image_path = "./data/zidane.jpg";  
            sail::Handle handle(tpu_id);  
        
            std::vector<std::pair<int, int>> alpha_beta = {{1, 0}, {1, 0}, {1, 0}};  
            sail::Decoder decoder(image_path, false, tpu_id);  
        
            sail::ImagePreProcess sail_ipp(batch_size, sail::sail_resize_type::BM_RESIZE_VPP_NEAREST, tpu_id, 20, 20, false);  
            sail_ipp.SetResizeImageAtrr(640, 640, false, sail::ImgDtype::DATA_TYPE_EXT_1N_BYTE);  
            sail_ipp.SetConvertAtrr(alpha_beta);  
            // sail_ipp.set_print_flag(true);  
            sail::BMImage bm_i;  
            for (int i = 0; i < batch_size; i++) {  
                decoder.read(handle, bm_i);  
                sail_ipp.PushImage(0, i, bm_i);  
            }  
            auto result = sail_ipp.GetBatchData();  
            decoder.release();  
        
            auto tensor = result[0];  
            auto t_npy = tensor.asnumpy();  
            auto result_img = t_npy[0].transpose({1, 2, 0});  
        
            cv::Mat raw_img = cv::imread(image_path);  
            cv::Mat resize_img = cv::resize(raw_img, cv::Size(640, 640), cv::INTER_NEAREST);  
            double max_diff = abs((resize_img.astype(double) - result_img.astype(double)).max());  
            double min_diff = abs((resize_img.astype(double) - result_img.astype(double)).min());  
            double diff = std::max(max_diff, min_diff);  
            std::cout << max_diff << " " << min_diff << " " << diff << std::endl;  
            return 0;  
        }
