EngineImagePreProcess
___________________________

带有预处理功能的图像推理接口，内部使用线程池的方式，Python下面有更高的效率。

构造函数
>>>>>>>>>>>>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: c

        EngineImagePreProcess(const std::string& bmodel_path, 
                            int tpu_id, 
                            bool use_mat_output=false,
                            std::vector<int> core_list = {});

**参数说明:**

* bmodel_path: string 

输入参数。输入模型的路径。

* tpu_id: int

输入参数。使用的智能视觉深度学习处理器 id。

* use_mat_output: bool

输入参数。是否使用OpenCV的Mat作为图片的输出，默认为False，不使用。

* use_mat_output: bool

输入参数。
使用支持多核推理的处理器和bmodel时，可以选择推理时使用的多个核心，默认使用从core0开始的N个core来做推理，N由当前bmodel决定。
对于仅支持单核推理的处理器和bmodel模型，仅支持选择推理使用的单个核心，参数的输入列表长度必须为1，若传入列表长度大于1，将自动在0号核心上推理。
默认为空不指定时，将默认从0号核心开始的N个core来做推理。

InitImagePreProcess
>>>>>>>>>>>>>>>>>>>>>>>>>>

初始化图像预处理模块。

**接口形式:**
    .. code-block:: c

        int InitImagePreProcess(
            sail_resize_type resize_mode,
            bool bgr2rgb=false,					    
            int queue_in_size=20, 
            int queue_out_size=20);

                    
**参数说明:**

* resize_mode: sail_resize_type

输入参数。内部尺度变换的方法。

* bgr2rgb: bool

输入参数。是否将图像有BGR转换为GRB。

* queue_in_size: int

输入参数。输入图像队列缓存的最大长度，默认为20。queue_in_size必须大于模型batch_size，若小于模型batch_size，将自动调整为模型batch_size。

* queue_out_size: int

输入参数。预处理结果Tensor队列缓存的最大长度，默认为20。queue_out_size必须大于模型batch_size，若小于模型batch_size，将自动调整为模型batch_size。

**返回值说明:**

成功返回0，其他值时失败。
           

SetPaddingAtrr
>>>>>>>>>>>>>>>>>>>

设置Padding的属性，只有在resize_mode为 BM_PADDING_VPP_NEAREST、BM_PADDING_TPU_NEAREST、BM_PADDING_TPU_LINEAR、BM_PADDING_TPU_BICUBIC时生效。

**接口形式:**
    .. code-block:: c

        int SetPaddingAtrr(
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
          
**返回值说明:**

成功返回0，其他值时失败。


SetConvertAtrr
>>>>>>>>>>>>>>>>>>>

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
>>>>>>>>>>>>>>

送入图像数据

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
                
输入参数。输入的图像。

**返回值说明:**

成功返回0，其他值时失败。


GetBatchData
>>>>>>>>>>>>>>>>>>>

获取一个batch的推理结果，调用此接口时，由于返回的结果类型为BMImage，所以use_mat_output必须为False。值得注意的是，该接口输出的tensor需要手动进行释放。

**接口形式:**
    .. code-block:: c

        std::tuple<std::map<std::string,sail::Tensor*>, 
            std::vector<BMImage>,
            std::vector<int>,
            std::vector<int>,
            std::vector<std::vector<int>>> GetBatchData();

**返回值说明:**

tuple[output_array, ost_images, channels, image_idxs, padding_attrs]

* output_array: std::map<std::string,sail::Tensor*>

推理结果。

* ost_images: std::vector<BMImage>

原始图片序列。

* channels: std::vector<int>

结果对应的原始图片的通道序列。

* image_idxs: std::vector<int>

结果对应的原始图片的编号序列。

* padding_attrs: std::vector<std::vector<int> >

填充图像的属性列表，填充的起始点坐标x、起始点坐标y、尺度变换之后的宽度、尺度变换之后的高度。



GetBatchData_CV
>>>>>>>>>>>>>>>>>>>>>>>

获取一个batch的推理结果，调用此接口时，由于返回的结果类型为cv::Mat，所以use_mat_output必须为True。值得注意的是，该接口输出的tensor需要手动进行释放。

**接口形式:**
    .. code-block:: c

        std::tuple<std::map<std::string,sail::Tensor*>, 
            std::vector<cv::Mat>,
            std::vector<int>,
            std::vector<int>,
            std::vector<std::vector<int>>> GetBatchData_CV();

**返回值说明:**

tuple[output_array, ost_images, channels, image_idxs, padding_attrs]

* output_array: std::map<std::string,sail::Tensor*>

推理结果。

* ost_images: std::vector<cv::Mat>

原始图片序列。

* channels: std::vector<int>

结果对应的原始图片的通道序列。

* image_idxs: std::vector<int>

结果对应的原始图片的编号序列。

* padding_attrs: std::vector<std::vector<int> >

填充图像的属性列表，填充的起始点坐标x、起始点坐标y、尺度变换之后的宽度、尺度变换之后的高度。


get_graph_name
>>>>>>>>>>>>>>>>

获取模型的运算图名称。

**接口形式:**
    .. code-block:: c

        std::string get_graph_name();

**返回值说明:**

返回模型的第一个运算图名称。

            
get_input_width
>>>>>>>>>>>>>>>>

获取模型输入的宽度。

**接口形式:**
    .. code-block:: c

        int get_input_width();

**返回值说明:**

返回模型输入的宽度。

            
get_input_height
>>>>>>>>>>>>>>>>>>>

获取模型输入的高度。

**接口形式:**
    .. code-block:: c

        int get_input_height();

**返回值说明:**

返回模型输入的宽度。

            
get_output_names
>>>>>>>>>>>>>>>>>>>

获取模型输出Tensor的名称。

**接口形式:**
    .. code-block:: c

        std::vector<std::string> get_output_names();

**返回值说明:**

返回模型所有输出Tensor的名称。
   
            
get_output_shape
>>>>>>>>>>>>>>>>>>>

获取指定输出Tensor的shape

**接口形式:**
    .. code-block:: c
        
        std::vector<int> get_output_shape(const std::string& tensor_name);

**参数说明:**

* tensor_name: string

指定的输出Tensor的名称。

**返回值说明:**

返回指定输出Tensor的shape。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <opencv2/opencv.hpp>  
        #include <fstream>  
        #include <iostream>  
        #include <vector>  
        #include <string>  
        
        using namespace std;  
  
        int main() {  
            int dev_id = 0;  
            sail::Handle handle(dev_id);  
            std::string image_path = "./data/zidane.jpg";  
            sail::Decoder decoder(image_path, true, dev_id);  
            std::string bmodel_path = "../../../sophon-demo/sample/YOLOv5/models/BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel";  
            std::vector<std::pair<float, float>> alpha_beta = {{1.0/255.0, 0}, {1.0/255.0, 0}, {1.0/255.0, 0}};  
        
            sail::sail_resize_type resize_type = sail::sail_resize_type::BM_PADDING_TPU_LINEAR;  
            sail::EngineImagePreProcess sail_engineipp(bmodel_path, dev_id, false);  
            sail_engineipp.InitImagePreProcess(resize_type, false, 20, 20);  
            sail_engineipp.SetPaddingAtrr();  
            sail_engineipp.SetConvertAtrr(alpha_beta);  
        
            int get_i_w = sail_engineipp.get_input_width();  
            int get_i_h = sail_engineipp.get_input_height();  
            std::string output_name = sail_engineipp.get_output_names()[0];  
            std::vector<int> output_shape = sail_engineipp.get_output_shape(output_name);  
        
            sail::BMImage bm_i;  
            decoder.read(handle, bm_i);  
            sail_engineipp.PushImage(0, 0, bm_i);  
            std::tuple<std::map<std::string,sail::Tensor*>,std::vector<BMImage>,std::vector<int>,std::vector<int>,std::vector<std::vector<int>>> res = sail_engineipp.GetBatchData(true);  
        
            std::cout << output_name << " " << output_shape << " " << get_i_h << " " << get_i_w << " " << res << std::endl;  
            return 0;  
        }