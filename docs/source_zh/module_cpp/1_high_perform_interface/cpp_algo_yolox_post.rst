algo_yolox_post
_________________________________

针对YOLOX模型的后处理接口，内部使用线程池的方式实现。

构造函数
>>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: c
          
        algo_yolox_post(const std::vector<int>& shape, 
                                int network_w=640, 
                                int network_h=640, 
                                int max_queue_size=20);

**参数说明:**

* shape: std::vector<int>

输入参数。输入数据的shape。

* network_w: int

输入参数。模型的输入宽度，默认为640。

* network_h: int

输入参数。模型的输入宽度，默认为640。

* max_queue_size: int

输入参数。缓存数据的最大长度。




push_data
>>>>>>>>>>>>>

输入数据，支持batchsize不为1的输入。

**接口形式:**
    .. code-block:: c

        int push_data(
            std::vector<int> channel_idx, 
            std::vector<int> image_idx, 
            TensorPTRWithName input_data, 
            std::vector<float> dete_threshold,
            std::vector<float> nms_threshold,
            std::vector<int> ost_w,
            std::vector<int> ost_h,
            std::vector<std::vector<int>> padding_attr);

**参数说明:**

* channel_idx: std::vector<int>

输入参数。输入图像序列的通道号。

* image_idx: std::vector<int>

输入参数。输入图像序列的编号。

* input_data: TensorPTRWithName

输入参数。输入数据。

* dete_threshold: std::vector<float>

输入参数。检测阈值序列。

* nms_threshold: std::vector<float>

输入参数。nms阈值序列。

* ost_w: std::vector<int>

输入参数。原始图片序列的宽。

* ost_h: std::vector<int>

输入参数。 原始图片序列的高。

* padding_attrs: std::vector<std::vector<int> >

输入参数。填充图像序列的属性列表，填充的起始点坐标x、起始点坐标y、尺度变换之后的宽度、尺度变换之后的高度。

**返回值说明:**

成功返回0，其他值表示失败。

get_result_npy
>>>>>>>>>>>>>>>>>

获取最终的检测结果

**接口形式:**
    .. code-block:: c

        std::tuple<std::vector<std::tuple<int, int, int, int ,int, float>>,int,int> get_result_npy();

**返回值说明:**
tuple[tuple[left, top, right, bottom, class_id, score],channel_idx, image_idx]

* left: int 

检测结果最左x坐标。

* top: int

检测结果最上y坐标。

* right: int

检测结果最右x坐标。

* bottom: int

检测结果最下y坐标。

* class_id: int

检测结果的类别编号。

* score: float

检测结果的分数。

* channel_idx: int

原始图像的通道号。

* image_idx: int

原始图像的编号。

**示例代码:**
    .. code-block:: c

        #include <stdio.h>
        #include <sail/cvwrapper.h>
        #include <sail/tensor.h>
        #include <sail/algokit.h>
        #include <iostream>
        #include <string>
        #include <vector>   
        #include <cmath>  
        
        int main() {  
            int tpu_id = 0;  
            sail::Handle handle(tpu_id);  
            std::string image_name = "../../../sophon-demo/sample/YOLOv5/datasets/test/3.jpg";  
            std::string bmodel_name = "../../../sophon-demo/sample/YOLOX/models/BM1684X/yolox_int8_1b.bmodel";  
        
            sail::Decoder decoder(image_name, true, tpu_id);  
            sail::BMImage BMimg = decoder.read(handle);  
        
            sail::EngineImagePreProcess engine_image_pre_process(bmodel_name, tpu_id, 0);  
            engine_image_pre_process.InitImagePreProcess(sail.sail_resize_type.BM_PADDING_TPU_LINEAR, true, 10, 10);  
            engine_image_pre_process.SetPaddingAtrr(114, 114, 114, 1);  
            std::vector<std::pair<float, float>> alpha_beta = {{1.0/255, 0}, {1.0/255, 0}, {1.0/255, 0}};  
            engine_image_pre_process.SetConvertAtrr(alpha_beta);  
            bool ret = engine_image_pre_process.PushImage(0, 0, BMimg);  
        
            std::map<std::string,sail::Tensor*> output_tensor_map;  
            std::vector<sail::BMImage> ost_images;  
            int channels = 0;  
            std::vector<int> imageidxs;  
            std::vector<int> padding_atrr;  
            engine_image_pre_process.GetBatchData(output_tensor_map, ost_images, channels, imageidxs, padding_atrr);  
        
            std::vector<int> width_list;  
            std::vector<int> height_list;  
            for (int index = 0; index < channels; index++) {  
                width_list.push_back(ost_images[index].width());  
                height_list.push_back(ost_images[index].height());  
            }  
        
            sail::algo_yolox_post yolox_post(std::vector<std::vector<int>>{{1, 3, 20, 20, 85}, {1, 3, 40, 40, 85}, {1, 3, 80, 80, 85}}, 640, 640, 10);  
            std::vector<float> dete_thresholds = {0.2f, 0.2f, 0.2f};  
            std::vector<float> nms_thresholds = {0.5f, 0.5f, 0.5f};  
            ret = yolox_post.push_data(channels, imageidxs, output_tensor_map, dete_thresholds, nms_thresholds, width_list, height_list, padding_atrr);  
        
            std::vector<std::tuple<int, int, int, int, int, float>> detection_results;
            int channel_idx, image_idx;

            std::tie(detection_results, channel_idx, image_idx) = yolox_post.get_result_npy();

            std::cout << "Detection Results:" << std::endl;
            for (const auto& detection : detection_results) {
                int left, top, right, bottom, class_id;
                float score;

                std::tie(left, top, right, bottom, class_id, score) = detection;

                std::cout << "Box: (" << left << ", " << top << ", " << right << ", " << bottom << ")"
                          << " Class ID: " << class_id << " Score: " << score << std::endl;
            }

            std::cout << "Channel Index: " << channel_idx << " Image Index: " << image_idx << std::endl;
            return 0;  
        }