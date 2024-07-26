tpu_kernel_api_yolov5_out_without_decode
____________________________________________

针对1输出的yolov5模型，使用智能视觉深度学习处理器Kernel对后处理进行加速，目前只支持BM1684x，且libsophon的版本必须不低于0.4.6（v23.03.01）。

构造函数
>>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: c
          
        tpu_kernel_api_yolov5_out_without_decode(int device_id, 
                                            const std::vector<int>& shapes, 
                                            int network_w=640, 
                                            int network_h=640,
                                            std::string module_file = "/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so");

**参数说明:**

* device_id: int

输入参数。使用的设备编号。

* shape: std::vector<int>

输入参数。输入数据的shape。

* network_w: int

输入参数。模型的输入宽度，默认为640。

* network_h: int

输入参数。模型的输入宽度，默认为640。

* module_file: string

输入参数。Kernel module文件路径，默认为"/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so"。


process
>>>>>>>>>>>>>

处理接口。

**接口形式1:**
    .. code-block:: c

        std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> process(
                                    TensorPTRWithName& input, 
                                    float dete_threshold, 
                                    float nms_threshold);

**参数说明1:**

* input_data: TensorPTRWithName

输入参数。输入数据，包含一个输出。

* dete_threshold: float

输入参数。检测阈值。

* nms_threshold: float

输入参数。nms阈值序列。

**接口形式2:**
    .. code-block:: c

        std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> process(
                                        std::map<std::string, Tensor&>& input, 
                                        float dete_threshold, 
                                        float nms_threshold);

**参数说明2:**

* input_data: std::map<std::string, Tensor&>

输入参数。输入数据，包含一个输出。

* dete_threshold: float

输入参数。检测阈值。

* nms_threshold: float

输入参数。nms阈值序列。

**返回值说明:**

std::vector<std::vector<std::tuple<left, top, right, bottom, class_id, score> > >

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

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/tpu_kernel_api.h>
        #include <opencv2/opencv.hpp>  
        #include <fstream>  
        #include <iostream>  
        #include <vector>  
        #include <string>  
        #include <math.h>  
        
        using namespace std;       
        
        void get_ratio(sail::BMImage& bmimg, int& tw, int& th, int& tx1, int& tx2, int& ty1, int& ty2) {  
            int img_w = bmimg.width();  
            int img_h = bmimg.height();  
            double r_w = 640.0 / img_w;  
            double r_h = 640.0 / img_h;  
            if (r_h > r_w) {  
                tw = 640;  
                th = static_cast<int>(r_w * img_h);  
                tx1 = tx2 = 0;  
                ty1 = static_cast<int>((640 - th) / 2);  
                ty2 = 640 - th - ty1;  
            } else {  
                tw = static_cast<int>(r_h * img_w);  
                th = 640;  
                tx1 = static_cast<int>((640 - tw) / 2);  
                tx2 = 640 - tw - tx1;  
                ty1 = ty2 = 0;  
            }  
        }
        
        int main() {  
            int tpu_id = 0;  
            std::string image_path = "../../../sophon-demo/sample/YOLOv5/datasets/test/3.jpg";  
            sail::Decoder decoder(image_path, true, tpu_id);  
            std::string bmodel_path = "../../../sophon-demo/sample/YOLOv5/models/BM1684X/yolov5s_v6.1_1output_int8_1b.bmodel";  
            sail::Handle handle(tpu_id);  
            std::vector<std::pair<float, float>> alpha_beta = {{1.0/255, 0}, {1.0/255, 0}, {1.0/255, 0}};  
        
            sail::sail_resize_type resize_type = sail::sail_resize_type::BM_PADDING_TPU_LINEAR;  
            sail::EngineImagePreProcess sail_engineipp(bmodel_path, tpu_id, false);  
            sail_engineipp.InitImagePreProcess(resize_type, true, 10, 10);  
            sail_engineipp.SetPaddingAtrr(114, 114, 114, 1);  
            bool ret1 = sail_engineipp.SetConvertAtrr(alpha_beta);  
        
            sail::BMImage bm_i;  
            decoder.read(handle, bm_i);  
            decoder.release();  
            int hw, ratio, txy;  
            get_ratio(bm_i, hw, ratio, txy);  
            bool ret3 = sail_engineipp.PushImage(0, 0, bm_i);  

            std::map<std::string,sail::Tensor*> output_tensor_map ;  
            std::vector<BMImage> ost_images ;  
            std::vector<int> channel_list ;  
            std::vector<int> imageidx_list ;  
            std::vector<std::vector<int>> padding_atrr ; 
            std::tuple<output_tensor_map, ost_images, channel_list ,imageidx_list, padding_atrr>all_out = sail_engineipp.GetBatchData(true);  
        
            sail::tpu_kernel_api_yolov5_out_without_decode tpu_kernel_1o(0, std::vector<int>{1, 25200, 85}, 640, 640, "/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so");  
            std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> out_boxs = tpu_kernel_1o->process(output_tensor_map, 0.5, 0.5);
            std::vector<std::vector<float, 6>> result;
            for (int bid = 0; bid < out_boxs[0].size(); bid++) { 
                std::vector<float, 6> temp_bbox;  
                temp_bbox[0] = out_boxs[0][bid].class_id;
                if (temp_bbox[0] == -1)continue;
                temp_bbox[1] = out_boxs[0][bid].score;
                temp_bbox[2] = (out_boxs[0][bid].width+ 0.5) / ratio;
                temp_bbox[3] =(out_boxs[0][bid].height+ 0.5) / ratio;
                float centerX = ((out_boxs[0][bid].left + out_boxs[0][bid].right) / 2 + 1 - tx1) / ratio - 1;
                float centerY = ((out_boxs[0][bid].top + out_boxs[0][bid].bottom) / 2 + 1 - ty1) / ratio - 1;
                temp_bbox[4] = MAX(int(centerX - temp_bbox.width / 2), 0);
                temp_bbox[5] = MAX(int(centerY - temp_bbox.height / 2), 0);
                result.push_back(temp_bbox);  
                }  
            }  