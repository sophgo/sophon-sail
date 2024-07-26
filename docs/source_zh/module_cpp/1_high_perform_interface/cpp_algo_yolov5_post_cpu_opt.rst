algo_yolov5_post_cpu_opt
____________________________________________

针对3输出或1输出的yolov5模型，对后处理进行了加速。

构造函数
>>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: c
          
        algo_yolov5_post_cpu_opt(const std::vector<std::vector<int>>& shape, 
                                    int network_w=640, int network_h=640);

**参数说明:**

* shape: std::vector<std::vector<int> >

输入参数。输入数据的shape。

* network_w: int

输入参数。模型的输入宽度，默认为640。

* network_h: int

输入参数。模型的输入宽度，默认为640。


process
>>>>>>>>>>>>>

处理接口。

**接口形式1:**
    .. code-block:: c

        std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> process(std::vector<TensorPTRWithName> &input_data, 
                std::vector<int> &ost_w,
                std::vector<int> &ost_h,
                std::vector<float> &dete_threshold,
                std::vector<float> &nms_threshold,
                bool input_keep_aspect_ratio,
                bool input_use_multiclass_nms);

**参数说明1:**

* input_data: std::vector<TensorPTRWithName>

输入参数。输入数据，包含三个输出或一个输出。

* ost_w: std::vector<int>

输入参数。原始图片的宽度。

* ost_h: std::vector<int>

输入参数。原始图片的高度。

* dete_threshold: std::vector<float>

输入参数。检测阈值。

* nms_threshold: std::vector<float>

输入参数。nms阈值序列。

* input_keep_aspect_ratio: bool

输入参数。输入图片是否保持纵横比。

* input_use_multiclass_nms: bool

输入参数。是否用多类nms。

**接口形式2:**
    .. code-block:: c

        std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> process(std::map<std::string, Tensor&>& input_data,
                std::vector<int> &ost_w,
                std::vector<int> &ost_h,
                std::vector<float> &dete_threshold,
                std::vector<float> &nms_threshold,
                bool input_keep_aspect_ratio,
                bool input_use_multiclass_nms);

**参数说明2:**

* input_data: std::map<std::string, Tensor&>

输入参数。输入数据，包含三个输出或一个输出。

* ost_w: std::vector<int>

输入参数。原始图片的宽度。

* ost_h: std::vector<int>

输入参数。原始图片的高度。

* dete_threshold: std::vector<float>

输入参数。检测阈值。

* nms_threshold: std::vector<float>

输入参数。nms阈值序列。

* input_keep_aspect_ratio: bool

输入参数。输入图片是否保持纵横比。

* input_use_multiclass_nms: bool

输入参数。是否用多类nms。

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


reset_anchors
>>>>>>>>>>>>>>>>

更新anchor尺寸.

**接口形式:**
    .. code-block:: c

        int reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new);

**参数说明:**

* anchors_new: std::vector<std::vector<std::vector<int> > >

要更新的anchor尺寸列表.

**返回值说明:**

成功返回0，其他值表示失败。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/tensor.h>
        #include <sail/algokit.h>
        #include <iostream>  
        #include <vector>  
        #include <queue>  
        #include <numeric>  
        #include <opencv2/opencv.hpp>  
  
        int main() {  
            int tpu_id = 0;  
            sail::Handle handle(tpu_id);  
            std::string image_name = "../../../sophon-demo/sample/YOLOv5/datasets/test/3.jpg";  
            std::string bmodel_name = "../../../sophon-demo/sample/YOLOv5/models/BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel";  
            sail::Decoder decoder(image_name, true, tpu_id);  
            auto bmimg = decoder.read(handle);  
            sail::EngineImagePreProcess engine_image_pre_process(bmodel_name, tpu_id, 0);  
            engine_image_pre_process.InitImagePreProcess(sail::sail_resize_type::BM_PADDING_TPU_LINEAR, true, 10, 10);  
            engine_image_pre_process.SetPaddingAtrr(114, 114, 114, 1);  
            std::vector<std::pair<float, float>> alpha_beta = {{1.0/255, 0}, {1.0/255, 0}, {1.0/255, 0}};  
            engine_image_pre_process.SetConvertAtrr(alpha_beta);  
            auto ret = engine_image_pre_process.PushImage(0, 0, bmimg);  
            auto output_tensor_map = engine_image_pre_process.GetBatchData(true);  
            std::vector<int> width_list;  
            std::vector<int> height_list;  
            for (int index = 0; index < output_tensor_map.size(); index++) {  
                width_list.push_back(output_tensor_map[index].width());  
                height_list.push_back(output_tensor_map[index].height());  
            }  
            auto yolov5_post = sail::algo_yolov5_post_cpu_opt(std::vector<std::vector<int>>{{1, 3, 20, 20, 85}, {1, 3, 40, 40, 85}, {1, 3, 80, 80, 85}}, 640, 640);  
            std::vector<float> dete_thresholds(output_tensor_map.size(), 0.2f);  
            std::vector<float> nms_thresholds(output_tensor_map.size(), 0.5f);  
            auto objs = yolov5_post.process(output_tensor_map, width_list, height_list, dete_thresholds, nms_thresholds, true, true);  
            std::cout << objs << std::endl;  
            return 0;  
        }