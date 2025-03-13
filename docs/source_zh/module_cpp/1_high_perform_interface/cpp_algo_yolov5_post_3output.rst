algo_yolov5_post_3output
_________________________________

针对以三输出YOLOv5模型的后处理接口，内部使用线程池的方式实现。

构造函数
>>>>>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: c
          
        algo_yolov5_post_3output(const std::vector<std::vector<int>>& shape, 
                                int network_w=640, 
                                int network_h=640, 
                                int max_queue_size=20,
                                bool input_use_multiclass_nms=true,
                                bool agnostic=false);

**参数说明:**

* shape: std::vector<std::vector<int>>

输入参数。输入数据的shape。

* network_w: int

输入参数。模型的输入宽度，默认为640。

* network_h: int

输入参数。模型的输入宽度，默认为640。

* max_queue_size: int

输入参数。缓存数据的最大长度。

* input_use_multiclass_nms: bool

输入参数。使用多分类NMS,每个框具有多个类别。

* agnostic: bool

输入参数。使用不考虑类别的NMS算法。


push_data
>>>>>>>>>>>>>

输入数据，支持任意batchsize的输入。

**接口形式1:**
    .. code-block:: c

        int push_data(
            std::vector<int> channel_idx, 
            std::vector<int> image_idx, 
            std::vector<TensorPTRWithName> input_data, 
            std::vector<float> dete_threshold,
            std::vector<float> nms_threshold,
            std::vector<int> ost_w,
            std::vector<int> ost_h,
            std::vector<std::vector<int>> padding_attr);

**参数说明1:**

* channel_idx: std::vector<int>

输入参数。输入图像序列的通道号。

* image_idx: std::vector<int>

输入参数。输入图像序列的编号。

* input_data: std::vector<TensorPTRWithName>

输入参数。输入数据，包含三个输出。

* dete_threshold: std::vector<float>

输入参数。检测阈值序列。

* nms_threshold: std::vector<float>

输入参数。nms阈值序列。

* ost_w: std::vector<int>

输入参数。原始图片序列的宽。

* ost_h: std::vector<int>

输入参数。 原始图片序列的高。

* padding_attrs: std::vector<std::vector<int>>

输入参数。填充图像序列的属性列表，填充的起始点坐标x、起始点坐标y、尺度变换之后的宽度、尺度变换之后的高度。

**返回值说明:**

成功返回0，其他值表示失败。

**接口形式2:**
    .. code-block:: c

        int push_data(
            std::vector<int> channel_idx, 
            std::vector<int> image_idx, 
            std::vector<TensorPTRWithName> input_data, 
            std::vector<std::vector<float>> dete_threshold,
            std::vector<float> nms_threshold,
            std::vector<int> ost_w,
            std::vector<int> ost_h,
            std::vector<std::vector<int>> padding_attr);

**参数说明2:**

* channel_idx: std::vector<int>

输入参数。输入图像序列的通道号。

* image_idx: std::vector<int>

输入参数。输入图像序列的编号。

* input_data: std::vector<TensorPTRWithName>

输入参数。输入数据，包含三个输出。

* dete_threshold: std::vector<std::vector<float>>

输入参数。检测阈值序列。

* nms_threshold: std::vector<float>

输入参数。nms阈值序列。

* ost_w: std::vector<int>

输入参数。原始图片序列的宽。

* ost_h: std::vector<int>

输入参数。 原始图片序列的高。

* padding_attrs: std::vector<std::vector<int>>

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

reset_anchors
>>>>>>>>>>>>>

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
        #include <vector>  
        #include <queue>  
        #include <numeric>  
        #include <iostream>  
        
        int main() {  
            int tpu_id = 0;  
            sail::Handle handle(tpu_id);  
            std::string image_path = "../../../sophon-demo/sample/YOLOv5/datasets/test/3.jpg";  
            std::string bmodel_path = "../../../sophon-demo/sample/YOLOv5/models/BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel";  
        
            sail::Decoder decoder(image_path, true, tpu_id);  
            sail::BMImage bmimg = decoder.read(handle);  
        
            sail::EngineImagePreProcess engine_image_pre_process(bmodel_path, tpu_id, 0);  
            engine_image_pre_process.InitImagePreProcess(sail::sail_resize_type::BM_PADDING_TPU_LINEAR, true, 10, 10);  
            engine_image_pre_process.SetPaddingAtrr(114, 114, 114, 1);  
            std::vector<std::pair<float, float>> alpha_beta = {{1.0/255, 0}, {1.0/255, 0}, {1.0/255, 0}};  
            engine_image_pre_process.SetConvertAtrr(alpha_beta);  
            bool ret = engine_image_pre_process.PushImage(0, 0, bmimg);  
        
            std::vector<sail::Tensor> output_tensor_map;  
            std::vector<sail::Image> ost_images;  
            std::vector<int> channels;  
            std::vector<int> imageidxs;  
            std::vector<float> padding_atrr;  
            engine_image_pre_process.GetBatchData(true, output_tensor_map, ost_images, channels, imageidxs, padding_atrr);  
        
            std::vector<int> width_list;  
            std::vector<int> height_list;  
            for (int index = 0; index < channels.size(); index++) {  
                width_list.push_back(ost_images[index].width());  
                height_list.push_back(ost_images[index].height());  
            }  
        
            sail::algo_yolov5_post_3output yolov5_post([[1, 3, 20, 20, 85], [1, 3, 40, 40, 85], [1, 3, 80, 80, 85]], 640, 640, 10);  
            std::vector<float> dete_thresholds(channels.size(), 0.2);  
            std::vector<float> nms_thresholds(channels.size(), 0.5);
            yolov5_post.push_data(channels, imageidxs, output_tensor_map, dete_thresholds, nms_thresholds, width_list, height_list, padding_atrr);  
            std::vector<std::tuple<int, int, int, int ,int, float>> objs;  
            std::vector<int> channel;  
            std::vector<int> image_idx;  
            yolov5_post.get_result_npy(&objs, &channel, &image_idx);  
            std::cout << "objs: " << objs << ", channel: " << channel << ", image idx: " << image_idx << std::endl;  
        
            return 0;  
        }