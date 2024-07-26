tpu_kernel_api_openpose_part_nms
____________________________________________

针对OpenPose模型，使用智能视觉深度学习处理器Kernel对part nms后处理进行加速，目前只支持BM1684x，且libsophon的版本必须不低于0.4.6（v23.03.01）。

构造函数
>>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: c
          
        tpu_kernel_api_openpose_part_nms(int device_id, 
                                            int network_c,
                                            std::string module_file = "/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so");

**参数说明:**

* device_id: int

输入参数。使用的设备编号。

* network_c: int

输入参数。输入通道数，对应关键点通道的数量。

* module_file: string

输入参数。Kernel module文件路径，默认为"/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so"。


process
>>>>>>>>>>>>>

处理接口。

**接口形式1:**
    .. code-block:: c

        std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<float>>, std::vector<std::vector<int>>> process(
                                    TensorPTRWithName& input_data, 
                                    std::vector<int>& shape, 
                                    std::vector<float>& threshold, 
                                    std::vector<int>& max_peak_num);

**参数说明1:**

* input_data: TensorPTRWithName

输入参数。输入数据。

* shape: std::vector<int>

输入参数。输入数据的宽和高。

* threshold: std::vector<float>

输入参数。检测阈值序列。

* max_peak_num: std::vector<int>

输入参数。最大被检测关键点的数量。

**接口形式2:**
    .. code-block:: c

        std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<float>>, std::vector<std::vector<int>>> process(
                                        std::map<std::string, Tensor&>& input_data, 
                                        std::vector<int>& shape, 
                                        std::vector<float>& threshold, 
                                        std::vector<int>& max_peak_num);

**参数说明2:**

* input_data: std::map<std::string, Tensor&>

输入参数。输入数据。

* shape: std::vector<int>

输入参数。输入数据的宽和高。

* threshold: std::vector<float>

输入参数。检测阈值序列。

* max_peak_num: std::vector<int>

输入参数。最大被检测关键点的数量。

**返回值说明:**

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<float>>, std::vector<std::vector<int>>>

* 第一个输出: std::vector<std::vector<int>> 

在每个通道被检测的关键点数量。

* 第二个输出: std::vector<std::vector<float>>

所有被检测的关键点的置信度。

* 第三个输出: std::vector<std::vector<int>>

所有被检测的关键点的拉平坐标。


reset_network_c
>>>>>>>>>>>>>>>>>>>>>

更新关键点通道数。

**接口形式:**
    .. code-block:: c

        int reset_network_c(int network_c_new);

**参数说明:**

* network_c_new: int

要更新的通道数。

**返回值说明:**

成功返回0，其他值表示失败。

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
        
        int main() {  
            int tpu_id = 0;  
            std::string image_path = "../../../sophon-demo/sample/OpenPose/datasets/test/3.jpg";  
            sail::Decoder decoder(image_path, true, tpu_id);  
            std::string bmodel_path = "../../../sophon-demo/sample/OpenPose/models/BM1684/pose_coco_fp32_1b.bmodel";  
            sail::Handle handle(tpu_id);  
            sail::Engine net(bmodel_path, tpu_id, sail::IOMode::SYSIO);  
            cv::Mat src_img = cv::imdecode(std::vector<uchar>(std::ifstream(image_path).readIntoVector()), -1);  
            std::string graph_name = net.get_graph_names()[0];  
            std::string input_name = net.get_input_names(graph_name)[0];  
            std::string output_name = net.get_output_names(graph_name)[0];  
            int h, w, _;  
            cv::split(src_img, {&h, &w, nullptr});  
            int net_h = net.get_input_shape(graph_name, input_name)[2];  
            int net_w = net.get_input_shape(graph_name, input_name)[3];  
            int out_h = net.get_output_shape(graph_name, output_name)[2];  
            float scale = std::min(static_cast<float>(net_h) / h, static_cast<float>(net_w) / w);  
        
            cv::Mat resize_img = cv::resize(src_img, cv::Size(0, 0), scale, scale, cv::INTER_CUBIC);  
            cv::Mat pad_img = cv::copyMakeBorder(resize_img, 0, net_h - resize_img.rows, 0, net_w - resize_img.cols, cv::BORDER_CONSTANT, value={114, 114, 114});  
            std::vector<float> img(net_h * net_w * 3);  
            cv::transform(pad_img.data, img.data(), net_h * net_w, 3, [](uchar v) { return (v - 128) / 255.0f; });  
        
            std::map<std::string, std::vector<float>> outputs = net.process(graph_name, {input_name: img});  
        
            std::vector<float> output(out_h * out_h * 3);  
            output = cv2::resize(output, (0, 0), fx=stride, fy=stride, interpolation=cv2::INTER_CUBIC);  
            output = output[:resize_img.shape[0], :resize_img.shape[1], :];  
            output = cv2::resize(output, (src_img.shape[1], src_img.shape[0]), interpolation=cv2::INTER_CUBIC);   
            std::vector<float> input_data;  
            for (int i = 0; i < output.size(); i += 3) {  
                input_data.push_back(cv::GaussianFilter(output.data() + i, sigma=3));  
            }  
            int point_num = int(net.get_output_shape(graph_name, output_name)[1] / 3) - 1;
            tpu_api_openpose_part_nms_postprocess_t api;

            bm_device_mem_t output_data, output_num;
            assert(BM_SUCCESS == bm_malloc_device_byte(
                                    handle, &output_data,
                                    sizeof(float) * api.input_c * input_h * input_w));
            assert(BM_SUCCESS == bm_malloc_device_byte(handle,
                                                        &output_num,
                                                        sizeof(int) * api.input_c));
            api.input_data_addr = bm_mem_get_device_addr(input_data);
            api.output_data_addr = bm_mem_get_device_addr(output_data);
            api.num_output_data_addr = bm_mem_get_device_addr(output_num);

            api.input_h = net_h;
            api.input_w = net_w;
            api.max_peak_num = 96;
            api.nms_thresh = 0.05;

            assert(BM_SUCCESS == tpu_kernel_launch(handle,
                                                    func_id, &api, sizeof(api)));
            bm_thread_sync(handle);

            bm_memcpy_d2s_partial(handle, num_result, output_num,
                                    sizeof(int) * api.input_c);
            const int peak_num = num_result[api.input_c - 1];
            bm_memcpy_d2s_partial(handle, score_out_result,
                                    input_data, peak_num * sizeof(float));
            bm_memcpy_d2s_partial_offset(handle, coor_out_result,
                                        input_data, peak_num * sizeof(int),
                                        peak_num * sizeof(float));

            bm_free_device(handle, input_data);
            bm_free_device(handle, output_data);
            bm_free_device(handle, output_num);
            return 0;
            }