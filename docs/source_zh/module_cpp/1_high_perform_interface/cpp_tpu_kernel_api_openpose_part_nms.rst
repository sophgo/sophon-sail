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