tpu_kernel_api_openpose_part_nms
____________________________________________

For the OpenPose model, Tensor Computing Processor Kernel is used to accelerate part nms post-processing. Currently, only BM1684x is supported, and the version of libsophon must be no less than 0.4.6 (v23.03.01).

Constructor
>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: c
          
        tpu_kernel_api_openpose_part_nms(int device_id, 
                                            int network_c,
                                            std::string module_file = "/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so");

**Paramters:**

* device_id: int

Input parameter. The used device ID.

* network_c: int

Input parameter. The input channel of the model, corresponding to the number of keypoint channels.

* module_file: string

Input parameter. Kernel module file path, the default is "/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so".


process
>>>>>>>>>>>>>

Processing interface

**Interface 1:**
    .. code-block:: c

        std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<float>>, std::vector<std::vector<int>>> process(
                                    TensorPTRWithName& input_data, 
                                    std::vector<int>& shape, 
                                    std::vector<float>& threshold, 
                                    std::vector<int>& max_peak_num);

**Paramters1:**

* input_data: TensorPTRWithName

Input parameter. Input data.

* shape: std::vector<int>

Input. Input data width and height.

* threshold: std::vector<float>

Input. Detection threshold.

* max_peak_num: std::vector<int>

Input. The max number of detected peaks.

**Interface 2:**
    .. code-block:: c

        std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<float>>, std::vector<std::vector<int>>> process(
                                        std::map<std::string, Tensor&>& input_data, 
                                        std::vector<int>& shape, 
                                        std::vector<float>& threshold, 
                                        std::vector<int>& max_peak_num);

**Paramters 2:**

* input_data: std::map<std::string, Tensor&>

Input parameter. Input data.

* shape: std::vector<int>

Input. Input data width and height.

* threshold: std::vector<float>

Input. Detection threshold.

* max_peak_num: std::vector<int>

Input. The max number of detected peaks.

**Returns:**

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<float>>, std::vector<std::vector<int>>>

* 1st output: std::vector<std::vector<int>>

The number of the detected peaks in each channel.

* 2nd: std::vector<std::vector<float>>

The scores of all detected peaks.

* 3rd: std::vector<std::vector<int>>

The flatten coordinates of all detected peaks.


reset_network_c
>>>>>>>>>>>>>>>>>>>>>

Update the channel number of the input.

**Interface:**
    .. code-block:: c

        int reset_network_c(int network_c_new);

**Paramters:**

* network_c_new: int

The number of channels to be updated.

**Returns:**

Returns 0 on success, other values indicate failure.