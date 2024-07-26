sail.tpu_kernel_api_openpose_part_nms
____________________________________________

The part nms post-processing is accelerated using the Tensor Computing Processor Kernel, which currently only supports BM1684x, and the version of libsophon must be no less than 0.4.6 (v23.03.01).

\_\_init\_\_
>>>>>>>>>>>>

**Interface:**
    .. code-block:: python
          
        def __init__(
                    self,
                    device_id: int,
                    network_c: int, 
                    module_file: str="/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so")

**Parameters:**

* device_id: int

Input. Device ID used.

* network_c: int

Input. Input channel of the model, corresponding to the number of keypoint channels.

* module_file: str

Input. File path of the kernel module, default is "/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so". 


process
>>>>>>>>>>>>>

Processing interface.

**Interface 1:**
    .. code-block:: python

        def process(self, 
            input_data: TensorPTRWithName, 
            shape: list[int],
            threshold: list[float],
            max_peak_num: list[int]) 
                -> tuple[list[list[int]], list[list[float]], list[list[int]]]

**Parameters 1:**

* input_data: TensorPTRWithName

Input. Input data.

* shape: list[int]

Input. Input data width and height.

* threshold: list[float]

Input. Detection threshold.

* max_peak_num: list[int]

Input. The max number of detected peaks.

**Interface 2:**
    .. code-block:: python

        def process(self, 
            input_data: dict[str, Tensor], 
            shape: list[int],
            threshold: list[float],
            max_peak_num: list[int]) 
                -> tuple[list[list[int]], list[list[float]], list[list[int]]]

**Parameters 2:**

* input_data: dict[str, Tensor]

Input. Input data.

* shape: list[int]

Input. Input data width and height.

* threshold: list[float]

Input. Detection threshold.

* max_peak_num: list[int]

Input. The max number of detected peaks.

**Returns:**

tuple[list[list[int]], list[list[float]], list[list[int]]]

* 1st output: list[list[int]] 

The number of the detected peaks in each channel.

* 2nd: list[list[float]]

The scores of all detected peaks.

* 3rd: list[list[int]]

The flatten coordinates of all detected peaks.


reset_network_c
>>>>>>>>>>>>>>>>>>

Update the channel number of the input.

**Interface:**
    .. code-block:: python

        def reset_network_c(self, network_c_new: int) -> int

**Parameters:**

* network_c_new: int

The number of channels to be updated.

**Returns:**

A return value of 0 indicates success, while other values indicate failure.