tpu_kernel_api_yolov5_out_without_decode
____________________________________________

For the 1-output yolov5 model, Tensor Computing Processor Kernel is used to accelerate post-processing. Currently, only BM1684x is supported, and the version of libsophon must be no less than 0.4.6 (v23.03.01).

Constructor
>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: c
          
        tpu_kernel_api_yolov5_out_without_decode(int device_id, 
                                            const std::vector<int>& shapes, 
                                            int network_w=640, 
                                            int network_h=640,
                                            std::string module_file = "/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so");

**Paramters:**

* device_id: int

Input parameter. The used device ID.

* shape: std::vector<int>

Input parameter. The shape of the input data.

* network_w: int

Input parameter. The input width of the model, default is 640.

* network_h: int

Input parameter. The input width of the model, default is 640.

* module_file: string

Input parameter. Kernel module file path, the default is "/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so".


process
>>>>>>>>>>>>>

Processing interface

**Interface 1:**
    .. code-block:: c

        std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> process(
                                    TensorPTRWithName& input, 
                                    float dete_threshold, 
                                    float nms_threshold);

**Paramters1:**

* input_data: TensorPTRWithName

Input parameter. Input data, including one output.

* dete_threshold: float

Input parameter. The detection threshold.

* nms_threshold: float

Input parameters. The nms threshold sequence.

**Interface 2:**
    .. code-block:: c

        std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> process(
                                        std::map<std::string, Tensor&>& input, 
                                        float dete_threshold, 
                                        float nms_threshold);

**Paramters 2:**

* input_data: std::map<std::string, Tensor&>

Input parameter. Input data, including one output.

* dete_threshold: float

Input parameter. The detection threshold.

* nms_threshold: float

Input parameters. The nms threshold sequence.

**Returns:**

std::vector<std::vector<std::tuple<left, top, right, bottom, class_id, score> > >

* left: int 

The leftmost x coordinate of the detection result.

* top: int

The uppermost y coordinate of the detection result.

* right: int

The rightmost x coordinate of the detection result.

* bottom: int

The lowest y coordinate of the detection result.

* class_id: int

The category ID of the test result.

* score: float

The score of the test result.