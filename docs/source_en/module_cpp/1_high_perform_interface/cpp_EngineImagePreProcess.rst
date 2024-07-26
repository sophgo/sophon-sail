EngineImagePreProcess
___________________________

The image inference interface with preprocessing function uses the thread pool internally, which is more efficient in the Python environment.

Constructor
>>>>>>>>>>>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: c

        EngineImagePreProcess(const std::string& bmodel_path, 
                            int tpu_id, 
                            bool use_mat_output=false,
                            std::vector<int> core_list = {});

**Parameters:**

* bmodel_path: string 

Input parameter. The path to input the model.

* tpu_id: int

Input parameter. The tpu id used.

* use_mat_output: bool

Input parameter. Whether to use Mat of OPENCV as the output of the image, the default is False, not used.

* use_mat_output: bool

When using the processor and BModel that support multi-core inference, you can choose multiple cores to use for inference. 
By default, N cores starting from core0 are used for inference, and N is determined by the current bmodel.
For processors and bmodel models that only support single-core inference, only a single core used for inference is supported, 
and the input list length of the parameter must be 1, if the length of the incoming list is greater than 1, the inference will be automatically on the core0.
Default is empty. If this parameter is not specified, the inference is performed by N cores starting from core 0.

InitImagePreProcess
>>>>>>>>>>>>>>>>>>>>>>>>>>

Initialize the image preprocessing module.

**Interface:**
    .. code-block:: c

        int InitImagePreProcess(
            sail_resize_type resize_mode,
            bool bgr2rgb=false,					    
            int queue_in_size=20, 
            int queue_out_size=20);

                    
**Parameters:**

* resize_mode: sail_resize_type

Input parameter. Internal scale transformation method.

* bgr2rgb: bool

Input parameter. Whether to convert images with BGR to RGB.

* queue_in_size: int

Input parameter. Enter the maximum length of the image queue buffer. The default is 20.
Must not be less than the batch_size of bmodel, if not, the value will be set to batch_size.

* queue_out_size: int

Input parameter. The maximum length of the preprocessing result Tensor queue cache, the default is 20.
Must not be less than the batch_size of bmodel, if not, the value will be set to batch_size.

**Returns:**

Returns 0 on success, fails on other values.
           

SetPaddingAtrr
>>>>>>>>>>>>>>>>>>>

Set the properties of Padding. This setting only takes effect when resize_mode is BM_PADDING_VPP_NEAREST, BM_PADDING_TPU_NEAREST, BM_PADDING_TPU_LINEAR, or BM_PADDING_TPU_BICUBIC.

**Interface:**
    .. code-block:: c

        int SetPaddingAtrr(
            int padding_b=114,
            int padding_g=114,	
            int padding_r=114, 
            int align=0);

**Parameters:**
* padding_b: int

Input parameter. The b channel pixel value to be padding, default is 114.

* padding_g: int

Input parameter. The g channel pixel value to be padding, default is 114.
                
* padding_r: int

Input parameter. The r channel pixel value to be padding, default is 114.

* align: int

Input parameter. The position of the image padding, 0 means padding from the upper left corner, 1 means padding in the center, the default is 0.
          
**Returns:**

Returns 0 on success, fails on other values.


SetConvertAtrr
>>>>>>>>>>>>>>>>>>>

Set the properties of the linear transformation.

**Interface:**
    .. code-block:: c

        int SetConvertAtrr(
            const std::tuple<
                std::pair<float, float>,
                std::pair<float, float>,
                std::pair<float, float>> &alpha_beta);

**Parameters:**

* alpha_beta: (a0, b0), (a1, b1), (a2, b2)。输入参数。

    a0 describes the linear transformation coefficient of the 0th channel;

    b0 describes the linear transformation offset of the 0th channel;

    a1 describes the linear transformation coefficient of the first channel;

    b1 describes the linear transformation offset of the first channel;

    a2 describes the linear transformation coefficient of the second channel;

    b2 describes the linear transformation offset of the second channel;

**Returns:**

If the setting is successful, 0 is returned. If other values are set, the setting fails.


PushImage
>>>>>>>>>>>>>>

Send image data

**Interface:**
    .. code-block:: c

        int PushImage(
            int channel_idx, 
            int image_idx, 
            BMImage &image);

**Parameters:**
* channel_idx: int

Input parameter. The channel id used to input the image.

* image_idx: int
                
Input parameter. The number of the input image.

* image: BMImage
                
Input parameter. The input image.

**Returns:**

Returns 0 on success, fails on other values.


GetBatchData
>>>>>>>>>>>>>>>>>>>

Obtain the inference results of a batch. When calling this interface, since the returned result type is BMImage, use_mat_output must be False. It is worth noting that output tensors must be manually released.

**Interface:**
    .. code-block:: c

        std::tuple<std::map<std::string,sail::Tensor*>, 
            std::vector<BMImage>,
            std::vector<int>,
            std::vector<int>,
            std::vector<std::vector<int>>> GetBatchData();

**Returns:**

tuple[output_array, ost_images, channels, image_idxs, padding_attrs]

* output_array: std::map<std::string,sail::Tensor*>

The result of inference.

* ost_images: std::vector<BMImage>

The original image sequence.

* channels: std::vector<int>

The channel sequence of the original image corresponding to the result.

* image_idxs: std::vector<int>

The number sequence of the original image corresponding to the result.

* padding_attrs: std::vector<std::vector<int> >

The attribute list of the filled image, the starting point coordinate x of the filling, the starting point coordinate y, the width after scale transformation, and the height after scale transformation.



GetBatchData_CV
>>>>>>>>>>>>>>>>>>>>>>>

Obtain the inference result of a batch. When calling this interface, since the returned result type is cv::Mat, use_mat_output must be True. It is worth noting that output tensors must be manually released.

**Interface:**
    .. code-block:: c

        std::tuple<std::map<std::string,sail::Tensor*>, 
            std::vector<cv::Mat>,
            std::vector<int>,
            std::vector<int>,
            std::vector<std::vector<int>>> GetBatchData_CV();

**Returns:**

tuple[output_array, ost_images, channels, image_idxs, padding_attrs]

* output_array: std::map<std::string,sail::Tensor*>

The result of inference.

* ost_images: std::vector<cv::Mat>

The original image sequence.

* channels: std::vector<int>

The channel sequence of the original image corresponding to the result.

* image_idxs: std::vector<int>

The number sequence of the original image corresponding to the result.

* padding_attrs: std::vector<std::vector<int> >

The attribute list of the filled image, the starting point coordinate x of the filling, the starting point coordinate y, the width after scale transformation, and the height after scale transformation.


get_graph_name
>>>>>>>>>>>>>>>>

Get the name of the operation graph of model.

**Interface:**
    .. code-block:: c

        std::string get_graph_name();

**Returns:**

Returns the name of the first operational graph of the model.

            
get_input_width
>>>>>>>>>>>>>>>>

Get the width of the model input.

**Interface:**
    .. code-block:: c

        int get_input_width();

**Returns:**

Returns the width of the model input.

            
get_input_height
>>>>>>>>>>>>>>>>>>>

Get the height of the model input.

**Interface:**
    .. code-block:: c

        int get_input_height();

**Returns:**

Returns the width of the model input.

            
get_output_names
>>>>>>>>>>>>>>>>>>>

Get the name of the model output Tensor.

**Interface:**
    .. code-block:: c

        std::vector<std::string> get_output_names();

**Returns:**

Returns the names of all output Tensors of the model.
   
            
get_output_shape
>>>>>>>>>>>>>>>>>>>

Get the shape of the specified output Tensor

**Interface:**
    .. code-block:: c
        
        std::vector<int> get_output_shape(const std::string& tensor_name);

**Parameters:**

* tensor_name: string

The name of the specified output Tensor.

**Returns:**

Returns the shape of the specified output Tensor.