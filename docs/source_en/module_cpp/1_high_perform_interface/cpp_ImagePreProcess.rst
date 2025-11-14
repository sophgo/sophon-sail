ImagePreProcess
______________________

General preprocessing interface, internally implemented using thread pool.

Constructor
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

**interface:**
    .. code-block:: c

        ImagePreProcess(
            int batch_size,
            sail_resize_type resize_mode,
            int tpu_id=0, 
            int queue_in_size=20, 
            int queue_out_size=20,
            bool use_mat_flag=false);


**Parameters:**

* batch_size: int

Input parameter. The batch size of the output results.

* resize_mode: sail_resize_type

Input parameter. Internal scale transformation method.

* tpu_id: int

Input parameter. The tpu id used, the default is 0.

* queue_in_size: int

Input parameter. Enter the maximum length of the image queue buffer. The default is 20.

* queue_out_size: int

Input parameter. The maximum length of the output Tensor queue cache, the default is 20.

* use_mat_output: bool

Input parameter. Whether to use Mat of OPENCV as the output of the image, the default is false, not used.

SetResizeImageAtrr
>>>>>>>>>>>>>>>>>>>>

Set the properties of the image resizing.

**Interface:**
    .. code-block:: c

        void SetResizeImageAtrr(			    
            int output_width,				    
            int output_height,				    
            bool bgr2rgb,					    
            bm_image_data_format_ext  dtype);	

**parameters:**
            
* output_width: int

Input parameter. Image width after scale transformation.

* output_height: int

Input parameter. Image height after scale transformation.

* bgr2rgb: bool

Input parameter. Whether to convert images with BGR to GRB.

* dtype: ImgDtype  

Input parameter. The data type after image scale conversion, the current version only supports BM_FLOAT32, BM_INT8, BM_UINT8. Can be set according to the input data type of the model.


SetPaddingAtrr
>>>>>>>>>>>>>>>>>>>>

Setting the properties of Padding only takes effect when resize_mode is BM_PADDING_VPP_NEAREST, BM_PADDING_TPU_NEAREST, BM_PADDING_TPU_LINEAR, or BM_PADDING_TPU_BICUBIC.

**interface:**
    .. code-block:: c

        void SetPaddingAtrr(		    
            int padding_b=114,		        
            int padding_g=114,		        
            int padding_r=114,		        
            int align=0);	

**Parameters:**
* padding_b: int

Input parameter. The b channel pixel value to be pdding, default is 114.

* padding_g: int

Input parameter. The g channel pixel value to be pdding, default is 114.
                
* padding_r: int

Input parameter. The r channel pixel value to be pdding, default is 114.

* align: int

Input parameter. Image padding is position, 0 means filling from the upper left corner, 1 means center filling, default is 0.


SetConvertAtrr
>>>>>>>>>>>>>>>>>>>>

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
>>>>>>>>>>>>>>>

Send data.

**interface:**
    .. code-block:: c

        int PushImage(
            int channel_idx, 
            int image_idx, 
            BMImage &image);

**Parameters:**

* channel_idx: int

Input parameter. Enter the channel number of the image.
                
* image_idx: int

Input parameter. Enter the number of the image.

* image: BMImage

Input parameter. The input image.

**返回值说明:**

Returns 0 if set successfully, other values indicate failure.
            
GetBatchData
>>>>>>>>>>>>>>>

Get the result of processing.

**Interface:**
    .. code-block:: c
        
        std::tuple<sail::Tensor, 
            std::vector<BMImage>,
            std::vector<int>,
            std::vector<int>,
            std::vector<std::vector<int>>> GetBatchData();
        
**Returns:**
tuple[data, images, channels, image_idxs, padding_attrs]

* data: Tensor

    The processed result Tensor.

* images: std::vector<BMImage>

    Original image sequence.

* channels: std::vector<int>

    The channel sequence of the original image.

* image_idxs: std::vector<int>

    Numbered sequence of original images.

* padding_attrs: std::vector<std::vector<int> >

    Attribute list of the filled image, filled starting point coordinate x, starting point coordinate y, width after scale transformation, height after scale transformation

set_print_flag
>>>>>>>>>>>>>>>

Set the flag for printing logs. If this interface is not called, the log will not be printed.

**Interface:**
    .. code-block:: c

        void set_print_flag(bool print_flag);
        
**Returns:**

* flag: bool

The printing flag, false means not printing, true means printing.
