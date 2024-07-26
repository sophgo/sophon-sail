sail.EngineImagePreProcess
___________________________

带有预处理功能的图像推理接口，内部使用线程池的方式，Python下面有更高的效率。

__init__
>>>>>>>>>

**接口形式:**
    .. code-block:: python

        def __init__(self,
                    bmodel_path: str, 
                    tpu_id: int,
                    use_mat_output: bool = False,
                    core_list:list = [])

**参数说明:**

* bmodel_path: str 

输入参数。输入模型的路径。

* tpu_id: int

输入参数。使用的智能视觉深度学习处理器 id。

* use_mat_output: bool

输入参数。是否使用OpenCV的Mat作为图片的输出，默认为False，不使用。

* use_mat_output: bool

输入参数。
使用支持多核推理的处理器和bmodel时，可以选择推理时使用的多个核心，默认使用从core0开始的N个core来做推理，N由当前bmodel决定。
对于仅支持单核推理的处理器和bmodel模型，仅支持选择推理使用的单个核心，参数的输入列表长度必须为1，若传入列表长度大于1，将自动在0号核心上推理。
默认为空不指定时，将默认从0号核心开始的N个core来做推理。

InitImagePreProcess
>>>>>>>>>>>>>>>>>>>>>>>>>>

初始化图像预处理模块。

**接口形式:**
    .. code-block:: python

        def InitImagePreProcess(self,
                    resize_mode: sail_resize_type, 
                    bgr2rgb: bool = False,
                    queue_in_size: int = 20, 
                    queue_out_size: int = 20) -> int

                    
**参数说明:**

* resize_mode: sail_resize_type

输入参数。内部尺度变换的方法。

* bgr2rgb: bool

输入参数。是否将图像有BGR转换为GRB。

* queue_in_size: int

输入参数。输入图像队列缓存的最大长度，默认为20。queue_in_size必须大于模型batch_size，若小于模型batch_size，将自动调整为模型batch_size。

* queue_out_size: int

输入参数。预处理结果Tensor队列缓存的最大长度，默认为20。queue_out_size必须大于模型batch_size，若小于模型batch_size，将自动调整为模型batch_size。

**返回值说明:**

成功返回0，其他值时失败。
           

SetPaddingAtrr
>>>>>>>>>>>>>>>>>>>

设置Padding的属性，只有在resize_mode为 BM_PADDING_VPP_NEAREST、BM_PADDING_TPU_NEAREST、BM_PADDING_TPU_LINEAR、BM_PADDING_TPU_BICUBIC时生效。

**接口形式:**
    .. code-block:: python

        def SetPaddingAtrr(self,
                    padding_b:int=114,
                    padding_g:int=114,
                    padding_r:int=114,
                    align:int=0) -> int 

**参数说明:**
* padding_b: int

输入参数。要pdding的b通道像素值，默认为114。

* padding_g: int

输入参数。要pdding的g通道像素值，默认为114。
                
* padding_r: int

输入参数。要pdding的r通道像素值，默认为114。

* align: int

输入参数。图像填充为位置，0表示从左上角开始填充，1表示居中填充，默认为0。
          
**返回值说明:**

成功返回0，其他值时失败。


SetConvertAtrr
>>>>>>>>>>>>>>>>>>>

设置线性变换的属性。

**接口形式:**
    .. code-block:: python

        def SetConvertAtrr(self, alpha_beta) -> int:

**参数说明:**

* alpha_beta: (a0, b0), (a1, b1), (a2, b2)。输入参数。

    a0 描述了第 0 个 channel 进行线性变换的系数；

    b0 描述了第 0 个 channel 进行线性变换的偏移；

    a1 描述了第 1 个 channel 进行线性变换的系数；

    b1 描述了第 1 个 channel 进行线性变换的偏移；

    a2 描述了第 2 个 channel 进行线性变换的系数；

    b2 描述了第 2 个 channel 进行线性变换的偏移；

**返回值说明:**

设置成功返回0，其他值时设置失败。


PushImage
>>>>>>>>>>>>>>

送入图像数据

**接口形式:**
    .. code-block:: python

        def PushImage(self,
                    channel_idx: int, 
                    image_idx: int, 
                    image: BMImage) -> int

**参数说明:**
* channel_idx: int

输入参数。输入图像的通道号。

* image_idx: int
                
输入参数。输入图像的编号。

* image: BMImage
                
输入参数。输入的图像。

**返回值说明:**

成功返回0，其他值时失败。


GetBatchData_Npy
>>>>>>>>>>>>>>>>>>>

获取一个batch的推理结果，调用此接口时，由于返回的结果类型为BMImage，所以use_mat_output必须为False。

**接口形式:**
    .. code-block:: python

        def GetBatchData_Npy(self) 
        -> tuple[[dict[str, ndarray], list[BMImage],list[int],list[int],list[list[int]]]]

**返回值说明:**

tuple[output_array, ost_images, channels, image_idxs, padding_attrs]

* output_array: dict[str, ndarray]

推理结果。

* ost_images: list[BMImage]

原始图片序列。

* channels: list[int]

结果对应的原始图片的通道序列。

* image_idxs: list[int]

结果对应的原始图片的编号序列。

* padding_attrs: list[list[int]]

填充图像的属性列表，填充的起始点坐标x、起始点坐标y、尺度变换之后的宽度、尺度变换之后的高度。



GetBatchData_Npy2
>>>>>>>>>>>>>>>>>>>>>>>

获取一个batch的推理结果，调用此接口时，由于返回的结果类型为numpy.ndarray[numpy.uint8]，所以use_mat_output必须为True。

**接口形式:**
    .. code-block:: python

        def GetBatchData_Npy2(self) 
            -> tuple[dict[str, ndarray], list[numpy.ndarray[numpy.uint8]],list[int],list[int],list[list[int]]]

**返回值说明:**

tuple[output_array, ost_images, channels, image_idxs, padding_attrs]

* output_array: dict[str, ndarray]

推理结果。

* ost_images: list[numpy.ndarray[numpy.uint8]]

原始图片序列。

* channels: list[int]

结果对应的原始图片的通道序列。

* image_idxs: list[int]

结果对应的原始图片的编号序列。

* padding_attrs: list[list[int]]

填充图像的属性列表，填充的起始点坐标x、起始点坐标y、尺度变换之后的宽度、尺度变换之后的高度。

GetBatchData
>>>>>>>>>>>>>>>>

获取一个batch的推理结果，调用此接口时，由于返回的结果类型为BMImage，所以use_mat_output必须为False。值得注意的是，该接口输出的tensor需要手动进行释放。

**接口形式:**
    .. code-block:: python
        
        def GetBatchData(self,
                    need_d2s: bool = True) 
                    -> tuple[list[TensorPTRWithName], list[BMImage],list[int],list[int],list[list[int]]]

**参数说明:**

* need_d2s: bool

是否需要将数据搬运至系统内存，默认为True,需要搬运。

**返回值说明:**

tuple[output_array, ost_images, channels, image_idxs, padding_attrs]

* output_array: list[TensorPTRWithName]

推理结果。

* ost_images: list[BMImage]

原始图片序列。

* channels: list[int]

结果对应的原始图片的通道序列。

* image_idxs: list[int]

结果对应的原始图片的编号序列。

* padding_attrs: list[list[int]]

填充图像的属性列表，填充的起始点坐标x、起始点坐标y、尺度变换之后的宽度、尺度变换之后的高度。


get_graph_name
>>>>>>>>>>>>>>>>

获取模型的运算图名称。

**接口形式:**
    .. code-block:: python

        def get_graph_name(self) -> str

**返回值说明:**

返回模型的第一个运算图名称。

            
get_input_width
>>>>>>>>>>>>>>>>

获取模型输入的宽度。

**接口形式:**
    .. code-block:: python

        def get_input_width(self) -> int

**返回值说明:**

返回模型输入的宽度。

            
get_input_height
>>>>>>>>>>>>>>>>>>>

获取模型输入的高度。

**接口形式:**
    .. code-block:: python

        def get_input_height(self) -> int

**返回值说明:**

返回模型输入的宽度。

            
get_output_names
>>>>>>>>>>>>>>>>>>>

获取模型输出Tensor的名称。

**接口形式:**
    .. code-block:: python

        def get_output_names(self) -> list[str]

**返回值说明:**

返回模型所有输出Tensor的名称。
   
            
get_output_shape
>>>>>>>>>>>>>>>>>>>

获取指定输出Tensor的shape

**接口形式:**
    .. code-block:: python
        
        def get_output_shape(self, tensor_name: str) -> list[int]

**参数说明:**

* tensor_name: str

指定的输出Tensor的名称。

**返回值说明:**

返回指定输出Tensor的shape。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np

        if __name__ == '__main__':
            dev_id = 0
            handle = sail.Handle(dev_id)
            image_path = "./data/zidane.jpg"
            decoder = sail.Decoder(image_path, True, dev_id)
            bmodel_path = '../../../sophon-demo/sample/YOLOv5/models/BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel'
            alpha_beta = (1.0/255.0, 0), (1.0/255.0, 0), (1.0/255.0, 0)

            resize_type = sail.sail_resize_type.BM_PADDING_TPU_LINEAR
            sail_engineipp = sail.EngineImagePreProcess(bmodel_path, tpu_id, False)
            sail_engineipp.InitImagePreProcess(resize_type, False, 20, 20)

            sail_engineipp.SetPaddingAtrr()
            sail_engineipp.SetConvertAtrr(alpha_beta)
            

            get_i_w = sail_engineipp.get_input_width()
            get_i_h = sail_engineipp.get_input_height()
            output_name = sail_engineipp.get_output_names()[0]
            output_shape = sail_engineipp.get_output_shape(output_name)

            bm_i = sail.BMImage()
            decoder.read(handle, bm_i)
            sail_engineipp.PushImage(0, 0, bm_i)

            res = sail_engineipp.GetBatchData(True)
            print(output_name,output_shape,get_i_h,get_i_w,res)