sail.ImagePreProcess
______________________

通用预处理接口，内部使用线程池的方式实现。

__init__
>>>>>>>>>

**接口形式:**
    .. code-block:: python

        def __init__(self,
                    batch_size: int, 
                    resize_mode:sail_resize_type,
                    tpu_id: int=0, 
                    queue_in_size: int=20, 
                    queue_out_size: int=20,
                    use_mat_output: bool = False)


**参数说明:**

* batch_size: int

输入参数。输出结果的batch size。

* resize_mode: sail_resize_type

输入参数。内部尺度变换的方法。

* tpu_id: int

输入参数。使用的智能视觉深度学习处理器 id，默认为0。

* queue_in_size: int

输入参数。输入图像队列缓存的最大长度，默认为20。

* queue_out_size: int

输入参数。输出Tensor队列缓存的最大长度，默认为20。

* use_mat_output: bool

输入参数。是否使用OpenCV的Mat作为图片的输出，默认为False，不使用。

SetResizeImageAtrr
>>>>>>>>>>>>>>>>>>>>

设置图像尺度变换的属性。

**接口形式:**
    .. code-block:: python

        def SetResizeImageAtrr(self,
                    output_width: int, 
                    output_height: int,
                    bgr2rgb: bool, 
                    dtype: ImgDtype) -> None

**参数说明:**
            
* output_width: int

输入参数。尺度变换之后的图像宽度。

* output_height: int

输入参数。尺度变换之后的图像高度。

* bgr2rgb: bool

输入参数。是否将图像有BGR转换为GRB。

* dtype: ImgDtype  

输入参数。图像尺度变换之后的数据类型，当前版本只支持BM_FLOAT32,BM_INT8,BM_UINT8。可根据模型的输入数据类型设置。


SetPaddingAtrr
>>>>>>>>>>>>>>>>>>>>

设置Padding的属性，只有在resize_mode为 BM_PADDING_VPP_NEAREST、BM_PADDING_TPU_NEAREST、BM_PADDING_TPU_LINEAR、BM_PADDING_TPU_BICUBIC时生效。

**接口形式:**
    .. code-block:: python

        def SetPaddingAtrr(self,
                    padding_b: int=114,
                    padding_g: int=114,
                    padding_r: int=114,
                    align: int=0) -> None

**参数说明:**
* padding_b: int

输入参数。要pdding的b通道像素值，默认为114。

* padding_g: int

输入参数。要pdding的g通道像素值，默认为114。
                
* padding_r: int

输入参数。要pdding的r通道像素值，默认为114。

* align: int

输入参数。图像填充为位置，0表示从左上角开始填充，1表示居中填充，默认为0。


SetConvertAtrr
>>>>>>>>>>>>>>>>>>>>

设置线性变换的属性。

**接口形式:**
    .. code-block:: python

        def SetConvertAtrr(self, alpha_beta) -> int 

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
>>>>>>>>>>>>>>>

送入数据。

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

输入参数。输入图像。

**返回值说明:**

设置成功返回0，其他值时表示失败。
            
GetBatchData
>>>>>>>>>>>>>>>

获取处理的结果。

**接口形式:**
    .. code-block:: python
        
        def GetBatchData(self) 
            -> tuple[Tensor, list[BMImage],list[int],list[int],list[list[int]]]
        """ Get the Batch Data object
        
**返回值说明:**
tuple[data, images, channels, image_idxs, padding_attrs]

* data: Tensor

    处理后的结果Tensor。

* images: list[BMImage]

    原始图像序列。

* channels: list[int]

    原始图像的通道序列。

* image_idxs: list[int]

    原始图像的编号序列。

* padding_attrs: list[list[int]]

    填充图像的属性列表，填充的起始点坐标x、起始点坐标y、尺度变换之后的宽度、尺度变换之后的高度

set_print_flag
>>>>>>>>>>>>>>>

设置打印日志的标志位，不调用此接口时不打印日志。

**接口形式:**
    .. code-block:: python

        def set_print_flag(self, flag: bool) -> None:
        
**返回值说明:**

* flag: bool

打印的标志位，False时表示不打印，True时表示打印。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        import cv2 as cv

        if __name__ == '__main__':
            tpu_id = 0
            batch_size = 1
            image_path = "./data/zidane.jpg"
            handle = sail.Handle(tpu_id)

            alpha_beta = (1, 0), (1, 0), (1, 0)
            decoder = sail.Decoder(image_path, False, tpu_id)

            sail_ipp = sail.ImagePreProcess(batch_size, sail.sail_resize_type.BM_RESIZE_VPP_NEAREST, tpu_id, 20, 20, False)

            sail_ipp.SetResizeImageAtrr(640, 640, False, sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE)
            ret1 = sail_ipp.SetConvertAtrr(alpha_beta)
            # sail_ipp.set_print_flag(True)
            bm_i = sail.BMImage()
            for i in range(0, batch_size):
                decoder.read(handle, bm_i)
                sail_ipp.PushImage(0, i, bm_i)

            result = sail_ipp.GetBatchData()
            decoder.release()

            tensor = result[0]
            t_npy = tensor.asnumpy()
            result_img = t_npy[0].transpose(1, 2, 0)

            raw_img = cv.imread(image_path)
            resize_img = cv.resize(raw_img, (640, 640), interpolation=cv.INTER_NEAREST)
            max_diff = abs((resize_img.astype(int) - result_img.astype(int)).max())
            min_diff = abs((resize_img.astype(int) - result_img.astype(int)).min())
            diff = max(max_diff, min_diff)
            print(max_diff,min_diff,diff)