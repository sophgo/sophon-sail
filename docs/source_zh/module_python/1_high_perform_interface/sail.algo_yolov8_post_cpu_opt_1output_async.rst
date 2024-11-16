sail.algo_yolov8_post_cpu_opt_1output_async
______________________________________________

针对以单输出YOLOv8模型的后处理接口，内部使用线程池的方式实现。

\_\_init\_\_
>>>>>>>>>>>>

**接口形式:**
    .. code-block:: python
          
        def __init__(
                    self,
                    shape: list[int], 
                    network_w:int = 640, 
                    network_h:int = 640, 
                    max_queue_size: int=20,
                    input_use_multiclass_nms: bool=True,
                    agnostic: bool=False)

**参数说明:**

* shape: list[int]

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


push_npy
>>>>>>>>>>

输入数据，只支持batchsize为1的输入，或者输入之前将数据拆分之后再送入接口。

**接口形式:**
    .. code-block:: python

        def push_npy(self, 
                channel_idx: int, 
                image_idx: int, 
                data: numpy.ndarray[Any, numpy.dtype[numpy.float_]], 
                dete_threshold: float, 
                nms_threshold: float,
                ost_w: int, 
                ost_h: int,
                padding_left: int,
                padding_top: int,
                padding_width: int,
                padding_height: int) -> int

**参数说明:**

* channel_idx: int

输入参数。输入图像的通道号。

* image_idx: int

输入参数。输入图像的编号。

* data: numpy.ndarray[Any, numpy.dtype[numpy.float\_]]

输入参数。输入数据。

* dete_threshold: float

输入参数。检测阈值。

* nms_threshold: float

输入参数。nms阈值。

* ost_w: int

输入参数。原始图片的宽。

* ost_h: int

输入参数。 原始图片的高。

* padding_left: int

输入参数。填充图像的起始点坐标x，参数可以通过通用预处理的接口中或者带有预处理的推理接口中获取，也可以自己计算。

* padding_top: int

输入参数。填充图像的起始点坐标y，参数可以通过通用预处理的接口中或者带有预处理的推理接口中获取，也可以自己计算。

* padding_width: int

输入参数。填充图像的宽度，参数可以通过通用预处理的接口中或者带有预处理的推理接口中获取，也可以自己计算。

* padding_height: int

输入参数。填充图像的高度，参数可以通过通用预处理的接口中或者带有预处理的推理接口中获取，也可以自己计算。

**返回值说明:**

成功返回0，其他值表示失败。


push_data
>>>>>>>>>>>>>

输入数据，只支持batchsize为1的输入，或者输入之前将数据拆分之后再送入接口。

**接口形式:**
    .. code-block:: python

        def push_data(self, 
            channel_idx: list[int], 
            image_idx: list[int], 
            input_data: TensorPTRWithName, 
            dete_threshold: list[float],
            nms_threshold: list[float],
            ost_w: list[int],
            ost_h: list[int],
            padding_attrs: list[list[int]]) -> int

**参数说明:**

* channel_idx: int

输入参数。输入图像序列的通道号。

* image_idx: int

输入参数。输入图像序列的编号。

* input_data: TensorPTRWithName

输入参数。输入数据。

* dete_threshold: float

输入参数。检测阈值序列。

* nms_threshold: float

输入参数。nms阈值序列。

* ost_w: int

输入参数。原始图片序列的宽。

* ost_h: int

输入参数。 原始图片序列的高。

* padding_attrs: list[list[int]]

输入参数。填充图像序列的属性列表，填充的起始点坐标x、起始点坐标y、尺度变换之后的宽度、尺度变换之后的高度。

**返回值说明:**

成功返回0，其他值表示失败。

get_result_npy
>>>>>>>>>>>>>>>>>

获取最终的检测结果

**接口形式:**
    .. code-block:: python

        def get_result_npy(self) 
                -> tuple[tuple[int, int, int, int, int, float],int, int]

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

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "../../../sophon-demo/sample/YOLOv8_det/datasets/test/3.jpg"
            bmodel_name = "../../../sophon-demo/sample/YOLOv8_det/models/BM1684X/yolov8s_opt_int8_1b.bmodel"
            decoder = sail.Decoder(image_name,True,tpu_id)
            bmimg = decoder.read(handle)
            engine_image_pre_process = sail.EngineImagePreProcess(bmodel_name, tpu_id, 0)
            engine_image_pre_process.InitImagePreProcess(sail.sail_resize_type.BM_PADDING_TPU_LINEAR, True, 10, 10)
            engine_image_pre_process.SetPaddingAtrr(114,114,114,1)
            alpha_beta = (1.0/255,0),(1.0/255,0),(1.0/255,0)
            engine_image_pre_process.SetConvertAtrr(alpha_beta)
            ret = engine_image_pre_process.PushImage(0,0, bmimg)
            output_tensor_map, ost_images, channels ,imageidxs, paddding_attrs = engine_image_pre_process.GetBatchData(True)
            width_list = []
            height_list= []
            for index, channel in enumerate(channels):
                width_list.append(ost_images[index].width())
                height_list.append(ost_images[index].height())
            yolov8_post = sail.algo_yolov8_post_cpu_opt_1output_async([1, 8400, 84],640,640,10)
            dete_thresholds = np.ones(len(channels),dtype=np.float32)
            nms_thresholds = np.ones(len(channels),dtype=np.float32)
            dete_thresholds = 0.2*dete_thresholds
            nms_thresholds = 0.5*nms_thresholds
            ret = yolov8_post.push_data(channels, imageidxs, output_tensor_map[0], dete_thresholds, nms_thresholds, width_list, height_list, paddding_attrs)
            # 以下是利用push_npy接口推送 numpy 数据的示例
            # for index, channel in enumerate(channels):
            #     ret = yolov8_post.push_npy(channel, index, output_tensor_map[index].get_data().asnumpy(), 0.2, 0.5, 
            #             ost_images[index].width(), ost_images[index].height(), 
            #             paddding_attrs[index][0], paddding_attrs[index][1], paddding_attrs[index][2], paddding_attrs[index][3])
            objs, channel, image_idx = yolov8_post.get_result_npy()
            print(objs, channel, image_idx)