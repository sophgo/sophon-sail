sail.algo_yolov5_post_3output
_________________________________

针对以三输出YOLOv5模型的后处理接口，内部使用线程池的方式实现。

\_\_init\_\_
>>>>>>>>>>>>

**接口形式:**
    .. code-block:: python
          
        def __init__(
                    self,
                    shape: list[list[int]], 
                    network_w:int = 640, 
                    network_h:int = 640, 
                    max_queue_size: int=20,
                    input_use_multiclass_nms: bool=True,
                    agnostic: bool=False)

**参数说明:**

* shape: list[list[int]]

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


push_data
>>>>>>>>>>>>>

输入数据，支持任意batchsize的输入。

**接口形式:**
    .. code-block:: python

        def push_data(self, 
            channel_idx: list[int], 
            image_idx: list[int], 
            input_data: list[TensorPTRWithName], 
            dete_threshold: list[float],
            nms_threshold: list[float],
            ost_w: list[int],
            ost_h: list[int],
            padding_attrs: list[list[int]]) -> int

**参数说明:**

* channel_idx: list[int]

输入参数。输入图像序列的通道号。

* image_idx: list[int]

输入参数。输入图像序列的编号。

* input_data: list[TensorPTRWithName],

输入参数。输入数据，包含三个输出。

* dete_threshold: list[float]

输入参数。检测阈值序列。

* nms_threshold: list[float]

输入参数。nms阈值序列。

* ost_w: list[int]

输入参数。原始图片序列的宽。

* ost_h: list[int]

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

reset_anchors
>>>>>>>>>>>>>

更新anchor尺寸.

**接口形式:**
    .. code-block:: python

        def reset_anchors(self, anchors_new: list[list[list[int]]]) -> int

**参数说明:**

* anchors_new: list[list[list[int]]]

要更新的anchor尺寸列表.

**返回值说明:**

成功返回0，其他值表示失败。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np
        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "../../../sophon-demo/sample/YOLOv5/datasets/test/3.jpg"
            bmodel_name = "../../../sophon-demo/sample/YOLOv5/models/BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel"
            decoder = sail.Decoder(image_name,True,tpu_id)
            bmimg = decoder.read(handle)
            engine_image_pre_process = sail.EngineImagePreProcess(bmodel_name, tpu_id, 0)
            engine_image_pre_process.InitImagePreProcess(sail.sail_resize_type.BM_PADDING_TPU_LINEAR, True, 10, 10)
            engine_image_pre_process.SetPaddingAtrr(114,114,114,1)
            alpha_beta = (1.0/255,0),(1.0/255,0),(1.0/255,0)
            engine_image_pre_process.SetConvertAtrr(alpha_beta)
            ret = engine_image_pre_process.PushImage(0,0, bmimg)
            engine_image_pre_process
            output_tensor_map, ost_images, channels ,imageidxs, padding_atrr = engine_image_pre_process.GetBatchData(True)
            width_list = []
            height_list= []
            for index, channel in enumerate(channels):
                width_list.append(ost_images[index].width())
                height_list.append(ost_images[index].height())
            yolov5_post = sail.algo_yolov5_post_3output([[1, 3, 20, 20, 85],[1, 3, 40, 40, 85],[1, 3, 80, 80, 85]],640,640,10)
            dete_thresholds = np.ones(len(channels),dtype=np.float32)
            nms_thresholds = np.ones(len(channels),dtype=np.float32)
            dete_thresholds = 0.2*dete_thresholds
            nms_thresholds = 0.5*nms_thresholds
            ret = yolov5_post.push_data(channels, imageidxs, output_tensor_map, dete_thresholds, nms_thresholds, width_list, height_list, padding_atrr)
            objs, channel, image_idx = yolov5_post.get_result_npy()
            print(objs, channel, image_idx)