sail.algo_yolov5_post_cpu_opt
____________________________________________

针对3输出或1输出的yolov5模型，对后处理进行了加速。

\_\_init\_\_
>>>>>>>>>>>>

**接口形式:**
    .. code-block:: python
          
        def __init__(
                    self,
                    shapes: list[list[int]], 
                    network_w: int = 640, 
                    network_h: int = 640)

**参数说明:**

* shapes: list[list[int]]

输入参数。输入数据的shape。

* network_w: int

输入参数。模型的输入宽度，默认为640。

* network_h: int

输入参数。模型的输入高度，默认为640。


process
>>>>>>>>>>>>>

处理接口。

**接口形式1:**
    .. code-block:: python

        def process(self, 
            input_data: list[TensorPTRWithName], 
            ost_w: list[int], 
            ost_h: list[int], 
            dete_threshold: list[float], 
            nms_threshold: list[float], 
            input_keep_aspect_ratio: bool, 
            input_use_multiclass_nms: bool) 
                -> list[list[tuple[int, int, int, int, int, float]]]

**参数说明1:**

* input_data: list[TensorPTRWithName]

输入参数。输入数据，包含三个输出或一个输出。

* ost_w: list[int]

输入参数。原始图片的宽度。

* ost_h: list[int]

输入参数。原始图片的高度。

* dete_threshold: list[float]

输入参数。检测阈值。

* nms_threshold: float

输入参数。nms阈值序列。

* input_keep_aspect_ratio: bool

输入参数。输入图片是否保持纵横比。

* input_use_multiclass_nms: bool

输入参数。是否用多类nms。

**接口形式2:**
    .. code-block:: python

        def process(self, 
            input_data: dict[str, Tensor], 
            ost_w: list[int], 
            ost_h: list[int], 
            dete_threshold: list[float], 
            nms_threshold: list[float],
            input_keep_aspect_ratio: bool, 
            input_use_multiclass_nms: bool) 
                -> list[list[tuple[int, int, int, int, int, float]]]

**参数说明2:**

* input_data: dict[str, Tensor]

输入参数。输入数据，包含三个输出或一个输出。

* ost_w: list[int]

输入参数。原始图片的宽度。

* ost_h: list[int]

输入参数。原始图片的高度。

* dete_threshold: list[float]

输入参数。检测阈值。

* nms_threshold: float

输入参数。nms阈值序列。

* input_keep_aspect_ratio: bool

输入参数。输入图片是否保持纵横比。

* input_use_multiclass_nms: bool

输入参数。是否用多类nms。

**返回值说明:**

list[list[tuple[left, top, right, bottom, class_id, score]]]

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
            output_tensor_map, ost_images, channels ,imageidxs, padding_atrr = engine_image_pre_process.GetBatchData(True)
            width_list = []
            height_list= []
            for index, channel in enumerate(channels):
                width_list.append(ost_images[index].width())
                height_list.append(ost_images[index].height())
            yolov5_post = sail.algo_yolov5_post_cpu_opt([[1, 3, 20, 20, 85],[1, 3, 40, 40, 85],[1, 3, 80, 80, 85]],640,640)
            dete_thresholds = np.ones(len(channels),dtype=np.float32)
            nms_thresholds = np.ones(len(channels),dtype=np.float32)
            dete_thresholds = 0.2*dete_thresholds
            nms_thresholds = 0.5*nms_thresholds
            objs = yolov5_post.process(output_tensor_map, width_list, height_list, dete_thresholds, nms_thresholds, True, True)
            print(objs)