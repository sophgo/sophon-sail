sail.algo_yolov8_seg_post_tpu_opt
____________________________________________

针对yolov8_seg模型，使用TPU对后处理进行了加速。

\_\_init\_\_
>>>>>>>>>>>>

**接口形式:**
    .. code-block:: python
          
        def __init__(
                    self,
                    bmodel_file: str,
                    dev_id: int,
                    detection_shape: list[int],
                    segmentation_shape: list[int],
                    network_h: int,
                    network_w: int)

**参数说明:**

* bmodel_file: str

输入参数。TPU后处理所使用的getmask bmodel的路径。

* dev_id: int

输入参数。智能视觉深度学习处理器的id号。

* detection_shape: list[int]

输入参数。yolov8_seg分割模型检测头的输出shape。

* segmentation_shape: list[int]

输入参数。yolov8_seg分割模型分割头的输出shape，即Prototype Mask的shape。

* network_h: int

输入参数。yolov8_seg分割模型的输入高度。

* network_w: int

输入参数。yolov8_seg分割模型的输入宽度。


process
>>>>>>>>>>>>>

处理接口。

**接口形式1:**
    .. code-block:: python

        def process(self, 
            detection_input: TensorPTRWithName, 
            segmentation_input: TensorPTRWithName, 
            ost_h: int, 
            ost_w: int,
            dete_threshold: float,
            nms_threshold: float,
            input_keep_aspect_ratio: bool,
            input_use_multiclass_nms: bool) 
                -> list[tuple[left, top, right, bottom, score, class_id, contour, mask]]

**参数说明1:**

* detection_input: TensorPTRWithName

输入参数。yolov8_seg分割模型检测头的输出数据。

* segmentation_input: TensorPTRWithName

输入参数。yolov8_seg分割模型分割头的输出数据，即Prototype Mask。

* ost_h: int

输入参数。原始图片的高度。

* ost_w: int

输入参数。原始图片的宽度。

* dete_threshold: float

输入参数。检测阈值。

* nms_threshold: float

输入参数。nms阈值。

* input_keep_aspect_ratio: bool

输入参数。输入图片是否保持纵横比。

* input_use_multiclass_nms: bool

输入参数。是否使用多类nms。

**接口形式2:**
    .. code-block:: python

        def process(self, 
            detection_input: dict[str, Tensor], 
            segmentation_input: dict[str, Tensor], 
            ost_h: int, 
            ost_w: int,
            dete_threshold: float,
            nms_threshold: float,
            input_keep_aspect_ratio: bool,
            input_use_multiclass_nms: bool) 
                -> list[tuple[left, top, right, bottom, score, class_id, contour, mask]]

**参数说明2:**

* detection_input: dict[str, Tensor]

输入参数。yolov8_seg分割模型检测头的输出数据。

* segmentation_input: dict[str, Tensor]

输入参数。yolov8_seg分割模型分割头的输出数据，即Prototype Mask。

* ost_h: int

输入参数。原始图片的高度。

* ost_w: int

输入参数。原始图片的宽度。

* dete_threshold: float

输入参数。检测阈值。

* nms_threshold: float

输入参数。nms阈值。

* input_keep_aspect_ratio: bool

输入参数。输入图片是否保持纵横比。

* input_use_multiclass_nms: bool

输入参数。是否使用多类nms。

**返回值说明:**

list[tuple[left, top, right, bottom, score, class_id, contour, mask]]

* left: int 

检测框的最左x坐标。

* top: int

检测框的最上y坐标。

* right: int

检测框的最右x坐标。

* bottom: int

检测框的最下y坐标。

* class_id: int

检测框内物体的类别编号。

* score: float

检测框内物体的分数。

* contour: list[float]

检测框内物体的轮廓。

* mask: numpy.ndarray

检测框内物体的分割掩码。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            
            image_name = "../../..//sophon-demo/sample/YOLOv8_seg/datasets/test/dog.jpg"
            yolov8_seg_bmodel_name = "../../..//sophon-demo/sample/YOLOv8_seg/models/BM1688/yolov8s_int8_1b.bmodel"
            get_mask_bmodel_name = "../../..//sophon-demo/sample/YOLOv8_seg/yolov8s_getmask_32_fp32.bmodel"
            
            decoder = sail.Decoder(image_name, True, tpu_id)
            bmimg = decoder.read(handle)
            
            engine_image_pre_process = sail.EngineImagePreProcess(yolov8_seg_bmodel_name, tpu_id, 0)
            engine_image_pre_process.InitImagePreProcess(sail.sail_resize_type.BM_PADDING_TPU_LINEAR, True, 10, 10)
            engine_image_pre_process.SetPaddingAtrr(114, 114, 114, 1)
            alpha_beta = (1.0/255, 0), (1.0/255, 0), (1.0/255, 0)
            engine_image_pre_process.SetConvertAtrr(alpha_beta)
            ret = engine_image_pre_process.PushImage(0, 0, bmimg)
            
            output_tensor_map, ost_images, channels ,imageidxs, padding_atrr = engine_image_pre_process.GetBatchData(True)
            
            width_list = []
            height_list= []
            for index, channel in enumerate(channels):
                width_list.append(ost_images[index].width())
                height_list.append(ost_images[index].height())
            
            yolov8_tpu_post = sail.algo_yolov8_seg_post_tpu_opt(get_mask_bmodel_name, tpu_id, [1, 116, 8400], [1, 32, 160, 160], 640, 640)
            
            dete_threshold = 0.25
            nms_threshold = 0.7
            
            results = yolov8_tpu_post.process(output_tensor_map[0], output_tensor_map[1], height_list[0], width_list[0], dete_threshold, nms_threshold, True, False)
            
            print(results)