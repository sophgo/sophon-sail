sail.tpu_kernel_api_yolov5_detect_out
____________________________________________

针对3输出的yolov5模型，使用智能视觉深度学习处理器Kernel对后处理进行加速，目前只支持BM1684x，且libsophon的版本必须不低于0.4.6（v23.03.01）。

\_\_init\_\_
>>>>>>>>>>>>

**接口形式:**
    .. code-block:: python
          
        def __init__(
                    self,
                    device_id: int,
                    shape: list[list[int]], 
                    network_w: int = 640, 
                    network_h: int = 640, 
                    module_file: str="/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so")

**参数说明:**

* device_id: int

输入参数。使用的设备编号。

* shape: list[list[int]]

输入参数。输入数据的shape。

* network_w: int

输入参数。模型的输入宽度，默认为640。

* network_h: int

输入参数。模型的输入宽度，默认为640。

* module_file: str

输入参数。Kernel module文件路径，默认为"/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so"。


process
>>>>>>>>>>>>>

处理接口。

**接口形式1:**
    .. code-block:: python

        def process(self, 
            input_data: list[TensorPTRWithName], 
            dete_threshold: float,
            nms_threshold: float,
            release_input: bool = False) 
                -> list[list[tuple[int, int, int, int, int, float]]]

**参数说明1:**

* input_data: list[TensorPTRWithName]

输入参数。输入数据，包含三个输出。

* dete_threshold: float

输入参数。检测阈值。

* nms_threshold: float

输入参数。nms阈值序列。

* release_input: bool

输入参数。释放输入的内存，默认为False。

**接口形式2:**
    .. code-block:: python

        def process(self, 
            input_data: dict[str, Tensor], 
            dete_threshold: float,
            nms_threshold: float,
            release_input: bool = False) 
                -> list[list[tuple[int, int, int, int, int, float]]]

**参数说明2:**

* input_data: dict[str, Tensor]

输入参数。输入数据，包含三个输出。

* dete_threshold: float

输入参数。检测阈值。

* nms_threshold: float

输入参数。nms阈值序列。

* release_input: bool

输入参数。释放输入的内存，默认为False。

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
        
        def get_ratio(bmimg):
            img_w = bmimg.width()
            img_h = bmimg.height()
            r_w = 640 / img_w
            r_h = 640 / img_h
            if r_h > r_w:
                tw = 640
                th = int(r_w * img_h)
                tx1 = tx2 = 0
                ty1 = int((640 - th) / 2)
                ty2 = 640 - th - ty1
            else:
                tw = int(r_h * img_w)
                th = 640
                tx1 = int((640 - tw) / 2)
                tx2 = 640 - tw - tx1
                ty1 = ty2 = 0

            ratio = (min(r_w, r_h), min(r_w, r_h))
            txy = (tx1, ty1)
            return (img_w, img_h), ratio, txy

        if __name__ == '__main__':
            tpu_id = 0
            image_path = '../../../sophon-demo/sample/YOLOv5/datasets/test/3.jpg'
            decoder = sail.Decoder(image_path, True, tpu_id)
            bmodel_path = '../../../sophon-demo/sample/YOLOv5/models/BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel'
            handle = sail.Handle(tpu_id)
            alpha_beta = (1.0/255,0),(1.0/255,0),(1.0/255,0)

            resize_type = sail.sail_resize_type.BM_PADDING_TPU_LINEAR
            sail_engineipp = sail.EngineImagePreProcess(bmodel_path, tpu_id, False)
            ret0 = sail_engineipp.InitImagePreProcess(resize_type, True, 10, 10)

            sail_engineipp.SetPaddingAtrr(114, 114, 114, 1)
            ret1 = sail_engineipp.SetConvertAtrr(alpha_beta)

            bm_i = sail.BMImage()
            
            decoder.read(handle, bm_i)
            decoder.release()
            hw, ratio, txy = get_ratio(bm_i)
            ret3 = sail_engineipp.PushImage(0, 0, bm_i)

            res = sail_engineipp.GetBatchData(True)
            output_tensor_map, ost_images, channel_list ,imageidx_list, padding_atrr = res
            tpu_kernel_3o = sail.tpu_kernel_api_yolov5_detect_out(0, [[1, 255, 80, 80], [1, 255, 40, 40], [1, 255, 20, 20]], 640, 640, "/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so")
            
            res_list = tpu_kernel_3o.process(output_tensor_map, 0.5, 0.5)

            result = []
            for i in range(len(res_list)):
                if len(res_list[i]) > 0:
                    result.append(np.array(res_list[i]))
                else:
                    result.append(np.empty((0,6)))

            for res in result:
                if len(res):
                    coords = res[:, :4]
                    
                    coords[:, [0, 2]] -= txy[0]
                    coords[:, [1, 3]] -= txy[1]
                    coords[:, [0, 2]] /= ratio[0]
                    coords[:, [1, 3]] /= ratio[1]

                    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, hw[0] - 1)
                    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, hw[1] - 1)
                    res[:, :4] = coords.round()
            print(result)