sail.tpu_kernel_api_openpose_part_nms
____________________________________________

使用智能视觉深度学习处理器 Kernel对part nms后处理进行加速，目前只支持BM1684x，且libsophon的版本必须不低于0.4.6（v23.03.01）。

\_\_init\_\_
>>>>>>>>>>>>

**接口形式:**
    .. code-block:: python
          
        def __init__(
                    self,
                    device_id: int,
                    network_c: int, 
                    module_file: str="/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so")

**参数说明:**

* device_id: int

输入参数。使用的设备编号。

* network_c: int

输入参数。输入通道数，对应关键点通道的数量。

* module_file: str

输入参数。Kernel module文件路径，默认为"/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so"。


process
>>>>>>>>>>>>>

处理接口。

**接口形式1:**
    .. code-block:: python

        def process(self, 
            input_data: TensorPTRWithName, 
            shape: list[int],
            threshold: list[float],
            max_peak_num: list[int]) 
                -> tuple[list[list[int]], list[list[float]], list[list[int]]]

**参数说明1:**

* input_data: TensorPTRWithName

输入参数。输入数据。

* shape: list[int]

输入参数。输入数据的宽和高。

* threshold: list[float]

输入参数。检测阈值序列。

* max_peak_num: list[int]

输入参数。最大被检测关键点的数量。

**接口形式2:**
    .. code-block:: python

        def process(self, 
            input_data: dict[str, Tensor], 
            shape: list[int],
            threshold: list[float],
            max_peak_num: list[int]) 
                -> tuple[list[list[int]], list[list[float]], list[list[int]]]

**参数说明2:**

* input_data: dict[str, Tensor]

输入参数。输入数据。

* shape: list[int]

输入参数。输入数据的宽和高。

* threshold: list[float]

输入参数。检测阈值序列。

* max_peak_num: list[int]

输入参数。最大被检测关键点的数量。

**返回值说明:**

tuple[list[list[int]], list[list[float]], list[list[int]]]

* 第一个输出: list[list[int]] 

在每个通道被检测的关键点数量。

* 第二个输出: list[list[float]]

所有被检测的关键点的置信度。

* 第三个输出: list[list[int]]

所有被检测的关键点的拉平坐标。


reset_network_c
>>>>>>>>>>>>>>>>>>

更新关键点通道数。

**接口形式:**
    .. code-block:: python

        def reset_network_c(self, network_c_new: int) -> int

**参数说明:**

* network_c_new: int

要更新的通道数。

**返回值说明:**

成功返回0，其他值表示失败。

**示例代码:**
    .. code-block:: python

        from scipy.ndimage import gaussian_filter
        import sophon.sail as sail
        import numpy as np
        import cv2

        if __name__ == '__main__':
            tpu_id = 0
            image_path = '../../../sophon-demo/sample/OpenPose/datasets/test/3.jpg'
            bmodel_path = '../../../sophon-demo/sample/OpenPose/models/BM1684x/pose_coco_fp32_1b.bmodel'
            handle = sail.Handle(tpu_id)
            net = sail.Engine(bmodel_path, tpu_id, sail.IOMode.SYSIO)
            src_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            graph_name = net.get_graph_names()[0]
            input_name = net.get_input_names(graph_name)[0]
            output_name = net.get_output_names(graph_name)[0]
            h, w, _ = src_img.shape
            net_h = net.get_input_shape(graph_name, input_name)[2]
            net_w = net.get_input_shape(graph_name, input_name)[3]
            out_h = net.get_output_shape(graph_name, output_name)[2]
            scale = min(net_h / h, net_w / w)

            resize_img = cv2.resize(src_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            pad_img = cv2.copyMakeBorder(resize_img,0,net_h - resize_img.shape[0],0,net_w - resize_img.shape[1],cv2.BORDER_CONSTANT,value=(114,114,114))
            img = np.transpose((pad_img.astype('float32')-128)/255, (2, 0, 1))
            img = np.stack([img])
            outputs = net.process(graph_name, {input_name: img})

            # output = np.transpose(list(outputs.values())[0], (1, 2, 0))
            output_array = list(outputs.values())[0]
            output = output_array[0]
            output = np.transpose(output, (1, 2, 0))
            
            stride = net_h / out_h
            output = cv2.resize(output, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            output = output[:resize_img.shape[0], :resize_img.shape[1], :]
            output = cv2.resize(output, (src_img.shape[1], src_img.shape[0]), interpolation=cv2.INTER_CUBIC)
            part_nms_input = np.array([gaussian_filter(output[:, :, j], sigma=3) for j in range(output.shape[-1])])
            point_num = int(net.get_output_shape(graph_name, output_name)[1] / 3) - 1
            part_nms_input = {"input1": sail.Tensor(handle, part_nms_input[:point_num][None])}
            part_nms_input['input1'].sync_s2d()
            tka_nms = sail.tpu_kernel_api_openpose_part_nms(tpu_id, point_num, "/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so")
            num_result, socre_result, coor_result = tka_nms.process(part_nms_input, [src_img.shape[1], src_img.shape[0]], [0.05], [96])
            print(num_result, socre_result, coor_result)
            