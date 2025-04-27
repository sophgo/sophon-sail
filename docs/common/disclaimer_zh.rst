声明
-------------

.. figure:: ../common/images/logo.png
   :width: 400px
   :height: 400px
   :scale: 50%
   :align: center
   :alt: SOPHGO LOGO

| **法律声明**
| 版权所有 © 算能 2022. 保留一切权利。
| 非经本公司书面许可，任何单位和个人不得擅自摘抄、复制本文档内容的部分或全部，并不得以任何形式传播。

| **注意**
| 您购买的产品、服务或特性等应受 算能 商业合同和条款的约束，
  本文档中描述的全部或部分产品、服务或特性可能不在您的购买或使用范围之内。
  除非合同另有约定， 算能 对本文档内容不做任何明示或默示的声明或保证。
  由于产品版本升级或其他原因，本文档内容会不定期进行更新。
  除非另有约定，本文档仅作为使用指导，本文档中的所有陈述、信息和建议不构成任何明示或暗示的担保。

| **技术支持**

:地址: 北京市海淀区丰豪东路9号院中关村集成电路设计园（ICPARK）1号楼
:邮编: 100094
:网址: https://www.sophgo.com/
:邮箱: sales@sophgo.com
:电话: +86-10-57590723
       +86-10-57590724



| **SAIL发布记录**

.. table::
   :width: 600
   :widths: 20 30 50

   ========== ========== ===================
      版本     发布日期    说明
   ========== ========== ===================
   V2.0.0     2019.09.20  第一次发布。
   ---------- ---------- -------------------
   V2.0.1     2019.11.16  V2.0.1版本发布。
   ---------- ---------- -------------------
   V2.0.3     2020.05.07  V2.0.3版本发布。
   ---------- ---------- -------------------
   V2.2.0     2020.10.12  V2.2.0版本发布。
   ---------- ---------- -------------------
   V2.3.0     2021.01.11  V2.3.0版本发布。
   ---------- ---------- -------------------
   V2.3.1     2021.03.09  V2.3.1版本发布。
   ---------- ---------- -------------------
   V2.3.2     2021.04.01  V2.3.2版本发布。
   ---------- ---------- -------------------
   V2.4.0     2021.05.23  V2.4.0版本发布。
   ---------- ---------- -------------------
   V2.5.0     2021.09.02  V2.5.0版本发布。
   ---------- ---------- -------------------
   V2.6.0     2022.01.30  V2.6.0版本修正后发布。
   ---------- ---------- -------------------
   V2.7.0     2022.03.16  V2.7.0版本发布, 20220531发布补丁版本。
   ---------- ---------- -------------------
   V3.0.0     2022.07.16  V3.0.0版本发布。
   ---------- ---------- -------------------
   V3.1.0     2022.11.01  V3.1.0版本发布。
   ---------- ---------- -------------------
   V3.2.0     2022.12.01  V3.2.0版本发布。
   ---------- ---------- -------------------
   V3.3.0     2023.01.01  V3.3.0版本发布。
   ---------- ---------- -------------------
   V3.4.0     2023.03.01  V3.4.0版本发布。
   ---------- ---------- -------------------
   V3.5.0     2023.05.01  V3.5.0版本发布。
   ---------- ---------- -------------------
   V3.6.0     2023.07.01  V3.6.0版本发布。
   ---------- ---------- -------------------
   V3.7.0     2023.10.01  V3.7.0版本发布。
   ---------- ---------- -------------------
   V3.8.0     2024.04.10  V3.8.0版本发布。
   ---------- ---------- -------------------
   V3.9.0     2024.09.14  V3.9.0版本发布。
   ---------- ---------- -------------------
   V3.10.0    2024.03.11  V3.10.0版本发布。
   ========== ========== ===================

| **V3.10.0 更新内容**

* Tensor添加查询单个元素占用的字节数接口element_size

* Tensor添加查询所有元素占用的总字节数接口nbytes

* BMImag添加获取裸数据接口asnumpy

* BMImage添加获取解码时间戳接口get_pts_dts

* Bmcv添加向量查询接口：faiss_indexflatL2 faiss_indexflatIP faiss_indexPQ_encode faiss_indexPQ_ADC faiss_indexPQ_SDC

* Bmcv添加傅里叶变换接口：stft istft

* Bmcv添加水印图片叠加接口bmcv_overlay

* yolov5后处理接口push_data添加对多类别置信度的支持

* 添加yolov8 seg后处理接口algo_yolov8_seg_post_tpu_opt

* EngineLLM添加支持bmrt flag的构造函数

* 优化Encoder推流兼容性

* 添加对Python 3.12版本的适配

| **V3.9.0 更新内容**

* BMImage添加基于bytes、numpy.ndarray等裸数据的构造接口

* 添加支持大模型推理的EngineLLM

* 添加图片拼接Blend模块

* Bmcv添加图片解码imread接口

* MultiDecoder添加查询状态接口get_channel_status

* Tensor添加获取设备号的接口device_id

* Tensor添加支持设置stride的d2d接口sync_d2d_stride

| **V3.8.0 更新内容**

* 添加获取底板温度接口get_board_temp

* 添加获取处理器温度接口get_chip_temp

* 添加获取tpu利用率接口get_tpu_util

* 添加获取vpu利用率接口get_vpu_util

* 添加获取vpp利用率接口get_vpp_util

* 添加获取处理器内存使用率接口get_dev_stat

* 添加获取日志等级接口set_loglevel

* Tensor添加d2d接口

* Tensor添加支持偏移量的d2s、s2接口

* Tensor添加对FLOAT16的支持

* Bmcv添加fillRectangle、putText、imencode、open_water、mosaic、gaussian_blur、transpose、Sobel等接口的支持

* 添加yolov8后处理异步接口algo_yolov8_post_1output_async

* 添加yolov8后处理CPU加速异步接口algo_yolov8_post_cpu_opt_1output_async

* 添加yolov5后处理CPU加速异步接口algo_yolov5_post_cpu_opt_async

* 添加deepsort同步处理接口deepsort_tracker_controller

* 添加deepsort异步处理接口deepsort_tracker_controller_async

* 添加bytetrack同步处理接口bytetrack_tracker_controller

* 添加sort同步处理接口sort_tracker_controller

* 添加sort异步处理接口sort_tracker_controller_async

* 添加使用tpu加速openpose后处理接口tpu_kernel_api_openpose_part_nms

* 添加对Windows的支持

* 添加对RISCV的支持

| **V3.7.0 更新内容**

* 添加yolox后处理异步接口algo_yolox_post

* 添加使用h264和h265解码裸流接口

* Tensor添加dump数据接口

* 添加使用yolov5后处理优化接口algo_yolov5_post_cpu_opt 

* 添加绘制多边形框功能接口

* 添加cv::Mat转换到BMImage的接口

* 添加对Python3.9，Python3.10，Python3.11版本的适配


| **V3.6.0 更新内容**
 
* Decoder添加保存视频接口dump。

* 基于BM1684X添加对单输出的yolov5模型后处理使用智能视觉深度学习处理器进行加速的接口tpu_kernel_api_yolov5_out_without_decode。

* 添加deepsort跟踪接口: deepsort_tracker_controller。

* 添加bytetrack跟踪接口: bytetrack_tracker_controller。

* convert_format接口添加可以指定图像格式。

* Bmcv添加convert_yuv420p_to_gray接口。

| **V3.5.0 更新内容**

* 添加视频及图片编码接口Encoder。

* Handle添加获取设备型号的接口get_target。

* 基于BM1684X添加对三输出的yolov5模型后处理使用智能视觉深度学习处理器进行加速的接口tpu_kernel_api_yolov5_detect_out。

* 添加了调用多线程推理框架的Python测试例程。

