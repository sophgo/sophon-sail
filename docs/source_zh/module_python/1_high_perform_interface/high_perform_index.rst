
SAIL Python 高性能接口
------------------------


.. toctree::
   :glob:
   :maxdepth: 1

   sail.TensorPTRWithName
   sail.ImagePreProcess
   sail.EngineImagePreProcess


Yolov5 后处理加速接口
=========================
Yolov5 针对单输出、三输出的YOLOv5模型的后处理加速接口

.. toctree::
   :glob:
   :maxdepth: 1

   sail.algo_yolov5_post_1output
   sail.algo_yolov5_post_3output
   sail.algo_yolov5_post_cpu_opt
   sail.algo_yolov5_post_cpu_opt_async
   sail.tpu_kernel_api_yolov5_detect_out
   sail.tpu_kernel_api_yolov5_out_without_decode


Yolox后处理加速接口
==========================

.. toctree::
   :glob:
   :maxdepth: 1

   sail.algo_yolox_post
   
   
Yolov8后处理加速接口
==========================

.. toctree::
   :glob:
   :maxdepth: 1

   sail.algo_yolov8_post_1output_async
   sail.algo_yolov8_post_cpu_opt_1output_async

sort 算法
==========================

.. toctree::
   :glob:
   :maxdepth: 1

   sail.algo_sort_tracker_controller
   sail.algo_sort_tracker_controller_async


deepsort 算法
===================

.. toctree::
   :glob:
   :maxdepth: 1

   sail.deepsort_tracker_controller
   sail.deepsort_tracker_controller_async


bytetrack 算法
==========================

.. toctree::
   :glob:
   :maxdepth: 1

   sail.bytetrack_tracker_controller

openpose 算法
======================
openpose  使用tpukernel对part nms后处理加速

.. toctree::
   :glob:
   :maxdepth: 1

   sail.tpu_kernel_api_openpose_part_nms

nms_rotated 接口
========================

.. toctree::
   :glob:
   :maxdepth: 1
   
   sail.nms_rotated


Yolov8_seg TPU后处理加速接口
=================================

.. toctree::
   :glob:
   :maxdepth: 1

   sail.algo_yolov8_seg_post_tpu_opt