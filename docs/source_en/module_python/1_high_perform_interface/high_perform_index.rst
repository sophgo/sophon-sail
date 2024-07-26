
SAIL Python high performance interface
----------------------------------------


.. toctree::
   :glob:
   :maxdepth: 1

   sail.TensorPTRWithName
   sail.ImagePreProcess
   sail.EngineImagePreProcess


Yolov5 post-processing acceleration interfaces
=====================================================
Post-processing acceleration interfaces for single-output and three-output YOLOv5 models.

.. toctree::
   :glob:
   :maxdepth: 1

   sail.algo_yolov5_post_1output
   sail.algo_yolov5_post_3output
   sail.algo_yolov5_post_cpu_opt
   sail.algo_yolov5_post_cpu_opt_async
   sail.tpu_kernel_api_yolov5_detect_out
   sail.tpu_kernel_api_yolov5_out_without_decode


Yolox post-processing acceleration interfaces
=====================================================

.. toctree::
   :glob:
   :maxdepth: 1

   sail.algo_yolox_post
   
   
Yolov8 post-processing acceleration interfaces
=====================================================

.. toctree::
   :glob:
   :maxdepth: 1

   sail.algo_yolov8_post_1output_async
   sail.algo_yolov8_post_cpu_opt_1output_async

sort 
==========================

.. toctree::
   :glob:
   :maxdepth: 1

   sail.algo_sort_tracker_controller
   sail.algo_sort_tracker_controller_async


deepsort
===================

.. toctree::
   :glob:
   :maxdepth: 1

   sail.deepsort_tracker_controller
   sail.deepsort_tracker_controller_async


bytetrack 
==========================

.. toctree::
   :glob:
   :maxdepth: 1

   sail.bytetrack_tracker_controller

openpose 
======================
Use tpukernel to speed up openpose post-processing of part nms.

.. toctree::
   :glob:
   :maxdepth: 1

   sail.tpu_kernel_api_openpose_part_nms

nms_rotated
==============

.. toctree::
   :glob:
   :maxdepth: 1

   sail.nms_rotated