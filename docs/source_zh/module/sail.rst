SAIL
====

SAIL (SOPHON Artificial Intelligent Library)是 SOPHON-SAIL 中的核心模块。
SAIL对 SOPHONSDK 中的 BMLib、sophon-mw(对于BM1688或者CV186AH处理器,名称更改为sophon-media)、BMCV、BMRuntime 进行了封装，
将 SOPHONSDK 中原有的 “加载 bmodel 并驱动智能视觉深度学习处理器推理”、“驱动智能视觉深度学习处理器做图像处理”、“驱动 VPU 做图像和视频解码”等功能抽象成更为简单的 C++ 接口对外提供；
并且使用 pybind11 再次封装，提供简洁易用的 python 接口。

目前，SAIL 模块中所有的类、枚举、函数都在 “sail” 命名空间下，
本单元中的文档将向您深入介绍可能用到的 SAIL 中的模块和类。
核心的类包括：

* Handle：

SDK 中 BMLib 的 bm_handle_t 的包装类，设备句柄，上下文信息，用来和内核驱动交互信息。

* Tensor：

SDK 中 BMLib 的包装类，封装了对 device memory 的管理以及与 system memory 的同步。

* Engine：

SDK 中 BMRuntime 的包装类，可以加载 bmodel 并驱动智能视觉深度学习处理器进行推理。
一个 Engine 实例可以加载一个任意的 bmodel，
自动地管理输入张量与输出张量对应的内存。

* Decoder：

使用VPU解码视频，JPU解码图像，均为硬件解码。

* Encoder：

使用VPU编码视频，JPU和软件编码图像。视频编码支持h264，h265的本地视频和rtsp/rtmp流；像素格式支持NV12和I420。jpeg硬件编码和其他图片格式的软件编码。

* Bmcv：

SDK 中 BMCV 的包装类，封装了一系列的图像处理函数，可以驱动智能视觉深度学习处理器进行图像处理。

