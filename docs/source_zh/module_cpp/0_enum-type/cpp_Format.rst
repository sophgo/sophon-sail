Format
______________

定义常用的图像格式。

**接口形式:**
    .. code-block:: c

        FORMAT_YUV420P
        FORMAT_YUV422P
        FORMAT_YUV444P
        FORMAT_NV12
        FORMAT_NV21
        FORMAT_NV16
        FORMAT_NV61
        FORMAT_NV24
        FORMAT_RGB_PLANAR
        FORMAT_BGR_PLANAR
        FORMAT_RGB_PACKED
        FORMAT_BGR_PACKED
        FORMAT_RGBP_SEPARATE
        FORMAT_BGRP_SEPARATE
        FORMAT_GRAY
        FORMAT_COMPRESSED

**参数说明:**

* FORMAT_YUV420P

表示预创建一个 YUV420 格式的图片，有三个 plane

* FORMAT_YUV422P

表示预创建一个 YUV422 格式的图片，有三个 plane

* FORMAT_YUV444P

表示预创建一个 YUV444 格式的图片，有三个 plane

* FORMAT_NV12

表示预创建一个 NV12 格式的图片，有两个 plane

* FORMAT_NV21

表示预创建一个 NV21 格式的图片，有两个 plane

* FORMAT_NV16

表示预创建一个 NV16 格式的图片，有两个 plane

* FORMAT_NV61

表示预创建一个 NV61 格式的图片，有两个 plane

* FORMAT_RGB_PLANAR

表示预创建一个 RGB 格式的图片，RGB 分开排列,有一个 plane

* FORMAT_BGR_PLANAR

表示预创建一个 BGR 格式的图片，BGR 分开排列,有一个 plane

* FORMAT_RGB_PACKED

表示预创建一个 RGB 格式的图片，RGB 交错排列,有一个 plane

* FORMAT_BGR_PACKED

表示预创建一个 BGR 格式的图片，BGR 交错排列,有一个 plane

* FORMAT_RGBP_SEPARATE

表示预创建一个 RGB planar 格式的图片，RGB 分开排列并各占一个 plane，共有 3 个 plane

* FORMAT_BGRP_SEPARATE

表示预创建一个 BGR planar 格式的图片，BGR 分开排列并各占一个 plane，共有 3 个 plane

* FORMAT_GRAY

表示预创建一个灰度图格式的图片，有一个 plane

* FORMAT_COMPRESSED

表示预创建一个 VPU 内部压缩格式的图片，共有四个 plane，分别存放内容如下：

plane0: Y 压缩表

plane1: Y 压缩数据

plane2: CbCr 压缩表

plane3: CbCr 压缩数据