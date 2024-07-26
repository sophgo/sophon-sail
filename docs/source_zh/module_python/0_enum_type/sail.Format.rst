sail.Format
______________

定义常用的图像格式。

**接口形式:**
    .. code-block:: python

        sail.Format.FORMAT_YUV420P
        sail.Format.FORMAT_YUV422P
        sail.Format.FORMAT_YUV444P
        sail.Format.FORMAT_NV12
        sail.Format.FORMAT_NV21
        sail.Format.FORMAT_NV16
        sail.Format.FORMAT_NV61
        sail.Format.FORMAT_NV24
        sail.Format.FORMAT_RGB_PLANAR
        sail.Format.FORMAT_BGR_PLANAR
        sail.Format.FORMAT_RGB_PACKED
        sail.Format.FORMAT_BGR_PACKED
        sail.Format.FORMAT_RGBP_SEPARATE
        sail.Format.FORMAT_BGRP_SEPARATE
        sail.Format.FORMAT_GRAY
        sail.Format.FORMAT_COMPRESSED

**参数说明:**

* sail.Format.FORMAT_YUV420P

表示预创建一个 YUV420 格式的图片，有三个 plane

* sail.Format.FORMAT_YUV422P

表示预创建一个 YUV422 格式的图片，有三个 plane

* sail.Format.FORMAT_YUV444P

表示预创建一个 YUV444 格式的图片，有三个 plane

* sail.Format.FORMAT_NV12

表示预创建一个 NV12 格式的图片，有两个 plane

* sail.Format.FORMAT_NV21

表示预创建一个 NV21 格式的图片，有两个 plane

* sail.Format.FORMAT_NV16

表示预创建一个 NV16 格式的图片，有两个 plane

* sail.Format.FORMAT_NV61

表示预创建一个 NV61 格式的图片，有两个 plane

* sail.Format.FORMAT_RGB_PLANAR

表示预创建一个 RGB 格式的图片，RGB 分开排列,有一个 plane

* sail.Format.FORMAT_BGR_PLANAR

表示预创建一个 BGR 格式的图片，BGR 分开排列,有一个 plane

* sail.Format.FORMAT_RGB_PACKED

表示预创建一个 RGB 格式的图片，RGB 交错排列,有一个 plane

* sail.Format.FORMAT_BGR_PACKED

表示预创建一个 BGR 格式的图片，BGR 交错排列,有一个 plane

* sail.Format.FORMAT_RGBP_SEPARATE

表示预创建一个 RGB planar 格式的图片，RGB 分开排列并各占一个 plane，共有 3 个 plane

* sail.Format.FORMAT_BGRP_SEPARATE

表示预创建一个 BGR planar 格式的图片，BGR 分开排列并各占一个 plane，共有 3 个 plane

* sail.Format.FORMAT_GRAY

表示预创建一个灰度图格式的图片，有一个 plane

* sail.Format.FORMAT_COMPRESSED

表示预创建一个 VPU 内部压缩格式的图片，共有四个 plane，分别存放内容如下：

plane0: Y 压缩表

plane1: Y 压缩数据

plane2: CbCr 压缩表

plane3: CbCr 压缩数据