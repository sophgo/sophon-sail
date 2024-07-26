Format
______________

Define commonly used image formats.

**Interface:**
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

**Paramters:**

* FORMAT_YUV420P

Pre-create an image in YUV420 format with three planes

* FORMAT_YUV422P

Pre-create an image in YUV422 format with three planes

* FORMAT_YUV444P

Pre-create an image in YUV444 format with three planes

* FORMAT_NV12

Pre-create an image in NV12 format with two planes

* FORMAT_NV21

Pre-create an image in NV21 format with two planes

* FORMAT_NV16

Pre-create an image in NV16 format with two planes

* FORMAT_NV61

Pre-create an image in NV61 format with two planes

* FORMAT_RGB_PLANAR

Pre-create an image in RGB format, with RGB arranged separately and a plane

* FORMAT_BGR_PLANAR

Pre-create an image in BGR format, with BGR arranged separately and a plane

* FORMAT_RGB_PACKED

Pre-create an image in RGB format, with RGB staggered and a plane

* FORMAT_BGR_PACKED

Pre-create an image in BGR format, with BGR staggered and a plane

* FORMAT_RGBP_SEPARATE

Pre-create an image in RGB planar format, in which RGB is arranged separately and each occupies a plane. There are 3 planes in total.

* FORMAT_BGRP_SEPARATE

Pre-create an image in BGR planar format, in which BGR is arranged separately and each occupies a plane. There are 3 planes in total.

* FORMAT_GRAY

Pre-create an image in grayscale format with a plane

* FORMAT_COMPRESSED

Pre-create an iamge in the VPU internal compression format. There are four planes, and the contents are stored as follows:

plane0: Y compression table

plane1: Y compressed data

plane2: CbCr compression table

plane3: CbCr compressed data