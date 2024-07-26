sail.ImgDtype
______________

定义几种图像的存储形式。

**接口形式:**
    .. code-block:: python

        sail.ImgDtype.DATA_TYPE_EXT_FLOAT32
        sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE
        sail.ImgDtype.DATA_TYPE_EXT_4N_BYTE
        sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE_SIGNED
        sail.ImgDtype.DATA_TYPE_EXT_4N_BYTE_SIGNED

**参数说明:**

* sail.ImgDtype.DATA_TYPE_EXT_FLOAT32

表示图片的数据类型为float32。

* sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE

表示图片的数据类型为uint8。

* sail.ImgDtype.DATA_TYPE_EXT_4N_BYTE

表示图片的数据类型为uint8，且每4张图片的数据交错排列，数据读写效率更高。

* sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE_SIGNED

表示图片的数据类型为int8。

* sail.ImgDtype.DATA_TYPE_EXT_4N_BYTE_SIGNED

表示图片的数据类型为int8，且每4张图片的数据交错排列，数据读写效率更高。