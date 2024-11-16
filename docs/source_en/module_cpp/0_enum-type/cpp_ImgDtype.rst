ImgDtype
______________

Define several image storage formats.
The type of values is BMCV enum ``bm_image_data_format_ext`` .

**Interface:**
    .. code-block:: c

        DATA_TYPE_EXT_FLOAT32
        DATA_TYPE_EXT_1N_BYTE
        DATA_TYPE_EXT_4N_BYTE
        DATA_TYPE_EXT_1N_BYTE_SIGNED
        DATA_TYPE_EXT_4N_BYTE_SIGNED

**Parameters:**

* DATA_TYPE_EXT_FLOAT32

The data type of the image is float32.

* DATA_TYPE_EXT_1N_BYTE

The data type of the image is uint8。

* DATA_TYPE_EXT_4N_BYTE

The data type of the image is uint8, and the data of each 4 images is staggered, and the data reading and writing efficiency is higher.

* DATA_TYPE_EXT_1N_BYTE_SIGNED

The data type of the image is int8。

* DATA_TYPE_EXT_4N_BYTE_SIGNED

The data type of the image is int8, and the data of each 4 images is staggered, and the data reading and writing efficiency is higher.