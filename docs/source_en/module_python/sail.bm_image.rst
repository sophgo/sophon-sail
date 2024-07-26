sail.bm_image
______________

Functions to get bm_image attributes.

width
>>>>>

Get width of image.

**Interface:**
    .. code-block:: python

        def width(self) -> int
            
**Returns:**

* width : int

The width of image.


height
>>>>>>>>

Get height of image.

**Interface:**
    .. code-block:: python

        def height(self) -> int
            
**Returns:**

* height : int

The height of image.


format
>>>>>>>>

Get format of image.

**Interface:**
    .. code-block:: python

        def format(self) -> bm_image_format_ext

**Returns:**

* format : bm_image_format_ext

Get the format of image.


dtype
>>>>>>>>

Get dtype of image.

**Interface:**
    .. code-block:: python

        def dtype(self) -> bm_image_data_format_ext
            
**Returns:**

* dtype : bm_image_data_format_ext

The dtype of image.

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            image_name = "your_image.jpg"
            decoder = sail.Decoder(image_name,True,tpu_id)
            BMimg = decoder.read(handle)# here is a sail.BMImage
            bmimg = BMimg.data()# here is a sail.bm_image
            print(bmimg.width(),bmimg.height(),bmimg.format(),bmimg.dtype()) 