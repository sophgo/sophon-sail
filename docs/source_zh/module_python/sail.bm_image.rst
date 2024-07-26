sail.bm_image
______________

bm_image是BMCV中的基本结构，封装了一张图像的主要信息，是后续BMImage和BMImageArray的内部元素。

**接口形式:**
    .. code-block:: python

        def width(self) -> int
           

**返回值说明:**

* width : int

返回图像的宽。


**接口形式:**
    .. code-block:: python

        def height(self) -> int

**返回值说明:**

* height : int

返回图像的高。


**接口形式:**
    .. code-block:: python

        def format(self) -> sail.Format


**返回值说明:**

* format : sail.Format

返回图像的格式。


**接口形式:**
    .. code-block:: python

        def dtype(self) -> sail.ImgDtype
         

**返回值说明:**

* dtype : sail.ImgDtype

返回图像的数据格式。

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