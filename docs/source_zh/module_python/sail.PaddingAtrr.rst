sail.PaddingAtrr
___________________


PaddingAtrr中存储了数据padding的各项属性，可通过配置PaddingAtrr进行数据填充


\_\_init\_\_
>>>>>>>>>>>>>>>

初始化PaddingAtrr

**接口形式:**
    .. code-block:: python

        def __init__(self) 

        def __init__(self, stx: int, sty: int, width: int, height: int, r: int, g: int, b: int)

**参数说明:**

* stx: int 

原图像相对于目标图像在x方向上的偏移量

* sty: int

原图像相对于目标图像在y方向上的偏移量

* width: int

在padding的同时可对原图像进行resize，width为原图像resize后的宽，若不进行resize，则width为原图像的宽

* height: int

在padding的同时可对原图像进行resize，height为原图像resize后的高，若不进行resize，则height为原图像的高

* r: int

padding时在R通道上填充的像素值

* g: int

padding时在G通道上填充的像素值

* b: int

padding时在B通道上填充的像素值


set_stx
>>>>>>>>>>>>>>>

设置原图像相对于目标图像在x方向上的偏移量

**接口形式:**
    .. code-block:: python

        def set_stx(self, stx: int) -> None

**参数说明:**

* stx: int

原图像相对于目标图像在x方向上的偏移量


set_sty
>>>>>>>>>>>>>>>

设置原图像相对于目标图像在y方向上的偏移量

**接口形式:**
    .. code-block:: python

        def set_sty(self, sty: int) -> None

**参数说明:**

* sty: int

原图像相对于目标图像在y方向上的偏移量


set_w
>>>>>>>>>>>>>>>

设置原图像resize后的width

**接口形式:**
    .. code-block:: python

        def set_w(self, width: int) -> None

**参数说明:**

* width: int

在padding的同时可对原图像进行resize，width为原图像resize后的宽，若不进行resize，则width为原图像的宽


set_h
>>>>>>>>>>>>>>>

设置原图像resize后的height

**接口形式:**
    .. code-block:: python

        def set_h(self, height: int) -> None

**参数说明:**

* height: int

在padding的同时可对原图像进行resize，height为原图像resize后的高，若不进行resize，则height为原图像的高


set_r
>>>>>>>>>>>>>>>

设置R通道上的padding值

**接口形式:**
    .. code-block:: python

        def set_r(self, r: int) -> None

**参数说明**

* r: int

R通道上的padding值


set_g
>>>>>>>>>>>>>>>

设置G通道上的padding值

**接口形式:**
    .. code-block:: python

        def set_g(self, g: int) -> None

**参数说明:**

* g: int

G通道上的padding值


set_b
>>>>>>>>>>>>>>>

设置B通道上的padding值

**接口形式:**
    .. code-block:: python

        def set_b(self, b: int) -> None

**参数说明**

* b: int

B通道上的padding值

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            tpu_id = 0
            handle = sail.Handle(tpu_id)
            bmcv = sail.Bmcv(handle)
            paddingatt = sail.PaddingAtrr()   
            paddingatt.set_stx(0)
            paddingatt.set_sty(0)
            paddingatt.set_w(resize_w)
            paddingatt.set_h(resize_h)
            paddingatt.set_r(114)
            paddingatt.set_g(114)
            paddingatt.set_b(114)
            output_temp = bmcv.crop_and_resize_padding(input,0,0,image_w,image_h,resize_w,resize_h,paddingatt)