sail.BMImageArray
__________________


BMImageArray是BMImage的数组，可为多张图片申请连续的内存空间。

在声明BMImageArray时需要根据图片数量指定不同的实例

例：4张图片时BMImageArray的构造方式如：  images = BMImageArray4D()


\_\_init\_\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

初始化BMImageArray。

**接口形式:**
    .. code-block:: python

        def __init__(self) :
        
        def __init__(self, handle: sail.Handle, h: int, w: int, format: sail.Format, dtype: sail.ImgDtype)

**参数说明:**

* handle: sail.Handle

设定BMImage所在的设备句柄。

* h: int

图像的高。

* w: int

图像的宽。

* format : sail.Format

图像的格式。

* dtype: sail.ImgDtype

图像的数据类型。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            # Create BMImageArray4D
            images = sail.BMImageArray4D() 

            # Create BMImageArray4D with parameters
            dev_id = 0
            handle = sail.Handle(dev_id)
            bmcv = sail.Bmcv(handle)
            decoder = sail.Decoder("your_image.jpg",True,dev_id)
            ori_img = decoder.read(handle)
            images = sail.BMImageArray4D(handle, ori_img.height(), ori_img.width(), ori_img.format(), ori_img.dtype()) 

__getitem__
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取index上的bm_image。

**接口形式:**
    .. code-block:: python

        def __getitem__(self, i: int)->sail.bm_image

**参数说明:**

* i: int

需要返回图像的index。

**返回值说明:**

* img: sail.bm_image

返回index上的图像。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            # Create BMImageArray4D with parameters
            dev_id = 0
            handle = sail.Handle(dev_id)
            bmcv = sail.Bmcv(handle)
            decoder = sail.Decoder("your_image.jpg",True,dev_id)
            ori_img = decoder.read(handle)
            images = sail.BMImageArray4D(handle, ori_img.height(), ori_img.width(), ori_img.format(), ori_img.dtype()) 

            images.copy_from(0,ori_img)
            # get the bm_image from index 0
            img_0 = images.__getitem__(0)
            print("image0 from bmimg_array:",img_0.width(),img_0.height(),img_0.dtype())


__setitem__
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

将图像拷贝到特定的索引上。

**接口形式:**
    .. code-block:: python

        def __setitem__(self, i: int, data: sail.bm_image)->None

**参数说明:**

* i: int

输入需要拷贝到的index

* data: sail.bm_image

需要拷贝的图像数据。


**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            # Create BMImageArray4D with parameters
            dev_id = 0
            handle = sail.Handle(dev_id)
            bmcv = sail.Bmcv(handle)
            decoder = sail.Decoder("your_image.jpg",True,dev_id)
            ori_img = decoder.read(handle)
            images = sail.BMImageArray4D(handle, ori_img.height(), ori_img.width(), ori_img.format(), ori_img.dtype()) 
            # copy image to the specified index
            images.__setitem__(3,ori_img.data())

copy_from
>>>>>>>>>>>>>>>

将图像拷贝到特定的索引上。

**接口形式:**
    .. code-block:: python

        def copy_from(self, i: int, data: sail.BMImage)->None

**参数说明:**

* i: int

输入需要拷贝到的index

* data: sail.BMImage

需要拷贝的图像数据。

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            # Create BMImageArray4D with parameters
            dev_id = 0
            handle = sail.Handle(dev_id)
            bmcv = sail.Bmcv(handle)
            decoder = sail.Decoder("your_image.jpg",True,dev_id)
            ori_img = decoder.read(handle)
            images = sail.BMImageArray4D(handle, ori_img.height(), ori_img.width(), ori_img.format(), ori_img.dtype()) 
            # copy image to the specified index
            images.copy_from(0,ori_img)


attach_from
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

将图像attach到特定的索引上，这里没有内存拷贝，所以需要原始数据已经被缓存。

**接口形式:**
    .. code-block:: python

        def attach_from(self, i: int, data: BMImage)->None

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            # Create BMImageArray4D with parameters
            dev_id = 0
            handle = sail.Handle(dev_id)
            bmcv = sail.Bmcv(handle)
            decoder = sail.Decoder("your_image.jpg",True,dev_id)
            ori_img = decoder.read(handle)
            images = sail.BMImageArray4D(handle, ori_img.height(), ori_img.width(), ori_img.format(), ori_img.dtype()) 
            # Attach image to the specified index
            images.attach_from(1,ori_img)

get_device_id
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取BMImageArray中的设备号。

**接口形式:**
    .. code-block:: python

        def get_device_id(self) -> int:

**返回值说明:**

* device_id: int

BMImageArray中的设备id号


**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            # Create BMImageArray4D with parameters
            dev_id = 0
            handle = sail.Handle(dev_id)
            bmcv = sail.Bmcv(handle)
            decoder = sail.Decoder("your_image.jpg",True,dev_id)
            ori_img = decoder.read(handle)
            images = sail.BMImageArray4D(handle, ori_img.height(), ori_img.width(), ori_img.format(), ori_img.dtype()) 
            # Get device id of this BMImageArray
            devid = images.get_device_id()
            print("device id:",devid)