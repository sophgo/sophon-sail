sail.BMImageArray
__________________

\_\_init\_\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: python

        def __init__(self) 
            
        
        def __init__(self, 
                    handle: sail.Handle, 
                    h: int, 
                    w: int, 
                    format: bm_image_format_ext, 
                    dtype: bm_image_data_format_ext)
            
Parameters

* handle : sail.Handle

Handle instance

* h : int

Height instance

* w : int

Width instance

* format : bm_image_format_ext

Format instance

* dtype : bm_image_data_format_ext

Dtype instance

**Sample:**
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

Get the bm_image from index i.

**Interface:**
    .. code-block:: python

        def __getitem__(self, i: int)-> sail.bm_image

**Parameters**

* i : int

Index of the specified location.

**Returns**

* img : sail.bm_image

result bm_image

**Sample:**
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
Copy the image to the specified index.

**Interface:**
    .. code-block:: python

        def __setitem__(self, i: int, data: sail.bm_image)

Parameters

* i: int

Index of the specified location.

* data: sail.bm_image

Input image

**Sample:**
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

Copy the image to the specified index.

**Interface:**
    .. code-block:: python

        def copy_from(self, i: int, data: sail.BMImage): 
            
**Parameters**

* i: int

Index of the specified location.

* data: sail.BMImage

Input image

**Sample:**
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

Attach the image to the specified index.(Because there is no memory copy, \
the original data needs to be cached)

**Interface:**
    .. code-block:: python

        def attach_from(self, i: int, data: BMImage):  
       
**Parameters:**

* i: int

Index of the specified location.

* data: BMImage

Input image.

**Sample:**
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

Get device id of this BMImageArray.

**Interface:**
    .. code-block:: python

        def get_device_id(self)  -> int:

**Sample:**
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