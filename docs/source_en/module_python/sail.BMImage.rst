sail.BMImage
____________

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
                    dtype: sail.bm_image_data_format_ext)

        def __init__(self, handle: sail.Handle, 
                     buffer: bytes | np.array, 
                     h: int, 
                     w: int, 
                     format: sail.Format, 
                     dtype: sail.ImgDtype = DATA_TYPE_EXT_1N_BYTE, 
                     strides: list[int] = None, 
                     offset: int = 0)

**Parameters**

* handle : sail.Handle

Handle instance

* h: int

The height of img

* w: int

The width of img

* format : bm_image_format_ext

The pixel format of img. 
Supported formats are listed in `sail.Format <0_enum_type/sail.Format.html>`_ .

* dtype: sail.bm_image_data_format_ext

The data type of img.
Supported data types are listed in `ImgDtype <0_enum-type/sail.ImgDtype.html>`_ .

* buffer: bytes | np.array

The buffer used to create a BMImage, which contains pixel values

* strides

The stride of the image when creating an image with a buffer. The unit is in bytes. 
The default is empty, indicating that it is the same as the data width of one row.
If specified, ensure the number of elements in the list matches the number of image planes.

* offset

The offset of valid data relative to the start address of the buffer when creating an image with a buffer. 
The unit is in bytes, and the default is 0.


width
>>>>>>>>>>>

Get the img width.

**Interface:**
    .. code-block:: python

        def width(self)-> int

**Returns**

* width : int

The width of img


height
>>>>>>>>>>>>>>>>>

Get the img height.

**Interface:**
    .. code-block:: python

        def height(self)-> int

**Returns**

* height : int

The height of img


format
>>>>>>>>>>>>>>>>>
Get the img format.

**Interface:**
    .. code-block:: python

        def format(self)-> bm_image_format_ext

**Returns**

* format : bm_image_format_ext

The format of img


dtype
>>>>>>>>>>>>>

Get the img dtype.

**Interface:**
    .. code-block:: python

        def dtype(self)-> bm_image_data_format_ext

Returns

* dtype: bm_image_data_format_ext

The data type of img


data
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get inner bm_image.  

**Interface:**
    .. code-block:: python
        
        def data(self)-> bm_image

**Returns**

* img : bm_image

the data of img


get_device_id
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Get device id of this image.

**Interface:**
    .. code-block:: python

        def get_device_id(self)-> int
            
**Returns**

* device_id : int   

Tensor Computing Processor ids of this image.


get_handle
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get Handle of BMImage.

**Interface:**
    .. code-block:: python

        def get_handle(self):

**Return:**

* Handle : Handle 

Return the Handle of BMImage.


asmat
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Convert to cv Mat

**Interface:**
    .. code-block:: python

        def asmat(self)-> numpy.ndarray[numpy.uint8]    
            
**Returns**

* image : numpy.ndarray[numpy.uint8]    

only support uint8


asnumpy
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Convert BMImage's data to numpy.ndarray, with original pixel format.

Supported pixel formats are listed in `sail.Format <0_enum_type/sail.Format.html>`_ .

Supported data type are DATA_TYPE_EXT_1N_BYTE、DATA_TYPE_EXT_1N_BYTE_SIGNED和DATA_TYPE_EXT_FLOAT32.

The returned ndarray's shape is corresponding to BMImage's pixel format.

.. list-table:: 
   :widths: 50 50
   :header-rows: 1

   * - pixel format
     - ndarray's shape
   * - FORMAT_BGR_PACKED / FORMAT_BGR_PACKED
     - (height, width, 3)
   * - FORMAT_ARGB_PACKED / FORMAT_ABGR_PACKED
     - (height, width, 4)
   * - FORMAT_GRAY
     - (1, height, width)
   * - FORMAT_BGR_PLANAR / FORMAT_RGB_PLANAR
     - (3, height, width)
   * - FORMAT_YUV444P
     - (3, height, width)
   * - else
     - (numel,)

In above table, ``numel`` means the number of elements in BMImage.
For example,
if pixel format is YUV420P or NV12, numel = height * width * 1.5 ;
if pixel format is BGR_PACKED or BGR_PLANAR, numel = height * width * 3 .

**接口形式:**
    .. code-block:: python

        def asnumpy(self) -> numpy.ndarray

**返回值说明:**

* image : numpy.ndarray

return data in BMImage.

**示例代码:**
    .. code-block:: python

        import sophon.sail as sail
        import numpy as np

        if __name__ == '__main__':
            devid = 0
            handle = sail.Handle(devid)
            height = 1080
            width = 1920
            dtype = sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE
            np_dtype = np.uint8

            # example for BGR_PLANAR
            format = sail.Format.FORMAT_BGR_PLANAR
            numel = int(height * width * 3)
            rawdata = np.random.randint(0, 255, (numel,), np_dtype)
            img = sail.BMImage(handle, rawdata, height, width, format, dtype)
            out_ndarray = img.asnumpy()
            assert out_ndarray.shape == (3, height, width)

            # example for YUV420P
            format = sail.Format.FORMAT_YUV420P
            numel = int(height * width * 1.5)
            rawdata = np.random.randint(0, 255, (numel,), np_dtype)
            img = sail.BMImage(handle, rawdata, height, width, format, dtype)
            out_ndarray = img.asnumpy()
            assert out_ndarray.shape == (numel,)


get_plane_num
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get plane number of this image

**Interface:**
    .. code-block:: python

        def get_plane_num(self)  -> int:


align
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Align the bm_image to 64 bytes

**Interface:**
    .. code-block:: python

        def align(self)  -> int:

**Returns**

* ret : int  

return if BMImage aligned,-1 failed,0 successed


check_align
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Check if the bm_image aligned 

**Interface:**
    .. code-block:: python

        def check_align(self)  -> bool:

**Returns**

* ret : bool  

return if BMImage aligned,1 aligned,0 unaligned


unalign
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Unalign the bm_image to source bm_image

**Interface:**
    .. code-block:: python

        def unalign(self)  -> int:

**Returns**

* ret : int  

return if BMImage unaligned,-1 failed,0 successed


check_contiguous_memory
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Check if the bm_image's memory contiguous

**Interface:**
    .. code-block:: python

        def check_contiguous_memory(self)  -> bool:

**Returns**

* ret : bool  

return if BMImage memory contiguous,1 contiguous,0 uncontiguous


**Sample:**
    .. code-block:: python

        import sophon.sail as sail

        if __name__ == '__main__':
            file_path = "your_image.jpg" # 请替换为您的文件路径
            dev_id = 0
            handle = sail.Handle(dev_id)
            decoder = sail.Decoder(file_path, False, dev_id)
            BMimg = sail.BMImage()
            ret = decoder.read(handle, BMimg)

            # get bm_image
            bm_image = BMimg.data()

            # get BMimg width,height,dtype,format,device_id,plane_num,handle
            print(BMimg.width(), BMimg.height(), BMimg.format(), BMimg.dtype(), BMimg.get_device_id(), BMimg.get_plane_num(), BMimg.get_handle())

            # get mat 
            np_data = BMimg.asmat()
            
            # align BMimg
            ret = BMimg.align()
            if ret == 0:
                print("align success")
            else:
                print("align failed")

            print(BMimg.check_align())

            # unalign BMimg
            ret = BMimg.unalign()
            if ret == 0:
                print("unalign success")
            else:
                print("unalign failed")

            # check contiguous memory
            print(BMimg.check_contiguous_memory())

            # create BMImage with data from buffer
            buf = bytes([i % 256 for i in range(int(200*100*3))])
            img_fromRawdata = sail.BMImage(handle, buf, 200, 100, sail.Format.FORMAT_BGR_PACKED)

get_pts_dts
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get pts and dts.

**Interface:**
    .. code-block:: python

        def get_pts_dts() -> list


**Returns**

* result : list

the value of pts and dts.

**Sample:**
    .. code-block:: python

        import sophon.sail as sail
        if __name__ == '__main__':
            input_file_path = 'your_rtsp_url'  
            dev_id = 0
            handle = sail.Handle(dev_id)
            decoder = sail.Decoder(input_file_path, True, dev_id)
            image = sail.BMImage()
            ret = decoder.read(handle, image)
            if ret == 0:
                print("Frame read successfully into bm_image")
                pts,dts=image.get_pts_dts()
                print("pts:",pts)
                print("dts:",dts)
            else:
                print("Failed to read frame into bm_image")