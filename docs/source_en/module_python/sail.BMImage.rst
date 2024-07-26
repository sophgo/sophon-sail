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

**Parameters**

* handle : sail.Handle

Handle instance

* h: int

The height of img

* w: int

The width of img

* format : bm_image_format_ext

The format of img

* dtype: sail.bm_image_data_format_ext

The data type of img


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
            file_path = '/data/jinyu.lu/jpu_test/1920x1080_yuvj420.jpg' # 请替换为您的文件路径
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
            if ret:
                print("align success")
            else:
                print("align failed")

            print(BMimg.check_align())

            # unalign BMimg
            ret = BMimg.unalign()
            if ret:
                print("unalign success")
            else:
                print("unalign failed")

            # check contiguous memory
            print(BMimg.check_contiguous_memory())