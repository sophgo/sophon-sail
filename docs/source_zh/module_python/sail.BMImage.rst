sail.BMImage
____________

BMImage封装了一张图片的全部信息，可利用Bmcv接口将BMImage转换为Tensor进行模型推理。

BMImage也是通过Bmcv接口进行其他图像处理操作的基本数据类型。

\_\_init\_\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

初始化BMImage。

**接口形式:**
    .. code-block:: python

        def __init__(self)

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


width
>>>>>>>>>>>

获取图像的宽。

**接口形式:**
    .. code-block:: python

        def width(self)->int

**返回值说明:**

* width : int

返回图像的宽。


height
>>>>>>>>>>>>>>>>>

获取图像的高。

**接口形式:**
    .. code-block:: python

        def height(self)->int

**返回值说明:**

* height : int

返回图像的高。


format
>>>>>>>>>>>>>>>>>

获取图像的格式。

**接口形式:**
    .. code-block:: python

        def format(self) -> sail.Format

**返回值说明:**

* format : sail.Format

返回图像的格式。


dtype
>>>>>>>>>>>>>

获取图像的数据类型。

**接口形式:**
    .. code-block:: python

        def dtype(self)->sail.ImgDtype

**返回值说明:**

* dtype: sail.ImgDtype

返回图像的数据类型。


data
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取BMImage内部的bm_image。

**接口形式:**
    .. code-block:: python
        
        def data(self) -> sail.bm_image

**返回值说明:**

* img : sail.bm_image

返回图像内部的bm_image。


get_device_id
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取BMImage中的设备id号。

**接口形式:**
    .. code-block:: python

        def get_device_id(self) -> int

**返回值说明:**

* device_id : int  

返回BMImage中的设备id号


get_handle
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取BMImage中的Handle。

**接口形式:**
    .. code-block:: python

        def get_handle(self):

**返回值说明:**

* Handle : Handle 

返回BMImage中的Handle

asmat
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

将BMImage中的数据转换成numpy.ndarray

**接口形式:**
    .. code-block:: python

        def asmat(self) -> numpy.ndarray[numpy.uint8]

**返回值说明:**

* image : numpy.ndarray[numpy.uint8]  

返回BMImage中的数据。


get_plane_num
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取BMImage中图像plane的数量。

**接口形式:**
    .. code-block:: python

        def get_plane_num(self)  -> int:

**返回值说明:**

* planes_num : int  

返回BMImage中图像plane的数量。


align
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

将BMImage 64对齐

**接口形式:**
    .. code-block:: python

        def align(self)  -> int:

**返回值说明:**

* ret : int  

返回BMImage是否对齐成功,-1代表失败,0代表成功


check_align
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取BMImage中图像是否对齐

**接口形式:**
    .. code-block:: python

        def check_align(self)  -> bool:

**返回值说明:**

* ret : bool  

1代表已对齐,0代表未对齐


unalign
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

将BMImage不对齐

**接口形式:**
    .. code-block:: python

        def unalign(self)  -> int:

**返回值说明:**

* ret : int  

返回BMImage是否不对齐成功,-1代表失败,0代表成功


check_contiguous_memory
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取BMImage中图像内存是否连续

**接口形式:**
    .. code-block:: python

        def check_contiguous_memory(self)  -> bool:

**返回值说明:**

* ret : bool  

1代表连续,0代表不连续

**示例代码:**
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