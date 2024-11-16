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

        def __init__(self, handle: sail.Handle, 
                     h: int, 
                     w: int, 
                     format: sail.Format, 
                     dtype: sail.ImgDtype)

        def __init__(self, handle: sail.Handle, 
                     buffer: bytes | np.array, 
                     h: int, 
                     w: int, 
                     format: sail.Format, 
                     dtype: sail.ImgDtype = DATA_TYPE_EXT_1N_BYTE, 
                     strides: list[int] = None, 
                     offset: int = 0)

**参数说明:**

* handle: sail.Handle

设定BMImage所在的设备句柄。

* h: int

图像的高。

* w: int

图像的宽。

* format : sail.Format

图像的像素格式。
支持 `sail.Format <0_enum_type/sail.Format.html>`_ 中的像素格式。

* dtype: sail.ImgDtype

图像的数据类型。
支持 `sail.ImgDtype <0_enum_type/sail.ImgDtype.html>`_ 中的像素格式。

* buffer: bytes | np.array

用于创建图像的保存了像素值的buffer

* strides

用buffer创建图像时图像的步长。单位为byte，默认为空，表示和一行的数据宽度相同。
如果需要指定，注意list中元素个数要与图像plane数一致

* offset

用buffer创建图像时有效数据相对buffer起始地址的偏移量。单位为byte，默认为0


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

将BMImage中的数据转换成cv2默认的BGR_PACKED像素格式的numpy.ndarray。

**接口形式:**
    .. code-block:: python

        def asmat(self) -> numpy.ndarray[numpy.uint8]

**返回值说明:**

* image : numpy.ndarray[numpy.uint8]  

返回BMImage中的数据，自动转换成BGR PACKED像素格式。


asnumpy
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

将BMImage中的图像裸数据转换成numpy.ndarray，并保持像素格式不变。

支持的像素格式为 `sail.Format <0_enum_type/sail.Format.html>`_ 中列出的格式。

支持的数据类型为DATA_TYPE_EXT_1N_BYTE、DATA_TYPE_EXT_1N_BYTE_SIGNED和DATA_TYPE_EXT_FLOAT32。

不同像素格式返回的ndarray的shape见下表：

.. list-table:: 
   :widths: 50 50
   :header-rows: 1

   * - 像素格式
     - 输出维度
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
   * - 其他
     - (numel,)

其中，numel表示该BMImage所含像素点的个数。
例如，
对于YUV420P或NV12，numel = height * width * 1.5 ；
对于BGR_PACKED或BGR_PLANAR，numel = height * width * 3 。

**接口形式:**
    .. code-block:: python

        def asnumpy(self) -> numpy.ndarray

**返回值说明:**

* image : numpy.ndarray

返回BMImage中的数据。

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

            # create BMImage with data from buffer
            buf = bytes([i % 256 for i in range(int(200*100*3))])
            img_fromRawdata = sail.BMImage(handle, buf, 200, 100, sail.Format.FORMAT_BGR_PACKED)