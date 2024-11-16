BMImage
____________

BMImage封装了一张图片的全部信息，可利用Bmcv接口将BMImage转换为Tensor进行模型推理。

BMImage也是通过Bmcv接口进行其他图像处理操作的基本数据类型。

构造函数
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

初始化BMImage。

**接口形式:**
    .. code-block:: c

        BMImage();

        BMImage(
            Handle&                  handle,
            int                      h,
            int                      w,
            bm_image_format_ext      format,
            bm_image_data_format_ext dtype);

        BMImage(
            Handle                   &handle,
            void*                    buffer,
            int                      h,
            int                      w,
            bm_image_format_ext      format,
            bm_image_data_format_ext dtype = DATA_TYPE_EXT_1N_BYTE,
            std::vector<int>         strides = {},
            size_t                   offset = 0);

**参数说明:**

* handle: Handle

设定BMImage所在的设备句柄。

* h: int

图像的高。

* w: int

图像的宽。

* format : bm_image_format_ext

图像的像素格式。
支持 `sail.Format <0_enum-type/cpp_Format.html>`_ 中的像素格式。

* dtype: bm_image_data_format_ext

图像的数据类型。
支持 `ImgDtype <0_enum-type/cpp_ImgDtype.html>`_ 中的数据类型。

* buffer: void*

用buffer创建图像时buffer的地址。

* strides: std::vector<int>

用buffer创建图像时图像的步长。单位为byte，默认为空，表示和一行的数据宽度相同。
如果需要指定，注意stride元素个数要与图像plane数一致

* offset: int

用buffer创建图像时有效数据相对buffer起始地址的偏移量。单位为byte，默认为0

width
>>>>>>>>>>>

获取图像的宽。

**接口形式:**
    .. code-block:: c

        int width();

**返回值说明:**

* width : int

返回图像的宽。


height
>>>>>>>>>>>>>>>>>

获取图像的高。

**接口形式:**
    .. code-block:: c

        int height();

**返回值说明:**

* height : int

返回图像的高。


format
>>>>>>>>>>>>>>>>>

获取图像的格式。

**接口形式:**
    .. code-block:: c

        bm_image_format_ext format();

**返回值说明:**

* format : bm_image_format_ext

返回图像的格式。


dtype
>>>>>>>>>>>>>

获取图像的数据类型。

**接口形式:**
    .. code-block:: c

        bm_image_data_format_ext dtype() const;

**返回值说明:**

* dtype: bm_image_data_format_ext

返回图像的数据类型。


data
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取BMImage内部的bm_image。

**接口形式:**
    .. code-block:: c
        
        bm_image& data();

**返回值说明:**

* img : bm_image

返回图像内部的bm_image。


get_device_id
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取BMImage中的设备id号。

**接口形式:**
    .. code-block:: c

        int get_device_id() const;

**返回值说明:**

* device_id : int  

返回BMImage中的设备id号


get_handle
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取BMImage中的Handle。

**接口形式:**
    .. code-block:: c

        Handle& get_handle();

**返回值说明:**

* Handle : Handle 

返回BMImage中的Handle


get_plane_num
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取BMImage中图像plane的数量。

**接口形式:**
    .. code-block:: c

        int get_plane_num() const;

**返回值说明:**

* planes_num : int  

返回BMImage中图像plane的数量。

align
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

将BMImage 64对齐

**接口形式:**
    .. code-block:: c

        int align();

**返回值说明:**

* ret : int  

返回BMImage是否对齐成功,-1代表失败,0代表成功

check_align
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取BMImage中图像是否对齐

**接口形式:**
    .. code-block:: c

        bool check_align()const;

**返回值说明:**

* ret : bool  

1代表已对齐,0代表未对齐

unalign
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

将BMImage不对齐

**接口形式:**
    .. code-block:: c

        int unalign();

**返回值说明:**

* ret : int  

返回BMImage是否不对齐成功,-1代表失败,0代表成功

check_contiguous_memory
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取BMImage中图像内存是否连续

**接口形式:**
    .. code-block:: c

        bool check_contiguous_memory()const;

**返回值说明:**

* ret : bool  

1代表连续,0代表不连续

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        int main() {
            int dev_id = 0;
            sail::Handle handle(dev_id);
            std::string image_name = "your_image.jpg";
            sail::Decoder decoder(image_name, true, dev_id);
            sail::BMImage BMimg = decoder.read(handle);

            // Get the image information
            int width = BMimg.width();  
            int height = BMimg.height();  
            bm_image_format_ext format = BMimg.format();  
            bm_image_data_format_ext dtype = BMimg.dtype();  

            // Convert BMImage to bm_image data structure  
            bm_image bmimg = BMimg.data();  

            // Get the device id and handle
            int device_id = BMimg.get_device_id();  
            sail::Handle handle_ = BMimg.get_handle();
            int plane_num = BMimg.get_plane_num();  
            std::cout << "Width: " << width << ", Height: " << height << ", Format: " << format << ", Data Type: " << dtype << ", Device ID: " << device_id << ", Plane Num: " << plane_num << std::endl;  

            int ret;
            // Align the image  
            ret = BMimg.align();  
            if (ret != 0) {  
                std::cout << "Failed to align the image!" << std::endl;  
            }
            std::cout << "is align: " << BMimg.check_align() << std::endl;      

            // unalign the image
            ret = BMimg.unalign();
            if (ret != 0) {  
                std::cout << "Failed to unalign the image!" << std::endl;    
            }
            std::cout << "is align: " << BMimg.check_align() << std::endl;

            // check contiguous memory
            std::cout << "is continues: " <<BMimg.check_contiguous_memory()<< std::endl;

            // create BMImage with data from buffer
            std::vector<uint8_t> buf(200 * 100 * 3);
            for (int i = 0; i < 200 * 100 * 3; ++i) {
                buf[i] = i % 256;
            }
            sail::BMImage img_fromRawdata(handle, buf.data(), 200, 100, sail::Format::FORMAT_BGR_PACKED);

            return 0;  
        }