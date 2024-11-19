BMImage
____________

BMImage encapsulates all the information of an image, and the Bmcv interface can be used to convert the BMImage into a Tensor for model inference.

BMImage is also the basic data type for other image processing operations through the Bmcv interface.

Constructor
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Initialize BMImage.

**Interface:**
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

**Parameters:**

* handle: Handle

Set the device handle where the BMImage is located.

* h: int

The height of the image.

* w: int

The width of the image.

* format : bm_image_format_ext

The format of the image.
Supported formats are listed in `sail.Format <0_enum-type/cpp_Format.html>`_ .

* dtype: bm_image_data_format_ext

The data type of the image.
Supported data types are listed in `ImgDtype <0_enum-type/cpp_ImgDtype.html>`_ .

* buffer: bytes | np.array

The address of the buffer when creating an image with a buffer.

* strides

The stride of the image when creating an image with a buffer. The unit is in bytes. 
The default is empty, indicating that it is the same as the data width of one row.
If specified, ensure the number of elements in the list matches the number of image planes.

* offset

The offset of valid data relative to the start address of the buffer when creating an image with a buffer. 
The unit is in bytes, and the default is 0.


width
>>>>>>>>>>>

Get the width of the image.

**Interface:**
    .. code-block:: c

        int width();

**Returns:**

* width : int

Returns the width of the image.


height
>>>>>>>>>>>>>>>>>

Get the height of the image.

**Interface:**
    .. code-block:: c

        int height();

**Returns:**

* height : int

Returns the height of the image.


format
>>>>>>>>>>>>>>>>>

Get the format of the image.

**Interface:**
    .. code-block:: c

        bm_image_format_ext format();

**Returns:**

* format : bm_image_format_ext

Returns the format of the image.


dtype
>>>>>>>>>>>>>

Get the data type of the image.

**Interface:**
    .. code-block:: c

        bm_image_data_format_ext dtype() const;

**Returns:**

* dtype: bm_image_data_format_ext

Returns the data type of the image.


data
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get bm_image inside BMImage.

**Interface:**
    .. code-block:: c
        
        bm_image& data();

**Returns:**

* img : bm_image

Returns bm_image inside the image.


get_device_id
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the device id number in BMImage.

**Interface:**
    .. code-block:: c

        int get_device_id() const;

**Returns**

* device_id : int  

Returns the device id in BMImage

get_handle
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get Handle of the BMImage.

**Interface:**
    .. code-block:: c

        Handle get_handle();

**Return:**

* Handle : Handle 

Return the Handle of BMImage.

get_plane_num
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the number of image planes in BMImage.

**Interface:**
    .. code-block:: c

        int get_plane_num() const;

**Returns:**

* planes_num : int  

Returns the number of image planes in BMImage.

align
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Align BMImage as 64 bits

**Interface:**
    .. code-block:: c

        int align();

**Returns:**

* ret : int  

Returns whether the BMImage is successfully aligned, -1 represents failure, 0 represents success

check_align
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get whether the image in BMImage is aligned

**Interface:**
    .. code-block:: c

        bool check_align()const;

**Returns:**

* ret : bool  

1 means aligned, 0 means not aligned

unalign
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

unalign the BMImage

**Interface:**
    .. code-block:: c

        int unalign();

**Returns:**

* ret : int  

Returns whether the BMImage is successfully unaligned, -1 means failure, 0 means success

check_contiguous_memory
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get whether the image memory in BMImage is continuous

**Interface:**
    .. code-block:: c

        bool check_contiguous_memory()const;

**Returns:**

* ret : bool  

1 means continuous, 0 means discontinuous


**Sample:**
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