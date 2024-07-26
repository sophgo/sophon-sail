BMImageArray
__________________


BMImageArray是BMImage的数组，可为多张图片申请连续的内存空间。

在声明BMImageArray时需要根据图片数量指定不同的实例

例：4张图片时BMImageArray的构造方式如：  images = BMImageArray<4>()


构造函数
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

初始化BMImageArray。

**接口形式:**
    .. code-block:: c

        BMImageArray();
        
        BMImageArray(
            Handle                   &handle,
            int                      h,
            int                      w,
            bm_image_format_ext      format,
            bm_image_data_format_ext dtype);

**参数说明:**

* handle: Handle

设定BMImage所在的设备句柄。

* h: int

图像的高。

* w: int

图像的宽。

* format : bm_image_format_ext

图像的格式。

* dtype: bm_image_data_format_ext

图像的数据类型。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        int main() {
            sail::Handle handle = sail::Handle(0);
            std::string image_path = "your_image.jpg"
            sail::Decoder decoder(image_path,false,0);
            sail::BMImage image;
            decoder.read(handle,image);

            // Create an instance of BMImageArray
            sail::BMImageArray<4> images = sail::BMImageArray<4>(handle,image.width(),image.height(),image.format(),image.dtype());

            return 0;
        }

copy_from
>>>>>>>>>>>>>>>

将图像拷贝到特定的索引上。

**接口形式:**
    .. code-block:: c

        int copy_from(int i, BMImage &data);

**参数说明:**

* i: int

输入需要拷贝到的index

* data: BMImage

需要拷贝的图像数据。

**返回值说明:**

* ret: int

返回0代表成功，其他代表失败。


**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        int main() {
            sail::Handle handle = sail::Handle(0);
            std::string image_path = "your_image.jpg"
            sail::Decoder decoder(image_path,false,0);
            sail::BMImage image;
            decoder.read(handle,image);

            // Create an instance of BMImageArray
            sail::BMImageArray<4> images = sail::BMImageArray<4>(handle,image.width(),image.height(),image.format(),image.dtype());
            // copy from BMImage
            int ret = images.copy_from(0,image);
            if (ret != 0) {
                std::cout << "copy_from failed" << std::endl;
                return -1;
            }
            return 0;
        }


attach_from
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

将图像attach到特定的索引上，这里没有内存拷贝，所以需要原始数据已经被缓存。

**接口形式:**
    .. code-block:: c

        int attach_from(int i, BMImage &data);
    
**参数说明:**

* i: int

输入需要拷贝到的index

* data: BMImage

需要拷贝的图像数据。

**返回值说明:**

* ret: int

返回0代表成功，其他代表失败。


**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        int main() {
            sail::Handle handle = sail::Handle(0);
            std::string image_path = "your_image.jpg"
            sail::Decoder decoder(image_path,false,0);
            sail::BMImage image;
            decoder.read(handle,image);

            // Create an instance of BMImageArray
            sail::BMImageArray<4> images = sail::BMImageArray<4>(handle,image.width(),image.height(),image.format(),image.dtype());
            // attach from BMImage
            ret = images.attach_from(1,image);
            if (ret != 0) {
                std::cout << "attach_from failed" << std::endl;
                return -1;
            }
            return 0;
        }

get_device_id
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取BMImageArray中的设备号。

**接口形式:**
    .. code-block:: c

        int get_device_id();

**返回值说明:**

* device_id: int

BMImageArray中的设备id号

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        int main() {
            sail::Handle handle = sail::Handle(0);
            std::string image_path = "/data/jinyu.lu/jpu_test/1920x1080_yuvj420.jpg";
            sail::Decoder decoder(image_path,false,0);
            sail::BMImage image;
            decoder.read(handle,image);

            // Create an instance of BMImageArray
            sail::BMImageArray<4> images = sail::BMImageArray<4>(handle,image.height(),image.width(),image.format(),image.dtype());

            // get devid
            int devid = images.get_device_id();
            std::cout << "devid: " << devid << std::endl;

            return 0;
        }
