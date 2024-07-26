BMImageArray
__________________


BMImageArray is an array of BMImage, which can apply for continuous memory space for multiple images.

When declaring BMImageArray, developer need to specify different instances according to the number of images.

Example: The construction method of BMImageArray when there are 4 images is as follows: images = BMImageArray<4>()


Constructor
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Initialize BMImageArray.

**Interface:**
    .. code-block:: c

        BMImageArray();
        
        BMImageArray(
            Handle                   &handle,
            int                      h,
            int                      w,
            bm_image_format_ext      format,
            bm_image_data_format_ext dtype);

**Parameters:**

* handle: Handle

Set the device handle where the BMImage is located.

* h: int

The height of the image.

* w: int

The width of the image.

* format : bm_image_format_ext

The format of the image.

* dtype: bm_image_data_format_ext

The data type of the image.

**Sample:**
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

Copies the image to a specific index.

**Interface:**
    .. code-block:: c

        int copy_from(int i, BMImage &data);

**Parameters:**

* i: int

Enter the index to be copied.

* data: BMImage

The image data that needs to be copied.


**Sample:**
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

Attach the image to a specific index. There is no memory copy, therefore, the original data needs to be cached.

**Interface:**
    .. code-block:: c

        int attach_from(int i, BMImage &data);

**Sample:**
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

Get the device number in BMImageArray.

**Parameters:**
    .. code-block:: c

        int get_device_id();

**Returns:**

* device_id: int

Device id number in BMImageArray.

**Sample:**
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