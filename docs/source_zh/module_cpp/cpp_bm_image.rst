bm_image
______________

bm_image是BMCV中的基本结构，封装了一张图像的主要信息，是后续BMImage和BMImageArray的内部元素。

**结构体:**
    .. code-block:: c

        struct bm_image {
            int width;
            int height;
            bm_image_format_ext image_format;
            bm_data_format_ext data_type;
            bm_image_private* image_private;
        };
                
bm_image 结构成员包括图片的宽高（width、height），图片格式 image_format，图片数据格式 data_type，以及该结构的私有数据。


**示例代码:**
    .. code-block:: c

        #include "cvwrapper.h"

        int main() {
            int dev_id = 0;
            sail::Handle handle(dev_id);
            std::string image_name = "your_image.jpg";
            sail::Decoder decoder(image_name, true, dev_id);
            sail::BMImage BMimg = decoder.read(handle);
            // Convert BMImage to bm_image data structure; here is a bm_image
            struct bm_image img = BMimg.data();

            // print image width, height, format and dtype
            std::cout << img.width << " " << img.height << " " << img.image_format << " " << img.data_type << std::endl;

            return 0;
        }
