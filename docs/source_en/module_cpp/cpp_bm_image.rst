bm_image
______________

bm_image is the basic structure in BMCV, which encapsulates the main information of an image and is the internal element of subsequent BMImage and BMImageArray.

**struct:**
    .. code-block:: c

        struct bm_image {
            int width;
            int height;
            bm_image_format_ext image_format;
            bm_data_format_ext data_type;
            bm_image_private* image_private;
        };
        

bm_image struct contains width,height,image_format,data_type and its private data.

**Sample:**
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
