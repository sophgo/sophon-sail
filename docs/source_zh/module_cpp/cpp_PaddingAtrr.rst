PaddingAtrr
___________________

PaddingAtrr中存储了数据padding的各项属性，可通过配置PaddingAtrr进行数据填充

    .. code-block:: c
    
        class PaddingAtrr {
        public:
                PaddingAtrr(){};
                PaddingAtrr(
                    unsigned int crop_start_x,
                    unsigned int crop_start_y,
                    unsigned int crop_width,
                    unsigned int crop_height,
                    unsigned char padding_value_r,
                    unsigned char padding_value_g,
                    unsigned char padding_value_b);
            PaddingAtrr(const PaddingAtrr& other);
            ~PaddingAtrr(){};
            void set_stx(unsigned int stx);
            void set_sty(unsigned int sty);
            void set_w(unsigned int w);
            void set_h(unsigned int h);
            void set_r(unsigned int r);
            void set_g(unsigned int g);
            void set_b(unsigned int b);

            unsigned int    dst_crop_stx; // Offset x information relative to the origin of dst image
            unsigned int    dst_crop_sty; // Offset y information relative to the origin of dst image
            unsigned int    dst_crop_w;   // The width after resize
            unsigned int    dst_crop_h;   // The height after resize
            unsigned char   padding_r;    // Pixel value information of R channel
            unsigned char   padding_g;    // Pixel value information of G channel
            unsigned char   padding_b;    // Pixel value information of B channel
        };



构造函数PaddingAtrr()
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

初始化PaddingAtrr

**接口形式:**
    .. code-block:: c

        PaddingAtrr()

        PaddingAtrr(
                unsigned int crop_start_x,
                unsigned int crop_start_y,
                unsigned int crop_width,
                unsigned int crop_height,
                unsigned char padding_value_r,
                unsigned char padding_value_g,
                unsigned char padding_value_b);

**参数说明:**

* crop_start_x: int 

原图像相对于目标图像在x方向上的偏移量

* crop_start_y: int

原图像相对于目标图像在y方向上的偏移量

* crop_width: int

在padding的同时可对原图像进行resize，width为原图像resize后的宽，若不进行resize，则width为原图像的宽

* crop_height: int

在padding的同时可对原图像进行resize，height为原图像resize后的高，若不进行resize，则height为原图像的高

* padding_value_r: int

padding时在R通道上填充的像素值

* padding_value_g: int

padding时在G通道上填充的像素值

* padding_value_b: int

padding时在B通道上填充的像素值


set_stx
>>>>>>>>>>>>>>>

设置原图像相对于目标图像在x方向上的偏移量

**接口形式:**
    .. code-block:: c

        void set_stx(unsigned int stx);

**参数说明:**

* stx: int

原图像相对于目标图像在x方向上的偏移量


set_sty
>>>>>>>>>>>>>>>

设置原图像相对于目标图像在y方向上的偏移量

**接口形式:**
    .. code-block:: c

        void set_sty(unsigned int sty);

**参数说明:**

* sty: int

原图像相对于目标图像在y方向上的偏移量


set_w
>>>>>>>>>>>>>>>

设置原图像resize后的width

**接口形式:**
    .. code-block:: c

        void set_w(unsigned int w);

**参数说明:**

* width: int

在padding的同时可对原图像进行resize，width为原图像resize后的宽，若不进行resize，则width为原图像的宽


set_h
>>>>>>>>>>>>>>>

设置原图像resize后的height

**接口形式:**
    .. code-block:: c

        void set_h(unsigned int h);

**参数说明:**

* height: int

在padding的同时可对原图像进行resize，height为原图像resize后的高，若不进行resize，则height为原图像的高


set_r
>>>>>>>>>>>>>>>

设置R通道上的padding值

**接口形式:**
    .. code-block:: c

        void set_r(unsigned int r);

**参数说明**

* r: int

R通道上的padding值


set_g
>>>>>>>>>>>>>>>

设置G通道上的padding值

**接口形式:**
    .. code-block:: c

        void set_g(unsigned int g);

**参数说明:**

* g: int

G通道上的padding值


set_b
>>>>>>>>>>>>>>>

设置B通道上的padding值

**接口形式:**
    .. code-block:: c

        void set_b(unsigned int b);

**参数说明**

* b: int

B通道上的padding值

**示例代码:**
    .. code-block:: c

        #include <stdio.h>
        #include <sail/cvwrapper.h>
        #include <iostream>
        #include <string>

        using namespace std;

        int main() {
            int tpu_id = 0;  
            sail::Handle handle(tpu_id);  
            std::string image_name = "../../../sophon-demo/sample/YOLOv5/datasets/test/3.jpg";  
            sail::Decoder decoder(image_name, true, tpu_id);  
            sail::BMImage BMimg = decoder.read(handle);  
            sail::Bmcv bmcv(handle);  
            sail::PaddingAtrr paddingatt;  
            paddingatt.set_stx(0);  
            paddingatt.set_sty(0);  
            paddingatt.set_w(640);  
            paddingatt.set_h(640);  
            paddingatt.set_r(114);  
            paddingatt.set_g(114);  
            paddingatt.set_b(114);  
            sail::BMImage BMimg3 = bmcv.crop_and_resize(BMimg, 0, 0, BMimg.width(), BMimg.height(), 640, 640, paddingatt);
            bmcv.imwrite("{}-{}.jpg".format(BMimg3.width(), BMimg3.height()), BMimg3);
            return 0;
        }