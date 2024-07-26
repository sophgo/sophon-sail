PaddingAtrr
___________________

PaddingAtrr stores various attributes of data padding, and data filling can be performed by configuring PaddingAtrr.

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



Constructor PaddingAtrr()
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Initialize PaddingAtrr

**Interface:**
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

**Parameters:**

* crop_start_x: int 

The offset of the original image relative to the target image in the x direction.

* crop_start_y: int

The offset of the original image relative to the target image in the y direction

* crop_width: int

The original image can be resized while padding. Width is the width of the original image after resize. If no resize is performed, width is the width of the original image.

* crop_height: int

The original image can be resized while padding. Height is the height of the original image after resize. If no resize is performed, height is the height of the original image.

* padding_value_r: int

The pixel value to fill on the R channel when padding.

* padding_value_g: int

The pixel value to fill on the G channel when padding.

* padding_value_b: int

The pixel value to fill on the B channel when padding


set_stx
>>>>>>>>>>>>>>>

Set the offset of the original image relative to the target image in the x direction

**Interface:**
    .. code-block:: c

        void set_stx(unsigned int stx);

**Parameters:**

* stx: int

The offset of the original image relative to the target image in the x direction


set_sty
>>>>>>>>>>>>>>>

Set the offset of the original image relative to the target image in the y direction

**Interface:**
    .. code-block:: c

        void set_sty(unsigned int sty);

**Parameters:**

* sty: int

The offset of the original image relative to the target image in the y direction


set_w
>>>>>>>>>>>>>>>

Set the width of the original image after resize

**Interface:**
    .. code-block:: c

        void set_w(unsigned int w);

**Parameters:**

* width: int

The original image can be resized while padding. Width is the width of the original image after resize. If no resize is performed, width is the width of the original image.


set_h
>>>>>>>>>>>>>>>

Set the height of the original image after resize

**Interface:**
    .. code-block:: c

        void set_h(unsigned int h);

**Parameters:**

* height: int

The original image can be resized while padding. Height is the height of the original image after resize. If no resize is performed, height is the height of the original image.


set_r
>>>>>>>>>>>>>>>

Set the padding value on the R channel

**Interface:**
    .. code-block:: c

        void set_r(unsigned int r);

**Parameters**

* r: int

The padding value on R channel


set_g
>>>>>>>>>>>>>>>

Set the padding value on the G channel

**Interface:**
    .. code-block:: c

        void set_g(unsigned int g);

**Parameters:**

* g: int

The padding value on G channel


set_b
>>>>>>>>>>>>>>>

Set the padding value on the B channel

**Interface:**
    .. code-block:: c

        void set_b(unsigned int b);

**Parameters**

* b: int

The padding value on channel B