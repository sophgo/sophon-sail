Decoder_RawStream
____________________

Original stream decoder for H264/H265 decoding。

Constructor
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Initialize Decoder。

**Interface:**
    .. code-block:: c

        Decoder_RawStream(int tpu_id, string  decformt);

**Parameters:**

* tpu_id: int

The Tensor Computing Processor id that used, which defaults to 0.

* decformat: string

Input image format, supports h264 and h265.


read
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Read a frame of image from Decoder.

**Interface:**
    .. code-block:: c

        int read(uint8_t* data, int data_size, sail::BMImage &image,bool continueFrame = false);
        
**Parameters:**

* data: uint8_t*

Input parameter. The binary data of the original stream.

* image: BMImage

Output parameter. Read data into the BMImage.

* continueFrame: bool

Input parameter. Whether to read frames continuously, the default is false.

**Returns:**

* judge_ret: int

Returns 0 if the read is successful and other values if failed.


read\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Read a frame of image from Decoder.

**Interface:**
    .. code-block:: c

        int read_(uint8_t* data, int data_size, bm_image &image,bool continueFrame = false);

        
**Parameters:**

* data: uint8_t*

Input parameter. The binary data of the original stream.

* image: bm_image

Output parameter. Read data into the bm_image.

* continueFrame: bool

Input parameter. Whether to read frames continuously, the default is false.

**Returns:**

* judge_ret: int

Returns 0 if the read is successful and other values if failed.


release
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Release Decoder resources.

**Interface:**
    .. code-block:: c
    
        void release();

**Example code:**
    .. code-block:: c 

        #include <sail/cvwrapper.h>
        using namespace std;

        int main(int argc, char *argv[]){

            const char *inputFile = "car.264";
            FILE *file = fopen(inputFile, "rb");
            if (!file) {
                fprintf(stderr, "Failed to open file for reading\n");
                return -1;
            }
        
            if(access("output",0)!=F_OK){
                mkdir("output",S_IRWXU);
            }
        
            fseek(file, 0, SEEK_END);
            int numBytes = ftell(file);
            cout << "infile size: " << numBytes << endl;
            fseek(file, 0, SEEK_SET);
        
            uint8_t *bs_buffer = (uint8_t *)av_malloc(numBytes);
            if (bs_buffer == nullptr) {
                cout << "av malloc for bs buffer failed" << endl;
                fclose(file);
                return -1;
            }
        
            fread(bs_buffer, sizeof(uint8_t), numBytes, file);
            fclose(file);
            file = nullptr;
        
            // create handle
            int dev_id=0;
            auto handle = sail::Handle(dev_id);
            bm_image  image;
        
            sail::Decoder_RawStream decoder_rawStream(dev_id,"h264");
            
            int frameCount =0;
            while(true){
                decoder_rawStream.read_(bs_buffer,numBytes,image,true);
                string out = "output/out_" + to_string(frameCount) + ".bmp";
                bm_image_write_to_bmp(image, out.c_str());
                frameCount++; 
            }
            av_free(bs_buffer);
            return 0;
        }


read_single_frame
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Read a frame of image from Decoder. Need to ensure that the input data only contains complete single frame data.

**Interface:**
    .. code-block:: c

        int read_single_frame(uint8_t* data, int data_size, sail::BMImage &image, bool continueFrame = false, bool needFlush = false);

**Parameters:**

* data: bytes

Input parameter. The binary data of the original stream.

* image: sail.BMImage

Output parameter. Read data into the BMImage.

* continue_frame: bool

Input parameter. Whether to read frames continuously, the default is false.

* need_flush: bool

Input parameter. Whether to flush the decoder buffer, the default is false.

**Returns:**

* judge_ret: int

Returns 0 if the read is successful and 1 if the read needs to continue, and other values if failed.

Returns 1 is normal for first frame, please continue to input the next frame data.

