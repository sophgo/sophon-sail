Decoder_RawStream
____________________

裸流解码器，可实现H264/H264的解码。

构造函数
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

初始化Decoder_RawStream。

**接口形式:**
    .. code-block:: c

        Decoder_RawStream(int tpu_id, string  decformt);

**参数说明:**

* tpu_id: int

设置智能视觉深度学习处理器的id号。

* decformat: string

输入参数。输入图像的格式，支持h264和h265


read
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

从Decoder_RawStream中读取一帧图像。

**接口形式:**
    .. code-block:: c

        int read(uint8_t* data, int data_size, sail::BMImage &image,bool continueFrame = false);
        
**参数说明:**

* data: uint8_t*

输入参数。裸流的二进制数据。

* image: BMImage

输出参数。将数据读取到image中。

* continueFrame: bool

输入参数。是否连续读帧,默认为false。

**返回值说明:**

* judge_ret: int

读取成功返回0，失败返回其他值。

read\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

从Decoder_RawStream中读取一帧图像。

**接口形式:**
    .. code-block:: c

        int read_(uint8_t* data, int data_size, bm_image &image,bool continueFrame = false);

        
**参数说明:**

* data: uint8_t*

输入参数。裸流的二进制数据。

* image: bm_image

输出参数。将数据读取到image中。

* continueFrame: bool

输入参数。是否连续读帧,默认为false。

**返回值说明:**

* judge_ret: int

读取成功返回0，失败返回其他值。


release
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

释放Decoder资源。

**接口形式:**
    .. code-block:: c
    
        void release();
        

**示例代码:**
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