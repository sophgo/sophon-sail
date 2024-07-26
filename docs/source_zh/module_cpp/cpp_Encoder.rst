Encoder
____________

编码器，可实现图像或视频的编码，以及保存视频文件、推rtsp/rtmp流。

构造函数
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

初始化Encoder。

图片编码器初始化：

**图片编码器:**
    .. code-block:: c

        Encoder();

视频编码器初始化：

**视频编码器接口形式1:**
    .. code-block:: c

        Encoder(const std::string &output_path,
                    Handle &handle,
                    const std::string &enc_fmt,
                    const std::string &pix_fmt,
                    const std::string &enc_params,
                    int cache_buffer_length=5,
                    int abort_policy=0);

**视频编码器接口形式2:**
    .. code-block:: c

        Encoder(const std::string &output_path,
                    int device_id,
                    const std::string &enc_fmt,
                    const std::string &pix_fmt,
                    const std::string &enc_params,
                    int cache_buffer_length=5
                    int abort_policy=0);

**参数说明:**

* output_path: string

输入参数。编码视频输出路径，支持本地文件（MP4，ts等）和rtsp/rtmp流。

* handle: sail.Handle

输入参数。编码器handle实例。（与device_id二选一）

* device_id: int

输入参数。编码器device_id。（与handle二选一，指定device_id时，编码器内部将会创建Handle）

* enc_fmt: string

输入参数。编码格式，支持h264_bm和h265_bm/hevc_bm。

* pix_fmt: string

输入参数。编码输出的像素格式，支持NV12和I420。推荐使用I420。

* enc_params: string

输入参数。编码参数， ``"width=1902:height=1080:gop=32:gop_preset=3:framerate=25:bitrate=2000"`` , 其中width和height是必须的，默认用bitrate控制质量，单位为kbps，参数中指定qp时bitrate失效。

* cache_buffer_length: int

输入参数。内部缓存队列长度，默认为5。sail.Encoder内部会维护一个缓存队列，从而在推流时提升流控容错。

* abort_policy: int

输入参数。缓存队列已满时，video_write接口的拒绝策略。设为0时，video_write接口立即返回-1。设为1时，pop队列头。设为2时，清空队列。设为3时，阻塞直到编码线程消耗一帧，队列产生空位。

is_opened
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

判断编码器是否打开。

**接口形式:**
    .. code-block:: c

        bool is_opened();

**返回值说明:**

* judge_ret: bool

编码器打开返回True，失败返回False。

**示例代码:**
    .. code-block:: c

        #include <sail/encoder.h>
        
        using namespace std;  
        int main() {  
            int dev_id = 0;
            sail::Handle handle(dev_id); 
            string out_path = "path/to/your/output/file";           
            string enc_fmt = "h264_bm";                           
            string pix_fmt = "I420";                              
            string enc_params = "width=1920:height=1080:bitrate=2000:gop=32:gop_preset=2:framerate=25";  
            int cache_buffer_length = 5;                      
            int abort_policy = 0;                             
            sail::Encoder encoder(out_path, handle, enc_fmt, pix_fmt, enc_params, cache_buffer_length, abort_policy);
            if(encoder.is_opened())
            {
                cout<<"succeed!"<<endl;
            }
            return 0;  
        }

pic_encode
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

编码一张图片，并返回编码后的data。

**接口形式1:**
    .. code-block:: c

        int pic_encode(std::string& ext, bm_image &image, std::vector<u_char>& data);

**接口形式2:**
    .. code-block:: c

        int pic_encode(std::string& ext, BMImage &image, std::vector<u_char>& data);
   
**参数说明:**

* ext: string

输入参数。图片编码格式。 ``".jpg"`` , ``".png"`` 等。

* image: bm_image/BMImage

输入参数。输入图片，只支持FORMAT_BGR_PACKED，DATA_TYPE_EXT_1N_BYTE的图片。

* data: vector<u_char>

输入参数。字节向量,用于保存编码后放在系统内存中的的数据。

**返回值说明:**

* size: int

编码后放在系统内存中的数据的有效长度。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/encoder.h>
        
        using namespace std;  
        int main() {  
            int dev_id = 0;
            sail::Handle handle(dev_id); 
            string image_path = "your_img_path";  
            sail::Decoder decoder(image_path,false,dev_id);
            sail::BMImage img = decoder.read(handle);    

            string out_path = "path/to/your/output/file";           
            string enc_fmt = "h264_bm";                           
            string pix_fmt = "I420";                              
            string enc_params = "width=1920:height=1080:bitrate=2000:gop=32:gop_preset=2:framerate=25";  
            int cache_buffer_length = 5;                      
            int abort_policy = 0;                             
            sail::Encoder encoder(out_path, handle, enc_fmt, pix_fmt, enc_params, cache_buffer_length, abort_policy); 

            vector<u_char> data;    
            string extension = ".jpg"; 
            int size = encoder.pic_encode(extension,img,data);                 
            //int size = encoder.pic_encode(extension,img.data(),data); //bm_image
  
            return 0;  
        }

video_write
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

向视频编码器送入一帧图像。异步接口，做格式转换后，放入内部的缓存队列中。

**接口形式1:**
    .. code-block:: c

        int video_write(bm_image &image);

**接口形式2:**
    .. code-block:: c

        int video_write(BMImage &image);
        
**参数说明:**

* image: bm_image/BMImage

在BM1684上，
当编码器像素格式（即pix_fmt）为I420时，待编码的image的shape可以与编码器的宽高不同；
当像素格式为NV12时，要求image的shape与编码器的宽高一致，内部使用bmcv_image_storage_convert做格式转换，可能占用NPU资源。

在BM1684X上，待编码的image的shape可以与编码器的宽高不同，内部使用bmcv_image_vpp_convert做resize和格式转换。

**返回值说明:**

* judge_ret: int

成功返回0，内部缓存队列已满返回-1。内部缓存队列中有一帧编码失败时返回-2。有一帧成功编码，但推流失败返回-3。未知的拒绝策略返回-4。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/encoder.h>
        
        using namespace std;  
        int main() {  
            int dev_id = 0;
            sail::Handle handle(dev_id); 
            string image_path = "your_img_path";  
            sail::Decoder decoder(image_path,false,dev_id);
            sail::BMImage img = decoder.read(handle);   
            string out_path = "out_put_path";            
            string enc_fmt = "h264_bm";                           
            string pix_fmt = "I420";                              
            string enc_params = "width=1920:height=1080:bitrate=2000:gop=32:gop_preset=2:framerate=25";  
            int cache_buffer_length = 5;                      
            int abort_policy = 0;                             
            sail::Encoder encoder(out_path, handle, enc_fmt, pix_fmt, enc_params, cache_buffer_length, abort_policy);
            int ret = encoder.video_write(img);
            // int ret = encoder.video_write(img.data());  //bm_image
            return 0;  
        }

release
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

释放编码器。

**示例代码:**
    .. code-block:: c

        #include <sail/encoder.h>
        
        using namespace std;  
        int main() {  
            int dev_id = 0;
            sail::Handle handle(dev_id); 
            string out_path = "path/to/your/output/file";           
            string enc_fmt = "h264_bm";                         
            string pix_fmt = "I420";                              
            string enc_params = "width=1920:height=1080:bitrate=2000:gop=32:gop_preset=2:framerate=25";  
            int cache_buffer_length = 5;                      
            int abort_policy = 0;                        
            sail::Encoder encoder(out_path, handle, enc_fmt, pix_fmt, enc_params, cache_buffer_length, abort_policy);
            encoder.release();
            return 0;  
        }