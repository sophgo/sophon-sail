Encoder
____________

Encoder can encode images or videos, save video files, and push rtsp/rtmp streams.

Constructor
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Initialize Encoder

Image encoder initialization

**Image Encoder Interface:**
    .. code-block:: c

        Encoder();

Vedio encoder initialization

**Vedio Encoder Interface 1:**
    .. code-block:: c

        Encoder(const std::string &output_path,
                    Handle &handle,
                    const std::string &enc_fmt,
                    const std::string &pix_fmt,
                    const std::string &enc_params,
                    int cache_buffer_length=5,
                    int abort_policy=0);

**Vedio Encoder Interface 2:**
    .. code-block:: c

        Encoder(const std::string &output_path,
                    int device_id,
                    const std::string &enc_fmt,
                    const std::string &pix_fmt,
                    const std::string &enc_params,
                    int cache_buffer_length=5
                    int abort_policy=0);

Memory mode encoder initialization (encode output to memory buffer, format is fragmented MP4):

**Memory Mode Encoder Interface:**
    .. code-block:: c

        Encoder(int device_id,
                    const std::string &enc_fmt,
                    const std::string &pix_fmt,
                    const std::string &enc_params,
                    bool memory_mode,
                    int cache_buffer_length=5,
                    int abort_policy=0);

**Parameters:**

* output_path: string

Input parameter. Encoded video output path, supports local files (MP4, ts, etc.) and rtsp/rtmp streams.
* handle: sail.Handle

Input parameter. Encoder handle instance. (Choose one between handle and device_id)

* device_id: int

Input parameter. Encoder device_id. (Choose one of device_id and handle. When device_id is specified, Handle will be created inside the encoder)

* enc_fmt: string

Input parameter. Encoding format, supports h264_bm and h265_bm/hevc_bm.

* pix_fmt: string

Input parameters. The pixel format of the encoding output supports NV12 and I420. I420 is recommed.

* enc_params: string

Input parameter. Encoding parameters, ``"width=1902:height=1080:gop=32:gop_preset=3:framerate=25:bitrate=2000"``, where width and height are required. By default, bitrate is used to control quality. Bitrate is invalid when qp is specified in the parameter. .

* cache_buffer_length: int

Input parameter. Internal cache queue length, default is 5. sail.Encoder internally maintains a cache queue to improve flow control fault tolerance when pushing streams.

* abort_policy: int

Input parameter. The rejection policy of the video_write interface when the cache queue is full. When set to 0, the video_write interface returns -1 immediately. When set to 1, pop the queue head. When set to 2, the queue is cleared. When set to 3, it blocks until the encoding thread consumes one frame and the queue becomes empty.

* memory_mode: bool

Input parameter. Whether to enable memory mode. When set to true, encoded output goes to a memory buffer instead of a file or stream. The output format is fragmented MP4. No output_path is needed in this mode. Use ``get_encoded_data()`` to retrieve the encoded data after encoding is complete.

is_opened
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Determine whether the encoder is turned on.

**Interface:**
    .. code-block:: c

        bool is_opened();

**Returns:**

* judge_ret: bool

Returns true if the encoder is turned on and false if it fails.

**Sample:**
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

Encode an image and return the encoded data.

**Interface 1:**
    .. code-block:: c

        int pic_encode(std::string& ext, bm_image &image, std::vector<u_char>& data);

**Interface 2:**
    .. code-block:: c

        int pic_encode(std::string& ext, BMImage &image, std::vector<u_char>& data);
   
**Parameters:**

* ext: string

Input parameter. Image encoding format. ``".jpg"``, ``".png"`` etc.

* image: bm_image/BMImage

Input parameter. Input pictures, only FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE pictures are supported.

* data: vector<u_char>

Input parameter. Byte vector,a container for holding the data encoded into system memory.

**Returns:**

* size: int

The size of the encoded data held in system memory.

**Sample:**
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

Send a frame of image to the video encoder. Asynchronous interface, after format conversion, is put into the internal cache queue.

**Interface 1:**
    .. code-block:: c

        int video_write(bm_image &image);

**Interface 2:**
    .. code-block:: c

        int video_write(BMImage &image);
        
**Parameters:**

* image: bm_image/BMImage

On the BM1684, 
when the pixel format (pix_fmt) of the encoder is set to I420, the shape of the image to be encoded can differ from the encoder's width and height. 
However, when the pixel format is NV12, the image shape must match the encoder's dimensions. In this case, a format conversion is performed internally using ``bmcv_image_storage_convert``, which may utilize NPU resources.

On the BM1684X, 
the shape of the image to be encoded can differ from the encoder's width and height. The internal resizing and format conversion are handled by ``bmcv_image_vpp_convert``.

**Returns:**

* judge_ret: int

Returns 0 on success, -1 when the internal cache queue is full. -2 is returned when there is a frame in the internal buffer queue that fails to encode. One frame was successfully encoded, but failed to push and returned -3. Unknown deny policy returns -4.

**Sample:**
    .. code-block:: c

        #include<sail/cvwrapper.h>
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

Release the encoder.

**Interface:**
    .. code-block:: c

        void release();

**Sample:**
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

reconnect
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Reconnect the encoder.

**Interface:**
    .. code-block:: c

        int reconnect();

**Return:**

* judge_ret: int

Successfully returned 0, failed to reconnect returned other values.

**Sample:**
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
            int ret = encoder.reconnect();
            if (ret != 0) {
                cout << "Reconnect failed!" << endl;
                return ret;
            }
            return 0;
        }

get_encoded_data
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get encoded data from memory buffer. Must be called after ``release()``. Returns fragmented MP4 data.

**Interface:**
    .. code-block:: c

        std::vector<uint8_t> get_encoded_data();

**Returns:**

* data: vector<uint8_t>

Encoded MP4 data. Can be used for network transmission, file saving, etc. Returns empty vector if the buffer is empty.

**Sample:**
    .. code-block:: c

        #include <sail/encoder.h>
        #include <fstream>

        using namespace std;
        int main() {
            int dev_id = 0;
            string enc_fmt = "h264_bm";
            string pix_fmt = "I420";
            string enc_params = "width=1920:height=1080:bitrate=2000:gop=32:gop_preset=2:framerate=25";
            // Create memory mode encoder
            sail::Encoder encoder(dev_id, enc_fmt, pix_fmt, enc_params, true);
            // Write frames...
            // encoder.video_write(img);
            // Release encoder (flush all buffered frames)
            encoder.release();
            // Get encoded data
            vector<uint8_t> mp4_data = encoder.get_encoded_data();
            cout << "Encoded size: " << mp4_data.size() << " bytes" << endl;
            // Save to file
            ofstream ofs("output.mp4", ios::binary);
            ofs.write(reinterpret_cast<char*>(mp4_data.data()), mp4_data.size());
            return 0;
        }

get_encoded_size
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get current size of the memory encoding buffer in bytes.

**Interface:**
    .. code-block:: c

        size_t get_encoded_size();

**Returns:**

* size: size_t

Current buffer size in bytes. Returns 0 if not in memory mode.

**Sample:**
    .. code-block:: c

        #include <sail/encoder.h>

        using namespace std;
        int main() {
            int dev_id = 0;
            string enc_fmt = "h264_bm";
            string pix_fmt = "I420";
            string enc_params = "width=1920:height=1080:bitrate=2000:gop=32:framerate=25";
            sail::Encoder encoder(dev_id, enc_fmt, pix_fmt, enc_params, true);
            // Check encoding progress after writing frames
            // encoder.video_write(img);
            cout << "Encoded so far: " << encoder.get_encoded_size() << " bytes" << endl;
            return 0;
        }

reset_encoded_data
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Reset the memory encoding buffer for reuse with the same encoder instance.

**Interface:**
    .. code-block:: c

        void reset_encoded_data();

**Sample:**
    .. code-block:: c

        #include <sail/encoder.h>

        using namespace std;
        int main() {
            int dev_id = 0;
            string enc_fmt = "h264_bm";
            string pix_fmt = "I420";
            string enc_params = "width=1920:height=1080:bitrate=2000:gop=32:framerate=25";
            sail::Encoder encoder(dev_id, enc_fmt, pix_fmt, enc_params, true);
            // ... encode first segment ...
            encoder.release();
            vector<uint8_t> data1 = encoder.get_encoded_data();
            // Reset buffer and reuse encoder for next segment
            encoder.reset_encoded_data();
            encoder.reconnect();
            // ... encode second segment ...
            return 0;
        }
