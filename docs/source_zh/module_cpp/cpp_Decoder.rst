Decoder
____________

解码器，可实现图像或视频的解码。

**图像解码像素格式支持说明:**

* 硬解支持jpeg baseline，其余支持软解；
* 视频支持硬解h264,h265。输出的像素格式为YUV-nv12、YUVJ420P或者YUV420P；

构造函数
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

初始化Decoder。

**接口形式:**
    .. code-block:: c

        Decoder(
           const std::string& file_path,
           bool               compressed = true,
           int                tpu_id = 0);

**参数说明:**

* file_path: str

图像或视频文件的Path或RTSP的URL。

* compressed: bool

是否将解码的输出压缩为NV12，default：true。
开启之后可以节省内存、节省带宽，但是输入视频必须要满足宽能被16整除才行，且输入必须为视频时才能生效。

* tpu_id: int

设置使用的智能视觉深度学习处理器 id号。


is_opened
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

判断源文件是否打开。

**接口形式:**
    .. code-block:: c

        bool is_opened();

**返回值说明:**

* judge_ret: bool

打开成功返回True，失败返回False。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        using namespace std;
        using namespace sail;

        int main() {
            string file_path = "your_video_file_path.mp4";
            int tpu_id = 0;

            Decoder decoder(file_path, true, tpu_id);
            if (!decoder.is_opened()) {
                cout << "Failed to open the file!" << endl;
                return -1;
            }
            return 0;
        }

read
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

从Decoder中读取一帧图像。

**接口形式:**
    .. code-block:: c

        int read(Handle& handle, BMImage& image);

        
**参数说明:**

* handle: Handle

输入参数。Decoder使用的智能视觉深度学习处理器的Handle。

* image: BMImage

输出参数。将数据读取到image中。

**返回值说明:**

* judge_ret: int

读取成功返回0，失败返回其他值。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        using namespace std;
        using namespace sail;

        int main() {
            string file_path = "your_video_file_path.mp4";
            int tpu_id = 0;

            Handle handle(tpu_id);
            Decoder decoder(file_path, true, tpu_id);
            BMImage image;

            int ret = decoder.read(handle, image);
            if (ret != 0) {
                cout << "Failed to read a frame!" << endl;
                return ret;
            }
            return 0;
        }

read\_
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

从Decoder中读取一帧图像。

**接口形式:**
    .. code-block:: c

        int read_(Handle& handle, bm_image& image);

        
**参数说明:**

* handle: Handle

输入参数。Decoder使用的智能视觉深度学习处理器的Handle。

* image: bm_image

输出参数。将数据读取到image中。

**返回值说明:**

* judge_ret: int

读取成功返回0，失败返回其他值。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        using namespace std;
        using namespace sail;

        int main() {
            string file_path = "your_video_file_path.mp4";
            int tpu_id = 0;

            Handle handle(tpu_id);
            Decoder decoder(file_path, true, tpu_id);
            BMImage image;
            bm_image bm_img = image.data();
            int ret = decoder.read_(handle, bm_img);
            if (ret != 0) {
                cout << "Failed to read a frame!" << endl;
                return ret;
            }

            return 0;
        }

get_frame_shape
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取Decoder中frame中的shape。

**接口形式:**
    .. code-block:: c

        std::vector<int> get_frame_shape();

**返回值说明:**

* frame_shape: std::vector<int>

返回当前frame的shape。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        using namespace sail;

        int main() {
            string file_path = "your_video_file_path.mp4";
            int tpu_id = 0;

            Decoder decoder(file_path, true, tpu_id);
            vector<int> frame_shape = decoder.get_frame_shape();

            for (auto dim : frame_shape) {
                cout << dim << " ";
            }
            cout << endl;

            return 0;
        }

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
        using namespace sail;

        int main() {
            string file_path = "your_video_file_path.mp4";
            int tpu_id = 0;

            Decoder decoder(file_path, true, tpu_id);
            decoder.release();

            return 0;
        }

reconnect
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Decoder再次连接。

**接口形式:**
    .. code-block:: c

        int reconnect();
        
**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        using namespace std;
        using namespace sail;

        int main() {
            string file_path = "your_video_file_path.mp4";
            int tpu_id = 0;
            Decoder decoder(file_path, true, tpu_id);
            if (decoder.reconnect() != 0) {
                cout << "Reconnect failed!" << endl;
                return -1;
            }

            return 0;
        }

enable_dump
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

开启解码器的dump输入视频功能（不经编码），并缓存最多1000帧未解码的视频。

**接口形式:**
    .. code-block:: c
    
        void enable_dump(int dump_max_seconds):

**参数说明:**

* dump_max_seconds: int

输入参数。dump视频的最大时长，也是内部AVpacket缓存队列的最大长度。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        using namespace std;
        using namespace sail;

        int main() {
            string file_path = "your_video_file_path.mp4";
            int tpu_id = 0;
            int dump_max_seconds = 100;  // 假设要dump的最大时长为100秒

            Decoder decoder(file_path, true, tpu_id);
            decoder.enable_dump(dump_max_seconds);

            return 0;
        }

disable_dump
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

关闭解码器的dump输入视频功能，并清空开启此功能时缓存的视频帧

**接口形式:**
    .. code-block:: c
    
        void disable_dump():
            """ Disable  input video dump without encode.
            """
**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        using namespace std;
        using namespace sail;

        int main() {
            string file_path = "your_video_file_path.mp4";
            int tpu_id = 0;

            Decoder decoder(file_path, true, tpu_id);
            decoder.enable_dump(100);
            decoder.disable_dump();

            return 0;
        }

dump
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

在调用此函数的时刻，dump下前后数秒的输入视频。由于未经编码，必须dump下前后数秒内所有帧所依赖的关键帧。因而接口的dump实现以gop为单位，实际dump下的视频时长将高于输入参数时长。误差取决于输入视频的gop_size，gop越大，误差越大。

**接口形式:**
    .. code-block:: c
    
        int dump(int dump_pre_seconds, int dump_post_seconds, std::string& file_path)

* dump_pre_seconds: int

输入参数。保存调用此接口时刻之前的数秒视频。

* dump_post_seconds: int

输入参数。保存调用此接口时刻之后的数秒视频。

* file_path: std::string&

输入参数。视频路径。

**返回值说明:**

* judge_ret: int

成功返回0，失败返回其他值。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        using namespace std;
        using namespace sail;

        int main() {
            string file_path = "your_video_file_path.mp4";
            string output_file_path = "output_video_path.mp4";
            int tpu_id = 0;
            int dump_pre_seconds = 3;
            int dump_post_seconds = 3;

            Decoder decoder(file_path, true, tpu_id);
            int ret = decoder.dump(dump_pre_seconds, dump_post_seconds, output_file_path);
            if (ret != 0) {
                cout << "Dump failed with error code: " << ret << endl;
                return ret;
            }

            return 0;
        }


get_pts_dts
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取pts和dts

**接口形式:**
    .. code-block:: c
    
        vector<double> get_pts_dts()
    
**返回值说明:**

* result: vector<double> 

输出结果。输出具体的pts和dts值。


**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        using namespace std;
        using namespace sail;

        int main() {
            string file_path = "your_video_file_path.mp4";
            int tpu_id = 0;

            Handle handle(tpu_id);
            Decoder decoder(file_path, true, tpu_id);
            BMImage image;

            int ret = decoder.read(handle, image);
            if (ret != 0) {
                cout << "Failed to read a frame!" << endl;
                return ret;
            }

            std::vector<int> pts_dts;
            pts_dts = decoder.get_pts_dts();
            cout << "pts: " << pts_dts[0] << endl;
            cout << "dts: " << pts_dts[1] << endl;
            return 0;
        }