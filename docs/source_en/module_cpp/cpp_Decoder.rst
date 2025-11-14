Decoder
____________

Decoder, which can decode images or videos.

Constructor
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Initialize Decoder。

**Interface:**
    .. code-block:: c

        Decoder(
           const std::string& file_path,
           bool               compressed = true,
           int                tpu_id = 0);

**Parameters:**

* file_path: str

The path or RTSP URL of the image or video file.

* compressed: bool

Whether to compress the decoded output to NV12, default is true

* tpu_id: int

Set the ID of Tensor Computing Processor.


is_opened
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Judge if the source is opened successfully.

**Interface:**
    .. code-block:: c

        bool is_opened();

**Returns:**

* judge_ret: bool

Returns true if the opening is successful and false if it fails.

**Sample:**
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

Read a frame of image from the Decoder.

**Interface:**
    .. code-block:: c

        int read(Handle& handle, BMImage& image);

        
**Parameters:**

* handle: Handle

Input parameter. Handle of Tensor Computing Processor used by Decoder.

* image: BMImage

Output parameter. Read data into image.

**Returns:**

* judge_ret: int

Returns 0 if the read is successful and other values if failed.

**Sample:**
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

Read a frame of image from the Decoder.

**Interface:**
    .. code-block:: c

        int read_(Handle& handle, bm_image& image);

        
**Parameters:**

* handle: Handle

Input parameter. Handle of Tensor Computing Processor used by Decoder.

* image: bm_image

Output parameter. Read data into image.

**Returns:**

* judge_ret: int

Returns 0 if the read is successful and other values if failed.

**Sample:**
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
            bm_image  bm_img = image.data();
            int ret = decoder.read_(handle, bm_img);
            if (ret != 0) {
                cout << "Failed to read a frame!" << endl;
                return ret;
            }
            return 0;
        }

get_frame_shape
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the shape in the frame in the Decoder.

**Interface:**
    .. code-block:: c

        std::vector<int> get_frame_shape();

**Returns:**

* frame_shape: std::vector<int>

Returns the shape of the current frame.

**Sample:**
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

Release Decoder resources.

**Interface:**
    .. code-block:: c
    
        void release();

**Sample:**
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

Decoder connects again.

**Interface:**
    .. code-block:: c

        int reconnect();

**Sample:**
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

Enable the dump input video function (without encoding) of decoder and cache up to 1000 frames of undecoded video.

**Interface:**
    .. code-block:: c
    
        void enable_dump(int dump_max_seconds):

**Parameters:**

* dump_max_seconds: int

Input parameter. The maximum duration of the dump video is also the maximum length of the internal AVpacket cache queue.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        using namespace std;
        using namespace sail;

        int main() {
            string file_path = "your_video_file_path.mp4";
            int tpu_id = 0;
            int dump_max_seconds = 100;  

            Decoder decoder(file_path, true, tpu_id);
            decoder.enable_dump(dump_max_seconds);

            return 0;
        }

disable_dump
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Turn off the dump input video function of decoder and clear the cached video frames when this function is turned on.

**Interface:**
    .. code-block:: c
    
        void disable_dump():
            """ Disable  input video dump without encode.
            """

**Sample:**
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

At the time of calling this function, dump the input video for several seconds before and after. Due to the lack of encoding, it is necessary to dump the keyframes that all frames depend on within a few seconds before and after. Therefore, the dump implementation of the interface is based on gop, and the actual video duration under dump will be higher than the input parameter duration. The error depends on the gop of the input video. The larger the size and gop, the larger the error.

**Interface:**
    .. code-block:: c
    
        int dump(int dump_pre_seconds, int dump_post_seconds, std::string& file_path)

* dump_pre_seconds: int

Input parameter. Save the video several seconds before calling this interface.

* dump_post_seconds: int

Input parameter. Save the video a few seconds after the time this interface is called.

* file_path: std::string&

Input parameter. Video path.

**Returns:**

* judge_ret: int

Returns 0 if successful and other values if failed.

**Sample:**
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

Get pts or dts.

**Interface:**
    .. code-block:: c
    
        vector<double> get_pts_dts()

**Returns**

* result : int

the value of pts and dts.

**Sample:**
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