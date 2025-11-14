MultiDecoder
____________________

多路解码接口，支持同时解码多路视频。

构造函数
>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: c

        MultiDecoder(
            int queue_size=10, 
            int tpu_id=0, 
            int discard_mode=0);

**参数说明:**

* queue_size: int

输入参数。每路视频，解码缓存图像队列的长度。

* tpu_id: int

输入参数。使用的智能视觉深度学习处理器 id，默认为0。

* discard_mode: int

输入参数。缓存达到最大值之后，数据的丢弃策略。0表示不再放数据进缓存；1表示先从队列中取出队列头的图片，丢弃之后再讲解码出来的图片缓存进去。默认为0。


set_read_timeout
>>>>>>>>>>>>>>>>>>>>

设置读取图片的超时时间，对read和read_接口生效，超时之后仍然没有获取到图像，结果就会返回。

**接口形式:**
    .. code-block:: c

        void set_read_timeout(int time_second);

**参数说明:**

* timeout: int

输入参数。超时时间，单位是秒。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/decoder_multi.h>

        using namespace std;  

        int main() {  
            int queue_size = 16;
            int dev_id = 0;
            int discard_mode = 0;
            sail::MultiDecoder multiDecoder(queue_size, dev_id, discard_mode);  
            multiDecoder.set_read_timeout(100);
            return 0;
        }
            
add_channel
>>>>>>>>>>>>>>>>

添加一个通道。

**接口形式1:**

通过该接口添加的通道，不会自动停止解码，会自动循环，直到该MultiDecoder析构，或者调用del_channel。

    .. code-block:: c

        int add_channel(
            const std::string&  file_path, 
            int                 frame_skip_num=0);
            
**参数说明1:**

* file_path: string

输入参数。视频的路径或者链接。

* frame_skip_num: int

输入参数。解码缓存的主动丢帧数，默认是0，不主动丢帧。

**返回值说明1**

返回视频对应的唯一的通道号。类型为整形。

**示例代码1:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/decoder_multi.h>

        using namespace std;  

        int main() {  
            int queue_size = 16;
            int dev_id = 0;
            int discard_mode = 0;
            sail::MultiDecoder multiDecoder(queue_size, dev_id, discard_mode); 
            vector<int> channel_list;  
            for (int i = 0; i < 4; i++) {  
                int idx = multiDecoder.add_channel("your_video_path");  
                if(idx<0) return -1;
                channel_list.push_back(idx);   
            }   
            return 0;
        }

**接口形式2:**

该接口添加通道时，支持设置循环次数。仅适用于解码本地视频文件的场景。

    .. code-block:: c

        int add_channel(
            const std::string&  file_path, 
            int                 frame_skip_num
            int                 loopnum);
            
**参数说明2:**

* file_path: string

输入参数。视频的路径或者链接。

* frame_skip_num: int

输入参数。解码缓存的主动丢帧数。设置为0表示不主动丢帧。

* loopnum: int

输入参数。解码循环次数。设置为0表示不循环，解码一遍后停止。

**返回值说明**

返回视频对应的唯一的通道号。类型为整形。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/decoder_multi.h>

        using namespace std;  

        int main() {  
            int queue_size = 16;
            int dev_id = 0;
            int discard_mode = 0;
            sail::MultiDecoder multiDecoder(queue_size, dev_id, discard_mode); 
            vector<int> channel_list;
            std::string file_path = "your_video_path";
            int frame_skip_num = 0;
            int loopnum = 0;
            for (int i = 0; i < 4; i++) {  
                int idx = multiDecoder.add_channel(file_path, frame_skip_num, loopnum);  
                if(idx<0) return -1;
                channel_list.push_back(idx);   
            }   
            return 0;
        }

del_channel
>>>>>>>>>>>>>>>

删除一个已经添加的视频通道。

**接口形式:**
    .. code-block:: c

        int del_channel(int channel_idx);

**参数说明:**

* channel_idx: int

输入参数。将要删除视频的通道号。

**返回值说明**

成功返回0，其他值时表示失败。


**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/decoder_multi.h>

        using namespace std;  

        int main() {  
            int queue_size = 16;
            int dev_id = 0;
            int discard_mode = 0;
            sail::MultiDecoder multiDecoder(queue_size, dev_id, discard_mode);  
            vector<int> channel_list;  
            for (int i = 0; i < 4; i++) {  
                int idx = multiDecoder.add_channel("your_video_path");
                if(idx<0) return -1;
                channel_list.push_back(idx);   
            } 
            int ret = multiDecoder.del_channel(0);
            if (ret!=0) {
                cout << "Failed!" << endl;
                return -1;
            }
            return 0;
        }

clear_queue
>>>>>>>>>>>>>>>

清除指定通道的图片缓存。

**接口形式:**
    .. code-block:: c

        int clear_queue(int channel_idx);


**参数说明:**

* channel_idx: int

输入参数。将要删除视频的通道号。

**返回值说明:**

成功返回0，其他值时表示失败。 

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/decoder_multi.h>

        using namespace std;  

        int main() {  
            int queue_size = 16;
            int dev_id = 0;
            int discard_mode = 0;
            sail::MultiDecoder multiDecoder(queue_size, dev_id, discard_mode);  
            vector<int> channel_list;  
            for (int i = 0; i < 4; i++) {  
                int idx = multiDecoder.add_channel("your_video_path");
                if(idx<0) return -1; 
                channel_list.push_back(idx);   
            } 
            int ret = multiDecoder.clear_queue(0);
            if (ret!=0) {
                cout << "Failed!" << endl;
                return -1;
            }
            return 0;
        }       

read
>>>>>>>>

从指定的视频通道中获取一张图片。

**接口形式1:**
    .. code-block:: c

        int read(
            int         channel_idx,
            BMImage&    image,
            int         read_mode=0);

**参数说明1:**

* channel_idx: int

输入参数。指定的视频通道号。

* image: BMImage

输出参数。解码出来的图片。

* read_mode: int

输入参数。获取图片的模式，0表示不等待，直接从缓存中读取一张，无论有没有读取到都会返回。其他的表示等到获取到图片之后或等待时间超时再返回。

**返回值说明1:**

成功返回0，其他值时表示失败。 
  
**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/decoder_multi.h>

        using namespace std;  

        int main() {  
            int queue_size = 16;
            int dev_id = 0;
            int discard_mode = 0;
            sail::MultiDecoder multiDecoder(queue_size, dev_id, discard_mode);  
            vector<int> channel_list;  
            for (int i = 0; i < 4; i++) {  
                int idx = multiDecoder.add_channel("your_video_path");  
                if(idx<0) return -1;
                channel_list.push_back(idx);  
            }  
            
            int count = 0;  
            while (true) {  
                count++;  
                for (int idx : channel_list) {  
                    sail::BMImage bmimg;  
                    int ret = multiDecoder.read(idx, bmimg, 1);   
                }  
                if (count == 20) {  
                    break;  
                }  
            }  
            return 0;
        }

**接口形式2:**
    .. code-block:: c

        BMImage read(int channel_idx);

**参数说明2:**

* channel_idx: int

输入参数。指定的视频通道号。

**返回值说明2:**

返回解码出来的图片，类型为BMImage。  

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/decoder_multi.h>
        
        using namespace std;  

        int main() {  
            int queue_size = 16;
            int dev_id = 0;
            int discard_mode = 0;
            sail::MultiDecoder multiDecoder(queue_size, dev_id, discard_mode);  
            vector<int> channel_list;  
            for (int i = 0; i < 4; i++) {  
                int idx = multiDecoder.add_channel("your_video_path");
                if(idx<0) return -1;  
                channel_list.push_back(idx);  
            }  
            
            int count = 0;  
            while (true) {  
                count++;  
                for (int idx : channel_list) {  
                    sail::BMImage bmimg = multiDecoder.read(idx);    
                }  
                if (count == 20) {  
                    break;  
                }  
            }  
            return 0;
        }        

read\_
>>>>>>>>

从指定的视频通道中获取一张图片，通常是要和BMImageArray一起使用。

**接口形式1:**
    .. code-block:: c

        int read_(
            int         channel_idx,
            bm_image&   image,
            int         read_mode=0);

**参数说明1:**

* channel_idx: int

输入参数。指定的视频通道号。

* image: bm_image

输出参数。解码出来的图片。

* read_mode: int

输入参数。获取图片的模式，0表示不等待，直接从缓存中读取一张，无论有没有读取到都会返回。其他的表示等到获取到图片之后或等待时间超时再返回。

**返回值说明1:**

成功返回0，其他值时表示失败。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/decoder_multi.h>
        
        using namespace std;  

        int main() {  
            int queue_size = 16;
            int dev_id = 0;
            int discard_mode = 0;
            sail::MultiDecoder multiDecoder(queue_size, dev_id, discard_mode);  
            vector<int> channel_list;  
            for (int i = 0; i < 4; i++) {  
                int idx = multiDecoder.add_channel("your_video_path");  
                if(idx<0) return -1;
                channel_list.push_back(idx);  
            }  
            
            int count = 0;  
            while (true) {  
                count++;  
                for (int idx : channel_list) {  
                    sail::BMImage image;
                    bm_image bmimg = image.data()
                    int ret = multiDecoder.read_(idx,bmimg,1);   
                }  
                if (count == 20) {  
                    break;  
                }  
            }  
            return 0;
        }        

**接口形式2:**
    .. code-block:: c

        bm_image read_(int channel_idx);

**参数说明2:**

* channel_idx: int

输入参数。指定的视频通道号。

**返回值说明2:**

返回解码出来的图片，类型为bm_image。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/decoder_multi.h>
        
        using namespace std;  

        int main() {  
            int queue_size = 16;
            int dev_id = 0;
            int discard_mode = 0;
            sail::MultiDecoder multiDecoder(queue_size, dev_id, discard_mode);  
            vector<int> channel_list;   
            for (int i = 0; i < 4; i++) {  
                int idx = multiDecoder.add_channel("your_video_path");  
                if(idx<0) return -1;
                channel_list.push_back(idx); 
            }  
            int count = 0;  
            while (true) {  
                count++;  
                for (int idx : channel_list) {  
                    bm_image bmimg =  multiDecoder.read_(idx);
                } 
                if (count == 20) {  
                    break;  
                }  
            }  
            return 0;
        }  

reconnect
>>>>>>>>>>>>>>

重连相应的通道的视频。

**接口形式:**
    .. code-block:: c
        
        int reconnect(int channel_idx);

**参数说明:**

* channel_idx: int

输入参数。输入图像的通道号。

**返回值说明**

成功返回0，其他值时表示失败。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/decoder_multi.h>
        using namespace std;  

        int main() {  
            int queue_size = 16;
            int dev_id = 0;
            int discard_mode = 0;
            sail::MultiDecoder multiDecoder(queue_size, dev_id, discard_mode);  
            vector<int> channel_list;  
            for (int i = 0; i < 4; i++) {  
                int idx = multiDecoder.add_channel("your_video_path");  
                if(idx<0) return -1;
                channel_list.push_back(idx);   
            } 
            int ret = multiDecoder.reconnect(0);
            if (ret!=0) {
                cout << "Failed!" << endl;
                return -1;
            }
            return 0;
        }

get_frame_shape
>>>>>>>>>>>>>>>>>>

获取相应通道的图像shape。

**接口形式:**
    .. code-block:: c

        std::vector<int> get_frame_shape(int channel_idx);
            
**参数说明:**

输入参数。输入图像的通道号。
        
**返回值说明**

返回一个由1，通道数，图像高度，图像宽度组成的list。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/decoder_multi.h>

        using namespace std;  

        int main() {  
            int queue_size = 16;
            int dev_id = 0;
            int discard_mode = 0;
            sail::MultiDecoder multiDecoder(queue_size, dev_id, discard_mode);  
            vector<int> channel_list;  
            for (int i = 0; i < 4; i++) {  
                int idx = multiDecoder.add_channel("your_video_path");  
                if(idx<0) return -1;
                channel_list.push_back(idx);   
                vector<int> shape = multiDecoder.get_frame_shape(idx);
            } 
            return 0;
        }

set_local_flag
>>>>>>>>>>>>>>>>>>

设置视频是否为本地视频。如果不调用则表示为视频为网络视频流。

**接口形式:**
    .. code-block:: c

        void set_local_flag(bool flag);
 
**参数说明:**

* flag: bool

标准位，如果为True，每路视频每秒固定解码25帧

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/decoder_multi.h>
        using namespace std;  

        int main() {  
            int queue_size = 16;
            int dev_id = 0;
            int discard_mode = 0;
            sail::MultiDecoder multiDecoder(queue_size, dev_id, discard_mode);  
            multiDecoder.set_local_flag(true);
            return 0;
        }

get_channel_fps
>>>>>>>>>>>>>>>>>>

Get the video fps of the specified channel

**接口形式:**
    .. code-block:: c

        float get_channel_fps(int channel_idx):
 
**参数说明:**

* channel_idx: int

指定需要获取视频帧数的视频通道号

**返回值说明**

返回指定视频通道的视频帧数

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/decoder_multi.h>

        using namespace std;  

        int main() {  
            int queue_size = 16;
            int dev_id = 0;
            int discard_mode = 0;
            sail::MultiDecoder  multiDecoder(queue_size, dev_id, discard_mode);  
            vector<int> channel_list;  
            for (int i = 0; i < 4; i++) {  
                int idx = multiDecoder.add_channel("your_video_path");  
                if(idx<0) return -1;
                channel_list.push_back(idx);   
                float fps = multiDecoder.get_channel_fps(idx);
            } 
            return 0;
        }

get_drop_num
>>>>>>>>>>>>>>>>>>

获取丢帧数。

**接口形式:**
    .. code-block:: c

        size_t get_drop_num(int channel_idx);
            
**参数说明:**

输入参数。输入图像的通道号。
        
**返回值说明**

返回一个数代表丢帧数

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/decoder_multi.h>

        using namespace std;  

        int main() {  
            int queue_size = 16;
            int dev_id = 0;
            int discard_mode = 0;
            sail::MultiDecoder multiDecoder(queue_size, dev_id, discard_mode);  
            vector<int> channel_list;  
            for (int i = 0; i < 4; i++) {  
                int idx = multiDecoder.add_channel("your_video_path");  
                if(idx<0) return -1;
                channel_list.push_back(idx);   
                size_t ret = multiDecoder.get_drop_num(idx);
            } 
            return 0;
        }
        
reset_drop_num
>>>>>>>>>>>>>>>>>>

设置丢帧数为0。

**接口形式:**
    .. code-block:: c

        void reset_drop_num(int channel_idx);
 
**参数说明:**

输入参数。输入图像的通道号。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        #include <sail/decoder_multi.h>
        using namespace std;  

        int main() {  
            int queue_size = 16;
            int dev_id = 0;
            int discard_mode = 0;
            sail::MultiDecoder multiDecoder(queue_size, dev_id, discard_mode);  
            vector<int> channel_list;  
            for (int i = 0; i < 4; i++) {  
                int idx = multiDecoder.add_channel("your_video_path");
                if(idx<0) return -1;
                channel_list.push_back(idx);   
                multiDecoder.reset_drop_num(idx);
            } 
            return 0;
        }

get_channel_status
>>>>>>>>>>>>>>>>>>

获取指定通道的解码器状态。

**接口形式:**
    .. code-block:: cpp

        sail::DecoderStatus get_channel_status(int channel_idx) const

**参数说明:**

    - ``channel_idx`` (int): 要查询状态的通道索引。

**返回值说明:**

    返回一个 ``sail::DecoderStatus`` 枚举值，表示指定通道的解码器状态。
    比如 ``sail::DecoderStatus::OPENED`` 或 ``sail::DecoderStatus::CLOSED``。

**示例代码:**
    .. code-block:: cpp

        #include <sail/cvwrapper.h>
        #include <sail/decoder_multi.h>

        int main() {
            int queue_size = 10;
            int dev_id = 0;
            int discard_mode = 0;
            sail::MultiDecoder multiDecoder(queue_size, dev_id, discard_mode);
            std::vector<int> channel_list;
            std::vector<std::string> file_path_list(4, "your_video_path");

            for (int i = 0; i < 4; ++i) {
                int idx = multiDecoder.add_channel(file_path_list.at(i));
                if (idx < 0) {
                    std::cout << "Failed to add channel " << i << ". Error: " << idx << std::endl;
                    continue;
                }

                channel_list.push_back(idx);
                sail::DecoderStatus status = multiDecoder.get_channel_status(idx);
                std::cout << "Channel " << i << " status: " << status << std::endl;
            }

            return 0;
        }

is_channel_eof
>>>>>>>>>>>>>>>>>>

查询某个通道的解码器是否已经到达文件结尾。

**接口形式:**
    .. code-block:: cpp

        bool is_channel_eof(int channel_idx) const

**参数说明:**

    - ``channel_idx`` : 要查询状态的通道索引。

**返回值说明:**

    如果该通道已经到达文件结尾（EOF, end of file），则返回True，否则返回False。
    如果不存在索引对应的通道，则抛出异常。

**示例代码:**
    .. code-block:: cpp

        #include <sail/cvwrapper.h>
        #include <sail/decoder_multi.h>

        int main() {
            std::string filepath{"./jellyfish_200frames.mkv"};
            sail::MultiDecoder md;
            md.set_local_flag(true);
            int frame_skip_num = 0;
            int loopnum = 0;  // no loop
            auto idx = md.add_channel(filepath, frame_skip_num, loopnum);
            if (idx != 0) {
                std::cerr << "add_channel fail, ret: " << idx << std::endl;
                return 1;
            }
            auto cnt = 0;
            while (true) {
                sail::BMImage img;
                int read_mode = 1;  // wait block
                auto ret = md.read(idx, img, read_mode);
                if (ret != 0) {
                    std::cout << "===================="
                            << "channel status: "
                            << static_cast<int>(md.get_channel_status(idx))
                            << ", channel eof: " << md.is_channel_eof(idx)
                            << std::endl;
                    if (md.get_channel_status(idx) == sail::DecoderStatus::CLOSED &&
                        md.is_channel_eof(idx)) {
                        std::cout << "channel " << idx << " reached EOF, total read "
                                << cnt << " images, decode thread will stop"
                                << std::endl;
                        break;
                    } else {
                        std::cout << "channel " << idx << " meet an error, total" << cnt
                                << " images, ret = " << ret << std::endl;
                    }
                } else {
                    cnt += 1;
                    std::cout << "channel " << idx << " read " << cnt << " images"
                            << std::endl;
                }
            }
            return 0;
        }