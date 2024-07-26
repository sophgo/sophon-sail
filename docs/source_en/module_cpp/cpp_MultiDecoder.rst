MultiDecoder
____________________

Multi-channel decoding interface supports decoding multiple channels of video at the same time.

Constructor
>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: c

        MultiDecoder(
            int queue_size=10, 
            int tpu_id=0, 
            int discard_mode=0);

**Parameters:**

* queue_size: int

Input parameter. For each channel of video, the length of the decoding cache image queue.

* tpu_id: int

Input parameter. The tpu id used, the default is 0.

* discard_mode: int

Input parameter. The data discarding strategy after the cache reaches the maximum size. 0 means not to put data into the cache; 1 means to take out the picture at the head of the queue from the queue, discard it, and then cache the decoded picture. Default is 0.


set_read_timeout
>>>>>>>>>>>>>>>>>>>>

Set the timeout for reading images, which takes effect on the read and read\_ interfaces. If the image is not obtained after the timeout, the result will be returned.

**Interface:**
    .. code-block:: c

        void set_read_timeout(int time_second);

**Parameters:**

* timeout: int

Input parameter. Timeout time, unit is seconds.

**Sample:**
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

Add a channel.

**Interface:**
    .. code-block:: c

        int add_channel(
            const std::string&  file_path, 
            int                 frame_skip_num=0);
            
**Parameters:**

* file_path: string

Input parameter. The path or link to the video.

* frame_skip_num: int

Input parameter. The number of active frame drops in the decoding cache. The default is 0, which means no active frame drops.

**Returns**

Returns the unique channel number corresponding to the video. Type is int.

**Sample:**
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

del_channel
>>>>>>>>>>>>>>>

Delete an already added video channel.

**Interface:**
    .. code-block:: c

        int del_channel(int channel_idx);

**Parameters:**

* channel_idx: int

Input parameter. The channel number of the video to be deleted.

**Returns:**

Returns 0 on success, other values indicate failure.

**Sample:**
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

Clear the image cache of the specified channel.

**Interface:**
    .. code-block:: c

        int clear_queue(int channel_idx);


**Parameters:**

* channel_idx: int

Input parameter. The channel number of the video to be deleted.

**Returns:**

Returns 0 on success, other values indicate failure.
            
**Sample:**
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

Get an image from the specified video channel.

**Interface 1:**
    .. code-block:: c

        int read(
            int         channel_idx,
            BMImage&    image,
            int         read_mode=0);

**Parameters 1:**

* channel_idx: int

Input parameter. Specified video channel number.

* image: BMImage

Output parameter. Decoded picture.

* read_mode: int

Input parameter. The mode for getting images, 0 means no waiting, read one directly from the cache, and it will return regardless of whether it is read or not. Others indicate waiting until the image is obtained before returning.

**Returns 1:**

Returns 0 on success, other values indicate failure.
  
**Sample:**
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

**Interface 2:**
    .. code-block:: c

        BMImage read(int channel_idx);

**Parameters 2:**

* channel_idx: int

Input parameter. Specified video channel number.

**返回值说明2:**

Returns the decoded image, type is BMImage.
            
**Sample:**
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

Get an image from the specified video channel, usually used with BMImageArray.

**Interface 1:**
    .. code-block:: c

        int read_(
            int         channel_idx,
            bm_image&   image,
            int         read_mode=0);

**Parameters 1:**

* channel_idx: int

Input parameter. Specified video channel number.

* image: bm_image

Output parameter. Decoded image.

* read_mode: int

Input parameter. The mode for getting images, 0 means no waiting, read one directly from the cache, and it will return regardless of whether it is read or not. Others indicate waiting until the image is obtained before returning.

**Returns 1:**

Returns 0 on success, other values indicate failure.

**Sample:**
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
            vector<vector<cv::Mat>> frame_list;  
            for (int i = 0; i < 4; i++) {  
                int idx = multiDecoder.add_channel("your_video_path");  
                if(idx<0) return -1;
                channel_list.push_back(idx);  
                frame_list.push_back(vector<cv::Mat>());  
            }  
            
            int count = 0;  
            while (true) {  
                count++;  
                for (int idx : channel_list) {  
                    sail::BMImage image;
                    sail::bm_image bmimg = image.data()
                    int ret = multiDecoder.read_(idx,(id,1);   
                }  
                if (count == 20) {  
                    break;  
                }  
            }  
            return 0;
        }        

**Interface 2:**
    .. code-block:: c

        bm_image read_(int channel_idx);

**Parameters 2:**

* channel_idx: int

Input parameter. Specified video channel number.

**Returns 2:**

Returns the decoded image, type bm_image.

**Sample:**
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

Reconnect video in the corresponding channel.

**Interface:**
    .. code-block:: c
        
        int reconnect(int channel_idx);

**Parameters:**

* channel_idx: int

Input parameter. Enter the channel number of the image.

**Returns**

Returns 0 on success, other values indicate failure.

**Sample:**
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

Get the image shape of the corresponding channel.

**Interface:**
    .. code-block:: c

        std::vector<int> get_frame_shape(int channel_idx);
            
**Parameters:**

Input parameters. Enter the channel number of the image.
        
**Returns:**

Returns a list consisting of 1, number of channels, image height, and image width.

**Sample:**
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

Set whether the video is a local video. If not called, the video is represented as a network video stream.

**Interface:**
    .. code-block:: c

        void set_local_flag(bool flag);
 
**Parameters:**

* flag: bool

Standard bit. If it is True, each video channel will be decoded at a fixed rate of 25 frames per second.

**Sample:**
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

**Interface**
    .. code-block:: c

        float get_channel_fps(int channel_idx):
 
**Parameters**

* channel_idx: int

The specified channel index

**Returns**

Returns the video fps of the specified channel

**Sample:**
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
                float fps = multiDecoder.get_channel_fps(idx);
            } 
            return 0;
        }

get_drop_num
>>>>>>>>>>>>>>>>>>

Obtain the number of dropped frames.

**Interface:**
    .. code-block:: c

        size_t get_drop_num(int channel_idx);
 
**Parameters:**

* channel_idx: int

The channel index of the input image.

**Sample:**
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

Set the number of dropped frames to num.

**Interface:**
    .. code-block:: c

        size_t reset_drop_num(int channel_idx);
 
**Parameters:**

* channel_idx: int

The channel index of the input image.

**Sample:**
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