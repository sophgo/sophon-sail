Basic function
_________________


主要用于获取或配置设备信息与属性。


get_available_tpu_num
>>>>>>>>>>>>>>>>>>>>>>>>>>

获取当前设备中可用智能视觉深度学习处理器的数量。

**接口形式:**
    .. code-block:: c

        int get_available_tpu_num();


**返回值说明:**

返回当前设备中可用智能视觉深度学习处理器的数量。

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        
        int main() {  
            int tpu_len = sail::get_available_tpu_num();  
            std::cout << "available tpu: " << tpu_len << std::endl;  
            return 0;  
        }

set_print_flag
>>>>>>>>>>>>>>>>>>>>>>>>>>

设置是否打印程序的计算耗时信息。

**接口形式:**
    .. code-block:: c

        int set_print_flag(bool print_flag);

**参数说明:**

* print_flag: bool

print_flag为True时，打印程序的计算主要的耗时信息，否则不打印。

**示例代码:**
    .. code-block:: c
    
        #include <sail/cvwrapper.h>
        
        int main() {  
            int ret = sail::set_print_flag(true);
            if (ret == 0){
                std::cout << "set print time success" << std::endl;
            }
            return 0;  
        }

set_dump_io_flag
>>>>>>>>>>>>>>>>>>>>>>>>>>

设置是否存储输入数据和输出数据。
每个输入或输出Tensor数据将被分别保存为一个独立的二进制文件。文件名的前缀为时间戳。

**接口形式:**
    .. code-block:: c
     
        int set_dump_io_flag(bool dump_io_flag);


**参数说明:**

* dump_io_flag: bool

dump_io_flag为True时，存储输入数据和输出数据，否则不存储。

**示例代码:**
    .. code-block:: c
    
        #include <sail/cvwrapper.h>
        
        int main() {  
            ret = sail::set_dump_io_flag(true);
            if (ret == 0){
                std::cout << "set save data success" << std::endl;
            }
            return 0;  
        }

set_loglevel
>>>>>>>>>>>>>>>>>>>>>>>>>>

设置运行过程中的日志级别为指定级别。
较低的日志等级通常用于生产环境，以减少性能开销和日志数据量，而较高的日志等级则适用于开发和调试，以便能够记录更详细的信息。

**接口形式:**
    .. code-block:: c++

        int set_loglevel(LogLevel loglevel);


**参数说明:**

* loglevel: LogLevel

期望的日志级别，为 ``sail::LogLevel`` 枚举值。可选的级别包括 ``TRACE``、``DEBUG``、``INFO``、``WARN``、``ERR``、``CRITICAL``、``OFF``，默认级别为 ``INFO``。

**返回值说明:**

返回类型：int

0：日志级别设置成功。
-1：传入了未知的日志级别，设置失败。

**示例代码:**
    .. code-block:: c++
    
        #include <sail/cvwrapper.h>
        
        int main() {
            int ret = sail::set_loglevel(sail::LogLevel::TRACE);
            if (ret == 0){
                std::cout << "Set log level successfully" << std::endl;
            }
            else{
                std::cout << "Unknown log level, set failed." << std::endl;
            }
            return 0;
        }


set_decoder_env
>>>>>>>>>>>>>>>>>>>>>>>>>>

通过环境变量设置Decoder（也包括MultiDecoder）的参数。
必须在Decoder构造前设置，否则使用默认值。主要适用于视频解码。

**接口形式:**
    .. code-block:: c

        int set_decoder_env(std::string env_name, std::string env_value);
            

**参数说明:**

* env_name: string

选择设置Decoder的属性名称，可选的属性名称有：

        - *'rtsp_transport'* 设置RTSP采用的传输协议。默认为TCP。
        - *'extra_frame_buffer_num'* 设置Decoder的最大缓存帧数。默认为5。
        - *'stimeout'* 设置阻塞超时时间，单位为ms。默认为20000000，即20s。
        - *'skip_non_idr'* 解码跳帧模式。0，关闭跳帧；1，跳过Non-RAP帧；2，跳过非参考帧。默认为0。
        - *'fflags'* 格式相关的flag。比如"nobuffer"。详细信息请参考ffmpeg官方文档。
        - *'rtsp_flags'* 设置RTSP是否自定义IO。默认为prefer_tcp。
        - *'refcounted_frames'* 是否使用引用计数机制。设置为1时，解码出来的图像需要程序手动释放，为0时由Decoder自动释放。
        - *'probesize'* 解析视频流时读取的最大字节数。默认为5000000。
        - *'analyzeduration'* 解析文件时读取的最大时长，单位为ms。默认为5000000。
        - *'buffer_size'* 设置缓存大小。
        - *'max_delay'* 设置最大时延。


* env_value: string

该属性的配置值

**示例代码:**
    .. code-block:: c

        #include <sail/cvwrapper.h>

        int main() {  
            sail::set_decoder_env("extra_frame_buffer_num", "3"); // 减小buffer以降低内存占用
            sail::set_decoder_env("probesize", "1024") // 减小probesize以降低拉流延迟
            sail::set_decoder_env("skip_non_idr", "2") // 跳过非参考帧
            int dev_id = 0;
            sail::Handle handle(dev_id);
            std::string video_path = "input_video.mp4";
            sail::Decoder decoder(video_path, true, dev_id);
            sail::BMImage bmimg = decoder.read(handle);
            return 0;
        }

base64_enc
>>>>>>>>>>>>>>>>>>>>>>>>>>

对数据进行base64编码，生成的对应的base64编码后的字符串。不支持BM1688和CV186AH PCIE模式。

**接口形式:**
    .. code-block:: c

        int base64_enc(Handle& handle, const void *data, uint32_t dlen, std::string& encoded);
            

**参数说明:**

* handle: Handle

设备的handle句柄，使用Handle(dev_id)创建

* data: void*

待编码的数据指针

* dlen: uint32_t

待编码的数据字节长度

* encoded: string

base64编码生成的字符串

**返回值说明**

base64编码成功返回0，否则返回-1

**示例代码:**
    .. code-block:: c
    
        #include <sail/base64.h>
        
        int main() {  
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);

            std::string data = "hello,world!";

            // base64 encode
            std::string base64_encoded;
            uint32_t dlen = data.length();
            ret = sail::base64_enc(handle, data.c_str(), dlen, base64_encoded);
            if (ret == 0){
                std::cout << dlen << std::endl;
                std::cout << "base64 encode success!" << "based 64:" << base64_encoded << " lens" << dlen << std::endl;
            }
            return 0;  
        }
        
base64_dec
>>>>>>>>>>>>>>>>>>>>>>>>>>

对数据进行base64编码，生成的对应的base64编码后的字符串。示例代码请参考base64_dec接口用法。不支持BM1688和CV186AH PCIE模式。

**接口形式:**
    .. code-block:: c

        int base64_dec(Handle& handle, const void *data, uint32_t dlen, uint8_t* p_outbuf, uint32_t *p_size);
            

**参数说明:**

* handle: Handle

设备的handle句柄，使用Handle(dev_id)创建

* data: void*

待解码的数据指针

* dlen: uint32_t

待解码的数据字节长度

* p_outbuf: uint8_t*

解码后的数据buffer

* p_size: uint32_t

输出数据。解码后的数据指针长度

**返回值说明**

base64解码成功返回0，否则返回-1

**示例代码:**
    .. code-block:: cpp
        
        #include <sail/base64.h>
        
        int main() {  
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);

            std::string data = "hello,world!";

            // base64 encode
            std::string base64_encoded;
            uint32_t dlen = data.length();
            ret = sail::base64_enc(handle, data.c_str(), dlen, base64_encoded);
            if (ret == 0){
                std::cout << dlen << std::endl;
                std::cout << "base64 encode success!" << "based 64:" << base64_encoded << "lens" << dlen << std::endl;
            }

            // base64_dec
            uint32_t dlen_based = base64_encoded.length();
            uint8_t out_data_buf[100]; // 假设有足够大的空间存放解码后的数据
            uint32_t out_data_size; // 用于存放解码后数据的长度
            ret =sail::base64_dec(handle, base64_encoded.c_str(), dlen_based, out_data_buf, &out_data_size);
            if (ret == 0){
                std::cout << "base64 decode success,data size is:" << out_data_size << std::endl;
                for(uint32_t i = 0; i < out_data_size; i++) {
                    std::cout << out_data_buf[i];
                }
                std::cout << std::endl;
            }
            return 0;
        }


get_tpu_util
>>>>>>>>>>>>>>>>>>>>>>>>>>

获取对应设备的处理器使用率

**接口形式:**
    .. code-block:: c

        int get_tpu_util(int dev_id);



**参数说明:**

* dev_id: int

需要获取处理器使用率的设备的ID。

**返回值说明:**

返回对应设备的处理器的使用率百分比。

**示例代码:**
    .. code-block:: c
    
        #include <sail/cvwrapper.h>
        
        int main() {  
            int tpu_util;
            tpu_util = sail::get_tpu_util(0); //获取dev0的处理器使用率
            std::cout << "tpu_util " << tpu_util << "%"<< std::endl;
            return 0;  
        }
        
get_vpu_util
>>>>>>>>>>>>>>>>>>>>>>>>>>

获取对应设备的VPU使用率

**接口形式:**
    .. code-block:: c

        std::vector<int> get_vpu_util(int dev_id);

**参数说明:**

* dev_id: int

需要获取VPU使用率的设备的ID。

**返回值说明:**

bm1684为5核vpu，返回值为长度为5的List，bm1684x为3核vpu， 返回值为长度为3的List。
List中的每项数据为对应核心的使用率百分比。

**示例代码:**
    .. code-block:: c
    
        #include <sail/cvwrapper.h>
        
        int main() {  
            std::vector<int> vpu_util;
            vpu_util = sail::get_vpu_util(0); //获取dev0的vpu处理器使用率

            for(int i = 0; i < vpu_util.size(); i++) {
                std::cout << "VPU ID: " << i << ", Util Value: " << vpu_util[i] << "%" << std::endl;
            }
            return 0;  
        }
        
get_vpp_util
>>>>>>>>>>>>>>>>>>>>>>>>>>

获取对应设备的VPP使用率

**接口形式:**
    .. code-block:: c

        std::vector<int> get_vpp_util(int dev_id);



**参数说明:**

* dev_id: int

需要获取VPP使用率的设备的ID。

**返回值说明:**

bm1684与bm1684x均为2核vpp，返回值为长度为2的List。
List中的每项数据为对应核心的使用率百分比。

**示例代码:**
    .. code-block:: c
    
        #include <sail/cvwrapper.h>
        
        int main() {  
            std::vector<int> vpp_util;
            vpp_util = sail::get_vpu_util(0); //获取dev0的vpu处理器使用率

            for(int i = 0; i < vpp_util.size(); i++) {
                std::cout << "VPU ID: " << i << ", Util Value: " << vpp_util[i] << "%" << std::endl;
            }
            return 0;  
        }
        
get_board_temp
>>>>>>>>>>>>>>>>>>>>>>>>>>

获取对应设备的板级温度。只支持PCIe模式。

**接口形式:**
    .. code-block:: c

        int get_board_temp(int dev_id);
        
**参数说明:**

* dev_id: int

需要获取对应板卡所在设备的ID。

**返回值说明:**

返回对应板卡的板级温度，默认单位摄氏度（℃）

**示例代码:**
    .. code-block:: c
      
        #include <sail/cvwrapper.h>
        
        int main() {  
            int board_temp;
            board_temp = sail::get_board_temp(0); 
            std::cout << "board_temp " << board_temp << "℃"<< std::endl;
            return 0;  
        }

get_chip_temp
>>>>>>>>>>>>>>>>>>>>>>>>>>

获取对应设备的处理器温度。只支持PCIe模式。

**接口形式:**
    .. code-block:: c

        int get_chip_temp(int dev_id);
        
**参数说明:**

* dev_id: int

需要获取对应板卡所在设备的ID。

**返回值说明:**

返回对应设备的处理器的温度，默认单位摄氏度（℃）。

**示例代码:**
    .. code-block:: c
    
        #include <sail/cvwrapper.h>
        
        int main() {  
            int chip_temp;
            chip_temp = sail::get_chip_temp(0); 
            std::cout << "chip_temp " << chip_temp << "℃"<< std::endl;
            return 0;  
        }

get_dev_stat
>>>>>>>>>>>>>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: c

        std::vector<int> get_dev_stat(int dev_id);
        
**参数说明:**

* dev_id: int

需要获取对应板卡所在设备的ID。

**返回值说明:**

返回对应设备的内存信息列表:[mem_total,mem_used,tpu_util]。

**示例代码:**
    .. code-block:: c
    
        #include <iostream>
        #include "cvwrapper.h"
        
        int main() {  
            std::vector<int> dev_stat;
            dev_stat = sail::get_dev_stat(0); 

            std::cout << "mem_total: " << dev_stat[0] << " MB" << std::endl;
            std::cout << "mem_used: " << dev_stat[1] << " MB" << std::endl;
            std::cout << "tpu_util: " << dev_stat[2] << " %" << std::endl;
            return 0;  
        }
