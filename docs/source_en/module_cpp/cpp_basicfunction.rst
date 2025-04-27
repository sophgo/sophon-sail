Basic function
_________________


Mainly used to obtain or configure device information and properties.


get_available_tpu_num
>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the number of available Tensor Computing Processors in the current device.

**Interface:**
    .. code-block:: c

        int get_available_tpu_num();


**Returns:**

Returns the number of Tensor Computing Processors available in the current device.

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        
        int main() {  
            int tpu_len = sail::get_available_tpu_num();  
            std::cout << "available tpu: " << tpu_len << std::endl;  
            return 0;  
        }

set_print_flag
>>>>>>>>>>>>>>>>>>>>>>>>>>

Set whether to print the program's calculation time information.

**Interface:**
    .. code-block:: c

        int set_print_flag(bool print_flag);

**Returns:**

* print_flag: bool

When print_flag is True, the main time-consuming information of the calculation of program is printed, otherwise it is not printed.

**Sample:**
    .. code-block:: c
    
        #include <sail/engine_multi.h>
        
        int main() {  
            int ret = sail::set_print_flag(true);
            if (ret == 0){
                std::cout << "set print time success" << std::endl;
            }
            return 0;  
        }

set_dump_io_flag
>>>>>>>>>>>>>>>>>>>>>>>>>>

Set whether to store input data and output data.

**Interface:**
    .. code-block:: c
     
        int set_dump_io_flag(bool dump_io_flag);


**Parameter:**

* dump_io_flag: bool

When dump_io_flag is True, input data and output data are stored, otherwise they are not stored.

**Sample:**
    .. code-block:: c
    
        #include <sail/engine_multi.h>
        
        int main() {  
            ret = sail::set_dump_io_flag(true);
            if (ret == 0){
                std::cout << "set save data success" << std::endl;
            }
            return 0;  
        }

set_loglevel
>>>>>>>>>>>>>>>>>>>>>>>>>>

Set the logging level to the specified level.
Lower log levels are typically used in production environments to reduce performance overhead and the volume of log data, 
while higher log levels are suitable for development and debugging in order to capture more detailed information.

**Interface:**
    .. code-block:: c++

        int set_loglevel(LogLevel loglevel);


**Parameters:**

* loglevel: LogLevel

The Target log level as a sail.LogLevel enum value. 
The optional values include ``TRACE``, ``DEBUG``, ``INFO``, ``WARN``, ``ERR``, ``CRITICAL``, ``OFF``, and the default level is ``INFO``.

**Returns:**

return: int

Returning 0 indicates the log level was set successfully, 
whereas returning -1 indicates a failure due to an unknown log level.

**Sample:**
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

Set the parameters for the Decoder (including MutlDecoder) through environment variables. 
These must be set before the Decoder is constructed; otherwise, the default values will be used. 
This is mainly applicable to video decoding.

**Interface:**
    .. code-block:: c

        int set_decoder_env(std::string env_name, std::string env_value);
            

**Parameter:**

* env_name: string

The property name to set for the Decoder. The available property names are as follows:

        - *'rtsp_transport'*: The transport protocol used for RTSP. The default is TCP.
        - *'extra_frame_buffer_num'*: The maximum number of cached frames for the Decoder. The default is 5.
        - *'stimeout'*: Raise error timeout, in milliseconds. The default is 20000000, i.e., 20 seconds.
        - *'skip_non_idr'*: Decoding frame skip mode. 0, no skip; 1, skip Non-RAP frames; 2, skip non-reference frames. The default is 0.
        - *'fflags'*: format flags, like "nobuffer". Read ffmpeg official docs for more details.
        - *'rtsp_flags'*: Set RTSP flags. The default is prefer_tcp.
        - *'refcounted_frames'*: When set to 1, the decoded images need to be manually released by the program; when set to 0, they are automatically released by the Decoder.
        - *'probesize'*: the max size of the data to analyze to get stream information. 5000000 by default.
        - *'analyzeduration'*: How many microseconds are analyzed to probe the input. 5000000 by default.
        - *'buffer_size'*: The maximum socket buffer size in bytes.
        - *'max_delay'*: Maximum demuxing delay in microseconds.



* env_value: string

The configuration value of this property

**Sample:**
    .. code-block:: c

        #include <sail/cvwrapper.h>
        
        int main() {  
            sail::set_decoder_env("extra_frame_buffer_num", "3"); // Decrease buffer num for lower memory usage
            sail::set_decoder_env("probesize", "1024") // Decrease probesize for lower latency
            sail::set_decoder_env("skip_non_idr", "2") // skip non-reference frames
            int dev_id = 0;
            sail::Handle handle(dev_id);
            std::string video_path = "input_video.mp4";
            sail::Decoder decoder(video_path, true, dev_id);
            sail::BMImage bmimg = decoder.read(handle);
            return 0;
        }

base64_enc
>>>>>>>>>>>>>>>>>>>>>>>>

Base64 encode the data to generate the corresponding base64 encoded string. BM1688 and CV186AH PCIE mode are not supported.
    .. code-block:: c

        int base64_enc(Handle& handle, const void *data, uint32_t dlen, std::string& encoded);

**Parameter:**

* handle: Handle

The handle of the device.

* data: const void*

The pointer to the data to be encoded.

* dlen: uint32_t

The byte length of the data to be encoded.

* encoded: string

encoded The string generated by base64 encoding.

**Returns**

Return 0 on successful base64 encoding, otherwise return -1.

**Sample:**
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
>>>>>>>>>>>>>>>>>>>>>>>>
Base64 encode the data to generate the corresponding base64 encoded string. BM1688 and CV186AH PCIE mode are not supported.

    .. code-block:: c

        int base64_dec(Handle& handle, const void *data, uint32_t dlen, uint8_t* p_outbuf, uint32_t *p_size);

**Parameter:**

* handle: Handle

The handle of the device.

* data: const void*

The pointer to the data to be decoded.

* dlen: uint32_t

The byte length of the data to be decoded.

* p_outbuf: uint8_t*

Pointer to the decoded data.

* p_size: uint32_t *

Length of the pointer to the decoded data.

**Returns**

Return 0 on successful base64 encoding, otherwise return -1.

**Sample:**
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
            uint8_t out_data_buf[100]; // set enough space for decoded data
            uint32_t out_data_size; // decoded data length 
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

Get the processor utilization of the specified device

**Interface:**
    .. code-block:: c

        int get_tpu_util(int dev_id);

**Parameter:**

* dev_id: int

Device ID.

**Return:**

Returns the processor utilization of the device corresponding to the ID.

**Sample:**
    .. code-block:: c
    
        #include <sail/cvwrapper.h>

        int main() {  
            int tpu_util;
            tpu_util = sail::get_tpu_util(0); 
            std::cout << "tpu_util " << tpu_util << "%"<< std::endl;
            return 0;  
        }
        
get_vpu_util
>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the VPU percent utilization of the specified device

**Interface:**
    .. code-block:: c

        std::vector<int> get_tpu_util(int dev_id);



**Parameter:**

* dev_id: int

Device ID.

**Return:**

The vpu of bm1684 is 5-core, and the return value is a list of length 5. The vpu of bm1684x is 3-core, and the return value is a list of length 3.
Each integer in the List is the percent utilization of the corresponding core.

**Sample:**
    .. code-block:: c
    
        #include <sail/cvwrapper.h>
        
        int main() {  
            std::vector<int> vpu_util;
            vpu_util = sail::get_vpu_util(0); 

            for(int i = 0; i < vpu_util.size(); i++) {
                std::cout << "VPU ID: " << i << ", Util Value: " << vpu_util[i] << "%" << std::endl;
            }
            return 0;  
        }

get_vpp_util
>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the VPP utilization of the specified device

**Interface:**
    .. code-block:: c

        std::vector<int> get_vpp_util(int dev_id);



**Parameter:**

* dev_id: int

Device ID.

**Return:**

The vpp of bm1684 and bm1684x are both 2-core, and the return value is a list of length 2.
Each integer in the List is the percent utilization of the corresponding core.

**Sample:**
    .. code-block:: c
    
        #include <sail/cvwrapper.h>
        
        int main() {  
            std::vector<int> vpp_util;
            vpp_util = sail::get_vpu_util(0); 

            for(int i = 0; i < vpp_util.size(); i++) {
                std::cout << "VPU ID: " << i << ", Util Value: " << vpp_util[i] << "%" << std::endl;
            }
            return 0;  
        }
        
get_board_temp
>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the temperature of the board.

**Interface:**
    .. code-block:: c

        int get_board_temp(int dev_id);

**Parameter:**

* dev_id: int

Device ID.

**Return:**

The board temperature for the corresponding card, with the default unit in Celsius (°C)

**Sample:**
    .. code-block:: c
    
        #include <sail/cvwrapper.h>
        
        int main() {  
            int board_temp;
            board_temp = sail::get_board_temp(0); 
            std::cout << "board_temp " << board_temp << "°C"<< std::endl;
            return 0;  
        }


get_chip_temp
>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the temperature of the chip.

**Interface:**
    .. code-block:: c

        int get_chip_temp(int dev_id);

**Parameter:**

* dev_id: int

Device ID.

**Return:**

The processor temperature for the corresponding card, with the default unit in Celsius (°C)

**Sample:**
    .. code-block:: c
    
        #include <sail/cvwrapper.h>
        
        int main() {  
            int chip_temp;
            chip_temp = sail::get_chip_temp(0); 
            std::cout << "chip_temp " << bchip_temp << "℃"<< std::endl;
            return 0;  
        }


get_dev_stat
>>>>>>>>>>>>>>>>>>>>>>>>>>

Get device memory information.

**Interface:**
    .. code-block:: c

        int get_dev_stat(int dev_id);

**Parameter:**

* dev_id: int

Device ID.

**Return:**

A list of memory information for the corresponding device: [mem_total, mem_used, tpu_util].

**Sample:**
    .. code-block:: c
    
        #include <sail/cvwrapper.h>
        
        int main() {  
            std::vector<int> dev_stat;
            dev_stat = sail::get_dev_stat(0); 

            std::cout << "mem_total: " << dev_stat[0] << " MB" << std::endl;
            std::cout << "mem_used: " << dev_stat[1] << " MB" << std::endl;
            std::cout << "tpu_util: " << dev_stat[2] << " %" << std::endl;
            return 0;  
        }