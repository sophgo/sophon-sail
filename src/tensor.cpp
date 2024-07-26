/* Copyright 2016-2022 by SOPHON Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

==============================================================================*/

#include <cstring>
#include <numeric>
#include <functional>
#include <fstream>
#include <iomanip> 
#include <regex>
#include <vector>
#include "bmlib_runtime.h"
#include "tensor.h"
#include "internal.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif  // _WIND32


namespace sail {

    int get_board_temp(int dev_id){
        bm_handle_t handle;
        bm_dev_request(&handle, dev_id);
        unsigned int board_temp=0;
        bm_get_board_temp(handle, &board_temp);
        return board_temp;
    }

    int get_chip_temp(int dev_id){
        bm_handle_t handle;
        bm_dev_request(&handle, dev_id);

        unsigned int chip_temp=0;
        bm_get_chip_temp(handle, &chip_temp);
        return chip_temp;
    }

    std::vector<int> get_dev_stat(int dev_id) {
        bm_dev_stat_t stat;
        bm_handle_t handle;
        
        bm_dev_request(&handle, dev_id);
        bm_get_stat(handle, &stat);
        std::vector<int> res={stat.mem_total, stat.mem_used,stat.tpu_util};
        return res;
    }

    int get_available_tpu_num() {
        int count = 0;
        bm_dev_getcount(&count);
        return count;
    }

    int get_tpu_util(int dev_id) {
        bm_dev_stat_t stat;
        bm_handle_t handle;

        bm_dev_request(&handle, dev_id);
        bm_get_stat(handle, &stat);

        return stat.tpu_util;
    }

    extern "C"{
        bm_status_t bm_get_vpu_instant_usage(bm_handle_t handle, int *vpu_usage) __attribute__((weak));
        bm_status_t bm_get_vpp_instant_usage(bm_handle_t handle, int *vpu_usage) __attribute__((weak));
    }
    std::vector<int> get_vpu_util(int dev_id){
        std::vector<int> res;
        int res_len;
        bm_handle_t handle;
        int vpu_usage[5];

        bm_dev_request(&handle, dev_id);

        unsigned int chipid = 0;
        bm_get_chipid(handle, &chipid);
        if(chipid == 0x1684){
            res_len = 5;
        }else{
            res_len = 3;
        }
        res.reserve(res_len);

#ifndef IS_SOC_MODE
        if(bm_get_vpu_instant_usage==nullptr){
            for(int i = 0; i < res_len; i++){
                res.push_back(-1);
            }
            return res;
        }
        bm_get_vpu_instant_usage(handle, vpu_usage);
        for(int i = 0; i < res_len; i++){
            res.push_back(vpu_usage[i]);
        }
#else
        std::ifstream file("/proc/vpuinfo");
        if(!file.is_open()){
            spdlog::error("File open failed!");
            exit(0);
        }
        std::string line;
        std::regex reg(R"(:([0-9]+)%\|([0-9]+)%)");
        while(std::getline(file, line)){
            std::smatch sm;
            bool ret = std::regex_search(line, sm ,reg);
            if(ret){
                int sm_int = std::stoi(sm[1]);
                res.push_back(sm_int);
            }
        }
#endif
        return res;
    }

    std::vector<int> get_vpp_util(int dev_id){
        std::vector<int> res;
        res.reserve(2);
        bm_handle_t handle;
        int vpp_usage[2];

        bm_dev_request(&handle, dev_id);

#ifndef IS_SOC_MODE
        if(bm_get_vpp_instant_usage==nullptr){
            for(int i = 0; i < 2; +i++){
                res.push_back(-1);
            }
            return res;
        }
        bm_get_vpp_instant_usage(handle, vpp_usage);
        for(int i = 0; i < 2; +i++){
            res.push_back(vpp_usage[i]);
        }
#else
        std::ifstream file("/proc/vppinfo");
        if(!file.is_open()){
            spdlog::error("File open failed!");
            exit(0);
        }
        std::string line;
        std::regex reg(R"(:([0-9]+)%\|([0-9]+)%)");
        while(std::getline(file, line)){
            std::smatch sm;
            bool ret = std::regex_search(line, sm ,reg);
            if(ret){
                int sm_int = std::stoi(sm[1]);
                res.push_back(sm_int);
            }
        }
#endif
        return res;
    }

#ifdef _WIN32
    int setenv(const char* name, const char* value, int overwrite)
    {
        int errcode = 0;
        if (!overwrite) {
            size_t envsize = 0;
            errcode = getenv_s(&envsize, NULL, 0, name);
            if (errcode || envsize) return errcode;
        }
        return _putenv_s(name, value);
    }
#endif

    int set_print_flag(bool print_flag)  {
        if(print_flag)
        return setenv("SAIL_PRINT_VIP_TIMES", "1", 1);
        else
        return setenv("SAIL_PRINT_VIP_TIMES", "0", 1);
    }

    int set_dump_io_flag(bool dump_io_flag){
        if(dump_io_flag)
        return setenv("SAIL_SAVE_IO_TENSORS", "1", 1);
        else
        return setenv("SAIL_SAVE_IO_TENSORS", "0", 1);
    }

    bool get_print_flag()  {
        const char *print_flag = getenv("SAIL_PRINT_VIP_TIMES");
        if(print_flag != nullptr && 0 == strcmp(print_flag,"1"))
            return true;
        return false;
    }

    void printEnvVarsWithPrefix(const std::string &prefix)
    {
        for (char **env = environ; *env != nullptr; ++env)
        {
            std::string envVar = *env;
            spdlog::debug("envVar: {}", envVar);
            if (envVar.find(prefix) == 0)
            {
                // find the first '=', which spilts a pair of NAME=value
                size_t equalPos = envVar.find('=');
                if (equalPos != std::string::npos)
                {
                    std::string name = envVar.substr(0, equalPos);
                    std::string value = envVar.substr(equalPos + 1);
                    SPDLOG_INFO("find envVar, name: {}, value: {}", name, value);
                }
            }
        }
    }

    // 获取系统的当前时间，单位微秒(us)
    double get_current_time_us()
    {
    #ifdef _WIN32
    // 从1601年1月1日0:0:0:000到1970年1月1日0:0:0:000的时间(单位100ns)
    #define EPOCHFILETIME   (116444736000000000UL)
        FILETIME ft;
        LARGE_INTEGER li;
        double tt = 0;
        GetSystemTimeAsFileTime(&ft);
        li.LowPart = ft.dwLowDateTime;
        li.HighPart = ft.dwHighDateTime;
        // 从1970年1月1日0:0:0:000到现在的微秒数(UTC时间)
        tt = (li.QuadPart - EPOCHFILETIME) /10;
        return tt;
    #else
        timeval tv;
        gettimeofday(&tv, 0);
        return (double)tv.tv_sec * 1000000 + (double)tv.tv_usec;
    #endif // _WIN32
        return 0;
    }

    std::unordered_map<bm_handle_t, Handle*> Handle::handle_map;
    std::mutex Handle::map_mutex;

    void delete_shaptr_bm_handle_t(bm_handle_t* handle_ptr){
        // SPDLOG_INFO("Start delete_shaptr_bm_handle_t!");
        delete handle_ptr;
        // SPDLOG_INFO("End delete_shaptr_bm_handle_t!");
    }

    void delete_shaptr_bm_handle_t_allocated(bm_handle_t* handle_ptr){
        // SPDLOG_INFO("Start delete_shaptr_bm_handle_t_allocated!");
        {
            std::lock_guard<std::mutex> lock(Handle::map_mutex);
            if(Handle::handle_map.find(handle_ptr[0]) != Handle::handle_map.end()){
                Handle::handle_map.erase(handle_ptr[0]);
            }
        }
        bm_dev_free(handle_ptr[0]);
        delete handle_ptr;
        // SPDLOG_INFO("End delete_shaptr_bm_handle_t_allocated!");
    }

    std::shared_ptr<bm_handle_t> make_shaptr_bm_handle_t(int dev_id){
        std::shared_ptr<bm_handle_t> ptr_temp = std::shared_ptr<bm_handle_t>(new bm_handle_t[1],delete_shaptr_bm_handle_t_allocated);
        if (bm_dev_query(dev_id)) {
            printf("Error: Invalid tpu id: %d!\n", dev_id);
            exit(SAIL_ERR_DEV_INIT);
        }
        bm_dev_request(&ptr_temp.get()[0], dev_id);
        return std::move(ptr_temp);
    }

    std::shared_ptr<bm_handle_t> make_shaptr_bm_handle_t(bm_handle_t handle){
        std::shared_ptr<bm_handle_t> ptr_temp = std::shared_ptr<bm_handle_t>(new bm_handle_t[1],delete_shaptr_bm_handle_t);
        ptr_temp.get()[0] = handle;
        return std::move(ptr_temp);
    }

    void get_sail_version(char* sail_version){
        char s_month[5];
        int month, day, year;
        int hour, minute, second;
        static const char month_names[] = "JanFebMarAprMayJunJulAugSepOctNovDec";
        sscanf(__DATE__, "%s %d %d", s_month, &day, &year);
        month = (strstr(month_names, s_month)-month_names)/3;
        sscanf(__TIME__, "%d:%d:%d", &hour, &minute, &second);
        sprintf(sail_version,"master(%d%02d%02d_%02d%02d%02d)",year,month+1,day,hour, minute, second);
    }

    class Handle::Handle_CC{
    public:
        explicit Handle_CC()
            : dev_id_(-1) {};

        explicit Handle_CC(bm_handle_t handle)  
            : dev_id_(-2) {
                std::lock_guard<std::mutex> lock(Handle::map_mutex);
                if(Handle::handle_map.find(handle) != Handle::handle_map.end()){
                    handle_ = Handle::handle_map[handle]->shaptr();
                    dev_id_ = bm_get_devid(handle);
                    // SPDLOG_INFO("Find sail::Handle* !");
                }else{
                    handle_ = make_shaptr_bm_handle_t(handle);
                    dev_id_ = bm_get_devid(handle);
                    // SPDLOG_INFO("Not find sail::Handle* !");
                }
                    
            };

        explicit Handle_CC(int dev_id) 
            : dev_id_(-1) {
            handle_ = make_shaptr_bm_handle_t(dev_id);
            dev_id_ = dev_id;
        };

        ~Handle_CC(){
            free();
        };

        /**
         * @brief Free inner bm_handle_t.
         */
        void free();

        std::shared_ptr<bm_handle_t> handle_;
        int dev_id_;
        std::string target;
    };

    void Handle::Handle_CC::free(){
        dev_id_ = -1;
    }

    Handle::Handle() : _impl(new Handle_CC()) {}

    Handle::Handle(bm_handle_t handle) : _impl(new Handle_CC(handle)) {}

    Handle::Handle(int dev_id) : _impl(new Handle_CC(dev_id)) {
        std::lock_guard<std::mutex> lock(Handle::map_mutex);
        if(Handle::handle_map.find(_impl->handle_.get()[0]) == Handle::handle_map.end()){
            Handle::handle_map[_impl->handle_.get()[0]] = this;
        }
    }

    Handle::Handle(const Handle &other): _impl(new Handle_CC()) {
        _impl->handle_ = other._impl->handle_;
        _impl->dev_id_ = other._impl->dev_id_;
    }

    Handle &Handle::operator=(const Handle &other) {
        if (this != &other) {
            _impl->free();
            _impl->handle_ = other._impl->handle_;
            _impl->dev_id_ = other._impl->dev_id_;
        }
        return *this;
    }

    Handle::~Handle() {
        delete _impl;
    }

    std::shared_ptr<bm_handle_t> Handle::shaptr(){
        return _impl->handle_;
    }

    bm_handle_t Handle::data() {
        // SPDLOG_INFO("num_list.use_count: {}",_impl->handle_.use_count());
        return _impl->handle_.get()[0];
    }

    int Handle::get_device_id() {
        return _impl->dev_id_;
    }

    std::string Handle::get_sn() {
        char sn_num[18]={0};
        bm_status_t r_value = bm_get_sn(this->data(),sn_num);
        if(r_value == BM_SUCCESS){
            return std::string(sn_num);
        }
        return "Error";
    }

    std::string Handle::get_target(){
        if (_impl->target.length() > 0){
            return _impl->target;
        }else{
            char board_name_char[40]={0}; 
            bm_status_t r_value = bm_get_board_name(this->data(),board_name_char);
            if(r_value == BM_SUCCESS){
                std::string board_name = std::string(board_name_char);
                if(board_name.find("1684x") != std::string::npos || board_name.find("1684X") != std::string::npos){
                    _impl->target = "BM1684X";
                }else if(board_name.find("1684") != std::string::npos){
                    _impl->target = "BM1684";
                }else if(board_name.find("1688") != std::string::npos){
                    _impl->target = "BM1688";
                }else if(board_name.find("CV186") != std::string::npos ||  board_name.find("cv186") != std::string::npos){
                    _impl->target = "CV186AH";
                }else if(board_name.find("Athena2") != std::string::npos){
                    auto num_cpus = std::thread::hardware_concurrency();
                    _impl->target = num_cpus == 6 ? "CV186AH" : "BM1688";
                }else{
                    _impl->target = "NOT 1684, 1684X, BM1688, or CV186AH !";
                }
                return _impl->target;
            }
        }
        return "Error";
    }

    inline int get_type_size(bm_data_type_t dtype) {
        int type_size = 0;
        switch (dtype) {
            case BM_FLOAT32:
                return sizeof(float);
            case BM_FLOAT16:
                return sizeof(float)/2;
            case BM_BFLOAT16:
                return sizeof(float)/2;
            case BM_INT8:
                return sizeof(int8_t);
            case BM_UINT8:
                return sizeof(uint8_t);
            case BM_INT16:
                return sizeof(int16_t);
            case BM_UINT16:
                return sizeof(uint16_t);
            case BM_INT32:
                return sizeof(int32_t);
            case BM_UINT32:
                return sizeof(uint32_t);
            default:
                return 0;
        }
    }

    /**
     * @brief A class manages the range of tesnor. Only used in tensor slice for now.
     * 
     * The range should be left close and right open: [start, end)
     */
    class Range{
    public:
        Range();
        Range(int s, int e);
        int size() const;
        int start, end;
    };

    inline
    Range::Range()
        : start(0), end(0) {}

    inline
    Range::Range(int s, int e){
        if (s >= e or s < 0){
            spdlog::error("Range is not valid!");
            exit(SAIL_ERR_TENSOR_INIT);
        }
        
        start = s;
        end = e;
    }

    inline
    int Range::size() const
    {
        return end - start;
    }

    class Tensor::Tensor_CC{
    public:
        Tensor_CC();

        ~Tensor_CC(){};

        explicit Tensor_CC(
            const Handle&           handle,
            const std::vector<int>& shape,
            bm_data_type_t          dtype,
            bool                    own_sys_data,
            bool                    own_dev_data);

        explicit Tensor_CC(
            const std::vector<int>& shape,
            bm_data_type_t          dtype);

        void free();
        
        void reset(const std::vector<int>& shape, bm_data_type_t dtype);

        void reset_sys_data(void *data, std::vector<int> &shape);

        void reset_dev_data(bm_device_mem_t data);

        void sync_s2d();

        void sync_s2d(int size);

        void sync_d2s();

        void sync_d2s(int size);

        void sync_d2d(Tensor_CC* src, int offset_src, int offset_dst, int len);

        void sync_d2d_stride(Tensor_CC* src, int stride_src, int stride_dst, int count);

        void sync_d2s(Tensor_CC* src, int offset_src, int offset_dst, int len);

        void sync_s2d(Tensor_CC* src, int offset_src, int offset_dst, int len);

        void sync_from(Tensor_CC* src);

        void sync_to(Tensor_CC* dst);

        void dump_data(std::string file_name, bool bin);

        bm_device_mem_t slice(std::vector<sail::Range> &ranges, bool d2d_flag);
        
        inline bool is_dev_data_valid() const;
        inline bool is_sys_data_valid() const;

#ifdef PYTHON

        explicit Tensor_CC(Handle handle, 
                        bm_data_type_t dtype,
                        const pybind11::buffer_info& buf, 
                        bool own_sys_data,
                        bool own_dev_data=true);

        void update_data(const pybind11::buffer_info& buf, int type_size);
        
#endif
    private:
        friend class Tensor;
        /// Handle instance.
        Handle handle_;

        /// Data type
        bm_data_type_t dtype_;

        /// device id
        int device_id_;

        /// Shape of the tensor.
        std::vector<int> shape_;

        /// Indicator of whether own the data pointer in system memory.
        bool own_sys_data_{false};
        bool own_sys_data_is_mmap_{false};

        /// Indicator of whether own the device memory struct.
        bool own_dev_data_{false};

        /// Data pointer in system memory of the tensor.
        void* sys_data_{nullptr};

        /// Instance of device memory structure.
        bm_device_mem_t dev_data_ {};
        /// data size
        uint32_t data_size_ {0};

    private:
        /**
         * @brief Judge if a tensor shape is valid.
         *
         * @param shape Shape of a tensor
         * @return True for valid and flase for invalid..
         */
        bool shape_is_valid(const std::vector<int>& shape);
    };

    bool Tensor::Tensor_CC::shape_is_valid(const std::vector<int>& shape){
        if (shape.empty()) {
            return false;
        }
        if (std::any_of(shape.begin(), shape.end(), [](int i) { return i <= 0; })) {
            return false;
        }
        return true;
    }

    Tensor::Tensor_CC::Tensor_CC():own_sys_data_(false),own_sys_data_is_mmap_(false),
        own_dev_data_(false),sys_data_(nullptr), dev_data_({}), data_size_(0),device_id_(-1){}

    Tensor::Tensor_CC::Tensor_CC(
        const Handle& handle,
        const std::vector<int>& shape,
        bm_data_type_t dtype,
        bool own_sys_data,
        bool own_dev_data)
        :handle_(handle), shape_(shape), dtype_(dtype),own_sys_data_(own_sys_data), 
        own_dev_data_(own_dev_data), sys_data_(nullptr), dev_data_({}), data_size_(0),device_id_(-1){
        int ret = 0;
        device_id_ = handle_.get_device_id();
        if (shape_is_valid(shape)) {
            int type_size = get_type_size(dtype);
            data_size_ = std::accumulate(shape_.begin(), shape_.end(),
                                         type_size, std::multiplies<int>());
            if (own_dev_data_) {
#if BMCV_VERSION_MAJOR > 1
                ret = bm_malloc_device_byte_heap_mask(handle_.data(), &dev_data_, 3, data_size_);
#else
                ret = bm_malloc_device_byte_heap_mask(handle_.data(), &dev_data_, 7, data_size_);
#endif
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("bm_malloc_device_type() err={}, size={}", ret, data_size_);
                }

                int c = 0;
                void* value = (void*)&c;
                ret = bm_memset_device_ext(handle_.data(), value, 1, dev_data_);
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("bm_memset_device failed, return={}", ret);
                }
            }
            if (own_sys_data_) {
#ifndef IS_SOC_MODE
                sys_data_ = malloc(data_size_);
#else
                if (own_dev_data_) {
                  bm_mem_mmap_device_mem(handle_.data(), &dev_data_,
                                         (unsigned long long*)&sys_data_);
                  own_sys_data_is_mmap_ = true;
                } else {
                  sys_data_ = malloc(data_size_);
             }
#endif
            }
            //} else {
            //  spdlog::error("tensor shape is not valid!");
            //  exit(SAIL_ERR_TENSOR_INIT);
        }        
    }

    Tensor::Tensor_CC::Tensor_CC(const std::vector<int>& shape,bm_data_type_t dtype)
    : shape_(shape), dtype_(dtype), own_sys_data_(true), own_dev_data_(false), 
    sys_data_(nullptr), dev_data_({}), data_size_(0), own_sys_data_is_mmap_(false),device_id_(-1){
        int type_size = get_type_size(dtype);
        data_size_ = std::accumulate(shape_.begin(), shape_.end(),
                                     type_size, std::multiplies<int>());
        if (data_size_ > 0) {
            sys_data_ = malloc(data_size_);
            memset(sys_data_,0,data_size_);
        }
    }

    inline bool Tensor::Tensor_CC::is_dev_data_valid() const {
        return dev_data_.u.device.device_addr != 0 && dev_data_.size != 0;
    }

    inline bool Tensor::Tensor_CC::is_sys_data_valid() const {
        return sys_data_ != nullptr;
    }

    void Tensor::Tensor_CC::free() {
        if (own_sys_data_ && sys_data_) {
            if (own_sys_data_is_mmap_) {
                bm_mem_unmap_device_mem(handle_.data(), sys_data_, data_size_);
            } else {
                std::free(sys_data_);
            }
            sys_data_ = nullptr;
        }

        if (own_dev_data_) {
            if (is_dev_data_valid()) bm_free_device(handle_.data(), dev_data_);
            dev_data_ = {};
        } else {
            if (sys_data_ != nullptr && dev_data_.size > 0) {
                if (own_sys_data_is_mmap_) {
                    bm_mem_unmap_device_mem(handle_.data(), sys_data_, dev_data_.size);
                    sys_data_ = nullptr;
                }
            }
        }
    }

    void Tensor::Tensor_CC::reset(const std::vector<int> &shape, bm_data_type_t dtype) {
        if (!shape_is_valid(shape)) {
            spdlog::error("Invalid tensor shape!");
            exit(SAIL_ERR_TENSOR_SHAPE);
        }

        int ret = 0;
        int size_shape = std::accumulate(shape.begin(), shape.end(),
                                         1, std::multiplies<int>());

        int data_size = size_shape * get_type_size(dtype);

        if (data_size_ != data_size) {
            if (own_dev_data_) {
                bm_free_device(handle_.data(), dev_data_);
#if BMCV_VERSION_MAJOR > 1
                ret = bm_malloc_device_byte_heap_mask(handle_.data(), &dev_data_, 3, data_size);
#else
                ret = bm_malloc_device_byte_heap_mask(handle_.data(), &dev_data_, 7, data_size);
#endif
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("bm_malloc_device_byte err={}, size={}", ret, data_size);
                }
            }
            if (own_sys_data_) {
#ifndef IS_SOC_MODE
                std::free(sys_data_);
                sys_data_ = malloc(data_size);
#else
                if (own_dev_data_) {
                  bm_mem_unmap_device_mem(handle_.data(), sys_data_, data_size_);
                  bm_mem_mmap_device_mem(handle_.data(), &dev_data_,
                                         (unsigned long long*)&sys_data_);
                  bm_mem_invalidate_device_mem(handle_.data(), &dev_data_);
                  own_sys_data_is_mmap_ = true;
                } else {
                  std::free(sys_data_);
                  sys_data_ = malloc(data_size);
                }
#endif
            }
        }
        dtype_ = dtype;
        shape_ = shape;
        data_size_ = data_size;
    }

    void Tensor::Tensor_CC::reset_sys_data(void *data, std::vector<int> &shape)
    {
        reset(shape, dtype_);
        if (own_dev_data_) {
            double process_start_time_befor = get_current_time_us();
            if (sys_data_ != nullptr) {
                memcpy(sys_data_, data, data_size_);
            } else {
                spdlog::error("Cannot reset_sys_data when own_dev_data is true.");
                exit(SAIL_ERR_TENSOR_DEVMEM);
            }
            PRINT_TIME_MS("memcpy_cpu_to_cpu_0", process_start_time_befor)
        } else if (own_sys_data_) {
            double process_start_time_befor = get_current_time_us();
            if (sys_data_ != nullptr) {
                memcpy(sys_data_, data, data_size_);
            }
            PRINT_TIME_MS("memcpy_cpu_to_cpu_1", process_start_time_befor)
        } else {
            sys_data_ = data;
        }
    }

    void Tensor::Tensor_CC::reset_dev_data(bm_device_mem_t data)  {
        if (own_dev_data_) {
            if (is_sys_data_valid() && is_dev_data_valid() && own_sys_data_is_mmap_) {
                bm_mem_unmap_device_mem(handle_.data(), sys_data_, dev_data_.size);
                printf("%s:%d\n", __FILE__, __LINE__);
                sys_data_ = nullptr;
            }

            bm_free_device(handle_.data(), dev_data_);
            dev_data_ = data;
            own_dev_data_ = false;
            // device memory changed, mmap will change too
#ifdef IS_SOC_MODE
            bm_mem_mmap_device_mem(handle_.data(), &dev_data_, (unsigned long long*)&sys_data_);
            own_sys_data_is_mmap_ = true;
#endif
        } else {
            if (is_sys_data_valid()) {
                if (own_sys_data_is_mmap_) {
#ifdef IS_SOC_MODE
                    bm_mem_unmap_device_mem(handle_.data(), sys_data_, dev_data_.size);
                    sys_data_ = nullptr;
#endif
                }
                dev_data_ = data;
#ifdef IS_SOC_MODE
                bm_mem_mmap_device_mem(handle_.data(), &dev_data_, (unsigned long long*)&sys_data_);
                own_sys_data_is_mmap_ = true;
#endif
            } else {
                dev_data_ = data;
#ifdef IS_SOC_MODE
                bm_mem_mmap_device_mem(handle_.data(), &dev_data_, (unsigned long long*)&sys_data_);
                own_sys_data_is_mmap_ = true;
#endif
            }
        }
        sync_d2s();
    }

    void Tensor::Tensor_CC::sync_d2s(int size) {
        if (is_dev_data_valid()) {
            if (is_sys_data_valid()) {
                if (!own_sys_data_is_mmap_) {
                    double process_start_time_d2s = get_current_time_us();
                    if (BM_SUCCESS != bm_memcpy_d2s_partial(handle_.data(), sys_data_, dev_data_, size)) {
                        SPDLOG_ERROR("bm_memcpy_d2s_partial() err");
                    }
                    PRINT_TIME_MS("bm_memcpy_d2s_partial", process_start_time_d2s)
                } else {
                    bm_mem_invalidate_device_mem(handle_.data(), &dev_data_);
                }
            }
        } else {
            if (is_sys_data_valid()) {
                if (!own_sys_data_is_mmap_) {
                    double process_start_time_d2s = get_current_time_us();
                    if (BM_SUCCESS != bm_memcpy_d2s_partial(handle_.data(), sys_data_, dev_data_, size)) {
                        SPDLOG_ERROR("bm_memcpy_d2s_partial() err");
                    }
                    PRINT_TIME_MS("bm_memcpy_d2s_partial", process_start_time_d2s)
                } else {
                    bm_mem_invalidate_device_mem(handle_.data(), &dev_data_);
                }
            }
        }
    }

    void Tensor::Tensor_CC::sync_s2d(int size) {
        if (is_sys_data_valid()) {
            if (own_sys_data_is_mmap_) {
                bm_mem_flush_partial_device_mem(handle_.data(), &dev_data_, 0, size);
            } else {
                double process_start_time = get_current_time_us();
                int ret = bm_memcpy_s2d_partial(handle_.data(), dev_data_, sys_data_, size);
                PRINT_TIME_MS("bm_memcpy_s2d_partial",process_start_time);

                if (ret != BM_SUCCESS) {
                    spdlog::error("bm_memcpy_s2d_partial(sys_data=%p, dev_data.addr=%p, size=%d)\n",
                                  sys_data_, (void *) dev_data_.u.device.device_addr, size);
                }
            }
        }
    }

    void Tensor::Tensor_CC::sync_d2s() {
        sync_d2s(data_size_);
    }

    void Tensor::Tensor_CC::sync_s2d() {
        if (is_sys_data_valid()) {
            if (own_sys_data_is_mmap_) {
                bm_mem_flush_device_mem(handle_.data(), &dev_data_);
            } else {
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(handle_.data(), dev_data_, sys_data_);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
            }
        }
    }

    void Tensor::Tensor_CC::sync_d2d(Tensor_CC* src, int offset_src, int offset_dst, int len){
        // Ensure data types match
        if (dtype_ != src->dtype_) {
            spdlog::error("sync_d2d: Data types do not match!");
            exit(SAIL_ERR_TENSOR_DTYPE);
        }

        // Ensure both source and destination data are on the device
        if (!src->is_dev_data_valid() || !is_dev_data_valid()) {
            spdlog::error("sync_d2d: Data is not on the device!");
            exit(SAIL_ERR_TENSOR_DEVMEM);
        }

        // Calculate the total number of elements for source and destination
        int size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
        int src_size = std::accumulate(src->shape_.begin(), src->shape_.end(), 1, std::multiplies<int>());

        // Check if source and destination data are within bounds
        if (offset_src + len > src_size) {
            spdlog::error("sync_d2d: Source data out of bounds, offset_src:{}, len:{}, src_size:{}!",offset_src, len, src_size);
            exit(SAIL_ERR_TENSOR_PARAM);
        }
        if (offset_dst + len > size) {
            spdlog::error("sync_d2d: Destination data out of bounds, offset_dst:{}, len:{}, dst_size:{}!",offset_dst, len, size);
            exit(SAIL_ERR_TENSOR_PARAM);
        }

        // Get the size of the data type
        int type_size = get_type_size(dtype_);

        // Perform the device-to-device memory copy
        bm_device_mem_t src_dev_data = src->dev_data_;
        bm_memcpy_d2d_byte(handle_.data(), dev_data_, offset_dst * type_size,
                      src_dev_data, offset_src * type_size, len * type_size);
    }

    void Tensor::Tensor_CC::sync_d2d_stride(Tensor_CC* src, int stride_src, int stride_dst, int count){
        // Get the size of the data type
        int type_size = get_type_size(dtype_);

        // Check the stride_dst value
        if (!(stride_dst = 1 || (stride_dst == 4 && stride_src == 1 && type_size == 1 ))){
            spdlog::error("sync_d2d_stride: stride_dst must be 1, EXCEPT: stride_dst == 4 && stride_src == 1 && Tensor_type_size == 1");
            throw SailTensorError("Wrong stride config!");
        }
        // Ensure data types match
        if (dtype_ != src->dtype_) {
            spdlog::error("sync_d2d_stride: Data types do not match!");
            throw DataTypeError("Dtype not match!");
        }

        // Ensure both source and destination data are on the device
        if (!src->is_dev_data_valid() || !is_dev_data_valid()) {
            spdlog::error("sync_d2d_stride: Data is not on the device!");
            throw MemoryError("Data is not on the device!");
        }

        // Calculate the total number of elements for source and destination
        int size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
        int src_size = std::accumulate(src->shape_.begin(), src->shape_.end(), 1, std::multiplies<int>());

        // Check if source and destination data are within bounds
        if (count * stride_src > src_size) {
            spdlog::error("sync_d2d_stride: Source data out of bounds, stride_src:{}, len:{}, src_size:{}!",stride_src, count, src_size);
            throw SailTensorError("Out of bounds!");
        }
        if (count * stride_dst > size) {
            spdlog::error("sync_d2d_stride: Destination data out of bounds, stride_dst:{}, len:{}, dst_size:{}!",stride_dst, count, size);
            throw SailTensorError("Out of bounds!");
        }


        // Perform the device-to-device memory copy
        bm_device_mem_t src_dev_data = src->dev_data_;
        int ret = bm_memcpy_d2d_stride(handle_.data(), dev_data_, stride_dst,
                      src_dev_data, stride_src, count, type_size);
        if(ret != BM_SUCCESS){
            throw MemoryError("bm_memcpy_d2d_stride failed!");
        }
    }

    void Tensor::Tensor_CC::sync_d2s(Tensor_CC* src, int offset_src, int offset_dst, int len)
    {
         // Ensure data types match
        if (dtype_ != src->dtype_) {
            spdlog::error("sync_d2s: Data types do not match!");
            throw DataTypeError("Dtype not match!");
        }

        // Calculate the total number of elements for source and destination
        int size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
        int src_size = std::accumulate(src->shape_.begin(), src->shape_.end(), 1, std::multiplies<int>());

        // Check if source and destination data are within bounds
        if (offset_src + len > src_size) {
            spdlog::error("sync_d2d: Source data out of bounds!");
            throw MemoryError("Source data out of bounds!");
        }
        if (offset_dst + len > size) {
            spdlog::error("sync_d2d: Destination data out of bounds!");
            throw MemoryError("Destination data out of bounds!");
        }

        // Get the size of the data type
        int type_size = get_type_size(dtype_);

        size = len * type_size;

        if (src->is_dev_data_valid()) {
            if (is_sys_data_valid()) {
                if(own_sys_data_is_mmap_){
                    bm_mem_invalidate_device_mem(handle_.data(), &dev_data_);
                }
                double process_start_time_d2s = get_current_time_us();
                if (bm_memcpy_d2s_partial_offset(handle_.data(), sys_data_ + offset_dst * type_size, src->dev_data_, size, offset_src * type_size) != 0) {
                    SPDLOG_ERROR("bm_memcpy_d2s_partial() err");
                    throw MemoryError("bm_memcpy_d2s_partial failed!");
                }
                PRINT_TIME_MS("bm_memcpy_d2s_partial", process_start_time_d2s)
            }else{
                SPDLOG_ERROR("Do not found system memory!");
                throw MemoryError("Do not found system memory!");
            }
        } else {
            SPDLOG_ERROR("Data is not on the device!");
            throw MemoryError("Data is not on the device!");
        }
    }

    void Tensor::Tensor_CC::sync_s2d(Tensor_CC* src, int offset_src, int offset_dst, int len)
    {
         // Ensure data types match
        if (dtype_ != src->dtype_) {
            spdlog::error("sync_s2d: Data types do not match!");
            throw DataTypeError("Dtype not match!");
        }

        // Calculate the total number of elements for source and destination
        int size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
        int src_size = std::accumulate(src->shape_.begin(), src->shape_.end(), 1, std::multiplies<int>());

        // Check if source and destination data are within bounds
        if (offset_src + len > src_size) {
            spdlog::error("sync_s2d: Source data out of bounds!");
            throw MemoryError("Source data out of bounds!");
        }
        if (offset_dst + len > size) {
            spdlog::error("sync_s2d: Destination data out of bounds!");
            throw MemoryError("Destination data out of bounds!");
        }

        // Get the size of the data type
        int type_size = get_type_size(dtype_);

        size = len * type_size;

        if (src->is_sys_data_valid() && is_dev_data_valid()) {
            double process_start_time = get_current_time_us();
            int ret = bm_memcpy_s2d_partial_offset(handle_.data(), dev_data_, src->sys_data_+ offset_src * type_size, size, offset_dst * type_size);
            PRINT_TIME_MS("bm_memcpy_s2d_partial",process_start_time);
            if (ret != BM_SUCCESS) {
                spdlog::error("bm_memcpy_s2d_partial(sys_data=%p, dev_data.addr=%p, size=%d)\n",
                                sys_data_, (void *) dev_data_.u.device.device_addr, size);
                throw MemoryError("bm_memcpy_s2d_partial failed!");
            }
        }else{
            SPDLOG_ERROR("Do not found device memory!");
            throw MemoryError("Do not found device memory!");
        }
    }


    void Tensor::Tensor_CC::sync_from(Tensor_CC* src)   {
        if (dtype_ != src->dtype_) {
            spdlog::error("sync_from: data type not match!");
            exit(SAIL_ERR_TENSOR_DTYPE);
        }
        int size = std::accumulate(shape_.begin(), shape_.end(),
                                   1, std::multiplies<int>());
        auto src_shape = src->shape_;
        int src_size = std::accumulate(src_shape.begin(), src_shape.end(),
                                       1, std::multiplies<int>());
        if (size != src_size) {
            spdlog::error("sync_from: tensor size not match!");
            exit(SAIL_ERR_TENSOR_SIZE);
        }
        auto src_handle = src->handle_.data();
        auto dtype_size = get_type_size(dtype_);
        void *src_sys_data = src->sys_data_;
        bool src_dev_data_valid = src->is_dev_data_valid();
        bm_device_mem_t src_dev_data = src->dev_data_;
        if (is_sys_data_valid()) {
            if (src_dev_data_valid) {
                double process_start_time_d2s = get_current_time_us();
                bm_memcpy_d2s(src_handle, sys_data_, src_dev_data);
                PRINT_TIME_MS("bm_memcpy_d2s", process_start_time_d2s)
            } else if (src_sys_data) {
                memcpy(sys_data_, src_sys_data, size * dtype_size);
            }
            if (is_dev_data_valid()) {
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(handle_.data(), dev_data_, sys_data_);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
            }
        } else if (is_dev_data_valid()) {
            if (src_sys_data) {
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(handle_.data(), dev_data_, src_sys_data);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
            } else if (src_dev_data_valid) {
                void *tmp = malloc(size * dtype_size);
                double process_start_time_d2s = get_current_time_us();
                bm_memcpy_d2s(handle_.data(), tmp, src_dev_data);
                PRINT_TIME_MS("bm_memcpy_d2s", process_start_time_d2s)
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(handle_.data(), dev_data_, tmp);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
                std::free(tmp);
            }
        }
    }

    void Tensor::Tensor_CC::sync_to(Tensor_CC* dst){
        if (dtype_ != dst->dtype_) {
            spdlog::error("dst_from: data type not match!");
            exit(SAIL_ERR_TENSOR_DTYPE);
        }
        int size = std::accumulate(shape_.begin(), shape_.end(),
                                   1, std::multiplies<int>());
        auto dst_shape = dst->shape_;
        int dst_size = std::accumulate(dst_shape.begin(), dst_shape.end(),
                                       1, std::multiplies<int>());
        if (size != dst_size) {
            spdlog::error("dst_from: tensor size not match!");
            exit(SAIL_ERR_TENSOR_SIZE);
        }
        auto dst_handle = dst->handle_.data();
        auto dtype_size = get_type_size(dtype_);
        void *dst_sys_data = dst->sys_data_;
        bool dst_dev_data_valid = dst->is_dev_data_valid();
        bm_device_mem_t dst_dev_data = dst->dev_data_;
        if (dst_sys_data) {
            if (is_dev_data_valid()) {
                double process_start_time_d2s = get_current_time_us();
                bm_memcpy_d2s(handle_.data(), dst_sys_data, dev_data_);
                PRINT_TIME_MS("bm_memcpy_d2s", process_start_time_d2s)
            } else if (sys_data_) {
                memcpy(dst_sys_data, sys_data_, size * dtype_size);
            }
            if (dst_dev_data_valid) {
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(dst_handle, dst_dev_data, sys_data_);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
            }
        } else if (dst_dev_data_valid) {
            if (sys_data_) {
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(dst_handle, dst_dev_data, sys_data_);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
            } else if (is_dev_data_valid()) {
                void *tmp = malloc(size * dtype_size);
                double process_start_time_d2s = get_current_time_us();
                bm_memcpy_d2s(handle_.data(), tmp, dev_data_);
                PRINT_TIME_MS("bm_memcpy_d2s", process_start_time_d2s)
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(handle_.data(), dst_dev_data, tmp);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
                std::free(tmp);
            }
        }
    }


    void Tensor::Tensor_CC::dump_data(std::string file_name, bool bin){
        // SPDLOG_INFO("dtype:{}",dtype_);
        // if sys_data_ and dev_data_ are nullptr
        if( !is_dev_data_valid() && !is_sys_data_valid()){
            SPDLOG_ERROR("Tensor is NULL! Can not dump!");
            return ;
        }

        void* buffer;
        // 无系统内存
        if (sys_data_ == nullptr){
            // 定义临时变量
            buffer = new char[data_size_];
            if (bm_memcpy_d2s_partial(handle_.data(), buffer, dev_data_, data_size_) != 0) {
                SPDLOG_ERROR("bm_memcpy_d2s_partial() err");
                delete[] buffer;
                return;
            }
        }else{
            buffer = sys_data_;
        }
           
        // 二进制存储文件
        if(bin){
            FILE* fp = fopen(file_name.c_str(),"wb+");
            fwrite(buffer,1,data_size_,fp);
            fclose(fp);
        // 十进制存储浮点数
        }else{
            std::ofstream outputFile(file_name);
            // 类型转换
            int row = shape_.back();
            int type_size = get_type_size(dtype_);
            if(dtype_ == BM_FLOAT32){
                SPDLOG_INFO("BM_FLOAT32");
                // 浮点数写入文件
                outputFile << std::setprecision(4) << std::fixed;
                float* floatPtr = static_cast<float*>(buffer);
                for(int i=0; i < int(data_size_ / type_size); ++i){
                    outputFile << floatPtr[i] << " ";
                    if( (i+1) % row == 0){
                        outputFile << std::endl;
                    }
                }
            }else if (dtype_ == BM_UINT8){
                SPDLOG_INFO("BM_UINT8");
                uint8_t* uint8Ptr = static_cast<uint8_t*>(buffer);
                for(int i=0; i < int(data_size_ / type_size); ++i){
                    outputFile << static_cast<int>(uint8Ptr[i]) << " ";
                    if( (i+1) % row == 0){
                        outputFile << std::endl;
                    }
                }
            }else if (dtype_ == BM_INT8){
                SPDLOG_INFO("BM_INT8");
                int8_t* int8Ptr = static_cast<int8_t*>(buffer);
                for(int i=0; i < int(data_size_ / type_size); ++i){
                    outputFile << static_cast<int>(int8Ptr[i]) << " ";
                    if( (i+1) % row == 0){
                        outputFile << std::endl;
                    }
                }
            }else if (dtype_ == BM_INT16){
                SPDLOG_INFO("BM_INT16");
                int16_t* int16Ptr = static_cast<int16_t*>(buffer);
                for(int i=0; i < int(data_size_ / type_size); ++i){
                    outputFile << static_cast<int>(int16Ptr[i]) << " ";
                    if( (i+1) % row == 0){
                        outputFile << std::endl;
                    }
                }
            }else if (dtype_ == BM_UINT16){
                SPDLOG_INFO("BM_UINT16");
                uint16_t* uint16Ptr = static_cast<uint16_t*>(buffer);
                for(int i=0; i < int(data_size_ / type_size); ++i){
                    outputFile << static_cast<int>(uint16Ptr[i]) << " ";
                    if( (i+1) % row == 0){
                        outputFile << std::endl;
                    }
                }
            }else if (dtype_ == BM_INT32){
                SPDLOG_INFO("BM_INT32");
                int32_t* int32Ptr = static_cast<int32_t*>(buffer);
                for(int i=0; i < int(data_size_ / type_size); ++i){
                    outputFile << static_cast<int>(int32Ptr[i]) << " ";
                    if( (i+1) % row == 0){
                        outputFile << std::endl;
                    }
                }
            }else if (dtype_ == BM_UINT32){
                SPDLOG_INFO("BM_UINT32");
                uint32_t* uint32Ptr = static_cast<uint32_t*>(buffer);
                for(int i=0; i < int(data_size_ / type_size); ++i){
                    outputFile << static_cast<int>(uint32Ptr[i]) << " ";
                    if( (i+1) % row == 0){
                        outputFile << std::endl;
                    }
                }
            }else{
                SPDLOG_ERROR("{} dtype can not dump! Support BM_FLOAT32/BM_UINT8/BM_INT8.",dtype_);
                delete[] buffer;
                return;
            }  
            // 关闭文件
            outputFile.close();
        }
        
        // 释放创建的缓冲buffer的内存
        if(sys_data_ == nullptr) delete[] buffer;

    }

    // 只支持设备内存操作，最多2维
    // 对于只有系统内存的情况，直接用numpy操作完再生成tensor就可以，不需要这个函数
    // c2c函数不能截取，只能copy整个结构体，因此每次先创建一个区域以后再copy过去
    bm_device_mem_t Tensor::Tensor_CC::slice(std::vector<sail::Range> &ranges, bool d2d_flag=true){
        if (ranges.size()>2 || shape_.size() > 2){
            spdlog::error("slicing not support shape > 2");
            exit(SAIL_ERR_TENSOR_INIT);
        }

        if (shape_.size()!=ranges.size()){
            spdlog::error("original tensor shape is not equal to slicing shape");
            exit(SAIL_ERR_TENSOR_INIT);
        }
        
        if(!is_dev_data_valid()){
            spdlog::error("no device mem from the original tensor");
            exit(SAIL_ERR_TENSOR_INIT);
        }

        std::vector<int> new_shape;
        for (int i = 0; i < ranges.size(); i++){
            if (ranges[i].end > shape_[i]){
                spdlog::error("slicing range is larger than origin tensor");
                exit(SAIL_ERR_TENSOR_INIT);
            }
            new_shape.emplace_back(ranges[i].size());
        }

        // 创建新dev_mem，并初始化0
        int ret = 0;
        bm_device_mem_t new_dev_data;
        size_t type_size = get_type_size(dtype_);
        size_t new_data_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                         type_size, std::multiplies<int>());
#if BMCV_VERSION_MAJOR > 1
        ret = bm_malloc_device_byte_heap_mask(handle_.data(), &new_dev_data, 3, new_data_size);
#else
        ret = bm_malloc_device_byte_heap_mask(handle_.data(), &new_dev_data, 7, new_data_size);
#endif
        if (ret != BM_SUCCESS){
            SPDLOG_ERROR("bm_malloc_device_type() err={}, size={}", ret, new_data_size);
        }
        int c = 0;
        void* value = (void*)&c;
        ret = bm_memset_device_ext(handle_.data(), value, 1, new_dev_data);
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("bm_memset_device failed, return={}", ret);
        }


        if (new_shape.size()==2){
            size_t dst_length = new_shape[1]*type_size;
            for (int i = ranges[0].start; i < ranges[0].end; i++ ){
                if (d2d_flag){
                    ret = bm_memcpy_d2d_byte(handle_.data(), 
                                new_dev_data, (i-ranges[0].start)*dst_length, 
                                dev_data_, (shape_[1]*i+ranges[1].start)*type_size,
                                dst_length);
                }else{
                    // 原tensor切出部分的一行
                    auto new_mem = bm_mem_from_device(
                                    bm_mem_get_device_addr(dev_data_)+(shape_[1]*i+ranges[1].start)*type_size, 
                                    dst_length);
                    // 新tensor一行
                    auto dst_mem = bm_mem_from_device(
                                    bm_mem_get_device_addr(new_dev_data)+(i-ranges[0].start)*dst_length,
                                    dst_length);
                    ret = bm_memcpy_c2c(handle_.data(), handle_.data(), new_mem, dst_mem, true);
                }
            }
        }else{
            if (d2d_flag){
                ret = bm_memcpy_d2d_byte(handle_.data(), 
                            new_dev_data, 0, 
                            dev_data_, ranges[0].start*type_size, 
                            type_size*new_shape[0]);
                
            }else{
                auto new_mem = bm_mem_from_device(
                                bm_mem_get_device_addr(dev_data_)+ranges[0].start*type_size,
                                new_shape[0]*type_size);
                ret = bm_memcpy_c2c(handle_.data(), handle_.data(), new_mem, new_dev_data, true);
                
            }
            
        }
        if (BM_SUCCESS != ret){
            if (d2d_flag){
                spdlog::error("d2d failed");
                exit(SAIL_ERR_DEV_MCOPY);
            }else{
                spdlog::error("c2c failed");
                exit(SAIL_ERR_DEV_MCOPY);
            }
                
        }

        return std::move(new_dev_data);
        
    }

#ifdef PYTHON

    Tensor::Tensor_CC::Tensor_CC(Handle handle, 
                    bm_data_type_t dtype,
                    const pybind11::buffer_info& buf, 
                    bool own_sys_data,
                    bool own_dev_data)
        :handle_(handle),dtype_(dtype),own_sys_data_(own_sys_data),
        own_dev_data_(own_dev_data),sys_data_(nullptr),dev_data_({}),device_id_(-1){
        if (buf.ndim < 1) {
            spdlog::error("Invalid tensor shape!");
            exit(SAIL_ERR_TENSOR_SHAPE);
        }
        shape_.clear();
        for (auto it : buf.shape) {
            shape_.push_back(static_cast<int>(it));
        }

        device_id_ = handle_.get_device_id();

        void* numpy_ptr = buf.ptr;

        pybind11::array_t<float> arr_float;
        pybind11::array_t<int8_t> arr_int8_t;
        pybind11::array_t<uint8_t> arr_uint8_t;
        pybind11::array_t<int32_t> arr_int32_t;

        pybind11::module np = pybind11::module::import("numpy");  // like 'import numpy as np'
        if(BM_FLOAT32 == dtype){
            pybind11::array_t<float> buf_temp(buf);
            arr_float = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_float.request().ptr;
        }else if(BM_INT8 == dtype){
            pybind11::array_t<int8_t> buf_temp(buf);
            arr_int8_t = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_int8_t.request().ptr;
        }else if(BM_UINT8 == dtype){
            pybind11::array_t<uint8_t> buf_temp(buf);
            arr_uint8_t = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_uint8_t.request().ptr;
        }else if(BM_INT32 == dtype){
            pybind11::array_t<int32_t> buf_temp(buf);
            arr_int32_t = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_int32_t.request().ptr;
        }else{
            SPDLOG_ERROR("Input Data Type not supported: {}",dtype);
            exit(SAIL_ERR_TENSOR_DTYPE);
        }

        // alloc dev_mem
        int data_size = std::accumulate(shape_.begin(), shape_.end(),
                        get_type_size(dtype), std::multiplies<int>());
        data_size_ = data_size;
        if (own_dev_data_) {
#if BMCV_VERSION_MAJOR > 1
            int ret = bm_malloc_device_byte_heap_mask(handle_.data(), &dev_data_, 3, data_size);
#else
            int ret = bm_malloc_device_byte_heap_mask(handle_.data(), &dev_data_, 7, data_size);
#endif
            if (BM_SUCCESS != ret) {
                SPDLOG_ERROR("bm_malloc_device_byte_heap_mask() err={}", ret);
                exit(SAIL_ERR_DEV_MALLOC);
            }
        }
        if (own_sys_data_) {
#ifndef IS_SOC_MODE
            sys_data_ = new uint8_t[data_size];
#else
            bm_mem_mmap_device_mem(handle_.data(), &dev_data_, (unsigned long long*)&sys_data_);
            own_sys_data_is_mmap_ = true;
#endif
            memcpy(sys_data_, numpy_ptr, data_size); 
        }else{
            double process_start_time = get_current_time_us();
            bm_memcpy_s2d(handle_.data(), dev_data_, numpy_ptr);
            int ret = bm_memcpy_s2d_partial(handle_.data(), dev_data_, numpy_ptr, data_size);
            
            if (ret != BM_SUCCESS) {
                spdlog::error("bm_memcpy_s2d_partial(sys_data=%p, dev_data.addr=%p, size=%d)\n",
                    numpy_ptr, (void *) dev_data_.u.device.device_addr, data_size);
            }

            PRINT_TIME_MS("bm_memcpy_s2d_partial",process_start_time);
        }
    }

    void Tensor::Tensor_CC::update_data(const pybind11::buffer_info& buf, int type_size)
    {
        if (buf.ndim != shape_.size()) {
            SPDLOG_ERROR("Invalid tensor shape dims {} vs. {}!",shape_.size(),buf.ndim);
            exit(SAIL_ERR_TENSOR_SHAPE);
        }
        std::vector<int> shape;
        for (auto it : buf.shape) {
            shape.push_back(static_cast<int>(it));
        }

        for (int i=0;i<shape.size();++i){ 
            if(shape[i] != shape_[i]){
                char str_shape_old[256]={};
                char str_shape_new[256]={};
                sprintf(str_shape_old,"[");
                sprintf(str_shape_new,"[");
                for (int j=0;j<shape.size();++j){
                    sprintf(&str_shape_new[strlen(str_shape_new)],"%d,",shape[j]);
                    sprintf(&str_shape_old[strlen(str_shape_old)],"%d,",shape_[j]);
                }
                str_shape_new[strlen(str_shape_new)-1] = ']';
                str_shape_old[strlen(str_shape_old)-1] = ']';
                SPDLOG_ERROR("Invalid tensor shape {} vs. {}!",str_shape_old,str_shape_new);
                exit(SAIL_ERR_TENSOR_SHAPE);
            }
        }
        size_t type_size_tmep = 1;
        if (dtype_ == BM_FLOAT32) {
            type_size_tmep = sizeof(float);
        } else if (dtype_ == BM_INT8) {
            type_size_tmep = sizeof(int8_t);
        } else if (dtype_ == BM_UINT8) {
            type_size_tmep = sizeof(uint8_t);
        } else if (dtype_ == BM_INT32) {
            type_size_tmep = sizeof(int32_t);
        } else if (dtype_ == BM_FLOAT16) {
            type_size_tmep = sizeof(float)/2;
        } else if (dtype_ == BM_BFLOAT16) {
            type_size_tmep = sizeof(float)/2;
        } else if (dtype_ == BM_UINT32) {
            type_size_tmep = sizeof(uint32_t);
        } else if (dtype_ == BM_UINT16) {
            type_size_tmep = sizeof(uint16_t);
        } else if (dtype_ == BM_INT16) {
            type_size_tmep = sizeof(int16_t);
        } 

        int old_size = std::accumulate(shape_.begin(), shape_.end(),
            type_size_tmep, std::multiplies<int>());
        int new_size = std::accumulate(shape.begin(), shape.end(),
            type_size, std::multiplies<int>());
        if (new_size > old_size) {
            spdlog::error("Data size exceeds tensor size!");
            exit(SAIL_ERR_TENSOR_SIZE);
        }

        void* numpy_ptr = buf.ptr;

        pybind11::array_t<float> arr_float;
        pybind11::array_t<int8_t> arr_int8_t;
        pybind11::array_t<uint8_t> arr_uint8_t;
        pybind11::array_t<int32_t> arr_int32_t;
        pybind11::array_t<uint32_t> arr_uint32_t;
        pybind11::array_t<uint16_t> arr_uint16_t;
        pybind11::array_t<int16_t> arr_int16_t;
        pybind11::module np = pybind11::module::import("numpy");  // like 'import numpy as np'
        if(BM_FLOAT32 == dtype_){
            pybind11::array_t<float> buf_temp(buf);
            arr_float = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_float.request().ptr;
        }else if(BM_INT8 == dtype_){
            pybind11::array_t<int8_t> buf_temp(buf);
            arr_int8_t = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_int8_t.request().ptr;
        }else if(BM_UINT8 == dtype_){
            pybind11::array_t<uint8_t> buf_temp(buf);
            arr_uint8_t = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_uint8_t.request().ptr;
        }else if(BM_INT32 == dtype_){
            pybind11::array_t<int32_t> buf_temp(buf);
            arr_int32_t = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_int32_t.request().ptr;
        }else if(BM_FLOAT16 == dtype_){
            pybind11::array_t<uint16_t> buf_temp(buf);
            arr_uint16_t = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_uint16_t.request().ptr;
        }else if(BM_BFLOAT16 == dtype_){
            pybind11::array_t<uint16_t> buf_temp(buf);
            arr_uint16_t = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_uint16_t.request().ptr;
        }else if(BM_UINT32 == dtype_){
            pybind11::array_t<uint32_t> buf_temp(buf);
            arr_uint32_t = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_uint32_t.request().ptr;
        }else if(BM_UINT16 == dtype_){
            pybind11::array_t<uint16_t> buf_temp(buf);
            arr_uint16_t = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_uint16_t.request().ptr;
        }else if(BM_INT16 == dtype_){
            pybind11::array_t<uint16_t> buf_temp(buf);
            arr_int16_t = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_int16_t.request().ptr;
        }

        if (is_sys_data_valid()){

#ifndef IS_SOC_MODE
    //    if (own_sys_data_) {
    //      std::free(sys_data_);
    //      own_sys_data_ = false;
    //    }
    //    sys_data_ = buf.ptr;
            memcpy(sys_data_, numpy_ptr, new_size);

#else
            memcpy(sys_data_, numpy_ptr, new_size);
#endif
        } else if(is_dev_data_valid()){
            double process_start_time = get_current_time_us();
            // bm_memcpy_s2d(handle_.data(), dev_data_, buf.ptr);
            int ret = bm_memcpy_s2d_partial(handle_.data(), dev_data_, numpy_ptr, new_size);
            
            if (ret != BM_SUCCESS) {
                spdlog::error("bm_memcpy_s2d_partial(sys_data=%p, dev_data.addr=%p, size=%d)\n",
                    numpy_ptr, (void *) dev_data_.u.device.device_addr, new_size);
            }
            PRINT_TIME_MS("bm_memcpy_s2d_partial",process_start_time);
        }else{
            spdlog::error("Can not found device memory or host memory!");
            exit(SAIL_ERR_TENSOR_EMPTY);
        }
    }

#endif

    Tensor::Tensor(
            const Handle &handle,
            const std::vector<int> &shape,
            bm_data_type_t dtype,
            bool own_sys_data,
            bool own_dev_data)
            : _impl(new Tensor_CC(handle,shape,dtype,own_sys_data,own_dev_data)){}

    Tensor::Tensor(
            const std::vector<int> &shape,
            bm_data_type_t dtype)
            : _impl(new Tensor_CC(shape,dtype)){}

    Tensor::Tensor(const Tensor &other):_impl(new Tensor_CC()) {
        _impl->handle_ = other._impl->handle_;
        _impl->dtype_ = other._impl->dtype_;
        _impl->shape_ = other._impl->shape_;
        _impl->device_id_ = other._impl->device_id_;
        _impl->own_sys_data_ = other._impl->own_sys_data_;
        _impl->own_dev_data_ = other._impl->own_dev_data_;
        _impl->own_sys_data_is_mmap_ = other._impl->own_sys_data_is_mmap_;
        int type_size = get_type_size(_impl->dtype_);
        _impl->data_size_ = std::accumulate(_impl->shape_.begin(), _impl->shape_.end(),
                                     type_size, std::multiplies<int>());
        if (_impl->own_dev_data_) {
#if BMCV_VERSION_MAJOR > 1
            int ret = bm_malloc_device_byte_heap_mask(_impl->handle_.data(), &_impl->dev_data_, 3, _impl->data_size_);
#else
            int ret = bm_malloc_device_byte_heap_mask(_impl->handle_.data(), &_impl->dev_data_, 7, _impl->data_size_);
#endif
            if (BM_SUCCESS != ret) {
                SPDLOG_ERROR("bm_malloc_device_byte_heap_mask() err={}", ret);
                exit(SAIL_ERR_TENSOR_INIT);
            }
        }else{
            _impl->dev_data_ = other._impl->dev_data_;
        }

#ifndef IS_SOC_MODE
        if (_impl->own_sys_data_) {
            _impl->sys_data_ = malloc(_impl->data_size_);
            memcpy(_impl->sys_data_, other._impl->sys_data_,_impl->data_size_);
            double process_start_time = get_current_time_us();
            bm_memcpy_s2d(_impl->handle_.data(), _impl->dev_data_, _impl->sys_data_);
            PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
        } else {
            void *tmp = malloc(_impl->data_size_);
            double process_start_time_d2s = get_current_time_us();
            bm_memcpy_d2s(_impl->handle_.data(), tmp, other._impl->dev_data_);
            PRINT_TIME_MS("bm_memcpy_d2s", process_start_time_d2s)
            double process_start_time = get_current_time_us();
            bm_memcpy_s2d(_impl->handle_.data(), _impl->dev_data_, tmp);
            PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
            std::free(tmp);
        }
#else
        if (_impl->own_sys_data_) {
          if (_impl->own_dev_data_) {
            bm_mem_mmap_device_mem(_impl->handle_.data(), &_impl->dev_data_,
                                   (unsigned long long*)&_impl->sys_data_);
            _impl->own_sys_data_is_mmap_ = true;
          } else {
            _impl->sys_data_ = malloc(_impl->data_size_);
          }
          memcpy(_impl->sys_data_, other._impl->sys_data_, _impl->data_size_);
        } else {
            void* tmp = malloc(_impl->data_size_);
            double process_start_time_d2s = get_current_time_us();
            bm_memcpy_d2s(_impl->handle_.data(), tmp, other._impl->dev_data_);
            PRINT_TIME_MS("bm_memcpy_d2s", process_start_time_d2s)
            double process_start_time = get_current_time_us();
            bm_memcpy_s2d(_impl->handle_.data(), _impl->dev_data_, tmp);
            PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
            std::free(tmp);
        }
#endif
    }

    Tensor::Tensor(const Tensor& other, const std::vector<std::pair<int, int>> &ranges, bool d2d_flag)
    :_impl(new Tensor_CC()){
        std::vector<sail::Range> cl_ranges;
        for (auto range: ranges){
            cl_ranges.emplace_back(range.first, range.second);
        } 
        bm_device_mem_t new_dev_data = other._impl->slice(cl_ranges, d2d_flag);
        std::vector<int> shape;
        for (auto range: cl_ranges){
            shape.emplace_back(range.size());
        }
        _impl->handle_ = other._impl->handle_;
        _impl->dtype_ = other._impl->dtype_;
        _impl->shape_ = shape;
        _impl->own_sys_data_ = false;
        _impl->own_dev_data_ = true;
        _impl->own_sys_data_is_mmap_ = false;
        _impl->sys_data_ = nullptr;
        _impl->dev_data_ = new_dev_data;
        _impl->data_size_ = bm_mem_get_device_size(new_dev_data);
    }

    Tensor::Tensor(Tensor &&other):_impl(new Tensor_CC()) {
        *this = std::move(other);
    }

    Tensor &Tensor::operator=(const Tensor &other) {
        if (this != &other) {
            free();
            _impl->handle_ = other._impl->handle_;
            _impl->dtype_ = other._impl->dtype_;
            _impl->shape_ = other._impl->shape_;
            _impl->device_id_ = other._impl->device_id_;
            _impl->own_sys_data_ = other._impl->own_sys_data_;
            _impl->own_dev_data_ = other._impl->own_dev_data_;
            _impl->own_sys_data_is_mmap_ = other._impl->own_sys_data_is_mmap_;
            int type_size = get_type_size(_impl->dtype_);
            _impl->data_size_ = std::accumulate(_impl->shape_.begin(), _impl->shape_.end(),
                                         type_size, std::multiplies<int>());
            if (_impl->own_dev_data_) {
#if BMCV_VERSION_MAJOR > 1
                int ret = bm_malloc_device_byte_heap_mask(_impl->handle_.data(), &_impl->dev_data_, 3, _impl->data_size_);
#else
                int ret = bm_malloc_device_byte_heap_mask(_impl->handle_.data(), &_impl->dev_data_, 7, _impl->data_size_);
#endif
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("bm_malloc_device_byte_heap_mask() err={}", ret);
                    exit(SAIL_ERR_TENSOR_INIT);
                }
            }
#ifndef IS_SOC_MODE
            if (_impl->own_sys_data_) {
                _impl->sys_data_ = malloc(_impl->data_size_);
                memcpy(_impl->sys_data_, other._impl->sys_data_, _impl->data_size_);
                if (_impl->own_dev_data_) {
                    double process_start_time = get_current_time_us();
                    bm_memcpy_s2d(_impl->handle_.data(), _impl->dev_data_, _impl->sys_data_);
                    PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
                }
            } else {
                void *tmp = malloc(_impl->data_size_);
                double process_start_time_d2s = get_current_time_us();
                bm_memcpy_d2s(_impl->handle_.data(), tmp, other._impl->dev_data_);
                PRINT_TIME_MS("bm_memcpy_d2s", process_start_time_d2s)
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(_impl->handle_.data(), _impl->dev_data_, tmp);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
                std::free(tmp);
            }
#else
            if (_impl->own_sys_data_) {
              if (_impl->own_dev_data_) {
                bm_mem_mmap_device_mem(_impl->handle_.data(), &_impl->dev_data_,
                                       (unsigned long long*)&_impl->sys_data_);
                _impl->own_sys_data_is_mmap_ = true;
              } else {
                _impl->sys_data_ = malloc(_impl->data_size_);
              }
              memcpy(_impl->sys_data_, other._impl->sys_data_, _impl->data_size_);
            } else {
                void* tmp = malloc(_impl->data_size_);
                double process_start_time_d2s = get_current_time_us();
                bm_memcpy_d2s(_impl->handle_.data(), tmp, other._impl->dev_data_);
                PRINT_TIME_MS("bm_memcpy_d2s", process_start_time_d2s)
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(_impl->handle_.data(), _impl->dev_data_, tmp);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
                std::free(tmp);
            }
#endif
        }
        return *this;
    }

    Tensor &Tensor::operator=(Tensor &&other) {
        if (this != &other) {
            std::swap(_impl->handle_, other._impl->handle_);
            std::swap(_impl->dtype_, other._impl->dtype_);
            std::swap(_impl->shape_, other._impl->shape_);
            // std::swap(_impl->device_id_, other._impl->device_id_);
            _impl->device_id_ = other._impl->device_id_;
            if (own_dev_data()){
                std::swap(_impl->own_dev_data_, other._impl->own_dev_data_);
            }else{
                _impl->own_dev_data_=other._impl->own_dev_data_;
            }
            if (own_sys_data()){
                std::swap(_impl->own_sys_data_, other._impl->own_sys_data_);
            }else{  
                _impl->own_sys_data_, other._impl->own_sys_data_;
            }
            std::swap(_impl->sys_data_, other._impl->sys_data_);
            std::swap(_impl->dev_data_, other._impl->dev_data_);
            std::swap(_impl->data_size_, other._impl->data_size_);
            std::swap(_impl->own_sys_data_is_mmap_, other._impl->own_sys_data_is_mmap_);
        }
        return *this;
    }

    void Tensor::free() {
        if(_impl) _impl->free();
    }

    Tensor::~Tensor() {
        if(_impl){
            free();
            delete _impl;
        }
    }

    Handle &Tensor::get_handle() {
        return _impl->handle_;
    }

    int Tensor::device_id() {
        return _impl->device_id_;
    }

    const std::vector<int> &Tensor::shape() const {
        return _impl->shape_;
    }

    bm_data_type_t Tensor::dtype() const {
        return _impl->dtype_;
    }

    void Tensor::reset(const std::vector<int> &shape, bm_data_type_t dtype) {
        return _impl->reset(shape, dtype);
    }

    void Tensor::reshape(const std::vector<int> &shape) {
        return reset(shape, _impl->dtype_);
    }

    void Tensor::reset_sys_data(void *data, std::vector<int> &shape) {
        return _impl->reset_sys_data(data, shape);
    }

    void Tensor::reset_dev_data(bm_device_mem_t data) {
        return _impl->reset_dev_data(data);
    }

    bool &Tensor::own_sys_data() {
        return _impl->own_sys_data_;
    }

    bool &Tensor::own_dev_data() {
        return _impl->own_dev_data_;
    }

    bm_device_mem_t Tensor::dev_data() {
        return _impl->dev_data_;
    }

    void *Tensor::sys_data() {
        return _impl->sys_data_;
    }

    void Tensor::sync_s2d() {
        return _impl->sync_s2d();
    }

    void Tensor::sync_s2d(int size) {
        return _impl->sync_s2d(size);
    }

    void Tensor::sync_d2s() {
                return _impl->sync_d2s();
    }

    void Tensor::sync_d2s(int size) {
                return _impl->sync_d2s(size);
    }

    void Tensor::sync_d2d(Tensor& src, int offset_src, int offset_dst, int len){
                return _impl->sync_d2d(src._impl,offset_src,offset_dst,len);
    }

    void Tensor::sync_d2d_stride(Tensor& src, int stride_src, int stride_dst, int count){
                return _impl->sync_d2d_stride(src._impl, stride_src, stride_dst, count);
    }

    void Tensor::sync_d2s(Tensor& src, int offset_src, int offset_dst, int len){
                return _impl->sync_d2s(src._impl,offset_src,offset_dst,len);
    }

    void Tensor::sync_s2d(Tensor& src, int offset_src, int offset_dst, int len){
        return _impl->sync_s2d(src._impl,offset_src,offset_dst,len);
    }
    
    
    void Tensor::sync_from(Tensor *src) {
        return _impl->sync_from(src->_impl);
    }

    void Tensor::sync_to(Tensor *dst) {
        return _impl->sync_to(dst->_impl);
    }

    void Tensor::scale_from(float *src, float scale) {
        int size = std::accumulate(_impl->shape_.begin(), _impl->shape_.end(),
                                   1, std::multiplies<int>());
        scale_from(src, scale, size);
    }

    void Tensor::dump_data(std::string file_name,bool bin){
        return _impl->dump_data(file_name, bin);
    }

    int Tensor::size()
    {
        return std::accumulate(this->shape().begin(), this->shape().end(), 1, std::multiplies<int>());
    }

#if USE_ASM_SSE

    void Tensor::scale_from(float *src, float scale, int size) {
        if (nullptr == _impl->sys_data_) {
            spdlog::error("When call scale_from, own_sys_data must be true");
            exit(EXIT_FAILURE);
        }
        double process_start_time_scale = get_current_time_us();
        AnyScale_SSE(src, BM_FLOAT32, _impl->sys_data_, _impl->dtype_, scale, size);
        PRINT_TIME_MS("AnyScale_SSE", process_start_time_scale)
    }

    void Tensor::scale_from_int32(int32_t *src, float scale, int size) {
        if (nullptr == _impl->sys_data_) {
            spdlog::error("When call scale_from_int32, own_sys_data must be true");
            exit(EXIT_FAILURE);
        }
        double process_start_time_scale = get_current_time_us();
        AnyScale_SSE(src, BM_INT32, _impl->sys_data_, _impl->dtype_, scale, size);
        PRINT_TIME_MS("AnyScale_SSE", process_start_time_scale)
    }

    void Tensor::scale_to(float *dst, float scale, int size) {

        if (nullptr == _impl->sys_data_) {
            SPDLOG_ERROR("When call scale_to, own_sys_data must be true");
            exit(EXIT_FAILURE);
        }

        double process_start_time_scale = get_current_time_us();
        AnyScale_SSE(_impl->sys_data_, _impl->dtype_, dst, BM_FLOAT32, scale, size);
        PRINT_TIME_MS("AnyScale_SSE", process_start_time_scale)
    }

#else // don't use asm language.

    void Tensor::scale_from(float *src, float scale, int size) {
        double process_start_time_scale = get_current_time_us();
        AnyScale(src, BM_FLOAT32, _impl->sys_data_, _impl->dtype_, scale, size);
        PRINT_TIME_MS("AnyScale", process_start_time_scale)
    }

    void Tensor::scale_to(float *dst, float scale, int size) {
        double process_start_time_scale = get_current_time_us();
        AnyScale(_impl->sys_data_, _impl->dtype_, dst, BM_FLOAT32, scale, size);
        PRINT_TIME_MS("AnyScale", process_start_time_scale)
    }


    void Tensor::scale_from_int32(int32_t* src, float scale, int size) {
      if (nullptr == _impl->sys_data_) {
        spdlog::error("When call scale_from_int32, own_sys_data must be true");
        exit(EXIT_FAILURE);
      }
        double process_start_time_scale = get_current_time_us();
        AnyScale(src, BM_INT32, _impl->sys_data_, _impl->dtype_, scale, size);
        PRINT_TIME_MS("AnyScale", process_start_time_scale)
    }
#endif

    void Tensor::scale_to(float *dst, float scale) {
        int size = std::accumulate(_impl->shape_.begin(), _impl->shape_.end(),
                                   1, std::multiplies<int>());
        scale_to(dst, scale, size);
    }
    
    void Tensor::memory_set(float c){
        void* value;
        if(_impl->dtype_ == BM_FLOAT32){
            float f32_c = c;
            value = (void*)&f32_c;
        } else if (_impl->dtype_ == BM_INT32){
            int32_t i32_c = static_cast<int32_t>(c);
            value = (void*)&i32_c;
        } else if (_impl->dtype_ == BM_INT16){
            int16_t i16_c = static_cast<int16_t>(c);
            value = (void*)&i16_c;
        } else if (_impl->dtype_ == BM_INT8){
            int8_t i8_c = static_cast<int8_t>(c);
            value = (void*)&i8_c;
        } else if (_impl->dtype_ == BM_UINT32){
            uint32_t ui32_c = static_cast<uint32_t>(c);
            value = (void*)&ui32_c;
        } else if (_impl->dtype_ == BM_UINT16){
            uint16_t ui16_c = static_cast<uint16_t>(c);
            value = (void*)&ui16_c;
        } else if (_impl->dtype_ == BM_UINT8){
            uint8_t ui8_c = static_cast<uint8_t>(c);
            value = (void*)&ui8_c;
        } else if (_impl->dtype_ == BM_FLOAT16){
            uint16_t f16_c = fp32_to_fp16(c);
            value = (void*)&f16_c;
        } else if (_impl->dtype_ == BM_BFLOAT16){
            uint16_t bf16_c = fp32_to_bf16(c);
            value = (void*)&bf16_c;
        } else {
            SPDLOG_ERROR("memory_set: Unsupport dtype {}.", _impl->dtype_);
        }
        memory_set(value);
    }

#ifdef PYTHON
    void Tensor::memory_set(pybind11::object c){
        pybind11::array array = pybind11::array::ensure(c);
        if (!array) {
            SPDLOG_ERROR("The provided object is not a numpy scalar.");
        }
        if (array.size() != 1) {
            SPDLOG_ERROR("The provided object is not a numpy scalar.");
        }
        pybind11::dtype dtype = array.dtype();
        std::string dtype_str = dtype.str();
        // SPDLOG_INFO("dtype_str : {}", dtype_str);
        if(dtype_str == "float32" && _impl->dtype_ == BM_FLOAT32
        || dtype_str == "float16" && _impl->dtype_ == BM_FLOAT16
        || dtype_str == "int32"   && _impl->dtype_ == BM_INT32
        || dtype_str == "int16"   && _impl->dtype_ == BM_INT16
        || dtype_str == "int8"    && _impl->dtype_ == BM_INT8
        || dtype_str == "uint32"  && _impl->dtype_ == BM_UINT32
        || dtype_str == "uint16"  && _impl->dtype_ == BM_UINT16
        || dtype_str == "uint8"   && _impl->dtype_ == BM_UINT8){
            void* value = const_cast<void*>(array.data());
            memory_set(value);
        } else{
            if(dtype_str == "float32"){
                float float_value = *reinterpret_cast<float*>(const_cast<void*>(array.data()));
                memory_set(float_value);
            }else if(dtype_str == "float64"){ //python default type float64.
                double double_value = *reinterpret_cast<double*>(const_cast<void*>(array.data()));
                memory_set(static_cast<float>(double_value));
            }else if(dtype_str == "int64" && _impl->dtype_ == BM_INT32){ //python default type int64, use int32 can keep max precision
                long long long_value = *reinterpret_cast<long long*>(const_cast<void*>(array.data()));
                int32_t int32_value = static_cast<int32_t>(long_value);
                memory_set((void*)&int32_value);
            }else if(dtype_str == "int64" && _impl->dtype_ == BM_UINT32){
                long long long_value = *reinterpret_cast<long long*>(const_cast<void*>(array.data()));
                uint32_t uint32_value = static_cast<uint32_t>(long_value);
                memory_set((void*)&uint32_value);
            }else if(dtype_str == "int64"){
                long long long_value = *reinterpret_cast<long long*>(const_cast<void*>(array.data()));
                memory_set(static_cast<float>(long_value));
            }else{
                SPDLOG_ERROR("Input scalar 's dtype {} cannot be convert to tensor's dtype {}. Only support np.float32 conversion now.", dtype_str, _impl->dtype_);                
            }
        }
    }
#endif

    void Tensor::memory_set(void* value)
    {
        int memset_mode = 1;
        int value_size = 1;
        if(_impl->dtype_ == BM_INT8 
        || _impl->dtype_ == BM_UINT8){
            memset_mode = 1;
            value_size = 1;
        } else if (_impl->dtype_ == BM_FLOAT16 
        || _impl->dtype_ == BM_INT16 
        || _impl->dtype_ == BM_UINT16
        || _impl->dtype_ == BM_BFLOAT16){
            memset_mode = 2;
            value_size = 2;
        } else if (_impl->dtype_ == BM_FLOAT32 
        || _impl->dtype_ == BM_INT32 
        || _impl->dtype_ == BM_UINT32){
            memset_mode = 4;
            value_size = 4;
        } else {
            SPDLOG_ERROR("Tensor dtype {} do not support memory_set.", _impl->dtype_);
        }
#ifndef IS_SOC_MODE
        if(_impl->sys_data_){
            if(_impl->dtype_ == BM_INT8 || _impl->dtype_ == BM_UINT8){
                memset(_impl->sys_data_,*(int*)value,_impl->data_size_);            
            }else if(_impl->dtype_ == BM_INT32 || _impl->dtype_ == BM_UINT32 || _impl->dtype_ == BM_FLOAT32){
                std::fill((float*)_impl->sys_data_, (float*)_impl->sys_data_ + _impl->data_size_ / value_size, *(float*)value);
            }else if(_impl->dtype_ == BM_FLOAT16 || _impl->dtype_ == BM_INT16 || _impl->dtype_ == BM_UINT16){
                std::fill((uint16_t*)_impl->sys_data_, (uint16_t*)_impl->sys_data_ + _impl->data_size_ / value_size, *(uint16_t*)value);
            }else{
                SPDLOG_ERROR("Tensor dtype {} do not support memory_set.", _impl->dtype_);
            }
        }
        if(_impl->is_dev_data_valid()){
            bm_memset_device_ext(_impl->handle_.data(), value, memset_mode, _impl->dev_data_);
        }
#else
        if(_impl->is_dev_data_valid()){
            bm_memset_device_ext(_impl->handle_.data(), value, memset_mode, _impl->dev_data_);
            if(_impl->sys_data_){
                bm_mem_invalidate_device_mem(_impl->handle_.data(), &_impl->dev_data_);
            }
        }
        else if(_impl->sys_data_){
            if(_impl->dtype_ == BM_INT8 || _impl->dtype_ == BM_UINT8){
                memset((char*)_impl->sys_data_,*(int*)value,_impl->data_size_);            
            }else if(_impl->dtype_ == BM_INT32 || _impl->dtype_ == BM_UINT32 || _impl->dtype_ == BM_FLOAT32){
                std::fill((float*)_impl->sys_data_, (float*)_impl->sys_data_ + _impl->data_size_ / value_size, *(float*)value);
            }else if(_impl->dtype_ == BM_FLOAT16 || _impl->dtype_ == BM_INT16 || _impl->dtype_ == BM_UINT16){
                std::fill((uint16_t*)_impl->sys_data_, (uint16_t*)_impl->sys_data_ + _impl->data_size_ / value_size, *(uint16_t*)value);
            }else{
                SPDLOG_ERROR("Tensor dtype {} do not support memory_set.", _impl->dtype_);
            }
        }
#endif
    }

    void Tensor::zeros(){
        memory_set((float)0);
    }

    void Tensor::ones(){
        memory_set((float)1);
    }

    inline bool Tensor::is_dev_data_valid() const {
        return _impl->is_dev_data_valid();
    }

    inline bool Tensor::is_sys_data_valid() const {
        return _impl->is_sys_data_valid();
    }
    
#ifdef PYTHON
    Tensor::Tensor(Handle handle, pybind11::array_t<float>&   data)
        :_impl (new Tensor_CC(handle, BM_FLOAT32, data.request(), 1)){
    }

    Tensor::Tensor(Handle handle, pybind11::array_t<int8_t>&  data)
        :_impl (new Tensor_CC(handle, BM_INT8, data.request(), 1)){
    }

    Tensor::Tensor(Handle handle, pybind11::array_t<uint8_t>& data)
        :_impl (new Tensor_CC(handle, BM_UINT8, data.request(), 1)){
    }
    
    Tensor::Tensor(Handle handle, pybind11::array_t<int32_t>& data)
        :_impl (new Tensor_CC(handle, BM_INT32, data.request(), 1)){
    }

    Tensor::Tensor(Handle handle, pybind11::array_t<float>&   data, bool own_sys_data, bool own_dev_data)
        :_impl (new Tensor_CC(handle, BM_FLOAT32, data.request(), own_sys_data, own_dev_data)){
    }

    Tensor::Tensor(Handle handle, pybind11::array_t<int8_t>&  data, bool own_sys_data, bool own_dev_data)
        :_impl (new Tensor_CC(handle, BM_INT8, data.request(), own_sys_data, own_dev_data)){
    }

    Tensor::Tensor(Handle handle, pybind11::array_t<uint8_t>& data, bool own_sys_data, bool own_dev_data)
        :_impl (new Tensor_CC(handle, BM_UINT8, data.request(), own_sys_data, own_dev_data)){
    }
    
    Tensor::Tensor(Handle handle, pybind11::array_t<int32_t>& data, bool own_sys_data, bool own_dev_data)
        :_impl (new Tensor_CC(handle, BM_INT32, data.request(), own_sys_data, own_dev_data)){
    }

    void Tensor::scale_from(pybind11::array_t<float> &data, float scale) {
        auto buf = data.request();
        int size = 1;
        for (auto it : buf.shape) {
            size *= static_cast<int>(it);
        }
        int tensor_size = std::accumulate(_impl->shape_.begin(), _impl->shape_.end(),
                                          1, std::multiplies<int>());
        if (size > tensor_size) {
            SPDLOG_ERROR("data size exceeds tensor size!");
            exit(SAIL_ERR_TENSOR_SIZE);
        }

        float* src = reinterpret_cast<float*>(buf.ptr);

        scale_from(src, scale, size);
    }

    void Tensor::scale_from(pybind11::array_t<int32_t> &data, float scale) {
        auto buf = data.request();
        int size = 1;
        for (auto it : buf.shape) {
            size *= static_cast<int>(it);
        }
        int tensor_size = std::accumulate(_impl->shape_.begin(), _impl->shape_.end(),
                                          1, std::multiplies<int>());
        if (size > tensor_size) {
            SPDLOG_ERROR("data size exceeds tensor size!");
            exit(SAIL_ERR_TENSOR_SIZE);
        }

        int32_t* src = reinterpret_cast<int32_t*>(buf.ptr);
        scale_from_int32(src, scale, size);
    }

    pybind11::array_t<float> Tensor::scale_to(float scale) {
        std::vector<ssize_t> shape;
        for (auto v : _impl->shape_) {
            shape.push_back(static_cast<ssize_t>(v));
        }
        auto ndarray = pybind11::array_t<float>(shape);
        float *dst = ndarray.mutable_data();
        scale_to(dst, scale);
        return std::move(ndarray);
    }

    pybind11::array_t<float> Tensor::scale_to(
            float scale,
            const std::vector<int> &shape) {
        int tensor_size = std::accumulate(_impl->shape_.begin(), _impl->shape_.end(),
                                          1, std::multiplies<int>());
        int size = std::accumulate(shape.begin(), shape.end(),
                                   1, std::multiplies<int>());
        std::vector<ssize_t> array_shape;
        for (auto v : shape) {
            array_shape.push_back(static_cast<ssize_t>(v));
        }
        auto ndarray = pybind11::array_t<float>(array_shape);
        if (size > tensor_size) {
            SPDLOG_ERROR("data size exceeds tensor size!");
            exit(SAIL_ERR_TENSOR_SIZE);
        }

        float *dst = ndarray.mutable_data();
        scale_to(dst, scale, size);

        return std::move(ndarray);
    }

    pybind11::object Tensor::asnumpy() {
        std::unique_ptr<uint8_t[]> ptr;
        void *data = _impl->sys_data_;
        if (!is_sys_data_valid()) {
            if (!is_dev_data_valid()) {
                SPDLOG_ERROR("asnumpy: sys_data=null and dev_data is null!");
                exit(SAIL_ERR_TENSOR_EMPTY);
            }
            ptr.reset(new uint8_t[_impl->data_size_]);
            data = ptr.get();
            double process_start_time_d2s = get_current_time_us();
            if (BM_SUCCESS != bm_memcpy_d2s_partial(_impl->handle_.data(), data, _impl->dev_data_, _impl->data_size_)) {
                SPDLOG_ERROR("bm_memcpy_d2s_partial() err");
            }
            PRINT_TIME_MS("bm_memcpy_d2s_partial", process_start_time_d2s)
        }

        // fill numpy array
        pybind11::ssize_t item_size = 1;
        std::string format;
        if (_impl->dtype_ == BM_FLOAT32) {
            item_size = sizeof(float);
            format = pybind11::format_descriptor<float>::format();
        } else if (_impl->dtype_ == BM_INT8) {
            item_size = sizeof(int8_t);
            format = pybind11::format_descriptor<int8_t>::format();
        } else if (_impl->dtype_ == BM_UINT8) {
            item_size = sizeof(uint8_t);
            format = pybind11::format_descriptor<uint8_t>::format();
        } else if (_impl->dtype_ == BM_INT32) {
            item_size = sizeof(int32_t);
            format = pybind11::format_descriptor<int32_t>::format();
        } else if (_impl->dtype_ == BM_UINT32) {
            item_size = sizeof(uint32_t);
            format = pybind11::format_descriptor<uint32_t>::format();
        } else if (_impl->dtype_ == BM_INT16){
            item_size = sizeof(int16_t);
            format = pybind11::format_descriptor<int16_t>::format();
        } else if (_impl->dtype_ == BM_UINT16 || _impl->dtype_ == BM_FLOAT16 || _impl->dtype_ == BM_BFLOAT16){
            item_size = sizeof(uint16_t);
            format = pybind11::format_descriptor<uint16_t>::format();
        }

        pybind11::ssize_t ndim = _impl->shape_.size();
        std::vector<pybind11::ssize_t> shape;
        for (auto it : _impl->shape_) {
            shape.push_back(it);
        }
        std::vector<pybind11::ssize_t> stride;
        for (size_t i = 1; i < _impl->shape_.size(); i++) {
            pybind11::ssize_t inner_stride = std::accumulate(shape.begin() + i,
                                                             shape.end(), item_size,
                                                             std::multiplies<pybind11::ssize_t>());
            stride.push_back(inner_stride);
        }
        stride.push_back(item_size);

        pybind11::buffer_info output_buf(data, item_size, format,
                                         ndim, shape, stride);
        if (_impl->dtype_ == BM_FLOAT32) {
            return std::move(pybind11::array_t<float>(output_buf));
        } else if (_impl->dtype_ == BM_INT8) {
            return std::move(pybind11::array_t<int8_t>(output_buf));
        } else if (_impl->dtype_ == BM_UINT8) {
            return std::move(pybind11::array_t<uint8_t>(output_buf));
        } else if (_impl->dtype_ == BM_INT32) {
            return std::move(pybind11::array_t<int32_t>(output_buf));
        } else if (_impl->dtype_ == BM_UINT32) {
            return std::move(pybind11::array_t<int32_t>(output_buf));
        } else if (_impl->dtype_ == BM_INT16){
            return std::move(pybind11::array_t<int16_t>(output_buf));
        } else if (_impl->dtype_ == BM_FLOAT16 || _impl->dtype_ == BM_BFLOAT16 || _impl->dtype_ == BM_UINT16){
            return std::move(pybind11::array_t<uint16_t>(output_buf));
        } else {
            return pybind11::cast<pybind11::none>(Py_None);;
        }
    }

    pybind11::object Tensor::asnumpy(const std::vector<int> &shape) {
        std::unique_ptr<uint8_t[]> ptr;
        void *data = _impl->sys_data_;
        if (!is_sys_data_valid()) {
            if (!is_dev_data_valid()) {
                SPDLOG_ERROR("asnumpy: sys_data=null and dev_data is null!");
                exit(SAIL_ERR_TENSOR_EMPTY);
            }
            ptr.reset(new uint8_t[_impl->data_size_]);
            data = ptr.get();
            double process_start_time_d2s = get_current_time_us();
            if (BM_SUCCESS != bm_memcpy_d2s_partial(_impl->handle_.data(), data, _impl->dev_data_, _impl->data_size_)) {
                SPDLOG_ERROR("bm_memcpy_d2s_partial() err");
            }
            PRINT_TIME_MS("bm_memcpy_d2s_partial", process_start_time_d2s)
        }
        // fill numpy array
        pybind11::ssize_t item_size = 1;
        std::string format;
        if (_impl->dtype_ == BM_FLOAT32) {
            item_size = sizeof(float);
            format = pybind11::format_descriptor<float>::format();
        } else if (_impl->dtype_ == BM_INT8) {
            item_size = sizeof(int8_t);
            format = pybind11::format_descriptor<int8_t>::format();
        } else if (_impl->dtype_ == BM_UINT8) {
            item_size = sizeof(uint8_t);
            format = pybind11::format_descriptor<uint8_t>::format();
        } else if (_impl->dtype_ == BM_INT32) {
            item_size = sizeof(int32_t);
            format = pybind11::format_descriptor<int32_t>::format();
        } else if (_impl->dtype_ == BM_UINT32) {
            item_size = sizeof(uint32_t);
            format = pybind11::format_descriptor<uint32_t>::format();
        } else if (_impl->dtype_ == BM_INT16){
            item_size = sizeof(int16_t);
            format = pybind11::format_descriptor<int16_t>::format();
        } else if (_impl->dtype_ == BM_UINT16 || _impl->dtype_ == BM_FLOAT16 || _impl->dtype_ == BM_BFLOAT16){
            item_size = sizeof(uint16_t);
            format = pybind11::format_descriptor<uint16_t>::format();
        }


        pybind11::ssize_t ndim = shape.size();
        std::vector<pybind11::ssize_t> stride;
        for (size_t i = 1; i < shape.size(); i++) {
            pybind11::ssize_t inner_stride = std::accumulate(shape.begin() + i,
                                                             shape.end(), item_size,
                                                             std::multiplies<pybind11::ssize_t>());
            stride.push_back(inner_stride);
        }
        stride.push_back(item_size);


        pybind11::buffer_info output_buf(data, item_size, format,
                                         ndim, shape, stride);
        if (_impl->dtype_ == BM_FLOAT32) {
            return std::move(pybind11::array_t<float>(output_buf));
        } else if (_impl->dtype_ == BM_INT8) {
            return std::move(pybind11::array_t<int8_t>(output_buf));
        } else if (_impl->dtype_ == BM_UINT8) {
            return std::move(pybind11::array_t<uint8_t>(output_buf));
        } else if (_impl->dtype_ == BM_INT32) {
            return std::move(pybind11::array_t<int32_t>(output_buf));
        } else if (_impl->dtype_ == BM_UINT32) {
            return std::move(pybind11::array_t<int32_t>(output_buf));
        } else if (_impl->dtype_ == BM_INT16){
            return std::move(pybind11::array_t<int16_t>(output_buf));
        } else if (_impl->dtype_ == BM_FLOAT16 || _impl->dtype_ == BM_BFLOAT16 || _impl->dtype_ == BM_UINT16){
            return std::move(pybind11::array_t<uint16_t>(output_buf));
        } else {
            return pybind11::cast<pybind11::none>(Py_None);;
        }
    }

    pybind11::array_t<long> Tensor::pysys_data() {
        //printf("pysys_data sys_data_=0x%x\n",sys_data_);

        std::vector<ssize_t> array_shape;
        array_shape.push_back(static_cast<ssize_t>(1));
        auto ndarray = pybind11::array_t<long>(array_shape);
        long *dst = ndarray.mutable_data();
        //long ldata = (long)(sys_data_);
        dst[0] = (long) _impl->sys_data_;
        //memcpy(dst, &ldata, 1 * sizeof(int));
        //printf("pysys_data sysd=0x%x\n",dst[0]);
        return std::move(ndarray);
    }

    void Tensor::update_data(const pybind11::buffer_info& buf, int type_size)
    {
        return _impl->update_data(buf,type_size);
    }

#endif

    void ReleaseTensorPtr(TensorPTRWithName& tensor_with_name)
    {
        if(tensor_with_name.data){
            delete tensor_with_name.data;
            tensor_with_name.data = NULL;
        }
    }

    /**
     * @brief Converts a LogLevel enum value to its corresponding string representation.
     *
     * @param loglevel The log level to be converted to a string.
     * @return std::string A string representation of the LogLevel.
     *         If the log level does not match any of the known cases, it returns "unknown".
     */
    std::string level_to_string(LogLevel loglevel)
    {
        switch (loglevel)
        {
        case LogLevel::TRACE:
            return "trace";
        case LogLevel::DEBUG:
            return "debug";
        case LogLevel::INFO:
            return "info";
        case LogLevel::WARN:
            return "warn";
        case LogLevel::ERROR:
            return "error";
        case LogLevel::CRITICAL:
            return "critical";
        case LogLevel::OFF:
            return "off";
        default:
            return "unknown";
        }
    }

    int set_loglevel(LogLevel loglevel)
    {
        std::string level_str = level_to_string(loglevel);
        if (level_str == "unknown")
        {
            spdlog::error("set_loglevel() failed. The input loglevel is unknown!");
            return -1;
        }
        spdlog::set_level(
            static_cast<spdlog::level::level_enum>(
                static_cast<int>(loglevel)));
        spdlog::info("set loglevel to {}", level_str);
        return 0;
    }

    std::string ret_to_string(int ret)
    {
        auto desc = MAP_RET_DESC.find(static_cast<sail_status_t>(ret));
        if(desc != MAP_RET_DESC.end()){
            return desc->second;
        }
        return std::string("");
    }

    int sail_status_to_exception_type(int ret)
    {
        return ret / 1000;
    }

    void check_return_status(int ret)
    {
        if (!ret) return;

        int exception_type = sail_status_to_exception_type(ret); // 100x -> 1 200x -> 2
        switch (exception_type)
        {
            case 1:
                throw SailDeviceError("Device related error!");
            case 2:
                throw SailTensorError("Tensor related error!");
            case 3:
                throw SailEngineError("Engine related error!");
            case 4:
                throw SailBMImageError("BMImage related error!");
            case 5:
                throw SailDecoderError("Decoder related error!");
            case 6:
                throw SailEncoderError("Encoder related error!");
            case 7:
                throw SailRuntimeError("Runtime related error!");
            default:
                // legacy return like 1 2 -1 ...
                throw SailUnknownError("Error occurred with status code " + std::to_string(ret));
        }
    }

}  // namespace sail
