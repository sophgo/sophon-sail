/* Copyright 2016-2022 by SOPHGO Technologies Inc. All rights reserved.

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

#include <spdlog/spdlog.h>
#include <decoder_multi.h>
#if defined(USE_BMCV) && defined(USE_OPENCV) && defined(USE_FFMPEG)
#include <opencv2/opencv.hpp>
#include <engine.h>
#include <numeric>
#include <sys/types.h>
#include <fstream>
#ifdef WIN
#else
#include <unistd.h>
#endif
#include "internal.h"


using namespace std;

namespace sail {
    static mutex md_print_flag_;
    void PrintThreadLog_Decoder(std::string file_name, int line, std::string message)
    {
        // if(md_print_flag_){
            std::lock_guard<std::mutex> lock_print(md_print_flag_);
            std::cout << "# File[" << file_name << ":" << line << "], ";
            std::cout << "Thread[" << std::this_thread::get_id()<<"], "<< message << std::endl;
        // }
    }

    class BMImageQueue{
    public:
        BMImageQueue(int max_queue_size);
        ~BMImageQueue();
        int Push(BMImage& image);   //压进数据
        int Pop(BMImage& image);    //取出第一张数据    
        int Pop();                  //丢弃第一张数据       
        bool IsEmpty();             //判断队列是否为空
        bool IsFull();              //判断队列是否已满
        int Reset();                //重置队列
        int GetMaxQueueSize();      //获取队列的最大长度
        int Size();                 //当前有效数据的长度
    private:
        int head_idx;           //头部元素的索引
        int end_idx;            //尾部元素的索引
        int que_size;           //队列长度
        bool empty_flag;        //d队列为空的标志位
        std::mutex mutex_data;  //互斥
        BMImage* ptr_;          //数据        
    };

    BMImageQueue::BMImageQueue(int max_queue_size)
    :head_idx(0),end_idx(-1),que_size(max_queue_size),empty_flag(true)
    {
        ptr_ = new BMImage[que_size];
    }

    BMImageQueue::~BMImageQueue(){
        Reset();
        std::lock_guard<std::mutex> lock(mutex_data);
        delete[] ptr_;
        ptr_ = NULL;
    }

    bool BMImageQueue::IsEmpty(){
        std::lock_guard<std::mutex> lock(mutex_data);
        return empty_flag;
    }

    bool BMImageQueue::IsFull(){
        std::lock_guard<std::mutex> lock(mutex_data);
        if(!empty_flag &&  head_idx+que_size-1 == end_idx){
            return true;
        }
        if(!empty_flag &&  head_idx-1 == end_idx){
            return true;
        }
        return false;
    }

    int BMImageQueue::Pop()
    {
        if(IsEmpty()){
            return 1;
        }
        std::lock_guard<std::mutex> lock(mutex_data);
        head_idx++;
        if(head_idx == end_idx+1){
            empty_flag = true;
        }
        if(head_idx == que_size){
            head_idx = 0;
        }
        return 0;
    }  

    int BMImageQueue::Pop(BMImage& image)
    {
        if(IsEmpty()){
            return 1;
        }
        std::lock_guard<std::mutex> lock(mutex_data);
        image = std::move(ptr_[head_idx]);
        head_idx++;
        if(head_idx == end_idx+1){
            empty_flag = true;
        }
        if(head_idx == que_size){
            head_idx = 0;
        }
        return 0;
    }    

    int BMImageQueue::Push(BMImage& image)
    {
        if(IsFull()){
            return 1;
        }
        std::lock_guard<std::mutex> lock(mutex_data);
        end_idx++;
        if(end_idx == que_size){
            end_idx = 0;
        }
        ptr_[end_idx] = std::move(image);
        empty_flag = false;
        return 0;
    }

    int BMImageQueue::Reset(){
        std::lock_guard<std::mutex> lock(mutex_data);
        head_idx = 0;
        end_idx = -1;
        empty_flag = true;
        return 0;
    }

    int BMImageQueue::GetMaxQueueSize()
    {
        std::lock_guard<std::mutex> lock(mutex_data);
        return que_size;
    }

    int BMImageQueue::Size()
    {
        if(IsEmpty()) return 0;
        std::lock_guard<std::mutex> lock(mutex_data);
        int size = end_idx-head_idx+1;
        if (size <= 0){
            size += que_size;
        }
        return size;
    }

    class ChannelDecoder{
    public:
        /** 
         * @brief Constructor
         * @param file_path Path or rtsp url to the video/image file.
         * @param tpu_id    TPU ID. You can use bm-smi to see available IDs.
         * @param queue_size max queue size
         * @param frame_skip_num frame skips numbers
         * @param channel_idx channel index
         * @param timeout_sec timeouts  
         * @param discard_mode discard mode.
         * @param discard_mode discard mode.
        */
        explicit ChannelDecoder( 
            const std::string&  file_path,
            int                 tpu_id,
            int                 queue_size =  10,
            int                 frame_skip_num = 0,
            int                 channel_idx = 0,
            int                 timeout_sec = 10, 
            int                 discard_mode = 0, 
            bool                local_video_flag = false);

        /**
         * @brief Destructor.
         */
        ~ChannelDecoder();

        int clear_queue();

        int read(BMImage &image, int read_mode = 0);

        int read_(bm_image &image, int read_mode = 0);

        int reconnect();

        std::vector<int> get_frame_shape();

        void set_local_flag(bool flag);     // 设置本地测试标志位

        Decoder* get_dec(); //获取decoder
        
        size_t get_drop_num();

        void reset_drop_num();

        DecoderStatus get_decoder_status();

    private:
        int tpu_id_;
        int queue_size_;
        int skip_number_;
        int timeout_sec_;
        int discard_mode_;
        int channel_idx_;

        BMImageQueue* pframes_queue;

        std::mutex mutex_decoder_c;

        std::condition_variable data_flag_cond;
        std::mutex mutex_data_flag;

        Decoder *decoder_;

        bool local_video_flag_;     // 本地视频的标志位,此标志位为1时解码会sleep,每秒只解码25fps


        bool stop_thread_flag;      //线程退出的标志位
        std::mutex mutex_stop_;     //线程退出互斥锁
        std::condition_variable exit_cond;  //线程已经退出的信号量
        std::mutex mutex_exit_;             //线程已经退出互斥锁
        bool exit_thread_flag;      //线程已经退出的标志位
        size_t drop_num;
        DecoderStatus decoder_status;

    private:
        void decoder_thread();      //解码线程

        void set_stop_flag(bool flag);       //设置线程退出的标志位

        bool get_stop_flag();       //获取线程退出的标志位

        void set_thread_exit();       //设置线程已经退出         

        void wait_thread_exit();      //等待线程退出

        void notify_data_flag();       //发送已经有缓存数据的信号
    
        int get_frame(BMImage &image);   //从队列中获取一张图片,成功返回0,否则返回其它

        sail_status_t set_decoder_status(DecoderStatus status);
    };

    ChannelDecoder::ChannelDecoder( 
            const std::string&  file_path,
            int                 tpu_id,
            int                 queue_size,
            int                 frame_skip_num,
            int                 channel_idx,
            int                 timeout_sec, 
            int                 discard_mode,
            bool                local_video_flag):
    tpu_id_(tpu_id),queue_size_(queue_size),skip_number_(frame_skip_num),
    timeout_sec_(timeout_sec), discard_mode_(discard_mode),stop_thread_flag(false),
    exit_thread_flag(true),decoder_(NULL),local_video_flag_(local_video_flag),
    channel_idx_(channel_idx),pframes_queue(NULL),drop_num(0),
    decoder_status(sail::DecoderStatus::NONE)
    {
        pframes_queue = new BMImageQueue(queue_size);
        decoder_ = new sail::Decoder(file_path,true,tpu_id_);
        if(decoder_ == NULL){
            SPDLOG_ERROR("Decoder failed! file_path:{}, tpu_id:{}!",file_path,tpu_id);
            throw SailDecoderError("invalid argument");
        }
        if(!decoder_->is_opened()){
            SPDLOG_ERROR("Decoder failed! file_path:{}, tpu_id:{}!",file_path,tpu_id);
            throw SailDecoderError("invalid argument");
        }
        sail_status_t ret = set_decoder_status(sail::DecoderStatus::OPENED);
        std::thread thread_decoder = std::thread(&ChannelDecoder::decoder_thread,this);
        thread_decoder.detach();
    }


    ChannelDecoder::~ChannelDecoder(){
        SPDLOG_INFO("Channel-{},Set Stop Flag!",channel_idx_);
        set_stop_flag(true);
        SPDLOG_INFO("Channel-{},Wait Thread Finshed: {}!",channel_idx_,get_stop_flag());
        wait_thread_exit();
        SPDLOG_INFO("Channel-{},End Thread Finshed!",channel_idx_);
        if(decoder_){
            SPDLOG_INFO("Channel-{},Start delete decoder_!",channel_idx_);
            delete decoder_;
            SPDLOG_INFO("Channel-{},End delete decoder_!",channel_idx_);
            decoder_ = NULL;
        }
        SPDLOG_INFO("Channel-{},All end!",channel_idx_);
        delete pframes_queue;
        pframes_queue = NULL;
    }

    int ChannelDecoder::clear_queue()
    {
        if(pframes_queue){
            return pframes_queue->Reset();
        }
        return 1;
    }

    int ChannelDecoder::read(BMImage &image, int read_mode)
    {
// #ifdef PYTHON
//         pybind11::gil_scoped_release gil_lock;
// #endif
        int ret = get_frame(image);
        if(read_mode == 0){
            return ret;
        }else if(ret != 0){
            {
                if(get_stop_flag()){
                    return -1;
                } 
            }

            double time_start = get_current_time_us();
            std::unique_lock<std::mutex> lck(mutex_data_flag);
            data_flag_cond.wait_for(lck,std::chrono::seconds(timeout_sec_));
            double time_use = (get_current_time_us() - time_start)/1000;

            char str_output[256] ={0};
            sprintf(str_output,"channel:[%d],Wait: %.2fms.", channel_idx_, time_use); 
            PrintThreadLog_Decoder(__FILE__, __LINE__, str_output);

            return get_frame(image);
        }
    }

    int ChannelDecoder::read_(bm_image &image, int read_mode)
    {
        BMImage image_temp;
        int ret = read(image_temp,read_mode);
        if (ret == 0){
            image= std::move(image_temp.data());
        }
        return ret;
    }

    int ChannelDecoder::reconnect()
    {
        std::lock_guard<std::mutex> lock(mutex_decoder_c);
        drop_num = 0;
        return decoder_->reconnect();
    }

    std::vector<int> ChannelDecoder::get_frame_shape()
    {
        std::lock_guard<std::mutex> lock(mutex_decoder_c);
        return decoder_->get_frame_shape();
    }

    void ChannelDecoder::decoder_thread()
    {
        sail::Handle handle(tpu_id_);
        int skip_count = 0;
        {        
            std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
            exit_thread_flag = false;
        }
//         SPDLOG_INFO("Create Decoder Thread, Channel:{}, pid:{}, tid:{} .",channel_idx_,getpid(),gettid());
        while(true) {
            double time_start = get_current_time_us();
            if(get_stop_flag()){
                break;
            }       
            sail::BMImage image_temp;
            int ret = 0;
            {           
                std::lock_guard<std::mutex> lock(mutex_decoder_c);
                ret = decoder_->read(handle, image_temp);
            }
            if (ret != 0)    {
                SPDLOG_INFO("Decoder read end or err={}", ret);
                drop_num = 0;
                try
                {
                    decoder_->reconnect(); 
                    SPDLOG_INFO("Try to reconnect channel {} .", channel_idx_);
                }
                catch (SailDecoderError)
                {
                    // set status to CLOSED
                    auto ret = set_decoder_status(sail::DecoderStatus::CLOSED);
                    SPDLOG_ERROR("Channel {} reconnection failed. This channel will be closed.", channel_idx_);
                    set_stop_flag(true);
                }
                continue;
            }
            bool notify_flag = false;
            if(skip_count >= skip_number_)  {
                skip_count = 0;
                if(!pframes_queue->IsFull()){
                    pframes_queue->Push(image_temp);
                }else if (discard_mode_ != 0){
                    pframes_queue->Pop();
                    drop_num++;
                    pframes_queue->Push(image_temp);

                }else{
                    drop_num++;
                    // char str_output[256] ={0};
                    // sprintf(str_output,"channle:[%d],pframes_queue size:[%d]", channel_idx_, pframes_queue.size()); 
                    // PrintThreadLog_Decoder(__FILE__, __LINE__, str_output);
                }
                notify_flag = true;
            }else{
                // SPDLOG_INFO("skip_count: {}", skip_count);
                skip_count++;
            }
            if(notify_flag){
                notify_data_flag();
            }
            if(local_video_flag_){
                double time_use = (get_current_time_us()-time_start)/1000;
                int time_sleep = (39.0 - time_use);
                if(time_sleep > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(time_sleep));
                }
            }
        }
        set_thread_exit();
        {
            std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
            exit_thread_flag = true;
        }
        SPDLOG_INFO("<<<<<<<<<<<<<<<<<<<<<<<<<<Channel-{},Decoder Thread Finshed!",channel_idx_);
    }

    int ChannelDecoder::get_frame(BMImage &image)
    {
        return pframes_queue->Pop(image);
    }

    sail_status_t ChannelDecoder::set_decoder_status(DecoderStatus status)
    {
        if (status < sail::DecoderStatus::NONE || status >= sail::DecoderStatus::STATUS_MAX)
        {
            return SAIL_ERR_DEC_BASIC;
        }
        decoder_status = status;
        return SAIL_SUCCESS;
    }

    Decoder* ChannelDecoder::get_dec(){
        return decoder_;
    }

    void ChannelDecoder::notify_data_flag()
    {
        std::unique_lock<std::mutex> lck(mutex_data_flag);
        data_flag_cond.notify_all();
    }

    bool ChannelDecoder::get_stop_flag()
    {
        std::lock_guard<std::mutex> lock(mutex_stop_);
        return stop_thread_flag;
    }

    void ChannelDecoder::set_stop_flag(bool flag)
    {
        std::lock_guard<std::mutex> lock(mutex_stop_);
        stop_thread_flag = flag;
    }

    void ChannelDecoder::set_thread_exit()
    {
        std::unique_lock<std::mutex> lck(mutex_exit_);
        exit_cond.notify_all();
    }   

    void ChannelDecoder::wait_thread_exit()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_exit_);
            if(exit_thread_flag){
                return;
            }
        }
        std::unique_lock<std::mutex> lck(mutex_exit_);
        exit_cond.wait(lck);
    }

    size_t ChannelDecoder::get_drop_num(){
        return drop_num;
    }

    void ChannelDecoder::reset_drop_num(){
        drop_num = 0;
    }

    DecoderStatus ChannelDecoder::get_decoder_status()
    {
        return decoder_status;
    }

    class MultiDecoder::MultiDecoder_CC{
    public:
        MultiDecoder_CC(
            int queue_size=10,
            int tpu_id=0, 
            int discard_mode=0)
            :timeout_sec_(10),queue_size_(queue_size),tpu_id_(tpu_id),
            discard_mode_(discard_mode),channel_count_(-1),local_video_flag_(false)
            {};

        ~MultiDecoder_CC(){
            std::lock_guard<std::mutex> lock(mutex_decoder);
            auto iter_temp = decoder_map_.begin();
            while(iter_temp != decoder_map_.end()){
                ChannelDecoder* temp_decoder = iter_temp->second;

                SPDLOG_INFO("Start delete channel {}",iter_temp->first);
                delete temp_decoder;
                SPDLOG_INFO("End delete channel {}",iter_temp->first);
                temp_decoder = NULL;
                iter_temp++;
            }
            decoder_map_.clear();
        }

        void set_read_timeout(int time_second)
        {
            timeout_sec_ = time_second;
        }

        int add_channel(const std::string& file_path, int frame_skip_num)
        {
            ChannelDecoder* decoder = new ChannelDecoder(file_path, 
                                                tpu_id_,
                                                queue_size_,
                                                frame_skip_num,
                                                channel_count_+1,
                                                timeout_sec_,
                                                discard_mode_,
                                                local_video_flag_);
            if (decoder == NULL){
                return -1;
            }
            insert_channel_decoder(decoder);  
            SPDLOG_INFO("Add video: {}, channel index: {}", file_path, channel_count_);
            return channel_count_;
        }

        int del_channel(int channel_idx){
            auto iter_temp = decoder_map_.find(channel_idx);
            if (iter_temp == decoder_map_.end()){
                SPDLOG_INFO("Can not found channel: {}", channel_idx);
                return 1;
            }
            ChannelDecoder* temp_decoder = iter_temp->second;
            if (temp_decoder){
                delete temp_decoder;
                temp_decoder = NULL;
            }
            decoder_map_.erase(iter_temp);
            return 0;
        }

        float get_channel_fps(int channel_idx){
            ChannelDecoder* decoder_temp = get_decoder(channel_idx);
            if(decoder_temp){
                Decoder* dec_tmp = decoder_temp->get_dec();
                return dec_tmp->get_fps();
            }
            return 1;
        }

        int clear_queue(int channel_idx){   //清空指定通道的缓存
            ChannelDecoder* decoder_temp = get_decoder(channel_idx);
            if(decoder_temp){
                return decoder_temp->clear_queue();
            }
            return 1;
        }  

        void set_local_flag(bool flag)
        {
            local_video_flag_ = flag;
        }

        int read(int channel_idx,BMImage& image,int read_mode)
        {
            ChannelDecoder* decoder_temp = get_decoder(channel_idx);
            if(decoder_temp){
               return decoder_temp->read(image, read_mode);
            }
            return -1;
        }

        BMImage read(int channel_idx){
            BMImage image;
            ChannelDecoder* decoder_temp = get_decoder(channel_idx);
            if(decoder_temp){
                decoder_temp->read(image,1);
            }
            return std::move(image);
        }	

        int read_(int channel_idx,bm_image& image,int read_mode)
        {
            ChannelDecoder* decoder_temp = get_decoder(channel_idx);
            if(decoder_temp){
               return decoder_temp->read_(image, read_mode);
            }
            return -1;
        }

        bm_image read_(int channel_idx){
            bm_image image;
            ChannelDecoder* decoder_temp = get_decoder(channel_idx);
            if(decoder_temp){
                decoder_temp->read_(image);
            }
            return std::move(image);
        }	

        int reconnect(int channel_idx){
            ChannelDecoder* decoder_temp = get_decoder(channel_idx);
            if(decoder_temp){
                return decoder_temp->reconnect();
            }
            return 1;
        }

        std::vector<int> get_frame_shape(int channel_idx)
        {
            ChannelDecoder* decoder_temp = get_decoder(channel_idx);
            if(decoder_temp){
                return decoder_temp->get_frame_shape();
            }
            std::vector<int> shape_temp;
            return shape_temp;
        }

        size_t get_drop_num(int channel_idx)
        {
            ChannelDecoder* decoder_temp = get_decoder(channel_idx);
            return decoder_temp->get_drop_num();
        }

        void reset_drop_num(int channel_idx)
        {
            ChannelDecoder* decoder_temp = get_decoder(channel_idx);
            decoder_temp->reset_drop_num();
            return ;
        }

        DecoderStatus get_channel_status(int channel_idx)
        {
            ChannelDecoder* decoder_idx = get_decoder(channel_idx);
            if (decoder_idx == nullptr)
            {
                SPDLOG_ERROR("Can not get status of channel {}", channel_idx);
                return sail::DecoderStatus::NONE;
            }
            return decoder_idx->get_decoder_status();
        }

    private:
        int timeout_sec_;       // read 超时的时间
        int queue_size_;        // 最大的队列长度
        int tpu_id_;            // 使用设备ID
        int discard_mode_;      // 缓存队列过长时的图片丢弃策略
        int channel_count_;     // number of channels

        std::map<int, ChannelDecoder*> decoder_map_;    // 通道号对应的解码器
        std::mutex mutex_decoder;                       // decoder_map_对应的互斥锁

        bool local_video_flag_;     // 本地视频测试标志位

        friend class MultiDecoder;

    private:
        void insert_channel_decoder(ChannelDecoder* decoder)    //将解码器插入到map中
        {
            std::lock_guard<std::mutex> lock(mutex_decoder);
            channel_count_++;
            decoder_map_.insert(std::pair<int, ChannelDecoder*>(channel_count_, decoder));
        }

        ChannelDecoder* get_decoder(int channel_idx){
            std::lock_guard<std::mutex> lock(mutex_decoder);
            auto iter_temp = decoder_map_.find(channel_idx);
            if (iter_temp == decoder_map_.end()){
                SPDLOG_INFO("Can not found channel: {}", channel_idx);
                return NULL;
            }else{
                return iter_temp->second;
            }
        }
    };

    MultiDecoder::MultiDecoder(int queue_size, int tpu_id, int discard_mode):_impl(new MultiDecoder_CC(queue_size, tpu_id, discard_mode))
    {}

    MultiDecoder::~MultiDecoder()
    {
        delete _impl;
    }

    void MultiDecoder::set_read_timeout(int time_second)
    {
        return _impl->set_read_timeout(time_second);
    }

    int MultiDecoder::add_channel(const std::string& file_path, int frame_skip_num)
    {
        return _impl->add_channel(file_path,frame_skip_num);
    }

    int MultiDecoder::del_channel(int channel_idx)
    {
        return _impl->del_channel(channel_idx);
    }

    float MultiDecoder::get_channel_fps(int channel_idx)
    {
        return _impl->get_channel_fps(channel_idx);
    }

    int MultiDecoder::clear_queue(int channel_idx)
    {
        return _impl->clear_queue(channel_idx);
    }

    int MultiDecoder::read(int channel_idx,BMImage& image,int read_mode)
    {
        return _impl->read(channel_idx,image,read_mode);
    }

    BMImage MultiDecoder::read(int channel_idx)
    {
        return _impl->read(channel_idx);
    }

    int MultiDecoder::read_(int channel_idx,bm_image& image,int read_mode)
    {
        return _impl->read_(channel_idx,image,read_mode);
    }

    bm_image MultiDecoder::read_(int channel_idx)
    {
        return _impl->read_(channel_idx);
    }

    int MultiDecoder::reconnect(int channel_idx)
    {
        return _impl->reconnect(channel_idx);
    }
    
    std::vector<int> MultiDecoder::get_frame_shape(int channel_idx)
    {
        return _impl->get_frame_shape(channel_idx);
    }

    void MultiDecoder::set_local_flag(bool flag)
    {
        return _impl->set_local_flag(flag);
    }

    size_t MultiDecoder::get_drop_num(int channel_idx)
    {
        return _impl->get_drop_num(channel_idx);
    }

    void MultiDecoder::reset_drop_num(int channel_idx)
    {
        _impl->reset_drop_num(channel_idx);
        return ;
    }

    DecoderStatus MultiDecoder::get_channel_status(int channel_idx)
    {
        return _impl->get_channel_status(channel_idx);
    }

    class Mutex_Flag{
    public:
        Mutex_Flag(){
            flag_ = true;
        }
        ~Mutex_Flag(){}

        bool get_flag(){
            std::lock_guard<std::mutex> lock_temp(mutex_flag);
            return flag_;
        }

        void set_flag(bool flag){ 
            std::lock_guard<std::mutex> lock_temp(mutex_flag);
            flag_ = flag;
        }

    private: 
        std::mutex mutex_flag;
        bool flag_;
    };
    
    class FixedLengthDqueue{
        public:
            FixedLengthDqueue(int max_length){
                max_length_=max_length;
                Image_queue = NULL;
            }
            ~FixedLengthDqueue(){
                if(Image_queue){
                    delete Image_queue;
                    Image_queue = NULL;
                }
            }

            void PushData(BMImage &img){
                if(!Image_queue){
                    Image_queue = new BMImageQueue(max_length_);
                }
                if(Image_queue->IsFull()){
                    Image_queue->Pop();
                }
                Image_queue->Push(img);
            }

            void PushData(BMImageArray<4> *img_array){
                if(Array4_queue.size()>=max_length_){
                    BMImageArray<4> *temp_array = Array4_queue.front();
                    delete temp_array;
                    temp_array = NULL;
                    Array4_queue.pop_front();
                }
                Array4_queue.push_back(img_array);
            }

            void PushData(BMImageArray<6> *img_array){
                if(Array6_queue.size()>=max_length_){
                    BMImageArray<6> *temp_array = Array6_queue.front();
                    delete temp_array;
                    temp_array = NULL;
                    Array6_queue.pop_front();
                }
                Array6_queue.push_back(img_array);
            }

            void PushData(BMImageArray<8> *img_array){
                if(Array8_queue.size()>=max_length_){
                    BMImageArray<8> *temp_array = Array8_queue.front();
                    delete temp_array;
                    temp_array = NULL;
                    Array8_queue.pop_front();
                }
                Array8_queue.push_back(img_array);
            }

            void PushData(BMImageArray<16> *img_array){
                if(Array16_queue.size()>=max_length_){
                    BMImageArray<16> *temp_array = Array16_queue.front();
                    delete temp_array;
                    temp_array = NULL;
                    Array16_queue.pop_front();
                }
                Array16_queue.push_back(img_array);
            }

            void PushData(BMImageArray<32> *img_array){
                if(Array32_queue.size()>=max_length_){
                    BMImageArray<32> *temp_array = Array32_queue.front();
                    delete temp_array;
                    temp_array = NULL;
                    Array32_queue.pop_front();
                }
                Array32_queue.push_back(img_array);
            }

            void PushData(BMImageArray<64> *img_array){
                if(Array64_queue.size()>=max_length_){
                    BMImageArray<64> *temp_array = Array64_queue.front();
                    delete temp_array;
                    temp_array = NULL;
                    Array64_queue.pop_front();
                }
                Array64_queue.push_back(img_array);
            }
        private:
            int max_length_;
            BMImageQueue*                    Image_queue; //缓存数据，防止Tensor被清空
            std::deque<BMImageArray<4>*>     Array4_queue; //缓存数据，防止Tensor被清空
            std::deque<BMImageArray<6>*>     Array6_queue; //缓存数据，防止Tensor被清空
            std::deque<BMImageArray<8>*>     Array8_queue; //缓存数据，防止Tensor被清空
            std::deque<BMImageArray<16>*>    Array16_queue; //缓存数据，防止Tensor被清空
            std::deque<BMImageArray<32>*>    Array32_queue; //缓存数据，防止Tensor被清空
            std::deque<BMImageArray<64>*>    Array64_queue; //缓存数据，防止Tensor被清空
    };


    class ImagePreProcess::ImagePreProcess_CC{
    public:
        ImagePreProcess_CC(
            int batch_size,                     //batch size
            sail_resize_type resize_mode,       //resize的方法
            int tpu_id,                         //设备编号
            int queue_size_in,                  //输入图像队列的最大长度
            int queue_size_out,                 //输出Tensor的最大的长度
            bool use_mat);                      //输出是否使用cvMat作为输出

        ~ImagePreProcess_CC();
       
    
        void SetResizeImageAtrr(				//设置输出图像的属性
            int output_width,				    //输出图像的宽度
            int output_height,				    //输出图像的高度
            bool bgr2rgb,					    //BGR转换RGB的标志位。
            bm_image_data_format_ext  dtype);	//输出的图像格式目前只支持DATA_TYPE_EXT_FLOAT32,DATA_TYPE_EXT_1N_BYTE, DATA_TYPE_EXT_1N_BYTE_SIGNED

        /* 返回Padding的起点坐标（x,y）图像缩放后的宽w,图像缩放后的高 */
       void SetPaddingAtrr(		        //设置输出图像的属性
            int padding_b,		                //输出图像的宽度
            int padding_g,		                //输出图像的高度
            int padding_r,		                //BGR转换RGB的标志位。
            int align);		                    //padding的位置，默认0，从左上角（0,0）点开始；1居中
        
        vector<int> CalcPaddingAtrr(
            int input_width, 
            int input_height, 
            int output_width,
            int output_height,
            int align);
         
        int SetConvertAtrr(
            const std::tuple<
                std::pair<float, float>,
                std::pair<float, float>,
                std::pair<float, float>> &alpha_beta);
            

        int PushImage(int channel_idx, int image_idx, BMImage& image);
        
        void exhausted();

        bool get_exhausted_flag();

        std::tuple<sail::Tensor, 
            std::vector<BMImage>,
            std::vector<int>,
            std::vector<int>,
            std::vector<std::vector<int>> > GetBatchData();       //获取一个Batch的数据结果Tensor

        std::tuple<sail::Tensor, 
            std::vector<cv::Mat>,
            std::vector<int>,
            std::vector<int>,
            std::vector<std::vector<int>> > GetBatchData_CV();    //获取一个Batch的数据结果Tensor

        void set_print_flag(bool print_flag);       //设置是否打印标志位
    private:
        BMImageQueue* InputImg_queue;         //输入图像队列
        std::deque<int>     ChannelIdx_queue;       //输入图像通道编号队列
        std::deque<int>     ImageIdx_queue;         //输入图像编号队列
        std::mutex mutex_input_data;                //输入互斥

        std::deque<BMImage> OutputImg_queue;        //输出原始图像
        std::deque<int>     OutputChannelIdx_queue; //输出图像通道编号队列
        std::deque<int>     OutputImgIdx_queue;     //输出图像编号队列
        std::deque<std::vector<int>>     PaddingAtrr_queue;    //输出图像Padding属性队列(start_x, start_y, width, height)
        std::mutex mutex_output_data_0;             //输出互斥
        std::mutex mutex_output_data_1;             //输出互斥

        std::deque<cv::Mat> OutputImg_cv_queue;     //输出原始图像
        bool use_mat_flag;                          //使用OpenCV作为原始图像输出,默认不使用
        std::mutex mutex_output_cv_data;            //输出互斥,只对OutputImg_cv_queue有效
        std::condition_variable cv_flag_cond;
        std::mutex cv_flag;
        Mutex_Flag tomat_exit_flag;                 //BMImage转换cvMat线程退出的标志位

        std::condition_variable input_flag_cond;
        std::mutex input_flag;

        BMImageQueue* Format_queue;               //转换格式之后的图像队列
        Mutex_Flag convert_format_exit_flag;        //格式转换线程退出的标志位
        std::condition_variable format_flag_cond;
        std::mutex format_flag;

        BMImageQueue* Resize_queue;               //Resize之后的图像队列
        Mutex_Flag resize_exit_flag;                //Resize线程退出的标志位
        std::condition_variable resize_flag_cond;
        std::mutex resize_flag;

        BMImageQueue* Convert_queue;              //经过线性变换之后的数据队列
        Mutex_Flag convert_exit_flag;               //线性变换线程退出的标志位
        std::condition_variable convert_flag_cond;
        std::mutex convert_flag;

        std::deque<Tensor>  OutputTensor_queue;     //输出Tensor队列
        std::mutex mutex_to_tensor;                 //读取线性变换之后的图像队列的互斥
        Mutex_Flag to_tensor_exit_flag;             //转换Tensor线程退出的标志位
        std::condition_variable outdata_flag_cond;
        std::mutex outdata_flag;

        FixedLengthDqueue *OutArray_queue;       //缓存数据，防止Tensor被清空

        Mutex_Flag stop_thread_flag;        //线程退出的标志位

        bool exit_thread_flag;              //线程已经退出的标志位
        std::mutex mutex_exit_;             //线程退出互斥
        std::condition_variable exit_cond;  //线程已经退出的信号量

        int queue_size_in_;                  
        int queue_size_out_;                 
        int tpu_id_;                         
        sail_resize_type resize_mode_;
        int batch_size_;
        bool use_padding_;      //是否使用padding
        bool had_padding_atrr_; //是否已经设置过padding属性
        bool had_resize_attr_;  //是否已经设置resize的属性

        int output_width_;				        //输出图像的宽度
        int output_height_;				        //输出图像的高度
        bool bgr2rgb_;					        //BGR转换RGB的标志位。
        bool use_vpp_flag_;                     //是否使用vpp的标志位。

        bm_image_data_format_ext img_dtype_;    //image dtype
        bm_image_format_ext out_format_;	    //线性变换之后的图像格式(bgr/rgb)

        bm_data_type_t  dtype_;                 //Tensor dtype
    
        int padding_r_;
        int padding_g_;
        int padding_b_;

        bool had_create_thread_flag;            //是否已经创建过线程的标志位

        bool set_convert_attr_flag;             //是否已经设置过线性变换的参数
        std::tuple<std::pair<float, float>, std::pair<float, float>,std::pair<float, float>> linear_trans_param_;

        bool print_flag_;

        int align_;     ///padding的位置,默认0，从左上角（0,0）点开始；1居中

        bool exhausted_flag_;

        friend class ImagePreProcess;

    private:
        int get_input_data(BMImage &image, int &channel, int &index);

        int push_result_image(BMImage& ostimage, int channel, int index);

        int push_padding_atrr(std::vector<int> padding_atrr);

        int push_tensor(Tensor& tensor_temp);

        int get_outque_size();

        int get_outimgque_data(BMImage& ostimage);

        void push_cvimgque_data(cv::Mat& cvimage);
        
        Tensor convert_to_Tensor_1N(sail::Handle& handle, Bmcv bmcv);
        
        Tensor convert_to_Tensor_4N(sail::Handle& handle, Bmcv bmcv);

        Tensor convert_to_Tensor_6N(sail::Handle& handle, Bmcv bmcv);

        Tensor convert_to_Tensor_8N(sail::Handle& handle,Bmcv bmcv);

        Tensor convert_to_Tensor_16N(sail::Handle& handle,Bmcv bmcv);

        Tensor convert_to_Tensor_32N(sail::Handle& handle,Bmcv bmcv);

        Tensor convert_to_Tensor_64N(sail::Handle& handle,Bmcv bmcv);

        int get_output_data(sail::Tensor& tensor, 
                            std::vector<sail::BMImage> &images,
                            std::vector<int> &channels, 
                            std::vector<int> &indexs,
                            std::vector<std::vector<int>> &padding_atrr);
        
        int get_output_cvdata(sail::Tensor& tensor, 
                        std::vector<cv::Mat> &images,
                        std::vector<int> &channels, 
                        std::vector<int> &indexs,
                        std::vector<std::vector<int>> &padding_atrr);

        void set_thread_exit();

        void wait_thread_exit();
        
        void convert_format_thread();

        void resize_thread();

        void convert_to_thread();

        void to_tensor_thread();

        void to_cvmat_thread();

    };

    ImagePreProcess::ImagePreProcess_CC::ImagePreProcess_CC(
            int batch_size,                     //batch size
            sail_resize_type resize_mode,       //resize的方法
            int tpu_id,                         //设备编号
            int queue_size_in,                  //输入图像队列的最大长度
            int queue_size_out,                 //输出Tensor的最大的长度
            bool use_mat)
            : queue_size_in_(queue_size_in),queue_size_out_(queue_size_out),tpu_id_(tpu_id),
                align_(0), padding_r_(114),padding_g_(114),padding_b_(114), resize_mode_(resize_mode),
                batch_size_(batch_size),use_mat_flag(use_mat),exhausted_flag_(false),
                Format_queue(NULL),Resize_queue(NULL),Convert_queue(NULL),InputImg_queue(NULL)
    {
        // 输入输出队列长度必须大于batch size
        if (queue_size_in < batch_size || queue_size_out < batch_size){
            queue_size_in_ = queue_size_in > batch_size ? queue_size_in : batch_size;
            queue_size_out_ = queue_size_out > batch_size ? queue_size_out : batch_size;
            spdlog::warn("Queue_size must not be less than the batch_size of bmodel. Now queue_size is set to batch_size");
        }

        Format_queue = new BMImageQueue(queue_size_out_);
        Resize_queue = new BMImageQueue(queue_size_out_);
        Convert_queue = new BMImageQueue(queue_size_out_);
        InputImg_queue = new BMImageQueue(queue_size_in_);
        switch(resize_mode_){
        case BM_RESIZE_VPP_NEAREST: 
        case BM_RESIZE_TPU_NEAREST: 
        case BM_RESIZE_TPU_LINEAR: 
        case BM_RESIZE_TPU_BICUBIC:
            use_padding_ = false;
            break;
        default:
            use_padding_ = true;
            break;
        }
        switch(resize_mode_){
        case BM_RESIZE_VPP_NEAREST:
        case BM_PADDING_VPP_NEAREST:
            use_vpp_flag_ = true;
        default:
            use_vpp_flag_ = false;
        }
        stop_thread_flag.set_flag(false);
        had_padding_atrr_ = false;
        exit_thread_flag = false;
        set_convert_attr_flag = false;
        had_resize_attr_ = false;

        if(batch_size_ != 1 && batch_size_ != 4 && batch_size_ != 8 && batch_size_ != 16 && batch_size_ != 32 && batch_size_ != 64){
            SPDLOG_ERROR("Batch size must be one of [1,4,8,16,32,64]!");
            throw SailEngineError("not supported");
        }
        had_create_thread_flag = false;
        print_flag_ = false;
        OutArray_queue = new FixedLengthDqueue(2*queue_size_out_+batch_size_);
    }

    ImagePreProcess::ImagePreProcess_CC::~ImagePreProcess_CC()
    {
        SPDLOG_INFO("Start Set Thread Exit Flag!");
        set_thread_exit();
        SPDLOG_INFO("End Set Thread Exit Flag, Waiting Thread Exit....");
        wait_thread_exit();
        SPDLOG_INFO("All Thread Finshed!");
        if(Format_queue){
            delete Format_queue;
            Format_queue = NULL;
        }
        if(Resize_queue){
            delete Resize_queue;
            Resize_queue = NULL;
        }
        if(Convert_queue){
            delete Convert_queue;
            Convert_queue = NULL;
        }
        delete OutArray_queue;
        OutArray_queue = NULL;
        if(InputImg_queue){
            delete InputImg_queue;
            InputImg_queue = NULL;
        }
        SPDLOG_INFO("All Exited!");
    }

    void ImagePreProcess::ImagePreProcess_CC::set_print_flag(bool print_flag)
    {
        print_flag_ = print_flag;
    }

    void ImagePreProcess::ImagePreProcess_CC::exhausted()
    {
        exhausted_flag_ = true;
    }

    bool ImagePreProcess::ImagePreProcess_CC::get_exhausted_flag()
    {
        bool ret = false;
        if(exhausted_flag_)
        {
            if(Format_queue->IsEmpty()&&Resize_queue->IsEmpty()&&Convert_queue->IsEmpty())
                ret = true;
        }
        return ret;
    }

    void ImagePreProcess::ImagePreProcess_CC::SetResizeImageAtrr(				
            int output_width,				    
            int output_height,				    
            bool bgr2rgb,					    
            bm_image_data_format_ext  dtype)
    {
        bgr2rgb_ = bgr2rgb;
        if(bgr2rgb_){
            out_format_ = FORMAT_RGB_PLANAR;
        }else{
            out_format_ = FORMAT_BGR_PLANAR;
        }

        output_width_ = output_width;
        output_height_ = output_height; 
        img_dtype_ = dtype;

        if(dtype == DATA_TYPE_EXT_FLOAT32){
            dtype_ = BM_FLOAT32;
        }else if (dtype == DATA_TYPE_EXT_1N_BYTE){
            dtype_ = BM_UINT8;
        }else if (dtype == DATA_TYPE_EXT_1N_BYTE_SIGNED){
            dtype_ = BM_INT8;
        }else{
            SPDLOG_ERROR("DType not supported!");
            throw SailEngineError("not supported");
        }
        had_resize_attr_ = true;
    }
    
    vector<int> ImagePreProcess::ImagePreProcess_CC::CalcPaddingAtrr(
        int input_width, 
        int input_height, 
        int output_width,
        int output_height,
        int align)		            //padding的位置，默认0，从左上角（0,0）点开始；1居中
    {
        float scale_w = float(output_width)/input_width;
        float scale_h = float(output_height)/input_height;

        int pad_w = output_width;
        int pad_h = output_height;
        
        float scale_min = scale_h;
        if(scale_w < scale_min){
            scale_min = scale_w;
            pad_h = input_height*scale_min;
        }else{
            pad_w = input_width*scale_min;
        }
        int start_x = 0;
        int start_y = 0;
        if (align == 1){
            start_x = (output_width - pad_w)/2;
            start_y = (output_height - pad_h)/2;
        }
        vector<int> result = {start_x, start_y, pad_w, pad_h};
        return result;      
    }
    void ImagePreProcess::ImagePreProcess_CC::SetPaddingAtrr(		//设置输出图像的属性
        int padding_b,		        //输出图像的宽度
        int padding_g,		        //输出图像的高度
        int padding_r,		        //BGR转换RGB的标志位。
        int align)		            //padding的位置，默认0，从左上角（0,0）点开始；1居中
    {
        padding_r_ = padding_r;
        padding_g_ = padding_g;
        padding_b_ = padding_b;
        align_ = align;
        had_padding_atrr_ = true;
        return;
    }

    int ImagePreProcess::ImagePreProcess_CC::SetConvertAtrr(
        const std::tuple<
            std::pair<float, float>,
            std::pair<float, float>,
            std::pair<float, float>> &alpha_beta
        ) {
        if(set_convert_attr_flag){
            return 1;
        }
        set_convert_attr_flag = true;
        linear_trans_param_ = alpha_beta;
        return 0;
    }

    int ImagePreProcess::ImagePreProcess_CC::PushImage(int channel_idx, int image_idx, BMImage& image)
    {
        std::lock_guard<std::mutex> lock_temp(mutex_input_data);
        if(ChannelIdx_queue.size() >= queue_size_in_){
            return 2;
        }

        InputImg_queue->Push(image);
        ChannelIdx_queue.push_back(channel_idx);
        ImageIdx_queue.push_back(image_idx);
        if(!had_create_thread_flag){
            had_create_thread_flag = true;
            if(use_mat_flag){
                std::thread thread_convert_cvmat = std::thread(&ImagePreProcess_CC::to_cvmat_thread,this);
                thread_convert_cvmat.detach();
            }
            std::thread thread_convert_format = std::thread(&ImagePreProcess_CC::convert_format_thread,this);
            thread_convert_format.detach();
            std::thread thread_resize = std::thread(&ImagePreProcess_CC::resize_thread,this);
            thread_resize.detach();
            std::thread thread_convert_to = std::thread(&ImagePreProcess_CC::convert_to_thread,this);
            thread_convert_to.detach();
            std::thread thread_to_tensor = std::thread(&ImagePreProcess_CC::to_tensor_thread,this);
            thread_to_tensor.detach();
        }

        std::unique_lock<std::mutex> lck(input_flag);
        input_flag_cond.notify_all(); 
        return 0;
    }

    std::tuple<sail::Tensor, 
        std::vector<BMImage>,
        std::vector<int>,
        std::vector<int>,
        std::vector<std::vector<int>> > ImagePreProcess::ImagePreProcess_CC::GetBatchData()     
    {
        if(use_mat_flag){
            SPDLOG_ERROR("USE cvMat Flag is TRUE, Can not use GetBatchData Function!");  
            throw SailEngineError("not supported");
        }
        sail::Tensor out_tensor;
        std::vector<sail::BMImage> images;
        std::vector<int> channels; 
        std::vector<int> indexs;
        std::vector<std::vector<int>> padding_atrr;

        int get_data_count = 0;
        while(true){
            if(get_output_data(out_tensor,images,channels,indexs,padding_atrr) == 0){
                break;
            }
            if(stop_thread_flag.get_flag()){
                break;
            }

            double time_start = get_current_time_us();
            std::unique_lock<std::mutex> lck(outdata_flag);
            outdata_flag_cond.wait_for(lck,std::chrono::seconds(1));
            if(print_flag_){
                double time_use = (get_current_time_us() - time_start)/1000;
                SPDLOG_INFO("to_tensor_thread sleep {} ms.",time_use);  
            }
        }
        return std::move(std::make_tuple(std::move(out_tensor),std::move(images),channels,indexs,padding_atrr));
    }

    std::tuple<sail::Tensor, 
        std::vector<cv::Mat>,
        std::vector<int>,
        std::vector<int>,
        std::vector<std::vector<int>> > ImagePreProcess::ImagePreProcess_CC::GetBatchData_CV()
    {
        if(!use_mat_flag){
            SPDLOG_ERROR("USE cvMat Flag is False, Can not use GetBatchData_CV Function!");  
            throw SailEngineError("not supported");
        }
        sail::Tensor out_tensor;
        std::vector<cv::Mat> images;
        std::vector<int> channels; 
        std::vector<int> indexs;
        std::vector<std::vector<int>> padding_atrr;

        int get_data_count = 0;
        while(true){
            if(get_output_cvdata(out_tensor,images,channels,indexs,padding_atrr) == 0){
                break;
            }
            if(stop_thread_flag.get_flag()){
                break;
            }

            double time_start = get_current_time_us();
            std::unique_lock<std::mutex> lck(outdata_flag);
            outdata_flag_cond.wait_for(lck,std::chrono::seconds(1));
            if(print_flag_){
                double time_use = (get_current_time_us() - time_start)/1000;
                SPDLOG_INFO("to_tensor_thread sleep {} ms.",time_use);  
            }
        }
        return std::move(std::make_tuple(std::move(out_tensor),std::move(images),channels,indexs,padding_atrr));
    }

    int ImagePreProcess::ImagePreProcess_CC::get_input_data(BMImage &image, int &channel, int &index)
    {
        std::lock_guard<std::mutex> lock_temp(mutex_input_data);
        if(InputImg_queue->IsEmpty()){
            return 1;
        }
        channel = ChannelIdx_queue.front();
        index = ImageIdx_queue.front();
        InputImg_queue->Pop(image);
        ChannelIdx_queue.pop_front();
        ImageIdx_queue.pop_front();
        return 0;
    }

    int ImagePreProcess::ImagePreProcess_CC::push_result_image(BMImage& ostimage, int channel, int index){
        
        {
            std::lock_guard<std::mutex> lock_temp(mutex_output_data_0);
            OutputImg_queue.push_back(std::move(ostimage));
        }
        {        
            std::lock_guard<std::mutex> lock_temp(mutex_output_data_1);
            OutputChannelIdx_queue.push_back(channel);
            OutputImgIdx_queue.push_back(index);
        }
        return 0;
    }

    int ImagePreProcess::ImagePreProcess_CC::push_padding_atrr(std::vector<int> padding_atrr){
        std::lock_guard<std::mutex> lock_temp(mutex_output_data_1);
        PaddingAtrr_queue.push_back(padding_atrr);
        return 0;
    }

    int ImagePreProcess::ImagePreProcess_CC::push_tensor(Tensor& tensor_temp){
        std::lock_guard<std::mutex> lock_temp(mutex_to_tensor);
        OutputTensor_queue.push_back(std::move(tensor_temp));
        return 0;
    }

    int ImagePreProcess::ImagePreProcess_CC::get_outque_size(){
        std::lock_guard<std::mutex> lock_temp(mutex_to_tensor);
        return OutputTensor_queue.size();
    }

    int ImagePreProcess::ImagePreProcess_CC::get_outimgque_data(sail::BMImage &image){
        std::lock_guard<std::mutex> lock_temp(mutex_output_data_0);
        if(OutputImg_queue.size() <= 0){
            return 1;
        }else{
            image = std::move(OutputImg_queue.front());
            OutputImg_queue.pop_front();
            return 0;
        }
    }

    void ImagePreProcess::ImagePreProcess_CC::push_cvimgque_data(cv::Mat& cvimage)
    {
        std::lock_guard<std::mutex> lock_temp(mutex_output_cv_data);
        OutputImg_cv_queue.push_back(std::move(cvimage));
    }

    Tensor ImagePreProcess::ImagePreProcess_CC::convert_to_Tensor_1N(sail::Handle& handle, Bmcv bmcv){
        sail::BMImage img_temp;
        Convert_queue->Pop(img_temp);
        sail::Tensor out_tensor = bmcv.bm_image_to_tensor(img_temp);
        OutArray_queue->PushData(img_temp);
        return std::move(out_tensor);
    }

    Tensor ImagePreProcess::ImagePreProcess_CC::convert_to_Tensor_4N(sail::Handle& handle, Bmcv bmcv){
        sail::BMImageArray<4> *img_array = new sail::BMImageArray<4>(handle,output_height_, output_width_, out_format_, img_dtype_);
        for(int i=0;i<4;++i){
            sail::BMImage img_temp;
            Convert_queue->Pop(img_temp);
            img_array->copy_from(i,img_temp);
        }
        sail::Tensor out_tensor = bmcv.bm_image_to_tensor(*img_array);
        OutArray_queue->PushData(img_array);
        return std::move(out_tensor);
    }

    Tensor ImagePreProcess::ImagePreProcess_CC::convert_to_Tensor_6N(sail::Handle& handle, Bmcv bmcv){
        sail::BMImageArray<6> *img_array = new sail::BMImageArray<6>(handle,output_height_, output_width_, out_format_, img_dtype_);
        for(int i=0;i<6;++i){
            sail::BMImage img_temp;
            Convert_queue->Pop(img_temp);
            img_array->copy_from(i,img_temp);
        }
        sail::Tensor out_tensor = bmcv.bm_image_to_tensor(*img_array);
        OutArray_queue->PushData(img_array);
        return std::move(out_tensor);
    }

    Tensor ImagePreProcess::ImagePreProcess_CC::convert_to_Tensor_8N(sail::Handle& handle,Bmcv bmcv){
        sail::BMImageArray<8> *img_array = new sail::BMImageArray<8>(handle,output_height_, output_width_, out_format_, img_dtype_);
        for(int i=0;i<8;++i){
            sail::BMImage img_temp;
            Convert_queue->Pop(img_temp);
            img_array->copy_from(i,img_temp);
        }
        sail::Tensor out_tensor = bmcv.bm_image_to_tensor(*img_array);
        OutArray_queue->PushData(img_array);
        return std::move(out_tensor);
    }

    Tensor ImagePreProcess::ImagePreProcess_CC::convert_to_Tensor_16N(sail::Handle& handle,Bmcv bmcv){
        sail::BMImageArray<16> *img_array= new sail::BMImageArray<16>(handle,output_height_, output_width_, out_format_, img_dtype_);
        for(int i=0;i<16;++i){
            sail::BMImage img_temp;
            Convert_queue->Pop(img_temp);
            img_array->copy_from(i,img_temp);
        }
        sail::Tensor out_tensor = bmcv.bm_image_to_tensor(*img_array);
        OutArray_queue->PushData(img_array);
        return std::move(out_tensor);
    }

    Tensor ImagePreProcess::ImagePreProcess_CC::convert_to_Tensor_32N(sail::Handle& handle,Bmcv bmcv){
        sail::BMImageArray<32> *img_array= new sail::BMImageArray<32>(handle,output_height_, output_width_, out_format_, img_dtype_);
        for(int i=0;i<32;++i){
            sail::BMImage img_temp;
            Convert_queue->Pop(img_temp);
            img_array->copy_from(i,img_temp);
        }
        sail::Tensor out_tensor = bmcv.bm_image_to_tensor(*img_array);
        OutArray_queue->PushData(img_array);
        return std::move(out_tensor);
    }

    Tensor ImagePreProcess::ImagePreProcess_CC::convert_to_Tensor_64N(sail::Handle& handle,Bmcv bmcv){
        sail::BMImageArray<64> *img_array= new sail::BMImageArray<64>(handle,output_height_, output_width_, out_format_, img_dtype_);
        for(int i=0;i<64;++i){
            sail::BMImage img_temp;
            Convert_queue->Pop(img_temp);
            img_array->copy_from(i,img_temp);
        }
        sail::Tensor out_tensor = bmcv.bm_image_to_tensor(*img_array);
        OutArray_queue->PushData(img_array);
        return std::move(out_tensor);
    }


    int ImagePreProcess::ImagePreProcess_CC::get_output_data(sail::Tensor& tensor, 
                        std::vector<sail::BMImage> &images,
                        std::vector<int> &channels, 
                        std::vector<int> &indexs,
                        std::vector<std::vector<int>> &padding_atrr)
    {
        {            
            std::lock_guard<std::mutex> lock_temp(mutex_to_tensor);
            if(OutputTensor_queue.size() <= 0){
                return 1;
            }
            tensor = std::move(OutputTensor_queue.front());
            OutputTensor_queue.pop_front();
        }
        {
            std::lock_guard<std::mutex> lock_temp(mutex_output_data_0);
            for(int i=0;i<batch_size_;++i){
                images.push_back(std::move(OutputImg_queue.front()));
                OutputImg_queue.pop_front();
            }
        }
        {
            std::lock_guard<std::mutex> lock_temp(mutex_output_data_1);
            for(int i=0;i<batch_size_;++i){
                channels.push_back(OutputChannelIdx_queue.front());
                indexs.push_back(OutputImgIdx_queue.front());
                padding_atrr.push_back(PaddingAtrr_queue.front());
                OutputChannelIdx_queue.pop_front();
                OutputImgIdx_queue.pop_front();
                PaddingAtrr_queue.pop_front();
            }
        }
        return 0;
    }

    int ImagePreProcess::ImagePreProcess_CC::get_output_cvdata(sail::Tensor& tensor, 
                        std::vector<cv::Mat> &images,
                        std::vector<int> &channels, 
                        std::vector<int> &indexs,
                        std::vector<std::vector<int>> &padding_atrr)
    {
        {
            std::lock_guard<std::mutex> lock_temp(mutex_output_cv_data);
            if(OutputImg_cv_queue.size() < batch_size_){
                return 2;
            }
        }

        {            
            std::lock_guard<std::mutex> lock_temp(mutex_to_tensor);
            if(OutputTensor_queue.size() <= 0){
                return 1;
            }
            tensor = std::move(OutputTensor_queue.front());
            OutputTensor_queue.pop_front();
        }
        {
            std::lock_guard<std::mutex> lock_temp(mutex_output_data_1);
            for(int i=0;i<batch_size_;++i){
                channels.push_back(OutputChannelIdx_queue.front());
                indexs.push_back(OutputImgIdx_queue.front());
                padding_atrr.push_back(PaddingAtrr_queue.front());
                OutputChannelIdx_queue.pop_front();
                OutputImgIdx_queue.pop_front();
                PaddingAtrr_queue.pop_front();
            }
        }
        {
            std::lock_guard<std::mutex> lock_temp(mutex_output_cv_data);
            for(int i=0;i<batch_size_;++i){
                images.push_back(std::move(OutputImg_cv_queue.front()));
                OutputImg_cv_queue.pop_front();
            }
        }
        return 0;
    }

    void ImagePreProcess::ImagePreProcess_CC::set_thread_exit()
    {
        stop_thread_flag.set_flag(true);
    }   

    void ImagePreProcess::ImagePreProcess_CC::wait_thread_exit()
    {
        while(true){
            if(!tomat_exit_flag.get_flag()){
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            if(!convert_format_exit_flag.get_flag()){
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            if(!resize_exit_flag.get_flag()){
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            if(!convert_exit_flag.get_flag()){
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            if(!to_tensor_exit_flag.get_flag()){
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            break;
        }

    }

    void ImagePreProcess::ImagePreProcess_CC::to_cvmat_thread()
    {
        static int save_idx = 0;
        if(!use_mat_flag){
            SPDLOG_INFO("Not cvMat for Output, Do not create to cvMat thread!");
            return ;
        }
//        SPDLOG_INFO("Create Convert to cv::Mat Thread, pid:{}, tid:{} .",getpid(),gettid());
        sail::Handle handle(tpu_id_);
        sail::Bmcv bmcv(handle);
        tomat_exit_flag.set_flag(false);
        while(true){
            if(stop_thread_flag.get_flag()){
                break;
            }
            sail::BMImage ost_image;
            int ret = get_outimgque_data(ost_image);
            if(ret != 0){
                std::unique_lock<std::mutex> lck(resize_flag);
                resize_flag_cond.wait_for(lck,std::chrono::milliseconds(50));
                continue;
            }
            cv::Mat image_temp = bmcv.bm_image_to_mat(ost_image);
            push_cvimgque_data(image_temp);
            std::unique_lock<std::mutex> lck(outdata_flag);
            outdata_flag_cond.notify_all(); 
        }
           
        convert_format_exit_flag.set_flag(true);
        SPDLOG_INFO("Convert to cv Mat Thread End!");
    }

    void ImagePreProcess::ImagePreProcess_CC::convert_format_thread(){
        if (!had_resize_attr_){
            SPDLOG_ERROR("Not SetResizeImageAtrr, Create convert format thread failed!");
            throw SailRuntimeError("invalid status");
        }
        if(!set_convert_attr_flag){
            SPDLOG_ERROR("Not Set LinearTransParam, Please Call SetConvertAtrr First!");
            throw SailRuntimeError("invalid status");
        }
//         SPDLOG_INFO("Create Convert Format Thread, pid:{}, tid:{} .",getpid(),gettid());
        sail::Handle handle(tpu_id_);
        sail::Bmcv bmcv(handle);
        convert_format_exit_flag.set_flag(false);

        int get_data_count = 0;
        while(true){
            if(stop_thread_flag.get_flag()){
                break;
            }

            if(queue_size_out_ >= 0 && Format_queue->IsFull()){ //如果已经处理的数据超过缓存,就先不处理
                double time_start = get_current_time_us();
                std::unique_lock<std::mutex> lck(resize_flag);
                resize_flag_cond.wait_for(lck,std::chrono::milliseconds(20));
                if(print_flag_){
                    double time_use = (get_current_time_us() - time_start)/1000;
                    SPDLOG_INFO("Format_queue full, convert_format_thread sleep {} ms.",time_use);
                }
                continue;
            }

            if(queue_size_out_ >= 0 && get_outque_size() >= queue_size_out_){ //如果已经处理的数据超过缓存,就先不处理
                double time_start = get_current_time_us();
                std::unique_lock<std::mutex> lck(resize_flag);
                resize_flag_cond.wait_for(lck,std::chrono::milliseconds(20));
                if(print_flag_){
                    double time_use = (get_current_time_us() - time_start)/1000;
                    SPDLOG_INFO("Output full:{},do not get data,convert_format_thread sleep {} ms.", get_outque_size(),time_use);
                }
                continue;
            }
            
            sail::BMImage image_temp;
            int channel_idx;
            int index;
            int ret = 0;

            ret = get_input_data(image_temp, channel_idx, index);
            
            if(ret != 0){
                double time_start = get_current_time_us();
                std::unique_lock<std::mutex> lck(input_flag);
                input_flag_cond.wait_for(lck,std::chrono::milliseconds(20));
                if(print_flag_){
                    double time_use = (get_current_time_us() - time_start)/1000;
                    SPDLOG_INFO("Do not get data, convert_format_thread sleep {} ms.",time_use);
                }
                continue;
            }
            double time_start_convert_format = get_current_time_us();
            get_data_count++;
            sail::BMImage format_image = sail::BMImage(handle, image_temp.height(), image_temp.width(), out_format_, DATA_TYPE_EXT_1N_BYTE);
            if(use_vpp_flag_){
                int ret = bmcv.vpp_convert_format(image_temp,format_image);
                // judge ret
                if (ret != 0){
                    SPDLOG_ERROR("vpp convert format error! ret:{}",ret);
                    throw SailBMImageError("bmcv api fail");
                }
            }else{
                int ret = bmcv.convert_format(image_temp,format_image);
                // judge ret
                if (ret != 0){
                    SPDLOG_ERROR("convert format error! ret:{}",ret);
                    throw SailBMImageError("bmcv api fail");
                }
            }
            {
                std::lock_guard<std::mutex> lock_temp(mutex_output_data_0);
                OutputImg_queue.push_back(std::move(image_temp));
            }
            {
                std::lock_guard<std::mutex> lock_temp(mutex_output_data_1);
                OutputChannelIdx_queue.push_back(channel_idx);
                OutputImgIdx_queue.push_back(index);
            }

            // char save_name_format[128] = {};
            // sprintf(save_name_format, "%08d.jpg",get_data_count-1);
            // bmcv.imwrite(save_name_format,format_image);
            Format_queue->Push(format_image);
            std::unique_lock<std::mutex> lck(format_flag);
            format_flag_cond.notify_all();   

            if(print_flag_){
                double time_use_convert_fotmat = (get_current_time_us() - time_start_convert_format)/1000;
                SPDLOG_INFO("convert_format time use: {} ms,Format_queue size:{} .",time_use_convert_fotmat,Format_queue->Size());
            }
        }
        convert_format_exit_flag.set_flag(true);
        SPDLOG_INFO("Convert Format Thread End!");
    }

    void ImagePreProcess::ImagePreProcess_CC::resize_thread(){
        if (!had_resize_attr_){
            SPDLOG_ERROR("Not SetResizeImageAtrr, Create resize thread failed!");
            throw SailRuntimeError("invalid status");
        }
        if(!set_convert_attr_flag){
            SPDLOG_ERROR("Not Set LinearTransParam, Please Call SetConvertAtrr First!");
            throw SailRuntimeError("invalid status");
        }
        sail::Handle handle(tpu_id_);
        sail::Bmcv bmcv(handle);
//         SPDLOG_INFO("Create Resize Thread, pid:{}, tid:{} .",getpid(),gettid());
        resize_exit_flag.set_flag(false);

        vector<int> pad_att_temp = {0,0,output_width_,output_height_};
        PaddingAtrr padding_atrr;
        if(had_padding_atrr_){
            padding_atrr.set_r(padding_r_);
            padding_atrr.set_g(padding_g_);
            padding_atrr.set_b(padding_b_);
        }
        
        while(true){
            if(stop_thread_flag.get_flag()){
                break;
            }
            sail::BMImage format_image;

            if(queue_size_out_ >= 0 && Resize_queue->IsFull()) { //如果已经处理的数据超过缓存,就先不处理
                double time_start = get_current_time_us();
                std::unique_lock<std::mutex> lck(convert_flag);
                convert_flag_cond.wait_for(lck,std::chrono::milliseconds(20));
                if(print_flag_){
                    double time_use = (get_current_time_us() - time_start)/1000;
                    SPDLOG_INFO("Resize_queue full, resize_thread sleep {} ms.",time_use);
                }
                continue;
            }

            if(Format_queue->Pop(format_image) != 0){
                double time_start = get_current_time_us();
                std::unique_lock<std::mutex> lck(format_flag);
                format_flag_cond.wait_for(lck,std::chrono::milliseconds(20));
                if(print_flag_){
                    double time_use = (get_current_time_us() - time_start)/1000;
                    SPDLOG_INFO("resize_thread sleep {} ms.",time_use);
                }
                continue;
            }

            double time_start_resize = get_current_time_us();
            spdlog::debug("IPP out_format_ {}",out_format_);
            sail::BMImage resize_image = sail::BMImage(handle, output_height_, output_width_, out_format_, DATA_TYPE_EXT_1N_BYTE);
            bool push_flag = false;
            spdlog::debug("IPP resize_mode_ {}",resize_mode_);
            switch(resize_mode_){
                case BM_RESIZE_VPP_NEAREST: 
                {                
                    int ret = bmcv.vpp_resize(format_image,resize_image,output_width_,output_height_);
                    // judge ret
                    if (ret != 0){
                        SPDLOG_ERROR("vpp resize error! ret:{}",ret);
                        throw SailBMImageError("bmcv api fail");
                    }
                    push_flag = true;
                    break;
                }
                case BM_RESIZE_TPU_NEAREST: 
                {                
                    int ret = bmcv.resize(format_image,resize_image,output_width_,output_height_,BMCV_INTER_NEAREST);
                    // judge ret
                    if (ret != 0){
                        SPDLOG_ERROR("resize error! ret:{}",ret);
                        throw SailBMImageError("bmcv api fail");
                    }
                    push_flag = true;
                    break;
                }
                case BM_RESIZE_TPU_LINEAR: 
                {                
                    int ret = bmcv.resize(format_image,resize_image,output_width_,output_height_,BMCV_INTER_LINEAR);
                    // judge ret
                    if (ret != 0){
                        SPDLOG_ERROR("resize error! ret:{}",ret);
                        throw SailBMImageError("bmcv api fail");
                    }
                    push_flag = true;
                    break;
                }
                case BM_RESIZE_TPU_BICUBIC: 
                {                
                    int ret = bmcv.resize(format_image,resize_image,output_width_,output_height_,BMCV_INTER_BICUBIC);
                    // judge ret
                    if (ret != 0){
                        SPDLOG_ERROR("resize error! ret:{}",ret);
                        throw SailBMImageError("bmcv api fail");
                    }
                    push_flag = true;
                    break;
                }
                case BM_PADDING_VPP_NEAREST: 
                {    
                    if(!had_padding_atrr_){
                        SPDLOG_ERROR("Not SetPaddingAtrr, Can not use BM_PADDING_VPP_NEAREST!");
                        throw SailRuntimeError("invalid status");
                    }
                    pad_att_temp = CalcPaddingAtrr(format_image.width(),format_image.height(),output_width_,output_height_,align_);
                    padding_atrr.set_stx(pad_att_temp[0]);
                    padding_atrr.set_sty(pad_att_temp[1]);
                    padding_atrr.set_w(pad_att_temp[2]);
                    padding_atrr.set_h(pad_att_temp[3]);
                    if (!format_image.check_align()) {
                        format_image.align();

                    }
                    if (!resize_image.check_align()) {
                        resize_image.align();
                    }
                    int ret = bmcv.vpp_crop_and_resize_padding(format_image,resize_image,0,0,format_image.width(),format_image.height(),
                        output_width_,output_height_, padding_atrr);
                    // judge ret
                    if (ret != 0) {
                        SPDLOG_ERROR("ImagePreProcess resize thread:VPP crop and resize padding failed!");
                        throw SailBMImageError("bmcv api fail");
                    }
                    push_flag = true;
                    break;
                }
                case BM_PADDING_TPU_NEAREST: 
                {
                    if(!had_padding_atrr_){
                        SPDLOG_ERROR("Not SetPaddingAtrr, Can not use BM_PADDING_TPU_NEAREST!");
                        throw SailRuntimeError("invalid status");
                    }
                    pad_att_temp = CalcPaddingAtrr(format_image.width(),format_image.height(),output_width_,output_height_,align_);
                    padding_atrr.set_stx(pad_att_temp[0]);
                    padding_atrr.set_sty(pad_att_temp[1]);
                    padding_atrr.set_w(pad_att_temp[2]);
                    padding_atrr.set_h(pad_att_temp[3]);
                    spdlog::debug("IPP padding_atrr {} {} {} {}",pad_att_temp[0], pad_att_temp[1], pad_att_temp[2], pad_att_temp[3]);
                    
                    sail::BMImage resize_image_p = bmcv.crop_and_resize_padding(format_image,0,0,format_image.width(),format_image.height(),
                        output_width_,output_height_, padding_atrr, BMCV_INTER_NEAREST);
                    Resize_queue->Push(resize_image_p);
                    std::unique_lock<std::mutex> lck(resize_flag);
                    resize_flag_cond.notify_all(); 
                    break;
                }
                case BM_PADDING_TPU_LINEAR: 
                {
                    if(!had_padding_atrr_){
                        SPDLOG_ERROR("Not SetPaddingAtrr, Can not use BM_PADDING_TPU_LINEAR!");
                        throw SailRuntimeError("invalid status");
                    }
                    pad_att_temp = CalcPaddingAtrr(format_image.width(),format_image.height(),output_width_,output_height_,align_);
                    padding_atrr.set_stx(pad_att_temp[0]);
                    padding_atrr.set_sty(pad_att_temp[1]);
                    padding_atrr.set_w(pad_att_temp[2]);
                    padding_atrr.set_h(pad_att_temp[3]);
                    sail::BMImage resize_image_p = bmcv.crop_and_resize_padding(format_image,0,0,format_image.width(),format_image.height(),
                        output_width_,output_height_, padding_atrr,BMCV_INTER_LINEAR);
                    Resize_queue->Push(resize_image_p);
                    std::unique_lock<std::mutex> lck(resize_flag);
                    resize_flag_cond.notify_all(); 
                    break;
                }
                case BM_PADDING_TPU_BICUBIC: 
                {
                    if(!had_padding_atrr_){
                        SPDLOG_ERROR("Not SetPaddingAtrr, Can not use BM_PADDING_TPU_BICUBIC!");
                        throw SailRuntimeError("invalid status");
                    }
                    pad_att_temp = CalcPaddingAtrr(format_image.width(),format_image.height(),output_width_,output_height_,align_);
                    padding_atrr.set_stx(pad_att_temp[0]);
                    padding_atrr.set_sty(pad_att_temp[1]);
                    padding_atrr.set_w(pad_att_temp[2]);
                    padding_atrr.set_h(pad_att_temp[3]);
                    sail::BMImage resize_image_p = bmcv.crop_and_resize_padding(format_image,0,0,format_image.width(),format_image.height(),
                        output_width_,output_height_, padding_atrr,BMCV_INTER_BICUBIC);
                    Resize_queue->Push(resize_image_p);
                    std::unique_lock<std::mutex> lck(resize_flag);
                    resize_flag_cond.notify_all(); 
                    break;
                }
                default:
                {
                    SPDLOG_ERROR("Error resize mode: Unknown{}!",resize_mode_);
                    throw SailRuntimeError("invalid argument");
                    break;
                }
            } 
            if(push_flag){
                Resize_queue->Push(resize_image);
                std::unique_lock<std::mutex> lck(resize_flag);
                resize_flag_cond.notify_all();  
            }
            push_padding_atrr(pad_att_temp);
            if(print_flag_){
                double time_use_resize = (get_current_time_us() - time_start_resize)/1000;
                SPDLOG_INFO("resize time use: {} ms,Resize_queue size:{} .",time_use_resize,Resize_queue->Size());
            }
        }
        resize_exit_flag.set_flag(true);
        SPDLOG_INFO("Resize Thread End!");
    }

    void ImagePreProcess::ImagePreProcess_CC::convert_to_thread()
    {
        if (!had_resize_attr_){
            SPDLOG_ERROR("Not SetResizeImageAtrr, Create convert to thread failed!");
            throw SailRuntimeError("invalid status");
        }
        if(!set_convert_attr_flag){
            SPDLOG_ERROR("Not Set LinearTransParam, Please Call SetConvertAtrr First!");
            throw SailRuntimeError("invalid status");
        }
        sail::Handle handle(tpu_id_);
        sail::Bmcv bmcv(handle);
//         SPDLOG_INFO("Create Convert To Thread, pid:{}, tid:{} .",getpid(),gettid());
        convert_exit_flag.set_flag(false);
        while(true){
            if(stop_thread_flag.get_flag()){
                break;
            }
            
            if(queue_size_out_ >= 0 && Convert_queue->IsFull()){     //如果目标缓存队列的长度太长
                double time_start = get_current_time_us();
                std::unique_lock<std::mutex> lck(outdata_flag);
                outdata_flag_cond.wait_for(lck,std::chrono::milliseconds(20));
                if(print_flag_){
                    double time_use = (get_current_time_us() - time_start)/1000;
                    SPDLOG_INFO("Convert_queue full, convert_to_thread sleep {} ms.",time_use);
                }
                continue;
            }

            double time_start_convert_to = get_current_time_us();
            sail::BMImage resize_image;
            if(Resize_queue->Pop(resize_image) != 0){
                double time_start = get_current_time_us();
                std::unique_lock<std::mutex> lck(resize_flag);
                resize_flag_cond.wait_for(lck,std::chrono::milliseconds(20));
                if(print_flag_){
                    double time_use = (get_current_time_us() - time_start)/1000;
                    SPDLOG_INFO("convert_to_thread sleep {} ms.",time_use);  
                }
                continue;
            }
            sail::BMImage out_image = sail::BMImage(handle, output_height_, output_width_, out_format_, img_dtype_);
            int ret = bmcv.convert_to(resize_image,out_image,linear_trans_param_);
            // judge ret
            if (ret != 0){
                SPDLOG_ERROR("convert to error! ret:{}",ret);
                throw SailBMImageError("bmcv api fail");
            }
            Convert_queue->Push(out_image);
            if(print_flag_){
                double time_use_convert_to = (get_current_time_us() - time_start_convert_to)/1000;
                SPDLOG_INFO("convert_to time use: {} ms,Convert_queue size:{} .",time_use_convert_to,Convert_queue->Size());
            }
            std::unique_lock<std::mutex> lck(convert_flag);
            convert_flag_cond.notify_all();  
        }
        convert_exit_flag.set_flag(true);
        SPDLOG_INFO("Convert To Thread End!");
    }

    void ImagePreProcess::ImagePreProcess_CC::to_tensor_thread(){
        if (!had_resize_attr_){
            SPDLOG_ERROR("Not SetResizeImageAtrr, Create resize thread failed!");
            throw SailRuntimeError("invalid status");
        }
        if(!set_convert_attr_flag){
            SPDLOG_ERROR("Not Set LinearTransParam, Please Call SetConvertAtrr First!");
            throw SailRuntimeError("invalid status");
        }
        sail::Handle handle(tpu_id_);
        sail::Bmcv bmcv(handle);
//         SPDLOG_INFO("Create To Tensor Thread, pid:{}, tid:{} .",getpid(),gettid());
        to_tensor_exit_flag.set_flag(false);
        while(true){
            if(stop_thread_flag.get_flag()){
                break;
            }
            int current_size_temp = Convert_queue->Size();
            if(current_size_temp < batch_size_){
                double time_start = get_current_time_us();
                std::unique_lock<std::mutex> lck(convert_flag);
                convert_flag_cond.wait_for(lck,std::chrono::seconds(1));
                if(print_flag_){
                    double time_use = (get_current_time_us() - time_start)/1000;
                    SPDLOG_INFO("to_tensor_thread sleep {} ms.",time_use);  
                }
                continue;
            }
            switch(batch_size_){
            case 1:
            {
                sail::Tensor out_tensor = convert_to_Tensor_1N(handle,bmcv);
                push_tensor(out_tensor);
                break;
            }
            case 4:
            {
                sail::Tensor out_tensor = convert_to_Tensor_4N(handle,bmcv);
                push_tensor(out_tensor);
                break;
            }
            case 6:
            {
                sail::Tensor out_tensor = convert_to_Tensor_6N(handle,bmcv);
                push_tensor(out_tensor);
                break;
            }
            case 8:
            {
                sail::Tensor out_tensor = convert_to_Tensor_8N(handle,bmcv);
                push_tensor(out_tensor);
                break;
            }
            case 16:
            {
                sail::Tensor out_tensor = convert_to_Tensor_16N(handle,bmcv);
                push_tensor(out_tensor);
                break;
            }
            case 32:
            {
                sail::Tensor out_tensor = convert_to_Tensor_32N(handle,bmcv);
                push_tensor(out_tensor);
                break;
            }

            case 64:
            {
                sail::Tensor out_tensor = convert_to_Tensor_64N(handle,bmcv);
                push_tensor(out_tensor);
                break;
            }

            default:{
                SPDLOG_ERROR("Batch size is not supported, bathsize must be 1,4,8,16,32,64!");
                throw SailEngineError("not supported");
                break;
            }
            }
            std::unique_lock<std::mutex> lck(outdata_flag);
            outdata_flag_cond.notify_all(); 
        }
        to_tensor_exit_flag.set_flag(true);
        SPDLOG_INFO("To Tensor Thread End!");
    }


    ImagePreProcess::ImagePreProcess(
            int batch_size,
            sail_resize_type resize_mode,
            int tpu_id, 
            int queue_in_size, 
            int queue_out_size,
            bool use_mat_flag)
        :_impl(new ImagePreProcess_CC(batch_size,resize_mode,tpu_id,queue_in_size,queue_out_size,use_mat_flag))
    {}

    ImagePreProcess::~ImagePreProcess()
    {
        delete _impl;
    }

        
    void ImagePreProcess::SetResizeImageAtrr(					
            int output_width,				    
            int output_height,				    
            bool bgr2rgb,					    
            bm_image_data_format_ext  dtype){
        return _impl->SetResizeImageAtrr(output_width,output_height,bgr2rgb,dtype);
    }
        
    void ImagePreProcess::SetPaddingAtrr(		//设置输出图像的属性
            int padding_b,		        //输出图像的宽度
            int padding_g,		        //输出图像的高度
            int padding_r,		        //BGR转换RGB的标志位。
            int align){
        return _impl->SetPaddingAtrr(padding_b,padding_g,padding_r,align);
    }

    int ImagePreProcess::SetConvertAtrr(
            const std::tuple<
                std::pair<float, float>,
                std::pair<float, float>,
                std::pair<float, float>> &alpha_beta){
        return _impl->SetConvertAtrr(alpha_beta);
    }

    int ImagePreProcess::PushImage(
            int channel_idx, 
            int image_idx, 
            BMImage &image){
        return _impl->PushImage(channel_idx,image_idx,image);
    }

    std::tuple<sail::Tensor,std::vector<BMImage>,std::vector<int>,std::vector<int>,std::vector<std::vector<int>>> ImagePreProcess::GetBatchData(){
        return std::move(_impl->GetBatchData());
    }
    std::tuple<sail::Tensor,std::vector<cv::Mat>,std::vector<int>,std::vector<int>,std::vector<std::vector<int>>> ImagePreProcess::GetBatchData_CV(){
        return std::move(_impl->GetBatchData_CV());
    }

    void ImagePreProcess::set_print_flag(bool print_flag){
        return _impl->set_print_flag(print_flag);
    }

    void ImagePreProcess::exhausted()
    {
        return _impl->exhausted();
    }

    bool ImagePreProcess::get_exhausted_flag()
    {
        return _impl->get_exhausted_flag();
    }

    void ImagePreProcess::stop_thread()
    {
        return _impl->set_thread_exit();
        // return _impl->stop_thread_flag.set_flag(true);
    }

    class EngineImagePreProcess::EngineImagePreProcess_CC{
    public:
        EngineImagePreProcess_CC(std::string bmodel_name,int tpu_id, bool use_mat_out, std::vector<int> core_list);

        ~EngineImagePreProcess_CC();

        int InitImagePreProcess(sail_resize_type resize_mode, bool bgr2rgb, int queue_in_size, int queue_out_size);

        int SetPaddingAtrr(int padding_b, int padding_g, int padding_r, int align);

        int SetConvertAtrr(const std::tuple<std::pair<float, float>, std::pair<float, float>,std::pair<float, float>> &alpha_beta);

        int PushImage(int channel_idx, int image_idx, BMImage &image);

        std::tuple<std::map<std::string,sail::Tensor*>, std::vector<BMImage>, std::vector<int>, std::vector<int>, std::vector<std::vector<int>>> GetBatchData();
    
        std::tuple<std::map<std::string,sail::Tensor*>, std::vector<cv::Mat>, std::vector<int>, std::vector<int>, std::vector<std::vector<int>>> GetBatchData_CV();

        void exhausted();

        bool get_exhausted_flag();

    private:
        sail::Engine* engine_;
        sail::ImagePreProcess* pre_process_;

        int tpu_id_;
        std::vector<int> core_list_;
        std::string input_name_;
        std::string graph_name_;

        int net_width_;
        int net_height_;
        int batch_size_;

        int max_queue_size_;

        float input_scale_;                 // bmodel input scale
        bm_image_data_format_ext dtype_;    // input bm_image dtype

        std::vector<std::string>        output_name_;
        std::vector<std::vector<int>>   output_shape_;
        std::vector<bm_data_type_t>     output_dtype_;
        std::map<std::string, float>    output_scale_;

        std::tuple<std::pair<float, float>, std::pair<float, float>,std::pair<float, float>> alpha_beta_;

        std::condition_variable pushImage_flag_cond;    //向图像预处理模块送数据成功后或发出此信号
        std::mutex pushImage_flag;

        std::condition_variable InferResult_flag_cond;      //从预处理模块中取数据成功后发送此信号
        std::mutex InferResult_flag;

        std::deque<std::map<std::string,sail::Tensor*>>  output_queue;
        std::deque<BMImage>               pframes_queue;
        std::deque<cv::Mat>               pcvframes_queue;
        std::deque<int>                   inputChannels_queue;
        std::deque<int>                   inputIndex_queue;
        std::deque<std::vector<int>>      paddingAtrr_queue;
        std::mutex mutex_input_data;                //输入互斥

        Mutex_Flag inference_exit_flag;             //推理线程退出的标志位
        Mutex_Flag stop_thread_flag;                //线程退出的标志位

        bool print_flag_;                           //打印的标志位

        bool had_create_thread_flag;

        bool use_mat_flag_;                         //是否使用Mat作为输出

        bool exhausted_flag_;

        void scale_output_tensor(sail::Tensor* tensor, float* data, float scale);

        friend class EngineImagePreProcess;
private:
        int PushResultData(
            std::map<std::string,sail::Tensor*> output_tensormap, 
            std::vector<BMImage>& imgs, 
            std::vector<int>& channel, 
            std::vector<int>& index, 
            std::vector<std::vector<int>>& padding_atrr);     //将推理结果及原始数据放入缓存

        int GetResultData(
            std::map<std::string,sail::Tensor*>& output_tensormap, 
            std::vector<BMImage>& imgs, 
            std::vector<int>& channel, 
            std::vector<int>& index, 
            std::vector<std::vector<int>>& padding_atrr);     //从缓存中取出原始数据及推理结果

        int PushResultData_CV(
            std::map<std::string,sail::Tensor*> output_tensormap, 
            std::vector<cv::Mat>& imgs, 
            std::vector<int>& channel, 
            std::vector<int>& index, 
            std::vector<std::vector<int>>& padding_atrr);     //将推理结果及原始数据放入缓存

        int GetResultData_CV(
            std::map<std::string,sail::Tensor*>& output_tensormap, 
            std::vector<cv::Mat>& imgs, 
            std::vector<int>& channel, 
            std::vector<int>& index, 
            std::vector<std::vector<int>>& padding_atrr);     //从缓存中取出原始数据及推理结果


        int get_queue_size();           //获取当前缓存的大小

        void InferenceThread();

        void set_thread_exit();       //设置线程已经退出         

        void wait_thread_exit();      //等待线程退出

    };

    EngineImagePreProcess::EngineImagePreProcess_CC::EngineImagePreProcess_CC(std::string bmodel_name,int tpu_id, bool use_mat_out, std::vector<int> core_list)
    :engine_(NULL),pre_process_(NULL),tpu_id_(tpu_id),core_list_(core_list),print_flag_(false),had_create_thread_flag(false),use_mat_flag_(use_mat_out),exhausted_flag_(false)
    {
        engine_ = new sail::Engine(bmodel_name, tpu_id, DEVIO);
        if (engine_ == NULL){
            SPDLOG_INFO("Create Engine failed,bmode_name:{}, tpu:{}",bmodel_name,tpu_id);
            throw SailEngineError("Engine related error");
        }
        graph_name_ = engine_->get_graph_names()[0];
        input_name_ = engine_->get_input_names(graph_name_)[0];
        output_name_ = engine_->get_output_names(graph_name_);

        bm_data_type_t input_dtype = engine_->get_input_dtype(graph_name_,input_name_);
        std::vector<int> input_shape = engine_->get_input_shape(graph_name_,input_name_);
        input_scale_ = engine_->get_input_scale(graph_name_,input_name_);

        dtype_ = DATA_TYPE_EXT_1N_BYTE_SIGNED;
        if (input_dtype == BM_FLOAT32){
            dtype_ = DATA_TYPE_EXT_FLOAT32;
        }

        for (int i = 0; i< output_name_.size(); ++i) {
            output_dtype_.push_back(engine_->get_output_dtype(graph_name_,output_name_[i]));
            output_shape_.push_back(engine_->get_output_shape(graph_name_,output_name_[i]));
            float scale_temp = engine_->get_output_scale(graph_name_,output_name_[i]);
            output_scale_.insert(std::pair<std::string, float>(output_name_[i],scale_temp));
        }

        batch_size_ = input_shape[0];
        net_height_ = input_shape[2];
        net_width_ = input_shape[3];
        alpha_beta_ = std::make_tuple(std::pair<float,float>(1.0, 0),std::pair<float,float>(1.0,0),std::pair<float,float>(1.0,0));
        max_queue_size_ = 5;
        inference_exit_flag.set_flag(true);
        stop_thread_flag.set_flag(false);
    }

    EngineImagePreProcess::EngineImagePreProcess_CC::~EngineImagePreProcess_CC()
    {
        if(pre_process_){
            pre_process_->stop_thread();
        }
        SPDLOG_INFO("Start Set Thread Exit Flag!");
        set_thread_exit();
        SPDLOG_INFO("End Set Thread Exit Flag, Waiting Thread Exit....");
        wait_thread_exit();
        SPDLOG_INFO("All Thread Finshed!");
     
        if(engine_){
            delete engine_;
            engine_ = NULL;
        }
        if(pre_process_){
            delete pre_process_;
            pre_process_ = NULL;
        }
    }

    int EngineImagePreProcess::EngineImagePreProcess_CC::InitImagePreProcess(sail_resize_type resize_mode, bool bgr2rgb, int queue_in_size, int queue_out_size)
    {
        pre_process_ = new ImagePreProcess(batch_size_, resize_mode, tpu_id_, queue_in_size, queue_out_size, use_mat_flag_);
        if(pre_process_ == NULL) {
            SPDLOG_ERROR("Create ImagePreProcess failed, tpu:{}!",tpu_id_);
            return 1;
        }
        pre_process_->SetResizeImageAtrr(net_width_,net_height_,bgr2rgb, dtype_);
        return 0;
    }

    int EngineImagePreProcess::EngineImagePreProcess_CC::SetPaddingAtrr(int padding_b, int padding_g, int padding_r, int align)
    {
            if(pre_process_ == NULL) {
                SPDLOG_ERROR("ImagePreProcess is NULL!");
                return 1;
            }
            pre_process_->SetPaddingAtrr(padding_b, padding_g, padding_r, align);
            return 0;
    }

    int EngineImagePreProcess::EngineImagePreProcess_CC::SetConvertAtrr(const std::tuple<std::pair<float, float>, std::pair<float, float>,std::pair<float, float>> &alpha_beta)
    {
        if(pre_process_ == NULL) {
            SPDLOG_ERROR("ImagePreProcess is NULL!");
            return 1;
        }
        std::get<0>(alpha_beta_).first = std::get<0>(alpha_beta).first * input_scale_;
        std::get<0>(alpha_beta_).second = std::get<0>(alpha_beta).second * input_scale_;

        std::get<1>(alpha_beta_).first = std::get<1>(alpha_beta).first * input_scale_;
        std::get<1>(alpha_beta_).second = std::get<1>(alpha_beta).second * input_scale_;

        std::get<2>(alpha_beta_).first = std::get<2>(alpha_beta).first * input_scale_;
        std::get<2>(alpha_beta_).second = std::get<2>(alpha_beta).second * input_scale_;
        
        pre_process_->SetConvertAtrr(alpha_beta_);
        return 0;
    }

    int EngineImagePreProcess::EngineImagePreProcess_CC::PushImage(int channel_idx, int image_idx, BMImage &image)
    {
        if(pre_process_ == NULL) {
            SPDLOG_ERROR("ImagePreProcess is NULL!");
            return 1;
        }
        int ret = pre_process_->PushImage(channel_idx, image_idx, image);
        if(ret == 0){
            std::unique_lock<std::mutex> lck(pushImage_flag);
            pushImage_flag_cond.notify_all(); 
        }
        if(!had_create_thread_flag){
            had_create_thread_flag = true;
            std::thread thread_inference = std::thread(&EngineImagePreProcess_CC::InferenceThread,this);
            thread_inference.detach();
        }
        return ret;
    }
    
    int EngineImagePreProcess::EngineImagePreProcess_CC::get_queue_size()
    {
        std::lock_guard<std::mutex> lock(mutex_input_data);
        return output_queue.size();
    }


    int EngineImagePreProcess::EngineImagePreProcess_CC::PushResultData(
        std::map<std::string,sail::Tensor*> output_tensormap, 
        std::vector<BMImage>& imgs, 
        std::vector<int>& channel, 
        std::vector<int>& index, 
        std::vector<std::vector<int>>& padding_atrr)
    {
        { 
            std::lock_guard<std::mutex> lock(mutex_input_data);
            output_queue.push_back(std::move(output_tensormap));
            for(int i = 0; i < imgs.size(); ++i){
                pframes_queue.push_back(std::move(imgs[i]));
                inputChannels_queue.push_back(channel[i]);
                inputIndex_queue.push_back(index[i]);
                paddingAtrr_queue.push_back(padding_atrr[i]);
            }
        }
        std::unique_lock<std::mutex> lck(InferResult_flag);
        InferResult_flag_cond.notify_all();
        return 0;
    }

    int EngineImagePreProcess::EngineImagePreProcess_CC::PushResultData_CV(
        std::map<std::string,sail::Tensor*> output_tensormap, 
        std::vector<cv::Mat>& imgs, 
        std::vector<int>& channel, 
        std::vector<int>& index, 
        std::vector<std::vector<int>>& padding_atrr)
    {
        { 
            std::lock_guard<std::mutex> lock(mutex_input_data);
            output_queue.push_back(std::move(output_tensormap));
            for(int i = 0; i < imgs.size(); ++i){
                pcvframes_queue.push_back(std::move(imgs[i]));
                inputChannels_queue.push_back(channel[i]);
                inputIndex_queue.push_back(index[i]);
                paddingAtrr_queue.push_back(padding_atrr[i]);
            }
        }
        std::unique_lock<std::mutex> lck(InferResult_flag);
        InferResult_flag_cond.notify_all();
        return 0;
    }

    int EngineImagePreProcess::EngineImagePreProcess_CC::GetResultData(
        std::map<std::string,sail::Tensor*>& output_tensormap, 
        std::vector<BMImage>& imgs, 
        std::vector<int>& channel, 
        std::vector<int>& index, 
        std::vector<std::vector<int>>& padding_atrr)
    {
        output_tensormap.clear();
        imgs.clear();
        channel.clear();
        index.clear();
        padding_atrr.clear();
        std::lock_guard<std::mutex> lock(mutex_input_data);
        if(output_queue.size() <= 0){
            return 1;
        }
        output_tensormap = std::move(output_queue.front());
        for(int i = 0; i < batch_size_; ++i){
            imgs.push_back(std::move(pframes_queue.front()));
            channel.push_back(inputChannels_queue.front());
            index.push_back(inputIndex_queue.front());
            padding_atrr.push_back(paddingAtrr_queue.front());
            pframes_queue.pop_front();
            inputChannels_queue.pop_front();
            inputIndex_queue.pop_front();
            paddingAtrr_queue.pop_front();
        }
        output_queue.pop_front();
        return 0;
    }

    int EngineImagePreProcess::EngineImagePreProcess_CC::GetResultData_CV(
        std::map<std::string,sail::Tensor*>& output_tensormap, 
        std::vector<cv::Mat>& imgs, 
        std::vector<int>& channel, 
        std::vector<int>& index, 
        std::vector<std::vector<int>>& padding_atrr)
    {
        output_tensormap.clear();
        imgs.clear();
        channel.clear();
        index.clear();
        padding_atrr.clear();
        std::lock_guard<std::mutex> lock(mutex_input_data);
        if(output_queue.size() <= 0){
            return 1;
        }
        output_tensormap = std::move(output_queue.front());
        for(int i = 0; i < batch_size_; ++i){
            imgs.push_back(std::move(pcvframes_queue.front()));
            channel.push_back(inputChannels_queue.front());
            index.push_back(inputIndex_queue.front());
            padding_atrr.push_back(paddingAtrr_queue.front());
            pcvframes_queue.pop_front();
            inputChannels_queue.pop_front();
            inputIndex_queue.pop_front();
            paddingAtrr_queue.pop_front();
        }
        output_queue.pop_front();
        return 0;
    }

    void EngineImagePreProcess::EngineImagePreProcess_CC::InferenceThread()
    {
//         SPDLOG_INFO("Create InferenceThread Thread, pid:{}, tid:{} .",getpid(),gettid());

        if (engine_ == NULL){
            SPDLOG_INFO("Engine is NULL!");
            throw SailRuntimeError("invalid status");
        }
        if(pre_process_ == NULL) {
            SPDLOG_ERROR("ImagePreProcess is NULL!");
            throw SailRuntimeError("invalid status");
        }
        sail::Handle handle(tpu_id_);
        sail::Bmcv bmcv(handle);
        bool use_opencv = use_mat_flag_;
        inference_exit_flag.set_flag(false);
        while(true){
            if(stop_thread_flag.get_flag()){
                break;
            }

            if(get_queue_size() >= max_queue_size_){ //如果已经处理的数据超过缓存,就先不处理
                double time_start = get_current_time_us();
                std::unique_lock<std::mutex> lck(pushImage_flag);
                pushImage_flag_cond.wait_for(lck,std::chrono::milliseconds(20));
                if(print_flag_){
                    double time_use = (get_current_time_us() - time_start)/1000;
                    SPDLOG_INFO("Output full:{},do not get data,InferenceThread sleep {} ms.", get_queue_size(),time_use);
                }
                continue;
            }
            
            if(!use_opencv){
                double time_start_get_data = get_current_time_us();
                auto result = pre_process_->GetBatchData();
                double time_end_get_data = get_current_time_us();

                sail::Tensor input_tensor = std::move(std::get<0>(result));
                std::vector<BMImage> imgs = std::move(std::get<1>(result));
                std::vector<int> channel = std::move(std::get<2>(result));
                std::vector<int> index = std::move(std::get<3>(result));
                std::vector<std::vector<int>> padding_atrr = std::move(std::get<4>(result));

                std::map<std::string,sail::Tensor*> input_tensormap;
                input_tensormap.insert(std::pair<std::string,sail::Tensor*>(input_name_,&input_tensor));
                std::map<std::string,sail::Tensor*> output_tensormap;
                for(int i = 0; i < output_name_.size();++i){
                    sail::Tensor* output_tensor = new sail::Tensor(handle, output_shape_[i], output_dtype_[i], true, true);
                    output_tensormap.insert(std::pair<std::string,sail::Tensor*>(output_name_[i], output_tensor));
                }
                if(channel.size() == 0){
                    continue;
                }
                engine_->process(graph_name_,input_tensormap,output_tensormap,core_list_);
                double time_end_inference = get_current_time_us();

                PushResultData(output_tensormap, imgs, channel, index, padding_atrr);
            }else{
                double time_start_get_data = get_current_time_us();
                auto result = pre_process_->GetBatchData_CV();
                double time_end_get_data = get_current_time_us();

                sail::Tensor input_tensor = std::move(std::get<0>(result));
                std::vector<cv::Mat> imgs = std::move(std::get<1>(result));
                std::vector<int> channel = std::move(std::get<2>(result));
                std::vector<int> index = std::move(std::get<3>(result));
                std::vector<std::vector<int>> padding_atrr = std::move(std::get<4>(result));

                std::map<std::string,sail::Tensor*> input_tensormap;
                input_tensormap.insert(std::pair<std::string,sail::Tensor*>(input_name_,&input_tensor));
                std::map<std::string,sail::Tensor*> output_tensormap;
                if(channel.size() == 0){
                    continue;
                }
                for(int i = 0; i < output_name_.size();++i){
                    sail::Tensor* output_tensor = new sail::Tensor(handle, output_shape_[i], output_dtype_[i], true, true);
                    output_tensormap.insert(std::pair<std::string,sail::Tensor*>(output_name_[i], output_tensor));
                }
                engine_->process(graph_name_,input_tensormap,output_tensormap,core_list_);
                double time_end_inference = get_current_time_us();

                PushResultData_CV(output_tensormap, imgs, channel, index, padding_atrr);
            }
           
        }
        inference_exit_flag.set_flag(true);
        SPDLOG_INFO("Inference Thread End!");
    }

    void EngineImagePreProcess::EngineImagePreProcess_CC::set_thread_exit()
    {
        stop_thread_flag.set_flag(true);
    }   

    void EngineImagePreProcess::EngineImagePreProcess_CC::wait_thread_exit()
    {
        while(true){
            if(!inference_exit_flag.get_flag()){
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            break;
        }

    }

    std::tuple<std::map<std::string,sail::Tensor*>, 
        std::vector<BMImage>,
        std::vector<int>,
        std::vector<int>,
        std::vector<std::vector<int>>> EngineImagePreProcess::EngineImagePreProcess_CC::GetBatchData(){
        
        std::map<std::string,sail::Tensor*> out_map;
        std::vector<sail::BMImage> images;
        std::vector<int> channels; 
        std::vector<int> indexs;
        std::vector<std::vector<int>> padding_atrr;

        while(true){
            if(GetResultData(out_map,images,channels,indexs,padding_atrr) == 0){
                break;
            }
            if(stop_thread_flag.get_flag()){
                break;
            }

            double time_start = get_current_time_us();
            std::unique_lock<std::mutex> lck(InferResult_flag);
            InferResult_flag_cond.wait_for(lck,std::chrono::seconds(1));
            if(print_flag_){
                double time_use = (get_current_time_us() - time_start)/1000;
                SPDLOG_INFO("GetBatchData sleep {} ms.",time_use);  
            }
        }
        return std::move(std::make_tuple(std::move(out_map),std::move(images),channels,indexs,padding_atrr));
    }

    std::tuple<std::map<std::string,sail::Tensor*>, 
        std::vector<cv::Mat>,
        std::vector<int>,
        std::vector<int>,
        std::vector<std::vector<int>>> EngineImagePreProcess::EngineImagePreProcess_CC::GetBatchData_CV(){
        
        std::map<std::string,sail::Tensor*> out_map;
        std::vector<cv::Mat> images;
        std::vector<int> channels; 
        std::vector<int> indexs;
        std::vector<std::vector<int>> padding_atrr;

        while(true){
            if(GetResultData_CV(out_map,images,channels,indexs,padding_atrr) == 0){
                break;
            }
            if(stop_thread_flag.get_flag()){
                break;
            }

            double time_start = get_current_time_us();
            std::unique_lock<std::mutex> lck(InferResult_flag);
            InferResult_flag_cond.wait_for(lck,std::chrono::seconds(1));
            if(print_flag_){
                double time_use = (get_current_time_us() - time_start)/1000;
                SPDLOG_INFO("to_tensor_thread sleep {} ms.",time_use);  
            }
        }
        return std::move(std::make_tuple(std::move(out_map),std::move(images),channels,indexs,padding_atrr));
    }

    void EngineImagePreProcess::EngineImagePreProcess_CC::scale_output_tensor(sail::Tensor* tensor, float* data, float scale){
        std::vector<int> shape = tensor->shape();
        int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        bm_data_type_t dtype = tensor->dtype();
        if (dtype == BM_FLOAT32) {
            float* src = reinterpret_cast<float*>(tensor);
            memcpy(data, src, size * sizeof(float));
        } else if (dtype == BM_INT8) {
            int8_t* src = reinterpret_cast<int8_t*>(tensor);
            engine_->scale_int8_to_fp32(src, data, scale, size);
        } else if (dtype == BM_UINT8) {
            uint8_t* src = reinterpret_cast<uint8_t*>(tensor);
            engine_->scale_uint8_to_fp32(src, data, scale, size);
        } else if (dtype == BM_INT32) {
            int32_t* src = reinterpret_cast<int32_t*>(tensor);
            engine_->scale_int32_to_fp32(src, data, scale, size);
        }else{
            SPDLOG_ERROR("scale_output_tensor() not support!");
            throw SailTensorError("not supported");
        }
    }

    void EngineImagePreProcess::EngineImagePreProcess_CC::exhausted()
    {
        exhausted_flag_ = true;
        pre_process_->exhausted();
    }

    bool EngineImagePreProcess::EngineImagePreProcess_CC::get_exhausted_flag()
    {
        bool ret = false;
        if(exhausted_flag_)
        {
            bool pre_process_stop = pre_process_->get_exhausted_flag();
            if(pre_process_stop)
                ret = true;
        }
        return ret;
    }

    EngineImagePreProcess::EngineImagePreProcess(const std::string& bmodel_path, int tpu_id, bool use_mat_output,std::vector<int> core_list)
    :_impl(new EngineImagePreProcess_CC(bmodel_path, tpu_id, use_mat_output,core_list))
    {

    }

    EngineImagePreProcess::~EngineImagePreProcess(){
        delete _impl;
    }

    int EngineImagePreProcess::InitImagePreProcess(
        sail_resize_type resize_mode,
        bool bgr2rgb,					    
        int queue_in_size, 
        int queue_out_size){
        return _impl->InitImagePreProcess(resize_mode,bgr2rgb,queue_in_size,queue_out_size);
    }

    int EngineImagePreProcess::SetPaddingAtrr(
        int padding_b,
        int padding_g,	
        int padding_r, 
        int align){
        return _impl->SetPaddingAtrr(padding_b,padding_g,padding_r,align);
    }

    int EngineImagePreProcess::SetConvertAtrr(
        const std::tuple<
            std::pair<float, float>,
            std::pair<float, float>,
            std::pair<float, float>> &alpha_beta){
        return _impl->SetConvertAtrr(alpha_beta);
    }

    int EngineImagePreProcess::PushImage(
        int channel_idx, 
        int image_idx, 
        BMImage &image){
        return _impl->PushImage(channel_idx,image_idx,image);
    }

    std::tuple<std::map<std::string,sail::Tensor*>, 
        std::vector<BMImage>,
        std::vector<int>,
        std::vector<int>,
        std::vector<std::vector<int>>> EngineImagePreProcess::GetBatchData(bool need_d2s){
        auto result = _impl->GetBatchData();
        std::map<std::string,sail::Tensor*> output_tensormap = std::move(std::get<0>(result));
        auto iter = output_tensormap.begin();
        while(iter != output_tensormap.end()){
            if (need_d2s)
                iter->second->sync_d2s();
            iter++;
        }
        return std::move(std::make_tuple(std::move(output_tensormap),std::move(std::get<1>(result)),std::get<2>(result),std::get<3>(result),std::get<4>(result)));
    }

    std::tuple<std::map<std::string,sail::Tensor*>, 
        std::vector<cv::Mat>,
        std::vector<int>,
        std::vector<int>,
        std::vector<std::vector<int>>> EngineImagePreProcess::GetBatchData_CV(bool need_d2s){
        auto result = _impl->GetBatchData_CV();
        std::map<std::string,sail::Tensor*> output_tensormap = std::move(std::get<0>(result));
        auto iter = output_tensormap.begin();
        while(iter != output_tensormap.end()){
            if(need_d2s)
                iter->second->sync_d2s();
            iter++;
        }
        return std::move(std::make_tuple(std::move(output_tensormap),std::move(std::get<1>(result)),std::get<2>(result),std::get<3>(result),std::get<4>(result)));
    }

#ifdef PYTHON

    std::tuple<std::vector<TensorPTRWithName>, 
        std::vector<BMImage>,
        std::vector<int>,
        std::vector<int>,
        std::vector<std::vector<int>>> EngineImagePreProcess::GetBatchData_py(bool need_d2s)
    {
        auto result = GetBatchData(need_d2s);
        std::map<std::string,sail::Tensor*> output_tensormap = std::get<0>(result);
        std::vector<TensorPTRWithName> result_tensorlist;
        auto iter = output_tensormap.begin();
        while(iter != output_tensormap.end()){
            TensorPTRWithName tensor_with_name;
            tensor_with_name.name = iter->first;
            tensor_with_name.data = iter->second;
            result_tensorlist.push_back(std::move(tensor_with_name));
            iter++;
        }
        return std::move(std::make_tuple(std::move(result_tensorlist),std::move(std::get<1>(result)),std::get<2>(result),std::get<3>(result),std::get<4>(result)));
    }
    pybind11::array_t<uint8_t> cvmat_to_numpy_tmp(cv::Mat cv_mat){
        std::vector<pybind11::ssize_t> shape;

        pybind11::ssize_t item_size = 1;
        std::string format;
        if (cv_mat.type()  == CV_8UC3) {
            shape.push_back(cv_mat.rows);
            shape.push_back(cv_mat.cols);
            shape.push_back(3);
            item_size = sizeof(uint8_t);
            format = pybind11::format_descriptor<uint8_t>::format();
        } else if(cv_mat.type()  == CV_8UC1){
            shape.push_back(cv_mat.rows);
            shape.push_back(cv_mat.cols);
            item_size = sizeof(uint8_t);
            format = pybind11::format_descriptor<uint8_t>::format();
        }else{
            SPDLOG_ERROR("Mat type not support: {}",cv_mat.type());
            throw SailBMImageError("not supported");
        }
        
        int stride_temp = FFALIGN(cv_mat.cols * 3 * item_size, SAIL_ALIGN); // ceiling to 64 * N

        pybind11::ssize_t ndim = shape.size();
        std::vector<pybind11::ssize_t> stride;
        for (size_t i = 1; i < shape.size(); i++) {
            stride.push_back(cv_mat.step[i-1]);
        }
        stride.push_back(item_size);
        pybind11::buffer_info output_buf(cv_mat.data, item_size, format,
                                            ndim, shape, stride);
        return std::move(pybind11::array_t<uint8_t>(output_buf));
    }

    std::tuple<std::map<std::string, pybind11::array_t<float>>, 
        std::vector<BMImage>,
        std::vector<int>,
        std::vector<int>,
        std::vector<std::vector<int>>> EngineImagePreProcess::GetBatchData_Npy(){
        
        std::tuple<std::map<std::string,sail::Tensor*>, 
        std::vector<BMImage>,
        std::vector<int>,
        std::vector<int>,
        std::vector<std::vector<int>>> result;
        double time_start;
        double time_end;
        {       
            pybind11::gil_scoped_release release;
            time_start = get_current_time_us();
            result = _impl->GetBatchData();
            time_end = get_current_time_us();
        }
        std::map<std::string,sail::Tensor*> output_tensors_map = std::get<0>(result);
        std::map<std::string, pybind11::array_t<float>> output;
        auto iter = output_tensors_map.begin();
        while(iter != output_tensors_map.end()){
            sail::Tensor* input_tensor = iter->second;
            auto iter_scale = _impl->output_scale_.find(iter->first);
            float scale = iter_scale->second;
            auto ndarray = pybind11::array_t<float>(input_tensor->shape());
            float* data = ndarray.mutable_data();
            if (input_tensor->dtype() == BM_INT8
                || input_tensor->dtype() == BM_UINT8
                || input_tensor->dtype() == BM_INT32) {
                input_tensor->sync_d2s();
                _impl->scale_output_tensor(input_tensor, data, scale);
            } else {
                bm_device_mem_t dev_data_ = input_tensor->dev_data();
                int size_shape = std::accumulate(input_tensor->shape().begin(), input_tensor->shape().end(),
                                            1, std::multiplies<int>());
                int data_size = size_shape * sizeof(float);
                int ret = bm_memcpy_d2s_partial(input_tensor->get_handle().data(), data, dev_data_, data_size);
                if (ret != BM_SUCCESS) {
                    spdlog::error("bm_memcpy_d2s_partial(sys_data=%p, dev_data.addr=%p, size=%d)\n",
                                    data, (void *) dev_data_.u.device.device_addr, data_size);
                    throw SailRuntimeError("bmlib api fail");
                }
            }
            output[pybind11::str(iter->first)] = ndarray;
            delete input_tensor;
            iter++;
        }
        double time_end_scale = get_current_time_us();
        // SPDLOG_INFO("GetResult: {}us, Scale: {}us",time_end-time_start, time_end_scale-time_end);

        return std::move(std::make_tuple(std::move(output),std::move(std::get<1>(result)),std::get<2>(result),std::get<3>(result),std::get<4>(result)));
    }

    std::tuple<std::map<std::string, pybind11::array_t<float>>, 
        std::vector<pybind11::array_t<uint8_t>>,
        std::vector<int>,
        std::vector<int>,
        std::vector<std::vector<int>>> EngineImagePreProcess::GetBatchData_Npy2(){
        
        std::tuple<std::map<std::string,sail::Tensor*>, 
        std::vector<cv::Mat>,
        std::vector<int>,
        std::vector<int>,
        std::vector<std::vector<int>>> result;
        double time_start;
        double time_end;
        {       
            pybind11::gil_scoped_release release;
            time_start = get_current_time_us();
            result = _impl->GetBatchData_CV();
            time_end = get_current_time_us();
        }
        std::map<std::string,sail::Tensor*> output_tensors_map = std::get<0>(result);
        std::map<std::string, pybind11::array_t<float>> output;
        auto iter = output_tensors_map.begin();
        while(iter != output_tensors_map.end()){
            sail::Tensor* input_tensor = iter->second;
            auto iter_scale = _impl->output_scale_.find(iter->first);
            float scale = iter_scale->second;
            auto ndarray = pybind11::array_t<float>(input_tensor->shape());
            float* data = ndarray.mutable_data();
            if (input_tensor->dtype() == BM_INT8
                || input_tensor->dtype() == BM_UINT8
                || input_tensor->dtype() == BM_INT32) {
                input_tensor->sync_d2s();
                _impl->scale_output_tensor(input_tensor, data, scale);
            } else {
                bm_device_mem_t dev_data_ = input_tensor->dev_data();
                int size_shape = std::accumulate(input_tensor->shape().begin(), input_tensor->shape().end(),
                                            1, std::multiplies<int>());
                int data_size = size_shape * sizeof(float);
                int ret = bm_memcpy_d2s_partial(input_tensor->get_handle().data(), data, dev_data_, data_size);
                if (ret != BM_SUCCESS) {
                    spdlog::error("bm_memcpy_d2s_partial(sys_data=%p, dev_data.addr=%p, size=%d)\n",
                                    data, (void *) dev_data_.u.device.device_addr, data_size);
                }
            }
            output[pybind11::str(iter->first)] = ndarray;
            delete input_tensor;
            iter++;
        }
        double time_end_scale = get_current_time_us();
        // SPDLOG_INFO("GetResult: {}us, Scale: {}us",time_end-time_start, time_end_scale-time_end);

        std::vector<pybind11::array_t<uint8_t>> mat_array;
        std::vector<cv::Mat> mat_tmp = std::move(std::get<1>(result));
        for (int i=0;i<mat_tmp.size();++i){
            mat_array.push_back(cvmat_to_numpy_tmp(mat_tmp[i]));
        }
        return std::move(std::make_tuple(std::move(output),std::move(mat_array),std::get<2>(result),std::get<3>(result),std::get<4>(result)));
    }

#endif

    std::string EngineImagePreProcess::get_graph_name()
    {
        return _impl->graph_name_;
    }

    int EngineImagePreProcess::get_input_width()
    {
        return _impl->net_width_;
    }

    int EngineImagePreProcess::get_input_height()
    {
        return _impl->net_height_;
    }

    std::vector<std::string> EngineImagePreProcess::get_output_names(){
        return _impl->output_name_;
    }

    std::vector<int> EngineImagePreProcess::get_output_shape(const std::string& tensor_name)
    {
        return _impl->engine_->get_output_shape(get_graph_name(),tensor_name);
    }

    void EngineImagePreProcess::exhausted()
    {
        return _impl->exhausted();
    }

    bool EngineImagePreProcess::get_exhausted_flag()
    {
        return _impl->get_exhausted_flag();
    }

    class ImageListDecoder      //通过此方法可以将图片预先加载到TPU中，主要为了防止出现图片解码瓶颈
    {
    public:
        /**
         * @brief Construct a new Image List Decoder object
         * 
         * @param image_list image name list
         * @param tpu_id     TPU ID. You can use bm-smi to see available IDs.
         * @param queue_size max queue size
         */
        explicit ImageListDecoder(
                    std::vector<std::string>& image_list,
                    int tpu_id,
                    int queue_size);

        /**
         * @brief Destroy the Image List Decoder object
         * 
         */
        ~ImageListDecoder();

        /**
         * @brief Set the Resize Attr object
         * 
         * @param width output width
         * @param height output height
         * @param resize_alg Resize algorithm, defalut BMCV_INTER_LINEAR
         * @return int, 0 for success and other for failure 
         */
        int setResizeAttr(int width, int height, bmcv_resize_algorithm resize_alg = BMCV_INTER_LINEAR);

        /**
         * @brief Start read images
         * 
         * @return int, 0 for success and other for failure 
         */
        int start();

        /**
         * @brief 
         * 
         * @param image 
         * @return int 0:success, 1:queue empty, -1:stop and exit
         */
        int read(BMImage &image);

        /**
         * @brief stop thread
         * 
         */
        void stop();

        /**
         * @brief Get the schedule object
         * 
         * @return int The number of decoded images
         */
        int get_schedule();
 
    private:
        /* data */
        BMImageQueue* pframes_queue;

        std::vector<std::string> image_name_list;
        std::condition_variable data_flag_cond;
        std::mutex mutex_data_flag;

        int current_index_;          //当前的进度
        int tpu_id_;                 //使用的dev

        bool stop_thread_flag;      //线程退出的标志位
        std::mutex mutex_stop_;     //线程退出互斥锁
        std::condition_variable exit_cond;  //线程已经退出的信号量
        std::mutex mutex_exit_;             //线程已经退出互斥锁
        bool exit_thread_flag;              //线程已经退出的标志位

        bmcv_resize_algorithm resize_alg_;      //如果设置resize, resize使用的算法
        bool resize_flag;                       //是否需要resize的标志位
        bool thread_flag;                       //线程是否已经开启的标志位
        int width_;                             //输出图片的宽
        int height_;                            //输出图片的高


    private:
        void decoder_thread();          //解码线程
        
        void set_stop_flag(bool flag);  //设置线程退出的标志位

        bool get_stop_flag();           //获取线程退出的标志位

        void notify_data_flag();        //发送已经有缓存数据的信号

        void set_thread_exit();         //设置线程已经退出     

        void wait_thread_exit();        //等待线程退出
    };
    
    ImageListDecoder::ImageListDecoder(
                    std::vector<std::string>& image_list,
                    int tpu_id,
                    int queue_size):
        image_name_list(image_list),current_index_(0),stop_thread_flag(false),
        exit_thread_flag(true), tpu_id_(tpu_id),thread_flag(false),resize_flag(false)
    {
        pframes_queue = new BMImageQueue(queue_size);
    }

    int ImageListDecoder::setResizeAttr(int width, int height, bmcv_resize_algorithm resize_alg)
    {
        if(thread_flag){
            SPDLOG_INFO("Needs to be called before the start()!");
            return 1;
        }
        width_ = width;
        height_ = height;
        resize_alg_ = resize_alg;
        resize_flag = true;
        return 0;
    }


    int ImageListDecoder::start()
    {
        if(!thread_flag){
            std::thread thread_decoder = std::thread(&ImageListDecoder::decoder_thread,this);
            thread_decoder.detach();
            thread_flag = true;
            return 0;
        }
        SPDLOG_INFO("Thread has been created!");
        return 1;

    }

    ImageListDecoder::~ImageListDecoder()
    {
        SPDLOG_INFO(">>Dev-{},Set Stop Flag!",tpu_id_);
        set_stop_flag(true);
        SPDLOG_INFO(">>Dev-{},Wait Thread Finshed: {}!",tpu_id_,get_stop_flag());
        wait_thread_exit(); 
        SPDLOG_INFO(">>Dev-{},All end!",tpu_id_);
        delete pframes_queue;
        pframes_queue = NULL;
    }
    
    void ImageListDecoder::set_stop_flag(bool flag)
    {
        std::lock_guard<std::mutex> lock(mutex_stop_);
        stop_thread_flag = flag;
    }

    int ImageListDecoder::get_schedule()
    {
        return current_index_;
    }

    bool ImageListDecoder::get_stop_flag()
    {
        std::lock_guard<std::mutex> lock(mutex_stop_);
        return stop_thread_flag;
    }

    void ImageListDecoder::notify_data_flag()
    {
        std::unique_lock<std::mutex> lck(mutex_data_flag);
        data_flag_cond.notify_all();
    }

    void ImageListDecoder::set_thread_exit()
    {
        std::unique_lock<std::mutex> lck(mutex_exit_);
        exit_cond.notify_all();
    }

    void ImageListDecoder::wait_thread_exit()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_exit_);
            if(exit_thread_flag){
                return;
            }
        }
        std::unique_lock<std::mutex> lck(mutex_exit_);
        exit_cond.wait(lck);
    }  

    void ImageListDecoder::stop()
    {
        return set_stop_flag(true);
    }

    void ImageListDecoder::decoder_thread()
    {
        sail::Handle handle(tpu_id_);
        sail::Bmcv bmcv(handle);
        {        
            std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
            exit_thread_flag = false;
        }
        while (true)
        {
            if(get_stop_flag()){
                break;
            }       
            if(pframes_queue->IsFull()){
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            std::ifstream file(image_name_list[current_index_], std::ios::binary | std::ios::ate);
            if(!file.is_open()){
                SPDLOG_INFO("0-Can not open file: {}",image_name_list[current_index_]);
                continue;
            }
            size_t size = file.tellg();
            file.seekg(0, std::ios::beg);
            std::vector<unsigned char> jpeg_data(size);
            if (!file.read(reinterpret_cast<char*>(jpeg_data.data()), size)){
                SPDLOG_INFO("1-Can not open file: {}",image_name_list[current_index_]);
                file.close();
                continue;
            }
            file.close();
            sail::BMImage image_temp = bmcv.imdecode(jpeg_data.data(),size);
            if(resize_flag){
                sail::BMImage image_resize = bmcv.resize(image_temp, width_,height_, resize_alg_);
                pframes_queue->Push(image_resize);
            }else{
                pframes_queue->Push(image_temp);
            }

            notify_data_flag();
            current_index_++;
            if(current_index_>=image_name_list.size()){
                stop();
                break;
            }
        }
        set_thread_exit();
        {
            std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
            exit_thread_flag = true;
        }
        SPDLOG_INFO(">>>>>Dev-{} Image List Decoder Thread Finshed!", tpu_id_);
    }

    int ImageListDecoder::read(BMImage &image)
    {
        int ret = pframes_queue->Pop(image);
        if(ret != 0){
            if(get_stop_flag()){
                ret = -1;
            } 
        }
        return ret;
    }

    class DecoderImages::DecoderImages_CC
    {
    public:
        DecoderImages_CC(
                    std::vector<std::string>& image_list,
                    int tpu_id,
                    int queue_size,
                    int thread_count=4);
        ~DecoderImages_CC();

        int setResizeAttr(int width, int height, bmcv_resize_algorithm resize_alg = BMCV_INTER_LINEAR);

        int start();

        int read(BMImage &image);

        void stop();

        int get_schedule();
    private:
        std::vector<ImageListDecoder*> decoder;
        std::vector<std::string>* image_name_list;
        int max_queue_size;

        int current_idx;
        int thread_num;
    private:
    };
    
    DecoderImages::DecoderImages_CC::DecoderImages_CC(
        std::vector<std::string>& image_list,
                    int tpu_id,
                    int queue_size,
                    int thread_count)
        :image_name_list(NULL),current_idx(0),thread_num(thread_count)
    {
        if(image_list.size() < thread_count){
            thread_num = 1;
        }
        max_queue_size = queue_size/thread_num;
        if(queue_size%thread_num != 0){
            max_queue_size += 1;
        }
        image_name_list = new std::vector<std::string>[thread_num];
        for(int i=0;i<image_list.size();i++){
            image_name_list[i%thread_num].push_back(image_list[i]);
        }
        for(int i=0;i<thread_num;i++){
            decoder.push_back(new ImageListDecoder(image_name_list[i],tpu_id,max_queue_size));
        }
    }
    
    DecoderImages::DecoderImages_CC::~DecoderImages_CC()
    {
        stop();
        for (size_t i = 0; i < thread_num; i++)        {
           delete decoder[i];
        }
        decoder.clear();
        delete []image_name_list;
    }

    int DecoderImages::DecoderImages_CC::setResizeAttr(int width, int height, bmcv_resize_algorithm resize_alg)
    {
        int ret = 0;
        for (size_t i = 0; i < thread_num; i++)        {
           ret = decoder[i]->setResizeAttr(width,height,resize_alg);
           if(ret != 0){
                return ret;
           }
        }
        return ret;
    }

    int DecoderImages::DecoderImages_CC::start()
    {
        int ret = 0;
        for (size_t i = 0; i < thread_num; i++)        {
           ret = decoder[i]->start();
           if(ret != 0){
                return ret;
           }
        }
        return ret;
    }
    
    int DecoderImages::DecoderImages_CC::read(BMImage &image)
    {
        int ret = decoder[current_idx%thread_num]->read(image);
        if(ret == 0){
            current_idx++;
        }
        return ret;
    }

    void DecoderImages::DecoderImages_CC::stop()
    {
        for (size_t i = 0; i < thread_num; i++)        {
           decoder[i]->stop();
        }
    }

    int DecoderImages::DecoderImages_CC::get_schedule()
    {
        int sch = 0;
        for (size_t i = 0; i < thread_num; i++)        {
           sch += decoder[i]->get_schedule();
        }
        return sch;
    }

    DecoderImages::DecoderImages(
                    std::vector<std::string>& image_list,
                    int tpu_id,
                    int queue_size)
        :_impl(new DecoderImages_CC(image_list,tpu_id,queue_size))
    {}

    DecoderImages::~DecoderImages()
    {
        delete _impl;
    }

    int DecoderImages::setResizeAttr(int width, int height, bmcv_resize_algorithm resize_alg)
    {
        return _impl->setResizeAttr(width,height,resize_alg);
    }

    int DecoderImages::start()
    {
        return _impl->start();
    }

    int DecoderImages::read(BMImage &image)
    {
        return _impl->read(image);
    }
        
    void DecoderImages::stop()
    {
        return _impl->stop();
    }

    int DecoderImages::get_schedule()
    {
        return _impl->get_schedule();
    }
}

#endif

