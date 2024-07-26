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
#include <engine.h>
#include <perf.h>
#include <numeric>
#include <sys/types.h>
#include "internal.h"


using namespace std;

namespace sail {
    static mutex perf_print_flag_;
    void PrintThreadLog_Perf(std::string file_name, int line, std::string message)
    {
        std::lock_guard<std::mutex> lock_print(perf_print_flag_);
        std::cout << "# File[" << file_name << ":" << line << "], ";
        std::cout << "Thread[" << std::this_thread::get_id()<<"], "<< message << std::endl;
    }

    class Perf_Mutex_Flag{
    public:
        Perf_Mutex_Flag(){
            flag_ = true;
        }
        ~Perf_Mutex_Flag(){}

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

    class TensorMapQueue
    {
    public:
        TensorMapQueue();
        ~TensorMapQueue();

        int push_data(std::map<std::string, Tensor*> tensor);      //push成功返回0，其他值表示失败

        int get_data(std::map<std::string, Tensor*>& tensor);       //从队列中获取数据,成功返回0，其他值表示失败

        int get_queue_size();                                       //获取当前queue的长度

        int clear_data();                                           //清空数据
    private:
        std::mutex mutex_data;  //互斥
        std::deque<std::map<std::string, Tensor*>> data_que;
    };
    
    TensorMapQueue::TensorMapQueue()
    {

    }

    int TensorMapQueue::get_queue_size()
    {
        int size_temp = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_data);
            size_temp = data_que.size();
        }
        return size_temp;
    }

    int TensorMapQueue::push_data(std::map<std::string, Tensor*> tensor)
    {
        data_que.push_back(std::move(tensor));
        return 0;
    }

    int TensorMapQueue::get_data(std::map<std::string, Tensor*>& tensor)
    {
        std::lock_guard<std::mutex> lock(mutex_data);
        if(data_que.size() <= 0){
            return 1;
        }
        tensor = std::move(data_que.front());
        data_que.pop_front();
        return 0;
    }

    int TensorMapQueue::clear_data()
    {
        std::lock_guard<std::mutex> lock(mutex_data);
        int size_temp = data_que.size();
        for (size_t i = 0; i < size_temp; i++)
        {
            std::map<std::string, Tensor*> output_tensormap = data_que.front();
            auto iter = output_tensormap.begin();
            while(iter != output_tensormap.end()){
                delete iter->second;
                iter++;
            }
            data_que.pop_front();
        }
        return 0;
    }

    TensorMapQueue::~TensorMapQueue()
    {
    }
    
    class MutexVariable
    {
    public:
        MutexVariable();
        ~MutexVariable();

        void notify_all(){
            std::unique_lock<std::mutex> lck(mutex_flag);
            flag_cond.notify_all();
        }
        void notify_one()
        {
            std::unique_lock<std::mutex> lck(mutex_flag);
            flag_cond.notify_one();
        }
        void wait_for_ms(int milliseconds)
        {
            std::unique_lock<std::mutex> lck(mutex_flag);
            flag_cond.wait_for(lck,std::chrono::milliseconds(milliseconds));
        }
    private:
        std::mutex mutex_flag;                   //互斥
        std::condition_variable flag_cond;       //信号量
    };
    
    MutexVariable::MutexVariable()
    {
    }
    
    MutexVariable::~MutexVariable()
    {
    }
    
    class Perf_single{
    public:
        /**
         * @brief Constructor
         * 
         * @param bmodel_path        Path to bmodel
         * @param tpu_id             ID of TPU.
         * @param max_que_size       max queue size.
         * @param thread_idx         Thread Index.
        */
        Perf_single(const std::string& bmodel_path, int tpu_id, int max_que_size, int thread_idx=0, IOMode mode=SYSO, bool free_input=false);

        ~Perf_single();

         /**
         * @brief Push Tensor  
         * 
         * @param tensor_index  index number of the Tensor map.
         * @param input_tensors Input tensor map,batch size must 1
         * @return 0 for success and other for failure 
         */
        int PushTensor(int tensor_index, std::map<std::string, Tensor*>& input_tensors);

        /**
         * @brief Stop Push Tensor map
         */
        void SetEnd();

        /**
         * @brief Get the Data object
         * 
         * @param tensor_index             Original Tensor Index
         * @param tensor_out               Output Tensor map
         * @return 0 for success and other for failure
         */
        int GetResult(int& tensor_index, std::map<std::string, Tensor*>& tensor_out);

        void set_print_flag(bool print_flag){
            print_flag_ = print_flag;
        }


    private:
        sail::Engine* engine_;

        bool input_dev;         //输入内存是否位于设备内存
        bool output_dev;        //输出内存是否位于设备内存

        int tpu_id_;
        int thread_idx_;
        std::string input_name_;
        std::string graph_name_;

        int net_width_;
        int net_height_;
        int batch_size_;

        int max_queue_size_;

        bm_data_type_t input_dtype_;

        std::vector<std::string>        output_name_;
        std::vector<std::vector<int>>   output_shape_;
        std::vector<bm_data_type_t>     output_dtype_;
        std::map<std::string, float>    output_scale_;

        bool print_flag_;

        std::deque<int>  tensor_index_que;          //输入数据的索引队列
        std::mutex mutex_index;                     //tensor_index_que互斥
        
        TensorMapQueue OstTensor_queue;             //原始Tensor队列
        MutexVariable ostdata_cond;                 //OstTensor_queue有数据被取走的信号量

        TensorMapQueue InferenceTensor_queue;      //推理输入Tensor队列
        MutexVariable inferencedata_cond;          //InferenceTensor_queue有数据被取走的信号量

        TensorMapQueue OutputTensor_queue;         //推理输出Tensor队列
        MutexVariable outdata_cond;                //OutputTensor_queue有数据被取走的信号量

        TensorMapQueue OutputTensor_result;         //最终输出Tensor队列
        MutexVariable outresult_cond;               //OutputTensor_result有数据被取走的信号量

        Perf_Mutex_Flag stop_thread_flag;           //停止所有线程的标志位,(强制退出)
        Perf_Mutex_Flag stop_push_flag;             //停止push数据的标志位,(等待所有数据处理完毕再推出)

        void split_d2s_thread();                    //Tensor拆分及d2s线程
        Perf_Mutex_Flag split_d2s_exit_flag;        //Tensor拆分及d2s线程已经退出的标志位

        void inference_thread();                     //推理线程
        Perf_Mutex_Flag inference_exit_flag;        //推理线程已经推出的标志位

        void mergetensor_thread();                  //凑batch线程
        Perf_Mutex_Flag mergetensor_exit_flag;      //推理线程已经推出的标志位

        bool flag_thread;                           //线程是否已经开启的标志位

        bool free_input_;                     //是否立即释放输入的内存

    private:
        void push_data_index(int tensor_index);      //放入队列中一个数据编号        

        int get_data_index();                        //从队列中获取一个数据编号

        bool check_batchsize(std::map<std::string, Tensor*>& input_tensors);   //检查batchsize是否为1
    friend class Perf;

    };

    Perf_single::Perf_single(const std::string& bmodel_name, int tpu_id, int max_que_size,int thread_idx, IOMode mode, bool free_input)
    :engine_(NULL),tpu_id_(tpu_id),max_queue_size_(max_que_size),print_flag_(false), flag_thread(false),thread_idx_(thread_idx),free_input_(free_input)
    {
        split_d2s_exit_flag.set_flag(true);
        inference_exit_flag.set_flag(true);
        mergetensor_exit_flag.set_flag(true);
        engine_ = new sail::Engine(bmodel_name, tpu_id, DEVIO);
        if (engine_ == NULL){
            SPDLOG_INFO("Create Engine failed,bmode_name:{}, tpu:{}",bmodel_name,tpu_id);
            exit(1);
        }
        graph_name_ = engine_->get_graph_names()[0];
        input_name_ = engine_->get_input_names(graph_name_)[0];
        output_name_ = engine_->get_output_names(graph_name_);

        input_dtype_ = engine_->get_input_dtype(graph_name_,input_name_);
        std::vector<int> input_shape = engine_->get_input_shape(graph_name_,input_name_);

        for (int i = 0; i< output_name_.size(); ++i) {
            output_dtype_.push_back(engine_->get_output_dtype(graph_name_,output_name_[i]));
            output_shape_.push_back(engine_->get_output_shape(graph_name_,output_name_[i]));
            float scale_temp = engine_->get_output_scale(graph_name_,output_name_[i]);
            output_scale_.insert(std::pair<std::string, float>(output_name_[i],scale_temp));
        }

        batch_size_ = input_shape[0];
        net_height_ = input_shape[2];
        net_width_ = input_shape[3];

        input_dev = true;
        output_dev = true;

        switch (mode){
            case SYSI:
                input_dev = false;
            break;
                case SYSO:
                output_dev = false;
                break;
            case SYSIO:
                input_dev = false;
                output_dev = false;
                break;
            default:
                break;
        }

        max_queue_size_ = max_queue_size_ > batch_size_ ? max_queue_size_ : batch_size_;
        stop_thread_flag.set_flag(false);
        stop_push_flag.set_flag(false);

    }

    Perf_single::~Perf_single(){
        SPDLOG_INFO("Perf_single with dev:{},thread:{},Set Stop Flag!",tpu_id_,thread_idx_);
        stop_thread_flag.set_flag(true);
         while(true){
            if(!split_d2s_exit_flag.get_flag()){
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            if(!inference_exit_flag.get_flag()){
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            if(!mergetensor_exit_flag.get_flag()){
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            break;
        }
        SPDLOG_INFO("All Perf_single Thread Finshed, DEV:[{}],Thread[{}]!",tpu_id_,thread_idx_);
    }
    void Perf_single::SetEnd()
    {
        stop_push_flag.set_flag(true);
    }


    void Perf_single::split_d2s_thread()
    {
        SPDLOG_INFO("Batch split and D2S Thread Starting with DEV:[{}],Thread[{}]!",tpu_id_,thread_idx_);
        split_d2s_exit_flag.set_flag(false);
        sail::Handle handle(tpu_id_);
        while (true)        {
            if(stop_thread_flag.get_flag()){
                break;
            }
            //最终结果的队列如果是满的，阻塞住拆分batch并d2s线程，也不再从推理结果队列取数据
            if(OutputTensor_result.get_queue_size() >= max_queue_size_){
                double time_start = get_current_time_us();
                outresult_cond.wait_for_ms(20);
                if(print_flag_){
                    double time_use = (get_current_time_us() - time_start)/1000;
                    SPDLOG_INFO("OutputTensor_result queue is full, split_d2s_thread sleep {} ms, DEV:[{}],Thread[{}].",time_use,tpu_id_,thread_idx_);
                }
                continue;
            }

            //如果从推理结果队列取数据取不到数据，sleep 2ms再重试
            std::map<std::string, Tensor*> tensor_out;
            int ret = OutputTensor_queue.get_data(tensor_out);
            if(ret != 0){
                if(stop_push_flag.get_flag() && inference_exit_flag.get_flag()){      //已经停止送数据且推理线程已经退出，当前拆batch线程退出。
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            outdata_cond.notify_all();   

            //拆分batch并进行d2s
            for (size_t i = 0; i < batch_size_; i++)
            {
                std::map<std::string,sail::Tensor*> temp_map;
                auto iter = tensor_out.begin();
                while (iter != tensor_out.end()){
                    std::string name = iter->first;
                    sail::Tensor* ost_out =  iter->second;
                    std::vector<int> ost_shape = ost_out->shape(); 
                    std::vector<int> dst_shape;
                    dst_shape.push_back(1);
                    int batch_len = 1;
                    for (size_t j = 1; j < ost_shape.size(); j++)   {
                        dst_shape.push_back(ost_shape[j]);
                        batch_len = batch_len*ost_shape[j];
                    }

                    sail::Tensor* dst_out = new sail::Tensor(handle,dst_shape, ost_out->dtype(),true,false);
                    if(output_dev)
                        dst_out->sync_d2d(*ost_out, batch_len*i, 0, batch_len);
                    else
                        dst_out->sync_d2s(*ost_out, batch_len*i, 0, batch_len);
                    temp_map.insert(std::pair<std::string, sail::Tensor*>(name,dst_out));
                    iter++;
                }
                OutputTensor_result.push_data(temp_map);
            }

            //d2s之后删除原来的tensor
            auto iter = tensor_out.begin();
            while (iter != tensor_out.end()){
                sail::Tensor* ost_out =  iter->second;
                delete ost_out;
                iter++;
            }
            tensor_out.clear();
        }
        split_d2s_exit_flag.set_flag(true);
        SPDLOG_INFO("Batch split and D2S Thread End with DEV:[{}],Thread[{}]!",tpu_id_,thread_idx_);
    }


    void Perf_single::inference_thread()
    {
        SPDLOG_INFO("Inference Thread Starting with DEV:[{}],Thread[{}]!",tpu_id_,thread_idx_);
        inference_exit_flag.set_flag(false);
        sail::Handle handle(tpu_id_);
        while (true)        {
            if(stop_thread_flag.get_flag()){
                break;
            }
            //推理结果的队列如果是满的，阻塞住推理线程，也不再从原始数据队列取数据
            if(OutputTensor_queue.get_queue_size() >= max_queue_size_){
                double time_start = get_current_time_us();
                outdata_cond.wait_for_ms(20);
                if(print_flag_){
                    double time_use = (get_current_time_us() - time_start)/1000;
                    SPDLOG_INFO("OutputTensor_queue queue is full, inference_thread sleep {} ms, DEV:[{}],Thread[{}].",time_use, tpu_id_,thread_idx_);
                }
                continue;
            }

            //如果从推理输入队列取数据取不到数据，sleep 2ms再重试
            std::map<std::string, Tensor*> input_tensormap;
            int ret = InferenceTensor_queue.get_data(input_tensormap);
            if(ret != 0){
                if(stop_push_flag.get_flag() && mergetensor_exit_flag.get_flag()){      //已经停止送数据且凑batch的线程已经退出，当前推理线程退出。
                    break;
                }
                if(print_flag_){
                    SPDLOG_INFO("Inference Tensor queue is empty, inference_thread sleep 2 ms, DEV:[{}],Thread[{}].",tpu_id_,thread_idx_);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            inferencedata_cond.notify_all();    //有数据从推理线程取走

            std::map<std::string,sail::Tensor*> output_tensormap;
            for(int i = 0; i < output_name_.size();++i){
                sail::Tensor* output_tensor = new sail::Tensor(handle, output_shape_[i], output_dtype_[i], false, true);
                output_tensormap.insert(std::pair<std::string,sail::Tensor*>(output_name_[i], output_tensor));
            }
            engine_->process(graph_name_,input_tensormap,output_tensormap);         //推理
            //推理之后删除推理输入的tensor
            auto iter = input_tensormap.begin();
            while (iter != input_tensormap.end()){
                sail::Tensor* ost_in =  iter->second;
                delete ost_in;
                iter++;
            }
            input_tensormap.clear();

            OutputTensor_queue.push_data(output_tensormap);                         //推理结果送入队列
        }
        inference_exit_flag.set_flag(true);
        SPDLOG_INFO("Inference Thread End with DEV:[{}],Thread[{}]!",tpu_id_,thread_idx_);
    }

    void Perf_single::mergetensor_thread()
    {
        SPDLOG_INFO("Merge Tensor Thread Starting with DEV:[{}],Thread[{}]!",tpu_id_,thread_idx_);
        mergetensor_exit_flag.set_flag(false);
        sail::Handle handle(tpu_id_);
        bool end_flag_merge = false;    //退出标志位
        while (true)        {
            if(end_flag_merge){
                break;
            }
            if(stop_thread_flag.get_flag()){
                break;
            }
            //推理输入的队列如果是满的，阻塞住凑batch线程，也不再从输入原始数据队列取数据
            if(InferenceTensor_queue.get_queue_size() >= max_queue_size_){
                double time_start = get_current_time_us();
                inferencedata_cond.wait_for_ms(20);
                if(print_flag_){
                    double time_use = (get_current_time_us() - time_start)/1000;
                    SPDLOG_INFO("InferenceTensor_queue is full, mergetensor_thread sleep {} ms, DEV:[{}],Thread[{}].",time_use, tpu_id_,thread_idx_);
                }
                continue;
            }

            //取一个batch的数据
            std::map<std::string, Tensor*> input_tensormap;
            for(int i=0;i<batch_size_;++i){
                if(stop_thread_flag.get_flag()){
                    break;
                }
                //如果从推理输入队列取数据取不到数据，sleep 2ms再重试
                std::map<std::string, Tensor*> ost_tensormap;
                int ret = OstTensor_queue.get_data(ost_tensormap);
                if(ret != 0){
                    if(stop_push_flag.get_flag()){
                        end_flag_merge=true;
                        break;
                    }
                    if(print_flag_){
                        SPDLOG_INFO("OST Tensor queue is empty, mergetensor_thread sleep 2 ms, DEV:[{}],Thread[{}].",tpu_id_,thread_idx_);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    i--;
                    continue;
                }
                ostdata_cond.notify_all();    //有数据从推理线程取走
                auto iter = ost_tensormap.begin();
                while (iter != ost_tensormap.end()){
                    std::string tensor_name = iter->first;
                    sail::Tensor* ost_in =  iter->second;

                    std::vector<int> ost_shape = ost_in->shape(); 
                    std::vector<int> dst_shape;
                    dst_shape.push_back(batch_size_);
                    int batch_len = 1;
                    for (size_t j = 1; j < ost_shape.size(); j++)   {
                        dst_shape.push_back(ost_shape[j]);
                        batch_len = batch_len*ost_shape[j];
                    }
                    auto iter_in = input_tensormap.find(tensor_name);
                    if(iter_in == input_tensormap.end()){
                        sail::Tensor* input_tensor = new sail::Tensor(handle, dst_shape, ost_in->dtype(), false, true);
                        if(input_dev)
                            input_tensor->sync_d2d(*ost_in, 0, 0, batch_len);
                        else
                            input_tensor->sync_s2d(*ost_in, 0, 0, batch_len);
                        input_tensormap.insert(std::pair<std::string,sail::Tensor*>(tensor_name, input_tensor));
                    }else{
                        sail::Tensor* input_tensor =  iter_in->second;
                        if(input_dev)
                            input_tensor->sync_d2d(*ost_in, 0, batch_len*i, batch_len);
                        else
                            input_tensor->sync_s2d(*ost_in, 0, batch_len*i, batch_len);
                    }
                    if(free_input_){
                        delete ost_in;
                    }
                    iter++;
                }
            }
            if(!end_flag_merge){
                InferenceTensor_queue.push_data(input_tensormap);
            }
        }
        mergetensor_exit_flag.set_flag(true);
        SPDLOG_INFO("Merge Tensor Thread End with DEV:[{}],Thread[{}]!",tpu_id_,thread_idx_);
    }

    int Perf_single::PushTensor(int tensor_index, std::map<std::string, Tensor*>& input_tensors)
    {
        if(!flag_thread){
            std::thread merger_tensor_thread = std::thread(&Perf_single::mergetensor_thread,this);
            merger_tensor_thread.detach();
            std::thread inference_thread = std::thread(&Perf_single::inference_thread,this);
            inference_thread.detach();
            std::thread split_tensor_thread = std::thread(&Perf_single::split_d2s_thread,this);
            split_tensor_thread.detach();
            flag_thread = true;
        }
        if(!check_batchsize(input_tensors)){
            return 1;
        }
        int ret = 0;
        while (true)        {
            if(stop_thread_flag.get_flag()){
                ret = 1;
                break;
            }
            //推理结果的队列如果是满的，阻塞住推理线程，也不再从原始数据队列取数据
            if(OstTensor_queue.get_queue_size() >= max_queue_size_){
                double time_start = get_current_time_us();
                ostdata_cond.wait_for_ms(20);
                if(print_flag_){
                    double time_use = (get_current_time_us() - time_start)/1000;
                    SPDLOG_INFO("OstTensor_queue is full, PushTensor sleep {} ms, DEV:[{}],Thread[{}], max_queue_size_:[{}], current size:[{}].",time_use, tpu_id_,thread_idx_,max_queue_size_,OstTensor_queue.get_queue_size());
                }
                continue;
            }
            OstTensor_queue.push_data(input_tensors);
            push_data_index(tensor_index);
            break;
        }
        return ret;


    }

    void Perf_single::push_data_index(int tensor_index)
    {
        std::lock_guard<std::mutex> lock_temp(mutex_index);
        tensor_index_que.push_back(tensor_index);
    }       

    int Perf_single::get_data_index()
    {
        std::lock_guard<std::mutex> lock_temp(mutex_index);
        int index = tensor_index_que.front();
        tensor_index_que.pop_front();
        return index;
    }

    bool Perf_single::check_batchsize(std::map<std::string, Tensor*>& input_tensors)
    {
        auto iter = input_tensors.begin();
        if(iter == input_tensors.end()){
            return false;
        }
        while (iter != input_tensors.end()){
            sail::Tensor* ost_in =  iter->second;
            if(ost_in->shape()[0] != 1){
                return false;
            }
            iter++;
        }
        return true;
    }

    int Perf_single::GetResult(int& tensor_index, std::map<std::string, Tensor*>& tensor_out){
        int ret = OutputTensor_result.get_data(tensor_out);
        if(ret != 0){           
            if(stop_push_flag.get_flag() && split_d2s_exit_flag.get_flag()){      //已经停止送数据且拆batch的线程已经退出
                return 2;   //取不到数据，且已经停止了送数据
            }
            return 1;       //取不到数据，且没有退出
        }
        outresult_cond.notify_all();  
        tensor_index = get_data_index();
        return 0;
    }

    struct Perf_Key
    {
    Perf_Key(): tpu_id(-1),thread_id(0) { }
    Perf_Key(int m, int n) : tpu_id(m),thread_id(n) { }
        int tpu_id;
        int thread_id;
    };

    inline bool operator<(Perf_Key const& left, Perf_Key const& right){
        if(left.tpu_id < right.tpu_id) {return true;}
        if(left.tpu_id > right.tpu_id) {return false;}
        return left.thread_id < right.thread_id;
    }
    

    class Perf::Perf_CC{
    public:
        Perf_CC(const std::string& bmodel_path, std::vector<int> tpu_id, int max_que_size, IOMode mode, int thread_count, bool free_input);
        ~Perf_CC();

        int PushTensor(int tensor_index, std::map<std::string, Tensor*>& input_tensors);

        void SetEnd();

        Perf_single* get_next_perf();       //获取下一个perf

    private:
        std::map<Perf_Key, Perf_single*> perf_singles;
        std::map<int, int> thread_indexs;       //当前push数据对应的线程map
        int offset_perf;            //第几个Perf_single
        int perf_single_size;       //perf_singles的数量
        int threads;                //每个芯片上运行的线程数

    friend class Perf;
    };

    Perf::Perf_CC::Perf_CC(const std::string& bmodel_path, std::vector<int> tpu_ids, int max_que_size, IOMode mode, int thread_count, bool free_input) 
    :perf_single_size(0),offset_perf(0),threads(thread_count)
    {
        for(int i=0;i<tpu_ids.size();++i){
            auto iter = thread_indexs.find(tpu_ids[i]);
            if(iter == thread_indexs.end()){
                for (size_t thread_idx = 0; thread_idx < thread_count; thread_idx++)
                {
                    Perf_single* perf_si = new Perf_single(bmodel_path, tpu_ids[i], max_que_size, thread_idx, mode, free_input);
                    perf_singles.insert(std::pair<Perf_Key,Perf_single*>(Perf_Key(tpu_ids[i],thread_idx), perf_si));
                    perf_single_size++;
                }
                thread_indexs[tpu_ids[i]]=0;
            }
        }
    }

    Perf::Perf_CC::~Perf_CC(){
        auto iter = perf_singles.begin();
        while(iter != perf_singles.end()){
            Perf_single* perf_si = iter->second;
            delete perf_si;
            iter++;
        }
        perf_singles.clear();
    }

    Perf_single* Perf::Perf_CC::get_next_perf()
    {
        if(offset_perf >= perf_single_size){
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            offset_perf = 0;
        }
        auto iter_temp = perf_singles.begin();
        for (size_t i = 0; i < offset_perf; i++)  {
            iter_temp++;
        }
        
        Perf_single* perf_s = iter_temp->second;
        offset_perf++;
        return perf_s;
    }


    int Perf::Perf_CC::PushTensor(int tensor_index, std::map<std::string, Tensor*>& input_tensors)
    {
        auto iter = input_tensors.begin();
        if(iter == input_tensors.end()){
            SPDLOG_ERROR("PushTensor Failed! Input Data is empty!");
            return 2;
        }
        int dev_id = iter->second->get_handle().get_device_id();

        int thread_idx = thread_indexs[dev_id];
        auto iter_perf = perf_singles.find(Perf_Key(dev_id,thread_idx));
        if(iter_perf == perf_singles.end()){
            SPDLOG_ERROR("PushTensor Failed! Can not find Engine with dev: {}!", dev_id);
            return 3;
        }
        thread_idx++;
        if (thread_idx >= threads){
            thread_idx = 0;
        }
        thread_indexs[dev_id]=thread_idx;
        return iter_perf->second->PushTensor(tensor_index, input_tensors);
    }

    void Perf::Perf_CC::SetEnd()
    {
        auto iter = perf_singles.begin();
        while(iter != perf_singles.end()){
            iter->second->SetEnd();
            iter++;
        }
    }


    Perf::Perf(const std::string& bmodel_path, std::vector<int> tpu_ids, int max_que_size, IOMode mode, int thread_count, bool free_input)
    :_impl(new Perf_CC(bmodel_path,tpu_ids,max_que_size,mode,thread_count,free_input))
    { }

    Perf::~Perf()
    {
        delete _impl;
    }

    int Perf::PushTensor(int tensor_index, std::vector<TensorPTRWithName>& input_tensors)
    {
        std::map<std::string, Tensor*> input_ts;
        for (int i=0;i<input_tensors.size();++i){
            input_ts[input_tensors[i].name]=input_tensors[i].data;
        }
        return _impl->PushTensor(tensor_index,input_ts);
    }

    void Perf::SetEnd()
    {
        return _impl->SetEnd();
    }

    std::tuple<int, std::vector<TensorPTRWithName>> Perf::GetResult()
    {
        int tensor_index = -1;
        std::map<std::string, Tensor*> tensor_out;
        std::vector<TensorPTRWithName> tensor_result;
        while(true){
            Perf_single* perf_s = _impl->get_next_perf();
            int ret = perf_s->GetResult(tensor_index, tensor_out);
            if(ret == 2){
                return std::move(std::make_tuple(tensor_index,tensor_result));
            }else if(ret == 1){
                continue;
            }
            break;
        }
        auto iter = tensor_out.begin();
        while(iter != tensor_out.end()){
            TensorPTRWithName data_s(iter->first,iter->second);
            tensor_result.push_back(std::move(data_s));
            iter++;
        }
        return std::move(std::make_tuple(tensor_index,tensor_result));
    }

    std::vector<std::string> Perf::get_graph_names()
    {
        auto iter = _impl->perf_singles.begin();
        if(iter == _impl->perf_singles.end()){
            SPDLOG_ERROR("Can not find any engine!");
            exit(1);
        }
        if(iter->second == NULL){
            SPDLOG_ERROR("Can not find any engine!");
            exit(1);
        }
        return iter->second->engine_->get_graph_names();
    }

    std::vector<std::string> Perf::get_input_names(const std::string& graph_name)
    {
        auto iter = _impl->perf_singles.begin();
        if(iter == _impl->perf_singles.end()){
            SPDLOG_ERROR("Can not find any engine!");
            exit(1);
        }
        if(iter->second == NULL){
            SPDLOG_ERROR("Can not find any engine!");
            exit(1);
        }
        return iter->second->engine_->get_input_names(graph_name);
    }

    std::vector<std::string> Perf::get_output_names(const std::string& graph_name)
    {
        auto iter = _impl->perf_singles.begin();
        if(iter == _impl->perf_singles.end()){
            SPDLOG_ERROR("Can not find any engine!");
            exit(1);
        }
        if(iter->second == NULL){
            SPDLOG_ERROR("Can not find any engine!");
            exit(1);
        }
        return iter->second->engine_->get_output_names(graph_name);
    }

    std::vector<int> Perf::get_input_shape(const std::string& graph_name, const std::string& tensor_name)
    {
        auto iter = _impl->perf_singles.begin();
        if(iter == _impl->perf_singles.end()){
            SPDLOG_ERROR("Can not find any engine!");
            exit(1);
        }
        if(iter->second == NULL){
            SPDLOG_ERROR("Can not find any engine!");
            exit(1);
        }
        return iter->second->engine_->get_input_shape(graph_name,tensor_name);
    }

    std::vector<int> Perf::get_output_shape(const std::string& graph_name, const std::string& tensor_name)
    {
        auto iter = _impl->perf_singles.begin();
        if(iter == _impl->perf_singles.end()){
            SPDLOG_ERROR("Can not find any engine!");
            exit(1);
        }
        if(iter->second == NULL){
            SPDLOG_ERROR("Can not find any engine!");
            exit(1);
        }
        return iter->second->engine_->get_output_shape(graph_name,tensor_name);

    }

    float Perf::get_input_scale(const std::string& graph_name, const std::string& tensor_name)
    {
        auto iter = _impl->perf_singles.begin();
        if(iter == _impl->perf_singles.end()){
            SPDLOG_ERROR("Can not find any engine!");
            exit(1);
        }
        if(iter->second == NULL){
            SPDLOG_ERROR("Can not find any engine!");
            exit(1);
        }
        return iter->second->engine_->get_input_scale(graph_name,tensor_name);
    }

    bm_data_type_t Perf::get_input_dtype(const std::string& graph_name, const std::string& tensor_name)
    {
        auto iter = _impl->perf_singles.begin();
        if(iter == _impl->perf_singles.end()){
            SPDLOG_ERROR("Can not find any engine!");
            exit(1);
        }
        if(iter->second == NULL){
            SPDLOG_ERROR("Can not find any engine!");
            exit(1);
        }
        return iter->second->engine_->get_input_dtype(graph_name,tensor_name);
    }

}

