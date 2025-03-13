#include <algokit.h>
#include <mutex>
#include <queue>
#include <spdlog/spdlog.h> 
#include <graph.h>
#include <algorithm>

#include "internal.h"

#ifdef PYTHON
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eval.h>
using namespace pybind11::literals;
#endif

using namespace std;

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#include "deepsort/tracker.h"
#include "deepsort/track.h"
#include "deepsort/model.h"
#include "bytetrack/bytetrack.h"
#include "sort/sort_tracker.h"
#include "sort/sort_track.h"
using namespace cv;
#endif

namespace sail {
    float sail_algo_overlap(float x1, float w1, float x2, float w2)
    {
        float l1 = x1;
        float l2 = x2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1;
        float r2 = x2 + w2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }

float sail_algo_box_intersection(DeteObjRect a, DeteObjRect b)
{
	float w = sail_algo_overlap(a.left, a.width, b.left, b.width);
	float h = sail_algo_overlap(a.top, a.height, b.top, b.height);
	if (w < 0 || h < 0) return 0;
	float area = w*h;
	return area;
}

float sail_algo_box_union(DeteObjRect a, DeteObjRect b)
{
	float i = sail_algo_box_intersection(a, b);
	float u = a.width*a.height + b.width*b.height - i;
	return u;
}

float sail_algo_box_iou(DeteObjRect a, DeteObjRect b)
{
	return sail_algo_box_intersection(a, b) / sail_algo_box_union(a, b);
}

static bool sail_algo_sort_ObjRect(DeteObjRect a, DeteObjRect b)
{
    return a.score > b.score;
}

static void sail_algo_nms_sorted_bboxes(std::vector<DeteObjRect>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    // sorted 
    std::sort(objects.begin(),objects.end(),sail_algo_sort_ObjRect);
    const int n = objects.size();

    for (int i = 0; i < n; i++)    {
        const DeteObjRect& a = objects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const DeteObjRect& b = objects[picked[j]];

            float iou = sail_algo_box_iou(a, b);
            if (iou > nms_threshold){
                keep = 0;
                break;
            }
                
        }
        if (keep)
            picked.push_back(i);
    }
}

float sigmoid(float x) {
    return 1.0 / (1 + expf(-x));
}

#ifdef USE_OPENCV
class algo_yolov5_post_1output::algo_yolov5_post_1output_cc{
public:
    explicit algo_yolov5_post_1output_cc(const std::vector<int>& shape, int network_w, int network_h, int max_queue_size, bool input_use_multiclass_nms, bool agnostic);
    ~algo_yolov5_post_1output_cc();

    int push_data(std::vector<int> channel_idx, 
                    std::vector<int> image_idx, 
                    sail::Tensor* input_data, 
                    std::vector<float> dete_threshold,
                    std::vector<float> nms_threshold,
                    std::vector<int> ost_w,
                    std::vector<int> ost_h,
                    std::vector<std::vector<int>> padding_attr);

    int push_data(std::vector<int> channel_idx, 
                    std::vector<int> image_idx, 
                    sail::Tensor* input_data, 
                    std::vector<std::vector<float>> dete_threshold,
                    std::vector<float> nms_threshold,
                    std::vector<int> ost_w,
                    std::vector<int> ost_h,
                    std::vector<std::vector<int>> padding_attr);

    int push_data(std::vector<int> channel_idx, 
                    std::vector<int> image_idx, 
                    float* input_data, 
                    std::vector<float> dete_threshold,
                    std::vector<float> nms_threshold,
                    std::vector<int> ost_w,
                    std::vector<int> ost_h,
                    std::vector<std::vector<int>> padding_attr);

    std::tuple<std::vector<DeteObjRect>,int,int> get_result();

    int get_buffer_size(){
        return buffer_size_;
    }

private:
    void post_thread();

    sail::Tensor* get_data(std::vector<int>& channel_idx, 
                std::vector<int>& image_idx, 
                std::vector<std::vector<float>> &dete_threshold, 
                std::vector<float> &nms_threshold,
                std::vector<int>& ost_w, 
                std::vector<int>& ost_h, 
                std::vector<int>& padding_left, 
                std::vector<int>& padding_top, 
                std::vector<int>& padding_width, 
                std::vector<int>& padding_height);

    void set_stop_flag(bool flag);  //设置线程退出的标志位

    bool get_stop_flag();           //获取线程退出的标志位

    void set_thread_exit();         //设置线程已经退出         

    void wait_thread_exit();        //等待线程退出

    void notify_data_push();        //

    void notify_result_push();       

    void notify_result_pop();        

    int get_result_once(std::vector<DeteObjRect>& result, int &channel_idx,int &image_idx);     //get detect only once, return 0 for success

    int push_result(std::vector<DeteObjRect> result, int channel_idx,int image_idx);            

    std::vector<int> data_shape_;   //
    int classes_;                   //
    int max_queue_size_;            //
    int buffer_size_;               //

    std::queue<int> channels_queue_;        //
    std::queue<int> image_idx_queue_;       //
    std::queue<sail::Tensor*> tensor_in_queue_;   //
    std::queue<std::vector<float>> dete_threshold_queue;   //

    std::queue<float> nms_threshold_queue;    //

    std::queue<int> ost_w_queue;            //
    std::queue<int> ost_h_queue;            //
    std::queue<int> padding_left_queue;     //
    std::queue<int> padding_top_queue;      //
    std::queue<int> padding_width_queue;    //
    std::queue<int> padding_height_queue;   //

    std::mutex mutex_data;

    std::queue<int> out_channels_queue_;                        //
    std::queue<int> out_image_idx_queue_;                       //
    std::queue<std::vector<DeteObjRect>> dete_result_queue_;    //
    std::mutex mutex_result;

    bool stop_thread_flag;      //线程退出的标志位
    std::mutex mutex_stop_;     //线程退出互斥锁
    std::condition_variable exit_cond;  //线程已经退出的信号量
    std::mutex mutex_exit_;             //线程已经退出互斥锁
    bool exit_thread_flag;      //线程已经退出的标志位

    std::condition_variable result_push_cond;       //后处理线程有结果送入结果队列的信号
    std::mutex mutex_result_push;

    std::condition_variable result_pop_cond;       //后处理线程有结果送入结果队列的信号
    std::mutex mutex_result_pop;

    std::condition_variable pushdata_flag_cond;     //有数据push进来的信号
    std::mutex mutex_pushdata_flag;

    bool post_thread_run;

    int network_width; //
    int network_height; //
    int batch_size; //

    // nms params 
    bool input_use_multiclass_nms;
    bool agnostic;
};

algo_yolov5_post_1output::algo_yolov5_post_1output_cc::algo_yolov5_post_1output_cc(
    const std::vector<int>& shape,
    int network_w,
    int network_h,
    int max_queue_size,
    bool input_use_multiclass_nms, 
    bool agnostic)
:max_queue_size_(max_queue_size),post_thread_run(false),stop_thread_flag(false),exit_thread_flag(true),
network_width(network_w),network_height(network_h)
{
    if (shape.size() != 3){
        SPDLOG_ERROR("ERROR DIMS, 3 vs. {}!",shape.size());
        throw SailRuntimeError("invalid argument");
    }
    buffer_size_ = 1;
    for(int i=0;i<shape.size();++i)    {
        data_shape_.push_back(shape[i]);
    }
    for(int i=1;i<shape.size();++i)    {
        buffer_size_ = buffer_size_ * shape[i];
    }
    classes_ = data_shape_[2]-5;
    batch_size = shape[0];

    // nms paras init
    this->input_use_multiclass_nms = input_use_multiclass_nms;
    this->agnostic = agnostic;
}

algo_yolov5_post_1output::algo_yolov5_post_1output_cc::~algo_yolov5_post_1output_cc()
{
    set_stop_flag(true);
    SPDLOG_INFO("Start Set Thread Exit Flag!");
    set_thread_exit();
    SPDLOG_INFO("End Set Thread Exit Flag, Waiting Thread Exit....");
    wait_thread_exit();
}

int algo_yolov5_post_1output::algo_yolov5_post_1output_cc::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        sail::Tensor* input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr){
 #ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    if(channel_idx.size() != batch_size ||
        image_idx.size() != batch_size ||
        input_data->shape()[0] != batch_size ||
        dete_threshold.size() != batch_size ||
        padding_attr.size() != batch_size ||
        ost_w.size() != batch_size ||
        ost_h.size() != batch_size ||
        nms_threshold.size() != batch_size){
        SPDLOG_ERROR("Input shape {},{},{}",input_data->shape()[0],input_data->shape()[1],input_data->shape()[2]);
        SPDLOG_ERROR("Input batch size mismatch, {} vs. {}, {}, {}, {}, {}, {}, {}, {}",
            batch_size, channel_idx.size(), image_idx.size(), input_data->shape()[0], dete_threshold.size(),padding_attr.size(),
            ost_w.size(), ost_h.size(), nms_threshold.size());
        return SAIL_ALGO_ERROR_BATCHSIZE;
    }

    if(data_shape_.size() != input_data->shape().size()){
        SPDLOG_ERROR("The shape of the pushed data is incorrect!");
        return SAIL_ALGO_ERROR_SHAPES;
    }else{
        for(int i=0;i<data_shape_.size();i++){
            if(data_shape_[i]!=input_data->shape()[i]){
                SPDLOG_ERROR("The shape of the pushed data is incorrect!");
                return SAIL_ALGO_ERROR_SHAPES;
            }
            else
                continue;
        }
    }
    
    std::lock_guard<std::mutex> lock(mutex_data);
    if(tensor_in_queue_.size() >= max_queue_size_) {
        return SAIL_ALGO_BUFFER_FULL;
    }
    for(int i=0; i <batch_size;++i) {
        channels_queue_.push(channel_idx[i]);
        image_idx_queue_.push(image_idx[i]);
        dete_threshold_queue.push({dete_threshold[i]});
        nms_threshold_queue.push(nms_threshold[i]);
        ost_w_queue.push(ost_w[i]);                    //
        ost_h_queue.push(ost_h[i]);                    //
        padding_left_queue.push(padding_attr[i][0]);      //
        padding_top_queue.push(padding_attr[i][1]);        //
        padding_width_queue.push(padding_attr[i][2]);    //
        padding_height_queue.push(padding_attr[i][3]);  //
    }
    tensor_in_queue_.push(std::move(input_data));

    if(!post_thread_run){
        std::thread thread_post = std::thread(&algo_yolov5_post_1output_cc::post_thread,this);
        thread_post.detach();
        post_thread_run = true;
    }
    notify_data_push();
    return SAIL_ALGO_SUCCESS;
};

int algo_yolov5_post_1output::algo_yolov5_post_1output_cc::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        sail::Tensor* input_data, 
        std::vector<std::vector<float>> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr){
 #ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    if(channel_idx.size() != batch_size ||
        image_idx.size() != batch_size ||
        input_data->shape()[0] != batch_size ||
        dete_threshold.size() != batch_size ||
        padding_attr.size() != batch_size ||
        ost_w.size() != batch_size ||
        ost_h.size() != batch_size ||
        nms_threshold.size() != batch_size){
        SPDLOG_ERROR("Input shape {},{},{}",input_data->shape()[0],input_data->shape()[1],input_data->shape()[2]);
        SPDLOG_ERROR("Input batch size mismatch, {} vs. {}, {}, {}, {}, {}, {}, {}, {}",
            batch_size, channel_idx.size(), image_idx.size(), input_data->shape()[0], dete_threshold.size(),padding_attr.size(),
            ost_w.size(), ost_h.size(), nms_threshold.size());
        return SAIL_ALGO_ERROR_BATCHSIZE;
    }

    if(data_shape_.size() != input_data->shape().size()){
        SPDLOG_ERROR("The shape of the pushed data is incorrect!");
        return SAIL_ALGO_ERROR_SHAPES;
    }else{
        for(int i=0;i<data_shape_.size();i++){
            if(data_shape_[i]!=input_data->shape()[i]){
                SPDLOG_ERROR("The shape of the pushed data is incorrect!");
                return SAIL_ALGO_ERROR_SHAPES;
            }
            else
                continue;
        }
    }
    
    std::lock_guard<std::mutex> lock(mutex_data);
    if(tensor_in_queue_.size() >= max_queue_size_) {
        return SAIL_ALGO_BUFFER_FULL;
    }
    for(int i=0; i <batch_size;++i) {
        channels_queue_.push(channel_idx[i]);
        image_idx_queue_.push(image_idx[i]);
        dete_threshold_queue.push(dete_threshold[i]);
        nms_threshold_queue.push(nms_threshold[i]);
        ost_w_queue.push(ost_w[i]);                    //
        ost_h_queue.push(ost_h[i]);                    //
        padding_left_queue.push(padding_attr[i][0]);      //
        padding_top_queue.push(padding_attr[i][1]);        //
        padding_width_queue.push(padding_attr[i][2]);    //
        padding_height_queue.push(padding_attr[i][3]);  //
    }
    tensor_in_queue_.push(std::move(input_data));

    if(!post_thread_run){
        std::thread thread_post = std::thread(&algo_yolov5_post_1output_cc::post_thread,this);
        thread_post.detach();
        post_thread_run = true;
    }
    notify_data_push();
    return SAIL_ALGO_SUCCESS;
};

sail::Tensor* algo_yolov5_post_1output::algo_yolov5_post_1output_cc::get_data(
        std::vector<int>& channel_idx, 
        std::vector<int>& image_idx, 
        std::vector<std::vector<float>> &dete_threshold, 
        std::vector<float> &nms_threshold,
        std::vector<int>& ost_w, 
        std::vector<int>& ost_h, 
        std::vector<int>& padding_left, 
        std::vector<int>& padding_top, 
        std::vector<int>& padding_width, 
        std::vector<int>& padding_height){
    sail::Tensor* input_data = NULL; 
    std::lock_guard<std::mutex> lock(mutex_data);
    if(tensor_in_queue_.empty()) {
        return input_data;
    }
    channel_idx.clear();
    image_idx.clear(); 
    dete_threshold.clear();
    nms_threshold.clear();
    ost_w.clear();
    ost_h.clear();
    padding_left.clear();
    padding_top.clear();
    padding_width.clear();
    padding_height.clear();
    for(int i=0;i<batch_size;++i){
        channel_idx.push_back(channels_queue_.front());
        image_idx.push_back(image_idx_queue_.front());
        dete_threshold.push_back(dete_threshold_queue.front());
        nms_threshold.push_back(nms_threshold_queue.front());

        ost_w.push_back(ost_w_queue.front());                    //
        ost_h.push_back(ost_h_queue.front());                    //
        padding_left.push_back(padding_left_queue.front());      //
        padding_top.push_back(padding_top_queue.front());        //
        padding_width.push_back(padding_width_queue.front());    //
        padding_height.push_back(padding_height_queue.front());  //

        channels_queue_.pop();
        image_idx_queue_.pop();
        dete_threshold_queue.pop();
        nms_threshold_queue.pop();
        ost_w_queue.pop();                    //
        ost_h_queue.pop();                    //
        padding_left_queue.pop();      //
        padding_top_queue.pop();        //
        padding_width_queue.pop();    //
        padding_height_queue.pop();  // 
    }

    input_data = tensor_in_queue_.front();
    tensor_in_queue_.pop(); //

    return std::move(input_data);
}

int algo_yolov5_post_1output::algo_yolov5_post_1output_cc::get_result_once(std::vector<DeteObjRect>& result, int &channel_idx,int &image_idx)
{
    std::lock_guard<std::mutex> lock(mutex_result);
    if(dete_result_queue_.empty()){
        return 1;
    }
    channel_idx = out_channels_queue_.front();
    image_idx = out_image_idx_queue_.front();
    std::vector<DeteObjRect> temp = dete_result_queue_.front();
    for(int i=0;i<temp.size();++i){
        result.push_back(temp[i]);
    }
    out_channels_queue_.pop();
    out_image_idx_queue_.pop();
    dete_result_queue_.pop();

    notify_result_pop();    
    return 0;
}

int algo_yolov5_post_1output::algo_yolov5_post_1output_cc::push_result(std::vector<DeteObjRect> result, int channel_idx,int image_idx)
{
    std::lock_guard<std::mutex> lock(mutex_result);
    if(dete_result_queue_.size() >= max_queue_size_){
        return 1;
    }
    out_channels_queue_.push(channel_idx);
    out_image_idx_queue_.push(image_idx);
    dete_result_queue_.push(std::move(result));
    return 0;
}

void algo_yolov5_post_1output::algo_yolov5_post_1output_cc::post_thread()
{
    {        
        std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
        exit_thread_flag = false;
    }
    // SPDLOG_INFO("Create To YOLOV5 1 output Thread, pid:{}, tid:{} .",getpid(),gettid());
    int idx = 0;
    while(true) {
        if(get_stop_flag()){
            break;
        }  
        std::vector<int> channel_idxs;
        std::vector<int> image_idxs;
        std::vector<std::vector<float>> dete_thresholds;
        std::vector<float> nms_thresholds;
        std::vector<int> ost_ws;
        std::vector<int> ost_hs;
        std::vector<int> padding_lefts;
        std::vector<int> padding_tops;
        std::vector<int> padding_widths;
        std::vector<int> padding_heights;

        sail::Tensor* in_data = get_data(channel_idxs, image_idxs, dete_thresholds, nms_thresholds,
            ost_ws, ost_hs, padding_lefts, padding_tops, padding_widths, padding_heights);
        if(in_data == NULL){
            std::unique_lock<std::mutex> lck(mutex_pushdata_flag);
            pushdata_flag_cond.wait_for(lck,std::chrono::milliseconds(5));
            // SPDLOG_INFO("Get Data failed, sleeping for 5 ms");
            continue;
        }
        float* data_ost = (float*)in_data->sys_data();
        const char *dump_flag = getenv("SAIL_SAVE_IO_TENSORS");
        if (dump_flag != nullptr && 0 == strcmp(dump_flag, "1")){
            double id_save = get_current_time_us();
            char save_name_temp[256]={0};
            sprintf(save_name_temp,"%s_%.0f.dat","algokit_yolov5_post_1out",id_save);
            dump_float32_data((const char* )save_name_temp, data_ost, buffer_size_*batch_size, data_shape_[1], data_shape_[2]);
        }

        // nms init 
        int max_wh = 7680;
        for (int index = 0; index < batch_size; ++index){
            int channel_idx = channel_idxs[index];
            int image_idx = image_idxs[index];
            std::vector<float> dete_threshold = dete_thresholds[index];
            float nms_threshold = nms_thresholds[index];
            int ost_w = ost_ws[index];
            int ost_h = ost_hs[index];
            int padding_left = padding_lefts[index];
            int padding_top = padding_tops[index];
            int padding_width = padding_widths[index];
            int padding_height = padding_heights[index];

            double time_start = get_current_time_us();
            float scale_w = (float)ost_w/padding_width;
            float scale_h = (float)ost_h/padding_height;
            float* data = data_ost + index*buffer_size_;
            float min_dete_threshold= *std::min_element(dete_threshold.begin(), dete_threshold.end());
            std::vector<DeteObjRect> dete_rects;

            for (int i = 0; i < data_shape_[1]; ++i){               
                float confidence = data[4];
                if (confidence >= min_dete_threshold) {
                    float *classes_scores = data + 5;
                    if(input_use_multiclass_nms){
                        if(classes_ != dete_threshold.size() && dete_threshold.size() != 1){
                            SPDLOG_ERROR("dete_threshold count Mismatch!");
                            set_thread_exit();
                            {
                                std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
                                exit_thread_flag = true;
                            }
                            return ;
                        }
                        for(int cls_id=0; cls_id<classes_; cls_id++){
                            float dete_threshold_=dete_threshold[cls_id<dete_threshold.size()?cls_id:0];
                            if(confidence * (*(classes_scores + cls_id)) > dete_threshold_){
                                DeteObjRect dete_rect;
                                dete_rect.score = confidence * (*(classes_scores + cls_id));
                                dete_rect.class_id = cls_id;
                                dete_rect.width = data[2];
                                dete_rect.height = data[3];

                                if(!agnostic){
                                    dete_rect.left = data[0] - 0.5 * data[2] + cls_id * max_wh;
                                    dete_rect.top = data[1] - 0.5 * data[3] +  + cls_id * max_wh;
                                }else{
                                    dete_rect.left = data[0] - 0.5 * data[2];
                                    dete_rect.top = data[1] - 0.5 * data[3];
                                }
                                dete_rect.right = dete_rect.left + dete_rect.width;
                                dete_rect.bottom = dete_rect.top + dete_rect.height;

                                dete_rects.push_back(dete_rect);

                            }
                        }
                    }else{
                        cv::Mat scores(1, classes_, CV_32FC1, classes_scores);
                        cv::Point class_id;
                        double max_class_score;
                        cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                        float dete_threshold_=dete_threshold[class_id.x<dete_threshold.size()?class_id.x:0];

                        DeteObjRect dete_rect;
                        if (confidence*max_class_score > dete_threshold_) {
                            dete_rect.score = confidence*max_class_score;
                            dete_rect.class_id = class_id.x;
                            dete_rect.width = data[2];
                            dete_rect.height = data[3];

                            if(!agnostic){
                                dete_rect.left = data[0] - 0.5 * data[2] + class_id.x * max_wh;
                                dete_rect.top = data[1] - 0.5 * data[3] +  + class_id.x * max_wh;
                            }else{
                                dete_rect.left = data[0] - 0.5 * data[2];
                                dete_rect.top = data[1] - 0.5 * data[3];
                            }
                            dete_rect.right = dete_rect.left + dete_rect.width;
                            dete_rect.bottom = dete_rect.top + dete_rect.height;

                            dete_rects.push_back(dete_rect);
                        }
                    }
                    
                }
                data += data_shape_[2];
            }
            std::vector<DeteObjRect> dect_result;
            std::vector<int> picked;
            double time_num_start = get_current_time_us();
            sail_algo_nms_sorted_bboxes(dete_rects, picked, nms_threshold);
            double nms_time_use = (get_current_time_us() - time_num_start)/1000;
            
            if(!agnostic){
                for(auto& box:dete_rects){
                    box.left -= box.class_id * max_wh;
                    box.top -= box.class_id * max_wh;
                    box.right = box.left + box.width;
                    box.bottom = box.top + box.height;
                }
            }
            for (size_t i = 0; i < picked.size(); i++)    {
                DeteObjRect dete_rect;
                dete_rect.left = (dete_rects[picked[i]].left - padding_left) * scale_w;
                dete_rect.top = (dete_rects[picked[i]].top - padding_top) * scale_h;
                dete_rect.left = dete_rect.left > 0 ? dete_rect.left : 0;
                dete_rect.top = dete_rect.top > 0 ? dete_rect.top : 0;

                dete_rect.right = (dete_rects[picked[i]].right - padding_left) * scale_w;
                dete_rect.bottom = (dete_rects[picked[i]].bottom - padding_top) * scale_h;
                dete_rect.right = dete_rect.right < ost_w ? dete_rect.right : ost_w-1;
                dete_rect.bottom = dete_rect.bottom < ost_h ? dete_rect.bottom : ost_h-1;

                dete_rect.width = dete_rect.right - dete_rect.left;
                dete_rect.height = dete_rect.bottom - dete_rect.top;

                // dete_rect.width = dete_rects[picked[i]].width * scale_w;
                // dete_rect.height = dete_rects[picked[i]].height * scale_h;

                // dete_rect.left = dete_rect.left > 0 ? dete_rect.left : 0;
                // dete_rect.top = dete_rect.top > 0 ? dete_rect.top : 0;
                // dete_rect.width = dete_rect.width < network_width ? dete_rect.width : network_width;
                // dete_rect.height = dete_rect.height < network_height ? dete_rect.height : network_height;

                // dete_rect.right = dete_rect.left + dete_rect.width < ost_w ? dete_rect.left + dete_rect.width : ost_w-1;
                // dete_rect.bottom = dete_rect.top + dete_rect.height < ost_h ? dete_rect.top + dete_rect.height : ost_h-1;
                

                dete_rect.score = dete_rects[picked[i]].score;
                dete_rect.class_id = dete_rects[picked[i]].class_id;

                dect_result.push_back(dete_rect);
            }
            double time_use = (get_current_time_us() - time_start)/1000;
            // SPDLOG_INFO("Yolov5 one output post process us {} ms, nms: {} ms. scale_w: {}, scale_h: {}",time_use, nms_time_use,scale_w,scale_h);  
            while (true) {
                int ret = push_result(dect_result, channel_idx, image_idx);
                if(ret == 0) {
                    notify_result_push();
                    break;
                }
                std::unique_lock<std::mutex> lck(mutex_result_pop);
                result_pop_cond.wait_for(lck,std::chrono::milliseconds(5));
                if(get_stop_flag()){
                    break;
                }  
            }
        }
        if(in_data){
            delete in_data;
        }
    }
    set_thread_exit();
    {
        std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
        exit_thread_flag = true;
    }
}

void algo_yolov5_post_1output::algo_yolov5_post_1output_cc::notify_data_push()
{
    std::unique_lock<std::mutex> lck(mutex_pushdata_flag);
    pushdata_flag_cond.notify_all();
}

void algo_yolov5_post_1output::algo_yolov5_post_1output_cc::notify_result_push()
{
    std::unique_lock<std::mutex> lck(mutex_result_push);
    result_push_cond.notify_all();
}

void algo_yolov5_post_1output::algo_yolov5_post_1output_cc::notify_result_pop()
{
    std::unique_lock<std::mutex> lck(mutex_result_pop);
    result_pop_cond.notify_all();
}

bool algo_yolov5_post_1output::algo_yolov5_post_1output_cc::get_stop_flag()
{
    std::lock_guard<std::mutex> lock(mutex_stop_);
    return stop_thread_flag;
}

void algo_yolov5_post_1output::algo_yolov5_post_1output_cc::set_stop_flag(bool flag)
{
    std::lock_guard<std::mutex> lock(mutex_stop_);
    stop_thread_flag = flag;
}

void algo_yolov5_post_1output::algo_yolov5_post_1output_cc::set_thread_exit()
{
    std::unique_lock<std::mutex> lck(mutex_exit_);
    exit_cond.notify_all();
}   

void algo_yolov5_post_1output::algo_yolov5_post_1output_cc::wait_thread_exit()
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

std::tuple<std::vector<DeteObjRect>,int,int> algo_yolov5_post_1output::algo_yolov5_post_1output_cc::get_result()
{
#ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    std::vector<DeteObjRect> results;
    int channel_idx = 0;
    int image_idx = 0;
    while(true){
        int ret = get_result_once(results, channel_idx, image_idx);
        if(ret == 0){
            break;
        }
        std::unique_lock<std::mutex> lck(mutex_result_push);
        result_push_cond.wait_for(lck,std::chrono::milliseconds(5));
        if(get_stop_flag()){
            break;
        } 
    }
    return std::make_tuple(std::move(results),channel_idx,image_idx);
}

algo_yolov5_post_1output::algo_yolov5_post_1output(const std::vector<int>& shape,int network_w, int network_h, int max_queue_size, bool input_use_multiclass_nms, bool agnostic)
:_impl (new algo_yolov5_post_1output_cc(shape,network_w,network_h,max_queue_size, input_use_multiclass_nms,agnostic))
{
}

algo_yolov5_post_1output::~algo_yolov5_post_1output()
{
    delete _impl;
}

#ifdef PYTHON
int algo_yolov5_post_1output::push_npy(int channel_idx, int image_idx, pybind11::array_t<float> ost_array, float dete_threshold, float nms_threshold,
        int ost_w, int ost_h, int padding_left, int padding_top, int padding_width, int padding_height)
{
    sail::Handle handle(0);
    std::vector<int> channel_idxs = {channel_idx};
    std::vector<int> image_idxs = {image_idx};
    std::vector<float> dete_thresholds = {dete_threshold};
    std::vector<float> nms_thresholds = {nms_threshold};
    std::vector<int> ost_ws = {ost_w};
    std::vector<int> ost_hs = {ost_h};
    std::vector<std::vector<int>> padding_attrs = {{padding_left,padding_top,padding_width,padding_height}};

    if (!pybind11::detail::check_flags(ost_array.ptr(), pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_)) {
        pybind11::module np = pybind11::module::import("numpy");        // like 'import numpy as np'
        pybind11::array_t<float> arr_c = np.attr("ascontiguousarray")(ost_array, "dtype"_a="float32");
        sail::Tensor* tensor_in = new sail::Tensor(handle, arr_c, true);
        return _impl->push_data(channel_idxs,image_idxs,tensor_in,dete_thresholds,nms_thresholds, ost_ws, ost_hs, padding_attrs);
    }else{
        sail::Tensor* tensor_in = new sail::Tensor(handle, ost_array, true);
        return _impl->push_data(channel_idxs,image_idxs,tensor_in,dete_thresholds,nms_thresholds, ost_ws, ost_hs, padding_attrs);
    }
}
#endif
    
int algo_yolov5_post_1output::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        TensorPTRWithName input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr)
{
    return _impl->push_data(channel_idx,image_idx,input_data.data,dete_threshold,nms_threshold, ost_w, ost_h, padding_attr);
}
int algo_yolov5_post_1output::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        TensorPTRWithName input_data, 
        std::vector<std::vector<float>> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr)
{
    return _impl->push_data(channel_idx,image_idx,input_data.data,dete_threshold,nms_threshold, ost_w, ost_h, padding_attr);
}
std::tuple<std::vector<DeteObjRect>,int,int> algo_yolov5_post_1output::get_result()
{
    return _impl->get_result();
}
std::tuple<std::vector<std::tuple<int, int, int, int ,int, float>>,int,int> algo_yolov5_post_1output::get_result_npy()
{
    auto result = _impl->get_result();
    std::vector<DeteObjRect> results = std::move(std::get<0>(result));
    int channel_idx = std::get<1>(result);
    int image_idx = std::get<2>(result);
    // SPDLOG_INFO("results num : {}",results.size());
    std::vector<std::tuple<int, int, int, int ,int, float>> objs;
    for (int i = 0; i < results.size(); ++i){
        int left_temp = results[i].left+0.5;
        int top_temp = results[i].top+0.5;
        int right_temp = results[i].right+0.5;
        int bottom_temp = results[i].bottom+0.5;
        int class_id_temp = results[i].class_id;
        float score_temp = results[i].score;

        objs.push_back(std::make_tuple(left_temp,
                                       top_temp,
                                       right_temp,
                                       bottom_temp,
                                       class_id_temp,
                                       score_temp));

    }
    return std::make_tuple(std::move(objs),channel_idx,image_idx);
}


class algo_yolov8_post_1output_async::algo_yolov8_post_1output_async_cc{
public:
    explicit algo_yolov8_post_1output_async_cc(const std::vector<int>& shape, int network_w, int network_h, int max_queue_size, bool input_use_multiclass_nms, bool agnostic);
    ~algo_yolov8_post_1output_async_cc();

    int push_data(std::vector<int> channel_idx, 
                    std::vector<int> image_idx, 
                    sail::Tensor* input_data, 
                    std::vector<float> dete_threshold,
                    std::vector<float> nms_threshold,
                    std::vector<int> ost_w,
                    std::vector<int> ost_h,
                    std::vector<std::vector<int>> padding_attr);

    int push_data(std::vector<int> channel_idx, 
                    std::vector<int> image_idx, 
                    float* input_data, 
                    std::vector<float> dete_threshold,
                    std::vector<float> nms_threshold,
                    std::vector<int> ost_w,
                    std::vector<int> ost_h,
                    std::vector<std::vector<int>> padding_attr);

    std::tuple<std::vector<DeteObjRect>,int,int> get_result();

    int get_buffer_size(){
        return buffer_size_;
    }

private:
    void post_thread();

    sail::Tensor* get_data(std::vector<int>& channel_idx, 
                std::vector<int>& image_idx, 
                std::vector<float> &dete_threshold, 
                std::vector<float> &nms_threshold,
                std::vector<int>& ost_w, 
                std::vector<int>& ost_h, 
                std::vector<int>& padding_left, 
                std::vector<int>& padding_top, 
                std::vector<int>& padding_width, 
                std::vector<int>& padding_height);

    void set_stop_flag(bool flag);  //设置线程退出的标志位

    bool get_stop_flag();           //获取线程退出的标志位

    void set_thread_exit();         //设置线程已经退出         

    void wait_thread_exit();        //等待线程退出

    void notify_data_push();        //

    void notify_result_push();       

    void notify_result_pop();        

    int get_result_once(std::vector<DeteObjRect>& result, int &channel_idx,int &image_idx);     //get detect only once, return 0 for success

    int push_result(std::vector<DeteObjRect> result, int channel_idx,int image_idx);            

    std::vector<int> data_shape_;   //
    int classes_;                   //
    int max_queue_size_;            //
    int buffer_size_;               //

    std::queue<int> channels_queue_;        //
    std::queue<int> image_idx_queue_;       //
    std::queue<sail::Tensor*> tensor_in_queue_;   //
    std::queue<float> dete_threshold_queue;   //
    std::queue<float> nms_threshold_queue;    //

    std::queue<int> ost_w_queue;            //
    std::queue<int> ost_h_queue;            //
    std::queue<int> padding_left_queue;     //
    std::queue<int> padding_top_queue;      //
    std::queue<int> padding_width_queue;    //
    std::queue<int> padding_height_queue;   //

    std::mutex mutex_data;

    std::queue<int> out_channels_queue_;                        //
    std::queue<int> out_image_idx_queue_;                       //
    std::queue<std::vector<DeteObjRect>> dete_result_queue_;    //
    std::mutex mutex_result;

    bool stop_thread_flag;      //线程退出的标志位
    std::mutex mutex_stop_;     //线程退出互斥锁
    std::condition_variable exit_cond;  //线程已经退出的信号量
    std::mutex mutex_exit_;             //线程已经退出互斥锁
    bool exit_thread_flag;      //线程已经退出的标志位

    std::condition_variable result_push_cond;       //后处理线程有结果送入结果队列的信号
    std::mutex mutex_result_push;

    std::condition_variable result_pop_cond;       //后处理线程有结果送入结果队列的信号
    std::mutex mutex_result_pop;

    std::condition_variable pushdata_flag_cond;     //有数据push进来的信号
    std::mutex mutex_pushdata_flag;

    bool post_thread_run;

    int network_width; //
    int network_height; //
    int batch_size; //

    // nms params 
    bool input_use_multiclass_nms;
    bool agnostic;
};

algo_yolov8_post_1output_async::algo_yolov8_post_1output_async_cc::algo_yolov8_post_1output_async_cc(
    const std::vector<int>& shape,
    int network_w,
    int network_h,
    int max_queue_size,
    bool input_use_multiclass_nms, 
    bool agnostic)
:max_queue_size_(max_queue_size),post_thread_run(false),stop_thread_flag(false),exit_thread_flag(true),
network_width(network_w),network_height(network_h)
{
    if (shape.size() != 3){
        SPDLOG_ERROR("ERROR DIMS, 3 vs. {}!",shape.size());
        throw SailRuntimeError("invalid argument");
    }
    buffer_size_ = 1;
    for(int i=0;i<shape.size();++i)    {
        data_shape_.push_back(shape[i]);
    }
    for(int i=1;i<shape.size();++i)    {
        buffer_size_ = buffer_size_ * shape[i];
    }
    classes_ = data_shape_[2]-5;
    batch_size = shape[0];

    // nms paras init
    this->input_use_multiclass_nms = input_use_multiclass_nms;
    this->agnostic = agnostic;
}

algo_yolov8_post_1output_async::algo_yolov8_post_1output_async_cc::~algo_yolov8_post_1output_async_cc()
{
    set_stop_flag(true);
    SPDLOG_INFO("Start Set Thread Exit Flag!");
    set_thread_exit();
    SPDLOG_INFO("End Set Thread Exit Flag, Waiting Thread Exit....");
    wait_thread_exit();
}

int algo_yolov8_post_1output_async::algo_yolov8_post_1output_async_cc::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        sail::Tensor* input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr){
 #ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    if(channel_idx.size() != batch_size ||
        image_idx.size() != batch_size ||
        input_data->shape()[0] != batch_size ||
        dete_threshold.size() != batch_size ||
        padding_attr.size() != batch_size ||
        ost_w.size() != batch_size ||
        ost_h.size() != batch_size ||
        nms_threshold.size() != batch_size){
        SPDLOG_ERROR("Input shape {},{},{}",input_data->shape()[0],input_data->shape()[1],input_data->shape()[2]);
        SPDLOG_ERROR("Input batch size mismatch, {} vs. {}, {}, {}, {}, {}, {}, {}, {}",
            batch_size, channel_idx.size(), image_idx.size(), input_data->shape()[0], dete_threshold.size(),padding_attr.size(),
            ost_w.size(), ost_h.size(), nms_threshold.size());
        return SAIL_ALGO_ERROR_BATCHSIZE;
    }

    if(data_shape_.size() != input_data->shape().size()){
        SPDLOG_ERROR("The shape of the pushed data is incorrect!");
        return SAIL_ALGO_ERROR_SHAPES;
    }else{
        for(int i=0;i<data_shape_.size();i++){
            if(data_shape_[i]!=input_data->shape()[i]){
                SPDLOG_ERROR("The shape of the pushed data is incorrect!");
                return SAIL_ALGO_ERROR_SHAPES;
            }
            else
                continue;
        }
    }
    
    std::lock_guard<std::mutex> lock(mutex_data);
    if(tensor_in_queue_.size() >= max_queue_size_) {
        return SAIL_ALGO_BUFFER_FULL;
    }
    for(int i=0; i <batch_size;++i) {
        channels_queue_.push(channel_idx[i]);
        image_idx_queue_.push(image_idx[i]);
        dete_threshold_queue.push(dete_threshold[i]);
        nms_threshold_queue.push(nms_threshold[i]);
        ost_w_queue.push(ost_w[i]);                    //
        ost_h_queue.push(ost_h[i]);                    //
        padding_left_queue.push(padding_attr[i][0]);      //
        padding_top_queue.push(padding_attr[i][1]);        //
        padding_width_queue.push(padding_attr[i][2]);    //
        padding_height_queue.push(padding_attr[i][3]);  //
    }
    tensor_in_queue_.push(std::move(input_data));

    if(!post_thread_run){
        std::thread thread_post = std::thread(&algo_yolov8_post_1output_async_cc::post_thread,this);
        thread_post.detach();
        post_thread_run = true;
    }
    notify_data_push();
    return SAIL_ALGO_SUCCESS;
};

sail::Tensor* algo_yolov8_post_1output_async::algo_yolov8_post_1output_async_cc::get_data(
        std::vector<int>& channel_idx, 
        std::vector<int>& image_idx, 
        std::vector<float> &dete_threshold, 
        std::vector<float> &nms_threshold,
        std::vector<int>& ost_w, 
        std::vector<int>& ost_h, 
        std::vector<int>& padding_left, 
        std::vector<int>& padding_top, 
        std::vector<int>& padding_width, 
        std::vector<int>& padding_height){
    sail::Tensor* input_data = NULL; 
    std::lock_guard<std::mutex> lock(mutex_data);
    if(tensor_in_queue_.empty()) {
        return input_data;
    }
    channel_idx.clear();
    image_idx.clear(); 
    dete_threshold.clear();
    nms_threshold.clear();
    ost_w.clear();
    ost_h.clear();
    padding_left.clear();
    padding_top.clear();
    padding_width.clear();
    padding_height.clear();
    for(int i=0;i<batch_size;++i){
        channel_idx.push_back(channels_queue_.front());
        image_idx.push_back(image_idx_queue_.front());
        dete_threshold.push_back(dete_threshold_queue.front());
        nms_threshold.push_back(nms_threshold_queue.front());

        ost_w.push_back(ost_w_queue.front());                    //
        ost_h.push_back(ost_h_queue.front());                    //
        padding_left.push_back(padding_left_queue.front());      //
        padding_top.push_back(padding_top_queue.front());        //
        padding_width.push_back(padding_width_queue.front());    //
        padding_height.push_back(padding_height_queue.front());  //

        channels_queue_.pop();
        image_idx_queue_.pop();
        dete_threshold_queue.pop();
        nms_threshold_queue.pop();
        ost_w_queue.pop();                    //
        ost_h_queue.pop();                    //
        padding_left_queue.pop();      //
        padding_top_queue.pop();        //
        padding_width_queue.pop();    //
        padding_height_queue.pop();  // 
    }

    input_data = tensor_in_queue_.front();
    tensor_in_queue_.pop(); //

    return std::move(input_data);
}

int algo_yolov8_post_1output_async::algo_yolov8_post_1output_async_cc::get_result_once(std::vector<DeteObjRect>& result, int &channel_idx,int &image_idx)
{
    std::lock_guard<std::mutex> lock(mutex_result);
    if(dete_result_queue_.empty()){
        return 1;
    }
    channel_idx = out_channels_queue_.front();
    image_idx = out_image_idx_queue_.front();
    std::vector<DeteObjRect> temp = dete_result_queue_.front();
    for(int i=0;i<temp.size();++i){
        result.push_back(temp[i]);
    }
    out_channels_queue_.pop();
    out_image_idx_queue_.pop();
    dete_result_queue_.pop();

    notify_result_pop();    
    return 0;
}

int algo_yolov8_post_1output_async::algo_yolov8_post_1output_async_cc::push_result(std::vector<DeteObjRect> result, int channel_idx,int image_idx)
{
    std::lock_guard<std::mutex> lock(mutex_result);
    if(dete_result_queue_.size() >= max_queue_size_){
        return 1;
    }
    out_channels_queue_.push(channel_idx);
    out_image_idx_queue_.push(image_idx);
    dete_result_queue_.push(std::move(result));
    return 0;
}

void algo_yolov8_post_1output_async::algo_yolov8_post_1output_async_cc::post_thread()
{
    {        
        std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
        exit_thread_flag = false;
    }
    // SPDLOG_INFO("Create To YOLOV8 1 output Thread, pid:{}, tid:{} .",getpid(),gettid());
    int idx = 0;
    while(true) {
        if(get_stop_flag()){
            break;
        }  
        std::vector<int> channel_idxs;
        std::vector<int> image_idxs;
        std::vector<float> dete_thresholds;
        std::vector<float> nms_thresholds;
        std::vector<int> ost_ws;
        std::vector<int> ost_hs;
        std::vector<int> padding_lefts;
        std::vector<int> padding_tops;
        std::vector<int> padding_widths;
        std::vector<int> padding_heights;

        sail::Tensor* in_data = get_data(channel_idxs, image_idxs, dete_thresholds, nms_thresholds,
            ost_ws, ost_hs, padding_lefts, padding_tops, padding_widths, padding_heights);
        if(in_data == NULL){
            std::unique_lock<std::mutex> lck(mutex_pushdata_flag);
            pushdata_flag_cond.wait_for(lck,std::chrono::milliseconds(5));
            // SPDLOG_INFO("Get Data failed, sleeping for 5 ms");
            continue;
        }
        float* data_ost = (float*)in_data->sys_data();
        const char *dump_flag = getenv("SAIL_SAVE_IO_TENSORS");
        if (dump_flag != nullptr && 0 == strcmp(dump_flag, "1")){
            double id_save = get_current_time_us();
            char save_name_temp[256]={0};
            sprintf(save_name_temp,"%s_%.0f.dat","algokit_yolov8_post_1out",id_save);
            dump_float32_data((const char* )save_name_temp, data_ost, buffer_size_*batch_size, data_shape_[1], data_shape_[2]);
        }

        // nms init 
        int max_wh = 7680;
        for (int index = 0; index < batch_size; ++index){
            int channel_idx = channel_idxs[index];
            int image_idx = image_idxs[index];
            float dete_threshold = dete_thresholds[index];
            float nms_threshold = nms_thresholds[index];
            int ost_w = ost_ws[index];
            int ost_h = ost_hs[index];
            int padding_left = padding_lefts[index];
            int padding_top = padding_tops[index];
            int padding_width = padding_widths[index];
            int padding_height = padding_heights[index];

            double time_start = get_current_time_us();
            float scale_w = (float)ost_w/padding_width;
            float scale_h = (float)ost_h/padding_height;
            float* data = data_ost + index*buffer_size_;

            std::vector<DeteObjRect> dete_rects;
        

            int min_idx = 0;
            int box_num = 0;
            int mask_num = 0;
            // Single output
            int m_class_num = data_shape_[1] - mask_num - 4;
            int feat_num = data_shape_[2];
            int nout = m_class_num + mask_num + 4;
            float* output_data = nullptr;
            std::vector<float> decoded_data;
            assert(box_num == 0 || box_num == data_shape_[1]);
            box_num = data_shape_[1];
            output_data = data_ost + index * feat_num * (m_class_num + mask_num + 4);

            // Candidates
            float* cls_conf = output_data + 4 * feat_num;
            for (int i = 0; i < feat_num; i++) {
                if(input_use_multiclass_nms){                // multilabel
                    for (int j = 0; j < m_class_num; j++) {
                        float cur_value = cls_conf[i + j * feat_num];
                        if (cur_value >= dete_threshold) {
                            DeteObjRect box;
                            box.score = cur_value;
                            box.class_id = j;
                            int c = box.class_id * max_wh;
                            float centerX = output_data[i + 0 * feat_num];
                            float centerY = output_data[i + 1 * feat_num];
                            float width = output_data[i + 2 * feat_num];
                            float height = output_data[i + 3 * feat_num];

                            box.left = centerX - width / 2 + c;
                            box.top = centerY - height / 2 + c;
                            box.right = box.left + width;
                            box.bottom = box.top + height;
                            box.width =  width;
                            box.height = height;
                            dete_rects.push_back(box);
                        }
                    }
                }
                else{
                    // best class
                    float max_value = 0.0;
                    int max_index = 0;
                    for (int j = 0; j < m_class_num; j++) {
                        float cur_value = cls_conf[i + j * feat_num];
                        if (cur_value > max_value) {
                            max_value = cur_value;
                            max_index = j;
                        }
                    }

                    if (max_value >= dete_threshold) {
                        DeteObjRect box;
                        box.score = max_value;
                        box.class_id = max_index;
                        int c = box.class_id * max_wh;
                        float centerX = output_data[i + 0 * feat_num];
                        float centerY = output_data[i + 1 * feat_num];
                        float width = output_data[i + 2 * feat_num];
                        float height = output_data[i + 3 * feat_num];

                        box.left = centerX - width / 2 + c;
                        box.top = centerY - height / 2 + c;
                        box.right = box.left + width;
                        box.bottom = box.top + height;
                        box.width =  width;
                        box.height = height;
                        dete_rects.push_back(box);
                       
                    }
                }
                
            }
    
            std::vector<DeteObjRect> dect_result;
            std::vector<int> picked;
            double time_num_start = get_current_time_us();
            sail_algo_nms_sorted_bboxes(dete_rects, picked, nms_threshold);
            double nms_time_use = (get_current_time_us() - time_num_start)/1000;

           
            
            
            for (size_t i = 0; i < picked.size(); i++)    {
                DeteObjRect dete_rect;
                int c = dete_rects[picked[i]].class_id * max_wh;
                dete_rects[picked[i]].left = dete_rects[picked[i]].left - c;
                dete_rects[picked[i]].top = dete_rects[picked[i]].top - c;
                dete_rects[picked[i]].right = dete_rects[picked[i]].right - c;
                dete_rects[picked[i]].bottom = dete_rects[picked[i]].bottom - c;
                dete_rect.left = (dete_rects[picked[i]].left - padding_left) * scale_w;
                dete_rect.top = (dete_rects[picked[i]].top - padding_top) * scale_h;
                dete_rect.left = dete_rect.left > 0 ? dete_rect.left : 0;
                dete_rect.top = dete_rect.top > 0 ? dete_rect.top : 0;

                dete_rect.right = (dete_rects[picked[i]].right - padding_left) * scale_w;
                dete_rect.bottom = (dete_rects[picked[i]].bottom - padding_top) * scale_h;
                dete_rect.right = dete_rect.right < ost_w ? dete_rect.right : ost_w-1;
                dete_rect.bottom = dete_rect.bottom < ost_h ? dete_rect.bottom : ost_h-1;

                dete_rect.width = dete_rect.right - dete_rect.left;
                dete_rect.height = dete_rect.bottom - dete_rect.top;

                // dete_rect.width = dete_rects[picked[i]].width * scale_w;
                // dete_rect.height = dete_rects[picked[i]].height * scale_h;

                // dete_rect.left = dete_rect.left > 0 ? dete_rect.left : 0;
                // dete_rect.top = dete_rect.top > 0 ? dete_rect.top : 0;
                // dete_rect.width = dete_rect.width < network_width ? dete_rect.width : network_width;
                // dete_rect.height = dete_rect.height < network_height ? dete_rect.height : network_height;

                // dete_rect.right = dete_rect.left + dete_rect.width < ost_w ? dete_rect.left + dete_rect.width : ost_w-1;
                // dete_rect.bottom = dete_rect.top + dete_rect.height < ost_h ? dete_rect.top + dete_rect.height : ost_h-1;
                

                dete_rect.score = dete_rects[picked[i]].score;
                dete_rect.class_id = dete_rects[picked[i]].class_id;

                dect_result.push_back(dete_rect);
            }
            double time_use = (get_current_time_us() - time_start)/1000;
            // SPDLOG_INFO("Yolov8 one output post process us {} ms, nms: {} ms. scale_w: {}, scale_h: {}",time_use, nms_time_use,scale_w,scale_h);  
            while (true) {
                int ret = push_result(dect_result, channel_idx, image_idx);
                if(ret == 0) {
                    notify_result_push();
                    break;
                }
                std::unique_lock<std::mutex> lck(mutex_result_pop);
                result_pop_cond.wait_for(lck,std::chrono::milliseconds(5));
                if(get_stop_flag()){
                    break;
                }  
            }
        }
        if(in_data){
            delete in_data;
        }
    }
    set_thread_exit();
    {
        std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
        exit_thread_flag = true;
    }
}

void algo_yolov8_post_1output_async::algo_yolov8_post_1output_async_cc::notify_data_push()
{
    std::unique_lock<std::mutex> lck(mutex_pushdata_flag);
    pushdata_flag_cond.notify_all();
}

void algo_yolov8_post_1output_async::algo_yolov8_post_1output_async_cc::notify_result_push()
{
    std::unique_lock<std::mutex> lck(mutex_result_push);
    result_push_cond.notify_all();
}

void algo_yolov8_post_1output_async::algo_yolov8_post_1output_async_cc::notify_result_pop()
{
    std::unique_lock<std::mutex> lck(mutex_result_pop);
    result_pop_cond.notify_all();
}

bool algo_yolov8_post_1output_async::algo_yolov8_post_1output_async_cc::get_stop_flag()
{
    std::lock_guard<std::mutex> lock(mutex_stop_);
    return stop_thread_flag;
}

void algo_yolov8_post_1output_async::algo_yolov8_post_1output_async_cc::set_stop_flag(bool flag)
{
    std::lock_guard<std::mutex> lock(mutex_stop_);
    stop_thread_flag = flag;
}

void algo_yolov8_post_1output_async::algo_yolov8_post_1output_async_cc::set_thread_exit()
{
    std::unique_lock<std::mutex> lck(mutex_exit_);
    exit_cond.notify_all();
}   

void algo_yolov8_post_1output_async::algo_yolov8_post_1output_async_cc::wait_thread_exit()
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

std::tuple<std::vector<DeteObjRect>,int,int> algo_yolov8_post_1output_async::algo_yolov8_post_1output_async_cc::get_result()
{
#ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    std::vector<DeteObjRect> results;
    int channel_idx = 0;
    int image_idx = 0;
    while(true){
        int ret = get_result_once(results, channel_idx, image_idx);
        if(ret == 0){
            break;
        }
        std::unique_lock<std::mutex> lck(mutex_result_push);
        result_push_cond.wait_for(lck,std::chrono::milliseconds(5));
        if(get_stop_flag()){
            break;
        } 
    }
    return std::make_tuple(std::move(results),channel_idx,image_idx);
}

algo_yolov8_post_1output_async::algo_yolov8_post_1output_async(const std::vector<int>& shape,int network_w, int network_h, int max_queue_size, bool input_use_multiclass_nms, bool agnostic)
:_impl (new algo_yolov8_post_1output_async_cc(shape,network_w,network_h,max_queue_size, input_use_multiclass_nms,agnostic))
{
}

algo_yolov8_post_1output_async::~algo_yolov8_post_1output_async()
{
    delete _impl;
}

#ifdef PYTHON
int algo_yolov8_post_1output_async::push_npy(int channel_idx, int image_idx, pybind11::array_t<float> ost_array, float dete_threshold, float nms_threshold,
        int ost_w, int ost_h, int padding_left, int padding_top, int padding_width, int padding_height)
{
    sail::Handle handle(0);
    std::vector<int> channel_idxs = {channel_idx};
    std::vector<int> image_idxs = {image_idx};
    std::vector<float> dete_thresholds = {dete_threshold};
    std::vector<float> nms_thresholds = {nms_threshold};
    std::vector<int> ost_ws = {ost_w};
    std::vector<int> ost_hs = {ost_h};
    std::vector<std::vector<int>> padding_attrs = {{padding_left,padding_top,padding_width,padding_height}};

    if (!pybind11::detail::check_flags(ost_array.ptr(), pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_)) {
        pybind11::module np = pybind11::module::import("numpy");        // like 'import numpy as np'
        pybind11::array_t<float> arr_c = np.attr("ascontiguousarray")(ost_array, "dtype"_a="float32");
        sail::Tensor* tensor_in = new sail::Tensor(handle, arr_c, true);
        return _impl->push_data(channel_idxs,image_idxs,tensor_in,dete_thresholds,nms_thresholds, ost_ws, ost_hs, padding_attrs);
    }else{
        sail::Tensor* tensor_in = new sail::Tensor(handle, ost_array, true);
        return _impl->push_data(channel_idxs,image_idxs,tensor_in,dete_thresholds,nms_thresholds, ost_ws, ost_hs, padding_attrs);
    }
}
#endif
    
int algo_yolov8_post_1output_async::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        TensorPTRWithName input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr)
{
    return _impl->push_data(channel_idx,image_idx,input_data.data,dete_threshold,nms_threshold, ost_w, ost_h, padding_attr);
}

std::tuple<std::vector<DeteObjRect>,int,int> algo_yolov8_post_1output_async::get_result()
{
    return _impl->get_result();
}
std::tuple<std::vector<std::tuple<int, int, int, int ,int, float>>,int,int> algo_yolov8_post_1output_async::get_result_npy()
{
    auto result = _impl->get_result();
    std::vector<DeteObjRect> results = std::move(std::get<0>(result));
    int channel_idx = std::get<1>(result);
    int image_idx = std::get<2>(result);
    // SPDLOG_INFO("results num : {}",results.size());
    std::vector<std::tuple<int, int, int, int ,int, float>> objs;
    for (int i = 0; i < results.size(); ++i){
        int left_temp = results[i].left+0.5;
        int top_temp = results[i].top+0.5;
        int right_temp = results[i].right+0.5;
        int bottom_temp = results[i].bottom+0.5;
        int class_id_temp = results[i].class_id;
        float score_temp = results[i].score;

        objs.push_back(std::make_tuple(left_temp,
                                       top_temp,
                                       right_temp,
                                       bottom_temp,
                                       class_id_temp,
                                       score_temp));

    }
    return std::make_tuple(std::move(objs),channel_idx,image_idx);
}


class algo_yolov8_post_cpu_opt_1output_async::algo_yolov8_post_cpu_opt_1output_async_cc{
public:
    explicit algo_yolov8_post_cpu_opt_1output_async_cc(const std::vector<int>& shape, int network_w, int network_h, int max_queue_size, bool input_use_multiclass_nms, bool agnostic);
    ~algo_yolov8_post_cpu_opt_1output_async_cc();

    int push_data(std::vector<int> channel_idx, 
                    std::vector<int> image_idx, 
                    sail::Tensor* input_data, 
                    std::vector<float> dete_threshold,
                    std::vector<float> nms_threshold,
                    std::vector<int> ost_w,
                    std::vector<int> ost_h,
                    std::vector<std::vector<int>> padding_attr);

    int push_data(std::vector<int> channel_idx, 
                    std::vector<int> image_idx, 
                    float* input_data, 
                    std::vector<float> dete_threshold,
                    std::vector<float> nms_threshold,
                    std::vector<int> ost_w,
                    std::vector<int> ost_h,
                    std::vector<std::vector<int>> padding_attr);

    std::tuple<std::vector<DeteObjRect>,int,int> get_result();

    int get_buffer_size(){
        return buffer_size_;
    }

private:
    void post_thread();

    sail::Tensor* get_data(std::vector<int>& channel_idx, 
                std::vector<int>& image_idx, 
                std::vector<float> &dete_threshold, 
                std::vector<float> &nms_threshold,
                std::vector<int>& ost_w, 
                std::vector<int>& ost_h, 
                std::vector<int>& padding_left, 
                std::vector<int>& padding_top, 
                std::vector<int>& padding_width, 
                std::vector<int>& padding_height);

    void set_stop_flag(bool flag);  //设置线程退出的标志位

    bool get_stop_flag();           //获取线程退出的标志位

    void set_thread_exit();         //设置线程已经退出         

    void wait_thread_exit();        //等待线程退出

    void notify_data_push();        //

    void notify_result_push();       

    void notify_result_pop();        

    int get_result_once(std::vector<DeteObjRect>& result, int &channel_idx,int &image_idx);     //get detect only once, return 0 for success

    int push_result(std::vector<DeteObjRect> result, int channel_idx,int image_idx);            

    std::vector<int> data_shape_;   //
    int classes_;                   //
    int max_queue_size_;            //
    int buffer_size_;               //

    std::queue<int> channels_queue_;        //
    std::queue<int> image_idx_queue_;       //
    std::queue<sail::Tensor*> tensor_in_queue_;   //
    std::queue<float> dete_threshold_queue;   //
    std::queue<float> nms_threshold_queue;    //

    std::queue<int> ost_w_queue;            //
    std::queue<int> ost_h_queue;            //
    std::queue<int> padding_left_queue;     //
    std::queue<int> padding_top_queue;      //
    std::queue<int> padding_width_queue;    //
    std::queue<int> padding_height_queue;   //

    std::mutex mutex_data;

    std::queue<int> out_channels_queue_;                        //
    std::queue<int> out_image_idx_queue_;                       //
    std::queue<std::vector<DeteObjRect>> dete_result_queue_;    //
    std::mutex mutex_result;

    bool stop_thread_flag;      //线程退出的标志位
    std::mutex mutex_stop_;     //线程退出互斥锁
    std::condition_variable exit_cond;  //线程已经退出的信号量
    std::mutex mutex_exit_;             //线程已经退出互斥锁
    bool exit_thread_flag;      //线程已经退出的标志位

    std::condition_variable result_push_cond;       //后处理线程有结果送入结果队列的信号
    std::mutex mutex_result_push;

    std::condition_variable result_pop_cond;       //后处理线程有结果送入结果队列的信号
    std::mutex mutex_result_pop;

    std::condition_variable pushdata_flag_cond;     //有数据push进来的信号
    std::mutex mutex_pushdata_flag;

    bool post_thread_run;

    int network_width; //
    int network_height; //
    int batch_size; //

    // nms params 
    bool input_use_multiclass_nms;
    bool agnostic;
};

algo_yolov8_post_cpu_opt_1output_async::algo_yolov8_post_cpu_opt_1output_async_cc::algo_yolov8_post_cpu_opt_1output_async_cc(
    const std::vector<int>& shape,
    int network_w,
    int network_h,
    int max_queue_size,
    bool input_use_multiclass_nms, 
    bool agnostic)
:max_queue_size_(max_queue_size),post_thread_run(false),stop_thread_flag(false),exit_thread_flag(true),
network_width(network_w),network_height(network_h)
{
    if (shape.size() != 3){
        SPDLOG_ERROR("ERROR DIMS, 3 vs. {}!",shape.size());
        throw SailRuntimeError("invalid argument");
    }
    buffer_size_ = 1;
    for(int i=0;i<shape.size();++i)    {
        data_shape_.push_back(shape[i]);
    }
    for(int i=1;i<shape.size();++i)    {
        buffer_size_ = buffer_size_ * shape[i];
    }
    classes_ = data_shape_[2]-5;
    batch_size = shape[0];

    // nms paras init
    this->input_use_multiclass_nms = input_use_multiclass_nms;
    this->agnostic = agnostic;
}

algo_yolov8_post_cpu_opt_1output_async::algo_yolov8_post_cpu_opt_1output_async_cc::~algo_yolov8_post_cpu_opt_1output_async_cc()
{
    set_stop_flag(true);
    SPDLOG_INFO("Start Set Thread Exit Flag!");
    set_thread_exit();
    SPDLOG_INFO("End Set Thread Exit Flag, Waiting Thread Exit....");
    wait_thread_exit();
}

int algo_yolov8_post_cpu_opt_1output_async::algo_yolov8_post_cpu_opt_1output_async_cc::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        sail::Tensor* input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr){
 #ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    if(channel_idx.size() != batch_size ||
        image_idx.size() != batch_size ||
        input_data->shape()[0] != batch_size ||
        dete_threshold.size() != batch_size ||
        padding_attr.size() != batch_size ||
        ost_w.size() != batch_size ||
        ost_h.size() != batch_size ||
        nms_threshold.size() != batch_size){
        SPDLOG_ERROR("Input shape {},{},{}",input_data->shape()[0],input_data->shape()[1],input_data->shape()[2]);
        SPDLOG_ERROR("Input batch size mismatch, {} vs. {}, {}, {}, {}, {}, {}, {}, {}",
            batch_size, channel_idx.size(), image_idx.size(), input_data->shape()[0], dete_threshold.size(),padding_attr.size(),
            ost_w.size(), ost_h.size(), nms_threshold.size());
        return SAIL_ALGO_ERROR_BATCHSIZE;
    }

    if(data_shape_.size() != input_data->shape().size()){
        SPDLOG_ERROR("The shape of the pushed data is incorrect!");
        return SAIL_ALGO_ERROR_SHAPES;
    }else{
        for(int i=0;i<data_shape_.size();i++){
            if(data_shape_[i]!=input_data->shape()[i]){
                SPDLOG_ERROR("The shape of the pushed data is incorrect!");
                return SAIL_ALGO_ERROR_SHAPES;
            }
            else
                continue;
        }
    }
    
    std::lock_guard<std::mutex> lock(mutex_data);
    if(tensor_in_queue_.size() >= max_queue_size_) {
        return SAIL_ALGO_BUFFER_FULL;
    }
    for(int i=0; i <batch_size;++i) {
        channels_queue_.push(channel_idx[i]);
        image_idx_queue_.push(image_idx[i]);
        dete_threshold_queue.push(dete_threshold[i]);
        nms_threshold_queue.push(nms_threshold[i]);
        ost_w_queue.push(ost_w[i]);                    //
        ost_h_queue.push(ost_h[i]);                    //
        padding_left_queue.push(padding_attr[i][0]);      //
        padding_top_queue.push(padding_attr[i][1]);        //
        padding_width_queue.push(padding_attr[i][2]);    //
        padding_height_queue.push(padding_attr[i][3]);  //
    }
    tensor_in_queue_.push(std::move(input_data));

    if(!post_thread_run){
        std::thread thread_post = std::thread(&algo_yolov8_post_cpu_opt_1output_async_cc::post_thread,this);
        thread_post.detach();
        post_thread_run = true;
    }
    notify_data_push();
    return SAIL_ALGO_SUCCESS;
};

sail::Tensor* algo_yolov8_post_cpu_opt_1output_async::algo_yolov8_post_cpu_opt_1output_async_cc::get_data(
        std::vector<int>& channel_idx, 
        std::vector<int>& image_idx, 
        std::vector<float> &dete_threshold, 
        std::vector<float> &nms_threshold,
        std::vector<int>& ost_w, 
        std::vector<int>& ost_h, 
        std::vector<int>& padding_left, 
        std::vector<int>& padding_top, 
        std::vector<int>& padding_width, 
        std::vector<int>& padding_height){
    sail::Tensor* input_data = NULL; 
    std::lock_guard<std::mutex> lock(mutex_data);
    if(tensor_in_queue_.empty()) {
        return input_data;
    }
    channel_idx.clear();
    image_idx.clear(); 
    dete_threshold.clear();
    nms_threshold.clear();
    ost_w.clear();
    ost_h.clear();
    padding_left.clear();
    padding_top.clear();
    padding_width.clear();
    padding_height.clear();
    for(int i=0;i<batch_size;++i){
        channel_idx.push_back(channels_queue_.front());
        image_idx.push_back(image_idx_queue_.front());
        dete_threshold.push_back(dete_threshold_queue.front());
        nms_threshold.push_back(nms_threshold_queue.front());

        ost_w.push_back(ost_w_queue.front());                    //
        ost_h.push_back(ost_h_queue.front());                    //
        padding_left.push_back(padding_left_queue.front());      //
        padding_top.push_back(padding_top_queue.front());        //
        padding_width.push_back(padding_width_queue.front());    //
        padding_height.push_back(padding_height_queue.front());  //

        channels_queue_.pop();
        image_idx_queue_.pop();
        dete_threshold_queue.pop();
        nms_threshold_queue.pop();
        ost_w_queue.pop();                    //
        ost_h_queue.pop();                    //
        padding_left_queue.pop();      //
        padding_top_queue.pop();        //
        padding_width_queue.pop();    //
        padding_height_queue.pop();  // 
    }

    input_data = tensor_in_queue_.front();
    tensor_in_queue_.pop(); //

    return std::move(input_data);
}

int algo_yolov8_post_cpu_opt_1output_async::algo_yolov8_post_cpu_opt_1output_async_cc::get_result_once(std::vector<DeteObjRect>& result, int &channel_idx,int &image_idx)
{
    std::lock_guard<std::mutex> lock(mutex_result);
    if(dete_result_queue_.empty()){
        return 1;
    }
    channel_idx = out_channels_queue_.front();
    image_idx = out_image_idx_queue_.front();
    std::vector<DeteObjRect> temp = dete_result_queue_.front();
    for(int i=0;i<temp.size();++i){
        result.push_back(temp[i]);
    }
    out_channels_queue_.pop();
    out_image_idx_queue_.pop();
    dete_result_queue_.pop();

    notify_result_pop();    
    return 0;
}

int algo_yolov8_post_cpu_opt_1output_async::algo_yolov8_post_cpu_opt_1output_async_cc::push_result(std::vector<DeteObjRect> result, int channel_idx,int image_idx)
{
    std::lock_guard<std::mutex> lock(mutex_result);
    if(dete_result_queue_.size() >= max_queue_size_){
        return 1;
    }
    out_channels_queue_.push(channel_idx);
    out_image_idx_queue_.push(image_idx);
    dete_result_queue_.push(std::move(result));
    return 0;
}

void algo_yolov8_post_cpu_opt_1output_async::algo_yolov8_post_cpu_opt_1output_async_cc::post_thread()
{
    {        
        std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
        exit_thread_flag = false;
    }
    // SPDLOG_INFO("Create To YOLOV8 1 output Thread, pid:{}, tid:{} .",getpid(),gettid());
    int idx = 0;
    while(true) {
        if(get_stop_flag()){
            break;
        }  
        std::vector<int> channel_idxs;
        std::vector<int> image_idxs;
        std::vector<float> dete_thresholds;
        std::vector<float> nms_thresholds;
        std::vector<int> ost_ws;
        std::vector<int> ost_hs;
        std::vector<int> padding_lefts;
        std::vector<int> padding_tops;
        std::vector<int> padding_widths;
        std::vector<int> padding_heights;

        sail::Tensor* in_data = get_data(channel_idxs, image_idxs, dete_thresholds, nms_thresholds,
            ost_ws, ost_hs, padding_lefts, padding_tops, padding_widths, padding_heights);
        if(in_data == NULL){
            std::unique_lock<std::mutex> lck(mutex_pushdata_flag);
            pushdata_flag_cond.wait_for(lck,std::chrono::milliseconds(5));
            // SPDLOG_INFO("Get Data failed, sleeping for 5 ms");
            continue;
        }
        float* data_ost = (float*)in_data->sys_data();
        const char *dump_flag = getenv("SAIL_SAVE_IO_TENSORS");
        if (dump_flag != nullptr && 0 == strcmp(dump_flag, "1")){
            double id_save = get_current_time_us();
            char save_name_temp[256]={0};
            sprintf(save_name_temp,"%s_%.0f.dat","algokit_yolov8_post_1out",id_save);
            dump_float32_data((const char* )save_name_temp, data_ost, buffer_size_*batch_size, data_shape_[1], data_shape_[2]);
        }

        // nms init 
        int max_wh = 7680;
        for (int index = 0; index < batch_size; ++index){
            int channel_idx = channel_idxs[index];
            int image_idx = image_idxs[index];
            float dete_threshold = dete_thresholds[index];
            float nms_threshold = nms_thresholds[index];
            int ost_w = ost_ws[index];
            int ost_h = ost_hs[index];
            int padding_left = padding_lefts[index];
            int padding_top = padding_tops[index];
            int padding_width = padding_widths[index];
            int padding_height = padding_heights[index];

            double time_start = get_current_time_us();
            float scale_w = (float)ost_w/padding_width;
            float scale_h = (float)ost_h/padding_height;
            float* data = data_ost + index*buffer_size_;

            std::vector<DeteObjRect> dete_rects;
        

            int min_idx = 0;
            int box_num = 0;
            int mask_num = 0;
            // Single output
            int m_class_num = data_shape_[2] - mask_num - 4;
            int feat_num = data_shape_[1];
            int nout = m_class_num + mask_num + 4;
            float* output_data = nullptr;
            std::vector<float> decoded_data;
            assert(box_num == 0 || box_num == data_shape_[2]);
            box_num = data_shape_[2];
            output_data = data_ost + index * feat_num * (m_class_num + mask_num + 4);

            // Candidates
            float* cls_conf = output_data + 4;
            for (int i = 0; i < feat_num; i++) {
                if(input_use_multiclass_nms){                // multilabel
                    for (int j = 0; j < m_class_num; j++) {
                        float cur_value = cls_conf[i*nout + j];
                        if (cur_value >= dete_threshold) {
                            DeteObjRect box;
                            box.score = cur_value;
                            box.class_id = j;
                            int c = box.class_id * max_wh;
                            float centerX = output_data[i*nout];
                            float centerY = output_data[i*nout + 1];
                            float width = output_data[i*nout + 2];
                            float height = output_data[i*nout + 3];

                            box.left = centerX - width / 2 + c;
                            box.top = centerY - height / 2 + c;
                            box.right = box.left + width;
                            box.bottom = box.top + height;
                            box.width =  width;
                            box.height = height;
                            dete_rects.push_back(box);
                        }
                    }
                }
                else{
                    // best class
                    float max_value = 0.0;
                    int max_index = 0;
                    for (int j = 0; j < m_class_num; j++) {
                        float cur_value = cls_conf[i*nout + j];
                        if (cur_value > max_value) {
                            max_value = cur_value;
                            max_index = j;
                        }
                    }

                    if (max_value >= dete_threshold) {
                        DeteObjRect box;
                        box.score = max_value;
                        box.class_id = max_index;
                        int c = box.class_id * max_wh;
                        float centerX = output_data[i*nout];
                        float centerY = output_data[i*nout + 1];
                        float width = output_data[i*nout + 2];
                        float height = output_data[i*nout + 3];

                        box.left = centerX - width / 2 + c;
                        box.top = centerY - height / 2 + c;
                        box.right = box.left + width;
                        box.bottom = box.top + height;
                        box.width =  width;
                        box.height = height;
                        dete_rects.push_back(box);
                       
                    }
                }
                
            }
    
            std::vector<DeteObjRect> dect_result;
            std::vector<int> picked;
            double time_num_start = get_current_time_us();
            sail_algo_nms_sorted_bboxes(dete_rects, picked, nms_threshold);
            double nms_time_use = (get_current_time_us() - time_num_start)/1000;

           
            
            
            for (size_t i = 0; i < picked.size(); i++)    {
                DeteObjRect dete_rect;
                int c = dete_rects[picked[i]].class_id * max_wh;
                dete_rects[picked[i]].left = dete_rects[picked[i]].left - c;
                dete_rects[picked[i]].top = dete_rects[picked[i]].top - c;
                dete_rects[picked[i]].right = dete_rects[picked[i]].right - c;
                dete_rects[picked[i]].bottom = dete_rects[picked[i]].bottom - c;
                dete_rect.left = (dete_rects[picked[i]].left - padding_left) * scale_w;
                dete_rect.top = (dete_rects[picked[i]].top - padding_top) * scale_h;
                dete_rect.left = dete_rect.left > 0 ? dete_rect.left : 0;
                dete_rect.top = dete_rect.top > 0 ? dete_rect.top : 0;

                dete_rect.right = (dete_rects[picked[i]].right - padding_left) * scale_w;
                dete_rect.bottom = (dete_rects[picked[i]].bottom - padding_top) * scale_h;
                dete_rect.right = dete_rect.right < ost_w ? dete_rect.right : ost_w-1;
                dete_rect.bottom = dete_rect.bottom < ost_h ? dete_rect.bottom : ost_h-1;

                dete_rect.width = dete_rect.right - dete_rect.left;
                dete_rect.height = dete_rect.bottom - dete_rect.top;

                // dete_rect.width = dete_rects[picked[i]].width * scale_w;
                // dete_rect.height = dete_rects[picked[i]].height * scale_h;

                // dete_rect.left = dete_rect.left > 0 ? dete_rect.left : 0;
                // dete_rect.top = dete_rect.top > 0 ? dete_rect.top : 0;
                // dete_rect.width = dete_rect.width < network_width ? dete_rect.width : network_width;
                // dete_rect.height = dete_rect.height < network_height ? dete_rect.height : network_height;

                // dete_rect.right = dete_rect.left + dete_rect.width < ost_w ? dete_rect.left + dete_rect.width : ost_w-1;
                // dete_rect.bottom = dete_rect.top + dete_rect.height < ost_h ? dete_rect.top + dete_rect.height : ost_h-1;
                

                dete_rect.score = dete_rects[picked[i]].score;
                dete_rect.class_id = dete_rects[picked[i]].class_id;

                dect_result.push_back(dete_rect);
            }
            double time_use = (get_current_time_us() - time_start)/1000;
            // SPDLOG_INFO("Yolov8 one output post process us {} ms, nms: {} ms. scale_w: {}, scale_h: {}",time_use, nms_time_use,scale_w,scale_h);  
            while (true) {
                int ret = push_result(dect_result, channel_idx, image_idx);
                if(ret == 0) {
                    notify_result_push();
                    break;
                }
                std::unique_lock<std::mutex> lck(mutex_result_pop);
                result_pop_cond.wait_for(lck,std::chrono::milliseconds(5));
                if(get_stop_flag()){
                    break;
                }  
            }
        }
        if(in_data){
            delete in_data;
        }
    }
    set_thread_exit();
    {
        std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
        exit_thread_flag = true;
    }
}

void algo_yolov8_post_cpu_opt_1output_async::algo_yolov8_post_cpu_opt_1output_async_cc::notify_data_push()
{
    std::unique_lock<std::mutex> lck(mutex_pushdata_flag);
    pushdata_flag_cond.notify_all();
}

void algo_yolov8_post_cpu_opt_1output_async::algo_yolov8_post_cpu_opt_1output_async_cc::notify_result_push()
{
    std::unique_lock<std::mutex> lck(mutex_result_push);
    result_push_cond.notify_all();
}

void algo_yolov8_post_cpu_opt_1output_async::algo_yolov8_post_cpu_opt_1output_async_cc::notify_result_pop()
{
    std::unique_lock<std::mutex> lck(mutex_result_pop);
    result_pop_cond.notify_all();
}

bool algo_yolov8_post_cpu_opt_1output_async::algo_yolov8_post_cpu_opt_1output_async_cc::get_stop_flag()
{
    std::lock_guard<std::mutex> lock(mutex_stop_);
    return stop_thread_flag;
}

void algo_yolov8_post_cpu_opt_1output_async::algo_yolov8_post_cpu_opt_1output_async_cc::set_stop_flag(bool flag)
{
    std::lock_guard<std::mutex> lock(mutex_stop_);
    stop_thread_flag = flag;
}

void algo_yolov8_post_cpu_opt_1output_async::algo_yolov8_post_cpu_opt_1output_async_cc::set_thread_exit()
{
    std::unique_lock<std::mutex> lck(mutex_exit_);
    exit_cond.notify_all();
}   

void algo_yolov8_post_cpu_opt_1output_async::algo_yolov8_post_cpu_opt_1output_async_cc::wait_thread_exit()
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

std::tuple<std::vector<DeteObjRect>,int,int> algo_yolov8_post_cpu_opt_1output_async::algo_yolov8_post_cpu_opt_1output_async_cc::get_result()
{
#ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    std::vector<DeteObjRect> results;
    int channel_idx = 0;
    int image_idx = 0;
    while(true){
        int ret = get_result_once(results, channel_idx, image_idx);
        if(ret == 0){
            break;
        }
        std::unique_lock<std::mutex> lck(mutex_result_push);
        result_push_cond.wait_for(lck,std::chrono::milliseconds(5));
        if(get_stop_flag()){
            break;
        } 
    }
    return std::make_tuple(std::move(results),channel_idx,image_idx);
}

algo_yolov8_post_cpu_opt_1output_async::algo_yolov8_post_cpu_opt_1output_async(const std::vector<int>& shape,int network_w, int network_h, int max_queue_size, bool input_use_multiclass_nms, bool agnostic)
:_impl (new algo_yolov8_post_cpu_opt_1output_async_cc(shape,network_w,network_h,max_queue_size, input_use_multiclass_nms,agnostic))
{
}

algo_yolov8_post_cpu_opt_1output_async::~algo_yolov8_post_cpu_opt_1output_async()
{
    delete _impl;
}

#ifdef PYTHON
int algo_yolov8_post_cpu_opt_1output_async::push_npy(int channel_idx, int image_idx, pybind11::array_t<float> ost_array, float dete_threshold, float nms_threshold,
        int ost_w, int ost_h, int padding_left, int padding_top, int padding_width, int padding_height)
{
    sail::Handle handle(0);
    std::vector<int> channel_idxs = {channel_idx};
    std::vector<int> image_idxs = {image_idx};
    std::vector<float> dete_thresholds = {dete_threshold};
    std::vector<float> nms_thresholds = {nms_threshold};
    std::vector<int> ost_ws = {ost_w};
    std::vector<int> ost_hs = {ost_h};
    std::vector<std::vector<int>> padding_attrs = {{padding_left,padding_top,padding_width,padding_height}};

    if (!pybind11::detail::check_flags(ost_array.ptr(), pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_)) {
        pybind11::module np = pybind11::module::import("numpy");        // like 'import numpy as np'
        pybind11::array_t<float> arr_c = np.attr("ascontiguousarray")(ost_array, "dtype"_a="float32");
        sail::Tensor* tensor_in = new sail::Tensor(handle, arr_c, true);
        return _impl->push_data(channel_idxs,image_idxs,tensor_in,dete_thresholds,nms_thresholds, ost_ws, ost_hs, padding_attrs);
    }else{
        sail::Tensor* tensor_in = new sail::Tensor(handle, ost_array, true);
        return _impl->push_data(channel_idxs,image_idxs,tensor_in,dete_thresholds,nms_thresholds, ost_ws, ost_hs, padding_attrs);
    }
}
#endif
    
int algo_yolov8_post_cpu_opt_1output_async::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        TensorPTRWithName input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr)
{
    return _impl->push_data(channel_idx,image_idx,input_data.data,dete_threshold,nms_threshold, ost_w, ost_h, padding_attr);
}

std::tuple<std::vector<DeteObjRect>,int,int> algo_yolov8_post_cpu_opt_1output_async::get_result()
{
    return _impl->get_result();
}
std::tuple<std::vector<std::tuple<int, int, int, int ,int, float>>,int,int> algo_yolov8_post_cpu_opt_1output_async::get_result_npy()
{
    auto result = _impl->get_result();
    std::vector<DeteObjRect> results = std::move(std::get<0>(result));
    int channel_idx = std::get<1>(result);
    int image_idx = std::get<2>(result);
    // SPDLOG_INFO("results num : {}",results.size());
    std::vector<std::tuple<int, int, int, int ,int, float>> objs;
    for (int i = 0; i < results.size(); ++i){
        int left_temp = results[i].left+0.5;
        int top_temp = results[i].top+0.5;
        int right_temp = results[i].right+0.5;
        int bottom_temp = results[i].bottom+0.5;
        int class_id_temp = results[i].class_id;
        float score_temp = results[i].score;

        objs.push_back(std::make_tuple(left_temp,
                                       top_temp,
                                       right_temp,
                                       bottom_temp,
                                       class_id_temp,
                                       score_temp));

    }
    return std::make_tuple(std::move(objs),channel_idx,image_idx);
}



// yolov5 post 3output
class algo_yolov5_post_3output::algo_yolov5_post_3output_cc{
public:
    explicit algo_yolov5_post_3output_cc(const std::vector<std::vector<int>>& shape, int network_w, int network_h, int max_queue_size, bool input_use_multiclass_nms, bool agnostic);
    ~algo_yolov5_post_3output_cc();

    int push_data(std::vector<int> channel_idx, 
                    std::vector<int> image_idx, 
                    std::vector<sail::Tensor *> input_data, 
                    std::vector<float> dete_threshold,
                    std::vector<float> nms_threshold,
                    std::vector<int> ost_w,
                    std::vector<int> ost_h,
                    std::vector<std::vector<int>> padding_attr);

    int push_data(std::vector<int> channel_idx, 
                    std::vector<int> image_idx, 
                    std::vector<sail::Tensor *> input_data, 
                    std::vector<std::vector<float>> dete_threshold,
                    std::vector<float> nms_threshold,
                    std::vector<int> ost_w,
                    std::vector<int> ost_h,
                    std::vector<std::vector<int>> padding_attr);
    std::tuple<std::vector<DeteObjRect>,int,int> get_result();

    int reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new);

private:
    void post_thread();

    std::vector<sail::Tensor*> get_data(std::vector<int>& channel_idx, 
                std::vector<int>& image_idx, 
                std::vector<std::vector<float>> &dete_threshold, 
                std::vector<float> &nms_threshold,
                std::vector<int>& ost_w, 
                std::vector<int>& ost_h, 
                std::vector<int>& padding_left, 
                std::vector<int>& padding_top, 
                std::vector<int>& padding_width, 
                std::vector<int>& padding_height);

    void set_stop_flag(bool flag);  //设置线程退出的标志位

    bool get_stop_flag();           //获取线程退出的标志位

    void set_thread_exit();         //设置线程已经退出         

    void wait_thread_exit();        //等待线程退出

    void notify_data_push();        //

    void notify_result_push();       

    void notify_result_pop();        

    int get_result_once(std::vector<DeteObjRect>& result, int &channel_idx,int &image_idx);     //get detect only once, return 0 for success

    int push_result(std::vector<DeteObjRect> result, int channel_idx,int image_idx);            

    std::vector<std::vector<int>> data_shape_;   //
    int classes_;                   //
    int max_queue_size_;            //

    std::queue<int> channels_queue_;        //
    std::queue<int> image_idx_queue_;       //
    std::queue<std::vector<sail::Tensor*>> tensor_in_queue_;   //
    std::queue<std::vector<float>> dete_threshold_queue;   //
    std::queue<float> nms_threshold_queue;    //

    std::queue<int> ost_w_queue;            //
    std::queue<int> ost_h_queue;            //
    std::queue<int> padding_left_queue;     //
    std::queue<int> padding_top_queue;      //
    std::queue<int> padding_width_queue;    //
    std::queue<int> padding_height_queue;   //

    std::mutex mutex_data;

    std::queue<int> out_channels_queue_;                        //
    std::queue<int> out_image_idx_queue_;                       //
    std::queue<std::vector<DeteObjRect>> dete_result_queue_;    //
    std::mutex mutex_result;

    bool stop_thread_flag;      //线程退出的标志位
    std::mutex mutex_stop_;     //线程退出互斥锁
    std::condition_variable exit_cond;  //线程已经退出的信号量
    std::mutex mutex_exit_;             //线程已经退出互斥锁
    bool exit_thread_flag;      //线程已经退出的标志位

    std::condition_variable result_push_cond;       //后处理线程有结果送入结果队列的信号
    std::mutex mutex_result_push;

    std::condition_variable result_pop_cond;       //后处理线程有结果送入结果队列的信号
    std::mutex mutex_result_pop;

    std::condition_variable pushdata_flag_cond;     //有数据push进来的信号
    std::mutex mutex_pushdata_flag;

    bool post_thread_run;

    int network_width; //
    int network_height; //
    int batch_size; //

    std::vector<std::vector<std::vector<int>>> anchors;
    int anchor_num;

    // nms paras 
    bool input_use_multiclass_nms, agnostic;
};

algo_yolov5_post_3output::algo_yolov5_post_3output_cc::algo_yolov5_post_3output_cc(
    const std::vector<std::vector<int>>& shape,
    int network_w,
    int network_h,
    int max_queue_size,
    bool input_use_multiclass_nms,
    bool agnostic)
:max_queue_size_(max_queue_size),post_thread_run(false),stop_thread_flag(false),exit_thread_flag(true),
network_width(network_w),network_height(network_h)
{
    if (shape.size() != 3){
        SPDLOG_ERROR("ERROR outputs num, 3 vs. {}!",shape.size());
        throw SailRuntimeError("invalid argument");
    }
    if (shape[0].size() != 5){
        SPDLOG_ERROR("ERROR DIM, 5 vs. {}!",shape[0].size());
        throw SailRuntimeError("invalid argument");
    }
    
    for(int i=0;i<shape.size();++i){
        std::vector<int> data_shape;
        for (int j=0;j<shape[0].size();++j){
            data_shape.push_back(shape[i][j]);
        }
        data_shape_.push_back(data_shape);
    }

    classes_ = data_shape_[0][4]-5;
    batch_size = shape[0][0];

    anchors = {{{10, 13}, {16, 30}, {33, 23}}, {{30, 61}, {62, 45}, {59, 119}}, {{116, 90}, {156, 198}, {373, 326}}};
    anchor_num = anchors[0].size();

    // nms params init
    this->input_use_multiclass_nms = input_use_multiclass_nms;
    this->agnostic = agnostic;
}

algo_yolov5_post_3output::algo_yolov5_post_3output_cc::~algo_yolov5_post_3output_cc()
{
    set_stop_flag(true);
    SPDLOG_INFO("Start Set Thread Exit Flag!");
    set_thread_exit();
    SPDLOG_INFO("End Set Thread Exit Flag, Waiting Thread Exit....");
    wait_thread_exit();
}

int algo_yolov5_post_3output::algo_yolov5_post_3output_cc::reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new){
#ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    anchors.clear();
    for(int i=0;i<anchors_new.size();++i){
        std::vector<std::vector<int>> anchor_;
        for (int j=0;j<anchors_new[0].size();++j){
            anchor_.push_back({anchors_new[i][j][0], anchors_new[i][j][1]});
        }
        anchors.push_back(anchor_);
    }
    anchor_num = anchors[0].size();
    SPDLOG_INFO("Reset Anchors, anchor_num:{} .", anchor_num);
    return 0;
}

int algo_yolov5_post_3output::algo_yolov5_post_3output_cc::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        std::vector<sail::Tensor*> input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr){
 #ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    if(channel_idx.size() != batch_size ||
        image_idx.size() != batch_size ||
        input_data[0]->shape()[0] != batch_size ||
        dete_threshold.size() != batch_size ||
        padding_attr.size() != batch_size ||
        ost_w.size() != batch_size ||
        ost_h.size() != batch_size ||
        nms_threshold.size() != batch_size){
        SPDLOG_ERROR("Input shape {},{},{},{},{}",input_data[0]->shape()[0],input_data[0]->shape()[1],input_data[0]->shape()[2],input_data[0]->shape()[3],input_data[0]->shape()[4]);
        SPDLOG_ERROR("Input batch size mismatch, {} vs. {}, {}, {}, {}, {}, {}, {}, {}",
            batch_size, channel_idx.size(), image_idx.size(), input_data[0]->shape()[0], dete_threshold.size(),padding_attr.size(),
            ost_w.size(), ost_h.size(), nms_threshold.size());
        return SAIL_ALGO_ERROR_BATCHSIZE;
    }
    std::lock_guard<std::mutex> lock(mutex_data);
    if(tensor_in_queue_.size() >= max_queue_size_) {
        return SAIL_ALGO_BUFFER_FULL;
    }
    for(int i=0; i <batch_size;++i) {
        channels_queue_.push(channel_idx[i]);
        image_idx_queue_.push(image_idx[i]);
        dete_threshold_queue.push({dete_threshold[i]});
        nms_threshold_queue.push(nms_threshold[i]);
        ost_w_queue.push(ost_w[i]);                    //
        ost_h_queue.push(ost_h[i]);                    //
        padding_left_queue.push(padding_attr[i][0]);      //
        padding_top_queue.push(padding_attr[i][1]);        //
        padding_width_queue.push(padding_attr[i][2]);    //
        padding_height_queue.push(padding_attr[i][3]);  //
    }
    tensor_in_queue_.push(std::move(input_data));

    if(!post_thread_run){
        std::thread thread_post = std::thread(&algo_yolov5_post_3output_cc::post_thread,this);
        thread_post.detach();
        post_thread_run = true;
    }
    notify_data_push();
    return SAIL_ALGO_SUCCESS;
};

int algo_yolov5_post_3output::algo_yolov5_post_3output_cc::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        std::vector<sail::Tensor*> input_data, 
        std::vector<std::vector<float>> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr){
 #ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    if(channel_idx.size() != batch_size ||
        image_idx.size() != batch_size ||
        input_data[0]->shape()[0] != batch_size ||
        dete_threshold.size() != batch_size ||
        padding_attr.size() != batch_size ||
        ost_w.size() != batch_size ||
        ost_h.size() != batch_size ||
        nms_threshold.size() != batch_size){
        SPDLOG_ERROR("Input shape {},{},{},{},{}",input_data[0]->shape()[0],input_data[0]->shape()[1],input_data[0]->shape()[2],input_data[0]->shape()[3],input_data[0]->shape()[4]);
        SPDLOG_ERROR("Input batch size mismatch, {} vs. {}, {}, {}, {}, {}, {}, {}, {}",
            batch_size, channel_idx.size(), image_idx.size(), input_data[0]->shape()[0], dete_threshold.size(),padding_attr.size(),
            ost_w.size(), ost_h.size(), nms_threshold.size());
        return SAIL_ALGO_ERROR_BATCHSIZE;
    }
    std::lock_guard<std::mutex> lock(mutex_data);
    if(tensor_in_queue_.size() >= max_queue_size_) {
        return SAIL_ALGO_BUFFER_FULL;
    }
    for(int i=0; i <batch_size;++i) {
        channels_queue_.push(channel_idx[i]);
        image_idx_queue_.push(image_idx[i]);
        dete_threshold_queue.push(dete_threshold[i]);
        nms_threshold_queue.push(nms_threshold[i]);
        ost_w_queue.push(ost_w[i]);                    //
        ost_h_queue.push(ost_h[i]);                    //
        padding_left_queue.push(padding_attr[i][0]);      //
        padding_top_queue.push(padding_attr[i][1]);        //
        padding_width_queue.push(padding_attr[i][2]);    //
        padding_height_queue.push(padding_attr[i][3]);  //
    }
    tensor_in_queue_.push(std::move(input_data));

    if(!post_thread_run){
        std::thread thread_post = std::thread(&algo_yolov5_post_3output_cc::post_thread,this);
        thread_post.detach();
        post_thread_run = true;
    }
    notify_data_push();
    return SAIL_ALGO_SUCCESS;
};

std::vector<sail::Tensor*> algo_yolov5_post_3output::algo_yolov5_post_3output_cc::get_data(
        std::vector<int>& channel_idx, 
        std::vector<int>& image_idx, 
        std::vector<std::vector<float>> &dete_threshold, 
        std::vector<float> &nms_threshold,
        std::vector<int>& ost_w, 
        std::vector<int>& ost_h, 
        std::vector<int>& padding_left, 
        std::vector<int>& padding_top, 
        std::vector<int>& padding_width, 
        std::vector<int>& padding_height){
    std::vector<sail::Tensor*> input_data; 
    std::lock_guard<std::mutex> lock(mutex_data);
    if(tensor_in_queue_.empty()) {
        // SPDLOG_INFO("tensor_in_queue_ is empty");
        return input_data;
    }
    channel_idx.clear();
    image_idx.clear(); 
    dete_threshold.clear();
    nms_threshold.clear();
    ost_w.clear();
    ost_h.clear();
    padding_left.clear();
    padding_top.clear();
    padding_width.clear();
    padding_height.clear();
    for(int i=0;i<batch_size;++i){
        channel_idx.push_back(channels_queue_.front());
        image_idx.push_back(image_idx_queue_.front());
        dete_threshold.push_back(dete_threshold_queue.front());
        nms_threshold.push_back(nms_threshold_queue.front());

        ost_w.push_back(ost_w_queue.front());                    //
        ost_h.push_back(ost_h_queue.front());                    //
        padding_left.push_back(padding_left_queue.front());      //
        padding_top.push_back(padding_top_queue.front());        //
        padding_width.push_back(padding_width_queue.front());    //
        padding_height.push_back(padding_height_queue.front());  //

        channels_queue_.pop();
        image_idx_queue_.pop();
        dete_threshold_queue.pop();
        nms_threshold_queue.pop();
        ost_w_queue.pop();                    //
        ost_h_queue.pop();                    //
        padding_left_queue.pop();      //
        padding_top_queue.pop();        //
        padding_width_queue.pop();    //
        padding_height_queue.pop();  // 
    }

    input_data = tensor_in_queue_.front();
    tensor_in_queue_.pop(); //

    return std::move(input_data);
}

int algo_yolov5_post_3output::algo_yolov5_post_3output_cc::get_result_once(std::vector<DeteObjRect>& result, int &channel_idx,int &image_idx)
{
    std::lock_guard<std::mutex> lock(mutex_result);
    if(dete_result_queue_.empty()){
        return 1;
    }
    channel_idx = out_channels_queue_.front();
    image_idx = out_image_idx_queue_.front();
    std::vector<DeteObjRect> temp = dete_result_queue_.front();
    for(int i=0;i<temp.size();++i){
        result.push_back(temp[i]);
    }
    out_channels_queue_.pop();
    out_image_idx_queue_.pop();
    dete_result_queue_.pop();

    notify_result_pop();    
    return 0;
}

int algo_yolov5_post_3output::algo_yolov5_post_3output_cc::push_result(std::vector<DeteObjRect> result, int channel_idx,int image_idx)
{
    std::lock_guard<std::mutex> lock(mutex_result);
    if(dete_result_queue_.size() >= max_queue_size_){
        return 1;
    }
    out_channels_queue_.push(channel_idx);
    out_image_idx_queue_.push(image_idx);
    dete_result_queue_.push(std::move(result));
    return 0;
}

void algo_yolov5_post_3output::algo_yolov5_post_3output_cc::post_thread()
{
    {        
        std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
        exit_thread_flag = false;
    }
    // SPDLOG_INFO("Create To YOLOV5 1 output Thread, pid:{}, tid:{} .",getpid(),gettid());
    int idx = 0;
    while(true) {
        if(get_stop_flag()){
            break;
        }  
        std::vector<int> channel_idxs;
        std::vector<int> image_idxs;
        std::vector<std::vector<float>> dete_thresholds;
        std::vector<float> nms_thresholds;
        std::vector<int> ost_ws;
        std::vector<int> ost_hs;
        std::vector<int> padding_lefts;
        std::vector<int> padding_tops;
        std::vector<int> padding_widths;
        std::vector<int> padding_heights;

        std::vector<sail::Tensor*> in_data = get_data(channel_idxs, image_idxs, dete_thresholds, nms_thresholds,
            ost_ws, ost_hs, padding_lefts, padding_tops, padding_widths, padding_heights);
        // reorder tensor
        std::sort(in_data.begin(), in_data.end(), [](sail::Tensor* a, sail::Tensor* b) {
            return a->shape()[3] > b->shape()[3];
        });
        if(channel_idxs.size() == 0){
            std::unique_lock<std::mutex> lck(mutex_pushdata_flag);
            pushdata_flag_cond.wait_for(lck,std::chrono::milliseconds(5));
            // SPDLOG_INFO("Get Data failed, sleeping for 5 ms");
            continue;
        }

        // nms paras 
        // int out_nout = input_use_multiclass_nms ? classes_ : 7;
        int max_wh = 7680;

        for (int index = 0; index < batch_size; ++index){
            int channel_idx = channel_idxs[index];
            int image_idx = image_idxs[index];
            std::vector<float> dete_threshold = dete_thresholds[index];
            float nms_threshold = nms_thresholds[index];
            int ost_w = ost_ws[index];
            int ost_h = ost_hs[index];
            int padding_left = padding_lefts[index];
            int padding_top = padding_tops[index];
            int padding_width = padding_widths[index];
            int padding_height = padding_heights[index];

            double time_start = get_current_time_us();
            float scale_w = (float)ost_w/padding_width;
            float scale_h = (float)ost_h/padding_height;
            float min_dete_threshold= *std::min_element(dete_threshold.begin(), dete_threshold.end());

            std::vector<DeteObjRect> dete_rects;
            for (int tidx = 0; tidx < in_data.size(); tidx++){
                float* data_ost = (float*)in_data[tidx]->sys_data();

                int feat_c = in_data[tidx]->shape()[1];
                int feat_h = in_data[tidx]->shape()[2];
                int feat_w = in_data[tidx]->shape()[3];
                int nout = in_data[tidx]->shape()[4];
                int area = feat_h * feat_w;
                assert(feat_c == anchor_num);
                int feature_size = feat_h * feat_w * nout;
                // SPDLOG_INFO("feat_c : {}, feat_h : {}, feat_w : {}, nout : {}, network_width : {}, network_height : {}", feat_c, feat_h, feat_w, nout, network_width, network_height);
                float* data = data_ost + index * feat_c * feature_size;
                for (int anchor_idx = 0; anchor_idx < anchor_num; anchor_idx++) {
                    float* ptr = data + anchor_idx * feature_size;
                    for (int i = 0; i < area; i++) {
                        ptr[0] = (sigmoid(ptr[0]) * 2 - 0.5 + i % feat_w) / feat_w * network_width;
                        ptr[1] = (sigmoid(ptr[1]) * 2 - 0.5 + i / feat_w) / feat_h * network_height;
                        ptr[2] = pow((sigmoid(ptr[2]) * 2), 2) * anchors[tidx][anchor_idx][0];
                        ptr[3] = pow((sigmoid(ptr[3]) * 2), 2) * anchors[tidx][anchor_idx][1];
                        float score = sigmoid(ptr[4]);
                        if (score > min_dete_threshold) {
                            for (int d = 5; d < nout; d++) {
                                ptr[d] = sigmoid(ptr[d]);
                            }
                            float *classes_scores = ptr + 5;
                            
                            if(input_use_multiclass_nms){
                                if(classes_ != dete_threshold.size() && dete_threshold.size() != 1){
                                    SPDLOG_ERROR("dete_threshold count Mismatch!");
                                    {
                                        std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
                                        exit_thread_flag = true;
                                    }
                                    return ;
                                }
                                for(int cls_id=0; cls_id<classes_; cls_id++){
                                    float dete_threshold_=dete_threshold[cls_id<dete_threshold.size()?cls_id:0];

                                    if(score * (*(classes_scores + cls_id)) > dete_threshold_){
                                        DeteObjRect dete_rect;
                                        dete_rect.score = score * (*(classes_scores + cls_id));
                                        dete_rect.class_id = cls_id;
                                        dete_rect.width = ptr[2];
                                        dete_rect.height = ptr[3];

                                        if(!agnostic){
                                            dete_rect.left = ptr[0] - 0.5 * ptr[2] + cls_id * max_wh;
                                            dete_rect.top = ptr[1] - 0.5 * ptr[3] +  + cls_id * max_wh;
                                        }else{
                                            dete_rect.left = ptr[0] - 0.5 * ptr[2];
                                            dete_rect.top = ptr[1] - 0.5 * ptr[3];
                                        }
                                        dete_rect.right = dete_rect.left + dete_rect.width;
                                        dete_rect.bottom = dete_rect.top + dete_rect.height;

                                        dete_rects.push_back(dete_rect);

                                    }
                                }
                            }else{
                                cv::Mat scores(1, classes_, CV_32FC1, classes_scores);
                                cv::Point class_id;
                                double max_class_score;
                                cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                                float dete_threshold_=dete_threshold[class_id.x<dete_threshold.size()?class_id.x:0];

                                DeteObjRect dete_rect;
                                if (score * max_class_score > dete_threshold_) {
                                    dete_rect.score = score * max_class_score;
                                    dete_rect.class_id = class_id.x;
                                    dete_rect.width = ptr[2];
                                    dete_rect.height = ptr[3];

                                    if(!agnostic){
                                        dete_rect.left = ptr[0] - 0.5 * ptr[2] + class_id.x * max_wh;
                                        dete_rect.top = ptr[1] - 0.5 * ptr[3] +  + class_id.x * max_wh;
                                    }else{
                                        dete_rect.left = ptr[0] - 0.5 * ptr[2];
                                        dete_rect.top = ptr[1] - 0.5 * ptr[3];
                                    }
                                    dete_rect.right = dete_rect.left + dete_rect.width;
                                    dete_rect.bottom = dete_rect.top + dete_rect.height;

                                    dete_rects.push_back(dete_rect);
                                }
                            }


                        }
                        ptr += nout;
                    }
                }
            }

            std::vector<DeteObjRect> dect_result;
            std::vector<int> picked;
            double time_num_start = get_current_time_us();
            sail_algo_nms_sorted_bboxes(dete_rects, picked, nms_threshold);
            double nms_time_use = (get_current_time_us() - time_num_start)/1000;

            if(!agnostic){
                for(auto& box:dete_rects){
                    box.left -= box.class_id * max_wh;
                    box.top -= box.class_id * max_wh;
                    box.right = box.left + box.width;
                    box.bottom = box.top + box.height;
                }
            }
            for (size_t i = 0; i < picked.size(); i++)    {
                DeteObjRect dete_rect;

                dete_rect.left = (dete_rects[picked[i]].left - padding_left) * scale_w;
                dete_rect.top = (dete_rects[picked[i]].top - padding_top) * scale_h;
                dete_rect.left = dete_rect.left > 0 ? dete_rect.left : 0;
                dete_rect.top = dete_rect.top > 0 ? dete_rect.top : 0;

                dete_rect.right = (dete_rects[picked[i]].right - padding_left) * scale_w;
                dete_rect.bottom = (dete_rects[picked[i]].bottom - padding_top) * scale_h;
                dete_rect.right = dete_rect.right < ost_w ? dete_rect.right : ost_w-1;
                dete_rect.bottom = dete_rect.bottom < ost_h ? dete_rect.bottom : ost_h-1;

                dete_rect.width = dete_rect.right - dete_rect.left;
                dete_rect.height = dete_rect.bottom - dete_rect.top;

                dete_rect.score = dete_rects[picked[i]].score;
                dete_rect.class_id = dete_rects[picked[i]].class_id;
                dect_result.push_back(dete_rect);
                // SPDLOG_INFO("dete_rect.left:{}, dete_rect.top:{}, dete_rect.right:{}, dete_rect.bottom:{}, dete_rect.score:{}, dete_rect.class_id:{}", dete_rect.left, dete_rect.top, dete_rect.right, dete_rect.bottom, dete_rect.score, dete_rect.class_id);
            }
            double time_use = (get_current_time_us() - time_start)/1000;
            // SPDLOG_INFO("dect_result num : {}",dect_result.size());  
            while (true) {
                int ret = push_result(dect_result, channel_idx, image_idx);
                if(ret == 0) {
                    notify_result_push();
                    break;
                }
                std::unique_lock<std::mutex> lck(mutex_result_pop);
                result_pop_cond.wait_for(lck,std::chrono::milliseconds(5));
                if(get_stop_flag()){
                    break;
                }  
            }
        }
        if(in_data[0]){
            for (int tidx = 0; tidx < in_data.size(); tidx++){
                delete in_data[tidx];
            }
        }
    }
    set_thread_exit();
    {
        std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
        exit_thread_flag = true;
    }
}

void algo_yolov5_post_3output::algo_yolov5_post_3output_cc::notify_data_push()
{
    std::unique_lock<std::mutex> lck(mutex_pushdata_flag);
    pushdata_flag_cond.notify_all();
}

void algo_yolov5_post_3output::algo_yolov5_post_3output_cc::notify_result_push()
{
    std::unique_lock<std::mutex> lck(mutex_result_push);
    result_push_cond.notify_all();
}

void algo_yolov5_post_3output::algo_yolov5_post_3output_cc::notify_result_pop()
{
    std::unique_lock<std::mutex> lck(mutex_result_pop);
    result_pop_cond.notify_all();
}

bool algo_yolov5_post_3output::algo_yolov5_post_3output_cc::get_stop_flag()
{
    std::lock_guard<std::mutex> lock(mutex_stop_);
    return stop_thread_flag;
}

void algo_yolov5_post_3output::algo_yolov5_post_3output_cc::set_stop_flag(bool flag)
{
    std::lock_guard<std::mutex> lock(mutex_stop_);
    stop_thread_flag = flag;
}

void algo_yolov5_post_3output::algo_yolov5_post_3output_cc::set_thread_exit()
{
    std::unique_lock<std::mutex> lck(mutex_exit_);
    exit_cond.notify_all();
}   

void algo_yolov5_post_3output::algo_yolov5_post_3output_cc::wait_thread_exit()
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

std::tuple<std::vector<DeteObjRect>,int,int> algo_yolov5_post_3output::algo_yolov5_post_3output_cc::get_result()
{
#ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    std::vector<DeteObjRect> results;
    int channel_idx = 0;
    int image_idx = 0;
    while(true){
        int ret = get_result_once(results, channel_idx, image_idx);
        if(ret == 0){
            break;
        }
        std::unique_lock<std::mutex> lck(mutex_result_push);
        result_push_cond.wait_for(lck,std::chrono::milliseconds(5));
        if(get_stop_flag()){
            break;
        } 
    }
    // SPDLOG_INFO("results num : {}",results.size()); 
    return std::make_tuple(std::move(results),channel_idx,image_idx);
}

algo_yolov5_post_3output::algo_yolov5_post_3output(const std::vector<std::vector<int>>& shape,int network_w, int network_h, int max_queue_size,bool input_use_multiclass_nms, bool agnostic)
:_impl (new algo_yolov5_post_3output_cc(shape,network_w,network_h,max_queue_size,input_use_multiclass_nms,agnostic))
{
}

algo_yolov5_post_3output::~algo_yolov5_post_3output()
{
    delete _impl;
}

int algo_yolov5_post_3output::reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new){
    return _impl->reset_anchors(anchors_new);
}   

int algo_yolov5_post_3output::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        std::vector<TensorPTRWithName> input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr)
{
    std::vector<sail::Tensor*> input_data_;
    for(int i=0;i<input_data.size();i++){
        input_data_.push_back(input_data[i].data);
        // SPDLOG_INFO(input_data[i].name);
    }
    return _impl->push_data(channel_idx,image_idx,input_data_,dete_threshold,nms_threshold, ost_w, ost_h, padding_attr);
}

int algo_yolov5_post_3output::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        std::vector<TensorPTRWithName> input_data, 
        std::vector<std::vector<float>> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr)
{
    std::vector<sail::Tensor*> input_data_;
    for(int i=0;i<input_data.size();i++){
        input_data_.push_back(input_data[i].data);
        // SPDLOG_INFO(input_data[i].name);
    }
    return _impl->push_data(channel_idx,image_idx,input_data_,dete_threshold,nms_threshold, ost_w, ost_h, padding_attr);
}

std::tuple<std::vector<DeteObjRect>,int,int> algo_yolov5_post_3output::get_result()
{
    return _impl->get_result();
}
std::tuple<std::vector<std::tuple<int, int, int, int ,int, float>>,int,int> algo_yolov5_post_3output::get_result_npy()
{
    auto result = _impl->get_result();
    std::vector<DeteObjRect> results = std::move(std::get<0>(result));
    int channel_idx = std::get<1>(result);
    int image_idx = std::get<2>(result);
    // SPDLOG_INFO("results num : {}",results.size());
    std::vector<std::tuple<int, int, int, int ,int, float>> objs;
    for (int i = 0; i < results.size(); ++i){
        int left_temp = results[i].left+0.5;
        int top_temp = results[i].top+0.5;
        int right_temp = results[i].right+0.5;
        int bottom_temp = results[i].bottom+0.5;
        int class_id_temp = results[i].class_id;
        float score_temp = results[i].score;

        objs.push_back(std::make_tuple(left_temp,
                                       top_temp,
                                       right_temp,
                                       bottom_temp,
                                       class_id_temp,
                                       score_temp));

    }
    return std::make_tuple(std::move(objs),channel_idx,image_idx);
}

// yolov5 post cpu opt async interface
class algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc{
public:
    explicit algo_yolov5_post_cpu_opt_async_cc(const std::vector<std::vector<int>>& shape, int network_w, int network_h, int max_queue_size, bool use_multiclass_nms);
    ~algo_yolov5_post_cpu_opt_async_cc();

    int push_data(std::vector<int> channel_idx, 
                    std::vector<int> image_idx, 
                    std::vector<sail::Tensor *> input_data, 
                    std::vector<float> dete_threshold,
                    std::vector<float> nms_threshold,
                    std::vector<int> ost_w,
                    std::vector<int> ost_h,
                    std::vector<std::vector<int>> padding_attr);

    int push_data(std::vector<int> channel_idx, 
                    std::vector<int> image_idx, 
                    std::vector<sail::Tensor *> input_data, 
                    std::vector<std::vector<float>> dete_threshold,
                    std::vector<float> nms_threshold,
                    std::vector<int> ost_w,
                    std::vector<int> ost_h,
                    std::vector<std::vector<int>> padding_attr);

    std::tuple<std::vector<DeteObjRect>,int,int> get_result();

    int reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new);

private:
    void post_thread();

    std::vector<sail::Tensor*> get_data(std::vector<int>& channel_idx, 
                std::vector<int>& image_idx, 
                std::vector<std::vector<float>> &dete_threshold, 
                std::vector<float> &nms_threshold,
                std::vector<int>& ost_w, 
                std::vector<int>& ost_h, 
                std::vector<int>& padding_left, 
                std::vector<int>& padding_top, 
                std::vector<int>& padding_width, 
                std::vector<int>& padding_height);

    void set_stop_flag(bool flag);  //设置线程退出的标志位

    bool get_stop_flag();           //获取线程退出的标志位

    void set_thread_exit();         //设置线程已经退出         

    void wait_thread_exit();        //等待线程退出

    void notify_data_push();        //

    void notify_result_push();       

    void notify_result_pop();        

    int get_result_once(std::vector<DeteObjRect>& result, int &channel_idx,int &image_idx);     //get detect only once, return 0 for success

    int push_result(std::vector<DeteObjRect> result, int channel_idx,int image_idx);            

    std::vector<std::vector<int>> data_shape_;   //
    int classes_;                   //
    int max_queue_size_;            //

    std::queue<int> channels_queue_;        //
    std::queue<int> image_idx_queue_;       //
    std::queue<std::vector<sail::Tensor*>> tensor_in_queue_;   //
    std::queue<std::vector<float>> dete_threshold_queue;   //
    std::queue<float> nms_threshold_queue;    //

    std::queue<int> ost_w_queue;            //
    std::queue<int> ost_h_queue;            //
    std::queue<int> padding_left_queue;     //
    std::queue<int> padding_top_queue;      //
    std::queue<int> padding_width_queue;    //
    std::queue<int> padding_height_queue;   //

    std::mutex mutex_data;

    std::queue<int> out_channels_queue_;                        //
    std::queue<int> out_image_idx_queue_;                       //
    std::queue<std::vector<DeteObjRect>> dete_result_queue_;    //
    std::mutex mutex_result;

    bool stop_thread_flag;      //线程退出的标志位
    std::mutex mutex_stop_;     //线程退出互斥锁
    std::condition_variable exit_cond;  //线程已经退出的信号量
    std::mutex mutex_exit_;             //线程已经退出互斥锁
    bool exit_thread_flag;      //线程已经退出的标志位

    std::condition_variable result_push_cond;       //后处理线程有结果送入结果队列的信号
    std::mutex mutex_result_push;

    std::condition_variable result_pop_cond;       //后处理线程有结果送入结果队列的信号
    std::mutex mutex_result_pop;

    std::condition_variable pushdata_flag_cond;     //有数据push进来的信号
    std::mutex mutex_pushdata_flag;

    bool post_thread_run;

    int network_width; //
    int network_height; //
    int batch_size; //

    std::vector<std::vector<std::vector<int>>> anchors;
    int anchor_num;

    bool use_multiclass_nms;
};

algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc::algo_yolov5_post_cpu_opt_async_cc(
    const std::vector<std::vector<int>>& shape,
    int network_w,
    int network_h,
    int max_queue_size,
    bool use_multiclass_nms)
:max_queue_size_(max_queue_size),post_thread_run(false),stop_thread_flag(false),exit_thread_flag(true),
network_width(network_w),network_height(network_h),use_multiclass_nms(use_multiclass_nms)
{
    if (shape.size() != 3){
        SPDLOG_ERROR("ERROR outputs num, 3 vs. {}!",shape.size());
        throw SailRuntimeError("invalid argument");
    }
    if (shape[0].size() != 5){
        SPDLOG_ERROR("ERROR DIM, 5 vs. {}!",shape[0].size());
        throw SailRuntimeError("invalid argument");
    }
    
    for(int i=0;i<shape.size();++i){
        std::vector<int> data_shape;
        for (int j=0;j<shape[0].size();++j){
            data_shape.push_back(shape[i][j]);
        }
        data_shape_.push_back(data_shape);
    }

    classes_ = data_shape_[0][4]-5;
    batch_size = shape[0][0];

    anchors = {{{10, 13}, {16, 30}, {33, 23}}, {{30, 61}, {62, 45}, {59, 119}}, {{116, 90}, {156, 198}, {373, 326}}};
    anchor_num = anchors[0].size();
}

algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc::~algo_yolov5_post_cpu_opt_async_cc()
{
    set_stop_flag(true);
    SPDLOG_INFO("Start Set Thread Exit Flag!");
    set_thread_exit();
    SPDLOG_INFO("End Set Thread Exit Flag, Waiting Thread Exit....");
    wait_thread_exit();
}

int algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc::reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new){
#ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    anchors.clear();
    for(int i=0;i<anchors_new.size();++i){
        std::vector<std::vector<int>> anchor_;
        for (int j=0;j<anchors_new[0].size();++j){
            anchor_.push_back({anchors_new[i][j][0], anchors_new[i][j][1]});
        }
        anchors.push_back(anchor_);
    }
    anchor_num = anchors[0].size();
    SPDLOG_INFO("Reset Anchors, anchor_num:{} .", anchor_num);
    return 0;
}

int algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        std::vector<sail::Tensor*> input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr){
 #ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    if(channel_idx.size() != batch_size ||
        image_idx.size() != batch_size ||
        input_data[0]->shape()[0] != batch_size ||
        dete_threshold.size() != batch_size ||
        padding_attr.size() != batch_size ||
        ost_w.size() != batch_size ||
        ost_h.size() != batch_size ||
        nms_threshold.size() != batch_size){
        SPDLOG_ERROR("Input shape {},{},{},{},{}",input_data[0]->shape()[0],input_data[0]->shape()[1],input_data[0]->shape()[2],input_data[0]->shape()[3],input_data[0]->shape()[4]);
        SPDLOG_ERROR("Input batch size mismatch, {} vs. {}, {}, {}, {}, {}, {}, {}, {}",
            batch_size, channel_idx.size(), image_idx.size(), input_data[0]->shape()[0], dete_threshold.size(),padding_attr.size(),
            ost_w.size(), ost_h.size(), nms_threshold.size());
        return SAIL_ALGO_ERROR_BATCHSIZE;
    }
    std::lock_guard<std::mutex> lock(mutex_data);
    if(tensor_in_queue_.size() >= max_queue_size_) {
        return SAIL_ALGO_BUFFER_FULL;
    }
    for(int i=0; i <batch_size;++i) {
        channels_queue_.push(channel_idx[i]);
        image_idx_queue_.push(image_idx[i]);
        dete_threshold_queue.push({dete_threshold[i]});
        nms_threshold_queue.push(nms_threshold[i]);
        ost_w_queue.push(ost_w[i]);                    //
        ost_h_queue.push(ost_h[i]);                    //
        padding_left_queue.push(padding_attr[i][0]);      //
        padding_top_queue.push(padding_attr[i][1]);        //
        padding_width_queue.push(padding_attr[i][2]);    //
        padding_height_queue.push(padding_attr[i][3]);  //
    }
    tensor_in_queue_.push(std::move(input_data));

    if(!post_thread_run){
        std::thread thread_post = std::thread(&algo_yolov5_post_cpu_opt_async_cc::post_thread,this);
        thread_post.detach();
        post_thread_run = true;
    }
    notify_data_push();
    return SAIL_ALGO_SUCCESS;
};

int algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        std::vector<sail::Tensor*> input_data, 
        std::vector<std::vector<float>> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr){
 #ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    if(channel_idx.size() != batch_size ||
        image_idx.size() != batch_size ||
        input_data[0]->shape()[0] != batch_size ||
        dete_threshold.size() != batch_size ||
        padding_attr.size() != batch_size ||
        ost_w.size() != batch_size ||
        ost_h.size() != batch_size ||
        nms_threshold.size() != batch_size){
        SPDLOG_ERROR("Input shape {},{},{},{},{}",input_data[0]->shape()[0],input_data[0]->shape()[1],input_data[0]->shape()[2],input_data[0]->shape()[3],input_data[0]->shape()[4]);
        SPDLOG_ERROR("Input batch size mismatch, {} vs. {}, {}, {}, {}, {}, {}, {}, {}",
            batch_size, channel_idx.size(), image_idx.size(), input_data[0]->shape()[0], dete_threshold.size(),padding_attr.size(),
            ost_w.size(), ost_h.size(), nms_threshold.size());
        return SAIL_ALGO_ERROR_BATCHSIZE;
    }
    std::lock_guard<std::mutex> lock(mutex_data);
    if(tensor_in_queue_.size() >= max_queue_size_) {
        return SAIL_ALGO_BUFFER_FULL;
    }
    for(int i=0; i <batch_size;++i) {
        channels_queue_.push(channel_idx[i]);
        image_idx_queue_.push(image_idx[i]);
        dete_threshold_queue.push(dete_threshold[i]);
        nms_threshold_queue.push(nms_threshold[i]);
        ost_w_queue.push(ost_w[i]);                    //
        ost_h_queue.push(ost_h[i]);                    //
        padding_left_queue.push(padding_attr[i][0]);      //
        padding_top_queue.push(padding_attr[i][1]);        //
        padding_width_queue.push(padding_attr[i][2]);    //
        padding_height_queue.push(padding_attr[i][3]);  //
    }
    tensor_in_queue_.push(std::move(input_data));

    if(!post_thread_run){
        std::thread thread_post = std::thread(&algo_yolov5_post_cpu_opt_async_cc::post_thread,this);
        thread_post.detach();
        post_thread_run = true;
    }
    notify_data_push();
    return SAIL_ALGO_SUCCESS;
};


std::vector<sail::Tensor*> algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc::get_data(
        std::vector<int>& channel_idx, 
        std::vector<int>& image_idx, 
        std::vector<std::vector<float>> &dete_threshold, 
        std::vector<float> &nms_threshold,
        std::vector<int>& ost_w, 
        std::vector<int>& ost_h, 
        std::vector<int>& padding_left, 
        std::vector<int>& padding_top, 
        std::vector<int>& padding_width, 
        std::vector<int>& padding_height){
    std::vector<sail::Tensor*> input_data; 
    std::lock_guard<std::mutex> lock(mutex_data);
    if(tensor_in_queue_.empty()) {
        // SPDLOG_INFO("tensor_in_queue_ is empty");
        return input_data;
    }
    channel_idx.clear();
    image_idx.clear(); 
    dete_threshold.clear();
    nms_threshold.clear();
    ost_w.clear();
    ost_h.clear();
    padding_left.clear();
    padding_top.clear();
    padding_width.clear();
    padding_height.clear();
    for(int i=0;i<batch_size;++i){
        channel_idx.push_back(channels_queue_.front());
        image_idx.push_back(image_idx_queue_.front());
        dete_threshold.push_back(dete_threshold_queue.front());
        nms_threshold.push_back(nms_threshold_queue.front());

        ost_w.push_back(ost_w_queue.front());                    //
        ost_h.push_back(ost_h_queue.front());                    //
        padding_left.push_back(padding_left_queue.front());      //
        padding_top.push_back(padding_top_queue.front());        //
        padding_width.push_back(padding_width_queue.front());    //
        padding_height.push_back(padding_height_queue.front());  //

        channels_queue_.pop();
        image_idx_queue_.pop();
        dete_threshold_queue.pop();
        nms_threshold_queue.pop();
        ost_w_queue.pop();                    //
        ost_h_queue.pop();                    //
        padding_left_queue.pop();      //
        padding_top_queue.pop();        //
        padding_width_queue.pop();    //
        padding_height_queue.pop();  // 
    }

    input_data = tensor_in_queue_.front();
    tensor_in_queue_.pop(); //

    return std::move(input_data);
}

int algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc::get_result_once(std::vector<DeteObjRect>& result, int &channel_idx,int &image_idx)
{
    std::lock_guard<std::mutex> lock(mutex_result);
    if(dete_result_queue_.empty()){
        return 1;
    }
    channel_idx = out_channels_queue_.front();
    image_idx = out_image_idx_queue_.front();
    std::vector<DeteObjRect> temp = dete_result_queue_.front();
    for(int i=0;i<temp.size();++i){
        result.push_back(temp[i]);
    }
    out_channels_queue_.pop();
    out_image_idx_queue_.pop();
    dete_result_queue_.pop();

    notify_result_pop();    
    return 0;
}

int algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc::push_result(std::vector<DeteObjRect> result, int channel_idx,int image_idx)
{
    std::lock_guard<std::mutex> lock(mutex_result);
    if(dete_result_queue_.size() >= max_queue_size_){
        return 1;
    }
    out_channels_queue_.push(channel_idx);
    out_image_idx_queue_.push(image_idx);
    dete_result_queue_.push(std::move(result));
    return 0;
}

void algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc::post_thread()
{
    {        
        std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
        exit_thread_flag = false;
    }
    // SPDLOG_INFO("Create To YOLOV5 1 output Thread, pid:{}, tid:{} .",getpid(),gettid());
    int idx = 0;
    while(true) {
        if(get_stop_flag()){
            break;
        }  
        std::vector<int> channel_idxs;
        std::vector<int> image_idxs;
        std::vector<std::vector<float>> dete_thresholds;
        std::vector<float> nms_thresholds;
        std::vector<int> ost_ws;
        std::vector<int> ost_hs;
        std::vector<int> padding_lefts;
        std::vector<int> padding_tops;
        std::vector<int> padding_widths;
        std::vector<int> padding_heights;

        std::vector<sail::Tensor*> in_data = get_data(channel_idxs, image_idxs, dete_thresholds, nms_thresholds,
            ost_ws, ost_hs, padding_lefts, padding_tops, padding_widths, padding_heights);
        // reorder tensor
        std::sort(in_data.begin(), in_data.end(), [](sail::Tensor* a, sail::Tensor* b) {
            return a->shape()[3] > b->shape()[3];
        });
        if(channel_idxs.size() == 0){
            std::unique_lock<std::mutex> lck(mutex_pushdata_flag);
            pushdata_flag_cond.wait_for(lck,std::chrono::milliseconds(5));
            // SPDLOG_INFO("Get Data failed, sleeping for 5 ms");
            continue;
        }

        int max_wh = 7680;
        for (int index = 0; index < batch_size; ++index){
            int channel_idx = channel_idxs[index];
            int image_idx = image_idxs[index];
            std::vector<float> dete_threshold = dete_thresholds[index];
            float nms_threshold = nms_thresholds[index];
            float min_dete_threshold= *std::min_element(dete_threshold.begin(), dete_threshold.end());

            float min_opposite_log_reciprocal_m_confThreshold_sub_one = - std::log(1 / min_dete_threshold - 1);
            int ost_w = ost_ws[index];
            int ost_h = ost_hs[index];
            int padding_left = padding_lefts[index];
            int padding_top = padding_tops[index];
            int padding_width = padding_widths[index];
            int padding_height = padding_heights[index];

            double time_start = get_current_time_us();
            float scale_w = (float)ost_w/padding_width;
            float scale_h = (float)ost_h/padding_height;

            std::vector<DeteObjRect> dete_rects;
            for (int tidx = 0; tidx < in_data.size(); tidx++){
                float* data_ost = (float*)in_data[tidx]->sys_data();

                int feat_c = in_data[tidx]->shape()[1];
                int feat_h = in_data[tidx]->shape()[2];
                int feat_w = in_data[tidx]->shape()[3];
                int nout = in_data[tidx]->shape()[4];
                int area = feat_h * feat_w;
                assert(feat_c == anchor_num);
                int feature_size = feat_h * feat_w * nout;
                // SPDLOG_INFO("feat_c : {}, feat_h : {}, feat_w : {}, nout : {}, network_width : {}, network_height : {}", feat_c, feat_h, feat_w, nout, network_width, network_height);
                float* data = data_ost + index * feat_c * feature_size;

                for (int anchor_idx = 0; anchor_idx < anchor_num; anchor_idx++) {
                    float* ptr = data + anchor_idx * feature_size;
                    for (int i = 0; i < area; i++) {
                        if(ptr[4] <= min_opposite_log_reciprocal_m_confThreshold_sub_one){
                            ptr += nout;
                            continue;
                        }
                        ptr[0] = (sigmoid(ptr[0]) * 2 - 0.5 + i % feat_w) / feat_w * network_width;
                        ptr[1] = (sigmoid(ptr[1]) * 2 - 0.5 + i / feat_w) / feat_h * network_height;
                        ptr[2] = pow((sigmoid(ptr[2]) * 2), 2) * anchors[tidx][anchor_idx][0];
                        ptr[3] = pow((sigmoid(ptr[3]) * 2), 2) * anchors[tidx][anchor_idx][1];
                        float score = sigmoid(ptr[4]);
                        // for single class nms
                        if (!use_multiclass_nms) {
                            float *classes_scores = ptr + 5;
                            cv::Mat scores(1, classes_, CV_32FC1, classes_scores);
                            cv::Point class_id;
                            double max_class_score;
                            float dete_threshold_=dete_threshold[class_id.x<dete_threshold.size()?class_id.x:0];
                            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                            DeteObjRect dete_rect;
                            float box_transformed_m_confThreshold = - std::log(score / dete_threshold_ - 1);
                            if (max_class_score > box_transformed_m_confThreshold) {
                                dete_rect.score = score * sigmoid(max_class_score);
                                dete_rect.class_id = class_id.x;
                                dete_rect.left = ptr[0] - 0.5 * ptr[2];
                                dete_rect.top = ptr[1] - 0.5 * ptr[3];
                                dete_rect.right = ptr[0] + 0.5 * ptr[2];
                                dete_rect.bottom = ptr[1] + 0.5 * ptr[3];
                                dete_rect.width = ptr[2];
                                dete_rect.height = ptr[3];

                                dete_rects.push_back(dete_rect);
                            }
                        }
                        // for multi class nms
                        else {
                            float *classes_scores = ptr + 5;
                            if(classes_ != dete_threshold.size() && dete_threshold.size() != 1){
                                SPDLOG_ERROR("dete_threshold count Mismatch!");
                                {
                                    std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
                                    exit_thread_flag = true;
                                }
                                return ;
                            }
                            for (int offset = 0; offset < classes_; offset++) {
                                float box_transformed_m_confThreshold = - std::log(score / dete_threshold[offset < dete_threshold.size()?offset:0] - 1);
                                if (*(classes_scores + offset) > box_transformed_m_confThreshold) {
                                    DeteObjRect dete_rect;
                                    dete_rect.left = ptr[0] - 0.5 * ptr[2] + offset * max_wh;
                                    dete_rect.top = ptr[1] - 0.5 * ptr[3] + offset * max_wh;
                                    dete_rect.right = ptr[0] + 0.5 * ptr[2] + offset * max_wh;
                                    dete_rect.bottom = ptr[1] + 0.5 * ptr[3] + offset * max_wh;
                                    dete_rect.width = ptr[2];
                                    dete_rect.height = ptr[3];
                                    dete_rect.class_id = offset;
                                    dete_rect.score = score * sigmoid(*(classes_scores + offset));
                                    dete_rects.push_back(dete_rect);
                                }
                            }
                        }
                        ptr += nout;
                    }
                }
            }

            std::vector<DeteObjRect> dect_result;
            std::vector<int> picked;
            double time_num_start = get_current_time_us();
            sail_algo_nms_sorted_bboxes(dete_rects, picked, nms_threshold);
            double nms_time_use = (get_current_time_us() - time_num_start)/1000;
            for (size_t i = 0; i < picked.size(); i++)    {
                DeteObjRect dete_rect;

                if (use_multiclass_nms) {
                    dete_rects[picked[i]].left -= dete_rects[picked[i]].class_id * max_wh;
                    dete_rects[picked[i]].top -= dete_rects[picked[i]].class_id * max_wh;
                    dete_rects[picked[i]].right -= dete_rects[picked[i]].class_id * max_wh;
                    dete_rects[picked[i]].bottom -= dete_rects[picked[i]].class_id * max_wh;
                }
                dete_rect.left = (dete_rects[picked[i]].left - padding_left) * scale_w;
                dete_rect.top = (dete_rects[picked[i]].top - padding_top) * scale_h;
                dete_rect.left = dete_rect.left > 0 ? dete_rect.left : 0;
                dete_rect.top = dete_rect.top > 0 ? dete_rect.top : 0;

                dete_rect.right = (dete_rects[picked[i]].right - padding_left) * scale_w;
                dete_rect.bottom = (dete_rects[picked[i]].bottom - padding_top) * scale_h;
                dete_rect.right = dete_rect.right < ost_w ? dete_rect.right : ost_w-1;
                dete_rect.bottom = dete_rect.bottom < ost_h ? dete_rect.bottom : ost_h-1;

                dete_rect.width = dete_rect.right - dete_rect.left;
                dete_rect.height = dete_rect.bottom - dete_rect.top;

                dete_rect.score = dete_rects[picked[i]].score;
                dete_rect.class_id = dete_rects[picked[i]].class_id;
                dect_result.push_back(dete_rect);
                // SPDLOG_INFO("dete_rect.left:{}, dete_rect.top:{}, dete_rect.right:{}, dete_rect.bottom:{}, dete_rect.score:{}, dete_rect.class_id:{}", dete_rect.left, dete_rect.top, dete_rect.right, dete_rect.bottom, dete_rect.score, dete_rect.class_id);
            }
            double time_use = (get_current_time_us() - time_start)/1000;
            // SPDLOG_INFO("dect_result num : {}",dect_result.size());  
            while (true) {
                int ret = push_result(dect_result, channel_idx, image_idx);
                if(ret == 0) {
                    notify_result_push();
                    break;
                }
                std::unique_lock<std::mutex> lck(mutex_result_pop);
                result_pop_cond.wait_for(lck,std::chrono::milliseconds(5));
                if(get_stop_flag()){
                    break;
                }  
            }
        }
        if(in_data[0]){
            for (int tidx = 0; tidx < in_data.size(); tidx++){
                delete in_data[tidx];
            }
        }
    }
    set_thread_exit();
    {
        std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
        exit_thread_flag = true;
    }
}

void algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc::notify_data_push()
{
    std::unique_lock<std::mutex> lck(mutex_pushdata_flag);
    pushdata_flag_cond.notify_all();
}

void algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc::notify_result_push()
{
    std::unique_lock<std::mutex> lck(mutex_result_push);
    result_push_cond.notify_all();
}

void algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc::notify_result_pop()
{
    std::unique_lock<std::mutex> lck(mutex_result_pop);
    result_pop_cond.notify_all();
}

bool algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc::get_stop_flag()
{
    std::lock_guard<std::mutex> lock(mutex_stop_);
    return stop_thread_flag;
}

void algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc::set_stop_flag(bool flag)
{
    std::lock_guard<std::mutex> lock(mutex_stop_);
    stop_thread_flag = flag;
}

void algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc::set_thread_exit()
{
    std::unique_lock<std::mutex> lck(mutex_exit_);
    exit_cond.notify_all();
}   

void algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc::wait_thread_exit()
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

std::tuple<std::vector<DeteObjRect>,int,int> algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async_cc::get_result()
{
#ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    std::vector<DeteObjRect> results;
    int channel_idx = 0;
    int image_idx = 0;
    while(true){
        int ret = get_result_once(results, channel_idx, image_idx);
        if(ret == 0){
            break;
        }
        std::unique_lock<std::mutex> lck(mutex_result_push);
        result_push_cond.wait_for(lck,std::chrono::milliseconds(5));
        if(get_stop_flag()){
            break;
        } 
    }
    // SPDLOG_INFO("results num : {}",results.size()); 
    return std::make_tuple(std::move(results),channel_idx,image_idx);
}

algo_yolov5_post_cpu_opt_async::algo_yolov5_post_cpu_opt_async(const std::vector<std::vector<int>>& shape,int network_w, int network_h, int max_queue_size, bool use_multiclass_nms)
:_impl (new algo_yolov5_post_cpu_opt_async_cc(shape,network_w,network_h,max_queue_size, use_multiclass_nms))
{
}

algo_yolov5_post_cpu_opt_async::~algo_yolov5_post_cpu_opt_async()
{
    delete _impl;
}

int algo_yolov5_post_cpu_opt_async::reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new){
    return _impl->reset_anchors(anchors_new);
}   

int algo_yolov5_post_cpu_opt_async::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        std::vector<TensorPTRWithName> input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr)
{
    std::vector<sail::Tensor*> input_data_;
    for(int i=0;i<input_data.size();i++){
        input_data_.push_back(input_data[i].data);
        // SPDLOG_INFO(input_data[i].name);
    }
    return _impl->push_data(channel_idx,image_idx,input_data_,dete_threshold,nms_threshold, ost_w, ost_h, padding_attr);
}
int algo_yolov5_post_cpu_opt_async::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        std::vector<TensorPTRWithName> input_data, 
        std::vector<std::vector<float>> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr)
{
    std::vector<sail::Tensor*> input_data_;
    for(int i=0;i<input_data.size();i++){
        input_data_.push_back(input_data[i].data);
        // SPDLOG_INFO(input_data[i].name);
    }
    return _impl->push_data(channel_idx,image_idx,input_data_,dete_threshold,nms_threshold, ost_w, ost_h, padding_attr);
}
std::tuple<std::vector<DeteObjRect>,int,int> algo_yolov5_post_cpu_opt_async::get_result()
{
    return _impl->get_result();
}
std::tuple<std::vector<std::tuple<int, int, int, int ,int, float>>,int,int> algo_yolov5_post_cpu_opt_async::get_result_npy()
{
    auto result = _impl->get_result();
    std::vector<DeteObjRect> results = std::move(std::get<0>(result));
    int channel_idx = std::get<1>(result);
    int image_idx = std::get<2>(result);
    // SPDLOG_INFO("results num : {}",results.size());
    std::vector<std::tuple<int, int, int, int ,int, float>> objs;
    for (int i = 0; i < results.size(); ++i){
        int left_temp = results[i].left+0.5;
        int top_temp = results[i].top+0.5;
        int right_temp = results[i].right+0.5;
        int bottom_temp = results[i].bottom+0.5;
        int class_id_temp = results[i].class_id;
        float score_temp = results[i].score;

        objs.push_back(std::make_tuple(left_temp,
                                       top_temp,
                                       right_temp,
                                       bottom_temp,
                                       class_id_temp,
                                       score_temp));

    }
    return std::make_tuple(std::move(objs),channel_idx,image_idx);
}

// 新增YOLOX后处理接口，多线程方式实现
class algo_yolox_post::algo_yolox_post_cc{
public:
    explicit algo_yolox_post_cc(const std::vector<int>& shape, int network_w, int network_h, int max_queue_size);
    ~algo_yolox_post_cc();

    int push_data(std::vector<int> channel_idx, 
                    std::vector<int> image_idx, 
                    sail::Tensor* input_data, 
                    std::vector<float> dete_threshold,
                    std::vector<float> nms_threshold,
                    std::vector<int> ost_w,
                    std::vector<int> ost_h,
                    std::vector<std::vector<int>> padding_attr);

    int push_data(std::vector<int> channel_idx, 
                    std::vector<int> image_idx, 
                    float* input_data, 
                    std::vector<float> dete_threshold,
                    std::vector<float> nms_threshold,
                    std::vector<int> ost_w,
                    std::vector<int> ost_h,
                    std::vector<std::vector<int>> padding_attr);

    std::tuple<std::vector<DeteObjRect>,int,int> get_result();

    int get_buffer_size(){
        return buffer_size_;
    }

private:
    void post_thread();

    sail::Tensor* get_data(std::vector<int>& channel_idx, 
                std::vector<int>& image_idx, 
                std::vector<float> &dete_threshold, 
                std::vector<float> &nms_threshold,
                std::vector<int>& ost_w, 
                std::vector<int>& ost_h, 
                std::vector<int>& padding_left, 
                std::vector<int>& padding_top, 
                std::vector<int>& padding_width, 
                std::vector<int>& padding_height);

    void set_stop_flag(bool flag);  //设置线程退出的标志位

    bool get_stop_flag();           //获取线程退出的标志位

    void set_thread_exit();         //设置线程已经退出         

    void wait_thread_exit();        //等待线程退出

    void notify_data_push();        //

    void notify_result_push();       

    void notify_result_pop();        

    int get_result_once(std::vector<DeteObjRect>& result, int &channel_idx,int &image_idx);     //get detect only once, return 0 for success

    int push_result(std::vector<DeteObjRect> result, int channel_idx,int image_idx);     

    int argmax(float* data, int num);       

    std::vector<int> data_shape_;   //
    int classes_;                   //
    int max_queue_size_;            //
    int buffer_size_;               //

    std::queue<int> channels_queue_;        //
    std::queue<int> image_idx_queue_;       //
    std::queue<sail::Tensor*> tensor_in_queue_;   //
    std::queue<float> dete_threshold_queue;   //
    std::queue<float> nms_threshold_queue;    //

    std::queue<int> ost_w_queue;            //
    std::queue<int> ost_h_queue;            //
    std::queue<int> padding_left_queue;     //
    std::queue<int> padding_top_queue;      //
    std::queue<int> padding_width_queue;    //
    std::queue<int> padding_height_queue;   //

    std::mutex mutex_data;

    std::queue<int> out_channels_queue_;                        //
    std::queue<int> out_image_idx_queue_;                       //
    std::queue<std::vector<DeteObjRect>> dete_result_queue_;    //
    std::mutex mutex_result;

    bool stop_thread_flag;      //线程退出的标志位
    std::mutex mutex_stop_;     //线程退出互斥锁
    std::condition_variable exit_cond;  //线程已经退出的信号量
    std::mutex mutex_exit_;             //线程已经退出互斥锁
    bool exit_thread_flag;      //线程已经退出的标志位

    std::condition_variable result_push_cond;       //后处理线程有结果送入结果队列的信号
    std::mutex mutex_result_push;

    std::condition_variable result_pop_cond;       //后处理线程有结果送入结果队列的信号
    std::mutex mutex_result_pop;

    std::condition_variable pushdata_flag_cond;     //有数据push进来的信号
    std::mutex mutex_pushdata_flag;

    bool post_thread_run;

    int network_width; //
    int network_height; //
    int batch_size; //
    int p6; //
};

algo_yolox_post::algo_yolox_post_cc::algo_yolox_post_cc(
    const std::vector<int>& shape,
    int network_w,
    int network_h,
    int max_queue_size)
:max_queue_size_(max_queue_size),post_thread_run(false),stop_thread_flag(false),exit_thread_flag(true),
network_width(network_w),network_height(network_h),p6(false)
{
    if (shape.size() != 3){
        SPDLOG_ERROR("ERROR DIMS, 3 vs. {}!",shape.size());
        throw SailRuntimeError("invalid argument");
    }
    for(int i=0;i<shape.size();++i)    {
        data_shape_.push_back(shape[i]);
    }
    buffer_size_ = 1;
    for(int i=1;i<shape.size();++i)    {
        buffer_size_ = buffer_size_ * shape[i];
    }
    classes_ = data_shape_[2]-5;
    batch_size = shape[0];
}

algo_yolox_post::algo_yolox_post_cc::~algo_yolox_post_cc()
{
    set_stop_flag(true);
    SPDLOG_INFO("Start Set Thread Exit Flag!");
    set_thread_exit();
    SPDLOG_INFO("End Set Thread Exit Flag, Waiting Thread Exit....");
    wait_thread_exit();
}

int algo_yolox_post::algo_yolox_post_cc::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        sail::Tensor* input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr){
#ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    if(channel_idx.size() != batch_size ||
        image_idx.size() != batch_size ||
        input_data->shape()[0] != batch_size ||
        dete_threshold.size() != batch_size ||
        padding_attr.size() != batch_size ||
        ost_w.size() != batch_size ||
        ost_h.size() != batch_size ||
        nms_threshold.size() != batch_size){
        SPDLOG_ERROR("Input shape {},{},{}",input_data->shape()[0],input_data->shape()[1],input_data->shape()[2]);
        SPDLOG_ERROR("Input batch size mismatch, {} vs. {}, {}, {}, {}, {}, {}, {}, {}",
            batch_size, channel_idx.size(), image_idx.size(), input_data->shape()[0], dete_threshold.size(),padding_attr.size(),
            ost_w.size(), ost_h.size(), nms_threshold.size());
        return SAIL_ALGO_ERROR_BATCHSIZE;
    }

    if(data_shape_.size() != input_data->shape().size()){
        SPDLOG_ERROR("The shape of the pushed data is incorrect!");
        return SAIL_ALGO_ERROR_SHAPES;
    }else{
        for(int i=0;i<data_shape_.size();i++){
            if(data_shape_[i]!=input_data->shape()[i]){
                SPDLOG_ERROR("The shape of the pushed data is incorrect!");
                return SAIL_ALGO_ERROR_SHAPES;
            }
            else
                continue;
        }
    }
    
    std::lock_guard<std::mutex> lock(mutex_data);
    if(tensor_in_queue_.size() >= max_queue_size_) {
        return SAIL_ALGO_BUFFER_FULL;
    }
    for(int i=0; i <batch_size;++i) {
        channels_queue_.push(channel_idx[i]);
        image_idx_queue_.push(image_idx[i]);
        dete_threshold_queue.push(dete_threshold[i]);
        nms_threshold_queue.push(nms_threshold[i]);
        ost_w_queue.push(ost_w[i]);                    //
        ost_h_queue.push(ost_h[i]);                    //
        padding_left_queue.push(padding_attr[i][0]);   //
        padding_top_queue.push(padding_attr[i][1]);    //
        padding_width_queue.push(padding_attr[i][2]);  //
        padding_height_queue.push(padding_attr[i][3]); //
    }
    tensor_in_queue_.push(std::move(input_data));

    if(!post_thread_run){
        std::thread thread_post = std::thread(&algo_yolox_post_cc::post_thread,this);
        thread_post.detach();
        post_thread_run = true;
    }
    notify_data_push();
    return SAIL_ALGO_SUCCESS;
};

sail::Tensor* algo_yolox_post::algo_yolox_post_cc::get_data(
        std::vector<int>& channel_idx, 
        std::vector<int>& image_idx, 
        std::vector<float> &dete_threshold, 
        std::vector<float> &nms_threshold,
        std::vector<int>& ost_w, 
        std::vector<int>& ost_h, 
        std::vector<int>& padding_left, 
        std::vector<int>& padding_top, 
        std::vector<int>& padding_width, 
        std::vector<int>& padding_height){
    sail::Tensor* input_data = NULL; 
    std::lock_guard<std::mutex> lock(mutex_data);
    if(tensor_in_queue_.empty()) {
        return input_data;
    }
    channel_idx.clear();
    image_idx.clear(); 
    dete_threshold.clear();
    nms_threshold.clear();
    ost_w.clear();
    ost_h.clear();
    padding_left.clear();
    padding_top.clear();
    padding_width.clear();
    padding_height.clear();
    for(int i=0;i<batch_size;++i){
        channel_idx.push_back(channels_queue_.front());
        image_idx.push_back(image_idx_queue_.front());
        dete_threshold.push_back(dete_threshold_queue.front());
        nms_threshold.push_back(nms_threshold_queue.front());

        ost_w.push_back(ost_w_queue.front());                    //
        ost_h.push_back(ost_h_queue.front());                    //
        padding_left.push_back(padding_left_queue.front());      //
        padding_top.push_back(padding_top_queue.front());        //
        padding_width.push_back(padding_width_queue.front());    //
        padding_height.push_back(padding_height_queue.front());  //

        channels_queue_.pop();
        image_idx_queue_.pop();
        dete_threshold_queue.pop();
        nms_threshold_queue.pop();
        ost_w_queue.pop();                    //
        ost_h_queue.pop();                    //
        padding_left_queue.pop();      //
        padding_top_queue.pop();        //
        padding_width_queue.pop();    //
        padding_height_queue.pop();  // 
    }

    input_data = tensor_in_queue_.front();
    tensor_in_queue_.pop(); //

    return std::move(input_data);
}

int algo_yolox_post::algo_yolox_post_cc::get_result_once(std::vector<DeteObjRect>& result, int &channel_idx,int &image_idx)
{
    std::lock_guard<std::mutex> lock(mutex_result);
    if(dete_result_queue_.empty()){
        return 1;
    }
    channel_idx = out_channels_queue_.front();
    image_idx = out_image_idx_queue_.front();
    std::vector<DeteObjRect> temp = dete_result_queue_.front();
    for(int i=0;i<temp.size();++i){
        result.push_back(temp[i]);
    }
    out_channels_queue_.pop();
    out_image_idx_queue_.pop();
    dete_result_queue_.pop();

    notify_result_pop();    
    return 0;
}

int algo_yolox_post::algo_yolox_post_cc::push_result(std::vector<DeteObjRect> result, int channel_idx,int image_idx)
{
    std::lock_guard<std::mutex> lock(mutex_result);
    if(dete_result_queue_.size() >= max_queue_size_){
        return 1;
    }
    out_channels_queue_.push(channel_idx);
    out_image_idx_queue_.push(image_idx);
    dete_result_queue_.push(std::move(result));
    return 0;
}

void algo_yolox_post::algo_yolox_post_cc::post_thread()
{
    {        
        std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
        exit_thread_flag = false;
    }
    // SPDLOG_INFO("Create To YOLOX Thread, pid:{}, tid:{} .",getpid(),gettid());
    // init grids 
    std::vector<int> strides = {8,16,32};
    if (p6){
        strides.push_back(64);
    }
    // get grids
    int outlen_diml = 0 ;
    for (int i =0;i<strides.size();++i){
        int layer_w = network_width / strides[i];
        int layer_h = network_height / strides[i];
        outlen_diml += layer_h * layer_w;
    }
    int* grids_x_          = new int[outlen_diml];
    int* grids_y_          = new int[outlen_diml];
    int* expanded_strides_ = new int[outlen_diml];
    int channel_len = 0;
    for (int i=0;i<strides.size();++i){
        int layer_w = network_width / strides[i];
        int layer_h = network_height / strides[i];
        for (int m = 0; m < layer_h; ++m){
            for (int n = 0; n < layer_w; ++n){
                grids_x_[channel_len + m * layer_w + n] = n;
                grids_y_[channel_len + m * layer_w + n] = m;
                expanded_strides_[channel_len + m * layer_w + n] = strides[i];
            }
        }
        channel_len += layer_w * layer_h;
    }
    // get idx of input imgs
    int idx = 0;
    while(true) {
        if(get_stop_flag()){
            // delete
            delete[] grids_x_;
            grids_x_ = nullptr;
            delete[] grids_y_;
            grids_y_ = nullptr;
            delete[] expanded_strides_;
            expanded_strides_ = nullptr;

            break;
        }  
        std::vector<int> channel_idxs;
        std::vector<int> image_idxs;
        std::vector<float> dete_thresholds;
        std::vector<float> nms_thresholds;
        std::vector<int> ost_ws;
        std::vector<int> ost_hs;
        std::vector<int> padding_lefts;
        std::vector<int> padding_tops;
        std::vector<int> padding_widths;
        std::vector<int> padding_heights;

        sail::Tensor* in_data = get_data(channel_idxs, image_idxs, dete_thresholds, nms_thresholds,
            ost_ws, ost_hs, padding_lefts, padding_tops, padding_widths, padding_heights);
        if(in_data == NULL){
            std::unique_lock<std::mutex> lck(mutex_pushdata_flag);
            pushdata_flag_cond.wait_for(lck,std::chrono::milliseconds(5));
            // SPDLOG_INFO("Get Data failed, sleeping for 5 ms");
            continue;
        }
        float* data_ost = (float*)in_data->sys_data();
        const char *dump_flag = getenv("SAIL_SAVE_IO_TENSORS");
        if (dump_flag != nullptr && 0 == strcmp(dump_flag, "1")){
            double id_save = get_current_time_us();
            char save_name_temp[256]={0};
            sprintf(save_name_temp,"%s_%.0f.dat","algokit_yolox_post",id_save);
            dump_float32_data((const char* )save_name_temp, data_ost, buffer_size_*batch_size, data_shape_[1], data_shape_[2]);
        }
        
        for (int index = 0; index < batch_size; ++index){
            int channel_idx = channel_idxs[index];
            int image_idx = image_idxs[index];
            float dete_threshold = dete_thresholds[index];
            float nms_threshold = nms_thresholds[index];
            int ost_w = ost_ws[index];
            int ost_h = ost_hs[index];
            int padding_left = padding_lefts[index];
            int padding_top = padding_tops[index];
            int padding_width = padding_widths[index];
            int padding_height = padding_heights[index];

            double time_start = get_current_time_us();
            float scale_w = (float)ost_w/padding_width;
            float scale_h = (float)ost_h/padding_height;


            // get output num
            int nout = in_data->shape()[2];
            int m_class_num = nout - 5;
            int box_num = in_data->shape()[1];

            float* output_data = nullptr;
            output_data = data_ost + index * box_num * nout;
            std::vector<DeteObjRect> dete_rects;
            for (int i = 0; i < box_num; i++) {
                float* ptr = output_data + i * nout;
                float score = ptr[4];
                int class_id = argmax(&ptr[5], m_class_num);
                float confidence = ptr[class_id + 5];
                if (confidence * score > dete_threshold) {
                    
                    float centerX = (ptr[0] + grids_x_[i]) * expanded_strides_[i];
                    float centerY = (ptr[1] + grids_y_[i]) * expanded_strides_[i];
                    float width = exp(ptr[2]) * expanded_strides_[i];
                    float height = exp(ptr[3]) * expanded_strides_[i];

                    DeteObjRect dete_rect;
                    dete_rect.width = width;
                    dete_rect.height = height;

                    // convert to left,top,right,bottom 
                    dete_rect.left = centerX  - dete_rect.width/2.0;
                    dete_rect.top  = centerY - dete_rect.height/2.0; 
                    dete_rect.right = centerX + dete_rect.width/2.0;
                    dete_rect.bottom  = centerY + dete_rect.height/2.0; 

                    dete_rect.class_id = class_id;
                    dete_rect.score = confidence * score;
                    
                    dete_rects.push_back(dete_rect);
                }
            }

            std::vector<DeteObjRect> dect_result;
            std::vector<int> picked;
            double time_num_start = get_current_time_us();
            sail_algo_nms_sorted_bboxes(dete_rects, picked, nms_threshold);
            double nms_time_use = (get_current_time_us() - time_num_start)/1000;
            for (size_t i = 0; i < picked.size(); i++)    {
                DeteObjRect dete_rect;
                dete_rect.left = (dete_rects[picked[i]].left - padding_left) * scale_w;
                dete_rect.top = (dete_rects[picked[i]].top - padding_top) * scale_h;
                dete_rect.width = dete_rects[picked[i]].width * scale_w;
                dete_rect.height = dete_rects[picked[i]].height * scale_h;

                dete_rect.left = dete_rect.left > 0 ? dete_rect.left : 0;
                dete_rect.top = dete_rect.top > 0 ? dete_rect.top : 0;
                // dete_rect.width = dete_rect.width < network_width ? dete_rect.width : network_width;
                // dete_rect.height = dete_rect.height < network_height ? dete_rect.height : network_height;

                dete_rect.right = dete_rect.left + dete_rect.width < ost_w ? dete_rect.left + dete_rect.width : ost_w-1;
                dete_rect.bottom = dete_rect.top + dete_rect.height < ost_h ? dete_rect.top + dete_rect.height : ost_h-1;
                
                dete_rect.score = dete_rects[picked[i]].score;
                dete_rect.class_id = dete_rects[picked[i]].class_id;

                dect_result.push_back(dete_rect);
            }
            double time_use = (get_current_time_us() - time_start)/1000;
            // SPDLOG_INFO("Yolox one output post process us {} ms, nms: {} ms. scale_w: {}, scale_h: {}",time_use, nms_time_use,scale_w,scale_h);  
            while (true) {
                int ret = push_result(dect_result, channel_idx, image_idx);
                if(ret == 0) {
                    notify_result_push();
                    break;
                }
                std::unique_lock<std::mutex> lck(mutex_result_pop);
                result_pop_cond.wait_for(lck,std::chrono::milliseconds(5));
                if(get_stop_flag()){
                    break;
                }  
            }
        }
        if(in_data){
            delete in_data;
        }
    }
    set_thread_exit();
    {
        std::lock_guard<std::mutex> lock(mutex_exit_);      //防止未收到退出信号导致卡死
        exit_thread_flag = true;
    }
}

void algo_yolox_post::algo_yolox_post_cc::notify_data_push()
{
    std::unique_lock<std::mutex> lck(mutex_pushdata_flag);
    pushdata_flag_cond.notify_all();
}

void algo_yolox_post::algo_yolox_post_cc::notify_result_push()
{
    std::unique_lock<std::mutex> lck(mutex_result_push);
    result_push_cond.notify_all();
}

void algo_yolox_post::algo_yolox_post_cc::notify_result_pop()
{
    std::unique_lock<std::mutex> lck(mutex_result_pop);
    result_pop_cond.notify_all();
}

bool algo_yolox_post::algo_yolox_post_cc::get_stop_flag()
{
    std::lock_guard<std::mutex> lock(mutex_stop_);
    return stop_thread_flag;
}

void algo_yolox_post::algo_yolox_post_cc::set_stop_flag(bool flag)
{
    std::lock_guard<std::mutex> lock(mutex_stop_);
    stop_thread_flag = flag;
}

void algo_yolox_post::algo_yolox_post_cc::set_thread_exit()
{
    std::unique_lock<std::mutex> lck(mutex_exit_);
    exit_cond.notify_all();
}   

void algo_yolox_post::algo_yolox_post_cc::wait_thread_exit()
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

int algo_yolox_post::algo_yolox_post_cc::argmax(float* data, int num) {
    float max_value = 0.0;
    int max_index = 0;
    for (int i = 0; i < num; ++i) {
        float value = data[i];
        if (value > max_value) {
            max_value = value;
            max_index = i;
        }
    }
    return max_index;
}


std::tuple<std::vector<DeteObjRect>,int,int> algo_yolox_post::algo_yolox_post_cc::get_result()
{
#ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    std::vector<DeteObjRect> results;
    int channel_idx = 0;
    int image_idx = 0;
    while(true){
        int ret = get_result_once(results, channel_idx, image_idx);
        if(ret == 0){
            break;
        }
        std::unique_lock<std::mutex> lck(mutex_result_push);
        result_push_cond.wait_for(lck,std::chrono::milliseconds(5));
        if(get_stop_flag()){
            break;
        } 
    }
    return std::make_tuple(std::move(results),channel_idx,image_idx);
}

algo_yolox_post::algo_yolox_post(const std::vector<int>& shape,int network_w, int network_h, int max_queue_size)
:_impl (new algo_yolox_post_cc(shape,network_w,network_h,max_queue_size))
{
}

algo_yolox_post::~algo_yolox_post()
{
    delete _impl;
}

#ifdef PYTHON
int algo_yolox_post::push_npy(int channel_idx, int image_idx, pybind11::array_t<float> ost_array, float dete_threshold, float nms_threshold,
        int ost_w, int ost_h, int padding_left, int padding_top, int padding_width, int padding_height)
{
    sail::Handle handle(0);
    std::vector<int> channel_idxs = {channel_idx};
    std::vector<int> image_idxs = {image_idx};
    std::vector<float> dete_thresholds = {dete_threshold};
    std::vector<float> nms_thresholds = {nms_threshold};
    std::vector<int> ost_ws = {ost_w};
    std::vector<int> ost_hs = {ost_h};
    std::vector<std::vector<int>> padding_attrs = {{padding_left,padding_top,padding_width,padding_height}};

    if (!pybind11::detail::check_flags(ost_array.ptr(), pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_)) {
        pybind11::module np = pybind11::module::import("numpy");        // like 'import numpy as np'
        pybind11::array_t<float> arr_c = np.attr("ascontiguousarray")(ost_array, "dtype"_a="float32");
        sail::Tensor* tensor_in = new sail::Tensor(handle, arr_c, true);
        return _impl->push_data(channel_idxs,image_idxs,tensor_in,dete_thresholds,nms_thresholds, ost_ws, ost_hs, padding_attrs);
    }else{
        sail::Tensor* tensor_in = new sail::Tensor(handle, ost_array, true);
        return _impl->push_data(channel_idxs,image_idxs,tensor_in,dete_thresholds,nms_thresholds, ost_ws, ost_hs, padding_attrs);
    }
}
#endif
    
int algo_yolox_post::push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        TensorPTRWithName input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr)
{
    return _impl->push_data(channel_idx,image_idx,input_data.data,dete_threshold,nms_threshold, ost_w, ost_h, padding_attr);
}

std::tuple<std::vector<DeteObjRect>,int,int> algo_yolox_post::get_result()
{
    return _impl->get_result();
}
std::tuple<std::vector<std::tuple<int, int, int, int ,int, float>>,int,int> algo_yolox_post::get_result_npy()
{
    auto result = _impl->get_result();
    std::vector<DeteObjRect> results = std::move(std::get<0>(result));
    int channel_idx = std::get<1>(result);
    int image_idx = std::get<2>(result);
    // SPDLOG_INFO("results num : {}",results.size());
    std::vector<std::tuple<int, int, int, int ,int, float>> objs;
    for (int i = 0; i < results.size(); ++i){
        int left_temp = results[i].left+0.5;
        int top_temp = results[i].top+0.5;
        int right_temp = results[i].right+0.5;
        int bottom_temp = results[i].bottom+0.5;
        int class_id_temp = results[i].class_id;
        float score_temp = results[i].score;

        objs.push_back(std::make_tuple(left_temp,
                                       top_temp,
                                       right_temp,
                                       bottom_temp,
                                       class_id_temp,
                                       score_temp));

    }
    return std::make_tuple(std::move(objs),channel_idx,image_idx);
}

class algo_yolov5_post_cpu_opt::algo_yolov5_post_cpu_opt_cc{
public:
    explicit algo_yolov5_post_cpu_opt_cc(const std::vector<std::vector<int>>& shapes, 
                                        int network_w, 
                                        int network_h);

    int reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new);



    int process(std::vector<sail::Tensor*> &input_data, 
                std::vector<int> &ost_w,
                std::vector<int> &ost_h,
                std::vector<std::vector<DeteObjRect>> &out_doxs,
                std::vector<std::vector<float>> &dete_threshold,
                std::vector<float> &nms_threshold,
                bool input_keep_aspect_ratio,
                bool input_use_multiclass_nms);

    int process(std::vector<TensorPTRWithName> &input_data, 
                std::vector<int> &ost_w,
                std::vector<int> &ost_h,
                std::vector<std::vector<DeteObjRect>> &out_doxs,
                std::vector<float> &dete_threshold,
                std::vector<float> &nms_threshold,
                bool input_keep_aspect_ratio,
                bool input_use_multiclass_nms);

    int process(std::vector<TensorPTRWithName> &input_data, 
                std::vector<int> &ost_w,
                std::vector<int> &ost_h,
                std::vector<std::vector<DeteObjRect>> &out_doxs,
                std::vector<std::vector<float>> &dete_threshold,
                std::vector<float> &nms_threshold,
                bool input_keep_aspect_ratio,
                bool input_use_multiclass_nms);

    ~algo_yolov5_post_cpu_opt_cc();

private:
    int net_w;
    int net_h;
    int class_num;
    int batch_size;
    int input_num;
    int min_idx;
    int network_box_num;
    int min_dim;
    int nout;
    int anchor_num;
    std::vector<std::vector<int>> input_shapes;
    std::vector<std::vector<std::vector<int>>> anchors;

    float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* pIsAligWidth);
    int argmax(float* data, int num);
    void NMS(std::vector<DeteObjRect>& dets, float nmsConfidence);
};

algo_yolov5_post_cpu_opt::algo_yolov5_post_cpu_opt_cc::algo_yolov5_post_cpu_opt_cc(const std::vector<std::vector<int>>& shapes, 
                                                                                int network_w, 
                                                                                int network_h)
:net_h(network_h),net_w(network_w)                                                                                    
{
    if (shapes.size() <= 0 || shapes[0].size() <= 0){
        SPDLOG_ERROR("ERROR Shapes size: {}!", shapes.size());
        throw SailRuntimeError("invalid argument");
    }

    input_num = shapes.size();
    class_num = shapes[0][shapes[0].size()-1] - 5;
    batch_size = shapes[0][0];
    if (batch_size <= 0 || class_num < 0){
        SPDLOG_ERROR("ERROR Shapes size: {}!", shapes.size());
        throw SailRuntimeError("invalid argument");
    }

    input_shapes.clear();
    for(int i = 0; i < input_num; ++i){
        if(shapes[i].size() != 3 && shapes[i].size() != 5){
            SPDLOG_ERROR("ERROR Shapes size: {}!", shapes.size());
            throw SailRuntimeError("invalid argument");
        }
        if(shapes[i][shapes[i].size() - 1] - 5 != class_num || shapes[i][0] != batch_size){
            SPDLOG_ERROR("ERROR Shapes size: {}!", shapes.size());
            throw SailRuntimeError("invalid argument");
        }
        std::vector<int> temp_shape;
        for(int j = 0; j < shapes[i].size(); ++j){
            temp_shape.push_back(shapes[i][j]);
        }
        input_shapes.push_back(temp_shape);
    }

    min_idx = 0;
    network_box_num = 0;
    min_dim = input_shapes[0].size();
    for (int i = 0; i < input_num; i++) {
        if (input_shapes[i].size() == 5) {
            network_box_num += input_shapes[i][1] * input_shapes[i][2] * input_shapes[i][3];
        }
        else {
            network_box_num += input_shapes[i][1];
        }
        if (min_dim > input_shapes[i].size()) {
            min_idx = i;
            min_dim = input_shapes[i].size();
        }
    }
    if (min_dim == 3 && input_num != 1) {
        SPDLOG_WARN("--> WARNING: the current bmodel has redundant outputs\n              you can remove the redundant outputs to improve performance\n\n");
    }
    if (network_box_num <= 0){
        SPDLOG_ERROR("ERROR Shapes size: {}!", shapes.size());
        throw SailRuntimeError("invalid argument");
    }

    anchors = {{{10, 13}, {16, 30}, {33, 23}}, {{30, 61}, {62, 45}, {59, 119}}, {{116, 90}, {156, 198}, {373, 326}}};
    anchor_num = 3;
    nout = input_shapes[min_idx][min_dim - 1];
}

int algo_yolov5_post_cpu_opt::algo_yolov5_post_cpu_opt_cc::reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new)
{
#ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    if(anchors_new.size() != input_num){
        SPDLOG_ERROR("Error anchor size!");
        return SAIL_ALGO_ERROR_SHAPES;
    }
    anchors.clear();
    for(int i=0;i<anchors_new.size();++i){
        std::vector<std::vector<int>> anchor_;
        for (int j=0;j<anchors_new[0].size();++j){
            anchor_.push_back({anchors_new[i][j][0], anchors_new[i][j][1]});
        }
        anchors.push_back(anchor_);
    }
    anchor_num = anchors[0].size();
    SPDLOG_INFO("Reset Anchors, anchor_num:{} .", anchor_num);
    return SAIL_ALGO_SUCCESS;
}

algo_yolov5_post_cpu_opt::algo_yolov5_post_cpu_opt_cc::~algo_yolov5_post_cpu_opt_cc(){

}

float algo_yolov5_post_cpu_opt::algo_yolov5_post_cpu_opt_cc::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* pIsAligWidth) {
    float ratio;
    float r_w = (float)dst_w / src_w;
    float r_h = (float)dst_h / src_h;
    if (r_h > r_w) {
        *pIsAligWidth = true;
        ratio = r_w;
    } else {
        *pIsAligWidth = false;
        ratio = r_h;
    }
    return ratio;
}

int algo_yolov5_post_cpu_opt::algo_yolov5_post_cpu_opt_cc::argmax(float* data, int num) {
    float max_value = 0.0;
    int max_index = 0;
    for (int i = 0; i < num; ++i) {
        float value = data[i];
        if (value > max_value) {
            max_value = value;
            max_index = i;
        }
    }

    return max_index;
}

void algo_yolov5_post_cpu_opt::algo_yolov5_post_cpu_opt_cc::NMS(std::vector<DeteObjRect>& dets, float nmsConfidence) {
    int length = dets.size();
    int index = length - 1;

    std::sort(dets.begin(), dets.end(), [](const DeteObjRect& a, const DeteObjRect& b) { return a.score < b.score; });

    std::vector<float> areas(length);
    for (int i = 0; i < length; i++) {
        areas[i] = dets[i].width * dets[i].height;
    }

    while (index > 0) {
        int i = 0;
        while (i < index) {
            float left = std::max(dets[index].left, dets[i].left);
            float top = std::max(dets[index].top, dets[i].top);
            float right = std::min(dets[index].right, dets[i].right);
            float bottom = std::min(dets[index].bottom, dets[i].bottom);
            float overlap = std::max(0.0f, right - left + 0.00001f) * std::max(0.0f, bottom - top + 0.00001f);
            if (overlap / (areas[index] + areas[i] - overlap) > nmsConfidence) {
                areas.erase(areas.begin() + i);
                dets.erase(dets.begin() + i);
                index--;
            } else {
                i++;
            }
        }
        index--;
    }
}

int algo_yolov5_post_cpu_opt::algo_yolov5_post_cpu_opt_cc::process(std::vector<sail::Tensor*> &input_data, 
                                                                std::vector<int> &ost_w,
                                                                std::vector<int> &ost_h,
                                                                std::vector<std::vector<DeteObjRect>> &out_doxs,
                                                                std::vector<std::vector<float>> &dete_thresholds,
                                                                std::vector<float> &nms_threshold,
                                                                bool input_keep_aspect_ratio,
                                                                bool input_use_multiclass_nms)
{
    if(input_num != input_data.size()){
        SPDLOG_ERROR("input_data count Mismatch!");
        return SAIL_ALGO_ERROR_SHAPES;
    }
    out_doxs.clear();

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        int box_num = network_box_num;
        int frame_width = ost_w[batch_idx];
        int frame_height = ost_h[batch_idx];
        std::vector<float> dete_threshold = dete_thresholds[batch_idx];
        float min_dete_threshold= *std::min_element(dete_threshold.begin(), dete_threshold.end());
        int tx1 = 0, ty1 = 0;
        float ratio_h = (float)net_h / frame_height, ratio_w = (float)net_w / frame_width;
        if (input_keep_aspect_ratio) {
            bool isAlignWidth = false;
            float ratio = get_aspect_scaled_ratio(frame_width, frame_height, net_w, net_h, &isAlignWidth);
            if (isAlignWidth) {
                ty1 = (int)((net_h - (int)(frame_height * ratio)) / 2);
            } else {
                tx1 = (int)((net_w - (int)(frame_width * ratio)) / 2);
            }
            ratio_h = ratio;
            ratio_w = ratio;
        }

        auto out_tensor = input_data[min_idx];
        int out_nout = 7;
        if (input_use_multiclass_nms)
            out_nout = nout;
        float min_opposite_log_reciprocal_m_confThreshold_sub_one = - std::log(1 / min_dete_threshold - 1);

        float* output_data = nullptr;
        std::vector<float> decoded_data;

        if (min_dim == 5) {
            if((int)decoded_data.size() != box_num*out_nout){
                decoded_data.resize(box_num*out_nout);
            }
            float* dst = decoded_data.data();

            for (int tidx = 0; tidx < input_num; ++tidx) {
                int feat_c = input_data[tidx]->shape()[1];
                int feat_h = input_data[tidx]->shape()[2];
                int feat_w = input_data[tidx]->shape()[3];
                int area = feat_h * feat_w;
                const int anchor_num = anchors[tidx].size();
                int feature_size = feat_h * feat_w * nout;
                assert(feat_c == anchor_num);

                output_data = reinterpret_cast<float*>(input_data[tidx]->sys_data());
                float* tensor_data = output_data + batch_idx * feat_c * area * nout;
                for (int anchor_idx = 0; anchor_idx < anchor_num; anchor_idx++) {
                    float* ptr = tensor_data + anchor_idx * feature_size;
                    for (int i = 0; i < area; i++) {
                        if(ptr[4] <= min_opposite_log_reciprocal_m_confThreshold_sub_one){
                            ptr += nout;
                            continue;
                        }
                        dst[0] = (sigmoid(ptr[0]) * 2 - 0.5 + i % feat_w) / feat_w * net_w;
                        dst[1] = (sigmoid(ptr[1]) * 2 - 0.5 + i / feat_w) / feat_h * net_h;
                        dst[2] = pow((sigmoid(ptr[2]) * 2), 2) * anchors[tidx][anchor_idx][0];
                        dst[3] = pow((sigmoid(ptr[3]) * 2), 2) * anchors[tidx][anchor_idx][1];
                        dst[4] = sigmoid(ptr[4]);
                        for(int d = 5; d < nout; d++)
                            dst[d] = ptr[d];
                       
                        dst += out_nout;
                        ptr += nout;
                    }
                }
            }
            output_data = decoded_data.data();
            box_num = (dst - output_data) / out_nout;
        } else {
            assert(box_num == 0 || box_num == out_tensor->shape()[1]);
            output_data = reinterpret_cast<float*>(input_data[0]->sys_data()) + batch_idx * box_num * nout;
        }

        // post 2: filter boxes
        int max_wh = 7680;
        bool agnostic = false;
        std::vector<DeteObjRect> yolobox_vec;
        for (int i = 0; i < box_num; i++) {
            float* ptr = output_data + i * out_nout;
            float score = ptr[4];
            
            if (input_use_multiclass_nms) {
                assert(min_dim == 5);
                float centerX = ptr[0];
                float centerY = ptr[1];
                float width = ptr[2];
                float height = ptr[3];
                if(class_num != dete_threshold.size() && dete_threshold.size() != 1){
                    SPDLOG_ERROR("dete_threshold count Mismatch!");
                    return SAIL_ALGO_ERROR_SHAPES;
                }
                
                for (int j = 0; j < class_num; j++) {
                    float box_transformed_m_confThreshold = - std::log(score / dete_threshold[j<dete_threshold.size()?j:0] - 1);
                    float confidence = ptr[5 + j];
                    int class_id = j;
                    if (confidence > box_transformed_m_confThreshold)
                    {
                        DeteObjRect box;
                        if (!agnostic)
                            box.left = centerX - width / 2 + class_id * max_wh;
                        else
                            box.left = centerX - width / 2;
                        if (box.left < 0) box.left = 0;
                        if (!agnostic)
                            box.top = centerY - height / 2 + class_id * max_wh;
                        else
                            box.top = centerY - height / 2;
                        if (box.top < 0) box.top = 0;
                        box.right = box.left + width;
                        box.bottom = box.top + height;
                        box.width = box.right - box.left;
                        box.height = box.bottom - box.top;
                        box.class_id = class_id;
                        box.score = sigmoid(confidence) * score;
                        yolobox_vec.push_back(box);
                    }
                }
            }
            else {
              
                
                ptr = output_data + i * nout;
                score = ptr[4];
                int class_id = argmax(&ptr[5], class_num);
                float confidence = ptr[class_id + 5];
                          
                float confThreshold_ = dete_threshold[class_id<dete_threshold.size()?class_id:0];
                float opposite_log_reciprocal_m_confThreshold_sub_one;
                if(min_dim != 5)
                    opposite_log_reciprocal_m_confThreshold_sub_one = confThreshold_ / score;
                else
                    opposite_log_reciprocal_m_confThreshold_sub_one = - std::log(score / confThreshold_ - 1);
                if (confidence > opposite_log_reciprocal_m_confThreshold_sub_one) {
                    float centerX = ptr[0];
                    float centerY = ptr[1];
                    float width = ptr[2];
                    float height = ptr[3];

                    DeteObjRect box;
                    if (!agnostic)
                        box.left = centerX - width / 2 + class_id * max_wh;
                    else
                        box.left = centerX - width / 2;
                    if (box.left < 0) box.left = 0;
                    if (!agnostic)
                        box.top = centerY - height / 2 + class_id * max_wh;
                    else
                        box.top = centerY - height / 2;
                    if (box.top < 0) box.top = 0;
                    box.right = box.left + width;
                    box.bottom = box.top + height;
                    box.width = box.right - box.left;
                    box.height = box.bottom - box.top;
                    box.class_id = class_id;
                    if(min_dim == 5)
                        confidence = sigmoid(confidence);
                    box.score = confidence * score;
                    yolobox_vec.push_back(box);
                }
            }
        }

        // post 3: nms
        NMS(yolobox_vec, nms_threshold[batch_idx]);
        if (!agnostic)
            for (auto& box : yolobox_vec){
                box.left -= box.class_id * max_wh;
                box.top -= box.class_id * max_wh;
                box.left = (box.left - tx1) / ratio_w;
                if (box.left < 0) box.left = 0;
                box.top = (box.top - ty1) / ratio_h;
                if (box.top < 0) box.top = 0;
                box.width = (box.width) / ratio_w;
                if (box.left + box.width >= frame_width)
                    box.width = frame_width - box.left;
                box.height = (box.height) / ratio_h;
                if (box.top + box.height >= frame_height)
                    box.height = frame_height - box.top;
                box.right = box.left + box.width;
                box.bottom = box.top + box.height;
            }
        else
            for (auto& box : yolobox_vec){
                box.left = (box.left - tx1) / ratio_w;
                if (box.left < 0) box.left = 0;
                box.top = (box.top - ty1) / ratio_h;
                if (box.top < 0) box.top = 0;
                box.width = (box.width) / ratio_w;
                if (box.left + box.width >= frame_width)
                    box.width = frame_width - box.left;
                box.height = (box.height) / ratio_h;
                if (box.top + box.height >= frame_height)
                    box.height = frame_height - box.top;
                box.right = box.left + box.width;
                box.bottom = box.top + box.height;
            }

        out_doxs.push_back(yolobox_vec);
    }

    return SAIL_ALGO_SUCCESS;
}

int algo_yolov5_post_cpu_opt::algo_yolov5_post_cpu_opt_cc::process(std::vector<TensorPTRWithName> &input_data, 
                                                                std::vector<int> &ost_w,
                                                                std::vector<int> &ost_h,
                                                                std::vector<std::vector<DeteObjRect>> &out_doxs,
                                                                std::vector<float> &dete_threshold,
                                                                std::vector<float> &nms_threshold,
                                                                bool input_keep_aspect_ratio,
                                                                bool input_use_multiclass_nms){
    std::vector<sail::Tensor*> input_list;
    std::vector<std::vector<float>> dete_threshold_;
    for (int i = 0; i < batch_size; i++) {
        dete_threshold_.push_back({dete_threshold[i]});
    }
    
    for (int i = 0; i < input_num; i++) {
        input_list.push_back(nullptr);
    }
    //重排序
    for(int i = 0; i < input_data.size(); ++i){
        sail::Tensor* data = input_data[i].data;
        const std::vector<int>& input_shape = data->shape();
        if(input_shape.size() != min_dim){
            SPDLOG_ERROR("Input Tensor shape Mismatch!");
            return SAIL_ALGO_ERROR_SHAPES;
        }
        if(batch_size != input_shape[0]){
            SPDLOG_ERROR("Input Batch Size Mismatch!");
            return SAIL_ALGO_ERROR_SHAPES;
        }
        bool has_match = false;
        for(int j = 0; j < input_shapes.size(); ++j){
            bool cur_match = true;
            for (int z = 0; z < min_dim; z++) {
                if(input_shape[z] != input_shapes[j][z])
                {
                    cur_match = false;
                    break;    
                }
            }
            if (cur_match)
            {
                if(input_list[j] != NULL){
                    SPDLOG_ERROR("Input Tensor shape Mismatch!");
                    return SAIL_ALGO_ERROR_SHAPES;
                }
                input_list[j] = data;
                has_match = true;
            }
            
        }
        if(!has_match){         //没有匹配到shape
            SPDLOG_ERROR("Input Tensor shape Mismatch!");
            return SAIL_ALGO_ERROR_SHAPES;
        }
    }

    // for(int i = 0; i < input_data.size(); ++i)
    //     input_list.push_back(input_data[i].data);

    return process(input_list, ost_w, ost_h, out_doxs, dete_threshold_, nms_threshold, input_keep_aspect_ratio, input_use_multiclass_nms);
}

int algo_yolov5_post_cpu_opt::algo_yolov5_post_cpu_opt_cc::process(std::vector<TensorPTRWithName> &input_data, 
                                                                std::vector<int> &ost_w,
                                                                std::vector<int> &ost_h,
                                                                std::vector<std::vector<DeteObjRect>> &out_doxs,
                                                                std::vector<std::vector<float>> &dete_threshold,
                                                                std::vector<float> &nms_threshold,
                                                                bool input_keep_aspect_ratio,
                                                                bool input_use_multiclass_nms){
    std::vector<sail::Tensor*> input_list;

    for (int i = 0; i < input_num; i++) {
        input_list.push_back(nullptr);
    }
    //重排序
    for(int i = 0; i < input_data.size(); ++i){
        sail::Tensor* data = input_data[i].data;
        const std::vector<int>& input_shape = data->shape();
        if(input_shape.size() != min_dim){
            SPDLOG_ERROR("Input Tensor shape Mismatch!");
            return SAIL_ALGO_ERROR_SHAPES;
        }
        if(batch_size != input_shape[0]){
            SPDLOG_ERROR("Input Batch Size Mismatch!");
            return SAIL_ALGO_ERROR_SHAPES;
        }
        bool has_match = false;
        for(int j = 0; j < input_shapes.size(); ++j){
            bool cur_match = true;
            for (int z = 0; z < min_dim; z++) {
                if(input_shape[z] != input_shapes[j][z])
                {
                    cur_match = false;
                    break;    
                }
            }
            if (cur_match)
            {
                if(input_list[j] != NULL){
                    SPDLOG_ERROR("Input Tensor shape Mismatch!");
                    return SAIL_ALGO_ERROR_SHAPES;
                }
                input_list[j] = data;
                has_match = true;
            }
            
        }
        if(!has_match){         //没有匹配到shape
            SPDLOG_ERROR("Input Tensor shape Mismatch!");
            return SAIL_ALGO_ERROR_SHAPES;
        }
    }

    // for(int i = 0; i < input_data.size(); ++i)
    //     input_list.push_back(input_data[i].data);

    return process(input_list, ost_w, ost_h, out_doxs, dete_threshold, nms_threshold, input_keep_aspect_ratio, input_use_multiclass_nms);
}

algo_yolov5_post_cpu_opt::algo_yolov5_post_cpu_opt(const std::vector<std::vector<int>>& shapes, 
                                            int network_w, 
                                            int network_h)
    :_impl (new algo_yolov5_post_cpu_opt_cc(shapes,network_w,network_h))
{}

algo_yolov5_post_cpu_opt::~algo_yolov5_post_cpu_opt()
{
    delete _impl;
}


int algo_yolov5_post_cpu_opt::process(std::vector<TensorPTRWithName> &input_data, 
                                        std::vector<int> &ost_w,
                                        std::vector<int> &ost_h,
                                        std::vector<std::vector<DeteObjRect>> &out_doxs,
                                        std::vector<float> &dete_threshold,
                                        std::vector<float> &nms_threshold,
                                        bool input_keep_aspect_ratio,
                                        bool input_use_multiclass_nms){
            
    return _impl->process(input_data, ost_w, ost_h, out_doxs, dete_threshold, nms_threshold, input_keep_aspect_ratio, input_use_multiclass_nms);
}

int algo_yolov5_post_cpu_opt::process(std::vector<TensorPTRWithName> &input_data, 
                                        std::vector<int> &ost_w,
                                        std::vector<int> &ost_h,
                                        std::vector<std::vector<DeteObjRect>> &out_doxs,
                                        std::vector<std::vector<float>> &dete_threshold,
                                        std::vector<float> &nms_threshold,
                                        bool input_keep_aspect_ratio,
                                        bool input_use_multiclass_nms){
            
    return _impl->process(input_data, ost_w, ost_h, out_doxs, dete_threshold, nms_threshold, input_keep_aspect_ratio, input_use_multiclass_nms);
}
std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> 
    algo_yolov5_post_cpu_opt::process(std::vector<TensorPTRWithName> &input_data, 
                                        std::vector<int> &ost_w,
                                        std::vector<int> &ost_h,
                                        std::vector<float> &dete_threshold,
                                        std::vector<float> &nms_threshold,
                                        bool input_keep_aspect_ratio,
                                        bool input_use_multiclass_nms)
{
    std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> results;
    results.clear();
    std::vector<std::vector<DeteObjRect>> out_doxs;
    int ret = process(input_data, ost_w, ost_h, out_doxs, dete_threshold, nms_threshold, input_keep_aspect_ratio, input_use_multiclass_nms);
    if(ret != SAIL_ALGO_SUCCESS){
        return results;
    }
    for(int i = 0; i < out_doxs.size(); ++i){
        std::vector<std::tuple<int, int, int, int ,int, float>> objs;
        for(int j = 0; j < out_doxs[i].size(); ++j){
            int left_temp = out_doxs[i][j].left;
            int top_temp = out_doxs[i][j].top;
            int right_temp = out_doxs[i][j].right;
            int bottom_temp = out_doxs[i][j].bottom;
            int class_id_temp = out_doxs[i][j].class_id;
            float score_temp = out_doxs[i][j].score;

            objs.push_back(std::make_tuple(left_temp,
                                        top_temp,
                                        right_temp,
                                        bottom_temp,
                                        class_id_temp,
                                        score_temp));
        }
        results.push_back(objs);
    }
    return results;
}
std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> 
    algo_yolov5_post_cpu_opt::process(std::vector<TensorPTRWithName> &input_data, 
                                        std::vector<int> &ost_w,
                                        std::vector<int> &ost_h,
                                        std::vector<std::vector<float>> &dete_threshold,
                                        std::vector<float> &nms_threshold,
                                        bool input_keep_aspect_ratio,
                                        bool input_use_multiclass_nms)
{
    std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> results;
    results.clear();
    std::vector<std::vector<DeteObjRect>> out_doxs;
    int ret = process(input_data, ost_w, ost_h, out_doxs, dete_threshold, nms_threshold, input_keep_aspect_ratio, input_use_multiclass_nms);
    if(ret != SAIL_ALGO_SUCCESS){
        return results;
    }
    for(int i = 0; i < out_doxs.size(); ++i){
        std::vector<std::tuple<int, int, int, int ,int, float>> objs;
        for(int j = 0; j < out_doxs[i].size(); ++j){
            int left_temp = out_doxs[i][j].left;
            int top_temp = out_doxs[i][j].top;
            int right_temp = out_doxs[i][j].right;
            int bottom_temp = out_doxs[i][j].bottom;
            int class_id_temp = out_doxs[i][j].class_id;
            float score_temp = out_doxs[i][j].score;

            objs.push_back(std::make_tuple(left_temp,
                                        top_temp,
                                        right_temp,
                                        bottom_temp,
                                        class_id_temp,
                                        score_temp));
        }
        results.push_back(objs);
    }
    return results;
}
std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> 
    algo_yolov5_post_cpu_opt::process(std::map<std::string, Tensor&>& input,
                                        std::vector<int> &ost_w,
                                        std::vector<int> &ost_h,
                                        std::vector<float> &dete_threshold,
                                        std::vector<float> &nms_threshold,
                                        bool input_keep_aspect_ratio,
                                        bool input_use_multiclass_nms)
{
    std::vector<TensorPTRWithName> input_data;
    auto iter_temp = input.begin();
    while(iter_temp != input.end()){
        TensorPTRWithName in_data_ptr;
        in_data_ptr.data = &iter_temp->second;
        in_data_ptr.name = iter_temp->first;
        input_data.push_back(in_data_ptr);
        iter_temp++;
    }
    return process(input_data, ost_w, ost_h, dete_threshold, nms_threshold, input_keep_aspect_ratio, input_use_multiclass_nms);
}
std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> 
    algo_yolov5_post_cpu_opt::process(std::map<std::string, Tensor&>& input,
                                        std::vector<int> &ost_w,
                                        std::vector<int> &ost_h,
                                        std::vector<std::vector<float>> &dete_threshold,
                                        std::vector<float> &nms_threshold,
                                        bool input_keep_aspect_ratio,
                                        bool input_use_multiclass_nms)
{
    std::vector<TensorPTRWithName> input_data;
    auto iter_temp = input.begin();
    while(iter_temp != input.end()){
        TensorPTRWithName in_data_ptr;
        in_data_ptr.data = &iter_temp->second;
        in_data_ptr.name = iter_temp->first;
        input_data.push_back(in_data_ptr);
        iter_temp++;
    }
    return process(input_data, ost_w, ost_h, dete_threshold, nms_threshold, input_keep_aspect_ratio, input_use_multiclass_nms);
}
int algo_yolov5_post_cpu_opt::reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new)
{
    return _impl->reset_anchors(anchors_new);
}

// algo_yolov8_seg_post_tpu_opt
class algo_yolov8_seg_post_tpu_opt::algo_yolov8_seg_post_tpu_opt_cc {
public:
    explicit algo_yolov8_seg_post_tpu_opt_cc(string bmodel_file, int dev_id, const vector<int>& detection_shape, const vector<int>& segmentation_shape, int network_h, int network_w);

    int process(sail::Tensor* &detection_input,
                sail::Tensor* &segmentation_input,
                int &ost_h,
                int &ost_w,
                vector<YoloV8Box> &yolov8_results,
                float &dete_threshold,
                float &nms_threshold,
                bool input_keep_aspect_ratio,
                bool input_use_multiclass_nms);

    int process(TensorPTRWithName &detection_input,
                TensorPTRWithName &segmentation_input,
                int &ost_h,
                int &ost_w,
                vector<YoloV8Box> &yolov8_results,
                float &dete_threshold,
                float &nms_threshold,
                bool input_keep_aspect_ratio,
                bool input_use_multiclass_nms);

    ~algo_yolov8_seg_post_tpu_opt_cc();
    cv::Mat crop_mask(cv::Mat& mask, const cv::Rect& box);
    vector<float> mask_to_contour(cv::Mat mask);

private:
    // getmask bmodel
    bm_handle_t tpu_mask_handle;
    void *bmrt = nullptr;
    const bm_net_info_t *netinfo;
    vector<string> network_names;
    int tpu_mask_num;
    int m_tpumask_net_h, m_tpumask_net_w;
    int mask_len;

    int per_feat_size;
    int m_class_num;
    int feat_num;
    int m_net_h, m_net_w;
    int max_det = 300;
    int max_wh = 7680; // (pixels) maximum box width and height

    struct Paras {
        int r_x;
        int r_y;
        int r_w;
        int r_h;
        int width;
        int height;
    };

    struct ImageInfo {
        cv::Size raw_size;
        cv::Vec4d trans;
    };

    float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* pIsAligWidt);
    void NMS(vector<YoloV8Box>& dets, float nmsConfidence);
    void clip_boxes(vector<YoloV8Box>& yolobox_vec, int src_w, int src_h);
    void getmask_tpu(vector<YoloV8Box>& yolov8box_input, int start, const bm_tensor_t& segmentation_tensor, Paras& paras, vector<YoloV8Box>& yolov8box_output, float confThreshold);
};

algo_yolov8_seg_post_tpu_opt::algo_yolov8_seg_post_tpu_opt_cc::algo_yolov8_seg_post_tpu_opt_cc(string bmodel_file, int dev_id, const vector<int>& detection_shape, const vector<int>& segmentation_shape, int network_h, int network_w):m_net_h(network_h), m_net_w(network_w) {

    // 1. get handle
    int ret = bm_dev_request(&tpu_mask_handle, dev_id);
    if (ret != BM_SUCCESS) {
        SPDLOG_ERROR("bm_dev_request error");
        throw SailRuntimeError("bm_dev_request error");
    }

    // 2. create bmrt and load bmodel
    bmrt = bmrt_create(tpu_mask_handle);
    if (!bmrt_load_bmodel(bmrt, bmodel_file.c_str())) {
        SPDLOG_ERROR("load getmask bmodel {} failed", bmodel_file);
        throw SailRuntimeError("load getmask bmodel failed");
    }

    // 3. get network names from bmodel
    const char **names = nullptr;
    int num = bmrt_get_network_number(bmrt);
    if (num > 1) {
        SPDLOG_INFO("This bmodel have {} networks, and this program will only take network 0", num);
    }

    bmrt_get_network_names(bmrt, &names);
    for (int i = 0; i < num; ++i) {
        network_names.emplace_back(names[i]);
    }
    free(names);

    // 4. get netinfo by netname
    netinfo = bmrt_get_network_info(bmrt, network_names[0].c_str());
    if (netinfo->stage_num > 1) {
        SPDLOG_INFO("This bmodel have {}  stages, and this program will only take stage 0", netinfo->stage_num);
    }

    // 5. initialize parameters.
    m_tpumask_net_h = netinfo->stages[0].input_shapes[1].dims[2];
    m_tpumask_net_w = netinfo->stages[0].input_shapes[1].dims[3];

    if (netinfo->stages[0].input_shapes[1].dims[1] != netinfo->stages[0].input_shapes[0].dims[2]) {
        SPDLOG_ERROR("The number of prototype mask does not match the mask coefficients");
        throw SailRuntimeError("The number of prototype mask does not match the mask coefficients");
    }

    tpu_mask_num = netinfo->stages[0].input_shapes[0].dims[1];
    mask_len = netinfo->stages[0].input_shapes[1].dims[1];

    if (mask_len != 32) {
        SPDLOG_ERROR("The number of prototype mask supports only 32 now");
        throw SailRuntimeError("The number of prototype mask supports only 32 now");
    }

    per_feat_size = detection_shape[1];
    m_class_num = per_feat_size - mask_len - 4;
    feat_num = detection_shape[2];

    // 6. compare
    if ((segmentation_shape[1] != mask_len) || (segmentation_shape[2] != m_tpumask_net_h) || (segmentation_shape[3] != m_tpumask_net_w)) {
        SPDLOG_ERROR("The shape of prototype mask of getmask bmodel does not match the segmentation_shape of the actual input");
        throw SailRuntimeError("The shape of prototype mask of getmask bmodel does not match the segmentation_shape of the actual input");
    }

}

algo_yolov8_seg_post_tpu_opt::algo_yolov8_seg_post_tpu_opt_cc::~algo_yolov8_seg_post_tpu_opt_cc() {
    if (bmrt != nullptr) {
        bmrt_destroy(bmrt);
        bmrt = nullptr;
    }
    if (tpu_mask_handle != nullptr) {
        bm_dev_free(tpu_mask_handle);
        tpu_mask_handle = nullptr;
    }
}

float algo_yolov8_seg_post_tpu_opt::algo_yolov8_seg_post_tpu_opt_cc::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* pIsAligWidth) {
    float ratio;
    float r_w = (float)dst_w / src_w;
    float r_h = (float)dst_h / src_h;
    if (r_h > r_w) {
        *pIsAligWidth = true;
        ratio = r_w;
    } else {
        *pIsAligWidth = false;
        ratio = r_h;
    }
    return ratio;
}

void algo_yolov8_seg_post_tpu_opt::algo_yolov8_seg_post_tpu_opt_cc::NMS(vector<YoloV8Box>& dets, float nmsConfidence) {
    int length = dets.size();
    int index = length - 1;

    std::sort(dets.begin(), dets.end(), [](const YoloV8Box& a, const YoloV8Box& b) { return a.score < b.score; });

    std::vector<float> areas(length);
    for (int i = 0; i < length; i++) {
        float width = dets[i].right - dets[i].left;
        float height = dets[i].bottom - dets[i].top;
        areas[i] = width * height;
    }

    while (index > 0) {
        int i = 0;
        while (i < index) {
            float left = std::max(dets[index].left, dets[i].left);
            float top = std::max(dets[index].top, dets[i].top);
            float right = std::min(dets[index].right, dets[i].right);
            float bottom = std::min(dets[index].bottom, dets[i].bottom);
            float overlap = std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
            if (overlap / (areas[index] + areas[i] - overlap) > nmsConfidence) {
                areas.erase(areas.begin() + i);
                dets.erase(dets.begin() + i);
                index--;
            } else {
                i++;
            }
        }
        index--;
    }
}

void algo_yolov8_seg_post_tpu_opt::algo_yolov8_seg_post_tpu_opt_cc::clip_boxes(vector<YoloV8Box>& yolobox_vec, int src_w, int src_h) {
    for (int i = 0; i < yolobox_vec.size(); i++) {
        yolobox_vec[i].left = std::max((float)0.0, std::min(yolobox_vec[i].left, (float)src_w));
        yolobox_vec[i].top = std::max((float)0.0, std::min(yolobox_vec[i].top, (float)src_h));
        yolobox_vec[i].right = std::max((float)0.0, std::min(yolobox_vec[i].right, (float)src_w));
        yolobox_vec[i].bottom = std::max((float)0.0, std::min(yolobox_vec[i].bottom, (float)src_h));
    }
}

cv::Mat algo_yolov8_seg_post_tpu_opt::algo_yolov8_seg_post_tpu_opt_cc::crop_mask(cv::Mat& mask, const cv::Rect& box) {
    cv::threshold(mask, mask, 0.5, 255, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8UC1); 
    
    cv::Mat cropped_mask(mask.size(), mask.type(), cv::Scalar(0));
    cv::add(cropped_mask(box), mask(box), cropped_mask(box));
    return cropped_mask;
}

vector<float> algo_yolov8_seg_post_tpu_opt::algo_yolov8_seg_post_tpu_opt_cc::mask_to_contour(cv::Mat mask) {
    vector<float> contour;
    contour.clear();
    vector<vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (!contours.empty()) {
        size_t max_contour_index = 0;
        int max_contour_length = 0;

        for (size_t i = 0; i < contours.size(); ++i) {
            if (contours[i].size() > max_contour_length) {
                max_contour_length = contours[i].size();
                max_contour_index = i;
            }
        }

        const vector<cv::Point>& longest_contour = contours[max_contour_index];
        for (const auto& point : longest_contour) {
            contour.emplace_back(static_cast<float>(point.x));
            contour.emplace_back(static_cast<float>(point.y));
        }
    }

    return contour;
}

void algo_yolov8_seg_post_tpu_opt::algo_yolov8_seg_post_tpu_opt_cc::getmask_tpu(vector<YoloV8Box>& yolov8box_input, int start, const bm_tensor_t& segmentation_tensor, Paras& paras, vector<YoloV8Box>& yolov8box_output, float confThreshold) {
    int mask_height = m_tpumask_net_h;
    int mask_width = m_tpumask_net_w;
    int actual_mask_num = MIN(tpu_mask_num, yolov8box_input.size() - start);

    netinfo->stages[0].input_shapes[0].dims[0] = 1;
    netinfo->stages[0].input_shapes[0].dims[1] = actual_mask_num;
    netinfo->stages[0].input_shapes[0].dims[2] = mask_len;

    //1. prepare bmodel inputs
    bm_tensor_t detection_tensor;
    bool ok = bmrt_tensor(&detection_tensor, bmrt, netinfo->input_dtypes[0], netinfo->stages[0].input_shapes[0]);
    if (!ok) {
        SPDLOG_ERROR("bmrt_tensor error");
        throw SailRuntimeError("bmrt_tensor error");
    }

    for (size_t i = start; i < start + actual_mask_num; i++) {
        int ret = bm_memcpy_s2d_partial_offset(tpu_mask_handle, detection_tensor.device_mem, reinterpret_cast<void*>(yolov8box_input[i].mask.data()), 32*4, 32*4*(i-start));
        if (ret != BM_SUCCESS) {
            SPDLOG_ERROR("bm_memcpy_s2d_partial_offset");
            throw SailRuntimeError("bm_memcpy_s2d_partial_offset");
        }
    }

    vector<bm_tensor_t> input_tensors = {detection_tensor, segmentation_tensor};
    vector<bm_tensor_t> output_tensors;
    
    // 2. run bmodel
    output_tensors.resize(netinfo->output_num);
    ok = bmrt_launch_tensor(bmrt, netinfo->name, input_tensors.data(), netinfo->input_num, output_tensors.data(), netinfo->output_num);
    if (!ok) {
        SPDLOG_ERROR("bmrt_launch_tensor error");
        throw SailRuntimeError("bmrt_launch_tensor error");
    }

    int ret = bm_thread_sync(tpu_mask_handle);
    if (ret != BM_SUCCESS) {
        SPDLOG_ERROR("bm_thread_sync error");
        throw SailRuntimeError("bm_thread_sync error");
    }

    bm_free_device(tpu_mask_handle, input_tensors[0].device_mem);

    // 3. get outputs
    bm_tensor_t output_tensor = output_tensors[0];
    float output0[1 * actual_mask_num * mask_height * mask_width];
    ret = bm_memcpy_d2s_partial(tpu_mask_handle, output0, output_tensor.device_mem, bmrt_tensor_bytesize(&output_tensor));
    if (ret != BM_SUCCESS) {
        SPDLOG_ERROR("bm_memcpy_d2s_partial error");
        throw SailRuntimeError("bm_memcpy_d2s_partial error");
    }
    for (int i = 0; i < output_tensors.size(); i++) {
        bm_free_device(tpu_mask_handle, output_tensors[i].device_mem);  
    }

    // 4. crop + mask
    for (int i = 0; i < actual_mask_num; i++){
        int yi = start + i;
        cv::Mat temp_mask(mask_height, mask_width, CV_32FC1, output0 + i * mask_height * mask_width);
        cv::Mat masks_feature = temp_mask(cv::Rect(paras.r_x, paras.r_y, paras.r_w, paras.r_h)); 
        cv::Mat mask;
        cv::resize(masks_feature, mask, cv::Size(paras.width, paras.height)); 
        
        cv::Rect box = cv::Rect{yolov8box_input[yi].left, yolov8box_input[yi].top, yolov8box_input[yi].right - yolov8box_input[yi].left, yolov8box_input[yi].bottom - yolov8box_input[yi].top};
        cv::Mat crop_mask_ = crop_mask(mask, box);
        vector<float> contour = mask_to_contour(crop_mask_);

        yolov8box_input[yi].mask_img = crop_mask_;
        yolov8box_input[yi].contour = contour;

        yolov8box_output.push_back(yolov8box_input[yi]);
    }

}

int algo_yolov8_seg_post_tpu_opt::algo_yolov8_seg_post_tpu_opt_cc::process(sail::Tensor* &detection_input,
                                                                        sail::Tensor* &segmentation_input,
                                                                        int &ost_h,
                                                                        int &ost_w,
                                                                        vector<YoloV8Box> &yolov8_results,
                                                                        float &dete_threshold,
                                                                        float &nms_threshold,
                                                                        bool input_keep_aspect_ratio,
                                                                        bool input_use_multiclass_nms)
    {   
        // post0: init
        vector<YoloV8Box> yolov8box_vec;
        int frame_width = ost_w;
        int frame_height = ost_h;  // Height and width of the original image
        
        // segmentation_data, one batch 
        float* segmentation_data = reinterpret_cast<float*>(segmentation_input->sys_data());
        bm_tensor_t segmentation_tensor;
        bool ok = bmrt_tensor(&segmentation_tensor, bmrt, netinfo->input_dtypes[1], netinfo->stages[0].input_shapes[1]);
        if (!ok) {
           SPDLOG_ERROR("bmrt_tensor error"); 
           throw SailRuntimeError("bmrt_tensor error");
        }
        int ret = bm_memcpy_s2d_partial(tpu_mask_handle, segmentation_tensor.device_mem, reinterpret_cast<void*>(segmentation_data), bmrt_tensor_bytesize(&segmentation_tensor));
        if (ret != BM_SUCCESS) {
            SPDLOG_ERROR("bm_memcpy_s2d_partial error"); 
            throw SailRuntimeError("bm_memcpy_s2d_partial error");
        }

        // input_keep_aspect_ratio
        float ratio = 0.0;
        int tx1 = 0, ty1 = 0;
        ImageInfo para;

        if (input_keep_aspect_ratio) {
            bool isAlignWidth = false;  
            ratio = get_aspect_scaled_ratio(frame_width, frame_height, m_net_w, m_net_h, &isAlignWidth);
            if (isAlignWidth) {
                ty1 = (int)((m_net_h - frame_height * ratio) / 2);  
            } else {
                tx1 = (int)((m_net_w - frame_width * ratio) / 2);
            }
            para = {cv::Size(frame_width, frame_height), {ratio, ratio, tx1, ty1}}; 

        }
        else {
            float ratio = 1;
            tx1 = 0;
            ty1 = 0;
            para = {cv::Size(frame_width, frame_height), {m_net_w / frame_width, m_net_h / frame_height, tx1, ty1}};
        }

        // post1: get output one batch
        float* detection_data = nullptr; 
        detection_data = reinterpret_cast<float*>(detection_input->sys_data());

        // post2:  get detections matrix nx6 (xyxy, score, class_id, mask)
        float* cls_conf = detection_data + 4 * feat_num;
        for (int i = 0; i < feat_num; i++) {
            // input_use_multiclass_nms
            if (input_use_multiclass_nms) {
                for (int j = 0; j < m_class_num; j++) {
                    float cur_value = cls_conf[i + j * feat_num];
                    if (cur_value >= dete_threshold) {
                        YoloV8Box box;
                        box.score = cur_value;
                        box.class_id = j;
                        int c = box.class_id * max_wh;
                        float centerX = detection_data[i + 0 * feat_num];
                        float centerY = detection_data[i + 1 * feat_num];
                        float width = detection_data[i + 2 * feat_num];
                        float height = detection_data[i + 3 * feat_num];

                        box.left = centerX - width / 2 + c;
                        box.top = centerY - height / 2 + c;
                        box.right = box.left + width;
                        box.bottom = box.top + height;
                        box.mask = vector<float>(detection_data + 4 + m_class_num, detection_data + per_feat_size);

                        yolov8box_vec.emplace_back(box);
                    }
                }
            }
            else 
            {
                // input_use_multiclass_nms == false
                float max_value = 0.0;
                int max_index = 0;
                for (int j = 0; j < m_class_num; j++) {
                    float cur_value = cls_conf[i + j * feat_num];
                    if (cur_value > max_value) {
                        max_value = cur_value;
                        max_index = j;
                    }
                }
                
               
                if (max_value >= dete_threshold) {
                    YoloV8Box box; 
                    box.score = max_value;
                    box.class_id = max_index;
                    int c = box.class_id * max_wh; 
                    float centerX = detection_data[i + 0 * feat_num];
                    float centerY = detection_data[i + 1 * feat_num];
                    float width = detection_data[i + 2 * feat_num];
                    float height = detection_data[i + 3 * feat_num];

                    box.left = centerX - width / 2 + c;
                    box.top = centerY - height / 2 + c;
                    box.right = box.left + width;
                    box.bottom = box.top + height;

                    for (int k = 0; k < mask_len; k++) {
                        box.mask.emplace_back(detection_data[i + (per_feat_size - mask_len + k) * feat_num]); 
                    }
                    yolov8box_vec.emplace_back(box);
                }

            }
        }

        // post3: nms
        NMS(yolov8box_vec, nms_threshold);
        if (yolov8box_vec.size() > max_det) {
            yolov8box_vec.erase(yolov8box_vec.begin(), yolov8box_vec.begin() + (yolov8box_vec.size() - max_det));
        }

        for (int i = 0; i < yolov8box_vec.size(); i++) {
            int c = yolov8box_vec[i].class_id * max_wh;
            yolov8box_vec[i].left = yolov8box_vec[i].left - c;
            yolov8box_vec[i].top = yolov8box_vec[i].top - c;
            yolov8box_vec[i].right = yolov8box_vec[i].right - c;
            yolov8box_vec[i].bottom = yolov8box_vec[i].bottom - c;
        }

        for (int i = 0; i < yolov8box_vec.size(); i++) {
            float centerx = ((yolov8box_vec[i].right + yolov8box_vec[i].left) / 2 - tx1) / ratio;
            float centery = ((yolov8box_vec[i].bottom + yolov8box_vec[i].top) / 2 - ty1) / ratio;
            float width = (yolov8box_vec[i].right - yolov8box_vec[i].left) / ratio;
            float height = (yolov8box_vec[i].bottom - yolov8box_vec[i].top) / ratio; 
            yolov8box_vec[i].left = centerx - width / 2;
            yolov8box_vec[i].top = centery - height / 2;
            yolov8box_vec[i].right = centerx + width / 2;
            yolov8box_vec[i].bottom = centery + height / 2;
        }

        clip_boxes(yolov8box_vec, frame_width, frame_height);

        // post4: get mask
        cv::Vec4f trans = para.trans;
        int r_x = floor(trans[2] / m_net_w * m_tpumask_net_w);
        int r_y = floor(trans[3] / m_net_h * m_tpumask_net_h);
        int r_w = m_tpumask_net_w - 2 * r_x;
        int r_h = m_tpumask_net_h - 2 * r_y;

        r_w = MAX(r_w, 1);
        r_h = MAX(r_h, 1);

        struct Paras paras={r_x, r_y, r_w, r_h, para.raw_size.width, para.raw_size.height};

        vector<YoloV8Box> yolobox_valid_vec;
        for(int i = 0; i < yolov8box_vec.size(); i++){
            if (yolov8box_vec[i].right > yolov8box_vec[i].left + 1 && yolov8box_vec[i].bottom > yolov8box_vec[i].top + 1){
                yolobox_valid_vec.push_back(yolov8box_vec[i]);
            }
        }  

        vector<YoloV8Box> yolov8box_output;
        if (yolobox_valid_vec.size() > 0) {
            int mask_times = (yolobox_valid_vec.size() + tpu_mask_num - 1) / tpu_mask_num;
            for (int i = 0; i < mask_times; i++){
                int start = i * tpu_mask_num;
                getmask_tpu(yolobox_valid_vec, start, segmentation_tensor, paras, yolov8box_output, dete_threshold);
            }
        }

        yolov8_results = yolov8box_output;

        bm_free_device(tpu_mask_handle, segmentation_tensor.device_mem);

        return SAIL_ALGO_SUCCESS;
    }


int algo_yolov8_seg_post_tpu_opt::algo_yolov8_seg_post_tpu_opt_cc::process(TensorPTRWithName &detection_input,
                                                                        TensorPTRWithName &segmentation_input,
                                                                        int &ost_h,
                                                                        int &ost_w,
                                                                        vector<YoloV8Box> &yolov8_results,
                                                                        float &dete_threshold,
                                                                        float &nms_threshold,
                                                                        bool input_keep_aspect_ratio,
                                                                        bool input_use_multiclass_nms) 
{
    sail::Tensor* detection_input_data = detection_input.data;
    sail::Tensor* segmentation_input_data = segmentation_input.data;

    return process(detection_input_data, segmentation_input_data, ost_h, ost_w, yolov8_results, dete_threshold, nms_threshold, input_keep_aspect_ratio, input_use_multiclass_nms);
}


algo_yolov8_seg_post_tpu_opt::algo_yolov8_seg_post_tpu_opt(string bmodel_file, int dev_id, 
                                                        const vector<int>& detection_shape, 
                                                        const vector<int>& segmentation_shape, 
                                                        int network_h, int network_w)
    :_impl (new algo_yolov8_seg_post_tpu_opt_cc(bmodel_file, dev_id, detection_shape, segmentation_shape, network_h, network_w))
{

}

algo_yolov8_seg_post_tpu_opt::~algo_yolov8_seg_post_tpu_opt()
{   
    delete _impl;
}


int algo_yolov8_seg_post_tpu_opt::process(TensorPTRWithName &detection_input,
                                        TensorPTRWithName &segmentation_input,
                                        int &ost_h,
                                        int &ost_w,
                                        vector<YoloV8Box> &yolov8_results,
                                        float &dete_threshold,
                                        float &nms_threshold,
                                        bool input_keep_aspect_ratio,
                                        bool input_use_multiclass_nms)
{  
    return _impl->process(detection_input, segmentation_input, ost_h, ost_w, yolov8_results, dete_threshold, nms_threshold, input_keep_aspect_ratio, input_use_multiclass_nms);
}


#ifdef PYTHON
pybind11::array_t<uint8_t> cvmat_to_numpy(cv::Mat cv_mat){
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
pybind11::list algo_yolov8_seg_post_tpu_opt::process(TensorPTRWithName &detection_input,
                            TensorPTRWithName &segmentation_input,
                            int &ost_h,
                            int &ost_w,
                            float &dete_threshold,
                            float &nms_threshold,
                            bool input_keep_aspect_ratio,
                            bool input_use_multiclass_nms)
{
    pybind11::list results;
    vector<YoloV8Box> yolov8_results;

    int ret = process(detection_input, segmentation_input, ost_h, ost_w, yolov8_results, dete_threshold, nms_threshold, input_keep_aspect_ratio, input_use_multiclass_nms);
    
    if (ret != SAIL_ALGO_SUCCESS) {
        return results;
    }

    for (int i = 0; i < yolov8_results.size(); ++i) {
        int left = yolov8_results[i].left;
        int top = yolov8_results[i].top;
        int right = yolov8_results[i].right;
        int bottom = yolov8_results[i].bottom;
        float score = yolov8_results[i].score;
        int class_id = yolov8_results[i].class_id;
        pybind11::array_t<uint8_t> mask = cvmat_to_numpy(yolov8_results[i].mask_img);
        vector<float> contour = yolov8_results[i].contour;

        results.append(pybind11::make_tuple(left, top, right, bottom, score, class_id, contour, mask));

    }

    return results;  
}


pybind11::list algo_yolov8_seg_post_tpu_opt::process(map<string, Tensor&> &detection_input,
                            map<string, Tensor&> &segmentation_input,
                            int &ost_h,
                            int &ost_w,
                            float &dete_threshold,
                            float &nms_threshold,
                            bool input_keep_aspect_ratio,
                            bool input_use_multiclass_nms)
{   
    TensorPTRWithName detection_input_temp;
    for (const auto& pair : detection_input) { 
        detection_input_temp.data = &pair.second; 
        detection_input_temp.name = pair.first;   
    }

    TensorPTRWithName segmentation_input_temp;
    for (const auto& pair : segmentation_input) { 
        segmentation_input_temp.data = &pair.second; 
        segmentation_input_temp.name = pair.first;
    }

    pybind11::list results;
    results = process(detection_input_temp, segmentation_input_temp, ost_h, ost_w, dete_threshold, nms_threshold, input_keep_aspect_ratio, input_use_multiclass_nms);

    return results; 
}
#endif

/*下面的内容都是跟目标跟踪有关*/
/*controller*/
class deepsort_tracker_controller::deepsort_tracker_controller_cc {
    public:
    explicit deepsort_tracker_controller_cc(float max_cosine_distance, int nn_budget, int k_feature_dim, float max_iou_distance = 0.7, int max_age = 30, int n_init = 3);
    ~deepsort_tracker_controller_cc();
    int process(const vector<DeteObjRect>& detected_objects, vector<Tensor>& feature, vector<TrackObjRect>& tracked_objects);
    int process(const vector<DeteObjRect>& detected_objects, vector<vector<float>>& feature, vector<TrackObjRect>& tracked_objects);

    private:
    int k_feature_dim;
    int frame_id;
    tracker* obj_tracker;
};

deepsort_tracker_controller::deepsort_tracker_controller_cc::deepsort_tracker_controller_cc(float max_cosine_distance, int nn_budget, int k_feature_dim, float max_iou_distance, int max_age, int n_init):
    frame_id(1),
    k_feature_dim(k_feature_dim) {
    obj_tracker = new tracker(max_cosine_distance, nn_budget, k_feature_dim, max_iou_distance, max_age, n_init);
}

deepsort_tracker_controller::deepsort_tracker_controller_cc::~deepsort_tracker_controller_cc() {
    delete obj_tracker;
}

int deepsort_tracker_controller::deepsort_tracker_controller_cc::process(const vector<DeteObjRect>& detected_objects, vector<Tensor>& feature, vector<TrackObjRect>& tracked_objects) {  
    DETECTIONS detections;
    for(int i = 0; i < detected_objects.size(); i++) {
        float start_x = detected_objects[i].left;
        float start_y = detected_objects[i].top;
        float crop_w = detected_objects[i].width; 
        float crop_h = detected_objects[i].height;

        cv::Mat box(1, 4, CV_32F);
        box.at<float>(0) = start_x;
        box.at<float>(1) = start_y;
        box.at<float>(2) = crop_w;
        box.at<float>(3) = crop_h;

        DETECTION_ROW d;
        d.tlwh = box.clone();
        d.confidence = detected_objects[i].score;
        d.class_id = detected_objects[i].class_id;
        detections.push_back(d);
    }

    obj_tracker->predict();
    assert(detected_objects.size() == feature.size() && "The number of detected objects must be equal to the number of features!");
    for(int i = 0; i < detected_objects.size(); i++) {
        if(feature[i].own_sys_data()) {
            // Allocate memory for the FEATURE matrix
            detections[i].feature = cv::Mat(1, k_feature_dim, CV_32F);
            memcpy(detections[i].feature.data, feature[i].sys_data(), k_feature_dim * sizeof(float));
        }
        else if(feature[i].own_dev_data()){
            feature[i].sync_d2s();
            memcpy(detections[i].feature.data, feature[i].sys_data(), k_feature_dim * sizeof(float));
        }
        else {
            return SAIL_ALGO_ERROR_D2S;
        }
    }
    obj_tracker->update(detections);
    int i = 0;
    for (Track& track : obj_tracker->tracks) {
        if ((!track.is_confirmed() || track.time_since_update > 1) && frame_id > 2) { //when frame_id < 2, there is no track.
            continue;
        }
        cv::Mat k = track.to_tlwh();
        TrackObjRect tem;
        tem.left = k.at<float>(0);
        tem.top = k.at<float>(1);
        tem.width = k.at<float>(2);
        tem.height = k.at<float>(3);
        tem.right = tem.left+tem.width;
        tem.bottom = tem.top+tem.height;
        tem.class_id = track.class_id;
        tem.track_id = track.track_id;
        tem.score = 1.0;
        tracked_objects.push_back(tem);
        i++;
    }
    frame_id++;
    return SAIL_ALGO_SUCCESS;
}

int deepsort_tracker_controller::deepsort_tracker_controller_cc::process(const vector<DeteObjRect>& detected_objects, vector<vector<float>>& feature, vector<TrackObjRect>& tracked_objects) {
    DETECTIONS detections;
    for(int i = 0; i < detected_objects.size(); i++) {
        float start_x = detected_objects[i].left;
        float start_y = detected_objects[i].top;
        float crop_w = detected_objects[i].width; 
        float crop_h = detected_objects[i].height;

        cv::Mat box(1, 4, CV_32F);
        box.at<float>(0) = start_x;
        box.at<float>(1) = start_y;
        box.at<float>(2) = crop_w;
        box.at<float>(3) = crop_h;

        DETECTION_ROW d;
        d.tlwh = box.clone();
        d.confidence = detected_objects[i].score;
        d.class_id = detected_objects[i].class_id;
        detections.push_back(d);
    }

    obj_tracker->predict();
    assert(detected_objects.size() == feature.size() && "The number of detected objects must be equal to the number of features!");
    for(int i = 0; i < detected_objects.size(); i++) {
        
        // Allocate memory for the FEATURE matrix
        detections[i].feature = cv::Mat(1, k_feature_dim, CV_32F, feature[i].data());
        // memcpy(detections[i].feature.data, feature[i].sys_data(), k_feature_dim * sizeof(float));
        
    }
    obj_tracker->update(detections);
    int i = 0;
    for (Track& track : obj_tracker->tracks) {
        if ((!track.is_confirmed() || track.time_since_update > 1) && frame_id > 2) { //when frame_id < 2, there is no track.
            continue;
        }
        cv::Mat k = track.to_tlwh();
        TrackObjRect tem;
        tem.left = k.at<float>(0);
        tem.top = k.at<float>(1);
        tem.width = k.at<float>(2);
        tem.height = k.at<float>(3);
        tem.right = tem.left+tem.width;
        tem.bottom = tem.top+tem.height;
        tem.class_id = track.class_id;
        tem.track_id = track.track_id;
        tem.score = 1.0;
        tracked_objects.push_back(tem);
        i++;
    }
    frame_id++;
    return SAIL_ALGO_SUCCESS;
}



deepsort_tracker_controller::deepsort_tracker_controller(float max_cosine_distance, int nn_budget, int k_feature_dim, float max_iou_distance, int max_age, int n_init):
    _impl(new deepsort_tracker_controller_cc(max_cosine_distance, nn_budget, k_feature_dim, max_iou_distance, max_age, n_init)) {
}

deepsort_tracker_controller::~deepsort_tracker_controller() {
    delete _impl;
}

int deepsort_tracker_controller::process(const vector<DeteObjRect>& detected_objects, vector<Tensor>& feature, vector<TrackObjRect>& tracked_objects) {
    return _impl->process(detected_objects, feature, tracked_objects);
}

int deepsort_tracker_controller::process(const vector<DeteObjRect>& detected_objects, vector<vector<float>>& feature, vector<TrackObjRect>& tracked_objects) {
    _impl->process(detected_objects, feature, tracked_objects);
}

// extractor feature输入为tensor
std::vector<std::tuple<int, int, int, int, int, float, int>> deepsort_tracker_controller::process(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short, vector<Tensor>& feature) {
    // 这里输入的detected_objects_short是按照yolov5多线程接口的get_result_npy写的
    // 输入顺序 [left, top, right, bottom, class_id, score]
    // 返回值[left, top, right, bottom, class_id, score, track_id]
    std::vector<DeteObjRect> detected_objects;
    std::vector<TrackObjRect> tracked_objects;

    for (auto item : detected_objects_short) {
        
        DeteObjRect obj;
        obj.left = std::get<0>(item);
        obj.top = std::get<1>(item);
        obj.right = std::get<2>(item);
        obj.bottom = std::get<3>(item);
        obj.class_id = std::get<4>(item);
        obj.score = std::get<5>(item);
        obj.width = obj.right-obj.left;
        obj.height = obj.bottom-obj.top;
        detected_objects.push_back(obj);
    }

    process(detected_objects, feature, tracked_objects);

    std::vector<std::tuple<int, int, int, int, int, float, int>> res;
    for (auto item: tracked_objects){
        int left_temp = item.left;
        int top_temp = item.top;
        int width_temp = item.width;
        int height_temp = item.height;
        int right_temp = left_temp+width_temp;
        int bottom_temp = top_temp+height_temp;
        int class_id_temp = item.class_id;
        float score_temp = item.score;
        int track_id_temp = item.track_id;

        res.push_back(std::make_tuple(left_temp,
                                       top_temp,
                                       right_temp,
                                       bottom_temp,
                                       class_id_temp,
                                       score_temp,
                                       track_id_temp));


    }

    return std::move(res);
}

// extractor feature输入为numpy
std::vector<std::tuple<int, int, int, int, int, float, int>> deepsort_tracker_controller::process(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short, vector<vector<float>>& feature) {
    // 这里输入的detected_objects_short是按照yolov5多线程接口的get_result_npy写的
    // 输入顺序 [left, top, right, bottom, class_id, score]
    // 返回值[left, top, right, bottom, class_id, score, track_id]
    std::vector<DeteObjRect> detected_objects;
    std::vector<TrackObjRect> tracked_objects;

    for (auto item : detected_objects_short) {
        
        DeteObjRect obj;
        obj.left = std::get<0>(item);
        obj.top = std::get<1>(item);
        obj.right = std::get<2>(item);
        obj.bottom = std::get<3>(item);
        obj.class_id = std::get<4>(item);
        obj.score = std::get<5>(item);
        obj.width = obj.right-obj.left;
        obj.height = obj.bottom-obj.top;
        detected_objects.push_back(obj);
    }

    _impl->process(detected_objects, feature, tracked_objects);

    std::vector<std::tuple<int, int, int, int, int, float, int>> res;
    for (auto item: tracked_objects){
        int left_temp = item.left;
        int top_temp = item.top;
        int width_temp = item.width;
        int height_temp = item.height;
        int right_temp = left_temp+width_temp;
        int bottom_temp = top_temp+height_temp;
        int class_id_temp = item.class_id;
        float score_temp = item.score;
        int track_id_temp = item.track_id;

        res.push_back(std::make_tuple(left_temp,
                                       top_temp,
                                       right_temp,
                                       bottom_temp,
                                       class_id_temp,
                                       score_temp,
                                       track_id_temp));


    }

    return std::move(res);
}


// deepsort 异步接口
class deepsort_tracker_controller_async::deepsort_tracker_controller_async_cc {
    public:
    explicit deepsort_tracker_controller_async_cc(float max_cosine_distance, int nn_budget, int k_feature_dim, float max_iou_distance, int max_age, int n_init, int queue_size);
    ~deepsort_tracker_controller_async_cc();

    int push_data(const vector<DeteObjRect>& detected_objects, vector<Tensor>& feature);
    int push_data(const vector<DeteObjRect>& detected_objects, vector<vector<float>>& feature);
    vector<TrackObjRect> get_result();
    void set_processing_timer(bool flag);

    private:
    // asynchronous interface
    std::queue<vector<DeteObjRect>> detected_objects_queue;
    std::queue<vector<Tensor>> feature_tensor_queue;
    std::queue<vector<vector<float>>> feature_float_queue;

    int result_buff_size;
    std::queue<vector<TrackObjRect>> result_queue;

    std::mutex input_mutex;
    std::mutex result_mutex;
    std::condition_variable result_empty;
    std::condition_variable result_full;
    std::condition_variable task_empty;

    bool init,exit;

    bool processing_timer_flag = false;

    void deepsort_tracker_worker_tensor();
    void deepsort_tracker_worker_float();
    thread worker;

    deepsort_tracker_controller* deepsort_tracker_controller_;

};

deepsort_tracker_controller_async::deepsort_tracker_controller_async_cc::deepsort_tracker_controller_async_cc(float max_cosine_distance, int nn_budget, int k_feature_dim, float max_iou_distance, int max_age, int n_init, int queue_size):
     result_buff_size(queue_size),
     init(false),exit(false),
     deepsort_tracker_controller_(new deepsort_tracker_controller(max_cosine_distance, nn_budget, k_feature_dim, max_iou_distance, max_age, n_init)){
}

deepsort_tracker_controller_async::deepsort_tracker_controller_async_cc::~deepsort_tracker_controller_async_cc() {
    exit = true;
    task_empty.notify_all();
    result_full.notify_all();
    worker.join();
    delete deepsort_tracker_controller_;
}

int deepsort_tracker_controller_async::deepsort_tracker_controller_async_cc::push_data(const vector<DeteObjRect>& detected_objects, vector<Tensor>& feature){
#ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    if(!init){
        worker = thread([this]{this->deepsort_tracker_worker_tensor();});
        init = true;
    }
    {
        unique_lock<mutex> lock(input_mutex);
        detected_objects_queue.push(detected_objects);
        feature_tensor_queue.push(feature);
        task_empty.notify_one();
    }
    
    return 0;
}

int deepsort_tracker_controller_async::deepsort_tracker_controller_async_cc::push_data(const vector<DeteObjRect>& detected_objects, vector<vector<float>>& feature){
#ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif
    if(!init){
        worker = thread([this]{this->deepsort_tracker_worker_float();});
        init = true;
    }

    {
        unique_lock<mutex> input_mutex;
        detected_objects_queue.push(detected_objects);
        feature_float_queue.push(feature);
        task_empty.notify_one();
    }
    return 0;
}

std::vector<TrackObjRect> deepsort_tracker_controller_async::deepsort_tracker_controller_async_cc::get_result(){
#ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif    
    unique_lock<mutex> lock(result_mutex);
    result_empty.wait(lock,[this]{return !result_queue.empty();});
    auto tracked_objects = move(result_queue.front());
    result_queue.pop();
    result_full.notify_one();
    return tracked_objects;
}

void deepsort_tracker_controller_async::deepsort_tracker_controller_async_cc::set_processing_timer(bool flag){
    processing_timer_flag = flag;
}

void deepsort_tracker_controller_async::deepsort_tracker_controller_async_cc::deepsort_tracker_worker_tensor(){
    while(true){
        vector<TrackObjRect> tracked_objects;
        vector<DeteObjRect> detected_objects;
        vector<Tensor> feature;
        {
            unique_lock<mutex> lock(input_mutex);
            task_empty.wait(lock,[this]{return exit || (!detected_objects_queue.empty() && !feature_tensor_queue.empty());});
            if(exit){
                SPDLOG_INFO("Deepsort_tracker_worker_tensor Thread Exit");
                return;
            }
            detected_objects = move(detected_objects_queue.front());
            detected_objects_queue.pop();

            feature = move(feature_tensor_queue.front());
            feature_tensor_queue.pop();
        }

        auto start = std::chrono::high_resolution_clock::now();
        int ret = deepsort_tracker_controller_->process(detected_objects,feature,tracked_objects);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = (end-start).count()*1e-9;
        if (processing_timer_flag)
            std::cout << "deepsort processing time: " << duration << " s" <<std::endl;

        {
            unique_lock<mutex> lock(result_mutex);
            std::queue<std::vector<TrackObjRect>> test;

            result_full.wait(lock,[this]{return exit || this->result_queue.size() < result_buff_size;});
            if(exit){
                SPDLOG_INFO("Deepsort_tracker_worker_tensor Thread Exit");
                return;
            }

            result_queue.emplace(move(tracked_objects));
            result_empty.notify_one();

        }
        
    }
}

void deepsort_tracker_controller_async::deepsort_tracker_controller_async_cc::deepsort_tracker_worker_float(){
    while(true){
        vector<TrackObjRect> tracked_objects;
        vector<DeteObjRect> detected_objects;
        vector<vector<float>> feature;
        {
            unique_lock<mutex> lock(input_mutex);
            task_empty.wait(lock,[this]{return exit || (!detected_objects_queue.empty() && !feature_float_queue.empty());});
            if(exit){
                SPDLOG_INFO("Deepsort_tracker_worker_float Thread Exit");
                return;
            }
            detected_objects = move(detected_objects_queue.front());
            detected_objects_queue.pop();

            feature = move(feature_float_queue.front());
            feature_float_queue.pop();
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        int ret = deepsort_tracker_controller_->process(detected_objects,feature,tracked_objects);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = (end-start).count()*1e-9;
        if (processing_timer_flag)
            std::cout << "deepsort processing time: " << duration << " s" <<std::endl;

        {
            unique_lock<mutex> lock(result_mutex);
            std::queue<std::vector<TrackObjRect>> test;

            result_full.wait(lock,[this]{return exit || this->result_queue.size() < result_buff_size;});
            if(exit){
                SPDLOG_INFO("Deepsort_tracker_worker_float Thread Exit");
                return;
            }

            result_queue.emplace(move(tracked_objects));
            result_empty.notify_one();

        }
        
    }
}


deepsort_tracker_controller_async::deepsort_tracker_controller_async(float max_cosine_distance, int nn_budget, int k_feature_dim, float max_iou_distance, int max_age, int n_init, int queue_size):
    _impl(new deepsort_tracker_controller_async_cc(max_cosine_distance, nn_budget, k_feature_dim, max_iou_distance, max_age, n_init,queue_size)){

}

deepsort_tracker_controller_async::~deepsort_tracker_controller_async(){
    SPDLOG_INFO("deepsort_tracker_controller_async finish work!");
    delete _impl;
}

// cpp func
int deepsort_tracker_controller_async::push_data(const vector<DeteObjRect>& detected_objects, vector<Tensor>& feature){
    return _impl->push_data(detected_objects,feature);
}

int deepsort_tracker_controller_async::push_data(const vector<DeteObjRect>& detected_objects, vector<vector<float>>& feature){
    return _impl->push_data(detected_objects,feature);
}

std::vector<TrackObjRect> deepsort_tracker_controller_async::get_result(){
    return _impl->get_result();
}

void deepsort_tracker_controller_async::set_processing_timer(bool flag){
    return _impl->set_processing_timer(flag);
}

// python func
int deepsort_tracker_controller_async::push_data(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short, vector<Tensor>& feature){
    std::vector<DeteObjRect> detected_objects;
    for (auto item : detected_objects_short) {
        DeteObjRect obj;
        obj.left = std::get<0>(item);
        obj.top = std::get<1>(item);
        obj.right = std::get<2>(item);
        obj.bottom = std::get<3>(item);
        obj.class_id = std::get<4>(item);
        obj.score = std::get<5>(item);
        obj.width = obj.right-obj.left;
        obj.height = obj.bottom-obj.top;
        detected_objects.push_back(obj);
    }
    return push_data(detected_objects,feature);
}

int deepsort_tracker_controller_async::push_data(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short, vector<vector<float>>& feature){
    std::vector<DeteObjRect> detected_objects;
    for (auto item : detected_objects_short) {
        DeteObjRect obj;
        obj.left = std::get<0>(item);
        obj.top = std::get<1>(item);
        obj.right = std::get<2>(item);
        obj.bottom = std::get<3>(item);
        obj.class_id = std::get<4>(item);
        obj.score = std::get<5>(item);
        obj.width = obj.right-obj.left;
        obj.height = obj.bottom-obj.top;
        detected_objects.push_back(obj);
    }
    return push_data(detected_objects,feature);
}

std::vector<std::tuple<int, int, int, int, int, float, int>> deepsort_tracker_controller_async::get_result_npy(){
    std::vector<TrackObjRect> tracked_objects = get_result();
    // transfer to tuple
    std::vector<std::tuple<int, int, int, int, int, float, int>> tracked_objects_npy;
    for(auto& track_object:tracked_objects){
        tracked_objects_npy.push_back(std::make_tuple(track_object.left,
                                                      track_object.top,
                                                      (int)(track_object.left)+(int)(track_object.width),
                                                      (int)(track_object.top)+(int)(track_object.height),
                                                      track_object.class_id,
                                                      track_object.score,
                                                      track_object.track_id));
    }
    return tracked_objects_npy;
}

class bytetrack_tracker_controller::bytetrack_tracker_controller_cc {
    public:
    explicit bytetrack_tracker_controller_cc(int frame_rate = 30, int track_buffer = 30);
    ~bytetrack_tracker_controller_cc();
    int process(const vector<DeteObjRect>& detected_objects, vector<TrackObjRect>& tracked_objects);

    private:
    BYTETracker* obj_tracker;
};

bytetrack_tracker_controller::bytetrack_tracker_controller_cc::bytetrack_tracker_controller_cc(int frame_rate, int track_buffer) {
    obj_tracker = new BYTETracker(frame_rate, track_buffer);
}

bytetrack_tracker_controller::bytetrack_tracker_controller_cc::~bytetrack_tracker_controller_cc() {
    delete obj_tracker;
}

int bytetrack_tracker_controller::bytetrack_tracker_controller_cc::process(const vector<DeteObjRect>& detected_objects, vector<TrackObjRect>& tracked_objects) {
    std::vector<std::vector<float>> objects;
    for(int i = 0; i < detected_objects.size(); i++) {
        std::vector<float> tmp = {detected_objects[i].left, detected_objects[i].top, detected_objects[i].width, detected_objects[i].height, detected_objects[i].score, float(detected_objects[i].class_id)};
        objects.push_back(tmp);
    }
    vector<STrack> output_stracks = obj_tracker->update(objects);
    tracked_objects.clear();
    for(auto& output_strack : output_stracks) {
        TrackObjRect tmp;
        tmp.left = output_strack.tlwh[0];
        tmp.top = output_strack.tlwh[1];
        tmp.width = output_strack.tlwh[2];
        tmp.height = output_strack.tlwh[3];
        tmp.score = output_strack.score;
        tmp.class_id = output_strack.class_id;
        tmp.track_id = output_strack.track_id;
        tracked_objects.push_back(tmp);
    }
    return SAIL_ALGO_SUCCESS;
}

bytetrack_tracker_controller::bytetrack_tracker_controller(int frame_rate, int track_buffer):
    _impl(new bytetrack_tracker_controller_cc(frame_rate, track_buffer)) {
}

bytetrack_tracker_controller::~bytetrack_tracker_controller() {
    delete _impl;
}

int bytetrack_tracker_controller::process(const vector<DeteObjRect>& detected_objects, vector<TrackObjRect>& tracked_objects) {
    return _impl->process(detected_objects, tracked_objects);
}



std::vector<std::tuple<int, int, int, int, int, float, int>> bytetrack_tracker_controller::process(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short) {
    // 这里输入的detected_objects_short是按照yolov5多线程接口的get_result_npy写的
    // 输入顺序 [left, top, right, bottom, class_id, score]
    // 返回值[left, top, right, bottom, class_id, score, track_id]
    std::vector<DeteObjRect> detected_objects;
    std::vector<TrackObjRect> tracked_objects;

    for (auto item : detected_objects_short) {
        
        DeteObjRect obj;
        obj.left = std::get<0>(item);
        obj.top = std::get<1>(item);
        obj.right = std::get<2>(item);
        obj.bottom = std::get<3>(item);
        obj.class_id = std::get<4>(item);
        obj.score = std::get<5>(item);
        obj.width = obj.right-obj.left;
        obj.height = obj.bottom-obj.top;
        detected_objects.push_back(obj);
    }

    process(detected_objects, tracked_objects);

    std::vector<std::tuple<int, int, int, int, int, float, int>> res;
    for (auto item: tracked_objects){
        int left_temp = item.left;
        int top_temp = item.top;
        int width_temp = item.width;
        int height_temp = item.height;
        int right_temp = left_temp+width_temp;
        int bottom_temp = top_temp+height_temp;
        int class_id_temp = item.class_id;
        float score_temp = item.score;
        int track_id_temp = item.track_id;

        res.push_back(std::make_tuple(left_temp,
                                       top_temp,
                                       right_temp,
                                       bottom_temp,
                                       class_id_temp,
                                       score_temp,
                                       track_id_temp));


    }

    return std::move(res);
}



// SORT算法接口实现
// 内部封装接口 
class sort_tracker_controller::sort_tracker_controller_cc{
    public:
        explicit sort_tracker_controller_cc(float max_iou_distance, int max_age, int n_init);
        ~sort_tracker_controller_cc();
        std::vector<std::tuple<int, int, int, int, int, float, int>> process(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short);
    
    private:
        int frame_id;
        sort_tracker* obj_tracker;
};

// sort 算法内部实现构造函数
sort_tracker_controller::sort_tracker_controller_cc::sort_tracker_controller_cc(float max_iou_distance = 0.7, 
                                                                                int max_age = 30, 
                                                                                int n_init = 3):frame_id(1){
    obj_tracker = new sort_tracker(max_iou_distance, max_age, n_init);
}

// sort 算法内部实现的析构函数
sort_tracker_controller::sort_tracker_controller_cc::~sort_tracker_controller_cc(){
    delete obj_tracker;
}


// sort 算法实现
// 输入顺序 [left, top, right, bottom, class_id, score]
std::vector<std::tuple<int, int, int, int, int, float, int>> sort_tracker_controller::sort_tracker_controller_cc::process(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short){

    // 创建输入的检测bbox
    DETECTIONS detections;
    for( auto& item : detected_objects_short){

        float start_x = std::get<0>(item);
        float start_y = std::get<1>(item);

        float crop_w = std::get<2>(item) - start_x;
        float crop_h = std::get<3>(item) - start_y;

        cv::Mat box(1, 4, CV_32F);
        box.at<float>(0) = start_x;
        box.at<float>(1) = start_y;
        box.at<float>(2) = crop_w;
        box.at<float>(3) = crop_h;

        DETECTION_ROW d;
        d.tlwh = box.clone();
        d.confidence = std::get<5>(item);
        d.class_id = std::get<4>(item);
        detections.push_back(d);
    
    }

    // 生成跟踪的目标
    std::vector<TrackObjRect> tracked_objects;

    // 卡尔曼滤波预测
    obj_tracker -> predict();

    // 卡尔曼滤波匹配更新
    obj_tracker->update(detections);
    
    // 返回生成的结果
    // 返回值[left, top, right, bottom, class_id, score, track_id]
    std::vector<std::tuple<int, int, int, int, int, float, int>> res;

    for(SortTrack& track : obj_tracker->tracks){

        if ((!track.is_confirmed() || track.time_since_update > 1) && frame_id > 2) { //when frame_id < 2, there is no track.
            continue;
        }

        cv::Mat k = track.to_tlwh();
        TrackObjRect tem;

        int left_temp = k.at<float>(0);
        int top_temp = k.at<float>(1);
        int right_temp = left_temp + k.at<float>(2);
        int bottom_temp = top_temp + k.at<float>(3);
        int class_id_temp = track.class_id;
        float score_temp = 1.0;
        int track_id_temp = track.track_id;

        res.push_back(std::make_tuple(left_temp,
                                top_temp,
                                right_temp,
                                bottom_temp,
                                class_id_temp,
                                score_temp,
                                track_id_temp));
    }

    frame_id ++;
    return res;

}

// 对外暴露的接口
// 构造函数
sort_tracker_controller::sort_tracker_controller(float max_iou_distance, int max_age, int n_init):
    _impl(new sort_tracker_controller_cc(max_iou_distance, max_age, n_init)){
}

// 析构函数
sort_tracker_controller::~sort_tracker_controller(){
    delete _impl;
    _impl = nullptr;
}

// 处理接口
std::vector<std::tuple<int, int, int, int, int, float, int>> sort_tracker_controller::process(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short){
    return _impl->process(detected_objects_short);
}


// SORT算法异步接口实现
class sort_tracker_controller_async::sort_tracker_controller_async_cc{
    public:
        explicit sort_tracker_controller_async_cc(float max_iou_distance = 0.3, int max_age = 30, int n_init = 3, int input_queue_size = 10, int result_queue_size = 10);
        ~sort_tracker_controller_async_cc();

        int push_data(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short);
        std::vector<std::tuple<int, int, int, int, int, float, int>> get_result_npy();

    private:
        // asynchronous interface
        int detected_buff_size;
        std::queue<vector<std::tuple<int, int, int, int ,int, float>>> detected_objects_queue;

        int result_buff_size;
        std::queue<vector<std::tuple<int, int, int, int, int, float, int>>> result_queue;

        std::mutex input_mutex;
        std::mutex result_mutex;
        std::condition_variable result_empty;
        std::condition_variable result_full;
        std::condition_variable task_empty;
        std::condition_variable task_full;

        bool init,exit;

        void sort_tracker_worker();
        thread worker;

        sort_tracker_controller* sort_tracker_controller_;
        
};

// 构造函数
sort_tracker_controller_async::sort_tracker_controller_async_cc::sort_tracker_controller_async_cc(
    float max_iou_distance, int max_age, int n_init, int input_queue_size, int result_queue_size):
        detected_buff_size(input_queue_size),result_buff_size(result_queue_size),exit(false),init(false),
        sort_tracker_controller_(new sort_tracker_controller(max_iou_distance,max_age,n_init)){

}


sort_tracker_controller_async::sort_tracker_controller_async_cc::~sort_tracker_controller_async_cc(){
    exit = true;
    task_empty.notify_all();
    task_full.notify_all();
    result_full.notify_all();
    result_empty.notify_all();


    worker.join();
    delete sort_tracker_controller_;
}       

int sort_tracker_controller_async::sort_tracker_controller_async_cc::push_data(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short){

#ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif

    if(!init){
        worker = thread([this]{this->sort_tracker_worker();});
        init = true;
    }

    {
        unique_lock<mutex> lock(input_mutex);
        if (this->detected_objects_queue.size() >= detected_buff_size){
            return -2;
        }
        // task_full.wait(lock,[this]{return exit || this->detected_objects_queue.size() < detected_buff_size ;});
        if(exit){
            SPDLOG_INFO("sort_tracker_worker_tensor Thread Exit");
            return -1;
        }
        detected_objects_queue.push(detected_objects_short);
    }
    task_empty.notify_one();
    
    return 0;
}

std::vector<std::tuple<int, int, int, int, int, float, int>> sort_tracker_controller_async::sort_tracker_controller_async_cc::get_result_npy(){

#ifdef PYTHON
    pybind11::gil_scoped_release release;
#endif

    std::vector<std::tuple<int, int, int, int, int, float, int>> tracked_objects;
    {
        unique_lock<mutex> lock(result_mutex);
        result_empty.wait(lock,[this]{return !result_queue.empty();});
        tracked_objects = move(result_queue.front());
        result_queue.pop();
    }

    result_full.notify_one();

    return tracked_objects;
}

void sort_tracker_controller_async::sort_tracker_controller_async_cc::sort_tracker_worker(){
    while(true){

        vector<std::tuple<int, int, int, int ,int, float>> detected_objects;
        {
            unique_lock<mutex> lock(input_mutex);
            task_empty.wait(lock,[this]{return exit || !detected_objects_queue.empty();});
            if(exit){
                SPDLOG_INFO("sort_tracker_worker Thread Exit");
                return;
            }
            detected_objects = move(detected_objects_queue.front());
            detected_objects_queue.pop();
        }

        task_full.notify_one();
        auto tracked_objects = sort_tracker_controller_->process(detected_objects);

        {
            unique_lock<mutex> lock(result_mutex);
            result_full.wait(lock,[this]{return exit || this->result_queue.size() < result_buff_size;});
            if(exit){
                SPDLOG_INFO("sort_tracker_worker_tensor Thread Exit");
                return;
            }

            result_queue.emplace(move(tracked_objects));
        }
        result_empty.notify_one();
        
    }
}


sort_tracker_controller_async::sort_tracker_controller_async(float max_iou_distance, int max_age, int n_init, int input_queue_size, int result_queue_size):
    _impl(new sort_tracker_controller_async_cc(max_iou_distance,max_age,n_init,input_queue_size,result_queue_size)){
}

sort_tracker_controller_async::~sort_tracker_controller_async(){
    delete _impl;
}

int sort_tracker_controller_async::push_data(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short){
    return  _impl->push_data(detected_objects_short);
}

std::vector<std::tuple<int, int, int, int, int, float, int>> sort_tracker_controller_async::get_result_npy(){
    return _impl->get_result_npy();
}
#endif //USE_OPENCV

// rotated nms

Point64D addPoints(Point64D p1, Point64D p2) {
    Point64D result;
    result.x = p1.x + p2.x;
    result.y = p1.y + p2.y;
    return result;
}

Point64D subtractPoints(Point64D p1, Point64D p2) {
    Point64D result;
    result.x = p1.x - p2.x;
    result.y = p1.y - p2.y;
    return result;
}

double dot_2d(const Point64D *A, const Point64D *B) {
    return A->x * B->x + A->y * B->y;
}


double cross_2d( Point64D *A, Point64D *B) {
   return A->x * B->y - A->y * B->x;
}

void get_rotated_vertices(RotatedBox32F *box, Point64D pts[4]) {
    double theta = box->a;
    double cosTheta2 = (double)cos(theta) * 0.5f;
    double sinTheta2 = (double)sin(theta) * 0.5f;

    pts[0].x = box->x_ctr + sinTheta2 * box->h + cosTheta2 * box->w;
    pts[0].y = box->y_ctr + cosTheta2 * box->h - sinTheta2 * box->w;
    pts[1].x = box->x_ctr - sinTheta2 * box->h + cosTheta2 * box->w;
    pts[1].y = box->y_ctr - cosTheta2 * box->h - sinTheta2 * box->w;
    pts[2].x = 2 * box->x_ctr - pts[0].x;
    pts[2].y = 2 * box->y_ctr - pts[0].y;
    pts[3].x = 2 * box->x_ctr - pts[1].x;
    pts[3].y = 2 * box->y_ctr - pts[1].y;
}

int get_intersection_points(const Point64D pts1[4], const Point64D pts2[4], Point64D intersections[24]) {
    Point64D vec1[4], vec2[4];
    for (int i = 0; i < 4; i++) {
        vec1[i] = subtractPoints(pts1[(i + 1) % 4], pts1[i]);
        vec2[i] = subtractPoints(pts2[(i + 1) % 4], pts2[i]);
    }

    int num = 0; 
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double det = cross_2d(&vec2[j], &vec1[i]);

            if (fabs(det) <= 1e-14) {
                continue;
            }

            Point64D vec12 = subtractPoints(pts2[j], pts1[i]);

            double t1 = cross_2d(&vec2[j], &vec12) / det;
            double t2 = cross_2d(&vec1[i], &vec12) / det;

            if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f) {
                intersections[num].x = pts1[i].x + vec1[i].x * t1;
                intersections[num].y = pts1[i].y + vec1[i].y * t1;
                num++;
            }
        }
    }

    {
        Point64D AB = vec2[0];
        Point64D DA = vec2[3];
        double ABdotAB = dot_2d(&AB, &AB);
        double ADdotAD = dot_2d(&DA, &DA);
        for (int i = 0; i < 4; i++) {
            Point64D AP = subtractPoints(pts1[i], pts2[0]);

            double APdotAB = dot_2d(&AP, &AB);
            double APdotAD = -dot_2d(&AP, &DA);

            if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
                (APdotAD <= ADdotAD)) {
                intersections[num] = pts1[i];
                num++;
            }
        }
    }

    {
        Point64D AB = vec1[0];
        Point64D DA = vec1[3];
        double ABdotAB = dot_2d(&AB, &AB);
        double ADdotAD = dot_2d(&DA, &DA);
        for (int i = 0; i < 4; i++) {
            Point64D AP = subtractPoints(pts2[i], pts1[0]);

            double APdotAB = dot_2d(&AP, &AB);
            double APdotAD = -dot_2d(&AP, &DA);

            if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
                (APdotAD <= ADdotAD)) {
                intersections[num] = pts2[i];
                num++;
            }
        }
    }

    return num;
}


int point_order_compare(const void *a, const void *b) {
    Point64D A = *((Point64D *)a);
    Point64D B = *((Point64D *)b);

    double temp = cross_2d(&A, &B);
    if (fabs(temp) < 1e-6) {
        return dot_2d(&A, &A) - dot_2d(&B, &B);
    } else {
        return temp < 0 ? 1 : -1;
    }
}


int convex_hull_graham(const Point64D p[24], const int num_in, Point64D q[24], int shift_to_zero) {

    int t = 0;
    for (int i = 1; i < num_in; i++) {
        if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x)) {
            t = i;
        }
    }
    Point64D start = p[t];

    for (int i = 0; i < num_in; i++) {
        q[i] = subtractPoints(p[i], start);
    }

    Point64D tmp = q[0];
    q[0] = q[t];
    q[t] = tmp;

    double dist[24];

    qsort(q + 1, num_in - 1, sizeof(Point64D), point_order_compare);

    for (int i = 0; i < num_in; i++) {
        dist[i] = dot_2d(&q[i], &q[i]);
    }

    int k;
    for (k = 1; k < num_in; k++) {
        if (dist[k] > 1e-8) {
            break;
        }
    }

    if (k == num_in) {
        q[0] = p[t];
        return 1;
    }

    q[1] = q[k];
    int m = 2;

    for (int i = k + 1; i < num_in; i++) {
        while (m > 1) {
            Point64D q1 = subtractPoints(q[i], q[m - 2]);
            Point64D q2 = subtractPoints(q[m - 1], q[m - 2]);
            if (q1.x * q2.y >= q2.x * q1.y)
                m--;
            else
                break;
        }

        q[m++] = q[i];
    }

    if (!shift_to_zero) {
        for (int i = 0; i < m; i++) {
            q[i] = addPoints(q[i], start);
        }
    }

    return m;
}



double polygon_area(const Point64D q[24], const int m) {
    if (m <= 2) {
        return 0;
    }

    double area = 0.0;
    for (int i = 1; i < m - 1; i++) {
        Point64D diff1 = subtractPoints(q[i], q[0]);
        Point64D diff2 = subtractPoints(q[i + 1], q[0]);
        area += fabs(cross_2d(&diff1, &diff2));
    }

    return area / 2.0;
}


double rotated_boxes_intersection(Point64D *pts1, Point64D *pts2) {
    Point64D intersectPts[24], orderedPts[24];

    int num = get_intersection_points(pts1, pts2, intersectPts);

    if (num <= 2) {
        return 0.0;
    }

    int num_convex = convex_hull_graham(intersectPts, num, orderedPts, 1);

    auto res = polygon_area(orderedPts, num_convex);

    return res;
}


double single_box_iou_rotated(RotatedBox32F box1_raw, RotatedBox32F box2_raw, Point64D *pts1, Point64D *pts2) {

    double area1 = box1_raw.w * box1_raw.h;
    double area2 = box2_raw.w * box2_raw.h;

    if (area1 < 1e-14 || area2 < 1e-14) {
        return 0.f;
    }

    double intersection = rotated_boxes_intersection(pts1, pts2);
    double iou = intersection / (area1 + area2 - intersection);
    return iou;
}

std::vector<int> nms_rotated(std::vector<std::vector<float>>& boxes, std::vector<float>& scores, float threshold) {
    std::vector<int> keep_index;
    if (boxes.size() != scores.size()){
      spdlog::error("nms_rotated: The sizes of boxes and scores are not equal, with {} boxes and {} scores", boxes.size(), scores.size());
      throw std::invalid_argument("The lengths of boxes and scores are inconsistent");
    }
    int dets_num = scores.size(); 
    int dets_dim = 5;
    int keep[dets_num];
    int order[dets_num];

    for (int i = 0; i < dets_num; ++i)
    {
        order[i] = i;
    }

    double suppressed[dets_num];
    memset(suppressed, 0, sizeof(float)*dets_num);

    std::sort(order, order+dets_num, [&scores](int i, int j){
      return scores[i]>scores[j];
    });

    Point64D pts_all[dets_num][4];
    RotatedBox32F rot_boxes[dets_num];

    for (int i = 0; i < dets_num; ++i){
      rot_boxes[i].x_ctr = boxes[i][0];
      rot_boxes[i].y_ctr = boxes[i][1];
      rot_boxes[i].w = boxes[i][2];
      rot_boxes[i].h = boxes[i][3];
      rot_boxes[i].a = boxes[i][4];
      get_rotated_vertices(&rot_boxes[i], pts_all[i]);
    }

    int k = 0;
    for (int i = 0; i < dets_num; ++i) {
        int idx = order[i];
        if (suppressed[idx] == 1) {
            continue;
        }

        keep[k] = idx;
        k += 1;

        for (int j = i + 1; j < dets_num; ++j) {
            int next_idx = order[j];

            if (suppressed[next_idx] == 1) {
                continue;
            }

            double iou = single_box_iou_rotated(rot_boxes[idx], rot_boxes[next_idx], pts_all[idx], pts_all[next_idx]);
 
            if (iou >= threshold) {
                suppressed[next_idx] = 1;
            }
        }
    }

    for (int i = 0; i < k; i++) {
       keep_index.emplace_back(keep[i]);
    }

    return keep_index;
}

}