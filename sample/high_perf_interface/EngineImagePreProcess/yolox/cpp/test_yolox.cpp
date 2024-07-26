#include "test_yolox.h"
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>

using namespace cv;

double get_time_us()
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

float overlap_FM(float x1, float w1, float x2, float w2)
{
	float l1 = x1;
	float l2 = x2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1;
	float r2 = x2 + w2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

float box_intersection_FM(ObjRect a, ObjRect b)
{
	float w = overlap_FM(a.left, a.width, b.left, b.width);
	float h = overlap_FM(a.top, a.height, b.top, b.height);
	if (w < 0 || h < 0) return 0;
	float area = w*h;
	return area;
}

float box_union_FM(ObjRect a, ObjRect b)
{
	float i = box_intersection_FM(a, b);
	float u = a.width*a.height + b.width*b.height - i;
	return u;
}

float box_iou_FM(ObjRect a, ObjRect b)
{
	return box_intersection_FM(a, b) / box_union_FM(a, b);
}

static bool sort_ObjRect(ObjRect a, ObjRect b)
{
    return a.score > b.score;
}

static void nms_sorted_bboxes(const std::vector<ObjRect>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = objects.size();

    for (int i = 0; i < n; i++)    {
        const ObjRect& a = objects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const ObjRect& b = objects[picked[j]];

            float iou = box_iou_FM(a, b);
            if (iou > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}

MultiProcessor::MultiProcessor(int tpu_id, std::vector<std::string> video_list, std::string bmodel_name, sail_resize_type resize_type, int queue_in_size, int queue_out_size)
{
    tpu_id_ = tpu_id;
    for(int i = 0; i < video_list.size(); ++i){
        video_list_.push_back(video_list[i]);
    }
    multi_decoder_ = new MultiDecoder(10, tpu_id_);
    multi_decoder_->set_local_flag(true);
    for(int i = 0; i < video_list.size(); ++i){
        // int channel_index = multi_decoder_->add_channel(video_list[i],1);
        int channel_index = multi_decoder_->add_channel(video_list[i]);
        video_list_map.insert(std::pair<int,std::string>(channel_index, video_list[i]));
    }
    flag_exit_ = false;
    thread_ended = true;
    alpha_beta = std::make_tuple(std::pair<float,float>(1,0),std::pair<float,float>(1,0),std::pair<float,float>(1,0));
    InitEngineImagePreProcess(tpu_id, bmodel_name, resize_type, queue_in_size, queue_out_size);
    start_decoder_thread();
}

MultiProcessor::~MultiProcessor()
{
    SPDLOG_INFO("Start set_exit_flag.....");
    set_exit_flag(true);
    SPDLOG_INFO("End set_exit_flag, and waiting for thread to finish....");

    while(true){
        {
            std::lock_guard<std::mutex> lock_mutex(mutex_thread_ended);
            if(thread_ended){
                break;
            }
        }
        SPDLOG_INFO("Thread Not finished, sleep 500ms!");
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        continue;
    }

    SPDLOG_INFO("Thread Finished!");
    SPDLOG_INFO("Start delete multi_decoder_");
    delete multi_decoder_;
    SPDLOG_INFO("Start delete engine_image_pre_process_");
    delete engine_image_pre_process_;
}

void  MultiProcessor::InitEngineImagePreProcess(int tpu_id, std::string bmodel_name, sail_resize_type resize_type, int queue_in_size, int queue_out_size)
{
    engine_image_pre_process_ = new EngineImagePreProcess(bmodel_name,tpu_id, true);
    engine_image_pre_process_->InitImagePreProcess(resize_type, false, queue_in_size,queue_out_size);
    engine_image_pre_process_->SetPaddingAtrr();
    engine_image_pre_process_->SetConvertAtrr(alpha_beta);
}

void MultiProcessor::decoder_and_preprocess()
{
    int image_idx = 0;
    sail::Handle handle(tpu_id_);
    sail::Bmcv bmcv(handle);
    while(true){
        thread_ended = false;
        if(get_exit_flag()){
            break;
        }
        auto iter = video_list_map.begin();
        while(iter != video_list_map.end()){
            int channel_index = iter->first;
            sail::BMImage image_temp;
            int ret = multi_decoder_->read(channel_index, image_temp);
            if (ret == 0)  {
                ret = engine_image_pre_process_->PushImage(channel_index, image_idx, image_temp);
                if(ret == 0)
                    image_idx++;
            }
            iter++;
        }
    }
    std::lock_guard<std::mutex> lock_mutex(mutex_thread_ended);
    thread_ended = true;
    SPDLOG_INFO("decoder_and_preprocess thread finished.....");
}

bool MultiProcessor::get_exit_flag()
{
    std::lock_guard<std::mutex> lock_mutex(mutex_exit);
    return flag_exit_;
}

void MultiProcessor::set_exit_flag(bool flag)
{
    std::lock_guard<std::mutex> lock_mutex(mutex_exit);
    flag_exit_ = flag;
}

void MultiProcessor::start_decoder_thread()
{
    std::thread thread_decoder = std::thread(&MultiProcessor::decoder_and_preprocess,this);
    thread_decoder.detach();
}

int MultiProcessor::get_input_width()
{
    return engine_image_pre_process_->get_input_width();
}

int MultiProcessor::get_input_height()
{
    return engine_image_pre_process_->get_input_height();
}

vector<int> MultiProcessor::get_output_shape()
{
    string output_name = engine_image_pre_process_->get_output_names()[0];
    return engine_image_pre_process_->get_output_shape(output_name);
}

std::tuple<std::map<std::string,sail::Tensor*>,std::vector<cv::Mat>,std::vector<int>,std::vector<int>,std::vector<std::vector<int>>> MultiProcessor::GetBatchData()
{
    return std::move(engine_image_pre_process_->GetBatchData_CV());
}

YoloX_PostForward::YoloX_PostForward(int net_w, int net_h, std::vector<int> strides):network_width(net_w),network_height(net_h)
{
  outlen_diml = 0;
  for (int i=0;i<strides.size();++i)  {
    int layer_w = net_w/strides[i];
    int layer_h = net_h/strides[i];
    outlen_diml += layer_h*layer_w;
  }
  grids_x_ = new int[outlen_diml];
  grids_y_ = new int[outlen_diml];
  expanded_strides_ = new int[outlen_diml];

  int channel_len = 0;
  for (int i=0;i<strides.size();++i)  {
    int layer_w = net_w/strides[i];
    int layer_h = net_h/strides[i];
    for (int m = 0; m < layer_h; ++m)   {
      for (int n = 0; n < layer_w; ++n)    {
          grids_x_[channel_len+m*layer_w+n] = n;
          grids_y_[channel_len+m*layer_w+n] = m;
          expanded_strides_[channel_len+m*layer_w+n] = strides[i];
      }
    }
    channel_len += layer_w * layer_h;
  }
}

YoloX_PostForward::~YoloX_PostForward()
{
  delete grids_x_;
  grids_x_ = NULL;
  delete grids_y_;
  grids_y_ = NULL;
  delete expanded_strides_;
  expanded_strides_ = NULL;
}

void YoloX_PostForward::process(float* data_ptr,std::vector<int> output_shape, std::vector<std::pair<int,int>> ost_size, 
  float threshold, float nms_threshold, std::vector<std::vector<ObjRect>> &detections)
{
  int size_one_batch = output_shape[1]*output_shape[2];
  int channels_resu_ = output_shape[2];
  int classes_ = channels_resu_ - 5;

  for (int batch_idx=0; batch_idx<ost_size.size();batch_idx++){
    int batch_start_ptr = size_one_batch * batch_idx;
    std::vector<ObjRect> dect_temp;
    dect_temp.clear();
    float scale_x = (float)ost_size[batch_idx].first/network_width;
    float scale_y = (float)ost_size[batch_idx].second/network_height;
    for (size_t i = 0; i < outlen_diml; i++)    {
        int ptr_start=i*channels_resu_;
        float box_objectness = data_ptr[batch_start_ptr + ptr_start+4];
        if(data_ptr[batch_start_ptr + ptr_start+4] >= threshold){
            float center_x = (data_ptr[batch_start_ptr +ptr_start+0] + grids_x_[i]) * expanded_strides_[i];
            float center_y = (data_ptr[batch_start_ptr +ptr_start+1] + grids_y_[i]) * expanded_strides_[i];
            float w_temp = exp(data_ptr[batch_start_ptr +ptr_start+2])*expanded_strides_[i];
            float h_temp = exp(data_ptr[batch_start_ptr +ptr_start+3])*expanded_strides_[i];
            float score = data_ptr[batch_start_ptr +ptr_start+4];
            center_x *= scale_x;
            center_y *= scale_y;
            w_temp *= scale_x;
            h_temp *= scale_y;
            float left = center_x - w_temp/2;
            float top = center_y - h_temp/2;
            float right = center_x + w_temp/2;
            float bottom = center_y + h_temp/2;

            // printf("[%.2f,%.2f,%.2f,%.2f]::::%f,%f\n",center_x,center_y,w_temp,h_temp,scale_x,scale_y);

            for (int class_idx = 0; class_idx < classes_; class_idx++)       {
                float box_cls_score = data_ptr[batch_start_ptr +ptr_start + 5 + class_idx];
                float box_prob = box_objectness * box_cls_score;
                if (box_prob > threshold)         {
                    ObjRect obj_temp;
                    obj_temp.width = w_temp;
                    obj_temp.height = h_temp;
                    obj_temp.left = left;
                    obj_temp.top = top;
                    obj_temp.right = right;
                    obj_temp.bottom = bottom;
                    obj_temp.score = box_prob;
                    obj_temp.class_id = class_idx;
                    dect_temp.push_back(obj_temp);
                }
            }
        }
    }

    std::sort(dect_temp.begin(),dect_temp.end(),sort_ObjRect);

    std::vector<ObjRect> dect_temp_batch;
    std::vector<int> picked;
    dect_temp_batch.clear();
    nms_sorted_bboxes(dect_temp, picked, nms_threshold);

    for (size_t i = 0; i < picked.size(); i++)    {
        dect_temp_batch.push_back(dect_temp[picked[i]]);
    }
    
    detections.push_back(dect_temp_batch);
  }
}

void YoloX_PostForward::process(float* data_ptr,std::vector<int> output_shape, std::vector<float> resize_scale, 
  float threshold, float nms_threshold, std::vector<std::vector<ObjRect>> &detections)
{
  int size_one_batch = output_shape[1]*output_shape[2];
  int channels_resu_ = output_shape[2];
  int classes_ = channels_resu_ - 5;


  for (int batch_idx=0; batch_idx<resize_scale.size();batch_idx++){
    int batch_start_ptr = size_one_batch * batch_idx;
    std::vector<ObjRect> dect_temp;
    dect_temp.clear();
    float scale_x = 1.0/resize_scale[batch_idx];
    float scale_y = 1.0/resize_scale[batch_idx];
    
    for (size_t i = 0; i < outlen_diml; i++)    {
        int ptr_start=i*channels_resu_;
        float box_objectness = data_ptr[batch_start_ptr + ptr_start+4];
        if(data_ptr[batch_start_ptr + ptr_start+4] >= threshold){
            float center_x = (data_ptr[batch_start_ptr +ptr_start+0] + grids_x_[i]) * expanded_strides_[i];
            float center_y = (data_ptr[batch_start_ptr +ptr_start+1] + grids_y_[i]) * expanded_strides_[i];
            float w_temp = exp(data_ptr[batch_start_ptr +ptr_start+2])*expanded_strides_[i];
            float h_temp = exp(data_ptr[batch_start_ptr +ptr_start+3])*expanded_strides_[i];
            float score = data_ptr[batch_start_ptr +ptr_start+4];
            center_x *= scale_x;
            center_y *= scale_y;
            w_temp *= scale_x;
            h_temp *= scale_y;
            float left = center_x - w_temp/2;
            float top = center_y - h_temp/2;
            float right = center_x + w_temp/2;
            float bottom = center_y + h_temp/2;

            // printf("[%.2f,%.2f,%.2f,%.2f]::::%f,%f\n",center_x,center_y,w_temp,h_temp,scale_x,scale_y);

            for (int class_idx = 0; class_idx < classes_; class_idx++)       {
                float box_cls_score = data_ptr[batch_start_ptr +ptr_start + 5 + class_idx];
                float box_prob = box_objectness * box_cls_score;
                if (box_prob > threshold)         {
                    ObjRect obj_temp;
                    obj_temp.width = w_temp;
                    obj_temp.height = h_temp;
                    obj_temp.left = left;
                    obj_temp.top = top;
                    obj_temp.right = right;
                    obj_temp.bottom = bottom;
                    obj_temp.score = box_prob;
                    obj_temp.class_id = class_idx;
                    dect_temp.push_back(obj_temp);
                }
            }
        }
    }

    std::sort(dect_temp.begin(),dect_temp.end(),sort_ObjRect);

    std::vector<ObjRect> dect_temp_batch;
    std::vector<int> picked;
    dect_temp_batch.clear();
    nms_sorted_bboxes(dect_temp, picked, nms_threshold);

    for (size_t i = 0; i < picked.size(); i++)    {
        dect_temp_batch.push_back(dect_temp[picked[i]]);
    }
    
    detections.push_back(dect_temp_batch);
  }
}

YoloxPostProcessThread::YoloxPostProcessThread(int net_w, int net_h, std::vector<int> strides)
{
    postprocessor = new YoloX_PostForward(net_w, net_h, strides);
    padding_flag = false;

    flag_exit_ = false;
    thread_ended = true;
}

YoloxPostProcessThread::~YoloxPostProcessThread()
{
    SPDLOG_INFO("Start ~YoloxPostProcessThread.....");
    set_exit_flag(true);
    SPDLOG_INFO("End set_exit_flag, and waiting for thread to finish....");

    while(true){
        {
            std::lock_guard<std::mutex> lock_mutex(mutex_thread_ended);
            if(thread_ended){
                break;
            }
        }
        SPDLOG_INFO("Thread Not finished, sleep 500ms!");
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        continue;
    }
    delete postprocessor;
    postprocessor = NULL;
}

void YoloxPostProcessThread::setPadding()
{
    padding_flag = true;
}

sail::Tensor* YoloxPostProcessThread::get_data(std::vector<cv::Mat>& imgs, std::vector<int>& channel, std::vector<int>& index, std::vector<std::vector<int>>& padding_atrr)
{
    sail::Tensor* tensor = NULL;
    std::lock_guard<std::mutex> lock_mutex(mutex_data);
    int result_size = tensor_que.size();
    if (result_size <= 0){
        return NULL;
    }
    // printf("get_data-0: output_tensor: %p\n",tensor_que.front());

    tensor = tensor_que.front();
    imgs = std::move(img_que.front());
    channel = std::move(channel_que.front());
    index = std::move(idx_que.front());
    padding_atrr = std::move(padding_atrr_que.front());

    tensor_que.pop_front();
    img_que.pop_front();
    channel_que.pop_front();
    idx_que.pop_front();
    padding_atrr_que.pop_front();
    // printf("get_data-1: output_tensor: %p\n",tensor);
    return tensor;
}

int YoloxPostProcessThread::push_data(sail::Tensor* tensor, std::vector<cv::Mat>& imgs, std::vector<int>& channel, std::vector<int>& index, std::vector<std::vector<int>>& padding_atrr)
{
    {
        std::lock_guard<std::mutex> lock_mutex(mutex_data);
        if(padding_atrr_que.size() >= 5){
            return 1;
        }
        tensor_que.push_back(std::move(tensor));
        img_que.push_back(std::move(imgs));
        channel_que.push_back(std::move(channel));
        idx_que.push_back(std::move(index));
        padding_atrr_que.push_back(std::move(padding_atrr));
    }
    {
        std::unique_lock<std::mutex> lck(data_flag);
        data_flag_cond.notify_all();
    }
    return 0;
}

bool YoloxPostProcessThread::get_exit_flag()
{
    std::lock_guard<std::mutex> lock_mutex(mutex_exit);
    return flag_exit_;
}

void YoloxPostProcessThread::set_exit_flag(bool flag)
{
    std::lock_guard<std::mutex> lock_mutex(mutex_exit);
    flag_exit_ = flag;
}

void YoloxPostProcessThread::setDevice(int dev_id)
{
    device_id = dev_id;
}

void YoloxPostProcessThread::setOutShape(std::vector<int> out_shape)
{
    output_shape.clear();
    for(int i=0;i<out_shape.size(); ++i){
        output_shape.push_back(out_shape[i]);
    }
    batch_size = output_shape[0];
}

void YoloxPostProcessThread::start_post_process_thread()
{
    std::thread thread_post_process = std::thread(&YoloxPostProcessThread::processThread,this);
    thread_post_process.detach();
}

void YoloxPostProcessThread::processThread()
{
    sail::Handle handle(device_id);
    sail::Bmcv bmcv(handle);
    float threshold_dete = 0.5;
    float threshold_nms = 0.5;
    thread_ended = false;
    while (true){
        double time_start_post = get_time_us();
        if(get_exit_flag()){
            break;
        }

        std::vector<cv::Mat> imgs;
        std::vector<int> channel;
        std::vector<int> index;
        std::vector<std::vector<int>> padding_atrr;

        sail::Tensor* tensor = get_data(imgs, channel, index, padding_atrr);
        if( tensor == NULL){
            std::unique_lock<std::mutex> lck(data_flag);
            data_flag_cond.wait_for(lck,std::chrono::seconds(1));
            continue;
        }

        tensor->sync_d2s();
        float* output_data = (float*)tensor->sys_data();
        std::vector<std::vector<ObjRect>> detections;

        std::vector<std::pair<int,int>> ost_size;
        std::vector<float> resize_scale;

        for(int i=0;i<imgs.size();++i){
            float sacle_w = (float)padding_atrr[i][2]/imgs[i].cols;
            float sacle_h = (float)padding_atrr[i][3]/imgs[i].rows;
            float sacle_min = sacle_w < sacle_h ? sacle_w : sacle_h;
            std::pair<int,int> video_size = std::pair<int,int>(imgs[i].cols, imgs[i].rows);

            padding_atrr[i][0] = padding_atrr[i][0]/sacle_min;
            padding_atrr[i][1] = padding_atrr[i][1]/sacle_min;
            ost_size.push_back(video_size);
            resize_scale.push_back(sacle_min);
        }
    
        if (padding_flag){
            postprocessor->process(output_data, output_shape, resize_scale, threshold_dete, threshold_nms, detections);
        }else{
            postprocessor->process(output_data, output_shape, ost_size, threshold_dete,threshold_nms,detections);
        }

        // for (size_t j = 0; j < imgs.size(); ++j)           {
        //     printf("%d,%d\n",padding_atrr[j][0],padding_atrr[j][1]);
        //     for (size_t box_idx = 0; box_idx < detections[j].size(); box_idx++)
        //     {
        //         cv::rectangle(imgs[j],cv::Rect(int(detections[j][box_idx].left-padding_atrr[j][0]), 
        //                                     int(detections[j][box_idx].top-padding_atrr[j][1]),
        //                                     int(detections[j][box_idx].width),
        //                                     int(detections[j][box_idx].height)),
        //             cv::Scalar(0, 0, 255), 2);
        //     }
        //     char save_name[256]={0};
        //     sprintf(save_name,"%08d_%02d_ost.jpg",index[j],channel[j]);
        //     cv::imwrite(save_name,imgs[j]);
        // }
        
        // printf("########: PostProcee: output_tensor: %p\n",tensor);
        delete tensor;
        // printf("********: PostProcee: SYS_PTR: %p\n",output_data);
        double time_end_post = get_time_us();
        // printf("POST time: %.0f us.\n", time_end_post-time_start_post);
    }
    std::lock_guard<std::mutex> lock_mutex(mutex_thread_ended);
    thread_ended = true;
}

int main(){
    int tpu_id = 0;
    int max_count = 5000;
    std::string bmodel_name = "/data/models/yolox_s_int8_bs4.bmodel";
    sail_resize_type resize_type = BM_RESIZE_VPP_NEAREST;
    // sail_resize_type resize_type = BM_RESIZE_TPU_NEAREST;
    // sail_resize_type resize_type = BM_RESIZE_TPU_LINEAR;
    // sail_resize_type resize_type = BM_RESIZE_TPU_BICUBIC;
    // sail_resize_type resize_type = BM_PADDING_VPP_NEAREST;
    // sail_resize_type resize_type = BM_PADDING_TPU_NEAREST;
    // sail_resize_type resize_type = BM_PADDING_TPU_LINEAR;
    // sail_resize_type resize_type = BM_PADDING_TPU_BICUBIC;
    bool padding_flag = true;
    if(resize_type == BM_RESIZE_VPP_NEAREST || resize_type == BM_RESIZE_TPU_NEAREST || resize_type == BM_RESIZE_TPU_NEAREST || resize_type == BM_RESIZE_TPU_BICUBIC){
        padding_flag = false;
    }
    std::vector<std::string> video_list;
    int queue_in_size = 20;
    int queue_out_size = 10;

    video_list.push_back("/data/video/001.mp4");
    video_list.push_back("/data/video/002.mp4");
    video_list.push_back("/data/video/003.mp4");
    video_list.push_back("/data/video/004.mp4");

    sail::Handle handle(tpu_id);
    sail::Bmcv bmcv(handle);
    MultiProcessor process(tpu_id,video_list,bmodel_name,resize_type,queue_in_size,queue_out_size);
    std::vector<int> strides;
    strides.push_back(8);
    strides.push_back(16);
    strides.push_back(32);

    YoloxPostProcessThread postprocessor(process.get_input_width(), process.get_input_height(), strides);

    postprocessor.setDevice(tpu_id);

    postprocessor.setOutShape(process.get_output_shape());

    postprocessor.start_post_process_thread();

    std::cout<<"::::::main thread: "<<getpid()<<","<<gettid()<<endl;
    for(int i = 0; i < max_count; ++i){

        double time_start = get_time_us();
        std::tuple<std::map<std::string,sail::Tensor*>,std::vector<cv::Mat>,std::vector<int>,std::vector<int>,std::vector<std::vector<int>>> result = process.GetBatchData();
        double time_end = get_time_us();
        
        auto result_map = std::move(std::get<0>(result)) ;
        auto iter = result_map.begin();
        if ( iter == result_map.end())
        {
            printf("Out put tensor map is empty!\n");
            return 1;
        }
        
        sail::Tensor* output_tensor = iter->second;
        std::vector<cv::Mat> imgs = std::move(std::get<1>(result));
        std::vector<int> channel = std::move(std::get<2>(result));
        std::vector<int> index = std::move(std::get<3>(result));
        std::vector<std::vector<int>> padding_atrr = std::move(std::get<4>(result));
        while(true){
            int ret = postprocessor.push_data(output_tensor,imgs,channel,index,padding_atrr);
            if(ret == 0){
                break;
            }else{
                printf("buffer full, sleep 20ms!\n");
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
        }
        printf("### Get data time: %.0f us, [%d]\n", time_end-time_start,i);
    }
    return 0;
}