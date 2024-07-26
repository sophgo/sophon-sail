#define USE_OPENCV 1
#define USE_FFMPEG 1
#define USE_BMCV 1

#include <sail/cvwrapper.h>
#include <sail/decoder_multi.h>
#include <sail/engine.h>
#include <sail/tensor.h>
#include <iostream>
#include <stdio.h>
#include <mutex>
#include <thread>

using namespace std;
using namespace sail;


class MultiProcessor{
public:
    MultiProcessor(int tpu_id, std::vector<std::string> video_list, std::string bmodel_name, sail_resize_type resize_type, int queue_in_size, int queue_out_size);
    ~MultiProcessor();

    std::tuple<std::map<std::string,sail::Tensor*>,std::vector<cv::Mat>,std::vector<int>,std::vector<int>,std::vector<std::vector<int>>> GetBatchData();

    int get_input_width();

    int get_input_height();

    vector<int> get_output_shape();
private:
    std::vector<std::string> video_list_;
    int tpu_id_;
    sail::MultiDecoder *multi_decoder_ = NULL;
    EngineImagePreProcess *engine_image_pre_process_ = NULL;
    std::map<int, std::string> video_list_map;

    std::string bmodel_name_;

    std::tuple<std::pair<float, float>, std::pair<float, float>,std::pair<float, float>> alpha_beta;

    std::mutex mutex_exit;
    bool flag_exit_;

    std::mutex mutex_thread_ended;
    bool thread_ended;
private:
    void InitEngineImagePreProcess(int tpu_id, std::string bmodel_name, sail_resize_type resize_type, int queue_in_size, int queue_out_size);

    void decoder_and_preprocess();

    bool get_exit_flag();

    void set_exit_flag(bool);

    void start_decoder_thread();
};

struct ObjRect
{
    unsigned int class_id;
    float score;
    float left;
    float top;
    float right;
    float bottom;
    float width;
    float height;
};


class YoloX_PostForward
{
private:
  /* data */
public:
  YoloX_PostForward(int net_w, int net_h, std::vector<int> strides);
  ~YoloX_PostForward();
  void process(float* data_ptr,std::vector<int> output_shape, std::vector<std::pair<int,int>> ost_size, 
    float threshold, float nms_threshold, std::vector<std::vector<ObjRect>> &detections);

  void process(float* data_ptr,std::vector<int> output_shape, std::vector<float> resize_scale, 
    float threshold, float nms_threshold, std::vector<std::vector<ObjRect>> &detections);

private:
  int outlen_diml ;
  int* grids_x_;
  int* grids_y_;
  int* expanded_strides_;
  int network_width;
  int network_height;
};

float box_iou_FM(ObjRect a, ObjRect b);


class YoloxPostProcessThread
{
  public:
    YoloxPostProcessThread(int net_w, int net_h, std::vector<int> strides);
    ~YoloxPostProcessThread();

    void setDevice(int dev_id);

    void setPadding();

    void setOutShape(std::vector<int> out_shape);

    void start_post_process_thread();

    int push_data(sail::Tensor* tensor, std::vector<cv::Mat>& imgs, std::vector<int>& channel, std::vector<int>& index, std::vector<std::vector<int>>& padding_atrr);
  private:

    sail::Tensor* get_data(std::vector<cv::Mat>& imgs, std::vector<int>& channel, std::vector<int>& index, std::vector<std::vector<int>>& padding_atrr);


    bool get_exit_flag();

    void set_exit_flag(bool);

    void processThread();


    YoloX_PostForward *postprocessor;

    int batch_size;
    int device_id;

    bool padding_flag;

    std::deque<sail::Tensor*> tensor_que;
    std::deque<std::vector<cv::Mat>> img_que;
    std::deque<std::vector<int>> channel_que;
    std::deque<std::vector<int>> idx_que;
    std::deque<std::vector<std::vector<int>>> padding_atrr_que;

    std::mutex mutex_data;

    std::mutex mutex_exit;
    bool flag_exit_;

    std::mutex mutex_thread_ended;
    bool thread_ended;

    std::condition_variable data_flag_cond;
    std::mutex data_flag;

    std::vector<int> output_shape;
};
