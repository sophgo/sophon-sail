#include <tpu_kernel_api.h>
#include <spdlog/spdlog.h>
#include <graph.h>
#include <tensor.h>
#include <algokit.h>

#ifndef TPUKERNRL_OFF

#ifdef PYTHON
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eval.h>
using namespace pybind11::literals;
#endif

using namespace std;

namespace sail {
    
#define SAIL_TK_MAX_YOLO_INPUT_NUM 8
#define SAIL_TK_MAX_YOLO_ANCHOR_NUM 8

typedef struct {
  unsigned long long bottom_addr[SAIL_TK_MAX_YOLO_INPUT_NUM];
  unsigned long long top_addr;
  unsigned long long detected_num_addr;
  int input_num;
  int batch_num;
  int hw_shape[SAIL_TK_MAX_YOLO_INPUT_NUM][2];
  int num_classes;
  int num_boxes;
  int keep_top_k;
  float nms_threshold;
  float confidence_threshold;
  float bias[SAIL_TK_MAX_YOLO_INPUT_NUM * SAIL_TK_MAX_YOLO_ANCHOR_NUM * 2];
  float anchor_scale[SAIL_TK_MAX_YOLO_INPUT_NUM];
  int clip_box;
} tpu_kernel_api_yolov5NMS_t;

typedef struct {
  unsigned long long bottom_addr;
  unsigned long long top_addr;
  unsigned long long detected_num_addr;
  int input_shape[3];
  int keep_top_k;
  float nms_threshold;
  float confidence_threshold;
  int agnostic_nms;
  int max_hw;
} tpu_kernel_api_yolov5NMS_v2_t;

typedef struct {
  unsigned long long input_data_addr;
  unsigned long long num_output_data_addr;
  unsigned long long output_data_addr;
  int input_c;
  int input_h;
  int input_w;
  int max_peak_num;
  float nms_thresh;
} tpu_kernel_api_openpose_part_nms_postprocess_t;

// yolov5 post 3output
class tpu_kernel_api_yolov5_detect_out::tpu_kernel_api_yolov5_detect_out_cc{
public:
    explicit tpu_kernel_api_yolov5_detect_out_cc(int device_id, 
                                            const std::vector<std::vector<int>>& shapes, 
                                            int network_w, 
                                            int network_h,
                                            std::string module_file);

    int reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new);

    int process(std::vector<unsigned long long> in_data_ptr, 
                std::vector<DeteObjRect> &out_doxs,
                float dete_threshold,
                float nms_threshold);

    int process(std::vector<TensorPTRWithName>& input, 
                std::vector<std::vector<DeteObjRect>> &out_doxs,
                float dete_threshold,
                float nms_threshold,
                bool release_input);

    ~tpu_kernel_api_yolov5_detect_out_cc();

private:
    int device_id_;
    int net_w;
    int net_h;
    int class_num;
    int batch_size;
    int max_dete_count;
    tpu_kernel_function_t func_id;
    tpu_kernel_api_yolov5NMS_t api_params;
    sail::Handle *handle_;
    sail::Tensor *output_tensor;
    sail::Tensor *output_num;
    std::vector<std::vector<int>> input_shapes;
};

tpu_kernel_api_yolov5_detect_out::tpu_kernel_api_yolov5_detect_out_cc::tpu_kernel_api_yolov5_detect_out_cc(int device_id, 
                                                                                                        const std::vector<std::vector<int>>& shapes, 
                                                                                                        int network_w, 
                                                                                                        int network_h,
                                                                                                        std::string module_file)
:device_id_(device_id), handle_(NULL), max_dete_count(0),batch_size(1),net_h(network_h),net_w(network_w),output_tensor(NULL),output_num(NULL)                                                                                     
{
    if (shapes.size() <= 0){
        SPDLOG_ERROR("ERROR Shapes size: {}!",shapes.size());
        exit(1);
    }
    memset(&api_params,0,sizeof(tpu_kernel_api_yolov5NMS_t));

    api_params.num_boxes = 3;
    api_params.bias[0] = 10;
    api_params.bias[1] = 13;

    api_params.bias[2] = 16;
    api_params.bias[3] = 30;

    api_params.bias[4] = 33;
    api_params.bias[5] = 23;

    api_params.bias[6] = 30;
    api_params.bias[7] = 61;

    api_params.bias[8] = 62;
    api_params.bias[9] = 45;

    api_params.bias[10] = 59;
    api_params.bias[11] = 119;

    api_params.bias[12] = 116;
    api_params.bias[13] = 90;

    api_params.bias[14] = 156;
    api_params.bias[15] = 198;

    api_params.bias[16] = 373;
    api_params.bias[17] = 326;

    
    int input_num = shapes.size();
    class_num = shapes[0][1]/3-5;
    input_shapes.clear();
    for(int i=0;i<input_num;++i){
        if(shapes[i].size() != 4){
            SPDLOG_ERROR("ERROR DIMS 4 vs. {}!",shapes[i].size());
            exit(1);
        }
        if(shapes[i][1]/3-5 != class_num){
            SPDLOG_ERROR("ERROR Shapes!");
            exit(1);
        }
        max_dete_count += 3*shapes[i][2]*shapes[i][3];
        std::vector<int> temp_shape;
        for(int j=0;j<shapes[i].size();++j){
            temp_shape.push_back(shapes[i][j]);
        }
        input_shapes.push_back(temp_shape);
    }
    handle_ = new Handle(device_id_);
    if(!handle_){
        SPDLOG_ERROR("Create Handle failed, using device {}!",device_id_);
        exit(1);
    }
    tpu_kernel_module_t tpu_module = tpu_kernel_load_module_file(handle_->data(), module_file.c_str()); 
    func_id = tpu_kernel_get_function(handle_->data(), tpu_module, "tpu_kernel_api_yolov5_detect_out");
    
    output_tensor = new Tensor(*handle_,{max_dete_count,7},BM_FLOAT32,true,true);
    output_num = new Tensor(*handle_,{1,1},BM_INT32,true,true);

    api_params.top_addr = bm_mem_get_device_addr(output_tensor->dev_data());
    api_params.detected_num_addr = bm_mem_get_device_addr(output_num->dev_data());

    api_params.input_num = input_num;
    api_params.batch_num = batch_size;
    for (int j = 0; j < input_num; ++j) {
        api_params.hw_shape[j][0] = shapes[j][2];
        api_params.hw_shape[j][1] = shapes[j][3];
        api_params.anchor_scale[j] = net_h / shapes[j][2];
    }
    api_params.num_classes = class_num;
    api_params.num_boxes = 3;
    api_params.keep_top_k = 200;
    api_params.clip_box = 1;
}

int tpu_kernel_api_yolov5_detect_out::tpu_kernel_api_yolov5_detect_out_cc::reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new)
{
    if(anchors_new.size() != api_params.input_num){
        SPDLOG_ERROR("Error anchor size!");
        return SAIL_TPU_KERNEL_ERROR_PARAM_MISMATCH;
    }
    api_params.num_boxes = anchors_new[0].size();
    for(int i=0;i<anchors_new.size();++i){
        for(int j=0;j<anchors_new[i].size();++j){
            api_params.bias[2*(i*anchors_new[j].size()+j)] = anchors_new[i][j][0];
            api_params.bias[2*(i*anchors_new[j].size()+j)+1] = anchors_new[i][j][1];
        }
    }
    return SAIL_TPU_KERNEL_SUCCESS;
}

tpu_kernel_api_yolov5_detect_out::tpu_kernel_api_yolov5_detect_out_cc::~tpu_kernel_api_yolov5_detect_out_cc(){
    if(output_tensor){
        delete output_tensor;
        output_tensor = NULL;
    }
    if(output_num){
        delete output_num;
        output_num = NULL;
    }
}

int tpu_kernel_api_yolov5_detect_out::tpu_kernel_api_yolov5_detect_out_cc::process(std::vector<unsigned long long> in_data_ptr, 
                                                                                    std::vector<DeteObjRect> &out_doxs,
                                                                                    float dete_threshold,
                                                                                    float nms_threshold)
{
    if(api_params.input_num != in_data_ptr.size()){
        SPDLOG_ERROR("Input Tensor count Mismatch!");
        return SAIL_TPU_KERNEL_ERROR_PARAM_MISMATCH;
    }
    for(int i=0;i<in_data_ptr.size();++i){
        api_params.bottom_addr[i] = in_data_ptr[i];
    }
    api_params.nms_threshold = nms_threshold;
    api_params.confidence_threshold = dete_threshold;
    tpu_kernel_launch(handle_->data(), func_id, &api_params, sizeof(api_params));
    bm_thread_sync(handle_->data());
    output_num->sync_d2s();
    int* box_count = (int*) output_num->sys_data();
    if(box_count[0] > 0) {
        output_tensor->sync_d2s(box_count[0]*7*sizeof(float));
    }
    out_doxs.clear();
    float* output_sys = (float*) output_tensor->sys_data();

    for(int bid=0;bid<box_count[0];++bid){
        DeteObjRect temp_bbox;
        temp_bbox.class_id = *(output_sys + 7 * bid + 1);
        if (temp_bbox.class_id == -1) {
            continue;
        }
        temp_bbox.score = *(output_sys + 7 * bid + 2);
        float centerX = *(output_sys + 7 * bid + 3);
        float centerY = *(output_sys + 7 * bid + 4);
        temp_bbox.width = *(output_sys + 7 * bid + 5);
        temp_bbox.height = *(output_sys + 7 * bid + 6);

        temp_bbox.left = centerX - temp_bbox.width/2;
        temp_bbox.top = centerY - temp_bbox.height/2;
        temp_bbox.right = centerX + temp_bbox.width/2;
        temp_bbox.bottom = centerY + temp_bbox.height/2;
        out_doxs.push_back(temp_bbox);
    }
    return SAIL_TPU_KERNEL_SUCCESS;
}

int tpu_kernel_api_yolov5_detect_out::tpu_kernel_api_yolov5_detect_out_cc::process(std::vector<TensorPTRWithName>& input_data, 
                                                                                std::vector<std::vector<DeteObjRect>> &out_doxs,
                                                                                float dete_threshold,
                                                                                float nms_threshold,
                                                                                bool release_input){
    if(input_data.size() != api_params.input_num){
        SPDLOG_ERROR("Input Tensor count Mismatch!");
        return SAIL_TPU_KERNEL_ERROR_PARAM_MISMATCH;
    }
    std::vector<sail::Tensor*> input_list;
    std::vector<int> each_batch_len;
    for(int i=0;i<api_params.input_num;++i){
        input_list.push_back(NULL);
        each_batch_len.push_back(0);
    }
    int batch_size = input_data[0].data->shape()[0];
    out_doxs.clear();
    //重排序
    for(int i=0; i<input_data.size();++i){
        sail::Tensor* data = input_data[i].data;
        const std::vector<int>& input_shape = data->shape();
        if(input_shape.size() != 4){
            SPDLOG_ERROR("Input Tensor shape Mismatch!");
            return SAIL_TPU_KERNEL_ERROR_SHAPE;
        }
        if(batch_size != input_shape[0]){
            SPDLOG_ERROR("Input Batch Size Mismatch!");
            return SAIL_TPU_KERNEL_ERROR_BATCH_SIZE;
        }
        bool has_match = false;
        for(int j=0;j<input_shapes.size();++j){
            if(input_shape[1] == input_shapes[j][1] && 
               input_shape[2] == input_shapes[j][2] && 
               input_shape[3] == input_shapes[j][3] ){
                if(input_list[j] != NULL){
                    SPDLOG_ERROR("Input Tensor shape Mismatch!");
                    return SAIL_TPU_KERNEL_ERROR_SHAPE;
                }
                input_list[j] = data;
                each_batch_len[j] = data->dev_data().size/batch_size;
                has_match = true;
            }
        }
        if(!has_match){         //没有匹配到shape
            SPDLOG_ERROR("Input Tensor shape Mismatch!");
            return SAIL_TPU_KERNEL_ERROR_SHAPE;
        }
    }
    for(int i=0;i<batch_size;++i){
        std::vector<unsigned long long> in_data_ptr;
        std::vector<DeteObjRect> out_detes;
        for(int j=0;j<input_list.size();++j){
            in_data_ptr.push_back(bm_mem_get_device_addr(input_list[j]->dev_data()) + i*each_batch_len[j]);
        }
        int ret = process(in_data_ptr, out_detes, dete_threshold, nms_threshold);
        if(ret != SAIL_TPU_KERNEL_SUCCESS){
            return ret;
        }
        out_doxs.push_back(out_detes);
    }

    if (release_input) {
        for (int tidx = 0; tidx < input_data.size(); tidx++){
            delete input_data[tidx].data;
        }
    }

    return SAIL_TPU_KERNEL_SUCCESS;
}


tpu_kernel_api_yolov5_detect_out::tpu_kernel_api_yolov5_detect_out(int device_id, 
                                            const std::vector<std::vector<int>>& shapes, 
                                            int network_w, 
                                            int network_h,
                                            std::string module_file)
    :_impl (new tpu_kernel_api_yolov5_detect_out_cc(device_id,shapes,network_w,network_h,module_file))
{}

tpu_kernel_api_yolov5_detect_out::~tpu_kernel_api_yolov5_detect_out()
{
    delete _impl;
}


int tpu_kernel_api_yolov5_detect_out::process(std::vector<TensorPTRWithName>& input, 
                                            std::vector<std::vector<DeteObjRect>> &out_doxs,
                                            float dete_threshold,
                                            float nms_threshold,
                                            bool release_input){
    return _impl->process(input,out_doxs,dete_threshold,nms_threshold, release_input);
}


std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> 
    tpu_kernel_api_yolov5_detect_out::process(std::vector<TensorPTRWithName>& input, float dete_threshold, float nms_threshold, bool release_input)
{
    std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> results;
    results.clear();
    std::vector<std::vector<DeteObjRect>> out_doxs;
    int ret = process(input,out_doxs,dete_threshold,nms_threshold,release_input);
    if(ret != SAIL_TPU_KERNEL_SUCCESS){
        return results;
    }
    for(int i=0;i<out_doxs.size();++i){
        std::vector<std::tuple<int, int, int, int ,int, float>> objs;
        for(int j=0;j<out_doxs[i].size();++j){
            int left_temp = out_doxs[i][j].left+0.5;
            int top_temp = out_doxs[i][j].top+0.5;
            int right_temp = out_doxs[i][j].right+0.5;
            int bottom_temp = out_doxs[i][j].bottom+0.5;
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
    tpu_kernel_api_yolov5_detect_out::process(std::map<std::string, Tensor&>& input, float dete_threshold, float nms_threshold, bool release_input)
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
    return process(input_data,dete_threshold,nms_threshold, release_input);
}

int tpu_kernel_api_yolov5_detect_out::reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new)
{
    return _impl->reset_anchors(anchors_new);
}

// yolov5 post 1output
class tpu_kernel_api_yolov5_out_without_decode::tpu_kernel_api_yolov5_out_without_decode_cc{
public:
    explicit tpu_kernel_api_yolov5_out_without_decode_cc(int device_id, 
                                            const std::vector<int>& shapes, 
                                            int network_w, 
                                            int network_h,
                                            std::string module_file);

    int process(unsigned long long in_data_ptr, 
                std::vector<DeteObjRect> &out_doxs,
                float dete_threshold,
                float nms_threshold);

    int process(TensorPTRWithName& input, 
                std::vector<std::vector<DeteObjRect>> &out_doxs,
                float dete_threshold,
                float nms_threshold);

    ~tpu_kernel_api_yolov5_out_without_decode_cc();

private:
    int device_id_;
    int net_w;
    int net_h;
    int class_num;
    int batch_size;
    int max_dete_count;
    tpu_kernel_function_t func_id;
    tpu_kernel_api_yolov5NMS_v2_t api_params;
    sail::Handle *handle_;
    sail::Tensor *output_tensor;
    sail::Tensor *output_num;
    std::vector<int> input_shapes;
};

tpu_kernel_api_yolov5_out_without_decode::tpu_kernel_api_yolov5_out_without_decode_cc::tpu_kernel_api_yolov5_out_without_decode_cc(int device_id, 
                                                                                                        const std::vector<int>& shapes, 
                                                                                                        int network_w, 
                                                                                                        int network_h,
                                                                                                        std::string module_file)
:device_id_(device_id), handle_(NULL), max_dete_count(0),net_h(network_h),net_w(network_w),output_tensor(NULL),output_num(NULL) {
    // shapes: [batch_size, max_dete_count, (class_num + 5) * 3]
    if (shapes.size() != 3){
        SPDLOG_ERROR("ERROR Shapes size: {}!",shapes.size());
        exit(1);
    }
    memset(&api_params, 0, sizeof(tpu_kernel_api_yolov5NMS_v2_t));
    
    batch_size = shapes[0];
    class_num = shapes[2]/3-5;
    input_shapes = shapes;
    handle_ = new Handle(device_id_);
    if(!handle_){
        SPDLOG_ERROR("Create Handle failed, using device {}!",device_id_);
        exit(1);
    }
    tpu_kernel_module_t tpu_module = tpu_kernel_load_module_file(handle_->data(), module_file.c_str()); 
    func_id = tpu_kernel_get_function(handle_->data(), tpu_module, "tpu_kernel_api_yolov5_out_without_decode");
    
    output_tensor = new Tensor(*handle_,{shapes[1],7},BM_FLOAT32,true,true);
    output_num = new Tensor(*handle_,{1,1},BM_INT32,true,true);

    api_params.top_addr = bm_mem_get_device_addr(output_tensor->dev_data());
    api_params.detected_num_addr = bm_mem_get_device_addr(output_num->dev_data());

    api_params.keep_top_k = 200;
    api_params.input_shape[0] = shapes[0]; //only support batchsize=1
    api_params.input_shape[1] = shapes[1];
    api_params.input_shape[2] = shapes[2];
    api_params.agnostic_nms = 0;
    api_params.max_hw = std::max(network_h, network_w);
}

tpu_kernel_api_yolov5_out_without_decode::tpu_kernel_api_yolov5_out_without_decode_cc::~tpu_kernel_api_yolov5_out_without_decode_cc(){
    if(output_tensor){
        delete output_tensor;
        output_tensor = NULL;
    }
    if(output_num){
        delete output_num;
        output_num = NULL;
    }
}

int tpu_kernel_api_yolov5_out_without_decode::tpu_kernel_api_yolov5_out_without_decode_cc::process(unsigned long long in_data_ptr, 
                                                                                    std::vector<DeteObjRect> &out_doxs,
                                                                                    float dete_threshold,
                                                                                    float nms_threshold) {
    api_params.bottom_addr = in_data_ptr;

    api_params.nms_threshold = nms_threshold;
    api_params.confidence_threshold = dete_threshold;
    tpu_kernel_launch(handle_->data(), func_id, &api_params, sizeof(api_params));
    bm_thread_sync(handle_->data());
    output_num->sync_d2s();
    int* box_count = (int*) output_num->sys_data();
    if(box_count[0] > 0) {
        output_tensor->sync_d2s(box_count[0]*7*sizeof(float));
    }
    out_doxs.clear();
    float* output_sys = (float*) output_tensor->sys_data();

    for(int bid=0;bid<box_count[0];++bid){
        DeteObjRect temp_bbox;
        temp_bbox.class_id = *(output_sys + 7 * bid + 1);
        if (temp_bbox.class_id == -1) {
            continue;
        }
        temp_bbox.score = *(output_sys + 7 * bid + 2);
        float centerX = *(output_sys + 7 * bid + 3);
        float centerY = *(output_sys + 7 * bid + 4);
        temp_bbox.width = *(output_sys + 7 * bid + 5);
        temp_bbox.height = *(output_sys + 7 * bid + 6);

        temp_bbox.left = centerX - temp_bbox.width/2;
        temp_bbox.top = centerY - temp_bbox.height/2;
        temp_bbox.right = centerX + temp_bbox.width/2;
        temp_bbox.bottom = centerY + temp_bbox.height/2;
        out_doxs.push_back(temp_bbox);
    }
    return SAIL_TPU_KERNEL_SUCCESS;
}

int tpu_kernel_api_yolov5_out_without_decode::tpu_kernel_api_yolov5_out_without_decode_cc::process(TensorPTRWithName& input_data, 
                                                                                std::vector<std::vector<DeteObjRect>> &out_doxs,
                                                                                float dete_threshold,
                                                                                float nms_threshold) {
    sail::Tensor* data = input_data.data;
    int each_batch_len = data->dev_data().size / batch_size;

    out_doxs.clear();
    for(int i=0;i<batch_size;++i){
        unsigned long long in_data_ptr = bm_mem_get_device_addr(data->dev_data()) + i * each_batch_len;
        std::vector<DeteObjRect> out_detes;
        int ret = process(in_data_ptr, out_detes, dete_threshold, nms_threshold);
        if(ret != SAIL_TPU_KERNEL_SUCCESS){
            return ret;
        }
        out_doxs.push_back(out_detes);
    }
    return SAIL_TPU_KERNEL_SUCCESS;
}


tpu_kernel_api_yolov5_out_without_decode::tpu_kernel_api_yolov5_out_without_decode(int device_id, 
                                            const std::vector<int>& shapes, 
                                            int network_w, 
                                            int network_h,
                                            std::string module_file)
    :_impl (new tpu_kernel_api_yolov5_out_without_decode_cc(device_id, shapes, network_w, network_h, module_file))
{}

tpu_kernel_api_yolov5_out_without_decode::~tpu_kernel_api_yolov5_out_without_decode()
{
    delete _impl;
}


int tpu_kernel_api_yolov5_out_without_decode::process(TensorPTRWithName& input, 
                                            std::vector<std::vector<DeteObjRect>> &out_doxs,
                                            float dete_threshold,
                                            float nms_threshold) {
    return _impl->process(input, out_doxs, dete_threshold, nms_threshold);
}


std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> 
    tpu_kernel_api_yolov5_out_without_decode::process(TensorPTRWithName& input, float dete_threshold, float nms_threshold) {
    std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> results;
    results.clear();
    std::vector<std::vector<DeteObjRect>> out_doxs;
    int ret = process(input, out_doxs, dete_threshold, nms_threshold);
    if(ret != SAIL_TPU_KERNEL_SUCCESS){
        return results;
    }
    for(int i=0;i<out_doxs.size();++i){
        std::vector<std::tuple<int, int, int, int ,int, float>> objs;
        for(int j=0;j<out_doxs[i].size();++j){
            int left_temp = out_doxs[i][j].left+0.5;
            int top_temp = out_doxs[i][j].top+0.5;
            int right_temp = out_doxs[i][j].right+0.5;
            int bottom_temp = out_doxs[i][j].bottom+0.5;
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
    tpu_kernel_api_yolov5_out_without_decode::process(std::map<std::string, Tensor&>& input, float dete_threshold, float nms_threshold) {
    TensorPTRWithName in_data_ptr;
    in_data_ptr.data = &input.begin()->second;
    in_data_ptr.name = input.begin()->first;
    return process(in_data_ptr, dete_threshold, nms_threshold);
}

// openpose part nms
class tpu_kernel_api_openpose_part_nms::tpu_kernel_api_openpose_part_nms_cc{
public:
    explicit tpu_kernel_api_openpose_part_nms_cc(int device_id, 
                                            int network_c, 
                                            std::string module_file);

    int reset_network_c(int network_c_new);

    int process(unsigned long long in_data_ptr, 
                std::vector<int>& shape,
                std::vector<int>& num_result, 
                std::vector<float>& score_out_result,
                std::vector<int>& coor_out_result,
                float threshold,
                int max_peak_num);

    int process(TensorPTRWithName& input, 
                std::vector<int>& shape,
                std::vector<std::vector<int>>& num_result, 
                std::vector<std::vector<float>>& score_out_result,
                std::vector<std::vector<int>>& coor_out_result,
                std::vector<float>& threshold,
                std::vector<int>& max_peak_num);

    ~tpu_kernel_api_openpose_part_nms_cc();

private:
    int device_id_;
    int channel_num;
    tpu_kernel_function_t func_id;
    tpu_kernel_api_openpose_part_nms_postprocess_t api_params;
    sail::Handle *handle_;
    sail::Tensor *output_tensor;
    sail::Tensor *output_num;
};

tpu_kernel_api_openpose_part_nms::tpu_kernel_api_openpose_part_nms_cc::tpu_kernel_api_openpose_part_nms_cc(int device_id, 
                                                                                                        int network_c,
                                                                                                        std::string module_file)
:device_id_(device_id), channel_num(network_c), handle_(NULL), output_tensor(NULL), output_num(NULL)                                                                                     
{
    if (network_c <= 0){
        SPDLOG_ERROR("ERROR channel size: {}!", network_c);
        exit(1);
    }
    memset(&api_params, 0, sizeof(tpu_kernel_api_openpose_part_nms_postprocess_t));
    api_params.input_c = network_c;

    handle_ = new Handle(device_id_);
    if(!handle_){
        SPDLOG_ERROR("Create Handle failed, using device {}!", device_id_);
        exit(1);
    }
    tpu_kernel_module_t tpu_module = tpu_kernel_load_module_file(handle_->data(), module_file.c_str()); 
    func_id = tpu_kernel_get_function(handle_->data(), tpu_module, "tpu_kernel_api_openpose_part_nms_postprocess");

    output_num = new Tensor(*handle_, {network_c}, BM_INT32, true, true);
    api_params.num_output_data_addr = bm_mem_get_device_addr(output_num->dev_data());
}

int tpu_kernel_api_openpose_part_nms::tpu_kernel_api_openpose_part_nms_cc::reset_network_c(int network_c_new)
{
    if(network_c_new <= 0){
        SPDLOG_ERROR("ERROR channel size: {}!", network_c_new);
        return SAIL_TPU_KERNEL_ERROR_PARAM_MISMATCH;
    }
    channel_num = network_c_new;
    if (output_num)
        delete output_num;
    output_num = new Tensor(*handle_, {channel_num}, BM_INT32, true, true);
    api_params.num_output_data_addr = bm_mem_get_device_addr(output_num->dev_data());
    api_params.input_c = channel_num;

    return SAIL_TPU_KERNEL_SUCCESS;
}

tpu_kernel_api_openpose_part_nms::tpu_kernel_api_openpose_part_nms_cc::~tpu_kernel_api_openpose_part_nms_cc(){
    if(output_tensor){
        delete output_tensor;
        output_tensor = NULL;
    }
    if(output_num){
        delete output_num;
        output_num = NULL;
    }
    if (handle_) {
        delete handle_;
        handle_ = NULL;
    }
}

int tpu_kernel_api_openpose_part_nms::tpu_kernel_api_openpose_part_nms_cc::process(unsigned long long in_data_ptr, 
                                                                                    std::vector<int>& shape,
                                                                                    std::vector<int>& num_result, 
                                                                                    std::vector<float>& score_out_result,
                                                                                    std::vector<int>& coor_out_result,
                                                                                    float threshold,
                                                                                    int max_peak_num)
{
    if(shape.size() != 2){
        SPDLOG_ERROR("ERROR shape size: {}, Only support dim2[width, height]!", shape.size());
        return SAIL_TPU_KERNEL_ERROR_PARAM_MISMATCH;
    }

    if (output_tensor)
        delete output_tensor;
    output_tensor = new Tensor(*handle_, {channel_num, shape[1], shape[0]}, BM_FLOAT32, true, true);
    api_params.output_data_addr = bm_mem_get_device_addr(output_tensor->dev_data());

    api_params.input_data_addr = in_data_ptr;
    api_params.input_h = shape[1];
    api_params.input_w = shape[0];
    api_params.max_peak_num = max_peak_num;
    api_params.nms_thresh = threshold;

    assert(BM_SUCCESS == tpu_kernel_launch(handle_->data(), func_id, &api_params, sizeof(api_params)));
    bm_thread_sync(handle_->data());
    
    output_num->sync_d2s();
    int* peak_count = (int*)output_num->sys_data();
    int peak_num = peak_count[channel_num - 1];

    if (num_result.size() != channel_num) num_result.resize(channel_num);
    if (score_out_result.size() != peak_num) score_out_result.resize(peak_num);
    if (coor_out_result.size() != peak_num) coor_out_result.resize(peak_num);
    if (peak_num == 0)
    {
        memcpy((void*)num_result.data(), (void*)peak_count, sizeof(int) * channel_num);
        return SAIL_TPU_KERNEL_SUCCESS;
    }

    bm_device_mem_t score_data_mem;
    bm_set_device_mem(&score_data_mem, peak_num * sizeof(float), in_data_ptr);
    bm_device_mem_t coor_data_mem;
    bm_set_device_mem(&coor_data_mem, peak_num * sizeof(int), in_data_ptr + peak_num * sizeof(float));
    std::vector<int> output_shape = {peak_num, 1};
    output_tensor->reset(output_shape, BM_FLOAT32);
    output_tensor->reset_dev_data(score_data_mem);
    float* score_output_data = (float*)output_tensor->sys_data();
    output_tensor->reset(output_shape, BM_UINT32);
    output_tensor->reset_dev_data(coor_data_mem);
    int* coor_output_data = (int*)output_tensor->sys_data();
    
    memcpy((void*)score_out_result.data(), (void*)score_output_data, sizeof(float) * peak_num);
    memcpy((void*)coor_out_result.data(), (void*)coor_output_data, sizeof(int) * peak_num);
    memcpy((void*)num_result.data(), (void*)peak_count, sizeof(int) * channel_num);
    return SAIL_TPU_KERNEL_SUCCESS;
}

int tpu_kernel_api_openpose_part_nms::tpu_kernel_api_openpose_part_nms_cc::process(TensorPTRWithName& input_data, 
                                                                                std::vector<int>& shape,
                                                                                std::vector<std::vector<int>>& num_result, 
                                                                                std::vector<std::vector<float>>& score_out_result,
                                                                                std::vector<std::vector<int>>& coor_out_result,
                                                                                std::vector<float>& threshold,
                                                                                std::vector<int>& max_peak_num){
    if(shape.size() != 2){
        SPDLOG_ERROR("ERROR shape size: {}, Only support dim2[width, height]!", shape.size());
        return SAIL_TPU_KERNEL_ERROR_PARAM_MISMATCH;
    }

    sail::Tensor* input = input_data.data;
    const std::vector<int>& input_shape = input->shape();
    if(input_shape.size() != 4){
        SPDLOG_ERROR("Input Tensor shape Mismatch!");
        return SAIL_TPU_KERNEL_ERROR_SHAPE;
    }
    int batch_size = input_data.data->shape()[0];
    if (threshold.size() != batch_size || max_peak_num.size() != batch_size) {
        SPDLOG_ERROR("Input threshold or max_peak_num shape Mismatch!");
        return SAIL_TPU_KERNEL_ERROR_PARAM_MISMATCH;
    }
    int each_batch_len = input->dev_data().size / batch_size;
    if (num_result.size() != batch_size) num_result.resize(batch_size);
    if (score_out_result.size() != batch_size) score_out_result.resize(batch_size);
    if (coor_out_result.size() != batch_size) coor_out_result.resize(batch_size);


    for(int i = 0; i < batch_size; ++i) {
        unsigned long long in_data_ptr = bm_mem_get_device_addr(input->dev_data()) + i * each_batch_len;
        int ret = process(in_data_ptr, shape, num_result[i], score_out_result[i], coor_out_result[i], threshold[i], max_peak_num[i]);
        if (ret != SAIL_TPU_KERNEL_SUCCESS) {
            return ret;
        }
    }
    return SAIL_TPU_KERNEL_SUCCESS;
}


tpu_kernel_api_openpose_part_nms::tpu_kernel_api_openpose_part_nms(int device_id, 
                                                                int network_c,
                                                                std::string module_file)
    :_impl (new tpu_kernel_api_openpose_part_nms_cc(device_id, network_c, module_file))
{}

tpu_kernel_api_openpose_part_nms::~tpu_kernel_api_openpose_part_nms()
{
    delete _impl;
}


int tpu_kernel_api_openpose_part_nms::process(TensorPTRWithName& input_data, 
                                            std::vector<int>& shape,
                                            std::vector<std::vector<int>>& num_result, 
                                            std::vector<std::vector<float>>& score_out_result,
                                            std::vector<std::vector<int>>& coor_out_result,
                                            std::vector<float>& threshold,
                                            std::vector<int>& max_peak_num) {
    return _impl->process(input_data, shape, num_result, score_out_result, coor_out_result, threshold, max_peak_num);
}


std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<float>>, std::vector<std::vector<int>>> 
    tpu_kernel_api_openpose_part_nms::process(TensorPTRWithName& input_data, std::vector<int>& shape, std::vector<float>& threshold, std::vector<int>& max_peak_num)
{
    std::vector<std::vector<int>> num_result;
    std::vector<std::vector<float>> score_out_result;
    std::vector<std::vector<int>> coor_out_result;
    int ret = process(input_data, shape, num_result, score_out_result, coor_out_result, threshold, max_peak_num);
    if(ret != SAIL_TPU_KERNEL_SUCCESS){
        return std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<float>>, std::vector<std::vector<int>>>();
    }
    std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<float>>, std::vector<std::vector<int>>> results = std::make_tuple(num_result, score_out_result, coor_out_result);
    return results;
}

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<float>>, std::vector<std::vector<int>>>  
    tpu_kernel_api_openpose_part_nms::process(std::map<std::string, Tensor&>& input, std::vector<int>& shape, std::vector<float>& threshold, std::vector<int>& max_peak_num)
{
    if (input.size() != 1) {
        SPDLOG_ERROR("Only support single input, but get {}!", input.size());
        exit(1);
    }
    
    TensorPTRWithName input_data;
    auto iter_temp = input.begin();
    input_data.data = &iter_temp->second;
    input_data.name = iter_temp->first;
    return process(input_data, shape, threshold, max_peak_num);
}

int tpu_kernel_api_openpose_part_nms::reset_network_c(int network_c_new)
{
    return _impl->reset_network_c(network_c_new);
}

}

#endif