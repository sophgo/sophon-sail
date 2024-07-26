/* Copyright 2016-2022 by Sophgo Technologies Inc. All rights reserved.

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

#include <numeric>
#include <cmath>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
#include "engine_llm.h"
#include "internal.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#ifdef PYTHON
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eval.h>

using namespace pybind11::literals;
#endif // PYTHON

namespace sail {

#ifdef BUILD_ENGINELLM
class GraphLLM
{
private:
    friend class EngineLLM;
    friend class EngineLLM_CC;
    /* data */
    const std::vector<bm_handle_t> &handles_;
    void* p_bmrt_;
    std::string name_;
    std::vector<int> device_ids_;

    bool is_dynamic_ = false;
    int input_num_ = 0;
    int output_num_ = 0;
    int stage_num_ = 0;
    int core_num_ = 0;
    int32_t addr_mode_ = 0;

    int max_stage_ = 0;
    int min_stage_ = 0;

    std::vector<std::string> input_names_;
    std::vector<bm_data_type_t> input_dtypes_;
    std::vector<float> input_scales_;
    std::vector<int> input_devids_;                 // 每个输入数据所在的device id
    std::vector<sail::Handle> input_handles_;       // 每个输入数据对应一个handle


    std::vector<std::string> output_names_;
    std::vector<bm_data_type_t> output_dtypes_;
    std::vector<float> output_scales_;
    std::vector<int> output_devids_;                // 每个输出数据所在的device id
    std::vector<sail::Handle> output_handles_;      // 每个输出数据对应一个handle

    // stage_index, tensor_index, shape
    std::vector<std::vector<std::vector<int>>> input_shapes_;
    std::vector<std::vector<std::vector<int>>> output_shapes_;

    // stage_index, tensor_index, mem
    std::vector<std::vector<bm_device_mem_t>> input_mems_;
    std::vector<std::vector<bm_device_mem_t>> output_mems_;

    // stage_index, tensor_index, Tensor
    std::vector<std::vector<sail::Tensor>> internal_input_tensors_;
    std::vector<std::vector<sail::Tensor>> internal_output_tensors_;

    bm_tensor_t *bm_in_  = nullptr;
    bm_tensor_t *bm_out_ = nullptr;

    int init_graph_info();
    int init_internal_tensors();
    int init_io_max_info();
    int attach_io_tensor(std::map<int, Tensor*>& input, std::map<int, Tensor*>& output);
    std::vector<int> input_name_to_indexes(const std::string &tensor_name) const;
    std::vector<int> output_name_to_indexes(const std::string &tensor_name) const;

    int init_internal_input_tensors();
    int init_internal_output_tensors();

public:
    GraphLLM(
        const std::vector<bm_handle_t>  &handle, 
        void*               p_bmrt, 
        const std::string&  name,
        std::vector<int>    tpu_ids);
    ~GraphLLM();

    int inference(std::map<int, Tensor *> &input,
                  std::map<int, Tensor *> &output,
                  const std::vector<int> &core_list);

    sail::Tensor *           get_input_tensor    (const int index, const int stage = 0);
    std::map<int, Tensor *>  get_input_tensors   (const int stage = 0);
    std::map<int, Tensor *>  get_input_tensors   (const std::string& tensor_name, const int stage = 0);
    std::vector<int>         get_input_shape     (const int index, const int stage = 0) const;
    bm_data_type_t           get_input_dtype     (const int index) const;
    float                    get_input_scale     (const int index) const;

    sail::Tensor *           get_output_tensor   (const int index, const int stage = 0);
    std::map<int, Tensor *>  get_output_tensors  (const int stage = 0);
    std::map<int, Tensor *>  get_output_tensors  (const std::string& tensor_name, const int stage = 0);
    std::vector<int>         get_output_shape    (const int index, const int stage = 0) const;
    bm_data_type_t           get_output_dtype    (const int index) const;
    float                    get_output_scale    (const int index) const;

    int get_input_tensor_devid(const int index) const;
    int get_output_tensor_devid(const int index) const;

    std::vector<int> get_input_max_shape(const int index) const;
    std::vector<int> get_output_max_shape(const int index) const;

    std::vector<int> get_input_min_shape(const int index) const;
    std::vector<int> get_output_min_shape(const int index) const;

    std::map<int, sail::Tensor*> create_stage_input_tensors(const int stage_index);
    std::map<int, sail::Tensor*> create_stage_output_tensors(const int stage_index);

    std::map<int, sail::Tensor*> create_max_input_tensors();
    std::map<int, sail::Tensor*> create_max_output_tensors();

    bool get_is_dynamic() const;
    int get_addr_mode() const;
    int get_stage_num() const;
    int get_input_num() const;
    int get_output_num() const;
    std::string get_input_name(const int index) const;
    std::string get_output_name(const int index) const;
    std::vector<std::string> get_input_names() const;
    std::vector<std::string> get_output_names() const;
};

GraphLLM::GraphLLM(
    const std::vector<bm_handle_t> &handles, 
    void*               p_bmrt, 
    const std::string&  name,
    std::vector<int>    tpu_ids)
    : handles_(handles), p_bmrt_(p_bmrt), name_(name), device_ids_(tpu_ids)
{
    spdlog::trace("net [{}] init graph", name_);
    if (!p_bmrt) {
        SPDLOG_ERROR("Error while constructing Graph: bmruntime is null");
        throw SailEngineError("Graph related error!");
    }
    check_return_status( init_graph_info() );
    // only create sail::Tensor when io alone, i.e., addr_mode = 1
    if (addr_mode_ == 1) 
    {
        check_return_status( init_internal_tensors() );
    }
}

GraphLLM::~GraphLLM()
{
    if(bm_in_){
        delete[] bm_in_;
        bm_in_ = nullptr;
    }
    if(bm_out_){
        delete[] bm_out_;
        bm_out_ = nullptr;
    }
}

// TODO how to def max stage
int GraphLLM::init_io_max_info()
{
    max_stage_ = 0;
    // int stage_i = 0;
    // for (int i = 0; i < stage_num_; i++)
    // {
    //     // std::vector<std::vector<std::vector<int>>> input_shapes_;
    //     // std::vector<std::vector<std::vector<int>>> output_shapes_;
    //     max_stage_ = stage_i;
    // }
    return SAIL_SUCCESS;
}

int GraphLLM::init_graph_info()
{
    spdlog::trace("net [{}] init graph info", name_);
    const bm_net_info_t* p_info = bmrt_get_network_info(p_bmrt_, name_.c_str());
    if(p_info == nullptr){
        return SAIL_ERR_ENGINE_GRAPH;
    }

    is_dynamic_ = p_info->is_dynamic;
    input_num_ = p_info->input_num;
    output_num_ = p_info->output_num;
    stage_num_ = p_info->stage_num;
    core_num_ = p_info->core_num;
    addr_mode_ = p_info->addr_mode;

    bm_in_  = new bm_tensor_t[input_num_];
    bm_out_ = new bm_tensor_t[output_num_];

    // init misc info
    for (int i = 0; i < input_num_; i++)
    {
        input_names_.emplace_back(std::string(p_info->input_names[i]));
        input_dtypes_.emplace_back(p_info->input_dtypes[i]);
        input_scales_.emplace_back(p_info->input_scales[i]);
        // TODO now assume device_ids_ are continuously increasing, like [4,5,6]
        int device_id = device_ids_[p_info->input_loc_devices[i]];
        input_devids_.emplace_back(device_id); // TODO whats this?
        input_handles_.emplace_back(sail::Handle(device_id));
    }
    for (int i = 0; i < output_num_; i++)
    {
        output_names_.emplace_back(std::string(p_info->output_names[i]));
        output_dtypes_.emplace_back(p_info->output_dtypes[i]);
        output_scales_.emplace_back(p_info->output_scales[i]);
        // TODO now assume device_ids_ are continuously increasing, like [4,5,6]
        int device_id = device_ids_[p_info->output_loc_devices[i]];
        output_devids_.emplace_back(device_id); // TODO whats this?
        output_handles_.emplace_back(sail::Handle(device_id));
    }

    // init shape
    for (int i = 0; i < stage_num_; i++)
    {
        // input shape
        std::vector<std::vector<int>> stage_i_in_shapes;
        for (int j = 0; j < input_num_; j++)
        {
            const bm_shape_t input_j_bmshape = p_info->stages[i].input_shapes[j];
            std::vector<int> input_j_shape(input_j_bmshape.dims,
                                           input_j_bmshape.dims + input_j_bmshape.num_dims);
            stage_i_in_shapes.emplace_back(input_j_shape);
        }
        input_shapes_.emplace_back(stage_i_in_shapes);

        // output shape
        std::vector<std::vector<int>> stage_i_out_shapes;
        for (int j = 0; j < output_num_; j++)
        {
            const bm_shape_t output_j_bmshape = p_info->stages[i].output_shapes[j];
            std::vector<int> output_j_shape(output_j_bmshape.dims, 
                                            output_j_bmshape.dims + output_j_bmshape.num_dims);
            stage_i_out_shapes.emplace_back(output_j_shape);
        }
        output_shapes_.emplace_back(stage_i_out_shapes);

        // parse device_mem info when io alone
        // if (0 == addr_mode_) continue;
        std::vector<bm_device_mem_t> stage_i_in_mems;
        for (int j = 0; j < input_num_; j++)
        {
            const bm_device_mem_t &input_j_mem = p_info->stages[i].input_mems[j];
            stage_i_in_mems.emplace_back(input_j_mem);
        }
        input_mems_.emplace_back(stage_i_in_mems);

        std::vector<bm_device_mem_t> stage_i_out_mems;
        for (int j = 0; j < output_num_; j++)
        {
            const bm_device_mem_t &output_j_mem = p_info->stages[i].output_mems[j];
            stage_i_out_mems.emplace_back(output_j_mem);
        }
        output_mems_.emplace_back(stage_i_out_mems);
    }
    
    int ret = init_io_max_info();
    return SAIL_SUCCESS;
}

int GraphLLM::init_internal_input_tensors()
{
    spdlog::trace("net [{}] init internal tensors", name_);
    if (input_mems_.size() == 0 || input_mems_.at(0).size() == 0)
    {
        SPDLOG_ERROR("input_mems is empty.");
        return SAIL_ERR_ENGINE_GRAPH;
    }

    if (0 != internal_input_tensors_.size())
    {
        if (internal_input_tensors_.size() != stage_num_ ||
            internal_input_tensors_.at(0).size() != input_num_)
        {
            SPDLOG_ERROR("internal input tensors are incomplete. "
                         "internal_input_tensors_.size(): {}, "
                         "internal_input_tensors_.size(): {}.", 
                         internal_input_tensors_.size(), 
                         internal_input_tensors_.at(0).size());
            return SAIL_ERR_ENGINE_GRAPH;
        }
        return SAIL_SUCCESS;
    }

    internal_input_tensors_.resize(stage_num_);
    for (int stage_index = 0; stage_index < stage_num_; stage_index++)
    {
        auto &stage_i_in_tensors = internal_input_tensors_.at(stage_index);
        stage_i_in_tensors.reserve(input_num_);
        for (int i = 0; i < input_num_; i++)
        {
            stage_i_in_tensors.emplace_back(sail::Tensor(input_handles_.at(i),
                                                         input_shapes_.at(stage_index).at(i),
                                                         input_dtypes_.at(i),
                                                         false, false));
            stage_i_in_tensors.at(i).reset_dev_data(input_mems_.at(stage_index).at(i));
        }
    }
    return SAIL_SUCCESS;
}

int GraphLLM::init_internal_output_tensors()
{
    spdlog::trace("net [{}] init internal tensors", name_);
    if (output_mems_.size() == 0 || output_mems_.at(0).size() == 0)
    {
        SPDLOG_ERROR("output_mems is empty.");
        return SAIL_ERR_ENGINE_GRAPH;
    }

    if (0 != internal_output_tensors_.size())
    {
        if (internal_output_tensors_.size() != stage_num_ ||
            internal_output_tensors_.at(0).size() != output_num_)
        {
            SPDLOG_ERROR("internal output tensors are incomplete. "
                         "internal_output_tensors_.size(): {}, "
                         "internal_output_tensors_.size(): {}.", 
                         internal_output_tensors_.size(), 
                         internal_output_tensors_.at(0).size());
            return SAIL_ERR_ENGINE_GRAPH;
        }
        return SAIL_SUCCESS;
    }

    internal_output_tensors_.resize(stage_num_);
    for (int stage_index = 0; stage_index < stage_num_; stage_index++)
    {
        auto &stage_i_out_tensors = internal_output_tensors_.at(stage_index);
        stage_i_out_tensors.reserve(output_num_);
        for (int i = 0; i < output_num_; i++)
        {
            stage_i_out_tensors.emplace_back(sail::Tensor(output_handles_.at(i),
                                                         output_shapes_.at(stage_index).at(i),
                                                         output_dtypes_.at(i),
                                                         false, false));
            stage_i_out_tensors.at(i).reset_dev_data(output_mems_.at(stage_index).at(i));
        }
    }
    return SAIL_SUCCESS;
}

int GraphLLM::init_internal_tensors()
{
    spdlog::trace("net [{}] init internal tensors", name_);
    int ret = SAIL_SUCCESS;
    ret = init_internal_input_tensors();
    if (SAIL_SUCCESS != ret)
    {
        SPDLOG_ERROR("init_internal_input_tensors failed! ret: {}.", ret);
        return SAIL_ERR_ENGINE_GRAPH;
    }
    ret = init_internal_output_tensors();
    if (SAIL_SUCCESS != ret)
    {
        SPDLOG_ERROR("init_internal_output_tensors failed! ret: {}.", ret);
        return SAIL_ERR_ENGINE_GRAPH;
    }
    return SAIL_SUCCESS;
}

int GraphLLM::attach_io_tensor(std::map<int, Tensor*>& input,
                               std::map<int, Tensor*>& output)
{
    if (!bm_in_ || !bm_out_) return SAIL_ERR_RUNTIME_BASIC;
    for (int i = 0; i < input_num_; i++)
    {
        bm_in_[i].dtype = input_dtypes_.at(i);
        bm_in_[i].device_mem = input[i]->dev_data();
        bm_in_[i].st_mode = BM_STORE_1N;

        const std::vector<int> shape_tmp = input.at(i)->shape();
        bm_in_[i].shape.num_dims = shape_tmp.size();
        for (int j = 0; j < shape_tmp.size(); ++j) {
            bm_in_[i].shape.dims[j] = shape_tmp.at(j);
        }
    }
    for (int i = 0; i < output_num_; i++)
    {
        bm_out_[i].dtype = output_dtypes_.at(i);
        bm_out_[i].device_mem = output[i]->dev_data();
        bm_out_[i].st_mode = BM_STORE_1N;

        const std::vector<int> shape_tmp = output.at(i)->shape();
        bm_out_[i].shape.num_dims = shape_tmp.size();
        for (int j = 0; j < shape_tmp.size(); ++j) {
            bm_out_[i].shape.dims[j] = shape_tmp.at(j);
        }
    }
    return SAIL_SUCCESS;
}

int GraphLLM::inference(std::map<int, Tensor*>& input,
                        std::map<int, Tensor*>& output,
                        const std::vector<int> &core_list)
{
    int ret = attach_io_tensor(input, output);
    if (ret) return ret;
    bool ok = bmrt_launch_tensor_ex(p_bmrt_,
                                    name_.c_str(),
                                    bm_in_,
                                    input_num_,
                                    bm_out_,
                                    output_num_,
                                    true,
                                    true);
    if (false == ok)
    {
        SPDLOG_ERROR("bmrt_launch_tensor_ex fail!");
        return SAIL_ERR_ENGINE_INFER;
    }
    ret = bm_thread_sync(handles_.at(0));
    if (BM_SUCCESS != ret)
    {
        SPDLOG_ERROR("bm_thread_sync fail! ret: {}.", ret);
        return SAIL_ERR_ENGINE_INFER;
    }
    return SAIL_SUCCESS;
}

std::vector<int> GraphLLM::input_name_to_indexes(const std::string &tensor_name) const
{
    std::vector<int> indexes = {};
    for (auto i = 0; i < input_num_; i++) {
        if (input_names_.at(i) == tensor_name) indexes.emplace_back(i);
    }
    return indexes;
}

std::vector<int> GraphLLM::output_name_to_indexes(const std::string &tensor_name) const
{
    std::vector<int> indexes = {};
    for (auto i = 0; i < output_num_; i++) {
        if (output_names_.at(i) == tensor_name) indexes.emplace_back(i);
    }
    return indexes;
}

std::map<int, Tensor *> GraphLLM::get_input_tensors(const int stage)
{
    std::map<int, Tensor *> ret_input_tensors = {};
    for (int i = 0; i < input_num_; i++) {
        ret_input_tensors.emplace(std::make_pair(i, &internal_input_tensors_.at(stage).at(i)));
    }
    return std::move(ret_input_tensors);
}

std::map<int, Tensor*> GraphLLM::get_input_tensors(const std::string& tensor_name, 
                                                   const int stage)
{
    std::map<int, Tensor *> ret_input_tensors = {};
    auto indexes = input_name_to_indexes(tensor_name);
    for (const auto &i : indexes) {
        ret_input_tensors.emplace(std::make_pair(i, &internal_input_tensors_.at(stage).at(i)));
    }
    return ret_input_tensors;
}

sail::Tensor* GraphLLM::get_input_tensor(const int index, const int stage)
{
    return &internal_input_tensors_.at(stage).at(index);
}

std::map<int, Tensor *> GraphLLM::get_output_tensors(const int stage)
{
    std::map<int, Tensor *> ret_output_tensors = {};
    for (int i = 0; i < output_num_; i++) {
        ret_output_tensors.emplace(std::make_pair(i, &internal_output_tensors_.at(stage).at(i)));
    }
    return ret_output_tensors;
}

std::map<int, Tensor*> GraphLLM::get_output_tensors(const std::string& tensor_name, 
                                                   const int stage)
{
    std::map<int, Tensor *> ret_output_tensors = {};
    auto indexes = output_name_to_indexes(tensor_name);
    for (const auto &i : indexes) {
        ret_output_tensors.emplace(std::make_pair(i, &internal_output_tensors_.at(stage).at(i)));
    }
    return ret_output_tensors;
}

sail::Tensor* GraphLLM::get_output_tensor(const int index, const int stage)
{
    return &internal_output_tensors_.at(stage).at(index);
}

std::vector<int> GraphLLM::get_input_shape(const int index, const int stage) const
{
    return input_shapes_.at(stage).at(index);
}

bm_data_type_t GraphLLM::get_input_dtype(const int index) const
{
    return input_dtypes_.at(index);
}

float GraphLLM::get_input_scale(const int index) const
{
    return input_scales_.at(index);
}

std::vector<int> GraphLLM::get_output_shape(const int index, const int stage) const
{
    return output_shapes_.at(stage).at(index);
}

bm_data_type_t GraphLLM::get_output_dtype(const int index) const
{
    return output_dtypes_.at(index);
}

float GraphLLM::get_output_scale(const int index) const
{    
    return output_scales_.at(index);
}

int GraphLLM::get_input_tensor_devid(const int index) const
{
    // TODO index check
    return input_devids_.at(index);
}
int GraphLLM::get_output_tensor_devid(const int index) const
{
    return output_devids_.at(index);
}

std::vector<int> GraphLLM::get_input_max_shape(const int index) const
{
    return input_shapes_.at(max_stage_).at(index);
}
std::vector<int> GraphLLM::get_output_max_shape(const int index) const
{
    return output_shapes_.at(max_stage_).at(index);
}

std::vector<int> GraphLLM::get_input_min_shape(const int index) const
{
    return input_shapes_.at(min_stage_).at(index);
}
std::vector<int> GraphLLM::get_output_min_shape(const int index) const
{
    return output_shapes_.at(min_stage_).at(index);
}

std::map<int, sail::Tensor*> GraphLLM::create_stage_input_tensors(const int stage_index)
{
    // if (addr_mode_ != 0) throw SailTensorError("this func is only for addr mode 0!");
    if (stage_index < 0 || stage_index >= stage_num_) 
        throw SailTensorError("input stage index invalid!");
    std::map<int, sail::Tensor*> ret_input_tensors = {};

    // // by this implement, created tensors are **bind** to this graph. tensor will be release when ~graph()
    // auto &stage_i_in_tensors = internal_input_tensors_.at(stage_index);
    // // target tensors have been created
    // if (stage_i_in_tensors.size() == input_num_)
    // {
    //     for (int i = 0; i < input_num_; i++){
    //         ret_input_tensors.emplace(std::make_pair(i, &stage_i_in_tensors.at(i)));
    //     }
    //     return ret_input_tensors;
    // }
    // // target tensors have not been created
    // stage_i_in_tensors.reserve(input_num_);
    // for (int i = 0; i < input_num_; i++)
    // {
    //     stage_i_in_tensors.emplace_back(sail::Tensor(input_handles_.at(i),
    //                                                  input_shapes_.at(stage_index).at(i),
    //                                                  input_dtypes_.at(i),
    //                                                  false, true));
    //     ret_input_tensors.emplace(std::make_pair(i, &stage_i_in_tensors.at(i)));
    // }
    // return ret_input_tensors;

    // by this implement, created tensors are **independent** to this graph.
    // TODO return a independent Tensor for Python
    for (int i = 0; i < input_num_; i++)
    {
        sail::Tensor *tensor_i = new sail::Tensor(input_handles_.at(i),
                                                  input_shapes_.at(stage_index).at(i),
                                                  input_dtypes_.at(i),
                                                  false, true);
        ret_input_tensors.emplace(std::make_pair(i, tensor_i));
    }
    return ret_input_tensors;
}

std::map<int, sail::Tensor*> GraphLLM::create_stage_output_tensors(const int stage_index)
{
    // if (addr_mode_ != 0) throw SailTensorError("this func is only for addr mode 0!");
    if (stage_index < 0 || stage_index >= stage_num_) 
        throw SailTensorError("output stage index invalid!");
    std::map<int, sail::Tensor*> ret_output_tensors = {};

    // auto &stage_i_out_tensors = internal_output_tensors_.at(stage_index);
    // // target tensors have been created
    // if (stage_i_out_tensors.size() == input_num_)
    // {
    //     for (int i = 0; i < input_num_; i++){
    //         ret_output_tensors.emplace(std::make_pair(i, &stage_i_out_tensors.at(i)));
    //     }
    //     return ret_output_tensors;
    // }
    // // target tensors have not been created
    // stage_i_out_tensors.reserve(output_num_);
    // for (int i = 0; i < output_num_; i++)
    // {
    //     stage_i_out_tensors.emplace_back(sail::Tensor(output_handles_.at(i),
    //                                                   output_shapes_.at(stage_index).at(i),
    //                                                   output_dtypes_.at(i),
    //                                                   false, true));
    //     ret_output_tensors.emplace(std::make_pair(i, &stage_i_out_tensors.at(i)));
    // }
    // return ret_output_tensors;

    for (int i = 0; i < output_num_; i++)
    {
        sail::Tensor *tensor_i = new sail::Tensor(output_handles_.at(i),
                                                  output_shapes_.at(stage_index).at(i),
                                                  output_dtypes_.at(i),
                                                  false, true);
        ret_output_tensors.emplace(std::make_pair(i, tensor_i));
    }
    return ret_output_tensors;
}

std::map<int, sail::Tensor*> GraphLLM::create_max_input_tensors()
{
    // if (addr_mode_ != 0) throw SailTensorError("this func is only for addr mode 0!");
    return create_stage_input_tensors(max_stage_);
}
std::map<int, sail::Tensor*> GraphLLM::create_max_output_tensors()
{
    // if (addr_mode_ != 0) throw SailTensorError("this func is only for addr mode 0!");
    return create_stage_output_tensors(max_stage_);
}

bool GraphLLM::get_is_dynamic() const {
    return is_dynamic_;
}
int GraphLLM::get_addr_mode() const {
    return addr_mode_;
}
int GraphLLM::get_stage_num() const {
    return stage_num_;
}
int GraphLLM::get_input_num() const {
    return input_num_;
}
int GraphLLM::get_output_num() const {
    return output_num_;
}
std::string GraphLLM::get_input_name(const int index) const {
    return input_names_.at(index);
}
std::string GraphLLM::get_output_name(const int index) const {
    return output_names_.at(index);
}
std::vector<std::string> GraphLLM::get_input_names() const {
    return input_names_;
}
std::vector<std::string> GraphLLM::get_output_names() const {
    return output_names_;
}


class EngineLLM::EngineLLM_CC{
public:
    /**
     * @brief Constructor loads bmodel from file.
     *
     * @param bmodel_path Path to bmodel
     * @param tpu_ids     TPU ID list. You can use bm-smi to see available IDs
     **/
    EngineLLM_CC(
        const std::string& bmodel_path,
        std::vector<int>   tpu_ids);

    /**
     * @brief Constructor loads bmodel from system memory.
     *
     * @param bmodel_ptr  Pointer to bmodel in system memory
     * @param bmodel_size Byte size of bmodel in system memory
     * @param tpu_ids     TPU ID list. You can use bm-smi to see available IDs.
     */
    EngineLLM_CC(
        const void*       bmodel_ptr,
        size_t            bmodel_size,
        std::vector<int>  tpu_ids);

#ifdef PYTHON
    EngineLLM_CC(
        pybind11::bytes&  bmodel_bytes,
        int               bmodel_size,
        std::vector<int>  tpu_id);

#endif // PYTHON

    ~EngineLLM_CC();

  /// Graph instance for each model.
  std::map<std::string, std::shared_ptr<GraphLLM>> graphs_;

  std::vector<std::string> get_graph_names() const;

  std::vector<int> get_device_ids() const;

  int get_stage_num(const std::string& graph_name) const;

  int get_input_num(const std::string& graph_name) const;

  int get_output_num(const std::string& graph_name) const;
    
  int get_addr_mode(const std::string& graph_name) const;

  std::string get_input_name(const std::string& graph_name, const int index) const;
  std::string get_output_name(const std::string& graph_name, const int index) const;
  std::vector<std::string> get_input_names(const std::string& graph_name) const;
  std::vector<std::string> get_output_names(const std::string& graph_name) const;

  bool get_is_dynamic(const std::string& graph_name) const;

// TODO multi cores
  void process(const std::string& graph_name,
               std::map<int, Tensor*>& input,
               std::map<int, Tensor*>& output,
               const std::vector<int> &core_list);

  std::map<int, Tensor *> get_input_tensors(const std::string& graph_name, const int stage = 0);

  std::map<int, Tensor *> get_input_tensors(const std::string& graph_name, 
                                            const std::string& tensor_name, 
                                            const int stage = 0);

  sail::Tensor* get_input_tensor(const std::string& graph_name, const int dev_index, const int stage = 0);

  std::map<int, Tensor *> get_output_tensors(const std::string& graph_name, const int stage = 0);

  std::map<int, Tensor *> get_output_tensors(const std::string& graph_name, 
                                            const std::string& tensor_name, 
                                            const int stage = 0);

  sail::Tensor* get_output_tensor(const std::string& graph_name, const int dev_index, const int stage = 0);

  std::map<int, Tensor *> get_input_tensors_addrmode0(const std::string &graph_name, const int stage = 0);
  std::map<int, Tensor *> get_output_tensors_addrmode0(const std::string &graph_name, const int stage = 0);

  std::vector<int> get_input_shape(const std::string& graph_name, const int index, const int stage = 0) const;

  std::vector<int> get_output_shape(const std::string& graph_name, const int index, const int stage = 0) const;

  bm_data_type_t get_input_dtype(const std::string& graph_name, const int index) const;

  bm_data_type_t get_output_dtype(const std::string& graph_name, const int index) const;
  
  float get_input_scale(const std::string& graph_name, const int index) const;

  float get_output_scale(const std::string& graph_name, const int index) const;

    int get_input_tensor_devid(const std::string& graph_name, const int index) const;
    int get_output_tensor_devid(const std::string& graph_name, const int index) const;

    std::vector<int> get_input_max_shape(const std::string& graph_name, const int index) const;
    std::vector<int> get_output_max_shape(const std::string& graph_name, const int index) const;

    std::vector<int> get_input_min_shape(const std::string& graph_name, const int index) const;
    std::vector<int> get_output_min_shape(const std::string& graph_name, const int index) const;

    std::map<int, sail::Tensor*> create_max_input_tensors(const std::string& graph_name);
    std::map<int, sail::Tensor*> create_max_output_tensors(const std::string& graph_name);

private:
  friend class EngineLLM;

  /// Pointer to bmruntime instance.
  void* p_bmrt_;

  /// Handle instance.
  std::vector<bm_handle_t> handles_;

  std::vector<int> device_ids_;

  // graph info
  int graph_num_;
  std::vector<std::string> graph_names_;

  int create_handles();
  int init_graph_names();
};

EngineLLM::EngineLLM_CC::EngineLLM_CC(const std::string& bmodel_path, std::vector<int> tpu_ids)
    :p_bmrt_(nullptr),device_ids_(tpu_ids),handles_({}),graph_num_(NULL),graph_names_({})
{
    // bm_dev_getcount
    if(device_ids_.size() <= 0){ // or tpu_id > tpu_num
        SPDLOG_ERROR("Input tpus is empty!");
        throw SailEngineError("EngineLLM device_ids error!");
    }
    // TODO sort device_ids_

    struct stat buffer;
    if (stat(bmodel_path.c_str(), &buffer) != 0) {
        SPDLOG_ERROR("bmodel {} does not exist", bmodel_path);
        throw SailEngineError("EngineLLM bmodel error!");
    }
    check_return_status( create_handles() );

    if(bmrt_load_bmodel(p_bmrt_, bmodel_path.c_str()) == false){
        SPDLOG_ERROR("Load bmodel {} failed", bmodel_path);
        throw SailEngineError("EngineLLM related error!");
    }

    int ret = init_graph_names(); // init graph_names

    for (auto graph_name : graph_names_) {
        // std::shared_ptr<GraphLLM> graph(new GraphLLM(handles_[0], p_bmrt_, graph_name, tpu_ids));
        auto graph = std::make_shared<GraphLLM>(handles_, p_bmrt_, graph_name, device_ids_);
        graphs_[graph_name] = graph;
    }
}


EngineLLM::EngineLLM_CC::EngineLLM_CC(
        const void*       bmodel_ptr,
        size_t            bmodel_size,
        std::vector<int>  tpu_ids)
    :p_bmrt_(nullptr),device_ids_(tpu_ids),handles_({}),graph_num_(NULL),graph_names_({})
{
    check_return_status( create_handles() );

    if (!bmrt_load_bmodel_data(p_bmrt_, bmodel_ptr, bmodel_size)) {
        SPDLOG_ERROR("Load bmodel failed");
        throw SailEngineError("EngineLLM related error!");
    }

    int ret = init_graph_names(); // init graph_names

    for (auto graph_name : graph_names_) {
        auto graph = std::make_shared<GraphLLM>(handles_, p_bmrt_, graph_name, device_ids_);
        graphs_[graph_name] = graph;
    }
}

#ifdef PYTHON
EngineLLM::EngineLLM_CC::EngineLLM_CC(
        pybind11::bytes&  bmodel_bytes,
        int               bmodel_size,
        std::vector<int>  tpu_ids)
    :p_bmrt_(nullptr),device_ids_(tpu_ids),handles_({}),graph_num_(NULL),graph_names_({})
{
    check_return_status( create_handles() );

    char* bmodel_ptr = nullptr;
    ssize_t size;

    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(bmodel_bytes.ptr(), &bmodel_ptr, &size)) {
        SPDLOG_ERROR("Unable to extract bytes contents!");
        throw SailEngineError("EngineLLM related error!");
    }

    if (bmodel_size != static_cast<int>(size)) {
        SPDLOG_ERROR("Wrong bmodel_size.");
        throw SailEngineError("EngineLLM related error!");
    }

    if (!bmrt_load_bmodel_data(p_bmrt_, (const void*)bmodel_ptr, bmodel_size)) {
        SPDLOG_ERROR("Load bmodel failed");
        throw SailEngineError("EngineLLM related error!");
    }

    int ret = SAIL_SUCCESS;
    ret = init_graph_names(); // init graph_names

    for (auto graph_name : graph_names_) {
        auto graph = std::make_shared<GraphLLM>(handles_, p_bmrt_, graph_name, device_ids_);
        graphs_[graph_name] = graph;
    }
}

#endif // PYTHON

EngineLLM::EngineLLM_CC::~EngineLLM_CC()
{
    bmrt_destroy(p_bmrt_);
    for (auto &handle : handles_) 
    {
        bm_dev_free(handle);
    }
}

int EngineLLM::EngineLLM_CC::create_handles()
{
    handles_.clear();
    for (auto dev_id : device_ids_) {
        if (BM_SUCCESS != bm_dev_query(dev_id)) {
            SPDLOG_ERROR("Query tpu id: {} failed! tpu {} may not exist!", dev_id, dev_id);
            return SAIL_ERR_DEV_HANDLE;
        }
        bm_handle_t h_temp;
        if (BM_SUCCESS != bm_dev_request(&h_temp, dev_id)) {
            SPDLOG_ERROR("Request tpu id: {} Failed!", dev_id);
            return SAIL_ERR_DEV_HANDLE;
        }
        handles_.push_back(h_temp);
    }
    p_bmrt_ = bmrt_create_ex(handles_.data(), handles_.size());
    if(p_bmrt_ == nullptr){
        SPDLOG_ERROR("Runtime Create Failed!");
        return SAIL_ERR_RUNTIME_BASIC;
    }
    return SAIL_SUCCESS;
}

int EngineLLM::EngineLLM_CC::init_graph_names() {
    // vector<string> graph_names;
    graph_num_ = bmrt_get_network_number(p_bmrt_);
    if (!graph_num_ || graph_num_ <= 0) return SAIL_ERR_ENGINE_GRAPH;
    graph_names_.clear();
    const char** names = nullptr;
    bmrt_get_network_names(p_bmrt_, &names);
    if (!names) return SAIL_ERR_ENGINE_GRAPH;
    for (int i = 0; i < graph_num_; ++i) {
        graph_names_.push_back(names[i]);
    }
    std::free(names);
    if (graph_num_ != graph_names_.size()) return SAIL_ERR_ENGINE_GRAPH;
    return SAIL_SUCCESS;        
}

std::vector<int> EngineLLM::EngineLLM_CC::get_device_ids() const {
    return device_ids_;
}

std::vector<std::string> EngineLLM::EngineLLM_CC::get_graph_names() const {
    return graph_names_;
}

int EngineLLM::EngineLLM_CC::get_addr_mode(const std::string& graph_name) const {
    return graphs_.at(graph_name)->get_addr_mode();
}

int EngineLLM::EngineLLM_CC::get_stage_num(const std::string& graph_name) const {
    return graphs_.at(graph_name)->get_stage_num();
}

int EngineLLM::EngineLLM_CC::get_input_num(const std::string& graph_name) const {
    return graphs_.at(graph_name)->get_input_num();
}

int EngineLLM::EngineLLM_CC::get_output_num(const std::string& graph_name) const {
    return graphs_.at(graph_name)->get_output_num();
}

bool EngineLLM::EngineLLM_CC::get_is_dynamic(const std::string& graph_name) const {
    return graphs_.at(graph_name)->get_is_dynamic();
}

std::string EngineLLM::EngineLLM_CC::get_input_name(
    const std::string& graph_name, const int index) const {
    return graphs_.at(graph_name)->get_input_name(index);
}

std::string EngineLLM::EngineLLM_CC::get_output_name(
    const std::string &graph_name, const int index) const {
    return graphs_.at(graph_name)->get_output_name(index);
}

std::vector<std::string> EngineLLM::EngineLLM_CC::get_input_names(
    const std::string& graph_name) const {
    return graphs_.at(graph_name)->get_input_names();
}

std::vector<std::string> EngineLLM::EngineLLM_CC::get_output_names(
    const std::string &graph_name) const {
    return graphs_.at(graph_name)->get_output_names();
}

// TODO multi cores
void EngineLLM::EngineLLM_CC::process(
    const std::string &graph_name,
    std::map<int, Tensor *> &input,
    std::map<int, Tensor *> &output,
    const std::vector<int> &core_list)
{
    int ret = graphs_.at(graph_name)->inference(input, output, core_list);
    check_return_status( ret );
}

std::map<int, Tensor *> EngineLLM::EngineLLM_CC::get_input_tensors(const std::string &graph_name, const int stage)
{
    if (1 != graphs_.at(graph_name)->get_addr_mode())
    {
        SPDLOG_ERROR("Can not get internal input tensors when addr_mode is not 1.");
        return std::map<int, Tensor *>{};
    }
    return std::move(graphs_.at(graph_name)->get_input_tensors(stage));
}

std::map<int, Tensor *> EngineLLM::EngineLLM_CC::get_input_tensors(
    const std::string &graph_name, const std::string &tensor_name, const int stage)
{
    if (1 != graphs_.at(graph_name)->get_addr_mode())
    {
        SPDLOG_ERROR("Can not get internal input tensors when addr_mode is not 1.");
        return std::map<int, Tensor *>{};
    }
    return std::move(graphs_.at(graph_name)->get_input_tensors(tensor_name, stage));
}

sail::Tensor* EngineLLM::EngineLLM_CC::get_input_tensor(const std::string& graph_name, const int index, const int stage)
{
    if (1 != graphs_.at(graph_name)->get_addr_mode())
    {
        SPDLOG_ERROR("Can not get internal input tensor when addr_mode is not 1.");
        return nullptr;
    }
    return std::move(graphs_.at(graph_name)->get_input_tensor(index, stage));
}

std::map<int, Tensor *> EngineLLM::EngineLLM_CC::get_output_tensors(const std::string& graph_name, const int stage)
{
    if (1 != graphs_.at(graph_name)->get_addr_mode())
    {
        SPDLOG_ERROR("Can not get internal output tensors when addr_mode is not 1.");
        return std::map<int, Tensor *>{};
    }
    return std::move(graphs_.at(graph_name)->get_output_tensors(stage));
}

std::map<int, Tensor *> EngineLLM::EngineLLM_CC::get_output_tensors(
    const std::string &graph_name, const std::string &tensor_name, const int stage)
{
    if (1 != graphs_.at(graph_name)->get_addr_mode())
    {
        SPDLOG_ERROR("Can not get internal output tensors when addr_mode is not 1.");
        return std::map<int, Tensor *>{};
    }
    return std::move(graphs_.at(graph_name)->get_output_tensors(tensor_name, stage));
}

sail::Tensor* EngineLLM::EngineLLM_CC::get_output_tensor(const std::string& graph_name, const int index, const int stage)
{
    if (1 != graphs_.at(graph_name)->get_addr_mode())
    {
        SPDLOG_ERROR("Can not get internal output tensor when addr_mode is not 1.");
        return nullptr;
    }
    return std::move(graphs_.at(graph_name)->get_output_tensor(index, stage));
}

std::map<int, Tensor *> EngineLLM::EngineLLM_CC::get_input_tensors_addrmode0(const std::string &graph_name, const int stage)
{
    SPDLOG_WARN("Try to get internal input tensors when addr_mode is not 1.");
    int ret = graphs_.at(graph_name)->init_internal_input_tensors();
    if (SAIL_SUCCESS != ret)
    {
        SPDLOG_ERROR("net [{}] init_internal_input_tensors failed! ret: {}.", graph_name, ret);
        return std::map<int, Tensor *>{};
    }
    return std::move(graphs_.at(graph_name)->get_input_tensors(stage));
}

std::map<int, Tensor *> EngineLLM::EngineLLM_CC::get_output_tensors_addrmode0(const std::string &graph_name, const int stage)
{
    SPDLOG_WARN("Try to get internal output tensors when addr_mode is not 1.");
    int ret = graphs_.at(graph_name)->init_internal_output_tensors();
    if (SAIL_SUCCESS != ret)
    {
        SPDLOG_ERROR("net [{}] init_internal_output_tensors failed! ret: {}.", graph_name, ret);
        return std::map<int, Tensor *>{};
    }
    return std::move(graphs_.at(graph_name)->get_output_tensors(stage));
}


std::vector<int> EngineLLM::EngineLLM_CC::get_input_shape(const std::string& graph_name, const int index, const int stage) const
{
    return graphs_.at(graph_name)->get_input_shape(index, stage);
}

std::vector<int> EngineLLM::EngineLLM_CC::get_output_shape(const std::string& graph_name, const int index, const int stage) const
{
    return graphs_.at(graph_name)->get_output_shape(index, stage);
}

bm_data_type_t EngineLLM::EngineLLM_CC::get_input_dtype(const std::string& graph_name, const int index) const
{
    return graphs_.at(graph_name)->get_input_dtype(index);
}

bm_data_type_t EngineLLM::EngineLLM_CC::get_output_dtype(const std::string& graph_name, const int index) const
{
    return graphs_.at(graph_name)->get_output_dtype(index);
}

float EngineLLM::EngineLLM_CC::get_input_scale(const std::string& graph_name, const int index) const
{
    return graphs_.at(graph_name)->get_input_scale(index);
}

float EngineLLM::EngineLLM_CC::get_output_scale(const std::string& graph_name, const int index) const
{
    return graphs_.at(graph_name)->get_output_scale(index);
}

int EngineLLM::EngineLLM_CC::get_input_tensor_devid(const std::string& graph_name, const int index) const {
    return graphs_.at(graph_name)->get_input_tensor_devid(index);
}
int EngineLLM::EngineLLM_CC::get_output_tensor_devid(const std::string& graph_name, const int index) const {
    return graphs_.at(graph_name)->get_output_tensor_devid(index);
}

std::vector<int> EngineLLM::EngineLLM_CC::get_input_max_shape(const std::string& graph_name, const int index) const {
    return graphs_.at(graph_name)->get_input_max_shape(index);
}
std::vector<int> EngineLLM::EngineLLM_CC::get_output_max_shape(const std::string& graph_name, const int index) const {
    return graphs_.at(graph_name)->get_output_max_shape(index);
}

std::vector<int> EngineLLM::EngineLLM_CC::get_input_min_shape(const std::string& graph_name, const int index) const {
    return graphs_.at(graph_name)->get_input_min_shape(index);
}
std::vector<int> EngineLLM::EngineLLM_CC::get_output_min_shape(const std::string& graph_name, const int index) const {
    return graphs_.at(graph_name)->get_output_min_shape(index);
}

std::map<int, sail::Tensor*> EngineLLM::EngineLLM_CC::create_max_input_tensors(const std::string& graph_name) {
    return graphs_.at(graph_name)->create_max_input_tensors();
}
std::map<int, sail::Tensor*> EngineLLM::EngineLLM_CC::create_max_output_tensors(const std::string& graph_name) {
    return graphs_.at(graph_name)->create_max_output_tensors();
}

EngineLLM::EngineLLM(
    const std::string& bmodel_path,
    std::vector<int>   tpu_ids)
    :_impl(new EngineLLM_CC(bmodel_path, tpu_ids)){
}

EngineLLM::EngineLLM(
    const void*       bmodel_ptr,
    size_t            bmodel_size,
    std::vector<int>  tpu_ids)
    :_impl(new EngineLLM_CC(bmodel_ptr,bmodel_size, tpu_ids)){
}

EngineLLM::~EngineLLM() {
    delete _impl;
}

int EngineLLM::get_addr_mode(const std::string& graph_name) const {
    return _impl->get_addr_mode(graph_name);
}

int EngineLLM::get_stage_num(const std::string& graph_name) const {
    return _impl->get_stage_num(graph_name);
}
int EngineLLM::get_input_num(const std::string& graph_name) const {
    return _impl->get_input_num(graph_name);
}
int EngineLLM::get_output_num(const std::string& graph_name) const {
    return _impl->get_output_num(graph_name);
}

bool EngineLLM::get_is_dynamic(const std::string& graph_name) const {
    return _impl->get_is_dynamic(graph_name);
}

std::map<int, Tensor *> EngineLLM::get_input_tensors(const std::string& graph_name, const int stage) {
    return _impl->get_input_tensors(graph_name, stage);
}

std::map<int, Tensor *> EngineLLM::get_input_tensors(const std::string& graph_name, const std::string& tensor_name, const int stage) {
    return _impl->get_input_tensors(graph_name, tensor_name, stage);
}

sail::Tensor* EngineLLM::get_input_tensor(const std::string& graph_name, int index, const int stage) {
    return _impl->get_input_tensor(graph_name, index, stage);
}

std::map<int, Tensor *> EngineLLM::get_output_tensors(const std::string& graph_name, const int stage) {
    return _impl->get_output_tensors(graph_name, stage);
}

std::map<int, Tensor *> EngineLLM::get_output_tensors(const std::string& graph_name, const std::string& tensor_name, const int stage) {
    return _impl->get_output_tensors(graph_name, tensor_name, stage);
}

sail::Tensor* EngineLLM::get_output_tensor(const std::string& graph_name, int index, const int stage) {
    return _impl->get_output_tensor(graph_name, index, stage);
}

std::map<int, Tensor *> EngineLLM::get_input_tensors_addrmode0(const std::string &graph_name, const int stage) {
    return _impl->get_input_tensors_addrmode0(graph_name, stage);
}

std::map<int, Tensor *> EngineLLM::get_output_tensors_addrmode0(const std::string &graph_name, const int stage) {
    return _impl->get_output_tensors_addrmode0(graph_name, stage);
}

std::vector<int> EngineLLM::get_device_ids() const {
    return _impl->get_device_ids();
}

std::vector<std::string> EngineLLM::get_graph_names() const {
    return _impl->get_graph_names();
}

std::string EngineLLM::get_input_name(const std::string& graph_name, const int index) const {
    return _impl->get_input_name(graph_name, index);
}

std::string EngineLLM::get_output_name(const std::string& graph_name, const int index) const {
    return _impl->get_output_name(graph_name, index);
}

std::vector<std::string> EngineLLM::get_input_names(const std::string& graph_name) const {
    return _impl->get_input_names(graph_name);
}

std::vector<std::string> EngineLLM::get_output_names(const std::string& graph_name) const {
    return _impl->get_output_names(graph_name);
}

std::vector<int> EngineLLM::get_input_shape(const std::string& graph_name, const int index, const int stage) const {
    return _impl->get_input_shape(graph_name, index, stage);
}

std::vector<int> EngineLLM::get_output_shape(const std::string& graph_name, const int index, const int stage) const {
    return _impl->get_output_shape(graph_name, index, stage);
}

bm_data_type_t EngineLLM::get_input_dtype(const std::string& graph_name, const int index) const
{
    return _impl->get_input_dtype(graph_name, index);
}

bm_data_type_t EngineLLM::get_output_dtype(const std::string& graph_name, const int index) const
{
    return _impl->get_output_dtype(graph_name, index);
}

float EngineLLM::get_input_scale(const std::string& graph_name, const int index) const
{
    return _impl->get_input_scale(graph_name, index);
}

float EngineLLM::get_output_scale(const std::string& graph_name, const int index) const
{
    return _impl->get_output_scale(graph_name, index);
}

    int EngineLLM::get_input_tensor_devid(const std::string& graph_name, const int index) const {
        return _impl->get_input_tensor_devid(graph_name, index);
    }
    int EngineLLM::get_output_tensor_devid(const std::string& graph_name, const int index) const {
        return _impl->get_output_tensor_devid(graph_name, index);
    }

    std::vector<int> EngineLLM::get_input_max_shape(const std::string& graph_name, const int index) const {
        return _impl->get_input_max_shape(graph_name, index);
    }
    std::vector<int> EngineLLM::get_output_max_shape(const std::string& graph_name, const int index) const {
        return _impl->get_output_max_shape(graph_name, index);
    }

    std::vector<int> EngineLLM::get_input_min_shape(const std::string& graph_name, const int index) const {
        return _impl->get_input_min_shape(graph_name, index);
    }
    std::vector<int> EngineLLM::get_output_min_shape(const std::string& graph_name, const int index) const {
        return _impl->get_output_min_shape(graph_name, index);
    }

    std::map<int, sail::Tensor*> EngineLLM::create_max_input_tensors(const std::string& graph_name){
        return _impl->create_max_input_tensors(graph_name);
    }
    std::map<int, sail::Tensor*> EngineLLM::create_max_output_tensors(const std::string& graph_name){
        return _impl->create_max_output_tensors(graph_name);
    }


// TODO multi cores
void EngineLLM::process(const std::string &graph_name, std::map<int, Tensor *> &input, 
                        std::map<int, Tensor *> &output, const std::vector<int> &core_list)
{
    return _impl->process(graph_name, input, output, core_list);
}

#ifdef PYTHON
EngineLLM::EngineLLM(
    pybind11::bytes&  bmodel_bytes,
    int               bmodel_size,
    std::vector<int>  tpu_ids)
    :_impl(new EngineLLM_CC(bmodel_bytes, bmodel_size, tpu_ids)){
}

#endif // PYTHON
#endif // BUILD_ENGINELLM



}
