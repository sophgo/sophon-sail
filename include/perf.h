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

/** @file     engine_multi.h
 *  @brief    Header file of Perf
 *  @author   zhiju.huang
 *  @version  3.6.0
 *  @date     2024-01-22
 */

#pragma once
#include "cvwrapper.h"
#include <deque>
#include <mutex>
#include <condition_variable>
#include <map>
#include <iostream>
#include <string>

#ifdef PYTHON
#include <type_traits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#endif

using namespace std;
namespace sail{

 class Perf{
    public:
        /**
         * @brief Constructor
         * 
         * @param bmodel_path        Path to bmodel
         * @param tpu_ids            ID of TPUs, there may be more than one TPU for PCIE mode.
         * @param max_que_size       max queue size.
         * @param mode               Specify the input/output tensors are in system memory
         *                           or device memory
         * @param thread_count       thread counts with each tpu.
         * @param free_input         release memory of input, default false.
        */
        Perf(const std::string& bmodel_path, std::vector<int> tpu_ids, int max_que_size, IOMode mode=SYSO,int thread_count=2, bool free_input=false);

        ~Perf();
 
         /**
         * @brief Push Tensors
         * 
         * @param tensor_index  index number of the Tensors.
         * @param input_tensors Input tensors with name
         * @return 0 for success and other for failure 
         */
        int PushTensor(int tensor_index, std::vector<TensorPTRWithName>& input_tensors);

        /**
         * @brief Stop Push Tensors
         */
        void SetEnd();

        /**
         * @brief Get the Result Data object
         * 
         * @return std::tuple<int,               Original Tensor Index
         * std::vector<TensorPTRWithName>>       Output Tensor Results
         */
        std::tuple<int, std::vector<TensorPTRWithName>> GetResult();

        /**
         * @brief Get all graph names in the loaded bomodels.
         *
         * @return All graph names
         */
        std::vector<std::string> get_graph_names();

        /**
         * @brief Get all input tensor names of the specified graph.
         *
         * @param graph_name The specified graph name
         * @return All the input tensor names of the graph
         */
        std::vector<std::string> get_input_names(const std::string& graph_name);

        /**
         * @brief Get all output tensor names of the specified graph.
         *
         * @param graph_name The specified graph name
         * @return All the output tensor names of the graph
         */
        std::vector<std::string> get_output_names(const std::string& graph_name);

        /**
         * @brief Get the shape of an input tensor in a graph.
         *
         * @param graph_name  The specified graph name
         * @param tensor_name The specified tensor name
         * @return The shape of the tensor
         */
        std::vector<int> get_input_shape(const std::string& graph_name, const std::string& tensor_name);

        /**
         * @brief Get the shape of an output tensor in a graph.
         *
         * @param graph_name  The specified graph name
         * @param tensor_name The specified tensor name
         * @return The shape of the tensor
         */
        std::vector<int> get_output_shape(const std::string& graph_name, const std::string& tensor_name);

        /**
         * @brief Get scale of an input tensor. Only used for int8 models.
         *
         * @param graph_name  The specified graph name
         * @param tensor_name The specified tensor name
         * @return Scale of the input tensor
         */
        float get_input_scale(const std::string& graph_name, const std::string& tensor_name);

        /**
         * @brief Get data type of an input tensor. Refer to bmdef.h as following.
         *   typedef enum {
         *     BM_FLOAT32 = 0,
         *     BM_FLOAT16 = 1,
         *     BM_INT8 = 2,
         *     BM_UINT8 = 3,
         *     BM_INT16 = 4,
         *     BM_UINT16 = 5,
         *     BM_INT32 = 6,
         *     BM_UINT32 = 7
         *   } bm_data_type_t;
         *
         * @param graph_name  The specified graph name
         * @param tensor_name The specified tensor name
         * @return Data type of the input tensor
         */
        bm_data_type_t get_input_dtype(const std::string& graph_name, const std::string& tensor_name);

    private:
        class Perf_CC;
        class Perf_CC* const _impl;

        /**
         * @brief Forbidden copy constructor.
         * @brief Copy constructor.
         *
         * @param other An other Perf instance.
         */
        Perf(const Perf& other) = delete;

        /**
         * @brief Forbidden assignment function.
         * @brief Assignment function.
         *
         * @param other An other Perf instance.
         * @return Reference of a Perf instance.
         */
        Perf& operator=(const Perf& other) = delete;

 };

}