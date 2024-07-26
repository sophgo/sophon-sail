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

/** @file     algokit.h
 *  @brief    Header file of algo kit
 *  @author   SOPHGO
 *  @version  3.3.0
 *  @date     2019-12-27
 */

#pragma once
#ifndef TPUKERNRL_OFF

#ifdef PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#endif
#include <tensor.h>
#include <string>
#include <vector>
#include <iostream>
#include <algokit.h>
#include <map>
#include "bmlib_runtime.h"
using namespace std;

namespace sail {
typedef enum {
  SAIL_TPU_KERNEL_SUCCESS = 0,
  SAIL_TPU_KERNEL_ERROR_PARAM_MISMATCH = 60001,
  SAIL_TPU_KERNEL_ERROR_SHAPE = 60002,
  SAIL_TPU_KERNEL_ERROR_BATCH_SIZE = 60003,
} sail_tpu_kernel_status;

class DECL_EXPORT tpu_kernel_api_yolov5_detect_out{
    public:
    /**
     * @brief Constructor
     * 
     * @param device_id         Device id.
     * @param shape             Input Data shape 
     * @param network_w         Network input width 
     * @param network_h         Network input height 
     * @param module_file       TPU Kernel module file.
     */
    explicit tpu_kernel_api_yolov5_detect_out(int device_id, 
                                            const std::vector<std::vector<int>>& shapes, 
                                            int network_w=640, 
                                            int network_h=640,
                                            std::string module_file = "/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so");

    /**
     * @brief Destructor 
     */  
    ~tpu_kernel_api_yolov5_detect_out();

    /**
     * @brief Porcess
     * 
     * @param input             Input Data 
     * @param out_doxs          Detect results 
     * @param dete_threshold    Detection threshold
     * @param nms_threshold     NMS threshold
     * @param release_input     Release input memory
     * @return 0 for success and other for failure
     */
    int process(std::vector<TensorPTRWithName>& input, 
                std::vector<std::vector<DeteObjRect>> &out_doxs,
                float dete_threshold,
                float nms_threshold,
                bool release_input = false);

    /**
     * @brief Porcess
     * 
     * @param input             Input Data 
     * @param dete_threshold    Detection threshold
     * @param nms_threshold     NMS threshold
     * @param release_input     Release input memory
     * @return Detect results[left,top,right,bottom, class_id, score]
     */
    std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> process(std::vector<TensorPTRWithName>& input, float dete_threshold, float nms_threshold, bool release_input = false);
    
    std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> process(std::map<std::string, Tensor&>& input, float dete_threshold, float nms_threshold, bool release_input = false);
    
    /**
     * @brief Reset Anchor
     * @param anchors  new anchors
     * @return 0 for success and other for failure
     */
    int reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new);

    private:
    class tpu_kernel_api_yolov5_detect_out_cc;
    class tpu_kernel_api_yolov5_detect_out_cc* const _impl;
};

class DECL_EXPORT tpu_kernel_api_yolov5_out_without_decode{
    public:
    /**
     * @brief Constructor
     * 
     * @param device_id         Device id.
     * @param shape             Input Data shape 
     * @param network_w         Network input width 
     * @param network_h         Network input height 
     * @param module_file       TPU Kernel module file.
     */
    explicit tpu_kernel_api_yolov5_out_without_decode(int device_id, 
                                            const std::vector<int>& shapes, 
                                            int network_w=640, 
                                            int network_h=640,
                                            std::string module_file = "/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so");

    /**
     * @brief Destructor 
     */  
    ~tpu_kernel_api_yolov5_out_without_decode();

    /**
     * @brief Porcess
     * 
     * @param input             Input Data 
     * @param out_doxs          Detect results 
     * @param dete_threshold    Detection threshold
     * @param nms_threshold     NMS threshold
     * @return 0 for success and other for failure
     */
    int process(TensorPTRWithName& input, 
                std::vector<std::vector<DeteObjRect>> &out_doxs,
                float dete_threshold,
                float nms_threshold);

    /**
     * @brief Porcess
     * 
     * @param input             Input Data 
     * @param dete_threshold    Detection threshold
     * @param nms_threshold     NMS threshold
     * @return Detect results[left,top,right,bottom, class_id, score]
     */
    std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> process(TensorPTRWithName& input, float dete_threshold, float nms_threshold);
    
    std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>> process(std::map<std::string, Tensor&>& input, float dete_threshold, float nms_threshold);

    private:
    class tpu_kernel_api_yolov5_out_without_decode_cc;
    class tpu_kernel_api_yolov5_out_without_decode_cc* const _impl;
};

class DECL_EXPORT tpu_kernel_api_openpose_part_nms{
    public:
    /**
     * @brief Constructor
     * 
     * @param device_id         Device id.
     * @param network_c         Pose output channel 
     * @param module_file       TPU Kernel module file.
     */
    explicit tpu_kernel_api_openpose_part_nms(int device_id, 
                                            int network_c,
                                            std::string module_file = "/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so");

    /**
     * @brief Destructor 
     */  
    ~tpu_kernel_api_openpose_part_nms();

    /**
     * @brief Porcess
     * 
     * @param input_data        Input Data 
     * @param shape             Input Data width and height
     * @param num_result        Peak num results
     * @param score_out_result  Peak score results 
     * @param coor_out_result   Peak coordinates results
     * @param threshold         Detection threshold
     * @param max_peak_num      Peak Detection maxium num
     * @return 0 for success and other for failure
     */
    int process(TensorPTRWithName& input_data, 
                std::vector<int>& shape,
                std::vector<std::vector<int>>& num_result, 
                std::vector<std::vector<float>>& score_out_result,
                std::vector<std::vector<int>>& coor_out_result,
                std::vector<float>& threshold,
                std::vector<int>& max_peak_num);

    /**
     * @brief Porcess
     * 
     * @param input_data        Input Data 
     * @param shape             Input Data width and height
     * @param threshold         Detection threshold
     * @param max_peak_num      Peak Detection maxium num
     * @return Peak detect results[nums, scores, coordinates]
     */
    std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<float>>, std::vector<std::vector<int>>> process(TensorPTRWithName& input_data, std::vector<int>& shape, std::vector<float>& threshold, std::vector<int>& max_peak_num);
    
    std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<float>>, std::vector<std::vector<int>>> process(std::map<std::string, Tensor&>& input_data, std::vector<int>& shape, std::vector<float>& threshold, std::vector<int>& max_peak_num);
    
    /**
     * @brief Reset Anchor
     * @param network_c_new  new channel
     * @return 0 for success and other for failure
     */
    int reset_network_c(int network_c_new);

    private:
    class tpu_kernel_api_openpose_part_nms_cc;
    class tpu_kernel_api_openpose_part_nms_cc* const _impl;
};

}
#endif