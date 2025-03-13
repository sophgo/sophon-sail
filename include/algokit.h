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

/** @file     algokit.h
 *  @brief    Header file of algo kit
 *  @author   SOPHGO
 *  @version  3.3.0
 *  @date     2019-12-27
 */

#pragma once
#ifdef PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#endif

#ifdef USE_OPENCV
#include <opencv2/core/mat.hpp>
#endif

#include <tensor.h>
#include <string>
#include <vector>
#include <map>
#include <iostream>
using namespace std;

typedef enum {
  SAIL_ALGO_SUCCESS = 0,
  SAIL_ALGO_BUFFER_FULL = 50001,    
  SAIL_ALGO_BUFFER_EMPRT = 50002,
  SAIL_ALGO_ERROR_D2S = 50003,
  SAIL_ALGO_ERROR_BATCHSIZE = 50004,
  SAIL_ALGO_ERROR_SHAPES = 50005,
} sail_algo_status;

namespace sail {

struct DeteObjRect {
    unsigned int class_id;
    float score;
    float left;
    float top;
    float right;
    float bottom;
    float width;
    float height;
};

struct TrackObjRect : public DeteObjRect {
    struct DeteObjRect;
    unsigned int track_id;
};

template <typename T = int>
struct Point {
  T x;
  T y;
};

using Point32I = Point<int>;
using Point32F = Point<float>;
using Point64D = Point<double>;

template <typename T = int>
struct RotatedBox{
  T x_ctr; 
  T y_ctr; 
  T w;   
  T h;   
  float a;   
} ;

using RotatedBox32I = RotatedBox<int>;
using RotatedBox32F = RotatedBox<float>;
using RotatedBox64D = RotatedBox<double>;

#ifdef USE_OPENCV
class DECL_EXPORT algo_yolov5_post_1output {
    public:
    /**
     * @brief Constructor
     * 
     * @param shape             Input Data shape 
     * @param network_w         Network input width 
     * @param network_h         Network input height 
     * @param max_queue_size    Data queue max size 
     */
    explicit algo_yolov5_post_1output(const std::vector<int>& shape, int network_w=640, int network_h=640, int max_queue_size=20,  bool input_use_multiclass_nms=true, bool agnostic=false);

    /**
     * @brief Destructor 
     */  
    ~algo_yolov5_post_1output();


    /**
     * @brief Push Data
     * 
     * @param channel_idx       Channel index number of the image.
     * @param image_idx         Image index number of the image.
     * @param data              Input Data ptr
     * @param dete_threshold    Detection threshold
     * @param nms_threshold     NMS threshold
     * @param ost_w             Original image width
     * @param ost_h             Original image height
     * @param padding_left      Padding left
     * @param padding_top       Padding top
     * @param padding_width     Padding width
     * @param padding_height    Padding height
     * @return 0 for success and other for failure 
     */
#ifdef PYTHON
    int push_npy(
        int channel_idx, 
        int image_idx, 
        pybind11::array_t<float> data, 
        float dete_threshold, 
        float nms_threshold,
        int ost_w, 
        int ost_h, 
        int padding_left, 
        int padding_top, 
        int padding_width, 
        int padding_height);
#endif

    /**
     * @brief Push Data
     * 
     * @param channel_idx       Channel index number of the image.
     * @param image_idx         Image index number of the image.
     * @param input_data        Input Data
     * @param dete_threshold    Detection threshold
     * @param nms_threshold     NMS threshold
     * @param ost_w             Original image width
     * @param ost_h             Original image height
     * @param padding_attr      Padding Attribute(start_x, start_y, width, height)
     * @return 0 for success and other for failure 
     */
    int push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        TensorPTRWithName input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr);
    int push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        TensorPTRWithName input_data, 
        std::vector<std::vector<float>> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr);    
    /**
     * @brief Get Result
     * 
     * @return Detect results, channle index, image index
     */
    std::tuple<std::vector<DeteObjRect>,int,int> get_result();

    /**
     * @brief Get Result with numpy array
     * 
     * @return Detect results[left,top,right,bottom, class_id, score], channle index, image index
     */
    std::tuple<std::vector<std::tuple<int, int, int, int ,int, float>>,int,int> get_result_npy();

    private:
    class algo_yolov5_post_1output_cc;
    class algo_yolov5_post_1output_cc* const _impl;

    /**
     * @brief Forbidden copy constructor.
     * @brief Copy constructor.
     *
     * @param other An other algo_yolov5_post_1output instance.
     */
    algo_yolov5_post_1output(const algo_yolov5_post_1output& other) = delete;

    /**
     * @brief Forbidden assignment function.
     * @brief Assignment function.
     *
     * @param other An other algo_yolov5_post_1output instance.
     * @return Reference of a algo_yolov5_post_1output instance.
     */
    algo_yolov5_post_1output& operator=(const algo_yolov5_post_1output& other) = delete;
};

class DECL_EXPORT algo_yolov8_post_1output_async {
    public:
    /**
     * @brief Constructor
     * 
     * @param shape             Input Data shape 
     * @param network_w         Network input width 
     * @param network_h         Network input height 
     * @param max_queue_size    Data queue max size 
     */
    explicit algo_yolov8_post_1output_async(const std::vector<int>& shape, int network_w=640, int network_h=640, int max_queue_size=20,  bool input_use_multiclass_nms=true, bool agnostic=false);

    /**
     * @brief Destructor 
     */  
    ~algo_yolov8_post_1output_async();


    /**
     * @brief Push Data
     * 
     * @param channel_idx       Channel index number of the image.
     * @param image_idx         Image index number of the image.
     * @param data              Input Data ptr
     * @param dete_threshold    Detection threshold
     * @param nms_threshold     NMS threshold
     * @param ost_w             Original image width
     * @param ost_h             Original image height
     * @param padding_left      Padding left
     * @param padding_top       Padding top
     * @param padding_width     Padding width
     * @param padding_height    Padding height
     * @return 0 for success and other for failure 
     */
#ifdef PYTHON
    int push_npy(
        int channel_idx, 
        int image_idx, 
        pybind11::array_t<float> data, 
        float dete_threshold, 
        float nms_threshold,
        int ost_w, 
        int ost_h, 
        int padding_left, 
        int padding_top, 
        int padding_width, 
        int padding_height);
#endif

    /**
     * @brief Push Data
     * 
     * @param channel_idx       Channel index number of the image.
     * @param image_idx         Image index number of the image.
     * @param input_data        Input Data
     * @param dete_threshold    Detection threshold
     * @param nms_threshold     NMS threshold
     * @param ost_w             Original image width
     * @param ost_h             Original image height
     * @param padding_attr      Padding Attribute(start_x, start_y, width, height)
     * @return 0 for success and other for failure 
     */
    int push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        TensorPTRWithName input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr);
        
    /**
     * @brief Get Result
     * 
     * @return Detect results, channle index, image index
     */
    std::tuple<std::vector<DeteObjRect>,int,int> get_result();

    /**
     * @brief Get Result with numpy array
     * 
     * @return Detect results[left,top,right,bottom, class_id, score], channle index, image index
     */
    std::tuple<std::vector<std::tuple<int, int, int, int ,int, float>>,int,int> get_result_npy();

    private:
    class algo_yolov8_post_1output_async_cc;
    class algo_yolov8_post_1output_async_cc* const _impl;

    /**
     * @brief Forbidden copy constructor.
     * @brief Copy constructor.
     *
     * @param other An other algo_yolov8_post_1output_async instance.
     */
    algo_yolov8_post_1output_async(const algo_yolov8_post_1output_async& other) = delete;

    /**
     * @brief Forbidden assignment function.
     * @brief Assignment function.
     *
     * @param other An other algo_yolov8_post_1output_async instance.
     * @return Reference of a algo_yolov8_post_1output_async instance.
     */
    algo_yolov8_post_1output_async& operator=(const algo_yolov8_post_1output_async& other) = delete;
};

class DECL_EXPORT algo_yolov8_post_cpu_opt_1output_async {
    public:
    /**
     * @brief Constructor
     * 
     * @param shape             Input Data shape 
     * @param network_w         Network input width 
     * @param network_h         Network input height 
     * @param max_queue_size    Data queue max size 
     */
    explicit algo_yolov8_post_cpu_opt_1output_async(const std::vector<int>& shape, int network_w=640, int network_h=640, int max_queue_size=20,  bool input_use_multiclass_nms=true, bool agnostic=false);

    /**
     * @brief Destructor 
     */  
    ~algo_yolov8_post_cpu_opt_1output_async();


    /**
     * @brief Push Data
     * 
     * @param channel_idx       Channel index number of the image.
     * @param image_idx         Image index number of the image.
     * @param data              Input Data ptr
     * @param dete_threshold    Detection threshold
     * @param nms_threshold     NMS threshold
     * @param ost_w             Original image width
     * @param ost_h             Original image height
     * @param padding_left      Padding left
     * @param padding_top       Padding top
     * @param padding_width     Padding width
     * @param padding_height    Padding height
     * @return 0 for success and other for failure 
     */
#ifdef PYTHON
    int push_npy(
        int channel_idx, 
        int image_idx, 
        pybind11::array_t<float> data, 
        float dete_threshold, 
        float nms_threshold,
        int ost_w, 
        int ost_h, 
        int padding_left, 
        int padding_top, 
        int padding_width, 
        int padding_height);
#endif

    /**
     * @brief Push Data
     * 
     * @param channel_idx       Channel index number of the image.
     * @param image_idx         Image index number of the image.
     * @param input_data        Input Data
     * @param dete_threshold    Detection threshold
     * @param nms_threshold     NMS threshold
     * @param ost_w             Original image width
     * @param ost_h             Original image height
     * @param padding_attr      Padding Attribute(start_x, start_y, width, height)
     * @return 0 for success and other for failure 
     */
    int push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        TensorPTRWithName input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr);
        
    /**
     * @brief Get Result
     * 
     * @return Detect results, channle index, image index
     */
    std::tuple<std::vector<DeteObjRect>,int,int> get_result();

    /**
     * @brief Get Result with numpy array
     * 
     * @return Detect results[left,top,right,bottom, class_id, score], channle index, image index
     */
    std::tuple<std::vector<std::tuple<int, int, int, int ,int, float>>,int,int> get_result_npy();

    private:
    class algo_yolov8_post_cpu_opt_1output_async_cc;
    class algo_yolov8_post_cpu_opt_1output_async_cc* const _impl;

    /**
     * @brief Forbidden copy constructor.
     * @brief Copy constructor.
     *
     * @param other An other algo_yolov8_post_cpu_opt_1output_async instance.
     */
    algo_yolov8_post_cpu_opt_1output_async(const algo_yolov8_post_cpu_opt_1output_async& other) = delete;

    /**
     * @brief Forbidden assignment function.
     * @brief Assignment function.
     *
     * @param other An other algo_yolov8_post_cpu_opt_1output_async instance.
     * @return Reference of a algo_yolov8_post_cpu_opt_1output_async instance.
     */
    algo_yolov8_post_cpu_opt_1output_async& operator=(const algo_yolov8_post_cpu_opt_1output_async& other) = delete;
};

// algo_yolov5_post_3output
class DECL_EXPORT algo_yolov5_post_3output {
    public:
    /**
     * @brief Constructor
     * 
     * @param shape             Input Data shape 
     * @param network_w         Network input width 
     * @param network_h         Network input height 
     * @param max_queue_size    Data queue max size 
     */
    explicit algo_yolov5_post_3output(const std::vector<std::vector<int>>& shape, int network_w=640, int network_h=640, int max_queue_size=20, bool input_use_multiclass_nms=true, bool agnostic=false);

    /**
     * @brief Destructor 
     */  
    ~algo_yolov5_post_3output();

    /**
     * @brief Push Data
     * 
     * @param channel_idx       Channel index number of the image.
     * @param image_idx         Image index number of the image.
     * @param input_data        Input Data
     * @param dete_threshold    Detection threshold
     * @param nms_threshold     NMS threshold
     * @param ost_w             Original image width
     * @param ost_h             Original image height
     * @param padding_attr      Padding Attribute(start_x, start_y, width, height)
     * @return 0 for success and other for failure 
     */
    int push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        std::vector<TensorPTRWithName> input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr);
    int push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        std::vector<TensorPTRWithName> input_data, 
        std::vector<std::vector<float>> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr);
            
    /**
     * @brief Get Result
     * 
     * @return Detect results, channle index, image index
     */
    std::tuple<std::vector<DeteObjRect>,int,int> get_result();

    /**
     * @brief Get Result with numpy array
     * 
     * @return Detect results[left,top,right,bottom, class_id, score], channle index, image index
     */
    std::tuple<std::vector<std::tuple<int, int, int, int ,int, float>>,int,int> get_result_npy();

    /**
     * @brief Reset Anchor
     * @param anchors  new anchors
     * @return 0 for success and other for failure
     */
    int reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new);

    private:
    class algo_yolov5_post_3output_cc;
    class algo_yolov5_post_3output_cc* const _impl;

    /**
     * @brief Forbidden copy constructor.
     * @brief Copy constructor.
     *
     * @param other An other algo_yolov5_post_3output instance.
     */
    algo_yolov5_post_3output(const algo_yolov5_post_3output& other) = delete;

    /**
     * @brief Forbidden assignment function.
     * @brief Assignment function.
     *
     * @param other An other algo_yolov5_post_3output instance.
     * @return Reference of a algo_yolov5_post_3output instance.
     */
    algo_yolov5_post_3output& operator=(const algo_yolov5_post_3output& other) = delete;
};

// algo_yolov5_post_cpu_opt_async
class DECL_EXPORT algo_yolov5_post_cpu_opt_async {
    public:
    /**
     * @brief Constructor
     * 
     * @param shape             Input Data shape 
     * @param network_w         Network input width 
     * @param network_h         Network input height 
     * @param max_queue_size    Data queue max size 
* @param use_multiclass_nms whether to use multi-class NMS
     */
    explicit algo_yolov5_post_cpu_opt_async(const std::vector<std::vector<int>>& shape, int network_w=640, int network_h=640, int max_queue_size=20, bool use_multiclass_nms=true);

    /**
     * @brief Destructor 
     */  
    ~algo_yolov5_post_cpu_opt_async();

    /**
     * @brief Push Data
     * 
     * @param channel_idx       Channel index number of the image.
     * @param image_idx         Image index number of the image.
     * @param input_data        Input Data
     * @param dete_threshold    Detection threshold
     * @param nms_threshold     NMS threshold
     * @param ost_w             Original image width
     * @param ost_h             Original image height
     * @param padding_attr      Padding Attribute(start_x, start_y, width, height)
     * @return 0 for success and other for failure 
     */
    int push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        std::vector<TensorPTRWithName> input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr);
    int push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        std::vector<TensorPTRWithName> input_data, 
        std::vector<std::vector<float>> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr);    
    /**
     * @brief Get Result
     * 
     * @return Detect results, channle index, image index
     */
    std::tuple<std::vector<DeteObjRect>,int,int> get_result();

    /**
     * @brief Get Result with numpy array
     * 
     * @return Detect results[left,top,right,bottom, class_id, score], channle index, image index
     */
    std::tuple<std::vector<std::tuple<int, int, int, int ,int, float>>,int,int> get_result_npy();

    /**
     * @brief Reset Anchor
     * @param anchors  new anchors
     * @return 0 for success and other for failure
     */
    int reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new);

    private:
    class algo_yolov5_post_cpu_opt_async_cc;
    class algo_yolov5_post_cpu_opt_async_cc* const _impl;

    /**
     * @brief Forbidden copy constructor.
     * @brief Copy constructor.
     *
     * @param other An other algo_yolov5_post_cpu_opt_async instance.
     */
    algo_yolov5_post_cpu_opt_async(const algo_yolov5_post_cpu_opt_async& other) = delete;

    /**
     * @brief Forbidden assignment function.
     * @brief Assignment function.
     *
     * @param other An other algo_yolov5_post_cpu_opt_async instance.
     * @return Reference of a algo_yolov5_post_cpu_opt_async instance.
     */
    algo_yolov5_post_cpu_opt_async& operator=(const algo_yolov5_post_cpu_opt_async& other) = delete;
};

// YOLOX后处理接口
// algo_yolox_post
class DECL_EXPORT algo_yolox_post {
    public:
    /**
     * @brief Constructor
     * 
     * @param shape             Input Data shape 
     * @param network_w         Network input width 
     * @param network_h         Network input height 
     * @param max_queue_size    Data queue max size 
     */
    explicit algo_yolox_post(const std::vector<int>& shape, int network_w=640, int network_h=640, int max_queue_size=20);

    /**
     * @brief Destructor 
     */  
    ~algo_yolox_post();


    /**
     * @brief Push Data
     * 
     * @param channel_idx       Channel index number of the image.
     * @param image_idx         Image index number of the image.
     * @param data              Input Data ptr
     * @param dete_threshold    Detection threshold
     * @param nms_threshold     NMS threshold
     * @param ost_w             Original image width
     * @param ost_h             Original image height
     * @param padding_left      Padding left
     * @param padding_top       Padding top
     * @param padding_width     Padding width
     * @param padding_height    Padding height
     * @return 0 for success and other for failure 
     */
#ifdef PYTHON
    int push_npy(
        int channel_idx, 
        int image_idx, 
        pybind11::array_t<float> data, 
        float dete_threshold, 
        float nms_threshold,
        int ost_w, 
        int ost_h, 
        int padding_left, 
        int padding_top, 
        int padding_width, 
        int padding_height);
#endif

    /**
     * @brief Push Data
     * 
     * @param channel_idx       Channel index number of the image.
     * @param image_idx         Image index number of the image.
     * @param input_data        Input Data
     * @param dete_threshold    Detection threshold
     * @param nms_threshold     NMS threshold
     * @param ost_w             Original image width
     * @param ost_h             Original image height
     * @param padding_attr      Padding Attribute(start_x, start_y, width, height)
     * @return 0 for success and other for failure 
     */
    int push_data(
        std::vector<int> channel_idx, 
        std::vector<int> image_idx, 
        TensorPTRWithName input_data, 
        std::vector<float> dete_threshold,
        std::vector<float> nms_threshold,
        std::vector<int> ost_w,
        std::vector<int> ost_h,
        std::vector<std::vector<int>> padding_attr);
        
    /**
     * @brief Get Result
     * 
     * @return Detect results, channle index, image index
     */
    std::tuple<std::vector<DeteObjRect>,int,int> get_result();

    /**
     * @brief Get Result with numpy array
     * 
     * @return Detect results[left,top,right,bottom, class_id, score], channle index, image index
     */
    std::tuple<std::vector<std::tuple<int, int, int, int ,int, float>>,int,int> get_result_npy();

    private:
    class algo_yolox_post_cc;
    class algo_yolox_post_cc* const _impl;

    /**
     * @brief Forbidden copy constructor.
     * @brief Copy constructor.
     *
     * @param other An other algo_yolox_post instance.
     */
    algo_yolox_post(const algo_yolox_post& other) = delete;

    /**
     * @brief Forbidden assignment function.
     * @brief Assignment function.
     *
     * @param other An other algo_yolox_post instance.
     * @return Reference of a algo_yolox_post instance.
     */
    algo_yolox_post& operator=(const algo_yolox_post& other) = delete;
};

// algo_yolov5_post_cpu_opt
class DECL_EXPORT algo_yolov5_post_cpu_opt {
    public:
    /**
     * @brief Constructor
     * 
     * @param shape             Input Data shape 
     * @param network_w         Network input width 
     * @param network_h         Network input height 
     */
    explicit algo_yolov5_post_cpu_opt(const std::vector<std::vector<int>>& shape, int network_w=640, int network_h=640);

    /**
     * @brief Destructor 
     */  
    ~algo_yolov5_post_cpu_opt();

    /**
     * @brief Process
     * 
     * @param input_data                    Input Data
     * @param ost_w                         Original image width
     * @param ost_h                         Original image height
     * @param out_doxs                      Detect results 
     * @param dete_threshold                Detection threshold
     * @param nms_threshold                 NMS threshold
     * @param input_keep_aspect_ratio       Input keeping aspect ratio  
     * @param input_use_multiclass_nms      Input with multiclass
     * @return 0 for success and other for failure 
     */
    int process(
        std::vector<TensorPTRWithName> &input_data, 
        std::vector<int> &ost_w,
        std::vector<int> &ost_h,
        std::vector<std::vector<DeteObjRect>> &out_doxs,
        std::vector<float> &dete_threshold,
        std::vector<float> &nms_threshold,
        bool input_keep_aspect_ratio,
        bool input_use_multiclass_nms);
    int process(
        std::vector<TensorPTRWithName> &input_data, 
        std::vector<int> &ost_w,
        std::vector<int> &ost_h,
        std::vector<std::vector<DeteObjRect>> &out_doxs,
        std::vector<std::vector<float>> &dete_threshold,
        std::vector<float> &nms_threshold,
        bool input_keep_aspect_ratio,
        bool input_use_multiclass_nms);    
    /**
     * @brief Porcess
     * 
     * @param input_data                    Input Data
     * @param ost_w                         Original image width
     * @param ost_h                         Original image height
     * @param dete_threshold                Detection threshold
     * @param nms_threshold                 NMS threshold
     * @param input_keep_aspect_ratio       Input keeping aspect ratio  
     * @param input_use_multiclass_nms      Input with multiclass
     * @return Detect results[left,top,right,bottom, class_id, score]
     */
    std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>>
         process(std::vector<TensorPTRWithName> &input_data, 
                std::vector<int> &ost_w,
                std::vector<int> &ost_h,
                std::vector<float> &dete_threshold,
                std::vector<float> &nms_threshold,
                bool input_keep_aspect_ratio,
                bool input_use_multiclass_nms);
    std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>>
         process(std::vector<TensorPTRWithName> &input_data, 
                std::vector<int> &ost_w,
                std::vector<int> &ost_h,
                std::vector<std::vector<float>> &dete_threshold,
                std::vector<float> &nms_threshold,
                bool input_keep_aspect_ratio,
                bool input_use_multiclass_nms);
    std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>>
         process(std::map<std::string, Tensor&>& input_data,
                std::vector<int> &ost_w,
                std::vector<int> &ost_h,
                std::vector<float> &dete_threshold,
                std::vector<float> &nms_threshold,
                bool input_keep_aspect_ratio,
                bool input_use_multiclass_nms);
    std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>>
         process(std::map<std::string, Tensor&>& input_data,
                std::vector<int> &ost_w,
                std::vector<int> &ost_h,
                std::vector<std::vector<float>> &dete_threshold,
                std::vector<float> &nms_threshold,
                bool input_keep_aspect_ratio,
                bool input_use_multiclass_nms);

    /**
     * @brief Reset Anchor
     * @param anchors  new anchors
     * @return 0 for success and other for failure
     */
    int reset_anchors(std::vector<std::vector<std::vector<int>>> anchors_new);

    private:
    class algo_yolov5_post_cpu_opt_cc;
    class algo_yolov5_post_cpu_opt_cc* const _impl;

    /**
     * @brief Forbidden copy constructor.
     * @brief Copy constructor.
     *
     * @param other An other algo_yolov5_post_cpu_opt instance.
     */
    algo_yolov5_post_cpu_opt(const algo_yolov5_post_cpu_opt& other) = delete;

    /**
     * @brief Forbidden assignment function.
     * @brief Assignment function.
     *
     * @param other An other algo_yolov5_post_cpu_opt instance.
     * @return Reference of a algo_yolov5_post_cpu_opt instance.
     */
    algo_yolov5_post_cpu_opt& operator=(const algo_yolov5_post_cpu_opt& other) = delete;
};

// algo_yolov8_seg_post_tpu_opt
struct YoloV8Box {
    float left;
    float top;
    float right;
    float bottom;
    float score;
    unsigned int class_id;
    std::vector<float> mask;
    std::vector<float> contour;
    cv::Mat mask_img;
};

class DECL_EXPORT algo_yolov8_seg_post_tpu_opt {
    public:
    /**
     * @brief Constructor
     * 
     * @param bmodel_file             TPU getmask bmodel 
     * @param dev_id                  device id 
     * @param detection_shape         The shapes of detection head
     * @param segmentation_shape      The shapes of segmentation head, that is Prototype Mask
     * @param network_h               The input height of yolov8 network
     * @param network_w               The input width of yolov8 network
     */
    explicit algo_yolov8_seg_post_tpu_opt(string bmodel_file, int dev_id, const vector<int>& detection_shape, const vector<int>& segmentation_shape, int network_h, int network_w);

     /**
     * @brief Destructor 
     */ 
    ~algo_yolov8_seg_post_tpu_opt();

    /**
     * @brief Process (single batch)
     * 
     * @param detection_input               The input data of detection head
     * @param segmentation_input            The input data of segmentation head, that is Prototype Mask
     * @param ost_h                         Original image height
     * @param ost_w                         Original image width 
     * @param dete_threshold                Detection threshold
     * @param nms_threshold                 NMS threshold
     * @param input_keep_aspect_ratio       Input keeping aspect ratio  
     * @param input_use_multiclass_nms      Input with multiclass
     * @return                              Segmentiont results[left, top, right, bottom, score, class_id, contour, mask] 
     */
    int process(TensorPTRWithName &detection_input,
                TensorPTRWithName &segmentation_input,
                int &ost_h,
                int &ost_w,
                vector<YoloV8Box> &yolov8_results,
                float &dete_threshold,
                float &nms_threshold,
                bool input_keep_aspect_ratio,
                bool input_use_multiclass_nms);
 
#ifdef PYTHON
    pybind11::list process(map<string, Tensor&> &detection_input,
                            map<string, Tensor&> &segmentation_input,
                            int &ost_h,
                            int &ost_w,
                            float &dete_threshold,
                            float &nms_threshold,
                            bool input_keep_aspect_ratio,
                            bool input_use_multiclass_nms);
    
    pybind11::list process(TensorPTRWithName &detection_input,
                            TensorPTRWithName &segmentation_input,
                            int &ost_h,
                            int &ost_w,
                            float &dete_threshold,
                            float &nms_threshold,
                            bool input_keep_aspect_ratio,
                            bool input_use_multiclass_nms);
#endif
    private:
    class algo_yolov8_seg_post_tpu_opt_cc;
    class algo_yolov8_seg_post_tpu_opt_cc* const _impl;

    /**
     * @brief Forbidden copy constructor.
     * @brief Copy constructor.
     *
     * @param other An other algo_yolov8_seg_post_tpu_opt instance.
     */
    algo_yolov8_seg_post_tpu_opt(const algo_yolov8_seg_post_tpu_opt& other) = delete;
    
    /**
     * @brief Forbidden assignment function.
     * @brief Assignment function.
     *
     * @param other An other algo_yolov8_seg_post_tpu_opt instance.
     * @return Reference of a algo_yolov8_seg_post_tpu_opt instance.
     */
    algo_yolov8_seg_post_tpu_opt& operator=(const algo_yolov8_seg_post_tpu_opt& other) = delete;
};

/*下面的内容都是跟目标跟踪有关*/
class DECL_EXPORT deepsort_tracker_controller {
    public:
    explicit deepsort_tracker_controller(float max_cosine_distance, int nn_budget, int k_feature_dim, float max_iou_distance = 0.7, int max_age = 30, int n_init = 3);
    ~deepsort_tracker_controller();
    int process(const vector<DeteObjRect>& detected_objects, vector<Tensor>& feature, vector<TrackObjRect>& tracked_objects);
    int process(const vector<DeteObjRect>& detected_objects, vector<vector<float>>& feature, vector<TrackObjRect>& tracked_objects);
    std::vector<std::tuple<int, int, int, int, int, float, int>> process(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short, vector<Tensor>& feature);
    std::vector<std::tuple<int, int, int, int, int, float, int>> process(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short, vector<vector<float>>& feature);

    private:
    class deepsort_tracker_controller_cc;
    class deepsort_tracker_controller_cc* const _impl;
};

// deepsort 异步处理接口
class DECL_EXPORT deepsort_tracker_controller_async{
    public:
    explicit deepsort_tracker_controller_async(float max_cosine_distance, int nn_budget, int k_feature_dim, float max_iou_distance = 0.7, int max_age = 30, int n_init = 3, int queue_size = 10);
    ~deepsort_tracker_controller_async();
     // asynchronous interface
    int push_data(const vector<DeteObjRect>& detected_objects, vector<Tensor>& feature);
    int push_data(const vector<DeteObjRect>& detected_objects, vector<vector<float>>& feature);
    std::vector<TrackObjRect> get_result();
    
    // python function
    int push_data(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short, vector<Tensor>& feature);
    int push_data(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short, vector<vector<float>>& feature);
    std::vector<std::tuple<int, int, int, int, int, float, int>> get_result_npy(); 
    void set_processing_timer(bool flag);

    private:
    class deepsort_tracker_controller_async_cc;
    class deepsort_tracker_controller_async_cc* const _impl;
};

class DECL_EXPORT bytetrack_tracker_controller {
    public:
    explicit bytetrack_tracker_controller(int frame_rate = 30, int track_buffer = 30);
    ~bytetrack_tracker_controller();
    int process(const vector<DeteObjRect>& detected_objects, vector<TrackObjRect>& tracked_objects);
    std::vector<std::tuple<int, int, int, int, int, float, int>> process(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short);
    
    private:
    class bytetrack_tracker_controller_cc;
    class bytetrack_tracker_controller_cc* const _impl;
};

// sort 跟踪算法接口
class DECL_EXPORT sort_tracker_controller{
    public:
        explicit sort_tracker_controller(float max_iou_distance = 0.7, int max_age = 30, int n_init = 3);
        ~sort_tracker_controller();
        // int process(const vector<DeteObjRect>& detected_objects, vector<TrackObjRect>& tracked_objects);
        std::vector<std::tuple<int, int, int, int, int, float, int>> process(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short);
    private:
        class sort_tracker_controller_cc;
        class sort_tracker_controller_cc* _impl;
};

// SORT算法异步接口
class DECL_EXPORT sort_tracker_controller_async{
    public:
    explicit sort_tracker_controller_async(float max_iou_distance = 0.7, int max_age = 30, int n_init = 3, int input_queue_size = 10, int result_queue_size = 10);
    ~sort_tracker_controller_async();
    
    // python function
    int push_data(const std::vector<std::tuple<int, int, int, int ,int, float>>& detected_objects_short);
    std::vector<std::tuple<int, int, int, int, int, float, int>> get_result_npy(); 

    private:
    class sort_tracker_controller_async_cc;
    class sort_tracker_controller_async_cc* const _impl;
};

#endif

// rotated nms
std::vector<int> nms_rotated(std::vector<std::vector<float>>& boxes, std::vector<float>& scores, float threshold);


}// namespace sail

