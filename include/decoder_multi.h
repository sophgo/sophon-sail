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

/** @file     engine_multi.h
 *  @brief    Header file of MultiEngine
 *  @author   SOPHGO
 *  @version  3.2.0
 *  @date     2022-10-21
 */

#pragma once
#include "cvwrapper.h"
#include <deque>
#include <mutex>
#include <condition_variable>
#include <map>
#include <iostream>
#include <string>

using namespace std;
namespace sail{
#if defined(USE_BMCV) && defined(USE_OPENCV) && defined(USE_FFMPEG)
    /**
     * @brief MultiDecoder
     * 
    */
    class DECL_EXPORT MultiDecoder{
    public:
        /**
         * @brief Constructor
         * 
         * @param queue_size    Max data queue size.
         * @param tpu_id        ID of TPU, there may be more than one TPU for PCIE mode.
         * @param discard_mode  Data discard policy when the queue is full. If 0, do not push the data to queue, else pop the data from queue and push new data to queue.
        */
        explicit MultiDecoder(int queue_size=10, int tpu_id=0, int discard_mode=0);

        /**
         * @brief Destructor 
        */    
        ~MultiDecoder();

        /**
         * @brief Set read frame timeout waiting time 
         * 
         * @param time_second Set read frame timeout waiting time in seconds
         */
        void set_read_timeout(int time_second);

        /**
         * @brief Add a channel to decode
         * 
         * @param file_path         Path or rtsp url to the video/image file.
         * @param frame_skip_num    Frame skip number.
         * @return  return channel index number.
         */
        int add_channel(
            const std::string&  file_path, 
            int                 frame_skip_num=0);

        /**
         * @brief Delete channel
         * 
         * @param channel_idx Channel index number.
         * @return 0 for success and other for failure. 
         */
        int del_channel(int channel_idx);

        /**
         * @brief Clear data cache queue
         * 
         * @param channel_idx Channel index number.
         * @return 0 for success and other for failure  
         */
        int clear_queue(int channel_idx);

        /**
         * @brief Get the fps of the video stream in a specified channel
         * 
         * @param channel_idx Channel index number.
         * @return Returns the fps of the video stream in the specified channel 
         */
        float get_channel_fps(int channel_idx);

        /**
         * @brief Read a BMImage from the MultiDecoder with a given channel.
         * 
         * @param channel_idx   Channel index number.
         * @param image         Reference of BMImage to be read to   
         * @param read_mode     Read data mode, 0 for not waiting data and other waiting data.
         * @return 0 for successed get data. 
         */
        int read(
            int         channel_idx,
            BMImage&    image,
            int         read_mode=0);

        /**
         * @brief Read a BMImage from the MultiDecoder with a given channel.
         * 
         * @param channel_idx Channel index number.
         * @return BMImage instance to be read to  
         */
        BMImage read(int channel_idx);

        /**
         * @brief Read a bm_image from the MultiDecoder with a given channel.
         * 
         * @param channel_idx   Channel index number.
         * @param image         Reference of bm_image to be read to 
         * @param read_mode     Read data mode, 0 for not waiting data and other waiting data.
         * @return 0 for successed get data.  
         */
        int read_(
            int         channel_idx,
            bm_image&   image,
            int         read_mode=0);
        
        /**
         * @brief Read a bm_image from the MultiDecoder with a given channel.
         * 
         * @param channel_idx  Channel index number.
         * @return bm_image instance to be read to.
         */
        bm_image read_(int channel_idx);

        /**
         * @brief Reconnect Decoder for instance channel.
         * 
         * @param channel_idx Channel index number.
         * @return 0 for success and other for failure. 
         */
        int reconnect(int channel_idx);

        /**
         * @brief Get frame shape for instance channel
         * 
         * @param channel_idx Channel index number.
         * @return Frame shape, [1, C, H, W]
         */
        std::vector<int> get_frame_shape(int channel_idx);

        /**
         * @brief Set local video flag
         * 
         * @param flag  If flag is True, Decode up to 25 frames per second
         */
        void set_local_flag(bool flag);

        /**
         * @brief Get drop num for instance channel.
         * 
         * @param channel_idx Channel index number.
         * @return num for instance channel. 
         */
        size_t get_drop_num(int channel_idx);
    
        /**
         * @brief Set drop num init 0 for instance channel.
         * 
         * @param channel_idx Channel index number. 
         */
        void reset_drop_num(int channel_idx);

        /**
         * @brief Get status of the specific channel.
         * 
         * @param channel_idx Channel index.
         * @return status of this channel. 0 for open and else for close 
         */
        DecoderStatus get_channel_status(int channel_idx);

    private:
        class MultiDecoder_CC;
        class MultiDecoder_CC* const _impl;

        /**
         * @brief Forbidden copy constructor.
         * @brief Copy constructor.
         *
         * @param other An other MultiDecoder instance.
         */
        MultiDecoder(const MultiDecoder& other) = delete;

        /**
         * @brief Forbidden assignment function.
         * @brief Assignment function.
         *
         * @param other An other MultiDecoder instance.
         * @return Reference of a MultiDecoder instance.
         */
        MultiDecoder& operator=(const MultiDecoder& other) = delete;
    };

    typedef enum sail_resize_type {
        BM_RESIZE_VPP_NEAREST = 0,
        BM_RESIZE_TPU_NEAREST = 1,
        BM_RESIZE_TPU_LINEAR = 2,
        BM_RESIZE_TPU_BICUBIC = 3,
        BM_PADDING_VPP_NEAREST = 4,
        BM_PADDING_TPU_NEAREST = 5,
        BM_PADDING_TPU_LINEAR = 6,
        BM_PADDING_TPU_BICUBIC = 7,
    };

    class DECL_EXPORT ImagePreProcess{
    public:
        /**
         * @brief Constructor
         * 
         * @param batch_size        Output batch size
         * @param resize_mode       Resize Methods 
         * @param tpu_id            ID of TPU, there may be more than one TPU for PCIE mode,default is 0.
         * @param queue_in_size     Max input image data queue size, default is 20.
         * @param queue_out_size    Max output tensor data queue size, default is 20.
         * @param use_mat_flag      Use cv Mat for output, default is false.
        */
        explicit ImagePreProcess(
            int batch_size,
            sail_resize_type resize_mode,
            int tpu_id=0, 
            int queue_in_size=20, 
            int queue_out_size=20,
            bool use_mat_flag=false);

        /**
         * @brief Destructor
         */
        ~ImagePreProcess();

        /**
         * @brief Set the Resize Image attribute
         * 
         * @param output_width  The width of resized image.
         * @param output_height The height of resized image.
         * @param bgr2rgb       The flag of convert BGR image to RGB.
         * @param dtype         The data type of resized image,Only supported BM_FLOAT32,BM_INT8,BM_UINT8     
         */
        void SetResizeImageAtrr(			    
            int output_width,				    
            int output_height,				    
            bool bgr2rgb,					    
            bm_image_data_format_ext  dtype);	


        /**
         * @brief Set the padding attribute object
         * 
         * @param padding_b padding value of b channel, dafault 114
         * @param padding_g padding value of g channel, dafault 114
         * @param padding_r padding value of r channel, dafault 114
         * @param align     padding position, default 0: start left top, 1 for center
         * @return padding position, {start point x,start point y, resize width, resize height}
         */
        void SetPaddingAtrr(		    
            int padding_b=114,		        
            int padding_g=114,		        
            int padding_r=114,		        
            int align=0);		            

        /**
         * @brief Set the linear transformation attribute.
         * 
         * @param alpha_beta (a0, b0), (a1, b1), (a2, b2) factors
         * @return 0 for success and other for failure 
         */
        int SetConvertAtrr(
            const std::tuple<
                std::pair<float, float>,
                std::pair<float, float>,
                std::pair<float, float>> &alpha_beta);

        /**
         * @brief Push Image
         * 
         * @param channel_idx   Channel index number of the image.
         * @param image_idx     Image index number of the image.
         * @param image         Input image
         * @return 0 for success and other for failure 
         */
        int PushImage(
            int channel_idx, 
            int image_idx, 
            BMImage &image);

        /**
         * @brief Get the Batch Data object
         * 
         * @return std::tuple<sail::Tensor,      Output Tensor map.
         * std::vector<BMImage>,                 Original Images
         * std::vector<int>,                     Original Channel index
         * std::vector<int>>                     Original Index
         * std::vector<std::vector<int>>>        Padding Attribute(start_x, start_y, width, height)
         */
        std::tuple<sail::Tensor, 
            std::vector<BMImage>,
            std::vector<int>,
            std::vector<int>,
            std::vector<std::vector<int>>> GetBatchData();

        /**
         * @brief Get the Batch Data object
         * 
         * @return std::tuple<sail::Tensor,      Output Tensor map.
         * std::vector<cv::Mat>,                 Original Images
         * std::vector<int>,                     Original Channel index
         * std::vector<int>>                     Original Index
         * std::vector<std::vector<int>>>        Padding Atrr(start_x, start_y, width, height)
         */
        std::tuple<sail::Tensor, 
            std::vector<cv::Mat>,
            std::vector<int>,
            std::vector<int>,
            std::vector<std::vector<int>>> GetBatchData_CV();

        /** @brief Print main process time use.
         *
         *  @param print_flag.
         */ 
        void set_print_flag(bool print_flag); 
        void exhausted();
        bool get_exhausted_flag();
        void stop_thread();

    private:
        class ImagePreProcess_CC;
        class ImagePreProcess_CC* const _impl;

        /**
         * @brief Forbidden copy constructor.
         * @brief Copy constructor.
         *
         * @param other An other ImagePreProcess instance.
         */
        ImagePreProcess(const ImagePreProcess& other) = delete;

        /**
         * @brief Forbidden assignment function.
         * @brief Assignment function.
         *
         * @param other An other ImagePreProcess instance.
         * @return Reference of a ImagePreProcess instance.
         */
        ImagePreProcess& operator=(const ImagePreProcess& other) = delete;
    };

    class EngineImagePreProcess{
    public:
        /**
         * @brief Constructor
         * 
         * @param bmodel_path        Path to bmodel
         * @param tpu_id             ID of TPU, there may be more than one TPU for PCIE mode.
         * @param use_mat_output     Use cv::Mat for output.
         * @param core_list          Core list of choosed TPU.
        */
        EngineImagePreProcess(const std::string& bmodel_path, int tpu_id, bool use_mat_output=false, std::vector<int> core_list={});
        ~EngineImagePreProcess();

         /**
         * @brief initialize ImagePreProcess
         * 
         * @param resize_mode       Resize Methods 
         * @param bgr2rgb           The flag of convert BGR image to RGB, default is false
         * @param queue_in_size     Max input image data queue size, default is 20.
         * @param queue_out_size    Max output tensor data queue size, default is 20.
         * @return 0 for success and other for failure 
        */
        int InitImagePreProcess(
            sail_resize_type resize_mode,
            bool bgr2rgb=false,					    
            int queue_in_size=20, 
            int queue_out_size=20);


        /**
         * @brief Set the padding attribute object
         * 
         * @param padding_b padding value of b channel, dafault 114
         * @param padding_g padding value of g channel, dafault 114
         * @param padding_r padding value of r channel, dafault 114
         * @param align     padding position, default 0: start left top, 1 for center
         * @return 0 for success and other for failure 
         */
        int SetPaddingAtrr(
            int padding_b=114,
            int padding_g=114,	
            int padding_r=114, 
            int align=0);

        /**
         * @brief Set the linear transformation attribute.
         * 
         * @param alpha_beta (a0, b0), (a1, b1), (a2, b2) factors
         * @return 0 for success and other for failure 
         */
        int SetConvertAtrr(
            const std::tuple<
                std::pair<float, float>,
                std::pair<float, float>,
                std::pair<float, float>> &alpha_beta);

        /**
         * @brief Push Image
         * 
         * @param channel_idx   Channel index number of the image.
         * @param image_idx     Image index number of the image.
         * @param image         Input image
         * @return 0 for success and other for failure 
         */
        int PushImage(
            int channel_idx, 
            int image_idx, 
            BMImage &image);

        /**
         * @brief Get the Batch Data object
         * 
         * @return std::tuple<std::map<std::string,sail::Tensor*>,      Output Tensor map.
         * std::vector<BMImage>,                                        Original Images
         * std::vector<int>,                                            Original Channel index
         * std::vector<int>>                                            Original Index
         * std::vector<std::vector<int>>>                               Padding Atrr(start_x, start_y, width, height)
         */
        std::tuple<std::map<std::string,sail::Tensor*>, 
            std::vector<BMImage>,
            std::vector<int>,
            std::vector<int>,
            std::vector<std::vector<int>>> GetBatchData(bool need_d2s = true);

        
        /**
         * @brief Get the Batch Data object
         * 
         * @return std::tuple<std::map<std::string,sail::Tensor*>,      Output Tensor map.
         * std::vector<cv::Mat>,                                        Original Images
         * std::vector<int>,                                            Original Channel index
         * std::vector<int>>                                            Original Index
         * std::vector<std::vector<int>>>                               Padding Atrr(start_x, start_y, width, height)
         */
        std::tuple<std::map<std::string,sail::Tensor*>, 
            std::vector<cv::Mat>,
            std::vector<int>,
            std::vector<int>,
            std::vector<std::vector<int>>> GetBatchData_CV(bool need_d2s = true);

#ifdef PYTHON
        std::tuple<std::vector<TensorPTRWithName>, 
            std::vector<BMImage>,
            std::vector<int>,
            std::vector<int>,
            std::vector<std::vector<int>>> GetBatchData_py(bool need_d2s = true);

        std::tuple<std::map<std::string, pybind11::array_t<float>>, 
            std::vector<BMImage>,
            std::vector<int>,
            std::vector<int>,
            std::vector<std::vector<int>>> GetBatchData_Npy();

        std::tuple<std::map<std::string, pybind11::array_t<float>>, 
            std::vector<pybind11::array_t<uint8_t>>,
            std::vector<int>,
            std::vector<int>,
            std::vector<std::vector<int>>> GetBatchData_Npy2();
#endif

        /**
         * @brief Get first graph name in the loaded bomodel.
         *
         * @return First graph name
         */
        std::string get_graph_name();


        /**
         * @brief Get model input width.
         * 
         * @return Model input width
        */
        int get_input_width();

        /**
         * @brief Get model input height.
         * 
         * @return Model input height
        */
        int get_input_height();

        /**
         * @brief Get all output tensor names of the first graph.
         *
         * @return All the output tensor names of the graph
         */
        std::vector<std::string> get_output_names();

        /**
         * @brief Get the shape of an output tensor in frist graph.
         *
         * @param tensor_name The specified tensor name
         * @return The shape of the tensor
         */
        std::vector<int> get_output_shape(const std::string& tensor_name);

        void exhausted();
        bool get_exhausted_flag();

    private:
        class EngineImagePreProcess_CC;
        class EngineImagePreProcess_CC* const _impl;

        /**
         * @brief Forbidden copy constructor.
         * @brief Copy constructor.
         *
         * @param other An other EngineImagePreProcess instance.
         */
        EngineImagePreProcess(const EngineImagePreProcess& other) = delete;

        /**
         * @brief Forbidden assignment function.
         * @brief Assignment function.
         *
         * @param other An other EngineImagePreProcess instance.
         * @return Reference of a EngineImagePreProcess instance.
         */
        EngineImagePreProcess& operator=(const EngineImagePreProcess& other) = delete;
    };
#endif
}