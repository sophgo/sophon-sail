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

#ifdef _WIN32
#include "getopt_win.h"
#else
#include <getopt.h>
#endif

#define USE_FFMPEG  1
#define USE_OPENCV  1
#define USE_BMCV    1

#include <string>
#include <vector>
#include <numeric>
#include <fstream>
#include <inireader.hpp>

#include <iostream>
#include <sail/cvwrapper.h>
#include <sail/engine.h>
#include <sail/tensor.h>



class PreProcessor {
 public:
  /**
   * @brief Constructor.
   *
   * @param scale Scale factor from float32 to int8
   */
  PreProcessor(float scale = 1.0);

  /**
   * @brief Execution function of preprocessing.
   *
   * @param input Input data
   * @param input Output data
   */
//   void process(cv::Mat& input, cv::Mat& output);

 protected:
  float ab_[6];
};

class BmcvPreProcessor: public PreProcessor{  //
 public:
  /**
   * @brief Constructor.
   *
   * @param bmcv  Reference to a Bmcv instance
   * @param scale Scale factor from float32 to int8
   */
  BmcvPreProcessor(sail::Bmcv& bmcv, float scale = 1.0);

  /**
   * @brief Execution function of preprocessing.
   *
   * @param input Input data
   * @param input Output data
   */
  void process(sail::BMImage& input, sail::BMImage& output);

  protected: sail::Bmcv& bmcv_;
};


class PostProcessor {
 public:
  /**
   * @brief Constructor.
   *
   * @param batch_size Batch size
   * @param class_num  Class number
   * @param top_k      Number of classification result
   */
  PostProcessor(size_t batch_size, size_t class_num, size_t top_k);

  /**
   * @brief Destructor.
   */
  ~PostProcessor() {}

  /**
   * @brief Execution function of postprocessing
   *
   * @param input The output data of inference
   * @return The classification result of each input
   *     @retval Ex:[[20,3,4,5,7],[6,30,1,2,3]] the index of classes
   */
  std::vector<std::vector<int>> process(float* input);

  /**
   * @brief Get correct result from given file.
   *
   * @param compare_path Path to correct result file
   * @return correct result
   */
  std::map<std::string, std::vector<int>> get_reference(
      const std::string& compare_path);

  /**
   * @brief Compare result.
   *
   * @param reference Correct result
   * @param result    Output result
   * @param dtype     Data type of model
   * @return Program state
   *     @retval true  Success
   *     @retval false Failure
   */
  bool compare(
    std::map<std::string, std::vector<int>>& reference,
    std::vector<int>&                        result,
    const std::string&                       dtype);

 private:
  size_t batch_size_;
  size_t class_num_;
  size_t top_k_;
};
