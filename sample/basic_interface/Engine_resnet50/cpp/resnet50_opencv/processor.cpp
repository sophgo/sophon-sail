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


#include "inireader.hpp"

#include "processor.h"

bool file_exists(const std::string& file_path) {
  std::ifstream f(file_path.c_str());
  return f.good();
}

PreProcessor::PreProcessor(
    int   height,
    int   width,
    float mean_0,
    float mean_1,
    float mean_2)
    : height_(height), width_(width) {
  linear_trans_param_ = std::make_tuple(std::make_pair(1,-mean_0),std::make_pair(1,-mean_1),std::make_pair(1,-mean_2));

}

void PreProcessor::process(sail::Handle& handle, sail::Tensor& input, cv::Mat& frame, sail::Bmcv& bmcv) {
  sail::BMImage bmimg;
  bmcv.mat_to_bm_image(frame, bmimg);
  sail::BMImage resized_bmimg;
  bmcv.vpp_resize(bmimg, resized_bmimg, width_, height_);
  sail::BMImage linear_bmimg = sail::BMImage(handle,width_,height_,FORMAT_BGR_PLANAR,DATA_TYPE_EXT_FLOAT32);
  bmcv.convert_to(resized_bmimg,linear_bmimg,linear_trans_param_);
  bmcv.bm_image_to_tensor(linear_bmimg, input);

}

PostProcessor::PostProcessor(
    size_t batch_size,
    size_t class_num,
    size_t top_k)
    : batch_size_(batch_size), class_num_(class_num), top_k_(top_k) {
  if (top_k > class_num) {
    spdlog::error("Error: top_k > class_num");
    throw;
  }
}

std::vector<std::vector<int>> PostProcessor::process(float* input) {
   std::vector<std::vector<int>> result;

  float *data = input;
  for (int i = 0; i < batch_size_; ++i) {
    // initialize original index locations
    std::vector<int> idx(class_num_);
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in data
    std::stable_sort(idx.begin(), idx.end(),
              [&data](int i1, int i2) {return data[i1] > data[i2];});
    idx.resize(top_k_);
    result.push_back(idx);
    data += class_num_;
  }
  return result;
}

std::map<std::string, std::vector<int>> PostProcessor::get_reference(
    const std::string& compare_path) {
  std::map<std::string, std::vector<int>> reference;
  if (!compare_path.empty()) {
    INIReader reader(compare_path);
    if (reader.ParseError()) {
      spdlog::error("Can't load reference file: {}!", compare_path);
      std::terminate();
    }
    reference["fp32"].push_back(reader.GetInteger("resnet50_fp32", "top_1", 0));
    reference["fp32"].push_back(reader.GetInteger("resnet50_fp32", "top_2", 0));
    reference["fp32"].push_back(reader.GetInteger("resnet50_fp32", "top_3", 0));
    reference["fp32"].push_back(reader.GetInteger("resnet50_fp32", "top_4", 0));
    reference["fp32"].push_back(reader.GetInteger("resnet50_fp32", "top_5", 0));
    reference["int8"].push_back(reader.GetInteger("resnet50_int8", "top_1", 0));
    reference["int8"].push_back(reader.GetInteger("resnet50_int8", "top_2", 0));
    reference["int8"].push_back(reader.GetInteger("resnet50_int8", "top_3", 0));
    reference["int8"].push_back(reader.GetInteger("resnet50_int8", "top_4", 0));
    reference["int8"].push_back(reader.GetInteger("resnet50_int8", "top_5", 0));
  }
  return std::move(reference);
}

bool PostProcessor::compare(
    std::map<std::string, std::vector<int>>& reference,
    std::vector<int>&                        result,
    const std::string&                       dtype) {
  if (reference.empty()) {
    spdlog::info("No verify_files file or verify_files err.");
    return true;
  }
  std::vector<int>& ref = reference[dtype];
  bool ret = true;
  for (int i = 0; i < result.size(); ++i) {
    if (result[i] != ref[i]) {
      ret = false;
      break;
    }
  }
  if (!ret) {
    spdlog::error("Result compare failed!");
    spdlog::info("Expected result: [{}]", fmt::join(ref, ", "));
  }
  return ret;
}
