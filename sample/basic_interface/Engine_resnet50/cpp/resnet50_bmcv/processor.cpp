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

#include "processor.h"

PreProcessor::PreProcessor(float scale) :
  ab_{1, -103.94, 1, -116.78, 1, -123.68} {
  for (int i = 0; i < 6; i ++) {
    ab_[i] *= scale;
  }
}

BmcvPreProcessor::BmcvPreProcessor(sail::Bmcv& bmcv, float scale)
    : PreProcessor(scale), bmcv_(bmcv) {
}

void BmcvPreProcessor::process(sail::BMImage& input, sail::BMImage& output) {
  sail::BMImage tmp;
  bmcv_.vpp_resize(input, tmp, 224, 224);
  bmcv_.convert_to(tmp, output,
                   std::make_tuple(std::make_pair(ab_[0], ab_[1]),
                                   std::make_pair(ab_[2], ab_[3]),
                                   std::make_pair(ab_[4], ab_[5])));
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
