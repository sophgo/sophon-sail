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

/**
 * @brief Load a bmodel and do inference.
 *
 * @param bmodel_path  Path to bmodel
 * @param input_path   Path to input image file
 * @param dev_id       ID of TPU to use
 * @return Program state
 *     @retval true  Success
 *     @retval false Failure
 */
bool inference(
    const std::string& bmodel_path,
    const std::string& input_path,
    int                dev_id) {
  // init Engine to load bmodel and allocate input and output tensors SYSIO
  sail::Engine engine(bmodel_path, dev_id, sail::SYSO);
  // get model info
  // only one model loaded for this engine
  // only one input tensor and only one output tensor in this graph
  auto graph_name = engine.get_graph_names().front();
  auto input_name = engine.get_input_names(graph_name).front();
  auto output_name = engine.get_output_names(graph_name).front();
  auto input_shape = engine.get_input_shape(graph_name, input_name);
  auto output_shape = engine.get_output_shape(graph_name, output_name);
  auto in_dtype = engine.get_input_dtype(graph_name, input_name);
  auto out_dtype = engine.get_output_dtype(graph_name, output_name);
  // prepare input_tensor and output_tensor
  sail::Handle& handle = engine.get_handle();
  sail::Tensor input = sail::Tensor(handle, input_shape, in_dtype, false, false);
  std::map<std::string, sail::Tensor*> input_tensors = {{input_name, &input}};
  sail::Tensor output = sail::Tensor(handle, output_shape, out_dtype, true,  true);
  std::map<std::string, sail::Tensor*> output_tensors = {{output_name, &output}}; 
  sail::Bmcv bmcv(handle);
  int in_size = std::accumulate(input_shape.begin(), input_shape.end(),
                              1, std::multiplies<int>());
  int out_size = std::accumulate(output_shape.begin(), output_shape.end(),
                              1, std::multiplies<int>());
  // init preprocessor and postprocessor
  PreProcessor preprocessor(input_shape[2], input_shape[3]);
  PostProcessor postprocessor(output_shape[0], output_shape[1], 5);

  // read image
  cv::Mat frame = cv::imread(input_path);
  // preprocess
  preprocessor.process(handle, input, frame, bmcv);
  // inference
  engine.process(graph_name,input_tensors,output_tensors);
  // postprocess
  float* output_data = reinterpret_cast<float*>(output.sys_data());
  auto result = postprocessor.process(output_data);
  // print result
  for (auto item : result) {
    spdlog::info("Top 5 on tpu {}: [{}]", dev_id, fmt::join(item, ", "));
  }

  return true;  
}

/// It is the simplest case for inference of one model on one TPU.
int main(int argc, char** argv) {
  // 设置默认参数
  std::string jpg_dir = "../../datasets/imagenet_val_1k/img/ILSVRC2012_val_00000075.JPEG"; // 默认的图片路径
  std::string bmodel_path = "../../models/BM1684X/resnet50_int8_1b.bmodel"; // 默认的模型文件路径
  int device_id = 0; // 默认的设备ID
  // 读取命令行参数，如果提供了则覆盖默认值 
  if (argc > 1) {
      jpg_dir = argv[1];
  }
  if (argc > 2) {
      bmodel_path = argv[2];
      cout << "bmodel file: " << bmodel_path << endl;
  }
  if (argc > 3) {
      device_id = std::stoi(argv[3]); // 转换字符串为整数
  }

  // load bmodel and do inference
  bool status = inference(bmodel_path, jpg_dir, device_id);
      if (status){
        return 0;
    }else{
        return 1;
    }
}
