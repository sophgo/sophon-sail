#pragma once
#include <map>
#include <set>
#include <vector>
#include <string>
#include <memory>
#include "tensor.h"
#include "cvwrapper.h"

#ifdef PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#endif

/// Namespace containing all symbols from the sail library.
namespace sail {


#ifdef BUILD_ENGINELLM
/**
 * @brief The main class of running deep learning inference on TPU.
 *
 * It's a high level encapsulation of BMRuntime. It automatically manage
 * memory of input and output tensors for both static and dynamic models.
 * It can load more than one model and runs on more than one TPU.
 */
class DECL_EXPORT EngineLLM {
 public:
  /**
   * @brief Constructor loads bmodel from file.
   *
   * @param bmodel_path Path to bmodel
   * @param tpu_ids     TPU ID list. You can use bm-smi to see available IDs
   */
  EngineLLM(
      const std::string& bmodel_path,
      std::vector<int>   tpu_ids);

  /**
   * @brief Constructor loads bmodel from system memory.
   *
   * @param bmodel_ptr  Pointer to bmodel in system memory
   * @param bmodel_size Byte size of bmodel in system memory
   * @param tpu_ids     TPU ID list. You can use bm-smi to see available IDs.
   */
  EngineLLM(
      const void*       bmodel_ptr,
      size_t            bmodel_size,
      std::vector<int>  tpu_ids);

  ~EngineLLM();

                                                        // get info about EngineLLM, i.e. bmodel
  /**
   * @brief Get device id of this EngineLLM.
   *
   * @return Device id list.
   */
  std::vector<int> get_device_ids() const;

  /**
   * @brief Get all graph names in the loaded bmodels.
   *
   * @return All graph names
   */
  std::vector<std::string> get_graph_names() const;

                                                        // get info about graph, i.e. net 
  /**
   * @brief Get address assign mode.
   *        0: basic mode
   *        1: io alone
   *
   * @param graph_name  graph_name The specified graph name
   * @return address assign mode.
   */
  int get_addr_mode(const std::string& graph_name) const;

  /**
   * @brief Get stage num of the specified graph.
   * @param graph_name The specified graph name
   *
   * @return stage num.
   */
  int get_stage_num(const std::string& graph_name) const;

  /**
   * @brief Get input num of the specified graph.
   * @param graph_name The specified graph name
   *
   * @return input num.
   */
  int get_input_num(const std::string& graph_name) const;

  /**
   * @brief Get output num of the specified graph.
   * @param graph_name The specified graph name
   *
   * @return output num.
   */
  int get_output_num(const std::string& graph_name) const;

  /**
   * @brief Jugde if the graph is dynamic.
   *
   * @param graph_name The specified graph name
   * @return 0 for dynamic and 1 for static
   */
  bool get_is_dynamic(const std::string& graph_name) const;

  std::string get_input_name(const std::string& graph_name, const int index) const;

  std::string get_output_name(const std::string& graph_name, const int index) const;

  /**
   * @brief Get all input tensor names of the specified graph.
   *
   * @param graph_name The specified graph name
   * @return All the input tensor names of the graph
   */
  std::vector<std::string> get_input_names(const std::string& graph_name) const;

  /**
   * @brief Get all output tensor names of the specified graph.
   *
   * @param graph_name The specified graph name
   * @return All the output tensor names of the graph
   */
  std::vector<std::string> get_output_names(const std::string& graph_name) const;

  /**
   * @brief Get the input tensors object. Only for addr_mode 1.
   * 
   * @param graph_name The specified graph name
   * @param stage stage
   * @return std::map<int, Tensor*> If addr_mode is not 1, return empty map.
   */
  std::map<int, Tensor*> get_input_tensors(const std::string& graph_name, const int stage = 0);

  /**
   * @brief Get the input tensors object. Only for addr_mode 1.
   * 
   * @param graph_name  The specified graph name
   * @param tensor_name The specified tensor name
   * @param stage stage
   * @return std::map<int, Tensor*> If addr_mode is not 1, return empty map.
   */
  std::map<int, Tensor*> get_input_tensors(const std::string &graph_name,
                                           const std::string &tensor_name,
                                           const int stage = 0);

  /**
   * @brief Get the input tensor object. Only for addr_mode 1.
   * 
   * @param graph_name  The specified graph name
   * @param index tensor index 
   * @param stage stage
   * @return Tensor* If addr_mode is not 1, return nullptr.
   */
  Tensor* get_input_tensor(const std::string& graph_name, const int index, const int stage = 0);

  /**
   * @brief Create a group of input Tensor, only for addr_mode 0.
   * 
   * @param graph_name graph_name The specified graph name
   * @param stage stage
   * @return std::map<int, Tensor*>
   */
  std::map<int, Tensor*> create_max_input_tensors(const std::string& graph_name);

  /**
   * @brief Get the output tensors object. Only for addr_mode 1.
   * 
   * @param graph_name 
   * @param stage stage
   * @return std::map<int, Tensor*> If addr_mode is not 1, return empty map.
   */
  std::map<int, Tensor*> get_output_tensors(const std::string& graph_name, const int stage = 0);

  /**
   * @brief Get the output tensors object. Only for addr_mode 1.
   * 
   * @param graph_name The specified graph name
   * @param tensor_name The specified tensor name
   * @param stage stage
   * @return std::map<int, Tensor*> If addr_mode is not 1, return empty map.
   */
  std::map<int, Tensor*> get_output_tensors(const std::string &graph_name,
                                                  const std::string &tensor_name,
                                                  const int stage = 0);

  /**
   * @brief Get the output tensor object. Only for addr_mode 1.
   * 
   * @param graph_name The specified graph name
   * @param index tensor index
   * @param stage stage
   * @return Tensor*. If addr_mode is not 1, return nullptr.
   */
  Tensor* get_output_tensor(const std::string &graph_name,
                                 const int index, const int stage = 0);

  /**
   * @brief Create a group of output Tensor, only for addr_mode 0.
   * 
   * @param graph_name graph_name The specified graph name
   * @param stage stage
   * @return std::map<int, Tensor*>
   */
  std::map<int, Tensor*> create_max_output_tensors(const std::string& graph_name);

  /**
   * @brief Get the input tensors object, even when addr_mode is 0.
   * 
   * @param graph_name The specified graph name
   * @param stage stage
   * @return std::map<int, Tensor*>
   */
  std::map<int, Tensor*> get_input_tensors_addrmode0(const std::string& graph_name, const int stage = 0);

  /**
   * @brief Get the output tensors object, even when addr_mode is 0.
   * 
   * @param graph_name The specified graph name
   * @param stage stage
   * @return std::map<int, Tensor*>
   */
  std::map<int, Tensor*> get_output_tensors_addrmode0(const std::string& graph_name, const int stage = 0);

  /**
   * @brief Get device id of the specified input tensor.
   *
   * @param graph_name The specified graph name
   * @param index input tensor index
   * @return device id
   */
  int get_input_tensor_devid(const std::string& graph_name, const int index) const;

  /**
   * @brief Get device id of the specified output tensor.
   *
   * @param graph_name The specified graph name
   * @param index output tensor index
   * @return device id
   */
  int get_output_tensor_devid(const std::string& graph_name, const int index) const;

  /**
   * @brief Get the shape of an input tensor in a graph.
   *
   * @param graph_name  The specified graph name
   * @param index input tensor index
   * @param stage stage
   * @return The shape of the tensor
   */
  std::vector<int> get_input_shape(const std::string &graph_name,
                                   const int index,
                                   const int stage = 0) const;

  /**
   * @brief Get the max shape in multiple stages' input tensors with same input index in a graph.
   *
   * @param graph_name  The specified graph name
   * @param index input tensor index
   * @return The shape of the tensor
   */
  std::vector<int> get_input_max_shape(const std::string &graph_name,
                                       const int index) const;

  /**
   * @brief Get the min shape in multiple stages' input tensors with same input index in a graph.
   *
   * @param graph_name  The specified graph name
   * @param index input tensor index
   * @return The shape of the tensor
   */
  std::vector<int> get_input_min_shape(const std::string &graph_name,
                                       const int index) const;

  /**
   * @brief Get the shape of an output tensor in a graph.
   *
   * @param graph_name  The specified graph name
   * @param index output tensor index
   * @param stage stage
   * @return The shape of the tensor
   */
  std::vector<int> get_output_shape(const std::string& graph_name,
                                    const int index,
                                    const int stage = 0) const;

  /**
   * @brief Get the max shape in multiple stages' output tensors with same output index in a graph.
   *
   * @param graph_name  The specified graph name
   * @param index output tensor index
   * @return The shape of the tensor
   */
  std::vector<int> get_output_max_shape(const std::string &graph_name,
                                        const int index) const;

  /**
   * @brief Get the min shape in multiple stages' output tensors with same output index in a graph.
   *
   * @param graph_name  The specified graph name
   * @param index output tensor index
   * @return The shape of the tensor
   */
  std::vector<int> get_output_min_shape(const std::string &graph_name,
                                        const int index) const;

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
   * @param index input tensor index
   * @return Data type of the input tensor
   */
  bm_data_type_t get_input_dtype(const std::string& graph_name,
                                 const int index) const;

  /**
   * @brief Get data type of an output tensor. Refer to bmdef.h as following.
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
   * @param index output tensor index
   * @return Data type of the outut tensor
   */
  bm_data_type_t get_output_dtype(const std::string& graph_name,
                                  const int index) const;

  /**
   * @brief Get scale of an input tensor. Only used for int8 models.
   *
   * @param graph_name  The specified graph name
   * @param index input tensor index
   * @return Scale of the input tensor
   */
  float get_input_scale(const std::string& graph_name,
                        const int index) const;

  /**
   * @brief Get scale of an output tensor. Only used for int8 models.
   *
   * @param graph_name  The specified graph name
   * @param index output tensor index
   * @return Scale of the output tensor
   */
  float get_output_scale(const std::string& graph_name,
                         const int index) const;

                                                        // process, inference
  /**
   * @brief Inference with provided input and output tensors.
   *
   * @param input     Input tensors
   * @param output    Output tensors
   */
  void process(
      const std::string&      graph_name,
      std::map<int, Tensor*>& input,
      std::map<int, Tensor*>& output,
      const std::vector<int> &core_list = {0});


#ifdef PYTHON
  EngineLLM(
      pybind11::bytes&  bmodel_bytes,
      int               bmodel_size,
      std::vector<int>  tpu_ids);

#endif

 private:

  class EngineLLM_CC;
  class EngineLLM_CC* const _impl;

  /**
   * @brief Forbidden copy constructor.
   * @brief Copy constructor.
   *
   * @param other An other EngineLLM instance.
   */
  EngineLLM(const EngineLLM& other) = delete;

  /**
   * @brief Forbidden assignment function.
   * @brief Assignment function.
   *
   * @param other An other EngineLLM instance.
   * @return Reference of a EngineLLM instance.
   */
  EngineLLM& operator=(const EngineLLM& other) = delete;
};

#endif

}