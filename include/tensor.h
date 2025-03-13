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

/** @file     tensor.hpp
 *  @brief    Header file of Tensor
 *  @author   SOPHON
 *  @version  2.0.3
 *  @date     2019-12-27
 */

#pragma once
#include <spdlog/spdlog.h>
#include <bmruntime_interface.h>
#include <bmlib_runtime.h>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <string.h>
#ifdef PYTHON
#include <type_traits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#endif

/// Namespace containing all symbols from the sail library.
namespace sail {

class MemoryError : public std::logic_error {
public:
    explicit MemoryError(const std::string &what) : std::logic_error(what) {}
};

class NotSupport : public std::logic_error {
public:
    explicit NotSupport(const std::string &what) : std::logic_error(what) {}
};

class DataTypeError : public std::logic_error {
public:
    explicit DataTypeError(const std::string &what) : std::logic_error(what) {}
};

class SailBaseError : public std::runtime_error {
public:                                          
  explicit SailBaseError(const std::string& what)
      : std::runtime_error(what) {}              
};

#define DEFINE_SAIL_EXCEPTION_TYPE(ExceptionName) \
class ExceptionName : public SailBaseError {      \
public:                                           \
  explicit ExceptionName(const std::string& what) \
      : SailBaseError(what) {}                    \
};

DEFINE_SAIL_EXCEPTION_TYPE(SailDeviceError)
DEFINE_SAIL_EXCEPTION_TYPE(SailTensorError)
DEFINE_SAIL_EXCEPTION_TYPE(SailEngineError)
DEFINE_SAIL_EXCEPTION_TYPE(SailBMImageError)
DEFINE_SAIL_EXCEPTION_TYPE(SailDecoderError)
DEFINE_SAIL_EXCEPTION_TYPE(SailEncoderError)
DEFINE_SAIL_EXCEPTION_TYPE(SailRuntimeError)
DEFINE_SAIL_EXCEPTION_TYPE(SailUnknownError)

#undef DEFINE_SAIL_EXCEPTION_TYPE
/* sail status */
// typedef enum sail_status_t {
//   SAIL_SUCCESS = 0,
//   SAIL_ERR_DEVICE_INIT = 1,    /* device query failed */
//   SAIL_ERR_TENSOR_INIT = 11,   /* sail tensor init failed */
//   SAIL_ERR_TENSOR_INNER = 12,  /* sail tensor inner processing failed */
//   SAIL_ERR_ENGINE_INIT = 21,   /* sail engine init failed */
//   SAIL_ERR_ENGINE_INNER = 22,  /* sail engine inner attribute judge failed */
//   SAIL_ERR_ENGINE_INPUT = 23,  /* sail engine input attribute judge failed */
//   SAIL_ERR_ENGINE_OUTPUT = 24, /* sail engine output attribute judge failed */
//   SAIL_ERR_ENGINE_INFER = 25,  /* sail engine inference failed */
//   SAIL_ERR_BMCV_INIT = 31,     /* sail bmcv init failed */
//   SAIL_ERR_BMCV_TRANS = 32,    /* sail bmcv data type transform failed */
//   SAIL_ERR_BMCV_FUNC = 33,     /* sail bmcv process failed */
//   SAIL_ERR_DECODER_INIT = 41,  /* sail decoder init failed */
//   SAIL_ERR_DECODER_READ = 42,  /* sail decoder get frame failed */
// } sail_status_t;

typedef enum sail_status_t {
    // success return
    SAIL_SUCCESS = 0,

    SAIL_ERR_DEV_BASIC      = 1001, // Failed due to a generic device error.
    SAIL_ERR_DEV_INIT       = 1002, // Failed to initialize device resources.
    SAIL_ERR_DEV_CLOSE      = 1003, // Failed to close device resources properly.
    SAIL_ERR_DEV_HANDLE     = 1004, // Failed due to an invalid device handle.
    SAIL_ERR_DEV_MALLOC     = 1005, // Failed to allocate device or system memory.
    SAIL_ERR_DEV_MCOPY      = 1006, // Failed to copy memory (d2d, d2s, s2d, ...).
    SAIL_ERR_DEV_MMAP       = 1007, // Failed to map or unmap device memory.
    SAIL_ERR_DEV_QFULL      = 1008, // Failed because the queue is full.
    SAIL_ERR_DEV_QEMPTY     = 1009, // Failed because the queue is empty.

    // Tensor-related errors.
    SAIL_ERR_TENSOR_BASIC   = 2001, // Failed due to a generic tensor error.
    SAIL_ERR_TENSOR_INIT    = 2002, // Failed to initialize a tensor.
    SAIL_ERR_TENSOR_EMPTY   = 2003, // Failed because the tensor is empty.
    SAIL_ERR_TENSOR_SHAPE   = 2004, // Failed due to an invalid tensor shape.
    SAIL_ERR_TENSOR_DTYPE   = 2005, // Failed due to an invalid tensor data type.
    SAIL_ERR_TENSOR_SIZE    = 2006, // Failed because tensor size does not match.
    SAIL_ERR_TENSOR_PARAM   = 2007, // Failed due to invalid tensor parameters.
    SAIL_ERR_TENSOR_SYSMEM  = 2008, // Failed due to invalid tensor system memory status.
    SAIL_ERR_TENSOR_DEVMEM  = 2009, // Failed due to invalid tensor device memory status.
    SAIL_ERR_TENSOR_CVT     = 2010, // Failed to convert the tensor.
    SAIL_ERR_TENSOR_DUMP    = 2011, // Failed to dump tensor data.

    // Engine-related errors.
    SAIL_ERR_ENGINE_BASIC   = 3001, // Failed due to a generic engine error.
    SAIL_ERR_ENGINE_INIT    = 3002, // Failed to initialize the engine.
    SAIL_ERR_ENGINE_BMODEL  = 3003, // Failed due to an invalid bmodel file.
    SAIL_ERR_ENGINE_LOAD    = 3004, // Failed to load the bmodel.
    SAIL_ERR_ENGINE_DTYPE   = 3005, // Failed due to an invalid data type for input/output.
    SAIL_ERR_ENGINE_SHAPE   = 3006, // Failed due to an invalid shape for input/output.
    SAIL_ERR_ENGINE_INPUT   = 3007, // Failed due to invalid input to the engine.
    SAIL_ERR_ENGINE_OUTPUT  = 3008, // Failed due to invalid output from the engine.
    SAIL_ERR_ENGINE_GRAPH   = 3009, // Failed due to an invalid computational graph.
    SAIL_ERR_ENGINE_INFER   = 3010, // Failed during the inference process.
    SAIL_ERR_ENGINE_PUSH    = 3011, // Failed to push data to the engine.
    SAIL_ERR_ENGINE_GET     = 3012, // Failed to get data from the engine.

    // BMImage-related errors.
    SAIL_ERR_BMI_BASIC      = 4001, // Failed due to a generic BMImage error.
    SAIL_ERR_BMI_INIT       = 4002, // Failed to initialize BMImage.
    SAIL_ERR_BMI_EMPTY      = 4003, // Failed because BMImage is empty.
    SAIL_ERR_BMI_SHAPE      = 4004, // Failed due to an invalid BMImage shape.
    SAIL_ERR_BMI_DTYPE      = 4005, // Failed due to an invalid BMImage data type.
    SAIL_ERR_BMI_FORMAT     = 4006, // Failed due to an invalid pixel format.
    SAIL_ERR_BMI_PARAM      = 4007, // Failed due to invalid BMImage parameters.
    SAIL_ERR_BMI_NOTSUP     = 4008, // Failed because the operation is not supported.
    SAIL_ERR_BMI_BMCV       = 4009, // Failed to execute a bmcv function for BMImage.
    SAIL_ERR_BMI_ALLOC      = 4010, // Failed to allocate memory for BMImage.
    SAIL_ERR_BMI_CVT        = 4011, // Failed to convert BMImage.
    SAIL_ERR_BMI_IPCINIT    = 4012, // Failed to initialize IPC resource.
    SAIL_ERR_BMI_IPCSEND    = 4013, // Failed on IPC send side.
    SAIL_ERR_BMI_IPCRCV     = 4014, // Failed on IPC receive side.

    // Decoder and MultiDecoder-related errors.
    SAIL_ERR_DEC_BASIC      = 5001, // Failed due to a generic decoder error.
    SAIL_ERR_DEC_INIT       = 5002, // Failed to initialize the decoder.
    SAIL_ERR_DEC_OPEN       = 5003, // Failed to open the input file for decoding.
    SAIL_ERR_DEC_FORMAT     = 5004, // Failed to parse the input format.
    SAIL_ERR_DEC_CODEC      = 5005, // Failed to open the codec for decoding.
    SAIL_ERR_DEC_READ       = 5006, // Failed to read a frame for decoding.
    SAIL_ERR_DEC_EOF        = 5007, // Failed because the end of the file was reached during decoding.
    SAIL_ERR_DEC_TIMEOUT    = 5008, // Failed due to a timeout while reading a frame.
    SAIL_ERR_DEC_CVT        = 5009, // Failed to convert the decoded frame.
    SAIL_ERR_DEC_DUMP       = 5010, // Failed to dump decoder data.
    SAIL_ERR_DEC_JPEG       = 5011, // Failed to decode a JPEG image.
    SAIL_ERR_DEC_MDADD      = 5012, // Failed to add a channel to the MultiDecoder.
    SAIL_ERR_DEC_MDDEL      = 5013, // Failed to delete a channel from the MultiDecoder.
    SAIL_ERR_DEC_MDCLR      = 5014, // Failed to clear the MultiDecoder queue.
    SAIL_ERR_DEC_MDRD       = 5015, // Failed to read a frame from the MultiDecoder.

    // Encoder-related errors.
    SAIL_ERR_ENC_BASIC      = 6001, // Failed due to a generic encoder error.
    SAIL_ERR_ENC_INIT       = 6002, // Failed to initialize the encoder.
    SAIL_ERR_ENC_OPEN       = 6003, // Failed to open the output file for encoding.
    SAIL_ERR_ENC_FORMAT     = 6004, // Failed to parse the output format.
    SAIL_ERR_ENC_CODEC      = 6005, // Failed to open the codec for encoding.
    SAIL_ERR_ENC_VDWR       = 6006, // Failed to write video data during encoding.
    SAIL_ERR_ENC_EOF        = 6007, // Failed because the end of the file was reached during encoding.
    SAIL_ERR_ENC_CVT        = 6008, // Failed to convert BMImage to frame for encoding.
    SAIL_ERR_ENC_DUMP       = 6009, // Failed to dump encoder data.
    SAIL_ERR_ENC_JPEG       = 6010, // Failed to encode a JPEG image.

    SAIL_ERR_RUNTIME_BASIC  = 7001  // Failed to create a runtime.
} sail_status_t;


const static std::unordered_map<sail_status_t, std::string> MAP_RET_DESC
{
    {sail_status_t::SAIL_ERR_DEV_BASIC, "Failed due to a generic device error."},
    {sail_status_t::SAIL_ERR_DEV_INIT, "Failed to initialize device resources."},
    {sail_status_t::SAIL_ERR_DEV_CLOSE, "Failed to close device resources properly."},
    {sail_status_t::SAIL_ERR_DEV_HANDLE, "Failed due to an invalid device handle."},
    {sail_status_t::SAIL_ERR_DEV_MALLOC, "Failed to allocate device or system memory."},
    {sail_status_t::SAIL_ERR_DEV_MCOPY, "Failed to copy memory (d2d, d2s, s2d, ...)."},
    {sail_status_t::SAIL_ERR_DEV_MMAP, "Failed to map or unmap device memory."},
    {sail_status_t::SAIL_ERR_DEV_QFULL, "Failed because the queue is full."},
    {sail_status_t::SAIL_ERR_DEV_QEMPTY, "Failed because the queue is empty."},

    // Tensor-related errors.
    {sail_status_t::SAIL_ERR_TENSOR_BASIC, "Failed due to a generic tensor error."},
    {sail_status_t::SAIL_ERR_TENSOR_INIT, "Failed to initialize a tensor."},
    {sail_status_t::SAIL_ERR_TENSOR_EMPTY, "Failed because the tensor is empty."},
    {sail_status_t::SAIL_ERR_TENSOR_SHAPE, "Failed due to an invalid tensor shape."},
    {sail_status_t::SAIL_ERR_TENSOR_DTYPE, "Failed due to an invalid tensor data type."},
    {sail_status_t::SAIL_ERR_TENSOR_SIZE, "Failed because tensor size does not match."},
    {sail_status_t::SAIL_ERR_TENSOR_PARAM, "Failed due to invalid tensor parameters."},
    {sail_status_t::SAIL_ERR_TENSOR_SYSMEM, "Failed due to invalid tensor system memory status."},
    {sail_status_t::SAIL_ERR_TENSOR_DEVMEM, "Failed due to invalid tensor device memory status."},
    {sail_status_t::SAIL_ERR_TENSOR_CVT, "Failed to convert the tensor."},
    {sail_status_t::SAIL_ERR_TENSOR_DUMP, "Failed to dump tensor data."},

    // Engine-related errors.
    {sail_status_t::SAIL_ERR_ENGINE_BASIC, "Failed due to a generic engine error."},
    {sail_status_t::SAIL_ERR_ENGINE_INIT, "Failed to initialize the engine."},
    {sail_status_t::SAIL_ERR_ENGINE_BMODEL, "Failed due to an invalid bmodel file."},
    {sail_status_t::SAIL_ERR_ENGINE_LOAD, "Failed to load the bmodel."},
    {sail_status_t::SAIL_ERR_ENGINE_DTYPE, "Failed due to an invalid data type for input/output."},
    {sail_status_t::SAIL_ERR_ENGINE_SHAPE, "Failed due to an invalid shape for input/output."},
    {sail_status_t::SAIL_ERR_ENGINE_INPUT, "Failed due to invalid input to the engine."},
    {sail_status_t::SAIL_ERR_ENGINE_OUTPUT, "Failed due to invalid output from the engine."},
    {sail_status_t::SAIL_ERR_ENGINE_GRAPH, "Failed due to an invalid computational graph."},
    {sail_status_t::SAIL_ERR_ENGINE_INFER, "Failed during the inference process."},
    {sail_status_t::SAIL_ERR_ENGINE_PUSH, "Failed to push data to the engine."},
    {sail_status_t::SAIL_ERR_ENGINE_GET, "Failed to get data from the engine."},

    // BMImage-related errors.
    {sail_status_t::SAIL_ERR_BMI_BASIC, "Failed due to a generic BMImage error."},
    {sail_status_t::SAIL_ERR_BMI_INIT, "Failed to initialize BMImage."},
    {sail_status_t::SAIL_ERR_BMI_EMPTY, "Failed because BMImage is empty."},
    {sail_status_t::SAIL_ERR_BMI_SHAPE, "Failed due to an invalid BMImage shape."},
    {sail_status_t::SAIL_ERR_BMI_DTYPE, "Failed due to an invalid BMImage data type."},
    {sail_status_t::SAIL_ERR_BMI_FORMAT, "Failed due to an invalid pixel format."},
    {sail_status_t::SAIL_ERR_BMI_PARAM, "Failed due to invalid BMImage parameters."},
    {sail_status_t::SAIL_ERR_BMI_NOTSUP, "Failed because the operation is not supported."},
    {sail_status_t::SAIL_ERR_BMI_BMCV, "Failed to execute a bmcv function for BMImage."},
    {sail_status_t::SAIL_ERR_BMI_ALLOC, "Failed to allocate memory for BMImage."},
    {sail_status_t::SAIL_ERR_BMI_CVT, "Failed to convert BMImage."},
    {sail_status_t::SAIL_ERR_BMI_IPCINIT, "Failed to initialize IPC resource."},
    {sail_status_t::SAIL_ERR_BMI_IPCSEND, "Failed on IPC send side."},
    {sail_status_t::SAIL_ERR_BMI_IPCRCV, "Failed on IPC receive side."},

    // Decoder and MultiDecoder-related errors.
    {sail_status_t::SAIL_ERR_DEC_BASIC, "Failed due to a generic decoder error."},
    {sail_status_t::SAIL_ERR_DEC_INIT, "Failed to initialize the decoder."},
    {sail_status_t::SAIL_ERR_DEC_OPEN, "Failed to open the input file for decoding."},
    {sail_status_t::SAIL_ERR_DEC_FORMAT, "Failed to parse the input format."},
    {sail_status_t::SAIL_ERR_DEC_CODEC, "Failed to open the codec for decoding."},
    {sail_status_t::SAIL_ERR_DEC_READ, "Failed to read a frame for decoding."},
    {sail_status_t::SAIL_ERR_DEC_EOF, "Failed because the end of the file was reached during decoding."},
    {sail_status_t::SAIL_ERR_DEC_TIMEOUT, "Failed due to a timeout while reading a frame."},
    {sail_status_t::SAIL_ERR_DEC_CVT, "Failed to convert the decoded frame."},
    {sail_status_t::SAIL_ERR_DEC_DUMP, "Failed to dump decoder data."},
    {sail_status_t::SAIL_ERR_DEC_JPEG, "Failed to decode a JPEG image."},
    {sail_status_t::SAIL_ERR_DEC_MDADD, "Failed to add a channel to the MultiDecoder."},
    {sail_status_t::SAIL_ERR_DEC_MDDEL, "Failed to delete a channel from the MultiDecoder."},
    {sail_status_t::SAIL_ERR_DEC_MDCLR, "Failed to clear the MultiDecoder queue."},
    {sail_status_t::SAIL_ERR_DEC_MDRD, "Failed to read a frame from the MultiDecoder."},

    // Encoder-related errors.
    {sail_status_t::SAIL_ERR_ENC_BASIC, "Failed due to a generic encoder error."},
    {sail_status_t::SAIL_ERR_ENC_INIT, "Failed to initialize the encoder."},
    {sail_status_t::SAIL_ERR_ENC_OPEN, "Failed to open the output file for encoding."},
    {sail_status_t::SAIL_ERR_ENC_FORMAT, "Failed to parse the output format."},
    {sail_status_t::SAIL_ERR_ENC_CODEC, "Failed to open the codec for encoding."},
    {sail_status_t::SAIL_ERR_ENC_VDWR, "Failed to write video data during encoding."},
    {sail_status_t::SAIL_ERR_ENC_EOF, "Failed because the end of the file was reached during encoding."},
    {sail_status_t::SAIL_ERR_ENC_CVT, "Failed to convert BMImage to frame for encoding."},
    {sail_status_t::SAIL_ERR_ENC_DUMP, "Failed to dump encoder data."},
    {sail_status_t::SAIL_ERR_ENC_JPEG, "Failed to encode a JPEG image."},
     
    //Runtime-related errors.
    {sail_status_t::SAIL_ERR_RUNTIME_BASIC, "Failed to create a runtime."}

};

#define BASENAME(x) (strrchr(x, '/') ? strrchr(x, '/') + 1 : x)

std::string ret_to_string(int ret);

int sail_status_to_exception_type(int ret);

/**
 * @brief Check funtion return value, throw exception if fail.
 *
 * @param ret value returned from other function.
 */
void check_return_status(int ret);

/**
 * @brief Used for inner check. If the returned value represents an error,
 *        it will print error info and throw an exception.
 *
 * @param ret value returned from other function, like bmlib, bmcv.
 */
#define SAIL_CHECK_RET(ret)                                        \
    do {                                                           \
        if (ret) {                                                 \
            spdlog::error("{} {}: {}: {}", ret_to_string(ret),     \
                          BASENAME(__FILE__), __func__, __LINE__); \
            check_return_status(ret);                              \
        }                                                          \
    } while (0)

enum class LogLevel: int
{
  TRACE = 0,
  DEBUG,
  INFO,
  WARN,
  ERROR,
  CRITICAL,
  OFF
};

/**
 * @brief Get the number of available TPUs.
 *
 * @return Number of available TPUs.
 */
int DECL_EXPORT get_available_tpu_num();

/**
 * @brief Get current time of system.
 *
 * @return current time.
 */
double DECL_EXPORT get_current_time_us();

#ifdef _WIN32
    int DECL_EXPORT setenv(const char* name, const char* value, int overwrite);
#endif

int DECL_EXPORT set_print_flag(bool print_flag);

int DECL_EXPORT set_dump_io_flag(bool dump_io_flag);

void DECL_EXPORT get_sail_version(char* sail_version);

bool DECL_EXPORT get_print_flag();

void DECL_EXPORT printEnvVarsWithPrefix(const std::string &prefix);

/**
   * @brief Get the processor percent utilization of the specified device
   *
   * @param dev_id Device id
   * @return The processor percent utilization of the specified device
   */
int get_tpu_util(int dev_id);

/**
   * @brief Get the VPU percent utilization of the specified device
   *
   * @param dev_id Device id
   * @return The VPU percent utilization of the specified device
   */
std::vector<int> get_vpu_util(int dev_id);

/**
   * @brief Get the VPP percent utilization of the specified device
   *
   * @param dev_id Device id
   * @return The VPP percent utilization of the specified device
   */
std::vector<int> get_vpp_util(int dev_id);

/**
   * @brief Get the board temperature of the specified device
   *
   * @param dev_id Device id
   * @return The board temperature of the specified device
   */
int get_board_temp(int dev_id);


/**
   * @brief Get the chip temperature of the specified device
   *
   * @param dev_id Device id
   * @return The chip temperature of the specified device
   */
int get_chip_temp(int dev_id);



/**
    * @brief Get the state of the specified device
    *
    * @param dev_id Device id
    * @return The state [mem_total,mem_used,tpu_util] used of the specified device
    */
std::vector<int> get_dev_stat(int dev_id);


  
/**
 * @brief Set the logging level to the specified level.
 * 
 * @param loglevel The desired logging level as a LogLevel enum value.
 * @return int Returns 0 if the log level was successfully set, or -1 if the
 *         input log level is unknown.
 */
int DECL_EXPORT set_loglevel(LogLevel loglevel=LogLevel::INFO);

#define PRINT_function_Time_ms(func_name, start_time, end_time) printf("Function[%s]-[%s] time use: %.4f ms \n",__FUNCTION__,func_name, abs(end_time-start_time)/1000.0);

#define PRINT_TIME_MS(func_name, start_time) if(get_print_flag()){ \
          PRINT_function_Time_ms(func_name, start_time, get_current_time_us());}


class DECL_EXPORT Handle {
 public:
  /**
   * @brief Default constructor.
   */
  explicit Handle();

  /**
   * This Function is not recommended, will be removed in future 
   *
   * @brief Constructor using existed bm_handle_t.
   *
   * @param handle A bm_handle_t
   */
  explicit Handle(bm_handle_t handle);

  /**
   * @brief Constructor with device id.
   *
   * @param dev_id Device id
   */
  explicit Handle(int dev_id);

  /**
   * @brief Copy constructor.
   *
   * @param other An other Handle instance.
   */
  Handle(const Handle& other);

  /**
   * @brief Assignment function.
   *
   * @param other An other Handle instance.
   * @return Reference of a Handle instance.
   */
  Handle& operator=(const Handle& other);

  /**
   * @brief Destructor.
   */
  ~Handle();

  /**
   * @brief Get inner bm_handle_t.
   *
   * @return Inner bm_handle_t
   */
  bm_handle_t data();

  /**
   * @brief Get device id of this handle.
   *
   * @return Device id.
   */
  int get_device_id();

  /**
   * @brief Get serial number
   * 
   * @return serial number
   */

  std::string get_sn();

  /**
   * @brief Get TPU chip type
   * 
   * @return TPU chip type
   */

  std::string get_target();

  static std::unordered_map<bm_handle_t, Handle*> handle_map;
  static std::mutex map_mutex;

  std::shared_ptr<bm_handle_t> shaptr();

 private:
  //   /**
  //  * @brief Forbidden copy constructor.
  //  */
  // Handle(const Handle& other) = delete;

  // /**
  //  * @brief Forbidden assignment function.
  //  */
  // Handle& operator=(const Handle& other) = delete;

  class Handle_CC;
  class Handle_CC* const _impl;
};

/**
 * @brief Indicator of where to store input and output tensors.
 */
enum IOMode {
  /// Input tensors are in system memory while output tensors are
  /// in device memory.
  SYSI,

  /// Input tensors are in device memory while output tensors are
  /// in system memory.
  SYSO,

  /// Both input and output tensors are in system memory.
  SYSIO,

  /// Both input and output tensors are in device memory.
  DEVIO
};

/**
 * @brief A class manages the system and device memory of a tensor.
 *
 * A tensor may only stores in sytem memory, or only stores in device memory,
 * or stores in both system memory and device memory. This class handles all
 * the conditions.
 */
class DECL_EXPORT Tensor {
 public:
  /**
   * @brief Common constructor.\n
   * @detail
   *  case 0: only allocate system memory
   *          (handle, shape, dtype, true, false) \n
   *  case 1: only allocate device memory
   *          (handle, shape, dtype, false, true) \n
   *  case 2: allocate system memory and device memory
   *          (handle, shape, dtype, true, true) \n
   *
   * @param handle       Handle instance
   * @param shape        Shape of the tensor
   * @param dtype        Data type
   * @param own_sys_data Indicator of whether own system memory
   * @param own_dev_data Indicator of whether own device memory
   */
  explicit Tensor(
      const Handle&           handle,
      const std::vector<int>& shape={},
      bm_data_type_t          dtype=BM_FLOAT32,
      bool                    own_sys_data=false,
      bool                    own_dev_data=false);

  /**
   * @brief Constructor of only system data.\n
   *
   * @param shape Shape of the tensor
   * @param dtype Data type
   */
  explicit Tensor(
      const std::vector<int>& shape={},
      bm_data_type_t          dtype=BM_FLOAT32);

  /**
   * @brief Constructor of device data.
   *
   * @param shape A vector of shape
   * @param t A part of device memory
   */
  explicit Tensor(
      const Handle&           handle,
      const std::vector<int>& shape,
      bm_data_type_t          dtype,
      bm_device_mem_t         t);

  /**
   * @brief Copy constructor.
   *
   * @param tensor A Tensor instance
   */
  Tensor(const Tensor& tensor);
  Tensor(Tensor&& tensor);

  /**
   * @brief Constructor to create a new tensor with some parts of other tensor, (a slice function like array[a:b,c:d] in python).
   *
   * @param tensor A Tensor instance
   * @param ranges The range of the tesnor you are trying to copy, support at most 2D
   * @param d2d_flag Whether use d2d or c2c to copy device data
   */
  Tensor(const Tensor& tensor, const std::vector<std::pair<int, int>> &ranges, bool d2d_flag=true);

  /**
   * @brief Constructor to create a new tensor from another existing tensor.
   *
   * @param src A Tensor instance
   * @param shape The shape of new tensor, which can not exceed src's shape.
   * @param offset element(not byte) index offset to the start of src's device memory
   * @param no_copy If no_copy flag is set to true, the created tensor will share 
                    the src's memory, instead of allocating new memory.
   */
  Tensor(const Tensor &src, const std::vector<int> &shape, unsigned int offset, bool no_copy = true);

  /**
   * @brief Assignment function.
   *
   * @param tensor A Tensor instance
   * @return A Tensor instance
   */
  Tensor& operator=(const Tensor& tensor);
  Tensor& operator=(Tensor&&  tensor);

  virtual ~Tensor();

  /**
   * @brief Scale data to tensor in system memory.
   *
   * @param src   Data of type float32 to be scaled from
   * @param scale Scale value
   */
  void scale_from(float* src, float scale);

  /**
   * @brief Scale data to tensor in system memory.
   *
   * @param src   Data of type float32 to be scaled from
   * @param scale Scale value
   * @param size  Size of data
   */
  void scale_from(float* src, float scale, int size);

  /**
   * @brief Scale int32 type data to tensor in system memory.
   *
   * @param src   Data of type int32 to be scaled from
   * @param scale Scale value
   * @param size  Size of data
   */
  void scale_from_int32(int32_t* src, float scale, int size);

  /**
   * @brief Scale tensor to data in system memory.
   *
   * @param dst   Data of type float32 to scaled to
   * @param scale Scale value
   */
  void scale_to(float* dst, float scale);

  /**
   * @brief Scale tensor to data in system memory.
   *
   * @param dst   Data of type float32 to scaled to
   * @param scale Scale value.
   * @param size  Size of data to scale to
   */
  void scale_to(float* dst, float scale, int size);

  /**
   * @brief Fill memory with a constant scalar, This interface can convert a scalar to tensor's dtype in physical memory.
   * 
   * @param c constant scalar
   */
  void memory_set(float c);

#ifdef PYTHON
  /**
   * @brief Fill memory with a numpy scalar.
   * 
   * @param c a numpy scalar
   */ 
  void memory_set(pybind11::object c);
#endif

  /**
   * @brief Fill memory with buffer of a scalar, the size of buffer to read depends on tensor's dtype.
   * 
   * @param value
   */ 
  void memory_set(void* value);

  /**
   * @brief Fill memory with zeros
   */
  void zeros();

  /**
   * @brief Fill memory with ones
   */
  void ones();

  /**
   * @brief Check whether device data is valid.
   * @param return true if device data is valid, otherwise false.
   */
  bool is_dev_data_valid() const;

  /**
   * @brief Check whether system data is valid.
   * @param return true if system data is valid, otherwise false.
   */
  bool is_sys_data_valid() const;

    /**
   * @breaf Dump Tensor data to file
   * 
   * @param file_name File path to dump tensor
   * @param bin Binary storage true/false
   * 
  */
  void dump_data(std::string file_name, bool bin = false);


#ifdef PYTHON
  /**
   * @brief Constructor allocates device memory of the tensor(py).
   *
   * @param handle Handle instance
   * @param data   Ndarray data
   */
    explicit Tensor(Handle handle, pybind11::array_t<float>&   data);
    explicit Tensor(Handle handle, pybind11::array_t<int8_t>&  data);
    explicit Tensor(Handle handle, pybind11::array_t<uint8_t>& data);
    explicit Tensor(Handle handle, pybind11::array_t<int32_t>& data);

    explicit Tensor(Handle handle, pybind11::array_t<float>&   data, bool own_sys_data, bool own_dev_data=true);
    explicit Tensor(Handle handle, pybind11::array_t<int8_t>&  data, bool own_sys_data, bool own_dev_data=true);
    explicit Tensor(Handle handle, pybind11::array_t<uint8_t>& data, bool own_sys_data, bool own_dev_data=true);
    explicit Tensor(Handle handle, pybind11::array_t<int32_t>& data, bool own_sys_data, bool own_dev_data=true);
  /**
   * @brief Get ndarray in system memory of the tensor.
   *
   * @return Ndarray data
   */
  pybind11::object asnumpy();

  /**
   * @brief Get ndarray in system memory of the tensor with specified shape.
   *
   * @return Ndarray data with specified shape.
   */
  pybind11::object asnumpy(const std::vector<int>& shape);

  /**
   *
   */
  pybind11::array_t<long> pysys_data();

  /**
   * @brief Scale data to tensor in system memory.
   *
   * @param data  Data of type float32 to be scaled from.
   * @param scale Scale value.
   */
  void scale_from(pybind11::array_t<float>& data, float scale);

  void scale_from(pybind11::array_t<int32_t>& data, float scale);
  /**
   * @brief Scale tensor to data in system memory.
   *
   * @param scale Scale value.
   * @return Ndarray data of type float32 to scale to.
   */
  pybind11::array_t<float> scale_to(float scale);

  /**
   * @brief Scale tensor to data in system memory.
   *
   * @param scale Scale value.
   * @param shape Shape of output data to scale to.
   * @return Ndarray data of type float32 to scale to.
   */
  pybind11::array_t<float> scale_to(float scale, const std::vector<int>& shape);

  /*pybind11::array_t<int32_t> scale_to(float scale);
  pybind11::array_t<int32_t> scale_to(
    float                   scale,
    const std::vector<int>& shape);*/

  /**
   * @brief Update system data of the tensor. The data size should not exceed
   *        the tensor size, and the tensor shape will not be changed.
   *
   * @param data Ndarray data with the same data type of the tensor.
   */
  template <typename T>
  void update_data(pybind11::array_t<T>& data) {
    return update_data(data.request(),sizeof(T));
  }
#endif

  /**
   * @brief Get Handle instance.
   *
   * @return Handle reference
   */
  Handle& get_handle();


  /**
   * @brief get device id
   * 
   * @return int device id
   */
  int device_id();

  /**
   * @brief Get shape of the tensor.
   *
   * @return Shape of the tensor
   */
  const std::vector<int>& shape() const;

  /**
   * @brief Get data type of the tensor.
   *
   * @return Data type of the tensor
   */
  bm_data_type_t dtype() const;

  /**
   * @brief Reset data type and shape of the tensor.
   *
   * @param shape Shape of the tensor
   * @param dtype Data type of the tensor
   */
  void reset(const std::vector<int>& shape, bm_data_type_t dtype);

  /**
   * @brief Reset shape of the tensor.
   *
   * @param shape Shape of the tensor
   */
  void reshape(const std::vector<int>& shape);

  /**
   * @brief Judge if the tensor owns data in system memory.
   *
   * @return True for owns data in system memory.
   */
  bool& own_sys_data();

  /**
   * @brief Judge if the tensor owns data in device memory.
   *
   * @return True for owns data in device memory.
   */
  bool& own_dev_data();

  /**
   * @brief Get data pointer in system memory of the tensor.
   *
   * @return Data pointer in system memory of the tensor
   */
  void* sys_data();

  /**
   * @brief Get pointer to device memory of the tensor.
   *
   * @return Instance of device memory structure of the tensor
   */
  bm_device_mem_t dev_data();

  /**
   * @brief Reset data pointer in system memory of the tensor.
   *
   * @param data  Data pointer in system memory of the tensor
   * @param shape Shape of the data
   */
  void reset_sys_data(
      void*             data,
      std::vector<int>& shape);

  /**
   * @brief Reset pointer to device memory of the tensor.
   *
   * @param data Instance of device memory structure
   */
  void reset_dev_data(bm_device_mem_t data);

  /**
   * @brief Copy data from system memory to device memory.
   */
  void sync_s2d();

  /**
   * @brief Copy data from system memory to device memory with specified size.
   *
   * @param size Byte size to be copied
   */
  void sync_s2d(int size);

  /**
   * @brief Copy data from device memory to system memory.
   */
  void sync_d2s();

  /**
   * @brief Copy data from device memory to system memory with specified size.
   *
   * @param size Byte size to be copied
   */
  void sync_d2s(int size);

  /**
   * @brief Copy data from device memory to device memory with specified size.
   *
   * @param Tensor Specifies the Tensor to be copied from.
   * @param offset_src Specifies the number of elements to offset in the source Tensor from where to start copying.
   * @param offset_dst Specifies the number of elements to offset in the destination Tensor from where to start copying.
   * @param len Specifies the length of the copy, i.e., the number of elements to copy.
   */
  void sync_d2d(Tensor& src, int offset_src, int offset_dst, int len);

  /**
   * @brief Copy data from device memory to device memory with specified size.
   *
   * @param Tensor Specifies the Tensor to be copied from.
   * @param stride_src Specifies the stride of the source Tensor.
   * @param stride_dst Specifies the stride of the destination Tensor.
   * @param count Specifies the number of elements to copy.
   */
  void sync_d2d_stride(Tensor& src, int stride_src, int stride_dst, int count);

  /**
   * @brief Copy data from device memory to system memory with specified size.
   *
   * @param Tensor Specifies the Tensor to be copied from.
   * @param offset_src Specifies the number of elements to offset in the source Tensor from where to start copying.
   * @param offset_dst Specifies the number of elements to offset in the destination Tensor from where to start copying.
   * @param len Specifies the length of the copy, i.e., the number of elements to copy.
   */
  void sync_d2s(Tensor& src, int offset_src, int offset_dst, int len);

    /**
   * @brief Copy data from system memory to device memory with specified size.
   *
   * @param Tensor Specifies the Tensor to be copied from.
   * @param offset_src Specifies the number of elements to offset in the source Tensor from where to start copying.
   * @param offset_dst Specifies the number of elements to offset in the destination Tensor from where to start copying.
   * @param len Specifies the length of the copy, i.e., the number of elements to copy.
   */
  void sync_s2d(Tensor& src, int offset_src, int offset_dst, int len);
  
  /**
   * @brief Copy data from another tensor to this tensor.
   *
   * @param src Another tensor pointer.
   */
  void sync_from(Tensor* src);

  /**
   * @brief Copy data from this tensor to another tensor.
   *
   * @param src Another tensor pointer.
   */
  void sync_to(Tensor* dst);

  /**
   * @brief Free system and device memroy of the tensor.
   */
  void free();

  /**
   * @brief Returns the number of elements in this tensor.
   * 
   * @return The number of elements in this tensor.
   */
  int size() const;

  /**
   * @brief Returns the size in bytes of an individual element.
   * 
   * @return The byte size of a single tensor element.
   */
  int element_size() const;

  /**
   * @brief Return the total number of bytes occupied by all elements of Tensor.
   * 
   * @return The total number of bytes occupied by all elements in Tensor.
   */
  int nbytes() const;


  // /**
  //  * @brief Get the value of index.
  //  *
  //  * @param index index.
  //  */
  // template<typename R>
  // R at(int index){
  //   void* ptr_sys = this->sys_data();
  //   if(!ptr_sys){
  //       SPDLOG_INFO("Do not found system memory, not support!");
  //       return -1;
  //   }
  //   bm_data_type_t dtype_temp = this->dtype();
  //   switch(dtype_temp) {
  //       case BM_FLOAT32:{
  //           float* ptr = (float*)ptr_sys;
  //           return ptr[index];
  //       }
  //       case BM_INT8:{
  //           int8_t* ptr = (int8_t*)ptr_sys;
  //           return ptr[index];
  //       }
  //       case BM_UINT8:{
  //           uint8_t* ptr = (uint8_t*)ptr_sys;
  //           return ptr[index];
  //       }
  //       case BM_INT32:{
  //           int* ptr = (int*)ptr_sys;
  //           return ptr[index];
  //       }
  //        case BM_UINT32:{
  //           uint* ptr = (uint*)ptr_sys;
  //           return ptr[index];
  //       }
  //       default:
  //       SPDLOG_INFO("Dtype not support!");
  //       return -1;
  //   }
  // }

 private:

  class Tensor_CC;
  class Tensor_CC* const _impl;

#ifdef PYTHON
  void update_data(const pybind11::buffer_info& buf, int type_size);
#endif
};

  struct TensorPTRWithName
  {
    TensorPTRWithName(): name(""),data(NULL) { }
    TensorPTRWithName(const std::string& n, sail::Tensor* d) : name(n), data(d) {} 
    std::string name;
    sail::Tensor* data;
  };
  
  /**
   * @brief Release TensorPTRWithName data.
   */
  void ReleaseTensorPtr(TensorPTRWithName& tensor_with_name);

}  // namespace sail
