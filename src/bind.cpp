/* Copyright 2016-2022 by SOPHON Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.  */

#include <fstream>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include "tensor.h"
#include "engine.h"
#include "cvwrapper.h"
#include "tools.h"
#include "base64.h"
#include "internal.h"
#include "engine_multi.h"
#include "decoder_multi.h"
#include "algokit.h"
#include "encoder.h"
#include "tpu_kernel_api.h"
#include "perf.h"
#include "engine_llm.h"

#ifdef USE_IPC
#include "ipc.h"
#endif // USE_IPC

using namespace sail;
namespace py = pybind11;

#ifdef USE_BMCV
template<std::size_t N>
static void declareBMImageArray(py::module &m) {
  std::stringstream ss; ss << "BMImageArray" << N << "D";
  py::class_<BMImageArray<N>>(m, ss.str().c_str())
    .def(py::init<>())
    .def(py::init<Handle&, int, int, bm_image_format_ext, bm_image_data_format_ext>())

    .def("__len__", &BMImageArray<N>::size)
    .def("__getitem__",
         [](BMImageArray<N> &v, size_t i) -> bm_image & {
           if (i >= v.size()) throw py::index_error();
           return v[i];
         }, py::return_value_policy::reference_internal)
    .def("__setitem__",
         [](BMImageArray<N> &v, size_t i, const bm_image &t) {
           if (i >= v.size()) throw py::index_error();
           if (v.check_need_free()) {
               if (v[0].width != t.width || v[0].height != t.height || v[0].image_format != t.image_format ||
                   v[0].data_type != t.data_type) {
                   printf("ERROR:__setitem__:  requires src image's format is same as dst\n");
                   printf("src(w=%d,h=%d, format=%s, dtype=%s\n", t.width, t.height,
                           bm_image_format_desc(t.image_format),
                           bm_image_data_type_desc(t.data_type));
                   printf("dst(w=%d,h=%d, format=%s, dtype=%s\n", v[i].width, v[i].height,
                          bm_image_format_desc(v[i].image_format),
                          bm_image_data_type_desc(v[i].data_type));
                   throw py::value_error();
               }
               bm_handle_t  handle = bm_image_get_handle(&v[0]);
               bmcv_copy_to_atrr_t attr;
               memset(&attr, 0, sizeof(attr));
               int ret = bmcv_image_copy_to(handle, attr, t, v[i]);
               if (BM_SUCCESS != ret) {
                   SPDLOG_ERROR("bmcv_image_copy_to err={}", ret);
                   throw py::value_error();
               }
           }else{
               //printf("__setitem__:\n");
               //print_image(t, "src:");
               int stride[3]={0};
               bm_handle_t  handle = bm_image_get_handle((bm_image*)&t);
               if (handle != nullptr) {
                   bm_image_get_stride(t, stride);
                  for(int temp_idx = 0; temp_idx < v.size(); ++temp_idx){
                    int ret = bm_image_create(handle, t.height, t.width, t.image_format, 
                      t.data_type, &v[temp_idx], stride);
                    if (BM_SUCCESS != ret) {
                      SPDLOG_ERROR("bm_image_create err={}", ret);
                      throw py::value_error();
                    }
                    // ret = bm_image_alloc_dev_mem_heap_mask(v[temp_idx], 6);
                    // if (BM_SUCCESS != ret) {
                    //   SPDLOG_ERROR("bm_image_alloc_dev_mem_heap_mask err={}", ret);
                    //   throw py::value_error();
                    // }
                  }
                  bm_image_alloc_contiguous_mem(N, v.data());
                  //  bm_device_mem_t dev_mem[3];
                  //  bm_image_get_device_mem(t, dev_mem);
                  //  bm_image_attach(v[i], dev_mem);
                  bmcv_copy_to_atrr_t attr;
                  memset(&attr, 0, sizeof(attr));
                  int ret = bmcv_image_copy_to(handle, attr, t, v[0]);
                  if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("bmcv_image_copy_to err={}", ret);
                    throw py::value_error();
                  }    
                  v.set_need_free(true);
               }else{
                   SPDLOG_ERROR("src image handle=nullptr");
                   throw py::value_error();
               }
           }
         }
    )
    .def("check_need_free", (bool (BMImageArray<N>::*) ()) &BMImageArray<N>::check_need_free)
    .def("set_need_free", (void (BMImageArray<N>::*) (bool))   &BMImageArray<N>::set_need_free)
    .def("create", (void (BMImageArray<N>::*)(Handle&, int, int, bm_image_format_ext, bm_image_data_format_ext)) &BMImageArray<N>::create)
    .def("copy_from",        &BMImageArray<N>::copy_from)
    .def("attach_from",      &BMImageArray<N>::attach_from)
    .def("get_device_id",    &BMImageArray<N>::get_device_id);
}

template<std::size_t N>
static void registerBMImageArrayFunctions(py::class_<Bmcv> &cls) {
  cls.def("bm_image_to_tensor",  (void            (Bmcv::*)(BMImageArray<N>&, Tensor&))       &Bmcv::bm_image_to_tensor)
     .def("bm_image_to_tensor",  (Tensor          (Bmcv::*)(BMImageArray<N>&))                &Bmcv::bm_image_to_tensor)
     .def("tensor_to_bm_image",  (void            (Bmcv::*)(Tensor&, BMImageArray<N>&, bool, std::string)) &Bmcv::tensor_to_bm_image, py::arg("tensor"), py::arg("img"), py::arg("bgr2rgb")=false, py::arg("layout")=std::string("nchw"))
     .def("tensor_to_bm_image",  (void            (Bmcv::*)(Tensor&, BMImageArray<N>&, bm_image_format_ext)) &Bmcv::tensor_to_bm_image, py::arg("tensor"), py::arg("img"), py::arg("format"))
     .def("crop_and_resize",     (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int, int, int, int, int, bmcv_resize_algorithm)) &Bmcv::crop_and_resize,
      py::arg("input"),py::arg("crop_x0"), py::arg("crop_y0"), py::arg("crop_w"),py::arg("crop_h"),py::arg("resize_w"), py::arg("resize_h"), py::arg("resize_alg")=BMCV_INTER_NEAREST)
     .def("crop",                (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int, int, int))           &Bmcv::crop)
     .def("resize",              (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int, bmcv_resize_algorithm)) &Bmcv::resize,
      py::arg("input"),py::arg("resize_w"), py::arg("resize_h"), py::arg("resize_alg")=BMCV_INTER_NEAREST)
     .def("vpp_resize",          (int             (Bmcv::*)(BMImageArray<N>&, BMImageArray<N>&, int, int, bmcv_resize_algorithm))   &Bmcv::vpp_resize,
      py::arg("input"),py::arg("output"), py::arg("resize_w"), py::arg("resize_h"), py::arg("resize_alg"))
     .def("vpp_crop_and_resize", (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int, int, int, int, int, bmcv_resize_algorithm)) &Bmcv::vpp_crop_and_resize,
      py::arg("input"), py::arg("crop_x0"), py::arg("crop_y0"), py::arg("crop_w"), py::arg("crop_h"), py::arg("resize_w"), py::arg("resize_h"), py::arg("resize_alg")=BMCV_INTER_NEAREST)
     .def("vpp_crop",            (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int, int, int))           &Bmcv::vpp_crop)
     .def("vpp_resize",          (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int, bmcv_resize_algorithm))  &Bmcv::vpp_resize,
      py::arg("input"), py::arg("resize_w"), py::arg("resize_h"), py::arg("resize_alg")=BMCV_INTER_NEAREST)
     .def("vpp_crop_and_resize_padding", (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int, int, int, int, int, PaddingAtrr&, bmcv_resize_algorithm)) &Bmcv::vpp_crop_and_resize_padding,
      py::arg("input"), py::arg("crop_x0"), py::arg("crop_y0"), py::arg("crop_w"), py::arg("crop_h"), py::arg("resize_w"), py::arg("resize_h"), py::arg("padding_in"), py::arg("resize_alg")=BMCV_INTER_NEAREST)
     .def("vpp_resize_padding",  (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int, PaddingAtrr&, bmcv_resize_algorithm))  &Bmcv::vpp_resize_padding,
      py::arg("input"), py::arg("resize_w"), py::arg("resize_h"), py::arg("padding_in"), py::arg("resize_alg")=BMCV_INTER_NEAREST)
     .def("yuv2bgr",             (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&))                               &Bmcv::yuv2bgr)
     .def("warp",                (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, const std::array<std::pair<std::tuple<float, float, float>, std::tuple<float, float, float>>, N>&, int, bool))              &Bmcv::warp,
      py::arg("input"), py::arg("matrix"), py::arg("use_bilinear")=0, py::arg("similar_to_opencv")=false)
     .def("warp",                (int (Bmcv::*)(BMImageArray<N>&, BMImageArray<N>&, const std::array<std::pair<std::tuple<float, float, float>, std::tuple<float, float, float>>, N>&, int, bool))              &Bmcv::warp,
      py::arg("input"), py::arg("output"), py::arg("matrix"), py::arg("use_bilinear")=0, py::arg("similar_to_opencv")=false)
     .def("convert_to",          (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, const std::tuple<std::pair<float, float>, std::pair<float, float>, std::pair<float, float>>&))                   &Bmcv::convert_to)
     .def("convert_to",          (int             (Bmcv::*)(BMImageArray<N>&, BMImageArray<N>&, const std::tuple<std::pair<float, float>, std::pair<float, float>, std::pair<float, float>>&)) &Bmcv::convert_to)
     .def("image_copy_to",          (int          (Bmcv::*)(BMImageArray<N>&, BMImageArray<N>&, int, int))  &Bmcv::image_copy_to,
        py::arg("input"),py::arg("output"), py::arg("start_x")=0, py::arg("start_y")=0)
     .def("image_copy_to_padding",  (int          (Bmcv::*)(BMImageArray<N>&, BMImageArray<N>&, unsigned int, unsigned int, unsigned int, int, int))  &Bmcv::image_copy_to_padding,
        py::arg("input"),py::arg("output"), py::arg("padding_r"), py::arg("padding_g"), py::arg("padding_b"),py::arg("start_x")=0, py::arg("start_y")=0);
}

// base64_decoder
template<typename DTYPE>
pybind11::array_t<DTYPE> get_ndarray(Handle& handle, uint32_t input_size, string& input){
    // 获取输出的字节数
    uint32_t output_size = input_size / 4 * 3 ;
    uint32_t byte_size = 0;

    // 计算预估的size,考虑到缓冲字节，可能会比真实值大1，后面再resize更正
    uint32_t real_size = output_size / sizeof(DTYPE);
    std::vector<ssize_t> shape;
    shape.push_back(real_size);
    
    pybind11::array_t<DTYPE> ndarray = pybind11::array_t<DTYPE>(shape);
    auto data = ndarray.mutable_data();

    // base64 decode
    base64_dec(handle,input.data(),input_size, reinterpret_cast<uint8_t*>(data), &byte_size);

    // compare size
    auto real_decode_size = byte_size / sizeof(DTYPE);
    assert(real_decode_size <= real_size);

    // resize
    shape[0] = real_decode_size;
    ndarray.resize(shape);

    return ndarray;
}
#endif


PYBIND11_MODULE(sail, m) {

  char temp[64] = {0};
  get_sail_version(temp);
  using namespace pybind11::literals;
  m.attr("__version__") = temp;
  m.doc() = "sophon inference module";

  m.def("get_board_temp", &get_board_temp);
  m.def("get_chip_temp", &get_chip_temp);
  m.def("get_dev_stat", &get_dev_stat);
  
  m.def("get_available_tpu_num", &get_available_tpu_num);
  m.def("get_tpu_util", &get_tpu_util);
  m.def("get_vpu_util", &get_vpu_util);
  m.def("get_vpp_util", &get_vpp_util);
  m.def("set_print_flag", &set_print_flag);
  m.def("set_dump_io_flag", &set_dump_io_flag);
  m.def("set_decoder_env",&set_decoder_env);
  m.def("_dryrun", &model_dryrun);
  m.def("_perf",   &multi_tpu_perf);
  py::register_exception<MemoryError>(m, "MemoryError");
  py::register_exception<NotSupport>(m, "NotSupport");
  py::register_exception<DataTypeError>(m, "DataTypeError");

#define REGISTER_SAIL_EXCEPTION_TYPE(ExceptionName) \
py::register_exception<ExceptionName>(m, #ExceptionName);

  REGISTER_SAIL_EXCEPTION_TYPE(SailDeviceError)
  REGISTER_SAIL_EXCEPTION_TYPE(SailTensorError)
  REGISTER_SAIL_EXCEPTION_TYPE(SailEngineError)
  REGISTER_SAIL_EXCEPTION_TYPE(SailBMImageError)
  REGISTER_SAIL_EXCEPTION_TYPE(SailDecoderError)
  REGISTER_SAIL_EXCEPTION_TYPE(SailEncoderError)
  REGISTER_SAIL_EXCEPTION_TYPE(SailUnknownError)

#undef REGISTER_SAIL_EXCEPTION_TYPE

  m.def("set_loglevel", &set_loglevel);

  py::enum_<bm_data_type_t>(m, "Dtype")
    .value("BM_FLOAT32", bm_data_type_t::BM_FLOAT32)
    .value("BM_INT8", bm_data_type_t::BM_INT8)
    .value("BM_UINT8", bm_data_type_t::BM_UINT8)
    .value("BM_FLOAT16", bm_data_type_t::BM_FLOAT16)
    .value("BM_BFLOAT16", bm_data_type_t::BM_BFLOAT16)
    .value("BM_INT32", bm_data_type_t::BM_INT32)
    .value("BM_UINT32", bm_data_type_t::BM_UINT32)
    .value("BM_INT16", bm_data_type_t::BM_INT16)
    .value("BM_UINT16", bm_data_type_t::BM_UINT16)
    //.value("BM_INT64", bm_data_type_t::BM_INT64)
    .export_values();

  py::enum_<IOMode>(m, "IOMode")
    .value("SYSI", IOMode::SYSI)
    .value("SYSO", IOMode::SYSO)
    .value("SYSIO", IOMode::SYSIO)
    .value("DEVIO", IOMode::DEVIO)
    .export_values();

  py::enum_<LogLevel>(m, "LogLevel")
    .value("TRACE",     LogLevel::TRACE)
    .value("DEBUG",     LogLevel::DEBUG)
    .value("INFO",      LogLevel::INFO)
    .value("WARN",      LogLevel::WARN)
    .value("ERROR",     LogLevel::ERROR)
    .value("CRITICAL",  LogLevel::CRITICAL)
    .value("OFF",       LogLevel::OFF)
    .export_values();

#ifdef USE_BMCV
  py::enum_<bmcv_resize_algorithm>(m, "bmcv_resize_algorithm")
    .value("BMCV_INTER_NEAREST", bmcv_resize_algorithm::BMCV_INTER_NEAREST)
    .value("BMCV_INTER_LINEAR", bmcv_resize_algorithm::BMCV_INTER_LINEAR)
    .value("BMCV_INTER_BICUBIC", bmcv_resize_algorithm::BMCV_INTER_BICUBIC)
    .export_values();
#endif

  py::class_<Handle>(m, "Handle")
    .def(py::init<int>())
    .def("get_device_id", &Handle::get_device_id)
    .def("get_sn",        &Handle::get_sn)
    .def("get_target",    &Handle::get_target);
  py::class_<Tensor>(m, "Tensor")
    .def(py::init<Handle, py::array_t<float>&>(),   py::arg("handle"), py::arg("data"))
    .def(py::init<Handle, py::array_t<int8_t>&>(),  py::arg("handle"), py::arg("data"))
    .def(py::init<Handle, py::array_t<uint8_t>&>(), py::arg("handle"), py::arg("data"))
    .def(py::init<Handle, py::array_t<int32_t>&>(), py::arg("handle"), py::arg("data"))
    .def(py::init<Handle, py::array_t<float>&, bool, bool>(),   py::arg("handle"), py::arg("data"), py::arg("own_sys_data"), py::arg("own_dev_data")=true)
    .def(py::init<Handle, py::array_t<int8_t>&, bool, bool>(),  py::arg("handle"), py::arg("data"), py::arg("own_sys_data"), py::arg("own_dev_data")=true)
    .def(py::init<Handle, py::array_t<uint8_t>&, bool, bool>(), py::arg("handle"), py::arg("data"), py::arg("own_sys_data"), py::arg("own_dev_data")=true)
    .def(py::init<Handle, py::array_t<int32_t>&, bool, bool>(), py::arg("handle"), py::arg("data"), py::arg("own_sys_data"), py::arg("own_dev_data")=true)
    .def(py::init<Handle, const std::vector<int>&, bm_data_type_t, bool, bool>(),
      py::arg("handle"),py::arg("shape"),py::arg("dtype"),py::arg("own_sys_data")=false,py::arg("own_dev_data")=false)
    .def(py::init<Tensor&, std::vector<std::pair<int, int>> &, bool>(), py::arg("tensor"), py::arg("ranges"), py::arg("d2d_flag")=true)
    .def(py::init<Tensor&, std::vector<int> &, unsigned int, bool>(), 
      py::arg("src"), py::arg("shape"), py::arg("offset"), py::arg("no_copy") = true,
      "Create a tensor from another existing tensor. The created Tensor can "
      "reuse src's memory without copy.")
    .def("shape",                 &Tensor::shape)
    .def("reshape",               &Tensor::reshape)
    .def("own_sys_data",          &Tensor::own_sys_data)
    .def("own_dev_data",          &Tensor::own_dev_data)
    .def("device_id",             &Tensor::device_id)
    .def("asnumpy",               (py::object (Tensor::*)()) &Tensor::asnumpy)
    .def("pysys_data",            (py::array_t<long> (Tensor::*)()) &Tensor::pysys_data)
    .def("asnumpy",               (py::object (Tensor::*)(const std::vector<int>&)) &Tensor::asnumpy)
    .def("update_data",           (void (Tensor::*)(pybind11::array_t<float>&)) &Tensor::update_data)
    .def("update_data",           (void (Tensor::*)(pybind11::array_t<int8_t>&)) &Tensor::update_data)
    .def("update_data",           (void (Tensor::*)(pybind11::array_t<uint8_t>&)) &Tensor::update_data)
    .def("update_data",           (void (Tensor::*)(pybind11::array_t<int32_t>&))   &Tensor::update_data)
    .def("update_data",           (void (Tensor::*)(pybind11::array_t<uint16_t>&))   &Tensor::update_data)
    .def("scale_from",            (void (Tensor::*)(pybind11::array_t<float>&, float)) &Tensor::scale_from)
    //.def("scale_from",            (void (Tensor::*)(pybind11::array_t<int32_t>&, float)) &Tensor::scale_from)
    .def("scale_to",              (py::array_t<float> (Tensor::*)(float)) &Tensor::scale_to)
    .def("scale_to",              (py::array_t<float> (Tensor::*)(float, const std::vector<int>&)) &Tensor::scale_to)
    //.def("scale_to",              (py::array_t<int32_t> (Tensor::*)(float)) &Tensor::scale_to)
    //.def("scale_to",              (py::array_t<int32_t> (Tensor::*)(float, const std::vector<int>&)) &Tensor::scale_to)
    .def("sync_s2d",              (void (Tensor::*)()) &Tensor::sync_s2d, "move all data from system to device")
    .def("sync_s2d",              (void (Tensor::*)(int)) &Tensor::sync_s2d, "move size data from system to device")
    .def("sync_s2d",              (void (Tensor::*)(Tensor&,int,int,int)) &Tensor::sync_s2d, "copy size data from system to device")
    .def("sync_d2s",              (void (Tensor::*)()) &Tensor::sync_d2s, "move all data from device to system")
    .def("sync_d2s",              (void (Tensor::*)(int)) &Tensor::sync_d2s, "move size data from device to system")
    .def("sync_d2s",              (void (Tensor::*)(Tensor&,int,int,int)) &Tensor::sync_d2s, "copy size data from device to system")
    .def("sync_d2d",              (void (Tensor::*)(Tensor&,int,int,int)) &Tensor::sync_d2d, "copy size data from device to device")
    .def("sync_d2d_stride",       (void (Tensor::*)(Tensor&,int,int,int)) &Tensor::sync_d2d_stride, "copy size data from device to device")
    .def("memory_set",            (void (Tensor::*)(pybind11::object)) &Tensor::memory_set, "fill memory with a constant scalar")
    .def("zeros",                 &Tensor::zeros)
    .def("ones",                  &Tensor::ones)
    .def("dtype",                 &Tensor::dtype)
    .def("size",                  &Tensor::size)
    .def("element_size",          &Tensor::element_size)
    .def("nbytes",                &Tensor::nbytes)
    .def("is_dev_data_valid",     &Tensor::is_dev_data_valid)
    .def("is_sys_data_valid",     &Tensor::is_sys_data_valid)
    .def("__getitem__",
         [](Tensor &v, size_t i) -> py::object {
            if (i >= v.size()) throw py::index_error();
            if (NULL == v.sys_data()) throw MemoryError("Can not find System memory!");
            void* ptr_sys = v.sys_data();
            bm_data_type_t dtype_temp = v.dtype();
            if(dtype_temp == BM_FLOAT32){
              float* ptr = (float*)ptr_sys;
              return py::cast(ptr[i]);
            }else if(dtype_temp == BM_INT32){
              int* ptr = (int*)ptr_sys;
              return py::cast(ptr[i]);
            }else if(dtype_temp == BM_INT8){
              int8_t* ptr = (int8_t*)ptr_sys;
              return py::cast(ptr[i]);
            }else if(dtype_temp == BM_UINT8){
              uint8_t* ptr = (uint8_t*)ptr_sys;
              return py::cast(ptr[i]);
            }else if(dtype_temp == BM_UINT32){
              uint* ptr = (uint*)ptr_sys;
              return py::cast(ptr[i]);
            }
            throw py::type_error();
         }, py::call_guard<py::gil_scoped_release>())
    .def("__len__",             &Tensor::size)
    .def("dump_data",           &Tensor::dump_data,"dump tensor to file",py::arg("file_name"),py::arg("bin") = false);

  py::class_<Engine>(m, "Engine")
    .def(py::init<int>())
    .def(py::init<const Handle&>())
    .def(py::init<const std::string&, int, IOMode>())
    .def(py::init<py::bytes&, int, int, IOMode>())
    .def("load", (bool (Engine::*)(const std::string&))&Engine::load)
    .def("load", (bool (Engine::*)(py::bytes&, int))&Engine::load)
    .def("get_handle",            (Handle& (Engine::*)())&Engine::get_handle)
    .def("get_device_id",         &Engine::get_device_id)
    .def("get_graph_names",       &Engine::get_graph_names)
    .def("set_io_mode",           &Engine::set_io_mode)
    .def("graph_is_dynamic",      &Engine::graph_is_dynamic)
    .def("get_input_names",       &Engine::get_input_names)
    .def("get_output_names",      &Engine::get_output_names)
    .def("get_max_input_shapes",  &Engine::get_max_input_shapes)
    .def("get_input_shape",       &Engine::get_input_shape)
    .def("get_max_output_shapes", &Engine::get_max_output_shapes)
    .def("get_output_shape",      &Engine::get_output_shape)
    .def("get_input_dtype",       &Engine::get_input_dtype)
    .def("get_output_dtype",      &Engine::get_output_dtype)
    .def("get_input_scale",       &Engine::get_input_scale)
    .def("get_output_scale",      &Engine::get_output_scale)
    .def("create_input_tensors_map",  (std::map<std::string, Tensor&> (Engine::*) (const std::string&, int)) &Engine::create_input_tensors_map,
      py::arg("graph_name"), py::arg("create_mode")=-1)
    .def("create_output_tensors_map", (std::map<std::string, Tensor&> (Engine::*) (const std::string&, int)) &Engine::create_output_tensors_map,
      py::arg("graph_name"), py::arg("create_mode")=-1)
    .def("process", (void (Engine::*)(const std::string&, std::map<std::string, Tensor&>&, std::map<std::string, Tensor&>&, std::vector<int> core_list)) &Engine::process, py::call_guard<py::gil_scoped_release>(), py::arg("graph_name"), py::arg("input"), py::arg("output"), py::arg("core_list") = std::vector<int>{})
    .def("process", (void (Engine::*)(const std::string&, std::map<std::string, Tensor&>&, std::map<std::string, std::vector<int>>&, std::map<std::string, Tensor&>&, std::vector<int> core_list)) &Engine::process, py::call_guard<py::gil_scoped_release>(), py::arg("graph_name"), py::arg("input"), py::arg("input_shapes"), py::arg("output"), py::arg("core_list") = std::vector<int>{})
    .def("process", (std::map<std::string, pybind11::array_t<float>> (Engine::*)(const std::string&, std::map<std::string, pybind11::array_t<float>>&, std::vector<int> core_list)) &Engine::process, py::arg("graph_name"), py::arg("input_tensors"), py::arg("core_list") = std::vector<int>{});
  
#ifdef BUILD_ENGINELLM
  py::class_<EngineLLM>(m, "EngineLLM")
    .def(py::init<const std::string&, std::vector<int>>())
    .def(py::init<py::bytes&, size_t, std::vector<int>>())
    
    .def("get_device_ids",      &EngineLLM::get_device_ids)
    .def("get_graph_names",     &EngineLLM::get_graph_names)
    .def("get_addr_mode",       &EngineLLM::get_addr_mode)
    .def("get_is_dynamic",      &EngineLLM::get_is_dynamic)
    .def("get_stage_num",       &EngineLLM::get_stage_num)
    .def("get_input_num",       &EngineLLM::get_input_num)
    .def("get_output_num",      &EngineLLM::get_output_num)
    .def("get_input_name",      &EngineLLM::get_input_name)
    .def("get_output_name",     &EngineLLM::get_output_name)
    .def("get_input_names",     &EngineLLM::get_input_names)
    .def("get_output_names",    &EngineLLM::get_output_names)
    
    .def("get_input_tensors",   (std::map<int, Tensor*> (EngineLLM::*) (const std::string&, const int))                     &EngineLLM::get_input_tensors,
      py::arg("graph_name"), py::arg("stage") = 0, py::return_value_policy::reference)
    .def("get_input_tensors",   (std::map<int, Tensor*> (EngineLLM::*) (const std::string&, const std::string&, const int)) &EngineLLM::get_input_tensors,
      py::arg("graph_name"), py::arg("tensor_name"), py::arg("stage") = 0, py::return_value_policy::reference)
    .def("get_output_tensors",  (std::map<int, Tensor*> (EngineLLM::*) (const std::string&, const int))                     &EngineLLM::get_output_tensors,
      py::arg("graph_name"), py::arg("stage") = 0, py::return_value_policy::reference)
    .def("get_output_tensors",  (std::map<int, Tensor*> (EngineLLM::*) (const std::string&, const std::string&, const int)) &EngineLLM::get_output_tensors,
      py::arg("graph_name"), py::arg("tensor_name"), py::arg("stage") = 0, py::return_value_policy::reference)
    .def("get_input_tensor",    &EngineLLM::get_input_tensor,
      py::arg("graph_name"), py::arg("index"), py::arg("stage") = 0, py::return_value_policy::reference)
    .def("get_output_tensor",   &EngineLLM::get_output_tensor,
      py::arg("graph_name"), py::arg("index"), py::arg("stage") = 0, py::return_value_policy::reference)
    .def("create_max_input_tensors",     &EngineLLM::create_max_input_tensors,  py::return_value_policy::take_ownership,
      py::arg("graph_name")) // py::call_guard<py::gil_scoped_release>()
    .def("create_max_output_tensors",    &EngineLLM::create_max_output_tensors, py::return_value_policy::take_ownership,
      py::arg("graph_name")) // py::call_guard<py::gil_scoped_release>()

    .def("get_input_tensors_addrmode0",   (std::map<int, Tensor*> (EngineLLM::*) (const std::string&, const int)) &EngineLLM::get_input_tensors_addrmode0,
      py::arg("graph_name"), py::arg("stage") = 0, py::return_value_policy::reference)
    .def("get_output_tensors_addrmode0",  (std::map<int, Tensor*> (EngineLLM::*) (const std::string&, const int)) &EngineLLM::get_output_tensors_addrmode0,
      py::arg("graph_name"), py::arg("stage") = 0, py::return_value_policy::reference)

    .def("get_input_tensor_devid",    &EngineLLM::get_input_tensor_devid)
    .def("get_output_tensor_devid",   &EngineLLM::get_output_tensor_devid)
    .def("get_input_shape",     &EngineLLM::get_input_shape,
      py::arg("graph_name"), py::arg("index"), py::arg("stage") = 0)
    .def("get_output_shape",    &EngineLLM::get_output_shape,
      py::arg("graph_name"), py::arg("index"), py::arg("stage") = 0)
    .def("get_input_dtype",     &EngineLLM::get_input_dtype)
    .def("get_output_dtype",    &EngineLLM::get_output_dtype)
    .def("get_input_scale",     &EngineLLM::get_input_scale)
    .def("get_output_scale",    &EngineLLM::get_output_scale)

    .def("get_input_max_shape",     &EngineLLM::get_input_max_shape)
    .def("get_input_min_shape",     &EngineLLM::get_input_min_shape)
    .def("get_output_max_shape",    &EngineLLM::get_output_max_shape)
    .def("get_output_min_shape",    &EngineLLM::get_output_min_shape)

    .def("process", (void (EngineLLM::*) (const std::string&, std::map<int, Tensor*>&, std::map<int, Tensor*>&, const std::vector<int> &)) &EngineLLM::process,
      py::call_guard<py::gil_scoped_release>(),
      py::arg("graph_name"), py::arg("input"), py::arg("output"), py::arg("core_list") = std::vector<int>{0});
#endif

#ifdef USE_FFMPEG
  py::class_<Frame>(m, "Frame")
    .def(py::init<>())
    .def("get_height",           &Frame::get_height)
    .def("get_width",            &Frame::get_width);

  py::enum_<DecoderStatus>(m, "DecoderStatus")
    .value("NONE", DecoderStatus::NONE)
    .value("OPENED", DecoderStatus::OPENED)
    .value("CLOSED", DecoderStatus::CLOSED)
    .value("STATUS_MAX", DecoderStatus::STATUS_MAX)
    .export_values();

  py::class_<Decoder>(m, "Decoder")
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, bool>())
    .def(py::init<const std::string&, bool, int>())
    .def("is_opened",            &Decoder::is_opened)
    .def("get_frame_shape",      &Decoder::get_frame_shape)
    .def("read",                 (BMImage  (Decoder::*)(Handle&))            &Decoder::read, py::call_guard<py::gil_scoped_release>())
    .def("read",                 (int      (Decoder::*)(Handle&, BMImage&))  &Decoder::read, py::call_guard<py::gil_scoped_release>())
    // .def("read_",                (bm_image (Decoder::*)(Handle&))            &Decoder::read_)
    .def("read_",                (int      (Decoder::*)(Handle&, bm_image&)) &Decoder::read_, py::call_guard<py::gil_scoped_release>())
    .def("get_fps",              (float      (Decoder::*)() const)             &Decoder::get_fps)
    .def("get_pts_dts",          &Decoder::get_pts_dts)
    .def("enable_dump",          (void     (Decoder::*)(int)) &Decoder::enable_dump)
    .def("disable_dump",         &Decoder::disable_dump)
    .def("dump",                 (int (Decoder::*)(int, int, std::string&)) &Decoder::dump)
    .def("release",              &Decoder::release)
    .def("reconnect",            &Decoder::reconnect);
  py::class_<Encoder>(m, "Encoder")
    .def(py::init<>())
    .def(py::init< const std::string&, int, const std::string&, const std::string&, const std::string&>())
    .def(py::init< const std::string&, int, const std::string&, const std::string&, const std::string&, int>())
    .def(py::init< const std::string&, int, const std::string&, const std::string&, const std::string&, int, int>())
    .def(py::init< const std::string&, Handle&, const std::string&, const std::string&, const std::string&>())
    .def(py::init< const std::string&, Handle&, const std::string&, const std::string&, const std::string&, int>())
    .def(py::init< const std::string&, Handle&, const std::string&, const std::string&, const std::string&, int, int>())
    .def("is_opened",            &Encoder::is_opened)
    .def("pic_encode",          (int  (Encoder::*)(std::string&, BMImage&, std::vector<u_char>&))  &Encoder::pic_encode)
    .def("pic_encode",          (int  (Encoder::*)(std::string&, bm_image&, std::vector<u_char>&))  &Encoder::pic_encode)
#ifdef PYTHON
    .def("pic_encode",          (py::array_t<uint8_t>  (Encoder::*)(std::string&, BMImage&))  &Encoder::pic_encode)
    .def("pic_encode",          (py::array_t<uint8_t>  (Encoder::*)(std::string&, bm_image&)) &Encoder::pic_encode)
#endif
    .def("video_write",          (int  (Encoder::*)(BMImage&))                &Encoder::video_write)
    .def("video_write",          (int  (Encoder::*)(bm_image&))               &Encoder::video_write)
    .def("release",              &Encoder::release);
#endif

#ifdef USE_BMCV
  py::enum_<bm_image_format_ext>(m, "Format")
    .value("FORMAT_YUV420P",       bm_image_format_ext::FORMAT_YUV420P)
    .value("FORMAT_YUV422P",       bm_image_format_ext::FORMAT_YUV422P)
    .value("FORMAT_YUV444P",       bm_image_format_ext::FORMAT_YUV444P)
    .value("FORMAT_NV12",          bm_image_format_ext::FORMAT_NV12)
    .value("FORMAT_NV21",          bm_image_format_ext::FORMAT_NV21)
    .value("FORMAT_NV16",          bm_image_format_ext::FORMAT_NV16)
    .value("FORMAT_NV61",          bm_image_format_ext::FORMAT_NV61)
    .value("FORMAT_NV24",          bm_image_format_ext::FORMAT_NV24)
    .value("FORMAT_RGB_PLANAR",    bm_image_format_ext::FORMAT_RGB_PLANAR)
    .value("FORMAT_BGR_PLANAR",    bm_image_format_ext::FORMAT_BGR_PLANAR)
    .value("FORMAT_RGB_PACKED",    bm_image_format_ext::FORMAT_RGB_PACKED)
    .value("FORMAT_BGR_PACKED",    bm_image_format_ext::FORMAT_BGR_PACKED)
    .value("FORMAT_RGBP_SEPARATE", bm_image_format_ext::FORMAT_RGBP_SEPARATE)
    .value("FORMAT_BGRP_SEPARATE", bm_image_format_ext::FORMAT_BGRP_SEPARATE)
    .value("FORMAT_GRAY",          bm_image_format_ext::FORMAT_GRAY)
    .value("FORMAT_COMPRESSED",    bm_image_format_ext::FORMAT_COMPRESSED)
    .value("FORMAT_ARGB_PACKED",    bm_image_format_ext::FORMAT_ARGB_PACKED)
    .value("FORMAT_ABGR_PACKED",    bm_image_format_ext::FORMAT_ABGR_PACKED)
    .export_values();

  py::enum_<bm_image_data_format_ext>(m, "ImgDtype")
    .value("DATA_TYPE_EXT_FLOAT32",        bm_image_data_format_ext::DATA_TYPE_EXT_FLOAT32)
    .value("DATA_TYPE_EXT_1N_BYTE",        bm_image_data_format_ext::DATA_TYPE_EXT_1N_BYTE)
    .value("DATA_TYPE_EXT_1N_BYTE_SIGNED", bm_image_data_format_ext::DATA_TYPE_EXT_1N_BYTE_SIGNED)
#if !(BMCV_VERSION_MAJOR > 1)
    .value("DATA_TYPE_EXT_4N_BYTE",        bm_image_data_format_ext::DATA_TYPE_EXT_4N_BYTE)
    .value("DATA_TYPE_EXT_4N_BYTE_SIGNED", bm_image_data_format_ext::DATA_TYPE_EXT_4N_BYTE_SIGNED)
#endif
    .export_values();

  /* cannot be instantiated in python, the only use case is: Decoder.read_(handle, BMImageArray[i]) */
  py::class_<bm_image>(m, "bm_image")
    .def("width",                [](bm_image &img) -> int { return img.width;  })
    .def("height",               [](bm_image &img) -> int { return img.height; })
    .def("format",               [](bm_image &img) -> bm_image_format_ext { return img.image_format; })
    .def("dtype",                [](bm_image &img) -> bm_image_data_format_ext { return img.data_type; });
    
  py::class_<BMImage>(m, "BMImage")
    .def(py::init<>())
    .def(py::init<bm_image&>())
    .def(py::init<Handle&, int, int, bm_image_format_ext, bm_image_data_format_ext>())
    .def(py::init<Handle&, pybind11::buffer&, int, int, bm_image_format_ext, bm_image_data_format_ext, std::vector<int>, size_t>(),
      py::arg("handle"),py::arg("buffer"),py::arg("h"),py::arg("w"),py::arg("format"),py::arg("dtype")=DATA_TYPE_EXT_1N_BYTE, 
      py::arg("strides")=std::vector<int>(),py::arg("offset")=0)
    .def("width",                &BMImage::width)
    .def("height",               &BMImage::height)
    .def("format",               &BMImage::format)
    .def("dtype",                &BMImage::dtype)
    .def("data",                 (bm_image (BMImage::*)() const)      &BMImage::data )
#if defined(USE_BMCV) && defined(USE_OPENCV) && defined(PYTHON)
    .def("asmat",                &BMImage::asmat)
#endif
    .def("get_plane_num",        (int (BMImage::*)() const)           &BMImage::get_plane_num)
    .def("need_to_free",         (bool (BMImage::*)() const)          &BMImage::need_to_free)
    .def("empty_check",          (int (BMImage::*)() const)           &BMImage::empty_check)
    .def("get_device_id",        (int (BMImage::*)() const)           &BMImage::get_device_id)
    .def("get_handle",           (Handle (BMImage::*)())              &BMImage::get_handle)
    .def("align",                (int (BMImage::*)())                &BMImage::align)
    .def("check_align",          (int (BMImage::*)() const)          &BMImage::check_align)
    .def("unalign",              (int (BMImage::*)())                &BMImage::unalign)
    .def("check_contiguous_memory",              (int (BMImage::*)() const)          &BMImage::check_contiguous_memory)
    .def("asnumpy",              (pybind11::array (BMImage::*)() const)          &BMImage::asnumpy)
    ;
    
  declareBMImageArray<2>(m); // BMImageArray2D
  declareBMImageArray<3>(m); // BMImageArray3D
  declareBMImageArray<4>(m); // BMImageArray4D
  declareBMImageArray<8>(m); // BMImageArray8D
  declareBMImageArray<16>(m); // BMImageArray16D
  declareBMImageArray<32>(m); // BMImageArray32D
  declareBMImageArray<64>(m); // BMImageArray64D
  declareBMImageArray<128>(m); // BMImageArray128D
  declareBMImageArray<256>(m); // BMImageArray256D

  py::class_<PaddingAtrr>(m, "PaddingAtrr")
    .def(py::init<>())
    .def(py::init<unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int>())
    .def("set_stx",              (void (PaddingAtrr::*)(unsigned int))   &PaddingAtrr::set_stx)
    .def("set_sty",              (void (PaddingAtrr::*)(unsigned int))   &PaddingAtrr::set_sty)
    .def("set_w",                (void (PaddingAtrr::*)(unsigned int))   &PaddingAtrr::set_w)
    .def("set_h",                (void (PaddingAtrr::*)(unsigned int))   &PaddingAtrr::set_h)
    .def("set_r",                (void (PaddingAtrr::*)(unsigned int))   &PaddingAtrr::set_r)
    .def("set_g",                (void (PaddingAtrr::*)(unsigned int))   &PaddingAtrr::set_g)
    .def("set_b",                (void (PaddingAtrr::*)(unsigned int))   &PaddingAtrr::set_b);

  auto cls = py::class_<Bmcv>(m, "Bmcv")
    .def(py::init<Handle&>())
    .def("bm_image_to_tensor",  (void    (Bmcv::*)(BMImage&, Tensor&))       &Bmcv::bm_image_to_tensor)
    .def("bm_image_to_tensor",  (Tensor  (Bmcv::*)(BMImage&))                &Bmcv::bm_image_to_tensor)
    .def("tensor_to_bm_image",  (void    (Bmcv::*)(Tensor&, BMImage&, bool, std::string)) &Bmcv::tensor_to_bm_image, py::arg("tensor"), py::arg("img"), py::arg("bgr2rgb")=false, py::arg("layout")=std::string("nchw"))
    .def("tensor_to_bm_image",  (void    (Bmcv::*)(Tensor&, BMImage&, bm_image_format_ext)) &Bmcv::tensor_to_bm_image, py::arg("tensor"), py::arg("img"), py::arg("format"))
    .def("tensor_to_bm_image",  (BMImage (Bmcv::*)(Tensor&, bool, std::string))           &Bmcv::tensor_to_bm_image, py::arg("tensor"), py::arg("bgr2rgb")=false, py::arg("layout")=std::string("nchw"))
    .def("tensor_to_bm_image",  (BMImage (Bmcv::*)(Tensor&, bm_image_format_ext))         &Bmcv::tensor_to_bm_image, py::arg("tensor"), py::arg("format"))
    .def("crop_and_resize",     (BMImage (Bmcv::*)(BMImage&, int, int, int, int, int, int, bmcv_resize_algorithm)) &Bmcv::crop_and_resize,
      py::arg("input"),py::arg("crop_x0"), py::arg("crop_y0"), py::arg("crop_w"),py::arg("crop_h"),py::arg("resize_w"), py::arg("resize_h"), py::arg("resize_alg")=BMCV_INTER_NEAREST)
    .def("crop",                (BMImage (Bmcv::*)(BMImage&, int, int, int, int))           &Bmcv::crop)
    .def("crop",                (std::vector<BMImage> (Bmcv::*)(BMImage&, std::vector<std::vector<int>> ))    &Bmcv::crop)
    .def("resize",              (BMImage (Bmcv::*)(BMImage&, int, int, bmcv_resize_algorithm)) &Bmcv::resize,
      py::arg("input"),py::arg("resize_w"), py::arg("resize_h"), py::arg("resize_alg")=BMCV_INTER_NEAREST)
    .def("vpp_crop_and_resize", (BMImage (Bmcv::*)(BMImage&, int, int, int, int, int, int, bmcv_resize_algorithm)) &Bmcv::vpp_crop_and_resize,
      py::arg("input"), py::arg("crop_x0"), py::arg("crop_y0"), py::arg("crop_w"),py::arg("crop_h"),py::arg("resize_w"), py::arg("resize_h"), py::arg("resize_alg")=BMCV_INTER_NEAREST)
    .def("vpp_crop",            (BMImage (Bmcv::*)(BMImage&, int, int, int, int))           &Bmcv::vpp_crop)
    .def("vpp_resize",          (BMImage (Bmcv::*)(BMImage&, int, int, bmcv_resize_algorithm))      &Bmcv::vpp_resize,
      py::arg("input"), py::arg("resize_w"), py::arg("resize_h"), py::arg("resize_alg")=BMCV_INTER_NEAREST)
    .def("vpp_resize",          (int     (Bmcv::*)(BMImage&, BMImage&, int, int, bmcv_resize_algorithm))           &Bmcv::vpp_resize,
      py::arg("input"), py::arg("output"), py::arg("resize_w"), py::arg("resize_h"), py::arg("resize_alg")=BMCV_INTER_NEAREST)
    .def("vpp_crop_and_resize_padding", (BMImage (Bmcv::*)(BMImage&, int, int, int, int, int, int, PaddingAtrr&, bmcv_resize_algorithm)) &Bmcv::vpp_crop_and_resize_padding,
      py::arg("input"), py::arg("crop_x0"), py::arg("crop_y0"), py::arg("crop_w"),py::arg("crop_h"),py::arg("resize_w"), py::arg("resize_h"), py::arg("padding_in"), py::arg("resize_alg")=BMCV_INTER_NEAREST)
    .def("vpp_resize_padding",  (BMImage (Bmcv::*)(BMImage&, int, int, PaddingAtrr&, bmcv_resize_algorithm))   &Bmcv::vpp_resize_padding,
      py::arg("input"), py::arg("resize_w"), py::arg("resize_h"), py::arg("padding_in"), py::arg("resize_alg")=BMCV_INTER_NEAREST)
    .def("yuv2bgr",             (BMImage (Bmcv::*)(BMImage&))                               &Bmcv::yuv2bgr)
    .def("warp",                (BMImage (Bmcv::*)(BMImage&, const std::pair<std::tuple<float, float, float>, std::tuple<float, float, float>>&, int, bool))                     &Bmcv::warp,
      py::arg("input"), py::arg("matrix"), py::arg("use_bilinear")=0, py::arg("similar_to_opencv")=false)
    .def("warp",                (int (Bmcv::*)(BMImage&, BMImage&, const std::pair<std::tuple<float, float, float>, std::tuple<float, float, float>>&, int, bool))                     &Bmcv::warp,
      py::arg("input"), py::arg("output"), py::arg("matrix"), py::arg("use_bilinear")=0, py::arg("similar_to_opencv")=false)
    .def("warp_perspective",    (BMImage (Bmcv::*)(BMImage&, const std::tuple<std::pair<int, int>,std::pair<int, int>,std::pair<int, int>,std::pair<int, int>>&, int, int, bm_image_format_ext, bm_image_data_format_ext, int))  &Bmcv::warp_perspective,
      py::arg("input"),py::arg("coordinate"),py::arg("output_width"),py::arg("output_height"),py::arg("format")=FORMAT_BGR_PLANAR,py::arg("dtype")=DATA_TYPE_EXT_1N_BYTE,py::arg("use_bilinear")=0)
    .def("convert_to",          (BMImage (Bmcv::*)(BMImage&, const std::tuple<std::pair<float, float>, std::pair<float, float>, std::pair<float, float>>&))           &Bmcv::convert_to)
    .def("convert_to",          (int     (Bmcv::*)(BMImage&, BMImage&, const std::tuple<std::pair<float, float>, std::pair<float, float>, std::pair<float, float>>&)) &Bmcv::convert_to)
    .def("rectangle",           (int (Bmcv::*)(const BMImage&, int, int, int, int, const std::tuple<int, int, int>&, int))         &Bmcv::rectangle,
     py::arg("image"), py::arg("x0"), py::arg("y0"), py::arg("w"), py::arg("h"), py::arg("color"), py::arg("thickness") = 1)
    .def("rectangle",           (int (Bmcv::*)(const bm_image&, int, int, int, int, const std::tuple<int, int, int>&, int))        &Bmcv::rectangle,
     py::arg("image"), py::arg("x0"), py::arg("y0"), py::arg("w"), py::arg("h"), py::arg("color"), py::arg("thickness") = 1)
    .def("rectangle_",           (int (Bmcv::*)(const bm_image&, int, int, int, int, const std::tuple<int, int, int>&, int))       &Bmcv::rectangle_,
     py::arg("image"), py::arg("x0"), py::arg("y0"), py::arg("w"), py::arg("h"), py::arg("color"), py::arg("thickness") = 1)
    .def("fillRectangle",       (int (Bmcv::*)(const BMImage&, int, int, int, int, const std::tuple<int, int, int>&))              &Bmcv::fillRectangle)
    .def("fillRectangle",       (int (Bmcv::*)(const bm_image&, int, int, int, int, const std::tuple<int, int, int>&))             &Bmcv::fillRectangle)
    .def("fillRectangle_",       (int (Bmcv::*)(const bm_image&, int, int, int, int, const std::tuple<int, int, int>&))            &Bmcv::fillRectangle_)
    .def("putText",             (int (Bmcv::*)(const BMImage&, const std::string&, int, int, const std::tuple<int, int, int>&, float, int))       &Bmcv::putText,
     py::arg("image"), py::arg("text"), py::arg("x"), py::arg("y"), py::arg("color"), py::arg("fontScale"), py::arg("thickness") = 1)
    .def("putText",             (int (Bmcv::*)(const bm_image&, const std::string&, int, int, const std::tuple<int, int, int>&, float, int))      &Bmcv::putText,
     py::arg("image"), py::arg("text"), py::arg("x"), py::arg("y"), py::arg("color"), py::arg("fontScale"), py::arg("thickness") = 1)
    .def("putText_",             (int (Bmcv::*)(const bm_image&, const std::string&, int, int, const std::tuple<int, int, int>&, float, int))     &Bmcv::putText_,
     py::arg("image"), py::arg("text"), py::arg("x"), py::arg("y"), py::arg("color"), py::arg("fontScale"), py::arg("thickness") = 1)
    .def("imwrite",             (int (Bmcv::*)(const std::string&, const BMImage&))       &Bmcv::imwrite)
    .def("imwrite",             (int (Bmcv::*)(const std::string&, const bm_image&))      &Bmcv::imwrite)
    .def("imwrite_",             (int (Bmcv::*)(const std::string&, const bm_image&))     &Bmcv::imwrite_)
    .def("get_handle",          &Bmcv::get_handle)
    .def("get_bm_data_type",         (bm_data_type_t (Bmcv::*)(bm_image_data_format_ext)) &Bmcv::get_bm_data_type)
    .def("get_bm_image_data_format", (bm_image_data_format_ext (Bmcv::*)(bm_data_type_t)) &Bmcv::get_bm_image_data_format)
    .def("vpp_convert_format",          (BMImage (Bmcv::*)(BMImage&, bm_image_format_ext)) &Bmcv::vpp_convert_format, py::arg("input"), py::arg("image_format")=FORMAT_BGR_PLANAR)
    .def("vpp_convert_format",          (int     (Bmcv::*)(BMImage&, BMImage&))           &Bmcv::vpp_convert_format)
    .def("convert_format",              (BMImage (Bmcv::*)(BMImage&, bm_image_format_ext)) &Bmcv::convert_format, py::arg("input"), py::arg("image_format")=FORMAT_BGR_PLANAR)
    .def("convert_format",              (int     (Bmcv::*)(BMImage&, BMImage&))           &Bmcv::convert_format)
    .def("crop_and_resize_padding",     (BMImage (Bmcv::*)(BMImage&, int, int, int, int, int, int, PaddingAtrr&, bmcv_resize_algorithm))   &Bmcv::crop_and_resize_padding,
      py::arg("input"),py::arg("crop_x0"), py::arg("crop_y0"), py::arg("crop_w"),py::arg("crop_h"),py::arg("resize_w"), py::arg("resize_h"), py::arg("padding_in"), py::arg("resize_alg")=BMCV_INTER_NEAREST)
    .def("image_add_weighted",          (int     (Bmcv::*)(BMImage&, float, BMImage&, float, float, BMImage&))      &Bmcv::image_add_weighted)
    .def("image_add_weighted",          (BMImage (Bmcv::*)(BMImage&, float, BMImage&, float, float))                &Bmcv::image_add_weighted)
    .def("image_copy_to",               (int     (Bmcv::*)(BMImage&, BMImage&, int, int))  &Bmcv::image_copy_to,
      py::arg("input"),py::arg("output"), py::arg("start_x")=0, py::arg("start_y")=0)
    .def("image_copy_to_padding",       (int     (Bmcv::*)(BMImage&, BMImage&, unsigned int, unsigned int, unsigned int, int, int))  &Bmcv::image_copy_to_padding,
      py::arg("input"),py::arg("output"), py::arg("padding_r"), py::arg("padding_g"), py::arg("padding_b"),py::arg("start_x")=0, py::arg("start_y")=0)
    .def("nms",  (pybind11::array_t<float> (Bmcv::*)(pybind11::array_t<float>, float)) &Bmcv::nms)
    .def("drawPoint",                   (int (Bmcv::*)(const BMImage&, std::pair<int, int>, std::tuple<unsigned char, unsigned char, unsigned char>, int))    &Bmcv::drawPoint,
      py::arg("image"), py::arg("center"), py::arg("color"), py::arg("radius"))
    .def("drawPoint",                   (int (Bmcv::*)(const bm_image&, std::pair<int, int>, std::tuple<unsigned char, unsigned char, unsigned char>, int))   &Bmcv::drawPoint,
      py::arg("image"), py::arg("center"), py::arg("color"), py::arg("radius"))
    .def("drawPoint_",                   (int (Bmcv::*)(const bm_image&, std::pair<int, int>, std::tuple<unsigned char, unsigned char, unsigned char>, int))  &Bmcv::drawPoint_,
      py::arg("image"), py::arg("center"), py::arg("color"), py::arg("radius"))
    .def("drawLines",           &Bmcv::drawLines)
    .def("polylines",           &Bmcv::polylines,
      py::arg("img"), py::arg("pts"), py::arg("isClosed"), py::arg("color"), py::arg("thickness") = 1, py::arg("shift") = 0)
    .def("mosaic",              &Bmcv::mosaic)
    .def("transpose",          (int     (Bmcv::*)(BMImage&, BMImage&))      &Bmcv::transpose)
    .def("transpose",          (BMImage (Bmcv::*)(BMImage&))                &Bmcv::transpose)    
    .def("watermark_superpose",           &Bmcv::watermark_superpose)
    .def("gaussian_blur",               (BMImage (Bmcv::*)(BMImage&, int, int, float, float)) &Bmcv::gaussian_blur,
      py::arg("input"), py::arg("kw"), py::arg("kh"), py::arg("sigmaX"), py::arg("sigmaY")=0.0f)
    .def("Sobel",                       (int     (Bmcv::*)(BMImage&, BMImage&, int, int, int, float, float))      &Bmcv::Sobel,
      py::arg("input"), py::arg("output"), py::arg("dx"), py::arg("dy"), py::arg("ksize") = 3, py::arg("scale") = 1.0, py::arg("delta") = 0.0)
    .def("Sobel",                       (BMImage (Bmcv::*)(BMImage&, int, int, int, float, float))                &Bmcv::Sobel,
      py::arg("input"), py::arg("dx"), py::arg("dy"), py::arg("ksize") = 3, py::arg("scale") = 1, py::arg("delta") = 0) 
    .def("imdecode",            (BMImage (Bmcv::*)(py::bytes))&Bmcv::imdecode)
    .def("imencode",            (pybind11::array_t<uint8_t> (Bmcv::*)(std::string&, BMImage&)) &Bmcv::imencode)
    .def("imread",              (BMImage (Bmcv::*)(const std::string &))             &Bmcv::imread,
      py::arg("filename"))
  #if BMCV_VERSION_MAJOR > 1
    .def("stft",           (std::tuple<pybind11::array_t<float>, pybind11::array_t<float>> (Bmcv::*)(pybind11::array_t<float>, pybind11::array_t<float>, bool, bool, int, int, int, int)) &Bmcv::stft,
          "Short-Time Fourier Transform",
      py::arg("input_real"), py::arg("input_imag"), py::arg("realInput"), py::arg("normalize"), py::arg("n_fft"), py::arg("hop_len"), py::arg("pad_mode"), py::arg("win_mode"))
    .def("stft",           (std::tuple<Tensor, Tensor> (Bmcv::*)(Tensor&, Tensor&, bool, bool, int, int, int, int)) &Bmcv::stft,
          "Short-Time Fourier Transform",
      py::arg("input_real"), py::arg("input_imag"), py::arg("realInput"), py::arg("normalize"), py::arg("n_fft"), py::arg("hop_len"), py::arg("pad_mode"), py::arg("win_mode"))
    .def("istft",            (std::tuple<pybind11::array_t<float>, pybind11::array_t<float>> (Bmcv::*)(pybind11::array_t<float>, pybind11::array_t<float>, bool, bool, int, int, int, int)) &Bmcv::istft,
          "Inverse Short-Time Fourier Transform",
      py::arg("input_real"), py::arg("input_imag"), py::arg("realInput"), py::arg("normalize"), py::arg("L"), py::arg("hop_len"), py::arg("pad_mode"), py::arg("win_mode"))
    .def("istft",            (std::tuple<Tensor, Tensor> (Bmcv::*)(Tensor&, Tensor&, bool, bool, int, int, int, int)) &Bmcv::istft,
          "Inverse Short-Time Fourier Transform",
      py::arg("input_real"), py::arg("input_imag"), py::arg("realInput"), py::arg("normalize"), py::arg("L"), py::arg("hop_len"), py::arg("pad_mode"), py::arg("win_mode"))
  #endif
    .def("fft",                 (std::vector<Tensor> (Bmcv::*)(bool, Tensor&)) &Bmcv::fft)
    .def("fft",                 (std::vector<Tensor> (Bmcv::*)(bool, Tensor&, Tensor&)) &Bmcv::fft)
    // .def("convert_yuv420p_to_gray",                  &Bmcv::convert_yuv420p_to_gray)
    .def("convert_yuv420p_to_gray",     (int (Bmcv::*)(BMImage&, BMImage&))               &Bmcv::convert_yuv420p_to_gray,
      py::arg("input"), py::arg("output"))
    .def("convert_yuv420p_to_gray",     (int (Bmcv::*)(bm_image&, bm_image&))             &Bmcv::convert_yuv420p_to_gray,
      py::arg("input"), py::arg("output"))
    .def("convert_yuv420p_to_gray_",     (int (Bmcv::*)(bm_image&, bm_image&))            &Bmcv::convert_yuv420p_to_gray_,
      py::arg("input"), py::arg("output"))
#if BMCV_VERSION_MAJOR > 1
    .def("bmcv_overlay", &Bmcv::bmcv_overlay, py::arg("image"), py::arg("overlay_info"), py::arg("overlay_image"))
#endif
#if defined(USE_BMCV) && defined(USE_OPENCV) && defined(PYTHON)
    .def("mat_to_bm_image",     (BMImage (Bmcv::*)(pybind11::array_t<uint8_t>&)) &Bmcv::mat_to_bm_image, py::arg("input_array").noconvert())
    .def("mat_to_bm_image",     (int (Bmcv::*)(pybind11::array_t<uint8_t>&, BMImage&)) &Bmcv::mat_to_bm_image, py::arg("input_array").noconvert(), py::arg("img"));
#endif

  registerBMImageArrayFunctions<2>(cls);
  registerBMImageArrayFunctions<3>(cls);
  registerBMImageArrayFunctions<4>(cls);
  registerBMImageArrayFunctions<8>(cls);
  // registerBMImageArrayFunctions<16>(cls);
  // registerBMImageArrayFunctions<32>(cls);
  // registerBMImageArrayFunctions<64>(cls);
  // registerBMImageArrayFunctions<128>(cls);
  // registerBMImageArrayFunctions<256>(cls);

  m.def("base64_encode", [](Handle& handle, py::bytes input_bytes) {
         std::string str1 = (std::string)input_bytes;
         std::string str2;
         base64_enc(handle, str1.data(), str1.size(), str2);
         return py::bytes(str2);
      }, "SOPHON base64 encoder");

  m.def("base64_decode", [](Handle& handle, py::bytes encode_bytes) {
      std::string input = (std::string)encode_bytes;
      uint32_t input_size = input.size();
      uint32_t output_size = input_size/4*3;
      uint32_t real_size = 0;
      uint8_t* out_data = new uint8_t[output_size];
      base64_dec(handle, input.data(), input_size, out_data, &real_size);
      auto ob= pybind11::bytes((char*)out_data, real_size);
      delete [] out_data;
      return ob;
      }, "SOPHON base64 decoder");

    m.def("base64_encode_array", [](Handle& handle, py::array input_arr) {
auto byte_size = input_arr.dtype().itemsize();
        std::string str2;
        base64_enc(handle, input_arr.data(), input_arr.size() * byte_size, str2);
        return py::bytes(str2);
    }, "SOPHON base64 encoder");

    m.def("base64_decode_asarray",[](Handle& handle, py::bytes encode_arr_bytes, std::string array_type = "uint8") -> py::object{

        // 将bytes数据转换为string
        std::string input = (std::string)encode_arr_bytes;
        uint32_t input_size = input.size();
        
        // 根据数据类型，返回对应的numpy数据
        if (array_type == "float"){
            return get_ndarray<float>(handle,input_size,input);
        }else if (array_type == "uint8"){
            return get_ndarray<uint8_t>(handle,input_size,input);
        }else if (array_type == "int8"){
            return get_ndarray<int8_t>(handle,input_size,input);
        }else if (array_type == "int16"){
            return get_ndarray<int16_t>(handle,input_size,input);
        }else if (array_type == "int32"){
            return get_ndarray<int32_t>(handle,input_size,input);
        }else if (array_type == "int64"){
            return get_ndarray<int64_t>(handle,input_size,input);
        }else {
            SPDLOG_ERROR("ERROR array_type, supported [float,uint8,int8,int16,int32,int64] vs {}", array_type);
            throw SailBMImageError("not supported");
        }
    },"base64 decoder for numpy.array",
    py::arg("handle"),py::arg("encode_arr_bytes"),py::arg("array_type")="uint8");

    #if defined(USE_BMCV) && defined(PYTHON)
      py::class_<Decoder_RawStream>(m, "Decoder_RawStream")
            .def(py::init<int, std::string>())
            .def("read_", (int (Decoder_RawStream::*)(pybind11::bytes, bm_image&, bool)) &Decoder_RawStream::read_)
            .def("read", (int (Decoder_RawStream::*)(pybind11::bytes, BMImage&, bool)) &Decoder_RawStream::read)
            .def("release",              &Decoder_RawStream::release);
    #endif

#if BMCV_VERSION_MAJOR > 1
  py::enum_<bm_stitch_wgt_mode>(m, "blend_wgt_mode")
    .value("BM_STITCH_WGT_YUV_SHARE", bm_stitch_wgt_mode::BM_STITCH_WGT_YUV_SHARE)
    .value("BM_STITCH_WGT_UV_SHARE", bm_stitch_wgt_mode::BM_STITCH_WGT_UV_SHARE)
    .value("BM_STITCH_WGT_SEP", bm_stitch_wgt_mode::BM_STITCH_WGT_SEP)
    .export_values();

  py::class_<Blend>(m, "Blend")
    .def(py::init<int, std::vector<std::vector<short>>, std::vector<std::vector<short>>, std::vector<std::vector<string>>, bm_stitch_wgt_mode>())
    .def("process", (int     (Blend::*)(std::vector<BMImage*>&, BMImage&)) &Blend::process)
    .def("process", (BMImage (Blend::*)(std::vector<BMImage*>&)) &Blend::process);
#endif

#endif
    py::class_<MultiEngine>(m, "MultiEngine")
    .def(py::init<const std::string&, std::vector<int>, bool, int>(),
      py::arg("bmodel_path"),py::arg("tpu_ids"), py::arg("sys_out")=true, py::arg("graph_idx")=0)
    .def("get_device_ids",        &MultiEngine::get_device_ids)
    .def("get_graph_names",       &MultiEngine::get_graph_names)
    .def("get_input_names",       &MultiEngine::get_input_names)
    .def("get_output_names",      &MultiEngine::get_output_names)
    .def("get_input_shape",       &MultiEngine::get_input_shape)
    .def("get_output_shape",      &MultiEngine::get_output_shape)
    .def("set_print_flag",        &MultiEngine::set_print_flag)
    .def("set_print_time",        &MultiEngine::set_print_time)
    .def("process", (std::vector<std::map<std::string, Tensor*>> (MultiEngine::*)(std::vector<std::map<std::string, Tensor*>>&)) &MultiEngine::process)
    .def("process", (std::map<std::string, pybind11::array_t<float>> (MultiEngine::*)(std::map<std::string, pybind11::array_t<float>>&)) &MultiEngine::process);
#if defined(USE_FFMPEG) && defined(USE_OPENCV)
    py::class_<MultiDecoder>(m, "MultiDecoder")
    .def(py::init<int, int, int>(),
      py::arg("queue_size")=10,py::arg("tpu_id")=0, py::arg("discard_mode")=0)
    .def("set_read_timeout",        &MultiDecoder::set_read_timeout)
    .def("del_channel",             &MultiDecoder::del_channel)
    .def("clear_queue",             &MultiDecoder::clear_queue)
    .def("get_channel_fps",         &MultiDecoder::get_channel_fps)
    .def("reconnect",               &MultiDecoder::reconnect)
    .def("get_frame_shape",         &MultiDecoder::get_frame_shape)
    .def("set_local_flag",          &MultiDecoder::set_local_flag)
    .def("get_drop_num",            &MultiDecoder::get_drop_num)
    .def("reset_drop_num",           &MultiDecoder::reset_drop_num)
    .def("add_channel",       (int  (MultiDecoder::*)(const std::string&, int))   &MultiDecoder::add_channel,
      py::arg("file_path"), py::arg("frame_skip_num")=0)
    .def("get_channel_status",      &MultiDecoder::get_channel_status)
    .def("read",              (int  (MultiDecoder::*)(int,BMImage&,int))          &MultiDecoder::read,
      py::arg("channel_idx"), py::arg("image"), py::arg("read_mode")=0, py::call_guard<py::gil_scoped_release>())
    .def("read",              (BMImage  (MultiDecoder::*)(int))                   &MultiDecoder::read, py::call_guard<py::gil_scoped_release>())
    .def("read_",              (int  (MultiDecoder::*)(int,bm_image&,int))        &MultiDecoder::read_,
      py::arg("channel_idx"), py::arg("image"), py::arg("read_mode")=0, py::call_guard<py::gil_scoped_release>())
    .def("read_",              (bm_image  (MultiDecoder::*)(int))                 &MultiDecoder::read_, py::call_guard<py::gil_scoped_release>());

    py::enum_<sail_resize_type>(m, "sail_resize_type")
    .value("BM_RESIZE_VPP_NEAREST", sail_resize_type::BM_RESIZE_VPP_NEAREST)
    .value("BM_RESIZE_TPU_NEAREST", sail_resize_type::BM_RESIZE_TPU_NEAREST)
    .value("BM_RESIZE_TPU_LINEAR", sail_resize_type::BM_RESIZE_TPU_LINEAR)
    .value("BM_RESIZE_TPU_BICUBIC", sail_resize_type::BM_RESIZE_TPU_BICUBIC)
    .value("BM_PADDING_VPP_NEAREST", sail_resize_type::BM_PADDING_VPP_NEAREST)
    .value("BM_PADDING_TPU_NEAREST", sail_resize_type::BM_PADDING_TPU_NEAREST)
    .value("BM_PADDING_TPU_LINEAR", sail_resize_type::BM_PADDING_TPU_LINEAR)
    .value("BM_PADDING_TPU_BICUBIC", sail_resize_type::BM_PADDING_TPU_BICUBIC)
    .export_values();

    py::class_<ImagePreProcess>(m, "ImagePreProcess")
    .def(py::init<int, sail_resize_type, int, int, int, bool>(),
      py::arg("batch_size"), py::arg("resize_mode"), py::arg("tpu_id")=0, py::arg("queue_in_size")=20, py::arg("queue_out_size")=20, py::arg("use_mat_flag")=false)
    .def("SetResizeImageAtrr",      &ImagePreProcess::SetResizeImageAtrr)
    .def("SetPaddingAtrr",          (void  (ImagePreProcess::*)(int,int,int,int))  &ImagePreProcess::SetPaddingAtrr,
      py::arg("padding_b")=114, py::arg("padding_g")=114,py::arg("padding_r")=114,py::arg("align")=0)
    .def("SetConvertAtrr",          &ImagePreProcess::SetConvertAtrr)
    .def("PushImage",               &ImagePreProcess::PushImage, py::call_guard<py::gil_scoped_release>())
    .def("GetBatchData",            &ImagePreProcess::GetBatchData, py::call_guard<py::gil_scoped_release>())
    .def("set_print_flag",          &ImagePreProcess::set_print_flag)
    .def("exhausted",                    &ImagePreProcess::exhausted)
    .def("get_exhausted_flag",           &ImagePreProcess::get_exhausted_flag);

    py::class_<EngineImagePreProcess>(m, "EngineImagePreProcess")
    .def(py::init<const std::string&, int, bool, std::vector<int>>(),
      py::arg("bmodel_path"), py::arg("tpu_id"), py::arg("use_mat_output")=false,py::arg("core_list")=std::vector<int>{})
    .def("InitImagePreProcess",       (int (EngineImagePreProcess::*)(sail_resize_type, bool, int, int))  &EngineImagePreProcess::InitImagePreProcess, 
      py::arg("resize_mode"), py::arg("bgr2rgb")=false,py::arg("queue_in_size")=20,py::arg("queue_out_size")=20)
    .def("SetPaddingAtrr",            (int (EngineImagePreProcess::*)(int,int,int,int))  &EngineImagePreProcess::SetPaddingAtrr,
      py::arg("padding_b")=114, py::arg("padding_g")=114,py::arg("padding_r")=114,py::arg("align")=0)
    .def("SetConvertAtrr",            &EngineImagePreProcess::SetConvertAtrr)
    .def("PushImage",                 &EngineImagePreProcess::PushImage,py::call_guard<py::gil_scoped_release>())
    .def("GetBatchData",              &EngineImagePreProcess::GetBatchData_py,py::call_guard<py::gil_scoped_release>(), py::arg("need_d2s")=true)
    .def("GetBatchData_Npy",          &EngineImagePreProcess::GetBatchData_Npy)
    .def("GetBatchData_Npy2",         &EngineImagePreProcess::GetBatchData_Npy2)
    .def("get_graph_name",            &EngineImagePreProcess::get_graph_name)
    .def("get_input_width",           &EngineImagePreProcess::get_input_width)
    .def("get_input_height",          &EngineImagePreProcess::get_input_height)
    .def("get_output_names",          &EngineImagePreProcess::get_output_names)
    .def("get_output_shape",          &EngineImagePreProcess::get_output_shape)
    .def("exhausted",                      &EngineImagePreProcess::exhausted)
    .def("get_exhausted_flag",             &EngineImagePreProcess::get_exhausted_flag);

#ifdef USE_IPC
    py::class_<IPC>(m, "IPC")
    .def(py::init<bool, const std::string&, const std::string&, bool, int> (), py::arg("isSender_"), py::arg("pipe_path"), py::arg("final_path"), py::arg("usec2c")=false, py::arg("queue_len")=20)
    .def("sendTensor", (void (IPC::*)(Tensor&, int, int)) &IPC::sendTensor)
    .def("receiveTensor", (std::tuple<Tensor, int, int> (IPC::*)()) &IPC::receiveTensor)
    .def("sendBMImage", (void (IPC::*)(BMImage&, int, int))  &IPC::sendBMImage)
    .def("receiveBMImage", (std::tuple<BMImage, int, int> (IPC::*)())  &IPC::receiveBMImage);
#endif // USE_IPC

    py::class_<algo_yolov5_post_1output>(m, "algo_yolov5_post_1output")
    .def(py::init<const std::vector<int>&, int, int, int, bool, bool> (), 
      py::arg("shape"),py::arg("network_w")=640,py::arg("network_h")=640,py::arg("max_queue_size")=20,py::arg("input_use_multiclass_nms")=true,py::arg("agnostic")=false)
    .def("push_data",    &algo_yolov5_post_1output::push_data)
    .def("push_npy",     &algo_yolov5_post_1output::push_npy)
    .def("get_result",   &algo_yolov5_post_1output::get_result)
    .def("get_result_npy",   &algo_yolov5_post_1output::get_result_npy);

    py::class_<algo_yolov8_post_1output_async>(m, "algo_yolov8_post_1output_async")
    .def(py::init<const std::vector<int>&, int, int, int, bool, bool> (), 
      py::arg("shape"),py::arg("network_w")=640,py::arg("network_h")=640,py::arg("max_queue_size")=20,py::arg("input_use_multiclass_nms")=true,py::arg("agnostic")=false)
    .def("push_data",    &algo_yolov8_post_1output_async::push_data)
    .def("push_npy",     &algo_yolov8_post_1output_async::push_npy)
    .def("get_result",   &algo_yolov8_post_1output_async::get_result)
    .def("get_result_npy",   &algo_yolov8_post_1output_async::get_result_npy);

    py::class_<algo_yolov8_post_cpu_opt_1output_async>(m, "algo_yolov8_post_cpu_opt_1output_async")
    .def(py::init<const std::vector<int>&, int, int, int, bool, bool> (), 
      py::arg("shape"),py::arg("network_w")=640,py::arg("network_h")=640,py::arg("max_queue_size")=20,py::arg("input_use_multiclass_nms")=true,py::arg("agnostic")=false)
    .def("push_data",    &algo_yolov8_post_cpu_opt_1output_async::push_data)
    .def("push_npy",     &algo_yolov8_post_cpu_opt_1output_async::push_npy)
    .def("get_result",   &algo_yolov8_post_cpu_opt_1output_async::get_result)
    .def("get_result_npy",   &algo_yolov8_post_cpu_opt_1output_async::get_result_npy);

    py::class_<algo_yolov5_post_3output>(m, "algo_yolov5_post_3output")
    .def(py::init<const std::vector<std::vector<int>>&, int, int, int,bool,bool> (), 
      py::arg("shape"),py::arg("network_w")=640,py::arg("network_h")=640,py::arg("max_queue_size")=20,py::arg("input_use_multiclass_nms")=true,py::arg("agnostic")=false)
    .def("reset_anchors",    &algo_yolov5_post_3output::reset_anchors)
    .def("push_data",    &algo_yolov5_post_3output::push_data)
    // .def("push_npy",     &algo_yolov5_post_3output::push_npy)
    .def("get_result",   &algo_yolov5_post_3output::get_result)
    .def("get_result_npy",   &algo_yolov5_post_3output::get_result_npy);

    py::class_<algo_yolov5_post_cpu_opt_async>(m, "algo_yolov5_post_cpu_opt_async")
    .def(py::init<const std::vector<std::vector<int>>&, int, int, int, bool> (), 
      py::arg("shape"),py::arg("network_w")=640,py::arg("network_h")=640,py::arg("max_queue_size")=20,py::arg("use_multiclass_nms")=true)
    .def("reset_anchors",    &algo_yolov5_post_cpu_opt_async::reset_anchors)
    .def("push_data",    &algo_yolov5_post_cpu_opt_async::push_data)
    // .def("push_npy",     &algo_yolov5_post_cpu_opt_async::push_npy)
    .def("get_result",   &algo_yolov5_post_cpu_opt_async::get_result)
    .def("get_result_npy",   &algo_yolov5_post_cpu_opt_async::get_result_npy);

    py::class_<algo_yolox_post>(m, "algo_yolox_post")
    .def(py::init<const std::vector<int>&, int, int, int> (), 
      py::arg("shape"),py::arg("network_w")=640,py::arg("network_h")=640,py::arg("max_queue_size")=20)
    .def("push_data",    &algo_yolox_post::push_data)
    .def("push_npy",     &algo_yolox_post::push_npy)
    .def("get_result",   &algo_yolox_post::get_result)
    .def("get_result_npy",  &algo_yolox_post::get_result_npy);

    py::class_<algo_yolov5_post_cpu_opt>(m, "algo_yolov5_post_cpu_opt")
    .def(py::init<const std::vector<std::vector<int>>&, int, int> (), 
      py::arg("shapes"),
      py::arg("network_w")=640,
      py::arg("network_h")=640)
    .def("process",           
        (std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>>  
          (algo_yolov5_post_cpu_opt::*)(std::vector<TensorPTRWithName>&, std::vector<int>&, std::vector<int>&, std::vector<float>&, std::vector<float>&, bool, bool))   
          &algo_yolov5_post_cpu_opt::process)
    .def("process",           
        (std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>>  
          (algo_yolov5_post_cpu_opt::*)(std::map<std::string, Tensor&>&, std::vector<int>&, std::vector<int>&, std::vector<float>&, std::vector<float>&, bool, bool))   
          &algo_yolov5_post_cpu_opt::process)
    .def("reset_anchors",     &algo_yolov5_post_cpu_opt::reset_anchors);

    py::class_<algo_yolov8_seg_post_tpu_opt>(m, "algo_yolov8_seg_post_tpu_opt")
    .def(py::init<std::string, int, const std::vector<int>&, const std::vector<int>&, int, int> (),
      py::arg("bmodel_file"),
      py::arg("dev_id"),
      py::arg("detection_shape"),
      py::arg("segmentation_shape"),
      py::arg("network_h"),
      py::arg("network_w"))
    .def("process",
        (py::list
          (algo_yolov8_seg_post_tpu_opt::*)(TensorPTRWithName&, TensorPTRWithName&, int&, int&, float&, float&, bool, bool))
          &algo_yolov8_seg_post_tpu_opt::process)
    .def("process",
        (py::list
          (algo_yolov8_seg_post_tpu_opt::*)(std::map<std::string, Tensor&>&, std::map<std::string, Tensor&>&, int&, int&, float&, float&, bool, bool))
          &algo_yolov8_seg_post_tpu_opt::process);

    py::class_<sort_tracker_controller>(m, "sort_tracker_controller")
    .def(py::init< float, int, int> (),
          py::arg("max_iou_distance") = 0.7, py::arg("max_age") = 30, py::arg("n_init") = 3)
    .def("process",
        (std::vector<std::tuple<int, int, int, int, int, float, int>> 
            (sort_tracker_controller::*)(const std::vector<std::tuple<int, int, int, int ,int, float>>&)) 
            &sort_tracker_controller::process);
    
     py::class_<sort_tracker_controller_async>(m, "sort_tracker_controller_async")
    .def(py::init<float, int, int, int, int> (),
           py::arg("max_iou_distance") = 0.7, py::arg("max_age") = 30, py::arg("n_init") = 3, py::arg("input_queue_size") = 10, py::arg("result_queue_size") = 10)
    .def("push_data",
        (int
            (sort_tracker_controller_async::*)(const std::vector<std::tuple<int, int, int, int ,int, float>>&))
            &sort_tracker_controller_async::push_data)
    .def("get_result_npy",   &sort_tracker_controller_async::get_result_npy);

    py::class_<deepsort_tracker_controller>(m, "deepsort_tracker_controller")
    .def(py::init<float, int, int, float, int, int> (),
          py::arg("max_cosine_distance"), py::arg("nn_budget"), py::arg("k_feature_dim"), py::arg("max_iou_distance") = 0.7, py::arg("max_age") = 30, py::arg("n_init") = 3)
    .def("process",
        (std::vector<std::tuple<int, int, int, int, int, float, int>> 
            (deepsort_tracker_controller::*)(const std::vector<std::tuple<int, int, int, int ,int, float>>&, vector<Tensor>&)) 
            &deepsort_tracker_controller::process)
    .def("process",
        (std::vector<std::tuple<int, int, int, int, int, float, int>> 
            (deepsort_tracker_controller::*)(const std::vector<std::tuple<int, int, int, int ,int, float>>&, vector<vector<float>>&)) 
            &deepsort_tracker_controller::process);

    py::class_<deepsort_tracker_controller_async>(m, "deepsort_tracker_controller_async")
    .def(py::init<float, int, int, float, int, int, int> (),
          py::arg("max_cosine_distance"), py::arg("nn_budget"), py::arg("k_feature_dim"), py::arg("max_iou_distance") = 0.7, py::arg("max_age") = 30, py::arg("n_init") = 3, py::arg("queue_size") = 10)
    .def("push_data",
        (int
            (deepsort_tracker_controller_async::*)(const std::vector<std::tuple<int, int, int, int ,int, float>>&, vector<Tensor>&))
            &deepsort_tracker_controller_async::push_data)
    .def("push_data",
        (int
            (deepsort_tracker_controller_async::*)(const std::vector<std::tuple<int, int, int, int ,int, float>>&, vector<vector<float>>&))
            &deepsort_tracker_controller_async::push_data)
    .def("get_result_npy",   &deepsort_tracker_controller_async::get_result_npy)
    .def("set_processing_timer",  &deepsort_tracker_controller_async::set_processing_timer);

    py::class_<bytetrack_tracker_controller>(m, "bytetrack_tracker_controller")
    .def(py::init<int, int> (),
          py::arg("frame_rate") = 30, py::arg("track_buffer") = 30)
    .def("process",    
        (std::vector<std::tuple<int, int, int, int, int, float, int>> 
            (bytetrack_tracker_controller::*)(const std::vector<std::tuple<int, int, int, int ,int, float>>&)) 
            &bytetrack_tracker_controller::process);

    py::class_<TensorPTRWithName>(m, "TensorPTRWithName")
    .def("get_name",    [](TensorPTRWithName &tensor_p) -> std::string { return tensor_p.name;  })
    .def("get_data",    [](TensorPTRWithName &tensor_p) -> sail::Tensor { return *tensor_p.data; });

    m.def("ReleaseTensorPtr",   &ReleaseTensorPtr, py::call_guard<py::gil_scoped_release>());
    m.def("CreateTensorPTRWithName", [](std::string name, Handle handle, pybind11::array_t<float>& data, bool own_sys_data, bool own_dev_data) {
          sail::Tensor* data_ptr = new sail::Tensor(handle, data, own_sys_data, own_dev_data);
          return sail::TensorPTRWithName(name,data_ptr);
      }, py::arg("name"), py::arg("handle"), py::arg("data"), py::arg("own_sys_data")=false, py::arg("own_dev_data")=true);
    m.def("CreateTensorPTRWithName", [](std::string name, Handle handle, pybind11::array_t<int8_t>& data, bool own_sys_data, bool own_dev_data) {
          sail::Tensor* data_ptr = new sail::Tensor(handle, data, own_sys_data, own_dev_data);
          return sail::TensorPTRWithName(name,data_ptr);
      }, py::arg("name"), py::arg("handle"), py::arg("data"), py::arg("own_sys_data")=false, py::arg("own_dev_data")=true);
    m.def("CreateTensorPTRWithName", [](std::string name, Handle handle, pybind11::array_t<uint8_t>& data, bool own_sys_data, bool own_dev_data) {
          sail::Tensor* data_ptr = new sail::Tensor(handle, data, own_sys_data, own_dev_data);
          return sail::TensorPTRWithName(name,data_ptr);
      }, py::arg("name"), py::arg("handle"), py::arg("data"), py::arg("own_sys_data")=false, py::arg("own_dev_data")=true);
    m.def("CreateTensorPTRWithName", [](std::string name, Handle handle, pybind11::array_t<int32_t>& data, bool own_sys_data, bool own_dev_data) {
          sail::Tensor* data_ptr = new sail::Tensor(handle, data, own_sys_data, own_dev_data);
          return sail::TensorPTRWithName(name,data_ptr);
      }, py::arg("name"), py::arg("handle"), py::arg("data"), py::arg("own_sys_data")=false, py::arg("own_dev_data")=true);
 
#ifndef TPUKERNRL_OFF
    py::class_<tpu_kernel_api_yolov5_detect_out>(m, "tpu_kernel_api_yolov5_detect_out")
    .def(py::init<int, const std::vector<std::vector<int>>&, int, int, std::string> (), 
      py::arg("device_id"),
      py::arg("shapes"),
      py::arg("network_w")=640,
      py::arg("network_h")=640,
      py::arg("module_file")="/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so")
    .def("process",           
        (std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>>  
          (tpu_kernel_api_yolov5_detect_out::*)(std::vector<TensorPTRWithName>&, float, float, bool))   
          &tpu_kernel_api_yolov5_detect_out::process, py::arg("input"), py::arg("dete_threshold"),py::arg("nms_threshold"),py::arg("release_input")=false)
    .def("process",           
        (std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>>  
          (tpu_kernel_api_yolov5_detect_out::*)(std::map<std::string, Tensor&>&, float, float, bool))   
          &tpu_kernel_api_yolov5_detect_out::process, py::arg("input"), py::arg("dete_threshold"),py::arg("nms_threshold"),py::arg("release_input")=false)
    .def("reset_anchors",     &tpu_kernel_api_yolov5_detect_out::reset_anchors);

    py::class_<tpu_kernel_api_openpose_part_nms>(m, "tpu_kernel_api_openpose_part_nms")
    .def(py::init<int, int, std::string> (), 
      py::arg("device_id"),
      py::arg("network_c"),
      py::arg("module_file")="/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so")
    .def("process",           
        (std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<float>>, std::vector<std::vector<int>>>   
          (tpu_kernel_api_openpose_part_nms::*)(TensorPTRWithName&, std::vector<int>&, std::vector<float>&, std::vector<int>&))   
          &tpu_kernel_api_openpose_part_nms::process)
    .def("process",           
        (std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<float>>, std::vector<std::vector<int>>>  
          (tpu_kernel_api_openpose_part_nms::*)(std::map<std::string, Tensor&>&, std::vector<int>&, std::vector<float>&, std::vector<int>&))   
          &tpu_kernel_api_openpose_part_nms::process)
    .def("reset_network_c",     &tpu_kernel_api_openpose_part_nms::reset_network_c);

    py::class_<tpu_kernel_api_yolov5_out_without_decode>(m, "tpu_kernel_api_yolov5_out_without_decode")
    .def(py::init<int, const std::vector<int>&, int, int, std::string> (), 
      py::arg("device_id"),
      py::arg("shapes"),
      py::arg("network_w")=640,
      py::arg("network_h")=640,
      py::arg("module_file")="/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so")
    .def("process",           
        (std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>>  
          (tpu_kernel_api_yolov5_out_without_decode::*)(TensorPTRWithName&, float, float))   
          &tpu_kernel_api_yolov5_out_without_decode::process)
    .def("process",           
        (std::vector<std::vector<std::tuple<int, int, int, int ,int, float>>>  
          (tpu_kernel_api_yolov5_out_without_decode::*)(std::map<std::string, Tensor&>&, float, float))   
          &tpu_kernel_api_yolov5_out_without_decode::process);
#endif
  py::class_<Perf>(m, "Perf")
    .def(py::init<const std::string&, std::vector<int>, int, IOMode, int, bool>()
      ,py::arg("bmodel_path"), py::arg("tpu_ids"), py::arg("max_que_size"),py::arg("mode")=SYSO, py::arg("thread_count")=2, py::arg("free_input")=false)
    .def("PushTensor",        &Perf::PushTensor, py::call_guard<py::gil_scoped_release>())
    .def("SetEnd",            &Perf::SetEnd)
    .def("get_graph_names",   &Perf::get_graph_names)
    .def("get_input_names",   &Perf::get_input_names)
    .def("get_output_names",  &Perf::get_output_names)
    .def("get_input_shape",   &Perf::get_input_shape)
    .def("get_input_scale",   &Perf::get_input_scale)
    .def("get_input_dtype",   &Perf::get_input_dtype)
    .def("GetResult",         &Perf::GetResult, py::call_guard<py::gil_scoped_release>());

    py::class_<DecoderImages>(m, "DecoderImages")
      .def(py::init<std::vector<std::string>&,int,int>())
      .def("setResizeAttr",    (int  (DecoderImages::*)(int,int,bmcv_resize_algorithm))  &DecoderImages::setResizeAttr,
          py::arg("width"), py::arg("height"),py::arg("resize_alg")=BMCV_INTER_LINEAR)
      .def("start",           &DecoderImages::start)
      .def("read",            &DecoderImages::read)
      .def("get_schedule",    &DecoderImages::get_schedule)
      .def("stop",            &DecoderImages::stop);


    m.def("argmax",   &argmax, py::call_guard<py::gil_scoped_release>());
    m.def("argmin",   &argmin, py::call_guard<py::gil_scoped_release>());
#endif 
    m.def("nms_rotated", &nms_rotated, py::arg("boxes"), py::arg("scores"), py::arg("threshold"));
}
