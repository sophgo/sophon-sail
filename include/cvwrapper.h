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

/** @file     cvwrapper.h
 *  @brief    Header file of BMCV and BMDECODE
 *  @version  2.0.3
 *  @date     2019-12-27
 */

#pragma once
#ifdef USE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavdevice/avdevice.h>
#include <libswscale/swscale.h>
}
#else
#ifndef FFALIGN
#define FFALIGN(x, a)(((x) + (a)-1) & ~((a)-1))
#endif
#endif

#ifdef USE_BMCV
#include <bmruntime_interface.h>
#include <bmlib_runtime.h>
#include <bmcv_api_ext.h>

#if !((BMCV_VERSION_MAJOR == 2) && !defined(BMCV_VERSION_MINOR))
#include <bmcv_api.h>
#endif

#endif

#ifdef PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#endif

#include <string>
#include <vector>
#include "tensor.h"
#include <iostream>
using namespace std;

namespace cv { class Mat; }

/// Namespace containing all symbols from the sail library.
namespace sail {

/**
 * @brief Set Decoder environment, must set befor Decoder Constructor, else use default values
 * refcounted_frames,extra_frame_buffer_num,rtsp_transport,stimeout,
 * rtsp_flags, buffer_size, max_delay, probesize, analyzeduration
 */
int DECL_EXPORT set_decoder_env(std::string env_name, std::string env_value);

class PaddingAtrr {
public:
    explicit PaddingAtrr(){};
    explicit PaddingAtrr(unsigned int crop_start_x,
        unsigned int crop_start_y,
        unsigned int crop_width,
        unsigned int crop_height,
        unsigned char padding_value_r,
        unsigned char padding_value_g,
        unsigned char padding_value_b);
    PaddingAtrr(const PaddingAtrr& other);

    void set_stx(unsigned int stx);
    void set_sty(unsigned int sty);
    void set_w(unsigned int w);
    void set_h(unsigned int h);
    void set_r(unsigned int r);
    void set_g(unsigned int g);
    void set_b(unsigned int b);

    unsigned int    dst_crop_stx;
    unsigned int    dst_crop_sty;
    unsigned int    dst_crop_w;
    unsigned int    dst_crop_h;
    unsigned char   padding_r;
    unsigned char   padding_g;
    unsigned char   padding_b;
};

#ifdef USE_OPENCV
/**
 * @brief Convert bm_data_type_t to opencv depth.
 *
 * @param dtype bm_data_type_t
 * @return opencv depth
 */
int get_cv_depth(bm_data_type_t dtype);

/**
 * @brief Convert data from cv::Mat to sail::Tensor.
 *
 * @param mat    Data with type cv:Mat
 * @param tensor Data with type sail::Tensor
 */
void mat_to_tensor(cv::Mat& mat, Tensor& tensor);

/**
 * @brief Convert data from vector of cv::Mat to sail::Tensor.
 *
 * @param mat    Data with type vector of cv:Mat
 * @param tensor Data with type sail::Tensor
 */
void mat_to_tensor(std::vector<cv::Mat>& mats, Tensor& tensor);
#endif

#ifdef USE_BMCV

class BMImage;

/**
 * @brief The wrapper of bmcv bm_image in sail for python api.
 */
class DECL_EXPORT BMImage {
 public:
  /**
   * @brief The default Constructor.
   */
  BMImage();
  /**
   * @brief Construct BMImage with bm_image.
   *
   * @param img Init input bm_image
   */
  BMImage(bm_image &img);
  /**
   * @brief The BMImage Constructor.
   *
   * @param handle A Handle instance
   * @param h      Image height
   * @param w      Image width
   * @param format Image format
   * @param dtype  Data type
   */
  BMImage(
      Handle&                  handle,
      int                      h,
      int                      w,
      bm_image_format_ext      format,
      bm_image_data_format_ext dtype);
  /**
   * @brief The BMImage Constructor.
   *
   * @param handle A Handle instance
   * @param h      Image height
   * @param w      Image width
   * @param format Image format
   * @param dtype  Data type
   */
  BMImage(
      Handle&                  handle,
      int                      h,
      int                      w,
      bm_image_format_ext      format,
      bm_image_data_format_ext dtype,
      int                      *stride);

  /**
   * @brief The BMImage Constructor. Create BMImage by an existing buffer.
   *
   * @param handle  A Handle instance
   * @param buffer  A buffer containing rawdata
   * @param h       Image height
   * @param w       Image width
   * @param format  Image format
   * @param dtype   Data type. Default: DATA_TYPE_EXT_1N_BYTE, i.e., uint8
   * @param strides Strides for each plane. Default: an empty std::vector<int>
   * @param offset  Start reading the buffer from this offset (in bytes); default: 0
   */
  BMImage(
      Handle                   &handle,
      void*                    buffer,
      int                      h,
      int                      w,
      bm_image_format_ext      format,
      bm_image_data_format_ext dtype = DATA_TYPE_EXT_1N_BYTE,
      std::vector<int>         strides = {},
      size_t                   offset = 0,
      int                      size = -1);

  /**
   * @brief The BMImage Constructor. Create BMImage by a bytes object.
   *
   * @param handle  A Handle instance
   * @param buffer  A buffer containing rawdata
   * @param h       Image height
   * @param w       Image width
   * @param format  Image format
   * @param dtype   Data type. Default: DATA_TYPE_EXT_1N_BYTE, i.e., uint8
   * @param strides Strides for each plane. Default: an empty std::vector<int>
   * @param offset  Start reading the buffer from this offset (in bytes); default: 0
   */
  #ifdef PYTHON
  BMImage(
      Handle&                     handle,
      pybind11::buffer            &buffer,
      int                         h,
      int                         w,
      bm_image_format_ext         format,
      bm_image_data_format_ext    dtype = DATA_TYPE_EXT_1N_BYTE,
      std::vector<int>            strides = {},
      size_t                      offset = 0);
  #endif // PYTHON

  /**
   * @brief The copy constructor of BMImage.
   */
  BMImage(BMImage &&other);
  /**
   * @brief The assignment function of BMImage.
   */
  BMImage& operator=(BMImage &&other);
  BMImage& operator=(bm_image &&other);
  virtual ~BMImage();
  /**
   * @brief Get inner bm_image
   *
   * @return The inner bm_image
   */
  bm_image& data();
  bm_image data() const;
  /**
   * @brief Get the img width.
   *
   * @return the width of img
   */
  int width() const;
  /**
   * @brief Get the img height.
   *
   * @return the height of img
   */
  int height() const;
  /**
   * @brief Get the img format.
   *
   * @return the format of img
   */
  bm_image_format_ext format() const;
  /**
   * @brief Get the img data type.
   *
   * @return the data type of img
   */
  bm_image_data_format_ext dtype() const;

  /**
   * @brief Get device id of this image.
   *
   * @return Device id.
   */
  int get_device_id() const;

  /**
   * @brief Get Handle of this BMImage.
   *
   * @return Handle.
   */
  Handle get_handle();

  bool need_to_free() const;
  int empty_check() const;
  int get_plane_num() const;
  void detach();
  /**
  * @brief Align the bm_image to 64 bytes.
  *
  * @return ret.
  */
  int align();
  /**
  * @brief Check if the bm_image aligned.
  *
  * @return ret.
  */
  int check_align() const;
  /**
  * @brief Unalign the bm_image to source bm_image.
  *
  * @return ret.
  */
  int unalign();
  /**
  * @brief Check if the bm_image's memory contiguous.
  *
  * @return ret.
  */
  int check_contiguous_memory() const;

#ifdef PYTHON
  /**
  * @brief Convert BMImage to numpy.ndarray containing raw data, 
  *        without color space convert.
  *
  * @return numpy.ndarray containing a copy of BMImage’s raw data.
  */
  pybind11::array asnumpy() const;
#endif

  /**
  * @brief set IPC flag. Normally false, only set to true if IPC::sendBMImage method is called.
  *
  * @return void.
  */
  void set_ipc_flag(bool f);
  
  /**
   * @brief get pts and dts
   * @return the pts and dts of inner frame.
  */
  vector<double> get_pts_dts();

#if defined(USE_BMCV) && defined(USE_OPENCV) && defined(PYTHON)
  pybind11::array_t<uint8_t> asmat();
#endif
 protected:
  /// inner bm_image
  void reset(int w, int h);

 private:
  class BMImage_CC;
  class BMImage_CC* const _impl;

  BMImage(const BMImage &other) = delete;
  BMImage& operator=(const BMImage &other) = delete;

  void create(
      Handle&                  handle,
      int                      h,
      int                      w,
      bm_image_format_ext      format,
      bm_image_data_format_ext dtype,
      int                      *stride=NULL);
  void destroy();
  void allocate();
  bool is_created() const;

#ifdef USE_OPENCV
  void cache_ost_mat(std::shared_ptr<cv::Mat>& ostrmat);
#endif

  void set_pts_dts(double pts, double dts);
  
  friend class Bmcv;
  friend class Decoder;
  friend class Blend;
};

template<std::size_t N>
class BMImageArray : public std::array<bm_image, N> {
 public:
  BMImageArray();
  BMImageArray(
      Handle                   &handle,
      int                      h,
      int                      w,
      bm_image_format_ext      format,
      bm_image_data_format_ext dtype);
  BMImageArray(
      Handle                   &handle,
      int                      h,
      int                      w,
      bm_image_format_ext      format,
      bm_image_data_format_ext dtype,
      int                      *stride);

  // BMImageArray(const std::vector<BMImage> &data);
  // int copy_from(Handle &handle, int i, BMImage &data);
  // int attach_from(Handle &handle, int i, BMImage &data);

  int copy_from(int i, BMImage &data);
  int attach_from(int i, BMImage &data);
  virtual ~BMImageArray();

  BMImageArray(BMImageArray &&other);
  BMImageArray& operator=(BMImageArray &&other);
  bool check_need_free() {return need_to_free_; };
  void set_need_free(bool value){need_to_free_ = value;};
  void create(
            Handle                   &handle,
            int                      h,
            int                      w,
            bm_image_format_ext      format,
            bm_image_data_format_ext dtype);

  void to_tensor(Tensor &tensor);

  /**
   * @brief Get device id of this image array.
   *
   * @return Device id.
   */
  int get_device_id();

 private:
  BMImageArray(const BMImageArray&) = delete;
  BMImageArray& operator=(const BMImageArray&) = delete;

  void create(
      Handle                   &handle,
      int                      h,
      int                      w,
      bm_image_format_ext      format,
      bm_image_data_format_ext dtype,
      int                      *stride);

  void create_not_alloc(
      Handle                   &handle,
      int                      h,
      int                      w,
      bm_image_format_ext      format,
      bm_image_data_format_ext dtype,
      int                      *stride);

  void destroy();
  //void allocate();
  bool is_created() const;
  bm_image_format_ext format(int index) const;
  void reset(int h, int w);

  bool need_to_free_;

  friend class Bmcv;
};

template class BMImageArray<1>;     //特化
template class BMImageArray<2>;     //特化
template class BMImageArray<3>;     //特化
template class BMImageArray<4>;     //特化
template class BMImageArray<6>;     //特化
template class BMImageArray<8>;     //特化
template class BMImageArray<16>;    //特化
template class BMImageArray<32>;    //特化
template class BMImageArray<64>;    //特化
template class BMImageArray<128>;   //特化
template class BMImageArray<256>;   //特化

#endif

#ifdef USE_FFMPEG
/**
 * @brief A class of image frame read by FFMPEG.
 *        It is an inner class used by Decoder.
 */
class Frame {
 public:
  /**
   * @brief Constructor.
   */
  Frame() {
    frame_ = av_frame_alloc();
  }
  ~Frame() {
    av_frame_free(&frame_);
  }
  /**
   * @brief Get the pointer of AVFrame instance.
   *
   * @return Pointer of AVFrame instance.
   */
  AVFrame* get() {
    return frame_;
  }
  /**
   * @brief Get height of the frame.
   *
   * @return Height of the frame
   */
  int get_height() {
    return frame_->height;
  }
  /**
   * @brief Get width of the frame.
   *
   * @return Width of the frame
   */
  int get_width() {
    return frame_->width;
  }

  Frame& operator =(const Frame& other){
      if (this == &other) {
          return *this;
      }

      if (this->frame_ != nullptr) {
          av_frame_free(&frame_);
      }
      this->frame_ = other.frame_;
      return *this;
  }

  void set_frame(AVFrame *frame) {
      if (this->frame_ != nullptr) {
          av_frame_free(&frame_);
      }
      this->frame_ = frame;
  }

 private:
  /// Pointer to AVFrame instance
  AVFrame* frame_;
};

enum class DecoderStatus: int
{
  NONE = -1,
  OPENED = 0,
  CLOSED = 1,
  STATUS_MAX
};

/**
 * @brief Decoder by VPU.
 *
 * Only format of AV_PIX_FMT_NV12 is supported.
 */
class DECL_EXPORT Decoder {
 public:
  /**
   * @brief Constructor.
   *
   * @param file_path  Path or rtsp url to the video/image file.
   * @param compressed Whether the format of decoded output is compressed NV12.
   * @param tpu_id     ID of TPU, there may be more than one TPU for PCIE mode.
   */
  explicit Decoder(
      const std::string& file_path,
      bool               compressed = true,
      int                tpu_id = 0);

  /**
   * @brief Destructor.
   */
  ~Decoder();

  /**
   * @brief Judge if the source is opened successfully.
   *
   * @return True if the source is opened successfully
   */
  bool is_opened();
  /**
   * @brief Get frame shape in the Decoder.
   *
   * @return Frame shape in the Decoder, [1, C, H, W]
   */
  std::vector<int> get_frame_shape();
  /**
   * @brief Read a BMImage from the image file.
   *
   * @param handle A bm_handle_t instance
   * @param image Reference of BMImage to be read to
   */
  int decode_jpeg(Handle& handle, BMImage& image);
  /**
   * @brief Read a bm_image from the image file.
   *
   * @param handle A bm_handle_t instance
   * @param image Reference of bm_image to be read to
   */
  int decode_jpeg(Handle& handle, bm_image& image);
  /**
   * @brief Read a BMImage from the Decoder.
   *
   * @param handle A bm_handle_t instance
   * @param image Reference of BMImage to be read to
   * @return 0 for success and 1 for failure
   */
  int read(Handle& handle, BMImage& image);
  /**
   * @brief Read a BMImage from the Decoder.
   *
   * @param handle A bm_handle_t instance
   * @return BMImage instance to be read to
   */
  BMImage read(Handle& handle);
  /**
   * @brief Read a bm_image from the Decoder.
   *
   * @param handle A bm_handle_t instance
   * @param image Reference of bm_image to be read to
   * @return 0 for success and 1 for failure
   */
  int read_(Handle& handle, bm_image& image);
  /**
   * @brief Read a bm_image from the Decoder.
   *
   * @param handle A bm_handle_t instance
   * @return bm_image instance to be read to
   */
  bm_image read_(Handle& handle);

  /**
   *  @brief Get the fps of the Video
   *  @return the fps of the video
   */
  float get_fps() const;

  /**
   * @brief Release Decoder.
   */
  void release();

  /**
   * @brief Reconnect Decoder.
   */
  int reconnect();

  /**
   * @brief enable video dump 
  */
  void enable_dump(int dump_max_seconds);

  /**
   * @brief disable video dump 
  */
  void disable_dump();

  /**
   * @brief get pts and dts
  */
  vector<double> get_pts_dts();

  /**
   * @brief video dump 
   * do not need to encode, just dump AVPacket(s) when decode
   * 
   * |---gop----|---gop----|---gop----|---gop----|
   * IPPPPPPPPPPIPPPPPPPPPPIPPPPPPPPPPIPPPPPPPPPPI
   *                  ^
   *                  (dump the stream before and after the current frame)
   * The frame sequence before the current frame must contain key frame. So the length of dump video is calculated based on gop size.
   * @param dump_pre_seconds dump video length(seconds) before dump moment
   * @param dump_post_seconds dump video length(seconds) after dump moment
   * @param  file_path output path
  */
  int dump(int dump_pre_seconds, int dump_post_seconds, std::string& file_path);

 private:
  class Decoder_CC;
  class Decoder_CC* const _impl;
};


class DECL_EXPORT Decoder_RawStream {
 public:
  /**
   * @brief Constructor.
   *
   * @param tpu_id     ID of TPU, there may be more than one TPU for PCIE mode.
   * @param decformt  decformt:h264 or h265.
   */
  Decoder_RawStream(
      int tpu_id,
      string  decformt);
  /**
   * @brief Destructor.
   */
  ~Decoder_RawStream();
  void release();

  int read_(uint8_t* data, int data_size, bm_image &image,bool continueFrame = false);
  int read(uint8_t* data, int data_size, sail::BMImage &image,bool continueFrame = false);

  #ifdef PYTHON
      int read_(pybind11::bytes data_bytes, bm_image& image, bool continueFrame = false);
      int read(pybind11::bytes data_bytes, BMImage& image, bool continueFrame = false);
  #endif


private:
    class Decoder_RawStream_CC;
    class Decoder_RawStream_CC* const _impl;

};


#endif

#ifdef USE_BMCV

#if BMCV_VERSION_MAJOR > 1
/**
 * @brief A class for image stitch.
 */
class DECL_EXPORT Blend{
public:
  /**
   * @brief blend init 
   * @param src_h       image height
   * @param ovlap_attr  overlapping areas width
   * @param bd_attr     Black border of images
   * @param wgt_phy_mem imgage weight
   * @param wgt_mode    weight mode
   */
  explicit Blend(int src_h, std::vector<std::vector<short>> ovlap_attr, std::vector<std::vector<short>> bd_attr, std::vector<std::vector<string>> wgt_phy_mem,bm_stitch_wgt_mode wgt_mode);
  ~Blend();

  /**
   * @brief blend process
   * @param input   input images
   * @param output  output image
   * @return 0 for success and other for failure
   */
  int process(std::vector<BMImage*> &input, BMImage &output);
  
  BMImage process(std::vector<BMImage*> &input);

private:
  class Blend_CC;
  Blend_CC* _impl;
};
#endif

/**
 * @brief A class for image processing by VPP/TPU.
 */
class DECL_EXPORT Bmcv {
 public:
  /**
   * @brief Constructor.
   *
   * @param handle A Handle instance
   */
  explicit Bmcv(Handle &handle);
  ~Bmcv();

#if defined(USE_BMCV) && defined(USE_OPENCV)
#if defined(PYTHON)
  BMImage mat_to_bm_image(pybind11::array_t<uint8_t> &mat);
  int mat_to_bm_image(pybind11::array_t<uint8_t> &input_array, BMImage &img);
#endif
  static int     mat_to_bm_image (cv::Mat &mat, BMImage &img);
  static BMImage mat_to_bm_image (cv::Mat &mat);

  static int     bm_image_to_mat(BMImage &img, cv::Mat &mat);
  static cv::Mat bm_image_to_mat(BMImage &img);
  #endif
  
  /**
   * @brief Convert BMImage to tensor.
   *
   * @param img      Input image
   * @param tensor   Output tensor
   */
  void   bm_image_to_tensor (BMImage &img, Tensor &tensor);
  Tensor bm_image_to_tensor (BMImage &img);

  template<std::size_t N> void   bm_image_to_tensor (BMImageArray<N> &imgs, Tensor &tensor);
  template<std::size_t N> Tensor bm_image_to_tensor (BMImageArray<N> &imgs);

  /**
   * @brief Convert tensor to BMImage.
   *
   * @param tensor   Input tensor
   * @param img      Output image
   * @param bgr2rgb  swap color channel
   * @param layout   layout of the tensor: "nchw" or "nhwc", default is "nchw"
   */
  void tensor_to_bm_image(Tensor &tensor, BMImage &img, bool bgr2rgb=false, std::string layout = std::string("nchw"));
  void tensor_to_bm_image(Tensor &tensor, BMImage &img, bm_image_format_ext format);
  BMImage tensor_to_bm_image (Tensor &tensor, bool bgr2rgb=false, std::string layout = std::string("nchw"));
  BMImage tensor_to_bm_image (Tensor &tensor, bm_image_format_ext format);
  template<std::size_t N> void  tensor_to_bm_image (Tensor &tensor, BMImageArray<N> &imgs, bool bgr2rgb=false, std::string layout = std::string("nchw"));
  template<std::size_t N> void  tensor_to_bm_image (Tensor &tensor, BMImageArray<N> &imgs, bm_image_format_ext format);

  /**
   * @brief Crop then resize an image.
   *
   * @param input    Input image
   * @param output   Output image
   * @param crop_x0  Start point x of the crop window
   * @param crop_y0  Start point y of the crop window
   * @param crop_w   Width of the crop window
   * @param crop_h   Height of the crop window
   * @param resize_w Target width
   * @param resize_h Target height
   * @param input_num   Input image count
   * @param resize_alg  Resize algorithm
   * @return 0 for success and other for failure
   */
  int crop_and_resize(
      bm_image                     *input,
      bm_image                     *output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      int                          input_num = 1,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  int crop_and_resize(
      BMImage                      &input,
      BMImage                      &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  BMImage crop_and_resize(
      BMImage                      &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  template<std::size_t N>
  int crop_and_resize(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  template<std::size_t N>
  BMImageArray<N> crop_and_resize(
      BMImageArray<N>              &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  bm_image crop_and_resize_padding(
      bm_image                     &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  BMImage crop_and_resize_padding(
      BMImage                     &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  /**
   * @brief Crop an image with given window.
   *
   * @param input    Input image
   * @param output   Output image
   * @param crop_x0  Start point x of the crop window
   * @param crop_y0  Start point y of the crop window
   * @param crop_w   Width of the crop window
   * @param crop_h   Height of the crop window
   * @return 0 for success and other for failure
   */
  int crop(
      BMImage                      &input,
      BMImage                      &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h);

  BMImage crop(
      BMImage                      &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h);

  vector<BMImage> crop(BMImage &input, vector<vector<int>> rects);

  template<std::size_t N>
  int crop(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h);

  template<std::size_t N>
  BMImageArray<N> crop(
      BMImageArray<N>              &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h);

  /**
   * @brief Resize an image with interpolation of INTER_NEAREST.
   *
   * @param input    Input image
   * @param output   Output image
   * @param resize_w Target width
   * @param resize_h Target height
   * @param resize_alg  Resize algorithm
   * @return 0 for success and other for failure
   */
  int resize(
      BMImage                      &input,
      BMImage                      &output,
      int                          resize_w,
      int                          resize_h,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  BMImage resize(
      BMImage                      &input,
      int                          resize_w,
      int                          resize_h,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  template<std::size_t N>
  int resize(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      int                          resize_w,
      int                          resize_h,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  template<std::size_t N>
  BMImageArray<N> resize(
      BMImageArray<N>              &input,
      int                          resize_w,
      int                          resize_h,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  /**
   * @brief Crop then resize an image using vpp.
   *
   * @param input    Input image
   * @param output   Output image
   * @param crop_x0  Start point x of the crop window
   * @param crop_y0  Start point y of the crop window
   * @param crop_w   Width of the crop window
   * @param crop_h   Height of the crop window
   * @param resize_w Target width
   * @param resize_h Target height
   * @return 0 for success and other for failure
   */
  int vpp_crop_and_resize(
      bm_image                     *input,
      bm_image                     *output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      int                          input_num = 1,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  int vpp_crop_and_resize(
      BMImage                      &input,
      BMImage                      &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  BMImage vpp_crop_and_resize(
      BMImage                      &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  int vpp_crop_and_resize_padding(
      bm_image                     *input,
      bm_image                     *output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in,
      int                          input_num = 1,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  int vpp_crop_and_resize_padding(
      BMImage                      &input,
      BMImage                      &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  BMImage vpp_crop_and_resize_padding(
      BMImage                      &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);


  template<std::size_t N>
  int vpp_crop_and_resize(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  template<std::size_t N>
  BMImageArray<N> vpp_crop_and_resize(
      BMImageArray<N>              &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);
  template<std::size_t N>
  int vpp_crop_and_resize_padding(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  template<std::size_t N>
  BMImageArray<N> vpp_crop_and_resize_padding(
      BMImageArray<N>              &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  /**
   * @brief Crop an image with given window using vpp.
   *
   * @param input    Input image
   * @param output   Output image
   * @param crop_x0  Start point x of the crop window
   * @param crop_y0  Start point y of the crop window
   * @param crop_w   Width of the crop window
   * @param crop_h   Height of the crop window
   * @return 0 for success and other for failure
   */
  int vpp_crop(
      BMImage                      &input,
      BMImage                      &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h);

  BMImage vpp_crop(
      BMImage                      &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h);

  template<std::size_t N>
  int vpp_crop(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h);

  template<std::size_t N>
  BMImageArray<N> vpp_crop(
      BMImageArray<N>              &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h);

  /**
   * @brief Resize an image with interpolation of INTER_NEAREST using vpp.
   *
   * @param input    Input image
   * @param output   Output image
   * @param resize_w Target width
   * @param resize_h Target height
   * @return 0 for success and other for failure
   */
  int vpp_resize(
      BMImage                      &input,
      BMImage                      &output,
      int                          resize_w,
      int                          resize_h,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  BMImage vpp_resize(
      BMImage                      &input,
      int                          resize_w,
      int                          resize_h,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  int vpp_resize_padding(
      BMImage                      &input,
      BMImage                      &output,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  BMImage vpp_resize_padding(
      BMImage                      &input,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  template<std::size_t N>
  int vpp_resize(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      int                          resize_w,
      int                          resize_h,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  template<std::size_t N>
  BMImageArray<N> vpp_resize(
      BMImageArray<N>              &input,
      int                          resize_w,
      int                          resize_h,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  template<std::size_t N>
  int vpp_resize_padding(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  template<std::size_t N>
  BMImageArray<N> vpp_resize_padding(
      BMImageArray<N>              &input,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in,
      bmcv_resize_algorithm        resize_alg = BMCV_INTER_NEAREST);

  /**
   * @brief Applies an affine transformation to an image.
   *
   * @param input    Input image
   * @param output   Output image
   * @param matrix   2x3 transformation matrix
   * @return 0 for success and other for failure
   */
  int warp(
      bm_image *input,
      bm_image *output,
      const std::pair<
        std::tuple<float, float, float>,
        std::tuple<float, float, float>> *matrix,
      int input_num = 1,
      int use_bilinear = 0,
      bool similar_to_opencv = false);

  int warp(
      BMImage                            &input,
      BMImage                            &output,
      const std::pair<
        std::tuple<float, float, float>,
        std::tuple<float, float, float>> &matrix,
      int                                use_bilinear = 0,
      bool                               similar_to_opencv = false);

  BMImage warp(
      BMImage                            &input,
      const std::pair<
        std::tuple<float, float, float>,
        std::tuple<float, float, float>> &matrix,
      int                                use_bilinear = 0,
      bool                               similar_to_opencv = false);

  template<std::size_t N>
  int warp(
      BMImageArray<N>                          &input,
      BMImageArray<N>                          &output,
      const std::array<
        std::pair<
          std::tuple<float, float, float>,
          std::tuple<float, float, float>>, N> &matrix,
      int                                      use_bilinear = 0,
      bool                                     similar_to_opencv = false);

  template<std::size_t N>
  BMImageArray<N> warp(
      BMImageArray<N>                          &input,
      const std::array<
        std::pair<
          std::tuple<float, float, float>,
          std::tuple<float, float, float>>, N> &matrix,
      int                                      use_bilinear = 0,
      bool                                     similar_to_opencv = false);

  /**
   * @brief Applies a linear transformation to an image.
   *
   * @param input        Input image
   * @param output       Output image
   * @param alpha_beta   (a0, b0), (a1, b1), (a2, b2) factors
   * @return 0 for success and other for failure
   */
  int convert_to(
      bm_image *input,
      bm_image *output,
      const std::tuple<
        std::pair<float, float>,
        std::pair<float, float>,
        std::pair<float, float>>   &alpha_beta,
        int input_num = 1);

  int convert_to(
      BMImage                      &input,
      BMImage                      &output,
      const std::tuple<
        std::pair<float, float>,
        std::pair<float, float>,
        std::pair<float, float>>   &alpha_beta);

  BMImage convert_to(
      BMImage                      &input,
      const std::tuple<
        std::pair<float, float>,
        std::pair<float, float>,
        std::pair<float, float>>   &alpha_beta);

  template<std::size_t N>
  int convert_to(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      const std::tuple<
        std::pair<float, float>,
        std::pair<float, float>,
        std::pair<float, float>>   &alpha_beta);

  template<std::size_t N>
  BMImageArray<N> convert_to(
      BMImageArray<N>              &input,
      const std::tuple<
        std::pair<float, float>,
        std::pair<float, float>,
        std::pair<float, float>>   &alpha_beta);

  /**
   * @brief Convert an image from YUV to BGR.
   *
   * @param input    Input image
   * @param output   Output image
   * @return 0 for success and other for failure
   */
  int yuv2bgr(
      BMImage                      &input,
      BMImage                      &output);

  BMImage yuv2bgr(
      BMImage                      &input);

  template<std::size_t N>
  int yuv2bgr(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output);

  template<std::size_t N>
  BMImageArray<N> yuv2bgr(
      BMImageArray<N>              &input);

  /**
   * @brief Convert an image to BGR PLANAR format using vpp.
   output.create
   * @param input    Input image
   * @param output   Output image
   * @return 0 for success and other for failure
   */
  int vpp_convert_format(
      BMImage          &input,
      BMImage          &output
  );

  BMImage vpp_convert_format(
      BMImage          &input,
      bm_image_format_ext image_format = FORMAT_BGR_PLANAR
  );

  template<std::size_t N>
  int vpp_convert_format(
      BMImageArray<N>  &input,
      BMImageArray<N>  &output
  );

  template<std::size_t N>
  BMImageArray<N> vpp_convert_format(
      BMImageArray<N>  &input
  );

  /**
   * @brief Convert an image to BGR PLANAR format.
   *
   * @param input    Input image
   * @param output   Output image
   * @return 0 for success and other for failure
   */
  int convert_format(
      BMImage          &input,
      BMImage          &output
  );

  BMImage convert_format(
      BMImage          &input,
      bm_image_format_ext image_format = FORMAT_BGR_PLANAR
  );

  template<std::size_t N>
  int convert_format(
      BMImageArray<N>  &input,
      BMImageArray<N>  &output
  );

  template<std::size_t N>
  BMImageArray<N> convert_format(
      BMImageArray<N>  &input
  );

  /**
   * @brief Draw a rectangle on input image.
   *
   * @param image      Input image
   * @param x0         Start point x of rectangle
   * @param y0         Start point y of rectangle
   * @param w          Width of rectangle
   * @param h          Height of rectangle
   * @param color      Color of rectangle
   * @param thickness  Thickness of rectangle
   * @return 0 for success and other for failure
   */
  int rectangle(
      const BMImage                   &image,
      int                             x0,
      int                             y0,
      int                             w,
      int                             h,
      const std::tuple<int, int, int> &color, // BGR
      int                             thickness=1
  );
  
  int rectangle(
      const bm_image                  &image,
      int                             x0,
      int                             y0,
      int                             w,
      int                             h,
      const std::tuple<int, int, int> &color, // BGR
      int                             thickness=1
  );

  int rectangle_(
      const bm_image                  &image,
      int                             x0,
      int                             y0,
      int                             w,
      int                             h,
      const std::tuple<int, int, int> &color, // BGR
      int                             thickness=1
  );

 /**
   * @brief Fill a rectangle on input image.
   *
   * @param image      Input image
   * @param x0         Start point x of rectangle
   * @param y0         Start point y of rectangle
   * @param w          Width of rectangle
   * @param h          Height of rectangle
   * @param color      Color of rectangle
   * @return 0 for success and other for failure
   */
  int fillRectangle(
      const BMImage                   &image,
      int                             x0,
      int                             y0,
      int                             w,
      int                             h,
      const std::tuple<int, int, int> &color
  );

  int fillRectangle(
      const bm_image                  &image,
      int                             x0,
      int                             y0,
      int                             w,
      int                             h,
      const std::tuple<int, int, int> &color
  );

  int fillRectangle_(
      const bm_image                  &image,
      int                             x0,
      int                             y0,
      int                             w,
      int                             h,
      const std::tuple<int, int, int> &color
  );

  /**
   * @brief put text on input image
   * 
   * @param image     Input image
   * @param text      Text string to be drawn
   * @param x         Start x
   * @param y         Start y
   * @param color     Color of text
   * @param fontScale Font scale factor that is multiplied by the font-specific base size
   * @param thickness Thickness of the lines used to draw a text
   * @return int 
   */
  int putText(
      const BMImage                   &image,
      const std::string               &text,
      int                             x,
      int                             y,
      const std::tuple<int, int, int> &color, // BGR
      float                           fontScale,
      int                             thickness=1
  );
  int putText(
      const bm_image                  &image,
      const std::string               &text,
      int                             x,
      int                             y,
      const std::tuple<int, int, int> &color, // BGR
      float                           fontScale,
      int                             thickness=1
  );
  int putText_(
      const bm_image                  &image,
      const std::string               &text,
      int                             x,
      int                             y,
      const std::tuple<int, int, int> &color, // BGR
      float                           fontScale,
      int                             thickness=1
  );

  /** @brief output = input1 * alpha + input2 * beta + gamma
   */
  int image_add_weighted(
      BMImage           &input1,
      float             alpha,
      BMImage           &input2,
      float             beta,
      float             gamma,
      BMImage           &output
  );

  BMImage image_add_weighted(
      BMImage           &input1,
      float             alpha,
      BMImage           &input2,
      float             beta,
      float             gamma
  );

  /**@brief Copy input image to output
   * @param input   Input image
   * @param output  Output image
   * @start_x       Target starting point x
   * @start_y       Target starting point y
   */
  int image_copy_to(bm_image input, bm_image output, int start_x, int start_y);

  int image_copy_to(BMImage &input, BMImage &output, int start_x = 0, int start_y = 0);

  template<std::size_t N>
  int image_copy_to(BMImageArray<N> &input, BMImageArray<N> &output, int start_x = 0, int start_y = 0);

  /**@brief Copy input image to output with padding
   * @param input   Input image
   * @param output  Output image
   * @param start_x       Target starting point x
   * @param start_y       Target starting point y
   * @param padding_r     padding value of r
   * @param padding_g     padding value of g
   * @param padding_b     padding value of b
   */
  int image_copy_to_padding(bm_image input, bm_image output,
    unsigned int padding_r, unsigned int padding_g, unsigned int padding_b,
    int start_x, int start_y);

  int image_copy_to_padding(BMImage &input, BMImage &output,
    unsigned int padding_r, unsigned int padding_g, unsigned int padding_b,
    int start_x = 0, int start_y = 0);
  
  template<std::size_t N>
  int image_copy_to_padding(BMImageArray<N> &input, BMImageArray<N> &output, 
    unsigned int padding_r, unsigned int padding_g, unsigned int padding_b,
    int start_x = 0, int start_y = 0);

  /**
   * @brief Save the image to the specified file.
   *
   * @param filename   Name of the file
   * @param image      Image to be saved
   * @return 0 for success and other for failure
   */
  int imwrite(
      const std::string &filename,
      const BMImage     &image
  );
  int imwrite(
      const std::string &filename,
      const bm_image     &image);
  int imwrite_(
      const std::string &filename,
      const bm_image     &image);
  /**
   * @brief Get Handle instance.
   *
   * @return Handle instance
   */
  Handle get_handle();

  /**
   * @brief Do nms use tpu
   * 
   * @param input_proposal input proposal objects
   * @param threshold      nms threshold
   * @param proposal_size  proposal size
   * @return result boxes [for c++, result memory should free by user]
   */
  nms_proposal_t* nms(face_rect_t *input_proposal,int proposal_size, float threshold);

  /**
   * @brief Applies a perspective transformation to an image.
   * 
   * @param input         Input image
   * @param coordinate    coordinate of left_top, right_top, left_bottom, right_bottom 
   * @param output_width  Output width
   * @param output_height Output height
   * @param format        Output format, only FORMAT_BGR_PLANAR,FORMAT_RGB_PLANAR
   * @param dtype         Output dtype, only DATA_TYPE_EXT_1N_BYTE,DATA_TYPE_EXT_4N_BYTE
   * @param use_bilinear  Bilinear use flag
   * @return Output image
   */

  BMImage warp_perspective(
    BMImage                     &input,
    const std::tuple<
      std::pair<int,int>,
      std::pair<int,int>,
      std::pair<int,int>,
      std::pair<int,int>>       &coordinate,
    int                         output_width,
    int                         output_height,
    bm_image_format_ext         format = FORMAT_BGR_PLANAR,
    bm_image_data_format_ext    dtype = DATA_TYPE_EXT_1N_BYTE,
    int                         use_bilinear = 0);

  /**
   * @brief Draw point.
   * 
   * @param image         Input image
   * @param center        Center of point
   * @param color         Color of point
   * @param radius        Radius of point
   */
  int drawPoint(
    const BMImage &image,
    std::pair<int,int> center,
    std::tuple<unsigned char, unsigned char, unsigned char> color,   // BGR
    int radius);

  int drawPoint(
    const bm_image  &image,
    std::pair<int,int> center,
    std::tuple<unsigned char, unsigned char, unsigned char> color,  // BGR
    int radius);

  int drawPoint_(
    const bm_image  &image,
    std::pair<int,int> center,
    std::tuple<unsigned char, unsigned char, unsigned char> color,  // BGR
    int radius);

  /**
   * @brief Draw lines.
   * 
   * @param image         Input image
   * @param start_points  start points of lines
   * @param end_points    end points if lines
   * @param line_num      nums of lines
   * @param color         Color of lines
   * @param thickness     thickness of lines
   */
  int drawLines(
    BMImage &image,
    std::vector<std::pair<int,int>> &start_points,
    std::vector<std::pair<int,int>> &end_points,
    int line_num,
    std::tuple<unsigned char, unsigned char, unsigned char> color,
    int thickness);

  /**
   * @brief Draw Polylines.
   * 
   * @param img           Input image
   * @param pts           The set of points of polylines
   * @param isClosed      Is the polygon closed 
   * @param color         Color of edges
   * @param thickness     Thickness of edges
   * @param shift         Scale the polylines
   */
  int polylines(
    BMImage &img,
    std::vector<std::vector<std::pair<int,int>>> &pts,
    bool isClosed,
    std::tuple<unsigned char, unsigned char, unsigned char> color,
    int thickness = 1,
    int shift = 0);

/**
   * @brief watermark superpose.
   * 
   * @param img           Input image
   * @param water_name    Path of bitmap
   * @param bitmap_num    Bitmap numbers of watermark
   * @param bitmap_type   Bitmap type of watermark
   * @param pitch         Width of watermark
   * @param rects         Rects of watermark
   * @param color         Color of watermark
   */
  int watermark_superpose(
        BMImage &img,
        string water_name,        
        int bitmap_type,
        int pitch,
        vector<vector<int>> rects,
        vector<int> color);

  /**
   * @brief Draw mosaic.
   * 
   * @param mosaic_num    Numbers of mosaic 
   * @param img           Input image
   * @param rects         The set of points of mosaic
   * @param is_expand     Surround the original mosaic with a macro block (8 pixels) 
   */
  int mosaic(
    int mosaic_num,
    BMImage &img,
    vector<vector<int>> rects,
    int is_expand);

  /**
   * @brief image transpose.
   * 
   * @param src           Input image
   * @param dst           Output image
   */
  int transpose(
    BMImage &src,
    BMImage &dst);

  /**
   * @brief image transpose.
   * 
   * @param src           Input image
   */
  BMImage transpose(
    BMImage &src);


   /**
   * @brief Edge detection Sobel operator.
   * 
   * @param input         Input BMImage 
   * @param output        Output BMImage
   * @param dx            Order of the derivative x
   * @param dy            Order of the derivative y
   * @param ksize         Size of the extended Sobel kernel; it must be -1, 1, 3, 5, or 7
   * @param scale         Optional scale factor for the computed derivative values; by default, no scaling is applied
   * @param delta         Optional delta value that is added to the results prior to storing them in dst
   */
  int Sobel(
        BMImage &input,
        BMImage &output,
        int dx,
        int dy,
        int ksize = 3,
        float scale = 1.0,
        float delta = 0.0);

 /**
   * @brief Edge detection Sobel operator.
   * 
   * @param input         Input BMImage 
   * @param dx            Order of the derivative x.
   * @param dy            Order of the derivative y
   * @param ksize         Size of the extended Sobel kernel; it must be -1, 1, 3, 5, or 7
   * @param scale         Optional scale factor for the computed derivative values; by default, no scaling is applied
   * @param delta         Optional delta value that is added to the results prior to storing them in dst
   */
  BMImage Sobel(
        BMImage &input,
        int dx,
        int dy,
        int ksize = 3,
        float scale = 1.0,
        float delta = 0.0);

  /**
   * @brief Load image from system memory.
   *
   * @param data_ptr  Pointer to data in system memory
   * @param data_size Byte size of data in system memory
   * @return BMImage
   */
  BMImage imdecode(const void* data_ptr, size_t data_size);

  /**
   * @brief Compresses the BMImage and stores it in the memory
   *
   * @param ext       File extension that defines the output format.
   * @param img       BMImage to be written
   * @param buf       Output buffer resized to fit the compressed BMImage
   * @return bool
   */
  bool imencode(std::string& ext, BMImage &img, std::vector<u_char>& buf);

  /**
   * @brief Compresses the bm_image and stores it in the memory
   *
   * @param ext       File extension that defines the output format.
   * @param img       bm_image to be written
   * @param buf       Output buffer resized to fit the compressed image
   * @return bool
   */
  bool imencode(std::string& ext, bm_image &img, std::vector<u_char>& buf);

  /**
   * @brief Read a jpeg image from file. Only support jpeg baseline image.
   * 
   * @param filename Name of file to be read.
   * @param dst The destination BMImage, which the image will be read into.
   * @return 0 for success and other for failure.
   */
  int     imread(const std::string &filename, BMImage &dst);

  /**
   *  @brief Read a jpeg image from file. Only support jpeg baseline image.
   * 
   *  @param filename Name of file to be read.
   *  @return The returned BMImage, which the image will be read into. The pixel format is in YUV-based.
   */
  BMImage imread(const std::string &filename);

  /**
   * @brief Short-Time Fourier Transform (STFT).
   *
   * @param input_real The real part of the input signal as a 2D numpy array.
   * @param input_imag The imaginary part of the input signal as a 2D numpy array.
   * @param realInput A boolean indicating whether the input signal is purely real. 
   *                  If true, the imaginary part is ignored.
   * @param normalize A boolean indicating whether to normalize the output.
   * @param n_fft The number of points used in the FFT. This determines the frequency resolution.
   * @param hop_len The number of samples to hop between successive frames. This controls the overlap.
   * @param pad_mode An integer indicating the padding mode to use when the input signal 
   *                 is shorter than n_fft:
   *                 - 0: Constant padding (pads with zeros).
   *                 - 1: Reflective padding (pads by reflecting the signal).
   * @param win_mode An integer specifying the window function to apply to each segment 
   *                  before computing the FFT:
   *                  - 0: Hann window.
   *                  - 1: Hamming window.
   * @return std::tuple<pybind11::array_t<float>, pybind11::array_t<float>> A tuple containing 
   *         two numpy arrays: the first array represents the real part of the STFT output, 
   *         and the second array represents the imaginary part of the STFT output. 
   *         Each array has shape (batch, n_fft / 2 + 1, num_frames), where num_frames is the 
   *         number of overlapping segments computed from the input signal.
   */
  #ifdef PYTHON
  std::tuple<pybind11::array_t<float>, pybind11::array_t<float>> stft(
      pybind11::array_t<float> input_real,
      pybind11::array_t<float> input_imag,
      bool realInput,
      bool normalize,
      int n_fft,
      int hop_len,
      int pad_mode,
      int win_mode
      );
  #endif
  std::tuple<Tensor, Tensor> stft(
      Tensor &input_real,
      Tensor &input_imag,
      bool realInput,
      bool normalize,
      int n_fft,
      int hop_len,
      int pad_mode,
      int win_mode
      );

  /**
   * @brief Inverse Short-Time Fourier Transform (ISTFT).
   *
   * @param input_real The real part of the STFT output as a 3D numpy array.
   * @param input_imag The imaginary part of the STFT output as a 3D numpy array.
   * @param realInput A boolean indicating whether the input STFT is purely real.
   *                  If true, the imaginary part is ignored.
   * @param normalize A boolean indicating whether to normalize the output.
   * @param L The length of the original time-domain signal to reconstruct.
   * @param hop_len The number of samples to hop between successive frames. This controls the overlap.
   * @param pad_mode An integer indicating the padding mode to use when the input signal 
   *                 is shorter than n_fft:
   *                 - 0: Constant padding (pads with zeros).
   *                 - 1: Reflective padding (pads by reflecting the signal).
   * @param win_mode An integer specifying the window function to apply to each segment 
   *                  before computing the FFT:
   *                  - 0: Hann window.
   *                  - 1: Hamming window.
   * @return pybind11::array_t<float> A numpy array representing the reconstructed time-domain signal. 
   *         The shape of the output array is (batch, signal_length), where signal_length is the length of the reconstructed signal.
   */
  #ifdef PYTHON
  std::tuple<pybind11::array_t<float>, pybind11::array_t<float>> istft(
      pybind11::array_t<float> input_real,
      pybind11::array_t<float> input_imag,
      bool realInput,
      bool normalize,
      int L,
      int hop_len,
      int pad_mode,
      int win_mode
      );
  #endif
  std::tuple<Tensor, Tensor> istft(
      Tensor &input_real,
      Tensor &input_imag,
      bool realInput,
      bool normalize,
      int L,
      int hop_len,
      int pad_mode,
      int win_mode
      );
  /**
   * @brief fft.
   * 
   * @param forward forward or Inverse transformation
   * @param input_real The real part of input
   * @param input_imag The imaginary part of input
   * @return std::vector<Tensor> The real and imaginary part of output
   */
  std::vector<Tensor> fft(bool forward, Tensor &input_real);
  std::vector<Tensor> fft(bool forward, Tensor &input_real, Tensor &input_imag);

  /**
   * @brief Convert a BMImage in yuv420p format to a BMImage in gray format.
   * 
   * @param image Input BMImage in yuv420p format
   * @return BMImage Output BMImage in gray format 
   */
  int convert_yuv420p_to_gray(BMImage& input, BMImage& output);
  int convert_yuv420p_to_gray(bm_image& input, bm_image& output);
  /**
   * @brief Convert a bm_image in yuv420p format to a bm_image in gray format.
   * 
   * @param image Input bm_image in yuv420p format
   * @return bm_image Output bm_image in gray format 
   */
  int convert_yuv420p_to_gray_(bm_image& input, bm_image& output);

  /**
   * @brief Gaussian_blur.
   *
   * @param input    Input image
   * @param output   Output image
   * @param kw       The size of kernel in the width direction
   * @param kh       The size of kernel in the height direction.
   * @param sigmaX   Gaussian kernel standard deviation in the X direction.
   * @param sigmaY   Gaussian kernel standard deviation in the Y direction.Default is 0.0, 
                      means that it is the same standard deviation as the Gaussian kernel in the X direction.
   * @return 0 for success and other for failure
   */
  int gaussian_blur(
      BMImage                      &input,
      BMImage                      &output,
      int                          kw,
      int                          kh,
      float                        sigmaX,
      float                        sigmaY = 0.0f);

  /**
   * @brief Gaussian_blur.
   *
   * @param input    Input image
   * @param kw       The size of kernel in the width direction
   * @param kh       The size of kernel in the height direction.
   * @param sigmaX   Gaussian kernel standard deviation in the X direction.
   * @param sigmaY   Gaussian kernel standard deviation in the Y direction.Default is 0.0, 
                      means that it is the same standard deviation as the Gaussian kernel in the X direction.
   * @return BMImage 
   */
  BMImage gaussian_blur(
      BMImage                      &input,
      int                          kw,
      int                          kh,
      float                        sigmaX,
      float                        sigmaY = 0.0f);

#if BMCV_VERSION_MAJOR > 1
  /**
   * @brief Add a watermark with a transparent channel to the image.
   * 
   * @param image           Input image
   * @param overlay_info    The position and size information of a set of watermarks in the format of [x,y,w,h]
   * @param overlay_image   A group of watermark
   * 
   * @return 0 for success and other for failure
   */
  int bmcv_overlay(
      BMImage&                      image, 
      std::vector<std::vector<int>> overlay_info, 
      std::vector<const BMImage *>        overlay_image);
#endif

// faiss series
  /**
   * @brief Calculate squared L2 distance between query vectors and database vectors, output the top topK L2sqr-values and the corresponding indices.
   *
   * @param query_vecs            Query vectors.
   * @param query_vecs_L2norm     Query vectors' square.
   * @param database_vecs         Database vectors.
   * @param database_vecs_L2norm  Database vectors' square.
   * @param vec_dims              The dimension of the Query vectors and Database vectors.
   * @param query_vecs_nums       The number of the Query vectors.
   * @param database_vecs_nums    The number of the Database vectors.
   * @param topK                  Get top topK values.  
   * @return std::tuple<pybind11::array_t<float>, pybind11::array_t<int>> A tuple containing
   *         two numpy arrays: the first array represents the squares of the top K most matching L2 distances,
   *         and the second array represents the  indices corresponding to the squares of the top K most matching L2 distances.
   *         Each array has shape (query_vecs_nums, topK).
   */
  #ifdef PYTHON
  // Python Interface 1
  std::tuple<pybind11::array_t<float>, pybind11::array_t<int>> faiss_indexflatL2(
    pybind11::array_t<float> query_vecs,
    pybind11::array_t<float> query_vecs_L2norm,
    pybind11::array_t<float> database_vecs,
    pybind11::array_t<float> database_vecs_L2norm,
    int vec_dims,
    int query_vecs_nums,
    int database_vecs_nums,
    int topK
  );

  // Python Interface 2
  std::tuple<pybind11::array_t<float>, pybind11::array_t<int>> faiss_indexflatL2(
    pybind11::array_t<float> query_vecs,
    pybind11::array_t<float> query_vecs_L2norm,
    sail::Tensor& database_vecs,
    sail::Tensor& database_vecs_L2norm,
    int vec_dims,
    int query_vecs_nums,
    int database_vecs_nums,
    int topK
  );
  #endif

  // C++ Interface 
  std::tuple<Tensor, Tensor> faiss_indexflatL2(
    Tensor &query_vecs,
    Tensor &query_vecs_L2norm,
    Tensor &database_vecs,
    Tensor& database_vecs_L2norm,
    int vec_dims,
    int query_vecs_nums,
    int database_vecs_nums,
    int topK
  );

  /**
   * @brief Calculate inner product distance between query vectors and database vectors, output the top K IP-values and the corresponding indices.
   *
   * @param query_vecs          Query vectors.
   * @param database_vecs       Database vectors.
   * @param vec_dims            The dimension of the Query vectors and Database vectors.
   * @param query_vecs_nums     The number of the Query vectors.
   * @param database_vecs_nums  The number of the Database vectors.
   * @param topK                Get top topK values.  
   * @return std::tuple<pybind11::array_t<OutputType>, pybind11::array_t<int>> A tuple containing
   *         two numpy arrays: the first array represents the squares of the top K most matching L2 distances,
   *         and the second array represents the  indices corresponding to the squares of the top K most matching L2 distances.
   *         Each array has shape (query_vecs_nums, topK).
   */
  #ifdef PYTHON
  // Python Interface 1
  std::tuple<pybind11::array_t<float>, pybind11::array_t<int>> faiss_indexflatIP(
    pybind11::array_t<float> query_vecs,
    sail::Tensor &database_vecs,
    int vec_dims,
    int query_vecs_nums,
    int database_vecs_nums,
    int topK
  );

  // Python Interface 2
  std::tuple<pybind11::array_t<float>, pybind11::array_t<int>> faiss_indexflatIP(
    pybind11::array_t<float> query_vecs,
    pybind11::array_t<float> database_vecs,
    int vec_dims,
    int query_vecs_nums,
    int database_vecs_nums,
    int topK
  );
  #endif

  // C++ Interface
  std::tuple<Tensor, Tensor> faiss_indexflatIP(
    Tensor &query_vecs,
    Tensor &database_vecs,
    int vec_dims,
    int query_vecs_nums,
    int database_vecs_nums,
    int topK
  );

// PQ_encoder
  /**
   * @brief Encode vectors into int8 PQcodes, output the encoded PQcodes.
   *
   * @param input_vecs            Input vectors.
   * @param centroids_vecs        Centroids vector.
   * @param encode_vecs_num       The number of Input vectors.
   * @param vec_dims              The dimension of the Input vectors and Centroids vector.
   * @param slice_num             The number of the sliced vector.
   * @param centroids_num         The number of the centroids.
   * @param IP_metric             0 indicates L2 distance calculation, 1 indicates the IP distance calculation.  
   * @return                      Encoded PQcodes.
   */
#ifdef PYTHON
  // Python Interface1 of faiss_indexPQ_encode
  Tensor faiss_indexPQ_encode(
    pybind11::array_t<float> input_vecs,
    Tensor &centroids_vecs,
    int encode_vecs_num,
    int vec_dims,
    int slice_num,
    int centroids_num,
    int IP_metric
  );

  // Python Interface2 of faiss_indexPQ_encode
  pybind11::array_t<uint8_t> faiss_indexPQ_encode(
    pybind11::array_t<float> input_vecs,
    pybind11::array_t<float> centroids_vecs,
    int encode_vecs_num,
    int vec_dims,
    int slice_num,
    int centroids_num,
    int IP_metric
  );

  // Python Interface3 of faiss_indexPQ_encode
  int faiss_indexPQ_encode(
    pybind11::array_t<float> input_vecs,
    Tensor &centroids_vecs,
    Tensor &encoded_vecs,
    int encode_vecs_num,
    int vec_dims,
    int slice_num,
    int centroids_num,
    int IP_metric
  );
#endif

 // C++ Interface of faiss_indexPQ_encode
 int faiss_indexPQ_encode(
    Tensor &input_vecs,
    Tensor &centroids_vecs,
    Tensor &encoded_vecs,
    int encode_vecs_num,
    int vec_dims,
    int slice_num,
    int centroids_num,
    int IP_metric
 );


// bmcv_faiss_indexPQ_ADC
  /**
   * @brief PQ Asymmetric Distance Computation, output the topK distance and label of x and q(ny).
   *
   * @param nxquery_vecs            Query vectors.
   * @param centroids_vecs          Centroids vectors.
   * @param nycodes_vecs            Encoded database vectors
   * @param vec_dims                The dimension of the Input vectors.
   * @param slice_num               The number of the sliced vectors.
   * @param centroids_num           The number of the centroids.
   * @param database_num            The number of the database vectors.
   * @param query_num               The number of the query vectors.
   * @param topK                    Get top topK values.
   * @param IP_metric               0 indicates L2 distance calculation, 1 indicates the IP distance calculation.  
   * @return                        The first array represents the top K most matching distances,and the second array 
   *                                represents the  indices corresponding to the top K most matching distances.
   */
#ifdef PYTHON
  // Python Interface1 of faiss_indexPQ_ADC
  std::tuple<pybind11::array_t<float>, pybind11::array_t<int>> faiss_indexPQ_ADC (
      pybind11::array_t<float> nxquery_vecs,
      pybind11::array_t<float> centroids_vecs,
      pybind11::array_t<uint8_t> nycodes_vecs,
      int vec_dims,   
      int slice_num,  
      int centroids_num,
      int database_vecs_num,
      int query_vecs_num,
      int topK,
      int IP_metric
  );

  // Python Interface 2 of faiss_indexPQ_ADC
  std::tuple<pybind11::array_t<float>, pybind11::array_t<int>> faiss_indexPQ_ADC (
      pybind11::array_t<float> nxquery_vecs,
      Tensor &centroids_vecs,
      Tensor &nycodes_vecs,
      int vec_dims,   
      int slice_num,  
      int centroids_num,
      int database_vecs_num,
      int query_vecs_num,
      int topK,
      int IP_metric
  );
#endif

  // C++ Interface of faiss_indexPQ_ADC
    std::tuple<Tensor, Tensor> faiss_indexPQ_ADC (
      Tensor &nxquery_vecs,
      Tensor &centroids_vecs,
      Tensor &nycodes_vecs,
      int vec_dims,   
      int slice_num,  
      int centroids_num,
      int database_vecs_num,
      int query_vecs_num,
      int topK,
      int IP_metric
  );

// bmcv_faiss_indexPQ_SDC
  /**
   * @brief  PQ Symmetric Distance Computation, output the topK distance and label of q(x) and q(ny).
   *
   * @param nxcodes_vecs            PQcodes of query vectors.
   * @param nycodes_vecs            PQcodes of database vectors.
   * @param sdc_table               sdc_table.
   * @param slice_num               The num of sliced vector.
   * @param centroids_num           The number of the centroids.
   * @param database_vecs_num       The number of the database vectors.
   * @param query_vecs_num          The number of the query vectors.
   * @param topK                    Get top topK values.
   * @param IP_metric               0 indicates L2 distance calculation, 1 indicates the IP distance calculation.  
   * @return                        The first array represents the top K most matching distances,and the second array 
   *                                represents the  indices corresponding to the top K most matching distances.
   */
#ifdef PYTHON
  // Python Interface1 of faiss_indexPQ_SDC
  std::tuple<pybind11::array_t<float>, pybind11::array_t<int>> faiss_indexPQ_SDC (
      pybind11::array_t<uint8_t> nxcodes_vecs,
      pybind11::array_t<uint8_t> nycodes_vecs,
      pybind11::array_t<float> sdc_table, 
      int slice_num,  
      int centroids_num,
      int database_vecs_num,
      int query_vecs_num,
      int topK,
      int IP_metric
  );

  // Python Interface2 of faiss_indexPQ_SDC
  std::tuple<pybind11::array_t<float>, pybind11::array_t<int>> faiss_indexPQ_SDC (
      pybind11::array_t<uint8_t> nxcodes_vecs,
      Tensor &nycodes_vecs,
      Tensor &sdc_table, 
      int slice_num,  
      int centroids_num,
      int database_vecs_num,
      int query_vecs_num,
      int topK,
      int IP_metric
  );

  // Python Interface3 of faiss_indexPQ_SDC
  std::tuple<pybind11::array_t<float>, pybind11::array_t<int>> faiss_indexPQ_SDC (
      Tensor &nxcodes_vecs,
      Tensor &nycodes_vecs,
      Tensor &sdc_table, 
      int slice_num,  
      int centroids_num,
      int database_vecs_num,
      int query_vecs_num,
      int topK,
      int IP_metric
  );
#endif

  // C++ Interface of faiss_indexPQ_SDC
  int faiss_indexPQ_SDC (
      Tensor &nxcodes_vecs,
      Tensor &nycodes_vecs,
      Tensor &sdc_table,
      Tensor &distance,
      Tensor &index, 
      int slice_num,  
      int centroids_num,
      int database_vecs_num,
      int query_vecs_num,
      int topK,
      int IP_metric
  );

#ifdef PYTHON
  pybind11::array_t<float> nms(pybind11::array_t<float> input_proposal, float threshold);

  BMImage imdecode(pybind11::bytes data_bytes);

  /**
   * @brief Compresses the image and stores it in the memory
   *
   * @param ext       File extension that defines the output format. Must include a leading period
   * @param img       BMImage to be written
   * @param buf       Output buffer resized to fit the compressed image
   * @return The encoded array
   */
  pybind11::array_t<uint8_t> imencode(std::string& ext, BMImage &img);

  
  /**
   * @brief Compresses the image and stores it in the memory
   *
   * @param ext       File extension that defines the output format. Must include a leading period
   * @param img       BMImage to be written
   * @param buf       Output buffer resized to fit the compressed image
   * @return The encoded array
   */
  pybind11::array_t<uint8_t> imencode(std::string& ext, bm_image &img);

#endif

  bm_data_type_t           get_bm_data_type(bm_image_data_format_ext fmt);
  bm_image_data_format_ext get_bm_image_data_format(bm_data_type_t dtype);

 private:
  Handle handle_;

  template<std::size_t N>
  void check_create(BMImageArray<N>& image,int height, int width, bm_image_data_format_ext dtype, bool reset = true,
    bm_image_format_ext fmt = FORMAT_BGR_PLANAR, int *stride = nullptr);

  template<std::size_t N>
  void check_create_not_alloc(BMImageArray<N>& image,int height, int width, bm_image_data_format_ext dtype, bool reset = true,
    bm_image_format_ext fmt = FORMAT_BGR_PLANAR, int *stride = nullptr);

  void check_create(BMImage& image,int height, int width, bm_image_data_format_ext dtype, 
    bm_image_format_ext fmt = FORMAT_BGR_PLANAR, int *stride = nullptr);
};

template<std::size_t N>
void Bmcv::check_create(BMImageArray<N>& image, int height, int width, bm_image_data_format_ext dtype, bool reset,
  bm_image_format_ext fmt, int *stride){
  if (reset && image.is_created()) {
      image.reset(height, width);
  }
  if(!image.is_created()){
    image.create(handle_, height, width, fmt, dtype, stride);
  }
}

template<std::size_t N>
void Bmcv::check_create_not_alloc(BMImageArray<N>& image, int height, int width, bm_image_data_format_ext dtype, bool reset,
  bm_image_format_ext fmt, int *stride){
  image.create_not_alloc(handle_, height, width, fmt, dtype, stride);
}

template<std::size_t N>
void Bmcv::bm_image_to_tensor (BMImageArray<N> &imgs, Tensor &tensor)
{
  return imgs.to_tensor(tensor);
}

template<std::size_t N>
Tensor Bmcv::bm_image_to_tensor (BMImageArray<N> &imgs)
{
  Tensor tensor(get_handle());
  bm_image_to_tensor(imgs, tensor);
  return std::move(tensor);
}

template<std::size_t N>
void Bmcv::tensor_to_bm_image (Tensor &tensor, BMImageArray<N> &imgs, bool bgr2rgb, std::string layout)
{
  auto shape = tensor.shape();
  int n = shape[0];
  if (n != N) {
    spdlog::error("Batch size mis-matched, Tensor batch size: {}, BMImageArray size: {}!",n,N);
    return;
  }
  int h, w;
  if(strcmp(layout.c_str(), "nchw") == 0) {
    h = shape[2];
    w = shape[3];
  } else if (strcmp(layout.c_str(), "nhwc") == 0) {
    h = shape[1];
    w = shape[2];
  } else {
    spdlog::error("Invalid layout!");
    return;
  }

  bm_image_data_format_ext dtype = get_bm_image_data_format(tensor.dtype());
  check_create_not_alloc(imgs, h, w, dtype, false, bgr2rgb ? FORMAT_RGB_PLANAR : FORMAT_BGR_PLANAR);

  bm_device_mem_t mem = tensor.dev_data();
  bm_image_attach_contiguous_mem(imgs.size(), imgs.data(), mem);
}

template<std::size_t N> 
void Bmcv::tensor_to_bm_image (Tensor &tensor, BMImageArray<N> &imgs, bm_image_format_ext format)
{
  auto shape = tensor.shape();
  int n = shape[0];
  if (n != N) {
    spdlog::error("Batch size mis-matched, Tensor batch size: {}, BMImageArray size: {}!",n,N);
    return;
  }
  int h, w;
  if(format == FORMAT_RGB_PLANAR || format == FORMAT_BGR_PLANAR) { //nchw
    h = shape[2];
    w = shape[3];
  } else if (format == FORMAT_RGB_PACKED || format == FORMAT_BGR_PACKED) { //nhwc
    h = shape[1];
    w = shape[2];
  } else {
    spdlog::error("Invalid format!");
    return;
  }

  bm_image_data_format_ext dtype = get_bm_image_data_format(tensor.dtype());
  check_create_not_alloc(imgs, h, w, dtype, false, format);

  bm_device_mem_t mem = tensor.dev_data();
  bm_image_attach_contiguous_mem(imgs.size(), imgs.data(), mem);
}

template<std::size_t N>
int Bmcv::crop_and_resize(
  BMImageArray<N> &input,
  BMImageArray<N> &output,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h,
  int             resize_w,
  int             resize_h,
  bmcv_resize_algorithm resize_alg
) {
  check_create(output, resize_h, resize_w, input[0].data_type);
  int ret = 0;
  for(int i = 0; i < N; ++i) {
    ret = crop_and_resize(&input.at(i), &output.at(i),
          crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, resize_alg);
    if (ret != 0){
      break;
    }
  }
  return ret;
}

template<std::size_t N>
BMImageArray<N> Bmcv::crop_and_resize(
  BMImageArray<N> &input,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h,
  int             resize_w,
  int             resize_h,
  bmcv_resize_algorithm resize_alg
) {
  BMImageArray<N> output;
  crop_and_resize(input, output, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, resize_alg);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::crop(
  BMImageArray<N>              &input,
  BMImageArray<N>              &output,
  int                          crop_x0,
  int                          crop_y0,
  int                          crop_w,
  int                          crop_h
) {
  return crop_and_resize(input, output, crop_x0, crop_y0, crop_w, crop_h, crop_w, crop_h);
}

template<std::size_t N>
BMImageArray<N> Bmcv::crop(
  BMImageArray<N>              &input,
  int                          crop_x0,
  int                          crop_y0,
  int                          crop_w,
  int                          crop_h
) {
  BMImageArray<N> output;
  crop(input, output, crop_x0, crop_y0, crop_w, crop_h);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::resize(
  BMImageArray<N>              &input,
  BMImageArray<N>              &output,
  int                          resize_w,
  int                          resize_h,
  bmcv_resize_algorithm        resize_alg
) {
  return crop_and_resize(input, output, 0, 0, input[0].width, input[0].height, resize_w, resize_h, resize_alg);
}

template<std::size_t N>
BMImageArray<N> Bmcv::resize(
  BMImageArray<N>              &input,
  int                          resize_w,
  int                          resize_h,
  bmcv_resize_algorithm        resize_alg
) {
  BMImageArray<N> output;
  resize(input, output, resize_w, resize_h, resize_alg);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::vpp_crop_and_resize(
  BMImageArray<N> &input,
  BMImageArray<N> &output,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h,
  int             resize_w,
  int             resize_h,
  bmcv_resize_algorithm resize_alg
) {

  check_create(output, resize_h, resize_w, input[0].data_type);
  int ret = 0;
  for (int i = 0; i < N; ++i){
    ret = vpp_crop_and_resize(&input.at(i), &output.at(i),
          crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, resize_alg);
    if (ret != 0){
      break;
    }
  }
  return ret;
}

template<std::size_t N>
BMImageArray<N> Bmcv::vpp_crop_and_resize(
  BMImageArray<N> &input,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h,
  int             resize_w,
  int             resize_h,
  bmcv_resize_algorithm resize_alg
) {
  BMImageArray<N> output;
  vpp_crop_and_resize(input, output, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, resize_alg);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::vpp_crop_and_resize_padding(
  BMImageArray<N> &input,
  BMImageArray<N> &output,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h,
  int             resize_w,
  int             resize_h,
  PaddingAtrr     &padding_in,
  bmcv_resize_algorithm resize_alg
) {
  check_create(output, resize_h, resize_w, input[0].data_type);
 
  return vpp_crop_and_resize_padding(input.data(), output.data(), crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h,padding_in, N, resize_alg);
}

template<std::size_t N>
BMImageArray<N> Bmcv::vpp_crop_and_resize_padding(
  BMImageArray<N> &input,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h,
  int             resize_w,
  int             resize_h,
  PaddingAtrr     &padding_in,
  bmcv_resize_algorithm resize_alg
) {
  BMImageArray<N> output;
  vpp_crop_and_resize_padding(input, output, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, padding_in, resize_alg);
  return std::move(output);
}
template<std::size_t N>
int Bmcv::vpp_crop(
  BMImageArray<N> &input,
  BMImageArray<N> &output,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h
) {
  return vpp_crop_and_resize(input, output, crop_x0, crop_y0, crop_w, crop_h, crop_w, crop_h);
}

template<std::size_t N>
BMImageArray<N> Bmcv::vpp_crop(
  BMImageArray<N> &input,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h
) {
  BMImageArray<N> output;
  vpp_crop(input, output, crop_x0, crop_y0, crop_w, crop_h);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::vpp_resize(
  BMImageArray<N> &input,
  BMImageArray<N> &output,
  int             resize_w,
  int             resize_h,
  bmcv_resize_algorithm resize_alg
) {
  return vpp_crop_and_resize(input, output, 0, 0, input[0].width, input[0].height, resize_w, resize_h, resize_alg);
}

template<std::size_t N>
BMImageArray<N> Bmcv::vpp_resize(
  BMImageArray<N> &input,
  int             resize_w,
  int             resize_h,
  bmcv_resize_algorithm resize_alg
) {
  BMImageArray<N> output;
  vpp_resize(input, output, resize_w, resize_h, resize_alg);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::vpp_resize_padding(
  BMImageArray<N> &input,
  BMImageArray<N> &output,
  int             resize_w,
  int             resize_h,
  PaddingAtrr     &padding_in,
  bmcv_resize_algorithm resize_alg
) {
  return vpp_crop_and_resize_padding(input, output, 0, 0, input[0].width, input[0].height, resize_w, resize_h, padding_in, resize_alg);
}

template<std::size_t N>
BMImageArray<N> Bmcv::vpp_resize_padding(
  BMImageArray<N> &input,
  int             resize_w,
  int             resize_h,
  PaddingAtrr     &padding_in,
  bmcv_resize_algorithm resize_alg
) {
  BMImageArray<N> output;
  vpp_resize_padding(input, output, resize_w, resize_h, padding_in, resize_alg);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::warp(
  BMImageArray<N>                          &input,
  BMImageArray<N>                          &output,
  const std::array<
    std::pair<
      std::tuple<float, float, float>,
      std::tuple<float, float, float>>, N> &matrix,
  int                                      use_bilinear,
  bool                                     similar_to_opencv
) {
  check_create(output, input[0].height, input[0].width, input[0].data_type, false, input[0].image_format);

  int ret = warp(input.data(), output.data(), matrix.data(), N, use_bilinear, similar_to_opencv);
  return ret;
}

template<std::size_t N>
BMImageArray<N> Bmcv::warp(
  BMImageArray<N>                          &input,
  const std::array<
    std::pair<
      std::tuple<float, float, float>,
      std::tuple<float, float, float>>, N> &matrix,
  int                                      use_bilinear,
  bool                                     similar_to_opencv
) {
  BMImageArray<N> output;
  warp(input, output, matrix, use_bilinear, similar_to_opencv);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::convert_to(
  BMImageArray<N>            &input,
  BMImageArray<N>            &output,
  const std::tuple<
    std::pair<float, float>,
    std::pair<float, float>,
    std::pair<float, float>> &alpha_beta
) {
  check_create(output, input[0].height, input[0].width, input[0].data_type);

  return convert_to(input.data(), output.data(), alpha_beta, N);
}

template<std::size_t N>
BMImageArray<N> Bmcv::convert_to(
  BMImageArray<N>            &input,
  const std::tuple<
    std::pair<float, float>,
    std::pair<float, float>,
    std::pair<float, float>> &alpha_beta
) {
  BMImageArray<N> output;
  convert_to(input, output, alpha_beta);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::yuv2bgr(
    BMImageArray<N> &input,
  BMImageArray<N> &output
) {
#if (BMCV_VERSION_MAJOR == 2) && !defined(BMCV_VERSION_MINOR)
  return vpp_convert_format(input,output);
#else
  check_create(output, input[0].height, input[0].width, input[0].data_type, false);
  int ret = bmcv_image_yuv2bgr_ext(handle_.data(), N, input.data(), output.data());
  return ret;
#endif
}

template<std::size_t N>
BMImageArray<N> Bmcv::yuv2bgr(
  BMImageArray<N> &input
) {
  BMImageArray<N> output;
  yuv2bgr(input, output);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::vpp_convert_format(
  BMImageArray<N> &input,
  BMImageArray<N> &output
) {
 
  check_create(output, input[0].height, input[0].width, input[0].data_type, false);

  int ret = 0;
  for (int i = 0; i < N; i ++) {
    ret = bmcv_image_vpp_convert(
      handle_.data(),
      1,
      input[i],
      &output[i]
    );
    if (ret != 0) break;
  }

  return ret;
}

template<std::size_t N>
BMImageArray<N> Bmcv::vpp_convert_format(
  BMImageArray<N> &input
) {
  BMImageArray<N> output;
  vpp_convert_format(input, output);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::convert_format(
  BMImageArray<N>  &input,
  BMImageArray<N>  &output
) {
  check_create(output,input[0].height,input[0].width,input[0].data_type, false);
  int ret = bmcv_image_storage_convert(
    handle_.data(),
    N,
    input.data(),
    output.data()
  );

  return ret;
}

template<std::size_t N>
BMImageArray<N> Bmcv::convert_format(
  BMImageArray<N>  &input
) {
  BMImageArray<N> output;
  convert_format(input, output);
  return std::move(output);
}

  template<std::size_t N>
  int Bmcv::image_copy_to(BMImageArray<N> &input, BMImageArray<N> &output, int start_x, int start_y)
  {
    if (!output.is_created() || !input.is_created()){
      SPDLOG_ERROR("input or output must be created before!");
      return SAIL_ERR_BMI_EMPTY;
    }
    for(int i = 0; i < N; ++i) {
      int ret = image_copy_to(input.at(i), output.at(i), start_x, start_y);
      if (ret != 0){
        SPDLOG_ERROR("image_copy_to failed, ret = {}, start_x {} start_y {}", ret, start_x, start_y);
        return ret;
      }
    }
    return BM_SUCCESS;
  }

  template<std::size_t N>
  int Bmcv::image_copy_to_padding(BMImageArray<N> &input, BMImageArray<N> &output, 
    unsigned int padding_r, unsigned int padding_g, unsigned int padding_b,
    int start_x, int start_y)
  {
    if (!output.is_created() || !input.is_created()){
      SPDLOG_ERROR("input or output must be created before!");
      return SAIL_ERR_BMI_EMPTY;
    }
    for(int i = 0; i < N; ++i) {
      int ret = image_copy_to_padding(input.at(i), output.at(i),padding_r, padding_g, padding_b, start_x, start_y);
      if (ret != 0){
        return ret;
      }
    }
    return BM_SUCCESS;
  }

  /**
   * @brief Get the indices of the maximum value
   * 
   * @param tensor input tensor
   * @return the indices of the maximum value.
   */
  int DECL_EXPORT argmax(sail::Tensor& tensor);

  /**
   * @brief Get the indices of the minimum value
   * 
   * @param tensor input tensor
   * @return the indices of the minimum value.
   */
  int DECL_EXPORT argmin(sail::Tensor& tensor);

#endif

}  // namespace sail
