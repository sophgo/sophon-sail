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
// Decoder_RawStream is only supported when USE_FFMPEG and USE_BMCV are defined
#if defined(USE_FFMPEG) && defined(USE_BMCV)
#include <string>

#include "bmcv_api_ext.h"

#ifdef PYTHON
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#endif  // PYTHON

namespace sail {
class BMImage;  // forward declaration
class DECL_EXPORT Decoder_RawStream {
   public:
    /**
     * @brief Constructor.
     *
     * @param tpu_id     ID of TPU
     * mode.
     * @param decformt  Coding format of raw video data to be decoded, h264 or
     * h265.
     */
    Decoder_RawStream(int tpu_id, std::string decformt);
    /**
     * @brief Destructor.
     */
    ~Decoder_RawStream();
    void release();

    /**
     * @brief Read a frame from the raw stream.
     * @param data        The raw stream data.
     * @param data_size   The size of the raw stream data.
     * @param image       The decoded image.
     * @param continueFrame  Whether to continue decoding the current frame.
     * @return 0 on success, -1 on failure.
     * @details
     * - Decoder will preserve decoding state internally, like how many data was
     * consumed
     */
    int read(uint8_t* data, int data_size, sail::BMImage& image,
             bool continueFrame = false);
    int read_(uint8_t* data, int data_size, bm_image& image,
              bool continueFrame = false);
    int read_single_frame(uint8_t* data, int data_size, sail::BMImage& image,
                         bool continueFrame = false, bool need_flush = false);

#ifdef PYTHON
    /**
     * @brief Read a frame from the raw stream.
     * @param data_bytes  The raw stream data.
     * @param image       The decoded image.
     * @param continueFrame  Whether to continue decoding the current frame.
     * @return 0 on success, -1 on failure.
     */
    int read(pybind11::bytes data_bytes, BMImage& image,
             bool continueFrame = false);
    int read_(pybind11::bytes data_bytes, bm_image& image,
              bool continueFrame = false);
    int read_single_frame(pybind11::bytes data_bytes, BMImage& image,
                         bool continueFrame = false, bool need_flush = false);
#endif

   private:
    class Decoder_RawStream_CC;
    class Decoder_RawStream_CC* const _impl;
};

}  // namespace sail

#endif  // USE_FFMPEG