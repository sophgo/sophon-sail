#pragma once
#include <regex>
#ifdef WIN
typedef unsigned char u_char;
#else
#include <unistd.h>
#include <sys/time.h>
#endif
#include <csignal>

#ifdef USE_BMCV
#include "bmcv_api_ext.h"

#if !(BMCV_VERSION_MAJOR > 1)
#include <bmcv_api.h>
#endif

#endif

#ifdef USE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#endif

#ifdef USE_FFMPEG
extern "C"
{
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/time.h>
#include <libswscale/swscale.h>
}
#endif

#include "cvwrapper.h"
#include "graph.h"
#include "tensor.h"

namespace sail {
#ifdef USE_FFMPEG

class DECL_EXPORT Encoder {
  public:
    Encoder();
    Encoder(const std::string &output_path, int device_id, const std::string &enc_fmt, const std::string &pix_fmt, const std::string &enc_params, int cache_buffer_length=5, int abort_policy=0);
    Encoder(const std::string &output_path, Handle& handle, const std::string &enc_fmt, const std::string &pix_fmt, const std::string &enc_params, int cache_buffer_length=5, int abort_policy=0);
    ~Encoder();

    bool is_opened();

    // c++ interface for pic encode, return data length
    int pic_encode(std::string& ext, BMImage &image, std::vector<u_char>& data);
    int pic_encode(std::string& ext, bm_image &image, std::vector<u_char>& data);

    // python interface for pic encode, return data on system memory as numpy
#ifdef PYTHON
    pybind11::array_t<uint8_t> pic_encode(std::string& ext, BMImage &image);
    pybind11::array_t<uint8_t> pic_encode(std::string& ext, bm_image &image);
#endif

    int video_write(BMImage &image);
    int video_write(bm_image &image);

    int reconnect();
    void release();

  private:
    class Encoder_CC;
    class Encoder_CC* const _impl;
};
#endif
}
