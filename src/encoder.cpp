#include "encoder.h"
#include <iostream>
#include <queue>

#if BMCV_VERSION_MAJOR > 1
#include "internal.h"
#endif

#ifdef USE_OPENCV
using namespace cv;
#endif

#ifdef USE_FFMPEG
namespace sail {
typedef struct{
        bm_image *bmImg;
        uint8_t* buf0;
        uint8_t* buf1;
        uint8_t* buf2;
}transcode_t;


void bmBufferDeviceMemFree(void *opaque, uint8_t *data)
{
    spdlog::debug("enter {}", __func__);
    if(opaque == NULL){
        spdlog::error("bm_image to avframe, create buffer error, parameter error\n");
    }
    transcode_t *testTranscoed = (transcode_t *)opaque;
    av_freep(&testTranscoed->buf0);
    testTranscoed->buf0 = NULL;

    int ret =  0;
    spdlog::debug("before bm_image_destroy()");
    ret = bm_image_destroy(*(testTranscoed->bmImg));
    spdlog::debug("after bm_image_destroy()");
    if(testTranscoed->bmImg){
        free(testTranscoed->bmImg);
        testTranscoed->bmImg =NULL;
    }
    if(ret != 0)
        spdlog::error("bm_image to avframe callback, bm_image destroy failed\n");
    free(testTranscoed);
    testTranscoed = NULL;
    return ;
}
void bmBufferDeviceMemFree2(void *opaque, uint8_t *data)
{
    return ;
}

bool string_start_with(const std::string &s, const std::string &head) {
    return s.compare(0, head.size(), head) == 0;
}

class Encoder::Encoder_CC
{
public:
    Encoder_CC();
    Encoder_CC(const std::string &output_path, int device_id, const std::string &enc_fmt,
                const std::string &pix_fmt, const std::string &enc_params, int cache_buffer_length=5, int abort_policy=0);

    ~Encoder_CC();

    bool is_opened();

    int pic_encode(std::string& ext, bm_image &image, std::vector<u_char>& data);

#ifdef PYTHON
    pybind11::array_t<uint8_t> pic_encode(std::string& ext, bm_image &image);
#endif

    int video_write(bm_image &image);

    int reconnect();

    void release();

private:

    enum OUTPUT_TYPE{
        RTSP_STREAM=0,
        RTMP_STREAM,
        BASE_STREAM,
        VIDEO_LOCAL_FILE
    };

    enum ERROR_CODE{
        CACHE_OVERFLOW=1,
        ERROR_ENCODE,
        ERROR_WRITE_INTERLEAVE,
        UNKNOEN_ABORT_POLICY
    };

    enum ABORT_POLICY{
        RETURN_NOW=0,
        POP_FRONT,
        POP_ALL,
        BLOCKING
    };

    unsigned int chip_id_;
    int tpu_id_;
    bm_handle_t handle_;
    std::string output_path_;
    std::string enc_fmt_;
    std::string enc_params_;
    std::map<std::string, int> params_map_;
    int abort_policy_;
    // FILE *fp;
    bool quit_flag;
    bool write_frame_start_;
    int cache_buffer_length_;
    std::mutex frame_process_lock;
    std::queue<AVFrame*> frame_process_q;
    std::thread write_frame_thread_;

    int encode_ret = 0;
    int write_frame_ret = 0;

    bool is_jpeg_;
    bool is_video_file_;
    bool is_rtsp_;
    bool is_rtmp_;
    bool opened_;

    int64_t first_frame_time = 0;
    int64_t frame_idx = 0;
    bool first_frame_flag = true;

    void *jpeg_addr;
    int jpeg_size;

    AVPixelFormat       pix_fmt_;
    AVCodec*            encoder_;
    AVDictionary *      enc_dict_;
    AVFormatContext*    enc_format_ctx_;
    AVCodecContext*     enc_ctx_;
    AVStream*           out_stream_;

    int get_output_type(std::string &output_path);
    void enc_params_prase();
    int map_bmformat_to_avformat(int bmformat);
    int bm_image_to_avframe(bm_handle_t &handle, bm_image *image, AVFrame *frame);
    int flush_encoder();
    void frame_process();
    void write_frame();
};

Encoder::Encoder()
    : _impl(new Encoder_CC())
{
}

Encoder::Encoder(
    const std::string &output_path, Handle& handle, const std::string &enc_fmt, const std::string &pix_fmt, const std::string &enc_params, int cache_buffer_length, int abort_policy)
    : _impl(new Encoder_CC(output_path, handle.get_device_id(), enc_fmt, pix_fmt, enc_params, cache_buffer_length, abort_policy))
{
}

Encoder::Encoder(
    const std::string &output_path, int device_id, const std::string &enc_fmt, const std::string &pix_fmt, const std::string &enc_params, int cache_buffer_length, int abort_policy)
    : _impl(new Encoder_CC(output_path, device_id, enc_fmt, pix_fmt, enc_params, cache_buffer_length, abort_policy))
{
}

Encoder::~Encoder()
{
    delete _impl;
}

bool Encoder::is_opened()
{
    return _impl->is_opened();
}

int Encoder::pic_encode(std::string& ext, BMImage &image, std::vector<u_char>& data)
{
    return _impl->pic_encode(ext, image.data(), data);
}

int Encoder::pic_encode(std::string& ext, bm_image &image, std::vector<u_char>& data)
{
    return _impl->pic_encode(ext, image, data);
}

#ifdef PYTHON
pybind11::array_t<uint8_t> Encoder::pic_encode(std::string& ext, BMImage &image)
{
    return _impl->pic_encode(ext, image.data());
}

pybind11::array_t<uint8_t> Encoder::pic_encode(std::string& ext, bm_image &image)
{
    return _impl->pic_encode(ext, image);
}
#endif

int Encoder::video_write(BMImage &image)
{
    return _impl->video_write(image.data());
}

int Encoder::video_write(bm_image &image)
{
    return _impl->video_write(image);
}

int Encoder::reconnect()
{
    return _impl->reconnect();
}

void Encoder::release()
{
    return _impl->release();
}

int Encoder::Encoder_CC::get_output_type(std::string &output_path)
{
    if(string_start_with(output_path, "rtsp://"))
        return RTSP_STREAM;
    if(string_start_with(output_path, "rtmp://"))
        return RTMP_STREAM;
    if(string_start_with(output_path, "tcp://") || string_start_with(output_path, "udp://"))
        return BASE_STREAM;
    return VIDEO_LOCAL_FILE;
}

int Encoder::Encoder_CC::map_bmformat_to_avformat(int bmformat)
{
    int format = 0;
    if (is_jpeg_)
    {
        switch (bmformat)
        {
        case FORMAT_YUV420P:
            format = AV_PIX_FMT_YUVJ420P;
            break;
        case FORMAT_YUV422P:
            format = AV_PIX_FMT_YUVJ422P;
            break;
        case FORMAT_YUV444P:
            format = AV_PIX_FMT_YUVJ444P;
            break;
        default:
            format = -1;
        }
    }
    else
    {
        switch (bmformat)
        {
        case FORMAT_YUV420P:
            format = AV_PIX_FMT_YUV420P;
            break;
        case FORMAT_NV12:
            format = AV_PIX_FMT_NV12;
            break;
        default:
            format = -1;
        }
    }
    return format;
}

void Encoder::Encoder_CC::enc_params_prase()
{
    params_map_.insert(std::pair<std::string, int>("width", 1920));
    params_map_.insert(std::pair<std::string, int>("height", 1080));
    params_map_.insert(std::pair<std::string, int>("framerate", 25));
    params_map_.insert(std::pair<std::string, int>("bitrate", 2000));
    params_map_.insert(std::pair<std::string, int>("gop", 32));
    params_map_.insert(std::pair<std::string, int>("gop_preset", 3));
    params_map_.insert(std::pair<std::string, int>("mb_rc", 0));
    params_map_.insert(std::pair<std::string, int>("qp", -1));
    params_map_.insert(std::pair<std::string, int>("bg", 0));
    params_map_.insert(std::pair<std::string, int>("nr", 0));
    params_map_.insert(std::pair<std::string, int>("weightp", 0));

    std::string s1;
    s1.append(1, ':');
    std::regex reg1(s1);

    std::string s2;
    s2.append(1, '=');
    std::regex reg2(s2);

    std::vector<std::string> elems(std::sregex_token_iterator(enc_params_.begin(), enc_params_.end(), reg1, -1),
                                   std::sregex_token_iterator());
    for (auto param : elems)
    {
        std::vector<std::string> key_value_(std::sregex_token_iterator(param.begin(), param.end(), reg2, -1),
                                       std::sregex_token_iterator());

        std::string temp_key = key_value_[0];
        std::string temp_value = key_value_[1];

        params_map_[temp_key] = std::stoi(temp_value);
        spdlog::info("encode params: {}={}", temp_key, temp_value);
    }
}

Encoder::Encoder_CC::Encoder_CC():opened_(false)
{
}

Encoder::Encoder_CC::Encoder_CC(const std::string &output_path, int device_id, const std::string &enc_fmt,
                                const std::string &pix_fmt, const std::string &enc_params, int cache_buffer_length, int abort_policy)
    : output_path_(output_path), chip_id_(0x1684), tpu_id_(device_id), is_jpeg_(false), is_rtsp_(false), is_rtmp_(false), is_video_file_(false),
    opened_(false), enc_ctx_(nullptr), enc_dict_(nullptr), enc_fmt_(enc_fmt), enc_params_(enc_params), pix_fmt_(AV_PIX_FMT_NONE), quit_flag(false),
    write_frame_start_(false), cache_buffer_length_(cache_buffer_length),abort_policy_(abort_policy)
{
    int ret = 0;
    ret = bm_dev_request(&handle_, tpu_id_);
    if (BM_SUCCESS != ret)
    {
        SPDLOG_ERROR("Encoder bm_dev_request fail, device_id: {}", tpu_id_);
    }
    ret = bm_get_chipid(handle_, &chip_id_);
    if (BM_SUCCESS != ret)
    {
        SPDLOG_ERROR("Encoder bm_get_chipid fail, device_id: {}", tpu_id_);
    }

    // get output format
    int output_type = get_output_type(output_path_);
    switch(output_type)
    {
        case RTSP_STREAM:
            spdlog::info("sail.Encoder: you are pushing a rtsp stream.");
            avformat_alloc_output_context2(&enc_format_ctx_, NULL, "rtsp", output_path_.c_str());
            is_rtsp_ = true;
            break;
        case RTMP_STREAM:
            spdlog::info("sail.Encoder: you are pushing a rtmp stream.");
            avformat_alloc_output_context2(&enc_format_ctx_, NULL, "flv", output_path_.c_str());
            is_rtmp_ = true;
            break;
        case BASE_STREAM:
            // if(string_start_with(enc_fmt_.c_str(), "h264_bm"))
            //     avformat_alloc_output_context2(&enc_format_ctx_, NULL, "h264", output_path_.c_str());
            // if(string_start_with(enc_fmt_.c_str(), "hevc_bm") || string_start_with(enc_fmt_.c_str(), "h265_bm") )
            //     avformat_alloc_output_context2(&enc_format_ctx_, NULL, "hevc", output_path_.c_str());
            spdlog::error("sail.Encoder: Not support tcp/udp stream yet.");
            throw std::runtime_error("Not support tcp/udp stream yet.");
            break;
        case VIDEO_LOCAL_FILE:
            spdlog::info("sail.Encoder: you are writing a local video file.");
            avformat_alloc_output_context2(&enc_format_ctx_, NULL, NULL, output_path_.c_str());
            break;
        default:
            throw std::runtime_error("Failed to alloc output context.");
            break;
    }

    if (!enc_format_ctx_) {
        spdlog::error("sail.Encoder: Could not create output context\n");
        throw std::runtime_error("Failed to alloc output context.");
    }

    // find encoder & alloc encoder context
    encoder_ = avcodec_find_encoder_by_name(enc_fmt_.c_str());
    if (!encoder_)
    {
        spdlog::error("Failed to find encoder: {} \n", enc_fmt_);
        throw std::runtime_error("Failed to find encoder.");
    }
    enc_ctx_ = avcodec_alloc_context3(encoder_);
    if (!encoder_)
    {
        spdlog::error("Failed to alloc context. \n");
        throw std::runtime_error("Failed to alloc context.");
    }

    // SPS PPS do not be tied with IDR, global header.
    if(enc_format_ctx_->oformat->flags & AVFMT_GLOBALHEADER)
        enc_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    enc_params_prase();

    if (pix_fmt == "I420")
    {
        pix_fmt_ = AV_PIX_FMT_YUV420P;
    }
    else if (pix_fmt == "NV12")
    {
        pix_fmt_ = AV_PIX_FMT_NV12;
    }
    else
    {
        spdlog::error("only support I420, NV12, check pix_fmt: {} \n.", pix_fmt);
        throw std::runtime_error("Not support encode pix format.");
    }
    enc_ctx_->codec_id      =   encoder_->id;
    enc_ctx_->pix_fmt       =   pix_fmt_;
    enc_ctx_->width         =   params_map_["width"];
    enc_ctx_->height        =   params_map_["height"];
    enc_ctx_->gop_size      =   params_map_["gop"];
    enc_ctx_->time_base     =   (AVRational){1, params_map_["framerate"]};
    enc_ctx_->framerate     =   (AVRational){params_map_["framerate"], 1};
    if(-1 == params_map_["qp"])
    {
        enc_ctx_->bit_rate_tolerance = params_map_["bitrate"]*1000;
        enc_ctx_->bit_rate      =   (int64_t)params_map_["bitrate"]*1000;
    }else{
        av_dict_set_int(&enc_dict_, "qp", params_map_["qp"], 0);
    }

    av_dict_set_int(&enc_dict_, "sophon_idx", tpu_id_, 0);
    av_dict_set_int(&enc_dict_, "gop_preset", params_map_["gop_preset"], 0);
    // av_dict_set_int(&enc_dict_, "mb_rc",      params_map_["mb_rc"],      0);    0);
    // av_dict_set_int(&enc_dict_, "bg",         params_map_["bg"],         0);
    // av_dict_set_int(&enc_dict_, "nr",         params_map_["nr"],         0);
    // av_dict_set_int(&enc_dict_, "weightp",    params_map_["weightp"],    0);
    av_dict_set_int(&enc_dict_, "is_dma_buffer", 1, 0);

    // open encoder
    ret = avcodec_open2(enc_ctx_, encoder_, &enc_dict_);
    if(ret < 0){
        spdlog::error("sail.Encoder: avcodec_open failed, return: {} \n.", ret);
        throw std::runtime_error("avcodec_open failed.");
    }
    av_dict_free(&enc_dict_);

    // new stream
    out_stream_ = avformat_new_stream(enc_format_ctx_, encoder_);
    out_stream_->time_base      = enc_ctx_->time_base;
    out_stream_->avg_frame_rate = enc_ctx_->framerate;
    out_stream_->r_frame_rate   = out_stream_->avg_frame_rate;

    ret = avcodec_parameters_from_context(out_stream_->codecpar, enc_ctx_);
    if(ret < 0)
    {
        spdlog::error("sail.Encoder: avcodec_parameters_from_context failed, return: {} \n.", ret);
        throw std::runtime_error("avcodec_parameters_from_context failed.");
    }

    if (!(enc_format_ctx_->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&enc_format_ctx_->pb, output_path_.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            spdlog::error("sail.Encoder: avio_open failed, return: {} \n.", ret);
            throw std::runtime_error("avio_open2 failed.");
        }
    }
    AVDictionary *header_options = NULL;
    if (output_type == RTSP_STREAM) {
        av_dict_set(&header_options, "rtsp_flags", "prefer_tcp", 0);
    }
    ret = avformat_write_header(enc_format_ctx_, &header_options);
    av_dict_free(&header_options);
        if (ret < 0) {
            spdlog::error("sail.Encoder: avformat_write_header failed, return: {} \n.", ret);
            throw std::runtime_error("avformat_write_header failed.");
        }
    opened_ = true;

    if(!write_frame_start_)
    {
        write_frame_thread_ = std::thread(&Encoder_CC::write_frame, this);
        write_frame_thread_.detach();
        write_frame_start_ = true;
    }
}

Encoder::Encoder_CC::~Encoder_CC()
{
    if(opened_)
        release();
}

int Encoder::Encoder_CC::bm_image_to_avframe(bm_handle_t& handle, bm_image *image, AVFrame *frame)
{
    if(!bm_image_is_attached(*image))
    {
        spdlog::error("sail.Encoder: input image does not attach device memory.");
        return BM_ERR_PARAM;
    }
    int enc_width = params_map_["width"];
    int enc_height = params_map_["height"];
    int plane = 0;
    bm_image *yuv_image = (bm_image*)malloc(sizeof(bm_image));
    bm_image_format_info info;
    int encode_stride = ((enc_width + 31) >> 5) << 5;

    if(pix_fmt_ == AV_PIX_FMT_YUV420P)
    {
        plane = 3;
        int stride_bmi[3] = {encode_stride, encode_stride /2, encode_stride /2};
        bm_image_create(handle, enc_height, enc_width, FORMAT_YUV420P, DATA_TYPE_EXT_1N_BYTE, yuv_image, stride_bmi);
    }
    if(pix_fmt_ == AV_PIX_FMT_NV12)
    {
        plane = 2;
        int stride_bmi[2] = {encode_stride, encode_stride};
        bm_image_create(handle, enc_height, enc_width, FORMAT_NV12, DATA_TYPE_EXT_1N_BYTE, yuv_image, stride_bmi);
    }

    int ret = 0;
    bmcv_rect_t crop_rect = {0, 0, image->width, image->height};
    switch(chip_id_)
    {
        case 0x1686A200:
            ret = bm_image_alloc_dev_mem_heap_mask(*yuv_image, 2);//vpp heap.
            if (BM_SUCCESS != ret)
            {
                spdlog::error("sail.Encoder: alloc vpu-heap device mem failed, ret: {}", ret);
                return ret;
            }
            ret = bmcv_image_vpp_convert(handle, 1, *image, yuv_image ,&crop_rect);
            if (BM_SUCCESS != ret)
            {
                bm_image_destroy(*yuv_image);
                spdlog::error("sail.Encoder: filter input image to encode pix_fmt and size failed, ret: {}", ret);
                return ret;
            }
            break;
        case 0x1686:
            ret = bm_image_alloc_dev_mem_heap_mask(*yuv_image, 4);
            if (BM_SUCCESS != ret)
            {
                spdlog::error("sail.Encoder: alloc vpu-heap device mem failed, ret: {}", ret);
                return ret;
            }
            ret = bmcv_image_vpp_convert(handle, 1, *image, yuv_image ,&crop_rect);
            if (BM_SUCCESS != ret)
            {
                bm_image_destroy(*yuv_image);
                spdlog::error("sail.Encoder: filter input image to encode pix_fmt and size failed, ret: {}", ret);
                return ret;
            }
            break;
        case 0x1684:
            spdlog::debug("bm_image_to_avframe src bm_image: width={}, height={}, image_format={}",
                            image->width, image->height, image->image_format);
            spdlog::debug("bm_image_to_avframe dst bm_image: width={}, height={}, image_format={}",
                            yuv_image->width, yuv_image->height, yuv_image->image_format);
            if(image->width==enc_width && image->height==enc_height)
            {
                ret = bm_image_alloc_dev_mem_heap_mask(*yuv_image, 4);
                if (BM_SUCCESS != ret)
                {
                    spdlog::error("sail.Encoder: alloc vpu-heap device mem failed, ret: {}", ret);
                    return ret;
                }
                ret = bmcv_image_storage_convert(handle, 1, image, yuv_image);
                if(ret!=BM_SUCCESS)
                {
                    bm_image_destroy(*yuv_image);
                    spdlog::error("sail.Encoder: filter input image to encode pix_fmt failed, ret: {}", ret);
                    return ret;
                }
            }
            else
            {
                // when input shape != output shape(encoder shape)
                // do format and resize
                if (pix_fmt_ == AV_PIX_FMT_YUV420P)
                {
                    if (image->image_format == FORMAT_BGR_PLANAR || image->image_format == FORMAT_BGR_PACKED ||
                        image->image_format == FORMAT_RGB_PLANAR || image->image_format == FORMAT_RGB_PACKED ||
                        image->image_format == FORMAT_GRAY)
                    {
                        // format
                        bm_image i420_image;
                        if (BM_SUCCESS != bm_image_create(handle, image->height, image->width, FORMAT_YUV420P, DATA_TYPE_EXT_1N_BYTE, &i420_image))
                        {
                            spdlog::error("sail.Encoder: create tmp I420 bm_image failed, ret: {}", ret);
                            return ret;
                        }
                        if (BM_SUCCESS != bm_image_alloc_dev_mem_heap_mask(i420_image, 6))
                        {
                            bm_image_destroy(i420_image);
                            i420_image.image_private = nullptr;
                            spdlog::error("sail.Encoder: alloc vpu-heap device mem failed, ret: {}", ret);
                            return ret;
                        }
                        ret = (image->image_format == FORMAT_GRAY) ? bmcv_image_storage_convert(handle, 1, image, &i420_image)
                                                                   : bmcv_image_vpp_convert(handle, 1, *image, &i420_image, &crop_rect);
                        if (BM_SUCCESS != ret)
                        {
                            bm_image_destroy(i420_image);
                            i420_image.image_private = nullptr;
                            spdlog::error("sail.Encoder: format input image to tmp I420 bm_image failed, ret: {}", ret);
                            return ret;
                        }

                        // resize
                        if (BM_SUCCESS != bm_image_alloc_dev_mem_heap_mask(*yuv_image, 4))
                        {
                            bm_image_destroy(*yuv_image);
                            yuv_image->image_private = nullptr;
                            spdlog::error("sail.Encoder: alloc vpu-heap device mem failed, ret: {}", ret);
                            return ret;
                        }
                        if (BM_SUCCESS != bmcv_image_vpp_convert(handle, 1, i420_image, yuv_image, &crop_rect))
                        {
                            bm_image_destroy(*yuv_image);
                            yuv_image->image_private = nullptr;
                            bm_image_destroy(i420_image);
                            i420_image.image_private = nullptr;
                            spdlog::error("sail.Encoder: resize input image to encode shape failed, "
                                          "format {} w {} h {} -> format {} w {} h{}, ret : {} ",
                                          image->image_format, image->width, image->height,
                                          yuv_image->image_format, yuv_image->width, yuv_image->height, ret);
                            return ret;
                        }
                        bm_image_destroy(i420_image);
                        i420_image.image_private = nullptr;
                    }
                    else if (image->image_format == FORMAT_YUV420P)
                    {
                        // only do resize
                        if (BM_SUCCESS != bm_image_alloc_dev_mem_heap_mask(*yuv_image, 4))
                        {
                            bm_image_destroy(*yuv_image);
                            yuv_image->image_private = nullptr;
                            spdlog::error("sail.Encoder: alloc vpu-heap device mem failed, ret: {}", ret);
                            return ret;
                        }
                        if (BM_SUCCESS != bmcv_image_vpp_convert(handle, 1, *image, yuv_image, &crop_rect))
                        {
                            bm_image_destroy(*yuv_image);
                            spdlog::error("sail.Encoder: resize input image to encode shape failed, "
                                          "format {} w {} h {} -> format {} w {} h{}, ret : {} ",
                                          image->image_format, image->width, image->height,
                                          yuv_image->image_format, yuv_image->width, yuv_image->height, ret);
                            return ret;
                        }
                    }
                    else
                    {
                        spdlog::error("sail.Encoder: convert failed. The shape of input image must match the shape specified by the encoder, when pix_fmt is {}.", image->image_format);
                        return BM_NOT_SUPPORTED;
                    }
                }
                else
                {
                    spdlog::error("sail.Encoder: convert failed. The shape of input image must match the shape specified by the encoder, when pix_fmt is NV12.");
                    return BM_NOT_SUPPORTED;
                }
            }
            break;
        default:
            spdlog::error("sail.Encoder: chip {} not supported yet.", chip_id_);
            return BM_NOT_SUPPORTED;
    }

    transcode_t *ImgOut  = NULL;
    ImgOut = (transcode_t *)malloc(sizeof(transcode_t));
    ImgOut->bmImg = yuv_image;

    if(ImgOut->bmImg->width > 0 && ImgOut->bmImg->height > 0
        && ImgOut->bmImg->height * ImgOut->bmImg->width <= 8192*4096) {
        ImgOut->buf0 = (uint8_t*)av_malloc(ImgOut->bmImg->height * ImgOut->bmImg->width * 3 / 2);
        ImgOut->buf1 = ImgOut->buf0 + (unsigned int)(ImgOut->bmImg->height * ImgOut->bmImg->width);
        if(plane == 3){
            ImgOut->buf2 = ImgOut->buf0 + (unsigned int)(ImgOut->bmImg->height * ImgOut->bmImg->width * 5 / 4);
        }
    }

    frame->buf[0] = av_buffer_create(ImgOut->buf0,ImgOut->bmImg->width * ImgOut->bmImg->height,
        bmBufferDeviceMemFree,ImgOut,AV_BUFFER_FLAG_READONLY);
    frame->buf[1] = av_buffer_create(ImgOut->buf1,ImgOut->bmImg->width * ImgOut->bmImg->height / 2 /2 ,
        bmBufferDeviceMemFree2,NULL,AV_BUFFER_FLAG_READONLY);
    frame->data[0] = ImgOut->buf0;
    frame->data[1] = ImgOut->buf0;

    if(plane == 3){
        frame->buf[2] = av_buffer_create(ImgOut->buf2,ImgOut->bmImg->width * ImgOut->bmImg->height / 2 /2 ,
            bmBufferDeviceMemFree2,NULL,AV_BUFFER_FLAG_READONLY);
        frame->data[2] = ImgOut->buf0;
    }

    if(plane == 3 && !frame->buf[2]){
        spdlog::error("sail.Encoder: encode pix format I420, but convert bm_image to AVFrame, buf[2] create failed \n");
        av_buffer_unref(&frame->buf[0]);
        av_buffer_unref(&frame->buf[1]);
        av_buffer_unref(&frame->buf[2]);
        free(ImgOut);
        free(image);
        return BM_ERR_FAILURE;
    }
    else if(plane == 2 && !frame->buf[1]){
        spdlog::error("sail.Encoder: encode pix format NV12, but convert bm_image to AVFrame, buf[1] create failed \n");
        av_buffer_unref(&frame->buf[0]);
        av_buffer_unref(&frame->buf[1]);
        free(ImgOut);
        free(image);
        return BM_ERR_FAILURE;
    }

    frame->format = (AVPixelFormat)map_bmformat_to_avformat(yuv_image->image_format);
    frame->height = image->height;
    frame->width = image->width;

    bm_device_mem_t* mems = new bm_device_mem_t[plane];
    bm_image_get_device_mem(*yuv_image, mems);
    bm_image_get_format_info(yuv_image, &info);

    for (int idx = 0; idx < plane; idx++)
    {
        frame->data[4 + idx] = (uint8_t *)mems[idx].u.device.device_addr;
        frame->linesize[idx] = info.stride[idx];
        frame->linesize[4 + idx] = info.stride[idx];
    }

    delete[] mems;

    return BM_SUCCESS;
}

int Encoder::Encoder_CC::pic_encode(std::string& ext, bm_image &image, std::vector<u_char>& data)
{
    cv::Mat mat;
    cv::bmcv::toMAT(&image, mat);
    cv::imencode(ext.c_str(), mat, data);
    return data.size();
}

#ifdef PYTHON
pybind11::array_t<uint8_t> Encoder::Encoder_CC::pic_encode(std::string& ext, bm_image &image)
{
    cv::Mat mat;
    cv::bmcv::toMAT(&image, mat);
    std::vector<u_char> data;
    cv::imencode(ext.c_str(), mat, data);
    // printf("size: %d \n", data.size());
    // FILE* fp = fopen("test.jpg", "wb+");
    // fwrite(data.data(), 1, data.size(), fp);
    // fclose(fp);
    std::vector<pybind11::ssize_t> shape = { pybind11::ssize_t(data.size())};
    std::vector<pybind11::ssize_t> strides = {sizeof(uint8_t)};
    pybind11::buffer_info output_buf(data.data(), sizeof(uint8_t), pybind11::format_descriptor<uint8_t>::format(),
                                         1, shape, strides);
    // return np_data;
    return std::move(pybind11::array_t<uint8_t>(output_buf));
}
#endif
bool Encoder::Encoder_CC::is_opened()
{
    return opened_;
}

void Encoder::Encoder_CC::write_frame()
{
    int64_t frame_interval = 1* 1000 * 1000 / params_map_["framerate"];

    while(!quit_flag)
    {
        frame_process_lock.lock();
        if(frame_process_q.empty())
        {
            frame_process_lock.unlock();
            av_usleep(10*1000);
            continue;
        }

        AVFrame* frame = frame_process_q.front();
        frame_process_q.pop();
        frame_process_lock.unlock();
        frame->pts = frame_idx++;

        AVPacket enc_pkt;
        av_init_packet(&enc_pkt);
        enc_pkt.data = nullptr;
        enc_pkt.size = 0;

        int got_output = 0;
        spdlog::trace("sail.Encoder: write_frame() avcodec_encode_video2() start");
        int ret = avcodec_encode_video2(enc_ctx_, &enc_pkt, frame, &got_output);
        spdlog::trace("sail.Encoder: write_frame() avcodec_encode_video2() finish, ret: {}", ret);
        av_frame_free(&frame);

        encode_ret = ret;
        if (ret < 0)
        {
            spdlog::error("sail.Encoder: encode failed for one frame in cache queue: {}", encode_ret);
            continue;
        }
        if (got_output == 0) {
            spdlog::debug("sail.Encoder: encoder no output for one frame in cache queue");
            continue;
        }

        spdlog::debug("sail.Encoder: write_frame() enc_pkt.pts={}, enc_pkt.dts={}", enc_pkt.pts, enc_pkt.dts);
        av_packet_rescale_ts(&enc_pkt, enc_ctx_->time_base,out_stream_->time_base);
        spdlog::debug("sail.Encoder: write_frame() rescaled enc_pkt.pts={}, enc_pkt.dts={}", enc_pkt.pts, enc_pkt.dts);
        
        if(is_rtsp_ || is_rtmp_)
        {
            if(first_frame_flag)
            {
                first_frame_time = av_gettime_relative();
                spdlog::debug("sail.Encoder: write_frame() first_frame_time: {}", first_frame_time);
                first_frame_flag = false;
            }
            else
            {
                int64_t this_frame_pts_time = first_frame_time + frame_idx * frame_interval;
                int64_t this_frame_time = av_gettime_relative();
                spdlog::debug("sail.Encoder: write_frame() elapsed_time: {}", this_frame_time - first_frame_time);
                int64_t wait_time = this_frame_pts_time - this_frame_time;
                spdlog::debug("sail.Encoder: write_frame() wait_time: {}", wait_time);
                if(wait_time > 0)
                    av_usleep(wait_time);
            }
        }
        
        spdlog::trace("sail.Encoder: write_frame() av_interleaved_write_frame() start, frame_idx: {}", frame_idx);
        ret = av_interleaved_write_frame(enc_format_ctx_, &enc_pkt);
        spdlog::trace("sail.Encoder: write_frame() av_interleaved_write_frame() finish, ret: {}", ret);
        if(ret < 0){
            write_frame_ret = ret;
            spdlog::error("sail.Encoder: av_interleaved_write_frame failed for one frame in cache queue: {}", write_frame_ret);
            continue;
        }
    }
    flush_encoder();
    write_frame_start_ = false;

    return;
}

int Encoder::Encoder_CC::video_write(bm_image &image)
{
    int ret = 0;
    // d2d convert, push into the cache queue.
    frame_process_lock.lock();
    if(frame_process_q.size()>cache_buffer_length_){
        switch(abort_policy_)
        {
            case RETURN_NOW:
                frame_process_lock.unlock();
                spdlog::error("sail.Encoder: cache queue is full, return now");
                return -CACHE_OVERFLOW;
            case POP_FRONT:
            {
                AVFrame* frame = frame_process_q.front();
                frame_process_q.pop();
                av_frame_free(&frame);
                break;
            }
            case POP_ALL:
            {
                while(frame_process_q.size()>0)
                {
                    AVFrame* frame = frame_process_q.front();
                    frame_process_q.pop();
                    av_frame_free(&frame);
                }
                break;
            }
            case BLOCKING:
                // unlock, encode thread will pop frame
                frame_process_lock.unlock();
                while(frame_process_q.size()>cache_buffer_length_)
                    av_usleep(20*1000);
                frame_process_lock.lock();
                break;
            default:
                frame_process_lock.unlock();
                spdlog::error("sail.Encoder: cache queue is full, but unknown abort policy");
                return -UNKNOEN_ABORT_POLICY;
        }
    }
    // push one frame
    AVFrame* frame = av_frame_alloc();
    spdlog::trace("sail.Encoder: video_write() bm_image_to_avframe() start");
    ret = bm_image_to_avframe(handle_, &image, frame);
    spdlog::trace("sail.Encoder: video_write() bm_image_to_avframe() finish, ret: {}", ret);
    if(ret==0)
    {
        frame_process_q.push(frame);
    }else{
        av_frame_free(&frame);
        spdlog::error("sail.Encoder: bm_image_to_avframe failed: {}", ret);
    }
    frame_process_lock.unlock();

    if(encode_ret < 0)
    {
        return -ERROR_ENCODE;
    }
    if(write_frame_ret < 0)
    {
        return -ERROR_WRITE_INTERLEAVE;
    }

    return ret;
}

int Encoder::Encoder_CC::flush_encoder()
{
    int ret = 0;
    if (!(enc_ctx_->codec->capabilities & AV_CODEC_CAP_DELAY))
        return 0;
    int64_t frame_interval = 1* 1000 * 1000 / params_map_["framerate"];
    
    SPDLOG_INFO("Encoder flushing cache ...");
    while (1) {
        AVPacket temp_enc_pkt;
        temp_enc_pkt.data = nullptr;
        temp_enc_pkt.size = 0;
        av_init_packet(&temp_enc_pkt);

        spdlog::debug("{} before avcodec_send_frame", __func__);
        ret = avcodec_send_frame(enc_ctx_, nullptr);
        spdlog::debug("{} avcodec_send_frame ret: {}", __func__, ret);
        while(1) {
            ret = avcodec_receive_packet(enc_ctx_, &temp_enc_pkt);
            spdlog::debug("Encoder flush avcodec_receive_packet ret: {}", ret);
            if (ret == AVERROR(EAGAIN)) {
                av_packet_unref(&temp_enc_pkt);
                break;
            } else if (ret == 0) {
                break;
            } else if (ret == AVERROR_EOF) {
                av_packet_unref(&temp_enc_pkt);
                SPDLOG_INFO("Encoder flush end");
                return ret;
            } else {
                SPDLOG_ERROR("Encoder avcodec_send_frame error, ret: {}", ret);
                return ret;
            }
        }

        if (ret != 0) {
            continue;
        }

        spdlog::debug("temp_enc_pkt.pts={}, temp_enc_pkt.dts={}", temp_enc_pkt.pts, temp_enc_pkt.dts);
        av_packet_rescale_ts(&temp_enc_pkt, this->enc_ctx_->time_base,this->out_stream_->time_base);
        spdlog::debug("rescaled temp_enc_pkt.pts={}, temp_enc_pkt.dts={}", temp_enc_pkt.pts, temp_enc_pkt.dts);
        ret = av_interleaved_write_frame(this->enc_format_ctx_, &temp_enc_pkt);
        if (ret < 0)
            break;
    }
    return ret;
}

int Encoder::Encoder_CC::reconnect()
{
    spdlog::info("sail.Encoder: reconnecting encoder.");

    if (opened_)
    {
        release();
    }
    int ret = 0;
    ret = bm_dev_request(&handle_, tpu_id_);
    if (BM_SUCCESS != ret)
    {
        SPDLOG_ERROR("Encoder bm_dev_request fail, device_id: {}", tpu_id_);
        return ret;
    }

    // get output format
    int output_type = get_output_type(output_path_);
    switch(output_type)
    {
        case RTSP_STREAM:
            spdlog::info("sail.Encoder: you are pushing a rtsp stream.");
            avformat_alloc_output_context2(&enc_format_ctx_, NULL, "rtsp", output_path_.c_str());
            is_rtsp_ = true;
            break;
        case RTMP_STREAM:
            spdlog::info("sail.Encoder: you are pushing a rtmp stream.");
            avformat_alloc_output_context2(&enc_format_ctx_, NULL, "flv", output_path_.c_str());
            is_rtmp_ = true;
            break;
        case BASE_STREAM:
            spdlog::error("sail.Encoder: Not support tcp/udp stream yet.");
            return -1;
        case VIDEO_LOCAL_FILE:
            spdlog::info("sail.Encoder: you are writing a local video file.");
            avformat_alloc_output_context2(&enc_format_ctx_, NULL, NULL, output_path_.c_str());
            break;
        default:
            spdlog::error("Failed to alloc output context.");
            return -1;
    }

    if (!enc_format_ctx_) {
        spdlog::error("sail.Encoder: Could not create output context\n");
        return -1;
    }

    // find encoder & alloc encoder context
    encoder_ = avcodec_find_encoder_by_name(enc_fmt_.c_str());
    if (!encoder_)
    {
        spdlog::error("Failed to find encoder: {} \n", enc_fmt_);
        return -1;
    }
    enc_ctx_ = avcodec_alloc_context3(encoder_);
    if (!enc_ctx_)
    {
        spdlog::error("Failed to alloc context. \n");
        return -1;
    }

    // SPS PPS do not be tied with IDR, global header.
    if(enc_format_ctx_->oformat->flags & AVFMT_GLOBALHEADER)
        enc_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    enc_params_prase();

    enc_ctx_->codec_id      =   encoder_->id;
    enc_ctx_->pix_fmt       =   pix_fmt_;
    enc_ctx_->width         =   params_map_["width"];
    enc_ctx_->height        =   params_map_["height"];
    enc_ctx_->gop_size      =   params_map_["gop"];
    enc_ctx_->time_base     =   (AVRational){1, params_map_["framerate"]};
    enc_ctx_->framerate     =   (AVRational){params_map_["framerate"], 1};
    if(-1 == params_map_["qp"])
    {
        enc_ctx_->bit_rate_tolerance = params_map_["bitrate"]*1000;
        enc_ctx_->bit_rate      =   (int64_t)params_map_["bitrate"]*1000;
    }else{
        av_dict_set_int(&enc_dict_, "qp", params_map_["qp"], 0);
    }

    av_dict_set_int(&enc_dict_, "sophon_idx", tpu_id_, 0);
    av_dict_set_int(&enc_dict_, "gop_preset", params_map_["gop_preset"], 0);
    av_dict_set_int(&enc_dict_, "is_dma_buffer", 1, 0);

    // open encoder
    ret = avcodec_open2(enc_ctx_, encoder_, &enc_dict_);
    if(ret < 0){
        spdlog::error("sail.Encoder: avcodec_open failed, return: {} \n.", ret);
        return ret;
    }
    av_dict_free(&enc_dict_);

    // new stream
    out_stream_ = avformat_new_stream(enc_format_ctx_, encoder_);
    out_stream_->time_base      = enc_ctx_->time_base;
    out_stream_->avg_frame_rate = enc_ctx_->framerate;
    out_stream_->r_frame_rate   = out_stream_->avg_frame_rate;

    ret = avcodec_parameters_from_context(out_stream_->codecpar, enc_ctx_);
    if(ret < 0)
    {
        spdlog::error("sail.Encoder: avcodec_parameters_from_context failed, return: {} \n.", ret);
        return ret;
    }

    if (!(enc_format_ctx_->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&enc_format_ctx_->pb, output_path_.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            spdlog::error("sail.Encoder: avio_open failed, return: {} \n.", ret);
            return ret;
        }
    }
    AVDictionary *header_options = NULL;
    if (output_type == RTSP_STREAM) {
        av_dict_set(&header_options, "rtsp_flags", "prefer_tcp", 0);
    }
    ret = avformat_write_header(enc_format_ctx_, &header_options);
    av_dict_free(&header_options);
    if (ret < 0) {
        spdlog::error("sail.Encoder: avformat_write_header failed, return: {} \n.", ret);
        return ret;
    }
    opened_ = true;
    quit_flag = false;
    first_frame_flag = true;
    frame_idx = 0;

    if(!write_frame_start_)
    {
        write_frame_thread_ = std::thread(&Encoder_CC::write_frame, this);
        write_frame_thread_.detach();
        write_frame_start_ = true;
    }
    spdlog::info("sail.Encoder: encoder opened.");
    return ret;
}

void Encoder::Encoder_CC::release()
{
    spdlog::info("sail.Encoder: releasing encoder.");

    // in case double release()
    if(quit_flag && !opened_)
    {
        spdlog::warn("sail.Encoder: encoder is likely double-released.");
        spdlog::info("sail.Encoder: released encoder completed.");
        return;
    }

    // tell write_frame() to quit and wait
    quit_flag = true;
    av_usleep(100*1000);
    while(write_frame_start_)
    {
        av_usleep(10*1000);
    }

    // release un-encoded frame(s)
    frame_process_lock.lock();
    while(frame_process_q.size()>0)
    {
        spdlog::debug("{}, frame_process_q.size: {}", __func__, frame_process_q.size());
        AVFrame* frame = frame_process_q.front();
        frame_process_q.pop();
        av_frame_free(&frame);
    }
    frame_process_lock.unlock();

    bm_dev_free(handle_);

    av_write_trailer(enc_format_ctx_);
    if (enc_dict_)
        av_dict_free(&enc_dict_);
    if (enc_ctx_)
        avcodec_free_context(&enc_ctx_);
    if (enc_format_ctx_ && !(enc_format_ctx_->oformat->flags & AVFMT_NOFILE))
        avio_closep(&enc_format_ctx_->pb);
    avformat_free_context(enc_format_ctx_);
    opened_ = false;
    spdlog::info("sail.Encoder: released encoder completed.");
}
} // namespace sail

#endif
