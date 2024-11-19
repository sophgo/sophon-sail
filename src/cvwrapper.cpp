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

#include "cvwrapper.h"
#include <spdlog/spdlog.h>
#include <stack>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <fstream>
#include "internal.h"
#include "tensor.h"

#ifdef USE_OPENCV

#include "opencv2/opencv.hpp"

#endif

#ifdef PYTHON
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eval.h>

using namespace pybind11::literals;
#endif

typedef unsigned char u8;

static int IMAGE_H = 1024;
static int IMAGE_W = 1920;

// Define constants for memory heap usage
#define USEING_MEM_HEAP2 4
#define USEING_MEM_HEAP1 2



#define AUTO_PTR(name, type, size) std::unique_ptr<type[]> up##name(new type[size]);\
                                   type *name=up##name.get();
namespace sail {
    inline bool string_start_with(const std::string &s, const std::string &head) {
        return s.compare(0, head.size(), head) == 0;
    }
    
    /**
     * @brief Get the decoder env int object
     * 
     * @param env_name 
     * @param value 
     * @return bool, if get value return true,else return false.
     */
    bool get_decoder_env_int(std::string env_name, int& value);

    /**
     * @brief Get the decoder env string object
     * 
     * @param env_name 
     * @param env_value return value
     * @return bool, if get value return true,else return false.
     */
    bool get_decoder_env_string(std::string env_name, std::string &env_value);

#ifdef USE_BMCV

    // Function to map AV format to BM format
    int map_avformat_to_bmformat(int avformat)
    {
        int format;
        switch(avformat){
            case AV_PIX_FMT_YUV420P: format = FORMAT_YUV420P; break;
            case AV_PIX_FMT_YUV422P: format = FORMAT_YUV422P; break;
            case AV_PIX_FMT_YUV444P: format = FORMAT_YUV444P; break;
            case AV_PIX_FMT_NV12:    format = FORMAT_NV12; break;
            case AV_PIX_FMT_NV16:    format = FORMAT_NV16; break;
            case AV_PIX_FMT_GRAY8:   format = FORMAT_GRAY; break;
            case AV_PIX_FMT_GBRP:    format = FORMAT_RGBP_SEPARATE; break;
            default: printf("unsupported av_pix_format %d\n", avformat); return -1;
        }
        return format;
    }
    // Function to AVFrame  to bm_image
    int avframe_to_bm_image(Handle& handle_,AVFrame &in, bm_image &out);

    bm_data_type_t get_bm_data_type_sail(bm_image_data_format_ext fmt) {
        std::string sfmt;
        switch (fmt) {
            case DATA_TYPE_EXT_FLOAT32:
                return BM_FLOAT32;
            case DATA_TYPE_EXT_1N_BYTE_SIGNED:
                return BM_INT8;
            case DATA_TYPE_EXT_1N_BYTE:
                return BM_UINT8;
#if !((BMCV_VERSION_MAJOR == 2) && !defined(BMCV_VERSION_MINOR))
            case DATA_TYPE_EXT_4N_BYTE_SIGNED:
                sfmt = "DATA_TYPE_EXT_4N_BYTE_SIGNED";
                break;
            case DATA_TYPE_EXT_4N_BYTE:
                sfmt = "DATA_TYPE_EXT_4N_BYTE";
                break;
#endif
            default:
                assert(0);
        }
        spdlog::error("No matching bm_data_type_t from bm_image_data_format_ext ({}).", sfmt);
        throw SailBMImageError("not supported");
    }

    bm_image_data_format_ext get_bm_image_data_format_sail(bm_data_type_t dtype) {
        std::string sfmt;
        switch (dtype) {
            case BM_FLOAT32:
                return DATA_TYPE_EXT_FLOAT32;
            case BM_INT8:
                return DATA_TYPE_EXT_1N_BYTE_SIGNED;
            case BM_UINT8:
                return DATA_TYPE_EXT_1N_BYTE;
            case BM_FLOAT16:
                sfmt = "BM_FLOAT16";
                break;
            case BM_INT16:
                sfmt = "BM_INT16";
                break;
            case BM_UINT16:
                sfmt = "BM_UINT16";
                break;
            case BM_INT32:
                sfmt = "BM_INT32";
                break;
            case BM_UINT32:
                sfmt = "BM_UINT32";
                break;
            default:
                assert(0);
        }
        spdlog::error("No matching bm_image_data_format_ext from bm_data_type_t ({}).", sfmt);
        throw SailBMImageError("not supported");
    }
#endif  //USE_BMCV

#ifdef USE_OPENCV

    inline bool check_shape(const std::vector<cv::Mat> &mats) {
        if (mats.empty()) return false;
        int h = mats[0].rows;
        int w = mats[0].cols;
        for (size_t i = 1; i < mats.size(); i++) {
            if (mats[i].rows != h || mats[i].cols != w) return false;
        }
        return true;
    }

    int get_cv_depth(bm_data_type_t dtype) {
        std::string sfmt;
        switch (dtype) {
            case BM_FLOAT32:
                return CV_32F;
            case BM_INT8:
                return CV_8S;
            case BM_UINT8:
                return CV_8U;
            case BM_FLOAT16:
                sfmt = "BM_FLOAT16";
                break;
            case BM_INT16:
                return CV_16S;
            case BM_UINT16:
                return CV_16U;
            case BM_INT32:
                return CV_32S;
            case BM_UINT32:
                sfmt = "BM_UINT32";
                break;
            default:
                assert(0);
        }
        spdlog::error("No matching cv dtype from bm_data_type_t ({}).", sfmt);
        throw SailBMImageError("not supported");
    }

    template<typename T>
    void mat_to_tensor_(std::vector<cv::Mat> &mats, Tensor &tensor) {
        if (!check_shape(mats)) {
            spdlog::error("mat_to_tensor(): check mat shape failed.");
            throw SailBMImageError("invalid argument");
        }
        if (!tensor.own_sys_data() || !tensor.own_dev_data()) {
            spdlog::error("mat_to_tensor(): tensor should own sys & dev memory.");
            throw SailBMImageError("invalid argument");
        }

        int n = static_cast<int>(mats.size());
        int c = 3;
        int h = mats[0].rows;
        int w = mats[0].cols;
        int depth = get_cv_depth(tensor.dtype());

        tensor.reshape({n, c, h, w});

        T *addr = reinterpret_cast<T *>(tensor.sys_data());
        for (auto mat : mats) {
            if (mat.depth() != depth) {
                mat.convertTo(mat, CV_MAKETYPE(depth, 3));
            }
            std::vector<cv::Mat> channels;
            for (int i = 0; i < mat.channels(); i++) {
                channels.emplace_back(h, w, CV_MAKETYPE(mat.depth(), 1), addr);
                addr += h * w;
            }
            cv::split(mat, channels);
        }
    }

    void mat_to_tensor(std::vector<cv::Mat> &mats, Tensor &tensor) {
        if (mats.empty()) return;
        if (mats[0].depth() == CV_32F) {
            mat_to_tensor_<float>(mats, tensor);
        } else {
            mat_to_tensor_<int8_t>(mats, tensor);
        }
    }

    void mat_to_tensor(cv::Mat &mat, sail::Tensor &tensor) {
        std::vector<cv::Mat> mats{mat};
        mat_to_tensor(mats, tensor);
    }

#endif

    inline bool is_pic_file(const std::string &filename) {
        size_t len = filename.size();
        return (filename.compare(len - 3, 3, "jpg") == 0 ||
                filename.compare(len - 3, 3, "JPG") == 0 ||
                filename.compare(len - 4, 4, "jpeg") == 0 ||
                filename.compare(len - 4, 4, "JPEG") == 0 ||
                filename.compare(len - 3, 3, "bmp") == 0 ||
                filename.compare(len - 3, 3, "BMP") == 0 ||
                filename.compare(len - 3, 3, "png") == 0 ||
                filename.compare(len - 3, 3, "PNG") == 0);
    }

    PaddingAtrr::PaddingAtrr(unsigned int crop_start_x,
                            unsigned int crop_start_y,
                            unsigned int crop_width,
                            unsigned int crop_height,
                            unsigned char padding_value_r,
                            unsigned char padding_value_g,
                            unsigned char padding_value_b):
      dst_crop_stx(crop_start_x),dst_crop_sty(crop_start_y),dst_crop_w(crop_width),dst_crop_h(crop_height),
      padding_r(padding_value_r),padding_g(padding_value_g),padding_b(padding_value_b){};

    PaddingAtrr::PaddingAtrr(const PaddingAtrr& other)
    {
        dst_crop_stx = other.dst_crop_stx;
        dst_crop_sty = other.dst_crop_sty;
        dst_crop_w = other.dst_crop_w;
        dst_crop_h = other.dst_crop_h;
        padding_r = other.padding_r;
        padding_g = other.padding_g;
        padding_b = other.padding_b;
    }

    void PaddingAtrr::set_stx(unsigned int stx) {
        dst_crop_stx = stx;
    }

    void PaddingAtrr::set_sty(unsigned int sty) {
        dst_crop_sty = sty;
    }

    void PaddingAtrr::set_w(unsigned int w) {
        dst_crop_w = w;
    }

    void PaddingAtrr::set_h(unsigned int h) {
        dst_crop_h = h;
    }

    void PaddingAtrr::set_r(unsigned int r) {
        padding_r = r;
    }

    void PaddingAtrr::set_g(unsigned int g) {
        padding_g = g;
    }

    void PaddingAtrr::set_b(unsigned int b) {
        padding_b = b;
    }

    int set_decoder_env(std::string env_name, std::string env_value)
    {
        std::string env_name_temp = std::string("SAIL_DECODER_")+env_name;
        return setenv(env_name_temp.c_str(), env_value.c_str(), 1);
    }

    bool get_decoder_env_int(std::string env_name, int& value)
    {
        std::string env_name_temp = std::string("SAIL_DECODER_")+env_name;
        const char *e_value = getenv(env_name_temp.c_str());
        if(e_value != nullptr){
            value = atoi(e_value);
            return true;
        }
        return false;
    }

    bool get_decoder_env_string(std::string env_name, std::string &env_value)
    {
        std::string env_name_temp = std::string("SAIL_DECODER_")+env_name;
        const char *e_value = getenv(env_name_temp.c_str());
        if(e_value != nullptr){
            env_value = std::string(e_value);
            return true;
        }
        return false;
    }

#ifdef USE_FFMPEG

    class Decoder::Decoder_CC{
    public:
        explicit Decoder_CC(
            const std::string& file_path,
            bool               compressed,
            int                tpu_id);

        ~Decoder_CC();

        int decode_jpeg(Handle& handle, bm_image& image);
        
        int read_(Handle& handle, bm_image& image);

        float get_fps() const;

        void release();

        void enable_dump(int dump_max_seconds);

        void disable_dump();

        int dump(int dump_pre_seconds, int dump_post_seconds, std::string& file_path);

    private:
        friend class Decoder;

        /**
         * @brief Grabs the next frame.
         *
         * @param frame Reference of frame to be read to
         * @return True for success and false for failure
         */
        bool grab(Frame& frame);
        /**
         * @brief Convert frame with format of AV_PIX_FMT_NV12 to bm_image.
         *
         * @param image Reference of BMImage to convert to
         */
        sail_status_t nv12_frame_to_image(Handle& handle, bm_image& image);
        void convert_to_yuv420p();
        void reset_decode(const std::string& file_path,
                        bool   compressed = true,
                        int    tpu_id = 0);
        void reconnect();

        /// Path to the Decoder file.
        std::string file_path_;
        /// TPU ID
        int tpu_id_;
        /// Pointer to an AVFormatContext instance.
        AVFormatContext* fmt_ctx_;
        /// Pointer to an AVCodecContext instance.
        AVCodecContext* video_dec_ctx_;
        /// Pointer to an AVStream instance.
        AVStream* video_stream_;
        /// Index of stream.
        int video_stream_idx_;
        /// An AVPacket instance.
        AVPacket pkt_;
        /// Count of Decoder frames.
        int video_frame_count_;
        /// Number of decoded frame of a packet.
        int got_frame_;
        /// Hight of frame.
        int height_;
        /// Width of frame.
        int width_;
        /// bm_handle
        bm_handle_t handle_;
        /// Decoded frame
        Frame frame_;
        /// Indicator of whether the frame is compressed.
        bool compressed_;
        /// Status of opening the Decoder file.
        bool opened_;
        /// Indicator of whether the input source is rtsp stream.
        bool is_rtsp_;
        /// Indicator of whether the input source is jpg image file.
        bool is_pic_file_;
        /// Flag of whether to read to end of the video.
        bool end_of_file_;
        int errcnt_;
        // video dump
        /// av_packet cache queue length
        int dump_buffer_length=1000;
        /// dump thread args
        typedef struct DUMP_THREAD_ARG{
            int fps;
            int dump_pre_frames;
            int dump_post_frames;
            const char* file_name;
        } DUMP_THREAD_ARG;

         //dts、pts
        bool keep_timestap;
        double pts_=0.0;
        double dts_=0.0;
        // get dts pts
        vector<double> get_pts_dts();

        int decoder_id=0;
        int key_frame_flag=0;
        bool enable_video_dump = false;
        std::mutex lock;
        std::deque<AVPacket*> pkt_cache_q;
        /// put av_packet into cache queue  
        void put_pkt(AVPacket* pkt);
        /// dump video write
        void dump_write(void* arg);

        std::string refcounted_frames_value;
        std::string extra_frame_buffer_num_value;
        std::string rtsp_transport_value;
        std::string stimeout_value;
        std::string rtsp_flags_value;
        int buffer_size_value;
        int max_delay_value;
        std::string probesize;
        std::string analyzeduration;
        std::string skip_non_idr;
        std::string fflags = "";
        int get_ffmpeg_valuse()
        {
            refcounted_frames_value = "1";
            extra_frame_buffer_num_value = "2";
            rtsp_transport_value = "tcp";
            stimeout_value = "20000000";
            rtsp_flags_value = "prefer_tcp";
            buffer_size_value = 1024000;
            max_delay_value = 500000;
            probesize = "0";
            analyzeduration = "0";
            get_decoder_env_string("refcounted_frames", refcounted_frames_value);
            get_decoder_env_string("extra_frame_buffer_num", extra_frame_buffer_num_value);
            get_decoder_env_string("rtsp_transport", rtsp_transport_value);
            get_decoder_env_string("stimeout", stimeout_value);
            get_decoder_env_string("rtsp_flags", rtsp_flags_value);
            get_decoder_env_int("buffer_size", buffer_size_value);
            get_decoder_env_int("max_delay", max_delay_value);
            get_decoder_env_string("probesize", probesize);             // 400
            get_decoder_env_string("analyzeduration", analyzeduration); // 100
            return 0;
        }

        bool need_flush_;
        bool flush();

#ifdef USE_OPENCV
          std::shared_ptr<cv::Mat> m1 = nullptr;
#endif
    };

    Decoder::Decoder_CC::Decoder_CC(
            const std::string &file_path,
            bool compressed,
            int tpu_id)
            : handle_(nullptr), file_path_(file_path), tpu_id_(tpu_id), fmt_ctx_(nullptr),
              video_dec_ctx_(nullptr), video_stream_(nullptr), video_stream_idx_(-1),
              video_frame_count_(0), got_frame_(0), height_(0), width_(0),
              compressed_(compressed), is_rtsp_(false), is_pic_file_(false), errcnt_(0),
              opened_(false), end_of_file_(false),keep_timestap(true),need_flush_(false) {
        printEnvVarsWithPrefix("SAIL_DECODER_");
        if (is_pic_file(file_path_)) {
            is_pic_file_ = true;
            opened_ = true;
            return;
        }
        std::cout << "decoder ctor: filepath=" << file_path << std::endl;
        // register all formats and codecs
        av_register_all();
        avdevice_register_all();
        AVDictionary *opts = NULL;
#ifndef IS_SOC_MODE
        av_dict_set_int(&opts, "pcie_board_id", tpu_id_, 0);
#endif
        if (file_path_.compare(0, 5, "rtsp:") == 0) {
            is_rtsp_ = true;
            get_ffmpeg_valuse();

            SPDLOG_INFO("refcounted_frames: {}",refcounted_frames_value);
            SPDLOG_INFO("extra_frame_buffer_num:{}",extra_frame_buffer_num_value);
            SPDLOG_INFO("rtsp_transport: {}",rtsp_transport_value);
            SPDLOG_INFO("stimeout: {}",stimeout_value);
            SPDLOG_INFO("rtsp_flags: {}",rtsp_flags_value);
            SPDLOG_INFO("buffer_size: {}",buffer_size_value);
            SPDLOG_INFO("max_delay: {}",max_delay_value);
            SPDLOG_INFO("probesize: {}",probesize);
            SPDLOG_INFO("analyzeduration: {}",analyzeduration);

            avformat_network_init();
            // Init the decoders, with reference counting
            av_dict_set(&opts, "refcounted_frames", refcounted_frames_value.c_str(), 0);
            // frame buffer set,same as opencv, ost is 20
            av_dict_set(&opts, "extra_frame_buffer_num", extra_frame_buffer_num_value.c_str(), 0);
            // set tcp
            av_dict_set(&opts, "rtsp_transport", rtsp_transport_value.c_str(), 0);
            // set timeout (same as opencv),ost is 10000000
            av_dict_set(&opts, "stimeout", stimeout_value.c_str(), 0);
            av_dict_set(&opts, "timeout", stimeout_value.c_str(), 0);

            // add same as opencv
            av_dict_set(&opts, "rtsp_flags", rtsp_flags_value.c_str(), 0);
            av_dict_set_int(&opts, "buffer_size", buffer_size_value, 0);
            av_dict_set_int(&opts, "max_delay", max_delay_value, 0);

            if(keep_timestap){
                //add timestamp
                av_dict_set(&opts, "keep_rtsp_timestamp", "1", 0);
            }

            if (probesize != "0" && analyzeduration != "0"){
                av_dict_set(&opts, "probesize", probesize.c_str(), 0);
                av_dict_set(&opts, "analyzeduration", analyzeduration.c_str(), 0);
            }
        }
        if (!is_pic_file_ && compressed_) {
            // set compressed output
            av_dict_set(&opts, "output_format", "101", 0);
        }
        skip_non_idr = "0";
        if (get_decoder_env_string("skip_non_idr", skip_non_idr)) {
            av_dict_set(&opts, "skip_non_idr", skip_non_idr.c_str(), 0);
            SPDLOG_INFO("skip_non_idr: {}", skip_non_idr);
        }
        if (get_decoder_env_string("fflags", fflags)) {
            av_dict_set(&opts, "fflags", fflags.c_str(), 0);
            SPDLOG_INFO("fflags: {}", fflags);
        }
        //av_log_set_level(AV_LOG_TRACE);
        std::cout << "open filepath=" << file_path << std::endl;
        AVInputFormat *input_fmt = nullptr;
        if (string_start_with(file_path, "/dev/video")) {
            input_fmt = av_find_input_format("video4linux2");
            if (input_fmt == nullptr) {
                printf("ERROR:can't find format: video4linux2\n");
            } else {
                printf("find video4linux2 success!\n");
                const char *pixfmt = getenv("SAIL_CAP_PIXFMT");
                if (pixfmt == nullptr) pixfmt = "mjpeg";
                av_dict_set(&opts, "pixel_format", pixfmt, 0);
            }
        }

        // open input file, and allocate format context
        int ret = avformat_open_input(&fmt_ctx_, file_path_.c_str(),
                                      input_fmt, &opts);
        if (ret < 0) {
            SPDLOG_ERROR("Failed to open input file: {}", file_path_);
            throw std::runtime_error("Failed to open input file");
        }
        // retrieve stream information
        ret = avformat_find_stream_info(fmt_ctx_, nullptr);
        if (ret < 0) {
            spdlog::error("Failed to find stream information.");
            throw std::runtime_error("Failed to find stream information");
        }

        ret = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (ret < 0) {
            spdlog::error("Failed to find a video stream.");
            throw std::runtime_error("Failed to find a video stream");
        }
        video_stream_idx_ = ret;
        video_stream_ = fmt_ctx_->streams[video_stream_idx_];
        
        AVCodec *dec = avcodec_find_decoder(video_stream_->codecpar->codec_id);
        if (!dec) {
            spdlog::error("Failed to find codec.");
            throw std::runtime_error("Failed to find codec");
        }

        video_dec_ctx_ = avcodec_alloc_context3(dec);
		avcodec_parameters_to_context(video_dec_ctx_, video_stream_->codecpar);

        ret = avcodec_open2(video_dec_ctx_, dec, &opts);
        if (ret < 0) {
            spdlog::error("Failed to open codec");
            throw std::runtime_error("Failed to open codec");

        }
        height_ = video_dec_ctx_->height;
        width_ = video_dec_ctx_->width;

        // initialize packet, set data to NULL, let the demuxer fill it
        av_init_packet(&pkt_);
        pkt_.data = nullptr;
        pkt_.size = 0;
        opened_ = true;
        // destroy avdict
        av_dict_free(&opts);
    }

    Decoder::Decoder_CC::~Decoder_CC() {
        release();
    }

    void Decoder::Decoder_CC::enable_dump(int dump_max_seconds)
    {
        int fps = get_fps();
        dump_buffer_length = dump_max_seconds*fps;
        decoder_id = video_dec_ctx_->codec_id;
        if(AV_CODEC_ID_H264!=decoder_id && AV_CODEC_ID_HEVC!=decoder_id)
        {
            spdlog::error("decoder video dump only support h264 and h265 {},{}", __FILE__, __LINE__);
            return;
        }
        key_frame_flag = (AV_CODEC_ID_H264==decoder_id) ? 0x07 : 0x20;
        enable_video_dump = true;
    }

    void Decoder::Decoder_CC::disable_dump()
    {
        enable_video_dump = false;
        while(pkt_cache_q.size()>0)
        {
            /* pop a pkt */
            AVPacket* temp_pkt = pkt_cache_q.front();
            av_packet_unref(temp_pkt);
            pkt_cache_q.pop_front();
        }
        return;
    }

    void Decoder::Decoder_CC::put_pkt(AVPacket* pkt)
    {
        std::lock_guard<std::mutex> my_lock_guard(lock);
        if(pkt_cache_q.size()>=dump_buffer_length)
        {
            /* pop a pkt */
            AVPacket* temp_pkt = pkt_cache_q.front();
            av_packet_unref(temp_pkt);
            pkt_cache_q.pop_front();
        }
        AVPacket* cache_pkt = av_packet_alloc();
        av_packet_ref(cache_pkt, pkt);
        pkt_cache_q.push_back(cache_pkt);

        return;
    }

    vector<double> Decoder::Decoder_CC::get_pts_dts()
    {
       return {pts_,dts_};
    }

    void Decoder::Decoder_CC::dump_write(void* arg)
    {
        SPDLOG_TRACE("start dump_write");
        bool h265_stream_file = false;
        DUMP_THREAD_ARG* dump_thread_arg = (DUMP_THREAD_ARG*)arg;
        int fps = dump_thread_arg->fps;
        int dump_pre_frames = dump_thread_arg->dump_pre_frames;
        int dump_post_frames = dump_thread_arg->dump_post_frames;
        const char* file_name = dump_thread_arg->file_name;
        if(strstr(file_name, ".h265")||strstr(file_name, ".H265"))
            h265_stream_file=true;

        int dump_frame_pos = video_frame_count_;

        SPDLOG_TRACE("start while");
        while(video_frame_count_<(dump_frame_pos+dump_post_frames))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }   

        SPDLOG_TRACE("start stack");
        std::stack<AVPacket*> write_cache_q;
        {
            std::lock_guard<std::mutex> my_lock_guard(lock);
            int pkt_count = pkt_cache_q.size();
            int pkts = dump_post_frames + dump_pre_frames; 
            for(int i=pkt_count; i>0; i--)
            {
                AVPacket* cache_pkt = pkt_cache_q[i-1];
                AVPacket* write_pkt = av_packet_alloc();
                av_packet_ref(write_pkt, cache_pkt);
                write_cache_q.push(write_pkt);
            }
        }

        for(int i=0; i<write_cache_q.size(); i++)
        {
            AVPacket* write_pkt = write_cache_q.top();
            int nal_type = 0;
            if(AV_CODEC_ID_H264==decoder_id)
            {
                if (0 == write_pkt->data[0] &&
                    0 == write_pkt->data[1] &&
                    0 == write_pkt->data[2] &&
                    1 == write_pkt->data[3]){
                    nal_type = write_pkt->data[4] & 0x1f;
                }else if(0 == write_pkt->data[0] &&
                        0 == write_pkt->data[1] &&
                        1 == write_pkt->data[2]){
                    nal_type = write_pkt->data[3] & 0x1f;
                }
            }
            if(AV_CODEC_ID_HEVC==decoder_id)
            {
                if (0 == write_pkt->data[0] &&
                    0 == write_pkt->data[1] &&
                    0 == write_pkt->data[2] &&
                    1 == write_pkt->data[3]){
                    nal_type = (write_pkt->data[4] & 0x7e) >> 1;
                }else if(0 == write_pkt->data[0] &&
                        0 == write_pkt->data[1] &&
                        1 == write_pkt->data[2]){
                    nal_type = (write_pkt->data[3] & 0x7e) >> 1;
                }
            }
            if((key_frame_flag==nal_type))
            {
                break;
            }
            av_packet_unref(write_pkt);
            write_cache_q.pop();
            if(write_cache_q.size()==dump_post_frames)
                spdlog::warn("dump pre seconds does not contain key frame, the video duration will be shorter than expected. {},{}", __FILE__, __LINE__);
            if(write_cache_q.size()==0)
            {
                spdlog::error("video-dump cache queue does not contain any key frame, maybe gop is too large, needs longer dump length. {},{}", __FILE__, __LINE__);
                return;
            }
        }

        SPDLOG_TRACE("start write");
        int time_stamp = 0;
        if(h265_stream_file)
        {
            FILE* fp = fopen(file_name, "wb+");
            for(int i=0; i<write_cache_q.size(); i++)
            {
                AVPacket* write_pkt = write_cache_q.top();                
                uchar* data = write_pkt->data;
                int size = write_pkt->size;

                if(i==0)
                {
                    void* head = malloc(1);
                    memset(head, 0, 1);
                    fwrite(head, 1, 1, fp);
                    free(head);
                }

                if(i==write_cache_q.size()-1)
                    size -= 1;

                fwrite(data, 1, size, fp);
                av_packet_unref(write_pkt);
                write_cache_q.pop();
            }
            fclose(fp);
        }else{
            AVFormatContext* out_fmt_ctx=NULL;
            avformat_alloc_output_context2(&out_fmt_ctx, NULL, NULL, file_name);
            AVOutputFormat* out_fmt = av_guess_format(NULL, file_name, NULL);
            if(out_fmt->video_codec == AV_CODEC_ID_NONE){
                spdlog::error("Unable to assign encoder automatically by file name. {},{}", __FILE__, __LINE__);
                return;
            }
            out_fmt_ctx->oformat = out_fmt;
            AVStream *in_stream = fmt_ctx_ -> streams[video_stream_idx_];

#if LIBAVCODEC_VERSION_MAJOR > 58
            //新版ffmpeg中AVStream中没有AVCodecContext，需额外声明一个中间变量AVCodecContext
            AVCodecContext* codec_ctx = avcodec_alloc_context3(avcodec_find_decoder(in_stream->codecpar->codec_id));
            AVStream *out_stream = avformat_new_stream(out_fmt_ctx, NULL);
#else
            AVStream *out_stream = avformat_new_stream(out_fmt_ctx, in_stream->codec->codec);
#endif
            if(!out_stream){
                spdlog::error("Faild allocating output stream {},{}", __FILE__, __LINE__);
            }

#if LIBAVCODEC_VERSION_MAJOR > 58
            int ret = avcodec_parameters_to_context(codec_ctx,in_stream->codecpar);
#else
            int ret = avcodec_copy_context(out_stream->codec, in_stream->codec);
#endif     
            if(ret < 0)
                spdlog::error("Failed to copy context from input to output stream codec context {},{}", __FILE__, __LINE__);

#if LIBAVCODEC_VERSION_MAJOR > 58   
            codec_ctx->codec_tag = 0;
            if(out_fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
                codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            ret = avcodec_parameters_from_context(out_stream->codecpar, codec_ctx);
            avcodec_free_context(&codec_ctx);
            if ( ret < 0) {
                spdlog::error("Failed to copy codec parameters from context to parameters");
                return;
            }
#else
            out_stream->codec->codec_tag = 0;
            if(out_fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
                out_stream->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
#endif

            if(!(out_fmt_ctx -> oformat ->flags & AVFMT_NOFILE)){
                ret = avio_open(&out_fmt_ctx->pb, file_name, AVIO_FLAG_WRITE);
                if(ret < 0){
                    spdlog::error("Could not open output URL {},{}", __FILE__, __LINE__);
                }
            }
            ret = avformat_write_header(out_fmt_ctx, NULL);
            if(ret < 0){
                spdlog::error("Error occurred when opening output URL {},{}", __FILE__, __LINE__);
            }

            while (!write_cache_q.empty())
            {
                SPDLOG_TRACE("write_cache_q.size(): {}", write_cache_q.size());
                AVPacket* write_pkt = write_cache_q.top();
                if(write_pkt->pts==AV_NOPTS_VALUE)
                {
                    write_pkt->dts = av_rescale_q_rnd(time_stamp, AVRational({1, fps}), out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
                    write_pkt->pts = av_rescale_q_rnd(time_stamp, AVRational({1, fps}), out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
                    write_pkt->duration = av_rescale_q_rnd(1, AVRational({1, fps}), out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
                    time_stamp++;
                }else{
                    write_pkt->dts = av_rescale_q_rnd(write_pkt->dts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
                    write_pkt->pts = av_rescale_q_rnd(write_pkt->pts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
                    write_pkt->duration = av_rescale_q_rnd(write_pkt->duration, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
                }

                av_interleaved_write_frame(out_fmt_ctx, write_pkt);
                av_packet_unref(write_pkt);
                write_cache_q.pop();
            }
            SPDLOG_TRACE("av_write_trailer");
            av_write_trailer(out_fmt_ctx);
        }
        return;
    }

    /***********************
     * do not need to encode, just dump AVPacket(s) when decode
     * 
     * |---gop----|---gop----|---gop----|---gop----|
     * IPPPPPPPPPPIPPPPPPPPPPIPPPPPPPPPPIPPPPPPPPPPI
     *                     ^
     *                     (dump the stream before and after the current frame)
     * The frame sequence before the current frame must contain key frame. So the length of dump video is calculated based on gop size.
    */
    int Decoder::Decoder_CC::dump(int dump_pre_seconds, int dump_post_seconds, std::string& file_name)
    {
        int fps = get_fps();

        int dump_pre_frames = dump_pre_seconds*fps;
        int dump_post_frames = dump_post_seconds*fps;

        if(dump_pre_frames+dump_post_frames > dump_buffer_length)
        {
            spdlog::error("dump video length larger than cache queue length. pre+psot: {}, cache: {}, check the max buffer length when you enable_dump() {},{}", dump_pre_frames+dump_post_frames, dump_buffer_length, __FILE__, __LINE__);
            return -1;
        }

        DUMP_THREAD_ARG* dump_thread_arg = (DUMP_THREAD_ARG*)malloc(sizeof(DUMP_THREAD_ARG));
        dump_thread_arg->fps = fps;
        dump_thread_arg->dump_pre_frames = dump_pre_frames;
        dump_thread_arg->dump_post_frames = dump_post_frames;
        char* file = (char*)malloc(file_name.length());
        strcpy(file,file_name.c_str());
        dump_thread_arg->file_name = file;
        
        std::thread dump_write_thread(&Decoder_CC::dump_write, this, (void*)dump_thread_arg);
        SPDLOG_TRACE("start thread");
        dump_write_thread.detach();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        free(dump_thread_arg);
        dump_thread_arg = nullptr;
        return 0;
    }

    bool Decoder::Decoder_CC::grab(Frame &frame) {
        if (end_of_file_ && !need_flush_) {
            return false;
        }
        AVFrame *p_frame = frame.get();
        bool valid = false;
        while (!valid && !need_flush_) {
            av_packet_unref(&pkt_);
            int ret = av_read_frame(fmt_ctx_, &pkt_);
            if (ret == AVERROR(EAGAIN)) {
                continue;
            }
            if (ret < 0) {
                if (ret == static_cast<int>(AVERROR_EOF)) {
                    spdlog::debug("Meet AVERROR_EOF. Need to flush decoder");
                    end_of_file_ = true;
                    need_flush_ = true;
                    break;
                }
                break;
            }
            if (pkt_.stream_index != video_stream_idx_) {
                av_packet_unref(&pkt_);
                continue;
            }

            // 获取时间戳（单位是 AVStream.time_base）
            int64_t pts = (int64_t )pkt_.pts;
            int64_t dts = (int64_t )pkt_.dts;
            AVRational time_base = fmt_ctx_->streams[video_stream_idx_]->time_base;
            // 转换时间戳为秒
            pts_ = pts * av_q2d(time_base);
            dts_ = dts * av_q2d(time_base);

            // decode video frame
            ret = avcodec_decode_video2(video_dec_ctx_, p_frame, &got_frame_, &pkt_);
            if (got_frame_) {
                valid = true;
            }

            // dump
            if(enable_video_dump)
                put_pkt(&pkt_);

        }
        if(need_flush_)
        {
            return flush();
        }
        if (valid) {
            ++video_frame_count_;
        }
        return valid;
    }

    bool Decoder::Decoder_CC::flush()
    {
        int ret = 0;
        AVFrame *p_frame = frame_.get();
        av_frame_unref(p_frame);
        ret = avcodec_send_packet(video_dec_ctx_, NULL);
        ret = avcodec_receive_frame(video_dec_ctx_, p_frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF || ret < 0)
        {
            spdlog::debug("All cached frames in decoder are flushed.");
            need_flush_ = false;
            return false;
        }
        ++video_frame_count_;
        return true;
    }

    int Decoder::Decoder_CC::decode_jpeg(Handle &handle, bm_image &image) {
        std::ifstream filestr;

        filestr.open(file_path_, std::ios::binary);
        // check if file exists
        if(filestr.good()){
            spdlog::debug("Decode pic exist: filepath={}",file_path_);
        }else{
            spdlog::error("Decode cannot open file. Please check the file path. filepath={}",file_path_);
            return SAIL_ERR_DEC_OPEN;
        }

        std::filebuf *pbuf = filestr.rdbuf();
        size_t size = pbuf->pubseekoff(0, std::ios::end, std::ios::in);
        char *buffer = new char[size];
        pbuf->pubseekpos(0, std::ios::in);
        pbuf->sgetn((char *) buffer, size);
        bm_handle_t bm_handle = handle.data();
        bm_image image1;
        memset(&image1, 0, sizeof(image1));
        int ret = BM_SUCCESS;
        // int ret = bmcv_image_jpeg_dec(bm_handle, (void **) &buffer, &size, 1, &image1);
        // if (BM_SUCCESS != ret) {
#ifdef USE_OPENCV
        //   spdlog::info(
        //       "bmcv_image_jpeg_dec err={}, fallback to software decode ...\n",
        //       ret);
          std::vector<char> pic(buffer, buffer+size);
          m1 = std::make_shared<cv::Mat>();
        //   m1.allocator = cv::hal::getAllocator();
          cv::imdecode(pic, cv::IMREAD_COLOR, m1.get(), handle.get_device_id());
          memset(&image, 0, sizeof(image));
          ret = cv::bmcv::toBMI(*m1, &image);
          if (ret != BM_SUCCESS) {
            spdlog::error("cv::bmcv::toBMI() err {},{}", __FILE__, __LINE__);
            ret = BM_NOT_SUPPORTED;
          }else {
            ret = BM_SUCCESS;
          }
#else
          ret = BM_NOT_SUPPORTED;
#endif
        // } else {
        //     if (image1.width % 2 != 0 || image1.height % 2 != 0) {
        //         // width and height align 2
        //         int new_w = FFALIGN(image1.width, 2);
        //         int new_h = FFALIGN(image1.height, 2);
        //         bm_image image2;
        //         int stride = FFALIGN(new_w, SAIL_ALIGN);
        //         ret = bm_image_create(bm_handle, new_h, new_w, FORMAT_BGR_PLANAR,
        //                         DATA_TYPE_EXT_1N_BYTE, &image2, &stride);
        //         assert(ret == 0);
        //         ret = bm_image_alloc_dev_mem_heap_mask(image2, 6);
        //         if (ret != 0) {
        //             SPDLOG_ERROR("bm_image_alloc_dev_mem_heap_mask err={}", ret);
        //             bm_image_destroy(image2);
        //             bm_image_destroy(image1);
        //             return ret;
        //         }
        //         ret = bmcv_image_vpp_csc_matrix_convert(bm_handle, 1, image1, &image2, CSC_YPbPr2RGB_BT601);
        //         if (0 != ret) {
        //             SPDLOG_ERROR("bmcv_image_vpp_convert error={}", ret);
        //             print_image(image1, "src:");
        //             print_image(image2, "dst:");
        //         }
        //         image = image2;
        //         bm_image_destroy(image1);
        //     } else {
        //         image = image1;
        //     }
        // }

        filestr.close();
        delete[] buffer;
        return ret;
    }

    sail_status_t Decoder::Decoder_CC::nv12_frame_to_image(Handle &handle, bm_image &image) {
        AVFrame *p_frame = frame_.get();
        bm_image temp_img;
        bm_status_t ret = BM_SUCCESS;
        if (compressed_) {
            spdlog::debug("coded_width {} coded_height {}",
                         video_dec_ctx_->coded_width, video_dec_ctx_->coded_height);
            spdlog::debug("width {} height {}", frame_.get_width(), frame_.get_height());
            ret = bm_image_create(handle.data(),
                                  video_dec_ctx_->coded_height,
                                  video_dec_ctx_->coded_width,
                                  FORMAT_COMPRESSED,
                                  DATA_TYPE_EXT_1N_BYTE,
                                  &temp_img);
            if (BM_SUCCESS != ret)
            {
                SPDLOG_ERROR("nv12_frame_to_image failed, bm_image_create ret: {}.", ret);
                return SAIL_ERR_BMI_INIT;
            }

            // calculate physical address of frame
            bm_device_mem_t input_addr[4];
            int size = frame_.get_height() * p_frame->linesize[4];
            input_addr[0] = bm_mem_from_device((unsigned long long) p_frame->data[6], size);
            size = (frame_.get_height() / 2) * p_frame->linesize[5];
            input_addr[1] = bm_mem_from_device((unsigned long long) p_frame->data[4], size);
            size = p_frame->linesize[6];
            input_addr[2] = bm_mem_from_device((unsigned long long) p_frame->data[7], size);
            size = p_frame->linesize[7];
            input_addr[3] = bm_mem_from_device((unsigned long long) p_frame->data[5], size);
            ret = bm_image_attach(temp_img, input_addr);
            if (BM_SUCCESS != ret)
            {
                SPDLOG_ERROR("nv12_frame_to_image failed, bm_image_attach ret: {}.", ret);
                return SAIL_ERR_BMI_BMCV;
            }

        } else {
            int stride[2];
            stride[0] = p_frame->linesize[4];
            stride[1] = p_frame->linesize[5];
            ret = bm_image_create(handle.data(),
                            frame_.get_height(),
                            frame_.get_width(),
                            FORMAT_NV12,
                            DATA_TYPE_EXT_1N_BYTE,
                            &temp_img,
                            stride);
            if (BM_SUCCESS != ret)
            {
                SPDLOG_ERROR("nv12_frame_to_image failed, bm_image_create ret: {}.", ret);
                return SAIL_ERR_BMI_INIT;
            }
            // calculate physical address of yuv data
            bm_device_mem_t input_addr[2];
            int size = p_frame->height * stride[0];
            input_addr[0] = bm_mem_from_device((unsigned long long) p_frame->data[4], size);
            size = p_frame->height / 2 * stride[1];
            input_addr[1] = bm_mem_from_device((unsigned long long) p_frame->data[5], size);
            // attach memory to bm_image
            bm_image_attach(temp_img, input_addr);
        }

        if (image.image_private == nullptr || image.width == 0 || image.height == 0) {
            ret = bm_image_create(handle.data(),
                                  frame_.get_height(),
                                  frame_.get_width(),
                                  FORMAT_BGR_PLANAR,
                                  DATA_TYPE_EXT_1N_BYTE,
                                  &image);
            if (BM_SUCCESS != ret)
            {
                SPDLOG_ERROR("nv12_frame_to_image failed, bm_image_create() ret: {}.", ret);
                return SAIL_ERR_BMI_INIT;
            }
            ret = bm_image_alloc_dev_mem(image, BMCV_HEAP1_ID);
            if (BM_SUCCESS != ret)
            {
                SPDLOG_ERROR("nv12_frame_to_image failed, alloc_dev_mem() ret: {}.", ret);
                return SAIL_ERR_DEV_MALLOC;
            }
        }

        bmcv_rect_t crop_rect = {0, 0, frame_.get_width(), frame_.get_height()};
        const char *env_csc_YPbPr2RGB = getenv("SAIL_USE_CSC_YPbPr2RGB");
        bool use_csc_YPbPr2RGB = env_csc_YPbPr2RGB != nullptr ? 0==strcmp(env_csc_YPbPr2RGB, "1"): false;
        if (use_csc_YPbPr2RGB){
            spdlog::trace("sail.Decoder: using CSC_YPbPr2RGB_BT601");
            ret = bmcv_image_vpp_csc_matrix_convert(handle.data(), 1, temp_img, &image,
                                                    CSC_YPbPr2RGB_BT601, nullptr, BMCV_INTER_LINEAR, &crop_rect);
            if (SAIL_SUCCESS != ret)
            {
                print_image(temp_img, " src:");
                print_image(image, " dst:");
                return SAIL_ERR_BMI_BMCV;
            }
        }
        ret = bmcv_image_vpp_convert(handle.data(), 1, temp_img, &image, &crop_rect);
        if (ret != 0) {
            SPDLOG_ERROR("bm_image_vpp_convert err={}", ret);
            print_image(temp_img, " src:");
            print_image(image, " dst:");
            return SAIL_ERR_BMI_BMCV;
        }
        bm_image_destroy(temp_img);
        return SAIL_SUCCESS;
    }

    int Decoder::Decoder_CC::read_(Handle &handle, bm_image &image) {
        handle_ = handle.data();
        
        int curr_id = bm_get_devid(handle_);
        if (curr_id != tpu_id_){
            SPDLOG_ERROR("Input Handle error, Decoder TPU:{} vs. Handle TPU:{}",tpu_id_,curr_id);
            return SAIL_ERR_DEV_HANDLE;
        }
        if (is_pic_file_) {
            return decode_jpeg(handle, image);
        }
        if (!opened_){
            return SAIL_ERR_DEC_OPEN;
        }
        //reconnect
        if (errcnt_ >= 20) {
            reset_decode(file_path_, compressed_, tpu_id_);
        }

        int ret = grab(frame_);
        if (!ret) {
            errcnt_++;
            return SAIL_ERR_DEC_READ;
        }
        AVFrame *p_frame = frame_.get();

        if (p_frame->height <= 0 || p_frame->width <= 0) {
            spdlog::error("fatal error: {} {}", p_frame->width, p_frame->height);
            errcnt_++;
            return SAIL_ERR_DEC_READ;
        }

        if (p_frame->format != AV_PIX_FMT_NV12 &&
            p_frame->format != AV_PIX_FMT_YUV420P &&
            p_frame->format != AV_PIX_FMT_YUVJ420P) {
            //Convert to YUV420P
            convert_to_yuv420p();
            p_frame = frame_.get();
        }

        // create bm_image with YUV-nv12 format
        if (p_frame->format == AV_PIX_FMT_NV12) {
            ret = nv12_frame_to_image(handle, image);
            if (SAIL_SUCCESS != ret)
            {
                SPDLOG_ERROR("Decoder read failed, nv12_frame_to_image() ret: {}.", ret);
                return SAIL_ERR_BMI_CVT;
            }
            // SPDLOG_INFO("decode with nv12");
        } else {
            // SPDLOG_INFO("decode with other");
            ret = bm_image_create(handle_, p_frame->height, p_frame->width,
                                  FORMAT_YUV420P,
                                  DATA_TYPE_EXT_1N_BYTE,
                                  &image);
            if (SAIL_SUCCESS != ret)
            {
                SPDLOG_ERROR("Decoder read failed, bm_image_create() ret: {}.", ret);
                return SAIL_ERR_BMI_INIT;
            }

            if (p_frame->data[4] != nullptr) {
                bm_mem_desc_t src_plane[4];
                src_plane[0] = bm_mem_from_device((uint64_t) p_frame->data[6], p_frame->linesize[6]);
                src_plane[1] = bm_mem_from_device((uint64_t) p_frame->data[4], p_frame->linesize[4] * p_frame->height);
                src_plane[2] = bm_mem_from_device((uint64_t) p_frame->data[7], p_frame->linesize[7]);
                src_plane[3] = bm_mem_from_device((uint64_t) p_frame->data[5],
                                                  p_frame->linesize[4] * p_frame->height / 2);
                bm_image_attach(image, src_plane);
            } else {
                void *src_plane[4];
#if (BMCV_VERSION_MAJOR == 2) && !defined(BMCV_VERSION_MINOR)
                ret = bm_image_alloc_dev_mem_heap_mask(image, 2);
#else
                ret = bm_image_alloc_dev_mem_heap_mask(image, 6);
#endif      
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("bm_image_alloc_dev_mem_heap_mask err={}", ret);
                    return SAIL_ERR_DEV_MALLOC;
                }
                // need copy to device memory
                src_plane[0] = p_frame->data[0];
                src_plane[1] = p_frame->data[1];
                src_plane[2] = p_frame->data[2];
                src_plane[3] = p_frame->data[3];

                ret = bm_image_copy_host_to_device(image, src_plane);
                assert(ret == 0);
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("bm_image_copy_host_to_device err={}", ret);
                    return SAIL_ERR_DEV_MCOPY;
                }
            }
        }
        errcnt_ = 0;
        return 0;
    }
    void Decoder::Decoder_CC::reconnect() {
        reset_decode(file_path_, compressed_, tpu_id_);
    }

    void Decoder::Decoder_CC::release() {
        disable_dump();
        if (!opened_) return;
        if (!is_pic_file_) {
            av_packet_unref(&pkt_);
            if (video_dec_ctx_) {
                avcodec_close(video_dec_ctx_);
                avcodec_free_context(&video_dec_ctx_);
            }
            avformat_close_input(&fmt_ctx_);
        }

        if(m1)
        {
            spdlog::trace("Decoder_CC::release() m1 != nullptr, and reset()");
            m1.reset();
        }

        handle_ = nullptr;
        fmt_ctx_ = nullptr;
        video_dec_ctx_ = nullptr;
        video_stream_ = nullptr;
        video_stream_idx_ = -1;
        video_frame_count_ = 0;
        got_frame_ = 0;
        height_ = width_ = 0;
        is_rtsp_ = false;
        is_pic_file_ = false;
        opened_ = false;
        end_of_file_ = false;
    }

    void Decoder::Decoder_CC::reset_decode(const std::string &file_path,
                               bool compressed,
                               int tpu_id) {
        release();
        if (is_pic_file(file_path_)) {
            is_pic_file_ = true;
            return;
        }

        std::cout << "decoder ctor: filepath=" << file_path << std::endl;
        // register all formats and codecs
        av_register_all();
        avdevice_register_all();
        AVDictionary *opts = NULL;
#ifndef IS_SOC_MODE
        av_dict_set_int(&opts, "pcie_board_id", tpu_id_, 0);
#endif
        if (file_path_.compare(0, 5, "rtsp:") == 0) {
            is_rtsp_ = true;
            avformat_network_init();
            // Init the decoders, with reference counting
            av_dict_set(&opts, "refcounted_frames", "1", 0);
            // frame buffer set,same as opencv, ost is 20
            av_dict_set(&opts, "extra_frame_buffer_num",
                        extra_frame_buffer_num_value.c_str(), 0);
            // set tcp
            av_dict_set(&opts, "rtsp_transport", rtsp_transport_value.c_str(),
                        0);
            // set timeout (same as opencv),ost is 10000000
            av_dict_set(&opts, "stimeout", stimeout_value.c_str(), 0);
            av_dict_set(&opts, "timeout", stimeout_value.c_str(), 0);

            // add same as opencv
            av_dict_set(&opts, "rtsp_flags", rtsp_flags_value.c_str(), 0);
            av_dict_set_int(&opts, "buffer_size", buffer_size_value, 0);
            av_dict_set_int(&opts, "max_delay", max_delay_value, 0);

            if(keep_timestap){
                //add timestamp
                av_dict_set(&opts, "keep_rtsp_timestamp", "1", 0);
            }
        }
        if (!is_pic_file_ && compressed_) {
            // set compressed output
            av_dict_set(&opts, "output_format", "101", 0);
        }
        skip_non_idr = "0";
        if (get_decoder_env_string("skip_non_idr", skip_non_idr)) {
            av_dict_set(&opts, "skip_non_idr", skip_non_idr.c_str(), 0);
            SPDLOG_INFO("skip_non_idr: {}", skip_non_idr);
        }
        //av_log_set_level(AV_LOG_TRACE);
        std::cout << "open filepath=" << file_path << std::endl;
        AVInputFormat *input_fmt = nullptr;
        if (string_start_with(file_path, "/dev/video")) {
            input_fmt = av_find_input_format("video4linux2");
            if (input_fmt == nullptr) {
                printf("ERROR:can't find format: video4linux2\n");
            } else {
                printf("find video4linux2 success!\n");
                const char *pixfmt = getenv("SAIL_CAP_PIXFMT");
                if (pixfmt == nullptr) pixfmt = "mjpeg";
                av_dict_set(&opts, "pixel_format", pixfmt, 0);
            }
        }

        // open input file, and allocate format context
        int ret = avformat_open_input(&fmt_ctx_, file_path_.c_str(),
                                      input_fmt, &opts);
        if (ret < 0) {
            SPDLOG_ERROR("Failed to open input file: {}", file_path_);
            throw SailDecoderError("Failed to open input file");
        }
        // retrieve stream information
        ret = avformat_find_stream_info(fmt_ctx_, nullptr);
        if (ret < 0) {
            spdlog::error("Failed to find stream information.");
            throw SailDecoderError("Failed to find stream information");
        }

        ret = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (ret < 0) {
            spdlog::error("Failed to find a video stream.");
            throw SailDecoderError("Failed to find a video stream");
        }
        video_stream_idx_ = ret;
        video_stream_ = fmt_ctx_->streams[video_stream_idx_];

        video_dec_ctx_ = avcodec_alloc_context3(NULL);
		avcodec_parameters_to_context(video_dec_ctx_, video_stream_->codecpar);

        AVCodec *dec = avcodec_find_decoder(video_dec_ctx_->codec_id);
        if (!dec) {
            spdlog::error("Failed to find codec.");
            throw SailDecoderError("Failed to find codec");
        }
        ret = avcodec_open2(video_dec_ctx_, dec, &opts);
        if (ret < 0) {
            spdlog::error("Failed to open codec");
            throw SailDecoderError("Failed to open codec");
        }
        height_ = video_dec_ctx_->height;
        width_ = video_dec_ctx_->width;

        // initialize packet, set data to NULL, let the demuxer fill it
        av_init_packet(&pkt_);
        pkt_.data = nullptr;
        pkt_.size = 0;
        opened_ = true;
        // destroy avdict
        av_dict_free(&opts);

    }

    void Decoder::Decoder_CC::convert_to_yuv420p() {
        AVFrame *src = frame_.get();

        struct SwsContext *convert_ctx = NULL;
        enum AVPixelFormat src_pix_fmt = (enum AVPixelFormat) src->format;
        if (src_pix_fmt == AV_PIX_FMT_YUVJ420P) {
            src_pix_fmt = AV_PIX_FMT_YUV420P;
        }

        AVFrame *dst = av_frame_alloc();
        enum AVPixelFormat dst_pix_fmt = AV_PIX_FMT_YUV420P;

        dst->width = src->width;
        dst->height = src->height;
        dst->format = dst_pix_fmt;

        int ret = av_frame_get_buffer(dst, 16);
        assert(ret == 0);

        convert_ctx = sws_getContext(src->width, src->height, src_pix_fmt, dst->width, dst->height, dst_pix_fmt,
                                     SWS_FAST_BILINEAR, NULL, NULL, NULL);
        assert(convert_ctx != nullptr);

        ret = sws_scale(convert_ctx, src->data, src->linesize, 0, src->height, dst->data, dst->linesize);
        assert(ret >= 0);

        sws_freeContext(convert_ctx);

        frame_.set_frame(dst);
    }

    float Decoder::Decoder_CC::get_fps() const {
        if (video_stream_) {
            return video_stream_->avg_frame_rate.num / (float) video_stream_->avg_frame_rate.den;
        } else
            return -1;
    }

// ref: https://ffmpeg.org/doxygen/trunk/demuxing_8c-source.html
    Decoder::Decoder(
            const std::string &file_path,
            bool compressed,
            int tpu_id)
            : _impl(new Decoder_CC(file_path,compressed,tpu_id)){
    }

    bool Decoder::is_opened() {
        return _impl->opened_;
    }

    std::vector<int> Decoder::get_frame_shape() {
        std::vector<int> shape(4);
        shape[0] = 1;
        shape[1] = 3;
        shape[2] = _impl->height_;
        shape[3] = _impl->width_;
        return std::move(shape);
    }

    Decoder::~Decoder() {
        // SPDLOG_INFO("Start ~Decoder()!");
        delete _impl;
        // SPDLOG_INFO("Start ~Decoder()!");
    }

    int Decoder::decode_jpeg(Handle &handle, BMImage &image) {
        return _impl->decode_jpeg(handle, image.data());
    }

    int Decoder::decode_jpeg(Handle &handle, bm_image &image) {
        return _impl->decode_jpeg(handle, image);
    }

    void Decoder::release()
    {
        return _impl->release();
    }

    int Decoder::reconnect()
    {
        if(is_opened()){
            _impl->reconnect();
            return 0;
        }else{
            SPDLOG_INFO("Decoder not opened!");
            return 1;
        }
    }

    int Decoder::read(Handle &handle, BMImage &image) {
        bm_image img;
        spdlog::trace("before Decoder::read(Handle&, BMImage&)");
        int ret = read_(handle, img);
        spdlog::trace("after  Decoder::read(Handle&, BMImage&)");
        if( ret != 0 ) {
            SPDLOG_INFO("Decoder read end or err={}", ret);
            return ret;
        }
        BMImage temp_img;
        temp_img = std::move(img);
        if(!image.empty_check()){
            spdlog::trace("!image.empty_check()");
            image = std::move(temp_img);
        }else if(image.width() == temp_img.width() && image.height() == temp_img.height() && image.format() == temp_img.format()){
            spdlog::trace("image == temp_img");
            image = std::move(temp_img);
        }else{
            spdlog::trace("bmcv_temp");
            Bmcv bmcv_temp(handle);
            ret = bmcv_temp.vpp_resize(temp_img, image, image.width(), image.height());
        }
#ifdef USE_OPENCV
        if(_impl->m1){
            spdlog::trace("_impl->m1 != nullptr");
            spdlog::debug("_impl->m1->u->refcount, {}", _impl->m1->u->refcount);
            std::shared_ptr<cv::Mat> cached_mat = std::make_shared<cv::Mat>(*(_impl->m1));
            image.cache_ost_mat(cached_mat);
        }
#endif
        return ret;
    }

    BMImage Decoder::read(Handle &handle) {
        BMImage image;
        read(handle, image);
        return std::move(image);
    }

    int Decoder::read_(Handle &handle, bm_image &image) {
        return _impl->read_(handle, image);
    }

    bm_image Decoder::read_(Handle &handle) {
        bm_image image;
        read_(handle, image);
        return image;
    }

    float Decoder::get_fps() const {
        return _impl->get_fps();
    }

    vector<double> Decoder::get_pts_dts() {
        return _impl->get_pts_dts();
    }

    void Decoder::enable_dump(int dump_max_seconds)
    {
        return _impl->enable_dump(dump_max_seconds);
    }

    void Decoder::disable_dump()
    {
        return _impl->disable_dump();
    }

    int Decoder::dump(int dump_pre_seconds, int dump_post_seconds, std::string& file_path)
    {
        return _impl->dump(dump_pre_seconds, dump_post_seconds, file_path);
    }

    class Decoder_RawStream::Decoder_RawStream_CC{
    public:
        explicit Decoder_RawStream_CC(int tpu_id,
            string  decformt);
        ~Decoder_RawStream_CC();

        int read_(uint8_t* data, int data_size, bm_image &image,bool continueFrame);
        int read(uint8_t* data, int data_size, sail::BMImage &image,bool continueFrame);
        void release();

        #ifdef PYTHON
            int read_(pybind11::bytes data_bytes, bm_image& image, bool continueFrame);
            int read(pybind11::bytes data_bytes, BMImage& image, bool continueFrame);
        #endif

        typedef struct {
        uint8_t* start;
        int      size;
        int      pos;
        } bs_buffer_t;

        //控制流向
        static int read_buffer(void *opaque, uint8_t *buf, int buf_size)
        {
            bs_buffer_t* bs = (bs_buffer_t*)opaque;

            int r = bs->size - bs->pos;
            if (r <= 0) {
                SPDLOG_INFO("EOF of AVIO.");
                return AVERROR_EOF;
            }

            uint8_t* p = bs->start + bs->pos;
            int len = (r >= buf_size) ? buf_size : r;
            memcpy(buf, p, len);
            //cout << "read " << len << endl;
            bs->pos += len;
            return len;
        }

    private:
    
    AVFormatContext *pFormatCtx = nullptr;
    AVCodecContext  *dec_ctx = nullptr;
    AVFrame         *pFrame = nullptr;
    AVPacket        *pkt;
    AVDictionary    *dict = nullptr;
    AVIOContext     *avio_ctx = nullptr;
    AVCodec         *pCodec = nullptr;
    AVInputFormat   *iformat = nullptr;

    int got_picture;
    int ret;
    sail::Handle handle;

    uint8_t* bs_buffer = nullptr;

    bs_buffer_t bs_obj = {0, 0, 0};

    int      aviobuf_size = 32*1024; // 32K
    uint8_t *aviobuffer = nullptr;

    // Select the correct decoder based on the input format
    const char* iformat_name;
    const char* decoder_input;    

    };

    Decoder_RawStream::Decoder_RawStream(
            int tpu_id,
            string  decformt)
            : _impl(new Decoder_RawStream_CC(tpu_id,decformt)){
    }

    Decoder_RawStream::~Decoder_RawStream() {
        delete _impl;
    }

    Decoder_RawStream::Decoder_RawStream_CC::~Decoder_RawStream_CC() {
        release();
    }

    void Decoder_RawStream::Decoder_RawStream_CC::release() {
        if (pFrame) {
            av_frame_free(&pFrame);
            pFrame = nullptr; 
        }
        
        if (pkt) {
            av_packet_free(&pkt); // 释放 AVPacket
            pkt = nullptr;
        }

        if (pFormatCtx) {
            avformat_close_input(&pFormatCtx); // 关闭输入流，如果 pFormatCtx 非空
            avformat_free_context(pFormatCtx); // 释放格式上下文
            pFormatCtx = nullptr; // 确保指针设置为 nullptr 避免野指针
        }

        if (dec_ctx) {
            avcodec_close(dec_ctx);
            dec_ctx = nullptr; 
        }
        if (dict) {
            av_dict_free(&dict);
            dict = nullptr; 
        }

        if (avio_ctx) {
            av_freep(&avio_ctx->buffer); // 释放 AVIO 上下文的缓冲区
            av_freep(&avio_ctx); // 释放 AVIO 上下文本身
            avio_ctx = nullptr;
        }
    }

    void Decoder_RawStream::release() {
        return _impl->release();
    }
    int Decoder_RawStream::read_(uint8_t* data, int data_size, bm_image &image, bool continueFrame){
        return _impl->read_(data, data_size, image, continueFrame);
    }

    int Decoder_RawStream::read(uint8_t* data, int data_size, sail::BMImage &image,bool continueFrame){
        return _impl->read(data, data_size, image, continueFrame);
    }

    Decoder_RawStream::Decoder_RawStream_CC::Decoder_RawStream_CC(int tpu_id,string  decformt){
        
        handle=sail::Handle(tpu_id);
        if (decformt == "h264") {
            // If the input format is h264, set the format name and decoder input accordingly
            iformat_name = "h264";
            decoder_input = "h264_bm";
        } else if (decformt == "h265") {
            // If the input format is h265, set the format name and decoder input accordingly
            iformat_name = "hevc";
            decoder_input = "hevc_bm";
        } else {
            // If the input format is neither h264 nor h265, throw an exception
            throw std::invalid_argument("Invalid decoder_input");
        }

        /* h264/h265 */
        iformat = av_find_input_format(iformat_name);
        if (iformat == NULL) {
            throw std::runtime_error("Failed to find input format."); 
        }

        pCodec = avcodec_find_decoder_by_name(decoder_input);
        if (pCodec == NULL) {
            throw std::runtime_error("Codec not found."); 
        }

        dec_ctx = avcodec_alloc_context3(pCodec);
        if (dec_ctx == NULL) {
            throw std::bad_alloc(); 
        }

        ret = avcodec_open2(dec_ctx, pCodec, &dict);
        if (ret < 0) {
            throw std::runtime_error("Could not open codec.");
        }

        pFormatCtx = avformat_alloc_context();
        if (pFormatCtx == nullptr) {
            throw std::bad_alloc(); 
        }

        pFrame = av_frame_alloc();
        if (pFrame == nullptr) {
            throw std::bad_alloc(); 
        }

        pkt = av_packet_alloc();
        if (pkt == nullptr) {
            throw std::bad_alloc(); 
        }

    }


     // Function to read from the decoder
    int  Decoder_RawStream::Decoder_RawStream_CC::read_(uint8_t* data, int data_size, bm_image &image, bool continueFrame) {

        bs_buffer = data;
        if (bs_buffer == nullptr) {
            throw std::runtime_error("Invalid h264 data");
        }
        
        bs_obj.start = bs_buffer;
        bs_obj.size  = data_size;

        //读取一帧
        if (continueFrame==false)
        {
            bs_obj.pos =0;
        }
            
        if (bs_obj.pos == 0) {
            // 每次从头开始时关闭并重新打开流
            if (avio_ctx) {
                avformat_close_input(&pFormatCtx); // 关闭输入流，同时会释放 pFormatCtx->pb
                av_freep(&avio_ctx->buffer); // 释放 AVIO 上下文的缓冲区
                av_freep(&avio_ctx); // 释放 AVIO 上下文本身
                avio_ctx = nullptr;
                avformat_free_context(pFormatCtx);
            }

            // 创建新的 AVIO 上下文
            avio_ctx = avio_alloc_context(aviobuffer, aviobuf_size, 0, (void*)(&bs_obj), read_buffer, NULL, NULL);
            if (!avio_ctx) {
                av_free(aviobuffer);
                throw std::runtime_error("avio_alloc_context failed");
            }

            // 确保 AVFormatContext 已正确分配
            if (!pFormatCtx) {
                pFormatCtx = avformat_alloc_context();
                if (!pFormatCtx) {
                    throw std::runtime_error("Failed to allocate AVFormatContext");
                }
            }
            pFormatCtx->pb = avio_ctx;

            // 打开输入流
            if (avformat_open_input(&pFormatCtx, NULL, iformat, NULL) < 0) {
                throw std::runtime_error("Couldn't open input stream.");
            }
        }
        
        av_packet_unref(pkt);
        if (av_read_frame(pFormatCtx, pkt) >= 0) {
            while (true)
            {
                avcodec_decode_video2(dec_ctx, pFrame, &got_picture, pkt);
                if(got_picture==0){
                    continue;   
                }else if(got_picture==1){
                    ret=avframe_to_bm_image(handle,*pFrame, image);
                    if (BM_SUCCESS!=ret)
                    {   
                        SPDLOG_ERROR("avframe_to_bm_image err={}", ret);
                        return ret;
                    }
                    return 0;
                }else{
                    SPDLOG_ERROR("Decode Error");
                    return ret;
                }
            }  
        }
        SPDLOG_INFO("Decode End.");
        return -1; // Or return other values as needed
    }

    // Function to read from the decoder
    int  Decoder_RawStream::Decoder_RawStream_CC::read(uint8_t* data, int data_size, sail::BMImage &image,bool continueFrame) {

        bs_buffer = data;
        if (bs_buffer == nullptr) {
            throw std::invalid_argument("Invalid h264 data");
        }
        
        bs_obj.start = bs_buffer;
        bs_obj.size  = data_size;

        //读取一帧
        if (continueFrame==false)
        {
            bs_obj.pos =0;
        }
            
        if (bs_obj.pos == 0) {
            // 每次从头开始时关闭并重新打开流
            if (avio_ctx) {
                avformat_close_input(&pFormatCtx); // 关闭输入流，同时会释放 pFormatCtx->pb
                av_freep(&avio_ctx->buffer); // 释放 AVIO 上下文的缓冲区
                av_freep(&avio_ctx); // 释放 AVIO 上下文本身
                avio_ctx = nullptr;
                avformat_free_context(pFormatCtx);
            }

            // 创建新的 AVIO 上下文
            avio_ctx = avio_alloc_context(aviobuffer, aviobuf_size, 0, (void*)(&bs_obj), read_buffer, NULL, NULL);
            if (!avio_ctx) {
                av_free(aviobuffer);
                throw std::runtime_error("avio_alloc_context failed");
            }

            // 确保 AVFormatContext 已正确分配
            if (!pFormatCtx) {
                pFormatCtx = avformat_alloc_context();
                if (!pFormatCtx) {
                    throw std::runtime_error("Failed to allocate AVFormatContext");
                }
            }
            pFormatCtx->pb = avio_ctx;

            // 打开输入流
            if (avformat_open_input(&pFormatCtx, NULL, iformat, NULL) < 0) {
                throw std::runtime_error("Couldn't open input stream.");
            }
        }
        av_packet_unref(pkt);
        if (av_read_frame(pFormatCtx, pkt) >= 0) {
            while (true)
            {
                avcodec_decode_video2(dec_ctx, pFrame, &got_picture, pkt);
                if(got_picture==0){
                    continue;   
                }else if(got_picture==1){
                    ret=avframe_to_bm_image(handle,*pFrame, image.data());
                    if (BM_SUCCESS!=ret)
                    {
                        SPDLOG_ERROR("avframe_to_bm_image err={}", ret);
                        return ret;
                    }
                    return 0;
                }else{
                    SPDLOG_ERROR("Decode Error");
                    return ret;
                }
            }
        }
        SPDLOG_INFO("Decode End.");
        return -1; // Or return other values as needed
    }


#endif //USE_FFMPEG

#ifdef USE_BMCV

    class BMImage::BMImage_CC{
    public:
        BMImage_CC();
        BMImage_CC(bm_image &img);
          
        BMImage_CC(
            Handle& handle,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext dtype);

        BMImage_CC(
            Handle& handle,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext dtype,
            int *stride);

        BMImage_CC(
            Handle &handle,
            void *buffer,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext dtype,
            std::vector<int> strides,
            size_t offset,
            int size);

        ~BMImage_CC(){
            if(mat_buffer){
                // delete mat_buffer;
                // mat_buffer = NULL;
            }
        };

#if defined(USE_BMCV) && defined(USE_OPENCV) && defined(PYTHON)
    pybind11::array_t<uint8_t> cvmat_to_numpy(cv::Mat cv_mat){
        std::vector<pybind11::ssize_t> shape;

        pybind11::ssize_t item_size = 1;
        std::string format;
        if (cv_mat.type()  == CV_8UC3) {
            shape.push_back(cv_mat.rows);
            shape.push_back(cv_mat.cols);
            shape.push_back(3);
            item_size = sizeof(uint8_t);
            format = pybind11::format_descriptor<uint8_t>::format();
        } else if(cv_mat.type()  == CV_8UC1){
            shape.push_back(cv_mat.rows);
            shape.push_back(cv_mat.cols);
            item_size = sizeof(uint8_t);
            format = pybind11::format_descriptor<uint8_t>::format();
        }else{
            SPDLOG_ERROR("Mat type not support: {}",cv_mat.type());
            throw SailBMImageError("not supported");
        }
        
        int stride_temp = FFALIGN(cv_mat.cols * 3 * item_size, SAIL_ALIGN); // ceiling to 64 * N

        pybind11::ssize_t ndim = shape.size();
        std::vector<pybind11::ssize_t> stride;
        for (size_t i = 1; i < shape.size(); i++) {
            stride.push_back(cv_mat.step[i-1]);
        }
        stride.push_back(item_size);
        pybind11::buffer_info output_buf(cv_mat.data, item_size, format,
                                            ndim, shape, stride);
        return std::move(pybind11::array_t<uint8_t>(output_buf));
    }
    pybind11::array_t<uint8_t> asmat()
    {
        if (!img_.image_private){
            SPDLOG_ERROR("BMImage is empty!");
            throw SailBMImageError("BMImage asmat() failed!");
        }
        cv::Mat o_mat;
        int ret = cv::bmcv::toMAT((bm_image *) &img_, o_mat, true);
        if(ret != BM_SUCCESS){
            SPDLOG_ERROR("cv::bmcv::toMAT Failed: {}",ret);
            throw SailBMImageError("BMImage asmat() failed!");
        }
        return std::move(cvmat_to_numpy(o_mat));
    }
#endif
    protected:
    /// inner bm_image
        void reset(int w, int h);
        bm_image img_;
        std::shared_ptr<bm_handle_t> handle_shaptr_;
        void cache_mat(std::shared_ptr<cv::Mat>& ost_img){
            spdlog::debug("BMImage_CC::cache_mat() before =, ost_img.use_count(): {}", ost_img.use_count());
            spdlog::debug("BMImage_CC::cache_mat() before =, ost_img->u->refcount: {}", ost_img->u->refcount);
            spdlog::debug("BMImage_CC::cache_mat() before =, mat_buffer.use_count(): {}", mat_buffer.use_count());
            mat_buffer = ost_img;
            spdlog::debug("BMImage_CC::cache_mat() after  =, ost_img.use_count(): {}", ost_img.use_count());
            spdlog::debug("BMImage_CC::cache_mat() after  =, ost_img->u->refcount: {}", ost_img->u->refcount);
            spdlog::debug("BMImage_CC::cache_mat() after  =, mat_buffer.use_count(): {}", mat_buffer.use_count());
            spdlog::debug("BMImage_CC::cache_mat() after  =, mat_buffer->u->refcount: {}", mat_buffer->u->refcount);
        }

    private:
        void create(
            Handle&                  handle,
            int                      h,
            int                      w,
            bm_image_format_ext      format,
            bm_image_data_format_ext dtype,
            int                      *stride=NULL);
        void destroy();
        void allocate();
        void detach();
        bool is_created() const;
        int align();
        int check_align() const;
        int unalign();
        int check_contiguous_memory() const;
#ifdef PYTHON
        pybind11::array asnumpy() const;
#endif
        void set_ipc_flag(bool f);
        bool need_to_free_;

        std::shared_ptr<cv::Mat> mat_buffer = nullptr;
        bool ipc_recv_flag = false;
        friend class BMImage;
        friend class Bmcv;
        friend class Decoder;
        Handle handle_;
    };

    BMImage::BMImage_CC::BMImage_CC():img_({}), need_to_free_(false) {
        img_.image_format = FORMAT_BGR_PLANAR;
        img_.data_type = DATA_TYPE_EXT_1N_BYTE;
        img_.width = 0;
        img_.height = 0;
    }

    BMImage::BMImage_CC::BMImage_CC(bm_image &img) : img_(img), need_to_free_(false) {
        bm_handle_t key_handle=bm_image_get_handle(&img_);
        std::lock_guard<std::mutex> lock(Handle::map_mutex);
        if(Handle::handle_map.find(key_handle) != Handle::handle_map.end())
        {
            Handle* h = Handle::handle_map[key_handle];
            if(h != nullptr) handle_shaptr_ = h->shaptr();
            // SPDLOG_INFO("Found sail::Handle!");
        }
        else{
            // SPDLOG_INFO("Not find sail::Handle!");
        }
    }

    BMImage::BMImage_CC::BMImage_CC(
            Handle &handle,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext dtype
    ) : img_({}), need_to_free_(false) {
        create(handle, h, w, format, dtype);
        allocate();
    }

    BMImage::BMImage_CC::BMImage_CC(
            Handle &handle,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext dtype,
            int *stride
    ) : img_({}), need_to_free_(false) {
        create(handle, h, w, format, dtype, stride);
        allocate();
    }

    BMImage::BMImage_CC::BMImage_CC(
        Handle &handle,
        void *buffer,
        int h,
        int w,
        bm_image_format_ext format,
        bm_image_data_format_ext dtype,
        std::vector<int> strides,
        size_t offset,
        int size) : img_({}), need_to_free_(false)
    {
        // check input params
        if (buffer == nullptr)
        {
            throw SailBMImageError("buf is invalid!");
        }
        if (h <= 0 || w <= 0)
        {
            throw SailBMImageError("height or width is invalid!");
        }

        int dtype_size = 0;
        switch (dtype)
        {
        case DATA_TYPE_EXT_FLOAT32:
#if !((BMCV_VERSION_MAJOR == 2) && !defined(BMCV_VERSION_MINOR))
        case DATA_TYPE_EXT_4N_BYTE_SIGNED:
        case DATA_TYPE_EXT_4N_BYTE:
#endif
            dtype_size = 4;
            break;
        case DATA_TYPE_EXT_1N_BYTE_SIGNED:
        case DATA_TYPE_EXT_1N_BYTE:
            dtype_size = 1;
            break;
        default:
            throw SailBMImageError("The input dtype is not supported!");
        }
        spdlog::debug("BMImage_CC dtype {}, dtype_size = {}", dtype, dtype_size);

        // create bm_image and alloc dev mem
        spdlog::debug("BMImage_CC strides.empty()? = {}", strides.empty());
        int *pstrides = strides.empty() ? nullptr : strides.data();
        create(handle, h, w, format, dtype, pstrides);
        allocate();

        // s2d
        int ret = 0;
        int plane_num = bm_image_get_plane_num(img_);
        spdlog::debug("BMImage_CC format {}, plane_num = {}", format, plane_num);
        std::vector<int> true_strides(plane_num);
        ret = bm_image_get_stride(img_, true_strides.data());
        if (BM_SUCCESS != ret)
        {
            throw SailBMImageError("BMImage init, bm_image_get_stride fail!");
        }
        spdlog::debug("BMImage_CC format {}, true strides[0] = {}",
                      format, true_strides[0]);

        std::vector<int> bytesizes(plane_num);
        ret = bm_image_get_byte_size(img_, bytesizes.data());
        if (BM_SUCCESS != ret)
        {
            SPDLOG_ERROR("bm_image_get_byte_size failed");
            throw SailBMImageError("BMImage init failed!");
        }
        std::vector<void *> s2d_buffers(plane_num);
        spdlog::debug("BMImage_CC buffer = {}", buffer);
        s2d_buffers[0] = buffer + offset;
        for (int i = 1; i < plane_num; i++)
        {
            spdlog::debug("plane {}, bytesizes = {}", i, bytesizes[i - 1]);
            char *buffer_i = static_cast<char *>(s2d_buffers[i - 1]) +
                             bytesizes[i - 1];
            s2d_buffers[i] = static_cast<void *>(buffer_i);
        }
        spdlog::trace("BMImage_CC before s2d");
        ret = bm_image_copy_host_to_device(img_, s2d_buffers.data());
        spdlog::trace("BMImage_CC after s2d");
        if (BM_SUCCESS != ret)
        {
            throw SailBMImageError("BMImage init, s2d failed!");
        }
    }

    void BMImage::BMImage_CC::create(
            Handle &handle,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext data_type,
            int *stride
    ) {
// #ifdef IS_SOC_MODE
//         int data_size = 1;
//         int default_stride[3] = {0};
//         if(NULL==stride)
//         {
//             switch(data_type){
//                 case DATA_TYPE_EXT_1N_BYTE:
//                 case DATA_TYPE_EXT_1N_BYTE_SIGNED:
//                     data_size = 1;
//                     break;
//                 case DATA_TYPE_EXT_FLOAT32:
//                 case DATA_TYPE_EXT_4N_BYTE:
//                 case DATA_TYPE_EXT_4N_BYTE_SIGNED:
//                     data_size = 4;
//                     break;
//                 case DATA_TYPE_EXT_FP16:
//                 case DATA_TYPE_EXT_BF16:
//                     data_size = 2;
//                     break;
//                 default:
//                     data_size = 1;
//                     break;
//             }
//             switch (format) {
//                 case FORMAT_YUV420P: {
//                     default_stride[0] = FFALIGN(w * data_size, SAIL_ALIGN);
//                     default_stride[1] = FFALIGN((FFALIGN(w, 2) >> 1) * data_size, SAIL_ALIGN);
//                     default_stride[2] = default_stride[1];
//                     break;
//                 }
//                 case FORMAT_YUV422P: {
//                     default_stride[0] = FFALIGN(w * data_size, SAIL_ALIGN);
//                     default_stride[1] = FFALIGN((FFALIGN(w, 2) >> 1) * data_size, SAIL_ALIGN);
//                     default_stride[2] = default_stride[1];
//                     break;
//                 }
//                 case FORMAT_YUV444P: {
//                     default_stride[0] = FFALIGN(w * data_size, SAIL_ALIGN);
//                     default_stride[1] = FFALIGN(w * data_size, SAIL_ALIGN);
//                     default_stride[2] = default_stride[1];
//                     break;
//                 }
//                 case FORMAT_NV12:
//                 case FORMAT_NV21: {
//                     default_stride[0] = FFALIGN(w * data_size, SAIL_ALIGN);
//                     default_stride[1] = FFALIGN(FFALIGN(w, 2) * data_size, SAIL_ALIGN);
//                     break;
//                 }
//                 case FORMAT_NV16:
//                 case FORMAT_NV61: {
//                     default_stride[0] = FFALIGN(w * data_size, SAIL_ALIGN);
//                     default_stride[1] = FFALIGN(FFALIGN(w, 2) * data_size, SAIL_ALIGN);
//                     break;
//                 }
//                 case FORMAT_GRAY: {
//                     default_stride[0] = FFALIGN(w * data_size, SAIL_ALIGN);
//                     break;
//                 }
//                 // case FORMAT_COMPRESSED: {
//                 //     image_private->plane_num = 4;
//                 //     break;
//                 // }
//                 case FORMAT_BGR_PACKED:
//                 case FORMAT_RGB_PACKED: {
//                     default_stride[0] = FFALIGN(w * 3 * data_size, SAIL_ALIGN);
//                     break;
//                 }
//                 case FORMAT_BGR_PLANAR:
//                 case FORMAT_RGB_PLANAR: {
//                     default_stride[0] = FFALIGN(w * data_size, SAIL_ALIGN);
//                     break;
//                 }
//                 case FORMAT_BGRP_SEPARATE:
//                 case FORMAT_RGBP_SEPARATE: {
//                     default_stride[0] = FFALIGN(w * data_size, SAIL_ALIGN);
//                     default_stride[1] = FFALIGN(w * data_size, SAIL_ALIGN);
//                     default_stride[2] = FFALIGN(w * data_size, SAIL_ALIGN);
//                     break;
//                 }
//             }
//             stride = default_stride;
//         }
// #endif
        destroy();
        bm_status_t ret = bm_image_create(handle.data(), h, w, format, data_type, &img_, stride);
        if(ret != BM_SUCCESS){
            SPDLOG_ERROR("bm_image_create failed, bm_image_create() ret {}", ret);
            throw SailBMImageError("BMImage create() failed!");
        }
        handle_shaptr_ = handle.shaptr();
        handle_ = handle;
    }

    void BMImage::BMImage_CC::destroy() {
        if (need_to_free_) {
            // static int destroy_image_0_count = 0;
            // destroy_image_0_count+= 1;

            // SPDLOG_INFO("BMImage: bm_image_free_contiguous_mem...{}",destroy_image_0_count);
            int ret = bm_image_free_contiguous_mem(1, &img_);
            if(ret !=0){
                SPDLOG_ERROR("bm_image_free_contiguous_mem failed!");
                throw SailBMImageError("bmcv api fail");
            }
            need_to_free_ = false;
        }
        if(ipc_recv_flag) {
            bm_handle_t handle_ = bm_image_get_handle(&img_);
            bm_device_mem_t mem[3];
            int ret = bm_image_get_device_mem(img_, mem);
            if(ret !=0){
                SPDLOG_ERROR("bm_image_get_device_mem failed!");
                throw SailBMImageError("bmcv api fail");
            }
            int planes = bm_image_get_plane_num(img_);
            for (int i = 0; i < planes; ++i) {
                bm_free_device(handle_, mem[i]);
            }
            delete img_.image_private;
            img_.image_private = nullptr;
            return;
        }
        detach();
        if(mat_buffer) {
            spdlog::trace("BMImage_CC::destroy() mat_buffer != nullptr, and reset()");
            spdlog::debug("BMImage_CC::destroy() before reset, mat_buffer use_count: {}", mat_buffer.use_count());
            spdlog::debug("BMImage_CC::destroy() before reset, mat_buffer refcount:  {}", mat_buffer->u->refcount);
            mat_buffer.reset();
        }
        if(is_created()){
            // SPDLOG_INFO("BMImage: bm_image_destroy...");
            bm_image_destroy(img_);
            img_.image_private = nullptr;
        }
    }
    void BMImage::BMImage_CC::detach()
    {
        if(is_created()){
            if(bm_image_is_attached(img_)){
                bm_image_detach(img_);
            }
        }
    }

    void BMImage::BMImage_CC::allocate() {
#if (BMCV_VERSION_MAJOR == 2) && !defined(BMCV_VERSION_MINOR)
        int ret = bm_image_alloc_contiguous_mem_heap_mask(1, &img_, 2);
#else
        int ret = bm_image_alloc_contiguous_mem_heap_mask(1, &img_, 6);
#endif
        if(ret != BM_SUCCESS){
            SPDLOG_ERROR("allocate failed, bm_image_alloc_contiguous_mem_heap_mask() ret {}", ret);
            throw SailBMImageError("BMImage destroy() failed!");
        }
        // bm_image_alloc_dev_mem_heap_mask(img_, 6);

        // bm_device_mem_t dev_mem[3];
        // bm_image_get_device_mem(img_, dev_mem);
        // printf("###: %s:%s:%d, Alloc %p\n",__FILE__,__FUNCTION__,__LINE__,bm_mem_get_device_addr(*dev_mem));

        need_to_free_ = true;
    }

    bool BMImage::BMImage_CC::is_created() const {
        return img_.image_private != nullptr;
    }

    void BMImage::BMImage_CC::reset(int w, int h)
    {
        if (img_.width != w || img_.height != h)
        {
            bm_handle_t bmHandle=bm_image_get_handle(&img_);
            destroy();
            if (bmHandle != nullptr) {
                bm_status_t ret = bm_image_create(bmHandle, h, w, img_.image_format, img_.data_type, &img_);
                if(ret != BM_SUCCESS){
                    SPDLOG_ERROR("allocate failed, bm_image_alloc_contiguous_mem_heap_mask() ret {}", ret);
                    throw SailBMImageError("BMImage reset() failed!");
                }
                std::lock_guard<std::mutex> lock(Handle::map_mutex);
                if(Handle::handle_map.find(bmHandle)!=Handle::handle_map.end()){
                    handle_shaptr_ = Handle::handle_map[bmHandle]->shaptr();
                    // SPDLOG_INFO("Found sail::Handle!");
                }
                else{
                    // SPDLOG_INFO("Not find sail::Handle!");
                } 
            }
        }

    }

    int BMImage::BMImage_CC::align() {
        int ret = check_align();
            if (ret) {
            return ret;
        }
        // if (!need_to_free_) {
            //     SPDLOG_ERROR("bm_image is not attach memory,can't align!");
            //     exit(1);
        // }
        int data_size = 1;
        switch (img_.data_type) {
            case DATA_TYPE_EXT_FLOAT32:
            data_size = 4;
            break;
#if !((BMCV_VERSION_MAJOR == 2) && !defined(BMCV_VERSION_MINOR))
            case DATA_TYPE_EXT_4N_BYTE:
            case DATA_TYPE_EXT_4N_BYTE_SIGNED:
            data_size = 4;
            break;
#endif
            default:
            data_size = 1;
            break;
        }
        int default_stride[4] = {0};
        int w = img_.width;
        int h = img_.height;
        {
            switch (img_.image_format) {
                case FORMAT_YUV420P: {
                    default_stride[0] = FFALIGN(w * data_size, 64);
                    default_stride[1] = FFALIGN((FFALIGN(w, 2) >> 1) * data_size, 64);
                    default_stride[2] = default_stride[1];
                    break;
                }
                case FORMAT_YUV422P: {
                    default_stride[0] = FFALIGN(w * data_size, 64);
                    default_stride[1] = FFALIGN((FFALIGN(w, 2) >> 1) * data_size, 64);
                    default_stride[2] = default_stride[1];
                    break;
                }
                case FORMAT_YUV444P: {
                    default_stride[0] = FFALIGN(w * data_size, 64);
                    default_stride[1] = FFALIGN(w * data_size, 64);
                    default_stride[2] = default_stride[1];
                    break;
                }
                case FORMAT_NV12:
                case FORMAT_NV21: {
                    default_stride[0] = FFALIGN(w * data_size, 64);
                    default_stride[1] = FFALIGN(FFALIGN(w, 2) * data_size, 64);
                    break;
                }
                case FORMAT_NV16:
                case FORMAT_NV61: {
                    default_stride[0] = FFALIGN(w * data_size, 64);
                    default_stride[1] = FFALIGN(FFALIGN(w, 2) * data_size, 64);
                    break;
                }
                case FORMAT_GRAY: {
                    default_stride[0] = FFALIGN(w * data_size, 64);
                    break;
                }
                // case FORMAT_COMPRESSED: {
                //     image_private->plane_num = 4;
                //     break;
                // }
                case FORMAT_BGR_PACKED:
                case FORMAT_RGB_PACKED: {
                    default_stride[0] = FFALIGN(w * 3 * data_size, 64);
                    break;
                }
                case FORMAT_BGR_PLANAR:
                case FORMAT_RGB_PLANAR: {
                    default_stride[0] = FFALIGN(w * data_size, 64);
                    break;
                }
                case FORMAT_BGRP_SEPARATE:
                case FORMAT_RGBP_SEPARATE: {
                    default_stride[0] = FFALIGN(w * data_size, 64);
                    default_stride[1] = FFALIGN(w * data_size, 64);
                    default_stride[2] = FFALIGN(w * data_size, 64);
                    break;
                }
            }
        }

        {
            bm_handle_t bmHandle = bm_image_get_handle(&img_);
            int ret=-1;
            if (bmHandle != nullptr) {
                bm_image temp_img;
                ret=bm_image_create(bmHandle, h, w, img_.image_format, img_.data_type,
                                &temp_img, default_stride);
                if (ret != BM_SUCCESS){
                    SPDLOG_ERROR("bm_image_create err={}", ret);
                    return ret;
                }
#if (BMCV_VERSION_MAJOR == 2) && !defined(BMCV_VERSION_MINOR)
                ret = bm_image_alloc_contiguous_mem_heap_mask(1, &temp_img, 2);
#else
                ret = bm_image_alloc_contiguous_mem_heap_mask(1, &temp_img, 6);
#endif
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("align failed, bm_image_alloc_contiguous_mem_heap_mask() ret {}", ret);
                    return ret;
                }
                ret = bmcv_width_align(bmHandle, img_, temp_img);
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("align failed, bm_image_alloc_contiguous_mem_heap_mask() ret {}", ret);
                    return ret;
                }
                destroy();
                img_ = temp_img;
                need_to_free_=true;
            }else{
                SPDLOG_INFO("BMImage is empty, Not in any device!");
            }
            return ret;
        }
    }
   int BMImage::BMImage_CC::check_align() const {
        if (!is_created()) {
            SPDLOG_ERROR("bm_image is not created,can't align!");
            return -1;
        }
        // if (!need_to_free_) {
            //     SPDLOG_ERROR("bm_image is not attach memory,can't align!");
            //     exit(1);
        // }
        bool if_aligned = true;
        int w = img_.width;
        int h = img_.height;

        int stride[8];
        bm_image_get_stride(img_, stride);
        #define CHECK_2_BASE(x, a) ((x & (a - 1)) == 0)
        {
            switch (img_.image_format) {
                case FORMAT_YUV420P: {
                    if_aligned &= CHECK_2_BASE(stride[0], 64);
                    if_aligned &= CHECK_2_BASE(stride[1], 64);
                    if_aligned &= CHECK_2_BASE(stride[2], 64);

                    break;
                }
                case FORMAT_YUV422P: {
                    if_aligned &= CHECK_2_BASE(stride[0], 64);
                    if_aligned &= CHECK_2_BASE(stride[1], 64);
                    if_aligned &= CHECK_2_BASE(stride[2], 64);
                    break;
                }
                case FORMAT_YUV444P: {
                    if_aligned &= CHECK_2_BASE(stride[0], 64);
                    if_aligned &= CHECK_2_BASE(stride[1], 64);
                    if_aligned &= CHECK_2_BASE(stride[2], 64);
                    break;
                }
                case FORMAT_NV12:
                case FORMAT_NV21: {
                    if_aligned &= CHECK_2_BASE(stride[0], 64);
                    if_aligned &= CHECK_2_BASE(stride[1], 64);
                    break;
                }
                case FORMAT_NV16:
                case FORMAT_NV61: {
                    if_aligned &= CHECK_2_BASE(stride[0], 64);
                    if_aligned &= CHECK_2_BASE(stride[1], 64);
                    break;
                }
                case FORMAT_GRAY: {
                    if_aligned &= CHECK_2_BASE(stride[0], 64);
                    break;
                }
                // case FORMAT_COMPRESSED: {
                //     image_private->plane_num = 4;
                //     break;
                // }
                case FORMAT_BGR_PACKED:
                case FORMAT_RGB_PACKED: {
                    if_aligned &= CHECK_2_BASE(stride[0], 64);
                    break;
                }
                case FORMAT_BGR_PLANAR:
                case FORMAT_RGB_PLANAR: {
                    if_aligned &= CHECK_2_BASE(stride[0], 64);
                    break;
                }
                case FORMAT_BGRP_SEPARATE:
                case FORMAT_RGBP_SEPARATE: {
                    if_aligned &= CHECK_2_BASE(stride[0], 64);
                    if_aligned &= CHECK_2_BASE(stride[1], 64);
                    if_aligned &= CHECK_2_BASE(stride[2], 64);
                    break;
                }
            }
        }

        return if_aligned;
    }
    int BMImage::BMImage_CC::unalign() {
        int ret = check_align();
            if (ret != 1) {
            return ret;
        }
        // if (!need_to_free_) {
            //     SPDLOG_ERROR("bm_image is not attach memory,can't align!");
            //     exit(1);
        // }
        int data_size = 1;
        switch (img_.data_type) {
            case DATA_TYPE_EXT_FLOAT32:
            data_size = 4;
            break;
#if !((BMCV_VERSION_MAJOR == 2) && !defined(BMCV_VERSION_MINOR))
            case DATA_TYPE_EXT_4N_BYTE:
            case DATA_TYPE_EXT_4N_BYTE_SIGNED:
            data_size = 4;
            break;
#endif
            default:
            data_size = 1;
            break;
        }
        int default_stride[4] = {0};
        int w = img_.width;
        int h = img_.height;

        {
            switch (img_.image_format) {
                case FORMAT_YUV420P: {
                    default_stride[0] = w * data_size;
                    default_stride[1] = (FFALIGN(w, 2) >> 1) * data_size;
                    default_stride[2] = default_stride[1];
                    break;
                }
                case FORMAT_YUV422P: {
                    default_stride[0] = w * data_size;
                    default_stride[1] = (FFALIGN(w, 2) >> 1) * data_size;
                    default_stride[2] = default_stride[1];
                    break;
                }
                case FORMAT_YUV444P: {
                    default_stride[0] = w * data_size;
                    default_stride[1] = w * data_size;
                    default_stride[2] = default_stride[1];
                    break;
                }
                case FORMAT_NV12:
                case FORMAT_NV21: {
                    default_stride[0] = w * data_size;
                    default_stride[1] = FFALIGN(w, 2) * data_size;
                    break;
                }
                case FORMAT_NV16:
                case FORMAT_NV61: {
                    default_stride[0] = w * data_size;
                    default_stride[1] = FFALIGN(w, 2) * data_size;
                    break;
                }
                case FORMAT_GRAY: {
                    default_stride[0] = w * data_size;
                    break;
                }
                // case FORMAT_COMPRESSED: {
                //     image_private->plane_num = 4;
                //     break;
                // }
                case FORMAT_BGR_PACKED:
                case FORMAT_RGB_PACKED: {
                    default_stride[0] = w * 3 * data_size;
                    break;
                }
                case FORMAT_BGR_PLANAR:
                case FORMAT_RGB_PLANAR: {
                    default_stride[0] = w * data_size;
                    break;
                }
                case FORMAT_BGRP_SEPARATE:
                case FORMAT_RGBP_SEPARATE: {
                    default_stride[0] = w * data_size;
                    default_stride[1] = w * data_size;
                    default_stride[2] = w * data_size;
                    break;
                }
            }
        }

        {
            bm_handle_t bmHandle = bm_image_get_handle(&img_);
            if (!bmHandle) {
                SPDLOG_INFO("BMImage is empty, not in any device!");
                return -1;
            }

            int ret=-1;
            bm_image temp_img;
            ret=bm_image_create(bmHandle, h, w, img_.image_format, img_.data_type,
                            &temp_img, default_stride);
            if (ret != BM_SUCCESS){
                SPDLOG_ERROR("bm_image_create err={}", ret);
                return ret;
            }
#if (BMCV_VERSION_MAJOR == 2) && !defined(BMCV_VERSION_MINOR)
            ret = bm_image_alloc_contiguous_mem_heap_mask(1, &temp_img, 2);
#else
            ret = bm_image_alloc_contiguous_mem_heap_mask(1, &temp_img, 6);
#endif
            if (BM_SUCCESS != ret) {
                SPDLOG_ERROR("bm_image_alloc_contiguous_mem_heap_mask err={}", ret);
                return ret;
            }
            ret = bmcv_width_align(bmHandle, img_, temp_img);
            if (BM_SUCCESS != ret) {
                SPDLOG_ERROR("bmcv_width_align err={}", ret);
                return ret;
            }
            destroy();
            img_ = temp_img;
            need_to_free_=true;
            return ret;
        }
    }
    int BMImage::BMImage_CC::check_contiguous_memory() const {
        if (!is_created()) {
            SPDLOG_ERROR("bm_image is not created!");
            return -1;
        }
        // if (!need_to_free_) {
        //     SPDLOG_ERROR("bm_image is not attach memory,can't align!");
        //     exit(1);
        // }
        int data_size = 1;
        switch (img_.data_type) {
            case DATA_TYPE_EXT_FLOAT32:
            data_size = 4;
            break;
#if !((BMCV_VERSION_MAJOR == 2) && !defined(BMCV_VERSION_MINOR))
            case DATA_TYPE_EXT_4N_BYTE:
            case DATA_TYPE_EXT_4N_BYTE_SIGNED:
            data_size = 4;
            break;
#endif
            default:
            data_size = 1;
            break;
        }
        bool if_contiguous = true;
        int w = img_.width;
        int h = img_.height;

        int stride[8];
        bm_image_get_stride(img_, stride);
        {
            switch (img_.image_format) {
                case FORMAT_YUV420P: {
                    if_contiguous &= (stride[0] == w * data_size);
                    if_contiguous &= (stride[1] == (FFALIGN(w, 2) >> 1) * data_size);
                    if_contiguous &= (stride[2] == stride[1]);
                    break;
                }
                case FORMAT_YUV422P: {
                    if_contiguous &= (stride[0] == w * data_size);
                    if_contiguous &= (stride[1] == (FFALIGN(w, 2) >> 1) * data_size);
                    if_contiguous &= (stride[2] == stride[1]);
                    break;
                }
                case FORMAT_YUV444P: {
                    if_contiguous &= (stride[0] == w * data_size);
                    if_contiguous &= (stride[1] == w * data_size);
                    if_contiguous &= (stride[2] == stride[1]);
                    break;
                }
                case FORMAT_NV12:
                case FORMAT_NV21: {
                    if_contiguous &= (stride[0] == w * data_size);
                    if_contiguous &= (stride[1] == FFALIGN(w, 2) * data_size);
                    break;
                }
                case FORMAT_NV16:
                case FORMAT_NV61: {
                    if_contiguous &= (stride[0] == w * data_size);
                    if_contiguous &= (stride[1] == FFALIGN(w, 2) * data_size);
                    break;
                }
                case FORMAT_GRAY: {
                    if_contiguous &= (stride[0] == w * data_size);
                    break;
                }
                // case FORMAT_COMPRESSED: {
                //     image_private->plane_num = 4;
                //     break;
                // }
                case FORMAT_BGR_PACKED:
                case FORMAT_RGB_PACKED: {
                    if_contiguous &= (stride[0] == w * 3 * data_size);
                    break;
                }
                case FORMAT_BGR_PLANAR:
                case FORMAT_RGB_PLANAR: {
                    if_contiguous &= (stride[0] == w * data_size);
                    break;
                }
                case FORMAT_BGRP_SEPARATE:
                case FORMAT_RGBP_SEPARATE: {
                    if_contiguous &= (stride[0] == w * data_size);
                    if_contiguous &= (stride[0] == w * data_size);
                    if_contiguous &= (stride[0] == w * data_size);
                    break;
                }
            }
        }

        return if_contiguous;
    }

#ifdef PYTHON  // asnumpy Python
    pybind11::array BMImage::BMImage_CC::asnumpy() const {
        if (is_created() == false) {
            SPDLOG_ERROR("This BMImage is empty!");
            throw SailBMImageError("Empty BMImage");
        }
        int plane_num = bm_image_get_plane_num(img_);
        std::vector<bm_device_mem_t> mem(plane_num);
        int ret = 0;
        ret = bm_image_get_device_mem(img_, mem.data());
        std::vector<int> bytesizes(plane_num);
        ret = bm_image_get_byte_size(img_, bytesizes.data());
        int total_bytesize = 0;
        for (int i = 0; i < plane_num; ++i) {
            total_bytesize += bytesizes.at(i);
        }
        pybind11::dtype out_dtype;
        int out_numel = total_bytesize;
        switch (img_.data_type) {
            case DATA_TYPE_EXT_FLOAT32:
                out_dtype = pybind11::dtype("float32");
                out_numel = int(total_bytesize / 4);
                break;
            case DATA_TYPE_EXT_1N_BYTE:
                out_dtype = pybind11::dtype("uint8");
                break;
            case DATA_TYPE_EXT_1N_BYTE_SIGNED:
                out_dtype = pybind11::dtype("int8");
            default:
                SPDLOG_ERROR("This BMImage's data type {} is not supported!",
                             img_.data_type);
                throw SailBMImageError("Not Support");
        }
        pybind11::array out_array(out_dtype, out_numel);
        void *ptr = out_array.request().ptr;
        pybind11::gil_scoped_release release;
        // d2s
        int dst_offset = 0;
        bm_handle_t handle = bm_image_get_handle(const_cast<bm_image *>(&img_));
        for (int i = 0; i < plane_num; ++i) {
            void *dst =
                static_cast<void *>(static_cast<char *>(ptr) + dst_offset);
            ret = bm_memcpy_d2s(handle, dst, mem[i]);
            if (ret != BM_SUCCESS) {
                SPDLOG_ERROR("{} failed for d2s error", __func__);
                throw SailBMImageError("Empty BMImage");
            }
            dst_offset += bytesizes[i];
        }
        // ajust shape
        std::vector<int> strides(plane_num);
        ret = bm_image_get_stride(img_, strides.data());
        std::vector<int> out_shape;
        switch (img_.image_format) {
            case FORMAT_BGR_PACKED:
            case FORMAT_RGB_PACKED:
                out_shape = {img_.height, int(strides[0] / 3), 3};
                break;
            case FORMAT_ARGB_PACKED:
            case FORMAT_ABGR_PACKED:
                out_shape = {img_.height, int(strides[0] / 4), 4};
                break;
            case FORMAT_BGR_PLANAR:
            case FORMAT_RGB_PLANAR:
            case FORMAT_YUV444P:
                out_shape = {3, img_.height, strides[0]};
                break;
            case FORMAT_GRAY:
                out_shape = {1, img_.height, strides[0]};
                break;
        }
        pybind11::gil_scoped_acquire gil;
        if (!out_shape.empty()) {
            out_array.resize(out_shape);
        }
        return out_array;
    }
#endif  // asnumpy PYTHON
    
    void BMImage::BMImage_CC::set_ipc_flag(bool f) {
        ipc_recv_flag = f;
    }

    BMImage::BMImage() : _impl(new BMImage_CC()){}

    BMImage::BMImage(bm_image &img) : _impl(new BMImage_CC(img)){}

    BMImage::BMImage(
            Handle &handle,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext dtype
    ) : _impl(new BMImage_CC(handle, h, w, format, dtype)) {}

    BMImage::BMImage(
            Handle &handle,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext dtype,
            int *stride
    ) : _impl(new BMImage_CC(handle, h, w, format, dtype, stride)) {}

    BMImage::BMImage(BMImage &&other) : _impl(new BMImage_CC()){
        *this = std::move(other);
    }

    BMImage::BMImage(
        Handle &handle,
        void *buffer,
        int h,
        int w,
        bm_image_format_ext format,
        bm_image_data_format_ext dtype,
        std::vector<int> strides,
        size_t offset,
        int size)
        : _impl(new BMImage_CC(handle, buffer, h, w, format, dtype, strides,
                               offset, size)) {}

#ifdef PYTHON
    BMImage::BMImage(
        Handle &handle,
        pybind11::buffer &buffer,
        int h,
        int w,
        bm_image_format_ext format,
        bm_image_data_format_ext dtype,
        std::vector<int> strides,
        size_t offset)
        : _impl(new BMImage_CC())
    {
        auto buf_info = buffer.request();
        // TODO check buffer size whether enough for h*strides
        auto shape = buf_info.shape;
        void *ptr = buf_info.ptr;
        *this = BMImage(handle, ptr, h, w, format, dtype, strides, offset);
    }
#endif // PYTHON

    BMImage &BMImage::operator=(BMImage &&other) {
        if (this != &other) {
            destroy();
            _impl->img_.width = other._impl->img_.width;
            _impl->img_.height = other._impl->img_.height;
            _impl->img_.image_format = other._impl->img_.image_format;
            _impl->img_.data_type = other._impl->img_.data_type;
            _impl->img_.image_private = other._impl->img_.image_private;
            _impl->handle_shaptr_ = other._impl->handle_shaptr_;
            _impl->need_to_free_ = other._impl->need_to_free_;
            other._impl->img_.image_private = nullptr;
            other._impl->need_to_free_ = false;
            _impl->mat_buffer = other._impl->mat_buffer;
            other._impl->mat_buffer = NULL;
            _impl->ipc_recv_flag = other._impl->ipc_recv_flag;
            other._impl->ipc_recv_flag = false;
        }
        return *this;
    }

    BMImage &BMImage::operator=(bm_image &&other) {
        spdlog::trace("BMImage::operator=(bm_image &&other)");
        destroy();
        _impl->img_.width = other.width;
        _impl->img_.height = other.height;
        _impl->img_.image_format = other.image_format;
        _impl->img_.data_type = other.data_type;
        _impl->img_.image_private = other.image_private;
        _impl->need_to_free_ = false;
        other = {};
        bm_handle_t key_handle=bm_image_get_handle(&(_impl->img_));
        std::lock_guard<std::mutex> lock(Handle::map_mutex);
        if(Handle::handle_map.find(key_handle) != Handle::handle_map.end())
        {
            Handle* h = Handle::handle_map[key_handle];
            if(h != nullptr) _impl->handle_shaptr_ = h->shaptr();
            // SPDLOG_INFO("Found sail::Handle!");
        }
        else{
            // SPDLOG_INFO("Not find sail::Handle!");
        }
        return *this;
    }

    int BMImage::get_device_id() const {
        bm_handle_t handle_temp = bm_image_get_handle(&_impl->img_);
        if(handle_temp){
            return bm_get_devid(handle_temp);
        }else{
            SPDLOG_INFO("BMImage is empty, Not in any device!");
            return -1;
        }
    }

    Handle BMImage::get_handle(){
        if(_impl->img_.image_private){
            if(_impl->handle_.get_device_id() >= 0){
                return _impl->handle_;
            }else{
                bm_handle_t handle_temp = bm_image_get_handle(&_impl->img_);
                return Handle(handle_temp);
            }
        }else{
            SPDLOG_ERROR("BMImage is empty, Not in any device!");
        }
    }

    BMImage::~BMImage() {
        destroy();
        delete _impl;
    }

    bm_image &BMImage::data() {
        return _impl->img_;
    }

    bm_image BMImage::data() const {
        return _impl->img_;
    }

    int BMImage::width() const { return _impl->img_.width; }

    int BMImage::height() const { return _impl->img_.height; }

    bm_image_format_ext BMImage::format() const { return _impl->img_.image_format; }

    bm_image_data_format_ext BMImage::dtype() const { return _impl->img_.data_type; }

    bool BMImage::need_to_free() const {
        return _impl->need_to_free_; 
    }

    int BMImage::empty_check() const {
        if (!_impl->img_.image_private)
            return 0;
        return 1;
    }

    int BMImage::get_plane_num() const {
        return bm_image_get_plane_num(_impl->img_);
    }
    int BMImage::align() { 
        return _impl->align(); 
    }
    int BMImage::check_align() const { 
        return _impl->check_align(); 
    }
    int BMImage::unalign() { 
        return _impl->unalign(); 
    }
    int BMImage::check_contiguous_memory() const { 
        return _impl->check_contiguous_memory(); 
    }

#ifdef PYTHON // asnumpy Python
    pybind11::array BMImage::asnumpy() const { 
        return _impl->asnumpy(); 
    }
#endif // asnumpy Python

    void BMImage::set_ipc_flag(bool f) {
        return _impl->set_ipc_flag(f);
    }
    void BMImage::create(
            Handle &handle,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext data_type,
            int *stride
    ) {
        return _impl->create(handle, h, w, format, data_type, stride);
    }

    void BMImage::destroy() {
        return _impl->destroy();
    }

    void BMImage::allocate() {
        return _impl->allocate();
    }

    bool BMImage::is_created() const {
        return _impl->is_created();
    }

#ifdef USE_OPENCV
    void BMImage::cache_ost_mat(std::shared_ptr<cv::Mat>& ostrmat){
        return _impl->cache_mat(ostrmat);
    }
#endif

    void BMImage::reset(int w, int h)
    {
        return _impl->reset(w, h);
    }
    void BMImage::detach()
    {
        return _impl->detach();
    }

#if defined(USE_BMCV) && defined(USE_OPENCV) && defined(PYTHON)
    pybind11::array_t<uint8_t> BMImage::asmat(){
        return _impl->asmat();
    }
#endif

    template<std::size_t N>
    BMImageArray<N>::BMImageArray() : need_to_free_(false) {
        for(int i= 0;i < N; ++i) {
            this->at(i).image_format = FORMAT_BGR_PLANAR;
            this->at(i).data_type = DATA_TYPE_EXT_1N_BYTE;
            this->at(i).width = this->at(i).height = 0;
            this->at(i).image_private = nullptr;
        }
    }

    template<std::size_t N>
    BMImageArray<N>::BMImageArray(
        Handle                   &handle,
        int                      h,
        int                      w,
        bm_image_format_ext      format,
        bm_image_data_format_ext dtype
    ) : need_to_free_(false) {
        create(handle, h, w, format, dtype);
    //allocate();
    }

    template<std::size_t N>
    BMImageArray<N>::BMImageArray(
        Handle                   &handle,
        int                      h,
        int                      w,
        bm_image_format_ext      format,
        bm_image_data_format_ext dtype,
        int                      *stride
    ) : need_to_free_(false) {
        create(handle, h, w, format, dtype, stride);
    //allocate();
    }

    // template<std::size_t N>
    // int BMImageArray<N>::attach_data(const std::vector<BMImage> &data){
    //     int ret = 0;
    //     return ret;
    // }

    // template<std::size_t N>
    // BMImageArray<N>::BMImageArray(const std::vector<BMImage> &data)
    // : need_to_free_(false) {
    //     if(data.size() != N){
    //         SPDLOG_ERROR("Error Input size {} vs. {}!", N, data.size());
    //         exit(SAIL_ERR_BMCV_INIT);
    //     }
    // }

    // template<std::size_t N>
    // BMImageArray<N>::BMImageArray(const BMImage &data){

    // }

    template<std::size_t N>
    int BMImageArray<N>::copy_from(int i, BMImage &data){
        int ret = 0;
        bm_handle_t handle = bm_image_get_handle(&data.data());
        if(is_created()){
            if (this->at(0).width        != data.width()  || 
                this->at(0).height       != data.height() || 
                this->at(0).image_format != data.format() ||
                this->at(0).data_type    != data.dtype()) {
                SPDLOG_ERROR("requires src image's format is same as dst");
                print_image(this->at(0));
                print_image(data.data());
            }
        }else{
            int stride[3]={0};
            bm_image_get_stride(data.data(), stride);
            for(int i=0; i<N; i++) {
                ret = bm_image_create(handle,
                    data.height(),
                    data.width(),
                    data.format(), 
                    data.dtype(),
                    &(*this)[i],stride);
                if (ret != BM_SUCCESS){
                    SPDLOG_ERROR("bm_image_create err={}", ret);
                    return ret;
                }
            }
        }
        if(!need_to_free_){
            for(int i=0; i<N; i++) {
                if(bm_image_is_attached((*this)[i])){
                    bm_image_detach((*this)[i]);
                }
                // ret = bm_image_alloc_dev_mem_heap_mask((*this)[i], 6);

                // bm_device_mem_t dev_mem[3];
                // bm_image_get_device_mem(this->data()[0], dev_mem);
                // printf("###: %s:%s:%d, Alloc %p\n",__FILE__,__FUNCTION__,__LINE__,bm_mem_get_device_addr(*dev_mem));

                
            }
#if (BMCV_VERSION_MAJOR == 2) && !defined(BMCV_VERSION_MINOR)
            ret = bm_image_alloc_contiguous_mem_heap_mask(N, this->data(),2);
#else
            ret = bm_image_alloc_contiguous_mem_heap_mask(N, this->data(),6);
#endif
            if (BM_SUCCESS != ret) {
                SPDLOG_ERROR("bm_image_alloc_contiguous_mem_heap_mask err={}", ret);
                return ret;
            }
            need_to_free_ = true;
        }
        bmcv_copy_to_atrr_t attr;
        memset(&attr, 0, sizeof(attr));
        ret = bmcv_image_copy_to(handle, attr, data.data(), (*this)[i]);
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("bmcv_image_copy_to err={}", ret);
        }
        return ret;
    }

    template<std::size_t N>
    int BMImageArray<N>::attach_from(int i, BMImage &data){
        int ret = 0;
        if(need_to_free_){
            destroy();
        }
        if(is_created()){
            if (this->at(0).width        != data.width()  || 
                this->at(0).height       != data.height() || 
                this->at(0).image_format != data.format() ||
                this->at(0).data_type    != data.dtype()) {
                SPDLOG_ERROR("requires src image's format is same as dst");
                print_image(this->at(0));
                print_image(data.data());
            }
        }else{
            bm_handle_t handle = bm_image_get_handle(&data.data());
            int stride[3]={0};
            bm_image_get_stride(data.data(), stride);
            for(int i=0; i<N; i++) {
                ret = bm_image_create(handle,
                    data.height(),
                    data.width(),
                    data.format(), 
                    data.dtype(),
                    &(*this)[i],stride);
                if (ret != BM_SUCCESS){
                    SPDLOG_ERROR("bm_image_create err={}", ret);
                    return ret;
                }
            }
        }
        bm_device_mem_t dev_mem[3];
        ret = bm_image_get_device_mem(data.data(), dev_mem);
        if (ret != BM_SUCCESS){
            SPDLOG_ERROR("bm_image_get_device_mem err={}", ret);
            return ret;
        }
        ret = bm_image_attach((*this)[i], dev_mem);
        if (ret != BM_SUCCESS){
            SPDLOG_ERROR("bm_image_attach err={}", ret);
        }
        // SPDLOG_INFO("bm_image_attach idx {}", i);
        return ret;
    }

    template<std::size_t N>
    BMImageArray<N>::~BMImageArray() {
        destroy();
    }

    template<std::size_t N>
    BMImageArray<N>::BMImageArray(BMImageArray<N> &&other) : need_to_free_(false) {
        *this = std::move(other);
    }

    template<std::size_t N>
    BMImageArray<N>& BMImageArray<N>::operator=(BMImageArray<N> &&other)
    {
        if (this != &other) {
            destroy();
            for (size_t i = 0; i < N; i ++) {
            this->at(i).width         = other.at(i).width;
            this->at(i).height        = other.at(i).height;
            this->at(i).image_format  = other.at(i).image_format;
            this->at(i).data_type     = other.at(i).data_type;
            this->at(i).image_private = other.at(i).image_private;
            other.at(i).image_private = nullptr;
            }
            this->need_to_free_ = other.need_to_free_;
            other.need_to_free_ = false;
        }
        return *this;
    }

    template<std::size_t N>
    bool BMImageArray<N>::is_created() const {
        return !this->empty() && (this->at(0).image_private != nullptr);
    }

    template<std::size_t N>
    bm_image_format_ext BMImageArray<N>::format(int index) const {
        return this->at(index).image_format;
    }

    template<std::size_t N>
    void BMImageArray<N>::create(
    Handle                   &handle,
    int                      h,
    int                      w,
    bm_image_format_ext      format,
    bm_image_data_format_ext dtype,
    int                      *stride
    ) {
        // clear old before.
        destroy();
        // create new instance
        for (size_t i = 0; i < N; i++) {
            bm_image_create(handle.data(), h, w, format, dtype, &this->at(i), stride);
        }
        // int ret = bm_image_alloc_contiguous_mem_heap_mask(N, this->data(),6);
        int ret = bm_image_alloc_contiguous_mem(N, this->data());
        if(ret != BM_SUCCESS) {
            char error_info[512]={0};
            sprintf(error_info,"bm_image_alloc_contiguous_mem error:%d,N[%d],h[%d],w[%d],format[%d],dtype[%d]", 
                ret, N, h, w, format, dtype);
            SPDLOG_ERROR(error_info);
            throw SailBMImageError("bmcv api fail");
        }
        need_to_free_ = true;
    }

    template<std::size_t N>
    void BMImageArray<N>::create_not_alloc(
    Handle                   &handle,
    int                      h,
    int                      w,
    bm_image_format_ext      format,
    bm_image_data_format_ext dtype,
    int                      *stride
    ) {
        // clear old before.
        destroy();
        // create new instance
        int ret = BM_SUCCESS;
        for (size_t i = 0; i < N; i++) {
            ret = bm_image_create(handle.data(), h, w, format, dtype, &this->at(i), stride);
            if(ret != BM_SUCCESS) {
                SPDLOG_ERROR("bm_image_create failed: {}!",ret);
                throw SailBMImageError("bmcv api fail");
            }
        }
        need_to_free_ = false;
    }

    template<std::size_t N>
    void BMImageArray<N>::create(
            Handle                   &handle,
            int                      h,
            int                      w,
            bm_image_format_ext      format,
            bm_image_data_format_ext dtype
    ) {
        create(handle, h, w, format, dtype, nullptr);
    }

    template<std::size_t N>
    void BMImageArray<N>::reset(
            int                      h,
            int                      w) {

        if (this->at(0).width != w || this->at(0).height != h)
        {
            SPDLOG_INFO("reset image, src({},{}) dst({},{})",
                    this->at(0).width,
                    this->at(0).height,
                    w, h);

            bm_handle_t bmHandle=nullptr;
            if (need_to_free_) {
                bmHandle = bm_image_get_handle(&this->at(0));
                bm_image_free_contiguous_mem(N, this->data());
                need_to_free_ = false;
            }
            if(is_created()){
                for(int i=0; i<N; i++) {
                    if(bm_image_is_attached((*this)[i])){
                        bm_image_detach((*this)[i]);
                    }
                    bm_image_destroy(this->at(i));
                    this->at(i).image_private=nullptr;
                }
            }
            if (bmHandle != nullptr) {
                for (size_t i = 0; i < N; i ++) {
                    bm_image_create(bmHandle, h, w, this->at(0).image_format, this->at(0).data_type, &this->at(i));
                }
            }

            int ret = bm_image_alloc_contiguous_mem(N, this->data());
            // int ret = bm_image_alloc_contiguous_mem_heap_mask(N, this->data(),6);
            if(ret != BM_SUCCESS) {
                SPDLOG_ERROR("bm_image_alloc_contiguous_mem err={}",ret);
                throw SailRuntimeError("device memory not enough");
            }
            need_to_free_ = true;
        }
    }

    template<std::size_t N>
    void BMImageArray<N>::destroy() {
        if (need_to_free_){
            int ret = bm_image_free_contiguous_mem(N, this->data());
            if(ret != 0){
                SPDLOG_ERROR("bm_image_free_contiguous_mem err={}", ret);
                throw SailBMImageError("bmcv api fail");
            }
            need_to_free_ = false;
        }

        if(is_created()){
            for (size_t i = 0; i < N; i++) {
                if(bm_image_is_attached((*this)[i])){
                    bm_image_detach((*this)[i]);
                }
                bm_image_destroy(this->at(i));
                this->at(i).image_private = nullptr;
            }
        }
    }

    template<std::size_t N>
    void BMImageArray<N>::to_tensor(Tensor &tensor){
        if(N <= 0){
            SPDLOG_ERROR("The size of the array must be greater than zero.");
            return;
        }
        if (this->at(0).image_format != FORMAT_RGB_PLANAR &&
            this->at(0).image_format != FORMAT_BGR_PLANAR &&
            this->at(0).image_format != FORMAT_BGR_PACKED &&
            this->at(0).image_format != FORMAT_BGR_PACKED)
        {
            SPDLOG_ERROR("Only support image format BGR or RGB. Not support {}. Please convert it first.", this->at(0).image_format) ;
            return;
        }
        int ret = 0;
        bm_data_type_t dtype = get_bm_data_type_sail(this->at(0).data_type);
        bm_device_mem_t addr;
        if (!need_to_free_) {
            SPDLOG_ERROR("input BMImage doesn't have continuous memory!");
            return;
        }
        ret = bm_image_get_contiguous_device_mem(N, this->data(), &addr);
        if (ret != BM_SUCCESS) {
            SPDLOG_ERROR("bm_image_to_tensor err={}", ret);
            throw SailBMImageError("bmcv api fail");
        }

        if (addr.u.device.device_addr == 0) {
            SPDLOG_ERROR("to_tensor: dev_data is null!");
            throw SailBMImageError("invalid argument");
        }

        int h = this->at(0).height;
        int w = this->at(0).width;

        tensor.reset({N, 3, h, w}, dtype);
        tensor.reset_dev_data(addr);
    }

    template<std::size_t N>
    int BMImageArray<N>::get_device_id(){
        if(N <= 0){
            SPDLOG_ERROR("The size of the array must be greater than zero.");
            return -1;
        }
        bm_handle_t handle_temp = bm_image_get_handle(&this->at(0));
        if(handle_temp){
            return bm_get_devid(handle_temp);
        }else{
            SPDLOG_INFO("BMImage is empty, Not in any device!");
            return -1;
        }
    }

    Bmcv::Bmcv(Handle &handle) : handle_(handle) {}

    Bmcv::~Bmcv() {}

#if defined(USE_BMCV) && defined(USE_OPENCV)
#if defined(PYTHON)
    BMImage Bmcv::mat_to_bm_image(pybind11::array_t<uint8_t> &input_array){
        BMImage img;
        int ret = mat_to_bm_image(input_array, img);
        SAIL_CHECK_RET(ret);
        return img; 
    }
    int Bmcv::mat_to_bm_image(pybind11::array_t<uint8_t> &input_array, BMImage &img){
        spdlog::trace("before import numpy");
        pybind11::module np = pybind11::module::import("numpy");  // like 'import numpy as np'
        spdlog::trace("after import numpy");
        pybind11::buffer_info buf = input_array.request();
        if (!pybind11::detail::check_flags(input_array.ptr(), pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_)) {
            pybind11::array_t<uint8_t> arr_c = np.attr("ascontiguousarray")(input_array, "dtype"_a="uint8");
            buf = arr_c.request();
        }
        if(buf.shape.size() != 3) {
            SPDLOG_ERROR("mat_to_bm_image failed, input mat's ndim must be 3!");
            return SAIL_ERR_BMI_SHAPE;
        }
        if(buf.shape[2] == 3){ //CV_8UC3
            cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, cv::SophonDevice(get_handle().get_device_id()));
            if(mat.step == buf.shape[1] * buf.shape[2]){
                memcpy(mat.data, buf.ptr, buf.size * buf.itemsize);
            }else{
                for(int y = 0; y < buf.shape[0]; ++y) {
                    memcpy(mat.data + y * mat.step, buf.ptr + y * buf.shape[1] * 3, buf.shape[1] * 3);
                }            
            }
            int ret = mat_to_bm_image(mat, img);
            if (ret) {
                SPDLOG_ERROR("mat_to_bm_image failed, ret {}.", ret);
                return SAIL_ERR_BMI_CVT;
            }
        }else{
            SPDLOG_ERROR("This mat has invalid channel num: %d", buf.shape[2]);
            return SAIL_ERR_BMI_SHAPE;
        }
        return 0;
    } 
#endif
    int Bmcv::mat_to_bm_image(cv::Mat &mat, BMImage &img) {
        if (mat.cols == 0 || mat.rows == 0) {
            SPDLOG_ERROR("mat_to_bm_image err = input mat must not empty!");
            return SAIL_ERR_BMI_EMPTY;
        }
        std::shared_ptr<cv::Mat> cached_mat = std::make_shared<cv::Mat>(mat);
        img.cache_ost_mat(cached_mat);
        bm_status_t ret = cv::bmcv::toBMI(mat, &img.data());
        if (ret) {
            SPDLOG_ERROR("mat_to_bm_image toBMI() failed, ret {}.", ret);
            return SAIL_ERR_BMI_SHAPE;
        }
        return 0;
    }

    BMImage Bmcv::mat_to_bm_image(cv::Mat &mat) {
        BMImage img;
        int ret = mat_to_bm_image(mat, img);
        SAIL_CHECK_RET(ret);
        // return std::move(img);
        return img;
    }

    int Bmcv::bm_image_to_mat(BMImage &img, cv::Mat &mat) {
        bm_status_t ret = cv::bmcv::toMAT(&img.data(), mat);
        if (ret) {
            SPDLOG_ERROR("bm_image_to_mat toMAT() failed, ret {}", ret);
            return SAIL_ERR_BMI_CVT;
        }
        return ret;
    }

    cv::Mat Bmcv::bm_image_to_mat(BMImage &img) {
        cv::Mat mat;
        int ret = bm_image_to_mat(img, mat);
        SAIL_CHECK_RET(ret);
        return std::move(mat);
    }

    void Bmcv::bm_image_to_tensor(BMImage &img, Tensor &tensor) {
        if(!img.check_contiguous_memory()){
            SPDLOG_ERROR("bm_image err = input bm_image must be contiguous!");
            return ;
        }
        bm_data_type_t dtype = get_bm_data_type(img.dtype());
        bm_device_mem_t addr;
        int ret = bm_image_get_device_mem(img.data(), &addr);
        if (ret != 0){
            SPDLOG_ERROR("bm_image_to_tensor err={}, bm_image_get_device_mem failed", ret);
            throw SailBMImageError("bmcv api fail");
        }

        int stride = 0;
        ret = bm_image_get_stride(img.data(),&stride);
        if (ret != 0){
            SPDLOG_ERROR("bm_image_to_tensor err={}, bm_image_get_stride failed", ret);
            throw SailBMImageError("bmcv api fail");
        }

        int h = img.height();
        int w = img.width();

        if (img.data().image_format == FORMAT_RGB_PLANAR || img.data().image_format == FORMAT_BGR_PLANAR) {
            tensor.reset({1, 3, h, w}, dtype);
        }else if(img.data().image_format == FORMAT_RGB_PACKED || img.data().image_format == FORMAT_BGR_PACKED){
            tensor.reset({1, h, w, 3}, dtype);
        }else{
            SPDLOG_ERROR("Image format not supported, Please convert it first.");
            throw SailBMImageError("not supported");
        }
            
        tensor.reset_dev_data(addr);
#if 0
#ifdef IS_SOC_MODE
        uint8_t * sys_data= new uint8_t[addr.size];
        double process_start_time_d2s = get_current_time_us();
        ret = bm_memcpy_d2s_partial(get_handle().data(), sys_data,addr, addr.size);
        if (ret != 0){
            SPDLOG_ERROR("bm_image_to_tensor err={}, bm_memcpy_d2s_partial failed", ret);
            throw SailBMImageError("bmlib api fail");
        }
        PRINT_TIME_MS("bm_memcpy_d2s_partial", process_start_time_d2s)
        delete [] sys_data;
#endif
#endif
    }

    Tensor Bmcv::bm_image_to_tensor(BMImage &img) {
        Tensor tensor(get_handle());
        bm_image_to_tensor(img, tensor);
        return std::move(tensor);
    }

    void Bmcv::tensor_to_bm_image(Tensor& tensor, BMImage& img, bool bgr2rgb,
                            std::string layout) {
        auto shape = tensor.shape();
        int h, w;
        if (strcmp(layout.c_str(), "nchw") == 0) {
            h = shape[2];
            w = shape[3];
        } else if (strcmp(layout.c_str(), "nhwc") == 0) {
            h = shape[1];
            w = shape[2];
        } else {
            throw std::invalid_argument("Invalid layout!");
        }

        bm_image_data_format_ext dtype = get_bm_image_data_format(tensor.dtype());

        if (img.need_to_free()) {
            img.destroy();
        }
        if(img.is_created()){
            img.detach();
        }
        else {
            int dtype_size = bm_image_data_type_size(dtype);
            // int stride = FFALIGN(w * dtype_size, SAIL_ALIGN); // ceiling to 64 * N
            if (strcmp(layout.c_str(), "nchw") == 0) {
                img.create(
                        handle_,
                        h,
                        w,
                        bgr2rgb ? FORMAT_RGB_PLANAR : FORMAT_BGR_PLANAR,
                        dtype);
            } else {
                img.create(
                        handle_,
                        h,
                        w,
                        bgr2rgb ? FORMAT_RGB_PACKED : FORMAT_BGR_PACKED,
                        dtype);
            }
        }

        bm_device_mem_t mem = tensor.dev_data();
        int ret = bm_image_attach(img.data(), &mem);
        if (ret != 0){
            SPDLOG_ERROR("tensor_to_bm_image err={}, bm_image_attach failed", ret);
            throw SailBMImageError("bmcv api fail");
        }
    }

    void Bmcv::tensor_to_bm_image(Tensor &tensor, BMImage &img, bm_image_format_ext format) {
        auto shape = tensor.shape();
        int h, w;
        if(format == FORMAT_RGB_PLANAR || format == FORMAT_BGR_PLANAR) { //nchw
            h = shape[2];
            w = shape[3];
        } else if (format == FORMAT_RGB_PACKED ||format == FORMAT_BGR_PACKED) { //nhwc
            h = shape[1];
            w = shape[2];
        } else {
            throw std::invalid_argument("Invalid format!");
        }

        bm_image_data_format_ext dtype = get_bm_image_data_format(tensor.dtype());

        if (img.need_to_free()) {
            img.destroy();
        }
        if(img.is_created()){
            img.detach();
        }
        else {
            int dtype_size = bm_image_data_type_size(dtype);
            // int stride = FFALIGN(w * dtype_size, SAIL_ALIGN); // ceiling to 64 * N
            img.create(
                    handle_,
                    h,
                    w,
                    format,
                    dtype);
        }

        bm_device_mem_t mem = tensor.dev_data();
        int ret = bm_image_attach(img.data(), &mem);
        if (ret != 0){
            SPDLOG_ERROR("tensor_to_bm_image err={}, bm_image_attach failed", ret);
            throw SailBMImageError("bmcv api fail");
        }
    }

    BMImage Bmcv::tensor_to_bm_image(Tensor &tensor, bool bgr2rgb/*false*/, std::string layout/*nchw*/) {
        BMImage img;
        tensor_to_bm_image(tensor, img, bgr2rgb, layout);
        return std::move(img);
    }

    BMImage Bmcv::tensor_to_bm_image(Tensor &tensor, bm_image_format_ext format) {
        BMImage img;
        tensor_to_bm_image(tensor, img, format);
        return std::move(img);
    }

    int Bmcv::crop_and_resize(
            bm_image *input,
            bm_image *output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h,
            int input_num,
            bmcv_resize_algorithm resize_alg
    ) {
        if(input_num <= 0){
            SPDLOG_ERROR("crop_and_resize error, invalid input_num[{}]!",input_num);
            return SAIL_ERR_BMI_PARAM;
        }
        bmcv_resize_t attr1;
        attr1.start_x = crop_x0;
        attr1.start_y = crop_y0;
        attr1.in_width = crop_w;
        attr1.in_height = crop_h;
        attr1.out_width = resize_w;
        attr1.out_height = resize_h;

        bmcv_resize_image attr0;
        attr0.resize_img_attr = &attr1;
        attr0.roi_num = 1;
        attr0.stretch_fit = 1;
        attr0.interpolation = resize_alg;

        switch(resize_alg){
            case BMCV_INTER_NEAREST: 
            case BMCV_INTER_LINEAR:
            case BMCV_INTER_BICUBIC:
            break;
            default:
            SPDLOG_INFO("Error resize_alg, use default BMCV_INTER_NEAREST");
            break;
        }

        int ret = bmcv_image_resize(
                handle_.data(),
                input_num,
                &attr0,
                input,
                output);
        if (ret) {
            SPDLOG_ERROR("crop_and_resize failed, bmcv_image_resize() ret {}", ret);
            return SAIL_ERR_BMI_BMCV;
        }
        return ret;
    }

    int Bmcv::crop_and_resize(
            BMImage &input,
            BMImage &output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h,
            bmcv_resize_algorithm resize_alg
    ) {
        
        if (output.is_created()) {
            if(output.width() != resize_w || output.height() != resize_h){
                SPDLOG_INFO("output will be reset to {}x{}",resize_w, resize_h);
            }
            output.reset(resize_w, resize_h);
        }

        if (!output.is_created()) {
            /* vpp limitation: 64-aligned */
            int dtype_size = bm_image_data_type_size(input.dtype());
            
            bm_image_format_ext temp_format = input.format();
            if(temp_format != FORMAT_BGR_PLANAR && temp_format != FORMAT_RGB_PLANAR){
                temp_format = FORMAT_BGR_PLANAR;
            }

            int stride = FFALIGN(resize_w * dtype_size, SAIL_ALIGN); // ceiling to 64 * N
            output.create(
                    handle_,
                    resize_h,
                    resize_w,
                    temp_format, // same as input format or FORMAT_RGB_PLANAR
                    input.dtype(),
                    &stride);
            output.allocate();
        }

        return crop_and_resize(&input.data(), &output.data(),
            crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, 1, resize_alg);
    }

    BMImage Bmcv::crop_and_resize(
            BMImage &input,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h,
            bmcv_resize_algorithm resize_alg) {
        BMImage output;
        int ret = crop_and_resize(input, output, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, resize_alg);
        SAIL_CHECK_RET(ret);
        return std::move(output);
    }

    int Bmcv::crop(
            BMImage &input,
            BMImage &output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h
    ) {
        return crop_and_resize(input, output, crop_x0, crop_y0, crop_w, crop_h, crop_w, crop_h);
    }

    BMImage Bmcv::crop(
            BMImage &input,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h) {
        BMImage output;
        int ret = crop(input, output, crop_x0, crop_y0, crop_w, crop_h);
        SAIL_CHECK_RET(ret);
        return std::move(output);
    }

    vector<BMImage> Bmcv::crop(BMImage &input, vector<vector<int>> rects){
        
        if(input.format()!=FORMAT_BGR_PACKED&&input.format()!=FORMAT_BGR_PLANAR
        &&input.format()!=FORMAT_RGB_PACKED&&input.format()!=FORMAT_RGB_PLANAR
        &&input.format()!=FORMAT_GRAY){
            SPDLOG_ERROR("Image format not supported, Please convert it first.");
        }
        if(rects.size()==0) SPDLOG_ERROR("Rects is NULL");
        vector<BMImage> out;
        bm_image *output=new bm_image[rects.size()];
        bmcv_rect_t* p_rects=new bmcv_rect_t[rects.size()];
        bm_status_t ret;
        for(int i=0;i<rects.size();i++){
            p_rects[i].start_x=rects[i][0];
            p_rects[i].start_y=rects[i][1];
            p_rects[i].crop_w=rects[i][2];
            p_rects[i].crop_h=rects[i][3];
            if(p_rects[i].start_x<0||p_rects[i].start_y<0||
                p_rects[i].crop_w+p_rects[i].start_x>=input.width()||
                p_rects[i].crop_h+p_rects[i].start_y>=input.height()){
                SPDLOG_ERROR("The crop rects is out of the image!");
                // should return and delete
            }
            ret = bm_image_create(handle_.data(),p_rects[i].crop_h,p_rects[i].crop_w,input.format(),input.dtype(),(output+i));
            if (ret) {
                SPDLOG_ERROR("crop failed, bm_image_create() ret {}", ret);
                delete[] p_rects;
                delete[] output;
                throw SailBMImageError("Bmcv crop() failed!");
            }
        }
        ret = bmcv_image_crop(handle_.data(),rects.size(),p_rects,input.data(),output);
        if (ret) {
            SPDLOG_ERROR("crop failed, bmcv_image_crop() ret {}", ret);
            delete[] p_rects;
            delete[] output;
            throw SailBMImageError("Bmcv crop() failed!");
        }
        for(int i =0;i<rects.size();i++){
            out.emplace_back(BMImage(output[i]));
        }
        delete[] p_rects;
        delete[] output;
        return std::move(out);
    }

    int Bmcv::resize(
            BMImage &input,
            BMImage &output,
            int resize_w,
            int resize_h,
            bmcv_resize_algorithm resize_alg
    ) {
        int ret;
        ret = crop_and_resize(input, output, 0, 0, input.data().width, input.data().height, resize_w, resize_h, resize_alg);
        if (ret) {
            SPDLOG_ERROR("resize failed, crop_and_resize() ret {}", ret);
        }
        return ret;
    }

    BMImage Bmcv::resize(
            BMImage &input,
            int resize_w,
            int resize_h,
            bmcv_resize_algorithm resize_alg) {
        BMImage output;
        int ret = resize(input, output, resize_w, resize_h, resize_alg);
        // judge ret
        SAIL_CHECK_RET(ret);
        return std::move(output);
    }

    int Bmcv::vpp_crop_and_resize(
            bm_image *input,
            bm_image *output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h,
            int input_num,
            bmcv_resize_algorithm resize_alg
    ) {
        if(input_num <= 0){
            printf("vpp_crop_and_resize error, invalid input_num[%d]!\n",input_num);
            return BM_ERR_DATA;
        }

        bmcv_rect rect;
        rect.start_x = crop_x0;
        rect.start_y = crop_y0;
        rect.crop_w = crop_w;
        rect.crop_h = crop_h;
        
        int ret = BM_SUCCESS;
        for (size_t i = 0; i < input_num; i++)        {
            ret = bmcv_image_vpp_convert(
                handle_.data(),
                1,
                input[i],
                &output[i],
                &rect,
                resize_alg);
            if(ret != BM_SUCCESS){
                break;
            }
        }
        if (ret == BM_NOT_SUPPORTED) {
            SPDLOG_WARN("vpp_crop_and_resize error, vpp not support, try tpu resize");
            print_image(input[0], " src:");
            print_image(output[0], " dst:");
            // vpp not support, try tpu resize
            bmcv_resize_t roi_attr[1];
            bmcv_resize_image resize_attr[1];
            memset(resize_attr, 0, sizeof(resize_attr));
            resize_attr[0].roi_num = 1;
            resize_attr[0].stretch_fit =1 ;
            resize_attr[0].interpolation = resize_alg;
            roi_attr[0].start_x = 0;
            roi_attr[0].start_y = 0;
            roi_attr[0].in_width = input[0].width;
            roi_attr[0].in_height = input[0].height;
            roi_attr[0].out_width = output[0].width;
            roi_attr[0].out_height = output[0].height;
            resize_attr[0].resize_img_attr = &roi_attr[0];
            ret = bmcv_image_resize(handle_.data(), input_num, resize_attr, input, output);
            if (BM_SUCCESS != ret) {
                SPDLOG_ERROR("bmcv_image_resize err={}", ret);
                print_image(input[0], " src:");
                print_image(output[0], " dst:");
                return SAIL_ERR_BMI_BMCV;
            }
        }else {
            if (BM_SUCCESS != ret) {
                SPDLOG_ERROR("bmcv_image_vpp_convert err={}", ret);
                print_image(input[0], " src:");
                print_image(output[0], " dst:");
                return SAIL_ERR_BMI_BMCV;
            }
        }


        return ret;
    }
    int Bmcv::vpp_crop_and_resize(
            BMImage &input,
            BMImage &output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h,
            bmcv_resize_algorithm resize_alg
    ) {
        if (output.is_created()) {
            if(output.width() != resize_w || output.height() != resize_h){
                SPDLOG_INFO("output will be reset to {}x{}",resize_w, resize_h);
            }
            output.reset(resize_w, resize_h);
        }

        if (!output.is_created()) {
            /* vpp limitation: 64-aligned */
            int dtype_size = bm_image_data_type_size(input.dtype());
            bm_image_format_ext temp_format = input.format();
            if(temp_format != FORMAT_BGR_PLANAR && temp_format != FORMAT_RGB_PLANAR){
                temp_format = FORMAT_BGR_PLANAR;
            }
            int stride = FFALIGN(resize_w * dtype_size, SAIL_ALIGN); // ceiling to 64 * N
            output.create(
                    handle_,
                    resize_h,
                    resize_w,
                    temp_format, // same as input format or FORMAT_RGB_PLANAR
                    input.dtype(),
                    &stride
            );
            output.allocate();
        }
        return vpp_crop_and_resize(&input.data(), &output.data(),
            crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, 1, resize_alg);
    }

    BMImage Bmcv::vpp_crop_and_resize(
            BMImage &input,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h,
            bmcv_resize_algorithm resize_alg) {
        BMImage output;
        int ret = vpp_crop_and_resize(input, output, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, resize_alg);
        SAIL_CHECK_RET(ret);
        return std::move(output);
    }

    int Bmcv::vpp_crop_and_resize_padding(
      bm_image                     *input,
      bm_image                     *output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in,
      int                          input_num,
      bmcv_resize_algorithm        resize_alg){
        if(input_num <= 0){
            printf("vpp_crop_and_resize_padding error, invalid input_num[%d]!\n",input_num);
            return SAIL_ERR_BMI_PARAM;
        }

        bmcv_rect rect;
        rect.start_x = crop_x0;
        rect.start_y = crop_y0;
        rect.crop_w = crop_w;
        rect.crop_h = crop_h;

        int ret = 0;
        for (size_t i = 0; i < input_num; i ++) {

            // filling output
            bmcv_padding_atrr_t padding;
            padding.dst_crop_stx = padding_in.dst_crop_stx;
            padding.dst_crop_sty = padding_in.dst_crop_sty;
            padding.dst_crop_w   = padding_in.dst_crop_w;
            padding.dst_crop_h   = padding_in.dst_crop_h;
            padding.padding_r   = padding_in.padding_r;
            padding.padding_g   = padding_in.padding_g;
            padding.padding_b   = padding_in.padding_b;
            padding.if_memset    = 1;

            int width  = output[i].width;
            int height = output[i].height;
  
            ret = bmcv_image_vpp_convert_padding(
                handle_.data(),
                1,
                input[i],
                &output[i],
                &padding,
                &rect,
                resize_alg);
            if (BM_SUCCESS != ret) {
                SPDLOG_ERROR("bmcv_image_vpp_convert_padding err={}", ret);
                print_image(input[i], " src:");
                print_image(output[i], " dst:");
                break;
            }
        }
        if (ret) {
            return SAIL_ERR_BMI_BMCV;
        }
        return ret;
    }

    int Bmcv::vpp_crop_and_resize_padding(
            BMImage &input,
            BMImage &output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h,
            PaddingAtrr &padding_in,
            bmcv_resize_algorithm resize_alg) {

        if (output.is_created()) {
            if(output.width() != resize_w || output.height() != resize_h){
                SPDLOG_INFO("output will be reset to {}x{}",resize_w, resize_h);
            }
            output.reset(resize_w, resize_h);
        }

        if (!output.is_created()) {
            /* vpp limitation: 64-aligned */
            int dtype_size = bm_image_data_type_size(input.dtype());
            bm_image_format_ext temp_format = input.format();
            if(temp_format != FORMAT_BGR_PLANAR && temp_format != FORMAT_RGB_PLANAR){
                temp_format = FORMAT_BGR_PLANAR;
            }
            int stride = FFALIGN(resize_w * dtype_size, SAIL_ALIGN); // ceiling to 64 * N
            output.create(
                    handle_,
                    resize_h,
                    resize_w,
                    temp_format, // same as input format or FORMAT_RGB_PLANAR
                    input.dtype(),
                    &stride
            );
            output.allocate();
        }

        return vpp_crop_and_resize_padding(&input.data(), &output.data(), crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, padding_in, 1, resize_alg);
    }

    BMImage Bmcv::vpp_crop_and_resize_padding(
            BMImage &input,
            int crop_x0,
            int crop_y0, 
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h,
            PaddingAtrr &padding_in,
            bmcv_resize_algorithm resize_alg) {
        BMImage output;
        int ret = vpp_crop_and_resize_padding(input, output, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, padding_in, resize_alg);
        SAIL_CHECK_RET(ret);
        return std::move(output);
    }

    int Bmcv::vpp_crop(
            BMImage &input,
            BMImage &output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h
    ) {
        return vpp_crop_and_resize(input, output, crop_x0, crop_y0, crop_w, crop_h, crop_w, crop_h);
    }

    BMImage Bmcv::vpp_crop(
            BMImage &input,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h) {
        BMImage output;
        int ret = vpp_crop(input, output, crop_x0, crop_y0, crop_w, crop_h);
        SAIL_CHECK_RET(ret);
        return std::move(output);
    }

    int Bmcv::vpp_resize(
            BMImage &input,
            BMImage &output,
            int resize_w,
            int resize_h,
            bmcv_resize_algorithm resize_alg
    ) {
        return vpp_crop_and_resize(input, output, 0, 0, input.data().width, input.data().height, resize_w, resize_h, resize_alg);
    }

    BMImage Bmcv::vpp_resize(
            BMImage &input,
            int resize_w,
            int resize_h,
            bmcv_resize_algorithm resize_alg) {
        BMImage output;
        int ret = vpp_resize(input, output, resize_w, resize_h, resize_alg);
        SAIL_CHECK_RET(ret);

        /*
        if (bm_image_get_plane_num(output.data()) != 1 || bm_image_get_plane_num(input.data()) != 3)
            std::cout << "****** vpp_resize: " << bm_image_get_plane_num(input.data()) << " "
                      << bm_image_get_plane_num(output.data()) << endl;
         */
        return std::move(output);
    }

    int Bmcv::vpp_resize_padding(
            BMImage &input,
            BMImage &output,
            int resize_w,
            int resize_h,
            PaddingAtrr &padding_in,
            bmcv_resize_algorithm resize_alg
    ) {
        return vpp_crop_and_resize_padding(input, output, 0, 0, input.data().width, input.data().height, resize_w,
                                           resize_h, padding_in, resize_alg);
    }

    BMImage Bmcv::vpp_resize_padding(
            BMImage &input,
            int resize_w,
            int resize_h,
            PaddingAtrr &padding_in,
            bmcv_resize_algorithm resize_alg) {
        BMImage output;
        int ret = vpp_resize_padding(input, output, resize_w, resize_h, padding_in, resize_alg);
        SAIL_CHECK_RET(ret);
        return std::move(output);
    }

    int Bmcv::warp(
        bm_image *input,
        bm_image *output,
        const std::pair<
        std::tuple<float, float, float>,
        std::tuple<float, float, float>> *matrix,
        int input_num,
        int use_bilinear,
        bool similar_to_opencv
    ){
        if(input_num <= 0 || !matrix){
            printf("warp error, invalid input_num[%d]!\n",input_num);
            return SAIL_ERR_BMI_PARAM;
        }
        for (int i = 0; i < input_num; i ++) {
            if (input[i].image_format != FORMAT_RGB_PLANAR && input[i].image_format != FORMAT_BGR_PLANAR) {
                SPDLOG_ERROR("Only support image format FORMAT_BGR_PLANAR or FORMAT_RGB_PLANAR. Please convert it first.");
                return SAIL_ERR_BMI_NOTSUP;
            }
        }
        
        bmcv_warp_image_matrix *attr0 = new bmcv_warp_image_matrix[input_num];
        bmcv_warp_matrix *attr1 = new bmcv_warp_matrix[input_num];
        for (int i = 0; i < input_num; i ++) {
            
            attr1[i].m[0] = std::get<0>(matrix[i].first);
            attr1[i].m[1] = std::get<1>(matrix[i].first);
            attr1[i].m[2] = std::get<2>(matrix[i].first);
            attr1[i].m[3] = std::get<0>(matrix[i].second);
            attr1[i].m[4] = std::get<1>(matrix[i].second);
            attr1[i].m[5] = std::get<2>(matrix[i].second);

            attr0[i].matrix = &attr1[i];
            attr0[i].matrix_num = 1;
        }

        int return_value;
        if (similar_to_opencv)
            return_value = bmcv_image_warp_affine_similar_to_opencv(handle_.data(), input_num, attr0, input, output, use_bilinear);
        else
            return_value = bmcv_image_warp_affine(handle_.data(), input_num, attr0, input, output, use_bilinear);
        delete []attr0;
        delete []attr1;
        if (return_value) {
            SPDLOG_WARN("warp failed, bmcv_image_warp_affine() ret {}.", return_value);
            return SAIL_ERR_BMI_BMCV;
        }
        return return_value;
    }

    int Bmcv::warp(
            BMImage &input,
            BMImage &output,
            const std::pair<
                    std::tuple<float, float, float>,
                    std::tuple<float, float, float>> &matrix,
            int use_bilinear,
            bool similar_to_opencv
    ) {
        if (!output.is_created()) {
            output.create(
                    handle_,
                    input.height(),
                    input.width(),
                    input.format(),
                    input.dtype()
            );
            output.allocate();
        }

        return warp(&input.data(), &output.data(), &matrix, 1, use_bilinear, similar_to_opencv);
    }

    BMImage Bmcv::warp(
            BMImage &input,
            const std::pair<
                    std::tuple<float, float, float>,
                    std::tuple<float, float, float>> &matrix,
            int use_bilinear,
            bool similar_to_opencv) {
        BMImage output;
        int ret = warp(input, output, matrix, use_bilinear, similar_to_opencv);
        SAIL_CHECK_RET(ret);
        return std::move(output);
    }

    int Bmcv::convert_to(
      bm_image *input,
      bm_image *output,
      const std::tuple<
        std::pair<float, float>,
        std::pair<float, float>,
        std::pair<float, float>>   &alpha_beta,
        int input_num){
        if(input_num <= 0){
            printf("bmcv_image_convert_to error, invalid input_num[%d]!\n",input_num);
            return SAIL_ERR_BMI_PARAM;
        }

        // Check if the input and output meet the requirements
        auto input_height = input[0].height;
        auto input_width = input[0].width;
        auto input_data_type = input[0].data_type;
        auto input_image_format = input[0].image_format;

        auto output_height = output[0].height;
        auto output_width = output[0].width;
        auto output_data_type = output[0].data_type;
        auto output_image_format = output[0].image_format;
        for(int i = 1; i < input_num; i++) {
            if(input[i].height != input_height || input[i].width != input_width || input[i].image_format != input_image_format || input[i].data_type != input_data_type) {
                SPDLOG_ERROR("convert_to error: The width, height, data type, and format of each input must be equal!");
                print_image(input[i], "src: ");
                return SAIL_ERR_BMI_PARAM;
            }
            if(output[i].height != output_height || output[i].width != output_width || output[i].image_format != output_image_format || output[i].data_type != output_data_type) {
                SPDLOG_ERROR("convert_to error: The width, height, data type, and format of each output must be equal!");
                print_image(output[i], "dst: ");
                return SAIL_ERR_BMI_PARAM;
            }
        }

        bmcv_convert_to_attr attr;
        attr.alpha_0 = std::get<0>(alpha_beta).first;
        attr.beta_0 = std::get<0>(alpha_beta).second;
        attr.alpha_1 = std::get<1>(alpha_beta).first;
        attr.beta_1 = std::get<1>(alpha_beta).second;
        attr.alpha_2 = std::get<2>(alpha_beta).first;
        attr.beta_2 = std::get<2>(alpha_beta).second;

        spdlog::debug("convert_to alpha_beta {} {} {} {} {} {}",
                     attr.alpha_0, attr.beta_0,
                     attr.alpha_1, attr.beta_1,
                     attr.alpha_2, attr.beta_2);
        
        int ret = BM_SUCCESS;
        if(input_image_format == FORMAT_BGR_PACKED) {
            if(handle_.get_target() == "1684") {
                if(input_data_type == DATA_TYPE_EXT_1N_BYTE_SIGNED) {
                    bm_image* input_bgr_planar_image = new bm_image[input_num];
                    for(int i = 0; i < input_num; i++) {
                        ret = bm_image_create(handle_.data(), input_height, input_width, FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE_SIGNED, &input_bgr_planar_image[i]);
                        if(BM_SUCCESS != ret) {
                            SPDLOG_ERROR("convert_to error: call bm_image_create failed!");
                            return ret;
                        }
                    }

                    for(int i = 0; i < input_num; i++) {
                        input[i].data_type = DATA_TYPE_EXT_1N_BYTE;
                        input_bgr_planar_image[i].data_type = DATA_TYPE_EXT_1N_BYTE;
                    }

                    ret = bmcv_image_storage_convert(handle_.data(), input_num, input, input_bgr_planar_image);
                    if(BM_SUCCESS != ret) {
                        SPDLOG_ERROR("convert_to error: call bmcv_image_storage_convert failed!");
                        for(int i = 0; i < input_num; i++) {
                            bm_image_destroy(input_bgr_planar_image[i]);
                        }
                        return ret;
                    }

                    for(int i = 0; i < input_num; i++) {
                        input[i].data_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
                        input_bgr_planar_image[i].data_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
                    }
                    
                    bm_image* output_bgr_planar_image = new bm_image[input_num];
                    for(int i = 0; i < input_num; i++) {
                        ret = bm_image_create(handle_.data(), output_height, output_width, FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE_SIGNED, &output_bgr_planar_image[i], &output_width);
                        if(BM_SUCCESS != ret) {
                            SPDLOG_ERROR("convert_to error: call bm_image_create failed!");
                            for(int i = 0; i < input_num; i++) {
                                bm_image_destroy(input_bgr_planar_image[i]);
                            }
                            return ret;
                        }
                    }

                    ret = bmcv_image_convert_to(handle_.data(), input_num, attr, input_bgr_planar_image, output_bgr_planar_image);
                    if(BM_SUCCESS != ret) {
                        SPDLOG_ERROR("convert_to error: call bmcv_image_convert_to failed!");
                        for(int i = 0; i < input_num; i++) {
                            bm_image_destroy(input_bgr_planar_image[i]);
                            bm_image_destroy(output_bgr_planar_image[i]);
                        }
                        return ret;
                    }

                    for(int i = 0; i < input_num; i++) {
                        output[i].data_type = DATA_TYPE_EXT_1N_BYTE;
                        output_bgr_planar_image[i].data_type = DATA_TYPE_EXT_1N_BYTE;
                    }

                    ret = bmcv_image_storage_convert(handle_.data(), input_num, output_bgr_planar_image, output);
                    if(BM_SUCCESS != ret) {
                        SPDLOG_ERROR("convert_to error: call bmcv_image_storage_convert failed!");
                        for(int i = 0; i < input_num; i++) {
                            bm_image_destroy(input_bgr_planar_image[i]);
                            bm_image_destroy(output_bgr_planar_image[i]);
                        }                    
                        return ret;
                    }

                    for(int i = 0; i < input_num; i++) {
                        output[i].data_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
                        output_bgr_planar_image[i].data_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
                    }

                    for(int i = 0; i < input_num; i++) {
                        bm_image_destroy(input_bgr_planar_image[i]);
                        bm_image_destroy(output_bgr_planar_image[i]);
                    }

                    delete[] input_bgr_planar_image;
                    delete[] output_bgr_planar_image;
                }
                else {
                    bm_image* input_bgr_planar_image = new bm_image[input_num];
                    for(int i = 0; i < input_num; i++) {
                        ret = bm_image_create(handle_.data(), input_height, input_width, FORMAT_BGR_PLANAR, input_data_type, &input_bgr_planar_image[i]);
                        if(BM_SUCCESS != ret) {
                            SPDLOG_ERROR("convert_to error: call bm_image_create failed!");
                            return ret;
                        }
                    }

                    ret = bmcv_image_storage_convert(handle_.data(), input_num, input, input_bgr_planar_image);
                    if(BM_SUCCESS != ret) {
                        SPDLOG_ERROR("convert_to error: call bmcv_image_storage_convert failed!");
                        for(int i = 0; i < input_num; i++) {
                            bm_image_destroy(input_bgr_planar_image[i]);
                        }
                        return ret;
                    }

                    bm_image* output_bgr_planar_image = new bm_image[input_num];
                    for(int i = 0; i < input_num; i++) {
                        ret = bm_image_create(handle_.data(), output_height, output_width, FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE, &output_bgr_planar_image[i], &output_width);
                        if(BM_SUCCESS != ret) {
                            SPDLOG_ERROR("convert_to error: call bm_image_create failed!");
                            for(int i = 0; i < input_num; i++) {
                                bm_image_destroy(input_bgr_planar_image[i]);
                            }
                            return ret;
                        }
                    }

                    ret = bmcv_image_convert_to(handle_.data(), input_num, attr, input_bgr_planar_image, output_bgr_planar_image);
                    if(BM_SUCCESS != ret) {
                        SPDLOG_ERROR("convert_to error: call bmcv_image_convert_to failed!");
                        for(int i = 0; i < input_num; i++) {
                            bm_image_destroy(input_bgr_planar_image[i]);
                            bm_image_destroy(output_bgr_planar_image[i]);
                        }
                        return ret;
                    }

                    ret = bmcv_image_storage_convert(handle_.data(), input_num, output_bgr_planar_image, output);
                    if(BM_SUCCESS != ret) {
                        SPDLOG_ERROR("convert_to error: call bmcv_image_storage_convert failed!");
                        for(int i = 0; i < input_num; i++) {
                            bm_image_destroy(input_bgr_planar_image[i]);
                            bm_image_destroy(output_bgr_planar_image[i]);
                        }                    
                        return ret;
                    }

                    for(int i = 0; i < input_num; i++) {
                        bm_image_destroy(input_bgr_planar_image[i]);
                        bm_image_destroy(output_bgr_planar_image[i]);
                    }    

                    delete[] input_bgr_planar_image;
                    delete[] output_bgr_planar_image;
                }
            }
            else {
                printf("convert_to error: not support FORMAT_BGR_PACKED now!\n");
                return BM_ERR_FAILURE;
            }
        }
        else {
            ret = bmcv_image_convert_to(handle_.data(), input_num, attr, input, output);
            if (ret != BM_SUCCESS) {
                printf("bmcv_image_convert_to error, src.format=%d, dst.format=%d", input->image_format, output->image_format);
            }
        }
        return BM_SUCCESS;
    }

    int Bmcv::convert_to(
            BMImage &input,
            BMImage &output,
            const std::tuple<
                    std::pair<float, float>,
                    std::pair<float, float>,
                    std::pair<float, float>> &alpha_beta
    ) {
        if (output.is_created()) {
            output.reset(input.width(), input.height());
        }

        if (!output.is_created()) {
            output.create(
                    handle_,
                    input.height(),
                    input.width(),
                    input.format(), // force to this format
                    input.dtype()
            );
            output.allocate();
        }
        return convert_to(&input.data(), &output.data(),alpha_beta,1);
    }

    BMImage Bmcv::convert_to(
            BMImage &input,
            const std::tuple<
                    std::pair<float, float>,
                    std::pair<float, float>,
                    std::pair<float, float>> &alpha_beta
    ) {
        BMImage output;
        int ret = convert_to(input, output, alpha_beta);
        SAIL_CHECK_RET(ret);
        return std::move(output);
    }

    int Bmcv::yuv2bgr(
            BMImage &input,
            BMImage &output
    ) {
#if (BMCV_VERSION_MAJOR == 2) && !defined(BMCV_VERSION_MINOR)
        return vpp_convert_format(input,output);
#else
        if (output.is_created()) {
            output.reset(input.width(), input.height());
        }
        if (!output.is_created()) {
            int stride = FFALIGN(input.width(), SAIL_ALIGN); // ceiling to 64 * N
            output.create(
                    handle_,
                    input.height(),
                    input.width(),
                    FORMAT_BGR_PLANAR, // force to this format
                    input.dtype(),
                    &stride
            );
            output.allocate();
        }

        int ret = bmcv_image_yuv2bgr_ext(handle_.data(), 1, &input.data(), &output.data());
        if (ret) {
            SPDLOG_ERROR("yuv2bgr failed, bmcv_image_yuv2bgr_ext() ret {}.", ret);
        }
        return ret;
#endif
    }

    BMImage Bmcv::yuv2bgr(BMImage &input) {
        BMImage output;
        int ret = yuv2bgr(input, output);
        SAIL_CHECK_RET(ret);
        return std::move(output);
    }

    int Bmcv::vpp_convert_format(
            BMImage &input,
            BMImage &output
    ) {
        if (output.is_created()) {
            if(output.width() != input.width() || output.height() != input.height()) {
                SPDLOG_INFO("output will be reset to {}x{}",input.width(), input.height());
            }
            output.reset(input.width(), input.height());
        }

        if (!output.is_created()) {
            /* vpp limitation: 64-aligned */
            int dtype_size = bm_image_data_type_size(input.dtype());
            int stride = FFALIGN(input.width() * dtype_size, SAIL_ALIGN); // ceiling to 64 * N

            output.create(
                    handle_,
                    input.height(),
                    input.width(),
                    FORMAT_BGR_PLANAR, // force to this format
                    input.dtype(),
                    &stride
            );
            output.allocate();
        }

        int ret = bmcv_image_vpp_convert(
                handle_.data(),
                1,
                input.data(),
                &output.data()
        );
        if (ret) {
            SPDLOG_ERROR("vpp_convert_format failed, bmcv_image_vpp_convert() ret {}.", ret);
            return SAIL_ERR_BMI_BMCV;
        }

        return ret;
    }

    BMImage Bmcv::vpp_convert_format(BMImage &input, bm_image_format_ext image_format) {
        BMImage output = sail::BMImage(handle_, input.height(), input.width(), image_format, input.dtype());
        int ret = vpp_convert_format(input, output);
        SAIL_CHECK_RET(ret);
        return output;
    }

    int Bmcv::convert_format(
            BMImage &input,
            BMImage &output
    ) {
        if (output.is_created()) {
            if(output.width() != input.width() || output.height() != input.height()) {
                SPDLOG_INFO("output will be reset to {}x{}",input.width(), input.height());
            }
            output.reset(input.width(), input.height());
        }

        if (!output.is_created()) {
            output.create(
                    handle_,
                    input.height(),
                    input.width(),
                    FORMAT_BGR_PLANAR, // force to this format
                    input.dtype()
            );
            output.allocate();
        }

        int ret = BM_SUCCESS;

        if(output.format() == FORMAT_GRAY) {
            // If it is 1684x, call the bmcv_image_storage_convert interface directly for conversion
            if(handle_.get_target() == "1684x") {
                ret = bmcv_image_storage_convert(handle_.data(), 1, &input.data(), &output.data());
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("convert_to_gray error: failed to call bmcv_image_storage_convert!");
                    print_image(input.data(), "src: ");
                    print_image(output.data(), "dst: ");
                    return ret;
                }
                return BM_SUCCESS;
            }

            // If it is 1684, then copy the Y channel of the image to construct a grayscale map
            if(input.format() != FORMAT_YUV420P) {
                BMImage input_ = vpp_convert_format(input, FORMAT_YUV420P);
                ret = convert_yuv420p_to_gray(input_, output);
                if(BM_SUCCESS != ret) {
                    SPDLOG_ERROR("convert_to_gray error: failed to call convert_yuv420p_to_gray!");
                    print_image(input.data(), "src: ");
                    print_image(output.data(), "dst: ");
                    return ret;
                }
            }
            else {
                ret = convert_yuv420p_to_gray(input, output);
                if(BM_SUCCESS != ret) {
                    SPDLOG_ERROR("convert_to_gray error: failed to call convert_yuv420p_to_gray!");
                    print_image(input.data(), "src: ");
                    print_image(output.data(), "dst: ");
                    return ret;
                }
            }
        }
        else {
            ret = bmcv_image_storage_convert(
                    handle_.data(),
                    1,
                    &input.data(),
                    &output.data()
            );
            if (ret != BM_SUCCESS) {
                SPDLOG_ERROR("bmcv_image_storage_convert Failed: {}",ret);
                return ret;
            }
        }
        return BM_SUCCESS;
    }

    BMImage Bmcv::convert_format(BMImage &input, bm_image_format_ext image_format) {
        BMImage output = sail::BMImage(handle_, input.height(), input.width(), image_format, input.dtype());
        if(BM_SUCCESS != convert_format(input, output)) {
            SPDLOG_ERROR("convert_format failed!");
            throw SailBMImageError("bmcv api fail");
        }

        return output;
    }

    int Bmcv::rectangle(
            const BMImage &input,
            int x0,
            int y0,
            int w,
            int h,
            const std::tuple<int, int, int> &color,
            int thickness
    ) {
        int left = x0 > 0 ? x0 : 0;
        int top = y0 > 0 ? y0 : 0;
        int right = x0 + w;
        int bottom = y0 + h;
        right = right > 0 ? right : 0;
        bottom = bottom > 0 ? bottom : 0;
        left = left < input.width()-1 ? left : input.width()-1;
        top = top < input.height()-1 ? top : input.height()-1;
        right = right < input.width()-1 ? right : input.width()-1;
        bottom = bottom < input.height()-1 ? bottom : input.height()-1;
        bmcv_rect rect = {left, top, right-left, bottom-top};
        int ret = bmcv_image_draw_rectangle(
                handle_.data(),
                input.data(),
                1,
                &rect,
                thickness,
                std::get<2>(color),  // R
                std::get<1>(color),  // G
                std::get<0>(color)); // B
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("bmcv_image_draw_rectangle() err={}", ret);
            return SAIL_ERR_BMI_BMCV;
        }
        return BM_SUCCESS;
    }

    int Bmcv::rectangle(
            const bm_image &input,
            int x0,
            int y0,
            int w,
            int h,
            const std::tuple<int, int, int> &color,
            int thickness
    ) {
        return rectangle_(input, x0, y0, w, h, color, thickness);
    }

    int Bmcv::rectangle_(
            const bm_image &input,
            int x0,
            int y0,
            int w,
            int h,
            const std::tuple<int, int, int> &color,
            int thickness
    ) {
        int left = x0 > 0 ? x0 : 0;
        int top = y0 > 0 ? y0 : 0;
        int right = x0 + w;
        int bottom = y0 + h;
        right = right > 0 ? right : 0;
        bottom = bottom > 0 ? bottom : 0;
        left = left < input.width-1 ? left : input.width-1;
        top = top < input.height-1 ? top : input.height-1;
        right = right < input.width-1 ? right : input.width-1;
        bottom = bottom < input.height-1 ? bottom : input.height-1;
        bmcv_rect rect = {left, top, right-left, bottom-top};
        int ret = bmcv_image_draw_rectangle(
                handle_.data(),
                input,
                1,
                &rect,
                thickness,
                std::get<2>(color),  // R
                std::get<1>(color),  // G
                std::get<0>(color)); // B
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("bmcv_image_draw_rectangle() err={}", ret);
            return ret;
        }
        return BM_SUCCESS;
    }
    int Bmcv::fillRectangle(
            const BMImage &input,
            int x0,
            int y0,
            int w,
            int h,
            const std::tuple<int, int, int> &color
    ) {
        int left = x0 > 0 ? x0 : 0;
        int top = y0 > 0 ? y0 : 0;
        int right = x0 + w;
        int bottom = y0 + h;
        right = right > 0 ? right : 0;
        bottom = bottom > 0 ? bottom : 0;
        left = left < input.width()-1 ? left : input.width()-1;
        top = top < input.height()-1 ? top : input.height()-1;
        right = right < input.width()-1 ? right : input.width()-1;
        bottom = bottom < input.height()-1 ? bottom : input.height()-1;
        bmcv_rect rect = {left, top, right-left, bottom-top};
        int ret = bmcv_image_fill_rectangle(
                handle_.data(),
                input.data(),
                1,
                &rect,
                std::get<2>(color),  // R
                std::get<1>(color),  // G
                std::get<0>(color)); // B
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("bmcv_image_fill_rectangle() err={}", ret);
            return SAIL_ERR_BMI_BMCV;
        }
        return BM_SUCCESS;
    }

    int Bmcv::fillRectangle(
            const bm_image &input,
            int x0,
            int y0,
            int w,
            int h,
            const std::tuple<int, int, int> &color
    ) {
        return fillRectangle_(input, x0, y0, w, h, color);
    }

    int Bmcv::fillRectangle_(
            const bm_image &input,
            int x0,
            int y0,
            int w,
            int h,
            const std::tuple<int, int, int> &color
    ) {
        int left = x0 > 0 ? x0 : 0;
        int top = y0 > 0 ? y0 : 0;
        int right = x0 + w;
        int bottom = y0 + h;
        right = right > 0 ? right : 0;
        bottom = bottom > 0 ? bottom : 0;
        left = left < input.width-1 ? left : input.width-1;
        top = top < input.height-1 ? top : input.height-1;
        right = right < input.width-1 ? right : input.width-1;
        bottom = bottom < input.height-1 ? bottom : input.height-1;
        bmcv_rect rect = {left, top, right-left, bottom-top};
        int ret = bmcv_image_fill_rectangle(
                handle_.data(),
                input,
                1,
                &rect,
                std::get<2>(color),  // R
                std::get<1>(color),  // G
                std::get<0>(color)); // B
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("bmcv_image_draw_rectangle() err={}", ret);
            return ret;
        }
        return BM_SUCCESS;
    }
    int Bmcv::putText(
        const BMImage                   &image,
        const std::string               &text,
        int                             x,
        int                             y,
        const std::tuple<int, int, int> &color, // BGR
        float                           fontScale,
        int                             thickness
    ){
        if(image.format() != FORMAT_GRAY &&
            image.format() != FORMAT_YUV420P &&
            image.format() != FORMAT_YUV422P &&
            image.format() != FORMAT_NV12 &&
            image.format() != FORMAT_NV21 &&
            image.format() != FORMAT_NV16 &&
            image.format() != FORMAT_NV61){
            SPDLOG_ERROR("input BMImage's pixel format is not supported!\n"
                         "Support pixel formats for putText: FORMAT_GRAY, "
                         "FORMAT_YUV420P, FORMAT_YUV422P, FORMAT_YUV444P, "
                         "FORMAT_NV12, FORMAT_NV21, FORMAT_NV16, FORMAT_NV61");
            print_image(image.data(),"input");
            return SAIL_ERR_BMI_NOTSUP;
        }
        bmcv_point_t org = {x,y};
        bmcv_color_t color_put = {std::get<2>(color), std::get<1>(color), std::get<0>(color)};
        int ret = bmcv_image_put_text(
            handle_.data(), 
            image.data(), 
            text.c_str(),
            org,
            color_put,
            fontScale,
            thickness);
         if (BM_SUCCESS != ret) {
            print_image(image.data(),"input");
            SPDLOG_ERROR("bmcv_image_put_text() err={}", ret);
            return SAIL_ERR_BMI_BMCV;
        }
        return BM_SUCCESS;
    }

    int Bmcv::putText(
        const bm_image                  &image,
        const std::string               &text,
        int                             x,
        int                             y,
        const std::tuple<int, int, int> &color, // BGR
        float                           fontScale,
        int                             thickness
    ){
        return putText_(image,text,x,y,color,fontScale,thickness);
    }

    int Bmcv::putText_(
        const bm_image                  &image,
        const std::string               &text,
        int                             x,
        int                             y,
        const std::tuple<int, int, int> &color, // BGR
        float                           fontScale,
        int                             thickness
    ){
        if(image.image_format != FORMAT_GRAY &&
            image.image_format != FORMAT_YUV420P &&
            image.image_format != FORMAT_YUV422P &&
            image.image_format != FORMAT_NV12 &&
            image.image_format != FORMAT_NV21 &&
            image.image_format != FORMAT_NV16 &&
            image.image_format != FORMAT_NV61){
            SPDLOG_ERROR("input BMImage's pixel format is not supported!\n"
                         "Support pixel formats for putText: FORMAT_GRAY, "
                         "FORMAT_YUV420P, FORMAT_YUV422P, FORMAT_YUV444P, "
                         "FORMAT_NV12, FORMAT_NV21, FORMAT_NV16, FORMAT_NV61");
            print_image(image,"input");
            return SAIL_ERR_BMI_NOTSUP;
        }
        bmcv_point_t org = {x,y};
        bmcv_color_t color_put = {std::get<2>(color), std::get<1>(color), std::get<0>(color)};
        int ret = bmcv_image_put_text(
            handle_.data(), 
            image, 
            text.c_str(),
            org,
            color_put,
            fontScale,
            thickness);
         if (BM_SUCCESS != ret) {
            print_image(image,"input");
            SPDLOG_ERROR("bmcv_image_put_text() err={}", ret);
            return SAIL_ERR_BMI_BMCV;
        }
        return BM_SUCCESS;
    }
    int Bmcv::image_add_weighted(
        BMImage           &input1,
        float             alpha,
        BMImage           &input2,
        float             beta,
        float             gamma,
        BMImage           &output){
        if (!input1.is_created()){
            SPDLOG_ERROR("input1 must be created before!");
            return BM_ERR_FAILURE;
        }
        if (!input2.is_created()){
            SPDLOG_ERROR("input2 must be created before!");
            return BM_ERR_FAILURE;
        }
        if (input1.dtype() != DATA_TYPE_EXT_1N_BYTE || input2.dtype() != DATA_TYPE_EXT_1N_BYTE){
            SPDLOG_ERROR("Input Dtype must be DATA_TYPE_EXT_1N_BYTE!");
            return BM_ERR_FAILURE;
        }
        if (input1.width() != input2.width() || input1.height() != input2.height() || input1.format() != input2.format()){
            SPDLOG_ERROR("The width, height and format of input2 must be consistent with that of input1!");
            return BM_ERR_FAILURE;
        }

        if (output.is_created()) {
            if (input1.width() != output.width() || input1.height() != output.height()){
                SPDLOG_ERROR("The width, height of output must be consistent with that of input1!");
                return BM_ERR_FAILURE;
            }
            if (output.format() != FORMAT_BGR_PLANAR || output.format() != FORMAT_RGB_PLANAR){
                SPDLOG_ERROR("The output format must be FORMAT_BGR_PLANAR!");
                return BM_ERR_FAILURE;
            }
            if (output.format() != input1.format())  {
                SPDLOG_ERROR("The output format must same as input1.format!");
                return BM_ERR_FAILURE;
            }
        }
        bm_image_format_ext temp_format = input1.format();
        bool convert_flag = false;
        if(temp_format != FORMAT_BGR_PLANAR && temp_format != FORMAT_RGB_PLANAR){
            temp_format = FORMAT_BGR_PLANAR;
            convert_flag = true;
        }
        if(!output.is_created()){ 
            int dtype_size = bm_image_data_type_size(input1.dtype());
            int stride = FFALIGN(input1.width() * dtype_size, SAIL_ALIGN); // ceiling to 64 * N

            output.create(
                    handle_,
                    input1.height(),
                    input1.width(),
                    temp_format, // force to this format
                    input1.dtype(),
                    &stride
            );
        }
        int ret = BM_SUCCESS;
        if(convert_flag){
            BMImage input1_temp = convert_format(input1);
            BMImage input2_temp = convert_format(input2);
            ret = bmcv_image_add_weighted(handle_.data(), input1_temp.data(), alpha, input2_temp.data(), beta, gamma, output.data());
        }else{
            ret = bmcv_image_add_weighted(handle_.data(), input1.data(), alpha, input2.data(), beta, gamma, output.data());
        }
        if (BM_SUCCESS != ret){
            SPDLOG_ERROR("bmcv_image_add_weighted err={}", ret);
            return SAIL_ERR_BMI_BMCV;
        }
        return ret;
    }

    BMImage Bmcv::image_add_weighted(
      BMImage           &input1,
      float             alpha,
      BMImage           &input2,
      float             beta,
      float             gamma){
        BMImage output;
        int ret = image_add_weighted(input1,alpha,input2,beta, gamma,output);
        SAIL_CHECK_RET(ret);
        return std::move(output);
    }
    
    int Bmcv::imwrite(
            const std::string &filename,
            const BMImage &input
    ) {
        int ret;
 #if defined USE_OPENCV && defined USE_BMCV
        // bm_image_write_to_bmp(input.data(), "./imwrite.jpg");
        cv::Mat cv_img;
        bm_image input_image = input.data();
        bm_image bgr_image;
        ret = bm_image_create(handle_.data(), input_image.height, input_image.width, FORMAT_BGR_PLANAR, input_image.data_type, &bgr_image);
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("imwrite error: call bm_image_create failed!");
            return ret;
        }

        ret = bmcv_image_storage_convert(handle_.data(), 1, &input_image, &bgr_image);
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("imwrite error: call bmcv_image_storage_convert failed!");
            return ret;
        }

        ret = cv::bmcv::toMAT((bm_image *) &bgr_image, cv_img, true);
        if (ret != 0) {
            SPDLOG_ERROR("cv::bmcv::toMat() err={}, filename={}", ret, filename);
            return ret;
        }

        if (!cv::imwrite(filename, cv_img)) {
            SPDLOG_ERROR("cv::imwrite failed");
            bm_image_destroy(bgr_image);
            return BM_ERR_FAILURE;
        }

        ret = bm_image_destroy(bgr_image);
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("imwrite error: call bm_image_destroy failed!");
            return ret;
        }
#else
        ret = bm_image_write_to_bmp(input.data(), filename.c_str());
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("bm_image_write_to_bmp() err={}", ret);
            return ret;
        }
#endif
        return BM_SUCCESS;
    }

    int Bmcv::imwrite(
            const std::string &filename,
            const bm_image &input
    ) {
        return imwrite_(filename, input);
    }

    int Bmcv::imwrite_(
            const std::string &filename,
            const bm_image &input
    ) {
#if defined USE_OPENCV && defined USE_BMCV
        // bm_image_write_to_bmp(input, "./imwrite_.jpg");
        int ret;
        cv::Mat cv_img;
        bm_image bgr_image;
        ret = bm_image_create(handle_.data(), input.height, input.width, FORMAT_BGR_PLANAR, input.data_type, &bgr_image);
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("imwrite error: call bm_image_create failed!");
            return ret;
        }

        ret = bmcv_image_storage_convert(handle_.data(), 1, &const_cast<bm_image&>(input), &bgr_image);
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("imwrite error: call bmcv_image_storage_convert failed!");
            return ret;
        }

        ret = cv::bmcv::toMAT((bm_image *) &bgr_image, cv_img, true);
        if (ret != 0) {
            SPDLOG_ERROR("cv::bmcv::toMat() err={}, filename={}", ret, filename);
            return ret;
        }

        if (!cv::imwrite(filename, cv_img)) {
            SPDLOG_ERROR("cv::imwrite failed");
            bm_image_destroy(bgr_image);
            return BM_ERR_FAILURE;
        }

        ret = bm_image_destroy(bgr_image);
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("imwrite error: call bm_image_destroy failed!");
            return ret;
        }
#else
        ret = bm_image_write_to_bmp(input, filename.c_str());
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("imwrite error: call bm_image_write_to_bmp failed!");
            return ret;
        }
#endif
        return ret;
    }


    Handle Bmcv::get_handle() {
        return handle_;
    }

    bm_data_type_t
    Bmcv::get_bm_data_type(
            bm_image_data_format_ext fmt
    ) {
        return get_bm_data_type_sail(fmt);
    }

    bm_image_data_format_ext
    Bmcv::get_bm_image_data_format(
            bm_data_type_t dtype
    ) {
        return get_bm_image_data_format_sail(dtype);
    }
#endif //USE_BMCV

    bm_image Bmcv::crop_and_resize_padding(
      bm_image &input,
      int crop_x0,
      int crop_y0,
      int crop_w,
      int crop_h,
      int resize_w,
      int resize_h,
      PaddingAtrr &padding_in,
      bmcv_resize_algorithm resize_alg
    ){
        bm_image_format_ext image_format = FORMAT_BGR_PLANAR;
        if (input.image_format == FORMAT_RGB_PLANAR){
            image_format = FORMAT_RGB_PLANAR;
        }
        bm_image bm_image_result;
        int ret = bm_image_create(handle_.data(),
            resize_h, resize_w, 
            image_format, DATA_TYPE_EXT_1N_BYTE,
            &bm_image_result);

        float scale_w = (float)padding_in.dst_crop_w/crop_w;
        float scale_h = (float)padding_in.dst_crop_h/crop_h;
        int temp_image_w = padding_in.dst_crop_w;
        int temp_image_h = padding_in.dst_crop_h;
        if(scale_w < scale_h) temp_image_h = crop_h*scale_w;
        else temp_image_w = crop_w*scale_h;
        bm_image bm_image_temp;
        ret = bm_image_create(
            handle_.data(),
            temp_image_h, temp_image_w, 
            image_format, DATA_TYPE_EXT_1N_BYTE,
            &bm_image_temp);
        if (BM_SUCCESS != ret){
            SPDLOG_ERROR("bm_image_create err={}", ret);
            bm_image_destroy(bm_image_temp);
            return bm_image_result;
        }
        bmcv_resize_t attr_rt;
        attr_rt.start_x = crop_x0;
        attr_rt.start_y = crop_y0;
        attr_rt.in_width = crop_w;
        attr_rt.in_height = crop_h;
        attr_rt.out_width = temp_image_w;
        attr_rt.out_height = temp_image_h;
        
        bmcv_resize_image attr_ri;
        attr_ri.resize_img_attr = &attr_rt;
        attr_ri.roi_num = 1;
        attr_ri.stretch_fit = 1;
        attr_ri.interpolation = resize_alg;

        spdlog::debug("crop_and_resize_padding attr_rt {} {} {} {} {} {}",
                     attr_rt.start_x, attr_rt.start_y, 
                     attr_rt.in_width, attr_rt.in_height,
                     attr_rt.out_width, attr_rt.out_height);
        spdlog::debug("crop_and_resize_padding resize_alg {}", resize_alg);

        ret = bmcv_image_resize(
            handle_.data(),
            1,
            &attr_ri,
            &input,
            &bm_image_temp);
        if (BM_SUCCESS != ret){
            SPDLOG_ERROR("bmcv_image_resize err={}", ret);
            throw SailBMImageError("bmcv api fail");
        }else{
            bmcv_copy_to_atrr_t copy_att;
            copy_att.start_x = padding_in.dst_crop_stx;
            copy_att.start_y = padding_in.dst_crop_sty;
            copy_att.padding_r = padding_in.padding_r;
            copy_att.padding_g = padding_in.padding_g;
            copy_att.padding_b = padding_in.padding_b;
            copy_att.if_padding = 1;
            ret = bmcv_image_copy_to(
                handle_.data(),
                copy_att,
                bm_image_temp,
                bm_image_result);
            if (BM_SUCCESS != ret){
                SPDLOG_ERROR("bmcv_image_resize err={}", ret);
                throw SailBMImageError("bmcv api fail");
            }
        }

        bm_image_destroy(bm_image_temp);
        return bm_image_result;
    }

    BMImage Bmcv::crop_and_resize_padding(
      BMImage &input,
      int crop_x0,
      int crop_y0,
      int crop_w,
      int crop_h,
      int resize_w,
      int resize_h,
      PaddingAtrr &padding_in,
      bmcv_resize_algorithm resize_alg
    ){
        spdlog::debug("crop_and_resize_padding resize_alg {}",resize_alg);
        bm_image bm_image_result = crop_and_resize_padding(
            input.data(),
            crop_x0, 
            crop_y0, 
            crop_w, 
            crop_h,
            resize_w, 
            resize_h, 
            padding_in,
            resize_alg);

        BMImage temp_img;
        temp_img = std::move(bm_image_result);
        return temp_img;
    }

    int Bmcv::image_copy_to(bm_image input, bm_image output, int start_x, int start_y)
    {
        if(input.width + start_x > output.width ){
            SPDLOG_ERROR("Input width add start_x must less than output width!");
            return 1;
        }
        if(input.height + start_y > output.height){
            SPDLOG_ERROR("Input height add start_y must less than output width!");
            return 1;
        }

        bmcv_copy_to_atrr_t copy_to_attr;
        copy_to_attr.start_x = start_x;
        copy_to_attr.start_y = start_y;
        copy_to_attr.if_padding = false;
        bm_status_t ret = bmcv_image_copy_to(handle_.data(),copy_to_attr,input,output);
        if(BM_SUCCESS != ret){
            SPDLOG_ERROR("bmcv_image_copy_to err {}!",ret);
            return ret;
        }
        return BM_SUCCESS;
    }

    int Bmcv::image_copy_to(BMImage &input, BMImage &output, int start_x, int start_y)
    {
        if(output.is_created()){
            if (input.format() != output.format()){
                SPDLOG_ERROR("Output Format must same as input!");
                throw SailBMImageError("mismatch BMImage");
            }
        }else{
            SPDLOG_ERROR("Output has not created!");
            throw SailBMImageError("invalid argument");
        }
        int ret = image_copy_to(input.data(),output.data(),start_x,start_y);
        if(ret != BM_SUCCESS){
            throw SailBMImageError("bmcv api fail");
        }
        return ret;
    }

    int Bmcv::image_copy_to_padding(bm_image input, bm_image output,
        unsigned int padding_r, unsigned int padding_g, unsigned int padding_b,
        int start_x, int start_y)
    {
        if(input.width + start_x > output.width ){
            SPDLOG_ERROR("Input width add start_x must less than output width!");
            return SAIL_ERR_BMI_PARAM;
        }
        if(input.height + start_y > output.height){
            SPDLOG_ERROR("Input height add start_y must less than output width!");
            return SAIL_ERR_BMI_PARAM;
        }

        bmcv_copy_to_atrr_t copy_to_attr;
        copy_to_attr.start_x = start_x;
        copy_to_attr.start_y = start_y;
        copy_to_attr.padding_r = padding_r;
        copy_to_attr.padding_g = padding_g;
        copy_to_attr.padding_b = padding_b;
        copy_to_attr.if_padding = true;
        bm_status_t ret = bmcv_image_copy_to(handle_.data(),copy_to_attr,input,output);
        if(BM_SUCCESS != ret){
            SPDLOG_ERROR("bmcv_image_copy_to err {}!",ret);
            return ret;
        }
        return BM_SUCCESS;
    }

    int Bmcv::image_copy_to_padding(BMImage &input, BMImage &output,
        unsigned int padding_r, unsigned int padding_g, unsigned int padding_b,
        int start_x, int start_y)
    {
        if(output.is_created()){
            if (input.format() != output.format()){
                SPDLOG_ERROR("Output Format must same as input!");
                throw SailBMImageError("mismatch BMImage");
            }
        }else{
            SPDLOG_ERROR("Output has not created!");
            throw SailBMImageError("invalid argument");
        }
        int ret = image_copy_to_padding(input.data(),output.data(),padding_r,padding_g,padding_b,start_x,start_y);
        if(ret != BM_SUCCESS){
            throw SailBMImageError("bmcv api fail");
        }
        return ret;      
    }

    nms_proposal_t* Bmcv::nms(face_rect_t *input_proposal, int proposal_size, float threshold)
    {
        nms_proposal_t *output_proposal = new nms_proposal_t;
        bmcv_nms(handle_.data(),
            bm_mem_from_system(input_proposal),
            proposal_size,
            threshold,
            bm_mem_from_system(output_proposal));
        return output_proposal;
    }
#if BMCV_VERSION_MAJOR > 1
    static int bm_dmem_read_bin(bm_handle_t handle, bm_device_mem_t* dmem, const char *input_name, unsigned int size){
        if (access(input_name, F_OK) != 0 || strlen(input_name) == 0 || 0 >= size){
            SPDLOG_ERROR("file is not exist or wrong size");
            return BM_ERR_FAILURE;
        }

        char input_ptr[size];
        FILE *fp_src = fopen(input_name, "rb+");

        if (fread(input_ptr, 1, size, fp_src) < (unsigned int)size){
            SPDLOG_ERROR("file size is less than {} required bytes", size);
            fclose(fp_src);
            return BM_ERR_FAILURE;
        };
        fclose(fp_src);

        if (BM_SUCCESS != bm_malloc_device_byte(handle, dmem, size)){
            SPDLOG_ERROR("bm_malloc_device_byte failed");
            return BM_ERR_FAILURE;
        }

        if (BM_SUCCESS != bm_memcpy_s2d(handle, *dmem, input_ptr)){
            SPDLOG_ERROR("bm_memcpy_s2d failed");
            return BM_ERR_FAILURE;
        }

        return BM_SUCCESS;
    }

    /**
    * @brief A class for image stitch.
    */
    class Blend::Blend_CC{
    public:
        /**
         * @brief blend init 
         * @param src_h       image height
         * @param ovlap_attr  overlapping areas width
         * @param bd_attr     Black border of images
         * @param wgt_phy_mem imgage weight
         * @param wgt_mode    weight mode
         */
        explicit Blend_CC(int src_h, std::vector<std::vector<short>> ovlap_attr, std::vector<std::vector<short>> bd_attr, std::vector<std::vector<string>> wgt_phy_mem,bm_stitch_wgt_mode wgt_mode);
        ~Blend_CC();

        /**
         * @brief blend process
         * @param input   input images
         * @param output  output image
         * @return 0 for success and other for failure
         */
        int process(std::vector<BMImage*> &input, BMImage &output);
        
        BMImage process(std::vector<BMImage*> &input);
        
        struct stitch_param blend_config;
        int ovlap_width;
        Handle handle_;
    };

    int Blend::Blend_CC::process(std::vector<BMImage*> &input, BMImage &output){
        //the num of input images
        int input_num = input.size();
        //input for bmcv_blending 
        bm_image blend_img[input_num];
        //if need convert
        bool need_convert[input_num];
        //output width 
        int output_width = 0;
        
        int ret;

        if(input_num < 2 || input_num > 4){
            SPDLOG_ERROR("The number of input images must be between 2 and 4");
            return BM_ERR_FAILURE;
        }
        
        //dst_format
        auto dst_format = input[0] -> format();
        if(!(   dst_format == FORMAT_RGBP_SEPARATE ||
                dst_format == FORMAT_GRAY ||
                dst_format == FORMAT_BGRP_SEPARATE ||
                dst_format == FORMAT_YUV420P ||
                dst_format == FORMAT_YUV422P ||
                dst_format == FORMAT_YUV444P ||
                dst_format == FORMAT_GRAY ))    dst_format = FORMAT_YUV420P;
        
        //convert format 
        for(int i = 0; i < input_num; i++){
            if(input[i] -> format() != dst_format){
                ret = bm_image_create(handle_.data(), input[i] -> height(), input[i] -> width(), dst_format, DATA_TYPE_EXT_1N_BYTE, &blend_img[i], NULL);
                if(ret != BM_SUCCESS) {
                    SPDLOG_ERROR("bm_image_create error");
                    goto fail;
                }

                ret = bm_image_alloc_dev_mem(blend_img[i], 1);
                if(ret != BM_SUCCESS) {
                    SPDLOG_ERROR("bm_image_alloc_dev_mem error");
                    goto fail;
                }

                ret = bmcv_image_vpp_convert(handle_.data(), 1, input[i] -> data(), &blend_img[i]);
                if(ret != BM_SUCCESS) {
                    SPDLOG_ERROR("bmcv_image_vpp_convert error");
                    goto fail;
                }

                need_convert[i] = true;
            }else{
                blend_img[i] = input[i] -> data();
                need_convert[i] = false;
            }
        }

        //create output BMImage
        for (int i = 0; i < input_num; i++){
            output_width += input[i] -> width();
        }
        output_width -= ovlap_width;
        // align 32
        output_width = (output_width + 32 -1) & ~(32 - 1);
        if (output.is_created()) {
            if (output_width != output.width() || input[0] -> height() != output.height()){
                output.reset(output_width, input[0] -> height());
                SPDLOG_INFO("output will be reset to {}x{}",output_width, input[0] -> height());
            }
            if (output.format() != dst_format){
                SPDLOG_ERROR("The output format must be the same as the input format {}!", dst_format);
                ret = BM_ERR_FAILURE;
                goto fail;
            }
        }
        if (!output.is_created()) {
            output.create(
                    handle_,
                    input[0] -> height(),
                    output_width,
                    dst_format, // force to this format
                    DATA_TYPE_EXT_1N_BYTE
            );
            output.allocate();
        }
        //blend work

        ret = bmcv_blending(handle_.data(), input_num, blend_img, output.data(), blend_config);
        if(ret != BM_SUCCESS){
            SPDLOG_ERROR("bmcv_blending error");
            goto fail;
        }
        //destroy 
    fail:
        for(int i = 0; i < input_num; i++) if(need_convert[i]) bm_image_destroy(blend_img[i]);
        return ret;
    }
        

    BMImage Blend::Blend_CC::process(std::vector<BMImage*> &input){
        BMImage output;
        int ret = process(input, output);
        if(ret != BM_SUCCESS){
            SPDLOG_ERROR("blend err={}", ret);
            throw SailRuntimeError("blend process error");
        }
        return std::move(output);
    }

    Blend::Blend_CC::Blend_CC(int src_h, std::vector<std::vector<short>> ovlap_attr, std::vector<std::vector<short>> bd_attr, std::vector<std::vector<string>> wgt_phy_mem,bm_stitch_wgt_mode wgt_mode) : handle_(0) {
        memset(&blend_config, 0, sizeof(blend_config));
        ovlap_width = 0;

        int ovlap_num = ovlap_attr[0].size();
        if(wgt_phy_mem.size() != ovlap_num){
            SPDLOG_ERROR("Wrong config; please check wgt_phy_mem.size(),ensure it's equal to the number of overlapping areas");
            throw SailRuntimeError("blend config error");
        }   
        
        for(int i = 0; i < ovlap_num; i++){
            blend_config.ovlap_attr.ovlp_lx[i] = ovlap_attr[0][i];
            blend_config.ovlap_attr.ovlp_rx[i] = ovlap_attr[1][i];
            int wgtwidth = blend_config.ovlap_attr.ovlp_rx[i] - blend_config.ovlap_attr.ovlp_lx[i] + 1;
            ovlap_width += wgtwidth;
            // align 16
            wgtwidth = (wgtwidth + 16 -1) & ~(16 - 1);
            int wgtheight = src_h;
            int wgt_len = wgtwidth * wgtheight;
            for (int j = 0; j < 2; j++) {
                int ret = bm_dmem_read_bin(handle_.data(), &blend_config.wgt_phy_mem[i][j], wgt_phy_mem[i][j].c_str(), wgt_len);
                if(ret != BM_SUCCESS){
                    SPDLOG_ERROR("bm_dmem_read_bin error");
                    throw SailRuntimeError("bm_dmem_read_bin error");
                }
            }
        }
    }

    Blend::Blend_CC::~Blend_CC() {}

    Blend::Blend(int src_h, std::vector<std::vector<short>> ovlap_attr, std::vector<std::vector<short>> bd_attr, std::vector<std::vector<string>> wgt_phy_mem,bm_stitch_wgt_mode wgt_mode):_impl(new Blend_CC(src_h, ovlap_attr, bd_attr, wgt_phy_mem, wgt_mode)){}

    int Blend::process(std::vector<BMImage*> &input, BMImage &output){
        return _impl -> process(input, output);
    }

    BMImage Blend::process(std::vector<BMImage*> &input){
        return _impl -> process(input);
    }
    
    Blend::~Blend(){
        delete _impl;
    }

    int Bmcv::bmcv_overlay(BMImage& image, std::vector<std::vector<int>> overlay_info, std::vector<const BMImage *> overlay_image){
        if (overlay_info.size() != overlay_image.size()){
            SPDLOG_ERROR("the number of overlay_info is not equal to overlay_image");
            throw SailBMImageError("parameter error");
        }

        int overlay_num = overlay_info.size();
        std::vector<bmcv_rect_t> overlay_info_bmcv;
        std::vector<bm_image> overlay_image_bmcv; 
        for (int i = 0; i < overlay_num; ++i){
            overlay_info_bmcv.emplace_back(bmcv_rect_t{overlay_info[i][0], overlay_info[i][1], overlay_info[i][2], overlay_info[i][3]});
            auto tmp = overlay_image[i]->data();
            if (tmp.image_format != FORMAT_ARGB_PACKED && tmp.image_format != FORMAT_ARGB4444_PACKED && tmp.image_format != FORMAT_ARGB1555_PACKED){
                SPDLOG_ERROR("overlay_image Format Error, unsupport format {}", tmp.image_format);
                throw SailBMImageError("parameter error"); 
            }
            overlay_image_bmcv.emplace_back(tmp);
        }
        auto ret = bmcv_image_overlay(handle_.data(), image.data(), overlay_num, overlay_info_bmcv.data(), overlay_image_bmcv.data());
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("overlay error: call bmcv_image_overlay failed!");
            return ret;
        }
        return ret;
    }
#endif
    BMImage Bmcv::warp_perspective(
        BMImage                     &input,
        const std::tuple<
            std::pair<int,int>,
            std::pair<int,int>,
            std::pair<int,int>,
            std::pair<int,int>>     &coordinate,
        int                         output_width,
        int                         output_height,
        bm_image_format_ext         format,
        bm_image_data_format_ext    dtype,
        int                         use_bilinear)
    {
        if (format != FORMAT_BGR_PLANAR && format != FORMAT_RGB_PLANAR){
            SPDLOG_ERROR("Output Format Error, Only support FORMAT_BGR_PLANAR and FORMAT_RGB_PLANAR!");
            throw SailBMImageError("not supported");
        }
#if (BMCV_VERSION_MAJOR == 2) && !defined(BMCV_VERSION_MINOR)
        if (dtype != DATA_TYPE_EXT_1N_BYTE){
            SPDLOG_ERROR("Output dtype Error, Only support DATA_TYPE_EXT_1N_BYTE!");
            throw SailBMImageError("not supported");
        }
#else
        if (dtype != DATA_TYPE_EXT_1N_BYTE && dtype != DATA_TYPE_EXT_4N_BYTE){
            SPDLOG_ERROR("Output dtype Error, Only support DATA_TYPE_EXT_1N_BYTE and DATA_TYPE_EXT_4N_BYTE!");
            throw SailBMImageError("not supported");
        }
#endif
        BMImage output_image = sail::BMImage(handle_, output_height, output_width, format, dtype);
        bmcv_perspective_image_coordinate coord[4];  
        coord[0].coordinate_num = 1;
        bmcv_perspective_coordinate coordinate_temp;

        coordinate_temp.x[0] = std::get<0>(coordinate).first;
        coordinate_temp.y[0] = std::get<0>(coordinate).second;
        coordinate_temp.x[1] = std::get<1>(coordinate).first;
        coordinate_temp.y[1] = std::get<1>(coordinate).second;
        coordinate_temp.x[2] = std::get<2>(coordinate).first;
        coordinate_temp.y[2] = std::get<2>(coordinate).second;
        coordinate_temp.x[3] = std::get<3>(coordinate).first;
        coordinate_temp.y[3] = std::get<3>(coordinate).second;
        coord[0].coordinate = &coordinate_temp;

        int ret = BM_SUCCESS;
        if(input.format() == format && input.dtype() == dtype){
            ret = bmcv_image_warp_perspective_with_coordinate(
                handle_.data(),1,coord,&input.data(), &output_image.data(), use_bilinear);
            if(ret != BM_SUCCESS){
                SPDLOG_ERROR("bmcv_image_warp_perspective_with_coordinate err={}", ret);
                throw SailBMImageError("bmcv api fail");
            }
        }else{
            BMImage input_convert = sail::BMImage(handle_, input.height(), input.width(), format, dtype);
            if(input.width() % 16 == 0 && input.height() % 2 == 0){
                vpp_convert_format(input,input_convert); 
            }else{
                convert_format(input,input_convert); 
            }
            ret = bmcv_image_warp_perspective_with_coordinate(
                handle_.data(),1,coord,&input_convert.data(), &output_image.data(), use_bilinear);
            if(ret != BM_SUCCESS){
                SPDLOG_ERROR("bmcv_image_warp_perspective_with_coordinate err={}", ret);
                throw SailBMImageError("bmcv api fail");
            }   
        }
        return output_image;
    }

    int Bmcv::drawPoint(
        const BMImage &image,
        std::pair<int,int> center,
        std::tuple<unsigned char, unsigned char, unsigned char> color,
        int radius){
        if(image.format() != FORMAT_GRAY &&
            image.format() != FORMAT_YUV420P &&
            image.format() != FORMAT_YUV422P &&
            image.format() != FORMAT_NV12 &&
            image.format() != FORMAT_NV21 &&
            image.format() != FORMAT_NV16 &&
            image.format() != FORMAT_NV61){
            SPDLOG_ERROR("input format not supported!");
            print_image(image.data(),"input");
            return BM_ERR_FAILURE;
        }
        if(center.first >= image.width() || center.second >= image.height()
            || center.first < 0 || center.second < 0){
            SPDLOG_ERROR("drawPoint failed, point is outside, center({},{}) vs. image width:{}, image height:{}",
                center.first, center.second, image.width(), image.height());
            return BM_ERR_FAILURE;
        }
       
        bmcv_point_t org_center = {center.first, center.second};
        bmcv_point_t point_start[2];
        bmcv_point_t point_end[2];
        point_start[0].x = center.first - radius;
        point_start[0].y = center.second;
        point_start[0].x = point_start[0].x > 0 ? point_start[0].x : 0;

        point_end[0].x = center.first + radius;
        point_end[0].y = center.second;
        point_end[0].x = point_end[0].x < image.width() - 1 ? point_end[0].x : image.width() - 1;

        point_start[1].x = center.first ;
        point_start[1].y = center.second - radius;
        point_start[1].y = point_start[1].y > 0 ? point_start[1].y : 0;

        point_end[1].x = center.first;
        point_end[1].y = center.second + radius; 
        point_end[1].y = point_end[1].y < image.height() - 1 ? point_end[1].y : image.height() - 1;

        bmcv_color_t color_put = {std::get<2>(color), std::get<1>(color), std::get<0>(color)};

        int thickness = radius / 2 + 1; 

        int ret = bmcv_image_draw_lines(
            handle_.data(),
            image.data(),
            point_start,
            point_end,
            2,
            color_put,
            thickness);

         if (BM_SUCCESS != ret) {
            print_image(image.data(),"input");
            SPDLOG_ERROR("bmcv_image_draw_lines() err={}", ret);
            return ret;
        }
        return BM_SUCCESS;
    }

    int Bmcv::drawPoint(
        const bm_image  &image,
        std::pair<int,int> center,
        std::tuple<unsigned char, unsigned char, unsigned char> color,
        int radius){
            return drawPoint_(image, center, color, radius);
        }

    int Bmcv::drawPoint_(
        const bm_image  &image,
        std::pair<int,int> center,
        std::tuple<unsigned char, unsigned char, unsigned char> color,
        int radius){

        if(image.image_format != FORMAT_GRAY &&
            image.image_format != FORMAT_YUV420P &&
            image.image_format != FORMAT_YUV422P &&
            image.image_format != FORMAT_NV12 &&
            image.image_format != FORMAT_NV21 &&
            image.image_format != FORMAT_NV16 &&
            image.image_format != FORMAT_NV61){
            SPDLOG_ERROR("input format not supported!");
            print_image(image,"input");
            return BM_ERR_FAILURE;
        }
        if(center.first >= image.width || center.second >= image.height
            || center.first < 0 || center.second < 0){
            SPDLOG_ERROR("drawPoint failed, point is outside, center({},{}) vs. image width:{}, image height:{}",
                center.first, center.second, image.width, image.height);
            return BM_ERR_FAILURE;
        }
       
        bmcv_point_t org_center = {center.first, center.second};
        bmcv_point_t point_start[2];
        bmcv_point_t point_end[2];
        point_start[0].x = center.first - radius;
        point_start[0].y = center.second;
        point_start[0].x = point_start[0].x > 0 ? point_start[0].x : 0;

        point_end[0].x = center.first + radius;
        point_end[0].y = center.second;
        point_end[0].x = point_end[0].x < image.width - 1 ? point_end[0].x : image.width - 1;

        point_start[1].x = center.first ;
        point_start[1].y = center.second - radius;
        point_start[1].y = point_start[1].y > 0 ? point_start[1].y : 0;

        point_end[1].x = center.first;
        point_end[1].y = center.second + radius; 
        point_end[1].y = point_end[1].y < image.height - 1 ? point_end[1].y : image.height - 1;

        bmcv_color_t color_put = {std::get<2>(color), std::get<1>(color), std::get<0>(color)};

        int thickness = radius / 2 + 1; 

        int ret = bmcv_image_draw_lines(
            handle_.data(),
            image,
            point_start,
            point_end,
            2,
            color_put,
            thickness);

         if (BM_SUCCESS != ret) {
            print_image(image,"input");
            SPDLOG_ERROR("bmcv_image_draw_lines() err={}", ret);
            return ret;
        }
        return BM_SUCCESS;
    }

    int Bmcv::drawLines(
        BMImage &image,
        std::vector<std::pair<int,int>> &start_points,
        std::vector<std::pair<int,int>> &end_points,
        int line_num,
        std::tuple<unsigned char, unsigned char, unsigned char> color,
        int thickness){

        if(image.format() != FORMAT_GRAY &&
            image.format() != FORMAT_YUV420P &&
            image.format() != FORMAT_YUV422P &&
            image.format() != FORMAT_YUV444P &&
            image.format() != FORMAT_NV12 &&
            image.format() != FORMAT_NV21 &&
            image.format() != FORMAT_NV16 &&
            image.format() != FORMAT_NV61){
            SPDLOG_ERROR("input format not supported!");
            print_image(image.data(),"input");
            return BM_ERR_FAILURE;
        }

        if(line_num!=start_points.size()||line_num!=end_points.size()){
            SPDLOG_ERROR("the size of points is error, line_num:{},start_points size:{},end_points size:{}",
                line_num,start_points.size(),end_points.size());
        }

        for(int i=0;i<line_num;i++){
            if(start_points[i].first >= image.width() || start_points[i].second >= image.height()
            || start_points[i].first < 0 || start_points[i].second < 0){
                SPDLOG_ERROR("drawLines failed, point is outside, point({},{}) vs. image width:{}, image height:{}",
                start_points[i].first, start_points[i].second, image.width(), image.height());
                return BM_ERR_FAILURE;
            }
            if(end_points[i].first >= image.width() || end_points[i].second >= image.height()
            || end_points[i].first < 0 || end_points[i].second < 0){
                SPDLOG_ERROR("drawLines failed, point is outside, point({},{}) vs. image width:{}, image height:{}",
                end_points[i].first, end_points[i].second, image.width(), image.height());
                return BM_ERR_FAILURE;
            }
        }

        bmcv_point_t* start_points_ = new bmcv_point_t[line_num];
        bmcv_point_t* end_points_ = new bmcv_point_t[line_num];
        bmcv_color_t color_ = {std::get<2>(color), std::get<1>(color), std::get<0>(color)};

        for(int i = 0; i < line_num; i++){
            start_points_[i].x = start_points[i].first;
            start_points_[i].y = start_points[i].second;
            end_points_[i].x = end_points[i].first;
            end_points_[i].y = end_points[i].second;
        }

        int ret = bmcv_image_draw_lines(
            handle_.data(),
            image.data(),
            start_points_,
            end_points_,
            line_num,
            color_,
            thickness);

        delete[] start_points_;
        delete[] end_points_;

        if (BM_SUCCESS != ret) {
            print_image(image.data(),"input");
            SPDLOG_ERROR("bmcv_image_draw_lines() err={}", ret);
            return ret;
        }

        return BM_SUCCESS;
    }


    BMImage Bmcv::imdecode(const void* jpeg_ptr, size_t jpeg_size) {
        BMImage Img;
        char *buffer = (char*)jpeg_ptr;
        // cv::Mat *m1 = new cv::Mat();
        auto m1 = std::make_shared<cv::Mat>();
        std::vector<char> pic(buffer, buffer+jpeg_size);
        cv::imdecode(pic, cv::IMREAD_COLOR, m1.get(), handle_.get_device_id());

        int ret = cv::bmcv::toBMI(*m1, &Img.data());
        if (ret != BM_SUCCESS) {
            spdlog::error("cv::bmcv::toBMI() err {},{}", __FILE__, __LINE__);
            throw SailBMImageError("Bmcv imdecode() failed!");
        }
        Img._impl->cache_mat(m1);
        return Img;
    }

    bool Bmcv::imencode(std::string& ext, bm_image &img, std::vector<u_char>& buf)
    {
        cv::Mat mat;
        bm_status_t ret_0 = cv::bmcv::toMAT(&img, mat);
        if (ret_0 != BM_SUCCESS){
            spdlog::error("cv::bmcv::toMAT() err {},{}", __FILE__, __LINE__);
            throw SailBMImageError("Bmcv imencode() failed!");
        }
        bool ret = cv::imencode(ext.c_str(), mat, buf);
        return ret;
    }

    bool Bmcv::imencode(std::string& ext, BMImage &img, std::vector<u_char>& buf)
    {
        return imencode(ext, img.data(), buf);
    }

    int Bmcv::imread(const std::string &filename, BMImage &dst)
    {
        std::string extension = filename.substr(filename.find_last_of('.') + 1);
        bool jpeg_decoded = false;

        // Modify environment variables for JPEG 2000
        std::string variableName = "OPENCV_IO_ENABLE_JASPER";
        std::string variableValue = "1"; 
        if (extension == "j2k" || extension == "jp2") 
        {
            const char* currentValue = getenv(variableName.c_str());
            if (currentValue == nullptr) 
            {
        #ifdef _WIN32
                if (SetEnvironmentVariable(variableName.c_str(), variableValue.c_str())) {
        #else
                if (setenv(variableName.c_str(), variableValue.c_str(), 1) == 0) {
        #endif
                    spdlog::info("Environment variable {} set to {} successfully.", variableName, variableValue);
                } else 
                {
                    spdlog::warn("Failed to set environment variable {}.", variableName);
                }
            }
        }

        if (extension == "jpeg" || extension == "jpg") {
            // read jpeg data from file
            std::ifstream file(filename, std::ios::binary | std::ios::ate);
            if (!file.is_open()) return SAIL_ERR_DEC_OPEN;
            size_t size = file.tellg();
            file.seekg(0, std::ios::beg);
            std::vector<unsigned char> jpeg_data(size);
            if (!file.read(reinterpret_cast<char*>(jpeg_data.data()), size))
                return SAIL_ERR_DEC_OPEN;
            file.close();
            bm_image &bmcv_dst = dst.data();
            memset((char*)&bmcv_dst, 0, sizeof(bm_image));

            // decode jpeg data to yuv
            auto jpeg_data_ptr = jpeg_data.data();
            int ret = bmcv_image_jpeg_dec(handle_.data(), reinterpret_cast<void**>(&jpeg_data_ptr), &size, 1, &bmcv_dst);
            if (ret == 0 && bmcv_dst.width > 0 && bmcv_dst.height > 0 &&
                (bmcv_dst.image_format == FORMAT_YUV420P || bmcv_dst.image_format == FORMAT_YUV422P ||
                bmcv_dst.image_format == FORMAT_YUV444P || bmcv_dst.image_format == FORMAT_GRAY))
            {
                jpeg_decoded = true;
            } else {
                spdlog::error("{}:{}:{} Hardware decode failed, trying software decode", __FILE__, __func__, __LINE__);
            }
        }

        if (!jpeg_decoded) {
            cv::Mat img = cv::imread(filename, cv::IMREAD_UNCHANGED);
            if (img.empty()) {
                spdlog::error("{}:{}:{} imread fail, image is empty", __FILE__, __func__, __LINE__);
                return SAIL_ERR_DEC_OPEN;
            }
            std::shared_ptr<cv::Mat> cached_mat = std::make_shared<cv::Mat>(img);
            dst.cache_ost_mat(cached_mat);
            bm_image &bmcv_dst = dst.data();
            memset(&bmcv_dst, 0, sizeof(bm_image));
            bm_handle_t handle = img.u->hid ? img.u->hid : cv::bmcv::getCard();
            bm_image_data_format_ext data_format;
            bm_image_format_ext image_format;

            if (!img.u || !img.u->addr) {
                spdlog::error("Memory allocated by user, no device memory assigned. Not support BMCV!");
                return BM_NOT_SUPPORTED;
            }
            if (img.type() == 16) { data_format = DATA_TYPE_EXT_1N_BYTE; image_format = FORMAT_BGR_PACKED; }
            else if (img.type() == 24) {data_format = DATA_TYPE_EXT_1N_BYTE; image_format = FORMAT_ABGR_PACKED; }
            else if (img.type() == 0) {data_format = DATA_TYPE_EXT_1N_BYTE; image_format = FORMAT_GRAY; }
            else return BM_NOT_SUPPORTED;
            int step[1] = { (int)img.step[0] };
            int ret = bm_image_create(handle, img.rows, img.cols, image_format, data_format, &bmcv_dst, step);
            if (ret != BM_SUCCESS) {
                spdlog::error("bm_image_create() err {},{}", __FILE__, __LINE__);
                return BM_NOT_SUPPORTED;
            }

            uint len, off;
            bm_device_mem_t mem;
            off = img.data - img.datastart;
            len = img.rows * img.step[0];
            mem = bm_mem_from_device(img.u->addr + off, len);
            ret = bm_image_attach(bmcv_dst, &mem);
            if (ret != BM_SUCCESS) {
                spdlog::error("bm_image_attach() err {},{}", __FILE__, __LINE__);
                return BM_NOT_SUPPORTED;
            }
        }

        return SAIL_SUCCESS;
    }

    BMImage Bmcv::imread(const std::string &filename)
    {
        BMImage dst;
        int ret = imread(filename, dst);
        SAIL_CHECK_RET(ret);
        return std::move(dst);
    }
#if BMCV_VERSION_MAJOR > 1
extern "C" {
    bm_status_t bmcv_stft(bm_handle_t handle, float* XRHost, float* XIHost, float* YRHost,
                    float* YIHost, int batch, int L, bool realInput,
                    int pad_mode, int n_fft, int win_mode, int hop_len, bool normalize) __attribute__((weak));
    bm_status_t bmcv_istft(bm_handle_t handle, float* XRHost, float* XIHost, float* YRHost,
                    float* YIHost, int batch, int L, bool realInput,
                    int pad_mode, int n_fft, int win_mode, int hop_len, bool normalize) __attribute__((weak));
}

#ifdef PYTHON
    std::tuple<pybind11::array_t<float>, pybind11::array_t<float>> Bmcv::stft(
        pybind11::array_t<float> input_real,
        pybind11::array_t<float> input_imag,
        bool realInput,
        bool normalize,
        int n_fft,
        int hop_len,
        int pad_mode,
        int win_mode
    )
    {
        if (!bmcv_stft) {
            SPDLOG_ERROR("stft is not available in this version, please upgrade sdk to v1.8 or later.");
            throw std::runtime_error("stft is not available in this version, please upgrade sdk to v1.8 or later.");
        }
        if (n_fft <= 0) {
            SPDLOG_ERROR("n_fft must be a positive integer.");
            throw std::invalid_argument("n_fft must be a positive integer.");
        }
        if (input_real.shape(1) < n_fft) {
            SPDLOG_ERROR("Input length must be greater than n_fft.");
            throw std::invalid_argument("Input length must be greater than n_fft.");
        }
        if (input_real.ndim() != 2 || input_imag.ndim() != 2) {
            SPDLOG_ERROR("Both inputs must be 2D arrays.");
            throw std::invalid_argument("Both inputs must be 2D arrays.");
        }
        if (input_real.shape(0) != input_imag.shape(0) || 
            input_real.shape(1) != input_imag.shape(1)) {
            SPDLOG_ERROR("Both input arrays must have the same shape.");
            throw std::invalid_argument("Both input arrays must have the same shape.");
        }

        // Check if input is a C-contiguous array
        if (!pybind11::detail::check_flags(input_real.ptr(), pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_)) {
            pybind11::module np = pybind11::module::import("numpy");
            input_real = np.attr("ascontiguousarray")(input_real, "dtype"_a="float32");
        }

        if (!pybind11::detail::check_flags(input_imag.ptr(), pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_)) {
            pybind11::module np = pybind11::module::import("numpy");
            input_imag = np.attr("ascontiguousarray")(input_imag, "dtype"_a="float32");
        }
        
        pybind11::buffer_info real_buf_info = input_real.request();
        pybind11::buffer_info imag_buf_info = input_imag.request();

        int batch = input_real.shape(0);
        int L = input_real.shape(1);
        int num_frames = 1 + L / hop_len; // col
        int row_num = n_fft / 2 + 1; // row
        pybind11::array_t<float> output_real = pybind11::array_t<float>({batch, row_num, num_frames});
        pybind11::array_t<float> output_imag = pybind11::array_t<float>({batch, row_num, num_frames});

        int ret = bmcv_stft(handle_.data(), static_cast<float*>(real_buf_info.ptr), static_cast<float*>(imag_buf_info.ptr), 
                static_cast<float*>(output_real.request().ptr), 
                static_cast<float*>(output_imag.request().ptr),
                batch, L, realInput, pad_mode, n_fft, win_mode, hop_len, normalize);
        
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("stft error: stft execute failed err={}", ret);
            throw std::runtime_error("stft execution failed with error code: " + std::to_string(ret));
        }

        return std::make_tuple(output_real, output_imag);
    }
#endif

    std::tuple<Tensor, Tensor> Bmcv::stft(
        Tensor &input_real,
        Tensor &input_imag,
        bool realInput,
        bool normalize,
        int n_fft,
        int hop_len,
        int pad_mode,
        int win_mode
    ) {
        if (!bmcv_stft) {
            SPDLOG_ERROR("stft is not available in this version, please upgrade sdk to v1.8 or later.");
            throw std::runtime_error("stft is not available in this version, please upgrade sdk to v1.8 or later.");
        }

        auto input_real_shape = input_real.shape();
        auto input_imag_shape = input_imag.shape();
        if (n_fft <= 0) {
            SPDLOG_ERROR("n_fft must be a positive integer.");
            throw std::invalid_argument("n_fft must be a positive integer.");
        }
        if (input_real_shape[1] < n_fft) {
            SPDLOG_ERROR("Input length must be greater than n_fft.");
            throw std::invalid_argument("Input length must be greater than n_fft.");
        }
        if (input_real_shape.size() != 2 || input_imag_shape.size() != 2) {
            SPDLOG_ERROR("Both inputs must be 2D arrays.");
            throw std::invalid_argument("Both inputs must be 2D arrays.");
        }
        if (input_real_shape[0] != input_imag_shape[0] || 
            input_real_shape[1] != input_imag_shape[1]) {
            SPDLOG_ERROR("Both input arrays must have the same shape.");
            throw std::invalid_argument("Both input arrays must have the same shape.");
        }

        int batch = input_real_shape[0];
        int L = input_real_shape[1];
        int num_frames = 1 + L / hop_len; // col
        int row_num = n_fft / 2 + 1; // row
        size_t size = batch * row_num * num_frames;

        std::vector<float> output_real(size);
        std::vector<float> output_imag(size);

        float* real_ptr = static_cast<float*>(input_real.sys_data());
        float* imag_ptr = static_cast<float*>(input_imag.sys_data());
        int ret = bmcv_stft(handle_.data(), real_ptr, imag_ptr, 
                            output_real.data(), 
                            output_imag.data(),
                            batch, L, realInput, pad_mode, n_fft, win_mode, hop_len, normalize);
        
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("stft error: stft execute failed err={}", ret);
            throw std::runtime_error("stft execution failed with error code: " + std::to_string(ret));
        }

        Tensor output_real_tensor = Tensor(handle_, {batch, row_num, num_frames}, BM_FLOAT32, true, false);
        Tensor output_imag_tensor = Tensor(handle_, {batch, row_num, num_frames}, BM_FLOAT32, true, false);
        
        std::vector<int> shape = {batch, row_num, num_frames};
        output_real_tensor.reset_sys_data(output_real.data(), shape);
        output_imag_tensor.reset_sys_data(output_imag.data(), shape);

        return std::make_tuple(output_real_tensor, output_imag_tensor);
    }

#ifdef PYTHON
    std::tuple<pybind11::array_t<float>, pybind11::array_t<float>> Bmcv::istft(
        pybind11::array_t<float> input_real,
        pybind11::array_t<float> input_imag,
        bool realInput,
        bool normalize,
        int L,
        int hop_len,
        int pad_mode,
        int win_mode
    )
    {
        if (!bmcv_istft) {
            SPDLOG_ERROR("istft is not available in this version, please upgrade sdk to v1.8 or later.");
            throw std::runtime_error("istft is not available in this version, please upgrade sdk to v1.8 or later.");
        }
        if (input_real.shape(0) != input_imag.shape(0) || 
            input_real.shape(1) != input_imag.shape(1) || 
            input_real.shape(2) != input_imag.shape(2)) {
            SPDLOG_ERROR("Input shapes do not match. Real shape: ({}, {}, {}), Imaginary shape: ({}, {}, {})", 
                        input_real.shape(0), input_real.shape(1), input_real.shape(2),
                        input_imag.shape(0), input_imag.shape(1), input_imag.shape(2));
            throw std::invalid_argument("Input shapes do not match.");
        }

        if (input_real.ndim() != 3 || input_imag.ndim() != 3) {
            SPDLOG_ERROR("Both inputs must be 3D arrays. Real ndim: {}, Imaginary ndim: {}", input_real.ndim(), input_imag.ndim());
            throw std::runtime_error("Both inputs must be 3D arrays.");
        }
        if (!pybind11::detail::check_flags(input_real.ptr(), pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_)) {
            pybind11::module np = pybind11::module::import("numpy");
            input_real = np.attr("ascontiguousarray")(input_real, "dtype"_a="float32");
        }

        if (!pybind11::detail::check_flags(input_imag.ptr(), pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_)) {
            pybind11::module np = pybind11::module::import("numpy");
            input_imag = np.attr("ascontiguousarray")(input_imag, "dtype"_a="float32");
        }

        int batch = input_real.shape(0);
        int n_fft = (input_real.shape(1) - 1) *2;
        
        pybind11::buffer_info real_buf_info = input_real.request();
        pybind11::buffer_info imag_buf_info = input_imag.request();
        pybind11::array_t<float> output_real = pybind11::array_t<float>({batch, L});
        pybind11::array_t<float> output_imag = pybind11::array_t<float>({batch, L});
        
        int ret = bmcv_istft(handle_.data(), static_cast<float*>(real_buf_info.ptr), static_cast<float*>(imag_buf_info.ptr), 
                            static_cast<float*>(output_real.request().ptr), 
                            static_cast<float*>(output_imag.request().ptr),
                            batch, L, realInput, pad_mode, n_fft, win_mode, hop_len, normalize);

        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("istft error: istft execute failed err={}", ret);
            throw std::runtime_error("istft execution failed with error code: " + std::to_string(ret));
        }

        return std::make_tuple(output_real, output_imag);
    }
#endif

    std::tuple<Tensor, Tensor> Bmcv::istft(
        Tensor &input_real,
        Tensor &input_imag,
        bool realInput,
        bool normalize,
        int L,
        int hop_len,
        int pad_mode,
        int win_mode) 
    {
        if (!bmcv_istft) {
            SPDLOG_ERROR("istft is not available in this version, please upgrade sdk to v1.8 or later.");
            throw std::runtime_error("istft is not available in this version, please upgrade sdk to v1.8 or later.");
        }
        auto input_real_shape = input_real.shape();
        auto input_imag_shape = input_imag.shape();

        if (input_real_shape.size() != 3 || input_imag_shape.size() != 3) {
            SPDLOG_ERROR("Both inputs must be 3D arrays. Real ndim: {}, Imaginary ndim: {}", input_real_shape.size(), input_imag_shape.size());
            throw std::invalid_argument("Both inputs must be 3D arrays.");
        }

        auto shapes_equal = [](const auto& shape1, const auto& shape2) {
            if (shape1.size() != shape2.size()) return false;
            for (size_t i = 0; i < shape1.size(); ++i) {
                if (shape1[i] != shape2[i]) return false;
            }
            return true;
        };

        if (!shapes_equal(input_real_shape, input_imag_shape)) {
            SPDLOG_ERROR("Both input arrays must have the same shape.");
            throw std::invalid_argument("Both input arrays must have the same shape.");
        }

        int batch = input_real_shape[0];
        int n_fft = (input_real_shape[1] - 1) * 2;
        size_t size = batch * L;

        std::vector<float> output_real(size);
        std::vector<float> output_imag(size);
        
        float* real_ptr = static_cast<float*>(input_real.sys_data());
        float* imag_ptr = static_cast<float*>(input_imag.sys_data());
        int ret = bmcv_istft(handle_.data(), real_ptr, imag_ptr, 
                            output_real.data(), 
                            output_imag.data(),
                            batch, L, realInput, pad_mode, n_fft, win_mode, hop_len, normalize);

        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("istft error: istft execute failed err={}", ret);
            throw std::runtime_error("istft execution failed with error code: " + std::to_string(ret));
        }

        Tensor output_real_tensor = Tensor(handle_, {batch, L}, BM_FLOAT32, true, false);
        Tensor output_imag_tensor = Tensor(handle_, {batch, L}, BM_FLOAT32, true, false);
        
        std::vector<int> shape = {batch, L};
        output_real_tensor.reset_sys_data(output_real.data(), shape);
        output_imag_tensor.reset_sys_data(output_imag.data(), shape);

        return std::make_tuple(output_real_tensor, output_imag_tensor);
    }
#endif

    std::vector<Tensor> Bmcv::fft(bool forward, Tensor &input_real)
    {
        if(input_real.dev_data().u.device.device_addr == 0 || input_real.dev_data().size <= 0)
        {
            SPDLOG_ERROR("fft error: input tensor does not own dev data!");
            throw SailBMImageError("invalid argument");
        }
        void *plan = nullptr;
        int ret = BM_SUCCESS;
        auto shape = input_real.shape();

        if(1 != shape[1])
        {
            SPDLOG_ERROR("fft error: input tensor should only has 1 channel, check channel !");
            throw SailBMImageError("invalid argument");
        }
        if(1 == shape[2])
        {
            ret = bmcv_fft_1d_create_plan(handle_.data(),shape[2],shape[3],forward,plan);
        }
        else if(1 == shape[0])
        {
            ret = bmcv_fft_2d_create_plan(handle_.data(),shape[2],shape[3],forward,plan);
        }
        else
        {
            SPDLOG_ERROR("fft error: dim must be 1 or 2, check dim!");
            throw SailBMImageError("invalid argument");
        }

        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("fft error: fft create plan failed err={}", ret);
            throw SailBMImageError("bmcv api fail");
        }

        bm_device_mem_t out_real_data;
        bm_device_mem_t out_imag_data;
        ret = bm_malloc_device_byte(handle_.data(), &out_real_data, shape[2]*shape[3] * 4);
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("bm_malloc_device_byte failed, size = {}, ret = {}",
                         shape[2] * shape[3] * 4, ret);
            throw SailRuntimeError("device memory not enough");
        }
        ret = bm_malloc_device_byte(handle_.data(), &out_imag_data, shape[2]*shape[3] * 4);
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("bm_malloc_device_byte failed, size = {}, ret = {}",
                         shape[2] * shape[3] * 4, ret);
            throw SailRuntimeError("device memory not enough");
        }

        ret = bmcv_fft_execute_real_input(handle_.data(),input_real.dev_data(),out_real_data,out_imag_data,plan);
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("fft error: fft execute failed err={}", ret);
            throw SailBMImageError("bmcv api fail");
        }

        bmcv_fft_destroy_plan(handle_.data(), plan);

        Tensor output_real = Tensor(handle_, {1, 1, shape[2], shape[3]}, BM_FLOAT32, true, false);
        Tensor output_imag = Tensor(handle_, {1, 1, shape[2], shape[3]}, BM_FLOAT32, true, false);
        output_real.reset_dev_data(out_real_data);
        output_imag.reset_dev_data(out_imag_data);

        std::vector<Tensor> outputs(2);
        outputs[0] = std::move(output_real);
        outputs[1] = std::move(output_imag);
        return outputs;
    }
    
    std::vector<Tensor> Bmcv::fft(bool forward, Tensor &input_real, Tensor &input_imag)
    {
        if(input_real.dev_data().u.device.device_addr == 0 || input_real.dev_data().size <= 0)
        {
            SPDLOG_ERROR("fft error: input tensor not own dev data!");
            throw SailBMImageError("invalid argument");
        }
        void *plan = nullptr;
        int ret = BM_SUCCESS;
        auto shape = input_real.shape();

        if(1 != shape[1])
        {
            SPDLOG_ERROR("fft error: input tensor should only has 1 channel, check channel !");
            throw SailBMImageError("invalid argument");
        }
        if(1 == shape[2])
        {
            bmcv_fft_1d_create_plan(handle_.data(),shape[2],shape[3],forward,plan);
        }
        else if(1 == shape[0])
        {
            bmcv_fft_2d_create_plan(handle_.data(),shape[2],shape[3],forward,plan);
        }
        else
        {
            SPDLOG_ERROR("fft error: dim must be 1 or 2, check dim!");
            throw SailBMImageError("invalid argument");
        }

        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("fft error: fft create paln failed err={}", ret);
            throw SailBMImageError("bmcv api fail");
        }

        bm_device_mem_t out_real_data;
        bm_device_mem_t out_imag_data;
        ret = bm_malloc_device_byte(handle_.data(), &out_real_data, shape[2]*shape[3] * 4);
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("bm_malloc_device_byte failed, size = {}, ret = {}",
                         shape[2] * shape[3] * 4, ret);
            throw SailRuntimeError("device memory not enough");
        }
        ret = bm_malloc_device_byte(handle_.data(), &out_imag_data, shape[2]*shape[3] * 4);
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("bm_malloc_device_byte failed, size = {}, ret = {}",
                         shape[2] * shape[3] * 4, ret);
            throw SailRuntimeError("device memory not enough");
        }

        bmcv_fft_execute(handle_.data(),input_real.dev_data(),input_imag.dev_data(),out_real_data,out_imag_data,plan);
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("fft error: fft execute failed err={}", ret);
            throw SailBMImageError("bmcv api fail");
        }

        bmcv_fft_destroy_plan(handle_.data(), plan);

        Tensor output_real = Tensor(handle_, {1, 1, shape[2], shape[3]}, BM_FLOAT32, true, false);
        Tensor output_imag = Tensor(handle_, {1, 1, shape[2], shape[3]}, BM_FLOAT32, true, false);
        output_real.reset_dev_data(out_real_data);
        output_imag.reset_dev_data(out_imag_data);

        std::vector<Tensor> outputs(2);
        outputs[0] = std::move(output_real);
        outputs[1] = std::move(output_imag);
        return outputs;
    }

    int Bmcv::convert_yuv420p_to_gray(BMImage& input, BMImage& output) {
        if(input.format() != FORMAT_YUV420P) {
            SPDLOG_ERROR("convert_yuv420p_to_gray error: Input format must be yuv420p!");
            print_image(input.data(), "src: ");
            return BM_ERR_FAILURE;
        }

        if (output.is_created()) {
            if(output.width() != input.width() || output.height() != input.height()) {
                SPDLOG_ERROR("convert_yuv420p_to_gray error: The width and height of the output and the output must be equal!");
                print_image(input.data(), "src: ");
                print_image(output.data(), "dst: ");
                return BM_ERR_FAILURE;
            }
            if(output.format() != FORMAT_GRAY) {
                SPDLOG_ERROR("convert_yuv420p_to_gray error: Output format must be gray!");
                print_image(output.data(), "dst: ");
                return BM_ERR_FAILURE;
            }
        }

        if (!output.is_created()) {
            output.create(
                    handle_,
                    input.height(),
                    input.width(),
                    FORMAT_GRAY, // force to this format
                    input.dtype()
            );
            output.allocate();
        }

        int ret = BM_SUCCESS;

        bm_device_mem_t yuv420p_device_addr[3] = {0};
        ret = bm_image_get_device_mem(input.data(), yuv420p_device_addr);
        if(BM_SUCCESS != ret) {
            SPDLOG_ERROR("convert_yuv420p_to_gray error: call bm_image_get_device_mem failed!");
            print_image(input.data(), "src: ");
            return ret;
        }

        int size;
        ret = bm_image_get_byte_size(output.data(), &size);
        if(BM_SUCCESS != ret) {
            SPDLOG_ERROR("convert_yuv420p_to_gray error: call bm_image_get_byte_size failed!");
            print_image(output.data(), "dst: ");
            return ret;
        }

        bm_device_mem_t gray_device_addr;
        ret = bm_malloc_device_byte(handle_.data(), &gray_device_addr, size);
        if(BM_SUCCESS != ret) {
            SPDLOG_ERROR("convert_yuv420p_to_gray error: call bm_malloc_device_byte failed!");
            print_image(output.data(), "dst: ");
            return ret;
        }

        ret = bm_memcpy_d2d_byte(handle_.data(), gray_device_addr, 0, yuv420p_device_addr[0], 0, size);
        if(BM_SUCCESS != ret) {
            SPDLOG_ERROR("convert_yuv420p_to_gray error: call bm_memcpy_d2d_byte failed!");
            print_image(input.data(), "src: ");
            print_image(output.data(), "dst: ");
            return ret;
        }

        ret = bm_image_attach(output.data(), &gray_device_addr);
        if(BM_SUCCESS != ret) {
            SPDLOG_ERROR("convert_yuv420p_to_gray error: call bm_image_attach failed!");
            print_image(output.data(), "dst: ");
            return ret;
        }

        return BM_SUCCESS;
    }

    int Bmcv::convert_yuv420p_to_gray(bm_image& input, bm_image& output) {
        return convert_yuv420p_to_gray_(input, output);
    }

    int Bmcv::convert_yuv420p_to_gray_(bm_image& input, bm_image& output) {
        if(input.image_format != FORMAT_YUV420P) {
            SPDLOG_ERROR("convert_yuv420p_to_gray_ error: Input format must be yuv420p!");
            print_image(input, "src: ");
            return BM_ERR_FAILURE;
        }

        if(output.width != input.width || output.height != input.height) {
            SPDLOG_ERROR("convert_yuv420p_to_gray_ error: The width and height of the output and the output must be equal!");
            print_image(input, "src: ");
            print_image(output, "dst: ");
            return BM_ERR_FAILURE;
        }
        if(output.image_format != FORMAT_GRAY) {
            SPDLOG_ERROR("convert_yuv420p_to_gray_ error: Output format must be gray!");
            print_image(output, "dst: ");
            return BM_ERR_FAILURE;
        }

        int ret = BM_SUCCESS;

        bm_device_mem_t yuv420p_device_addr[3] = {0};
        ret = bm_image_get_device_mem(input, yuv420p_device_addr);
        if(BM_SUCCESS != ret) {
            SPDLOG_ERROR("convert_yuv420p_to_gray_ error: call bm_image_get_device_mem failed!");
            print_image(input, "src: ");
            return ret;
        }

        int size;
        ret = bm_image_get_byte_size(output, &size);
        if(BM_SUCCESS != ret) {
            SPDLOG_ERROR("convert_yuv420p_to_gray_ error: call bm_image_get_byte_size failed!");
            print_image(output, "dst: ");
            return ret;
        }

        bm_device_mem_t gray_device_addr;
        ret = bm_malloc_device_byte(handle_.data(), &gray_device_addr, size);
        if(BM_SUCCESS != ret) {
            SPDLOG_ERROR("convert_yuv420p_to_gray_ error: call bm_malloc_device_byte failed!");
            print_image(output, "dst: ");
            return ret;
        }

        ret = bm_memcpy_d2d_byte(handle_.data(), gray_device_addr, 0, yuv420p_device_addr[0], 0, size);
        if(BM_SUCCESS != ret) {
            SPDLOG_ERROR("convert_yuv420p_to_gray_ error: call bm_memcpy_d2d_byte failed!");
            print_image(input, "src: ");
            print_image(output, "dst: ");
            return ret;
        }

        ret = bm_image_attach(output, &gray_device_addr);
        if(BM_SUCCESS != ret) {
            SPDLOG_ERROR("convert_yuv420p_to_gray_ error: call bm_image_attach failed!");
            print_image(output, "dst: ");
            return ret;
        }

        return BM_SUCCESS;
    }

    int Bmcv::polylines(
        BMImage &img,
        std::vector<std::vector<std::pair<int,int>>> &pts,
        bool isClosed,
        std::tuple<unsigned char, unsigned char, unsigned char> color,
        int thickness,
        int shift){
        

        if(img.format() != FORMAT_GRAY &&
            img.format() != FORMAT_YUV420P &&
            img.format() != FORMAT_YUV422P &&
            img.format() != FORMAT_YUV444P &&
            img.format() != FORMAT_NV12 &&
            img.format() != FORMAT_NV21 &&
            img.format() != FORMAT_NV16 &&
            img.format() != FORMAT_NV61){
            SPDLOG_ERROR("input format not supported!");
            print_image(img.data(),"input");
            return SAIL_ERR_BMI_NOTSUP;
        }
        
        int edge_num = 0;
        std::vector<std::pair<int,int>> start_points;
        std::vector<std::pair<int,int>> end_points;


        for(int i = 0; i < pts.size(); i++){
            std::vector<std::pair<int,int>> points = pts[i];
            
            if(points.size() < 2){
                points[0].first /= 1<<shift;
                points[0].second /= 1<<shift;
                start_points.push_back(points[0]);
                end_points.push_back(points[0]);
            }else if(points.size() < 3){
                points[0].first /= 1<<shift;
                points[0].second /= 1<<shift;
                points[1].first /= 1<<shift;
                points[1].second /= 1<<shift;
                start_points.push_back(points[0]);
                end_points.push_back(points[1]);
            }else{
                points[0].first /= 1<<shift;
                points[0].second /= 1<<shift;
                for(int i = 1; i < points.size(); i++){
                    points[i].first /= 1<<shift;
                    points[i].second /= 1<<shift;
                    start_points.push_back(points[i-1]);
                    end_points.push_back(points[i]);
                }
                if(isClosed){
                    start_points.push_back(points[points.size() - 1]);
                    end_points.push_back(points[0]);
                }
            }
        }
        edge_num = start_points.size();
        // printf("%d\n", edge_num);

        // for(int i = 0; i < edge_num; i++){
        //     printf("start points -> x: %d, y: %d\n", start_points[i].first, start_points[i].second);
        //     printf("end points -> x: %d, y: %d\n", end_points[i].first, end_points[i].second);
        // }

        for(int i = 0; i < edge_num; i++){
            if(start_points[i].first >= img.width() || start_points[i].second >= img.height()
            || start_points[i].first < 0 || start_points[i].second < 0){
                SPDLOG_ERROR("drawPolygon failed, point is outside, point({},{}) vs. image width:{}, image height:{}",
                start_points[i].first, start_points[i].second, img.width(), img.height());
                return BM_ERR_FAILURE;
            }
            if(end_points[i].first >= img.width() || end_points[i].second >= img.height()
            || end_points[i].first < 0 || end_points[i].second < 0){
                SPDLOG_ERROR("drawPolygon failed, point is outside, point({},{}) vs. image width:{}, image height:{}",
                end_points[i].first, end_points[i].second, img.width(), img.height());
                return BM_ERR_FAILURE;
            }
        }

        bmcv_point_t* start_points_ = new bmcv_point_t[edge_num];
        bmcv_point_t* end_points_ = new bmcv_point_t[edge_num];
        bmcv_color_t color_ = {std::get<2>(color), std::get<1>(color), std::get<0>(color)};

        for(int i = 0; i < edge_num; i++){
            start_points_[i].x = start_points[i].first;
            start_points_[i].y = start_points[i].second;
            end_points_[i].x = end_points[i].first;
            end_points_[i].y = end_points[i].second;
        }

        int ret = bmcv_image_draw_lines(
            handle_.data(),
            img.data(),
            start_points_,
            end_points_,
            edge_num,
            color_,
            thickness);

        delete[] start_points_;
        delete[] end_points_;

        if (BM_SUCCESS != ret) {
            print_image(img.data(),"input");
            SPDLOG_ERROR("bmcv_image_draw_lines() err={}", ret);
            return SAIL_ERR_BMI_BMCV;
        }

        return BM_SUCCESS;
    }

bm_status_t open_water(
        bm_handle_t           handle,
        const char *          src_name,
        int                   src_size,
        bm_device_mem_t *     dst){
        bm_status_t ret = BM_SUCCESS;
        unsigned char * src = new unsigned char [src_size];

        FILE * fp_src = fopen(src_name, "rb+");
        size_t read_size = fread((void *)src, src_size, 1, fp_src);
        printf("fread %ld byte\n", read_size);
        fclose(fp_src);

        ret = bm_malloc_device_byte(handle, dst, src_size);
        if(ret != BM_SUCCESS){
            printf("bm_malloc_device_byte fail %s: %s: %d\n", __FILE__, __func__, __LINE__);
            goto fail;
        }
        ret = bm_memcpy_s2d(handle, dst[0], src);
        if(ret != BM_SUCCESS){
            printf("bm_memcpy_s2d fail %s: %s: %d\n", __FILE__, __func__, __LINE__);
        }
    fail:
        delete [] src;
        return ret;
    }
    int Bmcv::watermark_superpose(
        BMImage &img,
        string water_name,
        int bitmap_type,
        int pitch,
        vector<vector<int>> rects,
        vector<int> color){
        if(img.format() != FORMAT_GRAY &&
            img.format() != FORMAT_YUV420P &&
            img.format() != FORMAT_YUV444P &&
            img.format() != FORMAT_NV12 &&
            img.format() != FORMAT_NV21 &&
            img.format() != FORMAT_RGB_PLANAR &&
            img.format() != FORMAT_BGR_PLANAR &&
            img.format() != FORMAT_RGB_PACKED &&
            img.format() != FORMAT_BGR_PACKED &&
            img.format() != FORMAT_RGBP_SEPARATE &&
            img.format() != FORMAT_BGRP_SEPARATE){
            SPDLOG_ERROR("input format not supported!");
            print_image(img.data(),"input");
            return BM_ERR_FAILURE;
        }
        if(img.dtype() != DATA_TYPE_EXT_1N_BYTE){
            SPDLOG_ERROR("input dtype not supported!");
            print_image(img.data(),"input");
            return BM_ERR_FAILURE;
        }
        bmcv_rect_t *rects_tmp=new bmcv_rect_t[rects.size()];
        int water_byte,water_w=0,water_h=0;
        for(int i=0;i<rects.size();i++){
            if (rects[i].size() > 4) {
                // 如果内部vector的大小大于3，删除多余的元素
                rects[i].resize(4);
            } else {
                // 如果内部vector的大小小于3，添加零直到大小为3
                while (rects[i].size() < 4) {
                    rects[i].push_back(0);
                }
            }
            rects_tmp->start_x=rects[i][0];
            rects_tmp->start_y=rects[i][1];
            rects_tmp->crop_w=rects[i][2];
            rects_tmp->crop_h=rects[i][3];
            if(!water_w){
                water_w=rects[i][2];
                water_h=rects[i][3];
                water_byte=water_w*water_h;
            }else{
                if(water_w!=rects[i][2] || water_h!=rects[i][3]){
                    SPDLOG_ERROR("watermark_superpose witdh and height must be same");
                    return BM_ERR_FAILURE;
                }

            }
            

            rects_tmp++;
        }    
        rects_tmp-=rects.size();
        if (color.size() > 3) {
            // 如果内部vector的大小大于3，删除多余的元素
            color.resize(3);
        } else {
            // 如果内部vector的大小小于3，添加零直到大小为3
            while (color.size() < 3) {
                color.push_back(0);
            }
        }

        bmcv_color_t color_tmp={color[0],color[1],color[2]};
        bm_device_mem_t water;
        if(bitmap_type == 0){
            if(open_water(handle_.data(), water_name.c_str(), water_byte, &water)!=BM_SUCCESS){
                SPDLOG_ERROR("watermark_superpose open error");
                return BM_ERR_FAILURE;
            }
        }
        if(bitmap_type == 1){
            water_byte = water_byte / 8;
            water_w = water_w / 8;
            if(open_water(handle_.data(), water_name.c_str(), water_byte, &water)!=BM_SUCCESS){
                SPDLOG_ERROR("watermark_superpose open error");
                return BM_ERR_FAILURE;
            }
        }

        
        int ret;
        ret = bmcv_image_watermark_repeat_superpose(handle_.data(),img.data(),water,rects.size(),bitmap_type,pitch,rects_tmp,color_tmp);

        
        return ret;
    }    

    int Bmcv::mosaic(
        int mosaic_num,
        BMImage &img,
        vector<vector<int>> rects,
        int is_expand){
        
        if(img.format() != FORMAT_GRAY &&
            img.format() != FORMAT_YUV420P &&
            img.format() != FORMAT_YUV444P &&
            img.format() != FORMAT_NV12 &&
            img.format() != FORMAT_NV21 &&
            img.format() != FORMAT_RGB_PLANAR &&
            img.format() != FORMAT_BGR_PLANAR &&
            img.format() != FORMAT_RGB_PACKED &&
            img.format() != FORMAT_BGR_PACKED &&
            img.format() != FORMAT_RGBP_SEPARATE &&
            img.format() != FORMAT_BGRP_SEPARATE){
            SPDLOG_ERROR("input format not supported!");
            print_image(img.data(),"input");
            return BM_ERR_FAILURE;
        }
        if(img.dtype() != DATA_TYPE_EXT_1N_BYTE){
            SPDLOG_ERROR("input data type not supported!");
            print_image(img.data(),"input");
        }
        if(rects.size()==0) SPDLOG_ERROR("Rects is NULL");

        
        bmcv_rect_t* p_rects=new bmcv_rect_t[rects.size()];
        for(int i=0;i<rects.size();i++){
                    p_rects[i].start_x=rects[i][0];
                    p_rects[i].start_y=rects[i][1];
                    p_rects[i].crop_w=rects[i][2];
                    p_rects[i].crop_h=rects[i][3];
                    if(p_rects[i].crop_w<=8||p_rects[i].crop_h<=8)
                        SPDLOG_ERROR("input mosaic witch or height <= 8, not supported!");
        }
        
        int ret = bmcv_image_mosaic(handle_.data(), mosaic_num, img.data(), p_rects, is_expand);
        if (BM_SUCCESS != ret) {
            print_image(img.data(),"input");
            SPDLOG_ERROR("bmcv_mosaic() err={}", ret);
            return ret;
        }
        return BM_SUCCESS;
    }

    int Bmcv::gaussian_blur( 
            BMImage &input,
            BMImage &output,
            int kw,
            int kh,
            float sigmaX,
            float sigmaY
    ){
        if(input.format() != FORMAT_BGR_PACKED &&
            input.format() != FORMAT_BGR_PLANAR &&
            input.format() != FORMAT_RGB_PACKED &&
            input.format() != FORMAT_RGB_PLANAR &&
            input.format() != FORMAT_RGBP_SEPARATE &&
            input.format() != FORMAT_BGRP_SEPARATE &&
            input.format() != FORMAT_YUV422P &&
            input.format() != FORMAT_YUV444P &&
            input.format() != FORMAT_YUV420P &&
            input.format() != FORMAT_GRAY ){
            SPDLOG_ERROR("input format not supported!");
            print_image(input.data(),"input");
            return BM_ERR_FAILURE;
        }

        if(input.dtype() != DATA_TYPE_EXT_1N_BYTE){
            SPDLOG_ERROR("input data type not supported!");
            print_image(input.data(),"input");
            return BM_ERR_FAILURE;
        }

        if (input.width() <= 2048-kw && input.width() > 0){ // avoid width overflow or input img null
        }else{
            SPDLOG_ERROR("input image width not supported!");
            print_image(input.data(),"input");
            return BM_ERR_FAILURE;
        }

        if (output.is_created()) {
            output.reset(input.height(),input.width());
        }else{
            output.create(
                handle_,
                input.height(),
                input.width(),
                input.format(), // force to this format
                input.dtype()
            );
            output.allocate();
        }

        int ret = bm_image_alloc_contiguous_mem(1, &output.data());

        if (BM_SUCCESS != ret){
            SPDLOG_ERROR("in bmcv_image_gaussian_blur: bm_image_alloc_dev_mem err={}", ret);
            return BM_ERR_FAILURE;
        }
        ret = bmcv_image_gaussian_blur(handle_.data(), input.data(), output.data(), kw, kh, sigmaX, sigmaY);
        if (BM_SUCCESS != ret){
            SPDLOG_ERROR("bmcv_image_gaussian_blur err={}", ret);
            return BM_ERR_FAILURE;
        }
        
        return BM_SUCCESS;
    }


    BMImage Bmcv::gaussian_blur(
            BMImage &input,
            int kw,
            int kh,
            float sigmaX,
            float sigmaY
    ){
        BMImage output;
        int ret = gaussian_blur(input, output, kw, kh, sigmaX, sigmaY);
        if (ret != BM_SUCCESS){
            SPDLOG_ERROR("bmcv_image_gaussian_blur: bm_image_get_byte_size err={}", ret);
            throw SailBMImageError("bmcv api fail");
        } 
        return std::move(output);
    }
    

    int Bmcv::transpose(
        BMImage &src,
        BMImage &dst
    ){
        if(src.height() == 0 && src.width() == 0){
            SPDLOG_ERROR("input image is empty!");
            return SAIL_ERR_BMI_EMPTY;
        }
        if(src.format() != FORMAT_GRAY &&
            src.format() != FORMAT_RGB_PLANAR &&
            src.format() != FORMAT_BGR_PLANAR ){
            SPDLOG_ERROR("input format not supported, color_space of input only support RGB_PLANAR/BGR_PLANAR/GRAY!!");
            print_image(src.data(),"input");
            return BM_ERR_FAILURE;
        }
        if(src.dtype() != DATA_TYPE_EXT_1N_BYTE_SIGNED &&
            src.dtype() != DATA_TYPE_EXT_1N_BYTE_SIGNED &&
            src.dtype() != DATA_TYPE_EXT_FLOAT32 ){
#if (BMCV_VERSION_MAJOR == 2) && !defined(BMCV_VERSION_MINOR)
            if (src.dtype() != DATA_TYPE_EXT_1N_BYTE){
                SPDLOG_ERROR("Input dtype Error!");
                print_image(src.data(),"input");
                return BM_ERR_FAILURE;
            }
#else
            if (src.dtype() != DATA_TYPE_EXT_1N_BYTE && src.dtype() != DATA_TYPE_EXT_4N_BYTE){
                print_image(src.data(),"input");
                SPDLOG_ERROR("Input dtype Error!");
                return BM_ERR_FAILURE;
            }
#endif
        }

        if (dst.is_created()) {
            dst.reset(src.width(),src.height()); // output height must be input width, output width must be input height
        }else{
            dst.create(
                handle_,
                src.width(),
                src.height(),
                src.format(), // force to this format
                src.dtype()
            );
            dst.allocate();
        }
        int ret = bmcv_image_transpose(handle_.data(), src.data(), dst.data());
        if (BM_SUCCESS != ret){
            SPDLOG_ERROR("bmcv_transpose err={}", ret);
            return BM_ERR_FAILURE;
        }
        return ret;
    }

    BMImage Bmcv::transpose(
        BMImage &src
    ){
        BMImage dst;
        int ret = transpose(src,dst);
        if(ret != 0){
            SPDLOG_ERROR("bmcv_transpose err={}", ret);
            throw SailBMImageError("bmcv api fail");
        }
        return std::move(dst);
    }
    
    int Bmcv::Sobel(
        BMImage &input,
        BMImage &output,
        int dx,
        int dy,
        int ksize,
        float scale,
        float delta){
        if(input.width() == 0 && input.height() == 0){
            SPDLOG_ERROR("Sobel:input data is empty!");
            return SAIL_ERR_BMI_EMPTY;
        }
        if(input.dtype() != DATA_TYPE_EXT_1N_BYTE){
            SPDLOG_ERROR("input data type not supported!");
            print_image(input.data(),"input");
            return BM_ERR_FAILURE;
        }
        if(input.width() - ksize >= 2048){
            SPDLOG_ERROR("input data width must less than 2048-ksize!");
            print_image(input.data(),"input");
            return BM_ERR_FAILURE;
        }
        if(output.is_created()) {
            output.reset(input.height(),input.width()); // output height must be input width, output width must be input height
        }else{
            output.create(
                handle_,
                input.height(),
                input.width(),
                input.format(), // force to this format
                input.dtype()
            );
            output.allocate();
        }
        int ret = bmcv_image_sobel(handle_.data(), input.data(), output.data(), dx, dy, ksize, scale, delta);
        if (BM_SUCCESS != ret){
            SPDLOG_ERROR("bmcv_Sobel err={}", ret);
            return BM_ERR_FAILURE;
        }
        return ret;
    }

    BMImage Bmcv::Sobel(
        BMImage &input,
        int dx,
        int dy,
        int ksize,
        float scale,
        float delta){
        BMImage output;
        int ret = Sobel(input, output, dx, dy, ksize, scale, delta);
        if (BM_SUCCESS != ret) {
            print_image(input.data(),"input");
            SPDLOG_ERROR("bmcv_Sobel() err={}", ret);
            throw SailBMImageError("bmcv api fail");
        }
        return std::move(output);
    }

    // Function to convert AVFrame to bm_image
    int avframe_to_bm_image(Handle& handle_,AVFrame &in, bm_image &out) {

        int plane                 = 0;
        int data_five_denominator = -1;
        int data_six_denominator  = -1;
        static int mem_flags = USEING_MEM_HEAP1;

        // Switch case to handle different formats
        switch(in.format){
            case AV_PIX_FMT_GRAY8:
                plane = 1;
                data_five_denominator = -1;
                data_six_denominator = -1;
                break;
            case AV_PIX_FMT_YUV420P:
                plane = 3;
                data_five_denominator = 4;
                data_six_denominator = 4;
                break;
            case AV_PIX_FMT_NV12:
                plane = 2;
                data_five_denominator = 2;
                data_six_denominator = -1;
                break;
            case AV_PIX_FMT_YUV422P:
                plane = 3;
                data_five_denominator = 2;
                data_six_denominator = 2;
                break;
            case AV_PIX_FMT_NV16:
                plane = 2;
                data_five_denominator = 2;
                data_six_denominator = -1;
                break;
            case AV_PIX_FMT_YUV444P:
            case AV_PIX_FMT_GBRP:
                plane = 3;
                data_five_denominator = 1;
                data_six_denominator = 1;
                break;
            default:
                printf("unsupported format, only gray,nv12,yuv420p,nv16,yuv422p horizontal,yuv444p,rgbp supported\n");
                break;
        }

        // Handle compressed NV12 format
        if (in.channel_layout == 101) {/* COMPRESSED NV12 FORMAT */
            if ((0 == in.height) || (0 == in.width) || \
                (0 == in.linesize[4]) || (0 == in.linesize[5]) || (0 == in.linesize[6]) || (0 == in.linesize[7]) || \
                (0 == in.data[4]) || (0 == in.data[5]) || (0 == in.data[6]) || (0 == in.data[7])) {
                printf("bm_image_from_frame: get yuv failed!!");
                return BM_ERR_PARAM;
            }
            bm_image cmp_bmimg;
            bm_image_create (handle_.data(),
                            in.height,
                            in.width,
                            FORMAT_COMPRESSED,
                            DATA_TYPE_EXT_1N_BYTE,
                            &cmp_bmimg);

            bm_device_mem_t input_addr[4];
            int size = in.height * in.linesize[4];
            input_addr[0] = bm_mem_from_device((unsigned long long)in.data[6], size);
            size = (in.height / 2) * in.linesize[5];
            input_addr[1] = bm_mem_from_device((unsigned long long)in.data[4], size);
            size = in.linesize[6];
            input_addr[2] = bm_mem_from_device((unsigned long long)in.data[7], size);
            size = in.linesize[7];
            input_addr[3] = bm_mem_from_device((unsigned long long)in.data[5], size);
            bm_image_attach(cmp_bmimg, input_addr);
            bm_image_create (handle_.data(),
                            in.height,
                            in.width,
                            FORMAT_YUV420P,
                            DATA_TYPE_EXT_1N_BYTE,
                            &out);
            //bm_image_dev_mem_alloc(out);
#if (BMCV_VERSION_MAJOR == 2) && !defined(BMCV_VERSION_MINOR)
            if(mem_flags == USEING_MEM_HEAP2 && bm_image_alloc_dev_mem_heap_mask(out,USEING_MEM_HEAP2) != BM_SUCCESS){
                mem_flags = USEING_MEM_HEAP1;
            }   
#endif
            if(mem_flags == USEING_MEM_HEAP1 && bm_image_alloc_dev_mem_heap_mask(out,USEING_MEM_HEAP1) != BM_SUCCESS){
                printf("bmcv allocate mem failed!!!");
            }

            bmcv_rect_t crop_rect = {0, 0, in.width, in.height};
            bmcv_image_vpp_convert(handle_.data(), 1, cmp_bmimg, &out, &crop_rect);
            bm_image_destroy(cmp_bmimg);
        }
        else {
            int stride[3];
            bm_image_format_ext bm_format;
            bm_device_mem_t input_addr[3] = {0};
            if(plane == 1){
                if ((0 == in.height) || (0 == in.width) ||(0 == in.linesize[4]) || (0 == in.data[4])) {
                    return BM_ERR_PARAM;
                }
                stride[0] = in.linesize[4];
            }
            else if (plane == 2){
                if ((0 == in.height) || (0 == in.width) || \
                (0 == in.linesize[4]) || (0 == in.linesize[5]) || \
                (0 == in.data[4]) || (0 == in.data[5])) {
                    return BM_ERR_PARAM;
                }

                stride[0] = in.linesize[4];
                stride[1] = in.linesize[5];
                printf("====stride[0][1]=%d %d width=%d \n", stride[0], stride[1], in.width);
            }
            else if(plane == 3){
                if ((0 == in.height) || (0 == in.width) || \
                (0 == in.linesize[4]) || (0 == in.linesize[5]) || (0 == in.linesize[6]) || \
                (0 == in.data[4]) || (0 == in.data[5]) || (0 == in.data[6])) {
                    return BM_ERR_PARAM;
                }

                stride[0] = in.linesize[4];
                stride[1] = in.linesize[5];
                stride[2] = in.linesize[6];
            }

            bm_format = (bm_image_format_ext)map_avformat_to_bmformat(in.format);
            bm_image_create (handle_.data(),
                            in.height,
                            in.width,
                            bm_format,
                            DATA_TYPE_EXT_1N_BYTE,
                            &out,
                            stride);

            int size = in.height * stride[0];
            input_addr[0] = bm_mem_from_device((unsigned long long)in.data[4], size);
            printf("==== size1=%d addr[0]=%d \n", size, input_addr[0].size);
            if(data_five_denominator != -1 ){
                size = in.height * stride[1] / data_five_denominator;
                input_addr[1] = bm_mem_from_device((unsigned long long)in.data[5], size);
                printf("==== size2=%d addr[1]=%d \n", size, input_addr[1].size);
            }
            if(data_six_denominator != -1){
                size = in.height * stride[2] / data_six_denominator;
                printf("==== size3=%d \n", size);
                input_addr[2] = bm_mem_from_device((unsigned long long)in.data[6], size);
            }
            bm_image_attach(out, input_addr);
        }
        return BM_SUCCESS;
    }

#ifdef PYTHON

    BMImage Bmcv::imdecode(pybind11::bytes jpeg_data){
        std::string str = (std::string)jpeg_data;
        return imdecode(str.c_str(), str.length());
    }

    pybind11::array_t<uint8_t> Bmcv::imencode(std::string& ext, bm_image &img)
    {
        cv::Mat mat;
        cv::bmcv::toMAT(&img, mat);
        std::vector<u_char> data;
        cv::imencode(ext.c_str(), mat, data);

        std::vector<pybind11::ssize_t> shape = { pybind11::ssize_t(data.size())};
        std::vector<pybind11::ssize_t> strides = {sizeof(uint8_t)};
        pybind11::buffer_info output_buf(data.data(), sizeof(uint8_t), pybind11::format_descriptor<uint8_t>::format(),
                                            1, shape, strides);
        return std::move(pybind11::array_t<uint8_t>(output_buf));
    }

    pybind11::array_t<uint8_t> Bmcv::imencode(std::string& ext, BMImage &img)
    {
        return imencode(ext, img.data());
    }

    pybind11::array_t<float> Bmcv::nms(pybind11::array_t<float> input_proposal, float threshold)
    {
        pybind11::module np = pybind11::module::import("numpy");  // like 'import numpy as np'
        pybind11::buffer_info buf = input_proposal.request();
        if(buf.ndim != 2){
            SPDLOG_ERROR("Input proposal dims must be 2");
            throw SailBMImageError("invalid argument");
        }
        int proposal_size = buf.shape[0];
        if (proposal_size > 56000){
            SPDLOG_ERROR("Input proposal max size is 56000");
            throw SailBMImageError("not supported");
        }
        if(buf.shape[1] != 5){
            SPDLOG_ERROR("Input proposal shape error, proposal must be [left,top,right,bottom,score]!");
            throw SailBMImageError("invalid argument");
        }
        if(buf.itemsize !=4 || buf.format != "f"){
            SPDLOG_ERROR("Type of Input proposal must be float32!");
            throw SailBMImageError("invalid argument");
        }
        face_rect_t *proposal_rand = (face_rect_t *)buf.ptr;
        if (!pybind11::detail::check_flags(input_proposal.ptr(), pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_)) {
            pybind11::array_t<float> arr_c = np.attr("ascontiguousarray")(input_proposal, "dtype"_a="float32");
            proposal_rand = (face_rect_t*)arr_c.request().ptr;
        }
        nms_proposal_t *output_proposal = new nms_proposal_t;
        bmcv_nms(handle_.data(),
            bm_mem_from_system(proposal_rand),
            proposal_size,
            threshold,
            bm_mem_from_system(output_proposal));
        int output_size = output_proposal->size;
        pybind11::list shape_temp;
        shape_temp.append(output_size);
        shape_temp.append(5);
        pybind11::array_t<float> arr = np.attr("zeros")(shape_temp, "dtype"_a="float32");
        memcpy((void *)arr.request().ptr, (void *)output_proposal->face_rect, output_size*5*sizeof(float)); 
        delete output_proposal;

        return std::move(arr);
    }
    
    int Decoder_RawStream::read_(pybind11::bytes data_bytes, bm_image& image, bool continueFrame){
        return _impl->read_(data_bytes, image, continueFrame);
    }

    int Decoder_RawStream::read(pybind11::bytes data_bytes, BMImage& image, bool continueFrame){
        return _impl->read(data_bytes, image, continueFrame);
    }
    
    int Decoder_RawStream::Decoder_RawStream_CC::read_(pybind11::bytes data_bytes, bm_image& image, bool continueFrame) {
    
        std::string data_str = pybind11::cast<std::string>(data_bytes);
        uint8_t* data_ptr = reinterpret_cast<uint8_t*>(const_cast<char*>(data_str.data()));
        int data_size = data_str.size();
        return read_(data_ptr, data_size, image, continueFrame);
    }

    int Decoder_RawStream::Decoder_RawStream_CC::read(pybind11::bytes data_bytes, BMImage& image, bool continueFrame) {
    
        std::string data_str = pybind11::cast<std::string>(data_bytes);
        uint8_t* data_ptr = reinterpret_cast<uint8_t*>(const_cast<char*>(data_str.data()));
        int data_size = data_str.size();
        return read(data_ptr, data_size, image, continueFrame);
    }

#endif // ! USE_BMCV

  int argmax(sail::Tensor& array)
  {
    const int array_len = std::accumulate(array.shape().begin(), array.shape().end(), 1, std::multiplies<int>());
    if(!array.sys_data()){
        SPDLOG_ERROR("argsort: Data is not on the system!");
        throw MemoryError("Data is not on the system!");
    }
    switch(array.dtype()){
        case BM_FLOAT32:{
            float* ptr = (float*)array.sys_data();
            return std::distance(ptr,std::max_element(ptr, ptr+array_len));
        }
            break;
        case BM_INT32:{
            int* ptr = (int*)array.sys_data();
            return std::distance(ptr,std::max_element(ptr, ptr+array_len));
        }
            break;
        case BM_INT8:{
            int8_t* ptr = (int8_t*)array.sys_data();
            return std::distance(ptr,std::max_element(ptr, ptr+array_len));
        }
            break;
        case BM_UINT8:{
            u_char* ptr = (u_char*)array.sys_data();
            return std::distance(ptr,std::max_element(ptr, ptr+array_len));
        }
            break;
        default:
            SPDLOG_ERROR("argsort: Data Type not support!");
            throw NotSupport("Data Type not support!");
    }
  }

  int argmin(sail::Tensor& array)
  {
    if(!array.sys_data()){
        SPDLOG_ERROR("argsort: Data is not on the system!");
        throw MemoryError("Data is not on the system!");
    }
    const int array_len = std::accumulate(array.shape().begin(), array.shape().end(), 1, std::multiplies<int>());
    switch(array.dtype()){
        case BM_FLOAT32:{
            float* ptr = (float*)array.sys_data();
            return std::distance(ptr,std::min_element(ptr, ptr+array_len));
        }
            break;
        case BM_INT32:{
            int* ptr = (int*)array.sys_data();
            return std::distance(ptr,std::min_element(ptr, ptr+array_len));
        }
            break;
        case BM_INT8:{
            int8_t* ptr = (int8_t*)array.sys_data();
            return std::distance(ptr,std::min_element(ptr, ptr+array_len));
        }
            break;
        case BM_UINT8:{
            u_char* ptr = (u_char*)array.sys_data();
            return std::distance(ptr,std::min_element(ptr, ptr+array_len));
        }
            break;
        default:
            SPDLOG_ERROR("argsort: Data Type not support!");
            throw NotSupport("Data Type not support!");
            break;
    }
  }
#endif // namespace sail
}  
