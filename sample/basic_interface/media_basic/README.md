# media_basic 

## 目录
- [media\_basic](#media_basic)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 准备数据](#2-准备数据)
  - [3. 程序运行](#3-程序运行)
    - [3.1 Python例程](#31-python例程)
    - [3.2 C++例程](#32-c例程)
      - [3.2.1 程序编译](#321-程序编译)
      - [3.2.2 运行测试](#322-运行测试)


## 1. 简介
本例程提供了图片编解码示例，具体包括：
```
./python
├── multi-channel_decode.py  # 使用 sail.multiDecoder 多路视频解码
├── push_stream.py           # 使用 sail.Encoder, sail.Decoder 进行图片、视频编解码
└── video_decoder.py         # 使用 sail.Decoder 解码并使用 sail.BMImageArray4D 准备4batch图片
```

```
./cpp
├── CMakeLists.txt
└── video_decoder.cpp        # 使用 sail::Decoder 解码并使用 sail::BMImageArray<4> 准备4batch图片
```


## 2. 准备数据
本例程在scripts目录下提供了相关模型和数据集的下载脚本download.sh，您也可以自己准备数据以运行例程。

```
chmod +x ./scripts/*
./scripts/download.sh
```

## 3. 程序运行
在运行程序前您需要参照[SAIL编译及安装](../../../README.md#编译及安装)准备运行环境。
### 3.1 Python例程

python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。各程序参数说明如下：


-  multi-channel_decode.py
```bash
usage: multi-channel_decode.py [--dev_id DEV_ID] [--video_url VIDEO_URL] [--is_local IS_LOCAL] \
[--read_timeout READ_TIMEOUT] [--test_duration TEST_DURATION] [--channel_num CHANNEL_NUM] \
[--is_save_jpeg IS_SAVE_JPEG]

--dev_id: 设备ID，用于指定使用哪个设备进行解码，默认为0。
--video_url: 视频URL，可以是RTSP流或文件路径。默认为"../datasets/test_car_person_1080P.mp4"。
--is_local: 指定视频源是否为本地视频。True表示本地，False表示非本地。默认为True。
--read_timeout: 读取视频流的超时时间（秒）。如果在此时间内未读取到数据，则视为超时。默认为5秒。
--test_duration: 测试持续时间（秒），即脚本将运行解码测试的时间长度。默认为20秒。
--channel_num: 通道数，表示将同时解码的视频通道数量。默认为10。
--is_save_jpeg: 是否保存JPEG图片。True表示保存JPEG格式的帧，False表示不保存。默认为True。

```

-  push_stream.py
```bash
usage: push_stream.py [--input_file_path INPUT_FILE_PATH] [--output_path OUTPUT_PATH] \
[--device_id DEVICE_ID] [--compressed_nv12 COMPRESSED_NV12] [--height HEIGHT] [--width WIDTH] \
[--enc_fmt ENC_FMT] [--bitrate BITRATE] [--pix_fmt PIX_FMT] [--gop GOP] [--gop_preset GOP_PRESET] \
[--framerate FRAMERATE]

--input_file_path: 输入视频文件的路径，例如 "input_video.mp4"；
--output_path: 推流输出的路径，默认是 "./output"；
--device_id: 使用的设备ID，默认为 0；
--compressed_nv12: 是否使用压缩的NV12格式，True 或 False，默认为 True；
--height: 视频帧的高度，默认为 1080；
--width: 视频帧的宽度，默认为 1920；
--enc_fmt: 编码格式，例如 "h264_bm"；
--bitrate: 码率（单位: Kbps），默认为 2000；
--pix_fmt: 像素格式，例如 "NV12"；
--gop: 关键帧间隔，默认为 32；
--gop_preset: GOP预设值，默认为 2；
--framerate: 帧率，默认为 25帧每秒。

```


-  video_decoder.py 
```bash
usage: video_decoder.py [--input_file_path INPUT_FILE_PATH] [--device_id DEVICE_ID] [--get_4batch GET_4BATCH]

--input_file_path: 视频文件的路径或RTSP流的URL。默认值为"../datasets/test_car_person_1080P.mp4"。
--device_id: 用于解码视频的设备ID。默认值为0。
--get_4batch: 是否生成batch_size为4的帧数据，即bmimgarray4D数据。设置为True时，将进行此操作；默认为False。
```

### 3.2 C++例程

C++程序运行前需要编译可执行文件

#### 3.2.1 程序编译

**x86/arm PCIe平台**

可以直接在PCIe平台上编译程序：

```bash
cd cpp
mkdir build && cd build
cmake .. && make # 生成可执行程序
```
编译完成后，会生成可执行程序。

**SoC平台**

您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中。详情见[SOPHON-SAIL使用手册](https://doc.sophgo.com/sdk-docs/v23.09.01-lts/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#id13)。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
cd cpp
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..  
make
cd ..
```
编译完成后，会生成SoC可执行程序。

#### 3.2.2 运行测试

可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：

```bash
usage:./video_decoder.pcie [params]

      --dev_id (value:0)
              device id
      --input  (value:../datasets/test_car_person_1080P.mp4)
              input video path
      --get_4batch (value:False)
              get 4 batch frame 
```

1. **注意：** CPP传参与python不同，需要用等于号，例如`./video_decoder.pcie --video_path=xxx`。


2. **注意：** 

若在SoC模式下执行报错:

```bash
error while loading shared libraries: libsail.so: cannot open shared object file: No such file or directory
```
请在SoC上设置如下环境变量:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
```
