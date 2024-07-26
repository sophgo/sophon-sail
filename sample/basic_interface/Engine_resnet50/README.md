# Engine_resnet50

## 目录
- [Engine\_resnet50](#engine_resnet50)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 准备数据](#2-准备数据)
  - [3. 程序运行](#3-程序运行)
    - [3.1 Python例程](#31-python例程)
    - [3.2 C++例程](#32-c例程)
      - [3.2.1 程序编译](#321-程序编译)
      - [3.2.2 运行测试](#322-运行测试)


## 1. 简介
本例程提供了使用sail.Engine运行resnet50算法示例，具体包括：
```
./python
├── resnet50_bmcv.py        # 使用sail.bmcv 预处理, sail.BMImage 和 sail.Tensor 存储中间数据
└── resnet50_opencv.py      # 使用 opencv 预处理
```

```
./cpp
├── resnet50_bmcv           # 使用 sail.bmcv 预处理, sail.BMImage 和 sail.Tensor 存储中间数据
│   ├── CMakeLists.txt
│   ├── main.cpp
│   ├── processor.cpp
│   ├── processor.h
│   └── README.md
└── resnet50_opencv         # 使用 opencv 预处理
    ├── CMakeLists.txt
    ├── main.cpp
    ├── processor.cpp
    ├── processor.h
    └── README.md
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


-  resnet50_bmcv.py
```bash
usage: resnet50_bmcv.py [--jpg_dir JPEG_DIR] [--bmodel_path BMODEL_PATH] [--device_id DEVICE_ID] 

--jpg_dir: 图片路径，默认为""
--bmodel_path: bmodel路径，默认为""
--device_id: 设备ID，用于指定使用的设备号，默认为0
```

-  resnet50_opencv.py
```bash
usage: resnet50_opencv.py [--jpg_dir JPEG_DIR] [--bmodel_path BMODEL_PATH] [--device_id DEVICE_ID] 

--jpg_dir: 图片路径，默认为""
--bmodel_path: bmodel路径，默认为""
--device_id: 设备ID，用于指定使用的设备号，默认为0
```

### 3.2 C++例程

C++程序运行前需要编译可执行文件

#### 3.2.1 程序编译

**x86/arm PCIe平台**

可以直接在PCIe平台上编译程序，以resnet50_bmcv为例：

```bash
cd resnet50_bmcv/cpp
mkdir build && cd build
cmake .. && make # 生成可执行程序
```
编译完成后，会生成可执行程序。

**SoC平台**

您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中。详情见[SOPHON-SAIL使用手册](https://doc.sophgo.com/sdk-docs/v23.09.01-lts/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#id13)。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
cd resnet50_bmcv/cpp
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
usage:./resnet50_bmcv.pcie [params]

    --jpg_dir (value:"../../datasets/imagenet_val_1k/img/ILSVRC2012_val_00000075.JPEG")
            图片路径
    --bmodel_path (value:"../../models/BM1684X/resnet50_int8_1b.bmodel")
            bmodel文件的路径
    --device_id (value:0)
            设备ID，用于指定使用的设备号，默认为0
```

1. **注意：** CPP传参与python不同，需要用等于号，例如`./resnet50_bmcv.pcie --jpg_dir=xxx`。


2. **注意：** 

若在SoC模式下执行报错:

```bash
error while loading shared libraries: libsail.so: cannot open shared object file: No such file or directory
```
请在SoC上设置如下环境变量:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
```
