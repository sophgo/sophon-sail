编译安装指南
==============

..
 .. |ver| replace:: 3.9.3

源码目录结构
______________


源码的目录结构如下:

.. parsed-literal::

    └── SOPHON-SAIL
        ├── 3rdparty
        │   ├── json
        │   ├── prebuild
        │   ├── pybind11
        │   ├── pybind11_new
        │   ├── spdlog
        │   └── ...
        ├── cmake                   # Cmake Files
        │   ├── BM168x_ARM_PCIE
        │   ├── BM168x_LoongArch64
        │   ├── BM168x_RISCV
        │   ├── BM168x_SOC
        │   └── ...
        ├── docs                    # Documentation codes
        │   ├── common
        │   ├── source_common
        │   ├── source_en
        │   └── source_zh
        ├── include                 # Includes
        ├── pyis                    # Stub files
        ├── pysail-dev              # Files for secondary dev
        │   └── ...
        ├── python                  # Wheel codes
        ├── python_wheels           # Python Wheels
        │   ├── arm_pcie
        │   ├── loongarch
        │   ├── soc_BM1684_BM1684X
        │   └── soc_BM1688
        │   └── ...
        ├── sample                  # Sample files
        │   └── ...
        ├── src                     # Source codes
        │   └── ...
        └── ...



其中3rdparty主要包含了编译sail需要依赖的第三方的一些头文件; 
cmake中是编译用到的一些cmake文件; 
include是sail的一些头文件; 
pyis文件夹内包含了对Python接口的描述，可用于类型提示和静态检查; 
pysail-dev文件夹内包含了用于二次开发其他Python模块的头文件和库；
python文件夹内包含了以下各平台下面python wheel的打包代码及脚本; 
python_wheels文件夹内是一些预编译出来的wheel包,arm_pcie、loongarch、soc三个文件夹分别是对应的平台; 
sample文件夹内是一些示例程序; 
src文件夹下面是各接口的实现代码。


SAIL的编译参数
_________________

*注意: BM1688处理器仅支持soc相关编译选项,BM1684和BM1684X处理器无此限制。*


* ``BUILD_TYPE`` : 编译的类型,目前有pcie、soc、arm_pcie、loongarch、windows五种模式, pcie是编译在x86主机上可用的SAIL包,soc表示使用交叉编译的方式,在x86主机上编译soc上可用的SAIL包,arm_pcie表示使用交叉编译的方式,在x86主机上编译插有bm168x卡的arm主机上可用的SAIL包,loongarch表示使用交叉编译的方式,在x86主机上编译插有bm168x卡的LoongArch64架构主机上可用的SAIL包，windows表示编译插有bm168x卡的windows主机上可用的SAIL包。默认pcie。
   
* ``ONLY_RUNTIME`` : 编译结果是否只包含运行时,而不包含bmcv,sophon-ffmpeg,sophon-opencv,如果此编译选项为ON,则SAIL的编解码及Bmcv接口不可用,只有推理接口可用。默认OFF。
   
* ``INSTALL_PREFIX`` : 执行make install时的安装路径,pcie模式下默认“/opt/sophon”,与libsophon的安装路径一致,交叉编译模式下默认“build_soc”。
   
* ``PYTHON_EXECUTABLE`` : 编译使用的“python3”的路径名称(路径+名称),默认使用当前系统中默认的python3。
   
* ``CUSTOM_PY_LIBDIR`` : 编译使用的python3的动态库的路径(只包含路径),默认使用当前系统中默认python3的动态库目录。
   
* ``LIBSOPHON_BASIC_PATH`` : 交叉编译模式下,libsophon的路径,如果配置不正确则会编译失败。pcie模式下面此编译选项不生效。
   
* ``FFMPEG_BASIC_PATH`` : 交叉编译模式下,sophon-ffmpeg的路径,如果配置不正确,且ONLY_RUNTIME为“OFF”时会编译失败。pcie模式下面此编译选项不生效。
   
* ``OPENCV_BASIC_PATH`` : 交叉编译模式下,sophon-opencv的路径,如果配置不正确,且ONLY_RUNTIME为“OFF”时会编译失败。pcie模式下面此编译选项不生效。

* ``TOOLCHAIN_BASIC_PATH`` : 交叉编译模式下,交叉编译器的路径,目前只有在BUILD_TYPE为loongarch时生效。

* ``BUILD_PYSAIL`` : 编译结果是否包含python版SAIL,默认为为“ON”,包含python版本SAIL。

* ``TARGET_TYPE`` : windows下的编译类型,当前支持 "release" 模式。

* ``RUNTIME_LIB`` : windows下的库类型,当前支持 "MT" 模式。



.. include:: 3_build/3.1_build_python.rst

.. include:: 3_build/3.2_build_cpp.rst

.. include:: 3_build/3.3_build_manual.rst