Compilation and Installation Guide
======================================

.. |ver| replace:: |version|

The Directory Structure of the Source Code
____________________________________________________


The directory structure of the source code is as follows:

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



Among them, 3rdparty mainly contains some header files of the third party on which sail needs to be compiled; 
cmake contains some cmake files used for compilation; 
include contains some header files of sail; 
pyis contains some stub files for Python interfaces; 
pysail-dev contains headers and libs for secondary developing other Python module;
python folder contains the packaging code and scripts of python whl for each platform; 
python_wheels contains come pre-compiled sail wheels; 
sample contains some code samples for developers; 
src folder contains the code of each interface.


SAIL Compilation and Installation
_____________________________________________________

*Note: BM1688 processor only supports soc related compilation options, BM1684 and BM1684X processors do not have this restriction.*

Compilation Parameters
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

* ``BUILD_TYPE`` : It is used to specify the type of compilation, currently there are three modes: pcie, soc and arm_pcie. pcie means to compile the SAIL package available on the x86 host, soc means to use cross-compilation to compile the SAIL package available on the soc of the x86 host. arm_pcie means to use cross-compilation to compile the SAIL package available on the x86 host with a bm168x card plugged in. The default is pcie.
   
* ``ONLY_RUNTIME`` : It is used to specify whether the compilation result contains only runtime but not bmcv, sophon-ffmpeg, or sophon-opencv; if this compilation option is ON, this SAIL encoding and decoding and the Bmcv interface are not available, and only the inference interface is available. The default is OFF.
   
* ``INSTALL_PREFIX`` : It is used to specify the installation path when executing make install, default "/opt/sophon" in pcie mode, same as the installation path of libsophon, default "build_soc" in cross-compile mode.
   
* ``PYTHON_EXECUTABLE`` : It is used to specify the path name (path+name) of "python3" for compilation; by default, the default python3 of the current system is used.
   
* ``CUSTOM_PY_LIBDIR`` : It is used to specify the path of the python3 dynamic library to be used for compilation (path only); by default, the current system's default python3 dynamic library directory is used.
   
* ``LIBSOPHON_BASIC_PATH`` : It is used to specify the path of libsophon in cross-compile mode; if the configuration is not correct, the compilation will fail. In pcie mode, this compilation option does not take effect.
   
* ``FFMPEG_BASIC_PATH`` : It is used to specify the path of sophon-ffmpeg in cross-compile mode; if the configuration is not correct and ONLY_RUNTIME is "OFF", the compilation will fail. In pcie mode, this compilation option does not take effect.
   
* ``OPENCV_BASIC_PATH`` : It is used to specify the path of sophon-opencv in cross-compile mode; if the configuration is not correct and ONLY_RUNTIME is "OFF", the compilation will fail. In pcie mode, this compilation option does not take effect.

* ``TOOLCHAIN_BASIC_PATH`` : It is used to specify the path of cross-compiler in cross-compile mode. Currently it only takes effect when BUILD_TYPE is loongarch.

* ``BUILD_PYSAIL`` : It is used to specify whether the compilation result contains the python version of SAIL. The default is "ON", which includes the python version of SAIL.



Compile dynamic libraries and header files that can be called by the C++ interface
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*Note: BM1688 and CV186AH processors only support SOC MODE chapter, BM1684 and BM1684X processors do not have this limitation.Note that for BM1688 or CV186AH, sophon-mw needs to be changed to sophon-media.*

PCIE MODE
:::::::::::

**Install libsophon, sophon-ffmpeg, sophon-opencv for SAIL**

The installation of libsophon, sophon-ffmpeg, and sophon-opencv can be found in the sophon's official documentation

**.Compiling SAIL with Multimedia Modules**

Compile SAIL with bmcv, sophon-ffmpeg, sophon-opencv using the default installation path

1. Download the SOPHON-SAIL source code, unpack it and go to its source code directory

2. Create the build folder "build" and go to the "build" folder

    .. parsed-literal::
        mkdir build && cd build   

3. Execute the compilation command

    .. parsed-literal::
        cmake -DBUILD_PYSAIL=OFF ..                                   
        make sail   

4. Install SAIL dynamic library and header files; the compiled result will be installed under the "/opt/sophon" directory

    .. parsed-literal::
        sudo make install   

**.Compiling SAIL without Multimedia Modules**

Compile SAIL without bmcv, sophon-ffmpeg, sophon-opencv using the default installation path

*The SAIL compiled in this way cannot use its Decoder, Bmcv, and other multimedia-related interfaces.*

1. Download the SOPHON-SAIL source code, unpack it and go to its source code directory

2. Create the build folder "build" and go to the "build" folder

    .. parsed-literal::
        mkdir build && cd build  

3. Execute the compilation command

    .. parsed-literal::
        mkdir build && cd build       

4. Install SAIL dynamic library and header files; the compiled result will be installed under the "/opt/sophon" directory

    .. parsed-literal::
        sudo make install  

SOC MODE
:::::::::::

**.Get the libsophon, sophon-ffmpeg, and sophon-opencv needed for cross-compilation**

*All compilation operations in this section are performed on the x86 host using cross-compilation.
The following examples choose to use libsophon version 0.4.1, sophon-ffmpeg version 0.4.1, and sophon-opencv version 0.4.1.*

1. Get "libsophon_soc_0.4.1_aarch64.tar.gz" from sophon's official website and unpack it

    .. parsed-literal::
        tar -xvf libsophon_soc_0.4.1_aarch64.tar.gz

The directory of libsophon after unpacking is "libsophon_soc_0.4.1_aarch64/opt/sophon/libsophon-0.4.1"

2. Get "sophon-mw-soc_0.4.1_aarch64.tar.gz" from sophon's official website and unpack it

    .. parsed-literal::
        tar -xvf sophon-mw-soc_0.4.1_aarch64.tar.gz

The directory of sophon-ffmpeg after unpacking is "sophon-mw-soc_0.4.1_aarch64/opt/sophon/sophon-ffmpeg_0.4.1".

The directory of sophon-opencv after unpacking is "sophon-mw-soc_0.4.1_aarch64/opt/sophon/sophon-opencv_0.4.1".


**.Install the gcc-aarch64-linux-gnu toolchain**

*If already installed, you can ignore this step*

    .. parsed-literal::
        sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

**.Compiling SAIL with Multimedia Modules**

Compile SAIL containing bmcv, sophon-ffmpeg, sophon-opencv through cross-compilation.

1. Download the SOPHON-SAIL source code, unpack it and go to its source code directory

2. Create the build folder "build" and go to the "build" folder

    .. parsed-literal::
        mkdir build && cd build 

3. Execute the compilation command

    .. code-block:: bash

        cmake -DBUILD_TYPE=soc -DBUILD_PYSAIL=OFF \
            -DCMAKE_TOOLCHAIN_FILE=../cmake/BM168x_SOC/ToolChain_aarch64_linux.cmake \
            -DLIBSOPHON_BASIC_PATH=libsophon_soc_0.4.1_aarch64/opt/sophon/libsophon-0.4.1 \
            -DFFMPEG_BASIC_PATH=sophon-mw-soc_0.4.1_aarch64/opt/sophon/sophon-ffmpeg_0.4.1 \
            -DOPENCV_BASIC_PATH=sophon-mw-soc_0.4.1_aarch64/opt/sophon/sophon-opencv_0.4.1 ..                                   
        make sail 

4. Install SAIL dynamic library and header files; The program will automatically create "build_soc" in the source code directory and the compiled result will be installed under "build_soc"

    .. parsed-literal::
        make install

5. Copy "sophon-sail" from the "build_soc" folder to the "/opt/sophon" directory on the target SOC, then can use SAIL on the target SOC host

**.Compiling SAIL without Multimedia Modules**

Compile SAIL that dose not include bmcv, sophon-ffmpeg, sophon-opencv through cross-compilation

*The SAIL compiled in this way cannot use its Decoder, Bmcv, and other multimedia-related interfaces.*

1. Download the SOPHON-SAIL source code, unpack it and go to its source code directory

2. Create the build folder "build" and go to the "build" folder

    .. parsed-literal::
        mkdir build && cd build 

3. Execute the compilation command

    .. code-block:: bash

        cmake -DBUILD_TYPE=soc  \
            -DBUILD_PYSAIL=OFF \
            -DONLY_RUNTIME=ON \
            -DCMAKE_TOOLCHAIN_FILE=../cmake/BM168x_SOC/ToolChain_aarch64_linux.cmake \
            -DLIBSOPHON_BASIC_PATH=libsophon_soc_0.4.1_aarch64/opt/sophon/libsophon-0.4.1 ..
        make sail   

4. Install SAIL dynamic library and header files; The program will automatically create "build_soc" in the source code directory and the compiled result will be installed under "build_soc"

    .. parsed-literal::
        make install    

5. Copy "sophon-sail" from the "build_soc" folder to the "/opt/sophon" directory on the target SOC, then can use SAIL on the target SOC host


ARM PCIE MODE
::::::::::::::::::

**.Get the libsophon, sophon-ffmpeg, and sophon-opencv needed for cross-compilation**

*All compilation operations in this section are performed on the x86 host using cross-compilation. The following examples choose to use libsophon version 0.4.1, sophon-ffmpeg version 0.4.1, and sophon-opencv version 0.4.1.*

1. Get "sophon-mw_0.4.1_aarch64.tar.gz" from sophon's official website and unpack it

    .. parsed-literal::
        tar -xvf libsophon_0.4.1_aarch64.tar.gz

The directory of libsophon after unpacking is "libsophon_0.4.1_aarch64/opt/sophon/libsophon-0.4.1"

2. Get "sophon-mw_0.4.1_aarch64.tar.gz" from sophon's official website and unpack it

    .. parsed-literal::
        tar -xvf sophon-mw_0.4.1_aarch64.tar.gz

The directory of sophon-ffmpeg after unpacking is "sophon-mw_0.4.1_aarch64/opt/sophon/sophon-ffmpeg_0.4.1".

The directory of sophon-opencv after unpacking is "sophon-mw_0.4.1_aarch64/opt/sophon/sophon-opencv_0.4.1".

**.Install the gcc-aarch64-linux-gnu toolchain**

*If already installed, you can ignore this step*

    .. parsed-literal::
        sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

**.Compiling SAIL with Multimedia Modules**

Compile SAIL containing bmcv, sophon-ffmpeg, sophon-opencv through cross-compilation.

1. Download the SOPHON-SAIL source code, unpack it and go to its source code directory

2. Create the build folder "build" and go to the "build" folder

    .. parsed-literal::
        mkdir build && cd build 

3. Execute the compilation command

    .. code-block:: bash

        cmake -DBUILD_TYPE=arm_pcie  \
            -DBUILD_PYSAIL=OFF \
            -DCMAKE_TOOLCHAIN_FILE=../cmake/BM168x_ARM_PCIE/ToolChain_aarch64_linux.cmake \
            -DLIBSOPHON_BASIC_PATH=libsophon_0.4.1_aarch64/opt/sophon/libsophon-0.4.1 \
            -DFFMPEG_BASIC_PATH=sophon-mw_0.4.1_aarch64/opt/sophon/sophon-ffmpeg_0.4.1 \
            -DOPENCV_BASIC_PATH=sophon-mw_0.4.1_aarch64/opt/sophon/sophon-opencv_0.4.1 ..                                   
        make sail   

4. Install SAIL dynamic library and header files; The program will automatically create "build_arm_pcie" in the source code directory and the compiled result will be installed under "build_arm_pcie"

    .. parsed-literal::
        make install

5. Copy "sophon-sail" from the "build_arm_pcie" folder to the "/opt/sophon" directory on the target ARM host, then can use SAIL on the target ARM host

**.Compiling SAIL without Multimedia Modules**

Compile SAIL that dose not include bmcv, sophon-ffmpeg, sophon-opencv through cross-compilation

*The SAIL compiled in this way cannot use its Decoder, Bmcv, and other multimedia-related interfaces.*

1. Download the SOPHON-SAIL source code, unpack it and go to its source code directory

2. Create the build folder "build" and go to the "build" folder

    .. parsed-literal::
        mkdir build && cd build

3. Execute the compilation command

    .. code-block:: bash

        cmake -DBUILD_TYPE=soc  \
            -DBUILD_PYSAIL=OFF \
            -DONLY_RUNTIME=ON \
            -DCMAKE_TOOLCHAIN_FILE=../cmake/BM168x_SOC/ToolChain_aarch64_linux.cmake \
            -DLIBSOPHON_BASIC_PATH=libsophon_soc_0.4.1_aarch64/opt/sophon/libsophon-0.4.1 ..
        make sail    

4. Install SAIL dynamic library and header files; The program will automatically create "build_arm_pcie" in the source code directory and the compiled result will be installed under "build_arm_pcie"

    .. parsed-literal::
        make install

5. Copy "sophon-sail" from the "build_arm_pcie" folder to the "/opt/sophon" directory on the target ARM host, then can use SAIL on the target ARM host

LOONGARCH64 MODE
::::::::::::::::::::

**.Install the loongarch64-linux-gnu toolchain**

Get the [cross-compiled toolchain](http://ftp.loongnix.cn/toolchain/gcc/release/loongarch/gcc8/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1.tar.xz) from the LoongArch64 official website, 
and unzip it locally. The directory structure after decompression is as follows:

.. parsed-literal::

    └── loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1
        ├── bin
        ├── lib
        ├── lib64
        ├── libexec
        ├── loongarch64-linux-gnu
        ├── share
        ├── sysroot
        └── versions 

**.Get the libsophon, sophon-ffmpeg, and sophon-opencv needed for cross-compilation**

*All compilation operations in this section are performed on the x86 host using cross-compilation. The following examples choose to use libsophon version 0.4.7, sophon-ffmpeg version 0.6.0, and sophon-opencv version 0.6.0.*

**.Compiling SAIL with Multimedia Modules**

Compile SAIL containing bmcv, sophon-ffmpeg, sophon-opencv through cross-compilation.

1. Download the SOPHON-SAIL source code, unpack it and go to its source code directory

2. Create the build folder "build" and go to the "build" folder

    .. parsed-literal::
        mkdir build && cd build 

3. Execute the compilation command

    .. parsed-literal::
        cmake -DBUILD_TYPE=loongarch  \
            -DBUILD_PYSAIL=OFF \
            -DTOOLCHAIN_BASIC_PATH=toolchains/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1 \
            -DCMAKE_TOOLCHAIN_FILE=../cmake/BM168x_LoongArch64/ToolChain_loongarch64_linux.cmake \
            -DLIBSOPHON_BASIC_PATH=libsophon_0.4.7_loongarch64/opt/sophon/libsophon-0.4.7 \
            -DFFMPEG_BASIC_PATH=sophon-mw_0.6.0_loongarch64/opt/sophon/sophon-ffmpeg_0.6.0 \
            -DOPENCV_BASIC_PATH=sophon-mw_0.6.0_loongarch64/opt/sophon/sophon-opencv_0.6.0  \
            ..
        make sail 

4. Install SAIL dynamic library and header files; The program will automatically create "build_loongarch" in the source code directory and the compiled result will be installed under "build_loongarch"

    .. parsed-literal::
        make install

5. Copy "sophon-sail" from the "build_loongarch" folder to the "/opt/sophon" directory on the target LOONGARCH host, then can use SAIL on the target LOONGARCH host

**.Compiling SAIL without Multimedia Modules**

Compile SAIL that dose not include bmcv, sophon-ffmpeg, sophon-opencv through cross-compilation

*The SAIL compiled in this way cannot use its Decoder, Bmcv, and other multimedia-related interfaces.*

1. Download the SOPHON-SAIL source code, unpack it and go to its source code directory

2. Create the build folder "build" and go to the "build" folder

    .. parsed-literal::
        mkdir build && cd build 

3. Execute the compilation command

    .. code-block:: bash

        cmake -DBUILD_TYPE=loongarch  \
            -DBUILD_PYSAIL=OFF \
            -DONLY_RUNTIME=ON \
            -DTOOLCHAIN_BASIC_PATH=toolchains/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1 \
            -DCMAKE_TOOLCHAIN_FILE=../cmake/BM168x_LoongArch64/ToolChain_loongarch64_linux.cmake \
            -DLIBSOPHON_BASIC_PATH=libsophon_0.4.7_loongarch64/opt/sophon/libsophon-0.4.7 \
            ..
        make sail

4. Install SAIL dynamic library and header files; The program will automatically create "build_loongarch" in the source code directory and the compiled result will be installed under "build_loongarch"

    .. parsed-literal::
        make install 

5. Copy "sophon-sail" from the "build_loongarch" folder to the "/opt/sophon" directory on the target LOONGARCH host, then can use SAIL on the target LOONGARCH host


Compile dynamic libraries and header files that can be called by the Python interface
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

*Note: BM1688 and CV186AH processors only support SOC MODE chapter, BM1684 and BM1684X processors do not have this limitation.Note that for BM1688 or CV186AH, sophon-mw needs to be changed to sophon-media.*

PCIE MODE
:::::::::::

**Install libsophon, sophon-ffmpeg, sophon-opencv for SAIL**

The installation of libsophon, sophon-ffmpeg, and sophon-opencv can be found in the sophon's official documentation

**.Compiling SAIL with Multimedia Modules**

Compile SAIL with bmcv, sophon-ffmpeg, sophon-opencv using the default installation path

*If you don't need to use the python interface, you can ignore sections 5 and 6*

1. Download the SOPHON-SAIL source code, unpack it and go to its source code directory

2. Create the build folder "build" and go to the "build" folder

    .. parsed-literal::
        mkdir build && cd build                   

3. Execute the compilation command

    .. parsed-literal::
        cmake ..                                   
        make pysail                                                                

4. Pack python wheel, the path of the generated wheel package is "python/dist" and the file name is "sophon-|ver|-py3-none-any.whl"

    .. parsed-literal::
        cd ../python 
        chmod +x sophon_whl.sh
        ./sophon_whl.sh  

5. Install python wheel  

    .. parsed-literal::
        pip3 install ./dist/sophon-|ver|-py3-none-any.whl --force-reinstall 

**.Compiling SAIL without Multimedia Modules**

Compile SAIL without bmcv, sophon-ffmpeg, sophon-opencv using the default installation path

*The SAIL compiled in this way cannot use its Decoder, Bmcv, and other multimedia-related interfaces.*

1. Download the SOPHON-SAIL source code, unpack it and go to its source code directory

2. Create the build folder "build" and go to the "build" folder

    .. parsed-literal::
        mkdir build && cd build                   

3. Execute the compilation command

    .. parsed-literal::
        cmake -DONLY_RUNTIME=ON ..                                   
        make pysail                                      

4. Pack python wheel, the path of the generated wheel package is "python/dist" and the file name is "sophon-|ver|-py3-none-any.whl"

    .. parsed-literal::
        cd ../python
        chmod +x sophon_whl.sh
        ./sophon_whl.sh  

5. Install python wheel  

    .. parsed-literal::
        pip3 install ./dist/sophon-|ver|-py3-none-any.whl --force-reinstall 


**.Compiling SAIL with a Specific Python Version**

If the python3 version in the production environment is not the same as the development environment, you can make it consistent by upgrading the python3 version.
You can also get the corresponding python3 package through the official python3 website.
Or you can download the already compiled python3 from [:ref:`Get Python3 for cross-compilation on the X86 host`].
That is, use the non-system default python3, compile SAIL containing bmcv, sophon-ffmpeg, and sophon-opencv, and package it in the "build_pcie" directory.
The path of python3 used in this example is "python_3.8.2/bin/python3", and the dynamic library directory of python3 is "python_3.8.2/lib".

1. Download the SOPHON-SAIL source code, unpack it and go to its source code directory

2. Create the build folder "build" and go to the "build" folder

    .. parsed-literal::
        mkdir build && cd build                   

3. Execute the compilation command

    .. parsed-literal::
        cmake -DPYTHON_EXECUTABLE=python_3.8.2/bin/python3 -DCUSTOM_PY_LIBDIR=python_3.8.2/lib ..                               
        make pysail                                       

4. Pack python wheel, the path of the generated wheel package is "python/dist" and the file name is "sophon-|ver|-py3-none-any.whl"

    .. parsed-literal::
        cd ../python 
        chmod +x sophon_whl.sh
        ./sophon_whl.sh  

7. Install python wheel  

Copy "sophon-|ver|-py3-none-any.whl" to the target machine, then execute the following installation command

    .. parsed-literal::
        pip3 install ./dist/sophon-|ver|-py3-none-any.whl --force-reinstall 

SOC MODE
>>>>>>>>>>>>>>>>>>>>>>>>>>

**.Get the libsophon, sophon-ffmpeg, and sophon-opencv needed for cross-compilation**

*All compilation operations in this section are performed on the x86 host using cross-compilation. 
The following examples choose to use libsophon version 0.4.1, sophon-ffmpeg version 0.4.1, and sophon-opencv version 0.4.1.*

1. Get "libsophon_soc_0.4.1_aarch64.tar.gz" from sophon's official website and unpack it

    .. parsed-literal::
        tar -xvf libsophon_soc_0.4.1_aarch64.tar.gz

The directory of libsophon after unpacking is "libsophon_soc_0.4.1_aarch64/opt/sophon/libsophon-0.4.1"

2. Get "sophon-mw-soc_0.4.1_aarch64.tar.gz" from sophon's official website and unpack it

    .. parsed-literal::
        tar -xvf sophon-mw-soc_0.4.1_aarch64.tar.gz

The directory of sophon-ffmpeg after unpacking is "sophon-mw-soc_0.4.1_aarch64/opt/sophon/sophon-ffmpeg_0.4.1".

The directory of sophon-opencv after unpacking is "sophon-mw-soc_0.4.1_aarch64/opt/sophon/sophon-opencv_0.4.1".


**.Install the gcc-aarch64-linux-gnu toolchain**

*If already installed, you can ignore this step*

    .. parsed-literal::
        sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

**.Compiling SAIL with Multimedia Modules**

Compile SAIL with bmcv, sophon-ffmpeg, and sophon-opencv by cross-compiling using the specified version of python3 (consistent with the version of python3 on the target SOC).
You can also get the corresponding python3 package through the official python3 website.
Or you can download the already compiled python3 from [:ref:`Get Python3 for cross-compilation on the X86 host`].
The path of python3 used in this example is "python_3.8.2/bin/python3", and the dynamic library directory of python3 is "python_3.8.2/lib".

*If you don't need to use the python interface, you can ignore sections 6 and 7*

1. Download the SOPHON-SAIL source code, unpack it and go to its source code directory

2. Create the build folder "build" and go to the "build" folder

    .. parsed-literal::
        mkdir build && cd build 

3. Execute the compilation command

    .. code-block:: bash

        cmake -DBUILD_TYPE=soc  \
            -DCMAKE_TOOLCHAIN_FILE=../cmake/BM168x_SOC/ToolChain_aarch64_linux.cmake \
            -DPYTHON_EXECUTABLE=python_3.8.2/bin/python3 \
            -DCUSTOM_PY_LIBDIR=python_3.8.2/lib \
            -DLIBSOPHON_BASIC_PATH=libsophon_soc_0.4.1_aarch64/opt/sophon/libsophon-0.4.1 \
            -DFFMPEG_BASIC_PATH=sophon-mw-soc_0.4.1_aarch64/opt/sophon/sophon-ffmpeg_0.4.1 \
            -DOPENCV_BASIC_PATH=sophon-mw-soc_0.4.1_aarch64/opt/sophon/sophon-opencv_0.4.1 ..                                   
        make pysail                                  

4. Pack python wheel, the path of the generated wheel package is "python/dist" and the file name is "sophon_arm-|ver|-py3-none-any.whl"

    .. parsed-literal::
        cd ../python 
        chmod +x sophon_whl.sh
        ./sophon_whl.sh  

5. Install python wheel  

Copy "sophon_arm-|ver|-py3-none-any.whl" to the target SOC, then execute the following installation command

    .. parsed-literal::
        pip3 install sophon_arm-|ver|-py3-none-any.whl --force-reinstall 

**.Compiling SAIL without Multimedia Modules**

Compile SAIL without bmcv, sophon-ffmpeg, and sophon-opencv by cross-compiling using the specified version of python3 (consistent with python3 on the target SOC).
You can also get the corresponding python3 package through the official python3 website.
Or you can download the already compiled python3 from [:ref:`Get Python3 for cross-compilation on the X86 host`].
The path of python3 used in this example is "python_3.8.2/bin/python3", and the dynamic library directory of python3 is "python_3.8.2/lib".

*The SAIL compiled in this way cannot use its Decoder, Bmcv, and other multimedia-related interfaces.*

1. Download the SOPHON-SAIL source code, unpack it and go to its source code directory

2. Create the build folder "build" and go to the "build" folder

    .. parsed-literal::
        mkdir build && cd build 

3. Execute the compilation command

    .. code-block:: bash

        cmake -DBUILD_TYPE=soc  \
            -DONLY_RUNTIME=ON \
            -DCMAKE_TOOLCHAIN_FILE=../cmake/BM168x_SOC/ToolChain_aarch64_linux.cmake \
            -DPYTHON_EXECUTABLE=python_3.8.2/bin/python3 \
            -DCUSTOM_PY_LIBDIR=python_3.8.2/lib \
            -DLIBSOPHON_BASIC_PATH=libsophon_soc_0.4.1_aarch64/opt/sophon/libsophon-0.4.1 ..
        make pysail                                   

4. Pack python wheel, the path of the generated wheel package is "python/dist" and the file name is "sophon_arm-|ver|-py3-none-any.whl"

    .. parsed-literal::
        cd ../python 
        chmod +x sophon_whl.sh
        ./sophon_whl.sh  

5. Install python wheel  

Copy "sophon_arm-|ver|-py3-none-any.whl" to the target SOC, then execute the following installation command

    .. parsed-literal::
        pip3 install sophon_arm-|ver|-py3-none-any.whl --force-reinstall 
 
 
ARM PCIE MODE
>>>>>>>>>>>>>>

**.Get the libsophon, sophon-ffmpeg, and sophon-opencv needed for cross-compilation**

*All compilation operations in this section are performed on the x86 host using cross-compilation. The following examples choose to use libsophon version 0.4.1, sophon-ffmpeg version 0.4.1, and sophon-opencv version 0.4.1.*

1. Get "libsophon_0.4.1_aarch64.tar.gz" from sophon's official website and unpack it

    .. parsed-literal::
        tar -xvf libsophon_0.4.1_aarch64.tar.gz

The directory of libsophon after unpacking is "libsophon_0.4.1_aarch64/opt/sophon/libsophon-0.4.1"

2. Get "sophon-mw_0.4.1_aarch64.tar.gz" from sophon's official website and unpack it

    .. parsed-literal::
        tar -xvf sophon-mw_0.4.1_aarch64.tar.gz

The directory of sophon-ffmpeg after unpacking is "sophon-mw_0.4.1_aarch64/opt/sophon/sophon-ffmpeg_0.4.1".

The directory of sophon-opencv after unpacking is "sophon-mw_0.4.1_aarch64/opt/sophon/sophon-opencv_0.4.1".


**.Install the gcc-aarch64-linux-gnu toolchain**

*If already installed, you can ignore this step*

    .. parsed-literal::
        sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

**.Compiling SAIL with Multimedia Modules**

Compile SAIL with bmcv, sophon-ffmpeg, and sophon-opencv by cross-compiling using the specified version of python3 (consistent with the version of python3 on the target ARM host).
You can also get the corresponding python3 package through the official python3 website.
Or you can download the already compiled python3 from [:ref:`Get Python3 for cross-compilation on the X86 host`].
The path of python3 used in this example is "python_3.8.2/bin/python3", and the dynamic library directory of python3 is "python_3.8.2/lib".

1. Download the SOPHON-SAIL source code, unpack it and go to its source code directory

2. Create the build folder "build" and go to the "build" folder

    .. parsed-literal::
        mkdir build && cd build 

3. Execute the compilation command

    .. code-block:: bash

        cmake -DBUILD_TYPE=arm_pcie  \
            -DCMAKE_TOOLCHAIN_FILE=../cmake/BM168x_ARM_PCIE/ToolChain_aarch64_linux.cmake \
            -DPYTHON_EXECUTABLE=python_3.8.2/bin/python3 \
            -DCUSTOM_PY_LIBDIR=python_3.8.2/lib \
            -DLIBSOPHON_BASIC_PATH=libsophon_0.4.1_aarch64/opt/sophon/libsophon-0.4.1 \
            -DFFMPEG_BASIC_PATH=sophon-mw_0.4.1_aarch64/opt/sophon/sophon-ffmpeg_0.4.1 \
            -DOPENCV_BASIC_PATH=sophon-mw_0.4.1_aarch64/opt/sophon/sophon-opencv_0.4.1 ..                                   
        make pysail                                   

4. Pack python wheel, the path of the generated wheel package is "python/dist" and the file name is "sophon_arm_pcie-|ver|-py3-none-any.whl"

    .. parsed-literal::
        cd ../python 
        chmod +x sophon_whl.sh
        ./sophon_whl.sh  

5. Install python wheel  

Copy "sophon_arm_pcie-|ver|-py3-none-any.whl" to the target ARM host, then execute the following installation command

    .. parsed-literal::
        pip3 install sophon_arm_pcie-|ver|-py3-none-any.whl --force-reinstall 

**.Compiling SAIL without Multimedia Modules**

Compile SAIL without bmcv, sophon-ffmpeg, and sophon-opencv by cross-compiling using the specified version of python3 (consistent with python3 on the target ARM host).
You can also get the corresponding python3 package through the official python3 website.
Or you can download the already compiled python3 from [:ref:`Get Python3 for cross-compilation on the X86 host`]..
The path of python3 used in this example is "python_3.8.2/bin/python3", and the dynamic library directory of python3 is "python_3.8.2/lib".

*The SAIL compiled in this way cannot use its Decoder, Bmcv, and other multimedia-related interfaces.*

1. Download the SOPHON-SAIL source code, unpack it and go to its source code directory

2. Create the build folder "build" and go to the "build" folder

    .. parsed-literal::
        mkdir build && cd build 

3. Execute the compilation command

    .. code-block:: bash

        cmake -DBUILD_TYPE=arm_pcie  \
            -DONLY_RUNTIME=ON \
            -DCMAKE_TOOLCHAIN_FILE=../cmake/BM168x_ARM_PCIE/ToolChain_aarch64_linux.cmake \
            -DPYTHON_EXECUTABLE=python_3.8.2/bin/python3 \
            -DCUSTOM_PY_LIBDIR=python_3.8.2/lib \
            -DLIBSOPHON_BASIC_PATH=libsophon_0.4.1_aarch64/opt/sophon/libsophon-0.4.1 ..
        make                                        

4. Pack python wheel, the path of the generated wheel package is "python/dist" and the file name is "sophon_arm_pcie-|ver|-py3-none-any.whl"

    .. parsed-literal::
        cd ../python 
        chmod +x sophon_whl.sh
        ./sophon_whl.sh 

5. Install python wheel  

Copy "sophon_arm_pcie-|ver|-py3-none-any.whl" to the target ARM host, then execute the following installation command

    .. parsed-literal::
        pip3 install sophon_arm_pcie-|ver|-py3-none-any.whl --force-reinstall 
 
LOONGARCH64 MODE
::::::::::::::::::::

**.Install the loongarch64-linux-gnu toolchain**

Get the [cross-compiled toolchain](http://ftp.loongnix.cn/toolchain/gcc/release/loongarch/gcc8/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1.tar.xz) from the LoongArch64 official website, 
and unzip it locally. The directory structure after decompression is as follows:

.. parsed-literal::

    └── loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1
        ├── bin
        ├── lib
        ├── lib64
        ├── libexec
        ├── loongarch64-linux-gnu
        ├── share
        ├── sysroot
        └── versions 

**.Get the libsophon, sophon-ffmpeg, and sophon-opencv needed for cross-compilation**

*All compilation operations in this section are performed on the x86 host using cross-compilation. The following examples choose to use libsophon version 0.4.7.*

**.Compiling SAIL with Multimedia Modules**

Compile SAIL without bmcv, sophon-ffmpeg, and sophon-opencv by cross-compiling using the specified version of python3 (consistent with python3 on the target ARM host).
You can also get the corresponding python3 package through the official python3 website.
Or you can download the already compiled python3 from [:ref:`Get Python3 for cross-compilation on the X86 host`]..
The path of python3 used in this example is "python_3.8.2/bin/python3", and the dynamic library directory of python3 is "python_3.8.2/lib".

*The SAIL compiled in this way cannot use its Decoder, Bmcv, and other multimedia-related interfaces.*

1. Download the SOPHON-SAIL source code, unpack it and go to its source code directory

2. Create the build folder "build" and go to the "build" folder

    .. parsed-literal::
        mkdir build && cd build 

3. Execute the compilation command

    .. code-block:: bash

        cmake -DBUILD_TYPE=loongarch  \
            -DONLY_RUNTIME=ON \
            -DTOOLCHAIN_BASIC_PATH=toolchains/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1 \
            -DCMAKE_TOOLCHAIN_FILE=../cmake/BM168x_LoongArch64/ToolChain_loongarch64_linux.cmake \
            -DPYTHON_EXECUTABLE=python_3.7.3/bin/python3 \
            -DCUSTOM_PY_LIBDIR=python_3.7.3/lib \
            -DLIBSOPHON_BASIC_PATH=libsophon_0.4.7_loongarch64/opt/sophon/libsophon-0.4.7 \
            ..
        make pysail

*The path in the cmake option needs to be adjusted according to the configuration of your environment*

* DLIBSOPHON_BASIC_PATH: The directory corresponding to the decompression of libsophon\_<x.y.z>_loongarch64.tar.gz under libsophon in SOPHONSDK。

4. Pack python wheel, the path of the generated wheel package is "python/dist" and the file name is "sophon_loongarch64-|ver|-py3-none-any.whl"

    .. parsed-literal::
        cd ../python 
        chmod +x sophon_whl.sh
        ./sophon_whl.sh 

5. Install python wheel  

Copy "sophon_loongarch64-|ver|-py3-none-any.whl" to the target ARM host, then execute the following installation command

    .. parsed-literal::
        pip3 install sophon_loongarch64-|ver|-py3-none-any.whl --force-reinstall 


Compile User Manual
>>>>>>>>>>>>>>>>>>>>>>>>>>>>

**.Install software packages**

.. parsed-literal::

    # Update apt
    sudo apt update
    # Install latex
    sudo apt install texlive-xetex texlive-latex-recommended
    # Install Sphinx
    pip3 install sphinx sphinx-autobuild sphinx_rtd_theme rst2pdf
    # Install the jieba Chinese text segmentation library to support Chinese search
    pip3 install jieba3k


**.Install fonts**

    [Fandol](https://ctan.org/pkg/fandol) - Four basic fonts for Chinese typesetting

.. parsed-literal::

    # Download the font
    wget http://mirrors.ctan.org/fonts/fandol.zip
    # Unpack the font package
    unzip fandol.zip
    # Copy and install the font package
    sudo cp -r fandol /usr/share/fonts/
    cp -r fandol ~/.fonts


**.Execute compilation**

.. parsed-literal::

   cd docs
   make pdf LANG=en

The compiled user manual path is "docs/build/SOPHON-SAIL_en.pdf"

*If the compilation still reports errors, you can run "sudo apt-get install texlive-lang-chinese", and then re-run the above command.*

Develop Programs Using SAIL's Python Interface
________________________________________________________________________________________

*Note: BM1688 and CV186AH processors only support SOC MODE chapter, BM1684 and BM1684X processors do not have this limitation.Note that for BM1688 or CV186AH, sophon-mw needs to be changed to sophon-media.*

PCIE MODE
>>>>>>>>>>>>>>>>>>>>>>>>>>>
After compiling SAIL with PCIE MODE and installing python wheel, you can call SAIL in python, the interface documentation can be found in the API chapter.

SOC MODE
>>>>>>>>>>>>>>

**.Use your own compiled Python wheel package**

After compiling SAIL by cross-compiling with SOC MODE, copy the python wheel to SOC and install it, then you can call SAIL in python, the interface document can be found in the API chapter.

**.Use the pre-compiled Python wheel package**

1. Check libsophon version and sophon-mw(sophon-ffmpeg,sophon-opencv) version on SOC

    .. parsed-literal::

        ls /opt/sophon/

2. Check Python3 version on SOC

    .. parsed-literal::

        python3 --version

3. You can find the corresponding version of the wheel package from the pre-compiled Python wheel package, copy it to the SOC, and install it; then, you can use python to call SAIL. Its interface documentation can be found in the API chapter.

ARM PCIE MODE
>>>>>>>>>>>>>>
After compiling SAIL by cross-compiling with ARM PCIE MODE, copy the python wheel to ARM host and install it, then you can call SAIL in python, the interface document can be found in the API chapter.

1. Check libsophon version and sophon-mw(sophon-ffmpeg,sophon-opencv) version on the ARM host

    .. parsed-literal::

        ls /opt/sophon/

2. Check Python3 version on the ARM host

    .. parsed-literal::

        python3 --version

3. You can find the corresponding version of the wheel package from the pre-compiled Python wheel package, copy it to the ARM host, and install it; then, you can use python to call SAIL. Its interface documentation can be found in the API chapter.


Develop Programs Using SAIL's C++ Interface
________________________________________________________________________________________

*Note: BM1688 and CV186AH processors only support SOC MODE chapter, BM1684 and BM1684X processors do not have this limitation.Note that for BM1688 or CV186AH, sophon-mw needs to be changed to sophon-media.*

PCIE MODE
>>>>>>>>>>>>>>>>>>>>>>>>>>>
After compiling SAIL with PCIE MODE and installing SAIL's c++ libraries by running "sudo make install" or by copying them.
It is recommended to use cmake to link the SAIL libraries to your application.
If you need to use SAIL multimedia-related functions, you also need to add libsophon, sophon-ffmpeg, sophon-opencv header file directory, and dynamic library directory to your program.
You can add the following paragraph to your program's CMakeLists.txt:

.. parsed-literal::

    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    # Add libsophon's header file directories

    set(SAIL_DIR  /opt/sophon/sophon-sail/lib/cmake)
    find_package(SAIL REQUIRED)
    include_directories(${SAIL_INCLUDE_DIRS})
    link_directories(${SAIL_LIB_DIRS})
    # Add SAIL header files and dynamic library directories

    set(OpenCV_DIR  /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    # Add the header file directories of sophon-opencv

    set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
    find_package(FFMPEG REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    link_directories(${FFMPEG_LIB_DIRS})
    # Add the header file directories and dynamic library directories of sophon-ffmpeg

    add_executable(${YOUR_TARGET_NAME} ${YOUR_SOURCE_FILES})
    target_link_libraries(${YOUR_TARGET_NAME} sail)


The functions in sail can be called from within your code:

.. code-block:: cpp

    #define USE_FFMPEG  1
    #define USE_OPENCV  1
    #define USE_BMCV    1

    #include <stdio.h>
    #include <sail/cvwrapper.h>
    #include <iostream>
    #include <string>

    using namespace std;

    int main() 
    {
        int device_id = 0;
        std::string video_path = "test.avi";
        sail::Decoder decoder(video_path,true,device_id);
        if(!decoder.is_opened()){
            printf("Video[%s] read failed!\n",video_path.c_str());
            exit(1) ;
        }
        
        sail::Handle handle(device_id);
        sail::Bmcv bmcv(handle);
        
        while(true){
            sail::BMImage ost_image = decoder.read(handle);
            bmcv.imwrite("test.jpg", ost_image);
            break;
        }

        return 0;
    }


SOC MODE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

**.Compile the program on the SOC device**

After installing libsophon, sophon-ffmpeg, sophon-opencv, and SAIL on the SOC device, you can use cmake to link the libraries in SAIL to your application by referring to the PCIE MODE development method.
If you need to use SAIL multimedia-related functions, you also need to add libsophon, sophon-ffmpeg, and sophon-opencv header file directories and dynamic library directories to your application.


**.Cross-compile programs on x86 hosts**

If you want to build a cross-compilation environment using SAIL, you will need libsophon, sophon-ffmpeg, sophon-opencv, and the gcc-aarch64-linux-gnu toolchain.

**.Create the "soc-sdk" folder**

Create the "soc-sdk" folder, the header files and dynamic libraries needed for subsequent cross-compilation will be stored in this directory.

    .. parsed-literal::
        mkdir soc-sdk

**.Get the libsophon,sophon-ffmpeg,sophon-opencv libraries needed for cross-compilation**

*The following examples choose to use libsophon version 0.4.1, sophon-ffmpeg version 0.4.1, and sophon-opencv version 0.4.1.*

1. Get "libsophon_soc_0.4.1_aarch64.tar.gz" from sophon's official website and unpack and copy it to the "soc-sdk" folder

    .. parsed-literal::
        tar -xvf libsophon_soc_0.4.1_aarch64.tar.gz
        cp -r libsophon_soc_0.4.1_aarch64/opt/sophon/libsophon-0.4.1/include soc-sdk
        cp -r libsophon_soc_0.4.1_aarch64/opt/sophon/libsophon-0.4.1/lib soc-sdk
        
The directory of libsophon after unpacking is "libsophon_soc_0.4.1_aarch64/opt/sophon/libsophon-0.4.1"

2. Get "sophon-mw-soc_0.4.1_aarch64.tar.gz" from sophon's official website and unpack and copy it to the "soc-sdk" folder

    .. parsed-literal::
        tar -xvf sophon-mw-soc_0.4.1_aarch64.tar.gz
        cp -r sophon-mw-soc_0.4.1_aarch64/opt/sophon/sophon-ffmpeg_0.4.1/include soc-sdk
        cp -r sophon-mw-soc_0.4.1_aarch64/opt/sophon/sophon-ffmpeg_0.4.1/lib soc-sdk
        cp -r sophon-mw-soc_0.4.1_aarch64/opt/sophon/sophon-opencv_0.4.1/include/opencv4/opencv2 soc-sdk/include
        cp -r sophon-mw-soc_0.4.1_aarch64/opt/sophon/sophon-opencv_0.4.1/lib soc-sdk

**.Copy the cross-compiled SAIL, i.e. "build_soc", to the "soc-sdk" folder**

    .. parsed-literal::
        cp build_soc/sophon-sail/include soc-sdk
        cp build_soc/sophon-sail/lib soc-sdk

**.Install the gcc-aarch64-linux-gnu toolchain**

*If already installed, you can ignore this step*

    .. parsed-literal::
        sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

After the above steps are configured, you can finish cross-compiling by configuring cmake. Add the following paragraph to your program's CMakeLists.txt:

*CMakeLists.txt needs to use "/opt/sophon/soc-sdk" as the absolute path to "soc-sdk", which needs to be configured according to the actual location of the file when it is applied.*

.. parsed-literal::

    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    
    include_directories("/opt/sophon/soc-sdk/include")
    include_directories("/opt/sophon/soc-sdk/include/sail")
    # Add the header file directories to be used for cross-compilation

    link_directories("/opt/sophon/soc-sdk/lib")
    # Add dynamic library directories to be used for cross-compilation

    add_executable(${YOUR_TARGET_NAME} ${YOUR_SOURCE_FILES})
    target_link_libraries(${YOUR_TARGET_NAME} sail)
    # sail is the library that needs to be linked


ARM PCIE MODE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

**.Compile the program on the ARM host**

After installing libsophon, sophon-ffmpeg, sophon-opencv, and SAIL on the ARM host, you can use cmake to link the libraries in SAIL to your application by referring to the PCIE MODE development method.
If you need to use SAIL multimedia-related functions, you also need to add libsophon, sophon-ffmpeg, and sophon-opencv header file directories and dynamic library directories to your application.


**.Cross-compile programs on x86 hosts**

If you want to build a cross-compilation environment using SAIL, you will need libsophon, sophon-ffmpeg, sophon-opencv, and the gcc-aarch64-linux-gnu toolchain.

**.Create the "arm_pcie-sdk" folder**

Create the "arm_pcie-sdk" folder, the header files and dynamic libraries needed for subsequent cross-compilation will be stored in this directory.

    .. parsed-literal::
        mkdir arm_pcie-sdk

**.Get the libsophon,sophon-ffmpeg,sophon-opencv libraries needed for cross-compilation**

*The following examples choose to use libsophon version 0.4.1, sophon-ffmpeg version 0.4.1, and sophon-opencv version 0.4.1.*

1. Get "libsophon_0.4.1_aarch64.tar.gz" from sophon's official website and unpack and copy it to the "arm_pcie-sdk" folder

    .. parsed-literal::
        tar -xvf libsophon_0.4.1_aarch64.tar.gz
        cp -r libsophon_0.4.1_aarch64/opt/sophon/libsophon-0.4.1/include arm_pcie-sdk
        cp -r libsophon_0.4.1_aarch64/opt/sophon/libsophon-0.4.1/lib arm_pcie-sdk
        
The directory of libsophon after unpacking is "libsophon_0.4.1_aarch64/opt/sophon/libsophon-0.4.1"

2. Get "sophon-mw_0.4.1_aarch64.tar.gz" from sophon's official website and unpack and copy it to the "arm_pcie-sdk" folder

    .. parsed-literal::
        tar -xvf sophon-mw_0.4.1_aarch64.tar.gz
        cp -r sophon-mw_0.4.1_aarch64/opt/sophon/sophon-ffmpeg_0.4.1/include arm_pcie-sdk
        cp -r sophon-mw_0.4.1_aarch64/opt/sophon/sophon-ffmpeg_0.4.1/lib arm_pcie-sdk
        cp -r sophon-mw_0.4.1_aarch64/opt/sophon/sophon-opencv_0.4.1/include/opencv4/opencv2 arm_pcie-sdk/include
        cp -r sophon-mw_0.4.1_aarch64/opt/sophon/sophon-opencv_0.4.1/lib arm_pcie-sdk

**.Copy the cross-compiled SAIL, i.e. "build_arm_pcie", to the "arm_pcie-sdk" folder**

    .. parsed-literal::
        cp build_arm_pcie/sophon-sail/include arm_pcie-sdk
        cp build_arm_pcie/sophon-sail/lib arm_pcie-sdk

**.Install the gcc-aarch64-linux-gnu toolchain**

*If already installed, you can ignore this step*

    .. parsed-literal::
        sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

After the above steps are configured, you can finish cross-compiling by configuring cmake. Add the following paragraph to your program's CMakeLists.txt:

*CMakeLists.txt needs to use "/opt/sophon/arm_pcie-sdk as the absolute path to "arm_pcie-sdk", which needs to be configured according to the actual location of the file when it is applied.*

.. parsed-literal::

    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    
    include_directories("/opt/sophon/arm_pcie-sdk/include")
    include_directories("/opt/sophon/arm_pcie-sdk/include/sail")
    # Add the header file directories to be used for cross-compilation

    link_directories("/opt/sophon/arm_pcie-sdk/lib")
    # Add dynamic library directories to be used for cross-compilation

    add_executable(${YOUR_TARGET_NAME} ${YOUR_SOURCE_FILES})
    target_link_libraries(${YOUR_TARGET_NAME} sail)
    # sail is the library that needs to be linked
