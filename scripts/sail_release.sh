#!/bin/bash
# set -ex
target=$1
shell_dir=$(dirname $(readlink -f "$0"))

SAIL_BASIC_PATH=$shell_dir/..
SAIL_RESULT_PATH=$SAIL_BASIC_PATH/release
VERSION_PATH=$SAIL_BASIC_PATH/git_version
SOURCECODE_PATH=$SAIL_RESULT_PATH/sophon-sail
exec < $VERSION_PATH
read -r line
SAIL_VERSION=$line

USER='AI'
PASSWD='SophgoRelease2022'
HOST='172.28.141.89'

if [ -d $SAIL_RESULT_PATH ]; then
    rm -rf $SAIL_RESULT_PATH
fi
mkdir -p $SAIL_RESULT_PATH
mkdir -p $SOURCECODE_PATH

if [ -d $SAIL_BASIC_PATH/build ]; then
    rm -rf $SAIL_BASIC_PATH/build
fi

if [ -d $SAIL_BASIC_PATH/docs/build ]; then
    rm -rf $SAIL_BASIC_PATH/docs/build
fi

if [ -d $SAIL_RESULT_PATH/docs ]; then
    rm -rf $SAIL_RESULT_PATH/docs
fi

mkdir -p $SAIL_RESULT_PATH/docs

function judge_ret() {
  if [[ $1 == 0 ]]; then
    echo -e "\033[32m Passed: $2 \033[0m"
    echo ""
  else
    echo -e "\033[31m Failed: $2 \033[0m"
    exit 1
  fi
  sleep 2
}

function download_python() {
    py_version=$1
    pythons_path=$SAIL_BASIC_PATH/pythons
    if [ ! -f $pythons_path ]; then
        mkdir -p $pythons_path
    fi
    pushd $pythons_path
    wget ftp://$USER:$PASSWD@$HOST/toolchains/pythons/Python-${py_version}.tar.gz  
    judge_ret $? "download_python V"${py_version}
    if [ -f $SAIL_BASIC_PATH/pythons/Python-${py_version} ]; then
        rm -rf $SAIL_BASIC_PATH/pythons/Python-${py_version}
    fi
    tar -xzvf Python-${py_version}.tar.gz
    judge_ret $? "tar -xzvf Python-"${py_version}".tar.gz"
    popd
}

function clean_python_tar() {
    py_version=$1
    pythons_path=$SAIL_BASIC_PATH/pythons
    rm -rf $pythons_path/Python-${py_version}.tar.gz
}

function clean_python_files() {
    pythons_path=$SAIL_BASIC_PATH/pythons
    rm -rf $pythons_path
}
function download_pythons() {
    # download_python "3.5.9"
    # download_python "3.6.5"
    # download_python "3.7.3"
    download_python "3.8.2"
    download_python "3.9.0"
    download_python "3.10.0"
    # download_python "3.11.0"

    # clean_python_tar "3.5.9"
    # clean_python_tar "3.6.5"
    # clean_python_tar "3.7.3"
    clean_python_tar "3.8.2"
    clean_python_tar "3.9.0"
    clean_python_tar "3.10.0"
    # clean_python_tar "3.11.0"
}

function download_loong_toolchain() {
    loong_toolchain_xz_path=$SAIL_BASIC_PATH/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1.tar.xz
    loong_toolchain_path=$SAIL_BASIC_PATH/loongarch

    if [ -f $loong_toolchain_xz_path ]; then
        rm -rf $loong_toolchain_xz_path
    fi

    if [ -d $loong_toolchain_path ]; then
        rm -rf $loong_toolchain_path
    fi

    pushd $SAIL_BASIC_PATH
    if [ ! -f $loong_toolchain_xz_path ]; then
        echo "Start download file: "$loong_toolchain_xz_path
        wget ftp://$USER:$PASSWD@$HOST/toolchains/gcc/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1.tar.xz  
        judge_ret $? "Downloaded"$loong_toolchain_xz_path
    else
        echo "File already existed: "$loong_toolchain_xz_path", need not to download!"
    fi

    tar -xf $loong_toolchain_xz_path 
    popd

    rm -rf $loong_toolchain_xz_path
}

function build_soc_pysail() {
    basic_version=$1
    libsophon_release_path=$2
    libsophon_version=$3
    sophonmw_release_path=$4
    sophonmw_version=$5
    chip_type=$6

    if [ $# -eq 7 ];then
        build_tpu_kernrl_off=$7
    else
        build_tpu_kernrl_off=0
    fi

    if [ $chip_type = "BM1688" ];then
        if [ $basic_version = "v1.2" ];then
            sophonmw_label="sophon_mw"
        elif [ $basic_version = "v1.3" ];then
            sophonmw_label="sophon-mw"
        else
            sophonmw_label="sophon-media"
        fi

        libsophon_soc_tar_url=ftp://$USER:$PASSWD@$HOST/athena2/sophon-img-edge-auto/release_build/${libsophon_release_path}/libsophon_soc_${libsophon_version}_aarch64.tar.gz
        sophonmw_url=ftp://$USER:$PASSWD@$HOST/athena2/sophon_media/release_build/${sophonmw_release_path}/${sophonmw_label}-soc_${sophonmw_version}_aarch64.tar.gz
    
        local whl_result_path=$SOURCECODE_PATH/python_wheels/soc_${chip_type}/${basic_version}_libsophon-${libsophon_version}_sophonmedia-${sophonmw_version}
        libsophon_dir=libsophon_soc_${libsophon_version}_aarch64
        sophonmw_dir=${sophonmw_label}-soc_${sophonmw_version}_aarch64
    else
        libsophon_soc_tar_url=ftp://$USER:$PASSWD@$HOST/sophon-sdk/sophon-img/release_build/${libsophon_release_path}/libsophon_soc_${libsophon_version}_aarch64.tar.gz
        sophonmw_url=ftp://$USER:$PASSWD@$HOST/sophon-sdk/sophon-mw/release_build/${sophonmw_release_path}/sophon-mw-soc_${sophonmw_version}_aarch64.tar.gz
        local whl_result_path=$SOURCECODE_PATH/python_wheels/soc_${chip_type}/libsophon-${libsophon_version}_sophonmw-${sophonmw_version}
        libsophon_dir=libsophon_soc_${libsophon_version}_aarch64
        sophonmw_dir=sophon-mw-soc_${sophonmw_version}_aarch64
    fi

    mkdir -p $whl_result_path

    soc_sdk_path=$SAIL_BASIC_PATH/soc-sdk
    libsophon_soc_tar=$soc_sdk_path/${libsophon_dir}.tar.gz
    sophonmw_soc_tar=$soc_sdk_path/${sophonmw_dir}.tar.gz

    if [ ! -d $soc_sdk_path ]; then
        mkdir -p $soc_sdk_path
    fi

    rm -rf $soc_sdk_path/*

    pushd $soc_sdk_path
    if [ ! -f $libsophon_soc_tar ]; then
        echo "Start download file: "$libsophon_soc_tar
        wget $libsophon_soc_tar_url
        judge_ret $? "Downloaded"$libsophon_soc_tar
    else
        echo "File already existed: "$libsophon_soc_tar", need not to download!"
    fi

    if [ ! -f $sophonmw_soc_tar ]; then
        echo "Start download file: "$sophonmw_soc_tar
        wget $sophonmw_url
        judge_ret $? "Downloaded"$sophonmw_soc_tar
    else
        echo "File already existed: "$sophonmw_soc_tar", need not to download!"
    fi

    tar -xzvf $libsophon_soc_tar 
    tar -xzvf $sophonmw_soc_tar 

    mv -f  ${libsophon_dir}/opt/sophon/libsophon-${libsophon_version} ./
    mv -f  ${sophonmw_dir}/opt/sophon/sophon-ffmpeg_${sophonmw_version} ./
    mv -f  ${sophonmw_dir}/opt/sophon/sophon-opencv_${sophonmw_version} ./
    rm -rf $libsophon_soc_tar 
    rm -rf $sophonmw_soc_tar
    rm -rf ${libsophon_dir}
    rm -rf ${sophonmw_dir}
    popd

    for python_version in 3.8.2 3.10.0 ;  do
        if [ ! -f $SAIL_BASIC_PATH/build ]; then
            rm -rf $SAIL_BASIC_PATH/build 
        fi
        mkdir -p $SAIL_BASIC_PATH/build 

        local temp_str=${python_version#3.*}
        local whl_release_path=$whl_result_path/py3${temp_str%.*}
        mkdir -p $whl_release_path 

        pushd $SAIL_BASIC_PATH/build
        if [ $build_tpu_kernrl_off -eq 1 ];then
        cmake -DBUILD_TYPE=soc  \
            -DCMAKE_TOOLCHAIN_FILE=$SAIL_BASIC_PATH/cmake/BM168x_SOC/ToolChain_aarch64_linux.cmake \
            -DPYTHON_EXECUTABLE=$SAIL_BASIC_PATH/pythons/Python-${python_version}/python_${python_version}/bin/python3 \
            -DCUSTOM_PY_LIBDIR=$SAIL_BASIC_PATH/pythons/Python-${python_version}/python_${python_version}/lib \
            -DLIBSOPHON_BASIC_PATH=$soc_sdk_path/libsophon-${libsophon_version} \
            -DFFMPEG_BASIC_PATH=$soc_sdk_path/sophon-ffmpeg_${sophonmw_version} \
            -DOPENCV_BASIC_PATH=$soc_sdk_path/sophon-opencv_${sophonmw_version} \
            -DTPUKERNRL_OFF=1\
            ..                                
        else
        cmake -DBUILD_TYPE=soc  \
            -DCMAKE_TOOLCHAIN_FILE=$SAIL_BASIC_PATH/cmake/BM168x_SOC/ToolChain_aarch64_linux.cmake \
            -DPYTHON_EXECUTABLE=$SAIL_BASIC_PATH/pythons/Python-${python_version}/python_${python_version}/bin/python3 \
            -DCUSTOM_PY_LIBDIR=$SAIL_BASIC_PATH/pythons/Python-${python_version}/python_${python_version}/lib \
            -DLIBSOPHON_BASIC_PATH=$soc_sdk_path/libsophon-${libsophon_version} \
            -DFFMPEG_BASIC_PATH=$soc_sdk_path/sophon-ffmpeg_${sophonmw_version} \
            -DOPENCV_BASIC_PATH=$soc_sdk_path/sophon-opencv_${sophonmw_version} \
            .. 
        fi
        core_nums=`cat /proc/cpuinfo| grep "processor"| wc -l`
        make pysail -j${core_nums}
        judge_ret $? "build_soc_pysail, python"$python_version",SDK version: "$1",libsophon: "$2", sophonmw: "$3
        popd

        pushd $SAIL_BASIC_PATH/python/soc
        chmod +x sophon_soc_whl.sh
        ./sophon_soc_whl.sh  
        judge_ret $? "build soc python"$python_version" wheel"
        mv ./dist/sophon*.whl $whl_release_path
        popd

    done
}

function build_soc_pysail_LTS() {
    basic_version=$1
    libsophon_release_path=$2
    libsophon_version=$3
    sophonmw_release_path=$4
    sophonmw_version=$5
    chip_type=$6

    if [ $# -eq 7 ];then
        build_tpu_kernrl_off=$7
    else
        build_tpu_kernrl_off=0
    fi

    if [ $chip_type = "BM1688" ];then
        if [ $basic_version = "v1.2" ];then
            sophonmw_label="sophon_mw"
        elif [ $basic_version = "v1.3" ];then
            sophonmw_label="sophon-mw"
        else
            sophonmw_label="sophon-media"
        fi

        libsophon_soc_tar_url=ftp://$USER:$PASSWD@$HOST/athena2/sophon-img-edge-auto/release_build/${libsophon_release_path}/libsophon_soc_${libsophon_version}_aarch64.tar.gz
        sophonmw_url=ftp://$USER:$PASSWD@$HOST/athena2/sophon_media/release_build/${sophonmw_release_path}/${sophonmw_label}-soc_${sophonmw_version}_aarch64.tar.gz
    
        local whl_result_path=$SOURCECODE_PATH/python_wheels/soc_${chip_type}/${basic_version}_libsophon-${libsophon_version}_sophonmedia-${sophonmw_version}
        libsophon_dir=libsophon_soc_${libsophon_version}_aarch64
        sophonmw_dir=${sophonmw_label}-soc_${sophonmw_version}_aarch64
    else
        libsophon_soc_tar_url=ftp://$USER:$PASSWD@$HOST/sophon-sdk/sophon-img/release_build/${libsophon_release_path}/libsophon_soc_${libsophon_version}-LTS_aarch64.tar.gz
        sophonmw_url=ftp://$USER:$PASSWD@$HOST/sophon-sdk/sophon-mw/release_build/${sophonmw_release_path}/sophon-mw-soc_${sophonmw_version}_aarch64.tar.gz
        local whl_result_path=$SOURCECODE_PATH/python_wheels/soc_${chip_type}/libsophon-${libsophon_version}_sophonmw-${sophonmw_version}
        libsophon_dir=libsophon_soc_${libsophon_version}-LTS_aarch64
        sophonmw_dir=sophon-mw-soc_${sophonmw_version}_aarch64
    fi

    mkdir -p $whl_result_path

    soc_sdk_path=$SAIL_BASIC_PATH/soc-sdk
    libsophon_soc_tar=$soc_sdk_path/${libsophon_dir}.tar.gz
    sophonmw_soc_tar=$soc_sdk_path/${sophonmw_dir}.tar.gz

    if [ ! -d $soc_sdk_path ]; then
        mkdir -p $soc_sdk_path
    fi

    rm -rf $soc_sdk_path/*

    pushd $soc_sdk_path
    if [ ! -f $libsophon_soc_tar ]; then
        echo "Start download file: "$libsophon_soc_tar
        wget $libsophon_soc_tar_url
        judge_ret $? "Downloaded"$libsophon_soc_tar
    else
        echo "File already existed: "$libsophon_soc_tar", need not to download!"
    fi

    if [ ! -f $sophonmw_soc_tar ]; then
        echo "Start download file: "$sophonmw_soc_tar
        wget $sophonmw_url
        judge_ret $? "Downloaded"$sophonmw_soc_tar
    else
        echo "File already existed: "$sophonmw_soc_tar", need not to download!"
    fi

    tar -xzvf $libsophon_soc_tar 
    tar -xzvf $sophonmw_soc_tar 

    mv -f  ${libsophon_dir}/opt/sophon/libsophon-${libsophon_version} ./
    mv -f  ${sophonmw_dir}/opt/sophon/sophon-ffmpeg_${sophonmw_version} ./
    mv -f  ${sophonmw_dir}/opt/sophon/sophon-opencv_${sophonmw_version} ./
    rm -rf $libsophon_soc_tar 
    rm -rf $sophonmw_soc_tar
    rm -rf ${libsophon_dir}
    rm -rf ${sophonmw_dir}
    popd

    for python_version in 3.8.2 3.10.0 ;  do
        if [ ! -f $SAIL_BASIC_PATH/build ]; then
            rm -rf $SAIL_BASIC_PATH/build 
        fi
        mkdir -p $SAIL_BASIC_PATH/build 

        local temp_str=${python_version#3.*}
        local whl_release_path=$whl_result_path/py3${temp_str%.*}
        mkdir -p $whl_release_path 

        pushd $SAIL_BASIC_PATH/build
        if [ $build_tpu_kernrl_off -eq 1 ];then
        cmake -DBUILD_TYPE=soc  \
            -DCMAKE_TOOLCHAIN_FILE=$SAIL_BASIC_PATH/cmake/BM168x_SOC/ToolChain_aarch64_linux.cmake \
            -DPYTHON_EXECUTABLE=$SAIL_BASIC_PATH/pythons/Python-${python_version}/python_${python_version}/bin/python3 \
            -DCUSTOM_PY_LIBDIR=$SAIL_BASIC_PATH/pythons/Python-${python_version}/python_${python_version}/lib \
            -DLIBSOPHON_BASIC_PATH=$soc_sdk_path/libsophon-${libsophon_version} \
            -DFFMPEG_BASIC_PATH=$soc_sdk_path/sophon-ffmpeg_${sophonmw_version} \
            -DOPENCV_BASIC_PATH=$soc_sdk_path/sophon-opencv_${sophonmw_version} \
            -DTPUKERNRL_OFF=1\
            ..                                
        else
        cmake -DBUILD_TYPE=soc  \
            -DCMAKE_TOOLCHAIN_FILE=$SAIL_BASIC_PATH/cmake/BM168x_SOC/ToolChain_aarch64_linux.cmake \
            -DPYTHON_EXECUTABLE=$SAIL_BASIC_PATH/pythons/Python-${python_version}/python_${python_version}/bin/python3 \
            -DCUSTOM_PY_LIBDIR=$SAIL_BASIC_PATH/pythons/Python-${python_version}/python_${python_version}/lib \
            -DLIBSOPHON_BASIC_PATH=$soc_sdk_path/libsophon-${libsophon_version} \
            -DFFMPEG_BASIC_PATH=$soc_sdk_path/sophon-ffmpeg_${sophonmw_version} \
            -DOPENCV_BASIC_PATH=$soc_sdk_path/sophon-opencv_${sophonmw_version} \
            .. 
        fi
        core_nums=`cat /proc/cpuinfo| grep "processor"| wc -l`
        make pysail -j${core_nums}
        judge_ret $? "build_soc_pysail, python"$python_version",SDK version: "$1",libsophon: "$2", sophonmw: "$3
        popd

        pushd $SAIL_BASIC_PATH/python/soc
        chmod +x sophon_soc_whl.sh
        ./sophon_soc_whl.sh  
        judge_ret $? "build soc python"$python_version" wheel"
        mv ./dist/sophon*.whl $whl_release_path
        popd

    done
}

function build_arm_pcie_pysail() {
    basic_version=$1
    libsophon_release_path=$2
    libsophon_version=$3
    sophonmw_release_path=$4
    sophonmw_version=$5

    if [ $# -eq 6 ];then
        build_tpu_kernrl_off=$6
    else
        build_tpu_kernrl_off=0
    fi

    libsophon_arm_pcie_tar_url=ftp://$USER:$PASSWD@$HOST/sophon-sdk/libsophon/release_build/${libsophon_release_path}/libsophon_${libsophon_version}_aarch64.tar.gz
    sophonmw_arm_pcie_tar_url=ftp://$USER:$PASSWD@$HOST/sophon-sdk/sophon-mw/release_build/${sophonmw_release_path}/sophon-mw_${sophonmw_version}_aarch64.tar.gz

    local whl_result_path=$SOURCECODE_PATH/python_wheels
    mkdir -p $whl_result_path
    local whl_result_path=$whl_result_path/arm_pcie
    mkdir -p $whl_result_path
    local whl_result_path=$whl_result_path/libsophon-${libsophon_version}_sophonmw-${sophonmw_version}
    mkdir -p $whl_result_path

    arm_pcie_sdk_path=$SAIL_BASIC_PATH/arm_pcie-sdk
    libsophon_arm_pcie_tar=$arm_pcie_sdk_path/libsophon_${libsophon_version}_aarch64.tar.gz
    sophonmw_arm_pcie_tar=$arm_pcie_sdk_path/sophon-mw_${sophonmw_version}_aarch64.tar.gz

    if [ -d $arm_pcie_sdk_path ]; then
        rm -rf $arm_pcie_sdk_path
    fi
    mkdir -p $arm_pcie_sdk_path

    pushd $arm_pcie_sdk_path
    if [ ! -f $libsophon_arm_pcie_tar ]; then
        echo "Start download file: "$libsophon_arm_pcie_tar
        wget $libsophon_arm_pcie_tar_url
        judge_ret $? "Downloaded "$libsophon_arm_pcie_tar
    else
        echo "File already existed: "$libsophon_arm_pcie_tar", need not to download!"
    fi

    if [ ! -f $sophonmw_arm_pcie_tar ]; then
        echo "Start download file: "$sophonmw_arm_pcie_tar
        wget $sophonmw_arm_pcie_tar_url
        judge_ret $? "Downloaded "$sophonmw_arm_pcie_tar
    else
        echo "File already existed: "$sophonmw_arm_pcie_tar", need not to download!"
    fi

    tar -xzvf $libsophon_arm_pcie_tar 
    tar -xzvf $sophonmw_arm_pcie_tar 
    mv -f libsophon_${libsophon_version}_aarch64/opt/sophon/libsophon-${libsophon_version} ./
    mv -f sophon-mw_${sophonmw_version}_aarch64/opt/sophon/sophon-ffmpeg_${sophonmw_version} ./
    mv -f sophon-mw_${sophonmw_version}_aarch64/opt/sophon/sophon-opencv_${sophonmw_version} ./
    rm -rf $libsophon_arm_pcie_tar 
    rm -rf $sophonmw_arm_pcie_tar
    rm -rf libsophon_${libsophon_version}_aarch64
    rm -rf sophon-mw_${sophonmw_version}_aarch64
    popd

    for python_version in 3.8.2 3.9.0 3.10.0 ;  do
        if [ ! -f $SAIL_BASIC_PATH/build ]; then
            rm -rf $SAIL_BASIC_PATH/build 
        fi
        mkdir -p $SAIL_BASIC_PATH/build 

        local temp_str=${python_version#3.*}
        local whl_release_path=$whl_result_path/py3${temp_str%.*}
        mkdir -p $whl_release_path 

        pushd $SAIL_BASIC_PATH/build
        if [ $build_tpu_kernrl_off -eq 1 ];then
        cmake -DBUILD_TYPE=arm_pcie  \
            -DCMAKE_TOOLCHAIN_FILE=$SAIL_BASIC_PATH/cmake/BM168x_ARM_PCIE/ToolChain_aarch64_linux.cmake \
            -DPYTHON_EXECUTABLE=$SAIL_BASIC_PATH/pythons/Python-${python_version}/python_${python_version}/bin/python3 \
            -DCUSTOM_PY_LIBDIR=$SAIL_BASIC_PATH/pythons/Python-${python_version}/python_${python_version}/lib \
            -DLIBSOPHON_BASIC_PATH=$arm_pcie_sdk_path/libsophon-${libsophon_version} \
            -DFFMPEG_BASIC_PATH=$arm_pcie_sdk_path/sophon-ffmpeg_${sophonmw_version} \
            -DOPENCV_BASIC_PATH=$arm_pcie_sdk_path/sophon-opencv_${sophonmw_version} \
            -DTPUKERNRL_OFF=1 \
            ..
        else
        cmake -DBUILD_TYPE=arm_pcie  \
            -DCMAKE_TOOLCHAIN_FILE=$SAIL_BASIC_PATH/cmake/BM168x_ARM_PCIE/ToolChain_aarch64_linux.cmake \
            -DPYTHON_EXECUTABLE=$SAIL_BASIC_PATH/pythons/Python-${python_version}/python_${python_version}/bin/python3 \
            -DCUSTOM_PY_LIBDIR=$SAIL_BASIC_PATH/pythons/Python-${python_version}/python_${python_version}/lib \
            -DLIBSOPHON_BASIC_PATH=$arm_pcie_sdk_path/libsophon-${libsophon_version} \
            -DFFMPEG_BASIC_PATH=$arm_pcie_sdk_path/sophon-ffmpeg_${sophonmw_version} \
            -DOPENCV_BASIC_PATH=$arm_pcie_sdk_path/sophon-opencv_${sophonmw_version} \
            ..
        fi

        core_nums=`cat /proc/cpuinfo| grep "processor"| wc -l`
        make pysail -j${core_nums}
        judge_ret $? "build_arm_pcie_pysail, python"$python_version",SDK version: "$1",libsophon: "$2", sophonmw: "$3
        popd

        pushd $SAIL_BASIC_PATH/python/arm_pcie
        chmod +x sophon_arm_pcie_whl.sh
        ./sophon_arm_pcie_whl.sh  
        judge_ret $? "build arm pcie python"$python_version" wheel"
        mv ./dist/sophon*.whl $whl_release_path
        popd

    done
}

function build_loongarch_pysail() {
    basic_version=$1
    libsophon_release_path=$2
    libsophon_version=$3
    sophonmw_release_path=$4
    sophonmw_version=$5

    if [ $# -eq 6 ];then
        build_tpu_kernrl_off=$6
    else
        build_tpu_kernrl_off=0
    fi


    libsophon_loongarch_tar_url=ftp://$USER:$PASSWD@$HOST/sophon-sdk/libsophon/release_build/${libsophon_release_path}/libsophon_${libsophon_version}_loongarch64.tar.gz
    sophonmw_loongarch_tar_url=ftp://$USER:$PASSWD@$HOST/sophon-sdk/sophon-mw/release_build/${sophonmw_release_path}/sophon-mw_${sophonmw_version}_loongarch64.tar.gz

    local whl_result_path=$SOURCECODE_PATH/python_wheels
    mkdir -p $whl_result_path
    local whl_result_path=$whl_result_path/loongarch
    mkdir -p $whl_result_path
    local whl_result_path=$whl_result_path/libsophon-${libsophon_version}_sophonmw-${sophonmw_version}
    mkdir -p $whl_result_path

    loongarch_sdk_path=$SAIL_BASIC_PATH/loongarch-sdk
    libsophon_loongarch_tar=$loongarch_sdk_path/libsophon_${libsophon_version}_loongarch64.tar.gz
    sophonmw_loongarch_tar=$loongarch_sdk_path/sophon-mw_${sophonmw_version}_loongarch64.tar.gz

    if [ -d $loongarch_sdk_path ]; then
        rm -rf $loongarch_sdk_path
    fi
    mkdir -p $loongarch_sdk_path

    pushd $loongarch_sdk_path
    if [ ! -f $libsophon_loongarch_tar ]; then
        echo "Start download file: "$libsophon_loongarch_tar
        wget $libsophon_loongarch_tar_url
        judge_ret $? "Downloaded"$libsophon_loongarch_tar
    else
        echo "File already existed: "$libsophon_loongarch_tar", need not to download!"
    fi

    if [ ! -f $sophonmw_loongarch_tar ]; then
        echo "Start download file: "$sophonmw_loongarch_tar
        wget $sophonmw_loongarch_tar_url
        judge_ret $? "Downloaded"$sophonmw_loongarch_tar

    else
        echo "File already existed: "$sophonmw_loongarch_tar", need not to download!"
    fi

    tar -xzvf $libsophon_loongarch_tar 
    tar -xzvf $sophonmw_loongarch_tar 
    mv -f libsophon_${libsophon_version}_loongarch64/opt/sophon/libsophon-${libsophon_version} ./
    mv -f sophon-mw_${sophonmw_version}_loongarch64/opt/sophon/sophon-ffmpeg_${sophonmw_version} ./
    mv -f sophon-mw_${sophonmw_version}_loongarch64/opt/sophon/sophon-opencv_${sophonmw_version} ./
    rm -rf $libsophon_loongarch_tar 
    rm -rf $sophonmw_loongarch_tar
    rm -rf libsophon_${libsophon_version}_loongarch64
    rm -rf sophon-mw_${sophonmw_version}_loongarch64
    popd

    for python_version in 3.8.2 3.9.0 3.10.0;  do
        if [ ! -f $SAIL_BASIC_PATH/build ]; then
            rm -rf $SAIL_BASIC_PATH/build 
        fi
        mkdir -p $SAIL_BASIC_PATH/build 

        local temp_str=${python_version#3.*}
        local whl_release_path=$whl_result_path/py3${temp_str%.*}
        mkdir -p $whl_release_path 

        pushd $SAIL_BASIC_PATH/build
        if [ $build_tpu_kernrl_off -eq 1 ];then
        cmake -DBUILD_TYPE=loongarch  \
            -DTOOLCHAIN_BASIC_PATH=$SAIL_BASIC_PATH/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1 \
            -DCMAKE_TOOLCHAIN_FILE=$SAIL_BASIC_PATH/cmake/BM168x_LoongArch64/ToolChain_loongarch64_linux.cmake \
            -DPYTHON_EXECUTABLE=$SAIL_BASIC_PATH/pythons/Python-${python_version}/python_${python_version}/bin/python3 \
            -DCUSTOM_PY_LIBDIR=$SAIL_BASIC_PATH/pythons/Python-${python_version}/python_${python_version}/lib \
            -DLIBSOPHON_BASIC_PATH=$loongarch_sdk_path/libsophon-${libsophon_version} \
            -DFFMPEG_BASIC_PATH=$loongarch_sdk_path/sophon-ffmpeg_${sophonmw_version} \
            -DOPENCV_BASIC_PATH=$loongarch_sdk_path/sophon-opencv_${sophonmw_version} \
            -DTPUKERNRL_OFF=1 \
            ..
        else
        cmake -DBUILD_TYPE=loongarch  \
            -DTOOLCHAIN_BASIC_PATH=$SAIL_BASIC_PATH/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1 \
            -DCMAKE_TOOLCHAIN_FILE=$SAIL_BASIC_PATH/cmake/BM168x_LoongArch64/ToolChain_loongarch64_linux.cmake \
            -DPYTHON_EXECUTABLE=$SAIL_BASIC_PATH/pythons/Python-${python_version}/python_${python_version}/bin/python3 \
            -DCUSTOM_PY_LIBDIR=$SAIL_BASIC_PATH/pythons/Python-${python_version}/python_${python_version}/lib \
            -DLIBSOPHON_BASIC_PATH=$loongarch_sdk_path/libsophon-${libsophon_version} \
            -DFFMPEG_BASIC_PATH=$loongarch_sdk_path/sophon-ffmpeg_${sophonmw_version} \
            -DOPENCV_BASIC_PATH=$loongarch_sdk_path/sophon-opencv_${sophonmw_version} \
            ..
        fi

        core_nums=`cat /proc/cpuinfo| grep "processor"| wc -l`
        make pysail -j${core_nums}
        judge_ret $? "build_loongarch_pysail, python"$python_version",SDK version: "$1",libsophon: "$2", sophonmw: "$3
        popd

        pushd $SAIL_BASIC_PATH/python/loongarch64
        chmod +x sophon_loongarch64_whl.sh
        ./sophon_loongarch64_whl.sh  
        judge_ret $? "build loongarch64 python"$python_version" wheel"
        mv ./dist/sophon*.whl $whl_release_path
        popd

    done
}

function fill_sources() {
    cp -r $SAIL_BASIC_PATH/3rdparty $SOURCECODE_PATH/
    cp -r $SAIL_BASIC_PATH/build_unix.cmake $SOURCECODE_PATH/
    cp -r $SAIL_BASIC_PATH/build_win.cmake $SOURCECODE_PATH/

    mkdir -p $SOURCECODE_PATH/cmake
    cp -r $SAIL_BASIC_PATH/cmake $SOURCECODE_PATH/

    cp -r $SAIL_BASIC_PATH/CMakeLists.txt $SOURCECODE_PATH/

    mkdir -p $SOURCECODE_PATH/docs
    cp -r $SAIL_BASIC_PATH/docs/common $SOURCECODE_PATH/docs/
    cp -r $SAIL_BASIC_PATH/docs/source_common $SOURCECODE_PATH/docs/
    cp -r $SAIL_BASIC_PATH/docs/source_zh $SOURCECODE_PATH/docs/
    cp -r $SAIL_BASIC_PATH/docs/source_en $SOURCECODE_PATH/docs/
    cp -r $SAIL_BASIC_PATH/docs/Makefile $SOURCECODE_PATH/docs/

    cp -r $SAIL_BASIC_PATH/include $SOURCECODE_PATH/
    
    mkdir -p $SOURCECODE_PATH/python
    mkdir -p $SOURCECODE_PATH/python/pcie
    mkdir -p $SOURCECODE_PATH/python/pcie/dist
    mkdir -p $SOURCECODE_PATH/python/pcie/sophon
    cp -r $SAIL_BASIC_PATH/python/__init__.py $SOURCECODE_PATH/python/
    cp -r $SAIL_BASIC_PATH/python/pcie/MANIFEST.in $SOURCECODE_PATH/python/pcie/
    cp -r $SAIL_BASIC_PATH/python/pcie/setup.py $SOURCECODE_PATH/python/pcie/
    cp -r $SAIL_BASIC_PATH/python/pcie/sophon_pcie_whl.sh $SOURCECODE_PATH/python/pcie/

    mkdir -p $SOURCECODE_PATH/python
    mkdir -p $SOURCECODE_PATH/python/soc
    mkdir -p $SOURCECODE_PATH/python/soc/dist
    mkdir -p $SOURCECODE_PATH/python/soc/sophon
    cp -r $SAIL_BASIC_PATH/python/soc/MANIFEST.in $SOURCECODE_PATH/python/soc/
    cp -r $SAIL_BASIC_PATH/python/soc/setup.py $SOURCECODE_PATH/python/soc/
    cp -r $SAIL_BASIC_PATH/python/soc/sophon_soc_whl.sh $SOURCECODE_PATH/python/soc/

    mkdir -p $SOURCECODE_PATH/python/arm_pcie
    mkdir -p $SOURCECODE_PATH/python/arm_pcie/dist
    mkdir -p $SOURCECODE_PATH/python/arm_pcie/sophon
    cp -r $SAIL_BASIC_PATH/python/arm_pcie/MANIFEST.in $SOURCECODE_PATH/python/arm_pcie/
    cp -r $SAIL_BASIC_PATH/python/arm_pcie/setup.py $SOURCECODE_PATH/python/arm_pcie/
    cp -r $SAIL_BASIC_PATH/python/arm_pcie/sophon_arm_pcie_whl.sh $SOURCECODE_PATH/python/arm_pcie/


    mkdir -p $SOURCECODE_PATH/python/loongarch64
    mkdir -p $SOURCECODE_PATH/python/loongarch64/dist
    mkdir -p $SOURCECODE_PATH/python/loongarch64/sophon
    cp -r $SAIL_BASIC_PATH/python/loongarch64/MANIFEST.in $SOURCECODE_PATH/python/loongarch64/
    cp -r $SAIL_BASIC_PATH/python/loongarch64/setup.py $SOURCECODE_PATH/python/loongarch64/
    cp -r $SAIL_BASIC_PATH/python/loongarch64/sophon_loongarch64_whl.sh $SOURCECODE_PATH/python/loongarch64/

    mkdir -p $SOURCECODE_PATH/python/riscv
    mkdir -p $SOURCECODE_PATH/python/riscv/dist
    mkdir -p $SOURCECODE_PATH/python/riscv/sophon
    cp -r $SAIL_BASIC_PATH/python/riscv/MANIFEST.in $SOURCECODE_PATH/python/riscv/
    cp -r $SAIL_BASIC_PATH/python/riscv/setup.py $SOURCECODE_PATH/python/riscv/
    cp -r $SAIL_BASIC_PATH/python/riscv/sophon_riscv_whl.sh $SOURCECODE_PATH/python/riscv/

    mkdir -p $SOURCECODE_PATH/python/windows
    mkdir -p $SOURCECODE_PATH/python/windows/dist
    mkdir -p $SOURCECODE_PATH/python/windows/sophon
    cp -r $SAIL_BASIC_PATH/python/windows/MANIFEST.in $SOURCECODE_PATH/python/windows/
    cp -r $SAIL_BASIC_PATH/python/windows/setup.py $SOURCECODE_PATH/python/windows/
    cp -r $SAIL_BASIC_PATH/python/windows/sophon_windows_whl.sh $SOURCECODE_PATH/python/windows/

    mkdir -p $SOURCECODE_PATH/python/sw64
    mkdir -p $SOURCECODE_PATH/python/sw64/dist
    mkdir -p $SOURCECODE_PATH/python/sw64/sophon
    cp -r $SAIL_BASIC_PATH/python/sw64/MANIFEST.in $SOURCECODE_PATH/python/sw64/
    cp -r $SAIL_BASIC_PATH/python/sw64/setup.py $SOURCECODE_PATH/python/sw64/
    cp -r $SAIL_BASIC_PATH/python/sw64/sophon_sw64_whl.sh $SOURCECODE_PATH/python/sw64/

    cp -r $SAIL_BASIC_PATH/src $SOURCECODE_PATH/
    cp -r $SAIL_BASIC_PATH/pyis $SOURCECODE_PATH/
    cp -r $SAIL_BASIC_PATH/git_version $SOURCECODE_PATH/
    cp -r $SAIL_BASIC_PATH/README.md $SOURCECODE_PATH/


    mkdir -p $SOURCECODE_PATH/sample
    cp -r $SAIL_BASIC_PATH/sample $SOURCECODE_PATH/
}

function build_docs(){
pushd $SAIL_BASIC_PATH/docs
    make clean
    make pdf  LANG=zh; ret=$?;   if [ $ret -ne 0 ]; then printf "make SophonSDK_doc zh pdf error"; exit 1; fi
    mv build/SOPHON-SAIL_zh.pdf $SAIL_RESULT_PATH/
    rm -rf build
    make pdf  LANG=en; ret=$?;   if [ $ret -ne 0 ]; then printf "make SophonSDK_doc en pdf error"; exit 1; fi
    mv build/SOPHON-SAIL_en.pdf $SAIL_RESULT_PATH/
popd
}

if [ "${target}" == "win" ]; then
    echo "-------------------------Start Build docs ----------------------------------------"
    build_docs
    echo "------------------------ Start packaging source code -----------------------------"
    fill_sources
    exit $?
fi

echo "-------------------------Start Build docs ----------------------------------------"
build_docs
echo "-------------------------Start Download pythons ----------------------------------"
download_pythons
echo "-------------------------Start Download loongarch toolchain ----------------------"
download_loong_toolchain
echo "-------------------------Start build soc pysail with Release_22.12.01-sdk --------"
build_soc_pysail "v22.12.01" "Master_20221227_195517" "0.4.4" "Release_20221227_040823" "0.5.1" "BM1684_BM1684X" "1"
echo "-------------------------Start build arm pcie pysail with Release_22.12.01-sdk ---"
build_arm_pcie_pysail "v22.12.01" "Release_20221227_025400" "0.4.4" "Release_20221227_040823" "0.5.1" "1"
echo "-------------------------Start build soc pysail with Release_23.03.01-sdk --------"
build_soc_pysail "v23.03.01" "Release_20230327_063808" "0.4.6" "Release_20230327_040051" "0.6.0" "BM1684_BM1684X"
echo "-------------------------Start build arm pcie pysail with Release_23.03.01-sdk ---"
build_arm_pcie_pysail "v23.03.01" "Release_20230327_025400" "0.4.6" "Release_20230327_040051" "0.6.0"
echo "-------------------------Start build soc pysail with Release_23.05.01-sdk --------"
build_soc_pysail "v23.05.01" "Release_20230605_054900" "0.4.8" "Release_20230605_032400" "0.6.3" "BM1684_BM1684X"
echo "-------------------------Start build arm pcie pysail with Release_23.05.01-sdk ---"
build_arm_pcie_pysail "v23.05.01" "Release_20230605_025400" "0.4.8" "Release_20230605_032400" "0.6.3"
echo "-------------------------Start build loongarch pysail with Release_23.05.01-sdk ---"
build_loongarch_pysail "v23.05.01" "Release_20230605_025400" "0.4.8" "Release_20230605_032400" "0.6.3"
echo "-------------------------Start build soc pysail with Release_23.10.01-sdk ---------"
build_soc_pysail "v23.10.01" "Release_20231116_213307" "0.5.0" "Release_20231116_113811" "0.7.3" "BM1684_BM1684X"
echo "-------------------------Start build arm pcie pysail with Release_23.10.01-sdk ----"
build_arm_pcie_pysail "v23.10.01" "Release_20231117_103705" "0.5.0" "Release_20231116_113811" "0.7.3"
echo "-------------------------Start build loongarch pysail with Release_23.10.01-sdk ---"
build_loongarch_pysail "v23.10.01" "Release_20231117_103705" "0.5.0" "Release_20231116_113811" "0.7.3"
echo "-------------------------Start build soc pysail with Release_24.04.01-sdk ---------"
build_soc_pysail "v24.04.01" "Release_20240624_155933" "0.5.1" "Release_20240619_173705" "0.10.0" "BM1684_BM1684X"
echo "-------------------------Start build arm pcie pysail with Release_24.04.01-sdk ----"
build_arm_pcie_pysail "v24.04.01" "Release_20240624_160933" "0.5.1" "Release_20240619_173705" "0.10.0"
echo "-------------------------Start build loongarch pysail with Release_24.04.01-sdk ---"
build_loongarch_pysail "v24.04.01" "Release_20240624_160933" "0.5.1" "Release_20240619_173705" "0.10.0"

echo "-------------------------Start build soc pysail with Release_23.09.01_LTS SP2-sdk --"
build_soc_pysail_LTS "23.09.01_LTS_SP2" "LTS_2309_20240116_212937" "0.4.9" "LTS_2309_20240116_152830" "0.8.0" "BM1684_BM1684X"

echo "-------------------------Start build soc pysail with Release_23.09.01_LTS SP3-sdk --"
build_soc_pysail_LTS "23.09.01_LTS_SP3" "LTS_2309_20240705_161328" "0.5.1" "LTS_2309_20240708_013900" "0.11.0" "BM1684_BM1684X"

echo "-------------------------Start build soc pysail with Gemini-sdk_v1.5 --------"
build_soc_pysail "v1.5" "20240328_164638" "0.4.9" "20240403_002500" "1.5.0" "BM1688" 
echo "-------------------------Start build soc pysail with Gemini-sdk_v1.6 --------"
build_soc_pysail "v1.6" "20240520_003934" "0.4.9" "20240511_002500" "1.6.0" "BM1688" 
echo "-------------------------Start build soc pysail with Gemini-sdk_v1.7 --------"
build_soc_pysail "v1.7" "20240621_003144" "0.4.9" "20240621_002500" "1.7.0" "BM1688" 

echo "------------------------ Start packaging source code -----------------------------"
fill_sources

pushd $SAIL_RESULT_PATH
    tar -czvf sophon-sail_$SAIL_VERSION.tar.gz sophon-sail
    rm -rf sophon-sail
popd

echo "-------------------------Sophon-sail Packaging End -------------------------------"
