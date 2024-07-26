#!/bin/bash
# set -ex
shell_dir=$(dirname $(readlink -f "$0"))

SAIL_BASIC_PATH=$shell_dir/..
SAIL_RESULT_PATH=$SAIL_BASIC_PATH/release
VERSION_PATH=$SAIL_BASIC_PATH/git_version
SOURCECODE_PATH=$SAIL_RESULT_PATH/sophon-sail
exec < $VERSION_PATH
read -r line
SAIL_VERSION=$line

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
  else
    echo -e "\033[31m Failed: $2 \033[0m"
    exit 1
  fi
  sleep 1
}

function download_python() {
    py_version=$1
    pythons_path=$SAIL_BASIC_PATH/pythons
    if [ ! -f $pythons_path ]; then
        mkdir -p $pythons_path
    fi
    pushd $pythons_path
    python3 -m dfss --url=open@sophgo.com:/toolchains/pythons/Python-${py_version}.tar.gz
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
    download_python "3.8.2"
    clean_python_tar "3.8.2"
}

function build_soc_pysail() {
    basic_version=$1
    libsophon_version=$2
    sophonmw_version=$3

    libsophon_soc_tar_url=open@sophgo.com:/sophon-sdk/release/${basic_version}/sophon-img/libsophon_soc_${libsophon_version}_aarch64.tar.gz
    sophonmw_url=open@sophgo.com:/sophon-sdk/release/${basic_version}/sophon-mw/sophon-mw-soc_${sophonmw_version}_aarch64.tar.gz

    if [ $# -eq 4 ];then
        build_tpu_kernrl_off=$4
    else
        build_tpu_kernrl_off=0
    fi

    local whl_result_path=$SOURCECODE_PATH/python_wheels
    mkdir -p $whl_result_path
    local whl_result_path=$whl_result_path/soc
    mkdir -p $whl_result_path
    local whl_result_path=$whl_result_path/libsophon-${libsophon_version}_sophonmw-${sophonmw_version}
    mkdir -p $whl_result_path

    soc_sdk_path=$SAIL_BASIC_PATH/soc-sdk
    libsophon_soc_tar=$soc_sdk_path/libsophon_soc_${libsophon_version}_aarch64.tar.gz
    sophonmw_soc_tar=$soc_sdk_path/sophon-mw-soc_${sophonmw_version}_aarch64.tar.gz

    if [ -d $soc_sdk_path ]; then
        rm -rf $soc_sdk_path
    fi
    mkdir -p $soc_sdk_path

    pushd $soc_sdk_path
    if [ ! -f $libsophon_soc_tar ]; then
        echo "Start download file: "$libsophon_soc_tar
        python3 -m dfss --url=$libsophon_soc_tar_url
        judge_ret $? "Downloaded"$libsophon_soc_tar
    else
        echo "File already existed: "$libsophon_soc_tar", need not to download!"
    fi

    if [ ! -f $sophonmw_soc_tar ]; then
        echo "Start download file: "$sophonmw_soc_tar
        python3 -m dfss --url=$sophonmw_url
        judge_ret $? "Downloaded"$sophonmw_soc_tar
    else
        echo "File already existed: "$sophonmw_soc_tar", need not to download!"
    fi

    tar -xzvf $libsophon_soc_tar 
    tar -xzvf $sophonmw_soc_tar 
    mv -f  libsophon_soc_${libsophon_version}_aarch64/opt/sophon/libsophon-${libsophon_version} ./
    mv -f  sophon-mw-soc_${sophonmw_version}_aarch64/opt/sophon/sophon-ffmpeg_${sophonmw_version} ./
    mv -f  sophon-mw-soc_${sophonmw_version}_aarch64/opt/sophon/sophon-opencv_${sophonmw_version} ./
    rm -rf $libsophon_soc_tar 
    rm -rf $sophonmw_soc_tar
    rm -rf libsophon_soc_${libsophon_version}_aarch64
    rm -rf sophon-mw-soc_${sophonmw_version}_aarch64
    popd

    for python_version in 3.8.2 ;  do
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

function build_docs(){
pushd $SAIL_BASIC_PATH/docs
    make clean
    make pdf  LANG=zh; ret=$?;   if [ $ret -ne 0 ]; then printf "make SophonSDK_doc zh pdf error"; exit 1; fi
    rm -rf build
    make pdf  LANG=en; ret=$?;   if [ $ret -ne 0 ]; then printf "make SophonSDK_doc en pdf error"; exit 1; fi
popd
}

function clear_result(){
    rm -rf $SAIL_BASIC_PATH/pythons
    rm -rf $SAIL_BASIC_PATH/release
    rm -rf $SAIL_BASIC_PATH/soc-sdk
    rm -rf $SAIL_BASIC_PATH/python/soc/sophon
    rm -rf $SAIL_BASIC_PATH/build
    rm -rf $SAIL_BASIC_PATH/docs/build
}

echo "-------------------------Start Install dfss --------------------------------------"
pip3 install dfss --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
build_docs
echo "-------------------------Start Download pythons ----------------------------------"
download_pythons
echo "-------------------------Start build soc pysail with Release_23.07.01-sdk --------"
build_soc_pysail "v23.07.01" "0.4.9" "0.7.0"
echo "-------------------------Start build arm pcie pysail with Release_23.07.01-sdk ---"
clear_result
echo "------------------------ Passed ---------------------------------------------------"
