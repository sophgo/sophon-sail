cmake_minimum_required(VERSION 3.10)
project(sail VERSION 3.10.2)

option(ONLY_RUNTIME "OFF for USE OpenCV,BM-FFMPEG,BMCV"  OFF)
option(BUILD_PYSAIL "ON for Build sail with python"      ON)

if (NOT DEFINED BUILD_TYPE)
    set(BUILD_TYPE "pcie")
endif()

if (NOT DEFINED LOCAL_ARCH)
    set(LOCAL_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
endif()
message(STATUS "CMAKE_HOST_SYSTEM_PROCESSOR: ${LOCAL_ARCH}")

if (NOT DEFINED PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE python3)
else()
    if (DEFINED CUSTOM_PY_LIBDIR)
    string(FIND ${CUSTOM_PY_LIBDIR} ./ res)
    if (${res} STREQUAL "0")
        string(REPLACE ./  ${PROJECT_BINARY_DIR}/ CUSTOM_PY_LIBDIR ${CUSTOM_PY_LIBDIR})
    endif()
    string(FIND ${CUSTOM_PY_LIBDIR} / res)
    if (NOT ${res} STREQUAL "0")
        set(CUSTOM_PY_LIBDIR ${PROJECT_BINARY_DIR}/${CUSTOM_PY_LIBDIR})
    endif()
    SET(ENV{LD_LIBRARY_PATH} ${CUSTOM_PY_LIBDIR}:$ENV{LD_LIBRARY_PATH})
    endif()
endif()

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH    ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_MODULE_PATH      ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD     14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
set(CMAKE_CXX_FLAGS_DEBUG    "-g")
set(CMAKE_CXX_FLAGS_RELEASE  "-O3")

set(BUILD_X86_PCIE OFF)
set(BUILD_SOC      OFF)
set(BUILD_ARM_PCIE OFF)
set(BUILD_LOONGARCH OFF)
set(BUILD_RISCV OFF)
set(BUILD_SW64 OFF)

set(CROSS_COMPILE OFF)

# native compile
if (("${LOCAL_ARCH}" STREQUAL "riscv64") OR
    ("${LOCAL_ARCH}" STREQUAL "sw_64") OR
    ("${LOCAL_ARCH}" STREQUAL "loongarch64"))
    set(BUILD_X86_PCIE ON)
    set(CMAKE_INSTALL_PREFIX /opt/sophon)
elseif ("${BUILD_TYPE}" STREQUAL "pcie")
    set(BUILD_X86_PCIE ON)
    if ("${LOCAL_ARCH}" STREQUAL "x86_64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64 -msse4.1 -mtune=native")
    elseif ("${LOCAL_ARCH}" STREQUAL "aarch64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -mtune=native")
    endif()
    set(CMAKE_INSTALL_PREFIX /opt/sophon)
# cross compile
elseif("${BUILD_TYPE}" STREQUAL "soc")
    add_definitions(-DIS_SOC_MODE=1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
    set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/build_soc)
    set(BUILD_SOC      ON)
    set(CROSS_COMPILE ON)
elseif("${BUILD_TYPE}" STREQUAL "arm_pcie")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
    set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/build_arm_pcie)
    set(BUILD_ARM_PCIE ON)
    set(CROSS_COMPILE ON)
elseif("${BUILD_TYPE}" STREQUAL "loongarch")
    set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/build_loongarch)
    set(BUILD_LOONGARCH ON)
    set(CROSS_COMPILE ON)
elseif("${BUILD_TYPE}" STREQUAL "riscv")
    set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/build_riscv)
    set(BUILD_RISCV ON)
    set(CROSS_COMPILE ON)
elseif("${BUILD_TYPE}" STREQUAL "sw_64")
    set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/build_sw64)
    set(BUILD_SW64 ON)
    set(CROSS_COMPILE ON)
else()
    MESSAGE(FATAL_ERROR "ERROR BUILD_TYPE!")
endif()

message(STATUS "CROSS_COMPILE: ${CROSS_COMPILE}")

if (DEFINED INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${INSTALL_PREFIX})
endif()

if (ONLY_RUNTIME)
    set(WITH_OPENCV OFF)
    set(WITH_FFMPEG OFF)
    set(WITH_BMCV OFF)
else()
    set(WITH_OPENCV ON)
    set(WITH_FFMPEG ON)
    set(WITH_BMCV ON)
endif()

message(STATUS "WITH_OPENCV: ${WITH_OPENCV}")
message(STATUS "WITH_FFMPEG: ${WITH_FFMPEG}")
message(STATUS "WITH_BMCV: ${WITH_BMCV}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

option(USE_IPC "Enable IPC module" OFF)
if((NOT ONLY_RUNTIME) AND USE_IPC)
    add_definitions(-DUSE_IPC=1)
endif()

if(DEFINED TPUKERNRL_OFF)
add_definitions(-DTPUKERNRL_OFF=1)
endif()

if (NOT CROSS_COMPILE)
    if (DEFINED LIBSOPHON_BASIC_PATH)
        message(STATUS "using customized LIBSOPHON_BASIC_PATH")
        set(LIBSOPHON_INCLUDE_DIRS ${LIBSOPHON_BASIC_PATH}/include)
        set(LIBSOPHON_LIB_DIRS ${LIBSOPHON_BASIC_PATH}/lib)
    else()
        find_package(libsophon REQUIRED)
    endif()
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    message(STATUS "LIBSOPHON_INCLUDE_DIRS: ${LIBSOPHON_INCLUDE_DIRS}")
    set(common_inc_dirs ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set(basic_include_path ${LIBSOPHON_INCLUDE_DIRS})
    set(basic_lib_path ${LIBSOPHON_LIB_DIRS})
    set(basic_link_libs bmlib bmrt)

    if (WITH_FFMPEG)
        add_definitions(-DUSE_FFMPEG=1)
        if (DEFINED FFMPEG_BASIC_PATH)
            message(STATUS "using customized FFMPEG_BASIC_PATH")
            set(FFMPEG_INCLUDE_DIRS ${FFMPEG_BASIC_PATH}/include)
            set(FFMPEG_LIB_DIRS ${FFMPEG_BASIC_PATH}/lib)
            file(GLOB FFMPEG_LIBS RELATIVE ${FFMPEG_LIB_DIRS} ${FFMPEG_LIB_DIRS}/*.so)
        else()
            set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
            find_package(FFMPEG REQUIRED)
        endif()
        set(basic_include_path ${basic_include_path} ${FFMPEG_INCLUDE_DIRS})
        set(basic_lib_path ${basic_lib_path} ${FFMPEG_LIB_DIRS})
        set(basic_link_libs ${basic_link_libs} ${FFMPEG_LIBS})
        message(STATUS "FFMPEG_INCLUDE_DIRS: ${FFMPEG_INCLUDE_DIRS}")
        message(STATUS "FFMPEG_LIB_DIRS: ${FFMPEG_LIB_DIRS}")
    endif()

    if (WITH_BMCV)
        add_definitions(-DUSE_BMCV=1)
        set(basic_link_libs ${basic_link_libs} ${the_libbmcv.so})
    endif()

    if (WITH_OPENCV)
        if (DEFINED OPENCV_BASIC_PATH)
            message(STATUS "using customized OPENCV_BASIC_PATH")
            set(OpenCV_INCLUDE_DIRS ${OPENCV_BASIC_PATH}/include/opencv4)
            set(OpenCV_LIB_DIRS ${OPENCV_BASIC_PATH}/lib)
            file(GLOB OpenCV_LIBS RELATIVE ${OpenCV_LIB_DIRS} ${OpenCV_LIB_DIRS}/*.so)
        else()
            set(OpenCV_DIR /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
            find_package(OpenCV REQUIRED)
        endif()
        add_definitions(-DUSE_OPENCV=1)
        set(basic_include_path ${basic_include_path} ${OpenCV_INCLUDE_DIRS})
        set(basic_lib_path ${basic_lib_path} ${OpenCV_LIB_DIRS})
        set(basic_link_libs ${basic_link_libs} ${OpenCV_LIBS})
        message(STATUS "OPENCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
    endif()

    message(STATUS "common_inc_dirs: ${common_inc_dirs}")
    message(STATUS "basic_include_path: ${basic_include_path}")
    message(STATUS "basic_lib_path: ${basic_lib_path}")
    message(STATUS "basic_link_libs: ${basic_link_libs}")
endif()


if (CROSS_COMPILE)
    if (NOT DEFINED LIBSOPHON_BASIC_PATH)
        message(FATAL_ERROR "Please set LIBSOPHON_BASIC_PATH!")
    endif()
    string(FIND ${LIBSOPHON_BASIC_PATH} / res)
    if (NOT ${res} MATCHES 0)
        set(LIBSOPHON_BASIC_PATH ${PROJECT_BINARY_DIR}/${LIBSOPHON_BASIC_PATH})
    endif()

    set(LIBSOPHON_INCLUDE_DIRS ${LIBSOPHON_BASIC_PATH}/include)
    set(LIBSOPHON_LIB_DIRS ${LIBSOPHON_BASIC_PATH}/lib)

    if (EXISTS ${LIBSOPHON_BASIC_PATH}/include/bmcv_api_ext.h)
        # read bmcv_api_ext.h content
        file(READ ${LIBSOPHON_BASIC_PATH}/include/bmcv_api_ext.h HEADER_CONTENT)
        # add BMCV_VERSION_MAJOR definitions
        string(REGEX MATCH "#define BMCV_VERSION_MAJOR ([0-9]+)" INCLUDE_BMCV_VERSION_MAJOR ${HEADER_CONTENT})
        if(INCLUDE_BMCV_VERSION_MAJOR)
            add_definitions(-DBMCV_VERSION_MAJOR=${CMAKE_MATCH_1})
            message(STATUS "WITH_SDK: GeminiSDK")

            string(REGEX MATCH "#define BMCV_VERSION_MINOR ([0-9]+)" INCLUDE_BMCV_VERSION_MINOR ${HEADER_CONTENT})
            if(INCLUDE_BMCV_VERSION_MINOR)
                add_definitions(-DBMCV_VERSION_MINOR=${CMAKE_MATCH_1})
            endif()
        else()
            message(STATUS "WITH_SDK: SOPHONSDK")
        endif()
    endif()

    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    set(common_inc_dirs ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set(basic_include_path ${LIBSOPHON_INCLUDE_DIRS})
    set(basic_lib_path ${LIBSOPHON_LIB_DIRS})
    set(basic_link_libs bmlib bmrt)
    
    if (WITH_FFMPEG)
        if (NOT DEFINED FFMPEG_BASIC_PATH)
            message(FATAL_ERROR "Please set FFMPEG_BASIC_PATH!")
        endif()

        string(FIND ${FFMPEG_BASIC_PATH} / res)
        if (NOT ${res} MATCHES 0)
            set(FFMPEG_BASIC_PATH ${PROJECT_BINARY_DIR}/${FFMPEG_BASIC_PATH})
        endif()

        add_definitions(-DUSE_FFMPEG=1)
        set(FFMPEG_INCLUDE_DIRS ${FFMPEG_BASIC_PATH}/include)
        set(FFMPEG_LIB_DIRS ${FFMPEG_BASIC_PATH}/lib)


        file(GLOB FFMPEG_LIBS RELATIVE ${FFMPEG_LIB_DIRS} ${FFMPEG_LIB_DIRS}/*.so)

        set(basic_include_path ${basic_include_path} ${FFMPEG_INCLUDE_DIRS})
        set(basic_lib_path ${basic_lib_path} ${FFMPEG_LIB_DIRS})
        set(basic_link_libs ${basic_link_libs} ${FFMPEG_LIBS})
        message(STATUS "FFMPEG_INCLUDE_DIRS: ${FFMPEG_INCLUDE_DIRS}")
        message(STATUS "FFMPEG_LIB_DIRS: ${FFMPEG_LIB_DIRS}")
    endif()

    if (WITH_BMCV)
        add_definitions(-DUSE_BMCV=1)
        set(basic_link_libs ${basic_link_libs} bmcv)
    endif()

    if (WITH_OPENCV)
        if (NOT DEFINED OPENCV_BASIC_PATH)
            message(FATAL_ERROR "Please set OPENCV_BASIC_PATH!")
        endif()

        string(FIND ${OPENCV_BASIC_PATH} / res)
        if (NOT ${res} MATCHES 0)
            set(OPENCV_BASIC_PATH ${PROJECT_BINARY_DIR}/${OPENCV_BASIC_PATH})
        endif()

        set(OpenCV_INCLUDE_DIRS ${OPENCV_BASIC_PATH}/include/opencv4)
        set(OpenCV_LIB_DIRS ${OPENCV_BASIC_PATH}/lib)

        file(GLOB OpenCV_LIBS RELATIVE ${OpenCV_LIB_DIRS} ${OpenCV_LIB_DIRS}/*.so)

        add_definitions(-DUSE_OPENCV=1)
        set(basic_include_path ${basic_include_path} ${OpenCV_INCLUDE_DIRS})
        set(basic_lib_path ${basic_lib_path} ${OpenCV_LIB_DIRS})
        set(basic_link_libs ${basic_link_libs} ${OpenCV_LIBS})
        message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
    endif()

    message(STATUS "common_inc_dirs: ${common_inc_dirs}")
    message(STATUS "basic_include_path: ${basic_include_path}")
    message(STATUS "basic_lib_path: ${basic_lib_path}")
    message(STATUS "basic_link_libs: ${basic_link_libs}")
endif()

message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

# read bmcv_api_ext.h content
file(READ ${LIBSOPHON_INCLUDE_DIRS}/bmdef.h BMRT_DEFINE)
# add BMCV_VERSION_MAJOR definitions
if(BMRT_DEFINE MATCHES "int32_t addr_mode;")
    add_definitions(-DBUILD_ENGINELLM=1)
    message(STATUS "build with engine_llm")
else()
    message(STATUS "build without engine_llm")
endif()

if(BMRT_DEFINE MATCHES "bm_runtime_flag_t")
    add_definitions(-DSAIL_WITH_BMRT_FLAG=1)
    set(sail_def_bmrt_flag "add_definitions(-DSAIL_WITH_BMRT_FLAG=1)")
    message(STATUS "build sail with bmrt flag")
else()
    message(STATUS "build sail without bmrt flag")
endif()

include_directories(${common_inc_dirs} ${basic_include_path})
link_directories(${basic_lib_path})

add_subdirectory(src)




