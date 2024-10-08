cmake_minimum_required(VERSION 3.10)
project(sail VERSION 3.1.0)

set(BUILD_WINDOWS ON)
add_definitions(-DWIN=1)

option(ONLY_RUNTIME "OFF for USE OpenCV,BM-FFMPEG,BMCV"  OFF)
option(BUILD_PYSAIL "ON for Build sail with python"      ON)

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

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_CXX_STANDARD     14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
set(CMAKE_CXX_FLAGS_DEBUG    "-g")
set(CMAKE_CXX_FLAGS_RELEASE  "-O3")

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/build_windows)
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

if(DEFINED TPUKERNRL_OFF)
add_definitions(-DTPUKERNRL_OFF=1)
endif()


if(NOT DEFINED TARGET_TYPE)
     message(FATAL_ERROR "Please set TARGET_TYPE!")
endif()

if(NOT DEFINED RUNTIME_LIB)
    message(FATAL_ERROR "Please set RUNTIME_LIB!")
endif()

if(TARGET_TYPE STREQUAL "release")
    if(RUNTIME_LIB STREQUAL "MD")
        set(CMAKE_CXX_FLAGS_RELEASE "/MD")
        set(CMAKE_C_FLAGS_RELEASE "/MD")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "/MT")
        set(CMAKE_C_FLAGS_RELEASE "/MT")
    endif()
else()
    if(RUNTIME_LIB STREQUAL "MD")
        set(CMAKE_CXX_FLAGS_DEBUG "/MDd")
        set(CMAKE_C_FLAGS_DEBUG "/MDd")
    else()
        set(CMAKE_CXX_FLAGS_DEBUG "/MTd")
        set(CMAKE_C_FLAGS_DEBUG "/MTd")
    endif()
endif()

if(NOT DEFINED LIBSOPHON_DIR)
    message(FATAL_ERROR "Please set LIBSOPHON_DIR!")
endif()
find_package(LIBSOPHON REQUIRED)
include_directories(${LIBSOPHON_INCLUDE_DIRS})
message(STATUS "LIBSOPHON_INCLUDE_DIRS: ${LIBSOPHON_INCLUDE_DIRS}")
set(common_inc_dirs ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(basic_include_path ${LIBSOPHON_INCLUDE_DIRS})
set(basic_lib_path ${LIBSOPHON_LIB_DIRS})
set(basic_link_libs libbmlib.lib libbmrt.lib)

if (WITH_FFMPEG)
    add_definitions(-DUSE_FFMPEG=1)
    if(NOT DEFINED FFMPEG_DIR)
        message(FATAL_ERROR "Please set FFMPEG_DIR!")
    endif()
    find_package(FFMPEG REQUIRED)
    set(basic_include_path ${basic_include_path} ${FFMPEG_INCLUDE_DIRS})
    set(basic_lib_path ${basic_lib_path} ${FFMPEG_LIB_DIRS})
    set(basic_link_libs ${basic_link_libs} ${FFMPEG_LIBS})
    message(STATUS "FFMPEG_INCLUDE_DIRS: ${FFMPEG_INCLUDE_DIRS}")
    message(STATUS "FFMPEG_LIB_DIRS: ${FFMPEG_LIB_DIRS}")
endif()

if (WITH_BMCV)
    add_definitions(-DUSE_BMCV=1)
    set(basic_link_libs ${basic_link_libs} libbmcv.lib)
endif()

if (WITH_OPENCV)
    if(NOT DEFINED OPENCV_DIR)
        message(FATAL_ERROR "Please set OPENCV_DIR!")
    endif()
    add_definitions(-DUSE_OPENCV=1)
    find_package(OPENCV REQUIRED)
    set(basic_include_path ${basic_include_path} ${OPENCV_INCLUDE_DIRS})
    set(basic_link_libs ${basic_link_libs} ${OPENCV_LIBS})
    message(STATUS "OPENCV_INCLUDE_DIRS: ${OPENCV_INCLUDE_DIRS}")
endif()

message(STATUS "common_inc_dirs: ${common_inc_dirs}")
message(STATUS "basic_include_path: ${basic_include_path}")
message(STATUS "basic_lib_path: ${basic_lib_path}")
message(STATUS "basic_link_libs: ${basic_link_libs}")



message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

include_directories(${common_inc_dirs} ${basic_include_path})
link_directories(${basic_lib_path})

add_subdirectory(src)