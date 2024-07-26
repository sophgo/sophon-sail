cmake_minimum_required(VERSION 3.10)
project(test_yolox)
 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -Wall")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(libsophon REQUIRED)
include_directories(${LIBSOPHON_INCLUDE_DIRS})
link_directories(${LIBSOPHON_LIB_DIRS})

set(SAIL_DIR  /opt/sophon/sophon-sail/lib/cmake)
find_package(SAIL REQUIRED)
include_directories(${SAIL_INCLUDE_DIRS})
link_directories(${SAIL_LIB_DIRS})

set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
find_package(FFMPEG REQUIRED)
include_directories(${FFMPEG_INCLUDE_DIRS})
link_directories(${FFMPEG_LIB_DIRS})

message(STATUS "SAIL_INCLUDE_DIRS: ${SAIL_INCLUDE_DIRS}")
message(STATUS "SAIL_LIB_DIRS: ${SAIL_LIB_DIRS}")
message(STATUS "FFMPEG_INCLUDE_DIRS: ${FFMPEG_INCLUDE_DIRS}")

set(CMAKE_CXX_FLAGS -pthread)

aux_source_directory(. SRC_FILES)
add_executable(test_yolox ${SRC_FILES})
target_link_libraries(test_yolox sail ${OpenCV_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${FFMPEG_LIBS} ${the_libbmcv.so})



