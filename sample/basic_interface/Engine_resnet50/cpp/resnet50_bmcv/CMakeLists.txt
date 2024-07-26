cmake_minimum_required(VERSION 3.10)
project(resnet50_bmcv)
 

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
# SET(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-O3")

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

if (${TARGET_ARCH} STREQUAL "pcie")
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})

    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    # use sail
    set(SAIL_DIR  /opt/sophon/sophon-sail/lib/cmake)
    find_package(SAIL REQUIRED)
    include_directories(${SAIL_INCLUDE_DIRS})
    link_directories(${SAIL_LIB_DIRS})

    set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
    find_package(FFMPEG REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    link_directories(${FFMPEG_LIB_DIRS})
    
    set(OpenCV_DIR /opt/sophon/sophon-opencv-latest/lib/cmake/)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})

    message(STATUS "SAIL_INCLUDE_DIRS: ${SAIL_INCLUDE_DIRS}")
    message(STATUS "SAIL_LIB_DIRS: ${SAIL_LIB_DIRS}")
    message(STATUS "FFMPEG_INCLUDE_DIRS: ${FFMPEG_INCLUDE_DIRS}")
    message(STATUS "OPENCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

    aux_source_directory(. SRC_FILES)
    add_executable(resnet50_bmcv.pcie ${SRC_FILES})
    target_link_libraries(resnet50_bmcv.pcie  ${FFMPEG_LIBS} ${OpenCV_LIBS} sail)

elseif (${TARGET_ARCH} STREQUAL "soc")
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    set(BM_LIBS bmlib bmrt bmcv yuv)
    set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)

    message("SOC-SDK: ${SDK}")
    include_directories("${SDK}/include/")
    include_directories("${SDK}/include/opencv4/")
    include_directories("${SDK}/include/sail/")
    link_directories("${SDK}/lib/")
    
    message("SDK: " ${SDK})

    aux_source_directory(. SRC_FILES)
    add_executable(resnet50_bmcv.soc ${SRC_FILES})
    target_link_libraries(resnet50_bmcv.soc ${BM_LIBS} ${OPENCV_LIBS} sail)

endif ()