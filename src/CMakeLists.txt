
# set sail source file
if (WITH_OPENCV)
set(CORE_SRC cvwrapper.cpp engine.cpp graph.cpp tensor.cpp tools.cpp base64.cpp engine_multi.cpp decoder_multi.cpp algokit.cpp encoder.cpp tpu_kernel_api.cpp 
perf.cpp engine_llm.cpp
deepsort/model.cpp 
kalmanfilter/kalmanfilter.cpp 
deepsort/linear_assignment.cpp 
deepsort/nn_matching.cpp 
deepsort/tracker.cpp 
deepsort/track.cpp 
deepsort/hungarianoper.cpp 
deepsort/munkres.cpp 
sort/sort_track.cpp 
sort/sort_tracker.cpp 
bytetrack/lapjv.cpp 
bytetrack/STrack.cpp 
bytetrack/utils.cpp 
bytetrack/bytetrack.cpp)
else()
set(CORE_SRC cvwrapper.cpp engine.cpp graph.cpp tensor.cpp tools.cpp base64.cpp engine_multi.cpp decoder_multi.cpp algokit.cpp encoder.cpp tpu_kernel_api.cpp perf.cpp engine_llm.cpp)
endif()

if((NOT ONLY_RUNTIME) AND USE_IPC)
    list(APPEND CORE_SRC ipc.cpp)
endif()

# build cpp api lib
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -Wall")
add_library(sail SHARED ${CORE_SRC})
target_link_libraries(sail ${basic_link_libs})

if (BUILD_PYSAIL)
# build python api lib
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print('{}.{}'.format(sys.version_info.major,sys.version_info.minor))"
    OUTPUT_VARIABLE PY_V
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if (PY_V VERSION_GREATER_EQUAL 3.8)
    add_subdirectory(../3rdparty/pybind11_new pybind.dir)
    message(STATUS,"USE pybind11 new!")
  else ()
    add_subdirectory(../3rdparty/pybind11 pybind.dir)
    message(STATUS,"USE pybind11 old!")
    endif ()

  add_library(pysail SHARED bind.cpp ${CORE_SRC})

  if(CUSTOM_PY_LIBDIR)
    link_directories(${CUSTOM_PY_LIBDIR})
    target_include_directories(pysail PUBLIC ${PYBIND11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
    message(STATUS "CUSTOM_PY_LIBDIR = ${CUSTOM_PY_LIBDIR}")
  else()
    target_include_directories(pysail PUBLIC ${PYBIND11_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/3rdparty/prebuild/include )
    endif()

  target_compile_definitions(pysail PUBLIC -DPYTHON=1)

  if(BUILD_WINDOWS)
    target_link_libraries(pysail PRIVATE ${basic_link_libs} pybind11::module pybind11::embed)
  else()
    target_link_libraries(pysail PRIVATE ${basic_link_libs} pybind11::module)
  endif()
  set(PYSAIL_SUFFIX "${PYTHON_MODULE_EXTENSION}")
  if(BUILD_SOC OR BUILD_ARM_PCIE)
    string(REPLACE "x86_64" "aarch64" PYSAIL_SUFFIX "${PYTHON_MODULE_EXTENSION}")
  endif()
  set_target_properties(pysail PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                          SUFFIX "${PYSAIL_SUFFIX}" 
                                          OUTPUT_NAME "sail")
  message(STATUS "PREFIX: " ${PYTHON_MODULE_PREFIX})
  message(STATUS "SUFFIX: " ${PYSAIL_SUFFIX})
endif()

set(sail_install_path "${CMAKE_INSTALL_PREFIX}/sophon-sail")

set(HEADERS ${CMAKE_SOURCE_DIR}/include/base64.h  
            ${CMAKE_SOURCE_DIR}/include/cvwrapper.h  
            ${CMAKE_SOURCE_DIR}/include/decoder_rawstream.h  
            ${CMAKE_SOURCE_DIR}/include/engine.h  
            ${CMAKE_SOURCE_DIR}/include/engine_llm.h 
            ${CMAKE_SOURCE_DIR}/include/engine_multi.h 
            ${CMAKE_SOURCE_DIR}/include/decoder_multi.h 
            ${CMAKE_SOURCE_DIR}/include/graph.h 
            ${CMAKE_SOURCE_DIR}/include/tensor.h  
            ${CMAKE_SOURCE_DIR}/include/algokit.h  
            ${CMAKE_SOURCE_DIR}/include/tools.h
            ${CMAKE_SOURCE_DIR}/include/tpu_kernel_api.h
            ${CMAKE_SOURCE_DIR}/include/encoder.h)
if((NOT ONLY_RUNTIME) AND USE_IPC)
  list(APPEND HEADERS ${CMAKE_SOURCE_DIR}/include/ipc.h)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")

if(BUILD_WINDOWS)
  INSTALL(TARGETS sail
          ARCHIVE DESTINATION  "${sail_install_path}/lib"
          LIBRARY DESTINATION "${sail_install_path}/lib"
          RUNTIME DESTINATION "${sail_install_path}/bin"
          PUBLIC_HEADER DESTINATION "${sail_install_path}/include/sail")
else()
  INSTALL(TARGETS sail
          LIBRARY DESTINATION "${sail_install_path}/lib"
          PUBLIC_HEADER DESTINATION "${sail_install_path}/include/sail")
endif()

INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/3rdparty/spdlog DESTINATION "${sail_install_path}/include/sail")

configure_file(${CMAKE_SOURCE_DIR}/cmake/SAILConfig.cmake.in ${CMAKE_SOURCE_DIR}/cmake/SAILConfig.cmake)

install(FILES "${CMAKE_SOURCE_DIR}/cmake/SAILConfig.cmake"
        DESTINATION "${sail_install_path}/lib/cmake/")