cmake_minimum_required(VERSION 3.18)
project(d2d_gpu LANGUAGES CXX)

# D2D_HOST_ONLY: build only the CUDA-free library + tests (decode, NMS,
#   letterbox reference, stats, frame I/O). Runs anywhere, incl. macOS.
# D2D_COMPILE_ONLY: compile the CUDA / TensorRT sources without linking.
#   CI uses this with nvcc + the TensorRT *header* packages only, so API
#   mistakes are caught without a GPU or the multi-GB TensorRT libraries.
option(D2D_HOST_ONLY "Build only the CUDA-free host library and tests" OFF)
option(D2D_COMPILE_ONLY "Compile GPU sources without linking (CI)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

enable_testing()

add_library(d2d_host STATIC
  src/letterbox.cpp src/postprocess.cpp src/stats.cpp src/frames.cpp)
target_include_directories(d2d_host PUBLIC include)

add_executable(test_host tests/test_host.cpp)
target_link_libraries(test_host PRIVATE d2d_host)
add_test(NAME host COMMAND test_host WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(host PROPERTIES LABELS host)

if(D2D_HOST_ONLY)
  return()
endif()

# ── CUDA ──────────────────────────────────────────────────────────────────────
# Default to a fat binary (T4, A100, A10/RTX 30xx, Orin, L4/RTX 40xx). On the
# target machine pass -DCMAKE_CUDA_ARCHITECTURES=native.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86;87;89")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 17)

# ── TensorRT ──────────────────────────────────────────────────────────────────
set(TENSORRT_ROOT "" CACHE PATH "TensorRT install prefix (tarball installs)")
set(_trt_hints ${TENSORRT_ROOT} /usr /usr/local/tensorrt /opt/tensorrt)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${_trt_hints} PATH_SUFFIXES include include/x86_64-linux-gnu include/aarch64-linux-gnu)
if(NOT TENSORRT_INCLUDE_DIR)
  message(FATAL_ERROR "NvInfer.h not found; set -DTENSORRT_ROOT=<prefix>")
endif()
message(STATUS "TensorRT headers: ${TENSORRT_INCLUDE_DIR}")

set(GPU_SOURCES src/engine.cpp src/pipeline.cpp src/preprocess.cu)
set(TOOLS build_engine bench multistream)

if(D2D_COMPILE_ONLY)
  add_library(d2d_gpu_objects OBJECT ${GPU_SOURCES}
    tools/build_engine.cpp tools/bench.cpp tools/multistream.cpp tests/test_preprocess_gpu.cu)
  target_include_directories(d2d_gpu_objects PRIVATE include tests ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(d2d_gpu_objects PRIVATE CUDA::cudart)
  return()
endif()

foreach(lib nvinfer nvonnxparser nvinfer_plugin)
  find_library(${lib}_LIB ${lib}
    HINTS ${_trt_hints} PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)
  if(NOT ${lib}_LIB)
    message(FATAL_ERROR "lib${lib} not found; set -DTENSORRT_ROOT=<prefix>")
  endif()
endforeach()

add_library(d2d_gpu STATIC ${GPU_SOURCES})
target_include_directories(d2d_gpu PUBLIC include ${TENSORRT_INCLUDE_DIR})
target_link_libraries(d2d_gpu PUBLIC d2d_host CUDA::cudart
  ${nvinfer_LIB} ${nvonnxparser_LIB} ${nvinfer_plugin_LIB})
if(TARGET CUDA::nvtx3)
  target_link_libraries(d2d_gpu PUBLIC CUDA::nvtx3)
endif()

foreach(tool ${TOOLS})
  add_executable(d2d_${tool} tools/${tool}.cpp)
  target_link_libraries(d2d_${tool} PRIVATE d2d_gpu)
endforeach()
find_package(Threads REQUIRED)
target_link_libraries(d2d_multistream PRIVATE Threads::Threads)

add_executable(test_preprocess_gpu tests/test_preprocess_gpu.cu)
target_include_directories(test_preprocess_gpu PRIVATE tests)
target_link_libraries(test_preprocess_gpu PRIVATE d2d_gpu)
add_test(NAME gpu_preprocess COMMAND test_preprocess_gpu)
set_tests_properties(gpu_preprocess PROPERTIES LABELS gpu)
