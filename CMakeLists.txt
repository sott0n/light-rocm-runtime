cmake_minimum_required(VERSION 3.18)

project(light_rocm_runtime VERSION 0.1.0 LANGUAGES C CXX)

set(LRRT_AMDGPU_TARGET "gfx1101" CACHE STRING "AMDGPU target for examples")
set(
  LRRT_TRITON_PYTHON
  "3.13"
  CACHE STRING "Python version or interpreter for Triton examples"
)
option(LRRT_ENABLE_TRITON_EXAMPLES "Build Triton examples" OFF)
option(LRRT_ENABLE_SPARSEWAVE_EXECUTOR "Build the SparseWave executor" OFF)
option(LRRT_ENABLE_IREE_ADAPTER "Build the experimental IREE HAL adapter" OFF)
option(LRRT_ENABLE_HSA "Build with HSA/ROCr support when available" ON)
option(LRRT_BUILD_BENCHMARKS "Build runtime benchmarks" OFF)
option(LRRT_BUILD_TRITON_BENCHMARKS "Build Triton runtime benchmarks" OFF)
set(
  LRRT_IREE_ROOT
  ""
  CACHE PATH "Path to an IREE install or source tree for the IREE HAL adapter"
)
set(
  LRRT_SPARSEWAVE_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sparsewave"
  CACHE PATH "Path to the SparseWave compiler source tree"
)
set(
  LRRT_SPARSEWAVE_BUILD_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/build-sparsewave/llvm"
  CACHE PATH "Path to the SparseWave LLVM build tree"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

find_path(
  HSA_INCLUDE_DIR
  NAMES hsa/hsa.h
  HINTS /opt/rocm/include
)
find_library(
  HSA_RUNTIME_LIBRARY
  NAMES hsa-runtime64
  HINTS /opt/rocm/lib /opt/rocm/lib64
)
find_program(AMDCLANGXX amdclang++ HINTS /opt/rocm/bin /opt/rocm/llvm/bin)
find_program(UV_EXECUTABLE uv HINTS "$ENV{HOME}/.local/bin")
find_program(PYTHON_EXECUTABLE python3)
find_program(
  CLANG_OFFLOAD_BUNDLER
  clang-offload-bundler
  HINTS /opt/rocm/llvm/bin
)
find_program(
  LLVM_READELF
  llvm-readelf
  HINTS /opt/rocm/llvm/bin
)

include(cmake/Hsaco.cmake)
include(cmake/TritonBundle.cmake)
include(cmake/IreeAdapter.cmake)

add_library(
  lrrt
  src/device_synchronization.cpp
  src/event.cpp
  src/launch.cpp
  src/memory.cpp
  src/module.cpp
  src/queue.cpp
  src/runtime.cpp
)
add_library(lrrt::lrrt ALIAS lrrt)

add_library(lrrt_bundle src/bundle.cpp)
add_library(lrrt::bundle ALIAS lrrt_bundle)
target_link_libraries(lrrt_bundle PUBLIC lrrt::lrrt)

set_target_properties(lrrt lrrt_bundle PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(
  lrrt
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(lrrt PUBLIC cxx_std_17)

if(LRRT_ENABLE_HSA AND HSA_INCLUDE_DIR AND HSA_RUNTIME_LIBRARY)
  target_compile_definitions(lrrt PRIVATE LRRT_ENABLE_HSA=1)
  target_include_directories(lrrt PRIVATE "${HSA_INCLUDE_DIR}")
  target_link_libraries(lrrt PRIVATE "${HSA_RUNTIME_LIBRARY}")
  message(STATUS "HSA runtime enabled: ${HSA_RUNTIME_LIBRARY}")
elseif(NOT LRRT_ENABLE_HSA)
  message(STATUS "HSA runtime disabled; building stub runtime")
else()
  message(STATUS "HSA runtime not found; building stub runtime")
endif()

enable_testing()

if(LRRT_ENABLE_SPARSEWAVE_EXECUTOR)
  add_subdirectory(executor/sparsewave)
endif()

if(LRRT_ENABLE_IREE_ADAPTER)
  lrrt_find_iree()
  add_subdirectory(executor/iree)
endif()

add_subdirectory(examples)
add_subdirectory(tests)

if(LRRT_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
