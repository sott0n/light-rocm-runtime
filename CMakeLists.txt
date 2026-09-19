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
set(
  LRRT_BACKEND
  "rocr"
  CACHE STRING "LRRT runtime backend (rocr or light-rocr)"
)
set_property(CACHE LRRT_BACKEND PROPERTY STRINGS rocr light-rocr)
option(
  LRRT_ENABLE_LIGHT_ROCR
  "Build the experimental self-authored light-rocr stack"
  OFF
)
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

string(TOLOWER "${LRRT_BACKEND}" LRRT_SELECTED_BACKEND)
if(
  NOT LRRT_SELECTED_BACKEND STREQUAL "rocr"
  AND NOT LRRT_SELECTED_BACKEND STREQUAL "light-rocr"
)
  message(FATAL_ERROR "LRRT_BACKEND must be rocr or light-rocr")
endif()

enable_testing()

if(LRRT_ENABLE_LIGHT_ROCR OR LRRT_SELECTED_BACKEND STREQUAL "light-rocr")
  add_subdirectory(light-rocr)
endif()

add_library(
  lrrt
  src/aql_producer.cpp
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
  add_executable(
    lrrt_rocr_baseline
    EXCLUDE_FROM_ALL
    tools/rocr_baseline.cpp
  )
  target_compile_features(lrrt_rocr_baseline PRIVATE cxx_std_17)
  target_include_directories(lrrt_rocr_baseline PRIVATE "${HSA_INCLUDE_DIR}")
  target_link_libraries(lrrt_rocr_baseline PRIVATE "${HSA_RUNTIME_LIBRARY}")
endif()

if(LRRT_SELECTED_BACKEND STREQUAL "light-rocr")
  if(NOT LRRT_AMDGPU_TARGET STREQUAL "gfx1101")
    message(FATAL_ERROR "The light-rocr LRRT backend currently requires gfx1101")
  endif()
  if(NOT TARGET light_rocr::transport_hsakmt)
    message(FATAL_ERROR "The light-rocr LRRT backend requires libhsakmt")
  endif()

  target_compile_definitions(lrrt PRIVATE LRRT_ENABLE_LIGHT_ROCR=1)
  target_link_libraries(lrrt PRIVATE light_rocr::transport_hsakmt)

  lrrt_add_hsaco_kernel(
    lrrt_internal_copy_kernel
    src/light_rocr_copy_kernel.cpp
    LRRT_INTERNAL_COPY_HSACO
    -O2
  )
  if(NOT LRRT_INTERNAL_COPY_HSACO)
    message(FATAL_ERROR "The light-rocr backend requires the ROCm compiler")
  endif()
  set(
    LRRT_INTERNAL_COPY_HEADER
    "${CMAKE_CURRENT_BINARY_DIR}/lrrt_internal_copy_kernel.inc"
  )
  add_custom_command(
    OUTPUT "${LRRT_INTERNAL_COPY_HEADER}"
    COMMAND
      "${CMAKE_COMMAND}" "-DINPUT=${LRRT_INTERNAL_COPY_HSACO}"
      "-DOUTPUT=${LRRT_INTERNAL_COPY_HEADER}" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
    DEPENDS
      "${LRRT_INTERNAL_COPY_HSACO}"
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
    VERBATIM
  )
  add_custom_target(
    lrrt_internal_copy_kernel_data
    DEPENDS "${LRRT_INTERNAL_COPY_HEADER}"
  )
  add_dependencies(lrrt lrrt_internal_copy_kernel_data)
  target_include_directories(lrrt PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  message(STATUS "LRRT backend: light-rocr")
elseif(LRRT_ENABLE_HSA AND HSA_INCLUDE_DIR AND HSA_RUNTIME_LIBRARY)
  target_compile_definitions(lrrt PRIVATE LRRT_ENABLE_HSA=1)
  target_include_directories(lrrt PRIVATE "${HSA_INCLUDE_DIR}")
  target_link_libraries(lrrt PRIVATE "${HSA_RUNTIME_LIBRARY}")
  message(STATUS "LRRT backend: ROCr (${HSA_RUNTIME_LIBRARY})")
elseif(NOT LRRT_ENABLE_HSA)
  message(STATUS "LRRT ROCr backend disabled; building stub runtime")
else()
  message(STATUS "LRRT ROCr backend unavailable; building stub runtime")
endif()

if(LRRT_BUILD_BENCHMARKS)
  target_compile_definitions(lrrt PRIVATE LRRT_ENABLE_LAUNCH_PROFILING=1)
endif()

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
