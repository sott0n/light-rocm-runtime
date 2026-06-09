cmake_minimum_required(VERSION 3.18)

project(light_rocm_runtime VERSION 0.1.0 LANGUAGES C CXX)

set(LRRT_AMDGPU_TARGET "gfx1101" CACHE STRING "AMDGPU target for examples")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

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
find_program(
  CLANG_OFFLOAD_BUNDLER
  clang-offload-bundler
  HINTS /opt/rocm/llvm/bin
)

add_library(lrrt src/runtime.cpp)
add_library(lrrt::lrrt ALIAS lrrt)

target_include_directories(
  lrrt
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(lrrt PUBLIC cxx_std_17)

if(HSA_INCLUDE_DIR AND HSA_RUNTIME_LIBRARY)
  target_compile_definitions(lrrt PRIVATE LRRT_ENABLE_HSA=1)
  target_include_directories(lrrt PRIVATE "${HSA_INCLUDE_DIR}")
  target_link_libraries(lrrt PRIVATE "${HSA_RUNTIME_LIBRARY}")
  message(STATUS "HSA runtime enabled: ${HSA_RUNTIME_LIBRARY}")
else()
  message(STATUS "HSA runtime not found; building stub runtime")
endif()

add_executable(lrrt_vector_add examples/vector_add/vector_add.cpp)
target_link_libraries(lrrt_vector_add PRIVATE lrrt::lrrt)

if(AMDCLANGXX AND CLANG_OFFLOAD_BUNDLER)
  set(VECTOR_ADD_BUNDLE "${CMAKE_CURRENT_BINARY_DIR}/vector_add_kernel.bundle")
  set(VECTOR_ADD_HSACO "${CMAKE_CURRENT_BINARY_DIR}/vector_add_kernel.hsaco")
  set(VECTOR_ADD_HOST_OBJECT
      "${CMAKE_CURRENT_BINARY_DIR}/vector_add_kernel.host.o")

  add_custom_command(
    OUTPUT "${VECTOR_ADD_HSACO}" "${VECTOR_ADD_BUNDLE}"
           "${VECTOR_ADD_HOST_OBJECT}"
    COMMAND
      "${AMDCLANGXX}" --offload-device-only
      "--offload-arch=${LRRT_AMDGPU_TARGET}" -x hip -include
      hip/hip_runtime.h
      "${CMAKE_CURRENT_SOURCE_DIR}/examples/vector_add/vector_add_kernel.cpp"
      -o "${VECTOR_ADD_BUNDLE}"
    COMMAND
      "${CLANG_OFFLOAD_BUNDLER}" -type=o
      "-targets=host-x86_64-unknown-linux-gnu-,hipv4-amdgcn-amd-amdhsa--${LRRT_AMDGPU_TARGET}"
      "-input=${VECTOR_ADD_BUNDLE}" "-output=${VECTOR_ADD_HOST_OBJECT}"
      "-output=${VECTOR_ADD_HSACO}" -unbundle
    DEPENDS examples/vector_add/vector_add_kernel.cpp
    VERBATIM
  )
  add_custom_target(lrrt_vector_add_hsaco DEPENDS "${VECTOR_ADD_HSACO}")
  add_dependencies(lrrt_vector_add lrrt_vector_add_hsaco)
  target_compile_definitions(
    lrrt_vector_add
    PRIVATE LRRT_VECTOR_ADD_HSACO="${VECTOR_ADD_HSACO}"
  )
endif()
