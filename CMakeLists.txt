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

function(lrrt_add_hsaco_kernel name source output_var)
  if(AMDCLANGXX AND CLANG_OFFLOAD_BUNDLER)
    set(bundle "${CMAKE_CURRENT_BINARY_DIR}/${name}.bundle")
    set(hsaco "${CMAKE_CURRENT_BINARY_DIR}/${name}.hsaco")
    set(host_object "${CMAKE_CURRENT_BINARY_DIR}/${name}.host.o")

    add_custom_command(
      OUTPUT "${hsaco}" "${bundle}" "${host_object}"
      COMMAND
        "${AMDCLANGXX}" --offload-device-only
        "--offload-arch=${LRRT_AMDGPU_TARGET}" -x hip -include
        hip/hip_runtime.h "${CMAKE_CURRENT_SOURCE_DIR}/${source}" -o
        "${bundle}"
      COMMAND
        "${CLANG_OFFLOAD_BUNDLER}" -type=o
        "-targets=host-x86_64-unknown-linux-gnu-,hipv4-amdgcn-amd-amdhsa--${LRRT_AMDGPU_TARGET}"
        "-input=${bundle}" "-output=${host_object}" "-output=${hsaco}"
        -unbundle
      DEPENDS "${source}"
      VERBATIM
    )
    add_custom_target(${name}_hsaco DEPENDS "${hsaco}")
    set(${output_var} "${hsaco}" PARENT_SCOPE)
  endif()
endfunction()

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

add_executable(lrrt_saxpy examples/saxpy/saxpy.cpp)
target_link_libraries(lrrt_saxpy PRIVATE lrrt::lrrt)

add_executable(lrrt_fill examples/fill/fill.cpp)
target_link_libraries(lrrt_fill PRIVATE lrrt::lrrt)

add_executable(lrrt_memory_bounds_tests tests/memory_bounds.cpp)
target_link_libraries(lrrt_memory_bounds_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_async_copy_tests tests/async_copy.cpp)
target_link_libraries(lrrt_async_copy_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_handle_registry_tests tests/handle_registry.cpp)
target_link_libraries(lrrt_handle_registry_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_scale_tests tests/scale.cpp)
target_link_libraries(lrrt_scale_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_copy_kernel_tests tests/copy_kernel.cpp)
target_link_libraries(lrrt_copy_kernel_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_event_timing_tests tests/event_timing.cpp)
target_link_libraries(lrrt_event_timing_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_cpp_wrapper_tests tests/cpp_wrapper.cpp)
target_link_libraries(lrrt_cpp_wrapper_tests PRIVATE lrrt::lrrt)

lrrt_add_hsaco_kernel(
  vector_add_kernel examples/vector_add/vector_add_kernel.cpp
  VECTOR_ADD_HSACO
)
lrrt_add_hsaco_kernel(saxpy_kernel examples/saxpy/saxpy_kernel.cpp SAXPY_HSACO)
lrrt_add_hsaco_kernel(fill_kernel examples/fill/fill_kernel.cpp FILL_HSACO)
lrrt_add_hsaco_kernel(scale_kernel tests/scale_kernel.cpp SCALE_HSACO)
lrrt_add_hsaco_kernel(copy_kernel tests/copy_kernel_kernel.cpp COPY_HSACO)

if(VECTOR_ADD_HSACO)
  add_dependencies(lrrt_vector_add vector_add_kernel_hsaco)
  add_dependencies(lrrt_handle_registry_tests vector_add_kernel_hsaco)
  target_compile_definitions(
    lrrt_vector_add
    PRIVATE LRRT_VECTOR_ADD_HSACO="${VECTOR_ADD_HSACO}"
  )
  target_compile_definitions(
    lrrt_handle_registry_tests
    PRIVATE LRRT_VECTOR_ADD_HSACO="${VECTOR_ADD_HSACO}"
  )
endif()
if(SAXPY_HSACO)
  add_dependencies(lrrt_saxpy saxpy_kernel_hsaco)
  target_compile_definitions(
    lrrt_saxpy
    PRIVATE LRRT_SAXPY_HSACO="${SAXPY_HSACO}"
  )
endif()
if(FILL_HSACO)
  add_dependencies(lrrt_fill fill_kernel_hsaco)
  target_compile_definitions(lrrt_fill PRIVATE LRRT_FILL_HSACO="${FILL_HSACO}")
endif()
if(SCALE_HSACO)
  add_dependencies(lrrt_scale_tests scale_kernel_hsaco)
  add_dependencies(lrrt_event_timing_tests scale_kernel_hsaco)
  add_dependencies(lrrt_cpp_wrapper_tests scale_kernel_hsaco)
  target_compile_definitions(
    lrrt_scale_tests
    PRIVATE LRRT_SCALE_HSACO="${SCALE_HSACO}"
  )
  target_compile_definitions(
    lrrt_event_timing_tests
    PRIVATE LRRT_SCALE_HSACO="${SCALE_HSACO}"
  )
  target_compile_definitions(
    lrrt_cpp_wrapper_tests
    PRIVATE LRRT_SCALE_HSACO="${SCALE_HSACO}"
  )
endif()
if(COPY_HSACO)
  add_dependencies(lrrt_copy_kernel_tests copy_kernel_hsaco)
  target_compile_definitions(
    lrrt_copy_kernel_tests
    PRIVATE LRRT_COPY_HSACO="${COPY_HSACO}"
  )
endif()

enable_testing()
add_test(NAME lrrt_memory_bounds_tests COMMAND lrrt_memory_bounds_tests)
add_test(NAME lrrt_async_copy_tests COMMAND lrrt_async_copy_tests)
add_test(NAME lrrt_handle_registry_tests COMMAND lrrt_handle_registry_tests)
add_test(NAME lrrt_scale_tests COMMAND lrrt_scale_tests)
add_test(NAME lrrt_copy_kernel_tests COMMAND lrrt_copy_kernel_tests)
add_test(NAME lrrt_event_timing_tests COMMAND lrrt_event_timing_tests)
add_test(NAME lrrt_cpp_wrapper_tests COMMAND lrrt_cpp_wrapper_tests)
