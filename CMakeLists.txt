cmake_minimum_required(VERSION 3.18)

project(light_rocm_runtime VERSION 0.1.0 LANGUAGES C CXX)

set(LRRT_AMDGPU_TARGET "gfx1101" CACHE STRING "AMDGPU target for examples")
set(LRRT_TRITON_PYTHON "3.13" CACHE STRING "Python version or interpreter for Triton examples")
option(LRRT_ENABLE_TRITON_EXAMPLES "Build Triton examples" OFF)
option(LRRT_BUILD_BENCHMARKS "Build runtime benchmarks" OFF)

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

function(lrrt_add_triton_bundle name script output_hsaco_var output_manifest_var)
  if(UV_EXECUTABLE)
    set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/${name}_bundle")
    set(hsaco "${output_dir}/kernels.hsaco")
    set(manifest "${output_dir}/manifest.json")
    set(outputs "${hsaco}" "${manifest}")
    foreach(byproduct IN LISTS ARGN)
      list(APPEND outputs "${output_dir}/${byproduct}")
    endforeach()

    add_custom_command(
      OUTPUT ${outputs}
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}"
      COMMAND
        "${CMAKE_COMMAND}" -E env
        "TRITON_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/triton-cache"
        "UV_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/uv-cache"
        "${UV_EXECUTABLE}" run --python "${LRRT_TRITON_PYTHON}" --no-project --with-requirements
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/triton/requirements.txt" python
        "${CMAKE_CURRENT_SOURCE_DIR}/${script}"
        "--arch=${LRRT_AMDGPU_TARGET}" "--output-dir=${output_dir}"
      DEPENDS
        examples/triton/requirements.txt
        "${script}"
      VERBATIM
    )
    add_custom_target(${name}_bundle DEPENDS ${outputs})
    set(${output_hsaco_var} "${hsaco}" PARENT_SCOPE)
    set(${output_manifest_var} "${manifest}" PARENT_SCOPE)
  endif()
endfunction()

add_library(lrrt src/runtime.cpp)
add_library(lrrt::lrrt ALIAS lrrt)

add_library(lrrt_bundle src/bundle.cpp)
add_library(lrrt::bundle ALIAS lrrt_bundle)
target_link_libraries(lrrt_bundle PUBLIC lrrt::lrrt)

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

if(LRRT_BUILD_BENCHMARKS)
  add_executable(
    lrrt_launch_overhead_benchmark
    benchmarks/launch_overhead.cpp
  )
  target_link_libraries(lrrt_launch_overhead_benchmark PRIVATE lrrt::lrrt)
  add_executable(
    lrrt_async_copy_launch_benchmark
    benchmarks/async_copy_launch.cpp
  )
  target_link_libraries(lrrt_async_copy_launch_benchmark PRIVATE lrrt::lrrt)
endif()

if(LRRT_ENABLE_TRITON_EXAMPLES AND UV_EXECUTABLE)
  add_executable(lrrt_triton_vector_add examples/triton/triton_vector_add.cpp)
  target_link_libraries(lrrt_triton_vector_add PRIVATE lrrt::bundle)
  add_executable(lrrt_triton_saxpy examples/triton/triton_saxpy.cpp)
  target_link_libraries(lrrt_triton_saxpy PRIVATE lrrt::bundle)
  add_executable(lrrt_triton_scale examples/triton/triton_scale.cpp)
  target_link_libraries(lrrt_triton_scale PRIVATE lrrt::bundle)
  add_executable(lrrt_triton_rmsnorm examples/triton/triton_rmsnorm.cpp)
  target_link_libraries(lrrt_triton_rmsnorm PRIVATE lrrt::bundle)
  add_executable(
    lrrt_triton_rmsnorm_precision_tests
    tests/triton_rmsnorm_precision.cpp
  )
  target_link_libraries(lrrt_triton_rmsnorm_precision_tests PRIVATE lrrt::bundle)
  add_executable(lrrt_triton_rope examples/triton/triton_rope.cpp)
  target_link_libraries(lrrt_triton_rope PRIVATE lrrt::bundle)
  add_executable(lrrt_triton_silu_mul examples/triton/triton_silu_mul.cpp)
  target_link_libraries(lrrt_triton_silu_mul PRIVATE lrrt::bundle)
elseif(LRRT_ENABLE_TRITON_EXAMPLES)
  message(STATUS "uv not found; skipping Triton example")
else()
  message(STATUS "Triton examples disabled")
endif()

add_executable(lrrt_memory_bounds_tests tests/memory_bounds.cpp)
target_link_libraries(lrrt_memory_bounds_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_async_copy_tests tests/async_copy.cpp)
target_link_libraries(lrrt_async_copy_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_async_copy_launch_tests tests/async_copy_launch.cpp)
target_link_libraries(lrrt_async_copy_launch_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_launch_async_copy_tests tests/launch_async_copy.cpp)
target_link_libraries(lrrt_launch_async_copy_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_explicit_dependencies_tests tests/explicit_dependencies.cpp)
target_link_libraries(lrrt_explicit_dependencies_tests PRIVATE lrrt::lrrt)

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

add_executable(lrrt_bundle_tests tests/bundle.cpp)
target_link_libraries(lrrt_bundle_tests PRIVATE lrrt::bundle)

lrrt_add_hsaco_kernel(
  vector_add_kernel examples/vector_add/vector_add_kernel.cpp
  VECTOR_ADD_HSACO
)
lrrt_add_hsaco_kernel(saxpy_kernel examples/saxpy/saxpy_kernel.cpp SAXPY_HSACO)
lrrt_add_hsaco_kernel(fill_kernel examples/fill/fill_kernel.cpp FILL_HSACO)
lrrt_add_hsaco_kernel(scale_kernel tests/scale_kernel.cpp SCALE_HSACO)
lrrt_add_hsaco_kernel(copy_kernel tests/copy_kernel_kernel.cpp COPY_HSACO)
lrrt_add_hsaco_kernel(
  async_copy_launch_kernel tests/async_copy_launch_kernel.cpp
  ASYNC_COPY_LAUNCH_HSACO
)
if(LRRT_BUILD_BENCHMARKS)
  lrrt_add_hsaco_kernel(
    launch_benchmark_kernel benchmarks/launch_kernel.cpp
    LAUNCH_BENCHMARK_HSACO
  )
  lrrt_add_hsaco_kernel(
    async_copy_launch_benchmark_kernel
    benchmarks/async_copy_launch_kernel.cpp
    ASYNC_COPY_LAUNCH_BENCHMARK_HSACO
  )
endif()
if(LRRT_ENABLE_TRITON_EXAMPLES)
  lrrt_add_triton_bundle(
    triton_vector_add examples/triton/vector_add_kernel.py
    TRITON_VECTOR_ADD_HSACO TRITON_VECTOR_ADD_MANIFEST
  )
  lrrt_add_triton_bundle(
    triton_saxpy examples/triton/saxpy_kernel.py
    TRITON_SAXPY_HSACO TRITON_SAXPY_MANIFEST
  )
  lrrt_add_triton_bundle(
    triton_scale examples/triton/scale_kernel.py
    TRITON_SCALE_HSACO TRITON_SCALE_MANIFEST
  )
  lrrt_add_triton_bundle(
    triton_rmsnorm examples/triton/rmsnorm_kernel.py
    TRITON_RMSNORM_HSACO TRITON_RMSNORM_MANIFEST
    kernels_fp32_2048.hsaco kernels_fp32_4096.hsaco
    kernels_fp16_1024.hsaco kernels_fp16_2048.hsaco kernels_fp16_4096.hsaco
    kernels_bf16_1024.hsaco kernels_bf16_2048.hsaco kernels_bf16_4096.hsaco
  )
  lrrt_add_triton_bundle(
    triton_rope examples/triton/rope_kernel.py
    TRITON_ROPE_HSACO TRITON_ROPE_MANIFEST
    kernels_128.hsaco
  )
  lrrt_add_triton_bundle(
    triton_silu_mul examples/triton/silu_mul_kernel.py
    TRITON_SILU_MUL_HSACO TRITON_SILU_MUL_MANIFEST
  )
endif()

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
if(ASYNC_COPY_LAUNCH_HSACO)
  add_dependencies(lrrt_async_copy_launch_tests async_copy_launch_kernel_hsaco)
  add_dependencies(lrrt_launch_async_copy_tests async_copy_launch_kernel_hsaco)
  add_dependencies(lrrt_explicit_dependencies_tests
                   async_copy_launch_kernel_hsaco)
  target_compile_definitions(
    lrrt_async_copy_launch_tests
    PRIVATE LRRT_ASYNC_COPY_LAUNCH_HSACO="${ASYNC_COPY_LAUNCH_HSACO}"
  )
  target_compile_definitions(
    lrrt_launch_async_copy_tests
    PRIVATE LRRT_ASYNC_COPY_LAUNCH_HSACO="${ASYNC_COPY_LAUNCH_HSACO}"
  )
  target_compile_definitions(
    lrrt_explicit_dependencies_tests
    PRIVATE LRRT_ASYNC_COPY_LAUNCH_HSACO="${ASYNC_COPY_LAUNCH_HSACO}"
  )
endif()
if(COPY_HSACO)
  add_dependencies(lrrt_copy_kernel_tests copy_kernel_hsaco)
  target_compile_definitions(
    lrrt_copy_kernel_tests
    PRIVATE LRRT_COPY_HSACO="${COPY_HSACO}"
  )
endif()
if(LAUNCH_BENCHMARK_HSACO)
  add_dependencies(
    lrrt_launch_overhead_benchmark
    launch_benchmark_kernel_hsaco
  )
  target_compile_definitions(
    lrrt_launch_overhead_benchmark
    PRIVATE LRRT_LAUNCH_BENCHMARK_HSACO="${LAUNCH_BENCHMARK_HSACO}"
  )
endif()
if(ASYNC_COPY_LAUNCH_BENCHMARK_HSACO)
  add_dependencies(
    lrrt_async_copy_launch_benchmark
    async_copy_launch_benchmark_kernel_hsaco
  )
  target_compile_definitions(
    lrrt_async_copy_launch_benchmark
    PRIVATE
      LRRT_ASYNC_COPY_LAUNCH_BENCHMARK_HSACO="${ASYNC_COPY_LAUNCH_BENCHMARK_HSACO}"
  )
endif()
if(TRITON_VECTOR_ADD_HSACO)
  add_dependencies(lrrt_triton_vector_add triton_vector_add_bundle)
  target_compile_definitions(
    lrrt_triton_vector_add
    PRIVATE
      LRRT_TRITON_VECTOR_ADD_MANIFEST="${TRITON_VECTOR_ADD_MANIFEST}"
  )
endif()
if(TRITON_SAXPY_HSACO)
  add_dependencies(lrrt_triton_saxpy triton_saxpy_bundle)
  target_compile_definitions(
    lrrt_triton_saxpy
    PRIVATE LRRT_TRITON_SAXPY_MANIFEST="${TRITON_SAXPY_MANIFEST}"
  )
endif()
if(TRITON_SCALE_HSACO)
  add_dependencies(lrrt_triton_scale triton_scale_bundle)
  target_compile_definitions(
    lrrt_triton_scale
    PRIVATE LRRT_TRITON_SCALE_MANIFEST="${TRITON_SCALE_MANIFEST}"
  )
endif()
if(TRITON_RMSNORM_HSACO)
  add_dependencies(lrrt_triton_rmsnorm triton_rmsnorm_bundle)
  add_dependencies(lrrt_triton_rmsnorm_precision_tests triton_rmsnorm_bundle)
  target_compile_definitions(
    lrrt_triton_rmsnorm
    PRIVATE LRRT_TRITON_RMSNORM_MANIFEST="${TRITON_RMSNORM_MANIFEST}"
  )
  target_compile_definitions(
    lrrt_triton_rmsnorm_precision_tests
    PRIVATE LRRT_TRITON_RMSNORM_MANIFEST="${TRITON_RMSNORM_MANIFEST}"
  )
endif()
if(TRITON_ROPE_HSACO)
  add_dependencies(lrrt_triton_rope triton_rope_bundle)
  target_compile_definitions(
    lrrt_triton_rope
    PRIVATE LRRT_TRITON_ROPE_MANIFEST="${TRITON_ROPE_MANIFEST}"
  )
endif()
if(TRITON_SILU_MUL_HSACO)
  add_dependencies(lrrt_triton_silu_mul triton_silu_mul_bundle)
  target_compile_definitions(
    lrrt_triton_silu_mul
    PRIVATE LRRT_TRITON_SILU_MUL_MANIFEST="${TRITON_SILU_MUL_MANIFEST}"
  )
endif()

enable_testing()
add_test(NAME lrrt_memory_bounds_tests COMMAND lrrt_memory_bounds_tests)
add_test(NAME lrrt_async_copy_tests COMMAND lrrt_async_copy_tests)
add_test(NAME lrrt_async_copy_launch_tests COMMAND lrrt_async_copy_launch_tests)
add_test(NAME lrrt_launch_async_copy_tests COMMAND lrrt_launch_async_copy_tests)
add_test(NAME lrrt_explicit_dependencies_tests
         COMMAND lrrt_explicit_dependencies_tests)
add_test(NAME lrrt_handle_registry_tests COMMAND lrrt_handle_registry_tests)
add_test(NAME lrrt_scale_tests COMMAND lrrt_scale_tests)
add_test(NAME lrrt_copy_kernel_tests COMMAND lrrt_copy_kernel_tests)
add_test(NAME lrrt_event_timing_tests COMMAND lrrt_event_timing_tests)
add_test(NAME lrrt_cpp_wrapper_tests COMMAND lrrt_cpp_wrapper_tests)
add_test(NAME lrrt_bundle_tests COMMAND lrrt_bundle_tests)
if(TARGET lrrt_triton_vector_add)
  add_test(NAME lrrt_triton_vector_add COMMAND lrrt_triton_vector_add)
endif()
if(TARGET lrrt_triton_saxpy)
  add_test(NAME lrrt_triton_saxpy COMMAND lrrt_triton_saxpy)
endif()
if(TARGET lrrt_triton_scale)
  add_test(NAME lrrt_triton_scale COMMAND lrrt_triton_scale)
endif()
if(TARGET lrrt_triton_rmsnorm)
  add_test(NAME lrrt_triton_rmsnorm COMMAND lrrt_triton_rmsnorm)
endif()
if(TARGET lrrt_triton_rmsnorm_precision_tests)
  add_test(
    NAME lrrt_triton_rmsnorm_precision_tests
    COMMAND lrrt_triton_rmsnorm_precision_tests
  )
endif()
if(TARGET lrrt_triton_rope)
  add_test(NAME lrrt_triton_rope COMMAND lrrt_triton_rope)
endif()
if(TARGET lrrt_triton_silu_mul)
  add_test(NAME lrrt_triton_silu_mul COMMAND lrrt_triton_silu_mul)
endif()
if(LRRT_ENABLE_TRITON_EXAMPLES AND UV_EXECUTABLE)
  if(PYTHON_EXECUTABLE AND LLVM_READELF)
    add_test(
      NAME lrrt_triton_vector_add_bundle_consistency
      COMMAND
        "${PYTHON_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/triton_bundle_consistency.py"
        "--manifest=${TRITON_VECTOR_ADD_MANIFEST}"
        "--llvm-readelf=${LLVM_READELF}"
    )
    add_test(
      NAME lrrt_triton_saxpy_bundle_consistency
      COMMAND
        "${PYTHON_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/triton_bundle_consistency.py"
        "--manifest=${TRITON_SAXPY_MANIFEST}"
        "--llvm-readelf=${LLVM_READELF}"
    )
    add_test(
      NAME lrrt_triton_scale_bundle_consistency
      COMMAND
        "${PYTHON_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/triton_bundle_consistency.py"
        "--manifest=${TRITON_SCALE_MANIFEST}"
        "--llvm-readelf=${LLVM_READELF}"
    )
    add_test(
      NAME lrrt_triton_rmsnorm_bundle_consistency
      COMMAND
        "${PYTHON_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/triton_bundle_consistency.py"
        "--manifest=${TRITON_RMSNORM_MANIFEST}"
        "--llvm-readelf=${LLVM_READELF}"
    )
    add_test(
      NAME lrrt_triton_rope_bundle_consistency
      COMMAND
        "${PYTHON_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/triton_bundle_consistency.py"
        "--manifest=${TRITON_ROPE_MANIFEST}"
        "--llvm-readelf=${LLVM_READELF}"
    )
    add_test(
      NAME lrrt_triton_silu_mul_bundle_consistency
      COMMAND
        "${PYTHON_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/triton_bundle_consistency.py"
        "--manifest=${TRITON_SILU_MUL_MANIFEST}"
        "--llvm-readelf=${LLVM_READELF}"
    )
  else()
    message(STATUS "python3 or llvm-readelf not found; skipping Triton bundle consistency test")
  endif()
endif()
