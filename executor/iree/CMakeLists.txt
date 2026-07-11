add_library(lrrt_iree_adapter INTERFACE)
add_library(lrrt::iree_adapter ALIAS lrrt_iree_adapter)

target_include_directories(
  lrrt_iree_adapter
  INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${LRRT_IREE_INCLUDE_DIR}"
)

target_link_libraries(lrrt_iree_adapter INTERFACE lrrt::lrrt)

add_executable(lrrt_iree_adapter_smoke smoke.cpp)
target_link_libraries(lrrt_iree_adapter_smoke PRIVATE lrrt::iree_adapter)
add_test(NAME lrrt_iree_adapter_smoke COMMAND lrrt_iree_adapter_smoke)

add_executable(lrrt_iree_adapter_contract_tests adapter_contract_test.cpp)
target_link_libraries(
  lrrt_iree_adapter_contract_tests
  PRIVATE lrrt::iree_adapter
)
add_test(
  NAME lrrt_iree_adapter_contract_tests
  COMMAND lrrt_iree_adapter_contract_tests
)

set(LRRT_IREE_RUNTIME_BUILD_DIR
    "${PROJECT_SOURCE_DIR}/build-iree-tools/runtime/src"
)
set(LRRT_IREE_BASE_LIB
    "${LRRT_IREE_RUNTIME_BUILD_DIR}/iree/base/libiree_base_base.a"
)
set(LRRT_IREE_HAL_LIB
    "${LRRT_IREE_RUNTIME_BUILD_DIR}/iree/hal/libiree_hal_hal.a"
)
set(LRRT_IREE_ASYNC_LIB
    "${LRRT_IREE_RUNTIME_BUILD_DIR}/iree/async/libiree_async_async.a"
)
set(LRRT_IREE_ASYNC_PLATFORM_LIB
    "${LRRT_IREE_RUNTIME_BUILD_DIR}/iree/async/libiree_async_platform.a"
)
set(LRRT_IREE_BASE_PATH_LIB
    "${LRRT_IREE_RUNTIME_BUILD_DIR}/iree/base/internal/libiree_base_internal_path.a"
)
set(LRRT_IREE_BASE_THREADING_LIB
    "${LRRT_IREE_RUNTIME_BUILD_DIR}/iree/base/threading/libiree_base_threading_threading.a"
)
set(LRRT_IREE_BASE_TIME_LIB
    "${LRRT_IREE_RUNTIME_BUILD_DIR}/iree/base/internal/libiree_base_internal_time.a"
)
set(LRRT_IREE_TOOLING_RUN_MODULE_LIB
    "${LRRT_IREE_RUNTIME_BUILD_DIR}/iree/tooling/libiree_tooling_run_module.a"
)
set(LRRT_IREE_EXECUTABLE_HEADER_LIB
    "${LRRT_IREE_RUNTIME_BUILD_DIR}/iree/hal/utils/libiree_hal_utils_executable_header.a"
)
set(LRRT_IREE_FILES_LIB
    "${LRRT_IREE_RUNTIME_BUILD_DIR}/iree/hal/utils/libiree_hal_utils_files.a"
)
set(LRRT_IREE_IO_FILE_HANDLE_LIB
    "${LRRT_IREE_RUNTIME_BUILD_DIR}/iree/io/libiree_io_file_handle.a"
)
set(LRRT_IREE_FLATCC_PARSING_LIB
    "${PROJECT_SOURCE_DIR}/build-iree-tools/build_tools/third_party/flatcc/libflatcc_parsing.a"
)

function(lrrt_configure_iree_hal_driver_target target_name)
  target_include_directories(
    ${target_name}
    PUBLIC
      "${PROJECT_SOURCE_DIR}"
      "${LRRT_IREE_INCLUDE_DIR}"
      "${PROJECT_SOURCE_DIR}/build-iree-tools/runtime/src"
      "${PROJECT_SOURCE_DIR}/third_party/iree/third_party/flatcc/include"
  )
  target_compile_definitions(
    ${target_name}
    PUBLIC IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl
  )
  target_link_libraries(
    ${target_name}
    PUBLIC
      "-Wl,--start-group"
      "${LRRT_IREE_HAL_LIB}"
      "${LRRT_IREE_ASYNC_PLATFORM_LIB}"
      "${LRRT_IREE_BASE_PATH_LIB}"
      "${LRRT_IREE_BASE_THREADING_LIB}"
      "${LRRT_IREE_BASE_LIB}"
      "${LRRT_IREE_BASE_TIME_LIB}"
      "${LRRT_IREE_EXECUTABLE_HEADER_LIB}"
      "${LRRT_IREE_FILES_LIB}"
      "${LRRT_IREE_ASYNC_LIB}"
      "${LRRT_IREE_IO_FILE_HANDLE_LIB}"
      "${LRRT_IREE_FLATCC_PARSING_LIB}"
      "-Wl,--end-group"
      lrrt::lrrt
  )
endfunction()

if(EXISTS "${LRRT_IREE_BASE_LIB}"
   AND EXISTS "${LRRT_IREE_HAL_LIB}"
   AND EXISTS "${LRRT_IREE_ASYNC_LIB}"
   AND EXISTS "${LRRT_IREE_ASYNC_PLATFORM_LIB}"
   AND EXISTS "${LRRT_IREE_BASE_PATH_LIB}"
   AND EXISTS "${LRRT_IREE_BASE_THREADING_LIB}"
   AND EXISTS "${LRRT_IREE_BASE_TIME_LIB}"
   AND EXISTS "${LRRT_IREE_EXECUTABLE_HEADER_LIB}"
   AND EXISTS "${LRRT_IREE_FILES_LIB}"
   AND EXISTS "${LRRT_IREE_IO_FILE_HANDLE_LIB}"
   AND EXISTS "${LRRT_IREE_FLATCC_PARSING_LIB}"
)
  add_library(
    lrrt_iree_hal_driver
    STATIC
      hal_driver.c
      registration/init.c
  )
  add_library(lrrt::iree_hal_driver ALIAS lrrt_iree_hal_driver)
  lrrt_configure_iree_hal_driver_target(lrrt_iree_hal_driver)

  add_library(
    lrrt_iree_hal_driver_testing
    STATIC
      hal_driver.c
      registration/init.c
  )
  lrrt_configure_iree_hal_driver_target(lrrt_iree_hal_driver_testing)
  target_compile_definitions(
    lrrt_iree_hal_driver_testing
    PRIVATE
      LRRT_IREE_HAL_DRIVER_TESTING
  )

  function(lrrt_add_iree_hal_driver_test target source)
    add_executable(${target} ${source})
    target_compile_definitions(
      ${target}
      PRIVATE
        LRRT_IREE_HAL_DRIVER_TESTING
    )
    target_link_libraries(${target} PRIVATE lrrt_iree_hal_driver_testing)
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(
      ${target}
      PROPERTIES
        LABELS "iree;gpu"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    )
  endfunction()

  lrrt_add_iree_hal_driver_test(
    lrrt_iree_hal_driver_registration_tests
    hal_driver_registration_test.cpp
  )
  lrrt_add_iree_hal_driver_test(
    lrrt_iree_hal_driver_allocator_tests
    hal_driver_allocator_test.cpp
  )
  lrrt_add_iree_hal_driver_test(
    lrrt_iree_hal_driver_transfer_tests
    hal_driver_transfer_test.cpp
  )
  lrrt_add_iree_hal_driver_test(
    lrrt_iree_hal_driver_unsupported_tests
    hal_driver_unsupported_test.cpp
  )
  lrrt_add_iree_hal_driver_test(
    lrrt_iree_hal_driver_executable_tests
    hal_driver_executable_test.cpp
  )
else()
  message(
    STATUS
      "IREE HAL driver registration test disabled; IREE runtime static libs "
      "were not found under ${LRRT_IREE_RUNTIME_BUILD_DIR}"
  )
endif()

if(TARGET lrrt_iree_hal_driver AND EXISTS "${LRRT_IREE_TOOLING_RUN_MODULE_LIB}")
  file(
    GLOB_RECURSE
    LRRT_IREE_RUNTIME_STATIC_LIBS
    CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/build-iree-tools/runtime/src/iree/*.a"
    "${PROJECT_SOURCE_DIR}/build-iree-tools/build_tools/third_party/*.a"
  )
  function(lrrt_add_iree_run_module_launcher target)
    add_executable(${target} run_module.c)
    target_compile_definitions(
      ${target}
      PRIVATE
        LRRT_IREE_RUN_MODULE_TOOL_NAME="${target}"
    )
    target_link_libraries(
      ${target}
      PRIVATE
        lrrt::iree_hal_driver
        "-Wl,--start-group"
        ${LRRT_IREE_RUNTIME_STATIC_LIBS}
        "-Wl,--end-group"
        dl
        m
        pthread
    )
  endfunction()

  lrrt_add_iree_run_module_launcher(lrrt_iree_run_module)
  lrrt_add_iree_run_module_launcher(lrrt_iree_run_module_smoke)

  if(LRRT_IREE_COMPILE_EXECUTABLE)
    lrrt_add_iree_vmfb_smoke(
      lrrt_iree_vmfb_probe
      lrrt_iree_run_module_vmfb_smoke
      lrrt_iree_vmfb
      tools/iree_minimal_mul.mlir
      minimal_mul
      simple_mul
      "4xf32=10 40 90 160"
      "--input=4xf32=1 2 3 4"
      "--input=4xf32=10 20 30 40"
    )

    lrrt_add_iree_vmfb_smoke(
      lrrt_iree_two_dispatch_vmfb_probe
      lrrt_iree_two_dispatch_run_module_vmfb_smoke
      lrrt_iree_two_dispatch_vmfb
      tools/iree_two_dispatch.mlir
      two_dispatch
      two_matmuls
      "2x2xf32=\\[2 6\\]\\[6 12\\]"
      "--input=2x2xf32=1 2 3 4"
      "--input=2x2xf32=1 0 0 1"
      "--input=2x2xf32=2 0 0 3"
    )

    lrrt_add_iree_vmfb_smoke(
      lrrt_iree_mixed_matmuls_vmfb_probe
      lrrt_iree_mixed_matmuls_run_module_vmfb_smoke
      lrrt_iree_mixed_matmuls_vmfb
      tools/iree_mixed_matmuls.mlir
      mixed_matmuls
      mixed_matmuls
      "2x3xf32=\\[9 12 15\\]\\[19 26 33\\]"
      "--input=2x2xf32=1 2 3 4"
      "--input=2x2xf32=1 0 0 1"
      "--input=2x3xf32=1 2 3 4 5 6"
    )

    lrrt_add_iree_vmfb_smoke(
      lrrt_iree_matmul_add_relu_vmfb_probe
      lrrt_iree_matmul_add_relu_run_module_vmfb_smoke
      lrrt_iree_matmul_add_relu_vmfb
      tools/iree_matmul_add_relu.mlir
      matmul_add_relu
      matmul_add_relu
      "2x2xf32=\\[9 0\\]\\[0 17\\]"
      "--input=2x2xf32=1 2 3 4"
      "--input=2x2xf32=1 0 0 1"
      "--input=2x2xf32=8 -3 -7 13"
    )

    lrrt_add_iree_vmfb_smoke(
      lrrt_iree_mini_decoder_block_vmfb_probe
      lrrt_iree_mini_decoder_block_run_module_vmfb_smoke
      lrrt_iree_mini_decoder_block_vmfb
      tools/iree_mini_decoder_block.mlir
      mini_decoder_block
      mini_decoder_block
      "2x2xf32=\\[7 10\\]\\[17 26\\]"
      "--input=2x2xf32=1 2 3 4"
      "--input=2x2xf32=1 0.5 2 1"
      "--input=2x2xf32=0 1 -1 0"
      "--input=2x2xf32=1 0 0 1"
      "--input=2x2xf32=1 2 3 4"
    )

    lrrt_add_iree_vmfb_smoke(
      lrrt_iree_rmsnorm_like_vmfb_probe
      lrrt_iree_rmsnorm_like_run_module_vmfb_smoke
      lrrt_iree_rmsnorm_like_vmfb
      tools/iree_rmsnorm_like.mlir
      rmsnorm_like
      rmsnorm_like
      "2x2xf32=\\[1 2\\]\\[1 2\\]"
      "--input=2x2xf32=1 1 2 2"
      "--input=2xf32=1 2"
    )

    lrrt_add_iree_vmfb_smoke(
      lrrt_iree_attention_score_vmfb_probe
      lrrt_iree_attention_score_run_module_vmfb_smoke
      lrrt_iree_attention_score_vmfb
      tools/iree_attention_score.mlir
      attention_score
      attention_score
      "2x2xf32=\\[5 11\\]\\[11 25\\]"
      "--input=2x2xf32=1 2 3 4"
      "--input=2x2xf32=1 3 2 4"
    )

    lrrt_add_iree_vmfb_smoke(
      lrrt_iree_attention_softmax_vmfb_probe
      lrrt_iree_attention_softmax_run_module_vmfb_smoke
      lrrt_iree_attention_softmax_vmfb
      tools/iree_attention_softmax.mlir
      attention_softmax
      attention_softmax
      "2x2xf32=\\[4 6\\]\\[4 6\\]"
      "--input=2x2xf32=0 0 0 0"
      "--input=2x2xf32=1 0 0 1"
      "--input=2x2xf32=2 4 6 8"
    )

    lrrt_add_iree_vmfb_smoke(
      lrrt_iree_kv_cache_read_vmfb_probe
      lrrt_iree_kv_cache_read_run_module_vmfb_smoke
      lrrt_iree_kv_cache_read_vmfb
      tools/iree_kv_cache_read.mlir
      kv_cache_read_attention
      kv_cache_read_attention
      "2x2xf32=\\[4 6\\]\\[4 6\\]"
      "--input=2x2xf32=0 0 0 0"
      "--input=2x3xf32=1 0 0 1 1 1"
      "--input=3x2xf32=2 4 4 6 6 8"
    )

    lrrt_add_iree_vmfb_smoke(
      lrrt_iree_kv_cache_update_vmfb_probe
      lrrt_iree_kv_cache_update_run_module_vmfb_smoke
      lrrt_iree_kv_cache_update_vmfb
      tools/iree_kv_cache_update.mlir
      kv_cache_update
      kv_cache_update
      "3x2xf32=\\[1 2\\]\\[3 4\\]\\[5 6\\]"
      "--input=3x2xf32=1 2 3 4 0 0"
      "--input=2xf32=5 6"
    )

    lrrt_add_iree_vmfb_smoke(
      lrrt_iree_attention_residual_vmfb_probe
      lrrt_iree_attention_residual_run_module_vmfb_smoke
      lrrt_iree_attention_residual_vmfb
      tools/iree_attention_residual.mlir
      attention_residual
      attention_residual
      "2x2xf32=\\[5 8\\]\\[7 10\\]"
      "--input=2x2xf32=1 2 3 4"
      "--input=2x2xf32=0 0 0 0"
      "--input=2x2xf32=1 0 0 1"
      "--input=2x2xf32=2 4 6 8"
    )

    lrrt_add_iree_vmfb_smoke(
      lrrt_iree_rope_apply_vmfb_probe
      lrrt_iree_rope_apply_run_module_vmfb_smoke
      lrrt_iree_rope_apply_vmfb
      tools/iree_rope_apply.mlir
      rope_apply
      rope_apply
      "2x2xf32=\\[1 2\\]\\[-4 3\\]"
      "--input=2x2xf32=1 2 3 4"
      "--input=2xf32=1 0"
      "--input=2xf32=0 1"
    )

    lrrt_add_iree_vmfb_smoke(
      lrrt_iree_qwen_ffn_mini_vmfb_probe
      lrrt_iree_qwen_ffn_mini_run_module_vmfb_smoke
      lrrt_iree_qwen_ffn_mini_vmfb
      tools/iree_qwen_ffn_mini.mlir
      qwen_ffn_mini
      qwen_ffn_mini
      "2x2xf32=\\[1 4\\]\\[9 16\\]"
      "--input=2x2xf32=1 2 3 4"
      "--input=2x2xf32=1 0 0 1"
      "--input=2x2xf32=1 0 0 1"
      "--input=2x2xf32=1 0 0 1"
    )

    lrrt_add_iree_vmfb_smoke(
      lrrt_iree_mini_decoder_layer_vmfb_probe
      lrrt_iree_mini_decoder_layer_run_module_vmfb_smoke
      lrrt_iree_mini_decoder_layer_vmfb
      tools/iree_mini_decoder_layer.mlir
      mini_decoder_layer
      mini_decoder_layer
      "2x2xf32=\\[30 72\\]\\[56 110\\]"
      "--input=2x2xf32=1 2 3 4"
      "--input=2x2xf32=0 0 0 0"
      "--input=2x2xf32=1 0 0 1"
      "--input=2x2xf32=2 4 6 8"
      "--input=2x2xf32=1 0 0 1"
      "--input=2x2xf32=1 0 0 1"
      "--input=2x2xf32=1 0 0 1"
    )

    lrrt_add_iree_vmfb_smoke(
      lrrt_iree_mini_decoder_layer_kv_cache_vmfb_probe
      lrrt_iree_mini_decoder_layer_kv_cache_run_module_vmfb_smoke
      lrrt_iree_mini_decoder_layer_kv_cache_vmfb
      tools/iree_mini_decoder_layer_kv_cache.mlir
      mini_decoder_layer_kv_cache
      mini_decoder_layer_kv_cache
      "2x2xf32=\\[30 72\\]\\[56 110\\]"
      "--input=2x2xf32=1 2 3 4"
      "--input=2x2xf32=0 0 0 0"
      "--input=2x3xf32=1 0 0 1 1 1"
      "--input=3x2xf32=2 4 4 6 6 8"
      "--input=2x2xf32=1 0 0 1"
      "--input=2x2xf32=1 0 0 1"
      "--input=2x2xf32=1 0 0 1"
    )

    lrrt_add_iree_vmfb_smoke(
      lrrt_iree_mini_decoder_layer_rope_kv_cache_vmfb_probe
      lrrt_iree_mini_decoder_layer_rope_kv_cache_run_module_vmfb_smoke
      lrrt_iree_mini_decoder_layer_rope_kv_cache_vmfb
      tools/iree_mini_decoder_layer_rope_kv_cache.mlir
      mini_decoder_layer_rope_kv_cache
      mini_decoder_layer_rope_kv_cache
      "2x2xf32=\\[30 72\\]\\[56 110\\]"
      "--input=2x2xf32=1 2 3 4"
      "--input=2x2xf32=1 2 3 4"
      "--input=2x3xf32=0 0 0 0 0 0"
      "--input=3x2xf32=2 4 4 6 6 8"
      "--input=2xf32=1 0"
      "--input=2xf32=0 1"
      "--input=2x2xf32=1 0 0 1"
      "--input=2x2xf32=1 0 0 1"
      "--input=2x2xf32=1 0 0 1"
    )

    lrrt_add_iree_vmfb_smoke(
      lrrt_iree_token_step_kv_cache_vmfb_probe
      lrrt_iree_token_step_kv_cache_run_module_vmfb_smoke
      lrrt_iree_token_step_kv_cache_vmfb
      tools/iree_token_step_kv_cache.mlir
      token_step_kv_cache
      token_step_kv_cache
      "2x2xf32=\\[4 6\\]\\[4 6\\]"
      "--input=2x2xf32=1 2 3 4"
      "--input=2x3xf32=0 0 0 0 0 0"
      "--input=2xf32=0 0"
      "--input=3x2xf32=2 4 4 6 0 0"
      "--input=2xf32=6 8"
      "--input=2xf32=1 0"
      "--input=2xf32=0 1"
    )

    lrrt_add_iree_vmfb_probe(
      lrrt_iree_multi_export_vmfb_probe
      lrrt_iree_multi_export_vmfb
      tools/iree_multi_export.mlir
      multi_export
    )
    lrrt_add_iree_vmfb_run(
      lrrt_iree_multi_export_mul_run_module_vmfb_smoke
      lrrt_iree_multi_export_vmfb
      multi_export
      simple_mul
      "4xf32=10 40 90 160"
      "--input=4xf32=1 2 3 4"
      "--input=4xf32=10 20 30 40"
    )
    lrrt_add_iree_vmfb_run(
      lrrt_iree_multi_export_add_run_module_vmfb_smoke
      lrrt_iree_multi_export_vmfb
      multi_export
      simple_add
      "4xf32=11 22 33 44"
      "--input=4xf32=1 2 3 4"
      "--input=4xf32=10 20 30 40"
    )
  else()
    message(
      STATUS
        "IREE lrrt VMFB smoke test disabled; iree-compile was not found"
    )
  endif()
elseif(NOT TARGET lrrt_iree_hal_driver)
  message(
    STATUS
      "IREE lrrt run-module smoke disabled; lrrt IREE HAL driver target "
      "was not available"
  )
else()
  message(
    STATUS
      "IREE lrrt run-module smoke disabled; IREE run_module static library "
      "was not found under ${LRRT_IREE_RUNTIME_BUILD_DIR}"
  )
endif()

if(LRRT_IREE_COMPILE_EXECUTABLE AND LRRT_IREE_RUN_MODULE_EXECUTABLE)
  set(LRRT_IREE_PROBE_HSACO
      "${PROJECT_SOURCE_DIR}/build-iree-probe/minimal_mul_${LRRT_AMDGPU_TARGET}.hsaco"
  )
  set(LRRT_IREE_PROBE_METADATA
      "${PROJECT_SOURCE_DIR}/build-iree-probe/minimal_mul_${LRRT_AMDGPU_TARGET}_metadata.json"
  )
  if(TARGET lrrt_iree_hal_driver_executable_tests)
    target_compile_definitions(
      lrrt_iree_hal_driver_executable_tests
      PRIVATE
        LRRT_IREE_PROBE_HSACO_PATH="${LRRT_IREE_PROBE_HSACO}"
        LRRT_IREE_PROBE_SYMBOL="simple_mul_dispatch_0_elementwise_4_f32"
    )
    set_tests_properties(
      lrrt_iree_hal_driver_executable_tests
      PROPERTIES
        FIXTURES_REQUIRED lrrt_iree_hsaco
    )
  endif()

  add_executable(lrrt_iree_hsaco_dispatch_smoke hsaco_dispatch_smoke.cpp)
  target_link_libraries(lrrt_iree_hsaco_dispatch_smoke PRIVATE lrrt::iree_adapter)
  target_compile_definitions(
    lrrt_iree_hsaco_dispatch_smoke
    PRIVATE
      LRRT_IREE_HSACO_PATH="${LRRT_IREE_PROBE_HSACO}"
      LRRT_IREE_METADATA_PATH="${LRRT_IREE_PROBE_METADATA}"
  )

  add_test(
    NAME lrrt_iree_emit_hsaco_probe
    COMMAND
      "${PROJECT_SOURCE_DIR}/tools/iree_compile_probe.sh"
      --iree-compile "${LRRT_IREE_COMPILE_EXECUTABLE}"
      --target "${LRRT_AMDGPU_TARGET}"
      --emit-hsaco
  )
  set_tests_properties(
    lrrt_iree_emit_hsaco_probe
    PROPERTIES
      FIXTURES_SETUP lrrt_iree_hsaco
      LABELS "iree"
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  )

  add_test(NAME lrrt_iree_hsaco_dispatch_smoke
           COMMAND lrrt_iree_hsaco_dispatch_smoke)
  set_tests_properties(
    lrrt_iree_hsaco_dispatch_smoke
    PROPERTIES
      FIXTURES_REQUIRED lrrt_iree_hsaco
      LABELS "iree;gpu"
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  )

  add_test(
    NAME lrrt_iree_baseline_probe
    COMMAND
      "${PROJECT_SOURCE_DIR}/tools/iree_compile_probe.sh"
      --iree-compile "${LRRT_IREE_COMPILE_EXECUTABLE}"
      --iree-run-module "${LRRT_IREE_RUN_MODULE_EXECUTABLE}"
      --run-baseline
  )
  set_tests_properties(
    lrrt_iree_baseline_probe
    PROPERTIES
      LABELS "iree;gpu"
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  )
else()
  message(
    STATUS
      "IREE baseline probe test disabled; iree-compile or iree-run-module "
      "was not found"
  )
endif()
