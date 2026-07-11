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
set(LRRT_IREE_FLATCC_PARSING_LIB
    "${PROJECT_SOURCE_DIR}/build-iree-tools/build_tools/third_party/flatcc/libflatcc_parsing.a"
)
if(EXISTS "${LRRT_IREE_BASE_LIB}"
   AND EXISTS "${LRRT_IREE_HAL_LIB}"
   AND EXISTS "${LRRT_IREE_ASYNC_LIB}"
   AND EXISTS "${LRRT_IREE_BASE_PATH_LIB}"
   AND EXISTS "${LRRT_IREE_BASE_THREADING_LIB}"
   AND EXISTS "${LRRT_IREE_BASE_TIME_LIB}"
   AND EXISTS "${LRRT_IREE_EXECUTABLE_HEADER_LIB}"
   AND EXISTS "${LRRT_IREE_FLATCC_PARSING_LIB}"
)
  add_executable(lrrt_iree_hal_driver_tests hal_driver.c hal_driver_test.cpp)
  target_include_directories(
    lrrt_iree_hal_driver_tests
    PRIVATE
      "${LRRT_IREE_INCLUDE_DIR}"
      "${PROJECT_SOURCE_DIR}/build-iree-tools/runtime/src"
      "${PROJECT_SOURCE_DIR}/third_party/iree/third_party/flatcc/include"
  )
  target_compile_definitions(
    lrrt_iree_hal_driver_tests
    PRIVATE
      IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl
      LRRT_IREE_HAL_DRIVER_TESTING
  )
  target_link_libraries(
    lrrt_iree_hal_driver_tests
    PRIVATE
      "${LRRT_IREE_HAL_LIB}"
      "${LRRT_IREE_ASYNC_LIB}"
      "${LRRT_IREE_BASE_PATH_LIB}"
      "${LRRT_IREE_BASE_THREADING_LIB}"
      "${LRRT_IREE_BASE_LIB}"
      "${LRRT_IREE_BASE_TIME_LIB}"
      "${LRRT_IREE_EXECUTABLE_HEADER_LIB}"
      "${LRRT_IREE_FLATCC_PARSING_LIB}"
      lrrt::lrrt
  )
  add_test(
    NAME lrrt_iree_hal_driver_tests
    COMMAND lrrt_iree_hal_driver_tests
  )
  set_tests_properties(
    lrrt_iree_hal_driver_tests
    PROPERTIES
      LABELS "iree;gpu"
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  )
else()
  message(
    STATUS
      "IREE HAL driver registration test disabled; IREE runtime static libs "
      "were not found under ${LRRT_IREE_RUNTIME_BUILD_DIR}"
  )
endif()

if(EXISTS "${LRRT_IREE_TOOLING_RUN_MODULE_LIB}")
  file(
    GLOB_RECURSE
    LRRT_IREE_RUNTIME_STATIC_LIBS
    CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/build-iree-tools/runtime/src/iree/*.a"
    "${PROJECT_SOURCE_DIR}/build-iree-tools/build_tools/third_party/*.a"
  )
  add_executable(lrrt_iree_run_module_smoke hal_driver.c run_module_smoke.c)
  target_include_directories(
    lrrt_iree_run_module_smoke
    PRIVATE
      "${PROJECT_SOURCE_DIR}"
      "${LRRT_IREE_INCLUDE_DIR}"
      "${PROJECT_SOURCE_DIR}/build-iree-tools/runtime/src"
      "${PROJECT_SOURCE_DIR}/third_party/iree/third_party/flatcc/include"
  )
  target_compile_definitions(
    lrrt_iree_run_module_smoke
    PRIVATE IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl
  )
  target_link_libraries(
    lrrt_iree_run_module_smoke
    PRIVATE
      lrrt::lrrt
      "-Wl,--start-group"
      ${LRRT_IREE_RUNTIME_STATIC_LIBS}
      "-Wl,--end-group"
      dl
      m
      pthread
  )

  if(LRRT_IREE_COMPILE_EXECUTABLE)
    function(lrrt_add_iree_vmfb_probe probe_test fixture input_mlir artifact_stem)
      set(_probe_dir "${PROJECT_SOURCE_DIR}/build-iree-probe/${artifact_stem}")
      add_test(
        NAME ${probe_test}
        COMMAND
          "${PROJECT_SOURCE_DIR}/tools/iree_compile_probe.sh"
          --iree-compile "${LRRT_IREE_COMPILE_EXECUTABLE}"
          --input "${PROJECT_SOURCE_DIR}/${input_mlir}"
          --target "${LRRT_AMDGPU_TARGET}"
          --artifact-stem "${artifact_stem}"
          --out-dir "${_probe_dir}"
          --try-vmfb
      )
      set_tests_properties(
        ${probe_test}
        PROPERTIES
          FIXTURES_SETUP ${fixture}
          LABELS "iree"
          WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      )
    endfunction()

    function(lrrt_add_iree_vmfb_run run_test fixture artifact_stem function_name expected_output)
      set(_probe_dir "${PROJECT_SOURCE_DIR}/build-iree-probe/${artifact_stem}")
      set(_probe_vmfb "${_probe_dir}/${artifact_stem}_${LRRT_AMDGPU_TARGET}.vmfb")
      add_test(
        NAME ${run_test}
        COMMAND
          lrrt_iree_run_module_smoke
          --device=lrrt
          "--module=${_probe_vmfb}"
          "--function=${function_name}"
          ${ARGN}
          --output=-
      )
      set_tests_properties(
        ${run_test}
        PROPERTIES
          FIXTURES_REQUIRED ${fixture}
          LABELS "iree;gpu"
          PASS_REGULAR_EXPRESSION "${expected_output}"
          WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      )
    endfunction()

    function(lrrt_add_iree_vmfb_smoke
             probe_test
             run_test
             fixture
             input_mlir
             artifact_stem
             function_name
             expected_output)
      lrrt_add_iree_vmfb_probe(
        ${probe_test}
        ${fixture}
        ${input_mlir}
        ${artifact_stem}
      )
      lrrt_add_iree_vmfb_run(
        ${run_test}
        ${fixture}
        ${artifact_stem}
        ${function_name}
        "${expected_output}"
        ${ARGN}
      )
    endfunction()

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
  if(TARGET lrrt_iree_hal_driver_tests)
    target_compile_definitions(
      lrrt_iree_hal_driver_tests
      PRIVATE
        LRRT_IREE_PROBE_HSACO_PATH="${LRRT_IREE_PROBE_HSACO}"
        LRRT_IREE_PROBE_SYMBOL="simple_mul_dispatch_0_elementwise_4_f32"
    )
    set_tests_properties(
      lrrt_iree_hal_driver_tests
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
