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
if(EXISTS "${LRRT_IREE_BASE_LIB}"
   AND EXISTS "${LRRT_IREE_HAL_LIB}"
   AND EXISTS "${LRRT_IREE_ASYNC_LIB}"
   AND EXISTS "${LRRT_IREE_BASE_PATH_LIB}"
   AND EXISTS "${LRRT_IREE_BASE_THREADING_LIB}"
   AND EXISTS "${LRRT_IREE_BASE_TIME_LIB}"
)
  add_executable(lrrt_iree_hal_driver_tests hal_driver.c hal_driver_test.cpp)
  target_include_directories(
    lrrt_iree_hal_driver_tests
    PRIVATE "${LRRT_IREE_INCLUDE_DIR}"
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
