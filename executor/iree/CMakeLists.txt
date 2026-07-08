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

if(LRRT_IREE_COMPILE_EXECUTABLE AND LRRT_IREE_RUN_MODULE_EXECUTABLE)
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
