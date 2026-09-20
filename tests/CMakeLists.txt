function(lrrt_add_test_executable target source library)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME ${target} COMMAND ${target})
endfunction()

function(lrrt_bind_hsaco target dependency define path)
  add_dependencies(${target} ${dependency})
  target_compile_definitions(${target} PRIVATE ${define}="${path}")
endfunction()

lrrt_add_test_executable(
  lrrt_aql_producer_tests
  aql_producer.cpp
  lrrt::lrrt
)
target_include_directories(
  lrrt_aql_producer_tests
  PRIVATE "${PROJECT_SOURCE_DIR}/src"
)

if(LRRT_SELECTED_BACKEND STREQUAL "light-rocr")
  set(
    LRRT_LIGHT_ROCR_GPU_TESTS
    lrrt_light_rocr_backend_tests
    lrrt_memory_bounds_tests
    lrrt_allocation_registry_tests
    lrrt_async_copy_tests
    lrrt_pinned_host_memory_tests
  )
  lrrt_add_test_executable(
    lrrt_light_rocr_backend_tests
    light_rocr_backend.cpp
    lrrt::lrrt
  )
  lrrt_add_test_executable(
    lrrt_light_rocr_kernarg_tests
    light_rocr_kernargs.cpp
    lrrt::lrrt
  )
  target_include_directories(
    lrrt_light_rocr_kernarg_tests
    PRIVATE "${PROJECT_SOURCE_DIR}/src"
  )
  target_link_libraries(
    lrrt_light_rocr_kernarg_tests
    PRIVATE light_rocr::loader
  )
  lrrt_add_test_executable(
    lrrt_memory_bounds_tests
    memory_bounds.cpp
    lrrt::lrrt
  )
  lrrt_add_test_executable(
    lrrt_allocation_registry_tests
    allocation_registry.cpp
    lrrt::lrrt
  )
  lrrt_add_test_executable(
    lrrt_async_copy_tests
    async_copy.cpp
    lrrt::lrrt
  )
  lrrt_add_test_executable(
    lrrt_pinned_host_memory_tests
    pinned_host_memory.cpp
    lrrt::lrrt
  )
  target_compile_definitions(
    lrrt_pinned_host_memory_tests
    PRIVATE LRRT_TEST_LIGHT_ROCR=1
  )
  lrrt_add_hsaco_kernel(
    light_rocr_copy_kernel
    tests/copy_kernel_kernel.cpp
    LIGHT_ROCR_COPY_HSACO
    -DLRRT_TEST_HIDDEN_KERNARGS=1
  )
  if(LIGHT_ROCR_COPY_HSACO)
    lrrt_add_test_executable(
      lrrt_copy_kernel_tests
      copy_kernel.cpp
      lrrt::lrrt
    )
    lrrt_bind_hsaco(
      lrrt_copy_kernel_tests
      light_rocr_copy_kernel_hsaco
      LRRT_COPY_HSACO
      "${LIGHT_ROCR_COPY_HSACO}"
    )
    target_compile_definitions(
      lrrt_copy_kernel_tests
      PRIVATE LRRT_TEST_HIDDEN_KERNARGS=1
    )
    lrrt_add_test_executable(
      lrrt_kernel_event_dependencies_tests
      kernel_event_dependencies.cpp
      lrrt::lrrt
    )
    lrrt_bind_hsaco(
      lrrt_kernel_event_dependencies_tests
      light_rocr_copy_kernel_hsaco
      LRRT_COPY_HSACO
      "${LIGHT_ROCR_COPY_HSACO}"
    )
    target_compile_definitions(
      lrrt_kernel_event_dependencies_tests
      PRIVATE LRRT_TEST_HIDDEN_KERNARGS=1
    )
    list(APPEND LRRT_LIGHT_ROCR_GPU_TESTS lrrt_copy_kernel_tests)
    list(
      APPEND LRRT_LIGHT_ROCR_GPU_TESTS lrrt_kernel_event_dependencies_tests
    )
  endif()

  get_property(VECTOR_ADD_HSACO GLOBAL PROPERTY LRRT_VECTOR_ADD_HSACO)
  if(VECTOR_ADD_HSACO)
    lrrt_add_test_executable(
      lrrt_handle_registry_tests
      handle_registry.cpp
      lrrt::lrrt
    )
    lrrt_bind_hsaco(
      lrrt_handle_registry_tests
      vector_add_kernel_hsaco
      LRRT_VECTOR_ADD_HSACO
      "${VECTOR_ADD_HSACO}"
    )
    list(APPEND LRRT_LIGHT_ROCR_GPU_TESTS lrrt_handle_registry_tests)
  endif()

  set_tests_properties(
    ${LRRT_LIGHT_ROCR_GPU_TESTS}
    PROPERTIES RESOURCE_LOCK light_rocr_gpu TIMEOUT 20
  )
  return()
endif()

set(
  LRRT_CORE_TESTS
  lrrt_memory_bounds_tests memory_bounds.cpp
  lrrt_allocation_registry_tests allocation_registry.cpp
  lrrt_runtime_api_tests runtime_api.cpp
  lrrt_memory_stats_tests memory_stats.cpp
  lrrt_pinned_host_memory_tests pinned_host_memory.cpp
  lrrt_double_buffer_pipeline_tests double_buffer_pipeline.cpp
  lrrt_device_buffer_ownership_tests device_buffer_ownership.cpp
  lrrt_queue_event_lifetime_tests queue_event_lifetime.cpp
  lrrt_module_buffer_lifetime_tests module_buffer_lifetime.cpp
  lrrt_async_copy_tests async_copy.cpp
  lrrt_async_copy_launch_tests async_copy_launch.cpp
  lrrt_launch_async_copy_tests launch_async_copy.cpp
  lrrt_explicit_dependencies_tests explicit_dependencies.cpp
  lrrt_multiple_queues_tests multiple_queues.cpp
  lrrt_handle_registry_tests handle_registry.cpp
  lrrt_scale_tests scale.cpp
  lrrt_copy_kernel_tests copy_kernel.cpp
  lrrt_kernel_event_dependencies_tests kernel_event_dependencies.cpp
  lrrt_event_timing_tests event_timing.cpp
  lrrt_cpp_wrapper_tests cpp_wrapper.cpp
)

while(LRRT_CORE_TESTS)
  list(POP_FRONT LRRT_CORE_TESTS target source)
  lrrt_add_test_executable(${target} ${source} lrrt::lrrt)
endwhile()

set(LRRT_BUNDLE_TESTS lrrt_bundle_tests bundle.cpp)

while(LRRT_BUNDLE_TESTS)
  list(POP_FRONT LRRT_BUNDLE_TESTS target source)
  lrrt_add_test_executable(${target} ${source} lrrt::bundle)
endwhile()

set(
  LRRT_TRITON_EXECUTOR_TESTS
  lrrt_triton_executor_tests executor/triton/triton_executor.cpp
)

set(LRRT_QWEN_EXECUTOR_TESTS lrrt_qwen_weight_bundle_tests qwen_weight_bundle.cpp)

if(PYTHON_EXECUTABLE)
  add_test(
    NAME lrrt_qwen_layer_converter_tests
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/tests/qwen_layer_converter_test.py"
  )
  add_test(
    NAME lrrt_qwen_benchmark_runner_tests
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/tests/qwen_benchmark_runner_test.py"
  )
  add_test(
    NAME lrrt_double_buffer_pipeline_sweep_tests
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/tests/double_buffer_pipeline_sweep_test.py"
  )
  add_test(
    NAME lrrt_qwen_e2e_runner_tests
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/tests/qwen_e2e_runner_test.py"
  )
  add_test(
    NAME lrrt_write_iree_qwen_decode_bundle_tests
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/tests/tools/iree/test_write_qwen_decode_bundle.py"
  )
  add_test(
    NAME lrrt_iree_metadata_summary_tests
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/tests/tools/iree/test_metadata_summary.py"
  )
  add_test(
    NAME lrrt_select_iree_qwen_linear_variant_tests
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/tests/tools/iree/test_select_qwen_linear_variant.py"
  )
endif()

while(LRRT_TRITON_EXECUTOR_TESTS)
  list(POP_FRONT LRRT_TRITON_EXECUTOR_TESTS target source)
  lrrt_add_test_executable(${target} ${source} lrrt::bundle)
  target_include_directories(
    ${target}
    PRIVATE "${PROJECT_SOURCE_DIR}/executor/triton"
  )
endwhile()

while(LRRT_QWEN_EXECUTOR_TESTS)
  list(POP_FRONT LRRT_QWEN_EXECUTOR_TESTS target source)
  lrrt_add_test_executable(${target} ${source} lrrt::bundle)
  target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}")
endwhile()

lrrt_add_hsaco_kernel(scale_kernel tests/scale_kernel.cpp SCALE_HSACO)
lrrt_add_hsaco_kernel(copy_kernel tests/copy_kernel_kernel.cpp COPY_HSACO)
lrrt_add_hsaco_kernel(
  async_copy_launch_kernel tests/async_copy_launch_kernel.cpp
  ASYNC_COPY_LAUNCH_HSACO
)

get_property(VECTOR_ADD_HSACO GLOBAL PROPERTY LRRT_VECTOR_ADD_HSACO)
if(VECTOR_ADD_HSACO)
  lrrt_bind_hsaco(
    lrrt_handle_registry_tests
    vector_add_kernel_hsaco
    LRRT_VECTOR_ADD_HSACO
    "${VECTOR_ADD_HSACO}"
  )
endif()

if(SCALE_HSACO)
  foreach(target IN ITEMS lrrt_scale_tests lrrt_event_timing_tests
                          lrrt_cpp_wrapper_tests)
    lrrt_bind_hsaco(${target} scale_kernel_hsaco LRRT_SCALE_HSACO
                    "${SCALE_HSACO}")
  endforeach()
endif()

if(ASYNC_COPY_LAUNCH_HSACO)
  foreach(
    target
    IN
    ITEMS
      lrrt_async_copy_launch_tests
      lrrt_launch_async_copy_tests
      lrrt_explicit_dependencies_tests
      lrrt_multiple_queues_tests
      lrrt_queue_event_lifetime_tests
      lrrt_module_buffer_lifetime_tests
      lrrt_double_buffer_pipeline_tests
  )
    lrrt_bind_hsaco(
      ${target}
      async_copy_launch_kernel_hsaco
      LRRT_ASYNC_COPY_LAUNCH_HSACO
      "${ASYNC_COPY_LAUNCH_HSACO}"
    )
  endforeach()
endif()

if(COPY_HSACO)
  lrrt_bind_hsaco(lrrt_copy_kernel_tests copy_kernel_hsaco LRRT_COPY_HSACO
                  "${COPY_HSACO}")
  lrrt_bind_hsaco(
    lrrt_kernel_event_dependencies_tests
    copy_kernel_hsaco
    LRRT_COPY_HSACO
    "${COPY_HSACO}"
  )
endif()
