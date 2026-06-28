function(lrrt_add_test_executable target source library)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME ${target} COMMAND ${target})
endfunction()

lrrt_add_test_executable(lrrt_memory_bounds_tests memory_bounds.cpp lrrt::lrrt)
lrrt_add_test_executable(lrrt_async_copy_tests async_copy.cpp lrrt::lrrt)
lrrt_add_test_executable(
  lrrt_async_copy_launch_tests
  async_copy_launch.cpp
  lrrt::lrrt
)
lrrt_add_test_executable(
  lrrt_launch_async_copy_tests
  launch_async_copy.cpp
  lrrt::lrrt
)
lrrt_add_test_executable(
  lrrt_explicit_dependencies_tests
  explicit_dependencies.cpp
  lrrt::lrrt
)
lrrt_add_test_executable(
  lrrt_multiple_queues_tests
  multiple_queues.cpp
  lrrt::lrrt
)
lrrt_add_test_executable(
  lrrt_handle_registry_tests
  handle_registry.cpp
  lrrt::lrrt
)
lrrt_add_test_executable(lrrt_scale_tests scale.cpp lrrt::lrrt)
lrrt_add_test_executable(lrrt_copy_kernel_tests copy_kernel.cpp lrrt::lrrt)
lrrt_add_test_executable(lrrt_event_timing_tests event_timing.cpp lrrt::lrrt)
lrrt_add_test_executable(lrrt_cpp_wrapper_tests cpp_wrapper.cpp lrrt::lrrt)
lrrt_add_test_executable(lrrt_bundle_tests bundle.cpp lrrt::bundle)

lrrt_add_test_executable(
  lrrt_triton_executor_tests
  triton_executor.cpp
  lrrt::bundle
)
target_include_directories(
  lrrt_triton_executor_tests
  PRIVATE "${PROJECT_SOURCE_DIR}/executor/triton"
)

lrrt_add_hsaco_kernel(scale_kernel tests/scale_kernel.cpp SCALE_HSACO)
lrrt_add_hsaco_kernel(copy_kernel tests/copy_kernel_kernel.cpp COPY_HSACO)
lrrt_add_hsaco_kernel(
  async_copy_launch_kernel tests/async_copy_launch_kernel.cpp
  ASYNC_COPY_LAUNCH_HSACO
)

get_property(VECTOR_ADD_HSACO GLOBAL PROPERTY LRRT_VECTOR_ADD_HSACO)
if(VECTOR_ADD_HSACO)
  add_dependencies(lrrt_handle_registry_tests vector_add_kernel_hsaco)
  target_compile_definitions(
    lrrt_handle_registry_tests
    PRIVATE LRRT_VECTOR_ADD_HSACO="${VECTOR_ADD_HSACO}"
  )
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
  add_dependencies(lrrt_multiple_queues_tests async_copy_launch_kernel_hsaco)
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
  target_compile_definitions(
    lrrt_multiple_queues_tests
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
