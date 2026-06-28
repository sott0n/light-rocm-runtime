add_executable(lrrt_memory_bounds_tests memory_bounds.cpp)
target_link_libraries(lrrt_memory_bounds_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_async_copy_tests async_copy.cpp)
target_link_libraries(lrrt_async_copy_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_async_copy_launch_tests async_copy_launch.cpp)
target_link_libraries(lrrt_async_copy_launch_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_launch_async_copy_tests launch_async_copy.cpp)
target_link_libraries(lrrt_launch_async_copy_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_explicit_dependencies_tests explicit_dependencies.cpp)
target_link_libraries(lrrt_explicit_dependencies_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_multiple_queues_tests multiple_queues.cpp)
target_link_libraries(lrrt_multiple_queues_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_handle_registry_tests handle_registry.cpp)
target_link_libraries(lrrt_handle_registry_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_scale_tests scale.cpp)
target_link_libraries(lrrt_scale_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_copy_kernel_tests copy_kernel.cpp)
target_link_libraries(lrrt_copy_kernel_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_event_timing_tests event_timing.cpp)
target_link_libraries(lrrt_event_timing_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_cpp_wrapper_tests cpp_wrapper.cpp)
target_link_libraries(lrrt_cpp_wrapper_tests PRIVATE lrrt::lrrt)

add_executable(lrrt_bundle_tests bundle.cpp)
target_link_libraries(lrrt_bundle_tests PRIVATE lrrt::bundle)

add_executable(lrrt_triton_executor_tests triton_executor.cpp)
target_include_directories(
  lrrt_triton_executor_tests
  PRIVATE "${PROJECT_SOURCE_DIR}/executor/triton"
)
target_link_libraries(lrrt_triton_executor_tests PRIVATE lrrt::bundle)

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

add_test(NAME lrrt_memory_bounds_tests COMMAND lrrt_memory_bounds_tests)
add_test(NAME lrrt_async_copy_tests COMMAND lrrt_async_copy_tests)
add_test(NAME lrrt_async_copy_launch_tests COMMAND lrrt_async_copy_launch_tests)
add_test(NAME lrrt_launch_async_copy_tests COMMAND lrrt_launch_async_copy_tests)
add_test(NAME lrrt_explicit_dependencies_tests
         COMMAND lrrt_explicit_dependencies_tests)
add_test(NAME lrrt_multiple_queues_tests COMMAND lrrt_multiple_queues_tests)
add_test(NAME lrrt_handle_registry_tests COMMAND lrrt_handle_registry_tests)
add_test(NAME lrrt_scale_tests COMMAND lrrt_scale_tests)
add_test(NAME lrrt_copy_kernel_tests COMMAND lrrt_copy_kernel_tests)
add_test(NAME lrrt_event_timing_tests COMMAND lrrt_event_timing_tests)
add_test(NAME lrrt_cpp_wrapper_tests COMMAND lrrt_cpp_wrapper_tests)
add_test(NAME lrrt_bundle_tests COMMAND lrrt_bundle_tests)
add_test(NAME lrrt_triton_executor_tests COMMAND lrrt_triton_executor_tests)
