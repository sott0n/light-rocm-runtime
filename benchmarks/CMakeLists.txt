add_executable(lrrt_launch_overhead_benchmark launch_overhead.cpp)
target_link_libraries(lrrt_launch_overhead_benchmark PRIVATE lrrt::lrrt)

add_executable(lrrt_async_copy_launch_benchmark async_copy_launch.cpp)
target_link_libraries(lrrt_async_copy_launch_benchmark PRIVATE lrrt::lrrt)

lrrt_add_hsaco_kernel(
  launch_benchmark_kernel benchmarks/launch_kernel.cpp
  LAUNCH_BENCHMARK_HSACO
)
lrrt_add_hsaco_kernel(
  async_copy_launch_benchmark_kernel
  benchmarks/async_copy_launch_kernel.cpp
  ASYNC_COPY_LAUNCH_BENCHMARK_HSACO
)

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
