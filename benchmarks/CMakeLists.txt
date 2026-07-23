add_executable(lrrt_launch_overhead_benchmark launch_overhead.cpp)
target_link_libraries(lrrt_launch_overhead_benchmark PRIVATE lrrt::lrrt)

add_executable(lrrt_async_copy_launch_benchmark async_copy_launch.cpp)
target_link_libraries(lrrt_async_copy_launch_benchmark PRIVATE lrrt::lrrt)

if(LRRT_BUILD_TRITON_BENCHMARKS)
  add_executable(
    lrrt_triton_mini_decoder_weight_bundle
    triton_mini_decoder_weight_bundle.cpp
  )
  target_link_libraries(
    lrrt_triton_mini_decoder_weight_bundle
    PRIVATE lrrt::bundle
  )
  target_include_directories(
    lrrt_triton_mini_decoder_weight_bundle
    PRIVATE
      "${PROJECT_SOURCE_DIR}"
      "${PROJECT_SOURCE_DIR}/executor/triton"
  )
endif()

if(LRRT_BUILD_TRITON_BENCHMARKS AND UV_EXECUTABLE)
  add_executable(
    lrrt_triton_mini_decoder_layer_benchmark
    triton_mini_decoder_layer.cpp
  )
  target_link_libraries(
    lrrt_triton_mini_decoder_layer_benchmark
    PRIVATE lrrt::bundle
  )
  target_include_directories(
    lrrt_triton_mini_decoder_layer_benchmark
    PRIVATE
      "${PROJECT_SOURCE_DIR}"
      "${PROJECT_SOURCE_DIR}/executor/triton"
  )
elseif(LRRT_BUILD_TRITON_BENCHMARKS)
  message(STATUS "uv not found; skipping Triton mini decoder layer benchmark")
endif()

lrrt_add_hsaco_kernel(
  launch_benchmark_kernel benchmarks/launch_kernel.cpp
  LAUNCH_BENCHMARK_HSACO
)
lrrt_add_hsaco_kernel(
  async_copy_launch_benchmark_kernel
  benchmarks/async_copy_launch_kernel.cpp
  ASYNC_COPY_LAUNCH_BENCHMARK_HSACO
)

if(LRRT_BUILD_TRITON_BENCHMARKS)
  lrrt_add_triton_bundle(
    triton_benchmark_rmsnorm examples/triton/rmsnorm_kernel.py
    TRITON_BENCHMARK_RMSNORM_HSACO TRITON_BENCHMARK_RMSNORM_MANIFEST
    kernels_fp32_2048.hsaco kernels_fp32_4096.hsaco
    kernels_fp16_1024.hsaco kernels_fp16_2048.hsaco kernels_fp16_4096.hsaco
    kernels_bf16_1024.hsaco kernels_bf16_2048.hsaco kernels_bf16_4096.hsaco
  )
  lrrt_add_triton_bundle(
    triton_benchmark_rope examples/triton/rope_kernel.py
    TRITON_BENCHMARK_ROPE_HSACO TRITON_BENCHMARK_ROPE_MANIFEST
    kernels_128.hsaco kernels_256.hsaco
  )
  lrrt_add_triton_bundle(
    triton_benchmark_silu_mul examples/triton/silu_mul_kernel.py
    TRITON_BENCHMARK_SILU_MUL_HSACO TRITON_BENCHMARK_SILU_MUL_MANIFEST
  )
  lrrt_add_triton_bundle(
    triton_benchmark_causal_softmax examples/triton/causal_softmax_kernel.py
    TRITON_BENCHMARK_CAUSAL_SOFTMAX_HSACO
    TRITON_BENCHMARK_CAUSAL_SOFTMAX_MANIFEST kernels_2048.hsaco
    kernels_4096.hsaco
  )
  lrrt_add_triton_bundle(
    triton_benchmark_matvec examples/triton/matvec_kernel.py
    TRITON_BENCHMARK_MATVEC_HSACO TRITON_BENCHMARK_MATVEC_MANIFEST
    kernels_2048.hsaco kernels_4096.hsaco
    kernels_fp16_1024.hsaco kernels_fp16_2048.hsaco kernels_fp16_4096.hsaco
    kernels_bf16_1024.hsaco kernels_bf16_2048.hsaco kernels_bf16_4096.hsaco
  )
  lrrt_add_triton_bundle(
    triton_benchmark_attention_score examples/triton/attention_score_kernel.py
    TRITON_BENCHMARK_ATTENTION_SCORE_HSACO
    TRITON_BENCHMARK_ATTENTION_SCORE_MANIFEST kernels_128.hsaco
    kernels_256.hsaco
  )
  lrrt_add_triton_bundle(
    triton_benchmark_value_aggregation
    examples/triton/value_aggregation_kernel.py
    TRITON_BENCHMARK_VALUE_AGGREGATION_HSACO
    TRITON_BENCHMARK_VALUE_AGGREGATION_MANIFEST kernels_2048.hsaco
    kernels_4096.hsaco
  )
  lrrt_add_triton_bundle(
    triton_benchmark_kv_cache examples/triton/kv_cache_kernel.py
    TRITON_BENCHMARK_KV_CACHE_HSACO TRITON_BENCHMARK_KV_CACHE_MANIFEST
    kernels_read.hsaco kernels_128.hsaco kernels_read_128.hsaco
    kernels_256.hsaco kernels_read_256.hsaco
  )
  lrrt_add_triton_bundle(
    triton_benchmark_vector_add examples/triton/vector_add_kernel.py
    TRITON_BENCHMARK_VECTOR_ADD_HSACO TRITON_BENCHMARK_VECTOR_ADD_MANIFEST
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

if(TRITON_BENCHMARK_RMSNORM_HSACO AND TRITON_BENCHMARK_MATVEC_HSACO AND
   TRITON_BENCHMARK_ATTENTION_SCORE_HSACO AND
   TRITON_BENCHMARK_CAUSAL_SOFTMAX_HSACO AND
   TRITON_BENCHMARK_VALUE_AGGREGATION_HSACO AND
   TRITON_BENCHMARK_VECTOR_ADD_HSACO AND TRITON_BENCHMARK_KV_CACHE_HSACO AND
   TRITON_BENCHMARK_ROPE_HSACO AND TRITON_BENCHMARK_SILU_MUL_HSACO)
  add_dependencies(
    lrrt_triton_mini_decoder_layer_benchmark
    triton_benchmark_rmsnorm_bundle
    triton_benchmark_matvec_bundle
    triton_benchmark_attention_score_bundle
    triton_benchmark_causal_softmax_bundle
    triton_benchmark_value_aggregation_bundle
    triton_benchmark_vector_add_bundle
    triton_benchmark_kv_cache_bundle
    triton_benchmark_rope_bundle
    triton_benchmark_silu_mul_bundle
  )
  target_compile_definitions(
    lrrt_triton_mini_decoder_layer_benchmark
    PRIVATE
      LRRT_TRITON_MINI_LAYER_RMSNORM_MANIFEST="${TRITON_BENCHMARK_RMSNORM_MANIFEST}"
      LRRT_TRITON_MINI_LAYER_MATVEC_MANIFEST="${TRITON_BENCHMARK_MATVEC_MANIFEST}"
      LRRT_TRITON_MINI_LAYER_ATTENTION_SCORE_MANIFEST="${TRITON_BENCHMARK_ATTENTION_SCORE_MANIFEST}"
      LRRT_TRITON_MINI_LAYER_CAUSAL_SOFTMAX_MANIFEST="${TRITON_BENCHMARK_CAUSAL_SOFTMAX_MANIFEST}"
      LRRT_TRITON_MINI_LAYER_VALUE_AGGREGATION_MANIFEST="${TRITON_BENCHMARK_VALUE_AGGREGATION_MANIFEST}"
      LRRT_TRITON_MINI_LAYER_VECTOR_ADD_MANIFEST="${TRITON_BENCHMARK_VECTOR_ADD_MANIFEST}"
      LRRT_TRITON_MINI_LAYER_KV_CACHE_MANIFEST="${TRITON_BENCHMARK_KV_CACHE_MANIFEST}"
      LRRT_TRITON_MINI_LAYER_ROPE_MANIFEST="${TRITON_BENCHMARK_ROPE_MANIFEST}"
      LRRT_TRITON_MINI_LAYER_SILU_MUL_MANIFEST="${TRITON_BENCHMARK_SILU_MUL_MANIFEST}"
  )
endif()
