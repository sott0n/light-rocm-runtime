function(lrrt_add_triton_executable target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE lrrt::bundle)
  add_test(NAME ${target} COMMAND ${target})
endfunction()

function(lrrt_bind_triton_manifest target bundle_target manifest_define
         manifest_path)
  add_dependencies(${target} ${bundle_target})
  target_compile_definitions(
    ${target}
    PRIVATE ${manifest_define}="${manifest_path}"
  )
endfunction()

lrrt_add_triton_executable(lrrt_triton_vector_add triton_vector_add.cpp)
lrrt_add_triton_executable(lrrt_triton_saxpy triton_saxpy.cpp)
lrrt_add_triton_executable(lrrt_triton_scale triton_scale.cpp)
lrrt_add_triton_executable(lrrt_triton_rmsnorm triton_rmsnorm.cpp)
lrrt_add_triton_executable(
  lrrt_triton_rmsnorm_precision_tests
  "${PROJECT_SOURCE_DIR}/tests/triton_rmsnorm_precision.cpp"
)
lrrt_add_triton_executable(lrrt_triton_rope triton_rope.cpp)
lrrt_add_triton_executable(lrrt_triton_silu_mul triton_silu_mul.cpp)
lrrt_add_triton_executable(lrrt_triton_softmax triton_softmax.cpp)
lrrt_add_triton_executable(
  lrrt_triton_causal_softmax
  triton_causal_softmax.cpp
)
lrrt_add_triton_executable(lrrt_triton_matvec triton_matvec.cpp)
lrrt_add_triton_executable(
  lrrt_triton_matvec_precision_tests
  "${PROJECT_SOURCE_DIR}/tests/triton_matvec_precision.cpp"
)
lrrt_add_triton_executable(lrrt_triton_qkv_projection triton_qkv_projection.cpp)
target_include_directories(
  lrrt_triton_qkv_projection
  PRIVATE "${PROJECT_SOURCE_DIR}/executor/triton"
)
lrrt_add_triton_executable(
  lrrt_triton_attention_score
  triton_attention_score.cpp
)
lrrt_add_triton_executable(
  lrrt_triton_value_aggregation
  triton_value_aggregation.cpp
)
lrrt_add_triton_executable(lrrt_triton_kv_cache triton_kv_cache.cpp)
target_include_directories(
  lrrt_triton_kv_cache
  PRIVATE "${PROJECT_SOURCE_DIR}/executor/triton"
)
lrrt_add_triton_executable(
  lrrt_triton_mini_attention
  triton_mini_attention.cpp
)
target_include_directories(
  lrrt_triton_mini_attention
  PRIVATE "${PROJECT_SOURCE_DIR}/executor/triton"
)

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
  kernels_128.hsaco kernels_256.hsaco
)
lrrt_add_triton_bundle(
  triton_silu_mul examples/triton/silu_mul_kernel.py
  TRITON_SILU_MUL_HSACO TRITON_SILU_MUL_MANIFEST
)
lrrt_add_triton_bundle(
  triton_softmax examples/triton/softmax_kernel.py
  TRITON_SOFTMAX_HSACO TRITON_SOFTMAX_MANIFEST
  kernels_2048.hsaco kernels_4096.hsaco
)
lrrt_add_triton_bundle(
  triton_causal_softmax examples/triton/causal_softmax_kernel.py
  TRITON_CAUSAL_SOFTMAX_HSACO TRITON_CAUSAL_SOFTMAX_MANIFEST
  kernels_2048.hsaco kernels_4096.hsaco
)
lrrt_add_triton_bundle(
  triton_matvec examples/triton/matvec_kernel.py
  TRITON_MATVEC_HSACO TRITON_MATVEC_MANIFEST
  kernels_2048.hsaco kernels_4096.hsaco
  kernels_fp16_1024.hsaco kernels_fp16_2048.hsaco kernels_fp16_4096.hsaco
  kernels_bf16_1024.hsaco kernels_bf16_2048.hsaco kernels_bf16_4096.hsaco
)
lrrt_add_triton_bundle(
  triton_attention_score examples/triton/attention_score_kernel.py
  TRITON_ATTENTION_SCORE_HSACO TRITON_ATTENTION_SCORE_MANIFEST
  kernels_128.hsaco kernels_256.hsaco
)
lrrt_add_triton_bundle(
  triton_value_aggregation examples/triton/value_aggregation_kernel.py
  TRITON_VALUE_AGGREGATION_HSACO TRITON_VALUE_AGGREGATION_MANIFEST
  kernels_2048.hsaco kernels_4096.hsaco
)
lrrt_add_triton_bundle(
  triton_kv_cache examples/triton/kv_cache_kernel.py
  TRITON_KV_CACHE_HSACO TRITON_KV_CACHE_MANIFEST
  kernels_read.hsaco kernels_128.hsaco kernels_read_128.hsaco
  kernels_256.hsaco kernels_read_256.hsaco
)

if(TRITON_VECTOR_ADD_HSACO)
  lrrt_bind_triton_manifest(
    lrrt_triton_vector_add
    triton_vector_add_bundle
    LRRT_TRITON_VECTOR_ADD_MANIFEST
    "${TRITON_VECTOR_ADD_MANIFEST}"
  )
endif()

if(TRITON_SAXPY_HSACO)
  lrrt_bind_triton_manifest(
    lrrt_triton_saxpy
    triton_saxpy_bundle
    LRRT_TRITON_SAXPY_MANIFEST
    "${TRITON_SAXPY_MANIFEST}"
  )
endif()

if(TRITON_SCALE_HSACO)
  lrrt_bind_triton_manifest(
    lrrt_triton_scale
    triton_scale_bundle
    LRRT_TRITON_SCALE_MANIFEST
    "${TRITON_SCALE_MANIFEST}"
  )
endif()

if(TRITON_RMSNORM_HSACO)
  lrrt_bind_triton_manifest(
    lrrt_triton_rmsnorm
    triton_rmsnorm_bundle
    LRRT_TRITON_RMSNORM_MANIFEST
    "${TRITON_RMSNORM_MANIFEST}"
  )
  lrrt_bind_triton_manifest(
    lrrt_triton_rmsnorm_precision_tests
    triton_rmsnorm_bundle
    LRRT_TRITON_RMSNORM_MANIFEST
    "${TRITON_RMSNORM_MANIFEST}"
  )
endif()

if(TRITON_ROPE_HSACO)
  lrrt_bind_triton_manifest(
    lrrt_triton_rope
    triton_rope_bundle
    LRRT_TRITON_ROPE_MANIFEST
    "${TRITON_ROPE_MANIFEST}"
  )
endif()

if(TRITON_SILU_MUL_HSACO)
  lrrt_bind_triton_manifest(
    lrrt_triton_silu_mul
    triton_silu_mul_bundle
    LRRT_TRITON_SILU_MUL_MANIFEST
    "${TRITON_SILU_MUL_MANIFEST}"
  )
endif()

if(TRITON_SOFTMAX_HSACO)
  lrrt_bind_triton_manifest(
    lrrt_triton_softmax
    triton_softmax_bundle
    LRRT_TRITON_SOFTMAX_MANIFEST
    "${TRITON_SOFTMAX_MANIFEST}"
  )
endif()

if(TRITON_CAUSAL_SOFTMAX_HSACO)
  lrrt_bind_triton_manifest(
    lrrt_triton_causal_softmax
    triton_causal_softmax_bundle
    LRRT_TRITON_CAUSAL_SOFTMAX_MANIFEST
    "${TRITON_CAUSAL_SOFTMAX_MANIFEST}"
  )
endif()

if(TRITON_MATVEC_HSACO)
  lrrt_bind_triton_manifest(
    lrrt_triton_matvec
    triton_matvec_bundle
    LRRT_TRITON_MATVEC_MANIFEST
    "${TRITON_MATVEC_MANIFEST}"
  )
  lrrt_bind_triton_manifest(
    lrrt_triton_matvec_precision_tests
    triton_matvec_bundle
    LRRT_TRITON_MATVEC_MANIFEST
    "${TRITON_MATVEC_MANIFEST}"
  )
  lrrt_bind_triton_manifest(
    lrrt_triton_qkv_projection
    triton_matvec_bundle
    LRRT_TRITON_QKV_PROJECTION_MATVEC_MANIFEST
    "${TRITON_MATVEC_MANIFEST}"
  )
endif()

if(TRITON_ATTENTION_SCORE_HSACO)
  lrrt_bind_triton_manifest(
    lrrt_triton_attention_score
    triton_attention_score_bundle
    LRRT_TRITON_ATTENTION_SCORE_MANIFEST
    "${TRITON_ATTENTION_SCORE_MANIFEST}"
  )
endif()

if(TRITON_VALUE_AGGREGATION_HSACO)
  lrrt_bind_triton_manifest(
    lrrt_triton_value_aggregation
    triton_value_aggregation_bundle
    LRRT_TRITON_VALUE_AGGREGATION_MANIFEST
    "${TRITON_VALUE_AGGREGATION_MANIFEST}"
  )
endif()

if(TRITON_KV_CACHE_HSACO)
  lrrt_bind_triton_manifest(
    lrrt_triton_kv_cache
    triton_kv_cache_bundle
    LRRT_TRITON_KV_CACHE_MANIFEST
    "${TRITON_KV_CACHE_MANIFEST}"
  )
endif()

if(TRITON_ATTENTION_SCORE_HSACO AND TRITON_CAUSAL_SOFTMAX_HSACO AND
   TRITON_VALUE_AGGREGATION_HSACO AND TRITON_VECTOR_ADD_HSACO AND
   TRITON_KV_CACHE_HSACO AND TRITON_ROPE_HSACO)
  add_dependencies(
    lrrt_triton_mini_attention
    triton_attention_score_bundle
    triton_causal_softmax_bundle
    triton_value_aggregation_bundle
    triton_vector_add_bundle
    triton_kv_cache_bundle
    triton_rope_bundle
  )
  target_compile_definitions(
    lrrt_triton_mini_attention
    PRIVATE
      LRRT_TRITON_MINI_ATTENTION_SCORE_MANIFEST="${TRITON_ATTENTION_SCORE_MANIFEST}"
      LRRT_TRITON_MINI_CAUSAL_SOFTMAX_MANIFEST="${TRITON_CAUSAL_SOFTMAX_MANIFEST}"
      LRRT_TRITON_MINI_VALUE_AGGREGATION_MANIFEST="${TRITON_VALUE_AGGREGATION_MANIFEST}"
      LRRT_TRITON_MINI_VECTOR_ADD_MANIFEST="${TRITON_VECTOR_ADD_MANIFEST}"
      LRRT_TRITON_MINI_KV_CACHE_MANIFEST="${TRITON_KV_CACHE_MANIFEST}"
      LRRT_TRITON_MINI_ROPE_MANIFEST="${TRITON_ROPE_MANIFEST}"
  )
endif()

if(PYTHON_EXECUTABLE AND LLVM_READELF)
  foreach(
    bundle_name
    vector_add
    saxpy
    scale
    rmsnorm
    rope
    silu_mul
    softmax
    causal_softmax
    matvec
    attention_score
    value_aggregation
    kv_cache
  )
    string(TOUPPER "${bundle_name}" bundle_var)
    set(manifest_var "TRITON_${bundle_var}_MANIFEST")
    add_test(
      NAME lrrt_triton_${bundle_name}_bundle_consistency
      COMMAND
        "${PYTHON_EXECUTABLE}"
        "${PROJECT_SOURCE_DIR}/tests/triton_bundle_consistency.py"
        "--manifest=${${manifest_var}}"
        "--llvm-readelf=${LLVM_READELF}"
    )
  endforeach()
else()
  message(
    STATUS
      "python3 or llvm-readelf not found; skipping Triton bundle consistency test"
  )
endif()
