add_executable(
  lrrt_iree_qwen_decode_bundle_tests
  "${PROJECT_SOURCE_DIR}/tests/executor/iree/qwen/decode_bundle_test.cpp"
)
target_include_directories(
  lrrt_iree_qwen_decode_bundle_tests
  PRIVATE "${PROJECT_SOURCE_DIR}"
)
target_link_libraries(
  lrrt_iree_qwen_decode_bundle_tests
  PRIVATE lrrt::iree_adapter
)
add_test(
  NAME lrrt_iree_qwen_decode_bundle_tests
  COMMAND lrrt_iree_qwen_decode_bundle_tests
)

if(TARGET lrrt_iree_vmfb_runner)
  add_executable(
    lrrt_iree_qwen_decode1_e2e
    qwen/qwen_decode1_e2e.cpp
  )
  target_link_libraries(
    lrrt_iree_qwen_decode1_e2e
    PRIVATE lrrt::iree_vmfb_runner
  )

  if(LRRT_IREE_COMPILE_EXECUTABLE)
    set(_qwen_decode_smoke_vmfb
        "${PROJECT_SOURCE_DIR}/build-iree-probe/mini_decoder_layer_rope_kv_cache_outputs/mini_decoder_layer_rope_kv_cache_outputs_${LRRT_AMDGPU_TARGET}.vmfb"
    )
    add_executable(
      lrrt_iree_qwen_decode_e2e_smoke
      "${PROJECT_SOURCE_DIR}/tests/executor/iree/qwen/decode_e2e_smoke.cpp"
    )
    target_link_libraries(
      lrrt_iree_qwen_decode_e2e_smoke
      PRIVATE lrrt::iree_vmfb_runner
    )
    add_test(
      NAME lrrt_iree_qwen_decode_e2e_smoke
      COMMAND
        lrrt_iree_qwen_decode_e2e_smoke
        "${_qwen_decode_smoke_vmfb}"
    )
    set_tests_properties(
      lrrt_iree_qwen_decode_e2e_smoke
      PROPERTIES
        FIXTURES_REQUIRED
          lrrt_iree_mini_decoder_layer_rope_kv_cache_outputs_vmfb
        LABELS "iree;gpu"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    )
  endif()
endif()
