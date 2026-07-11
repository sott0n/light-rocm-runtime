function(lrrt_find_iree)
  set(
    _iree_hints
    "${PROJECT_SOURCE_DIR}/third_party/iree"
    "${PROJECT_SOURCE_DIR}/build-iree-tools"
  )
  if(LRRT_IREE_ROOT)
    list(
      APPEND
      _iree_hints
      "${LRRT_IREE_ROOT}"
      "${LRRT_IREE_ROOT}/include"
      "${LRRT_IREE_ROOT}/runtime/src"
      "${LRRT_IREE_ROOT}/build"
      "${LRRT_IREE_ROOT}/build/runtime/src"
    )
  endif()

  find_path(
    LRRT_IREE_INCLUDE_DIR
    NAMES iree/base/api.h
    HINTS ${_iree_hints}
    PATH_SUFFIXES include runtime/src
  )

  find_program(
    LRRT_IREE_COMPILE_EXECUTABLE
    NAMES iree-compile
    HINTS ${_iree_hints}
    PATH_SUFFIXES bin tools iree/tools
  )

  find_program(
    LRRT_IREE_RUN_MODULE_EXECUTABLE
    NAMES iree-run-module
    HINTS ${_iree_hints}
    PATH_SUFFIXES bin tools iree/tools
  )

  if(NOT LRRT_IREE_INCLUDE_DIR)
    message(
      FATAL_ERROR
        "LRRT_ENABLE_IREE_ADAPTER=ON requires IREE headers. "
        "Set LRRT_IREE_ROOT to an IREE install or source tree containing "
        "iree/base/api.h."
    )
  endif()

  if(NOT EXISTS "${LRRT_IREE_INCLUDE_DIR}/iree/hal/api.h")
    message(
      FATAL_ERROR
        "LRRT_ENABLE_IREE_ADAPTER=ON found IREE base headers at "
        "${LRRT_IREE_INCLUDE_DIR}, but iree/hal/api.h is missing. "
        "Use an IREE runtime development tree or install."
    )
  endif()

  if(NOT LRRT_IREE_COMPILE_EXECUTABLE)
    message(
      STATUS
        "IREE compiler not found; IREE adapter validation tests that require "
        "iree-compile will be unavailable"
    )
  endif()

  if(NOT LRRT_IREE_RUN_MODULE_EXECUTABLE)
    message(
      STATUS
        "IREE runner not found; IREE adapter validation tests that require "
        "iree-run-module will be unavailable"
    )
  endif()

  message(STATUS "IREE adapter enabled")
  message(STATUS "IREE headers: ${LRRT_IREE_INCLUDE_DIR}")
  if(LRRT_IREE_COMPILE_EXECUTABLE)
    message(STATUS "IREE compiler: ${LRRT_IREE_COMPILE_EXECUTABLE}")
  endif()
  if(LRRT_IREE_RUN_MODULE_EXECUTABLE)
    message(STATUS "IREE runner: ${LRRT_IREE_RUN_MODULE_EXECUTABLE}")
  endif()
endfunction()

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

function(lrrt_add_iree_vmfb_run run_test fixture artifact_stem function_name
         expected_output)
  set(_probe_dir "${PROJECT_SOURCE_DIR}/build-iree-probe/${artifact_stem}")
  set(_probe_vmfb "${_probe_dir}/${artifact_stem}_${LRRT_AMDGPU_TARGET}.vmfb")
  add_test(
    NAME ${run_test}
    COMMAND
      lrrt_iree_run_module
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
