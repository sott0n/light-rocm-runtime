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
