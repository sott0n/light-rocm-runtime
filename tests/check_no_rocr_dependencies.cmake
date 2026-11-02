if(NOT DEFINED EXECUTABLE OR NOT EXISTS "${EXECUTABLE}")
  message(FATAL_ERROR "EXECUTABLE must name a built direct KFD executable")
endif()

file(
  GET_RUNTIME_DEPENDENCIES
  EXECUTABLES "${EXECUTABLE}"
  RESOLVED_DEPENDENCIES_VAR resolved_dependencies
  UNRESOLVED_DEPENDENCIES_VAR unresolved_dependencies
)

foreach(dependency IN LISTS resolved_dependencies unresolved_dependencies)
  get_filename_component(dependency_name "${dependency}" NAME)
  if(dependency_name MATCHES "^(libhsa-runtime64|libhsakmt|libamdhip64)\\.")
    message(
      FATAL_ERROR
      "direct KFD executable unexpectedly depends on ${dependency_name}"
    )
  endif()
endforeach()
