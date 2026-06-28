function(lrrt_add_triton_bundle name script output_hsaco_var
         output_manifest_var)
  if(UV_EXECUTABLE)
    set(script_path "${script}")
    if(NOT IS_ABSOLUTE "${script_path}")
      set(script_path "${PROJECT_SOURCE_DIR}/${script}")
    endif()

    set(requirements "${PROJECT_SOURCE_DIR}/examples/triton/requirements.txt")
    set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/${name}_bundle")
    set(hsaco "${output_dir}/kernels.hsaco")
    set(manifest "${output_dir}/manifest.json")
    set(outputs "${hsaco}" "${manifest}")
    foreach(byproduct IN LISTS ARGN)
      list(APPEND outputs "${output_dir}/${byproduct}")
    endforeach()

    add_custom_command(
      OUTPUT ${outputs}
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}"
      COMMAND
        "${CMAKE_COMMAND}" -E env
        "TRITON_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/triton-cache"
        "UV_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/uv-cache"
        "${UV_EXECUTABLE}" run --python "${LRRT_TRITON_PYTHON}"
        --no-project --with-requirements "${requirements}" python
        "${script_path}" "--arch=${LRRT_AMDGPU_TARGET}"
        "--output-dir=${output_dir}"
      DEPENDS "${requirements}" "${script_path}"
      VERBATIM
    )
    add_custom_target(${name}_bundle DEPENDS ${outputs})
    set(${output_hsaco_var} "${hsaco}" PARENT_SCOPE)
    set(${output_manifest_var} "${manifest}" PARENT_SCOPE)
  endif()
endfunction()
