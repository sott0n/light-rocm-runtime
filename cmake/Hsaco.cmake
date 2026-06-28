function(lrrt_add_hsaco_kernel name source output_var)
  if(AMDCLANGXX AND CLANG_OFFLOAD_BUNDLER)
    set(source_path "${source}")
    if(NOT IS_ABSOLUTE "${source_path}")
      set(source_path "${PROJECT_SOURCE_DIR}/${source}")
    endif()

    set(bundle "${CMAKE_CURRENT_BINARY_DIR}/${name}.bundle")
    set(hsaco "${CMAKE_CURRENT_BINARY_DIR}/${name}.hsaco")
    set(host_object "${CMAKE_CURRENT_BINARY_DIR}/${name}.host.o")

    add_custom_command(
      OUTPUT "${hsaco}" "${bundle}" "${host_object}"
      COMMAND
        "${AMDCLANGXX}" --offload-device-only
        "--offload-arch=${LRRT_AMDGPU_TARGET}" -x hip -include
        hip/hip_runtime.h "${source_path}" -o "${bundle}"
      COMMAND
        "${CLANG_OFFLOAD_BUNDLER}" -type=o
        "-targets=host-x86_64-unknown-linux-gnu-,hipv4-amdgcn-amd-amdhsa--${LRRT_AMDGPU_TARGET}"
        "-input=${bundle}" "-output=${host_object}" "-output=${hsaco}"
        -unbundle
      DEPENDS "${source_path}"
      VERBATIM
    )
    add_custom_target(${name}_hsaco DEPENDS "${hsaco}")
    set(${output_var} "${hsaco}" PARENT_SCOPE)
  endif()
endfunction()
