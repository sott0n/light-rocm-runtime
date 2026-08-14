set(
  LRRT_SPARSEWAVE_BUNDLE_EXECUTABLE
  "${LRRT_SPARSEWAVE_BUILD_DIR}/bin/sparsewave-bundle"
  CACHE FILEPATH "Path to sparsewave-bundle"
)
set(
  LRRT_SPARSEWAVE_OPT_EXECUTABLE
  "${LRRT_SPARSEWAVE_BUILD_DIR}/bin/sparsewave-opt"
  CACHE FILEPATH "Path to sparsewave-opt"
)
set(
  LRRT_SPARSEWAVE_PYTORCH_OPT_EXECUTABLE
  "${LRRT_SPARSEWAVE_BUILD_DIR}/bin/sparsewave-pytorch-opt"
  CACHE FILEPATH "Path to sparsewave-pytorch-opt"
)

if(NOT EXISTS "${LRRT_SPARSEWAVE_ROOT}/CMakeLists.txt")
  message(
    FATAL_ERROR
    "SparseWave source tree not found at ${LRRT_SPARSEWAVE_ROOT}; initialize "
    "the submodule or set LRRT_SPARSEWAVE_ROOT"
  )
endif()
if(NOT EXISTS "${LRRT_SPARSEWAVE_BUNDLE_EXECUTABLE}" OR NOT EXISTS
                                                       "${LRRT_SPARSEWAVE_OPT_EXECUTABLE}"
   OR NOT EXISTS "${LRRT_SPARSEWAVE_PYTORCH_OPT_EXECUTABLE}")
  message(
    FATAL_ERROR
    "SparseWave tools not found under ${LRRT_SPARSEWAVE_BUILD_DIR}; build "
    "SparseWave or set the LRRT_SPARSEWAVE tool paths"
  )
endif()
if(NOT LRRT_AMDGPU_TARGET STREQUAL "gfx1101")
  message(FATAL_ERROR "The initial SparseWave examples require gfx1101")
endif()
if(NOT UV_EXECUTABLE)
  message(FATAL_ERROR "The SparseWave PyTorch frontend requires uv")
endif()

get_property(SPARSEWAVE_PYTHON_DIR GLOBAL PROPERTY LRRT_SPARSEWAVE_PYTHON_DIR)
set(SPARSEWAVE_TORCH_PROJECT "${LRRT_SPARSEWAVE_ROOT}/examples/pytorch")
set(
  LRRT_SPARSEWAVE_UV_CACHE_DIR
  "${CMAKE_BINARY_DIR}/uv-cache"
  CACHE PATH "Cache used for the pinned SparseWave PyTorch frontend environment"
)

target_sources(
  _lrrt_sparsewave
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/spmm/python_binding.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sparse_attention/python_binding.cpp"
)
target_include_directories(
  _lrrt_sparsewave
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/spmm"
    "${CMAKE_CURRENT_SOURCE_DIR}/sparse_attention"
)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/applications.py"
  "${SPARSEWAVE_PYTHON_DIR}/lrrt_sparsewave_applications.py"
  COPYONLY
)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/spmm/application.py"
  "${SPARSEWAVE_PYTHON_DIR}/lrrt_sparsewave_spmm.py"
  COPYONLY
)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/sparse_attention/application.py"
  "${SPARSEWAVE_PYTHON_DIR}/lrrt_sparsewave_sparse_attention.py"
  COPYONLY
)

function(
  lrrt_add_sparsewave_torch_bundle
  application
  frontend
  operation
  manifest_variable
)
  set(input "${CMAKE_CURRENT_BINARY_DIR}/${application}.torch.mlir")
  set(bundle "${CMAKE_CURRENT_BINARY_DIR}/${application}.bundle")
  set(manifest "${bundle}/manifest.json")

  add_custom_command(
    OUTPUT "${input}"
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "UV_CACHE_DIR=${LRRT_SPARSEWAVE_UV_CACHE_DIR}"
      "UV_PROJECT_ENVIRONMENT=${CMAKE_BINARY_DIR}/sparsewave-pytorch-venv"
      "PYTHONPATH=${SPARSEWAVE_TORCH_PROJECT}"
      "${UV_EXECUTABLE}" run --project "${SPARSEWAVE_TORCH_PROJECT}" --frozen
      python "${CMAKE_CURRENT_SOURCE_DIR}/${application}/export.py"
      --output "${input}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/${application}/export.py"
      "${SPARSEWAVE_TORCH_PROJECT}/${frontend}"
      "${SPARSEWAVE_TORCH_PROJECT}/pyproject.toml"
      "${SPARSEWAVE_TORCH_PROJECT}/uv.lock"
      "${LRRT_SPARSEWAVE_ROOT}/python/sparsewave/torch_export/importer.py"
    COMMENT "Exporting the SparseWave ${application} graph from PyTorch"
    VERBATIM
  )

  set(outputs "${manifest}")
  if(application STREQUAL "spmm")
    list(APPEND outputs "${bundle}/kernels.hsaco")
  else()
    foreach(
      kernel
      IN
      ITEMS
        sparse_attention_scores
        sparse_attention_row_max
        sparse_attention_exp
        sparse_attention_row_sum
        sparse_attention_normalize
        sparse_attention_output
    )
      list(APPEND outputs "${bundle}/kernels/${kernel}.hsaco")
    endforeach()
  endif()

  set(bundle_command
      "${LRRT_SPARSEWAVE_BUNDLE_EXECUTABLE}"
      "${input}"
      --output "${bundle}"
      --sparsewave-opt "${LRRT_SPARSEWAVE_OPT_EXECUTABLE}"
      --sparsewave-pytorch-opt "${LRRT_SPARSEWAVE_PYTORCH_OPT_EXECUTABLE}"
      --chip "${LRRT_AMDGPU_TARGET}"
      --operation "${operation}"
      --rocm-path /opt/rocm
      --block-size 64
      --wavefront-size 32
  )
  if(application STREQUAL "spmm")
    list(APPEND bundle_command --mapping wave-per-row-tile --tile-size 4)
  endif()

  add_custom_command(
    OUTPUT ${outputs}
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${bundle}"
    COMMAND ${bundle_command}
    COMMAND "${LRRT_SPARSEWAVE_BUNDLE_EXECUTABLE}" --verify "${bundle}"
    DEPENDS
      "${input}"
      "${LRRT_SPARSEWAVE_BUNDLE_EXECUTABLE}"
      "${LRRT_SPARSEWAVE_OPT_EXECUTABLE}"
      "${LRRT_SPARSEWAVE_PYTORCH_OPT_EXECUTABLE}"
    COMMENT "Generating and verifying the SparseWave ${application} bundle"
    VERBATIM
  )

  add_custom_target(lrrt_sparsewave_${application}_bundle DEPENDS ${outputs})
  set(${manifest_variable} "${manifest}" PARENT_SCOPE)
endfunction()

lrrt_add_sparsewave_torch_bundle(
  spmm csr_spmm.py spmm SPARSEWAVE_SPMM_MANIFEST
)
lrrt_add_sparsewave_torch_bundle(
  sparse_attention sparse_attention.py sparse-attention
  SPARSEWAVE_ATTENTION_MANIFEST
)

add_dependencies(
  _lrrt_sparsewave
  lrrt_sparsewave_spmm_bundle
  lrrt_sparsewave_sparse_attention_bundle
)

add_test(
  NAME lrrt_sparsewave_python_spmm_e2e
  COMMAND
    "${CMAKE_COMMAND}" -E env
    "PYTHONPATH=${SPARSEWAVE_PYTHON_DIR}"
    "LRRT_SPARSEWAVE_SPMM_MANIFEST=${SPARSEWAVE_SPMM_MANIFEST}"
    "UV_CACHE_DIR=${LRRT_SPARSEWAVE_UV_CACHE_DIR}"
    "UV_PROJECT_ENVIRONMENT=${CMAKE_BINARY_DIR}/sparsewave-pytorch-venv"
    "${UV_EXECUTABLE}" run --project "${SPARSEWAVE_TORCH_PROJECT}" --frozen
    --with pytest==9.0.2 pytest -q
    "${PROJECT_SOURCE_DIR}/tests/executor/sparsewave/test_spmm_e2e.py"
)

add_test(
  NAME lrrt_sparsewave_python_attention_e2e
  COMMAND
    "${CMAKE_COMMAND}" -E env
    "PYTHONPATH=${SPARSEWAVE_PYTHON_DIR}"
    "LRRT_SPARSEWAVE_ATTENTION_MANIFEST=${SPARSEWAVE_ATTENTION_MANIFEST}"
    "UV_CACHE_DIR=${LRRT_SPARSEWAVE_UV_CACHE_DIR}"
    "UV_PROJECT_ENVIRONMENT=${CMAKE_BINARY_DIR}/sparsewave-pytorch-venv"
    "${UV_EXECUTABLE}" run --project "${SPARSEWAVE_TORCH_PROJECT}" --frozen
    --with pytest==9.0.2 pytest -q
    "${PROJECT_SOURCE_DIR}/tests/executor/sparsewave/test_attention_e2e.py"
)

add_custom_target(
  lrrt_sparsewave_sparse_attention_demo
  COMMAND
    "${CMAKE_COMMAND}" -E env
    "PYTHONPATH=${SPARSEWAVE_PYTHON_DIR}:${SPARSEWAVE_TORCH_PROJECT}"
    "UV_CACHE_DIR=${LRRT_SPARSEWAVE_UV_CACHE_DIR}"
    "UV_PROJECT_ENVIRONMENT=${CMAKE_BINARY_DIR}/sparsewave-pytorch-venv"
    "${UV_EXECUTABLE}" run --project "${SPARSEWAVE_TORCH_PROJECT}" --frozen
    python "${CMAKE_CURRENT_SOURCE_DIR}/sparse_attention/demo.py"
    --bundle-tool "${LRRT_SPARSEWAVE_BUNDLE_EXECUTABLE}"
    --sparsewave-opt "${LRRT_SPARSEWAVE_OPT_EXECUTABLE}"
    --sparsewave-pytorch-opt "${LRRT_SPARSEWAVE_PYTORCH_OPT_EXECUTABLE}"
    --chip "${LRRT_AMDGPU_TARGET}"
  DEPENDS _lrrt_sparsewave
  COMMENT "Running the SparseWave-to-lrrt SparseAttention E2E demo"
  USES_TERMINAL
)
