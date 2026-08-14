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

if(NOT EXISTS "${LRRT_SPARSEWAVE_ROOT}/CMakeLists.txt")
  message(
    FATAL_ERROR
    "SparseWave source tree not found at ${LRRT_SPARSEWAVE_ROOT}; initialize "
    "the submodule or set LRRT_SPARSEWAVE_ROOT"
  )
endif()
if(NOT EXISTS "${LRRT_SPARSEWAVE_BUNDLE_EXECUTABLE}" OR NOT EXISTS
                                                       "${LRRT_SPARSEWAVE_OPT_EXECUTABLE}")
  message(
    FATAL_ERROR
    "SparseWave tools not found under ${LRRT_SPARSEWAVE_BUILD_DIR}; build "
    "SparseWave or set LRRT_SPARSEWAVE_BUNDLE_EXECUTABLE and "
    "LRRT_SPARSEWAVE_OPT_EXECUTABLE"
  )
endif()
if(NOT LRRT_AMDGPU_TARGET STREQUAL "gfx1101")
  message(FATAL_ERROR "The initial SparseWave executor requires gfx1101")
endif()
if(NOT UV_EXECUTABLE)
  message(FATAL_ERROR "The SparseWave PyTorch frontend requires uv")
endif()

set(SPARSEWAVE_TORCH_PROJECT "${LRRT_SPARSEWAVE_ROOT}/examples/pytorch")
set(SPARSEWAVE_SPMM_INPUT "${CMAKE_CURRENT_BINARY_DIR}/spmm.torch.mlir")
set(SPARSEWAVE_SPMM_BUNDLE_DIR "${CMAKE_CURRENT_BINARY_DIR}/spmm.bundle")
set(SPARSEWAVE_SPMM_MANIFEST "${SPARSEWAVE_SPMM_BUNDLE_DIR}/manifest.json")
set(SPARSEWAVE_SPMM_HSACO "${SPARSEWAVE_SPMM_BUNDLE_DIR}/kernels.hsaco")
set(
  LRRT_SPARSEWAVE_UV_CACHE_DIR
  "${CMAKE_BINARY_DIR}/uv-cache"
  CACHE PATH "Cache used for the pinned SparseWave PyTorch frontend environment"
)

add_custom_command(
  OUTPUT "${SPARSEWAVE_SPMM_INPUT}"
  COMMAND
    "${CMAKE_COMMAND}" -E env
    "UV_CACHE_DIR=${LRRT_SPARSEWAVE_UV_CACHE_DIR}"
    "UV_PROJECT_ENVIRONMENT=${CMAKE_BINARY_DIR}/sparsewave-pytorch-venv"
    "${UV_EXECUTABLE}" run --project "${SPARSEWAVE_TORCH_PROJECT}" --frozen
    python "${CMAKE_CURRENT_SOURCE_DIR}/generate_spmm.py"
    --output "${SPARSEWAVE_SPMM_INPUT}"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/generate_spmm.py"
    "${SPARSEWAVE_TORCH_PROJECT}/pyproject.toml"
    "${SPARSEWAVE_TORCH_PROJECT}/uv.lock"
    "${LRRT_SPARSEWAVE_ROOT}/python/sparsewave/torch_export/importer.py"
  COMMENT "Exporting the SparseWave SpMM graph from PyTorch"
  VERBATIM
)

add_custom_command(
  OUTPUT "${SPARSEWAVE_SPMM_MANIFEST}" "${SPARSEWAVE_SPMM_HSACO}"
  COMMAND "${CMAKE_COMMAND}" -E rm -rf "${SPARSEWAVE_SPMM_BUNDLE_DIR}"
  COMMAND
    "${LRRT_SPARSEWAVE_BUNDLE_EXECUTABLE}"
    "${SPARSEWAVE_SPMM_INPUT}"
    --output "${SPARSEWAVE_SPMM_BUNDLE_DIR}"
    --sparsewave-opt "${LRRT_SPARSEWAVE_OPT_EXECUTABLE}"
    --chip "${LRRT_AMDGPU_TARGET}"
    --operation spmm
    --mapping wave-per-row-tile
    --block-size 64
    --tile-size 4
    --wavefront-size 32
  COMMAND
    "${LRRT_SPARSEWAVE_BUNDLE_EXECUTABLE}"
    --verify "${SPARSEWAVE_SPMM_BUNDLE_DIR}"
  DEPENDS
    "${SPARSEWAVE_SPMM_INPUT}"
    "${LRRT_SPARSEWAVE_BUNDLE_EXECUTABLE}"
    "${LRRT_SPARSEWAVE_OPT_EXECUTABLE}"
  COMMENT "Generating and verifying the SparseWave SpMM bundle"
  VERBATIM
)

add_custom_target(
  lrrt_sparsewave_spmm_bundle
  DEPENDS "${SPARSEWAVE_SPMM_MANIFEST}" "${SPARSEWAVE_SPMM_HSACO}"
)

add_library(lrrt_sparsewave_executor INTERFACE)
add_library(lrrt::sparsewave_executor ALIAS lrrt_sparsewave_executor)
target_include_directories(
  lrrt_sparsewave_executor
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(lrrt_sparsewave_executor INTERFACE lrrt::bundle)

add_executable(lrrt_sparsewave_spmm spmm_e2e.cpp)
add_dependencies(lrrt_sparsewave_spmm lrrt_sparsewave_spmm_bundle)
target_link_libraries(lrrt_sparsewave_spmm PRIVATE lrrt::sparsewave_executor)
target_compile_definitions(
  lrrt_sparsewave_spmm
  PRIVATE LRRT_SPARSEWAVE_SPMM_MANIFEST="${SPARSEWAVE_SPMM_MANIFEST}"
)

add_test(NAME lrrt_sparsewave_spmm_e2e COMMAND lrrt_sparsewave_spmm)

find_package(Python3 3.11 COMPONENTS Interpreter Development.SABIModule REQUIRED)
Python3_add_library(
  _lrrt_sparsewave MODULE USE_SABI 3.11 WITH_SOABI python_binding.cpp
)
target_link_libraries(_lrrt_sparsewave PRIVATE lrrt::sparsewave_executor)

set(SPARSEWAVE_PYTHON_DIR "${CMAKE_CURRENT_BINARY_DIR}/python")
set_target_properties(
  _lrrt_sparsewave
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${SPARSEWAVE_PYTHON_DIR}"
)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/lrrt_sparsewave.py"
  "${SPARSEWAVE_PYTHON_DIR}/lrrt_sparsewave.py"
  COPYONLY
)

add_custom_target(
  lrrt_sparsewave_python
  DEPENDS _lrrt_sparsewave
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
