find_package(Python3 3.11 COMPONENTS Interpreter Development.SABIModule REQUIRED)

Python3_add_library(
  _lrrt_sparsewave MODULE USE_SABI 3.11 WITH_SOABI python_module.cpp
)
target_link_libraries(_lrrt_sparsewave PRIVATE lrrt::bundle)
target_include_directories(
  _lrrt_sparsewave
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
)

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

add_custom_target(lrrt_sparsewave_python DEPENDS _lrrt_sparsewave)

set_property(
  GLOBAL PROPERTY LRRT_SPARSEWAVE_PYTHON_DIR "${SPARSEWAVE_PYTHON_DIR}"
)
