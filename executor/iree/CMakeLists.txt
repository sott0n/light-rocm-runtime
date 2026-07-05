add_library(lrrt_iree_adapter INTERFACE)
add_library(lrrt::iree_adapter ALIAS lrrt_iree_adapter)

target_include_directories(
  lrrt_iree_adapter
  INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${LRRT_IREE_INCLUDE_DIR}"
)

target_link_libraries(lrrt_iree_adapter INTERFACE lrrt::lrrt)
