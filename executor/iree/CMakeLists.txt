add_library(lrrt_iree_adapter INTERFACE)
add_library(lrrt::iree_adapter ALIAS lrrt_iree_adapter)

target_include_directories(
  lrrt_iree_adapter
  INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${LRRT_IREE_INCLUDE_DIR}"
)

target_link_libraries(lrrt_iree_adapter INTERFACE lrrt::lrrt)

add_executable(lrrt_iree_adapter_smoke smoke.cpp)
target_link_libraries(lrrt_iree_adapter_smoke PRIVATE lrrt::iree_adapter)
add_test(NAME lrrt_iree_adapter_smoke COMMAND lrrt_iree_adapter_smoke)
