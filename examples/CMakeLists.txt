add_executable(lrrt_vector_add vector_add/vector_add.cpp)
target_link_libraries(lrrt_vector_add PRIVATE lrrt::lrrt)

add_executable(lrrt_saxpy saxpy/saxpy.cpp)
target_link_libraries(lrrt_saxpy PRIVATE lrrt::lrrt)

add_executable(lrrt_fill fill/fill.cpp)
target_link_libraries(lrrt_fill PRIVATE lrrt::lrrt)

lrrt_add_hsaco_kernel(
  vector_add_kernel examples/vector_add/vector_add_kernel.cpp
  VECTOR_ADD_HSACO
)
lrrt_add_hsaco_kernel(saxpy_kernel examples/saxpy/saxpy_kernel.cpp SAXPY_HSACO)
lrrt_add_hsaco_kernel(fill_kernel examples/fill/fill_kernel.cpp FILL_HSACO)

if(VECTOR_ADD_HSACO)
  add_dependencies(lrrt_vector_add vector_add_kernel_hsaco)
  target_compile_definitions(
    lrrt_vector_add
    PRIVATE LRRT_VECTOR_ADD_HSACO="${VECTOR_ADD_HSACO}"
  )
  set_property(GLOBAL PROPERTY LRRT_VECTOR_ADD_HSACO "${VECTOR_ADD_HSACO}")
endif()

if(SAXPY_HSACO)
  add_dependencies(lrrt_saxpy saxpy_kernel_hsaco)
  target_compile_definitions(
    lrrt_saxpy
    PRIVATE LRRT_SAXPY_HSACO="${SAXPY_HSACO}"
  )
endif()

if(FILL_HSACO)
  add_dependencies(lrrt_fill fill_kernel_hsaco)
  target_compile_definitions(lrrt_fill PRIVATE LRRT_FILL_HSACO="${FILL_HSACO}")
endif()

if(LRRT_ENABLE_TRITON_EXAMPLES AND UV_EXECUTABLE)
  add_subdirectory(triton)
elseif(LRRT_ENABLE_TRITON_EXAMPLES)
  message(STATUS "uv not found; skipping Triton example")
else()
  message(STATUS "Triton examples disabled")
endif()
