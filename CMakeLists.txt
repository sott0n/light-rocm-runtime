cmake_minimum_required(VERSION 3.18)

project(light_rocm_runtime VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

add_library(lrrt src/runtime.cpp)
add_library(lrrt::lrrt ALIAS lrrt)

target_include_directories(
  lrrt
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(lrrt PUBLIC cxx_std_17)

add_executable(lrrt_vector_add examples/vector_add/vector_add.cpp)
target_link_libraries(lrrt_vector_add PRIVATE lrrt::lrrt)
