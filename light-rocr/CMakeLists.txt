cmake_minimum_required(VERSION 3.18)

project(light_rocr VERSION 0.1.0 LANGUAGES CXX)

option(LIGHT_ROCR_BUILD_TESTS "Build light-rocr host tests" ON)
option(LIGHT_ROCR_BUILD_TOOLS "Build light-rocr inspection tools" ON)

function(light_rocr_enable_warnings target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(
      ${target}
      PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
    )
  endif()
endfunction()

add_library(
  light_rocr_loader
  loader/elf_reader.cpp
  loader/load_plan_reader.cpp
  loader/metadata_reader.cpp
)
add_library(light_rocr::loader ALIAS light_rocr_loader)

target_compile_features(light_rocr_loader PUBLIC cxx_std_17)
target_include_directories(
  light_rocr_loader
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/loader/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(light_rocr_loader PROPERTIES POSITION_INDEPENDENT_CODE ON)
light_rocr_enable_warnings(light_rocr_loader)

if(LIGHT_ROCR_BUILD_TOOLS)
  add_executable(light_rocr_inspect_hsaco tools/inspect_hsaco.cpp)
  target_link_libraries(light_rocr_inspect_hsaco PRIVATE light_rocr::loader)
  light_rocr_enable_warnings(light_rocr_inspect_hsaco)
  set_target_properties(
    light_rocr_inspect_hsaco
    PROPERTIES OUTPUT_NAME light-rocr-inspect-hsaco
  )
endif()

if(LIGHT_ROCR_BUILD_TESTS)
  enable_testing()
  add_executable(light_rocr_loader_tests tests/loader/elf_reader_test.cpp)
  target_link_libraries(light_rocr_loader_tests PRIVATE light_rocr::loader)
  light_rocr_enable_warnings(light_rocr_loader_tests)
  add_test(NAME light_rocr_loader_tests COMMAND light_rocr_loader_tests)

  add_executable(
    light_rocr_metadata_tests
    tests/loader/metadata_reader_test.cpp
  )
  target_link_libraries(light_rocr_metadata_tests PRIVATE light_rocr::loader)
  light_rocr_enable_warnings(light_rocr_metadata_tests)
  add_test(NAME light_rocr_metadata_tests COMMAND light_rocr_metadata_tests)

  add_executable(
    light_rocr_load_plan_tests
    tests/loader/load_plan_reader_test.cpp
  )
  target_link_libraries(light_rocr_load_plan_tests PRIVATE light_rocr::loader)
  light_rocr_enable_warnings(light_rocr_load_plan_tests)
  add_test(NAME light_rocr_load_plan_tests COMMAND light_rocr_load_plan_tests)
endif()
