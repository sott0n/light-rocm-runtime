cmake_minimum_required(VERSION 3.18)

project(light_rocr VERSION 0.1.0 LANGUAGES CXX)

find_package(Threads REQUIRED)

option(LIGHT_ROCR_BUILD_TESTS "Build light-rocr host tests" ON)
option(LIGHT_ROCR_BUILD_TOOLS "Build light-rocr inspection tools" ON)
option(
  LIGHT_ROCR_ENABLE_HSAKMT
  "Build the libhsakmt transport when its development files are available"
  ON
)

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

add_library(
  light_rocr_core
  runtime/aql.cpp
  runtime/executable_image.cpp
  runtime/launch.cpp
  runtime/signal.cpp
  runtime/topology.cpp
)
add_library(light_rocr::core ALIAS light_rocr_core)
target_compile_features(light_rocr_core PUBLIC cxx_std_17)
target_include_directories(
  light_rocr_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(light_rocr_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(light_rocr_core PUBLIC light_rocr::loader Threads::Threads)
light_rocr_enable_warnings(light_rocr_core)

set(LIGHT_ROCR_HSAKMT_AVAILABLE OFF)
if(LIGHT_ROCR_ENABLE_HSAKMT)
  find_path(
    LIGHT_ROCR_HSAKMT_INCLUDE_DIR
    NAMES hsakmt/hsakmt.h
    HINTS /opt/rocm/include
  )
  find_library(
    LIGHT_ROCR_HSAKMT_LIBRARY
    NAMES hsakmt
    HINTS /opt/rocm/lib /opt/rocm/lib64
  )
  find_library(LIGHT_ROCR_DRM_LIBRARY NAMES drm)
  find_library(LIGHT_ROCR_DRM_AMDGPU_LIBRARY NAMES drm_amdgpu)
  find_library(LIGHT_ROCR_NUMA_LIBRARY NAMES numa)

  if(
    LIGHT_ROCR_HSAKMT_INCLUDE_DIR
    AND LIGHT_ROCR_HSAKMT_LIBRARY
    AND LIGHT_ROCR_DRM_LIBRARY
    AND LIGHT_ROCR_DRM_AMDGPU_LIBRARY
    AND LIGHT_ROCR_NUMA_LIBRARY
  )
    set(LIGHT_ROCR_HSAKMT_AVAILABLE ON)
    add_library(
      light_rocr_transport_hsakmt
      transport/hsakmt/executable_image.cpp
      transport/hsakmt/kernarg.cpp
      transport/hsakmt/memory.cpp
      transport/hsakmt/queue.cpp
      transport/hsakmt/signal.cpp
      transport/hsakmt/status.cpp
      transport/hsakmt/topology.cpp
    )
    add_library(
      light_rocr::transport_hsakmt
      ALIAS light_rocr_transport_hsakmt
    )
    target_compile_features(light_rocr_transport_hsakmt PUBLIC cxx_std_17)
    target_include_directories(
      light_rocr_transport_hsakmt
      PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/transport/hsakmt/include>
        $<INSTALL_INTERFACE:include>
    )
    target_include_directories(
      light_rocr_transport_hsakmt
      SYSTEM PRIVATE "${LIGHT_ROCR_HSAKMT_INCLUDE_DIR}"
    )
    target_link_libraries(
      light_rocr_transport_hsakmt
      PUBLIC light_rocr::core
      PRIVATE
        "${LIGHT_ROCR_HSAKMT_LIBRARY}"
        "${LIGHT_ROCR_DRM_AMDGPU_LIBRARY}"
        "${LIGHT_ROCR_DRM_LIBRARY}"
        "${LIGHT_ROCR_NUMA_LIBRARY}"
        Threads::Threads
        "${CMAKE_DL_LIBS}"
    )
    set_target_properties(
      light_rocr_transport_hsakmt
      PROPERTIES POSITION_INDEPENDENT_CODE ON
    )
    light_rocr_enable_warnings(light_rocr_transport_hsakmt)
    message(STATUS "light-rocr libhsakmt transport enabled")
  else()
    message(STATUS "light-rocr libhsakmt transport unavailable")
  endif()
endif()

if(LIGHT_ROCR_BUILD_TOOLS)
  add_executable(light_rocr_inspect_hsaco tools/inspect_hsaco.cpp)
  target_link_libraries(light_rocr_inspect_hsaco PRIVATE light_rocr::loader)
  light_rocr_enable_warnings(light_rocr_inspect_hsaco)
  set_target_properties(
    light_rocr_inspect_hsaco
    PROPERTIES OUTPUT_NAME light-rocr-inspect-hsaco
  )

  if(LIGHT_ROCR_HSAKMT_AVAILABLE)
    add_executable(light_rocr_check_image tools/check_image.cpp)
    target_link_libraries(
      light_rocr_check_image
      PRIVATE light_rocr::transport_hsakmt
    )
    light_rocr_enable_warnings(light_rocr_check_image)
    set_target_properties(
      light_rocr_check_image
      PROPERTIES OUTPUT_NAME light-rocr-check-image
    )

    add_executable(light_rocr_check_memory tools/check_memory.cpp)
    target_link_libraries(
      light_rocr_check_memory
      PRIVATE light_rocr::transport_hsakmt
    )
    light_rocr_enable_warnings(light_rocr_check_memory)
    set_target_properties(
      light_rocr_check_memory
      PROPERTIES OUTPUT_NAME light-rocr-check-memory
    )

    add_executable(light_rocr_check_queue tools/check_queue.cpp)
    target_link_libraries(
      light_rocr_check_queue
      PRIVATE light_rocr::transport_hsakmt
    )
    light_rocr_enable_warnings(light_rocr_check_queue)
    set_target_properties(
      light_rocr_check_queue
      PROPERTIES OUTPUT_NAME light-rocr-check-queue
    )

    add_executable(light_rocr_check_signal tools/check_signal.cpp)
    target_link_libraries(
      light_rocr_check_signal
      PRIVATE light_rocr::transport_hsakmt
    )
    light_rocr_enable_warnings(light_rocr_check_signal)
    set_target_properties(
      light_rocr_check_signal
      PROPERTIES OUTPUT_NAME light-rocr-check-signal
    )

    add_executable(light_rocr_check_dispatch tools/check_dispatch.cpp)
    target_link_libraries(
      light_rocr_check_dispatch
      PRIVATE light_rocr::transport_hsakmt
    )
    light_rocr_enable_warnings(light_rocr_check_dispatch)
    set_target_properties(
      light_rocr_check_dispatch
      PROPERTIES OUTPUT_NAME light-rocr-check-dispatch
    )

    add_executable(light_rocr_inspect_topology tools/inspect_topology.cpp)
    target_link_libraries(
      light_rocr_inspect_topology
      PRIVATE light_rocr::transport_hsakmt
    )
    light_rocr_enable_warnings(light_rocr_inspect_topology)
    set_target_properties(
      light_rocr_inspect_topology
      PROPERTIES OUTPUT_NAME light-rocr-inspect-topology
    )
  endif()
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

  add_executable(
    light_rocr_topology_tests
    tests/runtime/topology_test.cpp
  )
  target_link_libraries(light_rocr_topology_tests PRIVATE light_rocr::core)
  light_rocr_enable_warnings(light_rocr_topology_tests)
  add_test(NAME light_rocr_topology_tests COMMAND light_rocr_topology_tests)

  add_executable(light_rocr_signal_tests tests/runtime/signal_test.cpp)
  target_link_libraries(light_rocr_signal_tests PRIVATE light_rocr::core)
  light_rocr_enable_warnings(light_rocr_signal_tests)
  add_test(NAME light_rocr_signal_tests COMMAND light_rocr_signal_tests)

  add_executable(light_rocr_aql_tests tests/runtime/aql_test.cpp)
  target_link_libraries(light_rocr_aql_tests PRIVATE light_rocr::core)
  light_rocr_enable_warnings(light_rocr_aql_tests)
  add_test(NAME light_rocr_aql_tests COMMAND light_rocr_aql_tests)

  add_executable(
    light_rocr_executable_image_tests
    tests/runtime/executable_image_test.cpp
  )
  target_link_libraries(
    light_rocr_executable_image_tests
    PRIVATE light_rocr::core
  )
  light_rocr_enable_warnings(light_rocr_executable_image_tests)
  add_test(
    NAME light_rocr_executable_image_tests
    COMMAND light_rocr_executable_image_tests
  )

  add_executable(light_rocr_launch_tests tests/runtime/launch_test.cpp)
  target_link_libraries(light_rocr_launch_tests PRIVATE light_rocr::core)
  light_rocr_enable_warnings(light_rocr_launch_tests)
  add_test(NAME light_rocr_launch_tests COMMAND light_rocr_launch_tests)

  find_program(
    LIGHT_ROCR_AMDGPU_CLANG
    NAMES clang
    HINTS /opt/rocm/llvm/bin
  )
  find_program(
    LIGHT_ROCR_LLVM_OBJCOPY
    NAMES llvm-objcopy
    HINTS /opt/rocm/llvm/bin
  )
  if(LIGHT_ROCR_AMDGPU_CLANG AND LIGHT_ROCR_LLVM_OBJCOPY)
    set(
      LIGHT_ROCR_FIXED_GFX1101_BUILD_DIR
      "${CMAKE_CURRENT_BINARY_DIR}/fixed-gfx1101-store"
    )
    set(
      LIGHT_ROCR_FIXED_GFX1101_OBJECT
      "${LIGHT_ROCR_FIXED_GFX1101_BUILD_DIR}/fixed_gfx1101_store.o"
    )
    set(
      LIGHT_ROCR_FIXED_GFX1101_IMAGE
      "${LIGHT_ROCR_FIXED_GFX1101_BUILD_DIR}/fixed_gfx1101_store.co"
    )
    set(
      LIGHT_ROCR_FIXED_GFX1101_DESCRIPTOR
      "${LIGHT_ROCR_FIXED_GFX1101_BUILD_DIR}/fixed_gfx1101_store.rodata"
    )
    set(
      LIGHT_ROCR_FIXED_GFX1101_TEXT
      "${LIGHT_ROCR_FIXED_GFX1101_BUILD_DIR}/fixed_gfx1101_store.text"
    )
    add_custom_command(
      OUTPUT
        "${LIGHT_ROCR_FIXED_GFX1101_DESCRIPTOR}"
        "${LIGHT_ROCR_FIXED_GFX1101_TEXT}"
      BYPRODUCTS
        "${LIGHT_ROCR_FIXED_GFX1101_OBJECT}"
        "${LIGHT_ROCR_FIXED_GFX1101_IMAGE}"
      COMMAND
        "${CMAKE_COMMAND}" -E make_directory
        "${LIGHT_ROCR_FIXED_GFX1101_BUILD_DIR}"
      COMMAND
        "${LIGHT_ROCR_AMDGPU_CLANG}" -target amdgcn-amd-amdhsa -mcpu=gfx1101
        -c "${CMAKE_CURRENT_SOURCE_DIR}/tools/fixed_gfx1101_store.s"
        -o "${LIGHT_ROCR_FIXED_GFX1101_OBJECT}"
      COMMAND
        "${LIGHT_ROCR_AMDGPU_CLANG}" -target amdgcn-amd-amdhsa -mcpu=gfx1101
        -nostdlib -Wl,--section-start=.rodata=0x1000
        -Wl,--section-start=.text=0x1100
        "${LIGHT_ROCR_FIXED_GFX1101_OBJECT}"
        -o "${LIGHT_ROCR_FIXED_GFX1101_IMAGE}"
      COMMAND
        "${LIGHT_ROCR_LLVM_OBJCOPY}"
        "--dump-section=.rodata=${LIGHT_ROCR_FIXED_GFX1101_DESCRIPTOR}"
        "${LIGHT_ROCR_FIXED_GFX1101_IMAGE}"
      COMMAND
        "${LIGHT_ROCR_LLVM_OBJCOPY}"
        "--dump-section=.text=${LIGHT_ROCR_FIXED_GFX1101_TEXT}"
        "${LIGHT_ROCR_FIXED_GFX1101_IMAGE}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tools/fixed_gfx1101_store.s"
      COMMENT "Reassembling the fixed gfx1101 dispatch image"
      VERBATIM
    )
    add_custom_target(
      light_rocr_fixed_gfx1101_store_artifact
      DEPENDS
        "${LIGHT_ROCR_FIXED_GFX1101_DESCRIPTOR}"
        "${LIGHT_ROCR_FIXED_GFX1101_TEXT}"
    )
    add_executable(
      light_rocr_fixed_gfx1101_store_tests
      tests/runtime/fixed_gfx1101_store_test.cpp
    )
    target_compile_features(
      light_rocr_fixed_gfx1101_store_tests
      PRIVATE cxx_std_17
    )
    target_include_directories(
      light_rocr_fixed_gfx1101_store_tests
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tools"
    )
    add_dependencies(
      light_rocr_fixed_gfx1101_store_tests
      light_rocr_fixed_gfx1101_store_artifact
    )
    light_rocr_enable_warnings(light_rocr_fixed_gfx1101_store_tests)
    add_test(
      NAME light_rocr_fixed_gfx1101_store_tests
      COMMAND
        light_rocr_fixed_gfx1101_store_tests
        "${LIGHT_ROCR_FIXED_GFX1101_DESCRIPTOR}"
        "${LIGHT_ROCR_FIXED_GFX1101_TEXT}"
    )
  else()
    message(
      STATUS
      "light-rocr fixed gfx1101 reassembly oracle unavailable"
    )
  endif()

  find_path(
    LIGHT_ROCR_ROCR_SIGNAL_ABI_INCLUDE_DIR
    NAMES hsa/amd_hsa_signal.h
    HINTS /opt/rocm/include
  )
  if(LIGHT_ROCR_ROCR_SIGNAL_ABI_INCLUDE_DIR)
    add_executable(
      light_rocr_rocr_signal_abi_tests
      tests/runtime/rocr_signal_abi_test.cpp
    )
    target_include_directories(
      light_rocr_rocr_signal_abi_tests
      SYSTEM PRIVATE "${LIGHT_ROCR_ROCR_SIGNAL_ABI_INCLUDE_DIR}"
    )
    target_link_libraries(
      light_rocr_rocr_signal_abi_tests
      PRIVATE light_rocr::core
    )
    light_rocr_enable_warnings(light_rocr_rocr_signal_abi_tests)
    add_test(
      NAME light_rocr_rocr_signal_abi_tests
      COMMAND light_rocr_rocr_signal_abi_tests
    )

    add_executable(
      light_rocr_rocr_aql_abi_tests
      tests/runtime/rocr_aql_abi_test.cpp
    )
    target_include_directories(
      light_rocr_rocr_aql_abi_tests
      SYSTEM PRIVATE "${LIGHT_ROCR_ROCR_SIGNAL_ABI_INCLUDE_DIR}"
    )
    target_link_libraries(
      light_rocr_rocr_aql_abi_tests
      PRIVATE light_rocr::core
    )
    light_rocr_enable_warnings(light_rocr_rocr_aql_abi_tests)
    add_test(
      NAME light_rocr_rocr_aql_abi_tests
      COMMAND light_rocr_rocr_aql_abi_tests
    )
  endif()

  if(LIGHT_ROCR_HSAKMT_AVAILABLE)
    add_executable(
      light_rocr_hsakmt_kernarg_tests
      tests/hsakmt/kernarg_test.cpp
    )
    target_include_directories(
      light_rocr_hsakmt_kernarg_tests
      SYSTEM PRIVATE "${LIGHT_ROCR_HSAKMT_INCLUDE_DIR}"
    )
    target_link_libraries(
      light_rocr_hsakmt_kernarg_tests
      PRIVATE light_rocr::transport_hsakmt
    )
    light_rocr_enable_warnings(light_rocr_hsakmt_kernarg_tests)
    add_test(
      NAME light_rocr_hsakmt_kernarg_tests
      COMMAND light_rocr_hsakmt_kernarg_tests
    )

    add_executable(
      light_rocr_hsakmt_executable_image_tests
      tests/hsakmt/executable_image_test.cpp
    )
    target_include_directories(
      light_rocr_hsakmt_executable_image_tests
      SYSTEM PRIVATE "${LIGHT_ROCR_HSAKMT_INCLUDE_DIR}"
    )
    target_link_libraries(
      light_rocr_hsakmt_executable_image_tests
      PRIVATE light_rocr::transport_hsakmt
    )
    light_rocr_enable_warnings(light_rocr_hsakmt_executable_image_tests)
    add_test(
      NAME light_rocr_hsakmt_executable_image_tests
      COMMAND light_rocr_hsakmt_executable_image_tests
    )

    add_executable(
      light_rocr_hsakmt_memory_tests
      tests/hsakmt/memory_test.cpp
    )
    target_include_directories(
      light_rocr_hsakmt_memory_tests
      SYSTEM PRIVATE "${LIGHT_ROCR_HSAKMT_INCLUDE_DIR}"
    )
    target_link_libraries(
      light_rocr_hsakmt_memory_tests
      PRIVATE light_rocr::transport_hsakmt
    )
    light_rocr_enable_warnings(light_rocr_hsakmt_memory_tests)
    add_test(
      NAME light_rocr_hsakmt_memory_tests
      COMMAND light_rocr_hsakmt_memory_tests
    )

    add_executable(
      light_rocr_hsakmt_queue_tests
      tests/hsakmt/queue_test.cpp
    )
    target_include_directories(
      light_rocr_hsakmt_queue_tests
      SYSTEM PRIVATE "${LIGHT_ROCR_HSAKMT_INCLUDE_DIR}"
    )
    target_link_libraries(
      light_rocr_hsakmt_queue_tests
      PRIVATE light_rocr::transport_hsakmt
    )
    light_rocr_enable_warnings(light_rocr_hsakmt_queue_tests)
    add_test(
      NAME light_rocr_hsakmt_queue_tests
      COMMAND light_rocr_hsakmt_queue_tests
    )

    add_executable(
      light_rocr_hsakmt_signal_tests
      tests/hsakmt/signal_test.cpp
    )
    target_include_directories(
      light_rocr_hsakmt_signal_tests
      SYSTEM PRIVATE "${LIGHT_ROCR_HSAKMT_INCLUDE_DIR}"
    )
    target_link_libraries(
      light_rocr_hsakmt_signal_tests
      PRIVATE light_rocr::transport_hsakmt
    )
    light_rocr_enable_warnings(light_rocr_hsakmt_signal_tests)
    add_test(
      NAME light_rocr_hsakmt_signal_tests
      COMMAND light_rocr_hsakmt_signal_tests
    )

    add_executable(
      light_rocr_hsakmt_topology_tests
      tests/hsakmt/topology_test.cpp
    )
    target_include_directories(
      light_rocr_hsakmt_topology_tests
      SYSTEM PRIVATE "${LIGHT_ROCR_HSAKMT_INCLUDE_DIR}"
    )
    target_link_libraries(
      light_rocr_hsakmt_topology_tests
      PRIVATE light_rocr::transport_hsakmt
    )
    light_rocr_enable_warnings(light_rocr_hsakmt_topology_tests)
    add_test(
      NAME light_rocr_hsakmt_topology_tests
      COMMAND light_rocr_hsakmt_topology_tests
    )
  endif()
endif()
