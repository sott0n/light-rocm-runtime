#include "hal_driver.h"

#include <cstring>
#include <fstream>
#include <stdio.h>
#include <vector>

#include "lrrt/lrrt.h"

namespace {

bool string_view_equal(iree_string_view_t value, const char *expected) {
  return value.size == std::strlen(expected) &&
         std::memcmp(value.data, expected, value.size) == 0;
}

bool read_file(const char *path, std::vector<uint8_t> *out_data) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }
  file.seekg(0, std::ios::end);
  const std::streamoff size = file.tellg();
  if (size <= 0) {
    return false;
  }
  file.seekg(0, std::ios::beg);
  out_data->resize(static_cast<size_t>(size));
  return static_cast<bool>(
      file.read(reinterpret_cast<char *>(out_data->data()), size));
}

int test_lrrt_driver_registration() {
  iree_hal_driver_registry_t *registry = nullptr;
  iree_status_t status =
      iree_hal_driver_registry_allocate(iree_allocator_system(), &registry);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 1;
  }

  status = lrrt_iree_hal_driver_module_register(registry);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  iree_host_size_t driver_info_count = 0;
  iree_hal_driver_info_t *driver_infos = nullptr;
  status = iree_hal_driver_registry_enumerate(
      registry, iree_allocator_system(), &driver_info_count, &driver_infos);
  if (!iree_status_is_ok(status) || driver_info_count != 1 ||
      !string_view_equal(driver_infos[0].driver_name, "lrrt")) {
    iree_status_ignore(status);
    iree_allocator_free(iree_allocator_system(), driver_infos);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  iree_allocator_free(iree_allocator_system(), driver_infos);

  iree_hal_driver_t *driver = nullptr;
  status = iree_hal_driver_registry_try_create(
      registry, IREE_SV("lrrt"), iree_allocator_system(), &driver);
  if (!iree_status_is_ok(status) || !driver) {
    iree_status_ignore(status);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  iree_host_size_t device_info_count = 0;
  iree_hal_device_info_t *device_infos = nullptr;
  status = iree_hal_driver_query_available_devices(
      driver, iree_allocator_system(), &device_info_count, &device_infos);
  if (!iree_status_is_ok(status) || device_info_count != 1 ||
      !string_view_equal(device_infos[0].path, "default")) {
    iree_status_ignore(status);
    iree_allocator_free(iree_allocator_system(), device_infos);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  iree_allocator_free(iree_allocator_system(), device_infos);

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  iree_hal_device_t *device = nullptr;
  status = iree_hal_driver_create_default_device(
      driver, &create_params, iree_allocator_system(), &device);
  if (!iree_status_is_ok(status) || !device) {
    iree_status_ignore(status);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  iree_hal_allocator_t *allocator = iree_hal_device_allocator(device);
  if (!string_view_equal(iree_hal_device_id(device), "lrrt") || !allocator) {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  if (!iree_status_is_ok(iree_hal_allocator_trim(allocator))) {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  iree_host_size_t heap_count = 99;
  status =
      iree_hal_allocator_query_memory_heaps(allocator, 0, nullptr, &heap_count);
  if (!iree_status_is_ok(status) || heap_count != 0) {
    iree_status_ignore(status);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  iree_hal_buffer_params_t buffer_params = {};
  buffer_params.usage =
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  iree_device_size_t allocation_size = 128;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          allocator, buffer_params, allocation_size, nullptr, nullptr);
  if ((compatibility & IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE) == 0 ||
      (compatibility & IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER) == 0 ||
      (compatibility & IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH) == 0) {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  iree_hal_buffer_t *buffer = nullptr;
  status = iree_hal_allocator_allocate_buffer(allocator, buffer_params, 128,
                                              &buffer);
  if (!iree_status_is_ok(status) || !buffer ||
      iree_hal_buffer_allocation_size(buffer) != 128 ||
      iree_hal_buffer_byte_length(buffer) != 128 ||
      iree_hal_buffer_memory_type(buffer) !=
          IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL ||
      iree_hal_buffer_allowed_usage(buffer) !=
          (IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
           IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    iree_status_ignore(status);
    if (buffer) {
      iree_hal_buffer_release(buffer);
    }
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  iree_hal_buffer_release(buffer);

  iree_hal_buffer_t *source_buffer = nullptr;
  iree_hal_buffer_t *target_buffer = nullptr;
  status = iree_hal_allocator_allocate_buffer(allocator, buffer_params, 32,
                                              &source_buffer);
  if (!iree_status_is_ok(status) || !source_buffer) {
    iree_status_ignore(status);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  status = iree_hal_allocator_allocate_buffer(allocator, buffer_params, 32,
                                              &target_buffer);
  if (!iree_status_is_ok(status) || !target_buffer) {
    iree_status_ignore(status);
    iree_hal_buffer_release(source_buffer);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  const uint32_t source_values[4] = {3, 1, 4, 1};
  status = iree_hal_device_queue_update(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      iree_hal_semaphore_list_empty(), source_values, 0, source_buffer, 0,
      sizeof(source_values), IREE_HAL_UPDATE_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    iree_hal_buffer_release(target_buffer);
    iree_hal_buffer_release(source_buffer);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  status = iree_hal_device_queue_copy(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      iree_hal_semaphore_list_empty(), source_buffer, 0, target_buffer, 0,
      sizeof(source_values), IREE_HAL_COPY_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    iree_hal_buffer_release(target_buffer);
    iree_hal_buffer_release(source_buffer);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  void *target_device_ptr = nullptr;
  status = lrrt_iree_hal_buffer_device_pointer_for_test(target_buffer,
                                                        &target_device_ptr);
  if (!iree_status_is_ok(status) || !target_device_ptr) {
    iree_status_ignore(status);
    iree_hal_buffer_release(target_buffer);
    iree_hal_buffer_release(source_buffer);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  uint32_t copied_values[4] = {};
  lr_device_t lrrt_device = {0};
  if (lr_device_open(0, &lrrt_device) != LR_SUCCESS ||
      lr_memcpy(lrrt_device, copied_values, target_device_ptr,
                sizeof(copied_values),
                LR_MEMCPY_DEVICE_TO_HOST) != LR_SUCCESS ||
      std::memcmp(copied_values, source_values, sizeof(source_values)) != 0) {
    iree_hal_buffer_release(target_buffer);
    iree_hal_buffer_release(source_buffer);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  iree_hal_buffer_release(target_buffer);
  iree_hal_buffer_release(source_buffer);

  if (iree_hal_allocator_supports_virtual_memory(allocator)) {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  iree_hal_executable_cache_t *executable_cache = nullptr;
  status = iree_hal_executable_cache_create(device, IREE_SV("test-cache"),
                                            &executable_cache);
  if (!iree_status_is_ok(status) || !executable_cache) {
    iree_status_ignore(status);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  if (!iree_hal_executable_cache_can_prepare_format(
          executable_cache, (iree_hal_executable_caching_mode_t)0,
          IREE_SV("rocm-hsaco")) ||
      !iree_hal_executable_cache_can_prepare_format(
          executable_cache, (iree_hal_executable_caching_mode_t)0,
          IREE_SV("amdgpu-hsaco")) ||
      iree_hal_executable_cache_can_prepare_format(
          executable_cache, (iree_hal_executable_caching_mode_t)0,
          IREE_SV("vmfb"))) {
    iree_hal_executable_cache_release(executable_cache);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  const uint8_t hsaco_prefix[4] = {0x7f, 'E', 'L', 'F'};
  char inferred_format[32] = {};
  iree_host_size_t inferred_size = 0;
  status = iree_hal_executable_cache_infer_format(
      executable_cache, (iree_hal_executable_caching_mode_t)0,
      iree_make_const_byte_span(hsaco_prefix, sizeof(hsaco_prefix)),
      sizeof(inferred_format), inferred_format, &inferred_size);
  if (!iree_status_is_ok(status) ||
      std::strcmp(inferred_format, "rocm-hsaco") != 0 ||
      inferred_size != sizeof(hsaco_prefix)) {
    iree_status_ignore(status);
    iree_hal_executable_cache_release(executable_cache);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.executable_format = IREE_SV("rocm-hsaco");
  executable_params.executable_data =
      iree_make_const_byte_span(hsaco_prefix, sizeof(hsaco_prefix));
  iree_hal_executable_t *executable = nullptr;
  status = iree_hal_executable_cache_prepare_executable(
      executable_cache, &executable_params, &executable);
  if (iree_status_is_ok(status) || executable) {
    iree_status_ignore(status);
    if (executable) {
      iree_hal_executable_release(executable);
    }
    iree_hal_executable_cache_release(executable_cache);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  iree_status_ignore(status);

#if defined(LRRT_IREE_PROBE_HSACO_PATH) && defined(LRRT_IREE_PROBE_SYMBOL)
  std::vector<uint8_t> hsaco;
  if (!read_file(LRRT_IREE_PROBE_HSACO_PATH, &hsaco)) {
    iree_hal_executable_cache_release(executable_cache);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.executable_format = IREE_SV("rocm-hsaco");
  executable_params.executable_data =
      iree_make_const_byte_span(hsaco.data(), hsaco.size());
  status = iree_hal_executable_cache_prepare_executable(
      executable_cache, &executable_params, &executable);
  if (!iree_status_is_ok(status) || !executable ||
      iree_hal_executable_function_count(executable) != 0) {
    iree_status_ignore(status);
    if (executable) {
      iree_hal_executable_release(executable);
    }
    iree_hal_executable_cache_release(executable_cache);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  status = iree_hal_executable_lookup_function_by_name(
      executable, IREE_SV(LRRT_IREE_PROBE_SYMBOL), &function);
  if (!iree_status_is_ok(status) ||
      !iree_hal_executable_function_is_index_in_range(function, 1) ||
      iree_hal_executable_function_count(executable) != 1) {
    iree_status_ignore(status);
    iree_hal_executable_release(executable);
    iree_hal_executable_cache_release(executable_cache);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  iree_hal_executable_function_info_t function_info;
  status =
      iree_hal_executable_function_info(executable, function, &function_info);
  if (!iree_status_is_ok(status) ||
      !string_view_equal(function_info.name, LRRT_IREE_PROBE_SYMBOL) ||
      function_info.parameter_count != 0) {
    iree_status_ignore(status);
    iree_hal_executable_release(executable);
    iree_hal_executable_cache_release(executable_cache);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  iree_hal_buffer_t *lhs_buffer = nullptr;
  iree_hal_buffer_t *rhs_buffer = nullptr;
  iree_hal_buffer_t *out_buffer = nullptr;
  status = iree_hal_allocator_allocate_buffer(allocator, buffer_params,
                                              4 * sizeof(float), &lhs_buffer);
  if (!iree_status_is_ok(status) || !lhs_buffer) {
    iree_status_ignore(status);
    iree_hal_executable_release(executable);
    iree_hal_executable_cache_release(executable_cache);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  status = iree_hal_allocator_allocate_buffer(allocator, buffer_params,
                                              4 * sizeof(float), &rhs_buffer);
  if (!iree_status_is_ok(status) || !rhs_buffer) {
    iree_status_ignore(status);
    iree_hal_buffer_release(lhs_buffer);
    iree_hal_executable_release(executable);
    iree_hal_executable_cache_release(executable_cache);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  status = iree_hal_allocator_allocate_buffer(allocator, buffer_params,
                                              4 * sizeof(float), &out_buffer);
  if (!iree_status_is_ok(status) || !out_buffer) {
    iree_status_ignore(status);
    iree_hal_buffer_release(rhs_buffer);
    iree_hal_buffer_release(lhs_buffer);
    iree_hal_executable_release(executable);
    iree_hal_executable_cache_release(executable_cache);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  const float lhs_values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float rhs_values[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  const float expected_values[4] = {10.0f, 40.0f, 90.0f, 160.0f};
  status = iree_hal_device_queue_update(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      iree_hal_semaphore_list_empty(), lhs_values, 0, lhs_buffer, 0,
      sizeof(lhs_values), IREE_HAL_UPDATE_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_update(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        iree_hal_semaphore_list_empty(), rhs_values, 0, rhs_buffer, 0,
        sizeof(rhs_values), IREE_HAL_UPDATE_FLAG_NONE);
  }
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    iree_hal_buffer_release(out_buffer);
    iree_hal_buffer_release(rhs_buffer);
    iree_hal_buffer_release(lhs_buffer);
    iree_hal_executable_release(executable);
    iree_hal_executable_cache_release(executable_cache);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  iree_hal_buffer_ref_t dispatch_refs[3] = {
      iree_hal_make_indirect_buffer_ref(0, 0, sizeof(lhs_values)),
      iree_hal_make_indirect_buffer_ref(1, 0, sizeof(rhs_values)),
      iree_hal_make_indirect_buffer_ref(2, 0, sizeof(expected_values)),
  };
  const iree_hal_buffer_ref_list_t dispatch_bindings = {
      3,
      dispatch_refs,
  };
  iree_hal_buffer_binding_t binding_table_entries[3] = {
      {lhs_buffer, 0, sizeof(lhs_values)},
      {rhs_buffer, 0, sizeof(rhs_values)},
      {out_buffer, 0, sizeof(expected_values)},
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      3,
      binding_table_entries,
  };
  iree_hal_dispatch_config_t dispatch_config =
      iree_hal_make_static_dispatch_config(1, 1, 1);
  dispatch_config.workgroup_size[0] = 32;
  dispatch_config.workgroup_size[1] = 1;
  dispatch_config.workgroup_size[2] = 1;
  iree_hal_command_buffer_t *command_buffer = nullptr;
  status = iree_hal_command_buffer_create(
      device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/3, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_dispatch(
        command_buffer, executable, function, dispatch_config,
        iree_const_byte_span_empty(), dispatch_bindings,
        IREE_HAL_DISPATCH_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_execute(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        iree_hal_semaphore_list_empty(), command_buffer, binding_table,
        IREE_HAL_EXECUTE_FLAG_NONE);
  }
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    if (command_buffer) {
      iree_hal_command_buffer_release(command_buffer);
    }
    iree_hal_buffer_release(out_buffer);
    iree_hal_buffer_release(rhs_buffer);
    iree_hal_buffer_release(lhs_buffer);
    iree_hal_executable_release(executable);
    iree_hal_executable_cache_release(executable_cache);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  iree_hal_command_buffer_release(command_buffer);

  void *out_device_ptr = nullptr;
  status =
      lrrt_iree_hal_buffer_device_pointer_for_test(out_buffer, &out_device_ptr);
  float actual_values[4] = {};
  if (!iree_status_is_ok(status) || !out_device_ptr ||
      lr_memcpy(lrrt_device, actual_values, out_device_ptr,
                sizeof(actual_values),
                LR_MEMCPY_DEVICE_TO_HOST) != LR_SUCCESS ||
      std::memcmp(actual_values, expected_values, sizeof(expected_values)) !=
          0) {
    iree_status_ignore(status);
    iree_hal_buffer_release(out_buffer);
    iree_hal_buffer_release(rhs_buffer);
    iree_hal_buffer_release(lhs_buffer);
    iree_hal_executable_release(executable);
    iree_hal_executable_cache_release(executable_cache);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  iree_hal_buffer_release(out_buffer);
  iree_hal_buffer_release(rhs_buffer);
  iree_hal_buffer_release(lhs_buffer);
  iree_hal_executable_release(executable);
#endif

  iree_hal_executable_cache_release(executable_cache);

  iree_hal_device_capabilities_t capabilities;
  status = iree_hal_device_query_capabilities(device, &capabilities);
  if (!iree_status_is_ok(status) ||
      capabilities.flags != IREE_HAL_DEVICE_CAPABILITY_NONE) {
    iree_status_ignore(status);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  int64_t value = 0;
  status = iree_hal_device_query_i64(device, IREE_SV("hal.device.id"),
                                     IREE_SV("anything"), &value);
  if (iree_status_code(status) != IREE_STATUS_NOT_FOUND || value != 0) {
    iree_status_ignore(status);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  iree_status_ignore(status);

  iree_hal_device_release(device);
  iree_hal_driver_release(driver);
  iree_hal_driver_registry_free(registry);
  return 0;
}

} // namespace

int main() {
  if (test_lrrt_driver_registration() != 0) {
    fprintf(stderr, "lrrt IREE HAL driver registration test failed\n");
    return 1;
  }
  return 0;
}
