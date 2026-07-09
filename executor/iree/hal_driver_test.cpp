#include "hal_driver.h"

#include <cstring>
#include <stdio.h>

#include "lrrt/lrrt.h"

namespace {

bool string_view_equal(iree_string_view_t value, const char *expected) {
  return value.size == std::strlen(expected) &&
         std::memcmp(value.data, expected, value.size) == 0;
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
