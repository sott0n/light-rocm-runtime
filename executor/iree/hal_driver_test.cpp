#include "hal_driver.h"

#include <cstring>
#include <stdio.h>

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

  if (!string_view_equal(iree_hal_device_id(device), "lrrt") ||
      iree_hal_device_allocator(device) != nullptr) {
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
