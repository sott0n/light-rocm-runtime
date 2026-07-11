#include "hal_driver_test_common.hpp"

using namespace lrrt::iree::test;

namespace {

bool test_default_and_custom_registration() {
  if (!expect_ok(lrrt_iree_hal_register_all(), "lrrt_iree_hal_register_all")) {
    return false;
  }
  if (!expect_ok(lrrt_iree_hal_register_all(),
                 "lrrt_iree_hal_register_all idempotent")) {
    return false;
  }

  DriverPtr default_registry_driver;
  if (!expect_ok(iree_hal_driver_registry_try_create(
                     iree_hal_driver_registry_default(), IREE_SV("lrrt"),
                     iree_allocator_system(), default_registry_driver.out()),
                 "default registry create")) {
    return false;
  }

  RegistryPtr registry;
  if (!expect_ok(iree_hal_driver_registry_allocate(iree_allocator_system(),
                                                   registry.out()),
                 "registry allocate")) {
    return false;
  }
  if (!expect_ok(lrrt_iree_hal_register_all_available_drivers(registry.get()),
                 "register available drivers")) {
    return false;
  }

  iree_status_t duplicate_status =
      lrrt_iree_hal_driver_module_register(registry.get());
  if (!iree_status_is_already_exists(duplicate_status)) {
    iree_status_ignore(duplicate_status);
    return false;
  }
  iree_status_free(duplicate_status);

  iree_host_size_t driver_info_count = 0;
  iree_hal_driver_info_t *driver_infos = nullptr;
  iree_status_t status = iree_hal_driver_registry_enumerate(
      registry.get(), iree_allocator_system(), &driver_info_count,
      &driver_infos);
  const bool ok = iree_status_is_ok(status) && driver_info_count == 1 &&
                  string_view_equal(driver_infos[0].driver_name, "lrrt");
  iree_status_ignore(status);
  iree_allocator_free(iree_allocator_system(), driver_infos);
  return ok;
}

bool test_device_contract() {
  HalFixture fixture;
  if (!fixture.setup()) {
    return false;
  }

  iree_host_size_t device_info_count = 0;
  iree_hal_device_info_t *device_infos = nullptr;
  iree_status_t status = iree_hal_driver_query_available_devices(
      fixture.driver.get(), iree_allocator_system(), &device_info_count,
      &device_infos);
  const bool device_info_ok =
      iree_status_is_ok(status) && device_info_count == 1 &&
      string_view_equal(device_infos[0].path, "default");
  iree_status_ignore(status);
  iree_allocator_free(iree_allocator_system(), device_infos);
  if (!device_info_ok) {
    return false;
  }

  if (!string_view_equal(iree_hal_device_id(fixture.device.get()), "lrrt")) {
    return false;
  }

  int64_t query_value = 0;
  const char *accepted_device_ids[] = {"lrrt", "hip", "amdgpu"};
  for (const char *id : accepted_device_ids) {
    if (!expect_ok(iree_hal_device_query_i64(
                       fixture.device.get(), IREE_SV("hal.device.id"),
                       iree_make_cstring_view(id), &query_value),
                   "device id query") ||
        query_value != 1) {
      return false;
    }
  }
  if (!expect_ok(iree_hal_device_query_i64(
                     fixture.device.get(), IREE_SV("hal.executable.format"),
                     IREE_SV("rocm-hsaco-fb"), &query_value),
                 "executable format query") ||
      query_value != 1) {
    return false;
  }
  if (!expect_ok(iree_hal_device_query_i64(fixture.device.get(),
                                           IREE_SV("hal.device.id"),
                                           IREE_SV("anything"), &query_value),
                 "unknown device id query") ||
      query_value != 0) {
    return false;
  }

  iree_hal_device_capabilities_t capabilities;
  return expect_ok(iree_hal_device_query_capabilities(fixture.device.get(),
                                                      &capabilities),
                   "query capabilities") &&
         capabilities.flags == IREE_HAL_DEVICE_CAPABILITY_NONE;
}

bool test_semaphore_and_queue_dependencies() {
  HalFixture fixture;
  if (!fixture.setup()) {
    return false;
  }

  SemaphorePtr timeline_semaphore;
  uint64_t semaphore_value = 0;
  if (!expect_ok(iree_hal_semaphore_create(
                     fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                     /*initial_value=*/3, IREE_HAL_SEMAPHORE_FLAG_NONE,
                     timeline_semaphore.out()),
                 "semaphore create") ||
      iree_hal_device_query_semaphore_compatibility(fixture.device.get(),
                                                    timeline_semaphore.get()) !=
          IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL ||
      !expect_ok(
          iree_hal_semaphore_query(timeline_semaphore.get(), &semaphore_value),
          "semaphore query") ||
      semaphore_value != 3 ||
      !expect_ok(iree_hal_semaphore_signal(timeline_semaphore.get(), 5,
                                           /*frontier=*/nullptr),
                 "semaphore signal") ||
      !expect_ok(iree_hal_semaphore_wait(timeline_semaphore.get(), 5,
                                         iree_immediate_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE),
                 "semaphore wait") ||
      !expect_ok(
          iree_hal_semaphore_query(timeline_semaphore.get(), &semaphore_value),
          "semaphore query after wait") ||
      semaphore_value != 5) {
    return false;
  }

  SemaphorePtr update_signal;
  SemaphorePtr copy_signal;
  uint64_t update_signal_value = 1;
  uint64_t copy_signal_value = 2;
  iree_hal_semaphore_t *update_signal_ptrs[1] = {update_signal.get()};
  iree_hal_semaphore_t *copy_signal_ptrs[1] = {copy_signal.get()};
  if (!expect_ok(iree_hal_semaphore_create(
                     fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                     IREE_HAL_SEMAPHORE_FLAG_NONE, update_signal.out()),
                 "update semaphore create") ||
      !expect_ok(iree_hal_semaphore_create(
                     fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                     IREE_HAL_SEMAPHORE_FLAG_NONE, copy_signal.out()),
                 "copy semaphore create")) {
    return false;
  }
  update_signal_ptrs[0] = update_signal.get();
  copy_signal_ptrs[0] = copy_signal.get();
  const iree_hal_semaphore_list_t update_signal_list = {
      1,
      update_signal_ptrs,
      &update_signal_value,
  };
  const iree_hal_semaphore_list_t copy_wait_list = {
      1,
      update_signal_ptrs,
      &update_signal_value,
  };
  const iree_hal_semaphore_list_t copy_signal_list = {
      1,
      copy_signal_ptrs,
      &copy_signal_value,
  };
  uint64_t observed_copy_signal = 0;
  return expect_ok(iree_hal_device_queue_update(
                       fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                       iree_hal_semaphore_list_empty(), update_signal_list,
                       /*source_buffer=*/nullptr, /*source_offset=*/0,
                       /*target_buffer=*/nullptr, /*target_offset=*/0,
                       /*length=*/0, IREE_HAL_UPDATE_FLAG_NONE),
                   "queue update signal") &&
         expect_ok(iree_hal_device_queue_copy(
                       fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                       copy_wait_list, copy_signal_list,
                       /*source_buffer=*/nullptr, /*source_offset=*/0,
                       /*target_buffer=*/nullptr, /*target_offset=*/0,
                       /*length=*/0, IREE_HAL_COPY_FLAG_NONE),
                   "queue copy wait/signal") &&
         expect_ok(
             iree_hal_semaphore_query(copy_signal.get(), &observed_copy_signal),
             "copy signal query") &&
         observed_copy_signal == copy_signal_value;
}

bool test_registration_suite() {
  return test_default_and_custom_registration() && test_device_contract() &&
         test_semaphore_and_queue_dependencies();
}

} // namespace

int main() {
  return run_test("lrrt IREE HAL registration tests", test_registration_suite);
}
