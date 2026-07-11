#include "hal_driver.h"

#include <cstring>
#include <fstream>
#include <stdio.h>
#include <vector>

#include "iree/io/file_handle.h"
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
  iree_status_t status = lrrt_iree_hal_register_all();
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 1;
  }
  status = lrrt_iree_hal_register_all();
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 1;
  }

  iree_hal_driver_t *default_registry_driver = nullptr;
  status = iree_hal_driver_registry_try_create(
      iree_hal_driver_registry_default(), IREE_SV("lrrt"),
      iree_allocator_system(), &default_registry_driver);
  if (!iree_status_is_ok(status) || !default_registry_driver) {
    iree_status_ignore(status);
    return 1;
  }
  iree_hal_driver_release(default_registry_driver);

  iree_hal_driver_registry_t *registry = nullptr;
  status =
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

  int64_t query_value = 0;
  status = iree_hal_device_query_i64(device, IREE_SV("hal.device.id"),
                                     IREE_SV("lrrt"), &query_value);
  if (!iree_status_is_ok(status) || query_value != 1) {
    iree_status_ignore(status);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  status = iree_hal_device_query_i64(device, IREE_SV("hal.device.id"),
                                     IREE_SV("hip"), &query_value);
  if (!iree_status_is_ok(status) || query_value != 1) {
    iree_status_ignore(status);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  status = iree_hal_device_query_i64(device, IREE_SV("hal.device.id"),
                                     IREE_SV("amdgpu"), &query_value);
  if (!iree_status_is_ok(status) || query_value != 1) {
    iree_status_ignore(status);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  status = iree_hal_device_query_i64(device, IREE_SV("hal.executable.format"),
                                     IREE_SV("rocm-hsaco-fb"), &query_value);
  if (!iree_status_is_ok(status) || query_value != 1) {
    iree_status_ignore(status);
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

  iree_hal_semaphore_t *timeline_semaphore = nullptr;
  status = iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*initial_value=*/3, IREE_HAL_SEMAPHORE_FLAG_NONE, &timeline_semaphore);
  uint64_t semaphore_value = 0;
  if (!iree_status_is_ok(status) || !timeline_semaphore ||
      iree_hal_device_query_semaphore_compatibility(
          device, timeline_semaphore) != IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL ||
      !iree_status_is_ok(
          iree_hal_semaphore_query(timeline_semaphore, &semaphore_value)) ||
      semaphore_value != 3 ||
      !iree_status_is_ok(iree_hal_semaphore_signal(timeline_semaphore, 5,
                                                   /*frontier=*/nullptr)) ||
      !iree_status_is_ok(iree_hal_semaphore_wait(timeline_semaphore, 5,
                                                 iree_immediate_timeout(),
                                                 IREE_ASYNC_WAIT_FLAG_NONE)) ||
      !iree_status_is_ok(
          iree_hal_semaphore_query(timeline_semaphore, &semaphore_value)) ||
      semaphore_value != 5) {
    iree_status_ignore(status);
    if (timeline_semaphore) {
      iree_hal_semaphore_release(timeline_semaphore);
    }
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  iree_hal_semaphore_release(timeline_semaphore);

  iree_hal_semaphore_t *update_signal = nullptr;
  iree_hal_semaphore_t *copy_signal = nullptr;
  status =
      iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &update_signal);
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                  IREE_HAL_SEMAPHORE_FLAG_NONE, &copy_signal);
  }
  uint64_t update_signal_value = 1;
  uint64_t copy_signal_value = 2;
  iree_hal_semaphore_t *update_signal_ptrs[1] = {update_signal};
  iree_hal_semaphore_t *copy_signal_ptrs[1] = {copy_signal};
  iree_hal_semaphore_list_t update_signal_list = {
      1,
      update_signal_ptrs,
      &update_signal_value,
  };
  iree_hal_semaphore_list_t copy_wait_list = {
      1,
      update_signal_ptrs,
      &update_signal_value,
  };
  iree_hal_semaphore_list_t copy_signal_list = {
      1,
      copy_signal_ptrs,
      &copy_signal_value,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_update(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        update_signal_list, /*source_buffer=*/nullptr, /*source_offset=*/0,
        /*target_buffer=*/nullptr, /*target_offset=*/0, /*length=*/0,
        IREE_HAL_UPDATE_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_copy(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, copy_wait_list, copy_signal_list,
        /*source_buffer=*/nullptr, /*source_offset=*/0,
        /*target_buffer=*/nullptr, /*target_offset=*/0, /*length=*/0,
        IREE_HAL_COPY_FLAG_NONE);
  }
  uint64_t observed_copy_signal = 0;
  if (!iree_status_is_ok(status) ||
      !iree_status_is_ok(
          iree_hal_semaphore_query(copy_signal, &observed_copy_signal)) ||
      observed_copy_signal != copy_signal_value) {
    iree_status_ignore(status);
    if (copy_signal) {
      iree_hal_semaphore_release(copy_signal);
    }
    if (update_signal) {
      iree_hal_semaphore_release(update_signal);
    }
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  iree_hal_semaphore_release(copy_signal);
  iree_hal_semaphore_release(update_signal);

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

  iree_hal_buffer_params_t host_buffer_params = {};
  host_buffer_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
                             IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                             IREE_HAL_BUFFER_USAGE_MAPPING;
  host_buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  host_buffer_params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  compatibility = iree_hal_allocator_query_buffer_compatibility(
      allocator, host_buffer_params, 16, nullptr, nullptr);
  if ((compatibility & IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE) == 0 ||
      (compatibility & IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER) == 0) {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  iree_hal_buffer_t *mapped_buffer = nullptr;
  status = iree_hal_allocator_allocate_buffer(allocator, host_buffer_params, 16,
                                              &mapped_buffer);
  const uint32_t mapped_values[4] = {7, 11, 13, 17};
  uint32_t mapped_readback[4] = {};
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_write(mapped_buffer, 0, mapped_values,
                                       sizeof(mapped_values));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_read(mapped_buffer, 0, mapped_readback,
                                      sizeof(mapped_readback));
  }
  void *mapped_device_ptr = nullptr;
  uint32_t mapped_device_readback[4] = {};
  lr_device_t lrrt_device = {0};
  if (iree_status_is_ok(status)) {
    status = lrrt_iree_hal_buffer_device_pointer_for_test(mapped_buffer,
                                                          &mapped_device_ptr);
  }
  if (iree_status_is_ok(status) &&
      (lr_device_open(0, &lrrt_device) != LR_SUCCESS ||
       lr_memcpy(lrrt_device, mapped_device_readback, mapped_device_ptr,
                 sizeof(mapped_device_readback),
                 LR_MEMCPY_DEVICE_TO_HOST) != LR_SUCCESS)) {
    status =
        iree_make_status(IREE_STATUS_INTERNAL, "mapped device readback failed");
  }
  if (!iree_status_is_ok(status) ||
      std::memcmp(mapped_readback, mapped_values, sizeof(mapped_values)) != 0 ||
      std::memcmp(mapped_device_readback, mapped_values,
                  sizeof(mapped_values)) != 0) {
    iree_status_ignore(status);
    if (mapped_buffer) {
      iree_hal_buffer_release(mapped_buffer);
    }
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  const uint32_t mapped_updated_values[4] = {19, 23, 29, 31};
  uint32_t mapped_updated_readback[4] = {};
  status = iree_hal_device_queue_update(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      iree_hal_semaphore_list_empty(), mapped_updated_values, 0, mapped_buffer,
      0, sizeof(mapped_updated_values), IREE_HAL_UPDATE_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_read(mapped_buffer, 0, mapped_updated_readback,
                                      sizeof(mapped_updated_readback));
  }
  if (!iree_status_is_ok(status) ||
      std::memcmp(mapped_updated_readback, mapped_updated_values,
                  sizeof(mapped_updated_values)) != 0) {
    iree_status_ignore(status);
    if (mapped_buffer) {
      iree_hal_buffer_release(mapped_buffer);
    }
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  iree_hal_buffer_release(mapped_buffer);

  iree_hal_semaphore_t *alloc_signal = nullptr;
  iree_hal_semaphore_t *dealloc_signal = nullptr;
  status =
      iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &alloc_signal);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                       IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &dealloc_signal);
  }
  uint64_t alloc_signal_value = 7;
  uint64_t dealloc_signal_value = 8;
  iree_hal_semaphore_t *alloc_signal_ptrs[1] = {alloc_signal};
  iree_hal_semaphore_t *dealloc_signal_ptrs[1] = {dealloc_signal};
  iree_hal_semaphore_list_t alloc_signal_list = {
      1,
      alloc_signal_ptrs,
      &alloc_signal_value,
  };
  iree_hal_semaphore_list_t dealloc_wait_list = {
      1,
      alloc_signal_ptrs,
      &alloc_signal_value,
  };
  iree_hal_semaphore_list_t dealloc_signal_list = {
      1,
      dealloc_signal_ptrs,
      &dealloc_signal_value,
  };
  iree_hal_buffer_t *queue_buffer = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_alloca(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        alloc_signal_list, /*pool=*/nullptr, host_buffer_params, 16,
        (iree_hal_alloca_flags_t)0, &queue_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_dealloca(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, dealloc_wait_list,
        dealloc_signal_list, queue_buffer, (iree_hal_dealloca_flags_t)0);
  }
  uint64_t observed_dealloc_signal = 0;
  if (!iree_status_is_ok(status) || !queue_buffer ||
      !iree_status_is_ok(
          iree_hal_semaphore_query(dealloc_signal, &observed_dealloc_signal)) ||
      observed_dealloc_signal != dealloc_signal_value) {
    iree_status_ignore(status);
    if (queue_buffer) {
      iree_hal_buffer_release(queue_buffer);
    }
    if (dealloc_signal) {
      iree_hal_semaphore_release(dealloc_signal);
    }
    if (alloc_signal) {
      iree_hal_semaphore_release(alloc_signal);
    }
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  iree_hal_buffer_release(queue_buffer);
  iree_hal_semaphore_release(dealloc_signal);
  iree_hal_semaphore_release(alloc_signal);

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
  if (lr_memcpy(lrrt_device, copied_values, target_device_ptr,
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

  iree_hal_buffer_t *file_transfer_buffer = nullptr;
  status = iree_hal_allocator_allocate_buffer(allocator, buffer_params, 16,
                                              &file_transfer_buffer);
  if (!iree_status_is_ok(status) || !file_transfer_buffer) {
    iree_status_ignore(status);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  uint32_t file_source_values[4] = {8, 6, 7, 5};
  iree_io_file_handle_t *source_handle = nullptr;
  status = iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span(reinterpret_cast<uint8_t *>(file_source_values),
                          sizeof(file_source_values)),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &source_handle);
  iree_hal_file_t *source_file = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_file_import(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_READ,
        source_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &source_file);
  }
  if (source_handle) {
    iree_io_file_handle_release(source_handle);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_read(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        iree_hal_semaphore_list_empty(), source_file, /*source_offset=*/0,
        file_transfer_buffer, /*target_offset=*/0, sizeof(file_source_values),
        IREE_HAL_READ_FLAG_NONE);
  }
  void *file_transfer_device_ptr = nullptr;
  status = iree_status_join(
      status, lrrt_iree_hal_buffer_device_pointer_for_test(
                  file_transfer_buffer, &file_transfer_device_ptr));
  uint32_t file_readback_values[4] = {};
  if (iree_status_is_ok(status)) {
    if (lr_memcpy(lrrt_device, file_readback_values, file_transfer_device_ptr,
                  sizeof(file_readback_values),
                  LR_MEMCPY_DEVICE_TO_HOST) != LR_SUCCESS) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "queue read device readback failed");
    }
  }
  if (!iree_status_is_ok(status) || !source_file ||
      std::memcmp(file_readback_values, file_source_values,
                  sizeof(file_source_values)) != 0) {
    iree_status_ignore(status);
    if (source_file) {
      iree_hal_file_release(source_file);
    }
    iree_hal_buffer_release(file_transfer_buffer);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  iree_hal_file_release(source_file);

  uint32_t file_target_values[4] = {};
  iree_io_file_handle_t *target_handle = nullptr;
  status = iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(reinterpret_cast<uint8_t *>(file_target_values),
                          sizeof(file_target_values)),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &target_handle);
  iree_hal_file_t *target_file = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_file_import(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_WRITE,
        target_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &target_file);
  }
  if (target_handle) {
    iree_io_file_handle_release(target_handle);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_write(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        iree_hal_semaphore_list_empty(), file_transfer_buffer,
        /*source_offset=*/0, target_file, /*target_offset=*/0,
        sizeof(file_target_values), IREE_HAL_WRITE_FLAG_NONE);
  }
  if (!iree_status_is_ok(status) || !target_file ||
      std::memcmp(file_target_values, file_source_values,
                  sizeof(file_source_values)) != 0) {
    iree_status_ignore(status);
    if (target_file) {
      iree_hal_file_release(target_file);
    }
    iree_hal_buffer_release(file_transfer_buffer);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  iree_hal_file_release(target_file);
  iree_hal_buffer_release(file_transfer_buffer);

  iree_hal_buffer_t *cb_source_buffer = nullptr;
  iree_hal_buffer_t *cb_target_buffer = nullptr;
  status = iree_hal_allocator_allocate_buffer(allocator, buffer_params, 16,
                                              &cb_source_buffer);
  if (!iree_status_is_ok(status) || !cb_source_buffer) {
    iree_status_ignore(status);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  status = iree_hal_allocator_allocate_buffer(allocator, buffer_params, 16,
                                              &cb_target_buffer);
  if (!iree_status_is_ok(status) || !cb_target_buffer) {
    iree_status_ignore(status);
    iree_hal_buffer_release(cb_source_buffer);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

  const uint32_t cb_source_values[2] = {0x11111111u, 0x22222222u};
  const uint32_t cb_fill_pattern = 0xA5A5A5A5u;
  iree_hal_command_buffer_t *transfer_command_buffer = nullptr;
  status = iree_hal_command_buffer_create(
      device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/2, &transfer_command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(transfer_command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_update_buffer(
        transfer_command_buffer, cb_source_values, /*source_offset=*/0,
        iree_hal_make_buffer_ref(cb_source_buffer, 0, sizeof(cb_source_values)),
        IREE_HAL_UPDATE_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_fill_buffer(
        transfer_command_buffer,
        iree_hal_make_buffer_ref(cb_target_buffer, 0, 16), &cb_fill_pattern,
        sizeof(cb_fill_pattern), IREE_HAL_FILL_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_copy_buffer(
        transfer_command_buffer,
        iree_hal_make_indirect_buffer_ref(0, 0, sizeof(cb_source_values)),
        iree_hal_make_indirect_buffer_ref(1, 0, sizeof(cb_source_values)),
        IREE_HAL_COPY_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(transfer_command_buffer);
  }
  iree_hal_buffer_binding_t transfer_bindings[2] = {
      {cb_source_buffer, 0, 16},
      {cb_target_buffer, 0, 16},
  };
  const iree_hal_buffer_binding_table_t transfer_binding_table = {
      2,
      transfer_bindings,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_execute(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        iree_hal_semaphore_list_empty(), transfer_command_buffer,
        transfer_binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
  }
  void *cb_target_device_ptr = nullptr;
  status =
      iree_status_join(status, lrrt_iree_hal_buffer_device_pointer_for_test(
                                   cb_target_buffer, &cb_target_device_ptr));
  uint32_t cb_actual_values[4] = {};
  const uint32_t cb_expected_values[4] = {
      cb_source_values[0],
      cb_source_values[1],
      cb_fill_pattern,
      cb_fill_pattern,
  };
  if (!iree_status_is_ok(status) || !cb_target_device_ptr ||
      lr_memcpy(lrrt_device, cb_actual_values, cb_target_device_ptr,
                sizeof(cb_actual_values),
                LR_MEMCPY_DEVICE_TO_HOST) != LR_SUCCESS ||
      std::memcmp(cb_actual_values, cb_expected_values,
                  sizeof(cb_expected_values)) != 0) {
    iree_status_ignore(status);
    if (transfer_command_buffer) {
      iree_hal_command_buffer_release(transfer_command_buffer);
    }
    iree_hal_buffer_release(cb_target_buffer);
    iree_hal_buffer_release(cb_source_buffer);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  iree_hal_command_buffer_release(transfer_command_buffer);
  iree_hal_buffer_release(cb_target_buffer);
  iree_hal_buffer_release(cb_source_buffer);

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
  if (!iree_status_is_ok(status) || value != 0) {
    iree_status_ignore(status);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }

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
