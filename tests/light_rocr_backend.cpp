#include "lrrt/lrrt.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

bool expect_status(lr_status_t actual, lr_status_t expected,
                   const char *operation) {
  if (actual == expected) {
    return true;
  }
  std::fprintf(stderr, "%s returned %s, expected %s\n", operation,
               lr_status_string(actual), lr_status_string(expected));
  return false;
}

bool expect_empty_stats(const lr_memory_stats_t &stats, const char *operation) {
  const lr_memory_stats_t empty{};
  if (std::memcmp(&stats, &empty, sizeof(stats)) == 0) {
    return true;
  }
  std::fprintf(stderr, "%s returned non-empty memory statistics\n", operation);
  return false;
}

bool run_allocation_checks(lr_device_t device) {
  lr_memory_stats_t stats{};
  if (!expect_status(lr_get_memory_stats(device, &stats), LR_SUCCESS,
                     "lr_get_memory_stats initial") ||
      !expect_empty_stats(stats, "lr_get_memory_stats initial")) {
    return false;
  }

  void *allocation = nullptr;
  if (!expect_status(lr_malloc(device, 1, &allocation), LR_SUCCESS,
                     "lr_malloc one byte") ||
      allocation == nullptr) {
    std::fprintf(stderr, "lr_malloc returned a null allocation\n");
    return false;
  }

  if (!expect_status(lr_get_memory_stats(device, &stats), LR_SUCCESS,
                     "lr_get_memory_stats after allocation") ||
      stats.live_bytes != 1 || stats.peak_live_bytes != 1 ||
      stats.total_allocated_bytes != 1 || stats.allocation_count != 1) {
    std::fprintf(stderr, "allocation statistics did not use requested size\n");
    (void)lr_free(device, allocation);
    return false;
  }

  if (!expect_status(lr_reset_memory_stats(device), LR_SUCCESS,
                     "lr_reset_memory_stats") ||
      !expect_status(lr_get_memory_stats(device, &stats), LR_SUCCESS,
                     "lr_get_memory_stats after reset") ||
      stats.live_bytes != 1 || stats.peak_live_bytes != 1 ||
      stats.total_allocated_bytes != 0 || stats.allocation_count != 0) {
    std::fprintf(stderr,
                 "memory statistics reset lost live allocation state\n");
    (void)lr_free(device, allocation);
    return false;
  }

  void *subpointer = reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(allocation) + uintptr_t{1});
  if (!expect_status(lr_free(device, subpointer), LR_ERROR_INVALID_ARGUMENT,
                     "lr_free subpointer")) {
    (void)lr_free(device, allocation);
    return false;
  }

  uint8_t host_byte = 0x5a;
  if (!expect_status(lr_memcpy(device, allocation, &host_byte, 1,
                               LR_MEMCPY_HOST_TO_DEVICE),
                     LR_SUCCESS, "lr_memcpy host to device")) {
    (void)lr_free(device, allocation);
    return false;
  }
  host_byte = 0;
  if (!expect_status(lr_memcpy(device, &host_byte, allocation, 1,
                               LR_MEMCPY_DEVICE_TO_HOST),
                     LR_SUCCESS, "lr_memcpy device to host") ||
      host_byte != 0x5a) {
    std::fprintf(stderr, "lr_memcpy round trip did not preserve data\n");
    (void)lr_free(device, allocation);
    return false;
  }
  if (!expect_status(lr_get_memory_stats(device, &stats), LR_SUCCESS,
                     "lr_get_memory_stats after copies") ||
      stats.h2d_copy_bytes != 1 || stats.d2h_copy_bytes != 1 ||
      stats.d2d_copy_bytes != 0 || stats.memcpy_count != 2) {
    std::fprintf(stderr, "copy statistics did not record successful copies\n");
    (void)lr_free(device, allocation);
    return false;
  }

  void *host_allocation = reinterpret_cast<void *>(uintptr_t{1});
  if (!expect_status(lr_host_malloc(device, 1, &host_allocation),
                     LR_ERROR_NOT_SUPPORTED, "lr_host_malloc unsupported") ||
      host_allocation != nullptr) {
    std::fprintf(stderr, "unsupported lr_host_malloc wrote an allocation\n");
    (void)lr_free(device, allocation);
    return false;
  }

  lr_queue_t *queue = reinterpret_cast<lr_queue_t *>(uintptr_t{1});
  if (!expect_status(lr_queue_create(device, &queue), LR_ERROR_NOT_SUPPORTED,
                     "lr_queue_create unsupported") ||
      queue != nullptr) {
    std::fprintf(stderr, "unsupported lr_queue_create wrote a queue\n");
    (void)lr_free(device, allocation);
    return false;
  }

  lr_event_t *event = reinterpret_cast<lr_event_t *>(uintptr_t{1});
  if (!expect_status(lr_event_create(device, &event), LR_ERROR_NOT_SUPPORTED,
                     "lr_event_create unsupported") ||
      event != nullptr) {
    std::fprintf(stderr, "unsupported lr_event_create wrote an event\n");
    (void)lr_free(device, allocation);
    return false;
  }

  lr_module_t *module = reinterpret_cast<lr_module_t *>(uintptr_t{1});
  if (!expect_status(lr_module_load_hsaco(device, &host_byte, 1, &module),
                     LR_ERROR_NOT_SUPPORTED,
                     "lr_module_load_hsaco unsupported") ||
      module != nullptr) {
    std::fprintf(stderr, "unsupported lr_module_load_hsaco wrote a module\n");
    (void)lr_free(device, allocation);
    return false;
  }

  if (!expect_status(lr_synchronize(device), LR_ERROR_NOT_SUPPORTED,
                     "lr_synchronize unsupported") ||
      !expect_status(lr_free(device, allocation), LR_SUCCESS,
                     "lr_free allocation") ||
      !expect_status(lr_free(device, allocation), LR_ERROR_INVALID_ARGUMENT,
                     "lr_free stale allocation") ||
      !expect_status(lr_memcpy(device, allocation, &host_byte, 1,
                               LR_MEMCPY_HOST_TO_DEVICE),
                     LR_ERROR_INVALID_ARGUMENT, "lr_memcpy stale allocation") ||
      !expect_status(lr_get_memory_stats(device, &stats), LR_SUCCESS,
                     "lr_get_memory_stats after free") ||
      stats.live_bytes != 0 || stats.total_freed_bytes != 1 ||
      stats.free_count != 1 || stats.h2d_copy_bytes != 1 ||
      stats.d2h_copy_bytes != 1 || stats.d2d_copy_bytes != 0 ||
      stats.memcpy_count != 2) {
    return false;
  }

  return true;
}

bool open_only_device(lr_device_t *device, char *name, size_t name_size) {
  uint32_t count = 0;
  if (!expect_status(lr_device_count(&count), LR_SUCCESS, "lr_device_count") ||
      count != 1) {
    std::fprintf(stderr, "light-rocr backend exposed %u devices, expected 1\n",
                 count);
    return false;
  }
  if (!expect_status(lr_device_open(count, device), LR_ERROR_INVALID_ARGUMENT,
                     "lr_device_open out of range") ||
      !expect_status(lr_device_open(0, device), LR_SUCCESS, "lr_device_open") ||
      !expect_status(lr_device_name(*device, name, name_size), LR_SUCCESS,
                     "lr_device_name") ||
      name[0] == '\0') {
    std::fprintf(stderr, "light-rocr backend returned an empty device name\n");
    return false;
  }
  return true;
}

} // namespace

int main() {
  uint32_t count = 0;
  if (!expect_status(lr_device_count(&count), LR_ERROR_NOT_INITIALIZED,
                     "lr_device_count before init") ||
      !expect_status(lr_init(), LR_SUCCESS, "lr_init") ||
      !expect_status(lr_init(), LR_ERROR_ALREADY_INITIALIZED,
                     "lr_init twice")) {
    return 1;
  }

  lr_device_t device{};
  char name[128] = {};
  if (!open_only_device(&device, name, sizeof(name)) ||
      !run_allocation_checks(device)) {
    (void)lr_shutdown();
    return 1;
  }

  void *leaked_allocation = nullptr;
  if (!expect_status(lr_malloc(device, 5000, &leaked_allocation), LR_SUCCESS,
                     "lr_malloc shutdown-owned allocation") ||
      !expect_status(lr_shutdown(), LR_SUCCESS,
                     "lr_shutdown with live allocation") ||
      !expect_status(lr_device_count(&count), LR_ERROR_NOT_INITIALIZED,
                     "lr_device_count after shutdown")) {
    return 1;
  }

  if (!expect_status(lr_init(), LR_SUCCESS, "lr_init second session") ||
      !open_only_device(&device, name, sizeof(name))) {
    (void)lr_shutdown();
    return 1;
  }
  void *allocation = nullptr;
  if (!expect_status(lr_malloc(device, 32, &allocation), LR_SUCCESS,
                     "lr_malloc second session") ||
      !expect_status(lr_free(device, allocation), LR_SUCCESS,
                     "lr_free second session") ||
      !expect_status(lr_shutdown(), LR_SUCCESS, "lr_shutdown second session") ||
      !expect_status(lr_shutdown(), LR_ERROR_NOT_INITIALIZED,
                     "lr_shutdown twice")) {
    return 1;
  }

  std::printf("light_rocr_backend: device=%s lifecycle=ok memory=ok\n", name);
  return 0;
}
