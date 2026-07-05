#include "lrrt/lrrt.h"

#include <stdio.h>
#include <string.h>

static int expect_status(lr_status_t actual, lr_status_t expected,
                         const char *operation) {
  if (actual == expected) {
    return 1;
  }
  fprintf(stderr, "%s returned %s, expected %s\n", operation,
          lr_status_string(actual), lr_status_string(expected));
  return 0;
}

static int expect_stats_equal(const lr_memory_stats_t *actual,
                              const lr_memory_stats_t *expected,
                              const char *operation) {
  if (memcmp(actual, expected, sizeof(*actual)) == 0) {
    return 1;
  }
  fprintf(stderr,
          "%s changed stats unexpectedly:\n"
          "  live: %llu -> %llu\n"
          "  peak: %llu -> %llu\n"
          "  allocated: %llu -> %llu\n"
          "  freed: %llu -> %llu\n"
          "  allocations: %llu -> %llu\n"
          "  frees: %llu -> %llu\n"
          "  h2d: %llu -> %llu\n"
          "  d2h: %llu -> %llu\n"
          "  d2d: %llu -> %llu\n"
          "  memcpy_count: %llu -> %llu\n",
          operation, (unsigned long long)expected->live_bytes,
          (unsigned long long)actual->live_bytes,
          (unsigned long long)expected->peak_live_bytes,
          (unsigned long long)actual->peak_live_bytes,
          (unsigned long long)expected->total_allocated_bytes,
          (unsigned long long)actual->total_allocated_bytes,
          (unsigned long long)expected->total_freed_bytes,
          (unsigned long long)actual->total_freed_bytes,
          (unsigned long long)expected->allocation_count,
          (unsigned long long)actual->allocation_count,
          (unsigned long long)expected->free_count,
          (unsigned long long)actual->free_count,
          (unsigned long long)expected->h2d_copy_bytes,
          (unsigned long long)actual->h2d_copy_bytes,
          (unsigned long long)expected->d2h_copy_bytes,
          (unsigned long long)actual->d2h_copy_bytes,
          (unsigned long long)expected->d2d_copy_bytes,
          (unsigned long long)actual->d2d_copy_bytes,
          (unsigned long long)expected->memcpy_count,
          (unsigned long long)actual->memcpy_count);
  return 0;
}

int main(void) {
  lr_status_t status = lr_init();
  if (!expect_status(status, LR_SUCCESS, "lr_init")) {
    return 1;
  }

  uint32_t count = 0;
  status = lr_device_count(&count);
  if (!expect_status(status, LR_SUCCESS, "lr_device_count")) {
    lr_shutdown();
    return 1;
  }
  if (count == 0) {
    printf("memory_stats: skipped, no GPU devices\n");
    lr_shutdown();
    return 0;
  }

  lr_device_t device = {0};
  status = lr_device_open(0, &device);
  if (!expect_status(status, LR_SUCCESS, "lr_device_open")) {
    lr_shutdown();
    return 1;
  }

  status = lr_reset_memory_stats(device);
  if (!expect_status(status, LR_SUCCESS, "lr_reset_memory_stats initial")) {
    lr_shutdown();
    return 1;
  }

  lr_memory_stats_t stats = {};
  status = lr_get_memory_stats(device, &stats);
  if (!expect_status(status, LR_SUCCESS, "lr_get_memory_stats initial")) {
    lr_shutdown();
    return 1;
  }
  if (stats.live_bytes != 0 || stats.peak_live_bytes != 0 ||
      stats.total_allocated_bytes != 0 || stats.total_freed_bytes != 0 ||
      stats.allocation_count != 0 || stats.free_count != 0 ||
      stats.h2d_copy_bytes != 0 || stats.d2h_copy_bytes != 0 ||
      stats.d2d_copy_bytes != 0 || stats.memcpy_count != 0) {
    fprintf(stderr, "initial memory stats were not reset\n");
    lr_shutdown();
    return 1;
  }

  float host[64] = {};
  float out[64] = {};
  void *device_a = NULL;
  void *device_b = NULL;
  status = lr_malloc(device, sizeof(host), &device_a);
  if (!expect_status(status, LR_SUCCESS, "lr_malloc a")) {
    lr_shutdown();
    return 1;
  }
  status = lr_malloc(device, sizeof(host), &device_b);
  if (!expect_status(status, LR_SUCCESS, "lr_malloc b")) {
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }

  status =
      lr_memcpy(device, device_a, host, sizeof(host), LR_MEMCPY_HOST_TO_DEVICE);
  if (!expect_status(status, LR_SUCCESS, "host-to-device copy")) {
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }

  status = lr_get_memory_stats(device, &stats);
  if (!expect_status(status, LR_SUCCESS, "lr_get_memory_stats after setup")) {
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }
  if (stats.live_bytes != sizeof(host) * 2 ||
      stats.peak_live_bytes != sizeof(host) * 2 ||
      stats.total_allocated_bytes != sizeof(host) * 2 ||
      stats.total_freed_bytes != 0 || stats.allocation_count != 2 ||
      stats.free_count != 0 || stats.h2d_copy_bytes != sizeof(host) ||
      stats.d2h_copy_bytes != 0 || stats.d2d_copy_bytes != 0 ||
      stats.memcpy_count != 1) {
    fprintf(stderr, "setup memory stats mismatch\n");
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }

  lr_memory_stats_t before_failure = stats;
  status = lr_memcpy(device, device_a, host, sizeof(host) + sizeof(float),
                     LR_MEMCPY_HOST_TO_DEVICE);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "oversized failed copy")) {
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }
  status = lr_get_memory_stats(device, &stats);
  if (!expect_status(status, LR_SUCCESS,
                     "lr_get_memory_stats after failed copy")) {
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }
  if (!expect_stats_equal(&stats, &before_failure, "failed copy")) {
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }

  status = lr_free(device, static_cast<char *>(device_a) + sizeof(float));
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "failed subpointer free")) {
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }
  status = lr_get_memory_stats(device, &stats);
  if (!expect_status(status, LR_SUCCESS,
                     "lr_get_memory_stats after failed free")) {
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }
  if (!expect_stats_equal(&stats, &before_failure, "failed free")) {
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }

  status = lr_reset_memory_stats(device);
  if (!expect_status(status, LR_SUCCESS, "lr_reset_memory_stats live")) {
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }
  status = lr_get_memory_stats(device, &stats);
  if (!expect_status(status, LR_SUCCESS, "lr_get_memory_stats after reset")) {
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }
  if (stats.live_bytes != sizeof(host) * 2 ||
      stats.peak_live_bytes != sizeof(host) * 2 ||
      stats.total_allocated_bytes != 0 || stats.total_freed_bytes != 0 ||
      stats.allocation_count != 0 || stats.free_count != 0 ||
      stats.h2d_copy_bytes != 0 || stats.d2h_copy_bytes != 0 ||
      stats.d2d_copy_bytes != 0 || stats.memcpy_count != 0) {
    fprintf(stderr, "reset memory stats did not preserve live bytes only\n");
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }

  lr_event_t *event = NULL;
  status = lr_event_create(device, &event);
  if (!expect_status(status, LR_SUCCESS, "lr_event_create")) {
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }

  status = lr_memcpy_async(device, device_b, device_a, sizeof(host),
                           LR_MEMCPY_DEVICE_TO_DEVICE, event);
  if (!expect_status(status, LR_SUCCESS, "async device-to-device copy")) {
    lr_event_destroy(event);
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }
  status = lr_event_synchronize(event);
  if (!expect_status(status, LR_SUCCESS, "lr_event_synchronize")) {
    lr_event_destroy(event);
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }

  status =
      lr_memcpy(device, out, device_b, sizeof(out), LR_MEMCPY_DEVICE_TO_HOST);
  if (!expect_status(status, LR_SUCCESS, "device-to-host copy")) {
    lr_event_destroy(event);
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }

  status = lr_get_memory_stats(device, &stats);
  if (!expect_status(status, LR_SUCCESS,
                     "lr_get_memory_stats after async copy")) {
    lr_event_destroy(event);
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }
  if (stats.live_bytes != sizeof(host) * 2 ||
      stats.peak_live_bytes != sizeof(host) * 2 ||
      stats.total_allocated_bytes != 0 || stats.total_freed_bytes != 0 ||
      stats.allocation_count != 0 || stats.free_count != 0 ||
      stats.h2d_copy_bytes != 0 || stats.d2h_copy_bytes != sizeof(out) ||
      stats.d2d_copy_bytes != sizeof(host) || stats.memcpy_count != 2) {
    fprintf(stderr, "async copy memory stats mismatch\n");
    lr_event_destroy(event);
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }

  lr_event_destroy(event);
  lr_free(device, device_b);
  lr_free(device, device_a);
  lr_shutdown();

  printf("memory_stats: ok\n");
  return 0;
}
