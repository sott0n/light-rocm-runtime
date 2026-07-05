#include "lrrt/lrrt.h"

#include <stdio.h>

static int expect_status(lr_status_t actual, lr_status_t expected,
                         const char *operation) {
  if (actual == expected) {
    return 1;
  }
  fprintf(stderr, "%s returned %s, expected %s\n", operation,
          lr_status_string(actual), lr_status_string(expected));
  return 0;
}

int main(void) {
  uint32_t count = 123;
  lr_status_t status = lr_device_count(&count);
  if (!expect_status(status, LR_ERROR_NOT_INITIALIZED,
                     "lr_device_count before init")) {
    return 1;
  }

  status = lr_shutdown();
  if (!expect_status(status, LR_ERROR_NOT_INITIALIZED,
                     "lr_shutdown before init")) {
    return 1;
  }

  status = lr_init();
  if (!expect_status(status, LR_SUCCESS, "lr_init")) {
    return 1;
  }

  status = lr_init();
  if (!expect_status(status, LR_ERROR_ALREADY_INITIALIZED, "double lr_init")) {
    lr_shutdown();
    return 1;
  }

  status = lr_device_count(NULL);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "lr_device_count null")) {
    lr_shutdown();
    return 1;
  }

  status = lr_device_count(&count);
  if (!expect_status(status, LR_SUCCESS, "lr_device_count")) {
    lr_shutdown();
    return 1;
  }

  status = lr_device_open(0, NULL);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "lr_device_open null")) {
    lr_shutdown();
    return 1;
  }

  lr_device_t invalid_device = {count};
  lr_memory_stats_t stats = {};
  status = lr_get_memory_stats(invalid_device, &stats);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "lr_get_memory_stats invalid device")) {
    lr_shutdown();
    return 1;
  }

  if (count == 0) {
    printf("runtime_api: skipped GPU handle checks, no GPU devices\n");
    lr_shutdown();
    return 0;
  }

  lr_device_t device = {0};
  status = lr_device_open(0, &device);
  if (!expect_status(status, LR_SUCCESS, "lr_device_open")) {
    lr_shutdown();
    return 1;
  }

  status = lr_device_open(count, &device);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "lr_device_open out of range")) {
    lr_shutdown();
    return 1;
  }

  char name[1] = {};
  status = lr_device_name(device, name, sizeof(name));
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "lr_device_name too small")) {
    lr_shutdown();
    return 1;
  }

  status = lr_get_memory_stats(device, NULL);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "lr_get_memory_stats null")) {
    lr_shutdown();
    return 1;
  }

  status = lr_reset_memory_stats(invalid_device);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "lr_reset_memory_stats invalid device")) {
    lr_shutdown();
    return 1;
  }

  status = lr_queue_create(device, NULL);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "lr_queue_create null")) {
    lr_shutdown();
    return 1;
  }

  status = lr_queue_synchronize(NULL);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "lr_queue_synchronize null")) {
    lr_shutdown();
    return 1;
  }

  status = lr_queue_destroy(NULL);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "lr_queue_destroy null")) {
    lr_shutdown();
    return 1;
  }

  lr_queue_t *queue = NULL;
  status = lr_queue_create(device, &queue);
  if (!expect_status(status, LR_SUCCESS, "lr_queue_create")) {
    lr_shutdown();
    return 1;
  }

  status = lr_queue_destroy(queue);
  if (!expect_status(status, LR_SUCCESS, "lr_queue_destroy")) {
    lr_shutdown();
    return 1;
  }

  status = lr_queue_synchronize(queue);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "stale queue synchronize")) {
    lr_shutdown();
    return 1;
  }

  status = lr_queue_destroy(queue);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "stale queue destroy")) {
    lr_shutdown();
    return 1;
  }

  status = lr_event_create(device, NULL);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "lr_event_create null")) {
    lr_shutdown();
    return 1;
  }

  status = lr_event_synchronize(NULL);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "lr_event_synchronize null")) {
    lr_shutdown();
    return 1;
  }

  status = lr_event_destroy(NULL);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "lr_event_destroy null")) {
    lr_shutdown();
    return 1;
  }

  lr_event_t *event = NULL;
  status = lr_event_create(device, &event);
  if (!expect_status(status, LR_SUCCESS, "lr_event_create")) {
    lr_shutdown();
    return 1;
  }

  status = lr_event_synchronize(event);
  if (!expect_status(status, LR_SUCCESS, "unrecorded event synchronize")) {
    lr_event_destroy(event);
    lr_shutdown();
    return 1;
  }

  uint64_t elapsed_ns = 1;
  status = lr_event_elapsed_time_ns(event, event, &elapsed_ns);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "unrecorded event elapsed time")) {
    lr_event_destroy(event);
    lr_shutdown();
    return 1;
  }

  status = lr_event_destroy(event);
  if (!expect_status(status, LR_SUCCESS, "lr_event_destroy")) {
    lr_shutdown();
    return 1;
  }

  status = lr_event_synchronize(event);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "stale event synchronize")) {
    lr_shutdown();
    return 1;
  }

  status = lr_event_record(event);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT, "stale event record")) {
    lr_shutdown();
    return 1;
  }

  status = lr_event_destroy(event);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "stale event destroy")) {
    lr_shutdown();
    return 1;
  }

  lr_event_t *dependencies[1] = {NULL};
  status = lr_memcpy_async_with_dependencies(
      device, &stats, &stats, sizeof(stats), LR_MEMCPY_DEVICE_TO_DEVICE, event,
      dependencies, 1);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "async copy null dependency")) {
    lr_shutdown();
    return 1;
  }

  status = lr_synchronize(invalid_device);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "lr_synchronize invalid device")) {
    lr_shutdown();
    return 1;
  }

  status = lr_shutdown();
  if (!expect_status(status, LR_SUCCESS, "lr_shutdown")) {
    return 1;
  }

  status = lr_device_count(&count);
  if (!expect_status(status, LR_ERROR_NOT_INITIALIZED,
                     "lr_device_count after shutdown")) {
    return 1;
  }

  printf("runtime_api: ok\n");
  return 0;
}
