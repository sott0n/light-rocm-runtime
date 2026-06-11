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
    printf("memory_bounds: skipped, no GPU devices\n");
    lr_shutdown();
    return 0;
  }

  lr_device_t device = {0};
  status = lr_device_open(0, &device);
  if (!expect_status(status, LR_SUCCESS, "lr_device_open")) {
    lr_shutdown();
    return 1;
  }

  float host[64] = {0};
  void *device_buffer = NULL;
  status = lr_malloc(device, sizeof(host), &device_buffer);
  if (!expect_status(status, LR_SUCCESS, "lr_malloc")) {
    lr_shutdown();
    return 1;
  }

  status = lr_memcpy(device, device_buffer, host,
                     sizeof(host) + sizeof(float), LR_MEMCPY_HOST_TO_DEVICE);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "oversized host-to-device copy")) {
    lr_free(device, device_buffer);
    lr_shutdown();
    return 1;
  }

  lr_free(device, device_buffer);
  lr_shutdown();

  printf("memory_bounds: ok\n");
  return 0;
}
