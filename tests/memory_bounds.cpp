#include "lrrt/lrrt.h"

#include <math.h>
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
  for (int i = 0; i < 64; ++i) {
    host[i] = (float)i;
  }
  void *device_buffer = NULL;
  status = lr_malloc(device, sizeof(host), &device_buffer);
  if (!expect_status(status, LR_SUCCESS, "lr_malloc")) {
    lr_shutdown();
    return 1;
  }

  status = lr_memcpy(device, device_buffer, host, sizeof(host) + sizeof(float),
                     LR_MEMCPY_HOST_TO_DEVICE);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "oversized host-to-device copy")) {
    lr_free(device, device_buffer);
    lr_shutdown();
    return 1;
  }

  void *device_copy = NULL;
  status = lr_malloc(device, sizeof(host), &device_copy);
  if (!expect_status(status, LR_SUCCESS, "lr_malloc copy")) {
    lr_free(device, device_buffer);
    lr_shutdown();
    return 1;
  }

  status = lr_memcpy(device, device_buffer, host, sizeof(host),
                     LR_MEMCPY_HOST_TO_DEVICE);
  if (!expect_status(status, LR_SUCCESS, "host-to-device copy")) {
    lr_free(device, device_copy);
    lr_free(device, device_buffer);
    lr_shutdown();
    return 1;
  }

  float partial[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
  status = lr_memcpy(device, device_copy, partial, sizeof(partial),
                     LR_MEMCPY_HOST_TO_DEVICE);
  if (!expect_status(status, LR_SUCCESS, "partial host-to-device copy")) {
    lr_free(device, device_copy);
    lr_free(device, device_buffer);
    lr_shutdown();
    return 1;
  }

  status = lr_memcpy(device, static_cast<float *>(device_copy) + 1,
                     static_cast<float *>(device_buffer) + 2, 2 * sizeof(float),
                     LR_MEMCPY_DEVICE_TO_DEVICE);
  if (!expect_status(status, LR_SUCCESS, "subrange device-to-device copy")) {
    lr_free(device, device_copy);
    lr_free(device, device_buffer);
    lr_shutdown();
    return 1;
  }

  status = lr_memcpy(device, partial, device_copy, sizeof(partial),
                     LR_MEMCPY_DEVICE_TO_HOST);
  if (!expect_status(status, LR_SUCCESS, "partial device-to-host copy")) {
    lr_free(device, device_copy);
    lr_free(device, device_buffer);
    lr_shutdown();
    return 1;
  }
  if (fabsf(partial[0] + 1.0f) > 0.001f ||
      fabsf(partial[1] - host[2]) > 0.001f ||
      fabsf(partial[2] - host[3]) > 0.001f ||
      fabsf(partial[3] + 1.0f) > 0.001f) {
    fprintf(stderr, "subrange device copy mismatch\n");
    lr_free(device, device_copy);
    lr_free(device, device_buffer);
    lr_shutdown();
    return 1;
  }

  lr_free(device, device_copy);
  lr_free(device, device_buffer);
  lr_shutdown();

  printf("memory_bounds: ok\n");
  return 0;
}
