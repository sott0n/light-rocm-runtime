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
    printf("allocation_registry: skipped, no GPU devices\n");
    lr_shutdown();
    return 0;
  }

  lr_device_t device = {0};
  status = lr_device_open(0, &device);
  if (!expect_status(status, LR_SUCCESS, "lr_device_open")) {
    lr_shutdown();
    return 1;
  }

  float host[8] = {};
  float out[4] = {};
  for (int i = 0; i < 8; ++i) {
    host[i] = (float)i + 0.25f;
  }

  void *device_buffer = NULL;
  status = lr_malloc(device, sizeof(host), &device_buffer);
  if (!expect_status(status, LR_SUCCESS, "lr_malloc")) {
    lr_shutdown();
    return 1;
  }

  status = lr_free(device, host);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT, "host pointer free")) {
    lr_free(device, device_buffer);
    lr_shutdown();
    return 1;
  }

  status = lr_free(device, static_cast<float *>(device_buffer) + 1);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "allocation subpointer free")) {
    lr_free(device, device_buffer);
    lr_shutdown();
    return 1;
  }

  if (count > 1) {
    lr_device_t other_device = {1};
    status = lr_device_open(1, &other_device);
    if (!expect_status(status, LR_SUCCESS, "lr_device_open second device")) {
      lr_free(device, device_buffer);
      lr_shutdown();
      return 1;
    }

    status = lr_free(other_device, device_buffer);
    if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                       "wrong device free")) {
      lr_free(device, device_buffer);
      lr_shutdown();
      return 1;
    }
  }

  status = lr_memcpy(device, device_buffer, host, sizeof(host),
                     LR_MEMCPY_HOST_TO_DEVICE);
  if (!expect_status(status, LR_SUCCESS, "host-to-device full copy")) {
    lr_free(device, device_buffer);
    lr_shutdown();
    return 1;
  }

  status = lr_memcpy(device, out, static_cast<float *>(device_buffer) + 2,
                     sizeof(out), LR_MEMCPY_DEVICE_TO_HOST);
  if (!expect_status(status, LR_SUCCESS, "device-to-host subrange copy")) {
    lr_free(device, device_buffer);
    lr_shutdown();
    return 1;
  }
  for (int i = 0; i < 4; ++i) {
    if (fabsf(out[i] - host[i + 2]) > 0.001f) {
      fprintf(stderr, "subrange copy mismatch at %d: got %f expected %f\n", i,
              out[i], host[i + 2]);
      lr_free(device, device_buffer);
      lr_shutdown();
      return 1;
    }
  }

  status = lr_memcpy(device, out, static_cast<float *>(device_buffer) + 6,
                     sizeof(out), LR_MEMCPY_DEVICE_TO_HOST);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "device-to-host out-of-bounds subrange copy")) {
    lr_free(device, device_buffer);
    lr_shutdown();
    return 1;
  }

  status = lr_memcpy(device, static_cast<float *>(device_buffer) + 6, host,
                     sizeof(out), LR_MEMCPY_HOST_TO_DEVICE);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "host-to-device out-of-bounds subrange copy")) {
    lr_free(device, device_buffer);
    lr_shutdown();
    return 1;
  }

  status = lr_free(device, device_buffer);
  if (!expect_status(status, LR_SUCCESS, "lr_free")) {
    lr_shutdown();
    return 1;
  }

  lr_shutdown();
  printf("allocation_registry: ok\n");
  return 0;
}
