#include "lrrt/lrrt.h"

#include <string.h>
#include <stdio.h>

int main(void) {
  lr_status_t status = lr_init();
  if (status != LR_SUCCESS) {
    fprintf(stderr, "lr_init failed: %s\n", lr_status_string(status));
    return 1;
  }

  uint32_t count = 0;
  status = lr_device_count(&count);
  if (status != LR_SUCCESS) {
    fprintf(stderr, "lr_device_count failed: %s\n", lr_status_string(status));
    lr_shutdown();
    return 1;
  }

  printf("devices: %u\n", count);
  if (count > 0) {
    lr_device_t device = {0};
    status = lr_device_open(0, &device);
    if (status != LR_SUCCESS) {
      fprintf(stderr, "lr_device_open failed: %s\n", lr_status_string(status));
      lr_shutdown();
      return 1;
    }
    printf("opened device: %u\n", device.index);

    const float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    void *device_ptr = NULL;

    status = lr_malloc(device, sizeof(input), &device_ptr);
    if (status != LR_SUCCESS) {
      fprintf(stderr, "lr_malloc failed: %s\n", lr_status_string(status));
      lr_shutdown();
      return 1;
    }

    status = lr_memcpy(device, device_ptr, input, sizeof(input),
                       LR_MEMCPY_HOST_TO_DEVICE);
    if (status != LR_SUCCESS) {
      fprintf(stderr, "host-to-device copy failed: %s\n",
              lr_status_string(status));
      lr_free(device, device_ptr);
      lr_shutdown();
      return 1;
    }

    status = lr_memcpy(device, output, device_ptr, sizeof(output),
                       LR_MEMCPY_DEVICE_TO_HOST);
    if (status != LR_SUCCESS) {
      fprintf(stderr, "device-to-host copy failed: %s\n",
              lr_status_string(status));
      lr_free(device, device_ptr);
      lr_shutdown();
      return 1;
    }

    if (memcmp(input, output, sizeof(input)) != 0) {
      fprintf(stderr, "memory roundtrip mismatch\n");
      lr_free(device, device_ptr);
      lr_shutdown();
      return 1;
    }
    printf("memory roundtrip: ok\n");

    status = lr_free(device, device_ptr);
    if (status != LR_SUCCESS) {
      fprintf(stderr, "lr_free failed: %s\n", lr_status_string(status));
      lr_shutdown();
      return 1;
    }

    status = lr_synchronize(device);
    if (status != LR_SUCCESS) {
      fprintf(stderr, "lr_synchronize failed: %s\n", lr_status_string(status));
      lr_shutdown();
      return 1;
    }
  }

  lr_shutdown();
  return 0;
}
