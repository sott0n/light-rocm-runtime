#include "lrrt/lrrt.h"

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
  }

  lr_shutdown();
  return 0;
}
