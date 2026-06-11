#include "../common/example_utils.h"

#include <math.h>
#include <stdio.h>

#ifndef LRRT_FILL_HSACO
#define LRRT_FILL_HSACO "fill_kernel.hsaco"
#endif

typedef struct fill_args_t {
  float *out;
  float value;
  int n;
} fill_args_t;

int main(void) {
  try {
    lrrt_example::Runtime runtime;

    uint32_t count = 0;
    lrrt_example::check(lr_device_count(&count), "lr_device_count");
    printf("devices: %u\n", count);
    if (count == 0) {
      return 0;
    }

    lr_device_t device = {0};
    lrrt_example::check(lr_device_open(0, &device), "lr_device_open");
    printf("opened device: %u\n", device.index);

    const int n = 64;
    const float value = 7.25f;
    float out[n];

    lrrt_example::DeviceBuffer device_out(device, sizeof(out));
    std::vector<unsigned char> hsaco =
        lrrt_example::read_file(LRRT_FILL_HSACO);
    lrrt_example::Module module(device, hsaco);
    lr_kernel_t *kernel = module.kernel("fill");

    fill_args_t args = {(float *)device_out.data(), value, n};
    lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
    lrrt_example::check(lr_launch(kernel, &config, &args, sizeof(args)),
                        "lr_launch");
    lrrt_example::check(lr_memcpy(device, out, device_out.data(), sizeof(out),
                                  LR_MEMCPY_DEVICE_TO_HOST),
                        "copy out");

    for (int i = 0; i < n; ++i) {
      if (fabsf(out[i] - value) > 0.001f) {
        throw std::runtime_error("fill result mismatch");
      }
    }

    printf("fill: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
