#include "../common/example_utils.h"
#include "lrrt/lrrt.hpp"

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
    lrrt::Runtime runtime;

    uint32_t count = runtime.device_count();
    printf("devices: %u\n", count);
    if (count == 0) {
      return 0;
    }

    lrrt::Device device = runtime.open_device(0);
    printf("opened device: %u\n", device.index());

    const int n = 64;
    const float value = 7.25f;
    float out[n];

    lrrt::DeviceBuffer device_out(device, sizeof(out));
    std::vector<unsigned char> hsaco = lrrt_example::read_file(LRRT_FILL_HSACO);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel("fill");

    fill_args_t args = {(float *)device_out.data(), value, n};
    lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
    lrrt::launch(kernel, config, args);
    lrrt::copy_to_host(out, device_out, sizeof(out));

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
