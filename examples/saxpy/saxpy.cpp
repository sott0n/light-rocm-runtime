#include "../common/example_utils.h"
#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdio.h>

#ifndef LRRT_SAXPY_HSACO
#define LRRT_SAXPY_HSACO "saxpy_kernel.hsaco"
#endif

typedef struct saxpy_args_t {
  float alpha;
  const float *x;
  float *y;
  int n;
} saxpy_args_t;

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
    const float alpha = 2.5f;
    float x[n];
    float y[n];
    for (int i = 0; i < n; ++i) {
      x[i] = (float)i;
      y[i] = (float)(i * 3);
    }

    lrrt::DeviceBuffer device_x(device, sizeof(x));
    lrrt::DeviceBuffer device_y(device, sizeof(y));
    lrrt::copy_to_device(device_x, x);
    lrrt::copy_to_device(device_y, y);

    std::vector<unsigned char> hsaco =
        lrrt_example::read_file(LRRT_SAXPY_HSACO);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel("saxpy");

    saxpy_args_t args = {
        alpha,
        (const float *)device_x.data(),
        (float *)device_y.data(),
        n,
    };
    lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
    lrrt::launch(kernel, config, args);
    lrrt::copy_to_host(y, device_y);

    for (int i = 0; i < n; ++i) {
      float expected = alpha * x[i] + (float)(i * 3);
      if (fabsf(y[i] - expected) > 0.001f) {
        throw std::runtime_error("saxpy result mismatch");
      }
    }

    printf("saxpy: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
