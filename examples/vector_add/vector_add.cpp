#include "../common/example_utils.h"
#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdio.h>

#ifndef LRRT_VECTOR_ADD_HSACO
#define LRRT_VECTOR_ADD_HSACO "vector_add_kernel.hsaco"
#endif

typedef struct vector_add_args_t {
  const float *a;
  const float *b;
  float *c;
  int n;
} vector_add_args_t;

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
    float a[n];
    float b[n];
    float c[n];
    for (int i = 0; i < n; ++i) {
      a[i] = (float)i;
      b[i] = (float)(i * 2);
      c[i] = 0.0f;
    }

    lrrt::DeviceBuffer device_a(device, sizeof(a));
    lrrt::DeviceBuffer device_b(device, sizeof(b));
    lrrt::DeviceBuffer device_c(device, sizeof(c));
    lrrt::copy_to_device(device_a, a);
    lrrt::copy_to_device(device_b, b);

    std::vector<unsigned char> hsaco =
        lrrt_example::read_file(LRRT_VECTOR_ADD_HSACO);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel("vector_add");

    vector_add_args_t args = {
        (const float *)device_a.data(),
        (const float *)device_b.data(),
        (float *)device_c.data(),
        n,
    };
    lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
    lrrt::launch(kernel, config, args);
    device.synchronize();
    lrrt::copy_to_host(c, device_c);

    for (int i = 0; i < n; ++i) {
      if (fabsf(c[i] - (a[i] + b[i])) > 0.001f) {
        throw std::runtime_error("vector_add result mismatch");
      }
    }

    printf("vector_add: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
