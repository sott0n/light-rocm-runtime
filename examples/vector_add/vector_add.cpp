#include "../common/example_utils.h"

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
    float a[n];
    float b[n];
    float c[n];
    for (int i = 0; i < n; ++i) {
      a[i] = (float)i;
      b[i] = (float)(i * 2);
      c[i] = 0.0f;
    }

    lrrt_example::DeviceBuffer device_a(device, sizeof(a));
    lrrt_example::DeviceBuffer device_b(device, sizeof(b));
    lrrt_example::DeviceBuffer device_c(device, sizeof(c));
    lrrt_example::check(lr_memcpy(device, device_a.data(), a, sizeof(a),
                                  LR_MEMCPY_HOST_TO_DEVICE),
                        "copy a");
    lrrt_example::check(lr_memcpy(device, device_b.data(), b, sizeof(b),
                                  LR_MEMCPY_HOST_TO_DEVICE),
                        "copy b");

    std::vector<unsigned char> hsaco =
        lrrt_example::read_file(LRRT_VECTOR_ADD_HSACO);
    lrrt_example::Module module(device, hsaco);
    lr_kernel_t *kernel = module.kernel("vector_add");

    vector_add_args_t args = {
        (const float *)device_a.data(),
        (const float *)device_b.data(),
        (float *)device_c.data(),
        n,
    };
    lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
    lrrt_example::check(lr_launch(kernel, &config, &args, sizeof(args)),
                        "lr_launch");
    lrrt_example::check(lr_memcpy(device, c, device_c.data(), sizeof(c),
                                  LR_MEMCPY_DEVICE_TO_HOST),
                        "copy c back");

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
