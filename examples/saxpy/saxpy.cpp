#include "../common/example_utils.h"

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
    const float alpha = 2.5f;
    float x[n];
    float y[n];
    for (int i = 0; i < n; ++i) {
      x[i] = (float)i;
      y[i] = (float)(i * 3);
    }

    lrrt_example::DeviceBuffer device_x(device, sizeof(x));
    lrrt_example::DeviceBuffer device_y(device, sizeof(y));
    lrrt_example::check(lr_memcpy(device, device_x.data(), x, sizeof(x),
                                  LR_MEMCPY_HOST_TO_DEVICE),
                        "copy x");
    lrrt_example::check(lr_memcpy(device, device_y.data(), y, sizeof(y),
                                  LR_MEMCPY_HOST_TO_DEVICE),
                        "copy y");

    std::vector<unsigned char> hsaco =
        lrrt_example::read_file(LRRT_SAXPY_HSACO);
    lrrt_example::Module module(device, hsaco);
    lr_kernel_t *kernel = module.kernel("saxpy");

    saxpy_args_t args = {
        alpha,
        (const float *)device_x.data(),
        (float *)device_y.data(),
        n,
    };
    lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
    lrrt_example::check(lr_launch(kernel, &config, &args, sizeof(args)),
                        "lr_launch");
    lrrt_example::check(lr_memcpy(device, y, device_y.data(), sizeof(y),
                                  LR_MEMCPY_DEVICE_TO_HOST),
                        "copy y back");

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
