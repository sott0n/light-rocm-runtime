#include "../common/example_utils.h"
#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#ifndef LRRT_TRITON_VECTOR_ADD_HSACO
#define LRRT_TRITON_VECTOR_ADD_HSACO "kernels.hsaco"
#endif

#ifndef LRRT_TRITON_VECTOR_ADD_MANIFEST
#define LRRT_TRITON_VECTOR_ADD_MANIFEST "manifest.json"
#endif

typedef struct triton_vector_add_args_t {
  const float *x;
  const float *y;
  float *out;
  int32_t n;
  int32_t padding;
  void *triton_scratch_0;
  void *triton_scratch_1;
} triton_vector_add_args_t;

static_assert(sizeof(triton_vector_add_args_t) == 48,
              "Triton vector_add kernarg layout must match manifest");

static uint32_t ceil_div(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

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

    const uint32_t n = 1024;
    float x[n];
    float y[n];
    float out[n];
    for (uint32_t i = 0; i < n; ++i) {
      x[i] = (float)i;
      y[i] = (float)(i * 2);
      out[i] = 0.0f;
    }

    lrrt::DeviceBuffer device_x(device, sizeof(x));
    lrrt::DeviceBuffer device_y(device, sizeof(y));
    lrrt::DeviceBuffer device_out(device, sizeof(out));
    lrrt::copy_to_device(device_x, x);
    lrrt::copy_to_device(device_y, y);

    std::vector<unsigned char> manifest =
        lrrt_example::read_file(LRRT_TRITON_VECTOR_ADD_MANIFEST);
    printf("loaded Triton manifest: %zu bytes\n", manifest.size());

    std::vector<unsigned char> hsaco =
        lrrt_example::read_file(LRRT_TRITON_VECTOR_ADD_HSACO);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel("vector_add_kernel");

    triton_vector_add_args_t kernel_args = {
        (const float *)device_x.data(),
        (const float *)device_y.data(),
        (float *)device_out.data(),
        (int32_t)n,
        0,
        nullptr,
        nullptr,
    };

    const uint32_t workgroup_size = 128;
    const uint32_t program_count = ceil_div(n, 256);
    lr_launch_config_t config = {
        {program_count * workgroup_size, 1, 1},
        {workgroup_size, 1, 1},
        0,
    };
    lrrt::launch(kernel, config, kernel_args);
    device.synchronize();
    lrrt::copy_to_host(out, device_out);

    for (uint32_t i = 0; i < n; ++i) {
      if (fabsf(out[i] - (x[i] + y[i])) > 0.001f) {
        throw std::runtime_error("triton_vector_add result mismatch");
      }
    }

    printf("triton_vector_add: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
