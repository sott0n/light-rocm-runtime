#include "lrrt/bundle.hpp"
#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#ifndef LRRT_TRITON_ROPE_MANIFEST
#define LRRT_TRITON_ROPE_MANIFEST "manifest.json"
#endif

static void run_case(lrrt::Device &device, lrrt::Bundle &bundle,
                     uint32_t tokens, uint32_t heads, uint32_t head_dim) {
  if (head_dim == 0 || head_dim > 128 || head_dim % 2 != 0) {
    throw std::invalid_argument("RoPE head dimension must be even and <= 128");
  }

  const uint32_t rows = tokens * heads;
  const uint32_t elements = rows * head_dim;
  const uint32_t half = head_dim / 2;
  std::vector<float> x(elements);
  std::vector<float> cos(tokens * half);
  std::vector<float> sin(tokens * half);
  std::vector<float> out(elements, 0.0f);

  for (uint32_t i = 0; i < elements; ++i) {
    x[i] = 0.0625f * (float)((int32_t)(i % 37) - 18);
  }
  for (uint32_t token = 0; token < tokens; ++token) {
    for (uint32_t frequency = 0; frequency < half; ++frequency) {
      float exponent = -2.0f * (float)frequency / (float)head_dim;
      float inverse_frequency = powf(10000.0f, exponent);
      float angle = (float)(token + 7) * inverse_frequency;
      uint32_t index = token * half + frequency;
      cos[index] = cosf(angle);
      sin[index] = sinf(angle);
    }
  }

  lrrt::DeviceBuffer device_x(device, x.size() * sizeof(float));
  lrrt::DeviceBuffer device_cos(device, cos.size() * sizeof(float));
  lrrt::DeviceBuffer device_sin(device, sin.size() * sizeof(float));
  lrrt::DeviceBuffer device_out(device, out.size() * sizeof(float));
  lrrt::copy_to_device(device_x, x);
  lrrt::copy_to_device(device_cos, cos);
  lrrt::copy_to_device(device_sin, sin);

  lrrt::KernargBuffer kernel_args(bundle.manifest());
  kernel_args.set("x", (const float *)device_x.data());
  kernel_args.set("cos", (const float *)device_cos.data());
  kernel_args.set("sin", (const float *)device_sin.data());
  kernel_args.set("out", (float *)device_out.data());
  kernel_args.set("rows", (int32_t)rows);
  kernel_args.set("heads", (int32_t)heads);
  kernel_args.set("head_dim", (int32_t)head_dim);
  void *scratch = nullptr;
  kernel_args.set("_triton_global_scratch", scratch);
  kernel_args.set("_triton_profile_scratch", scratch);
  bundle.launch(rows, kernel_args);
  device.synchronize();
  lrrt::copy_to_host(out, device_out);

  float max_diff = 0.0f;
  uint32_t max_index = 0;
  float max_expected = 0.0f;
  float max_actual = 0.0f;
  for (uint32_t row = 0; row < rows; ++row) {
    uint32_t token = row / heads;
    uint32_t base = row * head_dim;
    for (uint32_t offset = 0; offset < head_dim; ++offset) {
      uint32_t frequency = offset % half;
      uint32_t partner = offset < half ? offset + half : offset - half;
      float rotated = offset < half ? -x[base + partner] : x[base + partner];
      float expected = x[base + offset] * cos[token * half + frequency] +
                       rotated * sin[token * half + frequency];
      float diff = fabsf(out[base + offset] - expected);
      if (diff > max_diff) {
        max_diff = diff;
        max_index = base + offset;
        max_expected = expected;
        max_actual = out[base + offset];
      }
    }
  }
  if (max_diff > 0.001f) {
    fprintf(stderr,
            "triton_rope head_dim=%u mismatch at %u: actual=%f expected=%f "
            "diff=%f\n",
            head_dim, max_index, max_actual, max_expected, max_diff);
    throw std::runtime_error("triton_rope result mismatch");
  }
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

    lrrt::Bundle rope_64(device, LRRT_TRITON_ROPE_MANIFEST, "rope_64");
    lrrt::Bundle rope_128(device, LRRT_TRITON_ROPE_MANIFEST, "rope_128");
    for (lrrt::Bundle *bundle : {&rope_64, &rope_128}) {
      printf("loaded Triton manifest for kernel: %s\n",
             bundle->manifest().name.c_str());
    }

    for (uint32_t head_dim : {48u, 64u, 96u, 128u}) {
      lrrt::Bundle &bundle = head_dim <= 64 ? rope_64 : rope_128;
      run_case(device, bundle, 5, 3, head_dim);
    }

    printf("triton_rope: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
