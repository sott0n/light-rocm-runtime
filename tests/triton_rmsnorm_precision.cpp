#include "lrrt/bundle.hpp"
#include "lrrt/lrrt.hpp"

#include "common/precision_utils.h"

#include <math.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#ifndef LRRT_TRITON_RMSNORM_MANIFEST
#define LRRT_TRITON_RMSNORM_MANIFEST "manifest.json"
#endif

typedef struct triton_rmsnorm_args_t {
  const void *x;
  const void *weight;
  void *out;
  float eps;
  int32_t rows;
  int32_t hidden;
  void *triton_global_scratch;
  void *triton_profile_scratch;
} triton_rmsnorm_args_t;

static_assert(sizeof(triton_rmsnorm_args_t) == 56,
              "Triton rmsnorm kernarg layout must match manifest");

using lrrt_test::data_type_name;
using lrrt_test::data_type_size;
using lrrt_test::DataType;
using lrrt_test::read_value;
using lrrt_test::round_value;
using lrrt_test::write_value;

static void run_case(lrrt::Device &device, DataType data_type, uint32_t rows,
                     uint32_t hidden) {
  uint32_t block_size = hidden <= 1024 ? 1024 : hidden <= 2048 ? 2048 : 4096;
  std::string kernel_name = std::string("rmsnorm_") +
                            data_type_name(data_type) + "_" +
                            std::to_string(block_size);
  lrrt::Bundle bundle(device, LRRT_TRITON_RMSNORM_MANIFEST,
                      kernel_name.c_str());
  lrrt::require_kernarg_layout(
      bundle.manifest(), sizeof(triton_rmsnorm_args_t),
      {
          offsetof(triton_rmsnorm_args_t, x),
          offsetof(triton_rmsnorm_args_t, weight),
          offsetof(triton_rmsnorm_args_t, out),
          offsetof(triton_rmsnorm_args_t, eps),
          offsetof(triton_rmsnorm_args_t, rows),
          offsetof(triton_rmsnorm_args_t, hidden),
          offsetof(triton_rmsnorm_args_t, triton_global_scratch),
          offsetof(triton_rmsnorm_args_t, triton_profile_scratch),
      });

  const uint32_t elements = rows * hidden;
  const float eps = 1.0e-5f;
  const size_t element_size = data_type_size(data_type);
  std::vector<unsigned char> x(elements * element_size);
  std::vector<unsigned char> weight(hidden * element_size);
  std::vector<unsigned char> out(elements * element_size, 0);
  for (uint32_t i = 0; i < hidden; ++i) {
    write_value(weight, i, data_type, 1.0f + 0.001f * (float)(i % 29));
  }
  for (uint32_t i = 0; i < elements; ++i) {
    write_value(x, i, data_type, 0.117f * (float)((int32_t)(i % 17) - 8));
  }

  lrrt::DeviceBuffer device_x(device, x.size());
  lrrt::DeviceBuffer device_weight(device, weight.size());
  lrrt::DeviceBuffer device_out(device, out.size());
  lrrt::copy_to_device(device_x, x.data(), x.size());
  lrrt::copy_to_device(device_weight, weight.data(), weight.size());

  triton_rmsnorm_args_t kernel_args = {
      device_x.data(), device_weight.data(), device_out.data(), eps,
      (int32_t)rows,   (int32_t)hidden,      nullptr,           nullptr,
  };
  lrrt::launch(bundle.kernel(), bundle.launch_config(rows), kernel_args);
  device.synchronize();
  lrrt::copy_to_host(out.data(), device_out, out.size());

  float max_diff = 0.0f;
  uint32_t max_index = 0;
  float max_expected = 0.0f;
  float max_actual = 0.0f;
  for (uint32_t row = 0; row < rows; ++row) {
    float sum_square = 0.0f;
    for (uint32_t col = 0; col < hidden; ++col) {
      float value = read_value(x, row * hidden + col, data_type);
      sum_square += value * value;
    }
    float scale = 1.0f / sqrtf(sum_square / (float)hidden + eps);
    for (uint32_t col = 0; col < hidden; ++col) {
      uint32_t index = row * hidden + col;
      float expected = read_value(x, index, data_type) * scale *
                       read_value(weight, col, data_type);
      expected = round_value(expected, data_type);
      float actual = read_value(out, index, data_type);
      float diff = fabsf(actual - expected);
      if (diff > max_diff) {
        max_diff = diff;
        max_index = index;
        max_expected = expected;
        max_actual = actual;
      }
    }
  }
  float tolerance = data_type == DataType::Fp16 ? 0.004f : 0.02f;
  if (max_diff > tolerance) {
    fprintf(stderr,
            "triton_rmsnorm dtype=%s hidden=%u mismatch at %u: actual=%f "
            "expected=%f diff=%f tolerance=%f\n",
            data_type_name(data_type), hidden, max_index, max_actual,
            max_expected, max_diff, tolerance);
    throw std::runtime_error("triton_rmsnorm precision mismatch");
  }
}

int main(void) {
  try {
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      return 0;
    }
    lrrt::Device device = runtime.open_device(0);

    for (DataType data_type : {DataType::Fp16, DataType::Bf16}) {
      for (uint32_t hidden : {1003u, 1536u, 3072u}) {
        run_case(device, data_type, 4, hidden);
      }
    }

    printf("triton_rmsnorm_precision: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
