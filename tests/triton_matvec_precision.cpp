#include "lrrt/bundle.hpp"
#include "lrrt/lrrt.hpp"

#include "common/precision_utils.h"

#include <math.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#ifndef LRRT_TRITON_MATVEC_MANIFEST
#define LRRT_TRITON_MATVEC_MANIFEST "manifest.json"
#endif

using lrrt_test::data_type_name;
using lrrt_test::data_type_size;
using lrrt_test::DataType;
using lrrt_test::read_value;
using lrrt_test::round_value;
using lrrt_test::write_value;

static void run_case(lrrt::Device &device, DataType data_type, uint32_t outputs,
                     uint32_t hidden) {
  uint32_t block_size = hidden <= 1024 ? 1024 : hidden <= 2048 ? 2048 : 4096;
  std::string kernel_name = std::string("matvec_") + data_type_name(data_type) +
                            "_" + std::to_string(block_size);
  lrrt::Bundle bundle(device, LRRT_TRITON_MATVEC_MANIFEST, kernel_name.c_str());

  const size_t element_size = data_type_size(data_type);
  std::vector<unsigned char> x(hidden * element_size);
  std::vector<unsigned char> weight(outputs * hidden * element_size);
  std::vector<unsigned char> out(outputs * element_size, 0);
  for (uint32_t i = 0; i < hidden; ++i) {
    write_value(x, i, data_type, 0.03125f * (float)((int32_t)(i % 23) - 11));
  }
  for (uint32_t row = 0; row < outputs; ++row) {
    for (uint32_t col = 0; col < hidden; ++col) {
      uint32_t index = row * hidden + col;
      int32_t lane = (int32_t)((index + row * 7) % 19) - 9;
      write_value(weight, index, data_type, 0.015625f * (float)lane);
    }
  }

  lrrt::DeviceBuffer device_x(device, x.size());
  lrrt::DeviceBuffer device_weight(device, weight.size());
  lrrt::DeviceBuffer device_out(device, out.size());
  lrrt::copy_to_device(device_x, x.data(), x.size());
  lrrt::copy_to_device(device_weight, weight.data(), weight.size());

  lrrt::KernargBuffer kernel_args = bundle.make_args();
  kernel_args.set("x", device_x.data());
  kernel_args.set("weight", device_weight.data());
  kernel_args.set("out", device_out.data());
  kernel_args.set("outputs", (int32_t)outputs);
  kernel_args.set("hidden", (int32_t)hidden);
  kernel_args.bind_optional_nulls();
  bundle.launch(outputs, kernel_args);
  device.synchronize();
  lrrt::copy_to_host(out.data(), device_out, out.size());

  float max_diff = 0.0f;
  uint32_t max_index = 0;
  float max_expected = 0.0f;
  float max_actual = 0.0f;
  for (uint32_t row = 0; row < outputs; ++row) {
    float expected = 0.0f;
    for (uint32_t col = 0; col < hidden; ++col) {
      expected += read_value(x, col, data_type) *
                  read_value(weight, row * hidden + col, data_type);
    }
    expected = round_value(expected, data_type);
    float actual = read_value(out, row, data_type);
    float diff = fabsf(actual - expected);
    if (diff > max_diff) {
      max_diff = diff;
      max_index = row;
      max_expected = expected;
      max_actual = actual;
    }
  }
  float tolerance = data_type == DataType::Fp16 ? 0.004f : 0.05f;
  if (max_diff > tolerance) {
    fprintf(stderr,
            "triton_matvec dtype=%s hidden=%u mismatch at %u: actual=%f "
            "expected=%f diff=%f tolerance=%f\n",
            data_type_name(data_type), hidden, max_index, max_actual,
            max_expected, max_diff, tolerance);
    throw std::runtime_error("triton_matvec precision mismatch");
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
      for (uint32_t hidden : {768u, 1536u, 3072u}) {
        run_case(device, data_type, 17, hidden);
      }
    }

    printf("triton_matvec_precision: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
