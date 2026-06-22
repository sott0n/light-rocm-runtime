#include "lrrt/lrrt.hpp"

#include <fstream>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#ifndef LRRT_ASYNC_COPY_LAUNCH_HSACO
#define LRRT_ASYNC_COPY_LAUNCH_HSACO "async_copy_launch_kernel.hsaco"
#endif

namespace {

struct ScaleArgs {
  const float *in;
  float *out;
  float alpha;
  int32_t index;
};

std::vector<unsigned char> read_file(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("failed to open launch-copy HSACO");
  }
  std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("launch-copy HSACO is empty");
  }
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> data(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    throw std::runtime_error("failed to read launch-copy HSACO");
  }
  return data;
}

} // namespace

int main() {
  try {
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      printf("launch_async_copy: skipped, no GPU devices\n");
      return 0;
    }
    lrrt::Device device = runtime.open_device(0);

    const int32_t n = 1 << 20;
    const float alpha = 2.5f;
    std::vector<float> input(n, 1.0f);
    input.back() = 3.0f;
    lrrt::DeviceBuffer kernel_input(device, input.size() * sizeof(float));
    lrrt::DeviceBuffer kernel_output(device, sizeof(float));
    lrrt::DeviceBuffer copy_output(device, sizeof(float));
    lrrt::DeviceBuffer chained_output(device, sizeof(float));
    lrrt::copy_to_device(kernel_input, input);

    std::vector<unsigned char> hsaco = read_file(LRRT_ASYNC_COPY_LAUNCH_HSACO);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel("async_copy_launch_kernel");
    const lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
    const ScaleArgs args = {
        static_cast<const float *>(kernel_input.data()),
        static_cast<float *>(kernel_output.data()),
        alpha,
        n - 1,
    };
    const ScaleArgs chained_args = {
        static_cast<const float *>(copy_output.data()),
        static_cast<float *>(chained_output.data()),
        alpha,
        0,
    };

    constexpr uint32_t dispatch_count = 64;
    for (uint32_t i = 0; i < dispatch_count; ++i) {
      lrrt::launch(kernel, config, args);
    }
    lrrt::Event copy_complete(device);
    lrrt::copy_device_to_device_async(copy_output, kernel_output, sizeof(float),
                                      copy_complete);
    lrrt::launch(kernel, config, chained_args);
    device.synchronize();

    float output = 0.0f;
    lrrt::copy_to_host(&output, chained_output, sizeof(output));
    const float expected = input.back() * alpha * alpha;
    if (output != expected) {
      fprintf(stderr, "launch_async_copy mismatch: actual=%f expected=%f\n",
              output, expected);
      return 1;
    }

    printf("launch_async_copy: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
