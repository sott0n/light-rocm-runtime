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
    throw std::runtime_error("failed to open async-copy launch HSACO");
  }
  std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("async-copy launch HSACO is empty");
  }
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> data(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    throw std::runtime_error("failed to read async-copy launch HSACO");
  }
  return data;
}

} // namespace

int main() {
  try {
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      printf("async_copy_launch: skipped, no GPU devices\n");
      return 0;
    }
    lrrt::Device device = runtime.open_device(0);

    const int32_t n = 1 << 20;
    const float alpha = 2.5f;
    std::vector<float> input(n);
    for (int32_t i = 0; i < n; ++i) {
      input[i] = 0.25f * (float)(i % 113);
    }

    lrrt::DeviceBuffer source(device, input.size() * sizeof(float));
    lrrt::DeviceBuffer staging(device, input.size() * sizeof(float));
    lrrt::DeviceBuffer result(device, sizeof(float));
    lrrt::copy_to_device(source, input);

    std::vector<unsigned char> hsaco = read_file(LRRT_ASYNC_COPY_LAUNCH_HSACO);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel("async_copy_launch_kernel");
    const lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
    const ScaleArgs args = {
        static_cast<const float *>(staging.data()),
        static_cast<float *>(result.data()),
        alpha,
        n - 1,
    };

    lrrt::Event copy_complete(device);
    lrrt::copy_device_to_device_async(
        staging, source, input.size() * sizeof(float), copy_complete);
    lrrt::launch(kernel, config, args);
    lrrt::launch(kernel, config, args);
    device.synchronize();
    float output = 0.0f;
    lrrt::copy_to_host(&output, result, sizeof(output));

    float expected = input.back() * alpha;
    if (output != expected) {
      fprintf(stderr, "async_copy_launch mismatch: actual=%f expected=%f\n",
              output, expected);
      return 1;
    }

    printf("async_copy_launch: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
