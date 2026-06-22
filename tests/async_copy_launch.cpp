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

void expect_result(lrrt::DeviceBuffer &result, float expected,
                   const char *scenario) {
  float output = 0.0f;
  lrrt::copy_to_host(&output, result, sizeof(output));
  if (output != expected) {
    fprintf(stderr, "async_copy_launch %s mismatch: actual=%f expected=%f\n",
            scenario, output, expected);
    throw std::runtime_error("async-copy launch result mismatch");
  }
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
    const float expected = input.back() * alpha;
    expect_result(result, expected, "multiple launch");

    lr_event_t *destroyed_early = nullptr;
    lrrt::check(lr_event_create(device.get(), &destroyed_early),
                "lr_event_create");
    lrrt::check(lr_memcpy_async(device.get(), staging.data(), source.data(),
                                staging.size(), LR_MEMCPY_DEVICE_TO_DEVICE,
                                destroyed_early),
                "lr_memcpy_async");
    lrrt::launch(kernel, config, args);
    lrrt::check(lr_event_destroy(destroyed_early), "lr_event_destroy");
    expect_result(result, expected, "early event destroy");

    constexpr uint32_t stress_launches = 2048;
    lrrt::Event stress_copy(device);
    lrrt::copy_device_to_device_async(staging, source, staging.size(),
                                      stress_copy);
    for (uint32_t i = 0; i < stress_launches; ++i) {
      lrrt::launch(kernel, config, args);
    }
    device.synchronize();
    expect_result(result, expected, "queue stress");

    printf("async_copy_launch: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
