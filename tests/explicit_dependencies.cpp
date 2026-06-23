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
    throw std::runtime_error("failed to open explicit-dependency HSACO");
  }
  std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("explicit-dependency HSACO is empty");
  }
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> data(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    throw std::runtime_error("failed to read explicit-dependency HSACO");
  }
  return data;
}

void expect_value(const lrrt::DeviceBuffer &buffer, float expected,
                  const char *scenario) {
  float actual = 0.0f;
  lrrt::copy_to_host(&actual, buffer, sizeof(actual));
  if (actual != expected) {
    fprintf(stderr, "explicit_dependencies %s: actual=%f expected=%f\n",
            scenario, actual, expected);
    throw std::runtime_error("explicit-dependency result mismatch");
  }
}

} // namespace

int main() {
  try {
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      printf("explicit_dependencies: skipped, no GPU devices\n");
      return 0;
    }
    lrrt::Device device = runtime.open_device(0);

    const int32_t n = 1 << 20;
    const float alpha = 2.5f;
    std::vector<float> input(n, 1.0f);
    input.back() = 3.0f;
    lrrt::DeviceBuffer source(device, input.size() * sizeof(float));
    lrrt::DeviceBuffer staging(device, input.size() * sizeof(float));
    lrrt::DeviceBuffer copied(device, input.size() * sizeof(float));
    lrrt::DeviceBuffer kernel_output(device, sizeof(float));
    lrrt::DeviceBuffer final_output(device, sizeof(float));
    lrrt::copy_to_device(source, input);

    std::vector<unsigned char> hsaco = read_file(LRRT_ASYNC_COPY_LAUNCH_HSACO);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel("async_copy_launch_kernel");
    const lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
    const ScaleArgs copied_args = {static_cast<const float *>(copied.data()),
                                   static_cast<float *>(kernel_output.data()),
                                   alpha, n - 1};

    lrrt::Event first_copy(device);
    lrrt::Event second_copy(device);
    lrrt::copy_device_to_device_async(staging, source, source.size(),
                                      first_copy, {});
    lrrt::copy_device_to_device_async(copied, staging, staging.size(),
                                      second_copy, {&first_copy});
    lrrt::launch(kernel, config, copied_args, {&second_copy});
    device.synchronize();
    expect_value(kernel_output, input.back() * alpha, "copy-copy-launch");

    const ScaleArgs source_args = {static_cast<const float *>(source.data()),
                                   static_cast<float *>(kernel_output.data()),
                                   alpha, n - 1};
    lrrt::launch(kernel, config, source_args, {});
    lrrt::Event kernel_complete(device);
    kernel_complete.record();
    lrrt::Event final_copy(device);
    lrrt::copy_device_to_device_async(final_output, kernel_output,
                                      sizeof(float), final_copy,
                                      {&kernel_complete});
    device.synchronize();
    expect_value(final_output, input.back() * alpha, "launch-copy");

    lrrt::Event unrecorded(device);
    lr_event_t *invalid_dependency[] = {unrecorded.get()};
    lr_status_t status =
        lr_launch_with_dependencies(kernel.get(), &config, &source_args,
                                    sizeof(source_args), invalid_dependency, 1);
    if (status != LR_ERROR_INVALID_ARGUMENT) {
      throw std::runtime_error("unrecorded dependency was accepted");
    }

    lr_event_t *duplicate_dependencies[] = {kernel_complete.get(),
                                            kernel_complete.get()};
    status = lr_launch_with_dependencies(kernel.get(), &config, &source_args,
                                         sizeof(source_args),
                                         duplicate_dependencies, 2);
    if (status != LR_ERROR_INVALID_ARGUMENT) {
      throw std::runtime_error("duplicate dependencies were accepted");
    }

    printf("explicit_dependencies: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
