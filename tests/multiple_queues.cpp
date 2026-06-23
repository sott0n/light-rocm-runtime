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
    throw std::runtime_error("failed to open multiple-queue HSACO");
  }
  std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("multiple-queue HSACO is empty");
  }
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> data(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    throw std::runtime_error("failed to read multiple-queue HSACO");
  }
  return data;
}

void expect_value(const lrrt::DeviceBuffer &buffer, float expected,
                  const char *scenario) {
  float actual = 0.0f;
  lrrt::copy_to_host(&actual, buffer, sizeof(actual));
  if (actual != expected) {
    fprintf(stderr, "multiple_queues %s: actual=%f expected=%f\n", scenario,
            actual, expected);
    throw std::runtime_error("multiple-queue result mismatch");
  }
}

} // namespace

int main() {
  try {
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      printf("multiple_queues: skipped, no GPU devices\n");
      return 0;
    }
    lrrt::Device device = runtime.open_device(0);

    const int32_t n = 1 << 20;
    const float alpha = 2.5f;
    std::vector<float> input(n, 1.0f);
    input.back() = 3.0f;
    lrrt::DeviceBuffer source(device, input.size() * sizeof(float));
    lrrt::DeviceBuffer staging(device, input.size() * sizeof(float));
    lrrt::DeviceBuffer first_output(device, sizeof(float));
    lrrt::DeviceBuffer copied_output(device, sizeof(float));
    lrrt::DeviceBuffer final_output(device, sizeof(float));
    lrrt::copy_to_device(source, input);

    std::vector<unsigned char> hsaco = read_file(LRRT_ASYNC_COPY_LAUNCH_HSACO);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel("async_copy_launch_kernel");
    lrrt::Queue first_queue(device);
    lrrt::Queue second_queue(device);
    const lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};

    const ScaleArgs first_args = {static_cast<const float *>(source.data()),
                                  static_cast<float *>(first_output.data()),
                                  alpha, n - 1};
    const ScaleArgs second_args = {
        static_cast<const float *>(first_output.data()),
        static_cast<float *>(final_output.data()), alpha, 0};
    lrrt::launch(first_queue, kernel, config, first_args);
    lrrt::Event first_complete(device);
    first_complete.record(first_queue);
    lrrt::launch(second_queue, kernel, config, second_args, {&first_complete});
    second_queue.synchronize();
    expect_value(final_output, input.back() * alpha * alpha, "cross-queue");

    lrrt::Event input_copied(device);
    lrrt::copy_device_to_device_async(staging, source, source.size(),
                                      input_copied, {});
    const ScaleArgs copied_args = {static_cast<const float *>(staging.data()),
                                   static_cast<float *>(first_output.data()),
                                   alpha, n - 1};
    lrrt::launch(first_queue, kernel, config, copied_args, {&input_copied});
    lrrt::Event kernel_complete(device);
    kernel_complete.record(first_queue);
    lrrt::Event output_copied(device);
    lrrt::copy_device_to_device_async(copied_output, first_output,
                                      sizeof(float), output_copied,
                                      {&kernel_complete});
    const ScaleArgs final_args = {
        static_cast<const float *>(copied_output.data()),
        static_cast<float *>(final_output.data()), alpha, 0};
    lrrt::launch(second_queue, kernel, config, final_args, {&output_copied});
    second_queue.synchronize();
    expect_value(final_output, input.back() * alpha * alpha,
                 "copy-compute-chain");

    lr_queue_t *destroyed_queue = nullptr;
    lrrt::check(lr_queue_create(device.get(), &destroyed_queue),
                "lr_queue_create");
    lrrt::check(lr_queue_destroy(destroyed_queue), "lr_queue_destroy");
    if (lr_queue_synchronize(destroyed_queue) != LR_ERROR_INVALID_ARGUMENT) {
      throw std::runtime_error("destroyed queue was accepted");
    }

    printf("multiple_queues: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
