#include "lrrt/lrrt.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#ifndef LRRT_ASYNC_COPY_LAUNCH_HSACO
#define LRRT_ASYNC_COPY_LAUNCH_HSACO "async_copy_launch_kernel.hsaco"
#endif

namespace {

struct ScaleArgs {
  const float *input;
  float *output;
  float alpha;
  int32_t index;
};

std::vector<unsigned char> read_file(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("failed to open double-buffer HSACO");
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("double-buffer HSACO is empty");
  }
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> data(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    throw std::runtime_error("failed to read double-buffer HSACO");
  }
  return data;
}

void expect_invalid_argument(const char *scenario,
                             const std::function<void()> &operation) {
  try {
    operation();
  } catch (const lrrt::Error &error) {
    if (error.status() == LR_ERROR_INVALID_ARGUMENT) {
      return;
    }
  }
  throw std::runtime_error(std::string(scenario) +
                           " did not report invalid argument");
}

} // namespace

int main() {
  try {
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      printf("double_buffer_pipeline: skipped, no GPU devices\n");
      return 0;
    }
    lrrt::Device device = runtime.open_device(0);
    lrrt::Queue queue(device);

    constexpr size_t elements = 4096;
    constexpr size_t chunks = 7;
    constexpr float alpha = 2.5f;
    const size_t bytes = elements * sizeof(float);

    std::vector<unsigned char> hsaco = read_file(LRRT_ASYNC_COPY_LAUNCH_HSACO);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel("async_copy_launch_kernel");
    lrrt::DeviceBuffer output(device, (chunks + 1) * sizeof(float));
    lrrt::PinnedHostDoubleBuffer pipeline(device, bytes);
    const lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};

    for (size_t chunk = 0; chunk < chunks; ++chunk) {
      auto &slot = pipeline.acquire();
      if (chunk == 0) {
        expect_invalid_argument("duplicate acquire",
                                [&] { static_cast<void>(pipeline.acquire()); });
        expect_invalid_argument(
            "zero-size copy", [&] { pipeline.copy_to_device_async(slot, 0); });
        expect_invalid_argument("oversized copy", [&] {
          pipeline.copy_to_device_async(slot, bytes + 1);
        });
      }

      auto *input = static_cast<float *>(slot.host_data());
      for (size_t i = 0; i < elements; ++i) {
        input[i] = static_cast<float>(chunk * elements + i);
      }

      pipeline.copy_to_device_async(slot, bytes);
      auto *output_value =
          static_cast<float *>(output.data()) + static_cast<ptrdiff_t>(chunk);
      const ScaleArgs args = {
          static_cast<const float *>(slot.device_buffer().data()), output_value,
          alpha, static_cast<int32_t>(elements - 1)};
      lrrt::launch(queue, kernel, config, args, {&slot.copy_complete()});
      pipeline.mark_work_submitted(slot, queue);
    }
    pipeline.finish();

    constexpr size_t final_chunk = chunks;
    constexpr float final_input = 1234.0f;
    auto &final_slot = pipeline.acquire();
    auto *final_values = static_cast<float *>(final_slot.host_data());
    std::fill(final_values, final_values + elements, final_input);
    pipeline.copy_to_device_async(final_slot, bytes);
    const ScaleArgs final_args = {
        static_cast<const float *>(final_slot.device_buffer().data()),
        static_cast<float *>(output.data()) + final_chunk, alpha,
        static_cast<int32_t>(elements - 1)};
    lrrt::launch(queue, kernel, config, final_args,
                 {&final_slot.copy_complete()});
    pipeline.mark_work_submitted(final_slot, queue);
    pipeline.finish();

    std::vector<float> actual(chunks + 1);
    lrrt::copy_to_host(actual, output);
    for (size_t chunk = 0; chunk < chunks; ++chunk) {
      const float input = static_cast<float>(chunk * elements + elements - 1);
      const float expected = input * alpha;
      if (std::fabs(actual[chunk] - expected) > 1.0e-5f) {
        fprintf(stderr,
                "double_buffer_pipeline chunk %zu: actual=%f expected=%f\n",
                chunk, actual[chunk], expected);
        return 1;
      }
    }
    if (std::fabs(actual[final_chunk] - final_input * alpha) > 1.0e-5f) {
      throw std::runtime_error("double-buffer reuse after finish failed");
    }

    printf("double_buffer_pipeline: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
