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
    throw std::runtime_error("failed to open queue-event lifetime HSACO");
  }
  std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("queue-event lifetime HSACO is empty");
  }
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> data(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    throw std::runtime_error("failed to read queue-event lifetime HSACO");
  }
  return data;
}

void expect_value(const lrrt::DeviceBuffer &buffer, float expected,
                  const char *scenario) {
  float actual = 0.0f;
  lrrt::copy_to_host(&actual, buffer, sizeof(actual));
  if (actual != expected) {
    fprintf(stderr, "queue_event_lifetime %s: actual=%f expected=%f\n",
            scenario, actual, expected);
    throw std::runtime_error("queue-event lifetime result mismatch");
  }
}

} // namespace

int main() {
  try {
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      printf("queue_event_lifetime: skipped, no GPU devices\n");
      return 0;
    }
    lrrt::Device device = runtime.open_device(0);

    const int32_t n = 1 << 20;
    const float alpha = 2.5f;
    std::vector<float> input(n, 1.0f);
    input.back() = 3.0f;

    lrrt::DeviceBuffer source(device, input.size() * sizeof(float));
    lrrt::DeviceBuffer staging(device, input.size() * sizeof(float));
    lrrt::DeviceBuffer output(device, sizeof(float));
    lrrt::copy_to_device(source, input);

    lr_event_t *copy_event = nullptr;
    lrrt::check(lr_event_create(device.get(), &copy_event), "lr_event_create");
    lrrt::check(lr_memcpy_async(device.get(), staging.data(), source.data(),
                                staging.size(), LR_MEMCPY_DEVICE_TO_DEVICE,
                                copy_event),
                "lr_memcpy_async");
    lrrt::check(lr_event_destroy(copy_event), "lr_event_destroy");

    std::vector<float> copied(n, 0.0f);
    lrrt::copy_to_host(copied, staging);
    if (copied.back() != input.back()) {
      throw std::runtime_error(
          "event destroy did not drain pending async copy");
    }
    if (lr_event_synchronize(copy_event) != LR_ERROR_INVALID_ARGUMENT) {
      throw std::runtime_error("destroyed event was accepted");
    }

    std::vector<unsigned char> hsaco = read_file(LRRT_ASYNC_COPY_LAUNCH_HSACO);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel("async_copy_launch_kernel");
    const lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
    const ScaleArgs args = {static_cast<const float *>(source.data()),
                            static_cast<float *>(output.data()), alpha, n - 1};

    lr_queue_t *queue = nullptr;
    lrrt::check(lr_queue_create(device.get(), &queue), "lr_queue_create");
    lrrt::check(
        lr_launch_on_queue(queue, kernel.get(), &config, &args, sizeof(args)),
        "lr_launch_on_queue");
    lrrt::check(lr_queue_destroy(queue), "lr_queue_destroy");
    expect_value(output, input.back() * alpha, "queue destroy drains dispatch");

    if (lr_queue_synchronize(queue) != LR_ERROR_INVALID_ARGUMENT) {
      throw std::runtime_error("destroyed queue was accepted");
    }

    lrrt::Queue live_queue(device);
    lrrt::launch(live_queue, kernel, config, args);
    live_queue.synchronize();
    expect_value(output, input.back() * alpha, "explicit queue synchronize");

    printf("queue_event_lifetime: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
