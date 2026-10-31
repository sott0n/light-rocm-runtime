#include "lrrt/lrrt.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef DIRECT_KFD_PIPELINE_HSACO
#error "DIRECT_KFD_PIPELINE_HSACO must name the test code object"
#endif

namespace {

constexpr size_t kElementCount = 64;

struct VectorAddArguments {
  const float *input_a;
  const float *input_b;
  float *output;
  int32_t element_count;
};

void expect_status(lr_status_t actual, lr_status_t expected,
                   const char *operation) {
  if (actual != expected) {
    throw std::runtime_error(std::string(operation) + " returned " +
                             lr_status_string(actual) + ", expected " +
                             lr_status_string(expected));
  }
}

std::vector<uint8_t> read_file(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("failed to open direct KFD pipeline fixture");
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("direct KFD pipeline fixture is empty");
  }
  std::vector<uint8_t> data(static_cast<size_t>(size));
  file.seekg(0, std::ios::beg);
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    throw std::runtime_error("failed to read direct KFD pipeline fixture");
  }
  return data;
}

} // namespace

int main() {
  try {
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      std::printf("direct_kfd_pipeline: skipped, no GPU devices\n");
      return 0;
    }

    lrrt::Device device = runtime.open_device(0);
    lrrt::Queue kernel_queue(device);
    constexpr size_t byte_count = kElementCount * sizeof(float);
    lrrt::PinnedHostBuffer input_a(device, byte_count);
    lrrt::PinnedHostBuffer input_b(device, byte_count);
    lrrt::PinnedHostBuffer output(device, byte_count);
    lrrt::DeviceBuffer device_a(device, byte_count);
    lrrt::DeviceBuffer device_b(device, byte_count);
    lrrt::DeviceBuffer device_output(device, byte_count);

    auto *input_a_values = static_cast<float *>(input_a.data());
    auto *input_b_values = static_cast<float *>(input_b.data());
    auto *output_values = static_cast<float *>(output.data());
    for (size_t index = 0; index < kElementCount; ++index) {
      input_a_values[index] = static_cast<float>(index) + 0.25F;
      input_b_values[index] = static_cast<float>(index * 2U) + 0.5F;
      output_values[index] = 0.0F;
    }

    std::vector<uint8_t> hsaco = read_file(DIRECT_KFD_PIPELINE_HSACO);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel("vector_add");
    const VectorAddArguments arguments{
        static_cast<const float *>(device_a.data()),
        static_cast<const float *>(device_b.data()),
        static_cast<float *>(device_output.data()),
        static_cast<int32_t>(kElementCount)};
    const lr_launch_config_t config = {
        {static_cast<uint32_t>(kElementCount), 1, 1},
        {static_cast<uint32_t>(kElementCount), 1, 1},
        0};

    lr_event_t *upload_a = nullptr;
    lr_event_t *upload_b = nullptr;
    lr_event_t *kernel_complete = nullptr;
    lr_event_t *download = nullptr;
    expect_status(lr_event_create(device.get(), &upload_a), LR_SUCCESS,
                  "create upload A event");
    expect_status(lr_event_create(device.get(), &upload_b), LR_SUCCESS,
                  "create upload B event");
    expect_status(lr_event_create(device.get(), &kernel_complete), LR_SUCCESS,
                  "create kernel completion event");
    expect_status(lr_event_create(device.get(), &download), LR_SUCCESS,
                  "create download event");

    expect_status(lr_memcpy_async(device.get(), device_a.data(), input_a.data(),
                                  byte_count, LR_MEMCPY_HOST_TO_DEVICE,
                                  upload_a),
                  LR_SUCCESS, "enqueue upload A");
    expect_status(lr_memcpy_async(device.get(), device_b.data(), input_b.data(),
                                  byte_count, LR_MEMCPY_HOST_TO_DEVICE,
                                  upload_b),
                  LR_SUCCESS, "enqueue upload B");
    lr_event_t *uploads[] = {upload_a, upload_b};
    expect_status(lr_launch_on_queue_with_dependencies(
                      kernel_queue.get(), kernel.get(), &config, &arguments,
                      sizeof(arguments), uploads, 2),
                  LR_SUCCESS, "enqueue dependent kernel");
    expect_status(lr_event_record_on_queue(kernel_complete, kernel_queue.get()),
                  LR_SUCCESS, "record kernel completion");
    lr_event_t *kernel_dependency[] = {kernel_complete};
    expect_status(lr_memcpy_async_with_dependencies(
                      device.get(), output.data(), device_output.data(),
                      byte_count, LR_MEMCPY_DEVICE_TO_HOST, download,
                      kernel_dependency, 1),
                  LR_SUCCESS, "enqueue dependent download");

    expect_status(lr_event_destroy(upload_a), LR_SUCCESS,
                  "destroy consumed upload A event");
    expect_status(lr_event_destroy(upload_b), LR_SUCCESS,
                  "destroy consumed upload B event");
    expect_status(lr_event_destroy(kernel_complete), LR_SUCCESS,
                  "destroy consumed kernel event");
    expect_status(lr_event_synchronize(download), LR_SUCCESS,
                  "synchronize download");

    for (size_t index = 0; index < kElementCount; ++index) {
      const float expected = input_a_values[index] + input_b_values[index];
      if (std::fabs(output_values[index] - expected) > 0.001F) {
        throw std::runtime_error("direct KFD pipeline result mismatch");
      }
    }
    expect_status(lr_event_destroy(download), LR_SUCCESS,
                  "destroy download event");

    std::printf("direct_kfd_pipeline: h2d-kernel-d2h=ok\n");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
