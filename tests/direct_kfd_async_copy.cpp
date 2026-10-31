#include "lrrt/lrrt.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect_status(lr_status_t actual, lr_status_t expected,
                   const char *operation) {
  if (actual != expected) {
    throw std::runtime_error(std::string(operation) + " returned " +
                             lr_status_string(actual) + ", expected " +
                             lr_status_string(expected));
  }
}

} // namespace

int main() {
  try {
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      std::printf("direct_kfd_async_copy: skipped, no GPU devices\n");
      return 0;
    }

    lrrt::Device device = runtime.open_device(0);
    device.reset_memory_stats();
    constexpr size_t element_count = 257;
    constexpr size_t byte_count = element_count * sizeof(float);
    lrrt::PinnedHostBuffer input(device, byte_count);
    lrrt::PinnedHostBuffer output(device, byte_count);
    lrrt::DeviceBuffer first(device, byte_count);
    lrrt::DeviceBuffer second(device, byte_count);
    lrrt::Event upload(device);
    lrrt::Event device_copy(device);
    lrrt::Event download(device);

    auto *input_values = static_cast<float *>(input.data());
    auto *output_values = static_cast<float *>(output.data());
    for (size_t index = 0; index < element_count; ++index) {
      input_values[index] = static_cast<float>(index) + 0.25F;
      output_values[index] = 0.0F;
    }

    expect_status(lr_memcpy_async(device.get(), first.data(), input.data(),
                                  byte_count, LR_MEMCPY_HOST_TO_DEVICE,
                                  upload.get()),
                  LR_SUCCESS, "async H2D copy");
    expect_status(lr_memcpy_async(device.get(), second.data(), first.data(),
                                  byte_count, LR_MEMCPY_DEVICE_TO_DEVICE,
                                  device_copy.get()),
                  LR_SUCCESS, "async D2D copy");
    expect_status(lr_memcpy_async(device.get(), output.data(), second.data(),
                                  byte_count, LR_MEMCPY_DEVICE_TO_HOST,
                                  download.get()),
                  LR_SUCCESS, "async D2H copy");
    download.synchronize();
    upload.synchronize();
    device_copy.synchronize();

    for (size_t index = 0; index < element_count; ++index) {
      if (std::fabs(output_values[index] - input_values[index]) > 0.001F) {
        throw std::runtime_error("asynchronous copy chain result mismatch");
      }
    }
    if (lrrt::duration_ns(upload) == 0 || lrrt::duration_ns(device_copy) == 0 ||
        lrrt::duration_ns(download) == 0) {
      throw std::runtime_error("asynchronous copy duration was not recorded");
    }

    std::vector<float> pageable(element_count, 1.0F);
    expect_status(lr_memcpy_async(device.get(), first.data(), pageable.data(),
                                  byte_count, LR_MEMCPY_HOST_TO_DEVICE,
                                  upload.get()),
                  LR_ERROR_NOT_SUPPORTED, "pageable async H2D copy");
    lr_event_t *dependencies[] = {upload.get()};
    expect_status(lr_memcpy_async_with_dependencies(
                      device.get(), second.data(), first.data(), byte_count,
                      LR_MEMCPY_DEVICE_TO_DEVICE, device_copy.get(),
                      dependencies, 1),
                  LR_ERROR_NOT_SUPPORTED, "explicit-dependency async copy");

    const lrrt::MemoryStats stats = device.memory_stats();
    if (stats.h2d_copy_bytes != byte_count ||
        stats.d2d_copy_bytes != byte_count ||
        stats.d2h_copy_bytes != byte_count || stats.memcpy_count != 3) {
      throw std::runtime_error("unexpected asynchronous copy statistics");
    }

    std::printf("direct_kfd_async_copy: h2d=ok d2d=ok d2h=ok\n");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
