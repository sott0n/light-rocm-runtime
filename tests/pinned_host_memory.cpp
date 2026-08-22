#include "lrrt/lrrt.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void expect_status(lr_status_t actual, lr_status_t expected,
                   const char *operation) {
  if (actual == expected) {
    return;
  }
  throw std::runtime_error(std::string(operation) + " returned " +
                           lr_status_string(actual) + ", expected " +
                           lr_status_string(expected));
}

} // namespace

int main() {
  try {
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      std::printf("pinned_host_memory: skipped, no GPU devices\n");
      return 0;
    }

    lrrt::Device device = runtime.open_device(0);
    device.reset_memory_stats();

    constexpr size_t element_count = 8;
    constexpr size_t byte_count = element_count * sizeof(float);
    {
      std::vector<float> pageable_input(element_count);
      std::vector<float> pageable_output(element_count, 0.0f);
      for (size_t i = 0; i < element_count; ++i) {
        pageable_input[i] = static_cast<float>(i) + 0.5f;
      }
      lrrt::DeviceBuffer device_buffer(device, byte_count);
      lrrt::Event copy_event(device);
      lrrt::copy_to_device_async(device_buffer, pageable_input.data(),
                                 byte_count, copy_event);
      copy_event.synchronize();
      lrrt::copy_to_host_async(pageable_output.data(), device_buffer,
                               byte_count, copy_event);
      copy_event.synchronize();
      for (size_t i = 0; i < element_count; ++i) {
        if (std::fabs(pageable_output[i] - pageable_input[i]) > 0.001f) {
          throw std::runtime_error("pageable async copy result mismatch");
        }
      }
    }

    {
      lrrt::PinnedHostBuffer first_input(device, byte_count);
      lrrt::PinnedHostBuffer input(std::move(first_input));
      lrrt::PinnedHostBuffer output(device, byte_count);
      lrrt::DeviceBuffer device_buffer(device, byte_count);
      lrrt::Event copy_event(device);

      auto *input_values = static_cast<float *>(input.data());
      auto *output_values = static_cast<float *>(output.data());
      for (size_t i = 0; i < element_count; ++i) {
        input_values[i] = static_cast<float>(i) + 0.25f;
        output_values[i] = 0.0f;
      }

      lrrt::copy_to_device_async(device_buffer, input.data(), input.size(),
                                 copy_event);
      copy_event.synchronize();
      lrrt::copy_to_host_async(output.data(), device_buffer, output.size(),
                               copy_event);
      copy_event.synchronize();

      for (size_t i = 0; i < element_count; ++i) {
        if (std::fabs(output_values[i] - input_values[i]) > 0.001f) {
          throw std::runtime_error("pinned async copy result mismatch");
        }
      }

      expect_status(lr_memcpy_async(device.get(), device_buffer.data(),
                                    input_values + 6, 4 * sizeof(float),
                                    LR_MEMCPY_HOST_TO_DEVICE, copy_event.get()),
                    LR_ERROR_INVALID_ARGUMENT,
                    "out-of-bounds pinned host copy");
      expect_status(lr_memcpy(device.get(), device_buffer.data(),
                              input_values + 6, 4 * sizeof(float),
                              LR_MEMCPY_HOST_TO_DEVICE),
                    LR_ERROR_INVALID_ARGUMENT,
                    "out-of-bounds synchronous pinned host copy");
      expect_status(lr_host_free(device.get(), input_values + 1),
                    LR_ERROR_INVALID_ARGUMENT, "pinned host subpointer free");

      if (runtime.device_count() > 1) {
        lrrt::Device other_device = runtime.open_device(1);
        expect_status(lr_host_free(other_device.get(), input.data()),
                      LR_ERROR_INVALID_ARGUMENT,
                      "pinned host wrong-device free");
      }

      lrrt::MemoryStats stats = device.memory_stats();
      if (stats.pinned_host_live_bytes != 2 * byte_count ||
          stats.pinned_host_peak_live_bytes < 2 * byte_count ||
          stats.pinned_host_total_allocated_bytes != 2 * byte_count ||
          stats.pinned_host_allocation_count != 2 ||
          stats.pinned_host_free_count != 0) {
        throw std::runtime_error("unexpected live pinned host statistics");
      }
    }

    lrrt::MemoryStats stats = device.memory_stats();
    if (stats.pinned_host_live_bytes != 0 ||
        stats.pinned_host_peak_live_bytes < 2 * byte_count ||
        stats.pinned_host_total_allocated_bytes != 2 * byte_count ||
        stats.pinned_host_total_freed_bytes != 2 * byte_count ||
        stats.pinned_host_allocation_count != 2 ||
        stats.pinned_host_free_count != 2) {
      throw std::runtime_error("unexpected freed pinned host statistics");
    }

    std::printf("pinned_host_memory: ok\n");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
