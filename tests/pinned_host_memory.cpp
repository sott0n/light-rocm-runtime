#include "lrrt/lrrt.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef LRRT_TEST_LIGHT_ROCR
#define LRRT_TEST_LIGHT_ROCR 0
#endif

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
#if LRRT_TEST_LIGHT_ROCR
      expect_status(lr_memcpy_async(device.get(), device_buffer.data(),
                                    pageable_input.data(), byte_count,
                                    LR_MEMCPY_HOST_TO_DEVICE, copy_event.get()),
                    LR_ERROR_NOT_SUPPORTED, "pageable async H2D copy");
      expect_status(lr_memcpy_async(device.get(), pageable_output.data(),
                                    device_buffer.data(), byte_count,
                                    LR_MEMCPY_DEVICE_TO_HOST, copy_event.get()),
                    LR_ERROR_NOT_SUPPORTED, "pageable async D2H copy");
#else
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
#endif
    }

    {
      lrrt::PinnedHostBuffer first_input(device, byte_count);
      lrrt::PinnedHostBuffer input(std::move(first_input));
      lrrt::PinnedHostBuffer output(device, byte_count);
      lrrt::DeviceBuffer device_buffer(device, byte_count);
      lrrt::Event copy_event(device);
      lrrt::Event download_event(device);

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

      std::fill(output_values, output_values + element_count, 0.0f);
      constexpr size_t subpointer_offset = sizeof(float);
      constexpr size_t subpointer_size = byte_count - 2 * sizeof(float);
      {
        lrrt::Event upload_event(device);
        auto *device_bytes = static_cast<unsigned char *>(device_buffer.data());
        expect_status(
            lr_memcpy_async(device.get(), device_bytes + subpointer_offset,
                            input_values + 1, subpointer_size,
                            LR_MEMCPY_HOST_TO_DEVICE, upload_event.get()),
            LR_SUCCESS, "pinned subpointer async H2D copy");
        lr_event_t *dependencies[] = {upload_event.get()};
        expect_status(lr_memcpy_async_with_dependencies(
                          device.get(), output_values + 1,
                          device_bytes + subpointer_offset, subpointer_size,
                          LR_MEMCPY_DEVICE_TO_HOST, download_event.get(),
                          dependencies, 1),
                      LR_SUCCESS, "dependent pinned subpointer async D2H copy");
      }
      download_event.synchronize();
      for (size_t i = 0; i < element_count; ++i) {
        const float expected =
            i == 0 || i + 1 == element_count ? 0.0f : input_values[i];
        if (std::fabs(output_values[i] - expected) > 0.001f) {
          throw std::runtime_error(
              "pinned subpointer async copy result mismatch");
        }
      }

      void *transient = nullptr;
      expect_status(lr_host_malloc(device.get(), byte_count, &transient),
                    LR_SUCCESS, "transient pinned host allocation");
      auto *transient_values = static_cast<float *>(transient);
      for (size_t i = 0; i < element_count; ++i) {
        transient_values[i] = static_cast<float>(i) + 0.75f;
      }
      expect_status(lr_memcpy_async(device.get(), device_buffer.data(),
                                    transient, byte_count,
                                    LR_MEMCPY_HOST_TO_DEVICE, copy_event.get()),
                    LR_SUCCESS, "transient pinned async H2D copy");
      expect_status(lr_host_free(device.get(), transient), LR_SUCCESS,
                    "free in-flight pinned host allocation");
      copy_event.synchronize();
      std::vector<float> transient_result(element_count, 0.0f);
      lrrt::copy_to_host(transient_result.data(), device_buffer, byte_count);
      for (size_t i = 0; i < element_count; ++i) {
        const float expected = static_cast<float>(i) + 0.75f;
        if (std::fabs(transient_result[i] - expected) > 0.001f) {
          throw std::runtime_error("freed pinned async copy result mismatch");
        }
      }

      expect_status(lr_memcpy_async(device.get(), device_buffer.data(),
                                    input_values + 6, 4 * sizeof(float),
                                    LR_MEMCPY_HOST_TO_DEVICE, copy_event.get()),
                    LR_ERROR_INVALID_ARGUMENT,
                    "out-of-bounds pinned host copy");
      expect_status(lr_memcpy_async(device.get(), output_values + 6,
                                    device_buffer.data(), 4 * sizeof(float),
                                    LR_MEMCPY_DEVICE_TO_HOST, copy_event.get()),
                    LR_ERROR_INVALID_ARGUMENT,
                    "out-of-bounds pinned host D2H copy");
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
          stats.pinned_host_peak_live_bytes < 3 * byte_count ||
          stats.pinned_host_total_allocated_bytes != 3 * byte_count ||
          stats.pinned_host_total_freed_bytes != byte_count ||
          stats.pinned_host_allocation_count != 3 ||
          stats.pinned_host_free_count != 1) {
        throw std::runtime_error("unexpected live pinned host statistics");
      }

      const uint64_t async_copy_bytes = 2 * byte_count + subpointer_size;
      const uint64_t expected_copy_bytes =
          async_copy_bytes +
          (LRRT_TEST_LIGHT_ROCR ? 0 : static_cast<uint64_t>(byte_count));
      const uint64_t expected_copy_count = LRRT_TEST_LIGHT_ROCR ? 6 : 8;
      if (stats.h2d_copy_bytes != expected_copy_bytes ||
          stats.d2h_copy_bytes != expected_copy_bytes ||
          stats.d2d_copy_bytes != 0 ||
          stats.memcpy_count != expected_copy_count) {
        throw std::runtime_error("unexpected pinned copy statistics");
      }
    }

    lrrt::MemoryStats stats = device.memory_stats();
    if (stats.pinned_host_live_bytes != 0 ||
        stats.pinned_host_peak_live_bytes < 3 * byte_count ||
        stats.pinned_host_total_allocated_bytes != 3 * byte_count ||
        stats.pinned_host_total_freed_bytes != 3 * byte_count ||
        stats.pinned_host_allocation_count != 3 ||
        stats.pinned_host_free_count != 3) {
      throw std::runtime_error("unexpected freed pinned host statistics");
    }

    std::printf("pinned_host_memory: ok\n");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
