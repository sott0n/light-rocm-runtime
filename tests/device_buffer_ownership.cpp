#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdio.h>

#include <stdexcept>

namespace {

void expect_stats(const lrrt::MemoryStats &stats, uint64_t live_bytes,
                  uint64_t total_allocated_bytes, uint64_t total_freed_bytes,
                  uint64_t allocation_count, uint64_t free_count,
                  const char *scenario) {
  if (stats.live_bytes == live_bytes &&
      stats.peak_live_bytes >= stats.live_bytes &&
      stats.total_allocated_bytes == total_allocated_bytes &&
      stats.total_freed_bytes == total_freed_bytes &&
      stats.allocation_count == allocation_count &&
      stats.free_count == free_count) {
    return;
  }
  throw std::runtime_error(std::string("unexpected memory stats for ") +
                           scenario);
}

} // namespace

int main() {
  try {
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      printf("device_buffer_ownership: skipped, no GPU devices\n");
      return 0;
    }

    lrrt::Device device = runtime.open_device(0);
    device.reset_memory_stats();

    const int n = 8;
    float input[n] = {};
    float output[n] = {};
    for (int i = 0; i < n; ++i) {
      input[i] = static_cast<float>(i) + 0.5f;
    }

    {
      lrrt::DeviceBuffer owner(device, sizeof(input));
      lrrt::MemoryStats stats = device.memory_stats();
      expect_stats(stats, sizeof(input), sizeof(input), 0, 1, 0,
                   "owner allocation");

      {
        lrrt::DeviceBuffer view =
            lrrt::DeviceBuffer::view(device.get(), owner.data(), owner.size());
        stats = device.memory_stats();
        expect_stats(stats, sizeof(input), sizeof(input), 0, 1, 0,
                     "view creation");

        lrrt::copy_to_device(view, input, sizeof(input));
        lrrt::copy_to_host(output, view, sizeof(output));
      }

      stats = device.memory_stats();
      expect_stats(stats, sizeof(input), sizeof(input), 0, 1, 0,
                   "view destruction");

      for (int i = 0; i < n; ++i) {
        if (fabsf(output[i] - input[i]) > 0.001f) {
          throw std::runtime_error("view copy result mismatch");
        }
      }

      float owner_output[n] = {};
      lrrt::copy_to_host(owner_output, owner, sizeof(owner_output));
      for (int i = 0; i < n; ++i) {
        if (fabsf(owner_output[i] - input[i]) > 0.001f) {
          throw std::runtime_error("owner allocation was not alive after view");
        }
      }
    }

    lrrt::MemoryStats stats = device.memory_stats();
    expect_stats(stats, 0, sizeof(input), sizeof(input), 1, 1,
                 "owner destruction");

    printf("device_buffer_ownership: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
