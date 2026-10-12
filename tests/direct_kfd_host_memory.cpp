#include "lrrt/lrrt.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef DIRECT_KFD_HOST_MEMORY_HSACO
#error "DIRECT_KFD_HOST_MEMORY_HSACO must name the test code object"
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

std::vector<unsigned char> read_hsaco() {
  std::ifstream file(DIRECT_KFD_HOST_MEMORY_HSACO,
                     std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("failed to open host-memory test HSACO");
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("host-memory test HSACO is empty");
  }
  std::vector<unsigned char> bytes(static_cast<size_t>(size));
  file.seekg(0, std::ios::beg);
  if (!file.read(reinterpret_cast<char *>(bytes.data()), size)) {
    throw std::runtime_error("failed to read host-memory test HSACO");
  }
  return bytes;
}

} // namespace

int main() {
  try {
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      std::printf("direct_kfd_host_memory: skipped, no GPU devices\n");
      return 0;
    }

    lrrt::Device device = runtime.open_device(0);
    device.reset_memory_stats();
    constexpr size_t kByteCount = kElementCount * sizeof(float);
    {
      lrrt::PinnedHostBuffer input_a(device, kByteCount);
      lrrt::PinnedHostBuffer input_b(device, kByteCount);
      lrrt::PinnedHostBuffer output(device, kByteCount);
      auto *a = static_cast<float *>(input_a.data());
      auto *b = static_cast<float *>(input_b.data());
      auto *c = static_cast<float *>(output.data());
      for (size_t index = 0; index < kElementCount; ++index) {
        a[index] = static_cast<float>(index) + 0.25F;
        b[index] = static_cast<float>(index * 2U) + 0.5F;
        c[index] = 0.0F;
      }

      const auto hsaco = read_hsaco();
      lrrt::Module module(device, hsaco);
      const lrrt::Kernel kernel = module.kernel("vector_add");
      const VectorAddArguments arguments{a, b, c,
                                         static_cast<int32_t>(kElementCount)};
      const lr_launch_config_t config = {
          {static_cast<uint32_t>(kElementCount), 1, 1},
          {static_cast<uint32_t>(kElementCount), 1, 1},
          0};
      lrrt::launch(kernel, config, arguments);
      device.synchronize();
      for (size_t index = 0; index < kElementCount; ++index) {
        if (std::fabs(c[index] - (a[index] + b[index])) > 0.001F) {
          throw std::runtime_error("GPU could not access pinned USERPTR");
        }
      }

      lrrt::DeviceBuffer device_buffer(device, kByteCount);
      lrrt::copy_to_device(device_buffer, a, kByteCount);
      std::array<float, kElementCount> copied{};
      lrrt::copy_to_host(copied.data(), device_buffer, kByteCount);
      for (size_t index = 0; index < kElementCount; ++index) {
        if (std::fabs(copied[index] - a[index]) > 0.001F) {
          throw std::runtime_error("pinned host copy result mismatch");
        }
      }

      expect_status(lr_memcpy(device.get(), device_buffer.data(), a + 62,
                              4 * sizeof(float), LR_MEMCPY_HOST_TO_DEVICE),
                    LR_ERROR_INVALID_ARGUMENT,
                    "out-of-bounds pinned host copy");
      expect_status(lr_host_free(device.get(), a + 1),
                    LR_ERROR_INVALID_ARGUMENT, "pinned host subpointer free");

      const lrrt::MemoryStats stats = device.memory_stats();
      if (stats.pinned_host_live_bytes != 3 * kByteCount ||
          stats.pinned_host_total_allocated_bytes != 3 * kByteCount ||
          stats.pinned_host_allocation_count != 3) {
        throw std::runtime_error("unexpected live pinned host statistics");
      }
    }

    const lrrt::MemoryStats stats = device.memory_stats();
    if (stats.pinned_host_live_bytes != 0 ||
        stats.pinned_host_total_freed_bytes != 3 * kByteCount ||
        stats.pinned_host_free_count != 3) {
      throw std::runtime_error("unexpected freed pinned host statistics");
    }

    std::printf("direct_kfd_host_memory: ok\n");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
