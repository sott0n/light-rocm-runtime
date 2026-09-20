#ifndef LIGHT_ROCR_TRANSPORT_KFD_MEMORY_INTERNAL_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_MEMORY_INTERNAL_HPP

#include "light_rocr/transport/kfd/memory_types.hpp"
#include "light_rocr/transport/kfd/vm_types.hpp"

#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <vector>

namespace light_rocr::transport::kfd::detail {

using IoctlFunction = int (*)(int fd, unsigned long request, void *arguments);
using MmapFunction = void *(*)(void *address, size_t length, int protection,
                               int flags, int fd, off_t offset);
using MunmapFunction = int (*)(void *address, size_t length);
using MadviseFunction = int (*)(void *address, size_t length, int advice);

struct MemorySyscalls {
  IoctlFunction ioctl_function = nullptr;
  MmapFunction mmap_function = nullptr;
  MunmapFunction munmap_function = nullptr;
  MadviseFunction madvise_function = nullptr;
};

struct RawGttAllocation {
  void *reservation_address = nullptr;
  uint64_t reservation_size = 0;
  void *host_address = nullptr;
  uint64_t size = 0;
  uint64_t handle = 0;
  std::vector<uint32_t> gpu_ids;
  uint32_t mapped_device_count = 0;
  uint32_t unmapped_device_count = 0;
  bool map_complete = false;
};

struct RawGttAllocationResult {
  MemoryStatus status;
  RawGttAllocation allocation;

  explicit operator bool() const { return static_cast<bool>(status); }
};

enum class GttAllocationUsage {
  General,
  Executable,
  AqlRing,
  Doorbell,
  Eop,
};

[[nodiscard]] RawGttAllocationResult
allocate_gtt(int kfd_fd, int render_fd, const ProcessAperture &aperture,
             uint64_t size, GttAllocationUsage usage, MemorySyscalls syscalls);
[[nodiscard]] inline RawGttAllocationResult
allocate_gtt(int kfd_fd, int render_fd, const ProcessAperture &aperture,
             uint64_t size, MemorySyscalls syscalls) {
  return allocate_gtt(kfd_fd, render_fd, aperture, size,
                      GttAllocationUsage::General, syscalls);
}
[[nodiscard]] MemoryStatus map_gtt(int kfd_fd, RawGttAllocation *allocation,
                                   const std::vector<uint32_t> &gpu_ids,
                                   MemorySyscalls syscalls);
[[nodiscard]] MemoryStatus unmap_gtt(int kfd_fd, RawGttAllocation *allocation,
                                     MemorySyscalls syscalls);
[[nodiscard]] MemoryStatus release_gtt(int kfd_fd, RawGttAllocation *allocation,
                                       MemorySyscalls syscalls);

} // namespace light_rocr::transport::kfd::detail

#endif
