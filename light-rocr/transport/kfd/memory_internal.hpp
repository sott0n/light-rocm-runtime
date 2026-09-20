#ifndef LIGHT_ROCR_TRANSPORT_KFD_MEMORY_INTERNAL_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_MEMORY_INTERNAL_HPP

#include "light_rocr/transport/kfd/memory_types.hpp"
#include "light_rocr/transport/kfd/vm_types.hpp"

#include <cstddef>
#include <cstdint>
#include <sys/types.h>

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
};

struct RawGttAllocationResult {
  MemoryStatus status;
  RawGttAllocation allocation;

  explicit operator bool() const { return static_cast<bool>(status); }
};

[[nodiscard]] RawGttAllocationResult
allocate_gtt(int kfd_fd, int render_fd, const ProcessAperture &aperture,
             uint64_t size, MemorySyscalls syscalls);
[[nodiscard]] MemoryStatus release_gtt(int kfd_fd, RawGttAllocation *allocation,
                                       MemorySyscalls syscalls);

} // namespace light_rocr::transport::kfd::detail

#endif
