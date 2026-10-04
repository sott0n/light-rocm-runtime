#ifndef LIGHT_ROCR_TRANSPORT_KFD_MEMORY_INTERNAL_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_MEMORY_INTERNAL_HPP

#include "light_rocr/transport/kfd/memory_types.hpp"
#include "light_rocr/transport/kfd/vm_types.hpp"

#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <vector>

namespace light_rocr::transport::kfd::detail {

inline constexpr uint64_t kScratchBackingAlignment = 64 * 1024;

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

struct RawMemoryAllocation {
  void *reservation_address = nullptr;
  uint64_t reservation_size = 0;
  void *host_address = nullptr;
  uint64_t gpu_address = 0;
  uint64_t size = 0;
  uint64_t handle = 0;
  std::vector<uint32_t> gpu_ids;
  uint32_t mapped_device_count = 0;
  uint32_t unmapped_device_count = 0;
  bool map_complete = false;
};

struct RawMemoryAllocationResult {
  MemoryStatus status;
  RawMemoryAllocation allocation;

  explicit operator bool() const { return static_cast<bool>(status); }
};

struct RawScratchAllocationResult {
  MemoryStatus status;
  RawMemoryAllocation allocation;
  uint64_t gpu_address = 0;
  bool integrated = false;
  bool gpu_mapped = false;

  explicit operator bool() const { return static_cast<bool>(status); }
};

enum class ScratchReservationResult {
  Acquired,
  AlreadyReserved,
  InvalidState,
};

enum class MemoryAllocationUsage {
  General,
  Vram,
  DeviceVram,
  Executable,
  AqlRing,
  Doorbell,
  Eop,
};

[[nodiscard]] RawMemoryAllocationResult
allocate_memory(int kfd_fd, int render_fd, const ProcessAperture &aperture,
                uint64_t size, MemoryAllocationUsage usage,
                MemorySyscalls syscalls);
[[nodiscard]] inline RawMemoryAllocationResult
allocate_memory(int kfd_fd, int render_fd, const ProcessAperture &aperture,
                uint64_t size, MemorySyscalls syscalls) {
  return allocate_memory(kfd_fd, render_fd, aperture, size,
                         MemoryAllocationUsage::General, syscalls);
}
[[nodiscard]] MemoryStatus map_memory(int kfd_fd,
                                      RawMemoryAllocation *allocation,
                                      const std::vector<uint32_t> &gpu_ids,
                                      MemorySyscalls syscalls);
[[nodiscard]] MemoryStatus unmap_memory(int kfd_fd,
                                        RawMemoryAllocation *allocation,
                                        MemorySyscalls syscalls);
[[nodiscard]] MemoryStatus release_memory(int kfd_fd,
                                          RawMemoryAllocation *allocation,
                                          MemorySyscalls syscalls);
[[nodiscard]] RawScratchAllocationResult
allocate_scratch(int kfd_fd, const ProcessAperture &aperture, uint64_t size,
                 bool integrated, MemorySyscalls syscalls);
[[nodiscard]] ScratchReservationResult
acquire_scratch_reservation(const std::shared_ptr<KfdState> &state,
                            uint32_t gpu_id);
void release_scratch_reservation(const std::shared_ptr<KfdState> &state,
                                 uint32_t gpu_id);

} // namespace light_rocr::transport::kfd::detail

#endif
