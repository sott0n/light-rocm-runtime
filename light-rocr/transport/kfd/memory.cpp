#include "light_rocr/transport/kfd/memory.hpp"

#include "memory_internal.hpp"
#include "session_state.hpp"

#include <linux/kfd_ioctl.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <system_error>
#include <utility>
#include <vector>

namespace light_rocr::transport::kfd {
namespace {

constexpr uint64_t kGuardPageCount = 2;

MemoryStatus system_failure(MemoryError error, int system_error,
                            const std::string &operation) {
  const std::error_code code(system_error, std::generic_category());
  return {error, system_error,
          operation + " failed: " + code.message() + " (" +
              std::to_string(system_error) + ")"};
}

int real_ioctl(int fd, unsigned long request, void *arguments) {
  return ::ioctl(fd, request, arguments);
}

void *real_mmap(void *address, size_t length, int protection, int flags, int fd,
                off_t offset) {
  return ::mmap(address, length, protection, flags, fd, offset);
}

int real_munmap(void *address, size_t length) {
  return ::munmap(address, length);
}

int real_madvise(void *address, size_t length, int advice) {
  return ::madvise(address, length, advice);
}

detail::MemorySyscalls real_syscalls() {
  return {real_ioctl, real_mmap, real_munmap, real_madvise};
}

bool invoke_ioctl(int fd, unsigned long request, void *arguments,
                  detail::IoctlFunction ioctl_function, int *system_error) {
  int result = 0;
  do {
    result = ioctl_function(fd, request, arguments);
  } while (result < 0 && errno == EINTR);
  if (result < 0) {
    *system_error = errno;
    return false;
  }
  return true;
}

bool valid_syscalls(detail::MemorySyscalls syscalls) {
  return syscalls.ioctl_function != nullptr &&
         syscalls.mmap_function != nullptr &&
         syscalls.munmap_function != nullptr &&
         syscalls.madvise_function != nullptr;
}

} // namespace

namespace detail {

ScratchReservationResult
acquire_scratch_reservation(const std::shared_ptr<KfdState> &state,
                            uint32_t gpu_id) {
  if (state == nullptr || gpu_id == 0) {
    return ScratchReservationResult::InvalidState;
  }
  const std::lock_guard<std::mutex> lock(state->mutex);
  const auto device = state->device_vms.find(gpu_id);
  if (device == state->device_vms.end() || !device->second.acquired ||
      !device->second.aperture_valid) {
    return ScratchReservationResult::InvalidState;
  }
  if (device->second.scratch_reserved) {
    return ScratchReservationResult::AlreadyReserved;
  }
  device->second.scratch_reserved = true;
  return ScratchReservationResult::Acquired;
}

void release_scratch_reservation(const std::shared_ptr<KfdState> &state,
                                 uint32_t gpu_id) {
  if (state == nullptr || gpu_id == 0) {
    return;
  }
  const std::lock_guard<std::mutex> lock(state->mutex);
  const auto device = state->device_vms.find(gpu_id);
  if (device != state->device_vms.end()) {
    device->second.scratch_reserved = false;
  }
}

namespace {

RawGttAllocationResult rollback_gtt(MemoryStatus status, int kfd_fd,
                                    RawGttAllocation allocation,
                                    MemorySyscalls syscalls) {
  const MemoryStatus cleanup = release_gtt(kfd_fd, &allocation, syscalls);
  if (!cleanup) {
    status.message += "; cleanup failed: " + cleanup.message;
  }
  return {std::move(status), std::move(allocation)};
}

RawScratchAllocationResult rollback_scratch(MemoryStatus status, int kfd_fd,
                                            RawGttAllocation allocation,
                                            uint64_t gpu_address,
                                            bool integrated,
                                            MemorySyscalls syscalls) {
  const MemoryStatus cleanup = release_gtt(kfd_fd, &allocation, syscalls);
  if (!cleanup) {
    status.message += "; cleanup failed: " + cleanup.message;
  }
  return {std::move(status), std::move(allocation), gpu_address, integrated,
          false};
}

} // namespace

RawGttAllocationResult allocate_gtt(int kfd_fd, int render_fd,
                                    const ProcessAperture &aperture,
                                    uint64_t size, GttAllocationUsage usage,
                                    MemorySyscalls syscalls) {
  if (kfd_fd < 0 || (usage != GttAllocationUsage::Doorbell && render_fd < 0) ||
      aperture.gpu_id == 0 || !valid_syscalls(syscalls)) {
    return {{MemoryError::InvalidSession, 0,
             "GTT allocation requires an acquired KFD VM"},
            {}};
  }
  if (size == 0 || size % kMemoryPageSize != 0 ||
      size > std::numeric_limits<size_t>::max() ||
      (usage == GttAllocationUsage::AqlRing &&
       size > std::numeric_limits<uint64_t>::max() / 2)) {
    return {{MemoryError::InvalidSize, 0,
             "GTT allocation size must be a non-zero multiple of 4096 bytes"},
            {}};
  }

  // KFD's AQL allocation contract reserves a second ring-sized span. The
  // packet producer only addresses the first span, but KFD uses the full
  // backing size for queue-memory handling.
  const uint64_t backing_size =
      usage == GttAllocationUsage::AqlRing ? size * 2 : size;
  if (backing_size > std::numeric_limits<uint64_t>::max() -
                         kGuardPageCount * kMemoryPageSize) {
    return {{MemoryError::InvalidSize, 0,
             "GTT allocation plus guard pages exceeds the address range"},
            {}};
  }
  const uint64_t reservation_size =
      backing_size + kGuardPageCount * kMemoryPageSize;
  if (reservation_size > std::numeric_limits<size_t>::max()) {
    return {{MemoryError::InvalidSize, 0,
             "GTT allocation plus guard pages exceeds the host size range"},
            {}};
  }

  std::vector<uint32_t> gpu_ids;
  try {
    gpu_ids.push_back(aperture.gpu_id);
  } catch (const std::bad_alloc &) {
    return {{MemoryError::AllocateState, 0,
             "failed to allocate GTT GPU mapping state"},
            {}};
  }
  void *reservation = syscalls.mmap_function(
      nullptr, static_cast<size_t>(reservation_size), PROT_NONE,
      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (reservation == MAP_FAILED) {
    return {system_failure(MemoryError::ReserveVa, errno,
                           "mmap(GTT address reservation)"),
            {}};
  }

  const uint64_t reservation_address =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(reservation));
  if (reservation_address >
      std::numeric_limits<uint64_t>::max() - kMemoryPageSize) {
    return rollback_gtt({MemoryError::ReserveVa, 0,
                         "reserved CPU VA overflows the host address range"},
                        kfd_fd,
                        {reservation, reservation_size, nullptr, size, 0,
                         std::move(gpu_ids), 0, 0, false},
                        syscalls);
  }
  const uint64_t host_address = reservation_address + kMemoryPageSize;
  const bool range_overflows =
      host_address > std::numeric_limits<uint64_t>::max() - (backing_size - 1);
  if (range_overflows || host_address < aperture.gpuvm_base ||
      host_address + backing_size - 1 > aperture.gpuvm_limit) {
    return rollback_gtt({MemoryError::ReserveVa, 0,
                         "reserved CPU VA is outside the KFD GPUVM aperture"},
                        kfd_fd,
                        {reservation, reservation_size, nullptr, size, 0,
                         std::move(gpu_ids), 0, 0, false},
                        syscalls);
  }

  kfd_ioctl_alloc_memory_of_gpu_args arguments{};
  arguments.va_addr = host_address;
  arguments.size = backing_size;
  arguments.gpu_id = aperture.gpu_id;
  if (usage == GttAllocationUsage::Doorbell) {
    arguments.flags = static_cast<uint32_t>(
        KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
        KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
        KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE);
  } else if (usage == GttAllocationUsage::Eop) {
    arguments.flags = static_cast<uint32_t>(
        KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
        KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
        KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE);
  } else {
    arguments.flags = static_cast<uint32_t>(
        KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
        KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
        KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE);
  }
  if (usage == GttAllocationUsage::Executable) {
    arguments.flags |= KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;
  } else if (usage == GttAllocationUsage::AqlRing) {
    arguments.flags |= KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
                       KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM |
                       KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED;
  }
  int system_error = 0;
  if (!invoke_ioctl(kfd_fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &arguments,
                    syscalls.ioctl_function, &system_error)) {
    return rollback_gtt(system_failure(MemoryError::AllocateGtt, system_error,
                                       "AMDKFD_IOC_ALLOC_MEMORY_OF_GPU(GTT)"),
                        kfd_fd,
                        {reservation, reservation_size, nullptr, size, 0,
                         std::move(gpu_ids), 0, 0, false},
                        syscalls);
  }
  if (arguments.handle == 0) {
    return rollback_gtt({MemoryError::AllocateGtt, 0,
                         "KFD returned an invalid zero GTT handle"},
                        kfd_fd,
                        {reservation, reservation_size, nullptr, size, 0,
                         std::move(gpu_ids), 0, 0, false},
                        syscalls);
  }

  RawGttAllocation allocation{reservation,
                              reservation_size,
                              nullptr,
                              size,
                              arguments.handle,
                              std::move(gpu_ids),
                              0,
                              0,
                              false};

  void *host = reinterpret_cast<void *>(static_cast<uintptr_t>(host_address));
  if (usage != GttAllocationUsage::Doorbell) {
    if (arguments.mmap_offset >
        static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
      return rollback_gtt(
          {MemoryError::MapHost, 0,
           "KFD returned a GTT mmap offset outside the host off_t range"},
          kfd_fd, std::move(allocation), syscalls);
    }
    const int protection =
        usage == GttAllocationUsage::Eop ? PROT_NONE : PROT_READ | PROT_WRITE;
    void *mapped = syscalls.mmap_function(
        host, static_cast<size_t>(size), protection, MAP_SHARED | MAP_FIXED,
        render_fd, static_cast<off_t>(arguments.mmap_offset));
    if (mapped == MAP_FAILED) {
      return rollback_gtt(system_failure(MemoryError::MapHost, errno,
                                         "mmap(GTT render-node mapping)"),
                          kfd_fd, std::move(allocation), syscalls);
    }
  }
  allocation.host_address = host;

  if (syscalls.madvise_function(host, static_cast<size_t>(size),
                                MADV_DONTFORK) != 0) {
    return rollback_gtt(system_failure(MemoryError::AdviseDontFork, errno,
                                       "madvise(MADV_DONTFORK)"),
                        kfd_fd, std::move(allocation), syscalls);
  }

  if (usage != GttAllocationUsage::Doorbell) {
    const MemoryStatus gpu_mapped =
        map_gtt(kfd_fd, &allocation, allocation.gpu_ids, syscalls);
    if (!gpu_mapped) {
      return rollback_gtt(gpu_mapped, kfd_fd, std::move(allocation), syscalls);
    }
  }

  return {{}, std::move(allocation)};
}

RawScratchAllocationResult allocate_scratch(int kfd_fd,
                                            const ProcessAperture &aperture,
                                            uint64_t size, bool integrated,
                                            MemorySyscalls syscalls) {
  if (kfd_fd < 0 || aperture.gpu_id == 0 || !valid_syscalls(syscalls)) {
    return {{MemoryError::InvalidSession, 0,
             "scratch allocation requires an acquired KFD VM"},
            {}};
  }
  if (size == 0 || size % kMemoryPageSize != 0 ||
      size > std::numeric_limits<uint64_t>::max() -
                 (kScratchBackingAlignment - 1U)) {
    return {{MemoryError::InvalidSize, 0,
             "scratch size must be a non-zero multiple of 4096 bytes"},
            {}};
  }
  const uint64_t aligned_size =
      (size + kScratchBackingAlignment - 1U) & ~(kScratchBackingAlignment - 1U);
  if (aligned_size > std::numeric_limits<uint64_t>::max() -
                         kScratchBackingAlignment - 2U * kMemoryPageSize) {
    return {{MemoryError::InvalidSize, 0,
             "scratch reservation size exceeds the address range"},
            {}};
  }
  const uint64_t reservation_size =
      aligned_size + kScratchBackingAlignment + 2U * kMemoryPageSize;
  if (reservation_size > std::numeric_limits<size_t>::max()) {
    return {{MemoryError::InvalidSize, 0,
             "scratch reservation exceeds the host size range"},
            {}};
  }

  std::vector<uint32_t> gpu_ids;
  if (!integrated) {
    try {
      gpu_ids.push_back(aperture.gpu_id);
    } catch (const std::bad_alloc &) {
      return {{MemoryError::AllocateState, 0,
               "failed to allocate scratch GPU mapping state"},
              {}};
    }
  }

  void *reservation = syscalls.mmap_function(
      nullptr, static_cast<size_t>(reservation_size), PROT_NONE,
      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (reservation == MAP_FAILED) {
    return {system_failure(MemoryError::ReserveVa, errno,
                           "mmap(scratch address reservation)"),
            {}};
  }

  const uint64_t reservation_address =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(reservation));
  if (reservation_address >
      std::numeric_limits<uint64_t>::max() - reservation_size) {
    return rollback_scratch(
        {MemoryError::ReserveVa, 0,
         "scratch reservation overflows the host address range"},
        kfd_fd,
        {reservation, reservation_size, nullptr, aligned_size, 0,
         std::move(gpu_ids), 0, 0, false},
        0, integrated, syscalls);
  }
  const uint64_t first_usable = reservation_address + kMemoryPageSize;
  const uint64_t gpu_address = (first_usable + kScratchBackingAlignment - 1U) &
                               ~(kScratchBackingAlignment - 1U);
  const uint64_t reservation_end = reservation_address + reservation_size;
  if (gpu_address < aperture.gpuvm_base || gpu_address > aperture.gpuvm_limit ||
      aligned_size - 1U > aperture.gpuvm_limit - gpu_address ||
      gpu_address > reservation_end ||
      reservation_end - gpu_address < kMemoryPageSize ||
      aligned_size > reservation_end - gpu_address - kMemoryPageSize) {
    return rollback_scratch(
        {MemoryError::ReserveVa, 0,
         "64 KiB-aligned scratch VA is outside the KFD GPUVM aperture"},
        kfd_fd,
        {reservation, reservation_size, nullptr, aligned_size, 0,
         std::move(gpu_ids), 0, 0, false},
        gpu_address, integrated, syscalls);
  }

  void *backing_address =
      reinterpret_cast<void *>(static_cast<uintptr_t>(gpu_address));
  RawGttAllocation allocation{reservation,
                              reservation_size,
                              backing_address,
                              aligned_size,
                              0,
                              std::move(gpu_ids),
                              0,
                              0,
                              false};
  if (integrated) {
    void *mapped = syscalls.mmap_function(
        backing_address, static_cast<size_t>(aligned_size),
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (mapped == MAP_FAILED) {
      return rollback_scratch(
          system_failure(MemoryError::MapHost, errno,
                         "mmap(integrated scratch backing)"),
          kfd_fd, std::move(allocation), gpu_address, integrated, syscalls);
    }
  }
  if (syscalls.madvise_function(backing_address,
                                static_cast<size_t>(aligned_size),
                                MADV_DONTFORK) != 0) {
    return rollback_scratch(system_failure(MemoryError::AdviseDontFork, errno,
                                           "madvise(scratch MADV_DONTFORK)"),
                            kfd_fd, std::move(allocation), gpu_address,
                            integrated, syscalls);
  }

  kfd_ioctl_set_scratch_backing_va_args scratch_arguments{};
  scratch_arguments.va_addr = gpu_address >> 16U;
  scratch_arguments.gpu_id = aperture.gpu_id;
  int system_error = 0;
  if (!invoke_ioctl(kfd_fd, AMDKFD_IOC_SET_SCRATCH_BACKING_VA,
                    &scratch_arguments, syscalls.ioctl_function,
                    &system_error)) {
    return rollback_scratch(
        system_failure(MemoryError::SetScratchBacking, system_error,
                       "AMDKFD_IOC_SET_SCRATCH_BACKING_VA"),
        kfd_fd, std::move(allocation), gpu_address, integrated, syscalls);
  }

  if (integrated) {
    return {{}, std::move(allocation), gpu_address, true, true};
  }

  kfd_ioctl_alloc_memory_of_gpu_args arguments{};
  arguments.va_addr = gpu_address;
  arguments.size = aligned_size;
  arguments.gpu_id = aperture.gpu_id;
  arguments.flags = static_cast<uint32_t>(KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
                                          KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE);
  if (!invoke_ioctl(kfd_fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &arguments,
                    syscalls.ioctl_function, &system_error)) {
    return rollback_scratch(
        system_failure(MemoryError::AllocateScratch, system_error,
                       "AMDKFD_IOC_ALLOC_MEMORY_OF_GPU(scratch)"),
        kfd_fd, std::move(allocation), gpu_address, integrated, syscalls);
  }
  if (arguments.handle == 0) {
    return rollback_scratch({MemoryError::AllocateScratch, 0,
                             "KFD returned an invalid zero scratch handle"},
                            kfd_fd, std::move(allocation), gpu_address,
                            integrated, syscalls);
  }
  allocation.handle = arguments.handle;

  const MemoryStatus mapped =
      map_gtt(kfd_fd, &allocation, allocation.gpu_ids, syscalls);
  if (!mapped) {
    return rollback_scratch(mapped, kfd_fd, std::move(allocation), gpu_address,
                            integrated, syscalls);
  }
  return {{}, std::move(allocation), gpu_address, false, true};
}

MemoryStatus map_gtt(int kfd_fd, RawGttAllocation *allocation,
                     const std::vector<uint32_t> &gpu_ids,
                     MemorySyscalls syscalls) {
  if (kfd_fd < 0 || allocation == nullptr || allocation->handle == 0 ||
      !valid_syscalls(syscalls)) {
    return {MemoryError::InvalidSession, 0,
            "GTT GPU mapping requires an allocated KFD buffer"};
  }
  if (gpu_ids.empty() ||
      gpu_ids.size() > std::numeric_limits<uint32_t>::max()) {
    return {MemoryError::InvalidNode, 0,
            "GTT GPU mapping requires at least one GPU ID"};
  }
  for (size_t index = 0; index < gpu_ids.size(); ++index) {
    if (gpu_ids[index] == 0) {
      return {MemoryError::InvalidNode, 0,
              "GTT GPU mapping received an invalid zero GPU ID"};
    }
    for (size_t previous = 0; previous < index; ++previous) {
      if (gpu_ids[previous] == gpu_ids[index]) {
        return {MemoryError::InvalidNode, 0,
                "GTT GPU mapping received a duplicate GPU ID"};
      }
    }
  }
  if (allocation->gpu_ids.empty()) {
    try {
      allocation->gpu_ids = gpu_ids;
    } catch (const std::bad_alloc &) {
      return {MemoryError::AllocateState, 0,
              "failed to allocate GTT GPU mapping state"};
    }
  } else if (allocation->gpu_ids != gpu_ids ||
             allocation->unmapped_device_count != 0) {
    return {MemoryError::MapToGpu, 0,
            "GTT GPU mapping retry does not match the pending mapping"};
  }

  const uint32_t device_count = static_cast<uint32_t>(gpu_ids.size());
  if (allocation->mapped_device_count > device_count) {
    return {MemoryError::MapToGpu, 0,
            "GTT GPU mapping retained invalid progress"};
  }
  if (allocation->mapped_device_count == device_count &&
      allocation->map_complete) {
    return {};
  }

  kfd_ioctl_map_memory_to_gpu_args arguments{};
  arguments.handle = allocation->handle;
  arguments.device_ids_array_ptr = static_cast<uint64_t>(
      reinterpret_cast<uintptr_t>(allocation->gpu_ids.data()));
  arguments.n_devices = device_count;
  arguments.n_success = allocation->mapped_device_count;
  const uint32_t previous_success = arguments.n_success;
  allocation->map_complete = false;
  int system_error = 0;
  const bool succeeded =
      invoke_ioctl(kfd_fd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &arguments,
                   syscalls.ioctl_function, &system_error);
  if (arguments.n_success > device_count) {
    allocation->mapped_device_count = device_count;
    return {MemoryError::MapToGpu, succeeded ? 0 : system_error,
            "AMDKFD_IOC_MAP_MEMORY_TO_GPU returned invalid progress"};
  }
  if (arguments.n_success < previous_success) {
    return {MemoryError::MapToGpu, succeeded ? 0 : system_error,
            "AMDKFD_IOC_MAP_MEMORY_TO_GPU returned non-monotonic progress"};
  }
  allocation->mapped_device_count = arguments.n_success;
  if (!succeeded) {
    return system_failure(MemoryError::MapToGpu, system_error,
                          "AMDKFD_IOC_MAP_MEMORY_TO_GPU(GTT)");
  }
  if (arguments.n_success != device_count) {
    return {MemoryError::MapToGpu, 0,
            "AMDKFD_IOC_MAP_MEMORY_TO_GPU completed without mapping every GPU"};
  }
  allocation->map_complete = true;
  return {};
}

MemoryStatus unmap_gtt(int kfd_fd, RawGttAllocation *allocation,
                       MemorySyscalls syscalls) {
  if (allocation == nullptr || allocation->mapped_device_count == 0) {
    if (allocation != nullptr) {
      allocation->gpu_ids.clear();
      allocation->unmapped_device_count = 0;
      allocation->map_complete = false;
    }
    return {};
  }
  if (kfd_fd < 0 || allocation->handle == 0 || !valid_syscalls(syscalls)) {
    return {MemoryError::InvalidSession, 0,
            "GTT GPU unmapping requires an allocated KFD buffer"};
  }
  if (allocation->mapped_device_count > allocation->gpu_ids.size() ||
      allocation->unmapped_device_count > allocation->mapped_device_count) {
    return {MemoryError::UnmapFromGpu, 0,
            "GTT GPU unmapping retained invalid progress"};
  }
  kfd_ioctl_unmap_memory_from_gpu_args arguments{};
  arguments.handle = allocation->handle;
  arguments.device_ids_array_ptr = static_cast<uint64_t>(
      reinterpret_cast<uintptr_t>(allocation->gpu_ids.data()));
  arguments.n_devices = allocation->mapped_device_count;
  arguments.n_success = allocation->unmapped_device_count;
  const uint32_t previous_success = arguments.n_success;
  int system_error = 0;
  const bool succeeded =
      invoke_ioctl(kfd_fd, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &arguments,
                   syscalls.ioctl_function, &system_error);
  if (arguments.n_success > allocation->mapped_device_count) {
    return {MemoryError::UnmapFromGpu, succeeded ? 0 : system_error,
            "AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU returned invalid progress"};
  }
  if (arguments.n_success < previous_success) {
    return {MemoryError::UnmapFromGpu, succeeded ? 0 : system_error,
            "AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU returned non-monotonic progress"};
  }
  allocation->unmapped_device_count = arguments.n_success;
  if (!succeeded) {
    return system_failure(MemoryError::UnmapFromGpu, system_error,
                          "AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU(GTT)");
  }
  if (arguments.n_success != allocation->mapped_device_count) {
    return {MemoryError::UnmapFromGpu, 0,
            "AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU completed without unmapping "
            "every GPU"};
  }

  allocation->gpu_ids.clear();
  allocation->mapped_device_count = 0;
  allocation->unmapped_device_count = 0;
  allocation->map_complete = false;
  return {};
}

MemoryStatus release_gtt(int kfd_fd, RawGttAllocation *allocation,
                         MemorySyscalls syscalls) {
  if (allocation == nullptr ||
      (allocation->reservation_address == nullptr && allocation->handle == 0)) {
    return {};
  }
  if (kfd_fd < 0 || !valid_syscalls(syscalls)) {
    return {MemoryError::InvalidSession, 0,
            "GTT release requires an open KFD session"};
  }
  const MemoryStatus unmapped = unmap_gtt(kfd_fd, allocation, syscalls);
  if (!unmapped) {
    return unmapped;
  }
  if (allocation->reservation_address != nullptr) {
    if (syscalls.munmap_function(
            allocation->reservation_address,
            static_cast<size_t>(allocation->reservation_size)) != 0) {
      return system_failure(MemoryError::UnmapHost, errno,
                            "munmap(GTT host mapping)");
    }
    allocation->reservation_address = nullptr;
    allocation->reservation_size = 0;
    allocation->host_address = nullptr;
  }

  if (allocation->handle != 0) {
    kfd_ioctl_free_memory_of_gpu_args arguments{};
    arguments.handle = allocation->handle;
    int system_error = 0;
    if (!invoke_ioctl(kfd_fd, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &arguments,
                      syscalls.ioctl_function, &system_error)) {
      return system_failure(MemoryError::FreeGtt, system_error,
                            "AMDKFD_IOC_FREE_MEMORY_OF_GPU(GTT)");
    }
  }
  *allocation = {};
  return {};
}

} // namespace detail

const char *memory_error_name(MemoryError error) {
  switch (error) {
  case MemoryError::None:
    return "none";
  case MemoryError::InvalidSession:
    return "invalid_session";
  case MemoryError::InvalidNode:
    return "invalid_node";
  case MemoryError::InvalidSize:
    return "invalid_size";
  case MemoryError::AllocateState:
    return "allocate_state";
  case MemoryError::AcquireVm:
    return "acquire_vm";
  case MemoryError::ReserveVa:
    return "reserve_va";
  case MemoryError::SetScratchBacking:
    return "set_scratch_backing";
  case MemoryError::AllocateGtt:
    return "allocate_gtt";
  case MemoryError::AllocateScratch:
    return "allocate_scratch";
  case MemoryError::ScratchAlreadyReserved:
    return "scratch_already_reserved";
  case MemoryError::MapHost:
    return "map_host";
  case MemoryError::AdviseDontFork:
    return "advise_dontfork";
  case MemoryError::MapToGpu:
    return "map_to_gpu";
  case MemoryError::UnmapFromGpu:
    return "unmap_from_gpu";
  case MemoryError::UnmapHost:
    return "unmap_host";
  case MemoryError::FreeGtt:
    return "free_gtt";
  }
  return "unknown";
}

GttAllocationResult
KfdSession::allocate_gtt(const runtime::Node &node, uint64_t size,
                         const std::string &dri_root) const {
  return allocate_gtt_impl(node, size, dri_root, GttUsage::General);
}

GttAllocationResult
KfdSession::allocate_executable_gtt(const runtime::Node &node, uint64_t size,
                                    const std::string &dri_root) const {
  return allocate_gtt_impl(node, size, dri_root, GttUsage::Executable);
}

ScratchAllocationResult
KfdSession::allocate_scratch(const runtime::Node &node, uint64_t size,
                             const std::string &dri_root) const {
  if (state_ == nullptr) {
    return {{MemoryError::InvalidSession, 0,
             "scratch allocation requires an open KFD session"},
            {}};
  }
  if (!node.is_gpu() || node.gpu_id == 0 || node.drm_render_minor < 0) {
    return {{MemoryError::InvalidNode, 0,
             "scratch allocation requires a GPU node with a render minor"},
            {}};
  }
  if (size == 0 || size % kMemoryPageSize != 0) {
    return {{MemoryError::InvalidSize, 0,
             "scratch size must be a non-zero multiple of 4096 bytes"},
            {}};
  }

  const VmResult acquired = acquire_vm(node, dri_root);
  if (!acquired) {
    return {{MemoryError::AcquireVm, acquired.status.system_error,
             "failed to acquire VM for scratch allocation: " +
                 acquired.status.message},
            {}};
  }

  const detail::ScratchReservationResult reserved =
      detail::acquire_scratch_reservation(state_, node.gpu_id);
  if (reserved == detail::ScratchReservationResult::InvalidState) {
    return {{MemoryError::AcquireVm, 0, "acquired KFD VM state is unavailable"},
            {}};
  }
  if (reserved == detail::ScratchReservationResult::AlreadyReserved) {
    return {{MemoryError::ScratchAlreadyReserved, 0,
             "scratch backing is already reserved for this process and GPU"},
            {}};
  }

  detail::RawScratchAllocationResult allocated = detail::allocate_scratch(
      state_->fd, acquired.aperture, size, node.integrated, real_syscalls());
  detail::RawGttAllocation &raw = allocated.allocation;
  if (!allocated && raw.reservation_address == nullptr && raw.handle == 0) {
    detail::release_scratch_reservation(state_, node.gpu_id);
    return {std::move(allocated.status), {}};
  }

  ScratchAllocation owner(
      state_, raw.reservation_address, raw.reservation_size,
      allocated.gpu_address, raw.size, raw.handle, std::move(raw.gpu_ids),
      raw.mapped_device_count, raw.unmapped_device_count, raw.map_complete,
      node.gpu_id, allocated.integrated, allocated.gpu_mapped);
  return {std::move(allocated.status), std::move(owner)};
}

GttAllocationResult KfdSession::allocate_gtt_impl(const runtime::Node &node,
                                                  uint64_t size,
                                                  const std::string &dri_root,
                                                  GttUsage usage) const {
  if (state_ == nullptr) {
    return {{MemoryError::InvalidSession, 0,
             "GTT allocation requires an open KFD session"},
            {}};
  }
  if (!node.is_gpu() || node.gpu_id == 0 || node.drm_render_minor < 0) {
    return {{MemoryError::InvalidNode, 0,
             "GTT allocation requires a GPU node with a render minor"},
            {}};
  }
  if (size == 0 || size % kMemoryPageSize != 0) {
    return {{MemoryError::InvalidSize, 0,
             "GTT allocation size must be a non-zero multiple of 4096 bytes"},
            {}};
  }

  const VmResult acquired = acquire_vm(node, dri_root);
  if (!acquired) {
    return {
        {MemoryError::AcquireVm, acquired.status.system_error,
         "failed to acquire VM for GTT allocation: " + acquired.status.message},
        {}};
  }

  int render_fd = -1;
  {
    const std::lock_guard<std::mutex> lock(state_->mutex);
    const auto device = state_->device_vms.find(node.gpu_id);
    if (device == state_->device_vms.end() || !device->second.acquired ||
        !device->second.aperture_valid) {
      return {
          {MemoryError::AcquireVm, 0, "acquired KFD VM state is unavailable"},
          {}};
    }
    render_fd = device->second.render_fd;
  }

  detail::RawGttAllocationResult allocated = detail::allocate_gtt(
      state_->fd, render_fd, acquired.aperture, size,
      usage == GttUsage::AqlRing      ? detail::GttAllocationUsage::AqlRing
      : usage == GttUsage::Doorbell   ? detail::GttAllocationUsage::Doorbell
      : usage == GttUsage::Eop        ? detail::GttAllocationUsage::Eop
      : usage == GttUsage::Executable ? detail::GttAllocationUsage::Executable
                                      : detail::GttAllocationUsage::General,
      real_syscalls());
  if (!allocated) {
    detail::RawGttAllocation &raw = allocated.allocation;
    if (raw.reservation_address == nullptr && raw.handle == 0) {
      return {std::move(allocated.status), {}};
    }
    return {std::move(allocated.status),
            GttAllocation(state_, raw.reservation_address, raw.reservation_size,
                          raw.host_address, raw.size, raw.handle,
                          std::move(raw.gpu_ids), raw.mapped_device_count,
                          raw.unmapped_device_count, raw.map_complete)};
  }
  detail::RawGttAllocation &raw = allocated.allocation;
  return {{},
          GttAllocation(state_, raw.reservation_address, raw.reservation_size,
                        raw.host_address, raw.size, raw.handle,
                        std::move(raw.gpu_ids), raw.mapped_device_count,
                        raw.unmapped_device_count, raw.map_complete)};
}

MemoryStatus
KfdSession::map_pending_allocation(GttAllocation *allocation) const {
  if (state_ == nullptr || allocation == nullptr ||
      allocation->state_ != state_ || allocation->handle_ == 0) {
    return {MemoryError::InvalidSession, 0,
            "GPU mapping requires an allocation owned by this KFD session"};
  }

  detail::RawGttAllocation raw;
  try {
    raw.gpu_ids = allocation->gpu_ids_;
  } catch (const std::bad_alloc &) {
    return {MemoryError::AllocateState, 0,
            "failed to copy pending GPU mapping state"};
  }
  raw.reservation_address = allocation->reservation_address_;
  raw.reservation_size = allocation->reservation_size_;
  raw.host_address = allocation->host_address_;
  raw.size = allocation->size_;
  raw.handle = allocation->handle_;
  raw.mapped_device_count = allocation->mapped_device_count_;
  raw.unmapped_device_count = allocation->unmapped_device_count_;
  raw.map_complete = allocation->map_complete_;

  const MemoryStatus status =
      detail::map_gtt(state_->fd, &raw, allocation->gpu_ids_, real_syscalls());
  allocation->gpu_ids_ = std::move(raw.gpu_ids);
  allocation->mapped_device_count_ = raw.mapped_device_count;
  allocation->unmapped_device_count_ = raw.unmapped_device_count;
  allocation->map_complete_ = raw.map_complete;
  return status;
}

GttAllocation::GttAllocation(GttAllocation &&other) noexcept
    : state_(std::move(other.state_)),
      reservation_address_(other.reservation_address_),
      reservation_size_(other.reservation_size_),
      host_address_(other.host_address_), size_(other.size_),
      handle_(other.handle_), gpu_ids_(std::move(other.gpu_ids_)),
      mapped_device_count_(other.mapped_device_count_),
      unmapped_device_count_(other.unmapped_device_count_),
      map_complete_(other.map_complete_) {
  other.reset();
}

GttAllocation::~GttAllocation() { (void)release(); }

MemoryStatus GttAllocation::release() {
  if (reservation_address_ == nullptr && handle_ == 0) {
    return {};
  }
  if (state_ == nullptr) {
    return {MemoryError::InvalidSession, 0,
            "GTT release requires an open KFD session"};
  }

  detail::RawGttAllocation raw{reservation_address_,
                               reservation_size_,
                               host_address_,
                               size_,
                               handle_,
                               std::move(gpu_ids_),
                               mapped_device_count_,
                               unmapped_device_count_,
                               map_complete_};
  MemoryStatus status = detail::release_gtt(state_->fd, &raw, real_syscalls());
  reservation_address_ = raw.reservation_address;
  reservation_size_ = raw.reservation_size;
  host_address_ = raw.host_address;
  size_ = raw.size;
  handle_ = raw.handle;
  gpu_ids_ = std::move(raw.gpu_ids);
  mapped_device_count_ = raw.mapped_device_count;
  unmapped_device_count_ = raw.unmapped_device_count;
  map_complete_ = raw.map_complete;
  if (status) {
    reset();
  }
  return status;
}

void GttAllocation::reset() {
  state_.reset();
  reservation_address_ = nullptr;
  reservation_size_ = 0;
  host_address_ = nullptr;
  size_ = 0;
  handle_ = 0;
  gpu_ids_.clear();
  mapped_device_count_ = 0;
  unmapped_device_count_ = 0;
  map_complete_ = false;
}

ScratchAllocation::ScratchAllocation(ScratchAllocation &&other) noexcept
    : state_(std::move(other.state_)),
      reservation_address_(other.reservation_address_),
      reservation_size_(other.reservation_size_),
      gpu_address_(other.gpu_address_), size_(other.size_),
      handle_(other.handle_), gpu_ids_(std::move(other.gpu_ids_)),
      mapped_device_count_(other.mapped_device_count_),
      unmapped_device_count_(other.unmapped_device_count_),
      map_complete_(other.map_complete_), gpu_id_(other.gpu_id_),
      integrated_(other.integrated_), gpu_mapped_(other.gpu_mapped_) {
  other.reset();
}

ScratchAllocation::~ScratchAllocation() { (void)release(); }

MemoryStatus ScratchAllocation::release() {
  if (reservation_address_ == nullptr && handle_ == 0) {
    return {};
  }
  if (state_ == nullptr) {
    return {MemoryError::InvalidSession, 0,
            "scratch release requires an open KFD session"};
  }

  detail::RawGttAllocation raw{
      reservation_address_,
      reservation_size_,
      reinterpret_cast<void *>(static_cast<uintptr_t>(gpu_address_)),
      size_,
      handle_,
      std::move(gpu_ids_),
      mapped_device_count_,
      unmapped_device_count_,
      map_complete_};
  const MemoryStatus status =
      detail::release_gtt(state_->fd, &raw, real_syscalls());
  reservation_address_ = raw.reservation_address;
  reservation_size_ = raw.reservation_size;
  size_ = raw.size;
  handle_ = raw.handle;
  gpu_ids_ = std::move(raw.gpu_ids);
  mapped_device_count_ = raw.mapped_device_count;
  unmapped_device_count_ = raw.unmapped_device_count;
  map_complete_ = raw.map_complete;
  gpu_mapped_ = integrated_ ? reservation_address_ != nullptr : map_complete_;
  if (!status) {
    return status;
  }

  detail::release_scratch_reservation(state_, gpu_id_);
  reset();
  return {};
}

void ScratchAllocation::reset() {
  state_.reset();
  reservation_address_ = nullptr;
  reservation_size_ = 0;
  gpu_address_ = 0;
  size_ = 0;
  handle_ = 0;
  gpu_ids_.clear();
  mapped_device_count_ = 0;
  unmapped_device_count_ = 0;
  map_complete_ = false;
  gpu_id_ = 0;
  integrated_ = false;
  gpu_mapped_ = false;
}

} // namespace light_rocr::transport::kfd
