#include "light_rocr/transport/kfd/memory.hpp"

#include "memory_internal.hpp"
#include "session_state.hpp"

#include <linux/kfd_ioctl.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <system_error>
#include <utility>

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

} // namespace

RawGttAllocationResult allocate_gtt(int kfd_fd, int render_fd,
                                    const ProcessAperture &aperture,
                                    uint64_t size, MemorySyscalls syscalls) {
  if (kfd_fd < 0 || render_fd < 0 || aperture.gpu_id == 0 ||
      !valid_syscalls(syscalls)) {
    return {{MemoryError::InvalidSession, 0,
             "GTT allocation requires an acquired KFD VM"},
            {}};
  }
  if (size == 0 || size % kMemoryPageSize != 0 ||
      size > std::numeric_limits<size_t>::max() ||
      size > std::numeric_limits<uint64_t>::max() -
                 kGuardPageCount * kMemoryPageSize) {
    return {{MemoryError::InvalidSize, 0,
             "GTT allocation size must be a non-zero multiple of 4096 bytes"},
            {}};
  }

  const uint64_t reservation_size = size + kGuardPageCount * kMemoryPageSize;
  if (reservation_size > std::numeric_limits<size_t>::max()) {
    return {{MemoryError::InvalidSize, 0,
             "GTT allocation plus guard pages exceeds the host size range"},
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
    return rollback_gtt(
        {MemoryError::ReserveVa, 0,
         "reserved CPU VA overflows the host address range"},
        kfd_fd, {reservation, reservation_size, nullptr, size, 0}, syscalls);
  }
  const uint64_t host_address = reservation_address + kMemoryPageSize;
  const bool range_overflows =
      host_address > std::numeric_limits<uint64_t>::max() - (size - 1);
  if (range_overflows || host_address < aperture.gpuvm_base ||
      host_address + size - 1 > aperture.gpuvm_limit) {
    return rollback_gtt(
        {MemoryError::ReserveVa, 0,
         "reserved CPU VA is outside the KFD GPUVM aperture"},
        kfd_fd, {reservation, reservation_size, nullptr, size, 0}, syscalls);
  }

  kfd_ioctl_alloc_memory_of_gpu_args arguments{};
  arguments.va_addr = host_address;
  arguments.size = size;
  arguments.gpu_id = aperture.gpu_id;
  arguments.flags = static_cast<uint32_t>(
      KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
      KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE);
  int system_error = 0;
  if (!invoke_ioctl(kfd_fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &arguments,
                    syscalls.ioctl_function, &system_error)) {
    return rollback_gtt(
        system_failure(MemoryError::AllocateGtt, system_error,
                       "AMDKFD_IOC_ALLOC_MEMORY_OF_GPU(GTT)"),
        kfd_fd, {reservation, reservation_size, nullptr, size, 0}, syscalls);
  }
  if (arguments.handle == 0) {
    return rollback_gtt(
        {MemoryError::AllocateGtt, 0,
         "KFD returned an invalid zero GTT handle"},
        kfd_fd, {reservation, reservation_size, nullptr, size, 0}, syscalls);
  }

  RawGttAllocation allocation{reservation, reservation_size, nullptr, size,
                              arguments.handle};

  if (arguments.mmap_offset >
      static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
    return rollback_gtt(
        {MemoryError::MapHost, 0,
         "KFD returned a GTT mmap offset outside the host off_t range"},
        kfd_fd, std::move(allocation), syscalls);
  }

  void *host = reinterpret_cast<void *>(static_cast<uintptr_t>(host_address));
  void *mapped = syscalls.mmap_function(
      host, static_cast<size_t>(size), PROT_READ | PROT_WRITE,
      MAP_SHARED | MAP_FIXED, render_fd,
      static_cast<off_t>(arguments.mmap_offset));
  if (mapped == MAP_FAILED) {
    return rollback_gtt(system_failure(MemoryError::MapHost, errno,
                                       "mmap(GTT render-node mapping)"),
                        kfd_fd, std::move(allocation), syscalls);
  }
  allocation.host_address = host;

  if (syscalls.madvise_function(host, static_cast<size_t>(size),
                                MADV_DONTFORK) != 0) {
    return rollback_gtt(system_failure(MemoryError::AdviseDontFork, errno,
                                       "madvise(MADV_DONTFORK)"),
                        kfd_fd, std::move(allocation), syscalls);
  }

  return {{}, std::move(allocation)};
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
  case MemoryError::AcquireVm:
    return "acquire_vm";
  case MemoryError::ReserveVa:
    return "reserve_va";
  case MemoryError::AllocateGtt:
    return "allocate_gtt";
  case MemoryError::MapHost:
    return "map_host";
  case MemoryError::AdviseDontFork:
    return "advise_dontfork";
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
      state_->fd, render_fd, acquired.aperture, size, real_syscalls());
  if (!allocated) {
    detail::RawGttAllocation &raw = allocated.allocation;
    if (raw.reservation_address == nullptr && raw.handle == 0) {
      return {std::move(allocated.status), {}};
    }
    return {std::move(allocated.status),
            GttAllocation(state_, raw.reservation_address, raw.reservation_size,
                          raw.host_address, raw.size, raw.handle)};
  }
  detail::RawGttAllocation &raw = allocated.allocation;
  return {{},
          GttAllocation(state_, raw.reservation_address, raw.reservation_size,
                        raw.host_address, raw.size, raw.handle)};
}

GttAllocation::GttAllocation(GttAllocation &&other) noexcept
    : state_(std::move(other.state_)),
      reservation_address_(other.reservation_address_),
      reservation_size_(other.reservation_size_),
      host_address_(other.host_address_), size_(other.size_),
      handle_(other.handle_) {
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

  detail::RawGttAllocation raw{reservation_address_, reservation_size_,
                               host_address_, size_, handle_};
  MemoryStatus status = detail::release_gtt(state_->fd, &raw, real_syscalls());
  reservation_address_ = raw.reservation_address;
  reservation_size_ = raw.reservation_size;
  host_address_ = raw.host_address;
  size_ = raw.size;
  handle_ = raw.handle;
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
}

} // namespace light_rocr::transport::kfd
