#include "light_rocr/transport/kfd/queue.hpp"

#include "light_rocr/runtime/amd_queue.hpp"
#include "light_rocr/transport/kfd/memory_types.hpp"
#include "light_rocr/transport/kfd/session.hpp"

#include "queue_internal.hpp"
#include "session_state.hpp"

#include <linux/kfd_ioctl.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <system_error>
#include <utility>

#if defined(__x86_64__)
#include <emmintrin.h>
#endif

namespace light_rocr::transport::kfd {
namespace {

static_assert(sizeof(runtime::AmdQueueV1) <= kMemoryPageSize);
static_assert(sizeof(uintptr_t) == sizeof(uint64_t));
static_assert(sizeof(off_t) == sizeof(uint64_t));
static_assert(__atomic_always_lock_free(sizeof(uint16_t), nullptr));
static_assert(__atomic_always_lock_free(sizeof(uint64_t), nullptr));

constexpr uint16_t kAqlInvalidPacketHeader = 1;
constexpr uint64_t kEopBufferSize = kMemoryPageSize;
constexpr uint32_t kDoorbellCount = 1024;
constexpr uint32_t kNormalQueuePriority = 7;
constexpr uint64_t kCwsrDebugBytesPerWave = 32;
constexpr uint64_t kCwsrDebugAlignment = 64;
constexpr uint64_t kGfx10ControlStackLimit = 0x7000;

struct CwsrHeader {
  uint32_t control_stack_offset = 0;
  uint32_t control_stack_size = 0;
  uint32_t wave_state_offset = 0;
  uint32_t wave_state_size = 0;
  uint32_t debug_offset = 0;
  uint32_t debug_size = 0;
  uint64_t error_reason = 0;
  uint32_t error_event_id = 0;
  uint32_t reserved = 0;
};

static_assert(sizeof(CwsrHeader) == 40);

AqlQueueStatus system_failure(AqlQueueError error, int system_error,
                              const std::string &operation) {
  const std::error_code code(system_error, std::generic_category());
  return {error, system_error,
          operation + " failed: " + code.message() + " (" +
              std::to_string(system_error) + ")"};
}

AqlQueueStatus allocation_failure(AqlQueueError error,
                                  const MemoryStatus &status,
                                  const char *allocation_name) {
  return {error, status.system_error,
          std::string(allocation_name) +
              " allocation failed: " + status.message};
}

bool is_power_of_two(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

bool align_up(uint64_t value, uint64_t alignment, uint64_t *aligned) {
  if (!is_power_of_two(alignment) ||
      value > std::numeric_limits<uint64_t>::max() - (alignment - 1)) {
    return false;
  }
  *aligned = (value + alignment - 1) & ~(alignment - 1);
  return true;
}

bool checked_add(uint64_t left, uint64_t right, uint64_t *sum) {
  if (left > std::numeric_limits<uint64_t>::max() - right) {
    return false;
  }
  *sum = left + right;
  return true;
}

bool checked_multiply(uint64_t left, uint64_t right, uint64_t *product) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    return false;
  }
  *product = left * right;
  return true;
}

uint32_t packed_gfx_version(runtime::GpuArchitecture architecture) {
  return (architecture.major << 16U) | (architecture.minor << 8U) |
         architecture.stepping;
}

uint64_t vgpr_size_per_compute_unit(uint32_t gfx_version) {
  if (gfx_version < 0x090008U) {
    return 0x40000;
  }
  if (gfx_version <= 0x09000aU) {
    return 0x80000;
  }
  if (gfx_version <= 0x09000cU) {
    return 0x40000;
  }
  if (gfx_version <= 0x090500U) {
    return 0x80000;
  }
  if (gfx_version < 0x0b0000U) {
    return 0x40000;
  }
  if (gfx_version <= 0x0c0001U) {
    return 0x60000;
  }
  return 0;
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

detail::QueueSyscalls real_syscalls() {
  return {real_ioctl, real_mmap, real_munmap, real_madvise};
}

void fence_before_doorbell_store() {
#if defined(__x86_64__)
  _mm_sfence();
#elif defined(__aarch64__)
  __asm__ __volatile__("dmb oshst" ::: "memory");
#else
#error "light-rocr needs an audited host MMIO store fence for this architecture"
#endif
}

void initialize_ring(GttAllocation *ring) {
  auto *bytes = static_cast<uint8_t *>(ring->host_address());
  std::memset(bytes, 0, static_cast<size_t>(ring->size()));
  for (uint64_t offset = 0; offset < ring->size(); offset += kAqlPacketSize) {
    std::memcpy(bytes + offset, &kAqlInvalidPacketHeader,
                sizeof(kAqlInvalidPacketHeader));
  }
}

uint64_t next_hsa_queue_id() {
  static std::atomic<uint64_t> next_id{0};
  return next_id.fetch_add(1, std::memory_order_relaxed);
}

} // namespace

namespace detail {
namespace {

bool valid_syscalls(QueueSyscalls syscalls) {
  return syscalls.ioctl_function != nullptr &&
         syscalls.mmap_function != nullptr &&
         syscalls.munmap_function != nullptr &&
         syscalls.madvise_function != nullptr;
}

bool invoke_ioctl(int fd, unsigned long request, void *arguments,
                  QueueIoctlFunction ioctl_function, int *system_error) {
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

} // namespace

AqlQueueStatus compute_cwsr_layout(const runtime::Node &node,
                                   CwsrLayout *layout) {
  if (layout == nullptr) {
    return {AqlQueueError::InvalidCwsrLayout, 0,
            "CWSR layout requires an output object"};
  }
  *layout = {};

  const uint32_t gfx_version = packed_gfx_version(node.architecture);
  const uint32_t compute_unit_count = node.compute_unit_count();
  const uint32_t xcc_count = node.xcc_count == 0 ? 1 : node.xcc_count;
  const uint64_t vgpr_size = vgpr_size_per_compute_unit(gfx_version);
  if (gfx_version < 0x090000U || vgpr_size == 0 || compute_unit_count == 0 ||
      compute_unit_count % xcc_count != 0 || node.simd_per_compute_unit == 0 ||
      node.maximum_waves_per_simd == 0 || node.lds_size_kb == 0) {
    return {AqlQueueError::InvalidCwsrLayout, 0,
            "CWSR requires complete supported GPU topology"};
  }

  const uint64_t compute_units_per_xcc = compute_unit_count / xcc_count;
  uint64_t wave_count = 0;
  uint64_t simd_count_per_xcc = 0;
  if (!checked_multiply(compute_units_per_xcc, node.simd_per_compute_unit,
                        &simd_count_per_xcc) ||
      !checked_multiply(simd_count_per_xcc, node.maximum_waves_per_simd,
                        &wave_count)) {
    return {AqlQueueError::InvalidCwsrLayout, 0, "CWSR wave count overflowed"};
  }
  const uint64_t bytes_per_wave = gfx_version >= 0x0a0100U ? 12 : 8;
  uint64_t control_stack_payload_size = 0;
  uint64_t control_stack_unaligned_size = 0;
  if (!checked_multiply(wave_count, bytes_per_wave,
                        &control_stack_payload_size) ||
      !checked_add(control_stack_payload_size, sizeof(CwsrHeader) + 8,
                   &control_stack_unaligned_size)) {
    return {AqlQueueError::InvalidCwsrLayout, 0,
            "CWSR control stack size overflowed"};
  }
  uint64_t derived_control_stack_size = 0;
  if (!align_up(control_stack_unaligned_size, kMemoryPageSize,
                &derived_control_stack_size)) {
    return {AqlQueueError::InvalidCwsrLayout, 0,
            "CWSR control stack size overflowed"};
  }
  if ((gfx_version & 0x3f0000U) == 0x0a0000U) {
    derived_control_stack_size =
        std::min(derived_control_stack_size, kGfx10ControlStackLimit);
  }

  const uint64_t lds_size = uint64_t{node.lds_size_kb} << 10U;
  constexpr uint64_t kSgprSizePerComputeUnit = 0x4000;
  constexpr uint64_t kHardwareRegisterSizePerComputeUnit = 0x1000;
  const uint64_t bytes_per_compute_unit = vgpr_size + kSgprSizePerComputeUnit +
                                          lds_size +
                                          kHardwareRegisterSizePerComputeUnit;
  uint64_t workgroup_data_size = 0;
  if (!checked_multiply(compute_units_per_xcc, bytes_per_compute_unit,
                        &workgroup_data_size)) {
    return {AqlQueueError::InvalidCwsrLayout, 0,
            "CWSR wave state size overflowed"};
  }
  uint64_t aligned_workgroup_data_size = 0;
  if (!align_up(workgroup_data_size, kMemoryPageSize,
                &aligned_workgroup_data_size)) {
    return {AqlQueueError::InvalidCwsrLayout, 0,
            "CWSR wave state size overflowed"};
  }

  const uint64_t control_stack_size = node.control_stack_size != 0
                                          ? node.control_stack_size
                                          : derived_control_stack_size;
  uint64_t derived_context_save_restore_size = 0;
  if (!checked_add(derived_control_stack_size, aligned_workgroup_data_size,
                   &derived_context_save_restore_size)) {
    return {AqlQueueError::InvalidCwsrLayout, 0,
            "CWSR context size overflowed"};
  }
  const uint64_t context_save_restore_size =
      node.cwsr_size != 0 ? node.cwsr_size : derived_context_save_restore_size;
  uint64_t debug_memory_unaligned_size = 0;
  uint64_t debug_memory_size = 0;
  if (!checked_multiply(wave_count, kCwsrDebugBytesPerWave,
                        &debug_memory_unaligned_size) ||
      !align_up(debug_memory_unaligned_size, kCwsrDebugAlignment,
                &debug_memory_size) ||
      control_stack_size > std::numeric_limits<uint32_t>::max() ||
      context_save_restore_size > std::numeric_limits<uint32_t>::max() ||
      debug_memory_size > std::numeric_limits<uint32_t>::max() ||
      context_save_restore_size >
          std::numeric_limits<uint32_t>::max() / xcc_count ||
      debug_memory_size > std::numeric_limits<uint32_t>::max() / xcc_count) {
    return {AqlQueueError::InvalidCwsrLayout, 0,
            "CWSR per-XCC size exceeds the KFD ABI"};
  }

  uint64_t per_xcc_size = 0;
  uint64_t total_size = 0;
  if (!checked_add(context_save_restore_size, debug_memory_size,
                   &per_xcc_size) ||
      !checked_multiply(per_xcc_size, xcc_count, &total_size)) {
    return {AqlQueueError::InvalidCwsrLayout, 0,
            "CWSR allocation size overflowed"};
  }
  uint64_t allocation_size = 0;
  if (!align_up(total_size, kMemoryPageSize, &allocation_size) ||
      allocation_size > std::numeric_limits<size_t>::max()) {
    return {AqlQueueError::InvalidCwsrLayout, 0,
            "CWSR allocation exceeds the host size range"};
  }

  layout->context_save_restore_size =
      static_cast<uint32_t>(context_save_restore_size);
  layout->control_stack_size = static_cast<uint32_t>(control_stack_size);
  layout->debug_memory_size = static_cast<uint32_t>(debug_memory_size);
  layout->xcc_count = xcc_count;
  layout->allocation_size = allocation_size;
  return {};
}

void initialize_cwsr(void *address, const CwsrLayout &layout) {
  std::memset(address, 0, static_cast<size_t>(layout.allocation_size));
  for (uint32_t xcc = 0; xcc < layout.xcc_count; ++xcc) {
    auto *header = reinterpret_cast<CwsrHeader *>(
        static_cast<uint8_t *>(address) +
        uint64_t{xcc} * layout.context_save_restore_size);
    header->debug_offset =
        (layout.xcc_count - xcc) * layout.context_save_restore_size;
    header->debug_size = layout.debug_memory_size * layout.xcc_count;
  }
}

RawAqlQueueResult create_aql_queue(int kfd_fd, const AqlQueueCreateInfo &info,
                                   QueueSyscalls syscalls) {
  if (kfd_fd < 0 || !valid_syscalls(syscalls)) {
    return {{AqlQueueError::InvalidSession, 0,
             "AQL queue creation requires an open KFD session"},
            {}};
  }
  if (info.gpu_id == 0 || info.ring_address == 0 ||
      info.read_pointer_address == 0 || info.write_pointer_address == 0 ||
      info.eop_buffer_address == 0 || info.eop_buffer_size == 0 ||
      info.context_save_restore_address == 0 ||
      info.context_save_restore_size == 0 || info.control_stack_size == 0) {
    return {{AqlQueueError::InvalidNode, 0,
             "AQL queue creation requires valid GPU resource addresses"},
            {}};
  }
  if (info.ring_size < kAqlRingMinimumSize ||
      info.ring_size > kAqlRingMaximumSize ||
      !is_power_of_two(info.ring_size) ||
      info.ring_size > std::numeric_limits<uint32_t>::max()) {
    return {{AqlQueueError::InvalidRingSize, 0,
             "AQL ring size must be a power of two from 4096 through 8388608"},
            {}};
  }
  if (info.doorbell_size != 4 && info.doorbell_size != 8) {
    return {{AqlQueueError::InvalidDoorbell, 0,
             "AQL queue requires a 4-byte or 8-byte doorbell"},
            {}};
  }
  if (info.doorbell_mapping_address == nullptr) {
    return {{AqlQueueError::InvalidDoorbell, 0,
             "AQL queue requires reserved GPUVM doorbell storage"},
            {}};
  }

  kfd_ioctl_create_queue_args arguments{};
  arguments.ring_base_address = info.ring_address;
  arguments.write_pointer_address = info.write_pointer_address;
  arguments.read_pointer_address = info.read_pointer_address;
  arguments.ring_size = static_cast<uint32_t>(info.ring_size);
  arguments.gpu_id = info.gpu_id;
  arguments.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  arguments.queue_percentage = KFD_MAX_QUEUE_PERCENTAGE;
  arguments.queue_priority = kNormalQueuePriority;
  arguments.eop_buffer_address = info.eop_buffer_address;
  arguments.eop_buffer_size = info.eop_buffer_size;
  arguments.ctx_save_restore_address = info.context_save_restore_address;
  arguments.ctx_save_restore_size = info.context_save_restore_size;
  arguments.ctl_stack_size = info.control_stack_size;

  int system_error = 0;
  if (!invoke_ioctl(kfd_fd, AMDKFD_IOC_CREATE_QUEUE, &arguments,
                    syscalls.ioctl_function, &system_error)) {
    return {system_failure(AqlQueueError::CreateQueue, system_error,
                           "AMDKFD_IOC_CREATE_QUEUE"),
            {}};
  }

  RawAqlQueue queue;
  queue.queue_id = arguments.queue_id;
  queue.active = true;
  queue.doorbell_mapping_size = std::max<uint64_t>(
      kMemoryPageSize,
      uint64_t{kDoorbellCount} * static_cast<uint64_t>(info.doorbell_size));
  const uint64_t mask = queue.doorbell_mapping_size - 1;
  const uint64_t mapping_offset = arguments.doorbell_offset & ~mask;
  const uint64_t doorbell_offset = arguments.doorbell_offset & mask;
  if (doorbell_offset > queue.doorbell_mapping_size - info.doorbell_size ||
      queue.doorbell_mapping_size > std::numeric_limits<size_t>::max()) {
    AqlQueueStatus status{AqlQueueError::InvalidDoorbell, 0,
                          "KFD returned invalid AQL doorbell offset " +
                              std::to_string(arguments.doorbell_offset) +
                              " for mapping size " +
                              std::to_string(queue.doorbell_mapping_size)};
    const AqlQueueStatus cleanup = release_aql_queue(kfd_fd, &queue, syscalls);
    if (!cleanup) {
      status.message += "; cleanup failed: " + cleanup.message;
    }
    return {std::move(status), std::move(queue)};
  }

  off_t mmap_offset = 0;
  // KFD mmap offsets use high bits as an unsigned type tag. Preserve that bit
  // pattern when crossing the signed Linux off_t API boundary.
  std::memcpy(&mmap_offset, &mapping_offset, sizeof(mmap_offset));
  void *mapping = syscalls.mmap_function(
      info.doorbell_mapping_address,
      static_cast<size_t>(queue.doorbell_mapping_size), PROT_READ | PROT_WRITE,
      MAP_SHARED | MAP_FIXED, kfd_fd, mmap_offset);
  if (mapping == MAP_FAILED) {
    AqlQueueStatus status = system_failure(AqlQueueError::MapDoorbell, errno,
                                           "mmap(KFD AQL doorbell)");
    const AqlQueueStatus cleanup = release_aql_queue(kfd_fd, &queue, syscalls);
    if (!cleanup) {
      status.message += "; cleanup failed: " + cleanup.message;
    }
    return {std::move(status), std::move(queue)};
  }

  queue.doorbell_mapping = mapping;
  queue.doorbell_address = reinterpret_cast<uintptr_t>(mapping) +
                           static_cast<uintptr_t>(doorbell_offset);
  if (syscalls.madvise_function(
          mapping, static_cast<size_t>(queue.doorbell_mapping_size),
          MADV_DONTFORK) != 0) {
    AqlQueueStatus status = system_failure(AqlQueueError::MapDoorbell, errno,
                                           "madvise(KFD AQL doorbell)");
    const AqlQueueStatus cleanup = release_aql_queue(kfd_fd, &queue, syscalls);
    if (!cleanup) {
      status.message += "; cleanup failed: " + cleanup.message;
    }
    return {std::move(status), std::move(queue)};
  }
  return {{}, std::move(queue)};
}

AqlQueueStatus release_aql_queue(int kfd_fd, RawAqlQueue *queue,
                                 QueueSyscalls syscalls) {
  if (queue == nullptr ||
      (!queue->active && queue->doorbell_mapping == nullptr)) {
    return {};
  }
  if (!valid_syscalls(syscalls) || (queue->active && kfd_fd < 0)) {
    return {AqlQueueError::InvalidSession, 0,
            "AQL queue release requires an open KFD session"};
  }
  if (queue->active) {
    kfd_ioctl_destroy_queue_args arguments{};
    arguments.queue_id = queue->queue_id;
    int system_error = 0;
    if (!invoke_ioctl(kfd_fd, AMDKFD_IOC_DESTROY_QUEUE, &arguments,
                      syscalls.ioctl_function, &system_error)) {
      return system_failure(AqlQueueError::DestroyQueue, system_error,
                            "AMDKFD_IOC_DESTROY_QUEUE");
    }
    queue->active = false;
  }
  if (queue->doorbell_mapping != nullptr) {
    if (syscalls.munmap_function(
            queue->doorbell_mapping,
            static_cast<size_t>(queue->doorbell_mapping_size)) != 0) {
      return system_failure(AqlQueueError::UnmapDoorbell, errno,
                            "munmap(KFD AQL doorbell)");
    }
    queue->doorbell_mapping = nullptr;
    queue->doorbell_mapping_size = 0;
    queue->doorbell_address = 0;
  }
  queue->queue_id = 0;
  return {};
}

} // namespace detail

struct AqlQueueState {
  AqlQueueState(std::shared_ptr<KfdState> session_state, uint32_t queue_gpu_id,
                uint64_t queue_ring_size)
      : session(std::move(session_state)), gpu_id(queue_gpu_id),
        ring_size(queue_ring_size) {}

  [[nodiscard]] bool owns_resources() const {
    return queue.active || queue.doorbell_mapping != nullptr ||
           (ring.has_value() && static_cast<bool>(*ring)) ||
           (control.has_value() && static_cast<bool>(*control)) ||
           (eop.has_value() && static_cast<bool>(*eop)) ||
           (cwsr.has_value() && static_cast<bool>(*cwsr)) ||
           (doorbell.has_value() && static_cast<bool>(*doorbell));
  }

  std::shared_ptr<KfdState> session;
  uint32_t gpu_id = 0;
  std::optional<GttAllocation> ring;
  std::optional<GttAllocation> control;
  std::optional<GttAllocation> eop;
  std::optional<GttAllocation> cwsr;
  std::optional<GttAllocation> doorbell;
  uint64_t ring_size = 0;
  detail::RawAqlQueue queue;
};

const char *aql_queue_error_name(AqlQueueError error) {
  switch (error) {
  case AqlQueueError::None:
    return "none";
  case AqlQueueError::InvalidSession:
    return "invalid_session";
  case AqlQueueError::InvalidNode:
    return "invalid_node";
  case AqlQueueError::InvalidRingSize:
    return "invalid_ring_size";
  case AqlQueueError::AcquireVm:
    return "acquire_vm";
  case AqlQueueError::AllocateState:
    return "allocate_state";
  case AqlQueueError::AllocateRing:
    return "allocate_ring";
  case AqlQueueError::AllocateControl:
    return "allocate_control";
  case AqlQueueError::AllocateEop:
    return "allocate_eop";
  case AqlQueueError::InvalidCwsrLayout:
    return "invalid_cwsr_layout";
  case AqlQueueError::AllocateCwsr:
    return "allocate_cwsr";
  case AqlQueueError::AllocateDoorbell:
    return "allocate_doorbell";
  case AqlQueueError::CreateQueue:
    return "create_queue";
  case AqlQueueError::InvalidDoorbell:
    return "invalid_doorbell";
  case AqlQueueError::MapDoorbell:
    return "map_doorbell";
  case AqlQueueError::DestroyQueue:
    return "destroy_queue";
  case AqlQueueError::UnmapDoorbell:
    return "unmap_doorbell";
  case AqlQueueError::ReleaseEop:
    return "release_eop";
  case AqlQueueError::ReleaseCwsr:
    return "release_cwsr";
  case AqlQueueError::ReleaseDoorbell:
    return "release_doorbell";
  case AqlQueueError::ReleaseControl:
    return "release_control";
  case AqlQueueError::ReleaseRing:
    return "release_ring";
  }
  return "unknown";
}

const char *aql_queue_primitive_error_name(AqlQueuePrimitiveError error) {
  switch (error) {
  case AqlQueuePrimitiveError::None:
    return "none";
  case AqlQueuePrimitiveError::InvalidQueue:
    return "invalid_queue";
  case AqlQueuePrimitiveError::InvalidIncrement:
    return "invalid_increment";
  case AqlQueuePrimitiveError::IndexOverflow:
    return "index_overflow";
  case AqlQueuePrimitiveError::InvalidDoorbell:
    return "invalid_doorbell";
  }
  return "unknown";
}

AqlQueueResult KfdSession::create_aql_queue(const runtime::Node &node,
                                            uint64_t ring_size,
                                            const std::string &dri_root) const {
  if (state_ == nullptr) {
    return {{AqlQueueError::InvalidSession, 0, "KFD session is not open"}, {}};
  }
  if (ring_size < kAqlRingMinimumSize || ring_size > kAqlRingMaximumSize ||
      !is_power_of_two(ring_size) || ring_size % kMemoryPageSize != 0) {
    return {{AqlQueueError::InvalidRingSize, 0,
             "AQL ring size must be a page-aligned power of two from 4096 "
             "through 8388608"},
            {}};
  }

  const uint32_t compute_unit_count = node.compute_unit_count();
  const uint64_t maximum_wave_id =
      uint64_t{node.maximum_waves_per_simd} * node.simd_per_compute_unit;
  if (!node.is_gpu() || node.gpu_id == 0 || node.drm_render_minor < 0 ||
      node.architecture.major < 9 || compute_unit_count == 0 ||
      maximum_wave_id == 0 ||
      maximum_wave_id - 1 > std::numeric_limits<uint32_t>::max()) {
    return {{AqlQueueError::InvalidNode, 0,
             "AQL queue requires complete GFX9+ GPU and render-node topology"},
            {}};
  }

  detail::CwsrLayout cwsr_layout;
  const AqlQueueStatus cwsr_layout_status =
      detail::compute_cwsr_layout(node, &cwsr_layout);
  if (!cwsr_layout_status) {
    return {cwsr_layout_status, {}};
  }

  const VmResult acquired = acquire_vm(node, dri_root);
  if (!acquired) {
    return {{AqlQueueError::AcquireVm, acquired.status.system_error,
             "failed to acquire VM for AQL queue: " + acquired.status.message},
            {}};
  }
  if (acquired.aperture.lds_base >> 32U == 0 ||
      acquired.aperture.scratch_base >> 32U == 0) {
    return {{AqlQueueError::InvalidNode, 0,
             "AQL queue requires non-zero KFD LDS and scratch apertures"},
            {}};
  }

  // Allocate all host bookkeeping before acquiring KFD resources. From this
  // point onward each acquired object can be retained for explicit cleanup.
  std::unique_ptr<AqlQueueState> queue_state;
  try {
    queue_state =
        std::make_unique<AqlQueueState>(state_, node.gpu_id, ring_size);
  } catch (const std::bad_alloc &) {
    return {{AqlQueueError::AllocateState, 0,
             "failed to allocate direct KFD AQL queue state"},
            {}};
  }

  auto ring = allocate_gtt_impl(node, ring_size, dri_root, GttUsage::AqlRing);
  queue_state->ring.emplace(std::move(ring.allocation));
  if (!ring) {
    AqlQueueStatus status = allocation_failure(AqlQueueError::AllocateRing,
                                               ring.status, "AQL ring");
    return {std::move(status), queue_state->owns_resources()
                                   ? AqlQueue(std::move(queue_state))
                                   : AqlQueue{}};
  }
  initialize_ring(&*queue_state->ring);

  auto control =
      allocate_gtt_impl(node, kMemoryPageSize, dri_root, GttUsage::General);
  queue_state->control.emplace(std::move(control.allocation));
  if (!control) {
    AqlQueueStatus status = allocation_failure(
        AqlQueueError::AllocateControl, control.status, "AQL queue control");
    return {std::move(status), AqlQueue(std::move(queue_state))};
  }
  auto eop = allocate_gtt_impl(node, kEopBufferSize, dri_root, GttUsage::Eop);
  queue_state->eop.emplace(std::move(eop.allocation));
  if (!eop) {
    AqlQueueStatus status = allocation_failure(AqlQueueError::AllocateEop,
                                               eop.status, "AQL EOP buffer");
    return {std::move(status), AqlQueue(std::move(queue_state))};
  }
  auto cwsr = allocate_gtt_impl(node, cwsr_layout.allocation_size, dri_root,
                                GttUsage::General);
  queue_state->cwsr.emplace(std::move(cwsr.allocation));
  if (!cwsr) {
    AqlQueueStatus status = allocation_failure(AqlQueueError::AllocateCwsr,
                                               cwsr.status, "AQL CWSR");
    return {std::move(status), AqlQueue(std::move(queue_state))};
  }
  detail::initialize_cwsr(queue_state->cwsr->host_address(), cwsr_layout);

  const uint64_t doorbell_mapping_size =
      std::max<uint64_t>(kMemoryPageSize, uint64_t{kDoorbellCount} * 8U);
  auto doorbell = allocate_gtt_impl(node, doorbell_mapping_size, dri_root,
                                    GttUsage::Doorbell);
  queue_state->doorbell.emplace(std::move(doorbell.allocation));
  if (!doorbell) {
    AqlQueueStatus status = allocation_failure(AqlQueueError::AllocateDoorbell,
                                               doorbell.status, "AQL doorbell");
    return {std::move(status), AqlQueue(std::move(queue_state))};
  }

  std::memset(queue_state->control->host_address(), 0,
              static_cast<size_t>(queue_state->control->size()));
  auto *amd_queue =
      ::new (queue_state->control->host_address()) runtime::AmdQueueV1;
  amd_queue->hsa_queue.type = runtime::kHsaQueueTypeMulti;
  amd_queue->hsa_queue.features = runtime::kHsaQueueFeatureKernelDispatch;
  amd_queue->hsa_queue.base_address = queue_state->ring->gpu_address();
  amd_queue->hsa_queue.size = static_cast<uint32_t>(ring_size / kAqlPacketSize);
  amd_queue->hsa_queue.id = next_hsa_queue_id();
  amd_queue->group_segment_aperture_base_hi =
      static_cast<uint32_t>(acquired.aperture.lds_base >> 32U);
  amd_queue->private_segment_aperture_base_hi =
      static_cast<uint32_t>(acquired.aperture.scratch_base >> 32U);
  amd_queue->max_cu_id = compute_unit_count - 1U;
  amd_queue->max_wave_id = static_cast<uint32_t>(maximum_wave_id - 1U);
  amd_queue->read_dispatch_id_field_base_byte_offset =
      static_cast<uint32_t>(offsetof(runtime::AmdQueueV1, read_dispatch_id));
  amd_queue->queue_properties = runtime::kAmdQueuePropertyIsPointer64;

  detail::AqlQueueCreateInfo info;
  info.gpu_id = node.gpu_id;
  info.ring_address = queue_state->ring->gpu_address();
  info.ring_size = ring_size;
  info.read_pointer_address = queue_state->control->gpu_address() +
                              offsetof(runtime::AmdQueueV1, read_dispatch_id);
  info.write_pointer_address = queue_state->control->gpu_address() +
                               offsetof(runtime::AmdQueueV1, write_dispatch_id);
  info.eop_buffer_address = queue_state->eop->gpu_address();
  info.eop_buffer_size = queue_state->eop->size();
  info.context_save_restore_address = queue_state->cwsr->gpu_address();
  info.context_save_restore_size = cwsr_layout.context_save_restore_size;
  info.control_stack_size = cwsr_layout.control_stack_size;
  info.doorbell_size = node.architecture.major >= 9 ? 8U : 4U;
  info.doorbell_mapping_address = queue_state->doorbell->host_address();

  detail::RawAqlQueueResult created =
      detail::create_aql_queue(state_->fd, info, real_syscalls());
  queue_state->queue = std::move(created.queue);
  if (!created) {
    return {std::move(created.status), AqlQueue(std::move(queue_state))};
  }
  const MemoryStatus doorbell_mapped =
      map_pending_allocation(&*queue_state->doorbell);
  if (!doorbell_mapped) {
    return {{AqlQueueError::MapDoorbell, doorbell_mapped.system_error,
             "AQL doorbell GPU mapping failed: " + doorbell_mapped.message},
            AqlQueue(std::move(queue_state))};
  }
  return {{}, AqlQueue(std::move(queue_state))};
}

AqlQueue::AqlQueue() = default;

AqlQueue::AqlQueue(std::unique_ptr<AqlQueueState> state)
    : state_(std::move(state)) {}

AqlQueue::AqlQueue(AqlQueue &&other) noexcept
    : state_(std::move(other.state_)) {}

AqlQueue &AqlQueue::operator=(AqlQueue &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  release_for_destruction();
  state_ = std::move(other.state_);
  return *this;
}

AqlQueue::~AqlQueue() { release_for_destruction(); }

uint32_t AqlQueue::queue_id() const {
  return state_ != nullptr ? state_->queue.queue_id : 0;
}

uintptr_t AqlQueue::doorbell_address() const {
  return state_ != nullptr ? state_->queue.doorbell_address : 0;
}

void *AqlQueue::ring_host_address() const {
  return state_ != nullptr && state_->ring.has_value()
             ? state_->ring->host_address()
             : nullptr;
}

uint64_t AqlQueue::ring_gpu_address() const {
  return state_ != nullptr && state_->ring.has_value()
             ? state_->ring->gpu_address()
             : 0;
}

uint64_t AqlQueue::ring_size() const {
  return state_ != nullptr ? state_->ring_size : 0;
}

uint64_t AqlQueue::packet_count() const {
  return state_ != nullptr ? state_->ring_size / kAqlPacketSize : 0;
}

uint64_t AqlQueue::read_index_acquire() const {
  if (state_ == nullptr || !state_->control.has_value() ||
      state_->control->host_address() == nullptr) {
    return 0;
  }
  const auto *control =
      static_cast<const runtime::AmdQueueV1 *>(state_->control->host_address());
  return __atomic_load_n(&control->read_dispatch_id, __ATOMIC_ACQUIRE);
}

uint64_t AqlQueue::write_index_relaxed() const {
  if (state_ == nullptr || !state_->control.has_value() ||
      state_->control->host_address() == nullptr) {
    return 0;
  }
  const auto *control =
      static_cast<const runtime::AmdQueueV1 *>(state_->control->host_address());
  return __atomic_load_n(&control->write_dispatch_id, __ATOMIC_RELAXED);
}

AqlQueueIndexResult AqlQueue::add_write_index_scacq_screl(uint64_t increment) {
  if (state_ == nullptr || !state_->queue.active ||
      !state_->control.has_value() ||
      state_->control->host_address() == nullptr) {
    return {{AqlQueuePrimitiveError::InvalidQueue, "AQL queue is not active"},
            0};
  }
  if (increment == 0) {
    return {{AqlQueuePrimitiveError::InvalidIncrement,
             "queue write-index increment must be non-zero"},
            0};
  }

  auto *control =
      static_cast<runtime::AmdQueueV1 *>(state_->control->host_address());
  uint64_t previous =
      __atomic_load_n(&control->write_dispatch_id, __ATOMIC_SEQ_CST);
  while (true) {
    if (previous > std::numeric_limits<uint64_t>::max() - increment) {
      return {{AqlQueuePrimitiveError::IndexOverflow,
               "queue write-index increment would overflow"},
              previous};
    }
    const uint64_t desired = previous + increment;
    if (__atomic_compare_exchange_n(&control->write_dispatch_id, &previous,
                                    desired, false, __ATOMIC_SEQ_CST,
                                    __ATOMIC_SEQ_CST)) {
      return {{}, previous};
    }
  }
}

AqlQueuePrimitiveStatus AqlQueue::store_doorbell_screlease(uint64_t value) {
  if (state_ == nullptr || !state_->queue.active) {
    return {AqlQueuePrimitiveError::InvalidQueue, "AQL queue is not active"};
  }
  if (state_->queue.doorbell_address == 0) {
    return {AqlQueuePrimitiveError::InvalidDoorbell,
            "AQL queue has no doorbell mapping"};
  }

  __atomic_thread_fence(__ATOMIC_RELEASE);
  fence_before_doorbell_store();
  auto *doorbell = reinterpret_cast<uint64_t *>(state_->queue.doorbell_address);
  __atomic_store_n(doorbell, value, __ATOMIC_RELAXED);
  return {};
}

AqlQueue::operator bool() const {
  return state_ != nullptr && state_->queue.active &&
         state_->queue.doorbell_address != 0;
}

bool KfdSession::owns_aql_queue(const AqlQueue &queue,
                                const runtime::Node &node) const {
  return state_ != nullptr && node.is_gpu() && node.gpu_id != 0 &&
         queue.state_ != nullptr && queue.state_->session == state_ &&
         queue.state_->gpu_id == node.gpu_id;
}

AqlQueueStatus AqlQueue::release() {
  if (state_ == nullptr) {
    return {};
  }
  const AqlQueueStatus queue_status = detail::release_aql_queue(
      state_->session != nullptr ? state_->session->fd : -1, &state_->queue,
      real_syscalls());
  if (!queue_status) {
    return queue_status;
  }
  if (state_->doorbell.has_value()) {
    const MemoryStatus doorbell_status = state_->doorbell->release();
    if (!doorbell_status) {
      return {AqlQueueError::ReleaseDoorbell, doorbell_status.system_error,
              "AQL doorbell cleanup failed: " + doorbell_status.message};
    }
    state_->doorbell.reset();
  }
  if (state_->cwsr.has_value()) {
    const MemoryStatus cwsr_status = state_->cwsr->release();
    if (!cwsr_status) {
      return {AqlQueueError::ReleaseCwsr, cwsr_status.system_error,
              "AQL CWSR cleanup failed: " + cwsr_status.message};
    }
    state_->cwsr.reset();
  }
  if (state_->eop.has_value()) {
    const MemoryStatus eop_status = state_->eop->release();
    if (!eop_status) {
      return {AqlQueueError::ReleaseEop, eop_status.system_error,
              "AQL EOP cleanup failed: " + eop_status.message};
    }
    state_->eop.reset();
  }
  if (state_->control.has_value()) {
    const MemoryStatus control_status = state_->control->release();
    if (!control_status) {
      return {AqlQueueError::ReleaseControl, control_status.system_error,
              "AQL queue control cleanup failed: " + control_status.message};
    }
    state_->control.reset();
  }
  if (state_->ring.has_value()) {
    const MemoryStatus ring_status = state_->ring->release();
    if (!ring_status) {
      return {AqlQueueError::ReleaseRing, ring_status.system_error,
              "AQL ring cleanup failed: " + ring_status.message};
    }
    state_->ring.reset();
  }
  state_.reset();
  return {};
}

void AqlQueue::release_for_destruction() noexcept {
  AqlQueueStatus status = release();
  if (!status) {
    // A destructor cannot return a retry token. Retry once for transient
    // destroy/munmap/unmap failures before preserving any remaining owner.
    status = release();
  }
  if (!status && state_ != nullptr && state_->owns_resources()) {
    (void)state_.release();
  }
}

} // namespace light_rocr::transport::kfd
