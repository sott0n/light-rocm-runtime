#ifndef LIGHT_ROCR_TRANSPORT_KFD_QUEUE_INTERNAL_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_QUEUE_INTERNAL_HPP

#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/kfd/queue.hpp"

#include <cstddef>
#include <cstdint>
#include <sys/types.h>

namespace light_rocr::transport::kfd::detail {

using QueueIoctlFunction = int (*)(int fd, unsigned long request,
                                   void *arguments);
using DoorbellMmapFunction = void *(*)(void *address, size_t length,
                                       int protection, int flags, int fd,
                                       off_t offset);
using DoorbellMunmapFunction = int (*)(void *address, size_t length);

struct QueueSyscalls {
  QueueIoctlFunction ioctl_function = nullptr;
  DoorbellMmapFunction mmap_function = nullptr;
  DoorbellMunmapFunction munmap_function = nullptr;
};

struct RawAqlQueue {
  uint32_t queue_id = 0;
  void *doorbell_mapping = nullptr;
  uint64_t doorbell_mapping_size = 0;
  uintptr_t doorbell_address = 0;
  bool active = false;
};

struct RawAqlQueueResult {
  AqlQueueStatus status;
  RawAqlQueue queue;

  explicit operator bool() const { return static_cast<bool>(status); }
};

struct AqlQueueCreateInfo {
  uint32_t gpu_id = 0;
  uint64_t ring_address = 0;
  uint64_t ring_size = 0;
  uint64_t read_pointer_address = 0;
  uint64_t write_pointer_address = 0;
  uint64_t eop_buffer_address = 0;
  uint64_t eop_buffer_size = 0;
  uint64_t context_save_restore_address = 0;
  uint32_t context_save_restore_size = 0;
  uint32_t control_stack_size = 0;
  uint32_t doorbell_size = 0;
};

struct CwsrLayout {
  uint32_t context_save_restore_size = 0;
  uint32_t control_stack_size = 0;
  uint32_t debug_memory_size = 0;
  uint32_t xcc_count = 0;
  uint64_t allocation_size = 0;
};

[[nodiscard]] AqlQueueStatus compute_cwsr_layout(const runtime::Node &node,
                                                 CwsrLayout *layout);
void initialize_cwsr(void *address, const CwsrLayout &layout);

[[nodiscard]] RawAqlQueueResult create_aql_queue(int kfd_fd,
                                                 const AqlQueueCreateInfo &info,
                                                 QueueSyscalls syscalls);
[[nodiscard]] AqlQueueStatus release_aql_queue(int kfd_fd, RawAqlQueue *queue,
                                               QueueSyscalls syscalls);

} // namespace light_rocr::transport::kfd::detail

#endif
