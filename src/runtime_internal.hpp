#ifndef LRRT_RUNTIME_INTERNAL_HPP_
#define LRRT_RUNTIME_INTERNAL_HPP_

#include "lrrt/lrrt.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#if LRRT_ENABLE_HSA
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#endif

#if LRRT_ENABLE_HSA
class LifetimePinCount {
public:
  void retain() { count_.fetch_add(1, std::memory_order_relaxed); }

  void release() {
    if (count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::lock_guard<std::mutex> lock(wait_mutex_);
      state_changed_.notify_all();
    }
  }

  bool empty() const { return count_.load(std::memory_order_acquire) == 0; }

  void wait_until_empty() {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    state_changed_.wait(lock, [this] { return empty(); });
  }

private:
  std::atomic<size_t> count_{0};
  std::mutex wait_mutex_;
  std::condition_variable state_changed_;
};

struct KernargBuffer {
  void *ptr;
  size_t size;
};

struct PendingDispatch {
  hsa_signal_t completion_signal;
  KernargBuffer kernarg;
};

struct PendingBarrier {
  // Reaching zero proves the queue consumed the barrier packet. Depending on
  // the producer, this may be a following dispatch signal or a synchronization
  // barrier's shared completion signal.
  hsa_signal_t retirement_signal;
  std::vector<lr_event_t *> dependencies;
};

struct PendingSynchronization {
  hsa_signal_t completion_signal;
  std::vector<hsa_signal_t> dispatch_dependencies;
};

struct QueueState {
  // Queue-local packet and retirement state is protected independently from
  // the runtime registry. When both locks are needed, acquire
  // g_devices_mutex before this mutex.
  std::mutex mutex;
  hsa_queue_t *queue;
  std::vector<PendingDispatch> pending_dispatches;
  std::vector<PendingDispatch> deferred_dispatches;
  std::vector<PendingBarrier> pending_barriers;
  std::vector<lr_event_t *> pending_events;
  std::vector<hsa_signal_t> signal_pool;
  std::vector<KernargBuffer> kernarg_pool;
  std::vector<PendingSynchronization> pending_synchronizations;
  std::vector<hsa_signal_t> progress_wait_signals;
  LifetimePinCount active_submissions;
  size_t active_synchronizers;
  bool destroying;
};
#endif

struct lr_event_t {
  lr_device_t device;
#if LRRT_ENABLE_HSA
  enum class Kind {
    None,
    Marker,
    AsyncCopy,
  };

  hsa_signal_t signal;
  Kind kind;
  bool pending;
  bool completed;
  // Queue-consumer tracking can be updated while the runtime registry lock is
  // released. Never acquire a QueueState mutex while holding this mutex.
  mutable std::mutex dependency_mutex;
  LifetimePinCount active_launch_dependencies;
  size_t active_synchronizers;
  bool destroying;
  std::unordered_set<QueueState *> dependency_queues;
  size_t dependency_count;
  uint64_t start_tick;
  uint64_t completion_tick;
  std::vector<lr_event_t *> dependencies;
  QueueState *recorded_queue;
  void *locked_host_ptr;
#endif
};

struct lr_queue_t {
  lr_device_t device;
  bool is_default;
#if LRRT_ENABLE_HSA
  QueueState state;
#endif
};

struct lr_module_t {
  lr_device_t device;
  std::vector<lr_kernel_t *> kernels;
#if LRRT_ENABLE_HSA
  LifetimePinCount active_submissions;
  bool destroying;
  hsa_code_object_reader_t reader;
  hsa_executable_t executable;
  hsa_loaded_code_object_t loaded_code_object;
#endif
};

struct lr_kernel_t {
  lr_module_t *module;
#if LRRT_ENABLE_HSA
  uint64_t object;
  uint32_t kernarg_size;
  uint32_t group_segment_size;
  uint32_t private_segment_size;
#endif
};

namespace lrrt_internal {

extern std::atomic<bool> g_initialized;

#if LRRT_ENABLE_HSA
struct DeviceState {
  hsa_agent_t agent;
  std::string name;
  hsa_region_t global_region;
  hsa_region_t kernarg_region;
  bool has_global_region;
  bool has_kernarg_region;
  lr_queue_t *default_queue;
  std::vector<lr_queue_t *> queues;
  std::vector<lr_event_t *> pending_events;
  lr_memory_stats_t memory_stats;
};

extern std::mutex g_devices_mutex;
extern std::condition_variable g_queue_state_changed;
extern std::condition_variable g_event_state_changed;
extern std::vector<DeviceState> g_devices;
extern hsa_agent_t g_host_agent;
extern bool g_has_host_agent;

lr_status_t to_lr_status(hsa_status_t status);
hsa_status_t create_queue(lr_device_t device_handle, DeviceState *device,
                          bool is_default, lr_queue_t **queue);
uint16_t packet_header(hsa_packet_type_t type);
uint16_t barrier_packet_header(hsa_packet_type_t type);
void publish_packet_header(uint16_t *header, uint16_t value);
uint16_t packet_setup(uint16_t dimensions);
uint16_t dispatch_dimensions(const lr_launch_config_t *config);
hsa_status_t acquire_signal_locked(QueueState *queue, hsa_signal_t *signal);
hsa_status_t acquire_kernarg_locked(QueueState *queue, hsa_region_t region,
                                    size_t size, KernargBuffer *kernarg);
lr_status_t event_wait_locked(lr_event_t *event,
                              QueueState *locked_queue = nullptr);
lr_status_t finish_event_wait_locked(lr_event_t *event,
                                     hsa_signal_value_t value,
                                     QueueState *locked_queue = nullptr);
void wait_for_event_synchronizers_locked(
    std::unique_lock<std::mutex> *devices_lock, lr_event_t *event);
void wait_for_all_event_synchronizers_locked(
    std::unique_lock<std::mutex> *devices_lock);
lr_status_t
wait_for_event_consumers_locked(std::unique_lock<std::mutex> *devices_lock,
                                DeviceState *device, lr_event_t *event);
lr_status_t drain_device_locked(DeviceState *device);
lr_status_t reap_completed_queue_work_locked(QueueState *queue);
lr_status_t collect_event_dependencies_locked(
    lr_device_t device, lr_event_t *const *dependencies,
    size_t dependency_count, const lr_event_t *completion_event,
    std::vector<lr_event_t *> *pending_dependencies);
bool event_has_queue_dependency(lr_event_t *event, QueueState *queue);
void retain_event_dependency(lr_event_t *event, QueueState *queue = nullptr);
void release_event_dependency(lr_event_t *event, QueueState *queue = nullptr);
size_t event_dependency_count(lr_event_t *event);
std::vector<QueueState *> event_dependency_queues(lr_event_t *event);
void clear_event_dependency_queues(lr_event_t *event);
size_t event_dependency_packet_count_locked(
    DeviceState *device, QueueState *queue,
    const std::vector<lr_event_t *> *explicit_dependencies);
lr_status_t enqueue_event_dependencies_locked(
    std::unique_lock<std::mutex> *devices_lock, DeviceState *device,
    std::unique_lock<std::mutex> *queue_lock, QueueState *queue,
    hsa_signal_t retirement_signal,
    const std::vector<lr_event_t *> *explicit_dependencies);
lr_status_t enqueue_explicit_event_dependencies_locally_locked(
    QueueState *queue, hsa_signal_t retirement_signal,
    const std::vector<lr_event_t *> &dependencies);
lr_status_t
ensure_queue_capacity_locked(std::unique_lock<std::mutex> *devices_lock,
                             std::unique_lock<std::mutex> *queue_lock,
                             QueueState *queue, size_t required_packets,
                             bool *lock_released = nullptr,
                             bool allow_destroying = false);
bool has_queue_capacity_locked(const QueueState *queue,
                               size_t required_packets);
void reap_completed_dispatches_locally_locked(QueueState *queue);
void reap_completed_barriers_locked(QueueState *queue);
bool valid_queue_locked(lr_queue_t *queue);
bool valid_event_locked(lr_event_t *event);
bool valid_kernel_locked(lr_kernel_t *kernel);
void release_device_queue_pools_locked(DeviceState *device,
                                       lr_status_t *result);
void destroy_all_queues_locked();
void wait_for_queue_submissions_locked(
    std::unique_lock<std::mutex> *devices_lock);
void wait_for_queue_synchronizers_locked(
    std::unique_lock<std::mutex> *devices_lock);
lr_status_t
enqueue_queue_synchronization_locked(QueueState *queue,
                                     hsa_signal_t *completion_signal,
                                     std::unique_lock<std::mutex> *devices_lock,
                                     std::unique_lock<std::mutex> *queue_lock);
lr_status_t enqueue_queue_tail_marker_locked(
    QueueState *queue, hsa_signal_t *completion_signal,
    std::unique_lock<std::mutex> *devices_lock,
    std::unique_lock<std::mutex> *queue_lock, bool allow_destroying = false);
lr_status_t finish_queue_synchronization_locked(QueueState *queue,
                                                hsa_signal_t completion_signal,
                                                hsa_signal_value_t wait_value);
lr_status_t synchronize_device(DeviceState *device,
                               std::unique_lock<std::mutex> *devices_lock);
void release_events_locked();
void release_modules_locked();
void wait_for_memory_operations_locked(
    std::unique_lock<std::mutex> *devices_lock);
void release_memory_allocations_locked(lr_status_t *result);
#endif

bool valid_device(lr_device_t device);

} // namespace lrrt_internal

#endif // LRRT_RUNTIME_INTERNAL_HPP_
