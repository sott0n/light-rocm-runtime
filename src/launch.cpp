#include "runtime_internal.hpp"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

using namespace lrrt_internal;

extern "C" {

static lr_status_t
launch_impl(lr_kernel_t *kernel, const lr_launch_config_t *config,
            const void *args, size_t args_size, lr_queue_t *execution_queue,
            bool use_default_queue, lr_event_t *const *explicit_dependencies,
            size_t dependency_count, bool use_implicit_dependencies) {
  if (!g_initialized.load()) {
    return LR_ERROR_NOT_INITIALIZED;
  }
  if (!kernel || !config || !args || args_size == 0) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (config->grid.x == 0 || config->grid.y == 0 || config->grid.z == 0 ||
      config->block.x == 0 || config->block.y == 0 || config->block.z == 0) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (config->grid.x < config->block.x || config->grid.y < config->block.y ||
      config->grid.z < config->block.z) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  if (config->block.x > UINT16_MAX || config->block.y > UINT16_MAX ||
      config->block.z > UINT16_MAX) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

#if LRRT_ENABLE_HSA
  std::lock_guard<std::mutex> lock(g_devices_mutex);
  if (!valid_kernel_locked(kernel)) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  lr_device_t device = kernel->module->device;
  if (device.index >= g_devices.size()) {
    return LR_ERROR_INVALID_ARGUMENT;
  }

  DeviceState &state = g_devices[device.index];
  if (use_default_queue) {
    execution_queue = state.default_queue;
  }
  if (!execution_queue || !valid_queue_locked(execution_queue) ||
      execution_queue->device.index != device.index ||
      !state.has_kernarg_region || args_size > kernel->kernarg_size) {
    return LR_ERROR_INVALID_ARGUMENT;
  }
  QueueState &queue = execution_queue->state;
  std::vector<lr_event_t *> event_dependencies;
  if (!use_implicit_dependencies) {
    lr_status_t dependency_status = collect_event_dependencies_locked(
        device, explicit_dependencies, dependency_count, nullptr,
        &event_dependencies);
    if (dependency_status != LR_SUCCESS) {
      return dependency_status;
    }
  }
  KernargBuffer kernarg{};
  hsa_status_t status = acquire_kernarg_locked(&queue, state.kernarg_region,
                                               kernel->kernarg_size, &kernarg);
  if (status != HSA_STATUS_SUCCESS) {
    return to_lr_status(status);
  }
  std::memset(kernarg.ptr, 0, kernel->kernarg_size);
  std::memcpy(kernarg.ptr, args, args_size);

  hsa_signal_t signal{};
  status = acquire_signal_locked(&queue, &signal);
  if (status != HSA_STATUS_SUCCESS) {
    queue.kernarg_pool.push_back(kernarg);
    return to_lr_status(status);
  }
  const std::vector<lr_event_t *> *dependencies =
      use_implicit_dependencies ? nullptr : &event_dependencies;
  lr_status_t dependency_status =
      enqueue_event_dependencies_locked(&state, &queue, signal, dependencies);
  if (dependency_status != LR_SUCCESS) {
    queue.signal_pool.push_back(signal);
    queue.kernarg_pool.push_back(kernarg);
    return dependency_status;
  }
  // Keep packets on the same lrrt queue completion-ordered. Several executor
  // pipelines pass one kernel's output directly to the next kernel.
  const bool wait_for_dependencies =
      !queue.pending_dispatches.empty() ||
      (use_implicit_dependencies ? !queue.pending_barriers.empty()
                                 : !event_dependencies.empty());

  const uint64_t index = hsa_queue_add_write_index_scacq_screl(queue.queue, 1);
  auto *packets =
      static_cast<hsa_kernel_dispatch_packet_t *>(queue.queue->base_address);
  hsa_kernel_dispatch_packet_t *packet =
      &packets[index & (queue.queue->size - 1)];
  std::memset(packet, 0, sizeof(*packet));
  packet->setup = packet_setup(dispatch_dimensions(config));
  packet->workgroup_size_x = static_cast<uint16_t>(config->block.x);
  packet->workgroup_size_y = static_cast<uint16_t>(config->block.y);
  packet->workgroup_size_z = static_cast<uint16_t>(config->block.z);
  packet->grid_size_x = config->grid.x;
  packet->grid_size_y = config->grid.y;
  packet->grid_size_z = config->grid.z;
  packet->private_segment_size = kernel->private_segment_size;
  packet->group_segment_size =
      kernel->group_segment_size + config->shared_memory_bytes;
  packet->kernel_object = kernel->object;
  packet->kernarg_address = kernarg.ptr;
  packet->completion_signal = signal;
  uint16_t header = wait_for_dependencies
                        ? barrier_packet_header(HSA_PACKET_TYPE_KERNEL_DISPATCH)
                        : packet_header(HSA_PACKET_TYPE_KERNEL_DISPATCH);
  publish_packet_header(&packet->header, header);

  hsa_signal_store_screlease(queue.queue->doorbell_signal, index);
  queue.pending_dispatches.push_back(PendingDispatch{signal, kernarg});
  return LR_SUCCESS;
#else
  return LR_ERROR_NOT_SUPPORTED;
#endif
}

lr_status_t lr_launch(lr_kernel_t *kernel, const lr_launch_config_t *config,
                      const void *args, size_t args_size) {
  return launch_impl(kernel, config, args, args_size, nullptr, true, nullptr, 0,
                     true);
}

lr_status_t lr_launch_with_dependencies(lr_kernel_t *kernel,
                                        const lr_launch_config_t *config,
                                        const void *args, size_t args_size,
                                        lr_event_t *const *dependencies,
                                        size_t dependency_count) {
  return launch_impl(kernel, config, args, args_size, nullptr, true,
                     dependencies, dependency_count, false);
}

lr_status_t lr_launch_on_queue(lr_queue_t *queue, lr_kernel_t *kernel,
                               const lr_launch_config_t *config,
                               const void *args, size_t args_size) {
  return launch_impl(kernel, config, args, args_size, queue, false, nullptr, 0,
                     false);
}

lr_status_t lr_launch_on_queue_with_dependencies(
    lr_queue_t *queue, lr_kernel_t *kernel, const lr_launch_config_t *config,
    const void *args, size_t args_size, lr_event_t *const *dependencies,
    size_t dependency_count) {
  return launch_impl(kernel, config, args, args_size, queue, false,
                     dependencies, dependency_count, false);
}

} // extern "C"
