#include "light_rocr/transport/kfd/vm.hpp"

#include "session_state.hpp"
#include "vm_internal.hpp"

#include <linux/kfd_ioctl.h>

#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <new>
#include <string>
#include <sys/ioctl.h>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace light_rocr::transport::kfd {
namespace {

namespace fs = std::filesystem;

constexpr unsigned kMaximumApertureQueryAttempts = 3;
constexpr uint32_t kMaximumProcessApertures = 4096;

VmStatus system_failure(VmError error, int system_error,
                        const std::string &operation) {
  const std::error_code code(system_error, std::generic_category());
  return {error, system_error,
          operation + " failed: " + code.message() + " (" +
              std::to_string(system_error) + ")"};
}

int real_ioctl(int fd, unsigned long request, void *arguments) {
  return ::ioctl(fd, request, arguments);
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

ProcessAperture convert_aperture(const kfd_process_device_apertures &aperture) {
  return {aperture.gpu_id,       aperture.lds_base,      aperture.lds_limit,
          aperture.scratch_base, aperture.scratch_limit, aperture.gpuvm_base,
          aperture.gpuvm_limit};
}

bool valid_range(uint64_t base, uint64_t limit) { return base <= limit; }

VmStatus validate_aperture(const ProcessAperture &aperture) {
  if (aperture.gpu_id == 0 ||
      !valid_range(aperture.lds_base, aperture.lds_limit) ||
      !valid_range(aperture.scratch_base, aperture.scratch_limit) ||
      !valid_range(aperture.gpuvm_base, aperture.gpuvm_limit)) {
    return {VmError::InvalidAperture, 0,
            "KFD returned an invalid process aperture for GPU " +
                std::to_string(aperture.gpu_id)};
  }
  return {};
}

VmStatus query_aperture_count(int kfd_fd, uint32_t *count,
                              detail::IoctlFunction ioctl_function) {
  kfd_ioctl_get_process_apertures_new_args arguments{};
  int system_error = 0;
  if (!invoke_ioctl(kfd_fd, AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, &arguments,
                    ioctl_function, &system_error)) {
    return system_failure(VmError::QueryApertureCount, system_error,
                          "AMDKFD_IOC_GET_PROCESS_APERTURES_NEW(count)");
  }
  if (arguments.num_of_nodes == 0 ||
      arguments.num_of_nodes > kMaximumProcessApertures) {
    return {VmError::InvalidApertureCount, 0,
            "KFD returned invalid process aperture count " +
                std::to_string(arguments.num_of_nodes)};
  }
  *count = arguments.num_of_nodes;
  return {};
}

} // namespace

namespace detail {

VmResult query_process_aperture(int kfd_fd, uint32_t gpu_id,
                                IoctlFunction ioctl_function) {
  if (kfd_fd < 0 || gpu_id == 0 || ioctl_function == nullptr) {
    return {{VmError::InvalidSession, 0,
             "process aperture query requires an open KFD session and GPU"},
            {}};
  }

  for (unsigned attempt = 0; attempt < kMaximumApertureQueryAttempts;
       ++attempt) {
    uint32_t count_before = 0;
    VmStatus status =
        query_aperture_count(kfd_fd, &count_before, ioctl_function);
    if (!status) {
      return {std::move(status), {}};
    }

    std::vector<kfd_process_device_apertures> raw_apertures;
    try {
      raw_apertures.resize(count_before);
    } catch (const std::bad_alloc &) {
      return {{VmError::AllocateApertures, 0,
               "failed to allocate process aperture query storage"},
              {}};
    }

    kfd_ioctl_get_process_apertures_new_args arguments{};
    arguments.kfd_process_device_apertures_ptr = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(raw_apertures.data()));
    arguments.num_of_nodes = count_before;
    int system_error = 0;
    if (!invoke_ioctl(kfd_fd, AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, &arguments,
                      ioctl_function, &system_error)) {
      return {system_failure(VmError::QueryApertures, system_error,
                             "AMDKFD_IOC_GET_PROCESS_APERTURES_NEW(data)"),
              {}};
    }
    if (arguments.num_of_nodes > count_before) {
      return {{VmError::InvalidApertureCount, 0,
               "KFD returned more process apertures than requested"},
              {}};
    }

    uint32_t count_after = 0;
    status = query_aperture_count(kfd_fd, &count_after, ioctl_function);
    if (!status) {
      return {std::move(status), {}};
    }
    if (count_before != arguments.num_of_nodes ||
        arguments.num_of_nodes != count_after) {
      continue;
    }

    ProcessAperture selected;
    bool found = false;
    for (const auto &raw_aperture : raw_apertures) {
      const ProcessAperture aperture = convert_aperture(raw_aperture);
      status = validate_aperture(aperture);
      if (!status) {
        return {std::move(status), {}};
      }
      if (aperture.gpu_id != gpu_id) {
        continue;
      }
      if (found) {
        return {{VmError::DuplicateAperture, 0,
                 "KFD returned duplicate process apertures for GPU " +
                     std::to_string(gpu_id)},
                {}};
      }
      selected = aperture;
      found = true;
    }
    if (!found) {
      return {{VmError::MissingAperture, 0,
               "KFD returned no process aperture for GPU " +
                   std::to_string(gpu_id)},
              {}};
    }
    return {{}, selected};
  }

  return {{VmError::TopologyChanged, 0,
           "KFD process aperture count changed during all query attempts"},
          {}};
}

} // namespace detail

const char *vm_error_name(VmError error) {
  switch (error) {
  case VmError::None:
    return "none";
  case VmError::InvalidSession:
    return "invalid_session";
  case VmError::InvalidNode:
    return "invalid_node";
  case VmError::OpenRenderNode:
    return "open_render_node";
  case VmError::AllocateState:
    return "allocate_state";
  case VmError::AcquireVm:
    return "acquire_vm";
  case VmError::QueryApertureCount:
    return "query_aperture_count";
  case VmError::InvalidApertureCount:
    return "invalid_aperture_count";
  case VmError::AllocateApertures:
    return "allocate_apertures";
  case VmError::QueryApertures:
    return "query_apertures";
  case VmError::TopologyChanged:
    return "topology_changed";
  case VmError::MissingAperture:
    return "missing_aperture";
  case VmError::DuplicateAperture:
    return "duplicate_aperture";
  case VmError::InvalidAperture:
    return "invalid_aperture";
  }
  return "unknown";
}

VmResult KfdSession::acquire_vm(const runtime::Node &node,
                                const std::string &dri_root) const {
  if (state_ == nullptr) {
    return {{VmError::InvalidSession, 0,
             "VM acquisition requires an open KFD session"},
            {}};
  }
  if (!node.is_gpu() || node.gpu_id == 0 || node.drm_render_minor < 0) {
    return {{VmError::InvalidNode, 0,
             "VM acquisition requires a GPU node with a render minor"},
            {}};
  }

  const std::lock_guard<std::mutex> lock(state_->mutex);
  auto existing = state_->device_vms.find(node.gpu_id);
  if (existing != state_->device_vms.end()) {
    if (existing->second.drm_render_minor != node.drm_render_minor) {
      return {{VmError::InvalidNode, 0,
               "GPU ID is associated with a different DRM render node"},
              {}};
    }
    if (existing->second.aperture_valid) {
      return {{}, existing->second.aperture};
    }
  }

  if (existing == state_->device_vms.end()) {
    const fs::path render_path =
        fs::path(dri_root) /
        ("renderD" + std::to_string(node.drm_render_minor));
    int render_fd = -1;
    do {
      render_fd = ::open(render_path.c_str(), O_RDWR | O_CLOEXEC);
    } while (render_fd < 0 && errno == EINTR);
    if (render_fd < 0) {
      const int system_error = errno;
      return {system_failure(VmError::OpenRenderNode, system_error,
                             "open(" + render_path.string() + ")"),
              {}};
    }

    try {
      existing = state_->device_vms
                     .emplace(node.gpu_id, DeviceVmState{render_fd,
                                                         node.drm_render_minor,
                                                         false,
                                                         false,
                                                         false,
                                                         {}})
                     .first;
    } catch (const std::bad_alloc &) {
      (void)::close(render_fd);
      return {{VmError::AllocateState, 0,
               "failed to allocate VM acquisition bookkeeping"},
              {}};
    }
  }

  DeviceVmState &device = existing->second;
  if (!device.acquired) {
    kfd_ioctl_acquire_vm_args arguments{};
    arguments.drm_fd = static_cast<uint32_t>(device.render_fd);
    arguments.gpu_id = node.gpu_id;
    int system_error = 0;
    if (!invoke_ioctl(state_->fd, AMDKFD_IOC_ACQUIRE_VM, &arguments, real_ioctl,
                      &system_error)) {
      (void)::close(device.render_fd);
      state_->device_vms.erase(existing);
      return {system_failure(VmError::AcquireVm, system_error,
                             "AMDKFD_IOC_ACQUIRE_VM"),
              {}};
    }
    device.acquired = true;
  }

  VmResult queried =
      detail::query_process_aperture(state_->fd, node.gpu_id, real_ioctl);
  if (!queried) {
    return queried;
  }
  device.aperture = queried.aperture;
  device.aperture_valid = true;
  return queried;
}

} // namespace light_rocr::transport::kfd
