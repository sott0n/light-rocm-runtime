#include "light_rocr/transport/kfd/session.hpp"

#include "session_state.hpp"

#include <linux/kfd_ioctl.h>

#include <cerrno>
#include <fcntl.h>
#include <mutex>
#include <new>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <unordered_map>

#ifndef AMDKFD_IOC_RUNTIME_ENABLE
struct kfd_ioctl_runtime_enable_args {
  __u64 r_debug;
  __u32 mode_mask;
  __u32 capabilities_mask;
};

#define KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK 1
#define AMDKFD_IOC_RUNTIME_ENABLE                                              \
  AMDKFD_IOWR(0x25, struct kfd_ioctl_runtime_enable_args)
#endif

namespace light_rocr::transport::kfd {

namespace {

struct SessionRegistry {
  std::mutex mutex;
  pid_t process_id = ::getpid();
  std::unordered_map<dev_t, std::shared_ptr<KfdState>> sessions;
};

SessionRegistry &session_registry() {
  static SessionRegistry registry;
  return registry;
}

void monitor_memory_faults(KfdState *state) {
  while (!state->stop_memory_fault_monitor.load(std::memory_order_acquire)) {
    kfd_event_data event{};
    event.event_id = state->memory_fault_event_id;
    kfd_ioctl_wait_events_args arguments{};
    arguments.events_ptr = reinterpret_cast<uint64_t>(&event);
    arguments.num_events = 1;
    arguments.wait_for_all = 1;
    arguments.timeout = 10;
    int result = 0;
    do {
      result = ::ioctl(state->fd, AMDKFD_IOC_WAIT_EVENTS, &arguments);
    } while (result < 0 && errno == EINTR);
    if (result == 0 && arguments.wait_result == KFD_IOC_WAIT_RESULT_COMPLETE) {
      state->memory_faulted.store(true, std::memory_order_release);
      return;
    }
    if (result < 0 || arguments.wait_result == KFD_IOC_WAIT_RESULT_FAIL) {
      return;
    }
  }
}

} // namespace

KfdState::KfdState(int opened_fd, runtime::KfdVersion queried_version)
    : fd(opened_fd), version(queried_version) {}

KfdState::~KfdState() {
  stop_memory_fault_monitor.store(true, std::memory_order_release);
  if (memory_fault_monitor.joinable()) {
    memory_fault_monitor.join();
  }
  for (auto &[gpu_id, device] : device_vms) {
    if (device.trap_handler_configured) {
      kfd_ioctl_set_trap_handler_args arguments{};
      arguments.gpu_id = gpu_id;
      (void)::ioctl(fd, AMDKFD_IOC_SET_TRAP_HANDLER, &arguments);
      device.trap_handler_configured = false;
    }
    if (device.trap_handler.has_value()) {
      (void)device.trap_handler->release();
      device.trap_handler.reset();
    }
  }
  if (queue_event_page != nullptr) {
    (void)queue_event_page->allocation.release();
    queue_event_page.reset();
  }
  if (memory_fault_event_active && fd >= 0) {
    kfd_ioctl_destroy_event_args arguments{};
    arguments.event_id = memory_fault_event_id;
    (void)::ioctl(fd, AMDKFD_IOC_DESTROY_EVENT, &arguments);
    memory_fault_event_active = false;
  }
  for (auto &[gpu_id, device] : device_vms) {
    (void)gpu_id;
    if (device.scratch_pool != nullptr &&
        device.scratch_pool->reservation_address != nullptr) {
      (void)::munmap(
          device.scratch_pool->reservation_address,
          static_cast<size_t>(device.scratch_pool->reservation_size));
    }
  }
  for (const auto &[gpu_id, device] : device_vms) {
    (void)gpu_id;
    if (device.render_fd >= 0) {
      (void)::close(device.render_fd);
    }
  }
  if (exception_debugging_enabled && fd >= 0) {
    kfd_ioctl_runtime_enable_args arguments{};
    (void)::ioctl(fd, AMDKFD_IOC_RUNTIME_ENABLE, &arguments);
    exception_debugging_enabled = false;
  }
  if (fd >= 0) {
    (void)::close(fd);
  }
}

namespace {

SessionStatus system_failure(SessionError error, int system_error,
                             const std::string &operation) {
  const std::error_code code(system_error, std::generic_category());
  return {error, system_error,
          operation + " failed: " + code.message() + " (" +
              std::to_string(system_error) + ")"};
}

bool version_less(runtime::KfdVersion left, runtime::KfdVersion right) {
  return left.major < right.major ||
         (left.major == right.major && left.minor < right.minor);
}

bool supports_runtime_enable(runtime::KfdVersion version) {
  return !version_less(version, runtime::KfdVersion{1, 13});
}

} // namespace

bool is_supported_kfd_version(runtime::KfdVersion version) {
  return version.major == kMinimumKfdVersion.major &&
         !version_less(version, kMinimumKfdVersion);
}

const char *session_error_name(SessionError error) {
  switch (error) {
  case SessionError::None:
    return "none";
  case SessionError::OpenKfd:
    return "open_kfd";
  case SessionError::InspectKfd:
    return "inspect_kfd";
  case SessionError::QueryKfdVersion:
    return "query_kfd_version";
  case SessionError::UnsupportedKfdVersion:
    return "unsupported_kfd_version";
  case SessionError::CreateMemoryFaultEvent:
    return "create_memory_fault_event";
  case SessionError::StartMemoryFaultMonitor:
    return "start_memory_fault_monitor";
  case SessionError::AllocateState:
    return "allocate_state";
  case SessionError::QueryClockCounters:
    return "query_clock_counters";
  }
  return "unknown";
}

SessionResult KfdSession::open(const std::string &device_path) {
  int fd = -1;
  do {
    fd = ::open(device_path.c_str(), O_RDWR | O_CLOEXEC);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    const int error = errno;
    return {system_failure(SessionError::OpenKfd, error,
                           "open(" + device_path + ")"),
            {}};
  }

  struct stat device_info{};
  if (::fstat(fd, &device_info) != 0) {
    const int error = errno;
    (void)::close(fd);
    return {system_failure(SessionError::InspectKfd, error,
                           "fstat(" + device_path + ")"),
            {}};
  }

  kfd_ioctl_get_version_args arguments{};
  int result = 0;
  do {
    result = ::ioctl(fd, AMDKFD_IOC_GET_VERSION, &arguments);
  } while (result < 0 && errno == EINTR);
  if (result < 0) {
    const int error = errno;
    (void)::close(fd);
    return {system_failure(SessionError::QueryKfdVersion, error,
                           "AMDKFD_IOC_GET_VERSION"),
            {}};
  }

  const runtime::KfdVersion version{arguments.major_version,
                                    arguments.minor_version};
  if (!is_supported_kfd_version(version)) {
    (void)::close(fd);
    return {{SessionError::UnsupportedKfdVersion, 0,
             "unsupported KFD UAPI version " + std::to_string(version.major) +
                 "." + std::to_string(version.minor) +
                 "; expected 1.x with x >= " +
                 std::to_string(kMinimumKfdVersion.minor)},
            {}};
  }

  SessionRegistry &registry = session_registry();
  const std::lock_guard<std::mutex> lock(registry.mutex);
  const pid_t process_id = ::getpid();
  if (registry.process_id != process_id) {
    registry.sessions.clear();
    registry.process_id = process_id;
  }

  const auto registered = registry.sessions.find(device_info.st_rdev);
  if (registered != registry.sessions.end()) {
    (void)::close(fd);
    return {{}, KfdSession(registered->second)};
  }

  std::shared_ptr<KfdState> state;
  try {
    state = std::make_shared<KfdState>(fd, version);
    if (supports_runtime_enable(version)) {
      kfd_ioctl_runtime_enable_args runtime_arguments{};
      runtime_arguments.r_debug =
          reinterpret_cast<uint64_t>(&state->runtime_debug);
      runtime_arguments.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
      do {
        result = ::ioctl(fd, AMDKFD_IOC_RUNTIME_ENABLE, &runtime_arguments);
      } while (result < 0 && errno == EINTR);
      state->exception_debugging_enabled = result == 0;
    }
    kfd_ioctl_create_event_args event_arguments{};
    event_arguments.event_type = KFD_IOC_EVENT_MEMORY;
    event_arguments.auto_reset = 0;
    do {
      result = ::ioctl(fd, AMDKFD_IOC_CREATE_EVENT, &event_arguments);
    } while (result < 0 && errno == EINTR);
    if (result < 0) {
      return {system_failure(SessionError::CreateMemoryFaultEvent, errno,
                             "AMDKFD_IOC_CREATE_EVENT(memory fault)"),
              {}};
    }
    state->memory_fault_event_id = event_arguments.event_id;
    state->memory_fault_event_active = true;
    try {
      state->memory_fault_monitor =
          std::thread(monitor_memory_faults, state.get());
    } catch (const std::system_error &error) {
      return {{SessionError::StartMemoryFaultMonitor, error.code().value(),
               "failed to start KFD memory fault monitor: " +
                   std::string(error.what())},
              {}};
    }
    registry.sessions[device_info.st_rdev] = state;
    return {{}, KfdSession(std::move(state))};
  } catch (const std::bad_alloc &) {
    if (state == nullptr) {
      (void)::close(fd);
    }
    return {{SessionError::AllocateState, 0,
             "failed to allocate direct KFD session state"},
            {}};
  }
}

namespace detail {

bool memory_fault_acquire(const std::shared_ptr<KfdState> &state) {
  return state != nullptr &&
         state->memory_faulted.load(std::memory_order_acquire);
}

} // namespace detail

runtime::KfdVersion KfdSession::version() const {
  return state_ ? state_->version : runtime::KfdVersion{};
}

ClockCountersResult
KfdSession::query_clock_counters(const runtime::Node &node) const {
  if (!state_) {
    return {{SessionError::QueryClockCounters, 0,
             "clock counters require an open KFD session"},
            {}};
  }
  if (node.gpu_id == 0) {
    return {{SessionError::QueryClockCounters, 0,
             "clock counters require a GPU node"},
            {}};
  }

  kfd_ioctl_get_clock_counters_args arguments{};
  arguments.gpu_id = node.gpu_id;
  int result = 0;
  do {
    result = ::ioctl(state_->fd, AMDKFD_IOC_GET_CLOCK_COUNTERS, &arguments);
  } while (result < 0 && errno == EINTR);
  if (result < 0) {
    return {system_failure(SessionError::QueryClockCounters, errno,
                           "AMDKFD_IOC_GET_CLOCK_COUNTERS"),
            {}};
  }
  if (arguments.system_clock_freq == 0) {
    return {{SessionError::QueryClockCounters, 0,
             "KFD returned a zero system clock frequency"},
            {}};
  }
  return {{},
          {arguments.gpu_clock_counter, arguments.cpu_clock_counter,
           arguments.system_clock_counter, arguments.system_clock_freq}};
}

} // namespace light_rocr::transport::kfd
