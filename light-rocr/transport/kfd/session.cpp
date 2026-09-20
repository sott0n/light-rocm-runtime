#include "light_rocr/transport/kfd/session.hpp"

#include "session_state.hpp"

#include <linux/kfd_ioctl.h>

#include <cerrno>
#include <fcntl.h>
#include <mutex>
#include <new>
#include <string>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <unordered_map>

namespace light_rocr::transport::kfd {

namespace {

struct SessionRegistry {
  std::mutex mutex;
  pid_t process_id = ::getpid();
  std::unordered_map<dev_t, std::weak_ptr<KfdState>> sessions;
};

SessionRegistry &session_registry() {
  static SessionRegistry registry;
  return registry;
}

} // namespace

KfdState::KfdState(int opened_fd, runtime::KfdVersion queried_version)
    : fd(opened_fd), version(queried_version) {}

KfdState::~KfdState() {
  if (fd >= 0) {
    (void)::close(fd);
  }
  for (const auto &[gpu_id, device] : device_vms) {
    (void)gpu_id;
    if (device.render_fd >= 0) {
      (void)::close(device.render_fd);
    }
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
  case SessionError::AllocateState:
    return "allocate_state";
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
    std::shared_ptr<KfdState> state = registered->second.lock();
    if (state != nullptr) {
      (void)::close(fd);
      return {{}, KfdSession(std::move(state))};
    }
  }

  std::shared_ptr<KfdState> state;
  try {
    state = std::make_shared<KfdState>(fd, version);
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

runtime::KfdVersion KfdSession::version() const {
  return state_ ? state_->version : runtime::KfdVersion{};
}

} // namespace light_rocr::transport::kfd
