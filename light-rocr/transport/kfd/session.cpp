#include "light_rocr/transport/kfd/session.hpp"

#include "session_state.hpp"

#include <linux/kfd_ioctl.h>

#include <cerrno>
#include <fcntl.h>
#include <new>
#include <string>
#include <sys/ioctl.h>
#include <system_error>
#include <unistd.h>

namespace light_rocr::transport::kfd {

KfdState::KfdState(int opened_fd, runtime::KfdVersion queried_version)
    : fd(opened_fd), version(queried_version) {}

KfdState::~KfdState() {
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
  const int fd = ::open(device_path.c_str(), O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    const int error = errno;
    return {system_failure(SessionError::OpenKfd, error,
                           "open(" + device_path + ")"),
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

  try {
    return {{}, KfdSession(std::make_shared<KfdState>(fd, version))};
  } catch (const std::bad_alloc &) {
    (void)::close(fd);
    return {{SessionError::AllocateState, 0,
             "failed to allocate direct KFD session state"},
            {}};
  }
}

runtime::KfdVersion KfdSession::version() const {
  return state_ ? state_->version : runtime::KfdVersion{};
}

} // namespace light_rocr::transport::kfd
