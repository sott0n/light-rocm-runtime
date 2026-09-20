#ifndef LIGHT_ROCR_TRANSPORT_KFD_SESSION_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_SESSION_HPP

#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/kfd/memory_types.hpp"
#include "light_rocr/transport/kfd/queue.hpp"
#include "light_rocr/transport/kfd/signal.hpp"
#include "light_rocr/transport/kfd/vm_types.hpp"

#include <memory>
#include <string>
#include <utility>

namespace light_rocr::transport::kfd {

inline constexpr runtime::KfdVersion kMinimumKfdVersion{1, 1};

enum class SessionError {
  None,
  OpenKfd,
  InspectKfd,
  QueryKfdVersion,
  UnsupportedKfdVersion,
  AllocateState,
};

struct SessionStatus {
  SessionError error = SessionError::None;
  int system_error = 0;
  std::string message;

  explicit operator bool() const { return error == SessionError::None; }
};

struct KfdState;
struct SessionResult;

class KfdSession {
public:
  KfdSession() = default;
  KfdSession(const KfdSession &) = delete;
  KfdSession &operator=(const KfdSession &) = delete;
  KfdSession(KfdSession &&) noexcept = default;
  KfdSession &operator=(KfdSession &&) noexcept = default;
  ~KfdSession() = default;

  [[nodiscard]] static SessionResult
  open(const std::string &device_path = "/dev/kfd");
  [[nodiscard]] VmResult
  acquire_vm(const runtime::Node &node,
             const std::string &dri_root = "/dev/dri") const;
  [[nodiscard]] GttAllocationResult
  allocate_gtt(const runtime::Node &node, uint64_t size,
               const std::string &dri_root = "/dev/dri") const;
  [[nodiscard]] GttAllocationResult
  allocate_executable_gtt(const runtime::Node &node, uint64_t size,
                          const std::string &dri_root = "/dev/dri") const;
  [[nodiscard]] AqlQueueResult
  create_aql_queue(const runtime::Node &node, uint64_t ring_size,
                   const std::string &dri_root = "/dev/dri") const;
  [[nodiscard]] UserSignalResult
  create_user_signal(const runtime::Node &node, int64_t initial_value,
                     const std::string &dri_root = "/dev/dri") const;
  [[nodiscard]] runtime::KfdVersion version() const;
  explicit operator bool() const { return state_ != nullptr; }

private:
  enum class GttUsage {
    General,
    Executable,
    AqlRing,
    Doorbell,
    Eop,
  };

  [[nodiscard]] GttAllocationResult
  allocate_gtt_impl(const runtime::Node &node, uint64_t size,
                    const std::string &dri_root, GttUsage usage) const;
  [[nodiscard]] MemoryStatus
  map_pending_allocation(GttAllocation *allocation) const;
  explicit KfdSession(std::shared_ptr<KfdState> state)
      : state_(std::move(state)) {}

  std::shared_ptr<KfdState> state_;
};

struct SessionResult {
  SessionStatus status;
  KfdSession session;

  explicit operator bool() const { return static_cast<bool>(status); }
};

[[nodiscard]] bool is_supported_kfd_version(runtime::KfdVersion version);
[[nodiscard]] const char *session_error_name(SessionError error);

} // namespace light_rocr::transport::kfd

#endif
