#ifndef LIGHT_ROCR_TRANSPORT_KFD_SIGNAL_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_SIGNAL_HPP

#include "light_rocr/runtime/signal.hpp"
#include "light_rocr/transport/kfd/memory_types.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace light_rocr::transport::kfd {

struct SignalPage;

enum class UserSignalError {
  None,
  InvalidSession,
  AllocateStorage,
  MisalignedStorage,
  ReleaseStorage,
};

struct UserSignalStatus {
  UserSignalError error = UserSignalError::None;
  int system_error = 0;
  std::string message;

  explicit operator bool() const { return error == UserSignalError::None; }
};

class UserSignal {
public:
  UserSignal() = default;
  UserSignal(const UserSignal &) = delete;
  UserSignal &operator=(const UserSignal &) = delete;
  UserSignal(UserSignal &&) noexcept = default;
  UserSignal &operator=(UserSignal &&) noexcept = delete;
  ~UserSignal();

  [[nodiscard]] const runtime::AmdSignal *host_address() const;
  [[nodiscard]] uint64_t gpu_handle() const;
  [[nodiscard]] int64_t load_relaxed() const;
  [[nodiscard]] int64_t load_acquire() const;
  void store_relaxed(int64_t value);
  void store_release(int64_t value);
  [[nodiscard]] runtime::SignalWaitResult
  wait_until_equal(int64_t expected_value,
                   std::chrono::steady_clock::time_point deadline) const;
  explicit operator bool() const { return page_ != nullptr; }

  [[nodiscard]] UserSignalStatus release();

private:
  friend class KfdSession;
  UserSignal(std::shared_ptr<SignalPage> page, size_t slot)
      : page_(std::move(page)), slot_(slot) {}
  [[nodiscard]] runtime::AmdSignal &abi();
  [[nodiscard]] const runtime::AmdSignal &abi() const;

  std::shared_ptr<SignalPage> page_;
  size_t slot_ = 0;
};

struct UserSignalResult {
  UserSignalStatus status;
  // A failed creation may retain partially allocated storage so release() can
  // retry cleanup without losing the KFD handle or reserved virtual address.
  UserSignal signal;

  explicit operator bool() const { return static_cast<bool>(status); }
};

[[nodiscard]] const char *user_signal_error_name(UserSignalError error);

} // namespace light_rocr::transport::kfd

#endif
