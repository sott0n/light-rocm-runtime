#include "light_rocr/transport/hsakmt/signal.hpp"

#include <cassert>
#include <cstdint>
#include <new>
#include <string>
#include <utility>

namespace light_rocr::transport::hsakmt {
namespace {

UserSignalStatus allocation_failure(const MemoryStatus &status) {
  return {UserSignalError::AllocateStorage, status.hsakmt_status,
          "user-signal storage allocation failed: " + status.message};
}

} // namespace

const char *user_signal_error_name(UserSignalError error) {
  switch (error) {
  case UserSignalError::None:
    return "none";
  case UserSignalError::InvalidSession:
    return "invalid_session";
  case UserSignalError::AllocateStorage:
    return "allocate_storage";
  case UserSignalError::MisalignedStorage:
    return "misaligned_storage";
  case UserSignalError::ReleaseStorage:
    return "release_storage";
  }
  return "unknown";
}

UserSignalResult KfdSession::create_user_signal(uint32_t gpu_node_id,
                                                int64_t initial_value) const {
  if (state_ == nullptr) {
    return {{UserSignalError::InvalidSession, 0, "KFD session is not open"},
            {}};
  }

  auto allocated = allocate_gtt(gpu_node_id, kMemoryPageSize);
  if (!allocated) {
    return {allocation_failure(allocated.status), {}};
  }

  const auto host_address =
      reinterpret_cast<uintptr_t>(allocated.allocation.host_address());
  const uint64_t gpu_address = allocated.allocation.gpu_address();
  if (host_address % alignof(runtime::AmdSignal) != 0 ||
      gpu_address % alignof(runtime::AmdSignal) != 0) {
    return {{UserSignalError::MisalignedStorage, 0,
             "user-signal CPU and GPU addresses must be 64-byte aligned"},
            {}};
  }

  auto *signal = ::new (allocated.allocation.host_address()) runtime::AmdSignal;
  runtime::initialize_user_signal(*signal, initial_value);
  return {{}, UserSignal(std::move(allocated.allocation))};
}

const runtime::AmdSignal *UserSignal::host_address() const {
  return static_cast<const runtime::AmdSignal *>(storage_.host_address());
}

uint64_t UserSignal::gpu_handle() const { return storage_.gpu_address(); }

runtime::AmdSignal &UserSignal::abi() {
  assert(storage_);
  return *static_cast<runtime::AmdSignal *>(storage_.host_address());
}

const runtime::AmdSignal &UserSignal::abi() const {
  assert(storage_);
  return *static_cast<const runtime::AmdSignal *>(storage_.host_address());
}

int64_t UserSignal::load_relaxed() const {
  return runtime::signal_load_relaxed(abi());
}

int64_t UserSignal::load_acquire() const {
  return runtime::signal_load_acquire(abi());
}

void UserSignal::store_relaxed(int64_t value) {
  runtime::signal_store_relaxed(abi(), value);
}

void UserSignal::store_release(int64_t value) {
  runtime::signal_store_release(abi(), value);
}

runtime::SignalWaitResult UserSignal::wait_until_equal(
    int64_t expected_value,
    std::chrono::steady_clock::time_point deadline) const {
  return runtime::signal_wait_until_equal(abi(), expected_value, deadline);
}

UserSignalStatus UserSignal::release() {
  const MemoryStatus status = storage_.release();
  if (!status) {
    return {UserSignalError::ReleaseStorage, status.hsakmt_status,
            "user-signal storage cleanup failed: " + status.message};
  }
  return {};
}

} // namespace light_rocr::transport::hsakmt
