#include "light_rocr/transport/kfd/signal.hpp"

#include "light_rocr/transport/kfd/session.hpp"

#include "session_state.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>

namespace light_rocr::transport::kfd {

struct SignalPage {
  static constexpr size_t kSlotCount =
      kMemoryPageSize / sizeof(runtime::AmdSignal);

  SignalPage(uint32_t page_gpu_id, MemoryAllocation page_allocation)
      : gpu_id(page_gpu_id), allocation(std::move(page_allocation)) {}

  [[nodiscard]] bool acquire(int64_t initial_value, size_t *slot) {
    const std::lock_guard<std::mutex> lock(mutex);
    if (!allocation || allocation.host_address() == nullptr ||
        allocation.gpu_address() == 0) {
      return false;
    }
    for (size_t index = 0; index < slots.size(); ++index) {
      if (slots[index]) {
        continue;
      }
      slots[index] = true;
      ++active_slots;
      auto *signal =
          static_cast<runtime::AmdSignal *>(allocation.host_address()) + index;
      ::new (signal) runtime::AmdSignal;
      runtime::initialize_user_signal(*signal, initial_value);
      *slot = index;
      return true;
    }
    return false;
  }

  [[nodiscard]] MemoryStatus release(size_t slot) {
    const std::lock_guard<std::mutex> lock(mutex);
    if (slot >= slots.size() || !slots[slot]) {
      return {};
    }
    if (active_slots == 1) {
      const MemoryStatus status = allocation.release();
      if (!status) {
        return status;
      }
    }
    slots[slot] = false;
    --active_slots;
    return {};
  }

  [[nodiscard]] runtime::AmdSignal *host_address(size_t slot) const {
    return static_cast<runtime::AmdSignal *>(allocation.host_address()) + slot;
  }

  [[nodiscard]] uint64_t gpu_address(size_t slot) const {
    return allocation.gpu_address() + slot * sizeof(runtime::AmdSignal);
  }

  uint32_t gpu_id = 0;
  MemoryAllocation allocation;
  mutable std::mutex mutex;
  std::array<bool, kSlotCount> slots{};
  size_t active_slots = 0;
};

namespace {

UserSignalStatus allocation_failure(const MemoryStatus &status) {
  return {UserSignalError::AllocateStorage, status.system_error,
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

UserSignalResult
KfdSession::create_user_signal(const runtime::Node &node, int64_t initial_value,
                               const std::string &dri_root) const {
  if (state_ == nullptr) {
    return {{UserSignalError::InvalidSession, 0, "KFD session is not open"},
            {}};
  }

  {
    const std::lock_guard<std::mutex> lock(state_->mutex);
    auto page = state_->signal_pages.begin();
    while (page != state_->signal_pages.end()) {
      std::shared_ptr<SignalPage> retained = page->lock();
      if (!retained) {
        page = state_->signal_pages.erase(page);
        continue;
      }
      if (retained->gpu_id == node.gpu_id) {
        size_t slot = 0;
        if (retained->acquire(initial_value, &slot)) {
          return {{}, UserSignal(std::move(retained), slot)};
        }
      }
      ++page;
    }
  }

  auto allocated = allocate_gtt(node, kMemoryPageSize, dri_root);
  if (!allocated) {
    if (!allocated.allocation) {
      return {allocation_failure(allocated.status), {}};
    }
    try {
      auto page = std::make_shared<SignalPage>(node.gpu_id,
                                               std::move(allocated.allocation));
      page->slots[0] = true;
      page->active_slots = 1;
      return {allocation_failure(allocated.status),
              UserSignal(std::move(page), 0)};
    } catch (const std::bad_alloc &) {
      return {allocation_failure(allocated.status), {}};
    }
  }

  const auto host_address =
      reinterpret_cast<uintptr_t>(allocated.allocation.host_address());
  const uint64_t gpu_address = allocated.allocation.gpu_address();
  if (host_address % alignof(runtime::AmdSignal) != 0 ||
      gpu_address % alignof(runtime::AmdSignal) != 0) {
    try {
      auto page = std::make_shared<SignalPage>(node.gpu_id,
                                               std::move(allocated.allocation));
      page->slots[0] = true;
      page->active_slots = 1;
      return {{UserSignalError::MisalignedStorage, 0,
               "user-signal CPU and GPU addresses must be 64-byte aligned"},
              UserSignal(std::move(page), 0)};
    } catch (const std::bad_alloc &) {
      return {{UserSignalError::AllocateStorage, 0,
               "failed to retain misaligned user-signal storage"},
              {}};
    }
  }

  try {
    auto page = std::make_shared<SignalPage>(node.gpu_id,
                                             std::move(allocated.allocation));
    size_t slot = 0;
    const bool acquired = page->acquire(initial_value, &slot);
    if (!acquired) {
      return {{UserSignalError::AllocateStorage, 0,
               "new user-signal page has no available slot"},
              {}};
    }
    {
      const std::lock_guard<std::mutex> lock(state_->mutex);
      state_->signal_pages.emplace_back(page);
    }
    return {{}, UserSignal(std::move(page), slot)};
  } catch (const std::bad_alloc &) {
    return {{UserSignalError::AllocateStorage, 0,
             "failed to allocate user-signal page state"},
            {}};
  }
}

const runtime::AmdSignal *UserSignal::host_address() const {
  return page_ ? page_->host_address(slot_) : nullptr;
}

uint64_t UserSignal::gpu_handle() const {
  return page_ ? page_->gpu_address(slot_) : 0;
}

runtime::AmdSignal &UserSignal::abi() {
  assert(page_);
  return *page_->host_address(slot_);
}

const runtime::AmdSignal &UserSignal::abi() const {
  assert(page_);
  return *page_->host_address(slot_);
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
  if (!page_) {
    return {};
  }
  const MemoryStatus status = page_->release(slot_);
  if (!status) {
    return {UserSignalError::ReleaseStorage, status.system_error,
            "user-signal storage cleanup failed: " + status.message};
  }
  page_.reset();
  slot_ = 0;
  return {};
}

UserSignal::~UserSignal() { (void)release(); }

} // namespace light_rocr::transport::kfd
