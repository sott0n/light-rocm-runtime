#include "iree_adapter.hpp"

#include <string>
#include <type_traits>
#include <vector>

namespace {

using lrrt::executor::iree::Buffer;
using lrrt::executor::iree::CommandQueue;
using lrrt::executor::iree::Device;
using lrrt::executor::iree::Executable;
using lrrt::executor::iree::Fence;
using lrrt::executor::iree::UnsupportedFeature;

static_assert(!std::is_copy_constructible<Device>::value,
              "IREE adapter Device must not be copyable");
static_assert(!std::is_copy_assignable<Device>::value,
              "IREE adapter Device must not be copy-assignable");
static_assert(!std::is_copy_constructible<Buffer>::value,
              "IREE adapter Buffer must not be copyable");
static_assert(!std::is_copy_assignable<Buffer>::value,
              "IREE adapter Buffer must not be copy-assignable");
static_assert(!std::is_copy_constructible<Executable>::value,
              "IREE adapter Executable must not be copyable");
static_assert(!std::is_copy_assignable<Executable>::value,
              "IREE adapter Executable must not be copy-assignable");
static_assert(!std::is_copy_constructible<CommandQueue>::value,
              "IREE adapter CommandQueue must not be copyable");
static_assert(!std::is_copy_assignable<CommandQueue>::value,
              "IREE adapter CommandQueue must not be copy-assignable");
static_assert(!std::is_copy_constructible<Fence>::value,
              "IREE adapter Fence must not be copyable");
static_assert(!std::is_copy_assignable<Fence>::value,
              "IREE adapter Fence must not be copy-assignable");

int test_unsupported_feature_message() {
  try {
    lrrt::executor::iree::require_unsupported("timeline semaphore");
  } catch (const UnsupportedFeature &error) {
    const std::string message = error.what();
    return message.find("unsupported IREE HAL adapter feature") !=
                       std::string::npos &&
                   message.find("timeline semaphore") != std::string::npos
               ? 0
               : 1;
  }
  return 1;
}

int test_dispatch_signature() {
  using Dispatch = void (CommandQueue::*)(
      const lrrt::Kernel &, const lr_launch_config_t &, const void *, size_t,
      const std::vector<const Fence *> &) const;
  Dispatch dispatch = &CommandQueue::dispatch;
  return dispatch ? 0 : 1;
}

} // namespace

int main() {
  if (test_unsupported_feature_message() != 0) {
    return 1;
  }
  if (test_dispatch_signature() != 0) {
    return 1;
  }
  return 0;
}
