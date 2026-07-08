#include "iree_adapter.hpp"

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using lrrt::executor::iree::BindingMetadata;
using lrrt::executor::iree::Buffer;
using lrrt::executor::iree::CommandQueue;
using lrrt::executor::iree::Device;
using lrrt::executor::iree::Executable;
using lrrt::executor::iree::ExecutableMetadata;
using lrrt::executor::iree::ExportMetadata;
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

ExecutableMetadata minimal_mul_metadata() {
  ExportMetadata export_metadata;
  export_metadata.symbol = "simple_mul_dispatch_0_elementwise_4_f32";
  export_metadata.ordinal = 0;
  export_metadata.workgroup_size = {32, 1, 1};
  export_metadata.subgroup_size = 32;
  export_metadata.bindings = {
      BindingMetadata{0, "storage_buffer", {"ReadOnly", "Indirect"}},
      BindingMetadata{1, "storage_buffer", {"ReadOnly", "Indirect"}},
      BindingMetadata{2, "storage_buffer", {"Indirect"}},
  };
  export_metadata.kernel.symbol = "simple_mul_dispatch_0_elementwise_4_f32";
  export_metadata.kernel.attributes = {
      "gpu.known_block_size",
      "rocdl.flat_work_group_size",
      "rocdl.kernel",
      "rocdl.reqd_work_group_size",
  };
  export_metadata.dispatch.executable = "simple_mul_dispatch_0";
  export_metadata.dispatch.variant = "rocm_hsaco_fb";
  export_metadata.dispatch.symbol = "simple_mul_dispatch_0_elementwise_4_f32";

  ExecutableMetadata metadata;
  metadata.target = "gfx1101";
  metadata.executable = "simple_mul_dispatch_0";
  metadata.variant = "rocm_hsaco_fb";
  metadata.exports = {std::move(export_metadata)};
  return metadata;
}

int test_metadata_contract() {
  const ExecutableMetadata metadata = minimal_mul_metadata();
  if (metadata.target != "gfx1101" || metadata.exports.size() != 1) {
    return 1;
  }

  const ExportMetadata &export_metadata = metadata.exports[0];
  if (export_metadata.bindings.size() != 3) {
    return 1;
  }
  if (!export_metadata.bindings[0].has_flag("ReadOnly") ||
      !export_metadata.bindings[0].has_flag("Indirect")) {
    return 1;
  }
  if (export_metadata.bindings[2].has_flag("ReadOnly")) {
    return 1;
  }
  if (!export_metadata.kernel.has_attribute("rocdl.kernel")) {
    return 1;
  }

  const lr_launch_config_t config =
      export_metadata.launch_config(lr_dim3_t{1, 1, 1}, 128);
  if (config.grid.x != 1 || config.grid.y != 1 || config.grid.z != 1) {
    return 1;
  }
  if (config.block.x != 32 || config.block.y != 1 || config.block.z != 1) {
    return 1;
  }
  if (config.shared_memory_bytes != 128) {
    return 1;
  }

  return 0;
}

} // namespace

int main() {
  if (test_unsupported_feature_message() != 0) {
    return 1;
  }
  if (test_dispatch_signature() != 0) {
    return 1;
  }
  if (test_metadata_contract() != 0) {
    return 1;
  }
  return 0;
}
