#include "iree_adapter.hpp"
#include "metadata_json.hpp"

#include <array>
#include <stdexcept>
#include <stdint.h>
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
using lrrt::executor::iree::KernargBuilder;
using lrrt::executor::iree::parse_executable_metadata_json;
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

const char *kMinimalMulMetadataJson = R"json(
{
  "target": "gfx1101",
  "executable": "simple_mul_dispatch_0",
  "variant": "rocm_hsaco_fb",
  "exports": [
    {
      "symbol": "simple_mul_dispatch_0_elementwise_4_f32",
      "ordinal": 0,
      "workgroup_size": [32, 1, 1],
      "subgroup_size": 32,
      "bindings": [
        {"index": 0, "type": "storage_buffer", "flags": ["ReadOnly", "Indirect"]},
        {"index": 1, "type": "storage_buffer", "flags": ["ReadOnly", "Indirect"]},
        {"index": 2, "type": "storage_buffer", "flags": ["Indirect"]}
      ],
      "kernel": {
        "symbol": "simple_mul_dispatch_0_elementwise_4_f32",
        "attributes": [
          "gpu.known_block_size",
          "rocdl.flat_work_group_size",
          "rocdl.kernel",
          "rocdl.reqd_work_group_size"
        ]
      },
      "dispatch": {
        "executable": "simple_mul_dispatch_0",
        "variant": "rocm_hsaco_fb",
        "symbol": "simple_mul_dispatch_0_elementwise_4_f32"
      }
    }
  ]
}
)json";

const char *kMultiExecutableMetadataJson = R"json(
{
  "target": "gfx1101",
  "executables": [
    {
      "executable": "mixed_matmuls_dispatch_0",
      "variant": "rocm_hsaco_fb",
      "exports": [
        {
          "symbol": "mixed_matmuls_dispatch_0_matmul_2x2x2_f32",
          "ordinal": 0,
          "workgroup_size": [2, 2, 1],
          "subgroup_size": 32,
          "bindings": [
            {"index": 0, "type": "storage_buffer", "flags": ["ReadOnly", "Indirect"]},
            {"index": 1, "type": "storage_buffer", "flags": ["ReadOnly", "Indirect"]},
            {"index": 2, "type": "storage_buffer", "flags": ["Indirect"]}
          ],
          "kernel": {
            "symbol": "mixed_matmuls_dispatch_0_matmul_2x2x2_f32",
            "attributes": ["rocdl.kernel"]
          },
          "dispatch": {
            "executable": "mixed_matmuls_dispatch_0",
            "variant": "rocm_hsaco_fb",
            "symbol": "mixed_matmuls_dispatch_0_matmul_2x2x2_f32"
          }
        }
      ]
    },
    {
      "executable": "mixed_matmuls_dispatch_1",
      "variant": "rocm_hsaco_fb",
      "exports": [
        {
          "symbol": "mixed_matmuls_dispatch_1_matmul_2x3x2_f32",
          "ordinal": 0,
          "workgroup_size": [2, 3, 1],
          "subgroup_size": 32,
          "bindings": [
            {"index": 0, "type": "storage_buffer", "flags": ["ReadOnly", "Indirect"]},
            {"index": 1, "type": "storage_buffer", "flags": ["ReadOnly", "Indirect"]},
            {"index": 2, "type": "storage_buffer", "flags": ["Indirect"]}
          ],
          "kernel": {
            "symbol": "mixed_matmuls_dispatch_1_matmul_2x3x2_f32",
            "attributes": ["rocdl.kernel"]
          },
          "dispatch": {
            "executable": "mixed_matmuls_dispatch_1",
            "variant": "rocm_hsaco_fb",
            "symbol": "mixed_matmuls_dispatch_1_matmul_2x3x2_f32"
          }
        }
      ]
    }
  ]
}
)json";

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

  const lr_launch_config_t workgroup_config =
      export_metadata.launch_config_for_workgroups(lr_dim3_t{2, 3, 4});
  if (workgroup_config.grid.x != 64 || workgroup_config.grid.y != 3 ||
      workgroup_config.grid.z != 4) {
    return 1;
  }
  if (workgroup_config.block.x != 32 || workgroup_config.block.y != 1 ||
      workgroup_config.block.z != 1) {
    return 1;
  }

  return 0;
}

int test_metadata_export_lookup() {
  const ExecutableMetadata metadata = minimal_mul_metadata();
  const char *symbol = "simple_mul_dispatch_0_elementwise_4_f32";

  const ExportMetadata *by_symbol = metadata.find_export_by_symbol(symbol);
  const ExportMetadata *by_ordinal = metadata.find_export_by_ordinal(0);
  if (!by_symbol || !by_ordinal || by_symbol != by_ordinal) {
    return 1;
  }

  const ExportMetadata &required_by_symbol =
      metadata.require_export_by_symbol(symbol);
  const ExportMetadata &required_by_ordinal =
      metadata.require_export_by_ordinal(0);
  if (&required_by_symbol != by_symbol || &required_by_ordinal != by_ordinal) {
    return 1;
  }

  if (metadata.find_export_by_symbol("missing") != nullptr ||
      metadata.find_export_by_ordinal(7) != nullptr) {
    return 1;
  }

  try {
    metadata.require_export_by_symbol("missing");
  } catch (const std::runtime_error &error) {
    const std::string message = error.what();
    if (message.find("missing IREE executable export symbol") ==
        std::string::npos) {
      return 1;
    }
    return 0;
  }

  return 1;
}

int test_kernarg_builder() {
  const ExecutableMetadata metadata = minimal_mul_metadata();
  const ExportMetadata &export_metadata = metadata.require_export_by_ordinal(0);
  KernargBuilder builder(export_metadata);

  void *lhs = reinterpret_cast<void *>(static_cast<uintptr_t>(0x1000));
  void *rhs = reinterpret_cast<void *>(static_cast<uintptr_t>(0x2000));
  void *out = reinterpret_cast<void *>(static_cast<uintptr_t>(0x3000));
  const std::vector<void *> buffers = {lhs, rhs, out};
  const std::vector<unsigned char> kernargs =
      builder.pack_global_buffers(buffers);

  if (kernargs.size() != 3 * sizeof(void *)) {
    return 1;
  }
  const void *const *packed =
      reinterpret_cast<const void *const *>(kernargs.data());
  if (packed[0] != lhs || packed[1] != rhs || packed[2] != out) {
    return 1;
  }

  try {
    builder.pack_global_buffers({lhs});
  } catch (const std::runtime_error &error) {
    const std::string message = error.what();
    if (message.find("IREE kernarg buffer count mismatch") ==
        std::string::npos) {
      return 1;
    }
    return 0;
  }

  return 1;
}

int test_metadata_json_parser() {
  const ExecutableMetadata metadata =
      parse_executable_metadata_json(kMinimalMulMetadataJson);
  if (metadata.target != "gfx1101" ||
      metadata.executable != "simple_mul_dispatch_0" ||
      metadata.variant != "rocm_hsaco_fb" || metadata.exports.size() != 1) {
    return 1;
  }

  const ExportMetadata &export_metadata = metadata.require_export_by_ordinal(0);
  if (export_metadata.symbol != "simple_mul_dispatch_0_elementwise_4_f32" ||
      export_metadata.workgroup_size != std::array<uint32_t, 3>{32, 1, 1} ||
      export_metadata.subgroup_size != 32 ||
      export_metadata.bindings.size() != 3) {
    return 1;
  }
  if (!export_metadata.bindings[0].has_flag("ReadOnly") ||
      !export_metadata.bindings[2].has_flag("Indirect") ||
      export_metadata.kernel.symbol != export_metadata.symbol ||
      !export_metadata.kernel.has_attribute("rocdl.kernel") ||
      export_metadata.dispatch.symbol != export_metadata.symbol) {
    return 1;
  }

  try {
    parse_executable_metadata_json(R"json({"target":"gfx1101"})json");
  } catch (const std::runtime_error &error) {
    const std::string message = error.what();
    return message.find("missing IREE metadata field: executable") !=
                   std::string::npos
               ? 0
               : 1;
  }
  return 1;
}

int test_multi_executable_metadata_json_parser() {
  const ExecutableMetadata metadata =
      parse_executable_metadata_json(kMultiExecutableMetadataJson);
  if (metadata.target != "gfx1101" ||
      metadata.executable != "mixed_matmuls_dispatch_0" ||
      metadata.variant != "rocm_hsaco_fb" || metadata.exports.size() != 2) {
    return 1;
  }

  const ExportMetadata &first = metadata.require_export_by_symbol(
      "mixed_matmuls_dispatch_0_matmul_2x2x2_f32");
  const ExportMetadata &second = metadata.require_export_by_symbol(
      "mixed_matmuls_dispatch_1_matmul_2x3x2_f32");
  if (first.dispatch.executable != "mixed_matmuls_dispatch_0" ||
      first.workgroup_size != std::array<uint32_t, 3>{2, 2, 1} ||
      second.dispatch.executable != "mixed_matmuls_dispatch_1" ||
      second.workgroup_size != std::array<uint32_t, 3>{2, 3, 1}) {
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
  if (test_metadata_export_lookup() != 0) {
    return 1;
  }
  if (test_kernarg_builder() != 0) {
    return 1;
  }
  if (test_metadata_json_parser() != 0) {
    return 1;
  }
  if (test_multi_executable_metadata_json_parser() != 0) {
    return 1;
  }
  return 0;
}
