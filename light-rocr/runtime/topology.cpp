#include "light_rocr/runtime/topology.hpp"

#include <cctype>
#include <iomanip>
#include <sstream>

namespace light_rocr::runtime {

bool Node::is_gpu() const { return simd_count != 0; }

uint32_t Node::compute_unit_count() const {
  if (simd_per_compute_unit == 0 || simd_count % simd_per_compute_unit != 0) {
    return 0;
  }
  return simd_count / simd_per_compute_unit;
}

std::string gfx_target_name(GpuArchitecture architecture) {
  if (architecture.major == 0) {
    return {};
  }

  std::ostringstream target;
  target << "gfx" << architecture.major << architecture.minor << std::hex
         << std::nouppercase << architecture.stepping;
  return target.str();
}

std::string canonical_gfx_target(const std::string &target_isa) {
  const size_t architecture_separator = target_isa.rfind("--");
  const size_t target_begin =
      architecture_separator == std::string::npos
          ? 0
          : architecture_separator + std::string("--").size();
  const size_t feature_separator = target_isa.find(':', target_begin);
  const std::string target =
      target_isa.substr(target_begin, feature_separator == std::string::npos
                                          ? std::string::npos
                                          : feature_separator - target_begin);

  if (target.size() < 5 || target.compare(0, 3, "gfx") != 0) {
    return {};
  }
  for (size_t index = 3; index < target.size(); ++index) {
    const unsigned char character = static_cast<unsigned char>(target[index]);
    if (std::isxdigit(character) == 0 || std::isupper(character) != 0) {
      return {};
    }
  }
  return target;
}

const char *memory_heap_type_name(MemoryHeapType heap_type) {
  switch (heap_type) {
  case MemoryHeapType::System:
    return "system";
  case MemoryHeapType::FrameBufferPublic:
    return "frame_buffer_public";
  case MemoryHeapType::FrameBufferPrivate:
    return "frame_buffer_private";
  case MemoryHeapType::Gds:
    return "gds";
  case MemoryHeapType::Lds:
    return "lds";
  case MemoryHeapType::Scratch:
    return "scratch";
  case MemoryHeapType::DeviceSvm:
    return "device_svm";
  case MemoryHeapType::MmioRemap:
    return "mmio_remap";
  case MemoryHeapType::Unknown:
    return "unknown";
  }
  return "unknown";
}

const char *gpu_selection_error_name(GpuSelectionError error) {
  switch (error) {
  case GpuSelectionError::None:
    return "none";
  case GpuSelectionError::InvalidTarget:
    return "invalid_target";
  case GpuSelectionError::NoMatchingGpu:
    return "no_matching_gpu";
  case GpuSelectionError::MultipleMatchingGpus:
    return "multiple_matching_gpus";
  }
  return "unknown";
}

GpuSelectionResult select_unique_gpu(const Topology &topology,
                                     const std::string &target_isa) {
  const std::string target = canonical_gfx_target(target_isa);
  if (target.empty()) {
    return {GpuSelectionError::InvalidTarget, 0,
            "invalid AMDGPU target ISA: " + target_isa};
  }

  size_t matching_node_index = 0;
  size_t match_count = 0;
  for (size_t index = 0; index < topology.nodes.size(); ++index) {
    const Node &node = topology.nodes[index];
    if (node.is_gpu() && gfx_target_name(node.architecture) == target) {
      matching_node_index = index;
      ++match_count;
    }
  }

  if (match_count == 0) {
    return {GpuSelectionError::NoMatchingGpu, 0,
            "no GPU node matches target " + target};
  }
  if (match_count != 1) {
    return {GpuSelectionError::MultipleMatchingGpus, 0,
            "target " + target + " matches " + std::to_string(match_count) +
                " GPU nodes; the initial runtime requires exactly one"};
  }
  return {GpuSelectionError::None, matching_node_index, {}};
}

} // namespace light_rocr::runtime
