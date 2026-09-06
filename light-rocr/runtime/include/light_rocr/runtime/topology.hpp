#ifndef LIGHT_ROCR_RUNTIME_TOPOLOGY_HPP
#define LIGHT_ROCR_RUNTIME_TOPOLOGY_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace light_rocr::runtime {

struct KfdVersion {
  uint32_t major = 0;
  uint32_t minor = 0;
};

struct GpuArchitecture {
  uint32_t major = 0;
  uint32_t minor = 0;
  uint32_t stepping = 0;
};

enum class MemoryHeapType {
  System,
  FrameBufferPublic,
  FrameBufferPrivate,
  Gds,
  Lds,
  Scratch,
  DeviceSvm,
  MmioRemap,
  Unknown,
};

struct MemoryBank {
  MemoryHeapType heap_type = MemoryHeapType::Unknown;
  uint64_t size = 0;
  uint64_t virtual_base_address = 0;
  uint32_t width = 0;
  uint32_t maximum_clock_mhz = 0;
  bool hot_pluggable = false;
  bool non_volatile = false;
};

struct Node {
  uint32_t node_id = 0;
  uint32_t gpu_id = 0;
  uint32_t cpu_core_count = 0;
  uint32_t simd_count = 0;
  uint32_t simd_per_compute_unit = 0;
  uint32_t wavefront_size = 0;
  uint32_t maximum_waves_per_simd = 0;
  uint16_t vendor_id = 0;
  uint16_t device_id = 0;
  uint32_t domain = 0;
  uint32_t location_id = 0;
  int32_t drm_render_minor = -1;
  uint64_t local_memory_size = 0;
  bool integrated = false;
  std::string name;
  GpuArchitecture architecture;
  std::vector<MemoryBank> memory_banks;

  [[nodiscard]] bool is_gpu() const;
  [[nodiscard]] uint32_t compute_unit_count() const;
};

struct Topology {
  KfdVersion kfd_version;
  std::vector<Node> nodes;
};

enum class GpuSelectionError {
  None,
  InvalidTarget,
  NoMatchingGpu,
  MultipleMatchingGpus,
};

struct GpuSelectionResult {
  GpuSelectionError error = GpuSelectionError::None;
  size_t node_index = 0;
  std::string message;

  explicit operator bool() const { return error == GpuSelectionError::None; }
};

[[nodiscard]] std::string gfx_target_name(GpuArchitecture architecture);
[[nodiscard]] std::string canonical_gfx_target(const std::string &target_isa);
[[nodiscard]] const char *memory_heap_type_name(MemoryHeapType heap_type);
[[nodiscard]] const char *gpu_selection_error_name(GpuSelectionError error);
[[nodiscard]] GpuSelectionResult
select_unique_gpu(const Topology &topology, const std::string &target_isa);

} // namespace light_rocr::runtime

#endif
