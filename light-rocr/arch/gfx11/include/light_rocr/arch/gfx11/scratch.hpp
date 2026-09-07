#ifndef LIGHT_ROCR_ARCH_GFX11_SCRATCH_HPP
#define LIGHT_ROCR_ARCH_GFX11_SCRATCH_HPP

#include "light_rocr/runtime/topology.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace light_rocr::arch::gfx11 {

inline constexpr uint32_t kScratchDescriptorAlignment = 256;
inline constexpr uint32_t kScratchBackingAlignment = 64 * 1024;
inline constexpr uint32_t kScratchLanesPerWave = 64;

enum class QueueScratchError {
  None,
  UnsupportedArchitecture,
  InvalidTopology,
  MissingScratchAperture,
  InvalidScratchAperture,
  PrivateSegmentTooLarge,
  SizeOverflow,
  ScratchApertureTooSmall,
  RegisterOverflow,
  InvalidBackingAddress,
};

struct QueueScratchStatus {
  QueueScratchError error = QueueScratchError::None;
  std::string message;

  explicit operator bool() const { return error == QueueScratchError::None; }
};

struct QueueScratchRequirements {
  uint32_t private_segment_size = 0;
  uint64_t allocation_size = 0;
  uint64_t backing_size_per_xcc = 0;
  uint32_t wave_size_units = 0;
  uint32_t waves_per_shader_engine = 0;
};

struct QueueScratchRequirementsResult {
  QueueScratchStatus status;
  QueueScratchRequirements requirements;

  explicit operator bool() const { return static_cast<bool>(status); }
};

struct QueueScratchControl {
  std::array<uint32_t, 4> resource_descriptor{};
  uint32_t compute_tmpring_size = 0;
  uint64_t backing_memory_location = 0;
  uint64_t backing_memory_byte_size = 0;
  uint32_t wave64_lane_byte_size = 0;
};

struct QueueScratchControlResult {
  QueueScratchStatus status;
  QueueScratchRequirements requirements;
  QueueScratchControl control;

  explicit operator bool() const { return static_cast<bool>(status); }
};

[[nodiscard]] QueueScratchRequirementsResult
queue_scratch_requirements(const runtime::Node &node,
                           uint32_t private_segment_size);

[[nodiscard]] QueueScratchControlResult
make_queue_scratch_control(const runtime::Node &node,
                           uint32_t private_segment_size,
                           uint64_t backing_gpu_address);

[[nodiscard]] const char *queue_scratch_error_name(QueueScratchError error);

} // namespace light_rocr::arch::gfx11

#endif
