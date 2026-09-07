#include "light_rocr/arch/gfx11/scratch.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace light_rocr::arch::gfx11 {
namespace {

constexpr uint64_t kPageSize = 4096;
constexpr uint32_t kMaximumPrivateSegmentSize = 262128;
constexpr uint32_t kMaximumWaveSizeUnits = (1U << 15U) - 1U;
constexpr uint32_t kMaximumWavesPerShaderEngine = (1U << 12U) - 1U;
constexpr uint64_t kMaximumDescriptorAddress = (uint64_t{1} << 48U) - 1U;
constexpr uint32_t kScratchSrdWord3 = 0x20814fac;

QueueScratchStatus failure(QueueScratchError error, std::string message) {
  return {error, std::move(message)};
}

bool multiply_overflows(uint64_t left, uint64_t right) {
  return right != 0 && left > std::numeric_limits<uint64_t>::max() / right;
}

bool round_up_overflows(uint64_t value, uint64_t alignment) {
  return value > std::numeric_limits<uint64_t>::max() - (alignment - 1U);
}

const runtime::MemoryBank *scratch_aperture(const runtime::Node &node,
                                            size_t *match_count) {
  const runtime::MemoryBank *result = nullptr;
  *match_count = 0;
  for (const runtime::MemoryBank &bank : node.memory_banks) {
    if (bank.heap_type == runtime::MemoryHeapType::Scratch) {
      result = &bank;
      ++*match_count;
    }
  }
  return result;
}

} // namespace

const char *queue_scratch_error_name(QueueScratchError error) {
  switch (error) {
  case QueueScratchError::None:
    return "none";
  case QueueScratchError::UnsupportedArchitecture:
    return "unsupported_architecture";
  case QueueScratchError::InvalidTopology:
    return "invalid_topology";
  case QueueScratchError::MissingScratchAperture:
    return "missing_scratch_aperture";
  case QueueScratchError::InvalidScratchAperture:
    return "invalid_scratch_aperture";
  case QueueScratchError::PrivateSegmentTooLarge:
    return "private_segment_too_large";
  case QueueScratchError::SizeOverflow:
    return "size_overflow";
  case QueueScratchError::ScratchApertureTooSmall:
    return "scratch_aperture_too_small";
  case QueueScratchError::RegisterOverflow:
    return "register_overflow";
  case QueueScratchError::InvalidBackingAddress:
    return "invalid_backing_address";
  }
  return "unknown";
}

QueueScratchRequirementsResult
queue_scratch_requirements(const runtime::Node &node,
                           uint32_t private_segment_size) {
  if (node.architecture.major != 11 || node.architecture.minor != 0 ||
      node.architecture.stepping != 1) {
    return {failure(QueueScratchError::UnsupportedArchitecture,
                    "queue scratch currently supports only gfx1101"),
            {}};
  }

  const uint32_t compute_unit_count = node.compute_unit_count();
  if (!node.is_gpu() || compute_unit_count == 0 || node.wavefront_size != 32 ||
      node.maximum_waves_per_simd == 0 || node.shader_engine_count == 0 ||
      node.maximum_scratch_waves_per_compute_unit == 0 || node.xcc_count != 1) {
    return {failure(QueueScratchError::InvalidTopology,
                    "gfx1101 scratch requires complete single-XCC topology"),
            {}};
  }

  size_t scratch_aperture_count = 0;
  const runtime::MemoryBank *aperture =
      scratch_aperture(node, &scratch_aperture_count);
  if (scratch_aperture_count == 0) {
    return {failure(QueueScratchError::MissingScratchAperture,
                    "GPU topology has no scratch aperture"),
            {}};
  }
  if (scratch_aperture_count != 1 || aperture == nullptr ||
      aperture->size == 0 || aperture->virtual_base_address == 0 ||
      aperture->virtual_base_address % kScratchBackingAlignment != 0) {
    return {failure(QueueScratchError::InvalidScratchAperture,
                    "GPU topology has an invalid scratch aperture"),
            {}};
  }

  QueueScratchRequirements requirements;
  requirements.private_segment_size = private_segment_size;
  if (private_segment_size == 0) {
    return {{}, requirements};
  }
  if (private_segment_size > kMaximumPrivateSegmentSize) {
    return {failure(QueueScratchError::PrivateSegmentTooLarge,
                    "private segment exceeds the gfx1101 queue limit"),
            {}};
  }

  uint64_t raw_size = private_segment_size;
  for (const uint64_t factor :
       {uint64_t{kScratchLanesPerWave},
        uint64_t{node.maximum_scratch_waves_per_compute_unit},
        uint64_t{compute_unit_count}}) {
    if (multiply_overflows(raw_size, factor)) {
      return {failure(QueueScratchError::SizeOverflow,
                      "scratch backing size overflows uint64"),
              {}};
    }
    raw_size *= factor;
  }
  if (round_up_overflows(raw_size, kPageSize)) {
    return {failure(QueueScratchError::SizeOverflow,
                    "page-aligned scratch backing size overflows uint64"),
            {}};
  }
  requirements.allocation_size =
      (raw_size + kPageSize - 1U) & ~(kPageSize - 1U);
  requirements.backing_size_per_xcc = requirements.allocation_size;
  if (requirements.allocation_size > aperture->size) {
    return {failure(QueueScratchError::ScratchApertureTooSmall,
                    "scratch backing exceeds the topology aperture"),
            {}};
  }
  if (requirements.backing_size_per_xcc >
      std::numeric_limits<uint32_t>::max()) {
    return {failure(QueueScratchError::RegisterOverflow,
                    "scratch SRD record count exceeds 32 bits"),
            {}};
  }

  const uint64_t wave_bytes =
      uint64_t{private_segment_size} * kScratchLanesPerWave;
  const uint64_t wave_size_units =
      (wave_bytes + kScratchDescriptorAlignment - 1U) /
      kScratchDescriptorAlignment;
  if (wave_size_units == 0 || wave_size_units > kMaximumWaveSizeUnits) {
    return {failure(QueueScratchError::RegisterOverflow,
                    "scratch wave size exceeds COMPUTE_TMPRING_SIZE"),
            {}};
  }
  requirements.wave_size_units = static_cast<uint32_t>(wave_size_units);

  const uint64_t total_waves = requirements.allocation_size /
                               (wave_size_units * kScratchDescriptorAlignment);
  const uint64_t configured_waves =
      std::min(total_waves / node.shader_engine_count,
               uint64_t{compute_unit_count} *
                   node.maximum_scratch_waves_per_compute_unit);
  if (configured_waves == 0 ||
      configured_waves > kMaximumWavesPerShaderEngine) {
    return {failure(QueueScratchError::RegisterOverflow,
                    "scratch wave count exceeds COMPUTE_TMPRING_SIZE"),
            {}};
  }
  requirements.waves_per_shader_engine =
      static_cast<uint32_t>(configured_waves);
  return {{}, requirements};
}

QueueScratchControlResult
make_queue_scratch_control(const runtime::Node &node,
                           uint32_t private_segment_size,
                           uint64_t backing_gpu_address) {
  QueueScratchRequirementsResult calculated =
      queue_scratch_requirements(node, private_segment_size);
  if (!calculated) {
    return {std::move(calculated.status), {}, {}};
  }

  QueueScratchControl control;
  control.resource_descriptor[3] = kScratchSrdWord3;
  if (calculated.requirements.allocation_size == 0) {
    if (backing_gpu_address != 0) {
      return {failure(QueueScratchError::InvalidBackingAddress,
                      "scratch-free queue must not have a backing address"),
              {},
              {}};
    }
    control.resource_descriptor[1] = 1U << 30U;
    return {{}, calculated.requirements, control};
  }

  if (backing_gpu_address == 0 ||
      backing_gpu_address % kScratchBackingAlignment != 0 ||
      backing_gpu_address > kMaximumDescriptorAddress ||
      calculated.requirements.allocation_size - 1U >
          kMaximumDescriptorAddress - backing_gpu_address) {
    return {failure(QueueScratchError::InvalidBackingAddress,
                    "scratch backing must be a 64 KiB-aligned 48-bit range"),
            {},
            {}};
  }

  control.resource_descriptor[0] = static_cast<uint32_t>(backing_gpu_address);
  control.resource_descriptor[1] =
      static_cast<uint32_t>((backing_gpu_address >> 32U) & 0xffffU) |
      (1U << 30U);
  control.resource_descriptor[2] =
      static_cast<uint32_t>(calculated.requirements.backing_size_per_xcc);
  control.compute_tmpring_size =
      (calculated.requirements.wave_size_units << 12U) |
      calculated.requirements.waves_per_shader_engine;
  control.backing_memory_location = backing_gpu_address;
  control.backing_memory_byte_size =
      calculated.requirements.backing_size_per_xcc;
  control.wave64_lane_byte_size = private_segment_size;
  return {{}, calculated.requirements, control};
}

} // namespace light_rocr::arch::gfx11
