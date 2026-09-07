#ifndef LIGHT_ROCR_RUNTIME_AMD_QUEUE_HPP
#define LIGHT_ROCR_RUNTIME_AMD_QUEUE_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace light_rocr::runtime {

inline constexpr uint32_t kHsaQueueTypeMulti = 0;
inline constexpr uint32_t kHsaQueueFeatureKernelDispatch = 1;
inline constexpr uint32_t kAmdQueuePropertyIsPointer64 = 1U << 1U;

// The public HSA queue prefix is reproduced here so the native runtime does
// not need a build-time dependency on ROCr. This project currently targets
// the 64-bit Linux ABI only.
struct HsaQueueAbi {
  uint32_t type = 0;
  uint32_t features = 0;
  uint64_t base_address = 0;
  uint64_t doorbell_signal = 0;
  uint32_t size = 0;
  uint32_t reserved = 0;
  uint64_t id = 0;
};

// Version-one AMD queue control block. The command processor may read this
// prefix through the queue read-pointer address, including its scratch SRD.
// Keeping caps zero tells firmware that no version-two async-reclaim fields
// are present.
struct alignas(64) AmdQueueV1 {
  HsaQueueAbi hsa_queue;
  uint32_t caps = 0;
  uint32_t reserved1[3]{};
  uint64_t write_dispatch_id = 0;
  uint32_t group_segment_aperture_base_hi = 0;
  uint32_t private_segment_aperture_base_hi = 0;
  uint32_t max_cu_id = 0;
  uint32_t max_wave_id = 0;
  uint64_t max_legacy_doorbell_dispatch_id_plus_1 = 0;
  uint32_t legacy_doorbell_lock = 0;
  uint32_t reserved2[9]{};
  uint64_t read_dispatch_id = 0;
  uint32_t read_dispatch_id_field_base_byte_offset = 0;
  uint32_t compute_tmpring_size = 0;
  uint32_t scratch_resource_descriptor[4]{};
  uint64_t scratch_backing_memory_location = 0;
  uint64_t scratch_backing_memory_byte_size = 0;
  uint32_t scratch_wave64_lane_byte_size = 0;
  uint32_t queue_properties = 0;
  uint32_t reserved3[2]{};
  uint64_t queue_inactive_signal = 0;
  uint32_t reserved4[14]{};
};

static_assert(sizeof(void *) == sizeof(uint64_t));
static_assert(std::is_standard_layout_v<HsaQueueAbi>);
static_assert(std::is_trivially_copyable_v<HsaQueueAbi>);
static_assert(sizeof(HsaQueueAbi) == 40);
static_assert(std::is_standard_layout_v<AmdQueueV1>);
static_assert(std::is_trivially_copyable_v<AmdQueueV1>);
static_assert(alignof(AmdQueueV1) == 64);
static_assert(sizeof(AmdQueueV1) == 256);
static_assert(offsetof(AmdQueueV1, caps) == 40);
static_assert(offsetof(AmdQueueV1, write_dispatch_id) == 56);
static_assert(offsetof(AmdQueueV1, read_dispatch_id) == 128);
static_assert(offsetof(AmdQueueV1, compute_tmpring_size) == 140);
static_assert(offsetof(AmdQueueV1, scratch_resource_descriptor) == 144);
static_assert(offsetof(AmdQueueV1, scratch_backing_memory_location) == 160);
static_assert(offsetof(AmdQueueV1, scratch_backing_memory_byte_size) == 168);
static_assert(offsetof(AmdQueueV1, scratch_wave64_lane_byte_size) == 176);
static_assert(offsetof(AmdQueueV1, queue_properties) == 180);

} // namespace light_rocr::runtime

#endif
