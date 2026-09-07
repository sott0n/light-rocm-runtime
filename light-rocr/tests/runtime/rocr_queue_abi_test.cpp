#include "light_rocr/runtime/amd_queue.hpp"

#include <hsa/amd_hsa_queue.h>

#include <cstddef>
#include <iostream>

int main() {
  using light_rocr::runtime::AmdQueueV1;
  using light_rocr::runtime::HsaQueueAbi;

  bool matches = true;
  matches = matches && sizeof(HsaQueueAbi) == sizeof(hsa_queue_t);
  matches = matches && offsetof(HsaQueueAbi, base_address) ==
                           offsetof(hsa_queue_t, base_address);
  matches = matches && offsetof(HsaQueueAbi, doorbell_signal) ==
                           offsetof(hsa_queue_t, doorbell_signal);
  matches = matches && offsetof(HsaQueueAbi, id) == offsetof(hsa_queue_t, id);
  matches = matches && sizeof(AmdQueueV1) == sizeof(amd_queue_t);
  matches = matches && alignof(AmdQueueV1) == alignof(amd_queue_t);
  matches =
      matches && offsetof(AmdQueueV1, caps) == offsetof(amd_queue_t, caps);
  matches = matches && offsetof(AmdQueueV1, write_dispatch_id) ==
                           offsetof(amd_queue_t, write_dispatch_id);
  matches =
      matches && offsetof(AmdQueueV1, group_segment_aperture_base_hi) ==
                     offsetof(amd_queue_t, group_segment_aperture_base_hi);
  matches =
      matches && offsetof(AmdQueueV1, private_segment_aperture_base_hi) ==
                     offsetof(amd_queue_t, private_segment_aperture_base_hi);
  matches = matches && offsetof(AmdQueueV1, read_dispatch_id) ==
                           offsetof(amd_queue_t, read_dispatch_id);
  matches = matches &&
            offsetof(AmdQueueV1, max_cu_id) == offsetof(amd_queue_t, max_cu_id);
  matches = matches && offsetof(AmdQueueV1, max_wave_id) ==
                           offsetof(amd_queue_t, max_wave_id);
  matches = matches &&
            offsetof(AmdQueueV1, read_dispatch_id_field_base_byte_offset) ==
                offsetof(amd_queue_t, read_dispatch_id_field_base_byte_offset);
  matches = matches && offsetof(AmdQueueV1, compute_tmpring_size) ==
                           offsetof(amd_queue_t, compute_tmpring_size);
  matches = matches && offsetof(AmdQueueV1, scratch_resource_descriptor) ==
                           offsetof(amd_queue_t, scratch_resource_descriptor);
  matches =
      matches && offsetof(AmdQueueV1, scratch_backing_memory_location) ==
                     offsetof(amd_queue_t, scratch_backing_memory_location);
  matches =
      matches && offsetof(AmdQueueV1, scratch_backing_memory_byte_size) ==
                     offsetof(amd_queue_t, scratch_backing_memory_byte_size);
  matches = matches && offsetof(AmdQueueV1, scratch_wave64_lane_byte_size) ==
                           offsetof(amd_queue_t, scratch_wave64_lane_byte_size);
  matches = matches && offsetof(AmdQueueV1, queue_properties) ==
                           offsetof(amd_queue_t, queue_properties);
  matches = matches && offsetof(AmdQueueV1, queue_inactive_signal) ==
                           offsetof(amd_queue_t, queue_inactive_signal);
  matches = matches &&
            light_rocr::runtime::kHsaQueueTypeMulti == HSA_QUEUE_TYPE_MULTI;
  matches = matches && light_rocr::runtime::kHsaQueueFeatureKernelDispatch ==
                           HSA_QUEUE_FEATURE_KERNEL_DISPATCH;
  matches = matches && light_rocr::runtime::kAmdQueuePropertyIsPointer64 ==
                           AMD_QUEUE_PROPERTIES_IS_PTR64;
  if (!matches) {
    std::cerr << "self-authored AMD queue ABI does not match installed ROCr "
                 "headers\n";
    return 1;
  }
  std::cout << "self-authored AMD queue ABI matches installed ROCr headers\n";
  return 0;
}
