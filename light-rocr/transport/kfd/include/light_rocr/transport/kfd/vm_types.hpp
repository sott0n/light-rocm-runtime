#ifndef LIGHT_ROCR_TRANSPORT_KFD_VM_TYPES_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_VM_TYPES_HPP

#include <cstdint>
#include <string>

namespace light_rocr::transport::kfd {

struct ProcessAperture {
  uint32_t gpu_id = 0;
  // KFD reports inclusive limits for each process virtual-address range.
  uint64_t lds_base = 0;
  uint64_t lds_limit = 0;
  uint64_t scratch_base = 0;
  uint64_t scratch_limit = 0;
  uint64_t gpuvm_base = 0;
  uint64_t gpuvm_limit = 0;
};

enum class VmError {
  None,
  InvalidSession,
  InvalidNode,
  OpenRenderNode,
  AllocateState,
  AcquireVm,
  QueryApertureCount,
  InvalidApertureCount,
  AllocateApertures,
  QueryApertures,
  TopologyChanged,
  MissingAperture,
  DuplicateAperture,
  InvalidAperture,
};

struct VmStatus {
  VmError error = VmError::None;
  int system_error = 0;
  std::string message;

  explicit operator bool() const { return error == VmError::None; }
};

struct VmResult {
  VmStatus status;
  ProcessAperture aperture;

  explicit operator bool() const { return static_cast<bool>(status); }
};

} // namespace light_rocr::transport::kfd

#endif
