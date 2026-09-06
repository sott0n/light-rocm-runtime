#include "light_rocr/runtime/aql.hpp"

#include <hsa/hsa.h>

#include <cstddef>
#include <iostream>

int main() {
  using light_rocr::runtime::AqlKernelDispatchPacket;
  bool matches = true;
  matches = matches && sizeof(AqlKernelDispatchPacket) ==
                           sizeof(hsa_kernel_dispatch_packet_t);
  matches = matches && offsetof(AqlKernelDispatchPacket, setup) ==
                           offsetof(hsa_kernel_dispatch_packet_t, setup);
  matches =
      matches && offsetof(AqlKernelDispatchPacket, kernel_object) ==
                     offsetof(hsa_kernel_dispatch_packet_t, kernel_object);
  matches =
      matches && offsetof(AqlKernelDispatchPacket, kernarg_address) ==
                     offsetof(hsa_kernel_dispatch_packet_t, kernarg_address);
  matches =
      matches && offsetof(AqlKernelDispatchPacket, completion_signal) ==
                     offsetof(hsa_kernel_dispatch_packet_t, completion_signal);
  matches = matches && light_rocr::runtime::kAqlPacketTypeInvalid ==
                           HSA_PACKET_TYPE_INVALID;
  matches = matches && light_rocr::runtime::kAqlPacketTypeKernelDispatch ==
                           HSA_PACKET_TYPE_KERNEL_DISPATCH;
  matches = matches &&
            light_rocr::runtime::kAqlFenceScopeSystem == HSA_FENCE_SCOPE_SYSTEM;
  if (!matches) {
    std::cerr
        << "self-authored AQL ABI does not match installed ROCr headers\n";
    return 1;
  }
  std::cout << "self-authored AQL ABI matches installed ROCr headers\n";
  return 0;
}
