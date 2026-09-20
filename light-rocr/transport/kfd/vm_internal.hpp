#ifndef LIGHT_ROCR_TRANSPORT_KFD_VM_INTERNAL_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_VM_INTERNAL_HPP

#include "light_rocr/transport/kfd/vm_types.hpp"

namespace light_rocr::transport::kfd::detail {

using IoctlFunction = int (*)(int fd, unsigned long request, void *arguments);

[[nodiscard]] VmResult query_process_aperture(int kfd_fd, uint32_t gpu_id,
                                              IoctlFunction ioctl_function);

} // namespace light_rocr::transport::kfd::detail

#endif
