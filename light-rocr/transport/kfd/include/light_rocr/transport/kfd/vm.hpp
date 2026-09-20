#ifndef LIGHT_ROCR_TRANSPORT_KFD_VM_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_VM_HPP

#include "light_rocr/transport/kfd/session.hpp"
#include "light_rocr/transport/kfd/vm_types.hpp"

namespace light_rocr::transport::kfd {

[[nodiscard]] const char *vm_error_name(VmError error);

} // namespace light_rocr::transport::kfd

#endif
