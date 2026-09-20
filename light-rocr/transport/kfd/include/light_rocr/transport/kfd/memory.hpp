#ifndef LIGHT_ROCR_TRANSPORT_KFD_MEMORY_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_MEMORY_HPP

#include "light_rocr/transport/kfd/memory_types.hpp"
#include "light_rocr/transport/kfd/session.hpp"

namespace light_rocr::transport::kfd {

[[nodiscard]] const char *memory_error_name(MemoryError error);

} // namespace light_rocr::transport::kfd

#endif
