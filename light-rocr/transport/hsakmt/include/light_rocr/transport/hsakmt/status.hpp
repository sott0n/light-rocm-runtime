#ifndef LIGHT_ROCR_TRANSPORT_HSAKMT_STATUS_HPP
#define LIGHT_ROCR_TRANSPORT_HSAKMT_STATUS_HPP

#include <cstdint>

namespace light_rocr::transport::hsakmt {

[[nodiscard]] const char *hsakmt_status_name(uint32_t status);

} // namespace light_rocr::transport::hsakmt

#endif
