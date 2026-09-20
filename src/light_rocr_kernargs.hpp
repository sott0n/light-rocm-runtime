#ifndef LRRT_LIGHT_ROCR_KERNARGS_HPP
#define LRRT_LIGHT_ROCR_KERNARGS_HPP

#include "light_rocr/loader/code_object.hpp"
#include "lrrt/lrrt.h"

#include <cstddef>

namespace lrrt_internal {

[[nodiscard]] bool populate_light_rocr_hidden_kernargs(
    const light_rocr::loader::KernelInfo &kernel,
    const lr_launch_config_t &config, void *kernarg, size_t kernarg_size);

} // namespace lrrt_internal

#endif
