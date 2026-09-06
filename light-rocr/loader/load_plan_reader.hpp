#ifndef LIGHT_ROCR_LOADER_LOAD_PLAN_READER_HPP_
#define LIGHT_ROCR_LOADER_LOAD_PLAN_READER_HPP_

#include "light_rocr/loader/code_object.hpp"

#include <cstddef>
#include <cstdint>

namespace light_rocr::loader::internal {

ParseError build_load_plan(const uint8_t *data, size_t size,
                           CodeObject *code_object);

} // namespace light_rocr::loader::internal

#endif // LIGHT_ROCR_LOADER_LOAD_PLAN_READER_HPP_
