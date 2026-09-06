#ifndef LIGHT_ROCR_LOADER_METADATA_READER_HPP_
#define LIGHT_ROCR_LOADER_METADATA_READER_HPP_

#include "light_rocr/loader/code_object.hpp"

#include <cstddef>
#include <cstdint>

namespace light_rocr::loader::internal {

ParseError decode_amdhsa_metadata(const uint8_t *data, size_t size,
                                  CodeObject *code_object);

} // namespace light_rocr::loader::internal

#endif // LIGHT_ROCR_LOADER_METADATA_READER_HPP_
