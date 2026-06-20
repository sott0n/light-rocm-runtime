#ifndef LRRT_TESTS_COMMON_PRECISION_UTILS_H_
#define LRRT_TESTS_COMMON_PRECISION_UTILS_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <stdexcept>
#include <vector>

namespace lrrt_test {

enum class DataType { Fp32, Fp16, Bf16 };

inline const char *data_type_name(DataType data_type) {
  switch (data_type) {
  case DataType::Fp32:
    return "fp32";
  case DataType::Fp16:
    return "fp16";
  case DataType::Bf16:
    return "bf16";
  }
  throw std::invalid_argument("unsupported data type");
}

inline size_t data_type_size(DataType data_type) {
  return data_type == DataType::Fp32 ? sizeof(float) : sizeof(uint16_t);
}

inline void write_value(std::vector<unsigned char> &storage, size_t index,
                        DataType data_type, float value) {
  unsigned char *destination =
      storage.data() + index * data_type_size(data_type);
  if (data_type == DataType::Fp32) {
    memcpy(destination, &value, sizeof(value));
  } else if (data_type == DataType::Fp16) {
    _Float16 half = (_Float16)value;
    memcpy(destination, &half, sizeof(half));
  } else {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    uint16_t bfloat16 = (uint16_t)(bits >> 16);
    memcpy(destination, &bfloat16, sizeof(bfloat16));
  }
}

inline float read_value(const std::vector<unsigned char> &storage, size_t index,
                        DataType data_type) {
  const unsigned char *source =
      storage.data() + index * data_type_size(data_type);
  if (data_type == DataType::Fp32) {
    float value = 0.0f;
    memcpy(&value, source, sizeof(value));
    return value;
  }
  if (data_type == DataType::Fp16) {
    _Float16 half = 0.0f;
    memcpy(&half, source, sizeof(half));
    return (float)half;
  }
  uint16_t bfloat16 = 0;
  memcpy(&bfloat16, source, sizeof(bfloat16));
  uint32_t bits = (uint32_t)bfloat16 << 16;
  float value = 0.0f;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

inline float round_value(float value, DataType data_type) {
  if (data_type == DataType::Fp32) {
    return value;
  }
  if (data_type == DataType::Fp16) {
    return (float)(_Float16)value;
  }
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  bits &= 0xffff0000u;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

} // namespace lrrt_test

#endif // LRRT_TESTS_COMMON_PRECISION_UTILS_H_
