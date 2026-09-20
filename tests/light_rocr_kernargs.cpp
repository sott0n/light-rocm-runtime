#include "light_rocr_kernargs.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

template <typename Value>
Value read_value(const std::array<unsigned char, 96> &bytes, size_t offset) {
  Value value{};
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

} // namespace

int main() {
  using light_rocr::loader::KernelArgumentKind;

  light_rocr::loader::KernelInfo kernel;
  kernel.kernarg_size = 96;
  kernel.arguments = {
      {0, 8, KernelArgumentKind::GlobalBuffer},
      {8, 4, KernelArgumentKind::HiddenBlockCountX},
      {12, 4, KernelArgumentKind::HiddenBlockCountY},
      {16, 4, KernelArgumentKind::HiddenBlockCountZ},
      {20, 2, KernelArgumentKind::HiddenGroupSizeX},
      {22, 2, KernelArgumentKind::HiddenGroupSizeY},
      {24, 2, KernelArgumentKind::HiddenGroupSizeZ},
      {26, 2, KernelArgumentKind::HiddenRemainderX},
      {28, 2, KernelArgumentKind::HiddenRemainderY},
      {30, 2, KernelArgumentKind::HiddenRemainderZ},
      {32, 2, KernelArgumentKind::HiddenGridDims},
      {36, 4, KernelArgumentKind::HiddenDynamicLdsSize},
      {40, 8, KernelArgumentKind::HiddenGlobalOffsetX},
  };
  std::array<unsigned char, 96> bytes{};
  bytes[0] = 0xa5;
  const lr_launch_config_t config = {{130, 15, 4}, {64, 4, 2}, 1234};
  if (!lrrt_internal::populate_light_rocr_hidden_kernargs(
          kernel, config, bytes.data(), bytes.size()) ||
      bytes[0] != 0xa5 || read_value<uint32_t>(bytes, 8) != 2 ||
      read_value<uint32_t>(bytes, 12) != 3 ||
      read_value<uint32_t>(bytes, 16) != 2 ||
      read_value<uint16_t>(bytes, 20) != 64 ||
      read_value<uint16_t>(bytes, 22) != 4 ||
      read_value<uint16_t>(bytes, 24) != 2 ||
      read_value<uint16_t>(bytes, 26) != 2 ||
      read_value<uint16_t>(bytes, 28) != 3 ||
      read_value<uint16_t>(bytes, 30) != 0 ||
      read_value<uint16_t>(bytes, 32) != 3 ||
      read_value<uint32_t>(bytes, 36) != 1234 ||
      read_value<uint64_t>(bytes, 40) != 0) {
    std::cerr << "hidden kernarg launch values were not populated correctly\n";
    return 1;
  }

  if (lrrt_internal::populate_light_rocr_hidden_kernargs(
          kernel, config, bytes.data(), bytes.size() - 1)) {
    std::cerr << "undersized kernarg buffer was accepted\n";
    return 1;
  }
  kernel.arguments[1].size = 2;
  if (lrrt_internal::populate_light_rocr_hidden_kernargs(
          kernel, config, bytes.data(), bytes.size())) {
    std::cerr << "invalid hidden argument size was accepted\n";
    return 1;
  }

  std::cout << "light-rocr hidden kernargs: ok\n";
  return 0;
}
