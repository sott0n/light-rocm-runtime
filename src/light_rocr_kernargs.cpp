#include "light_rocr_kernargs.hpp"

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace lrrt_internal {
namespace {

uint16_t dispatch_dimensions(const lr_launch_config_t &config) {
  if (config.grid.z > 1 || config.block.z > 1) {
    return 3;
  }
  if (config.grid.y > 1 || config.block.y > 1) {
    return 2;
  }
  return 1;
}

template <typename Value>
bool write_argument(unsigned char *bytes, size_t buffer_size,
                    const light_rocr::loader::KernelArgumentInfo &argument,
                    Value value) {
  static_assert(std::is_trivially_copyable_v<Value>);
  if (argument.size != sizeof(value) || argument.offset > buffer_size ||
      argument.size > buffer_size - argument.offset) {
    return false;
  }
  std::memcpy(bytes + argument.offset, &value, sizeof(value));
  return true;
}

} // namespace

bool populate_light_rocr_hidden_kernargs(
    const light_rocr::loader::KernelInfo &kernel,
    const lr_launch_config_t &config, void *kernarg, size_t kernarg_size) {
  if (kernarg_size < kernel.kernarg_size ||
      (kernel.kernarg_size != 0 && kernarg == nullptr) || config.grid.x == 0 ||
      config.grid.y == 0 || config.grid.z == 0 || config.block.x == 0 ||
      config.block.y == 0 || config.block.z == 0 ||
      config.block.x > UINT16_MAX || config.block.y > UINT16_MAX ||
      config.block.z > UINT16_MAX || config.grid.x < config.block.x ||
      config.grid.y < config.block.y || config.grid.z < config.block.z) {
    return false;
  }

  auto *bytes = static_cast<unsigned char *>(kernarg);
  const uint32_t grid[] = {config.grid.x, config.grid.y, config.grid.z};
  const uint32_t block[] = {config.block.x, config.block.y, config.block.z};
  using light_rocr::loader::KernelArgumentKind;
  for (const auto &argument : kernel.arguments) {
    bool written = true;
    switch (argument.kind) {
    case KernelArgumentKind::HiddenBlockCountX:
      written =
          write_argument(bytes, kernarg_size, argument, grid[0] / block[0]);
      break;
    case KernelArgumentKind::HiddenBlockCountY:
      written =
          write_argument(bytes, kernarg_size, argument, grid[1] / block[1]);
      break;
    case KernelArgumentKind::HiddenBlockCountZ:
      written =
          write_argument(bytes, kernarg_size, argument, grid[2] / block[2]);
      break;
    case KernelArgumentKind::HiddenGroupSizeX:
      written = write_argument(bytes, kernarg_size, argument,
                               static_cast<uint16_t>(block[0]));
      break;
    case KernelArgumentKind::HiddenGroupSizeY:
      written = write_argument(bytes, kernarg_size, argument,
                               static_cast<uint16_t>(block[1]));
      break;
    case KernelArgumentKind::HiddenGroupSizeZ:
      written = write_argument(bytes, kernarg_size, argument,
                               static_cast<uint16_t>(block[2]));
      break;
    case KernelArgumentKind::HiddenRemainderX:
      written = write_argument(bytes, kernarg_size, argument,
                               static_cast<uint16_t>(grid[0] % block[0]));
      break;
    case KernelArgumentKind::HiddenRemainderY:
      written = write_argument(bytes, kernarg_size, argument,
                               static_cast<uint16_t>(grid[1] % block[1]));
      break;
    case KernelArgumentKind::HiddenRemainderZ:
      written = write_argument(bytes, kernarg_size, argument,
                               static_cast<uint16_t>(grid[2] % block[2]));
      break;
    case KernelArgumentKind::HiddenGridDims:
      written = write_argument(bytes, kernarg_size, argument,
                               dispatch_dimensions(config));
      break;
    case KernelArgumentKind::HiddenDynamicLdsSize:
      written = write_argument(bytes, kernarg_size, argument,
                               config.shared_memory_bytes);
      break;
    default:
      break;
    }
    if (!written) {
      return false;
    }
  }
  return true;
}

} // namespace lrrt_internal
