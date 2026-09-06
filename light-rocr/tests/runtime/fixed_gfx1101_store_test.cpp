#include "fixed_gfx1101_store.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> read_file(const char *path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return {};
  }
  return {std::istreambuf_iterator<char>(stream),
          std::istreambuf_iterator<char>()};
}

} // namespace

int main(int argc, char **argv) {
  namespace fixed = light_rocr::tools::fixed_gfx1101_store;
  if (argc != 3) {
    std::cerr << "expected linked descriptor and text section paths\n";
    return 1;
  }

  const std::vector<uint8_t> descriptor_bytes = read_file(argv[1]);
  const std::vector<uint8_t> code_bytes = read_file(argv[2]);
  if (descriptor_bytes.size() != sizeof(fixed::KernelDescriptor) ||
      code_bytes.size() != sizeof(fixed::kCode)) {
    std::cerr << "assembled fixed-kernel section size does not match the "
                 "embedded image\n";
    return 1;
  }

  const fixed::KernelDescriptor descriptor;
  if (std::memcmp(descriptor_bytes.data(), &descriptor, sizeof(descriptor)) !=
      0) {
    std::cerr << "assembled fixed-kernel descriptor does not match the "
                 "embedded descriptor\n";
    return 1;
  }
  if (std::memcmp(code_bytes.data(), fixed::kCode.data(),
                  sizeof(fixed::kCode)) != 0) {
    std::cerr << "assembled fixed-kernel instructions do not match the "
                 "embedded instructions\n";
    return 1;
  }

  std::cout << "assembled fixed gfx1101 image matches embedded bytes\n";
  return 0;
}
