#include "light_rocr/loader/code_object.hpp"
#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/hsakmt/executable_image.hpp"
#include "light_rocr/transport/hsakmt/status.hpp"
#include "light_rocr/transport/hsakmt/topology.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

bool read_file(const std::string &path, std::vector<uint8_t> *bytes) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    std::cerr << "message=failed to open " << std::quoted(path) << '\n';
    return false;
  }
  const std::streampos end = input.tellg();
  if (end <= 0 ||
      static_cast<uintmax_t>(end) > std::numeric_limits<size_t>::max()) {
    std::cerr << "message=invalid file size for " << std::quoted(path) << '\n';
    return false;
  }
  bytes->resize(static_cast<size_t>(end));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char *>(bytes->data()),
             static_cast<std::streamsize>(bytes->size()));
  if (!input) {
    std::cerr << "message=failed to read " << std::quoted(path) << '\n';
    return false;
  }
  return true;
}

int fail_memory(const light_rocr::transport::hsakmt::MemoryStatus &status) {
  std::cerr << "memory_error="
            << light_rocr::transport::hsakmt::memory_error_name(status.error)
            << '\n';
  std::cerr << "hsakmt_status="
            << light_rocr::transport::hsakmt::hsakmt_status_name(
                   status.hsakmt_status)
            << '\n';
  std::cerr << "message=" << status.message << '\n';
  return 1;
}

int fail_image(
    const light_rocr::transport::hsakmt::ExecutableImageStatus &status) {
  std::cerr << "image_error="
            << light_rocr::transport::hsakmt::executable_image_error_name(
                   status.error)
            << '\n';
  if (status.memory_status.error !=
      light_rocr::transport::hsakmt::MemoryError::None) {
    std::cerr << "memory_error="
              << light_rocr::transport::hsakmt::memory_error_name(
                     status.memory_status.error)
              << '\n';
    std::cerr << "hsakmt_status="
              << light_rocr::transport::hsakmt::hsakmt_status_name(
                     status.memory_status.hsakmt_status)
              << '\n';
  }
  std::cerr << "message=" << status.message << '\n';
  return 1;
}

bool verify_copies(
    const std::vector<uint8_t> &bytes,
    const light_rocr::transport::hsakmt::ExecutableImage &image) {
  const auto *destination = static_cast<const uint8_t *>(image.host_address());
  const auto &plan = image.code_object().load_plan;
  for (const light_rocr::loader::LoadCopy &copy : plan.copies) {
    const uint64_t destination_offset =
        copy.virtual_address - plan.image_virtual_address;
    if (std::memcmp(destination + static_cast<size_t>(destination_offset),
                    bytes.data() + static_cast<size_t>(copy.file_offset),
                    static_cast<size_t>(copy.size)) != 0) {
      return false;
    }
  }
  return true;
}

bool verify_zero_fills(
    const light_rocr::transport::hsakmt::ExecutableImage &image) {
  const auto *destination = static_cast<const uint8_t *>(image.host_address());
  const auto &plan = image.code_object().load_plan;
  for (const light_rocr::loader::LoadZeroFill &zero_fill : plan.zero_fills) {
    const uint64_t destination_offset =
        zero_fill.virtual_address - plan.image_virtual_address;
    const uint8_t *begin =
        destination + static_cast<size_t>(destination_offset);
    if (!std::all_of(begin, begin + static_cast<size_t>(zero_fill.size),
                     [](uint8_t byte) { return byte == 0; })) {
      return false;
    }
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0] << " PATH_TO_HSACO [TARGET]\n";
    return 2;
  }
  const std::string target = argc == 3 ? argv[2] : "gfx1101";

  std::vector<uint8_t> bytes;
  if (!read_file(argv[1], &bytes)) {
    return 1;
  }
  const auto parsed =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  if (!parsed) {
    std::cerr << "parse_error="
              << light_rocr::loader::parse_error_code_name(parsed.error.code)
              << '\n';
    std::cerr << "error_offset=0x" << std::hex << parsed.error.offset
              << std::dec << '\n';
    std::cerr << "message=" << parsed.error.message << '\n';
    return 1;
  }

  const auto discovered = light_rocr::transport::hsakmt::discover_topology();
  if (!discovered) {
    std::cerr << "discovery_error="
              << light_rocr::transport::hsakmt::discovery_error_name(
                     discovered.error)
              << '\n';
    std::cerr << "message=" << discovered.message << '\n';
    return 1;
  }
  const auto selected =
      light_rocr::runtime::select_unique_gpu(discovered.topology, target);
  if (!selected) {
    std::cerr << "selection_error="
              << light_rocr::runtime::gpu_selection_error_name(selected.error)
              << '\n';
    std::cerr << "message=" << selected.message << '\n';
    return 1;
  }
  const auto &node = discovered.topology.nodes[selected.node_index];

  auto opened = light_rocr::transport::hsakmt::KfdSession::open();
  if (!opened) {
    return fail_memory(opened.status);
  }
  auto loaded = light_rocr::transport::hsakmt::materialize_executable_image(
      opened.session, node.node_id, bytes.data(), bytes.size(),
      parsed.code_object);
  if (!loaded) {
    return fail_image(loaded.status);
  }

  const bool copies_match = verify_copies(bytes, loaded.image);
  const bool zero_fills_match = verify_zero_fills(loaded.image);
  std::cout << "node_id=" << node.node_id << '\n';
  std::cout << "target="
            << light_rocr::runtime::gfx_target_name(node.architecture) << '\n';
  std::cout << "hsaco.size=" << bytes.size() << '\n';
  std::cout << "image.virtual_address=0x" << std::hex
            << loaded.image.image_virtual_address() << '\n';
  std::cout << "image.size=0x" << loaded.image.image_size() << '\n';
  std::cout << "image.allocation_size=0x" << loaded.image.allocation_size()
            << '\n';
  std::cout << "image.gpu_address=0x" << loaded.image.gpu_address() << std::dec
            << '\n';
  std::cout << "image.copy_verification=" << (copies_match ? "ok" : "failed")
            << '\n';
  std::cout << "image.zero_fill_verification="
            << (zero_fills_match ? "ok" : "failed") << '\n';
  std::cout << "kernel_count=" << loaded.image.kernels().size() << '\n';
  for (size_t index = 0; index < loaded.image.kernels().size(); ++index) {
    const auto &kernel = loaded.image.code_object().kernels[index];
    const auto &resolved = loaded.image.kernels()[index];
    std::cout << "kernel." << index << ".name=" << std::quoted(kernel.name)
              << '\n';
    std::cout << "kernel." << index << ".descriptor_virtual_address=0x"
              << std::hex << kernel.descriptor_virtual_address << '\n';
    std::cout << "kernel." << index << ".descriptor_gpu_address=0x"
              << resolved.descriptor_gpu_address << '\n';
    std::cout << "kernel." << index << ".code_entry_gpu_address=0x"
              << resolved.code_entry_gpu_address << std::dec << '\n';
  }

  const auto released = loaded.image.release();
  if (!released) {
    return fail_memory(released);
  }
  std::cout << "image.cleanup=ok\n";
  if (!copies_match || !zero_fills_match) {
    std::cerr << "message=materialized image verification failed\n";
    return 1;
  }
  return 0;
}
