#include "light_rocr/loader/code_object.hpp"

#include <cstdint>
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
    std::cerr << "failed to open: " << path << '\n';
    return false;
  }
  const std::streampos end = input.tellg();
  if (end <= 0 ||
      static_cast<uintmax_t>(end) > std::numeric_limits<size_t>::max()) {
    std::cerr << "invalid file size: " << path << '\n';
    return false;
  }
  bytes->resize(static_cast<size_t>(end));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char *>(bytes->data()),
             static_cast<std::streamsize>(bytes->size()));
  if (!input) {
    std::cerr << "failed to read: " << path << '\n';
    return false;
  }
  return true;
}

const char *target_name(uint32_t target_machine) {
  if (target_machine == light_rocr::loader::kAmdgpuMachineGfx1101) {
    return "gfx1101";
  }
  return "unknown";
}

std::string permissions(uint32_t flags) {
  std::string result;
  result += (flags & 4U) != 0 ? 'r' : '-';
  result += (flags & 2U) != 0 ? 'w' : '-';
  result += (flags & 1U) != 0 ? 'x' : '-';
  return result;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " PATH_TO_HSACO\n";
    return 2;
  }

  std::vector<uint8_t> bytes;
  if (!read_file(argv[1], &bytes)) {
    return 1;
  }

  const light_rocr::loader::ParseResult parsed =
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

  const light_rocr::loader::CodeObject &object = parsed.code_object;
  std::cout << "format=ELF64\n";
  std::cout << "endianness=little\n";
  std::cout << "os_abi=AMDGPU_HSA\n";
  std::cout << "abi_version=" << static_cast<unsigned int>(object.abi_version)
            << '\n';
  std::cout << "machine=EM_AMDGPU\n";
  std::cout << "target=" << target_name(object.target_machine) << '\n';
  std::cout << "elf_flags=0x" << std::hex << object.elf_flags << std::dec
            << '\n';
  std::cout << "load_segment_count=" << object.load_segments.size() << '\n';
  std::cout << "metadata_present=" << (object.has_metadata ? 1 : 0) << '\n';
  std::cout << "load_plan.image_virtual_address=0x" << std::hex
            << object.load_plan.image_virtual_address << '\n';
  std::cout << "load_plan.image_size=0x" << object.load_plan.image_size << '\n';
  std::cout << "load_plan.alignment=0x" << object.load_plan.alignment
            << std::dec << '\n';
  std::cout << "load_plan.copy_count=" << object.load_plan.copies.size()
            << '\n';
  std::cout << "load_plan.zero_fill_count="
            << object.load_plan.zero_fills.size() << '\n';
  std::cout << "load_plan.protection_count="
            << object.load_plan.protections.size() << '\n';
  std::cout << "load_plan.relocation_count="
            << object.load_plan.relocations.size() << '\n';

  if (object.has_metadata) {
    std::cout << "metadata_version=" << object.metadata_version.major << '.'
              << object.metadata_version.minor << '\n';
    std::cout << "target_isa=" << std::quoted(object.target_isa) << '\n';
    std::cout << "kernel_count=" << object.kernels.size() << '\n';
  }

  for (size_t index = 0; index < object.load_segments.size(); ++index) {
    const light_rocr::loader::LoadSegment &segment =
        object.load_segments[index];
    std::cout << "segment." << index << ".file_offset=0x" << std::hex
              << segment.file_offset << '\n';
    std::cout << "segment." << index << ".file_size=0x" << segment.file_size
              << '\n';
    std::cout << "segment." << index << ".virtual_address=0x"
              << segment.virtual_address << '\n';
    std::cout << "segment." << index << ".memory_size=0x" << segment.memory_size
              << '\n';
    std::cout << "segment." << index << ".alignment=0x" << segment.alignment
              << std::dec << '\n';
    std::cout << "segment." << index
              << ".permissions=" << permissions(segment.flags) << '\n';
  }

  for (size_t index = 0; index < object.kernels.size(); ++index) {
    const light_rocr::loader::KernelInfo &kernel = object.kernels[index];
    std::cout << "kernel." << index << ".name=" << std::quoted(kernel.name)
              << '\n';
    std::cout << "kernel." << index
              << ".symbol=" << std::quoted(kernel.symbol_name) << '\n';
    std::cout << "kernel." << index << ".descriptor_virtual_address=0x"
              << std::hex << kernel.descriptor_virtual_address << '\n';
    std::cout << "kernel." << index << ".code_entry_byte_offset=" << std::dec
              << kernel.code_entry_byte_offset << '\n';
    std::cout << "kernel." << index << ".code_entry_virtual_address=0x"
              << std::hex << kernel.code_entry_virtual_address << std::dec
              << '\n';
    std::cout << "kernel." << index << ".kernarg_size=" << kernel.kernarg_size
              << '\n';
    std::cout << "kernel." << index << ".metadata_kernarg_alignment="
              << kernel.metadata_kernarg_alignment << '\n';
    std::cout << "kernel." << index
              << ".kernarg_alignment=" << kernel.kernarg_alignment << '\n';
    std::cout << "kernel." << index
              << ".group_segment_size=" << kernel.group_segment_size << '\n';
    std::cout << "kernel." << index
              << ".private_segment_size=" << kernel.private_segment_size
              << '\n';
    std::cout << "kernel." << index
              << ".wavefront_size=" << kernel.wavefront_size << '\n';
    std::cout << "kernel." << index
              << ".uses_dynamic_stack=" << (kernel.uses_dynamic_stack ? 1 : 0)
              << '\n';
    std::cout << "kernel." << index << ".compute_pgm_rsrc1=0x" << std::hex
              << kernel.compute_pgm_rsrc1 << '\n';
    std::cout << "kernel." << index << ".compute_pgm_rsrc2=0x"
              << kernel.compute_pgm_rsrc2 << '\n';
    std::cout << "kernel." << index << ".compute_pgm_rsrc3=0x"
              << kernel.compute_pgm_rsrc3 << '\n';
    std::cout << "kernel." << index << ".kernel_code_properties=0x"
              << kernel.kernel_code_properties << '\n';
    std::cout << "kernel." << index << ".kernarg_preload=0x"
              << kernel.kernarg_preload << std::dec << '\n';
  }

  for (size_t index = 0; index < object.load_plan.relocations.size(); ++index) {
    const light_rocr::loader::Relocation &relocation =
        object.load_plan.relocations[index];
    std::cout << "relocation." << index << ".encoding="
              << (relocation.encoding ==
                          light_rocr::loader::RelocationEncoding::Rela
                      ? "RELA"
                      : "REL")
              << '\n';
    std::cout << "relocation." << index << ".target_virtual_address=0x"
              << std::hex << relocation.target_virtual_address << std::dec
              << '\n';
    std::cout << "relocation." << index << ".type="
              << light_rocr::loader::amdgpu_relocation_type_name(
                     relocation.type)
              << '\n';
    std::cout << "relocation." << index
              << ".symbol_index=" << relocation.symbol_index << '\n';
    if (relocation.encoding == light_rocr::loader::RelocationEncoding::Rela) {
      std::cout << "relocation." << index << ".addend=" << relocation.addend
                << '\n';
    }
  }
  return 0;
}
