#include "light_rocr/loader/code_object.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr size_t kProgramHeaderOffset = 0x40;
constexpr size_t kProgramHeaderSize = 56;
constexpr size_t kNoteOffset = 0x140;
constexpr size_t kDynamicSymbolOffset = 0x600;
constexpr size_t kDynamicStringOffset = 0x680;
constexpr size_t kHashOffset = 0x700;
constexpr size_t kDescriptorOffset = 0x800;
constexpr size_t kSecondDescriptorOffset = 0x840;
constexpr size_t kDynamicOffset = 0x900;
constexpr size_t kCodeOffset = 0x1000;
constexpr size_t kFileSize = 0x1100;

constexpr size_t kElfFlagsOffset = 48;
constexpr size_t kElfHeaderSizeOffset = 52;
constexpr size_t kElfProgramHeaderEntrySizeOffset = 54;
constexpr size_t kElfProgramHeaderCountOffset = 56;

constexpr size_t kNoteProgramHeader =
    kProgramHeaderOffset + 3 * kProgramHeaderSize;
constexpr size_t kDynamicProgramHeader =
    kProgramHeaderOffset + 2 * kProgramHeaderSize;
constexpr size_t kNoteDescriptorSizeField = kNoteOffset + 4;
constexpr size_t kNoteDescriptorOffset = kNoteOffset + 20;
constexpr size_t kKernelDescriptorKernargSize = kDescriptorOffset + 8;
constexpr size_t kKernelDescriptorCodeEntry = kDescriptorOffset + 16;
constexpr size_t kKernelSymbolValue = kDynamicSymbolOffset + 24 + 8;

void write_u16(std::vector<uint8_t> *bytes, size_t offset, uint16_t value) {
  (*bytes)[offset] = static_cast<uint8_t>(value & 0xffU);
  (*bytes)[offset + 1] = static_cast<uint8_t>((value >> 8U) & 0xffU);
}

void write_u32(std::vector<uint8_t> *bytes, size_t offset, uint32_t value) {
  for (size_t index = 0; index < 4; ++index) {
    (*bytes)[offset + index] = static_cast<uint8_t>(
        (value >> static_cast<unsigned int>(index * 8U)) & 0xffU);
  }
}

void write_u64(std::vector<uint8_t> *bytes, size_t offset, uint64_t value) {
  for (size_t index = 0; index < 8; ++index) {
    (*bytes)[offset + index] = static_cast<uint8_t>(
        (value >> static_cast<unsigned int>(index * 8U)) & 0xffU);
  }
}

void append_string(std::vector<uint8_t> *bytes, const std::string &value) {
  if (value.size() <= 31) {
    bytes->push_back(static_cast<uint8_t>(0xa0U | value.size()));
  } else {
    bytes->push_back(0xd9);
    bytes->push_back(static_cast<uint8_t>(value.size()));
  }
  bytes->insert(bytes->end(), value.begin(), value.end());
}

void append_uint(std::vector<uint8_t> *bytes, uint32_t value) {
  if (value <= 0x7f) {
    bytes->push_back(static_cast<uint8_t>(value));
  } else if (value <= 0xff) {
    bytes->push_back(0xcc);
    bytes->push_back(static_cast<uint8_t>(value));
  } else {
    bytes->push_back(0xce);
    bytes->push_back(static_cast<uint8_t>((value >> 24U) & 0xffU));
    bytes->push_back(static_cast<uint8_t>((value >> 16U) & 0xffU));
    bytes->push_back(static_cast<uint8_t>((value >> 8U) & 0xffU));
    bytes->push_back(static_cast<uint8_t>(value & 0xffU));
  }
}

void append_uint_field(std::vector<uint8_t> *bytes, const std::string &name,
                       uint32_t value) {
  append_string(bytes, name);
  append_uint(bytes, value);
}

const char *kernel_name(size_t index) {
  return index == 0 ? "test_kernel" : "second_kernel";
}

std::string kernel_symbol_name(size_t index) {
  return std::string(kernel_name(index)) + ".kd";
}

uint32_t kernel_kernarg_size(size_t index) { return index == 0 ? 24 : 32; }

uint32_t kernel_group_size(size_t index) { return index == 0 ? 4 : 0; }

uint32_t kernel_private_size(size_t index) { return index == 0 ? 8 : 16; }

std::vector<uint8_t> make_metadata(size_t kernel_count = 1) {
  std::vector<uint8_t> bytes;
  bytes.push_back(0x83); // root map(3)
  append_string(&bytes, "amdhsa.version");
  bytes.push_back(0x92);
  append_uint(&bytes, 1);
  append_uint(&bytes, 2);
  append_string(&bytes, "amdhsa.target");
  append_string(&bytes, "amdgcn-amd-amdhsa--gfx1101");
  append_string(&bytes, "amdhsa.kernels");
  bytes.push_back(static_cast<uint8_t>(0x90U | kernel_count));
  for (size_t index = 0; index < kernel_count; ++index) {
    bytes.push_back(0x88); // kernel map(8)
    append_string(&bytes, ".name");
    append_string(&bytes, kernel_name(index));
    append_string(&bytes, ".symbol");
    append_string(&bytes, kernel_symbol_name(index));
    append_uint_field(&bytes, ".kernarg_segment_size",
                      kernel_kernarg_size(index));
    append_uint_field(&bytes, ".kernarg_segment_align", 8);
    append_uint_field(&bytes, ".group_segment_fixed_size",
                      kernel_group_size(index));
    append_uint_field(&bytes, ".private_segment_fixed_size",
                      kernel_private_size(index));
    append_uint_field(&bytes, ".wavefront_size", 32);
    append_string(&bytes, ".uses_dynamic_stack");
    bytes.push_back(0xc2);
  }
  return bytes;
}

void write_program_header(std::vector<uint8_t> *bytes, size_t index,
                          uint32_t type, uint32_t flags, uint64_t file_offset,
                          uint64_t virtual_address, uint64_t file_size,
                          uint64_t memory_size, uint64_t alignment) {
  const size_t offset = kProgramHeaderOffset + index * kProgramHeaderSize;
  write_u32(bytes, offset, type);
  write_u32(bytes, offset + 4, flags);
  write_u64(bytes, offset + 8, file_offset);
  write_u64(bytes, offset + 16, virtual_address);
  write_u64(bytes, offset + 32, file_size);
  write_u64(bytes, offset + 40, memory_size);
  write_u64(bytes, offset + 48, alignment);
}

void write_dynamic_entry(std::vector<uint8_t> *bytes, size_t index,
                         uint64_t tag, uint64_t value) {
  const size_t offset = kDynamicOffset + index * 16;
  write_u64(bytes, offset, tag);
  write_u64(bytes, offset + 8, value);
}

void replace_metadata(std::vector<uint8_t> *bytes,
                      const std::vector<uint8_t> &metadata) {
  const size_t note_size = 20 + ((metadata.size() + 3) & ~size_t{3});
  std::fill(bytes->begin() + kNoteDescriptorOffset,
            bytes->begin() + kDynamicSymbolOffset, 0);
  write_u32(bytes, kNoteDescriptorSizeField,
            static_cast<uint32_t>(metadata.size()));
  write_u64(bytes, kNoteProgramHeader + 32, note_size);
  write_u64(bytes, kNoteProgramHeader + 40, note_size);
  std::copy(metadata.begin(), metadata.end(),
            bytes->begin() + kNoteDescriptorOffset);
}

std::vector<uint8_t> make_elf(size_t kernel_count = 1) {
  std::vector<uint8_t> bytes(kFileSize, 0);
  bytes[0] = 0x7f;
  bytes[1] = 'E';
  bytes[2] = 'L';
  bytes[3] = 'F';
  bytes[4] = 2;
  bytes[5] = 1;
  bytes[6] = 1;
  bytes[7] = light_rocr::loader::kAmdgpuHsaOsAbi;
  bytes[8] = 4;
  write_u16(&bytes, 16, 3);
  write_u16(&bytes, 18, 224);
  write_u32(&bytes, 20, 1);
  write_u64(&bytes, 32, kProgramHeaderOffset);
  write_u32(&bytes, kElfFlagsOffset, light_rocr::loader::kAmdgpuMachineGfx1101);
  write_u16(&bytes, kElfHeaderSizeOffset, 64);
  write_u16(&bytes, kElfProgramHeaderEntrySizeOffset, kProgramHeaderSize);
  write_u16(&bytes, kElfProgramHeaderCountOffset, 4);

  write_program_header(&bytes, 0, 1, 4, 0, 0, 0x960, 0x960, 0x1000);
  write_program_header(&bytes, 1, 1, 5, kCodeOffset, 0x2000, 0x100, 0x100,
                       0x1000);

  const std::vector<uint8_t> metadata = make_metadata(kernel_count);
  const size_t note_size = 20 + ((metadata.size() + 3) & ~size_t{3});
  write_program_header(&bytes, 2, 2, 6, kDynamicOffset, kDynamicOffset, 96, 96,
                       8);
  write_program_header(&bytes, 3, 4, 4, kNoteOffset, kNoteOffset, note_size,
                       note_size, 4);
  write_u32(&bytes, kNoteOffset, 7);
  write_u32(&bytes, kNoteDescriptorSizeField,
            static_cast<uint32_t>(metadata.size()));
  write_u32(&bytes, kNoteOffset + 8, 32);
  const std::string owner("AMDGPU\0", 7);
  std::copy(owner.begin(), owner.end(), bytes.begin() + kNoteOffset + 12);
  std::copy(metadata.begin(), metadata.end(),
            bytes.begin() + kNoteDescriptorOffset);

  std::vector<uint8_t> strings(1, 0);
  for (size_t index = 0; index < kernel_count; ++index) {
    const uint32_t name_offset = static_cast<uint32_t>(strings.size());
    const std::string symbol_name = kernel_symbol_name(index);
    strings.insert(strings.end(), symbol_name.begin(), symbol_name.end());
    strings.push_back(0);

    const size_t symbol = kDynamicSymbolOffset + (index + 1) * 24;
    write_u32(&bytes, symbol, name_offset);
    bytes[symbol + 4] = 0x11; // STB_GLOBAL | STT_OBJECT
    write_u16(&bytes, symbol + 6, 1);
    const size_t descriptor = kDescriptorOffset + index * 64;
    write_u64(&bytes, symbol + 8, descriptor);
    write_u64(&bytes, symbol + 16, 64);

    write_u32(&bytes, descriptor, kernel_group_size(index));
    write_u32(&bytes, descriptor + 4, kernel_private_size(index));
    write_u32(&bytes, descriptor + 8, kernel_kernarg_size(index));
    write_u64(&bytes, descriptor + 16, 0x2000 + index * 0x20 - descriptor);
    write_u32(&bytes, descriptor + 44, 0x20 + static_cast<uint32_t>(index));
    write_u32(&bytes, descriptor + 48, 0x1234 + static_cast<uint32_t>(index));
    write_u32(&bytes, descriptor + 52, 0x5678 + static_cast<uint32_t>(index));
    write_u16(&bytes, descriptor + 56, 1U << 10U);
  }
  std::copy(strings.begin(), strings.end(),
            bytes.begin() + kDynamicStringOffset);

  write_u32(&bytes, kHashOffset, 1);
  write_u32(&bytes, kHashOffset + 4, static_cast<uint32_t>(kernel_count + 1));

  write_dynamic_entry(&bytes, 0, 6, kDynamicSymbolOffset);
  write_dynamic_entry(&bytes, 1, 11, 24);
  write_dynamic_entry(&bytes, 2, 5, kDynamicStringOffset);
  write_dynamic_entry(&bytes, 3, 10, strings.size());
  write_dynamic_entry(&bytes, 4, 4, kHashOffset);
  write_dynamic_entry(&bytes, 5, 0, 0);
  return bytes;
}

struct TestContext {
  int failures = 0;

  void expect(bool condition, const std::string &message) {
    if (!condition) {
      ++failures;
      std::cerr << "  FAIL: " << message << '\n';
    }
  }
};

void expect_error(TestContext *context, const std::vector<uint8_t> &bytes,
                  light_rocr::loader::ParseErrorCode expected) {
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(!result, "parse unexpectedly succeeded");
  context->expect(
      result.error.code == expected,
      std::string("expected ") +
          light_rocr::loader::parse_error_code_name(expected) + ", got " +
          light_rocr::loader::parse_error_code_name(result.error.code));
}

void valid_metadata(TestContext *context) {
  const std::vector<uint8_t> bytes = make_elf();
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  context->expect(result.code_object.has_metadata, "metadata was not decoded");
  context->expect(result.code_object.metadata_version.major == 1 &&
                      result.code_object.metadata_version.minor == 2,
                  "unexpected metadata version");
  context->expect(result.code_object.target_isa == "amdgcn-amd-amdhsa--gfx1101",
                  "unexpected target ISA");
  context->expect(result.code_object.kernels.size() == 1,
                  "expected one kernel");
  if (result.code_object.kernels.size() != 1) {
    return;
  }
  const auto &kernel = result.code_object.kernels[0];
  context->expect(kernel.name == "test_kernel", "unexpected kernel name");
  context->expect(kernel.symbol_name == "test_kernel.kd",
                  "unexpected kernel symbol");
  context->expect(kernel.descriptor_virtual_address == kDescriptorOffset,
                  "unexpected descriptor address");
  context->expect(kernel.code_entry_virtual_address == 0x2000,
                  "unexpected code entry");
  context->expect(kernel.kernarg_size == 24, "unexpected kernarg size");
  context->expect(kernel.metadata_kernarg_alignment == 8,
                  "unexpected raw kernarg alignment");
  context->expect(kernel.kernarg_alignment == 16,
                  "ROCr-compatible kernarg alignment was not normalized");
  context->expect(kernel.group_segment_size == 4,
                  "unexpected group segment size");
  context->expect(kernel.private_segment_size == 8,
                  "unexpected private segment size");
  context->expect(kernel.wavefront_size == 32, "unexpected wavefront size");
  context->expect(kernel.compute_pgm_rsrc1 == 0x1234,
                  "unexpected compute_pgm_rsrc1");
  context->expect(kernel.compute_pgm_rsrc2 == 0x5678,
                  "unexpected compute_pgm_rsrc2");
  context->expect(kernel.compute_pgm_rsrc3 == 0x20,
                  "unexpected compute_pgm_rsrc3");
}

void multiple_kernels_with_reordered_symbols(TestContext *context) {
  auto bytes = make_elf(2);
  std::swap_ranges(bytes.begin() + kDynamicSymbolOffset + 24,
                   bytes.begin() + kDynamicSymbolOffset + 48,
                   bytes.begin() + kDynamicSymbolOffset + 48);
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  context->expect(result.code_object.kernels.size() == 2,
                  "expected two kernels");
  if (!result || result.code_object.kernels.size() != 2) {
    return;
  }
  context->expect(result.code_object.kernels[0].name == "test_kernel",
                  "metadata order was not preserved for the first kernel");
  context->expect(result.code_object.kernels[1].name == "second_kernel",
                  "metadata order was not preserved for the second kernel");
  context->expect(result.code_object.kernels[1].descriptor_virtual_address ==
                      kSecondDescriptorOffset,
                  "second descriptor was not resolved after symbol reorder");
  context->expect(result.code_object.kernels[1].kernarg_size == 32,
                  "unexpected second kernarg size");
  context->expect(result.code_object.kernels[1].private_segment_size == 16,
                  "unexpected second private segment size");
}

void truncated_prefixes(TestContext *context) {
  const auto bytes = make_elf();
  for (size_t prefix_size = 0; prefix_size < bytes.size(); ++prefix_size) {
    const auto result =
        light_rocr::loader::parse_code_object(bytes.data(), prefix_size);
    context->expect(!result, "truncated metadata fixture unexpectedly parsed");
  }
}

void malformed_note(TestContext *context) {
  auto bytes = make_elf();
  write_u64(&bytes, kNoteProgramHeader + 32, 16);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::MalformedNote);
}

void missing_dynamic_segment(TestContext *context) {
  auto bytes = make_elf();
  write_u32(&bytes, kDynamicProgramHeader, 6);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::MissingDynamicSegment);
}

void duplicate_dynamic_segment(TestContext *context) {
  auto bytes = make_elf();
  write_program_header(&bytes, 1, 2, 6, kDynamicOffset, kDynamicOffset, 96, 96,
                       8);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::DuplicateDynamicSegment);
}

void malformed_dynamic_table(TestContext *context) {
  auto bytes = make_elf();
  write_u64(&bytes, kDynamicProgramHeader + 32, 95);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidDynamicTable);
}

void missing_dynamic_entry(TestContext *context) {
  auto bytes = make_elf();
  write_dynamic_entry(&bytes, 4, 99, kHashOffset);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::MissingDynamicEntry);
}

void invalid_dynamic_hash(TestContext *context) {
  auto bytes = make_elf();
  write_u32(&bytes, kHashOffset, 0);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidDynamicTable);
}

void duplicate_metadata_note(TestContext *context) {
  auto bytes = make_elf();
  write_program_header(&bytes, 1, 4, 4, kNoteOffset, kNoteOffset,
                       20 + ((make_metadata().size() + 3) & ~size_t{3}),
                       20 + ((make_metadata().size() + 3) & ~size_t{3}), 4);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::DuplicateMetadataNote);
}

void unsupported_messagepack(TestContext *context) {
  auto bytes = make_elf();
  bytes[kNoteDescriptorOffset] = 0xc1;
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::UnsupportedMetadataEncoding);
}

void duplicate_messagepack_key(TestContext *context) {
  auto bytes = make_elf();
  replace_metadata(&bytes, {0x82, 0xa1, 'x', 0x00, 0xa1, 'x', 0x01});
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidMetadata);
}

void excessive_messagepack_depth(TestContext *context) {
  auto bytes = make_elf();
  std::vector<uint8_t> metadata(66, 0x91);
  metadata.push_back(0xc0);
  replace_metadata(&bytes, metadata);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidMetadata);
}

void excessive_messagepack_map_count(TestContext *context) {
  auto bytes = make_elf();
  replace_metadata(&bytes, {0xdf, 0xff, 0xff, 0xff, 0xff});
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidMetadata);
}

void truncated_messagepack(TestContext *context) {
  auto bytes = make_elf();
  const uint32_t descriptor_size =
      static_cast<uint32_t>(bytes[kNoteDescriptorSizeField]) |
      (static_cast<uint32_t>(bytes[kNoteDescriptorSizeField + 1]) << 8U);
  write_u32(&bytes, kNoteDescriptorSizeField, descriptor_size - 1);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidMetadata);
}

void target_mismatch(TestContext *context) {
  auto bytes = make_elf();
  const std::string target = "gfx1101";
  const auto found =
      std::search(bytes.begin(), bytes.end(), target.begin(), target.end());
  context->expect(found != bytes.end(), "target fixture text was not found");
  if (found != bytes.end()) {
    found[6] = '0';
  }
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::MetadataTargetMismatch);
}

void missing_symbol(TestContext *context) {
  auto bytes = make_elf();
  bytes[kDynamicStringOffset + 1] = 'X';
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::MissingKernelSymbol);
}

void descriptor_out_of_bounds(TestContext *context) {
  auto bytes = make_elf();
  write_u64(&bytes, kKernelSymbolValue, 0xa00);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::KernelDescriptorOutOfBounds);
}

void descriptor_metadata_mismatch(TestContext *context) {
  auto bytes = make_elf();
  write_u32(&bytes, kKernelDescriptorKernargSize, 32);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::KernelMetadataMismatch);
}

void invalid_descriptor_reserved_bytes(TestContext *context) {
  auto bytes = make_elf();
  bytes[kDescriptorOffset + 12] = 1;
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidKernelDescriptor);
}

void invalid_descriptor_reserved_property_bits(TestContext *context) {
  for (const unsigned int bit : {7U, 8U, 9U, 12U, 15U}) {
    auto bytes = make_elf();
    write_u16(&bytes, kDescriptorOffset + 56,
              static_cast<uint16_t>((1U << 10U) | (1U << bit)));
    expect_error(context, bytes,
                 light_rocr::loader::ParseErrorCode::InvalidKernelDescriptor);
  }
}

void dynamic_stack_flag(TestContext *context) {
  auto bytes = make_elf();
  const std::string field = ".uses_dynamic_stack";
  const auto found =
      std::search(bytes.begin(), bytes.end(), field.begin(), field.end());
  context->expect(found != bytes.end(), "dynamic-stack field was not found");
  if (found == bytes.end()) {
    return;
  }
  const size_t boolean_offset =
      static_cast<size_t>(found - bytes.begin()) + field.size();
  bytes[boolean_offset] = 0xc3;
  write_u16(&bytes, kDescriptorOffset + 56, (1U << 10U) | (1U << 11U));
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  if (result && result.code_object.kernels.size() == 1) {
    context->expect(result.code_object.kernels[0].uses_dynamic_stack,
                    "dynamic stack property was not decoded");
  }
}

void code_entry_out_of_bounds(TestContext *context) {
  auto bytes = make_elf();
  write_u64(&bytes, kKernelDescriptorCodeEntry, 0);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::KernelEntryOutOfBounds);
}

void code_entry_in_executable_bss(TestContext *context) {
  auto bytes = make_elf();
  const size_t executable_program_header =
      kProgramHeaderOffset + kProgramHeaderSize;
  write_u64(&bytes, executable_program_header + 32, 0x40);
  write_u64(&bytes, executable_program_header + 40, 0x100);
  write_u64(&bytes, kKernelDescriptorCodeEntry, 0x2040 - kDescriptorOffset);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::KernelEntryOutOfBounds);
}

void metadata_failure_is_transactional(TestContext *context) {
  auto bytes = make_elf(2);
  write_u32(&bytes, kSecondDescriptorOffset + 8, 64);
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(!result, "invalid second descriptor unexpectedly parsed");
  context->expect(!result.code_object.has_metadata,
                  "failed metadata parse was marked complete");
  context->expect(result.code_object.metadata_version.major == 0 &&
                      result.code_object.metadata_version.minor == 0,
                  "failed metadata parse exposed a partial version");
  context->expect(result.code_object.target_isa.empty(),
                  "failed metadata parse exposed a partial target");
  context->expect(result.code_object.kernels.empty(),
                  "failed metadata parse exposed partial kernels");
}

using TestFunction = std::function<void(TestContext *)>;

} // namespace

int main() {
  const std::vector<std::pair<const char *, TestFunction>> tests = {
      {"valid_metadata", valid_metadata},
      {"multiple_kernels_with_reordered_symbols",
       multiple_kernels_with_reordered_symbols},
      {"truncated_prefixes", truncated_prefixes},
      {"malformed_note", malformed_note},
      {"duplicate_metadata_note", duplicate_metadata_note},
      {"missing_dynamic_segment", missing_dynamic_segment},
      {"duplicate_dynamic_segment", duplicate_dynamic_segment},
      {"malformed_dynamic_table", malformed_dynamic_table},
      {"missing_dynamic_entry", missing_dynamic_entry},
      {"invalid_dynamic_hash", invalid_dynamic_hash},
      {"unsupported_messagepack", unsupported_messagepack},
      {"duplicate_messagepack_key", duplicate_messagepack_key},
      {"excessive_messagepack_depth", excessive_messagepack_depth},
      {"excessive_messagepack_map_count", excessive_messagepack_map_count},
      {"truncated_messagepack", truncated_messagepack},
      {"target_mismatch", target_mismatch},
      {"missing_symbol", missing_symbol},
      {"descriptor_out_of_bounds", descriptor_out_of_bounds},
      {"descriptor_metadata_mismatch", descriptor_metadata_mismatch},
      {"invalid_descriptor_reserved_bytes", invalid_descriptor_reserved_bytes},
      {"invalid_descriptor_reserved_property_bits",
       invalid_descriptor_reserved_property_bits},
      {"dynamic_stack_flag", dynamic_stack_flag},
      {"code_entry_out_of_bounds", code_entry_out_of_bounds},
      {"code_entry_in_executable_bss", code_entry_in_executable_bss},
      {"metadata_failure_is_transactional", metadata_failure_is_transactional},
  };

  int failures = 0;
  for (const auto &test : tests) {
    TestContext context;
    test.second(&context);
    if (context.failures != 0) {
      failures += context.failures;
      std::cerr << "test failed: " << test.first << '\n';
    }
  }
  if (failures != 0) {
    std::cerr << failures << " assertion(s) failed\n";
    return 1;
  }
  std::cout << tests.size() << " metadata loader tests passed\n";
  return 0;
}
