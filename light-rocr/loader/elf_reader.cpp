#include "light_rocr/loader/code_object.hpp"

#include "metadata_reader.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>
#include <vector>

namespace light_rocr::loader {
namespace {

constexpr size_t kElf64HeaderSize = 64;
constexpr size_t kElf64ProgramHeaderSize = 56;

constexpr size_t kEiClass = 4;
constexpr size_t kEiData = 5;
constexpr size_t kEiVersion = 6;
constexpr size_t kEiOsAbi = 7;
constexpr size_t kEiAbiVersion = 8;

constexpr uint8_t kElfClass64 = 2;
constexpr uint8_t kElfDataLittleEndian = 1;
constexpr uint8_t kElfCurrentVersion = 1;
constexpr uint16_t kElfTypeDynamic = 3;
constexpr uint16_t kElfMachineAmdgpu = 224;
constexpr uint16_t kExtendedProgramHeaderCount = 0xffff;

constexpr uint32_t kProgramHeaderTypeLoad = 1;
constexpr uint32_t kSegmentFlagExecute = 1;
constexpr uint32_t kSegmentFlagWrite = 2;
constexpr uint32_t kSegmentFlagRead = 4;
constexpr uint32_t kKnownSegmentFlags =
    kSegmentFlagExecute | kSegmentFlagWrite | kSegmentFlagRead;

constexpr size_t kElfTypeOffset = 16;
constexpr size_t kElfMachineOffset = 18;
constexpr size_t kElfVersionOffset = 20;
constexpr size_t kElfProgramHeaderOffset = 32;
constexpr size_t kElfFlagsOffset = 48;
constexpr size_t kElfHeaderSizeOffset = 52;
constexpr size_t kElfProgramHeaderEntrySizeOffset = 54;
constexpr size_t kElfProgramHeaderCountOffset = 56;

constexpr size_t kProgramHeaderTypeOffset = 0;
constexpr size_t kProgramHeaderFlagsOffset = 4;
constexpr size_t kProgramHeaderFileOffset = 8;
constexpr size_t kProgramHeaderVirtualAddressOffset = 16;
constexpr size_t kProgramHeaderFileSizeOffset = 32;
constexpr size_t kProgramHeaderMemorySizeOffset = 40;
constexpr size_t kProgramHeaderAlignmentOffset = 48;

ParseResult failure(ParseErrorCode code, uint64_t offset, std::string message) {
  ParseResult result;
  result.error = ParseError{code, offset, std::move(message)};
  return result;
}

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t *result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

bool range_is_in_file(uint64_t offset, uint64_t length, size_t file_size) {
  const uint64_t size = static_cast<uint64_t>(file_size);
  return offset <= size && length <= size - offset;
}

uint16_t read_u16(const uint8_t *data) {
  return static_cast<uint16_t>(data[0]) |
         static_cast<uint16_t>(static_cast<uint16_t>(data[1]) << 8U);
}

uint32_t read_u32(const uint8_t *data) {
  uint32_t value = 0;
  for (size_t byte = 0; byte < 4; ++byte) {
    value |= static_cast<uint32_t>(data[byte])
             << static_cast<unsigned int>(byte * 8U);
  }
  return value;
}

uint64_t read_u64(const uint8_t *data) {
  uint64_t value = 0;
  for (size_t byte = 0; byte < 8; ++byte) {
    value |= static_cast<uint64_t>(data[byte])
             << static_cast<unsigned int>(byte * 8U);
  }
  return value;
}

bool is_power_of_two(uint64_t value) {
  return value != 0 && (value & (value - 1U)) == 0;
}

ParseResult validate_load_segment(const LoadSegment &segment, size_t file_size,
                                  uint64_t program_header_offset) {
  if ((segment.flags & ~kKnownSegmentFlags) != 0) {
    return failure(ParseErrorCode::InvalidSegmentFlags,
                   program_header_offset + kProgramHeaderFlagsOffset,
                   "PT_LOAD has unknown permission flags");
  }
  if (segment.file_size > segment.memory_size) {
    return failure(ParseErrorCode::SegmentFileSizeExceedsMemorySize,
                   program_header_offset + kProgramHeaderFileSizeOffset,
                   "PT_LOAD file size exceeds memory size");
  }
  if (!range_is_in_file(segment.file_offset, segment.file_size, file_size)) {
    return failure(ParseErrorCode::SegmentFileRangeOutOfBounds,
                   program_header_offset + kProgramHeaderFileOffset,
                   "PT_LOAD file range is outside the code object");
  }

  uint64_t virtual_end = 0;
  if (!checked_add(segment.virtual_address, segment.memory_size,
                   &virtual_end)) {
    return failure(ParseErrorCode::SegmentVirtualRangeOverflow,
                   program_header_offset + kProgramHeaderVirtualAddressOffset,
                   "PT_LOAD virtual-address range overflows uint64");
  }

  if (segment.alignment > 1 && !is_power_of_two(segment.alignment)) {
    return failure(ParseErrorCode::InvalidSegmentAlignment,
                   program_header_offset + kProgramHeaderAlignmentOffset,
                   "PT_LOAD alignment is not zero, one, or a power of two");
  }
  if (segment.alignment > 1 &&
      segment.file_offset % segment.alignment !=
          segment.virtual_address % segment.alignment) {
    return failure(ParseErrorCode::IncongruentSegmentAlignment,
                   program_header_offset + kProgramHeaderAlignmentOffset,
                   "PT_LOAD file offset and virtual address are not congruent "
                   "modulo alignment");
  }
  return ParseResult{};
}

struct ParsedLoadSegment {
  LoadSegment segment;
  uint64_t program_header_offset = 0;
};

template <typename Offset, typename Size>
const ParsedLoadSegment *
find_overlapping_segment(const std::vector<ParsedLoadSegment> &segments,
                         Offset offset, Size range_size) {
  std::vector<const ParsedLoadSegment *> ordered;
  ordered.reserve(segments.size());
  for (const ParsedLoadSegment &segment : segments) {
    if (range_size(segment.segment) != 0) {
      ordered.push_back(&segment);
    }
  }

  std::sort(
      ordered.begin(), ordered.end(),
      [offset](const ParsedLoadSegment *lhs, const ParsedLoadSegment *rhs) {
        const uint64_t lhs_offset = offset(lhs->segment);
        const uint64_t rhs_offset = offset(rhs->segment);
        if (lhs_offset != rhs_offset) {
          return lhs_offset < rhs_offset;
        }
        return lhs->program_header_offset < rhs->program_header_offset;
      });

  for (size_t index = 1; index < ordered.size(); ++index) {
    const ParsedLoadSegment &previous = *ordered[index - 1];
    const ParsedLoadSegment &current = *ordered[index];
    const uint64_t previous_end =
        offset(previous.segment) + range_size(previous.segment);
    if (offset(current.segment) < previous_end) {
      return &current;
    }
  }
  return nullptr;
}

} // namespace

const char *parse_error_code_name(ParseErrorCode code) {
  switch (code) {
  case ParseErrorCode::None:
    return "none";
  case ParseErrorCode::InvalidArgument:
    return "invalid_argument";
  case ParseErrorCode::TruncatedElfHeader:
    return "truncated_elf_header";
  case ParseErrorCode::InvalidElfMagic:
    return "invalid_elf_magic";
  case ParseErrorCode::UnsupportedElfClass:
    return "unsupported_elf_class";
  case ParseErrorCode::UnsupportedEndianness:
    return "unsupported_endianness";
  case ParseErrorCode::UnsupportedElfVersion:
    return "unsupported_elf_version";
  case ParseErrorCode::UnsupportedOsAbi:
    return "unsupported_os_abi";
  case ParseErrorCode::UnsupportedAbiVersion:
    return "unsupported_abi_version";
  case ParseErrorCode::UnsupportedObjectType:
    return "unsupported_object_type";
  case ParseErrorCode::UnsupportedMachine:
    return "unsupported_machine";
  case ParseErrorCode::UnsupportedTarget:
    return "unsupported_target";
  case ParseErrorCode::InvalidElfHeaderSize:
    return "invalid_elf_header_size";
  case ParseErrorCode::MissingProgramHeaders:
    return "missing_program_headers";
  case ParseErrorCode::UnsupportedExtendedProgramHeaderCount:
    return "unsupported_extended_program_header_count";
  case ParseErrorCode::InvalidProgramHeaderEntrySize:
    return "invalid_program_header_entry_size";
  case ParseErrorCode::ProgramHeaderTableOutOfBounds:
    return "program_header_table_out_of_bounds";
  case ParseErrorCode::InvalidSegmentFlags:
    return "invalid_segment_flags";
  case ParseErrorCode::SegmentFileSizeExceedsMemorySize:
    return "segment_file_size_exceeds_memory_size";
  case ParseErrorCode::SegmentFileRangeOutOfBounds:
    return "segment_file_range_out_of_bounds";
  case ParseErrorCode::SegmentVirtualRangeOverflow:
    return "segment_virtual_range_overflow";
  case ParseErrorCode::InvalidSegmentAlignment:
    return "invalid_segment_alignment";
  case ParseErrorCode::IncongruentSegmentAlignment:
    return "incongruent_segment_alignment";
  case ParseErrorCode::OverlappingSegmentFileRanges:
    return "overlapping_segment_file_ranges";
  case ParseErrorCode::OverlappingSegmentVirtualRanges:
    return "overlapping_segment_virtual_ranges";
  case ParseErrorCode::MissingLoadSegments:
    return "missing_load_segments";
  case ParseErrorCode::MalformedNote:
    return "malformed_note";
  case ParseErrorCode::DuplicateMetadataNote:
    return "duplicate_metadata_note";
  case ParseErrorCode::UnsupportedMetadataEncoding:
    return "unsupported_metadata_encoding";
  case ParseErrorCode::InvalidMetadata:
    return "invalid_metadata";
  case ParseErrorCode::UnsupportedMetadataVersion:
    return "unsupported_metadata_version";
  case ParseErrorCode::MetadataTargetMismatch:
    return "metadata_target_mismatch";
  case ParseErrorCode::MissingDynamicSegment:
    return "missing_dynamic_segment";
  case ParseErrorCode::DuplicateDynamicSegment:
    return "duplicate_dynamic_segment";
  case ParseErrorCode::InvalidDynamicTable:
    return "invalid_dynamic_table";
  case ParseErrorCode::MissingDynamicEntry:
    return "missing_dynamic_entry";
  case ParseErrorCode::InvalidSymbolTable:
    return "invalid_symbol_table";
  case ParseErrorCode::InvalidStringTable:
    return "invalid_string_table";
  case ParseErrorCode::MissingKernelSymbol:
    return "missing_kernel_symbol";
  case ParseErrorCode::InvalidKernelSymbol:
    return "invalid_kernel_symbol";
  case ParseErrorCode::KernelDescriptorOutOfBounds:
    return "kernel_descriptor_out_of_bounds";
  case ParseErrorCode::InvalidKernelDescriptor:
    return "invalid_kernel_descriptor";
  case ParseErrorCode::KernelMetadataMismatch:
    return "kernel_metadata_mismatch";
  case ParseErrorCode::KernelEntryOutOfBounds:
    return "kernel_entry_out_of_bounds";
  }
  return "unknown";
}

ParseResult parse_code_object(const uint8_t *data, size_t size) {
  if (!data && size != 0) {
    return failure(ParseErrorCode::InvalidArgument, 0,
                   "non-empty code object has a null data pointer");
  }
  if (size < kElf64HeaderSize) {
    return failure(ParseErrorCode::TruncatedElfHeader, size,
                   "code object is smaller than an ELF64 header");
  }

  constexpr std::array<uint8_t, 4> kElfMagic = {0x7f, 'E', 'L', 'F'};
  if (!std::equal(kElfMagic.begin(), kElfMagic.end(), data)) {
    return failure(ParseErrorCode::InvalidElfMagic, 0,
                   "code object does not begin with the ELF magic");
  }
  if (data[kEiClass] != kElfClass64) {
    return failure(ParseErrorCode::UnsupportedElfClass, kEiClass,
                   "only ELF64 code objects are supported");
  }
  if (data[kEiData] != kElfDataLittleEndian) {
    return failure(ParseErrorCode::UnsupportedEndianness, kEiData,
                   "only little-endian code objects are supported");
  }
  if (data[kEiVersion] != kElfCurrentVersion ||
      read_u32(data + kElfVersionOffset) != kElfCurrentVersion) {
    return failure(ParseErrorCode::UnsupportedElfVersion, kEiVersion,
                   "ELF identification and header versions must be current");
  }
  if (data[kEiOsAbi] != kAmdgpuHsaOsAbi) {
    return failure(ParseErrorCode::UnsupportedOsAbi, kEiOsAbi,
                   "code object does not use the AMDGPU-HSA OS ABI");
  }
  if (data[kEiAbiVersion] < kMinimumSupportedAmdgpuHsaAbiVersion ||
      data[kEiAbiVersion] > kMaximumSupportedAmdgpuHsaAbiVersion) {
    return failure(ParseErrorCode::UnsupportedAbiVersion, kEiAbiVersion,
                   "only the observed AMDGPU-HSA ABI versions 3 and 4 are "
                   "supported");
  }
  if (read_u16(data + kElfTypeOffset) != kElfTypeDynamic) {
    return failure(ParseErrorCode::UnsupportedObjectType, kElfTypeOffset,
                   "AMDHSA code object must be an ELF shared object");
  }
  if (read_u16(data + kElfMachineOffset) != kElfMachineAmdgpu) {
    return failure(ParseErrorCode::UnsupportedMachine, kElfMachineOffset,
                   "ELF machine is not EM_AMDGPU");
  }
  if (read_u16(data + kElfHeaderSizeOffset) != kElf64HeaderSize) {
    return failure(ParseErrorCode::InvalidElfHeaderSize, kElfHeaderSizeOffset,
                   "ELF header size is not the ELF64 header size");
  }

  const uint32_t elf_flags = read_u32(data + kElfFlagsOffset);
  const uint32_t target_machine = elf_flags & kAmdgpuTargetMachineMask;
  if (target_machine != kAmdgpuMachineGfx1101) {
    return failure(ParseErrorCode::UnsupportedTarget, kElfFlagsOffset,
                   "initial light-rocr loader supports only gfx1101");
  }

  const uint16_t program_header_count =
      read_u16(data + kElfProgramHeaderCountOffset);
  if (program_header_count == 0) {
    return failure(ParseErrorCode::MissingProgramHeaders,
                   kElfProgramHeaderCountOffset,
                   "ELF code object has no program headers");
  }
  if (program_header_count == kExtendedProgramHeaderCount) {
    return failure(ParseErrorCode::UnsupportedExtendedProgramHeaderCount,
                   kElfProgramHeaderCountOffset,
                   "extended ELF program-header counts are not supported");
  }
  const uint16_t program_header_entry_size =
      read_u16(data + kElfProgramHeaderEntrySizeOffset);
  if (program_header_entry_size != kElf64ProgramHeaderSize) {
    return failure(ParseErrorCode::InvalidProgramHeaderEntrySize,
                   kElfProgramHeaderEntrySizeOffset,
                   "ELF program-header entry size is not the ELF64 size");
  }

  const uint64_t program_header_offset =
      read_u64(data + kElfProgramHeaderOffset);
  const uint64_t program_header_table_size =
      static_cast<uint64_t>(program_header_count) * kElf64ProgramHeaderSize;
  if (!range_is_in_file(program_header_offset, program_header_table_size,
                        size)) {
    return failure(ParseErrorCode::ProgramHeaderTableOutOfBounds,
                   kElfProgramHeaderOffset,
                   "ELF program-header table is outside the code object");
  }

  ParseResult result;
  result.code_object.abi_version = data[kEiAbiVersion];
  result.code_object.elf_flags = elf_flags;
  result.code_object.target_machine = target_machine;
  result.code_object.load_segments.reserve(program_header_count);
  std::vector<ParsedLoadSegment> parsed_load_segments;
  parsed_load_segments.reserve(program_header_count);

  for (uint16_t index = 0; index < program_header_count; ++index) {
    const uint64_t entry_offset =
        program_header_offset +
        static_cast<uint64_t>(index) * program_header_entry_size;
    const uint8_t *entry = data + static_cast<size_t>(entry_offset);
    if (read_u32(entry + kProgramHeaderTypeOffset) != kProgramHeaderTypeLoad) {
      continue;
    }

    LoadSegment segment;
    segment.flags = read_u32(entry + kProgramHeaderFlagsOffset);
    segment.file_offset = read_u64(entry + kProgramHeaderFileOffset);
    segment.virtual_address =
        read_u64(entry + kProgramHeaderVirtualAddressOffset);
    segment.file_size = read_u64(entry + kProgramHeaderFileSizeOffset);
    segment.memory_size = read_u64(entry + kProgramHeaderMemorySizeOffset);
    segment.alignment = read_u64(entry + kProgramHeaderAlignmentOffset);

    ParseResult validation = validate_load_segment(segment, size, entry_offset);
    if (!validation) {
      return validation;
    }
    result.code_object.load_segments.push_back(segment);
    parsed_load_segments.push_back(ParsedLoadSegment{segment, entry_offset});
  }

  if (result.code_object.load_segments.empty()) {
    return failure(ParseErrorCode::MissingLoadSegments, program_header_offset,
                   "ELF code object has no PT_LOAD segments");
  }

  const ParsedLoadSegment *file_overlap = find_overlapping_segment(
      parsed_load_segments,
      [](const LoadSegment &segment) { return segment.file_offset; },
      [](const LoadSegment &segment) { return segment.file_size; });
  if (file_overlap != nullptr) {
    return failure(ParseErrorCode::OverlappingSegmentFileRanges,
                   file_overlap->program_header_offset +
                       kProgramHeaderFileOffset,
                   "PT_LOAD segments have overlapping file ranges");
  }

  const ParsedLoadSegment *virtual_overlap = find_overlapping_segment(
      parsed_load_segments,
      [](const LoadSegment &segment) { return segment.virtual_address; },
      [](const LoadSegment &segment) { return segment.memory_size; });
  if (virtual_overlap != nullptr) {
    return failure(ParseErrorCode::OverlappingSegmentVirtualRanges,
                   virtual_overlap->program_header_offset +
                       kProgramHeaderVirtualAddressOffset,
                   "PT_LOAD segments have overlapping virtual ranges");
  }

  result.error =
      internal::decode_amdhsa_metadata(data, size, &result.code_object);

  return result;
}

} // namespace light_rocr::loader
