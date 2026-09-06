#ifndef LIGHT_ROCR_LOADER_CODE_OBJECT_HPP_
#define LIGHT_ROCR_LOADER_CODE_OBJECT_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace light_rocr::loader {

inline constexpr uint8_t kAmdgpuHsaOsAbi = 64;
inline constexpr uint8_t kMinimumSupportedAmdgpuHsaAbiVersion = 3;
inline constexpr uint8_t kMaximumSupportedAmdgpuHsaAbiVersion = 4;
inline constexpr uint32_t kAmdgpuTargetMachineMask = 0xff;
inline constexpr uint32_t kAmdgpuMachineGfx1101 = 0x46;

enum class ParseErrorCode {
  None = 0,
  InvalidArgument,
  TruncatedElfHeader,
  InvalidElfMagic,
  UnsupportedElfClass,
  UnsupportedEndianness,
  UnsupportedElfVersion,
  UnsupportedOsAbi,
  UnsupportedAbiVersion,
  UnsupportedObjectType,
  UnsupportedMachine,
  UnsupportedTarget,
  InvalidElfHeaderSize,
  MissingProgramHeaders,
  UnsupportedExtendedProgramHeaderCount,
  InvalidProgramHeaderEntrySize,
  ProgramHeaderTableOutOfBounds,
  InvalidSegmentFlags,
  SegmentFileSizeExceedsMemorySize,
  SegmentFileRangeOutOfBounds,
  SegmentVirtualRangeOverflow,
  InvalidSegmentAlignment,
  IncongruentSegmentAlignment,
  OverlappingSegmentFileRanges,
  OverlappingSegmentVirtualRanges,
  MissingLoadSegments,
  MalformedNote,
  DuplicateMetadataNote,
  UnsupportedMetadataEncoding,
  InvalidMetadata,
  UnsupportedMetadataVersion,
  MetadataTargetMismatch,
  MissingDynamicSegment,
  DuplicateDynamicSegment,
  InvalidDynamicTable,
  MissingDynamicEntry,
  InvalidSymbolTable,
  InvalidStringTable,
  MissingKernelSymbol,
  InvalidKernelSymbol,
  KernelDescriptorOutOfBounds,
  InvalidKernelDescriptor,
  KernelMetadataMismatch,
  KernelEntryOutOfBounds,
  InvalidRelocationTable,
  UnsupportedRelocationFormat,
  UnsupportedRelocationType,
  RelocationSymbolOutOfBounds,
  RelocationTargetOutOfBounds,
};

const char *parse_error_code_name(ParseErrorCode code);
const char *amdgpu_relocation_type_name(uint32_t type);

struct ParseError {
  ParseErrorCode code = ParseErrorCode::None;
  uint64_t offset = 0;
  std::string message;
};

struct LoadSegment {
  uint64_t file_offset = 0;
  uint64_t file_size = 0;
  uint64_t virtual_address = 0;
  uint64_t memory_size = 0;
  uint64_t alignment = 0;
  uint32_t flags = 0;
};

struct MetadataVersion {
  uint32_t major = 0;
  uint32_t minor = 0;
};

struct KernelInfo {
  // Source-level kernel name and ELF kernel-descriptor symbol name.
  std::string name;
  std::string symbol_name;

  uint64_t descriptor_virtual_address = 0;
  int64_t code_entry_byte_offset = 0;
  uint64_t code_entry_virtual_address = 0;

  uint32_t kernarg_size = 0;
  uint32_t metadata_kernarg_alignment = 0;
  uint32_t kernarg_alignment = 0;
  uint32_t group_segment_size = 0;
  uint32_t private_segment_size = 0;
  uint32_t wavefront_size = 0;
  bool uses_dynamic_stack = false;

  uint32_t compute_pgm_rsrc1 = 0;
  uint32_t compute_pgm_rsrc2 = 0;
  uint32_t compute_pgm_rsrc3 = 0;
  uint16_t kernel_code_properties = 0;
  uint16_t kernarg_preload = 0;
};

enum class RelocationEncoding {
  Rel,
  Rela,
};

struct LoadCopy {
  uint64_t file_offset = 0;
  uint64_t virtual_address = 0;
  uint64_t size = 0;
};

struct LoadZeroFill {
  uint64_t virtual_address = 0;
  uint64_t size = 0;
};

struct LoadProtection {
  uint64_t virtual_address = 0;
  uint64_t size = 0;
  uint32_t flags = 0;
};

struct Relocation {
  RelocationEncoding encoding = RelocationEncoding::Rela;
  uint64_t target_virtual_address = 0;
  uint32_t type = 0;
  uint32_t symbol_index = 0;
  int64_t addend = 0;
};

struct LoadPlan {
  uint64_t image_virtual_address = 0;
  uint64_t image_size = 0;
  uint64_t alignment = 1;
  std::vector<LoadCopy> copies;
  std::vector<LoadZeroFill> zero_fills;
  std::vector<LoadProtection> protections;
  std::vector<Relocation> relocations;
};

struct CodeObject {
  uint8_t abi_version = 0;
  uint32_t elf_flags = 0;
  uint32_t target_machine = 0;
  std::vector<LoadSegment> load_segments;
  bool has_metadata = false;
  MetadataVersion metadata_version;
  std::string target_isa;
  std::vector<KernelInfo> kernels;
  LoadPlan load_plan;
};

struct ParseResult {
  CodeObject code_object;
  ParseError error;

  explicit operator bool() const { return error.code == ParseErrorCode::None; }
};

ParseResult parse_code_object(const uint8_t *data, size_t size);

} // namespace light_rocr::loader

#endif // LIGHT_ROCR_LOADER_CODE_OBJECT_HPP_
