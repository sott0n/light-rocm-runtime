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
};

const char *parse_error_code_name(ParseErrorCode code);

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

struct CodeObject {
  uint8_t abi_version = 0;
  uint32_t elf_flags = 0;
  uint32_t target_machine = 0;
  std::vector<LoadSegment> load_segments;
};

struct ParseResult {
  CodeObject code_object;
  ParseError error;

  explicit operator bool() const { return error.code == ParseErrorCode::None; }
};

ParseResult parse_code_object(const uint8_t *data, size_t size);

} // namespace light_rocr::loader

#endif // LIGHT_ROCR_LOADER_CODE_OBJECT_HPP_
