#include "light_rocr/loader/code_object.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr size_t kElfHeaderSize = 64;
constexpr size_t kProgramHeaderSize = 56;
constexpr size_t kProgramHeaderOffset = kElfHeaderSize;

constexpr size_t kEiClass = 4;
constexpr size_t kEiData = 5;
constexpr size_t kEiVersion = 6;
constexpr size_t kEiOsAbi = 7;
constexpr size_t kEiAbiVersion = 8;
constexpr size_t kElfTypeOffset = 16;
constexpr size_t kElfMachineOffset = 18;
constexpr size_t kElfVersionOffset = 20;
constexpr size_t kElfProgramHeaderOffsetField = 32;
constexpr size_t kElfFlagsOffset = 48;
constexpr size_t kElfHeaderSizeOffset = 52;
constexpr size_t kElfProgramHeaderEntrySizeOffset = 54;
constexpr size_t kElfProgramHeaderCountOffset = 56;
constexpr size_t kProgramHeaderFileOffsetField = 8;
constexpr size_t kProgramHeaderVirtualAddressField = 16;

struct SegmentSpec {
  uint32_t type = 1;
  uint32_t flags = 4;
  uint64_t file_offset = 0;
  uint64_t virtual_address = 0;
  uint64_t file_size = 0x100;
  uint64_t memory_size = 0x100;
  uint64_t alignment = 0x1000;
};

void write_u16(std::vector<uint8_t> *bytes, size_t offset, uint16_t value) {
  (*bytes)[offset] = static_cast<uint8_t>(value & 0xffU);
  (*bytes)[offset + 1] = static_cast<uint8_t>((value >> 8U) & 0xffU);
}

void write_u32(std::vector<uint8_t> *bytes, size_t offset, uint32_t value) {
  for (size_t byte = 0; byte < 4; ++byte) {
    (*bytes)[offset + byte] = static_cast<uint8_t>(
        (value >> static_cast<unsigned int>(byte * 8U)) & 0xffU);
  }
}

void write_u64(std::vector<uint8_t> *bytes, size_t offset, uint64_t value) {
  for (size_t byte = 0; byte < 8; ++byte) {
    (*bytes)[offset + byte] = static_cast<uint8_t>(
        (value >> static_cast<unsigned int>(byte * 8U)) & 0xffU);
  }
}

void write_segment(std::vector<uint8_t> *bytes, size_t index,
                   const SegmentSpec &segment) {
  const size_t offset = kProgramHeaderOffset + index * kProgramHeaderSize;
  write_u32(bytes, offset, segment.type);
  write_u32(bytes, offset + 4, segment.flags);
  write_u64(bytes, offset + 8, segment.file_offset);
  write_u64(bytes, offset + 16, segment.virtual_address);
  write_u64(bytes, offset + 32, segment.file_size);
  write_u64(bytes, offset + 40, segment.memory_size);
  write_u64(bytes, offset + 48, segment.alignment);
}

std::vector<uint8_t> make_elf(
    std::vector<SegmentSpec> segments = {
        SegmentSpec{}, SegmentSpec{1, 5, 0x100, 0x1100, 0x40, 0x80, 0x1000}}) {
  const size_t program_header_table_end =
      kProgramHeaderOffset + segments.size() * kProgramHeaderSize;
  std::vector<uint8_t> bytes(std::max<size_t>(0x200, program_header_table_end),
                             0);
  bytes[0] = 0x7f;
  bytes[1] = 'E';
  bytes[2] = 'L';
  bytes[3] = 'F';
  bytes[kEiClass] = 2;
  bytes[kEiData] = 1;
  bytes[kEiVersion] = 1;
  bytes[kEiOsAbi] = light_rocr::loader::kAmdgpuHsaOsAbi;
  bytes[kEiAbiVersion] =
      light_rocr::loader::kMaximumSupportedAmdgpuHsaAbiVersion;
  write_u16(&bytes, kElfTypeOffset, 3);
  write_u16(&bytes, kElfMachineOffset, 224);
  write_u32(&bytes, kElfVersionOffset, 1);
  write_u64(&bytes, kElfProgramHeaderOffsetField, kProgramHeaderOffset);
  write_u32(&bytes, kElfFlagsOffset, light_rocr::loader::kAmdgpuMachineGfx1101);
  write_u16(&bytes, kElfHeaderSizeOffset, kElfHeaderSize);
  write_u16(&bytes, kElfProgramHeaderEntrySizeOffset, kProgramHeaderSize);
  write_u16(&bytes, kElfProgramHeaderCountOffset,
            static_cast<uint16_t>(segments.size()));
  for (size_t index = 0; index < segments.size(); ++index) {
    write_segment(&bytes, index, segments[index]);
  }
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

using TestFunction = std::function<void(TestContext *)>;

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

void expect_error_at(TestContext *context, const std::vector<uint8_t> &bytes,
                     light_rocr::loader::ParseErrorCode expected,
                     uint64_t expected_offset) {
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(!result, "parse unexpectedly succeeded");
  context->expect(
      result.error.code == expected,
      std::string("expected ") +
          light_rocr::loader::parse_error_code_name(expected) + ", got " +
          light_rocr::loader::parse_error_code_name(result.error.code));
  context->expect(result.error.offset == expected_offset,
                  "expected error at byte " + std::to_string(expected_offset) +
                      ", got " + std::to_string(result.error.offset));
}

void valid_code_object(TestContext *context) {
  const auto bytes = make_elf();
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  context->expect(result.code_object.abi_version == 4,
                  "unexpected ABI version");
  context->expect(result.code_object.target_machine == 0x46,
                  "unexpected target machine");
  context->expect(result.code_object.load_segments.size() == 2,
                  "expected two load segments");
  if (result.code_object.load_segments.size() == 2) {
    const auto &segment = result.code_object.load_segments[1];
    context->expect(segment.file_offset == 0x100,
                    "unexpected second file offset");
    context->expect(segment.virtual_address == 0x1100,
                    "unexpected second virtual address");
    context->expect(segment.file_size == 0x40, "unexpected second file size");
    context->expect(segment.memory_size == 0x80,
                    "unexpected second memory size");
    context->expect(segment.flags == 5, "unexpected second flags");
  }
}

void null_nonempty_input(TestContext *context) {
  const auto result = light_rocr::loader::parse_code_object(nullptr, 1);
  context->expect(!result, "parse unexpectedly succeeded");
  context->expect(result.error.code ==
                      light_rocr::loader::ParseErrorCode::InvalidArgument,
                  "expected invalid argument");
}

void observed_abi_versions(TestContext *context) {
  auto bytes = make_elf();
  bytes[kEiAbiVersion] = 3;
  auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  context->expect(result.code_object.abi_version == 3,
                  "expected IREE's ABI version 3");

  bytes[kEiAbiVersion] = 4;
  result = light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  context->expect(result.code_object.abi_version == 4,
                  "expected Clang's ABI version 4");
}

void truncated_header(TestContext *context) {
  std::vector<uint8_t> bytes(kElfHeaderSize - 1, 0);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::TruncatedElfHeader);
}

void truncated_prefixes(TestContext *context) {
  const auto bytes = make_elf();
  constexpr size_t kLastSegmentFileEnd = 0x140;
  for (size_t prefix_size = 0; prefix_size < kLastSegmentFileEnd;
       ++prefix_size) {
    const auto result =
        light_rocr::loader::parse_code_object(bytes.data(), prefix_size);
    context->expect(!result, "truncated prefix of " +
                                 std::to_string(prefix_size) +
                                 " bytes unexpectedly succeeded");
  }

  const auto exact_result =
      light_rocr::loader::parse_code_object(bytes.data(), kLastSegmentFileEnd);
  context->expect(static_cast<bool>(exact_result),
                  "code object ending at the final segment was rejected: " +
                      exact_result.error.message);
}

void invalid_identity(TestContext *context) {
  auto bytes = make_elf();
  bytes[0] = 0;
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidElfMagic);

  bytes = make_elf();
  bytes[kEiClass] = 1;
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::UnsupportedElfClass);

  bytes = make_elf();
  bytes[kEiData] = 2;
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::UnsupportedEndianness);

  bytes = make_elf();
  bytes[kEiVersion] = 0;
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::UnsupportedElfVersion);

  bytes = make_elf();
  bytes[kEiOsAbi] = 0;
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::UnsupportedOsAbi);

  bytes = make_elf();
  bytes[kEiAbiVersion] = 2;
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::UnsupportedAbiVersion);

  bytes = make_elf();
  bytes[kEiAbiVersion] = 5;
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::UnsupportedAbiVersion);
}

void invalid_header_fields(TestContext *context) {
  auto bytes = make_elf();
  write_u16(&bytes, kElfTypeOffset, 1);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::UnsupportedObjectType);

  bytes = make_elf();
  write_u16(&bytes, kElfMachineOffset, 62);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::UnsupportedMachine);

  bytes = make_elf();
  write_u32(&bytes, kElfFlagsOffset, 0x41);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::UnsupportedTarget);

  bytes = make_elf();
  write_u16(&bytes, kElfHeaderSizeOffset, 63);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidElfHeaderSize);
}

void invalid_program_header_table(TestContext *context) {
  auto bytes = make_elf();
  write_u16(&bytes, kElfProgramHeaderCountOffset, 0);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::MissingProgramHeaders);

  bytes = make_elf();
  write_u16(&bytes, kElfProgramHeaderCountOffset, 0xffff);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::
                   UnsupportedExtendedProgramHeaderCount);

  bytes = make_elf();
  write_u16(&bytes, kElfProgramHeaderEntrySizeOffset, 55);
  expect_error(
      context, bytes,
      light_rocr::loader::ParseErrorCode::InvalidProgramHeaderEntrySize);

  bytes = make_elf();
  write_u64(&bytes, kElfProgramHeaderOffsetField,
            std::numeric_limits<uint64_t>::max() - 8);
  expect_error(
      context, bytes,
      light_rocr::loader::ParseErrorCode::ProgramHeaderTableOutOfBounds);

  bytes = make_elf();
  write_u64(&bytes, kElfProgramHeaderOffsetField, bytes.size() - 1);
  expect_error(
      context, bytes,
      light_rocr::loader::ParseErrorCode::ProgramHeaderTableOutOfBounds);
}

void invalid_segment_sizes(TestContext *context) {
  auto bytes = make_elf({SegmentSpec{1, 4, 0, 0, 0x101, 0x100, 1}});
  expect_error(
      context, bytes,
      light_rocr::loader::ParseErrorCode::SegmentFileSizeExceedsMemorySize);

  bytes = make_elf({SegmentSpec{1, 4, 0x1f0, 0x1f0, 0x20, 0x20, 1}});
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::SegmentFileRangeOutOfBounds);

  bytes = make_elf({SegmentSpec{1, 4, 0, UINT64_MAX - 8, 0, 16, 1}});
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::SegmentVirtualRangeOverflow);

  bytes = make_elf({SegmentSpec{1, 8, 0, 0, 0x100, 0x100, 1}});
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidSegmentFlags);
}

void invalid_segment_alignment(TestContext *context) {
  auto bytes = make_elf({SegmentSpec{1, 4, 0, 0, 0x100, 0x100, 3}});
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidSegmentAlignment);

  bytes = make_elf({SegmentSpec{1, 4, 0x100, 0x1200, 0x40, 0x40, 0x1000}});
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::IncongruentSegmentAlignment);
}

void overlapping_segments(TestContext *context) {
  auto bytes = make_elf({SegmentSpec{1, 4, 0, 0, 0x100, 0x100, 1},
                         SegmentSpec{1, 5, 0x80, 0x200, 0x40, 0x40, 1}});
  expect_error_at(
      context, bytes,
      light_rocr::loader::ParseErrorCode::OverlappingSegmentFileRanges,
      kProgramHeaderOffset + kProgramHeaderSize +
          kProgramHeaderFileOffsetField);

  bytes = make_elf({SegmentSpec{1, 4, 0, 0, 0x100, 0x100, 1},
                    SegmentSpec{1, 5, 0x100, 0x80, 0x40, 0x80, 1}});
  expect_error_at(
      context, bytes,
      light_rocr::loader::ParseErrorCode::OverlappingSegmentVirtualRanges,
      kProgramHeaderOffset + kProgramHeaderSize +
          kProgramHeaderVirtualAddressField);
}

void segment_order_is_preserved(TestContext *context) {
  const auto bytes = make_elf({SegmentSpec{1, 4, 0x180, 0x3000, 0x20, 0x20, 1},
                               SegmentSpec{1, 5, 0, 0x1000, 0x40, 0x40, 1}});
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  context->expect(result.code_object.load_segments.size() == 2,
                  "expected two load segments");
  if (result.code_object.load_segments.size() == 2) {
    context->expect(result.code_object.load_segments[0].file_offset == 0x180,
                    "program-header order was not preserved");
    context->expect(result.code_object.load_segments[1].file_offset == 0,
                    "program-header order was not preserved");
  }

  const auto overlapping_bytes =
      make_elf({SegmentSpec{1, 4, 0x80, 0x3000, 0x40, 0x40, 1},
                SegmentSpec{1, 5, 0, 0x1000, 0x100, 0x100, 1}});
  expect_error_at(
      context, overlapping_bytes,
      light_rocr::loader::ParseErrorCode::OverlappingSegmentFileRanges,
      kProgramHeaderOffset + kProgramHeaderFileOffsetField);
}

void many_nonoverlapping_segments(TestContext *context) {
  constexpr size_t kSegmentCount = 8192;
  std::vector<SegmentSpec> segments(kSegmentCount);
  for (size_t index = 0; index < segments.size(); ++index) {
    segments[index].file_size = 0;
    segments[index].memory_size = 0;
    segments[index].virtual_address = static_cast<uint64_t>(index) * 0x1000;
    segments[index].alignment = 1;
  }

  const auto bytes = make_elf(std::move(segments));
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  context->expect(result.code_object.load_segments.size() == kSegmentCount,
                  "expected all nonoverlapping load segments");
}

void missing_load_segments(TestContext *context) {
  auto bytes = make_elf({SegmentSpec{4, 4, 0, 0, 0x100, 0x100, 1}});
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::MissingLoadSegments);
}

void empty_file_segment_is_valid(TestContext *context) {
  const auto bytes =
      make_elf({SegmentSpec{1, 6, 0x200, 0x200, 0, 0x80, 0x100}});
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  context->expect(result.code_object.load_segments.size() == 1,
                  "expected a BSS-only load segment");
}

} // namespace

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"valid code object", valid_code_object},
      {"null nonempty input", null_nonempty_input},
      {"observed ABI versions", observed_abi_versions},
      {"truncated header", truncated_header},
      {"truncated prefixes", truncated_prefixes},
      {"invalid identity", invalid_identity},
      {"invalid header fields", invalid_header_fields},
      {"invalid program-header table", invalid_program_header_table},
      {"invalid segment sizes", invalid_segment_sizes},
      {"invalid segment alignment", invalid_segment_alignment},
      {"overlapping segments", overlapping_segments},
      {"segment order is preserved", segment_order_is_preserved},
      {"many nonoverlapping segments", many_nonoverlapping_segments},
      {"missing load segments", missing_load_segments},
      {"empty file segment", empty_file_segment_is_valid},
  };

  int failures = 0;
  for (const auto &[name, function] : tests) {
    std::cout << "[ RUN      ] " << name << '\n';
    TestContext context;
    function(&context);
    failures += context.failures;
    if (context.failures == 0) {
      std::cout << "[       OK ] " << name << '\n';
    } else {
      std::cout << "[  FAILED  ] " << name << '\n';
    }
  }
  if (failures != 0) {
    std::cerr << failures << " assertion(s) failed\n";
    return 1;
  }
  std::cout << tests.size() << " test case(s) passed\n";
  return 0;
}
