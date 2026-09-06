#include "light_rocr/loader/code_object.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr size_t kElfHeaderSize = 64;
constexpr size_t kProgramHeaderSize = 56;
constexpr size_t kProgramHeaderOffset = kElfHeaderSize;
constexpr size_t kDynamicOffset = 0x100;
constexpr size_t kRelaOffset = 0x220;
constexpr size_t kRelOffset = 0x240;
constexpr size_t kHashOffset = 0x280;
constexpr uint64_t kRelocationTarget = 0x2d0;

constexpr int64_t kDynamicPltRelocationSize = 2;
constexpr int64_t kDynamicHash = 4;
constexpr int64_t kDynamicRela = 7;
constexpr int64_t kDynamicRelaSize = 8;
constexpr int64_t kDynamicRelaEntrySize = 9;
constexpr int64_t kDynamicRel = 17;
constexpr int64_t kDynamicRelSize = 18;
constexpr int64_t kDynamicRelEntrySize = 19;
constexpr int64_t kDynamicRelr = 36;
constexpr int64_t kDynamicRelaCount = 0x6ffffff9;

using DynamicEntry = std::pair<int64_t, uint64_t>;

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

struct SegmentSpec {
  uint32_t type = 1;
  uint32_t flags = 4;
  uint64_t file_offset = 0;
  uint64_t virtual_address = 0;
  uint64_t file_size = 0;
  uint64_t memory_size = 0;
  uint64_t alignment = 1;
};

void write_segment(std::vector<uint8_t> *bytes, size_t index,
                   const SegmentSpec &segment) {
  const size_t offset = kProgramHeaderOffset + index * kProgramHeaderSize;
  write_u32(bytes, offset, segment.type);
  write_u32(bytes, offset + 4, segment.flags);
  write_u64(bytes, offset + 8, segment.file_offset);
  write_u64(bytes, offset + 16, segment.virtual_address);
  write_u64(bytes, offset + 24, segment.virtual_address);
  write_u64(bytes, offset + 32, segment.file_size);
  write_u64(bytes, offset + 40, segment.memory_size);
  write_u64(bytes, offset + 48, segment.alignment);
}

std::vector<uint8_t>
make_elf(const std::vector<DynamicEntry> &dynamic_entries = {},
         bool terminate_dynamic = true, bool reverse_load_segments = false,
         bool include_dynamic = true) {
  std::vector<uint8_t> bytes(0x500, 0);
  bytes[0] = 0x7f;
  bytes[1] = 'E';
  bytes[2] = 'L';
  bytes[3] = 'F';
  bytes[4] = 2;
  bytes[5] = 1;
  bytes[6] = 1;
  bytes[7] = light_rocr::loader::kAmdgpuHsaOsAbi;
  bytes[8] = light_rocr::loader::kMaximumSupportedAmdgpuHsaAbiVersion;
  write_u16(&bytes, 16, 3);
  write_u16(&bytes, 18, 224);
  write_u32(&bytes, 20, 1);
  write_u64(&bytes, 32, kProgramHeaderOffset);
  write_u32(&bytes, 48, light_rocr::loader::kAmdgpuMachineGfx1101);
  write_u16(&bytes, 52, kElfHeaderSize);
  write_u16(&bytes, 54, kProgramHeaderSize);

  const SegmentSpec first_load{1, 6, 0, 0, 0x300, 0x340, 0x100};
  const SegmentSpec second_load{1, 5, 0x400, 0x600, 0x40, 0x80, 0x200};
  const SegmentSpec dynamic{
      2,
      6,
      kDynamicOffset,
      kDynamicOffset,
      static_cast<uint64_t>(
          (dynamic_entries.size() + (terminate_dynamic ? 1U : 0U)) * 16U),
      static_cast<uint64_t>(
          (dynamic_entries.size() + (terminate_dynamic ? 1U : 0U)) * 16U),
      8};

  std::vector<SegmentSpec> segments;
  if (reverse_load_segments) {
    segments = {second_load, first_load};
  } else {
    segments = {first_load, second_load};
  }
  if (include_dynamic) {
    segments.push_back(dynamic);
  }
  write_u16(&bytes, 56, static_cast<uint16_t>(segments.size()));
  for (size_t index = 0; index < segments.size(); ++index) {
    write_segment(&bytes, index, segments[index]);
  }

  size_t dynamic_offset = kDynamicOffset;
  for (const DynamicEntry &entry : dynamic_entries) {
    write_u64(&bytes, dynamic_offset, static_cast<uint64_t>(entry.first));
    write_u64(&bytes, dynamic_offset + 8, entry.second);
    dynamic_offset += 16;
  }
  if (terminate_dynamic) {
    write_u64(&bytes, dynamic_offset, 0);
    write_u64(&bytes, dynamic_offset + 8, 0);
  }

  write_u32(&bytes, kHashOffset, 1);
  write_u32(&bytes, kHashOffset + 4, 2);
  return bytes;
}

void write_relocation(std::vector<uint8_t> *bytes, size_t offset,
                      uint64_t target, uint32_t type, uint32_t symbol_index,
                      bool has_addend = false, int64_t addend = 0) {
  write_u64(bytes, offset, target);
  const uint64_t info = static_cast<uint64_t>(symbol_index) << 32U | type;
  write_u64(bytes, offset + 8, info);
  if (has_addend) {
    write_u64(bytes, offset + 16, static_cast<uint64_t>(addend));
  }
}

std::vector<DynamicEntry> rela_entries(uint64_t size = 24,
                                       uint64_t entry_size = 24,
                                       bool include_hash = true) {
  std::vector<DynamicEntry> entries = {
      {kDynamicRela, kRelaOffset},
      {kDynamicRelaSize, size},
      {kDynamicRelaEntrySize, entry_size},
  };
  if (include_hash) {
    entries.emplace_back(kDynamicHash, kHashOffset);
  }
  return entries;
}

std::vector<DynamicEntry> rel_entries(uint64_t size = 16,
                                      uint64_t entry_size = 16) {
  return {
      {kDynamicRel, kRelOffset},
      {kDynamicRelSize, size},
      {kDynamicRelEntrySize, entry_size},
      {kDynamicHash, kHashOffset},
  };
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
          light_rocr::loader::parse_error_code_name(result.error.code) + ": " +
          result.error.message);
}

void segment_plan(TestContext *context) {
  const auto bytes = make_elf();
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  const auto &plan = result.code_object.load_plan;
  context->expect(plan.image_virtual_address == 0,
                  "unexpected image virtual address");
  context->expect(plan.image_size == 0x680, "unexpected image size");
  context->expect(plan.alignment == 0x200, "unexpected image alignment");
  context->expect(plan.copies.size() == 2, "expected two copy operations");
  context->expect(plan.zero_fills.size() == 2,
                  "expected two zero-fill operations");
  context->expect(plan.protections.size() == 2,
                  "expected two protection operations");
  context->expect(plan.relocations.empty(), "expected no relocations");
  if (plan.copies.size() == 2) {
    context->expect(plan.copies[0].virtual_address == 0 &&
                        plan.copies[0].file_offset == 0 &&
                        plan.copies[0].size == 0x300,
                    "unexpected first copy operation");
    context->expect(plan.copies[1].virtual_address == 0x600 &&
                        plan.copies[1].file_offset == 0x400 &&
                        plan.copies[1].size == 0x40,
                    "unexpected second copy operation");
  }
  if (plan.zero_fills.size() == 2) {
    context->expect(plan.zero_fills[0].virtual_address == 0x300 &&
                        plan.zero_fills[0].size == 0x40,
                    "unexpected first zero-fill operation");
    context->expect(plan.zero_fills[1].virtual_address == 0x640 &&
                        plan.zero_fills[1].size == 0x40,
                    "unexpected second zero-fill operation");
  }
  if (plan.protections.size() == 2) {
    context->expect(plan.protections[0].virtual_address == 0 &&
                        plan.protections[0].size == 0x340 &&
                        plan.protections[0].flags == 6,
                    "unexpected first protection operation");
    context->expect(plan.protections[1].virtual_address == 0x600 &&
                        plan.protections[1].size == 0x80 &&
                        plan.protections[1].flags == 5,
                    "unexpected second protection operation");
  }
}

void plan_without_dynamic_segment(TestContext *context) {
  const auto bytes = make_elf({}, true, false, false);
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  context->expect(result.code_object.load_plan.relocations.empty(),
                  "object without PT_DYNAMIC has relocations");
}

void deterministic_segment_order(TestContext *context) {
  const auto bytes = make_elf({}, true, true);
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  const auto &copies = result.code_object.load_plan.copies;
  context->expect(copies.size() == 2, "expected two copy operations");
  if (copies.size() == 2) {
    context->expect(copies[0].virtual_address == 0 &&
                        copies[1].virtual_address == 0x600,
                    "copy operations are not sorted by virtual address");
  }
}

void zero_memory_segments_do_not_expand_image(TestContext *context) {
  auto bytes = make_elf();
  write_u64(&bytes, kProgramHeaderOffset + 32, 0);
  write_u64(&bytes, kProgramHeaderOffset + 40, 0);
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  const auto &plan = result.code_object.load_plan;
  context->expect(plan.image_virtual_address == 0x600,
                  "zero-memory segment changed the image base");
  context->expect(plan.image_size == 0x80,
                  "zero-memory segment changed the image size");
  context->expect(plan.alignment == 0x200,
                  "zero-memory segment changed the image alignment");
  context->expect(plan.copies.size() == 1 && plan.zero_fills.size() == 1 &&
                      plan.protections.size() == 1,
                  "zero-memory segment produced load operations");
}

void empty_image_plan(TestContext *context) {
  auto bytes = make_elf();
  write_u64(&bytes, kProgramHeaderOffset + 32, 0);
  write_u64(&bytes, kProgramHeaderOffset + 40, 0);
  write_u64(&bytes, kProgramHeaderOffset + kProgramHeaderSize + 32, 0);
  write_u64(&bytes, kProgramHeaderOffset + kProgramHeaderSize + 40, 0);
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  const auto &plan = result.code_object.load_plan;
  context->expect(plan.image_virtual_address == 0 && plan.image_size == 0 &&
                      plan.alignment == 1,
                  "empty image has a non-empty allocation plan");
  context->expect(plan.copies.empty() && plan.zero_fills.empty() &&
                      plan.protections.empty(),
                  "empty image produced load operations");
}

void rela_plan(TestContext *context) {
  auto bytes = make_elf(rela_entries());
  write_relocation(&bytes, kRelaOffset, kRelocationTarget, 3, 1, true, -7);
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  const auto &relocations = result.code_object.load_plan.relocations;
  context->expect(relocations.size() == 1, "expected one RELA relocation");
  if (relocations.size() == 1) {
    const auto &relocation = relocations[0];
    context->expect(relocation.encoding ==
                        light_rocr::loader::RelocationEncoding::Rela,
                    "unexpected relocation encoding");
    context->expect(relocation.target_virtual_address == kRelocationTarget,
                    "unexpected relocation target");
    context->expect(relocation.type == 3, "unexpected relocation type");
    context->expect(relocation.symbol_index == 1,
                    "unexpected relocation symbol index");
    context->expect(relocation.addend == -7, "unexpected relocation addend");
  }
}

void rel_plan(TestContext *context) {
  auto bytes = make_elf(rel_entries());
  write_relocation(&bytes, kRelOffset, kRelocationTarget, 13, 0);
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  const auto &relocations = result.code_object.load_plan.relocations;
  context->expect(relocations.size() == 1, "expected one REL relocation");
  if (relocations.size() == 1) {
    context->expect(relocations[0].encoding ==
                        light_rocr::loader::RelocationEncoding::Rel,
                    "unexpected relocation encoding");
    context->expect(relocations[0].type == 13,
                    "unexpected relative relocation type");
  }
}

void rel_and_rela_plan(TestContext *context) {
  auto entries = rel_entries();
  entries.pop_back();
  const auto rela = rela_entries();
  entries.insert(entries.end(), rela.begin(), rela.end());
  auto bytes = make_elf(entries);
  write_relocation(&bytes, kRelOffset, kRelocationTarget + 8, 13, 0);
  write_relocation(&bytes, kRelaOffset, kRelocationTarget, 3, 1, true, -7);
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  const auto &relocations = result.code_object.load_plan.relocations;
  context->expect(relocations.size() == 2, "expected REL and RELA records");
  if (relocations.size() == 2) {
    context->expect(
        relocations[0].target_virtual_address == kRelocationTarget &&
            relocations[1].target_virtual_address == kRelocationTarget + 8,
        "mixed relocation records are not deterministically "
        "ordered");
  }
}

void dynamic_tag_order_is_irrelevant(TestContext *context) {
  auto entries = rela_entries();
  std::reverse(entries.begin(), entries.end());
  auto bytes = make_elf(entries);
  write_relocation(&bytes, kRelaOffset, kRelocationTarget, 3, 1, true, -7);
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  context->expect(result.code_object.load_plan.relocations.size() == 1,
                  "reordered dynamic tags lost the relocation");
}

void truncated_relocation_prefixes(TestContext *context) {
  auto bytes = make_elf(rela_entries());
  write_relocation(&bytes, kRelaOffset, kRelocationTarget, 3, 1, true, -7);
  constexpr size_t kLastLoadFileEnd = 0x440;
  for (size_t prefix_size = 0; prefix_size < kLastLoadFileEnd; ++prefix_size) {
    const auto result =
        light_rocr::loader::parse_code_object(bytes.data(), prefix_size);
    context->expect(!result, "truncated relocation object prefix of " +
                                 std::to_string(prefix_size) +
                                 " bytes unexpectedly succeeded");
  }
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), kLastLoadFileEnd);
  context->expect(static_cast<bool>(result),
                  "object ending at the last PT_LOAD byte was rejected: " +
                      result.error.message);
}

void zero_sized_table(TestContext *context) {
  auto entries = rela_entries(0);
  entries.erase(entries.begin() + 3);
  const auto bytes = make_elf(entries);
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(static_cast<bool>(result), result.error.message);
  context->expect(result.code_object.load_plan.relocations.empty(),
                  "zero-sized table produced relocations");
}

void missing_relocation_tag(TestContext *context) {
  const auto bytes =
      make_elf({{kDynamicRela, kRelaOffset}, {kDynamicRelaSize, 24}});
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidRelocationTable);
}

void duplicate_relocation_tag(TestContext *context) {
  auto entries = rela_entries();
  entries.insert(entries.begin() + 1, {kDynamicRela, kRelaOffset});
  const auto bytes = make_elf(entries);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidDynamicTable);
}

void invalid_relocation_entry_size(TestContext *context) {
  const auto bytes = make_elf(rela_entries(24, 16));
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidRelocationTable);
}

void partial_relocation_entry(TestContext *context) {
  const auto bytes = make_elf(rela_entries(25));
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidRelocationTable);
}

void relocation_table_out_of_bounds(TestContext *context) {
  auto entries = rela_entries();
  entries[0].second = 0x320;
  const auto bytes = make_elf(entries);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidRelocationTable);
}

void overlapping_relocation_tables(TestContext *context) {
  auto entries = rel_entries();
  entries[0].second = kRelaOffset;
  entries.pop_back();
  const auto rela = rela_entries();
  entries.insert(entries.end(), rela.begin(), rela.end());
  const auto bytes = make_elf(entries);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidRelocationTable);
}

void unknown_relocation_type(TestContext *context) {
  auto bytes = make_elf(rela_entries());
  write_relocation(&bytes, kRelaOffset, kRelocationTarget, 12, 0, true);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::UnsupportedRelocationType);
}

void failed_plan_is_transactional(TestContext *context) {
  auto bytes = make_elf(rela_entries());
  write_relocation(&bytes, kRelaOffset, kRelocationTarget, 12, 0, true);
  const auto result =
      light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
  context->expect(!result, "invalid relocation unexpectedly succeeded");
  const auto &plan = result.code_object.load_plan;
  context->expect(plan.image_virtual_address == 0 && plan.image_size == 0 &&
                      plan.alignment == 1,
                  "failed load plan published partial image state");
  context->expect(plan.copies.empty() && plan.zero_fills.empty() &&
                      plan.protections.empty() && plan.relocations.empty(),
                  "failed load plan published partial operations");
}

void relocation_symbol_out_of_bounds(TestContext *context) {
  auto bytes = make_elf(rela_entries());
  write_relocation(&bytes, kRelaOffset, kRelocationTarget, 3, 2, true);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::RelocationSymbolOutOfBounds);
}

void relocation_target_out_of_bounds(TestContext *context) {
  auto bytes = make_elf(rela_entries());
  write_relocation(&bytes, kRelaOffset, 0x400, 3, 0, true);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::RelocationTargetOutOfBounds);
}

void missing_hash_for_relocations(TestContext *context) {
  auto bytes = make_elf(rela_entries(24, 24, false));
  write_relocation(&bytes, kRelaOffset, kRelocationTarget, 3, 0, true);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidRelocationTable);
}

void relative_count_exceeds_table(TestContext *context) {
  auto entries = rela_entries();
  entries.emplace_back(kDynamicRelaCount, 2);
  const auto bytes = make_elf(entries);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidRelocationTable);
}

void invalid_relative_prefix(TestContext *context) {
  auto entries = rela_entries();
  entries.emplace_back(kDynamicRelaCount, 1);
  auto bytes = make_elf(entries);
  write_relocation(&bytes, kRelaOffset, kRelocationTarget, 3, 0, true);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidRelocationTable);
}

void known_relocation_types(TestContext *context) {
  constexpr uint32_t known_types[] = {0, 1, 2, 3,  4,  5,  6,
                                      7, 8, 9, 10, 11, 13, 14};
  for (const uint32_t type : known_types) {
    auto bytes = make_elf(rela_entries());
    write_relocation(&bytes, kRelaOffset, kRelocationTarget, type, 0, true);
    const auto result =
        light_rocr::loader::parse_code_object(bytes.data(), bytes.size());
    context->expect(static_cast<bool>(result),
                    "known relocation type " + std::to_string(type) +
                        " was rejected: " + result.error.message);
  }
}

void unsupported_plt_relocations(TestContext *context) {
  const auto bytes = make_elf({{kDynamicPltRelocationSize, 24}});
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::UnsupportedRelocationFormat);
}

void unsupported_relr_relocations(TestContext *context) {
  const auto bytes = make_elf({{kDynamicRelr, kRelOffset}});
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::UnsupportedRelocationFormat);
}

void unterminated_dynamic_table(TestContext *context) {
  const auto bytes = make_elf({{kDynamicHash, kHashOffset}}, false);
  expect_error(context, bytes,
               light_rocr::loader::ParseErrorCode::InvalidDynamicTable);
}

void relocation_names(TestContext *context) {
  context->expect(std::string(light_rocr::loader::amdgpu_relocation_type_name(
                      3)) == "R_AMDGPU_ABS64",
                  "known relocation name is wrong");
  context->expect(std::string(light_rocr::loader::amdgpu_relocation_type_name(
                      12)) == "R_AMDGPU_UNKNOWN",
                  "unknown relocation name is wrong");
}

} // namespace

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"segment_plan", segment_plan},
      {"plan_without_dynamic_segment", plan_without_dynamic_segment},
      {"deterministic_segment_order", deterministic_segment_order},
      {"zero_memory_segments_do_not_expand_image",
       zero_memory_segments_do_not_expand_image},
      {"empty_image_plan", empty_image_plan},
      {"rela_plan", rela_plan},
      {"rel_plan", rel_plan},
      {"rel_and_rela_plan", rel_and_rela_plan},
      {"dynamic_tag_order_is_irrelevant", dynamic_tag_order_is_irrelevant},
      {"truncated_relocation_prefixes", truncated_relocation_prefixes},
      {"zero_sized_table", zero_sized_table},
      {"missing_relocation_tag", missing_relocation_tag},
      {"duplicate_relocation_tag", duplicate_relocation_tag},
      {"invalid_relocation_entry_size", invalid_relocation_entry_size},
      {"partial_relocation_entry", partial_relocation_entry},
      {"relocation_table_out_of_bounds", relocation_table_out_of_bounds},
      {"overlapping_relocation_tables", overlapping_relocation_tables},
      {"unknown_relocation_type", unknown_relocation_type},
      {"failed_plan_is_transactional", failed_plan_is_transactional},
      {"relocation_symbol_out_of_bounds", relocation_symbol_out_of_bounds},
      {"relocation_target_out_of_bounds", relocation_target_out_of_bounds},
      {"missing_hash_for_relocations", missing_hash_for_relocations},
      {"relative_count_exceeds_table", relative_count_exceeds_table},
      {"invalid_relative_prefix", invalid_relative_prefix},
      {"known_relocation_types", known_relocation_types},
      {"unsupported_plt_relocations", unsupported_plt_relocations},
      {"unsupported_relr_relocations", unsupported_relr_relocations},
      {"unterminated_dynamic_table", unterminated_dynamic_table},
      {"relocation_names", relocation_names},
  };

  int failures = 0;
  for (const auto &test : tests) {
    TestContext context;
    test.second(&context);
    if (context.failures == 0) {
      std::cout << "PASS: " << test.first << '\n';
    } else {
      std::cerr << "FAIL: " << test.first << " (" << context.failures
                << " checks)\n";
      failures += context.failures;
    }
  }
  if (failures != 0) {
    std::cerr << failures << " load-plan checks failed\n";
    return 1;
  }
  std::cout << tests.size() << " load-plan tests passed\n";
  return 0;
}
