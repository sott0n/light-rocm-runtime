#include "load_plan_reader.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace light_rocr::loader {
namespace {

constexpr size_t kElfProgramHeaderOffset = 32;
constexpr size_t kElfProgramHeaderEntrySizeOffset = 54;
constexpr size_t kElfProgramHeaderCountOffset = 56;

constexpr size_t kProgramHeaderTypeOffset = 0;
constexpr size_t kProgramHeaderFileOffset = 8;
constexpr size_t kProgramHeaderFileSizeOffset = 32;

constexpr uint32_t kProgramHeaderTypeDynamic = 2;
constexpr size_t kElf64DynamicEntrySize = 16;
constexpr size_t kElf64RelSize = 16;
constexpr size_t kElf64RelaSize = 24;

constexpr int64_t kDynamicNull = 0;
constexpr int64_t kDynamicPltRelocationSize = 2;
constexpr int64_t kDynamicHash = 4;
constexpr int64_t kDynamicRela = 7;
constexpr int64_t kDynamicRelaSize = 8;
constexpr int64_t kDynamicRelaEntrySize = 9;
constexpr int64_t kDynamicRel = 17;
constexpr int64_t kDynamicRelSize = 18;
constexpr int64_t kDynamicRelEntrySize = 19;
constexpr int64_t kDynamicPltRelocationType = 20;
constexpr int64_t kDynamicJumpRelocation = 23;
constexpr int64_t kDynamicRelrSize = 35;
constexpr int64_t kDynamicRelr = 36;
constexpr int64_t kDynamicRelrEntrySize = 37;
constexpr int64_t kDynamicRelaCount = 0x6ffffff9;
constexpr int64_t kDynamicRelCount = 0x6ffffffa;

ParseError failure(ParseErrorCode code, uint64_t offset, std::string message) {
  return ParseError{code, offset, std::move(message)};
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

int64_t read_i64(const uint8_t *data) {
  const uint64_t bits = read_u64(data);
  int64_t value = 0;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool virtual_address_to_file(const CodeObject &code_object, uint64_t address,
                             uint64_t length, uint64_t *file_offset) {
  for (const LoadSegment &segment : code_object.load_segments) {
    if (address < segment.virtual_address) {
      continue;
    }
    const uint64_t delta = address - segment.virtual_address;
    if (delta <= segment.file_size && length <= segment.file_size - delta) {
      *file_offset = segment.file_offset + delta;
      return true;
    }
  }
  return false;
}

bool virtual_address_in_memory(const CodeObject &code_object, uint64_t address,
                               uint64_t length) {
  for (const LoadSegment &segment : code_object.load_segments) {
    if (address < segment.virtual_address) {
      continue;
    }
    const uint64_t delta = address - segment.virtual_address;
    if (delta <= segment.memory_size && length <= segment.memory_size - delta) {
      return true;
    }
  }
  return false;
}

struct DynamicValue {
  bool present = false;
  uint64_t value = 0;
  uint64_t file_offset = 0;
};

struct RelocationTableInfo {
  DynamicValue address;
  DynamicValue size;
  DynamicValue entry_size;
  DynamicValue relative_count;
};

struct DynamicRelocationInfo {
  DynamicValue hash;
  RelocationTableInfo rel;
  RelocationTableInfo rela;
};

ParseError set_dynamic_value(DynamicValue *destination, uint64_t value,
                             uint64_t file_offset, const char *tag_name) {
  if (destination->present) {
    return failure(ParseErrorCode::InvalidDynamicTable, file_offset,
                   std::string("dynamic table contains duplicate ") + tag_name);
  }
  destination->present = true;
  destination->value = value;
  destination->file_offset = file_offset;
  return {};
}

ParseError read_dynamic_relocation_info(const uint8_t *data, size_t size,
                                        DynamicRelocationInfo *info) {
  const uint64_t program_header_offset =
      read_u64(data + kElfProgramHeaderOffset);
  const uint16_t entry_size = read_u16(data + kElfProgramHeaderEntrySizeOffset);
  const uint16_t count = read_u16(data + kElfProgramHeaderCountOffset);
  uint64_t dynamic_offset = 0;
  uint64_t dynamic_size = 0;
  bool found_dynamic = false;

  for (uint16_t index = 0; index < count; ++index) {
    const uint64_t header_offset =
        program_header_offset + static_cast<uint64_t>(index) * entry_size;
    const uint8_t *header = data + static_cast<size_t>(header_offset);
    if (read_u32(header + kProgramHeaderTypeOffset) !=
        kProgramHeaderTypeDynamic) {
      continue;
    }
    if (found_dynamic) {
      return failure(ParseErrorCode::DuplicateDynamicSegment, header_offset,
                     "code object has more than one PT_DYNAMIC segment");
    }
    found_dynamic = true;
    dynamic_offset = read_u64(header + kProgramHeaderFileOffset);
    dynamic_size = read_u64(header + kProgramHeaderFileSizeOffset);
    if (!range_is_in_file(dynamic_offset, dynamic_size, size) ||
        dynamic_size == 0 || dynamic_size % kElf64DynamicEntrySize != 0) {
      return failure(ParseErrorCode::InvalidDynamicTable,
                     header_offset + kProgramHeaderFileOffset,
                     "PT_DYNAMIC does not contain complete ELF64 entries");
    }
  }
  if (!found_dynamic) {
    return {};
  }

  bool found_null = false;
  for (uint64_t offset = 0; offset < dynamic_size;
       offset += kElf64DynamicEntrySize) {
    const uint64_t file_offset = dynamic_offset + offset;
    const uint8_t *entry = data + static_cast<size_t>(file_offset);
    const int64_t tag = read_i64(entry);
    const uint64_t value = read_u64(entry + 8);
    if (tag == kDynamicNull) {
      found_null = true;
      break;
    }

    ParseError error;
    switch (tag) {
    case kDynamicHash:
      error = set_dynamic_value(&info->hash, value, file_offset + 8, "DT_HASH");
      break;
    case kDynamicRel:
      error = set_dynamic_value(&info->rel.address, value, file_offset + 8,
                                "DT_REL");
      break;
    case kDynamicRelSize:
      error = set_dynamic_value(&info->rel.size, value, file_offset + 8,
                                "DT_RELSZ");
      break;
    case kDynamicRelEntrySize:
      error = set_dynamic_value(&info->rel.entry_size, value, file_offset + 8,
                                "DT_RELENT");
      break;
    case kDynamicRelCount:
      error = set_dynamic_value(&info->rel.relative_count, value,
                                file_offset + 8, "DT_RELCOUNT");
      break;
    case kDynamicRela:
      error = set_dynamic_value(&info->rela.address, value, file_offset + 8,
                                "DT_RELA");
      break;
    case kDynamicRelaSize:
      error = set_dynamic_value(&info->rela.size, value, file_offset + 8,
                                "DT_RELASZ");
      break;
    case kDynamicRelaEntrySize:
      error = set_dynamic_value(&info->rela.entry_size, value, file_offset + 8,
                                "DT_RELAENT");
      break;
    case kDynamicRelaCount:
      error = set_dynamic_value(&info->rela.relative_count, value,
                                file_offset + 8, "DT_RELACOUNT");
      break;
    case kDynamicPltRelocationSize:
    case kDynamicPltRelocationType:
    case kDynamicJumpRelocation:
      return failure(ParseErrorCode::UnsupportedRelocationFormat, file_offset,
                     "PLT relocations are outside the observed HSACO subset");
    case kDynamicRelr:
    case kDynamicRelrSize:
    case kDynamicRelrEntrySize:
      return failure(ParseErrorCode::UnsupportedRelocationFormat, file_offset,
                     "RELR relocations are outside the observed HSACO subset");
    default:
      break;
    }
    if (error.code != ParseErrorCode::None) {
      return error;
    }
  }
  if (!found_null) {
    return failure(ParseErrorCode::InvalidDynamicTable, dynamic_offset,
                   "dynamic table has no DT_NULL terminator");
  }
  return {};
}

bool table_is_present(const RelocationTableInfo &table) {
  return table.address.present || table.size.present ||
         table.entry_size.present;
}

ParseError validate_relocation_table_info(const RelocationTableInfo &table,
                                          size_t expected_entry_size,
                                          const char *table_name) {
  if (!table_is_present(table)) {
    if (table.relative_count.present && table.relative_count.value != 0) {
      return failure(ParseErrorCode::InvalidRelocationTable,
                     table.relative_count.file_offset,
                     std::string(table_name) +
                         " count is present without a relocation table");
    }
    return {};
  }
  if (!table.address.present || !table.size.present ||
      !table.entry_size.present) {
    const uint64_t offset =
        table.address.present
            ? table.address.file_offset
            : (table.size.present ? table.size.file_offset
                                  : table.entry_size.file_offset);
    return failure(ParseErrorCode::InvalidRelocationTable, offset,
                   std::string(table_name) +
                       " requires address, size, and entry-size tags");
  }
  if (table.entry_size.value != expected_entry_size) {
    return failure(
        ParseErrorCode::InvalidRelocationTable, table.entry_size.file_offset,
        std::string(table_name) + " entry size is not the ELF64 size");
  }
  if (table.size.value % expected_entry_size != 0) {
    return failure(ParseErrorCode::InvalidRelocationTable,
                   table.size.file_offset,
                   std::string(table_name) +
                       " byte size is not a whole number of entries");
  }
  const uint64_t entry_count = table.size.value / expected_entry_size;
  if (table.relative_count.present &&
      table.relative_count.value > entry_count) {
    return failure(ParseErrorCode::InvalidRelocationTable,
                   table.relative_count.file_offset,
                   std::string(table_name) +
                       " relative relocation count exceeds its entry count");
  }
  return {};
}

ParseError
validate_disjoint_relocation_tables(const RelocationTableInfo &rel,
                                    const RelocationTableInfo &rela) {
  if (!table_is_present(rel) || !table_is_present(rela) ||
      rel.size.value == 0 || rela.size.value == 0) {
    return {};
  }
  uint64_t rel_end = 0;
  uint64_t rela_end = 0;
  if (!checked_add(rel.address.value, rel.size.value, &rel_end) ||
      !checked_add(rela.address.value, rela.size.value, &rela_end)) {
    return failure(ParseErrorCode::InvalidRelocationTable,
                   rela.address.file_offset,
                   "relocation table virtual range overflows uint64");
  }
  if (rel.address.value < rela_end && rela.address.value < rel_end) {
    return failure(ParseErrorCode::InvalidRelocationTable,
                   rela.address.file_offset,
                   "DT_REL and DT_RELA tables overlap");
  }
  return {};
}

bool is_known_amdgpu_relocation(uint32_t type) {
  switch (type) {
  case 0:
  case 1:
  case 2:
  case 3:
  case 4:
  case 5:
  case 6:
  case 7:
  case 8:
  case 9:
  case 10:
  case 11:
  case 13:
  case 14:
    return true;
  default:
    return false;
  }
}

uint64_t relocation_width(uint32_t type) {
  switch (type) {
  case 0:
    return 0;
  case 14:
    return 2;
  case 1:
  case 2:
  case 4:
  case 6:
  case 7:
  case 8:
  case 9:
  case 10:
  case 11:
    return 4;
  case 3:
  case 5:
  case 13:
    return 8;
  default:
    return 0;
  }
}

ParseError read_symbol_count(const uint8_t *data, const CodeObject &code_object,
                             const DynamicValue &hash, uint32_t *count) {
  if (!hash.present) {
    return failure(ParseErrorCode::InvalidRelocationTable, 0,
                   "relocations require DT_HASH to bound symbol indices");
  }
  uint64_t hash_offset = 0;
  if (!virtual_address_to_file(code_object, hash.value, 8, &hash_offset)) {
    return failure(ParseErrorCode::InvalidRelocationTable, hash.file_offset,
                   "DT_HASH is not backed by a PT_LOAD file range");
  }
  const uint32_t bucket_count =
      read_u32(data + static_cast<size_t>(hash_offset));
  const uint32_t symbol_count =
      read_u32(data + static_cast<size_t>(hash_offset + 4));
  if (bucket_count == 0) {
    return failure(ParseErrorCode::InvalidRelocationTable, hash.file_offset,
                   "DT_HASH has no symbol buckets");
  }
  const uint64_t word_count =
      static_cast<uint64_t>(bucket_count) + symbol_count;
  if (word_count >
      (std::numeric_limits<uint64_t>::max() - 8) / sizeof(uint32_t)) {
    return failure(ParseErrorCode::InvalidRelocationTable, hash.file_offset,
                   "DT_HASH size overflows uint64");
  }
  const uint64_t hash_size = 8 + word_count * sizeof(uint32_t);
  if (!virtual_address_to_file(code_object, hash.value, hash_size,
                               &hash_offset)) {
    return failure(ParseErrorCode::InvalidRelocationTable, hash.file_offset,
                   "DT_HASH extends outside PT_LOAD file ranges");
  }
  *count = symbol_count;
  return {};
}

ParseError read_relocation_table(const uint8_t *data,
                                 const CodeObject &code_object,
                                 const RelocationTableInfo &table,
                                 RelocationEncoding encoding,
                                 uint32_t symbol_count,
                                 std::vector<Relocation> *relocations) {
  if (!table_is_present(table) || table.size.value == 0) {
    return {};
  }
  const uint64_t entry_size =
      encoding == RelocationEncoding::Rela ? kElf64RelaSize : kElf64RelSize;
  uint64_t table_offset = 0;
  if (!virtual_address_to_file(code_object, table.address.value,
                               table.size.value, &table_offset)) {
    return failure(ParseErrorCode::InvalidRelocationTable,
                   table.address.file_offset,
                   "relocation table is not backed by a PT_LOAD file range");
  }

  const uint64_t entry_count = table.size.value / entry_size;
  for (uint64_t index = 0; index < entry_count; ++index) {
    const uint64_t entry_offset = table_offset + index * entry_size;
    const uint8_t *entry = data + static_cast<size_t>(entry_offset);
    Relocation relocation;
    relocation.encoding = encoding;
    relocation.target_virtual_address = read_u64(entry);
    const uint64_t info = read_u64(entry + 8);
    relocation.type = static_cast<uint32_t>(info & 0xffffffffU);
    relocation.symbol_index = static_cast<uint32_t>(info >> 32U);
    if (encoding == RelocationEncoding::Rela) {
      relocation.addend = read_i64(entry + 16);
    }

    if (!is_known_amdgpu_relocation(relocation.type)) {
      return failure(ParseErrorCode::UnsupportedRelocationType,
                     entry_offset + 8,
                     "relocation uses an unknown AMDGPU type " +
                         std::to_string(relocation.type));
    }
    if (relocation.symbol_index >= symbol_count) {
      return failure(ParseErrorCode::RelocationSymbolOutOfBounds,
                     entry_offset + 8,
                     "relocation symbol index is outside DT_HASH bounds");
    }
    if (index < table.relative_count.value &&
        (relocation.type != 13 || relocation.symbol_index != 0)) {
      return failure(ParseErrorCode::InvalidRelocationTable, entry_offset + 8,
                     "relative relocation prefix contains a non-relative "
                     "entry");
    }
    const uint64_t width = relocation_width(relocation.type);
    if (width != 0 &&
        !virtual_address_in_memory(code_object,
                                   relocation.target_virtual_address, width)) {
      return failure(ParseErrorCode::RelocationTargetOutOfBounds, entry_offset,
                     "relocation target is outside PT_LOAD memory ranges");
    }
    relocations->push_back(relocation);
  }
  return {};
}

ParseError append_relocations(const uint8_t *data, size_t size,
                              const CodeObject &code_object,
                              LoadPlan *load_plan) {
  DynamicRelocationInfo dynamic;
  ParseError error = read_dynamic_relocation_info(data, size, &dynamic);
  if (error.code != ParseErrorCode::None) {
    return error;
  }
  error = validate_relocation_table_info(dynamic.rel, kElf64RelSize, "DT_REL");
  if (error.code != ParseErrorCode::None) {
    return error;
  }
  error =
      validate_relocation_table_info(dynamic.rela, kElf64RelaSize, "DT_RELA");
  if (error.code != ParseErrorCode::None) {
    return error;
  }
  error = validate_disjoint_relocation_tables(dynamic.rel, dynamic.rela);
  if (error.code != ParseErrorCode::None) {
    return error;
  }
  const bool has_relocations =
      (table_is_present(dynamic.rel) && dynamic.rel.size.value != 0) ||
      (table_is_present(dynamic.rela) && dynamic.rela.size.value != 0);
  if (!has_relocations) {
    return {};
  }

  uint32_t symbol_count = 0;
  error = read_symbol_count(data, code_object, dynamic.hash, &symbol_count);
  if (error.code != ParseErrorCode::None) {
    return error;
  }
  error = read_relocation_table(data, code_object, dynamic.rel,
                                RelocationEncoding::Rel, symbol_count,
                                &load_plan->relocations);
  if (error.code != ParseErrorCode::None) {
    return error;
  }
  error = read_relocation_table(data, code_object, dynamic.rela,
                                RelocationEncoding::Rela, symbol_count,
                                &load_plan->relocations);
  if (error.code != ParseErrorCode::None) {
    return error;
  }
  std::sort(load_plan->relocations.begin(), load_plan->relocations.end(),
            [](const Relocation &lhs, const Relocation &rhs) {
              if (lhs.target_virtual_address != rhs.target_virtual_address) {
                return lhs.target_virtual_address < rhs.target_virtual_address;
              }
              if (lhs.encoding != rhs.encoding) {
                return lhs.encoding < rhs.encoding;
              }
              if (lhs.type != rhs.type) {
                return lhs.type < rhs.type;
              }
              if (lhs.symbol_index != rhs.symbol_index) {
                return lhs.symbol_index < rhs.symbol_index;
              }
              return lhs.addend < rhs.addend;
            });
  return {};
}

} // namespace

const char *amdgpu_relocation_type_name(uint32_t type) {
  switch (type) {
  case 0:
    return "R_AMDGPU_NONE";
  case 1:
    return "R_AMDGPU_ABS32_LO";
  case 2:
    return "R_AMDGPU_ABS32_HI";
  case 3:
    return "R_AMDGPU_ABS64";
  case 4:
    return "R_AMDGPU_REL32";
  case 5:
    return "R_AMDGPU_REL64";
  case 6:
    return "R_AMDGPU_ABS32";
  case 7:
    return "R_AMDGPU_GOTPCREL";
  case 8:
    return "R_AMDGPU_GOTPCREL32_LO";
  case 9:
    return "R_AMDGPU_GOTPCREL32_HI";
  case 10:
    return "R_AMDGPU_REL32_LO";
  case 11:
    return "R_AMDGPU_REL32_HI";
  case 13:
    return "R_AMDGPU_RELATIVE64";
  case 14:
    return "R_AMDGPU_REL16";
  default:
    return "R_AMDGPU_UNKNOWN";
  }
}

namespace internal {

ParseError build_load_plan(const uint8_t *data, size_t size,
                           CodeObject *code_object) {
  LoadPlan plan;
  uint64_t image_end = 0;
  plan.image_virtual_address = std::numeric_limits<uint64_t>::max();

  for (const LoadSegment &segment : code_object->load_segments) {
    if (segment.memory_size == 0) {
      continue;
    }
    plan.image_virtual_address =
        std::min(plan.image_virtual_address, segment.virtual_address);
    uint64_t segment_end = 0;
    if (!checked_add(segment.virtual_address, segment.memory_size,
                     &segment_end)) {
      return failure(ParseErrorCode::SegmentVirtualRangeOverflow,
                     segment.virtual_address,
                     "load-plan segment range overflows uint64");
    }
    image_end = std::max(image_end, segment_end);
    plan.alignment = std::max(plan.alignment, segment.alignment);

    if (segment.file_size != 0) {
      plan.copies.push_back(LoadCopy{
          segment.file_offset, segment.virtual_address, segment.file_size});
    }
    if (segment.memory_size > segment.file_size) {
      uint64_t zero_address = 0;
      if (!checked_add(segment.virtual_address, segment.file_size,
                       &zero_address)) {
        return failure(ParseErrorCode::SegmentVirtualRangeOverflow,
                       segment.virtual_address,
                       "load-plan zero-fill address overflows uint64");
      }
      plan.zero_fills.push_back(
          LoadZeroFill{zero_address, segment.memory_size - segment.file_size});
    }
    plan.protections.push_back(LoadProtection{
        segment.virtual_address, segment.memory_size, segment.flags});
  }
  if (plan.image_virtual_address == std::numeric_limits<uint64_t>::max()) {
    plan.image_virtual_address = 0;
  } else {
    plan.image_size = image_end - plan.image_virtual_address;
  }
  std::sort(plan.copies.begin(), plan.copies.end(),
            [](const LoadCopy &lhs, const LoadCopy &rhs) {
              if (lhs.virtual_address != rhs.virtual_address) {
                return lhs.virtual_address < rhs.virtual_address;
              }
              return lhs.file_offset < rhs.file_offset;
            });
  std::sort(plan.zero_fills.begin(), plan.zero_fills.end(),
            [](const LoadZeroFill &lhs, const LoadZeroFill &rhs) {
              return lhs.virtual_address < rhs.virtual_address;
            });
  std::sort(plan.protections.begin(), plan.protections.end(),
            [](const LoadProtection &lhs, const LoadProtection &rhs) {
              return lhs.virtual_address < rhs.virtual_address;
            });

  ParseError error = append_relocations(data, size, *code_object, &plan);
  if (error.code != ParseErrorCode::None) {
    return error;
  }
  code_object->load_plan = std::move(plan);
  return {};
}

} // namespace internal
} // namespace light_rocr::loader
