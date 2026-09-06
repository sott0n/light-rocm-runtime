#include "metadata_reader.hpp"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace light_rocr::loader::internal {
namespace {

constexpr size_t kElfProgramHeaderOffset = 32;
constexpr size_t kElfProgramHeaderEntrySizeOffset = 54;
constexpr size_t kElfProgramHeaderCountOffset = 56;

constexpr size_t kElf64SymbolSize = 24;
constexpr size_t kKernelDescriptorSize = 64;
constexpr size_t kElf64DynamicEntrySize = 16;

constexpr uint32_t kProgramHeaderTypeNote = 4;
constexpr uint32_t kProgramHeaderTypeDynamic = 2;
constexpr uint32_t kAmdgpuMetadataNoteType = 32;
constexpr uint8_t kSymbolTypeObject = 1;
constexpr uint16_t kUndefinedSection = 0;
constexpr uint16_t kKernelPropertyWavefrontSize32 = 1U << 10U;
constexpr uint16_t kKernelPropertyUsesDynamicStack = 1U << 11U;
constexpr uint32_t kMinimumKernargAlignment = 16;
constexpr size_t kMaximumMetadataDepth = 64;

constexpr size_t kProgramHeaderTypeOffset = 0;
constexpr size_t kProgramHeaderFileOffset = 8;
constexpr size_t kProgramHeaderFileSizeOffset = 32;

constexpr size_t kSymbolNameOffset = 0;
constexpr size_t kSymbolInfoOffset = 4;
constexpr size_t kSymbolSectionIndexOffset = 6;
constexpr size_t kSymbolValueOffset = 8;
constexpr size_t kSymbolSizeOffset = 16;

constexpr size_t kDescriptorGroupSegmentSizeOffset = 0;
constexpr size_t kDescriptorPrivateSegmentSizeOffset = 4;
constexpr size_t kDescriptorKernargSizeOffset = 8;
constexpr size_t kDescriptorCodeEntryOffset = 16;
constexpr size_t kDescriptorComputePgmRsrc3Offset = 44;
constexpr size_t kDescriptorComputePgmRsrc1Offset = 48;
constexpr size_t kDescriptorComputePgmRsrc2Offset = 52;
constexpr size_t kDescriptorKernelPropertiesOffset = 56;
constexpr size_t kDescriptorKernargPreloadOffset = 58;

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
  static_assert(sizeof(value) == sizeof(bits), "unexpected int64 size");
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool is_power_of_two(uint32_t value) {
  return value != 0 && (value & (value - 1U)) == 0;
}

bool align4(uint64_t value, uint64_t *aligned) {
  uint64_t with_padding = 0;
  if (!checked_add(value, 3, &with_padding)) {
    return false;
  }
  *aligned = with_padding & ~uint64_t{3};
  return true;
}

struct MetadataBytes {
  const uint8_t *data = nullptr;
  size_t size = 0;
  uint64_t file_offset = 0;
};

ParseError find_metadata_note(const uint8_t *data, size_t size,
                              MetadataBytes *metadata) {
  const uint64_t program_header_offset =
      read_u64(data + kElfProgramHeaderOffset);
  const uint16_t entry_size = read_u16(data + kElfProgramHeaderEntrySizeOffset);
  const uint16_t count = read_u16(data + kElfProgramHeaderCountOffset);

  for (uint16_t index = 0; index < count; ++index) {
    const uint64_t header_offset =
        program_header_offset + static_cast<uint64_t>(index) * entry_size;
    const uint8_t *header = data + static_cast<size_t>(header_offset);
    if (read_u32(header + kProgramHeaderTypeOffset) != kProgramHeaderTypeNote) {
      continue;
    }

    const uint64_t note_offset = read_u64(header + kProgramHeaderFileOffset);
    const uint64_t note_size = read_u64(header + kProgramHeaderFileSizeOffset);
    if (!range_is_in_file(note_offset, note_size, size)) {
      return failure(ParseErrorCode::MalformedNote,
                     header_offset + kProgramHeaderFileOffset,
                     "PT_NOTE file range is outside the code object");
    }

    uint64_t cursor = note_offset;
    const uint64_t end = note_offset + note_size;
    while (cursor < end) {
      if (end - cursor < 12) {
        return failure(ParseErrorCode::MalformedNote, cursor,
                       "note header is truncated");
      }
      const uint32_t name_size = read_u32(data + static_cast<size_t>(cursor));
      const uint32_t descriptor_size =
          read_u32(data + static_cast<size_t>(cursor + 4));
      const uint32_t type = read_u32(data + static_cast<size_t>(cursor + 8));
      const uint64_t name_offset = cursor + 12;
      uint64_t aligned_name_size = 0;
      uint64_t aligned_descriptor_size = 0;
      if (!align4(name_size, &aligned_name_size) ||
          !align4(descriptor_size, &aligned_descriptor_size)) {
        return failure(ParseErrorCode::MalformedNote, cursor,
                       "note size alignment overflows");
      }
      uint64_t descriptor_offset = 0;
      uint64_t next = 0;
      if (!checked_add(name_offset, aligned_name_size, &descriptor_offset) ||
          !checked_add(descriptor_offset, aligned_descriptor_size, &next) ||
          next > end) {
        return failure(ParseErrorCode::MalformedNote, cursor,
                       "note name or descriptor is truncated");
      }

      constexpr char kAmdgpuOwner[] = "AMDGPU";
      const bool is_amdgpu =
          name_size == sizeof(kAmdgpuOwner) &&
          std::memcmp(data + static_cast<size_t>(name_offset), kAmdgpuOwner,
                      sizeof(kAmdgpuOwner)) == 0;
      if (is_amdgpu && type == kAmdgpuMetadataNoteType) {
        if (metadata->data != nullptr) {
          return failure(ParseErrorCode::DuplicateMetadataNote, cursor,
                         "code object has more than one AMDGPU metadata note");
        }
        metadata->data = data + static_cast<size_t>(descriptor_offset);
        metadata->size = descriptor_size;
        metadata->file_offset = descriptor_offset;
      }
      cursor = next;
    }
  }
  return {};
}

struct MessagePackValue {
  enum class Kind { Nil, Boolean, Unsigned, Signed, String, Array, Map };

  Kind kind = Kind::Nil;
  uint64_t offset = 0;
  bool boolean = false;
  uint64_t unsigned_integer = 0;
  int64_t signed_integer = 0;
  std::string string;
  std::vector<MessagePackValue> array;
  std::vector<std::pair<std::string, MessagePackValue>> map;
};

class MessagePackReader {
public:
  explicit MessagePackReader(MetadataBytes bytes) : bytes_(bytes) {}

  ParseError parse(MessagePackValue *value) {
    ParseError error = parse_value(0, value);
    if (error.code != ParseErrorCode::None) {
      return error;
    }
    if (cursor_ != bytes_.size) {
      return fail(ParseErrorCode::InvalidMetadata, cursor_,
                  "metadata has trailing MessagePack values");
    }
    return {};
  }

private:
  ParseError fail(ParseErrorCode code, size_t relative_offset,
                  std::string message) const {
    return failure(code, bytes_.file_offset + relative_offset,
                   std::move(message));
  }

  bool take(size_t count, const uint8_t **start) {
    if (count > bytes_.size - cursor_) {
      return false;
    }
    *start = bytes_.data + cursor_;
    cursor_ += count;
    return true;
  }

  bool read_big_endian(size_t count, uint64_t *value) {
    const uint8_t *start = nullptr;
    if (!take(count, &start)) {
      return false;
    }
    uint64_t result = 0;
    for (size_t index = 0; index < count; ++index) {
      result = (result << 8U) | start[index];
    }
    *value = result;
    return true;
  }

  ParseError read_length(size_t width, size_t marker_offset, size_t *length) {
    uint64_t value = 0;
    if (!read_big_endian(width, &value)) {
      return fail(ParseErrorCode::InvalidMetadata, marker_offset,
                  "MessagePack length is truncated");
    }
    if (value > std::numeric_limits<size_t>::max()) {
      return fail(ParseErrorCode::InvalidMetadata, marker_offset,
                  "MessagePack length exceeds host size_t");
    }
    *length = static_cast<size_t>(value);
    return {};
  }

  ParseError parse_string(size_t length, size_t marker_offset,
                          MessagePackValue *value) {
    const uint8_t *start = nullptr;
    if (!take(length, &start)) {
      return fail(ParseErrorCode::InvalidMetadata, marker_offset,
                  "MessagePack string is truncated");
    }
    value->kind = MessagePackValue::Kind::String;
    value->string.assign(reinterpret_cast<const char *>(start), length);
    return {};
  }

  ParseError parse_array(size_t count, size_t depth, size_t marker_offset,
                         MessagePackValue *value) {
    if (count > bytes_.size - cursor_) {
      return fail(ParseErrorCode::InvalidMetadata, marker_offset,
                  "MessagePack array count exceeds remaining bytes");
    }
    value->kind = MessagePackValue::Kind::Array;
    value->array.reserve(count);
    for (size_t index = 0; index < count; ++index) {
      value->array.emplace_back();
      ParseError error = parse_value(depth + 1, &value->array.back());
      if (error.code != ParseErrorCode::None) {
        return error;
      }
    }
    return {};
  }

  ParseError parse_map(size_t count, size_t depth, size_t marker_offset,
                       MessagePackValue *value) {
    if (count > (bytes_.size - cursor_) / 2) {
      return fail(ParseErrorCode::InvalidMetadata, marker_offset,
                  "MessagePack map count exceeds remaining bytes");
    }
    value->kind = MessagePackValue::Kind::Map;
    value->map.reserve(count);
    std::set<std::string> keys;
    for (size_t index = 0; index < count; ++index) {
      MessagePackValue key;
      ParseError error = parse_value(depth + 1, &key);
      if (error.code != ParseErrorCode::None) {
        return error;
      }
      if (key.kind != MessagePackValue::Kind::String) {
        return fail(ParseErrorCode::InvalidMetadata,
                    static_cast<size_t>(key.offset - bytes_.file_offset),
                    "AMDHSA metadata map key is not a string");
      }
      if (!keys.insert(key.string).second) {
        return fail(ParseErrorCode::InvalidMetadata,
                    static_cast<size_t>(key.offset - bytes_.file_offset),
                    "AMDHSA metadata map contains a duplicate key");
      }
      MessagePackValue mapped;
      error = parse_value(depth + 1, &mapped);
      if (error.code != ParseErrorCode::None) {
        return error;
      }
      value->map.emplace_back(std::move(key.string), std::move(mapped));
    }
    return {};
  }

  ParseError parse_value(size_t depth, MessagePackValue *value) {
    if (depth > kMaximumMetadataDepth) {
      return fail(ParseErrorCode::InvalidMetadata, cursor_,
                  "MessagePack nesting exceeds the loader limit");
    }
    const size_t marker_offset = cursor_;
    const uint8_t *marker_pointer = nullptr;
    if (!take(1, &marker_pointer)) {
      return fail(ParseErrorCode::InvalidMetadata, marker_offset,
                  "MessagePack value is truncated");
    }
    const uint8_t marker = *marker_pointer;
    value->offset = bytes_.file_offset + marker_offset;

    if (marker <= 0x7f) {
      value->kind = MessagePackValue::Kind::Unsigned;
      value->unsigned_integer = marker;
      return {};
    }
    if (marker >= 0xe0) {
      value->kind = MessagePackValue::Kind::Signed;
      value->signed_integer = static_cast<int8_t>(marker);
      return {};
    }
    if ((marker & 0xe0U) == 0xa0U) {
      return parse_string(marker & 0x1fU, marker_offset, value);
    }
    if ((marker & 0xf0U) == 0x90U) {
      return parse_array(marker & 0x0fU, depth, marker_offset, value);
    }
    if ((marker & 0xf0U) == 0x80U) {
      return parse_map(marker & 0x0fU, depth, marker_offset, value);
    }

    if (marker == 0xc0) {
      value->kind = MessagePackValue::Kind::Nil;
      return {};
    }
    if (marker == 0xc2 || marker == 0xc3) {
      value->kind = MessagePackValue::Kind::Boolean;
      value->boolean = marker == 0xc3;
      return {};
    }

    size_t length = 0;
    ParseError error;
    switch (marker) {
    case 0xcc:
    case 0xcd:
    case 0xce:
    case 0xcf: {
      const size_t width = size_t{1} << static_cast<size_t>(marker - 0xccU);
      uint64_t integer = 0;
      if (!read_big_endian(width, &integer)) {
        return fail(ParseErrorCode::InvalidMetadata, marker_offset,
                    "MessagePack unsigned integer is truncated");
      }
      value->kind = MessagePackValue::Kind::Unsigned;
      value->unsigned_integer = integer;
      return {};
    }
    case 0xd0:
    case 0xd1:
    case 0xd2:
    case 0xd3: {
      const size_t width = size_t{1} << static_cast<size_t>(marker - 0xd0U);
      uint64_t bits = 0;
      if (!read_big_endian(width, &bits)) {
        return fail(ParseErrorCode::InvalidMetadata, marker_offset,
                    "MessagePack signed integer is truncated");
      }
      if (width < sizeof(bits) &&
          (bits & (uint64_t{1} << (width * 8U - 1U))) != 0) {
        bits |= std::numeric_limits<uint64_t>::max() << (width * 8U);
      }
      value->kind = MessagePackValue::Kind::Signed;
      std::memcpy(&value->signed_integer, &bits, sizeof(bits));
      return {};
    }
    case 0xd9:
    case 0xda:
    case 0xdb:
      error = read_length(size_t{1} << static_cast<size_t>(marker - 0xd9U),
                          marker_offset, &length);
      if (error.code != ParseErrorCode::None) {
        return error;
      }
      return parse_string(length, marker_offset, value);
    case 0xdc:
    case 0xdd:
      error = read_length(marker == 0xdc ? 2 : 4, marker_offset, &length);
      if (error.code != ParseErrorCode::None) {
        return error;
      }
      return parse_array(length, depth, marker_offset, value);
    case 0xde:
    case 0xdf:
      error = read_length(marker == 0xde ? 2 : 4, marker_offset, &length);
      if (error.code != ParseErrorCode::None) {
        return error;
      }
      return parse_map(length, depth, marker_offset, value);
    default:
      return fail(ParseErrorCode::UnsupportedMetadataEncoding, marker_offset,
                  "AMDHSA metadata uses an unsupported MessagePack type");
    }
  }

  MetadataBytes bytes_;
  size_t cursor_ = 0;
};

const MessagePackValue *find_field(const MessagePackValue &map,
                                   const std::string &name) {
  if (map.kind != MessagePackValue::Kind::Map) {
    return nullptr;
  }
  const auto found =
      std::find_if(map.map.begin(), map.map.end(),
                   [&name](const auto &entry) { return entry.first == name; });
  return found == map.map.end() ? nullptr : &found->second;
}

ParseError required_field(const MessagePackValue &map, const char *name,
                          MessagePackValue::Kind kind,
                          const MessagePackValue **value) {
  *value = find_field(map, name);
  if (*value == nullptr) {
    return failure(ParseErrorCode::InvalidMetadata, map.offset,
                   std::string("AMDHSA metadata is missing ") + name);
  }
  if ((*value)->kind != kind) {
    return failure(ParseErrorCode::InvalidMetadata, (*value)->offset,
                   std::string("AMDHSA metadata field has the wrong type: ") +
                       name);
  }
  return {};
}

ParseError metadata_uint32(const MessagePackValue &map, const char *name,
                           uint32_t *result) {
  const MessagePackValue *value = find_field(map, name);
  if (value == nullptr) {
    return failure(ParseErrorCode::InvalidMetadata, map.offset,
                   std::string("AMDHSA metadata is missing ") + name);
  }
  uint64_t integer = 0;
  if (value->kind == MessagePackValue::Kind::Unsigned) {
    integer = value->unsigned_integer;
  } else if (value->kind == MessagePackValue::Kind::Signed &&
             value->signed_integer >= 0) {
    integer = static_cast<uint64_t>(value->signed_integer);
  } else {
    return failure(ParseErrorCode::InvalidMetadata, value->offset,
                   std::string("AMDHSA metadata field is not a nonnegative ") +
                       "integer: " + name);
  }
  if (integer > std::numeric_limits<uint32_t>::max()) {
    return failure(ParseErrorCode::InvalidMetadata, value->offset,
                   std::string("AMDHSA metadata field exceeds uint32: ") +
                       name);
  }
  *result = static_cast<uint32_t>(integer);
  return {};
}

struct MetadataKernel {
  KernelInfo info;
  uint64_t metadata_offset = 0;
};

ParseError decode_metadata_document(const MessagePackValue &root,
                                    MetadataVersion *metadata_version,
                                    std::string *target_isa,
                                    std::vector<MetadataKernel> *kernels) {
  if (root.kind != MessagePackValue::Kind::Map) {
    return failure(ParseErrorCode::InvalidMetadata, root.offset,
                   "AMDHSA metadata root is not a map");
  }

  const MessagePackValue *version = nullptr;
  ParseError error = required_field(root, "amdhsa.version",
                                    MessagePackValue::Kind::Array, &version);
  if (error.code != ParseErrorCode::None) {
    return error;
  }
  if (version->array.size() != 2) {
    return failure(ParseErrorCode::InvalidMetadata, version->offset,
                   "amdhsa.version must contain major and minor values");
  }
  const auto version_integer = [](const MessagePackValue &value,
                                  uint32_t *result) {
    uint64_t integer = std::numeric_limits<uint64_t>::max();
    if (value.kind == MessagePackValue::Kind::Unsigned) {
      integer = value.unsigned_integer;
    } else if (value.kind == MessagePackValue::Kind::Signed &&
               value.signed_integer >= 0) {
      integer = static_cast<uint64_t>(value.signed_integer);
    }
    if (integer > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    *result = static_cast<uint32_t>(integer);
    return true;
  };
  if (!version_integer(version->array[0], &metadata_version->major) ||
      !version_integer(version->array[1], &metadata_version->minor)) {
    return failure(ParseErrorCode::InvalidMetadata, version->offset,
                   "amdhsa.version values must be uint32 integers");
  }
  if (metadata_version->major != 1 || metadata_version->minor != 2) {
    return failure(ParseErrorCode::UnsupportedMetadataVersion, version->offset,
                   "initial light-rocr loader supports AMDHSA metadata 1.2");
  }

  const MessagePackValue *target = nullptr;
  error = required_field(root, "amdhsa.target", MessagePackValue::Kind::String,
                         &target);
  if (error.code != ParseErrorCode::None) {
    return error;
  }
  *target_isa = target->string;
  if (target->string != "amdgcn-amd-amdhsa--gfx1101" &&
      target->string != "amdgcn-amd-amdhsa-unknown-gfx1101") {
    return failure(ParseErrorCode::MetadataTargetMismatch, target->offset,
                   "AMDHSA metadata target does not match gfx1101");
  }

  const MessagePackValue *kernel_array = nullptr;
  error = required_field(root, "amdhsa.kernels", MessagePackValue::Kind::Array,
                         &kernel_array);
  if (error.code != ParseErrorCode::None) {
    return error;
  }
  if (kernel_array->array.empty()) {
    return failure(ParseErrorCode::InvalidMetadata, kernel_array->offset,
                   "amdhsa.kernels is empty");
  }

  kernels->reserve(kernel_array->array.size());
  std::set<std::string> kernel_names;
  std::set<std::string> symbol_names;
  for (const MessagePackValue &kernel_value : kernel_array->array) {
    if (kernel_value.kind != MessagePackValue::Kind::Map) {
      return failure(ParseErrorCode::InvalidMetadata, kernel_value.offset,
                     "amdhsa.kernels entry is not a map");
    }
    MetadataKernel kernel;
    kernel.metadata_offset = kernel_value.offset;
    const MessagePackValue *field = nullptr;
    error = required_field(kernel_value, ".name",
                           MessagePackValue::Kind::String, &field);
    if (error.code != ParseErrorCode::None) {
      return error;
    }
    kernel.info.name = field->string;
    error = required_field(kernel_value, ".symbol",
                           MessagePackValue::Kind::String, &field);
    if (error.code != ParseErrorCode::None) {
      return error;
    }
    kernel.info.symbol_name = field->string;
    if (kernel.info.name.empty() || kernel.info.symbol_name.empty()) {
      return failure(ParseErrorCode::InvalidMetadata, field->offset,
                     "kernel name and symbol must not be empty");
    }
    constexpr char kKernelDescriptorSuffix[] = ".kd";
    if (kernel.info.symbol_name.size() < sizeof(kKernelDescriptorSuffix) - 1 ||
        kernel.info.symbol_name.compare(
            kernel.info.symbol_name.size() -
                (sizeof(kKernelDescriptorSuffix) - 1),
            sizeof(kKernelDescriptorSuffix) - 1,
            kKernelDescriptorSuffix) != 0) {
      return failure(ParseErrorCode::InvalidMetadata, field->offset,
                     "kernel descriptor symbol does not end in .kd");
    }

    const struct IntegerField {
      const char *name;
      uint32_t *destination;
    } integer_fields[] = {
        {".kernarg_segment_size", &kernel.info.kernarg_size},
        {".kernarg_segment_align", &kernel.info.metadata_kernarg_alignment},
        {".group_segment_fixed_size", &kernel.info.group_segment_size},
        {".private_segment_fixed_size", &kernel.info.private_segment_size},
        {".wavefront_size", &kernel.info.wavefront_size},
    };
    for (const IntegerField &integer_field : integer_fields) {
      error = metadata_uint32(kernel_value, integer_field.name,
                              integer_field.destination);
      if (error.code != ParseErrorCode::None) {
        return error;
      }
    }
    if (!is_power_of_two(kernel.info.metadata_kernarg_alignment)) {
      const MessagePackValue *alignment =
          find_field(kernel_value, ".kernarg_segment_align");
      return failure(ParseErrorCode::InvalidMetadata, alignment->offset,
                     "kernel argument alignment is not a power of two");
    }
    kernel.info.kernarg_alignment = std::max(
        kernel.info.metadata_kernarg_alignment, kMinimumKernargAlignment);
    if (kernel.info.wavefront_size != 32 && kernel.info.wavefront_size != 64) {
      const MessagePackValue *wavefront =
          find_field(kernel_value, ".wavefront_size");
      return failure(ParseErrorCode::InvalidMetadata, wavefront->offset,
                     "kernel wavefront size must be 32 or 64");
    }

    error = required_field(kernel_value, ".uses_dynamic_stack",
                           MessagePackValue::Kind::Boolean, &field);
    if (error.code != ParseErrorCode::None) {
      return error;
    }
    kernel.info.uses_dynamic_stack = field->boolean;

    if (!kernel_names.insert(kernel.info.name).second ||
        !symbol_names.insert(kernel.info.symbol_name).second) {
      return failure(ParseErrorCode::InvalidMetadata, kernel_value.offset,
                     "AMDHSA metadata contains a duplicate kernel name or "
                     "symbol");
    }
    kernels->push_back(std::move(kernel));
  }
  return {};
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

struct DynamicValue {
  bool present = false;
  uint64_t value = 0;
  uint64_t file_offset = 0;
};

struct DynamicInfo {
  DynamicValue symbol_table;
  DynamicValue symbol_entry_size;
  DynamicValue string_table;
  DynamicValue string_table_size;
  DynamicValue hash_table;
};

ParseError set_dynamic_value(DynamicValue *destination, uint64_t value,
                             uint64_t file_offset) {
  if (destination->present) {
    return failure(ParseErrorCode::InvalidDynamicTable, file_offset,
                   "dynamic table contains a duplicate required tag");
  }
  destination->present = true;
  destination->value = value;
  destination->file_offset = file_offset;
  return {};
}

ParseError read_dynamic_info(const uint8_t *data, size_t size,
                             DynamicInfo *info) {
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
    return failure(ParseErrorCode::MissingDynamicSegment,
                   kElfProgramHeaderOffset,
                   "kernel metadata requires a PT_DYNAMIC segment");
  }

  constexpr int64_t kDynamicNull = 0;
  constexpr int64_t kDynamicHash = 4;
  constexpr int64_t kDynamicStringTable = 5;
  constexpr int64_t kDynamicSymbolTable = 6;
  constexpr int64_t kDynamicStringTableSize = 10;
  constexpr int64_t kDynamicSymbolEntrySize = 11;
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
      error = set_dynamic_value(&info->hash_table, value, file_offset + 8);
      break;
    case kDynamicStringTable:
      error = set_dynamic_value(&info->string_table, value, file_offset + 8);
      break;
    case kDynamicSymbolTable:
      error = set_dynamic_value(&info->symbol_table, value, file_offset + 8);
      break;
    case kDynamicStringTableSize:
      error =
          set_dynamic_value(&info->string_table_size, value, file_offset + 8);
      break;
    case kDynamicSymbolEntrySize:
      error =
          set_dynamic_value(&info->symbol_entry_size, value, file_offset + 8);
      break;
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
  if (!info->hash_table.present || !info->string_table.present ||
      !info->string_table_size.present || !info->symbol_table.present ||
      !info->symbol_entry_size.present) {
    return failure(ParseErrorCode::MissingDynamicEntry, dynamic_offset,
                   "dynamic table is missing HASH, STRTAB, STRSZ, SYMTAB, or "
                   "SYMENT");
  }
  if (info->symbol_entry_size.value != kElf64SymbolSize) {
    return failure(ParseErrorCode::InvalidSymbolTable,
                   info->symbol_entry_size.file_offset,
                   "DT_SYMENT is not the ELF64 symbol size");
  }
  return {};
}

bool read_string(const uint8_t *data, uint64_t table_offset,
                 uint64_t table_size, uint32_t name_offset, std::string *name) {
  if (name_offset >= table_size) {
    return false;
  }
  const char *begin = reinterpret_cast<const char *>(
      data + static_cast<size_t>(table_offset + name_offset));
  const size_t remaining =
      static_cast<size_t>(table_size - static_cast<uint64_t>(name_offset));
  const void *terminator = std::memchr(begin, '\0', remaining);
  if (terminator == nullptr) {
    return false;
  }
  const auto *end = static_cast<const char *>(terminator);
  name->assign(begin, static_cast<size_t>(end - begin));
  return true;
}

struct Symbol {
  uint64_t value = 0;
  uint64_t size = 0;
  uint16_t section_index = 0;
  uint8_t type = 0;
  uint64_t file_offset = 0;
};

struct NamedSymbol {
  std::string name;
  Symbol symbol;
};

ParseError read_dynamic_symbols(const uint8_t *data, size_t size,
                                const CodeObject &code_object,
                                std::vector<NamedSymbol> *symbols) {
  DynamicInfo dynamic;
  ParseError error = read_dynamic_info(data, size, &dynamic);
  if (error.code != ParseErrorCode::None) {
    return error;
  }

  uint64_t hash_offset = 0;
  if (!virtual_address_to_file(code_object, dynamic.hash_table.value, 8,
                               &hash_offset)) {
    return failure(ParseErrorCode::InvalidDynamicTable,
                   dynamic.hash_table.file_offset,
                   "DT_HASH is not backed by a PT_LOAD file range");
  }
  const uint32_t bucket_count =
      read_u32(data + static_cast<size_t>(hash_offset));
  const uint32_t symbol_count =
      read_u32(data + static_cast<size_t>(hash_offset + 4));
  if (bucket_count == 0) {
    return failure(ParseErrorCode::InvalidDynamicTable,
                   dynamic.hash_table.file_offset,
                   "DT_HASH has no symbol buckets");
  }
  const uint64_t hash_word_count =
      static_cast<uint64_t>(bucket_count) + symbol_count;
  const uint64_t hash_size = 8 + hash_word_count * sizeof(uint32_t);
  if (!virtual_address_to_file(code_object, dynamic.hash_table.value, hash_size,
                               &hash_offset)) {
    return failure(ParseErrorCode::InvalidDynamicTable,
                   dynamic.hash_table.file_offset,
                   "DT_HASH extends outside PT_LOAD file ranges");
  }

  const uint64_t symbol_table_size =
      static_cast<uint64_t>(symbol_count) * kElf64SymbolSize;
  uint64_t symbol_table_offset = 0;
  if (!virtual_address_to_file(code_object, dynamic.symbol_table.value,
                               symbol_table_size, &symbol_table_offset)) {
    return failure(ParseErrorCode::InvalidSymbolTable,
                   dynamic.symbol_table.file_offset,
                   "DT_SYMTAB extends outside PT_LOAD file ranges");
  }
  uint64_t string_table_offset = 0;
  if (dynamic.string_table_size.value == 0 ||
      !virtual_address_to_file(code_object, dynamic.string_table.value,
                               dynamic.string_table_size.value,
                               &string_table_offset)) {
    return failure(ParseErrorCode::InvalidStringTable,
                   dynamic.string_table.file_offset,
                   "DT_STRTAB extends outside PT_LOAD file ranges");
  }

  symbols->reserve(symbol_count);
  for (uint32_t index = 0; index < symbol_count; ++index) {
    const uint64_t entry_offset =
        symbol_table_offset + static_cast<uint64_t>(index) * kElf64SymbolSize;
    const uint8_t *entry = data + static_cast<size_t>(entry_offset);
    NamedSymbol named;
    if (!read_string(data, string_table_offset, dynamic.string_table_size.value,
                     read_u32(entry + kSymbolNameOffset), &named.name)) {
      return failure(ParseErrorCode::InvalidStringTable,
                     entry_offset + kSymbolNameOffset,
                     "dynamic symbol name is outside DT_STRTAB");
    }
    named.symbol.file_offset = entry_offset;
    named.symbol.value = read_u64(entry + kSymbolValueOffset);
    named.symbol.size = read_u64(entry + kSymbolSizeOffset);
    named.symbol.section_index = read_u16(entry + kSymbolSectionIndexOffset);
    named.symbol.type = entry[kSymbolInfoOffset] & 0x0fU;
    symbols->push_back(std::move(named));
  }
  std::sort(symbols->begin(), symbols->end(),
            [](const NamedSymbol &lhs, const NamedSymbol &rhs) {
              if (lhs.name != rhs.name) {
                return lhs.name < rhs.name;
              }
              return lhs.symbol.file_offset < rhs.symbol.file_offset;
            });
  return {};
}

ParseError find_kernel_symbol(const std::vector<NamedSymbol> &symbols,
                              const MetadataKernel &kernel, Symbol *symbol) {
  const auto found = std::lower_bound(
      symbols.begin(), symbols.end(), kernel.info.symbol_name,
      [](const NamedSymbol &candidate, const std::string &name) {
        return candidate.name < name;
      });
  if (found == symbols.end() || found->name != kernel.info.symbol_name) {
    return failure(ParseErrorCode::MissingKernelSymbol, kernel.metadata_offset,
                   "metadata kernel symbol was not found in DT_SYMTAB: " +
                       kernel.info.symbol_name);
  }
  if (std::next(found) != symbols.end() &&
      std::next(found)->name == kernel.info.symbol_name) {
    return failure(ParseErrorCode::InvalidKernelSymbol,
                   std::next(found)->symbol.file_offset,
                   "kernel descriptor symbol is defined more than once");
  }
  *symbol = found->symbol;
  constexpr uint16_t kLowestReservedSectionIndex = 0xff00;
  if (symbol->type != kSymbolTypeObject ||
      symbol->section_index == kUndefinedSection ||
      symbol->section_index >= kLowestReservedSectionIndex ||
      symbol->size != kKernelDescriptorSize) {
    return failure(ParseErrorCode::InvalidKernelSymbol, symbol->file_offset,
                   "kernel descriptor symbol has invalid type, section, or "
                   "size");
  }
  return {};
}

bool address_has_executable_bytes(const CodeObject &code_object,
                                  uint64_t address) {
  constexpr uint32_t kExecute = 1;
  for (const LoadSegment &segment : code_object.load_segments) {
    if ((segment.flags & kExecute) == 0 || address < segment.virtual_address) {
      continue;
    }
    if (address - segment.virtual_address < segment.file_size) {
      return true;
    }
  }
  return false;
}

bool add_signed_offset(uint64_t base, int64_t offset, uint64_t *result) {
  if (offset >= 0) {
    return checked_add(base, static_cast<uint64_t>(offset), result);
  }
  const uint64_t magnitude = uint64_t{0} - static_cast<uint64_t>(offset);
  if (magnitude > base) {
    return false;
  }
  *result = base - magnitude;
  return true;
}

ParseError decode_kernel_descriptor(const uint8_t *data,
                                    const CodeObject &code_object,
                                    const Symbol &symbol,
                                    MetadataKernel *kernel) {
  uint64_t descriptor_offset = 0;
  if (!virtual_address_to_file(code_object, symbol.value, kKernelDescriptorSize,
                               &descriptor_offset)) {
    return failure(ParseErrorCode::KernelDescriptorOutOfBounds,
                   symbol.file_offset,
                   "kernel descriptor is not backed by a PT_LOAD file range");
  }
  const uint8_t *descriptor = data + static_cast<size_t>(descriptor_offset);
  if (symbol.value % kKernelDescriptorSize != 0 ||
      !std::all_of(descriptor + 12, descriptor + 16,
                   [](uint8_t byte) { return byte == 0; }) ||
      !std::all_of(descriptor + 24, descriptor + 44,
                   [](uint8_t byte) { return byte == 0; }) ||
      !std::all_of(descriptor + 60, descriptor + 64,
                   [](uint8_t byte) { return byte == 0; })) {
    return failure(ParseErrorCode::InvalidKernelDescriptor, descriptor_offset,
                   "kernel descriptor alignment or reserved bytes are invalid");
  }
  kernel->info.descriptor_virtual_address = symbol.value;
  kernel->info.code_entry_byte_offset =
      read_i64(descriptor + kDescriptorCodeEntryOffset);
  if (!add_signed_offset(symbol.value, kernel->info.code_entry_byte_offset,
                         &kernel->info.code_entry_virtual_address) ||
      !address_has_executable_bytes(code_object,
                                    kernel->info.code_entry_virtual_address)) {
    return failure(ParseErrorCode::KernelEntryOutOfBounds,
                   descriptor_offset + kDescriptorCodeEntryOffset,
                   "kernel code entry does not point into an executable "
                   "PT_LOAD segment");
  }

  const uint32_t descriptor_group =
      read_u32(descriptor + kDescriptorGroupSegmentSizeOffset);
  const uint32_t descriptor_private =
      read_u32(descriptor + kDescriptorPrivateSegmentSizeOffset);
  const uint32_t descriptor_kernarg =
      read_u32(descriptor + kDescriptorKernargSizeOffset);
  if (descriptor_group != kernel->info.group_segment_size ||
      descriptor_private != kernel->info.private_segment_size ||
      descriptor_kernarg != kernel->info.kernarg_size) {
    return failure(ParseErrorCode::KernelMetadataMismatch, descriptor_offset,
                   "kernel descriptor sizes do not match AMDHSA metadata");
  }

  kernel->info.compute_pgm_rsrc3 =
      read_u32(descriptor + kDescriptorComputePgmRsrc3Offset);
  kernel->info.compute_pgm_rsrc1 =
      read_u32(descriptor + kDescriptorComputePgmRsrc1Offset);
  kernel->info.compute_pgm_rsrc2 =
      read_u32(descriptor + kDescriptorComputePgmRsrc2Offset);
  kernel->info.kernel_code_properties =
      read_u16(descriptor + kDescriptorKernelPropertiesOffset);
  kernel->info.kernarg_preload =
      read_u16(descriptor + kDescriptorKernargPreloadOffset);
  constexpr uint16_t kKnownKernelProperties = 0x0c7f;
  if ((kernel->info.kernel_code_properties & ~kKnownKernelProperties) != 0) {
    return failure(ParseErrorCode::InvalidKernelDescriptor,
                   descriptor_offset + kDescriptorKernelPropertiesOffset,
                   "kernel descriptor uses reserved property bits");
  }

  const uint32_t descriptor_wavefront = (kernel->info.kernel_code_properties &
                                         kKernelPropertyWavefrontSize32) != 0
                                            ? 32
                                            : 64;
  const bool descriptor_dynamic_stack = (kernel->info.kernel_code_properties &
                                         kKernelPropertyUsesDynamicStack) != 0;
  if (descriptor_wavefront != kernel->info.wavefront_size ||
      descriptor_dynamic_stack != kernel->info.uses_dynamic_stack) {
    return failure(ParseErrorCode::KernelMetadataMismatch,
                   descriptor_offset + kDescriptorKernelPropertiesOffset,
                   "kernel descriptor properties do not match AMDHSA "
                   "metadata");
  }
  return {};
}

} // namespace

ParseError decode_amdhsa_metadata(const uint8_t *data, size_t size,
                                  CodeObject *code_object) {
  MetadataBytes metadata;
  ParseError error = find_metadata_note(data, size, &metadata);
  if (error.code != ParseErrorCode::None || metadata.data == nullptr) {
    return error;
  }

  MessagePackValue root;
  error = MessagePackReader(metadata).parse(&root);
  if (error.code != ParseErrorCode::None) {
    return error;
  }

  MetadataVersion metadata_version;
  std::string target_isa;
  std::vector<MetadataKernel> kernels;
  error =
      decode_metadata_document(root, &metadata_version, &target_isa, &kernels);
  if (error.code != ParseErrorCode::None) {
    return error;
  }

  std::vector<NamedSymbol> symbols;
  error = read_dynamic_symbols(data, size, *code_object, &symbols);
  if (error.code != ParseErrorCode::None) {
    return error;
  }
  std::vector<KernelInfo> decoded_kernels;
  decoded_kernels.reserve(kernels.size());
  for (MetadataKernel &kernel : kernels) {
    Symbol symbol;
    error = find_kernel_symbol(symbols, kernel, &symbol);
    if (error.code != ParseErrorCode::None) {
      return error;
    }
    error = decode_kernel_descriptor(data, *code_object, symbol, &kernel);
    if (error.code != ParseErrorCode::None) {
      return error;
    }
    decoded_kernels.push_back(std::move(kernel.info));
  }
  code_object->metadata_version = metadata_version;
  code_object->target_isa = std::move(target_isa);
  code_object->kernels = std::move(decoded_kernels);
  code_object->has_metadata = true;
  return {};
}

} // namespace light_rocr::loader::internal
