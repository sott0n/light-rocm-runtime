#ifndef LRRT_EXECUTOR_IREE_METADATA_JSON_HPP_
#define LRRT_EXECUTOR_IREE_METADATA_JSON_HPP_

#include "iree_adapter.hpp"

#include <stdint.h>

#include <array>
#include <cctype>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lrrt::executor::iree {
namespace detail {

struct JsonValue {
  enum class Type { kObject, kArray, kString, kNumber };

  Type type = Type::kObject;
  std::map<std::string, JsonValue> object;
  std::vector<JsonValue> array;
  std::string string;
  uint64_t number = 0;
};

class JsonParser {
public:
  JsonParser(const char *data, size_t size) : data_(data), size_(size) {
    if (!data_ || size_ == 0) {
      throw std::runtime_error("IREE metadata JSON is empty");
    }
  }

  JsonValue parse() {
    JsonValue value = parse_value();
    skip_ws();
    if (position_ != size_) {
      throw std::runtime_error("IREE metadata JSON has trailing content");
    }
    return value;
  }

private:
  JsonValue parse_value() {
    skip_ws();
    if (position_ >= size_) {
      throw std::runtime_error("unexpected end of IREE metadata JSON");
    }
    switch (data_[position_]) {
    case '{':
      return parse_object();
    case '[':
      return parse_array();
    case '"':
      return JsonValue{JsonValue::Type::kString, {}, {}, parse_string(), 0};
    default:
      if (std::isdigit(static_cast<unsigned char>(data_[position_]))) {
        return parse_number();
      }
      throw std::runtime_error("unexpected IREE metadata JSON token");
    }
  }

  JsonValue parse_object() {
    expect('{');
    JsonValue value;
    value.type = JsonValue::Type::kObject;
    skip_ws();
    if (consume('}')) {
      return value;
    }
    while (true) {
      skip_ws();
      if (position_ >= size_ || data_[position_] != '"') {
        throw std::runtime_error("IREE metadata object key is not a string");
      }
      std::string key = parse_string();
      skip_ws();
      expect(':');
      value.object.emplace(std::move(key), parse_value());
      skip_ws();
      if (consume('}')) {
        return value;
      }
      expect(',');
    }
  }

  JsonValue parse_array() {
    expect('[');
    JsonValue value;
    value.type = JsonValue::Type::kArray;
    skip_ws();
    if (consume(']')) {
      return value;
    }
    while (true) {
      value.array.push_back(parse_value());
      skip_ws();
      if (consume(']')) {
        return value;
      }
      expect(',');
    }
  }

  std::string parse_string() {
    expect('"');
    std::string value;
    while (position_ < size_) {
      const char c = data_[position_++];
      if (c == '"') {
        return value;
      }
      if (c == '\\') {
        if (position_ >= size_) {
          throw std::runtime_error("unterminated IREE metadata string escape");
        }
        const char escaped = data_[position_++];
        switch (escaped) {
        case '"':
        case '\\':
        case '/':
          value.push_back(escaped);
          break;
        case 'n':
          value.push_back('\n');
          break;
        case 'r':
          value.push_back('\r');
          break;
        case 't':
          value.push_back('\t');
          break;
        default:
          throw std::runtime_error("unsupported IREE metadata string escape");
        }
      } else {
        value.push_back(c);
      }
    }
    throw std::runtime_error("unterminated IREE metadata string");
  }

  JsonValue parse_number() {
    uint64_t number = 0;
    while (position_ < size_ &&
           std::isdigit(static_cast<unsigned char>(data_[position_]))) {
      number = number * 10 + static_cast<uint64_t>(data_[position_] - '0');
      ++position_;
    }
    JsonValue value;
    value.type = JsonValue::Type::kNumber;
    value.number = number;
    return value;
  }

  void skip_ws() {
    while (position_ < size_ &&
           std::isspace(static_cast<unsigned char>(data_[position_]))) {
      ++position_;
    }
  }

  bool consume(char expected) {
    if (position_ < size_ && data_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  void expect(char expected) {
    skip_ws();
    if (position_ >= size_ || data_[position_] != expected) {
      throw std::runtime_error("unexpected IREE metadata JSON structure");
    }
    ++position_;
  }

  const char *data_ = nullptr;
  size_t size_ = 0;
  size_t position_ = 0;
};

inline const JsonValue &require_field(const JsonValue &object,
                                      const char *field) {
  if (object.type != JsonValue::Type::kObject) {
    throw std::runtime_error("IREE metadata value is not an object");
  }
  const auto it = object.object.find(field);
  if (it == object.object.end()) {
    throw std::runtime_error(std::string("missing IREE metadata field: ") +
                             field);
  }
  return it->second;
}

inline const JsonValue *find_field(const JsonValue &object, const char *field) {
  if (object.type != JsonValue::Type::kObject) {
    throw std::runtime_error("IREE metadata value is not an object");
  }
  const auto it = object.object.find(field);
  return it == object.object.end() ? nullptr : &it->second;
}

inline std::string require_string(const JsonValue &object, const char *field) {
  const JsonValue &value = require_field(object, field);
  if (value.type != JsonValue::Type::kString) {
    throw std::runtime_error(
        std::string("IREE metadata field is not string: ") + field);
  }
  return value.string;
}

inline uint32_t require_u32(const JsonValue &object, const char *field) {
  const JsonValue &value = require_field(object, field);
  if (value.type != JsonValue::Type::kNumber || value.number > UINT32_MAX) {
    throw std::runtime_error(std::string("IREE metadata field is not u32: ") +
                             field);
  }
  return static_cast<uint32_t>(value.number);
}

inline std::vector<std::string> require_string_array(const JsonValue &object,
                                                     const char *field) {
  const JsonValue &value = require_field(object, field);
  if (value.type != JsonValue::Type::kArray) {
    throw std::runtime_error(std::string("IREE metadata field is not array: ") +
                             field);
  }
  std::vector<std::string> strings;
  strings.reserve(value.array.size());
  for (const JsonValue &item : value.array) {
    if (item.type != JsonValue::Type::kString) {
      throw std::runtime_error(std::string("IREE metadata array item is not "
                                           "string: ") +
                               field);
    }
    strings.push_back(item.string);
  }
  return strings;
}

inline std::array<uint32_t, 3> require_dim3_array(const JsonValue &object,
                                                  const char *field) {
  const JsonValue &value = require_field(object, field);
  if (value.type != JsonValue::Type::kArray || value.array.size() != 3) {
    throw std::runtime_error(std::string("IREE metadata field is not dim3: ") +
                             field);
  }
  std::array<uint32_t, 3> dims = {1, 1, 1};
  for (size_t i = 0; i < dims.size(); ++i) {
    const JsonValue &item = value.array[i];
    if (item.type != JsonValue::Type::kNumber || item.number > UINT32_MAX) {
      throw std::runtime_error(std::string("IREE metadata dim is not u32: ") +
                               field);
    }
    dims[i] = static_cast<uint32_t>(item.number);
  }
  return dims;
}

inline BindingMetadata parse_binding_metadata(const JsonValue &value) {
  BindingMetadata binding;
  binding.index = require_u32(value, "index");
  binding.type = require_string(value, "type");
  binding.flags = require_string_array(value, "flags");
  return binding;
}

inline KernelMetadata parse_kernel_metadata(const JsonValue &value) {
  KernelMetadata kernel;
  kernel.symbol = require_string(value, "symbol");
  kernel.attributes = require_string_array(value, "attributes");
  return kernel;
}

inline DispatchMetadata parse_dispatch_metadata(const JsonValue &value) {
  DispatchMetadata dispatch;
  dispatch.executable = require_string(value, "executable");
  dispatch.variant = require_string(value, "variant");
  dispatch.symbol = require_string(value, "symbol");
  return dispatch;
}

inline ExportMetadata parse_export_metadata(const JsonValue &value) {
  ExportMetadata export_metadata;
  export_metadata.symbol = require_string(value, "symbol");
  export_metadata.ordinal = require_u32(value, "ordinal");
  export_metadata.workgroup_size = require_dim3_array(value, "workgroup_size");
  export_metadata.subgroup_size = require_u32(value, "subgroup_size");

  const JsonValue &bindings = require_field(value, "bindings");
  if (bindings.type != JsonValue::Type::kArray) {
    throw std::runtime_error("IREE metadata bindings is not array");
  }
  export_metadata.bindings.reserve(bindings.array.size());
  for (const JsonValue &binding : bindings.array) {
    export_metadata.bindings.push_back(parse_binding_metadata(binding));
  }

  export_metadata.kernel =
      parse_kernel_metadata(require_field(value, "kernel"));
  export_metadata.dispatch =
      parse_dispatch_metadata(require_field(value, "dispatch"));
  return export_metadata;
}

inline void parse_executable_entry_metadata(const JsonValue &value,
                                            ExecutableMetadata *metadata) {
  const std::string executable = require_string(value, "executable");
  const std::string variant = require_string(value, "variant");
  if (metadata->executable.empty()) {
    metadata->executable = executable;
  }
  if (metadata->variant.empty()) {
    metadata->variant = variant;
  }

  const JsonValue &exports = require_field(value, "exports");
  if (exports.type != JsonValue::Type::kArray || exports.array.empty()) {
    throw std::runtime_error("IREE metadata exports must be a non-empty array");
  }
  metadata->exports.reserve(metadata->exports.size() + exports.array.size());
  for (const JsonValue &export_value : exports.array) {
    metadata->exports.push_back(parse_export_metadata(export_value));
  }
}

} // namespace detail

inline ExecutableMetadata parse_executable_metadata_json(const void *data,
                                                         size_t size) {
  detail::JsonParser parser(static_cast<const char *>(data), size);
  const detail::JsonValue root = parser.parse();

  ExecutableMetadata metadata;
  metadata.target = detail::require_string(root, "target");

  if (const detail::JsonValue *executables =
          detail::find_field(root, "executables")) {
    if (executables->type != detail::JsonValue::Type::kArray ||
        executables->array.empty()) {
      throw std::runtime_error(
          "IREE metadata executables must be a non-empty array");
    }
    for (const detail::JsonValue &executable_value : executables->array) {
      detail::parse_executable_entry_metadata(executable_value, &metadata);
    }
  } else {
    detail::parse_executable_entry_metadata(root, &metadata);
  }
  return metadata;
}

inline ExecutableMetadata
parse_executable_metadata_json(const std::vector<unsigned char> &data) {
  return parse_executable_metadata_json(data.data(), data.size());
}

inline ExecutableMetadata
parse_executable_metadata_json(const std::string &data) {
  return parse_executable_metadata_json(data.data(), data.size());
}

} // namespace lrrt::executor::iree

#endif // LRRT_EXECUTOR_IREE_METADATA_JSON_HPP_
