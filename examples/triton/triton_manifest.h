#ifndef LRRT_EXAMPLES_TRITON_MANIFEST_H_
#define LRRT_EXAMPLES_TRITON_MANIFEST_H_

#include "lrrt/lrrt.h"

#include <stddef.h>
#include <stdint.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace lrrt_example {
namespace triton {

struct KernelManifest {
  std::string name;
  std::string symbol;
  std::string code_object;
  uint32_t block[3];
  uint32_t grid_multiplier;
  uint32_t grid_divisor;
  size_t kernarg_size;
  uint32_t shared_memory_bytes;
  std::vector<size_t> arg_offsets;
};

inline uint32_t ceil_div(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

class ManifestParser {
public:
  explicit ManifestParser(const std::vector<unsigned char> &data)
      : text_(data.begin(), data.end()) {}

  KernelManifest parse_first_kernel() const {
    KernelManifest manifest{};
    const size_t kernels = require("\"kernels\"");
    const size_t kernel = require("{", kernels);

    manifest.name = read_string("name", kernel);
    manifest.symbol = read_string("symbol", kernel);
    manifest.code_object = read_string("code_object", kernel);
    manifest.kernarg_size = read_size("kernarg_size", kernel);
    const size_t shared_memory_bytes =
        read_optional_size("shared_memory_bytes", kernel, 0);
    if (shared_memory_bytes > UINT32_MAX) {
      throw std::runtime_error("Triton shared memory requirement is too large");
    }
    manifest.shared_memory_bytes =
        static_cast<uint32_t>(shared_memory_bytes);
    read_u32_array("block", kernel, manifest.block);
    read_grid_expr(kernel, &manifest.grid_divisor, &manifest.grid_multiplier);
    manifest.arg_offsets = read_arg_offsets(kernel);

    if (manifest.symbol.empty() || manifest.code_object.empty() ||
        manifest.kernarg_size == 0 || manifest.block[0] == 0 ||
        manifest.block[1] == 0 || manifest.block[2] == 0 ||
        manifest.grid_divisor == 0 || manifest.grid_multiplier == 0) {
      throw std::runtime_error("invalid Triton manifest");
    }
    return manifest;
  }

private:
  size_t require(const char *needle, size_t from = 0) const {
    size_t position = text_.find(needle, from);
    if (position == std::string::npos) {
      throw std::runtime_error(std::string("missing manifest field ") + needle);
    }
    return position;
  }

  size_t field_value(const char *field, size_t from) const {
    std::string key = std::string("\"") + field + "\"";
    size_t key_position = require(key.c_str(), from);
    size_t colon = require(":", key_position + key.size());
    return skip_space(colon + 1);
  }

  size_t skip_space(size_t position) const {
    while (position < text_.size() &&
           (text_[position] == ' ' || text_[position] == '\n' ||
            text_[position] == '\r' || text_[position] == '\t')) {
      ++position;
    }
    return position;
  }

  std::string read_string(const char *field, size_t from) const {
    size_t position = field_value(field, from);
    if (position >= text_.size() || text_[position] != '"') {
      throw std::runtime_error(std::string("manifest field is not string: ") +
                               field);
    }
    ++position;
    size_t end = text_.find('"', position);
    if (end == std::string::npos) {
      throw std::runtime_error(std::string("unterminated manifest string: ") +
                               field);
    }
    return text_.substr(position, end - position);
  }

  size_t read_size(const char *field, size_t from) const {
    size_t position = field_value(field, from);
    size_t end = position;
    while (end < text_.size() && text_[end] >= '0' && text_[end] <= '9') {
      ++end;
    }
    if (end == position) {
      throw std::runtime_error(std::string("manifest field is not integer: ") +
                               field);
    }
    return static_cast<size_t>(
        std::stoull(text_.substr(position, end - position)));
  }

  size_t read_optional_size(const char *field, size_t from,
                            size_t default_value) const {
    std::string key = std::string("\"") + field + "\"";
    size_t field_position = text_.find(key, from);
    size_t current_symbol = text_.find("\"symbol\"", from);
    size_t next_symbol = current_symbol == std::string::npos
                             ? std::string::npos
                             : text_.find("\"symbol\"", current_symbol + 1);
    if (field_position == std::string::npos ||
        (next_symbol != std::string::npos && field_position > next_symbol)) {
      return default_value;
    }
    return read_size(field, from);
  }

  void read_u32_array(const char *field, size_t from, uint32_t out[3]) const {
    size_t position = field_value(field, from);
    if (position >= text_.size() || text_[position] != '[') {
      throw std::runtime_error(std::string("manifest field is not array: ") +
                               field);
    }
    ++position;
    for (size_t i = 0; i < 3; ++i) {
      position = skip_space(position);
      size_t end = position;
      while (end < text_.size() && text_[end] >= '0' && text_[end] <= '9') {
        ++end;
      }
      if (end == position) {
        throw std::runtime_error(std::string("invalid manifest array: ") +
                                 field);
      }
      out[i] = static_cast<uint32_t>(
          std::stoul(text_.substr(position, end - position)));
      position = skip_space(end);
      if (i < 2) {
        if (position >= text_.size() || text_[position] != ',') {
          throw std::runtime_error(std::string("invalid manifest array: ") +
                                   field);
        }
        ++position;
      }
    }
  }

  void read_grid_expr(size_t from, uint32_t *divisor,
                      uint32_t *multiplier) const {
    const std::string expr = read_first_grid_string(from);
    const std::string prefix = "ceil_div(n, ";
    if (expr.compare(0, prefix.size(), prefix) != 0) {
      throw std::runtime_error("unsupported Triton grid expression");
    }
    size_t divisor_begin = prefix.size();
    size_t divisor_end = expr.find(')', divisor_begin);
    size_t multiply = expr.find('*', divisor_end);
    if (divisor_end == std::string::npos || multiply == std::string::npos) {
      throw std::runtime_error("unsupported Triton grid expression");
    }
    *divisor = static_cast<uint32_t>(
        std::stoul(expr.substr(divisor_begin, divisor_end - divisor_begin)));
    *multiplier =
        static_cast<uint32_t>(std::stoul(expr.substr(multiply + 1)));
  }

  std::string read_first_grid_string(size_t from) const {
    size_t position = field_value("grid", from);
    if (position >= text_.size() || text_[position] != '[') {
      throw std::runtime_error("manifest grid is not array");
    }
    position = skip_space(position + 1);
    if (position >= text_.size() || text_[position] != '"') {
      throw std::runtime_error("manifest grid[0] is not expression string");
    }
    ++position;
    size_t end = text_.find('"', position);
    if (end == std::string::npos) {
      throw std::runtime_error("unterminated manifest grid expression");
    }
    return text_.substr(position, end - position);
  }

  std::vector<size_t> read_arg_offsets(size_t from) const {
    std::vector<size_t> offsets;
    size_t args = field_value("args", from);
    size_t position = args;
    while (true) {
      size_t offset_field = text_.find("\"offset\"", position);
      if (offset_field == std::string::npos) {
        break;
      }
      size_t args_end = text_.find("\"kernarg_size\"", args);
      if (args_end != std::string::npos && offset_field > args_end) {
        break;
      }
      size_t colon = require(":", offset_field);
      size_t value = skip_space(colon + 1);
      size_t end = value;
      while (end < text_.size() && text_[end] >= '0' && text_[end] <= '9') {
        ++end;
      }
      if (end == value) {
        throw std::runtime_error("invalid manifest arg offset");
      }
      offsets.push_back(
          static_cast<size_t>(std::stoull(text_.substr(value, end - value))));
      position = end;
    }
    if (offsets.empty()) {
      throw std::runtime_error("manifest has no arg offsets");
    }
    return offsets;
  }

  std::string text_;
};

inline KernelManifest parse_first_kernel_manifest(
    const std::vector<unsigned char> &data) {
  return ManifestParser(data).parse_first_kernel();
}

inline lr_launch_config_t launch_config_from_manifest(
    const KernelManifest &manifest, uint32_t n) {
  const uint32_t programs = ceil_div(n, manifest.grid_divisor);
  return {
      {programs * manifest.grid_multiplier, 1, 1},
      {manifest.block[0], manifest.block[1], manifest.block[2]},
      manifest.shared_memory_bytes,
  };
}

} // namespace triton
} // namespace lrrt_example

#endif // LRRT_EXAMPLES_TRITON_MANIFEST_H_
