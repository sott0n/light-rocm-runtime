#include "lrrt/bundle.hpp"

#include <stdio.h>
#include <string.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::vector<unsigned char> read_file(const char *path) {
  if (!path || path[0] == '\0') {
    throw std::invalid_argument("bundle path is empty");
  }

  FILE *file = fopen(path, "rb");
  if (!file) {
    throw std::runtime_error(std::string("failed to open ") + path);
  }
  if (fseek(file, 0, SEEK_END) != 0) {
    fclose(file);
    throw std::runtime_error(std::string("failed to seek ") + path);
  }
  long length = ftell(file);
  if (length <= 0) {
    fclose(file);
    throw std::runtime_error(std::string("empty file ") + path);
  }
  rewind(file);

  std::vector<unsigned char> data(static_cast<size_t>(length));
  if (fread(data.data(), 1, data.size(), file) != data.size()) {
    fclose(file);
    throw std::runtime_error(std::string("failed to read ") + path);
  }
  fclose(file);
  return data;
}

bool has_parent_component(const std::string &path) {
  size_t begin = 0;
  while (begin <= path.size()) {
    size_t end = path.find('/', begin);
    if (path.substr(begin, end - begin) == "..") {
      return true;
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return false;
}

std::string bundle_file_path(const char *manifest_path,
                             const std::string &file_name) {
  if (file_name.empty() || file_name[0] == '/' ||
      has_parent_component(file_name)) {
    throw std::runtime_error("bundle code object path must stay in bundle");
  }

  std::string path(manifest_path);
  size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) {
    return file_name;
  }
  return path.substr(0, slash + 1) + file_name;
}

class ManifestParser {
public:
  ManifestParser(const void *data, size_t size)
      : text_(data ? static_cast<const char *>(data) : "", size) {
    if (!data || size == 0) {
      throw std::invalid_argument("bundle manifest is empty");
    }
  }

  std::vector<lrrt::KernelManifest> parse_kernels() const {
    std::vector<lrrt::KernelManifest> manifests;
    const std::string target = read_string("target", 0, text_.size());
    if (target.empty()) {
      throw std::runtime_error("bundle manifest target is empty");
    }

    const size_t kernels = require("\"kernels\"", 0, text_.size());
    size_t position = field_value("kernels", kernels, text_.size());
    if (position >= text_.size() || text_[position] != '[') {
      throw std::runtime_error("manifest kernels is not array");
    }

    position = skip_space(position + 1, text_.size());
    while (position < text_.size() && text_[position] != ']') {
      if (text_[position] != '{') {
        throw std::runtime_error("invalid kernel manifest array");
      }
      const size_t kernel_end = matching_brace(position);
      lrrt::KernelManifest manifest = parse_kernel(position, kernel_end);
      manifest.target = target;
      for (const lrrt::KernelManifest &existing : manifests) {
        if (existing.name == manifest.name) {
          throw std::runtime_error("duplicate kernel name in bundle manifest");
        }
      }
      manifests.push_back(std::move(manifest));

      position = skip_space(kernel_end, text_.size());
      if (position < text_.size() && text_[position] == ',') {
        position = skip_space(position + 1, text_.size());
        if (position >= text_.size() || text_[position] == ']') {
          throw std::runtime_error("invalid kernel manifest array");
        }
      } else if (position >= text_.size() || text_[position] != ']') {
        throw std::runtime_error("invalid kernel manifest array");
      }
    }
    if (position >= text_.size() || manifests.empty()) {
      throw std::runtime_error("bundle manifest has no kernels");
    }
    return manifests;
  }

private:
  lrrt::KernelManifest parse_kernel(size_t kernel_begin,
                                    size_t kernel_end) const {
    lrrt::KernelManifest manifest{};

    manifest.name = read_string("name", kernel_begin, kernel_end);
    manifest.symbol = read_string("symbol", kernel_begin, kernel_end);
    manifest.code_object = read_string("code_object", kernel_begin, kernel_end);
    if (manifest.code_object.empty() || manifest.code_object[0] == '/' ||
        has_parent_component(manifest.code_object)) {
      throw std::runtime_error("bundle code object path must stay in bundle");
    }
    manifest.kernarg_size = read_size("kernarg_size", kernel_begin, kernel_end);

    const size_t shared_memory_bytes =
        read_optional_size("shared_memory_bytes", kernel_begin, kernel_end, 0);
    if (shared_memory_bytes > UINT32_MAX) {
      throw std::runtime_error("bundle shared memory requirement is too large");
    }
    manifest.shared_memory_bytes = static_cast<uint32_t>(shared_memory_bytes);

    read_u32_array("block", kernel_begin, kernel_end, manifest.block);
    read_grid_expr(kernel_begin, kernel_end, &manifest.grid_divisor,
                   &manifest.grid_multiplier);
    manifest.args = read_args(kernel_begin, kernel_end);
    for (const lrrt::KernelArgument &arg : manifest.args) {
      manifest.arg_offsets.push_back(arg.offset);
    }
    validate_args(manifest.args, manifest.kernarg_size);

    if (manifest.name.empty() || manifest.symbol.empty() ||
        manifest.kernarg_size == 0 || manifest.block[0] == 0 ||
        manifest.block[1] == 0 || manifest.block[2] == 0 ||
        manifest.grid_divisor == 0 || manifest.grid_multiplier == 0) {
      throw std::runtime_error("invalid bundle manifest");
    }
    return manifest;
  }

  size_t require(const char *needle, size_t from, size_t limit) const {
    size_t position = text_.find(needle, from);
    if (position == std::string::npos || position >= limit) {
      throw std::runtime_error(std::string("missing manifest token ") + needle);
    }
    return position;
  }

  size_t matching_brace(size_t begin) const {
    size_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = begin; i < text_.size(); ++i) {
      char value = text_[i];
      if (in_string) {
        if (escaped) {
          escaped = false;
        } else if (value == '\\') {
          escaped = true;
        } else if (value == '"') {
          in_string = false;
        }
        continue;
      }
      if (value == '"') {
        in_string = true;
      } else if (value == '{') {
        ++depth;
      } else if (value == '}') {
        if (--depth == 0) {
          return i + 1;
        }
      }
    }
    throw std::runtime_error("unterminated kernel manifest object");
  }

  size_t skip_space(size_t position, size_t limit) const {
    while (position < limit &&
           (text_[position] == ' ' || text_[position] == '\n' ||
            text_[position] == '\r' || text_[position] == '\t')) {
      ++position;
    }
    return position;
  }

  size_t field_value(const char *field, size_t from, size_t limit) const {
    std::string key = std::string("\"") + field + "\"";
    size_t key_position = require(key.c_str(), from, limit);
    size_t colon = require(":", key_position + key.size(), limit);
    return skip_space(colon + 1, limit);
  }

  std::string read_string(const char *field, size_t from, size_t limit) const {
    size_t position = field_value(field, from, limit);
    if (position >= limit || text_[position] != '"') {
      throw std::runtime_error(std::string("manifest field is not string: ") +
                               field);
    }
    ++position;
    size_t end = position;
    while (end < limit && text_[end] != '"') {
      if (text_[end] == '\\') {
        throw std::runtime_error("escaped manifest strings are unsupported");
      }
      ++end;
    }
    if (end >= limit) {
      throw std::runtime_error(std::string("unterminated manifest string: ") +
                               field);
    }
    return text_.substr(position, end - position);
  }

  size_t read_size_at(size_t position, size_t limit, const char *field) const {
    size_t end = position;
    while (end < limit && text_[end] >= '0' && text_[end] <= '9') {
      ++end;
    }
    if (end == position) {
      throw std::runtime_error(std::string("manifest field is not integer: ") +
                               field);
    }
    if (end < limit && text_[end] != ' ' && text_[end] != '\n' &&
        text_[end] != '\r' && text_[end] != '\t' && text_[end] != ',' &&
        text_[end] != ']' && text_[end] != '}') {
      throw std::runtime_error(std::string("invalid manifest integer: ") +
                               field);
    }
    return static_cast<size_t>(
        std::stoull(text_.substr(position, end - position)));
  }

  size_t read_size(const char *field, size_t from, size_t limit) const {
    return read_size_at(field_value(field, from, limit), limit, field);
  }

  size_t read_optional_size(const char *field, size_t from, size_t limit,
                            size_t default_value) const {
    std::string key = std::string("\"") + field + "\"";
    size_t position = text_.find(key, from);
    if (position == std::string::npos || position >= limit) {
      return default_value;
    }
    return read_size(field, from, limit);
  }

  bool read_optional_bool(const char *field, size_t from, size_t limit,
                          bool default_value) const {
    std::string key = std::string("\"") + field + "\"";
    size_t key_position = text_.find(key, from);
    if (key_position == std::string::npos || key_position >= limit) {
      return default_value;
    }
    size_t position = field_value(field, from, limit);
    if (text_.compare(position, 4, "true") == 0) {
      return true;
    }
    if (text_.compare(position, 5, "false") == 0) {
      return false;
    }
    throw std::runtime_error(std::string("manifest field is not boolean: ") +
                             field);
  }

  uint32_t read_u32_at(size_t position, size_t limit, const char *field) const {
    size_t value = read_size_at(position, limit, field);
    if (value > UINT32_MAX) {
      throw std::runtime_error(std::string("manifest field is too large: ") +
                               field);
    }
    return static_cast<uint32_t>(value);
  }

  void read_u32_array(const char *field, size_t from, size_t limit,
                      uint32_t out[3]) const {
    size_t position = field_value(field, from, limit);
    if (position >= limit || text_[position] != '[') {
      throw std::runtime_error(std::string("manifest field is not array: ") +
                               field);
    }
    ++position;
    for (size_t i = 0; i < 3; ++i) {
      position = skip_space(position, limit);
      out[i] = read_u32_at(position, limit, field);
      while (position < limit && text_[position] >= '0' &&
             text_[position] <= '9') {
        ++position;
      }
      position = skip_space(position, limit);
      if (i < 2) {
        if (position >= limit || text_[position] != ',') {
          throw std::runtime_error(std::string("invalid manifest array: ") +
                                   field);
        }
        ++position;
      }
    }
    position = skip_space(position, limit);
    if (position >= limit || text_[position] != ']') {
      throw std::runtime_error(std::string("invalid manifest array: ") + field);
    }
  }

  std::string read_grid_string(size_t from, size_t limit) const {
    size_t position = field_value("grid", from, limit);
    if (position >= limit || text_[position] != '[') {
      throw std::runtime_error("manifest grid is not array");
    }
    position = skip_space(position + 1, limit);
    if (position >= limit || text_[position] != '"') {
      throw std::runtime_error("manifest grid[0] is not expression string");
    }
    ++position;
    size_t end = text_.find('"', position);
    if (end == std::string::npos || end >= limit) {
      throw std::runtime_error("unterminated manifest grid expression");
    }
    std::string expr = text_.substr(position, end - position);

    position = skip_space(end + 1, limit);
    if (position >= limit || text_[position] != ',') {
      throw std::runtime_error("invalid manifest grid array");
    }
    position = skip_space(position + 1, limit);
    if (read_u32_at(position, limit, "grid") != 1) {
      throw std::runtime_error("bundle grid y dimension must be 1");
    }
    while (position < limit && text_[position] >= '0' &&
           text_[position] <= '9') {
      ++position;
    }

    position = skip_space(position, limit);
    if (position >= limit || text_[position] != ',') {
      throw std::runtime_error("invalid manifest grid array");
    }
    position = skip_space(position + 1, limit);
    if (read_u32_at(position, limit, "grid") != 1) {
      throw std::runtime_error("bundle grid z dimension must be 1");
    }
    while (position < limit && text_[position] >= '0' &&
           text_[position] <= '9') {
      ++position;
    }

    position = skip_space(position, limit);
    if (position >= limit || text_[position] != ']') {
      throw std::runtime_error("invalid manifest grid array");
    }
    return expr;
  }

  void read_grid_expr(size_t from, size_t limit, uint32_t *divisor,
                      uint32_t *multiplier) const {
    const std::string expr = read_grid_string(from, limit);
    const std::string prefix = "ceil_div(n, ";
    if (expr.compare(0, prefix.size(), prefix) != 0) {
      throw std::runtime_error("unsupported bundle grid expression");
    }
    size_t divisor_begin = prefix.size();
    size_t divisor_end = expr.find(')', divisor_begin);
    if (divisor_end == std::string::npos) {
      throw std::runtime_error("unsupported bundle grid expression");
    }
    size_t multiply = divisor_end + 1;
    while (multiply < expr.size() && expr[multiply] == ' ') {
      ++multiply;
    }
    if (multiply >= expr.size() || expr[multiply] != '*') {
      throw std::runtime_error("unsupported bundle grid expression");
    }
    size_t multiplier_begin = multiply + 1;
    while (multiplier_begin < expr.size() && expr[multiplier_begin] == ' ') {
      ++multiplier_begin;
    }

    *divisor = parse_grid_integer(expr, divisor_begin, divisor_end);
    *multiplier = parse_grid_integer(expr, multiplier_begin, expr.size());
  }

  uint32_t parse_grid_integer(const std::string &expr, size_t begin,
                              size_t end) const {
    while (begin < end && expr[begin] == ' ') {
      ++begin;
    }
    while (end > begin && expr[end - 1] == ' ') {
      --end;
    }
    if (begin == end) {
      throw std::runtime_error("unsupported bundle grid expression");
    }

    uint64_t value = 0;
    for (size_t i = begin; i < end; ++i) {
      if (expr[i] < '0' || expr[i] > '9') {
        throw std::runtime_error("unsupported bundle grid expression");
      }
      value = value * 10 + static_cast<uint64_t>(expr[i] - '0');
      if (value > UINT32_MAX) {
        throw std::runtime_error("bundle grid expression is too large");
      }
    }
    return static_cast<uint32_t>(value);
  }

  std::vector<lrrt::KernelArgument> read_args(size_t from, size_t limit) const {
    std::vector<lrrt::KernelArgument> args_out;
    size_t args = field_value("args", from, limit);
    if (args >= limit || text_[args] != '[') {
      throw std::runtime_error("manifest args is not array");
    }
    size_t position = skip_space(args + 1, limit);
    while (position < limit && text_[position] != ']') {
      if (text_[position] != '{') {
        throw std::runtime_error("invalid manifest args array");
      }
      const size_t arg_end = matching_brace(position);
      lrrt::KernelArgument arg{};
      arg.name = read_string("name", position, arg_end);
      arg.type = read_string("type", position, arg_end);
      arg.offset = read_size("offset", position, arg_end);
      arg.size = read_size("size", position, arg_end);
      arg.optional = read_optional_bool("optional", position, arg_end, false);
      args_out.push_back(std::move(arg));

      position = skip_space(arg_end, limit);
      if (position < limit && text_[position] == ',') {
        position = skip_space(position + 1, limit);
        if (position >= limit || text_[position] == ']') {
          throw std::runtime_error("invalid manifest args array");
        }
      } else if (position >= limit || text_[position] != ']') {
        throw std::runtime_error("invalid manifest args array");
      }
    }
    if (args_out.empty()) {
      throw std::runtime_error("bundle manifest has no arguments");
    }
    return args_out;
  }

  void validate_args(const std::vector<lrrt::KernelArgument> &args,
                     size_t kernarg_size) const {
    size_t previous = 0;
    for (size_t i = 0; i < args.size(); ++i) {
      const lrrt::KernelArgument &arg = args[i];
      if (arg.name.empty() || arg.type.empty() || arg.size == 0) {
        throw std::runtime_error("invalid bundle argument manifest");
      }
      if (arg.offset >= kernarg_size || arg.size > kernarg_size ||
          arg.offset + arg.size > kernarg_size) {
        throw std::runtime_error("bundle argument offset exceeds kernarg size");
      }
      if (i > 0 && arg.offset <= previous) {
        throw std::runtime_error(
            "bundle argument offsets must be strictly increasing");
      }
      for (size_t j = 0; j < i; ++j) {
        if (args[j].name == arg.name) {
          throw std::runtime_error("duplicate bundle argument name");
        }
      }
      previous = arg.offset;
    }
  }

  std::string text_;
};

} // namespace

namespace lrrt {

KernelManifest parse_bundle_manifest(const void *data, size_t size) {
  return parse_bundle_manifests(data, size).front();
}

KernelManifest parse_bundle_manifest(const void *data, size_t size,
                                     const char *kernel_name) {
  if (!kernel_name || kernel_name[0] == '\0') {
    throw std::invalid_argument("bundle kernel name is empty");
  }
  for (KernelManifest &manifest : parse_bundle_manifests(data, size)) {
    if (manifest.name == kernel_name) {
      return std::move(manifest);
    }
  }
  throw std::runtime_error(std::string("bundle kernel not found: ") +
                           kernel_name);
}

std::vector<KernelManifest> parse_bundle_manifests(const void *data,
                                                   size_t size) {
  return ManifestParser(data, size).parse_kernels();
}

lr_launch_config_t launch_config_from_manifest(const KernelManifest &manifest,
                                               uint32_t n) {
  if (manifest.grid_divisor == 0 || manifest.grid_multiplier == 0) {
    throw std::runtime_error("invalid bundle launch configuration");
  }
  const uint64_t programs =
      (static_cast<uint64_t>(n) + manifest.grid_divisor - 1) /
      manifest.grid_divisor;
  const uint64_t grid_x = programs * manifest.grid_multiplier;
  if (grid_x > UINT32_MAX) {
    throw std::runtime_error("bundle grid size is too large");
  }
  return {
      {static_cast<uint32_t>(grid_x), 1, 1},
      {manifest.block[0], manifest.block[1], manifest.block[2]},
      manifest.shared_memory_bytes,
  };
}

void require_kernarg_layout(const KernelManifest &manifest, size_t kernarg_size,
                            const std::vector<size_t> &arg_offsets) {
  if (manifest.kernarg_size != kernarg_size ||
      manifest.arg_offsets != arg_offsets) {
    throw std::runtime_error("bundle manifest does not match kernarg layout");
  }
}

KernargBuffer::KernargBuffer(const KernelManifest &manifest)
    : data_(manifest.kernarg_size, 0), args_(manifest.args),
      bound_(manifest.args.size(), false) {
  if (data_.empty() || args_.empty()) {
    throw std::runtime_error("invalid bundle kernarg layout");
  }
}

void KernargBuffer::set_raw(size_t index, const void *value,
                            size_t value_size) {
  if (!value && value_size != 0) {
    throw std::invalid_argument("bundle kernarg value is null");
  }
  if (index >= args_.size()) {
    throw std::out_of_range("bundle kernarg index is out of range");
  }

  const KernelArgument &arg = args_[index];
  if (arg.offset > data_.size() || arg.size > data_.size() ||
      arg.offset + arg.size > data_.size()) {
    throw std::runtime_error("invalid bundle kernarg offset");
  }
  if (value_size > arg.size) {
    throw std::runtime_error("bundle kernarg value exceeds argument slot");
  }
  memcpy(data_.data() + arg.offset, value, value_size);
  bound_[index] = true;
}

void KernargBuffer::set_raw(const char *name, const void *value,
                            size_t value_size) {
  if (!name || name[0] == '\0') {
    throw std::invalid_argument("bundle kernarg name is empty");
  }
  for (size_t i = 0; i < args_.size(); ++i) {
    if (args_[i].name == name) {
      set_raw(i, value, value_size);
      return;
    }
  }
  throw std::out_of_range(std::string("bundle kernarg not found: ") + name);
}

void KernargBuffer::validate() const {
  for (size_t i = 0; i < args_.size(); ++i) {
    if (!args_[i].optional && !bound_[i]) {
      throw std::runtime_error(
          std::string("missing required bundle kernarg: ") + args_[i].name);
    }
  }
}

Bundle::Bundle(Device device, const char *manifest_path)
    : manifest_(load_manifest(manifest_path, nullptr)),
      module_(device, read_code_object(manifest_path, manifest_)),
      kernel_(module_.kernel(manifest_.symbol.c_str())) {}

Bundle::Bundle(Device device, const char *manifest_path,
               const char *kernel_name)
    : manifest_(load_manifest(manifest_path, kernel_name)),
      module_(device, read_code_object(manifest_path, manifest_)),
      kernel_(module_.kernel(manifest_.symbol.c_str())) {}

lr_launch_config_t Bundle::launch_config(uint32_t n) const {
  return launch_config_from_manifest(manifest_, n);
}

void Bundle::launch(uint32_t n, const KernargBuffer &args) const {
  args.validate();
  lrrt::launch(kernel_, launch_config(n), args.data(), args.size());
}

void Bundle::launch(uint32_t n, const KernargBuffer &args,
                    const std::vector<const Event *> &dependencies) const {
  args.validate();
  lrrt::launch(kernel_, launch_config(n), args.data(), args.size(),
               dependencies);
}

void Bundle::launch(const Queue &queue, uint32_t n, const KernargBuffer &args,
                    const std::vector<const Event *> &dependencies) const {
  args.validate();
  lrrt::launch(queue, kernel_, launch_config(n), args.data(), args.size(),
               dependencies);
}

KernelManifest Bundle::load_manifest(const char *manifest_path,
                                     const char *kernel_name) {
  std::vector<unsigned char> data = read_file(manifest_path);
  if (kernel_name) {
    return parse_bundle_manifest(data, kernel_name);
  }
  return parse_bundle_manifest(data);
}

std::vector<unsigned char>
Bundle::read_code_object(const char *manifest_path,
                         const KernelManifest &manifest) {
  std::string hsaco_path =
      bundle_file_path(manifest_path, manifest.code_object);
  return read_file(hsaco_path.c_str());
}

} // namespace lrrt
