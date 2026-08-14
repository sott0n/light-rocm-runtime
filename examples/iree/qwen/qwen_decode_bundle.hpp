#ifndef LRRT_EXAMPLES_IREE_QWEN_DECODE_BUNDLE_HPP_
#define LRRT_EXAMPLES_IREE_QWEN_DECODE_BUNDLE_HPP_

#include "executor/iree/metadata_json.hpp"

#include <cstddef>
#include <cstdint>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace lrrt::examples::iree::qwen {

namespace detail = lrrt::executor::iree::detail;

struct QwenDecodeBundleManifest {
  uint32_t manifest_version = 0;
  std::string target;
  std::filesystem::path layer_vmfb;
  std::filesystem::path tail_vmfb;
  std::string layer_export;
  std::string tail_export;
  std::string precision = "f32";
  uint32_t sequence_capacity = 0;
  uint32_t max_cache_tokens = 0;
  uint32_t kv_cache_tokens = 0;
  uint32_t kv_cache_dim = 0;
};

inline std::string read_text_file(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to open IREE Qwen bundle manifest: " +
                             path.string());
  }
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

inline std::filesystem::path
require_relative_bundle_path(const std::filesystem::path &bundle_dir,
                             const std::string &field,
                             const std::string &value) {
  const std::filesystem::path relative(value);
  if (relative.empty() || relative.is_absolute()) {
    throw std::runtime_error(
        "IREE Qwen bundle field must be a relative path: " + field);
  }
  for (const std::filesystem::path &part : relative) {
    if (part == "..") {
      throw std::runtime_error("IREE Qwen bundle path must not contain '..': " +
                               field);
    }
  }
  return bundle_dir / relative;
}

inline QwenDecodeBundleManifest
parse_qwen_decode_bundle_manifest(const std::filesystem::path &bundle_dir,
                                  const std::filesystem::path &manifest_path) {
  const std::string text = read_text_file(manifest_path);
  detail::JsonParser parser(text.data(), text.size());
  const detail::JsonValue root = parser.parse();

  QwenDecodeBundleManifest manifest;
  manifest.manifest_version = detail::require_u32(root, "manifest_version");
  if (manifest.manifest_version != 1 && manifest.manifest_version != 2) {
    throw std::runtime_error("unsupported IREE Qwen bundle manifest_version: " +
                             std::to_string(manifest.manifest_version));
  }
  manifest.target = detail::require_string(root, "target");
  if (manifest.target.empty()) {
    throw std::runtime_error("IREE Qwen bundle target must not be empty");
  }

  manifest.layer_vmfb = require_relative_bundle_path(
      bundle_dir, "layer_vmfb", detail::require_string(root, "layer_vmfb"));
  manifest.tail_vmfb = require_relative_bundle_path(
      bundle_dir, "tail_vmfb", detail::require_string(root, "tail_vmfb"));
  manifest.layer_export = detail::require_string(root, "layer_export");
  manifest.tail_export = detail::require_string(root, "tail_export");
  if (manifest.layer_export.empty() || manifest.tail_export.empty()) {
    throw std::runtime_error("IREE Qwen bundle exports must not be empty");
  }
  if (manifest.manifest_version >= 2) {
    manifest.precision = detail::require_string(root, "precision");
  }
  if (manifest.precision != "f32" && manifest.precision != "f16" &&
      manifest.precision != "bf16") {
    throw std::runtime_error(
        "IREE Qwen bundle precision must be f32, f16, or bf16");
  }

  manifest.sequence_capacity = detail::require_u32(root, "sequence_capacity");
  manifest.max_cache_tokens = detail::require_u32(root, "max_cache_tokens");
  const detail::JsonValue &shape =
      detail::require_field(root, "kv_cache_shape");
  if (shape.type != detail::JsonValue::Type::kArray ||
      shape.array.size() != 2) {
    throw std::runtime_error(
        "IREE Qwen bundle kv_cache_shape must be [tokens, dim]");
  }
  for (const detail::JsonValue &dim : shape.array) {
    if (dim.type != detail::JsonValue::Type::kNumber ||
        dim.number > UINT32_MAX) {
      throw std::runtime_error(
          "IREE Qwen bundle kv_cache_shape dims must be u32");
    }
  }
  manifest.kv_cache_tokens = static_cast<uint32_t>(shape.array[0].number);
  manifest.kv_cache_dim = static_cast<uint32_t>(shape.array[1].number);
  if (manifest.max_cache_tokens == 0 ||
      manifest.kv_cache_tokens != manifest.max_cache_tokens) {
    throw std::runtime_error(
        "IREE Qwen bundle max_cache_tokens must match kv_cache_shape[0]");
  }
  if (manifest.sequence_capacity == 0 ||
      manifest.sequence_capacity > manifest.max_cache_tokens) {
    throw std::runtime_error("IREE Qwen bundle sequence_capacity must be in "
                             "(0, max_cache_tokens]");
  }
  if (manifest.kv_cache_dim == 0) {
    throw std::runtime_error("IREE Qwen bundle kv_cache_shape[1] is zero");
  }
  return manifest;
}

inline QwenDecodeBundleManifest
load_qwen_decode_bundle_manifest(const std::filesystem::path &bundle_dir) {
  return parse_qwen_decode_bundle_manifest(bundle_dir,
                                           bundle_dir / "manifest.json");
}

} // namespace lrrt::examples::iree::qwen

#endif // LRRT_EXAMPLES_IREE_QWEN_DECODE_BUNDLE_HPP_
