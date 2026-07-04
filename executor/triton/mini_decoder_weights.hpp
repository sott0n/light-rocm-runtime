#ifndef LRRT_EXECUTOR_TRITON_MINI_DECODER_WEIGHTS_HPP_
#define LRRT_EXECUTOR_TRITON_MINI_DECODER_WEIGHTS_HPP_

#include "mini_decoder_layer.hpp"

#include <stdint.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace lrrt::executor::triton::mini {

struct DecoderLayerShape {
  uint32_t keys;
  uint32_t hidden;
  uint32_t heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t intermediate;

  uint32_t q_dim() const { return heads * head_dim; }
  uint32_t kv_dim() const { return kv_heads * head_dim; }
  uint32_t qkv_dim() const { return q_dim(); }
};

struct DecoderLayerWeights {
  DecoderLayerShape shape;
  float rope_theta = 10000.0f;
  std::vector<float> attention_norm_weight;
  std::vector<float> mlp_norm_weight;
  std::vector<float> q_weight;
  std::vector<float> k_weight;
  std::vector<float> v_weight;
  std::vector<float> out_weight;
  std::vector<float> gate_weight;
  std::vector<float> up_weight;
  std::vector<float> down_weight;
};

struct ModelTailWeights {
  uint32_t hidden;
  uint32_t vocab;
  std::vector<uint32_t> token_ids;
  std::vector<float> token_embeddings;
  std::vector<float> final_norm_weight;
  std::vector<float> lm_head_weight;
};

namespace detail {

struct TensorSpec {
  const char *name;
  size_t count;
  std::vector<float> DecoderLayerWeights::*field;
};

struct TailTensorSpec {
  const char *name;
  size_t count;
  std::vector<float> ModelTailWeights::*field;
};

inline void validate_shape(const DecoderLayerShape &shape) {
  if (shape.keys == 0 || shape.hidden == 0 || shape.heads == 0 ||
      shape.kv_heads == 0 || shape.head_dim == 0 || shape.head_dim % 2 != 0 ||
      shape.intermediate == 0) {
    throw std::runtime_error("mini decoder weight manifest has invalid shape");
  }
  if (shape.kv_heads > shape.heads || shape.heads % shape.kv_heads != 0) {
    throw std::runtime_error(
        "mini decoder weight manifest has invalid GQA shape");
  }
  if (shape.heads > std::numeric_limits<uint32_t>::max() / shape.head_dim) {
    throw std::overflow_error("mini decoder q dimension overflows uint32");
  }
  if (shape.kv_heads > std::numeric_limits<uint32_t>::max() / shape.head_dim) {
    throw std::overflow_error("mini decoder kv dimension overflows uint32");
  }
}

inline size_t checked_multiply(size_t left, size_t right, const char *context) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    throw std::overflow_error(std::string(context) + " overflows size_t");
  }
  return left * right;
}

inline std::vector<unsigned char> read_file(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("failed to open mini decoder weight file: " +
                             path);
  }
  std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("mini decoder weight file is empty: " + path);
  }
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> data(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    throw std::runtime_error("failed to read mini decoder weight file: " +
                             path);
  }
  return data;
}

inline std::string read_text_file(const char *path) {
  std::vector<unsigned char> data = read_file(path);
  return std::string(reinterpret_cast<const char *>(data.data()), data.size());
}

inline bool has_parent_component(const std::string &path) {
  size_t begin = 0;
  while (begin <= path.size()) {
    size_t end = path.find('/', begin);
    std::string component = path.substr(
        begin, end == std::string::npos ? std::string::npos : end - begin);
    if (component == "..") {
      return true;
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return false;
}

inline std::string relative_path(const char *manifest_path,
                                 const std::string &file_name) {
  if (file_name.empty() || file_name[0] == '/' ||
      has_parent_component(file_name)) {
    throw std::runtime_error("invalid mini decoder weight data path: " +
                             file_name);
  }

  std::string path(manifest_path);
  size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) {
    return file_name;
  }
  return path.substr(0, slash + 1) + file_name;
}

inline size_t skip_ws(const std::string &text, size_t position) {
  while (position < text.size() &&
         std::isspace(static_cast<unsigned char>(text[position])) != 0) {
    ++position;
  }
  return position;
}

inline size_t field_position(const std::string &text, const char *field) {
  std::string needle = std::string("\"") + field + "\"";
  size_t position = text.find(needle);
  if (position == std::string::npos) {
    throw std::runtime_error("missing mini decoder weight manifest field: " +
                             std::string(field));
  }
  position = text.find(':', position + needle.size());
  if (position == std::string::npos) {
    throw std::runtime_error("invalid mini decoder weight manifest field: " +
                             std::string(field));
  }
  return skip_ws(text, position + 1);
}

inline std::string read_string_field(const std::string &text,
                                     const char *field) {
  size_t position = field_position(text, field);
  if (position >= text.size() || text[position] != '"') {
    throw std::runtime_error("mini decoder weight manifest field is not a "
                             "string: " +
                             std::string(field));
  }
  ++position;
  size_t end = text.find('"', position);
  if (end == std::string::npos) {
    throw std::runtime_error("unterminated mini decoder weight manifest "
                             "string: " +
                             std::string(field));
  }
  return text.substr(position, end - position);
}

inline uint32_t read_u32_field(const std::string &text, const char *field) {
  size_t position = field_position(text, field);
  uint64_t value = 0;
  bool saw_digit = false;
  while (position < text.size() &&
         std::isdigit(static_cast<unsigned char>(text[position])) != 0) {
    saw_digit = true;
    value = value * 10 + static_cast<uint64_t>(text[position] - '0');
    if (value > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("mini decoder weight manifest integer is too "
                               "large: " +
                               std::string(field));
    }
    ++position;
  }
  if (!saw_digit) {
    throw std::runtime_error("mini decoder weight manifest field is not an "
                             "integer: " +
                             std::string(field));
  }
  return static_cast<uint32_t>(value);
}

inline std::vector<uint32_t> read_u32_array_field(const std::string &text,
                                                  const char *field) {
  size_t position = field_position(text, field);
  if (position >= text.size() || text[position] != '[') {
    throw std::runtime_error("mini decoder weight manifest field is not an "
                             "array: " +
                             std::string(field));
  }
  ++position;
  std::vector<uint32_t> values;
  while (true) {
    position = skip_ws(text, position);
    if (position >= text.size()) {
      throw std::runtime_error("unterminated mini decoder weight manifest "
                               "array: " +
                               std::string(field));
    }
    if (text[position] == ']') {
      return values;
    }
    uint64_t value = 0;
    bool saw_digit = false;
    while (position < text.size() &&
           std::isdigit(static_cast<unsigned char>(text[position])) != 0) {
      saw_digit = true;
      value = value * 10 + static_cast<uint64_t>(text[position] - '0');
      if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("mini decoder weight manifest integer is too "
                                 "large: " +
                                 std::string(field));
      }
      ++position;
    }
    if (!saw_digit) {
      throw std::runtime_error("mini decoder weight manifest array contains a "
                               "non-integer: " +
                               std::string(field));
    }
    values.push_back(static_cast<uint32_t>(value));
    position = skip_ws(text, position);
    if (position < text.size() && text[position] == ',') {
      ++position;
      continue;
    }
    if (position < text.size() && text[position] == ']') {
      return values;
    }
    throw std::runtime_error("mini decoder weight manifest array is invalid: " +
                             std::string(field));
  }
}

inline float read_f32_field(const std::string &text, const char *field) {
  size_t position = field_position(text, field);
  const char *begin = text.c_str() + position;
  char *end = nullptr;
  errno = 0;
  float value = std::strtof(begin, &end);
  if (begin == end || errno == ERANGE || value <= 0.0f) {
    throw std::runtime_error("mini decoder weight manifest field is not a "
                             "positive float: " +
                             std::string(field));
  }
  return value;
}

inline uint64_t read_u64_field(const std::string &text, size_t begin,
                               size_t end, const char *field,
                               const std::string &tensor_name) {
  std::string needle = std::string("\"") + field + "\"";
  size_t position = text.find(needle, begin);
  if (position == std::string::npos || position >= end) {
    throw std::runtime_error("mini decoder weight tensor '" + tensor_name +
                             "' is missing field: " + field);
  }
  position = text.find(':', position + needle.size());
  if (position == std::string::npos || position >= end) {
    throw std::runtime_error("invalid mini decoder weight tensor '" +
                             tensor_name + "' field: " + field);
  }
  position = skip_ws(text, position + 1);
  uint64_t value = 0;
  bool saw_digit = false;
  while (position < end &&
         std::isdigit(static_cast<unsigned char>(text[position])) != 0) {
    saw_digit = true;
    uint64_t digit = static_cast<uint64_t>(text[position] - '0');
    if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
      throw std::runtime_error("mini decoder weight tensor '" + tensor_name +
                               "' field is too large: " + field);
    }
    value = value * 10 + digit;
    ++position;
  }
  if (!saw_digit) {
    throw std::runtime_error("mini decoder weight tensor '" + tensor_name +
                             "' field is not an integer: " + field);
  }
  return value;
}

inline bool tensor_name_at(const std::string &text, size_t position,
                           const char *expected) {
  position = text.find(':', position);
  if (position == std::string::npos) {
    return false;
  }
  position = skip_ws(text, position + 1);
  if (position >= text.size() || text[position] != '"') {
    return false;
  }
  ++position;
  size_t end = text.find('"', position);
  if (end == std::string::npos) {
    return false;
  }
  return text.substr(position, end - position) == expected;
}

inline std::pair<size_t, size_t> tensor_bounds(const std::string &text,
                                               const char *name) {
  std::string needle = "\"name\"";
  size_t name_position = std::string::npos;
  size_t search = 0;
  size_t matches = 0;
  while (true) {
    size_t candidate = text.find(needle, search);
    if (candidate == std::string::npos) {
      break;
    }
    if (tensor_name_at(text, candidate + needle.size(), name)) {
      name_position = candidate;
      ++matches;
    }
    search = candidate + needle.size();
  }
  if (matches == 0) {
    throw std::runtime_error("missing mini decoder weight tensor: " +
                             std::string(name));
  }
  if (matches > 1) {
    throw std::runtime_error("duplicate mini decoder weight tensor: " +
                             std::string(name));
  }
  size_t begin = text.rfind('{', name_position);
  size_t end = text.find('}', name_position);
  if (begin == std::string::npos || end == std::string::npos || begin > end) {
    throw std::runtime_error("invalid mini decoder weight tensor object: " +
                             std::string(name));
  }
  return {begin, end};
}

inline std::vector<TensorSpec> tensor_specs(const DecoderLayerShape &shape) {
  size_t hidden = shape.hidden;
  size_t q_dim = shape.q_dim();
  size_t kv_dim = shape.kv_dim();
  size_t intermediate = shape.intermediate;
  return {
      {"attention_norm_weight", hidden,
       &DecoderLayerWeights::attention_norm_weight},
      {"mlp_norm_weight", hidden, &DecoderLayerWeights::mlp_norm_weight},
      {"q_weight", checked_multiply(q_dim, hidden, "q_weight"),
       &DecoderLayerWeights::q_weight},
      {"k_weight", checked_multiply(kv_dim, hidden, "k_weight"),
       &DecoderLayerWeights::k_weight},
      {"v_weight", checked_multiply(kv_dim, hidden, "v_weight"),
       &DecoderLayerWeights::v_weight},
      {"out_weight", checked_multiply(hidden, q_dim, "out_weight"),
       &DecoderLayerWeights::out_weight},
      {"gate_weight", checked_multiply(intermediate, hidden, "gate_weight"),
       &DecoderLayerWeights::gate_weight},
      {"up_weight", checked_multiply(intermediate, hidden, "up_weight"),
       &DecoderLayerWeights::up_weight},
      {"down_weight", checked_multiply(hidden, intermediate, "down_weight"),
       &DecoderLayerWeights::down_weight},
  };
}

inline std::vector<TailTensorSpec>
tail_tensor_specs(uint32_t hidden, uint32_t vocab, uint32_t tokens) {
  return {
      {"token_embeddings", checked_multiply(tokens, hidden, "token_embeddings"),
       &ModelTailWeights::token_embeddings},
      {"final_norm_weight", hidden, &ModelTailWeights::final_norm_weight},
      {"lm_head_weight", checked_multiply(vocab, hidden, "lm_head_weight"),
       &ModelTailWeights::lm_head_weight},
  };
}

inline void append_tensor(std::ofstream &data, std::ofstream &manifest,
                          const char *name, const std::vector<float> &values,
                          uint64_t *offset, bool *first) {
  if (!first || !offset) {
    throw std::invalid_argument("mini decoder weight writer state is null");
  }
  if (!data || !manifest) {
    throw std::runtime_error("mini decoder weight output stream is invalid");
  }
  if (!*first) {
    manifest << ",\n";
  }
  *first = false;
  manifest << "    {\"name\":\"" << name << "\",\"offset\":" << *offset
           << ",\"count\":" << values.size() << "}";
  size_t bytes = checked_multiply(values.size(), sizeof(float), name);
  if (bytes > 0) {
    data.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(bytes));
    if (!data) {
      throw std::runtime_error("failed to write mini decoder weight tensor: " +
                               std::string(name));
    }
  }
  if (*offset > std::numeric_limits<uint64_t>::max() - bytes) {
    throw std::overflow_error("mini decoder weight data offset overflows");
  }
  *offset += static_cast<uint64_t>(bytes);
}

inline void copy_tensor(const std::vector<unsigned char> &data, uint64_t offset,
                        size_t count, const std::string &name,
                        std::vector<float> *out) {
  size_t bytes = checked_multiply(count, sizeof(float), name.c_str());
  if (offset > std::numeric_limits<size_t>::max() ||
      bytes >
          std::numeric_limits<size_t>::max() - static_cast<size_t>(offset) ||
      static_cast<size_t>(offset) + bytes > data.size()) {
    throw std::runtime_error("mini decoder weight tensor '" + name +
                             "' is out of range");
  }
  out->resize(count);
  std::memcpy(out->data(), data.data() + static_cast<size_t>(offset), bytes);
}

} // namespace detail

inline DecoderLayerWeights
load_decoder_layer_weights(const char *manifest_path) {
  if (!manifest_path || manifest_path[0] == '\0') {
    throw std::invalid_argument("mini decoder weight manifest path is empty");
  }

  std::string manifest = detail::read_text_file(manifest_path);
  if (detail::read_u32_field(manifest, "version") != 1) {
    throw std::runtime_error(
        "unsupported mini decoder weight manifest version");
  }
  std::string format = detail::read_string_field(manifest, "format");
  if (format != "lrrt.mini_decoder_weights") {
    throw std::runtime_error(
        "unsupported mini decoder weight manifest format: " + format);
  }
  std::string dtype = detail::read_string_field(manifest, "dtype");
  if (dtype != "f32") {
    throw std::runtime_error("unsupported mini decoder weight dtype: " + dtype);
  }

  DecoderLayerWeights weights{};
  weights.shape = {
      detail::read_u32_field(manifest, "keys"),
      detail::read_u32_field(manifest, "hidden"),
      detail::read_u32_field(manifest, "heads"),
      0,
      detail::read_u32_field(manifest, "head_dim"),
      detail::read_u32_field(manifest, "intermediate"),
  };
  if (manifest.find("\"kv_heads\"") != std::string::npos) {
    weights.shape.kv_heads = detail::read_u32_field(manifest, "kv_heads");
  } else {
    weights.shape.kv_heads = weights.shape.heads;
  }
  if (manifest.find("\"rope_theta\"") != std::string::npos) {
    weights.rope_theta = detail::read_f32_field(manifest, "rope_theta");
  }
  detail::validate_shape(weights.shape);

  std::string data_file = detail::read_string_field(manifest, "data");
  std::vector<unsigned char> data =
      detail::read_file(detail::relative_path(manifest_path, data_file));

  uint64_t expected_size = 0;
  std::unordered_set<std::string> seen;
  for (const detail::TensorSpec &spec : detail::tensor_specs(weights.shape)) {
    auto [begin, end] = detail::tensor_bounds(manifest, spec.name);
    uint64_t offset =
        detail::read_u64_field(manifest, begin, end, "offset", spec.name);
    uint64_t count =
        detail::read_u64_field(manifest, begin, end, "count", spec.name);
    if (count != spec.count) {
      throw std::runtime_error("mini decoder weight tensor '" +
                               std::string(spec.name) +
                               "' has unexpected count");
    }
    uint64_t bytes = static_cast<uint64_t>(
        detail::checked_multiply(spec.count, sizeof(float), spec.name));
    if (offset > std::numeric_limits<uint64_t>::max() - bytes) {
      throw std::runtime_error("mini decoder weight tensor '" +
                               std::string(spec.name) + "' overflows offset");
    }
    expected_size = std::max(expected_size, offset + bytes);
    if (!seen.insert(spec.name).second) {
      throw std::runtime_error("duplicate mini decoder weight tensor spec");
    }
    detail::copy_tensor(data, offset, spec.count, spec.name,
                        &(weights.*(spec.field)));
  }
  if (expected_size != data.size()) {
    throw std::runtime_error("mini decoder weight data size mismatch");
  }

  return weights;
}

inline ModelTailWeights load_model_tail_weights(const char *manifest_path) {
  if (!manifest_path || manifest_path[0] == '\0') {
    throw std::invalid_argument("mini model tail manifest path is empty");
  }

  std::string manifest = detail::read_text_file(manifest_path);
  if (detail::read_u32_field(manifest, "version") != 1) {
    throw std::runtime_error("unsupported mini model tail manifest version");
  }
  std::string format = detail::read_string_field(manifest, "format");
  if (format != "lrrt.mini_model_tail_weights") {
    throw std::runtime_error("unsupported mini model tail manifest format: " +
                             format);
  }
  std::string dtype = detail::read_string_field(manifest, "dtype");
  if (dtype != "f32") {
    throw std::runtime_error("unsupported mini model tail dtype: " + dtype);
  }

  ModelTailWeights weights{};
  weights.hidden = detail::read_u32_field(manifest, "hidden");
  weights.vocab = detail::read_u32_field(manifest, "vocab");
  weights.token_ids = detail::read_u32_array_field(manifest, "token_ids");
  if (weights.hidden == 0 || weights.vocab == 0 || weights.token_ids.empty()) {
    throw std::runtime_error("mini model tail manifest has invalid shape");
  }
  for (uint32_t token_id : weights.token_ids) {
    if (token_id >= weights.vocab) {
      throw std::runtime_error("mini model tail manifest has invalid token id");
    }
  }
  std::string data_file = detail::read_string_field(manifest, "data");
  std::vector<unsigned char> data =
      detail::read_file(detail::relative_path(manifest_path, data_file));

  uint64_t expected_size = 0;
  std::vector<detail::TailTensorSpec> specs = detail::tail_tensor_specs(
      weights.hidden, weights.vocab,
      static_cast<uint32_t>(weights.token_ids.size()));
  for (const detail::TailTensorSpec &spec : specs) {
    auto [begin, end] = detail::tensor_bounds(manifest, spec.name);
    uint64_t offset =
        detail::read_u64_field(manifest, begin, end, "offset", spec.name);
    uint64_t count =
        detail::read_u64_field(manifest, begin, end, "count", spec.name);
    if (count != spec.count) {
      throw std::runtime_error("mini model tail tensor '" +
                               std::string(spec.name) +
                               "' has unexpected count");
    }
    uint64_t bytes = static_cast<uint64_t>(
        detail::checked_multiply(spec.count, sizeof(float), spec.name));
    if (offset > std::numeric_limits<uint64_t>::max() - bytes) {
      throw std::runtime_error("mini model tail tensor '" +
                               std::string(spec.name) + "' overflows offset");
    }
    expected_size = std::max(expected_size, offset + bytes);
    detail::copy_tensor(data, offset, spec.count, spec.name,
                        &(weights.*(spec.field)));
  }
  if (expected_size != data.size()) {
    throw std::runtime_error("mini model tail data size mismatch");
  }

  return weights;
}

inline void copy_decoder_layer_inputs(DecoderLayer &executor,
                                      const std::vector<float> &hidden_states,
                                      const DecoderLayerWeights &weights,
                                      const std::vector<float> &cos,
                                      const std::vector<float> &sin) {
  executor.copy_inputs(
      hidden_states, weights.attention_norm_weight, weights.mlp_norm_weight,
      weights.q_weight, weights.k_weight, weights.v_weight, weights.out_weight,
      weights.gate_weight, weights.up_weight, weights.down_weight, cos, sin);
}

inline void write_decoder_layer_weights(const char *manifest_path,
                                        const char *data_file_name,
                                        const DecoderLayerWeights &weights) {
  if (!manifest_path || manifest_path[0] == '\0' || !data_file_name ||
      data_file_name[0] == '\0') {
    throw std::invalid_argument("mini decoder weight output path is empty");
  }
  detail::validate_shape(weights.shape);
  if (weights.rope_theta <= 0.0f ||
      weights.attention_norm_weight.size() != weights.shape.hidden ||
      weights.mlp_norm_weight.size() != weights.shape.hidden ||
      weights.q_weight.size() !=
          static_cast<size_t>(weights.shape.q_dim()) * weights.shape.hidden ||
      weights.k_weight.size() !=
          static_cast<size_t>(weights.shape.kv_dim()) * weights.shape.hidden ||
      weights.v_weight.size() !=
          static_cast<size_t>(weights.shape.kv_dim()) * weights.shape.hidden ||
      weights.out_weight.size() !=
          static_cast<size_t>(weights.shape.hidden) * weights.shape.q_dim() ||
      weights.gate_weight.size() !=
          static_cast<size_t>(weights.shape.intermediate) *
              weights.shape.hidden ||
      weights.up_weight.size() !=
          static_cast<size_t>(weights.shape.intermediate) *
              weights.shape.hidden ||
      weights.down_weight.size() != static_cast<size_t>(weights.shape.hidden) *
                                        weights.shape.intermediate) {
    throw std::runtime_error("mini decoder weights have invalid tensor sizes");
  }

  std::string data_path =
      detail::relative_path(manifest_path, std::string(data_file_name));
  std::ofstream data(data_path, std::ios::binary | std::ios::trunc);
  if (!data) {
    throw std::runtime_error(
        "failed to create mini decoder weight data file: " + data_path);
  }
  std::ofstream manifest(manifest_path, std::ios::binary | std::ios::trunc);
  if (!manifest) {
    throw std::runtime_error("failed to create mini decoder weight manifest: " +
                             std::string(manifest_path));
  }

  manifest << "{\n"
           << "  \"format\": \"lrrt.mini_decoder_weights\",\n"
           << "  \"version\": 1,\n"
           << "  \"dtype\": \"f32\",\n"
           << "  \"data\": \"" << data_file_name << "\",\n"
           << "  \"keys\": " << weights.shape.keys << ",\n"
           << "  \"hidden\": " << weights.shape.hidden << ",\n"
           << "  \"heads\": " << weights.shape.heads << ",\n"
           << "  \"kv_heads\": " << weights.shape.kv_heads << ",\n"
           << "  \"head_dim\": " << weights.shape.head_dim << ",\n"
           << "  \"intermediate\": " << weights.shape.intermediate << ",\n"
           << "  \"rope_theta\": " << weights.rope_theta << ",\n"
           << "  \"tensors\": [\n";
  uint64_t offset = 0;
  bool first = true;
  detail::append_tensor(data, manifest, "attention_norm_weight",
                        weights.attention_norm_weight, &offset, &first);
  detail::append_tensor(data, manifest, "mlp_norm_weight",
                        weights.mlp_norm_weight, &offset, &first);
  detail::append_tensor(data, manifest, "q_weight", weights.q_weight, &offset,
                        &first);
  detail::append_tensor(data, manifest, "k_weight", weights.k_weight, &offset,
                        &first);
  detail::append_tensor(data, manifest, "v_weight", weights.v_weight, &offset,
                        &first);
  detail::append_tensor(data, manifest, "out_weight", weights.out_weight,
                        &offset, &first);
  detail::append_tensor(data, manifest, "gate_weight", weights.gate_weight,
                        &offset, &first);
  detail::append_tensor(data, manifest, "up_weight", weights.up_weight, &offset,
                        &first);
  detail::append_tensor(data, manifest, "down_weight", weights.down_weight,
                        &offset, &first);
  manifest << "\n  ]\n}\n";
  if (!manifest) {
    throw std::runtime_error("failed to write mini decoder weight manifest");
  }
}

inline void write_model_tail_weights(const char *manifest_path,
                                     const char *data_file_name,
                                     const ModelTailWeights &weights) {
  if (!manifest_path || manifest_path[0] == '\0' || !data_file_name ||
      data_file_name[0] == '\0') {
    throw std::invalid_argument("mini model tail output path is empty");
  }
  if (weights.hidden == 0 || weights.vocab == 0 || weights.token_ids.empty() ||
      weights.token_embeddings.size() !=
          static_cast<size_t>(weights.token_ids.size()) * weights.hidden ||
      weights.final_norm_weight.size() != weights.hidden ||
      weights.lm_head_weight.size() !=
          static_cast<size_t>(weights.vocab) * weights.hidden) {
    throw std::runtime_error("mini model tail weights have invalid sizes");
  }
  for (uint32_t token_id : weights.token_ids) {
    if (token_id >= weights.vocab) {
      throw std::runtime_error("mini model tail weights have invalid token id");
    }
  }

  std::string data_path =
      detail::relative_path(manifest_path, std::string(data_file_name));
  std::ofstream data(data_path, std::ios::binary | std::ios::trunc);
  if (!data) {
    throw std::runtime_error("failed to create mini model tail data file: " +
                             data_path);
  }
  std::ofstream manifest(manifest_path, std::ios::binary | std::ios::trunc);
  if (!manifest) {
    throw std::runtime_error("failed to create mini model tail manifest: " +
                             std::string(manifest_path));
  }

  manifest << "{\n"
           << "  \"format\": \"lrrt.mini_model_tail_weights\",\n"
           << "  \"version\": 1,\n"
           << "  \"dtype\": \"f32\",\n"
           << "  \"data\": \"" << data_file_name << "\",\n"
           << "  \"hidden\": " << weights.hidden << ",\n"
           << "  \"vocab\": " << weights.vocab << ",\n"
           << "  \"token_ids\": [";
  for (size_t i = 0; i < weights.token_ids.size(); ++i) {
    if (i != 0) {
      manifest << ", ";
    }
    manifest << weights.token_ids[i];
  }
  manifest << "],\n"
           << "  \"tensors\": [\n";
  uint64_t offset = 0;
  bool first = true;
  detail::append_tensor(data, manifest, "token_embeddings",
                        weights.token_embeddings, &offset, &first);
  detail::append_tensor(data, manifest, "final_norm_weight",
                        weights.final_norm_weight, &offset, &first);
  detail::append_tensor(data, manifest, "lm_head_weight",
                        weights.lm_head_weight, &offset, &first);
  manifest << "\n  ]\n}\n";
  if (!manifest) {
    throw std::runtime_error("failed to write mini model tail manifest");
  }
}

} // namespace lrrt::executor::triton::mini

#endif // LRRT_EXECUTOR_TRITON_MINI_DECODER_WEIGHTS_HPP_
