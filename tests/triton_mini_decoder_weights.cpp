#include "mini_decoder_weights.hpp"

#include <math.h>
#include <stdio.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using lrrt::executor::triton::mini::DecoderLayerShape;
using lrrt::executor::triton::mini::DecoderLayerWeights;

struct TensorInput {
  const char *name;
  size_t count;
  float seed;
};

std::string test_path(const char *name) {
  return std::string("/tmp/lrrt_mini_decoder_weights_") +
         std::to_string((long long)getpid()) + "_" + name;
}

void write_file(const std::string &path, const std::string &data) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to create test file: " + path);
  }
  file.write(data.data(), static_cast<std::streamsize>(data.size()));
}

void write_binary(const std::string &path, const std::vector<float> &values) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to create test binary: " + path);
  }
  file.write(reinterpret_cast<const char *>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
}

std::vector<TensorInput> tensors(const DecoderLayerShape &shape) {
  const size_t q_dim = shape.heads * shape.head_dim;
  const size_t kv_dim = shape.kv_heads * shape.head_dim;
  return {
      {"attention_norm_weight", shape.hidden, 1.0f},
      {"mlp_norm_weight", shape.hidden, 2.0f},
      {"q_weight", q_dim * shape.hidden, 3.0f},
      {"k_weight", kv_dim * shape.hidden, 4.0f},
      {"v_weight", kv_dim * shape.hidden, 5.0f},
      {"out_weight", shape.hidden * q_dim, 6.0f},
      {"gate_weight", shape.intermediate * shape.hidden, 7.0f},
      {"up_weight", shape.intermediate * shape.hidden, 8.0f},
      {"down_weight", shape.hidden * shape.intermediate, 9.0f},
  };
}

std::string manifest_for(const DecoderLayerShape &shape,
                         const std::string &data_file,
                         const char *skip_tensor = nullptr,
                         const char *dtype = "f32") {
  std::string manifest = "{\n"
                         "  \"format\": \"lrrt.mini_decoder_weights\",\n"
                         "  \"version\": 1,\n"
                         "  \"dtype\": \"" +
                         std::string(dtype) +
                         "\",\n"
                         "  \"data\": \"" +
                         data_file +
                         "\",\n"
                         "  \"keys\": " +
                         std::to_string(shape.keys) +
                         ",\n"
                         "  \"hidden\": " +
                         std::to_string(shape.hidden) +
                         ",\n"
                         "  \"heads\": " +
                         std::to_string(shape.heads) +
                         ",\n"
                         "  \"kv_heads\": " +
                         std::to_string(shape.kv_heads) +
                         ",\n"
                         "  \"head_dim\": " +
                         std::to_string(shape.head_dim) +
                         ",\n"
                         "  \"intermediate\": " +
                         std::to_string(shape.intermediate) +
                         ",\n"
                         "  \"tensors\": [\n";

  size_t offset = 0;
  bool first = true;
  for (const TensorInput &tensor : tensors(shape)) {
    if (skip_tensor && std::string(skip_tensor) == tensor.name) {
      offset += tensor.count * sizeof(float);
      continue;
    }
    if (!first) {
      manifest += ",\n";
    }
    first = false;
    manifest += "    {\"name\":\"" + std::string(tensor.name) +
                "\",\"offset\":" + std::to_string(offset) +
                ",\"count\":" + std::to_string(tensor.count) + "}";
    offset += tensor.count * sizeof(float);
  }
  manifest += "\n  ]\n}\n";
  return manifest;
}

std::vector<float> data_for(const DecoderLayerShape &shape) {
  std::vector<float> values;
  for (const TensorInput &tensor : tensors(shape)) {
    for (size_t i = 0; i < tensor.count; ++i) {
      values.push_back(tensor.seed + 0.001f * static_cast<float>(i));
    }
  }
  return values;
}

void expect_close(float actual, float expected, const char *label) {
  if (fabsf(actual - expected) > 0.00001f) {
    throw std::runtime_error(std::string(label) + " mismatch");
  }
}

template <typename Function>
void expect_throw(Function function, const char *needle, const char *label) {
  try {
    function();
  } catch (const std::exception &error) {
    if (std::string(error.what()).find(needle) != std::string::npos) {
      return;
    }
    fprintf(stderr, "%s: wrong error: %s\n", label, error.what());
    throw;
  }
  throw std::runtime_error(std::string(label) + " did not throw");
}

void test_load_weights(void) {
  DecoderLayerShape shape{4, 8, 2, 1, 4, 12};
  std::string data_path = test_path("weights.bin");
  std::string manifest_path = test_path("manifest.json");
  write_binary(data_path, data_for(shape));
  write_file(manifest_path, manifest_for(shape, "lrrt_mini_decoder_weights_"
                                                "weights.bin"));

  DecoderLayerWeights weights =
      lrrt::executor::triton::mini::load_decoder_layer_weights(
          manifest_path.c_str());
  if (weights.shape.keys != shape.keys ||
      weights.shape.hidden != shape.hidden ||
      weights.shape.heads != shape.heads ||
      weights.shape.kv_heads != shape.kv_heads ||
      weights.shape.head_dim != shape.head_dim ||
      weights.shape.intermediate != shape.intermediate) {
    throw std::runtime_error("loaded shape mismatch");
  }
  expect_close(weights.attention_norm_weight.front(), 1.0f,
               "attention_norm_weight");
  expect_close(weights.mlp_norm_weight.back(), 2.007f, "mlp_norm_weight");
  expect_close(weights.q_weight.front(), 3.0f, "q_weight");
  expect_close(weights.down_weight.back(), 9.095f, "down_weight");
}

void test_validation_errors(void) {
  DecoderLayerShape shape{4, 8, 2, 1, 4, 12};
  std::string data_path = test_path("bad_weights.bin");
  std::string manifest_path = test_path("bad_manifest.json");
  write_binary(data_path, data_for(shape));

  write_file(manifest_path,
             manifest_for(shape, "lrrt_mini_decoder_weights_bad_weights.bin",
                          "down_weight"));
  expect_throw(
      [&] {
        lrrt::executor::triton::mini::load_decoder_layer_weights(
            manifest_path.c_str());
      },
      "missing mini decoder weight tensor: down_weight",
      "missing tensor validation");

  write_file(manifest_path, manifest_for(shape, "../bad_weights.bin"));
  expect_throw(
      [&] {
        lrrt::executor::triton::mini::load_decoder_layer_weights(
            manifest_path.c_str());
      },
      "invalid mini decoder weight data path", "data path validation");

  write_file(manifest_path,
             manifest_for(shape, "lrrt_mini_decoder_weights_bad_weights.bin",
                          nullptr, "bf16"));
  expect_throw(
      [&] {
        lrrt::executor::triton::mini::load_decoder_layer_weights(
            manifest_path.c_str());
      },
      "unsupported mini decoder weight dtype", "dtype validation");
}

} // namespace

int main(void) {
  try {
    test_load_weights();
    test_validation_errors();
    printf("triton_mini_decoder_weights: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
