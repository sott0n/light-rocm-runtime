#include "mini_decoder_weights.hpp"

#include <limits>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

namespace {

uint32_t parse_u32(const char *text, const char *label) {
  if (!text || text[0] == '\0') {
    throw std::invalid_argument(std::string(label) + " is empty");
  }
  char *end = nullptr;
  unsigned long value = strtoul(text, &end, 10);
  if (!end || *end != '\0' || value == 0 ||
      value > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(std::string(label) +
                                " must be a positive uint32");
  }
  return static_cast<uint32_t>(value);
}

void fill_norm_weights(std::vector<float> &attention, std::vector<float> &mlp) {
  for (uint32_t i = 0; i < attention.size(); ++i) {
    attention[i] = 1.0f + 0.001f * (float)(i % 29);
    mlp[i] = 1.0f - 0.001f * (float)(i % 17);
  }
}

lrrt::executor::triton::mini::DecoderLayerWeights
make_weights(const lrrt::executor::triton::mini::DecoderLayerShape &shape) {
  const uint32_t qkv_dim = shape.qkv_dim();
  lrrt::executor::triton::mini::DecoderLayerWeights weights{};
  weights.shape = shape;
  weights.attention_norm_weight.resize(shape.hidden);
  weights.mlp_norm_weight.resize(shape.hidden);
  weights.q_weight.resize(qkv_dim * shape.hidden);
  weights.k_weight.resize(qkv_dim * shape.hidden);
  weights.v_weight.resize(qkv_dim * shape.hidden);
  weights.out_weight.resize(shape.hidden * qkv_dim);
  weights.gate_weight.resize(shape.intermediate * shape.hidden);
  weights.up_weight.resize(shape.intermediate * shape.hidden);
  weights.down_weight.resize(shape.hidden * shape.intermediate);

  fill_norm_weights(weights.attention_norm_weight, weights.mlp_norm_weight);
  lrrt::executor::triton::mini::fill_projection_weight(weights.q_weight,
                                                       shape.hidden, 1);
  lrrt::executor::triton::mini::fill_projection_weight(weights.k_weight,
                                                       shape.hidden, 2);
  lrrt::executor::triton::mini::fill_projection_weight(weights.v_weight,
                                                       shape.hidden, 3);
  lrrt::executor::triton::mini::fill_projection_weight(weights.out_weight,
                                                       qkv_dim, 4);
  lrrt::executor::triton::mini::fill_projection_weight(weights.gate_weight,
                                                       shape.hidden, 5);
  lrrt::executor::triton::mini::fill_projection_weight(weights.up_weight,
                                                       shape.hidden, 6);
  lrrt::executor::triton::mini::fill_projection_weight(weights.down_weight,
                                                       shape.intermediate, 7);
  return weights;
}

void print_usage(void) {
  fprintf(stderr,
          "usage: lrrt_triton_mini_decoder_weight_bundle <weights.json> "
          "<keys> <hidden> <heads> <head_dim> <intermediate>\n");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 7) {
      print_usage();
      return 1;
    }
    lrrt::executor::triton::mini::DecoderLayerShape shape{
        parse_u32(argv[2], "keys"),         parse_u32(argv[3], "hidden"),
        parse_u32(argv[4], "heads"),        parse_u32(argv[5], "head_dim"),
        parse_u32(argv[6], "intermediate"),
    };
    lrrt::executor::triton::mini::DecoderLayerWeights weights =
        make_weights(shape);
    lrrt::executor::triton::mini::write_decoder_layer_weights(
        argv[1], "weights.bin", weights);
    printf("wrote mini decoder weight bundle: %s\n", argv[1]);
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
