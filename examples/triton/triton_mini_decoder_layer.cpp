#include "mini_decoder_layer.hpp"

static void reference_rmsnorm(const std::vector<float> &x,
                              const std::vector<float> &weight, uint32_t rows,
                              uint32_t hidden, std::vector<float> &out) {
  for (uint32_t row = 0; row < rows; ++row) {
    float sum_square = 0.0f;
    for (uint32_t col = 0; col < hidden; ++col) {
      float value = x[row * hidden + col];
      sum_square += value * value;
    }
    float scale = 1.0f / sqrtf(sum_square / (float)hidden + 1.0e-5f);
    for (uint32_t col = 0; col < hidden; ++col) {
      uint32_t index = row * hidden + col;
      out[index] = x[index] * scale * weight[col];
    }
  }
}

static void reference_rmsnorm_vector(const std::vector<float> &x,
                                     const std::vector<float> &weight,
                                     std::vector<float> &out) {
  float sum_square = 0.0f;
  for (float value : x) {
    sum_square += value * value;
  }
  float scale = 1.0f / sqrtf(sum_square / (float)x.size() + 1.0e-5f);
  for (uint32_t i = 0; i < x.size(); ++i) {
    out[i] = x[i] * scale * weight[i];
  }
}

static void reference_projection(const std::vector<float> &x,
                                 const std::vector<float> &weight,
                                 uint32_t input, uint32_t row,
                                 std::vector<float> &out) {
  for (uint32_t projection = 0; projection < out.size(); ++projection) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < input; ++col) {
      sum += x[row * input + col] * weight[projection * input + col];
    }
    out[projection] = sum;
  }
}

static void reference_vector_projection(const std::vector<float> &x,
                                        const std::vector<float> &weight,
                                        uint32_t input,
                                        std::vector<float> &out) {
  for (uint32_t projection = 0; projection < out.size(); ++projection) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < input; ++col) {
      sum += x[col] * weight[projection * input + col];
    }
    out[projection] = sum;
  }
}

static void run_case(lrrt::Device &device, uint32_t keys, uint32_t hidden,
                     uint32_t heads, uint32_t head_dim, uint32_t intermediate,
                     uint32_t valid_keys) {
  const uint32_t qkv_dim = heads * head_dim;
  std::vector<float> hidden_states(keys * hidden);
  std::vector<float> attention_norm_weight(hidden);
  std::vector<float> mlp_norm_weight(hidden);
  std::vector<float> attention_norm_states(keys * hidden, 0.0f);
  std::vector<float> q_weight(qkv_dim * hidden);
  std::vector<float> k_weight(qkv_dim * hidden);
  std::vector<float> v_weight(qkv_dim * hidden);
  std::vector<float> out_weight(hidden * qkv_dim);
  std::vector<float> gate_weight(intermediate * hidden);
  std::vector<float> up_weight(intermediate * hidden);
  std::vector<float> down_weight(hidden * intermediate);
  std::vector<float> cos(keys * (head_dim / 2));
  std::vector<float> sin(keys * (head_dim / 2));
  std::vector<float> out(hidden, 0.0f);
  const uint32_t half = head_dim / 2;
  const float scale = 1.0f / sqrtf((float)head_dim);

  for (uint32_t i = 0; i < hidden_states.size(); ++i) {
    hidden_states[i] =
        0.03125f * (float)((int32_t)((i * 5 + i / hidden) % 31) - 15);
  }
  for (uint32_t i = 0; i < hidden; ++i) {
    attention_norm_weight[i] = 1.0f + 0.001f * (float)(i % 29);
    mlp_norm_weight[i] = 1.0f - 0.001f * (float)(i % 17);
  }
  lrrt::executor::triton::mini::fill_projection_weight(q_weight, hidden, 1);
  lrrt::executor::triton::mini::fill_projection_weight(k_weight, hidden, 2);
  lrrt::executor::triton::mini::fill_projection_weight(v_weight, hidden, 3);
  lrrt::executor::triton::mini::fill_projection_weight(out_weight, qkv_dim, 4);
  lrrt::executor::triton::mini::fill_projection_weight(gate_weight, hidden, 5);
  lrrt::executor::triton::mini::fill_projection_weight(up_weight, hidden, 6);
  lrrt::executor::triton::mini::fill_projection_weight(down_weight,
                                                       intermediate, 7);
  for (uint32_t token = 0; token < keys; ++token) {
    for (uint32_t frequency = 0; frequency < half; ++frequency) {
      float exponent = -2.0f * (float)frequency / (float)head_dim;
      float inverse_frequency = powf(10000.0f, exponent);
      float angle = (float)(token + 7) * inverse_frequency;
      uint32_t index = token * half + frequency;
      cos[index] = cosf(angle);
      sin[index] = sinf(angle);
    }
  }

  lrrt::executor::triton::mini::DecoderLayer executor(
      device, keys, hidden, heads, head_dim, intermediate);
  executor.copy_inputs(hidden_states, attention_norm_weight, mlp_norm_weight,
                       q_weight, k_weight, v_weight, out_weight, gate_weight,
                       up_weight, down_weight, cos, sin);
  executor.run(valid_keys);
  executor.synchronize();
  executor.copy_output(out);

  reference_rmsnorm(hidden_states, attention_norm_weight, keys, hidden,
                    attention_norm_states);

  const uint32_t query_position = valid_keys - 1;
  std::vector<float> q(qkv_dim);
  std::vector<float> k(keys * qkv_dim, 0.0f);
  std::vector<float> v(keys * qkv_dim, 0.0f);
  reference_projection(attention_norm_states, q_weight, hidden, query_position,
                       q);
  for (uint32_t row = 0; row < valid_keys; ++row) {
    std::vector<float> k_row(qkv_dim);
    std::vector<float> v_row(qkv_dim);
    reference_projection(attention_norm_states, k_weight, hidden, row, k_row);
    reference_projection(attention_norm_states, v_weight, hidden, row, v_row);
    for (uint32_t col = 0; col < qkv_dim; ++col) {
      k[row * qkv_dim + col] = k_row[col];
      v[row * qkv_dim + col] = v_row[col];
    }
  }

  std::vector<float> attention_out(qkv_dim, 0.0f);
  for (uint32_t head = 0; head < heads; ++head) {
    const uint32_t head_offset = head * head_dim;
    std::vector<float> q_rope(head_dim);
    for (uint32_t offset = 0; offset < head_dim; ++offset) {
      uint32_t frequency = offset % half;
      uint32_t partner = offset < half ? offset + half : offset - half;
      uint32_t index = head_offset + offset;
      uint32_t partner_index = head_offset + partner;
      float rotated = offset < half ? -q[partner_index] : q[partner_index];
      q_rope[offset] = q[index] * cos[query_position * half + frequency] +
                       rotated * sin[query_position * half + frequency];
    }

    std::vector<float> reference_scores(keys, 0.0f);
    for (uint32_t row = 0; row < valid_keys; ++row) {
      for (uint32_t col = 0; col < head_dim; ++col) {
        uint32_t frequency = col % half;
        uint32_t partner = col < half ? col + half : col - half;
        uint32_t base = row * qkv_dim + head_offset;
        float rotated = col < half ? -k[base + partner] : k[base + partner];
        float k_rope = k[base + col] * cos[row * half + frequency] +
                       rotated * sin[row * half + frequency];
        reference_scores[row] += q_rope[col] * k_rope;
      }
      reference_scores[row] *= scale;
    }

    float max_score = reference_scores[0];
    for (uint32_t row = 1; row < valid_keys; ++row) {
      max_score = fmaxf(max_score, reference_scores[row]);
    }
    float denominator = 0.0f;
    for (uint32_t row = 0; row < valid_keys; ++row) {
      denominator += expf(reference_scores[row] - max_score);
    }

    std::vector<float> reference_probs(keys, 0.0f);
    for (uint32_t row = 0; row < valid_keys; ++row) {
      reference_probs[row] =
          expf(reference_scores[row] - max_score) / denominator;
    }

    for (uint32_t col = 0; col < head_dim; ++col) {
      for (uint32_t row = 0; row < valid_keys; ++row) {
        attention_out[head_offset + col] +=
            reference_probs[row] * v[row * qkv_dim + head_offset + col];
      }
    }
  }

  std::vector<float> projected_attention(hidden, 0.0f);
  reference_vector_projection(attention_out, out_weight, qkv_dim,
                              projected_attention);

  std::vector<float> attention_residual(hidden, 0.0f);
  for (uint32_t col = 0; col < hidden; ++col) {
    attention_residual[col] =
        hidden_states[query_position * hidden + col] + projected_attention[col];
  }

  std::vector<float> mlp_norm_hidden(hidden, 0.0f);
  std::vector<float> gate(intermediate, 0.0f);
  std::vector<float> up(intermediate, 0.0f);
  std::vector<float> activated(intermediate, 0.0f);
  std::vector<float> projected_mlp(hidden, 0.0f);
  reference_rmsnorm_vector(attention_residual, mlp_norm_weight,
                           mlp_norm_hidden);
  reference_vector_projection(mlp_norm_hidden, gate_weight, hidden, gate);
  reference_vector_projection(mlp_norm_hidden, up_weight, hidden, up);
  for (uint32_t i = 0; i < intermediate; ++i) {
    activated[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
  }
  reference_vector_projection(activated, down_weight, intermediate,
                              projected_mlp);

  float max_diff = 0.0f;
  uint32_t max_index = 0;
  float max_expected = 0.0f;
  float max_actual = 0.0f;
  for (uint32_t col = 0; col < hidden; ++col) {
    float expected = attention_residual[col] + projected_mlp[col];
    float diff = fabsf(out[col] - expected);
    if (diff > max_diff) {
      max_diff = diff;
      max_index = col;
      max_expected = expected;
      max_actual = out[col];
    }
  }
  if (max_diff > 0.02f) {
    fprintf(stderr,
            "triton_mini_decoder_layer keys=%u hidden=%u heads=%u head_dim=%u "
            "intermediate=%u valid_keys=%u mismatch at %u: actual=%f "
            "expected=%f diff=%f\n",
            keys, hidden, heads, head_dim, intermediate, valid_keys, max_index,
            max_actual, max_expected, max_diff);
    throw std::runtime_error("triton_mini_decoder_layer result mismatch");
  }
}

#ifndef LRRT_TRITON_MINI_DECODER_LAYER_NO_MAIN
int main(void) {
  try {
    lrrt::Runtime runtime;
    uint32_t count = runtime.device_count();
    printf("devices: %u\n", count);
    if (count == 0) {
      return 0;
    }

    lrrt::Device device = runtime.open_device(0);
    printf("opened device: %u\n", device.index());
    run_case(device, 16, 768, 1, 64, 2048, 7);
    run_case(device, 32, 768, 2, 64, 2048, 19);
    run_case(device, 64, 1024, 2, 128, 3072, 33);

    printf("triton_mini_decoder_layer: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
#endif
