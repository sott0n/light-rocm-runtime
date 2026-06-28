#include "triton_executor.hpp"

#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#ifndef LRRT_TRITON_MINI_MLP_RMSNORM_MANIFEST
#define LRRT_TRITON_MINI_MLP_RMSNORM_MANIFEST "manifest.json"
#endif

#ifndef LRRT_TRITON_MINI_MLP_MATVEC_MANIFEST
#define LRRT_TRITON_MINI_MLP_MATVEC_MANIFEST "manifest.json"
#endif

#ifndef LRRT_TRITON_MINI_MLP_SILU_MUL_MANIFEST
#define LRRT_TRITON_MINI_MLP_SILU_MUL_MANIFEST "manifest.json"
#endif

#ifndef LRRT_TRITON_MINI_MLP_VECTOR_ADD_MANIFEST
#define LRRT_TRITON_MINI_MLP_VECTOR_ADD_MANIFEST "manifest.json"
#endif

namespace tex = lrrt::executor::triton;

static uint32_t select_block(uint32_t size) {
  if (size <= 1024) {
    return 1024;
  }
  if (size <= 2048) {
    return 2048;
  }
  return 4096;
}

class MiniMlpExecutor {
public:
  MiniMlpExecutor(lrrt::Device &device, uint32_t hidden, uint32_t intermediate)
      : queue_(device), bundles_(device), buffers_(device), hidden_(hidden),
        intermediate_(intermediate) {
    bundles_.add("rmsnorm", LRRT_TRITON_MINI_MLP_RMSNORM_MANIFEST,
                 "rmsnorm_fp32_" + std::to_string(select_block(hidden)));
    bundles_.add("gate_projection", LRRT_TRITON_MINI_MLP_MATVEC_MANIFEST,
                 "matvec_fp32_" + std::to_string(select_block(hidden)));
    bundles_.add("up_projection", LRRT_TRITON_MINI_MLP_MATVEC_MANIFEST,
                 "matvec_fp32_" + std::to_string(select_block(hidden)));
    bundles_.add("down_projection", LRRT_TRITON_MINI_MLP_MATVEC_MANIFEST,
                 "matvec_fp32_" + std::to_string(select_block(intermediate)));
    bundles_.add("silu_mul", LRRT_TRITON_MINI_MLP_SILU_MUL_MANIFEST,
                 "silu_mul");
    bundles_.add("residual_add", LRRT_TRITON_MINI_MLP_VECTOR_ADD_MANIFEST,
                 "vector_add");

    buffers_.allocate<float>("hidden", hidden);
    buffers_.allocate<float>("norm_weight", hidden);
    buffers_.allocate<float>("norm_hidden", hidden);
    buffers_.allocate<float>("gate_weight", intermediate * hidden);
    buffers_.allocate<float>("up_weight", intermediate * hidden);
    buffers_.allocate<float>("down_weight", hidden * intermediate);
    buffers_.allocate<float>("gate", intermediate);
    buffers_.allocate<float>("up", intermediate);
    buffers_.allocate<float>("activated", intermediate);
    buffers_.allocate<float>("projected", hidden);
    buffers_.allocate<float>("out", hidden);
  }

  void copy_inputs(const std::vector<float> &hidden,
                   const std::vector<float> &norm_weight,
                   const std::vector<float> &gate_weight,
                   const std::vector<float> &up_weight,
                   const std::vector<float> &down_weight) {
    if (hidden.size() != hidden_ || norm_weight.size() != hidden_ ||
        gate_weight.size() != intermediate_ * hidden_ ||
        up_weight.size() != intermediate_ * hidden_ ||
        down_weight.size() != hidden_ * intermediate_) {
      throw std::runtime_error("mini MLP input shape mismatch");
    }

    std::vector<float> hidden_zero(hidden_, 0.0f);
    std::vector<float> intermediate_zero(intermediate_, 0.0f);
    buffers_.copy_to("hidden", hidden);
    buffers_.copy_to("norm_weight", norm_weight);
    buffers_.copy_to("norm_hidden", hidden_zero);
    buffers_.copy_to("gate_weight", gate_weight);
    buffers_.copy_to("up_weight", up_weight);
    buffers_.copy_to("down_weight", down_weight);
    buffers_.copy_to("gate", intermediate_zero);
    buffers_.copy_to("up", intermediate_zero);
    buffers_.copy_to("activated", intermediate_zero);
    buffers_.copy_to("projected", hidden_zero);
    buffers_.copy_to("out", hidden_zero);
  }

  void run() {
    tex::launch(queue_, bundles_.get("rmsnorm"), 1,
                {
                    tex::arg("x", buffers_.ptr<float>("hidden")),
                    tex::arg("weight", buffers_.ptr<float>("norm_weight")),
                    tex::arg("out", buffers_.ptr<float>("norm_hidden")),
                    tex::arg("eps", 1.0e-5f),
                    tex::arg("rows", (int32_t)1),
                    tex::arg("hidden", (int32_t)hidden_),
                });
    tex::launch(queue_, bundles_.get("gate_projection"), intermediate_,
                {
                    tex::arg("x", buffers_.ptr<float>("norm_hidden")),
                    tex::arg("weight", buffers_.ptr<float>("gate_weight")),
                    tex::arg("out", buffers_.ptr<float>("gate")),
                    tex::arg("outputs", (int32_t)intermediate_),
                    tex::arg("hidden", (int32_t)hidden_),
                });
    tex::launch(queue_, bundles_.get("up_projection"), intermediate_,
                {
                    tex::arg("x", buffers_.ptr<float>("norm_hidden")),
                    tex::arg("weight", buffers_.ptr<float>("up_weight")),
                    tex::arg("out", buffers_.ptr<float>("up")),
                    tex::arg("outputs", (int32_t)intermediate_),
                    tex::arg("hidden", (int32_t)hidden_),
                });
    tex::launch(queue_, bundles_.get("silu_mul"), intermediate_,
                {
                    tex::arg("gate", buffers_.ptr<float>("gate")),
                    tex::arg("up", buffers_.ptr<float>("up")),
                    tex::arg("out", buffers_.ptr<float>("activated")),
                    tex::arg("n", (int32_t)intermediate_),
                });
    tex::launch(queue_, bundles_.get("down_projection"), hidden_,
                {
                    tex::arg("x", buffers_.ptr<float>("activated")),
                    tex::arg("weight", buffers_.ptr<float>("down_weight")),
                    tex::arg("out", buffers_.ptr<float>("projected")),
                    tex::arg("outputs", (int32_t)hidden_),
                    tex::arg("hidden", (int32_t)intermediate_),
                });
    tex::launch(queue_, bundles_.get("residual_add"), hidden_,
                {
                    tex::arg("x", buffers_.ptr<float>("hidden")),
                    tex::arg("y", buffers_.ptr<float>("projected")),
                    tex::arg("out", buffers_.ptr<float>("out")),
                    tex::arg("n", (int32_t)hidden_),
                });
  }

  void synchronize() const { queue_.synchronize(); }

  void copy_output(std::vector<float> &out) const {
    if (out.size() != hidden_) {
      throw std::runtime_error("mini MLP output shape mismatch");
    }
    buffers_.copy_from(out, "out");
  }

private:
  lrrt::Queue queue_;
  tex::BundleSet bundles_;
  tex::BufferSet buffers_;
  uint32_t hidden_;
  uint32_t intermediate_;
};

static void fill_projection_weight(std::vector<float> &weight, uint32_t input,
                                   uint32_t seed) {
  for (uint32_t row = 0; row < weight.size() / input; ++row) {
    for (uint32_t col = 0; col < input; ++col) {
      uint32_t index = row * input + col;
      int32_t lane = (int32_t)((index + row * 7 + seed * 11) % 23) - 11;
      weight[index] = 0.0078125f * (float)lane;
    }
  }
}

static void reference_rmsnorm(const std::vector<float> &x,
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
                                 uint32_t input, std::vector<float> &out) {
  for (uint32_t row = 0; row < out.size(); ++row) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < input; ++col) {
      sum += x[col] * weight[row * input + col];
    }
    out[row] = sum;
  }
}

static void run_case(lrrt::Device &device, uint32_t hidden,
                     uint32_t intermediate) {
  std::vector<float> input(hidden);
  std::vector<float> norm_weight(hidden);
  std::vector<float> gate_weight(intermediate * hidden);
  std::vector<float> up_weight(intermediate * hidden);
  std::vector<float> down_weight(hidden * intermediate);
  std::vector<float> out(hidden, 0.0f);
  for (uint32_t i = 0; i < hidden; ++i) {
    input[i] = 0.03125f * (float)((int32_t)((i * 5) % 31) - 15);
    norm_weight[i] = 1.0f + 0.001f * (float)(i % 29);
  }
  fill_projection_weight(gate_weight, hidden, 1);
  fill_projection_weight(up_weight, hidden, 2);
  fill_projection_weight(down_weight, intermediate, 3);

  MiniMlpExecutor executor(device, hidden, intermediate);
  executor.copy_inputs(input, norm_weight, gate_weight, up_weight, down_weight);
  executor.run();
  executor.synchronize();
  executor.copy_output(out);

  std::vector<float> norm_hidden(hidden);
  std::vector<float> gate(intermediate);
  std::vector<float> up(intermediate);
  std::vector<float> activated(intermediate);
  std::vector<float> projected(hidden);
  reference_rmsnorm(input, norm_weight, norm_hidden);
  reference_projection(norm_hidden, gate_weight, hidden, gate);
  reference_projection(norm_hidden, up_weight, hidden, up);
  for (uint32_t i = 0; i < intermediate; ++i) {
    activated[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
  }
  reference_projection(activated, down_weight, intermediate, projected);

  float max_diff = 0.0f;
  uint32_t max_index = 0;
  float max_actual = 0.0f;
  float max_expected = 0.0f;
  for (uint32_t i = 0; i < hidden; ++i) {
    float expected = input[i] + projected[i];
    float diff = fabsf(out[i] - expected);
    if (diff > max_diff) {
      max_diff = diff;
      max_index = i;
      max_actual = out[i];
      max_expected = expected;
    }
  }
  if (max_diff > 0.01f) {
    fprintf(stderr,
            "triton_mini_mlp hidden=%u intermediate=%u mismatch at %u: "
            "actual=%f expected=%f diff=%f\n",
            hidden, intermediate, max_index, max_actual, max_expected,
            max_diff);
    throw std::runtime_error("triton_mini_mlp result mismatch");
  }
}

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
    run_case(device, 768, 2048);
    run_case(device, 1024, 3072);

    printf("triton_mini_mlp: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
