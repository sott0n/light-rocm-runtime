#include "triton_executor.hpp"

#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#ifndef LRRT_TRITON_QKV_PROJECTION_MATVEC_MANIFEST
#define LRRT_TRITON_QKV_PROJECTION_MATVEC_MANIFEST "manifest.json"
#endif

namespace tex = lrrt::executor::triton;

static uint32_t select_matvec_block(uint32_t hidden) {
  if (hidden <= 1024) {
    return 1024;
  }
  if (hidden <= 2048) {
    return 2048;
  }
  return 4096;
}

class QkvProjectionExecutor {
public:
  QkvProjectionExecutor(lrrt::Device &device, uint32_t hidden,
                        uint32_t projection)
      : queue_(device), bundles_(device), buffers_(device), hidden_(hidden),
        projection_(projection) {
    std::string kernel_name =
        "matvec_fp32_" + std::to_string(select_matvec_block(hidden));
    bundles_.add("q", LRRT_TRITON_QKV_PROJECTION_MATVEC_MANIFEST, kernel_name);
    bundles_.add("k", LRRT_TRITON_QKV_PROJECTION_MATVEC_MANIFEST, kernel_name);
    bundles_.add("v", LRRT_TRITON_QKV_PROJECTION_MATVEC_MANIFEST, kernel_name);

    buffers_.allocate<float>("hidden", hidden_);
    buffers_.allocate<float>("q_weight", projection_ * hidden_);
    buffers_.allocate<float>("k_weight", projection_ * hidden_);
    buffers_.allocate<float>("v_weight", projection_ * hidden_);
    buffers_.allocate<float>("q", projection_);
    buffers_.allocate<float>("k", projection_);
    buffers_.allocate<float>("v", projection_);
  }

  void copy_inputs(const std::vector<float> &hidden,
                   const std::vector<float> &q_weight,
                   const std::vector<float> &k_weight,
                   const std::vector<float> &v_weight) {
    if (hidden.size() != hidden_ || q_weight.size() != projection_ * hidden_ ||
        k_weight.size() != projection_ * hidden_ ||
        v_weight.size() != projection_ * hidden_) {
      throw std::runtime_error("qkv projection input shape mismatch");
    }

    std::vector<float> zeros(projection_, 0.0f);
    buffers_.copy_to("hidden", hidden);
    buffers_.copy_to("q_weight", q_weight);
    buffers_.copy_to("k_weight", k_weight);
    buffers_.copy_to("v_weight", v_weight);
    buffers_.copy_to("q", zeros);
    buffers_.copy_to("k", zeros);
    buffers_.copy_to("v", zeros);
  }

  void run() {
    launch_projection("q", "q_weight", "q");
    launch_projection("k", "k_weight", "k");
    launch_projection("v", "v_weight", "v");
  }

  void synchronize() const { queue_.synchronize(); }

  void copy_outputs(std::vector<float> &q, std::vector<float> &k,
                    std::vector<float> &v) const {
    if (q.size() != projection_ || k.size() != projection_ ||
        v.size() != projection_) {
      throw std::runtime_error("qkv projection output shape mismatch");
    }
    buffers_.copy_from(q, "q");
    buffers_.copy_from(k, "k");
    buffers_.copy_from(v, "v");
  }

private:
  void launch_projection(const char *bundle_name, const char *weight_name,
                         const char *out_name) {
    tex::launch(queue_, bundles_.get(bundle_name), projection_,
                {
                    tex::arg("x", buffers_.ptr<float>("hidden")),
                    tex::arg("weight", buffers_.ptr<float>(weight_name)),
                    tex::arg("out", buffers_.ptr<float>(out_name)),
                    tex::arg("outputs", (int32_t)projection_),
                    tex::arg("hidden", (int32_t)hidden_),
                });
  }

  lrrt::Queue queue_;
  tex::BundleSet bundles_;
  tex::BufferSet buffers_;
  uint32_t hidden_;
  uint32_t projection_;
};

static void fill_hidden(std::vector<float> &hidden) {
  for (uint32_t i = 0; i < hidden.size(); ++i) {
    hidden[i] = 0.03125f * (float)((int32_t)((i * 5) % 31) - 15);
  }
}

static void fill_projection_weight(std::vector<float> &weight, uint32_t hidden,
                                   uint32_t seed) {
  for (uint32_t row = 0; row < weight.size() / hidden; ++row) {
    for (uint32_t col = 0; col < hidden; ++col) {
      uint32_t index = row * hidden + col;
      int32_t lane = (int32_t)((index + row * 7 + seed * 11) % 23) - 11;
      weight[index] = 0.0078125f * (float)lane;
    }
  }
}

static void reference_projection(const std::vector<float> &hidden,
                                 const std::vector<float> &weight,
                                 std::vector<float> &out) {
  const uint32_t hidden_size = (uint32_t)hidden.size();
  for (uint32_t row = 0; row < out.size(); ++row) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < hidden_size; ++col) {
      sum += hidden[col] * weight[row * hidden_size + col];
    }
    out[row] = sum;
  }
}

static void check_projection(const char *name, const std::vector<float> &actual,
                             const std::vector<float> &expected) {
  float max_diff = 0.0f;
  uint32_t max_index = 0;
  for (uint32_t i = 0; i < actual.size(); ++i) {
    float diff = fabsf(actual[i] - expected[i]);
    if (diff > max_diff) {
      max_diff = diff;
      max_index = i;
    }
  }

  if (max_diff > 0.002f) {
    fprintf(stderr,
            "triton_qkv_projection %s mismatch at %u: actual=%f "
            "expected=%f diff=%f\n",
            name, max_index, actual[max_index], expected[max_index], max_diff);
    throw std::runtime_error("triton_qkv_projection result mismatch");
  }
}

static void run_case(lrrt::Device &device, uint32_t hidden,
                     uint32_t projection) {
  std::vector<float> input(hidden);
  std::vector<float> q_weight(projection * hidden);
  std::vector<float> k_weight(projection * hidden);
  std::vector<float> v_weight(projection * hidden);
  std::vector<float> q(projection, 0.0f);
  std::vector<float> k(projection, 0.0f);
  std::vector<float> v(projection, 0.0f);
  std::vector<float> expected_q(projection, 0.0f);
  std::vector<float> expected_k(projection, 0.0f);
  std::vector<float> expected_v(projection, 0.0f);

  fill_hidden(input);
  fill_projection_weight(q_weight, hidden, 1);
  fill_projection_weight(k_weight, hidden, 2);
  fill_projection_weight(v_weight, hidden, 3);

  QkvProjectionExecutor executor(device, hidden, projection);
  executor.copy_inputs(input, q_weight, k_weight, v_weight);
  executor.run();
  executor.synchronize();
  executor.copy_outputs(q, k, v);

  reference_projection(input, q_weight, expected_q);
  reference_projection(input, k_weight, expected_k);
  reference_projection(input, v_weight, expected_v);
  check_projection("q", q, expected_q);
  check_projection("k", k, expected_k);
  check_projection("v", v, expected_v);
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
    run_case(device, 768, 128);
    run_case(device, 1536, 128);

    printf("triton_qkv_projection: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
