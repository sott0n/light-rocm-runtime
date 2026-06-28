#include "triton_executor.hpp"

#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#ifndef LRRT_TRITON_KV_CACHE_MANIFEST
#define LRRT_TRITON_KV_CACHE_MANIFEST "manifest.json"
#endif

namespace tex = lrrt::executor::triton;

static uint32_t select_block_size(uint32_t head_dim) {
  if (head_dim <= 64) {
    return 64;
  }
  if (head_dim <= 128) {
    return 128;
  }
  return 256;
}

static void fill_token(std::vector<float> &k, std::vector<float> &v,
                       uint32_t position, uint32_t head_dim) {
  for (uint32_t col = 0; col < head_dim; ++col) {
    int32_t k_lane = (int32_t)((position * 17 + col * 5) % 41) - 20;
    int32_t v_lane = (int32_t)((position * 19 + col * 7) % 43) - 21;
    k[col] = 0.015625f * (float)k_lane;
    v[col] = 0.015625f * (float)v_lane;
  }
}

static void check_vector(const char *name, const std::vector<float> &actual,
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
  if (max_diff > 0.0001f) {
    fprintf(stderr,
            "triton_kv_cache %s mismatch at %u: actual=%f expected=%f "
            "diff=%f\n",
            name, max_index, actual[max_index], expected[max_index], max_diff);
    throw std::runtime_error("triton_kv_cache result mismatch");
  }
}

static void run_case(lrrt::Device &device, uint32_t max_tokens,
                     uint32_t head_dim) {
  uint32_t block_size = select_block_size(head_dim);
  std::string update_name =
      "kv_cache_update_fp32_" + std::to_string(block_size);
  std::string read_name = "kv_cache_read_fp32_" + std::to_string(block_size);
  lrrt::Queue queue(device);
  tex::BundleSet bundles(device);
  bundles.add("update", LRRT_TRITON_KV_CACHE_MANIFEST, update_name);
  bundles.add("read", LRRT_TRITON_KV_CACHE_MANIFEST, read_name);

  std::vector<float> k(head_dim);
  std::vector<float> v(head_dim);
  std::vector<float> k_read(head_dim, 0.0f);
  std::vector<float> v_read(head_dim, 0.0f);
  std::vector<float> k_cache(max_tokens * head_dim, 0.0f);
  std::vector<float> v_cache(max_tokens * head_dim, 0.0f);

  tex::BufferSet buffers(device);
  buffers.allocate<float>("k", k.size());
  buffers.allocate<float>("v", v.size());
  buffers.allocate<float>("k_read", k_read.size());
  buffers.allocate<float>("v_read", v_read.size());
  buffers.allocate<float>("k_cache", k_cache.size());
  buffers.allocate<float>("v_cache", v_cache.size());
  buffers.copy_to("k_cache", k_cache);
  buffers.copy_to("v_cache", v_cache);

  for (uint32_t position = 0; position < max_tokens; ++position) {
    fill_token(k, v, position, head_dim);
    buffers.copy_to("k", k);
    buffers.copy_to("v", v);

    tex::launch(queue, bundles.get("update"), 1,
                {
                    tex::arg("k", buffers.ptr<float>("k")),
                    tex::arg("v", buffers.ptr<float>("v")),
                    tex::arg("k_cache", buffers.ptr<float>("k_cache")),
                    tex::arg("v_cache", buffers.ptr<float>("v_cache")),
                    tex::arg("position", (int32_t)position),
                    tex::arg("max_tokens", (int32_t)max_tokens),
                    tex::arg("head_dim", (int32_t)head_dim),
                });

    for (uint32_t col = 0; col < head_dim; ++col) {
      k_cache[position * head_dim + col] = k[col];
      v_cache[position * head_dim + col] = v[col];
    }
  }

  uint32_t read_position = max_tokens / 2;
  tex::launch(queue, bundles.get("read"), 1,
              {
                  tex::arg("k_cache", buffers.ptr<float>("k_cache")),
                  tex::arg("v_cache", buffers.ptr<float>("v_cache")),
                  tex::arg("k", buffers.ptr<float>("k_read")),
                  tex::arg("v", buffers.ptr<float>("v_read")),
                  tex::arg("position", (int32_t)read_position),
                  tex::arg("max_tokens", (int32_t)max_tokens),
                  tex::arg("head_dim", (int32_t)head_dim),
              });

  queue.synchronize();
  buffers.copy_from(k_read, "k_read");
  buffers.copy_from(v_read, "v_read");

  std::vector<float> expected_k(head_dim);
  std::vector<float> expected_v(head_dim);
  for (uint32_t col = 0; col < head_dim; ++col) {
    expected_k[col] = k_cache[read_position * head_dim + col];
    expected_v[col] = v_cache[read_position * head_dim + col];
  }
  check_vector("k_read", k_read, expected_k);
  check_vector("v_read", v_read, expected_v);

  std::vector<float> actual_k_cache(k_cache.size(), 0.0f);
  std::vector<float> actual_v_cache(v_cache.size(), 0.0f);
  buffers.copy_from(actual_k_cache, "k_cache");
  buffers.copy_from(actual_v_cache, "v_cache");
  check_vector("k_cache", actual_k_cache, k_cache);
  check_vector("v_cache", actual_v_cache, v_cache);
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
    run_case(device, 8, 64);
    run_case(device, 11, 128);
    run_case(device, 13, 192);

    printf("triton_kv_cache: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
