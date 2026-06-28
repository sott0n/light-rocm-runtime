#include "lrrt/bundle.hpp"
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
  lrrt::Bundle update_bundle(device, LRRT_TRITON_KV_CACHE_MANIFEST,
                             update_name.c_str());
  lrrt::Bundle read_bundle(device, LRRT_TRITON_KV_CACHE_MANIFEST,
                           read_name.c_str());

  std::vector<float> k(head_dim);
  std::vector<float> v(head_dim);
  std::vector<float> k_read(head_dim, 0.0f);
  std::vector<float> v_read(head_dim, 0.0f);
  std::vector<float> k_cache(max_tokens * head_dim, 0.0f);
  std::vector<float> v_cache(max_tokens * head_dim, 0.0f);

  lrrt::DeviceBuffer device_k(device, k.size() * sizeof(float));
  lrrt::DeviceBuffer device_v(device, v.size() * sizeof(float));
  lrrt::DeviceBuffer device_k_read(device, k_read.size() * sizeof(float));
  lrrt::DeviceBuffer device_v_read(device, v_read.size() * sizeof(float));
  lrrt::DeviceBuffer device_k_cache(device, k_cache.size() * sizeof(float));
  lrrt::DeviceBuffer device_v_cache(device, v_cache.size() * sizeof(float));
  lrrt::copy_to_device(device_k_cache, k_cache);
  lrrt::copy_to_device(device_v_cache, v_cache);

  for (uint32_t position = 0; position < max_tokens; ++position) {
    fill_token(k, v, position, head_dim);
    lrrt::copy_to_device(device_k, k);
    lrrt::copy_to_device(device_v, v);

    lrrt::KernargBuffer update_args = update_bundle.make_args();
    update_args.set("k", (const float *)device_k.data());
    update_args.set("v", (const float *)device_v.data());
    update_args.set("k_cache", (float *)device_k_cache.data());
    update_args.set("v_cache", (float *)device_v_cache.data());
    update_args.set("position", (int32_t)position);
    update_args.set("max_tokens", (int32_t)max_tokens);
    update_args.set("head_dim", (int32_t)head_dim);
    update_args.bind_optional_nulls();
    update_bundle.launch(1, update_args);

    for (uint32_t col = 0; col < head_dim; ++col) {
      k_cache[position * head_dim + col] = k[col];
      v_cache[position * head_dim + col] = v[col];
    }
  }

  uint32_t read_position = max_tokens / 2;
  lrrt::KernargBuffer read_args = read_bundle.make_args();
  read_args.set("k_cache", (const float *)device_k_cache.data());
  read_args.set("v_cache", (const float *)device_v_cache.data());
  read_args.set("k", (float *)device_k_read.data());
  read_args.set("v", (float *)device_v_read.data());
  read_args.set("position", (int32_t)read_position);
  read_args.set("max_tokens", (int32_t)max_tokens);
  read_args.set("head_dim", (int32_t)head_dim);
  read_args.bind_optional_nulls();
  read_bundle.launch(1, read_args);

  device.synchronize();
  lrrt::copy_to_host(k_read, device_k_read);
  lrrt::copy_to_host(v_read, device_v_read);

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
  lrrt::copy_to_host(actual_k_cache, device_k_cache);
  lrrt::copy_to_host(actual_v_cache, device_v_cache);
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
