#include "lrrt/bundle.hpp"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <stdexcept>
#include <vector>

namespace {

int g_failures = 0;

void expect(bool condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s\n", message);
    ++g_failures;
  }
}

template <typename Function>
void expect_throw(Function function, const char *message) {
  try {
    function();
    fprintf(stderr, "FAIL: %s\n", message);
    ++g_failures;
  } catch (const std::exception &) {
  }
}

lrrt::KernelManifest parse(const char *json) {
  return lrrt::parse_bundle_manifest(json, strlen(json));
}

lrrt::KernelManifest parse(const char *json, const char *kernel_name) {
  return lrrt::parse_bundle_manifest(json, strlen(json), kernel_name);
}

const char *kManifest = R"json(
{
  "target": "gfx1101",
  "kernels": [
    {
      "name": "scale",
      "symbol": "scale_kernel",
      "code_object": "kernels.hsaco",
      "args": [
        {"name": "x", "type": "ptr", "offset": 0, "size": 8},
        {"name": "out", "type": "ptr", "offset": 8, "size": 8}
      ],
      "kernarg_size": 16,
      "block": [128, 1, 1],
      "grid": ["ceil_div(n, 256) * 128", 1, 1],
      "shared_memory_bytes": 16
    }
  ]
}
)json";

void test_parse_manifest() {
  lrrt::KernelManifest manifest = parse(kManifest);
  expect(manifest.name == "scale", "kernel name");
  expect(manifest.symbol == "scale_kernel", "kernel symbol");
  expect(manifest.code_object == "kernels.hsaco", "code object path");
  expect(manifest.kernarg_size == 16, "kernarg size");
  expect(manifest.block[0] == 128 && manifest.block[1] == 1 &&
             manifest.block[2] == 1,
         "block dimensions");
  expect(manifest.grid_divisor == 256, "grid divisor");
  expect(manifest.grid_multiplier == 128, "grid multiplier");
  expect(manifest.shared_memory_bytes == 16, "shared memory size");
  expect(manifest.arg_offsets == std::vector<size_t>({0, 8}),
         "argument offsets");

  lr_launch_config_t config = lrrt::launch_config_from_manifest(manifest, 513);
  expect(config.grid.x == 384 && config.grid.y == 1 && config.grid.z == 1,
         "total grid dimensions");
  expect(config.block.x == 128 && config.block.y == 1 && config.block.z == 1,
         "launch block dimensions");
  expect(config.shared_memory_bytes == 16, "launch dynamic shared memory");

  lrrt::require_kernarg_layout(manifest, 16, {0, 8});
  expect_throw([&] { lrrt::require_kernarg_layout(manifest, 24, {0, 8}); },
               "kernarg size mismatch must fail");
  expect_throw([&] { lrrt::require_kernarg_layout(manifest, 16, {0, 16}); },
               "kernarg offset mismatch must fail");
}

void test_multiple_kernels() {
  const char *json = R"json(
  {
    "kernels": [
      {
        "name": "first",
        "symbol": "first_kernel",
        "code_object": "first.hsaco",
        "args": [{"offset": 0}],
        "kernarg_size": 8,
        "block": [64, 1, 1],
        "grid": ["ceil_div(n, 64) * 64", 1, 1]
      },
      {
        "name": "second",
        "symbol": "second_kernel",
        "code_object": "second.hsaco",
        "args": [{"offset": 0}],
        "kernarg_size": 8,
        "block": [64, 1, 1],
        "grid": ["ceil_div(n, 64) * 64", 1, 1],
        "shared_memory_bytes": 64
      }
    ]
  }
  )json";

  lrrt::KernelManifest manifest = parse(json);
  expect(manifest.name == "first", "first kernel is selected");
  expect(manifest.shared_memory_bytes == 0,
         "missing shared memory defaults to zero");

  std::vector<lrrt::KernelManifest> manifests =
      lrrt::parse_bundle_manifests(json, strlen(json));
  expect(manifests.size() == 2, "all kernels are parsed");
  expect(manifests[0].name == "first" && manifests[1].name == "second",
         "kernel manifest order is preserved");

  lrrt::KernelManifest second = parse(json, "second");
  expect(second.name == "second", "kernel is selected by name");
  expect(second.symbol == "second_kernel", "selected kernel symbol");
  expect(second.code_object == "second.hsaco", "selected code object");
  expect(second.shared_memory_bytes == 64,
         "selected kernel shared memory size");

  expect_throw([&] { parse(json, "missing"); },
               "missing kernel name must fail");
  expect_throw([&] { parse(json, ""); }, "empty kernel name must fail");
}

void test_invalid_manifests() {
  expect_throw([] { lrrt::parse_bundle_manifest(nullptr, 0); },
               "empty manifest must fail");
  expect_throw([] { parse(R"json({"kernels":[]})json"); },
               "empty kernel array must fail");
  expect_throw(
      [] {
        parse(R"json(
          {"kernels":[
            {"name":"same","symbol":"a","code_object":"a.hsaco",
             "args":[{"offset":0}],"kernarg_size":8,
             "block":[64,1,1],"grid":["ceil_div(n, 64) * 64",1,1]},
            {"name":"same","symbol":"b","code_object":"b.hsaco",
             "args":[{"offset":0}],"kernarg_size":8,
             "block":[64,1,1],"grid":["ceil_div(n, 64) * 64",1,1]}
          ]}
        )json");
      },
      "duplicate kernel names must fail");
  expect_throw(
      [] { parse(R"json({"kernels":[{"name":"bad","symbol":"k"}]})json"); },
      "missing required fields must fail");
  expect_throw(
      [] {
        parse(R"json(
          {"kernels":[{
            "name":"bad","symbol":"k","code_object":"k.hsaco",
            "args":[{"offset":0}],"kernarg_size":8,
            "block":[64,1,1],"grid":["n * 64",1,1]
          }]}
        )json");
      },
      "unsupported grid expression must fail");
  expect_throw(
      [] {
        parse(R"json(
          {"kernels":[{
            "name":"bad","symbol":"k","code_object":"k.hsaco",
            "args":[{"offset":0}],"kernarg_size":8,
            "block":[64,1,1],
            "grid":["ceil_div(n, 64) * 64garbage",1,1]
          }]}
        )json");
      },
      "grid expression suffix must fail");
  expect_throw(
      [] {
        parse(R"json(
          {"kernels":[{
            "name":"bad","symbol":"k","code_object":"k.hsaco",
            "args":[{"offset":0}],"kernarg_size":8,
            "block":[64,1,1,2],
            "grid":["ceil_div(n, 64) * 64",1,1]
          }]}
        )json");
      },
      "extra block dimensions must fail");
  expect_throw(
      [] {
        parse(R"json(
          {"kernels":[{
            "name":"bad","symbol":"k","code_object":"k.hsaco",
            "args":[{"offset":0}],"kernarg_size":8,
            "block":[64,1,1],"grid":["ceil_div(n, 64) * 64",1,1],
            "shared_memory_bytes":4294967296
          }]}
        )json");
      },
      "oversized shared memory must fail");
}

void test_grid_overflow() {
  lrrt::KernelManifest manifest = parse(kManifest);
  manifest.grid_divisor = 1;
  manifest.grid_multiplier = UINT32_MAX;
  expect_throw([&] { lrrt::launch_config_from_manifest(manifest, 2); },
               "oversized launch grid must fail");
}

} // namespace

int main() {
  test_parse_manifest();
  test_multiple_kernels();
  test_invalid_manifests();
  test_grid_overflow();

  if (g_failures != 0) {
    fprintf(stderr, "%d bundle tests failed\n", g_failures);
    return 1;
  }
  printf("bundle tests: ok\n");
  return 0;
}
