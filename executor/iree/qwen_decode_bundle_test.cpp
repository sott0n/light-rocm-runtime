#include "qwen_decode_bundle.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>

namespace {

std::filesystem::path test_dir() {
  const char *tmpdir = std::getenv("TMPDIR");
  const std::filesystem::path base = tmpdir ? tmpdir : "/tmp";
  return base / "lrrt_iree_qwen_decode_bundle_test";
}

void write_file(const std::filesystem::path &path, const std::string &text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to write test file: " + path.string());
  }
  file << text;
}

std::string manifest_text(const char *layer_vmfb = "layer.vmfb",
                          const char *tail_vmfb = "tail.vmfb",
                          uint32_t max_cache_tokens = 8,
                          uint32_t shape_tokens = 8, uint32_t shape_dim = 128) {
  return std::string("{\n") +
         "  \"manifest_version\": 1,\n"
         "  \"target\": \"gfx1101\",\n"
         "  \"layer_vmfb\": \"" +
         layer_vmfb +
         "\",\n"
         "  \"tail_vmfb\": \"" +
         tail_vmfb +
         "\",\n"
         "  \"layer_export\": \"qwen_decode_layer_kv_cache_max8\",\n"
         "  \"tail_export\": \"qwen_decode1_tail\",\n"
         "  \"max_cache_tokens\": " +
         std::to_string(max_cache_tokens) +
         ",\n"
         "  \"kv_cache_shape\": [" +
         std::to_string(shape_tokens) + ", " + std::to_string(shape_dim) +
         "]\n"
         "}\n";
}

void expect(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void expect_throw_contains(const std::function<void()> &fn, const char *needle,
                           const char *message) {
  try {
    fn();
  } catch (const std::exception &error) {
    if (std::string(error.what()).find(needle) == std::string::npos) {
      throw std::runtime_error(std::string(message) +
                               ": unexpected error: " + error.what());
    }
    return;
  }
  throw std::runtime_error(std::string(message) + ": expected throw");
}

void test_parse_manifest() {
  const std::filesystem::path dir = test_dir() / "valid";
  write_file(dir / "manifest.json", manifest_text());

  const auto manifest =
      lrrt::executor::iree::load_qwen_decode_bundle_manifest(dir);
  expect(manifest.manifest_version == 1, "manifest version");
  expect(manifest.target == "gfx1101", "target");
  expect(manifest.layer_vmfb == dir / "layer.vmfb", "layer vmfb path");
  expect(manifest.tail_vmfb == dir / "tail.vmfb", "tail vmfb path");
  expect(manifest.layer_export == "qwen_decode_layer_kv_cache_max8",
         "layer export");
  expect(manifest.tail_export == "qwen_decode1_tail", "tail export");
  expect(manifest.max_cache_tokens == 8, "max cache tokens");
  expect(manifest.kv_cache_tokens == 8, "kv cache tokens");
  expect(manifest.kv_cache_dim == 128, "kv cache dim");
}

void test_invalid_manifest() {
  const std::filesystem::path dir = test_dir() / "invalid";

  write_file(dir / "absolute" / "manifest.json",
             manifest_text("/tmp/layer.vmfb"));
  expect_throw_contains(
      [&] {
        lrrt::executor::iree::load_qwen_decode_bundle_manifest(dir /
                                                               "absolute");
      },
      "relative path", "absolute VMFB path must fail");

  write_file(dir / "parent" / "manifest.json", manifest_text("../layer.vmfb"));
  expect_throw_contains(
      [&] {
        lrrt::executor::iree::load_qwen_decode_bundle_manifest(dir / "parent");
      },
      "must not contain '..'", "parent VMFB path must fail");

  write_file(dir / "shape" / "manifest.json",
             manifest_text("layer.vmfb", "tail.vmfb", 8, 4, 128));
  expect_throw_contains(
      [&] {
        lrrt::executor::iree::load_qwen_decode_bundle_manifest(dir / "shape");
      },
      "must match kv_cache_shape[0]",
      "max_cache_tokens and shape mismatch must fail");
}

} // namespace

int main() {
  try {
    std::filesystem::remove_all(test_dir());
    test_parse_manifest();
    test_invalid_manifest();
    std::printf("qwen_decode_bundle_test: ok\n");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "qwen_decode_bundle_test: %s\n", error.what());
    return 1;
  }
}
