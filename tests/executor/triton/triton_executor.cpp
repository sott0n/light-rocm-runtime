#include "triton_executor.hpp"

#include <algorithm>
#include <limits>
#include <stdio.h>
#include <string.h>

#include <exception>
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
void expect_throw_contains(Function function, const char *text,
                           const char *message) {
  try {
    function();
    fprintf(stderr, "FAIL: %s\n", message);
    ++g_failures;
  } catch (const std::exception &error) {
    if (strstr(error.what(), text) == nullptr) {
      fprintf(stderr, "FAIL: %s: got '%s'\n", message, error.what());
      ++g_failures;
    }
  }
}

lrrt::Device dummy_device() {
  lr_device_t device = {0};
  return lrrt::Device(device);
}

void test_unknown_names() {
  namespace tex = lrrt::executor::triton;

  tex::BundleSet bundles(dummy_device());
  expect_throw_contains([&] { bundles.get("missing"); },
                        "unknown Triton executor bundle 'missing'",
                        "unknown bundle must include requested name");
  expect_throw_contains([&] { bundles.get("missing"); }, "available bundles",
                        "unknown bundle must include available bundle list");

  tex::BufferSet buffers(dummy_device());
  expect_throw_contains([&] { buffers.get("input"); },
                        "unknown Triton executor buffer 'input'",
                        "unknown buffer must include requested name");
  expect_throw_contains([&] { buffers.get("input"); }, "available buffers",
                        "unknown buffer must include available buffer list");
}

void test_allocation_overflow() {
  namespace tex = lrrt::executor::triton;

  tex::BufferSet buffers(dummy_device());
  const size_t count =
      std::numeric_limits<size_t>::max() / sizeof(uint64_t) + 1;
  expect_throw_contains(
      [&] { buffers.allocate<uint64_t>("huge", count); },
      "Triton executor buffer allocation overflows size_t for buffer 'huge'",
      "overflowing allocation must include buffer name");
}

void test_empty_launch_arg_name() {
  namespace tex = lrrt::executor::triton;

  const int value = 1;
  expect_throw_contains([&] { (void)tex::arg<int>(nullptr, value); },
                        "Triton executor launch argument name is empty",
                        "null launch argument name must fail clearly");
  expect_throw_contains([&] { (void)tex::arg<int>("", value); },
                        "Triton executor launch argument name is empty",
                        "empty launch argument name must fail clearly");
}

void test_buffer_offset_errors() {
  namespace tex = lrrt::executor::triton;

  lrrt::Runtime runtime;
  if (runtime.device_count() == 0) {
    printf("triton_executor: skipped GPU offset check, no GPU devices\n");
    return;
  }

  lrrt::Device device = runtime.open_device(0);
  tex::BufferSet buffers(device);
  buffers.allocate<float>("values", 4);

  expect(buffers.ptr<float>("values", 4) != nullptr,
         "one-past-end pointer is allowed for empty slices");
  expect_throw_contains(
      [&] { buffers.ptr<float>("values", 5); },
      "Triton executor buffer offset is out of range for 'values'",
      "out-of-range offset must include buffer name");
  expect_throw_contains(
      [&] { buffers.ptr<float>("values", 5); }, "byte_offset=20",
      "out-of-range offset must include computed byte offset");
  expect_throw_contains([&] { buffers.ptr<float>("values", 5); },
                        "buffer_size=16",
                        "out-of-range offset must include buffer size");
}

void test_arena_contract() {
  namespace tex = lrrt::executor::triton;

  lrrt::Runtime runtime;
  if (runtime.device_count() == 0) {
    printf("triton_executor: skipped GPU arena check, no GPU devices\n");
    return;
  }

  lrrt::Device device = runtime.open_device(0);
  device.reset_memory_stats();

  tex::BufferSet buffers(device, true);
  expect(buffers.uses_arena(), "arena buffer set must report arena mode");
  expect_throw_contains([&] { buffers.get("a"); },
                        "Triton executor buffer set is not finalized",
                        "arena buffers must reject lookup before finalize");
  expect_throw_contains(
      [&] { buffers.byte_offset("a"); },
      "Triton executor buffer set does not have arena offsets",
      "arena offsets must reject lookup before finalize");

  buffers.allocate<float>("a", 3);
  buffers.allocate<uint32_t>("b", 5);
  buffers.finalize();

  const size_t offset_a = buffers.byte_offset("a");
  const size_t offset_b = buffers.byte_offset("b");
  expect(offset_a == 0, "first arena buffer must start at offset zero");
  expect(offset_b == 256, "second arena buffer must be 256-byte aligned");
  expect(buffers.arena_size() == offset_b + 5 * sizeof(uint32_t),
         "arena size must include aligned buffers");

  lrrt::MemoryStats stats = device.memory_stats();
  expect(stats.live_bytes == buffers.arena_size(),
         "arena must be the only live allocation");
  expect(stats.total_allocated_bytes == buffers.arena_size(),
         "arena allocation size must be counted");
  expect(stats.allocation_count == 1,
         "arena buffer set must allocate one device buffer");

  std::vector<float> values_a = {1.0f, 2.0f, 3.0f};
  std::vector<uint32_t> values_b = {4, 5, 6, 7, 8};
  buffers.copy_to("a", values_a);
  buffers.copy_to("b", values_b);

  std::vector<float> out_a(3, 0.0f);
  std::vector<uint32_t> out_b(5, 0);
  buffers.copy_from(out_a, "a");
  buffers.copy_from(out_b, "b");
  expect(std::equal(out_a.begin(), out_a.end(), values_a.begin()),
         "arena view copy for float buffer must round trip");
  expect(std::equal(out_b.begin(), out_b.end(), values_b.begin()),
         "arena view copy for uint32 buffer must round trip");

  expect_throw_contains(
      [&] { buffers.allocate<float>("late", 1); },
      "Triton executor arena buffer set is finalized",
      "arena buffer set must reject allocation after finalize");
  expect_throw_contains([&] { buffers.byte_offset("missing"); },
                        "unknown Triton executor buffer 'missing'",
                        "missing arena offset must include buffer name");
}

} // namespace

int main() {
  test_unknown_names();
  test_allocation_overflow();
  test_empty_launch_arg_name();
  test_buffer_offset_errors();
  test_arena_contract();

  if (g_failures != 0) {
    return 1;
  }

  printf("triton_executor: ok\n");
  return 0;
}
