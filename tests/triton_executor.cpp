#include "triton_executor.hpp"

#include <limits>
#include <stdio.h>
#include <string.h>

#include <exception>

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

} // namespace

int main() {
  test_unknown_names();
  test_allocation_overflow();
  test_empty_launch_arg_name();
  test_buffer_offset_errors();

  if (g_failures != 0) {
    return 1;
  }

  printf("triton_executor: ok\n");
  return 0;
}
