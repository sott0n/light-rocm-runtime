#include "lrrt/lrrt.h"

#include <array>
#include <cstddef>
#include <cstdio>

namespace {

bool expect_status(lr_status_t actual, lr_status_t expected,
                   const char *operation) {
  if (actual == expected) {
    return true;
  }
  std::fprintf(stderr, "%s returned %s, expected %s\n", operation,
               lr_status_string(actual), lr_status_string(expected));
  return false;
}

bool open_device(lr_device_t *device, const char *init_operation,
                 const char *open_operation) {
  return expect_status(lr_init(), LR_SUCCESS, init_operation) &&
         expect_status(lr_device_open(0, device), LR_SUCCESS, open_operation);
}

} // namespace

int main() {
  lr_device_t device{};
  if (!open_device(&device, "lr_init first session",
                   "lr_device_open first session")) {
    (void)lr_shutdown();
    return 1;
  }

  std::array<unsigned char, 17> input{};
  std::array<unsigned char, 17> output{};
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<unsigned char>(index * 7 + 3);
  }

  void *first = nullptr;
  void *second = nullptr;
  if (!expect_status(lr_malloc(device, input.size(), &first), LR_SUCCESS,
                     "lr_malloc first") ||
      !expect_status(lr_malloc(device, input.size(), &second), LR_SUCCESS,
                     "lr_malloc second") ||
      !expect_status(lr_memcpy(device, first, input.data(), input.size(),
                               LR_MEMCPY_HOST_TO_DEVICE),
                     LR_SUCCESS, "lr_memcpy host to device") ||
      !expect_status(lr_memcpy(device, second, first, input.size(),
                               LR_MEMCPY_DEVICE_TO_DEVICE),
                     LR_SUCCESS, "lr_memcpy device to device") ||
      !expect_status(lr_memcpy(device, output.data(), second, output.size(),
                               LR_MEMCPY_DEVICE_TO_HOST),
                     LR_SUCCESS, "lr_memcpy device to host") ||
      output != input) {
    std::fprintf(stderr, "direct KFD memory round trip mismatch\n");
    (void)lr_free(device, second);
    (void)lr_free(device, first);
    (void)lr_shutdown();
    return 1;
  }

  auto overlap_expected = input;
  for (size_t index = overlap_expected.size() - 1; index != 0; --index) {
    overlap_expected[index] = overlap_expected[index - 1];
  }
  if (!expect_status(lr_memcpy(device, static_cast<unsigned char *>(first) + 1,
                               first, input.size() - 1,
                               LR_MEMCPY_DEVICE_TO_DEVICE),
                     LR_SUCCESS, "lr_memcpy overlapping device memory") ||
      !expect_status(lr_memcpy(device, output.data(), first, output.size(),
                               LR_MEMCPY_DEVICE_TO_HOST),
                     LR_SUCCESS, "lr_memcpy overlapped result") ||
      output != overlap_expected) {
    std::fprintf(stderr, "direct KFD overlapping copy mismatch\n");
    (void)lr_free(device, second);
    (void)lr_free(device, first);
    (void)lr_shutdown();
    return 1;
  }

  lr_memory_stats_t stats{};
  if (!expect_status(lr_get_memory_stats(device, &stats), LR_SUCCESS,
                     "lr_get_memory_stats") ||
      stats.live_bytes != input.size() * 2 || stats.allocation_count != 2 ||
      stats.h2d_copy_bytes != input.size() ||
      stats.d2d_copy_bytes != input.size() * 2 - 1 ||
      stats.d2h_copy_bytes != output.size() * 2 || stats.memcpy_count != 5 ||
      !expect_status(lr_free(device, second), LR_SUCCESS, "lr_free second") ||
      !expect_status(lr_free(device, second), LR_ERROR_INVALID_ARGUMENT,
                     "lr_free stale") ||
      !expect_status(lr_shutdown(), LR_SUCCESS,
                     "lr_shutdown with live allocation")) {
    (void)lr_free(device, first);
    (void)lr_shutdown();
    return 1;
  }

  if (!open_device(&device, "lr_init second session",
                   "lr_device_open second session") ||
      !expect_status(lr_shutdown(), LR_SUCCESS, "lr_shutdown second session")) {
    (void)lr_shutdown();
    return 1;
  }

  std::printf("direct_kfd_memory: round_trip=ok cleanup=ok\n");
  return 0;
}
