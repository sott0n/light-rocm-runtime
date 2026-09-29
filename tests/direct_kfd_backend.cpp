#include "lrrt/lrrt.h"

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

} // namespace

int main() {
  uint32_t count = 0;
  if (!expect_status(lr_device_count(&count), LR_ERROR_NOT_INITIALIZED,
                     "lr_device_count before init") ||
      !expect_status(lr_init(), LR_SUCCESS, "lr_init") ||
      !expect_status(lr_init(), LR_ERROR_ALREADY_INITIALIZED,
                     "lr_init twice") ||
      !expect_status(lr_device_count(&count), LR_SUCCESS, "lr_device_count") ||
      count != 1) {
    (void)lr_shutdown();
    return 1;
  }

  lr_device_t device{};
  char name[128] = {};
  if (!expect_status(lr_device_open(count, &device), LR_ERROR_INVALID_ARGUMENT,
                     "lr_device_open out of range") ||
      !expect_status(lr_synchronize(device), LR_ERROR_INVALID_ARGUMENT,
                     "lr_synchronize before open") ||
      !expect_status(lr_device_open(0, &device), LR_SUCCESS,
                     "lr_device_open") ||
      !expect_status(lr_device_open(0, &device), LR_SUCCESS,
                     "lr_device_open twice") ||
      !expect_status(lr_device_name(device, name, sizeof(name)), LR_SUCCESS,
                     "lr_device_name") ||
      name[0] == '\0' ||
      !expect_status(lr_synchronize(device), LR_SUCCESS,
                     "lr_synchronize idle device") ||
      !expect_status(lr_shutdown(), LR_SUCCESS, "lr_shutdown") ||
      !expect_status(lr_device_count(&count), LR_ERROR_NOT_INITIALIZED,
                     "lr_device_count after shutdown") ||
      !expect_status(lr_shutdown(), LR_ERROR_NOT_INITIALIZED,
                     "lr_shutdown twice")) {
    (void)lr_shutdown();
    return 1;
  }

  std::printf("direct_kfd_backend: device=%s lifecycle=ok\n", name);
  return 0;
}
