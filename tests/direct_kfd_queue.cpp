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
  if (!expect_status(lr_init(), LR_SUCCESS, "lr_init")) {
    return 1;
  }

  lr_device_t device{};
  lr_queue_t *unopened_queue = nullptr;
  if (!expect_status(lr_queue_create(device, &unopened_queue),
                     LR_ERROR_INVALID_ARGUMENT,
                     "lr_queue_create before device open") ||
      unopened_queue != nullptr ||
      !expect_status(lr_device_open(0, &device), LR_SUCCESS,
                     "lr_device_open") ||
      !expect_status(lr_queue_create(device, nullptr),
                     LR_ERROR_INVALID_ARGUMENT, "lr_queue_create null")) {
    (void)lr_shutdown();
    return 1;
  }

  lr_queue_t *first = nullptr;
  lr_queue_t *second = nullptr;
  if (!expect_status(lr_queue_create(device, &first), LR_SUCCESS,
                     "lr_queue_create first") ||
      !expect_status(lr_queue_create(device, &second), LR_SUCCESS,
                     "lr_queue_create second") ||
      first == nullptr || second == nullptr || first == second ||
      !expect_status(lr_queue_synchronize(first), LR_SUCCESS,
                     "lr_queue_synchronize first") ||
      !expect_status(lr_queue_synchronize(second), LR_SUCCESS,
                     "lr_queue_synchronize second") ||
      !expect_status(lr_queue_destroy(first), LR_SUCCESS,
                     "lr_queue_destroy first") ||
      !expect_status(lr_queue_destroy(first), LR_ERROR_INVALID_ARGUMENT,
                     "lr_queue_destroy stale") ||
      !expect_status(lr_queue_synchronize(first), LR_ERROR_INVALID_ARGUMENT,
                     "lr_queue_synchronize stale") ||
      !expect_status(lr_queue_destroy(second), LR_SUCCESS,
                     "lr_queue_destroy second")) {
    (void)lr_shutdown();
    return 1;
  }

  lr_queue_t *shutdown_owned = nullptr;
  if (!expect_status(lr_queue_create(device, &shutdown_owned), LR_SUCCESS,
                     "lr_queue_create before shutdown") ||
      !expect_status(lr_shutdown(), LR_SUCCESS,
                     "lr_shutdown with live queue") ||
      !expect_status(lr_init(), LR_SUCCESS, "lr_init second session") ||
      !expect_status(lr_device_open(0, &device), LR_SUCCESS,
                     "lr_device_open second session")) {
    (void)lr_shutdown();
    return 1;
  }

  lr_queue_t *reopened_queue = nullptr;
  if (!expect_status(lr_queue_create(device, &reopened_queue), LR_SUCCESS,
                     "lr_queue_create second session") ||
      !expect_status(lr_queue_destroy(reopened_queue), LR_SUCCESS,
                     "lr_queue_destroy second session") ||
      !expect_status(lr_shutdown(), LR_SUCCESS, "lr_shutdown second session")) {
    (void)lr_shutdown();
    return 1;
  }

  std::printf("direct_kfd_queue: lifecycle=ok cleanup=ok\n");
  return 0;
}
