#include "lrrt/lrrt.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#ifndef DIRECT_KFD_MODULE_HSACO
#error "DIRECT_KFD_MODULE_HSACO must name the test code object"
#endif

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

bool read_file(const char *path, std::vector<uint8_t> *data) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return false;
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    return false;
  }
  data->resize(static_cast<size_t>(size));
  file.seekg(0, std::ios::beg);
  return static_cast<bool>(
      file.read(reinterpret_cast<char *>(data->data()), size));
}

} // namespace

int main() {
  std::vector<uint8_t> hsaco;
  if (!read_file(DIRECT_KFD_MODULE_HSACO, &hsaco)) {
    std::fprintf(stderr, "failed to read direct KFD module fixture\n");
    return 1;
  }

  if (!expect_status(lr_init(), LR_SUCCESS, "lr_init")) {
    return 1;
  }

  lr_device_t device{};
  lr_module_t *unopened_module = reinterpret_cast<lr_module_t *>(1);
  if (!expect_status(lr_module_load_hsaco(device, hsaco.data(), hsaco.size(),
                                          &unopened_module),
                     LR_ERROR_INVALID_ARGUMENT,
                     "lr_module_load_hsaco before device open") ||
      unopened_module != nullptr ||
      !expect_status(lr_device_open(0, &device), LR_SUCCESS,
                     "lr_device_open")) {
    (void)lr_shutdown();
    return 1;
  }

  uint8_t malformed = 0;
  lr_module_t *malformed_module = reinterpret_cast<lr_module_t *>(1);
  if (!expect_status(lr_module_load_hsaco(device, &malformed, sizeof(malformed),
                                          &malformed_module),
                     LR_ERROR_INVALID_ARGUMENT,
                     "lr_module_load_hsaco malformed") ||
      malformed_module != nullptr) {
    (void)lr_shutdown();
    return 1;
  }

  lr_module_t *module = nullptr;
  if (!expect_status(
          lr_module_load_hsaco(device, hsaco.data(), hsaco.size(), &module),
          LR_SUCCESS, "lr_module_load_hsaco") ||
      module == nullptr) {
    (void)lr_shutdown();
    return 1;
  }
  hsaco.assign(hsaco.size(), 0);

  lr_kernel_t *kernel = nullptr;
  lr_kernel_t *symbol_kernel = nullptr;
  lr_kernel_t *missing_kernel = reinterpret_cast<lr_kernel_t *>(1);
  if (!expect_status(lr_kernel_get(module, "vector_add", &kernel), LR_SUCCESS,
                     "lr_kernel_get name") ||
      kernel == nullptr ||
      !expect_status(lr_kernel_get(module, "vector_add.kd", &symbol_kernel),
                     LR_SUCCESS, "lr_kernel_get symbol") ||
      symbol_kernel == nullptr ||
      !expect_status(lr_kernel_get(module, "missing_kernel", &missing_kernel),
                     LR_ERROR_INVALID_ARGUMENT, "lr_kernel_get missing") ||
      missing_kernel != nullptr ||
      !expect_status(lr_module_destroy(module), LR_SUCCESS,
                     "lr_module_destroy") ||
      !expect_status(lr_module_destroy(module), LR_ERROR_INVALID_ARGUMENT,
                     "lr_module_destroy stale") ||
      !expect_status(lr_kernel_get(module, "vector_add", &kernel),
                     LR_ERROR_INVALID_ARGUMENT,
                     "lr_kernel_get destroyed module")) {
    (void)lr_shutdown();
    return 1;
  }

  if (!read_file(DIRECT_KFD_MODULE_HSACO, &hsaco)) {
    (void)lr_shutdown();
    return 1;
  }
  lr_module_t *shutdown_owned = nullptr;
  if (!expect_status(lr_module_load_hsaco(device, hsaco.data(), hsaco.size(),
                                          &shutdown_owned),
                     LR_SUCCESS, "lr_module_load_hsaco before shutdown") ||
      !expect_status(lr_shutdown(), LR_SUCCESS,
                     "lr_shutdown with live module") ||
      !expect_status(lr_init(), LR_SUCCESS, "lr_init second session") ||
      !expect_status(lr_device_open(0, &device), LR_SUCCESS,
                     "lr_device_open second session")) {
    (void)lr_shutdown();
    return 1;
  }

  lr_module_t *reopened_module = nullptr;
  if (!expect_status(lr_module_load_hsaco(device, hsaco.data(), hsaco.size(),
                                          &reopened_module),
                     LR_SUCCESS, "lr_module_load_hsaco second session") ||
      !expect_status(lr_module_destroy(reopened_module), LR_SUCCESS,
                     "lr_module_destroy second session") ||
      !expect_status(lr_shutdown(), LR_SUCCESS, "lr_shutdown second session")) {
    (void)lr_shutdown();
    return 1;
  }

  std::printf("direct_kfd_module: load=ok lookup=ok cleanup=ok\n");
  return 0;
}
