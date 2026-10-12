#include "lrrt/lrrt.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#ifndef DIRECT_KFD_FAULT_HSACO
#error "DIRECT_KFD_FAULT_HSACO must name the test code object"
#endif

namespace {

struct VectorAddArguments {
  const float *input_a;
  const float *input_b;
  float *output;
  int32_t element_count;
};

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
  if (!read_file(DIRECT_KFD_FAULT_HSACO, &hsaco)) {
    std::fprintf(stderr, "failed to read direct KFD fault fixture\n");
    return 1;
  }

  lr_device_t device{};
  lr_module_t *module = nullptr;
  lr_kernel_t *kernel = nullptr;
  void *input_a = nullptr;
  void *input_b = nullptr;
  void *output = nullptr;
  constexpr size_t kElementCount = 64;
  if (lr_init() != LR_SUCCESS || lr_device_open(0, &device) != LR_SUCCESS ||
      lr_module_load_hsaco(device, hsaco.data(), hsaco.size(), &module) !=
          LR_SUCCESS ||
      lr_kernel_get(module, "vector_add", &kernel) != LR_SUCCESS ||
      lr_malloc(device, kElementCount * sizeof(float), &input_a) !=
          LR_SUCCESS ||
      lr_malloc(device, kElementCount * sizeof(float), &input_b) !=
          LR_SUCCESS ||
      lr_malloc(device, kElementCount * sizeof(float), &output) != LR_SUCCESS) {
    std::fprintf(stderr, "failed to prepare direct KFD fault dispatch\n");
    return 1;
  }
  if (lr_free(device, output) != LR_SUCCESS) {
    std::fprintf(stderr, "failed to unmap direct KFD fault target\n");
    return 1;
  }

  const VectorAddArguments arguments{
      static_cast<const float *>(input_a), static_cast<const float *>(input_b),
      static_cast<float *>(output), static_cast<int32_t>(kElementCount)};
  const lr_launch_config_t config = {
      {static_cast<uint32_t>(kElementCount), 1, 1},
      {static_cast<uint32_t>(kElementCount), 1, 1},
      0};
  if (lr_launch(kernel, &config, &arguments, sizeof(arguments)) != LR_SUCCESS) {
    std::fprintf(stderr, "failed to submit direct KFD fault dispatch\n");
    return 1;
  }

  const lr_status_t status = lr_synchronize(device);
  if (status != LR_ERROR_DEVICE_FAULT) {
    std::fprintf(stderr, "lr_synchronize returned %s, expected %s\n",
                 lr_status_string(status),
                 lr_status_string(LR_ERROR_DEVICE_FAULT));
    return 1;
  }
  std::printf("direct KFD memory fault detected\n");
  return 0;
}
