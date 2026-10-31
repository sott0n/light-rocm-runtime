#include "lrrt/lrrt.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#ifndef DIRECT_KFD_LAUNCH_HSACO
#error "DIRECT_KFD_LAUNCH_HSACO must name the test code object"
#endif

namespace {

constexpr size_t kElementCount = 64;

struct VectorAddArguments {
  const float *input_a;
  const float *input_b;
  float *output;
  int32_t element_count;
};

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

bool output_matches(const std::array<float, kElementCount> &input_a,
                    const std::array<float, kElementCount> &input_b,
                    const std::array<float, kElementCount> &output) {
  for (size_t index = 0; index < kElementCount; ++index) {
    if (std::fabs(output[index] - (input_a[index] + input_b[index])) > 0.001F) {
      std::fprintf(stderr, "vector add mismatch at %zu\n", index);
      return false;
    }
  }
  return true;
}

bool dependent_output_matches(const std::array<float, kElementCount> &input_a,
                              const std::array<float, kElementCount> &input_b,
                              const std::array<float, kElementCount> &output) {
  for (size_t index = 0; index < kElementCount; ++index) {
    if (std::fabs(output[index] - (input_a[index] + 2.0F * input_b[index])) >
        0.001F) {
      std::fprintf(stderr, "dependent vector add mismatch at %zu\n", index);
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  std::vector<uint8_t> hsaco;
  if (!read_file(DIRECT_KFD_LAUNCH_HSACO, &hsaco)) {
    std::fprintf(stderr, "failed to read direct KFD launch fixture\n");
    return 1;
  }

  lr_device_t device{};
  lr_module_t *module = nullptr;
  lr_kernel_t *kernel = nullptr;
  void *device_a = nullptr;
  void *device_b = nullptr;
  void *device_output = nullptr;
  void *device_dependency_output = nullptr;
  if (!expect_status(lr_init(), LR_SUCCESS, "lr_init") ||
      !expect_status(lr_device_open(0, &device), LR_SUCCESS,
                     "lr_device_open") ||
      !expect_status(
          lr_module_load_hsaco(device, hsaco.data(), hsaco.size(), &module),
          LR_SUCCESS, "lr_module_load_hsaco") ||
      !expect_status(lr_kernel_get(module, "vector_add", &kernel), LR_SUCCESS,
                     "lr_kernel_get") ||
      !expect_status(
          lr_malloc(device, kElementCount * sizeof(float), &device_a),
          LR_SUCCESS, "lr_malloc input a") ||
      !expect_status(
          lr_malloc(device, kElementCount * sizeof(float), &device_b),
          LR_SUCCESS, "lr_malloc input b") ||
      !expect_status(
          lr_malloc(device, kElementCount * sizeof(float), &device_output),
          LR_SUCCESS, "lr_malloc output") ||
      !expect_status(lr_malloc(device, kElementCount * sizeof(float),
                               &device_dependency_output),
                     LR_SUCCESS, "lr_malloc dependency output")) {
    (void)lr_shutdown();
    return 1;
  }

  std::array<float, kElementCount> input_a{};
  std::array<float, kElementCount> input_b{};
  std::array<float, kElementCount> output{};
  for (size_t index = 0; index < kElementCount; ++index) {
    input_a[index] = static_cast<float>(index);
    input_b[index] = static_cast<float>(index * 2U);
  }
  if (!expect_status(lr_memcpy(device, device_a, input_a.data(),
                               sizeof(input_a), LR_MEMCPY_HOST_TO_DEVICE),
                     LR_SUCCESS, "copy input a") ||
      !expect_status(lr_memcpy(device, device_b, input_b.data(),
                               sizeof(input_b), LR_MEMCPY_HOST_TO_DEVICE),
                     LR_SUCCESS, "copy input b")) {
    (void)lr_shutdown();
    return 1;
  }

  const VectorAddArguments arguments{static_cast<const float *>(device_a),
                                     static_cast<const float *>(device_b),
                                     static_cast<float *>(device_output),
                                     static_cast<int32_t>(kElementCount)};
  const lr_launch_config_t config = {
      {static_cast<uint32_t>(kElementCount), 1, 1},
      {static_cast<uint32_t>(kElementCount), 1, 1},
      0};

  lr_queue_t *queue = nullptr;
  lr_queue_t *second_queue = nullptr;
  output.fill(0.0F);
  if (!expect_status(lr_queue_create(device, &queue), LR_SUCCESS,
                     "lr_queue_create") ||
      !expect_status(lr_queue_create(device, &second_queue), LR_SUCCESS,
                     "lr_queue_create second") ||
      !expect_status(lr_launch_on_queue(queue, kernel, &config, &arguments,
                                        sizeof(arguments)),
                     LR_SUCCESS, "lr_launch_on_queue") ||
      !expect_status(lr_launch_on_queue(second_queue, kernel, &config,
                                        &arguments, sizeof(arguments)),
                     LR_SUCCESS, "lr_launch_on_queue second") ||
      !expect_status(lr_launch_on_queue(queue, kernel, &config, &arguments,
                                        sizeof(arguments)),
                     LR_SUCCESS, "lr_launch_on_queue ordered") ||
      !expect_status(lr_queue_synchronize(queue), LR_SUCCESS,
                     "lr_queue_synchronize") ||
      !expect_status(lr_queue_synchronize(second_queue), LR_SUCCESS,
                     "lr_queue_synchronize second") ||
      !expect_status(lr_launch_on_queue(queue, kernel, &config, &arguments,
                                        sizeof(arguments)),
                     LR_SUCCESS, "lr_launch_on_queue before destroy") ||
      !expect_status(lr_queue_destroy(queue), LR_SUCCESS,
                     "lr_queue_destroy with pending launch") ||
      !expect_status(lr_queue_destroy(second_queue), LR_SUCCESS,
                     "lr_queue_destroy second") ||
      !expect_status(lr_memcpy(device, output.data(), device_output,
                               sizeof(output), LR_MEMCPY_DEVICE_TO_HOST),
                     LR_SUCCESS, "copy explicit queue output") ||
      !output_matches(input_a, input_b, output)) {
    (void)lr_shutdown();
    return 1;
  }

  lr_queue_t *producer_queue = nullptr;
  lr_queue_t *consumer_queue = nullptr;
  std::array<lr_event_t *, 6> dependencies{};
  const lr_launch_config_t long_config = {
      {16U * 1024U * 1024U, 1, 1}, {64, 1, 1}, 0};
  const VectorAddArguments dependent_arguments{
      static_cast<const float *>(device_output),
      static_cast<const float *>(device_b),
      static_cast<float *>(device_dependency_output),
      static_cast<int32_t>(kElementCount)};
  bool dependency_test_ok =
      expect_status(lr_queue_create(device, &producer_queue), LR_SUCCESS,
                    "create dependency producer queue") &&
      expect_status(lr_queue_create(device, &consumer_queue), LR_SUCCESS,
                    "create dependency consumer queue") &&
      expect_status(lr_launch_on_queue(producer_queue, kernel, &long_config,
                                       &arguments, sizeof(arguments)),
                    LR_SUCCESS, "launch dependency producer");
  for (size_t index = 0; dependency_test_ok && index < dependencies.size();
       ++index) {
    dependency_test_ok =
        expect_status(lr_event_create(device, &dependencies[index]), LR_SUCCESS,
                      "create kernel dependency event") &&
        expect_status(
            lr_event_record_on_queue(dependencies[index], producer_queue),
            LR_SUCCESS, "record kernel dependency event");
  }
  lr_event_t *duplicate_dependencies[] = {dependencies[0], dependencies[0]};
  if (dependency_test_ok) {
    dependency_test_ok =
        expect_status(lr_launch_on_queue_with_dependencies(
                          consumer_queue, kernel, &config, &arguments,
                          sizeof(arguments), duplicate_dependencies, 2),
                      LR_ERROR_INVALID_ARGUMENT,
                      "reject duplicate kernel dependencies") &&
        expect_status(lr_launch_on_queue_with_dependencies(
                          consumer_queue, kernel, &config, &dependent_arguments,
                          sizeof(dependent_arguments), dependencies.data(),
                          dependencies.size()),
                      LR_SUCCESS, "launch with six kernel dependencies");
  }
  for (lr_event_t *&dependency : dependencies) {
    if (dependency &&
        !expect_status(lr_event_destroy(dependency), LR_SUCCESS,
                       "destroy consumed kernel dependency event")) {
      dependency_test_ok = false;
    }
    dependency = nullptr;
  }
  if (producer_queue &&
      !expect_status(lr_queue_destroy(producer_queue), LR_SUCCESS,
                     "destroy dependency producer queue")) {
    dependency_test_ok = false;
  }
  if (consumer_queue &&
      !expect_status(lr_queue_destroy(consumer_queue), LR_SUCCESS,
                     "destroy dependency consumer queue")) {
    dependency_test_ok = false;
  }
  output.fill(0.0F);
  if (!expect_status(lr_memcpy(device, output.data(), device_dependency_output,
                               sizeof(output), LR_MEMCPY_DEVICE_TO_HOST),
                     LR_SUCCESS, "copy kernel dependency output") ||
      !dependent_output_matches(input_a, input_b, output)) {
    dependency_test_ok = false;
  }
  if (!dependency_test_ok) {
    (void)lr_shutdown();
    return 1;
  }

  if (!expect_status(lr_launch(kernel, &config, &arguments, sizeof(arguments)),
                     LR_SUCCESS, "lr_launch default") ||
      !expect_status(lr_launch(kernel, &config, &arguments, sizeof(arguments)),
                     LR_SUCCESS, "lr_launch default ordered") ||
      !expect_status(lr_synchronize(device), LR_SUCCESS, "lr_synchronize") ||
      !expect_status(lr_memcpy(device, output.data(), device_output,
                               sizeof(output), LR_MEMCPY_DEVICE_TO_HOST),
                     LR_SUCCESS, "copy default output") ||
      !output_matches(input_a, input_b, output)) {
    (void)lr_shutdown();
    return 1;
  }

  output.fill(0.0F);
  if (!expect_status(lr_launch(kernel, &config, &arguments, sizeof(arguments)),
                     LR_SUCCESS, "lr_launch before module destroy") ||
      !expect_status(lr_module_destroy(module), LR_SUCCESS,
                     "lr_module_destroy with pending launch") ||
      !expect_status(lr_memcpy(device, output.data(), device_output,
                               sizeof(output), LR_MEMCPY_DEVICE_TO_HOST),
                     LR_SUCCESS, "copy module destroy output") ||
      !output_matches(input_a, input_b, output) ||
      !expect_status(lr_free(device, device_dependency_output), LR_SUCCESS,
                     "lr_free dependency output") ||
      !expect_status(lr_free(device, device_output), LR_SUCCESS,
                     "lr_free output") ||
      !expect_status(lr_free(device, device_b), LR_SUCCESS,
                     "lr_free input b") ||
      !expect_status(lr_free(device, device_a), LR_SUCCESS,
                     "lr_free input a") ||
      !expect_status(lr_shutdown(), LR_SUCCESS, "lr_shutdown")) {
    (void)lr_shutdown();
    return 1;
  }

  std::printf(
      "direct_kfd_launch: default=ok explicit_two_queue=ok dependencies=ok "
      "cleanup=ok\n");
  return 0;
}
