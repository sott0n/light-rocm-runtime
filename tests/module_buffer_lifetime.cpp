#include "lrrt/lrrt.hpp"

#include <atomic>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <thread>
#include <vector>

#ifndef LRRT_ASYNC_COPY_LAUNCH_HSACO
#define LRRT_ASYNC_COPY_LAUNCH_HSACO "async_copy_launch_kernel.hsaco"
#endif

namespace {

struct ScaleArgs {
  const float *in;
  float *out;
  float alpha;
  int32_t index;
};

struct WaitArgs {
  unsigned long long iterations;
  unsigned long long *output;
};

std::vector<unsigned char> read_file(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("failed to open module-buffer lifetime HSACO");
  }
  std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("module-buffer lifetime HSACO is empty");
  }
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> data(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    throw std::runtime_error("failed to read module-buffer lifetime HSACO");
  }
  return data;
}

void expect_value(lr_device_t device, const void *device_buffer, float expected,
                  const char *scenario) {
  float actual = 0.0f;
  lrrt::check(lr_memcpy(device, &actual, device_buffer, sizeof(actual),
                        LR_MEMCPY_DEVICE_TO_HOST),
              "lr_memcpy device to host");
  if (actual != expected) {
    fprintf(stderr, "module_buffer_lifetime %s: actual=%f expected=%f\n",
            scenario, actual, expected);
    throw std::runtime_error("module-buffer lifetime result mismatch");
  }
}

void test_module_destroy_drains(lrrt::Device device,
                                const std::vector<unsigned char> &hsaco,
                                const std::vector<float> &input) {
  void *device_input = nullptr;
  void *device_output = nullptr;
  lrrt::check(
      lr_malloc(device.get(), input.size() * sizeof(float), &device_input),
      "lr_malloc input");
  lrrt::check(lr_malloc(device.get(), sizeof(float), &device_output),
              "lr_malloc output");
  lrrt::check(lr_memcpy(device.get(), device_input, input.data(),
                        input.size() * sizeof(float), LR_MEMCPY_HOST_TO_DEVICE),
              "lr_memcpy host to device");

  lr_module_t *module = nullptr;
  lrrt::check(
      lr_module_load_hsaco(device.get(), hsaco.data(), hsaco.size(), &module),
      "lr_module_load_hsaco");
  lr_kernel_t *kernel = nullptr;
  lrrt::check(lr_kernel_get(module, "async_copy_launch_kernel", &kernel),
              "lr_kernel_get");

  const float alpha = 2.5f;
  const ScaleArgs args = {static_cast<const float *>(device_input),
                          static_cast<float *>(device_output), alpha,
                          static_cast<int32_t>(input.size() - 1)};
  const lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
  lrrt::check(lr_launch(kernel, &config, &args, sizeof(args)), "lr_launch");
  lrrt::check(lr_module_destroy(module), "lr_module_destroy");
  expect_value(device.get(), device_output, input.back() * alpha,
               "module destroy drains dispatch");

  if (lr_launch(kernel, &config, &args, sizeof(args)) !=
      LR_ERROR_INVALID_ARGUMENT) {
    throw std::runtime_error("destroyed module left kernel handle usable");
  }
  if (lr_module_destroy(module) != LR_ERROR_INVALID_ARGUMENT) {
    throw std::runtime_error("destroyed module handle was accepted");
  }

  lrrt::check(lr_free(device.get(), device_output), "lr_free output");
  lrrt::check(lr_free(device.get(), device_input), "lr_free input");
}

void test_module_destroy_releases_runtime_lock(
    lrrt::Device device, const std::vector<unsigned char> &hsaco,
    const std::vector<float> &input) {
  lrrt::DeviceBuffer source(device, input.size() * sizeof(float));
  lrrt::DeviceBuffer output(device, sizeof(float));
  lrrt::DeviceBuffer wait_output(device, sizeof(unsigned long long));
  lrrt::copy_to_device(source, input);

  lr_module_t *destroyed_module = nullptr;
  lrrt::check(lr_module_load_hsaco(device.get(), hsaco.data(), hsaco.size(),
                                   &destroyed_module),
              "lr_module_load_hsaco");
  lr_kernel_t *wait_kernel = nullptr;
  lrrt::check(
      lr_kernel_get(destroyed_module, "queue_wait_kernel", &wait_kernel),
      "lr_kernel_get wait");

  lrrt::Module concurrent_module(device, hsaco);
  lrrt::Kernel concurrent_kernel =
      concurrent_module.kernel("async_copy_launch_kernel");
  lrrt::Queue wait_queue(device);
  lrrt::Queue concurrent_queue(device);
  const lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
  const WaitArgs wait_args = {
      50000ULL,
      static_cast<unsigned long long *>(wait_output.data()),
  };
  lrrt::check(lr_launch_on_queue(wait_queue.get(), wait_kernel, &config,
                                 &wait_args, sizeof(wait_args)),
              "lr_launch_on_queue wait");

  std::atomic<bool> destroy_started{false};
  std::atomic<bool> destroy_completed{false};
  lr_status_t destroy_status = LR_ERROR_RUNTIME;
  std::thread destroyer([&] {
    destroy_started.store(true, std::memory_order_release);
    destroy_status = lr_module_destroy(destroyed_module);
    destroy_completed.store(true, std::memory_order_release);
  });
  while (!destroy_started.load(std::memory_order_acquire)) {
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  const ScaleArgs args = {
      static_cast<const float *>(source.data()),
      static_cast<float *>(output.data()),
      2.5f,
      static_cast<int32_t>(input.size() - 1),
  };
  lrrt::launch(concurrent_queue, concurrent_kernel, config, args);
  if (destroy_completed.load(std::memory_order_acquire)) {
    destroyer.join();
    throw std::runtime_error(
        "module destruction blocked an independent module launch");
  }
  if (lr_launch_on_queue(wait_queue.get(), wait_kernel, &config, &wait_args,
                         sizeof(wait_args)) != LR_ERROR_INVALID_ARGUMENT) {
    destroyer.join();
    throw std::runtime_error("destroying module accepted a new launch");
  }

  destroyer.join();
  lrrt::check(destroy_status, "lr_module_destroy");
  concurrent_queue.synchronize();
  expect_value(device.get(), output.data(), input.back() * 2.5f,
               "concurrent launch during module destroy");
}

void expect_free_releases_runtime_lock(lrrt::Device device,
                                       const lrrt::Kernel &wait_kernel,
                                       void *allocation, bool host_allocation) {
  lrrt::DeviceBuffer wait_output(device, sizeof(unsigned long long));
  lrrt::Queue wait_queue(device);
  const lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
  const WaitArgs wait_args = {
      50000ULL,
      static_cast<unsigned long long *>(wait_output.data()),
  };
  lrrt::launch(wait_queue, wait_kernel, config, wait_args);

  std::atomic<bool> free_started{false};
  std::atomic<bool> free_completed{false};
  lr_status_t free_status = LR_ERROR_RUNTIME;
  std::thread freer([&] {
    free_started.store(true, std::memory_order_release);
    free_status = host_allocation ? lr_host_free(device.get(), allocation)
                                  : lr_free(device.get(), allocation);
    free_completed.store(true, std::memory_order_release);
  });
  while (!free_started.load(std::memory_order_acquire)) {
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  lr_memory_stats_t stats{};
  lrrt::check(lr_get_memory_stats(device.get(), &stats),
              "lr_get_memory_stats during free");
  const bool completed_while_probing =
      free_completed.load(std::memory_order_acquire);

  freer.join();
  lrrt::check(free_status, host_allocation ? "lr_host_free" : "lr_free");
  if (completed_while_probing) {
    throw std::runtime_error(
        "memory free held the runtime lock while waiting for GPU work");
  }
}

void test_free_releases_runtime_lock(lrrt::Device device,
                                     const std::vector<unsigned char> &hsaco) {
  lrrt::Module module(device, hsaco);
  lrrt::Kernel wait_kernel = module.kernel("queue_wait_kernel");

  void *device_allocation = nullptr;
  lrrt::check(
      lr_malloc(device.get(), sizeof(unsigned long long), &device_allocation),
      "lr_malloc free lock test");
  expect_free_releases_runtime_lock(device, wait_kernel, device_allocation,
                                    false);

  void *host_allocation = nullptr;
  lrrt::check(lr_host_malloc(device.get(), sizeof(unsigned long long),
                             &host_allocation),
              "lr_host_malloc free lock test");
  expect_free_releases_runtime_lock(device, wait_kernel, host_allocation, true);
}

void test_free_drains(lrrt::Device device,
                      const std::vector<unsigned char> &hsaco,
                      const std::vector<float> &input) {
  void *device_input = nullptr;
  void *device_output = nullptr;
  lrrt::check(
      lr_malloc(device.get(), input.size() * sizeof(float), &device_input),
      "lr_malloc input");
  lrrt::check(lr_malloc(device.get(), sizeof(float), &device_output),
              "lr_malloc output");
  lrrt::check(lr_memcpy(device.get(), device_input, input.data(),
                        input.size() * sizeof(float), LR_MEMCPY_HOST_TO_DEVICE),
              "lr_memcpy host to device");

  lrrt::Module module(device, hsaco);
  lrrt::Kernel kernel = module.kernel("async_copy_launch_kernel");
  const float alpha = 2.5f;
  const ScaleArgs args = {static_cast<const float *>(device_input),
                          static_cast<float *>(device_output), alpha,
                          static_cast<int32_t>(input.size() - 1)};
  const lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
  lrrt::launch(kernel, config, args);
  lrrt::check(lr_free(device.get(), device_output), "lr_free output");

  if (lr_memcpy(device.get(), device_output, input.data(), sizeof(float),
                LR_MEMCPY_HOST_TO_DEVICE) != LR_ERROR_INVALID_ARGUMENT) {
    throw std::runtime_error("freed allocation was accepted by copy");
  }

  lrrt::check(lr_free(device.get(), device_input), "lr_free input");
}

} // namespace

int main() {
  try {
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      printf("module_buffer_lifetime: skipped, no GPU devices\n");
      return 0;
    }
    lrrt::Device device = runtime.open_device(0);

    const int32_t n = 1 << 20;
    std::vector<float> input(n, 1.0f);
    input.back() = 3.0f;
    std::vector<unsigned char> hsaco = read_file(LRRT_ASYNC_COPY_LAUNCH_HSACO);

    test_module_destroy_drains(device, hsaco, input);
    test_module_destroy_releases_runtime_lock(device, hsaco, input);
    test_free_releases_runtime_lock(device, hsaco);
    test_free_drains(device, hsaco, input);

    printf("module_buffer_lifetime: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
