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

struct GateArgs {
  const unsigned long long *gate;
};

std::vector<unsigned char> read_file(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("failed to open multiple-queue HSACO");
  }
  std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("multiple-queue HSACO is empty");
  }
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> data(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    throw std::runtime_error("failed to read multiple-queue HSACO");
  }
  return data;
}

void expect_value(const lrrt::DeviceBuffer &buffer, float expected,
                  const char *scenario) {
  float actual = 0.0f;
  lrrt::copy_to_host(&actual, buffer, sizeof(actual));
  if (actual != expected) {
    fprintf(stderr, "multiple_queues %s: actual=%f expected=%f\n", scenario,
            actual, expected);
    throw std::runtime_error("multiple-queue result mismatch");
  }
}

} // namespace

int main() {
  try {
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      printf("multiple_queues: skipped, no GPU devices\n");
      return 0;
    }
    lrrt::Device device = runtime.open_device(0);

    const int32_t n = 1 << 20;
    const float alpha = 2.5f;
    std::vector<float> input(n, 1.0f);
    input.back() = 3.0f;
    lrrt::DeviceBuffer source(device, input.size() * sizeof(float));
    lrrt::DeviceBuffer staging(device, input.size() * sizeof(float));
    lrrt::DeviceBuffer first_output(device, sizeof(float));
    lrrt::DeviceBuffer copied_output(device, sizeof(float));
    lrrt::DeviceBuffer final_output(device, sizeof(float));
    lrrt::DeviceBuffer wait_output(device, sizeof(unsigned long long));
    lrrt::copy_to_device(source, input);

    std::vector<unsigned char> hsaco = read_file(LRRT_ASYNC_COPY_LAUNCH_HSACO);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel("async_copy_launch_kernel");
    lrrt::Kernel wait_kernel = module.kernel("queue_wait_kernel");
    lrrt::Kernel gate_kernel = module.kernel("queue_gate_kernel");
    lrrt::Queue first_queue(device);
    lrrt::Queue second_queue(device);
    const lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};

    const WaitArgs wait_args = {
        50000ULL,
        static_cast<unsigned long long *>(wait_output.data()),
    };
    lrrt::launch(first_queue, wait_kernel, config, wait_args);
    std::atomic<bool> synchronize_started{false};
    std::atomic<bool> synchronize_completed{false};
    std::thread synchronizer([&] {
      synchronize_started.store(true, std::memory_order_release);
      first_queue.synchronize();
      synchronize_completed.store(true, std::memory_order_release);
    });
    while (!synchronize_started.load(std::memory_order_acquire)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    const ScaleArgs concurrent_args = {
        static_cast<const float *>(source.data()),
        static_cast<float *>(copied_output.data()), alpha, n - 1};
    lrrt::launch(second_queue, kernel, config, concurrent_args);
    if (synchronize_completed.load(std::memory_order_acquire)) {
      synchronizer.join();
      throw std::runtime_error(
          "queue synchronization blocked an independent queue launch");
    }
    synchronizer.join();
    second_queue.synchronize();

    lrrt::launch(first_queue, wait_kernel, config, wait_args);
    lrrt::Event synchronized_event(device);
    synchronized_event.record(first_queue);
    std::atomic<bool> event_synchronize_started{false};
    std::atomic<bool> event_synchronize_completed{false};
    std::thread event_synchronizer([&] {
      event_synchronize_started.store(true, std::memory_order_release);
      synchronized_event.synchronize();
      event_synchronize_completed.store(true, std::memory_order_release);
    });
    while (!event_synchronize_started.load(std::memory_order_acquire)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    lrrt::launch(second_queue, kernel, config, concurrent_args);
    if (event_synchronize_completed.load(std::memory_order_acquire)) {
      event_synchronizer.join();
      throw std::runtime_error(
          "event synchronization blocked an independent queue launch");
    }
    event_synchronizer.join();
    second_queue.synchronize();

    lrrt::launch(first_queue, wait_kernel, config, wait_args);
    std::atomic<bool> device_synchronize_started{false};
    std::atomic<bool> device_synchronize_completed{false};
    std::thread device_synchronizer([&] {
      device_synchronize_started.store(true, std::memory_order_release);
      device.synchronize();
      device_synchronize_completed.store(true, std::memory_order_release);
    });
    while (!device_synchronize_started.load(std::memory_order_acquire)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    lrrt::launch(second_queue, kernel, config, concurrent_args);
    if (device_synchronize_completed.load(std::memory_order_acquire)) {
      device_synchronizer.join();
      throw std::runtime_error(
          "device synchronization blocked a later queue launch");
    }
    device_synchronizer.join();
    second_queue.synchronize();

    lr_queue_t *device_synchronized_queue = nullptr;
    lrrt::check(lr_queue_create(device.get(), &device_synchronized_queue),
                "lr_queue_create");
    lrrt::check(lr_launch_on_queue(device_synchronized_queue, wait_kernel.get(),
                                   &config, &wait_args, sizeof(wait_args)),
                "lr_launch_on_queue");
    std::atomic<bool> queue_device_synchronize_started{false};
    std::atomic<bool> queue_device_synchronize_completed{false};
    lr_status_t queue_device_synchronize_status = LR_ERROR_RUNTIME;
    std::thread queue_device_synchronizer([&] {
      queue_device_synchronize_started.store(true, std::memory_order_release);
      queue_device_synchronize_status = lr_synchronize(device.get());
      queue_device_synchronize_completed.store(true, std::memory_order_release);
    });
    while (!queue_device_synchronize_started.load(std::memory_order_acquire)) {
    }
    lrrt::launch(second_queue, kernel, config, concurrent_args);
    if (queue_device_synchronize_completed.load(std::memory_order_acquire)) {
      queue_device_synchronizer.join();
      lrrt::check(lr_queue_destroy(device_synchronized_queue),
                  "lr_queue_destroy");
      throw std::runtime_error(
          "device synchronization completed before queue destroy test");
    }

    std::atomic<bool> queue_destroy_completed{false};
    lr_status_t queue_destroy_status = LR_ERROR_RUNTIME;
    std::thread queue_destroyer([&] {
      queue_destroy_status = lr_queue_destroy(device_synchronized_queue);
      queue_destroy_completed.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (queue_destroy_completed.load(std::memory_order_acquire)) {
      queue_device_synchronizer.join();
      queue_destroyer.join();
      throw std::runtime_error(
          "queue destroy did not wait for device synchronization");
    }
    queue_device_synchronizer.join();
    queue_destroyer.join();
    lrrt::check(queue_device_synchronize_status, "lr_synchronize");
    lrrt::check(queue_destroy_status, "lr_queue_destroy");
    second_queue.synchronize();

    lr_event_t *device_synchronized_event = nullptr;
    lrrt::check(lr_event_create(device.get(), &device_synchronized_event),
                "lr_event_create");
    lrrt::launch(first_queue, wait_kernel, config, wait_args);
    lrrt::check(
        lr_event_record_on_queue(device_synchronized_event, first_queue.get()),
        "lr_event_record_on_queue");
    std::atomic<bool> event_device_synchronize_started{false};
    std::atomic<bool> event_device_synchronize_completed{false};
    lr_status_t event_device_synchronize_status = LR_ERROR_RUNTIME;
    std::thread event_device_synchronizer([&] {
      event_device_synchronize_started.store(true, std::memory_order_release);
      event_device_synchronize_status = lr_synchronize(device.get());
      event_device_synchronize_completed.store(true, std::memory_order_release);
    });
    while (!event_device_synchronize_started.load(std::memory_order_acquire)) {
    }
    lrrt::launch(second_queue, kernel, config, concurrent_args);
    if (event_device_synchronize_completed.load(std::memory_order_acquire)) {
      event_device_synchronizer.join();
      lrrt::check(lr_event_destroy(device_synchronized_event),
                  "lr_event_destroy");
      throw std::runtime_error(
          "device synchronization completed before event destroy test");
    }

    std::atomic<bool> event_destroy_completed{false};
    lr_status_t event_destroy_status = LR_ERROR_RUNTIME;
    std::thread event_destroyer([&] {
      event_destroy_status = lr_event_destroy(device_synchronized_event);
      event_destroy_completed.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (event_destroy_completed.load(std::memory_order_acquire)) {
      event_device_synchronizer.join();
      event_destroyer.join();
      throw std::runtime_error(
          "event destroy did not wait for device synchronization");
    }
    event_device_synchronizer.join();
    event_destroyer.join();
    lrrt::check(event_device_synchronize_status, "lr_synchronize");
    lrrt::check(event_destroy_status, "lr_event_destroy");
    second_queue.synchronize();

    lr_event_t *reused_while_synchronizing = nullptr;
    lrrt::check(lr_event_create(device.get(), &reused_while_synchronizing),
                "lr_event_create");
    lrrt::launch(first_queue, wait_kernel, config, wait_args);
    lrrt::check(
        lr_event_record_on_queue(reused_while_synchronizing, first_queue.get()),
        "lr_event_record_on_queue");
    std::atomic<bool> reuse_synchronize_started{false};
    lr_status_t reuse_synchronize_status = LR_ERROR_RUNTIME;
    std::thread reuse_synchronizer([&] {
      reuse_synchronize_started.store(true, std::memory_order_release);
      reuse_synchronize_status =
          lr_event_synchronize(reused_while_synchronizing);
    });
    while (!reuse_synchronize_started.load(std::memory_order_acquire)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    lrrt::check(
        lr_event_record_on_queue(reused_while_synchronizing, first_queue.get()),
        "lr_event_record_on_queue");
    reuse_synchronizer.join();
    lrrt::check(reuse_synchronize_status, "lr_event_synchronize");
    lrrt::check(lr_event_synchronize(reused_while_synchronizing),
                "lr_event_synchronize");
    lrrt::check(lr_event_destroy(reused_while_synchronizing),
                "lr_event_destroy");

    lr_event_t *destroyed_while_synchronizing = nullptr;
    lrrt::check(lr_event_create(device.get(), &destroyed_while_synchronizing),
                "lr_event_create");
    lrrt::launch(first_queue, wait_kernel, config, wait_args);
    lrrt::check(lr_event_record_on_queue(destroyed_while_synchronizing,
                                         first_queue.get()),
                "lr_event_record_on_queue");
    std::atomic<bool> destroy_synchronize_started{false};
    lr_status_t destroy_synchronize_status = LR_ERROR_RUNTIME;
    std::thread destroy_synchronizer([&] {
      destroy_synchronize_started.store(true, std::memory_order_release);
      destroy_synchronize_status =
          lr_event_synchronize(destroyed_while_synchronizing);
    });
    while (!destroy_synchronize_started.load(std::memory_order_acquire)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    lrrt::check(lr_event_destroy(destroyed_while_synchronizing),
                "lr_event_destroy");
    destroy_synchronizer.join();
    lrrt::check(destroy_synchronize_status, "lr_event_synchronize");

    lr_queue_t *lifetime_queue = nullptr;
    lrrt::check(lr_queue_create(device.get(), &lifetime_queue),
                "lr_queue_create");
    lrrt::check(lr_launch_on_queue(lifetime_queue, wait_kernel.get(), &config,
                                   &wait_args, sizeof(wait_args)),
                "lr_launch_on_queue");
    std::atomic<bool> lifetime_synchronize_started{false};
    lr_status_t lifetime_synchronize_status = LR_ERROR_RUNTIME;
    std::thread lifetime_synchronizer([&] {
      lifetime_synchronize_started.store(true, std::memory_order_release);
      lifetime_synchronize_status = lr_queue_synchronize(lifetime_queue);
    });
    while (!lifetime_synchronize_started.load(std::memory_order_acquire)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    lrrt::check(lr_queue_destroy(lifetime_queue), "lr_queue_destroy");
    lifetime_synchronizer.join();
    lrrt::check(lifetime_synchronize_status, "lr_queue_synchronize");

    lrrt::Queue backpressure_queue(device);
    lrrt::Queue independent_queue(device);
    lrrt::DeviceBuffer gate(device, sizeof(unsigned long long));
    const unsigned long long closed_gate = 0;
    lrrt::copy_to_device(gate, &closed_gate, sizeof(closed_gate));
    const GateArgs gate_args = {
        static_cast<const unsigned long long *>(gate.data()),
    };
    const WaitArgs filler_args = {
        0,
        static_cast<unsigned long long *>(wait_output.data()),
    };

    void *synchronous_copy_device = nullptr;
    void *synchronous_copy_host = nullptr;
    lrrt::check(lr_malloc(device.get(), sizeof(unsigned long long),
                          &synchronous_copy_device),
                "lr_malloc synchronous copy device buffer");
    lrrt::check(lr_host_malloc(device.get(), sizeof(unsigned long long),
                               &synchronous_copy_host),
                "lr_host_malloc synchronous copy host buffer");
    *static_cast<unsigned long long *>(synchronous_copy_host) = 42;

    lrrt::launch(first_queue, gate_kernel, config, gate_args);
    std::atomic<bool> synchronous_copy_started{false};
    std::atomic<bool> synchronous_copy_completed{false};
    lr_status_t synchronous_copy_status = LR_ERROR_RUNTIME;
    std::thread synchronous_copier([&] {
      synchronous_copy_started.store(true, std::memory_order_release);
      synchronous_copy_status = lr_memcpy(
          device.get(), synchronous_copy_device, synchronous_copy_host,
          sizeof(unsigned long long), LR_MEMCPY_HOST_TO_DEVICE);
      synchronous_copy_completed.store(true, std::memory_order_release);
    });
    while (!synchronous_copy_started.load(std::memory_order_acquire)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    lrrt::launch(independent_queue, kernel, config, concurrent_args);
    if (synchronous_copy_completed.load(std::memory_order_acquire)) {
      const unsigned long long open_gate_after_failure = 1;
      lrrt::Event gate_opened_after_failure(device);
      lrrt::copy_to_device_async(gate, &open_gate_after_failure,
                                 sizeof(open_gate_after_failure),
                                 gate_opened_after_failure);
      synchronous_copier.join();
      gate_opened_after_failure.synchronize();
      lrrt::check(lr_free(device.get(), synchronous_copy_device),
                  "lr_free failed synchronous copy buffer");
      lrrt::check(lr_host_free(device.get(), synchronous_copy_host),
                  "lr_host_free failed synchronous copy buffer");
      throw std::runtime_error(
          "synchronous copy did not remain blocked by earlier queue work");
    }

    std::atomic<bool> device_free_completed{false};
    std::atomic<bool> host_free_completed{false};
    lr_status_t device_free_status = LR_ERROR_RUNTIME;
    lr_status_t host_free_status = LR_ERROR_RUNTIME;
    std::thread device_freer([&] {
      device_free_status = lr_free(device.get(), synchronous_copy_device);
      device_free_completed.store(true, std::memory_order_release);
    });
    std::thread host_freer([&] {
      host_free_status = lr_host_free(device.get(), synchronous_copy_host);
      host_free_completed.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    if (device_free_completed.load(std::memory_order_acquire) ||
        host_free_completed.load(std::memory_order_acquire)) {
      const unsigned long long open_gate_after_failure = 1;
      lrrt::Event gate_opened_after_failure(device);
      lrrt::copy_to_device_async(gate, &open_gate_after_failure,
                                 sizeof(open_gate_after_failure),
                                 gate_opened_after_failure);
      synchronous_copier.join();
      device_freer.join();
      host_freer.join();
      throw std::runtime_error(
          "synchronous copy allocations were not pinned while waiting");
    }

    const unsigned long long open_gate_for_synchronous_copy = 1;
    lrrt::Event synchronous_copy_gate_opened(device);
    lrrt::copy_to_device_async(gate, &open_gate_for_synchronous_copy,
                               sizeof(open_gate_for_synchronous_copy),
                               synchronous_copy_gate_opened);
    synchronous_copier.join();
    device_freer.join();
    host_freer.join();
    lrrt::check(synchronous_copy_status, "lr_memcpy concurrent wait");
    lrrt::check(device_free_status, "lr_free concurrent copy");
    lrrt::check(host_free_status, "lr_host_free concurrent copy");
    synchronous_copy_gate_opened.synchronize();
    first_queue.synchronize();
    independent_queue.synchronize();

    lrrt::copy_to_device(gate, &closed_gate, sizeof(closed_gate));
    constexpr uint32_t queue_capacity = 1024;
    lrrt::launch(backpressure_queue, gate_kernel, config, gate_args);
    for (uint32_t i = 1; i < queue_capacity; ++i) {
      lrrt::launch(backpressure_queue, wait_kernel, config, filler_args);
    }

    std::atomic<bool> backpressure_launch_started{false};
    std::atomic<bool> backpressure_launch_completed{false};
    std::thread backpressure_launcher([&] {
      backpressure_launch_started.store(true, std::memory_order_release);
      lrrt::launch(backpressure_queue, wait_kernel, config, filler_args);
      backpressure_launch_completed.store(true, std::memory_order_release);
    });
    while (!backpressure_launch_started.load(std::memory_order_acquire)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    lrrt::launch(independent_queue, kernel, config, concurrent_args);
    if (backpressure_launch_completed.load(std::memory_order_acquire)) {
      backpressure_launcher.join();
      throw std::runtime_error(
          "queue backpressure did not overlap an independent queue launch");
    }
    const unsigned long long open_gate = 1;
    lrrt::Event gate_opened(device);
    lrrt::copy_to_device_async(gate, &open_gate, sizeof(open_gate),
                               gate_opened);
    backpressure_launcher.join();
    gate_opened.synchronize();
    backpressure_queue.synchronize();
    independent_queue.synchronize();

    const ScaleArgs first_args = {static_cast<const float *>(source.data()),
                                  static_cast<float *>(first_output.data()),
                                  alpha, n - 1};
    const ScaleArgs second_args = {
        static_cast<const float *>(first_output.data()),
        static_cast<float *>(final_output.data()), alpha, 0};

    lrrt::copy_to_device(gate, &closed_gate, sizeof(closed_gate));
    lrrt::Event consumer_wait_event(device);
    consumer_wait_event.record(first_queue);
    lrrt::launch(second_queue, gate_kernel, config, gate_args,
                 {&consumer_wait_event});

    std::atomic<bool> consumer_wait_started{false};
    std::atomic<bool> consumer_wait_completed{false};
    lr_status_t consumer_wait_status = LR_ERROR_RUNTIME;
    std::thread consumer_waiter([&] {
      consumer_wait_started.store(true, std::memory_order_release);
      consumer_wait_status = lr_event_record_on_queue(consumer_wait_event.get(),
                                                      first_queue.get());
      consumer_wait_completed.store(true, std::memory_order_release);
    });
    while (!consumer_wait_started.load(std::memory_order_acquire)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    lrrt::launch(independent_queue, kernel, config, concurrent_args);
    if (consumer_wait_completed.load(std::memory_order_acquire)) {
      consumer_waiter.join();
      throw std::runtime_error(
          "event consumer wait did not remain blocked by its consumer queue");
    }

    lrrt::Event consumer_gate_opened(device);
    lrrt::copy_to_device_async(gate, &open_gate, sizeof(open_gate),
                               consumer_gate_opened);
    consumer_waiter.join();
    lrrt::check(consumer_wait_status, "lr_event_record_on_queue");
    consumer_gate_opened.synchronize();
    second_queue.synchronize();
    independent_queue.synchronize();

    lrrt::copy_to_device(gate, &closed_gate, sizeof(closed_gate));
    lrrt::Event copy_consumer_wait_event(device);
    copy_consumer_wait_event.record(first_queue);
    lrrt::launch(second_queue, gate_kernel, config, gate_args);
    lrrt::Event copy_consumer_gate(device);
    copy_consumer_gate.record(second_queue);
    lrrt::Event dependent_copy_wait_event(device);
    lrrt::copy_device_to_device_async(
        copied_output, first_output, sizeof(float), dependent_copy_wait_event,
        {&copy_consumer_wait_event, &copy_consumer_gate});

    std::atomic<bool> copy_consumer_wait_started{false};
    std::atomic<bool> copy_consumer_wait_completed{false};
    lr_status_t copy_consumer_wait_status = LR_ERROR_RUNTIME;
    std::thread copy_consumer_waiter([&] {
      copy_consumer_wait_started.store(true, std::memory_order_release);
      copy_consumer_wait_status = lr_event_record_on_queue(
          copy_consumer_wait_event.get(), first_queue.get());
      copy_consumer_wait_completed.store(true, std::memory_order_release);
    });
    while (!copy_consumer_wait_started.load(std::memory_order_acquire)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    lrrt::launch(independent_queue, kernel, config, concurrent_args);
    if (copy_consumer_wait_completed.load(std::memory_order_acquire)) {
      copy_consumer_waiter.join();
      throw std::runtime_error(
          "event consumer wait did not remain blocked by its copy consumer");
    }

    lrrt::Event copy_consumer_gate_opened(device);
    lrrt::copy_to_device_async(gate, &open_gate, sizeof(open_gate),
                               copy_consumer_gate_opened);
    copy_consumer_waiter.join();
    lrrt::check(copy_consumer_wait_status, "lr_event_record_on_queue");
    copy_consumer_gate_opened.synchronize();
    dependent_copy_wait_event.synchronize();
    second_queue.synchronize();
    independent_queue.synchronize();

    lrrt::launch(first_queue, kernel, config, first_args);
    lrrt::Event first_complete(device);
    first_complete.record(first_queue);
    lrrt::launch(second_queue, kernel, config, second_args, {&first_complete});
    second_queue.synchronize();
    expect_value(final_output, input.back() * alpha * alpha, "cross-queue");

    const ScaleArgs same_queue_first_args = {
        static_cast<const float *>(source.data()),
        static_cast<float *>(first_output.data()), alpha, n - 1};
    const ScaleArgs same_queue_second_args = {
        static_cast<const float *>(first_output.data()),
        static_cast<float *>(final_output.data()), alpha, 0};
    lrrt::launch(first_queue, kernel, config, same_queue_first_args);
    lrrt::launch(first_queue, kernel, config, same_queue_second_args);
    first_queue.synchronize();
    expect_value(final_output, input.back() * alpha * alpha, "same-queue");

    lrrt::Event input_copied(device);
    lrrt::copy_device_to_device_async(staging, source, source.size(),
                                      input_copied, {});
    const ScaleArgs copied_args = {static_cast<const float *>(staging.data()),
                                   static_cast<float *>(first_output.data()),
                                   alpha, n - 1};
    lrrt::launch(first_queue, kernel, config, copied_args, {&input_copied});
    lrrt::Event kernel_complete(device);
    kernel_complete.record(first_queue);
    lrrt::Event output_copied(device);
    lrrt::copy_device_to_device_async(copied_output, first_output,
                                      sizeof(float), output_copied,
                                      {&kernel_complete});
    const ScaleArgs final_args = {
        static_cast<const float *>(copied_output.data()),
        static_cast<float *>(final_output.data()), alpha, 0};
    lrrt::launch(second_queue, kernel, config, final_args, {&output_copied});
    second_queue.synchronize();
    expect_value(final_output, input.back() * alpha * alpha,
                 "copy-compute-chain");

    lrrt::Event reusable_event(device);
    lrrt::launch(first_queue, kernel, config, first_args);
    reusable_event.record(first_queue);
    lrrt::launch(second_queue, kernel, config, second_args, {&reusable_event});
    lrrt::Event dependent_copy(device);
    lrrt::copy_device_to_device_async(copied_output, first_output,
                                      sizeof(float), dependent_copy,
                                      {&reusable_event});

    reusable_event.record(first_queue);
    second_queue.synchronize();
    dependent_copy.synchronize();
    expect_value(final_output, input.back() * alpha * alpha,
                 "event queue consumer");
    expect_value(copied_output, input.back() * alpha, "event copy consumer");

    const ScaleArgs reused_args = {
        static_cast<const float *>(copied_output.data()),
        static_cast<float *>(final_output.data()), alpha, 0};
    lrrt::launch(second_queue, kernel, config, reused_args, {&reusable_event});
    lrrt::copy_device_to_device_async(copied_output, final_output,
                                      sizeof(float), reusable_event, {});
    reusable_event.synchronize();
    expect_value(copied_output, input.back() * alpha * alpha,
                 "event reused for copy");

    {
      lrrt::Event destroyed_event(device);
      lrrt::launch(first_queue, kernel, config, first_args);
      destroyed_event.record(first_queue);
      lrrt::launch(second_queue, kernel, config, second_args,
                   {&destroyed_event});
    }
    expect_value(final_output, input.back() * alpha * alpha,
                 "destroyed event consumer");

    lrrt::DeviceBuffer first_burst_output(device, sizeof(float));
    lrrt::DeviceBuffer second_burst_output(device, sizeof(float));
    const ScaleArgs first_burst_args = {
        static_cast<const float *>(source.data()),
        static_cast<float *>(first_burst_output.data()), alpha, n - 1};
    const ScaleArgs second_burst_args = {
        static_cast<const float *>(source.data()),
        static_cast<float *>(second_burst_output.data()), alpha, n - 1};
    constexpr uint32_t burst_launches = 2048;
    for (uint32_t i = 0; i < burst_launches; ++i) {
      lrrt::launch(first_queue, kernel, config, first_burst_args);
      lrrt::launch(second_queue, kernel, config, second_burst_args);
    }
    first_queue.synchronize();
    second_queue.synchronize();
    expect_value(first_burst_output, input.back() * alpha,
                 "first saturated queue");
    expect_value(second_burst_output, input.back() * alpha,
                 "second saturated queue");

    lr_queue_t *destroyed_queue = nullptr;
    lrrt::check(lr_queue_create(device.get(), &destroyed_queue),
                "lr_queue_create");
    lrrt::check(lr_queue_destroy(destroyed_queue), "lr_queue_destroy");
    if (lr_queue_synchronize(destroyed_queue) != LR_ERROR_INVALID_ARGUMENT) {
      throw std::runtime_error("destroyed queue was accepted");
    }

    printf("multiple_queues: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
