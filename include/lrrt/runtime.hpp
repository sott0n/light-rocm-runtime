#ifndef LRRT_RUNTIME_HPP_
#define LRRT_RUNTIME_HPP_

#include "lrrt/error.hpp"

#include <stdint.h>
#include <string>
#include <vector>

namespace lrrt {

using MemoryStats = lr_memory_stats_t;

class Device {
public:
  explicit Device(lr_device_t device) : device_(device) {}

  lr_device_t get() const { return device_; }
  uint32_t index() const { return device_.index; }
  std::string name() const {
    char name[64] = {};
    check(lr_device_name(device_, name, sizeof(name)), "lr_device_name");
    return std::string(name);
  }
  void synchronize() const { check(lr_synchronize(device_), "lr_synchronize"); }
  MemoryStats memory_stats() const {
    MemoryStats stats{};
    check(lr_get_memory_stats(device_, &stats), "lr_get_memory_stats");
    return stats;
  }
  void reset_memory_stats() const {
    check(lr_reset_memory_stats(device_), "lr_reset_memory_stats");
  }

private:
  lr_device_t device_;
};

inline void synchronize(Device device) {
  check(lr_synchronize(device.get()), "lr_synchronize");
}

class Queue {
public:
  explicit Queue(lr_device_t device) : queue_(nullptr) {
    check(lr_queue_create(device, &queue_), "lr_queue_create");
  }

  explicit Queue(Device device) : Queue(device.get()) {}

  ~Queue() { reset(); }

  Queue(Queue &&other) noexcept : queue_(other.queue_) {
    other.queue_ = nullptr;
  }

  Queue(const Queue &) = delete;
  Queue &operator=(const Queue &) = delete;

  Queue &operator=(Queue &&other) noexcept {
    if (this != &other) {
      reset();
      queue_ = other.queue_;
      other.queue_ = nullptr;
    }
    return *this;
  }

  lr_queue_t *get() const { return queue_; }

  void synchronize() const {
    check(lr_queue_synchronize(queue_), "lr_queue_synchronize");
  }

private:
  void reset() noexcept {
    if (queue_) {
      lr_queue_destroy(queue_);
      queue_ = nullptr;
    }
  }

  lr_queue_t *queue_;
};

class Event {
public:
  explicit Event(lr_device_t device) : event_(nullptr) {
    check(lr_event_create(device, &event_), "lr_event_create");
  }

  explicit Event(Device device) : Event(device.get()) {}

  ~Event() { reset(); }

  Event(Event &&other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
  }

  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;

  Event &operator=(Event &&other) noexcept {
    if (this != &other) {
      reset();
      event_ = other.event_;
      other.event_ = nullptr;
    }
    return *this;
  }

  lr_event_t *get() const { return event_; }

  void record() const { check(lr_event_record(event_), "lr_event_record"); }

  void record(const Queue &queue) const {
    check(lr_event_record_on_queue(event_, queue.get()),
          "lr_event_record_on_queue");
  }

  void synchronize() const {
    check(lr_event_synchronize(event_), "lr_event_synchronize");
  }

private:
  void reset() noexcept {
    if (event_) {
      lr_event_destroy(event_);
      event_ = nullptr;
    }
  }

  lr_event_t *event_;
};

inline std::vector<lr_event_t *>
event_handles(const std::vector<const Event *> &dependencies) {
  std::vector<lr_event_t *> handles;
  handles.reserve(dependencies.size());
  for (const Event *dependency : dependencies) {
    handles.push_back(dependency ? dependency->get() : nullptr);
  }
  return handles;
}

inline uint64_t elapsed_time_ns(const Event &start, const Event &end) {
  uint64_t elapsed_ns = 0;
  check(lr_event_elapsed_time_ns(start.get(), end.get(), &elapsed_ns),
        "lr_event_elapsed_time_ns");
  return elapsed_ns;
}

inline uint64_t duration_ns(const Event &event) {
  uint64_t value = 0;
  check(lr_event_duration_ns(event.get(), &value), "lr_event_duration_ns");
  return value;
}

class Runtime {
public:
  Runtime() { check(lr_init(), "lr_init"); }
  ~Runtime() { lr_shutdown(); }

  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

  uint32_t device_count() const {
    uint32_t count = 0;
    check(lr_device_count(&count), "lr_device_count");
    return count;
  }

  Device open_device(uint32_t index) const {
    lr_device_t device = {0};
    check(lr_device_open(index, &device), "lr_device_open");
    return Device(device);
  }
};

} // namespace lrrt

#endif // LRRT_RUNTIME_HPP_
