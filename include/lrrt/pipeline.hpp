#ifndef LRRT_PIPELINE_HPP_
#define LRRT_PIPELINE_HPP_

#include "lrrt/memory.hpp"

#include <array>

namespace lrrt {

class PinnedHostDoubleBuffer {
public:
  class Slot {
  public:
    void *host_data() const { return host_.data(); }
    size_t size() const { return host_.size(); }
    DeviceBuffer &device_buffer() { return device_; }
    const DeviceBuffer &device_buffer() const { return device_; }
    const Event &copy_complete() const { return copy_complete_; }

  private:
    friend class PinnedHostDoubleBuffer;

    enum class State { available, acquired, copy_submitted, work_submitted };

    Slot(Device device, size_t size)
        : host_(device, size), device_(device, size), copy_complete_(device),
          work_complete_(device) {}

    PinnedHostBuffer host_;
    DeviceBuffer device_;
    Event copy_complete_;
    Event work_complete_;
    State state_ = State::available;
  };

  PinnedHostDoubleBuffer(Device device, size_t slot_size)
      : slots_{Slot(device, slot_size), Slot(device, slot_size)} {}

  PinnedHostDoubleBuffer(const PinnedHostDoubleBuffer &) = delete;
  PinnedHostDoubleBuffer &operator=(const PinnedHostDoubleBuffer &) = delete;

  Slot &acquire() {
    Slot &slot = slots_[next_slot_];
    if (slot.state_ == Slot::State::work_submitted) {
      slot.work_complete_.synchronize();
      slot.state_ = Slot::State::available;
    }
    if (slot.state_ != Slot::State::available) {
      throw Error(LR_ERROR_INVALID_ARGUMENT,
                  "double-buffer slot is already acquired");
    }
    slot.state_ = Slot::State::acquired;
    return slot;
  }

  void copy_to_device_async(Slot &slot, size_t size) {
    require_slot(slot);
    if (slot.state_ != Slot::State::acquired) {
      throw Error(LR_ERROR_INVALID_ARGUMENT,
                  "double-buffer slot must be acquired before copy");
    }
    if (size == 0 || size > slot.size()) {
      throw Error(LR_ERROR_INVALID_ARGUMENT,
                  "double-buffer copy size is out of range");
    }
    lrrt::copy_to_device_async(slot.device_, slot.host_.data(), size,
                               slot.copy_complete_, {});
    slot.state_ = Slot::State::copy_submitted;
  }

  void mark_work_submitted(Slot &slot, const Queue &queue) {
    require_slot(slot);
    if (slot.state_ != Slot::State::copy_submitted) {
      throw Error(LR_ERROR_INVALID_ARGUMENT,
                  "double-buffer copy must be submitted before device work");
    }
    slot.work_complete_.record(queue);
    slot.state_ = Slot::State::work_submitted;
    next_slot_ ^= 1;
  }

  void finish() {
    for (Slot &slot : slots_) {
      if (slot.state_ == Slot::State::work_submitted) {
        slot.work_complete_.synchronize();
      } else if (slot.state_ == Slot::State::copy_submitted) {
        slot.copy_complete_.synchronize();
      }
      slot.state_ = Slot::State::available;
    }
  }

private:
  void require_slot(const Slot &slot) const {
    if (&slot != &slots_[0] && &slot != &slots_[1]) {
      throw Error(LR_ERROR_INVALID_ARGUMENT,
                  "slot does not belong to this double buffer");
    }
  }

  std::array<Slot, 2> slots_;
  size_t next_slot_ = 0;
};

} // namespace lrrt

#endif // LRRT_PIPELINE_HPP_
