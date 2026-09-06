#ifndef LIGHT_ROCR_TRANSPORT_HSAKMT_KERNARG_HPP
#define LIGHT_ROCR_TRANSPORT_HSAKMT_KERNARG_HPP

#include "light_rocr/loader/code_object.hpp"
#include "light_rocr/runtime/launch.hpp"
#include "light_rocr/transport/hsakmt/memory.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace light_rocr::transport::hsakmt {

using KernargBufferError = runtime::KernargBufferError;

struct KernargBufferStatus {
  KernargBufferError error = KernargBufferError::None;
  MemoryStatus memory_status;
  std::string message;

  explicit operator bool() const { return error == KernargBufferError::None; }
};

struct KernargBufferResult;

class KernargBuffer {
public:
  KernargBuffer() = default;
  KernargBuffer(const KernargBuffer &) = delete;
  KernargBuffer &operator=(const KernargBuffer &) = delete;
  KernargBuffer(KernargBuffer &&other) noexcept;
  KernargBuffer &operator=(KernargBuffer &&) noexcept = delete;
  ~KernargBuffer() = default;

  [[nodiscard]] const void *host_address() const {
    return buffer_ ? allocation_.host_address() : nullptr;
  }
  [[nodiscard]] uint64_t gpu_address() const { return buffer_.gpu_address(); }
  [[nodiscard]] uint64_t allocation_size() const {
    return buffer_ ? allocation_.size() : 0;
  }
  [[nodiscard]] uint32_t kernarg_size() const { return buffer_.kernarg_size(); }
  [[nodiscard]] uint32_t alignment() const { return buffer_.alignment(); }
  [[nodiscard]] const runtime::KernargBufferInfo &runtime_buffer() const {
    return buffer_;
  }
  [[nodiscard]] bool owns_allocation() const {
    return static_cast<bool>(allocation_);
  }
  explicit operator bool() const { return static_cast<bool>(buffer_); }

  [[nodiscard]] MemoryStatus release();

private:
  friend KernargBufferResult create_kernarg_buffer(const KfdSession &, uint32_t,
                                                   const loader::KernelInfo &,
                                                   const void *, size_t);

  KernargBuffer(MemoryAllocation allocation, runtime::KernargBufferInfo buffer)
      : allocation_(std::move(allocation)), buffer_(std::move(buffer)) {}

  void invalidate();

  MemoryAllocation allocation_;
  runtime::KernargBufferInfo buffer_;
};

struct KernargBufferResult {
  KernargBufferStatus status;
  KernargBuffer buffer;

  explicit operator bool() const { return static_cast<bool>(status); }
};

[[nodiscard]] KernargBufferResult
create_kernarg_buffer(const KfdSession &session, uint32_t gpu_node_id,
                      const loader::KernelInfo &kernel, const void *arguments,
                      size_t arguments_size);

[[nodiscard]] const char *kernarg_buffer_error_name(KernargBufferError error);

} // namespace light_rocr::transport::hsakmt

#endif
