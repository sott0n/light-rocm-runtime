#ifndef LIGHT_ROCR_TRANSPORT_KFD_CODE_CACHE_HPP
#define LIGHT_ROCR_TRANSPORT_KFD_CODE_CACHE_HPP

#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/kfd/memory.hpp"
#include "light_rocr/transport/kfd/queue.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <utility>

namespace light_rocr::transport::kfd {

class ExecutableImage;

enum class CodeCacheError {
  None,
  InvalidArgument,
  AllocateState,
  AllocateCommand,
  AllocateSignal,
  QueueFull,
  ReservePacket,
  RingDoorbell,
  WaitForCompletion,
  DestroyQueue,
  ReleaseSignal,
  ReleaseCommand,
};

struct CodeCacheStatus {
  CodeCacheError error = CodeCacheError::None;
  int system_error = 0;
  std::string message;

  explicit operator bool() const { return error == CodeCacheError::None; }
};

struct CodeCacheOperationState;
struct CodeCacheResult;

// Owns temporary command and signal storage when cache invalidation cannot
// finish cleanup. If a submitted packet is still pending, release() destroys
// the same queue before releasing GPU-visible storage.
class CodeCacheOperation {
public:
  CodeCacheOperation();
  CodeCacheOperation(const CodeCacheOperation &) = delete;
  CodeCacheOperation &operator=(const CodeCacheOperation &) = delete;
  CodeCacheOperation(CodeCacheOperation &&other) noexcept;
  CodeCacheOperation &operator=(CodeCacheOperation &&) noexcept = delete;
  ~CodeCacheOperation();

  [[nodiscard]] bool owns_resources() const;
  explicit operator bool() const { return owns_resources(); }
  [[nodiscard]] CodeCacheStatus release(AqlQueue &queue);

private:
  friend struct CodeCacheResult;
  friend CodeCacheResult
  freeze_executable_image(const KfdSession &, const runtime::Node &, AqlQueue &,
                          const ExecutableImage &,
                          std::chrono::steady_clock::time_point,
                          const std::string &);

  explicit CodeCacheOperation(std::unique_ptr<CodeCacheOperationState> state);
  void release_for_destruction() noexcept;

  std::unique_ptr<CodeCacheOperationState> state_;
};

struct CodeCacheResult {
  CodeCacheStatus status;
  // A failed operation may retain GPU-visible resources for explicit retry.
  CodeCacheOperation operation;

  explicit operator bool() const { return static_cast<bool>(status); }
};

// The caller must serialize producers for queue and ensure that no executable
// allocation being replaced is still in use by another queue.
[[nodiscard]] CodeCacheResult
freeze_executable_image(const KfdSession &session, const runtime::Node &node,
                        AqlQueue &queue, const ExecutableImage &image,
                        std::chrono::steady_clock::time_point deadline,
                        const std::string &dri_root = "/dev/dri");

[[nodiscard]] const char *code_cache_error_name(CodeCacheError error);

} // namespace light_rocr::transport::kfd

#endif
