#ifndef LIGHT_ROCR_TRANSPORT_HSAKMT_CODE_CACHE_HPP
#define LIGHT_ROCR_TRANSPORT_HSAKMT_CODE_CACHE_HPP

#include "light_rocr/transport/hsakmt/memory.hpp"
#include "light_rocr/transport/hsakmt/queue.hpp"

#include <cstdint>
#include <string>

namespace light_rocr::transport::hsakmt {

class ExecutableImage;

enum class CodeCacheError {
  None,
  InvalidArgument,
  AllocateCommand,
  AllocateSignal,
  QueueFull,
  ReservePacket,
  RingDoorbell,
  WaitForCompletion,
  ReleaseSignal,
  ReleaseCommand,
};

struct CodeCacheStatus {
  CodeCacheError error = CodeCacheError::None;
  uint32_t hsakmt_status = 0;
  std::string message;

  explicit operator bool() const { return error == CodeCacheError::None; }
};

// The caller must serialize producers for queue and ensure that no executable
// allocation being replaced is still in use by another queue.
[[nodiscard]] CodeCacheStatus
freeze_executable_image(const KfdSession &session, uint32_t gpu_node_id,
                        AqlQueue &queue, const ExecutableImage &image);

[[nodiscard]] const char *code_cache_error_name(CodeCacheError error);

} // namespace light_rocr::transport::hsakmt

#endif
