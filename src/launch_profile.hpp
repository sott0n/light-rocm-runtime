#ifndef LRRT_LAUNCH_PROFILE_HPP_
#define LRRT_LAUNCH_PROFILE_HPP_

#include "lrrt/lrrt.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace lrrt_internal {

enum class LaunchProfilePhase : size_t {
  GlobalLockWait,
  QueueLockWait,
  DependencyCollection,
  QueueCapacity,
  ResourceAcquisition,
  DependencyRegistration,
  PacketPublication,
  EventPinRelease,
  SubmissionPinRelease,
  Count,
};

struct LaunchProfile {
  uint64_t launch_count = 0;
  uint64_t total_ns = 0;
  // Diagnostic wall time from the initial registry-lock acquisition until its
  // first release. This overlaps the phase timings and is not part of their
  // categorized total.
  uint64_t initial_global_lock_hold_ns = 0;
  std::array<uint64_t, static_cast<size_t>(LaunchProfilePhase::Count)>
      phase_ns{};
};

// These hooks are private to the runtime benchmarks. Profiling is disabled by
// default and collected independently for each calling thread.
LRRT_API void set_thread_launch_profiling(bool enabled);
LRRT_API void reset_thread_launch_profile();
LRRT_API LaunchProfile thread_launch_profile();

} // namespace lrrt_internal

#endif
