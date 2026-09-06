#include "light_rocr/runtime/signal.hpp"

#include <hsa/amd_hsa_signal.h>

#include <cstddef>

namespace {

using light_rocr::runtime::AmdSignal;

static_assert(sizeof(AmdSignal) == sizeof(amd_signal_t));
static_assert(alignof(AmdSignal) == alignof(amd_signal_t));
static_assert(offsetof(AmdSignal, kind) == offsetof(amd_signal_t, kind));
static_assert(offsetof(AmdSignal, value) == offsetof(amd_signal_t, value));
static_assert(offsetof(AmdSignal, event_mailbox_ptr) ==
              offsetof(amd_signal_t, event_mailbox_ptr));
static_assert(offsetof(AmdSignal, event_id) ==
              offsetof(amd_signal_t, event_id));
static_assert(offsetof(AmdSignal, start_ts) ==
              offsetof(amd_signal_t, start_ts));
static_assert(offsetof(AmdSignal, end_ts) == offsetof(amd_signal_t, end_ts));
static_assert(offsetof(AmdSignal, queue_ptr) ==
              offsetof(amd_signal_t, queue_ptr));
static_assert(light_rocr::runtime::kAmdSignalKindUser == AMD_SIGNAL_KIND_USER);

} // namespace

int main() { return 0; }
