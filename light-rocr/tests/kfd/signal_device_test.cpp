#include "light_rocr/runtime/topology.hpp"
#include "light_rocr/transport/kfd/session.hpp"
#include "light_rocr/transport/kfd/signal.hpp"
#include "light_rocr/transport/kfd/topology.hpp"

#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

int main() {
  auto opened = light_rocr::transport::kfd::KfdSession::open();
  if (!opened) {
    std::cerr << opened.status.message << '\n';
    return 1;
  }
  const auto discovered =
      light_rocr::transport::kfd::discover_topology(opened.session);
  if (!discovered) {
    std::cerr << discovered.message << '\n';
    return 1;
  }
  const auto selected =
      light_rocr::runtime::select_unique_gpu(discovered.topology, "gfx1101");
  if (!selected) {
    std::cerr << selected.message << '\n';
    return 1;
  }

  const auto &node = discovered.topology.nodes[selected.node_index];
  constexpr size_t kSignalsPerPage =
      light_rocr::transport::kfd::kMemoryPageSize /
      sizeof(light_rocr::runtime::AmdSignal);
  std::vector<light_rocr::transport::kfd::UserSignal> signals;
  signals.reserve(kSignalsPerPage + 1);
  for (size_t index = 0; index < kSignalsPerPage + 1; ++index) {
    auto created = opened.session.create_user_signal(
        node, static_cast<int64_t>(index + 1));
    if (!created) {
      std::cerr << created.status.message << '\n';
      return 1;
    }
    signals.push_back(std::move(created.signal));
  }

  const uint64_t first = signals.front().gpu_handle();
  for (size_t index = 0; index < kSignalsPerPage; ++index) {
    const uint64_t expected =
        first + index * sizeof(light_rocr::runtime::AmdSignal);
    if (signals[index].gpu_handle() != expected ||
        signals[index].load_relaxed() != static_cast<int64_t>(index + 1)) {
      std::cerr << "user signals were not packed into one GTT page\n";
      return 1;
    }
  }
  if (signals[kSignalsPerPage].gpu_handle() /
          light_rocr::transport::kfd::kMemoryPageSize ==
      first / light_rocr::transport::kfd::kMemoryPageSize) {
    std::cerr << "user-signal page capacity was exceeded\n";
    return 1;
  }

  constexpr size_t kReusableSlot = 17;
  const uint64_t released_handle = signals[kReusableSlot].gpu_handle();
  const auto released = signals[kReusableSlot].release();
  if (!released) {
    std::cerr << released.message << '\n';
    return 1;
  }
  auto reused = opened.session.create_user_signal(node, 99);
  if (!reused || reused.signal.gpu_handle() != released_handle ||
      reused.signal.load_relaxed() != 99) {
    std::cerr << (reused ? "released user-signal slot was not reused"
                         : reused.status.message)
              << '\n';
    return 1;
  }

  for (auto &signal : signals) {
    const auto status = signal.release();
    if (!status) {
      std::cerr << status.message << '\n';
      return 1;
    }
  }
  const auto reused_released = reused.signal.release();
  if (!reused_released) {
    std::cerr << reused_released.message << '\n';
    return 1;
  }

  std::cout << "packed " << kSignalsPerPage
            << " user signals per GTT page and reused one slot\n";
  return 0;
}
