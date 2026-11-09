#ifndef LRRT_BENCHMARKS_PERSISTENT_GRAPH_CONTEXT_HPP_
#define LRRT_BENCHMARKS_PERSISTENT_GRAPH_CONTEXT_HPP_

#include <cstdint>
#include <type_traits>

namespace lrrt::benchmarks::persistent_graph {

constexpr uint32_t kContextMagic = 0x5047424du;
constexpr uint32_t kIncrementOpcode = 0;

struct Context {
  uint32_t magic;
  uint32_t task_count;
  uint32_t *task_counters;
};

static_assert(std::is_standard_layout_v<Context>);

} // namespace lrrt::benchmarks::persistent_graph

#endif // LRRT_BENCHMARKS_PERSISTENT_GRAPH_CONTEXT_HPP_
