#ifndef LRRT_BENCHMARKS_QWEN_HIP_GRAPH_HPP_
#define LRRT_BENCHMARKS_QWEN_HIP_GRAPH_HPP_

#include "lrrt/lrrt.h"
#include "lrrt/persistent_graph/abi.hpp"

#include <hip/hip_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace lrrt::benchmarks {

inline void check_hip(hipError_t status, const char *operation) {
  if (status == hipSuccess) {
    return;
  }
  throw std::runtime_error(std::string(operation) + ": " +
                           hipGetErrorString(status));
}

class QwenHipGraph {
public:
  template <typename HostLaunch>
  QwenHipGraph(const std::vector<unsigned char> &image,
               const std::vector<HostLaunch> &decoder_launches,
               const std::vector<HostLaunch> &tail_launches, uint32_t steps,
               uint32_t layers, bool stage_profile = false)
      : stream_(nullptr), module_(nullptr), function_(nullptr), graph_(nullptr),
        executable_(nullptr), start_(nullptr), end_(nullptr),
        stage_profile_(stage_profile) {
    try {
      check_hip(hipSetDevice(0), "hipSetDevice");
      check_hip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
                "hipStreamCreateWithFlags");
      check_hip(hipEventCreate(&start_), "hipEventCreate start");
      check_hip(hipEventCreate(&end_), "hipEventCreate end");
      check_hip(hipModuleLoadData(&module_, image.data()), "hipModuleLoadData");
      check_hip(hipModuleGetFunction(
                    &function_, module_,
                    stage_profile
                        ? "qwen_stateful_graph_matched_profile_step_kernel"
                        : "qwen_stateful_graph_matched_step_kernel"),
                "hipModuleGetFunction");
      check_hip(hipGraphCreate(&graph_, 0), "hipGraphCreate");

      const size_t node_count =
          static_cast<size_t>(steps) *
          (static_cast<size_t>(layers) * decoder_launches.size() +
           tail_launches.size());
      if (node_count > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Qwen HIP Graph has too many nodes");
      }
      nodes_.reserve(node_count);
      hipGraphNode_t previous = nullptr;
      for (uint32_t step = 0; step < steps; ++step) {
        for (uint32_t layer = 0; layer < layers; ++layer) {
          for (const HostLaunch &launch : decoder_launches) {
            previous = add_kernel(launch, previous);
          }
        }
        for (const HostLaunch &launch : tail_launches) {
          previous = add_kernel(launch, previous);
        }
      }
      check_hip(hipGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0),
                "hipGraphInstantiate");
      check_hip(hipGraphUpload(executable_, stream_), "hipGraphUpload");
      synchronize();
    } catch (...) {
      reset();
      throw;
    }
  }

  ~QwenHipGraph() { reset(); }

  QwenHipGraph(const QwenHipGraph &) = delete;
  QwenHipGraph &operator=(const QwenHipGraph &) = delete;

  void launch() {
    check_hip(hipGraphLaunch(executable_, stream_), "hipGraphLaunch");
  }

  void synchronize() {
    check_hip(hipStreamSynchronize(stream_), "hipStreamSynchronize");
  }

  void record_start() {
    check_hip(hipEventRecord(start_, stream_), "hipEventRecord start");
  }

  void record_end() {
    check_hip(hipEventRecord(end_, stream_), "hipEventRecord end");
  }

  double elapsed_time_ns() const {
    float milliseconds = 0.0f;
    check_hip(hipEventElapsedTime(&milliseconds, start_, end_),
              "hipEventElapsedTime");
    return static_cast<double>(milliseconds) * 1.0e6;
  }

  size_t node_count() const { return nodes_.size(); }

private:
  void reset() noexcept {
    if (stream_) {
      static_cast<void>(hipStreamSynchronize(stream_));
    }
    if (executable_) {
      static_cast<void>(hipGraphExecDestroy(executable_));
    }
    if (graph_) {
      static_cast<void>(hipGraphDestroy(graph_));
    }
    if (module_) {
      static_cast<void>(hipModuleUnload(module_));
    }
    if (start_) {
      static_cast<void>(hipEventDestroy(start_));
    }
    if (end_) {
      static_cast<void>(hipEventDestroy(end_));
    }
    if (stream_) {
      static_cast<void>(hipStreamDestroy(stream_));
    }
    stream_ = nullptr;
    module_ = nullptr;
    function_ = nullptr;
    graph_ = nullptr;
    executable_ = nullptr;
    start_ = nullptr;
    end_ = nullptr;
  }

  struct KernelNode {
    persistent_graph::HostStepArgs args{};
    std::array<void *, 5> parameters{};
  };

  template <typename HostLaunch>
  hipGraphNode_t add_kernel(const HostLaunch &launch,
                            hipGraphNode_t dependency) {
    if (launch.config.block.x == 0 || launch.config.block.y == 0 ||
        launch.config.block.z == 0 ||
        launch.config.grid.x % launch.config.block.x != 0 ||
        launch.config.grid.y % launch.config.block.y != 0 ||
        launch.config.grid.z % launch.config.block.z != 0) {
      throw std::runtime_error("HIP Graph launch has invalid dimensions");
    }
    nodes_.emplace_back();
    KernelNode &node = nodes_.back();
    node.args = launch.args;
    if (stage_profile_) {
      node.args.reserved = static_cast<uint32_t>(nodes_.size() - 1u);
    }
    node.parameters = {
        &node.args.state,    &node.args.tasks,   &node.args.task_count,
        &node.args.reserved, &node.args.context,
    };
    hipKernelNodeParams parameters{};
    parameters.blockDim = dim3(launch.config.block.x, launch.config.block.y,
                               launch.config.block.z);
    parameters.func = function_;
    parameters.gridDim = dim3(launch.config.grid.x / launch.config.block.x,
                              launch.config.grid.y / launch.config.block.y,
                              launch.config.grid.z / launch.config.block.z);
    parameters.kernelParams = node.parameters.data();
    parameters.sharedMemBytes = launch.config.shared_memory_bytes;

    hipGraphNode_t graph_node = nullptr;
    check_hip(hipGraphAddKernelNode(&graph_node, graph_,
                                    dependency ? &dependency : nullptr,
                                    dependency ? 1 : 0, &parameters),
              "hipGraphAddKernelNode");
    return graph_node;
  }

  hipStream_t stream_;
  hipModule_t module_;
  hipFunction_t function_;
  hipGraph_t graph_;
  hipGraphExec_t executable_;
  hipEvent_t start_;
  hipEvent_t end_;
  bool stage_profile_;
  std::vector<KernelNode> nodes_;
};

} // namespace lrrt::benchmarks

#endif // LRRT_BENCHMARKS_QWEN_HIP_GRAPH_HPP_
