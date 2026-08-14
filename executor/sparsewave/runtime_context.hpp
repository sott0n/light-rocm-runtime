#ifndef LRRT_EXECUTOR_SPARSEWAVE_RUNTIME_CONTEXT_HPP_
#define LRRT_EXECUTOR_SPARSEWAVE_RUNTIME_CONTEXT_HPP_

#include "lrrt/lrrt.hpp"

#include <memory>
#include <stdexcept>

namespace lrrt::executor::sparsewave {

struct RuntimeContext {
  RuntimeContext() : runtime(), device(open_device(runtime)) {}

  static lrrt::Device open_device(lrrt::Runtime &runtime) {
    if (runtime.device_count() == 0) {
      throw std::runtime_error("SparseWave requires an AMDGPU device");
    }
    return runtime.open_device(0);
  }

  lrrt::Runtime runtime;
  lrrt::Device device;
};

std::shared_ptr<RuntimeContext> get_runtime_context();

} // namespace lrrt::executor::sparsewave

#endif // LRRT_EXECUTOR_SPARSEWAVE_RUNTIME_CONTEXT_HPP_
