#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "application_bindings.hpp"
#include "runtime_context.hpp"

#include <memory>

namespace lrrt::executor::sparsewave {

namespace {

std::weak_ptr<RuntimeContext> runtime_context;

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_lrrt_sparsewave", nullptr, -1,
                      nullptr};

} // namespace

std::shared_ptr<RuntimeContext> get_runtime_context() {
  std::shared_ptr<RuntimeContext> context = runtime_context.lock();
  if (!context) {
    context = std::make_shared<RuntimeContext>();
    runtime_context = context;
  }
  return context;
}

} // namespace lrrt::executor::sparsewave

PyMODINIT_FUNC PyInit__lrrt_sparsewave() {
  PyObject *module = PyModule_Create(&lrrt::executor::sparsewave::module);
  if (!module) {
    return nullptr;
  }
  if (lrrt::executor::sparsewave::add_spmm_bindings(module) < 0 ||
      lrrt::executor::sparsewave::add_sparse_attention_bindings(module) < 0) {
    Py_DECREF(module);
    return nullptr;
  }
  return module;
}
