#ifndef LRRT_EXECUTOR_SPARSEWAVE_APPLICATION_BINDINGS_HPP_
#define LRRT_EXECUTOR_SPARSEWAVE_APPLICATION_BINDINGS_HPP_

#define PY_SSIZE_T_CLEAN
#include <Python.h>

namespace lrrt::executor::sparsewave {

int add_spmm_bindings(PyObject *module);
int add_sparse_attention_bindings(PyObject *module);

} // namespace lrrt::executor::sparsewave

#endif // LRRT_EXECUTOR_SPARSEWAVE_APPLICATION_BINDINGS_HPP_
