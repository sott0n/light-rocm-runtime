#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "executor.hpp"
#include "runtime_context.hpp"

#include <stdint.h>

#include <cstring>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

namespace lrrt::executor::sparsewave {

namespace {

constexpr const char *kCapsuleName = "lrrt_sparsewave.SparseAttentionExecutor";

struct Program {
  Program(std::shared_ptr<RuntimeContext> context, const char *manifest_path,
          lrrt::examples::sparsewave::sparse_attention::Problem problem)
      : context(std::move(context)),
        executor(this->context->device, manifest_path, problem) {}

  std::shared_ptr<RuntimeContext> context;
  lrrt::examples::sparsewave::sparse_attention::Executor executor;
};

template <typename T>
std::vector<T> copy_input(unsigned long long address, size_t count) {
  if (count == 0) {
    return {};
  }
  const auto *data =
      reinterpret_cast<const T *>(static_cast<uintptr_t>(address));
  return std::vector<T>(data, data + count);
}

void destroy_program(PyObject *capsule) {
  void *pointer = PyCapsule_GetPointer(capsule, kCapsuleName);
  if (pointer) {
    delete static_cast<Program *>(pointer);
  } else {
    PyErr_Clear();
  }
}

PyObject *load(PyObject *, PyObject *args) {
  const char *manifest_path = nullptr;
  unsigned long long output_rows = 0;
  unsigned long long key_value_rows = 0;
  unsigned long long head_dimension = 0;
  unsigned long long value_columns = 0;
  if (!PyArg_ParseTuple(args, "sKKKK", &manifest_path, &output_rows,
                        &key_value_rows, &head_dimension, &value_columns)) {
    return nullptr;
  }

  try {
    if (output_rows > UINT32_MAX || key_value_rows > UINT32_MAX ||
        head_dimension > UINT32_MAX || value_columns > UINT32_MAX) {
      throw std::invalid_argument(
          "SparseWave SparseAttention dimensions exceed uint32");
    }
    auto program = std::make_unique<Program>(
        get_runtime_context(), manifest_path,
        lrrt::examples::sparsewave::sparse_attention::Problem{
            static_cast<uint32_t>(output_rows),
            static_cast<uint32_t>(key_value_rows),
            static_cast<uint32_t>(head_dimension),
            static_cast<uint32_t>(value_columns)});
    PyObject *capsule =
        PyCapsule_New(program.get(), kCapsuleName, destroy_program);
    if (!capsule) {
      return nullptr;
    }
    program.release();
    return capsule;
  } catch (const std::exception &error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
}

PyObject *execute(PyObject *, PyObject *args) {
  PyObject *capsule = nullptr;
  unsigned long long nonzeros = 0;
  unsigned long long row_offsets_address = 0;
  unsigned long long column_indices_address = 0;
  unsigned long long mask_values_address = 0;
  unsigned long long query_address = 0;
  unsigned long long transposed_key_address = 0;
  unsigned long long value_address = 0;
  unsigned long long output_address = 0;
  if (!PyArg_ParseTuple(
          args, "OKKKKKKKK", &capsule, &nonzeros, &row_offsets_address,
          &column_indices_address, &mask_values_address, &query_address,
          &transposed_key_address, &value_address, &output_address)) {
    return nullptr;
  }

  auto *program =
      static_cast<Program *>(PyCapsule_GetPointer(capsule, kCapsuleName));
  if (!program) {
    return nullptr;
  }

  try {
    if (nonzeros > INT32_MAX) {
      throw std::invalid_argument(
          "SparseWave SparseAttention nonzeros exceed int32");
    }
    if (row_offsets_address == 0 || query_address == 0 ||
        transposed_key_address == 0 || value_address == 0 ||
        output_address == 0 ||
        (nonzeros != 0 &&
         (column_indices_address == 0 || mask_values_address == 0))) {
      throw std::invalid_argument(
          "SparseWave SparseAttention received a null tensor pointer");
    }

    const auto &problem = program->executor.problem();
    const lrrt::examples::sparsewave::sparse_attention::Inputs inputs{
        copy_input<int32_t>(row_offsets_address, problem.output_rows + 1),
        copy_input<int32_t>(column_indices_address, nonzeros),
        copy_input<float>(mask_values_address, nonzeros),
        copy_input<float>(query_address,
                          static_cast<size_t>(problem.output_rows) *
                              problem.head_dimension),
        copy_input<float>(transposed_key_address,
                          static_cast<size_t>(problem.head_dimension) *
                              problem.key_value_rows),
        copy_input<float>(value_address,
                          static_cast<size_t>(problem.key_value_rows) *
                              problem.value_columns)};
    const std::vector<float> output = program->executor.execute(inputs);
    std::memcpy(
        reinterpret_cast<void *>(static_cast<uintptr_t>(output_address)),
        output.data(), output.size() * sizeof(float));
    Py_RETURN_NONE;
  } catch (const std::exception &error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
}

PyMethodDef methods[] = {
    {"load_sparse_attention", load, METH_VARARGS,
     "Load a compiled SparseWave SparseAttention bundle."},
    {"execute_sparse_attention", execute, METH_VARARGS,
     "Execute a loaded SparseWave SparseAttention bundle."},
    {nullptr, nullptr, 0, nullptr},
};

} // namespace

int add_sparse_attention_bindings(PyObject *module) {
  return PyModule_AddFunctions(module, methods);
}

} // namespace lrrt::executor::sparsewave
