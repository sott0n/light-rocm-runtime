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

constexpr const char *kCapsuleName = "lrrt_sparsewave.SpmmExecutor";

struct Program {
  Program(std::shared_ptr<RuntimeContext> context, const char *manifest_path,
          lrrt::examples::sparsewave::spmm::Problem problem)
      : context(std::move(context)),
        executor(this->context->device, manifest_path, problem) {}

  std::shared_ptr<RuntimeContext> context;
  lrrt::examples::sparsewave::spmm::Executor executor;
};

template <typename T>
std::vector<T> copy_input(unsigned long long address, size_t count) {
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
  unsigned long long rows = 0;
  unsigned long long columns = 0;
  unsigned long long rhs_columns = 0;
  if (!PyArg_ParseTuple(args, "sKKK", &manifest_path, &rows, &columns,
                        &rhs_columns)) {
    return nullptr;
  }

  try {
    if (rows > UINT32_MAX || columns > UINT32_MAX || rhs_columns > UINT32_MAX) {
      throw std::invalid_argument("SparseWave SpMM dimensions exceed uint32");
    }
    auto program = std::make_unique<Program>(
        get_runtime_context(), manifest_path,
        lrrt::examples::sparsewave::spmm::Problem{
            static_cast<uint32_t>(rows), static_cast<uint32_t>(columns),
            static_cast<uint32_t>(rhs_columns)});
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
  unsigned long long values_address = 0;
  unsigned long long rhs_address = 0;
  unsigned long long output_address = 0;
  if (!PyArg_ParseTuple(args, "OKKKKKK", &capsule, &nonzeros,
                        &row_offsets_address, &column_indices_address,
                        &values_address, &rhs_address, &output_address)) {
    return nullptr;
  }

  auto *program =
      static_cast<Program *>(PyCapsule_GetPointer(capsule, kCapsuleName));
  if (!program) {
    return nullptr;
  }

  try {
    if (nonzeros > INT32_MAX) {
      throw std::invalid_argument("SparseWave SpMM nonzeros exceed int32");
    }
    if (row_offsets_address == 0 || column_indices_address == 0 ||
        values_address == 0 || rhs_address == 0 || output_address == 0) {
      throw std::invalid_argument(
          "SparseWave SpMM received a null tensor pointer");
    }

    const auto &problem = program->executor.problem();
    const lrrt::examples::sparsewave::spmm::Inputs inputs{
        copy_input<int32_t>(row_offsets_address, problem.rows + 1),
        copy_input<int32_t>(column_indices_address, nonzeros),
        copy_input<float>(values_address, nonzeros),
        copy_input<float>(rhs_address, static_cast<size_t>(problem.columns) *
                                           problem.rhs_columns)};
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
    {"load_spmm", load, METH_VARARGS,
     "Load a compiled SparseWave CSR SpMM bundle."},
    {"execute_spmm", execute, METH_VARARGS,
     "Execute a loaded SparseWave CSR SpMM bundle."},
    {nullptr, nullptr, 0, nullptr},
};

} // namespace

int add_spmm_bindings(PyObject *module) {
  return PyModule_AddFunctions(module, methods);
}

} // namespace lrrt::executor::sparsewave
