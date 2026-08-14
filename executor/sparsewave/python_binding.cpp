#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "lrrt/executor/sparsewave/spmm.hpp"

#include <stdint.h>

#include <cstring>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

namespace {

constexpr const char *kSpmmCapsuleName = "lrrt_sparsewave.SpmmExecutor";

struct RuntimeContext {
  RuntimeContext() : runtime(), device(open_device(runtime)) {}

  static lrrt::Device open_device(lrrt::Runtime &runtime) {
    if (runtime.device_count() == 0) {
      throw std::runtime_error("SparseWave SpMM requires an AMDGPU device");
    }
    return runtime.open_device(0);
  }

  lrrt::Runtime runtime;
  lrrt::Device device;
};

std::weak_ptr<RuntimeContext> runtime_context;

std::shared_ptr<RuntimeContext> get_runtime_context() {
  std::shared_ptr<RuntimeContext> context = runtime_context.lock();
  if (!context) {
    context = std::make_shared<RuntimeContext>();
    runtime_context = context;
  }
  return context;
}

struct SpmmProgram {
  SpmmProgram(std::shared_ptr<RuntimeContext> context,
              const char *manifest_path,
              lrrt::executor::sparsewave::SpmmProblem problem)
      : context(std::move(context)),
        executor(this->context->device, manifest_path, problem) {}

  std::shared_ptr<RuntimeContext> context;
  lrrt::executor::sparsewave::SpmmExecutor executor;
};

template <typename T>
std::vector<T> copy_input(unsigned long long address, size_t count) {
  const auto *data =
      reinterpret_cast<const T *>(static_cast<uintptr_t>(address));
  return std::vector<T>(data, data + count);
}

void destroy_spmm_program(PyObject *capsule) {
  void *pointer = PyCapsule_GetPointer(capsule, kSpmmCapsuleName);
  if (pointer) {
    delete static_cast<SpmmProgram *>(pointer);
  } else {
    PyErr_Clear();
  }
}

PyObject *load_spmm(PyObject *, PyObject *args) {
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
    auto program = std::make_unique<SpmmProgram>(
        get_runtime_context(), manifest_path,
        lrrt::executor::sparsewave::SpmmProblem{
            static_cast<uint32_t>(rows), static_cast<uint32_t>(columns),
            static_cast<uint32_t>(rhs_columns)});
    PyObject *capsule =
        PyCapsule_New(program.get(), kSpmmCapsuleName, destroy_spmm_program);
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

PyObject *execute_spmm(PyObject *, PyObject *args) {
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

  auto *program = static_cast<SpmmProgram *>(
      PyCapsule_GetPointer(capsule, kSpmmCapsuleName));
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
    const lrrt::executor::sparsewave::CsrMatrix matrix{
        copy_input<int32_t>(row_offsets_address, problem.rows + 1),
        copy_input<int32_t>(column_indices_address, nonzeros),
        copy_input<float>(values_address, nonzeros)};
    const std::vector<float> rhs =
        copy_input<float>(rhs_address, static_cast<size_t>(problem.columns) *
                                           problem.rhs_columns);
    const std::vector<float> output = program->executor.execute(matrix, rhs);
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
    {"load_spmm", load_spmm, METH_VARARGS,
     "Load a compiled SparseWave CSR SpMM bundle."},
    {"execute_spmm", execute_spmm, METH_VARARGS,
     "Execute a loaded SparseWave CSR SpMM bundle."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_lrrt_sparsewave", nullptr, -1,
                      methods};

} // namespace

PyMODINIT_FUNC PyInit__lrrt_sparsewave() { return PyModule_Create(&module); }
