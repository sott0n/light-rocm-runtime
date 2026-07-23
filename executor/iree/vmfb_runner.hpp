#ifndef LRRT_EXECUTOR_IREE_VMFB_RUNNER_HPP_
#define LRRT_EXECUTOR_IREE_VMFB_RUNNER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "executor/iree/registration/init.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/buffer_view_util.h"
#include "iree/io/file_contents.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/function_io.h"
#include "iree/tooling/function_util.h"
#include "iree/vm/bytecode/module.h"

namespace lrrt::iree_executor {

template <typename T, void (*ReleaseFn)(T *)> class ScopedPtr {
public:
  ScopedPtr() = default;
  explicit ScopedPtr(T *value) : ptr_(value) {}
  ScopedPtr(const ScopedPtr &) = delete;
  ScopedPtr &operator=(const ScopedPtr &) = delete;
  ScopedPtr(ScopedPtr &&other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }
  ScopedPtr &operator=(ScopedPtr &&other) noexcept {
    if (this != &other) {
      reset(other.ptr_);
      other.ptr_ = nullptr;
    }
    return *this;
  }
  ~ScopedPtr() { reset(); }

  T *get() const { return ptr_; }
  T **out() {
    reset();
    return &ptr_;
  }
  void reset(T *value = nullptr) {
    if (ptr_) {
      ReleaseFn(ptr_);
    }
    ptr_ = value;
  }
  T *release() {
    T *value = ptr_;
    ptr_ = nullptr;
    return value;
  }

private:
  T *ptr_ = nullptr;
};

using VmInstancePtr = ScopedPtr<iree_vm_instance_t, iree_vm_instance_release>;
using VmContextPtr = ScopedPtr<iree_vm_context_t, iree_vm_context_release>;
using VmModulePtr = ScopedPtr<iree_vm_module_t, iree_vm_module_release>;
using VmListPtr = ScopedPtr<iree_vm_list_t, iree_vm_list_release>;
using DeviceListPtr =
    ScopedPtr<iree_hal_device_list_t, iree_hal_device_list_free>;
using AllocatorPtr =
    ScopedPtr<iree_hal_allocator_t, iree_hal_allocator_release>;
using BufferViewPtr =
    ScopedPtr<iree_hal_buffer_view_t, iree_hal_buffer_view_release>;
using FileContentsPtr =
    ScopedPtr<iree_io_file_contents_t, iree_io_file_contents_free>;

struct BufferViewReplacement {
  iree_host_size_t index = 0;
  iree_hal_buffer_view_t *view = nullptr;
};

inline iree_status_t replace_buffer_view_input(iree_vm_list_t *inputs,
                                               iree_host_size_t index,
                                               iree_hal_buffer_view_t *view) {
  iree_vm_ref_t ref = iree_hal_buffer_view_retain_ref(view);
  iree_status_t status = iree_vm_list_set_ref_move(inputs, index, &ref);
  if (!iree_status_is_ok(status)) {
    iree_vm_ref_release(&ref);
  }
  return status;
}

inline iree_status_t pop_front_buffer_view(iree_vm_list_t *outputs,
                                           BufferViewPtr *out_view) {
  iree_vm_ref_t ref = {0};
  IREE_RETURN_IF_ERROR(iree_vm_list_pop_front_ref_move(outputs, &ref));

  iree_hal_buffer_view_t *view = nullptr;
  iree_status_t status = iree_hal_buffer_view_check_deref(ref, &view);
  if (iree_status_is_ok(status)) {
    out_view->reset(view);
    ref.ptr = nullptr;
    ref.type = IREE_VM_REF_TYPE_NULL;
  }
  iree_vm_ref_release(&ref);
  return status;
}

class VmfbRunner {
public:
  iree_status_t initialize(const char *module_path) {
    std::vector<const char *> module_paths = {module_path};
    return initialize(module_paths);
  }

  iree_status_t initialize(const std::vector<const char *> &module_paths) {
    if (module_paths.empty()) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "at least one VMFB module path is required");
    }
    IREE_RETURN_IF_ERROR(lrrt_iree_hal_register_all());

    iree_allocator_t host_allocator = iree_allocator_system();
    IREE_RETURN_IF_ERROR(
        iree_tooling_create_instance(host_allocator, instance_.out()));

    modules_.clear();
    modules_.reserve(module_paths.size());
    std::vector<iree_vm_module_t *> user_modules;
    user_modules.reserve(module_paths.size());
    for (const char *path : module_paths) {
      VmModulePtr module;
      IREE_RETURN_IF_ERROR(load_module(path, &module));
      user_modules.push_back(module.get());
      modules_.push_back(std::move(module));
    }

    IREE_RETURN_IF_ERROR(iree_tooling_create_context_from_flags(
        instance_.get(), user_modules.size(), user_modules.data(),
        IREE_SV("lrrt"), host_allocator, context_.out(), device_list_.out(),
        allocator_.out(),
        /*out_replay_recorder=*/NULL));
    device_ = iree_hal_device_list_at(device_list_.get(), 0);

    return iree_ok_status();
  }

  iree_status_t initialize(const char *module_path, const char *function_name) {
    IREE_RETURN_IF_ERROR(initialize(module_path));
    return lookup_function(function_name, &function_);
  }

  iree_status_t lookup_function(const char *function_name,
                                iree_vm_function_t *out_function) {
    for (VmModulePtr &module : modules_) {
      iree_status_t status = iree_vm_module_lookup_function_by_name(
          module.get(), IREE_VM_FUNCTION_LINKAGE_EXPORT,
          iree_make_cstring_view(function_name), out_function);
      if (iree_status_is_ok(status)) {
        return status;
      }
      iree_status_ignore(status);
    }
    return iree_make_status(IREE_STATUS_NOT_FOUND, "VMFB export not found: %s",
                            function_name);
  }

  iree_status_t invoke(const std::vector<std::string> &input_specs,
                       const std::vector<BufferViewReplacement> &replacements,
                       iree_host_size_t output_count,
                       std::vector<BufferViewPtr> *outputs) {
    return invoke(function_, input_specs, replacements, output_count, outputs);
  }

  iree_status_t invoke(const iree_vm_function_t &function,
                       const std::vector<std::string> &input_specs,
                       const std::vector<BufferViewReplacement> &replacements,
                       iree_host_size_t output_count,
                       std::vector<BufferViewPtr> *outputs) {
    outputs->clear();

    const iree_vm_function_signature_t signature =
        iree_vm_function_signature(&function);
    iree_string_view_t arguments_cconv = iree_string_view_empty();
    iree_string_view_t results_cconv = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_vm_function_call_get_cconv_fragments(
        &signature, &arguments_cconv, &results_cconv));
    (void)results_cconv;

    std::vector<iree_string_view_t> specs;
    specs.reserve(input_specs.size());
    for (const std::string &spec : input_specs) {
      specs.push_back(iree_make_string_view(spec.data(), spec.size()));
    }
    const iree_string_view_list_t spec_list = {specs.size(), specs.data()};

    VmListPtr inputs;
    IREE_RETURN_IF_ERROR(iree_tooling_parse_variants(
        arguments_cconv, spec_list, device_, allocator_.get(),
        iree_allocator_system(), inputs.out()));

    for (const BufferViewReplacement &replacement : replacements) {
      IREE_RETURN_IF_ERROR(replace_buffer_view_input(
          inputs.get(), replacement.index, replacement.view));
    }

    VmListPtr raw_outputs;
    IREE_RETURN_IF_ERROR(
        iree_vm_list_create(iree_vm_make_undefined_type_def(), output_count,
                            iree_allocator_system(), raw_outputs.out()));

    iree_hal_fence_t *finish_fence = NULL;
    iree_status_t status =
        iree_tooling_append_async_fences(inputs.get(), function, device_,
                                         /*wait_fence=*/NULL, &finish_fence);
    if (iree_status_is_ok(status)) {
      status =
          iree_vm_invoke(context_.get(), function, IREE_VM_INVOCATION_FLAG_NONE,
                         /*policy=*/NULL, inputs.get(), raw_outputs.get(),
                         iree_allocator_system());
    }
    if (iree_status_is_ok(status) && finish_fence) {
      status = iree_hal_fence_wait(finish_fence, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
    }
    iree_hal_fence_release(finish_fence);
    IREE_RETURN_IF_ERROR(status);

    outputs->reserve(output_count);
    for (iree_host_size_t i = 0; i < output_count; ++i) {
      BufferViewPtr output;
      IREE_RETURN_IF_ERROR(pop_front_buffer_view(raw_outputs.get(), &output));
      outputs->push_back(std::move(output));
    }
    return iree_ok_status();
  }

  iree_status_t make_f32_buffer_view(const std::vector<float> &data,
                                     const std::vector<iree_hal_dim_t> &shape,
                                     BufferViewPtr *out_view) {
    const iree_hal_buffer_params_t buffer_params = {
        .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
    };
    const iree_const_byte_span_t initial_data = {
        reinterpret_cast<const uint8_t *>(data.data()),
        data.size() * sizeof(float),
    };
    return iree_hal_buffer_view_allocate_buffer_copy(
        device_, allocator_.get(), shape.size(), shape.data(),
        IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        buffer_params, initial_data, out_view->out());
  }

  iree_status_t invoke_views(const iree_vm_function_t &function,
                             const std::vector<iree_hal_buffer_view_t *> &views,
                             iree_host_size_t output_count,
                             std::vector<BufferViewPtr> *outputs) {
    outputs->clear();

    VmListPtr inputs;
    IREE_RETURN_IF_ERROR(
        iree_vm_list_create(iree_vm_make_undefined_type_def(), views.size(),
                            iree_allocator_system(), inputs.out()));
    for (iree_hal_buffer_view_t *view : views) {
      iree_vm_ref_t ref = iree_hal_buffer_view_retain_ref(view);
      iree_status_t status = iree_vm_list_push_ref_move(inputs.get(), &ref);
      if (!iree_status_is_ok(status)) {
        iree_vm_ref_release(&ref);
        return status;
      }
    }

    VmListPtr raw_outputs;
    IREE_RETURN_IF_ERROR(
        iree_vm_list_create(iree_vm_make_undefined_type_def(), output_count,
                            iree_allocator_system(), raw_outputs.out()));

    iree_hal_fence_t *finish_fence = NULL;
    iree_status_t status =
        iree_tooling_append_async_fences(inputs.get(), function, device_,
                                         /*wait_fence=*/NULL, &finish_fence);
    if (iree_status_is_ok(status)) {
      status =
          iree_vm_invoke(context_.get(), function, IREE_VM_INVOCATION_FLAG_NONE,
                         /*policy=*/NULL, inputs.get(), raw_outputs.get(),
                         iree_allocator_system());
    }
    if (iree_status_is_ok(status) && finish_fence) {
      status = iree_hal_fence_wait(finish_fence, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
    }
    iree_hal_fence_release(finish_fence);
    IREE_RETURN_IF_ERROR(status);

    outputs->reserve(output_count);
    for (iree_host_size_t i = 0; i < output_count; ++i) {
      BufferViewPtr output;
      IREE_RETURN_IF_ERROR(pop_front_buffer_view(raw_outputs.get(), &output));
      outputs->push_back(std::move(output));
    }
    return iree_ok_status();
  }

private:
  iree_status_t load_module(const char *module_path, VmModulePtr *out_module) {
    iree_allocator_t host_allocator = iree_allocator_system();
    FileContentsPtr flatbuffer_contents;
    IREE_RETURN_IF_ERROR(
        iree_io_file_contents_read(iree_make_cstring_view(module_path),
                                   host_allocator, flatbuffer_contents.out()));
    iree_status_t status = iree_vm_bytecode_module_create(
        instance_.get(), IREE_VM_BYTECODE_MODULE_FLAG_NONE,
        flatbuffer_contents.get()->const_buffer,
        iree_io_file_contents_deallocator(flatbuffer_contents.get()),
        host_allocator, out_module->out());
    if (iree_status_is_ok(status)) {
      flatbuffer_contents.release();
    }
    return status;
  }

  VmInstancePtr instance_;
  std::vector<VmModulePtr> modules_;
  VmContextPtr context_;
  DeviceListPtr device_list_;
  AllocatorPtr allocator_;
  iree_hal_device_t *device_ = nullptr;
  iree_vm_function_t function_ = {};
};

} // namespace lrrt::iree_executor

#endif // LRRT_EXECUTOR_IREE_VMFB_RUNNER_HPP_
