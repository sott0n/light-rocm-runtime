#ifndef LRRT_EXECUTOR_IREE_VMFB_RUNNER_HPP_
#define LRRT_EXECUTOR_IREE_VMFB_RUNNER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "executor/iree/registration/init.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
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
  iree_status_t initialize(const char *module_path, const char *function_name) {
    IREE_RETURN_IF_ERROR(lrrt_iree_hal_register_all());

    iree_allocator_t host_allocator = iree_allocator_system();
    IREE_RETURN_IF_ERROR(
        iree_tooling_create_instance(host_allocator, instance_.out()));

    FileContentsPtr flatbuffer_contents;
    IREE_RETURN_IF_ERROR(
        iree_io_file_contents_read(iree_make_cstring_view(module_path),
                                   host_allocator, flatbuffer_contents.out()));
    iree_status_t status = iree_vm_bytecode_module_create(
        instance_.get(), IREE_VM_BYTECODE_MODULE_FLAG_NONE,
        flatbuffer_contents.get()->const_buffer,
        iree_io_file_contents_deallocator(flatbuffer_contents.get()),
        host_allocator, module_.out());
    if (iree_status_is_ok(status)) {
      flatbuffer_contents.release();
    }
    IREE_RETURN_IF_ERROR(status);

    iree_vm_module_t *user_modules[1] = {module_.get()};
    IREE_RETURN_IF_ERROR(iree_tooling_create_context_from_flags(
        instance_.get(), 1, user_modules, IREE_SV("lrrt"), host_allocator,
        context_.out(), device_list_.out(), allocator_.out(),
        /*out_replay_recorder=*/NULL));
    device_ = iree_hal_device_list_at(device_list_.get(), 0);

    return iree_vm_module_lookup_function_by_name(
        module_.get(), IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_make_cstring_view(function_name), &function_);
  }

  iree_status_t invoke(const std::vector<std::string> &input_specs,
                       const std::vector<BufferViewReplacement> &replacements,
                       iree_host_size_t output_count,
                       std::vector<BufferViewPtr> *outputs) {
    outputs->clear();

    const iree_vm_function_signature_t signature =
        iree_vm_function_signature(&function_);
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
        iree_tooling_append_async_fences(inputs.get(), function_, device_,
                                         /*wait_fence=*/NULL, &finish_fence);
    if (iree_status_is_ok(status)) {
      status = iree_vm_invoke(context_.get(), function_,
                              IREE_VM_INVOCATION_FLAG_NONE,
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
  VmInstancePtr instance_;
  VmModulePtr module_;
  VmContextPtr context_;
  DeviceListPtr device_list_;
  AllocatorPtr allocator_;
  iree_hal_device_t *device_ = nullptr;
  iree_vm_function_t function_ = {};
};

} // namespace lrrt::iree_executor

#endif // LRRT_EXECUTOR_IREE_VMFB_RUNNER_HPP_
