#include <cmath>
#include <cstdio>

#include "executor/iree/registration/init.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/function_io.h"
#include "iree/tooling/function_util.h"
#include "iree/vm/bytecode/module.h"

namespace {

template <typename T, void (*ReleaseFn)(T *)> class ScopedPtr {
public:
  ScopedPtr() = default;
  ScopedPtr(const ScopedPtr &) = delete;
  ScopedPtr &operator=(const ScopedPtr &) = delete;
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

bool report_status(iree_status_t status, const char *label) {
  if (iree_status_is_ok(status)) {
    return true;
  }
  std::fprintf(stderr, "%s failed\n", label);
  iree_status_fprint(stderr, status);
  iree_status_free(status);
  return false;
}

iree_status_t replace_buffer_view_input(iree_vm_list_t *inputs,
                                        iree_host_size_t index,
                                        iree_hal_buffer_view_t *view) {
  iree_vm_ref_t ref = iree_hal_buffer_view_retain_ref(view);
  iree_status_t status = iree_vm_list_set_ref_move(inputs, index, &ref);
  if (!iree_status_is_ok(status)) {
    iree_vm_ref_release(&ref);
  }
  return status;
}

iree_status_t pop_front_buffer_view(iree_vm_list_t *outputs,
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

iree_status_t
invoke_step(iree_vm_context_t *context, const iree_vm_function_t *function,
            iree_hal_device_t *device, iree_hal_allocator_t *allocator,
            iree_hal_buffer_view_t *key_cache,
            iree_hal_buffer_view_t *value_cache, const char *new_value_spec,
            BufferViewPtr *out_key_cache, BufferViewPtr *out_value_cache,
            BufferViewPtr *out_context) {
  VmListPtr inputs;
  const iree_vm_function_signature_t signature =
      iree_vm_function_signature(function);
  iree_string_view_t arguments_cconv = iree_string_view_empty();
  iree_string_view_t results_cconv = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_vm_function_call_get_cconv_fragments(
      &signature, &arguments_cconv, &results_cconv));
  (void)results_cconv;

  const iree_string_view_t specs[] = {
      IREE_SV("2x2xf32=1 2 3 4"),
      IREE_SV("2x3xf32=0 0 0 0 0 0"),
      IREE_SV("2xf32=0 0"),
      IREE_SV("3x2xf32=2 4 4 6 0 0"),
      iree_make_cstring_view(new_value_spec),
      IREE_SV("2xf32=1 0"),
      IREE_SV("2xf32=0 1"),
  };
  const iree_string_view_list_t spec_list = {IREE_ARRAYSIZE(specs), specs};

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_tooling_parse_variants(arguments_cconv, spec_list, device,
                                         allocator, iree_allocator_system(),
                                         inputs.out());
  }
  if (iree_status_is_ok(status) && key_cache) {
    status = replace_buffer_view_input(inputs.get(), 1, key_cache);
  }
  if (iree_status_is_ok(status) && value_cache) {
    status = replace_buffer_view_input(inputs.get(), 3, value_cache);
  }

  VmListPtr outputs;
  if (iree_status_is_ok(status)) {
    status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 3,
                                 iree_allocator_system(), outputs.out());
  }
  iree_hal_fence_t *finish_fence = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_tooling_append_async_fences(inputs.get(), *function, device,
                                         /*wait_fence=*/NULL, &finish_fence);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(context, *function, IREE_VM_INVOCATION_FLAG_NONE,
                            /*policy=*/NULL, inputs.get(), outputs.get(),
                            iree_allocator_system());
  }
  if (iree_status_is_ok(status) && finish_fence) {
    status = iree_hal_fence_wait(finish_fence, iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_fence_release(finish_fence);
  if (iree_status_is_ok(status)) {
    status = pop_front_buffer_view(outputs.get(), out_key_cache);
  }
  if (iree_status_is_ok(status)) {
    status = pop_front_buffer_view(outputs.get(), out_value_cache);
  }
  if (iree_status_is_ok(status)) {
    status = pop_front_buffer_view(outputs.get(), out_context);
  }
  return status;
}

bool expect_context(iree_hal_buffer_view_t *context,
                    const float (&expected)[4]) {
  if (iree_hal_buffer_view_shape_rank(context) != 2 ||
      iree_hal_buffer_view_shape_dim(context, 0) != 2 ||
      iree_hal_buffer_view_shape_dim(context, 1) != 2) {
    std::fprintf(stderr, "unexpected context shape\n");
    return false;
  }

  float values[4] = {};
  iree_hal_buffer_t *buffer = iree_hal_buffer_view_buffer(context);
  if (!report_status(
          iree_hal_buffer_map_read(buffer, 0, values, sizeof(values)),
          "iree_hal_buffer_map_read(context)")) {
    return false;
  }

  for (int i = 0; i < 4; ++i) {
    if (std::fabs(values[i] - expected[i]) > 1e-4f) {
      std::fprintf(stderr, "context[%d]: expected %.6g, got %.6g\n", i,
                   expected[i], values[i]);
      return false;
    }
  }
  return true;
}

iree_status_t run_smoke(const char *module_path) {
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_register_all());

  iree_allocator_t host_allocator = iree_allocator_system();
  VmInstancePtr instance;
  IREE_RETURN_IF_ERROR(
      iree_tooling_create_instance(host_allocator, instance.out()));

  iree_io_file_contents_t *flatbuffer_contents = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_io_file_contents_read(iree_make_cstring_view(module_path),
                                 host_allocator, &flatbuffer_contents));
  VmModulePtr module;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
      instance.get(), IREE_VM_BYTECODE_MODULE_FLAG_NONE,
      flatbuffer_contents->const_buffer,
      iree_io_file_contents_deallocator(flatbuffer_contents), host_allocator,
      module.out()));

  iree_vm_module_t *user_modules[1] = {module.get()};
  VmContextPtr context;
  DeviceListPtr device_list;
  AllocatorPtr allocator;
  IREE_RETURN_IF_ERROR(iree_tooling_create_context_from_flags(
      instance.get(), 1, user_modules, IREE_SV("lrrt"), host_allocator,
      context.out(), device_list.out(), allocator.out(),
      /*out_replay_recorder=*/NULL));

  iree_hal_device_t *device = iree_hal_device_list_at(device_list.get(), 0);

  iree_vm_function_t function;
  IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_name(
      module.get(), IREE_VM_FUNCTION_LINKAGE_EXPORT,
      IREE_SV("token_step_kv_cache_outputs"), &function));

  BufferViewPtr first_key_cache;
  BufferViewPtr first_value_cache;
  BufferViewPtr first_context;
  IREE_RETURN_IF_ERROR(invoke_step(
      context.get(), &function, device, allocator.get(), /*key_cache=*/nullptr,
      /*value_cache=*/nullptr, "2xf32=6 8", &first_key_cache,
      &first_value_cache, &first_context));

  const float first_expected[4] = {4.0f, 6.0f, 4.0f, 6.0f};
  if (!expect_context(first_context.get(), first_expected)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "first context did not match expected values");
  }

  BufferViewPtr second_key_cache;
  BufferViewPtr second_value_cache;
  BufferViewPtr second_context;
  IREE_RETURN_IF_ERROR(
      invoke_step(context.get(), &function, device, allocator.get(),
                  first_key_cache.get(), first_value_cache.get(), "2xf32=8 10",
                  &second_key_cache, &second_value_cache, &second_context));

  const float second_expected[4] = {14.0f / 3.0f, 20.0f / 3.0f, 14.0f / 3.0f,
                                    20.0f / 3.0f};
  if (!expect_context(second_context.get(), second_expected)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "second context did not match expected values");
  }

  return iree_ok_status();
}

} // namespace

int main(int argc, char **argv) {
  iree_flags_set_usage("lrrt_iree_token_step_kv_cache_device_resident_smoke",
                       "Runs token-step KV cache twice without host handoff.");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <token_step_kv_cache_outputs.vmfb>\n",
                 argv[0]);
    return 2;
  }

  iree_status_t status = run_smoke(argv[1]);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }

  std::puts("iree_token_step_kv_cache_device_resident: ok");
  return 0;
}
