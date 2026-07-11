#include "hal_driver_test_common.hpp"

using namespace lrrt::iree::test;

namespace {

bool test_executable_cache_contract() {
  HalFixture fixture;
  if (!fixture.setup()) {
    return false;
  }

  ExecutableCachePtr executable_cache;
  if (!expect_ok(iree_hal_executable_cache_create(fixture.device.get(),
                                                  IREE_SV("test-cache"),
                                                  executable_cache.out()),
                 "executable cache create")) {
    return false;
  }

  if (!iree_hal_executable_cache_can_prepare_format(
          executable_cache.get(), (iree_hal_executable_caching_mode_t)0,
          IREE_SV("rocm-hsaco")) ||
      !iree_hal_executable_cache_can_prepare_format(
          executable_cache.get(), (iree_hal_executable_caching_mode_t)0,
          IREE_SV("amdgpu-hsaco")) ||
      iree_hal_executable_cache_can_prepare_format(
          executable_cache.get(), (iree_hal_executable_caching_mode_t)0,
          IREE_SV("vmfb"))) {
    return false;
  }

  const uint8_t hsaco_prefix[4] = {0x7f, 'E', 'L', 'F'};
  char inferred_format[32] = {};
  iree_host_size_t inferred_size = 0;
  if (!expect_ok(
          iree_hal_executable_cache_infer_format(
              executable_cache.get(), (iree_hal_executable_caching_mode_t)0,
              iree_make_const_byte_span(hsaco_prefix, sizeof(hsaco_prefix)),
              sizeof(inferred_format), inferred_format, &inferred_size),
          "infer hsaco format") ||
      std::strcmp(inferred_format, "rocm-hsaco") != 0 ||
      inferred_size != sizeof(hsaco_prefix)) {
    return false;
  }

  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.executable_format = IREE_SV("rocm-hsaco");
  executable_params.executable_data =
      iree_make_const_byte_span(hsaco_prefix, sizeof(hsaco_prefix));
  ExecutablePtr executable;
  iree_status_t status = iree_hal_executable_cache_prepare_executable(
      executable_cache.get(), &executable_params, executable.out());
  if (iree_status_is_ok(status) || executable.get()) {
    iree_status_ignore(status);
    return false;
  }
  iree_status_ignore(status);
  return true;
}

bool test_hsaco_dispatch() {
#if defined(LRRT_IREE_PROBE_HSACO_PATH) && defined(LRRT_IREE_PROBE_SYMBOL)
  HalFixture fixture;
  if (!fixture.setup()) {
    return false;
  }

  ExecutableCachePtr executable_cache;
  if (!expect_ok(iree_hal_executable_cache_create(fixture.device.get(),
                                                  IREE_SV("test-cache"),
                                                  executable_cache.out()),
                 "executable cache create")) {
    return false;
  }

  std::vector<uint8_t> hsaco;
  if (!read_file(LRRT_IREE_PROBE_HSACO_PATH, &hsaco)) {
    return false;
  }
  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.executable_format = IREE_SV("rocm-hsaco");
  executable_params.executable_data =
      iree_make_const_byte_span(hsaco.data(), hsaco.size());
  ExecutablePtr executable;
  if (!expect_ok(
          iree_hal_executable_cache_prepare_executable(
              executable_cache.get(), &executable_params, executable.out()),
          "prepare hsaco executable") ||
      iree_hal_executable_function_count(executable.get()) != 0) {
    return false;
  }

  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  if (!expect_ok(
          iree_hal_executable_lookup_function_by_name(
              executable.get(), IREE_SV(LRRT_IREE_PROBE_SYMBOL), &function),
          "lookup hsaco function") ||
      !iree_hal_executable_function_is_index_in_range(function, 1) ||
      iree_hal_executable_function_count(executable.get()) != 1) {
    return false;
  }

  iree_hal_executable_function_info_t function_info;
  if (!expect_ok(iree_hal_executable_function_info(executable.get(), function,
                                                   &function_info),
                 "query function info") ||
      !string_view_equal(function_info.name, LRRT_IREE_PROBE_SYMBOL) ||
      function_info.parameter_count != 0) {
    return false;
  }

  const iree_hal_buffer_params_t params = device_buffer_params();
  BufferPtr lhs_buffer;
  BufferPtr rhs_buffer;
  BufferPtr out_buffer;
  const float lhs_values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float rhs_values[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  const float expected_values[4] = {10.0f, 40.0f, 90.0f, 160.0f};
  if (!expect_ok(iree_hal_allocator_allocate_buffer(fixture.allocator, params,
                                                    sizeof(lhs_values),
                                                    lhs_buffer.out()),
                 "allocate lhs") ||
      !expect_ok(iree_hal_allocator_allocate_buffer(fixture.allocator, params,
                                                    sizeof(rhs_values),
                                                    rhs_buffer.out()),
                 "allocate rhs") ||
      !expect_ok(iree_hal_allocator_allocate_buffer(fixture.allocator, params,
                                                    sizeof(expected_values),
                                                    out_buffer.out()),
                 "allocate out") ||
      !expect_ok(iree_hal_device_queue_update(
                     fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                     iree_hal_semaphore_list_empty(),
                     iree_hal_semaphore_list_empty(), lhs_values, 0,
                     lhs_buffer.get(), 0, sizeof(lhs_values),
                     IREE_HAL_UPDATE_FLAG_NONE),
                 "update lhs") ||
      !expect_ok(iree_hal_device_queue_update(
                     fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                     iree_hal_semaphore_list_empty(),
                     iree_hal_semaphore_list_empty(), rhs_values, 0,
                     rhs_buffer.get(), 0, sizeof(rhs_values),
                     IREE_HAL_UPDATE_FLAG_NONE),
                 "update rhs")) {
    return false;
  }

  iree_hal_buffer_ref_t dispatch_refs[3] = {
      iree_hal_make_indirect_buffer_ref(0, 0, sizeof(lhs_values)),
      iree_hal_make_indirect_buffer_ref(1, 0, sizeof(rhs_values)),
      iree_hal_make_indirect_buffer_ref(2, 0, sizeof(expected_values)),
  };
  const iree_hal_buffer_ref_list_t dispatch_bindings = {3, dispatch_refs};
  iree_hal_buffer_binding_t binding_table_entries[3] = {
      {lhs_buffer.get(), 0, sizeof(lhs_values)},
      {rhs_buffer.get(), 0, sizeof(rhs_values)},
      {out_buffer.get(), 0, sizeof(expected_values)},
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      3,
      binding_table_entries,
  };
  iree_hal_dispatch_config_t dispatch_config =
      iree_hal_make_static_dispatch_config(1, 1, 1);
  dispatch_config.workgroup_size[0] = 32;
  dispatch_config.workgroup_size[1] = 1;
  dispatch_config.workgroup_size[2] = 1;

  CommandBufferPtr command_buffer;
  if (!expect_ok(
          iree_hal_command_buffer_create(
              fixture.device.get(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
              IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
              /*binding_capacity=*/3, command_buffer.out()),
          "dispatch command buffer create") ||
      !expect_ok(iree_hal_command_buffer_begin(command_buffer.get()),
                 "dispatch command buffer begin") ||
      !expect_ok(iree_hal_command_buffer_dispatch(
                     command_buffer.get(), executable.get(), function,
                     dispatch_config, iree_const_byte_span_empty(),
                     dispatch_bindings, IREE_HAL_DISPATCH_FLAG_NONE),
                 "record dispatch") ||
      !expect_ok(iree_hal_command_buffer_end(command_buffer.get()),
                 "dispatch command buffer end") ||
      !expect_ok(iree_hal_device_queue_execute(
                     fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                     iree_hal_semaphore_list_empty(),
                     iree_hal_semaphore_list_empty(), command_buffer.get(),
                     binding_table, IREE_HAL_EXECUTE_FLAG_NONE),
                 "execute dispatch command buffer")) {
    return false;
  }

  void *out_device_ptr = nullptr;
  float actual_values[4] = {};
  lr_device_t lrrt_device = {0};
  return expect_ok(lrrt_iree_hal_buffer_device_pointer_for_test(
                       out_buffer.get(), &out_device_ptr),
                   "out device pointer") &&
         lr_device_open(0, &lrrt_device) == LR_SUCCESS &&
         lr_memcpy(lrrt_device, actual_values, out_device_ptr,
                   sizeof(actual_values),
                   LR_MEMCPY_DEVICE_TO_HOST) == LR_SUCCESS &&
         std::memcmp(actual_values, expected_values, sizeof(expected_values)) ==
             0;
#else
  return true;
#endif
}

bool test_executable_suite() {
  return test_executable_cache_contract() && test_hsaco_dispatch();
}

} // namespace

int main() {
  return run_test("lrrt IREE HAL executable tests", test_executable_suite);
}
