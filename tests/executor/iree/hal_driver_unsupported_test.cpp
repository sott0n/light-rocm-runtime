#include "hal_driver_test_common.hpp"

#include "iree/hal/channel.h"
#include "iree/hal/event.h"

using namespace lrrt::iree::test;

namespace {

iree_status_t no_op_host_call(void *user_data, const uint64_t args[4],
                              iree_hal_host_call_context_t *context) {
  (void)user_data;
  (void)args;
  (void)context;
  return iree_ok_status();
}

bool test_unsupported_contracts() {
  HalFixture fixture;
  if (!fixture.setup()) {
    return false;
  }

  const iree_hal_buffer_params_t host_params = host_buffer_params();
  if (iree_hal_allocator_supports_virtual_memory(fixture.allocator)) {
    return false;
  }

  uint32_t external_storage[4] = {};
  iree_hal_external_buffer_t external_buffer = {};
  external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION;
  external_buffer.size = sizeof(external_storage);
  external_buffer.handle.host_allocation.ptr = external_storage;
  BufferPtr imported_buffer;
  iree_status_t status = iree_hal_allocator_import_buffer(
      fixture.allocator, host_params, &external_buffer,
      iree_hal_buffer_release_callback_null(), imported_buffer.out());
  if (!expect_unimplemented(status, "external buffer import") ||
      imported_buffer.get() != nullptr) {
    return false;
  }

  BufferPtr export_source_buffer;
  iree_hal_external_buffer_t exported_buffer = {};
  if (!expect_ok(iree_hal_allocator_allocate_buffer(fixture.allocator,
                                                    host_params, 16,
                                                    export_source_buffer.out()),
                 "allocate export source buffer")) {
    return false;
  }
  status = iree_hal_allocator_export_buffer(
      fixture.allocator, export_source_buffer.get(),
      IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &exported_buffer);
  if (!expect_unimplemented(status, "external buffer export") ||
      exported_buffer.type != IREE_HAL_EXTERNAL_BUFFER_TYPE_NONE) {
    return false;
  }

  iree_device_size_t minimum_page_size = 1;
  iree_device_size_t recommended_page_size = 1;
  status = iree_hal_allocator_virtual_memory_query_granularity(
      fixture.allocator, host_params, &minimum_page_size,
      &recommended_page_size);
  if (!expect_unavailable(status, "virtual memory granularity") ||
      minimum_page_size != 0 || recommended_page_size != 0) {
    return false;
  }
  iree_hal_buffer_t *virtual_buffer = reinterpret_cast<iree_hal_buffer_t *>(1);
  status = iree_hal_allocator_virtual_memory_reserve(
      fixture.allocator, IREE_HAL_QUEUE_AFFINITY_ANY, 4096, &virtual_buffer);
  if (!expect_unavailable(status, "virtual memory reserve") ||
      virtual_buffer != nullptr) {
    return false;
  }

  iree_hal_channel_t *channel = reinterpret_cast<iree_hal_channel_t *>(1);
  status =
      iree_hal_channel_create(fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                              iree_hal_channel_params_t{}, &channel);
  if (!expect_unimplemented(status, "channel creation") || channel != nullptr) {
    if (channel) {
      iree_hal_channel_release(channel);
    }
    return false;
  }

  iree_hal_event_t *event = reinterpret_cast<iree_hal_event_t *>(1);
  status =
      iree_hal_event_create(fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                            IREE_HAL_EVENT_FLAG_NONE, &event);
  if (!expect_unimplemented(status, "event creation") || event != nullptr) {
    if (event) {
      iree_hal_event_release(event);
    }
    return false;
  }

  const iree_hal_host_call_t host_call =
      iree_hal_make_host_call(no_op_host_call, nullptr);
  status = iree_hal_device_queue_host_call(
      fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
      host_call, nullptr, IREE_HAL_HOST_CALL_FLAG_NONE);
  return expect_unimplemented(status, "queue host call");
}

} // namespace

int main() {
  return run_test("lrrt IREE HAL unsupported tests",
                  test_unsupported_contracts);
}
