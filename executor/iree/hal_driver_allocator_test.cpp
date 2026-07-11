#include "hal_driver_test_common.hpp"

using namespace lrrt::iree::test;

namespace {

bool test_allocator_buffers() {
  HalFixture fixture;
  if (!fixture.setup()) {
    return false;
  }

  if (!expect_ok(iree_hal_allocator_trim(fixture.allocator),
                 "allocator trim")) {
    return false;
  }

  iree_host_size_t heap_count = 99;
  if (!expect_ok(iree_hal_allocator_query_memory_heaps(fixture.allocator, 0,
                                                       nullptr, &heap_count),
                 "query memory heaps") ||
      heap_count != 0) {
    return false;
  }

  const iree_hal_buffer_params_t device_params = device_buffer_params();
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          fixture.allocator, device_params, 128, nullptr, nullptr);
  if ((compatibility & IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE) == 0 ||
      (compatibility & IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER) == 0 ||
      (compatibility & IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH) == 0) {
    return false;
  }

  BufferPtr buffer;
  if (!expect_ok(iree_hal_allocator_allocate_buffer(
                     fixture.allocator, device_params, 128, buffer.out()),
                 "allocate device buffer") ||
      iree_hal_buffer_allocation_size(buffer.get()) != 128 ||
      iree_hal_buffer_byte_length(buffer.get()) != 128 ||
      iree_hal_buffer_memory_type(buffer.get()) !=
          IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL ||
      iree_hal_buffer_allowed_usage(buffer.get()) !=
          (IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
           IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    return false;
  }

  const iree_hal_buffer_params_t host_params = host_buffer_params();
  compatibility = iree_hal_allocator_query_buffer_compatibility(
      fixture.allocator, host_params, 16, nullptr, nullptr);
  if ((compatibility & IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE) == 0 ||
      (compatibility & IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER) == 0) {
    return false;
  }

  BufferPtr mapped_buffer;
  const uint32_t mapped_values[4] = {7, 11, 13, 17};
  uint32_t mapped_readback[4] = {};
  uint32_t mapped_device_readback[4] = {};
  void *mapped_device_ptr = nullptr;
  lr_device_t lrrt_device = {0};
  return expect_ok(iree_hal_allocator_allocate_buffer(
                       fixture.allocator, host_params, 16, mapped_buffer.out()),
                   "allocate mapped buffer") &&
         expect_ok(iree_hal_buffer_map_write(mapped_buffer.get(), 0,
                                             mapped_values,
                                             sizeof(mapped_values)),
                   "mapped buffer write") &&
         expect_ok(iree_hal_buffer_map_read(mapped_buffer.get(), 0,
                                            mapped_readback,
                                            sizeof(mapped_readback)),
                   "mapped buffer read") &&
         expect_ok(lrrt_iree_hal_buffer_device_pointer_for_test(
                       mapped_buffer.get(), &mapped_device_ptr),
                   "mapped device pointer") &&
         lr_device_open(0, &lrrt_device) == LR_SUCCESS &&
         lr_memcpy(lrrt_device, mapped_device_readback, mapped_device_ptr,
                   sizeof(mapped_device_readback),
                   LR_MEMCPY_DEVICE_TO_HOST) == LR_SUCCESS &&
         std::memcmp(mapped_readback, mapped_values, sizeof(mapped_values)) ==
             0 &&
         std::memcmp(mapped_device_readback, mapped_values,
                     sizeof(mapped_values)) == 0;
}

bool test_queue_alloca_dealloca() {
  HalFixture fixture;
  if (!fixture.setup()) {
    return false;
  }

  const iree_hal_buffer_params_t host_params = host_buffer_params();
  SemaphorePtr alloc_signal;
  SemaphorePtr dealloc_signal;
  uint64_t alloc_signal_value = 7;
  uint64_t dealloc_signal_value = 8;
  if (!expect_ok(iree_hal_semaphore_create(
                     fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                     IREE_HAL_SEMAPHORE_FLAG_NONE, alloc_signal.out()),
                 "alloc semaphore create") ||
      !expect_ok(iree_hal_semaphore_create(
                     fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                     IREE_HAL_SEMAPHORE_FLAG_NONE, dealloc_signal.out()),
                 "dealloc semaphore create")) {
    return false;
  }
  iree_hal_semaphore_t *alloc_signal_ptrs[1] = {alloc_signal.get()};
  iree_hal_semaphore_t *dealloc_signal_ptrs[1] = {dealloc_signal.get()};
  const iree_hal_semaphore_list_t alloc_signal_list = {
      1,
      alloc_signal_ptrs,
      &alloc_signal_value,
  };
  const iree_hal_semaphore_list_t dealloc_wait_list = {
      1,
      alloc_signal_ptrs,
      &alloc_signal_value,
  };
  const iree_hal_semaphore_list_t dealloc_signal_list = {
      1,
      dealloc_signal_ptrs,
      &dealloc_signal_value,
  };

  BufferPtr queue_buffer;
  uint64_t observed_dealloc_signal = 0;
  return expect_ok(iree_hal_device_queue_alloca(
                       fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                       iree_hal_semaphore_list_empty(), alloc_signal_list,
                       /*pool=*/nullptr, host_params, 16,
                       (iree_hal_alloca_flags_t)0, queue_buffer.out()),
                   "queue alloca") &&
         queue_buffer.get() != nullptr &&
         expect_ok(iree_hal_device_queue_dealloca(
                       fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                       dealloc_wait_list, dealloc_signal_list,
                       queue_buffer.get(), (iree_hal_dealloca_flags_t)0),
                   "queue dealloca") &&
         expect_ok(iree_hal_semaphore_query(dealloc_signal.get(),
                                            &observed_dealloc_signal),
                   "dealloc signal query") &&
         observed_dealloc_signal == dealloc_signal_value;
}

bool test_allocator_suite() {
  return test_allocator_buffers() && test_queue_alloca_dealloca();
}

} // namespace

int main() {
  return run_test("lrrt IREE HAL allocator tests", test_allocator_suite);
}
