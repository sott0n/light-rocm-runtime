#include "hal_driver_test_common.hpp"

#include "iree/io/file_handle.h"

using namespace lrrt::iree::test;

namespace {

bool test_queue_update_copy() {
  HalFixture fixture;
  if (!fixture.setup()) {
    return false;
  }

  const iree_hal_buffer_params_t params = device_buffer_params();
  BufferPtr source_buffer;
  BufferPtr target_buffer;
  lr_device_t lrrt_device = {0};
  const uint32_t source_values[4] = {3, 1, 4, 1};
  uint32_t copied_values[4] = {};
  void *target_device_ptr = nullptr;
  return expect_ok(iree_hal_allocator_allocate_buffer(fixture.allocator, params,
                                                      32, source_buffer.out()),
                   "allocate source buffer") &&
         expect_ok(iree_hal_allocator_allocate_buffer(fixture.allocator, params,
                                                      32, target_buffer.out()),
                   "allocate target buffer") &&
         expect_ok(iree_hal_device_queue_update(
                       fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                       iree_hal_semaphore_list_empty(),
                       iree_hal_semaphore_list_empty(), source_values, 0,
                       source_buffer.get(), 0, sizeof(source_values),
                       IREE_HAL_UPDATE_FLAG_NONE),
                   "queue update") &&
         expect_ok(iree_hal_device_queue_copy(
                       fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                       iree_hal_semaphore_list_empty(),
                       iree_hal_semaphore_list_empty(), source_buffer.get(), 0,
                       target_buffer.get(), 0, sizeof(source_values),
                       IREE_HAL_COPY_FLAG_NONE),
                   "queue copy") &&
         expect_ok(lrrt_iree_hal_buffer_device_pointer_for_test(
                       target_buffer.get(), &target_device_ptr),
                   "target device pointer") &&
         lr_device_open(0, &lrrt_device) == LR_SUCCESS &&
         lr_memcpy(lrrt_device, copied_values, target_device_ptr,
                   sizeof(copied_values),
                   LR_MEMCPY_DEVICE_TO_HOST) == LR_SUCCESS &&
         std::memcmp(copied_values, source_values, sizeof(source_values)) == 0;
}

bool test_queue_read_write_file() {
  HalFixture fixture;
  if (!fixture.setup()) {
    return false;
  }

  const iree_hal_buffer_params_t params = device_buffer_params();
  BufferPtr transfer_buffer;
  uint32_t file_source_values[4] = {8, 6, 7, 5};
  uint32_t file_readback_values[4] = {};
  uint32_t file_target_values[4] = {};
  void *transfer_device_ptr = nullptr;
  lr_device_t lrrt_device = {0};

  FileHandlePtr source_handle;
  FilePtr source_file;
  FileHandlePtr target_handle;
  FilePtr target_file;
  return expect_ok(iree_hal_allocator_allocate_buffer(
                       fixture.allocator, params, 16, transfer_buffer.out()),
                   "allocate file transfer buffer") &&
         expect_ok(iree_io_file_handle_wrap_host_allocation(
                       IREE_IO_FILE_ACCESS_READ,
                       iree_make_byte_span(
                           reinterpret_cast<uint8_t *>(file_source_values),
                           sizeof(file_source_values)),
                       iree_io_file_handle_release_callback_null(),
                       iree_allocator_system(), source_handle.out()),
                   "wrap source file handle") &&
         expect_ok(iree_hal_file_import(
                       fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                       IREE_HAL_MEMORY_ACCESS_READ, source_handle.get(),
                       IREE_HAL_EXTERNAL_FILE_FLAG_NONE, source_file.out()),
                   "import source file") &&
         expect_ok(iree_hal_device_queue_read(
                       fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                       iree_hal_semaphore_list_empty(),
                       iree_hal_semaphore_list_empty(), source_file.get(),
                       /*source_offset=*/0, transfer_buffer.get(),
                       /*target_offset=*/0, sizeof(file_source_values),
                       IREE_HAL_READ_FLAG_NONE),
                   "queue read") &&
         expect_ok(lrrt_iree_hal_buffer_device_pointer_for_test(
                       transfer_buffer.get(), &transfer_device_ptr),
                   "file transfer device pointer") &&
         lr_device_open(0, &lrrt_device) == LR_SUCCESS &&
         lr_memcpy(lrrt_device, file_readback_values, transfer_device_ptr,
                   sizeof(file_readback_values),
                   LR_MEMCPY_DEVICE_TO_HOST) == LR_SUCCESS &&
         std::memcmp(file_readback_values, file_source_values,
                     sizeof(file_source_values)) == 0 &&
         expect_ok(iree_io_file_handle_wrap_host_allocation(
                       IREE_IO_FILE_ACCESS_WRITE,
                       iree_make_byte_span(
                           reinterpret_cast<uint8_t *>(file_target_values),
                           sizeof(file_target_values)),
                       iree_io_file_handle_release_callback_null(),
                       iree_allocator_system(), target_handle.out()),
                   "wrap target file handle") &&
         expect_ok(iree_hal_file_import(
                       fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                       IREE_HAL_MEMORY_ACCESS_WRITE, target_handle.get(),
                       IREE_HAL_EXTERNAL_FILE_FLAG_NONE, target_file.out()),
                   "import target file") &&
         expect_ok(iree_hal_device_queue_write(
                       fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                       iree_hal_semaphore_list_empty(),
                       iree_hal_semaphore_list_empty(), transfer_buffer.get(),
                       /*source_offset=*/0, target_file.get(),
                       /*target_offset=*/0, sizeof(file_target_values),
                       IREE_HAL_WRITE_FLAG_NONE),
                   "queue write") &&
         std::memcmp(file_target_values, file_source_values,
                     sizeof(file_source_values)) == 0;
}

bool test_transfer_command_buffer() {
  HalFixture fixture;
  if (!fixture.setup()) {
    return false;
  }

  const iree_hal_buffer_params_t params = device_buffer_params();
  BufferPtr source_buffer;
  BufferPtr target_buffer;
  CommandBufferPtr command_buffer;
  const uint32_t source_values[2] = {0x11111111u, 0x22222222u};
  const uint32_t fill_pattern = 0xA5A5A5A5u;
  const uint32_t expected_values[4] = {
      source_values[0],
      source_values[1],
      fill_pattern,
      fill_pattern,
  };
  uint32_t actual_values[4] = {};
  void *target_device_ptr = nullptr;
  lr_device_t lrrt_device = {0};
  iree_hal_buffer_binding_t bindings[2] = {
      {nullptr, 0, 16},
      {nullptr, 0, 16},
  };
  const iree_hal_buffer_binding_table_t binding_table = {2, bindings};

  if (!expect_ok(iree_hal_allocator_allocate_buffer(fixture.allocator, params,
                                                    16, source_buffer.out()),
                 "allocate command source buffer") ||
      !expect_ok(iree_hal_allocator_allocate_buffer(fixture.allocator, params,
                                                    16, target_buffer.out()),
                 "allocate command target buffer")) {
    return false;
  }
  bindings[0].buffer = source_buffer.get();
  bindings[1].buffer = target_buffer.get();

  return expect_ok(iree_hal_command_buffer_create(
                       fixture.device.get(),
                       IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                       IREE_HAL_COMMAND_CATEGORY_TRANSFER,
                       IREE_HAL_QUEUE_AFFINITY_ANY, /*binding_capacity=*/2,
                       command_buffer.out()),
                   "command buffer create") &&
         expect_ok(iree_hal_command_buffer_begin(command_buffer.get()),
                   "command buffer begin") &&
         expect_ok(iree_hal_command_buffer_update_buffer(
                       command_buffer.get(), source_values,
                       /*source_offset=*/0,
                       iree_hal_make_buffer_ref(source_buffer.get(), 0,
                                                sizeof(source_values)),
                       IREE_HAL_UPDATE_FLAG_NONE),
                   "command update buffer") &&
         expect_ok(iree_hal_command_buffer_fill_buffer(
                       command_buffer.get(),
                       iree_hal_make_buffer_ref(target_buffer.get(), 0, 16),
                       &fill_pattern, sizeof(fill_pattern),
                       IREE_HAL_FILL_FLAG_NONE),
                   "command fill buffer") &&
         expect_ok(
             iree_hal_command_buffer_copy_buffer(
                 command_buffer.get(),
                 iree_hal_make_indirect_buffer_ref(0, 0, sizeof(source_values)),
                 iree_hal_make_indirect_buffer_ref(1, 0, sizeof(source_values)),
                 IREE_HAL_COPY_FLAG_NONE),
             "command copy buffer") &&
         expect_ok(iree_hal_command_buffer_end(command_buffer.get()),
                   "command buffer end") &&
         expect_ok(iree_hal_device_queue_execute(
                       fixture.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                       iree_hal_semaphore_list_empty(),
                       iree_hal_semaphore_list_empty(), command_buffer.get(),
                       binding_table, IREE_HAL_EXECUTE_FLAG_NONE),
                   "queue execute transfer command buffer") &&
         expect_ok(lrrt_iree_hal_buffer_device_pointer_for_test(
                       target_buffer.get(), &target_device_ptr),
                   "command target device pointer") &&
         lr_device_open(0, &lrrt_device) == LR_SUCCESS &&
         lr_memcpy(lrrt_device, actual_values, target_device_ptr,
                   sizeof(actual_values),
                   LR_MEMCPY_DEVICE_TO_HOST) == LR_SUCCESS &&
         std::memcmp(actual_values, expected_values, sizeof(expected_values)) ==
             0;
}

bool test_transfer_suite() {
  return test_queue_update_copy() && test_queue_read_write_file() &&
         test_transfer_command_buffer();
}

} // namespace

int main() {
  return run_test("lrrt IREE HAL transfer tests", test_transfer_suite);
}
