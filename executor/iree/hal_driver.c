#include "hal_driver.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/hal/resource.h"
#include "lrrt/lrrt.h"

typedef struct lrrt_iree_hal_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_string_view_t identifier;
} lrrt_iree_hal_driver_t;

typedef struct lrrt_iree_hal_allocator_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  lr_device_t device;
  iree_hal_device_t *hal_device;
} lrrt_iree_hal_allocator_t;

typedef struct lrrt_iree_hal_buffer_t {
  iree_hal_buffer_t base;
  iree_allocator_t host_allocator;
  iree_hal_allocator_t *allocator;
  lr_device_t device;
  void *device_ptr;
} lrrt_iree_hal_buffer_t;

typedef struct lrrt_iree_hal_executable_cache_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_string_view_t identifier;
} lrrt_iree_hal_executable_cache_t;

typedef struct lrrt_iree_hal_device_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_allocator_t *device_allocator;
  iree_string_view_t identifier;
  iree_hal_device_topology_info_t topology_info;
} lrrt_iree_hal_device_t;

static const iree_hal_driver_vtable_t lrrt_iree_hal_driver_vtable;
static const iree_hal_allocator_vtable_t lrrt_iree_hal_allocator_vtable;
static const iree_hal_buffer_vtable_t lrrt_iree_hal_buffer_vtable;
static const iree_hal_executable_cache_vtable_t
    lrrt_iree_hal_executable_cache_vtable;
static const iree_hal_device_vtable_t lrrt_iree_hal_device_vtable;

static const iree_string_view_t kLrrtIreeHalExecutableFormatRocmHsaco =
    IREE_SVL("rocm-hsaco");
static const iree_string_view_t kLrrtIreeHalExecutableFormatAmdgpuHsaco =
    IREE_SVL("amdgpu-hsaco");

static uint32_t g_lrrt_iree_hal_runtime_ref_count = 0;
static bool g_lrrt_iree_hal_owns_runtime = false;

static lrrt_iree_hal_driver_t *
lrrt_iree_hal_driver_cast(iree_hal_driver_t *base_driver) {
  IREE_HAL_ASSERT_TYPE(base_driver, &lrrt_iree_hal_driver_vtable);
  return (lrrt_iree_hal_driver_t *)base_driver;
}

static lrrt_iree_hal_allocator_t *
lrrt_iree_hal_allocator_cast(iree_hal_allocator_t *base_allocator) {
  IREE_HAL_ASSERT_TYPE(base_allocator, &lrrt_iree_hal_allocator_vtable);
  return (lrrt_iree_hal_allocator_t *)base_allocator;
}

static const lrrt_iree_hal_allocator_t *
lrrt_iree_hal_allocator_const_cast(const iree_hal_allocator_t *base_allocator) {
  IREE_HAL_ASSERT_TYPE(base_allocator, &lrrt_iree_hal_allocator_vtable);
  return (const lrrt_iree_hal_allocator_t *)base_allocator;
}

static lrrt_iree_hal_device_t *
lrrt_iree_hal_device_cast(iree_hal_device_t *base_device) {
  IREE_HAL_ASSERT_TYPE(base_device, &lrrt_iree_hal_device_vtable);
  return (lrrt_iree_hal_device_t *)base_device;
}

static iree_status_t lrrt_iree_hal_status_from_lr(lr_status_t status,
                                                  const char *operation) {
  switch (status) {
  case LR_SUCCESS:
    return iree_ok_status();
  case LR_ERROR_INVALID_ARGUMENT:
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s failed: %s",
                            operation, lr_status_string(status));
  case LR_ERROR_NOT_INITIALIZED:
  case LR_ERROR_ALREADY_INITIALIZED:
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "%s failed: %s",
                            operation, lr_status_string(status));
  case LR_ERROR_NOT_SUPPORTED:
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "%s failed: %s",
                            operation, lr_status_string(status));
  case LR_ERROR_RUNTIME:
  default:
    return iree_make_status(IREE_STATUS_INTERNAL, "%s failed: %s", operation,
                            lr_status_string(status));
  }
}

static iree_status_t lrrt_iree_hal_copy_string_view(iree_string_view_t value,
                                                    iree_host_size_t capacity,
                                                    char *buffer) {
  if (value.size >= capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "string buffer too small: need %" PRIhsz
                            " bytes, have %" PRIhsz,
                            value.size + 1, capacity);
  }
  memcpy(buffer, value.data, value.size);
  buffer[value.size] = '\0';
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_runtime_retain(void) {
  if (g_lrrt_iree_hal_runtime_ref_count == 0) {
    lr_status_t lr_status = lr_init();
    if (lr_status == LR_SUCCESS) {
      g_lrrt_iree_hal_owns_runtime = true;
    } else if (lr_status == LR_ERROR_ALREADY_INITIALIZED) {
      g_lrrt_iree_hal_owns_runtime = false;
    } else {
      return lrrt_iree_hal_status_from_lr(lr_status, "lr_init");
    }
  }
  ++g_lrrt_iree_hal_runtime_ref_count;
  return iree_ok_status();
}

static void lrrt_iree_hal_runtime_release(void) {
  if (g_lrrt_iree_hal_runtime_ref_count == 0) {
    return;
  }
  --g_lrrt_iree_hal_runtime_ref_count;
  if (g_lrrt_iree_hal_runtime_ref_count == 0 && g_lrrt_iree_hal_owns_runtime) {
    lr_shutdown();
    g_lrrt_iree_hal_owns_runtime = false;
  }
}

static iree_status_t
lrrt_iree_hal_allocator_create(iree_allocator_t host_allocator,
                               iree_hal_device_t *hal_device,
                               iree_hal_allocator_t **out_allocator) {
  IREE_ASSERT_ARGUMENT(out_allocator);
  *out_allocator = NULL;

  IREE_RETURN_IF_ERROR(lrrt_iree_hal_runtime_retain());

  uint32_t device_count = 0;
  lr_status_t lr_status = lr_device_count(&device_count);
  if (lr_status != LR_SUCCESS) {
    lrrt_iree_hal_runtime_release();
    return lrrt_iree_hal_status_from_lr(lr_status, "lr_device_count");
  }
  if (device_count == 0) {
    lrrt_iree_hal_runtime_release();
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "lrrt HAL allocator found no AMD GPU devices");
  }

  lr_device_t lrrt_device = {0};
  lr_status = lr_device_open(0, &lrrt_device);
  if (lr_status != LR_SUCCESS) {
    lrrt_iree_hal_runtime_release();
    return lrrt_iree_hal_status_from_lr(lr_status, "lr_device_open");
  }

  lrrt_iree_hal_allocator_t *allocator = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*allocator), (void **)&allocator);
  if (!iree_status_is_ok(status)) {
    lrrt_iree_hal_runtime_release();
    return status;
  }
  iree_hal_resource_initialize(&lrrt_iree_hal_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;
  allocator->device = lrrt_device;
  allocator->hal_device = hal_device;
  *out_allocator = (iree_hal_allocator_t *)allocator;
  return iree_ok_status();
}

static iree_status_t
lrrt_iree_hal_device_create(iree_string_view_t identifier,
                            iree_allocator_t host_allocator,
                            iree_hal_device_t **out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;

  lrrt_iree_hal_device_t *device = NULL;
  const iree_host_size_t total_size =
      iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void **)&device));

  iree_hal_resource_initialize(&lrrt_iree_hal_device_vtable, &device->resource);
  device->host_allocator = host_allocator;
  device->device_allocator = NULL;
  memset(&device->topology_info, 0, sizeof(device->topology_info));
  iree_string_view_append_to_buffer(identifier, &device->identifier,
                                    (char *)device +
                                        iree_sizeof_struct(*device));

  iree_status_t status = lrrt_iree_hal_allocator_create(
      host_allocator, (iree_hal_device_t *)device, &device->device_allocator);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, device);
    return status;
  }

  *out_device = (iree_hal_device_t *)device;
  return iree_ok_status();
}

static iree_status_t
lrrt_iree_hal_driver_create(iree_string_view_t identifier,
                            iree_allocator_t host_allocator,
                            iree_hal_driver_t **out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = NULL;

  lrrt_iree_hal_driver_t *driver = NULL;
  const iree_host_size_t total_size =
      iree_sizeof_struct(*driver) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void **)&driver));

  iree_hal_resource_initialize(&lrrt_iree_hal_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(identifier, &driver->identifier,
                                    (char *)driver +
                                        iree_sizeof_struct(*driver));

  *out_driver = (iree_hal_driver_t *)driver;
  return iree_ok_status();
}

static void lrrt_iree_hal_driver_destroy(iree_hal_driver_t *base_driver) {
  lrrt_iree_hal_driver_t *driver = lrrt_iree_hal_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  iree_allocator_free(host_allocator, driver);
}

static void
lrrt_iree_hal_allocator_destroy(iree_hal_allocator_t *base_allocator) {
  lrrt_iree_hal_allocator_t *allocator =
      lrrt_iree_hal_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  lrrt_iree_hal_runtime_release();
  iree_allocator_free(host_allocator, allocator);
}

static void lrrt_iree_hal_device_destroy(iree_hal_device_t *base_device) {
  lrrt_iree_hal_device_t *device = lrrt_iree_hal_device_cast(base_device);
  iree_allocator_t host_allocator = device->host_allocator;
  iree_hal_allocator_release(device->device_allocator);
  iree_allocator_free(host_allocator, device);
}

static iree_allocator_t lrrt_iree_hal_allocator_host_allocator(
    const iree_hal_allocator_t *base_allocator) {
  const lrrt_iree_hal_allocator_t *allocator =
      lrrt_iree_hal_allocator_const_cast(base_allocator);
  return allocator->host_allocator;
}

static iree_status_t
lrrt_iree_hal_allocator_trim(iree_hal_allocator_t *base_allocator) {
  (void)base_allocator;
  return iree_ok_status();
}

static void lrrt_iree_hal_allocator_query_statistics(
    iree_hal_allocator_t *base_allocator,
    iree_hal_allocator_statistics_t *out_statistics) {
  (void)base_allocator;
  IREE_ASSERT_ARGUMENT(out_statistics);
  memset(out_statistics, 0, sizeof(*out_statistics));
}

static iree_status_t lrrt_iree_hal_allocator_query_memory_heaps(
    iree_hal_allocator_t *base_allocator, iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t *heaps, iree_host_size_t *out_count) {
  (void)base_allocator;
  (void)capacity;
  (void)heaps;
  if (out_count) {
    *out_count = 0;
  }
  return iree_ok_status();
}

static iree_hal_buffer_compatibility_t
lrrt_iree_hal_allocator_query_buffer_compatibility(
    iree_hal_allocator_t *base_allocator, iree_hal_buffer_params_t *params,
    iree_device_size_t *allocation_size) {
  (void)base_allocator;
  if (!params || !allocation_size || *allocation_size == 0) {
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }
  if ((params->type & IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) != 0) {
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }
  if ((params->type & IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE) == 0 &&
      (params->type & IREE_HAL_MEMORY_TYPE_OPTIMAL) == 0) {
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }
  if ((params->usage & IREE_HAL_BUFFER_USAGE_MAPPING) != 0) {
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }

  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;
  if ((params->usage & IREE_HAL_BUFFER_USAGE_TRANSFER) != 0) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
  }
  if ((params->usage & IREE_HAL_BUFFER_USAGE_DISPATCH) != 0) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
  }
  return compatibility;
}

#define LRRT_IREE_HAL_ALLOCATOR_UNIMPLEMENTED(name)                            \
  iree_make_status(IREE_STATUS_UNIMPLEMENTED,                                  \
                   "lrrt HAL allocator " name " is not implemented yet")

static iree_status_t
lrrt_iree_hal_allocator_allocate_buffer(iree_hal_allocator_t *base_allocator,
                                        const iree_hal_buffer_params_t *params,
                                        iree_device_size_t allocation_size,
                                        iree_hal_buffer_t **out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;

  if (!params || allocation_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "lrrt HAL allocation requires non-zero size");
  }

  iree_device_size_t checked_allocation_size = allocation_size;
  iree_hal_buffer_params_t checked_params = *params;
  if (lrrt_iree_hal_allocator_query_buffer_compatibility(
          base_allocator, &checked_params, &checked_allocation_size) ==
      IREE_HAL_BUFFER_COMPATIBILITY_NONE) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "lrrt HAL allocator only supports device-visible, "
                            "non-mappable buffers");
  }

  lrrt_iree_hal_allocator_t *allocator =
      lrrt_iree_hal_allocator_cast(base_allocator);
  void *device_ptr = NULL;
  lr_status_t lr_status =
      lr_malloc(allocator->device, (size_t)allocation_size, &device_ptr);
  if (lr_status != LR_SUCCESS) {
    return lrrt_iree_hal_status_from_lr(lr_status, "lr_malloc");
  }

  lrrt_iree_hal_buffer_t *buffer = NULL;
  iree_status_t status = iree_allocator_malloc(
      allocator->host_allocator, sizeof(*buffer), (void **)&buffer);
  if (!iree_status_is_ok(status)) {
    lr_free(allocator->device, device_ptr);
    return status;
  }

  iree_hal_buffer_placement_t placement = {
      .device = allocator->hal_device,
      .queue_affinity = params->queue_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
      .reserved = 0,
  };
  iree_hal_buffer_initialize(placement, &buffer->base, allocation_size, 0,
                             allocation_size, params->type, params->access,
                             params->usage, &lrrt_iree_hal_buffer_vtable,
                             &buffer->base);
  buffer->host_allocator = allocator->host_allocator;
  buffer->allocator = base_allocator;
  buffer->device = allocator->device;
  buffer->device_ptr = device_ptr;
  iree_hal_allocator_retain(base_allocator);

  *out_buffer = &buffer->base;
  return iree_ok_status();
}

static void
lrrt_iree_hal_allocator_deallocate_buffer(iree_hal_allocator_t *base_allocator,
                                          iree_hal_buffer_t *buffer) {
  (void)base_allocator;
  iree_hal_buffer_release(buffer);
}

static lrrt_iree_hal_buffer_t *
lrrt_iree_hal_buffer_cast(iree_hal_buffer_t *base_buffer) {
  IREE_HAL_ASSERT_TYPE(base_buffer, &lrrt_iree_hal_buffer_vtable);
  return (lrrt_iree_hal_buffer_t *)base_buffer;
}

static iree_status_t lrrt_iree_hal_buffer_device_range(
    iree_hal_buffer_t *base_buffer, iree_device_size_t offset,
    iree_device_size_t length, void **out_device_ptr) {
  IREE_ASSERT_ARGUMENT(out_device_ptr);
  *out_device_ptr = NULL;
  if (!iree_hal_resource_is(base_buffer, &lrrt_iree_hal_buffer_vtable)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer is not an lrrt HAL buffer");
  }
  iree_device_size_t byte_length = iree_hal_buffer_byte_length(base_buffer);
  if (offset > byte_length || length > byte_length - offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer range out of bounds: offset=%" PRIu64
                            ", length=%" PRIu64 ", byte_length=%" PRIu64,
                            (uint64_t)offset, (uint64_t)length,
                            (uint64_t)byte_length);
  }
  lrrt_iree_hal_buffer_t *buffer = lrrt_iree_hal_buffer_cast(base_buffer);
  *out_device_ptr = (uint8_t *)buffer->device_ptr + offset;
  return iree_ok_status();
}

#if defined(LRRT_IREE_HAL_DRIVER_TESTING)
iree_status_t
lrrt_iree_hal_buffer_device_pointer_for_test(iree_hal_buffer_t *buffer,
                                             void **out_device_ptr) {
  return lrrt_iree_hal_buffer_device_range(buffer, 0, 0, out_device_ptr);
}
#endif

static void lrrt_iree_hal_buffer_destroy(iree_hal_buffer_t *base_buffer) {
  lrrt_iree_hal_buffer_t *buffer = lrrt_iree_hal_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  if (buffer->device_ptr) {
    lr_free(buffer->device, buffer->device_ptr);
    buffer->device_ptr = NULL;
  }
  iree_hal_allocator_release(buffer->allocator);
  iree_allocator_free(host_allocator, buffer);
}

static iree_status_t lrrt_iree_hal_buffer_map_range(
    iree_hal_buffer_t *base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t *out_mapping) {
  (void)base_buffer;
  (void)mapping_mode;
  (void)memory_access;
  (void)local_byte_offset;
  (void)local_byte_length;
  IREE_ASSERT_ARGUMENT(out_mapping);
  memset(out_mapping, 0, sizeof(*out_mapping));
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "lrrt HAL buffers are device-local and not host "
                          "mappable yet");
}

static iree_status_t lrrt_iree_hal_buffer_unmap_range(
    iree_hal_buffer_t *base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t *mapping) {
  (void)base_buffer;
  (void)local_byte_offset;
  (void)local_byte_length;
  (void)mapping;
  return iree_ok_status();
}

static iree_status_t
lrrt_iree_hal_buffer_invalidate_range(iree_hal_buffer_t *base_buffer,
                                      iree_device_size_t local_byte_offset,
                                      iree_device_size_t local_byte_length) {
  (void)base_buffer;
  (void)local_byte_offset;
  (void)local_byte_length;
  return iree_ok_status();
}

static iree_status_t
lrrt_iree_hal_buffer_flush_range(iree_hal_buffer_t *base_buffer,
                                 iree_device_size_t local_byte_offset,
                                 iree_device_size_t local_byte_length) {
  (void)base_buffer;
  (void)local_byte_offset;
  (void)local_byte_length;
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_allocator_import_buffer(
    iree_hal_allocator_t *base_allocator,
    const iree_hal_buffer_params_t *params,
    iree_hal_external_buffer_t *external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t **out_buffer) {
  (void)base_allocator;
  (void)params;
  (void)external_buffer;
  (void)release_callback;
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  return LRRT_IREE_HAL_ALLOCATOR_UNIMPLEMENTED("buffer import");
}

static iree_status_t lrrt_iree_hal_allocator_export_buffer(
    iree_hal_allocator_t *base_allocator, iree_hal_buffer_t *buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t *out_external_buffer) {
  (void)base_allocator;
  (void)buffer;
  (void)requested_type;
  (void)requested_flags;
  IREE_ASSERT_ARGUMENT(out_external_buffer);
  memset(out_external_buffer, 0, sizeof(*out_external_buffer));
  return LRRT_IREE_HAL_ALLOCATOR_UNIMPLEMENTED("buffer export");
}

static bool lrrt_iree_hal_allocator_supports_virtual_memory(
    iree_hal_allocator_t *base_allocator) {
  (void)base_allocator;
  return false;
}

static iree_status_t lrrt_iree_hal_allocator_virtual_memory_query_granularity(
    iree_hal_allocator_t *base_allocator, iree_hal_buffer_params_t params,
    iree_device_size_t *out_minimum_page_size,
    iree_device_size_t *out_recommended_page_size) {
  (void)base_allocator;
  (void)params;
  IREE_ASSERT_ARGUMENT(out_minimum_page_size);
  IREE_ASSERT_ARGUMENT(out_recommended_page_size);
  *out_minimum_page_size = 0;
  *out_recommended_page_size = 0;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "lrrt HAL allocator virtual memory is unavailable");
}

static iree_status_t lrrt_iree_hal_allocator_virtual_memory_reserve(
    iree_hal_allocator_t *base_allocator,
    iree_hal_queue_affinity_t queue_affinity, iree_device_size_t size,
    iree_hal_buffer_t **out_virtual_buffer) {
  (void)base_allocator;
  (void)queue_affinity;
  (void)size;
  IREE_ASSERT_ARGUMENT(out_virtual_buffer);
  *out_virtual_buffer = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "lrrt HAL allocator virtual memory is unavailable");
}

static iree_status_t lrrt_iree_hal_allocator_virtual_memory_release(
    iree_hal_allocator_t *base_allocator, iree_hal_buffer_t *virtual_buffer) {
  (void)base_allocator;
  (void)virtual_buffer;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "lrrt HAL allocator virtual memory is unavailable");
}

static iree_status_t lrrt_iree_hal_allocator_physical_memory_allocate(
    iree_hal_allocator_t *base_allocator, iree_hal_buffer_params_t params,
    iree_device_size_t size, iree_allocator_t host_allocator,
    iree_hal_physical_memory_t **out_physical_memory) {
  (void)base_allocator;
  (void)params;
  (void)size;
  (void)host_allocator;
  IREE_ASSERT_ARGUMENT(out_physical_memory);
  *out_physical_memory = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "lrrt HAL allocator virtual memory is unavailable");
}

static iree_status_t lrrt_iree_hal_allocator_physical_memory_free(
    iree_hal_allocator_t *base_allocator,
    iree_hal_physical_memory_t *physical_memory) {
  (void)base_allocator;
  (void)physical_memory;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "lrrt HAL allocator virtual memory is unavailable");
}

static iree_status_t lrrt_iree_hal_allocator_virtual_memory_map(
    iree_hal_allocator_t *base_allocator, iree_hal_buffer_t *virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t *physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size) {
  (void)base_allocator;
  (void)virtual_buffer;
  (void)virtual_offset;
  (void)physical_memory;
  (void)physical_offset;
  (void)size;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "lrrt HAL allocator virtual memory is unavailable");
}

static iree_status_t lrrt_iree_hal_allocator_virtual_memory_unmap(
    iree_hal_allocator_t *base_allocator, iree_hal_buffer_t *virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  (void)base_allocator;
  (void)virtual_buffer;
  (void)virtual_offset;
  (void)size;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "lrrt HAL allocator virtual memory is unavailable");
}

static iree_status_t lrrt_iree_hal_allocator_virtual_memory_protect(
    iree_hal_allocator_t *base_allocator, iree_hal_buffer_t *virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_protection_t protection) {
  (void)base_allocator;
  (void)virtual_buffer;
  (void)virtual_offset;
  (void)size;
  (void)queue_affinity;
  (void)protection;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "lrrt HAL allocator virtual memory is unavailable");
}

static iree_status_t lrrt_iree_hal_allocator_virtual_memory_advise(
    iree_hal_allocator_t *base_allocator, iree_hal_buffer_t *virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_advice_t advice) {
  (void)base_allocator;
  (void)virtual_buffer;
  (void)virtual_offset;
  (void)size;
  (void)queue_affinity;
  (void)advice;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "lrrt HAL allocator virtual memory is unavailable");
}

static lrrt_iree_hal_executable_cache_t *lrrt_iree_hal_executable_cache_cast(
    iree_hal_executable_cache_t *base_executable_cache) {
  IREE_HAL_ASSERT_TYPE(base_executable_cache,
                       &lrrt_iree_hal_executable_cache_vtable);
  return (lrrt_iree_hal_executable_cache_t *)base_executable_cache;
}

static iree_status_t lrrt_iree_hal_executable_cache_create(
    iree_string_view_t identifier, iree_allocator_t host_allocator,
    iree_hal_executable_cache_t **out_executable_cache) {
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = NULL;

  lrrt_iree_hal_executable_cache_t *executable_cache = NULL;
  const iree_host_size_t total_size =
      iree_sizeof_struct(*executable_cache) + identifier.size;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_size,
                                             (void **)&executable_cache));
  iree_hal_resource_initialize(&lrrt_iree_hal_executable_cache_vtable,
                               &executable_cache->resource);
  executable_cache->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(identifier, &executable_cache->identifier,
                                    (char *)executable_cache +
                                        iree_sizeof_struct(*executable_cache));

  *out_executable_cache = (iree_hal_executable_cache_t *)executable_cache;
  return iree_ok_status();
}

static void lrrt_iree_hal_executable_cache_destroy(
    iree_hal_executable_cache_t *base_executable_cache) {
  lrrt_iree_hal_executable_cache_t *executable_cache =
      lrrt_iree_hal_executable_cache_cast(base_executable_cache);
  iree_allocator_t host_allocator = executable_cache->host_allocator;
  iree_allocator_free(host_allocator, executable_cache);
}

static iree_status_t lrrt_iree_hal_executable_cache_infer_format(
    iree_hal_executable_cache_t *base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char *executable_format,
    iree_host_size_t *out_inferred_size) {
  (void)base_executable_cache;
  (void)caching_mode;
  if (iree_const_byte_span_is_empty(executable_data)) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "lrrt HAL executable format inference requires "
                            "non-empty HSACO data");
  }
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_copy_string_view(
      kLrrtIreeHalExecutableFormatRocmHsaco, executable_format_capacity,
      executable_format));
  *out_inferred_size = executable_data.data_length;
  return iree_ok_status();
}

static bool lrrt_iree_hal_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t *base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  (void)base_executable_cache;
  (void)caching_mode;
  return iree_string_view_equal(executable_format,
                                kLrrtIreeHalExecutableFormatRocmHsaco) ||
         iree_string_view_equal(executable_format,
                                kLrrtIreeHalExecutableFormatAmdgpuHsaco);
}

static iree_status_t lrrt_iree_hal_executable_cache_prepare_executable(
    iree_hal_executable_cache_t *base_executable_cache,
    const iree_hal_executable_params_t *executable_params,
    iree_hal_executable_t **out_executable) {
  (void)base_executable_cache;
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  if (!executable_params) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "lrrt HAL executable preparation requires executable params");
  }
  if (!lrrt_iree_hal_executable_cache_can_prepare_format(
          base_executable_cache, executable_params->caching_mode,
          executable_params->executable_format)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "lrrt HAL executable cache cannot prepare format "
                            "'%.*s'",
                            (int)executable_params->executable_format.size,
                            executable_params->executable_format.data);
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "lrrt HAL executable preparation is not implemented yet");
}

static iree_string_view_t
lrrt_iree_hal_device_id(iree_hal_device_t *base_device) {
  lrrt_iree_hal_device_t *device = lrrt_iree_hal_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t
lrrt_iree_hal_device_host_allocator(iree_hal_device_t *base_device) {
  lrrt_iree_hal_device_t *device = lrrt_iree_hal_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t *
lrrt_iree_hal_device_allocator(iree_hal_device_t *base_device) {
  lrrt_iree_hal_device_t *device = lrrt_iree_hal_device_cast(base_device);
  return device->device_allocator;
}

static void
lrrt_iree_hal_device_replace_allocator(iree_hal_device_t *base_device,
                                       iree_hal_allocator_t *new_allocator) {
  lrrt_iree_hal_device_t *device = lrrt_iree_hal_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void lrrt_iree_hal_device_replace_channel_provider(
    iree_hal_device_t *base_device, iree_hal_channel_provider_t *new_provider) {
  (void)base_device;
  (void)new_provider;
}

static iree_status_t lrrt_iree_hal_device_trim(iree_hal_device_t *base_device) {
  (void)base_device;
  return iree_ok_status();
}

static iree_status_t
lrrt_iree_hal_device_query_i64(iree_hal_device_t *base_device,
                               iree_string_view_t category,
                               iree_string_view_t key, int64_t *out_value) {
  (void)base_device;
  (void)category;
  (void)key;
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "lrrt HAL device query is not implemented yet");
}

static iree_status_t lrrt_iree_hal_device_query_capabilities(
    iree_hal_device_t *base_device,
    iree_hal_device_capabilities_t *out_capabilities) {
  (void)base_device;
  IREE_ASSERT_ARGUMENT(out_capabilities);
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t *
lrrt_iree_hal_device_topology_info(iree_hal_device_t *base_device) {
  lrrt_iree_hal_device_t *device = lrrt_iree_hal_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t
lrrt_iree_hal_device_refine_topology_edge(iree_hal_device_t *src_device,
                                          iree_hal_device_t *dst_device,
                                          iree_hal_topology_edge_t *edge) {
  (void)src_device;
  (void)dst_device;
  (void)edge;
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_device_assign_topology_info(
    iree_hal_device_t *base_device,
    const iree_hal_device_topology_info_t *topology_info) {
  lrrt_iree_hal_device_t *device = lrrt_iree_hal_device_cast(base_device);
  if (topology_info) {
    memcpy(&device->topology_info, topology_info,
           sizeof(device->topology_info));
  } else {
    memset(&device->topology_info, 0, sizeof(device->topology_info));
  }
  return iree_ok_status();
}

#define LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED(name)                               \
  iree_make_status(IREE_STATUS_UNIMPLEMENTED,                                  \
                   "lrrt HAL device " name " is not implemented yet")

static iree_status_t lrrt_iree_hal_device_require_empty_semaphore_lists(
    const char *operation, const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  if (!iree_hal_semaphore_list_is_empty(wait_semaphore_list) ||
      !iree_hal_semaphore_list_is_empty(signal_semaphore_list)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "lrrt HAL device %s only supports empty semaphore lists", operation);
  }
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_device_create_channel(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t **out_channel) {
  (void)device;
  (void)queue_affinity;
  (void)params;
  IREE_ASSERT_ARGUMENT(out_channel);
  *out_channel = NULL;
  return LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED("channel creation");
}

static iree_status_t lrrt_iree_hal_device_create_command_buffer(
    iree_hal_device_t *device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t **out_command_buffer) {
  (void)device;
  (void)mode;
  (void)command_categories;
  (void)queue_affinity;
  (void)binding_capacity;
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;
  return LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED("command buffer creation");
}

static iree_status_t lrrt_iree_hal_device_create_event(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t **out_event) {
  (void)device;
  (void)queue_affinity;
  (void)flags;
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;
  return LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED("event creation");
}

static iree_status_t lrrt_iree_hal_device_create_executable_cache(
    iree_hal_device_t *device, iree_string_view_t identifier,
    iree_hal_executable_cache_t **out_executable_cache) {
  lrrt_iree_hal_device_t *lrrt_device = lrrt_iree_hal_device_cast(device);
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  return lrrt_iree_hal_executable_cache_create(
      identifier, lrrt_device->host_allocator, out_executable_cache);
}

static iree_status_t lrrt_iree_hal_device_import_file(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t *handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t **out_file) {
  (void)device;
  (void)queue_affinity;
  (void)access;
  (void)handle;
  (void)flags;
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;
  return LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED("file import");
}

static iree_status_t lrrt_iree_hal_device_create_semaphore(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t **out_semaphore) {
  (void)device;
  (void)queue_affinity;
  (void)initial_value;
  (void)flags;
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  return LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED("semaphore creation");
}

static iree_hal_semaphore_compatibility_t
lrrt_iree_hal_device_query_semaphore_compatibility(
    iree_hal_device_t *device, iree_hal_semaphore_t *semaphore) {
  (void)device;
  (void)semaphore;
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_NONE;
}

static iree_status_t lrrt_iree_hal_device_query_queue_pool_backend(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t *out_backend) {
  (void)device;
  (void)queue_affinity;
  IREE_ASSERT_ARGUMENT(out_backend);
  memset(out_backend, 0, sizeof(*out_backend));
  return LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED("queue pool backend");
}

static iree_status_t lrrt_iree_hal_device_queue_alloca(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t *pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t **out_buffer) {
  (void)device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)pool;
  (void)params;
  (void)allocation_size;
  (void)flags;
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  return LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED("queue allocation");
}

static iree_status_t lrrt_iree_hal_device_queue_dealloca(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t *buffer, iree_hal_dealloca_flags_t flags) {
  (void)device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)buffer;
  (void)flags;
  return LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED("queue deallocation");
}

static iree_status_t lrrt_iree_hal_device_queue_fill(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void *pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  (void)device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)target_buffer;
  (void)target_offset;
  (void)length;
  (void)pattern;
  (void)pattern_length;
  (void)flags;
  return LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED("queue fill");
}

static iree_status_t lrrt_iree_hal_device_queue_update(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void *source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  (void)queue_affinity;
  (void)flags;
  lrrt_iree_hal_device_t *lrrt_device = lrrt_iree_hal_device_cast(device);
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_device_require_empty_semaphore_lists(
      "queue update", wait_semaphore_list, signal_semaphore_list));
  if (length == 0) {
    return iree_ok_status();
  }
  if ((iree_hal_buffer_allowed_usage(target_buffer) &
       IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET) == 0) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "queue update target buffer lacks transfer-target "
                            "usage");
  }

  void *target_device_ptr = NULL;
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_buffer_device_range(
      target_buffer, target_offset, length, &target_device_ptr));
  const uint8_t *source =
      (const uint8_t *)source_buffer + (iree_host_size_t)source_offset;
  lrrt_iree_hal_allocator_t *allocator =
      lrrt_iree_hal_allocator_cast(lrrt_device->device_allocator);
  lr_status_t lr_status =
      lr_memcpy(allocator->device, target_device_ptr, source, (size_t)length,
                LR_MEMCPY_HOST_TO_DEVICE);
  return lrrt_iree_hal_status_from_lr(lr_status, "lr_memcpy");
}

static iree_status_t lrrt_iree_hal_device_queue_copy(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t *source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  (void)queue_affinity;
  (void)flags;
  lrrt_iree_hal_device_t *lrrt_device = lrrt_iree_hal_device_cast(device);
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_device_require_empty_semaphore_lists(
      "queue copy", wait_semaphore_list, signal_semaphore_list));
  if (length == 0) {
    return iree_ok_status();
  }
  if ((iree_hal_buffer_allowed_usage(source_buffer) &
       IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE) == 0) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "queue copy source buffer lacks transfer-source "
                            "usage");
  }
  if ((iree_hal_buffer_allowed_usage(target_buffer) &
       IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET) == 0) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "queue copy target buffer lacks transfer-target "
                            "usage");
  }

  void *source_device_ptr = NULL;
  void *target_device_ptr = NULL;
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_buffer_device_range(
      source_buffer, source_offset, length, &source_device_ptr));
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_buffer_device_range(
      target_buffer, target_offset, length, &target_device_ptr));
  lrrt_iree_hal_allocator_t *allocator =
      lrrt_iree_hal_allocator_cast(lrrt_device->device_allocator);
  lr_status_t lr_status =
      lr_memcpy(allocator->device, target_device_ptr, source_device_ptr,
                (size_t)length, LR_MEMCPY_DEVICE_TO_DEVICE);
  return lrrt_iree_hal_status_from_lr(lr_status, "lr_memcpy");
}

static iree_status_t lrrt_iree_hal_device_queue_read(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t *source_file, uint64_t source_offset,
    iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  (void)device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)source_file;
  (void)source_offset;
  (void)target_buffer;
  (void)target_offset;
  (void)length;
  (void)flags;
  return LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED("queue read");
}

static iree_status_t lrrt_iree_hal_device_queue_write(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t *source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t *target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  (void)device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)source_buffer;
  (void)source_offset;
  (void)target_file;
  (void)target_offset;
  (void)length;
  (void)flags;
  return LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED("queue write");
}

static iree_status_t lrrt_iree_hal_device_queue_host_call(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  (void)device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)call;
  (void)args;
  (void)flags;
  return LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED("queue host call");
}

static iree_status_t lrrt_iree_hal_device_queue_dispatch(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t *executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  (void)device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)executable;
  (void)function;
  (void)config;
  (void)constants;
  (void)bindings;
  (void)flags;
  return LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED("queue dispatch");
}

static iree_status_t lrrt_iree_hal_device_queue_execute(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t *command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  (void)device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)command_buffer;
  (void)binding_table;
  (void)flags;
  return LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED("queue execute");
}

static iree_status_t
lrrt_iree_hal_device_queue_flush(iree_hal_device_t *device,
                                 iree_hal_queue_affinity_t queue_affinity) {
  (void)device;
  (void)queue_affinity;
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_device_profiling_begin(
    iree_hal_device_t *device,
    const iree_hal_device_profiling_options_t *options) {
  (void)device;
  (void)options;
  return iree_ok_status();
}

static iree_status_t
lrrt_iree_hal_device_profiling_flush(iree_hal_device_t *device) {
  (void)device;
  return iree_ok_status();
}

static iree_status_t
lrrt_iree_hal_device_profiling_end(iree_hal_device_t *device) {
  (void)device;
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_device_external_capture_begin(
    iree_hal_device_t *device,
    const iree_hal_device_external_capture_options_t *options) {
  (void)device;
  (void)options;
  return LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED("external capture begin");
}

static iree_status_t
lrrt_iree_hal_device_external_capture_end(iree_hal_device_t *device) {
  (void)device;
  return LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED("external capture end");
}

static iree_status_t lrrt_iree_hal_driver_query_available_devices(
    iree_hal_driver_t *base_driver, iree_allocator_t host_allocator,
    iree_host_size_t *out_device_info_count,
    iree_hal_device_info_t **out_device_infos) {
  IREE_ASSERT_ARGUMENT(base_driver);
  IREE_ASSERT_ARGUMENT(out_device_info_count);
  IREE_ASSERT_ARGUMENT(out_device_infos);
  *out_device_info_count = 0;
  *out_device_infos = NULL;

  static const iree_string_view_t device_path = IREE_SVL("default");
  static const iree_string_view_t device_name =
      IREE_SVL("lrrt default AMD GPU");
  const iree_host_size_t total_size =
      sizeof(iree_hal_device_info_t) + device_path.size + device_name.size;

  iree_hal_device_info_t *device_infos = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_size,
                                             (void **)&device_infos));
  memset(device_infos, 0, sizeof(*device_infos));

  uint8_t *buffer_ptr = (uint8_t *)device_infos + sizeof(*device_infos);
  device_infos[0].device_id = 1;
  buffer_ptr += iree_string_view_append_to_buffer(
      device_path, &device_infos[0].path, (char *)buffer_ptr);
  iree_string_view_append_to_buffer(device_name, &device_infos[0].name,
                                    (char *)buffer_ptr);

  *out_device_info_count = 1;
  *out_device_infos = device_infos;
  return iree_ok_status();
}

static iree_status_t
lrrt_iree_hal_driver_dump_device_info(iree_hal_driver_t *base_driver,
                                      iree_hal_device_id_t device_id,
                                      iree_string_builder_t *builder) {
  (void)base_driver;
  (void)device_id;
  (void)builder;
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_driver_create_device_by_id(
    iree_hal_driver_t *base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t *params,
    const iree_hal_device_create_params_t *create_params,
    iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
  (void)base_driver;
  (void)param_count;
  (void)params;
  (void)create_params;
  IREE_ASSERT_ARGUMENT(out_device);
  if (device_id != IREE_HAL_DEVICE_ID_DEFAULT && device_id != 1) {
    *out_device = NULL;
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "unknown lrrt HAL device id: %" PRIuPTR,
                            (uintptr_t)device_id);
  }
  return lrrt_iree_hal_device_create(IREE_SV("lrrt"), host_allocator,
                                     out_device);
}

static iree_status_t lrrt_iree_hal_driver_create_device_by_path(
    iree_hal_driver_t *base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t *params,
    const iree_hal_device_create_params_t *create_params,
    iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
  (void)base_driver;
  (void)driver_name;
  (void)param_count;
  (void)params;
  (void)create_params;
  IREE_ASSERT_ARGUMENT(out_device);
  if (!iree_string_view_is_empty(device_path) &&
      !iree_string_view_equal(device_path, IREE_SV("default"))) {
    *out_device = NULL;
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "unknown lrrt HAL device path: '%.*s'",
                            (int)device_path.size, device_path.data);
  }
  return lrrt_iree_hal_device_create(IREE_SV("lrrt"), host_allocator,
                                     out_device);
}

static const iree_hal_driver_vtable_t lrrt_iree_hal_driver_vtable = {
    .destroy = lrrt_iree_hal_driver_destroy,
    .query_available_devices = lrrt_iree_hal_driver_query_available_devices,
    .dump_device_info = lrrt_iree_hal_driver_dump_device_info,
    .create_device_by_id = lrrt_iree_hal_driver_create_device_by_id,
    .create_device_by_path = lrrt_iree_hal_driver_create_device_by_path,
};

static const iree_hal_allocator_vtable_t lrrt_iree_hal_allocator_vtable = {
    .destroy = lrrt_iree_hal_allocator_destroy,
    .host_allocator = lrrt_iree_hal_allocator_host_allocator,
    .trim = lrrt_iree_hal_allocator_trim,
    .query_statistics = lrrt_iree_hal_allocator_query_statistics,
    .query_memory_heaps = lrrt_iree_hal_allocator_query_memory_heaps,
    .query_buffer_compatibility =
        lrrt_iree_hal_allocator_query_buffer_compatibility,
    .allocate_buffer = lrrt_iree_hal_allocator_allocate_buffer,
    .deallocate_buffer = lrrt_iree_hal_allocator_deallocate_buffer,
    .import_buffer = lrrt_iree_hal_allocator_import_buffer,
    .export_buffer = lrrt_iree_hal_allocator_export_buffer,
    .supports_virtual_memory = lrrt_iree_hal_allocator_supports_virtual_memory,
    .virtual_memory_query_granularity =
        lrrt_iree_hal_allocator_virtual_memory_query_granularity,
    .virtual_memory_reserve = lrrt_iree_hal_allocator_virtual_memory_reserve,
    .virtual_memory_release = lrrt_iree_hal_allocator_virtual_memory_release,
    .physical_memory_allocate =
        lrrt_iree_hal_allocator_physical_memory_allocate,
    .physical_memory_free = lrrt_iree_hal_allocator_physical_memory_free,
    .virtual_memory_map = lrrt_iree_hal_allocator_virtual_memory_map,
    .virtual_memory_unmap = lrrt_iree_hal_allocator_virtual_memory_unmap,
    .virtual_memory_protect = lrrt_iree_hal_allocator_virtual_memory_protect,
    .virtual_memory_advise = lrrt_iree_hal_allocator_virtual_memory_advise,
};

static const iree_hal_buffer_vtable_t lrrt_iree_hal_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = lrrt_iree_hal_buffer_destroy,
    .map_range = lrrt_iree_hal_buffer_map_range,
    .unmap_range = lrrt_iree_hal_buffer_unmap_range,
    .invalidate_range = lrrt_iree_hal_buffer_invalidate_range,
    .flush_range = lrrt_iree_hal_buffer_flush_range,
};

static const iree_hal_executable_cache_vtable_t
    lrrt_iree_hal_executable_cache_vtable = {
        .destroy = lrrt_iree_hal_executable_cache_destroy,
        .infer_format = lrrt_iree_hal_executable_cache_infer_format,
        .can_prepare_format = lrrt_iree_hal_executable_cache_can_prepare_format,
        .prepare_executable = lrrt_iree_hal_executable_cache_prepare_executable,
};

static const iree_hal_device_vtable_t lrrt_iree_hal_device_vtable = {
    .destroy = lrrt_iree_hal_device_destroy,
    .id = lrrt_iree_hal_device_id,
    .host_allocator = lrrt_iree_hal_device_host_allocator,
    .device_allocator = lrrt_iree_hal_device_allocator,
    .replace_device_allocator = lrrt_iree_hal_device_replace_allocator,
    .replace_channel_provider = lrrt_iree_hal_device_replace_channel_provider,
    .trim = lrrt_iree_hal_device_trim,
    .query_i64 = lrrt_iree_hal_device_query_i64,
    .query_capabilities = lrrt_iree_hal_device_query_capabilities,
    .topology_info = lrrt_iree_hal_device_topology_info,
    .refine_topology_edge = lrrt_iree_hal_device_refine_topology_edge,
    .assign_topology_info = lrrt_iree_hal_device_assign_topology_info,
    .create_channel = lrrt_iree_hal_device_create_channel,
    .create_command_buffer = lrrt_iree_hal_device_create_command_buffer,
    .create_event = lrrt_iree_hal_device_create_event,
    .create_executable_cache = lrrt_iree_hal_device_create_executable_cache,
    .import_file = lrrt_iree_hal_device_import_file,
    .create_semaphore = lrrt_iree_hal_device_create_semaphore,
    .query_semaphore_compatibility =
        lrrt_iree_hal_device_query_semaphore_compatibility,
    .query_queue_pool_backend = lrrt_iree_hal_device_query_queue_pool_backend,
    .queue_alloca = lrrt_iree_hal_device_queue_alloca,
    .queue_dealloca = lrrt_iree_hal_device_queue_dealloca,
    .queue_fill = lrrt_iree_hal_device_queue_fill,
    .queue_update = lrrt_iree_hal_device_queue_update,
    .queue_copy = lrrt_iree_hal_device_queue_copy,
    .queue_read = lrrt_iree_hal_device_queue_read,
    .queue_write = lrrt_iree_hal_device_queue_write,
    .queue_host_call = lrrt_iree_hal_device_queue_host_call,
    .queue_dispatch = lrrt_iree_hal_device_queue_dispatch,
    .queue_execute = lrrt_iree_hal_device_queue_execute,
    .queue_flush = lrrt_iree_hal_device_queue_flush,
    .profiling_begin = lrrt_iree_hal_device_profiling_begin,
    .profiling_flush = lrrt_iree_hal_device_profiling_flush,
    .profiling_end = lrrt_iree_hal_device_profiling_end,
    .external_capture_begin = lrrt_iree_hal_device_external_capture_begin,
    .external_capture_end = lrrt_iree_hal_device_external_capture_end,
};

static iree_status_t lrrt_iree_hal_driver_factory_enumerate(
    void *self, iree_host_size_t *out_driver_info_count,
    const iree_hal_driver_info_t **out_driver_infos) {
  (void)self;
  IREE_ASSERT_ARGUMENT(out_driver_info_count);
  IREE_ASSERT_ARGUMENT(out_driver_infos);
  static const iree_hal_driver_info_t driver_infos[1] = {{
      .driver_name = IREE_SVL("lrrt"),
      .full_name = IREE_SVL("lrrt HAL driver skeleton"),
  }};
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_driver_factory_try_create(
    void *self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t **out_driver) {
  (void)self;
  if (!iree_string_view_equal(driver_name, IREE_SV("lrrt"))) {
    *out_driver = NULL;
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "lrrt HAL driver factory does not provide '%.*s'",
                            (int)driver_name.size, driver_name.data);
  }
  return lrrt_iree_hal_driver_create(driver_name, host_allocator, out_driver);
}

iree_status_t
lrrt_iree_hal_driver_module_register(iree_hal_driver_registry_t *registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = lrrt_iree_hal_driver_factory_enumerate,
      .try_create = lrrt_iree_hal_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
