#include "hal_driver.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "executor/iree/registration/driver_module.h"
#include "iree/async/semaphore.h"
#include "iree/hal/resource.h"
#include "iree/hal/utils/executable_header.h"
#include "iree/hal/utils/file_registry.h"
#include "iree/schemas/hip_executable_def_reader.h"
#include "iree/schemas/hip_executable_def_verifier.h"
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
  void *host_shadow_ptr;
  iree_device_size_t host_shadow_size;
} lrrt_iree_hal_buffer_t;

typedef struct lrrt_iree_hal_executable_cache_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  lr_device_t device;
  iree_string_view_t identifier;
} lrrt_iree_hal_executable_cache_t;

typedef struct lrrt_iree_hal_executable_function_t {
  iree_string_view_t name;
  lr_kernel_t *kernel;
  uint32_t default_workgroup_size[3];
} lrrt_iree_hal_executable_function_t;

typedef struct lrrt_iree_hal_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  lr_module_t *module;
  iree_host_size_t function_count;
  iree_host_size_t function_capacity;
  lrrt_iree_hal_executable_function_t *functions;
} lrrt_iree_hal_executable_t;

typedef struct lrrt_iree_hal_device_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_allocator_t *device_allocator;
  iree_string_view_t identifier;
  iree_hal_device_topology_info_t topology_info;
} lrrt_iree_hal_device_t;

typedef struct lrrt_iree_hal_semaphore_t {
  iree_async_semaphore_t async;
  iree_allocator_t host_allocator;
} lrrt_iree_hal_semaphore_t;

enum { LRRT_IREE_HAL_MAX_INLINE_BINDINGS = 64 };

typedef enum lrrt_iree_hal_command_kind_t {
  LRRT_IREE_HAL_COMMAND_UPDATE,
  LRRT_IREE_HAL_COMMAND_COPY,
  LRRT_IREE_HAL_COMMAND_FILL,
  LRRT_IREE_HAL_COMMAND_DISPATCH,
} lrrt_iree_hal_command_kind_t;

typedef struct lrrt_iree_hal_dispatch_record_t {
  iree_hal_executable_t *executable;
  iree_hal_executable_function_t function;
  iree_hal_dispatch_config_t config;
  iree_hal_dispatch_flags_t flags;
  iree_host_size_t binding_count;
  iree_hal_buffer_ref_t *bindings;
} lrrt_iree_hal_dispatch_record_t;

typedef struct lrrt_iree_hal_update_record_t {
  void *source_buffer;
  iree_host_size_t source_offset;
  iree_hal_buffer_ref_t target_ref;
  iree_hal_update_flags_t flags;
} lrrt_iree_hal_update_record_t;

typedef struct lrrt_iree_hal_copy_record_t {
  iree_hal_buffer_ref_t source_ref;
  iree_hal_buffer_ref_t target_ref;
  iree_hal_copy_flags_t flags;
} lrrt_iree_hal_copy_record_t;

typedef struct lrrt_iree_hal_fill_record_t {
  uint8_t pattern[4];
  iree_host_size_t pattern_length;
  iree_hal_buffer_ref_t target_ref;
  iree_hal_fill_flags_t flags;
} lrrt_iree_hal_fill_record_t;

typedef struct lrrt_iree_hal_command_record_t {
  lrrt_iree_hal_command_kind_t kind;
  union {
    lrrt_iree_hal_update_record_t update;
    lrrt_iree_hal_copy_record_t copy;
    lrrt_iree_hal_fill_record_t fill;
    lrrt_iree_hal_dispatch_record_t dispatch;
  } payload;
} lrrt_iree_hal_command_record_t;

typedef struct lrrt_iree_hal_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  iree_host_size_t command_count;
  iree_host_size_t command_capacity;
  lrrt_iree_hal_command_record_t *commands;
} lrrt_iree_hal_command_buffer_t;

static const iree_hal_driver_vtable_t lrrt_iree_hal_driver_vtable;
static const iree_hal_allocator_vtable_t lrrt_iree_hal_allocator_vtable;
static const iree_hal_buffer_vtable_t lrrt_iree_hal_buffer_vtable;
static const iree_hal_executable_cache_vtable_t
    lrrt_iree_hal_executable_cache_vtable;
static const iree_hal_executable_vtable_t lrrt_iree_hal_executable_vtable;
static const iree_hal_semaphore_vtable_t lrrt_iree_hal_semaphore_vtable;
static const iree_hal_command_buffer_vtable_t
    lrrt_iree_hal_command_buffer_vtable;
static const iree_hal_device_vtable_t lrrt_iree_hal_device_vtable;

static const iree_string_view_t kLrrtIreeHalExecutableFormatRocmHsaco =
    IREE_SVL("rocm-hsaco");
static const iree_string_view_t kLrrtIreeHalExecutableFormatAmdgpuHsaco =
    IREE_SVL("amdgpu-hsaco");
static const iree_string_view_t kLrrtIreeHalExecutableFormatRocmHsacoFb =
    IREE_SVL("rocm-hsaco-fb");

static uint32_t g_lrrt_iree_hal_runtime_ref_count = 0;
static bool g_lrrt_iree_hal_owns_runtime = false;

static bool lrrt_iree_hal_trace_enabled(void) {
  const char *value = getenv("LRRT_IREE_TRACE");
  return value && value[0] != '\0' && strcmp(value, "0") != 0 &&
         strcmp(value, "false") != 0 && strcmp(value, "FALSE") != 0;
}

static void lrrt_iree_hal_tracef(const char *format, ...) {
  if (!lrrt_iree_hal_trace_enabled()) {
    return;
  }
  fprintf(stderr, "[lrrt-iree] ");
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fputc('\n', stderr);
}

static void lrrt_iree_hal_trace_status(const char *operation,
                                       iree_status_t status) {
  lrrt_iree_hal_tracef("%s end status=%s", operation,
                       iree_status_is_ok(status) ? "ok" : "error");
}

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

static lrrt_iree_hal_command_buffer_t *
lrrt_iree_hal_command_buffer_cast(iree_hal_command_buffer_t *base_buffer) {
  IREE_HAL_ASSERT_TYPE(base_buffer, &lrrt_iree_hal_command_buffer_vtable);
  return (lrrt_iree_hal_command_buffer_t *)base_buffer;
}

static lrrt_iree_hal_semaphore_t *
lrrt_iree_hal_semaphore_cast(iree_hal_semaphore_t *base_semaphore) {
  IREE_HAL_ASSERT_TYPE(base_semaphore, &lrrt_iree_hal_semaphore_vtable);
  return (lrrt_iree_hal_semaphore_t *)base_semaphore;
}

static uint64_t
lrrt_iree_hal_semaphore_query(iree_async_semaphore_t *base_semaphore);

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
  lrrt_iree_hal_tracef("runtime retain begin ref_count=%u",
                       g_lrrt_iree_hal_runtime_ref_count);
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
  lrrt_iree_hal_tracef("runtime retain end ref_count=%u owns_runtime=%d",
                       g_lrrt_iree_hal_runtime_ref_count,
                       g_lrrt_iree_hal_owns_runtime ? 1 : 0);
  return iree_ok_status();
}

static void lrrt_iree_hal_runtime_release(void) {
  if (g_lrrt_iree_hal_runtime_ref_count == 0) {
    return;
  }
  lrrt_iree_hal_tracef("runtime release begin ref_count=%u",
                       g_lrrt_iree_hal_runtime_ref_count);
  --g_lrrt_iree_hal_runtime_ref_count;
  if (g_lrrt_iree_hal_runtime_ref_count == 0 && g_lrrt_iree_hal_owns_runtime) {
    lrrt_iree_hal_tracef("runtime shutdown");
    lr_shutdown();
    g_lrrt_iree_hal_owns_runtime = false;
  }
  lrrt_iree_hal_tracef("runtime release end ref_count=%u owns_runtime=%d",
                       g_lrrt_iree_hal_runtime_ref_count,
                       g_lrrt_iree_hal_owns_runtime ? 1 : 0);
}

static iree_status_t
lrrt_iree_hal_allocator_create(iree_allocator_t host_allocator,
                               iree_hal_device_t *hal_device,
                               iree_hal_allocator_t **out_allocator) {
  IREE_ASSERT_ARGUMENT(out_allocator);
  *out_allocator = NULL;

  lrrt_iree_hal_tracef("allocator create begin");
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
  lrrt_iree_hal_tracef("allocator found device_count=%u", device_count);

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
  lrrt_iree_hal_tracef("allocator create end device_index=0");
  return iree_ok_status();
}

static iree_status_t
lrrt_iree_hal_device_create(iree_string_view_t identifier,
                            iree_allocator_t host_allocator,
                            iree_hal_device_t **out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;

  lrrt_iree_hal_tracef("device create begin id='%.*s'", (int)identifier.size,
                       identifier.data);
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
  lrrt_iree_hal_tracef("device create end id='%.*s'", (int)identifier.size,
                       identifier.data);
  return iree_ok_status();
}

static iree_status_t
lrrt_iree_hal_driver_create(iree_string_view_t identifier,
                            iree_allocator_t host_allocator,
                            iree_hal_driver_t **out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = NULL;

  lrrt_iree_hal_tracef("driver create begin id='%.*s'", (int)identifier.size,
                       identifier.data);
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
  lrrt_iree_hal_tracef("driver create end id='%.*s'", (int)identifier.size,
                       identifier.data);
  return iree_ok_status();
}

static void lrrt_iree_hal_driver_destroy(iree_hal_driver_t *base_driver) {
  lrrt_iree_hal_driver_t *driver = lrrt_iree_hal_driver_cast(base_driver);
  lrrt_iree_hal_tracef("driver destroy id='%.*s'", (int)driver->identifier.size,
                       driver->identifier.data);
  iree_allocator_t host_allocator = driver->host_allocator;
  iree_allocator_free(host_allocator, driver);
}

static void
lrrt_iree_hal_allocator_destroy(iree_hal_allocator_t *base_allocator) {
  lrrt_iree_hal_allocator_t *allocator =
      lrrt_iree_hal_allocator_cast(base_allocator);
  lrrt_iree_hal_tracef("allocator destroy");
  iree_allocator_t host_allocator = allocator->host_allocator;
  lrrt_iree_hal_runtime_release();
  iree_allocator_free(host_allocator, allocator);
}

static void lrrt_iree_hal_device_destroy(iree_hal_device_t *base_device) {
  lrrt_iree_hal_device_t *device = lrrt_iree_hal_device_cast(base_device);
  lrrt_iree_hal_tracef("device destroy id='%.*s'", (int)device->identifier.size,
                       device->identifier.data);
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
  if ((params->type & IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE) == 0 &&
      (params->type & IREE_HAL_MEMORY_TYPE_OPTIMAL) == 0 &&
      (params->type & IREE_HAL_MEMORY_TYPE_HOST_LOCAL) == 0) {
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
                            "lrrt HAL allocator only supports device-visible "
                            "or host-local buffers");
  }

  lrrt_iree_hal_allocator_t *allocator =
      lrrt_iree_hal_allocator_cast(base_allocator);
  lrrt_iree_hal_tracef("buffer allocate size=%" PRIu64 " type=0x%" PRIx64
                       " usage=0x%" PRIx64,
                       (uint64_t)allocation_size, (uint64_t)params->type,
                       (uint64_t)params->usage);
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
  buffer->host_shadow_ptr = NULL;
  buffer->host_shadow_size = 0;
  iree_hal_allocator_retain(base_allocator);

  *out_buffer = &buffer->base;
  lrrt_iree_hal_tracef(
      "buffer allocate end buffer=%p device_ptr=%p size=%" PRIu64,
      (void *)&buffer->base, device_ptr, (uint64_t)allocation_size);
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
  if (!base_buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "buffer is null");
  }
  if (!iree_hal_resource_is(base_buffer, &lrrt_iree_hal_buffer_vtable)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer is not an lrrt HAL buffer");
  }
  iree_device_size_t byte_length = iree_hal_buffer_byte_length(base_buffer);
  if (offset > byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer range out of bounds: offset=%" PRIu64
                            ", byte_length=%" PRIu64,
                            (uint64_t)offset, (uint64_t)byte_length);
  }
  if (length == IREE_HAL_WHOLE_BUFFER) {
    length = byte_length - offset;
  }
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

static iree_status_t
lrrt_iree_hal_buffer_ref_device_range(iree_hal_buffer_ref_t buffer_ref,
                                      void **out_device_ptr) {
  IREE_ASSERT_ARGUMENT(out_device_ptr);
  *out_device_ptr = NULL;
  if (!buffer_ref.buffer) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unresolved indirect HAL buffer references are "
                            "not supported by lrrt dispatch");
  }
  return lrrt_iree_hal_buffer_device_range(buffer_ref.buffer, buffer_ref.offset,
                                           buffer_ref.length, out_device_ptr);
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
  lrrt_iree_hal_tracef("buffer destroy buffer=%p device_ptr=%p size=%" PRIu64,
                       (void *)base_buffer, buffer->device_ptr,
                       (uint64_t)iree_hal_buffer_allocation_size(base_buffer));
  iree_allocator_t host_allocator = buffer->host_allocator;
  if (buffer->host_shadow_ptr) {
    iree_allocator_free(host_allocator, buffer->host_shadow_ptr);
    buffer->host_shadow_ptr = NULL;
    buffer->host_shadow_size = 0;
  }
  if (buffer->device_ptr) {
    lr_free(buffer->device, buffer->device_ptr);
    buffer->device_ptr = NULL;
  }
  iree_hal_allocator_release(buffer->allocator);
  iree_allocator_free(host_allocator, buffer);
}

static iree_status_t
lrrt_iree_hal_buffer_ensure_host_shadow(lrrt_iree_hal_buffer_t *buffer) {
  iree_device_size_t allocation_size =
      iree_hal_buffer_allocation_size(&buffer->base);
  if (buffer->host_shadow_ptr && buffer->host_shadow_size >= allocation_size) {
    return iree_ok_status();
  }
  if (allocation_size > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "lrrt HAL buffer is too large to host-map");
  }
  void *host_shadow_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(buffer->host_allocator,
                                             (iree_host_size_t)allocation_size,
                                             &host_shadow_ptr));
  if (buffer->host_shadow_ptr) {
    iree_allocator_free(buffer->host_allocator, buffer->host_shadow_ptr);
  }
  buffer->host_shadow_ptr = host_shadow_ptr;
  buffer->host_shadow_size = allocation_size;
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_buffer_map_range(
    iree_hal_buffer_t *base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t *out_mapping) {
  (void)mapping_mode;
  IREE_ASSERT_ARGUMENT(out_mapping);
  lrrt_iree_hal_buffer_t *buffer = lrrt_iree_hal_buffer_cast(base_buffer);
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_buffer_ensure_host_shadow(buffer));
  if (iree_all_bits_set(memory_access, IREE_HAL_MEMORY_ACCESS_READ)) {
    lr_status_t lr_status = lr_memcpy(
        buffer->device, (uint8_t *)buffer->host_shadow_ptr + local_byte_offset,
        (uint8_t *)buffer->device_ptr + local_byte_offset,
        (size_t)local_byte_length, LR_MEMCPY_DEVICE_TO_HOST);
    IREE_RETURN_IF_ERROR(lrrt_iree_hal_status_from_lr(lr_status, "lr_memcpy"));
  }
  out_mapping->contents = iree_make_byte_span(
      (uint8_t *)buffer->host_shadow_ptr + local_byte_offset,
      (iree_host_size_t)local_byte_length);
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_buffer_unmap_range(
    iree_hal_buffer_t *base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t *mapping) {
  lrrt_iree_hal_buffer_t *buffer = lrrt_iree_hal_buffer_cast(base_buffer);
  if (mapping &&
      iree_all_bits_set(mapping->impl.allowed_access,
                        IREE_HAL_MEMORY_ACCESS_WRITE) &&
      buffer->host_shadow_ptr) {
    lr_status_t lr_status = lr_memcpy(
        buffer->device, (uint8_t *)buffer->device_ptr + local_byte_offset,
        (uint8_t *)buffer->host_shadow_ptr + local_byte_offset,
        (size_t)local_byte_length, LR_MEMCPY_HOST_TO_DEVICE);
    IREE_RETURN_IF_ERROR(lrrt_iree_hal_status_from_lr(lr_status, "lr_memcpy"));
  }
  return iree_ok_status();
}

static iree_status_t
lrrt_iree_hal_buffer_invalidate_range(iree_hal_buffer_t *base_buffer,
                                      iree_device_size_t local_byte_offset,
                                      iree_device_size_t local_byte_length) {
  lrrt_iree_hal_buffer_t *buffer = lrrt_iree_hal_buffer_cast(base_buffer);
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_buffer_ensure_host_shadow(buffer));
  lr_status_t lr_status = lr_memcpy(
      buffer->device, (uint8_t *)buffer->host_shadow_ptr + local_byte_offset,
      (uint8_t *)buffer->device_ptr + local_byte_offset,
      (size_t)local_byte_length, LR_MEMCPY_DEVICE_TO_HOST);
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_status_from_lr(lr_status, "lr_memcpy"));
  return iree_ok_status();
}

static iree_status_t
lrrt_iree_hal_buffer_flush_range(iree_hal_buffer_t *base_buffer,
                                 iree_device_size_t local_byte_offset,
                                 iree_device_size_t local_byte_length) {
  lrrt_iree_hal_buffer_t *buffer = lrrt_iree_hal_buffer_cast(base_buffer);
  if (!buffer->host_shadow_ptr) {
    return iree_ok_status();
  }
  lr_status_t lr_status = lr_memcpy(
      buffer->device, (uint8_t *)buffer->device_ptr + local_byte_offset,
      (uint8_t *)buffer->host_shadow_ptr + local_byte_offset,
      (size_t)local_byte_length, LR_MEMCPY_HOST_TO_DEVICE);
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_status_from_lr(lr_status, "lr_memcpy"));
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

static lrrt_iree_hal_executable_t *
lrrt_iree_hal_executable_cast(iree_hal_executable_t *base_executable) {
  IREE_HAL_ASSERT_TYPE(base_executable, &lrrt_iree_hal_executable_vtable);
  return (lrrt_iree_hal_executable_t *)base_executable;
}

static void
lrrt_iree_hal_executable_destroy(iree_hal_executable_t *base_executable) {
  lrrt_iree_hal_executable_t *executable =
      lrrt_iree_hal_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  for (iree_host_size_t i = 0; i < executable->function_count; ++i) {
    iree_allocator_free(host_allocator,
                        (void *)executable->functions[i].name.data);
  }
  iree_allocator_free(host_allocator, executable->functions);
  if (executable->module) {
    lr_module_destroy(executable->module);
  }
  iree_allocator_free(host_allocator, executable);
}

static iree_host_size_t lrrt_iree_hal_executable_function_count(
    iree_hal_executable_t *base_executable) {
  lrrt_iree_hal_executable_t *executable =
      lrrt_iree_hal_executable_cast(base_executable);
  return executable->function_count;
}

static iree_status_t lrrt_iree_hal_executable_function_info(
    iree_hal_executable_t *base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t *out_info) {
  lrrt_iree_hal_executable_t *executable =
      lrrt_iree_hal_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "lrrt HAL executable function index %" PRIu64
                            " is out of range; function count is %" PRIhsz,
                            function.value, executable->function_count);
  }
  memset(out_info, 0, sizeof(*out_info));
  out_info->name =
      executable->functions[iree_hal_executable_function_index(function)].name;
  out_info->flags = IREE_HAL_EXECUTABLE_FUNCTION_FLAG_NONE;
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_executable_function_parameters(
    iree_hal_executable_t *base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t *out_parameters) {
  lrrt_iree_hal_executable_t *executable =
      lrrt_iree_hal_executable_cast(base_executable);
  (void)capacity;
  (void)out_parameters;
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "lrrt HAL executable function index %" PRIu64
                            " is out of range; function count is %" PRIhsz,
                            function.value, executable->function_count);
  }
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_executable_add_function(
    lrrt_iree_hal_executable_t *executable, iree_string_view_t name,
    const uint32_t default_workgroup_size[3],
    iree_hal_executable_function_t *out_function) {
  char *kernel_name = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      executable->host_allocator, name.size + 1, (void **)&kernel_name));
  memcpy(kernel_name, name.data, name.size);
  kernel_name[name.size] = '\0';

  lr_kernel_t *kernel = NULL;
  lrrt_iree_hal_tracef("executable add function name='%.*s'", (int)name.size,
                       name.data);
  lr_status_t lr_status =
      lr_kernel_get(executable->module, kernel_name, &kernel);
  if (lr_status != LR_SUCCESS) {
    iree_allocator_free(executable->host_allocator, kernel_name);
    if (lr_status == LR_ERROR_INVALID_ARGUMENT) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "lrrt HAL executable function '%.*s' not found",
                              (int)name.size, name.data);
    }
    return lrrt_iree_hal_status_from_lr(lr_status, "lr_kernel_get");
  }

  if (executable->function_count == executable->function_capacity) {
    iree_status_t status = iree_allocator_grow_array(
        executable->host_allocator, executable->function_count + 1,
        sizeof(*executable->functions), &executable->function_capacity,
        (void **)&executable->functions);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(executable->host_allocator, kernel_name);
      return status;
    }
  }
  const iree_host_size_t function_index = executable->function_count++;
  executable->functions[function_index].name =
      iree_make_string_view(kernel_name, name.size);
  executable->functions[function_index].kernel = kernel;
  memcpy(executable->functions[function_index].default_workgroup_size,
         default_workgroup_size,
         sizeof(executable->functions[function_index].default_workgroup_size));
  if (out_function) {
    *out_function =
        iree_hal_executable_function_from_index((uint32_t)function_index);
  }
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_executable_lookup_function_by_name(
    iree_hal_executable_t *base_executable, iree_string_view_t name,
    iree_hal_executable_function_t *out_function) {
  IREE_ASSERT_ARGUMENT(out_function);
  *out_function = iree_hal_executable_function_invalid();
  if (iree_string_view_is_empty(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "lrrt HAL executable function name is empty");
  }

  lrrt_iree_hal_tracef("executable lookup function name='%.*s'", (int)name.size,
                       name.data);
  lrrt_iree_hal_executable_t *executable =
      lrrt_iree_hal_executable_cast(base_executable);
  for (iree_host_size_t i = 0; i < executable->function_count; ++i) {
    if (iree_string_view_equal(executable->functions[i].name, name)) {
      *out_function = iree_hal_executable_function_from_index((uint32_t)i);
      lrrt_iree_hal_tracef(
          "executable lookup function hit name='%.*s' index=%" PRIhsz,
          (int)name.size, name.data, i);
      return iree_ok_status();
    }
  }

  const uint32_t no_default_workgroup_size[3] = {0, 0, 0};
  return lrrt_iree_hal_executable_add_function(
      executable, name, no_default_workgroup_size, out_function);
}

static iree_status_t lrrt_iree_hal_executable_kernel_for_function(
    iree_hal_executable_t *base_executable,
    iree_hal_executable_function_t function, lr_kernel_t **out_kernel) {
  IREE_ASSERT_ARGUMENT(out_kernel);
  *out_kernel = NULL;
  lrrt_iree_hal_executable_t *executable =
      lrrt_iree_hal_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "lrrt HAL executable function index %" PRIu64
                            " is out of range; function count is %" PRIhsz,
                            function.value, executable->function_count);
  }
  lr_kernel_t *kernel =
      executable->functions[iree_hal_executable_function_index(function)]
          .kernel;
  if (!kernel) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "lrrt HAL executable function has no kernel");
  }
  *out_kernel = kernel;
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_executable_resolve_workgroup_size(
    iree_hal_executable_t *base_executable,
    iree_hal_executable_function_t function, iree_hal_dispatch_config_t *config,
    const char *operation) {
  lrrt_iree_hal_executable_t *executable =
      lrrt_iree_hal_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "lrrt HAL executable function index %" PRIu64
                            " is out of range; function count is %" PRIhsz,
                            function.value, executable->function_count);
  }
  if (config->workgroup_size[0] != 0 && config->workgroup_size[1] != 0 &&
      config->workgroup_size[2] != 0) {
    return iree_ok_status();
  }

  lrrt_iree_hal_executable_function_t *function_state =
      &executable->functions[iree_hal_executable_function_index(function)];
  if (function_state->default_workgroup_size[0] == 0 ||
      function_state->default_workgroup_size[1] == 0 ||
      function_state->default_workgroup_size[2] == 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "lrrt HAL %s requires explicit workgroup size metadata", operation);
  }
  config->workgroup_size[0] = function_state->default_workgroup_size[0];
  config->workgroup_size[1] = function_state->default_workgroup_size[1];
  config->workgroup_size[2] = function_state->default_workgroup_size[2];
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_executable_lookup_global_by_name(
    iree_hal_executable_t *base_executable, iree_string_view_t name,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t **out_buffer) {
  (void)base_executable;
  (void)name;
  (void)queue_affinity;
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "lrrt HAL executable globals are not implemented");
}

static iree_status_t
lrrt_iree_hal_executable_create(lr_module_t *module,
                                iree_allocator_t host_allocator,
                                iree_hal_executable_t **out_executable) {
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  lrrt_iree_hal_executable_t *executable = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*executable), (void **)&executable));
  iree_hal_resource_initialize(&lrrt_iree_hal_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->module = module;
  executable->function_count = 0;
  executable->function_capacity = 0;
  executable->functions = NULL;
  *out_executable = (iree_hal_executable_t *)executable;
  lrrt_iree_hal_tracef("executable create module=%p", (void *)module);
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_executable_cache_create(
    iree_string_view_t identifier, iree_allocator_t host_allocator,
    lr_device_t device, iree_hal_executable_cache_t **out_executable_cache) {
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
  executable_cache->device = device;
  iree_string_view_append_to_buffer(identifier, &executable_cache->identifier,
                                    (char *)executable_cache +
                                        iree_sizeof_struct(*executable_cache));

  *out_executable_cache = (iree_hal_executable_cache_t *)executable_cache;
  lrrt_iree_hal_tracef("executable cache create id='%.*s'",
                       (int)identifier.size, identifier.data);
  return iree_ok_status();
}

static void lrrt_iree_hal_executable_cache_destroy(
    iree_hal_executable_cache_t *base_executable_cache) {
  lrrt_iree_hal_executable_cache_t *executable_cache =
      lrrt_iree_hal_executable_cache_cast(base_executable_cache);
  lrrt_iree_hal_tracef("executable cache destroy id='%.*s'",
                       (int)executable_cache->identifier.size,
                       executable_cache->identifier.data);
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
                                kLrrtIreeHalExecutableFormatAmdgpuHsaco) ||
         iree_string_view_equal(executable_format,
                                kLrrtIreeHalExecutableFormatRocmHsacoFb);
}

static iree_status_t
lrrt_iree_hal_extract_rocm_hsaco_fb(iree_const_byte_span_t executable_data,
                                    iree_const_byte_span_t *out_hsaco_data) {
  IREE_ASSERT_ARGUMENT(out_hsaco_data);
  *out_hsaco_data = iree_const_byte_span_empty();

  iree_const_byte_span_t flatbuffer_data = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_read_executable_flatbuffer_header(
      executable_data, /*unsafe_infer_size=*/false,
      iree_hal_hip_ExecutableDef_file_identifier, &flatbuffer_data));

  int verify_ret = iree_hal_hip_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "rocm-hsaco-fb verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_hip_ExecutableDef_table_t executable_def =
      iree_hal_hip_ExecutableDef_as_root(flatbuffer_data.data);
  iree_hal_hip_ModuleDef_vec_t modules_vec =
      iree_hal_hip_ExecutableDef_modules_get(executable_def);
  iree_host_size_t module_count = iree_hal_hip_ModuleDef_vec_len(modules_vec);
  if (module_count != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "lrrt HAL executable cache requires exactly one rocm-hsaco-fb module, "
        "got %" PRIhsz,
        module_count);
  }

  iree_hal_hip_ModuleDef_table_t module_def =
      iree_hal_hip_ModuleDef_vec_at(modules_vec, 0);
  const char *hsaco_image = iree_hal_hip_ModuleDef_hsaco_image_get(module_def);
  iree_host_size_t hsaco_size = flatbuffers_string_len(hsaco_image);
  if (!hsaco_image || hsaco_size == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "lrrt HAL executable cache requires non-empty rocm-hsaco-fb module "
        "data");
  }

  *out_hsaco_data =
      iree_make_const_byte_span((const uint8_t *)hsaco_image, hsaco_size);
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_register_rocm_hsaco_fb_exports(
    lrrt_iree_hal_executable_t *executable,
    iree_const_byte_span_t executable_data) {
  iree_const_byte_span_t flatbuffer_data = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_read_executable_flatbuffer_header(
      executable_data, /*unsafe_infer_size=*/false,
      iree_hal_hip_ExecutableDef_file_identifier, &flatbuffer_data));

  iree_hal_hip_ExecutableDef_table_t executable_def =
      iree_hal_hip_ExecutableDef_as_root(flatbuffer_data.data);
  iree_hal_hip_ExportDef_vec_t exports_vec =
      iree_hal_hip_ExecutableDef_exports_get(executable_def);
  iree_host_size_t export_count = iree_hal_hip_ExportDef_vec_len(exports_vec);
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    iree_hal_hip_ExportDef_table_t export_def =
        iree_hal_hip_ExportDef_vec_at(exports_vec, i);
    if (!export_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "rocm-hsaco-fb exports[%" PRIhsz "] is null", i);
    }
    uint32_t module_ordinal =
        iree_hal_hip_ExportDef_module_ordinal_get(export_def);
    if (module_ordinal != 0) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "lrrt HAL executable cache only supports rocm-hsaco-fb exports from "
          "module 0, got module %u",
          module_ordinal);
    }
    const char *kernel_name =
        iree_hal_hip_ExportDef_kernel_name_get(export_def);
    if (!kernel_name || flatbuffers_string_len(kernel_name) == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "rocm-hsaco-fb exports[%" PRIhsz "] has an empty kernel name", i);
    }
    const iree_hal_hip_BlockDims_t *block_dims =
        iree_hal_hip_ExportDef_block_dims_get(export_def);
    if (!block_dims || block_dims->x == 0 || block_dims->y == 0 ||
        block_dims->z == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "rocm-hsaco-fb exports[%" PRIhsz "] has invalid block dims", i);
    }
    const uint32_t default_workgroup_size[3] = {block_dims->x, block_dims->y,
                                                block_dims->z};
    IREE_RETURN_IF_ERROR(lrrt_iree_hal_executable_add_function(
        executable,
        iree_make_string_view(kernel_name, flatbuffers_string_len(kernel_name)),
        default_workgroup_size, NULL));
  }
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_executable_cache_prepare_executable(
    iree_hal_executable_cache_t *base_executable_cache,
    const iree_hal_executable_params_t *executable_params,
    iree_hal_executable_t **out_executable) {
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
  if (iree_const_byte_span_is_empty(executable_params->executable_data)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "lrrt HAL executable preparation requires non-empty HSACO data");
  }

  lrrt_iree_hal_tracef("executable prepare begin format='%.*s' bytes=%" PRIhsz,
                       (int)executable_params->executable_format.size,
                       executable_params->executable_format.data,
                       executable_params->executable_data.data_length);
  iree_const_byte_span_t module_data = executable_params->executable_data;
  if (iree_string_view_equal(executable_params->executable_format,
                             kLrrtIreeHalExecutableFormatRocmHsacoFb)) {
    iree_status_t extract_status = lrrt_iree_hal_extract_rocm_hsaco_fb(
        executable_params->executable_data, &module_data);
    if (!iree_status_is_ok(extract_status)) {
      lrrt_iree_hal_trace_status("executable prepare", extract_status);
      return extract_status;
    }
  }

  lrrt_iree_hal_executable_cache_t *executable_cache =
      lrrt_iree_hal_executable_cache_cast(base_executable_cache);
  lr_module_t *module = NULL;
  lr_status_t lr_status =
      lr_module_load_hsaco(executable_cache->device, module_data.data,
                           module_data.data_length, &module);
  if (lr_status != LR_SUCCESS) {
    iree_status_t status =
        lrrt_iree_hal_status_from_lr(lr_status, "lr_module_load_hsaco");
    lrrt_iree_hal_trace_status("executable prepare", status);
    return status;
  }

  iree_status_t status = lrrt_iree_hal_executable_create(
      module, executable_cache->host_allocator, out_executable);
  if (!iree_status_is_ok(status)) {
    lr_module_destroy(module);
    lrrt_iree_hal_trace_status("executable prepare", status);
    return status;
  }
  if (iree_string_view_equal(executable_params->executable_format,
                             kLrrtIreeHalExecutableFormatRocmHsacoFb)) {
    status = lrrt_iree_hal_register_rocm_hsaco_fb_exports(
        lrrt_iree_hal_executable_cast(*out_executable),
        executable_params->executable_data);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_executable_release(*out_executable);
    *out_executable = NULL;
  }
  lrrt_iree_hal_trace_status("executable prepare", status);
  return status;
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
  lrrt_iree_hal_device_t *device = lrrt_iree_hal_device_cast(base_device);
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value = iree_string_view_match_pattern(device->identifier, key) ||
                         iree_string_view_equal(key, IREE_SV("hip")) ||
                         iree_string_view_equal(key, IREE_SV("amdgpu"))
                     ? 1
                     : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value =
        iree_string_view_equal(key, kLrrtIreeHalExecutableFormatRocmHsaco) ||
                iree_string_view_equal(
                    key, kLrrtIreeHalExecutableFormatAmdgpuHsaco) ||
                iree_string_view_equal(key,
                                       kLrrtIreeHalExecutableFormatRocmHsacoFb)
            ? 1
            : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.device"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      *out_value = 1;
      return iree_ok_status();
    }
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND, "unknown lrrt HAL device query '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
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

static void
lrrt_iree_hal_dispatch_record_reset(lrrt_iree_hal_dispatch_record_t *record,
                                    iree_allocator_t host_allocator) {
  if (!record) {
    return;
  }
  for (iree_host_size_t i = 0; i < record->binding_count; ++i) {
    if (record->bindings[i].buffer) {
      iree_hal_buffer_release(record->bindings[i].buffer);
    }
  }
  iree_allocator_free(host_allocator, record->bindings);
  record->bindings = NULL;
  record->binding_count = 0;
  if (record->executable) {
    iree_hal_executable_release(record->executable);
    record->executable = NULL;
  }
}

static void lrrt_iree_hal_buffer_ref_retain(iree_hal_buffer_ref_t ref) {
  if (ref.buffer) {
    iree_hal_buffer_retain(ref.buffer);
  }
}

static void lrrt_iree_hal_buffer_ref_release(iree_hal_buffer_ref_t ref) {
  if (ref.buffer) {
    iree_hal_buffer_release(ref.buffer);
  }
}

static void
lrrt_iree_hal_update_record_reset(lrrt_iree_hal_update_record_t *record,
                                  iree_allocator_t host_allocator) {
  if (!record) {
    return;
  }
  lrrt_iree_hal_buffer_ref_release(record->target_ref);
  iree_allocator_free(host_allocator, record->source_buffer);
  memset(record, 0, sizeof(*record));
}

static void
lrrt_iree_hal_copy_record_reset(lrrt_iree_hal_copy_record_t *record) {
  if (!record) {
    return;
  }
  lrrt_iree_hal_buffer_ref_release(record->source_ref);
  lrrt_iree_hal_buffer_ref_release(record->target_ref);
  memset(record, 0, sizeof(*record));
}

static void
lrrt_iree_hal_fill_record_reset(lrrt_iree_hal_fill_record_t *record) {
  if (!record) {
    return;
  }
  lrrt_iree_hal_buffer_ref_release(record->target_ref);
  memset(record, 0, sizeof(*record));
}

static void
lrrt_iree_hal_command_record_reset(lrrt_iree_hal_command_record_t *record,
                                   iree_allocator_t host_allocator) {
  if (!record) {
    return;
  }
  switch (record->kind) {
  case LRRT_IREE_HAL_COMMAND_UPDATE:
    lrrt_iree_hal_update_record_reset(&record->payload.update, host_allocator);
    break;
  case LRRT_IREE_HAL_COMMAND_COPY:
    lrrt_iree_hal_copy_record_reset(&record->payload.copy);
    break;
  case LRRT_IREE_HAL_COMMAND_FILL:
    lrrt_iree_hal_fill_record_reset(&record->payload.fill);
    break;
  case LRRT_IREE_HAL_COMMAND_DISPATCH:
    lrrt_iree_hal_dispatch_record_reset(&record->payload.dispatch,
                                        host_allocator);
    break;
  }
  memset(record, 0, sizeof(*record));
}

static iree_status_t lrrt_iree_hal_command_buffer_append(
    lrrt_iree_hal_command_buffer_t *command_buffer,
    lrrt_iree_hal_command_record_t **out_record) {
  IREE_ASSERT_ARGUMENT(out_record);
  *out_record = NULL;
  if (command_buffer->command_count == command_buffer->command_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        command_buffer->host_allocator, command_buffer->command_count + 1,
        sizeof(*command_buffer->commands), &command_buffer->command_capacity,
        (void **)&command_buffer->commands));
  }
  lrrt_iree_hal_command_record_t *record =
      &command_buffer->commands[command_buffer->command_count];
  memset(record, 0, sizeof(*record));
  ++command_buffer->command_count;
  *out_record = record;
  return iree_ok_status();
}

static iree_status_t
lrrt_iree_hal_semaphore_create(uint64_t initial_value,
                               iree_allocator_t host_allocator,
                               iree_hal_semaphore_t **out_semaphore) {
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;

  lrrt_iree_hal_semaphore_t *semaphore = NULL;
  iree_host_size_t frontier_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_async_semaphore_layout(sizeof(*semaphore), /*frontier_capacity=*/0,
                                  &frontier_offset, &total_size));
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void **)&semaphore));
  memset(semaphore, 0, total_size);

  iree_atomic_ref_count_init(&semaphore->async.ref_count);
  semaphore->async.vtable =
      (const iree_async_semaphore_vtable_t *)&lrrt_iree_hal_semaphore_vtable;
  semaphore->async.proactor = NULL;
  iree_atomic_store(&semaphore->async.timeline_value, (int64_t)initial_value,
                    iree_memory_order_release);
  iree_atomic_store(&semaphore->async.last_untainted_value,
                    (int64_t)initial_value, iree_memory_order_release);
  iree_slim_mutex_initialize(&semaphore->async.mutex);
  iree_atomic_store(&semaphore->async.failure_status, 0,
                    iree_memory_order_release);
  semaphore->async.timepoints_head = NULL;
  semaphore->async.frontier_capacity = 0;
  semaphore->async.frontier =
      (iree_async_frontier_t *)((uint8_t *)semaphore + frontier_offset);
  iree_async_frontier_initialize(semaphore->async.frontier, 0);
  semaphore->host_allocator = host_allocator;

  *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);
  lrrt_iree_hal_tracef("semaphore create initial_value=%" PRIu64,
                       initial_value);
  return iree_ok_status();
}

static void
lrrt_iree_hal_semaphore_destroy(iree_async_semaphore_t *base_semaphore) {
  lrrt_iree_hal_semaphore_t *semaphore =
      lrrt_iree_hal_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));
  lrrt_iree_hal_tracef("semaphore destroy current_value=%" PRIu64,
                       lrrt_iree_hal_semaphore_query(base_semaphore));
  iree_allocator_t host_allocator = semaphore->host_allocator;
  iree_async_semaphore_deinitialize(&semaphore->async);
  iree_allocator_free(host_allocator, semaphore);
}

static uint64_t
lrrt_iree_hal_semaphore_query(iree_async_semaphore_t *base_semaphore) {
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &base_semaphore->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    return iree_hal_status_as_semaphore_failure(failure);
  }
  return (uint64_t)iree_atomic_load(&base_semaphore->timeline_value,
                                    iree_memory_order_acquire);
}

static iree_status_t
lrrt_iree_hal_semaphore_signal(iree_async_semaphore_t *base_semaphore,
                               uint64_t new_value,
                               const iree_async_frontier_t *frontier) {
  lrrt_iree_hal_tracef("semaphore signal value=%" PRIu64, new_value);
  iree_status_t status = iree_async_semaphore_advance_timeline(
      base_semaphore, new_value, frontier);
  if (iree_status_is_ok(status)) {
    iree_async_semaphore_dispatch_timepoints(base_semaphore, new_value);
    return iree_ok_status();
  }

  uint64_t current_value = 0;
  iree_status_t query_status = iree_hal_semaphore_query(
      iree_hal_semaphore_cast(base_semaphore), &current_value);
  if (iree_status_is_ok(query_status) && current_value >= new_value) {
    iree_status_ignore(status);
    return iree_ok_status();
  }
  iree_status_ignore(query_status);
  return status;
}

static iree_status_t
lrrt_iree_hal_semaphore_wait(iree_hal_semaphore_t *base_semaphore,
                             uint64_t value, iree_timeout_t timeout,
                             iree_async_wait_flags_t flags) {
  lrrt_iree_hal_tracef("semaphore wait value=%" PRIu64, value);
  iree_status_t status = iree_async_semaphore_multi_wait(
      IREE_ASYNC_WAIT_MODE_ALL, (iree_async_semaphore_t **)&base_semaphore,
      &value, 1, timeout, flags, iree_allocator_system());
  lrrt_iree_hal_trace_status("semaphore wait", status);
  return status;
}

static iree_status_t lrrt_iree_hal_semaphore_import_timepoint(
    iree_hal_semaphore_t *base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  (void)base_semaphore;
  (void)value;
  (void)queue_affinity;
  (void)external_timepoint;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "lrrt HAL semaphore timepoint import is not "
                          "implemented yet");
}

static iree_status_t lrrt_iree_hal_semaphore_export_timepoint(
    iree_hal_semaphore_t *base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t *IREE_RESTRICT out_external_timepoint) {
  (void)base_semaphore;
  (void)value;
  (void)queue_affinity;
  (void)requested_type;
  (void)requested_flags;
  IREE_ASSERT_ARGUMENT(out_external_timepoint);
  memset(out_external_timepoint, 0, sizeof(*out_external_timepoint));
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "lrrt HAL semaphore timepoint export is not "
                          "implemented yet");
}

static bool
lrrt_iree_hal_semaphore_is_lrrt(iree_hal_semaphore_t *base_semaphore) {
  return iree_hal_resource_is(base_semaphore, &lrrt_iree_hal_semaphore_vtable);
}

static iree_status_t lrrt_iree_hal_device_validate_semaphore_list(
    const char *operation, iree_hal_semaphore_list_t semaphore_list) {
  if (iree_hal_semaphore_list_is_empty(semaphore_list)) {
    return iree_ok_status();
  }
  if (!semaphore_list.semaphores || !semaphore_list.payload_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "lrrt HAL device %s semaphore list is malformed",
                            operation);
  }
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    if (!lrrt_iree_hal_semaphore_is_lrrt(semaphore_list.semaphores[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "lrrt HAL device %s requires lrrt-owned semaphores", operation);
    }
  }
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_device_wait_semaphore_list(
    const char *operation, iree_hal_semaphore_list_t semaphore_list) {
  IREE_RETURN_IF_ERROR(
      lrrt_iree_hal_device_validate_semaphore_list(operation, semaphore_list));
  lrrt_iree_hal_tracef("%s wait semaphores=%" PRIhsz, operation,
                       semaphore_list.count);
  return iree_hal_semaphore_list_wait(semaphore_list, iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t lrrt_iree_hal_device_finish_queue_operation(
    const char *operation, iree_hal_semaphore_list_t signal_semaphore_list,
    iree_status_t status) {
  iree_status_t signal_list_status =
      lrrt_iree_hal_device_validate_semaphore_list(operation,
                                                   signal_semaphore_list);
  if (!iree_status_is_ok(signal_list_status)) {
    iree_status_ignore(status);
    return signal_list_status;
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
    lrrt_iree_hal_tracef("%s fail semaphores=%" PRIhsz, operation,
                         signal_semaphore_list.count);
    return status;
  }
  lrrt_iree_hal_tracef("%s signal semaphores=%" PRIhsz, operation,
                       signal_semaphore_list.count);
  return iree_hal_semaphore_list_signal(signal_semaphore_list, NULL);
}

static iree_status_t lrrt_iree_hal_device_begin_synchronous_submission(
    const char *operation, iree_hal_semaphore_list_t wait_semaphore_list) {
  lrrt_iree_hal_tracef("%s begin", operation);
  return lrrt_iree_hal_device_wait_semaphore_list(operation,
                                                  wait_semaphore_list);
}

static iree_status_t lrrt_iree_hal_device_end_synchronous_submission(
    const char *operation, iree_hal_semaphore_list_t signal_semaphore_list,
    iree_status_t status) {
  iree_status_t final_status = lrrt_iree_hal_device_finish_queue_operation(
      operation, signal_semaphore_list, status);
  lrrt_iree_hal_trace_status(operation, final_status);
  return final_status;
}

static const iree_hal_semaphore_vtable_t lrrt_iree_hal_semaphore_vtable = {
    .async =
        {
            .destroy = lrrt_iree_hal_semaphore_destroy,
            .query = lrrt_iree_hal_semaphore_query,
            .signal = lrrt_iree_hal_semaphore_signal,
        },
    .wait = lrrt_iree_hal_semaphore_wait,
    .import_timepoint = lrrt_iree_hal_semaphore_import_timepoint,
    .export_timepoint = lrrt_iree_hal_semaphore_export_timepoint,
};

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
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  lrrt_iree_hal_device_t *lrrt_device = lrrt_iree_hal_device_cast(device);
  const iree_host_size_t validation_size =
      iree_hal_command_buffer_validation_state_size(mode, binding_capacity);
  const iree_host_size_t total_size =
      iree_sizeof_struct(lrrt_iree_hal_command_buffer_t) + validation_size;
  lrrt_iree_hal_command_buffer_t *command_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      lrrt_device->host_allocator, total_size, (void **)&command_buffer));
  void *validation_state =
      validation_size
          ? (uint8_t *)command_buffer + iree_sizeof_struct(*command_buffer)
          : NULL;
  iree_hal_command_buffer_initialize(
      lrrt_device->device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, validation_state, &lrrt_iree_hal_command_buffer_vtable,
      &command_buffer->base);
  command_buffer->host_allocator = lrrt_device->host_allocator;
  command_buffer->command_count = 0;
  command_buffer->command_capacity = 0;
  command_buffer->commands = NULL;

  *out_command_buffer = &command_buffer->base;
  lrrt_iree_hal_tracef("command buffer create mode=0x%x categories=0x%x "
                       "binding_capacity=%" PRIhsz,
                       (uint32_t)mode, (uint32_t)command_categories,
                       binding_capacity);
  return iree_ok_status();
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
  lrrt_iree_hal_allocator_t *allocator =
      lrrt_iree_hal_allocator_cast(lrrt_device->device_allocator);
  lrrt_iree_hal_tracef("device create executable cache id='%.*s'",
                       (int)identifier.size, identifier.data);
  return lrrt_iree_hal_executable_cache_create(
      identifier, lrrt_device->host_allocator, allocator->device,
      out_executable_cache);
}

static iree_status_t lrrt_iree_hal_device_import_file(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t *handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t **out_file) {
  (void)flags;
  IREE_ASSERT_ARGUMENT(out_file);
  lrrt_iree_hal_device_t *lrrt_device = lrrt_iree_hal_device_cast(device);
  lrrt_iree_hal_tracef("device import file access=0x%" PRIx64,
                       (uint64_t)access);
  return iree_hal_file_from_handle(
      lrrt_device->device_allocator, queue_affinity, access, handle,
      /*proactor=*/NULL, lrrt_device->host_allocator, out_file);
}

static iree_status_t lrrt_iree_hal_device_create_semaphore(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t **out_semaphore) {
  lrrt_iree_hal_device_t *lrrt_device = lrrt_iree_hal_device_cast(device);
  (void)queue_affinity;
  (void)flags;
  IREE_ASSERT_ARGUMENT(out_semaphore);
  lrrt_iree_hal_tracef("device create semaphore initial_value=%" PRIu64,
                       initial_value);
  return lrrt_iree_hal_semaphore_create(
      initial_value, lrrt_device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
lrrt_iree_hal_device_query_semaphore_compatibility(
    iree_hal_device_t *device, iree_hal_semaphore_t *semaphore) {
  (void)device;
  if (!lrrt_iree_hal_semaphore_is_lrrt(semaphore)) {
    return IREE_HAL_SEMAPHORE_COMPATIBILITY_NONE;
  }
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL;
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
  (void)queue_affinity;
  (void)flags;
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  if (pool) {
    return LRRT_IREE_HAL_DEVICE_UNIMPLEMENTED("queue allocation pool");
  }

  IREE_RETURN_IF_ERROR(lrrt_iree_hal_device_begin_synchronous_submission(
      "queue allocation", wait_semaphore_list));
  lrrt_iree_hal_tracef("queue allocation size=%" PRIu64,
                       (uint64_t)allocation_size);

  iree_hal_buffer_params_t local_params = params;
  local_params.queue_affinity = queue_affinity;
  local_params.type &= ~IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  local_params.type |= IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  local_params.usage &= ~IREE_HAL_BUFFER_USAGE_MAPPING;

  iree_status_t status = iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), local_params, allocation_size,
      out_buffer);
  return lrrt_iree_hal_device_end_synchronous_submission(
      "queue allocation", signal_semaphore_list, status);
}

static iree_status_t lrrt_iree_hal_device_queue_dealloca(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t *buffer, iree_hal_dealloca_flags_t flags) {
  (void)queue_affinity;
  (void)flags;
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_device_begin_synchronous_submission(
      "queue deallocation", wait_semaphore_list));
  iree_status_t status = iree_ok_status();
  if (buffer) {
    iree_hal_allocator_deallocate_buffer(iree_hal_device_allocator(device),
                                         buffer);
  }
  return lrrt_iree_hal_device_end_synchronous_submission(
      "queue deallocation", signal_semaphore_list, status);
}

static iree_status_t lrrt_iree_hal_device_fill_now(
    iree_hal_device_t *device, iree_hal_buffer_t *target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const void *pattern, iree_host_size_t pattern_length) {
  if (length == 0) {
    return iree_ok_status();
  }
  if (!pattern ||
      (pattern_length != 1 && pattern_length != 2 && pattern_length != 4)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue fill pattern length must be 1, 2, or 4");
  }
  if ((target_offset % pattern_length) != 0 || (length % pattern_length) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue fill range is not pattern-aligned");
  }
  if ((iree_hal_buffer_allowed_usage(target_buffer) &
       IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET) == 0) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "queue fill target buffer lacks transfer-target "
                            "usage");
  }

  void *target_device_ptr = NULL;
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_buffer_device_range(
      target_buffer, target_offset, length, &target_device_ptr));

  lrrt_iree_hal_device_t *lrrt_device = lrrt_iree_hal_device_cast(device);
  void *scratch = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      lrrt_device->host_allocator, (iree_host_size_t)length, &scratch));
  uint8_t *bytes = (uint8_t *)scratch;
  for (iree_device_size_t offset = 0; offset < length;
       offset += pattern_length) {
    memcpy(bytes + offset, pattern, pattern_length);
  }

  lrrt_iree_hal_allocator_t *allocator =
      lrrt_iree_hal_allocator_cast(lrrt_device->device_allocator);
  lr_status_t lr_status =
      lr_memcpy(allocator->device, target_device_ptr, scratch, (size_t)length,
                LR_MEMCPY_HOST_TO_DEVICE);
  iree_allocator_free(lrrt_device->host_allocator, scratch);
  return lrrt_iree_hal_status_from_lr(lr_status, "lr_memcpy");
}

static iree_status_t lrrt_iree_hal_device_queue_fill(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void *pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  (void)queue_affinity;
  (void)flags;
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_device_begin_synchronous_submission(
      "queue fill", wait_semaphore_list));
  lrrt_iree_hal_tracef("queue fill target_offset=%" PRIu64 " length=%" PRIu64
                       " pattern_length=%" PRIhsz,
                       (uint64_t)target_offset, (uint64_t)length,
                       pattern_length);
  iree_status_t status = lrrt_iree_hal_device_fill_now(
      device, target_buffer, target_offset, length, pattern, pattern_length);
  return lrrt_iree_hal_device_end_synchronous_submission(
      "queue fill", signal_semaphore_list, status);
}

static iree_status_t lrrt_iree_hal_device_update_now(
    iree_hal_device_t *device, const void *source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t *target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  if (length == 0) {
    return iree_ok_status();
  }
  if ((iree_hal_buffer_allowed_usage(target_buffer) &
       IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET) == 0) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "queue update target buffer lacks "
                            "transfer-target usage");
  }

  void *target_device_ptr = NULL;
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_buffer_device_range(
      target_buffer, target_offset, length, &target_device_ptr));
  const uint8_t *source =
      (const uint8_t *)source_buffer + (iree_host_size_t)source_offset;
  lrrt_iree_hal_device_t *lrrt_device = lrrt_iree_hal_device_cast(device);
  lrrt_iree_hal_allocator_t *allocator =
      lrrt_iree_hal_allocator_cast(lrrt_device->device_allocator);
  lr_status_t lr_status =
      lr_memcpy(allocator->device, target_device_ptr, source, (size_t)length,
                LR_MEMCPY_HOST_TO_DEVICE);
  return lrrt_iree_hal_status_from_lr(lr_status, "lr_memcpy");
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
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_device_begin_synchronous_submission(
      "queue update", wait_semaphore_list));
  lrrt_iree_hal_tracef("queue update source_offset=%" PRIhsz
                       " target_offset=%" PRIu64 " length=%" PRIu64,
                       source_offset, (uint64_t)target_offset,
                       (uint64_t)length);
  iree_status_t status =
      lrrt_iree_hal_device_update_now(device, source_buffer, source_offset,
                                      target_buffer, target_offset, length);
  return lrrt_iree_hal_device_end_synchronous_submission(
      "queue update", signal_semaphore_list, status);
}

static iree_status_t lrrt_iree_hal_device_copy_now(
    iree_hal_device_t *device, iree_hal_buffer_t *source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t *target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
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
  lrrt_iree_hal_device_t *lrrt_device = lrrt_iree_hal_device_cast(device);
  lrrt_iree_hal_allocator_t *allocator =
      lrrt_iree_hal_allocator_cast(lrrt_device->device_allocator);
  lr_status_t lr_status =
      lr_memcpy(allocator->device, target_device_ptr, source_device_ptr,
                (size_t)length, LR_MEMCPY_DEVICE_TO_DEVICE);
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
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_device_begin_synchronous_submission(
      "queue copy", wait_semaphore_list));
  lrrt_iree_hal_tracef("queue copy source_offset=%" PRIu64
                       " target_offset=%" PRIu64 " length=%" PRIu64,
                       (uint64_t)source_offset, (uint64_t)target_offset,
                       (uint64_t)length);
  iree_status_t status =
      lrrt_iree_hal_device_copy_now(device, source_buffer, source_offset,
                                    target_buffer, target_offset, length);
  return lrrt_iree_hal_device_end_synchronous_submission(
      "queue copy", signal_semaphore_list, status);
}

static iree_status_t lrrt_iree_hal_device_queue_read(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t *source_file, uint64_t source_offset,
    iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  (void)queue_affinity;
  (void)flags;
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_device_begin_synchronous_submission(
      "queue read", wait_semaphore_list));
  lrrt_iree_hal_tracef("queue read source_offset=%" PRIu64
                       " target_offset=%" PRIu64 " length=%" PRIu64,
                       source_offset, (uint64_t)target_offset,
                       (uint64_t)length);

  iree_status_t status = iree_ok_status();
  if (length != 0) {
    status =
        iree_hal_file_validate_access(source_file, IREE_HAL_MEMORY_ACCESS_READ);
  }
  if (iree_status_is_ok(status) && length != 0 &&
      (iree_hal_buffer_allowed_usage(target_buffer) &
       IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET) == 0) {
    status = iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                              "queue read target buffer lacks "
                              "transfer-target usage");
  }
  if (iree_status_is_ok(status) && length != 0 &&
      length > UINT64_MAX - source_offset) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "queue read file range overflows uint64");
  }
  if (iree_status_is_ok(status) && length != 0) {
    uint64_t file_length = iree_hal_file_length(source_file);
    if (file_length > 0 && source_offset + length > file_length) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "queue read range [%" PRIu64 ", %" PRIu64
                                ") exceeds file length %" PRIu64,
                                source_offset, source_offset + (uint64_t)length,
                                file_length);
    }
  }
  if (iree_status_is_ok(status) && length != 0) {
    iree_hal_buffer_t *storage_buffer =
        iree_hal_file_storage_buffer(source_file);
    if (storage_buffer) {
      status = lrrt_iree_hal_device_copy_now(
          device, storage_buffer, (iree_device_size_t)source_offset,
          target_buffer, target_offset, length);
    } else if (!iree_hal_file_supports_synchronous_io(source_file)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "queue read source file does not support "
                                "synchronous I/O");
    } else {
      status = iree_hal_file_read(source_file, source_offset, target_buffer,
                                  target_offset, length);
    }
  }
  return lrrt_iree_hal_device_end_synchronous_submission(
      "queue read", signal_semaphore_list, status);
}

static iree_status_t lrrt_iree_hal_device_queue_write(
    iree_hal_device_t *device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t *source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t *target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  (void)queue_affinity;
  (void)flags;
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_device_begin_synchronous_submission(
      "queue write", wait_semaphore_list));
  lrrt_iree_hal_tracef("queue write source_offset=%" PRIu64
                       " target_offset=%" PRIu64 " length=%" PRIu64,
                       (uint64_t)source_offset, target_offset,
                       (uint64_t)length);

  iree_status_t status = iree_ok_status();
  if (length != 0) {
    status = iree_hal_file_validate_access(target_file,
                                           IREE_HAL_MEMORY_ACCESS_WRITE);
  }
  if (iree_status_is_ok(status) && length != 0 &&
      (iree_hal_buffer_allowed_usage(source_buffer) &
       IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE) == 0) {
    status = iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                              "queue write source buffer lacks "
                              "transfer-source usage");
  }
  if (iree_status_is_ok(status) && length != 0 &&
      length > UINT64_MAX - target_offset) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "queue write file range overflows uint64");
  }
  if (iree_status_is_ok(status) && length != 0) {
    iree_hal_buffer_t *storage_buffer =
        iree_hal_file_storage_buffer(target_file);
    if (storage_buffer) {
      status = lrrt_iree_hal_device_copy_now(
          device, source_buffer, source_offset, storage_buffer,
          (iree_device_size_t)target_offset, length);
    } else if (!iree_hal_file_supports_synchronous_io(target_file)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "queue write target file does not support "
                                "synchronous I/O");
    } else {
      status = iree_hal_file_write(target_file, target_offset, source_buffer,
                                   source_offset, length);
    }
  }
  return lrrt_iree_hal_device_end_synchronous_submission(
      "queue write", signal_semaphore_list, status);
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

static iree_status_t lrrt_iree_hal_device_dispatch_now(
    iree_hal_executable_t *executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_dispatch_flags_t flags) {
  if (constants.data_length != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "lrrt HAL dispatch does not support inline "
                            "constants yet");
  }
  if (flags & (IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS |
               IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS |
               IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS |
               IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_ARGUMENTS |
               IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_ARGUMENTS)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "lrrt HAL dispatch only supports direct static arguments");
  }
  if (config.workgroup_count_ref.buffer) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "lrrt HAL dispatch does not support indirect "
                            "workgroup counts yet");
  }
  if (config.workgroup_count[0] == 0 || config.workgroup_count[1] == 0 ||
      config.workgroup_count[2] == 0) {
    return iree_ok_status();
  }
  iree_hal_dispatch_config_t resolved_config = config;
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_executable_resolve_workgroup_size(
      executable, function, &resolved_config, "dispatch"));
  if (bindings.count > LRRT_IREE_HAL_MAX_INLINE_BINDINGS) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "lrrt HAL dispatch binding count %" PRIhsz
                            " exceeds inline limit %d",
                            bindings.count, LRRT_IREE_HAL_MAX_INLINE_BINDINGS);
  }
  if (bindings.count != 0 && !bindings.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "lrrt HAL dispatch binding list is null");
  }

  lr_kernel_t *kernel = NULL;
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_executable_kernel_for_function(
      executable, function, &kernel));
  const lrrt_iree_hal_executable_t *lrrt_executable =
      lrrt_iree_hal_executable_cast(executable);
  const uint32_t function_index = iree_hal_executable_function_index(function);
  const lrrt_iree_hal_executable_function_t *function_state =
      &lrrt_executable->functions[function_index];

  void *kernargs[LRRT_IREE_HAL_MAX_INLINE_BINDINGS] = {0};
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    iree_hal_buffer_ref_t binding = {0};
    IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
        binding_table, bindings.values[i], &binding));
    if (!binding.buffer) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "lrrt HAL dispatch binding %" PRIhsz " resolved to null buffer", i);
    }
    if ((iree_hal_buffer_allowed_usage(binding.buffer) &
         IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE) == 0) {
      return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                              "lrrt HAL dispatch binding %" PRIhsz
                              " lacks dispatch-storage usage",
                              i);
    }
    IREE_RETURN_IF_ERROR(
        lrrt_iree_hal_buffer_ref_device_range(binding, &kernargs[i]));
  }

  const lr_launch_config_t launch_config = {
      {resolved_config.workgroup_count[0] * resolved_config.workgroup_size[0],
       resolved_config.workgroup_count[1] * resolved_config.workgroup_size[1],
       resolved_config.workgroup_count[2] * resolved_config.workgroup_size[2]},
      {resolved_config.workgroup_size[0], resolved_config.workgroup_size[1],
       resolved_config.workgroup_size[2]},
      resolved_config.dynamic_workgroup_local_memory};
  lrrt_iree_hal_tracef(
      "dispatch launch function='%.*s' wg_count=%ux%ux%u wg_size=%ux%ux%u "
      "bindings=%" PRIhsz " local_memory=%u",
      (int)function_state->name.size, function_state->name.data,
      resolved_config.workgroup_count[0], resolved_config.workgroup_count[1],
      resolved_config.workgroup_count[2], resolved_config.workgroup_size[0],
      resolved_config.workgroup_size[1], resolved_config.workgroup_size[2],
      bindings.count, resolved_config.dynamic_workgroup_local_memory);
  return lrrt_iree_hal_status_from_lr(
      lr_launch(kernel, &launch_config, kernargs,
                bindings.count * sizeof(kernargs[0])),
      "lr_launch");
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
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_device_begin_synchronous_submission(
      "queue dispatch", wait_semaphore_list));
  lrrt_iree_hal_tracef("queue dispatch bindings=%" PRIhsz, bindings.count);
  iree_status_t status = lrrt_iree_hal_device_dispatch_now(
      executable, function, config, constants, bindings,
      iree_hal_buffer_binding_table_empty(), flags);
  return lrrt_iree_hal_device_end_synchronous_submission(
      "queue dispatch", signal_semaphore_list, status);
}

static iree_status_t lrrt_iree_hal_command_buffer_require_dispatch_shape(
    const char *operation, iree_hal_executable_t *executable,
    iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  if (constants.data_length != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "lrrt HAL command buffer %s does not support "
                            "inline constants yet",
                            operation);
  }
  if (flags & (IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS |
               IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS |
               IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS |
               IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_ARGUMENTS |
               IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_ARGUMENTS)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "lrrt HAL command buffer %s only supports direct "
                            "static arguments",
                            operation);
  }
  if (config.workgroup_count_ref.buffer) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "lrrt HAL command buffer %s does not support "
                            "indirect workgroup counts yet",
                            operation);
  }
  iree_hal_dispatch_config_t resolved_config = config;
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_executable_resolve_workgroup_size(
      executable, function, &resolved_config, operation));
  if (bindings.count > LRRT_IREE_HAL_MAX_INLINE_BINDINGS) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "lrrt HAL command buffer %s binding count %" PRIhsz
                            " exceeds inline limit %d",
                            operation, bindings.count,
                            LRRT_IREE_HAL_MAX_INLINE_BINDINGS);
  }
  if (bindings.count != 0 && !bindings.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "lrrt HAL command buffer %s binding list is null",
                            operation);
  }
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_command_buffer_record_buffer_ref(
    iree_hal_command_buffer_t *base_command_buffer, iree_hal_buffer_ref_t ref) {
  if (ref.buffer) {
    return iree_ok_status();
  }
  const uint32_t required_binding_count = ref.buffer_slot + 1;
  if (required_binding_count > base_command_buffer->binding_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "lrrt HAL command buffer indirect binding slot %u "
                            "exceeds capacity %u",
                            ref.buffer_slot,
                            base_command_buffer->binding_capacity);
  }
  if (required_binding_count > base_command_buffer->binding_count) {
    base_command_buffer->binding_count = required_binding_count;
  }
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_command_buffer_resolve_buffer_ref(
    iree_hal_buffer_binding_table_t binding_table, iree_hal_buffer_ref_t ref,
    iree_hal_buffer_ref_t *out_ref) {
  IREE_ASSERT_ARGUMENT(out_ref);
  *out_ref = (iree_hal_buffer_ref_t){0};
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_binding_table_resolve_ref(binding_table, ref, out_ref));
  if (!out_ref->buffer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "lrrt HAL command buffer buffer reference resolved to null buffer");
  }
  return iree_ok_status();
}

static void lrrt_iree_hal_command_buffer_destroy(
    iree_hal_command_buffer_t *base_command_buffer) {
  lrrt_iree_hal_command_buffer_t *command_buffer =
      lrrt_iree_hal_command_buffer_cast(base_command_buffer);
  lrrt_iree_hal_tracef("command buffer destroy commands=%" PRIhsz,
                       command_buffer->command_count);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  for (iree_host_size_t i = 0; i < command_buffer->command_count; ++i) {
    lrrt_iree_hal_command_record_reset(&command_buffer->commands[i],
                                       host_allocator);
  }
  iree_allocator_free(host_allocator, command_buffer->commands);
  iree_allocator_free(host_allocator, command_buffer);
}

static iree_status_t lrrt_iree_hal_command_buffer_begin(
    iree_hal_command_buffer_t *base_command_buffer) {
  lrrt_iree_hal_command_buffer_t *command_buffer =
      lrrt_iree_hal_command_buffer_cast(base_command_buffer);
  for (iree_host_size_t i = 0; i < command_buffer->command_count; ++i) {
    lrrt_iree_hal_command_record_reset(&command_buffer->commands[i],
                                       command_buffer->host_allocator);
  }
  command_buffer->command_count = 0;
  base_command_buffer->binding_count = 0;
  lrrt_iree_hal_tracef("command buffer begin");
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_command_buffer_end(
    iree_hal_command_buffer_t *base_command_buffer) {
  lrrt_iree_hal_command_buffer_t *command_buffer =
      lrrt_iree_hal_command_buffer_cast(base_command_buffer);
  lrrt_iree_hal_tracef("command buffer end commands=%" PRIhsz,
                       command_buffer->command_count);
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t *base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t *location) {
  (void)base_command_buffer;
  (void)label;
  (void)label_color;
  (void)location;
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_command_buffer_end_debug_group(
    iree_hal_command_buffer_t *base_command_buffer) {
  (void)base_command_buffer;
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_command_buffer_execution_barrier(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t *memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t *buffer_barriers) {
  (void)base_command_buffer;
  (void)source_stage_mask;
  (void)target_stage_mask;
  (void)flags;
  (void)memory_barrier_count;
  (void)memory_barriers;
  (void)buffer_barrier_count;
  (void)buffer_barriers;
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_command_buffer_signal_event(
    iree_hal_command_buffer_t *base_command_buffer, iree_hal_event_t *event,
    iree_hal_execution_stage_t source_stage_mask) {
  (void)base_command_buffer;
  (void)event;
  (void)source_stage_mask;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "lrrt HAL command buffer events are not implemented");
}

static iree_status_t lrrt_iree_hal_command_buffer_reset_event(
    iree_hal_command_buffer_t *base_command_buffer, iree_hal_event_t *event,
    iree_hal_execution_stage_t source_stage_mask) {
  (void)base_command_buffer;
  (void)event;
  (void)source_stage_mask;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "lrrt HAL command buffer events are not implemented");
}

static iree_status_t lrrt_iree_hal_command_buffer_wait_events(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t **events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t *memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t *buffer_barriers) {
  (void)base_command_buffer;
  (void)event_count;
  (void)events;
  (void)source_stage_mask;
  (void)target_stage_mask;
  (void)memory_barrier_count;
  (void)memory_barriers;
  (void)buffer_barrier_count;
  (void)buffer_barriers;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "lrrt HAL command buffer events are not implemented");
}

static iree_status_t lrrt_iree_hal_command_buffer_advise_buffer(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  (void)base_command_buffer;
  (void)buffer_ref;
  (void)flags;
  (void)arg0;
  (void)arg1;
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_command_buffer_fill_buffer(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void *pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  if (!pattern ||
      pattern_length > sizeof(((lrrt_iree_hal_fill_record_t *)0)->pattern)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "lrrt HAL command buffer fill pattern is invalid");
  }
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_command_buffer_record_buffer_ref(
      base_command_buffer, target_ref));
  lrrt_iree_hal_command_buffer_t *command_buffer =
      lrrt_iree_hal_command_buffer_cast(base_command_buffer);
  lrrt_iree_hal_command_record_t *command = NULL;
  IREE_RETURN_IF_ERROR(
      lrrt_iree_hal_command_buffer_append(command_buffer, &command));
  command->kind = LRRT_IREE_HAL_COMMAND_FILL;
  lrrt_iree_hal_fill_record_t *record = &command->payload.fill;
  memcpy(record->pattern, pattern, pattern_length);
  record->pattern_length = pattern_length;
  record->target_ref = target_ref;
  record->flags = flags;
  lrrt_iree_hal_buffer_ref_retain(record->target_ref);
  lrrt_iree_hal_tracef("command buffer record fill length=%" PRIu64,
                       (uint64_t)target_ref.length);
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_command_buffer_update_buffer(
    iree_hal_command_buffer_t *base_command_buffer, const void *source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  if (target_ref.length != 0 && !source_buffer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "lrrt HAL command buffer update source buffer is null");
  }
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_command_buffer_record_buffer_ref(
      base_command_buffer, target_ref));
  lrrt_iree_hal_command_buffer_t *command_buffer =
      lrrt_iree_hal_command_buffer_cast(base_command_buffer);
  lrrt_iree_hal_command_record_t *command = NULL;
  IREE_RETURN_IF_ERROR(
      lrrt_iree_hal_command_buffer_append(command_buffer, &command));
  command->kind = LRRT_IREE_HAL_COMMAND_UPDATE;
  lrrt_iree_hal_update_record_t *record = &command->payload.update;
  record->target_ref = target_ref;
  record->flags = flags;
  if (target_ref.length != 0) {
    iree_status_t status = iree_allocator_malloc(
        command_buffer->host_allocator, (iree_host_size_t)target_ref.length,
        &record->source_buffer);
    if (!iree_status_is_ok(status)) {
      --command_buffer->command_count;
      memset(command, 0, sizeof(*command));
      return status;
    }
    memcpy(record->source_buffer,
           (const uint8_t *)source_buffer + source_offset,
           (iree_host_size_t)target_ref.length);
  }
  lrrt_iree_hal_buffer_ref_retain(record->target_ref);
  lrrt_iree_hal_tracef("command buffer record update length=%" PRIu64,
                       (uint64_t)target_ref.length);
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_command_buffer_copy_buffer(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_command_buffer_record_buffer_ref(
      base_command_buffer, source_ref));
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_command_buffer_record_buffer_ref(
      base_command_buffer, target_ref));
  lrrt_iree_hal_command_buffer_t *command_buffer =
      lrrt_iree_hal_command_buffer_cast(base_command_buffer);
  lrrt_iree_hal_command_record_t *command = NULL;
  IREE_RETURN_IF_ERROR(
      lrrt_iree_hal_command_buffer_append(command_buffer, &command));
  command->kind = LRRT_IREE_HAL_COMMAND_COPY;
  lrrt_iree_hal_copy_record_t *record = &command->payload.copy;
  record->source_ref = source_ref;
  record->target_ref = target_ref;
  record->flags = flags;
  lrrt_iree_hal_buffer_ref_retain(record->source_ref);
  lrrt_iree_hal_buffer_ref_retain(record->target_ref);
  lrrt_iree_hal_tracef("command buffer record copy length=%" PRIu64,
                       (uint64_t)target_ref.length);
  return iree_ok_status();
}

static iree_status_t lrrt_iree_hal_command_buffer_collective(
    iree_hal_command_buffer_t *base_command_buffer, iree_hal_channel_t *channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  (void)base_command_buffer;
  (void)channel;
  (void)op;
  (void)param;
  (void)send_ref;
  (void)recv_ref;
  (void)element_count;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "lrrt HAL command buffer collective is not implemented yet");
}

static iree_status_t lrrt_iree_hal_command_buffer_dispatch(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_executable_t *executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_command_buffer_require_dispatch_shape(
      "command buffer dispatch", executable, function, config, constants,
      bindings, flags));
  lrrt_iree_hal_command_buffer_t *command_buffer =
      lrrt_iree_hal_command_buffer_cast(base_command_buffer);

  lrrt_iree_hal_command_record_t *command = NULL;
  IREE_RETURN_IF_ERROR(
      lrrt_iree_hal_command_buffer_append(command_buffer, &command));
  command->kind = LRRT_IREE_HAL_COMMAND_DISPATCH;
  lrrt_iree_hal_dispatch_record_t *record = &command->payload.dispatch;
  if (bindings.count != 0) {
    iree_status_t status = iree_allocator_malloc(
        command_buffer->host_allocator,
        bindings.count * sizeof(*record->bindings), (void **)&record->bindings);
    if (!iree_status_is_ok(status)) {
      --command_buffer->command_count;
      memset(command, 0, sizeof(*command));
      return status;
    }
    memcpy(record->bindings, bindings.values,
           bindings.count * sizeof(*record->bindings));
  }
  record->executable = executable;
  record->function = function;
  record->config = config;
  record->flags = flags;
  record->binding_count = bindings.count;

  iree_hal_executable_retain(executable);
  for (iree_host_size_t i = 0; i < record->binding_count; ++i) {
    if (record->bindings[i].buffer) {
      iree_hal_buffer_retain(record->bindings[i].buffer);
    } else {
      iree_status_t status = lrrt_iree_hal_command_buffer_record_buffer_ref(
          base_command_buffer, record->bindings[i]);
      if (!iree_status_is_ok(status)) {
        lrrt_iree_hal_command_record_reset(command,
                                           command_buffer->host_allocator);
        --command_buffer->command_count;
        return status;
      }
    }
  }

  lrrt_iree_hal_tracef("command buffer record dispatch bindings=%" PRIhsz,
                       record->binding_count);
  return iree_ok_status();
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
  (void)flags;
  IREE_RETURN_IF_ERROR(lrrt_iree_hal_device_begin_synchronous_submission(
      "queue execute", wait_semaphore_list));
  if (!command_buffer) {
    return lrrt_iree_hal_device_end_synchronous_submission(
        "queue execute", signal_semaphore_list, iree_ok_status());
  }

  iree_status_t status = iree_hal_command_buffer_validate_submission(
      command_buffer, binding_table);
  if (iree_status_is_ok(status) &&
      !iree_hal_resource_is(&command_buffer->resource,
                            &lrrt_iree_hal_command_buffer_vtable)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "lrrt HAL queue execute requires an lrrt command buffer");
  }
  if (iree_status_is_ok(status)) {
    lrrt_iree_hal_command_buffer_t *lrrt_command_buffer =
        lrrt_iree_hal_command_buffer_cast(command_buffer);
    lrrt_iree_hal_tracef("queue execute commands=%" PRIhsz,
                         lrrt_command_buffer->command_count);
    for (iree_host_size_t i = 0; i < lrrt_command_buffer->command_count; ++i) {
      const lrrt_iree_hal_command_record_t *command =
          &lrrt_command_buffer->commands[i];
      switch (command->kind) {
      case LRRT_IREE_HAL_COMMAND_UPDATE: {
        const lrrt_iree_hal_update_record_t *record = &command->payload.update;
        lrrt_iree_hal_tracef("queue execute command[%" PRIhsz
                             "]=update length=%" PRIu64,
                             i, (uint64_t)record->target_ref.length);
        iree_hal_buffer_ref_t target_ref = {0};
        status = lrrt_iree_hal_command_buffer_resolve_buffer_ref(
            binding_table, record->target_ref, &target_ref);
        if (iree_status_is_ok(status)) {
          status = lrrt_iree_hal_device_update_now(
              device, record->source_buffer, record->source_offset,
              target_ref.buffer, target_ref.offset, target_ref.length);
        }
        break;
      }
      case LRRT_IREE_HAL_COMMAND_COPY: {
        const lrrt_iree_hal_copy_record_t *record = &command->payload.copy;
        lrrt_iree_hal_tracef("queue execute command[%" PRIhsz
                             "]=copy length=%" PRIu64,
                             i, (uint64_t)record->target_ref.length);
        iree_hal_buffer_ref_t source_ref = {0};
        iree_hal_buffer_ref_t target_ref = {0};
        status = lrrt_iree_hal_command_buffer_resolve_buffer_ref(
            binding_table, record->source_ref, &source_ref);
        if (iree_status_is_ok(status)) {
          status = lrrt_iree_hal_command_buffer_resolve_buffer_ref(
              binding_table, record->target_ref, &target_ref);
        }
        if (iree_status_is_ok(status)) {
          status = lrrt_iree_hal_device_copy_now(
              device, source_ref.buffer, source_ref.offset, target_ref.buffer,
              target_ref.offset, target_ref.length);
        }
        break;
      }
      case LRRT_IREE_HAL_COMMAND_FILL: {
        const lrrt_iree_hal_fill_record_t *record = &command->payload.fill;
        lrrt_iree_hal_tracef("queue execute command[%" PRIhsz
                             "]=fill length=%" PRIu64,
                             i, (uint64_t)record->target_ref.length);
        iree_hal_buffer_ref_t target_ref = {0};
        status = lrrt_iree_hal_command_buffer_resolve_buffer_ref(
            binding_table, record->target_ref, &target_ref);
        if (iree_status_is_ok(status)) {
          status = lrrt_iree_hal_device_fill_now(
              device, target_ref.buffer, target_ref.offset, target_ref.length,
              record->pattern, record->pattern_length);
        }
        break;
      }
      case LRRT_IREE_HAL_COMMAND_DISPATCH: {
        const lrrt_iree_hal_dispatch_record_t *record =
            &command->payload.dispatch;
        lrrt_iree_hal_tracef("queue execute command[%" PRIhsz
                             "]=dispatch bindings=%" PRIhsz,
                             i, record->binding_count);
        const iree_hal_buffer_ref_list_t bindings = {
            record->binding_count,
            record->bindings,
        };
        status = lrrt_iree_hal_device_dispatch_now(
            record->executable, record->function, record->config,
            iree_const_byte_span_empty(), bindings, binding_table,
            record->flags);
        break;
      }
      }
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
  }
  return lrrt_iree_hal_device_end_synchronous_submission(
      "queue execute", signal_semaphore_list, status);
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
  lrrt_iree_hal_tracef("driver query available devices count=1");
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
  lrrt_iree_hal_tracef("driver create device by id=%" PRIuPTR,
                       (uintptr_t)device_id);
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
  lrrt_iree_hal_tracef("driver create device by path='%.*s'",
                       (int)device_path.size, device_path.data);
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

static const iree_hal_executable_vtable_t lrrt_iree_hal_executable_vtable = {
    .destroy = lrrt_iree_hal_executable_destroy,
    .function_count = lrrt_iree_hal_executable_function_count,
    .function_info = lrrt_iree_hal_executable_function_info,
    .function_parameters = lrrt_iree_hal_executable_function_parameters,
    .lookup_function_by_name = lrrt_iree_hal_executable_lookup_function_by_name,
    .lookup_global_by_name = lrrt_iree_hal_executable_lookup_global_by_name,
};

static const iree_hal_command_buffer_vtable_t
    lrrt_iree_hal_command_buffer_vtable = {
        .destroy = lrrt_iree_hal_command_buffer_destroy,
        .begin = lrrt_iree_hal_command_buffer_begin,
        .end = lrrt_iree_hal_command_buffer_end,
        .begin_debug_group = lrrt_iree_hal_command_buffer_begin_debug_group,
        .end_debug_group = lrrt_iree_hal_command_buffer_end_debug_group,
        .execution_barrier = lrrt_iree_hal_command_buffer_execution_barrier,
        .signal_event = lrrt_iree_hal_command_buffer_signal_event,
        .reset_event = lrrt_iree_hal_command_buffer_reset_event,
        .wait_events = lrrt_iree_hal_command_buffer_wait_events,
        .advise_buffer = lrrt_iree_hal_command_buffer_advise_buffer,
        .fill_buffer = lrrt_iree_hal_command_buffer_fill_buffer,
        .update_buffer = lrrt_iree_hal_command_buffer_update_buffer,
        .copy_buffer = lrrt_iree_hal_command_buffer_copy_buffer,
        .collective = lrrt_iree_hal_command_buffer_collective,
        .dispatch = lrrt_iree_hal_command_buffer_dispatch,
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
  lrrt_iree_hal_tracef("driver factory enumerate count=%" PRIhsz,
                       *out_driver_info_count);
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
  lrrt_iree_hal_tracef("driver factory create name='%.*s'",
                       (int)driver_name.size, driver_name.data);
  return lrrt_iree_hal_driver_create(driver_name, host_allocator, out_driver);
}

iree_status_t
lrrt_iree_hal_driver_module_register(iree_hal_driver_registry_t *registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = lrrt_iree_hal_driver_factory_enumerate,
      .try_create = lrrt_iree_hal_driver_factory_try_create,
  };
  lrrt_iree_hal_tracef("driver module register");
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
