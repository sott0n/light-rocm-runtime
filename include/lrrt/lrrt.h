#ifndef LRRT_LRRT_H_
#define LRRT_LRRT_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define LRRT_API __declspec(dllexport)
#else
#define LRRT_API __attribute__((visibility("default")))
#endif

typedef enum lr_status_t {
  LR_SUCCESS = 0,
  LR_ERROR_INVALID_ARGUMENT = 1,
  LR_ERROR_NOT_INITIALIZED = 2,
  LR_ERROR_ALREADY_INITIALIZED = 3,
  LR_ERROR_NOT_SUPPORTED = 4,
  LR_ERROR_RUNTIME = 5
} lr_status_t;

typedef enum lr_memcpy_kind_t {
  LR_MEMCPY_HOST_TO_DEVICE = 0,
  LR_MEMCPY_DEVICE_TO_HOST = 1,
  LR_MEMCPY_DEVICE_TO_DEVICE = 2
} lr_memcpy_kind_t;

typedef struct lr_device_t {
  uint32_t index;
} lr_device_t;

typedef struct lr_event_t lr_event_t;
typedef struct lr_module_t lr_module_t;
typedef struct lr_kernel_t lr_kernel_t;

typedef struct lr_dim3_t {
  uint32_t x;
  uint32_t y;
  uint32_t z;
} lr_dim3_t;

typedef struct lr_launch_config_t {
  lr_dim3_t grid;
  lr_dim3_t block;
  uint32_t shared_memory_bytes;
} lr_launch_config_t;

LRRT_API const char *lr_status_string(lr_status_t status);

LRRT_API lr_status_t lr_init(void);

/* Waits for pending work on all devices before releasing runtime resources. */
LRRT_API lr_status_t lr_shutdown(void);

LRRT_API lr_status_t lr_device_count(uint32_t *count);
LRRT_API lr_status_t lr_device_open(uint32_t index, lr_device_t *device);

LRRT_API lr_status_t lr_event_create(lr_device_t device, lr_event_t **event);

/*
 * Waits for the event if it is still pending, then releases the event handle.
 */
LRRT_API lr_status_t lr_event_destroy(lr_event_t *event);

/*
 * Enqueues an event marker after previously submitted work on the event's
 * device. Re-recording a pending event waits for the previous marker first.
 */
LRRT_API lr_status_t lr_event_record(lr_event_t *event);

/* Waits for a recorded event marker to complete. */
LRRT_API lr_status_t lr_event_synchronize(lr_event_t *event);

/*
 * Returns the elapsed time in nanoseconds between two completed event markers.
 */
LRRT_API lr_status_t lr_event_elapsed_time_ns(const lr_event_t *start,
                                              const lr_event_t *end,
                                              uint64_t *elapsed_ns);

LRRT_API lr_status_t lr_malloc(lr_device_t device, size_t size, void **ptr);

/* Waits for pending work on the device before releasing ptr. */
LRRT_API lr_status_t lr_free(lr_device_t device, void *ptr);

/* Waits for pending work on the device before performing the copy. */
LRRT_API lr_status_t lr_memcpy(lr_device_t device, void *dst, const void *src,
                               size_t size, lr_memcpy_kind_t kind);

/*
 * Starts an asynchronous copy and records completion in event. The event must
 * belong to the same device. The copy receives device-side dependencies for
 * earlier kernel dispatches instead of waiting on the host. Later lr_launch
 * calls similarly depend on pending async copies. Use lr_event_synchronize to
 * wait for copy completion explicitly.
 */
LRRT_API lr_status_t lr_memcpy_async(lr_device_t device, void *dst,
                                     const void *src, size_t size,
                                     lr_memcpy_kind_t kind, lr_event_t *event);

LRRT_API lr_status_t lr_module_load_hsaco(lr_device_t device, const void *image,
                                          size_t image_size,
                                          lr_module_t **module);

/* Waits for pending work on the module's device before destroying it. */
LRRT_API lr_status_t lr_module_destroy(lr_module_t *module);

LRRT_API lr_status_t lr_kernel_get(lr_module_t *module, const char *name,
                                   lr_kernel_t **kernel);

/*
 * Enqueues a kernel dispatch and returns after the packet is submitted to the
 * device queue. Kernel completion and execution errors are observed by
 * lr_synchronize or by later APIs that implicitly wait for pending work.
 */
LRRT_API lr_status_t lr_launch(lr_kernel_t *kernel,
                               const lr_launch_config_t *config,
                               const void *args, size_t args_size);

/* Waits for all previously enqueued work on the device. */
LRRT_API lr_status_t lr_synchronize(lr_device_t device);

#ifdef __cplusplus
}
#endif

#endif // LRRT_LRRT_H_
