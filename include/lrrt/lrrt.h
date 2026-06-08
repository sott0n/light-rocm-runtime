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
LRRT_API lr_status_t lr_shutdown(void);

LRRT_API lr_status_t lr_device_count(uint32_t *count);
LRRT_API lr_status_t lr_device_open(uint32_t index, lr_device_t *device);

LRRT_API lr_status_t lr_malloc(lr_device_t device, size_t size, void **ptr);
LRRT_API lr_status_t lr_free(lr_device_t device, void *ptr);
LRRT_API lr_status_t lr_memcpy(lr_device_t device, void *dst, const void *src,
                               size_t size, lr_memcpy_kind_t kind);

LRRT_API lr_status_t lr_module_load_hsaco(lr_device_t device,
                                          const void *image,
                                          size_t image_size,
                                          lr_module_t **module);
LRRT_API lr_status_t lr_module_destroy(lr_module_t *module);

LRRT_API lr_status_t lr_kernel_get(lr_module_t *module, const char *name,
                                   lr_kernel_t **kernel);
LRRT_API lr_status_t lr_launch(lr_kernel_t *kernel,
                               const lr_launch_config_t *config,
                               const void *args, size_t args_size);
LRRT_API lr_status_t lr_synchronize(lr_device_t device);

#ifdef __cplusplus
}
#endif

#endif  // LRRT_LRRT_H_
