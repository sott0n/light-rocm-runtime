#ifndef LRRT_EXECUTOR_IREE_HAL_DRIVER_H_
#define LRRT_EXECUTOR_IREE_HAL_DRIVER_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t
lrrt_iree_hal_driver_module_register(iree_hal_driver_registry_t *registry);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LRRT_EXECUTOR_IREE_HAL_DRIVER_H_
