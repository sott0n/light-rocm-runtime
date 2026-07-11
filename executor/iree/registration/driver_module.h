#ifndef LRRT_EXECUTOR_IREE_REGISTRATION_DRIVER_MODULE_H_
#define LRRT_EXECUTOR_IREE_REGISTRATION_DRIVER_MODULE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t
lrrt_iree_hal_driver_module_register(iree_hal_driver_registry_t *registry);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LRRT_EXECUTOR_IREE_REGISTRATION_DRIVER_MODULE_H_
