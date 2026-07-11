#ifndef LRRT_EXECUTOR_IREE_REGISTRATION_INIT_H_
#define LRRT_EXECUTOR_IREE_REGISTRATION_INIT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t lrrt_iree_hal_register_all_available_drivers(
    iree_hal_driver_registry_t *registry);

iree_status_t lrrt_iree_hal_register_all(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LRRT_EXECUTOR_IREE_REGISTRATION_INIT_H_
