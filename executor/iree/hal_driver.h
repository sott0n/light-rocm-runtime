#ifndef LRRT_EXECUTOR_IREE_HAL_DRIVER_H_
#define LRRT_EXECUTOR_IREE_HAL_DRIVER_H_

#include "executor/iree/registration/driver_module.h"
#include "executor/iree/registration/init.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(LRRT_IREE_HAL_DRIVER_TESTING)
iree_status_t
lrrt_iree_hal_buffer_device_pointer_for_test(iree_hal_buffer_t *buffer,
                                             void **out_device_ptr);
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LRRT_EXECUTOR_IREE_HAL_DRIVER_H_
