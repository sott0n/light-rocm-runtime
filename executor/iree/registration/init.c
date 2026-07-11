#include "executor/iree/registration/init.h"

#include "executor/iree/registration/driver_module.h"
#include "iree/hal/driver_registry.h"

iree_status_t lrrt_iree_hal_register_all_available_drivers(
    iree_hal_driver_registry_t *registry) {
  return lrrt_iree_hal_driver_module_register(registry);
}

iree_status_t lrrt_iree_hal_register_all(void) {
  iree_status_t status = lrrt_iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_free(status);
    return iree_ok_status();
  }
  return status;
}
