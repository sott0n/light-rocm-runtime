#include <stdio.h>
#include <stdlib.h>

#include "executor/iree/hal_driver.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/driver_registry.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/run_module.h"
#include "iree/vm/api.h"

static const char kUsage[] =
    "Runs an IREE module after registering the experimental lrrt HAL driver.\n"
    "\n"
    "This is a local smoke runner for validating the seamless IREE tooling "
    "path:\n"
    "  lrrt_iree_run_module_smoke --device=lrrt --module=model.vmfb ...\n";

int main(int argc, char **argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_status_t status = lrrt_iree_hal_register_all();
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(EXIT_FAILURE);
    return EXIT_FAILURE;
  }

  iree_flags_set_usage("lrrt_iree_run_module_smoke", kUsage);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_vm_instance_t *instance = NULL;
  status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_tooling_create_instance(host_allocator, &instance);
  }

  int exit_code = EXIT_SUCCESS;
  if (iree_status_is_ok(status)) {
    status = iree_tooling_run_module_from_flags(instance, host_allocator,
                                                &exit_code);
  }

  iree_vm_instance_release(instance);

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
