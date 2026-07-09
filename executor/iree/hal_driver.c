#include "hal_driver.h"

#include <stddef.h>
#include <string.h>

#include "iree/hal/resource.h"

typedef struct lrrt_iree_hal_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_string_view_t identifier;
} lrrt_iree_hal_driver_t;

static const iree_hal_driver_vtable_t lrrt_iree_hal_driver_vtable;

static lrrt_iree_hal_driver_t *
lrrt_iree_hal_driver_cast(iree_hal_driver_t *base_driver) {
  IREE_HAL_ASSERT_TYPE(base_driver, &lrrt_iree_hal_driver_vtable);
  return (lrrt_iree_hal_driver_t *)base_driver;
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
  (void)device_id;
  (void)param_count;
  (void)params;
  (void)create_params;
  (void)host_allocator;
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "lrrt HAL device creation is not implemented yet");
}

static iree_status_t lrrt_iree_hal_driver_create_device_by_path(
    iree_hal_driver_t *base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t *params,
    const iree_hal_device_create_params_t *create_params,
    iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
  (void)base_driver;
  (void)driver_name;
  (void)device_path;
  (void)param_count;
  (void)params;
  (void)create_params;
  (void)host_allocator;
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "lrrt HAL device creation is not implemented yet");
}

static const iree_hal_driver_vtable_t lrrt_iree_hal_driver_vtable = {
    .destroy = lrrt_iree_hal_driver_destroy,
    .query_available_devices = lrrt_iree_hal_driver_query_available_devices,
    .dump_device_info = lrrt_iree_hal_driver_dump_device_info,
    .create_device_by_id = lrrt_iree_hal_driver_create_device_by_id,
    .create_device_by_path = lrrt_iree_hal_driver_create_device_by_path,
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
