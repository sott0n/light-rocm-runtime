#include "lrrt/lrrt.h"

#include <stdio.h>
#include <stdlib.h>

#ifndef LRRT_VECTOR_ADD_HSACO
#define LRRT_VECTOR_ADD_HSACO "vector_add_kernel.hsaco"
#endif

typedef struct vector_add_args_t {
  const float *a;
  const float *b;
  float *c;
  int n;
} vector_add_args_t;

static int read_file(const char *path, void **data, size_t *size) {
  FILE *file = fopen(path, "rb");
  if (!file) {
    return 0;
  }
  if (fseek(file, 0, SEEK_END) != 0) {
    fclose(file);
    return 0;
  }
  long length = ftell(file);
  if (length <= 0) {
    fclose(file);
    return 0;
  }
  rewind(file);

  void *buffer = malloc((size_t)length);
  if (!buffer) {
    fclose(file);
    return 0;
  }
  if (fread(buffer, 1, (size_t)length, file) != (size_t)length) {
    free(buffer);
    fclose(file);
    return 0;
  }
  fclose(file);
  *data = buffer;
  *size = (size_t)length;
  return 1;
}

static int expect_status(lr_status_t actual, lr_status_t expected,
                         const char *operation) {
  if (actual == expected) {
    return 1;
  }
  fprintf(stderr, "%s returned %s, expected %s\n", operation,
          lr_status_string(actual), lr_status_string(expected));
  return 0;
}

int main(void) {
  lr_status_t status = lr_init();
  if (!expect_status(status, LR_SUCCESS, "lr_init")) {
    return 1;
  }

  uint32_t count = 0;
  status = lr_device_count(&count);
  if (!expect_status(status, LR_SUCCESS, "lr_device_count")) {
    lr_shutdown();
    return 1;
  }
  if (count == 0) {
    printf("handle_registry: skipped, no GPU devices\n");
    lr_shutdown();
    return 0;
  }

  lr_device_t device = {0};
  status = lr_device_open(0, &device);
  if (!expect_status(status, LR_SUCCESS, "lr_device_open")) {
    lr_shutdown();
    return 1;
  }

  float a[64] = {0};
  float b[64] = {0};
  float c[64] = {0};
  void *device_a = NULL;
  void *device_b = NULL;
  void *device_c = NULL;

  status = lr_malloc(device, sizeof(a), &device_a);
  if (!expect_status(status, LR_SUCCESS, "lr_malloc a")) {
    lr_shutdown();
    return 1;
  }
  status = lr_malloc(device, sizeof(b), &device_b);
  if (!expect_status(status, LR_SUCCESS, "lr_malloc b")) {
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }
  status = lr_malloc(device, sizeof(c), &device_c);
  if (!expect_status(status, LR_SUCCESS, "lr_malloc c")) {
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }

  void *hsaco = NULL;
  size_t hsaco_size = 0;
  if (!read_file(LRRT_VECTOR_ADD_HSACO, &hsaco, &hsaco_size)) {
    fprintf(stderr, "failed to read HSACO: %s\n", LRRT_VECTOR_ADD_HSACO);
    lr_free(device, device_c);
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }

  lr_module_t *module = NULL;
  status = lr_module_load_hsaco(device, hsaco, hsaco_size, &module);
  free(hsaco);
  if (!expect_status(status, LR_SUCCESS, "lr_module_load_hsaco")) {
    lr_free(device, device_c);
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }

  lr_kernel_t *kernel = NULL;
  status = lr_kernel_get(module, "vector_add", &kernel);
  if (!expect_status(status, LR_SUCCESS, "lr_kernel_get")) {
    lr_module_destroy(module);
    lr_free(device, device_c);
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }

  status = lr_module_destroy(module);
  if (!expect_status(status, LR_SUCCESS, "lr_module_destroy")) {
    lr_free(device, device_c);
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }

  vector_add_args_t args = {
      (const float *)device_a,
      (const float *)device_b,
      (float *)device_c,
      64,
  };
  lr_launch_config_t config = {
      {64, 1, 1},
      {64, 1, 1},
      0,
  };

  status = lr_launch(kernel, &config, &args, sizeof(args));
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "stale kernel launch")) {
    lr_free(device, device_c);
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }

  status = lr_module_destroy(module);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "stale module destroy")) {
    lr_free(device, device_c);
    lr_free(device, device_b);
    lr_free(device, device_a);
    lr_shutdown();
    return 1;
  }

  lr_free(device, device_c);
  lr_free(device, device_b);
  lr_free(device, device_a);
  lr_shutdown();

  printf("handle_registry: ok\n");
  return 0;
}
