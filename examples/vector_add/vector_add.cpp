#include "lrrt/lrrt.h"

#include <math.h>
#include <string.h>
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

int main(void) {
  lr_status_t status = lr_init();
  if (status != LR_SUCCESS) {
    fprintf(stderr, "lr_init failed: %s\n", lr_status_string(status));
    return 1;
  }

  uint32_t count = 0;
  status = lr_device_count(&count);
  if (status != LR_SUCCESS) {
    fprintf(stderr, "lr_device_count failed: %s\n", lr_status_string(status));
    lr_shutdown();
    return 1;
  }

  printf("devices: %u\n", count);
  if (count > 0) {
    lr_device_t device = {0};
    status = lr_device_open(0, &device);
    if (status != LR_SUCCESS) {
      fprintf(stderr, "lr_device_open failed: %s\n", lr_status_string(status));
      lr_shutdown();
      return 1;
    }
    printf("opened device: %u\n", device.index);

    const int n = 64;
    float a[n];
    float b[n];
    float c[n];
    for (int i = 0; i < n; ++i) {
      a[i] = (float)i;
      b[i] = (float)(i * 2);
      c[i] = 0.0f;
    }

    void *device_a = NULL;
    void *device_b = NULL;
    void *device_c = NULL;

    status = lr_malloc(device, sizeof(a), &device_a);
    if (status != LR_SUCCESS) {
      fprintf(stderr, "lr_malloc a failed: %s\n", lr_status_string(status));
      lr_shutdown();
      return 1;
    }
    status = lr_malloc(device, sizeof(b), &device_b);
    if (status != LR_SUCCESS) {
      fprintf(stderr, "lr_malloc b failed: %s\n", lr_status_string(status));
      lr_free(device, device_a);
      lr_shutdown();
      return 1;
    }
    status = lr_malloc(device, sizeof(c), &device_c);
    if (status != LR_SUCCESS) {
      fprintf(stderr, "lr_malloc c failed: %s\n", lr_status_string(status));
      lr_free(device, device_b);
      lr_free(device, device_a);
      lr_shutdown();
      return 1;
    }

    status = lr_memcpy(device, device_a, a, sizeof(a),
                       LR_MEMCPY_HOST_TO_DEVICE);
    if (status != LR_SUCCESS) {
      fprintf(stderr, "host-to-device copy a failed: %s\n",
              lr_status_string(status));
      lr_free(device, device_c);
      lr_free(device, device_b);
      lr_free(device, device_a);
      lr_shutdown();
      return 1;
    }
    status = lr_memcpy(device, device_b, b, sizeof(b),
                       LR_MEMCPY_HOST_TO_DEVICE);
    if (status != LR_SUCCESS) {
      fprintf(stderr, "host-to-device copy b failed: %s\n",
              lr_status_string(status));
      lr_free(device, device_c);
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
    if (status != LR_SUCCESS) {
      fprintf(stderr, "lr_module_load_hsaco failed: %s\n",
              lr_status_string(status));
      lr_free(device, device_c);
      lr_free(device, device_b);
      lr_free(device, device_a);
      lr_shutdown();
      return 1;
    }

    lr_kernel_t *kernel = NULL;
    status = lr_kernel_get(module, "vector_add", &kernel);
    if (status != LR_SUCCESS) {
      fprintf(stderr, "lr_kernel_get failed: %s\n", lr_status_string(status));
      lr_module_destroy(module);
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
        n,
    };
    lr_launch_config_t config = {
        {64, 1, 1},
        {64, 1, 1},
        0,
    };

    status = lr_launch(kernel, &config, &args, sizeof(args));
    if (status != LR_SUCCESS) {
      fprintf(stderr, "lr_launch failed: %s\n", lr_status_string(status));
      lr_module_destroy(module);
      lr_free(device, device_c);
      lr_free(device, device_b);
      lr_free(device, device_a);
      lr_shutdown();
      return 1;
    }

    status = lr_memcpy(device, c, device_c, sizeof(c),
                       LR_MEMCPY_DEVICE_TO_HOST);
    if (status != LR_SUCCESS) {
      fprintf(stderr, "device-to-host copy c failed: %s\n",
              lr_status_string(status));
      lr_module_destroy(module);
      lr_free(device, device_c);
      lr_free(device, device_b);
      lr_free(device, device_a);
      lr_shutdown();
      return 1;
    }

    for (int i = 0; i < n; ++i) {
      if (fabsf(c[i] - (a[i] + b[i])) > 0.001f) {
        fprintf(stderr, "vector_add mismatch at %d: got %f expected %f\n", i,
                c[i], a[i] + b[i]);
        lr_module_destroy(module);
        lr_free(device, device_c);
        lr_free(device, device_b);
        lr_free(device, device_a);
        lr_shutdown();
        return 1;
      }
    }
    printf("vector_add: ok\n");

    lr_module_destroy(module);
    lr_free(device, device_c);
    lr_free(device, device_b);
    lr_free(device, device_a);
  }

  lr_shutdown();
  return 0;
}
