#include "lrrt/lrrt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef LRRT_SCALE_HSACO
#define LRRT_SCALE_HSACO "scale_kernel.hsaco"
#endif

typedef struct scale_args_t {
  const float *in;
  float *out;
  float alpha;
  int n;
} scale_args_t;

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
    printf("event_timing: skipped, no GPU devices\n");
    lr_shutdown();
    return 0;
  }

  lr_device_t device = {0};
  status = lr_device_open(0, &device);
  if (!expect_status(status, LR_SUCCESS, "lr_device_open")) {
    lr_shutdown();
    return 1;
  }

  const int n = 64;
  const float alpha = 3.0f;
  float in[n];
  float out[n];
  for (int i = 0; i < n; ++i) {
    in[i] = (float)i + 1.0f;
    out[i] = 0.0f;
  }

  void *device_in = NULL;
  void *device_out = NULL;
  status = lr_malloc(device, sizeof(in), &device_in);
  if (!expect_status(status, LR_SUCCESS, "lr_malloc in")) {
    lr_shutdown();
    return 1;
  }
  status = lr_malloc(device, sizeof(out), &device_out);
  if (!expect_status(status, LR_SUCCESS, "lr_malloc out")) {
    lr_free(device, device_in);
    lr_shutdown();
    return 1;
  }
  status =
      lr_memcpy(device, device_in, in, sizeof(in), LR_MEMCPY_HOST_TO_DEVICE);
  if (!expect_status(status, LR_SUCCESS, "copy in")) {
    lr_free(device, device_out);
    lr_free(device, device_in);
    lr_shutdown();
    return 1;
  }

  void *hsaco = NULL;
  size_t hsaco_size = 0;
  if (!read_file(LRRT_SCALE_HSACO, &hsaco, &hsaco_size)) {
    fprintf(stderr, "failed to read HSACO: %s\n", LRRT_SCALE_HSACO);
    lr_free(device, device_out);
    lr_free(device, device_in);
    lr_shutdown();
    return 1;
  }

  lr_module_t *module = NULL;
  status = lr_module_load_hsaco(device, hsaco, hsaco_size, &module);
  free(hsaco);
  if (!expect_status(status, LR_SUCCESS, "lr_module_load_hsaco")) {
    lr_free(device, device_out);
    lr_free(device, device_in);
    lr_shutdown();
    return 1;
  }

  lr_kernel_t *kernel = NULL;
  status = lr_kernel_get(module, "scale", &kernel);
  if (!expect_status(status, LR_SUCCESS, "lr_kernel_get")) {
    lr_module_destroy(module);
    lr_free(device, device_out);
    lr_free(device, device_in);
    lr_shutdown();
    return 1;
  }

  lr_event_t *start = NULL;
  lr_event_t *end = NULL;
  status = lr_event_create(device, &start);
  if (!expect_status(status, LR_SUCCESS, "lr_event_create start")) {
    lr_module_destroy(module);
    lr_free(device, device_out);
    lr_free(device, device_in);
    lr_shutdown();
    return 1;
  }
  status = lr_event_create(device, &end);
  if (!expect_status(status, LR_SUCCESS, "lr_event_create end")) {
    lr_event_destroy(start);
    lr_module_destroy(module);
    lr_free(device, device_out);
    lr_free(device, device_in);
    lr_shutdown();
    return 1;
  }

  status = lr_event_record(start);
  if (!expect_status(status, LR_SUCCESS, "lr_event_record start")) {
    lr_event_destroy(end);
    lr_event_destroy(start);
    lr_module_destroy(module);
    lr_free(device, device_out);
    lr_free(device, device_in);
    lr_shutdown();
    return 1;
  }

  scale_args_t args = {
      (const float *)device_in,
      (float *)device_out,
      alpha,
      n,
  };
  lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
  status = lr_launch(kernel, &config, &args, sizeof(args));
  if (!expect_status(status, LR_SUCCESS, "lr_launch")) {
    lr_event_destroy(end);
    lr_event_destroy(start);
    lr_module_destroy(module);
    lr_free(device, device_out);
    lr_free(device, device_in);
    lr_shutdown();
    return 1;
  }

  status = lr_event_record(end);
  if (!expect_status(status, LR_SUCCESS, "lr_event_record end")) {
    lr_event_destroy(end);
    lr_event_destroy(start);
    lr_module_destroy(module);
    lr_free(device, device_out);
    lr_free(device, device_in);
    lr_shutdown();
    return 1;
  }

  status = lr_event_synchronize(end);
  if (!expect_status(status, LR_SUCCESS, "lr_event_synchronize end")) {
    lr_event_destroy(end);
    lr_event_destroy(start);
    lr_module_destroy(module);
    lr_free(device, device_out);
    lr_free(device, device_in);
    lr_shutdown();
    return 1;
  }
  status = lr_event_synchronize(start);
  if (!expect_status(status, LR_SUCCESS, "lr_event_synchronize start")) {
    lr_event_destroy(end);
    lr_event_destroy(start);
    lr_module_destroy(module);
    lr_free(device, device_out);
    lr_free(device, device_in);
    lr_shutdown();
    return 1;
  }

  uint64_t elapsed_ns = 0;
  status = lr_event_elapsed_time_ns(start, end, &elapsed_ns);
  if (!expect_status(status, LR_SUCCESS, "lr_event_elapsed_time_ns")) {
    lr_event_destroy(end);
    lr_event_destroy(start);
    lr_module_destroy(module);
    lr_free(device, device_out);
    lr_free(device, device_in);
    lr_shutdown();
    return 1;
  }
  if (elapsed_ns == 0) {
    fprintf(stderr, "event elapsed time was zero\n");
    lr_event_destroy(end);
    lr_event_destroy(start);
    lr_module_destroy(module);
    lr_free(device, device_out);
    lr_free(device, device_in);
    lr_shutdown();
    return 1;
  }

  uint64_t duration_ns = 0;
  status = lr_event_duration_ns(end, &duration_ns);
  if (!expect_status(status, LR_SUCCESS, "lr_event_duration_ns")) {
    lr_event_destroy(end);
    lr_event_destroy(start);
    lr_module_destroy(module);
    lr_free(device, device_out);
    lr_free(device, device_in);
    lr_shutdown();
    return 1;
  }

  status =
      lr_memcpy(device, out, device_out, sizeof(out), LR_MEMCPY_DEVICE_TO_HOST);
  if (!expect_status(status, LR_SUCCESS, "copy out")) {
    lr_event_destroy(end);
    lr_event_destroy(start);
    lr_module_destroy(module);
    lr_free(device, device_out);
    lr_free(device, device_in);
    lr_shutdown();
    return 1;
  }

  for (int i = 0; i < n; ++i) {
    float expected = alpha * in[i];
    if (fabsf(out[i] - expected) > 0.001f) {
      fprintf(stderr, "event_timing mismatch at %d: got %f expected %f\n", i,
              out[i], expected);
      lr_event_destroy(end);
      lr_event_destroy(start);
      lr_module_destroy(module);
      lr_free(device, device_out);
      lr_free(device, device_in);
      lr_shutdown();
      return 1;
    }
  }

  lr_event_destroy(end);
  lr_event_destroy(start);
  lr_module_destroy(module);
  lr_free(device, device_out);
  lr_free(device, device_in);
  lr_shutdown();
  printf("event_timing: ok (%llu ns)\n", (unsigned long long)elapsed_ns);
  return 0;
}
