#include "lrrt/lrrt.h"

#include <math.h>
#include <stdio.h>

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
    printf("async_copy: skipped, no GPU devices\n");
    lr_shutdown();
    return 0;
  }

  lr_device_t device = {0};
  status = lr_device_open(0, &device);
  if (!expect_status(status, LR_SUCCESS, "lr_device_open")) {
    lr_shutdown();
    return 1;
  }

  const int n = 65;
  float in[n];
  float out[n];
  for (int i = 0; i < n; ++i) {
    in[i] = (float)i + 0.25f;
    out[i] = 0.0f;
  }

  void *device_src = NULL;
  void *device_dst = NULL;
  status = lr_malloc(device, sizeof(in), &device_src);
  if (!expect_status(status, LR_SUCCESS, "lr_malloc src")) {
    lr_shutdown();
    return 1;
  }
  status = lr_malloc(device, sizeof(out), &device_dst);
  if (!expect_status(status, LR_SUCCESS, "lr_malloc dst")) {
    lr_free(device, device_src);
    lr_shutdown();
    return 1;
  }

  status =
      lr_memcpy(device, device_src, in, sizeof(in), LR_MEMCPY_HOST_TO_DEVICE);
  if (!expect_status(status, LR_SUCCESS, "copy in")) {
    lr_free(device, device_dst);
    lr_free(device, device_src);
    lr_shutdown();
    return 1;
  }

  lr_event_t *event = NULL;
  status = lr_event_create(device, &event);
  if (!expect_status(status, LR_SUCCESS, "lr_event_create")) {
    lr_free(device, device_dst);
    lr_free(device, device_src);
    lr_shutdown();
    return 1;
  }

  status = lr_memcpy_async(device, device_dst, device_src, sizeof(in),
                           LR_MEMCPY_DEVICE_TO_DEVICE, event);
  if (!expect_status(status, LR_SUCCESS, "lr_memcpy_async")) {
    lr_event_destroy(event);
    lr_free(device, device_dst);
    lr_free(device, device_src);
    lr_shutdown();
    return 1;
  }

  status = lr_event_synchronize(event);
  if (!expect_status(status, LR_SUCCESS, "lr_event_synchronize")) {
    lr_event_destroy(event);
    lr_free(device, device_dst);
    lr_free(device, device_src);
    lr_shutdown();
    return 1;
  }

  status =
      lr_memcpy(device, out, device_dst, sizeof(out), LR_MEMCPY_DEVICE_TO_HOST);
  if (!expect_status(status, LR_SUCCESS, "copy out")) {
    lr_event_destroy(event);
    lr_free(device, device_dst);
    lr_free(device, device_src);
    lr_shutdown();
    return 1;
  }

  for (int i = 0; i < n; ++i) {
    if (fabsf(out[i] - in[i]) > 0.001f) {
      fprintf(stderr, "async_copy mismatch at %d: got %f expected %f\n", i,
              out[i], in[i]);
      lr_event_destroy(event);
      lr_free(device, device_dst);
      lr_free(device, device_src);
      lr_shutdown();
      return 1;
    }
  }

  for (int i = 0; i < n; ++i) {
    out[i] = 0.0f;
  }
  status =
      lr_memcpy(device, device_dst, out, sizeof(out), LR_MEMCPY_HOST_TO_DEVICE);
  if (!expect_status(status, LR_SUCCESS, "reset destination")) {
    lr_event_destroy(event);
    lr_free(device, device_dst);
    lr_free(device, device_src);
    lr_shutdown();
    return 1;
  }

  lr_event_t *dependency = NULL;
  status = lr_event_create(device, &dependency);
  if (!expect_status(status, LR_SUCCESS, "lr_event_create dependency")) {
    lr_event_destroy(event);
    lr_free(device, device_dst);
    lr_free(device, device_src);
    lr_shutdown();
    return 1;
  }
  status = lr_event_record(dependency);
  if (!expect_status(status, LR_SUCCESS, "lr_event_record dependency")) {
    lr_event_destroy(dependency);
    lr_event_destroy(event);
    lr_free(device, device_dst);
    lr_free(device, device_src);
    lr_shutdown();
    return 1;
  }

  lr_event_t *dependencies[] = {dependency};
  status = lr_memcpy_async_with_dependencies(
      device, device_dst, device_src, sizeof(in), LR_MEMCPY_DEVICE_TO_DEVICE,
      event, dependencies, 1);
  if (!expect_status(status, LR_SUCCESS, "lr_memcpy_async_with_dependencies")) {
    lr_event_destroy(dependency);
    lr_event_destroy(event);
    lr_free(device, device_dst);
    lr_free(device, device_src);
    lr_shutdown();
    return 1;
  }
  status = lr_event_destroy(dependency);
  if (!expect_status(status, LR_SUCCESS, "lr_event_destroy dependency") ||
      !expect_status(lr_event_synchronize(event), LR_SUCCESS,
                     "synchronize dependent copy")) {
    lr_event_destroy(event);
    lr_free(device, device_dst);
    lr_free(device, device_src);
    lr_shutdown();
    return 1;
  }
  status =
      lr_memcpy(device, out, device_dst, sizeof(out), LR_MEMCPY_DEVICE_TO_HOST);
  if (!expect_status(status, LR_SUCCESS, "copy dependent result")) {
    lr_event_destroy(event);
    lr_free(device, device_dst);
    lr_free(device, device_src);
    lr_shutdown();
    return 1;
  }
  for (int i = 0; i < n; ++i) {
    if (fabsf(out[i] - in[i]) > 0.001f) {
      fprintf(stderr,
              "dependent async_copy mismatch at %d: got %f expected %f\n", i,
              out[i], in[i]);
      lr_event_destroy(event);
      lr_free(device, device_dst);
      lr_free(device, device_src);
      lr_shutdown();
      return 1;
    }
  }

  status = lr_memcpy_async(device, device_dst, device_src, sizeof(in) + 1,
                           LR_MEMCPY_DEVICE_TO_DEVICE, event);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "reject out-of-bounds async copy")) {
    lr_event_destroy(event);
    lr_free(device, device_dst);
    lr_free(device, device_src);
    lr_shutdown();
    return 1;
  }

  lr_event_t *self_dependency[] = {event};
  status = lr_memcpy_async_with_dependencies(
      device, device_dst, device_src, sizeof(in), LR_MEMCPY_DEVICE_TO_DEVICE,
      event, self_dependency, 1);
  if (!expect_status(status, LR_ERROR_INVALID_ARGUMENT,
                     "reject completion event dependency")) {
    lr_event_destroy(event);
    lr_free(device, device_dst);
    lr_free(device, device_src);
    lr_shutdown();
    return 1;
  }

  lr_event_destroy(event);
  lr_free(device, device_dst);
  lr_free(device, device_src);
  lr_shutdown();
  printf("async_copy: ok\n");
  return 0;
}
