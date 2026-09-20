#include "lrrt/lrrt.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef LRRT_COPY_HSACO
#define LRRT_COPY_HSACO "copy_kernel.hsaco"
#endif

typedef struct copy_args_t {
  const float *in;
  float *out;
  int n;
} copy_args_t;

static int expect_status(lr_status_t actual, lr_status_t expected,
                         const char *operation) {
  if (actual == expected) {
    return 1;
  }
  fprintf(stderr, "%s returned %s, expected %s\n", operation,
          lr_status_string(actual), lr_status_string(expected));
  return 0;
}

static int read_file(const char *path, void **data, size_t *size) {
  FILE *file = fopen(path, "rb");
  if (!file || fseek(file, 0, SEEK_END) != 0) {
    if (file) {
      fclose(file);
    }
    return 0;
  }
  const long length = ftell(file);
  if (length <= 0) {
    fclose(file);
    return 0;
  }
  rewind(file);
  *data = malloc((size_t)length);
  if (!*data || fread(*data, 1, (size_t)length, file) != (size_t)length) {
    free(*data);
    *data = NULL;
    fclose(file);
    return 0;
  }
  fclose(file);
  *size = (size_t)length;
  return 1;
}

static int check_output(const float *actual, const float *expected, int count) {
  for (int index = 0; index < count; ++index) {
    if (fabsf(actual[index] - expected[index]) > 0.001f) {
      fprintf(stderr,
              "kernel_event_dependencies mismatch at %d: got %f expected %f\n",
              index, actual[index], expected[index]);
      return 0;
    }
  }
  return 1;
}

int main(void) {
  if (!expect_status(lr_init(), LR_SUCCESS, "lr_init")) {
    return 1;
  }
  uint32_t device_count = 0;
  if (!expect_status(lr_device_count(&device_count), LR_SUCCESS,
                     "lr_device_count")) {
    lr_shutdown();
    return 1;
  }
  if (device_count == 0) {
    printf("kernel_event_dependencies: skipped, no GPU devices\n");
    lr_shutdown();
    return 0;
  }

  lr_device_t device = {0};
  if (!expect_status(lr_device_open(0, &device), LR_SUCCESS,
                     "lr_device_open")) {
    lr_shutdown();
    return 1;
  }

  const int n = 64;
  float input[n];
  float output[n];
  for (int index = 0; index < n; ++index) {
    input[index] = (float)(index * 7 + 1);
    output[index] = 0.0f;
  }

  void *device_input = NULL;
  void *device_intermediate = NULL;
  void *device_output = NULL;
  void *hsaco = NULL;
  size_t hsaco_size = 0;
  lr_module_t *module = NULL;
  lr_kernel_t *kernel = NULL;
  lr_queue_t *producer_queue = NULL;
  lr_queue_t *consumer_queue = NULL;
  lr_event_t *events[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
  int ok =
      expect_status(lr_malloc(device, sizeof(input), &device_input), LR_SUCCESS,
                    "lr_malloc input") &&
      expect_status(lr_malloc(device, sizeof(input), &device_intermediate),
                    LR_SUCCESS, "lr_malloc intermediate") &&
      expect_status(lr_malloc(device, sizeof(input), &device_output),
                    LR_SUCCESS, "lr_malloc output") &&
      expect_status(lr_memcpy(device, device_input, input, sizeof(input),
                              LR_MEMCPY_HOST_TO_DEVICE),
                    LR_SUCCESS, "copy input") &&
      read_file(LRRT_COPY_HSACO, &hsaco, &hsaco_size) &&
      expect_status(lr_module_load_hsaco(device, hsaco, hsaco_size, &module),
                    LR_SUCCESS, "lr_module_load_hsaco") &&
      expect_status(lr_kernel_get(module, "copy_kernel", &kernel), LR_SUCCESS,
                    "lr_kernel_get") &&
      expect_status(lr_queue_create(device, &producer_queue), LR_SUCCESS,
                    "create producer queue") &&
      expect_status(lr_queue_create(device, &consumer_queue), LR_SUCCESS,
                    "create consumer queue");
  free(hsaco);
  if (!ok) {
    lr_shutdown();
    return 1;
  }

  for (size_t index = 0; index < 6; ++index) {
    if (!expect_status(lr_event_create(device, &events[index]), LR_SUCCESS,
                       "lr_event_create")) {
      lr_shutdown();
      return 1;
    }
  }

#ifdef LRRT_TEST_HIDDEN_KERNARGS
  const lr_launch_config_t config = {{192, 1, 1}, {64, 1, 1}, 0};
#else
  const lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
#endif
  const copy_args_t producer_args = {(const float *)device_input,
                                     (float *)device_intermediate, n};
  const copy_args_t consumer_args = {(const float *)device_intermediate,
                                     (float *)device_output, n};
  ok = expect_status(lr_launch_on_queue(producer_queue, kernel, &config,
                                        &producer_args, sizeof(producer_args)),
                     LR_SUCCESS, "producer launch");
  for (size_t index = 0; ok && index < 6; ++index) {
    ok = expect_status(lr_event_record_on_queue(events[index], producer_queue),
                       LR_SUCCESS, "record dependency event");
  }
  if (ok) {
    ok = expect_status(lr_launch_on_queue_with_dependencies(
                           consumer_queue, kernel, &config, &consumer_args,
                           sizeof(consumer_args), events, 6),
                       LR_SUCCESS, "dependent queue launch");
  }
  for (size_t index = 0; ok && index < 6; ++index) {
    ok = expect_status(lr_event_destroy(events[index]), LR_SUCCESS,
                       "destroy consumed event");
    events[index] = NULL;
  }
  if (ok) {
    ok = expect_status(lr_memcpy(device, output, device_output, sizeof(output),
                                 LR_MEMCPY_DEVICE_TO_HOST),
                       LR_SUCCESS, "copy dependent output") &&
         check_output(output, input, n);
  }

  lr_event_t *validation_event = NULL;
  if (ok) {
    ok = expect_status(lr_event_create(device, &validation_event), LR_SUCCESS,
                       "create validation event");
  }
  lr_event_t *single_dependency[] = {validation_event};
  if (ok) {
    ok = expect_status(lr_launch_with_dependencies(
                           kernel, &config, &consumer_args,
                           sizeof(consumer_args), single_dependency, 1),
                       LR_ERROR_INVALID_ARGUMENT,
                       "reject unrecorded dependency");
  }
  if (ok) {
    for (int index = 0; index < n; ++index) {
      output[index] = 0.0f;
    }
    ok = expect_status(lr_memcpy(device, device_output, output, sizeof(output),
                                 LR_MEMCPY_HOST_TO_DEVICE),
                       LR_SUCCESS, "reset default queue output");
  }
  if (ok) {
    ok =
        expect_status(lr_launch_on_queue(producer_queue, kernel, &config,
                                         &producer_args, sizeof(producer_args)),
                      LR_SUCCESS, "second producer launch") &&
        expect_status(
            lr_event_record_on_queue(validation_event, producer_queue),
            LR_SUCCESS, "record validation event");
  }
  lr_event_t *duplicate_dependencies[] = {validation_event, validation_event};
  if (ok) {
    ok = expect_status(lr_launch_with_dependencies(
                           kernel, &config, &consumer_args,
                           sizeof(consumer_args), duplicate_dependencies, 2),
                       LR_ERROR_INVALID_ARGUMENT,
                       "reject duplicate dependencies") &&
         expect_status(lr_launch_with_dependencies(
                           kernel, &config, &consumer_args,
                           sizeof(consumer_args), single_dependency, 1),
                       LR_SUCCESS, "dependent default queue launch") &&
         expect_status(lr_event_destroy(validation_event), LR_SUCCESS,
                       "destroy default queue dependency");
    validation_event = NULL;
  }
  if (ok) {
    ok = expect_status(lr_memcpy(device, output, device_output, sizeof(output),
                                 LR_MEMCPY_DEVICE_TO_HOST),
                       LR_SUCCESS, "copy default queue output") &&
         check_output(output, input, n);
  }

  if (validation_event) {
    lr_event_destroy(validation_event);
  }
  for (size_t index = 0; index < 6; ++index) {
    if (events[index]) {
      lr_event_destroy(events[index]);
    }
  }
  if (consumer_queue) {
    ok = expect_status(lr_queue_destroy(consumer_queue), LR_SUCCESS,
                       "destroy consumer queue") &&
         ok;
  }
  if (producer_queue) {
    ok = expect_status(lr_queue_destroy(producer_queue), LR_SUCCESS,
                       "destroy producer queue") &&
         ok;
  }
  if (module) {
    ok = expect_status(lr_module_destroy(module), LR_SUCCESS,
                       "lr_module_destroy") &&
         ok;
  }
  if (device_output) {
    ok = expect_status(lr_free(device, device_output), LR_SUCCESS,
                       "lr_free output") &&
         ok;
  }
  if (device_intermediate) {
    ok = expect_status(lr_free(device, device_intermediate), LR_SUCCESS,
                       "lr_free intermediate") &&
         ok;
  }
  if (device_input) {
    ok = expect_status(lr_free(device, device_input), LR_SUCCESS,
                       "lr_free input") &&
         ok;
  }
  ok = expect_status(lr_shutdown(), LR_SUCCESS, "lr_shutdown") && ok;
  if (!ok) {
    return 1;
  }
  printf("kernel_event_dependencies: ok\n");
  return 0;
}
