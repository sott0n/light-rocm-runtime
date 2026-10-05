#include "lrrt/lrrt.h"

#include <cstdint>
#include <iostream>

namespace {

bool expect(lr_status_t actual, lr_status_t expected, const char *operation) {
  if (actual == expected) {
    return true;
  }
  std::cerr << operation << " returned " << lr_status_string(actual)
            << ", expected " << lr_status_string(expected) << '\n';
  return false;
}

} // namespace

int main() {
  if (!expect(lr_init(), LR_SUCCESS, "lr_init")) {
    return 1;
  }

  lr_device_t device{};
  lr_queue_t *queue = nullptr;
  lr_event_t *start = nullptr;
  lr_event_t *end = nullptr;
  bool correct =
      expect(lr_device_open(0, &device), LR_SUCCESS, "lr_device_open") &&
      expect(lr_queue_create(device, &queue), LR_SUCCESS, "lr_queue_create") &&
      expect(lr_event_create(device, &start), LR_SUCCESS,
             "lr_event_create(start)") &&
      expect(lr_event_create(device, &end), LR_SUCCESS, "lr_event_create(end)");

  uint64_t elapsed_ns = 0;
  uint64_t duration_ns = 0;
  if (correct) {
    correct = expect(lr_event_elapsed_time_ns(start, end, &elapsed_ns),
                     LR_ERROR_INVALID_ARGUMENT,
                     "unrecorded lr_event_elapsed_time_ns") &&
              expect(lr_event_record_on_queue(start, queue), LR_SUCCESS,
                     "lr_event_record_on_queue(start)") &&
              expect(lr_event_record_on_queue(end, queue), LR_SUCCESS,
                     "lr_event_record_on_queue(end)") &&
              expect(lr_event_synchronize(end), LR_SUCCESS,
                     "lr_event_synchronize(end)") &&
              expect(lr_event_synchronize(start), LR_SUCCESS,
                     "lr_event_synchronize(start)") &&
              expect(lr_event_elapsed_time_ns(start, end, &elapsed_ns),
                     LR_SUCCESS, "lr_event_elapsed_time_ns") &&
              expect(lr_event_duration_ns(end, &duration_ns), LR_SUCCESS,
                     "lr_event_duration_ns");
    if (correct && elapsed_ns == 0) {
      std::cerr << "profiled event interval was zero\n";
      correct = false;
    }
  }

  if (start &&
      !expect(lr_event_destroy(start), LR_SUCCESS, "lr_event_destroy(start)")) {
    correct = false;
  }
  if (correct) {
    correct = expect(lr_event_record_on_queue(end, queue), LR_SUCCESS,
                     "lr_event_record_on_queue(re-record)") &&
              expect(lr_queue_destroy(queue), LR_SUCCESS,
                     "lr_queue_destroy with pending event") &&
              expect(lr_event_synchronize(end), LR_SUCCESS,
                     "lr_event_synchronize after queue destroy");
    queue = nullptr;
  }
  if (end &&
      !expect(lr_event_destroy(end), LR_SUCCESS, "lr_event_destroy(end)")) {
    correct = false;
  }
  if (queue && !expect(lr_queue_destroy(queue), LR_SUCCESS,
                       "lr_queue_destroy cleanup")) {
    correct = false;
  }
  if (!expect(lr_shutdown(), LR_SUCCESS, "lr_shutdown")) {
    correct = false;
  }

  if (correct) {
    std::cout << "direct_kfd_event: elapsed_ns=" << elapsed_ns
              << " duration_ns=" << duration_ns << '\n';
  }
  return correct ? 0 : 1;
}
