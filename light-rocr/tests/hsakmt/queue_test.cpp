#include "light_rocr/transport/hsakmt/queue.hpp"

#include <hsakmt/hsakmt.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using light_rocr::transport::hsakmt::kAqlPacketSize;
using light_rocr::transport::hsakmt::kAqlRingDefaultSize;
using light_rocr::transport::hsakmt::kMemoryPageSize;

constexpr uint64_t kRingGpuAddress = 0x300000;
constexpr uint64_t kControlGpuAddress = 0x400000;
constexpr HSA_QUEUEID kQueueId = 0x12345678;

alignas(4096) std::array<uint8_t, kAqlRingDefaultSize> ring_memory;
alignas(4096) std::array<uint8_t, kMemoryPageSize> control_memory;
uint64_t doorbell = 0;

struct AllocationCall {
  uint32_t preferred_node = 0;
  uint64_t size = 0;
  HsaMemFlags flags{};
  void *address = nullptr;
};

struct MapCall {
  void *address = nullptr;
  uint64_t size = 0;
  HsaMemMapFlags flags{};
  uint32_t node = 0;
};

struct FakeKmt {
  HSAKMT_STATUS open_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS acquire_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS create_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS destroy_status = HSAKMT_STATUS_SUCCESS;
  size_t fail_allocation_call = 0;
  size_t fail_unmap_call = 0;
  size_t unmap_calls = 0;
  HsaQueueResource queue_resource_input{};
  uint32_t queue_node = 0;
  HSA_QUEUE_TYPE queue_type = HSA_QUEUE_TYPE_SIZE;
  uint32_t queue_percentage = 0;
  HSA_QUEUE_PRIORITY queue_priority = HSA_QUEUE_PRIORITY_SIZE;
  void *queue_address = nullptr;
  uint64_t queue_size = 0;
  HsaEvent *queue_event = reinterpret_cast<HsaEvent *>(uintptr_t{1});
  std::vector<AllocationCall> allocations;
  std::vector<MapCall> maps;
  std::vector<std::string> calls;
};

FakeKmt fake;

void reset_fake() {
  fake = {};
  std::fill(ring_memory.begin(), ring_memory.end(), uint8_t{0xa5});
  std::fill(control_memory.begin(), control_memory.end(), uint8_t{0xa5});
  doorbell = UINT64_MAX;
}

const char *allocation_name(void *address) {
  if (address == ring_memory.data()) {
    return "ring";
  }
  if (address == control_memory.data()) {
    return "control";
  }
  return "unknown";
}

struct TestContext {
  int failures = 0;

  void expect(bool condition, const std::string &message) {
    if (!condition) {
      ++failures;
      std::cerr << "  FAIL: " << message << '\n';
    }
  }
};

using TestFunction = std::function<void(TestContext *)>;

void expect_calls(TestContext *context,
                  const std::vector<std::string> &expected) {
  context->expect(fake.calls == expected, "unexpected KMT call sequence");
}

void successful_round_trip(TestContext *context) {
  reset_fake();
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto created = opened.session.create_aql_queue(7, kAqlRingDefaultSize);
    context->expect(static_cast<bool>(created), created.status.message);
    context->expect(created.queue.queue_id() == kQueueId,
                    "queue ID was not retained");
    context->expect(created.queue.doorbell_address() ==
                        reinterpret_cast<uintptr_t>(&doorbell),
                    "doorbell address was not retained");
    context->expect(created.queue.ring_host_address() == ring_memory.data() &&
                        created.queue.ring_gpu_address() == kRingGpuAddress &&
                        created.queue.ring_size() == kAqlRingDefaultSize,
                    "ring properties were not retained");

    context->expect(fake.allocations.size() == 2,
                    "queue did not make two allocations");
    if (fake.allocations.size() == 2) {
      HsaMemFlags expected_ring_flags{};
      expected_ring_flags.ui32.NonPaged = 1;
      expected_ring_flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
      expected_ring_flags.ui32.HostAccess = 1;
      expected_ring_flags.ui32.ExecuteAccess = 1;
      expected_ring_flags.ui32.AQLQueueMemory = 1;
      expected_ring_flags.ui32.NoNUMABind = 1;
      context->expect(fake.allocations[0].preferred_node == 0 &&
                          fake.allocations[0].size == kAqlRingDefaultSize &&
                          fake.allocations[0].flags.Value ==
                              expected_ring_flags.Value,
                      "unexpected AQL ring allocation policy");

      HsaMemFlags expected_control_flags{};
      expected_control_flags.ui32.NonPaged = 1;
      expected_control_flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
      expected_control_flags.ui32.HostAccess = 1;
      expected_control_flags.ui32.NoNUMABind = 1;
      context->expect(fake.allocations[1].preferred_node == 0 &&
                          fake.allocations[1].size == kMemoryPageSize &&
                          fake.allocations[1].flags.Value ==
                              expected_control_flags.Value,
                      "unexpected queue-control allocation policy");
    }
    context->expect(fake.maps.size() == 2, "queue did not make two GPU maps");
    if (fake.maps.size() == 2) {
      HsaMemMapFlags expected_map_flags{};
      expected_map_flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
      expected_map_flags.ui32.HostAccess = 1;
      context->expect(fake.maps[0].address == ring_memory.data() &&
                          fake.maps[0].size == kAqlRingDefaultSize &&
                          fake.maps[0].node == 7 &&
                          fake.maps[0].flags.Value == expected_map_flags.Value,
                      "unexpected AQL ring mapping policy");
      context->expect(fake.maps[1].address == control_memory.data() &&
                          fake.maps[1].size == kMemoryPageSize &&
                          fake.maps[1].node == 7 &&
                          fake.maps[1].flags.Value == expected_map_flags.Value,
                      "unexpected queue-control mapping policy");
    }

    bool ring_initialized = true;
    for (size_t offset = 0; offset < ring_memory.size();
         offset += static_cast<size_t>(kAqlPacketSize)) {
      ring_initialized = ring_initialized && ring_memory[offset] == 1;
      ring_initialized =
          ring_initialized &&
          std::all_of(ring_memory.data() + offset + 1,
                      ring_memory.data() + offset + kAqlPacketSize,
                      [](uint8_t value) { return value == 0; });
    }
    context->expect(ring_initialized,
                    "ring packets were not initialized as invalid");
    context->expect(std::all_of(control_memory.begin(), control_memory.end(),
                                [](uint8_t value) { return value == 0; }),
                    "queue indexes were not zero-initialized");

    context->expect(
        fake.queue_node == 7 && fake.queue_type == HSA_QUEUE_COMPUTE_AQL &&
            fake.queue_percentage == 100 &&
            fake.queue_priority == HSA_QUEUE_PRIORITY_NORMAL &&
            fake.queue_address == reinterpret_cast<void *>(kRingGpuAddress) &&
            fake.queue_size == kAqlRingDefaultSize &&
            fake.queue_event == nullptr,
        "unexpected hsaKmtCreateQueue arguments");
    context->expect(fake.queue_resource_input.QueueRptrValue ==
                            kControlGpuAddress &&
                        fake.queue_resource_input.QueueWptrValue ==
                            kControlGpuAddress + 64 &&
                        fake.queue_resource_input.ErrorReason == nullptr,
                    "unexpected AQL index addresses");

    const auto released = created.queue.release();
    context->expect(static_cast<bool>(released), released.message);
    expect_calls(context, {"open", "acquire", "allocate:ring", "map:ring:7",
                           "allocate:control", "map:control:7",
                           "create:7:65536", "destroy", "unmap:control",
                           "free:control", "unmap:ring", "free:ring"});
  }
  context->expect(fake.calls.size() >= 2 &&
                      fake.calls[fake.calls.size() - 2] == "release" &&
                      fake.calls.back() == "close",
                  "session did not close after queue cleanup");
}

void invalid_inputs_do_not_allocate(TestContext *context) {
  reset_fake();
  light_rocr::transport::hsakmt::KfdSession invalid;
  auto created = invalid.create_aql_queue(1, kAqlRingDefaultSize);
  context->expect(
      created.status.error ==
          light_rocr::transport::hsakmt::AqlQueueError::InvalidSession,
      "invalid session was accepted");

  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const std::array<uint64_t, 5> invalid_sizes = {
        0, 1024, 4095, 12288,
        2 * light_rocr::transport::hsakmt::kAqlRingMaximumSize};
    for (const uint64_t size : invalid_sizes) {
      created = opened.session.create_aql_queue(1, size);
      context->expect(
          created.status.error ==
              light_rocr::transport::hsakmt::AqlQueueError::InvalidRingSize,
          "invalid ring size was accepted");
    }
    expect_calls(context, {"open", "acquire"});
  }
}

void ring_allocation_failure_stops_creation(TestContext *context) {
  reset_fake();
  fake.fail_allocation_call = 1;
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto created =
        opened.session.create_aql_queue(3, kAqlRingDefaultSize);
    context->expect(!created, "ring allocation failure unexpectedly succeeded");
    context->expect(
        created.status.error ==
            light_rocr::transport::hsakmt::AqlQueueError::AllocateRing,
        "wrong ring allocation failure");
  }
  expect_calls(context,
               {"open", "acquire", "allocate:ring", "release", "close"});
}

void control_allocation_failure_releases_ring(TestContext *context) {
  reset_fake();
  fake.fail_allocation_call = 2;
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto created =
        opened.session.create_aql_queue(3, kAqlRingDefaultSize);
    context->expect(!created,
                    "control allocation failure unexpectedly succeeded");
    context->expect(
        created.status.error ==
            light_rocr::transport::hsakmt::AqlQueueError::AllocateControl,
        "wrong control allocation failure");
  }
  expect_calls(context, {"open", "acquire", "allocate:ring", "map:ring:3",
                         "allocate:control", "unmap:ring", "free:ring",
                         "release", "close"});
}

void create_failure_releases_both_allocations(TestContext *context) {
  reset_fake();
  fake.create_status = HSAKMT_STATUS_OUT_OF_RESOURCES;
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    const auto created =
        opened.session.create_aql_queue(5, kAqlRingDefaultSize);
    context->expect(!created, "queue creation failure unexpectedly succeeded");
    context->expect(
        created.status.error ==
            light_rocr::transport::hsakmt::AqlQueueError::CreateQueue,
        "wrong queue creation failure");
  }
  expect_calls(context, {"open", "acquire", "allocate:ring", "map:ring:5",
                         "allocate:control", "map:control:5", "create:5:65536",
                         "unmap:control", "free:control", "unmap:ring",
                         "free:ring", "release", "close"});
}

void destroy_failure_is_retryable(TestContext *context) {
  reset_fake();
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto created = opened.session.create_aql_queue(2, kAqlRingDefaultSize);
    fake.destroy_status = HSAKMT_STATUS_ERROR;
    auto released = created.queue.release();
    context->expect(
        released.error ==
            light_rocr::transport::hsakmt::AqlQueueError::DestroyQueue,
        "destroy failure was not reported");
    context->expect(static_cast<bool>(created.queue),
                    "failed destroy discarded the live queue");
    context->expect(fake.calls.back() == "destroy",
                    "memory was released after failed queue destroy");

    fake.destroy_status = HSAKMT_STATUS_SUCCESS;
    released = created.queue.release();
    context->expect(static_cast<bool>(released), released.message);
  }
  expect_calls(context, {"open", "acquire", "allocate:ring", "map:ring:2",
                         "allocate:control", "map:control:2", "create:2:65536",
                         "destroy", "destroy", "unmap:control", "free:control",
                         "unmap:ring", "free:ring", "release", "close"});
}

void partial_cleanup_keeps_index_accessors_safe(TestContext *context) {
  reset_fake();
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto created = opened.session.create_aql_queue(2, kAqlRingDefaultSize);
    fake.fail_unmap_call = 2;
    auto released = created.queue.release();
    context->expect(
        released.error ==
            light_rocr::transport::hsakmt::AqlQueueError::ReleaseRing,
        "ring cleanup failure was not reported");
    context->expect(!static_cast<bool>(created.queue),
                    "partially released queue remained active");
    context->expect(created.queue.read_index_acquire() == 0 &&
                        created.queue.write_index_relaxed() == 0,
                    "index accessors touched released control storage");

    fake.fail_unmap_call = 0;
    released = created.queue.release();
    context->expect(static_cast<bool>(released), released.message);
  }
}

void queue_keeps_session_open(TestContext *context) {
  reset_fake();
  light_rocr::transport::hsakmt::AqlQueue queue;
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    auto created = opened.session.create_aql_queue(4, kAqlRingDefaultSize);
    queue = std::move(created.queue);
  }
  context->expect(fake.calls.back() == "create:4:65536",
                  "KFD closed while a queue was live");
  const auto released = queue.release();
  context->expect(static_cast<bool>(released), released.message);
  context->expect(fake.calls.size() >= 2 &&
                      fake.calls[fake.calls.size() - 2] == "release" &&
                      fake.calls.back() == "close",
                  "session did not close after the surviving queue");
}

void destructor_cleans_up_in_order(TestContext *context) {
  reset_fake();
  {
    auto opened = light_rocr::transport::hsakmt::KfdSession::open();
    {
      const auto created =
          opened.session.create_aql_queue(6, kAqlRingDefaultSize);
      context->expect(static_cast<bool>(created), created.status.message);
    }
  }
  expect_calls(context, {"open", "acquire", "allocate:ring", "map:ring:6",
                         "allocate:control", "map:control:6", "create:6:65536",
                         "destroy", "unmap:control", "free:control",
                         "unmap:ring", "free:ring", "release", "close"});
}

light_rocr::runtime::AqlKernelDispatchPacket valid_packet() {
  light_rocr::runtime::KernelDispatchSpec spec;
  spec.kernel_object = 0x500000;
  spec.kernarg_address = 0x600000;
  spec.completion_signal = 0x700000;
  return light_rocr::runtime::make_kernel_dispatch_packet(spec).packet;
}

void publishes_packet_and_rings_doorbell(TestContext *context) {
  reset_fake();
  auto opened = light_rocr::transport::hsakmt::KfdSession::open();
  auto created = opened.session.create_aql_queue(7, kAqlRingDefaultSize);
  const auto packet = valid_packet();
  const auto submitted = created.queue.submit_kernel_dispatch(packet);
  context->expect(static_cast<bool>(submitted), submitted.message);
  context->expect(submitted.packet_id == 0 && submitted.read_index == 0 &&
                      submitted.write_index == 1,
                  "first submission returned incorrect queue indexes");

  uint64_t observed_write_index = 0;
  std::memcpy(&observed_write_index, control_memory.data() + 64,
              sizeof(observed_write_index));
  context->expect(observed_write_index == 1,
                  "queue write index was not advanced");
  context->expect(doorbell == 0,
                  "first submission did not ring packet ID zero");

  light_rocr::runtime::AqlKernelDispatchPacket observed{};
  std::memcpy(&observed, ring_memory.data(), sizeof(observed));
  context->expect(observed.header == packet.header &&
                      observed.setup == packet.setup &&
                      observed.kernel_object == packet.kernel_object &&
                      observed.kernarg_address == packet.kernarg_address &&
                      observed.completion_signal == packet.completion_signal,
                  "published ring packet does not match the source packet");
  context->expect(created.queue.read_index_acquire() == 0 &&
                      created.queue.write_index_relaxed() == 1,
                  "queue index accessors returned incorrect values");
}

void rejects_invalid_packet_without_publication(TestContext *context) {
  reset_fake();
  auto opened = light_rocr::transport::hsakmt::KfdSession::open();
  auto created = opened.session.create_aql_queue(7, kAqlRingDefaultSize);
  auto packet = valid_packet();
  packet.reserved2 = 1;
  doorbell = UINT64_MAX;
  const auto submitted = created.queue.submit_kernel_dispatch(packet);
  context->expect(
      submitted.error ==
          light_rocr::transport::hsakmt::AqlSubmitError::InvalidPacket,
      "invalid packet was accepted");
  context->expect(created.queue.write_index_relaxed() == 0 &&
                      doorbell == UINT64_MAX && ring_memory[0] == 1,
                  "invalid packet changed the ring, index, or doorbell");
}

void reports_full_queue_without_publication(TestContext *context) {
  reset_fake();
  auto opened = light_rocr::transport::hsakmt::KfdSession::open();
  auto created = opened.session.create_aql_queue(7, kAqlRingDefaultSize);
  const uint64_t packet_count = kAqlRingDefaultSize / kAqlPacketSize;
  std::memcpy(control_memory.data() + 64, &packet_count, sizeof(packet_count));
  doorbell = UINT64_MAX;
  const auto submitted = created.queue.submit_kernel_dispatch(valid_packet());
  context->expect(submitted.error ==
                      light_rocr::transport::hsakmt::AqlSubmitError::QueueFull,
                  "full queue accepted a packet");
  context->expect(submitted.read_index == 0 &&
                      submitted.write_index == packet_count &&
                      doorbell == UINT64_MAX && ring_memory[0] == 1,
                  "full queue changed the ring or doorbell");
}

void submit_error_names(TestContext *context) {
  context->expect(
      std::string(light_rocr::transport::hsakmt::aql_submit_error_name(
          light_rocr::transport::hsakmt::AqlSubmitError::QueueFull)) ==
          "queue_full",
      "unexpected submit error name");
}

void inactive_queue_rejects_submission(TestContext *context) {
  light_rocr::transport::hsakmt::AqlQueue queue;
  const auto submitted = queue.submit_kernel_dispatch(valid_packet());
  context->expect(
      submitted.error ==
          light_rocr::transport::hsakmt::AqlSubmitError::InvalidQueue,
      "inactive queue accepted a packet");
}

} // namespace

extern "C" HSAKMT_STATUS hsaKmtOpenKFD() {
  fake.calls.emplace_back("open");
  return fake.open_status;
}

extern "C" HSAKMT_STATUS hsaKmtCloseKFD() {
  fake.calls.emplace_back("close");
  return HSAKMT_STATUS_SUCCESS;
}

extern "C" HSAKMT_STATUS
hsaKmtAcquireSystemProperties(HsaSystemProperties *properties) {
  fake.calls.emplace_back("acquire");
  if (fake.acquire_status == HSAKMT_STATUS_SUCCESS) {
    properties->NumNodes = 2;
  }
  return fake.acquire_status;
}

extern "C" HSAKMT_STATUS hsaKmtReleaseSystemProperties() {
  fake.calls.emplace_back("release");
  return HSAKMT_STATUS_SUCCESS;
}

extern "C" HSAKMT_STATUS hsaKmtAllocMemory(HSAuint32 preferred_node,
                                           HSAuint64 size, HsaMemFlags flags,
                                           void **address) {
  const size_t call_number = fake.allocations.size() + 1;
  void *result = call_number == 1 ? static_cast<void *>(ring_memory.data())
                                  : static_cast<void *>(control_memory.data());
  fake.allocations.push_back({preferred_node, size, flags, result});
  fake.calls.push_back(std::string("allocate:") + allocation_name(result));
  if (fake.fail_allocation_call == call_number) {
    return HSAKMT_STATUS_NO_MEMORY;
  }
  *address = result;
  return HSAKMT_STATUS_SUCCESS;
}

extern "C" HSAKMT_STATUS hsaKmtMapMemoryToGPUNodes(
    void *address, HSAuint64 size, HSAuint64 *alternate_gpu_address,
    HsaMemMapFlags flags, HSAuint64 node_count, HSAuint32 *nodes) {
  if (node_count != 1) {
    return HSAKMT_STATUS_INVALID_PARAMETER;
  }
  const uint32_t node = nodes[0];
  fake.maps.push_back({address, size, flags, node});
  fake.calls.push_back(std::string("map:") + allocation_name(address) + ":" +
                       std::to_string(node));
  *alternate_gpu_address =
      address == ring_memory.data() ? kRingGpuAddress : kControlGpuAddress;
  return HSAKMT_STATUS_SUCCESS;
}

extern "C" HSAKMT_STATUS hsaKmtUnmapMemoryToGPU(void *address) {
  fake.calls.push_back(std::string("unmap:") + allocation_name(address));
  ++fake.unmap_calls;
  if (fake.fail_unmap_call == fake.unmap_calls) {
    return HSAKMT_STATUS_ERROR;
  }
  return HSAKMT_STATUS_SUCCESS;
}

extern "C" HSAKMT_STATUS hsaKmtFreeMemory(void *address, HSAuint64) {
  fake.calls.push_back(std::string("free:") + allocation_name(address));
  return HSAKMT_STATUS_SUCCESS;
}

extern "C" HSAKMT_STATUS
hsaKmtCreateQueue(HSAuint32 node_id, HSA_QUEUE_TYPE type,
                  HSAuint32 queue_percentage, HSA_QUEUE_PRIORITY priority,
                  void *queue_address, HSAuint64 queue_size_in_bytes,
                  HsaEvent *event, HsaQueueResource *queue_resource) {
  fake.calls.push_back("create:" + std::to_string(node_id) + ":" +
                       std::to_string(queue_size_in_bytes));
  fake.queue_node = node_id;
  fake.queue_type = type;
  fake.queue_percentage = queue_percentage;
  fake.queue_priority = priority;
  fake.queue_address = queue_address;
  fake.queue_size = queue_size_in_bytes;
  fake.queue_event = event;
  fake.queue_resource_input = *queue_resource;
  if (fake.create_status == HSAKMT_STATUS_SUCCESS) {
    queue_resource->QueueId = kQueueId;
    queue_resource->Queue_DoorBell_aql = &doorbell;
  }
  return fake.create_status;
}

extern "C" HSAKMT_STATUS hsaKmtDestroyQueue(HSA_QUEUEID queue_id) {
  fake.calls.emplace_back("destroy");
  if (queue_id != kQueueId) {
    return HSAKMT_STATUS_INVALID_HANDLE;
  }
  return fake.destroy_status;
}

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"successful_round_trip", successful_round_trip},
      {"invalid_inputs_do_not_allocate", invalid_inputs_do_not_allocate},
      {"ring_allocation_failure_stops_creation",
       ring_allocation_failure_stops_creation},
      {"control_allocation_failure_releases_ring",
       control_allocation_failure_releases_ring},
      {"create_failure_releases_both_allocations",
       create_failure_releases_both_allocations},
      {"destroy_failure_is_retryable", destroy_failure_is_retryable},
      {"partial_cleanup_keeps_index_accessors_safe",
       partial_cleanup_keeps_index_accessors_safe},
      {"queue_keeps_session_open", queue_keeps_session_open},
      {"destructor_cleans_up_in_order", destructor_cleans_up_in_order},
      {"publishes_packet_and_rings_doorbell",
       publishes_packet_and_rings_doorbell},
      {"rejects_invalid_packet_without_publication",
       rejects_invalid_packet_without_publication},
      {"reports_full_queue_without_publication",
       reports_full_queue_without_publication},
      {"submit_error_names", submit_error_names},
      {"inactive_queue_rejects_submission", inactive_queue_rejects_submission},
  };

  TestContext context;
  for (const auto &test : tests) {
    std::cout << "[ RUN      ] " << test.first << '\n';
    const int failures_before = context.failures;
    test.second(&context);
    if (context.failures == failures_before) {
      std::cout << "[       OK ] " << test.first << '\n';
    } else {
      std::cout << "[  FAILED  ] " << test.first << '\n';
    }
  }

  if (context.failures != 0) {
    std::cerr << context.failures << " assertion(s) failed\n";
    return 1;
  }
  std::cout << tests.size() << " test(s) passed\n";
  return 0;
}
