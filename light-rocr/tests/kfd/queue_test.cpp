#include "queue_internal.hpp"

#include <linux/kfd_ioctl.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kQueueId = 37;
constexpr uint64_t kDoorbellMappingOffset = 0xea46000000000000ULL;
constexpr uint64_t kDoorbellOffset = 0x38;
constexpr uintptr_t kDoorbellMappingAddress = 0x500000;

struct FakeSystem {
  bool create_fails = false;
  bool map_fails = false;
  bool destroy_fails = false;
  bool unmap_fails = false;
  bool advise_fails = false;
  bool contract_valid = true;
  std::vector<std::string> calls;
};

FakeSystem *fake = nullptr;

int fake_ioctl(int fd, unsigned long request, void *arguments) {
  if (fake == nullptr || fd != 17) {
    errno = EINVAL;
    return -1;
  }
  if (request == AMDKFD_IOC_CREATE_QUEUE) {
    auto *args = static_cast<kfd_ioctl_create_queue_args *>(arguments);
    fake->contract_valid =
        fake->contract_valid && args->ring_base_address == 0x100000 &&
        args->write_pointer_address == 0x200038 &&
        args->read_pointer_address == 0x200080 && args->ring_size == 65536 &&
        args->gpu_id == 42 &&
        args->queue_type == KFD_IOC_QUEUE_TYPE_COMPUTE_AQL &&
        args->queue_percentage == KFD_MAX_QUEUE_PERCENTAGE &&
        args->queue_priority == 7 && args->eop_buffer_address == 0x300000 &&
        args->eop_buffer_size == 4096 &&
        args->ctx_save_restore_address == 0x400000 &&
        args->ctx_save_restore_size == 0x1b72000 &&
        args->ctl_stack_size == 0x6000;
    fake->calls.emplace_back("create");
    if (fake->create_fails) {
      errno = EINVAL;
      return -1;
    }
    args->queue_id = kQueueId;
    args->doorbell_offset = kDoorbellMappingOffset + kDoorbellOffset;
    return 0;
  }
  if (request == AMDKFD_IOC_DESTROY_QUEUE) {
    const auto *args = static_cast<kfd_ioctl_destroy_queue_args *>(arguments);
    fake->contract_valid = fake->contract_valid && args->queue_id == kQueueId;
    fake->calls.emplace_back("destroy");
    if (fake->destroy_fails) {
      errno = EBUSY;
      return -1;
    }
    return 0;
  }
  fake->contract_valid = false;
  errno = EINVAL;
  return -1;
}

void *fake_mmap(void *address, size_t length, int protection, int flags, int fd,
                off_t offset) {
  fake->contract_valid =
      fake->contract_valid &&
      address == reinterpret_cast<void *>(kDoorbellMappingAddress) &&
      length == 8192 && protection == (PROT_READ | PROT_WRITE) &&
      flags == (MAP_SHARED | MAP_FIXED) && fd == 17 &&
      offset == static_cast<off_t>(kDoorbellMappingOffset);
  fake->calls.emplace_back("map");
  if (fake->map_fails) {
    errno = ENXIO;
    return MAP_FAILED;
  }
  return reinterpret_cast<void *>(kDoorbellMappingAddress);
}

int fake_munmap(void *address, size_t length) {
  fake->contract_valid =
      fake->contract_valid &&
      address == reinterpret_cast<void *>(kDoorbellMappingAddress) &&
      length == 8192;
  fake->calls.emplace_back("unmap");
  if (fake->unmap_fails) {
    errno = EBUSY;
    return -1;
  }
  return 0;
}

int fake_madvise(void *address, size_t length, int advice) {
  fake->contract_valid =
      fake->contract_valid &&
      address == reinterpret_cast<void *>(kDoorbellMappingAddress) &&
      length == 8192 && advice == MADV_DONTFORK;
  fake->calls.emplace_back("advise");
  if (fake->advise_fails) {
    errno = EINVAL;
    return -1;
  }
  return 0;
}

light_rocr::transport::kfd::detail::QueueSyscalls syscalls() {
  return {fake_ioctl, fake_mmap, fake_munmap, fake_madvise};
}

light_rocr::transport::kfd::detail::AqlQueueCreateInfo create_info() {
  light_rocr::transport::kfd::detail::AqlQueueCreateInfo info;
  info.gpu_id = 42;
  info.ring_address = 0x100000;
  info.ring_size = 65536;
  info.read_pointer_address = 0x200080;
  info.write_pointer_address = 0x200038;
  info.eop_buffer_address = 0x300000;
  info.eop_buffer_size = 4096;
  info.context_save_restore_address = 0x400000;
  info.context_save_restore_size = 0x1b72000;
  info.control_stack_size = 0x6000;
  info.doorbell_size = 8;
  info.doorbell_mapping_address =
      reinterpret_cast<void *>(kDoorbellMappingAddress);
  return info;
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

void successful_round_trip(TestContext *context) {
  FakeSystem state;
  fake = &state;
  auto created = light_rocr::transport::kfd::detail::create_aql_queue(
      17, create_info(), syscalls());
  context->expect(created && created.queue.active &&
                      created.queue.queue_id == kQueueId &&
                      created.queue.doorbell_mapping ==
                          reinterpret_cast<void *>(kDoorbellMappingAddress) &&
                      created.queue.doorbell_address ==
                          kDoorbellMappingAddress + kDoorbellOffset,
                  created.status.message);
  auto released = light_rocr::transport::kfd::detail::release_aql_queue(
      17, &created.queue, syscalls());
  context->expect(released && !created.queue.active &&
                      created.queue.doorbell_mapping == nullptr &&
                      created.queue.doorbell_address == 0,
                  released.message);
  context->expect(state.contract_valid, "KFD queue syscall contract changed");
  context->expect(state.calls == std::vector<std::string>{"create", "map",
                                                          "advise", "destroy",
                                                          "unmap"},
                  "queue lifecycle order changed");
  fake = nullptr;
}

void create_failure_has_no_owned_queue(TestContext *context) {
  FakeSystem state;
  state.create_fails = true;
  fake = &state;
  auto created = light_rocr::transport::kfd::detail::create_aql_queue(
      17, create_info(), syscalls());
  context->expect(
      !created &&
          created.status.error ==
              light_rocr::transport::kfd::AqlQueueError::CreateQueue &&
          !created.queue.active && created.queue.doorbell_mapping == nullptr,
      "failed queue ioctl retained ownership");
  context->expect(state.calls == std::vector<std::string>{"create"},
                  "failed queue ioctl performed cleanup on an unowned queue");
  fake = nullptr;
}

void map_failure_destroys_queue(TestContext *context) {
  FakeSystem state;
  state.map_fails = true;
  fake = &state;
  auto created = light_rocr::transport::kfd::detail::create_aql_queue(
      17, create_info(), syscalls());
  context->expect(
      !created &&
          created.status.error ==
              light_rocr::transport::kfd::AqlQueueError::MapDoorbell &&
          !created.queue.active,
      "doorbell map failure retained a destroyed queue");
  context->expect(state.calls ==
                      std::vector<std::string>{"create", "map", "destroy"},
                  "doorbell map rollback order changed");
  fake = nullptr;
}

void failed_rollback_is_retryable(TestContext *context) {
  FakeSystem state;
  state.map_fails = true;
  state.destroy_fails = true;
  fake = &state;
  auto created = light_rocr::transport::kfd::detail::create_aql_queue(
      17, create_info(), syscalls());
  context->expect(!created && created.queue.active &&
                      created.status.message.find("cleanup failed") !=
                          std::string::npos,
                  "destroy failure discarded the live KFD queue");

  state.destroy_fails = false;
  const auto released = light_rocr::transport::kfd::detail::release_aql_queue(
      17, &created.queue, syscalls());
  context->expect(released && !created.queue.active,
                  "retained queue could not be destroyed on retry");
  context->expect(state.calls == std::vector<std::string>{"create", "map",
                                                          "destroy", "destroy"},
                  "destroy retry lifecycle changed");
  fake = nullptr;
}

void madvise_failure_releases_mapping_and_queue(TestContext *context) {
  FakeSystem state;
  state.advise_fails = true;
  fake = &state;
  auto created = light_rocr::transport::kfd::detail::create_aql_queue(
      17, create_info(), syscalls());
  context->expect(
      !created &&
          created.status.error ==
              light_rocr::transport::kfd::AqlQueueError::MapDoorbell &&
          !created.queue.active && created.queue.doorbell_mapping == nullptr,
      "doorbell madvise failure retained queue resources");
  context->expect(state.calls == std::vector<std::string>{"create", "map",
                                                          "advise", "destroy",
                                                          "unmap"},
                  "doorbell madvise rollback order changed");
  fake = nullptr;
}

void unmap_failure_does_not_redestroy(TestContext *context) {
  FakeSystem state;
  fake = &state;
  auto created = light_rocr::transport::kfd::detail::create_aql_queue(
      17, create_info(), syscalls());
  state.unmap_fails = true;
  auto released = light_rocr::transport::kfd::detail::release_aql_queue(
      17, &created.queue, syscalls());
  context->expect(!released && !created.queue.active &&
                      created.queue.doorbell_mapping != nullptr,
                  "doorbell unmap failure discarded retry state");

  state.unmap_fails = false;
  released = light_rocr::transport::kfd::detail::release_aql_queue(
      17, &created.queue, syscalls());
  context->expect(released && created.queue.doorbell_mapping == nullptr,
                  "doorbell unmap retry did not complete");
  context->expect(state.calls == std::vector<std::string>{"create", "map",
                                                          "advise", "destroy",
                                                          "unmap", "unmap"},
                  "doorbell unmap retry destroyed the queue twice");
  fake = nullptr;
}

void invalid_inputs_issue_no_syscall(TestContext *context) {
  FakeSystem state;
  fake = &state;
  auto info = create_info();
  info.ring_size = 4095;
  auto created = light_rocr::transport::kfd::detail::create_aql_queue(
      17, info, syscalls());
  context->expect(
      !created &&
          created.status.error ==
              light_rocr::transport::kfd::AqlQueueError::InvalidRingSize,
      "invalid ring size was accepted");
  info = create_info();
  info.doorbell_size = 16;
  created = light_rocr::transport::kfd::detail::create_aql_queue(17, info,
                                                                 syscalls());
  context->expect(
      !created &&
          created.status.error ==
              light_rocr::transport::kfd::AqlQueueError::InvalidDoorbell,
      "invalid doorbell size was accepted");
  info = create_info();
  info.context_save_restore_address = 0;
  created = light_rocr::transport::kfd::detail::create_aql_queue(17, info,
                                                                 syscalls());
  context->expect(
      !created && created.status.error ==
                      light_rocr::transport::kfd::AqlQueueError::InvalidNode,
      "missing CWSR allocation was accepted");
  context->expect(state.calls.empty(), "invalid input issued a syscall");
  fake = nullptr;
}

void gfx1101_cwsr_layout(TestContext *context) {
  light_rocr::runtime::Node node;
  node.simd_count = 120;
  node.simd_per_compute_unit = 2;
  node.maximum_waves_per_simd = 16;
  node.lds_size_kb = 64;
  node.xcc_count = 1;
  node.architecture = {11, 0, 1};

  light_rocr::transport::kfd::detail::CwsrLayout layout;
  const auto status =
      light_rocr::transport::kfd::detail::compute_cwsr_layout(node, &layout);
  context->expect(status && layout.control_stack_size == 0x6000 &&
                      layout.context_save_restore_size == 0x1b72000 &&
                      layout.debug_memory_size == 0xf000 &&
                      layout.allocation_size == 0x1b81000,
                  status.message.empty() ? "unexpected gfx1101 CWSR layout"
                                         : status.message);
  if (!status) {
    return;
  }

  std::vector<uint8_t> storage(static_cast<size_t>(layout.allocation_size));
  light_rocr::transport::kfd::detail::initialize_cwsr(storage.data(), layout);
  uint32_t debug_offset = 0;
  uint32_t debug_size = 0;
  std::memcpy(&debug_offset, storage.data() + 16, sizeof(debug_offset));
  std::memcpy(&debug_size, storage.data() + 20, sizeof(debug_size));
  context->expect(debug_offset == layout.context_save_restore_size &&
                      debug_size == layout.debug_memory_size,
                  "CWSR header does not describe the debugger area");
}

void kernel_cwsr_sizes_take_precedence(TestContext *context) {
  light_rocr::runtime::Node node;
  node.simd_count = 120;
  node.simd_per_compute_unit = 2;
  node.maximum_waves_per_simd = 16;
  node.lds_size_kb = 64;
  node.xcc_count = 1;
  node.cwsr_size = 0x2000000;
  node.control_stack_size = 0x8000;
  node.architecture = {11, 0, 1};

  light_rocr::transport::kfd::detail::CwsrLayout layout;
  const auto status =
      light_rocr::transport::kfd::detail::compute_cwsr_layout(node, &layout);
  context->expect(status && layout.context_save_restore_size == 0x2000000 &&
                      layout.control_stack_size == 0x8000,
                  "kernel CWSR sizes were not preferred");
}

void inactive_queue_rejects_producer_operations(TestContext *context) {
  light_rocr::transport::kfd::AqlQueue queue;
  const auto reserved = queue.add_write_index_scacq_screl(1);
  const auto doorbell = queue.store_doorbell_screlease(0);
  context->expect(
      !reserved &&
          reserved.status.error ==
              light_rocr::transport::kfd::AqlQueuePrimitiveError::InvalidQueue,
      "inactive queue accepted a write-index reservation");
  context->expect(
      !doorbell &&
          doorbell.error ==
              light_rocr::transport::kfd::AqlQueuePrimitiveError::InvalidQueue,
      "inactive queue accepted a doorbell store");
}

} // namespace

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"successful round trip", successful_round_trip},
      {"create failure", create_failure_has_no_owned_queue},
      {"map failure", map_failure_destroys_queue},
      {"failed rollback", failed_rollback_is_retryable},
      {"madvise failure", madvise_failure_releases_mapping_and_queue},
      {"unmap retry", unmap_failure_does_not_redestroy},
      {"invalid inputs", invalid_inputs_issue_no_syscall},
      {"gfx1101 CWSR layout", gfx1101_cwsr_layout},
      {"kernel CWSR sizes", kernel_cwsr_sizes_take_precedence},
      {"inactive producer operations",
       inactive_queue_rejects_producer_operations},
  };

  int failures = 0;
  for (const auto &[name, test] : tests) {
    TestContext context;
    test(&context);
    if (context.failures == 0) {
      std::cout << "PASS: " << name << '\n';
    } else {
      std::cerr << "FAIL: " << name << " (" << context.failures << ")\n";
      failures += context.failures;
    }
  }
  return failures == 0 ? 0 : 1;
}
