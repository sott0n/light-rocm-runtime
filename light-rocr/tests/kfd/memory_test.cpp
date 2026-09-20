#include "light_rocr/transport/kfd/memory.hpp"

#include "memory_internal.hpp"

#include <linux/kfd_ioctl.h>

#include <cerrno>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr uintptr_t kReservationAddress = 0x100000;
constexpr uintptr_t kHostAddress = kReservationAddress + 4096;
constexpr uint64_t kHandle = 0x1234;
constexpr uint64_t kMmapOffset = 0x200000;

static_assert(
    !std::is_move_assignable_v<light_rocr::transport::kfd::GttAllocation>);

struct FakeSystem {
  bool allocate_fails = false;
  bool host_mmap_fails = false;
  bool madvise_fails = false;
  bool map_fails = false;
  bool unmap_gpu_fails = false;
  bool munmap_fails = false;
  bool free_fails = false;
  uint32_t map_result_n_success = std::numeric_limits<uint32_t>::max();
  uint32_t unmap_result_n_success = std::numeric_limits<uint32_t>::max();
  std::vector<uint32_t> expected_gpu_ids{42};
  std::vector<uint32_t> map_starts;
  std::vector<uint32_t> unmap_starts;
  bool contract_valid = true;
  unsigned mmap_calls = 0;
  std::vector<std::string> calls;
};

FakeSystem *fake = nullptr;

int fake_ioctl(int fd, unsigned long request, void *arguments) {
  if (fake == nullptr || fd != 17) {
    errno = EINVAL;
    return -1;
  }
  if (request == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU) {
    auto *args = static_cast<kfd_ioctl_alloc_memory_of_gpu_args *>(arguments);
    const uint32_t expected_flags = static_cast<uint32_t>(
        KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
        KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
        KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE);
    fake->contract_valid = fake->contract_valid &&
                           args->va_addr == kHostAddress &&
                           args->size == 8192 && args->gpu_id == 42 &&
                           args->flags == expected_flags;
    fake->calls.emplace_back("allocate");
    if (fake->allocate_fails) {
      errno = ENOMEM;
      return -1;
    }
    args->handle = kHandle;
    args->mmap_offset = kMmapOffset;
    return 0;
  }
  if (request == AMDKFD_IOC_FREE_MEMORY_OF_GPU) {
    const auto *args =
        static_cast<kfd_ioctl_free_memory_of_gpu_args *>(arguments);
    fake->contract_valid = fake->contract_valid && args->handle == kHandle;
    fake->calls.emplace_back("free");
    if (fake->free_fails) {
      errno = EIO;
      return -1;
    }
    return 0;
  }
  if (request == AMDKFD_IOC_MAP_MEMORY_TO_GPU) {
    auto *args = static_cast<kfd_ioctl_map_memory_to_gpu_args *>(arguments);
    const auto *gpu_ids = reinterpret_cast<const uint32_t *>(
        static_cast<uintptr_t>(args->device_ids_array_ptr));
    fake->contract_valid =
        fake->contract_valid && args->handle == kHandle && gpu_ids != nullptr &&
        args->n_devices == fake->expected_gpu_ids.size() &&
        std::vector<uint32_t>(gpu_ids, gpu_ids + args->n_devices) ==
            fake->expected_gpu_ids;
    fake->calls.emplace_back("map_gpu");
    fake->map_starts.push_back(args->n_success);
    args->n_success =
        fake->map_result_n_success == std::numeric_limits<uint32_t>::max()
            ? args->n_devices
            : fake->map_result_n_success;
    if (fake->map_fails) {
      errno = EBUSY;
      return -1;
    }
    return 0;
  }
  if (request == AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU) {
    auto *args = static_cast<kfd_ioctl_unmap_memory_from_gpu_args *>(arguments);
    const auto *gpu_ids = reinterpret_cast<const uint32_t *>(
        static_cast<uintptr_t>(args->device_ids_array_ptr));
    fake->contract_valid =
        fake->contract_valid && args->handle == kHandle && gpu_ids != nullptr &&
        args->n_devices <= fake->expected_gpu_ids.size() &&
        std::vector<uint32_t>(gpu_ids, gpu_ids + args->n_devices) ==
            std::vector<uint32_t>(fake->expected_gpu_ids.begin(),
                                  fake->expected_gpu_ids.begin() +
                                      args->n_devices);
    fake->calls.emplace_back("unmap_gpu");
    fake->unmap_starts.push_back(args->n_success);
    args->n_success =
        fake->unmap_result_n_success == std::numeric_limits<uint32_t>::max()
            ? args->n_devices
            : fake->unmap_result_n_success;
    if (fake->unmap_gpu_fails) {
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
  if (fake == nullptr) {
    errno = EINVAL;
    return MAP_FAILED;
  }
  ++fake->mmap_calls;
  if (fake->mmap_calls == 1) {
    fake->contract_valid =
        fake->contract_valid && address == nullptr && length == 16384 &&
        protection == PROT_NONE &&
        flags == (MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE) && fd == -1 &&
        offset == 0;
    fake->calls.emplace_back("reserve");
    return reinterpret_cast<void *>(kReservationAddress);
  }
  fake->contract_valid = fake->contract_valid &&
                         address == reinterpret_cast<void *>(kHostAddress) &&
                         length == 8192 &&
                         protection == (PROT_READ | PROT_WRITE) &&
                         flags == (MAP_SHARED | MAP_FIXED) && fd == 23 &&
                         offset == static_cast<off_t>(kMmapOffset);
  fake->calls.emplace_back("map_host");
  if (fake->host_mmap_fails) {
    errno = ENXIO;
    return MAP_FAILED;
  }
  return address;
}

int fake_munmap(void *address, size_t length) {
  if (fake == nullptr) {
    errno = EINVAL;
    return -1;
  }
  fake->contract_valid =
      fake->contract_valid &&
      address == reinterpret_cast<void *>(kReservationAddress) &&
      length == 16384;
  fake->calls.emplace_back("unmap");
  if (fake->munmap_fails) {
    errno = EBUSY;
    return -1;
  }
  return 0;
}

int fake_madvise(void *address, size_t length, int advice) {
  if (fake == nullptr) {
    errno = EINVAL;
    return -1;
  }
  fake->contract_valid = fake->contract_valid &&
                         address == reinterpret_cast<void *>(kHostAddress) &&
                         length == 8192 && advice == MADV_DONTFORK;
  fake->calls.emplace_back("dontfork");
  if (fake->madvise_fails) {
    errno = EINVAL;
    return -1;
  }
  return 0;
}

light_rocr::transport::kfd::detail::MemorySyscalls syscalls() {
  return {fake_ioctl, fake_mmap, fake_munmap, fake_madvise};
}

light_rocr::transport::kfd::ProcessAperture aperture() {
  light_rocr::transport::kfd::ProcessAperture result;
  result.gpu_id = 42;
  result.gpuvm_base = 0x4000;
  result.gpuvm_limit = 0x7fffffffffff;
  return result;
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
  auto allocated = light_rocr::transport::kfd::detail::allocate_gtt(
      17, 23, aperture(), 8192, syscalls());
  context->expect(static_cast<bool>(allocated), allocated.status.message);
  context->expect(state.contract_valid, "allocation syscall contract changed");
  context->expect(allocated.allocation.host_address ==
                          reinterpret_cast<void *>(kHostAddress) &&
                      allocated.allocation.size == 8192 &&
                      allocated.allocation.handle == kHandle &&
                      allocated.allocation.gpu_ids ==
                          std::vector<uint32_t>{42} &&
                      allocated.allocation.mapped_device_count == 1 &&
                      allocated.allocation.map_complete,
                  "allocation result was not retained");
  auto released = light_rocr::transport::kfd::detail::release_gtt(
      17, &allocated.allocation, syscalls());
  context->expect(static_cast<bool>(released), released.message);
  context->expect(
      state.calls == std::vector<std::string>{"reserve", "allocate", "map_host",
                                              "dontfork", "map_gpu",
                                              "unmap_gpu", "unmap", "free"},
      "unexpected successful allocation call order");
  context->expect(allocated.allocation.handle == 0,
                  "released allocation retained its handle");
  fake = nullptr;
}

void invalid_size_does_not_reserve(TestContext *context) {
  FakeSystem state;
  fake = &state;
  const auto allocated = light_rocr::transport::kfd::detail::allocate_gtt(
      17, 23, aperture(), 4095, syscalls());
  context->expect(!allocated, "unaligned allocation unexpectedly succeeded");
  context->expect(allocated.status.error ==
                      light_rocr::transport::kfd::MemoryError::InvalidSize,
                  "wrong invalid-size error");
  context->expect(state.calls.empty(), "invalid size issued a syscall");
  fake = nullptr;
}

void allocation_failure_releases_va(TestContext *context) {
  FakeSystem state;
  state.allocate_fails = true;
  fake = &state;
  const auto allocated = light_rocr::transport::kfd::detail::allocate_gtt(
      17, 23, aperture(), 8192, syscalls());
  context->expect(!allocated, "failed ioctl unexpectedly succeeded");
  context->expect(
      allocated.status.error ==
              light_rocr::transport::kfd::MemoryError::AllocateGtt &&
          allocated.status.system_error == ENOMEM,
      "allocation errno was not preserved");
  context->expect(state.calls ==
                      std::vector<std::string>{"reserve", "allocate", "unmap"},
                  "failed allocation did not release its VA");
  fake = nullptr;
}

void rollback_unmap_failure_retains_va(TestContext *context) {
  FakeSystem state;
  state.allocate_fails = true;
  state.munmap_fails = true;
  fake = &state;
  auto allocated = light_rocr::transport::kfd::detail::allocate_gtt(
      17, 23, aperture(), 8192, syscalls());
  context->expect(!allocated, "failed ioctl unexpectedly succeeded");
  context->expect(allocated.allocation.reservation_address ==
                          reinterpret_cast<void *>(kReservationAddress) &&
                      allocated.allocation.handle == 0,
                  "rollback munmap failure discarded the reserved VA");
  context->expect(allocated.status.message.find("cleanup failed") !=
                      std::string::npos,
                  "rollback munmap failure was not added to the diagnostic");

  state.munmap_fails = false;
  const auto released = light_rocr::transport::kfd::detail::release_gtt(
      17, &allocated.allocation, syscalls());
  context->expect(static_cast<bool>(released), released.message);
  context->expect(allocated.allocation.reservation_address == nullptr &&
                      allocated.allocation.handle == 0,
                  "retained VA could not be released");
  fake = nullptr;
}

void host_map_failure_frees_handle(TestContext *context) {
  FakeSystem state;
  state.host_mmap_fails = true;
  fake = &state;
  const auto allocated = light_rocr::transport::kfd::detail::allocate_gtt(
      17, 23, aperture(), 8192, syscalls());
  context->expect(!allocated, "failed host map unexpectedly succeeded");
  context->expect(allocated.status.error ==
                          light_rocr::transport::kfd::MemoryError::MapHost &&
                      allocated.status.system_error == ENXIO,
                  "host mmap errno was not preserved");
  context->expect(state.calls == std::vector<std::string>{"reserve", "allocate",
                                                          "map_host", "unmap",
                                                          "free"},
                  "host mmap failure cleanup order changed");
  fake = nullptr;
}

void cleanup_failure_retains_handle(TestContext *context) {
  FakeSystem state;
  state.host_mmap_fails = true;
  state.free_fails = true;
  fake = &state;
  auto allocated = light_rocr::transport::kfd::detail::allocate_gtt(
      17, 23, aperture(), 8192, syscalls());
  context->expect(!allocated, "failed host map unexpectedly succeeded");
  context->expect(allocated.allocation.host_address == nullptr &&
                      allocated.allocation.handle == kHandle,
                  "cleanup failure discarded the retryable KFD handle");
  context->expect(allocated.status.message.find("cleanup failed") !=
                      std::string::npos,
                  "cleanup failure was not added to the diagnostic");

  state.free_fails = false;
  const auto released = light_rocr::transport::kfd::detail::release_gtt(
      17, &allocated.allocation, syscalls());
  context->expect(static_cast<bool>(released), released.message);
  context->expect(allocated.allocation.handle == 0,
                  "retained cleanup handle could not be released");
  fake = nullptr;
}

void dontfork_failure_cleans_up(TestContext *context) {
  FakeSystem state;
  state.madvise_fails = true;
  fake = &state;
  const auto allocated = light_rocr::transport::kfd::detail::allocate_gtt(
      17, 23, aperture(), 8192, syscalls());
  context->expect(!allocated, "failed madvise unexpectedly succeeded");
  context->expect(allocated.status.error ==
                      light_rocr::transport::kfd::MemoryError::AdviseDontFork,
                  "wrong madvise error");
  context->expect(
      state.calls == std::vector<std::string>{"reserve", "allocate", "map_host",
                                              "dontfork", "unmap", "free"},
      "madvise failure cleanup order changed");
  fake = nullptr;
}

void map_failure_cleans_up(TestContext *context) {
  FakeSystem state;
  state.map_fails = true;
  state.map_result_n_success = 0;
  fake = &state;
  const auto allocated = light_rocr::transport::kfd::detail::allocate_gtt(
      17, 23, aperture(), 8192, syscalls());
  context->expect(!allocated, "failed GPU map unexpectedly succeeded");
  context->expect(allocated.status.error ==
                          light_rocr::transport::kfd::MemoryError::MapToGpu &&
                      allocated.status.system_error == EBUSY,
                  "GPU map errno was not preserved");
  context->expect(
      state.calls == std::vector<std::string>{"reserve", "allocate", "map_host",
                                              "dontfork", "map_gpu", "unmap",
                                              "free"},
      "failed GPU map did not clean up the host mapping and handle");
  fake = nullptr;
}

void partial_map_cleanup_is_retryable(TestContext *context) {
  FakeSystem state;
  state.map_fails = true;
  state.map_result_n_success = 1;
  state.unmap_gpu_fails = true;
  state.unmap_result_n_success = 0;
  fake = &state;
  auto allocated = light_rocr::transport::kfd::detail::allocate_gtt(
      17, 23, aperture(), 8192, syscalls());
  context->expect(!allocated,
                  "partially failed GPU map unexpectedly succeeded");
  context->expect(allocated.allocation.mapped_device_count == 1 &&
                      allocated.allocation.host_address != nullptr &&
                      allocated.allocation.handle == kHandle,
                  "failed partial-map cleanup discarded owned resources");
  context->expect(allocated.status.message.find("cleanup failed") !=
                      std::string::npos,
                  "partial-map cleanup failure was not diagnosed");
  context->expect(state.calls.back() == "unmap_gpu",
                  "partial-map cleanup continued after GPU unmap failed");

  state.unmap_gpu_fails = false;
  state.unmap_result_n_success = std::numeric_limits<uint32_t>::max();
  const auto released = light_rocr::transport::kfd::detail::release_gtt(
      17, &allocated.allocation, syscalls());
  context->expect(released && allocated.allocation.handle == 0,
                  "partial-map cleanup could not be retried");
  context->expect(state.calls ==
                      std::vector<std::string>{
                          "reserve", "allocate", "map_host", "dontfork",
                          "map_gpu", "unmap_gpu", "unmap_gpu", "unmap", "free"},
                  "partial-map cleanup retry order changed");
  fake = nullptr;
}

void partial_progress_is_retryable(TestContext *context) {
  FakeSystem state;
  state.expected_gpu_ids = {42, 43, 44};
  state.map_fails = true;
  state.map_result_n_success = 2;
  fake = &state;
  light_rocr::transport::kfd::detail::RawGttAllocation allocation;
  allocation.handle = kHandle;

  auto status = light_rocr::transport::kfd::detail::map_gtt(
      17, &allocation, state.expected_gpu_ids, syscalls());
  context->expect(!status &&
                      status.error ==
                          light_rocr::transport::kfd::MemoryError::MapToGpu &&
                      allocation.mapped_device_count == 2,
                  "partial GPU map progress was not retained");

  state.map_result_n_success = 1;
  status = light_rocr::transport::kfd::detail::map_gtt(
      17, &allocation, state.expected_gpu_ids, syscalls());
  context->expect(!status && allocation.mapped_device_count == 2,
                  "non-monotonic GPU map progress was accepted");

  state.map_result_n_success = std::numeric_limits<uint32_t>::max();
  status = light_rocr::transport::kfd::detail::map_gtt(
      17, &allocation, state.expected_gpu_ids, syscalls());
  context->expect(!status && allocation.mapped_device_count == 3 &&
                      !allocation.map_complete,
                  "failed full-progress GPU map was marked complete");

  state.map_fails = false;
  status = light_rocr::transport::kfd::detail::map_gtt(
      17, &allocation, state.expected_gpu_ids, syscalls());
  context->expect(status && allocation.mapped_device_count == 3 &&
                      allocation.map_complete &&
                      state.map_starts == std::vector<uint32_t>{0, 2, 2, 3},
                  "GPU map retry did not resume after the mapped prefix");

  state.unmap_gpu_fails = true;
  state.unmap_result_n_success = 1;
  status = light_rocr::transport::kfd::detail::unmap_gtt(17, &allocation,
                                                         syscalls());
  context->expect(
      !status &&
          status.error ==
              light_rocr::transport::kfd::MemoryError::UnmapFromGpu &&
          allocation.unmapped_device_count == 1,
      "partial GPU unmap progress was not retained");

  state.unmap_result_n_success = 0;
  status = light_rocr::transport::kfd::detail::unmap_gtt(17, &allocation,
                                                         syscalls());
  context->expect(!status && allocation.unmapped_device_count == 1,
                  "non-monotonic GPU unmap progress was accepted");

  state.unmap_result_n_success = 4;
  status = light_rocr::transport::kfd::detail::unmap_gtt(17, &allocation,
                                                         syscalls());
  context->expect(!status && allocation.unmapped_device_count == 1,
                  "out-of-range GPU unmap progress was retained");

  state.unmap_result_n_success = std::numeric_limits<uint32_t>::max();
  status = light_rocr::transport::kfd::detail::unmap_gtt(17, &allocation,
                                                         syscalls());
  context->expect(!status && allocation.unmapped_device_count == 3 &&
                      allocation.map_complete,
                  "failed full-progress GPU unmap was marked complete");

  state.unmap_gpu_fails = false;
  status = light_rocr::transport::kfd::detail::unmap_gtt(17, &allocation,
                                                         syscalls());
  context->expect(
      status && allocation.gpu_ids.empty() &&
          allocation.mapped_device_count == 0 && !allocation.map_complete &&
          state.unmap_starts == std::vector<uint32_t>{0, 1, 1, 1, 3},
      "GPU unmap retry did not resume after the unmapped prefix");

  status = light_rocr::transport::kfd::detail::release_gtt(17, &allocation,
                                                           syscalls());
  context->expect(status && allocation.handle == 0,
                  "mapped allocation handle was not released");
  context->expect(state.contract_valid, "GPU map/unmap contract changed");
  fake = nullptr;
}

void gpu_unmap_failure_blocks_release(TestContext *context) {
  FakeSystem state;
  fake = &state;
  auto allocated = light_rocr::transport::kfd::detail::allocate_gtt(
      17, 23, aperture(), 8192, syscalls());
  state.unmap_gpu_fails = true;
  state.unmap_result_n_success = 0;
  auto released = light_rocr::transport::kfd::detail::release_gtt(
      17, &allocated.allocation, syscalls());
  context->expect(!released &&
                      released.error ==
                          light_rocr::transport::kfd::MemoryError::UnmapFromGpu,
                  "GPU unmap failure was not reported");
  context->expect(allocated.allocation.host_address != nullptr &&
                      allocated.allocation.handle == kHandle,
                  "GPU unmap failure released dependent resources");
  context->expect(state.calls.back() == "unmap_gpu",
                  "release continued after GPU unmap failed");

  state.unmap_gpu_fails = false;
  state.unmap_result_n_success = std::numeric_limits<uint32_t>::max();
  released = light_rocr::transport::kfd::detail::release_gtt(
      17, &allocated.allocation, syscalls());
  context->expect(released && allocated.allocation.handle == 0,
                  "GPU unmap retry did not finish release");
  fake = nullptr;
}

void release_failures_are_retryable(TestContext *context) {
  FakeSystem state;
  fake = &state;
  auto allocated = light_rocr::transport::kfd::detail::allocate_gtt(
      17, 23, aperture(), 8192, syscalls());
  state.munmap_fails = true;
  auto released = light_rocr::transport::kfd::detail::release_gtt(
      17, &allocated.allocation, syscalls());
  context->expect(!released &&
                      released.error ==
                          light_rocr::transport::kfd::MemoryError::UnmapHost,
                  "munmap failure was not reported");
  context->expect(allocated.allocation.host_address != nullptr &&
                      allocated.allocation.handle == kHandle,
                  "munmap failure discarded allocation state");

  state.munmap_fails = false;
  state.free_fails = true;
  released = light_rocr::transport::kfd::detail::release_gtt(
      17, &allocated.allocation, syscalls());
  context->expect(!released &&
                      released.error ==
                          light_rocr::transport::kfd::MemoryError::FreeGtt,
                  "free failure was not reported");
  context->expect(allocated.allocation.host_address == nullptr &&
                      allocated.allocation.handle == kHandle,
                  "free failure discarded the retryable handle");

  state.free_fails = false;
  released = light_rocr::transport::kfd::detail::release_gtt(
      17, &allocated.allocation, syscalls());
  context->expect(static_cast<bool>(released), released.message);
  context->expect(state.calls.back() == "free" &&
                      allocated.allocation.handle == 0,
                  "free retry did not finish the release");
  fake = nullptr;
}

void invalid_public_inputs(TestContext *context) {
  light_rocr::transport::kfd::KfdSession session;
  light_rocr::runtime::Node node;
  node.gpu_id = 42;
  node.simd_count = 1;
  node.drm_render_minor = 128;
  const auto allocated = session.allocate_gtt(node, 4096);
  context->expect(
      !allocated && allocated.status.error ==
                        light_rocr::transport::kfd::MemoryError::InvalidSession,
      "invalid session was accepted");
}

} // namespace

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"successful round trip", successful_round_trip},
      {"invalid size", invalid_size_does_not_reserve},
      {"allocation failure", allocation_failure_releases_va},
      {"rollback unmap failure", rollback_unmap_failure_retains_va},
      {"host map failure", host_map_failure_frees_handle},
      {"cleanup failure", cleanup_failure_retains_handle},
      {"dontfork failure", dontfork_failure_cleans_up},
      {"map failure", map_failure_cleans_up},
      {"partial map cleanup", partial_map_cleanup_is_retryable},
      {"partial GPU progress", partial_progress_is_retryable},
      {"GPU unmap failure", gpu_unmap_failure_blocks_release},
      {"retryable release", release_failures_are_retryable},
      {"invalid public inputs", invalid_public_inputs},
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
