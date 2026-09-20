#include "light_rocr/transport/kfd/memory.hpp"

#include "memory_internal.hpp"

#include <linux/kfd_ioctl.h>

#include <cerrno>
#include <cstdint>
#include <functional>
#include <iostream>
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
  bool munmap_fails = false;
  bool free_fails = false;
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
                      allocated.allocation.handle == kHandle,
                  "allocation result was not retained");
  auto released = light_rocr::transport::kfd::detail::release_gtt(
      17, &allocated.allocation, syscalls());
  context->expect(static_cast<bool>(released), released.message);
  context->expect(
      state.calls == std::vector<std::string>{"reserve", "allocate", "map_host",
                                              "dontfork", "unmap", "free"},
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
