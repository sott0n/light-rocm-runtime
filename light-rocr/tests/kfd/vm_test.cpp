#include "light_rocr/transport/kfd/vm.hpp"

#include "vm_internal.hpp"

#include <linux/kfd_ioctl.h>

#include <cerrno>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

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

struct IoctlResponse {
  uint32_t count = 0;
  std::vector<kfd_process_device_apertures> apertures;
  int system_error = 0;
};

struct FakeIoctlState {
  std::vector<IoctlResponse> responses;
  size_t next_response = 0;
  bool contract_valid = true;
};

FakeIoctlState *g_fake_ioctl = nullptr;
bool g_policy_contract_valid = false;
int g_policy_error = 0;

int fake_policy_ioctl(int fd, unsigned long request, void *arguments) {
  if (fd != 17 || request != AMDKFD_IOC_SET_MEMORY_POLICY ||
      arguments == nullptr) {
    errno = EINVAL;
    return -1;
  }
  const auto *policy =
      static_cast<kfd_ioctl_set_memory_policy_args *>(arguments);
  g_policy_contract_valid =
      policy->gpu_id == 22 &&
      policy->default_policy == KFD_IOC_CACHE_POLICY_NONCOHERENT &&
      policy->alternate_policy == KFD_IOC_CACHE_POLICY_COHERENT &&
      policy->alternate_aperture_base == 2U * 1024U * 1024U &&
      policy->alternate_aperture_size ==
          (uint64_t{1} << 47U) - 2U * 1024U * 1024U;
  if (g_policy_error != 0) {
    errno = g_policy_error;
    return -1;
  }
  return 0;
}

int fake_ioctl(int fd, unsigned long request, void *arguments) {
  if (g_fake_ioctl == nullptr || fd != 17 ||
      request != AMDKFD_IOC_GET_PROCESS_APERTURES_NEW ||
      g_fake_ioctl->next_response >= g_fake_ioctl->responses.size()) {
    if (g_fake_ioctl != nullptr) {
      g_fake_ioctl->contract_valid = false;
    }
    errno = EINVAL;
    return -1;
  }

  const IoctlResponse &response =
      g_fake_ioctl->responses[g_fake_ioctl->next_response++];
  if (response.system_error != 0) {
    errno = response.system_error;
    return -1;
  }

  auto *ioctl_arguments =
      static_cast<kfd_ioctl_get_process_apertures_new_args *>(arguments);
  if (ioctl_arguments->num_of_nodes == 0) {
    if (!response.apertures.empty() ||
        ioctl_arguments->kfd_process_device_apertures_ptr != 0) {
      g_fake_ioctl->contract_valid = false;
    }
  } else {
    if (ioctl_arguments->num_of_nodes < response.apertures.size() ||
        ioctl_arguments->kfd_process_device_apertures_ptr == 0) {
      g_fake_ioctl->contract_valid = false;
      errno = EINVAL;
      return -1;
    }
    auto *destination =
        reinterpret_cast<kfd_process_device_apertures *>(static_cast<uintptr_t>(
            ioctl_arguments->kfd_process_device_apertures_ptr));
    for (size_t index = 0; index < response.apertures.size(); ++index) {
      destination[index] = response.apertures[index];
    }
  }
  ioctl_arguments->num_of_nodes = response.count;
  return 0;
}

kfd_process_device_apertures aperture(uint32_t gpu_id, uint64_t base) {
  kfd_process_device_apertures result{};
  result.gpu_id = gpu_id;
  result.lds_base = base;
  result.lds_limit = base + 0xffff;
  result.scratch_base = base + 0x100000;
  result.scratch_limit = base + 0x1fffff;
  result.gpuvm_base = base + 0x200000;
  result.gpuvm_limit = base + 0x2fffff;
  return result;
}

light_rocr::transport::kfd::VmResult run_query(FakeIoctlState *state,
                                               uint32_t gpu_id) {
  g_fake_ioctl = state;
  auto result = light_rocr::transport::kfd::detail::query_process_aperture(
      17, gpu_id, fake_ioctl);
  g_fake_ioctl = nullptr;
  return result;
}

void successful_query(TestContext *context) {
  FakeIoctlState state{{{2, {}, 0},
                        {2, {aperture(11, 0x1000), aperture(22, 0x400000)}, 0},
                        {2, {}, 0}}};
  const auto result = run_query(&state, 22);
  context->expect(static_cast<bool>(result), result.status.message);
  context->expect(state.contract_valid && state.next_response == 3,
                  "two-pass aperture ioctl contract was not followed");
  context->expect(result.aperture.gpu_id == 22 &&
                      result.aperture.lds_base == 0x400000 &&
                      result.aperture.gpuvm_limit == 0x6fffff,
                  "selected aperture was not converted");
}

void memory_policy_matches_rocr(TestContext *context) {
  g_policy_contract_valid = false;
  g_policy_error = 0;
  const auto status =
      light_rocr::transport::kfd::detail::configure_memory_policy(
          17, 22, fake_policy_ioctl);
  context->expect(static_cast<bool>(status), status.message);
  context->expect(g_policy_contract_valid,
                  "memory policy does not match ROCr's KFD contract");
}

void memory_policy_failure_preserves_errno(TestContext *context) {
  g_policy_contract_valid = false;
  g_policy_error = EIO;
  const auto status =
      light_rocr::transport::kfd::detail::configure_memory_policy(
          17, 22, fake_policy_ioctl);
  context->expect(
      !status && g_policy_contract_valid &&
          status.error ==
              light_rocr::transport::kfd::VmError::ConfigureMemoryPolicy &&
          status.system_error == EIO,
      "memory policy failure did not preserve its cause");
  g_policy_error = 0;
}

void changed_count_is_retried(TestContext *context) {
  FakeIoctlState state{{{1, {}, 0},
                        {1, {aperture(11, 0x1000)}, 0},
                        {2, {}, 0},
                        {2, {}, 0},
                        {2, {aperture(11, 0x1000), aperture(22, 0x400000)}, 0},
                        {2, {}, 0}}};
  const auto result = run_query(&state, 22);
  context->expect(static_cast<bool>(result), result.status.message);
  context->expect(state.contract_valid && state.next_response == 6,
                  "changed aperture count was not retried");
}

void duplicate_gpu_is_rejected(TestContext *context) {
  FakeIoctlState state{{{2, {}, 0},
                        {2, {aperture(22, 0x1000), aperture(22, 0x400000)}, 0},
                        {2, {}, 0}}};
  const auto result = run_query(&state, 22);
  context->expect(!result, "duplicate GPU aperture unexpectedly succeeded");
  context->expect(result.status.error ==
                      light_rocr::transport::kfd::VmError::DuplicateAperture,
                  "wrong duplicate-aperture error");
}

void invalid_range_is_rejected(TestContext *context) {
  auto invalid = aperture(22, 0x400000);
  invalid.scratch_limit = invalid.scratch_base - 1;
  FakeIoctlState state{{{1, {}, 0}, {1, {invalid}, 0}, {1, {}, 0}}};
  const auto result = run_query(&state, 22);
  context->expect(!result, "invalid aperture range unexpectedly succeeded");
  context->expect(result.status.error ==
                      light_rocr::transport::kfd::VmError::InvalidAperture,
                  "wrong invalid-aperture error");
}

void count_failure_preserves_errno(TestContext *context) {
  FakeIoctlState state{{{0, {}, EIO}}};
  const auto result = run_query(&state, 22);
  context->expect(!result, "failed count query unexpectedly succeeded");
  context->expect(
      result.status.error ==
              light_rocr::transport::kfd::VmError::QueryApertureCount &&
          result.status.system_error == EIO,
      "count query error or errno was not preserved");
}

void invalid_session_is_reported(TestContext *context) {
  light_rocr::transport::kfd::KfdSession session;
  light_rocr::runtime::Node node;
  node.gpu_id = 22;
  node.simd_count = 1;
  node.drm_render_minor = 128;
  const auto result = session.acquire_vm(node);
  context->expect(!result, "invalid session unexpectedly acquired a VM");
  context->expect(result.status.error ==
                      light_rocr::transport::kfd::VmError::InvalidSession,
                  "wrong invalid-session error");
}

} // namespace

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"successful query", successful_query},
      {"ROCr memory policy", memory_policy_matches_rocr},
      {"memory policy failure", memory_policy_failure_preserves_errno},
      {"changed count retry", changed_count_is_retried},
      {"duplicate GPU", duplicate_gpu_is_rejected},
      {"invalid range", invalid_range_is_rejected},
      {"count failure", count_failure_preserves_errno},
      {"invalid session", invalid_session_is_reported},
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
