#include "light_rocr/transport/hsakmt/topology.hpp"

#include <hsakmt/hsakmt.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct FakeKmt {
  HSAKMT_STATUS open_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS version_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS acquire_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS node_status = HSAKMT_STATUS_SUCCESS;
  uint32_t failing_node = 0;
  HSAKMT_STATUS memory_status = HSAKMT_STATUS_SUCCESS;
  uint32_t failing_memory_node = 0;
  HSAKMT_STATUS release_status = HSAKMT_STATUS_SUCCESS;
  HSAKMT_STATUS close_status = HSAKMT_STATUS_SUCCESS;
  std::vector<std::string> calls;
};

FakeKmt fake;

void reset_fake() { fake = {}; }

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

void successful_discovery(TestContext *context) {
  reset_fake();
  const auto result = light_rocr::transport::hsakmt::discover_topology();
  context->expect(static_cast<bool>(result), result.message);
  expect_calls(context, {"open", "version", "acquire", "node:0", "memory:0",
                         "node:1", "memory:1", "release", "close"});
  context->expect(result.topology.kfd_version.major == 1 &&
                      result.topology.kfd_version.minor == 16,
                  "KFD version was not converted");
  context->expect(result.topology.nodes.size() == 2,
                  "unexpected topology node count");
  if (result.topology.nodes.size() != 2) {
    return;
  }

  const auto &cpu = result.topology.nodes[0];
  context->expect(!cpu.is_gpu() && cpu.cpu_core_count == 64,
                  "CPU node was not converted");
  context->expect(cpu.memory_banks.size() == 1 &&
                      cpu.memory_banks[0].heap_type ==
                          light_rocr::runtime::MemoryHeapType::System,
                  "system bank was not converted");

  const auto &gpu = result.topology.nodes[1];
  context->expect(gpu.is_gpu() && gpu.gpu_id == 1234,
                  "GPU node was not converted");
  context->expect(gpu.compute_unit_count() == 60,
                  "GPU compute units were not derived");
  context->expect(gpu.shader_engine_count == 3 &&
                      gpu.maximum_scratch_waves_per_compute_unit == 32 &&
                      gpu.xcc_count == 1,
                  "GPU scratch topology was not converted");
  context->expect(light_rocr::runtime::gfx_target_name(gpu.architecture) ==
                      "gfx1101",
                  "GPU architecture was not converted");
  context->expect(gpu.name == "test-gpu", "GPU name was not converted");
  context->expect(gpu.domain == 7 && gpu.location_id == 0x1200,
                  "GPU PCI location was not converted");
  context->expect(
      gpu.memory_banks.size() == 1 &&
          gpu.memory_banks[0].heap_type ==
              light_rocr::runtime::MemoryHeapType::FrameBufferPublic &&
          gpu.memory_banks[0].size == 8ULL * 1024 * 1024 * 1024,
      "frame-buffer bank was not converted");
}

void open_failure_has_no_cleanup(TestContext *context) {
  reset_fake();
  fake.open_status = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;
  const auto result = light_rocr::transport::hsakmt::discover_topology();
  context->expect(!result, "open failure unexpectedly succeeded");
  context->expect(result.error ==
                      light_rocr::transport::hsakmt::DiscoveryError::OpenKfd,
                  "wrong open failure");
  expect_calls(context, {"open"});
}

void version_failure_closes_kfd(TestContext *context) {
  reset_fake();
  fake.version_status = HSAKMT_STATUS_DRIVER_MISMATCH;
  const auto result = light_rocr::transport::hsakmt::discover_topology();
  context->expect(!result, "version failure unexpectedly succeeded");
  context->expect(
      result.error ==
          light_rocr::transport::hsakmt::DiscoveryError::QueryKfdVersion,
      "wrong version failure");
  expect_calls(context, {"open", "version", "close"});
}

void acquire_failure_closes_kfd(TestContext *context) {
  reset_fake();
  fake.acquire_status = HSAKMT_STATUS_ERROR;
  const auto result = light_rocr::transport::hsakmt::discover_topology();
  context->expect(!result, "acquire failure unexpectedly succeeded");
  context->expect(result.error == light_rocr::transport::hsakmt::
                                      DiscoveryError::AcquireSystemProperties,
                  "wrong acquire failure");
  expect_calls(context, {"open", "version", "acquire", "close"});
}

void node_failure_releases_and_closes(TestContext *context) {
  reset_fake();
  fake.node_status = HSAKMT_STATUS_INVALID_NODE_UNIT;
  fake.failing_node = 1;
  const auto result = light_rocr::transport::hsakmt::discover_topology();
  context->expect(!result, "node failure unexpectedly succeeded");
  context->expect(
      result.error ==
          light_rocr::transport::hsakmt::DiscoveryError::QueryNodeProperties,
      "wrong node failure");
  expect_calls(context, {"open", "version", "acquire", "node:0", "memory:0",
                         "node:1", "release", "close"});
}

void memory_failure_releases_and_closes(TestContext *context) {
  reset_fake();
  fake.memory_status = HSAKMT_STATUS_ERROR;
  fake.failing_memory_node = 1;
  const auto result = light_rocr::transport::hsakmt::discover_topology();
  context->expect(!result, "memory failure unexpectedly succeeded");
  context->expect(
      result.error ==
          light_rocr::transport::hsakmt::DiscoveryError::QueryMemoryProperties,
      "wrong memory failure");
  expect_calls(context, {"open", "version", "acquire", "node:0", "memory:0",
                         "node:1", "memory:1", "release", "close"});
}

void cleanup_failures_are_reported(TestContext *context) {
  reset_fake();
  fake.release_status = HSAKMT_STATUS_ERROR;
  auto result = light_rocr::transport::hsakmt::discover_topology();
  context->expect(result.error == light_rocr::transport::hsakmt::
                                      DiscoveryError::ReleaseSystemProperties,
                  "release failure was not reported");
  context->expect(fake.calls.back() == "close",
                  "KFD was not closed after release failure");

  reset_fake();
  fake.close_status = HSAKMT_STATUS_ERROR;
  result = light_rocr::transport::hsakmt::discover_topology();
  context->expect(result.error ==
                      light_rocr::transport::hsakmt::DiscoveryError::CloseKfd,
                  "close failure was not reported");
}

} // namespace

extern "C" HSAKMT_STATUS hsaKmtOpenKFD() {
  fake.calls.emplace_back("open");
  return fake.open_status;
}

extern "C" HSAKMT_STATUS hsaKmtCloseKFD() {
  fake.calls.emplace_back("close");
  return fake.close_status;
}

extern "C" HSAKMT_STATUS hsaKmtGetVersion(HsaVersionInfo *version) {
  fake.calls.emplace_back("version");
  if (fake.version_status == HSAKMT_STATUS_SUCCESS) {
    version->KernelInterfaceMajorVersion = 1;
    version->KernelInterfaceMinorVersion = 16;
  }
  return fake.version_status;
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
  return fake.release_status;
}

extern "C" HSAKMT_STATUS
hsaKmtGetNodeProperties(HSAuint32 node_id, HsaNodeProperties *properties) {
  fake.calls.push_back("node:" + std::to_string(node_id));
  if (node_id == fake.failing_node &&
      fake.node_status != HSAKMT_STATUS_SUCCESS) {
    return fake.node_status;
  }

  std::memset(properties, 0, sizeof(*properties));
  properties->NumMemoryBanks = 1;
  if (node_id == 0) {
    properties->NumCPUCores = 64;
    return HSAKMT_STATUS_SUCCESS;
  }

  properties->NumFComputeCores = 120;
  properties->NumSIMDPerCU = 2;
  properties->WaveFrontSize = 32;
  properties->MaxWavesPerSIMD = 16;
  properties->NumShaderBanks = 3;
  properties->MaxSlotsScratchCU = 32;
  properties->NumXcc = 1;
  properties->EngineId.ui32.Major = 11;
  properties->EngineId.ui32.Minor = 0;
  properties->EngineId.ui32.Stepping = 1;
  properties->VendorId = 0x1002;
  properties->DeviceId = 0x1234;
  properties->Domain = 7;
  properties->LocationId = 0x1200;
  properties->DrmRenderMinor = 128;
  properties->LocalMemSize = 8ULL * 1024 * 1024 * 1024;
  properties->KFDGpuID = 1234;
  constexpr char kName[] = "test-gpu";
  std::copy(kName, kName + sizeof(kName), properties->AMDName);
  return HSAKMT_STATUS_SUCCESS;
}

extern "C" HSAKMT_STATUS
hsaKmtGetNodeMemoryProperties(HSAuint32 node_id, HSAuint32 bank_count,
                              HsaMemoryProperties *properties) {
  fake.calls.push_back("memory:" + std::to_string(node_id));
  if (node_id == fake.failing_memory_node &&
      fake.memory_status != HSAKMT_STATUS_SUCCESS) {
    return fake.memory_status;
  }
  if (bank_count != 1) {
    return HSAKMT_STATUS_INVALID_PARAMETER;
  }
  std::memset(properties, 0, sizeof(*properties));
  properties->HeapType =
      node_id == 0 ? HSA_HEAPTYPE_SYSTEM : HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC;
  properties->SizeInBytes =
      node_id == 0 ? 64ULL * 1024 * 1024 * 1024 : 8ULL * 1024 * 1024 * 1024;
  return HSAKMT_STATUS_SUCCESS;
}

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"successful_discovery", successful_discovery},
      {"open_failure_has_no_cleanup", open_failure_has_no_cleanup},
      {"version_failure_closes_kfd", version_failure_closes_kfd},
      {"acquire_failure_closes_kfd", acquire_failure_closes_kfd},
      {"node_failure_releases_and_closes", node_failure_releases_and_closes},
      {"memory_failure_releases_and_closes",
       memory_failure_releases_and_closes},
      {"cleanup_failures_are_reported", cleanup_failures_are_reported},
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
