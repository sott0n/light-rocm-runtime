#include "light_rocr/transport/hsakmt/topology.hpp"

#include <hsakmt/hsakmt.h>

#include <algorithm>
#include <cstring>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace light_rocr::transport::hsakmt {
namespace {

class KfdScope {
public:
  KfdScope() = default;
  KfdScope(const KfdScope &) = delete;
  KfdScope &operator=(const KfdScope &) = delete;

  ~KfdScope() {
    if (active_) {
      (void)hsaKmtCloseKFD();
    }
  }

  HSAKMT_STATUS close() {
    active_ = false;
    return hsaKmtCloseKFD();
  }

private:
  bool active_ = true;
};

class SystemPropertiesScope {
public:
  SystemPropertiesScope() = default;
  SystemPropertiesScope(const SystemPropertiesScope &) = delete;
  SystemPropertiesScope &operator=(const SystemPropertiesScope &) = delete;

  ~SystemPropertiesScope() {
    if (active_) {
      (void)hsaKmtReleaseSystemProperties();
    }
  }

  HSAKMT_STATUS release() {
    active_ = false;
    return hsaKmtReleaseSystemProperties();
  }

private:
  bool active_ = true;
};

DiscoveryResult failure(DiscoveryError error, HSAKMT_STATUS status,
                        const std::string &operation) {
  return {error,
          static_cast<uint32_t>(status),
          {},
          operation + " failed with " +
              hsakmt_status_name(static_cast<uint32_t>(status)) + " (" +
              std::to_string(static_cast<uint32_t>(status)) + ")"};
}

runtime::MemoryHeapType convert_heap_type(HSA_HEAPTYPE heap_type) {
  switch (heap_type) {
  case HSA_HEAPTYPE_SYSTEM:
    return runtime::MemoryHeapType::System;
  case HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC:
    return runtime::MemoryHeapType::FrameBufferPublic;
  case HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE:
    return runtime::MemoryHeapType::FrameBufferPrivate;
  case HSA_HEAPTYPE_GPU_GDS:
    return runtime::MemoryHeapType::Gds;
  case HSA_HEAPTYPE_GPU_LDS:
    return runtime::MemoryHeapType::Lds;
  case HSA_HEAPTYPE_GPU_SCRATCH:
    return runtime::MemoryHeapType::Scratch;
  case HSA_HEAPTYPE_DEVICE_SVM:
    return runtime::MemoryHeapType::DeviceSvm;
  case HSA_HEAPTYPE_MMIO_REMAP:
    return runtime::MemoryHeapType::MmioRemap;
  case HSA_HEAPTYPE_NUMHEAPTYPES:
  case HSA_HEAPTYPE_SIZE:
    return runtime::MemoryHeapType::Unknown;
  }
  return runtime::MemoryHeapType::Unknown;
}

std::string bounded_string(const HSAuint8 *bytes, size_t maximum_size) {
  const char *characters = reinterpret_cast<const char *>(bytes);
  const size_t size = ::strnlen(characters, maximum_size);
  return std::string(characters, size);
}

runtime::MemoryBank convert_memory_bank(const HsaMemoryProperties &properties) {
  runtime::MemoryBank bank;
  bank.heap_type = convert_heap_type(properties.HeapType);
  bank.size = properties.SizeInBytes;
  bank.virtual_base_address = properties.VirtualBaseAddress;
  bank.width = properties.Width;
  bank.maximum_clock_mhz = properties.MemoryClockMax;
  bank.hot_pluggable = properties.Flags.ui32.HotPluggable != 0;
  bank.non_volatile = properties.Flags.ui32.NonVolatile != 0;
  return bank;
}

runtime::Node convert_node(uint32_t node_id,
                           const HsaNodeProperties &properties,
                           std::vector<runtime::MemoryBank> memory_banks) {
  runtime::Node node;
  node.node_id = node_id;
  node.gpu_id = properties.KFDGpuID;
  node.cpu_core_count = properties.NumCPUCores;
  node.simd_count = properties.NumFComputeCores;
  node.simd_per_compute_unit = properties.NumSIMDPerCU;
  node.wavefront_size = properties.WaveFrontSize;
  node.maximum_waves_per_simd = properties.MaxWavesPerSIMD;
  node.vendor_id = properties.VendorId;
  node.device_id = properties.DeviceId;
  node.domain = properties.Domain;
  node.location_id = properties.LocationId;
  node.drm_render_minor = properties.DrmRenderMinor;
  node.local_memory_size = properties.LocalMemSize;
  node.integrated = properties.Integrated != 0;
  node.name = bounded_string(properties.AMDName, HSA_PUBLIC_NAME_SIZE);
  node.architecture = {properties.EngineId.ui32.Major,
                       properties.EngineId.ui32.Minor,
                       properties.EngineId.ui32.Stepping};
  node.memory_banks = std::move(memory_banks);
  return node;
}

} // namespace

const char *discovery_error_name(DiscoveryError error) {
  switch (error) {
  case DiscoveryError::None:
    return "none";
  case DiscoveryError::OpenKfd:
    return "open_kfd";
  case DiscoveryError::QueryKfdVersion:
    return "query_kfd_version";
  case DiscoveryError::AcquireSystemProperties:
    return "acquire_system_properties";
  case DiscoveryError::QueryNodeProperties:
    return "query_node_properties";
  case DiscoveryError::QueryMemoryProperties:
    return "query_memory_properties";
  case DiscoveryError::ReleaseSystemProperties:
    return "release_system_properties";
  case DiscoveryError::CloseKfd:
    return "close_kfd";
  }
  return "unknown";
}

DiscoveryResult discover_topology() {
  HSAKMT_STATUS status = hsaKmtOpenKFD();
  if (status != HSAKMT_STATUS_SUCCESS) {
    return failure(DiscoveryError::OpenKfd, status, "hsaKmtOpenKFD");
  }
  KfdScope kfd;

  runtime::Topology topology;
  HsaVersionInfo version{};
  status = hsaKmtGetVersion(&version);
  if (status != HSAKMT_STATUS_SUCCESS) {
    return failure(DiscoveryError::QueryKfdVersion, status, "hsaKmtGetVersion");
  }
  topology.kfd_version = {version.KernelInterfaceMajorVersion,
                          version.KernelInterfaceMinorVersion};

  HsaSystemProperties system_properties{};
  status = hsaKmtAcquireSystemProperties(&system_properties);
  if (status != HSAKMT_STATUS_SUCCESS) {
    return failure(DiscoveryError::AcquireSystemProperties, status,
                   "hsaKmtAcquireSystemProperties");
  }
  SystemPropertiesScope system_properties_scope;

  topology.nodes.reserve(system_properties.NumNodes);
  DiscoveryResult result;
  for (uint32_t node_id = 0; node_id < system_properties.NumNodes; ++node_id) {
    HsaNodeProperties node_properties{};
    status = hsaKmtGetNodeProperties(node_id, &node_properties);
    if (status != HSAKMT_STATUS_SUCCESS) {
      result = failure(DiscoveryError::QueryNodeProperties, status,
                       "hsaKmtGetNodeProperties(node " +
                           std::to_string(node_id) + ")");
      break;
    }

    std::vector<HsaMemoryProperties> raw_memory_banks(
        node_properties.NumMemoryBanks);
    if (!raw_memory_banks.empty()) {
      status = hsaKmtGetNodeMemoryProperties(
          node_id, node_properties.NumMemoryBanks, raw_memory_banks.data());
      if (status != HSAKMT_STATUS_SUCCESS) {
        result = failure(DiscoveryError::QueryMemoryProperties, status,
                         "hsaKmtGetNodeMemoryProperties(node " +
                             std::to_string(node_id) + ")");
        break;
      }
    }

    std::vector<runtime::MemoryBank> memory_banks;
    memory_banks.reserve(raw_memory_banks.size());
    std::transform(raw_memory_banks.begin(), raw_memory_banks.end(),
                   std::back_inserter(memory_banks), convert_memory_bank);
    topology.nodes.push_back(
        convert_node(node_id, node_properties, std::move(memory_banks)));
  }

  const HSAKMT_STATUS release_status = system_properties_scope.release();
  const HSAKMT_STATUS close_status = kfd.close();
  if (result.error != DiscoveryError::None) {
    return result;
  }
  if (release_status != HSAKMT_STATUS_SUCCESS) {
    return failure(DiscoveryError::ReleaseSystemProperties, release_status,
                   "hsaKmtReleaseSystemProperties");
  }
  if (close_status != HSAKMT_STATUS_SUCCESS) {
    return failure(DiscoveryError::CloseKfd, close_status, "hsaKmtCloseKFD");
  }

  result.topology = std::move(topology);
  return result;
}

} // namespace light_rocr::transport::hsakmt
