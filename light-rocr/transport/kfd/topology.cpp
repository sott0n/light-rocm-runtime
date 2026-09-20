#include "light_rocr/transport/kfd/topology.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace light_rocr::transport::kfd {
namespace {

namespace fs = std::filesystem;
using Properties = std::unordered_map<std::string, uint64_t>;

constexpr uint64_t kMemoryHotPluggable = 1U << 0;
constexpr uint64_t kMemoryNonVolatile = 1U << 1;
constexpr unsigned kMaximumSnapshotAttempts = 3;

struct Failure {
  DiscoveryError error = DiscoveryError::None;
  int system_error = 0;
  std::string message;
};

DiscoveryResult failure_result(const Failure &failure) {
  return {failure.error, failure.system_error, {}, failure.message};
}

bool parse_unsigned(const std::string &text, uint64_t *value) {
  if (text.empty()) {
    return false;
  }
  const char *begin = text.data();
  const char *end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, *value, 10);
  return parsed.ec == std::errc{} && parsed.ptr == end;
}

std::string trim(std::string value) {
  const size_t begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return {};
  }
  const size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

bool read_text(const fs::path &path, std::string *text, Failure *failure,
               DiscoveryError read_error) {
  errno = 0;
  std::ifstream input(path);
  if (!input) {
    const int error = errno;
    *failure = {read_error, error, "failed to open " + path.string()};
    return false;
  }

  std::ostringstream contents;
  contents << input.rdbuf();
  if (input.bad()) {
    const int error = errno;
    *failure = {read_error, error, "failed to read " + path.string()};
    return false;
  }
  *text = contents.str();
  return true;
}

bool read_scalar(const fs::path &path, uint64_t *value, Failure *failure,
                 DiscoveryError read_error, DiscoveryError parse_error) {
  std::string contents;
  if (!read_text(path, &contents, failure, read_error)) {
    return false;
  }
  contents = trim(std::move(contents));
  if (!parse_unsigned(contents, value)) {
    *failure = {parse_error, 0, "invalid unsigned integer in " + path.string()};
    return false;
  }
  return true;
}

bool read_properties(const fs::path &path, Properties *properties,
                     Failure *failure, DiscoveryError read_error,
                     DiscoveryError parse_error) {
  std::string contents;
  if (!read_text(path, &contents, failure, read_error)) {
    return false;
  }

  std::istringstream lines(contents);
  std::string line;
  size_t line_number = 0;
  while (std::getline(lines, line)) {
    ++line_number;
    if (trim(line).empty()) {
      continue;
    }
    std::istringstream fields(line);
    std::string name;
    std::string value_text;
    std::string extra;
    uint64_t value = 0;
    if (!(fields >> name >> value_text) || (fields >> extra) ||
        !parse_unsigned(value_text, &value) ||
        !properties->emplace(name, value).second) {
      *failure = {parse_error, 0,
                  "invalid property at " + path.string() + ":" +
                      std::to_string(line_number)};
      return false;
    }
  }
  return true;
}

bool require_property(const Properties &properties, const std::string &name,
                      uint64_t maximum, uint64_t *value, Failure *failure,
                      DiscoveryError error, const fs::path &path) {
  const auto found = properties.find(name);
  if (found == properties.end()) {
    *failure = {error, 0, "missing property " + name + " in " + path.string()};
    return false;
  }
  if (found->second > maximum) {
    *failure = {error, 0,
                "property " + name + " is out of range in " + path.string()};
    return false;
  }
  *value = found->second;
  return true;
}

uint64_t optional_property(const Properties &properties,
                           const std::string &name) {
  const auto found = properties.find(name);
  return found == properties.end() ? 0 : found->second;
}

runtime::MemoryHeapType convert_heap_type(uint64_t heap_type) {
  switch (heap_type) {
  case 0:
    return runtime::MemoryHeapType::System;
  case 1:
    return runtime::MemoryHeapType::FrameBufferPublic;
  case 2:
    return runtime::MemoryHeapType::FrameBufferPrivate;
  case 3:
    return runtime::MemoryHeapType::Gds;
  case 4:
    return runtime::MemoryHeapType::Lds;
  case 5:
    return runtime::MemoryHeapType::Scratch;
  default:
    return runtime::MemoryHeapType::Unknown;
  }
}

bool read_memory_bank(const fs::path &bank_path, runtime::MemoryBank *bank,
                      Failure *failure) {
  const fs::path properties_path = bank_path / "properties";
  Properties properties;
  if (!read_properties(properties_path, &properties, failure,
                       DiscoveryError::ReadMemoryBank,
                       DiscoveryError::InvalidMemoryBank)) {
    return false;
  }

  uint64_t heap_type = 0;
  uint64_t size = 0;
  uint64_t flags = 0;
  uint64_t width = 0;
  uint64_t clock = 0;
  if (!require_property(properties, "heap_type",
                        std::numeric_limits<uint32_t>::max(), &heap_type,
                        failure, DiscoveryError::InvalidMemoryBank,
                        properties_path) ||
      !require_property(properties, "size_in_bytes",
                        std::numeric_limits<uint64_t>::max(), &size, failure,
                        DiscoveryError::InvalidMemoryBank, properties_path) ||
      !require_property(properties, "flags",
                        std::numeric_limits<uint32_t>::max(), &flags, failure,
                        DiscoveryError::InvalidMemoryBank, properties_path) ||
      !require_property(properties, "width",
                        std::numeric_limits<uint32_t>::max(), &width, failure,
                        DiscoveryError::InvalidMemoryBank, properties_path) ||
      !require_property(properties, "mem_clk_max",
                        std::numeric_limits<uint32_t>::max(), &clock, failure,
                        DiscoveryError::InvalidMemoryBank, properties_path)) {
    return false;
  }

  bank->heap_type = convert_heap_type(heap_type);
  bank->size = size;
  bank->width = static_cast<uint32_t>(width);
  bank->maximum_clock_mhz = static_cast<uint32_t>(clock);
  bank->hot_pluggable = (flags & kMemoryHotPluggable) != 0;
  bank->non_volatile = (flags & kMemoryNonVolatile) != 0;
  return true;
}

std::string architecture_name(runtime::GpuArchitecture architecture) {
  const uint32_t packed = (architecture.major << 16U) |
                          (architecture.minor << 8U) | architecture.stepping;
  std::ostringstream name;
  name << "GFX" << std::hex << std::nouppercase << std::setw(6)
       << std::setfill('0') << packed;
  return name.str();
}

bool read_node(uint32_t node_id, const fs::path &node_path, runtime::Node *node,
               Failure *failure) {
  const fs::path properties_path = node_path / "properties";
  Properties properties;
  if (!read_properties(properties_path, &properties, failure,
                       DiscoveryError::ReadNode, DiscoveryError::InvalidNode)) {
    return false;
  }

  uint64_t gpu_id = 0;
  if (!read_scalar(node_path / "gpu_id", &gpu_id, failure,
                   DiscoveryError::ReadNode, DiscoveryError::InvalidNode) ||
      gpu_id > std::numeric_limits<uint32_t>::max()) {
    if (failure->error == DiscoveryError::None) {
      *failure = {DiscoveryError::InvalidNode, 0,
                  "GPU ID is out of range in " + node_path.string()};
    }
    return false;
  }

  struct Field {
    const char *name;
    uint64_t maximum;
    uint64_t value = 0;
  } fields[] = {
      {"cpu_cores_count", std::numeric_limits<uint32_t>::max()},
      {"simd_count", std::numeric_limits<uint32_t>::max()},
      {"simd_per_cu", std::numeric_limits<uint32_t>::max()},
      {"wave_front_size", std::numeric_limits<uint32_t>::max()},
      {"max_waves_per_simd", std::numeric_limits<uint32_t>::max()},
      {"array_count", std::numeric_limits<uint32_t>::max()},
      {"simd_arrays_per_engine", std::numeric_limits<uint32_t>::max()},
      {"max_slots_scratch_cu", std::numeric_limits<uint32_t>::max()},
      {"gfx_target_version", std::numeric_limits<uint32_t>::max()},
      {"vendor_id", std::numeric_limits<uint16_t>::max()},
      {"device_id", std::numeric_limits<uint16_t>::max()},
      {"domain", std::numeric_limits<uint32_t>::max()},
      {"location_id", std::numeric_limits<uint32_t>::max()},
      {"drm_render_minor", std::numeric_limits<int32_t>::max()},
      {"mem_banks_count", std::numeric_limits<uint32_t>::max()},
  };
  for (Field &field : fields) {
    if (!require_property(properties, field.name, field.maximum, &field.value,
                          failure, DiscoveryError::InvalidNode,
                          properties_path)) {
      return false;
    }
  }

  const uint32_t simd_count = static_cast<uint32_t>(fields[1].value);
  const uint32_t arrays_per_engine = static_cast<uint32_t>(fields[6].value);
  const uint32_t array_count = static_cast<uint32_t>(fields[5].value);
  if (simd_count != 0 &&
      (arrays_per_engine == 0 || array_count % arrays_per_engine != 0)) {
    *failure = {DiscoveryError::InvalidNode, 0,
                "invalid shader-engine topology in " +
                    properties_path.string()};
    return false;
  }

  const uint32_t gfx_version = static_cast<uint32_t>(fields[8].value);
  runtime::GpuArchitecture architecture{(gfx_version / 10000U) % 100U,
                                        (gfx_version / 100U) % 100U,
                                        gfx_version % 100U};
  if (simd_count != 0 && architecture.major == 0) {
    *failure = {DiscoveryError::InvalidNode, 0,
                "GPU node has no gfx target version in " +
                    properties_path.string()};
    return false;
  }

  node->node_id = node_id;
  node->gpu_id = static_cast<uint32_t>(gpu_id);
  node->cpu_core_count = static_cast<uint32_t>(fields[0].value);
  node->simd_count = simd_count;
  node->simd_per_compute_unit = static_cast<uint32_t>(fields[2].value);
  node->wavefront_size = static_cast<uint32_t>(fields[3].value);
  node->maximum_waves_per_simd = static_cast<uint32_t>(fields[4].value);
  node->shader_engine_count =
      arrays_per_engine == 0 ? 0 : array_count / arrays_per_engine;
  node->maximum_scratch_waves_per_compute_unit =
      static_cast<uint32_t>(fields[7].value);
  const uint64_t xcc_count = optional_property(properties, "num_xcc");
  if (xcc_count > std::numeric_limits<uint32_t>::max()) {
    *failure = {DiscoveryError::InvalidNode, 0,
                "property num_xcc is out of range in " +
                    properties_path.string()};
    return false;
  }
  node->xcc_count = static_cast<uint32_t>(xcc_count);
  node->vendor_id = static_cast<uint16_t>(fields[9].value);
  node->device_id = static_cast<uint16_t>(fields[10].value);
  node->domain = static_cast<uint32_t>(fields[11].value);
  node->location_id = static_cast<uint32_t>(fields[12].value);
  node->drm_render_minor = static_cast<int32_t>(fields[13].value);
  node->local_memory_size = optional_property(properties, "local_mem_size");
  node->integrated = node->cpu_core_count != 0 && node->simd_count != 0;
  node->architecture = architecture;

  std::string sysfs_name;
  if (!read_text(node_path / "name", &sysfs_name, failure,
                 DiscoveryError::ReadNode)) {
    return false;
  }
  node->name = node->is_gpu() ? architecture_name(architecture)
                              : trim(std::move(sysfs_name));

  const uint32_t memory_bank_count = static_cast<uint32_t>(fields[14].value);
  node->memory_banks.reserve(memory_bank_count);
  for (uint32_t bank_id = 0; bank_id < memory_bank_count; ++bank_id) {
    runtime::MemoryBank bank;
    if (!read_memory_bank(node_path / "mem_banks" / std::to_string(bank_id),
                          &bank, failure)) {
      return false;
    }
    node->memory_banks.push_back(bank);
  }
  return true;
}

bool parse_node_id(const std::string &name, uint32_t *node_id) {
  uint64_t value = 0;
  if (!parse_unsigned(name, &value) ||
      value > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  *node_id = static_cast<uint32_t>(value);
  return true;
}

bool read_snapshot(const fs::path &topology_path, runtime::Topology *topology,
                   Failure *failure) {
  const fs::path nodes_path = topology_path / "nodes";
  std::error_code error;
  fs::directory_iterator entries(nodes_path, error);
  if (error) {
    *failure = {DiscoveryError::EnumerateNodes, error.value(),
                "failed to enumerate " + nodes_path.string() + ": " +
                    error.message()};
    return false;
  }

  std::vector<std::pair<uint32_t, fs::path>> node_paths;
  const fs::directory_iterator end;
  while (entries != end) {
    const fs::directory_entry entry = *entries;
    if (!entry.is_directory(error)) {
      if (error) {
        *failure = {DiscoveryError::EnumerateNodes, error.value(),
                    "failed to inspect " + entry.path().string() + ": " +
                        error.message()};
        return false;
      }
    } else {
      uint32_t node_id = 0;
      if (parse_node_id(entry.path().filename().string(), &node_id)) {
        node_paths.emplace_back(node_id, entry.path());
      }
    }

    entries.increment(error);
    if (error) {
      *failure = {DiscoveryError::EnumerateNodes, error.value(),
                  "failed while enumerating " + nodes_path.string() + ": " +
                      error.message()};
      return false;
    }
  }
  std::sort(node_paths.begin(), node_paths.end(),
            [](const auto &left, const auto &right) {
              return left.first < right.first;
            });
  if (node_paths.empty()) {
    *failure = {DiscoveryError::EnumerateNodes, 0,
                "no KFD topology nodes found in " + nodes_path.string()};
    return false;
  }

  topology->nodes.reserve(node_paths.size());
  for (const auto &[node_id, node_path] : node_paths) {
    runtime::Node node;
    if (!read_node(node_id, node_path, &node, failure)) {
      return false;
    }
    topology->nodes.push_back(std::move(node));
  }
  return true;
}

} // namespace

const char *discovery_error_name(DiscoveryError error) {
  switch (error) {
  case DiscoveryError::None:
    return "none";
  case DiscoveryError::InvalidSession:
    return "invalid_session";
  case DiscoveryError::ReadGeneration:
    return "read_generation";
  case DiscoveryError::EnumerateNodes:
    return "enumerate_nodes";
  case DiscoveryError::ReadNode:
    return "read_node";
  case DiscoveryError::InvalidNode:
    return "invalid_node";
  case DiscoveryError::ReadMemoryBank:
    return "read_memory_bank";
  case DiscoveryError::InvalidMemoryBank:
    return "invalid_memory_bank";
  case DiscoveryError::TopologyChanged:
    return "topology_changed";
  }
  return "unknown";
}

DiscoveryResult read_topology_sysfs(const std::string &topology_path,
                                    runtime::KfdVersion version) {
  for (unsigned attempt = 0; attempt < kMaximumSnapshotAttempts; ++attempt) {
    Failure generation_failure;
    uint64_t generation_before = 0;
    if (!read_scalar(fs::path(topology_path) / "generation_id",
                     &generation_before, &generation_failure,
                     DiscoveryError::ReadGeneration,
                     DiscoveryError::ReadGeneration)) {
      return failure_result(generation_failure);
    }

    runtime::Topology topology;
    topology.kfd_version = version;
    Failure snapshot_failure;
    const bool snapshot_read =
        read_snapshot(topology_path, &topology, &snapshot_failure);

    uint64_t generation_after = 0;
    if (!read_scalar(fs::path(topology_path) / "generation_id",
                     &generation_after, &generation_failure,
                     DiscoveryError::ReadGeneration,
                     DiscoveryError::ReadGeneration)) {
      return failure_result(generation_failure);
    }
    if (generation_before != generation_after) {
      continue;
    }
    if (!snapshot_read) {
      return failure_result(snapshot_failure);
    }
    return {{}, 0, std::move(topology), {}};
  }

  return {DiscoveryError::TopologyChanged,
          0,
          {},
          "KFD topology changed during all snapshot attempts"};
}

DiscoveryResult discover_topology(const KfdSession &session,
                                  const std::string &topology_path) {
  if (!session) {
    return {DiscoveryError::InvalidSession,
            0,
            {},
            "direct KFD topology discovery requires an open session"};
  }
  return read_topology_sysfs(topology_path, session.version());
}

} // namespace light_rocr::transport::kfd
