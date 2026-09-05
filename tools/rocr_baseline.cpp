#include <hsa/hsa.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct AgentRecord {
  hsa_agent_t agent{};
  hsa_device_type_t type = HSA_DEVICE_TYPE_CPU;
  std::string name;
};

struct RegionRecord {
  hsa_region_t region{};
  hsa_region_segment_t segment = HSA_REGION_SEGMENT_READONLY;
  uint32_t flags = 0;
  size_t size = 0;
  size_t allocation_max_size = 0;
  size_t allocation_granule = 0;
  size_t allocation_alignment = 0;
  bool allocation_allowed = false;
};

std::string status_string(hsa_status_t status) {
  const char *message = nullptr;
  if (hsa_status_string(status, &message) == HSA_STATUS_SUCCESS && message) {
    return message;
  }
  return "unknown HSA status";
}

bool check(hsa_status_t status, const char *operation) {
  if (status == HSA_STATUS_SUCCESS) {
    return true;
  }
  std::cerr << operation << " failed: " << status_string(status) << " ("
            << static_cast<int>(status) << ")\n";
  return false;
}

const char *device_type_name(hsa_device_type_t type) {
  switch (type) {
  case HSA_DEVICE_TYPE_CPU:
    return "cpu";
  case HSA_DEVICE_TYPE_GPU:
    return "gpu";
  case HSA_DEVICE_TYPE_DSP:
    return "dsp";
  case HSA_DEVICE_TYPE_AIE:
    return "aie";
  }
  return "unknown";
}

const char *profile_name(hsa_profile_t profile) {
  switch (profile) {
  case HSA_PROFILE_BASE:
    return "base";
  case HSA_PROFILE_FULL:
    return "full";
  }
  return "unknown";
}

const char *region_segment_name(hsa_region_segment_t segment) {
  switch (segment) {
  case HSA_REGION_SEGMENT_GLOBAL:
    return "global";
  case HSA_REGION_SEGMENT_READONLY:
    return "readonly";
  case HSA_REGION_SEGMENT_PRIVATE:
    return "private";
  case HSA_REGION_SEGMENT_GROUP:
    return "group";
  case HSA_REGION_SEGMENT_KERNARG:
    return "kernarg";
  }
  return "unknown";
}

hsa_status_t collect_agent(hsa_agent_t agent, void *data) {
  auto *agents = static_cast<std::vector<AgentRecord> *>(data);
  hsa_device_type_t type = HSA_DEVICE_TYPE_CPU;
  hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  char name[64] = {};
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  agents->push_back(AgentRecord{agent, type, name});
  return HSA_STATUS_SUCCESS;
}

hsa_status_t collect_region(hsa_region_t region, void *data) {
  auto *regions = static_cast<std::vector<RegionRecord> *>(data);
  RegionRecord record{};
  record.region = region;

  hsa_status_t status =
      hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &record.segment);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  status = hsa_region_get_info(region, HSA_REGION_INFO_SIZE, &record.size);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  status = hsa_region_get_info(region, HSA_REGION_INFO_ALLOC_MAX_SIZE,
                               &record.allocation_max_size);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  status = hsa_region_get_info(region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
                               &record.allocation_allowed);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  if (record.segment == HSA_REGION_SEGMENT_GLOBAL) {
    status = hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS,
                                 &record.flags);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }
  if (record.allocation_allowed) {
    status = hsa_region_get_info(region, HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE,
                                 &record.allocation_granule);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
    status =
        hsa_region_get_info(region, HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT,
                            &record.allocation_alignment);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }

  regions->push_back(record);
  return HSA_STATUS_SUCCESS;
}

template <typename T>
bool agent_info(hsa_agent_t agent, hsa_agent_info_t attribute, T *value,
                const char *operation) {
  return check(hsa_agent_get_info(agent, attribute, value), operation);
}

bool read_file(const std::string &path, std::vector<char> *bytes) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    std::cerr << "failed to open HSACO: " << path << "\n";
    return false;
  }
  const std::streampos end = input.tellg();
  if (end <= 0 ||
      static_cast<uintmax_t>(end) > std::numeric_limits<size_t>::max()) {
    std::cerr << "invalid HSACO size: " << path << "\n";
    return false;
  }
  bytes->resize(static_cast<size_t>(end));
  input.seekg(0, std::ios::beg);
  input.read(bytes->data(), static_cast<std::streamsize>(bytes->size()));
  if (!input) {
    std::cerr << "failed to read HSACO: " << path << "\n";
    return false;
  }
  return true;
}

struct KernelPrintState {
  size_t count = 0;
};

hsa_status_t print_kernel(hsa_executable_t, hsa_agent_t,
                          hsa_executable_symbol_t symbol, void *data) {
  hsa_symbol_kind_t kind = HSA_SYMBOL_KIND_VARIABLE;
  hsa_status_t status = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind);
  if (status != HSA_STATUS_SUCCESS || kind != HSA_SYMBOL_KIND_KERNEL) {
    return status;
  }

  uint32_t name_length = 0;
  status = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_length);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  std::vector<char> name(name_length + 1, '\0');
  status = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data());
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  uint64_t object = 0;
  uint32_t kernarg_size = 0;
  uint32_t kernarg_alignment = 0;
  uint32_t group_segment_size = 0;
  uint32_t private_segment_size = 0;
  bool dynamic_callstack = false;
  const struct Query {
    hsa_executable_symbol_info_t attribute;
    void *value;
  } queries[] = {
      {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &object},
      {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg_size},
      {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
       &kernarg_alignment},
      {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
       &group_segment_size},
      {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
       &private_segment_size},
      {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK, &dynamic_callstack},
  };
  for (const Query &query : queries) {
    status =
        hsa_executable_symbol_get_info(symbol, query.attribute, query.value);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }

  auto *state = static_cast<KernelPrintState *>(data);
  const size_t index = state->count++;
  std::cout << "kernel." << index << ".name=" << std::quoted(name.data())
            << "\n";
  std::cout << "kernel." << index << ".object=0x" << std::hex << object
            << std::dec << "\n";
  std::cout << "kernel." << index << ".kernarg_size=" << kernarg_size << "\n";
  std::cout << "kernel." << index << ".kernarg_alignment=" << kernarg_alignment
            << "\n";
  std::cout << "kernel." << index
            << ".group_segment_size=" << group_segment_size << "\n";
  std::cout << "kernel." << index
            << ".private_segment_size=" << private_segment_size << "\n";
  std::cout << "kernel." << index
            << ".dynamic_callstack=" << (dynamic_callstack ? 1 : 0) << "\n";
  return HSA_STATUS_SUCCESS;
}

bool print_loaded_hsaco(hsa_agent_t agent, hsa_profile_t profile,
                        const std::string &path) {
  std::vector<char> bytes;
  if (!read_file(path, &bytes)) {
    return false;
  }

  hsa_code_object_reader_t reader{};
  hsa_executable_t executable{};
  hsa_loaded_code_object_t loaded{};
  if (!check(hsa_code_object_reader_create_from_memory(bytes.data(),
                                                       bytes.size(), &reader),
             "hsa_code_object_reader_create_from_memory")) {
    return false;
  }

  hsa_status_t status = hsa_executable_create_alt(
      profile, HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR, nullptr, &executable);
  if (status != HSA_STATUS_SUCCESS) {
    check(status, "hsa_executable_create_alt");
    hsa_code_object_reader_destroy(reader);
    return false;
  }

  status = hsa_executable_load_agent_code_object(executable, agent, reader,
                                                 nullptr, &loaded);
  if (status == HSA_STATUS_SUCCESS) {
    status = hsa_executable_freeze(executable, nullptr);
  }
  if (status != HSA_STATUS_SUCCESS) {
    check(status, "load/freeze executable");
    hsa_executable_destroy(executable);
    hsa_code_object_reader_destroy(reader);
    return false;
  }

  std::cout << "hsaco.path=" << std::quoted(path) << "\n";
  std::cout << "hsaco.size=" << bytes.size() << "\n";
  KernelPrintState print_state;
  status = hsa_executable_iterate_agent_symbols(executable, agent, print_kernel,
                                                &print_state);
  std::cout << "hsaco.kernel_count=" << print_state.count << "\n";

  const hsa_status_t executable_status = hsa_executable_destroy(executable);
  const hsa_status_t reader_status = hsa_code_object_reader_destroy(reader);
  return check(status, "hsa_executable_iterate_agent_symbols") &&
         check(executable_status, "hsa_executable_destroy") &&
         check(reader_status, "hsa_code_object_reader_destroy");
}

void usage(const char *program) {
  std::cerr << "Usage: " << program
            << " [--device-index INDEX] [--hsaco PATH]\n";
}

bool parse_index(const std::string &text, size_t *value) {
  try {
    size_t consumed = 0;
    const unsigned long long parsed = std::stoull(text, &consumed, 10);
    if (consumed != text.size() ||
        parsed > std::numeric_limits<size_t>::max()) {
      return false;
    }
    *value = static_cast<size_t>(parsed);
    return true;
  } catch (...) {
    return false;
  }
}

} // namespace

int main(int argc, char **argv) {
  size_t device_index = 0;
  std::string hsaco_path;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--device-index" && i + 1 < argc) {
      if (!parse_index(argv[++i], &device_index)) {
        std::cerr << "invalid device index\n";
        return 2;
      }
    } else if (argument == "--hsaco" && i + 1 < argc) {
      hsaco_path = argv[++i];
    } else if (argument == "--help" || argument == "-h") {
      usage(argv[0]);
      return 0;
    } else {
      usage(argv[0]);
      return 2;
    }
  }

  if (!check(hsa_init(), "hsa_init")) {
    return 1;
  }

  int result = 1;
  do {
    uint16_t hsa_version_major = 0;
    uint16_t hsa_version_minor = 0;
    uint64_t timestamp_frequency = 0;
    if (!check(hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR,
                                   &hsa_version_major),
               "hsa_system_get_info(version major)") ||
        !check(hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MINOR,
                                   &hsa_version_minor),
               "hsa_system_get_info(version minor)") ||
        !check(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY,
                                   &timestamp_frequency),
               "hsa_system_get_info(timestamp frequency)")) {
      break;
    }

    std::vector<AgentRecord> agents;
    if (!check(hsa_iterate_agents(collect_agent, &agents),
               "hsa_iterate_agents")) {
      break;
    }

    std::vector<AgentRecord> gpu_agents;
    size_t cpu_count = 0;
    for (const AgentRecord &agent : agents) {
      if (agent.type == HSA_DEVICE_TYPE_GPU) {
        gpu_agents.push_back(agent);
      } else if (agent.type == HSA_DEVICE_TYPE_CPU) {
        ++cpu_count;
      }
    }
    if (device_index >= gpu_agents.size()) {
      std::cerr << "GPU device index " << device_index << " is unavailable; "
                << gpu_agents.size() << " GPU agent(s) found\n";
      break;
    }

    const AgentRecord &selected = gpu_agents[device_index];
    hsa_profile_t profile = HSA_PROFILE_FULL;
    uint32_t node = 0;
    uint32_t wavefront_size = 0;
    uint16_t workgroup_max_dim[3] = {};
    uint32_t workgroup_max_size = 0;
    hsa_dim3_t grid_max_dim{};
    uint32_t queue_min_size = 0;
    uint32_t queue_max_size = 0;
    hsa_queue_type32_t queue_type = HSA_QUEUE_TYPE_MULTI;

    if (!agent_info(selected.agent, HSA_AGENT_INFO_PROFILE, &profile,
                    "hsa_agent_get_info(profile)") ||
        !agent_info(selected.agent, HSA_AGENT_INFO_NODE, &node,
                    "hsa_agent_get_info(node)") ||
        !agent_info(selected.agent, HSA_AGENT_INFO_WAVEFRONT_SIZE,
                    &wavefront_size, "hsa_agent_get_info(wavefront size)") ||
        !agent_info(selected.agent, HSA_AGENT_INFO_WORKGROUP_MAX_DIM,
                    &workgroup_max_dim,
                    "hsa_agent_get_info(workgroup max dimensions)") ||
        !agent_info(selected.agent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE,
                    &workgroup_max_size,
                    "hsa_agent_get_info(workgroup max size)") ||
        !agent_info(selected.agent, HSA_AGENT_INFO_GRID_MAX_DIM, &grid_max_dim,
                    "hsa_agent_get_info(grid max dimensions)") ||
        !agent_info(selected.agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE,
                    &queue_min_size, "hsa_agent_get_info(queue min size)") ||
        !agent_info(selected.agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                    &queue_max_size, "hsa_agent_get_info(queue max size)") ||
        !agent_info(selected.agent, HSA_AGENT_INFO_QUEUE_TYPE, &queue_type,
                    "hsa_agent_get_info(queue type)")) {
      break;
    }

    std::vector<RegionRecord> regions;
    if (!check(
            hsa_agent_iterate_regions(selected.agent, collect_region, &regions),
            "hsa_agent_iterate_regions")) {
      break;
    }

    std::cout << "schema.version=1\n";
    std::cout << "system.hsa_version=" << hsa_version_major << '.'
              << hsa_version_minor << "\n";
    std::cout << "system.timestamp_frequency=" << timestamp_frequency << "\n";
    std::cout << "system.agent_count=" << agents.size() << "\n";
    std::cout << "system.cpu_agent_count=" << cpu_count << "\n";
    std::cout << "system.gpu_agent_count=" << gpu_agents.size() << "\n";
    for (size_t i = 0; i < agents.size(); ++i) {
      std::cout << "agent." << i << ".name=" << std::quoted(agents[i].name)
                << "\n";
      std::cout << "agent." << i << ".type=" << device_type_name(agents[i].type)
                << "\n";
    }
    std::cout << "selected.device_index=" << device_index << "\n";
    std::cout << "selected.name=" << std::quoted(selected.name) << "\n";
    std::cout << "selected.node=" << node << "\n";
    std::cout << "selected.profile=" << profile_name(profile) << "\n";
    std::cout << "selected.wavefront_size=" << wavefront_size << "\n";
    std::cout << "selected.workgroup_max_dim=" << workgroup_max_dim[0] << ','
              << workgroup_max_dim[1] << ',' << workgroup_max_dim[2] << "\n";
    std::cout << "selected.workgroup_max_size=" << workgroup_max_size << "\n";
    std::cout << "selected.grid_max_dim=" << grid_max_dim.x << ','
              << grid_max_dim.y << ',' << grid_max_dim.z << "\n";
    std::cout << "selected.queue_min_size=" << queue_min_size << "\n";
    std::cout << "selected.queue_max_size=" << queue_max_size << "\n";
    std::cout << "selected.queue_type=" << static_cast<uint32_t>(queue_type)
              << "\n";
    std::cout << "selected.region_count=" << regions.size() << "\n";

    size_t selected_global_region = regions.size();
    size_t selected_kernarg_region = regions.size();
    for (size_t i = 0; i < regions.size(); ++i) {
      const RegionRecord &region = regions[i];
      std::cout << "region." << i
                << ".segment=" << region_segment_name(region.segment) << "\n";
      std::cout << "region." << i << ".size=" << region.size << "\n";
      std::cout << "region." << i
                << ".allocation_max_size=" << region.allocation_max_size
                << "\n";
      std::cout << "region." << i
                << ".allocation_allowed=" << (region.allocation_allowed ? 1 : 0)
                << "\n";
      std::cout << "region." << i << ".global_flags=0x" << std::hex
                << region.flags << std::dec << "\n";
      std::cout << "region." << i
                << ".allocation_granule=" << region.allocation_granule << "\n";
      std::cout << "region." << i
                << ".allocation_alignment=" << region.allocation_alignment
                << "\n";

      if (region.segment == HSA_REGION_SEGMENT_GLOBAL &&
          region.allocation_allowed &&
          selected_global_region == regions.size() &&
          (region.flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) != 0) {
        selected_global_region = i;
      }
      if (region.segment == HSA_REGION_SEGMENT_GLOBAL &&
          region.allocation_allowed &&
          selected_kernarg_region == regions.size() &&
          (region.flags & HSA_REGION_GLOBAL_FLAG_KERNARG) != 0) {
        selected_kernarg_region = i;
      }
    }
    std::cout << "selected.global_region=";
    if (selected_global_region == regions.size()) {
      std::cout << "none\n";
    } else {
      std::cout << selected_global_region << "\n";
    }
    std::cout << "selected.kernarg_region=";
    if (selected_kernarg_region == regions.size()) {
      std::cout << "none\n";
    } else {
      std::cout << selected_kernarg_region << "\n";
    }

    uint32_t requested_queue_size = 1024;
    while (requested_queue_size > queue_max_size) {
      requested_queue_size >>= 1;
    }
    requested_queue_size = std::max(requested_queue_size, 1U);
    hsa_queue_t *queue = nullptr;
    if (!check(hsa_queue_create(selected.agent, requested_queue_size,
                                HSA_QUEUE_TYPE_MULTI, nullptr, nullptr,
                                UINT32_MAX, UINT32_MAX, &queue),
               "hsa_queue_create")) {
      break;
    }
    std::cout << "queue.requested_size=" << requested_queue_size << "\n";
    std::cout << "queue.actual_size=" << queue->size << "\n";
    std::cout << "queue.type=" << static_cast<uint32_t>(queue->type) << "\n";
    std::cout << "queue.features=0x" << std::hex << queue->features << std::dec
              << "\n";
    std::cout << "queue.id=" << queue->id << "\n";
    std::cout << "queue.base_address=0x" << std::hex
              << reinterpret_cast<uintptr_t>(queue->base_address) << std::dec
              << "\n";
    std::cout << "queue.doorbell_signal=0x" << std::hex
              << queue->doorbell_signal.handle << std::dec << "\n";
    if (!check(hsa_queue_destroy(queue), "hsa_queue_destroy")) {
      break;
    }

    if (!hsaco_path.empty() &&
        !print_loaded_hsaco(selected.agent, profile, hsaco_path)) {
      break;
    }
    result = 0;
  } while (false);

  if (!check(hsa_shut_down(), "hsa_shut_down")) {
    return 1;
  }
  return result;
}
