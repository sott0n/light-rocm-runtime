#include "light_rocr/transport/kfd/topology.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

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

class TemporaryTopology {
public:
  TemporaryTopology() {
    static unsigned counter = 0;
    path_ = fs::temp_directory_path() /
            ("light-rocr-kfd-topology-" + std::to_string(::getpid()) + "-" +
             std::to_string(counter++));
    fs::create_directories(path_ / "nodes");
    write("generation_id", "7\n");
  }

  TemporaryTopology(const TemporaryTopology &) = delete;
  TemporaryTopology &operator=(const TemporaryTopology &) = delete;

  ~TemporaryTopology() {
    std::error_code error;
    fs::remove_all(path_, error);
  }

  const fs::path &path() const { return path_; }

  void write(const fs::path &relative_path, const std::string &contents) {
    const fs::path destination = path_ / relative_path;
    fs::create_directories(destination.parent_path());
    std::ofstream output(destination);
    output << contents;
  }

private:
  fs::path path_;
};

const char *cpu_properties() {
  return "cpu_cores_count 64\n"
         "simd_count 0\n"
         "simd_per_cu 0\n"
         "wave_front_size 0\n"
         "max_waves_per_simd 0\n"
         "array_count 0\n"
         "simd_arrays_per_engine 0\n"
         "max_slots_scratch_cu 0\n"
         "gfx_target_version 0\n"
         "vendor_id 0\n"
         "device_id 0\n"
         "domain 0\n"
         "location_id 0\n"
         "drm_render_minor 0\n"
         "mem_banks_count 1\n";
}

const char *gpu_properties() {
  return "cpu_cores_count 0\n"
         "simd_count 120\n"
         "simd_per_cu 2\n"
         "wave_front_size 32\n"
         "max_waves_per_simd 16\n"
         "lds_size_in_kb 64\n"
         "array_count 6\n"
         "simd_arrays_per_engine 2\n"
         "max_slots_scratch_cu 32\n"
         "gfx_target_version 110001\n"
         "vendor_id 4098\n"
         "device_id 29822\n"
         "domain 7\n"
         "location_id 8960\n"
         "drm_render_minor 128\n"
         "local_mem_size 17163091968\n"
         "num_xcc 1\n"
         "cwsr_size 28835840\n"
         "ctl_stack_size 24576\n"
         "mem_banks_count 1\n"
         "unknown_future_property 99\n";
}

void add_node(TemporaryTopology *topology, uint32_t node_id, uint32_t gpu_id,
              const std::string &name, const std::string &properties,
              const std::string &bank_properties) {
  const fs::path node = fs::path("nodes") / std::to_string(node_id);
  topology->write(node / "gpu_id", std::to_string(gpu_id) + "\n");
  topology->write(node / "name", name + "\n");
  topology->write(node / "properties", properties);
  topology->write(node / "mem_banks/0/properties", bank_properties);
}

void successful_snapshot(TestContext *context) {
  TemporaryTopology topology;
  add_node(&topology, 2, 43288, "ip discovery", gpu_properties(),
           "heap_type 2\n"
           "size_in_bytes 17163091968\n"
           "flags 3\n"
           "width 256\n"
           "mem_clk_max 1218\n");
  add_node(&topology, 0, 0, "test-cpu", cpu_properties(),
           "heap_type 0\n"
           "size_in_bytes 68719476736\n"
           "flags 0\n"
           "width 64\n"
           "mem_clk_max 2667\n");

  const auto result = light_rocr::transport::kfd::read_topology_sysfs(
      topology.path().string(), {1, 14});
  context->expect(static_cast<bool>(result), result.message);
  context->expect(result.topology.kfd_version.major == 1 &&
                      result.topology.kfd_version.minor == 14,
                  "KFD version was not retained");
  context->expect(result.topology.nodes.size() == 2, "unexpected node count");
  if (result.topology.nodes.size() != 2) {
    return;
  }

  const auto &cpu = result.topology.nodes[0];
  context->expect(cpu.node_id == 0 && !cpu.is_gpu() &&
                      cpu.cpu_core_count == 64 && cpu.name == "test-cpu",
                  "CPU node was not converted");

  const auto &gpu = result.topology.nodes[1];
  context->expect(gpu.node_id == 2 && gpu.gpu_id == 43288 && gpu.is_gpu(),
                  "GPU identity was not converted");
  context->expect(gpu.compute_unit_count() == 60 &&
                      gpu.shader_engine_count == 3 && gpu.xcc_count == 1 &&
                      gpu.lds_size_kb == 64 && gpu.cwsr_size == 28835840 &&
                      gpu.control_stack_size == 24576,
                  "GPU compute topology was not converted");
  context->expect(gpu.maximum_scratch_waves_per_compute_unit == 32,
                  "GPU scratch topology was not converted");
  context->expect(gpu.architecture.major == 11 && gpu.architecture.minor == 0 &&
                      gpu.architecture.stepping == 1 && gpu.name == "GFX0b0001",
                  "gfx target was not converted");
  context->expect(gpu.vendor_id == 0x1002 && gpu.device_id == 0x747e &&
                      gpu.domain == 7 && gpu.location_id == 0x2300 &&
                      gpu.drm_render_minor == 128,
                  "GPU PCI identity was not converted");
  context->expect(gpu.memory_banks.size() == 1,
                  "unexpected GPU memory-bank count");
  if (!gpu.memory_banks.empty()) {
    const auto &bank = gpu.memory_banks[0];
    context->expect(
        bank.heap_type ==
                light_rocr::runtime::MemoryHeapType::FrameBufferPrivate &&
            bank.size == 17163091968ULL && bank.width == 256 &&
            bank.maximum_clock_mhz == 1218 && bank.hot_pluggable &&
            bank.non_volatile,
        "GPU memory bank was not converted");
  }
}

void missing_generation_is_reported(TestContext *context) {
  TemporaryTopology topology;
  fs::remove(topology.path() / "generation_id");
  const auto result = light_rocr::transport::kfd::read_topology_sysfs(
      topology.path().string(), {1, 14});
  context->expect(!result, "missing generation unexpectedly succeeded");
  context->expect(
      result.error ==
          light_rocr::transport::kfd::DiscoveryError::ReadGeneration,
      "wrong missing-generation error");
}

void malformed_gpu_topology_is_rejected(TestContext *context) {
  TemporaryTopology topology;
  std::string properties = gpu_properties();
  const size_t position = properties.find("simd_arrays_per_engine 2");
  properties.replace(position, std::string("simd_arrays_per_engine 2").size(),
                     "simd_arrays_per_engine 4");
  add_node(&topology, 1, 43288, "ip discovery", properties,
           "heap_type 2\n"
           "size_in_bytes 17163091968\n"
           "flags 0\n"
           "width 256\n"
           "mem_clk_max 1218\n");

  const auto result = light_rocr::transport::kfd::read_topology_sysfs(
      topology.path().string(), {1, 14});
  context->expect(!result, "malformed shader topology unexpectedly succeeded");
  context->expect(result.error ==
                      light_rocr::transport::kfd::DiscoveryError::InvalidNode,
                  "wrong malformed-node error");
}

void missing_memory_bank_is_reported(TestContext *context) {
  TemporaryTopology topology;
  const fs::path node = "nodes/1";
  topology.write(node / "gpu_id", "43288\n");
  topology.write(node / "name", "ip discovery\n");
  topology.write(node / "properties", gpu_properties());

  const auto result = light_rocr::transport::kfd::read_topology_sysfs(
      topology.path().string(), {1, 14});
  context->expect(!result, "missing memory bank unexpectedly succeeded");
  context->expect(
      result.error ==
          light_rocr::transport::kfd::DiscoveryError::ReadMemoryBank,
      "wrong missing-memory-bank error");
}

template <typename BeforeClose>
bool write_fifo(const fs::path &path, const std::string &value,
                BeforeClose before_close) {
  int fd = -1;
  for (unsigned attempt = 0; attempt < 1000; ++attempt) {
    fd = ::open(path.c_str(), O_WRONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd >= 0) {
      break;
    }
    if (errno != ENXIO) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (fd < 0) {
    return false;
  }

  const ssize_t written = ::write(fd, value.data(), value.size());
  before_close();
  const int close_result = ::close(fd);
  // Let the current reader observe EOF and close before opening the FIFO for
  // the next generation read.
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  return written == static_cast<ssize_t>(value.size()) && close_result == 0;
}

void changed_generation_retries_transient_failure(TestContext *context) {
  TemporaryTopology topology;
  add_node(&topology, 1, 43288, "ip discovery", gpu_properties(),
           "heap_type 2\n"
           "size_in_bytes 17163091968\n"
           "flags 0\n"
           "width 256\n"
           "mem_clk_max 1218\n");

  const fs::path properties_path = topology.path() / "nodes/1/properties";
  fs::remove(properties_path);
  const fs::path generation_path = topology.path() / "generation_id";
  fs::remove(generation_path);
  if (::mkfifo(generation_path.c_str(), 0600) != 0) {
    context->expect(false, "failed to create generation FIFO");
    return;
  }

  std::atomic<bool> writer_succeeded{true};
  std::thread writer([&] {
    const bool succeeded =
        write_fifo(generation_path, "7\n", [] {}) &&
        write_fifo(
            generation_path, "8\n",
            [&] { topology.write("nodes/1/properties", gpu_properties()); }) &&
        write_fifo(generation_path, "8\n", [] {}) &&
        write_fifo(generation_path, "8\n", [] {});
    writer_succeeded.store(succeeded);
  });

  const auto result = light_rocr::transport::kfd::read_topology_sysfs(
      topology.path().string(), {1, 14});
  writer.join();

  context->expect(writer_succeeded.load(), "generation FIFO writer failed");
  context->expect(static_cast<bool>(result), result.message);
  context->expect(result.topology.nodes.size() == 1 &&
                      result.topology.nodes[0].gpu_id == 43288,
                  "changed topology was not read on retry");
}

void invalid_session_is_reported(TestContext *context) {
  light_rocr::transport::kfd::KfdSession session;
  const auto result = light_rocr::transport::kfd::discover_topology(session);
  context->expect(!result, "invalid session unexpectedly succeeded");
  context->expect(
      result.error ==
          light_rocr::transport::kfd::DiscoveryError::InvalidSession,
      "wrong invalid-session error");
}

} // namespace

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"successful snapshot", successful_snapshot},
      {"missing generation", missing_generation_is_reported},
      {"malformed GPU topology", malformed_gpu_topology_is_rejected},
      {"missing memory bank", missing_memory_bank_is_reported},
      {"changed generation retry",
       changed_generation_retries_transient_failure},
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
