#include "iree_adapter.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <stdio.h>
#include <utility>
#include <vector>

#ifndef LRRT_IREE_HSACO_PATH
#define LRRT_IREE_HSACO_PATH "build-iree-probe/minimal_mul_gfx1101.hsaco"
#endif

#ifndef LRRT_IREE_KERNEL_SYMBOL
#define LRRT_IREE_KERNEL_SYMBOL "simple_mul_dispatch_0_elementwise_4_f32"
#endif

namespace {

using lrrt::executor::iree::BindingMetadata;
using lrrt::executor::iree::ExecutableMetadata;
using lrrt::executor::iree::ExportMetadata;
using lrrt::executor::iree::KernargBuilder;

std::vector<unsigned char> read_file(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("failed to open IREE HSACO");
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("IREE HSACO is empty");
  }
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> data(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    throw std::runtime_error("failed to read IREE HSACO");
  }
  return data;
}

void expect_close(float actual, float expected, size_t index) {
  if (std::fabs(actual - expected) > 0.0001f) {
    fprintf(stderr, "iree_hsaco_dispatch: output[%zu]=%f expected=%f\n", index,
            actual, expected);
    throw std::runtime_error("IREE HSACO dispatch result mismatch");
  }
}

ExecutableMetadata simple_mul_metadata() {
  ExportMetadata export_metadata;
  export_metadata.symbol = LRRT_IREE_KERNEL_SYMBOL;
  export_metadata.ordinal = 0;
  export_metadata.workgroup_size = {32, 1, 1};
  export_metadata.subgroup_size = 32;
  export_metadata.bindings = {
      BindingMetadata{0, "storage_buffer", {"ReadOnly", "Indirect"}},
      BindingMetadata{1, "storage_buffer", {"ReadOnly", "Indirect"}},
      BindingMetadata{2, "storage_buffer", {"Indirect"}},
  };
  export_metadata.kernel.symbol = LRRT_IREE_KERNEL_SYMBOL;
  export_metadata.dispatch.executable = "simple_mul_dispatch_0";
  export_metadata.dispatch.variant = "rocm_hsaco_fb";
  export_metadata.dispatch.symbol = LRRT_IREE_KERNEL_SYMBOL;

  ExecutableMetadata metadata;
  metadata.target = "gfx1101";
  metadata.executable = "simple_mul_dispatch_0";
  metadata.variant = "rocm_hsaco_fb";
  metadata.exports = {std::move(export_metadata)};
  return metadata;
}

void run_simple_mul(lrrt::Device device, const lrrt::Kernel &kernel,
                    const ExportMetadata &export_metadata) {
  const std::vector<float> lhs = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> rhs = {10.0f, 20.0f, 30.0f, 40.0f};
  const std::vector<float> expected = {10.0f, 40.0f, 90.0f, 160.0f};
  std::vector<float> actual(expected.size(), 0.0f);

  lrrt::DeviceBuffer device_lhs(device, lhs.size() * sizeof(float));
  lrrt::DeviceBuffer device_rhs(device, rhs.size() * sizeof(float));
  lrrt::DeviceBuffer device_out(device, actual.size() * sizeof(float));

  lrrt::copy_to_device(device_lhs, lhs.data(), lhs.size() * sizeof(float));
  lrrt::copy_to_device(device_rhs, rhs.data(), rhs.size() * sizeof(float));

  const std::vector<void *> buffers = {
      device_lhs.data(),
      device_rhs.data(),
      device_out.data(),
  };
  const std::vector<unsigned char> kernargs =
      KernargBuilder(export_metadata).pack_global_buffers(buffers);
  const lr_launch_config_t config =
      export_metadata.launch_config_for_workgroups(lr_dim3_t{1, 1, 1});
  lrrt::launch(kernel, config, kernargs.data(), kernargs.size());
  lrrt::check(lr_synchronize(device.get()), "lr_synchronize");

  lrrt::copy_to_host(actual.data(), device_out, actual.size() * sizeof(float));
  for (size_t i = 0; i < actual.size(); ++i) {
    expect_close(actual[i], expected[i], i);
  }
}

} // namespace

int main() {
  try {
    lrrt::Runtime runtime;
    if (runtime.device_count() == 0) {
      printf("iree_hsaco_dispatch_smoke: skipped, no GPU devices\n");
      return 0;
    }

    lrrt::Device device = runtime.open_device(0);
    const ExecutableMetadata metadata = simple_mul_metadata();
    const ExportMetadata &export_metadata =
        metadata.require_export_by_ordinal(0);
    const std::vector<unsigned char> hsaco = read_file(LRRT_IREE_HSACO_PATH);
    lrrt::Module module(device, hsaco);
    lrrt::Kernel kernel = module.kernel(export_metadata.symbol.c_str());
    if (!kernel.get()) {
      throw std::runtime_error("IREE kernel lookup returned null");
    }
    run_simple_mul(device, kernel, export_metadata);

    printf("iree_hsaco_dispatch_smoke: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
