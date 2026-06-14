# light-rocm-runtime

A small experimental runtime for launching AMD GPU kernels from ROCm code
objects.

`light-rocm-runtime` aims to stay small while providing a low-overhead
dispatcher and predictable resource manager for compiler-generated AMD GPU
workloads. Compiler or executor layers are expected to own operator lowering,
kernel generation, scheduling, and tensor graph semantics; this runtime owns the
execution path for loading code objects, managing resources, dispatching
kernels, and synchronizing work.

The first target is intentionally narrow:

- Linux with AMDGPU and an installed ROCm stack
- load a pre-built `.hsaco`
- allocate device memory
- copy host/device buffers
- launch one kernel
- synchronize and copy results back

This project is not trying to replace the full HIP runtime. The implementation
uses the HSA/ROCr API directly and keeps a small public C ABI.

In the longer compiler-oriented stack, the runtime is the boundary between
generated kernel bundles and the AMD GPU. Its performance role is to keep
dispatch, code object, memory, and synchronization overhead predictable; kernel
math performance and pipeline scheduling remain the responsibility of compiler
and executor layers.

## Build

```sh
cmake -S . -B build
cmake --build build
```

The runtime uses the HSA/ROCr API directly for device memory, code object
loading, and kernel dispatch.

## C++ Usage

The C++ wrapper in `lrrt/lrrt.hpp` keeps the C ABI underneath, but provides
small RAII/value helpers for the common flow:

```cpp
#include "lrrt/lrrt.hpp"

#include <vector>

std::vector<unsigned char> read_hsaco(const char *path);

struct args_t {
  const float *in;
  float *out;
  int n;
};

int main() {
  lrrt::Runtime runtime;
  lrrt::Device device = runtime.open_device(0);

  std::vector<float> in(64, 1.0f);
  std::vector<float> out(64, 0.0f);

  lrrt::DeviceBuffer device_in(device, in.size() * sizeof(float));
  lrrt::DeviceBuffer device_out(device, out.size() * sizeof(float));
  lrrt::copy_to_device(device_in, in.data(), in.size() * sizeof(float));

  std::vector<unsigned char> hsaco = read_hsaco("my_kernel.hsaco");
  lrrt::Module module(device, hsaco);
  lrrt::Kernel kernel = module.kernel("my_kernel");

  args_t args = {
      static_cast<const float *>(device_in.data()),
      static_cast<float *>(device_out.data()),
      static_cast<int>(in.size()),
  };
  lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
  lrrt::launch(kernel, config, args);
  device.synchronize();

  lrrt::copy_to_host(out.data(), device_out, out.size() * sizeof(float));
}
```

`Runtime`, `DeviceBuffer`, and `Module` manage their runtime resources through
constructors and destructors. `Device` and `Kernel` are lightweight wrappers
around the C handles; their lifetimes are still tied to the runtime/module that
created them.

## Development GPU

This repository is currently developed and tested on:

- GPU: AMD Radeon RX 7800 XT
- HSA agent name: `gfx1101`
- ISA names:
  - `amdgcn-amd-amdhsa--gfx1101`
  - `amdgcn-amd-amdhsa--gfx11-generic`
- Compute units: 60
- Wavefront size: 32
- Workgroup max size: 1024
- Queue max size: 131072
- VRAM reported by ROCm: 16760832 KB
- XNACK: disabled

## AMD Dependencies

The current implementation uses the ROCr/HSA runtime directly. The versions
below describe the AMD stack used for development:

- ROCm: `6.4.4`
- ROCr/HSA runtime package: `hsa-rocr 1.15.0.60404-129~22.04`
- ROCr/HSA development package: `hsa-rocr-dev 1.15.0.60404-129~22.04`
- HSA runtime API: `1.15`
- HSA runtime extension API: `1.7`
- HSA runtime library: `/opt/rocm-6.4.4/lib/libhsa-runtime64.so.1`
- ROCm LLVM: `19.0.0.25224.60404-129~22.04`
- AMD Comgr: `3.0.0.60404-129~22.04`
- rocminfo: `1.0.0.60404-129~22.04`
