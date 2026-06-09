# light-rocm-runtime

A small experimental runtime for launching AMD GPU kernels from ROCm code
objects.

The first target is intentionally narrow:

- Linux with AMDGPU and an installed ROCm stack
- load a pre-built `.hsaco`
- allocate device memory
- copy host/device buffers
- launch one kernel
- synchronize and copy results back

This project is not trying to replace the full HIP runtime. The initial
implementation will use the HSA/ROCr API directly and keep a small public C ABI.

## Build

```sh
cmake -S . -B build
cmake --build build
```

The current runtime is a scaffold. HSA-backed execution will be added behind the
same public API.

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
