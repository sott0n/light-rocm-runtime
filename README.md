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

The C/HSA runtime remains available as `lrrt::lrrt`. Compiler integrations can
use the separate `lrrt::bundle` C++ target, which parses the kernel manifest,
loads its HSACO, validates the kernarg layout, and derives the launch config.

## Build

```sh
cmake -S . -B build
cmake --build build
```

The runtime uses the HSA/ROCr API directly for device memory, code object
loading, and kernel dispatch.

Triton examples are opt-in and are not part of the default build. They use `uv`
to resolve `examples/triton/requirements.txt` and compile Triton kernels with
Python 3.13 by default:

```sh
cmake -S . -B build-triton -DLRRT_ENABLE_TRITON_EXAMPLES=ON
cmake --build build-triton
ctest --test-dir build-triton --output-on-failure -R lrrt_triton
```

Use `-DLRRT_TRITON_PYTHON=/path/to/python3.13` or another `uv --python` value
to override the Python used for Triton bundle generation.

## Launch Overhead Benchmark

The launch benchmark is opt-in and uses a minimal pre-built kernel to separate
host enqueue cost, synchronization cost, sustained dispatch throughput, and
device-side batch time:

```sh
cmake -S . -B build-bench -DLRRT_BUILD_BENCHMARKS=ON
cmake --build build-bench
./build-bench/lrrt_launch_overhead_benchmark
./build-bench/lrrt_async_copy_launch_benchmark
```

Pass an optional dispatch count to replace the default `10000` iterations.
Results are printed as a table with microseconds per launch and sustained
launches per second for straightforward comparison across runtime changes.
Interactive terminal output uses color; set `NO_COLOR=1` to disable it. Color
is disabled automatically when output is redirected or `TERM=dumb`.

- **Idle synchronize**: host cost of synchronizing an idle device
- **Host enqueue**: host enqueue cost while the queue has capacity
- **Device batch interval**: device event interval across a dispatch batch
- **Submit and synchronize**: sustained batched dispatch cost
- **Launch round trip**: serialized launch and synchronization cost

The values depend on the GPU, CPU, system load, and ROCm version. The benchmark
does not enforce performance thresholds; use repeated runs in the same
environment when comparing runtime changes.

The async dependency benchmark compares explicit host waits with device-side
dependencies in both copy-to-launch and launch-to-copy directions. It reports
submission time and the complete operation round trip. Pass an optional
iteration count; the default is `100` with a 4 MiB D2D copy in the
copy-to-launch direction.

## Development

Install the pre-commit runner and enable the repository hooks:

```sh
uv tool install pre-commit
pre-commit install
```

The hooks run Ruff linting and formatting for Python, clang-format for C/C++,
and basic file checks. Run the complete set manually with:

```sh
pre-commit run --all-files
```

## Execution Semantics

`lr_launch` is asynchronous with respect to the host. A successful return means
the kernel dispatch packet was enqueued on the device queue; it does not mean
the kernel has finished running.

Use `lr_synchronize` to wait for all previously enqueued work on a device.
Kernel execution errors are reported by `lr_synchronize` or by a later API call
that must drain pending work for safety.

Additional compute queues can be created with `lr_queue_create`. Kernels
submitted with `lr_launch_on_queue` do not receive implicit dependencies, so
independent queues can make progress concurrently. Cross-queue ordering is
expressed by recording an event with `lr_event_record_on_queue` and passing it
to `lr_launch_on_queue_with_dependencies`. `lr_queue_synchronize` waits for one
queue, while `lr_synchronize` remains the device-wide synchronization point.

The current memory and lifetime APIs are conservative:

- `lr_memcpy` waits for pending work before copying
- `lr_memcpy_async` adds device-side dependencies on earlier kernel dispatches,
  starts the copy, and reports completion through an event
- `lr_free` waits before releasing a runtime-managed allocation
- `lr_module_destroy` waits before destroying code object state for that device
- `lr_shutdown` waits for all devices before releasing runtime resources

These implicit synchronization points keep pointer and module lifetimes
predictable while the runtime is small. Code that wants explicit control should
call `lr_synchronize` before reading results or destroying resources that may be
used by queued kernels.

Events provide a lightweight way to mark queue progress and measure elapsed
time. `lr_event_record` enqueues a marker after previously submitted work on the
event's device, `lr_event_synchronize` waits for that marker, and
`lr_event_elapsed_time_ns` reports the nanoseconds between two completed
markers.

Asynchronous copies and kernel dispatches are ordered through device-side
signals. `lr_memcpy_async` passes pending dispatch completion signals to HSA as
copy dependencies, while `lr_launch` inserts HSA barrier packets for pending
async-copy signals. Neither direction waits for completion on the host.
Destroying or re-recording an event that is still referenced by a queued
dependency drains the device before reusing its signal.

## Memory Statistics

The runtime exposes lightweight per-device memory counters through
`lr_get_memory_stats` and `lr_reset_memory_stats`. The C++ wrapper provides the
same data as `lrrt::Device::memory_stats()` and resets it with
`lrrt::Device::reset_memory_stats()`.

`lr_memory_stats_t` tracks work that flows through lrrt:

- currently live bytes and peak live bytes for runtime-managed allocations
- total allocated and freed bytes
- allocation and free counts
- host-to-device, device-to-host, and device-to-device copy bytes
- total copy call count

These counters are intended for runtime debugging and benchmark regression
checks. For example, they can show whether an executor accidentally added many
small allocations, copied a full model more than once, or stopped using a
device-to-device handoff path.

The counters are not a ROCm global memory query. They do not report total GPU
VRAM, system-wide free VRAM, allocations made outside lrrt, kernel-side scratch
usage, or memory owned by HIP, PyTorch, Triton, or another runtime in the same
process. They only cover allocations and copies submitted through this runtime.

`lr_reset_memory_stats` preserves the current live byte count, sets the peak to
that current live value, and clears the accumulated allocation, free, and copy
totals. This makes it useful for measuring one benchmark phase after setup
buffers have already been allocated.

See [Resource Ownership Policy](docs/resource-ownership.md) for the boundary
between runtime-owned allocations, C++ RAII helpers, and executor-owned arena
policy.

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
  lrrt::Event start(device);
  lrrt::Event end(device);
  start.record();
  lrrt::launch(kernel, config, args);
  end.record();
  end.synchronize();
  start.synchronize();
  uint64_t elapsed_ns = lrrt::elapsed_time_ns(start, end);

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
