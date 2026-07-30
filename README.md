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

## Design Summary

`light-rocm-runtime` is an execution substrate, not a compiler, tensor runtime,
or full HIP replacement. The core is intentionally limited to:

- device discovery and low-level resource ownership
- device memory allocation and copies
- pre-built HSACO loading and kernel symbol resolution
- asynchronous kernel dispatch through HSA queues
- event dependencies, synchronization, and lightweight timing

Compiler layers own kernel generation, target selection, and static launch
metadata. Executor layers own tensor and weight lifetimes, memory arenas,
workspaces, KV caches, queue selection, and graph or model scheduling. The
runtime keeps these policies outside the core so the same dispatch path can be
used by Triton, IREE, and other compiler integrations.

The default execution path favors predictable lifetimes: launches are
asynchronous, but synchronous copies and resource destruction may drain pending
device work. Multiple queues can overlap independent work, while cross-queue
ordering must be expressed with explicit event dependencies.

The current implementation has deliberate limitations. It requires
Linux/AMDGPU and ROCr/HSA, loads pre-built HSACO only, is not HIP API
compatible, and does not provide peer-to-peer multi-GPU scheduling, managed
memory APIs, stream-ordered allocation, graph capture, dynamic launch
inference, or a general tensor/kernel ABI. Initialization and runtime state are
process-global, and several safety-oriented operations introduce broad
synchronization.

See [Runtime Design](docs/design.md) for the complete responsibility boundaries,
execution model, limitations, intentional non-goals, and criteria for extending
the runtime core.

## Executors

The executor layer currently supports two compiler integration paths:

- **Triton** loads generated kernel bundles and provides the reference path for
  executor-level scheduling, validation, and benchmarking.
- **IREE** provides an experimental HAL adapter that runs compiled IREE modules
  through the same lrrt resource and dispatch path.

Qwen2/Qwen2.5 0.5B serves as the end-to-end model integration. The IREE path
covers checkpoint conversion, prompt prefill, device-resident KV cache, and
autoregressive text generation; the Triton executor runs the full decoder stack
for correctness validation and detailed benchmarking. See the [Qwen IREE and
Triton execution guide](examples/qwen/README.md) for prerequisites, build and
conversion steps, validation, and benchmark commands.

## Build

```sh
cmake -S . -B build
cmake --build build
```

The runtime uses the HSA/ROCr API directly for device memory, code object
loading, and kernel dispatch.

Triton executor examples are opt-in and are not part of the default build. They
use `uv` to resolve `examples/triton/requirements.txt` and compile Triton
kernels with Python 3.13 by default:

```sh
cmake -S . -B build-triton -DLRRT_ENABLE_TRITON_EXAMPLES=ON
cmake --build build-triton
ctest --test-dir build-triton --output-on-failure -R lrrt_triton
```

Use `-DLRRT_TRITON_PYTHON=/path/to/python3.13` or another `uv --python` value
to override the Python used for Triton bundle generation.

The IREE executor and its experimental HAL adapter are also opt-in and are not
part of the default build. They use the pinned IREE submodule:

```sh
git submodule update --init third_party/iree
cmake -S . -B build-iree \
  -DLRRT_ENABLE_IREE_ADAPTER=ON
```

Use `-DLRRT_IREE_ROOT=/path/to/iree` to point at a different IREE source or
install tree. The Qwen guide covers the additional IREE compiler-tool build and
model-specific setup required for end-to-end execution.

## Benchmarks

Runtime benchmarks are opt-in:

```sh
cmake -S . -B build-bench -DLRRT_BUILD_BENCHMARKS=ON
cmake --build build-bench
```

| Executable | Measures |
| --- | --- |
| `lrrt_launch_overhead_benchmark` | Kernel enqueue, synchronization, dispatch throughput, and round-trip cost |
| `lrrt_async_copy_launch_benchmark` | Host waits compared with device-side copy/launch dependencies |
| `lrrt_pinned_host_transfer_benchmark` | Pageable and pinned H2D/D2H latency and bandwidth |
| `lrrt_double_buffer_pipeline_benchmark` | Sequential and double-buffered CPU preparation, H2D, and GPU work |

Run them from the build directory:

```sh
./build-bench/lrrt_launch_overhead_benchmark
./build-bench/lrrt_async_copy_launch_benchmark
./build-bench/lrrt_pinned_host_transfer_benchmark
./build-bench/lrrt_double_buffer_pipeline_benchmark
```

Results vary with the GPU, CPU, system load, and ROCm version. Use repeated
runs in the same environment when comparing changes.

## Pinned Host Double-buffer Pipeline

`lrrt::PinnedHostDoubleBuffer` overlaps preparation and H2D transfer for one
chunk with GPU work for another:

```cpp
lrrt::Queue queue(device);
lrrt::PinnedHostDoubleBuffer pipeline(device, chunk_bytes);

for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
  auto &slot = pipeline.acquire(); // waits only when this slot is reused
  prepare_chunk(slot.host_data(), chunk_bytes, chunk);
  pipeline.copy_to_device_async(slot, chunk_bytes);

  Args args = make_args(slot.device_buffer(), chunk);
  lrrt::launch(queue, kernel, config, args, {&slot.copy_complete()});
  pipeline.mark_work_submitted(slot, queue);
}
pipeline.finish();
```

`mark_work_submitted` records a marker after the work already enqueued on
`queue`. Consequently, all device work that consumes a slot must be submitted
before this call. `finish` drains both slots and makes them available for
another pipeline pass.

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
markers. `lr_event_duration_ns` reports the profiled duration of one completed
operation, including exact copy-engine duration for an asynchronous copy.

Asynchronous copies and kernel dispatches are ordered through device-side
signals. `lr_memcpy_async` passes pending dispatch completion signals to HSA as
copy dependencies, while `lr_launch` inserts HSA barrier packets for pending
async-copy signals. Neither direction waits for completion on the host.
Destroying or re-recording an event that is still referenced by a queued
dependency first retires completed queue barriers without blocking. The runtime
drains the device before reusing the signal only when an active queued
operation still references it.

## Memory Statistics

The runtime exposes lightweight per-device memory counters through
`lr_get_memory_stats` and `lr_reset_memory_stats`. The C++ wrapper provides the
same data as `lrrt::Device::memory_stats()` and resets it with
`lrrt::Device::reset_memory_stats()`.

`lr_memory_stats_t` tracks work that flows through lrrt:

- live, peak, allocated, and freed bytes for device and pinned-host allocations
- device and pinned-host allocation and free counts
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

`Runtime`, `DeviceBuffer`, `PinnedHostBuffer`, and `Module` manage their runtime
resources through constructors and destructors. `PinnedHostBuffer` provides
device-scoped page-locked host storage for predictable asynchronous H2D and D2H
copies. `Device` and `Kernel` are lightweight wrappers around the C handles;
their lifetimes are still tied to the runtime/module that created them.

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
