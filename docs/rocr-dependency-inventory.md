# ROCr Dependency Inventory and Baseline

## Purpose

This document defines the current ROCr dependency contract that the native
runtime must replace. The existing ROCr backend remains the correctness oracle
while the replacement progresses through two transports:

```text
current:     LRRT -> ROCr/libhsa-runtime64 -> libhsakmt -> /dev/kfd
phase one:   LRRT -> native runtime/loader -> libhsakmt -> /dev/kfd
phase two:   LRRT -> native runtime/loader -> /dev/kfd
```

The inventory intentionally covers only direct HSA calls made by the LRRT core.
Compiler and executor integrations call the LRRT API and do not add HSA calls
to this contract.

## Supported baseline

The first native implementation is constrained to:

- Linux and one process
- one `gfx1101` GPU
- the AMDGPU code-object ABI produced by the repository's current toolchain
- compute AQL queues and ordinary kernel dispatch
- user signals with active polling
- VRAM/GTT allocations required by current LRRT workloads

Multi-GPU behavior, IPC, XNACK variants, debugging, complete HSA compatibility,
and KFD internals below the public UAPI are outside this first baseline.

## Direct HSA function inventory

The core currently calls 35 distinct HSA functions. Source groups below are
the ownership boundary to preserve during backend extraction.

| Responsibility | HSA functions | Current source | Native/KMT replacement | Direct-KFD replacement |
| --- | --- | --- | --- | --- |
| Runtime lifecycle | `hsa_init`, `hsa_shut_down` | `src/runtime.cpp` | `hsaKmtOpenKFD`, topology snapshot lifetime, native registries | open/close `/dev/kfd`, UAPI version check, native registries |
| Agent discovery | `hsa_iterate_agents`, `hsa_agent_get_info` | `src/runtime.cpp`, `src/module.cpp`, `src/queue.cpp` | `hsaKmtAcquireSystemProperties`, `hsaKmtGetNodeProperties` | KFD topology sysfs plus KFD version/properties ioctls |
| Memory-region discovery | `hsa_agent_iterate_regions`, `hsa_region_get_info` | `src/runtime.cpp` | node memory properties and aperture information | topology memory banks and process apertures |
| System timebase | `hsa_system_get_info` | `src/event.cpp` | KMT clock counters or a defined host-clock fallback | `AMDKFD_IOC_GET_CLOCK_COUNTERS` or a defined host-clock fallback |
| Device allocation | `hsa_memory_allocate`, `hsa_memory_assign_agent`, `hsa_memory_free` | `src/memory.cpp`, `src/queue.cpp` | `hsaKmtAllocMemory`, `hsaKmtMapMemoryToGPU`, unmap/free | alloc/map/unmap/free memory ioctls and VA management |
| Host pinning | `hsa_amd_memory_lock`, `hsa_amd_memory_unlock` | `src/memory.cpp`, `src/event.cpp` | KMT user-pointer memory registration and mapping | KFD USERPTR allocation/map/unmap/free |
| Synchronous copy | `hsa_memory_copy` | `src/memory.cpp` | CPU mapping or native copy kernel | CPU mapping or native copy kernel |
| Async copy | `hsa_amd_memory_async_copy` | `src/memory.cpp` | copy kernel first; SDMA is a later feature | copy kernel first; direct SDMA queue later |
| Queue lifecycle | `hsa_queue_create`, `hsa_queue_destroy` | `src/queue.cpp` | ring/EOP allocation and `hsaKmtCreateQueue`/destroy | queue create/destroy ioctls and doorbell mmap |
| Queue publication | `hsa_queue_add_write_index_scacq_screl`, `hsa_signal_store_screlease` | `src/queue.cpp`, `src/event.cpp`, `src/launch.cpp` | native atomic write index and mapped doorbell store | same; transport only creates the queue and mapping |
| Signal lifecycle | `hsa_signal_create`, `hsa_signal_destroy` | `src/event.cpp`, `src/queue.cpp` | aligned native signal storage; KMT event only for blocking waits | aligned native signal storage; KFD event ioctl only for blocking waits |
| Signal operations | `hsa_signal_load_scacquire`, `hsa_signal_store_relaxed`, `hsa_signal_store_screlease`, `hsa_signal_wait_scacquire` | `src/device_synchronization.cpp`, `src/event.cpp`, `src/queue.cpp` | native atomics and active polling initially | same |
| Code-object reader | `hsa_code_object_reader_create_from_memory`, `hsa_code_object_reader_destroy` | `src/module.cpp` | bounded AMDHSA ELF parser | same loader; transport-independent |
| Executable lifecycle | `hsa_executable_create_alt`, `hsa_executable_load_agent_code_object`, `hsa_executable_freeze`, `hsa_executable_destroy` | `src/module.cpp` | build a load plan, allocate executable segments, copy and relocate | same loader; allocation supplied by KFD transport |
| Kernel discovery | `hsa_executable_iterate_agent_symbols`, `hsa_executable_symbol_get_info` | `src/module.cpp` | metadata/symbol/descriptor lookup | same loader; transport-independent |
| Profiling setup | `hsa_amd_profiling_set_profiler_enabled`, `hsa_amd_profiling_async_copy_enable` | `src/queue.cpp`, `src/memory.cpp` | initially unsupported; add native timestamp capture later | initially unsupported; add native timestamp capture later |
| Profiling queries | `hsa_amd_profiling_get_dispatch_time`, `hsa_amd_profiling_get_async_copy_time` | `src/event.cpp` | initially host timing; GPU timestamps are a later feature | initially host timing; GPU timestamps are a later feature |

### Functions by exact name

```text
hsa_agent_get_info
hsa_agent_iterate_regions
hsa_amd_memory_async_copy
hsa_amd_memory_lock
hsa_amd_memory_unlock
hsa_amd_profiling_async_copy_enable
hsa_amd_profiling_get_async_copy_time
hsa_amd_profiling_get_dispatch_time
hsa_amd_profiling_set_profiler_enabled
hsa_code_object_reader_create_from_memory
hsa_code_object_reader_destroy
hsa_executable_create_alt
hsa_executable_destroy
hsa_executable_freeze
hsa_executable_iterate_agent_symbols
hsa_executable_load_agent_code_object
hsa_executable_symbol_get_info
hsa_init
hsa_iterate_agents
hsa_memory_allocate
hsa_memory_assign_agent
hsa_memory_copy
hsa_memory_free
hsa_queue_add_write_index_scacq_screl
hsa_queue_create
hsa_queue_destroy
hsa_region_get_info
hsa_shut_down
hsa_signal_create
hsa_signal_destroy
hsa_signal_load_scacquire
hsa_signal_store_relaxed
hsa_signal_store_screlease
hsa_signal_wait_scacquire
hsa_system_get_info
```

To reproduce the list from the current source tree:

```sh
rg -o --no-filename \
  --glob '!third_party/**' --glob '!build*/**' \
  '\bhsa[A-Za-z0-9_]*(?=\s*\()' \
  src include tests examples executor benchmarks CMakeLists.txt -P |
  sort -u
```

Any newly introduced HSA call must be added to this contract or rejected in
review. Native code must not acquire an untracked HSA dependency.

## ROCr oracle diagnostic

`lrrt_rocr_baseline` queries ROCr directly instead of going through LRRT. It
records the values that the native runtime and loader must reproduce or explain:

- system timestamp frequency and HSA agent inventory
- reported HSA system version
- selected GPU node, profile, wavefront, grid, workgroup, and queue limits
- every legacy HSA region and LRRT's selected global/kernarg regions
- an actual ROCr queue's size, type, features, base, and doorbell signal
- optional HSACO loading and all kernel symbol properties consumed by LRRT

The target is excluded from the default build and exists only when ROCr/HSA is
available:

```sh
cmake -S . -B build
cmake --build build --target lrrt_rocr_baseline -j2
./build/lrrt_rocr_baseline \
  --device-index 0 \
  --hsaco ./build/vector_add_kernel.hsaco
```

The output is line-oriented `key=value` data so it can be diffed against future
KMT and direct-KFD diagnostics. Addresses, queue IDs, and doorbell handles are
expected to vary between processes; topology, limits, region properties, and
kernel segment properties are the stable comparison fields.

### Measured `gfx1101` oracle

The baseline tool was run on 2026-09-06 against
`build/vector_add_kernel.hsaco`. Stable values from that run were:

| Property | ROCr result |
| --- | --- |
| Reported HSA system version | 1.15 |
| GPU agents | 1 |
| Selected GPU/node | `gfx1101` / node 1 |
| HSA profile | base |
| Wavefront size | 32 |
| Workgroup maximum | 1024 work-items |
| Queue minimum/maximum | 64 / 131072 packets |
| LRRT requested/actual queue | 1024 / 1024 packets |
| Queue features | `0x1` |
| GPU-visible regions | 7 legacy HSA regions |
| Selected device-global region | region 0, coarse-grained, 4096-byte granule/alignment |
| Selected kernarg region | region 4, fine-grained+kernarg, 4096-byte granule/alignment |
| Kernel symbol | `vector_add.kd` |
| ROCr kernarg size/alignment | 288 / 16 bytes |
| Fixed group/private segment | 0 / 272 bytes |
| Dynamic callstack | false |

The absolute queue ID, mappings, doorbell handle, memory capacities, and kernel
object address are deliberately not recorded as invariants. They vary with
process state or available host/device memory.

The code-object metadata declares an 8-byte kernarg alignment, while ROCr's
`HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT` query returned
16 bytes for the loaded kernel. The loader work must explain and reproduce this
normalization instead of assuming that the raw metadata value is the final HSA
executable property.

Baseline validation on the same host and date completed with all 29 tests in
the default `build` tree passing. The standalone `lrrt_vector_add` example also
reported one device and `vector_add: ok`. This is the initial correctness
snapshot; later backend gates must run the same commands rather than relying on
the recorded result.

## Code-object baseline

Inspect the same artifact independently of ROCr:

```sh
/opt/rocm/llvm/bin/llvm-readelf \
  --file-header --program-headers --notes --symbols --relocs \
  ./build/vector_add_kernel.hsaco
```

At the time this contract was introduced, the repository's `gfx1101`
vector-add artifact had the following properties:

- ELF64, little endian, `ET_DYN`, `EM_AMDGPU`
- AMDGPU-HSA OS/ABI, ABI version 4
- ELF flags selecting `gfx1101`
- four `PT_LOAD` segments, including an executable segment
- AMDGPU metadata target `amdgcn-amd-amdhsa--gfx1101`
- metadata version `[1, 2]`
- exported `vector_add` function and 64-byte `vector_add.kd` object
- no dynamic relocations in this particular artifact
- 288-byte kernarg segment, 8-byte kernarg alignment
- 0-byte fixed group segment and 272-byte fixed private segment
- wavefront size 32

The current artifact corpus spans two AMDGPU-HSA ELF ABI versions. Clang and
Triton artifacts use ABI version 4, while the current IREE raw-HSACO probe uses
ABI version 3. The initial light-rocr loader therefore accepts both versions
and retains the version in its parsed code-object model for later
version-specific metadata handling.

The absence of relocations is a property of this smoke artifact, not permission
to omit relocation support from the loader. The loader test corpus must include
every relocation form observed in Clang, Triton, and IREE artifacts used by the
project.

### Native metadata comparison

`light-rocr-inspect-hsaco` now decodes AMDHSA metadata 1.2, resolves each
metadata kernel to its 64-byte `.kd` dynamic symbol through `PT_DYNAMIC`, and
validates the descriptor against the metadata without requiring ELF section
headers. On 2026-09-06 its stable output was compared with
`lrrt_rocr_baseline` for Clang, Triton, and IREE artifacts. The comparison uses
symbol names rather than iteration order because ROCr's symbol iteration order
is not metadata order.

| Artifact | Kernels | kernarg size | public alignment | group/private | dynamic stack |
| --- | ---: | ---: | ---: | ---: | --- |
| Clang `vector_add_kernel.hsaco` | 1 | 288 | 16 | 0 / 272 | false |
| Triton `triton_rmsnorm_bundle/kernels.hsaco` | 1 | 56 | 16 | 0 / 0 | false |
| IREE `minimal_mul_gfx1101.hsaco` | 1 | 24 | 16 | 0 / 0 | false |
| Clang `async_copy_launch_kernel.hsaco` | 3 | 280, 272, 264 | 16 | 0 / 140, 0 / 140, 0 / 92 | false |

All stable executable properties matched ROCr. Descriptor virtual addresses
are pre-load ELF addresses in the native inspector, whereas ROCr reports
process-specific relocated kernel-object addresses, so those absolute values
are deliberately not compared. A corpus scan also parsed all 208 HSACO files
then present under the repository's `build`, `build-triton`, and
`build-iree-probe` trees without GPU access.

## Executable dependency baseline

The current dependency edge can be verified with:

```sh
ldd ./build/lrrt_vector_add | rg 'hsa|hsakmt|drm|elf'
```

The current ROCr build directly loads `libhsa-runtime64.so.1`. The native/KMT
milestone must remove that entry. The final direct-KFD milestone must also have
no `libhsakmt.so` entry.

## Correctness gates

Use the following order so a failure identifies one new layer at a time:

```sh
cmake --build build -j2
timeout 60s ctest --test-dir build --output-on-failure
./build/lrrt_vector_add
./build/lrrt_rocr_baseline --hsaco ./build/vector_add_kernel.hsaco
```

Compiler integration gates follow the core baseline:

1. Triton vector-add
2. Triton RMSNorm and matvec
3. Triton mini decoder layer
4. IREE raw-HSACO dispatch smoke
5. Qwen one layer
6. Qwen E2E

The first native runtime does not need to match ROCr performance. It must match
observable correctness, resource lifetime behavior, queue ordering, and kernel
properties for the supported feature subset.

## Backend feature matrix

| Feature | ROCr oracle | Native/KMT first gate | Native/KFD final gate |
| --- | --- | --- | --- |
| GPU discovery | Required | Required | Required |
| Coarse device allocation | Required | Required | Required |
| Kernarg allocation | Required | Required | Required |
| Host pinning | Required | Deferred | Required for Qwen gate |
| Synchronous copy | Required | Required | Required |
| Compute AQL queue | Required | Required | Required |
| User signal active wait | Required | Required | Required |
| Barrier-and packet | Required | Required after basic dispatch | Required |
| Custom HSACO loader | Oracle properties | Required | Required |
| Multiple queues | Required | Required after basic dispatch | Required |
| Async copy | Required | Copy-kernel implementation accepted initially | Required for full parity |
| SDMA queue | ROCr implementation detail | Deferred | Stretch after Qwen correctness |
| Dispatch timing | Required | Host timing accepted initially | GPU timing deferred |
| Async-copy timing | Required | Deferred | Deferred |
| Triton/IREE HSACO | Required | Required | Required |
| Qwen E2E | Required | Required before removing KMT | Final acceptance gate |

## Boundary decisions

- The custom loader owns ELF validation, metadata parsing, segment layout,
  relocation, and kernel descriptor discovery.
- Runtime transports own topology, virtual memory, mappings, queue creation,
  doorbells, and optional interrupt events.
- Queue packet construction, native signal atomics, synchronization semantics,
  and resource lifetime live above the transport.
- The ROCr backend remains available throughout development and must use the
  same public LRRT API tests.
- An HSA-compatible shared-library frontend is optional and must not become a
  prerequisite for the native LRRT path.

## Primary specifications and implementation references

- ROCr Runtime: <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocr-runtime>
- libhsakmt: <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocr-runtime/libhsakmt>
- Linux KFD UAPI: <https://github.com/torvalds/linux/blob/master/include/uapi/linux/kfd_ioctl.h>
- LLVM AMDGPU usage and code-object ABI: <https://llvm.org/docs/AMDGPUUsage.html>
- HSA system architecture: <https://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf>
- IREE AMDGPU ABI definitions: `third_party/iree/runtime/src/iree/hal/drivers/amdgpu/abi`
- IREE HSACO metadata parser: `third_party/iree/runtime/src/iree/hal/drivers/amdgpu/util/hsaco_metadata.c`
