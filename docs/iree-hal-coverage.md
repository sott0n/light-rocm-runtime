# IREE HAL Adapter Coverage

This document tracks the current implementation coverage of the experimental
lrrt-backed IREE HAL adapter. It is a status table for what `--device=lrrt`
can currently exercise through the repo-local smoke path, not a full IREE HAL
compatibility claim.

The adapter goal remains narrow: let IREE own VMFB loading, graph/runtime
semantics, and HAL command construction while lrrt provides a low-overhead
dispatcher and predictable resource manager underneath the HAL boundary.

## Status Legend

| Status | Meaning |
| --- | --- |
| ✅ | Implemented and covered by repo tests or smoke runs |
| ✅/❌ | Partially implemented; useful path exists but the full IREE HAL behavior is not covered |
| ❌ | Explicitly unsupported or not implemented |

## Driver And Device

| Area | Current status | Current behavior | Coverage |
| --- | --- | --- | --- |
| Driver module registration | ✅ | Registers the `lrrt` HAL driver factory into an explicit IREE registry. | `lrrt_iree_hal_driver_registration_tests` |
| Default registry registration | ✅ | `lrrt_iree_hal_register_all` idempotently registers `lrrt` with IREE's default registry. | `lrrt_iree_hal_driver_registration_tests`, `lrrt_iree_run_module_smoke` |
| Driver creation by name | ✅ | Creates the native `lrrt` HAL driver from the registered factory. | `lrrt_iree_hal_driver_registration_tests` |
| Device enumeration | ✅ | Reports one placeholder `default` device. | `lrrt_iree_hal_driver_registration_tests` |
| Device creation | ✅ | Creates one `iree_hal_device_t` backed by lrrt device index `0`. | `lrrt_iree_hal_driver_registration_tests` |
| Device queries | ✅/❌ | Answers the minimal device-id and executable-format queries needed by the smoke path. | `lrrt_iree_hal_driver_registration_tests` |
| Device capabilities | ✅/❌ | Returns an empty capability set. | `lrrt_iree_hal_driver_registration_tests` |
| Multiple devices | ❌ | The driver exposes only one `default` device. | Documented unsupported behavior |
| Device topology policy | ❌ | No meaningful topology assignment or refinement is implemented. | Not covered beyond the vtable shape |

## Allocator And Buffers

| Area | Current status | Current behavior | Coverage |
| --- | --- | --- | --- |
| Allocator creation | ✅ | The HAL device owns an allocator that opens the lrrt runtime/device needed by buffers. | `lrrt_iree_hal_driver_allocator_tests` |
| Allocator trim | ✅/❌ | Accepts trim as a no-op for the current simple allocator. | `lrrt_iree_hal_driver_allocator_tests` |
| Memory heap query | ✅/❌ | Reports no explicit heap descriptors. | `lrrt_iree_hal_driver_allocator_tests` |
| Buffer compatibility query | ✅ | Reports allocatable transfer/dispatch compatibility for supported buffer params. | `lrrt_iree_hal_driver_allocator_tests` |
| Device-local buffer allocation | ✅ | Allocates HAL buffers backed by `lr_malloc` and releases them through `lr_free`. | `lrrt_iree_hal_driver_allocator_tests` |
| Host-local mapped buffer path | ✅/❌ | Uses an adapter-owned host shadow buffer and syncs with `lr_memcpy`; it is not a direct GPU mapping. | `lrrt_iree_hal_driver_allocator_tests` |
| Buffer map/read/write | ✅/❌ | Map APIs operate through the host shadow copy bridge. | `lrrt_iree_hal_driver_allocator_tests` |
| Buffer invalidate/flush | ✅/❌ | Syncs the host shadow and device allocation when a shadow allocation exists. | Covered by implementation contract; needs narrower tests |
| Queue alloca/dealloca | ✅/❌ | Allocates and releases buffers through synchronous adapter queue operations. | `lrrt_iree_hal_driver_allocator_tests` |
| Allocation pools | ❌ | Non-null queue allocation pools are rejected. | Implementation contract |
| External buffer import/export | ❌ | Returns explicit `UNIMPLEMENTED`; no external handle ownership exists yet. | `lrrt_iree_hal_driver_unsupported_tests` |
| Virtual/physical memory APIs | ❌ | Virtual memory support is reported unavailable. | `lrrt_iree_hal_driver_unsupported_tests` |
| Direct host-visible GPU mapping | ❌ | The adapter exposes a host shadow, not a direct mapped GPU allocation. | Documented buffer mapping behavior |

## Transfers And Files

| Area | Current status | Current behavior | Coverage |
| --- | --- | --- | --- |
| Queue update | ✅ | Synchronously copies host bytes into an lrrt-backed HAL buffer. | `lrrt_iree_hal_driver_transfer_tests` |
| Queue copy | ✅ | Synchronously copies between lrrt-backed HAL buffers. | `lrrt_iree_hal_driver_transfer_tests` |
| Queue fill | ✅/❌ | Synchronously materializes a host pattern and writes it to the target buffer. | Command-buffer transfer test |
| File import | ✅/❌ | Wraps IREE file handles for queue read/write smoke paths. | `lrrt_iree_hal_driver_transfer_tests` |
| Queue read | ✅/❌ | Reads host-backed file data into an lrrt-backed HAL buffer. | `lrrt_iree_hal_driver_transfer_tests` |
| Queue write | ✅/❌ | Writes an lrrt-backed HAL buffer into a host-backed file handle. | `lrrt_iree_hal_driver_transfer_tests` |
| DMA/async transfer semantics | ❌ | Current transfer operations are synchronous adapter operations over `lr_memcpy`. | Documented adapter limitation |

## Executables And Dispatch

| Area | Current status | Current behavior | Coverage |
| --- | --- | --- | --- |
| Executable cache creation | ✅ | Creates a HAL executable cache on the lrrt device. | `lrrt_iree_hal_driver_executable_tests` |
| Raw HSACO format negotiation | ✅ | Recognizes `rocm-hsaco` and `amdgpu-hsaco` formats. | `lrrt_iree_hal_driver_executable_tests` |
| HSACO format inference | ✅/❌ | Infers `rocm-hsaco` from an ELF-like HSACO prefix. | `lrrt_iree_hal_driver_executable_tests` |
| Raw HSACO prepare | ✅ | Loads raw HSACO bytes with `lr_module_load_hsaco`. | `lrrt_iree_hal_driver_executable_tests` |
| `rocm-hsaco-fb` VMFB executable payload | ✅/❌ | Repo-local smoke runs can consume IREE VMFBs whose executable payload resolves to the supported HSACO path. | `lrrt_iree_*_run_module_vmfb_smoke` |
| Function lookup by name | ✅ | Lazily resolves exported symbols with `lr_kernel_get`. | `lrrt_iree_hal_driver_executable_tests` |
| Function info | ✅/❌ | Reports the minimal function metadata needed by the current path. | `lrrt_iree_hal_driver_executable_tests` |
| Queue dispatch | ✅/❌ | Dispatches static workgroup counts with explicit workgroup size and pointer-only storage-buffer bindings. | `lrrt_iree_hal_driver_executable_tests`, VMFB smoke tests |
| Command-buffer dispatch | ✅/❌ | Records dispatch commands and replays them through the same lrrt launch path. | `lrrt_iree_hal_driver_executable_tests` |
| Dispatch constants | ❌ | Non-empty inline constants are rejected. | Implementation contract |
| Indirect workgroup counts | ❌ | Dispatch requires static workgroup counts. | Implementation contract |
| General IREE ABI packing | ❌ | Only pointer-only storage-buffer kernargs are packed. | Implementation contract |
| Dynamic shape dispatch policy | ❌ | The adapter does not infer dynamic shapes or dispatch sizes. | Non-goal for current adapter |

## Command Buffers And Queue Execution

| Area | Current status | Current behavior | Coverage |
| --- | --- | --- | --- |
| Command buffer creation | ✅ | Creates one-shot command buffers with IREE validation state. | `lrrt_iree_hal_driver_transfer_tests`, `lrrt_iree_hal_driver_executable_tests` |
| Command buffer update | ✅ | Records host-to-device update commands. | `lrrt_iree_hal_driver_transfer_tests` |
| Command buffer fill | ✅ | Records fill commands for transfer replay. | `lrrt_iree_hal_driver_transfer_tests` |
| Command buffer copy | ✅ | Records copy commands, including indirect binding slots. | `lrrt_iree_hal_driver_transfer_tests` |
| Command buffer dispatch | ✅/❌ | Records static dispatch commands with supported binding forms. | `lrrt_iree_hal_driver_executable_tests` |
| Queue execute replay | ✅/❌ | Replays update/fill/copy/dispatch records synchronously and in order. | `lrrt_iree_hal_driver_transfer_tests`, `lrrt_iree_hal_driver_executable_tests` |
| Indirect binding table resolution | ✅/❌ | Resolves indirect buffer slots supplied through the submission binding table. | `lrrt_iree_hal_driver_transfer_tests`, `lrrt_iree_hal_driver_executable_tests` |
| Command-buffer events | ❌ | Signal/reset/wait event commands return `UNIMPLEMENTED`. | Implementation contract |
| Command-buffer collectives | ❌ | Collective commands return `UNIMPLEMENTED`. | Implementation contract |
| Command-buffer optimization | ❌ | Commands are recorded and replayed directly; no fusion or optimization exists. | Documented adapter limitation |

## Synchronization

| Area | Current status | Current behavior | Coverage |
| --- | --- | --- | --- |
| HAL semaphore creation/query/signal/wait | ✅ | Provides lrrt-owned host-side timeline semaphores. | `lrrt_iree_hal_driver_registration_tests` |
| Queue wait/signal lists | ✅/❌ | Queue operations wait on and signal lrrt-owned semaphores around synchronous work. | `lrrt_iree_hal_driver_registration_tests`, transfer/dispatch tests |
| Device-native semaphore implementation | ❌ | Semaphores are host-side ordering objects, not GPU-native synchronization primitives. | Documented limitation |
| External semaphore timepoints | ❌ | Import/export of external timepoints returns `UNIMPLEMENTED`. | Implementation contract |
| HAL events | ❌ | Event creation and command-buffer event operations are unsupported. | `lrrt_iree_hal_driver_unsupported_tests` |

## Tooling And VMFB Smoke Coverage

| Area | Current status | Current behavior | Coverage |
| --- | --- | --- | --- |
| Repo-local run-module smoke runner | ✅ | `lrrt_iree_run_module_smoke` registers the lrrt driver then uses IREE run-module tooling. | `lrrt_iree_*_run_module_vmfb_smoke` |
| Single dispatch VMFB | ✅ | Runs the minimal multiply VMFB through `--device=lrrt`. | `lrrt_iree_run_module_vmfb_smoke` |
| Ordered multi-dispatch VMFB | ✅ | Runs a two-dispatch graph and validates command ordering. | `lrrt_iree_two_dispatch_run_module_vmfb_smoke` |
| Mixed matmul VMFB | ✅ | Runs a VMFB with more than one generated dispatch shape. | `lrrt_iree_mixed_matmuls_run_module_vmfb_smoke` |
| Multi-export VMFB | ✅ | Runs separate VMFB exports through the same adapter path. | `lrrt_iree_multi_export_*_run_module_vmfb_smoke` |
| Upstream-style external `iree-run-module --device=lrrt` | ❌ | The seamless path still depends on repo-local registration/linkage. | Remaining integration task |
| Dynamic plugin packaging | ❌ | No packaged external driver plugin exists yet. | Remaining integration task |

## Explicit Non-Coverage

| Area | Current status | Reason |
| --- | --- | --- |
| IREE compiler implementation | ❌ | The compiler remains outside lrrt. |
| IREE VM/Flow/Stream runtime implementation | ❌ | IREE owns high-level runtime semantics. |
| PyTorch/XLA/frontend import | ❌ | Frontend integration is outside the runtime adapter. |
| Full HIP HAL parity | ❌ | The adapter only implements the subset needed for the lrrt dispatch/resource-management path. |
| Performance-oriented queue scheduling | ❌ | Current HAL queue operations are synchronous correctness bridges; optimization comes later. |

## Current Reading

At this point the adapter is strong enough to validate that IREE-generated
ROCm dispatches can flow through an lrrt-backed HAL device for small static
VMFBs. The remaining "seamless" gap is not the core dispatch path; it is
packaging/registration outside the repo-local smoke runner plus broader HAL
surface coverage for programs that require more than static pointer-only
dispatches and synchronous transfer semantics.
