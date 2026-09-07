# light-rocr

`light-rocr` is the self-authored low-level AMDGPU runtime stack used by
`light-rocm-runtime`. It will contain the AMDHSA code-object loader, runtime
primitives, and interchangeable libhsakmt and direct-KFD transports.

The loader is a host-only ELF, AMDHSA metadata, and load-plan parser. It has no
dependency on ROCr, libhsakmt, KFD, a GPU, libelf, or LLVM libraries. The
runtime now also has a small transport-independent topology model and an
optional libhsakmt transport that discovers KFD nodes, allocates mapped GTT
and VRAM memory, creates compute AQL queues, and manages their lifetimes
without loading `libhsa-runtime64.so`. It also provides the hardware-visible
AMD user-signal ABI, CPU atomic operations, bounded active waits, and mapped
GTT-backed signal ownership.
The queue path now owns a fixed `gfx1101` scratch backing allocation and
populates the firmware-visible AMD queue control block for kernels with a
known private-segment requirement.
The first fixed `gfx1101` AQL dispatch now runs without ROCr: light-rocr writes
the packet, advances its queue index, rings the KMT-mapped doorbell, observes a
GPU completion-signal decrement, and verifies a GPU-written canary value.
The normal Loader `LoadPlan` can now be materialized into an executable GTT
image as well: file-backed ranges are copied, BSS and allocation padding are
zeroed, and kernel descriptor virtual addresses are resolved to GPU addresses.
The common runtime can also materialize caller-provided kernarg bytes into a
metadata-sized logical buffer and build an AQL dispatch packet directly from a
loaded kernel's resolved descriptor and supported segment sizes.

The initial parser accepts the AMDGPU-HSA ABI versions observed in the current
artifact corpus: version 4 from Clang/Triton and version 3 from IREE.

## Standalone build

```sh
cmake -S light-rocr -B build-light-rocr
cmake --build build-light-rocr
ctest --test-dir build-light-rocr --output-on-failure
```

Inspect an AMDHSA code object:

```sh
./build-light-rocr/light-rocr-inspect-hsaco \
  ./build/vector_add_kernel.hsaco
```

Inspect the live KFD topology and select the unique `gfx1101` node:

```sh
./build-light-rocr/light-rocr-inspect-topology
```

Exercise live GTT and topology-selected VRAM allocations and GPU mappings:

```sh
./build-light-rocr/light-rocr-check-memory
```

Create a live compute AQL queue with enough fixed scratch backing for the
repository's 272-byte-private-segment Clang vector-add kernel, then destroy it
without publishing a packet or writing its doorbell:

```sh
./build-light-rocr/light-rocr-check-queue
```

Create a GPU-visible native user signal, exercise its CPU atomics and bounded
wait, and release its mapping without publishing a packet or writing a
doorbell:

```sh
./build-light-rocr/light-rocr-check-signal
```

Publish one fixed, scratch-free `gfx1101` kernel-dispatch packet and verify its
GPU store and native completion signal with bounded waits:

```sh
./build-light-rocr/light-rocr-check-dispatch
```

Materialize a normal HSACO in executable GPU-visible memory and verify its
copies, zero fills, address translation, and cleanup without creating a queue
or ringing a doorbell:

```sh
./build-light-rocr/light-rocr-check-image \
  ./build/vector_add_kernel.hsaco
```

The topology tool is built when the public `hsakmt` headers, library, and its
DRM/NUMA dependencies are available. `LIGHT_ROCR_ENABLE_HSAKMT=OFF` keeps the
standalone loader and host-only topology tests buildable without them. Live
topology discovery opens `/dev/kfd`, takes and releases a KMT system-property
snapshot, records CPU/GPU nodes and memory-bank properties, and closes KFD on
every successful or failed path. GPU selection accepts either `gfx1101` or a
full AMDHSA target string and deliberately rejects zero or multiple matching
nodes in the initial single-GPU runtime. Host tests replace the seven KMT entry
points used by discovery, so conversion, call ordering, and every cleanup path
remain testable without a GPU.

When installed HSA headers are available, the test build adds optional
compile-time ABI oracles comparing the self-authored signal, 64-byte AQL
kernel-dispatch packet, and 256-byte AMD queue-control definitions with the
public definitions used by ROCr.
The runtime and libhsakmt tests do not require those ROCr headers, and the
oracles do not link `libhsa-runtime64.so`.

The memory transport opens KFD and retains a system-property snapshot for the
whole session. `allocate_gtt` creates 4 KiB-aligned, non-pageable system memory
and maps it only to the selected GPU node. `allocate_vram` requires a topology
frame-buffer heap, prevents substitution with system memory, and follows the
heap's CPU-visibility contract: public frame buffers expose a host address,
while private frame buffers expose only their GPU address. Both return a
move-only `MemoryAllocation`. An allocation shares ownership of the session,
so KFD cannot close while mapped memory remains alive. Explicit or RAII cleanup
orders unmap before free; explicit cleanup failures retain enough state to
retry. The live diagnostic writes and reads GTT (and public VRAM when present),
then checks VRAM mapping and cleanup without dereferencing private VRAM.
`allocate_executable_gtt` uses the same ownership rules while requesting KFD's
execute page attribute for the initial fixed kernel image.
`allocate_scratch` follows ROCr's distinct KMT path: it reserves the selected
GPU's hidden-private backing VA with the scratch flag, maps physical backing
through `hsaKmtMapMemoryToGPU`, never exposes the reserved VA as a CPU pointer,
and ignores the alternate mapping address on the current gfx9+ contract. A
process-wide, per-GPU lifetime guard serializes libhsakmt's single scratch
reservation contract across every `KfdSession`; a concurrent second request is
rejected as `ScratchAlreadyReserved` before entering KMT. The guard remains held
after a cleanup failure and is released only when the backing allocation is
successfully freed.

The transport-independent executable-image runtime consumes the Loader's
already-checked `CodeObject` and original HSACO bytes. Before a transport
allocates memory it revalidates the complete `LoadPlan`, rejects relocations in
the initial materialization subset, checks ordered and non-overlapping copy,
zero-fill, and protection ranges, and validates file-backed kernel descriptors
and executable code entries. It also owns checked image realization and
virtual-to-GPU address translation, so libhsakmt and a later direct-KFD
transport share the same rules and host-only tests.

The common runtime rounds the validated image span to 4 KiB. The libhsakmt
wrapper requests an allocation of that size and delegates writes and
GPU-address resolution back to the common runtime. Its move-only
`ExecutableImage` retains an independent copy of the
parsed code object and the resolved descriptor/code-entry GPU addresses. A
truthy image is GPU-usable; cleanup ownership is reported separately because a
successful unmap followed by a failed free must invalidate every published GPU
address while retaining enough allocation state to retry the free. Explicit
cleanup remains retryable through the underlying allocation. Segment-specific
page protections and relocation application are deliberately deferred; the
current artifact corpus uses at most 4 KiB load alignment and contains no
dynamic relocations.

The transport-independent launch layer validates the metadata-declared kernarg
size and normalized alignment before writing memory. It accepts the same
prefix contract as LRRT: caller bytes may be shorter than the complete kernarg
segment, while only the remaining metadata-declared hidden-argument area is
zeroed. Bytes beyond that logical segment remain untouched, allowing a later
transport to suballocate multiple kernargs from larger backing storage.
Zero-sized kernarg segments use a canonical allocation-free representation.
The libhsakmt wrapper currently rounds each non-empty buffer to a dedicated
ordinary mapped GTT page and rejects alignments above that page alignment. It
uses the same move-only ownership, retryable cleanup, and stale-GPU-address
invalidation rules as executable images. If validation fails after mapping and
cleanup also fails, the failed result retains unpublished allocation ownership
so cleanup can be retried explicitly.

`make_kernel_launch_packet` combines caller geometry with the selected loaded
kernel's descriptor GPU address, fixed and dynamic group segment sizes, kernarg
address, and completion signal. It rejects a mismatched kernarg layout,
dynamic-stack kernels, and group-size arithmetic overflow before delegating
final packet ABI checks to the AQL layer. Fixed private-segment sizes now flow
into the packet; queue submission rejects them before publication unless the
queue owns at least that much per-lane scratch capacity. This unit constructs
but does not submit the packet itself.

The queue transport allocates a 64 KiB default ring as executable AQL queue
memory and initializes every 64-byte packet header to the invalid type. A
separate mapped GTT page begins with light-rocr's self-authored, version-one
AMD queue control block. Besides the read and write dispatch IDs, that block
records the public HSA queue prefix, LDS/private aperture bases, CU/wave limits,
scratch resource descriptor, `COMPUTE_TMPRING_SIZE`, backing VA/size, and the
64-bit pointer property. Its layout is checked against the installed ROCr
header in an optional oracle test, while production code has no ROCr-header
dependency. Queue capabilities remain zero, explicitly excluding ROCr's
version-two dynamic async-reclaim protocol.

For a non-zero configured private segment, the gfx1101 platform module sizes a
fixed backing for all reported physical scratch slots, encodes the 256-byte
SRD granularity and per-shader-engine wave count, and checks aperture, address,
integer, and register-width bounds before queue creation. `hsaKmtCreateQueue`
registers the control block's ABI-defined read/write field addresses and
returns the mapped 64-bit doorbell. The public HSA queue ID comes from a
process-wide monotonic counter and remains distinct from the reusable KMT queue
handle. The move-only `AqlQueue` destroys the hardware queue before releasing
scratch, control, and ring storage in that order; explicit failures remain
retryable. If scratch mapping and its immediate cleanup both fail during queue
creation, the failed result carries an inactive `AqlQueue` cleanup owner whose
`release()` can be retried. In this libhsakmt phase, libhsakmt itself owns the
EOP buffer, CWSR storage, and doorbell mapping. Those resources become
light-rocr's explicit responsibility only in the later direct-KFD transport.
The initial fixed policy supports one concurrent scratch-backed queue per GPU
process; a second reservation is rejected before KMT. A shared pool and
multiple scratch-backed queues are deferred with the broader multi-queue
runtime work.

The transport-independent AQL definition validates the initial narrow policy:
one-, two-, or three-dimensional dispatch geometry, 64-byte-aligned AMD kernel
descriptors and completion signals, at least 16-byte-aligned kernarg, zeroed
reserved fields, the gfx1101 limits of 1,024 total work-items per workgroup and
64 KiB of group-segment LDS, and system-scope acquire/release fences. The
current queue is single-producer.
Submission rejects an invalid packet, a private segment above the queue's
fixed scratch capacity, or a full queue before mutation. It writes the payload
into an invalid ring slot, publishes its 16-bit header with release ordering,
advances the 64-bit write index, executes the host-architecture MMIO store
fence, and finally writes the packet ID to the mapped 64-bit doorbell.

The native user signal is the AMD hardware ABI object itself: a 64-byte object
on a 64-byte boundary, with the user kind at offset 0 and its 64-bit atomic
value at offset 8. Its AQL handle is the GPU virtual address of the complete
object rather than the address of the value field. The remaining mailbox,
timestamp, queue, and reserved fields start at zero. `light_rocr_core` provides
lock-free relaxed/release stores, relaxed/acquire loads, and a deadline-bounded
active wait. The KMT transport places one signal in a mapped GTT page and keeps
the KFD session alive for the mapping's lifetime. Interrupt-backed waits and
signal pooling are intentionally deferred.

The fixed dispatch diagnostic deliberately bypasses normal ELF loading. It
places a self-contained AMDHSA descriptor and the audited 48-byte instruction
sequence from `tools/fixed_gfx1101_store.s` in executable GTT. A 16-byte
kernarg supplies a GTT output address and canary. One work-item stores that
canary, while packet completion decrements the native signal from one to zero.
Both observations, queue-index progress, a two-second signal deadline, and
ordered queue-before-memory cleanup are checked. This isolates AQL
queue/packet/doorbell/signal behavior; ordinary HSACO segment materialization
now uses a separate diagnostic and relocation remains deferred. When AMDGPU
Clang and
`llvm-objcopy` are available, an additional host test reassembles and links the
checked-in source at the fixed layout, then compares both its descriptor and
instruction bytes with the embedded image.

The inspector reports checked `PT_LOAD` records and, when an AMDGPU metadata
note is present, the metadata version, target ISA, kernel inventory, ELF
`.kd` symbol addresses, code entry offsets, segment sizes, wavefront size, and
raw kernel-descriptor resource flags. It reports both the metadata kernarg
alignment and the public alignment, whose minimum is normalized to 16 bytes to
match the observed ROCr executable property. It also reports the virtual image
span and alignment, file-copy, zero-fill, protection, and relocation operation
counts that a later GPU allocator will consume.

The initial executable subset is intentionally narrow: metadata version 1.2,
the two observed canonical `gfx1101` target strings, string-keyed MessagePack
maps, ELF64 dynamic symbols discoverable through `PT_DYNAMIC` and `DT_HASH`,
and 64-byte AMDHSA kernel descriptors. Section headers are not required.
Unknown MessagePack types, malformed/duplicate notes or dynamic tables,
inconsistent metadata and descriptors, reserved descriptor bits, and
out-of-range symbols or code entries are rejected. ELF objects without an
AMDGPU metadata note remain parseable and produce an empty kernel inventory; a
later executable operation decides whether kernels are required.

`LoadPlan` construction is likewise section-header-independent. It emits
deterministically ordered copy, zero-fill, and final-protection ranges from
checked `PT_LOAD` records. Dynamic ELF64 `REL` and `RELA` tables are bounded by
their `PT_DYNAMIC` tags and `DT_HASH` symbol count; known AMDGPU relocation
records are represented but are not applied yet. Unknown AMDGPU types, invalid
targets or symbol indices, partial tables, and unobserved PLT/RELR encodings are
rejected explicitly. A 2026-09-06 scan of all 208 HSACOs then present under
`build`, `build-triton`, and `build-iree-probe` found no relocation records, so
the first GPU-loading path can be gated on an empty relocation list while the
parser preserves that boundary rather than assuming it.

The main project can include this subproject with
`-DLRRT_ENABLE_LIGHT_ROCR=ON`.

## Component boundary

```text
loader/       bounded AMDHSA ELF parsing and LoadPlan construction
runtime/      device, memory, queue, signal, executable, and dispatch logic
transport/    libhsakmt and direct public-KFD implementations
arch/         GPU-generation-specific layouts and operations
```

`runtime/` owns transport-independent topology, selection, executable-image
validation, realization, address translation, kernarg materialization, and
launch-packet construction semantics.
`transport/hsakmt/` is the first concrete KFD transport; additional component
directories are added when they receive their first implementation.
