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

Create and immediately destroy a live compute AQL queue without publishing a
packet or writing its doorbell:

```sh
./build-light-rocr/light-rocr-check-queue
```

Create a GPU-visible native user signal, exercise its CPU atomics and bounded
wait, and release its mapping without publishing a packet or writing a
doorbell:

```sh
./build-light-rocr/light-rocr-check-signal
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

When an installed `hsa/amd_hsa_signal.h` is available, the test build adds an
optional compile-time ABI oracle comparing the self-authored signal definition
with ROCr's definition. Neither the runtime nor the libhsakmt signal tests
require that ROCr header, and the oracle does not link `libhsa-runtime64.so`.

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

The queue transport allocates a 64 KiB default ring as executable AQL queue
memory and initializes every 64-byte packet header to the invalid type. A
separate mapped GTT page holds cache-line-separated read and write indexes.
`hsaKmtCreateQueue` registers those GPU addresses, creates the compute AQL
queue, and returns its mapped 64-bit doorbell. The move-only `AqlQueue` owns
both allocations and always destroys the queue before unmapping them; an
explicit destroy failure leaves the queue and mappings live so cleanup can be
retried. In this libhsakmt phase, libhsakmt itself owns the EOP buffer, CWSR
storage, and doorbell mapping. Those resources become light-rocr's explicit
responsibility only in the later direct-KFD transport.

The native user signal is the AMD hardware ABI object itself: a 64-byte object
on a 64-byte boundary, with the user kind at offset 0 and its 64-bit atomic
value at offset 8. Its AQL handle is the GPU virtual address of the complete
object rather than the address of the value field. The remaining mailbox,
timestamp, queue, and reserved fields start at zero. `light_rocr_core` provides
lock-free relaxed/release stores, relaxed/acquire loads, and a deadline-bounded
active wait. The KMT transport places one signal in a mapped GTT page and keeps
the KFD session alive for the mapping's lifetime. Interrupt-backed waits and
signal pooling are intentionally deferred.

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

`runtime/` owns transport-independent topology and selection semantics.
`transport/hsakmt/` is the first concrete KFD transport; additional component
directories are added when they receive their first implementation.
