# zig-gc

A precise, tri-color **mark-sweep garbage collector** in pure Zig, generic over
an embedder *binding*. The default is non-moving; an opt-in stop-the-world
relocation pass is available to bindings with complete rewrite coverage. No
dependencies.

Built to unblock Phase 7 (GIL removal / true parallelism) of the
[zig-js](../zig-js) JavaScript engine, but the core is runtime-agnostic — it
follows the [MMTk](https://www.mmtk.io/) model of a reusable collector mechanism
driven by a small, language-specific binding. Full design rationale:
`zig-js/docs/threads/P7-gc-design.md`.

## Why precise + a non-moving default

- **Precise** (the collector traces only real references) — a conservative GC
  would falsely retain the `f64`s and NaN-boxed words that fill a JS heap, and
  precise reachability is *required* to clear weak references and fire
  finalizers correctly.
- **Non-moving by default** — bindings pay no relocation metadata, pointer
  rewriting, or read-barrier cost unless they explicitly implement the complete
  compaction contract. This remains the concurrency-friendly baseline.
- **Optional failure-atomic compaction** — a full stop-the-world collection can
  reserve every destination first, rewrite a complete live graph, and then
  commit address changes without finalizing moved cells.
- **Mark-sweep, tri-color** — incrementalizable and concurrentizable behind a
  write barrier, which is the path to lock-free parallelism.

## The binding contract

The collector owns allocation, the mark stack, sweep, weak-edge processing, and
finalizers. The embedder supplies, at comptime, a type `B` with:

```zig
const Kind = enum { ... };                          // your cell taxonomy
fn traceRoots(ctx: *B, v: anytype) void;            // mark every root
fn trace(cell: *anyopaque, kind: Kind, v: anytype) void; // mark a cell's edges
fn finalize(ctx: *B, cell: *anyopaque, kind: Kind) void; // a cell is dying
```

Inside `traceRoots`/`trace`, call `v.mark(ptr)` for each strong reference and
`v.markWeak(&slot)` (where `slot: *?*anyopaque`) for each weak slot. A weak slot
whose target dies is set to `null` *before* the target's storage is freed, so it
never dangles.

Embedders that need to root native stack or register-spill words can opt into
conservative marking from `traceRoots` with `v.markConservativeWord(word)` or
`v.markConservativeWords(start, count)`. Exact payload pointers and interior
payload pointers keep the owning cell alive; unrelated words are ignored. This
does not replace precise heap tracing — it is a stack-root escape hatch for
runtimes whose frame layout is not fully described yet.

An embedder with exact slab metadata may also replace the collector's per-cell
live-payload hash for slab-owned allocations with two optional hooks:

```zig
fn usesOwnedCellStorage(ctx: *B, total: usize) bool;
fn ownsCellAllocation(ctx: *B, allocation: *anyopaque) bool;
```

`usesOwnedCellStorage` must return `true` only when every successful aligned
allocation of `total` bytes comes from storage covered by the ownership hook.
`ownsCellAllocation` receives a candidate header address and must validate it
without dereferencing unowned memory; it must accept only exact issued cell-slot
starts. The collector checks its live header magic after ownership succeeds,
clears that magic before freeing a cell, and retains the hash/all-cells fallback
for other allocations and bindings without these hooks.

Embedders that can prove no weak semantic state has been published may avoid
empty ephemeron, weak-slot, and after-weak passes with:

```zig
fn hasWeakWork(ctx: *B) bool;
```

Returning `false` is a correctness promise that the heap has no weak slots,
ephemeron edges, or after-weak cleanup records. A monotonic false-to-true flag
is sufficient and may conservatively remain true for the rest of the heap
lifetime. Bindings without this hook always run the weak passes.

An embedder whose backing allocator can reserve several equal-size slabs under
one size-class lock may also provide:

```zig
fn allocateCellBatch(ctx: *B, total: usize, out: []*anyopaque) usize;
```

The hook writes 16-byte-aligned allocation-base pointers into a prefix of
`out` and returns its length. The slabs must have exactly the same ownership
and free semantics as individual `backing.alignedAlloc` results. A zero prefix
asks the collector to use its normal allocation and recovery path; a short,
non-zero prefix is published immediately so the caller can initialize and
commit it before a later request performs recovery. The slabs stay private
until `zig-gc` initializes their headers and publishes the whole prefix under
one metadata lock.

Bindings that additionally prove every cell uses that owned storage and can
publish exact slot ownership may provide:

```zig
fn allCellsUseOwnedStorage(ctx: *B) bool;
fn publishCellAllocationBatch(
    ctx: *B,
    payloads: []*anyopaque,
    total: usize,
    payload_offset: usize,
) void;
fn ownedCellIterator(ctx: *B) Iterator; // Iterator.next() ?*anyopaque
```

For batches of at least 64 cells while marking is inactive, `zig-gc` initializes
and chains the private headers before taking the allocation-metadata lock,
splices the chain and updates its counters in O(1), then releases that lock
before the binding publishes its ownership bitmap. Payloads remain private
until `createBatch` returns. Short batches, a changed nursery mode, and active
marking retain the compact per-cell publication path, including born-grey
semantics. `ownedCellIterator` is consumed only at a quiescent collection or
teardown boundary (or while publication is excluded); it must yield every
published allocation base exactly once and exclude private or reclaimed slots.
When present, it also replaces the collector's intrusive all-cells list.
Large owned batches then publish aggregate live/nursery deltas through stable
per-thread cache-line shards: publishers only read the heap gate and update
their own shard, while a collector closes the gate, drains active publishers,
and folds all deltas once before marking. Active marking retains the serialized
born-grey path.

### Optional relocation

A binding opts into `collectAndCompact()` only by defining all three graph
hooks:

```zig
fn canRelocate(ctx: *B, cell: *anyopaque, kind: Kind) bool;
fn relocateRoots(ctx: *B, v: anytype) void;
fn relocateCell(ctx: *B, cell: *anyopaque, kind: Kind, v: anytype) void;
```

`canRelocate` may pin any cell. Both rewrite hooks must replace every pointer
with `v.resolve(old)`; the resolver returns the destination for a moved cell and
the original address for a pinned cell. Pinned cells are still passed to
`relocateCell` because they may point at moved children. The collector exposes
`StableCellId`, `RelocationRecord`, and `RelocationVisitor` for plan auditing.

Bindings may also opt into a paired post-commit audit while forwarding records
are still live:

```zig
fn verifyRelocationRoots(ctx: *B, v: anytype) void;
fn verifyRelocationCell(ctx: *B, cell: *anyopaque, kind: Kind, v: anytype) void;
```

The collector calls these only after every destination and index/publication
update commits. They must be allocation-free and infallible; an embedder may
use `v.moved(pointer)` to trap if any audited root or live edge still contains
an old payload address.

The default destination path uses the heap backing allocator. Owned slab
bindings may instead make unpublished reservations and atomically update their
publication metadata with:

```zig
fn reserveRelocationCell(ctx: *B, total: usize) ?*anyopaque;
fn releaseRelocationReservation(ctx: *B, allocation: *anyopaque, total: usize) void;
fn commitRelocationCell(ctx: *B, old: *anyopaque, new: *anyopaque, total: usize) void;
```

All scratch capacity and destinations are reserved before mutation. Any failure
rolls back every reservation and returns `.out_of_memory` with the old graph,
indexes, publication state, and accounting intact. Once copying begins, graph
rewrite and commit hooks must be allocation-free and infallible. Relocation is
stop-the-world only; concurrent/native safepoints remain an embedder concern.

## Usage

```zig
const gc = @import("gc");

var rt = MyRuntime{};                 // holds your roots + finalizer state
var heap = gc.Heap(MyRuntime).init(allocator, &rt);
defer heap.deinit();                  // frees + finalizes everything left
heap.setNurseryTenuringAge(3);        // retain survivors for three minor cycles
heap.setNurseryEnabled(true);

const obj = try heap.create(MyObject, .object); // uninitialized payload
obj.* = .{ ... };                      // initialize before the next safepoint

var objects: [32]*MyObject = undefined;
const count = try heap.createBatch(MyObject, .object, &objects); // one metadata lock
for (objects[0..count]) |item| item.* = .{ ... }; // initialize before a safepoint

heap.maybeCollect();                   // call at safepoints; collects past a threshold
heap.collect();                        // or force a full stop-the-world cycle
const compacted = heap.collectAndCompact(); // opt-in binding: collect + relocate
const stats = heap.accounting();       // race-safe generation + heap telemetry
```

`create` and `createBatch` return uninitialized payloads — initialize them
before the next collection so a cycle never traces a half-built cell. Batched
creation allocates private slabs first, then publishes their headers, all-cells
links, nursery accounting, and born-grey state under one metadata lock; large
all-owned batches use the O(1) splice described above. Cells are 16-byte aligned
behind a 32-byte header on 64-bit targets. That header includes a process-unique
relocation-stable ID, checked 32-bit payload size, kind, generation age, and one
atomic byte for mark/young/remembered flags; recovering it from a payload is
O(1). A short
batch reports its successfully published prefix so the caller can commit that
work before the next allocation performs recovery or reports OOM, preserving
sequential failure ordering.

`accounting()` snapshots live/young/promoted totals, collection counts, the
configured tenuring age, the latest minor survivor/reclamation/promotion bytes,
cumulative young-input/survivor/reclamation/promotion bytes across all minor
cycles, and the post-sweep byte size of the last full collection. The cumulative
counters are historical and do not reset on a full collection or nursery toggle.
Old-container and conservative-target cards persist while survivors remain
young, so an unchanged old-to-young edge stays sound across repeated minors.
Full collection or nursery disable explicitly tenures the remaining young
prefix.

## Status

The collector supports stop-the-world, incremental, concurrent, parallel, and
configurable multi-age nursery paths, plus opt-in stop-the-world relocation. The
default remains non-moving with one-cycle tenuring until an embedder selects a
higher age. Unit and TSan gates cover cycles, weak/finalization semantics,
allocation/publication races, relocation rollback, pinned/moved graphs, and
exact accounting.

## License

MIT
