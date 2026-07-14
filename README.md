# zig-gc

A precise, non-moving, tri-color **mark-sweep garbage collector** in pure Zig,
generic over an embedder *binding*. No dependencies.

Built to unblock Phase 7 (GIL removal / true parallelism) of the
[zig-js](../zig-js) JavaScript engine, but the core is runtime-agnostic — it
follows the [MMTk](https://www.mmtk.io/) model of a reusable collector mechanism
driven by a small, language-specific binding. Full design rationale:
`zig-js/docs/threads/P7-gc-design.md`.

## Why precise + non-moving

- **Precise** (the collector traces only real references) — a conservative GC
  would falsely retain the `f64`s and NaN-boxed words that fill a JS heap, and
  precise reachability is *required* to clear weak references and fire
  finalizers correctly.
- **Non-moving** — no compaction, no pointer rewriting, no read barriers. Simple
  to get correct, and it matches the reference design for concurrent JS GC
  (WebKit's Riptide is non-moving).
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

## Usage

```zig
const gc = @import("gc");

var rt = MyRuntime{};                 // holds your roots + finalizer state
var heap = gc.Heap(MyRuntime).init(allocator, &rt);
defer heap.deinit();                  // frees + finalizes everything left

const obj = try heap.create(MyObject, .object); // uninitialized payload
obj.* = .{ ... };                      // initialize before the next safepoint

var objects: [32]*MyObject = undefined;
try heap.createBatch(MyObject, .object, &objects); // one metadata publication lock
for (objects) |item| item.* = .{ ... }; // initialize every item before a safepoint

heap.maybeCollect();                   // call at safepoints; collects past a threshold
heap.collect();                        // or force a full stop-the-world cycle
```

`create` and `createBatch` return uninitialized payloads — initialize them
before the next collection so a cycle never traces a half-built cell. Batched
creation allocates private slabs first, then publishes their headers, all-cells
links, nursery accounting, and born-grey state under one metadata lock. Cells
are 16-byte aligned with a single-word header; recovering a header from a
payload is O(1).

## Status

**M1 scaffold**: a working stop-the-world collector (no write barrier needed
while the world is stopped) with cycles, garbage reclamation, weak edges, and
finalizers — see `src/heap.zig` tests (`zig build test`). The same core
incrementalizes (M2: insertion write barrier, lazy sweep) and concurrentizes
(M3: per-object locks, drop the GIL) per the staged plan in the zig-js design
note.

## License

MIT
