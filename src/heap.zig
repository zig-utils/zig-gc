//! A precise, non-moving, tri-color mark-sweep collector, generic over an
//! embedder *binding*. The collector owns the mechanism (allocation, the mark
//! stack, sweep, weak-edge processing, finalizers); the embedder supplies the
//! policy at comptime: how to enumerate roots, how to trace one cell, and what
//! to do when a cell dies.
//!
//! Design: docs/threads/P7-gc-design.md in zig-utils/zig-js. M1 is
//! stop-the-world (no write barrier needed); the same core incrementalizes and
//! concurrentizes for Phase-7 GIL removal.
//!
//! The binding is any type `B` exposing:
//!   const Kind = enum {...};                       // the cell taxonomy
//!   fn traceRoots(ctx: *B, v: anytype) void;       // mark every root
//!   fn trace(cell: *anyopaque, kind: Kind, v: anytype) void; // mark a cell's edges
//!   fn finalize(ctx: *B, cell: *anyopaque, kind: Kind) void; // a cell is dying
//!
//! Optional hooks:
//!   fn hasWeakWork(ctx: *B) bool;
//!   fn freeCellStorageBatch(ctx: *B, total: usize, allocations: []*anyopaque) void;
//!   fn traceEphemeron(ctx: *B, cell: *anyopaque, kind: Kind, v: anytype) void;
//!   fn afterWeak(ctx: *B, cell: *anyopaque, kind: Kind) void;
//!   fn traceOldOnMinor(kind: Kind) bool;
//!   fn classifyConservativeInterior(ctx: *B, address: usize) InteriorOwnership;
//!   fn allCellsUseOwnedStorage(ctx: *B) bool;
//!   fn publishCellAllocation(ctx: *B, allocation: *anyopaque, total: usize) void;
//!   fn publishCellAllocationBatch(ctx: *B, payloads: []*anyopaque, total: usize, payload_offset: usize) void;
//!   fn unpublishCellAllocation(ctx: *B, allocation: *anyopaque, total: usize) void;
//!   fn unpublishCellAllocationBatch(ctx: *B, total: usize, allocations: []*anyopaque) void;
//!   fn ownedCellIterator(ctx: *B) Iterator; // Iterator.next() ?*anyopaque
//!   fn canRelocate(ctx: *B, cell: *anyopaque, kind: Kind) bool;
//!   fn relocateRoots(ctx: *B, v: anytype) void;
//!   fn relocateCell(ctx: *B, cell: *anyopaque, kind: Kind, v: anytype) void;
//!   fn verifyRelocationRoots(ctx: *B, v: anytype) void;
//!   fn verifyRelocationCell(ctx: *B, cell: *anyopaque, kind: Kind, v: anytype) void;
//!   fn reserveRelocationCell(ctx: *B, total: usize) ?*anyopaque;
//!   fn releaseRelocationReservation(ctx: *B, allocation: *anyopaque, total: usize) void;
//!   fn commitRelocationCell(ctx: *B, old: *anyopaque, new: *anyopaque, total: usize) void;
//!   fn collectionPhaseBoundary(ctx: *B, boundary: CollectionPhaseBoundary) void;
//!
//! Inside trace/traceRoots the binding calls `v.mark(ptr)` for each strong
//! reference and `v.markWeak(&slot)` for each weak slot (`*?*anyopaque`).
//! `traceEphemeron` may call `v.isMarked(key)` and then `v.mark(value)` for
//! WeakMap-style key/value edges.

const std = @import("std");
const builtin = @import("builtin");

/// Semantic collection boundaries for opt-in embedder profiling. The generic
/// heap deliberately owns no clock or counters; bindings that omit the hook pay
/// no runtime cost. `prepare_begin` is emitted only after a collection is known
/// to run, and `post_sweep_end` follows the optional `afterSweep` hook.
pub const CollectionPhaseBoundary = enum {
    full_prepare_begin,
    full_trace_begin,
    full_sweep_begin,
    full_sweep_end,
    full_post_sweep_end,
    minor_prepare_begin,
    minor_trace_begin,
    minor_sweep_begin,
    minor_sweep_end,
    minor_post_sweep_end,
};

/// Optional binding result for conservative interior-address classification.
/// `allocation` is the exact cell allocation base (the header address), while
/// `owned_empty` means the address lies in owned storage but cannot name an
/// issued allocation. `outside` permits the collector's generic fallback unless
/// the binding separately proves that every cell uses its owned storage.
pub const InteriorOwnership = union(enum) {
    outside,
    owned_empty,
    allocation: *anyopaque,
};

/// Process-unique, address-independent identity for one collector cell. IDs are
/// non-zero, never recycled, and remain attached to a cell when compaction
/// relocates its storage. Embedders may use them for diagnostics such as heap
/// snapshots, but must not expose them as mutable language state.
pub const StableCellId = enum(u64) {
    _,

    pub fn init(raw: u64) StableCellId {
        std.debug.assert(raw != 0);
        return @enumFromInt(raw);
    }
};

var next_stable_cell_id: std.atomic.Value(u64) = .init(1);

fn allocateStableCellId() StableCellId {
    var current = next_stable_cell_id.load(.monotonic);
    while (true) {
        if (current == 0 or current == std.math.maxInt(u64))
            @panic("zig-gc stable cell identity exhausted");
        if (next_stable_cell_id.cmpxchgWeak(current, current + 1, .monotonic, .monotonic)) |observed| {
            current = observed;
            continue;
        }
        return .init(current);
    }
}

pub const RelocationState = enum {
    reserved,
    copied,
    rewritten,
};

pub fn RelocationRecord(comptime Kind: type) type {
    return struct {
        id: StableCellId,
        kind: Kind,
        size: usize,
        old_payload: *anyopaque,
        new_payload: *anyopaque,
        state: RelocationState = .reserved,
    };
}

/// Infallible old→new resolver handed to binding rewrite hooks after every
/// destination has been reserved. Cells absent from the plan are pinned and
/// retain their address. Dead weak targets have already been cleared before
/// relocation begins, so every non-null weak target follows the same mapping.
pub fn RelocationVisitor(comptime Kind: type) type {
    return struct {
        const Record = RelocationRecord(Kind);
        records: []const Record,
        index: *const std.AutoHashMapUnmanaged(usize, usize),

        pub fn resolve(self: *const @This(), old_payload: *anyopaque) *anyopaque {
            const record_index = self.index.get(@intFromPtr(old_payload)) orelse return old_payload;
            const record = &self.records[record_index];
            std.debug.assert(record.old_payload == old_payload);
            return record.new_payload;
        }

        pub fn moved(self: *const @This(), old_payload: *anyopaque) bool {
            return self.index.contains(@intFromPtr(old_payload));
        }

        pub fn stableId(self: *const @This(), old_payload: *anyopaque) ?StableCellId {
            const record_index = self.index.get(@intFromPtr(old_payload)) orelse return null;
            return self.records[record_index].id;
        }
    };
}

const use_pthread_weak_lock = switch (builtin.os.tag) {
    .linux,
    .macos,
    .ios,
    .tvos,
    .watchos,
    .visionos,
    .freebsd,
    .netbsd,
    .openbsd,
    .dragonfly,
    => true,
    else => false,
};

const WeakLock = if (use_pthread_weak_lock) struct {
    mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,

    inline fn lock(self: *@This()) void {
        std.debug.assert(std.c.pthread_mutex_lock(&self.mutex) == .SUCCESS);
    }

    inline fn unlock(self: *@This()) void {
        std.debug.assert(std.c.pthread_mutex_unlock(&self.mutex) == .SUCCESS);
    }

    inline fn deinit(self: *@This()) void {
        std.debug.assert(std.c.pthread_mutex_destroy(&self.mutex) == .SUCCESS);
    }
} else struct {
    state: std.atomic.Value(u32) = .init(0),

    inline fn lock(self: *@This()) void {
        while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }

    inline fn unlock(self: *@This()) void {
        self.state.store(0, .release);
    }

    inline fn deinit(_: *@This()) void {}
};

// Weak-slot tracing is collector scratch state, not heap policy. Serialize it
// with one process-wide lock so independent heaps created/destroyed on parallel
// threads do not expose allocator address reuse to ThreadSanitizer as two
// different per-heap mutexes protecting the same scratch page.
var global_weak_lock: WeakLock = .{};

pub fn Heap(comptime Binding: type) type {
    return struct {
        const Self = @This();
        pub const Kind = Binding.Kind;
        pub const CellMetadata = struct {
            id: StableCellId,
            kind: Kind,
            size: usize,
        };
        pub const RelocationRecordType = RelocationRecord(Kind);
        pub const RelocationVisitorType = RelocationVisitor(Kind);
        const has_owned_cell_iterator = @hasDecl(Binding, "ownedCellIterator");
        const supports_relocation = @hasDecl(Binding, "canRelocate") and
            @hasDecl(Binding, "relocateRoots") and
            @hasDecl(Binding, "relocateCell");
        const relocation_verify_hooks = @as(u2, @intFromBool(@hasDecl(Binding, "verifyRelocationRoots"))) +
            @as(u2, @intFromBool(@hasDecl(Binding, "verifyRelocationCell")));
        const custom_relocation_storage_hooks = @as(u2, @intFromBool(@hasDecl(Binding, "reserveRelocationCell"))) +
            @as(u2, @intFromBool(@hasDecl(Binding, "releaseRelocationReservation"))) +
            @as(u2, @intFromBool(@hasDecl(Binding, "commitRelocationCell")));
        comptime {
            if (custom_relocation_storage_hooks != 0 and custom_relocation_storage_hooks != 3)
                @compileError("relocation storage hooks must be supplied as reserve/release/commit trio");
            if (relocation_verify_hooks != 0 and relocation_verify_hooks != 2)
                @compileError("relocation verification hooks must be supplied as roots/cell pair");
        }
        const OwnedCellIterator = if (has_owned_cell_iterator)
            @typeInfo(@TypeOf(Binding.ownedCellIterator)).@"fn".return_type.?
        else
            void;
        pub const min_nursery_threshold_bytes: usize = 4 * 1024 * 1024;
        pub const default_nursery_threshold_bytes: usize = 4 * 1024 * 1024;
        pub const default_tenuring_age: u8 = 1;
        const tenured_age: u8 = std.math.maxInt(u8);
        const min_retained_scratch_entries: usize = 4096;
        const publication_shard_count = if (has_owned_cell_iterator) 64 else 0;

        /// One fixed 32-byte header on 64-bit targets: stable identity, the
        /// all-cells link, checked payload size, kind/age, and four independently
        /// updated concurrent state bits packed into one atomic byte.
        const header_magic: u64 = 0x7a67_6763_5f68_6561;
        const header_marked: u8 = 1 << 0;
        const header_young: u8 = 1 << 1;
        const header_remembered_owner: u8 = 1 << 2;
        const header_remembered_target: u8 = 1 << 3;
        pub const Header = struct {
            magic: u64,
            stable_id: StableCellId,
            next: ?*Header,
            size: u32,
            kind: Kind,
            age: u8,
            flags: u8,
        };

        comptime {
            if (@sizeOf(usize) == 8 and @sizeOf(Header) != 32)
                @compileError("zig-gc Header must remain 32 bytes on 64-bit targets");
        }

        inline fn headerFlagLoad(h: *const Header, mask: u8, comptime order: std.builtin.AtomicOrder) bool {
            return @atomicLoad(u8, &h.flags, order) & mask != 0;
        }

        inline fn headerPayloadSize(h: *const Header) usize {
            return h.size;
        }

        fn headerFlagStore(h: *Header, mask: u8, value: bool, comptime order: std.builtin.AtomicOrder) void {
            var current = @atomicLoad(u8, &h.flags, .monotonic);
            while (true) {
                const updated = if (value) current | mask else current & ~mask;
                if (@cmpxchgWeak(u8, &h.flags, current, updated, order, .monotonic)) |observed| {
                    current = observed;
                    continue;
                }
                return;
            }
        }

        fn headerFlagSetIfClear(h: *Header, mask: u8, comptime order: std.builtin.AtomicOrder) bool {
            var current = @atomicLoad(u8, &h.flags, .monotonic);
            while (true) {
                if (current & mask != 0) return false;
                if (@cmpxchgWeak(u8, &h.flags, current, current | mask, order, .monotonic)) |observed| {
                    current = observed;
                    continue;
                }
                return true;
            }
        }

        /// Fixed offset from a header to its payload. 16-byte aligned so any
        /// cell whose `@alignOf <= 16` (every normal Zig struct on 64-bit) is
        /// correctly aligned, and `payload - header_stride` recovers the header
        /// in O(1) regardless of cell type.
        const header_stride = std.mem.alignForward(usize, @sizeOf(Header), 16);

        /// Uniform heap walk. Exact owned-storage bindings enumerate their
        /// published slots in slab order; every other binding retains the
        /// intrusive-list fallback. Owned iteration is only requested at a
        /// heap-quiescent boundary or while `alloc_lock` excludes publication.
        const CellIterator = struct {
            heap: *Self,
            intrusive: ?*Header,
            owned: OwnedCellIterator = undefined,
            use_owned: bool = false,

            fn init(heap: *Self) @This() {
                var result = @This(){ .heap = heap, .intrusive = heap.all };
                if (comptime has_owned_cell_iterator) {
                    if (heap.bindingAllCellsUseOwnedStorage()) {
                        result.owned = Binding.ownedCellIterator(heap.ctx);
                        result.use_owned = true;
                    }
                }
                return result;
            }

            fn next(self: *@This()) ?*Header {
                if (comptime has_owned_cell_iterator) {
                    if (self.use_owned) {
                        const allocation = self.owned.next() orelse return null;
                        return @ptrCast(@alignCast(allocation));
                    }
                }
                const header = self.intrusive orelse return null;
                self.intrusive = header.next;
                return header;
            }
        };

        const PublicationShard = struct {
            owner: std.atomic.Value(usize) = .init(0),
            active: std.atomic.Value(bool) = .init(false),
            live_cells: usize = 0,
            live_bytes: std.atomic.Value(usize) = .init(0),
        };

        // Keep independent mutator counters on separate cache lines. The
        // binding's owned bitmap remains the authoritative cell walk; shards
        // carry only aggregate deltas since the last collector fold.
        const CacheAlignedPublicationShard = struct {
            shard: PublicationShard = .{},
            padding: [64 - @sizeOf(PublicationShard)]u8 = @splat(0),
        };

        const PublicationTotals = struct {
            live_cells: usize = 0,
            live_bytes: usize = 0,
            young_cells: usize = 0,
            young_bytes: usize = 0,
        };

        /// Exact backing allocation size for a cell payload type. Embedders that
        /// claim all cells use owned storage use this in an exhaustive comptime
        /// proof over their cell taxonomy.
        pub fn cellAllocationBytes(comptime T: type) usize {
            if (@sizeOf(T) > std.math.maxInt(u32))
                @compileError("zig-gc cell payload exceeds the 32-bit header size field");
            return header_stride + @sizeOf(T);
        }

        backing: std.mem.Allocator,
        /// Scratch allocator for the collector's own bookkeeping that the marker
        /// thread and a mutator both touch during a *concurrent* mark — the
        /// `mark_stack` (grown by the marker) and `barrier_buf` (grown by the
        /// mutator's barrier/born-grey path). Defaults to `backing`, so M1/M2 are
        /// byte-identical; for concurrent marking (M3) the embedder installs a
        /// thread-safe allocator via `setAuxAllocator` so these two threads don't
        /// race on a shared non-thread-safe allocator. Cell slabs stay on
        /// `backing`, which only the (GIL-serialized) mutator allocates from
        /// while the marker runs, so it needs no such guarantee.
        aux: std.mem.Allocator = undefined,
        ctx: *Binding,
        /// Intrusive singly-linked list of every live cell (sweep walks it).
        all: ?*Header = null,
        /// Exact live-payload index for broad "maybe managed" barrier/root
        /// inputs. The all-list remains authoritative; this map is a fast path
        /// maintained under `alloc_lock` alongside `all`. If it ever cannot grow,
        /// lookup falls back to the all-list walk, preserving the same safety
        /// semantics for wild/stale pointers.
        payload_index: std.AutoHashMapUnmanaged(usize, *Header) = .empty,
        /// Large owned batches may publish through one stable per-thread shard
        /// without acquiring or modifying a heap-wide word. Collectors close
        /// this read-mostly gate and drain active publishers before folding the
        /// aggregate deltas. Marking keeps it closed and uses the proven global
        /// born-grey path.
        sharded_publication_gate: std.atomic.Value(bool) = .init(true),
        publication_shards: [publication_shard_count]CacheAlignedPublicationShard = @splat(.{}),
        live_cells: usize = 0,
        bytes_live: usize = 0,
        /// Collect when `bytes_live` crosses this; reset to 2× live after a
        /// collection (a simple allocation-rate-agnostic growth policy).
        threshold_bytes: usize = 64 * 1024,
        mark_stack: std.ArrayListUnmanaged(*Header) = .empty,
        weak_slots: std.ArrayListUnmanaged(*?*anyopaque) = .empty,
        /// Weak roots owned by concurrent embedders. Unlike cell-internal weak
        /// slots, these may be read or cleared while marking is active, so the
        /// collector snapshots with acquire and clears with a compare-exchange.
        weak_atomic_slots: std.ArrayListUnmanaged(*std.atomic.Value(?*anyopaque)) = .empty,
        marked_count: usize = 0,
        collections: usize = 0,
        full_collections: usize = 0,
        last_full_collection_bytes: usize = 0,
        minor_collections: usize = 0,
        promoted_cells: usize = 0,
        promoted_bytes: usize = 0,
        young_cells: usize = 0,
        young_bytes: usize = 0,
        last_minor_young_bytes: usize = 0,
        last_minor_reclaimed_bytes: usize = 0,
        last_minor_survived_cells: usize = 0,
        last_minor_survived_bytes: usize = 0,
        last_minor_promoted_bytes: usize = 0,
        total_minor_young_bytes: usize = 0,
        total_minor_reclaimed_bytes: usize = 0,
        total_minor_survived_bytes: usize = 0,
        total_minor_promoted_bytes: usize = 0,
        nursery_threshold_bytes: usize = default_nursery_threshold_bytes,
        tenuring_age: u8 = default_tenuring_age,
        nursery_enabled: bool = false,
        collection_kind: CollectionKind = .full,
        remembered_owners: std.ArrayListUnmanaged(*Header) = .empty,
        remembered_targets: std.ArrayListUnmanaged(*Header) = .empty,
        remember_lock: std.atomic.Value(u32) = .init(0),
        nursery_force_full: std.atomic.Value(bool) = .init(false),
        /// True between `startMarking` and `finishMarking`: an incremental mark
        /// is in progress and the mutator runs between `markStep`s. While set,
        /// the insertion `writeBarrier` shades stored-into cells grey, and freshly
        /// allocated cells are born grey (so they survive the in-progress cycle
        /// and their creation-time field writes are caught when traced; later
        /// stores are caught by the barrier). M3 runs this phase concurrently
        /// with mutators (`concurrent`); M2 keeps it under the GIL.
        ///
        /// Atomic because the *parallel* M3 model toggles it on the collector
        /// while peer mutators read it on their `create` / `writeBarrier` hot
        /// paths with no GIL. The load is `.acquire` / store `.release`, which
        /// on x86_64/arm64 is a plain `mov` — the M1/M2 fast path is unchanged.
        marking: std.atomic.Value(bool) = .init(false),
        /// True during a *concurrent* mark (M3): the marker runs on its own
        /// thread while mutators keep executing. The mark-claim then uses an
        /// atomic compare-and-set on the cell's mark bit (so marker and mutator
        /// never double-push), and the mutator's `writeBarrier` hands greyed
        /// cells to the marker through the lock-guarded `barrier_buf` instead of
        /// touching the marker-private `mark_stack`. Atomic for the same reason
        /// as `marking` — peer mutators read it lock-free under parallel M3.
        concurrent: std.atomic.Value(bool) = .init(false),
        /// Mutator→marker hand-off buffer for concurrent marking, guarded by
        /// `barrier_lock` (a brief atomic spinlock; no std.Thread.Mutex in this
        /// zig). The marker drains it into `mark_stack` between local rounds.
        barrier_buf: std.ArrayListUnmanaged(*Header) = .empty,
        barrier_lock: std.atomic.Value(u32) = .init(0),
        /// Cells allocated *during* a concurrent mark. They are born marked (so
        /// the in-progress cycle never sweeps them) but are NOT handed to the
        /// marker mid-cycle — the mutator is still initializing the payload, so
        /// tracing it concurrently would read a half-built cell (and race the
        /// init that overwrites a per-object lock). Instead they accumulate here
        /// (mutator-private) and are folded into the mark stack at the
        /// world-stopped `finishConcurrentMark`, where the mutator is quiescent
        /// and every payload is complete. Uses `aux` (page allocator) so it
        /// doesn't contend with the marker's `mark_stack` on the cell backing.
        born_concurrent: std.ArrayListUnmanaged(*Header) = .empty,
        /// Cells the marker chose to *trace* at the world-stopped finish rather
        /// than mid-cycle (via `Visitor.deferToFinish`) — those whose mutable
        /// storage can't be read safely while the mutator runs. They are already
        /// marked (won't be swept); only their outgoing edges are discovered at
        /// finish. Marker-thread-private during rounds; drained at finish after
        /// the marker has joined. Uses `aux`.
        deferred_trace: std.ArrayListUnmanaged(*Header) = .empty,
        /// True when *multiple* mutators allocate concurrently (the post-GIL
        /// model). Then `create`'s shared-state bookkeeping — the `all`-list
        /// prepend, the live/bytes counters, and the born-cell hand-off — runs
        /// under `alloc_lock`, and `backing` must itself be thread-safe. Off (the
        /// default, and today's single-GIL'd-mutator model) → that bookkeeping is
        /// lock-free, so single-threaded allocation pays nothing.
        parallel: bool = false,
        /// True when a dedicated marker thread can read allocation metadata
        /// (`all`, `payload_index`, address snapshots) while the single mutator
        /// keeps allocating. This is intentionally distinct from `parallel`:
        /// there is still only one allocator-mutator, so the embedder must not
        /// take the parallel finish/abort path merely to make marker lookups
        /// TSan-clean.
        concurrent_marker_metadata: bool = false,
        alloc_lock: std.atomic.Value(u32) = .init(0),
        /// Test-only telemetry used to prove batched publication acquires the
        /// heap metadata lock once. The increment compiles out of release
        /// builds; keeping the field unconditional avoids changing Self's type
        /// between test declarations.
        alloc_lock_acquisitions_for_testing: usize = 0,
        sharded_publications_for_testing: std.atomic.Value(usize) = .init(0),
        /// Address-sorted snapshot of live cell extents, used to answer interior
        /// pointer queries (`markConservativeWord`) in O(log n) instead of an
        /// O(n) walk of the `all` list per word. Built lazily on the first
        /// conservative mark within a collection (so precise collections never
        /// pay for it) and reused for the rest of that collection's mark phase,
        /// which neither allocates nor frees cells. See `headerForInteriorAddress`.
        addr_index: std.ArrayListUnmanaged(AddrEntry) = .empty,
        addr_index_built: bool = false,

        const AddrEntry = struct { start: usize, end: usize, header: *Header };
        const CollectionKind = enum { full, minor };

        pub const Visitor = struct {
            heap: *Self,
            /// Set if the mark stack couldn't grow mid-collection. A real M2/M3
            /// collector pre-sizes or falls back to a conservative re-scan; M1
            /// surfaces it so the embedder can grow the reserve and retry.
            oom: bool = false,

            /// Mark a strong reference. Null-safe and idempotent (tri-color:
            /// white→grey on first sight, pushed once). The white→grey claim is
            /// atomic under a concurrent mark so the marker and a mutator's
            /// `writeBarrier` never both push the same cell.
            pub fn mark(v: *Visitor, cell: ?*anyopaque) void {
                const p = cell orelse return;
                const h = Self.headerOf(p);
                if (h.magic != header_magic) std.debug.panic("GC mark of non-GC cell at 0x{x}", .{@intFromPtr(p)});
                if (v.heap.collection_kind == .minor and !headerFlagLoad(h, header_young, .monotonic)) return;
                if (!v.heap.claimMark(h)) return; // already grey/black
                v.heap.mark_stack.append(v.heap.aux, h) catch {
                    v.oom = true;
                };
            }

            /// Whether `cell` is one of this heap's managed payloads. Bindings use
            /// this before marking legacy/embedder pointers that may still point
            /// outside the GC heap. Unlike `mark`, this predicate must tolerate
            /// stale or wild values: root tracers often use it specifically at
            /// mixed ownership boundaries, where a header peek would turn a bad
            /// legacy pointer into a collector crash. So this walks the heap's
            /// live-cell list for an exact payload match instead of reading from
            /// the candidate address. The walk is intentionally paid only by
            /// "maybe managed" compatibility edges; precise edges should call
            /// `mark` directly.
            pub fn isManaged(v: *Visitor, cell: ?*anyopaque) bool {
                const p = cell orelse return false;
                return v.heap.headerForPayload(p) != null;
            }

            /// Whether a cell is already black/grey in the current collection.
            /// Used by ephemeron tables: if the key is live, the value is a
            /// strong edge; if the key stays white, the entry is weak.
            pub fn isMarked(v: *Visitor, cell: ?*anyopaque) bool {
                const p = cell orelse return false;
                if (v.heap.collection_kind == .minor and !headerFlagLoad(Self.headerOf(p), header_young, .monotonic)) return true;
                // Atomic: the parallel marker reads `marked` (here, in the sweep
                // phase, and in conservative marking) while a mutator's `claimMark`
                // write barrier CASes it (`.acq_rel`); a plain read races that. A
                // relaxed load is a plain mov — the marking-phase handshake already
                // orders it — so this is free.
                return headerFlagLoad(Self.headerOf(p), header_marked, .monotonic);
            }

            /// Conservatively mark a machine word if it points at the payload
            /// of a managed cell. This is intentionally opt-in: precise
            /// embedders should keep using `mark`, while runtimes that need to
            /// root native stacks can scan a stack/register spill range without
            /// teaching the collector about their frame layout.
            pub fn markConservativeWord(v: *Visitor, word: usize) void {
                const h = v.heap.headerForInteriorAddress(word) orelse return;
                if (v.heap.collection_kind == .minor and !headerFlagLoad(h, header_young, .monotonic)) return;
                // `claimMark` is the atomic CAS under a concurrent/parallel mark
                // (and a plain check otherwise), so a conservative root claim
                // can't race a peer mutator's `writeBarrier` claim on the same
                // cell. `mark_stack` stays collector-private, so the append is
                // single-threaded.
                if (!v.heap.claimMark(h)) return;
                v.heap.mark_stack.append(v.heap.aux, h) catch {
                    v.oom = true;
                };
            }

            /// Scan a word-aligned range, inclusive of `start` and spanning
            /// `words` machine words. The caller owns choosing safe stack or
            /// register-spill bounds for its platform.
            pub fn markConservativeWords(v: *Visitor, start: [*]const usize, words: usize) void {
                var i: usize = 0;
                while (i < words) : (i += 1) v.markConservativeWord(start[i]);
            }

            /// Whether this trace is running on the marker thread *concurrently*
            /// with live mutators (M3). Bindings whose cells have internally
            /// mutable storage (a growable slot/element vector behind a lock)
            /// must, when this is true, read that storage under the same lock the
            /// mutator takes — otherwise the marker's read races a mutator's
            /// append/realloc. False under stop-the-world (M1) and GIL-held
            /// incremental (M2) marking, where the world is quiescent during the
            /// read, so the binding can skip the lock on those paths.
            pub fn concurrent(v: *Visitor) bool {
                return v.heap.concurrent.load(.acquire);
            }

            /// Defer this (already-marked) cell's *tracing* to the world-stopped
            /// `finishConcurrentMark`. For cells whose mutable storage is too
            /// entangled to read safely mid-mark (e.g. a generator whose `exec`
            /// is the live VM stack, or an iterator helper whose fields update
            /// around JS callbacks): the binding calls this from `trace` when
            /// `concurrent()`, so the cell survives this cycle (it is marked) but
            /// its children are discovered at finish, when the mutator is
            /// quiescent and the storage is stable. A no-op outside a concurrent
            /// mark (the caller should just trace normally then).
            pub fn deferToFinish(v: *Visitor, cell: *anyopaque) void {
                v.heap.deferred_trace.append(v.heap.aux, Self.headerOf(cell)) catch {
                    v.oom = true;
                };
            }

            /// Register a weak slot. After marking completes, if its target
            /// stayed white the slot is set to null (the cell is dying).
            pub fn markWeak(v: *Visitor, slot: *?*anyopaque) void {
                v.heap.lockWeak();
                defer v.heap.unlockWeak();
                v.heap.weak_slots.append(v.heap.aux, slot) catch {
                    v.oom = true;
                };
            }

            /// Register an externally synchronized weak slot. Clearing uses a
            /// CAS so a concurrent embedder clear cannot race a plain store and
            /// a future retargeting API cannot lose a newly published target.
            pub fn markWeakAtomic(v: *Visitor, slot: *std.atomic.Value(?*anyopaque)) void {
                v.heap.lockWeak();
                defer v.heap.unlockWeak();
                v.heap.weak_atomic_slots.append(v.heap.aux, slot) catch {
                    v.oom = true;
                };
            }
        };

        pub fn init(backing: std.mem.Allocator, ctx: *Binding) Self {
            return .{ .backing = backing, .aux = backing, .ctx = ctx };
        }

        /// Install a thread-safe scratch allocator for concurrent marking (M3).
        /// Must be called right after `init`, before any allocation, since
        /// `mark_stack`/`barrier_buf` must be freed with the same allocator they
        /// were grown with. A no-op conceptually for M1/M2 (leave it as `backing`).
        pub fn setAuxAllocator(self: *Self, aux: std.mem.Allocator) void {
            self.aux = aux;
        }

        /// Enable multi-mutator allocation (the post-GIL model): `create`'s
        /// shared-state bookkeeping runs under `alloc_lock`. `backing` must be
        /// thread-safe. Leave off (default) for the single-GIL'd-mutator model so
        /// allocation pays no lock.
        pub fn setParallel(self: *Self, parallel: bool) void {
            self.parallel = parallel;
        }

        /// Serialize allocation metadata against a single dedicated marker
        /// thread. Use this for concurrent marking without enabling the
        /// multi-mutator `parallel` collector protocol.
        pub fn setConcurrentMarkerMetadata(self: *Self, enabled: bool) void {
            self.concurrent_marker_metadata = enabled;
        }

        /// Enable the non-moving nursery. New cells start at age zero; a minor
        /// collection reclaims unreachable young cells, advances live survivors,
        /// and tenures cells that reach `tenuring_age`. Existing cells stay old,
        /// so enabling this after heap initialization is safe. Disabling with a
        /// pending nursery tenures that entire young prefix without collecting:
        /// later old allocations can therefore never split the prefix before a
        /// re-enable.
        pub fn setNurseryEnabled(self: *Self, enabled: bool) void {
            std.debug.assert(!self.marking.load(.acquire));
            if (self.parallel) self.lockAlloc();
            defer if (self.parallel) self.unlockAlloc();
            const reopen_shards = self.closeAndFoldPublicationShards();
            defer if (reopen_shards) self.reopenShardedPublication();
            if (!enabled and self.nursery_enabled and self.young_cells != 0)
                self.tenureYoungPrefix();
            if (!enabled) self.clearRemembered();
            self.nursery_enabled = enabled;
        }

        /// Select how many successful minor collections a young cell must
        /// survive before promotion. One preserves the original single-cycle
        /// nursery policy; larger values retain a measured multi-age nursery.
        pub fn setNurseryTenuringAge(self: *Self, age: u8) void {
            std.debug.assert(age > 0 and age < tenured_age);
            std.debug.assert(!self.marking.load(.acquire));
            self.tenuring_age = age;
        }

        /// Race-safe accounting snapshot for embedders that expose heap usage
        /// or collection telemetry. `live_bytes` includes the collector header
        /// and payload allocation for every currently live cell. The
        /// last-full value changes only after a completed full sweep, never
        /// after a nursery-only cycle or an in-progress concurrent mark.
        pub const Accounting = struct {
            live_cells: usize,
            live_bytes: usize,
            last_full_collection_bytes: usize,
            collections: usize,
            full_collections: usize,
            minor_collections: usize,
            young_cells: usize,
            young_bytes: usize,
            promoted_cells: usize,
            promoted_bytes: usize,
            tenuring_age: u8,
            last_minor_young_bytes: usize,
            last_minor_reclaimed_bytes: usize,
            last_minor_survived_cells: usize,
            last_minor_survived_bytes: usize,
            last_minor_promoted_bytes: usize,
            /// Historical byte totals across successful minor collections.
            /// These do not reset during a full collection or nursery toggle.
            total_minor_young_bytes: usize,
            total_minor_reclaimed_bytes: usize,
            total_minor_survived_bytes: usize,
            total_minor_promoted_bytes: usize,
        };

        pub const CompactionStatus = enum {
            unsupported,
            no_candidates,
            out_of_memory,
            compacted,
        };

        pub const CompactionResult = struct {
            status: CompactionStatus,
            moved_cells: usize = 0,
            moved_bytes: usize = 0,
        };

        pub fn accounting(self: *Self) Accounting {
            if (self.syncAllocMetadata()) self.lockAlloc();
            defer if (self.syncAllocMetadata()) self.unlockAlloc();
            const reopen_shards = self.closeAndFoldPublicationShards();
            defer if (reopen_shards and !self.marking.load(.acquire)) self.reopenShardedPublication();
            return .{
                .live_cells = self.live_cells,
                .live_bytes = self.bytes_live,
                .last_full_collection_bytes = self.last_full_collection_bytes,
                .collections = self.collections,
                .full_collections = self.full_collections,
                .minor_collections = self.minor_collections,
                .young_cells = self.young_cells,
                .young_bytes = self.young_bytes,
                .promoted_cells = self.promoted_cells,
                .promoted_bytes = self.promoted_bytes,
                .tenuring_age = self.tenuring_age,
                .last_minor_young_bytes = self.last_minor_young_bytes,
                .last_minor_reclaimed_bytes = self.last_minor_reclaimed_bytes,
                .last_minor_survived_cells = self.last_minor_survived_cells,
                .last_minor_survived_bytes = self.last_minor_survived_bytes,
                .last_minor_promoted_bytes = self.last_minor_promoted_bytes,
                .total_minor_young_bytes = self.total_minor_young_bytes,
                .total_minor_reclaimed_bytes = self.total_minor_reclaimed_bytes,
                .total_minor_survived_bytes = self.total_minor_survived_bytes,
                .total_minor_promoted_bytes = self.total_minor_promoted_bytes,
            };
        }

        fn tenureYoungPrefix(self: *Self) void {
            var promoted_cells: usize = 0;
            var promoted_bytes: usize = 0;
            const owned_enumeration = self.bindingEnumeratesOwnedCells();
            var it = self.cellIterator();
            while (it.next()) |h| {
                if (!headerFlagLoad(h, header_young, .monotonic)) {
                    if (owned_enumeration) continue;
                    break;
                }
                headerFlagStore(h, header_young, false, .release);
                h.age = tenured_age;
                promoted_cells += 1;
                promoted_bytes += header_stride + headerPayloadSize(h);
            }
            std.debug.assert(promoted_cells == self.young_cells);
            std.debug.assert(promoted_bytes == self.young_bytes);
            self.young_cells = 0;
            self.young_bytes = 0;
            self.promoted_cells += promoted_cells;
            self.promoted_bytes += promoted_bytes;
        }

        fn doubledBytes(bytes: usize) usize {
            const max = std.math.maxInt(usize);
            if (bytes > max / 2) return max;
            return bytes * 2;
        }

        fn nextNurseryThreshold(self: *Self, young_bytes: usize, survivor_bytes: usize) usize {
            const survivor_target = @max(min_nursery_threshold_bytes, doubledBytes(survivor_bytes));
            const decay_floor = @max(min_nursery_threshold_bytes, self.nursery_threshold_bytes / 2);
            const adaptive = @max(survivor_target, decay_floor);
            // Decay remains gradual, but growth should not jump past the young
            // batch that was just measured. Otherwise high-survival bursts can
            // skip the next quiescent minor and carry a whole dead batch until a
            // later boundary.
            if (adaptive <= self.nursery_threshold_bytes or young_bytes == 0) return adaptive;
            const observed_cap = @max(self.nursery_threshold_bytes, @max(min_nursery_threshold_bytes, young_bytes));
            return @min(adaptive, observed_cap);
        }

        fn headerOf(payload: *anyopaque) *Header {
            const raw: [*]u8 = @ptrCast(payload);
            return @ptrCast(@alignCast(raw - header_stride));
        }

        fn payloadOf(h: *Header) *anyopaque {
            const raw: [*]u8 = @ptrCast(h);
            return @ptrCast(raw + header_stride);
        }

        fn payloadKey(payload: *anyopaque) usize {
            return @intFromPtr(payload);
        }

        inline fn bindingOwnsCellAllocation(self: *Self, allocation: *anyopaque) bool {
            if (@hasDecl(Binding, "ownsCellAllocation"))
                return self.ctx.ownsCellAllocation(allocation);
            return false;
        }

        inline fn bindingUsesOwnedCellStorage(self: *Self, total: usize) bool {
            if (@hasDecl(Binding, "usesOwnedCellStorage"))
                return self.ctx.usesOwnedCellStorage(total);
            return false;
        }

        inline fn bindingClassifyConservativeInterior(self: *Self, address: usize) ?InteriorOwnership {
            if (@hasDecl(Binding, "classifyConservativeInterior"))
                return self.ctx.classifyConservativeInterior(address);
            return null;
        }

        inline fn bindingAllCellsUseOwnedStorage(self: *Self) bool {
            if (@hasDecl(Binding, "allCellsUseOwnedStorage"))
                return self.ctx.allCellsUseOwnedStorage();
            return false;
        }

        inline fn bindingEnumeratesOwnedCells(self: *Self) bool {
            if (comptime !has_owned_cell_iterator) return false;
            return self.bindingAllCellsUseOwnedStorage();
        }

        fn reserveRelocationCell(self: *Self, total: usize) ?*anyopaque {
            if (@hasDecl(Binding, "reserveRelocationCell"))
                return self.ctx.reserveRelocationCell(total);
            const slab = self.backing.alignedAlloc(u8, .@"16", total) catch return null;
            return slab.ptr;
        }

        fn releaseRelocationReservation(self: *Self, allocation: *anyopaque, total: usize) void {
            if (@hasDecl(Binding, "releaseRelocationReservation")) {
                self.ctx.releaseRelocationReservation(allocation, total);
                return;
            }
            const base: [*]align(16) u8 = @ptrCast(@alignCast(allocation));
            self.backing.free(base[0..total]);
        }

        fn commitRelocationCell(self: *Self, old: *Header, new: *Header, total: usize) void {
            if (@hasDecl(Binding, "commitRelocationCell")) {
                self.ctx.commitRelocationCell(old, new, total);
                return;
            }
            self.bindingUnpublishCellAllocation(old, total);
            self.bindingPublishCellAllocation(new, total);
            const base: [*]align(16) u8 = @ptrCast(@alignCast(old));
            self.backing.free(base[0..total]);
        }

        inline fn cellIterator(self: *Self) CellIterator {
            return CellIterator.init(self);
        }

        fn currentPublicationShard(self: *Self) ?*PublicationShard {
            const thread_id: usize = @intCast(std.Thread.getCurrentId());
            const owner = thread_id +% 1;
            const token = if (owner == 0) std.math.maxInt(usize) else owner;
            var empty: ?*PublicationShard = null;
            for (&self.publication_shards) |*entry| {
                const shard = &entry.shard;
                const existing = shard.owner.load(.acquire);
                if (existing == token) return shard;
                if (existing == 0 and empty == null) empty = shard;
            }
            const shard = empty orelse return null;
            if (shard.owner.cmpxchgStrong(0, token, .acq_rel, .acquire)) |existing| {
                if (existing == token) return shard;
                // Another first-use registration won this slot. Retry the
                // bounded table so each OS thread keeps one stable shard.
                return self.currentPublicationShard();
            }
            return shard;
        }

        fn beginShardedPublication(self: *Self, nursery_snapshot: bool) ?*PublicationShard {
            if (!self.bindingEnumeratesOwnedCells()) return null;
            if (!self.sharded_publication_gate.load(.seq_cst)) return null;
            const shard = self.currentPublicationShard() orelse return null;
            shard.active.store(true, .seq_cst);
            if (!self.sharded_publication_gate.load(.seq_cst) or
                self.marking.load(.acquire) or self.nursery_enabled != nursery_snapshot)
            {
                shard.active.store(false, .seq_cst);
                return null;
            }
            return shard;
        }

        fn finishShardedPublication(self: *Self, shard: *PublicationShard, count: usize, bytes: usize) void {
            shard.live_cells += count;
            _ = shard.live_bytes.fetchAdd(bytes, .monotonic);
            if (builtin.is_test) _ = self.sharded_publications_for_testing.fetchAdd(1, .monotonic);
            shard.active.store(false, .seq_cst);
        }

        fn publicationTotals(self: *Self) PublicationTotals {
            if (!self.bindingEnumeratesOwnedCells()) return .{};
            var totals = PublicationTotals{};
            for (&self.publication_shards) |*entry| {
                const shard = &entry.shard;
                totals.live_bytes += shard.live_bytes.load(.acquire);
            }
            if (self.nursery_enabled) {
                totals.young_cells = totals.live_cells;
                totals.young_bytes = totals.live_bytes;
            }
            return totals;
        }

        /// Stop new private publishers, wait for any batch that passed the gate,
        /// and fold every shard exactly once. Sequentially consistent gate/
        /// active handshakes close the "collector saw idle while publisher saw
        /// open" race without any heap-wide publisher RMW.
        fn closeAndFoldPublicationShards(self: *Self) bool {
            if (!self.bindingEnumeratesOwnedCells()) return false;
            const was_open = self.sharded_publication_gate.swap(false, .seq_cst);
            for (&self.publication_shards) |*entry| {
                const shard = &entry.shard;
                while (shard.active.load(.seq_cst)) std.atomic.spinLoopHint();
                const cells = shard.live_cells;
                shard.live_cells = 0;
                const bytes = shard.live_bytes.swap(0, .acq_rel);
                self.live_cells += cells;
                self.bytes_live += bytes;
                if (self.nursery_enabled) {
                    self.young_cells += cells;
                    self.young_bytes += bytes;
                }
            }
            return was_open;
        }

        inline fn reopenShardedPublication(self: *Self) void {
            if (self.bindingEnumeratesOwnedCells())
                self.sharded_publication_gate.store(true, .seq_cst);
        }

        inline fn bindingPublishCellAllocation(self: *Self, allocation: *anyopaque, total: usize) void {
            if (@hasDecl(Binding, "publishCellAllocation"))
                self.ctx.publishCellAllocation(allocation, total);
        }

        inline fn bindingPublishCellAllocationBatch(self: *Self, payloads: []*anyopaque, total: usize, payload_offset: usize) void {
            if (@hasDecl(Binding, "publishCellAllocationBatch")) {
                self.ctx.publishCellAllocationBatch(payloads, total, payload_offset);
            } else if (@hasDecl(Binding, "publishCellAllocation")) {
                for (payloads) |payload| {
                    const base: *anyopaque = @ptrFromInt(@intFromPtr(payload) - payload_offset);
                    self.ctx.publishCellAllocation(base, total);
                }
            }
        }

        inline fn bindingUnpublishCellAllocation(self: *Self, allocation: *anyopaque, total: usize) void {
            if (@hasDecl(Binding, "unpublishCellAllocation"))
                self.ctx.unpublishCellAllocation(allocation, total);
        }

        inline fn bindingUnpublishCellAllocationBatch(self: *Self, total: usize, allocations: []*anyopaque) void {
            if (@hasDecl(Binding, "unpublishCellAllocationBatch")) {
                self.ctx.unpublishCellAllocationBatch(total, allocations);
            } else {
                for (allocations) |allocation| self.bindingUnpublishCellAllocation(allocation, total);
            }
        }

        fn syncAllocMetadata(self: *Self) bool {
            return self.parallel or self.concurrent_marker_metadata;
        }

        fn headerForPayloadSlowLocked(self: *Self, payload: *anyopaque) ?*Header {
            var it = self.cellIterator();
            while (it.next()) |h| {
                if (payloadOf(h) == payload) return h;
            }
            return null;
        }

        fn headerForPayload(self: *Self, payload: *anyopaque) ?*Header {
            if (self.syncAllocMetadata()) self.lockAlloc();
            defer if (self.syncAllocMetadata()) self.unlockAlloc();
            const candidate = headerOf(payload);
            // An embedder with exact slab ownership metadata can validate the
            // candidate address before we dereference it. Header magic supplies
            // liveness; the hash/list path remains authoritative otherwise.
            if (self.bindingOwnsCellAllocation(@ptrCast(candidate)) and candidate.magic == header_magic)
                return candidate;
            if (self.payload_index.get(payloadKey(payload))) |h| return h;
            return self.headerForPayloadSlowLocked(payload);
        }

        /// Return immutable diagnostics metadata for a live cell. The embedder
        /// must call this at a heap-quiescent boundary, so collection cannot
        /// reclaim the returned cell between lookup and the metadata reads.
        /// Null and unmanaged/stale payloads return null.
        pub fn cellMetadata(self: *Self, payload: ?*anyopaque) ?CellMetadata {
            const ptr = payload orelse return null;
            const h = self.headerForPayload(ptr) orelse return null;
            return .{ .id = h.stable_id, .kind = h.kind, .size = headerPayloadSize(h) };
        }

        fn indexPayloadLocked(self: *Self, h: *Header) void {
            if (self.bindingUsesOwnedCellStorage(header_stride + headerPayloadSize(h))) return;
            self.payload_index.put(self.backing, payloadKey(payloadOf(h)), h) catch {};
        }

        fn unindexPayloadLocked(self: *Self, h: *Header) void {
            if (self.bindingUsesOwnedCellStorage(header_stride + headerPayloadSize(h))) return;
            _ = self.payload_index.remove(payloadKey(payloadOf(h)));
        }

        /// Whether `p` is a live (marked) cell — O(1). `p` must be null or a
        /// pointer to a cell allocated by this heap. This is the read a binding
        /// uses for **isMarked-based weak clearing**: deciding a weak key's /
        /// finalizer target's liveness in the world-stopped finish pass by its
        /// mark bit, instead of pre-registering an interior `&slot` weak pointer
        /// that a concurrent mutator append could dangle by reallocating the
        /// buffer it points into. Call only with marks still valid (before sweep).
        pub fn isLive(self: *Self, p: ?*anyopaque) bool {
            const ptr = p orelse return false;
            if (self.collection_kind == .minor and !headerFlagLoad(headerOf(ptr), header_young, .monotonic)) return true;
            return headerFlagLoad(headerOf(ptr), header_marked, .monotonic); // vs concurrent claimMark CAS
        }

        fn headerForInteriorAddress(self: *Self, address: usize) ?*Header {
            if (self.bindingClassifyConservativeInterior(address)) |ownership| switch (ownership) {
                .allocation => |base| {
                    const h: *Header = @ptrCast(@alignCast(base));
                    if (h.magic != header_magic) return null;
                    const start = @intFromPtr(payloadOf(h));
                    const size = @max(headerPayloadSize(h), 1);
                    return if (address >= start and address < start + size) h else null;
                },
                .owned_empty => return null,
                .outside => if (self.bindingAllCellsUseOwnedStorage()) return null,
            };
            // Lazily build the address-sorted index on first use within a
            // collection (reset by `collect`). The mark phase never allocates or
            // frees cells, so a snapshot stays valid for the whole phase.
            if (!self.addr_index_built) self.buildAddrIndex();
            const items = self.addr_index.items;
            // Binary search for the rightmost extent whose `start <= address`.
            var lo: usize = 0;
            var hi: usize = items.len;
            while (lo < hi) {
                const mid = lo + (hi - lo) / 2;
                if (items[mid].start <= address) lo = mid + 1 else hi = mid;
            }
            if (lo == 0) return null;
            const e = items[lo - 1];
            if (address < e.end) return e.header;
            return null;
        }

        fn buildAddrIndex(self: *Self) void {
            // Snapshot the all-list under `alloc_lock` so a peer mutator's
            // `create` (which prepends to `all`) can't race the walk. New cells
            // born after this snapshot aren't in the index — fine: they are born
            // marked, so a conservative pointer into one needs no index hit.
            if (self.syncAllocMetadata()) self.lockAlloc();
            defer if (self.syncAllocMetadata()) self.unlockAlloc();
            self.addr_index.clearRetainingCapacity();
            var it = self.cellIterator();
            while (it.next()) |h| {
                const start = @intFromPtr(payloadOf(h));
                // A zero-size payload would make an empty extent that no interior
                // address can fall into; record at least one byte so an exact
                // payload pointer still resolves.
                const size = @max(headerPayloadSize(h), 1);
                self.addr_index.append(self.aux, .{
                    .start = start,
                    .end = start + size,
                    .header = h,
                }) catch {
                    // On OOM, fall back to leaving the index partial; a missed
                    // interior pointer is a missed (conservative) root, which is
                    // unsound, so panic rather than silently under-mark.
                    std.debug.panic("GC: out of memory building conservative address index", .{});
                };
            }
            std.mem.sort(AddrEntry, self.addr_index.items, {}, struct {
                fn lessThan(_: void, a: AddrEntry, b: AddrEntry) bool {
                    return a.start < b.start;
                }
            }.lessThan);
            self.addr_index_built = true;
        }

        inline fn allocateCellSlab(self: *Self, comptime T: type) ![]align(16) u8 {
            const total = header_stride + @sizeOf(T);
            return self.backing.alignedAlloc(u8, .@"16", total) catch |err| blk: {
                if (err == error.OutOfMemory and @hasDecl(Binding, "recoverAllocationFailure")) {
                    if (Binding.recoverAllocationFailure(self.ctx)) {
                        break :blk try self.backing.alignedAlloc(u8, .@"16", total);
                    }
                }
                return err;
            };
        }

        /// Publish one privately allocated header. The caller serializes
        /// allocation metadata when required and supplies one marking-state
        /// snapshot for the complete publication batch.
        inline fn publishCellLocked(self: *Self, comptime T: type, kind: Kind, h: *Header, born_grey: bool, publish_binding: bool) *T {
            const total = cellAllocationBytes(T);
            // Field-wise init (not a struct literal) so packed flags are written
            // *atomically*: under a parallel concurrent mark the marker may
            // `claimMark` this born-grey cell (an atomic CAS) the instant a peer
            // links it behind a traced object, and a non-atomic store racing
            // that CAS is a data race. The CAS only fails (born grey ⇒ already
            // marked), so the marker never traces the half-built payload — the
            // other header fields stay private until the world-stopped finish.
            // `magic`/`next`/`size`/`kind` are only read at that finish (or by
            // sweep under `alloc_lock`), so they need no atomic.
            h.magic = header_magic;
            h.stable_id = allocateStableCellId();
            const owned_enumeration = self.bindingEnumeratesOwnedCells();
            h.next = if (owned_enumeration) null else self.all;
            h.size = @intCast(@sizeOf(T));
            h.kind = kind;
            @atomicStore(u8, &h.flags, (if (born_grey) header_marked else 0) |
                (if (self.nursery_enabled) header_young else 0), .release);
            h.age = if (self.nursery_enabled) 0 else tenured_age;
            // An owned backing may reserve a slot before this private header is
            // initialized. Publish only after every field is complete so its
            // classifier's synchronization makes later header reads race-free.
            if (publish_binding) self.bindingPublishCellAllocation(h, total);
            if (!owned_enumeration) self.all = h;
            self.indexPayloadLocked(h);
            self.live_cells += 1;
            self.bytes_live += total;
            if (self.nursery_enabled) {
                self.young_cells += 1;
                self.young_bytes += total;
            }
            if (born_grey) {
                if (self.concurrent.load(.acquire)) {
                    // Allocated on the mutator thread during a concurrent mark.
                    // Born marked (survives this cycle), but DON'T hand it to the
                    // marker yet — the caller is still initializing this payload,
                    // so the marker must not trace it concurrently. Defer it to
                    // `finishConcurrentMark` (world stopped, payload complete).
                    _ = @atomicRmw(usize, &self.marked_count, .Add, 1, .monotonic);
                    self.born_concurrent.append(self.aux, h) catch {};
                } else {
                    // A parallel mutator can observe a stale `marking=true`
                    // just after abort/finish cleared `concurrent`. The cell is
                    // born marked but no collection is active enough to consume
                    // it; do not touch the marker-private stack from this thread.
                    if (!self.parallel) {
                        self.marked_count += 1;
                        self.mark_stack.append(self.aux, h) catch {};
                    }
                }
            }
            return @ptrCast(@alignCast(@as([*]u8, @ptrCast(h)) + header_stride));
        }

        /// Allocate a GC-managed cell of type `T` tagged `kind`. The returned
        /// pointer is uninitialized payload; the caller writes it before the
        /// next safepoint (so a collection never traces a half-built cell).
        pub fn create(self: *Self, comptime T: type, kind: Kind) !*T {
            comptime std.debug.assert(@alignOf(T) <= 16);
            // The slab alloc happens before the lock: `backing` is thread-safe in
            // `parallel` mode, and `h` is private until linked into `all` below.
            const slab = try self.allocateCellSlab(T);
            const h: *Header = @ptrCast(@alignCast(slab.ptr));
            // Shared-state bookkeeping (all-list prepend, counters, born-cell
            // hand-off) is serialized across mutators in `parallel` mode and
            // against a dedicated marker in single-mutator concurrent mode;
            // otherwise it remains lock-free.
            if (self.syncAllocMetadata()) self.lockAlloc();
            defer if (self.syncAllocMetadata()) self.unlockAlloc();
            // Allocate-grey during an incremental mark: a cell created while
            // marking is in progress is born marked AND queued for tracing, so it
            // survives this cycle and — crucially — its *creation-time* field
            // writes are caught when it is traced (the caller fully initializes
            // the payload before the next safepoint, where the next `markStep`
            // runs). This is what lets the embedder barrier only post-creation
            // mutations instead of every initializing store. Children added after
            // it is traced are caught by the insertion `writeBarrier`.
            return self.publishCellLocked(T, kind, h, self.marking.load(.acquire), true);
        }

        /// Allocate several same-kind cells privately, then publish the
        /// successfully allocated prefix under one metadata lock. Returning a
        /// short prefix defers recovery/OOM until the caller has initialized and
        /// consumed those cells, preserving sequential allocation failure
        /// ordering. Every returned payload is uninitialized and must be fully
        /// initialized before the caller's next safepoint, just like `create`.
        pub fn createBatch(self: *Self, comptime T: type, kind: Kind, out: []*T) !usize {
            comptime std.debug.assert(@alignOf(T) <= 16);
            if (out.len == 0) return 0;

            var allocated: usize = 0;
            if (@hasDecl(Binding, "allocateCellBatch")) {
                // Reuse the caller's pointer array as scratch for private slab
                // bases. Pointer representations are identical; the binding
                // fills only the prefix it reports, and we replace that prefix
                // with payload pointers before publishing it below.
                const raw: []*anyopaque = @as([*]*anyopaque, @ptrCast(out.ptr))[0..out.len];
                allocated = Binding.allocateCellBatch(self.ctx, cellAllocationBytes(T), raw);
                std.debug.assert(allocated <= out.len);
                for (raw[0..allocated], 0..) |slab, i| {
                    const base: [*]u8 = @ptrCast(slab);
                    out[i] = @ptrCast(@alignCast(base + header_stride));
                }
                // A short non-empty prefix means the backing reached its local
                // limit. Publish it now so the caller can initialize and commit
                // the matching work before a later request performs recovery.
                if (allocated != 0) return self.publishBatch(T, kind, out[0..allocated]);
            }
            while (allocated < out.len) {
                // Only the first allocation uses recovery. If a later private
                // slab fails, publish the successful prefix and let the caller
                // commit its corresponding work before requesting another
                // batch, at which point normal recovery/OOM ordering resumes.
                const slab = self.backing.alignedAlloc(u8, .@"16", cellAllocationBytes(T)) catch {
                    if (allocated != 0) break;
                    const recovered = try self.allocateCellSlab(T);
                    out[allocated] = @ptrCast(@alignCast(recovered.ptr + header_stride));
                    allocated += 1;
                    continue;
                };
                out[allocated] = @ptrCast(@alignCast(slab.ptr + header_stride));
                allocated += 1;
            }

            return self.publishBatch(T, kind, out[0..allocated]);
        }

        inline fn publishBatch(self: *Self, comptime T: type, kind: Kind, out: []*T) usize {
            const total = cellAllocationBytes(T);
            // Private header chaining pays for itself on genuinely amortized
            // batches. Keep short batches on the compact proven loop: their
            // lock hold is already small, while building a second private
            // chain adds work without relieving meaningful contention.
            const owned_fast_path = self.bindingAllCellsUseOwnedStorage() and out.len >= 64;
            const owned_enumeration = self.bindingEnumeratesOwnedCells();
            const nursery_snapshot = self.nursery_enabled;
            if (owned_fast_path) {
                if (self.beginShardedPublication(nursery_snapshot)) |shard| {
                    for (out) |payload| {
                        const h = headerOf(payload);
                        h.magic = header_magic;
                        h.stable_id = allocateStableCellId();
                        h.next = null;
                        h.size = @intCast(@sizeOf(T));
                        h.kind = kind;
                        @atomicStore(u8, &h.flags, if (nursery_snapshot) header_young else 0, .release);
                        h.age = if (nursery_snapshot) 0 else tenured_age;
                    }
                    const raw: []*anyopaque = @as([*]*anyopaque, @ptrCast(out.ptr))[0..out.len];
                    self.bindingPublishCellAllocationBatch(raw, total, header_stride);
                    self.finishShardedPublication(shard, out.len, total * out.len);
                    return out.len;
                }
            }
            var private_head: ?*Header = null;
            var private_tail: ?*Header = null;
            if (owned_fast_path and !self.marking.load(.acquire)) {
                for (out) |payload| {
                    const h = headerOf(payload);
                    h.magic = header_magic;
                    h.stable_id = allocateStableCellId();
                    h.next = if (owned_enumeration) null else private_head;
                    h.size = @intCast(@sizeOf(T));
                    h.kind = kind;
                    @atomicStore(u8, &h.flags, if (nursery_snapshot) header_young else 0, .release);
                    h.age = if (nursery_snapshot) 0 else tenured_age;
                    if (!owned_enumeration) {
                        if (private_tail == null) private_tail = h;
                        private_head = h;
                    }
                }
            }

            if (self.syncAllocMetadata()) self.lockAlloc();
            if (owned_fast_path and !self.marking.load(.acquire) and self.nursery_enabled == nursery_snapshot) {
                if (!owned_enumeration) {
                    private_tail.?.next = self.all;
                    self.all = private_head;
                }
                self.live_cells += out.len;
                self.bytes_live += total * out.len;
                if (nursery_snapshot) {
                    self.young_cells += out.len;
                    self.young_bytes += total * out.len;
                }
                if (self.syncAllocMetadata()) self.unlockAlloc();

                // Every header is complete and linked before the binding
                // publishes its ownership bitmap. Do that second publication
                // after releasing alloc_lock: bindings commonly serialize
                // their bitmap by size class, and nesting that lock inside the
                // global heap-publication lock turns unrelated mutators into
                // one long convoy. The payloads remain private to the caller
                // until createBatch returns, while an exact lookup during this
                // short interval can still use the all-list fallback under
                // alloc_lock.
                const raw: []*anyopaque = @as([*]*anyopaque, @ptrCast(out.ptr))[0..out.len];
                self.bindingPublishCellAllocationBatch(raw, total, header_stride);
                return out.len;
            } else {
                const born_grey = self.marking.load(.acquire);
                for (out) |payload| {
                    const h = headerOf(payload);
                    const published = self.publishCellLocked(T, kind, h, born_grey, false);
                    std.debug.assert(published == payload);
                }
            }
            const raw: []*anyopaque = @as([*]*anyopaque, @ptrCast(out.ptr))[0..out.len];
            self.bindingPublishCellAllocationBatch(raw, total, header_stride);
            if (self.syncAllocMetadata()) self.unlockAlloc();
            return out.len;
        }

        /// Collect if the heap has grown past the threshold. Call at safepoints
        /// (the engine's `(steps & 1023)` checkpoints) and after large allocs.
        pub fn maybeCollect(self: *Self) void {
            if (self.nursery_enabled and self.shouldCollectYoung()) {
                self.collectYoung();
            } else if (self.shouldCollect()) {
                self.collect();
            }
        }

        /// Dijkstra insertion write barrier. The embedder calls this whenever it
        /// stores a reference to `cell` into a heap object during an incremental
        /// mark: it shades `cell` grey so a reference newly hidden behind an
        /// already-black object is never missed (the black→white invariant). A
        /// no-op when not marking, when `cell` is null, or when `cell` is not a
        /// managed payload (the embedder may store non-cell pointers) — so it is
        /// cheap and safe to call broadly. Idempotent (already-grey/black: skip).
        pub fn writeBarrier(self: *Self, cell: ?*anyopaque) void {
            self.rememberStrongStore(null, cell);
            self.incrementalBarrier(cell);
        }

        /// Owner-aware insertion barrier. In nursery mode an old `owner` is
        /// remembered when it receives a young child, so minor collection only
        /// rescans dirty old containers. The incremental/full barrier remains
        /// identical to `writeBarrier`.
        pub fn writeBarrierFrom(self: *Self, owner: ?*anyopaque, cell: ?*anyopaque) void {
            self.rememberStrongStore(owner, cell);
            self.incrementalBarrier(cell);
        }

        /// Fast owner-aware barrier for exact live managed payloads allocated by
        /// this heap. Unlike `writeBarrierFrom`, this deliberately does not
        /// classify arbitrary pointers through the live-payload index. The
        /// caller must provide non-null payload starts from this heap; use the
        /// tolerant barrier whenever either pointer may be external or stale.
        pub fn writeBarrierFromManaged(self: *Self, owner: *anyopaque, cell: *anyopaque) void {
            const child = headerOf(cell);
            std.debug.assert(child.magic == header_magic);
            if (self.nursery_enabled and headerFlagLoad(child, header_young, .acquire)) {
                const parent = headerOf(owner);
                std.debug.assert(parent.magic == header_magic);
                if (!headerFlagLoad(parent, header_young, .acquire)) self.rememberOwner(parent);
            }
            if (!self.marking.load(.acquire)) return;
            self.incrementalBarrierHeader(child);
        }

        /// Remember an old container whose weak slots changed. This does not mark
        /// the weak target; it merely ensures minor GC revisits the container to
        /// apply normal weak/ephemeron semantics.
        pub fn writeBarrierWeak(self: *Self, owner: ?*anyopaque) void {
            if (!self.nursery_enabled) return;
            const p = owner orelse {
                self.nursery_force_full.store(true, .release);
                return;
            };
            const h = self.headerForPayload(p) orelse {
                self.nursery_force_full.store(true, .release);
                return;
            };
            if (!headerFlagLoad(h, header_young, .acquire)) self.rememberOwner(h);
        }

        fn rememberStrongStore(self: *Self, owner: ?*anyopaque, cell: ?*anyopaque) void {
            if (!self.nursery_enabled) return;
            const child_ptr = cell orelse return;
            const child = self.headerForPayload(child_ptr) orelse return;
            if (!headerFlagLoad(child, header_young, .acquire)) return;
            if (owner) |owner_ptr| {
                if (self.headerForPayload(owner_ptr)) |parent| {
                    if (!headerFlagLoad(parent, header_young, .acquire)) self.rememberOwner(parent);
                    return;
                }
            }
            // Compatibility path for embedders that only provide the child. It
            // is conservative but sound: retain that target for this nursery
            // cycle, then tenure it and discard the entry.
            if (!headerFlagSetIfClear(child, header_remembered_target, .acq_rel)) return;
            self.lockRemember();
            self.remembered_targets.append(self.aux, child) catch {
                headerFlagStore(child, header_remembered_target, false, .release);
                self.nursery_force_full.store(true, .release);
            };
            self.unlockRemember();
        }

        fn rememberOwner(self: *Self, h: *Header) void {
            if (!headerFlagSetIfClear(h, header_remembered_owner, .acq_rel)) return;
            self.lockRemember();
            self.remembered_owners.append(self.aux, h) catch {
                headerFlagStore(h, header_remembered_owner, false, .release);
                self.nursery_force_full.store(true, .release);
            };
            self.unlockRemember();
        }

        fn incrementalBarrier(self: *Self, cell: ?*anyopaque) void {
            if (!self.marking.load(.acquire)) return;
            const p = cell orelse return;
            const h = self.headerForPayload(p) orelse return;
            self.incrementalBarrierHeader(h);
        }

        fn incrementalBarrierHeader(self: *Self, h: *Header) void {
            if (!self.claimMark(h)) return;
            if (self.concurrent.load(.acquire)) {
                // Hand the greyed cell to the marker thread (it owns mark_stack).
                self.lockBarrier();
                // A parallel collector can abort between the lock-free
                // `concurrent` check above and this critical section. Re-check
                // while holding the hand-off lock so a stale mutator cannot
                // append into `barrier_buf` after abort cleared it.
                if (self.marking.load(.acquire) and self.concurrent.load(.acquire))
                    self.barrier_buf.append(self.aux, h) catch {};
                self.unlockBarrier();
            } else {
                // In a parallel heap, mutators must never append to the
                // marker-private stack. This branch is reachable only from a
                // stale `marking=true` observation during abort/finish, after
                // `concurrent` has been cleared; the claim is harmless and the
                // next cycle re-whitens every cell.
                if (self.parallel) return;
                self.mark_stack.append(self.aux, h) catch {};
            }
        }

        /// Atomically claim a white cell as grey (returns true once per cell).
        /// Under a concurrent mark the claim is a compare-and-set so the marker
        /// and a mutator's `writeBarrier` never both push the same cell; the
        /// single-threaded path is a plain check.
        fn claimMark(self: *Self, h: *Header) bool {
            if (self.parallel or self.concurrent.load(.acquire)) {
                if (!headerFlagSetIfClear(h, header_marked, .acq_rel)) return false;
                _ = @atomicRmw(usize, &self.marked_count, .Add, 1, .monotonic);
                return true;
            }
            if (headerFlagLoad(h, header_marked, .monotonic)) return false;
            headerFlagStore(h, header_marked, true, .monotonic);
            self.marked_count += 1;
            return true;
        }

        inline fn lockBarrier(self: *Self) void {
            while (self.barrier_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
        }
        inline fn unlockBarrier(self: *Self) void {
            self.barrier_lock.store(0, .release);
        }

        inline fn lockRemember(self: *Self) void {
            while (self.remember_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
        }
        inline fn unlockRemember(self: *Self) void {
            self.remember_lock.store(0, .release);
        }

        inline fn lockWeak(self: *Self) void {
            _ = self;
            global_weak_lock.lock();
        }
        inline fn unlockWeak(self: *Self) void {
            _ = self;
            global_weak_lock.unlock();
        }

        inline fn lockAlloc(self: *Self) void {
            var spins: usize = 0;
            while (self.alloc_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) {
                while (self.alloc_lock.load(.monotonic) != 0) : (spins += 1) {
                    // Read-spin while a peer owns the short publication
                    // section instead of issuing a failing CAS on every pass;
                    // yield periodically so many mutators do not starve the
                    // current owner on CPU-saturated heaps.
                    if ((spins & 0x3f) == 0x3f)
                        std.Thread.yield() catch {}
                    else
                        std.atomic.spinLoopHint();
                }
            }
            if (builtin.is_test) self.alloc_lock_acquisitions_for_testing += 1;
        }
        inline fn unlockAlloc(self: *Self) void {
            self.alloc_lock.store(0, .release);
        }

        inline fn publishCollectionPhase(self: *Self, boundary: CollectionPhaseBoundary) void {
            if (comptime @hasDecl(Binding, "collectionPhaseBoundary"))
                Binding.collectionPhaseBoundary(self.ctx, boundary);
        }

        /// Begin an incremental mark: whiten all cells and grey the roots. The
        /// mutator then runs between `markStep`s with the `writeBarrier` active.
        pub fn startMarking(self: *Self) void {
            self.publishCollectionPhase(.full_prepare_begin);
            self.collection_kind = .full;
            _ = self.closeAndFoldPublicationShards();
            var it = self.cellIterator();
            while (it.next()) |h| headerFlagStore(h, header_marked, false, .monotonic);
            self.marked_count = 0;
            self.mark_stack.clearRetainingCapacity();
            // Reserve the mark stack to the live-cell count up front — the grey
            // set can never exceed it (each cell is greyed at most once). A
            // *concurrent* marker then appends into this pre-allocated buffer for
            // the whole cycle without reallocating from `aux`, so its writes never
            // reuse a page the mutator just freed from a side-store (e.g. a
            // growing WeakMap's `weak_entries`) mid-mark — the cross-thread
            // allocator page-reuse ThreadSanitizer flags on the no-GIL concurrent
            // path. Reserved before roots are traced (still at the safepoint) so
            // even the initial root push is covered. Cells born during the cycle
            // are folded in at the world-stopped finish, where a grow is safe.
            self.mark_stack.ensureTotalCapacity(self.aux, self.live_cells) catch {};
            self.lockWeak();
            self.weak_slots.clearRetainingCapacity();
            self.weak_atomic_slots.clearRetainingCapacity();
            // Weak-slot registration can run while a concurrent marker traces
            // object cells and mutators grow their own side storage. Reserve the
            // per-cycle scratch up front, at the safepoint, so `markWeak` never
            // reallocates from `aux` mid-mark and reuses a page concurrently with
            // a mutator's object-backing write (the same allocator-reuse TSan
            // class the mark-stack reservation avoids).
            self.weak_slots.ensureTotalCapacity(self.aux, self.live_cells) catch {};
            self.weak_atomic_slots.ensureTotalCapacity(self.aux, self.live_cells) catch {};
            self.unlockWeak();
            self.addr_index_built = false;
            self.marking.store(true, .release);
            self.publishCollectionPhase(.full_trace_begin);
            var v = Visitor{ .heap = self };
            Binding.traceRoots(self.ctx, &v);
        }

        /// Process up to `budget` grey cells (0 = unbounded). Returns true when
        /// the mark stack is empty (the grey set is drained for now). Because the
        /// barrier keeps shading during mutation, "drained" is not final until
        /// `finishMarking` re-checks under a stop.
        pub fn markStep(self: *Self, budget: usize) bool {
            var v = Visitor{ .heap = self };
            var n: usize = 0;
            while (self.mark_stack.pop()) |h| {
                Binding.trace(payloadOf(h), h.kind, &v);
                n += 1;
                if (budget != 0 and n >= budget) return self.mark_stack.items.len == 0;
            }
            return true;
        }

        /// Finish an incremental mark (stop-the-world tail): re-scan the roots,
        /// drain the grey set, run the ephemeron fixpoint and weak processing,
        /// then sweep. The root re-scan closes the one gap the heap-store
        /// insertion barrier doesn't: a reachable-but-white cell the mutator
        /// moved onto a *root* (an operand stack, a native frame, the microtask
        /// queue) after `startMarking` snapshotted them. Heap→heap moves are
        /// covered by the barrier; root moves are covered here. (For a
        /// stop-the-world `collect()` no mutation happened, so this re-scan only
        /// re-touches already-marked roots — cheap and harmless.)
        pub fn finishMarking(self: *Self) void {
            std.debug.assert(self.marking.load(.acquire));
            var v = Visitor{ .heap = self };
            Binding.traceRoots(self.ctx, &v);
            while (self.mark_stack.pop()) |h| {
                Binding.trace(payloadOf(h), h.kind, &v);
            }
            self.marking.store(false, .release);
            self.publishCollectionPhase(.full_sweep_begin);
            self.sweepPhase(&v);
            self.publishCollectionPhase(.full_sweep_end);
            self.reopenShardedPublication();
            self.runAfterSweep();
            self.publishCollectionPhase(.full_post_sweep_end);
        }

        /// A full stop-the-world cycle: mark from roots, clear dead weak edges,
        /// sweep (finalizing) the white cells. Equivalent to
        /// `startMarking` + drain + `finishMarking`, kept as the default.
        pub fn collect(self: *Self) void {
            self.startMarking();
            _ = self.markStep(0); // drain fully
            self.finishMarking();
        }

        /// Run a full collection and then compact every live cell accepted by
        /// the binding. Destination reservation is failure-atomic: allocation
        /// and the old→new index complete before the first old byte, root, edge,
        /// publication bit, or heap index is changed. Rewrite/commit hooks are
        /// therefore infallible and execute only with a complete plan.
        pub fn collectAndCompact(self: *Self) CompactionResult {
            if (comptime !supports_relocation) return .{ .status = .unsupported };
            self.collect();
            return self.compactLiveCells();
        }

        fn compactLiveCells(self: *Self) CompactionResult {
            if (comptime !supports_relocation) return .{ .status = .unsupported };
            std.debug.assert(!self.marking.load(.acquire));
            std.debug.assert(!self.concurrent.load(.acquire));

            if (self.parallel) self.lockAlloc();
            defer if (self.parallel) self.unlockAlloc();
            const reopen_shards = self.closeAndFoldPublicationShards();
            defer if (reopen_shards) self.reopenShardedPublication();

            var records: std.ArrayListUnmanaged(RelocationRecordType) = .empty;
            defer records.deinit(self.aux);
            var index: std.AutoHashMapUnmanaged(usize, usize) = .empty;
            defer index.deinit(self.aux);
            records.ensureTotalCapacity(self.aux, self.live_cells) catch
                return .{ .status = .out_of_memory };
            const index_capacity = std.math.cast(u32, self.live_cells) orelse
                return .{ .status = .out_of_memory };
            index.ensureTotalCapacity(self.aux, index_capacity) catch
                return .{ .status = .out_of_memory };

            var moved_bytes: usize = 0;
            var cells = self.cellIterator();
            while (cells.next()) |old_header| {
                const old_payload = payloadOf(old_header);
                if (!Binding.canRelocate(self.ctx, old_payload, old_header.kind)) continue;
                const total = header_stride + headerPayloadSize(old_header);
                const new_header = self.reserveRelocationCell(total) orelse {
                    for (records.items) |record|
                        self.releaseRelocationReservation(headerOf(record.new_payload), header_stride + record.size);
                    return .{ .status = .out_of_memory };
                };
                std.debug.assert(new_header != @as(*anyopaque, @ptrCast(old_header)));
                std.debug.assert(@intFromPtr(new_header) % 16 == 0);
                const new_payload = payloadOf(@ptrCast(@alignCast(new_header)));
                const record_index = records.items.len;
                records.appendAssumeCapacity(.{
                    .id = old_header.stable_id,
                    .kind = old_header.kind,
                    .size = headerPayloadSize(old_header),
                    .old_payload = old_payload,
                    .new_payload = new_payload,
                });
                index.putAssumeCapacity(@intFromPtr(old_payload), record_index);
                moved_bytes += total;
            }
            if (records.items.len == 0) return .{ .status = .no_candidates };

            for (records.items) |*record| {
                const total = header_stride + record.size;
                const old_bytes: [*]const u8 = @ptrCast(headerOf(record.old_payload));
                const new_bytes: [*]u8 = @ptrCast(headerOf(record.new_payload));
                @memcpy(new_bytes[0..total], old_bytes[0..total]);
                record.state = .copied;
            }

            const visitor = RelocationVisitorType{ .records = records.items, .index = &index };
            Binding.relocateRoots(self.ctx, &visitor);
            // Pinned cells can still point at moved cells, so rewrite the
            // complete live graph. Moved cells are visited at their copied
            // destination; pinned cells remain at their original address.
            var rewrite_cells = self.cellIterator();
            while (rewrite_cells.next()) |old_header| {
                const current_payload = visitor.resolve(payloadOf(old_header));
                Binding.relocateCell(self.ctx, current_payload, old_header.kind, &visitor);
            }
            for (records.items) |*record| record.state = .rewritten;

            // Repair the generic intrusive list before freeing any old header.
            // Owned-storage bindings enumerate their own publication bitmaps and
            // keep `all == null`, so this loop is empty for that fast path.
            var old_cursor = self.all;
            while (old_cursor) |old_header| {
                const old_next = old_header.next;
                const current_payload = visitor.resolve(payloadOf(old_header));
                const current_header = headerOf(current_payload);
                current_header.next = if (old_next) |next|
                    headerOf(visitor.resolve(payloadOf(next)))
                else
                    null;
                old_cursor = old_next;
            }
            if (self.all) |old_head| self.all = headerOf(visitor.resolve(payloadOf(old_head)));

            for (records.items) |record| {
                const old_header = headerOf(record.old_payload);
                const new_header = headerOf(record.new_payload);
                const total = header_stride + record.size;
                self.unindexPayloadLocked(old_header);
                self.indexPayloadLocked(new_header);
                // The old liveness witness disappears only after roots, edges,
                // and the generic list/index all point at the destination.
                old_header.magic = 0;
                self.commitRelocationCell(old_header, new_header, total);
            }
            if (comptime relocation_verify_hooks == 2) {
                // Forwarding records are still live, while the binding's
                // publication/index metadata already names only committed
                // destinations. Verification must be allocation-free and may
                // trap on any stale old payload retained by the embedder.
                Binding.verifyRelocationRoots(self.ctx, &visitor);
                var verify_cells = self.cellIterator();
                while (verify_cells.next()) |header|
                    Binding.verifyRelocationCell(self.ctx, payloadOf(header), header.kind, &visitor);
            }
            self.weak_slots.clearRetainingCapacity();
            self.weak_atomic_slots.clearRetainingCapacity();
            self.addr_index_built = false;
            return .{
                .status = .compacted,
                .moved_cells = records.items.len,
                .moved_bytes = moved_bytes,
            };
        }

        /// Run a stop-the-world nursery cycle. Old roots are not recursively
        /// rescanned; dirty old containers and conservative child-only barrier
        /// targets supply the old-to-young frontier. Live young cells advance
        /// one age and are tenured only when they reach `tenuring_age`.
        pub fn collectYoung(self: *Self) void {
            std.debug.assert(!self.marking.load(.acquire));
            std.debug.assert(!self.concurrent.load(.acquire));
            if (!self.nursery_enabled) return;
            if (self.young_bytes + self.publicationTotals().young_bytes == 0) return;
            if (self.nursery_force_full.load(.acquire)) {
                self.collect();
                return;
            }

            self.publishCollectionPhase(.minor_prepare_begin);
            if (self.parallel) self.lockAlloc();
            _ = self.closeAndFoldPublicationShards();
            self.collection_kind = .minor;
            const owned_enumeration = self.bindingEnumeratesOwnedCells();
            var old_head: ?*Header = null;
            if (owned_enumeration) {
                var cells = self.cellIterator();
                while (cells.next()) |h| {
                    if (headerFlagLoad(h, header_young, .monotonic))
                        headerFlagStore(h, header_marked, false, .monotonic);
                }
            } else {
                old_head = self.all;
                while (old_head) |h| {
                    if (!headerFlagLoad(h, header_young, .monotonic)) break;
                    headerFlagStore(h, header_marked, false, .monotonic);
                    old_head = h.next;
                }
            }
            self.marked_count = 0;
            self.mark_stack.clearRetainingCapacity();
            self.mark_stack.ensureTotalCapacity(self.aux, self.young_cells) catch {};
            self.lockWeak();
            self.weak_slots.clearRetainingCapacity();
            self.weak_atomic_slots.clearRetainingCapacity();
            // A minor cycle can still trace old roots/remembered owners that
            // register weak slots, so reserve to the full live-cell bound rather
            // than only the young-cell count.
            self.weak_slots.ensureTotalCapacity(self.aux, self.live_cells) catch {};
            self.weak_atomic_slots.ensureTotalCapacity(self.aux, self.live_cells) catch {};
            self.unlockWeak();
            self.addr_index_built = false;
            self.marking.store(true, .release);
            if (self.parallel) self.unlockAlloc();

            self.publishCollectionPhase(.minor_trace_begin);
            var v = Visitor{ .heap = self };
            Binding.traceRoots(self.ctx, &v);
            // Bindings may identify mutable side-cell kinds whose post-creation
            // stores are not fully owner-barriered. Treat those old cells as
            // remembered owners for this cycle. The sweep already walks `all`,
            // so this adds only a kind check per old cell and traces edges solely
            // for the selected kinds.
            if (@hasDecl(Binding, "traceOldOnMinor")) {
                if (owned_enumeration) {
                    var old_it = self.cellIterator();
                    while (old_it.next()) |h| {
                        if (!headerFlagLoad(h, header_young, .monotonic) and Binding.traceOldOnMinor(h.kind))
                            self.rememberOwner(h);
                    }
                } else {
                    var old_it = old_head;
                    while (old_it) |h| : (old_it = h.next) {
                        if (Binding.traceOldOnMinor(h.kind)) self.rememberOwner(h);
                    }
                }
            }
            self.lockRemember();
            for (self.remembered_owners.items) |h| {
                if (h.magic == header_magic and !headerFlagLoad(h, header_young, .monotonic))
                    Binding.trace(payloadOf(h), h.kind, &v);
            }
            for (self.remembered_targets.items) |h| v.mark(payloadOf(h));
            self.unlockRemember();
            while (self.mark_stack.pop()) |h| Binding.trace(payloadOf(h), h.kind, &v);
            self.retainRememberedForMinorSweep();
            self.marking.store(false, .release);
            self.publishCollectionPhase(.minor_sweep_begin);
            self.sweepPhase(&v);
            self.publishCollectionPhase(.minor_sweep_end);
            self.collection_kind = .full;
            self.reopenShardedPublication();
            self.runAfterSweep();
            self.publishCollectionPhase(.minor_post_sweep_end);
        }

        // ---- Concurrent marking (M3) -------------------------------------
        //
        // The marker runs on its own thread while mutators keep executing. The
        // mutator publishes greyed cells (insertion barrier + born-grey allocs)
        // into `barrier_buf`; the marker drains its private `mark_stack` and
        // folds in `barrier_buf` each round. White→grey claims are atomic
        // (`claimMark`), so no cell is double-pushed. Roots are traced at
        // `beginConcurrentMark` while the world is stopped, and re-scanned at
        // `finishConcurrentMark` (also stopped) to catch cells the mutator moved
        // onto a root mid-mark. This is the "make the collector concurrent" step
        // of M3 (docs/threads/P7-gc-design.md); dropping the GIL is the rest.

        /// Begin a concurrent mark. Call with the world stopped (no mutator
        /// running): whitens all cells and greys the roots into `mark_stack`,
        /// then flips `concurrent` on so mutators route through `barrier_buf`.
        pub fn beginConcurrentMark(self: *Self) void {
            self.startMarking(); // whiten + trace roots into mark_stack
            self.barrier_buf.clearRetainingCapacity();
            self.born_concurrent.clearRetainingCapacity();
            self.deferred_trace.clearRetainingCapacity();
            self.concurrent.store(true, .release);
        }

        /// Begin a concurrent mark for the *parallel* (multi-mutator, GIL-free)
        /// model: peer mutators keep allocating and mutating on other threads
        /// while this runs. Unlike `beginConcurrentMark`, the world is NOT
        /// stopped, so:
        ///   - The whiten pass + state reset run under `alloc_lock`, the same
        ///     leaf lock `create` takes, so the O(n) `all`-list walk and the
        ///     born-grey mark-bit set can't race a peer's prepend. `alloc_lock`
        ///     is a leaf (never held across a safepoint or a per-structure
        ///     lock), so guarding the walk with it cannot deadlock a mutator
        ///     that is parked or spinning for an object lock. The whiten *stores*
        ///     to `marked` are atomic because `alloc_lock` does NOT serialize the
        ///     barrier path: a peer still finishing a store from the *previous*
        ///     cycle (it read `marking` true before the prior finish cleared it)
        ///     reaches `claimMark`, whose CAS atomically touches `marked` without
        ///     `alloc_lock`. That CAS is always a benign no-op here — its target
        ///     was reachable-and-marked last cycle, so the strong CAS fails and
        ///     `writeBarrier` returns before mutating any list (a successful CAS
        ///     would mean a swept-garbage target, which the terminal root
        ///     handshake already rules out). Atomic-vs-atomic is race-free; the
        ///     new cycle's happens-before is the `marking`=true release below.
        ///   - `marking`+`concurrent` are published *while still holding*
        ///     `alloc_lock`, so no cell can be prepended between "whitened" and
        ///     "barrier armed": a `create` that wins the lock after us already
        ///     sees `marking` true and is born grey.
        ///   - After arming the barrier, `Binding.traceRoots` greys the
        ///     embedder's roots. The embedder is responsible for making that
        ///     trace touch only roots that are *safe* to read while peers run
        ///     (its global/realm state, parked-peer stacks, the collector's own
        ///     stack) and for layering each *running* peer's own roots in via a
        ///     safepoint handshake (`src/root_handshake.zig`) — a running peer's
        ///     live VM/native stack can't be read by another thread. Peers'
        ///     concurrent stores shade through `barrier_buf` and their
        ///     allocations are born grey, so nothing reachable is missed.
        /// Requires `parallel`.
        pub fn beginConcurrentMarkParallel(self: *Self) void {
            std.debug.assert(self.parallel);
            self.publishCollectionPhase(.full_prepare_begin);
            self.collection_kind = .full;
            self.lockAlloc();
            _ = self.closeAndFoldPublicationShards();
            var it = self.cellIterator();
            // Atomic store: a lagging peer's `claimMark` CAS (barrier path, no
            // `alloc_lock`) can touch the same `marked` byte concurrently. See
            // the whiten note in the doc comment above.
            while (it.next()) |h| headerFlagStore(h, header_marked, false, .monotonic);
            self.marked_count = 0;
            self.mark_stack.clearRetainingCapacity();
            // Pre-reserve the mark stack to the live-cell count (see the note in
            // `startMarking`): the concurrent/parallel marker then appends into
            // it for the whole cycle without reallocating from `aux`, so its
            // writes never reuse a page a mutator just freed from a side-store
            // mid-mark — the cross-thread allocator page-reuse TSan flags.
            self.mark_stack.ensureTotalCapacity(self.aux, self.live_cells) catch {};
            self.lockWeak();
            self.weak_slots.clearRetainingCapacity();
            self.weak_atomic_slots.clearRetainingCapacity();
            // See `startMarking`: weak-slot scratch is marker-side state, but it
            // must not allocate from `aux` while parallel mutators are growing
            // object backing stores.
            self.weak_slots.ensureTotalCapacity(self.aux, self.live_cells) catch {};
            self.weak_atomic_slots.ensureTotalCapacity(self.aux, self.live_cells) catch {};
            self.unlockWeak();
            self.barrier_buf.clearRetainingCapacity();
            self.born_concurrent.clearRetainingCapacity();
            self.deferred_trace.clearRetainingCapacity();
            self.addr_index_built = false;
            self.marking.store(true, .release);
            self.concurrent.store(true, .release);
            self.unlockAlloc();
            // Grey the embedder's roots into `mark_stack` (only the collector
            // pushes there; peers route through `barrier_buf`). Mark claims are
            // atomic now that `concurrent` is set, so this is safe to run while
            // peers allocate.
            self.publishCollectionPhase(.full_trace_begin);
            var v = Visitor{ .heap = self };
            Binding.traceRoots(self.ctx, &v);
        }

        /// One marker-thread round: trace everything currently grey, then fold
        /// in whatever the mutator handed off. Returns true when both the local
        /// stack and the hand-off buffer were empty this round (a quiescent
        /// point — not final until the world is stopped for `finishConcurrentMark`).
        pub fn concurrentMarkRound(self: *Self) bool {
            var v = Visitor{ .heap = self };
            while (self.mark_stack.pop()) |h| {
                Binding.trace(payloadOf(h), h.kind, &v);
            }
            self.lockBarrier();
            const handed = self.barrier_buf.items.len;
            if (handed > 0) {
                self.mark_stack.appendSlice(self.aux, self.barrier_buf.items) catch {
                    v.oom = true;
                };
                self.barrier_buf.clearRetainingCapacity();
            }
            self.unlockBarrier();
            return handed == 0 and self.mark_stack.items.len == 0;
        }

        /// Finish a concurrent mark (call with the world stopped): fold in any
        /// remaining hand-off, re-scan roots, drain, run the ephemeron/weak pass,
        /// and sweep. After this `concurrent`/`marking` are off.
        pub fn finishConcurrentMark(self: *Self) void {
            std.debug.assert(self.marking.load(.acquire) and self.concurrent.load(.acquire));
            self.concurrent.store(false, .release); // world is stopped; claims need not be atomic now
            var v = Visitor{ .heap = self };
            self.mark_stack.appendSlice(self.aux, self.barrier_buf.items) catch {};
            self.barrier_buf.clearRetainingCapacity();
            // Fold in cells born during the cycle: now world-stopped, their
            // payloads are complete, so tracing them is safe. They are already
            // marked; tracing discovers and marks their children.
            self.mark_stack.appendSlice(self.aux, self.born_concurrent.items) catch {};
            self.born_concurrent.clearRetainingCapacity();
            // Cells whose tracing was deferred (mutable storage unsafe to read
            // mid-mark) are traced now — world stopped, storage stable. They are
            // already marked; trace discovers their (possibly white) children.
            for (self.deferred_trace.items) |h| Binding.trace(payloadOf(h), h.kind, &v);
            self.deferred_trace.clearRetainingCapacity();
            Binding.traceRoots(self.ctx, &v); // catch cells moved onto a root mid-mark
            while (self.mark_stack.pop()) |h| {
                Binding.trace(payloadOf(h), h.kind, &v);
            }
            self.marking.store(false, .release);
            self.publishCollectionPhase(.full_sweep_begin);
            self.sweepPhase(&v);
            self.publishCollectionPhase(.full_sweep_end);
            self.reopenShardedPublication();
            self.runAfterSweep();
            self.publishCollectionPhase(.full_post_sweep_end);
        }

        /// Pending mutator-allocated cells not yet folded into the mark (M3
        /// parallel). The driver watches this for *stability* across two
        /// all-published handshake rounds: a stable count means no peer is
        /// mid-allocation, so every born cell's payload is fully initialized and
        /// safe to fold at `finishConcurrentMarkParallel`.
        pub fn bornPendingLen(self: *Self) usize {
            self.lockAlloc();
            defer self.unlockAlloc();
            return self.born_concurrent.items.len;
        }

        /// Cells whose tracing the marker deferred to finish (generators /
        /// iterator helpers whose mutable `exec`/`inner` can't be read while the
        /// owning mutator runs). The parallel driver refuses to finish (aborts)
        /// while this is non-empty, because a *running* peer's deferred cell
        /// can't be traced soundly — only a world-stopped or quiescent finish can.
        pub fn deferredPendingLen(self: *Self) usize {
            return self.deferred_trace.items.len;
        }

        /// Finish a concurrent mark in the PARALLEL model — peers keep running,
        /// no stop-the-world. The caller (the engine's mid-script driver) must
        /// have confirmed via the root handshake that **every peer published the
        /// current generation**, that `born_concurrent` is **stable** (no peer
        /// mid-allocation, so every born payload is initialized), and that
        /// `deferred_trace` is **empty**. Given that, marking has reached closure
        /// over all live roots, and any store a peer makes from here can only
        /// shade a cell reachable from its already-published (and traced) roots —
        /// i.e. an already-marked cell — so it is sound to drain and sweep
        /// without freezing the world. Claims stay atomic until the final
        /// `marking=false`; the sweep runs under `alloc_lock` (`sweepPhase`).
        /// Returns true if it swept, false if it had to bail (a peer allocated
        /// during the finish, so newly-born cells would be untraced and their
        /// un-barriered creation-time references could be missed); on false the
        /// caller aborts (`abortConcurrentMarkParallel`), freeing nothing.
        pub fn finishConcurrentMarkParallel(self: *Self) bool {
            std.debug.assert(self.parallel and self.marking.load(.acquire) and self.concurrent.load(.acquire));
            var v = Visitor{ .heap = self };
            // Fold the born cells (initialized, per the caller's stability
            // guarantee) and any deferred cells under `alloc_lock` so a peer
            // can't append a fresh half-built born cell while we read the list.
            // Tracing these born cells is what catches their creation-time
            // references (e.g. an `Environment.parent`) which the insertion
            // barrier does NOT cover — so their (possibly white) parents survive.
            self.lockAlloc();
            self.mark_stack.appendSlice(self.aux, self.born_concurrent.items) catch {};
            self.born_concurrent.clearRetainingCapacity();
            for (self.deferred_trace.items) |h| self.mark_stack.append(self.aux, h) catch {};
            self.deferred_trace.clearRetainingCapacity();
            self.unlockAlloc();
            // Re-scan the collector-safe roots (the driver leaves `traceRoots` in
            // parallel mode: realm roots locked + the collector's own interpreter;
            // running peers self-published), then drain to closure, folding any
            // late barrier hand-offs. Post-closure hand-offs are necessarily
            // already-marked (see the doc comment), so the loop terminates.
            Binding.traceRoots(self.ctx, &v);
            var passes: usize = 0;
            while (passes < 64) : (passes += 1) {
                while (self.mark_stack.pop()) |h| Binding.trace(payloadOf(h), h.kind, &v);
                self.lockBarrier();
                const handed = self.barrier_buf.items.len;
                if (handed > 0) {
                    self.mark_stack.appendSlice(self.aux, self.barrier_buf.items) catch {};
                    self.barrier_buf.clearRetainingCapacity();
                }
                self.unlockBarrier();
                if (handed == 0 and self.mark_stack.items.len == 0) break;
            }
            // Sweep ONLY if no peer allocated during the trace/drain above — i.e.
            // `born_concurrent` is still empty. A non-empty list means new cells
            // were born whose creation-time references we didn't trace and whose
            // (possibly white) targets could be swept; rather than risk that, bail
            // and let the caller abort (freeing nothing is always sound). Hold
            // `alloc_lock` across the check AND the sweep so no cell can be born
            // between them. The sweep takes no other lock that a peer holds while
            // waiting on `alloc_lock`, so this can't deadlock.
            self.lockAlloc();
            if (self.born_concurrent.items.len != 0) {
                self.unlockAlloc();
                return false;
            }
            // Closure reached, no new allocation. Disarm the barrier (a later
            // peer store now no-ops: its target is reachable-from-roots → already
            // marked, or genuinely unreachable → correctly white), then sweep
            // while still holding `alloc_lock` (so the all-list is stable).
            self.marking.store(false, .release);
            self.concurrent.store(false, .release);
            self.publishCollectionPhase(.full_sweep_begin);
            self.sweepPhaseLocked(&v); // alloc_lock already held
            self.publishCollectionPhase(.full_sweep_end);
            self.reopenShardedPublication();
            self.unlockAlloc();
            self.runAfterSweep();
            self.publishCollectionPhase(.full_post_sweep_end);
            return true;
        }

        /// Whether live bytes have crossed the collection threshold, read under
        /// `alloc_lock` in `parallel` mode so a mid-script collector's safepoint
        /// check doesn't race a peer's `create` updating `bytes_live`.
        pub fn shouldCollect(self: *Self) bool {
            if (self.parallel) self.lockAlloc();
            defer if (self.parallel) self.unlockAlloc();
            const pending = self.publicationTotals();
            return self.bytes_live + pending.live_bytes >= self.threshold_bytes;
        }

        /// Whether tenured bytes alone have crossed the full-heap threshold.
        /// Generational embedders use this at quiescent boundaries so a large
        /// young batch receives a minor collection before it can force a full
        /// trace. Mid-script collectors that cannot run minor GC should continue
        /// using `shouldCollect()` over total bytes.
        pub fn shouldCollectOld(self: *Self) bool {
            if (self.parallel) self.lockAlloc();
            defer if (self.parallel) self.unlockAlloc();
            const pending = self.publicationTotals();
            return (self.bytes_live + pending.live_bytes) - (self.young_bytes + pending.young_bytes) >= self.threshold_bytes;
        }

        /// Whether the nursery has reached its collection threshold, or a
        /// remembered-set allocation failure requires the next nursery request
        /// to fall back to a full collection.
        pub fn shouldCollectYoung(self: *Self) bool {
            if (!self.nursery_enabled) return false;
            if (self.nursery_force_full.load(.acquire)) return true;
            if (self.parallel) self.lockAlloc();
            defer if (self.parallel) self.unlockAlloc();
            const pending = self.publicationTotals();
            return self.young_bytes + pending.young_bytes >= self.nursery_threshold_bytes;
        }

        /// Abort an in-progress parallel concurrent mark WITHOUT sweeping: clear
        /// the marking state and the scratch buffers, freeing nothing. Sound at
        /// any time — an aborted mark just leaves some cells marked (re-whitened
        /// by the next `beginConcurrentMarkParallel` / `startMarking`). The driver
        /// calls this when it can't reach a stable finish within its round budget
        /// (heavy continuous allocation, or a deferred generator), falling back to
        /// the next quiescent `collect`.
        pub fn abortConcurrentMarkParallel(self: *Self) void {
            self.marking.store(false, .release);
            self.concurrent.store(false, .release);
            self.lockBarrier();
            self.barrier_buf.clearRetainingCapacity();
            self.unlockBarrier();
            self.lockAlloc();
            self.born_concurrent.clearRetainingCapacity();
            self.unlockAlloc();
            self.mark_stack.clearRetainingCapacity();
            self.deferred_trace.clearRetainingCapacity();
            self.reopenShardedPublication();
        }

        /// The ephemeron-fixpoint + weak-edge + sweep tail shared by the
        /// stop-the-world and incremental paths. `v` is a live Visitor over self.
        ///
        /// Under `parallel` the whole tail runs holding `alloc_lock`: the sweep
        /// free-loop unlinks from `all` and adjusts the live/bytes counters, and
        /// the ephemeron/weak walks read `all` + mark bits — all of which a peer
        /// mutator's `create` touches under the same lock. The collector reaches
        /// this only after the embedder's terminal root handshake, so peer
        /// mutators are at safepoints and not tracing the cells being read.
        /// Finalizers run here must not allocate from this heap (they would
        /// re-enter `lockAlloc` — a non-reentrant leaf spinlock); this engine's
        /// finalizers only release native side storage, never allocate cells.
        fn sweepPhase(self: *Self, v: *Visitor) void {
            {
                if (self.parallel) self.lockAlloc();
                defer if (self.parallel) self.unlockAlloc();
                self.sweepPhaseLocked(v);
            }
        }

        /// Let the embedder drain work queued by cell finalizers after the
        /// collector has completed sweep accounting and released alloc_lock.
        /// The hook is optional and is invoked exactly once per successful
        /// full, nursery, or concurrent sweep; aborted cycles do not invoke it.
        fn runAfterSweep(self: *Self) void {
            if (@hasDecl(Binding, "afterSweep")) Binding.afterSweep(self.ctx);
        }

        /// Retire one address-local, same-size run supplied by an authoritative
        /// owned-cell iterator. Classification is withdrawn for the complete
        /// run before the first finalizer executes; storage reuse remains after
        /// every finalizer, payload-index removal, and header retirement.
        fn sweepDeadOwnedRun(
            self: *Self,
            total: usize,
            allocations: []*anyopaque,
            reclaimed_cells: *usize,
            reclaimed_bytes: *usize,
            reclaimed_young_bytes: *usize,
        ) void {
            self.bindingUnpublishCellAllocationBatch(total, allocations);
            for (allocations) |allocation| {
                const h: *Header = @ptrCast(@alignCast(allocation));
                const young = headerFlagLoad(h, header_young, .monotonic);
                Binding.finalize(self.ctx, payloadOf(h), h.kind);
                self.unindexPayloadLocked(h);
                h.magic = 0;
                reclaimed_cells.* += 1;
                reclaimed_bytes.* += total;
                if (young) reclaimed_young_bytes.* += total;
            }
            if (@hasDecl(Binding, "freeCellStorageBatch")) {
                Binding.freeCellStorageBatch(self.ctx, total, allocations);
            } else {
                for (allocations) |allocation| {
                    const base: [*]align(16) u8 = @ptrCast(@alignCast(allocation));
                    self.backing.free(base[0..total]);
                }
            }
        }

        /// The sweep tail proper, assuming `alloc_lock` is already held under
        /// `parallel` (the parallel finish holds it across its born-empty check
        /// and the sweep so no cell is born in between).
        fn sweepPhaseLocked(self: *Self, v: *Visitor) void {
            // Full marking does not consume the generational frontier. Drop its
            // pre-mark snapshot before freeing anything: unlike minor GC, a full
            // sweep may reclaim remembered owners/targets, so clearing their bits
            // afterward would dereference dead headers. Parallel mutators can add
            // fresh entries after this point; closure guarantees those cells are
            // marked, and the final clear below safely discards those late cards.
            if (self.collection_kind == .full) self.clearRemembered();

            // A binding can conservatively prove that it has never published
            // weak semantic state. In that common case all three passes below
            // are empty, and avoiding their all-list walks is material for
            // allocation-heavy nursery cycles. Bindings without the hook keep
            // the original unconditional behavior.
            const has_weak_work = if (@hasDecl(Binding, "hasWeakWork")) Binding.hasWeakWork(self.ctx) else true;
            if (has_weak_work) {
                // 2b. Ephemerons (WeakMap-style edges): if a marked table has a
                // marked key, its value becomes strong. Iterate to a fixed point
                // so values can keep further keys alive through weak-map chains.
                if (@hasDecl(Binding, "traceEphemeron")) {
                    while (true) {
                        const before = self.marked_count;
                        var eit = self.cellIterator();
                        while (eit.next()) |h| {
                            if (self.shouldProcessMarkedCell(h)) Binding.traceEphemeron(self.ctx, payloadOf(h), h.kind, v);
                        }
                        while (self.mark_stack.pop()) |h| {
                            Binding.trace(payloadOf(h), h.kind, v);
                        }
                        if (self.marked_count == before) break;
                    }
                }

                // 3. weak edges whose target died are cleared *before* the
                // sweep frees it, so no slot ever dangles.
                self.lockWeak();
                for (self.weak_slots.items) |slot| {
                    if (slot.*) |target| {
                        if (!self.isLive(target)) slot.* = null;
                    }
                }
                for (self.weak_atomic_slots.items) |slot| {
                    if (slot.load(.acquire)) |target| {
                        if (!self.isLive(target))
                            _ = slot.cmpxchgStrong(target, null, .acq_rel, .acquire);
                    }
                }
                self.unlockWeak();

                // External weak owners may need to publish host cleanup after
                // an atomic slot is cleared. Run that hook outside `weak_lock`:
                // host finalizers can release handles or otherwise re-enter the
                // embedder and must never execute under collector scratch locks.
                if (@hasDecl(Binding, "afterWeakRoots")) Binding.afterWeakRoots(self.ctx);

                if (@hasDecl(Binding, "afterWeak")) {
                    var wit = self.cellIterator();
                    while (wit.next()) |h| {
                        if (self.shouldProcessMarkedCell(h)) Binding.afterWeak(self.ctx, payloadOf(h), h.kind);
                    }
                }
            }

            // 4. sweep the white cells.
            const minor = self.collection_kind == .minor;
            var cycle_young_bytes: usize = 0;
            var cycle_reclaimed_young_bytes: usize = 0;
            var cycle_reclaimed_cells: usize = 0;
            var cycle_reclaimed_bytes: usize = 0;
            var cycle_survived_cells: usize = 0;
            var cycle_survived_bytes: usize = 0;
            var cycle_retained_young_cells: usize = 0;
            var cycle_retained_young_bytes: usize = 0;
            var cycle_promoted_cells: usize = 0;
            var cycle_promoted_bytes: usize = 0;
            const release_batch_capacity = 64;
            var release_batch: [release_batch_capacity]*anyopaque = undefined;
            var release_batch_len: usize = 0;
            var release_batch_bytes: usize = 0;
            // The largest default owned slab contains 1,024 64-byte cells.
            // This fixed scratch admits a complete chunk without collector
            // allocation; bindings decide whether a run is actually eligible
            // for whole-run reclamation.
            const owned_run_capacity = 1024;
            var owned_run: [owned_run_capacity]*anyopaque = undefined;
            var owned_run_len: usize = 0;
            var owned_run_bytes: usize = 0;
            var prev: ?*Header = null;
            const owned_enumeration = self.bindingEnumeratesOwnedCells();
            const batch_owned_runs = owned_enumeration and @hasDecl(Binding, "unpublishCellAllocationBatch");
            var cells = self.cellIterator();
            while (cells.next()) |h| {
                const young = headerFlagLoad(h, header_young, .monotonic);
                if (minor and !young) {
                    if (owned_run_len != 0) {
                        self.sweepDeadOwnedRun(owned_run_bytes, owned_run[0..owned_run_len], &cycle_reclaimed_cells, &cycle_reclaimed_bytes, &cycle_reclaimed_young_bytes);
                        owned_run_len = 0;
                    }
                    if (owned_enumeration) continue;
                    break;
                }
                const next = h.next;
                const total = header_stride + headerPayloadSize(h);
                if (minor and young) cycle_young_bytes += total;
                if (headerFlagLoad(h, header_marked, .monotonic)) {
                    if (owned_run_len != 0) {
                        self.sweepDeadOwnedRun(owned_run_bytes, owned_run[0..owned_run_len], &cycle_reclaimed_cells, &cycle_reclaimed_bytes, &cycle_reclaimed_young_bytes);
                        owned_run_len = 0;
                    }
                    if (young) {
                        cycle_survived_cells += 1;
                        cycle_survived_bytes += total;
                        const next_age = h.age + 1;
                        if (!minor or next_age >= self.tenuring_age) {
                            headerFlagStore(h, header_young, false, .release);
                            h.age = tenured_age;
                            cycle_promoted_cells += 1;
                            cycle_promoted_bytes += total;
                        } else {
                            h.age = next_age;
                            cycle_retained_young_cells += 1;
                            cycle_retained_young_bytes += total;
                        }
                    }
                    prev = h;
                } else if (batch_owned_runs) {
                    if (owned_run_len != 0 and owned_run_bytes != total) {
                        self.sweepDeadOwnedRun(owned_run_bytes, owned_run[0..owned_run_len], &cycle_reclaimed_cells, &cycle_reclaimed_bytes, &cycle_reclaimed_young_bytes);
                        owned_run_len = 0;
                    }
                    owned_run_bytes = total;
                    owned_run[owned_run_len] = h;
                    owned_run_len += 1;
                    if (owned_run_len == owned_run.len) {
                        self.sweepDeadOwnedRun(owned_run_bytes, &owned_run, &cycle_reclaimed_cells, &cycle_reclaimed_bytes, &cycle_reclaimed_young_bytes);
                        owned_run_len = 0;
                    }
                } else {
                    // Stop owned classifiers from returning this slot before
                    // finalization clears side state or its header is reused.
                    self.bindingUnpublishCellAllocation(h, total);
                    Binding.finalize(self.ctx, payloadOf(h), h.kind);
                    self.unindexPayloadLocked(h);
                    // Slab ownership outlives a freed slot. Clear the live-cell
                    // witness before returning storage so a stale payload cannot
                    // pass the optional ownership hook.
                    h.magic = 0;
                    if (!owned_enumeration) {
                        if (prev) |p| p.next = next else self.all = next;
                    }
                    cycle_reclaimed_cells += 1;
                    cycle_reclaimed_bytes += total;
                    if (young) {
                        cycle_reclaimed_young_bytes += total;
                    }
                    if (@hasDecl(Binding, "freeCellStorageBatch")) {
                        if (release_batch_len != 0 and release_batch_bytes != total) {
                            Binding.freeCellStorageBatch(self.ctx, release_batch_bytes, release_batch[0..release_batch_len]);
                            release_batch_len = 0;
                        }
                        release_batch_bytes = total;
                        release_batch[release_batch_len] = h;
                        release_batch_len += 1;
                        if (release_batch_len == release_batch.len) {
                            Binding.freeCellStorageBatch(self.ctx, release_batch_bytes, &release_batch);
                            release_batch_len = 0;
                        }
                    } else {
                        const base: [*]align(16) u8 = @ptrCast(@alignCast(h));
                        self.backing.free(base[0..total]);
                    }
                }
            }
            if (owned_run_len != 0)
                self.sweepDeadOwnedRun(owned_run_bytes, owned_run[0..owned_run_len], &cycle_reclaimed_cells, &cycle_reclaimed_bytes, &cycle_reclaimed_young_bytes);
            if (@hasDecl(Binding, "freeCellStorageBatch") and release_batch_len != 0)
                Binding.freeCellStorageBatch(self.ctx, release_batch_bytes, release_batch[0..release_batch_len]);

            // Binding calls inside the sweep can otherwise force these hot
            // heap fields to be reloaded and stored for every dead cell. The
            // list/header mutations remain immediate; publish their exact
            // aggregate accounting after every finalizer and backing release
            // has completed.
            std.debug.assert(self.live_cells >= cycle_reclaimed_cells);
            std.debug.assert(self.bytes_live >= cycle_reclaimed_bytes);
            self.live_cells -= cycle_reclaimed_cells;
            self.bytes_live -= cycle_reclaimed_bytes;
            // Every collection kind traverses the complete young prefix. A
            // minor retains sub-threshold survivors in that prefix; a full
            // explicitly tenures every survivor and resets the frontier.
            self.young_cells = if (minor) cycle_retained_young_cells else 0;
            self.young_bytes = if (minor) cycle_retained_young_bytes else 0;

            self.collections += 1;
            self.promoted_cells += cycle_promoted_cells;
            self.promoted_bytes += cycle_promoted_bytes;
            if (minor) {
                self.minor_collections += 1;
                self.last_minor_young_bytes = cycle_young_bytes;
                self.last_minor_reclaimed_bytes = cycle_reclaimed_young_bytes;
                self.last_minor_survived_cells = cycle_survived_cells;
                self.last_minor_survived_bytes = cycle_survived_bytes;
                self.last_minor_promoted_bytes = cycle_promoted_bytes;
                self.total_minor_young_bytes +|= cycle_young_bytes;
                self.total_minor_reclaimed_bytes +|= cycle_reclaimed_young_bytes;
                self.total_minor_survived_bytes +|= cycle_survived_bytes;
                self.total_minor_promoted_bytes +|= cycle_promoted_bytes;
                self.nursery_threshold_bytes = self.nextNurseryThreshold(cycle_young_bytes, cycle_survived_bytes);
            } else {
                self.full_collections += 1;
                self.last_full_collection_bytes = self.bytes_live;
                self.threshold_bytes = @max(64 * 1024, self.bytes_live * 2);
            }
            if (!minor) self.clearRemembered();
            self.nursery_force_full.store(false, .release);
            self.trimCollectorScratch();
        }

        fn shouldProcessMarkedCell(self: *Self, h: *Header) bool {
            if (self.collection_kind == .full) return headerFlagLoad(h, header_marked, .monotonic);
            if (headerFlagLoad(h, header_young, .monotonic)) return headerFlagLoad(h, header_marked, .monotonic);
            return headerFlagLoad(h, header_remembered_owner, .monotonic);
        }

        fn clearRemembered(self: *Self) void {
            self.lockRemember();
            for (self.remembered_owners.items) |h| headerFlagStore(h, header_remembered_owner, false, .release);
            for (self.remembered_targets.items) |h| headerFlagStore(h, header_remembered_target, false, .release);
            self.remembered_owners.clearRetainingCapacity();
            self.remembered_targets.clearRetainingCapacity();
            self.unlockRemember();
        }

        /// Keep old-container cards across repeated minor collections. Without
        /// this, a multi-age child reachable only through an unchanged old edge
        /// would disappear on its second cycle. Conservative target-only cards
        /// retain only targets proven live in this cycle, so sweep never leaves
        /// a dangling header in the card list.
        fn retainRememberedForMinorSweep(self: *Self) void {
            self.lockRemember();
            defer self.unlockRemember();
            var retained: usize = 0;
            for (self.remembered_targets.items) |h| {
                const live_young = h.magic == header_magic and
                    headerFlagLoad(h, header_young, .monotonic) and
                    headerFlagLoad(h, header_marked, .monotonic);
                if (live_young) {
                    self.remembered_targets.items[retained] = h;
                    retained += 1;
                } else if (h.magic == header_magic) {
                    headerFlagStore(h, header_remembered_target, false, .release);
                }
            }
            self.remembered_targets.items.len = retained;
        }

        fn retainedScratchEntryLimit(self: *Self) usize {
            const max = std.math.maxInt(usize);
            const live_bound = if (self.live_cells > max / 2) max else self.live_cells * 2;
            return @max(min_retained_scratch_entries, live_bound);
        }

        fn trimEmptyScratchList(self: *Self, comptime T: type, list: *std.ArrayListUnmanaged(T)) void {
            if (list.items.len != 0) return;
            if (list.capacity <= self.retainedScratchEntryLimit()) return;
            list.clearAndFree(self.aux);
        }

        fn trimCollectorScratch(self: *Self) void {
            self.trimEmptyScratchList(*Header, &self.mark_stack);
            self.lockWeak();
            self.weak_slots.clearRetainingCapacity();
            self.weak_atomic_slots.clearRetainingCapacity();
            self.trimEmptyScratchList(*?*anyopaque, &self.weak_slots);
            self.trimEmptyScratchList(*std.atomic.Value(?*anyopaque), &self.weak_atomic_slots);
            self.unlockWeak();
            self.lockRemember();
            self.trimEmptyScratchList(*Header, &self.remembered_owners);
            self.trimEmptyScratchList(*Header, &self.remembered_targets);
            self.unlockRemember();
            self.trimEmptyScratchList(*Header, &self.barrier_buf);
            self.trimEmptyScratchList(*Header, &self.born_concurrent);
            self.trimEmptyScratchList(*Header, &self.deferred_trace);
        }

        fn deinitImpl(self: *Self, free_cell_storage: bool) void {
            _ = self.closeAndFoldPublicationShards();
            var cells = self.cellIterator();
            while (cells.next()) |h| {
                self.bindingUnpublishCellAllocation(h, header_stride + headerPayloadSize(h));
                Binding.finalize(self.ctx, payloadOf(h), h.kind);
                if (free_cell_storage) {
                    const total = header_stride + headerPayloadSize(h);
                    const base: [*]align(16) u8 = @ptrCast(@alignCast(h));
                    self.backing.free(base[0..total]);
                }
            }
            self.all = null;
            self.payload_index.deinit(self.backing);
            self.payload_index = .empty;
            self.live_cells = 0;
            self.bytes_live = 0;
            self.last_full_collection_bytes = 0;
            self.young_cells = 0;
            self.young_bytes = 0;
            self.mark_stack.deinit(self.aux);
            self.weak_slots.deinit(self.aux);
            self.weak_atomic_slots.deinit(self.aux);
            self.addr_index.deinit(self.aux);
            self.barrier_buf.deinit(self.aux);
            self.born_concurrent.deinit(self.aux);
            self.deferred_trace.deinit(self.aux);
            self.remembered_owners.deinit(self.aux);
            self.remembered_targets.deinit(self.aux);
        }

        /// Free every remaining cell (finalizing each) and the internal lists.
        /// The embedder calls this at context teardown — equivalent to the old
        /// arena `deinit`, but finalizers run.
        pub fn deinit(self: *Self) void {
            self.deinitImpl(true);
        }

        /// Finalize every remaining cell and release collector side buffers, but
        /// do not return individual cell allocations to `backing`. Use only when
        /// the embedder owns those allocations through a slab/arena that it will
        /// reclaim wholesale immediately afterward. Cell finalizers still run in
        /// full, so side storage and host resources are released normally.
        pub fn deinitRetainingCellStorage(self: *Self) void {
            self.deinitImpl(false);
        }
    };
}

// ---------------------------------------------------------------------------
// Tests — a toy runtime binding exercising cycles, garbage, weak edges, and
// finalizers, asserting *exact* reclamation (the collector is precise).
// ---------------------------------------------------------------------------

test "relocation visitor resolves moved cells and preserves pinned cells" {
    const Kind = enum { object, string };
    const Record = RelocationRecord(Kind);
    const Visitor = RelocationVisitor(Kind);
    var old_object: u64 = 1;
    var new_object: u64 = 2;
    var pinned_string: u64 = 3;
    const records = [_]Record{.{
        .id = .init(17),
        .kind = .object,
        .size = @sizeOf(u64),
        .old_payload = &old_object,
        .new_payload = &new_object,
    }};
    var index: std.AutoHashMapUnmanaged(usize, usize) = .empty;
    defer index.deinit(std.testing.allocator);
    try index.put(std.testing.allocator, @intFromPtr(&old_object), 0);
    const visitor = Visitor{ .records = &records, .index = &index };

    try std.testing.expectEqual(@as(*anyopaque, &new_object), visitor.resolve(&old_object));
    try std.testing.expectEqual(@as(*anyopaque, &pinned_string), visitor.resolve(&pinned_string));
    try std.testing.expect(visitor.moved(&old_object));
    try std.testing.expect(!visitor.moved(&pinned_string));
    try std.testing.expectEqual(@as(u64, 17), @intFromEnum(visitor.stableId(&old_object).?));
    try std.testing.expectEqual(@as(?StableCellId, null), visitor.stableId(&pinned_string));
}

const TestRT = struct {
    pub const Kind = enum { node };

    const Node = struct {
        strong: ?*Node = null,
        weak: ?*anyopaque = null,
        id: u32 = 0,
    };

    roots: std.ArrayListUnmanaged(*Node) = .empty,
    finalized: std.ArrayListUnmanaged(u32) = .empty,
    conservative_words: []const usize = &.{},
    atomic_weak_root: ?*std.atomic.Value(?*anyopaque) = null,
    after_weak_roots_calls: usize = 0,
    after_sweep_calls: usize = 0,
    finalized_at_last_after_sweep: usize = 0,
    alloc_lock_probe: ?*std.atomic.Value(u32) = null,
    after_sweep_under_alloc_lock: bool = false,
    publication_gate_probe: ?*std.atomic.Value(bool) = null,
    after_sweep_before_publication_reopened: bool = false,
    collection_phase_boundaries: [64]CollectionPhaseBoundary = undefined,
    collection_phase_boundary_len: usize = 0,
    phase_boundary_len_at_after_sweep: usize = 0,
    relocation_reserve_limit: usize = std.math.maxInt(usize),
    relocation_reserve_calls: usize = 0,
    relocation_rollbacks: usize = 0,
    relocation_commits: usize = 0,
    relocation_root_verifications: usize = 0,
    relocation_cell_verifications: usize = 0,

    pub fn traceRoots(self: *TestRT, v: anytype) void {
        for (self.roots.items) |n| v.mark(n);
        for (self.conservative_words) |word| v.markConservativeWord(word);
        if (self.atomic_weak_root) |slot| v.markWeakAtomic(slot);
    }

    pub fn afterWeakRoots(self: *TestRT) void {
        self.after_weak_roots_calls += 1;
    }

    pub fn afterSweep(self: *TestRT) void {
        self.after_sweep_calls += 1;
        self.phase_boundary_len_at_after_sweep = self.collection_phase_boundary_len;
        self.finalized_at_last_after_sweep = self.finalized.items.len;
        if (self.alloc_lock_probe) |lock|
            self.after_sweep_under_alloc_lock = lock.load(.monotonic) != 0;
        if (self.publication_gate_probe) |gate|
            self.after_sweep_before_publication_reopened = !gate.load(.acquire);
    }

    pub fn collectionPhaseBoundary(self: *TestRT, boundary: CollectionPhaseBoundary) void {
        if (self.collection_phase_boundary_len < self.collection_phase_boundaries.len) {
            self.collection_phase_boundaries[self.collection_phase_boundary_len] = boundary;
            self.collection_phase_boundary_len += 1;
        }
    }

    pub fn trace(cell: *anyopaque, kind: Kind, v: anytype) void {
        switch (kind) {
            .node => {
                const n: *Node = @ptrCast(@alignCast(cell));
                // Under a concurrent mark the mutator may store into `strong`
                // while we read it (it then fires the barrier, so the new target
                // is marked regardless). The read is `.acquire` and the mutator's
                // publishing store `.release`, so when the marker first reaches a
                // freshly built target through this slot it also observes that
                // target's field initialization (the release/acquire edge stands
                // in for the per-structure lock the real engine binding takes
                // around slot/element reads under a concurrent mark).
                const s = if (v.concurrent()) @atomicLoad(?*Node, &n.strong, .acquire) else n.strong;
                v.mark(s);
                v.markWeak(&n.weak);
            },
        }
    }

    pub fn canRelocate(_: *TestRT, cell: *anyopaque, kind: Kind) bool {
        return switch (kind) {
            .node => @as(*Node, @ptrCast(@alignCast(cell))).id != 99,
        };
    }

    pub fn relocateRoots(self: *TestRT, v: anytype) void {
        for (self.roots.items) |*root|
            root.* = @ptrCast(@alignCast(v.resolve(root.*)));
    }

    pub fn relocateCell(_: *TestRT, cell: *anyopaque, kind: Kind, v: anytype) void {
        switch (kind) {
            .node => {
                const n: *Node = @ptrCast(@alignCast(cell));
                if (n.strong) |strong| n.strong = @ptrCast(@alignCast(v.resolve(strong)));
                if (n.weak) |weak| n.weak = v.resolve(weak);
            },
        }
    }

    pub fn verifyRelocationRoots(self: *TestRT, v: anytype) void {
        self.relocation_root_verifications += 1;
        for (self.roots.items) |root|
            std.debug.assert(!v.moved(root));
    }

    pub fn verifyRelocationCell(self: *TestRT, cell: *anyopaque, kind: Kind, v: anytype) void {
        self.relocation_cell_verifications += 1;
        std.debug.assert(!v.moved(cell));
        switch (kind) {
            .node => {
                const node: *Node = @ptrCast(@alignCast(cell));
                if (node.strong) |strong| std.debug.assert(!v.moved(strong));
                if (node.weak) |weak| std.debug.assert(!v.moved(weak));
            },
        }
    }

    pub fn reserveRelocationCell(self: *TestRT, total: usize) ?*anyopaque {
        if (self.relocation_reserve_calls == self.relocation_reserve_limit) return null;
        self.relocation_reserve_calls += 1;
        const slab = std.testing.allocator.alignedAlloc(u8, .@"16", total) catch return null;
        return slab.ptr;
    }

    pub fn releaseRelocationReservation(self: *TestRT, allocation: *anyopaque, total: usize) void {
        self.relocation_rollbacks += 1;
        const base: [*]align(16) u8 = @ptrCast(@alignCast(allocation));
        std.testing.allocator.free(base[0..total]);
    }

    pub fn commitRelocationCell(self: *TestRT, old: *anyopaque, _: *anyopaque, total: usize) void {
        self.relocation_commits += 1;
        const base: [*]align(16) u8 = @ptrCast(@alignCast(old));
        std.testing.allocator.free(base[0..total]);
    }

    pub fn finalize(self: *TestRT, cell: *anyopaque, kind: Kind) void {
        switch (kind) {
            .node => {
                const n: *Node = @ptrCast(@alignCast(cell));
                self.finalized.append(std.testing.allocator, n.id) catch {};
            },
        }
    }
};

const BatchAllocTestRT = struct {
    pub const Kind = enum { node };
    const Node = struct { id: u32 = 0 };

    allocator: std.mem.Allocator,
    limit: usize = std.math.maxInt(usize),
    batch_calls: usize = 0,
    publish_batch_calls: usize = 0,
    published_cells: usize = 0,
    alloc_lock_probe: ?*std.atomic.Value(u32) = null,
    published_while_alloc_locked: bool = false,

    pub fn allocateCellBatch(self: *BatchAllocTestRT, total: usize, out: []*anyopaque) usize {
        self.batch_calls += 1;
        const wanted = @min(self.limit, out.len);
        var allocated: usize = 0;
        while (allocated < wanted) : (allocated += 1) {
            const slab = self.allocator.alignedAlloc(u8, .@"16", total) catch break;
            out[allocated] = @ptrCast(slab.ptr);
        }
        return allocated;
    }

    pub fn publishCellAllocationBatch(self: *BatchAllocTestRT, payloads: []*anyopaque, _: usize, payload_offset: usize) void {
        self.publish_batch_calls += 1;
        self.published_cells += payloads.len;
        if (self.alloc_lock_probe) |lock|
            self.published_while_alloc_locked = lock.load(.monotonic) != 0;
        for (payloads) |payload| std.debug.assert(@intFromPtr(payload) > payload_offset);
    }

    pub fn allCellsUseOwnedStorage(_: *BatchAllocTestRT) bool {
        return true;
    }

    pub fn traceRoots(_: *BatchAllocTestRT, _: anytype) void {}
    pub fn trace(_: *anyopaque, _: Kind, _: anytype) void {}
    pub fn finalize(_: *BatchAllocTestRT, _: *anyopaque, _: Kind) void {}
};

const ShardedBatchTestRT = struct {
    pub const Kind = enum { node };
    const Node = struct { id: usize = 0 };
    const max_cells = 4096;

    published: [max_cells]std.atomic.Value(?*anyopaque) = @splat(.init(null)),
    next_published: std.atomic.Value(usize) = .init(0),
    pause_allocate: std.atomic.Value(bool) = .init(false),
    allocate_ready: std.atomic.Value(bool) = .init(false),
    release_allocate: std.atomic.Value(bool) = .init(false),
    pause_publish: std.atomic.Value(bool) = .init(false),
    publish_ready: std.atomic.Value(bool) = .init(false),
    release_publish: std.atomic.Value(bool) = .init(false),

    pub const Iterator = struct {
        rt: *ShardedBatchTestRT,
        index: usize = 0,

        pub fn next(self: *Iterator) ?*anyopaque {
            while (self.index < self.rt.published.len) {
                const index = self.index;
                self.index += 1;
                if (self.rt.published[index].load(.acquire)) |allocation| return allocation;
            }
            return null;
        }
    };

    pub fn ownedCellIterator(self: *ShardedBatchTestRT) Iterator {
        return .{ .rt = self };
    }
    pub fn allCellsUseOwnedStorage(_: *ShardedBatchTestRT) bool {
        return true;
    }
    pub fn usesOwnedCellStorage(_: *ShardedBatchTestRT, _: usize) bool {
        return true;
    }
    pub fn allocateCellBatch(self: *ShardedBatchTestRT, total: usize, out: []*anyopaque) usize {
        var allocated: usize = 0;
        while (allocated < out.len) : (allocated += 1) {
            const slab = std.heap.page_allocator.alignedAlloc(u8, .@"16", total) catch break;
            out[allocated] = @ptrCast(slab.ptr);
        }
        if (self.pause_allocate.load(.acquire)) {
            self.allocate_ready.store(true, .release);
            while (!self.release_allocate.load(.acquire)) std.atomic.spinLoopHint();
        }
        return allocated;
    }
    pub fn publishCellAllocation(self: *ShardedBatchTestRT, allocation: *anyopaque, _: usize) void {
        const index = self.next_published.fetchAdd(1, .monotonic);
        std.debug.assert(index < self.published.len);
        self.published[index].store(allocation, .release);
    }
    pub fn publishCellAllocationBatch(self: *ShardedBatchTestRT, payloads: []*anyopaque, _: usize, payload_offset: usize) void {
        if (self.pause_publish.load(.acquire)) {
            self.publish_ready.store(true, .release);
            while (!self.release_publish.load(.acquire)) std.atomic.spinLoopHint();
        }
        const start = self.next_published.fetchAdd(payloads.len, .monotonic);
        std.debug.assert(start + payloads.len <= self.published.len);
        for (payloads, 0..) |payload, i| {
            const allocation: *anyopaque = @ptrFromInt(@intFromPtr(payload) - payload_offset);
            self.published[start + i].store(allocation, .release);
        }
    }
    pub fn unpublishCellAllocation(self: *ShardedBatchTestRT, allocation: *anyopaque, _: usize) void {
        for (&self.published) |*slot| {
            if (slot.load(.acquire) == allocation) {
                if (slot.cmpxchgStrong(allocation, null, .acq_rel, .acquire) == null) return;
            }
        }
        unreachable;
    }
    pub fn ownsCellAllocation(self: *ShardedBatchTestRT, allocation: *anyopaque) bool {
        for (&self.published) |*slot| if (slot.load(.acquire) == allocation) return true;
        return false;
    }
    pub fn freeCellStorageBatch(_: *ShardedBatchTestRT, total: usize, allocations: []*anyopaque) void {
        for (allocations) |allocation| {
            const base: [*]align(16) u8 = @ptrCast(@alignCast(allocation));
            std.heap.page_allocator.free(base[0..total]);
        }
    }
    pub fn traceRoots(_: *ShardedBatchTestRT, _: anytype) void {}
    pub fn trace(_: *anyopaque, _: Kind, _: anytype) void {}
    pub fn finalize(_: *ShardedBatchTestRT, _: *anyopaque, _: Kind) void {}
};

const SweepBatchTestRT = struct {
    pub const Kind = enum { node, large_node };
    const Node = struct { value: usize = 0 };
    const LargeNode = struct { bytes: [160]u8 = @splat(0) };

    allocator: std.mem.Allocator,
    release_calls: usize = 0,
    released_cells: usize = 0,
    smallest_batch: usize = std.math.maxInt(usize),
    largest_batch: usize = 0,

    pub fn traceRoots(_: *SweepBatchTestRT, _: anytype) void {}
    pub fn trace(_: *anyopaque, _: Kind, _: anytype) void {}
    pub fn finalize(_: *SweepBatchTestRT, _: *anyopaque, _: Kind) void {}

    pub fn freeCellStorageBatch(self: *SweepBatchTestRT, total: usize, allocations: []*anyopaque) void {
        self.release_calls += 1;
        self.released_cells += allocations.len;
        self.smallest_batch = @min(self.smallest_batch, allocations.len);
        self.largest_batch = @max(self.largest_batch, allocations.len);
        for (allocations) |allocation| {
            const base: [*]align(16) u8 = @ptrCast(@alignCast(allocation));
            self.allocator.free(base[0..total]);
        }
    }
};

const OwnedCellTestRT = struct {
    pub const Kind = enum { node };
    const Node = struct { value: u32 = 0 };

    owned_header: ?*anyopaque = null,
    owned_start: usize = 0,
    owned_end: usize = 0,
    owned_empty_address: usize = 0,
    conservative_words: []const usize = &.{},
    classify_calls: usize = 0,
    live: bool = false,
    publish_calls: usize = 0,
    unpublish_calls: usize = 0,

    pub fn traceRoots(self: *OwnedCellTestRT, v: anytype) void {
        for (self.conservative_words) |word| v.markConservativeWord(word);
    }
    pub fn trace(_: *anyopaque, _: Kind, _: anytype) void {}
    pub fn finalize(self: *OwnedCellTestRT, _: *anyopaque, _: Kind) void {
        self.live = false;
    }
    pub fn publishCellAllocation(self: *OwnedCellTestRT, allocation: *anyopaque, _: usize) void {
        std.debug.assert(!self.live);
        self.owned_header = allocation;
        self.live = true;
        self.publish_calls += 1;
    }
    pub fn unpublishCellAllocation(self: *OwnedCellTestRT, allocation: *anyopaque, _: usize) void {
        std.debug.assert(self.live and self.owned_header == allocation);
        self.live = false;
        self.unpublish_calls += 1;
    }
    pub fn usesOwnedCellStorage(_: *OwnedCellTestRT, _: usize) bool {
        return true;
    }
    pub fn ownsCellAllocation(self: *OwnedCellTestRT, allocation: *anyopaque) bool {
        return self.live and allocation == self.owned_header;
    }
    pub fn classifyConservativeInterior(self: *OwnedCellTestRT, address: usize) InteriorOwnership {
        self.classify_calls += 1;
        if (address == self.owned_empty_address) return .owned_empty;
        if (self.owned_header) |header| {
            if (address >= self.owned_start and address < self.owned_end)
                return .{ .allocation = header };
        }
        return .outside;
    }
    pub fn allCellsUseOwnedStorage(_: *OwnedCellTestRT) bool {
        return true;
    }
};

const OwnedIterationTestRT = struct {
    pub const Kind = enum { node };
    const Node = struct {
        id: u32 = 0,
        strong: ?*Node = null,
        weak: ?*anyopaque = null,
    };

    roots: [2]?*Node = .{ null, null },
    published: [16]?*anyopaque = @splat(null),
    finalized: [16]u32 = @splat(0),
    finalized_len: usize = 0,
    unpublish_batch_calls: usize = 0,
    largest_unpublish_batch: usize = 0,
    finalizer_saw_published: bool = false,

    pub const Iterator = struct {
        rt: *OwnedIterationTestRT,
        index: usize = 0,

        pub fn next(self: *Iterator) ?*anyopaque {
            while (self.index < self.rt.published.len) {
                const index = self.index;
                self.index += 1;
                if (self.rt.published[index]) |allocation| return allocation;
            }
            return null;
        }
    };

    pub fn ownedCellIterator(self: *OwnedIterationTestRT) Iterator {
        return .{ .rt = self };
    }
    pub fn allCellsUseOwnedStorage(_: *OwnedIterationTestRT) bool {
        return true;
    }
    pub fn usesOwnedCellStorage(_: *OwnedIterationTestRT, _: usize) bool {
        return true;
    }
    pub fn ownsCellAllocation(self: *OwnedIterationTestRT, allocation: *anyopaque) bool {
        for (self.published) |candidate| if (candidate == allocation) return true;
        return false;
    }
    pub fn publishCellAllocation(self: *OwnedIterationTestRT, allocation: *anyopaque, _: usize) void {
        for (&self.published) |*slot| {
            if (slot.* == null) {
                slot.* = allocation;
                return;
            }
        }
        unreachable;
    }
    pub fn unpublishCellAllocation(self: *OwnedIterationTestRT, allocation: *anyopaque, _: usize) void {
        for (&self.published) |*slot| {
            if (slot.* == allocation) {
                slot.* = null;
                return;
            }
        }
        unreachable;
    }
    pub fn unpublishCellAllocationBatch(self: *OwnedIterationTestRT, _: usize, allocations: []*anyopaque) void {
        self.unpublish_batch_calls += 1;
        self.largest_unpublish_batch = @max(self.largest_unpublish_batch, allocations.len);
        for (allocations) |allocation| self.unpublishCellAllocation(allocation, 0);
    }
    pub fn traceRoots(self: *OwnedIterationTestRT, v: anytype) void {
        for (self.roots) |root| v.mark(root);
    }
    pub fn trace(cell: *anyopaque, _: Kind, v: anytype) void {
        const node: *Node = @ptrCast(@alignCast(cell));
        v.mark(node.strong);
        v.markWeak(&node.weak);
    }
    pub fn traceOldOnMinor(_: Kind) bool {
        return true;
    }
    pub fn finalize(self: *OwnedIterationTestRT, cell: *anyopaque, _: Kind) void {
        const node: *Node = @ptrCast(@alignCast(cell));
        const allocation: *anyopaque = @ptrFromInt(@intFromPtr(cell) - Heap(OwnedIterationTestRT).header_stride);
        if (self.ownsCellAllocation(allocation)) self.finalizer_saw_published = true;
        self.finalized[self.finalized_len] = node.id;
        self.finalized_len += 1;
    }
};

const FreeCountingAllocator = struct {
    inner: std.mem.Allocator,
    free_calls: usize = 0,
    watched_free_len: ?usize = null,
    watched_free_calls: usize = 0,

    fn allocFn(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *FreeCountingAllocator = @ptrCast(@alignCast(ctx));
        return self.inner.vtable.alloc(self.inner.ptr, len, alignment, ret_addr);
    }

    fn resizeFn(ctx: *anyopaque, mem: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *FreeCountingAllocator = @ptrCast(@alignCast(ctx));
        return self.inner.vtable.resize(self.inner.ptr, mem, alignment, new_len, ret_addr);
    }

    fn remapFn(ctx: *anyopaque, mem: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *FreeCountingAllocator = @ptrCast(@alignCast(ctx));
        return self.inner.vtable.remap(self.inner.ptr, mem, alignment, new_len, ret_addr);
    }

    fn freeFn(ctx: *anyopaque, mem: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *FreeCountingAllocator = @ptrCast(@alignCast(ctx));
        self.free_calls += 1;
        if (self.watched_free_len == mem.len) self.watched_free_calls += 1;
        self.inner.vtable.free(self.inner.ptr, mem, alignment, ret_addr);
    }

    const vtable: std.mem.Allocator.VTable = .{
        .alloc = allocFn,
        .resize = resizeFn,
        .remap = remapFn,
        .free = freeFn,
    };

    fn allocator(self: *FreeCountingAllocator) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &vtable };
    }
};

test "stable cell metadata is process-unique, batched, and never recycled" {
    const a = std.testing.allocator;
    var first_rt = TestRT{};
    defer first_rt.roots.deinit(a);
    defer first_rt.finalized.deinit(a);
    var second_rt = TestRT{};
    defer second_rt.roots.deinit(a);
    defer second_rt.finalized.deinit(a);

    var first = Heap(TestRT).init(a, &first_rt);
    defer first.deinit();
    var second = Heap(TestRT).init(a, &second_rt);
    defer second.deinit();

    const N = TestRT.Node;
    const rooted = try first.create(N, .node);
    rooted.* = .{ .id = 1 };
    try first_rt.roots.append(a, rooted);
    const other_heap = try second.create(N, .node);
    other_heap.* = .{ .id = 2 };

    var batch: [4]*N = undefined;
    try std.testing.expectEqual(batch.len, try first.createBatch(N, .node, &batch));
    for (batch, 0..) |node, i| node.* = .{ .id = @intCast(i + 10) };

    var ids: [6]StableCellId = undefined;
    ids[0] = first.cellMetadata(rooted).?.id;
    ids[1] = second.cellMetadata(other_heap).?.id;
    for (batch, 0..) |node, i| ids[i + 2] = first.cellMetadata(node).?.id;
    for (ids, 0..) |id, i| {
        try std.testing.expect(@intFromEnum(id) != 0);
        for (ids[0..i]) |prior| try std.testing.expect(id != prior);
    }
    try std.testing.expectEqual(@sizeOf(N), first.cellMetadata(rooted).?.size);
    try std.testing.expectEqual(TestRT.Kind.node, first.cellMetadata(rooted).?.kind);

    var unmanaged: N = .{ .id = 99 };
    try std.testing.expectEqual(@as(?Heap(TestRT).CellMetadata, null), first.cellMetadata(&unmanaged));
    try std.testing.expectEqual(@as(?Heap(TestRT).CellMetadata, null), first.cellMetadata(null));

    const doomed = try first.create(N, .node);
    doomed.* = .{ .id = 20 };
    const doomed_id = first.cellMetadata(doomed).?.id;
    first.collect();
    try std.testing.expectEqual(@as(?Heap(TestRT).CellMetadata, null), first.cellMetadata(doomed));
    const replacement = try first.create(N, .node);
    replacement.* = .{ .id = 21 };
    try std.testing.expect(first.cellMetadata(replacement).?.id != doomed_id);
}

test "64-bit header stays 32 bytes and packed flags transition independently" {
    const H = Heap(TestRT);
    if (@sizeOf(usize) == 8) try std.testing.expectEqual(@as(usize, 32), @sizeOf(H.Header));

    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);
    var heap = H.init(a, &rt);
    defer heap.deinit();
    heap.setNurseryEnabled(true);

    const node = try heap.create(TestRT.Node, .node);
    node.* = .{ .id = 1 };
    const header = H.headerOf(node);
    try std.testing.expect(H.headerFlagLoad(header, H.header_young, .acquire));
    try std.testing.expect(!H.headerFlagLoad(header, H.header_marked, .acquire));
    try std.testing.expect(H.headerFlagSetIfClear(header, H.header_marked, .acq_rel));
    try std.testing.expect(!H.headerFlagSetIfClear(header, H.header_marked, .acq_rel));
    try std.testing.expect(H.headerFlagSetIfClear(header, H.header_remembered_owner, .acq_rel));
    try std.testing.expect(H.headerFlagSetIfClear(header, H.header_remembered_target, .acq_rel));
    H.headerFlagStore(header, H.header_marked, false, .release);
    try std.testing.expect(!H.headerFlagLoad(header, H.header_marked, .acquire));
    try std.testing.expect(H.headerFlagLoad(header, H.header_young, .acquire));
    try std.testing.expect(H.headerFlagLoad(header, H.header_remembered_owner, .acquire));
    try std.testing.expect(H.headerFlagLoad(header, H.header_remembered_target, .acquire));
}

test "mark-sweep: cycles survive via a root, garbage is swept, weak edges clear" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();

    const N = TestRT.Node;
    // root -> A <-> B (a cycle); C is reachable only by a weak edge from B.
    const A = try heap.create(N, .node);
    A.* = .{ .id = 1 };
    const B = try heap.create(N, .node);
    B.* = .{ .id = 2 };
    const C = try heap.create(N, .node);
    C.* = .{ .id = 3 };
    A.strong = B;
    B.strong = A;
    B.weak = C;
    try rt.roots.append(a, A);

    try std.testing.expectEqual(@as(usize, 3), heap.live_cells);

    heap.collect();
    // The cycle is kept alive by the root; C (weak-only) is collected.
    try std.testing.expectEqual(@as(usize, 2), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 1), rt.finalized.items.len);
    try std.testing.expectEqual(@as(u32, 3), rt.finalized.items[0]);
    // The weak edge was nulled before C's storage was freed.
    try std.testing.expectEqual(@as(?*anyopaque, null), B.weak);

    // Drop the root: the whole cycle becomes unreachable and is reclaimed.
    rt.roots.clearRetainingCapacity();
    heap.collect();
    try std.testing.expectEqual(@as(usize, 0), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 3), rt.finalized.items.len);
}

test "stop-the-world compaction rewrites cycles weak roots and pinned edges" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();

    const N = TestRT.Node;
    const old_a = try heap.create(N, .node);
    old_a.* = .{ .id = 1 };
    const old_b = try heap.create(N, .node);
    old_b.* = .{ .id = 2 };
    const dead = try heap.create(N, .node);
    dead.* = .{ .id = 3 };
    const weak_survivor = try heap.create(N, .node);
    weak_survivor.* = .{ .id = 4 };
    const pinned = try heap.create(N, .node);
    pinned.* = .{ .id = 99 };
    old_a.strong = old_b;
    old_a.weak = weak_survivor;
    old_b.strong = old_a;
    old_b.weak = dead;
    pinned.strong = old_b;
    try rt.roots.append(a, old_a);
    try rt.roots.append(a, pinned);
    try rt.roots.append(a, weak_survivor);

    const old_a_id = heap.cellMetadata(old_a).?.id;
    const old_b_id = heap.cellMetadata(old_b).?.id;
    const weak_survivor_id = heap.cellMetadata(weak_survivor).?.id;
    const pinned_id = heap.cellMetadata(pinned).?.id;

    const before = heap.accounting();
    const result = heap.collectAndCompact();
    try std.testing.expectEqual(Heap(TestRT).CompactionStatus.compacted, result.status);
    try std.testing.expectEqual(@as(usize, 3), result.moved_cells);
    try std.testing.expect(result.moved_bytes != 0);
    try std.testing.expectEqual(@as(usize, 1), rt.finalized.items.len);
    try std.testing.expectEqual(@as(u32, 3), rt.finalized.items[0]);

    const new_a = rt.roots.items[0];
    const new_b = new_a.strong.?;
    try std.testing.expect(new_a != old_a);
    try std.testing.expect(new_b != old_b);
    try std.testing.expectEqual(new_a, new_b.strong.?);
    try std.testing.expectEqual(@as(*anyopaque, rt.roots.items[2]), new_a.weak.?);
    try std.testing.expect(rt.roots.items[2] != weak_survivor);
    try std.testing.expectEqual(@as(?*anyopaque, null), new_b.weak);
    try std.testing.expectEqual(pinned, rt.roots.items[1]);
    try std.testing.expectEqual(new_b, pinned.strong.?);
    try std.testing.expectEqual(old_a_id, heap.cellMetadata(new_a).?.id);
    try std.testing.expectEqual(old_b_id, heap.cellMetadata(new_b).?.id);
    try std.testing.expectEqual(weak_survivor_id, heap.cellMetadata(rt.roots.items[2]).?.id);
    try std.testing.expectEqual(pinned_id, heap.cellMetadata(pinned).?.id);
    try std.testing.expectEqual(@as(usize, 1), rt.relocation_root_verifications);
    try std.testing.expectEqual(@as(usize, 4), rt.relocation_cell_verifications);

    const after = heap.accounting();
    try std.testing.expectEqual(before.live_cells - 1, after.live_cells);
    try std.testing.expectEqual(before.live_bytes - Heap(TestRT).cellAllocationBytes(N), after.live_bytes);

    rt.roots.clearRetainingCapacity();
    heap.collect();
    try std.testing.expectEqual(@as(usize, 0), heap.accounting().live_cells);
    try std.testing.expectEqual(@as(usize, 5), rt.finalized.items.len);
}

test "compaction reports no candidates for an empty or entirely pinned heap" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);
    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();

    try std.testing.expectEqual(
        Heap(TestRT).CompactionStatus.no_candidates,
        heap.collectAndCompact().status,
    );
    const pinned = try heap.create(TestRT.Node, .node);
    pinned.* = .{ .id = 99 };
    try rt.roots.append(a, pinned);
    const result = heap.collectAndCompact();
    try std.testing.expectEqual(Heap(TestRT).CompactionStatus.no_candidates, result.status);
    try std.testing.expectEqual(pinned, rt.roots.items[0]);
    try std.testing.expectEqual(@as(usize, 0), rt.relocation_reserve_calls);
}

test "compaction destination OOM rolls back every reservation without graph mutation" {
    const a = std.testing.allocator;
    for (0..2) |fail_at| {
        var rt = TestRT{ .relocation_reserve_limit = fail_at };
        defer rt.roots.deinit(a);
        defer rt.finalized.deinit(a);
        var heap = Heap(TestRT).init(a, &rt);
        defer heap.deinit();

        const first = try heap.create(TestRT.Node, .node);
        first.* = .{ .id = 10 };
        const second = try heap.create(TestRT.Node, .node);
        second.* = .{ .id = 11 };
        first.strong = second;
        second.strong = first;
        try rt.roots.append(a, first);
        const before = heap.accounting();

        const result = heap.collectAndCompact();
        try std.testing.expectEqual(Heap(TestRT).CompactionStatus.out_of_memory, result.status);
        try std.testing.expectEqual(@as(usize, 0), result.moved_cells);
        try std.testing.expectEqual(first, rt.roots.items[0]);
        try std.testing.expectEqual(second, first.strong.?);
        try std.testing.expectEqual(first, second.strong.?);
        try std.testing.expectEqual(before.live_cells, heap.accounting().live_cells);
        try std.testing.expectEqual(before.live_bytes, heap.accounting().live_bytes);
        try std.testing.expectEqual(fail_at, rt.relocation_reserve_calls);
        try std.testing.expectEqual(fail_at, rt.relocation_rollbacks);
        try std.testing.expectEqual(@as(usize, 0), rt.relocation_commits);
        try std.testing.expectEqual(@as(usize, 0), rt.finalized.items.len);
    }
}

test "post-sweep hook observes finalizers after the parallel allocation lock is released" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();
    heap.setParallel(true);
    rt.alloc_lock_probe = &heap.alloc_lock;
    rt.publication_gate_probe = &heap.sharded_publication_gate;
    heap.setNurseryEnabled(true);

    const nursery_garbage = try heap.create(TestRT.Node, .node);
    nursery_garbage.* = .{ .id = 34 };
    heap.collectYoung();
    try std.testing.expectEqual(@as(usize, 1), rt.after_sweep_calls);
    try std.testing.expectEqual(@as(usize, 1), rt.finalized_at_last_after_sweep);
    try std.testing.expect(!rt.after_sweep_under_alloc_lock);
    try std.testing.expect(!rt.after_sweep_before_publication_reopened);

    const full_garbage = try heap.create(TestRT.Node, .node);
    full_garbage.* = .{ .id = 35 };
    heap.collect();
    try std.testing.expectEqual(@as(usize, 2), rt.after_sweep_calls);
    try std.testing.expectEqual(@as(usize, 2), rt.finalized_at_last_after_sweep);
    try std.testing.expect(!rt.after_sweep_under_alloc_lock);
    try std.testing.expect(!rt.after_sweep_before_publication_reopened);

    heap.beginConcurrentMarkParallel();
    heap.abortConcurrentMarkParallel();
    try std.testing.expectEqual(@as(usize, 2), rt.after_sweep_calls);

    const concurrent_garbage = try heap.create(TestRT.Node, .node);
    concurrent_garbage.* = .{ .id = 36 };
    heap.beginConcurrentMarkParallel();
    while (!heap.concurrentMarkRound()) {}
    try std.testing.expect(heap.finishConcurrentMarkParallel());
    try std.testing.expectEqual(@as(usize, 3), rt.after_sweep_calls);
    try std.testing.expectEqual(@as(usize, 3), rt.finalized_at_last_after_sweep);
    try std.testing.expect(!rt.after_sweep_under_alloc_lock);
    try std.testing.expect(!rt.after_sweep_before_publication_reopened);
}

test "collection phase boundaries distinguish minor and force-full ordering" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();

    // Disabled and empty nursery calls are true no-ops for observers.
    heap.collectYoung();
    heap.setNurseryEnabled(true);
    heap.collectYoung();
    try std.testing.expectEqual(@as(usize, 0), rt.collection_phase_boundary_len);

    const nursery_garbage = try heap.create(TestRT.Node, .node);
    nursery_garbage.* = .{ .id = 41 };
    heap.collectYoung();
    try std.testing.expectEqualSlices(CollectionPhaseBoundary, &.{
        .minor_prepare_begin,
        .minor_trace_begin,
        .minor_sweep_begin,
        .minor_sweep_end,
        .minor_post_sweep_end,
    }, rt.collection_phase_boundaries[0..rt.collection_phase_boundary_len]);
    // `afterSweep` runs between the sweep-end and post-sweep-end boundaries.
    try std.testing.expectEqual(@as(usize, 4), rt.phase_boundary_len_at_after_sweep);

    rt.collection_phase_boundary_len = 0;
    const full_garbage = try heap.create(TestRT.Node, .node);
    full_garbage.* = .{ .id = 42 };
    heap.nursery_force_full.store(true, .release);
    heap.collectYoung();
    try std.testing.expectEqualSlices(CollectionPhaseBoundary, &.{
        .full_prepare_begin,
        .full_trace_begin,
        .full_sweep_begin,
        .full_sweep_end,
        .full_post_sweep_end,
    }, rt.collection_phase_boundaries[0..rt.collection_phase_boundary_len]);
    try std.testing.expectEqual(@as(usize, 4), rt.phase_boundary_len_at_after_sweep);
}

test "deinit finalizes every remaining cell" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);

    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        const n = try heap.create(TestRT.Node, .node);
        n.* = .{ .id = i };
    }
    try std.testing.expectEqual(@as(usize, 10), heap.live_cells);

    heap.deinit(); // no roots → everything is garbage, all finalized
    try std.testing.expectEqual(@as(usize, 10), rt.finalized.items.len);
}

test "sweep releases mixed cell storage through bounded same-size binding batches" {
    const a = std.testing.allocator;
    var rt = SweepBatchTestRT{ .allocator = a };
    var heap = Heap(SweepBatchTestRT).init(a, &rt);
    defer heap.deinit();

    for (0..65) |value| {
        const node = try heap.create(SweepBatchTestRT.Node, .node);
        node.* = .{ .value = value };
    }
    for (0..3) |_| {
        const node = try heap.create(SweepBatchTestRT.LargeNode, .large_node);
        node.* = .{};
    }
    for (0..65) |value| {
        const node = try heap.create(SweepBatchTestRT.Node, .node);
        node.* = .{ .value = value + 65 };
    }
    heap.collect();

    try std.testing.expectEqual(@as(usize, 0), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 133), rt.released_cells);
    try std.testing.expectEqual(@as(usize, 5), rt.release_calls);
    try std.testing.expectEqual(@as(usize, 1), rt.smallest_batch);
    try std.testing.expectEqual(@as(usize, 64), rt.largest_batch);
}

test "accounting snapshots live and last-full bytes without minor-cycle drift" {
    const a = std.testing.allocator;
    const H = Heap(TestRT);
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = H.init(a, &rt);
    defer heap.deinit();
    heap.setParallel(true);
    heap.setNurseryEnabled(true);

    const live = try heap.create(TestRT.Node, .node);
    live.* = .{ .id = 1 };
    const dead = try heap.create(TestRT.Node, .node);
    dead.* = .{ .id = 2 };
    try rt.roots.append(a, live);

    const cell_bytes = H.cellAllocationBytes(TestRT.Node);
    const before = heap.accounting();
    try std.testing.expectEqual(@as(usize, 2), before.live_cells);
    try std.testing.expectEqual(cell_bytes * 2, before.live_bytes);
    try std.testing.expectEqual(@as(usize, 0), before.last_full_collection_bytes);

    heap.collectYoung();
    const after_minor = heap.accounting();
    try std.testing.expectEqual(@as(usize, 1), after_minor.live_cells);
    try std.testing.expectEqual(cell_bytes, after_minor.live_bytes);
    try std.testing.expectEqual(@as(usize, 0), after_minor.last_full_collection_bytes);
    try std.testing.expectEqual(@as(usize, 1), after_minor.minor_collections);

    heap.collect();
    const after_full = heap.accounting();
    try std.testing.expectEqual(cell_bytes, after_full.live_bytes);
    try std.testing.expectEqual(cell_bytes, after_full.last_full_collection_bytes);
    try std.testing.expectEqual(@as(usize, 2), after_full.collections);
    try std.testing.expectEqual(@as(usize, 1), after_full.full_collections);
}

test "bulk deinit finalizes cells without individual backing frees" {
    const a = std.testing.allocator;
    const H = Heap(TestRT);
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    var counting = FreeCountingAllocator{
        .inner = arena_state.allocator(),
        .watched_free_len = H.header_stride + @sizeOf(TestRT.Node),
    };
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = H.init(counting.allocator(), &rt);
    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        const n = try heap.create(TestRT.Node, .node);
        n.* = .{ .id = i };
    }

    heap.deinitRetainingCellStorage();
    try std.testing.expectEqual(@as(usize, 10), rt.finalized.items.len);
    try std.testing.expectEqual(@as(usize, 0), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 0), heap.bytes_live);
    try std.testing.expectEqual(@as(usize, 0), counting.watched_free_calls);
}

test "nursery reclaims young garbage and tenures root survivors" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();
    heap.setNurseryEnabled(true);

    const live = try heap.create(TestRT.Node, .node);
    live.* = .{ .id = 1 };
    const garbage = try heap.create(TestRT.Node, .node);
    garbage.* = .{ .id = 2 };
    try rt.roots.append(a, live);
    heap.nursery_threshold_bytes = 512 * 1024;

    try std.testing.expectEqual(@as(usize, 2), heap.young_cells);
    const young_bytes_before = heap.young_bytes;
    heap.collectYoung();

    try std.testing.expectEqual(@as(usize, 1), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 0), heap.young_cells);
    try std.testing.expectEqual(@as(usize, 1), heap.promoted_cells);
    try std.testing.expectEqual(young_bytes_before, heap.last_minor_young_bytes);
    try std.testing.expectEqual(young_bytes_before / 2, heap.last_minor_promoted_bytes);
    try std.testing.expectEqual(young_bytes_before / 2, heap.last_minor_reclaimed_bytes);
    try std.testing.expectEqual(Heap(TestRT).min_nursery_threshold_bytes, heap.nursery_threshold_bytes);
    try std.testing.expectEqual(@as(usize, 1), heap.minor_collections);
    try std.testing.expectEqual(@as(usize, 0), heap.full_collections);
    try std.testing.expectEqual(@as(u32, 2), rt.finalized.items[0]);
    heap.threshold_bytes = 1;
    try std.testing.expect(heap.shouldCollectOld());
}

test "multi-age nursery retains survivors and promotes at the configured age" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    const H = Heap(TestRT);
    var heap = H.init(a, &rt);
    defer heap.deinit();
    heap.setNurseryTenuringAge(3);
    heap.setNurseryEnabled(true);

    const survivor = try heap.create(TestRT.Node, .node);
    survivor.* = .{ .id = 1 };
    const later_garbage = try heap.create(TestRT.Node, .node);
    later_garbage.* = .{ .id = 2 };
    try rt.roots.append(a, survivor);
    try rt.roots.append(a, later_garbage);
    const node_bytes = heap.young_bytes / 2;

    heap.collectYoung();
    try std.testing.expectEqual(@as(usize, 2), heap.young_cells);
    try std.testing.expectEqual(@as(u8, 1), H.headerOf(survivor).age);
    try std.testing.expectEqual(@as(usize, 2), heap.last_minor_survived_cells);
    try std.testing.expectEqual(@as(usize, 0), heap.last_minor_promoted_bytes);

    _ = rt.roots.pop();
    heap.collectYoung();
    try std.testing.expectEqual(@as(usize, 1), heap.young_cells);
    try std.testing.expectEqual(@as(u8, 2), H.headerOf(survivor).age);
    try std.testing.expectEqual(@as(u32, 2), rt.finalized.items[0]);

    heap.collectYoung();
    const stats = heap.accounting();
    try std.testing.expectEqual(@as(usize, 0), stats.young_cells);
    try std.testing.expectEqual(@as(usize, 1), stats.promoted_cells);
    try std.testing.expectEqual(@as(u8, 3), stats.tenuring_age);
    try std.testing.expectEqual(@as(usize, 1), stats.last_minor_survived_cells);
    try std.testing.expect(stats.last_minor_promoted_bytes > 0);
    try std.testing.expectEqual(5 * node_bytes, stats.total_minor_young_bytes);
    try std.testing.expectEqual(4 * node_bytes, stats.total_minor_survived_bytes);
    try std.testing.expectEqual(node_bytes, stats.total_minor_reclaimed_bytes);
    try std.testing.expectEqual(node_bytes, stats.total_minor_promoted_bytes);
    try std.testing.expectEqual(
        stats.total_minor_young_bytes,
        stats.total_minor_survived_bytes + stats.total_minor_reclaimed_bytes,
    );
    try std.testing.expectEqual(H.tenured_age, H.headerOf(survivor).age);
}

test "multi-age owner and weak cards persist across minor collections" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();
    heap.setNurseryTenuringAge(3);
    heap.setNurseryEnabled(true);

    const owner = try heap.create(TestRT.Node, .node);
    owner.* = .{ .id = 1 };
    try rt.roots.append(a, owner);
    heap.collectYoung();
    heap.collectYoung();
    heap.collectYoung();

    const child = try heap.create(TestRT.Node, .node);
    child.* = .{ .id = 2 };
    owner.strong = child;
    heap.writeBarrierFromManaged(owner, child);
    heap.collectYoung();
    heap.collectYoung();
    heap.collectYoung();
    try std.testing.expectEqual(@as(usize, 2), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 0), heap.young_cells);

    const weak_target = try heap.create(TestRT.Node, .node);
    weak_target.* = .{ .id = 3 };
    owner.weak = weak_target;
    try rt.roots.append(a, weak_target);
    heap.writeBarrierWeak(owner);
    heap.collectYoung();
    _ = rt.roots.pop();
    heap.collectYoung();
    try std.testing.expectEqual(@as(?*anyopaque, null), owner.weak);
    try std.testing.expectEqual(@as(u32, 3), rt.finalized.items[0]);

    owner.strong = null;
    heap.collect();
    try std.testing.expectEqual(@as(usize, 1), heap.live_cells);
}

test "full collection and nursery disable tenure multi-age survivors" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    const H = Heap(TestRT);
    var heap = H.init(a, &rt);
    defer heap.deinit();
    heap.setNurseryTenuringAge(4);
    heap.setNurseryEnabled(true);

    const full_survivor = try heap.create(TestRT.Node, .node);
    full_survivor.* = .{ .id = 1 };
    try rt.roots.append(a, full_survivor);
    heap.collectYoung();
    try std.testing.expectEqual(@as(usize, 1), heap.young_cells);
    heap.collect();
    try std.testing.expectEqual(@as(usize, 0), heap.young_cells);
    try std.testing.expectEqual(H.tenured_age, H.headerOf(full_survivor).age);

    const disabled_survivor = try heap.create(TestRT.Node, .node);
    disabled_survivor.* = .{ .id = 2 };
    try rt.roots.append(a, disabled_survivor);
    heap.collectYoung();
    try std.testing.expectEqual(@as(usize, 1), heap.young_cells);
    heap.setNurseryEnabled(false);
    try std.testing.expectEqual(@as(usize, 0), heap.young_cells);
    try std.testing.expectEqual(H.tenured_age, H.headerOf(disabled_survivor).age);
}

test "nursery disable tenures the pending prefix before old allocations" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();
    heap.setNurseryEnabled(true);

    const first = try heap.create(TestRT.Node, .node);
    first.* = .{ .id = 1 };
    const second = try heap.create(TestRT.Node, .node);
    second.* = .{ .id = 2 };
    const promoted_before = heap.promoted_cells;
    heap.setNurseryEnabled(false);
    try std.testing.expectEqual(@as(usize, 0), heap.young_cells);
    try std.testing.expectEqual(promoted_before + 2, heap.promoted_cells);

    const old = try heap.create(TestRT.Node, .node);
    old.* = .{ .id = 3 };
    heap.setNurseryEnabled(true);
    const live = try heap.create(TestRT.Node, .node);
    live.* = .{ .id = 4 };
    const garbage = try heap.create(TestRT.Node, .node);
    garbage.* = .{ .id = 5 };
    try rt.roots.append(a, live);
    heap.collectYoung();

    try std.testing.expectEqual(@as(usize, 4), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 0), heap.young_cells);
    try std.testing.expectEqual(@as(u32, 5), rt.finalized.items[0]);
}

test "minor whitening and sweep stop at the old-generation boundary" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    const H = Heap(TestRT);
    var heap = H.init(a, &rt);
    defer heap.deinit();
    heap.setNurseryEnabled(true);

    const old = try heap.create(TestRT.Node, .node);
    old.* = .{ .id = 1 };
    try rt.roots.append(a, old);
    heap.collectYoung();

    const live = try heap.create(TestRT.Node, .node);
    live.* = .{ .id = 2 };
    const garbage = try heap.create(TestRT.Node, .node);
    garbage.* = .{ .id = 3 };
    try rt.roots.append(a, live);

    const old_header = H.headerOf(old);
    const saved_next = old_header.next;
    old_header.next = @ptrFromInt(@as(usize, 0x1000));
    heap.collectYoung();
    old_header.next = saved_next;

    try std.testing.expectEqual(@as(usize, 2), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 0), heap.young_cells);
    try std.testing.expectEqual(@as(u32, 3), rt.finalized.items[0]);
}

test "nursery threshold growth is capped by observed young batch" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();
    heap.setNurseryEnabled(true);

    heap.nursery_threshold_bytes = 8 * 1024 * 1024;

    try std.testing.expectEqual(
        @as(usize, 12 * 1024 * 1024),
        heap.nextNurseryThreshold(12 * 1024 * 1024, 8 * 1024 * 1024),
    );
    try std.testing.expectEqual(
        Heap(TestRT).min_nursery_threshold_bytes,
        heap.nextNurseryThreshold(64 * 1024, 64 * 1024),
    );
}

test "collector scratch frees oversized empty buffers after spike" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    const H = Heap(TestRT);
    var heap = H.init(a, &rt);
    defer heap.deinit();

    const retained = H.min_retained_scratch_entries;
    const oversized = retained + 1;
    try heap.mark_stack.ensureTotalCapacityPrecise(a, oversized);
    try heap.weak_slots.ensureTotalCapacityPrecise(a, oversized);
    try heap.weak_atomic_slots.ensureTotalCapacityPrecise(a, oversized);
    try heap.remembered_owners.ensureTotalCapacityPrecise(a, oversized);
    try heap.remembered_targets.ensureTotalCapacityPrecise(a, oversized);
    try heap.barrier_buf.ensureTotalCapacityPrecise(a, oversized);
    try heap.born_concurrent.ensureTotalCapacityPrecise(a, oversized);
    try heap.deferred_trace.ensureTotalCapacityPrecise(a, oversized);
    heap.live_cells = 1;

    heap.trimCollectorScratch();

    try std.testing.expectEqual(@as(usize, 0), heap.mark_stack.capacity);
    try std.testing.expectEqual(@as(usize, 0), heap.weak_slots.capacity);
    try std.testing.expectEqual(@as(usize, 0), heap.weak_atomic_slots.capacity);
    try std.testing.expectEqual(@as(usize, 0), heap.remembered_owners.capacity);
    try std.testing.expectEqual(@as(usize, 0), heap.remembered_targets.capacity);
    try std.testing.expectEqual(@as(usize, 0), heap.barrier_buf.capacity);
    try std.testing.expectEqual(@as(usize, 0), heap.born_concurrent.capacity);
    try std.testing.expectEqual(@as(usize, 0), heap.deferred_trace.capacity);

    try heap.mark_stack.ensureTotalCapacityPrecise(a, retained);
    heap.trimCollectorScratch();
    try std.testing.expectEqual(retained, heap.mark_stack.capacity);
}

test "atomic weak roots survive only while strongly reachable" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();

    const target = try heap.create(TestRT.Node, .node);
    target.* = .{ .id = 71 };
    var slot = std.atomic.Value(?*anyopaque).init(@ptrCast(target));
    rt.atomic_weak_root = &slot;
    try rt.roots.append(a, target);

    heap.collect();
    try std.testing.expectEqual(@as(?*anyopaque, @ptrCast(target)), slot.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 1), rt.after_weak_roots_calls);

    rt.roots.clearRetainingCapacity();
    heap.collect();
    try std.testing.expectEqual(@as(?*anyopaque, null), slot.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), heap.live_cells);
    try std.testing.expectEqualSlices(u32, &.{71}, rt.finalized.items);
    try std.testing.expectEqual(@as(usize, 2), rt.after_weak_roots_calls);
}

test "nursery owner barrier preserves old-to-young edges" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();
    heap.setNurseryEnabled(true);

    const owner = try heap.create(TestRT.Node, .node);
    owner.* = .{ .id = 1 };
    try rt.roots.append(a, owner);
    heap.collectYoung();

    const child = try heap.create(TestRT.Node, .node);
    child.* = .{ .id = 2 };
    owner.strong = child;
    heap.writeBarrierFrom(owner, child);
    heap.collectYoung();

    try std.testing.expectEqual(@as(usize, 2), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 0), heap.young_cells);
    try std.testing.expectEqual(@as(usize, 0), rt.finalized.items.len);
    owner.strong = null;
    heap.collect();
    try std.testing.expectEqual(@as(usize, 1), heap.live_cells);
    try std.testing.expectEqual(@as(u32, 2), rt.finalized.items[0]);
}

test "exact managed owner barrier preserves old-to-young edges" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();
    heap.setNurseryEnabled(true);

    const owner = try heap.create(TestRT.Node, .node);
    owner.* = .{ .id = 1 };
    try rt.roots.append(a, owner);
    heap.collectYoung();

    const child = try heap.create(TestRT.Node, .node);
    child.* = .{ .id = 2 };
    owner.strong = child;
    heap.writeBarrierFromManaged(owner, child);
    heap.collectYoung();

    try std.testing.expectEqual(@as(usize, 2), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 0), heap.young_cells);
    try std.testing.expectEqual(@as(usize, 0), rt.finalized.items.len);
    owner.strong = null;
    heap.collect();
    try std.testing.expectEqual(@as(usize, 1), heap.live_cells);
    try std.testing.expectEqual(@as(u32, 2), rt.finalized.items[0]);
}

test "nursery weak barrier clears an old container's dead young target" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();
    heap.setNurseryEnabled(true);

    const owner = try heap.create(TestRT.Node, .node);
    owner.* = .{ .id = 1 };
    try rt.roots.append(a, owner);
    heap.collectYoung();

    const target = try heap.create(TestRT.Node, .node);
    target.* = .{ .id = 2 };
    owner.weak = target;
    heap.writeBarrierWeak(owner);
    heap.collectYoung();

    try std.testing.expectEqual(@as(?*anyopaque, null), owner.weak);
    try std.testing.expectEqual(@as(usize, 1), heap.live_cells);
    try std.testing.expectEqual(@as(u32, 2), rt.finalized.items[0]);
}

test "nursery child-only barrier is a conservative compatibility root" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();
    heap.setNurseryEnabled(true);

    const child = try heap.create(TestRT.Node, .node);
    child.* = .{ .id = 1 };
    heap.writeBarrier(child);
    heap.collectYoung();

    try std.testing.expectEqual(@as(usize, 1), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 0), heap.young_cells);
    heap.collect();
    try std.testing.expectEqual(@as(usize, 0), heap.live_cells);
}

test "write barriers tolerate non-heap pointer values" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();
    heap.setNurseryEnabled(true);

    var stack_value: usize = 0;
    const stack_ptr: *anyopaque = @ptrCast(&stack_value);
    const wild: *anyopaque = @ptrFromInt(@as(usize, 0x1000_0000_0000));

    // Nursery compatibility paths: maybe-managed child/owner pointers must be
    // rejected by heap membership, not by peeking at their candidate headers.
    heap.writeBarrier(stack_ptr);
    heap.writeBarrier(wild);
    heap.writeBarrierFrom(stack_ptr, wild);
    heap.writeBarrierFrom(wild, stack_ptr);
    heap.writeBarrierWeak(stack_ptr);
    heap.writeBarrierWeak(wild);

    // Incremental/concurrent barrier paths have the same public contract.
    heap.marking.store(true, .release);
    defer heap.marking.store(false, .release);
    heap.writeBarrier(stack_ptr);
    heap.writeBarrier(wild);
    heap.writeBarrierFrom(stack_ptr, wild);
    heap.writeBarrierFrom(wild, stack_ptr);
}

test "mark-sweep: optional conservative words root payload and interior pointers" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();

    const A = try heap.create(TestRT.Node, .node);
    A.* = .{ .id = 1 };
    const B = try heap.create(TestRT.Node, .node);
    B.* = .{ .id = 2 };
    const C = try heap.create(TestRT.Node, .node);
    C.* = .{ .id = 3 };

    const b_interior = @intFromPtr(B) + @offsetOf(TestRT.Node, "id");
    const words = [_]usize{ @intFromPtr(A), b_interior, 0x1234 };
    rt.conservative_words = &words;

    heap.collect();

    try std.testing.expectEqual(@as(usize, 2), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 1), rt.finalized.items.len);
    try std.testing.expectEqual(@as(u32, 3), rt.finalized.items[0]);
}

test "visitor isManaged tolerates non-heap pointer values" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();

    const live = try heap.create(TestRT.Node, .node);
    live.* = .{ .id = 7 };

    var visitor = Heap(TestRT).Visitor{ .heap = &heap };
    try std.testing.expect(visitor.isManaged(live));

    var stack_value: usize = 0;
    try std.testing.expect(!visitor.isManaged(@ptrCast(&stack_value)));

    const wild: *anyopaque = @ptrFromInt(@as(usize, 0x1000_0000_0000));
    try std.testing.expect(!visitor.isManaged(wild));
}

test "owned cell hook replaces the payload index and rejects stale cells" {
    const H = Heap(OwnedCellTestRT);
    var rt = OwnedCellTestRT{};
    var heap = H.init(std.testing.allocator, &rt);
    defer heap.deinit();

    const live = try heap.create(OwnedCellTestRT.Node, .node);
    live.* = .{ .value = 7 };

    try std.testing.expectEqual(@as(usize, 1), rt.publish_calls);
    try std.testing.expectEqual(@as(usize, 0), heap.payload_index.count());
    var visitor = H.Visitor{ .heap = &heap };
    try std.testing.expect(visitor.isManaged(live));

    heap.collect();
    try std.testing.expectEqual(@as(usize, 1), rt.unpublish_calls);
    try std.testing.expect(!visitor.isManaged(live));
}

test "owned conservative classification skips the generic address index" {
    const H = Heap(OwnedCellTestRT);
    var rt = OwnedCellTestRT{};
    var heap = H.init(std.testing.allocator, &rt);
    defer heap.deinit();

    const live = try heap.create(OwnedCellTestRT.Node, .node);
    live.* = .{ .value = 7 };
    const header = @intFromPtr(live) - H.header_stride;
    rt.owned_header = @ptrFromInt(header);
    rt.owned_start = header;
    rt.owned_end = header + H.cellAllocationBytes(OwnedCellTestRT.Node) + 16;
    rt.owned_empty_address = 0x1111;

    const words = [_]usize{
        header,
        @intFromPtr(live),
        @intFromPtr(live) + 1,
        @intFromPtr(live) + @sizeOf(OwnedCellTestRT.Node),
        rt.owned_empty_address,
        0x2222,
    };
    rt.conservative_words = &words;
    heap.collect();

    try std.testing.expectEqual(@as(usize, 1), heap.live_cells);
    try std.testing.expect(rt.classify_calls >= words.len);
    try std.testing.expect(!heap.addr_index_built);
    try std.testing.expectEqual(@as(usize, 0), heap.addr_index.items.len);
}

test "owned cell iteration replaces intrusive publication across minor and full sweeps" {
    const a = std.testing.allocator;
    const H = Heap(OwnedIterationTestRT);
    var rt = OwnedIterationTestRT{};
    var heap = H.init(a, &rt);
    heap.setNurseryEnabled(true);

    const old = try heap.create(OwnedIterationTestRT.Node, .node);
    old.* = .{ .id = 1 };
    rt.roots[0] = old;
    try std.testing.expectEqual(@as(?*H.Header, null), heap.all);
    heap.collectYoung();

    const live = try heap.create(OwnedIterationTestRT.Node, .node);
    live.* = .{ .id = 2 };
    const garbage = try heap.create(OwnedIterationTestRT.Node, .node);
    garbage.* = .{ .id = 3 };
    old.weak = garbage;
    rt.roots[1] = live;
    heap.collectYoung();

    try std.testing.expectEqual(@as(?*H.Header, null), heap.all);
    try std.testing.expectEqual(@as(usize, 2), heap.live_cells);
    try std.testing.expectEqual(@as(?*anyopaque, null), old.weak);
    try std.testing.expectEqual(@as(usize, 1), rt.finalized_len);
    try std.testing.expectEqual(@as(u32, 3), rt.finalized[0]);
    try std.testing.expectEqual(@as(usize, 1), rt.unpublish_batch_calls);
    try std.testing.expectEqual(@as(usize, 1), rt.largest_unpublish_batch);
    try std.testing.expect(!rt.finalizer_saw_published);

    rt.roots = .{ null, null };
    heap.collect();
    try std.testing.expectEqual(@as(usize, 0), heap.live_cells);
    for (rt.published) |allocation| try std.testing.expectEqual(@as(?*anyopaque, null), allocation);
    try std.testing.expectEqual(@as(usize, 2), rt.unpublish_batch_calls);
    try std.testing.expectEqual(@as(usize, 2), rt.largest_unpublish_batch);
    try std.testing.expect(!rt.finalizer_saw_published);

    const teardown = try heap.create(OwnedIterationTestRT.Node, .node);
    teardown.* = .{ .id = 4 };
    try std.testing.expectEqual(@as(?*H.Header, null), heap.all);
    heap.deinit();
    try std.testing.expectEqual(@as(usize, 4), rt.finalized_len);
    for (rt.published) |allocation| try std.testing.expectEqual(@as(?*anyopaque, null), allocation);
}

const EphRT = struct {
    pub const Kind = enum { node, table };

    const Node = struct {
        strong: ?*Node = null,
        id: u32 = 0,
    };
    const Entry = struct {
        key: ?*Node = null,
        value: ?*Node = null,
    };
    const Table = struct {
        entries: std.ArrayListUnmanaged(Entry) = .empty,
    };

    roots: std.ArrayListUnmanaged(*anyopaque) = .empty,
    finalized: std.ArrayListUnmanaged(u32) = .empty,
    has_weak_work: bool = true,
    weak_work_checks: usize = 0,
    ephemeron_calls: usize = 0,
    after_weak_calls: usize = 0,

    pub fn hasWeakWork(self: *EphRT) bool {
        self.weak_work_checks += 1;
        return self.has_weak_work;
    }

    pub fn traceRoots(self: *EphRT, v: anytype) void {
        for (self.roots.items) |r| v.mark(r);
    }

    pub fn traceOldOnMinor(kind: Kind) bool {
        return kind == .table;
    }

    pub fn trace(cell: *anyopaque, kind: Kind, v: anytype) void {
        switch (kind) {
            .node => {
                const n: *Node = @ptrCast(@alignCast(cell));
                v.mark(n.strong);
            },
            .table => {
                const t: *Table = @ptrCast(@alignCast(cell));
                for (t.entries.items) |*e| v.markWeak(@ptrCast(&e.key));
            },
        }
    }

    pub fn traceEphemeron(self: *EphRT, cell: *anyopaque, kind: Kind, v: anytype) void {
        self.ephemeron_calls += 1;
        switch (kind) {
            .node => {},
            .table => {
                const t: *Table = @ptrCast(@alignCast(cell));
                for (t.entries.items) |e| {
                    if (v.isMarked(e.key)) v.mark(e.value);
                }
            },
        }
    }

    pub fn afterWeak(self: *EphRT, cell: *anyopaque, kind: Kind) void {
        self.after_weak_calls += 1;
        switch (kind) {
            .node => {},
            .table => {
                const t: *Table = @ptrCast(@alignCast(cell));
                var i: usize = 0;
                while (i < t.entries.items.len) {
                    if (t.entries.items[i].key == null) {
                        _ = t.entries.orderedRemove(i);
                    } else {
                        i += 1;
                    }
                }
            },
        }
    }

    pub fn finalize(self: *EphRT, cell: *anyopaque, kind: Kind) void {
        switch (kind) {
            .node => {
                const n: *Node = @ptrCast(@alignCast(cell));
                self.finalized.append(std.testing.allocator, n.id) catch {};
            },
            .table => {
                const t: *Table = @ptrCast(@alignCast(cell));
                t.entries.deinit(std.testing.allocator);
            },
        }
    }
};

test "binding can skip weak passes until weak work exists" {
    const a = std.testing.allocator;
    var rt = EphRT{ .has_weak_work = false };
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(EphRT).init(a, &rt);
    defer heap.deinit();
    const node = try heap.create(EphRT.Node, .node);
    node.* = .{ .id = 1 };
    try rt.roots.append(a, node);

    heap.collect();
    try std.testing.expectEqual(@as(usize, 1), rt.weak_work_checks);
    try std.testing.expectEqual(@as(usize, 0), rt.ephemeron_calls);
    try std.testing.expectEqual(@as(usize, 0), rt.after_weak_calls);

    rt.has_weak_work = true;
    heap.collect();
    try std.testing.expectEqual(@as(usize, 2), rt.weak_work_checks);
    try std.testing.expect(rt.ephemeron_calls > 0);
    try std.testing.expect(rt.after_weak_calls > 0);
}

test "mark-sweep: ephemeron values stay live only while keys are live" {
    const a = std.testing.allocator;
    var rt = EphRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(EphRT).init(a, &rt);
    defer heap.deinit();

    const N = EphRT.Node;
    const T = EphRT.Table;
    const key = try heap.create(N, .node);
    key.* = .{ .id = 1 };
    const value = try heap.create(N, .node);
    value.* = .{ .id = 2 };
    const dead_key = try heap.create(N, .node);
    dead_key.* = .{ .id = 3 };
    const dead_value = try heap.create(N, .node);
    dead_value.* = .{ .id = 4 };
    const table = try heap.create(T, .table);
    table.* = .{};
    try table.entries.append(a, .{ .key = key, .value = value });
    try table.entries.append(a, .{ .key = dead_key, .value = dead_value });
    try rt.roots.append(a, table);
    try rt.roots.append(a, key);

    heap.collect();
    try std.testing.expectEqual(@as(usize, 3), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 1), table.entries.items.len);
    try std.testing.expectEqual(key, table.entries.items[0].key.?);
    try std.testing.expectEqual(value, table.entries.items[0].value.?);

    _ = rt.roots.pop();
    heap.collect();
    try std.testing.expectEqual(@as(usize, 1), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 0), table.entries.items.len);
}

test "nursery binding-selected old ephemerons retain values only for live keys" {
    const a = std.testing.allocator;
    var rt = EphRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(EphRT).init(a, &rt);
    defer heap.deinit();
    heap.setNurseryEnabled(true);

    const key = try heap.create(EphRT.Node, .node);
    key.* = .{ .id = 1 };
    const table = try heap.create(EphRT.Table, .table);
    table.* = .{};
    try rt.roots.append(a, table);
    try rt.roots.append(a, key);
    heap.collectYoung(); // establish old table + old live key

    const value = try heap.create(EphRT.Node, .node);
    value.* = .{ .id = 2 };
    const dead_key = try heap.create(EphRT.Node, .node);
    dead_key.* = .{ .id = 3 };
    const dead_value = try heap.create(EphRT.Node, .node);
    dead_value.* = .{ .id = 4 };
    try table.entries.append(a, .{ .key = key, .value = value });
    try table.entries.append(a, .{ .key = dead_key, .value = dead_value });
    heap.collectYoung();

    try std.testing.expectEqual(@as(usize, 3), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 0), heap.young_cells);
    try std.testing.expectEqual(@as(usize, 1), table.entries.items.len);
    try std.testing.expectEqual(key, table.entries.items[0].key.?);
    try std.testing.expectEqual(value, table.entries.items[0].value.?);
    try std.testing.expectEqual(@as(usize, 2), rt.finalized.items.len);
}

test "incremental mark: stepped drain matches stop-the-world reachability" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();

    const N = TestRT.Node;
    // root -> A <-> B (cycle); B -weak-> C; D is unreferenced garbage.
    const A = try heap.create(N, .node);
    A.* = .{ .id = 1 };
    const B = try heap.create(N, .node);
    B.* = .{ .id = 2 };
    const C = try heap.create(N, .node);
    C.* = .{ .id = 3 };
    const D = try heap.create(N, .node);
    D.* = .{ .id = 4 };
    A.strong = B;
    B.strong = A;
    B.weak = C;
    try rt.roots.append(a, A);

    // Drive marking in single-cell steps with no mutation between them: the
    // result must match `collect()` — A,B kept by the cycle/root; C (weak-only)
    // and D (garbage) swept.
    heap.startMarking();
    var steps: usize = 0;
    while (!heap.markStep(1)) : (steps += 1) {}
    heap.finishMarking();
    try std.testing.expectEqual(@as(usize, 2), heap.live_cells);
    try std.testing.expect(!heap.marking.load(.acquire));
}

test "incremental mark: insertion barrier saves a cell reparented behind a black object" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();

    const N = TestRT.Node;
    const holder = try heap.create(N, .node); // a root → traced black
    holder.* = .{ .id = 1 };
    const donor = try heap.create(N, .node); // not a root → would be swept
    donor.* = .{ .id = 2 };
    const orphan = try heap.create(N, .node); // reachable only via donor
    orphan.* = .{ .id = 3 };
    donor.strong = orphan;
    try rt.roots.append(a, holder);

    // Snapshot the roots, then fully trace the grey set: `holder` goes black
    // (strong == null). `donor`/`orphan` stay white (donor isn't a root).
    heap.startMarking();
    _ = heap.markStep(0);

    // Mutator hides `orphan` behind the already-black `holder` and drops the
    // only other path. The store fires the insertion barrier, shading `orphan`.
    holder.strong = orphan;
    heap.writeBarrier(orphan);
    donor.strong = null;

    heap.finishMarking();

    // holder + orphan survive (orphan via the barrier); donor is swept.
    try std.testing.expectEqual(@as(usize, 2), heap.live_cells);
    try std.testing.expect(holder.strong == orphan);
    try std.testing.expectEqual(@as(usize, 1), rt.finalized.items.len);
    try std.testing.expectEqual(@as(u32, 2), rt.finalized.items[0]); // donor
}

test "incremental mark: exact managed owner barrier preserves an inserted child" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();

    const N = TestRT.Node;
    const holder = try heap.create(N, .node);
    holder.* = .{ .id = 1 };
    const donor = try heap.create(N, .node);
    donor.* = .{ .id = 2 };
    const orphan = try heap.create(N, .node);
    orphan.* = .{ .id = 3 };
    donor.strong = orphan;
    try rt.roots.append(a, holder);

    heap.startMarking();
    _ = heap.markStep(0);

    holder.strong = orphan;
    heap.writeBarrierFromManaged(holder, orphan);
    donor.strong = null;

    heap.finishMarking();

    try std.testing.expectEqual(@as(usize, 2), heap.live_cells);
    try std.testing.expect(holder.strong == orphan);
    try std.testing.expectEqual(@as(usize, 1), rt.finalized.items.len);
    try std.testing.expectEqual(@as(u32, 2), rt.finalized.items[0]);
}

test "incremental mark: cells allocated mid-cycle are born grey and survive" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    defer heap.deinit();

    const N = TestRT.Node;
    const root = try heap.create(N, .node);
    root.* = .{ .id = 1 };
    try rt.roots.append(a, root);

    heap.startMarking();
    _ = heap.markStep(0); // root is black now

    // A cell allocated during marking is born grey (marked + queued), so it
    // survives this cycle and its creation-time fields would be traced.
    const fresh = try heap.create(N, .node);
    fresh.* = .{ .id = 2 };
    try std.testing.expect(Heap(TestRT).headerFlagLoad(
        Heap(TestRT).headerOf(fresh),
        Heap(TestRT).header_marked,
        .monotonic,
    ));

    heap.finishMarking();
    try std.testing.expectEqual(@as(usize, 2), heap.live_cells); // root + fresh
    try std.testing.expectEqual(@as(usize, 0), rt.finalized.items.len);
}

test "concurrent mark: a mutator racing the marker behind the barrier loses no live cell" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    // Concurrent marking: marker (mark_stack) and mutator (barrier_buf) grow GC
    // scratch from `aux` on different threads, so it must be thread-safe — the
    // page allocator is. Cell slabs stay on the (single-threaded here) backing.
    heap.setAuxAllocator(std.heap.page_allocator);
    defer heap.deinit();

    const N = TestRT.Node;
    // A rooted `holder`, plus a pool of initially-unreferenced nodes that the
    // mutator will reparent under `holder` *during* the concurrent mark — the
    // classic concurrent hazard (a white cell hidden behind a black root).
    const holder = try heap.create(N, .node);
    holder.* = .{ .id = 0 };
    try rt.roots.append(a, holder);

    const pool_n = 2000;
    var pool: [pool_n]*N = undefined;
    for (&pool, 0..) |*slot, i| {
        const node = try heap.create(N, .node);
        node.* = .{ .id = @intCast(i + 1) };
        slot.* = node;
    }

    // Begin the concurrent mark with the world stopped (only `holder` is rooted;
    // the pool is all white).
    heap.beginConcurrentMark();

    const Shared = struct {
        heap: *Heap(TestRT),
        holder: *N,
        pool: []*N,
        done: std.atomic.Value(bool) = .init(false),

        fn mutate(s: *@This()) void {
            // Chain each pool node under holder and fire the insertion barrier —
            // exactly what the engine's store funnels do. The marker is running
            // concurrently; without the barrier these would be swept.
            var prev = s.holder;
            for (s.pool) |node| {
                // Atomic store pairs with the marker's atomic load of `strong`
                // (relaxed: ordering of the new edge is established by the
                // barrier, not this store), then shade the new target grey.
                @atomicStore(?*N, &prev.strong, node, .monotonic);
                s.heap.writeBarrier(node);
                prev = node;
            }
            s.done.store(true, .release);
        }
        fn markLoop(s: *@This()) void {
            // Drain rounds until the mutator is done and a final round is clean.
            while (true) {
                const quiescent = s.heap.concurrentMarkRound();
                if (s.done.load(.acquire) and quiescent) break;
                std.atomic.spinLoopHint();
            }
        }
    };
    var shared = Shared{ .heap = &heap, .holder = holder, .pool = pool[0..] };

    const mut = try std.Thread.spawn(.{}, Shared.mutate, .{&shared});
    const mk = try std.Thread.spawn(.{}, Shared.markLoop, .{&shared});
    mut.join();
    mk.join();

    // World stopped again: finish + sweep.
    heap.finishConcurrentMark();

    // Every pool node is now reachable holder→n1→n2→…; none must have been
    // swept, and nothing should have been finalized.
    try std.testing.expectEqual(@as(usize, pool_n + 1), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 0), rt.finalized.items.len);
    // Spot-check the chain is intact.
    var cur: ?*N = holder.strong;
    var count: usize = 0;
    while (cur) |c| : (cur = c.strong) count += 1;
    try std.testing.expectEqual(@as(usize, pool_n), count);
}

test "concurrent mark: cells never reparented onto a root are still swept" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    heap.setAuxAllocator(std.heap.page_allocator); // thread-safe GC scratch (see above)
    defer heap.deinit();

    const N = TestRT.Node;
    const root = try heap.create(N, .node);
    root.* = .{ .id = 0 };
    try rt.roots.append(a, root);
    // A garbage node that the mutator touches but never links to a root.
    const garbage = try heap.create(N, .node);
    garbage.* = .{ .id = 99 };

    heap.beginConcurrentMark();
    _ = heap.concurrentMarkRound(); // marker drains the rooted graph
    // Mutator writes garbage's own field (no barrier needed for self) — it stays
    // unreachable from any root.
    garbage.strong = null;
    heap.finishConcurrentMark();

    // root survives; garbage is collected.
    try std.testing.expectEqual(@as(usize, 1), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 1), rt.finalized.items.len);
    try std.testing.expectEqual(@as(u32, 99), rt.finalized.items[0]);
}

test "concurrent mark: cells allocated mid-cycle are deferred (born_concurrent) and survive" {
    // The production-driver hazard: the mutator ALLOCATES during the concurrent
    // window. Born cells must NOT be traced by the marker while the mutator is
    // still initializing them — they accumulate in `born_concurrent` and are
    // folded in (and traced) at the world-stopped finish. Here the mutator
    // creates a chain of fresh nodes under a rooted holder while the marker runs;
    // all must survive, and a fresh node never linked anywhere also survives this
    // cycle (born marked).
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    heap.setAuxAllocator(std.heap.page_allocator);
    defer heap.deinit();

    const N = TestRT.Node;
    const holder = try heap.create(N, .node);
    holder.* = .{ .id = 0 };
    try rt.roots.append(a, holder);

    heap.beginConcurrentMark();

    const Shared = struct {
        heap: *Heap(TestRT),
        holder: *N,
        done: std.atomic.Value(bool) = .init(false),
        fn mutate(s: *@This()) void {
            // Allocate fresh nodes mid-cycle and chain them under the rooted
            // holder. `create` defers each to born_concurrent (born marked); the
            // link store fires the barrier too. A real init writes the payload
            // right after create — exactly the half-built window the deferral
            // protects.
            var prev = s.holder;
            var i: u32 = 1;
            while (i <= 1000) : (i += 1) {
                const node = s.heap.create(N, .node) catch return;
                node.* = .{ .id = i };
                @atomicStore(?*N, &prev.strong, node, .monotonic);
                s.heap.writeBarrier(node);
                prev = node;
            }
            s.done.store(true, .release);
        }
        fn markLoop(s: *@This()) void {
            while (true) {
                const q = s.heap.concurrentMarkRound();
                if (s.done.load(.acquire) and q) break;
                std.atomic.spinLoopHint();
            }
        }
    };
    var shared = Shared{ .heap = &heap, .holder = holder };
    const mut = try std.Thread.spawn(.{}, Shared.mutate, .{&shared});
    const mk = try std.Thread.spawn(.{}, Shared.markLoop, .{&shared});
    mut.join();
    mk.join();
    heap.finishConcurrentMark();

    // holder + 1000 fresh nodes all survive.
    try std.testing.expectEqual(@as(usize, 1001), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 0), rt.finalized.items.len);
    var cur: ?*N = holder.strong;
    var count: usize = 0;
    while (cur) |c| : (cur = c.strong) count += 1;
    try std.testing.expectEqual(@as(usize, 1000), count);
}

test "parallel: multiple mutators allocate concurrently without corrupting the heap" {
    // The first GIL-removal prerequisite: cell allocation must be thread-safe so
    // several mutators can `create` at once. With `setParallel`, the all-list
    // prepend + counters run under `alloc_lock` and `backing` is the thread-safe
    // page allocator. N threads each allocate M cells; afterward the heap must
    // hold exactly N*M cells (no lost/double-linked nodes), every cell's header
    // magic intact, and the `all` list length must match the counter.
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    heap.setAuxAllocator(std.heap.page_allocator);
    heap.backing = std.heap.page_allocator; // thread-safe cell slabs for parallel alloc
    heap.setParallel(true);
    defer heap.deinit();

    const N = TestRT.Node;
    const threads = 8;
    const per = 2000;
    const Worker = struct {
        heap: *Heap(TestRT),
        fn run(s: *@This()) void {
            var i: usize = 0;
            while (i < per) : (i += 1) {
                const node = s.heap.create(N, .node) catch return;
                node.* = .{ .id = @intCast(i) }; // fully initialize (no marker running)
            }
        }
    };
    var w = Worker{ .heap = &heap };
    var pool: [threads]std.Thread = undefined;
    for (&pool) |*t| t.* = try std.Thread.spawn(.{}, Worker.run, .{&w});
    for (&pool) |*t| t.join();

    // Exactly N*M live cells, and the all-list length agrees with the counter —
    // a lost/double-linked prepend or a torn counter (the race a non-thread-safe
    // create would cause) would make these mismatch.
    try std.testing.expectEqual(@as(usize, threads * per), heap.live_cells);
    var walked: usize = 0;
    var it = heap.all;
    while (it) |hdr| : (it = hdr.next) walked += 1;
    try std.testing.expectEqual(@as(usize, threads * per), walked);
}

test "createBatch publishes nursery cells under one allocation lock" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    heap.setParallel(true);
    heap.setNurseryEnabled(true);
    defer heap.deinit();

    const N = TestRT.Node;
    var nodes: [32]*N = undefined;
    const locks_before = heap.alloc_lock_acquisitions_for_testing;
    try std.testing.expectEqual(nodes.len, try heap.createBatch(N, .node, &nodes));
    try std.testing.expectEqual(locks_before + 1, heap.alloc_lock_acquisitions_for_testing);
    for (nodes, 0..) |node, i| node.* = .{ .id = @intCast(i) };

    const bytes = nodes.len * Heap(TestRT).cellAllocationBytes(N);
    try std.testing.expectEqual(nodes.len, heap.live_cells);
    try std.testing.expectEqual(bytes, heap.bytes_live);
    try std.testing.expectEqual(nodes.len, heap.young_cells);
    try std.testing.expectEqual(bytes, heap.young_bytes);
    var walked: usize = 0;
    var it = heap.all;
    while (it) |hdr| : (it = hdr.next) walked += 1;
    try std.testing.expectEqual(nodes.len, walked);
}

test "createBatch keeps a short embedder batch on the compact publication path" {
    const a = std.testing.allocator;
    var rt = BatchAllocTestRT{ .allocator = a, .limit = 3 };
    const H = Heap(BatchAllocTestRT);
    var heap = H.init(a, &rt);
    heap.setParallel(true);
    heap.setNurseryEnabled(true);
    defer heap.deinit();
    rt.alloc_lock_probe = &heap.alloc_lock;

    const N = BatchAllocTestRT.Node;
    var nodes: [8]*N = undefined;
    const locks_before = heap.alloc_lock_acquisitions_for_testing;
    try std.testing.expectEqual(@as(usize, 3), try heap.createBatch(N, .node, &nodes));
    try std.testing.expectEqual(@as(usize, 1), rt.batch_calls);
    try std.testing.expectEqual(@as(usize, 1), rt.publish_batch_calls);
    try std.testing.expectEqual(@as(usize, 3), rt.published_cells);
    try std.testing.expect(rt.published_while_alloc_locked);
    try std.testing.expectEqual(locks_before + 1, heap.alloc_lock_acquisitions_for_testing);
    try std.testing.expectEqual(nodes[2], @as(*N, @ptrCast(@alignCast(H.payloadOf(heap.all.?)))));
    try std.testing.expectEqual(nodes[1], @as(*N, @ptrCast(@alignCast(H.payloadOf(heap.all.?.next.?)))));
    try std.testing.expectEqual(nodes[0], @as(*N, @ptrCast(@alignCast(H.payloadOf(heap.all.?.next.?.next.?)))));
    for (nodes[0..3], 0..) |node, i| node.* = .{ .id = @intCast(i) };
    try std.testing.expectEqual(@as(usize, 3), heap.live_cells);
    try std.testing.expectEqual(@as(usize, 3), heap.young_cells);
}

test "large owned createBatch splices before publishing outside the allocation lock" {
    const a = std.testing.allocator;
    var rt = BatchAllocTestRT{ .allocator = a };
    const H = Heap(BatchAllocTestRT);
    var heap = H.init(a, &rt);
    heap.setParallel(true);
    heap.setNurseryEnabled(true);
    defer heap.deinit();
    rt.alloc_lock_probe = &heap.alloc_lock;

    const N = BatchAllocTestRT.Node;
    var nodes: [64]*N = undefined;
    const locks_before = heap.alloc_lock_acquisitions_for_testing;
    try std.testing.expectEqual(nodes.len, try heap.createBatch(N, .node, &nodes));
    try std.testing.expectEqual(@as(usize, 1), rt.publish_batch_calls);
    try std.testing.expectEqual(nodes.len, rt.published_cells);
    try std.testing.expect(!rt.published_while_alloc_locked);
    try std.testing.expectEqual(locks_before + 1, heap.alloc_lock_acquisitions_for_testing);
    try std.testing.expectEqual(nodes[nodes.len - 1], @as(*N, @ptrCast(@alignCast(H.payloadOf(heap.all.?)))));
    try std.testing.expectEqual(nodes.len, heap.live_cells);
    try std.testing.expectEqual(nodes.len, heap.young_cells);
}

test "parallel owned batches publish through private aggregate shards" {
    var rt = ShardedBatchTestRT{};
    const H = Heap(ShardedBatchTestRT);
    var heap = H.init(std.heap.page_allocator, &rt);
    heap.setNurseryEnabled(true);
    heap.setParallel(true);
    defer heap.deinit();

    const threads = 8;
    const batches_per_thread = 8;
    const batch_size = 64;
    const Worker = struct {
        heap: *H,
        lane: usize,

        fn run(self: *@This()) void {
            var nodes: [batch_size]*ShardedBatchTestRT.Node = undefined;
            for (0..batches_per_thread) |batch| {
                const count = self.heap.createBatch(ShardedBatchTestRT.Node, .node, &nodes) catch unreachable;
                std.debug.assert(count == nodes.len);
                for (nodes, 0..) |node, i| node.* = .{ .id = self.lane * 10_000 + batch * batch_size + i };
            }
        }
    };
    var workers: [threads]Worker = undefined;
    var pool: [threads]std.Thread = undefined;
    for (&pool, &workers, 0..) |*thread, *worker, lane| {
        worker.* = .{ .heap = &heap, .lane = lane };
        thread.* = try std.Thread.spawn(.{}, Worker.run, .{worker});
    }
    for (&pool) |*thread| thread.join();

    const expected = threads * batches_per_thread * batch_size;
    try std.testing.expectEqual(@as(usize, 0), heap.alloc_lock_acquisitions_for_testing);
    try std.testing.expectEqual(@as(usize, threads * batches_per_thread), heap.sharded_publications_for_testing.load(.monotonic));
    try std.testing.expectEqual(@as(?*H.Header, null), heap.all);
    const before = heap.accounting();
    try std.testing.expectEqual(@as(usize, expected), before.live_cells);
    try std.testing.expectEqual(expected * H.cellAllocationBytes(ShardedBatchTestRT.Node), before.live_bytes);

    heap.collectYoung();
    const after = heap.accounting();
    try std.testing.expectEqual(@as(usize, 0), after.live_cells);
    for (&rt.published) |*slot| try std.testing.expectEqual(@as(?*anyopaque, null), slot.load(.acquire));
}

test "parallel mark closes sharded publication before and after header publication" {
    var rt = ShardedBatchTestRT{};
    const H = Heap(ShardedBatchTestRT);
    var heap = H.init(std.heap.page_allocator, &rt);
    heap.setNurseryEnabled(true);
    heap.setParallel(true);
    defer heap.deinit();

    const batch_size = 64;
    const Publisher = struct {
        heap: *H,
        base: usize,

        fn run(self: *@This()) void {
            var nodes: [batch_size]*ShardedBatchTestRT.Node = undefined;
            const count = self.heap.createBatch(ShardedBatchTestRT.Node, .node, &nodes) catch unreachable;
            std.debug.assert(count == nodes.len);
            for (nodes, 0..) |node, i| node.* = .{ .id = self.base + i };
        }
    };
    const waitFor = struct {
        fn flag(value: *std.atomic.Value(bool)) void {
            while (!value.load(.acquire)) std.Thread.yield() catch {};
        }
    }.flag;

    // Raw reservations stay absent from the iterator. Once marking closes the
    // shard gate, the resumed publisher takes the existing born-grey fallback.
    rt.pause_allocate.store(true, .release);
    var raw_publisher = Publisher{ .heap = &heap, .base = 1000 };
    const raw_thread = try std.Thread.spawn(.{}, Publisher.run, .{&raw_publisher});
    waitFor(&rt.allocate_ready);
    heap.beginConcurrentMarkParallel();
    rt.release_allocate.store(true, .release);
    raw_thread.join();
    heap.abortConcurrentMarkParallel();
    try std.testing.expectEqual(@as(usize, batch_size), heap.accounting().live_cells);
    heap.collectYoung();

    // This publisher has initialized every header and is stopped immediately
    // before bitmap publication. The collector must observe its active shard
    // and cannot complete begin/whitening until publication and counters land.
    rt.pause_allocate.store(false, .release);
    rt.pause_publish.store(true, .release);
    var header_publisher = Publisher{ .heap = &heap, .base = 2000 };
    const header_thread = try std.Thread.spawn(.{}, Publisher.run, .{&header_publisher});
    waitFor(&rt.publish_ready);

    const Collector = struct {
        heap: *H,
        started: std.atomic.Value(bool) = .init(false),
        completed: std.atomic.Value(bool) = .init(false),

        fn run(self: *@This()) void {
            self.started.store(true, .release);
            self.heap.beginConcurrentMarkParallel();
            self.completed.store(true, .release);
        }
    };
    var collector = Collector{ .heap = &heap };
    const collector_thread = try std.Thread.spawn(.{}, Collector.run, .{&collector});
    waitFor(&collector.started);
    while (heap.sharded_publication_gate.load(.seq_cst)) std.Thread.yield() catch {};
    try std.testing.expect(!collector.completed.load(.acquire));
    rt.release_publish.store(true, .release);
    header_thread.join();
    collector_thread.join();
    try std.testing.expect(collector.completed.load(.acquire));
    heap.abortConcurrentMarkParallel();
    try std.testing.expectEqual(@as(usize, batch_size), heap.accounting().live_cells);
    heap.collectYoung();
}

test "createBatch falls back normally when the embedder batch is empty" {
    const a = std.testing.allocator;
    var rt = BatchAllocTestRT{ .allocator = a, .limit = 0 };
    var heap = Heap(BatchAllocTestRT).init(a, &rt);
    defer heap.deinit();

    const N = BatchAllocTestRT.Node;
    var nodes: [4]*N = undefined;
    try std.testing.expectEqual(nodes.len, try heap.createBatch(N, .node, &nodes));
    try std.testing.expectEqual(@as(usize, 1), rt.batch_calls);
    for (nodes, 0..) |node, i| node.* = .{ .id = @intCast(i) };
    try std.testing.expectEqual(nodes.len, heap.live_cells);
}

test "owned createBatch keeps the marking publication fallback" {
    const a = std.testing.allocator;
    var rt = BatchAllocTestRT{ .allocator = a };
    const H = Heap(BatchAllocTestRT);
    var heap = H.init(a, &rt);
    heap.setParallel(true);
    defer heap.deinit();
    rt.alloc_lock_probe = &heap.alloc_lock;
    heap.marking.store(true, .release);

    const N = BatchAllocTestRT.Node;
    var nodes: [4]*N = undefined;
    try std.testing.expectEqual(nodes.len, try heap.createBatch(N, .node, &nodes));
    var header = heap.all;
    var count: usize = 0;
    while (header) |h| : (header = h.next) {
        try std.testing.expect(Heap(BatchAllocTestRT).headerFlagLoad(
            h,
            Heap(BatchAllocTestRT).header_marked,
            .acquire,
        ));
        count += 1;
    }
    try std.testing.expectEqual(nodes.len, count);
    try std.testing.expectEqual(@as(usize, 1), rt.publish_batch_calls);
    try std.testing.expect(rt.published_while_alloc_locked);
}

test "createBatch publishes a partial prefix before preserving OOM order" {
    const a = std.testing.allocator;
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 2 });
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(failing.allocator(), &rt);
    var nodes: [4]*TestRT.Node = undefined;
    const count = try heap.createBatch(TestRT.Node, .node, &nodes);
    try std.testing.expectEqual(@as(usize, 2), count);
    for (nodes[0..count], 0..) |node, i| node.* = .{ .id = @intCast(i) };
    try std.testing.expectEqual(count, heap.live_cells);
    try std.testing.expectError(error.OutOfMemory, heap.createBatch(TestRT.Node, .node, nodes[count..]));
    try std.testing.expectEqual(count, heap.live_cells);
    heap.deinit();
    try std.testing.expectEqual(failing.allocations, failing.deallocations);
    try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
}

test "parallel: batched mutators preserve the all-list and amortize publication locks" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    heap.setAuxAllocator(std.heap.page_allocator);
    heap.backing = std.heap.page_allocator;
    heap.setParallel(true);
    defer heap.deinit();

    const N = TestRT.Node;
    const threads = 8;
    const per = 2048;
    const batch_size = 32;
    const Worker = struct {
        heap: *Heap(TestRT),
        fn run(s: *@This()) void {
            var created: usize = 0;
            var nodes: [batch_size]*N = undefined;
            while (created < per) {
                const count = @min(batch_size, per - created);
                const published = s.heap.createBatch(N, .node, nodes[0..count]) catch return;
                for (nodes[0..published], 0..) |node, i|
                    node.* = .{ .id = @intCast(created + i) };
                created += published;
            }
        }
    };
    var worker = Worker{ .heap = &heap };
    var pool: [threads]std.Thread = undefined;
    for (&pool) |*thread| thread.* = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    for (&pool) |*thread| thread.join();

    try std.testing.expectEqual(@as(usize, threads * per), heap.live_cells);
    try std.testing.expectEqual(@as(usize, threads * (per / batch_size)), heap.alloc_lock_acquisitions_for_testing);
    var walked: usize = 0;
    var it = heap.all;
    while (it) |hdr| : (it = hdr.next) walked += 1;
    try std.testing.expectEqual(@as(usize, threads * per), walked);
}

test "parallel concurrent mark: collector marks + sweeps while mutators allocate, no GIL" {
    // The mid-script parallel-GC mechanism (issue #1 M3): a concurrent mark runs
    // while several mutators allocate and build object graphs on other threads
    // with no GIL. Exercises the three parallel-safe paths together —
    //   - `beginConcurrentMarkParallel`: whitens + arms the barrier under
    //     `alloc_lock` while peers are mid-`create`;
    //   - `concurrentMarkRound`: the marker drains its stack + the peers'
    //     `barrier_buf` hand-off while they keep mutating;
    //   - `sweepPhase` under `parallel`: the final sweep unlinks dead cells from
    //     `all` while (quiesced-at-finish) peers could still hold `alloc_lock`.
    // Determinism: each mutator builds a PRIVATE chain under its OWN rooted
    // holder, so a chain node is always reachable from a root and survives; the
    // pre-begin garbage is never linked, stays white, and is reclaimed. TSan
    // proves the whiten/mark/sweep vs. parallel-`create`/`writeBarrier` paths
    // are race-free.
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.roots.deinit(a);
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    heap.setAuxAllocator(std.heap.page_allocator);
    heap.backing = std.heap.page_allocator; // thread-safe cell slabs
    heap.setParallel(true);
    defer heap.deinit();

    const N = TestRT.Node;
    const threads = 4;
    const chain_per = 500;
    const garbage_n = 800;

    // Rooted holders, one per mutator (so chains never share a slot).
    var holders: [threads]*N = undefined;
    for (&holders, 0..) |*slot, i| {
        const h = try heap.create(N, .node);
        h.* = .{ .id = @intCast(1000 + i) };
        slot.* = h;
        try rt.roots.append(a, h);
    }
    // Pre-begin garbage: white, never linked → must be swept.
    var g: usize = 0;
    while (g < garbage_n) : (g += 1) {
        const node = try heap.create(N, .node);
        node.* = .{ .id = 7 };
    }

    const Shared = struct {
        heap: *Heap(TestRT),
        holders: []*N,
        started: std.atomic.Value(u32) = .init(0),
        done: std.atomic.Value(u32) = .init(0),

        fn markLoop(s: *@This()) void {
            // Drain rounds until every mutator is done and a final round is
            // clean (nothing left grey / handed off).
            while (true) {
                const q = s.heap.concurrentMarkRound();
                if (s.done.load(.acquire) == threads and q) break;
                std.atomic.spinLoopHint();
            }
        }
    };
    const Worker = struct {
        s: *Shared,
        idx: usize,
        fn run(w: *@This()) void {
            _ = w.s.started.fetchAdd(1, .release);
            // Build a private chain of fresh nodes under this thread's holder.
            // Each `create` is a parallel allocation (born grey once the mark is
            // armed); the link store fires the insertion barrier.
            var prev = w.s.holders[w.idx];
            var i: u32 = 0;
            while (i < chain_per) : (i += 1) {
                const node = w.s.heap.create(N, .node) catch return;
                node.* = .{ .id = i };
                // Release store: publishes `node`'s initialization to the marker,
                // which reads this slot with an acquire load (see `TestRT.trace`).
                @atomicStore(?*N, &prev.strong, node, .release);
                w.s.heap.writeBarrier(node);
                prev = node;
            }
            _ = w.s.done.fetchAdd(1, .release);
        }
    };
    var shared = Shared{ .heap = &heap, .holders = holders[0..] };

    // Spawn mutators first; begin the mark while they are already allocating, so
    // the whiten pass races live `create` calls.
    var workers: [threads]Worker = undefined;
    for (&workers, 0..) |*w, i| w.* = .{ .s = &shared, .idx = i };
    var mpool: [threads]std.Thread = undefined;
    for (&mpool, 0..) |*t, i| t.* = try std.Thread.spawn(.{}, Worker.run, .{&workers[i]});
    // Let at least one mutator get into its allocation loop before arming.
    while (shared.started.load(.acquire) == 0) std.atomic.spinLoopHint();
    heap.beginConcurrentMarkParallel();
    const mk = try std.Thread.spawn(.{}, Shared.markLoop, .{&shared});

    for (&mpool) |*t| t.join();
    mk.join();
    // Mutators quiesced (joined): finish + sweep.
    heap.finishConcurrentMark();

    // Every holder + its full chain survives; the pre-begin garbage is gone.
    try std.testing.expectEqual(@as(usize, threads * (1 + chain_per)), heap.live_cells);
    try std.testing.expectEqual(@as(usize, garbage_n), rt.finalized.items.len);
    for (holders) |h| {
        var cur: ?*N = h.strong;
        var count: usize = 0;
        while (cur) |c| : (cur = c.strong) count += 1;
        try std.testing.expectEqual(@as(usize, chain_per), count);
    }
}

test "parallel concurrent mark: stale barrier append after abort is ignored" {
    const a = std.testing.allocator;
    var rt = TestRT{};
    defer rt.finalized.deinit(a);

    var heap = Heap(TestRT).init(a, &rt);
    heap.setAuxAllocator(std.heap.page_allocator);
    heap.setParallel(true);
    defer heap.deinit();

    const node = try heap.create(TestRT.Node, .node);
    node.* = .{ .id = 42 };
    const hdr = Heap(TestRT).headerOf(node);
    Heap(TestRT).headerFlagStore(hdr, Heap(TestRT).header_marked, false, .monotonic);
    heap.marked_count = 0;
    heap.marking.store(true, .release);
    heap.concurrent.store(true, .release);

    heap.lockBarrier();
    const Worker = struct {
        heap: *Heap(TestRT),
        node: *TestRT.Node,
        fn run(w: *@This()) void {
            w.heap.writeBarrier(w.node);
        }
    };
    var worker = Worker{ .heap = &heap, .node = node };
    const t = try std.Thread.spawn(.{}, Worker.run, .{&worker});

    while (!Heap(TestRT).headerFlagLoad(hdr, Heap(TestRT).header_marked, .acquire)) std.atomic.spinLoopHint();
    heap.marking.store(false, .release);
    heap.concurrent.store(false, .release);
    heap.barrier_buf.clearRetainingCapacity();
    heap.unlockBarrier();
    t.join();

    try std.testing.expectEqual(@as(usize, 0), heap.barrier_buf.items.len);
}
