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
//!   fn traceEphemeron(ctx: *B, cell: *anyopaque, kind: Kind, v: anytype) void;
//!   fn afterWeak(ctx: *B, cell: *anyopaque, kind: Kind) void;
//!   fn traceOldOnMinor(kind: Kind) bool;
//!
//! Inside trace/traceRoots the binding calls `v.mark(ptr)` for each strong
//! reference and `v.markWeak(&slot)` for each weak slot (`*?*anyopaque`).
//! `traceEphemeron` may call `v.isMarked(key)` and then `v.mark(value)` for
//! WeakMap-style key/value edges.

const std = @import("std");
const builtin = @import("builtin");

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
        pub const min_nursery_threshold_bytes: usize = 64 * 1024;
        pub const default_nursery_threshold_bytes: usize = 256 * 1024;

        /// One machine-word-ish header in front of every cell: the all-cells
        /// list link, the payload size (to free), the kind tag (to dispatch
        /// trace/finalize without RTTI), and the mark bit.
        const header_magic: u64 = 0x7a67_6763_5f68_6561;
        pub const Header = struct {
            magic: u64,
            next: ?*Header,
            size: usize,
            kind: Kind,
            marked: bool,
            young: bool,
            remembered_owner: bool,
            remembered_target: bool,
        };

        /// Fixed offset from a header to its payload. 16-byte aligned so any
        /// cell whose `@alignOf <= 16` (every normal Zig struct on 64-bit) is
        /// correctly aligned, and `payload - header_stride` recovers the header
        /// in O(1) regardless of cell type.
        const header_stride = std.mem.alignForward(usize, @sizeOf(Header), 16);

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
        live_cells: usize = 0,
        bytes_live: usize = 0,
        /// Collect when `bytes_live` crosses this; reset to 2× live after a
        /// collection (a simple allocation-rate-agnostic growth policy).
        threshold_bytes: usize = 64 * 1024,
        mark_stack: std.ArrayListUnmanaged(*Header) = .empty,
        weak_slots: std.ArrayListUnmanaged(*?*anyopaque) = .empty,
        marked_count: usize = 0,
        collections: usize = 0,
        full_collections: usize = 0,
        minor_collections: usize = 0,
        promoted_cells: usize = 0,
        promoted_bytes: usize = 0,
        young_cells: usize = 0,
        young_bytes: usize = 0,
        last_minor_young_bytes: usize = 0,
        last_minor_reclaimed_bytes: usize = 0,
        last_minor_promoted_bytes: usize = 0,
        nursery_threshold_bytes: usize = default_nursery_threshold_bytes,
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
                if (v.heap.collection_kind == .minor and !@atomicLoad(bool, &h.young, .monotonic)) return;
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
                if (v.heap.collection_kind == .minor and !@atomicLoad(bool, &Self.headerOf(p).young, .monotonic)) return true;
                // Atomic: the parallel marker reads `marked` (here, in the sweep
                // phase, and in conservative marking) while a mutator's `claimMark`
                // write barrier CASes it (`.acq_rel`); a plain read races that. A
                // relaxed load is a plain mov — the marking-phase handshake already
                // orders it — so this is free.
                return @atomicLoad(bool, &Self.headerOf(p).marked, .monotonic);
            }

            /// Conservatively mark a machine word if it points at the payload
            /// of a managed cell. This is intentionally opt-in: precise
            /// embedders should keep using `mark`, while runtimes that need to
            /// root native stacks can scan a stack/register spill range without
            /// teaching the collector about their frame layout.
            pub fn markConservativeWord(v: *Visitor, word: usize) void {
                const h = v.heap.headerForInteriorAddress(word) orelse return;
                if (v.heap.collection_kind == .minor and !@atomicLoad(bool, &h.young, .monotonic)) return;
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

        /// Enable the non-moving one-cycle nursery. New cells are young; a minor
        /// collection reclaims unreachable young cells and immediately tenures
        /// every survivor. Existing cells stay old, so enabling this after heap
        /// initialization is safe.
        pub fn setNurseryEnabled(self: *Self, enabled: bool) void {
            std.debug.assert(!self.marking.load(.acquire));
            self.nursery_enabled = enabled;
        }

        fn doubledBytes(bytes: usize) usize {
            const max = std.math.maxInt(usize);
            if (bytes > max / 2) return max;
            return bytes * 2;
        }

        fn nextNurseryThreshold(self: *Self, promoted_bytes: usize) usize {
            const promoted_target = @max(min_nursery_threshold_bytes, doubledBytes(promoted_bytes));
            const decay_floor = @max(min_nursery_threshold_bytes, self.nursery_threshold_bytes / 2);
            return @max(promoted_target, decay_floor);
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

        fn syncAllocMetadata(self: *Self) bool {
            return self.parallel or self.concurrent_marker_metadata;
        }

        fn headerForPayloadSlowLocked(self: *Self, payload: *anyopaque) ?*Header {
            var it = self.all;
            while (it) |h| : (it = h.next) {
                if (payloadOf(h) == payload) return h;
            }
            return null;
        }

        fn headerForPayload(self: *Self, payload: *anyopaque) ?*Header {
            if (self.syncAllocMetadata()) self.lockAlloc();
            defer if (self.syncAllocMetadata()) self.unlockAlloc();
            if (self.payload_index.get(payloadKey(payload))) |h| return h;
            return self.headerForPayloadSlowLocked(payload);
        }

        fn indexPayloadLocked(self: *Self, h: *Header) void {
            self.payload_index.put(self.backing, payloadKey(payloadOf(h)), h) catch {};
        }

        fn unindexPayloadLocked(self: *Self, h: *Header) void {
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
            if (self.collection_kind == .minor and !@atomicLoad(bool, &headerOf(ptr).young, .monotonic)) return true;
            return @atomicLoad(bool, &headerOf(ptr).marked, .monotonic); // vs concurrent claimMark CAS
        }

        fn headerForInteriorAddress(self: *Self, address: usize) ?*Header {
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
            var it = self.all;
            while (it) |h| : (it = h.next) {
                const start = @intFromPtr(payloadOf(h));
                // A zero-size payload would make an empty extent that no interior
                // address can fall into; record at least one byte so an exact
                // payload pointer still resolves.
                const size = if (h.size == 0) 1 else h.size;
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

        /// Allocate a GC-managed cell of type `T` tagged `kind`. The returned
        /// pointer is uninitialized payload; the caller writes it before the
        /// next safepoint (so a collection never traces a half-built cell).
        pub fn create(self: *Self, comptime T: type, kind: Kind) !*T {
            comptime std.debug.assert(@alignOf(T) <= 16);
            const total = header_stride + @sizeOf(T);
            // The slab alloc happens before the lock: `backing` is thread-safe in
            // `parallel` mode, and `h` is private until linked into `all` below.
            const slab = self.backing.alignedAlloc(u8, .@"16", total) catch |err| blk: {
                if (err == error.OutOfMemory and @hasDecl(Binding, "recoverAllocationFailure")) {
                    if (Binding.recoverAllocationFailure(self.ctx)) {
                        break :blk try self.backing.alignedAlloc(u8, .@"16", total);
                    }
                }
                return err;
            };
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
            const born_grey = self.marking.load(.acquire);
            // Field-wise init (not a struct literal) so `marked` is written
            // *atomically*: under a parallel concurrent mark the marker may
            // `claimMark` this born-grey cell (an atomic CAS) the instant a peer
            // links it behind a traced object, and a non-atomic store racing
            // that CAS is a data race. The CAS only fails (born grey ⇒ already
            // marked), so the marker never traces the half-built payload — the
            // other header fields stay private until the world-stopped finish.
            // `magic`/`next`/`size`/`kind` are only read at that finish (or by
            // sweep under `alloc_lock`), so they need no atomic.
            h.magic = header_magic;
            h.next = self.all;
            h.size = @sizeOf(T);
            h.kind = kind;
            @atomicStore(bool, &h.marked, born_grey, .release);
            @atomicStore(bool, &h.young, self.nursery_enabled, .release);
            @atomicStore(bool, &h.remembered_owner, false, .release);
            @atomicStore(bool, &h.remembered_target, false, .release);
            self.all = h;
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
                    if (self.parallel) return @ptrCast(@alignCast(slab.ptr + header_stride));
                    self.marked_count += 1;
                    self.mark_stack.append(self.aux, h) catch {};
                }
            }
            return @ptrCast(@alignCast(slab.ptr + header_stride));
        }

        /// Collect if the heap has grown past the threshold. Call at safepoints
        /// (the engine's `(steps & 1023)` checkpoints) and after large allocs.
        pub fn maybeCollect(self: *Self) void {
            if (self.nursery_enabled and self.shouldCollectYoung()) {
                self.collectYoung();
            } else if (self.bytes_live >= self.threshold_bytes) {
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
            if (!@atomicLoad(bool, &h.young, .acquire)) self.rememberOwner(h);
        }

        fn rememberStrongStore(self: *Self, owner: ?*anyopaque, cell: ?*anyopaque) void {
            if (!self.nursery_enabled) return;
            const child_ptr = cell orelse return;
            const child = self.headerForPayload(child_ptr) orelse return;
            if (!@atomicLoad(bool, &child.young, .acquire)) return;
            if (owner) |owner_ptr| {
                if (self.headerForPayload(owner_ptr)) |parent| {
                    if (!@atomicLoad(bool, &parent.young, .acquire)) self.rememberOwner(parent);
                    return;
                }
            }
            // Compatibility path for embedders that only provide the child. It
            // is conservative but sound: retain that target for this nursery
            // cycle, then tenure it and discard the entry.
            if (@cmpxchgStrong(bool, &child.remembered_target, false, true, .acq_rel, .monotonic) != null) return;
            self.lockRemember();
            self.remembered_targets.append(self.aux, child) catch {
                @atomicStore(bool, &child.remembered_target, false, .release);
                self.nursery_force_full.store(true, .release);
            };
            self.unlockRemember();
        }

        fn rememberOwner(self: *Self, h: *Header) void {
            if (@cmpxchgStrong(bool, &h.remembered_owner, false, true, .acq_rel, .monotonic) != null) return;
            self.lockRemember();
            self.remembered_owners.append(self.aux, h) catch {
                @atomicStore(bool, &h.remembered_owner, false, .release);
                self.nursery_force_full.store(true, .release);
            };
            self.unlockRemember();
        }

        fn incrementalBarrier(self: *Self, cell: ?*anyopaque) void {
            if (!self.marking.load(.acquire)) return;
            const p = cell orelse return;
            const h = self.headerForPayload(p) orelse return;
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
                if (@cmpxchgStrong(bool, &h.marked, false, true, .acq_rel, .monotonic) != null) return false;
                _ = @atomicRmw(usize, &self.marked_count, .Add, 1, .monotonic);
                return true;
            }
            if (h.marked) return false;
            h.marked = true;
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
            while (self.alloc_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
        }
        inline fn unlockAlloc(self: *Self) void {
            self.alloc_lock.store(0, .release);
        }

        /// Begin an incremental mark: whiten all cells and grey the roots. The
        /// mutator then runs between `markStep`s with the `writeBarrier` active.
        pub fn startMarking(self: *Self) void {
            self.collection_kind = .full;
            var it = self.all;
            while (it) |h| : (it = h.next) h.marked = false;
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
            // Weak-slot registration can run while a concurrent marker traces
            // object cells and mutators grow their own side storage. Reserve the
            // per-cycle scratch up front, at the safepoint, so `markWeak` never
            // reallocates from `aux` mid-mark and reuses a page concurrently with
            // a mutator's object-backing write (the same allocator-reuse TSan
            // class the mark-stack reservation avoids).
            self.weak_slots.ensureTotalCapacity(self.aux, self.live_cells) catch {};
            self.unlockWeak();
            self.addr_index_built = false;
            self.marking.store(true, .release);
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
            self.sweepPhase(&v);
        }

        /// A full stop-the-world cycle: mark from roots, clear dead weak edges,
        /// sweep (finalizing) the white cells. Equivalent to
        /// `startMarking` + drain + `finishMarking`, kept as the default.
        pub fn collect(self: *Self) void {
            self.startMarking();
            _ = self.markStep(0); // drain fully
            self.finishMarking();
        }

        /// Run a stop-the-world nursery cycle. Old roots are not recursively
        /// rescanned; dirty old containers and conservative child-only barrier
        /// targets supply the old-to-young frontier. Every live young cell is
        /// tenured, so the nursery is empty when this returns.
        pub fn collectYoung(self: *Self) void {
            std.debug.assert(!self.marking.load(.acquire));
            std.debug.assert(!self.concurrent.load(.acquire));
            if (!self.nursery_enabled or self.young_cells == 0) return;
            if (self.nursery_force_full.load(.acquire)) {
                self.collect();
                return;
            }

            if (self.parallel) self.lockAlloc();
            self.collection_kind = .minor;
            var it = self.all;
            while (it) |h| : (it = h.next) {
                if (@atomicLoad(bool, &h.young, .monotonic)) @atomicStore(bool, &h.marked, false, .monotonic);
            }
            self.marked_count = 0;
            self.mark_stack.clearRetainingCapacity();
            self.mark_stack.ensureTotalCapacity(self.aux, self.young_cells) catch {};
            self.lockWeak();
            self.weak_slots.clearRetainingCapacity();
            // A minor cycle can still trace old roots/remembered owners that
            // register weak slots, so reserve to the full live-cell bound rather
            // than only the young-cell count.
            self.weak_slots.ensureTotalCapacity(self.aux, self.live_cells) catch {};
            self.unlockWeak();
            self.addr_index_built = false;
            self.marking.store(true, .release);
            if (self.parallel) self.unlockAlloc();

            var v = Visitor{ .heap = self };
            Binding.traceRoots(self.ctx, &v);
            // Bindings may identify mutable side-cell kinds whose post-creation
            // stores are not fully owner-barriered. Treat those old cells as
            // remembered owners for this cycle. The sweep already walks `all`,
            // so this adds only a kind check per old cell and traces edges solely
            // for the selected kinds.
            if (@hasDecl(Binding, "traceOldOnMinor")) {
                var old_it = self.all;
                while (old_it) |h| : (old_it = h.next) {
                    if (!@atomicLoad(bool, &h.young, .monotonic) and Binding.traceOldOnMinor(h.kind)) self.rememberOwner(h);
                }
            }
            self.lockRemember();
            for (self.remembered_owners.items) |h| {
                if (h.magic == header_magic and !@atomicLoad(bool, &h.young, .monotonic))
                    Binding.trace(payloadOf(h), h.kind, &v);
            }
            for (self.remembered_targets.items) |h| v.mark(payloadOf(h));
            self.unlockRemember();
            while (self.mark_stack.pop()) |h| Binding.trace(payloadOf(h), h.kind, &v);
            self.marking.store(false, .release);
            self.sweepPhase(&v);
            self.collection_kind = .full;
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
            self.collection_kind = .full;
            self.lockAlloc();
            var it = self.all;
            // Atomic store: a lagging peer's `claimMark` CAS (barrier path, no
            // `alloc_lock`) can touch the same `marked` byte concurrently. See
            // the whiten note in the doc comment above.
            while (it) |h| : (it = h.next) @atomicStore(bool, &h.marked, false, .monotonic);
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
            // See `startMarking`: weak-slot scratch is marker-side state, but it
            // must not allocate from `aux` while parallel mutators are growing
            // object backing stores.
            self.weak_slots.ensureTotalCapacity(self.aux, self.live_cells) catch {};
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
            self.sweepPhase(&v);
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
            self.sweepPhaseLocked(&v); // alloc_lock already held
            self.unlockAlloc();
            return true;
        }

        /// Whether live bytes have crossed the collection threshold, read under
        /// `alloc_lock` in `parallel` mode so a mid-script collector's safepoint
        /// check doesn't race a peer's `create` updating `bytes_live`.
        pub fn shouldCollect(self: *Self) bool {
            if (self.parallel) self.lockAlloc();
            defer if (self.parallel) self.unlockAlloc();
            return self.bytes_live >= self.threshold_bytes;
        }

        /// Whether tenured bytes alone have crossed the full-heap threshold.
        /// Generational embedders use this at quiescent boundaries so a large
        /// young batch receives a minor collection before it can force a full
        /// trace. Mid-script collectors that cannot run minor GC should continue
        /// using `shouldCollect()` over total bytes.
        pub fn shouldCollectOld(self: *Self) bool {
            if (self.parallel) self.lockAlloc();
            defer if (self.parallel) self.unlockAlloc();
            return self.bytes_live - self.young_bytes >= self.threshold_bytes;
        }

        /// Whether the nursery has reached its collection threshold, or a
        /// remembered-set allocation failure requires the next nursery request
        /// to fall back to a full collection.
        pub fn shouldCollectYoung(self: *Self) bool {
            if (!self.nursery_enabled) return false;
            if (self.nursery_force_full.load(.acquire)) return true;
            if (self.parallel) self.lockAlloc();
            defer if (self.parallel) self.unlockAlloc();
            return self.young_bytes >= self.nursery_threshold_bytes;
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
            if (self.parallel) self.lockAlloc();
            defer if (self.parallel) self.unlockAlloc();
            self.sweepPhaseLocked(v);
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

            // 2b. Ephemerons (WeakMap-style edges): if a marked table has a
            // marked key, its value becomes strong. Iterate to a fixed point so
            // values can keep further keys alive through chains of weak maps.
            if (@hasDecl(Binding, "traceEphemeron")) {
                while (true) {
                    const before = self.marked_count;
                    var eit = self.all;
                    while (eit) |h| : (eit = h.next) {
                        if (self.shouldProcessMarkedCell(h)) Binding.traceEphemeron(self.ctx, payloadOf(h), h.kind, v);
                    }
                    while (self.mark_stack.pop()) |h| {
                        Binding.trace(payloadOf(h), h.kind, v);
                    }
                    if (self.marked_count == before) break;
                }
            }

            // 3. weak edges whose target died are cleared *before* the sweep
            //    frees it, so no slot ever dangles.
            self.lockWeak();
            for (self.weak_slots.items) |slot| {
                if (slot.*) |target| {
                    if (!self.isLive(target)) slot.* = null;
                }
            }
            self.unlockWeak();

            if (@hasDecl(Binding, "afterWeak")) {
                var wit = self.all;
                while (wit) |h| : (wit = h.next) {
                    if (self.shouldProcessMarkedCell(h)) Binding.afterWeak(self.ctx, payloadOf(h), h.kind);
                }
            }

            // 4. sweep the white cells.
            const minor = self.collection_kind == .minor;
            var cycle_young_bytes: usize = 0;
            var cycle_reclaimed_young_bytes: usize = 0;
            var cycle_promoted_cells: usize = 0;
            var cycle_promoted_bytes: usize = 0;
            var prev: ?*Header = null;
            var cur = self.all;
            while (cur) |h| {
                const next = h.next;
                const young = @atomicLoad(bool, &h.young, .monotonic);
                const total = header_stride + h.size;
                if (minor and young) cycle_young_bytes += total;
                if (minor and !young) {
                    prev = h;
                } else if (@atomicLoad(bool, &h.marked, .monotonic)) {
                    if (young) {
                        @atomicStore(bool, &h.young, false, .release);
                        self.young_cells -= 1;
                        self.young_bytes -= total;
                        cycle_promoted_cells += 1;
                        cycle_promoted_bytes += total;
                    }
                    prev = h;
                } else {
                    Binding.finalize(self.ctx, payloadOf(h), h.kind);
                    self.unindexPayloadLocked(h);
                    if (prev) |p| p.next = next else self.all = next;
                    self.live_cells -= 1;
                    self.bytes_live -= total;
                    if (young) {
                        self.young_cells -= 1;
                        self.young_bytes -= total;
                        cycle_reclaimed_young_bytes += total;
                    }
                    const base: [*]align(16) u8 = @ptrCast(@alignCast(h));
                    self.backing.free(base[0..total]);
                }
                cur = next;
            }

            self.collections += 1;
            self.promoted_cells += cycle_promoted_cells;
            self.promoted_bytes += cycle_promoted_bytes;
            if (minor) {
                self.minor_collections += 1;
                self.last_minor_young_bytes = cycle_young_bytes;
                self.last_minor_reclaimed_bytes = cycle_reclaimed_young_bytes;
                self.last_minor_promoted_bytes = cycle_promoted_bytes;
                self.nursery_threshold_bytes = self.nextNurseryThreshold(cycle_promoted_bytes);
            } else {
                self.full_collections += 1;
                self.threshold_bytes = @max(64 * 1024, self.bytes_live * 2);
            }
            self.clearRemembered();
            self.nursery_force_full.store(false, .release);
        }

        fn shouldProcessMarkedCell(self: *Self, h: *Header) bool {
            if (self.collection_kind == .full) return @atomicLoad(bool, &h.marked, .monotonic);
            if (@atomicLoad(bool, &h.young, .monotonic)) return @atomicLoad(bool, &h.marked, .monotonic);
            return @atomicLoad(bool, &h.remembered_owner, .monotonic);
        }

        fn clearRemembered(self: *Self) void {
            self.lockRemember();
            for (self.remembered_owners.items) |h| @atomicStore(bool, &h.remembered_owner, false, .release);
            for (self.remembered_targets.items) |h| @atomicStore(bool, &h.remembered_target, false, .release);
            self.remembered_owners.clearRetainingCapacity();
            self.remembered_targets.clearRetainingCapacity();
            self.unlockRemember();
        }

        fn deinitImpl(self: *Self, free_cell_storage: bool) void {
            var cur = self.all;
            while (cur) |h| {
                const next = h.next;
                Binding.finalize(self.ctx, payloadOf(h), h.kind);
                if (free_cell_storage) {
                    const total = header_stride + h.size;
                    const base: [*]align(16) u8 = @ptrCast(@alignCast(h));
                    self.backing.free(base[0..total]);
                }
                cur = next;
            }
            self.all = null;
            self.payload_index.deinit(self.backing);
            self.payload_index = .empty;
            self.live_cells = 0;
            self.bytes_live = 0;
            self.young_cells = 0;
            self.young_bytes = 0;
            self.mark_stack.deinit(self.aux);
            self.weak_slots.deinit(self.aux);
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

    pub fn traceRoots(self: *TestRT, v: anytype) void {
        for (self.roots.items) |n| v.mark(n);
        for (self.conservative_words) |word| v.markConservativeWord(word);
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

    pub fn finalize(self: *TestRT, cell: *anyopaque, kind: Kind) void {
        switch (kind) {
            .node => {
                const n: *Node = @ptrCast(@alignCast(cell));
                self.finalized.append(std.testing.allocator, n.id) catch {};
            },
        }
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
    try std.testing.expectEqual(@as(usize, 256 * 1024), heap.nursery_threshold_bytes);
    try std.testing.expectEqual(@as(usize, 1), heap.minor_collections);
    try std.testing.expectEqual(@as(usize, 0), heap.full_collections);
    try std.testing.expectEqual(@as(u32, 2), rt.finalized.items[0]);
    heap.threshold_bytes = 1;
    try std.testing.expect(heap.shouldCollectOld());
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
        _ = self;
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
        _ = self;
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
    try std.testing.expect(Heap(TestRT).headerOf(fresh).marked);

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
    @atomicStore(bool, &hdr.marked, false, .monotonic);
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

    while (!@atomicLoad(bool, &hdr.marked, .acquire)) std.atomic.spinLoopHint();
    heap.marking.store(false, .release);
    heap.concurrent.store(false, .release);
    heap.barrier_buf.clearRetainingCapacity();
    heap.unlockBarrier();
    t.join();

    try std.testing.expectEqual(@as(usize, 0), heap.barrier_buf.items.len);
}
