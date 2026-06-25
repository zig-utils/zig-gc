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
//!
//! Inside trace/traceRoots the binding calls `v.mark(ptr)` for each strong
//! reference and `v.markWeak(&slot)` for each weak slot (`*?*anyopaque`).
//! `traceEphemeron` may call `v.isMarked(key)` and then `v.mark(value)` for
//! WeakMap-style key/value edges.

const std = @import("std");

pub fn Heap(comptime Binding: type) type {
    return struct {
        const Self = @This();
        pub const Kind = Binding.Kind;

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
        live_cells: usize = 0,
        bytes_live: usize = 0,
        /// Collect when `bytes_live` crosses this; reset to 2× live after a
        /// collection (a simple allocation-rate-agnostic growth policy).
        threshold_bytes: usize = 64 * 1024,
        mark_stack: std.ArrayListUnmanaged(*Header) = .empty,
        weak_slots: std.ArrayListUnmanaged(*?*anyopaque) = .empty,
        marked_count: usize = 0,
        collections: usize = 0,
        /// True between `startMarking` and `finishMarking`: an incremental mark
        /// is in progress and the mutator runs between `markStep`s. While set,
        /// the insertion `writeBarrier` shades stored-into cells grey, and freshly
        /// allocated cells are born grey (so they survive the in-progress cycle
        /// and their creation-time field writes are caught when traced; later
        /// stores are caught by the barrier). M3 runs this phase concurrently
        /// with mutators (`concurrent`); M2 keeps it under the GIL.
        marking: std.atomic.Value(bool) = .init(false),
        /// True during a *concurrent* mark (M3): the marker runs on its own
        /// thread while mutators keep executing. The mark-claim then uses an
        /// atomic compare-and-set on the cell's mark bit (so marker and mutator
        /// never double-push), and the mutator's `writeBarrier` hands greyed
        /// cells to the marker through the lock-guarded `barrier_buf` instead of
        /// touching the marker-private `mark_stack`.
        concurrent: std.atomic.Value(bool) = .init(false),
        /// Born-colour flag, decoupled from `marking` only during the
        /// parallel finish window: cells born while this is set are born
        /// marked (survive the in-flight sweep). `marking` (barrier-active)
        /// turns off before the sweep; `born_black` stays on until after it.
        born_black: std.atomic.Value(bool) = .init(false),
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
                if (!v.heap.claimMark(h)) return; // already grey/black
                v.heap.mark_stack.append(v.heap.aux, h) catch {
                    v.oom = true;
                };
            }

            /// Whether `cell` is one of this heap's managed payloads. Bindings use
            /// this before marking legacy/embedder pointers that may still point
            /// outside the GC heap. O(1): reads the candidate header's magic
            /// (exactly the check `mark` makes), so it does **not** walk the
            /// all-cells list — which would both cost O(n) per call and race the
            /// mutator's `create` prepending to that list during a concurrent
            /// mark. Callers must only pass pointers into mapped memory (a GC cell
            /// or another live engine allocation, e.g. an arena cell), never a
            /// freed or wild pointer; arena bytes simply won't match the magic.
            pub fn isManaged(v: *Visitor, cell: ?*anyopaque) bool {
                _ = v;
                const p = cell orelse return false;
                if (!std.mem.isAligned(@intFromPtr(p), 16)) return false;
                return Self.headerOf(p).magic == header_magic;
            }

            /// Whether a cell is already black/grey in the current collection.
            /// Used by ephemeron tables: if the key is live, the value is a
            /// strong edge; if the key stays white, the entry is weak.
            pub fn isMarked(v: *Visitor, cell: ?*anyopaque) bool {
                _ = v;
                const p = cell orelse return false;
                return Self.headerOf(p).marked;
            }

            /// Conservatively mark a machine word if it points at the payload
            /// of a managed cell. This is intentionally opt-in: precise
            /// embedders should keep using `mark`, while runtimes that need to
            /// root native stacks can scan a stack/register spill range without
            /// teaching the collector about their frame layout.
            pub fn markConservativeWord(v: *Visitor, word: usize) void {
                const h = v.heap.headerForInteriorAddress(word) orelse return;
                if (h.marked) return;
                h.marked = true;
                v.heap.marked_count += 1;
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
                v.heap.weak_slots.append(v.heap.backing, slot) catch {
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

        fn headerOf(payload: *anyopaque) *Header {
            const raw: [*]u8 = @ptrCast(payload);
            return @ptrCast(@alignCast(raw - header_stride));
        }

        fn payloadOf(h: *Header) *anyopaque {
            const raw: [*]u8 = @ptrCast(h);
            return @ptrCast(raw + header_stride);
        }

        fn headerForPayload(self: *Self, payload: *anyopaque) ?*Header {
            var it = self.all;
            while (it) |h| : (it = h.next) {
                if (payloadOf(h) == payload) return h;
            }
            return null;
        }

        /// Whether `p` is a live (marked) cell — O(1). `p` must be null or a
        /// pointer to a cell allocated by this heap. This is the read a binding
        /// uses for **isMarked-based weak clearing**: deciding a weak key's /
        /// finalizer target's liveness in the world-stopped finish pass by its
        /// mark bit, instead of pre-registering an interior `&slot` weak pointer
        /// that a concurrent mutator append could dangle by reallocating the
        /// buffer it points into. Call only with marks still valid (before sweep).
        pub fn isLive(self: *Self, p: ?*anyopaque) bool {
            _ = self;
            const ptr = p orelse return false;
            return headerOf(ptr).marked;
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
            self.addr_index.clearRetainingCapacity();
            var it = self.all;
            while (it) |h| : (it = h.next) {
                const start = @intFromPtr(payloadOf(h));
                // A zero-size payload would make an empty extent that no interior
                // address can fall into; record at least one byte so an exact
                // payload pointer still resolves.
                const size = if (h.size == 0) 1 else h.size;
                self.addr_index.append(self.backing, .{
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
            const slab = try self.backing.alignedAlloc(u8, .@"16", total);
            const h: *Header = @ptrCast(@alignCast(slab.ptr));
            // Shared-state bookkeeping (all-list prepend, counters, born-cell
            // hand-off) is serialized across mutators in `parallel` mode; lock-free
            // otherwise (single GIL'd mutator).
            if (self.parallel) self.lockAlloc();
            defer if (self.parallel) self.unlockAlloc();
            // Allocate-grey during an incremental mark: a cell created while
            // marking is in progress is born marked AND queued for tracing, so it
            // survives this cycle and — crucially — its *creation-time* field
            // writes are caught when it is traced (the caller fully initializes
            // the payload before the next safepoint, where the next `markStep`
            // runs). This is what lets the embedder barrier only post-creation
            // mutations instead of every initializing store. Children added after
            // it is traced are caught by the insertion `writeBarrier`.
            const born_grey = self.born_black.load(.acquire);
            h.* = .{ .magic = header_magic, .next = self.all, .size = @sizeOf(T), .kind = kind, .marked = born_grey };
            self.all = h;
            self.live_cells += 1;
            // Atomic: a parallel mutator at a GC safepoint reads `bytes_live` for
            // its collection-threshold check (Context.collectMidScriptParallel)
            // without taking `alloc_lock`, so the increment must be atomic to pair
            // with that read race-free. Identical value single-threaded.
            _ = @atomicRmw(usize, &self.bytes_live, .Add, total, .monotonic);
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
                    self.marked_count += 1;
                    self.mark_stack.append(self.aux, h) catch {};
                }
            }
            return @ptrCast(@alignCast(slab.ptr + header_stride));
        }

        /// Collect if the heap has grown past the threshold. Call at safepoints
        /// (the engine's `(steps & 1023)` checkpoints) and after large allocs.
        pub fn maybeCollect(self: *Self) void {
            if (self.bytes_live >= self.threshold_bytes) self.collect();
        }

        /// Dijkstra insertion write barrier. The embedder calls this whenever it
        /// stores a reference to `cell` into a heap object during an incremental
        /// mark: it shades `cell` grey so a reference newly hidden behind an
        /// already-black object is never missed (the black→white invariant). A
        /// no-op when not marking, when `cell` is null, or when `cell` is not a
        /// managed payload (the embedder may store non-cell pointers) — so it is
        /// cheap and safe to call broadly. Idempotent (already-grey/black: skip).
        pub fn writeBarrier(self: *Self, cell: ?*anyopaque) void {
            if (!self.marking.load(.acquire)) return;
            const p = cell orelse return;
            const h = Self.headerOf(p);
            if (h.magic != header_magic) return;
            if (!self.claimMark(h)) return;
            if (self.concurrent.load(.acquire)) {
                // Hand the greyed cell to the marker thread (it owns mark_stack).
                self.lockBarrier();
                self.barrier_buf.append(self.aux, h) catch {};
                self.unlockBarrier();
            } else {
                self.mark_stack.append(self.aux, h) catch {};
            }
        }

        /// Atomically claim a white cell as grey (returns true once per cell).
        /// Under a concurrent mark the claim is a compare-and-set so the marker
        /// and a mutator's `writeBarrier` never both push the same cell; the
        /// single-threaded path is a plain check.
        fn claimMark(self: *Self, h: *Header) bool {
            if (self.concurrent.load(.acquire)) {
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

        inline fn lockAlloc(self: *Self) void {
            while (self.alloc_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
        }
        inline fn unlockAlloc(self: *Self) void {
            self.alloc_lock.store(0, .release);
        }

        /// Begin an incremental mark: whiten all cells and grey the roots. The
        /// mutator then runs between `markStep`s with the `writeBarrier` active.
        pub fn startMarking(self: *Self) void {
            var it = self.all;
            while (it) |h| : (it = h.next) h.marked = false;
            self.marked_count = 0;
            self.mark_stack.clearRetainingCapacity();
            self.weak_slots.clearRetainingCapacity();
            self.addr_index_built = false;
            self.born_black.store(true, .release);
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
            self.born_black.store(false, .release);
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
            self.born_black.store(false, .release);
            self.sweepPhase(&v);
        }

        // -- Parallel (no-GIL, multi-mutator) concurrent collection (issue #1
        //    Phase 7 / M3). The world is never stopped: the collector gathers
        //    roots via the embedder's ragged root-publication handshake
        //    (src/root_handshake.zig) — its own + parked peers via `greyRoots`,
        //    running peers self-publish via `writeBarrier` at their safepoints —
        //    marks concurrently behind the insertion barrier, then sweeps. See
        //    docs/threads/P7-gc-design.md.

        /// Begin a parallel concurrent mark. Unlike `beginConcurrentMark` the
        /// world is NOT stopped: whiten the all-cell list under `alloc_lock` (so a
        /// concurrent `create` prepend can't race the walk; safe to write `marked`
        /// non-atomically here because `marking` is still off, so no barrier or
        /// marker reads colours yet), then publish `concurrent` BEFORE `marking`
        /// so a mutator that observes `marking` also observes `concurrent` and
        /// routes stores to the marker-owned `barrier_buf`, never `mark_stack`.
        /// No roots are traced here — the collector calls `greyRoots` next.
        pub fn beginConcurrentMarkParallel(self: *Self) void {
            self.lockAlloc();
            var it = self.all;
            while (it) |h| : (it = h.next) h.marked = false;
            self.marked_count = 0;
            self.mark_stack.clearRetainingCapacity();
            self.weak_slots.clearRetainingCapacity();
            self.addr_index_built = false;
            self.barrier_buf.clearRetainingCapacity();
            self.born_concurrent.clearRetainingCapacity();
            self.deferred_trace.clearRetainingCapacity();
            self.concurrent.store(true, .release); // before marking (see above)
            self.born_black.store(true, .release);
            self.marking.store(true, .release);
            self.unlockAlloc();
        }

        /// Collector: grey the embedder's traced roots (its own native stack +
        /// parked peers' frozen stacks + persistent VM roots) onto `mark_stack`.
        /// Running peers are NOT traced here — they self-publish via `writeBarrier`
        /// (into `barrier_buf`) at their safepoints. Call after
        /// `beginConcurrentMarkParallel` and again for each finish-phase re-scan.
        pub fn greyRoots(self: *Self) void {
            var v = Visitor{ .heap = self };
            Binding.traceRoots(self.ctx, &v);
        }

        /// Collector: turn the barrier off once the mark has reached a fixpoint
        /// (every running peer re-greyed its roots and the grey set drained, so no
        /// white cell is reachable by any mutator). `born_black` stays on so cells
        /// still allocated during the finish survive this sweep. A mutator store
        /// racing this either greys an already-marked cell (harmless) or skips the
        /// barrier but can only reach already-marked cells. The embedder then runs
        /// ONE post-flip handshake so every peer passes a safepoint (finishing any
        /// in-flight barrier) before `sweepConcurrentParallel`.
        pub fn markOffParallel(self: *Self) void {
            self.marking.store(false, .release);
        }

        /// Collector: finish + sweep for the parallel model. Call after
        /// `markOffParallel` AND the post-flip handshake (so no peer is still
        /// barriering — claims can be non-atomic again). Fold the last hand-off /
        /// born / deferred cells, drain, then sweep under `alloc_lock`: the sweep
        /// walks `all`, which `create` prepends to under the same lock reading
        /// `born_black`, so a racing allocation either lands before the sweep (born
        /// marked, kept) or after it (added post-sweep, reclaimed next cycle) —
        /// never freed live. `born_black` is cleared last, after the sweep.
        pub fn sweepConcurrentParallel(self: *Self) void {
            std.debug.assert(!self.marking.load(.acquire));
            var v = Visitor{ .heap = self };
            self.lockBarrier();
            self.mark_stack.appendSlice(self.aux, self.barrier_buf.items) catch {};
            self.barrier_buf.clearRetainingCapacity();
            self.unlockBarrier();
            self.mark_stack.appendSlice(self.aux, self.born_concurrent.items) catch {};
            self.born_concurrent.clearRetainingCapacity();
            for (self.deferred_trace.items) |h| Binding.trace(payloadOf(h), h.kind, &v);
            self.deferred_trace.clearRetainingCapacity();
            self.concurrent.store(false, .release);
            while (self.mark_stack.pop()) |h| {
                Binding.trace(payloadOf(h), h.kind, &v);
            }
            self.lockAlloc();
            self.sweepPhase(&v);
            self.born_black.store(false, .release);
            self.unlockAlloc();
        }

        /// Collector: build the address-sorted interior-pointer index now, before
        /// opening the root handshake. Running peers' `barrierConservativeWord`
        /// self-publish then only *reads* the index — safe concurrently — because
        /// the collector built it first (the handshake's release/acquire
        /// establishes the happens-before so peers observe `addr_index_built`).
        pub fn ensureAddrIndex(self: *Self) void {
            // The build walks the `all` list; under parallel mutation a concurrent
            // `create` prepends to it, so freeze allocation for the walk.
            self.lockAlloc();
            defer self.unlockAlloc();
            if (!self.addr_index_built) self.buildAddrIndex();
        }

        /// Mutator self-publish for the parallel concurrent mark: conservatively
        /// grey the cell that stack `word` points into (interior-pointer aware),
        /// routing the grey to the marker via `barrier_buf` — never the
        /// marker-owned `mark_stack`. The collector must have called
        /// `ensureAddrIndex` before any peer calls this; if the index isn't built
        /// yet this is a no-op (the cell, if live, is still reachable from another
        /// root or is born-marked, and the collector's re-scan catches stragglers).
        pub fn barrierConservativeWord(self: *Self, word: usize) void {
            if (!self.addr_index_built) return;
            const h = self.headerForInteriorAddress(word) orelse return;
            if (!self.claimMark(h)) return;
            self.lockBarrier();
            self.barrier_buf.append(self.aux, h) catch {};
            self.unlockBarrier();
        }

        /// `barrierConservativeWord` over a word-aligned range (a peer's own stack
        /// + spilled registers). Same routing + safety contract.
        pub fn barrierConservativeWords(self: *Self, start: [*]const usize, words: usize) void {
            var i: usize = 0;
            while (i < words) : (i += 1) self.barrierConservativeWord(start[i]);
        }

        /// The ephemeron-fixpoint + weak-edge + sweep tail shared by the
        /// stop-the-world and incremental paths. `v` is a live Visitor over self.
        fn sweepPhase(self: *Self, v: *Visitor) void {
            // 2b. Ephemerons (WeakMap-style edges): if a marked table has a
            // marked key, its value becomes strong. Iterate to a fixed point so
            // values can keep further keys alive through chains of weak maps.
            if (@hasDecl(Binding, "traceEphemeron")) {
                while (true) {
                    const before = self.marked_count;
                    var eit = self.all;
                    while (eit) |h| : (eit = h.next) {
                        if (h.marked) Binding.traceEphemeron(self.ctx, payloadOf(h), h.kind, v);
                    }
                    while (self.mark_stack.pop()) |h| {
                        Binding.trace(payloadOf(h), h.kind, v);
                    }
                    if (self.marked_count == before) break;
                }
            }

            // 3. weak edges whose target died are cleared *before* the sweep
            //    frees it, so no slot ever dangles.
            for (self.weak_slots.items) |slot| {
                if (slot.*) |target| {
                    if (!headerOf(target).marked) slot.* = null;
                }
            }

            if (@hasDecl(Binding, "afterWeak")) {
                var wit = self.all;
                while (wit) |h| : (wit = h.next) {
                    if (h.marked) Binding.afterWeak(self.ctx, payloadOf(h), h.kind);
                }
            }

            // 4. sweep the white cells.
            var prev: ?*Header = null;
            var cur = self.all;
            while (cur) |h| {
                const next = h.next;
                if (h.marked) {
                    prev = h;
                } else {
                    Binding.finalize(self.ctx, payloadOf(h), h.kind);
                    if (prev) |p| p.next = next else self.all = next;
                    const total = header_stride + h.size;
                    self.live_cells -= 1;
                    self.bytes_live -= total;
                    const base: [*]align(16) u8 = @ptrCast(@alignCast(h));
                    self.backing.free(base[0..total]);
                }
                cur = next;
            }

            self.collections += 1;
            self.threshold_bytes = @max(64 * 1024, self.bytes_live * 2);
        }

        /// Free every remaining cell (finalizing each) and the internal lists.
        /// The embedder calls this at context teardown — equivalent to the old
        /// arena `deinit`, but finalizers run.
        pub fn deinit(self: *Self) void {
            var cur = self.all;
            while (cur) |h| {
                const next = h.next;
                Binding.finalize(self.ctx, payloadOf(h), h.kind);
                const total = header_stride + h.size;
                const base: [*]align(16) u8 = @ptrCast(@alignCast(h));
                self.backing.free(base[0..total]);
                cur = next;
            }
            self.all = null;
            self.live_cells = 0;
            self.bytes_live = 0;
            self.mark_stack.deinit(self.aux);
            self.weak_slots.deinit(self.backing);
            self.addr_index.deinit(self.backing);
            self.barrier_buf.deinit(self.aux);
            self.born_concurrent.deinit(self.aux);
            self.deferred_trace.deinit(self.aux);
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
                // is marked regardless). A bare reference slot the marker reads
                // and a mutator writes must be accessed atomically to be
                // race-free per the memory model — a relaxed load/store, which is
                // a plain mov on x86_64/arm64. This is the pattern the engine
                // binding uses for its non-collection pointer slots (e.g. proto).
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

test "parallel concurrent collection: collect mid-script while mutators run (no-GIL M3)" {
    // The first true MID-SCRIPT parallel collection (issue #1 Phase 7 / M3): the
    // collector runs `beginConcurrentMarkParallel` → root-publication handshake →
    // concurrent rounds → re-scan to a fixpoint → `markOffParallel` → post-flip
    // handshake → `sweepConcurrentParallel`, WHILE N mutators keep allocating and
    // growing rooted graphs the whole time (no world stop, no GIL). Each mutator
    // greys its own root at a safepoint (the ragged handshake — modelled inline
    // here exactly as src/root_handshake.zig drives it in the engine). The
    // collector must keep every live chain intact (no live cell swept → no UAF;
    // DebugAllocator-clean proves it) and reclaim the pre-existing garbage.
    if (@import("builtin").single_threaded) return error.SkipZigTest;
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
    const nworkers = 4;
    const per = 1500; // live nodes each worker chains onto its (published) root
    const garbage = 600; // pre-existing unreachable nodes — must be reclaimed

    // Pre-existing garbage: born white BEFORE the cycle, reachable from nothing.
    var g: u32 = 0;
    while (g < garbage) : (g += 1) {
        const junk = try heap.create(N, .node);
        junk.* = .{ .id = 1_000_000 + g };
    }
    // Per-worker root nodes (also born white; greyed only via the handshake — they
    // are NOT in rt.roots, so the collector reaches them solely through a mutator
    // publishing its own root, exactly the running-peer case).
    var roots: [nworkers]*N = undefined;
    for (&roots) |*r| {
        r.* = try heap.create(N, .node);
        r.*.* = .{ .id = 0 };
    }

    const Shared = struct {
        hs_request: std.atomic.Value(u64) = .init(0),
        hs_acks: std.atomic.Value(u32) = .init(0),
        done_building: std.atomic.Value(u32) = .init(0),
        collection_done: std.atomic.Value(bool) = .init(false),
    };
    const Worker = struct {
        heap: *Heap(TestRT),
        sh: *Shared,
        root: *N,
        base: u32,
        published: u64 = 0,
        fn safepoint(s: *@This()) void {
            const req = s.sh.hs_request.load(.acquire);
            if (req == 0 or s.published == req) return;
            // Publish: grey my own root (a no-op after markOff — barrier off — but
            // the ack still signals "passed a safepoint", which is all the
            // post-flip handshake needs).
            s.heap.writeBarrier(s.root);
            s.published = req;
            _ = s.sh.hs_acks.fetchAdd(1, .release);
        }
        fn run(s: *@This()) void {
            var i: u32 = 1;
            while (i <= per) : (i += 1) {
                const node = s.heap.create(N, .node) catch return;
                node.* = .{ .id = s.base + i };
                const old = @atomicLoad(?*N, &s.root.strong, .monotonic);
                @atomicStore(?*N, &node.strong, old, .monotonic);
                @atomicStore(?*N, &s.root.strong, node, .release);
                s.heap.writeBarrier(node); // node newly reachable from root → grey
                if (i % 64 == 0) s.safepoint();
            }
            _ = s.sh.done_building.fetchAdd(1, .release);
            // Keep hitting safepoints so the collector's handshakes complete.
            while (!s.sh.collection_done.load(.acquire)) {
                s.safepoint();
                std.atomic.spinLoopHint();
            }
        }
    };

    var shared = Shared{};
    var workers: [nworkers]Worker = undefined;
    for (&workers, 0..) |*w, t| w.* = .{ .heap = &heap, .sh = &shared, .root = roots[t], .base = @as(u32, @intCast(t)) * per };
    var pool: [nworkers]std.Thread = undefined;
    for (&pool, 0..) |*th, t| th.* = try std.Thread.spawn(.{}, Worker.run, .{&workers[t]});

    // --- Collector (this thread): a full mid-script parallel cycle. ---
    heap.beginConcurrentMarkParallel();
    heap.greyRoots(); // own + parked roots (none here; running peers self-publish)
    var cycle: u64 = 1;
    while (true) {
        const before = @atomicLoad(usize, &heap.marked_count, .monotonic);
        shared.hs_acks.store(0, .monotonic);
        shared.hs_request.store(cycle, .release);
        // Gather every running peer's roots, draining grey work meanwhile.
        while (shared.hs_acks.load(.acquire) < nworkers) {
            _ = heap.concurrentMarkRound();
            std.atomic.spinLoopHint();
        }
        while (!heap.concurrentMarkRound()) {} // fully drain this round's grey set
        shared.hs_request.store(0, .release);
        cycle += 1;
        // Fixpoint: all peers stopped building AND this re-scan greyed nothing new.
        if (shared.done_building.load(.acquire) == nworkers and
            @atomicLoad(usize, &heap.marked_count, .monotonic) == before) break;
    }
    heap.markOffParallel(); // barrier off; born_black stays on through the sweep
    // Post-flip handshake: every peer passes a safepoint after markOff, so no
    // barrier is in flight when sweepConcurrentParallel turns `concurrent` off.
    shared.hs_acks.store(0, .monotonic);
    shared.hs_request.store(cycle, .release);
    while (shared.hs_acks.load(.acquire) < nworkers) std.atomic.spinLoopHint();
    shared.hs_request.store(0, .release);
    heap.sweepConcurrentParallel();
    shared.collection_done.store(true, .release);
    for (&pool) |*th| th.join();

    // Correctness 1: every live chain is intact — no live cell was swept. Each
    // root's chain has exactly `per` nodes with the right id multiset.
    for (&roots, 0..) |r, t| {
        const base: u32 = @as(u32, @intCast(t)) * per;
        var count: usize = 0;
        var expect_sum: u64 = 0;
        var got_sum: u64 = 0;
        var cur = @atomicLoad(?*N, &r.strong, .monotonic);
        while (cur) |c| : (cur = c.strong) {
            count += 1;
            got_sum += c.id;
        }
        var k: u32 = 1;
        while (k <= per) : (k += 1) expect_sum += base + k;
        try std.testing.expectEqual(@as(usize, per), count);
        try std.testing.expectEqual(expect_sum, got_sum);
    }
    // Correctness 2: the pre-existing garbage was reclaimed (collector swept).
    try std.testing.expectEqual(@as(usize, garbage), rt.finalized.items.len);
    // Correctness 3: live_cells == N roots + N*per chain nodes (garbage gone).
    try std.testing.expectEqual(@as(usize, nworkers * (per + 1)), heap.live_cells);
    try std.testing.expect(heap.collections >= 1);
}

test "parallel concurrent collection: peers self-publish CONSERVATIVELY (interior pointers, no-GIL M3)" {
    // Like the test above, but running peers publish their root via
    // `barrierConservativeWord` (interior-pointer resolution → grey via
    // barrier_buf) instead of the exact-pointer `writeBarrier`. This is the
    // engine's actual root-publish mode (a peer conservatively scans its own
    // native stack at a safepoint), and it exercises the address-index built by
    // the collector before the handshake opens being *read* concurrently by N
    // peers — the hazard that makes conservative self-publish the hard part of
    // wiring this into the live engine. Live chains intact + garbage reclaimed +
    // (CI) TSan-clean proves the index read + barrier routing is race-free.
    if (@import("builtin").single_threaded) return error.SkipZigTest;
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
    const nworkers = 4;
    const per = 1500;
    const garbage = 600;

    var g: u32 = 0;
    while (g < garbage) : (g += 1) {
        const junk = try heap.create(N, .node);
        junk.* = .{ .id = 1_000_000 + g };
    }
    var roots: [nworkers]*N = undefined;
    for (&roots) |*r| {
        r.* = try heap.create(N, .node);
        r.*.* = .{ .id = 0 };
    }

    const Shared = struct {
        hs_request: std.atomic.Value(u64) = .init(0),
        hs_acks: std.atomic.Value(u32) = .init(0),
        done_building: std.atomic.Value(u32) = .init(0),
        collection_done: std.atomic.Value(bool) = .init(false),
    };
    const Worker = struct {
        heap: *Heap(TestRT),
        sh: *Shared,
        root: *N,
        base: u32,
        published: u64 = 0,
        fn safepoint(s: *@This()) void {
            const req = s.sh.hs_request.load(.acquire);
            if (req == 0 or s.published == req) return;
            // Conservative interior-pointer publish (the engine's mode): treat the
            // live root as a stack word and route it through barrier_buf. The
            // happens-before from this acquire-load of `req` (paired with the
            // collector's release-store in `open`) guarantees we observe the
            // collector's built addr-index.
            s.heap.barrierConservativeWord(@intFromPtr(s.root));
            s.published = req;
            _ = s.sh.hs_acks.fetchAdd(1, .release);
        }
        fn run(s: *@This()) void {
            var i: u32 = 1;
            while (i <= per) : (i += 1) {
                const node = s.heap.create(N, .node) catch return;
                node.* = .{ .id = s.base + i };
                const old = @atomicLoad(?*N, &s.root.strong, .monotonic);
                @atomicStore(?*N, &node.strong, old, .monotonic);
                @atomicStore(?*N, &s.root.strong, node, .release);
                s.heap.writeBarrier(node);
                if (i % 64 == 0) s.safepoint();
            }
            _ = s.sh.done_building.fetchAdd(1, .release);
            while (!s.sh.collection_done.load(.acquire)) {
                s.safepoint();
                std.atomic.spinLoopHint();
            }
        }
    };

    var shared = Shared{};
    var workers: [nworkers]Worker = undefined;
    for (&workers, 0..) |*w, t| w.* = .{ .heap = &heap, .sh = &shared, .root = roots[t], .base = @as(u32, @intCast(t)) * per };
    var pool: [nworkers]std.Thread = undefined;
    for (&pool, 0..) |*th, t| th.* = try std.Thread.spawn(.{}, Worker.run, .{&workers[t]});

    heap.beginConcurrentMarkParallel();
    heap.ensureAddrIndex(); // BEFORE the handshake: peers only read the index
    heap.greyRoots();
    var cycle: u64 = 1;
    while (true) {
        const before = @atomicLoad(usize, &heap.marked_count, .monotonic);
        shared.hs_acks.store(0, .monotonic);
        shared.hs_request.store(cycle, .release);
        while (shared.hs_acks.load(.acquire) < nworkers) {
            _ = heap.concurrentMarkRound();
            std.atomic.spinLoopHint();
        }
        while (!heap.concurrentMarkRound()) {}
        shared.hs_request.store(0, .release);
        cycle += 1;
        if (shared.done_building.load(.acquire) == nworkers and
            @atomicLoad(usize, &heap.marked_count, .monotonic) == before) break;
    }
    heap.markOffParallel();
    shared.hs_acks.store(0, .monotonic);
    shared.hs_request.store(cycle, .release);
    while (shared.hs_acks.load(.acquire) < nworkers) std.atomic.spinLoopHint();
    shared.hs_request.store(0, .release);
    heap.sweepConcurrentParallel();
    shared.collection_done.store(true, .release);
    for (&pool) |*th| th.join();

    for (&roots, 0..) |r, t| {
        const base: u32 = @as(u32, @intCast(t)) * per;
        var count: usize = 0;
        var expect_sum: u64 = 0;
        var got_sum: u64 = 0;
        var cur = @atomicLoad(?*N, &r.strong, .monotonic);
        while (cur) |c| : (cur = c.strong) {
            count += 1;
            got_sum += c.id;
        }
        var k: u32 = 1;
        while (k <= per) : (k += 1) expect_sum += base + k;
        try std.testing.expectEqual(@as(usize, per), count);
        try std.testing.expectEqual(expect_sum, got_sum);
    }
    try std.testing.expectEqual(@as(usize, garbage), rt.finalized.items.len);
    try std.testing.expectEqual(@as(usize, nworkers * (per + 1)), heap.live_cells);
    try std.testing.expect(heap.collections >= 1);
}
