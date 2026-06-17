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
        /// stores are caught by the barrier). M3 will run this phase concurrently
        /// with mutators; M2 keeps it under the GIL.
        marking: bool = false,
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
            /// white→grey on first sight, pushed once).
            pub fn mark(v: *Visitor, cell: ?*anyopaque) void {
                const p = cell orelse return;
                const h = Self.headerOf(p);
                if (h.magic != header_magic) std.debug.panic("GC mark of non-GC cell at 0x{x}", .{@intFromPtr(p)});
                if (h.marked) return;
                h.marked = true;
                v.heap.marked_count += 1;
                v.heap.mark_stack.append(v.heap.backing, h) catch {
                    v.oom = true;
                };
            }

            /// Whether `cell` is one of this heap's managed payloads. Bindings use
            /// this before marking legacy/embedder pointers that may still point
            /// outside the GC heap.
            pub fn isManaged(v: *Visitor, cell: ?*anyopaque) bool {
                const p = cell orelse return false;
                return v.heap.headerForPayload(p) != null;
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
                v.heap.mark_stack.append(v.heap.backing, h) catch {
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

            /// Register a weak slot. After marking completes, if its target
            /// stayed white the slot is set to null (the cell is dying).
            pub fn markWeak(v: *Visitor, slot: *?*anyopaque) void {
                v.heap.weak_slots.append(v.heap.backing, slot) catch {
                    v.oom = true;
                };
            }
        };

        pub fn init(backing: std.mem.Allocator, ctx: *Binding) Self {
            return .{ .backing = backing, .ctx = ctx };
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
            const slab = try self.backing.alignedAlloc(u8, .@"16", total);
            const h: *Header = @ptrCast(@alignCast(slab.ptr));
            // Allocate-grey during an incremental mark: a cell created while
            // marking is in progress is born marked AND queued for tracing, so it
            // survives this cycle and — crucially — its *creation-time* field
            // writes are caught when it is traced (the caller fully initializes
            // the payload before the next safepoint, where the next `markStep`
            // runs). This is what lets the embedder barrier only post-creation
            // mutations instead of every initializing store. Children added after
            // it is traced are caught by the insertion `writeBarrier`.
            const born_grey = self.marking;
            h.* = .{ .magic = header_magic, .next = self.all, .size = @sizeOf(T), .kind = kind, .marked = born_grey };
            self.all = h;
            self.live_cells += 1;
            self.bytes_live += total;
            if (born_grey) {
                self.marked_count += 1;
                self.mark_stack.append(self.backing, h) catch {};
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
            if (!self.marking) return;
            const p = cell orelse return;
            const h = Self.headerOf(p);
            if (h.magic != header_magic) return;
            if (h.marked) return;
            h.marked = true;
            self.marked_count += 1;
            self.mark_stack.append(self.backing, h) catch {};
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
            self.marking = true;
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
            std.debug.assert(self.marking);
            var v = Visitor{ .heap = self };
            Binding.traceRoots(self.ctx, &v);
            while (self.mark_stack.pop()) |h| {
                Binding.trace(payloadOf(h), h.kind, &v);
            }
            self.marking = false;
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
            self.mark_stack.deinit(self.backing);
            self.weak_slots.deinit(self.backing);
            self.addr_index.deinit(self.backing);
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
                v.mark(n.strong);
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
    try std.testing.expect(!heap.marking);
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
