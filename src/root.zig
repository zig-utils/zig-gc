//! zig-gc — a precise, non-moving, tri-color mark-sweep garbage collector
//! generic over an embedder binding. See `heap.zig` and the design note in
//! zig-utils/zig-js `docs/threads/P7-gc-design.md`.

pub const Heap = @import("heap.zig").Heap;
pub const InteriorOwnership = @import("heap.zig").InteriorOwnership;
pub const CellAllocation = @import("heap.zig").CellAllocation;

test {
    _ = @import("heap.zig");
}
