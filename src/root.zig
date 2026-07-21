//! zig-gc — a precise, non-moving, tri-color mark-sweep garbage collector
//! generic over an embedder binding. See `heap.zig` and the design note in
//! zig-utils/zig-js `docs/threads/P7-gc-design.md`.

pub const Heap = @import("heap.zig").Heap;
pub const CollectionPhaseBoundary = @import("heap.zig").CollectionPhaseBoundary;
pub const InteriorOwnership = @import("heap.zig").InteriorOwnership;
pub const RelocationRecord = @import("heap.zig").RelocationRecord;
pub const RelocationState = @import("heap.zig").RelocationState;
pub const RelocationVisitor = @import("heap.zig").RelocationVisitor;
pub const StableCellId = @import("heap.zig").StableCellId;

test {
    _ = @import("heap.zig");
}
