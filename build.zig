const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The importable module: `@import("gc")` once a consumer adds this package.
    _ = b.addModule("gc", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    // `-Dtsan` builds the unit tests under ThreadSanitizer — the concurrency
    // gate for the parallel/concurrent collector paths (issue #1 M3). The
    // parallel marking/allocation/sweep tests must stay race-free under it.
    const tsan = b.option(bool, "tsan", "Build unit tests with ThreadSanitizer") orelse false;

    // Unit tests over the collector core.
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
            .sanitize_thread = tsan,
        }),
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run zig-gc unit tests");
    test_step.dependOn(&run_tests.step);
}
