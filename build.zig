const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The importable module: `@import("gc")` once a consumer adds this package.
    _ = b.addModule("gc", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    // ThreadSanitizer gate for the concurrent/parallel collector (issue #1 M3).
    // TSan can't build on the macOS dev box (bundled libcxx), so this runs on
    // Linux CI: `zig build test -Dtsan=true`.
    const tsan = b.option(bool, "tsan", "Build the tests with ThreadSanitizer") orelse false;

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
