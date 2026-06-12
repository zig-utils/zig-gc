const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The importable module: `@import("gc")` once a consumer adds this package.
    _ = b.addModule("gc", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    // Unit tests over the collector core.
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run zig-gc unit tests");
    test_step.dependOn(&run_tests.step);
}
