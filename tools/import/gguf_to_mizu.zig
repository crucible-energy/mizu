const std = @import("std");
const importer = @import("mizu_importer.zig");

pub fn main(init: std.process.Init) !void {
    try importer.run(init, .gguf);
}
