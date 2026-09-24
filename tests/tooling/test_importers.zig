const std = @import("std");
const Io = std.Io;
const Dir = Io.Dir;
const Allocator = std.mem.Allocator;

const TensorFixture = struct {
    name: []const u8,
    shape: []const u64,
    ggml_type: u32,
    data_offset: u64,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(init.gpa);
    defer init.gpa.free(args);
    if (args.len == 3 and std.mem.eql(u8, args[1], "--check-packbuffer-tree")) {
        if (try packbufferTreeHasDistinctSpans(allocator, init.io, args[2])) return;
        try report(init.io, "no packbuffer with distinct tensor spans and source offsets found under {s}\n", .{args[2]});
        std.process.exit(1);
    }
    if (args.len == 3 and std.mem.eql(u8, args[1], "--write-integrated-gguf")) {
        try writeGgufFixture(init.io, args[2], false, false, true);
        return;
    }
    if (args.len != 4) {
        try report(init.io, "usage: test_importers GGUF_IMPORTER SAFETENSORS_IMPORTER FIXTURE_ROOT\n", .{});
        return error.InvalidArguments;
    }

    const gguf_importer = args[1];
    const safetensors_importer = args[2];
    const fixture_root = args[3];
    try ensureDirectory(init.io, fixture_root);
    try testPackbufferInspector(allocator, init.io, args[0], fixture_root);
    try testSafetensors(allocator, init.io, safetensors_importer, fixture_root);
    try testGguf(allocator, init.io, gguf_importer, fixture_root);
    try report(init.io, "test_importers: PASS\n", .{});
}

fn testPackbufferInspector(allocator: Allocator, io: Io, executable: []const u8, root: []const u8) !void {
    const fixture_root = try join(allocator, root, "packbuffer-inspector");
    const fixture_dir = try join(allocator, fixture_root, "nested");
    const fixture_path = try join(allocator, fixture_dir, "weights.packbuffer");
    try ensureDirectory(io, fixture_dir);
    try writePackbufferFixture(io, fixture_path, true);
    _ = try runCommand(allocator, io, executable, &.{ "--check-packbuffer-tree", fixture_root }, 0, null);
    try writePackbufferFixture(io, fixture_path, false);
    _ = try runCommand(allocator, io, executable, &.{ "--check-packbuffer-tree", fixture_root }, 1, "no packbuffer with distinct tensor spans");
}

fn writePackbufferFixture(io: Io, path: []const u8, distinct: bool) !void {
    var bytes: [32 + 2 * 104]u8 = undefined;
    @memset(&bytes, 0);
    writeLeU32(bytes[4..8], 2);
    writeLeU32(bytes[12..16], 104);
    writeLeU32(bytes[16..20], 2);
    for (0..2) |index| {
        const record = bytes[32 + index * 104 ..][0..104];
        writeLeI64(record[56..64], if (distinct) @as(i64, @intCast(index + 1)) * 1024 else 1024);
        writeLeI64(record[96..104], if (distinct) @as(i64, @intCast(index + 1)) * 2048 else 2048);
    }
    var file = try Dir.createFileAbsolute(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, &bytes);
}

fn packbufferTreeHasDistinctSpans(allocator: Allocator, io: Io, root: []const u8) !bool {
    var directory = try Dir.openDirAbsolute(io, root, .{ .iterate = true });
    defer directory.close(io);
    var walker = try directory.walk(allocator);
    defer walker.deinit();
    while (try walker.next(io)) |entry| {
        if (entry.kind != .file or std.mem.indexOf(u8, entry.basename, ".packbuffer") == null) continue;
        const path = try std.fs.path.join(allocator, &.{ root, entry.path });
        if (try packbufferFileHasDistinctSpans(io, path)) return true;
    }
    return false;
}

fn packbufferFileHasDistinctSpans(io: Io, path: []const u8) !bool {
    var file = try Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size < 32) return false;

    var reader = file.readerStreaming(io, &.{});
    var header: [32]u8 = undefined;
    reader.interface.readSliceAll(&header) catch return false;
    const version = readLeU32(header[4..8]);
    const entry_bytes = readLeU32(header[12..16]);
    const count = readLeU32(header[16..20]);
    if (version < 2 or entry_bytes < 104 or entry_bytes > 1024 * 1024 or count < 2 or count > 10_000_000) return false;
    const records_size = std.math.mul(u64, @as(u64, count), @as(u64, entry_bytes)) catch return false;
    if (records_size > stat.size - 32) return false;

    var record: [104]u8 = undefined;
    var skip_buffer: [4096]u8 = undefined;
    var first_span: ?i64 = null;
    var first_offset: ?i64 = null;
    var other_span = false;
    var other_offset = false;
    for (0..count) |_| {
        reader.interface.readSliceAll(&record) catch return false;
        const span = readLeI64(record[56..64]);
        const offset = readLeI64(record[96..104]);
        if (span > 0) {
            if (first_span) |first| {
                if (span != first) other_span = true;
            } else {
                first_span = span;
            }
        }
        if (offset >= 0) {
            if (first_offset) |first| {
                if (offset != first) other_offset = true;
            } else {
                first_offset = offset;
            }
        }
        var remaining = entry_bytes - 104;
        while (remaining > 0) {
            const skip_size = @min(remaining, @as(u32, @intCast(skip_buffer.len)));
            const skipped: usize = @intCast(skip_size);
            reader.interface.readSliceAll(skip_buffer[0..skipped]) catch return false;
            remaining -= skip_size;
        }
    }
    return other_span and other_offset;
}

fn readLeU32(bytes: []const u8) u32 {
    return @as(u32, bytes[0]) | (@as(u32, bytes[1]) << 8) | (@as(u32, bytes[2]) << 16) | (@as(u32, bytes[3]) << 24);
}

fn readLeI64(bytes: []const u8) i64 {
    var value: u64 = 0;
    for (bytes[0..8], 0..) |byte, index| value |= @as(u64, byte) << @intCast(index * 8);
    return @bitCast(value);
}

fn writeLeU32(bytes: []u8, value: u32) void {
    for (0..4) |index| bytes[index] = @truncate(value >> @intCast(index * 8));
}

fn writeLeI64(bytes: []u8, value: i64) void {
    const raw: u64 = @bitCast(value);
    for (0..8) |index| bytes[index] = @truncate(raw >> @intCast(index * 8));
}

fn testSafetensors(allocator: Allocator, io: Io, importer: []const u8, root: []const u8) !void {
    const source = try join(allocator, root, "hf-source");
    try ensureDirectory(io, source);
    try writeText(io, try join(allocator, source, "config.json"), "{\"_name_or_path\":\"Qwen/Qwen-3.5-VL-9B\",\"_commit_hash\":\"fixture-qwen\",\"model_type\":\"qwen3_5_vl\",\"vision_config\":{\"hidden_size\":1280}}\n");
    try writeText(io, try join(allocator, source, "tokenizer_config.json"), "{\"tokenizer_class\":\"QwenTokenizer\"}\n");
    try writeSafetensorsHeaderFixture(io, try join(allocator, source, "model-00001-of-00002.safetensors"), "{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[4,4],\"data_offsets\":[0,32]},\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"BF16\",\"shape\":[4,4],\"data_offsets\":[32,64]}}", 64);
    try writeSafetensorsHeaderFixture(io, try join(allocator, source, "model-00002-of-00002.safetensors"), "{\"model.norm.weight\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]},\"lm_head.weight\":{\"dtype\":\"BF16\",\"shape\":[4,4],\"data_offsets\":[16,48]},\"visual.position_embedding.weight\":{\"dtype\":\"F16\",\"shape\":[4,4],\"data_offsets\":[48,80]},\"vision_tower.vision_model.embeddings.class_embedding\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[80,88]},\"visual.merger.mlp.0.weight\":{\"dtype\":\"F16\",\"shape\":[4,4],\"data_offsets\":[88,120]}}", 120);
    try writeText(io, try join(allocator, source, "model.safetensors.index.json"), "{\"weight_map\":{\"model.embed_tokens.weight\":\"model-00001-of-00002.safetensors\",\"model.layers.0.self_attn.q_proj.weight\":\"model-00001-of-00002.safetensors\",\"model.norm.weight\":\"model-00002-of-00002.safetensors\",\"lm_head.weight\":\"model-00002-of-00002.safetensors\",\"visual.position_embedding.weight\":\"model-00002-of-00002.safetensors\",\"vision_tower.vision_model.embeddings.class_embedding\":\"model-00002-of-00002.safetensors\",\"visual.merger.mlp.0.weight\":\"model-00002-of-00002.safetensors\"}}\n");

    const bundle = try join(allocator, root, "hf-bundle");
    const summary = try runCommand(allocator, io, importer, &.{ source, "--output-root", bundle, "--link-mode", "copy" }, 0, null);
    try expectContains(summary.stdout, "family: qwen3_5");
    try expectContains(summary.stdout, "tensor_count: 7");
    try expectFileContains(allocator, io, try join(allocator, bundle, "manifest.mizu"), "family = qwen3_5");
    try expectFileContains(allocator, io, try join(allocator, bundle, "manifest.mizu"), "source_hash_text = ");
    try expectFileContains(allocator, io, try join(allocator, bundle, "mizu_import/tensors.tsv"), "model.embed_tokens.weight|embedding_table|bf16|row_major");
    try expectFileContains(allocator, io, try join(allocator, bundle, "mizu_import/tensors.tsv"), "model.layers.0.self_attn.q_proj.weight|decoder_stack|bf16|packed|weights/model-00001-of-00002.safetensors");
    try expectFileContains(allocator, io, try join(allocator, bundle, "mizu_import/tensors.tsv"), "lm_head.weight|token_projection|bf16|row_major");
    try expectFileContains(allocator, io, try join(allocator, bundle, "mizu_import/tensors.tsv"), "visual.position_embedding.weight|vision_encoder|f16|packed");
    try expectFileContains(allocator, io, try join(allocator, bundle, "mizu_import/tensors.tsv"), "vision_tower.vision_model.embeddings.class_embedding|vision_encoder|f16|vector");
    try expectFileContains(allocator, io, try join(allocator, bundle, "mizu_import/modalities.tsv"), "1|image|image|encoded_bytes|u8");
    try expectFileContains(allocator, io, try join(allocator, bundle, "mizu_import/projector/projector_assets.mizu"), "visual.merger.mlp.0.weight|weights/model-00002-of-00002.safetensors");
    try expectFileContains(allocator, io, try join(allocator, bundle, "mizu_import/projector/projector_assets.mizu"), "vision_tower.vision_model.embeddings.class_embedding|weights/model-00002-of-00002.safetensors");
    try expectFileExists(io, try join(allocator, bundle, "mizu_import/weights/model-00001-of-00002.safetensors"));
    try expectFileExists(io, try join(allocator, bundle, "mizu_import/weights/model-00002-of-00002.safetensors"));

    const before = try readFile(allocator, io, try join(allocator, bundle, "manifest.mizu"));
    _ = try runCommand(allocator, io, importer, &.{ source, "--output-root", bundle, "--link-mode", "copy" }, 2, "refusing to overwrite");
    const after = try readFile(allocator, io, try join(allocator, bundle, "manifest.mizu"));
    try expectEqualBytes(before, after, "safetensors output was modified without --force");
    _ = try runCommand(allocator, io, importer, &.{ source, "--output-root", bundle, "--link-mode", "copy", "--force" }, 0, null);

    const dry_run = try join(allocator, root, "hf-dry-run");
    _ = try runCommand(allocator, io, importer, &.{ source, "--output-root", dry_run, "--dry-run" }, 0, null);
    try expectNotExists(io, try join(allocator, dry_run, "manifest.mizu"));

    const link_bundle = try join(allocator, root, "hf-link-bundle");
    _ = try runCommand(allocator, io, importer, &.{ source, "--output-root", link_bundle, "--link-mode", "symlink" }, 0, null);
    try expectSymlink(io, try join(allocator, link_bundle, "mizu_import/weights/model-00001-of-00002.safetensors"));

    const gemma_source = try join(allocator, root, "gemma-source");
    try ensureDirectory(io, gemma_source);
    try writeText(io, try join(allocator, gemma_source, "config.json"), "{\"_name_or_path\":\"Google/Gemma4-21B\",\"model_type\":\"gemma4\",\"vision_config\":{\"hidden_size\":1152}}\n");
    try writeSafetensorsHeaderFixture(io, try join(allocator, gemma_source, "model.safetensors"), "{\"embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[4,4],\"data_offsets\":[0,32]},\"decoder.layers.0.mlp.up_proj.weight\":{\"dtype\":\"BF16\",\"shape\":[4,4],\"data_offsets\":[32,64]},\"mm_projector.weight\":{\"dtype\":\"F16\",\"shape\":[4,4],\"data_offsets\":[64,96]}}", 96);
    const gemma_bundle = try join(allocator, root, "gemma-bundle");
    _ = try runCommand(allocator, io, importer, &.{ gemma_source, "--output-root", gemma_bundle, "--link-mode", "copy" }, 0, null);
    try expectFileContains(allocator, io, try join(allocator, gemma_bundle, "manifest.mizu"), "family = gemma4");
    try expectFileContains(allocator, io, try join(allocator, gemma_bundle, "mizu_import/tensors.tsv"), "mm_projector.weight|multimodal_projector");

    const outside = try join(allocator, root, "outside.safetensors");
    try writeSafetensorsFixture(io, outside);
    const escape = try join(allocator, root, "hf-escape");
    try ensureDirectory(io, escape);
    const escape_link = try join(allocator, escape, "escape.safetensors");
    try Dir.symLinkAbsolute(io, outside, escape_link, .{});
    try writeText(io, try join(allocator, escape, "config.json"), "{\"model_type\":\"qwen3_5\"}\n");
    try writeText(io, try join(allocator, escape, "model.safetensors.index.json"), "{\"weight_map\":{\"weight\":\"escape.safetensors\"}}\n");
    _ = try runCommand(allocator, io, importer, &.{ escape, "--output-root", try join(allocator, root, "hf-escape-output") }, 2, "escapes model root");

    const unsafe_index = try join(allocator, root, "hf-unsafe-index");
    try ensureDirectory(io, unsafe_index);
    try writeText(io, try join(allocator, unsafe_index, "config.json"), "{\"model_type\":\"qwen3_5\"}\n");
    try writeText(io, try join(allocator, unsafe_index, "model.safetensors.index.json"), "{\"weight_map\":{\"weight\":\"../outside.safetensors\"}}\n");
    _ = try runCommand(allocator, io, importer, &.{ unsafe_index, "--output-root", try join(allocator, root, "hf-unsafe-index-output") }, 2, "unsafe shard path");

    const bad_range = try join(allocator, root, "hf-bad-range");
    try ensureDirectory(io, bad_range);
    try writeText(io, try join(allocator, bad_range, "config.json"), "{\"model_type\":\"qwen3_5\"}\n");
    try writeSafetensorsInvalidFixture(io, try join(allocator, bad_range, "model.safetensors"));
    _ = try runCommand(allocator, io, importer, &.{ bad_range, "--output-root", try join(allocator, root, "hf-bad-range-output") }, 2, "expected 512 bytes from dtype/shape");
    try expectNotExists(io, try join(allocator, root, "hf-bad-range-output/manifest.mizu"));

    const overlap = try join(allocator, root, "hf-overlap");
    try ensureDirectory(io, overlap);
    try writeText(io, try join(allocator, overlap, "config.json"), "{\"model_type\":\"qwen3_5\"}\n");
    try writeSafetensorsOverlapFixture(io, try join(allocator, overlap, "model.safetensors"));
    _ = try runCommand(allocator, io, importer, &.{ overlap, "--output-root", try join(allocator, root, "hf-overlap-output") }, 2, "safetensors ranges overlap");
    try expectNotExists(io, try join(allocator, root, "hf-overlap-output/manifest.mizu"));

    const oversized = try join(allocator, root, "hf-oversized");
    try ensureDirectory(io, oversized);
    try writeText(io, try join(allocator, oversized, "config.json"), "{\"model_type\":\"qwen3_5\"}\n");
    try writeOversizedSafetensorsHeader(io, try join(allocator, oversized, "model.safetensors"));
    _ = try runCommand(allocator, io, importer, &.{ oversized, "--output-root", try join(allocator, root, "hf-oversized-output") }, 2, "unreasonable or truncated safetensors header");
    try expectNotExists(io, try join(allocator, root, "hf-oversized-output/manifest.mizu"));

    const unsafe_output = try join(allocator, root, "hf-unsafe-output");
    const external_output = try join(allocator, root, "hf-external-output");
    try ensureDirectory(io, unsafe_output);
    try ensureDirectory(io, external_output);
    try Dir.symLinkAbsolute(io, external_output, try join(allocator, unsafe_output, "mizu_import"), .{});
    _ = try runCommand(allocator, io, importer, &.{ source, "--output-root", unsafe_output }, 2, "not a safe directory");
    try expectNotExists(io, try join(allocator, external_output, "tensors.tsv"));

    const injected = try join(allocator, root, "hf-injected");
    const injected_result = try runCommand(allocator, io, importer, &.{ source, "--output-root", injected, "--source-revision", "good\nprojector_present = true" }, 2, "unsupported line break in source revision");
    try expectNotExists(io, try join(allocator, injected, "manifest.mizu"));
    _ = injected_result;
}

fn testGguf(allocator: Allocator, io: Io, importer: []const u8, root: []const u8) !void {
    const model = try join(allocator, root, "qwen35.gguf");
    const projector = try join(allocator, root, "mmproj-qwen35.gguf");
    try writeGgufFixture(io, model, false, false, false);
    try writeGgufFixture(io, projector, true, false, false);
    const bundle = try join(allocator, root, "gguf-bundle");
    const summary = try runCommand(allocator, io, importer, &.{ model, "--projector-gguf", projector, "--output-root", bundle, "--link-mode", "copy" }, 0, null);
    try expectContains(summary.stdout, "family: qwen3_5");
    try expectContains(summary.stdout, "gguf_file_count: 2");
    try expectFileContains(allocator, io, try join(allocator, bundle, "manifest.mizu"), "projector_present = true");
    try expectFileContains(allocator, io, try join(allocator, bundle, "mizu_import/layout.mizu"), "gguf_inventory = gguf_tensors.tsv");
    try expectFileContains(allocator, io, try join(allocator, bundle, "mizu_import/tensors.tsv"), "token_embd.weight|embedding_table|f16|row_major");
    try expectFileContains(allocator, io, try join(allocator, bundle, "mizu_import/tensors.tsv"), "mm.0.weight|multimodal_projector|f16|packed");
    const inventory = try readFile(allocator, io, try join(allocator, bundle, "mizu_import/gguf_tensors.tsv"));
    try expectContains(inventory, "source_kind|ggml_type|normalized_dtype");
    try expectContains(inventory, "token_embd.weight|model|q4_k|f16|row_major|weights/qwen35.gguf|0|");
    try expectContains(inventory, "mm.0.weight|projector|f16|f16|packed|weights/mmproj-qwen35.gguf|131072|");
    try expectGgufSourceOffset(inventory, "token_embd.weight");
    try expectFileContains(allocator, io, try join(allocator, bundle, "mizu_import/projector/projector_assets.mizu"), "mm.0.weight|weights/mmproj-qwen35.gguf|offset=131072|ggml_type=f16");

    const repeat_bundle = try join(allocator, root, "gguf-repeat");
    _ = try runCommand(allocator, io, importer, &.{ model, "--projector-gguf", projector, "--output-root", repeat_bundle, "--link-mode", "copy" }, 0, null);
    for ([_][]const u8{ "manifest.mizu", "mizu_import/layout.mizu", "mizu_import/tensors.tsv", "mizu_import/gguf_tensors.tsv", "mizu_import/projector.mizu" }) |relative| {
        const first = try readFile(allocator, io, try join(allocator, bundle, relative));
        const second = try readFile(allocator, io, try join(allocator, repeat_bundle, relative));
        try expectEqualBytes(first, second, "GGUF output is not deterministic");
    }

    const dry_run = try join(allocator, root, "gguf-dry-run");
    _ = try runCommand(allocator, io, importer, &.{ model, "--output-root", dry_run, "--dry-run" }, 0, null);
    try expectNotExists(io, try join(allocator, dry_run, "manifest.mizu"));

    const malformed = try join(allocator, root, "truncated.gguf");
    try writeText(io, malformed, "GGUF\x03");
    _ = try runCommand(allocator, io, importer, &.{ malformed, "--output-root", try join(allocator, root, "truncated-output") }, 2, "truncated GGUF header");
    try expectNotExists(io, try join(allocator, root, "truncated-output/manifest.mizu"));

    const unsupported = try join(allocator, root, "unsupported.gguf");
    try writeGgufFixture(io, unsupported, false, true, false);
    _ = try runCommand(allocator, io, importer, &.{ unsupported, "--output-root", try join(allocator, root, "unsupported-output") }, 2, "unsupported GGML type id 99");
    try expectNotExists(io, try join(allocator, root, "unsupported-output/manifest.mizu"));

    const oversized_string = try join(allocator, root, "oversized-string.gguf");
    try writeOversizedGgufString(io, oversized_string);
    _ = try runCommand(allocator, io, importer, &.{ oversized_string, "--output-root", try join(allocator, root, "oversized-string-output") }, 2, "unreasonable or truncated GGUF string");
    try expectNotExists(io, try join(allocator, root, "oversized-string-output/manifest.mizu"));

    const collision_dir = try join(allocator, root, "collision-projector");
    try ensureDirectory(io, collision_dir);
    const collision = try join(allocator, collision_dir, "qwen35.gguf");
    try writeGgufFixture(io, collision, false, false, false);
    _ = try runCommand(allocator, io, importer, &.{ model, "--projector-gguf", collision, "--output-root", try join(allocator, root, "collision-output") }, 2, "GGUF basename would collide");
    try expectNotExists(io, try join(allocator, root, "collision-output/manifest.mizu"));

    const gemma_model = try join(allocator, root, "gemma4.gguf");
    const gemma_bundle = try join(allocator, root, "gemma4-bundle");
    try writeGemmaGgufFixture(io, gemma_model);
    _ = try runCommand(allocator, io, importer, &.{ gemma_model, "--output-root", gemma_bundle, "--link-mode", "copy" }, 0, null);
    try expectFileContains(allocator, io, try join(allocator, gemma_bundle, "manifest.mizu"), "family = gemma4");
    try expectFileContains(allocator, io, try join(allocator, gemma_bundle, "manifest.mizu"), "projector_present = false");
    try expectFileContains(allocator, io, try join(allocator, gemma_bundle, "mizu_import/tensors.tsv"), "token_embd.weight|embedding_table|f16|row_major|weights/gemma4.gguf|256x256|q5_k");
}

fn writeSafetensorsFixture(io: Io, path: []const u8) !void {
    const header = "{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[4,4],\"data_offsets\":[0,32]},\"visual.merger.weight\":{\"dtype\":\"F16\",\"shape\":[4,4],\"data_offsets\":[32,64]}}";
    try writeSafetensorsHeaderFixture(io, path, header, 64);
}

fn writeSafetensorsHeaderFixture(io: Io, path: []const u8, header: []const u8, payload_size: usize) !void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, header.len, .little);
    var file = try Dir.createFileAbsolute(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, &bytes);
    try file.writeStreamingAll(io, header);
    const payload = try std.heap.page_allocator.alloc(u8, payload_size);
    defer std.heap.page_allocator.free(payload);
    @memset(payload, 0);
    try file.writeStreamingAll(io, payload);
}

fn writeSafetensorsInvalidFixture(io: Io, path: []const u8) !void {
    const header = "{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[16,16],\"data_offsets\":[0,64]}}";
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, header.len, .little);
    var file = try Dir.createFileAbsolute(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, &bytes);
    try file.writeStreamingAll(io, header);
    const payload = try std.heap.page_allocator.alloc(u8, 64);
    defer std.heap.page_allocator.free(payload);
    @memset(payload, 0);
    try file.writeStreamingAll(io, payload);
}

fn writeSafetensorsOverlapFixture(io: Io, path: []const u8) !void {
    const header = "{\"first.weight\":{\"dtype\":\"BF16\",\"shape\":[4,4],\"data_offsets\":[0,32]},\"second.weight\":{\"dtype\":\"BF16\",\"shape\":[4,4],\"data_offsets\":[16,48]}}";
    var file = try Dir.createFileAbsolute(io, path, .{});
    defer file.close(io);
    var size: [8]u8 = undefined;
    std.mem.writeInt(u64, &size, header.len, .little);
    try file.writeStreamingAll(io, &size);
    try file.writeStreamingAll(io, header);
    try file.writeStreamingAll(io, &([_]u8{0} ** 64));
}

fn writeOversizedSafetensorsHeader(io: Io, path: []const u8) !void {
    var file = try Dir.createFileAbsolute(io, path, .{ .truncate = true });
    defer file.close(io);
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, 256 * 1024 * 1024 + 1, .little);
    try file.writeStreamingAll(io, &bytes);
}

fn writeOversizedGgufString(io: Io, path: []const u8) !void {
    var file = try Dir.createFileAbsolute(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, "GGUF");
    var u32_bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &u32_bytes, 3, .little);
    try file.writeStreamingAll(io, &u32_bytes);
    var u64_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &u64_bytes, 1, .little);
    try file.writeStreamingAll(io, &u64_bytes);
    std.mem.writeInt(u64, &u64_bytes, 0, .little);
    try file.writeStreamingAll(io, &u64_bytes);
    std.mem.writeInt(u64, &u64_bytes, 256 * 1024 * 1024 + 1, .little);
    try file.writeStreamingAll(io, &u64_bytes);
}

fn writeGgufFixture(io: Io, path: []const u8, projector: bool, unsupported_type: bool, integrated: bool) !void {
    const allocator = std.heap.page_allocator;
    var bytes: std.ArrayList(u8) = .empty;
    try appendBytes(allocator, &bytes, "GGUF");
    try appendU32(allocator, &bytes, 3);
    const tensors = if (projector) &[_]TensorFixture{
        .{ .name = "v.blk.0.attn_qkv.weight", .shape = &.{ 256, 256 }, .ggml_type = 1, .data_offset = 0 },
        .{ .name = "mm.0.weight", .shape = &.{ 256, 256 }, .ggml_type = 1, .data_offset = 131072 },
    } else if (integrated) &[_]TensorFixture{
        .{ .name = "token_embd.weight", .shape = &.{ 256, 256 }, .ggml_type = 12, .data_offset = 0 },
        .{ .name = "blk.0.attn_qkv.weight", .shape = &.{ 256, 256 }, .ggml_type = 13, .data_offset = 65536 },
        .{ .name = "output_norm.weight", .shape = &.{256}, .ggml_type = 0, .data_offset = 131072 },
        .{ .name = "output.weight", .shape = &.{ 256, 256 }, .ggml_type = 12, .data_offset = 196608 },
        .{ .name = "v.blk.0.attn_qkv.weight", .shape = &.{ 256, 256 }, .ggml_type = 1, .data_offset = 262144 },
        .{ .name = "mm.0.weight", .shape = &.{ 256, 256 }, .ggml_type = 1, .data_offset = 393216 },
        .{ .name = "mm.2.bias", .shape = &.{256}, .ggml_type = 0, .data_offset = 524288 },
    } else &[_]TensorFixture{
        .{ .name = "token_embd.weight", .shape = &.{ 256, 256 }, .ggml_type = if (unsupported_type) 99 else 12, .data_offset = 0 },
        .{ .name = "blk.0.attn_qkv.weight", .shape = &.{ 256, 256 }, .ggml_type = 13, .data_offset = 65536 },
        .{ .name = "output_norm.weight", .shape = &.{256}, .ggml_type = 0, .data_offset = 131072 },
        .{ .name = "output.weight", .shape = &.{ 256, 256 }, .ggml_type = 12, .data_offset = 196608 },
    };
    try appendU64(allocator, &bytes, tensors.len);
    try appendU64(allocator, &bytes, if (projector) 5 else if (integrated) 7 else 6);
    if (projector) {
        try appendMetadataString(allocator, &bytes, "general.architecture", "clip");
        try appendMetadataString(allocator, &bytes, "general.name", "Qwen3.5 projector");
        try appendMetadataString(allocator, &bytes, "general.type", "mmproj");
        try appendMetadataU32(allocator, &bytes, "general.file_type", 1);
        try appendMetadataBool(allocator, &bytes, "clip.has_vision_encoder", true);
    } else {
        try appendMetadataString(allocator, &bytes, "general.architecture", "qwen35");
        try appendMetadataString(allocator, &bytes, "general.name", "Qwen3.5 9B");
        try appendMetadataString(allocator, &bytes, "general.type", "model");
        try appendMetadataU32(allocator, &bytes, "general.file_type", 15);
        try appendMetadataU32(allocator, &bytes, "general.quantization_version", 2);
        try appendMetadataString(allocator, &bytes, "tokenizer.ggml.model", "gpt2");
        if (integrated) try appendMetadataBool(allocator, &bytes, "clip.has_vision_encoder", true);
    }
    for (tensors) |tensor| {
        try appendString(allocator, &bytes, tensor.name);
        try appendU32(allocator, &bytes, @intCast(tensor.shape.len));
        for (tensor.shape) |dimension| try appendU64(allocator, &bytes, dimension);
        try appendU32(allocator, &bytes, tensor.ggml_type);
        try appendU64(allocator, &bytes, tensor.data_offset);
    }
    const aligned_header_length = (bytes.items.len + 31) & ~@as(usize, 31);
    try appendZeros(allocator, &bytes, aligned_header_length - bytes.items.len);
    var payload_length: u64 = 0;
    for (tensors) |tensor| {
        const size = try ggmlSize(tensor);
        payload_length = @max(payload_length, tensor.data_offset + size);
    }
    try appendZeros(allocator, &bytes, payload_length);
    var file = try Dir.createFileAbsolute(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, bytes.items);
}

fn writeGemmaGgufFixture(io: Io, path: []const u8) !void {
    const allocator = std.heap.page_allocator;
    var bytes: std.ArrayList(u8) = .empty;
    try appendBytes(allocator, &bytes, "GGUF");
    try appendU32(allocator, &bytes, 3);
    const tensors = [_]TensorFixture{
        .{ .name = "token_embd.weight", .shape = &.{ 256, 256 }, .ggml_type = 13, .data_offset = 0 },
        .{ .name = "blk.0.ffn_gate.weight", .shape = &.{ 256, 256 }, .ggml_type = 13, .data_offset = 45056 },
        .{ .name = "output_norm.weight", .shape = &.{256}, .ggml_type = 0, .data_offset = 90112 },
    };
    try appendU64(allocator, &bytes, tensors.len);
    try appendU64(allocator, &bytes, 6);
    try appendMetadataString(allocator, &bytes, "general.architecture", "gemma4");
    try appendMetadataString(allocator, &bytes, "general.name", "Gemma-4 fixture");
    try appendMetadataString(allocator, &bytes, "general.type", "model");
    try appendMetadataU32(allocator, &bytes, "general.file_type", 29);
    try appendMetadataU32(allocator, &bytes, "general.quantization_version", 2);
    try appendMetadataString(allocator, &bytes, "tokenizer.ggml.model", "gemma4");
    for (tensors) |tensor| {
        try appendString(allocator, &bytes, tensor.name);
        try appendU32(allocator, &bytes, @intCast(tensor.shape.len));
        for (tensor.shape) |dimension| try appendU64(allocator, &bytes, dimension);
        try appendU32(allocator, &bytes, tensor.ggml_type);
        try appendU64(allocator, &bytes, tensor.data_offset);
    }
    const aligned_header_length = (bytes.items.len + 31) & ~@as(usize, 31);
    try appendZeros(allocator, &bytes, aligned_header_length - bytes.items.len);
    try appendZeros(allocator, &bytes, 91136);
    var file = try Dir.createFileAbsolute(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, bytes.items);
}

fn expectGgufSourceOffset(inventory: []const u8, tensor_name: []const u8) !void {
    var lines = std.mem.splitScalar(u8, inventory, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, tensor_name) or line.len <= tensor_name.len or line[tensor_name.len] != '|') continue;
        var fields = std.mem.splitScalar(u8, line, '|');
        for (0..6) |_| _ = fields.next() orelse return error.MalformedGgufInventoryRow;
        const data_offset_text = fields.next() orelse return error.MalformedGgufInventoryRow;
        const source_offset_text = fields.next() orelse return error.MalformedGgufInventoryRow;
        const data_offset = try std.fmt.parseInt(u64, data_offset_text, 10);
        const source_offset = try std.fmt.parseInt(u64, source_offset_text, 10);
        if (source_offset <= data_offset) return error.InvalidGgufSourceOffset;
        return;
    }
    return error.MissingGgufInventoryRow;
}

fn ggmlSize(tensor: TensorFixture) !u64 {
    const quant: [2]u64 = switch (tensor.ggml_type) {
        0 => .{ 1, 4 },
        1 => .{ 1, 2 },
        12 => .{ 256, 144 },
        13 => .{ 256, 176 },
        99 => .{ 256, 144 },
        else => return error.InvalidFixtureGgmlType,
    };
    if (tensor.shape[0] % quant[0] != 0) return error.InvalidFixtureShape;
    var rows: u64 = 1;
    for (tensor.shape[1..]) |dimension| rows *= dimension;
    return rows * (tensor.shape[0] / quant[0]) * quant[1];
}

fn appendMetadataString(allocator: Allocator, bytes: *std.ArrayList(u8), key: []const u8, value: []const u8) !void {
    try appendString(allocator, bytes, key);
    try appendU32(allocator, bytes, 8);
    try appendString(allocator, bytes, value);
}

fn appendMetadataU32(allocator: Allocator, bytes: *std.ArrayList(u8), key: []const u8, value: u32) !void {
    try appendString(allocator, bytes, key);
    try appendU32(allocator, bytes, 4);
    try appendU32(allocator, bytes, value);
}

fn appendMetadataBool(allocator: Allocator, bytes: *std.ArrayList(u8), key: []const u8, value: bool) !void {
    try appendString(allocator, bytes, key);
    try appendU32(allocator, bytes, 7);
    try bytes.append(allocator, @intFromBool(value));
}

fn appendString(allocator: Allocator, bytes: *std.ArrayList(u8), text: []const u8) !void {
    try appendU64(allocator, bytes, text.len);
    try appendBytes(allocator, bytes, text);
}

fn appendU32(allocator: Allocator, bytes: *std.ArrayList(u8), value: u32) !void {
    var encoded: [4]u8 = undefined;
    std.mem.writeInt(u32, &encoded, value, .little);
    try appendBytes(allocator, bytes, &encoded);
}

fn appendU64(allocator: Allocator, bytes: *std.ArrayList(u8), value: u64) !void {
    var encoded: [8]u8 = undefined;
    std.mem.writeInt(u64, &encoded, value, .little);
    try appendBytes(allocator, bytes, &encoded);
}

fn appendBytes(allocator: Allocator, bytes: *std.ArrayList(u8), value: []const u8) !void {
    try bytes.appendSlice(allocator, value);
}

fn appendZeros(allocator: Allocator, bytes: *std.ArrayList(u8), count: u64) !void {
    const length: usize = @intCast(count);
    const zeros = try allocator.alloc(u8, length);
    @memset(zeros, 0);
    try bytes.appendSlice(allocator, zeros);
}

fn runCommand(allocator: Allocator, io: Io, executable: []const u8, arguments: []const []const u8, expected_exit: u8, stderr_contains: ?[]const u8) !std.process.RunResult {
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.append(allocator, executable);
    try argv.appendSlice(allocator, arguments);
    const result = try std.process.run(allocator, io, .{ .argv = argv.items });
    const actual_exit: u8 = switch (result.term) {
        .exited => |code| code,
        else => 255,
    };
    if (actual_exit != expected_exit) {
        try report(io, "command {s}: expected exit {d}, got {d}\nstdout:\n{s}\nstderr:\n{s}", .{ executable, expected_exit, actual_exit, result.stdout, result.stderr });
        return error.UnexpectedCommandExit;
    }
    if (stderr_contains) |needle| try expectContains(result.stderr, needle);
    return result;
}

fn expectContains(text: []const u8, needle: []const u8) !void {
    if (std.mem.indexOf(u8, text, needle) == null) return error.ExpectedTextMissing;
}

fn expectFileContains(allocator: Allocator, io: Io, path: []const u8, needle: []const u8) !void {
    const bytes = try readFile(allocator, io, path);
    try expectContains(bytes, needle);
}

fn readFile(allocator: Allocator, io: Io, path: []const u8) ![]u8 {
    var file = try Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    var reader = file.readerStreaming(io, &.{});
    return reader.interface.readAlloc(allocator, @intCast(stat.size));
}

fn expectEqualBytes(expected: []const u8, actual: []const u8, message: []const u8) !void {
    if (!std.mem.eql(u8, expected, actual)) {
        std.debug.print("{s}\nexpected: {s}\nactual: {s}\n", .{ message, expected, actual });
        return error.BytesDiffer;
    }
}

fn ensureDirectory(io: Io, path: []const u8) !void {
    var dir = try Dir.cwd().createDirPathOpen(io, path, .{});
    dir.close(io);
}

fn writeText(io: Io, path: []const u8, content: []const u8) !void {
    var file = try Dir.createFileAbsolute(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, content);
}

fn expectFileExists(io: Io, path: []const u8) !void {
    const file = try Dir.openFileAbsolute(io, path, .{});
    file.close(io);
}

fn expectNotExists(io: Io, path: []const u8) !void {
    if (Dir.openFileAbsolute(io, path, .{})) |file| {
        file.close(io);
        return error.UnexpectedOutputExists;
    } else |_| {}
}

fn expectSymlink(io: Io, path: []const u8) !void {
    const parent_path = std.fs.path.dirname(path) orelse return error.InvalidFixturePath;
    const name = std.fs.path.basename(path);
    var parent = try Dir.openDirAbsolute(io, parent_path, .{});
    defer parent.close(io);
    const stat = try parent.statFile(io, name, .{ .follow_symlinks = false });
    if (stat.kind != .sym_link) return error.ExpectedSymlink;
}

fn join(allocator: Allocator, root: []const u8, child: []const u8) ![]const u8 {
    return std.fs.path.join(allocator, &.{ root, child });
}

fn report(io: Io, comptime format: []const u8, args: anytype) !void {
    var buffer: [8192]u8 = undefined;
    var writer = Io.File.stderr().writerStreaming(io, &buffer);
    try writer.interface.print(format, args);
    try writer.interface.flush();
}
