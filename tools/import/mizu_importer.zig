const std = @import("std");
const Io = std.Io;
const Dir = Io.Dir;
const Allocator = std.mem.Allocator;

pub const Mode = enum { safetensors, gguf };
const LinkMode = enum { symlink, copy };
const max_safe_i64: u64 = 0x7fff_ffff_ffff_ffff;
const max_reasonable_count: u64 = 10_000_000;
const max_safetensors_header_bytes: u64 = 256 * 1024 * 1024;
const max_gguf_string_bytes: u64 = 256 * 1024 * 1024;
const max_gguf_header_bytes: u64 = 256 * 1024 * 1024;
const import_layout_version = 1;

const Tensor = struct {
    name: []const u8,
    role: []const u8,
    dtype: []const u8,
    storage_type: []const u8,
    layout: []const u8,
    shape: []const u64,
    source_kind: []const u8,
    source_path: []const u8,
    source_rel: []const u8,
    bundle_rel: []const u8,
    data_offset: u64 = 0,
    source_offset: u64 = 0,
};

const SourceFile = struct {
    path: []const u8,
    name: []const u8,
    kind: []const u8,
    size: u64,
    version: u32 = 0,
    metadata: std.StringHashMap(MetadataValue),
};

const MetadataValue = union(enum) {
    string: []const u8,
    signed: i64,
    unsigned: u64,
    float: f64,
    boolean: bool,
    other,
};

const Bundle = struct {
    model_path: []const u8,
    output_root: []const u8,
    family: []const u8,
    source_model_id: []const u8,
    source_revision: []const u8,
    source_hash_text: []const u8,
    tokenizer_name: []const u8,
    has_projector: bool,
    projector_revision: []const u8,
    tensors: []Tensor,
    sources: []SourceFile,
    mode: Mode,
};

const Cli = struct {
    model_path: ?[]const u8 = null,
    projector_path: ?[]const u8 = null,
    output_root: ?[]const u8 = null,
    family: []const u8 = "auto",
    source_model_id: []const u8 = "",
    source_revision: []const u8 = "",
    link_mode: LinkMode = .symlink,
    force: bool = false,
    dry_run: bool = false,
    help: bool = false,
};

pub fn run(init: std.process.Init, mode: Mode) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(init.gpa);
    defer init.gpa.free(args);

    const cli = parseArgs(args[1..], mode) catch |err| {
        try writeErr(init.io, "mizu-import: invalid arguments ({s}); use --help for usage\n", .{@errorName(err)});
        std.process.exit(2);
    };
    if (cli.help) {
        try printHelp(init.io, mode);
        return;
    }

    const bundle = buildBundle(allocator, init.io, mode, cli) catch |err| switch (err) {
        error.ReportedDiagnostic => std.process.exit(2),
        else => {
            try writeErr(init.io, "mizu-import: {s}\n", .{@errorName(err)});
            std.process.exit(2);
        },
    };
    if (!cli.dry_run) {
        writeBundle(allocator, init.io, bundle, cli.link_mode, cli.force) catch |err| switch (err) {
            error.ReportedDiagnostic => std.process.exit(2),
            else => {
                try writeErr(init.io, "mizu-import: output failed ({s})\n", .{@errorName(err)});
                std.process.exit(2);
            },
        };
    }
    try printSummary(init.io, bundle);
}

fn parseArgs(args: []const []const u8, mode: Mode) !Cli {
    var cli = Cli{};
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            cli.help = true;
        } else if (std.mem.eql(u8, arg, "--output-root")) {
            cli.output_root = try optionValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--family")) {
            cli.family = try optionValue(args, &index);
            if (!std.mem.eql(u8, cli.family, "auto") and
                !std.mem.eql(u8, cli.family, "qwen3_5") and
                !std.mem.eql(u8, cli.family, "gemma4")) return error.InvalidFamily;
        } else if (std.mem.eql(u8, arg, "--source-model-id")) {
            cli.source_model_id = try optionValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--source-revision")) {
            cli.source_revision = try optionValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--link-mode")) {
            const value = try optionValue(args, &index);
            if (std.mem.eql(u8, value, "symlink")) {
                cli.link_mode = .symlink;
            } else if (std.mem.eql(u8, value, "copy")) {
                cli.link_mode = .copy;
            } else return error.InvalidLinkMode;
        } else if (std.mem.eql(u8, arg, "--projector-gguf")) {
            if (mode != .gguf) return error.UnsupportedOption;
            cli.projector_path = try optionValue(args, &index);
        } else if (std.mem.eql(u8, arg, "--force")) {
            cli.force = true;
        } else if (std.mem.eql(u8, arg, "--dry-run")) {
            cli.dry_run = true;
        } else if (std.mem.startsWith(u8, arg, "-")) {
            return error.UnknownOption;
        } else if (cli.model_path == null) {
            cli.model_path = arg;
        } else {
            return error.UnexpectedArgument;
        }
    }
    if (cli.help) return cli;
    if (cli.model_path == null) return error.MissingModelPath;
    return cli;
}

fn optionValue(args: []const []const u8, index: *usize) ![]const u8 {
    if (index.* + 1 >= args.len) return error.MissingOptionValue;
    index.* += 1;
    return args[index.*];
}

fn printHelp(io: Io, mode: Mode) !void {
    if (mode == .gguf) {
        try writeOut(io, "Usage: zig run tools/import/gguf_to_mizu.zig -- MODEL.gguf [options]\n", .{});
        try writeOut(io, "  --projector-gguf PATH  Optional paired projector GGUF.\n", .{});
    } else {
        try writeOut(io, "Usage: zig run tools/import/hf_safetensors_to_mizu.zig -- MODEL_DIR [options]\n", .{});
    }
    try writeOut(io, "  --output-root PATH     Bundle destination. Defaults beside the input.\n", .{});
    try writeOut(io, "  --family auto|qwen3_5|gemma4\n", .{});
    try writeOut(io, "  --source-model-id ID   Override source model identity.\n", .{});
    try writeOut(io, "  --source-revision REV  Override source revision identity.\n", .{});
    try writeOut(io, "  --link-mode symlink|copy\n", .{});
    try writeOut(io, "  --dry-run              Validate and summarize without writing.\n", .{});
    try writeOut(io, "  --force                Replace existing generated files and weights.\n", .{});
    try writeOut(io, "  --help                 Show this help.\n", .{});
}

fn buildBundle(allocator: Allocator, io: Io, mode: Mode, cli: Cli) !Bundle {
    const input_path = try absolutePath(allocator, io, cli.model_path.?);
    const model_path = try canonicalInput(allocator, io, input_path, mode == .safetensors);
    const output_root = if (cli.output_root) |path|
        try canonicalOutputRoot(allocator, io, path)
    else if (mode == .safetensors)
        model_path
    else
        try defaultOutputRoot(allocator, model_path);

    if (mode == .safetensors) return buildSafetensorsBundle(allocator, io, model_path, output_root, cli);
    return buildGgufBundle(allocator, io, model_path, output_root, cli);
}

fn canonicalOutputRoot(allocator: Allocator, io: Io, path: []const u8) ![]const u8 {
    var candidate = try absolutePath(allocator, io, path);
    var missing_components: std.ArrayList([]const u8) = .empty;
    while (true) {
        var dir = Dir.openDirAbsolute(io, candidate, .{}) catch |err| switch (err) {
            error.FileNotFound, error.NotDir => {
                const parent = std.fs.path.dirname(candidate) orelse return candidate;
                if (std.mem.eql(u8, parent, candidate)) return candidate;
                try missing_components.append(allocator, std.fs.path.basename(candidate));
                candidate = parent;
                continue;
            },
            else => return err,
        };
        defer dir.close(io);
        var buffer: [Dir.max_path_bytes]u8 = undefined;
        const real_parent_length = try dir.realPath(io, &buffer);
        candidate = try allocator.dupe(u8, buffer[0..real_parent_length]);
        for (0..missing_components.items.len) |index| {
            const component = missing_components.items[missing_components.items.len - index - 1];
            candidate = try std.fs.path.join(allocator, &.{ candidate, component });
        }
        return candidate;
    }
}

fn absolutePath(allocator: Allocator, io: Io, path: []const u8) ![]const u8 {
    if (std.fs.path.isAbsolute(path)) return std.fs.path.resolve(allocator, &.{path});
    const cwd = try std.process.currentPathAlloc(io, allocator);
    return std.fs.path.resolve(allocator, &.{ cwd, path });
}

fn canonicalInput(allocator: Allocator, io: Io, path: []const u8, directory: bool) ![]const u8 {
    if (directory) {
        var dir = Dir.openDirAbsolute(io, path, .{}) catch {
            try writeErr(io, "model root does not exist or is not a directory: {s}\n", .{path});
            return error.ReportedDiagnostic;
        };
        defer dir.close(io);
        var buffer: [Dir.max_path_bytes]u8 = undefined;
        const length = try dir.realPath(io, &buffer);
        return allocator.dupe(u8, buffer[0..length]);
    }
    const resolved = Dir.realPathFileAbsoluteAlloc(io, path, allocator) catch {
        try writeErr(io, "model file does not exist or cannot be resolved: {s}\n", .{path});
        return error.ReportedDiagnostic;
    };
    return resolved;
}

fn defaultOutputRoot(allocator: Allocator, model_path: []const u8) ![]const u8 {
    const parent = std.fs.path.dirname(model_path) orelse ".";
    const stem = pathStem(model_path);
    const name = try std.fmt.allocPrint(allocator, "{s}.mizu", .{stem});
    return std.fs.path.join(allocator, &.{ parent, name });
}

fn pathIsWithin(root: []const u8, candidate: []const u8) bool {
    if (std.mem.eql(u8, root, candidate)) return true;
    if (!std.mem.startsWith(u8, candidate, root)) return false;
    if (root.len == 0 or root[root.len - 1] == std.fs.path.sep) return true;
    return candidate.len > root.len and candidate[root.len] == std.fs.path.sep;
}

fn safeRelative(allocator: Allocator, root: []const u8, file: []const u8) ![]const u8 {
    if (!pathIsWithin(root, file) or std.mem.eql(u8, root, file)) return error.InputPathEscapesRoot;
    const start = if (root[root.len - 1] == std.fs.path.sep) root.len else root.len + 1;
    return std.fs.path.join(allocator, &.{file[start..]});
}

fn readJsonFile(allocator: Allocator, io: Io, path: []const u8, optional: bool) !std.json.Value {
    var file = Dir.openFileAbsolute(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => if (optional) return .{ .object = .{} } else {
            try writeErr(io, "required JSON file is missing: {s}\n", .{path});
            return error.ReportedDiagnostic;
        },
        else => return err,
    };
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size > max_safetensors_header_bytes) {
        try writeErr(io, "JSON file is too large to inspect safely: {s}\n", .{path});
        return error.ReportedDiagnostic;
    }
    var file_reader = file.readerStreaming(io, &.{});
    const bytes = try file_reader.interface.readAlloc(allocator, @intCast(stat.size));
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch {
        try writeErr(io, "invalid JSON in {s}\n", .{path});
        return error.ReportedDiagnostic;
    };
    if (parsed.value != .object) {
        try writeErr(io, "expected JSON object in {s}\n", .{path});
        return error.ReportedDiagnostic;
    }
    return parsed.value;
}

fn jsonString(value: std.json.Value, default: []const u8) []const u8 {
    return if (value == .string) value.string else default;
}

fn jsonObjectGet(value: std.json.Value, key: []const u8) std.json.Value {
    if (value != .object) return .null;
    return value.object.get(key) orelse .null;
}

fn jsonHasKey(value: std.json.Value, key: []const u8) bool {
    return value == .object and value.object.contains(key);
}

fn buildSafetensorsBundle(
    allocator: Allocator,
    io: Io,
    model_root: []const u8,
    output_root: []const u8,
    cli: Cli,
) !Bundle {
    const config_path = try std.fs.path.join(allocator, &.{ model_root, "config.json" });
    const tokenizer_config_path = try std.fs.path.join(allocator, &.{ model_root, "tokenizer_config.json" });
    const config = try readJsonFile(allocator, io, config_path, true);
    const tokenizer_config = try readJsonFile(allocator, io, tokenizer_config_path, true);
    const shard_paths = try discoverSafetensorsShards(allocator, io, model_root);
    if (shard_paths.len == 0) {
        try writeErr(io, "no .safetensors files found under {s}\n", .{model_root});
        return error.ReportedDiagnostic;
    }

    var tensors: std.ArrayList(Tensor) = .empty;
    var sources: std.ArrayList(SourceFile) = .empty;
    var output_names = std.StringHashMap(void).init(allocator);
    for (shard_paths) |path| {
        const name = std.fs.path.basename(path);
        if (output_names.contains(name)) {
            try writeErr(io, "safetensors shards have colliding basenames under weights/: {s}\n", .{name});
            return error.ReportedDiagnostic;
        }
        try output_names.put(name, {});
        const relative = try safeRelative(allocator, model_root, path);
        const parsed = try parseSafetensorsFile(allocator, io, path, relative, name);
        try tensors.appendSlice(allocator, parsed.tensors);
        try sources.append(allocator, parsed.source);
    }
    if (tensors.items.len == 0) {
        try writeErr(io, "no tensors found in safetensors headers under {s}\n", .{model_root});
        return error.ReportedDiagnostic;
    }
    std.mem.sort(Tensor, tensors.items, {}, tensorNameLessThan);

    const family = try resolveSafetensorsFamily(allocator, cli.family, config, model_root, io);
    const source_model_id = try resolveSafetensorsModelId(cli.source_model_id, config, model_root);
    const source_revision = resolveSafetensorsRevision(cli.source_revision, config);
    const tokenizer_name = resolveSafetensorsTokenizer(tokenizer_config, config, family);
    try validateLineField(io, source_model_id, "source model id");
    try validateLineField(io, source_revision, "source revision");
    try validateLineField(io, tokenizer_name, "tokenizer name");
    const source_hash_text = try safetensorsSourceHash(allocator, source_model_id, source_revision, tensors.items);

    var has_projector = configIndicatesProjector(config);
    for (tensors.items) |tensor| {
        if (isProjectorSideRole(tensor.role)) has_projector = true;
    }
    const projector_revision_text = try std.fmt.allocPrint(allocator, "{s}:projector", .{source_hash_text});
    const projector_revision = try stablePositiveI64(allocator, projector_revision_text);

    return .{
        .model_path = model_root,
        .output_root = output_root,
        .family = family,
        .source_model_id = source_model_id,
        .source_revision = source_revision,
        .source_hash_text = source_hash_text,
        .tokenizer_name = tokenizer_name,
        .has_projector = has_projector,
        .projector_revision = projector_revision,
        .tensors = try tensors.toOwnedSlice(allocator),
        .sources = try sources.toOwnedSlice(allocator),
        .mode = .safetensors,
    };
}

const ParsedGguf = struct {
    source: SourceFile,
    tensors: []GgufTensorHeader,
};

const GgufTensorHeader = struct {
    name: []const u8,
    shape: []const u64,
    ggml_type: []const u8,
    data_offset: u64,
    source_offset: u64 = 0,
    source_kind: []const u8,
    bundle_rel: []const u8,
};

fn buildGgufBundle(allocator: Allocator, io: Io, model_path: []const u8, output_root: []const u8, cli: Cli) !Bundle {
    var sources: std.ArrayList(SourceFile) = .empty;
    var tensors: std.ArrayList(Tensor) = .empty;
    const model = try parseGgufFile(allocator, io, model_path, "model");
    try sources.append(allocator, model.source);
    try appendGgufTensors(allocator, io, &tensors, model);

    if (cli.projector_path) |projector_input| {
        const projector_absolute = try absolutePath(allocator, io, projector_input);
        const projector_path = try canonicalInput(allocator, io, projector_absolute, false);
        if (std.mem.eql(u8, std.fs.path.basename(model_path), std.fs.path.basename(projector_path))) {
            try writeErr(io, "GGUF basename would collide under mizu_import/weights: {s}\n", .{std.fs.path.basename(model_path)});
            return error.ReportedDiagnostic;
        }
        const projector = try parseGgufFile(allocator, io, projector_path, "projector");
        try sources.append(allocator, projector.source);
        try appendGgufTensors(allocator, io, &tensors, projector);
    }
    if (tensors.items.len == 0) {
        try writeErr(io, "no tensors found in GGUF header for {s}\n", .{model_path});
        return error.ReportedDiagnostic;
    }
    std.mem.sort(Tensor, tensors.items, {}, tensorSourceNameLessThan);

    const family = try resolveGgufFamily(allocator, cli.family, model.source, io);
    const source_model_id = try resolveGgufModelId(allocator, cli.source_model_id, model.source, model_path);
    const source_revision = try resolveGgufRevision(allocator, cli.source_revision, sources.items);
    const tokenizer_name = try resolveGgufTokenizer(allocator, model.source, family);
    try validateLineField(io, source_model_id, "source model id");
    try validateLineField(io, source_revision, "source revision");
    try validateLineField(io, tokenizer_name, "tokenizer name");
    const source_hash_text = try ggufSourceHash(allocator, source_model_id, source_revision, sources.items, tensors.items);
    var has_projector = false;
    for (tensors.items) |tensor| if (isProjectorSideRole(tensor.role)) {
        has_projector = true;
    };
    const revision_input = try std.fmt.allocPrint(allocator, "{s}:projector", .{source_hash_text});

    return .{
        .model_path = model_path,
        .output_root = output_root,
        .family = family,
        .source_model_id = source_model_id,
        .source_revision = source_revision,
        .source_hash_text = source_hash_text,
        .tokenizer_name = tokenizer_name,
        .has_projector = has_projector,
        .projector_revision = try stablePositiveI64(allocator, revision_input),
        .tensors = try tensors.toOwnedSlice(allocator),
        .sources = try sources.toOwnedSlice(allocator),
        .mode = .gguf,
    };
}

fn appendGgufTensors(allocator: Allocator, io: Io, tensors: *std.ArrayList(Tensor), parsed: ParsedGguf) !void {
    try validateTsvField(io, parsed.source.name, parsed.source.path);
    const general_type = try metadataText(allocator, parsed.source.metadata, "general.type");
    for (parsed.tensors) |tensor| {
        const role = try classifyGgufTensor(allocator, tensor.name, tensor.source_kind, general_type);
        try validateTsvField(io, tensor.name, parsed.source.path);
        try tensors.append(allocator, .{
            .name = tensor.name,
            .role = role,
            .dtype = normalizeGgmlDtype(tensor.ggml_type),
            .storage_type = tensor.ggml_type,
            .layout = inferLayout(role, tensor.shape),
            .shape = tensor.shape,
            .source_kind = tensor.source_kind,
            .source_path = parsed.source.path,
            .source_rel = parsed.source.name,
            .bundle_rel = tensor.bundle_rel,
            .data_offset = tensor.data_offset,
            .source_offset = tensor.source_offset,
        });
    }
}

fn parseGgufFile(allocator: Allocator, io: Io, path: []const u8, source_kind: []const u8) !ParsedGguf {
    var file = try Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    var raw_reader = file.readerStreaming(io, &.{});
    var reader = GgufReader{ .allocator = allocator, .io = io, .reader = &raw_reader.interface, .path = path, .file_size = stat.size };

    var magic: [4]u8 = undefined;
    try reader.readExact(&magic);
    if (!std.mem.eql(u8, &magic, "GGUF")) {
        try writeErr(io, "invalid GGUF magic in {s}\n", .{path});
        return error.ReportedDiagnostic;
    }
    const version = try reader.readU32();
    if (version != 2 and version != 3) {
        try writeErr(io, "unsupported GGUF version in {s}: {d}\n", .{ path, version });
        return error.ReportedDiagnostic;
    }
    const tensor_count = try reader.readU64();
    const metadata_count = try reader.readU64();
    if (tensor_count == 0 or tensor_count > max_reasonable_count or metadata_count > max_reasonable_count) {
        try writeErr(io, "unreasonable GGUF tensor or metadata count in {s}\n", .{path});
        return error.ReportedDiagnostic;
    }

    var metadata = std.StringHashMap(MetadataValue).init(allocator);
    var index: u64 = 0;
    while (index < metadata_count) : (index += 1) {
        const key = try reader.readString();
        const value = try reader.readMetadataValue();
        try metadata.put(key, value);
    }

    var records: std.ArrayList(GgufTensorHeader) = .empty;
    index = 0;
    while (index < tensor_count) : (index += 1) {
        const name = try reader.readString();
        const rank = try reader.readU32();
        if (rank == 0 or rank > 8) {
            try writeErr(io, "tensor {s} in {s} has invalid rank {d}\n", .{ name, path, rank });
            return error.ReportedDiagnostic;
        }
        const shape = try allocator.alloc(u64, rank);
        for (shape) |*dimension| {
            dimension.* = try reader.readU64();
            if (dimension.* == 0 or dimension.* > max_reasonable_count) {
                try writeErr(io, "tensor {s} in {s} has invalid dimensions\n", .{ name, path });
                return error.ReportedDiagnostic;
            }
        }
        const ggml_type_id = try reader.readU32();
        const ggml_type = ggmlTypeName(ggml_type_id) orelse {
            try writeErr(io, "tensor {s} in {s} has unsupported GGML type id {d}\n", .{ name, path, ggml_type_id });
            return error.ReportedDiagnostic;
        };
        const data_offset = try reader.readU64();
        if (data_offset > max_safe_i64) {
            try writeErr(io, "tensor {s} in {s} has unreasonable data offset\n", .{ name, path });
            return error.ReportedDiagnostic;
        }
        try records.append(allocator, .{
            .name = name,
            .shape = shape,
            .ggml_type = ggml_type,
            .data_offset = data_offset,
            .source_kind = source_kind,
            .bundle_rel = try std.fs.path.join(allocator, &.{ "weights", std.fs.path.basename(path) }),
        });
    }

    const alignment = metadataInteger(metadata, "general.alignment", 32);
    const tensor_data_start = alignOffset(reader.bytes_read, if (alignment <= 0) 32 else @intCast(alignment));
    if (tensor_data_start >= stat.size) {
        try writeErr(io, "GGUF tensor data starts beyond EOF in {s}\n", .{path});
        return error.ReportedDiagnostic;
    }
    for (records.items) |*tensor| {
        const source_offset = std.math.add(u64, tensor_data_start, tensor.data_offset) catch {
            try writeErr(io, "tensor {s} in {s} has unreasonable source offset\n", .{ tensor.name, path });
            return error.ReportedDiagnostic;
        };
        if (source_offset > max_safe_i64 or source_offset >= stat.size) {
            try writeErr(io, "tensor {s} in {s} points beyond EOF at offset {d}\n", .{ tensor.name, path, source_offset });
            return error.ReportedDiagnostic;
        }
        const byte_count = try ggmlTensorByteCount(io, path, tensor.name, tensor.shape, tensor.ggml_type);
        if (byte_count > stat.size - source_offset) {
            try writeErr(io, "tensor {s} in {s} points beyond EOF at offset {d} with byte size {d}\n", .{ tensor.name, path, source_offset, byte_count });
            return error.ReportedDiagnostic;
        }
        tensor.source_offset = source_offset;
    }

    return .{
        .source = .{
            .path = path,
            .name = std.fs.path.basename(path),
            .kind = source_kind,
            .size = stat.size,
            .version = version,
            .metadata = metadata,
        },
        .tensors = try records.toOwnedSlice(allocator),
    };
}

const GgufReader = struct {
    allocator: Allocator,
    io: Io,
    reader: *Io.Reader,
    path: []const u8,
    file_size: u64,
    bytes_read: u64 = 0,

    fn readExact(self: *GgufReader, out: []u8) !void {
        if (out.len > self.file_size - self.bytes_read) {
            try writeErr(self.io, "truncated GGUF header in {s}\n", .{self.path});
            return error.ReportedDiagnostic;
        }
        if (self.bytes_read > max_gguf_header_bytes or out.len > max_gguf_header_bytes - self.bytes_read) {
            try writeErr(self.io, "GGUF header exceeds the {d}-byte inspection limit in {s}\n", .{ max_gguf_header_bytes, self.path });
            return error.ReportedDiagnostic;
        }
        self.reader.readSliceAll(out) catch {
            try writeErr(self.io, "truncated GGUF header in {s}\n", .{self.path});
            return error.ReportedDiagnostic;
        };
        self.bytes_read += out.len;
    }

    fn readU32(self: *GgufReader) !u32 {
        var bytes: [4]u8 = undefined;
        try self.readExact(&bytes);
        return std.mem.readInt(u32, &bytes, .little);
    }

    fn readU64(self: *GgufReader) !u64 {
        var bytes: [8]u8 = undefined;
        try self.readExact(&bytes);
        return std.mem.readInt(u64, &bytes, .little);
    }

    fn readString(self: *GgufReader) ![]const u8 {
        const size = try self.readU64();
        if (size > max_gguf_string_bytes or size > self.file_size - self.bytes_read) {
            try writeErr(self.io, "unreasonable or truncated GGUF string in {s}\n", .{self.path});
            return error.ReportedDiagnostic;
        }
        const value = try self.allocator.alloc(u8, @intCast(size));
        try self.readExact(value);
        if (!std.unicode.utf8ValidateSlice(value)) {
            try writeErr(self.io, "invalid UTF-8 GGUF string in {s}\n", .{self.path});
            return error.ReportedDiagnostic;
        }
        return value;
    }

    fn skipString(self: *GgufReader) !void {
        const size = try self.readU64();
        if (size > max_gguf_string_bytes or size > self.file_size - self.bytes_read) {
            try writeErr(self.io, "unreasonable GGUF array string in {s}\n", .{self.path});
            return error.ReportedDiagnostic;
        }
        var remaining = size;
        var buffer: [4096]u8 = undefined;
        while (remaining > 0) {
            const span: usize = @intCast(@min(remaining, buffer.len));
            try self.readExact(buffer[0..span]);
            remaining -= span;
        }
    }

    fn readMetadataValue(self: *GgufReader) !MetadataValue {
        const value_type = try self.readU32();
        if (value_type == 9) {
            const element_type = try self.readU32();
            if (element_type == 9 or element_type > 12) {
                try writeErr(self.io, "unsupported GGUF metadata array element type {d} in {s}\n", .{ element_type, self.path });
                return error.ReportedDiagnostic;
            }
            const count = try self.readU64();
            if (count > max_reasonable_count) {
                try writeErr(self.io, "unreasonable GGUF metadata array length in {s}\n", .{self.path});
                return error.ReportedDiagnostic;
            }
            var index: u64 = 0;
            while (index < count) : (index += 1) {
                if (element_type == 8) try self.skipString() else _ = try self.readScalar(element_type);
            }
            return .other;
        }
        if (value_type > 12) {
            try writeErr(self.io, "unsupported GGUF metadata value type {d} in {s}\n", .{ value_type, self.path });
            return error.ReportedDiagnostic;
        }
        return self.readScalar(value_type);
    }

    fn readScalar(self: *GgufReader, value_type: u32) !MetadataValue {
        return switch (value_type) {
            0 => .{ .unsigned = try self.readU8() },
            1 => .{ .signed = @as(i8, @bitCast(try self.readU8())) },
            2 => .{ .unsigned = try self.readU16() },
            3 => .{ .signed = @as(i16, @bitCast(try self.readU16())) },
            4 => .{ .unsigned = try self.readU32() },
            5 => .{ .signed = @as(i64, @as(i32, @bitCast(try self.readU32()))) },
            6 => .{ .float = @as(f64, @floatCast(@as(f32, @bitCast(try self.readU32())))) },
            7 => .{ .boolean = (try self.readU8()) != 0 },
            8 => .{ .string = try self.readString() },
            10 => .{ .unsigned = try self.readU64() },
            11 => .{ .signed = @bitCast(try self.readU64()) },
            12 => .{ .float = @as(f64, @bitCast(try self.readU64())) },
            else => error.ReportedDiagnostic,
        };
    }

    fn readU8(self: *GgufReader) !u8 {
        var byte: [1]u8 = undefined;
        try self.readExact(&byte);
        return byte[0];
    }

    fn readU16(self: *GgufReader) !u16 {
        var bytes: [2]u8 = undefined;
        try self.readExact(&bytes);
        return std.mem.readInt(u16, &bytes, .little);
    }
};

const ParsedSafetensors = struct {
    tensors: []Tensor,
    source: SourceFile,
};

fn discoverSafetensorsShards(allocator: Allocator, io: Io, model_root: []const u8) ![][]const u8 {
    const index_path = try std.fs.path.join(allocator, &.{ model_root, "model.safetensors.index.json" });
    if (pathExists(io, index_path)) {
        const index = try readJsonFile(allocator, io, index_path, false);
        const weight_map = jsonObjectGet(index, "weight_map");
        if (weight_map != .object or weight_map.object.count() == 0) {
            try writeErr(io, "expected non-empty weight_map in {s}\n", .{index_path});
            return error.ReportedDiagnostic;
        }
        var unique = std.StringHashMap(void).init(allocator);
        var iterator = weight_map.object.iterator();
        while (iterator.next()) |entry| {
            if (entry.value_ptr.* != .string) {
                try writeErr(io, "safetensors weight_map must map tensor names to shard paths in {s}\n", .{index_path});
                return error.ReportedDiagnostic;
            }
            const shard_name = entry.value_ptr.string;
            if (shard_name.len == 0 or std.fs.path.isAbsolute(shard_name) or pathHasParentTraversal(shard_name)) {
                try writeErr(io, "unsafe shard path in {s}: {s}\n", .{ index_path, shard_name });
                return error.ReportedDiagnostic;
            }
            if (!std.mem.endsWith(u8, shard_name, ".safetensors")) {
                try writeErr(io, "unsupported shard path in {s}: {s}\n", .{ index_path, shard_name });
                return error.ReportedDiagnostic;
            }
            try unique.put(shard_name, {});
        }
        var paths: std.ArrayList([]const u8) = .empty;
        var names = unique.keyIterator();
        while (names.next()) |name| {
            const candidate = try std.fs.path.join(allocator, &.{ model_root, name.* });
            const canonical = canonicalInput(allocator, io, candidate, false) catch {
                try writeErr(io, "safetensors index references missing or unsafe shard: {s}\n", .{name.*});
                return error.ReportedDiagnostic;
            };
            if (!pathIsWithin(model_root, canonical)) {
                try writeErr(io, "safetensors shard escapes model root: {s}\n", .{name.*});
                return error.ReportedDiagnostic;
            }
            try paths.append(allocator, canonical);
        }
        std.mem.sort([]const u8, paths.items, {}, stringLessThan);
        return paths.toOwnedSlice(allocator);
    }

    var model_dir = try Dir.openDirAbsolute(io, model_root, .{ .iterate = true });
    defer model_dir.close(io);
    var walker = try model_dir.walk(allocator);
    defer walker.deinit();
    var paths: std.ArrayList([]const u8) = .empty;
    while (try walker.next(io)) |entry| {
        if (entry.kind != .file or std.mem.indexOfScalar(u8, entry.path, std.fs.path.sep) != null) continue;
        if (!std.mem.endsWith(u8, entry.basename, ".safetensors")) continue;
        const candidate = try std.fs.path.join(allocator, &.{ model_root, entry.basename });
        const canonical = try canonicalInput(allocator, io, candidate, false);
        if (!pathIsWithin(model_root, canonical)) {
            try writeErr(io, "safetensors shard escapes model root: {s}\n", .{entry.basename});
            return error.ReportedDiagnostic;
        }
        try paths.append(allocator, canonical);
    }
    std.mem.sort([]const u8, paths.items, {}, stringLessThan);
    return paths.toOwnedSlice(allocator);
}

fn parseSafetensorsFile(
    allocator: Allocator,
    io: Io,
    path: []const u8,
    relative: []const u8,
    basename: []const u8,
) !ParsedSafetensors {
    var file = try Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size < 8) {
        try writeErr(io, "invalid safetensors header in {s}\n", .{path});
        return error.ReportedDiagnostic;
    }
    var header_size_bytes: [8]u8 = undefined;
    var reader = file.readerStreaming(io, &.{});
    reader.interface.readSliceAll(&header_size_bytes) catch {
        try writeErr(io, "truncated safetensors header in {s}\n", .{path});
        return error.ReportedDiagnostic;
    };
    const header_size = std.mem.readInt(u64, &header_size_bytes, .little);
    if (header_size == 0 or header_size > max_safetensors_header_bytes or header_size > stat.size - 8) {
        try writeErr(io, "unreasonable or truncated safetensors header in {s}: {d} bytes\n", .{ path, header_size });
        return error.ReportedDiagnostic;
    }
    const header_bytes = try allocator.alloc(u8, @intCast(header_size));
    reader.interface.readSliceAll(header_bytes) catch {
        try writeErr(io, "truncated safetensors header in {s}\n", .{path});
        return error.ReportedDiagnostic;
    };
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, header_bytes, .{}) catch {
        try writeErr(io, "invalid safetensors JSON header in {s}\n", .{path});
        return error.ReportedDiagnostic;
    };
    const header = parsed.value;
    if (header != .object) {
        try writeErr(io, "expected safetensors header object in {s}\n", .{path});
        return error.ReportedDiagnostic;
    }
    const payload_size = stat.size - 8 - header_size;
    var tensors: std.ArrayList(Tensor) = .empty;
    var offsets: std.ArrayList([2]u64) = .empty;
    var iterator = header.object.iterator();
    while (iterator.next()) |entry| {
        const tensor_name = entry.key_ptr.*;
        if (std.mem.eql(u8, tensor_name, "__metadata__")) continue;
        if (entry.value_ptr.* != .object) {
            try writeErr(io, "tensor metadata for {s} in {s} is not an object\n", .{ tensor_name, path });
            return error.ReportedDiagnostic;
        }
        const dtype_name = jsonString(jsonObjectGet(entry.value_ptr.*, "dtype"), "");
        const dtype = safetensorsDtype(dtype_name) orelse {
            try writeErr(io, "unsupported safetensors dtype `{s}` for tensor {s} in {s}\n", .{ dtype_name, tensor_name, path });
            return error.ReportedDiagnostic;
        };
        const shape_value = jsonObjectGet(entry.value_ptr.*, "shape");
        if (shape_value != .array or shape_value.array.items.len == 0 or shape_value.array.items.len > 8) {
            try writeErr(io, "tensor {s} in {s} has invalid shape\n", .{ tensor_name, path });
            return error.ReportedDiagnostic;
        }
        const shape = try allocator.alloc(u64, shape_value.array.items.len);
        for (shape_value.array.items, 0..) |dimension, shape_index| {
            if (dimension != .integer or dimension.integer <= 0) {
                try writeErr(io, "tensor {s} in {s} has invalid shape\n", .{ tensor_name, path });
                return error.ReportedDiagnostic;
            }
            shape[shape_index] = @intCast(dimension.integer);
        }
        const offsets_value = jsonObjectGet(entry.value_ptr.*, "data_offsets");
        if (offsets_value != .array or offsets_value.array.items.len != 2 or
            offsets_value.array.items[0] != .integer or offsets_value.array.items[1] != .integer)
        {
            try writeErr(io, "tensor metadata for {s} in {s} is missing valid data_offsets\n", .{ tensor_name, path });
            return error.ReportedDiagnostic;
        }
        const start_signed = offsets_value.array.items[0].integer;
        const end_signed = offsets_value.array.items[1].integer;
        if (start_signed < 0 or end_signed <= start_signed) {
            try writeErr(io, "tensor {s} in {s} has invalid data_offsets\n", .{ tensor_name, path });
            return error.ReportedDiagnostic;
        }
        const start: u64 = @intCast(start_signed);
        const end: u64 = @intCast(end_signed);
        if (end > payload_size) {
            try writeErr(io, "tensor {s} in {s} points beyond EOF\n", .{ tensor_name, path });
            return error.ReportedDiagnostic;
        }
        const expected_size = try safetensorsByteCount(io, path, tensor_name, dtype_name, shape);
        if (end - start != expected_size) {
            try writeErr(io, "tensor {s} in {s} has data_offsets but expected {d} bytes from dtype/shape\n", .{ tensor_name, path, expected_size });
            return error.ReportedDiagnostic;
        }
        try offsets.append(allocator, .{ start, end });
        const role = try classifySafetensorsTensor(allocator, tensor_name);
        try validateTsvField(io, tensor_name, path);
        try tensors.append(allocator, .{
            .name = tensor_name,
            .role = role,
            .dtype = dtype,
            .storage_type = dtype,
            .layout = inferLayout(role, shape),
            .shape = shape,
            .source_kind = "model",
            .source_path = path,
            .source_rel = relative,
            .bundle_rel = try std.fs.path.join(allocator, &.{ "weights", basename }),
        });
    }

    if (offsets.items.len > 1) {
        std.mem.sort([2]u64, offsets.items, {}, offsetStartLessThan);
        for (1..offsets.items.len) |offset_index| {
            if (offsets.items[offset_index][0] < offsets.items[offset_index - 1][1]) {
                try writeErr(io, "safetensors ranges overlap in {s}\n", .{path});
                return error.ReportedDiagnostic;
            }
        }
    }

    const meta = std.StringHashMap(MetadataValue).init(allocator);
    return .{
        .tensors = try tensors.toOwnedSlice(allocator),
        .source = .{
            .path = path,
            .name = basename,
            .kind = "model",
            .size = @intCast(stat.size),
            .metadata = meta,
        },
    };
}

fn safetensorsDtype(value: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, value, "U8")) return "u8";
    if (std.mem.eql(u8, value, "I32")) return "i32";
    if (std.mem.eql(u8, value, "F16")) return "f16";
    if (std.mem.eql(u8, value, "BF16")) return "bf16";
    if (std.mem.eql(u8, value, "F32")) return "f32";
    return null;
}

fn safetensorsByteCount(io: Io, path: []const u8, tensor_name: []const u8, dtype_name: []const u8, shape: []const u64) !u64 {
    const element_bytes: u64 = if (std.mem.eql(u8, dtype_name, "U8")) 1 else if (std.mem.eql(u8, dtype_name, "I32") or std.mem.eql(u8, dtype_name, "F32")) 4 else 2;
    var count: u64 = 1;
    for (shape) |dimension| {
        if (count > max_safe_i64 / dimension) {
            try writeErr(io, "tensor {s} in {s} has unreasonable shape\n", .{ tensor_name, path });
            return error.ReportedDiagnostic;
        }
        count *= dimension;
    }
    if (count > max_safe_i64 / element_bytes) {
        try writeErr(io, "tensor {s} in {s} has unreasonable byte size\n", .{ tensor_name, path });
        return error.ReportedDiagnostic;
    }
    return count * element_bytes;
}

fn resolveSafetensorsFamily(allocator: Allocator, requested: []const u8, config: std.json.Value, model_root: []const u8, io: Io) ![]const u8 {
    if (!std.mem.eql(u8, requested, "auto")) return requested;
    const identity = try std.mem.join(allocator, " ", &.{
        jsonString(jsonObjectGet(config, "model_type"), ""),
        jsonString(jsonObjectGet(config, "_name_or_path"), ""),
        std.fs.path.basename(model_root),
        model_root,
    });
    const lowered = try asciiLower(allocator, identity);
    if (std.mem.indexOf(u8, lowered, "qwen") != null) return "qwen3_5";
    if (std.mem.indexOf(u8, lowered, "gemma") != null) return "gemma4";
    try writeErr(io, "could not infer model family; pass --family qwen3_5 or --family gemma4\n", .{});
    return error.ReportedDiagnostic;
}

fn resolveSafetensorsModelId(override: []const u8, config: std.json.Value, model_root: []const u8) ![]const u8 {
    if (override.len > 0) return override;
    for ([_][]const u8{ "_name_or_path", "name_or_path", "model_type" }) |key| {
        const value = jsonString(jsonObjectGet(config, key), "");
        const trimmed = std.mem.trim(u8, value, " \t\r\n");
        if (trimmed.len > 0) return trimmed;
    }
    return std.fs.path.basename(model_root);
}

fn resolveSafetensorsRevision(override: []const u8, config: std.json.Value) []const u8 {
    if (override.len > 0) return override;
    for ([_][]const u8{ "_commit_hash", "revision", "transformers_version" }) |key| {
        const value = jsonString(jsonObjectGet(config, key), "");
        const trimmed = std.mem.trim(u8, value, " \t\r\n");
        if (trimmed.len > 0) return trimmed;
    }
    return "imported-local";
}

fn resolveSafetensorsTokenizer(config_tokenizer: std.json.Value, config: std.json.Value, family: []const u8) []const u8 {
    for ([_]std.json.Value{ config_tokenizer, config }) |source| {
        for ([_][]const u8{ "tokenizer_class", "model_type" }) |key| {
            const value = jsonString(jsonObjectGet(source, key), "");
            if (std.mem.trim(u8, value, " \t\r\n").len > 0) return value;
        }
    }
    return family;
}

fn configIndicatesProjector(config: std.json.Value) bool {
    if (config != .object) return false;
    var iterator = config.object.iterator();
    while (iterator.next()) |entry| {
        const key = entry.key_ptr.*;
        if (containsIgnoreCase(key, "vision") or containsIgnoreCase(key, "projector")) return true;
        if (std.mem.eql(u8, key, "visual") or std.mem.eql(u8, key, "mm_vision_tower")) return true;
    }
    return false;
}

fn safetensorsSourceHash(allocator: Allocator, source_model_id: []const u8, source_revision: []const u8, tensors: []Tensor) ![]const u8 {
    const ordered = try allocator.dupe(Tensor, tensors);
    std.mem.sort(Tensor, ordered, {}, tensorNameLessThan);
    var digest: [32]u8 = undefined;
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(source_model_id);
    hasher.update(&.{0});
    hasher.update(source_revision);
    for (ordered) |tensor| {
        hasher.update(&.{0});
        hasher.update(tensor.name);
        hasher.update("|");
        hasher.update(tensor.dtype);
        hasher.update("|");
        try hashShape(allocator, &hasher, tensor.shape);
        hasher.update("|");
        hasher.update(tensor.source_rel);
    }
    hasher.final(&digest);
    return hexDigest(allocator, digest);
}

fn hashShape(allocator: Allocator, hasher: *std.crypto.hash.sha2.Sha256, shape: []const u64) !void {
    for (shape, 0..) |dimension, index| {
        if (index > 0) hasher.update("x");
        const text = try std.fmt.allocPrint(allocator, "{d}", .{dimension});
        hasher.update(text);
    }
}

fn stablePositiveI64(allocator: Allocator, text: []const u8) ![]const u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &digest, .{});
    const value = @as(u64, @bitCast(std.mem.readInt(i64, digest[0..8], .little))) & max_safe_i64;
    return std.fmt.allocPrint(allocator, "{d}", .{if (value == 0) 1 else value});
}

fn hexDigest(allocator: Allocator, digest: [32]u8) ![]const u8 {
    const out = try allocator.alloc(u8, digest.len * 2);
    const alphabet = "0123456789abcdef";
    for (digest, 0..) |byte, index| {
        out[index * 2] = alphabet[byte >> 4];
        out[index * 2 + 1] = alphabet[byte & 0xf];
    }
    return out;
}

fn classifySafetensorsTensor(allocator: Allocator, name: []const u8) ![]const u8 {
    const lowered = try asciiLower(allocator, name);
    if (containsAny(lowered, &.{ "mm_projector", "multi_modal_projector", "projector", "visual.merger", "vision_projector" })) return "multimodal_projector";
    if (isSafetensorsVisionName(lowered)) return "vision_encoder";
    if (containsAny(lowered, &.{ "embed_tokens", "token_embedding", "token_embd" })) return "embedding_table";
    if (std.mem.endsWith(u8, lowered, "lm_head.weight") or std.mem.eql(u8, lowered, "output.weight") or std.mem.indexOf(u8, lowered, "output_projection") != null) return "token_projection";
    if (std.mem.indexOf(u8, lowered, "norm") != null) return "normalization";
    if (containsAny(lowered, &.{ ".layers.", ".blocks.", "blk.", ".blk.", "decoder", "self_attn", ".mlp.", "attn_", "ffn_" })) return "decoder_stack";
    return "model_tensor";
}

fn isSafetensorsVisionName(lowered: []const u8) bool {
    for ([_][]const u8{ "vision", "visual.", ".visual.", "vision_tower.", ".vision_tower.", "vision_model.", ".vision_model.", "image_tower.", ".image_tower." }) |needle| {
        if (std.mem.indexOf(u8, lowered, needle) != null) return true;
    }
    return false;
}

fn isProjectorSideRole(role: []const u8) bool {
    return std.mem.eql(u8, role, "multimodal_projector") or std.mem.eql(u8, role, "vision_encoder");
}

fn inferLayout(role: []const u8, shape: []const u64) []const u8 {
    if (shape.len == 1) return "vector";
    if (std.mem.eql(u8, role, "decoder_stack") or isProjectorSideRole(role)) return "packed";
    if (shape.len == 2) return "row_major";
    return "tensor";
}

fn containsAny(text: []const u8, needles: []const []const u8) bool {
    for (needles) |needle| if (std.mem.indexOf(u8, text, needle) != null) return true;
    return false;
}

fn asciiLower(allocator: Allocator, text: []const u8) ![]const u8 {
    const out = try allocator.dupe(u8, text);
    for (out) |*byte| byte.* = std.ascii.toLower(byte.*);
    return out;
}

fn validateTsvField(io: Io, field: []const u8, path: []const u8) !void {
    if (std.mem.indexOfAny(u8, field, "|\r\n\x00") != null) {
        try writeErr(io, "unsupported TSV delimiter or line break in tensor name `{s}` in {s}\n", .{ field, path });
        return error.ReportedDiagnostic;
    }
}

fn validateLineField(io: Io, field: []const u8, label: []const u8) !void {
    if (std.mem.indexOfAny(u8, field, "\r\n\x00") != null) {
        try writeErr(io, "unsupported line break in {s}\n", .{label});
        return error.ReportedDiagnostic;
    }
}

fn containsIgnoreCase(text: []const u8, needle: []const u8) bool {
    if (needle.len > text.len) return false;
    for (0..text.len - needle.len + 1) |start| {
        if (std.ascii.eqlIgnoreCase(text[start .. start + needle.len], needle)) return true;
    }
    return false;
}

fn pathHasParentTraversal(path: []const u8) bool {
    var components = std.mem.tokenizeAny(u8, path, "/\\");
    while (components.next()) |component| if (std.mem.eql(u8, component, "..")) return true;
    return false;
}

fn pathExists(io: Io, path: []const u8) bool {
    const file = Dir.openFileAbsolute(io, path, .{}) catch return false;
    file.close(io);
    return true;
}

fn tensorNameLessThan(_: void, left: Tensor, right: Tensor) bool {
    return std.mem.lessThan(u8, left.name, right.name);
}

fn offsetStartLessThan(_: void, left: [2]u64, right: [2]u64) bool {
    return left[0] < right[0];
}

fn stringLessThan(_: void, left: []const u8, right: []const u8) bool {
    return std.mem.lessThan(u8, left, right);
}

fn writeBundle(allocator: Allocator, io: Io, bundle: Bundle, link_mode: LinkMode, force: bool) !void {
    var root_dir = try Dir.cwd().createDirPathOpen(io, bundle.output_root, .{});
    defer root_dir.close(io);
    var import_dir = root_dir.createDirPathOpen(io, "mizu_import", .{ .open_options = .{ .follow_symlinks = false } }) catch {
        try writeErr(io, "output path mizu_import is not a safe directory under {s}\n", .{bundle.output_root});
        return error.ReportedDiagnostic;
    };
    defer import_dir.close(io);
    var weights_dir = import_dir.createDirPathOpen(io, "weights", .{ .open_options = .{ .follow_symlinks = false } }) catch {
        try writeErr(io, "output path mizu_import/weights is not a safe directory\n", .{});
        return error.ReportedDiagnostic;
    };
    defer weights_dir.close(io);
    var projector_dir = import_dir.createDirPathOpen(io, "projector", .{ .open_options = .{ .follow_symlinks = false } }) catch {
        try writeErr(io, "output path mizu_import/projector is not a safe directory\n", .{});
        return error.ReportedDiagnostic;
    };
    defer projector_dir.close(io);

    try ensureCanReplace(io, &root_dir, "manifest.mizu", force);
    for ([_][]const u8{ "layout.mizu", "tensors.tsv", "modalities.tsv", "projector.mizu" }) |name| {
        try ensureCanReplace(io, &import_dir, name, force);
    }
    if (bundle.mode == .gguf) try ensureCanReplace(io, &import_dir, "gguf_tensors.tsv", force);
    try ensureCanReplace(io, &projector_dir, "projector_assets.mizu", force);
    for (bundle.sources) |source| try ensureCanReplace(io, &weights_dir, source.name, force);

    try writeAtomic(io, &root_dir, "manifest.mizu", try renderRootManifest(allocator, bundle), force);
    try writeAtomic(io, &import_dir, "layout.mizu", try renderLayout(allocator, bundle), force);
    try writeAtomic(io, &import_dir, "tensors.tsv", try renderTensors(allocator, bundle), force);
    if (bundle.mode == .gguf) try writeAtomic(io, &import_dir, "gguf_tensors.tsv", try renderGgufTensors(allocator, bundle), force);
    try writeAtomic(io, &import_dir, "modalities.tsv", renderModalities(bundle), force);
    try writeAtomic(io, &import_dir, "projector.mizu", try renderProjector(allocator, bundle), force);
    try writeAtomic(io, &projector_dir, "projector_assets.mizu", try renderProjectorAssets(allocator, bundle), force);
    try materializeSources(allocator, io, bundle, &weights_dir, link_mode, force);
}

fn ensureCanReplace(io: Io, dir: *Dir, path: []const u8, force: bool) !void {
    const stat = dir.statFile(io, path, .{ .follow_symlinks = false }) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    if (!force) {
        try writeErr(io, "refusing to overwrite {s}; pass --force\n", .{path});
        return error.ReportedDiagnostic;
    }
    if (stat.kind != .file and stat.kind != .sym_link) {
        try writeErr(io, "cannot replace non-file output: {s}\n", .{path});
        return error.ReportedDiagnostic;
    }
}

fn writeAtomic(io: Io, dir: *Dir, path: []const u8, bytes: []const u8, replace: bool) !void {
    var atomic_file = try dir.createFileAtomic(io, path, .{ .replace = replace });
    defer atomic_file.deinit(io);
    try atomic_file.file.writeStreamingAll(io, bytes);
    try atomic_file.file.sync(io);
    if (replace) try atomic_file.replace(io) else try atomic_file.link(io);
}

fn materializeSources(allocator: Allocator, io: Io, bundle: Bundle, weights_dir: *Dir, link_mode: LinkMode, force: bool) !void {
    const weights_path = try std.fs.path.join(allocator, &.{ bundle.output_root, "mizu_import", "weights" });
    const cwd = try std.process.currentPathAlloc(io, allocator);
    for (bundle.sources) |source| {
        if (force) {
            const existing = weights_dir.statFile(io, source.name, .{ .follow_symlinks = false }) catch |err| switch (err) {
                error.FileNotFound => null,
                else => return err,
            };
            if (existing) |stat| {
                if (stat.kind != .file and stat.kind != .sym_link) {
                    try writeErr(io, "cannot replace directory with imported source: {s}\n", .{source.name});
                    return error.ReportedDiagnostic;
                }
                try weights_dir.deleteFile(io, source.name);
            }
        }
        if (link_mode == .copy) {
            try Dir.copyFileAbsolute(source.path, try std.fs.path.join(allocator, &.{ weights_path, source.name }), io, .{ .replace = force });
        } else {
            const relative_target = try std.fs.path.relative(allocator, cwd, null, weights_path, source.path);
            try weights_dir.symLinkAtomic(io, relative_target, source.name, .{});
        }
    }
}

fn renderRootManifest(allocator: Allocator, bundle: Bundle) ![]const u8 {
    return std.fmt.allocPrint(
        allocator,
        "# Generated by tools/import/{s}.zig\n" ++
            "family = {s}\nsource_model_id = {s}\nsource_revision = {s}\nsource_hash_text = {s}\n" ++
            "tokenizer = {s}\nmodel_features = {s}\nprojector_present = {s}\n",
        .{
            if (bundle.mode == .gguf) "gguf_to_mizu" else "hf_safetensors_to_mizu",
            bundle.family,
            bundle.source_model_id,
            bundle.source_revision,
            bundle.source_hash_text,
            bundle.tokenizer_name,
            if (bundle.has_projector) "multimodal,projector" else "none",
            if (bundle.has_projector) "true" else "false",
        },
    );
}

fn renderLayout(allocator: Allocator, bundle: Bundle) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try appendFmt(allocator, &out, "# Generated by tools/import/{s}.zig\n", .{if (bundle.mode == .gguf) "gguf_to_mizu" else "hf_safetensors_to_mizu"});
    try appendFmt(
        allocator,
        &out,
        "layout_version = {d}\nfamily = {s}\nsource_model_id = {s}\nsource_revision = {s}\nsource_hash_text = {s}\n" ++
            "tokenizer = {s}\ntensor_inventory = tensors.tsv\n",
        .{ import_layout_version, bundle.family, bundle.source_model_id, bundle.source_revision, bundle.source_hash_text, bundle.tokenizer_name },
    );
    if (bundle.mode == .gguf) try appendText(allocator, &out, "gguf_inventory = gguf_tensors.tsv\n");
    try appendFmt(
        allocator,
        &out,
        "modality_inventory = {s}\nprojector_inventory = {s}\nmodel_features = {s}\nprojector_present = {s}\n",
        .{
            if (bundle.has_projector) "modalities.tsv" else "-",
            if (bundle.has_projector) "projector.mizu" else "-",
            if (bundle.has_projector) "multimodal,projector" else "none",
            if (bundle.has_projector) "true" else "false",
        },
    );
    if (bundle.has_projector) try appendFmt(
        allocator,
        &out,
        "projector_slot = image\nprojector_placeholder_count = 1\nprojector_input_dtype = u8\n" ++
            "projector_embedding_dtype = bf16\nprojector_revision = {s}\n",
        .{bundle.projector_revision},
    );
    return out.toOwnedSlice(allocator);
}

fn renderTensors(allocator: Allocator, bundle: Bundle) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try appendText(allocator, &out, "# tensor_name|tensor_role|dtype|layout_name|relative_path|shape|storage_type\n");
    const tensors = try allocator.dupe(Tensor, bundle.tensors);
    if (bundle.mode == .gguf) std.mem.sort(Tensor, tensors, {}, tensorSourceNameLessThan) else std.mem.sort(Tensor, tensors, {}, tensorNameLessThan);
    for (tensors) |tensor| {
        try appendFmt(allocator, &out, "{s}|{s}|{s}|{s}|{s}|", .{ tensor.name, tensor.role, tensor.dtype, tensor.layout, tensor.bundle_rel });
        try appendShape(allocator, &out, tensor.shape);
        try appendFmt(allocator, &out, "|{s}\n", .{tensor.storage_type});
    }
    return out.toOwnedSlice(allocator);
}

fn renderGgufTensors(allocator: Allocator, bundle: Bundle) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try appendText(allocator, &out, "# tensor_name|source_kind|ggml_type|normalized_dtype|layout_name|relative_path|data_offset|source_offset|shape\n");
    const tensors = try allocator.dupe(Tensor, bundle.tensors);
    std.mem.sort(Tensor, tensors, {}, tensorSourceNameLessThan);
    for (tensors) |tensor| {
        try appendFmt(allocator, &out, "{s}|{s}|{s}|{s}|{s}|{s}|{d}|{d}|", .{ tensor.name, tensor.source_kind, tensor.storage_type, tensor.dtype, tensor.layout, tensor.bundle_rel, tensor.data_offset, tensor.source_offset });
        try appendShape(allocator, &out, tensor.shape);
        try appendText(allocator, &out, "\n");
    }
    return out.toOwnedSlice(allocator);
}

fn renderModalities(bundle: Bundle) []const u8 {
    return if (bundle.has_projector)
        "# placeholder_ordinal|slot_name|modality_kind|storage_kind|dtype\n1|image|image|encoded_bytes|u8\n"
    else
        "# no multimodal projector detected\n";
}

fn renderProjector(allocator: Allocator, bundle: Bundle) ![]const u8 {
    if (!bundle.has_projector) return "present = false\n";
    return std.fmt.allocPrint(
        allocator,
        "present = true\nslot = image\nplaceholder_count = 1\ninput_modality_kind = image\n" ++
            "input_dtype = u8\nembedding_dtype = bf16\nrevision_identity = {s}\n" ++
            "artifact_path = projector/projector_assets.mizu\n",
        .{bundle.projector_revision},
    );
}

fn renderProjectorAssets(allocator: Allocator, bundle: Bundle) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try appendText(allocator, &out, "# projector tensor inventory\n");
    for (bundle.tensors) |tensor| {
        if (!isProjectorSideRole(tensor.role)) continue;
        if (bundle.mode == .gguf) {
            try appendFmt(allocator, &out, "{s}|{s}|offset={d}|ggml_type={s}|source_offset={d}\n", .{ tensor.name, tensor.bundle_rel, tensor.data_offset, tensor.storage_type, tensor.source_offset });
        } else {
            try appendFmt(allocator, &out, "{s}|{s}\n", .{ tensor.name, tensor.bundle_rel });
        }
    }
    if (out.items.len == "# projector tensor inventory\n".len) {
        try appendText(allocator, &out, if (bundle.mode == .gguf)
            "# no projector-like tensors were detected\n"
        else
            "# no projector-like tensors were detected; config requested projector presence\n");
    }
    return out.toOwnedSlice(allocator);
}

fn appendShape(allocator: Allocator, out: *std.ArrayList(u8), shape: []const u64) !void {
    for (shape, 0..) |dimension, index| {
        if (index > 0) try appendText(allocator, out, "x");
        try appendFmt(allocator, out, "{d}", .{dimension});
    }
}

fn appendText(allocator: Allocator, out: *std.ArrayList(u8), text: []const u8) !void {
    try out.appendSlice(allocator, text);
}

fn appendFmt(allocator: Allocator, out: *std.ArrayList(u8), comptime format: []const u8, args: anytype) !void {
    try out.appendSlice(allocator, try std.fmt.allocPrint(allocator, format, args));
}

fn tensorSourceNameLessThan(_: void, left: Tensor, right: Tensor) bool {
    if (!std.mem.eql(u8, left.source_kind, right.source_kind)) return std.mem.lessThan(u8, left.source_kind, right.source_kind);
    return std.mem.lessThan(u8, left.name, right.name);
}

fn resolveGgufFamily(allocator: Allocator, requested: []const u8, model: SourceFile, io: Io) ![]const u8 {
    if (!std.mem.eql(u8, requested, "auto")) return requested;
    const identity = try std.mem.join(allocator, " ", &.{
        try metadataText(allocator, model.metadata, "general.architecture"),
        try metadataText(allocator, model.metadata, "general.name"),
        try metadataText(allocator, model.metadata, "general.basename"),
        model.name,
    });
    const lowered = try asciiLower(allocator, identity);
    if (std.mem.indexOf(u8, lowered, "qwen") != null) return "qwen3_5";
    if (std.mem.indexOf(u8, lowered, "gemma") != null) return "gemma4";
    try writeErr(io, "could not infer model family; pass --family qwen3_5 or --family gemma4\n", .{});
    return error.ReportedDiagnostic;
}

fn resolveGgufModelId(allocator: Allocator, override: []const u8, model: SourceFile, model_path: []const u8) ![]const u8 {
    if (override.len > 0) return override;
    for ([_][]const u8{ "general.name", "general.basename", "general.architecture" }) |key| {
        const value = try metadataText(allocator, model.metadata, key);
        if (std.mem.trim(u8, value, " \t\r\n").len > 0) return value;
    }
    return pathStem(model_path);
}

fn resolveGgufRevision(allocator: Allocator, override: []const u8, sources: []const SourceFile) ![]const u8 {
    if (override.len > 0) return override;
    var out: std.ArrayList(u8) = .empty;
    try appendFmt(allocator, &out, "gguf-v{d}", .{sources[0].version});
    const file_type = std.mem.trim(u8, try metadataText(allocator, sources[0].metadata, "general.file_type"), " \t\r\n");
    if (file_type.len > 0) try appendFmt(allocator, &out, ":filetype-{s}", .{file_type});
    const quant_version = std.mem.trim(u8, try metadataText(allocator, sources[0].metadata, "general.quantization_version"), " \t\r\n");
    if (quant_version.len > 0) try appendFmt(allocator, &out, ":quantv-{s}", .{quant_version});
    if (sources.len > 1) try appendFmt(allocator, &out, ":projector-gguf-v{d}", .{sources[1].version});
    return out.toOwnedSlice(allocator);
}

fn resolveGgufTokenizer(allocator: Allocator, model: SourceFile, family: []const u8) ![]const u8 {
    const tokenizer = std.mem.trim(u8, try metadataText(allocator, model.metadata, "tokenizer.ggml.model"), " \t\r\n");
    return if (tokenizer.len > 0) tokenizer else family;
}

fn metadataText(allocator: Allocator, metadata: std.StringHashMap(MetadataValue), key: []const u8) ![]const u8 {
    const value = metadata.get(key) orelse return "";
    return switch (value) {
        .string => |text| text,
        .signed => |number| std.fmt.allocPrint(allocator, "{d}", .{number}),
        .unsigned => |number| std.fmt.allocPrint(allocator, "{d}", .{number}),
        .float => |number| std.fmt.allocPrint(allocator, "{d}", .{number}),
        .boolean => |boolean| if (boolean) "True" else "False",
        .other => "",
    };
}

fn metadataInteger(metadata: std.StringHashMap(MetadataValue), key: []const u8, default: i64) i64 {
    const value = metadata.get(key) orelse return default;
    return switch (value) {
        .signed => |number| number,
        .unsigned => |number| if (number <= max_safe_i64) @intCast(number) else default,
        .boolean => |boolean| if (boolean) 1 else 0,
        .float => |number| if (number >= -9223372036854775808.0 and number < 9223372036854775808.0) @intFromFloat(number) else default,
        .string => |text| std.fmt.parseInt(i64, text, 10) catch default,
        .other => default,
    };
}

fn ggufSourceHash(allocator: Allocator, source_model_id: []const u8, source_revision: []const u8, sources: []const SourceFile, tensors: []Tensor) ![]const u8 {
    var digest = std.crypto.hash.sha2.Sha256.init(.{});
    digest.update(source_model_id);
    digest.update(&.{0});
    digest.update(source_revision);
    for (sources) |source| {
        digest.update("\x00file|");
        digest.update(source.kind);
        digest.update("|");
        digest.update(source.name);
        digest.update("|");
        const size = try std.fmt.allocPrint(allocator, "{d}", .{source.size});
        digest.update(size);
        digest.update("|");
        digest.update(try metadataText(allocator, source.metadata, "general.architecture"));
        digest.update("|");
        digest.update(try metadataText(allocator, source.metadata, "general.type"));
    }
    const ordered = try allocator.dupe(Tensor, tensors);
    std.mem.sort(Tensor, ordered, {}, tensorSourceNameLessThan);
    for (ordered) |tensor| {
        digest.update("\x00tensor|");
        digest.update(tensor.source_kind);
        digest.update("|");
        digest.update(tensor.name);
        digest.update("|");
        digest.update(tensor.storage_type);
        digest.update("|");
        try hashShape(allocator, &digest, tensor.shape);
        digest.update("|");
        digest.update(try std.fmt.allocPrint(allocator, "{d}", .{tensor.data_offset}));
        digest.update("|");
        digest.update(try std.fmt.allocPrint(allocator, "{d}", .{tensor.source_offset}));
    }
    var output: [32]u8 = undefined;
    digest.final(&output);
    return hexDigest(allocator, output);
}

fn pathStem(path: []const u8) []const u8 {
    const name = std.fs.path.basename(path);
    if (std.fs.path.extension(name).len == 0) return name;
    return name[0 .. name.len - std.fs.path.extension(name).len];
}

fn ggmlTypeName(type_id: u32) ?[]const u8 {
    return switch (type_id) {
        0 => "f32",
        1 => "f16",
        2 => "q4_0",
        3 => "q4_1",
        6 => "q5_0",
        7 => "q5_1",
        8 => "q8_0",
        9 => "q8_1",
        10 => "q2_k",
        11 => "q3_k",
        12 => "q4_k",
        13 => "q5_k",
        14 => "q6_k",
        15 => "q8_k",
        16 => "iq2_xxs",
        17 => "iq2_xs",
        18 => "iq3_xxs",
        19 => "iq1_s",
        20 => "iq4_nl",
        21 => "iq3_s",
        22 => "iq2_s",
        23 => "iq4_xs",
        24 => "i8",
        25 => "i16",
        26 => "i32",
        27 => "i64",
        28 => "f64",
        29 => "iq1_m",
        30 => "bf16",
        31 => "q4_0_4_4",
        32 => "q4_0_4_8",
        33 => "q4_0_8_8",
        34 => "tq1_0",
        35 => "tq2_0",
        else => null,
    };
}

const QuantSize = struct { elements: u64, bytes: u64 };

fn ggmlQuantSize(name: []const u8) ?QuantSize {
    if (std.mem.eql(u8, name, "f32")) return .{ .elements = 1, .bytes = 4 };
    if (std.mem.eql(u8, name, "f16") or std.mem.eql(u8, name, "bf16") or std.mem.eql(u8, name, "i16")) return .{ .elements = 1, .bytes = 2 };
    if (std.mem.eql(u8, name, "i8")) return .{ .elements = 1, .bytes = 1 };
    if (std.mem.eql(u8, name, "i32")) return .{ .elements = 1, .bytes = 4 };
    if (std.mem.eql(u8, name, "i64") or std.mem.eql(u8, name, "f64")) return .{ .elements = 1, .bytes = 8 };
    if (containsAny(name, &.{ "q4_0", "iq4_nl" })) return .{ .elements = 32, .bytes = 18 };
    if (std.mem.eql(u8, name, "q4_1")) return .{ .elements = 32, .bytes = 20 };
    if (std.mem.eql(u8, name, "q5_0")) return .{ .elements = 32, .bytes = 22 };
    if (std.mem.eql(u8, name, "q5_1")) return .{ .elements = 32, .bytes = 24 };
    if (std.mem.eql(u8, name, "q8_0")) return .{ .elements = 32, .bytes = 34 };
    if (std.mem.eql(u8, name, "q8_1")) return .{ .elements = 32, .bytes = 36 };
    if (std.mem.eql(u8, name, "q2_k")) return .{ .elements = 256, .bytes = 84 };
    if (std.mem.eql(u8, name, "q3_k") or std.mem.eql(u8, name, "iq3_s")) return .{ .elements = 256, .bytes = 110 };
    if (std.mem.eql(u8, name, "q4_k")) return .{ .elements = 256, .bytes = 144 };
    if (std.mem.eql(u8, name, "q5_k")) return .{ .elements = 256, .bytes = 176 };
    if (std.mem.eql(u8, name, "q6_k")) return .{ .elements = 256, .bytes = 210 };
    if (std.mem.eql(u8, name, "q8_k")) return .{ .elements = 256, .bytes = 292 };
    if (std.mem.eql(u8, name, "iq3_xxs")) return .{ .elements = 256, .bytes = 98 };
    if (std.mem.eql(u8, name, "iq2_xxs") or std.mem.eql(u8, name, "tq2_0")) return .{ .elements = 256, .bytes = 66 };
    if (std.mem.eql(u8, name, "iq2_xs")) return .{ .elements = 256, .bytes = 74 };
    if (std.mem.eql(u8, name, "iq1_s")) return .{ .elements = 256, .bytes = 50 };
    if (std.mem.eql(u8, name, "iq2_s")) return .{ .elements = 256, .bytes = 82 };
    if (std.mem.eql(u8, name, "iq4_xs")) return .{ .elements = 256, .bytes = 136 };
    if (std.mem.eql(u8, name, "iq1_m")) return .{ .elements = 256, .bytes = 56 };
    if (std.mem.eql(u8, name, "tq1_0")) return .{ .elements = 256, .bytes = 54 };
    if (std.mem.eql(u8, name, "q4_0_4_4") or std.mem.eql(u8, name, "q4_0_4_8") or std.mem.eql(u8, name, "q4_0_8_8")) return .{ .elements = 32, .bytes = 18 };
    return null;
}

fn normalizeGgmlDtype(name: []const u8) []const u8 {
    if (std.mem.eql(u8, name, "f32") or std.mem.eql(u8, name, "f16") or std.mem.eql(u8, name, "bf16") or std.mem.eql(u8, name, "i32")) return name;
    if (std.mem.eql(u8, name, "i8")) return "u8";
    if (std.mem.eql(u8, name, "i16") or std.mem.eql(u8, name, "i64") or std.mem.eql(u8, name, "f64")) return "f32";
    return "f16";
}

fn ggmlTensorByteCount(io: Io, path: []const u8, name: []const u8, shape: []const u64, ggml_type: []const u8) !u64 {
    const quant = ggmlQuantSize(ggml_type) orelse {
        try writeErr(io, "tensor {s} in {s} has unsupported GGML type {s}\n", .{ name, path, ggml_type });
        return error.ReportedDiagnostic;
    };
    if (shape.len == 0 or shape[0] % quant.elements != 0) {
        try writeErr(io, "tensor {s} in {s} has a shape incompatible with GGML type {s}\n", .{ name, path, ggml_type });
        return error.ReportedDiagnostic;
    }
    var rows: u64 = 1;
    for (shape[1..]) |dimension| {
        if (rows > max_safe_i64 / dimension) {
            try writeErr(io, "tensor {s} in {s} has unreasonable row count\n", .{ name, path });
            return error.ReportedDiagnostic;
        }
        rows *= dimension;
    }
    const row_bytes = (shape[0] / quant.elements) * quant.bytes;
    if (rows > max_safe_i64 / row_bytes) {
        try writeErr(io, "tensor {s} in {s} has unreasonable byte size\n", .{ name, path });
        return error.ReportedDiagnostic;
    }
    return rows * row_bytes;
}

fn classifyGgufTensor(allocator: Allocator, name: []const u8, source_kind: []const u8, general_type: []const u8) ![]const u8 {
    const lowered = try asciiLower(allocator, name);
    const is_projector = std.mem.eql(u8, source_kind, "projector") or std.mem.eql(u8, general_type, "mmproj");
    if (is_projector) {
        if (isGgufProjectorName(lowered)) return "multimodal_projector";
        if (isGgufVisionName(lowered, true)) return "vision_encoder";
        return "multimodal_projector";
    }
    if (isGgufProjectorName(lowered)) return "multimodal_projector";
    if (isGgufVisionName(lowered, false)) return "vision_encoder";
    if (containsAny(lowered, &.{ "token_embd", "embed_tokens", "embedding" })) return "embedding_table";
    if (std.mem.eql(u8, lowered, "output.weight") or containsAny(lowered, &.{ "lm_head", "output_projection" })) return "token_projection";
    if (std.mem.indexOf(u8, lowered, "norm") != null) return "normalization";
    if (containsAny(lowered, &.{ "blk.", ".blk.", "decoder", "attn_", "ffn_", "ssm_" })) return "decoder_stack";
    return "model_tensor";
}

fn isGgufProjectorName(lowered: []const u8) bool {
    return std.mem.startsWith(u8, lowered, "mm.") or containsAny(lowered, &.{ "projector", "merger" });
}

fn isGgufVisionName(lowered: []const u8, broad: bool) bool {
    if (std.mem.startsWith(u8, lowered, "v.") or std.mem.indexOf(u8, lowered, "vision") != null) return true;
    return broad and containsAny(lowered, &.{ "patch", "position" });
}

fn alignOffset(offset: u64, alignment: u64) u64 {
    if (alignment <= 1) return offset;
    const remainder = offset % alignment;
    if (remainder == 0) return offset;
    return offset + (alignment - remainder);
}

fn printSummary(io: Io, bundle: Bundle) !void {
    try writeOut(io, "family: {s}\nsource_model_id: {s}\nsource_revision: {s}\ntensor_count: {d}\n", .{ bundle.family, bundle.source_model_id, bundle.source_revision, bundle.tensors.len });
    if (bundle.mode == .gguf) {
        try writeOut(io, "gguf_file_count: {d}\n", .{bundle.sources.len});
    } else {
        try writeOut(io, "shard_count: {d}\n", .{bundle.sources.len});
    }
    try writeOut(io, "projector_present: {s}\noutput_root: {s}\n", .{ if (bundle.has_projector) "true" else "false", bundle.output_root });
}

fn writeOut(io: Io, comptime format: []const u8, args: anytype) !void {
    var buffer: [4096]u8 = undefined;
    var writer = Io.File.stdout().writerStreaming(io, &buffer);
    try writer.interface.print(format, args);
    try writer.interface.flush();
}

fn writeErr(io: Io, comptime format: []const u8, args: anytype) !void {
    var buffer: [4096]u8 = undefined;
    var writer = Io.File.stderr().writerStreaming(io, &buffer);
    try writer.interface.print(format, args);
    try writer.interface.flush();
}

test "GGML block sizes cover supported legacy and quantized formats" {
    const testing = std.testing;
    try testing.expectEqual(QuantSize{ .elements = 256, .bytes = 98 }, ggmlQuantSize("iq3_xxs").?);
    try testing.expectEqual(QuantSize{ .elements = 256, .bytes = 66 }, ggmlQuantSize("tq2_0").?);
    try testing.expectEqual(QuantSize{ .elements = 256, .bytes = 110 }, ggmlQuantSize("q3_k").?);
    try testing.expectEqual(QuantSize{ .elements = 32, .bytes = 18 }, ggmlQuantSize("q4_0_4_8").?);
    try testing.expect(ggmlQuantSize("unsupported") == null);
}

test "safetensors and GGUF role classification keeps format-specific precedence" {
    const testing = std.testing;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    try testing.expectEqualStrings("multimodal_projector", try classifySafetensorsTensor(allocator, "visual.merger.weight"));
    try testing.expectEqualStrings("vision_encoder", try classifySafetensorsTensor(allocator, "vision_model.encoder.weight"));
    try testing.expectEqualStrings("embedding_table", try classifySafetensorsTensor(allocator, "model.embed_tokens.weight"));
    try testing.expectEqualStrings("vision_encoder", try classifyGgufTensor(allocator, "v.blk.0.attn.weight", "model", "model"));
    try testing.expectEqualStrings("multimodal_projector", try classifyGgufTensor(allocator, "mm.0.weight", "model", "model"));
    try testing.expectEqualStrings("vision_encoder", try classifyGgufTensor(allocator, "position_embedding.weight", "projector", "mmproj"));
}

test "input path checks use component boundaries and reject traversal" {
    const testing = std.testing;
    try testing.expect(pathIsWithin("/models/qwen", "/models/qwen/shard.safetensors"));
    try testing.expect(!pathIsWithin("/models/qwen", "/models/qwen-old/shard.safetensors"));
    try testing.expect(pathHasParentTraversal("nested/../outside.safetensors"));
    try testing.expect(!pathHasParentTraversal("nested/shard.safetensors"));
    try testing.expectEqual(@as(u64, 64), alignOffset(33, 32));
    try testing.expectEqual(@as(u64, 64), alignOffset(64, 32));
}

test "projector detection ignores key case" {
    const testing = std.testing;
    try testing.expect(containsIgnoreCase("Vision_Config", "vision"));
    try testing.expect(!containsIgnoreCase("decoder_config", "vision"));
}
