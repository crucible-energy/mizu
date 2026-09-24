const std = @import("std");
const Io = std.Io;
const Dir = Io.Dir;
const Allocator = std.mem.Allocator;
const EnvMap = std.process.Environ.Map;

const EnvPair = struct { key: []const u8, value: []const u8 };

const Context = struct {
    allocator: Allocator,
    io: Io,
    source_env: *EnvMap,
    repo_root: []const u8,
    fixture_root: []const u8,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(init.gpa);
    defer init.gpa.free(args);
    if (args.len != 2) return error.ExpectedFixtureRoot;
    const cwd = try std.process.currentPathAlloc(init.io, allocator);
    const context = Context{
        .allocator = allocator,
        .io = init.io,
        .source_env = init.environ_map,
        .repo_root = cwd,
        .fixture_root = args[1],
    };
    try testFormatter(context);
    try testPreCommit(context);
    try testPrePushCheck(context);
    try testPrePushHook(context);
    try writeOut(init.io, "test_devtools: PASS\n", .{});
}

fn testFormatter(context: Context) !void {
    const repo = try join(context.allocator, context.fixture_root, "formatter-repo");
    try initRepo(context, repo, false, false);
    const formatter = try repoTool(context, "scripts/format-local.sh");
    const restage_invalid = try runRaw(context, &.{ "bash", formatter, "--all", "--write", "--restage" }, repo, &.{});
    try expectExit(restage_invalid.term, 2);
    try expectContains(restage_invalid.stderr, "--restage requires --staged");

    const script = try join(context.allocator, repo, "script.sh");
    try writeBytes(context.io, script, "#!/usr/bin/env bash\r\necho hi   \r\n");
    try setPermissions(context.io, script, 0o755);
    const markdown = try join(context.allocator, repo, "notes.md");
    try writeText(context.io, markdown, "line with hard break  \nnext line\n");
    const attributes = try join(context.allocator, repo, ".gitattributes");
    try writeText(context.io, attributes, "*.md whitespace=-trailing-space\n");
    _ = try run(context, &.{ "git", "add", "script.sh", "notes.md", ".gitattributes" }, repo, &.{});
    _ = try run(context, &.{ "bash", formatter, "--all", "--write" }, repo, &.{});
    _ = try run(context, &.{ "git", "add", "script.sh", "notes.md", ".gitattributes" }, repo, &.{});
    _ = try run(context, &.{ "bash", formatter, "--all", "--check" }, repo, &.{});
    _ = try run(context, &.{ "git", "diff", "--cached", "--check" }, repo, &.{});
    try expectEqual("normalized shell script", try readFile(context.allocator, context.io, script), "#!/usr/bin/env bash\necho hi\n");
    try expectEqualInt("executable mode preserved", try permissionBits(context.io, script), 0o755);
    try expectEqual("markdown hard break preserved", try readFile(context.allocator, context.io, markdown), "line with hard break  \nnext line\n");
    _ = try run(context, &.{ "git", "commit", "-qm", "baseline" }, repo, &.{});

    const partial = try join(context.allocator, repo, "staged_partial.txt");
    try writeText(context.io, partial, "alpha\nbeta\ngamma\n");
    _ = try run(context, &.{ "git", "add", "staged_partial.txt" }, repo, &.{});
    _ = try run(context, &.{ "git", "commit", "-qm", "add partial fixture" }, repo, &.{});
    try writeText(context.io, partial, "alpha staged\nbeta\ngamma\n");
    _ = try run(context, &.{ "git", "add", "staged_partial.txt" }, repo, &.{});
    try writeText(context.io, partial, "alpha staged\nbeta unstaged\ngamma\n");
    _ = try run(context, &.{ "bash", formatter, "--staged", "--write", "--restage" }, repo, &.{});
    try expectEqual("partial staged blob isolated from worktree edits", try gitShow(context, repo, ":staged_partial.txt"), "alpha staged\nbeta\ngamma\n");
    try expectEqual("partial unstaged worktree preserved", try readFile(context.allocator, context.io, partial), "alpha staged\nbeta unstaged\ngamma\n");

    const staged_newline = try join(context.allocator, repo, "staged_newline.txt");
    try writeText(context.io, staged_newline, "baseline\n");
    _ = try run(context, &.{ "git", "add", "staged_newline.txt" }, repo, &.{});
    _ = try run(context, &.{ "git", "commit", "-qm", "add newline fixture" }, repo, &.{});
    try writeBytes(context.io, staged_newline, "needs newline");
    _ = try run(context, &.{ "git", "add", "staged_newline.txt" }, repo, &.{});
    try writeText(context.io, staged_newline, "needs newline\n");
    _ = try run(context, &.{ "bash", formatter, "--staged", "--write", "--restage" }, repo, &.{});
    _ = try run(context, &.{ "git", "diff", "--cached", "--check" }, repo, &.{});
    try expectEqual("staged newline normalized", try gitShow(context, repo, ":staged_newline.txt"), "needs newline\n");
    try expectEqual("clean worktree left intact", try readFile(context.allocator, context.io, staged_newline), "needs newline\n");

    const staged_edit = try join(context.allocator, repo, "staged_no_restage.txt");
    try writeText(context.io, staged_edit, "baseline\n");
    _ = try run(context, &.{ "git", "add", "staged_no_restage.txt" }, repo, &.{});
    _ = try run(context, &.{ "git", "commit", "-qm", "add no-restage fixture" }, repo, &.{});
    try writeBytes(context.io, staged_edit, "needs newline");
    _ = try run(context, &.{ "git", "add", "staged_no_restage.txt" }, repo, &.{});
    try writeText(context.io, staged_edit, "local edit stays\n");
    _ = try run(context, &.{ "bash", formatter, "--staged", "--write" }, repo, &.{});
    try expectEqual("staged blob normalized without restage", try gitShow(context, repo, ":staged_no_restage.txt"), "needs newline\n");
    try expectEqual("no-restage local edit preserved", try readFile(context.allocator, context.io, staged_edit), "local edit stays\n");

    const deleted = try join(context.allocator, repo, "deleted_after_stage.txt");
    try writeText(context.io, deleted, "baseline\n");
    _ = try run(context, &.{ "git", "add", "deleted_after_stage.txt" }, repo, &.{});
    _ = try run(context, &.{ "git", "commit", "-qm", "add delete fixture" }, repo, &.{});
    try writeBytes(context.io, deleted, "needs newline");
    _ = try run(context, &.{ "git", "add", "deleted_after_stage.txt" }, repo, &.{});
    try Dir.cwd().deleteFile(context.io, deleted);
    _ = try run(context, &.{ "bash", formatter, "--staged", "--write", "--restage" }, repo, &.{});
    try expectEqual("deleted path staged blob normalized", try gitShow(context, repo, ":deleted_after_stage.txt"), "needs newline\n");
    try expectFileAbsent(context.io, deleted);
}

fn testPreCommit(context: Context) !void {
    const repo = try join(context.allocator, context.fixture_root, "pre-commit-repo");
    try initRepo(context, repo, true, false);
    _ = try run(context, &.{ "bash", "scripts/install-local-hooks.sh" }, repo, &.{});
    try expectEqual("core.hooksPath", try gitConfig(context, repo, "core.hooksPath"), ".githooks");
    try expect(try permissionBits(context.io, try join(context.allocator, repo, ".githooks/pre-commit")) & 0o100 != 0, "pre-commit hook is executable");

    const staged_only = try join(context.allocator, repo, "staged_only.txt");
    try writeBytes(context.io, staged_only, "needs newline   \r\n");
    _ = try run(context, &.{ "git", "add", "staged_only.txt" }, repo, &.{});
    _ = try run(context, &.{ "git", "commit", "-qm", "normalize staged-only file" }, repo, &.{});
    try expectEqual("staged-only committed blob", try gitShow(context, repo, "HEAD:staged_only.txt"), "needs newline\n");
    try expectEqual("staged-only worktree normalized", try readFile(context.allocator, context.io, staged_only), "needs newline\n");

    const partial = try join(context.allocator, repo, "partial.txt");
    try writeText(context.io, partial, "alpha\nbeta\ngamma\n");
    _ = try run(context, &.{ "git", "add", "partial.txt" }, repo, &.{});
    _ = try run(context, &.{ "git", "commit", "-qm", "add partial fixture" }, repo, &.{});
    try writeText(context.io, partial, "alpha staged   \nbeta\ngamma");
    _ = try run(context, &.{ "git", "add", "partial.txt" }, repo, &.{});
    try writeText(context.io, partial, "alpha staged   \nbeta unstaged\ngamma\n");
    _ = try run(context, &.{ "git", "commit", "-qm", "commit partial fixture" }, repo, &.{});
    try expectEqual("partial staged commit normalized", try gitShow(context, repo, "HEAD:partial.txt"), "alpha staged\nbeta\ngamma\n");
    try expectEqual("partial worktree preserved", try readFile(context.allocator, context.io, partial), "alpha staged   \nbeta unstaged\ngamma\n");
}

fn testPrePushCheck(context: Context) !void {
    const tools_repo = try join(context.allocator, context.fixture_root, "pre-push-check-repo");
    try initRepo(context, tools_repo, true, true);
    const docs_base = try gitRevParse(context, tools_repo, "HEAD");
    _ = try run(context, &.{ "git", "checkout", "-qb", "feat/docs-only" }, tools_repo, &.{});
    try writeText(context.io, try join(context.allocator, tools_repo, "README.md"), "Current docs change.\n");
    _ = try run(context, &.{ "git", "add", "README.md" }, tools_repo, &.{});
    _ = try run(context, &.{ "git", "commit", "-qm", "docs-only" }, tools_repo, &.{});
    const docs_head = try gitRevParse(context, tools_repo, "HEAD");
    const docs_log = try join(context.allocator, context.fixture_root, "docs-make.log");
    const docs_result = try runPrePushCheck(context, tools_repo, docs_log, try fmt(context.allocator, "refs/heads/feat/docs-only {s} refs/heads/feat/docs-only {s}\n", .{ docs_head, docs_base }), &.{});
    try expectExit(docs_result.term, 0);
    try expectEqual("docs-only make targets", try readFile(context.allocator, context.io, docs_log), "test\n");
    try expectNotContains(concat(context.allocator, docs_result.stdout, docs_result.stderr), "Escalating to make check-debug");

    const no_ref_log = try join(context.allocator, context.fixture_root, "no-ref-make.log");
    const no_ref_result = try runPrePushCheck(context, tools_repo, no_ref_log, "", &.{});
    try expectExit(no_ref_result.term, 0);
    try expectEqual("no-ref pre-push make targets", try readFile(context.allocator, context.io, no_ref_log), "test\n");
    try expectContains(concat(context.allocator, no_ref_result.stdout, no_ref_result.stderr), "Mizu pre-push gate passed on branch: feat/docs-only");
    try expectNotContains(concat(context.allocator, no_ref_result.stdout, no_ref_result.stderr), "unbound variable");

    const source_repo = try join(context.allocator, context.fixture_root, "pre-push-main-source-repo");
    try initRepo(context, source_repo, true, true);
    try writeText(context.io, try join(context.allocator, source_repo, "README.md"), "Current main-only change.\n");
    _ = try run(context, &.{ "git", "add", "README.md" }, source_repo, &.{});
    _ = try run(context, &.{ "git", "commit", "-qm", "main-only" }, source_repo, &.{});
    const source_head = try gitRevParse(context, source_repo, "HEAD");
    const zero_oid = "0000000000000000000000000000000000000000";
    const main_allow_log = try join(context.allocator, context.fixture_root, "main-allow-make.log");
    const main_line = try fmt(context.allocator, "refs/heads/main {s} refs/heads/main {s}\n", .{ source_head, zero_oid });
    const allow_main = [_]EnvPair{.{ .key = "MIZU_ALLOW_MAIN_PUSH", .value = "1" }};
    const main_result = try runPrePushCheck(context, source_repo, main_allow_log, main_line, &allow_main);
    try expectExit(main_result.term, 0);
    try expectEqual("allowed main push make targets", try readFile(context.allocator, context.io, main_allow_log), "test\n");
    try expectNotContains(concat(context.allocator, main_result.stdout, main_result.stderr), "Escalating to make check-debug");

    _ = try run(context, &.{ "git", "checkout", "-qb", "feat/source-main-guard" }, source_repo, &.{});
    const rejected_log = try join(context.allocator, context.fixture_root, "source-main-rejected.log");
    const source_line = try fmt(context.allocator, "refs/heads/main {s} refs/heads/feat/from-main {s}\n", .{ source_head, zero_oid });
    const source_result = try runPrePushCheck(context, source_repo, rejected_log, source_line, &.{});
    try expectExit(source_result.term, 2);
    try expectContains(concat(context.allocator, source_result.stdout, source_result.stderr), "Refusing push from main (refs/heads/main -> refs/heads/feat/from-main). Use a feature branch.");
    try expectFileAbsent(context.io, rejected_log);

    const source_allow_log = try join(context.allocator, context.fixture_root, "source-main-allow-make.log");
    const source_allow_result = try runPrePushCheck(context, source_repo, source_allow_log, source_line, &allow_main);
    try expectExit(source_allow_result.term, 0);
    try expectEqual("allowed source-main make targets", try readFile(context.allocator, context.io, source_allow_log), "test\n");
    try expectNotContains(concat(context.allocator, source_allow_result.stdout, source_allow_result.stderr), "Escalating to make check-debug");

    const runtime_repo = try join(context.allocator, context.fixture_root, "pre-push-runtime-repo");
    try initRepo(context, runtime_repo, true, true);
    _ = try run(context, &.{ "git", "checkout", "-qb", "feat/runtime" }, runtime_repo, &.{});
    _ = try run(context, &.{ "git", "branch", "--set-upstream-to=main" }, runtime_repo, &.{});
    try writeText(context.io, try join(context.allocator, runtime_repo, "src/runtime/touch.f90"), "program touch\nend program touch\n");
    _ = try run(context, &.{ "git", "add", "src/runtime/touch.f90" }, runtime_repo, &.{});
    _ = try run(context, &.{ "git", "commit", "-qm", "runtime-change" }, runtime_repo, &.{});
    const runtime_head = try gitRevParse(context, runtime_repo, "HEAD");
    const runtime_log = try join(context.allocator, context.fixture_root, "runtime-make.log");
    const runtime_line = try fmt(context.allocator, "refs/heads/feat/runtime {s} refs/heads/feat/runtime {s}\n", .{ runtime_head, zero_oid });
    const runtime_result = try runPrePushCheck(context, runtime_repo, runtime_log, runtime_line, &.{});
    try expectExit(runtime_result.term, 0);
    try expectEqual("runtime make targets", try readFile(context.allocator, context.io, runtime_log), "test\ncheck-debug\n");
    try expectContains(concat(context.allocator, runtime_result.stdout, runtime_result.stderr), "Escalating to make check-debug for sensitive path: src/runtime/touch.f90");
}

fn testPrePushHook(context: Context) !void {
    const repo = try join(context.allocator, context.fixture_root, "pre-push-hook-repo");
    try initRepo(context, repo, true, true);
    const remote = try join(context.allocator, context.fixture_root, "remote.git");
    try ensureDirectory(context.io, context.fixture_root);
    _ = try run(context, &.{ "git", "init", "--bare", "-q", remote }, context.fixture_root, &.{});
    _ = try run(context, &.{ "git", "remote", "add", "origin", remote }, repo, &.{});
    _ = try run(context, &.{ "git", "push", "-q", "origin", "main" }, repo, &.{});
    _ = try run(context, &.{ "bash", "scripts/install-local-hooks.sh" }, repo, &.{});
    try expectEqual("installed core.hooksPath", try gitConfig(context, repo, "core.hooksPath"), ".githooks");
    try expect(try permissionBits(context.io, try join(context.allocator, repo, ".githooks/pre-push")) & 0o100 != 0, "pre-push hook is executable");

    const fake_bin = try join(context.allocator, context.fixture_root, "fake-bin");
    try ensureDirectory(context.io, fake_bin);
    try writeFakeMake(context, fake_bin);

    try writeText(context.io, try join(context.allocator, repo, "README.md"), "Current main-only change.\n");
    _ = try run(context, &.{ "git", "add", "README.md" }, repo, &.{});
    _ = try run(context, &.{ "git", "commit", "-qm", "main-only" }, repo, &.{});
    const main_log = try join(context.allocator, context.fixture_root, "main-push-make.log");
    const main_result = try gitPush(context, repo, fake_bin, main_log, "main", &.{});
    try expectExit(main_result.term, 1);
    try expectContains(concat(context.allocator, main_result.stdout, main_result.stderr), "Refusing push to main (refs/heads/main -> refs/heads/main). Use a feature branch.");
    try expectFileAbsent(context.io, main_log);

    const allow_main = [_]EnvPair{.{ .key = "MIZU_ALLOW_MAIN_PUSH", .value = "1" }};
    const main_allow_log = try join(context.allocator, context.fixture_root, "main-allow-push-make.log");
    const main_allow_result = try gitPush(context, repo, fake_bin, main_allow_log, "main", &allow_main);
    try expectExit(main_allow_result.term, 0);
    try expectEqual("allowed main push targets", try readFile(context.allocator, context.io, main_allow_log), "test\n");
    try expectNotContains(concat(context.allocator, main_allow_result.stdout, main_allow_result.stderr), "Escalating to make check-debug");
    try expectContains(concat(context.allocator, main_allow_result.stdout, main_allow_result.stderr), "Mizu pre-push gate passed on branch: main");

    _ = try run(context, &.{ "git", "checkout", "-qb", "feat/source-main-guard" }, repo, &.{});
    const source_log = try join(context.allocator, context.fixture_root, "main-source-make.log");
    const source_result = try gitPush(context, repo, fake_bin, source_log, "main:feat/from-main", &.{});
    try expectExit(source_result.term, 1);
    try expectContains(concat(context.allocator, source_result.stdout, source_result.stderr), "Refusing push from main (refs/heads/main -> refs/heads/feat/from-main). Use a feature branch.");
    try expectFileAbsent(context.io, source_log);
    const source_allow_log = try join(context.allocator, context.fixture_root, "main-source-allow-make.log");
    const source_allow_result = try gitPush(context, repo, fake_bin, source_allow_log, "main:feat/from-main", &allow_main);
    try expectExit(source_allow_result.term, 0);
    try expectEqual("allowed main source targets", try readFile(context.allocator, context.io, source_allow_log), "test\n");
    try expectNotContains(concat(context.allocator, source_allow_result.stdout, source_allow_result.stderr), "Escalating to make check-debug");
    try expectContains(concat(context.allocator, source_allow_result.stdout, source_allow_result.stderr), "Mizu pre-push gate passed on branch: feat/source-main-guard");

    _ = try run(context, &.{ "git", "checkout", "-q", "main" }, repo, &.{});
    _ = try run(context, &.{ "git", "checkout", "-qb", "feat/docs-only" }, repo, &.{});
    _ = try run(context, &.{ "git", "branch", "--set-upstream-to=main" }, repo, &.{});
    try writeText(context.io, try join(context.allocator, repo, "README.md"), "Current docs change.\n");
    _ = try run(context, &.{ "git", "add", "README.md" }, repo, &.{});
    _ = try run(context, &.{ "git", "commit", "-qm", "docs-only" }, repo, &.{});
    const docs_log = try join(context.allocator, context.fixture_root, "docs-push-make.log");
    const docs_result = try gitPush(context, repo, fake_bin, docs_log, "feat/docs-only", &.{});
    try expectExit(docs_result.term, 0);
    try expectEqual("docs push targets", try readFile(context.allocator, context.io, docs_log), "test\n");
    try expectNotContains(concat(context.allocator, docs_result.stdout, docs_result.stderr), "Escalating to make check-debug");
    try expectContains(concat(context.allocator, docs_result.stdout, docs_result.stderr), "Mizu pre-push gate passed on branch: feat/docs-only");

    _ = try run(context, &.{ "git", "checkout", "-q", "main" }, repo, &.{});
    _ = try run(context, &.{ "git", "checkout", "-qb", "feat/runtime" }, repo, &.{});
    _ = try run(context, &.{ "git", "branch", "--set-upstream-to=main" }, repo, &.{});
    try writeText(context.io, try join(context.allocator, repo, "src/runtime/touch.f90"), "program touch\nend program touch\n");
    _ = try run(context, &.{ "git", "add", "src/runtime/touch.f90" }, repo, &.{});
    _ = try run(context, &.{ "git", "commit", "-qm", "runtime-change" }, repo, &.{});
    const runtime_log = try join(context.allocator, context.fixture_root, "runtime-push-make.log");
    const runtime_result = try gitPush(context, repo, fake_bin, runtime_log, "feat/runtime", &.{});
    try expectExit(runtime_result.term, 0);
    try expectEqual("runtime push targets", try readFile(context.allocator, context.io, runtime_log), "test\ncheck-debug\n");
    const runtime_output = concat(context.allocator, runtime_result.stdout, runtime_result.stderr);
    try expectContains(runtime_output, "Escalating to make check-debug for sensitive path: src/runtime/touch.f90");
    try expectContains(runtime_output, "Mizu pre-push gate passed on branch: feat/runtime");
}

const ProcessOutput = struct {
    term: std.process.Child.Term,
    stdout: []u8,
    stderr: []u8,
};

fn initRepo(context: Context, path: []const u8, copy_tools: bool, include_make: bool) !void {
    try ensureDirectory(context.io, path);
    _ = try run(context, &.{ "git", "init", "-q" }, path, &.{});
    _ = try run(context, &.{ "git", "config", "user.name", "Mizu Zig Test" }, path, &.{});
    _ = try run(context, &.{ "git", "config", "user.email", "mizu-zig-test@example.com" }, path, &.{});
    _ = try run(context, &.{ "git", "symbolic-ref", "HEAD", "refs/heads/main" }, path, &.{});
    if (copy_tools) try copyRepoTools(context, path);
    try writeText(context.io, try join(context.allocator, path, "README.md"), "Current baseline.\n");
    if (include_make) {
        try writeText(context.io, try join(context.allocator, path, "Makefile"), "test:\n\t@:\n\ncheck-debug:\n\t@:\n");
        try writeText(context.io, try join(context.allocator, path, "src/runtime/baseline.f90"), "program baseline\nend program baseline\n");
    }
    _ = try run(context, &.{ "git", "add", "." }, path, &.{});
    _ = try run(context, &.{ "git", "commit", "-qm", "baseline" }, path, &.{});
}

fn copyRepoTools(context: Context, destination: []const u8) !void {
    for ([_][]const u8{
        "scripts/format-local.sh",
        "scripts/install-local-hooks.sh",
        "scripts/mizu-pre-push-check.sh",
        ".githooks/pre-commit",
        ".githooks/pre-push",
    }) |relative| {
        try Dir.copyFileAbsolute(
            try join(context.allocator, context.repo_root, relative),
            try join(context.allocator, destination, relative),
            context.io,
            .{ .make_path = true },
        );
    }
}

fn repoTool(context: Context, relative: []const u8) ![]const u8 {
    return join(context.allocator, context.repo_root, relative);
}

fn run(context: Context, argv: []const []const u8, cwd: []const u8, overrides: []const EnvPair) !std.process.RunResult {
    const result = try runRaw(context, argv, cwd, overrides);
    if (exitCode(result.term) != 0) {
        try writeErr(context.io, "command failed: {s}\nstdout:\n{s}\nstderr:\n{s}\n", .{ argv[0], result.stdout, result.stderr });
        return error.CommandFailed;
    }
    return result;
}

fn runRaw(context: Context, argv: []const []const u8, cwd: []const u8, overrides: []const EnvPair) !std.process.RunResult {
    var environ = try buildEnv(context, overrides);
    defer environ.deinit();
    return std.process.run(context.allocator, context.io, .{
        .argv = argv,
        .cwd = .{ .path = cwd },
        .environ_map = &environ,
    });
}

fn runInput(context: Context, argv: []const []const u8, cwd: []const u8, input_path: []const u8, stdout_path: []const u8, stderr_path: []const u8, overrides: []const EnvPair) !ProcessOutput {
    var environ = try buildEnv(context, overrides);
    defer environ.deinit();
    var input = try Dir.openFileAbsolute(context.io, input_path, .{});
    var output: Io.File = undefined;
    var errors: Io.File = undefined;
    var input_open = true;
    var output_open = false;
    var errors_open = false;
    defer {
        if (errors_open) errors.close(context.io);
        if (output_open) output.close(context.io);
        if (input_open) input.close(context.io);
    }
    output = try Dir.createFileAbsolute(context.io, stdout_path, .{ .truncate = true });
    output_open = true;
    errors = try Dir.createFileAbsolute(context.io, stderr_path, .{ .truncate = true });
    errors_open = true;
    var child = try std.process.spawn(context.io, .{
        .argv = argv,
        .cwd = .{ .path = cwd },
        .environ_map = &environ,
        .stdin = .{ .file = input },
        .stdout = .{ .file = output },
        .stderr = .{ .file = errors },
    });
    defer child.kill(context.io);
    const term = try child.wait(context.io);
    input.close(context.io);
    input_open = false;
    output.close(context.io);
    output_open = false;
    errors.close(context.io);
    errors_open = false;
    return .{
        .term = term,
        .stdout = try readFile(context.allocator, context.io, stdout_path),
        .stderr = try readFile(context.allocator, context.io, stderr_path),
    };
}

fn runPrePushCheck(context: Context, repo: []const u8, log_path: []const u8, input: []const u8, extra_env: []const EnvPair) !ProcessOutput {
    const fake_bin = try join(context.allocator, context.fixture_root, "pre-push-check-fake-bin");
    try ensureDirectory(context.io, fake_bin);
    try writeFakeMake(context, fake_bin);
    const overrides = try makeOverrides(context, fake_bin, log_path, extra_env);
    const input_path = try fmt(context.allocator, "{s}.input", .{log_path});
    const stdout_path = try fmt(context.allocator, "{s}.stdout", .{log_path});
    const stderr_path = try fmt(context.allocator, "{s}.stderr", .{log_path});
    try writeText(context.io, input_path, input);
    return runInput(context, &.{ "bash", "scripts/mizu-pre-push-check.sh" }, repo, input_path, stdout_path, stderr_path, overrides);
}

fn gitPush(context: Context, repo: []const u8, fake_bin: []const u8, log_path: []const u8, refspec: []const u8, extra_env: []const EnvPair) !std.process.RunResult {
    const overrides = try makeOverrides(context, fake_bin, log_path, extra_env);
    return runRaw(context, &.{ "git", "push", "-q", "origin", refspec }, repo, overrides);
}

fn makeOverrides(context: Context, fake_bin: []const u8, log_path: []const u8, extra: []const EnvPair) ![]const EnvPair {
    const inherited_path = context.source_env.get("PATH") orelse "/usr/bin:/bin";
    const path = try fmt(context.allocator, "{s}:{s}", .{ fake_bin, inherited_path });
    const values = try context.allocator.alloc(EnvPair, 2 + extra.len);
    values[0] = .{ .key = "PATH", .value = path };
    values[1] = .{ .key = "MIZU_TEST_MAKE_LOG", .value = log_path };
    @memcpy(values[2..], extra);
    return values;
}

fn buildEnv(context: Context, overrides: []const EnvPair) !EnvMap {
    var environ = EnvMap.init(context.allocator);
    errdefer environ.deinit();
    for (context.source_env.keys(), context.source_env.values()) |key, value| {
        if (std.mem.startsWith(u8, key, "GIT_") or
            std.mem.eql(u8, key, "MIZU_ALLOW_MAIN_PUSH") or
            std.mem.eql(u8, key, "MIZU_TEST_MAKE_LOG")) continue;
        try environ.put(key, value);
    }
    for (overrides) |override| try environ.put(override.key, override.value);
    return environ;
}

fn writeFakeMake(context: Context, fake_bin: []const u8) !void {
    const path = try join(context.allocator, fake_bin, "make");
    try writeText(context.io, path, "#!/usr/bin/env bash\n" ++
        "set -euo pipefail\n" ++
        "printf '%s\\n' \"$1\" >> \"$MIZU_TEST_MAKE_LOG\"\n");
    try setPermissions(context.io, path, 0o755);
}

fn gitConfig(context: Context, cwd: []const u8, key: []const u8) ![]const u8 {
    const result = try run(context, &.{ "git", "config", "--local", "--get", key }, cwd, &.{});
    return std.mem.trim(u8, result.stdout, " \t\r\n");
}

fn gitRevParse(context: Context, cwd: []const u8, rev: []const u8) ![]const u8 {
    const result = try run(context, &.{ "git", "rev-parse", rev }, cwd, &.{});
    return std.mem.trim(u8, result.stdout, " \t\r\n");
}

fn gitShow(context: Context, cwd: []const u8, spec: []const u8) ![]const u8 {
    const result = try run(context, &.{ "git", "show", spec }, cwd, &.{});
    return result.stdout;
}

fn ensureDirectory(io: Io, path: []const u8) !void {
    var dir = try Dir.cwd().createDirPathOpen(io, path, .{});
    dir.close(io);
}

fn writeText(io: Io, path: []const u8, value: []const u8) !void {
    if (std.fs.path.dirname(path)) |parent| try ensureDirectory(io, parent);
    var file = try Dir.createFileAbsolute(io, path, .{ .truncate = true, .permissions = @enumFromInt(0o644) });
    defer file.close(io);
    try file.writeStreamingAll(io, value);
}

fn writeBytes(io: Io, path: []const u8, value: []const u8) !void {
    try writeText(io, path, value);
}

fn readFile(allocator: Allocator, io: Io, path: []const u8) ![]u8 {
    var file = try Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    var reader = file.readerStreaming(io, &.{});
    return reader.interface.readAlloc(allocator, @intCast(stat.size));
}

fn setPermissions(io: Io, path: []const u8, mode: u16) !void {
    var file = try Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    try file.setPermissions(io, @enumFromInt(mode));
}

fn permissionBits(io: Io, path: []const u8) !u16 {
    var file = try Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    return @truncate(@intFromEnum(stat.permissions) & 0o7777);
}

fn expectFileAbsent(io: Io, path: []const u8) !void {
    if (Dir.openFileAbsolute(io, path, .{})) |file| {
        file.close(io);
        return error.ExpectedFileAbsent;
    } else |_| {}
}

fn expectEqual(label: []const u8, actual: []const u8, expected: []const u8) !void {
    if (!std.mem.eql(u8, actual, expected)) {
        std.debug.print("{s}: expected {s}, got {s}\n", .{ label, expected, actual });
        return error.ValuesDiffer;
    }
}

fn expectEqualInt(label: []const u8, actual: u16, expected: u16) !void {
    if (actual != expected) {
        std.debug.print("{s}: expected {d}, got {d}\n", .{ label, expected, actual });
        return error.ValuesDiffer;
    }
}

fn expectContains(value: []const u8, expected: []const u8) !void {
    if (std.mem.indexOf(u8, value, expected) == null) return error.ExpectedTextMissing;
}

fn expectNotContains(value: []const u8, needle: []const u8) !void {
    if (std.mem.indexOf(u8, value, needle) != null) return error.UnexpectedTextFound;
}

fn expect(condition: bool, label: []const u8) !void {
    if (!condition) {
        std.debug.print("failed: {s}\n", .{label});
        return error.ExpectationFailed;
    }
}

fn expectExit(term: std.process.Child.Term, expected: u8) !void {
    const actual = exitCode(term);
    if (actual != expected) {
        std.debug.print("expected exit {d}, got {d}\n", .{ expected, actual });
        return error.UnexpectedExit;
    }
}

fn exitCode(term: std.process.Child.Term) u8 {
    return switch (term) {
        .exited => |code| code,
        else => 255,
    };
}

fn concat(allocator: Allocator, left: []const u8, right: []const u8) []const u8 {
    return std.mem.concat(allocator, u8, &.{ left, right }) catch unreachable;
}

fn fmt(allocator: Allocator, comptime format: []const u8, args: anytype) ![]const u8 {
    return std.fmt.allocPrint(allocator, format, args);
}

fn join(allocator: Allocator, root: []const u8, child: []const u8) ![]const u8 {
    return std.fs.path.join(allocator, &.{ root, child });
}

fn writeOut(io: Io, comptime format: []const u8, args: anytype) !void {
    var buffer: [4096]u8 = undefined;
    var writer = Io.File.stdout().writerStreaming(io, &buffer);
    try writer.interface.print(format, args);
    try writer.interface.flush();
}

fn writeErr(io: Io, comptime format: []const u8, args: anytype) !void {
    var buffer: [8192]u8 = undefined;
    var writer = Io.File.stderr().writerStreaming(io, &buffer);
    try writer.interface.print(format, args);
    try writer.interface.flush();
}
