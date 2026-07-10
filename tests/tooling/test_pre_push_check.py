#!/usr/bin/env python3
"""Regression-test repo-local pre-push validation behavior."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_PUSH = REPO_ROOT / "scripts" / "mizu-pre-push-check.sh"
FORMATTER = REPO_ROOT / "scripts" / "format-local.sh"
INSTALL_HOOKS = REPO_ROOT / "scripts" / "install-local-hooks.sh"
PRE_COMMIT_HOOK = REPO_ROOT / ".githooks" / "pre-commit"
PRE_PUSH_HOOK = REPO_ROOT / ".githooks" / "pre-push"
ZERO_OID = "0" * 40


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mizu_pre_push_check_") as temp_root:
        temp_root_path = Path(temp_root)
        docs_repo = init_repo(temp_root_path / "docs")
        docs_base = git_rev_parse(docs_repo, "HEAD")
        run(["git", "checkout", "-qb", "feat/docs-only"], cwd=docs_repo)
        write_text(docs_repo / "README.md", "Current docs change.\n")
        run(["git", "add", "README.md"], cwd=docs_repo)
        run(["git", "commit", "-qm", "docs-only"], cwd=docs_repo)
        docs_head = git_rev_parse(docs_repo, "HEAD")
        docs_log = temp_root_path / "docs-make.log"
        docs_completed = run_pre_push(
            docs_repo,
            docs_log,
            f"refs/heads/feat/docs-only {docs_head} refs/heads/feat/docs-only {docs_base}\n",
        )
        expect_equal("docs-only push return code", docs_completed.returncode, 0)
        expect_equal("docs-only make targets", docs_log.read_text(encoding="utf-8"), "test\n")
        if "Escalating to make check-debug" in docs_completed.stdout:
            raise AssertionError("docs-only push should not escalate to check-debug")

        source_repo = init_repo(temp_root_path / "source-main")
        write_text(source_repo / "README.md", "Current main-only change.\n")
        run(["git", "add", "README.md"], cwd=source_repo)
        run(["git", "commit", "-qm", "main-only"], cwd=source_repo)
        source_head = git_rev_parse(source_repo, "HEAD")
        allow_main_env = {"MIZU_ALLOW_MAIN_PUSH": "1"}
        main_allow_log = temp_root_path / "main-allow-make.log"
        main_allow_completed = run_pre_push(
            source_repo,
            main_allow_log,
            f"refs/heads/main {source_head} refs/heads/main {ZERO_OID}\n",
            extra_env=allow_main_env,
        )
        expect_equal("allowed main push return code", main_allow_completed.returncode, 0)
        expect_equal("allowed main push make targets", main_allow_log.read_text(encoding="utf-8"), "test\n")
        if "Escalating to make check-debug" in combined_output(main_allow_completed):
            raise AssertionError(f"allowed main push should not escalate: {combined_output(main_allow_completed)!r}")

        run(["git", "checkout", "-qb", "feat/source-main-guard"], cwd=source_repo)
        source_log = temp_root_path / "source-main-make.log"
        source_completed = run_pre_push(
            source_repo,
            source_log,
            f"refs/heads/main {source_head} refs/heads/feat/from-main {ZERO_OID}\n",
        )
        expect_push_rejected(
            "source-main push",
            source_completed,
            source_log,
            "Refusing push from main (refs/heads/main -> refs/heads/feat/from-main). Use a feature branch.",
        )
        source_allow_log = temp_root_path / "source-main-allow-make.log"
        source_allow_completed = run_pre_push(
            source_repo,
            source_allow_log,
            f"refs/heads/main {source_head} refs/heads/feat/from-main {ZERO_OID}\n",
            extra_env=allow_main_env,
        )
        expect_equal("allowed source-main push return code", source_allow_completed.returncode, 0)
        expect_equal("allowed source-main make targets", source_allow_log.read_text(encoding="utf-8"), "test\n")
        if "Escalating to make check-debug" in combined_output(source_allow_completed):
            raise AssertionError(f"allowed source-main push should not escalate: {combined_output(source_allow_completed)!r}")

        runtime_repo = init_repo(temp_root_path / "runtime")
        run(["git", "checkout", "-qb", "feat/runtime"], cwd=runtime_repo)
        run(["git", "branch", "--set-upstream-to=main"], cwd=runtime_repo)
        write_text(runtime_repo / "src" / "runtime" / "touch.f90", "program touch\nend program touch\n")
        run(["git", "add", "src/runtime/touch.f90"], cwd=runtime_repo)
        run(["git", "commit", "-qm", "runtime-change"], cwd=runtime_repo)
        runtime_head = git_rev_parse(runtime_repo, "HEAD")
        runtime_log = temp_root_path / "runtime-make.log"
        runtime_completed = run_pre_push(
            runtime_repo,
            runtime_log,
            f"refs/heads/feat/runtime {runtime_head} refs/heads/feat/runtime {ZERO_OID}\n",
        )
        expect_equal("runtime push return code", runtime_completed.returncode, 0)
        expect_equal("runtime make targets", runtime_log.read_text(encoding="utf-8"), "test\ncheck-debug\n")
        if "Escalating to make check-debug for sensitive path: src/runtime/touch.f90" not in runtime_completed.stdout:
            raise AssertionError(f"missing debug escalation message: {runtime_completed.stdout!r}")

    print("test_pre_push_check: PASS")
    return 0


def init_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True)
    run(["git", "init", "-q"], cwd=repo_root)
    run(["git", "config", "user.name", "Mizu Test"], cwd=repo_root)
    run(["git", "config", "user.email", "mizu-test@example.com"], cwd=repo_root)
    run(["git", "checkout", "-qb", "main"], cwd=repo_root)

    copy_text(FORMATTER, repo_root / "scripts" / "format-local.sh")
    copy_text(INSTALL_HOOKS, repo_root / "scripts" / "install-local-hooks.sh")
    copy_text(PRE_PUSH, repo_root / "scripts" / "mizu-pre-push-check.sh")
    copy_text(PRE_COMMIT_HOOK, repo_root / ".githooks" / "pre-commit")
    copy_text(PRE_PUSH_HOOK, repo_root / ".githooks" / "pre-push")

    write_text(repo_root / "README.md", "Current baseline.\n")
    write_text(repo_root / "Makefile", "test:\n\t@:\n\ncheck-debug:\n\t@:\n")
    write_text(repo_root / "src" / "runtime" / "baseline.f90", "program baseline\nend program baseline\n")

    run(["git", "add", "."], cwd=repo_root)
    run(["git", "commit", "-qm", "baseline"], cwd=repo_root)
    return repo_root


def run_pre_push(
    repo_root: Path,
    log_path: Path,
    stdin_text: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    fake_make_dir = repo_root / ".fake-bin"
    fake_make_dir.mkdir(exist_ok=True)
    fake_make_path = fake_make_dir / "make"
    fake_make_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$1\" >> \"$MIZU_TEST_MAKE_LOG\"\n",
        encoding="utf-8",
    )
    fake_make_path.chmod(0o755)

    env = sandbox_env()
    env["PATH"] = f"{fake_make_dir}:{env['PATH']}"
    env["MIZU_TEST_MAKE_LOG"] = str(log_path)
    if extra_env is not None:
        env.update(extra_env)
    return subprocess.run(
        ["bash", "scripts/mizu-pre-push-check.sh"],
        cwd=repo_root,
        input=stdin_text,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def run(args: list[str], cwd: Path) -> None:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=sandbox_env(),
    )
    if completed.returncode == 0:
        return
    print(completed.stdout)
    print(completed.stderr, file=sys.stderr)
    raise SystemExit(completed.returncode)


def git_rev_parse(cwd: Path, rev: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", rev],
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=sandbox_env(),
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)
    return completed.stdout.strip()


def copy_text(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    destination.chmod(source.stat().st_mode)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def expect_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def expect_push_rejected(
    label: str,
    completed: subprocess.CompletedProcess[str],
    log_path: Path,
    message: str,
) -> None:
    if completed.returncode == 0:
        raise AssertionError(f"{label}: push should fail")
    if log_path.exists():
        raise AssertionError(f"{label}: hook should fail before invoking make: {log_path.read_text(encoding='utf-8')!r}")
    output = combined_output(completed)
    if message not in output:
        raise AssertionError(f"{label}: missing rejection message {message!r} in {output!r}")


def combined_output(completed: subprocess.CompletedProcess[str]) -> str:
    return completed.stdout + completed.stderr


def sandbox_env() -> dict[str, str]:
    return {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}


if __name__ == "__main__":
    raise SystemExit(main())
