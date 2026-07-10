#!/usr/bin/env python3
"""Regression-test repo-local pre-commit hook behavior."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMATTER = REPO_ROOT / "scripts" / "format-local.sh"
INSTALL_HOOKS = REPO_ROOT / "scripts" / "install-local-hooks.sh"
PRE_PUSH = REPO_ROOT / "scripts" / "mizu-pre-push-check.sh"
PRE_COMMIT_HOOK = REPO_ROOT / ".githooks" / "pre-commit"
PRE_PUSH_HOOK = REPO_ROOT / ".githooks" / "pre-push"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mizu_pre_commit_hook_") as temp_root:
        repo_root = init_repo(Path(temp_root) / "repo")

        run(["bash", "scripts/install-local-hooks.sh"], cwd=repo_root)
        expect_equal("core.hooksPath", git_config(repo_root, "core.hooksPath"), ".githooks")
        if not os.access(repo_root / ".githooks" / "pre-commit", os.X_OK):
            raise AssertionError("pre-commit hook should be executable after install")

        staged_only_path = repo_root / "staged_only.txt"
        staged_only_path.write_bytes(b"needs newline   \r\n")
        run(["git", "add", "staged_only.txt"], cwd=repo_root)
        run(["git", "commit", "-qm", "normalize staged-only file"], cwd=repo_root)
        expect_equal("staged-only committed blob", git_show_head(repo_root, "staged_only.txt"), "needs newline\n")
        expect_equal("staged-only worktree normalized", staged_only_path.read_text(encoding="utf-8"), "needs newline\n")

        partial_path = repo_root / "partial.txt"
        partial_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        run(["git", "add", "partial.txt"], cwd=repo_root)
        run(["git", "commit", "-qm", "add partial fixture"], cwd=repo_root)

        partial_path.write_text("alpha staged   \nbeta\ngamma", encoding="utf-8")
        run(["git", "add", "partial.txt"], cwd=repo_root)
        partial_path.write_text("alpha staged   \nbeta unstaged\ngamma\n", encoding="utf-8")

        run(["git", "commit", "-qm", "commit partial fixture"], cwd=repo_root)
        expect_equal("partial committed blob", git_show_head(repo_root, "partial.txt"), "alpha staged\nbeta\ngamma\n")
        expect_equal(
            "partial worktree preserved",
            partial_path.read_text(encoding="utf-8"),
            "alpha staged   \nbeta unstaged\ngamma\n",
        )

    print("test_pre_commit_hook: PASS")
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

    run(["git", "add", "."], cwd=repo_root)
    run(["git", "commit", "-qm", "baseline"], cwd=repo_root)
    return repo_root


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


def git_config(cwd: Path, key: str) -> str:
    completed = subprocess.run(
        ["git", "config", "--local", "--get", key],
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


def git_show_head(cwd: Path, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
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
    return completed.stdout


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


def sandbox_env() -> dict[str, str]:
    return {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}


if __name__ == "__main__":
    raise SystemExit(main())
