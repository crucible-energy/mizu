#!/usr/bin/env python3
"""Regression-test installed repo-local pre-push hook behavior."""

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


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mizu_pre_push_hook_") as temp_root:
        temp_root_path = Path(temp_root)
        repo_root = init_repo(temp_root_path / "repo")
        remote_root = init_bare_remote(temp_root_path / "remote.git")

        run(["git", "remote", "add", "origin", str(remote_root)], cwd=repo_root)
        run(["git", "push", "-q", "origin", "main"], cwd=repo_root)

        run(["bash", "scripts/install-local-hooks.sh"], cwd=repo_root)
        expect_equal("core.hooksPath", git_config(repo_root, "core.hooksPath"), ".githooks")
        if not os.access(repo_root / ".githooks" / "pre-push", os.X_OK):
            raise AssertionError("pre-push hook should be executable after install")

        fake_make_dir = temp_root_path / "fake-bin"
        fake_make_dir.mkdir()
        fake_make_path = fake_make_dir / "make"
        fake_make_path.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "printf '%s\\n' \"$1\" >> \"$MIZU_TEST_MAKE_LOG\"\n",
            encoding="utf-8",
        )
        fake_make_path.chmod(0o755)

        run(["git", "checkout", "-qb", "feat/docs-only"], cwd=repo_root)
        run(["git", "branch", "--set-upstream-to=main"], cwd=repo_root)
        write_text(repo_root / "README.md", "Current docs change.\n")
        run(["git", "add", "README.md"], cwd=repo_root)
        run(["git", "commit", "-qm", "docs-only"], cwd=repo_root)
        docs_log = temp_root_path / "docs-make.log"
        docs_completed = git_push(repo_root, fake_make_dir, docs_log, "feat/docs-only")
        expect_equal("docs-only push return code", docs_completed.returncode, 0)
        expect_equal("docs-only make targets", docs_log.read_text(encoding="utf-8"), "test\n")
        docs_output = combined_output(docs_completed)
        if "Escalating to make check-debug" in docs_output:
            raise AssertionError(f"docs-only push should not escalate: {docs_output!r}")
        if "Mizu pre-push gate passed on branch: feat/docs-only" not in docs_output:
            raise AssertionError(f"missing success message for docs-only push: {docs_output!r}")

        run(["git", "checkout", "-q", "main"], cwd=repo_root)
        run(["git", "checkout", "-qb", "feat/runtime"], cwd=repo_root)
        run(["git", "branch", "--set-upstream-to=main"], cwd=repo_root)
        write_text(repo_root / "src" / "runtime" / "touch.f90", "program touch\nend program touch\n")
        run(["git", "add", "src/runtime/touch.f90"], cwd=repo_root)
        run(["git", "commit", "-qm", "runtime-change"], cwd=repo_root)
        runtime_log = temp_root_path / "runtime-make.log"
        runtime_completed = git_push(repo_root, fake_make_dir, runtime_log, "feat/runtime")
        expect_equal("runtime push return code", runtime_completed.returncode, 0)
        expect_equal("runtime make targets", runtime_log.read_text(encoding="utf-8"), "test\ncheck-debug\n")
        runtime_output = combined_output(runtime_completed)
        if "Escalating to make check-debug for sensitive path: src/runtime/touch.f90" not in runtime_output:
            raise AssertionError(f"missing debug escalation message: {runtime_output!r}")
        if "Mizu pre-push gate passed on branch: feat/runtime" not in runtime_output:
            raise AssertionError(f"missing success message for runtime push: {runtime_output!r}")

    print("test_pre_push_hook: PASS")
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


def init_bare_remote(remote_root: Path) -> Path:
    remote_root.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "init", "--bare", "-q", str(remote_root)], cwd=remote_root.parent)
    return remote_root


def git_push(
    repo_root: Path,
    fake_make_dir: Path,
    log_path: Path,
    branch: str,
) -> subprocess.CompletedProcess[str]:
    env = sandbox_env()
    env["PATH"] = f"{fake_make_dir}:{env['PATH']}"
    env["MIZU_TEST_MAKE_LOG"] = str(log_path)
    return subprocess.run(
        ["git", "push", "-q", "origin", branch],
        cwd=repo_root,
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


def combined_output(completed: subprocess.CompletedProcess[str]) -> str:
    return completed.stdout + completed.stderr


def sandbox_env() -> dict[str, str]:
    return {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}


if __name__ == "__main__":
    raise SystemExit(main())
