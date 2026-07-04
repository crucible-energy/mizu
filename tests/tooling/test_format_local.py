#!/usr/bin/env python3
"""Regression-test repo-local formatter behavior."""

from __future__ import annotations

import stat
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMATTER = REPO_ROOT / "scripts" / "format-local.sh"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mizu_format_local_") as temp_root:
        repo_root = Path(temp_root) / "repo"
        repo_root.mkdir()

        run(["git", "init", "-q"], cwd=repo_root)
        run(["git", "config", "user.name", "Mizu Test"], cwd=repo_root)
        run(["git", "config", "user.email", "mizu-test@example.com"], cwd=repo_root)

        completed = run_completed(["bash", str(FORMATTER), "--all", "--write", "--restage"], cwd=repo_root)
        expect_equal("restage without staged should fail", completed.returncode, 2)
        if "--restage requires --staged" not in completed.stderr:
            raise AssertionError(f"missing expected stderr: {completed.stderr!r}")

        script_path = repo_root / "script.sh"
        script_path.write_bytes(b"#!/usr/bin/env bash\r\necho hi   \r\n")
        script_path.chmod(0o755)
        markdown_path = repo_root / "notes.md"
        markdown_path.write_text("line with hard break  \nnext line\n", encoding="utf-8")
        gitattributes_path = repo_root / ".gitattributes"
        gitattributes_path.write_text("*.md whitespace=-trailing-space\n", encoding="utf-8")
        run(["git", "add", "script.sh"], cwd=repo_root)
        run(["git", "add", "notes.md", ".gitattributes"], cwd=repo_root)

        run(["bash", str(FORMATTER), "--all", "--write"], cwd=repo_root)
        run(["git", "add", "script.sh", "notes.md", ".gitattributes"], cwd=repo_root)
        run(["bash", str(FORMATTER), "--all", "--check"], cwd=repo_root)
        run(["git", "diff", "--cached", "--check"], cwd=repo_root)

        expect_equal(
            "normalized shell script contents",
            script_path.read_text(encoding="utf-8"),
            "#!/usr/bin/env bash\necho hi\n",
        )
        expect_equal(
            "executable mode preserved",
            stat.S_IMODE(script_path.stat().st_mode),
            0o755,
        )
        expect_equal(
            "markdown hard break preserved",
            markdown_path.read_text(encoding="utf-8"),
            "line with hard break  \nnext line\n",
        )

        run(["git", "commit", "-qm", "baseline"], cwd=repo_root)
        run(["bash", str(FORMATTER), "--staged", "--write", "--restage"], cwd=repo_root)

        staged_partial_path = repo_root / "staged_partial.txt"
        staged_partial_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        run(["git", "add", "staged_partial.txt"], cwd=repo_root)
        run(["git", "commit", "-qm", "add partial fixture"], cwd=repo_root)

        staged_partial_path.write_text("alpha staged\nbeta\ngamma\n", encoding="utf-8")
        run(["git", "add", "staged_partial.txt"], cwd=repo_root)
        staged_partial_path.write_text("alpha staged\nbeta unstaged\ngamma\n", encoding="utf-8")

        run(["bash", str(FORMATTER), "--staged", "--write", "--restage"], cwd=repo_root)
        expect_equal(
            "partial staged blob preserved without unstaged bleed",
            git_show(repo_root, "staged_partial.txt"),
            "alpha staged\nbeta\ngamma\n",
        )
        expect_equal(
            "partial staged worktree preserved",
            staged_partial_path.read_text(encoding="utf-8"),
            "alpha staged\nbeta unstaged\ngamma\n",
        )

        staged_newline_path = repo_root / "staged_newline.txt"
        staged_newline_path.write_text("baseline\n", encoding="utf-8")
        run(["git", "add", "staged_newline.txt"], cwd=repo_root)
        run(["git", "commit", "-qm", "add newline fixture"], cwd=repo_root)

        staged_newline_path.write_bytes(b"needs newline")
        run(["git", "add", "staged_newline.txt"], cwd=repo_root)
        staged_newline_path.write_text("needs newline\n", encoding="utf-8")

        run(["bash", str(FORMATTER), "--staged", "--write", "--restage"], cwd=repo_root)
        run(["git", "diff", "--cached", "--check"], cwd=repo_root)
        expect_equal(
            "staged blob normalized even when worktree is already clean",
            git_show(repo_root, "staged_newline.txt"),
            "needs newline\n",
        )
        expect_equal(
            "clean worktree left intact",
            staged_newline_path.read_text(encoding="utf-8"),
            "needs newline\n",
        )

        staged_edit_path = repo_root / "staged_no_restage.txt"
        staged_edit_path.write_text("baseline\n", encoding="utf-8")
        run(["git", "add", "staged_no_restage.txt"], cwd=repo_root)
        run(["git", "commit", "-qm", "add no-restage fixture"], cwd=repo_root)

        staged_edit_path.write_bytes(b"needs newline")
        run(["git", "add", "staged_no_restage.txt"], cwd=repo_root)
        staged_edit_path.write_text("local edit stays\n", encoding="utf-8")

        run(["bash", str(FORMATTER), "--staged", "--write"], cwd=repo_root)
        expect_equal(
            "no-restage staged blob normalized in index",
            git_show(repo_root, "staged_no_restage.txt"),
            "needs newline\n",
        )
        expect_equal(
            "no-restage local worktree edits preserved",
            staged_edit_path.read_text(encoding="utf-8"),
            "local edit stays\n",
        )

        deleted_path = repo_root / "deleted_after_stage.txt"
        deleted_path.write_text("baseline\n", encoding="utf-8")
        run(["git", "add", "deleted_after_stage.txt"], cwd=repo_root)
        run(["git", "commit", "-qm", "add delete fixture"], cwd=repo_root)

        deleted_path.write_bytes(b"needs newline")
        run(["git", "add", "deleted_after_stage.txt"], cwd=repo_root)
        deleted_path.unlink()

        run(["bash", str(FORMATTER), "--staged", "--write", "--restage"], cwd=repo_root)
        expect_equal(
            "deleted-path staged blob normalized in index",
            git_show(repo_root, "deleted_after_stage.txt"),
            "needs newline\n",
        )
        if deleted_path.exists():
            raise AssertionError("unstaged deletion should be preserved in worktree")

    print("test_format_local: PASS")
    return 0


def run(args: list[str], cwd: Path) -> None:
    completed = run_completed(args, cwd)
    if completed.returncode == 0:
        return
    print(completed.stdout)
    print(completed.stderr, file=sys.stderr)
    raise SystemExit(completed.returncode)


def run_completed(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed


def expect_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def git_show(cwd: Path, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f":{path}"],
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)
    return completed.stdout


if __name__ == "__main__":
    raise SystemExit(main())
