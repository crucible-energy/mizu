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

        staged_partial_path = repo_root / "staged_partial.txt"
        staged_partial_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        run(["git", "add", "staged_partial.txt"], cwd=repo_root)
        run(["git", "commit", "-qm", "add partial fixture"], cwd=repo_root)

        staged_partial_path.write_text("alpha staged\nbeta\ngamma\n", encoding="utf-8")
        run(["git", "add", "staged_partial.txt"], cwd=repo_root)
        staged_partial_path.write_text("alpha staged\nbeta unstaged\ngamma\n", encoding="utf-8")

        run(["bash", str(FORMATTER), "--staged", "--write", "--restage"], cwd=repo_root)

    print("test_format_local: PASS")
    return 0


def run(args: list[str], cwd: Path) -> None:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode == 0:
        return
    print(completed.stdout)
    print(completed.stderr, file=sys.stderr)
    raise SystemExit(completed.returncode)


def expect_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


if __name__ == "__main__":
    raise SystemExit(main())
