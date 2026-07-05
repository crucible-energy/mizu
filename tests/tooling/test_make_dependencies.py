#!/usr/bin/env python3
"""Regression-test direct Make target dependencies."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    run(["make", "build/tests/test_cache_keys", "build/tests/test_opaque_handles"], cwd=REPO_ROOT)

    expect_target_is_out_of_date(
        "unit test binary should track its Fortran sources",
        "build/tests/test_cache_keys",
        REPO_ROOT / "src" / "cache" / "mod_cache_keys.f90",
    )
    expect_target_is_out_of_date(
        "contract test binary should track the public C header",
        "build/tests/test_opaque_handles",
        REPO_ROOT / "include" / "mizu.h",
    )

    print("test_make_dependencies: PASS")
    return 0


def expect_target_is_out_of_date(label: str, target: str, dependency: Path) -> None:
    dependency_stat = dependency.stat()
    bumped_mtime_ns = max(dependency_stat.st_mtime_ns, time.time_ns()) + 5_000_000_000

    try:
        os.utime(dependency, ns=(dependency_stat.st_atime_ns, bumped_mtime_ns))
        completed = run_completed(["make", "-q", target], cwd=REPO_ROOT)
        if completed.returncode == 1:
            return
        raise AssertionError(
            f"{label}: expected `make -q {target}` to return 1, got {completed.returncode}; "
            f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
        )
    finally:
        os.utime(dependency, ns=(dependency_stat.st_atime_ns, dependency_stat.st_mtime_ns))


def run(args: list[str], cwd: Path) -> None:
    completed = run_completed(args, cwd)
    if completed.returncode == 0:
        return
    print(completed.stdout)
    print(completed.stderr, file=sys.stderr)
    raise SystemExit(completed.returncode)


def run_completed(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


if __name__ == "__main__":
    raise SystemExit(main())
