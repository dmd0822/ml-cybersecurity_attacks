"""Smoke tests to ensure all Python files compile.

These tests are intentionally lightweight and do not execute pipeline entrypoints.
They help catch syntax errors across the repository.
"""

from __future__ import annotations

import py_compile
from pathlib import Path


def _iter_python_files(repo_root: Path) -> list[Path]:
    """Return a sorted list of Python files to compile."""

    include_roots = [repo_root / "src", repo_root / "entrypoints"]

    files: list[Path] = []
    for root in include_roots:
        if not root.exists():
            continue
        files.extend(root.rglob("*.py"))

    # Filter out common noise folders.
    out: list[Path] = []
    for path in files:
        if any(part in {"__pycache__", ".venv", ".git"} for part in path.parts):
            continue
        out.append(path)

    return sorted(set(out))


def test_all_python_files_compile() -> None:
    """Compile every Python file under src/ and entrypoints/."""

    repo_root = Path(__file__).resolve().parents[1]
    python_files = _iter_python_files(repo_root)

    assert python_files, "Expected to find Python files under src/ or entrypoints/."

    failures: list[str] = []
    for path in python_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:  # pragma: no cover
            failures.append(f"{path}: {exc}")

    assert not failures, "\n".join(failures)
