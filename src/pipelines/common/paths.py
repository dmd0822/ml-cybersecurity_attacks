"""Path helpers for this repository.

Centralizes folder conventions so entry points and notebooks can reuse
the same layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoPaths:
    """Resolved repository paths."""

    repo_root: Path
    config_dir: Path
    data_dir: Path
    raw_dir: Path
    preprocessed_dir: Path


def get_repo_paths(repo_root: Path) -> RepoPaths:
    """Resolve commonly used repository directories.

    Parameters:
        repo_root: The repository root path.

    Returns:
        RepoPaths: Resolved paths.
    """

    data_dir = repo_root / "data"
    return RepoPaths(
        repo_root=repo_root,
        config_dir=repo_root / "config",
        data_dir=data_dir,
        raw_dir=data_dir / "01-raw",
        preprocessed_dir=data_dir / "02-preprocessed",
    )
