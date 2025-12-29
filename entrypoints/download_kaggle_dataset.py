"""Download and snapshot the Kaggle cyber-security-attacks dataset.

This script is an *entry point* intended to be run manually during dataset
preparation.

Workflow:
1) Download and unzip the Kaggle dataset into `downloads/` (ignored by git).
2) Copy an immutable, dated snapshot into `data/01-raw/`.

Kaggle authentication:
- Ensure your Kaggle API token exists at:
    - Windows: %USERPROFILE%\\.kaggle\\kaggle.json
  - macOS/Linux: ~/.kaggle/kaggle.json

Dataset:
- Kaggle slug: teamincribo/cyber-security-attacks

Notes:
- This script does not alter any files under `data/01-raw/` in-place.
  If the snapshot folder already exists, it will fail unless `--force`
  is provided.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET = "teamincribo/cyber-security-attacks"


@dataclass(frozen=True)
class SnapshotProvenance:
    """Metadata recorded alongside a raw data snapshot."""

    kaggle_dataset: str
    downloaded_utc: str
    snapshot_date: str
    download_dir: str
    snapshot_dir: str


def _utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_command(args: list[str]) -> None:
    """Run a subprocess command, raising on failure.

    Parameters:
        args: Command + args to execute.

    Raises:
        RuntimeError: If the command fails.
    """

    completed = subprocess.run(args, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "(no stderr)"
        stdout = completed.stdout.strip() or "(no stdout)"
        raise RuntimeError(
            "Command failed:\n"
            f"  cmd: {args}\n"
            f"  exit: {completed.returncode}\n"
            f"  stdout: {stdout}\n"
            f"  stderr: {stderr}"
        )


def _sanitize_dataset_slug(dataset: str) -> str:
    """Convert a Kaggle dataset slug into a filesystem-friendly name."""

    return dataset.replace("/", "_").replace(" ", "_")


def _iter_files(root: Path) -> Iterable[Path]:
    """Yield all files under a directory."""

    yield from (p for p in root.rglob("*") if p.is_file())


def download_and_unzip(dataset: str, downloads_dir: Path, force: bool) -> Path:
    """Download and unzip a Kaggle dataset into the downloads directory.

    Parameters:
        dataset: Kaggle dataset slug, e.g. "owner/dataset".
        downloads_dir: Root downloads directory.
        force: If True, overwrite existing download folder contents.

    Returns:
        Path: Directory containing the extracted dataset files.
    """

    downloads_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = downloads_dir / _sanitize_dataset_slug(dataset)
    if extracted_dir.exists() and force:
        shutil.rmtree(extracted_dir)

    extracted_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(extracted_dir),
        "--unzip",
    ]

    _run_command(cmd)

    # Kaggle sometimes nests files inside additional folders; we accept that.
    file_count = sum(1 for _ in _iter_files(extracted_dir))
    if file_count == 0:
        raise RuntimeError(
            "Kaggle download completed but no files were found in "
            f"{extracted_dir}."
        )

    return extracted_dir


def snapshot_raw_data(
    *,
    extracted_dir: Path,
    raw_root: Path,
    dataset: str,
    snapshot_date: str,
    force: bool,
) -> Path:
    """Copy a dated, immutable snapshot of downloaded data into data/01-raw.

    Parameters:
        extracted_dir: Directory with extracted Kaggle files.
        raw_root: `data/01-raw` directory.
        dataset: Kaggle dataset slug.
        snapshot_date: Date string (YYYY-MM-DD).
        force: If True, overwrite existing snapshot directory.

    Returns:
        Path: The snapshot directory path.
    """

    raw_root.mkdir(parents=True, exist_ok=True)

    dataset_dir = raw_root / _sanitize_dataset_slug(dataset)
    snapshot_dir = dataset_dir / snapshot_date

    if snapshot_dir.exists():
        if not force:
            raise RuntimeError(
                f"Snapshot already exists: {snapshot_dir}. "
                "Re-run with --force to overwrite."
            )
        shutil.rmtree(snapshot_dir)

    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Copy extracted data into the snapshot.
    shutil.copytree(extracted_dir, snapshot_dir, dirs_exist_ok=True)

    provenance = SnapshotProvenance(
        kaggle_dataset=dataset,
        downloaded_utc=_utc_now_iso(),
        snapshot_date=snapshot_date,
        download_dir=str(extracted_dir),
        snapshot_dir=str(snapshot_dir),
    )
    (snapshot_dir / "provenance.json").write_text(
        json.dumps(asdict(provenance), indent=2), encoding="utf-8"
    )

    return snapshot_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Download a Kaggle dataset and snapshot it into data/01-raw."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Kaggle dataset slug (default: teamincribo/cyber-security-attacks).",
    )
    parser.add_argument(
        "--snapshot-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Snapshot date folder name (default: today, YYYY-MM-DD).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing download/snapshot directories.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""

    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    downloads_dir = repo_root / "downloads"
    raw_root = repo_root / "data" / "01-raw"

    extracted_dir = download_and_unzip(
        dataset=args.dataset, downloads_dir=downloads_dir, force=args.force
    )

    snapshot_dir = snapshot_raw_data(
        extracted_dir=extracted_dir,
        raw_root=raw_root,
        dataset=args.dataset,
        snapshot_date=args.snapshot_date,
        force=args.force,
    )

    print(f"Downloaded dataset to: {extracted_dir}")
    print(f"Snapshotted raw data to: {snapshot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
