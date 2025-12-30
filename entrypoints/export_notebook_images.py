"""Export embedded image outputs from a Jupyter notebook.

This repo's EDA notebook renders charts inline (via `plt.show()`), which means
no PNG files are written to disk. The ML report generator, however, benefits
from stable filenames that can be referenced from Markdown.

This script extracts `image/png` cell outputs from an `.ipynb` file and writes
PNG files into an output directory.

Usage:
  python entrypoints/export_notebook_images.py \
    --notebook notebooks/analysis_general_2025-12-29.ipynb \
    --outdir reports/assets/analysis_general_2025-12-29

Notes:
- Filenames are stable and based on cell order and output index.
- Only `image/png` outputs are exported.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any


def _safe_write_bytes(path: Path, content: bytes) -> None:
    """Write bytes to `path`, creating parent directories if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _iter_png_outputs(nb: dict[str, Any]) -> list[tuple[int, int, bytes]]:
    """Return a list of (cell_index, output_index, png_bytes) from notebook JSON."""

    extracted: list[tuple[int, int, bytes]] = []
    cells = nb.get("cells", [])
    for cell_index, cell in enumerate(cells):
        outputs = cell.get("outputs") or []
        for output_index, output in enumerate(outputs):
            data = output.get("data") or {}
            png = data.get("image/png")
            if not png:
                continue

            if isinstance(png, list):
                png_b64 = "".join(str(part) for part in png)
            else:
                png_b64 = str(png)

            try:
                extracted.append(
                    (cell_index, output_index, base64.b64decode(png_b64))
                )
            except Exception as exc:  # pragma: no cover
                raise ValueError(
                    "Failed to decode image/png output "
                    f"(cell={cell_index}, output={output_index})."
                ) from exc

    return extracted


def export_notebook_images(notebook_path: Path, out_dir: Path) -> list[Path]:
    """Export all `image/png` outputs from a notebook into `out_dir`.

    Returns:
        List of written file paths.
    """

    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    written: list[Path] = []
    for cell_index, output_index, png_bytes in _iter_png_outputs(nb):
        filename = f"cell_{cell_index:03d}_output_{output_index:02d}.png"
        out_path = out_dir / filename
        _safe_write_bytes(out_path, png_bytes)
        written.append(out_path)

    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export embedded image/png outputs from a Jupyter notebook.",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        required=True,
        help="Path to the .ipynb notebook file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory to write exported PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    notebook_path: Path = args.notebook
    out_dir: Path = args.outdir

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    written = export_notebook_images(notebook_path=notebook_path, out_dir=out_dir)

    print(f"Exported {len(written)} PNG(s) to: {out_dir}")


if __name__ == "__main__":
    main()
