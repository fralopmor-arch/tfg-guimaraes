# Utility to archive curve outputs

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class CurveFile:
    path: Path
    size_bytes: int


def _collect_curve_files(curves_dir: Path) -> list[CurveFile]:
    files: list[CurveFile] = []
    for path in curves_dir.glob("*_curves.csv"):
        stat = path.stat()
        files.append(CurveFile(path=path, size_bytes=stat.st_size))
    return files


def _build_archive_path(archive_dir: Path, now: datetime | None = None) -> Path:
    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return archive_dir / f"curves_archive_{ts}.zip"


def _write_zip_archive(archive_path: Path, files: list[CurveFile], root_dir: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for curve in files:
            arcname = curve.path.relative_to(root_dir).as_posix()
            zf.write(curve.path, arcname=arcname)


def _write_manifest(manifest_path: Path, archived: list[CurveFile]) -> None:
    data = {
        "archived_count": len(archived),
        "kept_count": 0,
        "archived_files": [
            {
                "path": item.path.as_posix(),
                "size_bytes": item.size_bytes,
            }
            for item in archived
        ],
        "kept_files": [],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _sum_bytes(items: list[CurveFile]) -> int:
    return sum(item.size_bytes for item in items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive all per-motor curve CSV outputs into a timestamped zip and delete originals."
        )
    )
    parser.add_argument(
        "--curves-dir",
        type=Path,
        default=Path("outputs/curves"),
        help="Directory containing *_curves.csv files.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("outputs/curves_archive"),
        help="Directory where zip archives and manifests will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    curves_dir: Path = args.curves_dir
    archive_dir: Path = args.archive_dir

    if not curves_dir.exists():
        print(f"No action: curves directory does not exist: {curves_dir.as_posix()}")
        return 0

    all_curves = _collect_curve_files(curves_dir)
    if not all_curves:
        print(f"No action: no curve CSV files found in {curves_dir.as_posix()}")
        return 0

    archived = all_curves

    print(f"found={len(all_curves)} archive={len(archived)}")
    print(f"archive_bytes={_sum_bytes(archived)}")

    archive_path = _build_archive_path(archive_dir)
    manifest_path = archive_path.with_suffix(".manifest.json")

    _write_zip_archive(archive_path, archived, root_dir=curves_dir)
    _write_manifest(manifest_path, archived=archived)

    deleted = 0
    for item in archived:
        item.path.unlink(missing_ok=True)
        deleted += 1

    print(f"archive_created={archive_path.as_posix()}")
    print(f"manifest_created={manifest_path.as_posix()}")
    print(f"deleted={deleted}")
    print("kept_in_place=0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
