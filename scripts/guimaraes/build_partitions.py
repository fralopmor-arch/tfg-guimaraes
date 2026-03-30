from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.guimaraes.catalog_schema import (
        MotorRecord,
        enforce_grouping_constraints,
        motor_record_from_row,
    )
    from scripts.guimaraes.mode_config import ExecutionMode, ModeConfig, normalize_mode, resolve_mode_config
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from catalog_schema import MotorRecord, enforce_grouping_constraints, motor_record_from_row
    from mode_config import ExecutionMode, ModeConfig, normalize_mode, resolve_mode_config


@dataclass(frozen=True)
class RawRowRecord:
    row: dict[str, str]
    record: MotorRecord


@dataclass(frozen=True)
class PartitionBatch:
    source_name: str
    mode: ExecutionMode
    group_key: tuple[str, str, str, str, str]
    batch_index: int
    rows_count: int
    csv_path: Path


@dataclass(frozen=True)
class PartitionBuildResult:
    batches: list[PartitionBatch]
    manifest_path: Path
    reused_existing: bool


def _sanitize_segment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.strip())
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return collapsed or "unknown"


def _group_segment(group_key: tuple[str, str, str, str, str]) -> str:
    manufacturer, efficiency_class, starting_category, frequency_hz, pole_pairs = group_key
    parts = [
        _sanitize_segment(manufacturer),
        _sanitize_segment(efficiency_class),
        _sanitize_segment(starting_category),
        _sanitize_segment(f"{frequency_hz}Hz"),
        _sanitize_segment(f"pp{pole_pairs}"),
    ]
    return "__".join(parts)


def _load_rows_with_records(input_master: Path) -> tuple[list[str], list[RawRowRecord]]:
    with input_master.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError("Input CSV must include a header row")

        loaded: list[RawRowRecord] = []
        for index, row in enumerate(reader, start=1):
            record = motor_record_from_row(row, row_index=index)
            loaded.append(RawRowRecord(row=row, record=record))
    return fieldnames, loaded


def _group_by_method_dimensions(rows: list[RawRowRecord]) -> dict[tuple[str, str, str, str, str], list[RawRowRecord]]:
    grouped: dict[tuple[str, str, str, str, str], list[RawRowRecord]] = {}
    for item in rows:
        grouped.setdefault(item.record.group_key, []).append(item)

    for group_key, grouped_rows in grouped.items():
        group_name = "|".join(group_key)
        enforce_grouping_constraints(group_name, [item.record for item in grouped_rows])

    return grouped


def _sort_group_rows(rows: list[RawRowRecord]) -> list[RawRowRecord]:
    return sorted(
        rows,
        key=lambda item: (
            item.record.motor_id,
            item.record.rated_power_w,
            item.record.rated_voltage_v,
        ),
    )


def _chunk_rows(rows: list[RawRowRecord], chunk_size: int) -> list[list[RawRowRecord]]:
    return [rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)]


def _build_mode_batches(
    rows: list[RawRowRecord],
    *,
    mode: ExecutionMode,
    config: ModeConfig,
) -> list[list[RawRowRecord]]:
    ordered_rows = _sort_group_rows(rows)
    if mode == "smoke":
        return [ordered_rows[: config.smoke_size]] if ordered_rows else []
    if mode == "fast":
        return [ordered_rows[: config.fast_size]] if ordered_rows else []
    if mode == "intermediate":
        return _chunk_rows(ordered_rows, config.intermediate_batch_size)

    if config.full_batch_size is None:
        return [ordered_rows] if ordered_rows else []
    return _chunk_rows(ordered_rows, config.full_batch_size)


def _clean_existing_mode_dirs(output_root: Path, source_name: str, mode: ExecutionMode) -> None:
    source_root = output_root / source_name
    if not source_root.exists():
        return
    for path in source_root.rglob(mode):
        if path.is_dir():
            shutil.rmtree(path)


def _build_input_signature(input_master: Path) -> dict[str, object]:
    stat = input_master.stat()
    return {
        "path": input_master.as_posix(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _manifest_config_payload(config: ModeConfig) -> dict[str, int | None]:
    return {
        "smoke_size": config.smoke_size,
        "fast_size": config.fast_size,
        "intermediate_batch_size": config.intermediate_batch_size,
        "full_batch_size": config.full_batch_size,
    }


def _manifest_path(*, output_root: Path, source_name: str, mode: ExecutionMode) -> Path:
    return output_root / source_name / f"manifest_{mode}.json"


def _batch_from_manifest_row(source_name: str, mode: ExecutionMode, row: Mapping[str, object]) -> PartitionBatch:
    group_key_payload = row["group_key"]
    if not isinstance(group_key_payload, Mapping):
        raise ValueError("Invalid manifest format: group_key must be an object")

    group_key = (
        str(group_key_payload["manufacturer"]),
        str(group_key_payload["efficiency_class"]),
        str(group_key_payload["starting_torque_category"]),
        str(group_key_payload["frequency_hz"]),
        str(group_key_payload["pole_pairs"]),
    )
    return PartitionBatch(
        source_name=source_name,
        mode=mode,
        group_key=group_key,
        batch_index=int(row["batch_index"]),
        rows_count=int(row["rows_count"]),
        csv_path=Path(str(row["csv_path"])),
    )


def _try_load_cached_partitions(
    *,
    input_master: Path,
    source_name: str,
    mode: ExecutionMode,
    output_root: Path,
    config: ModeConfig,
) -> PartitionBuildResult | None:
    manifest_path = _manifest_path(output_root=output_root, source_name=source_name, mode=mode)
    if not manifest_path.exists():
        return None

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    expected_signature = _build_input_signature(input_master)
    cached_signature = data.get("input_signature")
    if cached_signature != expected_signature:
        return None

    if data.get("source_name") != source_name or data.get("mode") != mode:
        return None

    if data.get("mode_config") != _manifest_config_payload(config):
        return None

    raw_batches = data.get("batches")
    if not isinstance(raw_batches, list):
        return None

    batches: list[PartitionBatch] = []
    for row in raw_batches:
        if not isinstance(row, Mapping):
            return None
        batch = _batch_from_manifest_row(source_name, mode, row)
        if not batch.csv_path.exists():
            return None
        batches.append(batch)

    return PartitionBuildResult(batches=batches, manifest_path=manifest_path, reused_existing=True)


def build_partitions(
    *,
    input_master: Path,
    source_name: str,
    mode: ExecutionMode,
    output_root: Path,
    config: ModeConfig,
) -> list[PartitionBatch]:
    fieldnames, loaded = _load_rows_with_records(input_master)
    grouped = _group_by_method_dimensions(loaded)

    _clean_existing_mode_dirs(output_root, source_name, mode)

    batches: list[PartitionBatch] = []
    for group_key in sorted(grouped.keys()):
        group_rows = grouped[group_key]
        for batch_index, batch_rows in enumerate(
            _build_mode_batches(group_rows, mode=mode, config=config),
            start=1,
        ):
            if not batch_rows:
                continue

            group_segment = _group_segment(group_key)
            batch_dir = output_root / source_name / group_segment / mode
            batch_dir.mkdir(parents=True, exist_ok=True)
            batch_path = batch_dir / f"batch_{batch_index:03d}.csv"

            with batch_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in batch_rows:
                    writer.writerow(row.row)

            batches.append(
                PartitionBatch(
                    source_name=source_name,
                    mode=mode,
                    group_key=group_key,
                    batch_index=batch_index,
                    rows_count=len(batch_rows),
                    csv_path=batch_path,
                )
            )

    return batches


def write_partition_manifest(
    *,
    input_master: Path,
    source_name: str,
    mode: ExecutionMode,
    config: ModeConfig,
    output_root: Path,
    batches: list[PartitionBatch],
) -> Path:
    manifest_path = _manifest_path(output_root=output_root, source_name=source_name, mode=mode)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "input_master": input_master.as_posix(),
        "source_name": source_name,
        "mode": mode,
        "input_signature": _build_input_signature(input_master),
        "mode_config": {
            "smoke_size": config.smoke_size,
            "fast_size": config.fast_size,
            "intermediate_batch_size": config.intermediate_batch_size,
            "full_batch_size": config.full_batch_size,
        },
        "total_batches": len(batches),
        "total_rows": sum(batch.rows_count for batch in batches),
        "batches": [
            {
                "group_key": {
                    "manufacturer": batch.group_key[0],
                    "efficiency_class": batch.group_key[1],
                    "starting_torque_category": batch.group_key[2],
                    "frequency_hz": batch.group_key[3],
                    "pole_pairs": batch.group_key[4],
                },
                "batch_index": batch.batch_index,
                "rows_count": batch.rows_count,
                "csv_path": batch.csv_path.as_posix(),
            }
            for batch in batches
        ],
    }
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return manifest_path


def build_or_load_partitions(
    *,
    input_master: Path,
    source_name: str,
    mode: ExecutionMode,
    output_root: Path,
    config: ModeConfig,
    reuse_if_unchanged: bool,
) -> PartitionBuildResult:
    if reuse_if_unchanged:
        cached = _try_load_cached_partitions(
            input_master=input_master,
            source_name=source_name,
            mode=mode,
            output_root=output_root,
            config=config,
        )
        if cached is not None:
            return cached

    batches = build_partitions(
        input_master=input_master,
        source_name=source_name,
        mode=mode,
        output_root=output_root,
        config=config,
    )
    manifest_path = write_partition_manifest(
        input_master=input_master,
        source_name=source_name,
        mode=mode,
        config=config,
        output_root=output_root,
        batches=batches,
    )
    return PartitionBuildResult(batches=batches, manifest_path=manifest_path, reused_existing=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic per-group CSV partitions from one source master CSV. "
            "Groups follow Guimaraes method dimensions: manufacturer, efficiency class, "
            "starting category, frequency and pole-pair count."
        )
    )
    parser.add_argument("--input-master", required=True, help="Source CSV master file")
    parser.add_argument(
        "--source-name",
        help="Source label used in output paths. Defaults to input file stem.",
    )
    parser.add_argument("--mode", required=True, choices=["smoke", "fast", "intermediate", "full"])
    parser.add_argument(
        "--output-root",
        default="data/partitions",
        help="Root output folder for partition CSV files",
    )
    parser.add_argument("--smoke-size", type=int, help="Smoke sample size per coherent group (5-10)")
    parser.add_argument("--fast-size", type=int, help="Fast sample size per coherent group (20-50)")
    parser.add_argument(
        "--intermediate-batch-size",
        type=int,
        help="Batch size for intermediate mode",
    )
    parser.add_argument(
        "--full-batch-size",
        type=int,
        help="Optional batch size for full mode. Omit for one full batch per coherent group.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    mode = normalize_mode(args.mode)
    config = resolve_mode_config(
        smoke_size=args.smoke_size,
        fast_size=args.fast_size,
        intermediate_batch_size=args.intermediate_batch_size,
        full_batch_size=args.full_batch_size,
    )

    input_master = Path(args.input_master)
    source_name = args.source_name or input_master.stem
    output_root = Path(args.output_root)

    batches = build_partitions(
        input_master=input_master,
        source_name=source_name,
        mode=mode,
        output_root=output_root,
        config=config,
    )
    manifest_path = write_partition_manifest(
        input_master=input_master,
        source_name=source_name,
        mode=mode,
        config=config,
        output_root=output_root,
        batches=batches,
    )

    print(f"mode={mode} source={source_name}")
    print(f"batches={len(batches)} rows={sum(batch.rows_count for batch in batches)}")
    print(f"manifest={manifest_path.as_posix()}")
    for batch in batches:
        print(
            f"batch={batch.batch_index:03d} rows={batch.rows_count} "
            f"group={batch.group_key[0]}|{batch.group_key[1]}|{batch.group_key[2]}|{batch.group_key[3]}|{batch.group_key[4]} "
            f"path={batch.csv_path.as_posix()}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
