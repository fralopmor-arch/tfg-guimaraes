from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

try:
    from build_partitions import build_or_load_partitions
    from mode_config import ExecutionMode, ModeConfig, resolve_mode_config
    from validate_guimaraes import (
        DEFAULT_CURRENT_PCT_THRESHOLD,
        DEFAULT_EFFICIENCY_ABS_THRESHOLD,
        LegacyRuntimeConfig,
        DEFAULT_POWER_FACTOR_ABS_THRESHOLD,
        DEFAULT_RATIO_PCT_THRESHOLD,
        DEFAULT_TORQUE_PCT_THRESHOLD,
        ValidationRuntimeConfig,
        resolve_validation_runtime_config,
        run_comparison_validation,
        run_validation,
    )
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from scripts.guimaraes.build_partitions import build_or_load_partitions
    from scripts.guimaraes.mode_config import ExecutionMode, ModeConfig, resolve_mode_config
    from scripts.guimaraes.validate_guimaraes import (
        DEFAULT_CURRENT_PCT_THRESHOLD,
        DEFAULT_EFFICIENCY_ABS_THRESHOLD,
        LegacyRuntimeConfig,
        DEFAULT_POWER_FACTOR_ABS_THRESHOLD,
        DEFAULT_RATIO_PCT_THRESHOLD,
        DEFAULT_TORQUE_PCT_THRESHOLD,
        ValidationRuntimeConfig,
        resolve_validation_runtime_config,
        run_comparison_validation,
        run_validation,
    )
def _resolve_modes(value: str) -> list[ExecutionMode]:
    mode = value.strip().lower()
    if mode == "all":
        return ["smoke", "fast", "intermediate", "full"]
    if mode in {"smoke", "fast", "intermediate", "full"}:
        return [mode]  # type: ignore[list-item]
    raise ValueError("mode must be one of: smoke, fast, intermediate, full, all")


def _build_report_path(
    *,
    partition_csv: Path,
    partitions_root: Path,
    source_name: str,
    reports_root: Path,
) -> Path:
    relative = partition_csv.relative_to(partitions_root / source_name)
    target = reports_root / source_name / relative
    return target.with_name(f"{target.stem}_report.json")


def _build_curves_dir(
    *,
    export_curves_dir: Path | None,
    source_name: str,
    mode: ExecutionMode,
    batch_csv: Path,
) -> Path | None:
    if export_curves_dir is None:
        return None
    return export_curves_dir / source_name / mode / batch_csv.stem


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _merge_stage_rollup(target: dict[str, dict[str, float]], source: dict[str, Any], *, prefix: str = "") -> None:
    for stage_name, payload in source.items():
        if not isinstance(payload, dict):
            continue
        key = f"{prefix}{stage_name}"
        slot = target.setdefault(key, {"seconds": 0.0, "count": 0.0})
        slot["seconds"] += float(payload.get("seconds", 0.0))
        slot["count"] += float(payload.get("count", 0.0))


def _legacy_profile_defaults(mode: ExecutionMode, fidelity: str) -> tuple[int, float]:
    if fidelity == "benchmark":
        return 600, 1.0

    iterative_by_mode = {
        "smoke": (140, 0.40),
        "fast": (220, 0.55),
        "intermediate": (350, 0.80),
        "full": (600, 1.0),
    }
    return iterative_by_mode[mode]


def _resolve_runtime_config_for_mode(
    *,
    mode: ExecutionMode,
    legacy_fidelity: str,
    legacy_breakdown_samples: int | None,
    legacy_grid_scale: float | None,
) -> ValidationRuntimeConfig:
    default_samples, default_scale = _legacy_profile_defaults(mode, legacy_fidelity)
    runtime = resolve_validation_runtime_config(
        legacy_breakdown_samples=legacy_breakdown_samples or default_samples,
        legacy_grid_scale=legacy_grid_scale if legacy_grid_scale is not None else default_scale,
    )

    # Keep deterministic mode untouched for benchmark/final runs by clamping only legacy knobs.
    base = runtime.legacy_solver
    return ValidationRuntimeConfig(
        legacy_solver=LegacyRuntimeConfig(
            breakdown_samples=max(20, base.breakdown_samples),
            beta_min_tenths=base.beta_min_tenths,
            beta_max_tenths=base.beta_max_tenths,
            alpha_min_tenths=base.alpha_min_tenths,
            alpha_max_tenths=base.alpha_max_tenths,
            r1_steps=max(3, base.r1_steps),
            gr_steps=max(3, base.gr_steps),
            rm_steps=max(3, base.rm_steps),
            xm_steps=max(3, base.xm_steps),
        )
    )


def _run_mode(
    *,
    mode: ExecutionMode,
    input_master: Path,
    source_name: str,
    partitions_root: Path,
    reports_root: Path,
    config: ModeConfig,
    current_threshold_pct: float,
    torque_threshold_pct: float,
    efficiency_threshold_abs: float,
    pf_threshold_abs: float,
    ratio_threshold_pct: float,
    export_curves_dir: Path | None,
    curve_points: int,
    solver: str,
    compare_with_legacy: bool,
    reuse_partitions: bool,
    legacy_fidelity: str,
    legacy_breakdown_samples: int | None,
    legacy_grid_scale: float | None,
) -> dict[str, Any]:
    mode_start = time.perf_counter()
    runtime_config = _resolve_runtime_config_for_mode(
        mode=mode,
        legacy_fidelity=legacy_fidelity,
        legacy_breakdown_samples=legacy_breakdown_samples,
        legacy_grid_scale=legacy_grid_scale,
    )

    partition_start = time.perf_counter()
    partition_result = build_or_load_partitions(
        input_master=input_master,
        source_name=source_name,
        mode=mode,
        output_root=partitions_root,
        config=config,
        reuse_if_unchanged=reuse_partitions,
    )
    partition_seconds = time.perf_counter() - partition_start
    batches = partition_result.batches
    manifest_path = partition_result.manifest_path

    print(
        "mode={} partitions={} manifest={} reused_partitions={} build_seconds={:.3f}".format(
            mode,
            len(batches),
            manifest_path.as_posix(),
            partition_result.reused_existing,
            partition_seconds,
        )
    )

    batch_summaries: list[dict[str, Any]] = []
    total_validated = 0
    total_skipped = 0
    total_rows = 0
    total_pass = 0
    all_batches_pass = True
    accumulated_stages: dict[str, dict[str, float]] = {}
    batch_validation_seconds_total = 0.0
    batch_write_seconds_total = 0.0

    total_batches = len(batches)
    for index, batch in enumerate(batches, start=1):
        curves_dir = _build_curves_dir(
            export_curves_dir=export_curves_dir,
            source_name=source_name,
            mode=mode,
            batch_csv=batch.csv_path,
        )

        validate_start = time.perf_counter()
        if compare_with_legacy:
            report = run_comparison_validation(
                input_csv=batch.csv_path,
                current_pct_threshold=current_threshold_pct,
                torque_pct_threshold=torque_threshold_pct,
                efficiency_abs_threshold=efficiency_threshold_abs,
                power_factor_abs_threshold=pf_threshold_abs,
                ratio_pct_threshold=ratio_threshold_pct,
                curves_output_dir=curves_dir,
                curve_points=curve_points,
                runtime_config=runtime_config,
            )
            report_for_counts = report["deterministic"]
        else:
            report = run_validation(
                input_csv=batch.csv_path,
                current_pct_threshold=current_threshold_pct,
                torque_pct_threshold=torque_threshold_pct,
                efficiency_abs_threshold=efficiency_threshold_abs,
                power_factor_abs_threshold=pf_threshold_abs,
                ratio_pct_threshold=ratio_threshold_pct,
                curves_output_dir=curves_dir,
                curve_points=curve_points,
                solver=solver,
                runtime_config=runtime_config,
            )
            report_for_counts = report
        batch_validation_seconds = time.perf_counter() - validate_start
        batch_validation_seconds_total += batch_validation_seconds

        report_timing = report_for_counts.get("timing")
        if isinstance(report_timing, dict):
            stages = report_timing.get("stages")
            if isinstance(stages, dict):
                _merge_stage_rollup(accumulated_stages, stages)
        if compare_with_legacy and isinstance(report.get("timing"), dict):
            comparison_timing = report.get("timing", {})
            accumulated_stages.setdefault("comparison.wrapper", {"seconds": 0.0, "count": 0.0})
            accumulated_stages["comparison.wrapper"]["seconds"] += float(
                comparison_timing.get("total_runtime_seconds", 0.0)
            )
            accumulated_stages["comparison.wrapper"]["count"] += 1.0
            det_stages = report.get("deterministic", {}).get("timing", {}).get("stages", {})
            if isinstance(det_stages, dict):
                _merge_stage_rollup(accumulated_stages, det_stages, prefix="deterministic.")
            leg_stages = report.get("legacy", {}).get("timing", {}).get("stages", {})
            if isinstance(leg_stages, dict):
                _merge_stage_rollup(accumulated_stages, leg_stages, prefix="legacy.")

        report_path = _build_report_path(
            partition_csv=batch.csv_path,
            partitions_root=partitions_root,
            source_name=source_name,
            reports_root=reports_root,
        )
        write_start = time.perf_counter()
        _write_json(report_path, report)
        write_seconds = time.perf_counter() - write_start
        batch_write_seconds_total += write_seconds

        validated_count = len(report_for_counts["results"])
        skipped_count = len(report_for_counts["skipped"])
        passed_count = sum(1 for row in report_for_counts["results"] if row["pass_overall"])
        pass_rate = passed_count / validated_count if validated_count else 0.0

        total_rows += batch.rows_count
        total_validated += validated_count
        total_skipped += skipped_count
        total_pass += passed_count
        all_batches_pass = all_batches_pass and report_for_counts["overall_pass"]

        batch_summary = {
            "batch_index": index,
            "partition_csv": batch.csv_path.as_posix(),
            "report_json": report_path.as_posix(),
            "rows_count": batch.rows_count,
            "validated_count": validated_count,
            "skipped_count": skipped_count,
            "pass_rate": pass_rate,
            "overall_pass": report_for_counts["overall_pass"],
            "timing": {
                "validation_seconds": batch_validation_seconds,
                "write_report_seconds": write_seconds,
            },
        }
        if compare_with_legacy:
            batch_summary["comparison_r2"] = report["comparison"]["regression_r2"]
            if isinstance(report.get("timing"), dict):
                batch_summary["comparison_timing"] = report["timing"]
        batch_summaries.append(batch_summary)

        print(
            f"[{index}/{total_batches}] mode={mode} rows={batch.rows_count} "
            f"validated={validated_count} skipped={skipped_count} pass_rate={pass_rate:.3f} "
            f"validate_s={batch_validation_seconds:.3f} write_s={write_seconds:.3f}"
        )

    mode_seconds = time.perf_counter() - mode_start
    stage_total_seconds = sum(item["seconds"] for item in accumulated_stages.values())
    ranked_hotspots = sorted(
        (
            {
                "stage": stage,
                "seconds": values["seconds"],
                "count": int(values["count"]),
                "pct_of_accumulated_stages": (
                    (values["seconds"] * 100.0 / stage_total_seconds) if stage_total_seconds > 0.0 else 0.0
                ),
            }
            for stage, values in accumulated_stages.items()
        ),
        key=lambda item: float(item["seconds"]),
        reverse=True,
    )

    aggregate = {
        "mode": mode,
        "solver": solver,
        "compare_with_legacy": compare_with_legacy,
        "input_master": input_master.as_posix(),
        "source_name": source_name,
        "partition_manifest": manifest_path.as_posix(),
        "partition_reused": partition_result.reused_existing,
        "runtime_config": {
            "legacy_fidelity": legacy_fidelity,
            "legacy_solver": {
                "breakdown_samples": runtime_config.legacy_solver.breakdown_samples,
                "beta_min_tenths": runtime_config.legacy_solver.beta_min_tenths,
                "beta_max_tenths": runtime_config.legacy_solver.beta_max_tenths,
                "alpha_min_tenths": runtime_config.legacy_solver.alpha_min_tenths,
                "alpha_max_tenths": runtime_config.legacy_solver.alpha_max_tenths,
                "r1_steps": runtime_config.legacy_solver.r1_steps,
                "gr_steps": runtime_config.legacy_solver.gr_steps,
                "rm_steps": runtime_config.legacy_solver.rm_steps,
                "xm_steps": runtime_config.legacy_solver.xm_steps,
            },
        },
        "totals": {
            "batches": total_batches,
            "rows_count": total_rows,
            "validated_count": total_validated,
            "skipped_count": total_skipped,
            "pass_count": total_pass,
            "overall_pass_rate": (total_pass / total_validated) if total_validated else 0.0,
            "overall_pass": all_batches_pass and total_batches > 0,
        },
        "timing": {
            "partition_stage_seconds": partition_seconds,
            "batch_validation_seconds": batch_validation_seconds_total,
            "batch_report_write_seconds": batch_write_seconds_total,
            "mode_total_runtime_seconds": mode_seconds,
            "accumulated_stage_seconds": stage_total_seconds,
            "hotspots": ranked_hotspots,
        },
        "batch_summaries": batch_summaries,
    }

    aggregate_path = reports_root / source_name / mode / "aggregate_report.json"
    _write_json(aggregate_path, aggregate)
    print(f"mode={mode} aggregate={aggregate_path.as_posix()}")
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic partitions and run Guimaraes validation in modes: "
            "smoke, fast, intermediate, full (or all)."
        )
    )
    parser.add_argument("--input-master", required=True, help="Source master CSV path")
    parser.add_argument("--source-name", help="Source label for output folders (default: input stem)")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["smoke", "fast", "intermediate", "full", "all"],
        help="Execution mode",
    )
    parser.add_argument(
        "--partitions-root",
        default="data/partitions",
        help="Root directory for partition CSV files",
    )
    parser.add_argument(
        "--reports-root",
        default="outputs/validation_modes",
        help="Root directory for validation mode reports",
    )

    parser.add_argument("--smoke-size", type=int, help="Smoke sample size per coherent group (5-10)")
    parser.add_argument("--fast-size", type=int, help="Fast sample size per coherent group (20-50)")
    parser.add_argument("--intermediate-batch-size", type=int, help="Intermediate fixed batch size")
    parser.add_argument(
        "--full-batch-size",
        type=int,
        help="Optional full mode batch size; omit to process all rows in one group batch",
    )

    parser.add_argument("--current-threshold-pct", type=float, default=DEFAULT_CURRENT_PCT_THRESHOLD)
    parser.add_argument("--torque-threshold-pct", type=float, default=DEFAULT_TORQUE_PCT_THRESHOLD)
    parser.add_argument("--efficiency-threshold-abs", type=float, default=DEFAULT_EFFICIENCY_ABS_THRESHOLD)
    parser.add_argument("--pf-threshold-abs", type=float, default=DEFAULT_POWER_FACTOR_ABS_THRESHOLD)
    parser.add_argument("--ratio-threshold-pct", type=float, default=DEFAULT_RATIO_PCT_THRESHOLD)
    parser.add_argument(
        "--export-curves-dir",
        help="Optional root directory for curve exports. Curves are written per mode/batch subfolder.",
    )
    parser.add_argument(
        "--curve-points",
        type=int,
        default=300,
        help="Number of slip samples used when exporting curves",
    )
    parser.add_argument(
        "--solver",
        default="deterministic",
        choices=["deterministic", "legacy", "auto"],
        help="Estimator used by batch validation",
    )
    parser.add_argument(
        "--compare-with-legacy",
        action="store_true",
        help="Generate per-batch deterministic-vs-legacy comparison reports",
    )
    parser.add_argument(
        "--legacy-fidelity",
        choices=["iterative", "benchmark"],
        default="iterative",
        help="Legacy-solver effort preset: iterative for daily smoke/fast, benchmark for high-fidelity reports.",
    )
    parser.add_argument(
        "--legacy-breakdown-samples",
        type=int,
        help="Optional override for legacy breakdown search samples.",
    )
    parser.add_argument(
        "--legacy-grid-scale",
        type=float,
        help="Optional override scaling for legacy grid-search loops (1.0 baseline).",
    )
    parser.add_argument(
        "--force-rebuild-partitions",
        action="store_true",
        help="Force partition rebuild even when manifest/input/mode configuration are unchanged.",
    )
    parser.add_argument(
        "--log-level",
        default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for solver/guard events (default: ERROR to reduce terminal overhead).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")

    input_master = Path(args.input_master)
    source_name = args.source_name or input_master.stem
    partitions_root = Path(args.partitions_root)
    reports_root = Path(args.reports_root)
    config = resolve_mode_config(
        smoke_size=args.smoke_size,
        fast_size=args.fast_size,
        intermediate_batch_size=args.intermediate_batch_size,
        full_batch_size=args.full_batch_size,
    )

    aggregates: list[dict[str, Any]] = []
    for mode in _resolve_modes(args.mode):
        aggregate = _run_mode(
            mode=mode,
            input_master=input_master,
            source_name=source_name,
            partitions_root=partitions_root,
            reports_root=reports_root,
            config=config,
            current_threshold_pct=args.current_threshold_pct,
            torque_threshold_pct=args.torque_threshold_pct,
            efficiency_threshold_abs=args.efficiency_threshold_abs,
            pf_threshold_abs=args.pf_threshold_abs,
            ratio_threshold_pct=args.ratio_threshold_pct,
            export_curves_dir=Path(args.export_curves_dir) if args.export_curves_dir else None,
            curve_points=args.curve_points,
            solver=args.solver,
            compare_with_legacy=args.compare_with_legacy,
            reuse_partitions=not args.force_rebuild_partitions,
            legacy_fidelity=args.legacy_fidelity,
            legacy_breakdown_samples=args.legacy_breakdown_samples,
            legacy_grid_scale=args.legacy_grid_scale,
        )
        aggregates.append(aggregate)

    summary = {
        "input_master": input_master.as_posix(),
        "source_name": source_name,
        "executed_modes": [item["mode"] for item in aggregates],
        "mode_aggregates": aggregates,
    }
    summary_path = reports_root / source_name / "summary.json"
    _write_json(summary_path, summary)
    print(f"summary={summary_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
