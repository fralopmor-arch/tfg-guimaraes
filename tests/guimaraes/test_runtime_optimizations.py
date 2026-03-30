from __future__ import annotations

import csv
from pathlib import Path

from scripts.guimaraes.build_partitions import build_or_load_partitions
from scripts.guimaraes.mode_config import resolve_mode_config
from scripts.guimaraes.run_validation_modes import _resolve_runtime_config_for_mode, build_parser
from scripts.guimaraes.validate_guimaraes import (
    resolve_validation_runtime_config,
    run_comparison_validation,
    run_validation,
)


def _write_subset_csv(src: Path, dst: Path, *, rows: int) -> None:
    with src.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        items = list(reader)
        fieldnames = list(reader.fieldnames or [])

    assert fieldnames
    assert len(items) >= rows

    with dst.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in items[:rows]:
            writer.writerow(row)


def test_runtime_profile_is_lighter_for_smoke_iterative() -> None:
    iterative = _resolve_runtime_config_for_mode(
        mode="smoke",
        legacy_fidelity="iterative",
        legacy_breakdown_samples=None,
        legacy_grid_scale=None,
    )
    benchmark = _resolve_runtime_config_for_mode(
        mode="smoke",
        legacy_fidelity="benchmark",
        legacy_breakdown_samples=None,
        legacy_grid_scale=None,
    )

    assert iterative.legacy_solver.breakdown_samples < benchmark.legacy_solver.breakdown_samples
    assert iterative.legacy_solver.r1_steps < benchmark.legacy_solver.r1_steps
    assert iterative.legacy_solver.gr_steps < benchmark.legacy_solver.gr_steps


def test_partition_manifest_reuse_skips_rebuild(tmp_path: Path) -> None:
    src = Path("data/catalog_weg_w21_aluminium_multimounting_p9_p21_extracted.csv")
    subset = tmp_path / "subset.csv"
    _write_subset_csv(src, subset, rows=10)

    config = resolve_mode_config(smoke_size=5)
    output_root = tmp_path / "partitions"

    first = build_or_load_partitions(
        input_master=subset,
        source_name="test_source",
        mode="smoke",
        output_root=output_root,
        config=config,
        reuse_if_unchanged=True,
    )
    assert first.reused_existing is False
    assert first.batches

    mtimes = {batch.csv_path: batch.csv_path.stat().st_mtime_ns for batch in first.batches}

    second = build_or_load_partitions(
        input_master=subset,
        source_name="test_source",
        mode="smoke",
        output_root=output_root,
        config=config,
        reuse_if_unchanged=True,
    )
    assert second.reused_existing is True
    assert [item.csv_path for item in second.batches] == [item.csv_path for item in first.batches]
    assert {batch.csv_path: batch.csv_path.stat().st_mtime_ns for batch in second.batches} == mtimes


def test_validation_and_comparison_include_timing_metadata(tmp_path: Path) -> None:
    src = Path("data/catalog_weg_w21_aluminium_multimounting_p9_p21_extracted.csv")
    subset = tmp_path / "subset_small.csv"
    _write_subset_csv(src, subset, rows=3)

    runtime = resolve_validation_runtime_config(legacy_breakdown_samples=40, legacy_grid_scale=0.25)

    validation_report = run_validation(
        input_csv=subset,
        current_pct_threshold=15.0,
        torque_pct_threshold=15.0,
        efficiency_abs_threshold=0.05,
        power_factor_abs_threshold=0.05,
        ratio_pct_threshold=15.0,
        curves_output_dir=None,
        curve_points=40,
        solver="deterministic",
        runtime_config=runtime,
    )
    assert "timing" in validation_report
    assert "runtime_config" in validation_report
    assert "stages" in validation_report["timing"]

    comparison_report = run_comparison_validation(
        input_csv=subset,
        current_pct_threshold=15.0,
        torque_pct_threshold=15.0,
        efficiency_abs_threshold=0.05,
        power_factor_abs_threshold=0.05,
        ratio_pct_threshold=15.0,
        curves_output_dir=None,
        curve_points=40,
        runtime_config=runtime,
    )
    assert "timing" in comparison_report
    assert comparison_report["timing"]["legacy_vs_deterministic_multiplier"] >= 0.0


def test_mode_runner_parser_defaults_are_iterative_and_quiet() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--input-master",
        "data/catalog_weg_w22_three_phase_eu_p30_p37_extracted.csv",
        "--mode",
        "smoke",
    ])

    assert args.compare_with_legacy is False
    assert args.legacy_fidelity == "iterative"
    assert args.log_level == "ERROR"


def test_mode_runner_parser_accepts_nameplate_solver() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--input-master",
        "data/catalog_weg_w22_three_phase_eu_p30_p37_extracted.csv",
        "--mode",
        "smoke",
        "--solver",
        "deterministic-nameplate",
    ])

    assert args.solver == "deterministic-nameplate"
