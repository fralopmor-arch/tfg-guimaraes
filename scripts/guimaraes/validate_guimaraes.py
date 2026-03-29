from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from scripts.catalog_schema import (
        MotorRecord,
        enforce_grouping_constraints,
        group_records_by_id_or_tags,
        load_catalog_csv,
    )
    from scripts.guimaraes_model import (
        estimate_parameters,
        evaluate_vs_slip,
        predict_characteristic_points,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from catalog_schema import (
        MotorRecord,
        enforce_grouping_constraints,
        group_records_by_id_or_tags,
        load_catalog_csv,
    )
    from guimaraes_model import estimate_parameters, evaluate_vs_slip, predict_characteristic_points

LOGGER = logging.getLogger(__name__)

DEFAULT_CURRENT_PCT_THRESHOLD = 15.0
DEFAULT_TORQUE_PCT_THRESHOLD = 15.0
DEFAULT_EFFICIENCY_ABS_THRESHOLD = 0.05
DEFAULT_POWER_FACTOR_ABS_THRESHOLD = 0.05
DEFAULT_RATIO_PCT_THRESHOLD = 15.0
DEFAULT_CURVE_POINTS = 300


@dataclass(frozen=True)
class MotorValidationResult:
    group_name: str
    motor_id: str
    predicted_current_a: float
    nominal_current_a: float
    error_current_pct: float
    pass_current: bool
    predicted_torque_nm: float
    nominal_torque_nm: float
    error_torque_pct: float
    pass_torque: bool
    predicted_efficiency: float
    nominal_efficiency: float
    error_efficiency_abs: float
    pass_efficiency: bool
    predicted_power_factor: float
    nominal_power_factor: float
    error_power_factor_abs: float
    pass_power_factor: bool
    predicted_ist_in: float
    nominal_ist_in: float
    error_ist_in_pct: float
    pass_ist_in: bool
    predicted_mst_mn: float
    nominal_mst_mn: float
    error_mst_mn_pct: float
    pass_mst_mn: bool
    predicted_mk_mn: float
    nominal_mk_mn: float
    error_mk_mn_pct: float
    pass_mk_mn: bool
    predicted_breakdown_slip: float
    nominal_error_pct: float
    start_error_pct: float
    breakdown_error_pct: float
    curve_csv_path: str | None
    pass_overall: bool


def _sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _build_curve_rows(record: MotorRecord, params: Any, curve_points: int) -> list[dict[str, float]]:
    points = max(curve_points, 10)
    slips = [0.001 + ((1.0 - 0.001) * idx / (points - 1)) for idx in range(points)]
    curve = evaluate_vs_slip(record, params, slips)

    rows: list[dict[str, float]] = []
    for point in curve:
        absorbed_power_w = (
            (3.0 ** 0.5) * record.rated_voltage_v * point.current_a * point.power_factor
        )
        rows.append(
            {
                "slip": point.slip,
                "torque_nm": point.torque_nm,
                "current_a": point.current_a,
                "absorbed_power_w": absorbed_power_w,
                "efficiency": point.efficiency,
                "power_factor": point.power_factor,
            }
        )
    return rows


def _export_curves_csv(
    curves_output_dir: Path,
    group_name: str,
    record: MotorRecord,
    params: Any,
    curve_points: int,
) -> Path:
    curves_output_dir.mkdir(parents=True, exist_ok=True)
    safe_group = _sanitize_name(group_name)
    safe_motor = _sanitize_name(record.motor_id)
    output_path = curves_output_dir / f"{safe_group}__{safe_motor}_curves.csv"

    rows = _build_curve_rows(record, params, curve_points)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "slip",
                "torque_nm",
                "current_a",
                "absorbed_power_w",
                "efficiency",
                "power_factor",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def _error_pct(predicted: float, nominal: float) -> float:
    if nominal == 0:
        return 0.0 if predicted == 0 else 100.0
    return abs(predicted - nominal) * 100.0 / abs(nominal)


def _validate_one_motor(
    group_name: str,
    record: MotorRecord,
    current_pct_threshold: float,
    torque_pct_threshold: float,
    efficiency_abs_threshold: float,
    power_factor_abs_threshold: float,
    ratio_pct_threshold: float,
    curves_output_dir: Path | None,
    curve_points: int,
) -> MotorValidationResult:
    if not record.has_block3_targets:
        raise ValueError(f"Motor {record.motor_id} does not include required Block III ratios")

    params = estimate_parameters(record)
    points = predict_characteristic_points(record, params)
    prediction = points.nominal

    err_current = _error_pct(prediction.current_a, record.rated_current_a)
    err_torque = _error_pct(prediction.torque_nm, record.rated_torque_nm)
    err_eff = abs(prediction.efficiency - record.efficiency)
    err_pf = abs(prediction.power_factor - record.power_factor)

    predicted_ist_in = points.start.current_a / record.rated_current_a
    predicted_mst_mn = points.start.torque_nm / record.rated_torque_nm
    predicted_mk_mn = points.breakdown.torque_nm / record.rated_torque_nm

    nominal_ist_in = float(record.ist_in)
    nominal_mst_mn = float(record.mst_mn)
    nominal_mk_mn = float(record.mk_mn)

    err_ist_in = _error_pct(predicted_ist_in, nominal_ist_in)
    err_mst_mn = _error_pct(predicted_mst_mn, nominal_mst_mn)
    err_mk_mn = _error_pct(predicted_mk_mn, nominal_mk_mn)
    nominal_error_pct = 0.5 * (err_current + err_torque)
    start_error_pct = 0.5 * (err_ist_in + err_mst_mn)
    breakdown_error_pct = err_mk_mn

    pass_current = err_current <= current_pct_threshold
    pass_torque = err_torque <= torque_pct_threshold
    pass_efficiency = err_eff <= efficiency_abs_threshold
    pass_power_factor = err_pf <= power_factor_abs_threshold
    pass_ist_in = err_ist_in <= ratio_pct_threshold
    pass_mst_mn = err_mst_mn <= ratio_pct_threshold
    pass_mk_mn = err_mk_mn <= ratio_pct_threshold
    pass_overall = (
        pass_current
        and pass_torque
        and pass_efficiency
        and pass_power_factor
        and pass_ist_in
        and pass_mst_mn
        and pass_mk_mn
    )

    curve_csv_path: str | None = None
    if curves_output_dir is not None:
        curve_path = _export_curves_csv(
            curves_output_dir=curves_output_dir,
            group_name=group_name,
            record=record,
            params=params,
            curve_points=curve_points,
        )
        curve_csv_path = curve_path.as_posix()

    LOGGER.info(
        "Validation %s -> Ierr=%.3f%% Terr=%.3f%% EtaErr=%.4f PFErr=%.4f Ist/In=%.3f%% Mst/Mn=%.3f%% Mk/Mn=%.3f%% Pass=%s",
        record.motor_id,
        err_current,
        err_torque,
        err_eff,
        err_pf,
        err_ist_in,
        err_mst_mn,
        err_mk_mn,
        pass_overall,
    )

    return MotorValidationResult(
        group_name=group_name,
        motor_id=record.motor_id,
        predicted_current_a=prediction.current_a,
        nominal_current_a=record.rated_current_a,
        error_current_pct=err_current,
        pass_current=pass_current,
        predicted_torque_nm=prediction.torque_nm,
        nominal_torque_nm=record.rated_torque_nm,
        error_torque_pct=err_torque,
        pass_torque=pass_torque,
        predicted_efficiency=prediction.efficiency,
        nominal_efficiency=record.efficiency,
        error_efficiency_abs=err_eff,
        pass_efficiency=pass_efficiency,
        predicted_power_factor=prediction.power_factor,
        nominal_power_factor=record.power_factor,
        error_power_factor_abs=err_pf,
        pass_power_factor=pass_power_factor,
        predicted_ist_in=predicted_ist_in,
        nominal_ist_in=nominal_ist_in,
        error_ist_in_pct=err_ist_in,
        pass_ist_in=pass_ist_in,
        predicted_mst_mn=predicted_mst_mn,
        nominal_mst_mn=nominal_mst_mn,
        error_mst_mn_pct=err_mst_mn,
        pass_mst_mn=pass_mst_mn,
        predicted_mk_mn=predicted_mk_mn,
        nominal_mk_mn=nominal_mk_mn,
        error_mk_mn_pct=err_mk_mn,
        pass_mk_mn=pass_mk_mn,
        predicted_breakdown_slip=points.breakdown.slip,
        nominal_error_pct=nominal_error_pct,
        start_error_pct=start_error_pct,
        breakdown_error_pct=breakdown_error_pct,
        curve_csv_path=curve_csv_path,
        pass_overall=pass_overall,
    )


def _summarize_group(results: list[MotorValidationResult]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "motors": 0,
            "mean_abs_current_error_pct": 0.0,
            "mean_abs_torque_error_pct": 0.0,
            "mean_abs_efficiency_error": 0.0,
            "mean_abs_pf_error": 0.0,
            "mean_abs_ist_in_error_pct": 0.0,
            "mean_abs_mst_mn_error_pct": 0.0,
            "mean_abs_mk_mn_error_pct": 0.0,
            "mean_nominal_error_pct": 0.0,
            "mean_start_error_pct": 0.0,
            "mean_breakdown_error_pct": 0.0,
            "overall_pass_rate": 0.0,
        }
    return {
        "motors": total,
        "mean_abs_current_error_pct": sum(item.error_current_pct for item in results) / total,
        "mean_abs_torque_error_pct": sum(item.error_torque_pct for item in results) / total,
        "mean_abs_efficiency_error": sum(item.error_efficiency_abs for item in results) / total,
        "mean_abs_pf_error": sum(item.error_power_factor_abs for item in results) / total,
        "mean_abs_ist_in_error_pct": sum(item.error_ist_in_pct for item in results) / total,
        "mean_abs_mst_mn_error_pct": sum(item.error_mst_mn_pct for item in results) / total,
        "mean_abs_mk_mn_error_pct": sum(item.error_mk_mn_pct for item in results) / total,
        "mean_nominal_error_pct": sum(item.nominal_error_pct for item in results) / total,
        "mean_start_error_pct": sum(item.start_error_pct for item in results) / total,
        "mean_breakdown_error_pct": sum(item.breakdown_error_pct for item in results) / total,
        "overall_pass_rate": sum(1 for item in results if item.pass_overall) / total,
    }


def run_validation(
    input_csv: Path,
    current_pct_threshold: float,
    torque_pct_threshold: float,
    efficiency_abs_threshold: float,
    power_factor_abs_threshold: float,
    ratio_pct_threshold: float,
    curves_output_dir: Path | None,
    curve_points: int,
) -> dict[str, Any]:
    records = load_catalog_csv(input_csv)
    groups = group_records_by_id_or_tags(records)

    all_results: list[MotorValidationResult] = []
    summary_by_group: dict[str, dict[str, Any]] = {}
    skipped_motors: list[dict[str, str]] = []

    for group_name in sorted(groups.keys()):
        group_records = groups[group_name]
        enforce_grouping_constraints(group_name, group_records)

        group_results: list[MotorValidationResult] = []
        for record in group_records:
            if not record.has_block3_targets:
                skipped_motors.append(
                    {
                        "group_name": group_name,
                        "motor_id": record.motor_id,
                        "reason": "missing Block III columns (Ist_In, Mst_Mn, Mk_Mn)",
                    }
                )
                continue
            result = _validate_one_motor(
                group_name=group_name,
                record=record,
                current_pct_threshold=current_pct_threshold,
                torque_pct_threshold=torque_pct_threshold,
                efficiency_abs_threshold=efficiency_abs_threshold,
                power_factor_abs_threshold=power_factor_abs_threshold,
                ratio_pct_threshold=ratio_pct_threshold,
                curves_output_dir=curves_output_dir,
                curve_points=curve_points,
            )
            group_results.append(result)
            all_results.append(result)

        summary_by_group[group_name] = _summarize_group(group_results)

    return {
        "thresholds": {
            "current_pct": current_pct_threshold,
            "torque_pct": torque_pct_threshold,
            "efficiency_abs": efficiency_abs_threshold,
            "power_factor_abs": power_factor_abs_threshold,
            "ratio_pct": ratio_pct_threshold,
        },
        "results": [asdict(item) for item in all_results],
        "summary_by_group": summary_by_group,
        "skipped": skipped_motors,
        "overall_pass": all(item.pass_overall for item in all_results) if all_results else False,
    }


def _print_console_report(report: dict[str, Any]) -> None:
    thresholds = report["thresholds"]
    print("Thresholds:")
    print(f"  current_pct <= {thresholds['current_pct']}")
    print(f"  torque_pct <= {thresholds['torque_pct']}")
    print(f"  efficiency_abs <= {thresholds['efficiency_abs']}")
    print(f"  power_factor_abs <= {thresholds['power_factor_abs']}")
    print(f"  ratio_pct <= {thresholds['ratio_pct']}")
    print()

    print("Per-motor results:")
    for row in report["results"]:
        print(
            f"  [{row['group_name']}] {row['motor_id']} | "
            f"I={row['predicted_current_a']:.4f}/{row['nominal_current_a']:.4f}A "
            f"({row['error_current_pct']:.2f}% {'PASS' if row['pass_current'] else 'FAIL'}) | "
            f"T={row['predicted_torque_nm']:.4f}/{row['nominal_torque_nm']:.4f}Nm "
            f"({row['error_torque_pct']:.2f}% {'PASS' if row['pass_torque'] else 'FAIL'}) | "
            f"eta={row['predicted_efficiency']:.4f}/{row['nominal_efficiency']:.4f} "
            f"({row['error_efficiency_abs']:.4f} {'PASS' if row['pass_efficiency'] else 'FAIL'}) | "
            f"pf={row['predicted_power_factor']:.4f}/{row['nominal_power_factor']:.4f} "
            f"({row['error_power_factor_abs']:.4f} {'PASS' if row['pass_power_factor'] else 'FAIL'}) | "
            f"Ist/In={row['predicted_ist_in']:.4f}/{row['nominal_ist_in']:.4f} "
            f"({row['error_ist_in_pct']:.2f}% {'PASS' if row['pass_ist_in'] else 'FAIL'}) | "
            f"Mst/Mn={row['predicted_mst_mn']:.4f}/{row['nominal_mst_mn']:.4f} "
            f"({row['error_mst_mn_pct']:.2f}% {'PASS' if row['pass_mst_mn'] else 'FAIL'}) | "
            f"Mk/Mn={row['predicted_mk_mn']:.4f}/{row['nominal_mk_mn']:.4f} "
            f"({row['error_mk_mn_pct']:.2f}% {'PASS' if row['pass_mk_mn'] else 'FAIL'}) | "
            f"smax={row['predicted_breakdown_slip']:.4f} | "
            f"NomErr={row['nominal_error_pct']:.2f}% StartErr={row['start_error_pct']:.2f}% BrErr={row['breakdown_error_pct']:.2f}% | "
            f"overall={'PASS' if row['pass_overall'] else 'FAIL'}"
        )
        if row.get("curve_csv_path"):
            print(f"      curves: {row['curve_csv_path']}")

    print()
    print("Summary by group:")
    for group_name, summary in report["summary_by_group"].items():
        print(
            f"  {group_name}: motors={summary['motors']}, "
            f"mean|I|%={summary['mean_abs_current_error_pct']:.3f}, "
            f"mean|T|%={summary['mean_abs_torque_error_pct']:.3f}, "
            f"mean|eta|={summary['mean_abs_efficiency_error']:.4f}, "
            f"mean|pf|={summary['mean_abs_pf_error']:.4f}, "
            f"mean|Ist/In|%={summary['mean_abs_ist_in_error_pct']:.3f}, "
            f"mean|Mst/Mn|%={summary['mean_abs_mst_mn_error_pct']:.3f}, "
            f"mean|Mk/Mn|%={summary['mean_abs_mk_mn_error_pct']:.3f}, "
            f"meanNomErr%={summary['mean_nominal_error_pct']:.3f}, "
            f"meanStartErr%={summary['mean_start_error_pct']:.3f}, "
            f"meanBrErr%={summary['mean_breakdown_error_pct']:.3f}, "
            f"pass_rate={summary['overall_pass_rate']:.3f}"
        )

    if report["skipped"]:
        print()
        print("Skipped motors:")
        for row in report["skipped"]:
            print(f"  [{row['group_name']}] {row['motor_id']} -> {row['reason']}")

    print()
    print(f"Overall PASS: {report['overall_pass']}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Guimaraes-model nominal reproduction from CSV. "
            "Default thresholds: "
            f"current_pct={DEFAULT_CURRENT_PCT_THRESHOLD}, "
            f"torque_pct={DEFAULT_TORQUE_PCT_THRESHOLD}, "
            f"efficiency_abs={DEFAULT_EFFICIENCY_ABS_THRESHOLD}, "
            f"power_factor_abs={DEFAULT_POWER_FACTOR_ABS_THRESHOLD}, "
            f"ratio_pct={DEFAULT_RATIO_PCT_THRESHOLD}."
        )
    )
    parser.add_argument("--input", required=True, help="Input CSV file with motor catalog rows")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--current-threshold-pct", type=float, default=DEFAULT_CURRENT_PCT_THRESHOLD)
    parser.add_argument("--torque-threshold-pct", type=float, default=DEFAULT_TORQUE_PCT_THRESHOLD)
    parser.add_argument("--efficiency-threshold-abs", type=float, default=DEFAULT_EFFICIENCY_ABS_THRESHOLD)
    parser.add_argument("--pf-threshold-abs", type=float, default=DEFAULT_POWER_FACTOR_ABS_THRESHOLD)
    parser.add_argument("--ratio-threshold-pct", type=float, default=DEFAULT_RATIO_PCT_THRESHOLD)
    parser.add_argument(
        "--export-curves-dir",
        help="Optional output directory to export per-motor curve CSV files",
    )
    parser.add_argument(
        "--curve-points",
        type=int,
        default=DEFAULT_CURVE_POINTS,
        help=f"Number of slip samples in [0.001, 1.0] for curve export (default: {DEFAULT_CURVE_POINTS})",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")

    report = run_validation(
        input_csv=Path(args.input),
        current_pct_threshold=args.current_threshold_pct,
        torque_pct_threshold=args.torque_threshold_pct,
        efficiency_abs_threshold=args.efficiency_threshold_abs,
        power_factor_abs_threshold=args.pf_threshold_abs,
        ratio_pct_threshold=args.ratio_threshold_pct,
        curves_output_dir=Path(args.export_curves_dir) if args.export_curves_dir else None,
        curve_points=args.curve_points,
    )

    _print_console_report(report)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report written to: {out_path}")


if __name__ == "__main__":
    main()
