from __future__ import annotations

import argparse
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
    from scripts.guimaraes_model import estimate_parameters, predict_nominal
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from catalog_schema import (
        MotorRecord,
        enforce_grouping_constraints,
        group_records_by_id_or_tags,
        load_catalog_csv,
    )
    from guimaraes_model import estimate_parameters, predict_nominal

LOGGER = logging.getLogger(__name__)

DEFAULT_CURRENT_PCT_THRESHOLD = 15.0
DEFAULT_TORQUE_PCT_THRESHOLD = 15.0
DEFAULT_EFFICIENCY_ABS_THRESHOLD = 0.05
DEFAULT_POWER_FACTOR_ABS_THRESHOLD = 0.05


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
    pass_overall: bool


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
) -> MotorValidationResult:
    params = estimate_parameters(record)
    prediction = predict_nominal(record, params)

    err_current = _error_pct(prediction.rated_current_a, record.rated_current_a)
    err_torque = _error_pct(prediction.rated_torque_nm, record.rated_torque_nm)
    err_eff = abs(prediction.efficiency - record.efficiency)
    err_pf = abs(prediction.power_factor - record.power_factor)

    pass_current = err_current <= current_pct_threshold
    pass_torque = err_torque <= torque_pct_threshold
    pass_efficiency = err_eff <= efficiency_abs_threshold
    pass_power_factor = err_pf <= power_factor_abs_threshold
    pass_overall = pass_current and pass_torque and pass_efficiency and pass_power_factor

    LOGGER.info(
        "Validation %s -> Ierr=%.3f%% Terr=%.3f%% EtaErr=%.4f PFErr=%.4f Pass=%s",
        record.motor_id,
        err_current,
        err_torque,
        err_eff,
        err_pf,
        pass_overall,
    )

    return MotorValidationResult(
        group_name=group_name,
        motor_id=record.motor_id,
        predicted_current_a=prediction.rated_current_a,
        nominal_current_a=record.rated_current_a,
        error_current_pct=err_current,
        pass_current=pass_current,
        predicted_torque_nm=prediction.rated_torque_nm,
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
            "overall_pass_rate": 0.0,
        }
    return {
        "motors": total,
        "mean_abs_current_error_pct": sum(item.error_current_pct for item in results) / total,
        "mean_abs_torque_error_pct": sum(item.error_torque_pct for item in results) / total,
        "mean_abs_efficiency_error": sum(item.error_efficiency_abs for item in results) / total,
        "mean_abs_pf_error": sum(item.error_power_factor_abs for item in results) / total,
        "overall_pass_rate": sum(1 for item in results if item.pass_overall) / total,
    }


def run_validation(
    input_csv: Path,
    current_pct_threshold: float,
    torque_pct_threshold: float,
    efficiency_abs_threshold: float,
    power_factor_abs_threshold: float,
) -> dict[str, Any]:
    records = load_catalog_csv(input_csv)
    groups = group_records_by_id_or_tags(records)

    all_results: list[MotorValidationResult] = []
    summary_by_group: dict[str, dict[str, Any]] = {}

    for group_name in sorted(groups.keys()):
        group_records = groups[group_name]
        enforce_grouping_constraints(group_name, group_records)

        group_results: list[MotorValidationResult] = []
        for record in group_records:
            result = _validate_one_motor(
                group_name=group_name,
                record=record,
                current_pct_threshold=current_pct_threshold,
                torque_pct_threshold=torque_pct_threshold,
                efficiency_abs_threshold=efficiency_abs_threshold,
                power_factor_abs_threshold=power_factor_abs_threshold,
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
        },
        "results": [asdict(item) for item in all_results],
        "summary_by_group": summary_by_group,
    }


def _print_console_report(report: dict[str, Any]) -> None:
    thresholds = report["thresholds"]
    print("Thresholds:")
    print(f"  current_pct <= {thresholds['current_pct']}")
    print(f"  torque_pct <= {thresholds['torque_pct']}")
    print(f"  efficiency_abs <= {thresholds['efficiency_abs']}")
    print(f"  power_factor_abs <= {thresholds['power_factor_abs']}")
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
            f"overall={'PASS' if row['pass_overall'] else 'FAIL'}"
        )

    print()
    print("Summary by group:")
    for group_name, summary in report["summary_by_group"].items():
        print(
            f"  {group_name}: motors={summary['motors']}, "
            f"mean|I|%={summary['mean_abs_current_error_pct']:.3f}, "
            f"mean|T|%={summary['mean_abs_torque_error_pct']:.3f}, "
            f"mean|eta|={summary['mean_abs_efficiency_error']:.4f}, "
            f"mean|pf|={summary['mean_abs_pf_error']:.4f}, "
            f"pass_rate={summary['overall_pass_rate']:.3f}"
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Guimaraes-model nominal reproduction from CSV. "
            "Default thresholds: "
            f"current_pct={DEFAULT_CURRENT_PCT_THRESHOLD}, "
            f"torque_pct={DEFAULT_TORQUE_PCT_THRESHOLD}, "
            f"efficiency_abs={DEFAULT_EFFICIENCY_ABS_THRESHOLD}, "
            f"power_factor_abs={DEFAULT_POWER_FACTOR_ABS_THRESHOLD}."
        )
    )
    parser.add_argument("--input", required=True, help="Input CSV file with motor catalog rows")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--current-threshold-pct", type=float, default=DEFAULT_CURRENT_PCT_THRESHOLD)
    parser.add_argument("--torque-threshold-pct", type=float, default=DEFAULT_TORQUE_PCT_THRESHOLD)
    parser.add_argument("--efficiency-threshold-abs", type=float, default=DEFAULT_EFFICIENCY_ABS_THRESHOLD)
    parser.add_argument("--pf-threshold-abs", type=float, default=DEFAULT_POWER_FACTOR_ABS_THRESHOLD)
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
    )

    _print_console_report(report)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report written to: {out_path}")


if __name__ == "__main__":
    main()
