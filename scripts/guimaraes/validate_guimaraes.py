from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from scripts.guimaraes.catalog_schema import (
        MotorRecord,
        enforce_grouping_constraints,
        group_records_by_id_or_tags,
        load_catalog_csv,
    )
    from scripts.guimaraes.guimaraes_model import (
        LegacyRuntimeConfig,
        estimate_parameters,
        estimate_parameters_deterministic,
        estimate_parameters_deterministic_nameplate,
        estimate_parameters_legacy,
        evaluate_vs_slip,
        normalize_for_typical_curve,
        predict_characteristic_points,
        rated_slip,
        rotor_reactance_at_slip,
        rotor_resistance_at_slip,
        set_performance_hook,
    )
    from scripts.guimaraes.physical_guards import apply_physical_guards
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from catalog_schema import (
        MotorRecord,
        enforce_grouping_constraints,
        group_records_by_id_or_tags,
        load_catalog_csv,
    )
    from guimaraes_model import (
        LegacyRuntimeConfig,
        estimate_parameters,
        estimate_parameters_deterministic,
        estimate_parameters_deterministic_nameplate,
        estimate_parameters_legacy,
        evaluate_vs_slip,
        normalize_for_typical_curve,
        predict_characteristic_points,
        rated_slip,
        rotor_reactance_at_slip,
        rotor_resistance_at_slip,
        set_performance_hook,
    )
    from physical_guards import apply_physical_guards

LOGGER = logging.getLogger(__name__)

DEFAULT_CURRENT_PCT_THRESHOLD = 15.0
DEFAULT_TORQUE_PCT_THRESHOLD = 15.0
DEFAULT_EFFICIENCY_ABS_THRESHOLD = 0.05
DEFAULT_POWER_FACTOR_ABS_THRESHOLD = 0.05
DEFAULT_RATIO_PCT_THRESHOLD = 15.0
DEFAULT_CURVE_POINTS = 300
DEFAULT_SOLVER = "deterministic"
DEFAULT_TYPICAL_CURVE_POINTS = 40

_TYPICAL_PARAMETER_FIELDS: dict[str, str] = {
    "normalized_r1": "r1",
    "normalized_x1": "x1",
    "normalized_r2": "r2",
    "normalized_x2": "x2",
    "normalized_r20": "r20",
    "normalized_x20": "x20",
    "normalized_rm": "rm",
    "normalized_xm": "xm",
}


@dataclass(frozen=True)
class ValidationRuntimeConfig:
    legacy_solver: LegacyRuntimeConfig = field(default_factory=LegacyRuntimeConfig)


@dataclass
class TimingCollector:
    seconds_by_stage: dict[str, float] = field(default_factory=dict)
    counts_by_stage: dict[str, int] = field(default_factory=dict)

    def add(self, stage: str, elapsed_seconds: float, count: int = 1) -> None:
        self.seconds_by_stage[stage] = self.seconds_by_stage.get(stage, 0.0) + max(elapsed_seconds, 0.0)
        self.counts_by_stage[stage] = self.counts_by_stage.get(stage, 0) + max(count, 0)

    @contextmanager
    def track(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, time.perf_counter() - start)

    def to_report(self) -> dict[str, Any]:
        total_seconds = sum(self.seconds_by_stage.values())
        stages: dict[str, dict[str, float | int]] = {}
        for stage in sorted(self.seconds_by_stage.keys()):
            seconds = self.seconds_by_stage[stage]
            count = self.counts_by_stage.get(stage, 0)
            share_pct = (seconds * 100.0 / total_seconds) if total_seconds > 0.0 else 0.0
            stages[stage] = {
                "seconds": seconds,
                "count": count,
                "pct_of_timed_stages": share_pct,
            }
        return {
            "total_timed_seconds": total_seconds,
            "stages": stages,
        }


@dataclass(frozen=True)
class MotorValidationResult:
    solver: str
    group_name: str
    motor_id: str
    has_block3_targets: bool
    rated_power_kw: float
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
    nominal_breakdown_slip: float
    error_breakdown_slip_pct: float
    pass_breakdown_slip: bool
    normalized_r1: float
    normalized_r2: float
    normalized_x2: float
    normalized_r20: float
    normalized_x20: float
    normalized_rm: float
    normalized_xm: float
    normalized_x1: float
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


def _nominal_breakdown_slip(record: MotorRecord) -> float:
    s_r = rated_slip(record)
    mk = max(float(record.mk_mn or 1.0), 1.0)
    if mk <= 1.0:
        return min(max(3.0 * s_r, 0.02), 0.35)
    root = math.sqrt(max((mk ** 2) - 1.0, 0.0))
    return min(max(s_r * (mk + root), s_r + 1e-4), 0.98)


def _pick_solver(solver: str, runtime_config: ValidationRuntimeConfig):
    if solver == "deterministic":
        return (
            lambda record: estimate_parameters_deterministic(record)
            if record.has_block3_targets
            else estimate_parameters_deterministic_nameplate(record)
        )
    if solver == "deterministic-nameplate":
        return estimate_parameters_deterministic_nameplate
    if solver == "legacy":
        return lambda record: estimate_parameters_legacy(record, runtime_config=runtime_config.legacy_solver)
    return lambda record: estimate_parameters(record, legacy_config=runtime_config.legacy_solver)


def _summarize_guard_events(guard_events: list[dict[str, Any]]) -> dict[str, Any]:
    by_action = Counter(str(item.get("action", "unknown")) for item in guard_events)
    by_severity = Counter(str(item.get("severity", "unknown")) for item in guard_events)
    by_reason = Counter(str(item.get("reason", "unknown")) for item in guard_events)
    top_reasons = [
        {"reason": reason, "count": count}
        for reason, count in by_reason.most_common(10)
    ]
    return {
        "total_events": len(guard_events),
        "by_action": dict(by_action),
        "by_severity": dict(by_severity),
        "top_reasons": top_reasons,
    }


def _scale_steps(base: int, scale: float, *, minimum: int) -> int:
    return max(minimum, int(round(base * max(scale, 0.05))))


def resolve_validation_runtime_config(
    *,
    legacy_breakdown_samples: int | None = None,
    legacy_grid_scale: float | None = None,
) -> ValidationRuntimeConfig:
    base = LegacyRuntimeConfig()
    grid_scale = 1.0 if legacy_grid_scale is None else legacy_grid_scale

    return ValidationRuntimeConfig(
        legacy_solver=LegacyRuntimeConfig(
            breakdown_samples=max(20, legacy_breakdown_samples or base.breakdown_samples),
            beta_min_tenths=base.beta_min_tenths,
            beta_max_tenths=base.beta_max_tenths,
            alpha_min_tenths=base.alpha_min_tenths,
            alpha_max_tenths=base.alpha_max_tenths,
            r1_steps=_scale_steps(base.r1_steps, grid_scale, minimum=3),
            gr_steps=_scale_steps(base.gr_steps, grid_scale, minimum=3),
            rm_steps=_scale_steps(base.rm_steps, grid_scale, minimum=3),
            xm_steps=_scale_steps(base.xm_steps, grid_scale, minimum=3),
        )
    )


def _regression_metrics(x_values: list[float], y_values: list[float]) -> dict[str, float]:
    count = len(x_values)
    if count == 0:
        return {"count": 0.0, "slope": 0.0, "intercept": 0.0, "r2": 0.0}

    x_mean = sum(x_values) / count
    y_mean = sum(y_values) / count
    den = sum((x - x_mean) ** 2 for x in x_values)
    if den <= 1e-12:
        slope = 0.0
        intercept = y_mean
    else:
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        slope = num / den
        intercept = y_mean - (slope * x_mean)

    residuals = [(y - (slope * x + intercept)) for x, y in zip(x_values, y_values)]
    ss_res = sum(err * err for err in residuals)
    ss_tot = sum((y - y_mean) ** 2 for y in y_values)
    if ss_tot <= 1e-12:
        r2 = 1.0
    else:
        r2 = max(min(1.0 - (ss_res / ss_tot), 1.0), -1.0)

    return {
        "count": float(count),
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
    }


def _collect_regression_bundle(results: list[MotorValidationResult]) -> dict[str, dict[str, float]]:
    if not results:
        empty = _regression_metrics([], [])
        return {
            "ist_in": empty,
            "mst_mn": empty,
            "mk_mn": empty,
            "rated_torque": empty,
            "breakdown_slip": empty,
        }

    block3_rows = [row for row in results if row.has_block3_targets]

    return {
        "ist_in": _regression_metrics(
            [row.nominal_ist_in for row in block3_rows],
            [row.predicted_ist_in for row in block3_rows],
        ),
        "mst_mn": _regression_metrics(
            [row.nominal_mst_mn for row in block3_rows],
            [row.predicted_mst_mn for row in block3_rows],
        ),
        "mk_mn": _regression_metrics(
            [row.nominal_mk_mn for row in block3_rows],
            [row.predicted_mk_mn for row in block3_rows],
        ),
        "rated_torque": _regression_metrics(
            [row.nominal_torque_nm for row in results],
            [row.predicted_torque_nm for row in results],
        ),
        "breakdown_slip": _regression_metrics(
            [row.nominal_breakdown_slip for row in block3_rows],
            [row.predicted_breakdown_slip for row in block3_rows],
        ),
    }


def _collect_typical_curve_metrics(results: list[MotorValidationResult]) -> dict[str, dict[str, float]]:
    if not results:
        empty = _regression_metrics([], [])
        return {field_name: empty for field_name in _TYPICAL_PARAMETER_FIELDS}

    powers = [row.rated_power_kw for row in results]
    metrics: dict[str, dict[str, float]] = {}
    for field_name in _TYPICAL_PARAMETER_FIELDS:
        metrics[field_name] = _regression_metrics(
            powers,
            [float(getattr(row, field_name)) for row in results],
        )
    return metrics


def _fit_power_law(xs: list[float], ys: list[float]) -> dict[str, float | bool]:
    filtered: list[tuple[float, float]] = [
        (x, y) for x, y in zip(xs, ys) if x > 0.0 and y > 0.0 and math.isfinite(x) and math.isfinite(y)
    ]
    if len(filtered) < 2:
        return {
            "valid": False,
            "count": float(len(filtered)),
            "a": 0.0,
            "b": 0.0,
            "r2_log": 0.0,
        }

    log_x = [math.log(item[0]) for item in filtered]
    log_y = [math.log(item[1]) for item in filtered]
    count = len(log_x)
    x_mean = sum(log_x) / count
    y_mean = sum(log_y) / count
    den = sum((x - x_mean) ** 2 for x in log_x)
    if den <= 1e-12:
        slope = 0.0
        intercept = y_mean
    else:
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(log_x, log_y))
        slope = num / den
        intercept = y_mean - (slope * x_mean)

    predictions = [intercept + (slope * x) for x in log_x]
    y_mean = sum(log_y) / len(log_y)
    ss_res = sum((y - y_hat) ** 2 for y, y_hat in zip(log_y, predictions))
    ss_tot = sum((y - y_mean) ** 2 for y in log_y)
    r2_log = 1.0 if ss_tot <= 1e-12 else max(min(1.0 - (ss_res / ss_tot), 1.0), -1.0)

    return {
        "valid": True,
        "count": float(len(filtered)),
        "a": math.exp(intercept),
        "b": slope,
        "r2_log": r2_log,
    }


def _sample_power_law_curve(
    fit: dict[str, float | bool],
    min_power_kw: float,
    max_power_kw: float,
    points: int = DEFAULT_TYPICAL_CURVE_POINTS,
) -> list[dict[str, float]]:
    if not bool(fit.get("valid", False)):
        return []

    p_min = max(min_power_kw, 1e-6)
    p_max = max(max_power_kw, p_min)
    count = max(points, 2)
    a = float(fit["a"])
    b = float(fit["b"])

    if abs(p_max - p_min) <= 1e-12:
        y_val = a * (p_min ** b)
        return [{"rated_power_kw": p_min, "predicted_normalized": y_val}]

    samples: list[dict[str, float]] = []
    for idx in range(count):
        power_kw = p_min + ((p_max - p_min) * idx / (count - 1))
        samples.append(
            {
                "rated_power_kw": power_kw,
                "predicted_normalized": a * (power_kw ** b),
            }
        )
    return samples


def _collect_typical_power_laws(results: list[MotorValidationResult]) -> dict[str, dict[str, float | bool]]:
    if not results:
        return {
            field_name: {"valid": False, "count": 0.0, "a": 0.0, "b": 0.0, "r2_log": 0.0}
            for field_name in _TYPICAL_PARAMETER_FIELDS
        }

    powers = [row.rated_power_kw for row in results]
    fits: dict[str, dict[str, float | bool]] = {}
    for field_name in _TYPICAL_PARAMETER_FIELDS:
        values = [float(getattr(row, field_name)) for row in results]
        fits[field_name] = _fit_power_law(powers, values)
    return fits


def _collect_typical_power_law_curves(
    results: list[MotorValidationResult],
    fits: dict[str, dict[str, float | bool]],
) -> dict[str, list[dict[str, float]]]:
    if not results:
        return {field_name: [] for field_name in _TYPICAL_PARAMETER_FIELDS}

    powers = [row.rated_power_kw for row in results]
    min_power = min(powers)
    max_power = max(powers)
    curves: dict[str, list[dict[str, float]]] = {}
    for field_name, fit in fits.items():
        curves[field_name] = _sample_power_law_curve(fit, min_power, max_power)
    return curves


def _validate_one_motor(
    solver: str,
    solver_fn: Any,
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
    has_block3_targets = record.has_block3_targets

    params = solver_fn(record)
    points = predict_characteristic_points(record, params)
    prediction = points.nominal

    err_current = _error_pct(prediction.current_a, record.rated_current_a)
    err_torque = _error_pct(prediction.torque_nm, record.rated_torque_nm)
    err_eff = abs(prediction.efficiency - record.efficiency)
    err_pf = abs(prediction.power_factor - record.power_factor)

    predicted_ist_in = points.start.current_a / record.rated_current_a
    predicted_mst_mn = points.start.torque_nm / record.rated_torque_nm
    predicted_mk_mn = points.breakdown.torque_nm / record.rated_torque_nm

    r2_rated = rotor_resistance_at_slip(params, params.slip_rated)
    x2_rated = rotor_reactance_at_slip(params, params.slip_rated)

    if has_block3_targets:
        nominal_ist_in = float(record.ist_in)
        nominal_mst_mn = float(record.mst_mn)
        nominal_mk_mn = float(record.mk_mn)
        nominal_breakdown_slip = _nominal_breakdown_slip(record)

        err_ist_in = _error_pct(predicted_ist_in, nominal_ist_in)
        err_mst_mn = _error_pct(predicted_mst_mn, nominal_mst_mn)
        err_mk_mn = _error_pct(predicted_mk_mn, nominal_mk_mn)
        err_breakdown_slip = _error_pct(points.breakdown.slip, nominal_breakdown_slip)
    else:
        nominal_ist_in = predicted_ist_in
        nominal_mst_mn = predicted_mst_mn
        nominal_mk_mn = predicted_mk_mn
        nominal_breakdown_slip = points.breakdown.slip
        err_ist_in = 0.0
        err_mst_mn = 0.0
        err_mk_mn = 0.0
        err_breakdown_slip = 0.0
    nominal_error_pct = 0.5 * (err_current + err_torque)
    start_error_pct = 0.5 * (err_ist_in + err_mst_mn)
    breakdown_error_pct = err_mk_mn

    pass_current = err_current <= current_pct_threshold
    pass_torque = err_torque <= torque_pct_threshold
    pass_efficiency = err_eff <= efficiency_abs_threshold
    pass_power_factor = err_pf <= power_factor_abs_threshold
    pass_ist_in = (err_ist_in <= ratio_pct_threshold) if has_block3_targets else True
    pass_mst_mn = (err_mst_mn <= ratio_pct_threshold) if has_block3_targets else True
    pass_mk_mn = (err_mk_mn <= ratio_pct_threshold) if has_block3_targets else True
    pass_breakdown_slip = (err_breakdown_slip <= ratio_pct_threshold) if has_block3_targets else True
    pass_overall = (
        pass_current
        and pass_torque
        and pass_efficiency
        and pass_power_factor
        and pass_ist_in
        and pass_mst_mn
        and pass_mk_mn
        and pass_breakdown_slip
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
        "Validation[%s] %s -> Ierr=%.3f%% Terr=%.3f%% EtaErr=%.4f PFErr=%.4f Ist/In=%.3f%% Mst/Mn=%.3f%% Mk/Mn=%.3f%% sK=%.3f%% Pass=%s",
        solver,
        record.motor_id,
        err_current,
        err_torque,
        err_eff,
        err_pf,
        err_ist_in,
        err_mst_mn,
        err_mk_mn,
        err_breakdown_slip,
        pass_overall,
    )

    return MotorValidationResult(
        solver=solver,
        group_name=group_name,
        motor_id=record.motor_id,
        has_block3_targets=has_block3_targets,
        rated_power_kw=record.rated_power_w / 1000.0,
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
        nominal_breakdown_slip=nominal_breakdown_slip,
        error_breakdown_slip_pct=err_breakdown_slip,
        pass_breakdown_slip=pass_breakdown_slip,
        normalized_r1=normalize_for_typical_curve(params.r1_ohm, record, "r1"),
        normalized_r2=normalize_for_typical_curve(r2_rated, record, "r2"),
        normalized_x2=normalize_for_typical_curve(x2_rated, record, "x2"),
        normalized_r20=normalize_for_typical_curve(params.r2_base_ohm, record, "r20"),
        normalized_x20=normalize_for_typical_curve(params.x2_ohm, record, "x20"),
        normalized_rm=normalize_for_typical_curve(params.rm_ohm, record, "rm"),
        normalized_xm=normalize_for_typical_curve(params.xm_ohm, record, "xm"),
        normalized_x1=normalize_for_typical_curve(params.x1_ohm, record, "x1"),
        nominal_error_pct=nominal_error_pct,
        start_error_pct=start_error_pct,
        breakdown_error_pct=breakdown_error_pct,
        curve_csv_path=curve_csv_path,
        pass_overall=pass_overall,
    )


def _summarize_group(results: list[MotorValidationResult]) -> dict[str, Any]:
    total = len(results)
    block3_rows = [item for item in results if item.has_block3_targets]
    block3_total = len(block3_rows)
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
            "mean_abs_breakdown_slip_error_pct": 0.0,
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
        "mean_abs_ist_in_error_pct": (
            sum(item.error_ist_in_pct for item in block3_rows) / block3_total if block3_total else 0.0
        ),
        "mean_abs_mst_mn_error_pct": (
            sum(item.error_mst_mn_pct for item in block3_rows) / block3_total if block3_total else 0.0
        ),
        "mean_abs_mk_mn_error_pct": (
            sum(item.error_mk_mn_pct for item in block3_rows) / block3_total if block3_total else 0.0
        ),
        "mean_abs_breakdown_slip_error_pct": (
            sum(item.error_breakdown_slip_pct for item in block3_rows) / block3_total if block3_total else 0.0
        ),
        "mean_nominal_error_pct": sum(item.nominal_error_pct for item in results) / total,
        "mean_start_error_pct": (
            sum(item.start_error_pct for item in block3_rows) / block3_total if block3_total else 0.0
        ),
        "mean_breakdown_error_pct": (
            sum(item.breakdown_error_pct for item in block3_rows) / block3_total if block3_total else 0.0
        ),
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
    solver: str = DEFAULT_SOLVER,
    runtime_config: ValidationRuntimeConfig | None = None,
) -> dict[str, Any]:
    runtime = runtime_config or ValidationRuntimeConfig()
    timer = TimingCollector()

    def _performance_hook(stage: str, elapsed_seconds: float, count: int) -> None:
        timer.add(stage, elapsed_seconds, count)

    set_performance_hook(_performance_hook)
    overall_start = time.perf_counter()

    with timer.track("load_catalog_csv"):
        records = load_catalog_csv(input_csv)
    with timer.track("group_records"):
        groups = group_records_by_id_or_tags(records)

    solver_fn = _pick_solver(solver, runtime)

    all_results: list[MotorValidationResult] = []
    summary_by_group: dict[str, dict[str, Any]] = {}
    regression_by_group: dict[str, dict[str, dict[str, float]]] = {}
    typical_curve_metrics_by_group: dict[str, dict[str, dict[str, float]]] = {}
    typical_power_laws_by_group: dict[str, dict[str, dict[str, float | bool]]] = {}
    typical_power_law_curves_by_group: dict[str, dict[str, list[dict[str, float]]]] = {}
    skipped_motors: list[dict[str, str]] = []
    guard_events: list[dict[str, Any]] = []

    for group_name in sorted(groups.keys()):
        group_records = groups[group_name]
        with timer.track("group_constraints"):
            enforce_grouping_constraints(group_name, group_records)

        group_results: list[MotorValidationResult] = []
        for record in group_records:
            validated_record = record
            if record.has_block3_targets:
                with timer.track("physical_guards"):
                    guard_result = apply_physical_guards(record)
                guard_events.extend(asdict(item) for item in guard_result.events)
                if not guard_result.accepted or guard_result.record is None:
                    skipped_motors.append(
                        {
                            "group_name": group_name,
                            "motor_id": record.motor_id,
                            "reason": "rejected by physical guards",
                        }
                    )
                    continue
                validated_record = guard_result.record

            with timer.track("validate_motor"):
                result = _validate_one_motor(
                    solver=solver,
                    solver_fn=solver_fn,
                    group_name=group_name,
                    record=validated_record,
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

        with timer.track("group_summary"):
            summary_by_group[group_name] = _summarize_group(group_results)
            regression_by_group[group_name] = _collect_regression_bundle(group_results)
            typical_curve_metrics_by_group[group_name] = _collect_typical_curve_metrics(group_results)
            typical_power_laws_by_group[group_name] = _collect_typical_power_laws(group_results)
            typical_power_law_curves_by_group[group_name] = _collect_typical_power_law_curves(
                group_results,
                typical_power_laws_by_group[group_name],
            )

    rejected_count = sum(1 for item in skipped_motors if item["reason"] == "rejected by physical guards")
    corrected_count = sum(1 for item in guard_events if item["action"] == "auto_correct")
    guard_summary = _summarize_guard_events(guard_events)

    with timer.track("report_generation"):
        report = {
            "solver": solver,
            "thresholds": {
                "current_pct": current_pct_threshold,
                "torque_pct": torque_pct_threshold,
                "efficiency_abs": efficiency_abs_threshold,
                "power_factor_abs": power_factor_abs_threshold,
                "ratio_pct": ratio_pct_threshold,
            },
            "runtime_config": {
                "legacy_solver": {
                    "breakdown_samples": runtime.legacy_solver.breakdown_samples,
                    "beta_min_tenths": runtime.legacy_solver.beta_min_tenths,
                    "beta_max_tenths": runtime.legacy_solver.beta_max_tenths,
                    "alpha_min_tenths": runtime.legacy_solver.alpha_min_tenths,
                    "alpha_max_tenths": runtime.legacy_solver.alpha_max_tenths,
                    "r1_steps": runtime.legacy_solver.r1_steps,
                    "gr_steps": runtime.legacy_solver.gr_steps,
                    "rm_steps": runtime.legacy_solver.rm_steps,
                    "xm_steps": runtime.legacy_solver.xm_steps,
                }
            },
            "results": [asdict(item) for item in all_results],
            "summary_by_group": summary_by_group,
            "regression_by_group": regression_by_group,
            "regression_overall": _collect_regression_bundle(all_results),
            "typical_curve_metrics_by_group": typical_curve_metrics_by_group,
            "typical_power_laws_by_group": typical_power_laws_by_group,
            "typical_power_law_curves_by_group": typical_power_law_curves_by_group,
            "skipped": skipped_motors,
            "guard_audit": {
                "events": guard_events,
                "event_summary": guard_summary,
                "auto_correct_count": corrected_count,
                "reject_count": rejected_count,
            },
            "overall_pass": all(item.pass_overall for item in all_results) if all_results else False,
        }

    total_seconds = time.perf_counter() - overall_start
    report["timing"] = {
        **timer.to_report(),
        "total_runtime_seconds": total_seconds,
    }
    set_performance_hook(None)
    return report


def run_comparison_validation(
    input_csv: Path,
    current_pct_threshold: float,
    torque_pct_threshold: float,
    efficiency_abs_threshold: float,
    power_factor_abs_threshold: float,
    ratio_pct_threshold: float,
    curve_points: int,
    curves_output_dir: Path | None = None,
    runtime_config: ValidationRuntimeConfig | None = None,
) -> dict[str, Any]:
    comparison_start = time.perf_counter()

    deterministic_start = time.perf_counter()
    deterministic_report = run_validation(
        input_csv=input_csv,
        current_pct_threshold=current_pct_threshold,
        torque_pct_threshold=torque_pct_threshold,
        efficiency_abs_threshold=efficiency_abs_threshold,
        power_factor_abs_threshold=power_factor_abs_threshold,
        ratio_pct_threshold=ratio_pct_threshold,
        curves_output_dir=curves_output_dir,
        curve_points=curve_points,
        solver="deterministic",
        runtime_config=runtime_config,
    )
    deterministic_seconds = time.perf_counter() - deterministic_start

    legacy_start = time.perf_counter()
    legacy_report = run_validation(
        input_csv=input_csv,
        current_pct_threshold=current_pct_threshold,
        torque_pct_threshold=torque_pct_threshold,
        efficiency_abs_threshold=efficiency_abs_threshold,
        power_factor_abs_threshold=power_factor_abs_threshold,
        ratio_pct_threshold=ratio_pct_threshold,
        curves_output_dir=None,
        curve_points=curve_points,
        solver="legacy",
        runtime_config=runtime_config,
    )
    legacy_seconds = time.perf_counter() - legacy_start

    det_results = deterministic_report["results"]
    leg_results = legacy_report["results"]

    def _mean(rows: list[dict[str, Any]], key: str) -> float:
        if not rows:
            return 0.0
        return sum(float(row[key]) for row in rows) / len(rows)

    comparison = {
        "mean_errors": {
            "current_pct": {
                "deterministic": _mean(det_results, "error_current_pct"),
                "legacy": _mean(leg_results, "error_current_pct"),
            },
            "torque_pct": {
                "deterministic": _mean(det_results, "error_torque_pct"),
                "legacy": _mean(leg_results, "error_torque_pct"),
            },
            "efficiency_abs": {
                "deterministic": _mean(det_results, "error_efficiency_abs"),
                "legacy": _mean(leg_results, "error_efficiency_abs"),
            },
            "power_factor_abs": {
                "deterministic": _mean(det_results, "error_power_factor_abs"),
                "legacy": _mean(leg_results, "error_power_factor_abs"),
            },
        },
        "pass_rate": {
            "deterministic": _mean(det_results, "pass_overall"),
            "legacy": _mean(leg_results, "pass_overall"),
        },
        "regression_r2": {
            "deterministic": {
                key: deterministic_report["regression_overall"][key]["r2"]
                for key in deterministic_report["regression_overall"].keys()
            },
            "legacy": {
                key: legacy_report["regression_overall"][key]["r2"]
                for key in legacy_report["regression_overall"].keys()
            },
        },
    }

    total_seconds = time.perf_counter() - comparison_start
    return {
        "comparison": comparison,
        "timing": {
            "deterministic_seconds": deterministic_seconds,
            "legacy_seconds": legacy_seconds,
            "legacy_vs_deterministic_multiplier": (
                legacy_seconds / deterministic_seconds if deterministic_seconds > 0.0 else 0.0
            ),
            "total_runtime_seconds": total_seconds,
        },
        "deterministic": deterministic_report,
        "legacy": legacy_report,
    }


def _print_console_report(report: dict[str, Any], *, summary_only: bool = False) -> None:
    print(f"Solver: {report.get('solver', 'n/a')}")
    thresholds = report["thresholds"]
    print("Thresholds:")
    print(f"  current_pct <= {thresholds['current_pct']}")
    print(f"  torque_pct <= {thresholds['torque_pct']}")
    print(f"  efficiency_abs <= {thresholds['efficiency_abs']}")
    print(f"  power_factor_abs <= {thresholds['power_factor_abs']}")
    print(f"  ratio_pct <= {thresholds['ratio_pct']}")
    print()

    if not summary_only:
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
                f"smax={row['predicted_breakdown_slip']:.4f}/{row['nominal_breakdown_slip']:.4f} "
                f"({row['error_breakdown_slip_pct']:.2f}% {'PASS' if row['pass_breakdown_slip'] else 'FAIL'}) | "
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
            f"mean|smax|%={summary['mean_abs_breakdown_slip_error_pct']:.3f}, "
            f"meanNomErr%={summary['mean_nominal_error_pct']:.3f}, "
            f"meanStartErr%={summary['mean_start_error_pct']:.3f}, "
            f"meanBrErr%={summary['mean_breakdown_error_pct']:.3f}, "
            f"pass_rate={summary['overall_pass_rate']:.3f}"
        )

    print()
    print("Regression (slope, R2) overall:")
    for metric_name, values in report["regression_overall"].items():
        print(f"  {metric_name}: slope={values['slope']:.4f}, r2={values['r2']:.4f}, n={int(values['count'])}")

    print()
    print("Physical guards:")
    guard = report["guard_audit"]
    print(f"  auto_correct_count={guard['auto_correct_count']} reject_count={guard['reject_count']}")

    timing = report.get("timing")
    if isinstance(timing, dict):
        print()
        print("Timing:")
        total_runtime = float(timing.get("total_runtime_seconds", 0.0))
        print(f"  total_runtime_seconds={total_runtime:.3f}")
        stages = timing.get("stages")
        if isinstance(stages, dict):
            top = sorted(
                (
                    (name, values)
                    for name, values in stages.items()
                    if isinstance(values, dict)
                ),
                key=lambda item: float(item[1].get("seconds", 0.0)),
                reverse=True,
            )[:8]
            for stage_name, stage_values in top:
                print(
                    "  {}: {:.3f}s ({:.1f}%) count={}".format(
                        stage_name,
                        float(stage_values.get("seconds", 0.0)),
                        float(stage_values.get("pct_of_timed_stages", 0.0)),
                        int(stage_values.get("count", 0)),
                    )
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
    parser.add_argument(
        "--solver",
        default=DEFAULT_SOLVER,
        choices=["deterministic", "deterministic-nameplate", "legacy", "auto"],
        help="Estimator pipeline: deterministic (Eq.11-34), legacy (optimization), or auto",
    )
    parser.add_argument(
        "--compare-with-legacy",
        action="store_true",
        help="Generate deterministic-vs-legacy comparison report in one run",
    )
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
    parser.add_argument(
        "--legacy-breakdown-samples",
        type=int,
        default=None,
        help="Override legacy breakdown search samples (default: 600)",
    )
    parser.add_argument(
        "--legacy-grid-scale",
        type=float,
        default=None,
        help="Scale legacy solver grid-search steps (1.0=baseline fidelity)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only thresholds, group summary, skipped rows and overall PASS (omit per-motor lines).",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    runtime_config = resolve_validation_runtime_config(
        legacy_breakdown_samples=args.legacy_breakdown_samples,
        legacy_grid_scale=args.legacy_grid_scale,
    )

    if args.compare_with_legacy:
        report = run_comparison_validation(
            input_csv=Path(args.input),
            current_pct_threshold=args.current_threshold_pct,
            torque_pct_threshold=args.torque_threshold_pct,
            efficiency_abs_threshold=args.efficiency_threshold_abs,
            power_factor_abs_threshold=args.pf_threshold_abs,
            ratio_pct_threshold=args.ratio_threshold_pct,
            curves_output_dir=Path(args.export_curves_dir) if args.export_curves_dir else None,
            curve_points=args.curve_points,
            runtime_config=runtime_config,
        )

        det = report["deterministic"]
        leg = report["legacy"]
        print("Comparison summary:")
        print(
            "  pass_rate deterministic={:.3f} legacy={:.3f}".format(
                report["comparison"]["pass_rate"]["deterministic"],
                report["comparison"]["pass_rate"]["legacy"],
            )
        )
        print(
            "  mean|I| deterministic={:.3f} legacy={:.3f}".format(
                report["comparison"]["mean_errors"]["current_pct"]["deterministic"],
                report["comparison"]["mean_errors"]["current_pct"]["legacy"],
            )
        )
        print("\nDeterministic:")
        _print_console_report(det, summary_only=args.summary_only)
        print("\nLegacy:")
        _print_console_report(leg, summary_only=True)

        if args.output:
            out_path = Path(args.output)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nJSON report written to: {out_path}")
        return

    report = run_validation(
        input_csv=Path(args.input),
        current_pct_threshold=args.current_threshold_pct,
        torque_pct_threshold=args.torque_threshold_pct,
        efficiency_abs_threshold=args.efficiency_threshold_abs,
        power_factor_abs_threshold=args.pf_threshold_abs,
        ratio_pct_threshold=args.ratio_threshold_pct,
        curves_output_dir=Path(args.export_curves_dir) if args.export_curves_dir else None,
        curve_points=args.curve_points,
        solver=args.solver,
        runtime_config=runtime_config,
    )

    _print_console_report(report, summary_only=args.summary_only)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report written to: {out_path}")


if __name__ == "__main__":
    main()
