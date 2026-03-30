from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from scripts.guimaraes.catalog_schema import MotorRecord, enforce_grouping_constraints
from scripts.guimaraes.guimaraes_model import (
    _block3_efficiency_pf_points,
    _linear_fit,
    _slip_from_load_fraction,
    estimate_parameters_deterministic,
    estimate_parameters_deterministic_nameplate,
    predict_characteristic_points,
    rated_slip,
    rotor_resistance_at_slip,
    solve_deterministic_blocks,
)
from scripts.guimaraes.physical_guards import apply_physical_guards
from scripts.guimaraes.validate_guimaraes import run_validation


def _sample_record(**overrides: object) -> MotorRecord:
    base = {
        "motor_id": "TEST_MOTOR",
        "rated_voltage_v": 400.0,
        "rated_power_w": 11000.0,
        "frequency_hz": 50.0,
        "poles": 4,
        "pole_pairs": None,
        "rated_current_a": 21.0,
        "rated_torque_nm": 72.0,
        "efficiency": 0.91,
        "power_factor": 0.86,
        "efficiency_class": "IE3",
        "starting_torque_category": "Normal torque (IEC design N)",
        "manufacturer": "WEG",
        "group_id": None,
        "ist_in": 7.0,
        "mst_mn": 2.2,
        "mk_mn": 2.8,
        "eff_50": 0.88,
        "eff_75": 0.90,
        "eff_100": 0.91,
        "pf_50": 0.70,
        "pf_75": 0.80,
        "pf_100": 0.86,
    }
    base.update(overrides)
    return MotorRecord(**base)


def test_group_constraints_include_frequency_and_pole_pairs() -> None:
    row_a = _sample_record(motor_id="A", frequency_hz=50.0, poles=4)
    row_b = _sample_record(motor_id="B", frequency_hz=60.0, poles=4)

    with pytest.raises(ValueError):
        enforce_grouping_constraints("mixed_frequency", [row_a, row_b])


def test_physical_guards_auto_correct_and_reject_paths() -> None:
    corrected = _sample_record(mk_mn=1.5, mst_mn=1.8, eff_75=0.84, pf_75=0.68)
    result_ok = apply_physical_guards(corrected)
    assert result_ok.accepted
    assert result_ok.record is not None
    assert result_ok.record.mk_mn >= result_ok.record.mst_mn
    assert result_ok.record.eff_75 >= result_ok.record.eff_50
    assert any(event.action == "auto_correct" for event in result_ok.events)

    rejected = _sample_record(rated_torque_nm=1.0)
    result_bad = apply_physical_guards(rejected)
    assert not result_bad.accepted
    assert any(event.action == "reject" for event in result_bad.events)


def test_deterministic_block_anchors_are_consistent() -> None:
    state = solve_deterministic_blocks(_sample_record())

    assert state.r1_ohm > 0.0
    assert state.r2_start_ohm > 0.0
    assert state.x2_start_ohm > 0.0
    assert state.rm_ohm > 0.0
    assert state.xm_ohm > 0.0

    r2_rated_from_law = state.r2_start_ohm * (
        math.exp(state.gr * ((1.0 - state.slip_rated) ** 0.5))
    )
    assert r2_rated_from_law == pytest.approx(state.r2_rated_ohm, rel=1e-6)


def test_eq15_rotor_resistance_regression_drives_r2_rated() -> None:
    record = _sample_record()
    state = solve_deterministic_blocks(record)

    slip_r = rated_slip(record)
    xs_r2: list[float] = []
    ys_r2: list[float] = []
    for load_fraction, eta, pf in _block3_efficiency_pf_points(record):
        eta_safe = max(min(eta, 0.9995), 1e-4)
        pf_safe = max(min(pf, 0.9995), 0.05)
        p2 = record.rated_power_w * load_fraction
        pin = p2 / eta_safe
        i1 = pin / max((math.sqrt(3.0) * record.rated_voltage_v * pf_safe), 1e-9)
        slip_i = _slip_from_load_fraction(slip_r, load_fraction)
        xs_r2.append(3.0 * (i1 ** 2))
        ys_r2.append((p2 * slip_i) / max(1.0 - slip_i, 1e-9))

    r2_fit, _ = _linear_fit(xs_r2, ys_r2)

    assert r2_fit > 0.0
    assert r2_fit <= state.r2_start_ohm
    assert state.r2_rated_ohm == pytest.approx(r2_fit, rel=1e-6)


def test_rotor_resistance_is_monotonic_in_deterministic_block() -> None:
    record = _sample_record()
    params = estimate_parameters_deterministic(record)
    points = predict_characteristic_points(record, params)

    r_start = rotor_resistance_at_slip(params, 1.0)
    r_breakdown = rotor_resistance_at_slip(params, points.breakdown.slip)
    r_rated = rotor_resistance_at_slip(params, params.slip_rated)

    assert r_start >= r_breakdown
    assert r_breakdown >= r_rated


def test_end_to_end_smoke_report_contains_regressions(tmp_path: Path) -> None:
    src = Path("data/catalog_weg_w21_aluminium_multimounting_p9_p21_extracted.csv")
    out_csv = tmp_path / "sample.csv"

    with src.open("r", encoding="utf-8-sig", newline="") as handle_in:
        reader = csv.DictReader(handle_in)
        rows = list(reader)
        assert len(rows) >= 3
        fieldnames = list(reader.fieldnames or [])

    with out_csv.open("w", encoding="utf-8-sig", newline="") as handle_out:
        writer = csv.DictWriter(handle_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows[:3]:
            writer.writerow(row)

    report = run_validation(
        input_csv=out_csv,
        current_pct_threshold=15.0,
        torque_pct_threshold=15.0,
        efficiency_abs_threshold=0.05,
        power_factor_abs_threshold=0.05,
        ratio_pct_threshold=15.0,
        curves_output_dir=None,
        curve_points=40,
        solver="deterministic",
    )

    assert "regression_overall" in report
    assert "ist_in" in report["regression_overall"]
    assert "breakdown_slip" in report["regression_overall"]
    assert "guard_audit" in report


def test_nameplate_only_mode_validates_without_block3(tmp_path: Path) -> None:
    out_csv = tmp_path / "nameplate_only.csv"
    fieldnames = [
        "motor_id",
        "rated_voltage_v",
        "rated_power_w",
        "frequency_hz",
        "poles",
        "rated_current_a",
        "rated_torque_nm",
        "efficiency",
        "power_factor",
        "efficiency_class",
        "starting_torque_category",
        "manufacturer",
    ]

    with out_csv.open("w", encoding="utf-8-sig", newline="") as handle_out:
        writer = csv.DictWriter(handle_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "motor_id": "NP_001",
                "rated_voltage_v": 400.0,
                "rated_power_w": 11000.0,
                "frequency_hz": 50.0,
                "poles": 4,
                "rated_current_a": 21.0,
                "rated_torque_nm": 72.0,
                "efficiency": 0.91,
                "power_factor": 0.86,
                "efficiency_class": "IE3",
                "starting_torque_category": "Normal torque (IEC design N)",
                "manufacturer": "WEG",
            }
        )

    report = run_validation(
        input_csv=out_csv,
        current_pct_threshold=15.0,
        torque_pct_threshold=15.0,
        efficiency_abs_threshold=0.05,
        power_factor_abs_threshold=0.05,
        ratio_pct_threshold=15.0,
        curves_output_dir=None,
        curve_points=40,
        solver="deterministic-nameplate",
    )

    assert report["skipped"] == []
    assert len(report["results"]) == 1
    row = report["results"][0]
    assert row["has_block3_targets"] is False
    assert row["pass_overall"] in {True, False}


def test_deterministic_auto_routes_nameplate_without_block3(tmp_path: Path) -> None:
    out_csv = tmp_path / "nameplate_auto.csv"
    fieldnames = [
        "motor_id",
        "rated_voltage_v",
        "rated_power_w",
        "frequency_hz",
        "poles",
        "rated_current_a",
        "rated_torque_nm",
        "efficiency",
        "power_factor",
        "efficiency_class",
        "starting_torque_category",
        "manufacturer",
    ]

    with out_csv.open("w", encoding="utf-8-sig", newline="") as handle_out:
        writer = csv.DictWriter(handle_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "motor_id": "NP_AUTO_001",
                "rated_voltage_v": 400.0,
                "rated_power_w": 11000.0,
                "frequency_hz": 50.0,
                "poles": 4,
                "rated_current_a": 21.0,
                "rated_torque_nm": 72.0,
                "efficiency": 0.91,
                "power_factor": 0.86,
                "efficiency_class": "IE3",
                "starting_torque_category": "Normal torque (IEC design N)",
                "manufacturer": "WEG",
            }
        )

    report = run_validation(
        input_csv=out_csv,
        current_pct_threshold=15.0,
        torque_pct_threshold=15.0,
        efficiency_abs_threshold=0.05,
        power_factor_abs_threshold=0.05,
        ratio_pct_threshold=15.0,
        curves_output_dir=None,
        curve_points=40,
        solver="deterministic",
    )

    assert report["skipped"] == []
    assert len(report["results"]) == 1
    assert report["results"][0]["has_block3_targets"] is False
    assert "solver_deterministic_nameplate" in report["timing"]["stages"]


def test_nameplate_solver_keeps_rotor_resistance_monotonic() -> None:
    record = _sample_record(
        ist_in=None,
        mst_mn=None,
        mk_mn=None,
        eff_50=None,
        eff_75=None,
        eff_100=None,
        pf_50=None,
        pf_75=None,
        pf_100=None,
    )
    params = estimate_parameters_deterministic_nameplate(record)
    points = predict_characteristic_points(record, params)

    r_start = rotor_resistance_at_slip(params, 1.0)
    r_breakdown = rotor_resistance_at_slip(params, points.breakdown.slip)
    r_rated = rotor_resistance_at_slip(params, params.slip_rated)

    assert r_start >= r_breakdown
    assert r_breakdown >= r_rated


def test_typical_model_exports_include_power_laws_and_curves(tmp_path: Path) -> None:
    src = Path("data/catalog_weg_w21_aluminium_multimounting_p9_p21_extracted.csv")
    out_csv = tmp_path / "sample_typical.csv"

    with src.open("r", encoding="utf-8-sig", newline="") as handle_in:
        reader = csv.DictReader(handle_in)
        rows = list(reader)
        assert len(rows) >= 4
        fieldnames = list(reader.fieldnames or [])

    with out_csv.open("w", encoding="utf-8-sig", newline="") as handle_out:
        writer = csv.DictWriter(handle_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows[:4]:
            writer.writerow(row)

    report = run_validation(
        input_csv=out_csv,
        current_pct_threshold=15.0,
        torque_pct_threshold=15.0,
        efficiency_abs_threshold=0.05,
        power_factor_abs_threshold=0.05,
        ratio_pct_threshold=15.0,
        curves_output_dir=None,
        curve_points=40,
        solver="deterministic",
    )

    assert "typical_power_laws_by_group" in report
    assert "typical_power_law_curves_by_group" in report

    target_keys = {
        "normalized_r1",
        "normalized_x1",
        "normalized_r2",
        "normalized_x2",
        "normalized_r20",
        "normalized_x20",
        "normalized_rm",
        "normalized_xm",
    }

    for group_name in report["summary_by_group"].keys():
        metrics = report["typical_curve_metrics_by_group"][group_name]
        fits = report["typical_power_laws_by_group"][group_name]
        curves = report["typical_power_law_curves_by_group"][group_name]

        assert target_keys.issubset(set(metrics.keys()))
        assert target_keys.issubset(set(fits.keys()))
        assert target_keys.issubset(set(curves.keys()))

        for key in target_keys:
            assert "a" in fits[key]
            assert "b" in fits[key]
            assert "r2_log" in fits[key]
            assert isinstance(curves[key], list)
