from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from scripts.guimaraes.catalog_schema import MotorRecord, enforce_grouping_constraints
from scripts.guimaraes.guimaraes_model import solve_deterministic_blocks
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
