from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace

try:
    from scripts.guimaraes.catalog_schema import MotorRecord
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from catalog_schema import MotorRecord

LOGGER = logging.getLogger(__name__)

_EPS = 1e-6
_MAX_EFF = 0.9995
_MAX_PF = 0.9995
_MIN_PF = 0.05


@dataclass(frozen=True)
class GuardEvent:
    motor_id: str
    severity: str
    action: str
    field: str
    reason: str
    old_value: str | None = None
    new_value: str | None = None


@dataclass(frozen=True)
class GuardResult:
    accepted: bool
    record: MotorRecord | None
    events: list[GuardEvent]


def _event(
    events: list[GuardEvent],
    *,
    record: MotorRecord,
    severity: str,
    action: str,
    field: str,
    reason: str,
    old_value: str | None = None,
    new_value: str | None = None,
) -> None:
    payload = GuardEvent(
        motor_id=record.motor_id,
        severity=severity,
        action=action,
        field=field,
        reason=reason,
        old_value=old_value,
        new_value=new_value,
    )
    events.append(payload)
    log_fn = LOGGER.error if severity == "error" else LOGGER.warning
    log_fn(
        "[%s] %s field=%s reason=%s old=%s new=%s",
        record.motor_id,
        action,
        field,
        reason,
        old_value,
        new_value,
    )


def _clip_probability(
    value: float,
    *,
    field: str,
    record: MotorRecord,
    events: list[GuardEvent],
    low: float,
    high: float,
) -> float:
    if low <= value <= high:
        return value
    clipped = min(max(value, low), high)
    _event(
        events,
        record=record,
        severity="warning",
        action="auto_correct",
        field=field,
        reason="value clipped to physical probability bounds",
        old_value=f"{value:.6f}",
        new_value=f"{clipped:.6f}",
    )
    return clipped


def _validate_required_positive(record: MotorRecord, events: list[GuardEvent]) -> bool:
    checks = {
        "rated_voltage_v": record.rated_voltage_v,
        "rated_power_w": record.rated_power_w,
        "frequency_hz": record.frequency_hz,
        "rated_current_a": record.rated_current_a,
        "rated_torque_nm": record.rated_torque_nm,
    }
    valid = True
    for field, value in checks.items():
        if value <= 0.0:
            valid = False
            _event(
                events,
                record=record,
                severity="error",
                action="reject",
                field=field,
                reason="must be strictly positive",
                old_value=f"{value}",
            )
    return valid


def _compute_rated_slip(record: MotorRecord) -> float:
    pole_pairs = record.resolved_pole_pairs
    ns = 60.0 * record.frequency_hz / float(pole_pairs)
    omega_r = record.rated_power_w / max(record.rated_torque_nm, _EPS)
    nr = omega_r * 60.0 / (2.0 * math.pi)
    return (ns - nr) / max(ns, _EPS)


def _harmonize_eff_pf_load_points(record: MotorRecord, events: list[GuardEvent]) -> MotorRecord:
    eff50 = record.eff_50
    eff75 = record.eff_75
    eff100 = record.eff_100 if record.eff_100 is not None else record.efficiency
    pf50 = record.pf_50
    pf75 = record.pf_75
    pf100 = record.pf_100 if record.pf_100 is not None else record.power_factor

    if eff50 is None or eff75 is None or eff100 is None:
        _event(
            events,
            record=record,
            severity="error",
            action="reject",
            field="eff_50/eff_75/eff_100",
            reason="missing full efficiency load points required by paper",
        )
        return record

    if pf50 is None or pf75 is None or pf100 is None:
        _event(
            events,
            record=record,
            severity="error",
            action="reject",
            field="pf_50/pf_75/pf_100",
            reason="missing full power-factor load points required by paper",
        )
        return record

    eff50 = _clip_probability(eff50, field="eff_50", record=record, events=events, low=_EPS, high=_MAX_EFF)
    eff75 = _clip_probability(eff75, field="eff_75", record=record, events=events, low=_EPS, high=_MAX_EFF)
    eff100 = _clip_probability(eff100, field="eff_100", record=record, events=events, low=_EPS, high=_MAX_EFF)

    pf50 = _clip_probability(pf50, field="pf_50", record=record, events=events, low=_MIN_PF, high=_MAX_PF)
    pf75 = _clip_probability(pf75, field="pf_75", record=record, events=events, low=_MIN_PF, high=_MAX_PF)
    pf100 = _clip_probability(pf100, field="pf_100", record=record, events=events, low=_MIN_PF, high=_MAX_PF)

    # Mild monotonic enforcement avoids inverted load behavior due to extraction noise.
    eff75_fix = max(eff75, eff50)
    eff100_fix = max(eff100, eff75_fix)
    if abs(eff75_fix - eff75) > 0.0 or abs(eff100_fix - eff100) > 0.0:
        _event(
            events,
            record=record,
            severity="warning",
            action="auto_correct",
            field="eff_50/75/100",
            reason="enforced non-decreasing efficiency with load",
            old_value=f"{eff50:.6f},{eff75:.6f},{eff100:.6f}",
            new_value=f"{eff50:.6f},{eff75_fix:.6f},{eff100_fix:.6f}",
        )
    eff75 = eff75_fix
    eff100 = eff100_fix

    pf75_fix = max(pf75, pf50)
    pf100_fix = max(pf100, pf75_fix)
    if abs(pf75_fix - pf75) > 0.0 or abs(pf100_fix - pf100) > 0.0:
        _event(
            events,
            record=record,
            severity="warning",
            action="auto_correct",
            field="pf_50/75/100",
            reason="enforced non-decreasing power factor with load",
            old_value=f"{pf50:.6f},{pf75:.6f},{pf100:.6f}",
            new_value=f"{pf50:.6f},{pf75_fix:.6f},{pf100_fix:.6f}",
        )
    pf75 = pf75_fix
    pf100 = pf100_fix

    return replace(
        record,
        eff_50=eff50,
        eff_75=eff75,
        eff_100=eff100,
        efficiency=eff100,
        pf_50=pf50,
        pf_75=pf75,
        pf_100=pf100,
        power_factor=pf100,
    )


def apply_physical_guards(record: MotorRecord) -> GuardResult:
    events: list[GuardEvent] = []

    if not _validate_required_positive(record, events):
        return GuardResult(accepted=False, record=None, events=events)

    if record.ist_in is None or record.mst_mn is None or record.mk_mn is None:
        _event(
            events,
            record=record,
            severity="error",
            action="reject",
            field="ist_in/mst_mn/mk_mn",
            reason="missing start/rated/breakdown anchor ratios",
        )
        return GuardResult(accepted=False, record=None, events=events)

    ist_in = record.ist_in
    mst_mn = record.mst_mn
    mk_mn = record.mk_mn

    if ist_in <= 1.0:
        _event(
            events,
            record=record,
            severity="warning",
            action="auto_correct",
            field="ist_in",
            reason="start current ratio must exceed 1.0; clipped",
            old_value=f"{ist_in:.6f}",
            new_value="1.050000",
        )
        ist_in = 1.05

    if mst_mn <= _EPS:
        _event(
            events,
            record=record,
            severity="error",
            action="reject",
            field="mst_mn",
            reason="starting torque ratio must be positive",
            old_value=f"{mst_mn:.6f}",
        )
        return GuardResult(accepted=False, record=None, events=events)

    if mk_mn < mst_mn:
        mk_fix = mst_mn
        _event(
            events,
            record=record,
            severity="warning",
            action="auto_correct",
            field="mk_mn",
            reason="breakdown torque ratio cannot be lower than starting torque ratio",
            old_value=f"{mk_mn:.6f}",
            new_value=f"{mk_fix:.6f}",
        )
        mk_mn = mk_fix

    try:
        slip = _compute_rated_slip(record)
    except ValueError as exc:
        _event(
            events,
            record=record,
            severity="error",
            action="reject",
            field="poles/pole_pairs",
            reason=str(exc),
        )
        return GuardResult(accepted=False, record=None, events=events)

    if not (0.0005 < slip < 0.5):
        _event(
            events,
            record=record,
            severity="error",
            action="reject",
            field="rated_slip",
            reason="rated slip must be in (0.0005, 0.5)",
            old_value=f"{slip:.6f}",
        )
        return GuardResult(accepted=False, record=None, events=events)

    corrected = replace(record, ist_in=ist_in, mst_mn=mst_mn, mk_mn=mk_mn)
    corrected = _harmonize_eff_pf_load_points(corrected, events)

    rejected = any(event.action == "reject" for event in events)
    if rejected:
        return GuardResult(accepted=False, record=None, events=events)

    return GuardResult(accepted=True, record=corrected, events=events)
