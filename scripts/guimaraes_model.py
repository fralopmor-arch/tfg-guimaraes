from __future__ import annotations

import logging
import math
from dataclasses import dataclass

try:
    from scripts.catalog_schema import MotorRecord
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from catalog_schema import MotorRecord

LOGGER = logging.getLogger(__name__)

_EPS = 1e-9


@dataclass(frozen=True)
class EquivalentCircuitParameters:
    r1_ohm: float
    r2_base_ohm: float
    x1_ohm: float
    x2_ohm: float
    rm_ohm: float
    xm_ohm: float
    slip_rated: float
    gr: float


@dataclass(frozen=True)
class NominalPrediction:
    rated_current_a: float
    rated_torque_nm: float
    efficiency: float
    power_factor: float


def synchronous_speed_rpm(frequency_hz: float, pole_pairs: int) -> float:
    return 60.0 * frequency_hz / float(pole_pairs)


def rated_speed_rpm_from_power_and_torque(power_w: float, torque_nm: float) -> float:
    omega = power_w / max(torque_nm, _EPS)
    return omega * 60.0 / (2.0 * math.pi)


def rated_slip(record: MotorRecord) -> float:
    pole_pairs = record.resolved_pole_pairs
    ns = synchronous_speed_rpm(record.frequency_hz, pole_pairs)
    nr = rated_speed_rpm_from_power_and_torque(record.rated_power_w, record.rated_torque_nm)
    slip = (ns - nr) / max(ns, _EPS)
    return max(min(slip, 0.5), 0.001)


def reference_impedance(v_ll: float, pole_pairs: int, exponent_u: float) -> float:
    return (v_ll ** 2) / (((2.0 * pole_pairs) ** exponent_u) * 1.0e4)


def rotor_resistance_at_slip(params: EquivalentCircuitParameters, slip: float) -> float:
    safe_slip = max(min(slip, 1.0), 0.0)
    return params.r2_base_ohm * math.exp(params.gr * math.sqrt(max(1.0 - safe_slip, 0.0)))


def _phase_voltage(v_ll: float) -> float:
    return v_ll / math.sqrt(3.0)


def _torque_from_params(v_phase: float, omega_sync: float, slip: float, params: EquivalentCircuitParameters) -> float:
    safe_slip = max(slip, 0.001)
    r2_slip = rotor_resistance_at_slip(params, safe_slip)
    r2_effective = r2_slip / safe_slip
    x_total = params.x1_ohm + params.x2_ohm
    den = (params.r1_ohm + r2_effective) ** 2 + x_total ** 2
    return 3.0 * (v_phase ** 2) * r2_effective / max(omega_sync * den, _EPS)


def estimate_parameters(record: MotorRecord) -> EquivalentCircuitParameters:
    pole_pairs = record.resolved_pole_pairs
    slip = rated_slip(record)
    v_phase = _phase_voltage(record.rated_voltage_v)
    ns = synchronous_speed_rpm(record.frequency_hz, pole_pairs)
    omega_sync = 2.0 * math.pi * ns / 60.0

    z_mag = v_phase / max(record.rated_current_a, _EPS)
    angle = math.acos(max(min(record.power_factor, 1.0), 0.0))
    r_total = z_mag * math.cos(angle)
    x_total = max(z_mag * math.sin(angle), 1e-4)

    torque_gain = 3.0 * (v_phase ** 2) / max(omega_sync * (r_total ** 2 + x_total ** 2), _EPS)
    r2_effective_target = max(record.rated_torque_nm / max(torque_gain, _EPS), 1e-4)
    r1 = max(r_total - r2_effective_target, 1e-4)
    r2_effective = max(r_total - r1, 1e-4)

    if r2_effective_target > r_total:
        LOGGER.warning(
            "Motor %s: torque-implied rotor branch exceeds real impedance budget; clamped for physical consistency",
            record.motor_id,
        )

    gr = 0.45
    r2_rated = max(r2_effective * slip, 1e-4)
    r2_base = r2_rated / math.exp(gr * math.sqrt(max(1.0 - slip, 0.0)))

    x1 = 0.5 * x_total
    x2 = 0.5 * x_total

    rm = max(8.0 * reference_impedance(record.rated_voltage_v, pole_pairs, exponent_u=-0.25), 1.0)
    xm = max(3.0 * reference_impedance(record.rated_voltage_v, pole_pairs, exponent_u=0.333), 1e-4)

    params = EquivalentCircuitParameters(
        r1_ohm=r1,
        r2_base_ohm=r2_base,
        x1_ohm=x1,
        x2_ohm=x2,
        rm_ohm=rm,
        xm_ohm=xm,
        slip_rated=slip,
        gr=gr,
    )

    LOGGER.info(
        "Solved %s -> r1=%.5f, r2_base=%.5f, x1=%.5f, x2=%.5f, rm=%.5f, xm=%.5f, s=%.5f",
        record.motor_id,
        params.r1_ohm,
        params.r2_base_ohm,
        params.x1_ohm,
        params.x2_ohm,
        params.rm_ohm,
        params.xm_ohm,
        params.slip_rated,
    )
    return params


def predict_nominal(record: MotorRecord, params: EquivalentCircuitParameters) -> NominalPrediction:
    pole_pairs = record.resolved_pole_pairs
    ns = synchronous_speed_rpm(record.frequency_hz, pole_pairs)
    omega_sync = 2.0 * math.pi * ns / 60.0
    slip = params.slip_rated

    v_phase = _phase_voltage(record.rated_voltage_v)
    r2_slip = rotor_resistance_at_slip(params, slip)
    r2_effective = r2_slip / max(slip, 0.001)

    z_real = params.r1_ohm + r2_effective
    z_imag = params.x1_ohm + params.x2_ohm
    z_abs = math.sqrt(z_real ** 2 + z_imag ** 2)

    current = v_phase / max(z_abs, _EPS)
    power_factor = max(min(z_real / max(z_abs, _EPS), 1.0), 0.0)

    torque = _torque_from_params(v_phase, omega_sync, slip, params)
    omega_mech = (1.0 - slip) * omega_sync
    output_power = torque * omega_mech
    input_power = math.sqrt(3.0) * record.rated_voltage_v * current * power_factor
    efficiency = max(min(output_power / max(input_power, _EPS), 1.0), 0.0)

    return NominalPrediction(
        rated_current_a=current,
        rated_torque_nm=torque,
        efficiency=efficiency,
        power_factor=power_factor,
    )
