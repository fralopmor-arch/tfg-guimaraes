from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable
from typing import Iterable

try:
    from scripts.guimaraes.catalog_schema import MotorRecord
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
    gx: float


@dataclass(frozen=True)
class NominalPrediction:
    rated_current_a: float
    rated_torque_nm: float
    efficiency: float
    power_factor: float


@dataclass(frozen=True)
class OperatingPoint:
    slip: float
    current_a: float
    torque_nm: float
    efficiency: float
    power_factor: float


@dataclass(frozen=True)
class CharacteristicPoints:
    nominal: OperatingPoint
    start: OperatingPoint
    breakdown: OperatingPoint


@dataclass(frozen=True)
class DeterministicBlockState:
    slip_rated: float
    slip_breakdown: float
    r1_ohm: float
    prot_w: float
    r2_start_ohm: float
    r2_rated_ohm: float
    r2_breakdown_ohm: float
    gr: float
    x1_ohm: float
    x2_start_ohm: float
    x2_breakdown_ohm: float
    x2_rated_ohm: float
    gx: float
    rm_ohm: float
    xm_ohm: float


@dataclass(frozen=True)
class LegacyRuntimeConfig:
    breakdown_samples: int = 600
    beta_min_tenths: int = 8
    beta_max_tenths: int = 16
    alpha_min_tenths: int = 1
    alpha_max_tenths: int = 9
    r1_steps: int = 31
    gr_steps: int = 51
    rm_steps: int = 21
    xm_steps: int = 31


_RM_EXPONENT_U = -0.25
_XM_EXPONENT_U = 0.333
_X1_EXPONENT_U = -0.333

_PerformanceHook = Callable[[str, float, int], None]
_PERFORMANCE_HOOK: _PerformanceHook | None = None


def set_performance_hook(hook: _PerformanceHook | None) -> None:
    global _PERFORMANCE_HOOK
    _PERFORMANCE_HOOK = hook


def _record_performance(stage: str, elapsed_seconds: float, count: int = 1) -> None:
    if _PERFORMANCE_HOOK is None:
        return
    _PERFORMANCE_HOOK(stage, elapsed_seconds, count)


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


def rotor_reactance_at_slip(params: EquivalentCircuitParameters, slip: float) -> float:
    safe_slip = max(min(slip, 1.0), 0.0)
    return params.x2_ohm * math.exp(params.gx * math.sqrt(max(1.0 - safe_slip, 0.0)))


def _magnetizing_impedance(params: EquivalentCircuitParameters) -> complex:
    z_rm = complex(params.rm_ohm, 0.0)
    z_xm = complex(0.0, params.xm_ohm)
    den = z_rm + z_xm
    if abs(den) <= _EPS:
        den = complex(_EPS, 0.0)
    return (z_rm * z_xm) / den


def _thevenin_equivalent(params: EquivalentCircuitParameters, v_phase: float) -> tuple[complex, complex]:
    z1 = complex(params.r1_ohm, params.x1_ohm)
    zm = _magnetizing_impedance(params)
    z_sum = z1 + zm
    if abs(z_sum) <= _EPS:
        z_sum = complex(_EPS, 0.0)
    v_th = v_phase * (zm / z_sum)
    z_th = (z1 * zm) / z_sum
    return v_th, z_th


def _phase_voltage(v_ll: float) -> float:
    return v_ll / math.sqrt(3.0)


def _torque_from_params(v_phase: float, omega_sync: float, slip: float, params: EquivalentCircuitParameters) -> float:
    safe_slip = max(slip, 0.001)
    r2_slip = rotor_resistance_at_slip(params, safe_slip)
    x2_slip = rotor_reactance_at_slip(params, safe_slip)
    z2 = complex(r2_slip / safe_slip, x2_slip)
    v_th, z_th = _thevenin_equivalent(params, v_phase)
    den = z_th + z2
    if abs(den) <= _EPS:
        den = complex(_EPS, 0.0)
    i2 = v_th / den
    p_ag = 3.0 * (abs(i2) ** 2) * (r2_slip / safe_slip)
    return p_ag / max(omega_sync, _EPS)


def _solve_breakdown_slip(r1: float, r2_start: float, gr: float, mk_target: float, k_torque: float) -> tuple[float, float]:
    y = (k_torque / max(mk_target, _EPS)) - r1
    if y <= abs(r1):
        return 1.0, 0.0

    x_total_k = math.sqrt(max((y ** 2) - (r1 ** 2), 0.0))
    s = 0.2
    for _ in range(50):
        r2_k = r2_start * math.exp(gr * math.sqrt(max(1.0 - s, 0.0)))
        s_next = max(min(r2_k / max(y, _EPS), 1.0), 0.001)
        if abs(s_next - s) < 1e-7:
            s = s_next
            break
        s = s_next
    return s, x_total_k


def _gr_from_rated(r2_start: float, r2_rated: float, slip_rated: float) -> float:
    a = math.sqrt(max(1.0 - slip_rated, 0.0))
    if a <= _EPS:
        return 0.0
    ratio = max(r2_rated, _EPS) / max(r2_start, _EPS)
    return math.log(max(ratio, _EPS)) / a


def _gr_from_rated_torque(
    r1: float,
    r2_start: float,
    x_total_rated: float,
    slip_rated: float,
    torque_target: float,
    k_torque: float,
) -> float | None:
    a = max(torque_target, _EPS)
    b = (2.0 * torque_target * r1) - k_torque
    c = torque_target * ((r1 ** 2) + (x_total_rated ** 2))
    disc = (b ** 2) - (4.0 * a * c)
    if disc < 0.0:
        return None

    sqrt_disc = math.sqrt(max(disc, 0.0))
    y1 = (-b + sqrt_disc) / (2.0 * a)
    y2 = (-b - sqrt_disc) / (2.0 * a)
    y = max(y1, y2)
    if y <= _EPS:
        return None

    r2_rated = y * max(slip_rated, _EPS)
    return _gr_from_rated(r2_start, r2_rated, slip_rated)


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if len(xs) != len(ys) or not xs:
        raise ValueError("linear fit requires non-empty vectors with equal sizes")

    n = float(len(xs))
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    den = sum((x - x_mean) ** 2 for x in xs)
    if den <= _EPS:
        return 0.0, y_mean

    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = num / den
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _block3_efficiency_pf_points(record: MotorRecord) -> list[tuple[float, float, float]]:
    eff_100 = record.eff_100 if record.eff_100 is not None else record.efficiency
    pf_100 = record.pf_100 if record.pf_100 is not None else record.power_factor
    return [
        (0.50, float(record.eff_50 or eff_100), float(record.pf_50 or pf_100)),
        (0.75, float(record.eff_75 or eff_100), float(record.pf_75 or pf_100)),
        (1.00, float(eff_100), float(pf_100)),
    ]


def _slip_from_load_fraction(slip_rated: float, load_fraction: float) -> float:
    # Eq.13 form with clipping to avoid negative discriminant from noisy data.
    disc = 1.0 - (4.0 * slip_rated * (1.0 - slip_rated) * load_fraction)
    disc = max(disc, 0.0)
    slip = 0.5 * (1.0 - math.sqrt(disc))
    return max(min(slip, 0.5), 0.001)


def _breakdown_slip_from_ratio(slip_rated: float, mk_mn: float | None) -> float:
    if mk_mn is None or mk_mn <= 1.0:
        return min(max(3.0 * slip_rated, 0.02), 0.35)
    root = math.sqrt(max((mk_mn ** 2) - 1.0, 0.0))
    sk = slip_rated * (mk_mn + root)
    return min(max(sk, slip_rated + 1e-4), 0.98)


def _reference_defaults(record: MotorRecord) -> tuple[float, float, float]:
    pole_pairs = record.resolved_pole_pairs
    rm_ref = max(8.0 * reference_impedance(record.rated_voltage_v, pole_pairs, _RM_EXPONENT_U), 1e-3)
    xm_ref = max(3.0 * reference_impedance(record.rated_voltage_v, pole_pairs, _XM_EXPONENT_U), 1e-4)
    x1_ref = max(0.25 * reference_impedance(record.rated_voltage_v, pole_pairs, _X1_EXPONENT_U), 1e-4)
    return rm_ref, xm_ref, x1_ref


def normalize_for_typical_curve(parameter_ohm: float, record: MotorRecord, parameter_name: str) -> float:
    exponent_by_name = {
        "rm": _RM_EXPONENT_U,
        "xm": _XM_EXPONENT_U,
        "x1": _X1_EXPONENT_U,
    }
    if parameter_name not in exponent_by_name:
        raise ValueError(f"Unsupported parameter_name '{parameter_name}' for normalization")
    z_ref = reference_impedance(record.rated_voltage_v, record.resolved_pole_pairs, exponent_by_name[parameter_name])
    return parameter_ohm / max(z_ref, _EPS)


def solve_deterministic_blocks(record: MotorRecord) -> DeterministicBlockState:
    pole_pairs = record.resolved_pole_pairs
    slip_r = rated_slip(record)
    v_phase = _phase_voltage(record.rated_voltage_v)
    i_rated = max(record.rated_current_a, _EPS)
    ns = synchronous_speed_rpm(record.frequency_hz, pole_pairs)
    nr = ns * (1.0 - slip_r)
    ns_nr = ns / max(nr, _EPS)

    # Block I (Eq.11-22): resistive calibration.
    xs: list[float] = []
    ys: list[float] = []
    xs_r2: list[float] = []
    ys_r2: list[float] = []
    for load_fraction, eta, pf in _block3_efficiency_pf_points(record):
        eta_safe = max(min(eta, 0.9995), 1e-4)
        pf_safe = max(min(pf, 0.9995), 0.05)
        p2 = record.rated_power_w * load_fraction
        pin = p2 / eta_safe
        i1 = pin / max((math.sqrt(3.0) * record.rated_voltage_v * pf_safe), _EPS)
        slip_i = _slip_from_load_fraction(slip_r, load_fraction)
        x_i = 3.0 * (i1 ** 2)
        xs.append(x_i)
        ys.append(pin - (p2 / max(1.0 - slip_i, _EPS)))
        xs_r2.append(x_i)
        ys_r2.append((p2 * slip_i) / max(1.0 - slip_i, _EPS))

    r1_fit, prot_fit = _linear_fit(xs, ys)
    r2_fit, _ = _linear_fit(xs_r2, ys_r2)
    r1 = max(r1_fit, 1e-4)
    prot = max(prot_fit, 1e-3)

    z_rated = v_phase / i_rated
    pf_100 = record.pf_100 if record.pf_100 is not None else record.power_factor
    phi_100 = math.acos(max(min(pf_100, 0.9995), 0.0))
    r_total = z_rated * math.cos(phi_100)

    # Eq.15 explicit path: r2 is the slope of P2*s/(1-s) versus 3*I1^2.
    if math.isfinite(r2_fit) and r2_fit > 0.0:
        r2_rated = max(r2_fit, 1e-4)
    else:
        r2_rated = max((r_total - r1) * slip_r, 1e-4)

    i_start = i_rated * max(record.ist_in or 1.0, 1.0)
    mst_mn = max(record.mst_mn or 1.0, 1e-4)
    r2_start = max(
        (record.rated_power_w * mst_mn) / max(3.0 * (i_start ** 2) * ns_nr, _EPS),
        1e-4,
    )

    gr = _gr_from_rated(r2_start, r2_rated, slip_r)
    r2_k = r2_start * math.exp(gr * math.sqrt(max(1.0 - _breakdown_slip_from_ratio(slip_r, record.mk_mn), 0.0)))

    # Keep rotor resistance law monotonic from start -> breakdown -> rated.
    if r2_rated < r2_start:
        r2_rated = max(r2_start * 1.05, r2_rated)
        gr = _gr_from_rated(r2_start, r2_rated, slip_r)

    sk = _breakdown_slip_from_ratio(slip_r, record.mk_mn)
    r2_k = r2_start * math.exp(gr * math.sqrt(max(1.0 - sk, 0.0)))

    # Block II (Eq.23-28): reactive calibration.
    x2_k = max(r2_k / max(sk, _EPS), 1e-4)
    mk_mn = max(record.mk_mn or 1.0, 1.0)
    x2_start_sq = (2.0 * r2_start * x2_k * mk_mn / max(mst_mn, _EPS)) - (r2_start ** 2)
    x2_start = math.sqrt(max(x2_start_sq, 1e-8))

    den_gx = math.sqrt(max(1.0 - sk, 0.0))
    if den_gx <= _EPS:
        gx = 0.0
    else:
        gx = math.log(max(x2_k, _EPS) / max(x2_start, _EPS)) / den_gx

    x2_rated = x2_start * math.exp(gx * math.sqrt(max(1.0 - slip_r, 0.0)))
    ist_in = max(record.ist_in or 1.0, 1.0)
    z_start_abs = v_phase / max(i_rated * ist_in, _EPS)
    x1_sq = (
        (z_start_abs ** 2)
        - ((r1 + r2_start) ** 2)
        - ((x2_start ** 2) / max(1.0 + ((slip_r / max(sk, _EPS)) ** 2), _EPS))
    )
    x1 = math.sqrt(max(x1_sq, 1e-8))

    # Block III (Eq.29-34): magnetizing calibration.
    pin_rated = record.rated_power_w / max(record.efficiency, _EPS)
    prot_eq29 = pin_rated - (record.rated_power_w / max(1.0 - slip_r, _EPS)) - (3.0 * r1 * (i_rated ** 2))
    e_internal = v_phase - (i_rated * math.sqrt((r1 ** 2) + (x1 ** 2)))

    rm_ref, xm_ref, x1_ref = _reference_defaults(record)
    if e_internal <= _EPS or prot_eq29 <= _EPS:
        rm = rm_ref
        xm = xm_ref
    else:
        rm = max((3.0 * (e_internal ** 2)) / prot_eq29, 1e-3)
        i0p = e_internal / max(rm, _EPS)
        i1q = i_rated * math.sin(phi_100)
        i0q_sq = (i1q ** 2) - (i0p ** 2)
        if i0q_sq <= _EPS:
            xm = xm_ref
        else:
            xm = max(e_internal / math.sqrt(i0q_sq), 1e-4)

    # Fallbacks only where deterministic equations are singular/noisy.
    if not math.isfinite(x1) or x1 <= 0.0:
        x1 = x1_ref
    if not math.isfinite(r1) or r1 <= 0.0:
        r1 = max(0.15 * z_rated, 1e-4)
    if not math.isfinite(gr):
        gr = 0.0
    if not math.isfinite(gx):
        gx = 0.0

    return DeterministicBlockState(
        slip_rated=slip_r,
        slip_breakdown=sk,
        r1_ohm=r1,
        prot_w=max(prot_eq29, prot),
        r2_start_ohm=r2_start,
        r2_rated_ohm=r2_rated,
        r2_breakdown_ohm=r2_k,
        gr=gr,
        x1_ohm=x1,
        x2_start_ohm=x2_start,
        x2_breakdown_ohm=x2_k,
        x2_rated_ohm=x2_rated,
        gx=gx,
        rm_ohm=rm,
        xm_ohm=xm,
    )


def estimate_parameters_deterministic(record: MotorRecord) -> EquivalentCircuitParameters:
    solver_start = time.perf_counter()
    try:
        state = solve_deterministic_blocks(record)
        params = EquivalentCircuitParameters(
            r1_ohm=state.r1_ohm,
            r2_base_ohm=state.r2_start_ohm,
            x1_ohm=state.x1_ohm,
            x2_ohm=state.x2_start_ohm,
            rm_ohm=state.rm_ohm,
            xm_ohm=state.xm_ohm,
            slip_rated=state.slip_rated,
            gr=state.gr,
            gx=state.gx,
        )

        LOGGER.info(
            "Deterministic solve %s -> r1=%.5f r2s=%.5f x1=%.5f x2s=%.5f rm=%.5f xm=%.5f sR=%.5f sk=%.5f",
            record.motor_id,
            params.r1_ohm,
            state.r2_start_ohm,
            params.x1_ohm,
            state.x2_start_ohm,
            params.rm_ohm,
            params.xm_ohm,
            state.slip_rated,
            state.slip_breakdown,
        )
        return params
    finally:
        _record_performance("solver_deterministic", time.perf_counter() - solver_start)


def estimate_parameters(
    record: MotorRecord,
    *,
    legacy_config: LegacyRuntimeConfig | None = None,
) -> EquivalentCircuitParameters:
    try:
        return estimate_parameters_deterministic(record)
    except Exception as exc:
        LOGGER.warning(
            "Motor %s: deterministic Eq.11-34 chain failed (%s); falling back to legacy estimator",
            record.motor_id,
            exc,
        )
        return estimate_parameters_legacy(record, runtime_config=legacy_config)


def estimate_parameters_legacy(
    record: MotorRecord,
    *,
    runtime_config: LegacyRuntimeConfig | None = None,
) -> EquivalentCircuitParameters:
    runtime = runtime_config or LegacyRuntimeConfig()
    solver_start = time.perf_counter()

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

    r2_rated = max(r2_effective * slip, 1e-4)
    r2_base = r2_rated
    gr = 0.45

    if record.has_block3_targets and record.ist_in is not None and record.mst_mn is not None:
        i_start_target = max(record.ist_in * record.rated_current_a, 1e-4)
        z_start_abs = v_phase / i_start_target
        start_torque_target = max(record.mst_mn * record.rated_torque_nm, 1e-4)

        # Eq. (6)/(16)-style direct relation at start.
        k_torque = 3.0 * (v_phase ** 2) / max(omega_sync, _EPS)
        r2_base = max(start_torque_target * (z_start_abs ** 2) / max(k_torque, _EPS), 1e-4)

        feasible_r1_max = max(math.sqrt(max(z_start_abs ** 2, 0.0)) - r2_base - 1e-4, 1e-4)
        r1 = min(r1, feasible_r1_max)
        x_start_total = math.sqrt(max((z_start_abs ** 2) - ((r1 + r2_base) ** 2), 1e-8))

        r2_rated = max(slip * max(r_total - r1, 1e-4), 1e-4)
        gr = _gr_from_rated(r2_base, r2_rated, slip)

        breakdown_target = None
        sk_target = 0.2
        x_total_k_target = x_start_total
        if record.mk_mn is not None:
            breakdown_target = record.mk_mn * record.rated_torque_nm
            sk_target, x_total_k_target = _solve_breakdown_slip(
                r1=r1,
                r2_start=r2_base,
                gr=gr,
                mk_target=breakdown_target,
                k_torque=k_torque,
            )

        rm = max(8.0 * reference_impedance(record.rated_voltage_v, pole_pairs, exponent_u=-0.25), 1.0)
        xm = max(3.0 * reference_impedance(record.rated_voltage_v, pole_pairs, exponent_u=0.333), 1e-4)

        best_r1 = r1
        best_gr = gr
        best_gx = 0.0
        best_x1 = 0.5 * x_start_total
        best_x2_base = 0.5 * x_start_total
        baseline = EquivalentCircuitParameters(
            r1_ohm=r1,
            r2_base_ohm=r2_base,
            x1_ohm=best_x1,
            x2_ohm=best_x2_base,
            rm_ohm=rm,
            xm_ohm=xm,
            slip_rated=slip,
            gr=best_gr,
            gx=best_gx,
        )
        baseline_rated = evaluate_operating_point(record, baseline, slip)
        baseline_start = evaluate_operating_point(record, baseline, 1.0)
        best_error = (
            abs(baseline_rated.current_a - record.rated_current_a) / max(record.rated_current_a, _EPS) / 0.15
            + abs(baseline_rated.torque_nm - record.rated_torque_nm) / max(record.rated_torque_nm, _EPS) / 0.15
            + abs(baseline_rated.power_factor - record.power_factor) / 0.05
            + 0.5 * (abs((baseline_start.current_a / record.rated_current_a) - record.ist_in) / max(record.ist_in, _EPS) / 0.15)
            + 0.5 * (abs((baseline_start.torque_nm / record.rated_torque_nm) - record.mst_mn) / max(record.mst_mn, _EPS) / 0.15)
        )
        if breakdown_target is not None:
            baseline_breakdown = find_breakdown_point(record, baseline, samples=runtime.breakdown_samples)
            best_error += abs(baseline_breakdown.torque_nm - breakdown_target) / max(breakdown_target, _EPS) / 0.15
            best_error += 0.5 * (abs(baseline_breakdown.slip - sk_target) / max(sk_target, 0.02) / 0.25)

        for beta_idx in range(runtime.beta_min_tenths, runtime.beta_max_tenths + 1):
            beta = beta_idx / 10.0
            x_start_candidate = x_start_total * beta
            for alpha_idx in range(runtime.alpha_min_tenths, runtime.alpha_max_tenths + 1):
                alpha = alpha_idx / 10.0
                x1 = alpha * x_start_candidate
                x2_base = (1.0 - alpha) * x_start_candidate

                # First pass gx from current sk_target, then close gr from rated torque equation.
                x2_k_target = x_total_k_target - x1
                if x2_base <= _EPS or x2_k_target <= _EPS:
                    continue

                a_k = math.sqrt(max(1.0 - sk_target, 0.0))
                if a_k <= _EPS:
                    gx = 0.0
                else:
                    gx = math.log(max(x2_k_target, _EPS) / max(x2_base, _EPS)) / a_k

                x_total_rated = x1 + (x2_base * math.exp(gx * math.sqrt(max(1.0 - slip, 0.0))))
                gr_candidate = _gr_from_rated_torque(
                    r1=r1,
                    r2_start=r2_base,
                    x_total_rated=x_total_rated,
                    slip_rated=slip,
                    torque_target=record.rated_torque_nm,
                    k_torque=k_torque,
                )
                if gr_candidate is None:
                    continue

                sk_candidate = sk_target
                if breakdown_target is not None:
                    sk_candidate, _ = _solve_breakdown_slip(
                        r1=r1,
                        r2_start=r2_base,
                        gr=gr_candidate,
                        mk_target=breakdown_target,
                        k_torque=k_torque,
                    )

                params_candidate = EquivalentCircuitParameters(
                    r1_ohm=r1,
                    r2_base_ohm=r2_base,
                    x1_ohm=x1,
                    x2_ohm=x2_base,
                    rm_ohm=rm,
                    xm_ohm=xm,
                    slip_rated=slip,
                    gr=gr_candidate,
                    gx=gx,
                )
                rated_candidate = evaluate_operating_point(record, params_candidate, slip)
                start_candidate = evaluate_operating_point(record, params_candidate, 1.0)

                err_current_pct = abs(rated_candidate.current_a - record.rated_current_a) / max(record.rated_current_a, _EPS)
                err_torque_pct = abs(rated_candidate.torque_nm - record.rated_torque_nm) / max(record.rated_torque_nm, _EPS)
                err_pf_abs = abs(rated_candidate.power_factor - record.power_factor)
                err_ist_pct = abs((start_candidate.current_a / record.rated_current_a) - record.ist_in) / max(record.ist_in, _EPS)
                err_mst_pct = abs((start_candidate.torque_nm / record.rated_torque_nm) - record.mst_mn) / max(record.mst_mn, _EPS)

                error = (err_current_pct / 0.15) + (err_torque_pct / 0.15) + (err_pf_abs / 0.05)
                error += 0.5 * (err_ist_pct / 0.15) + 0.5 * (err_mst_pct / 0.15)

                if breakdown_target is not None:
                    breakdown_candidate = find_breakdown_point(record, params_candidate, samples=runtime.breakdown_samples)
                    err_mk_pct = abs(breakdown_candidate.torque_nm - breakdown_target) / max(breakdown_target, _EPS)
                    error += err_mk_pct / 0.15
                    err_sk = abs(breakdown_candidate.slip - sk_candidate) / max(sk_candidate, 0.02)
                    error += 0.5 * (err_sk / 0.25)

                if error < best_error:
                    best_error = error
                    best_gr = gr_candidate
                    best_gx = gx
                    best_x1 = x1
                    best_x2_base = x2_base

        # Additive V3 step: keep x1/x2/gx from the best beta-alpha candidate and
        # refine (r1, gr) to close rated PF/torque while preserving Block III fits.
        ratio_threshold = 0.15
        current_threshold = 0.15
        torque_threshold = 0.15
        pf_threshold = 0.05

        r1_span = max(0.02 * r_total, 0.25 * max(best_r1, 1e-4), 1e-4)
        r1_min = max(1e-4, best_r1 - r1_span)
        r1_max = min(feasible_r1_max, best_r1 + r1_span)
        if r1_max < r1_min:
            r1_min = max(1e-4, min(best_r1, feasible_r1_max))
            r1_max = r1_min

        gr_span = 0.35
        gr_min = max(best_gr - gr_span, -1.0)
        gr_max = min(best_gr + gr_span, 2.0)

        refined_params: EquivalentCircuitParameters | None = None
        refined_key: tuple[float, float, float, float] | None = None
        best_unconstrained_key: tuple[float, float, float, float] | None = None
        best_unconstrained_metrics: tuple[float, float, float, float, float, float] | None = None

        total_candidates = 0
        finite_candidates = 0
        pass_ist_count = 0
        pass_mst_count = 0
        pass_mk_count = 0
        pass_all_count = 0

        r1_steps = max(runtime.r1_steps, 1)
        gr_steps = max(runtime.gr_steps, 1)
        for r1_idx in range(r1_steps):
            if r1_steps == 1:
                r1_candidate = r1_min
            else:
                r1_candidate = r1_min + (r1_max - r1_min) * (r1_idx / (r1_steps - 1))
            for gr_idx in range(gr_steps):
                total_candidates += 1
                if gr_steps == 1:
                    gr_candidate = gr_min
                else:
                    gr_candidate = gr_min + (gr_max - gr_min) * (gr_idx / (gr_steps - 1))

                params_candidate = EquivalentCircuitParameters(
                    r1_ohm=r1_candidate,
                    r2_base_ohm=r2_base,
                    x1_ohm=best_x1,
                    x2_ohm=best_x2_base,
                    rm_ohm=rm,
                    xm_ohm=xm,
                    slip_rated=slip,
                    gr=gr_candidate,
                    gx=best_gx,
                )

                rated_candidate = evaluate_operating_point(record, params_candidate, slip)
                start_candidate = evaluate_operating_point(record, params_candidate, 1.0)
                if (
                    not math.isfinite(rated_candidate.current_a)
                    or not math.isfinite(rated_candidate.torque_nm)
                    or not math.isfinite(rated_candidate.power_factor)
                    or not math.isfinite(start_candidate.current_a)
                    or not math.isfinite(start_candidate.torque_nm)
                ):
                    continue
                finite_candidates += 1

                err_current = abs(rated_candidate.current_a - record.rated_current_a) / max(record.rated_current_a, _EPS)
                err_torque = abs(rated_candidate.torque_nm - record.rated_torque_nm) / max(record.rated_torque_nm, _EPS)
                err_pf = abs(rated_candidate.power_factor - record.power_factor)
                err_ist = abs((start_candidate.current_a / record.rated_current_a) - record.ist_in) / max(record.ist_in, _EPS)
                err_mst = abs((start_candidate.torque_nm / record.rated_torque_nm) - record.mst_mn) / max(record.mst_mn, _EPS)

                err_mk = 0.0
                if breakdown_target is not None:
                    breakdown_candidate = find_breakdown_point(record, params_candidate, samples=runtime.breakdown_samples)
                    if not math.isfinite(breakdown_candidate.torque_nm):
                        continue
                    err_mk = abs((breakdown_candidate.torque_nm / record.rated_torque_nm) - record.mk_mn) / max(record.mk_mn, _EPS)

                if err_ist <= ratio_threshold:
                    pass_ist_count += 1
                if err_mst <= ratio_threshold:
                    pass_mst_count += 1
                if err_mk <= ratio_threshold:
                    pass_mk_count += 1

                hard_ok = err_ist <= ratio_threshold and err_mst <= ratio_threshold and err_mk <= ratio_threshold
                if hard_ok:
                    pass_all_count += 1

                # Primary objective: rated PF and rated torque. Current is secondary.
                objective_primary = (err_pf / pf_threshold) + (err_torque / torque_threshold)
                objective_secondary = (err_current / current_threshold) + 0.35 * ((err_ist + err_mst + err_mk) / ratio_threshold)
                key = (objective_primary, objective_secondary, err_pf, err_torque)

                if best_unconstrained_key is None or key < best_unconstrained_key:
                    best_unconstrained_key = key
                    best_unconstrained_metrics = (
                        err_current,
                        err_torque,
                        err_pf,
                        err_ist,
                        err_mst,
                        err_mk,
                    )

                if not hard_ok:
                    continue

                if refined_key is None or key < refined_key:
                    refined_key = key
                    refined_params = params_candidate

        if best_unconstrained_metrics is not None:
            LOGGER.debug(
                (
                    "Motor %s: post-refinement diagnostics -> total=%d finite=%d pass(ist)=%d pass(mst)=%d "
                    "pass(mk)=%d pass(all)=%d best_unconstrained(errI=%.3f%% errT=%.3f%% errPF=%.4f errIst=%.3f%% errMst=%.3f%% errMk=%.3f%%)"
                ),
                record.motor_id,
                total_candidates,
                finite_candidates,
                pass_ist_count,
                pass_mst_count,
                pass_mk_count,
                pass_all_count,
                100.0 * best_unconstrained_metrics[0],
                100.0 * best_unconstrained_metrics[1],
                best_unconstrained_metrics[2],
                100.0 * best_unconstrained_metrics[3],
                100.0 * best_unconstrained_metrics[4],
                100.0 * best_unconstrained_metrics[5],
            )

        if refined_params is not None:
            r1 = refined_params.r1_ohm
            gr = refined_params.gr
            gx = refined_params.gx
            x1 = refined_params.x1_ohm
            x2 = refined_params.x2_ohm
            LOGGER.info(
                "Motor %s: post-refined (r1, gr) with hard Block III constraints",
                record.motor_id,
            )
        else:
            r1 = best_r1
            gr = best_gr
            gx = best_gx
            x1 = best_x1
            x2 = best_x2_base
            LOGGER.info(
                "Motor %s: no hard-feasible post-refinement candidate; keeping beta-alpha best",
                record.motor_id,
            )

        # Deterministic PF/current closure stage: refine magnetizing branch while
        # preserving start and breakdown quality targets.
        rm_base = rm
        xm_base = xm
        rm_scale_min = 0.25
        rm_scale_max = 1.25
        xm_scale_min = 0.04
        xm_scale_max = 1.20
        rm_steps = max(runtime.rm_steps, 1)
        xm_steps = max(runtime.xm_steps, 1)

        best_mag_params: tuple[float, float] | None = None
        best_mag_key: tuple[float, float, float] | None = None
        mag_total_candidates = 0
        mag_feasible_candidates = 0

        for rm_idx in range(rm_steps):
            rm_scale = rm_scale_min + (rm_scale_max - rm_scale_min) * (rm_idx / max(rm_steps - 1, 1))
            rm_candidate = max(rm_base * rm_scale, 1e-3)
            for xm_idx in range(xm_steps):
                mag_total_candidates += 1
                xm_scale = xm_scale_min + (xm_scale_max - xm_scale_min) * (xm_idx / max(xm_steps - 1, 1))
                xm_candidate = max(xm_base * xm_scale, 1e-4)

                params_candidate = EquivalentCircuitParameters(
                    r1_ohm=r1,
                    r2_base_ohm=r2_base,
                    x1_ohm=x1,
                    x2_ohm=x2,
                    rm_ohm=rm_candidate,
                    xm_ohm=xm_candidate,
                    slip_rated=slip,
                    gr=gr,
                    gx=gx,
                )

                rated_candidate = evaluate_operating_point(record, params_candidate, slip)
                start_candidate = evaluate_operating_point(record, params_candidate, 1.0)
                breakdown_candidate = find_breakdown_point(record, params_candidate, samples=runtime.breakdown_samples)
                if (
                    not math.isfinite(rated_candidate.current_a)
                    or not math.isfinite(rated_candidate.torque_nm)
                    or not math.isfinite(rated_candidate.power_factor)
                    or not math.isfinite(start_candidate.current_a)
                    or not math.isfinite(start_candidate.torque_nm)
                    or not math.isfinite(breakdown_candidate.torque_nm)
                ):
                    continue

                err_current = abs(rated_candidate.current_a - record.rated_current_a) / max(record.rated_current_a, _EPS)
                err_torque = abs(rated_candidate.torque_nm - record.rated_torque_nm) / max(record.rated_torque_nm, _EPS)
                err_pf = abs(rated_candidate.power_factor - record.power_factor)
                err_ist = abs((start_candidate.current_a / record.rated_current_a) - record.ist_in) / max(record.ist_in, _EPS)
                err_mst = abs((start_candidate.torque_nm / record.rated_torque_nm) - record.mst_mn) / max(record.mst_mn, _EPS)
                err_mk = abs((breakdown_candidate.torque_nm / record.rated_torque_nm) - record.mk_mn) / max(record.mk_mn, _EPS)

                hard_ok = err_ist <= ratio_threshold and err_mst <= ratio_threshold and err_mk <= ratio_threshold
                if not hard_ok:
                    continue
                mag_feasible_candidates += 1

                # Prioritize PF closure, then rated current and torque.
                key = (
                    err_pf / pf_threshold,
                    err_current / current_threshold,
                    err_torque / torque_threshold,
                )
                if best_mag_key is None or key < best_mag_key:
                    best_mag_key = key
                    best_mag_params = (rm_candidate, xm_candidate)

        if best_mag_params is not None:
            rm = best_mag_params[0]
            xm = best_mag_params[1]
            LOGGER.debug(
                "Motor %s: magnetizing refinement selected rm=%.5f, xm=%.5f (%d/%d feasible)",
                record.motor_id,
                rm,
                xm,
                mag_feasible_candidates,
                mag_total_candidates,
            )
        else:
            rm = rm_base
            xm = xm_base
            LOGGER.debug(
                "Motor %s: magnetizing refinement found no hard-feasible candidate; keeping baseline rm/xm",
                record.motor_id,
            )
    else:
        a = math.sqrt(max(1.0 - slip, 0.0))
        if record.ist_in is not None and record.ist_in > 0.0 and a > _EPS:
            i_start_target = record.ist_in * record.rated_current_a
            z_start_abs = v_phase / max(i_start_target, _EPS)
            start_real_budget_sq = max((z_start_abs ** 2) - (x_total ** 2), 0.0)
            start_real_budget = math.sqrt(start_real_budget_sq)
            ratio = (start_real_budget - r1) / max(r2_rated, _EPS)
            if ratio > 0.0:
                gr = -math.log(ratio) / a

        r2_base = r2_rated / math.exp(gr * a)
        gx = 0.0
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
        gx=gx,
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
    _record_performance("solver_legacy", time.perf_counter() - solver_start)
    return params


def predict_nominal(record: MotorRecord, params: EquivalentCircuitParameters) -> NominalPrediction:
    point = evaluate_operating_point(record, params, params.slip_rated)

    return NominalPrediction(
        rated_current_a=point.current_a,
        rated_torque_nm=point.torque_nm,
        efficiency=point.efficiency,
        power_factor=point.power_factor,
    )


def evaluate_operating_point(record: MotorRecord, params: EquivalentCircuitParameters, slip: float) -> OperatingPoint:
    pole_pairs = record.resolved_pole_pairs
    ns = synchronous_speed_rpm(record.frequency_hz, pole_pairs)
    omega_sync = 2.0 * math.pi * ns / 60.0

    safe_slip = max(min(slip, 1.0), 0.001)
    v_phase = _phase_voltage(record.rated_voltage_v)
    r2_slip = rotor_resistance_at_slip(params, safe_slip)
    x2_slip = rotor_reactance_at_slip(params, safe_slip)
    z1 = complex(params.r1_ohm, params.x1_ohm)
    zm = _magnetizing_impedance(params)
    z2 = complex(r2_slip / safe_slip, x2_slip)
    den_parallel = zm + z2
    if abs(den_parallel) <= _EPS:
        den_parallel = complex(_EPS, 0.0)
    z_parallel = (zm * z2) / den_parallel
    z_in = z1 + z_parallel
    if abs(z_in) <= _EPS:
        z_in = complex(_EPS, 0.0)
    i1 = v_phase / z_in

    current = abs(i1)
    if abs(z_in) <= _EPS:
        power_factor = 0.0
    else:
        power_factor = max(min(math.cos(math.atan2(z_in.imag, z_in.real)), 1.0), 0.0)

    torque = _torque_from_params(v_phase, omega_sync, safe_slip, params)
    omega_mech = (1.0 - safe_slip) * omega_sync
    output_power = torque * omega_mech
    input_power = math.sqrt(3.0) * record.rated_voltage_v * current * power_factor
    efficiency = max(min(output_power / max(input_power, _EPS), 1.0), 0.0)

    return OperatingPoint(
        slip=safe_slip,
        current_a=current,
        torque_nm=torque,
        efficiency=efficiency,
        power_factor=power_factor,
    )


def evaluate_vs_slip(
    record: MotorRecord,
    params: EquivalentCircuitParameters,
    slips: Iterable[float],
) -> list[OperatingPoint]:
    return [evaluate_operating_point(record, params, slip) for slip in slips]


def find_breakdown_point(
    record: MotorRecord,
    params: EquivalentCircuitParameters,
    samples: int = 2000,
) -> OperatingPoint:
    start = time.perf_counter()
    best_point = evaluate_operating_point(record, params, 0.001)
    for idx in range(1, samples + 1):
        slip = idx / samples
        point = evaluate_operating_point(record, params, slip)
        if point.torque_nm > best_point.torque_nm:
            best_point = point
    elapsed = time.perf_counter() - start
    _record_performance("breakdown_search", elapsed)
    _record_performance("breakdown_search_evaluations", 0.0, count=max(samples, 0))
    return best_point


def predict_characteristic_points(record: MotorRecord, params: EquivalentCircuitParameters) -> CharacteristicPoints:
    nominal = evaluate_operating_point(record, params, params.slip_rated)
    start = evaluate_operating_point(record, params, 1.0)
    breakdown = find_breakdown_point(record, params)
    return CharacteristicPoints(nominal=nominal, start=start, breakdown=breakdown)
