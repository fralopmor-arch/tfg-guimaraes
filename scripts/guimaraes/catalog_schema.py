from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

REQUIRED_NOMINAL_FIELDS: tuple[str, ...] = (
    "rated_voltage_v",
    "rated_power_w",
    "frequency_hz",
    "rated_current_a",
    "rated_torque_nm",
    "efficiency",
    "power_factor",
)

REQUIRED_GROUPING_FIELDS: tuple[str, ...] = (
    "efficiency_class",
    "starting_torque_category",
    "manufacturer",
)

REQUIRED_BLOCK3_FIELDS: tuple[str, ...] = (
    "ist_in",
    "mst_mn",
    "mk_mn",
    "eff_50",
    "eff_75",
    "eff_100",
    "pf_50",
    "pf_75",
    "pf_100",
)

BLOCK3_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "ist_in": ("Ist_In", "ist_in", "Ist_I1", "ist_i1"),
    "mst_mn": ("Mst_Mn", "mst_mn"),
    "mk_mn": ("Mk_Mn", "mk_mn"),
    "eff_50": ("eff_50", "Eff50", "eff50"),
    "eff_75": ("eff_75", "Eff75", "eff75"),
    "eff_100": ("eff_100", "Eff100", "eff100", "efficiency"),
    "pf_50": ("pf_50", "PF50", "pf50"),
    "pf_75": ("pf_75", "PF75", "pf75"),
    "pf_100": ("pf_100", "PF100", "pf100", "power_factor"),
}

OPTIONAL_IDENTIFIER_FIELDS: tuple[str, ...] = (
    "motor_id",
    "group_id",
)


@dataclass(frozen=True)
class MotorRecord:
    motor_id: str
    rated_voltage_v: float
    rated_power_w: float
    frequency_hz: float
    poles: int | None
    pole_pairs: int | None
    rated_current_a: float
    rated_torque_nm: float
    efficiency: float
    power_factor: float
    efficiency_class: str
    starting_torque_category: str
    manufacturer: str
    group_id: str | None = None
    ist_in: float | None = None
    mst_mn: float | None = None
    mk_mn: float | None = None
    eff_50: float | None = None
    eff_75: float | None = None
    eff_100: float | None = None
    pf_50: float | None = None
    pf_75: float | None = None
    pf_100: float | None = None

    @property
    def resolved_pole_pairs(self) -> int:
        if self.pole_pairs is not None:
            return self.pole_pairs
        if self.poles is None:
            raise ValueError(f"Motor {self.motor_id}: poles or pole_pairs is required")
        if self.poles <= 0 or self.poles % 2 != 0:
            raise ValueError(f"Motor {self.motor_id}: poles must be a positive even integer")
        return self.poles // 2

    @property
    def group_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.manufacturer,
            self.efficiency_class,
            self.starting_torque_category,
            f"{self.frequency_hz:.3f}",
            str(self.resolved_pole_pairs),
        )

    @property
    def has_block3_targets(self) -> bool:
        return self.ist_in is not None and self.mst_mn is not None and self.mk_mn is not None


def _to_float(data: dict[str, str], key: str) -> float:
    value = data.get(key, "").strip()
    if value == "":
        raise ValueError(f"Missing required numeric field: {key}")
    return float(value)


def _to_optional_int(data: dict[str, str], key: str) -> int | None:
    value = data.get(key, "").strip()
    if value == "":
        return None
    return int(value)


def _to_text(data: dict[str, str], key: str) -> str:
    value = data.get(key, "").strip()
    if value == "":
        raise ValueError(f"Missing required text field: {key}")
    return value


def _to_optional_float_alias(data: dict[str, str], aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        raw = data.get(alias)
        if raw is None:
            continue
        value = raw.strip()
        if value == "":
            continue
        return float(value)
    return None


def motor_record_from_row(row: dict[str, str], row_index: int) -> MotorRecord:
    motor_id = row.get("motor_id", "").strip() or f"row_{row_index}"
    return MotorRecord(
        motor_id=motor_id,
        rated_voltage_v=_to_float(row, "rated_voltage_v"),
        rated_power_w=_to_float(row, "rated_power_w"),
        frequency_hz=_to_float(row, "frequency_hz"),
        poles=_to_optional_int(row, "poles"),
        pole_pairs=_to_optional_int(row, "pole_pairs"),
        rated_current_a=_to_float(row, "rated_current_a"),
        rated_torque_nm=_to_float(row, "rated_torque_nm"),
        efficiency=_to_float(row, "efficiency"),
        power_factor=_to_float(row, "power_factor"),
        efficiency_class=_to_text(row, "efficiency_class"),
        starting_torque_category=_to_text(row, "starting_torque_category"),
        manufacturer=_to_text(row, "manufacturer"),
        group_id=row.get("group_id", "").strip() or None,
        ist_in=_to_optional_float_alias(row, BLOCK3_FIELD_ALIASES["ist_in"]),
        mst_mn=_to_optional_float_alias(row, BLOCK3_FIELD_ALIASES["mst_mn"]),
        mk_mn=_to_optional_float_alias(row, BLOCK3_FIELD_ALIASES["mk_mn"]),
        eff_50=_to_optional_float_alias(row, BLOCK3_FIELD_ALIASES["eff_50"]),
        eff_75=_to_optional_float_alias(row, BLOCK3_FIELD_ALIASES["eff_75"]),
        eff_100=_to_optional_float_alias(row, BLOCK3_FIELD_ALIASES["eff_100"]),
        pf_50=_to_optional_float_alias(row, BLOCK3_FIELD_ALIASES["pf_50"]),
        pf_75=_to_optional_float_alias(row, BLOCK3_FIELD_ALIASES["pf_75"]),
        pf_100=_to_optional_float_alias(row, BLOCK3_FIELD_ALIASES["pf_100"]),
    )


def _validate_csv_header(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("CSV header is required")

    clean_headers = {name.strip() for name in fieldnames if name is not None}

    required = set(REQUIRED_NOMINAL_FIELDS) | set(REQUIRED_GROUPING_FIELDS)
    missing_required = sorted(name for name in required if name not in clean_headers)
    if missing_required:
        raise ValueError(f"CSV missing required column(s): {', '.join(missing_required)}")

    if "poles" not in clean_headers and "pole_pairs" not in clean_headers:
        raise ValueError("CSV must include at least one of these columns: poles, pole_pairs")

    missing_block3: list[str] = []
    for canonical in REQUIRED_BLOCK3_FIELDS:
        aliases = BLOCK3_FIELD_ALIASES[canonical]
        if not any(alias in clean_headers for alias in aliases):
            missing_block3.append(canonical)
    if missing_block3:
        LOGGER.warning(
            "CSV does not fully cover Block III columns. Missing: %s",
            ", ".join(missing_block3),
        )


def load_catalog_csv(file_path: str | Path) -> list[MotorRecord]:
    path = Path(file_path)
    rows: list[MotorRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_csv_header(reader.fieldnames)
        for index, row in enumerate(reader, start=1):
            record = motor_record_from_row(row, row_index=index)
            rows.append(record)
    LOGGER.info("Loaded %d motor records from %s", len(rows), path)
    return rows


def group_records_by_id_or_tags(records: list[MotorRecord]) -> dict[str, list[MotorRecord]]:
    groups: dict[str, list[MotorRecord]] = {}
    for record in records:
        if record.group_id:
            group_name = record.group_id
        else:
            group_name = "|".join(record.group_key)
        groups.setdefault(group_name, []).append(record)
    LOGGER.info("Created %d group(s) for validation", len(groups))
    return groups


def enforce_grouping_constraints(group_name: str, records: list[MotorRecord]) -> None:
    manufacturers = {entry.manufacturer for entry in records}
    efficiency_classes = {entry.efficiency_class for entry in records}
    starting_categories = {entry.starting_torque_category for entry in records}
    frequencies = {round(entry.frequency_hz, 6) for entry in records}
    pole_pairs = {entry.resolved_pole_pairs for entry in records}

    if len(manufacturers) > 1:
        raise ValueError(f"Invalid mixed group '{group_name}': multiple manufacturers")
    if len(efficiency_classes) > 1:
        raise ValueError(f"Invalid mixed group '{group_name}': multiple efficiency classes")
    if len(starting_categories) > 1:
        raise ValueError(f"Invalid mixed group '{group_name}': multiple starting-torque categories")
    if len(frequencies) > 1:
        raise ValueError(f"Invalid mixed group '{group_name}': multiple frequencies")
    if len(pole_pairs) > 1:
        raise ValueError(f"Invalid mixed group '{group_name}': multiple pole-pair counts")


def as_dict(record: MotorRecord) -> dict[str, Any]:
    return {
        "motor_id": record.motor_id,
        "rated_voltage_v": record.rated_voltage_v,
        "rated_power_w": record.rated_power_w,
        "frequency_hz": record.frequency_hz,
        "poles": record.poles,
        "pole_pairs": record.pole_pairs,
        "rated_current_a": record.rated_current_a,
        "rated_torque_nm": record.rated_torque_nm,
        "efficiency": record.efficiency,
        "power_factor": record.power_factor,
        "efficiency_class": record.efficiency_class,
        "starting_torque_category": record.starting_torque_category,
        "manufacturer": record.manufacturer,
        "group_id": record.group_id,
        "ist_in": record.ist_in,
        "mst_mn": record.mst_mn,
        "mk_mn": record.mk_mn,
        "eff_50": record.eff_50,
        "eff_75": record.eff_75,
        "eff_100": record.eff_100,
        "pf_50": record.pf_50,
        "pf_75": record.pf_75,
        "pf_100": record.pf_100,
    }
