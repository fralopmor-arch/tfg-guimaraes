from __future__ import annotations

import csv
import math
import re
import argparse
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

NORMALIZED_HEADERS = [
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
    "group_id",
    "ist_in",
    "mst_mn",
    "mk_mn",
    "eff_50",
    "pf_50",
    "eff_75",
    "pf_75",
]


def parse_decimal(text: str) -> Optional[float]:
    value = text.strip()
    if not value:
        return None

    value = value.replace(" ", "").replace("%", "")
    if value.count(",") == 1 and value.count(".") > 1:
        value = value.replace(".", "").replace(",", ".")
    else:
        value = value.replace(",", ".")

    if re.match(r"^[-+]?\d*\.?\d+$", value) is None:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def extract_first_number(text: str) -> float:
    match = re.search(r"-?\d+(?:[\.,]\d+)?", text)
    if match is None:
        raise ValueError(f"No numeric value found in: {text}")
    parsed = parse_decimal(match.group(0))
    if parsed is None:
        raise ValueError(f"Unable to parse numeric value: {text}")
    return parsed


def to_decimal_percent(text: str) -> float:
    return extract_first_number(text) / 100.0


def kw_to_w(power_kw: float) -> float:
    return power_kw * 1000.0

SOURCE_PATH = Path("catalog/WEG_IE4_14_Mar_2026.xls")
OUTPUT_PATH = Path("data/catalog_weg_ie4_xls_raw_full.csv")
NORMALIZED_OUTPUT_PATH = Path("data/catalog_weg_ie4_xls_normalized_full.csv")

@dataclass
class Cell:
    text: str
    colspan: int = 1
    rowspan: int = 1


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[Cell]]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._current_table: list[list[Cell]] | None = None
        self._current_row: list[Cell] | None = None
        self._cell_parts: list[str] = []
        self._cell_colspan = 1
        self._cell_rowspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = dict(attrs)
        t = tag.lower()
        if t == "table":
            self._in_table = True
            self._current_table = []
            return
        if not self._in_table:
            return
        if t == "tr":
            self._in_row = True
            self._current_row = []
        elif t in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._cell_parts = []
            self._cell_colspan = _safe_int(attrs_map.get("colspan"), default=1)
            self._cell_rowspan = _safe_int(attrs_map.get("rowspan"), default=1)
        elif t == "br" and self._in_cell:
            self._cell_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"td", "th"} and self._in_cell and self._current_row is not None:
            text = _normalize_text("".join(self._cell_parts))
            self._current_row.append(
                Cell(
                    text=text,
                    colspan=max(1, self._cell_colspan),
                    rowspan=max(1, self._cell_rowspan),
                )
            )
            self._in_cell = False
        elif t == "tr" and self._in_row:
            if self._current_table is not None and self._current_row is not None:
                self._current_table.append(self._current_row)
            self._in_row = False
            self._current_row = None
        elif t == "table" and self._in_table:
            if self._current_table is not None:
                self.tables.append(self._current_table)
            self._in_table = False
            self._current_table = None

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)


def _safe_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_text(value: str) -> str:
    text = unescape(value)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _expand_table(rows: list[list[Cell]]) -> list[list[str]]:
    if not rows:
        return []

    ncols = max(sum(cell.colspan for cell in row) for row in rows)
    active_spans: dict[int, tuple[int, str]] = {}
    expanded: list[list[str]] = []

    for row in rows:
        current: list[str | None] = [None] * ncols

        for col, (remaining, text) in list(active_spans.items()):
            if remaining > 0:
                current[col] = text
                active_spans[col] = (remaining - 1, text)
            if active_spans[col][0] == 0:
                del active_spans[col]

        col_ptr = 0
        for cell in row:
            while col_ptr < ncols and current[col_ptr] is not None:
                col_ptr += 1
            if col_ptr >= ncols:
                break

            span_end = min(ncols, col_ptr + cell.colspan)
            for c in range(col_ptr, span_end):
                current[c] = cell.text
                if cell.rowspan > 1:
                    active_spans[c] = (cell.rowspan - 1, cell.text)
            col_ptr = span_end

        expanded.append(["" if value is None else value for value in current])

    return expanded


def _sanitize_header(text: str) -> str:
    cleaned = text.lower().strip()
    cleaned = re.sub(r"[%]", " pct ", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "column"


def _build_headers(row1: list[str], row2: list[str]) -> list[str]:
    headers: list[str] = []
    counts: dict[str, int] = {}

    for h1, h2 in zip(row1, row2):
        top = _normalize_text(h1)
        sub = _normalize_text(h2)
        if sub and top and top != sub:
            combined = f"{top} {sub}"
        elif sub:
            combined = sub
        else:
            combined = top

        key = _sanitize_header(combined)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > 1:
            key = f"{key}_{counts[key]}"
        headers.append(key)

    return headers


def _select_main_table(tables: list[list[list[Cell]]]) -> list[list[Cell]]:
    candidates: list[list[list[Cell]]] = []
    for table in tables:
        if not table:
            continue
        text_blob = " ".join(cell.text.lower() for row in table for cell in row)
        if "rated current" in text_blob and "power factor" in text_blob:
            candidates.append(table)
    if candidates:
        return max(candidates, key=lambda t: len(t))
    if not tables:
        raise RuntimeError("No HTML tables found in source file.")
    return max(tables, key=lambda t: len(t))


def parse_source(source_path: Path) -> tuple[list[str], list[list[str]], int]:
    parser = _TableParser()
    parser.feed(source_path.read_text(encoding="utf-8", errors="ignore"))
    table = _select_main_table(parser.tables)
    expanded = _expand_table(table)
    if len(expanded) < 3:
        raise RuntimeError("Main table does not have enough rows for 2-level header + body.")

    headers = _build_headers(expanded[0], expanded[1])
    body = [row for row in expanded[2:] if any(cell.strip() for cell in row)]
    dropped_empty = len(expanded[2:]) - len(body)

    return headers, body, dropped_empty


def write_csv(headers: list[str], rows: list[list[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def _parse_frequency_hz(text: str) -> float:
    return extract_first_number(text)


def _parse_voltage_and_current(voltage_text: str, current_text: str) -> tuple[float, float]:
    voltage_values = [
        parse_decimal(token)
        for token in re.findall(r"\d+(?:[\.,]\d+)?", voltage_text)
        if parse_decimal(token) is not None
    ]
    current_values = [
        parse_decimal(token)
        for token in re.findall(r"\d+(?:[\.,]\d+)?", current_text)
        if parse_decimal(token) is not None
    ]
    if not voltage_values or not current_values:
        raise ValueError(
            f"Unable to parse voltage/current pair from '{voltage_text}' and '{current_text}'"
        )

    target_voltage = max(voltage_values)
    selected_current = current_values[min(len(current_values), len(voltage_values)) - 1]
    if len(voltage_values) == len(current_values):
        selected_current = current_values[voltage_values.index(target_voltage)]
    elif len(current_values) == 1:
        selected_current = current_values[0]

    return target_voltage, selected_current


def _parse_ratio_percent(text: str) -> float:
    return to_decimal_percent(text)


def _build_motor_id(power_kw: float, poles: int, voltage_v: float, row_index: int) -> str:
    power_token = str(power_kw).replace(".", "p")
    return f"WEG_IE4_{power_token}kW_{poles}P_{int(round(voltage_v))}V_{row_index:03d}"


def _row_value_or_default(row: dict[str, str], keys: list[str], default: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if value is None:
            continue
        normalized = value.strip()
        if normalized:
            return normalized
    return default


def _build_group_id(manufacturer: str, efficiency_class: str, frequency_hz: float) -> str:
    manufacturer_token = re.sub(r"[^A-Za-z0-9]+", "_", manufacturer.upper()).strip("_") or "GEN"
    class_token = re.sub(r"[^A-Za-z0-9]+", "", efficiency_class.upper()) or "UNK"
    if abs(frequency_hz - round(frequency_hz)) < 1e-9:
        freq_token = str(int(round(frequency_hz)))
    else:
        freq_token = str(frequency_hz).replace(".", "p")
    return f"{manufacturer_token}_{class_token}_{freq_token}HZ"


def _rated_torque_nm(power_w: float, speed_rpm: float) -> float:
    omega = 2.0 * math.pi * speed_rpm / 60.0
    if omega <= 0.0:
        raise ValueError(f"Invalid full-load speed: {speed_rpm}")
    return power_w / omega


def normalize_ie4_csv(
    raw_csv_path: Path,
    normalized_csv_path: Path,
    *,
    default_efficiency_class: str,
    default_starting_torque_category: str,
    default_manufacturer: str,
    default_group_id: Optional[str],
) -> int:
    with raw_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    normalized_rows: list[dict[str, str]] = []
    for row_index, row in enumerate(rows, start=1):
        frequency_hz = _parse_frequency_hz(row["frequency"])
        power_kw = extract_first_number(row["output_kw"])
        power_w = kw_to_w(power_kw)
        poles = int(extract_first_number(row["poles"]))
        voltage_v, rated_current_a = _parse_voltage_and_current(
            row["voltage"], row["rated_current"]
        )
        speed_rpm = extract_first_number(row["full_load_speed"])
        torque_nm = _rated_torque_nm(power_w, speed_rpm)
        mst_mn = _parse_ratio_percent(row["locked_rotor_torque"])
        mk_mn = _parse_ratio_percent(row["breakdown_torque"])
        ist_in = extract_first_number(row["locked_rotor_current_il_in"])
        efficiency_class = _row_value_or_default(
            row,
            ["efficiency_class", "efficiency class", "clase de eficiencia"],
            default_efficiency_class,
        )
        starting_torque_category = _row_value_or_default(
            row,
            ["starting_torque_category", "starting torque category", "categoria de par de arranque"],
            default_starting_torque_category,
        )
        manufacturer = _row_value_or_default(
            row,
            ["manufacturer", "fabricante"],
            default_manufacturer,
        )
        group_id = _row_value_or_default(row, ["group_id", "group id"], "")
        if not group_id:
            group_id = default_group_id or _build_group_id(
                manufacturer=manufacturer,
                efficiency_class=efficiency_class,
                frequency_hz=frequency_hz,
            )

        normalized_rows.append(
            {
                "motor_id": _build_motor_id(power_kw, poles, voltage_v, row_index),
                "rated_voltage_v": f"{voltage_v:.0f}",
                "rated_power_w": f"{power_w:.3f}",
                "frequency_hz": f"{frequency_hz:.1f}",
                "poles": str(poles),
                "rated_current_a": f"{rated_current_a:.5f}",
                "rated_torque_nm": f"{torque_nm:.5f}",
                "efficiency": f"{to_decimal_percent(row['efficiency_pct_100_pct']):.5f}",
                "power_factor": f"{extract_first_number(row['power_factor_100_pct']):.5f}",
                "efficiency_class": efficiency_class,
                "starting_torque_category": starting_torque_category,
                "manufacturer": manufacturer,
                "group_id": group_id,
                "ist_in": f"{ist_in:.5f}",
                "mst_mn": f"{mst_mn:.5f}",
                "mk_mn": f"{mk_mn:.5f}",
                "eff_50": f"{to_decimal_percent(row['efficiency_pct_50_pct']):.5f}",
                "pf_50": f"{extract_first_number(row['power_factor_50_pct']):.5f}",
                "eff_75": f"{to_decimal_percent(row['efficiency_pct_75_pct']):.5f}",
                "pf_75": f"{extract_first_number(row['power_factor_75_pct']):.5f}",
            }
        )

    normalized_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with normalized_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NORMALIZED_HEADERS)
        writer.writeheader()
        writer.writerows(normalized_rows)

    return len(normalized_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and normalize WEG IE4 catalog data")
    parser.add_argument("--source-xls", type=Path, default=SOURCE_PATH)
    parser.add_argument("--raw-output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--normalized-output", type=Path, default=NORMALIZED_OUTPUT_PATH)
    parser.add_argument(
        "--efficiency-class",
        default="IE4",
        help="Fallback value used when efficiency class is missing in source rows.",
    )
    parser.add_argument(
        "--starting-torque-category",
        default="Normal torque (IEC design N)",
        help="Fallback value used when starting torque category is missing in source rows.",
    )
    parser.add_argument(
        "--manufacturer",
        default="WEG",
        help="Fallback manufacturer used when source rows do not provide one.",
    )
    parser.add_argument(
        "--group-id",
        default="",
        help=(
            "Optional fixed group_id for normalized rows. "
            "When omitted, it is auto-generated from manufacturer, efficiency class and frequency."
        ),
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip XLS extraction and only normalize an existing raw CSV",
    )
    args = parser.parse_args()

    if not args.skip_extract:
        if not args.source_xls.exists():
            raise FileNotFoundError(f"Missing source file: {args.source_xls}")
        headers, rows, dropped_empty = parse_source(args.source_xls)
        write_csv(headers, rows, args.raw_output)
        print(f"CSV written: {args.raw_output}")
        print(f"Columns: {len(headers)}")
        print(f"Rows: {len(rows)}")
        print(f"Dropped empty/spacer rows: {dropped_empty}")

    if not args.raw_output.exists():
        raise FileNotFoundError(f"Missing raw CSV for normalization: {args.raw_output}")

    normalized_count = normalize_ie4_csv(
        args.raw_output,
        args.normalized_output,
        default_efficiency_class=args.efficiency_class,
        default_starting_torque_category=args.starting_torque_category,
        default_manufacturer=args.manufacturer,
        default_group_id=args.group_id.strip() or None,
    )
    print(f"Normalized CSV written: {args.normalized_output}")
    print(f"Normalized rows: {normalized_count}")


if __name__ == "__main__":
    main()
