"""Catalog Markdown -> normalized motor CSV extractor.

This script reads one selected markdown file from `catalog/` and extracts
motor records from markdown tables and numeric blocks found between optional
page ranges (`# Page N` markers). Output rows are normalized to the Guimaraes
schema expected by project validation scripts.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse

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

KGM_TO_NM = 9.80665


def normalize_label(text: str) -> str:
    value = unicodedata.normalize("NFKD", text)
    value = "".join(ch for ch in value if not unicodedata.combining(ch)).lower()
    value = re.sub(r"[^a-z0-9%/()]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


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


def kw_to_w(power_kw: float) -> float:
    return power_kw * 1000.0


def kgfm_to_nm(torque_kgfm: float) -> float:
    return torque_kgfm * KGM_TO_NM


def fmt_float(value: Optional[float], places: int = 5) -> str:
    if value is None:
        return ""
    return f"{value:.{places}f}"


CATALOG_DIR = Path("catalog")
OUTPUT_DIR = Path("data")

TARGET_FIELDS = list(NORMALIZED_HEADERS)

TABLE_FIELDS = [
    "rated_power_kw",
    "rated_power_hp",
    "output_frame",
    "inertia_j_kgm2",
    "max_locked_rotor_time_hot_s",
    "max_locked_rotor_time_cold_s",
    "weight_kg",
    "noise_dba",
    "rated_speed_rpm",
    "eff_100",
    "pf_100",
]

EXTRA_FIELDS = ["source_file", "source_page", "missing_fields", "parse_warnings"]

# Table header aliases to target fields.
HEADER_ALIASES = {
    "motor id": "motor_id",
    "id": "motor_id",
    "voltage": "rated_voltage_v",
    "rated voltage": "rated_voltage_v",
    "tension": "rated_voltage_v",
    "tensao": "rated_voltage_v",
    "potencia kw": "rated_power_w",
    "kw": "rated_power_w",
    "power kw": "rated_power_w",
    "frequency": "frequency_hz",
    "frecuencia": "frequency_hz",
    "frequencia": "frequency_hz",
    "poles": "poles",
    "polos": "poles",
    "corriente nominal in a": "rated_current_a",
    "full load current in a": "rated_current_a",
    "rated current": "rated_current_a",
    "current": "rated_current_a",
    "par nominal tn kgfm": "rated_torque_nm",
    "rated torque": "rated_torque_nm",
    "torque": "rated_torque_nm",
    "rendimiento": "efficiency",
    "efficiency": "efficiency",
    "factor de potencia": "power_factor",
    "power factor": "power_factor",
    "efficiency class": "efficiency_class",
    "clase de rendimiento": "efficiency_class",
    "starting torque category": "starting_torque_category",
    "manufacturer": "manufacturer",
    "fabricante": "manufacturer",
    "group id": "group_id",
    "il/in": "ist_in",
    "corriente con rotor trabado il/in": "ist_in",
    "locked rotor current il/in": "ist_in",
    "ta/tn": "mst_mn",
    "par de arranque ta/tn": "mst_mn",
    "starting torque ta/tn": "mst_mn",
    "tm/tn": "mk_mn",
    "par maximo tm/tn": "mk_mn",
    "breakdown torque tm/tn": "mk_mn",
    "eff 50": "eff_50",
    "efficiency 50": "eff_50",
    "pf 50": "pf_50",
    "power factor 50": "pf_50",
    "eff 75": "eff_75",
    "efficiency 75": "eff_75",
    "pf 75": "pf_75",
    "power factor 75": "pf_75",
}

ROMAN_TO_POLES = {"ii": 2, "iv": 4, "vi": 6, "viii": 8, "x": 10, "xii": 12}
DEFAULT_EFFICIENCY_CLASS = "UNK"
DEFAULT_STARTING_TORQUE_CATEGORY = "Normal torque (IEC design N)"
DEFAULT_MANUFACTURER = "WEG"


def parse_markdown_tables(text: str) -> List[Tuple[List[str], List[List[str]]]]:
    """Return list of (headers, rows) for each markdown table found."""
    lines = text.splitlines()
    tables = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|") and line.count("|") >= 2:
            # find table block
            block = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1

            if len(block) >= 2:
                header_line = block[0]
                # skip separator line if present
                if re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+$", block[1]):
                    data_lines = block[2:]
                else:
                    data_lines = block[1:]

                headers = [h.strip() for h in header_line.strip("|").split("|")]
                rows = [[c.strip() for c in r.strip("|").split("|")] for r in data_lines]
                tables.append((headers, rows))
            continue
        i += 1
    return tables


def page_split(text: str) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    current_page: Optional[int] = None
    current_lines: List[str] = []
    page_re = re.compile(r"^#\s*Page\s+(\d+)\s*$", re.IGNORECASE)
    for raw in text.splitlines():
        m = page_re.match(raw.strip())
        if m:
            if current_page is not None:
                pages.append((current_page, "\n".join(current_lines)))
            current_page = int(m.group(1))
            current_lines = []
            continue
        if current_page is not None:
            current_lines.append(raw)
    if current_page is not None:
        pages.append((current_page, "\n".join(current_lines)))
    return pages


def extract_page_range(text: str, start: Optional[int], end: Optional[int]) -> List[Tuple[int, str]]:
    pages = page_split(text)
    if start is None and end is None:
        return pages
    out: List[Tuple[int, str]] = []
    for page_no, page_text in pages:
        if start is not None and page_no < start:
            continue
        if end is not None and page_no > end:
            continue
        out.append((page_no, page_text))
    return out


def extract_context_from_text(text: str, file_name: str) -> Dict[str, Optional[str]]:
    ntext = normalize_label(text)
    file_norm = normalize_label(file_name)

    manufacturer = ""
    if "weg" in file_norm or " weg " in f" {ntext} ":
        manufacturer = "WEG"

    frequency_hz = ""
    mf = re.search(r"(\d{2})\s*hz", ntext)
    if mf:
        frequency_hz = mf.group(1)

    rated_voltage_v = ""
    mv = re.search(r"(\d{3,4}(?:[.,]\d+)?)\s*v\s*\(\s*\d{2}\s*hz\s*\)", ntext)
    if not mv:
        mv = re.search(r"(\d{3,4}(?:[.,]\d+)?)\s*v", ntext)
    if mv:
        v = parse_decimal(mv.group(1))
        if v is not None:
            rated_voltage_v = f"{v:.0f}"

    poles = ""
    mp = re.search(r"\b(ii|iv|vi|viii|x|xii)\s*polos\b", ntext)
    if mp:
        poles_int = ROMAN_TO_POLES.get(mp.group(1))
        if poles_int is not None:
            poles = str(poles_int)
    if not poles:
        mp2 = re.search(r"\b(\d{1,2})\s*p\b", file_norm)
        if mp2:
            poles = mp2.group(1)

    efficiency_class = ""
    mie = re.search(r"\bie\s*([1-5])\b", ntext)
    if not mie:
        mie = re.search(r"\bie\s*([1-5])\b", file_norm)
    if mie:
        efficiency_class = f"IE{mie.group(1)}"

    starting_torque_category = ""
    if "design n" in ntext or "normal torque" in ntext:
        starting_torque_category = "Normal torque (IEC design N)"

    return {
        "manufacturer": manufacturer,
        "frequency_hz": frequency_hz,
        "rated_voltage_v": rated_voltage_v,
        "poles": poles,
        "efficiency_class": efficiency_class,
        "starting_torque_category": starting_torque_category,
    }


def _is_numeric_token(token: str) -> bool:
    t = token.strip()
    if not t:
        return False
    return parse_decimal(t) is not None


def _clean_token(token: str) -> str:
    return token.strip().strip("|")


def parse_numeric_blocks(page_text: str, page_no: int, ctx: Dict[str, Optional[str]]) -> List[Dict[str, str]]:
    """Parse space-separated motor rows from PDF->MD text blocks.

    Expected order (best effort):
    kW HP frame Tn(kgfm) Il/In Ta/Tn Tm/Tn ... Eff50 Eff75 Eff100 PF50 PF75 PF100 In(A)
    """
    rows: List[Dict[str, str]] = []
    lines = [ln.strip() for ln in page_text.splitlines()]
    for line in lines:
        if not line:
            continue
        if re.search(r"[a-zA-Z]", line) and not re.search(r"\d", line):
            continue

        raw_tokens = [t for t in re.split(r"\s+", line) if t]
        tokens = [_clean_token(t) for t in raw_tokens]
        if len(tokens) < 18:
            continue
        if parse_decimal(tokens[0]) is None or parse_decimal(tokens[1]) is None:
            continue
        if not re.search(r"[a-zA-Z/]", tokens[2]):
            continue

        kW = parse_decimal(tokens[0])
        hp = parse_decimal(tokens[1])
        frame = tokens[2]
        torque_kgm = parse_decimal(tokens[3])
        ist_in = parse_decimal(tokens[4])
        mst_mn = parse_decimal(tokens[5])
        mk_mn = parse_decimal(tokens[6])
        inertia = parse_decimal(tokens[7]) if len(tokens) > 7 else None
        hot_time = parse_decimal(tokens[8]) if len(tokens) > 8 else None
        cold_time = parse_decimal(tokens[9]) if len(tokens) > 9 else None
        weight = parse_decimal(tokens[10]) if len(tokens) > 10 else None
        noise = parse_decimal(tokens[11]) if len(tokens) > 11 else None
        rated_speed = parse_decimal(tokens[12]) if len(tokens) > 12 else None

        eff_50 = parse_decimal(tokens[13]) if len(tokens) > 13 else None
        eff_75 = parse_decimal(tokens[14]) if len(tokens) > 14 else None
        eff_100 = parse_decimal(tokens[15]) if len(tokens) > 15 else None
        pf_50 = parse_decimal(tokens[16]) if len(tokens) > 16 else None
        pf_75 = parse_decimal(tokens[17]) if len(tokens) > 17 else None
        pf_100 = parse_decimal(tokens[18]) if len(tokens) > 18 else None
        rated_current = parse_decimal(tokens[19]) if len(tokens) > 19 else None

        if eff_50 and eff_50 > 1.5:
            eff_50 = eff_50 / 100.0
        if eff_75 and eff_75 > 1.5:
            eff_75 = eff_75 / 100.0
        if eff_100 and eff_100 > 1.5:
            eff_100 = eff_100 / 100.0

        record = {field: "" for field in TARGET_FIELDS}
        record["rated_power_w"] = fmt_float(kw_to_w(kW) if kW is not None else None, 3)
        record["rated_torque_nm"] = fmt_float(kgfm_to_nm(torque_kgm) if torque_kgm is not None else None, 5)
        record["ist_in"] = fmt_float(ist_in)
        record["mst_mn"] = fmt_float(mst_mn)
        record["mk_mn"] = fmt_float(mk_mn)
        record["eff_50"] = fmt_float(eff_50)
        record["pf_50"] = fmt_float(pf_50)
        record["eff_75"] = fmt_float(eff_75)
        record["pf_75"] = fmt_float(pf_75)
        record["efficiency"] = fmt_float(eff_100)
        record["power_factor"] = fmt_float(pf_100)
        record["rated_current_a"] = fmt_float(rated_current)
        record["rated_power_kw"] = fmt_float(kW, 3)
        record["rated_power_hp"] = fmt_float(hp, 3)
        record["output_frame"] = frame
        record["inertia_j_kgm2"] = fmt_float(inertia, 5)
        record["max_locked_rotor_time_hot_s"] = fmt_float(hot_time, 3)
        record["max_locked_rotor_time_cold_s"] = fmt_float(cold_time, 3)
        record["weight_kg"] = fmt_float(weight, 3)
        record["noise_dba"] = fmt_float(noise, 3)
        record["rated_speed_rpm"] = fmt_float(rated_speed, 3)
        record["eff_100"] = fmt_float(eff_100)
        record["pf_100"] = fmt_float(pf_100)

        for key in ("rated_voltage_v", "frequency_hz", "poles", "efficiency_class", "starting_torque_category", "manufacturer"):
            if ctx.get(key):
                record[key] = str(ctx[key])

        record["source_page"] = str(page_no)
        record["parse_warnings"] = ""
        rows.append(record)
    return rows


def _is_frame_token(token: str) -> bool:
    return bool(re.match(r"^\d{2,4}[a-z]?/[a-z]\*?$", token.lower()))


def parse_stacked_numeric_blocks(page_text: str, page_no: int, ctx: Dict[str, Optional[str]]) -> List[Dict[str, str]]:
    """Parse rows when each table value is placed on its own line.

    This format appears in some PDF->markdown outputs where one logical row is
    represented as a sequence of 20 single-token lines.
    """
    tokens: List[str] = []
    for raw in page_text.splitlines():
        t = raw.strip().strip("|")
        if not t or " " in t:
            continue
        if parse_decimal(t) is not None or _is_frame_token(t):
            tokens.append(t)

    rows: List[Dict[str, str]] = []
    i = 0
    while i + 19 < len(tokens):
        # expected row anchor: kW, HP, FRAME, ...
        if parse_decimal(tokens[i]) is None or parse_decimal(tokens[i + 1]) is None or not _is_frame_token(tokens[i + 2]):
            i += 1
            continue
        chunk = tokens[i : i + 20]
        # tail value should be rated current, numeric.
        if parse_decimal(chunk[19]) is None:
            i += 1
            continue

        kW = parse_decimal(chunk[0])
        hp = parse_decimal(chunk[1])
        frame = chunk[2]
        torque_kgm = parse_decimal(chunk[3])
        ist_in = parse_decimal(chunk[4])
        mst_mn = parse_decimal(chunk[5])
        mk_mn = parse_decimal(chunk[6])
        inertia = parse_decimal(chunk[7])
        hot_time = parse_decimal(chunk[8])
        cold_time = parse_decimal(chunk[9])
        weight = parse_decimal(chunk[10])
        noise = parse_decimal(chunk[11])
        rated_speed = parse_decimal(chunk[12])
        eff_50 = parse_decimal(chunk[13])
        eff_75 = parse_decimal(chunk[14])
        eff_100 = parse_decimal(chunk[15])
        pf_50 = parse_decimal(chunk[16])
        pf_75 = parse_decimal(chunk[17])
        pf_100 = parse_decimal(chunk[18])
        rated_current = parse_decimal(chunk[19])

        if eff_50 and eff_50 > 1.5:
            eff_50 = eff_50 / 100.0
        if eff_75 and eff_75 > 1.5:
            eff_75 = eff_75 / 100.0
        if eff_100 and eff_100 > 1.5:
            eff_100 = eff_100 / 100.0

        record = {field: "" for field in TARGET_FIELDS}
        record["rated_power_w"] = fmt_float(kw_to_w(kW) if kW is not None else None, 3)
        record["rated_torque_nm"] = fmt_float(kgfm_to_nm(torque_kgm) if torque_kgm is not None else None, 5)
        record["ist_in"] = fmt_float(ist_in)
        record["mst_mn"] = fmt_float(mst_mn)
        record["mk_mn"] = fmt_float(mk_mn)
        record["eff_50"] = fmt_float(eff_50)
        record["pf_50"] = fmt_float(pf_50)
        record["eff_75"] = fmt_float(eff_75)
        record["pf_75"] = fmt_float(pf_75)
        record["efficiency"] = fmt_float(eff_100)
        record["power_factor"] = fmt_float(pf_100)
        record["rated_current_a"] = fmt_float(rated_current)
        record["rated_power_kw"] = fmt_float(kW, 3)
        record["rated_power_hp"] = fmt_float(hp, 3)
        record["output_frame"] = frame
        record["inertia_j_kgm2"] = fmt_float(inertia, 5)
        record["max_locked_rotor_time_hot_s"] = fmt_float(hot_time, 3)
        record["max_locked_rotor_time_cold_s"] = fmt_float(cold_time, 3)
        record["weight_kg"] = fmt_float(weight, 3)
        record["noise_dba"] = fmt_float(noise, 3)
        record["rated_speed_rpm"] = fmt_float(rated_speed, 3)
        record["eff_100"] = fmt_float(eff_100)
        record["pf_100"] = fmt_float(pf_100)

        for key in ("rated_voltage_v", "frequency_hz", "poles", "efficiency_class", "starting_torque_category", "manufacturer"):
            if ctx.get(key):
                record[key] = str(ctx[key])

        record["source_page"] = str(page_no)
        record["parse_warnings"] = ""
        rows.append(record)
        i += 20

    return rows


def map_table_row(headers: List[str], row: List[str], ctx: Dict[str, Optional[str]]) -> Dict[str, str]:
    mapped: Dict[str, str] = {field: "" for field in TARGET_FIELDS}
    warnings: List[str] = []
    for h, cell in zip(headers, row):
        key = normalize_label(h)
        target = HEADER_ALIASES.get(key)
        if not target:
            continue
        c = cell.strip()
        if target == "rated_power_w":
            val = parse_decimal(c)
            mapped[target] = fmt_float(val * 1000.0 if val is not None else None, 3)
        elif target == "rated_torque_nm":
            val = parse_decimal(c)
            if val is not None and "kgfm" in key:
                val = kgfm_to_nm(val)
            mapped[target] = fmt_float(val)
        elif target in {"efficiency", "power_factor", "eff_50", "eff_75", "pf_50", "pf_75", "ist_in", "mst_mn", "mk_mn", "rated_current_a"}:
            val = parse_decimal(c)
            if target in {"efficiency", "eff_50", "eff_75"} and val is not None and val > 1.5:
                val = val / 100.0
            mapped[target] = fmt_float(val)
        else:
            mapped[target] = c

    for key in ("rated_voltage_v", "frequency_hz", "poles", "efficiency_class", "starting_torque_category", "manufacturer"):
        if not mapped.get(key) and ctx.get(key):
            mapped[key] = str(ctx[key])

    mapped["parse_warnings"] = "; ".join(warnings)
    return mapped


def _file_default_output(single_md: Path, start_page: Optional[int], end_page: Optional[int]) -> Path:
    stem = single_md.stem
    if start_page is not None or end_page is not None:
        s = start_page if start_page is not None else "start"
        e = end_page if end_page is not None else "end"
        name = f"{stem}__p{s}-{e}_extracted.csv"
    else:
        name = f"{stem}_extracted.csv"
    return OUTPUT_DIR / name


def _build_group_id(record: Dict[str, str]) -> str:
    manufacturer = (record.get("manufacturer") or "UNK").upper()
    ie_class = record.get("efficiency_class") or "UNK"
    freq = record.get("frequency_hz") or "UNK"
    return f"{manufacturer}_{ie_class}_{freq}HZ"


def _kw_token_for_id(power_w: str) -> str:
    p = parse_decimal(power_w)
    if p is None:
        return "UNK"
    kw = p / 1000.0
    return f"{kw:.1f}".replace(".", "p") + "kW"


def _synthesize_motor_id(record: Dict[str, str], index: int) -> str:
    manufacturer = (record.get("manufacturer") or "MOTOR").upper()
    ie_class = (record.get("efficiency_class") or "UNK").upper()
    kw_token = _kw_token_for_id(record.get("rated_power_w", ""))
    poles = record.get("poles") or "XP"
    voltage = record.get("rated_voltage_v") or "XV"
    return f"{manufacturer}_{ie_class}_{kw_token}_{poles}P_{voltage}V_{index:03d}"


def _infer_poles_from_speed(speed_rpm_text: str, frequency_hz_text: str) -> str:
    speed = parse_decimal(speed_rpm_text)
    frequency = parse_decimal(frequency_hz_text)
    if speed is None or frequency is None or speed <= 0 or frequency <= 0:
        return ""

    best_poles: Optional[int] = None
    best_error = float("inf")
    for poles in (2, 4, 6, 8, 10, 12):
        synchronous_speed = 120.0 * frequency / poles
        if synchronous_speed <= 0:
            continue
        slip = (synchronous_speed - speed) / synchronous_speed
        # Prefer physically plausible slips, but keep best fallback if needed.
        if slip < -0.02:
            continue
        error = abs(slip)
        if error < best_error:
            best_error = error
            best_poles = poles
    if best_poles is None:
        return ""
    return str(best_poles)


def _merge_non_empty(base: Dict[str, Optional[str]], overrides: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    merged = dict(base)
    for key, value in overrides.items():
        if value is None:
            continue
        text = str(value).strip()
        if text:
            merged[key] = text
    return merged


def _finalize_record(
    record: Dict[str, str],
    index: int,
    source_file: str,
    *,
    default_efficiency_class: str,
    default_starting_torque_category: str,
    default_manufacturer: str,
) -> Dict[str, str]:
    out = {field: record.get(field, "") for field in TARGET_FIELDS}
    for field in TABLE_FIELDS:
        out[field] = record.get(field, "")
    out["source_file"] = source_file
    out["source_page"] = record.get("source_page", "")

    if not out.get("poles", "").strip():
        out["poles"] = _infer_poles_from_speed(
            out.get("rated_speed_rpm", ""),
            out.get("frequency_hz", ""),
        )

    if not out.get("efficiency_class", "").strip():
        out["efficiency_class"] = default_efficiency_class.strip() or DEFAULT_EFFICIENCY_CLASS
    if not out.get("starting_torque_category", "").strip():
        out["starting_torque_category"] = (
            default_starting_torque_category.strip() or DEFAULT_STARTING_TORQUE_CATEGORY
        )
    if not out.get("manufacturer", "").strip():
        out["manufacturer"] = default_manufacturer.strip() or DEFAULT_MANUFACTURER

    if not out["group_id"]:
        out["group_id"] = _build_group_id(out)
    if not out["motor_id"]:
        out["motor_id"] = _synthesize_motor_id(out, index)

    missing = [f for f in TARGET_FIELDS if not (out.get(f) or "").strip()]
    out["missing_fields"] = ",".join(missing)
    out["parse_warnings"] = record.get("parse_warnings", "")
    return out


def scan_catalog(
    single_md: Path,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    *,
    default_efficiency_class: str,
    default_starting_torque_category: str,
    default_manufacturer: str,
) -> List[Dict[str, str]]:
    if not single_md.exists():
        raise FileNotFoundError(f"File not found: {single_md}")

    text = single_md.read_text(encoding="utf-8")
    pages = extract_page_range(text, start_page, end_page)
    if not pages:
        return []

    full_ctx = extract_context_from_text("\n".join(p for _, p in pages), single_md.name)

    extracted_rows: List[Dict[str, str]] = []
    for page_no, page_text in pages:
        page_ctx = _merge_non_empty(full_ctx, extract_context_from_text(page_text, single_md.name))

        # Strategy 1: markdown tables with aliases
        for headers, rows in parse_markdown_tables(page_text):
            mapped_headers = [HEADER_ALIASES.get(normalize_label(h), "") for h in headers]
            if not any(mapped_headers):
                continue
            for row in rows:
                mapped = map_table_row(headers, row, page_ctx)
                mapped["source_page"] = str(page_no)
                extracted_rows.append(mapped)

        # Strategy 2: numerical blocks from PDF->MD conversion.
        extracted_rows.extend(parse_numeric_blocks(page_text, page_no, page_ctx))
        # Strategy 3: stacked one-token-per-line numerical blocks.
        extracted_rows.extend(parse_stacked_numeric_blocks(page_text, page_no, page_ctx))

    finalized: List[Dict[str, str]] = []
    for i, row in enumerate(extracted_rows, 1):
        finalized.append(
            _finalize_record(
                row,
                i,
                single_md.name,
                default_efficiency_class=default_efficiency_class,
                default_starting_torque_category=default_starting_torque_category,
                default_manufacturer=default_manufacturer,
            )
        )
    return finalized


def write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = TARGET_FIELDS + TABLE_FIELDS + EXTRA_FIELDS

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ordered)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in ordered})


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract normalized motor fields from one catalog markdown file")
    parser.add_argument("--file", "-f", required=True, help="Name of the markdown file inside catalog/ to process")
    parser.add_argument("--start-page", type=int, help="Start page number to process (inclusive)")
    parser.add_argument("--end-page", type=int, help="End page number to process (inclusive)")
    parser.add_argument("--output", "-o", help="Optional output CSV path. Defaults to data/<input>_extracted.csv")
    parser.add_argument(
        "--default-efficiency-class",
        default=DEFAULT_EFFICIENCY_CLASS,
        help="Fallback efficiency class used when no class is detected in source rows.",
    )
    parser.add_argument(
        "--default-starting-torque-category",
        default=DEFAULT_STARTING_TORQUE_CATEGORY,
        help="Fallback starting torque category used when missing in source rows.",
    )
    parser.add_argument(
        "--default-manufacturer",
        default=DEFAULT_MANUFACTURER,
        help="Fallback manufacturer used when missing in source rows.",
    )
    args = parser.parse_args()

    if args.start_page is not None and args.end_page is not None and args.start_page > args.end_page:
        raise ValueError("--start-page cannot be greater than --end-page")

    file_name = Path(args.file).name
    if file_name != args.file:
        raise ValueError("--file must be a file name inside catalog/ (no directories)")

    single = CATALOG_DIR / file_name
    output_path = Path(args.output) if args.output else _file_default_output(single, args.start_page, args.end_page)

    rows = scan_catalog(
        single_md=single,
        start_page=args.start_page,
        end_page=args.end_page,
        default_efficiency_class=args.default_efficiency_class,
        default_starting_torque_category=args.default_starting_torque_category,
        default_manufacturer=args.default_manufacturer,
    )
    if not rows:
        print("No matching data found for selected file/page range.")
        return
    write_csv(rows, output_path)

    rows_with_missing = sum(1 for r in rows if r.get("missing_fields"))
    print(f"Extracted {len(rows)} records to {output_path}")
    print(f"Rows with missing target fields: {rows_with_missing}")


if __name__ == "__main__":
    main()
