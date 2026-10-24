"""BEA input-output use tables and the SIC -> NAICS -> BEA crosswalks (keyless).

Sources (fetched by ``scripts/ingest_bea_io.py fetch``):

* BEA ``AllTablesIO.zip`` (16.3 MB) and ``AllTablesSUP.zip`` (20.4 MB), static files of
  the Industry Economic Accounts. ``IOUse_Before_Redefinitions_PRO_1997-2023_Summary.xlsx``
  holds one sheet per data year 1997-2023: commodities (rows) x industries (columns),
  producers' prices, millions of dollars, 71 industries. Only ONE vintage is served
  (workbook ``docProps/core.xml`` modified 2024-09-06): every historical year carries the
  revisions of that vintage (design review M10). The vintage date is stored per row
  (``vintage_modified_utc``); this module does not choose a valid_from rule: it stores
  ``data_year`` and the vintage, and ``ts`` = 1 January of (data year + 2), the rule of the
  study-15 v2 design (the edge builder may apply any stricter rule).
* Census 1987-SIC -> 1997-NAICS concordance (official, part indicators) and the 1997
  Economic Census SIC x NAICS bridge (establishments, shipments/receipts, payroll,
  employment per pair), plus the official NAICS 1997->2002->2007->2012 concordances, all
  as mirrored in the Eckert-Fort-Schott-Yang (2021, NBER w26632) replication release on
  GitHub (www.census.gov returns a Cloudflare 403 to the declared-contact User-Agent).
  The EFSY ``full_sic87_naics97`` weights (official + Economic Census + hybrid) are
  stored too.
* The BEA detail -> summary -> related 2017 NAICS definitions from the 2017 detail use
  table's ``NAICS Codes`` sheet.

No openpyxl / xlrd in the environment: :func:`read_xlsx_sheet` is a minimal reader of
the OOXML sheet XML (shared strings, inline strings, numbers).
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from eqxs.data_zones import EQUITIES_ZONES_PATH, safe_load_parquet, sealed_start

REPO: Final[Path] = Path(__file__).resolve().parents[2]
BEA_ROOT: Final[Path] = REPO / "data_cache" / "links" / "bea"
RAW_DIR: Final[Path] = BEA_ROOT / "raw"
USE_ROOT: Final[Path] = BEA_ROOT / "use_summary"
STATIC_DIR: Final[Path] = BEA_ROOT / "static"
MANIFEST: Final[Path] = BEA_ROOT / "manifest.jsonl"
_GH = "https://github.com/fpeckert/website/releases/download/cbp-data/"
SOURCES: Final[dict[str, dict[str, str]]] = {
    "bea_all_tables_io": {
        "url": "https://apps.bea.gov/industry/iTables%20Static%20Files/AllTablesIO.zip",
        "file": "AllTablesIO.zip", "provenance": "BEA Industry Economic Accounts static file"},
    "bea_all_tables_sup": {
        "url": "https://apps.bea.gov/industry/iTables%20Static%20Files/AllTablesSUP.zip",
        "file": "AllTablesSUP.zip", "provenance": "BEA Industry Economic Accounts static file"},
    "efsy_raw_concordances": {
        "url": _GH + "efsy_raw_concordances.zip", "file": "efsy_raw_concordances.zip",
        "provenance": "Eckert, Fort, Schott, Yang (2021) NBER w26632 replication files "
                      "(mirror of Census concordances + 1997 Economic Census bridge)"},
    "efsy_final_concordances": {
        "url": _GH + "efsy_final_concordances.zip", "file": "efsy_final_concordances.zip",
        "provenance": "Eckert, Fort, Schott, Yang (2021) NBER w26632 replication files"},
    "efsy_readme_concordance": {
        "url": _GH + "efsy_readme_concordance_revised.txt",
        "file": "efsy_readme_concordance_revised.txt",
        "provenance": "Eckert, Fort, Schott, Yang (2021) NBER w26632 replication files"},
}
USE_SUMMARY_MEMBER: Final[str] = "IOUse_Before_Redefinitions_PRO_1997-2023_Summary.xlsx"
DETAIL_USE_MEMBERS: Final[dict[int, str]] = {
    2017: "IOUse_Before_Redefinitions_PRO_2017_Detail.xlsx"}
NAICS_SHEET_MEMBER: Final[str] = "IOUse_Before_Redefinitions_PRO_2017_Detail.xlsx"
VALID_FROM_LAG_YEARS: Final[int] = 2

_NS: Final[str] = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_RNS: Final[str] = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


# ---------------------------------------------------------------------------
# Minimal xlsx reader
# ---------------------------------------------------------------------------
def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref)
    n = 0
    for ch in letters.group(0) if letters else "A":
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def xlsx_sheet_names(book: zipfile.ZipFile) -> dict[str, str]:
    """Sheet name -> member path inside the xlsx."""
    wb = ET.fromstring(book.read("xl/workbook.xml"))
    rels = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
    tgt = {r.get("Id"): r.get("Target") for r in rels}
    out = {}
    for s in wb.iter(f"{_NS}sheet"):
        t = tgt[s.get(f"{_RNS}id")]
        t = t.lstrip("/")
        out[s.get("name")] = t if t.startswith("xl/") else "xl/" + t
    return out


def _shared_strings(book: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in book.namelist():
        return []
    root = ET.fromstring(book.read("xl/sharedStrings.xml"))
    return ["".join(t.text or "" for t in si.iter(f"{_NS}t")) for si in root.iter(f"{_NS}si")]


def read_xlsx_sheet(book: zipfile.ZipFile, sheet: str,
                    shared: Sequence[str] | None = None) -> list[list[Any]]:
    """Rows of one sheet as lists (strings / floats / None), ragged rows padded."""
    names = xlsx_sheet_names(book)
    ss = list(shared) if shared is not None else _shared_strings(book)
    root = ET.fromstring(book.read(names[sheet]))
    rows: list[list[Any]] = []
    for r in root.iter(f"{_NS}row"):
        cells: dict[int, Any] = {}
        for c in r.iter(f"{_NS}c"):
            t = c.get("t")
            v = c.find(f"{_NS}v")
            if t == "s" and v is not None:
                val: Any = ss[int(v.text)]
            elif t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter(f"{_NS}t"))
            elif t in ("str", "e"):
                val = v.text if v is not None else None
            elif t == "b":
                val = bool(int(v.text)) if v is not None else None
            else:
                val = float(v.text) if (v is not None and v.text not in (None, "")) else None
            cells[_col_index(c.get("r") or "A")] = val
        idx = int(r.get("r")) - 1 if r.get("r") else len(rows)
        while len(rows) < idx:
            rows.append([])
        rows.append([cells.get(i) for i in range(max(cells) + 1)] if cells else [])
    return rows


def docprops(book: zipfile.ZipFile) -> dict[str, str]:
    """``docProps/core.xml`` created / modified / lastModifiedBy."""
    if "docProps/core.xml" not in book.namelist():
        return {}
    root = ET.fromstring(book.read("docProps/core.xml"))
    out = {}
    for el in root:
        tag = el.tag.split("}")[-1]
        if tag in ("created", "modified", "lastModifiedBy", "creator", "title"):
            out[tag] = (el.text or "").strip()
    return out


def open_member(outer: Path | str, member: str) -> zipfile.ZipFile:
    """Open an xlsx stored inside a zip (read into memory; members are < 5 MB)."""
    with zipfile.ZipFile(outer) as z:
        return zipfile.ZipFile(io.BytesIO(z.read(member)))


# ---------------------------------------------------------------------------
# Use table parsing
# ---------------------------------------------------------------------------
TOTAL_ROW_CODES: Final[dict[str, str]] = {
    "total intermediate": "T005", "total value added": "T006", "total industry output": "T008"}
TOTAL_COL_CODES: Final[dict[str, str]] = {
    "total intermediate": "T001", "total final uses": "T019", "total commodity output": "T007"}


def code_str(x: Any) -> str:
    """Cell -> code string (``211.0`` -> ``'211'``; ``None`` -> ``''``)."""
    if x is None:
        return ""
    if isinstance(x, float):
        return str(int(x)) if x.is_integer() else repr(x)
    return str(x).strip()


def _total_code(name: str, table: Mapping[str, str]) -> str:
    n = (name or "").strip().lower()
    for k, v in table.items():
        if n.startswith(k):
            return v
    return ""


def _find_header(rows: Sequence[Sequence[Any]]) -> int:
    """Index of the row holding the column (industry) codes: the row whose second cell
    is 'Commodities/Industries' (summary layout) or whose first cell is 'Code' (detail
    layout), else the first row with >= 20 non-empty cells after column 1."""
    for i, r in enumerate(rows[:30]):
        if len(r) > 1 and isinstance(r[1], str) and r[1].strip().lower().startswith(
                "commodities/industries"):
            return i
        if r and isinstance(r[0], str) and r[0].strip().lower() == "code":
            return i
    for i, r in enumerate(rows[:30]):
        if sum(code_str(x) != "" for x in r[2:]) >= 20:
            return i
    raise ValueError("no header row")


def parse_use_sheet(rows: Sequence[Sequence[Any]], data_year: int) -> pd.DataFrame:
    """One BEA use sheet -> long frame of every (row, column) value.

    Layout (BEA 2024 vintage): a code row (``Commodities/Industries`` in column 1, column
    codes ``111CA`` ... ``GSLE``, a blank-coded ``Total Intermediate`` column, final uses
    ``F010`` ...), the next row holding names (``IOCode``, ``Name``, ...); data rows:
    commodity code, name, values (``...`` = suppressed -> NaN). Detail layout (2017
    benchmark workbook): a ``Code`` row with the codes, names in the row above, blank
    cells (-> NaN; BEA leaves zero cells blank in the detail tables). Uncoded total rows /
    columns get BEA's standard codes from their names (rows T005 / T006 / T008, columns
    T001 / T019 / T007)."""
    h = _find_header(rows)
    codes = [code_str(x) for x in rows[h]]
    detail = bool(codes) and codes[0].lower() == "code"
    if detail:  # detail layout: names in the row above the codes
        names = [code_str(x) for x in rows[h - 1]] if h > 0 else []
        start = h + 1
    else:
        names = [code_str(x) for x in rows[h + 1]] if h + 1 < len(rows) else []
        start = h + 2 if names and names[:2] == ["IOCode", "Name"] else h + 1
    for j in range(2, len(codes)):
        if not codes[j] and j < len(names):
            codes[j] = _total_code(names[j], TOTAL_COL_CODES)
    recs = []
    for r in rows[start:]:
        if len(r) < 3:
            continue
        rn = code_str(r[1])
        rc = code_str(r[0]) or _total_code(rn, TOTAL_ROW_CODES)
        if not rc:
            continue
        for j in range(2, len(codes)):
            cc = codes[j]
            if not cc:
                continue
            v = r[j] if j < len(r) else None
            if isinstance(v, str):
                t = v.replace(",", "").strip()
                v = np.nan if t in ("...", "---", "(D)", "") else pd.to_numeric(t, errors="coerce")
            recs.append((rc, rn, cc, names[j] if j < len(names) else "",
                         np.nan if v is None else float(v)))
    df = pd.DataFrame(recs, columns=["row_code", "row_name", "col_code", "col_name", "value_musd"])
    df.insert(0, "data_year", np.int16(data_year))
    return df


def is_industry_col(code: str) -> bool:
    """Industry columns (vs totals ``T0..`` and final uses ``F0..``)."""
    return bool(code) and not code.startswith(("T0", "T1")) and not re.match(r"^F\d", code)


def is_commodity_row(code: str) -> bool:
    """Commodity rows (vs totals ``T0..`` and value-added ``V0..``)."""
    return bool(code) and not code.startswith(("T0", "V0", "T1"))


def use_long(raw: pd.DataFrame) -> pd.DataFrame:
    """Commodity x industry flows with shares of the buyer's total intermediate inputs.

    ``share_of_buyer_inputs`` = flow / the buyer column's ``T005`` (Total intermediate
    inputs) row; ``share_of_buyer_inputs_sum`` uses the column sum of commodity rows
    instead (equal up to scrap / noncomparable imports rows, which are commodity rows).
    Diagonal entries (commodity code == industry code) are KEPT; ``is_diagonal`` flags
    them (study 15 v2 zeroes them in the edge builder, not here)."""
    d = raw[raw["col_code"].map(is_industry_col)]
    tot = d[d["row_code"] == "T005"].set_index("col_code")["value_musd"]
    flows = d[d["row_code"].map(is_commodity_row)].copy()
    colsum = flows.groupby("col_code")["value_musd"].sum(min_count=1)
    flows["buyer_total_intermediate_musd"] = flows["col_code"].map(tot)
    flows["share_of_buyer_inputs"] = flows["value_musd"] / flows["buyer_total_intermediate_musd"]
    flows["share_of_buyer_inputs_sum"] = flows["value_musd"] / flows["col_code"].map(colsum)
    flows["is_diagonal"] = flows["row_code"] == flows["col_code"]
    return flows.rename(columns={"row_code": "sell_code", "row_name": "sell_name",
                                 "col_code": "buy_code", "col_name": "buy_name"}).reset_index(
        drop=True)


def attach_vintage(df: pd.DataFrame, vintage_modified: str, *, source_file: str,
                   lag_years: int = VALID_FROM_LAG_YEARS) -> pd.DataFrame:
    """Add ``vintage_modified_utc`` (workbook docProps modified), ``valid_from`` =
    ``ts`` = 1 January of (data_year + ``lag_years``) UTC, ``valid_to`` one year later."""
    out = df.copy()
    vm = pd.Timestamp(vintage_modified)
    vm = vm.tz_localize("UTC") if vm.tzinfo is None else vm.tz_convert("UTC")
    out["vintage_modified_utc"] = vm
    y = out["data_year"].astype(int) + lag_years
    out["ts"] = pd.to_datetime(y.astype(str) + "-01-01", utc=True)
    out["valid_from"] = out["ts"]
    out["valid_to"] = pd.to_datetime((y + 1).astype(str) + "-01-01", utc=True)
    out["vintage_after_valid_from"] = out["vintage_modified_utc"] > out["valid_from"]
    out["source_file"] = source_file
    return out


# ---------------------------------------------------------------------------
# Crosswalks
# ---------------------------------------------------------------------------
def _read_csv_member(zpath: Path | str, member: str) -> list[list[str]]:
    with zipfile.ZipFile(zpath) as z:
        text = z.read(member).decode("latin-1")
    return list(csv.reader(io.StringIO(text.replace("\r\n", "\n").replace("\r", "\n"))))


def parse_census_sic87_naics97(rows: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Census 1987 SIC -> 1997 NAICS concordance (``SIC, Part Indicator, SIC title,
    1997 NAICS, NAICS title``): one row per (sic4, naics97) pair; ``part`` = the SIC is
    split across several NAICS; sector / 2-digit header rows without a NAICS dropped."""
    recs = []
    for r in rows[1:]:
        if len(r) < 4:
            continue
        sic = r[0].strip()
        nai = r[3].strip()
        if not re.fullmatch(r"\d{4}", sic) or not re.fullmatch(r"\d{2,6}", nai):
            continue
        recs.append((sic, r[1].strip().lower() == "pt", r[2].strip(), nai,
                     r[4].strip() if len(r) > 4 else ""))
    df = pd.DataFrame(recs, columns=["sic4", "sic_part", "sic_title", "naics97", "naics97_title"])
    return df.drop_duplicates(["sic4", "naics97"]).reset_index(drop=True)


def parse_ec97_bridge(rows: Sequence[Sequence[str]]) -> pd.DataFrame:
    """1997 Economic Census SIC x NAICS bridge -> one row per (sic4, naics97) with
    establishments, shipments/receipts ($1000), payroll, employment, and each pair's share
    of the SIC's shipments (``ship_share``) and employment (``emp_share``). ``imputed``
    (EFSY's code, 0 = published Census cell, > 0 = imputed by EFSY for a suppressed cell;
    about 10 % of pairs and 2 % of shipments) is carried through (max over duplicates)."""
    hdr = [h.strip() for h in rows[0]]
    df = pd.DataFrame([r for r in rows[1:] if len(r) == len(hdr)], columns=hdr)
    df = df.rename(columns={"SIC_code": "sic4", "NAICS_code": "naics97", "Estabs": "estabs",
                            "Shipments": "shipments_kusd", "Payroll": "payroll_kusd",
                            "Employment": "employment", "SIC_description": "sic_title",
                            "NAICS_description": "naics97_title", "SIC_sector": "sic_sector",
                            "NAICS_sector": "naics_sector", "imputed": "imputed",
                            "taxable": "taxable"})
    df["sic4"] = df["sic4"].str.strip().str.zfill(4)
    df["naics97"] = df["naics97"].str.strip()
    for c in ("estabs", "shipments_kusd", "payroll_kusd", "employment", "imputed"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["sic4"].str.fullmatch(r"\d{4}") & df["naics97"].str.fullmatch(r"\d{6}")]
    g = df.groupby(["sic4", "naics97"], as_index=False).agg(
        estabs=("estabs", "sum"), shipments_kusd=("shipments_kusd", "sum"),
        payroll_kusd=("payroll_kusd", "sum"), employment=("employment", "sum"),
        sic_title=("sic_title", "first"), naics97_title=("naics97_title", "first"),
        sic_sector=("sic_sector", "first"), naics_sector=("naics_sector", "first"),
        imputed=("imputed", "max"))
    tot = g.groupby("sic4")["shipments_kusd"].transform("sum")
    g["ship_share"] = np.where(tot > 0, g["shipments_kusd"] / tot.where(tot > 0), np.nan)
    et = g.groupby("sic4")["employment"].transform("sum")
    g["emp_share"] = np.where(et > 0, g["employment"] / et.where(et > 0), np.nan)
    return g


def parse_naics_concordance(rows: Sequence[Sequence[str]], src: str, dst: str) -> pd.DataFrame:
    """Census NAICS-vintage concordance CSV -> (``src``, ``dst``) pairs of 6-digit codes.
    Columns are found by header names containing the two vintage years."""
    hdr = [h.strip() for h in rows[0]]
    ys, yd = src[-2:], dst[-2:]

    def find(y: str) -> int:
        for i, h in enumerate(hdr):
            hl = h.lower()
            if (y in hl) and ("code" in hl or "naics" in hl) and "title" not in hl:
                return i
        raise ValueError(f"no column for {y} in {hdr}")

    i, j = find(ys), find(yd)
    recs = []
    for r in rows[1:]:
        if len(r) <= max(i, j):
            continue
        a = re.sub(r"\D", "", r[i])
        b = re.sub(r"\D", "", r[j])
        if len(a) == 6 and len(b) == 6:
            recs.append((a, b))
    df = pd.DataFrame(recs, columns=[src, dst]).drop_duplicates().reset_index(drop=True)
    n = df.groupby(src)[dst].transform("count")
    df["n_dst"] = n.astype("int16")
    return df


def parse_bea_naics_sheet(rows: Sequence[Sequence[Any]]) -> pd.DataFrame:
    """BEA 2017 detail ``NAICS Codes`` sheet -> one row per BEA detail industry:
    (sector, summary, u_summary, detail, title, related 2017 NAICS code string).

    The sheet is hierarchical, one level per column: a sector row has its code in column
    0 (title in 1), a summary row its code in column 1 (title in 2), an underlying-summary
    row its code in column 2 (title in 3), a detail row its code in column 3, title in 4,
    notes in 5 and the related 2017 NAICS codes (``"11111-2"``, ``"1121A0"`` style ranges,
    ``"n.a."``) in column 6."""
    recs = []
    sector = summary = usum = None
    for r in rows:
        r = list(r) + [None] * (8 - len(r))
        c = [code_str(x) for x in r[:8]]
        if c[0] and re.fullmatch(r"[0-9A-Z]{1,6}", c[0]):
            sector, summary, usum = c[0], None, None
        elif not c[0] and c[1] and re.fullmatch(r"[0-9A-Z]{2,7}", c[1]):
            summary, usum = c[1], None
        elif not c[0] and not c[1] and c[2] and re.fullmatch(r"[0-9A-Z]{2,7}", c[2]):
            usum = c[2]
        elif not any(c[:3]) and c[3] and re.fullmatch(r"[0-9A-Z]{4,7}", c[3]):
            recs.append((sector, summary, usum, c[3], c[4], c[6]))
    return pd.DataFrame(recs, columns=["bea_sector", "bea_summary", "bea_u_summary",
                                       "bea_detail", "bea_detail_title", "naics17_codes"])


def expand_bea_naics(defs: pd.DataFrame) -> pd.DataFrame:
    """One row per (BEA detail, NAICS-2017 prefix). Ranges ``3361-3`` expand to
    ``3361, 3362, 3363``; ``n.a.`` rows are dropped."""
    recs = []
    for _, r in defs.iterrows():
        for tok in re.split(r"[,;]", str(r["naics17_codes"] or "")):
            t = tok.strip().replace("*", "")
            m = re.fullmatch(r"(\d+)-(\d+)", t)
            if m:
                a, b = m.group(1), m.group(2)
                stem, lo = a[: len(a) - len(b)], int(a[len(a) - len(b):])
                for k in range(lo, int(b) + 1):
                    recs.append((r["bea_summary"], r["bea_detail"], f"{stem}{k:0{len(b)}d}"))
            elif re.fullmatch(r"\d{2,6}", t):
                recs.append((r["bea_summary"], r["bea_detail"], t))
    return pd.DataFrame(recs, columns=["bea_summary", "bea_detail", "naics17_prefix"]
                        ).drop_duplicates().reset_index(drop=True)


def naics_to_bea(naics: str, prefixes: pd.DataFrame) -> list[str]:
    """BEA summary codes for one NAICS code; see :func:`naics_to_bea_method`."""
    return naics_to_bea_method(naics, prefixes)[0]


def naics_to_bea_method(naics: str, prefixes: pd.DataFrame) -> tuple[list[str], str]:
    """BEA summary codes whose NAICS-2017 prefixes match ``naics`` at the LONGEST
    matching prefix length (so ``336111`` -> the ``3361`` detail, not a broader one),
    method ``'prefix'``. A 2012 code is matched against the 2017 prefixes directly (no
    free 2012 -> 2017 concordance is reachable keyless). Where the code was renumbered
    in 2017 (``333911`` -> ``333914``, ``517110`` -> ``517311``) no prefix matches; the
    fallback takes every BEA prefix sharing the code's first k digits, k = 5, 4, 3
    (method ``'truncated_k'``)."""
    best: list[str] = []
    best_len = 0
    for p, s in zip(prefixes["naics17_prefix"], prefixes["bea_summary"], strict=True):
        if naics.startswith(p):
            if len(p) > best_len:
                best_len, best = len(p), [s]
            elif len(p) == best_len and s not in best:
                best.append(s)
    if best:
        return best, "prefix"
    for k in (5, 4, 3):
        hit = sorted({s for p, s in zip(prefixes["naics17_prefix"], prefixes["bea_summary"],
                                        strict=True) if len(p) >= k and p[:k] == naics[:k]})
        if hit:
            return hit, f"truncated_{k}"
    return [], "none"


def chain_naics(start: pd.DataFrame, steps: Sequence[pd.DataFrame], src: str,
                cols: Sequence[str]) -> pd.DataFrame:
    """Compose concordances: ``start`` has column ``src``; each step maps
    ``cols[i] -> cols[i+1]``. Many-to-many; returns every reachable final code."""
    cur = start.rename(columns={src: cols[0]})
    for k, st in enumerate(steps):
        cur = cur.merge(st.loc[:, [cols[k], cols[k + 1]]], on=cols[k], how="left")
    return cur


def resolve_sic_naics(sic: str, sic_naics: pd.DataFrame,
                      share_col: str = "ship_share") -> tuple[pd.DataFrame, str]:
    """1997 NAICS rows of one SEC SIC code with shares summing to 1 within the SIC.

    SEC assigns some group-level codes (``7370``, ``1000``, ``2860``) and some SEC-only
    codes (``3576``, ``6189``, ``6199``) that are not 1987 SIC 4-digit industries.
    Resolution: exact 4-digit match (``level='sic4'``); else every 4-digit SIC of its
    3-digit group (``'sic3_group'``); else, only if the code ends in ``00``, its 2-digit
    group (``'sic2_group'``); group shares are
    re-weighted by each member SIC's shipments (``shipments_kusd``) when present, else
    averaged. ``'none'`` when nothing matches (e.g. SEC 6770 blank checks)."""
    g = sic_naics[sic_naics["sic4"] == sic]
    if len(g):
        return g.loc[:, ["naics97", share_col]].rename(columns={share_col: "share"}), "sic4"
    for level, k, need in (("sic3_group", 3, ""), ("sic2_group", 2, "00")):
        if sic.endswith(need):
            grp = sic_naics[sic_naics["sic4"].str[:k] == sic[:k]]
            if grp.empty:
                continue
            w = grp["shipments_kusd"] if "shipments_kusd" in grp else None
            if w is not None and w.fillna(0).sum() > 0:
                agg = grp.assign(_w=w.fillna(0)).groupby("naics97")["_w"].sum()
            else:
                agg = grp.assign(_w=grp[share_col].fillna(0)).groupby("naics97")["_w"].sum()
            tot = float(agg.sum())
            if tot <= 0:
                continue
            return pd.DataFrame({"naics97": agg.index, "share": (agg / tot).values}), level
    return pd.DataFrame({"naics97": [], "share": []}), "none"


def sic_bea_ambiguity(sics: Iterable[str], sic_naics: pd.DataFrame, naics_bea: pd.DataFrame,
                      *, share_col: str = "ship_share", dominant_min: float = 0.5
                      ) -> pd.DataFrame:
    """Per SEC SIC: the set of BEA summary codes reached through EVERY 1997 NAICS of the
    SIC (``n_bea_any``), the dominant NAICS (largest share) with its share and its BEA
    code(s), ``dominant_ok`` = share >= ``dominant_min`` and exactly one BEA code (the v2
    rule: drop the firm otherwise), plus the BEA share mass (``bea_top_share``: the
    largest BEA code's total share over all the SIC's NAICS). ``naics_bea`` columns:
    ``naics97``, ``bea_summary``."""
    nb = naics_bea.groupby("naics97")["bea_summary"].agg(lambda s: sorted(set(s.dropna())))
    recs = []
    for s in sorted(set(sics)):
        g, level = resolve_sic_naics(s, sic_naics, share_col)
        if g.empty:
            recs.append((s, level, 0, [], None, np.nan, [], False, np.nan, None))
            continue
        allb: set[str] = set()
        mass: dict[str, float] = {}
        for n, sh in zip(g["naics97"], g["share"], strict=True):
            bs = nb.get(n, [])
            allb |= set(bs)
            for bcode in bs:
                mass[bcode] = mass.get(bcode, 0.0) + (float(sh) if pd.notna(sh) else 0.0) / len(bs)
        gg = g.sort_values("share", ascending=False, na_position="last")
        dom = gg.iloc[0]
        dshare = float(dom["share"]) if pd.notna(dom["share"]) else np.nan
        db = nb.get(dom["naics97"], [])
        top = max(mass.items(), key=lambda kv: kv[1]) if mass else (None, np.nan)
        recs.append((s, level, len(allb), sorted(allb), dom["naics97"], dshare, list(db),
                     bool(pd.notna(dshare) and dshare >= dominant_min and len(db) == 1),
                     top[1], top[0]))
    return pd.DataFrame(recs, columns=["sic4", "match_level", "n_bea_any", "bea_any",
                                       "dominant_naics97", "dominant_share", "bea_dominant",
                                       "dominant_ok", "bea_top_share", "bea_top"])


# ---------------------------------------------------------------------------
# Build + loaders
# ---------------------------------------------------------------------------
def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def build_use_tables(raw_zip: Path = RAW_DIR / "AllTablesIO.zip",
                     member: str = USE_SUMMARY_MEMBER, out: Path = USE_ROOT
                     ) -> tuple[pd.DataFrame, dict[str, Any]]:
    book = open_member(raw_zip, member)
    props = docprops(book)
    ss = _shared_strings(book)
    frames = []
    for name in xlsx_sheet_names(book):
        if not re.fullmatch(r"\d{4}", name):
            continue
        raw = parse_use_sheet(read_xlsx_sheet(book, name, ss), int(name))
        frames.append(use_long(raw))
    long = pd.concat(frames, ignore_index=True)
    long = attach_vintage(long, props.get("modified", ""), source_file=member)
    y = long["ts"].dt.year
    parts = {}
    for yr, g in long.groupby(y):
        _write(g.reset_index(drop=True), out / f"year={int(yr)}" / "part.parquet")
        parts[str(int(yr))] = int(len(g))
    meta = {"docprops": props, "member": member, "partitions_by_ts_year": parts}
    return long, meta


DETAIL_ROOT: Final[Path] = BEA_ROOT / "use_detail"
DETAIL_MEMBER: Final[str] = "IOUse_Before_Redefinitions_PRO_2017_Detail.xlsx"


def build_detail_tables(raw_zip: Path = RAW_DIR / "AllTablesIO.zip",
                        member: str = DETAIL_MEMBER, out: Path = DETAIL_ROOT
                        ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Benchmark detail use tables (sheets 2007, 2012, 2017 of the 2017 detail workbook,
    ~400 industries), same long schema and ``ts`` rule as the summary table."""
    book = open_member(raw_zip, member)
    props = docprops(book)
    ss = _shared_strings(book)
    frames = []
    for name in xlsx_sheet_names(book):
        if re.fullmatch(r"\d{4}", name):
            frames.append(use_long(parse_use_sheet(read_xlsx_sheet(book, name, ss), int(name))))
    long = attach_vintage(pd.concat(frames, ignore_index=True), props.get("modified", ""),
                          source_file=member)
    parts = {}
    for yr, g in long.groupby(long["ts"].dt.year):
        _write(g.reset_index(drop=True), out / f"year={int(yr)}" / "part.parquet")
        parts[str(int(yr))] = int(len(g))
    return long, {"docprops": props, "member": member, "partitions_by_ts_year": parts}


def load_use_detail(**kw: Any) -> pd.DataFrame:
    """Benchmark detail use tables; same arguments as :func:`load_use_summary`."""
    kw.setdefault("root", DETAIL_ROOT)
    return load_use_summary(**kw)


def load_use_summary(*, start: pd.Timestamp | str | None = None,
                     end: pd.Timestamp | str | None = None, data_years: Iterable[int] | None = None,
                     columns: Sequence[str] | None = None, allow_sealed: bool = False,
                     unlock_token: str | None = None, root: Path = USE_ROOT,
                     zone_config: Path | str = EQUITIES_ZONES_PATH) -> pd.DataFrame:
    """Long summary use table rows with ``start <= ts < end`` (``ts`` = valid_from).
    No partition ending after the equities sealed start is opened without a token."""
    from eqxs.sec_eightk_items import _utc, year_partitions

    if not root.exists():
        raise FileNotFoundError(f"{root} missing - run scripts/ingest_bea_io.py build")
    s, e = _utc(start), _utc(end)
    sealed = None if (allow_sealed and unlock_token) else sealed_start(zone_config)
    frames = []
    for f in year_partitions(root, start=s, end=e, sealed=sealed):
        df = safe_load_parquet(f, zone_config=zone_config, allow_sealed=allow_sealed,
                               unlock_token=unlock_token, caller_intent="bea_io.load_use_summary")
        t = pd.to_datetime(df["ts"], utc=True)
        m = np.ones(len(df), dtype=bool)
        if s is not None:
            m &= np.asarray(t >= s)
        if e is not None:
            m &= np.asarray(t < e)
        df = df.loc[m]
        if data_years is not None:
            df = df[df["data_year"].isin([int(x) for x in data_years])]
        if columns is not None:
            df = df.loc[:, [c for c in columns if c in df.columns]]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(columns) if columns else [])
    return pd.concat(frames, ignore_index=True)


STATIC_TABLES: Final[tuple[str, ...]] = (
    "sic87_naics97_census", "sic87_naics97_ec97_bridge", "sic87_naics97_efsy",
    "naics97_naics02", "naics02_naics07", "naics07_naics12", "bea_detail_naics17",
    "bea_naics17_prefix", "naics97_bea_summary", "sic4_bea_summary", "sic4_naics97_weights",
)


def load_static_link(name: str, *, root: Path = STATIC_DIR) -> pd.DataFrame:
    """Crosswalk tables (no time axis; read through the tripwire anyway)."""
    if name not in STATIC_TABLES:
        raise ValueError(f"{name!r} not in {STATIC_TABLES}")
    return safe_load_parquet(root / f"{name}.parquet", zone_config=EQUITIES_ZONES_PATH,
                             caller_intent=f"bea_io.load_static_link({name})")


def _provenance(df: pd.DataFrame, source: str, member: str) -> pd.DataFrame:
    out = df.copy()
    out["source"] = source
    out["source_member"] = member
    return out


def build_crosswalks(raw_dir: Path = RAW_DIR, out: Path = STATIC_DIR,
                     sec_sics: Iterable[str] | None = None) -> dict[str, Any]:
    efsy_raw = raw_dir / "efsy_raw_concordances.zip"
    efsy_fin = raw_dir / "efsy_final_concordances.zip"
    pre = "efsy_raw_concordances/"
    census = parse_census_sic87_naics97(
        _read_csv_member(efsy_raw, pre + "1987_SIC_to_1997_NAICS.csv"))
    bridge = parse_ec97_bridge(_read_csv_member(efsy_raw, pre + "SIC_NAICS_concordance.csv"))
    fin_rows = _read_csv_member(efsy_fin, "efsy_final_concordances/full_sic87_naics97.csv")
    efsy = pd.DataFrame(fin_rows[1:], columns=[h.strip() for h in fin_rows[0]])
    efsy = efsy[efsy["sic87"].str.fullmatch(r"\d{4}") & efsy["naics97"].str.fullmatch(r"\d{6}")]
    for c in [c for c in efsy.columns if c.startswith(("weight", "mapping"))]:
        efsy[c] = pd.to_numeric(efsy[c], errors="coerce")
    efsy = efsy.rename(columns={"sic87": "sic4"})
    n97_02 = parse_naics_concordance(
        _read_csv_member(efsy_raw, pre + "1997_NAICS_to_2002_NAICS.csv"),
                                     "naics97", "naics02")
    n02_07 = parse_naics_concordance(_read_csv_member(efsy_raw, pre + "2002_to_2007_NAICS.csv"),
                                     "naics02", "naics07")
    n07_12 = parse_naics_concordance(_read_csv_member(efsy_raw, pre + "2007_to_2012_NAICS.csv"),
                                     "naics07", "naics12")
    book = open_member(raw_dir / "AllTablesIO.zip", NAICS_SHEET_MEMBER)
    sheet = next(n for n in xlsx_sheet_names(book) if "naics" in n.lower())
    defs = parse_bea_naics_sheet(read_xlsx_sheet(book, sheet))
    prefixes = expand_bea_naics(defs)
    # 1997 NAICS -> 2012 NAICS (many-to-many) -> BEA summary (longest prefix)
    n97 = pd.DataFrame({"naics97": sorted(set(census["naics97"].str.ljust(6, "0"))
                                          | set(bridge["naics97"]))})
    ch = chain_naics(n97, [n97_02, n02_07, n07_12], "naics97",
                     ["naics97", "naics02", "naics07", "naics12"])
    ch["naics12"] = ch["naics12"].fillna(ch["naics07"]).fillna(ch["naics02"]).fillna(ch["naics97"])
    cache: dict[str, tuple[list[str], str]] = {}
    recs = []
    for a, b in ch.loc[:, ["naics97", "naics12"]].drop_duplicates().itertuples(index=False):
        if b not in cache:
            cache[b] = naics_to_bea_method(str(b), prefixes)
        for s in cache[b][0]:
            recs.append((a, b, s, cache[b][1]))
    n97_bea = pd.DataFrame(recs, columns=["naics97", "naics12", "bea_summary", "method"]
                           ).drop_duplicates()
    # SIC4 -> BEA summary: economic-census bridge where it covers the SIC, else the
    # official concordance (no weights -> equal, flagged)
    br = bridge.loc[:, ["sic4", "naics97", "ship_share", "emp_share", "shipments_kusd"]].assign(
        weight_source="ec97_ship")
    cen_only = census[~census["sic4"].isin(bridge["sic4"])].copy()
    cen_only["naics97"] = cen_only["naics97"].str.ljust(6, "0")
    k = cen_only.groupby("sic4")["naics97"].transform("count")
    cen_only = cen_only.assign(ship_share=1.0 / k, emp_share=np.nan, shipments_kusd=np.nan,
                               weight_source="census_official_equal")
    sic_naics = pd.concat([br, cen_only.loc[:, ["sic4", "naics97", "ship_share", "emp_share",
                                                "shipments_kusd", "weight_source"]]],
                          ignore_index=True)
    sics = sorted(set(sec_sics)) if sec_sics is not None else sorted(sic_naics["sic4"].unique())
    amb = sic_bea_ambiguity(sics, sic_naics, n97_bea.loc[:, ["naics97", "bea_summary"]])
    amb = amb.merge(sic_naics.groupby("sic4")["weight_source"].first().reset_index(), on="sic4",
                    how="left")
    tables = {
        "sic87_naics97_census": _provenance(census, "census_1987sic_1997naics_via_efsy",
                                            pre + "1987_SIC_to_1997_NAICS.csv"),
        "sic87_naics97_ec97_bridge": _provenance(bridge, "census_ec97_bridge_via_efsy",
                                                 pre + "SIC_NAICS_concordance.csv"),
        "sic87_naics97_efsy": _provenance(efsy, "efsy_2021_final",
                                          "efsy_final_concordances/full_sic87_naics97.csv"),
        "naics97_naics02": _provenance(n97_02, "census_via_efsy",
                                       pre + "1997_NAICS_to_2002_NAICS.csv"),
        "naics02_naics07": _provenance(n02_07, "census_via_efsy", pre + "2002_to_2007_NAICS.csv"),
        "naics07_naics12": _provenance(n07_12, "census_via_efsy", pre + "2007_to_2012_NAICS.csv"),
        "bea_detail_naics17": _provenance(defs, "bea_2017_detail_naics_sheet", NAICS_SHEET_MEMBER),
        "bea_naics17_prefix": _provenance(prefixes, "bea_2017_detail_naics_sheet",
                                          NAICS_SHEET_MEMBER),
        "naics97_bea_summary": _provenance(n97_bea, "derived_chain_97_02_07_12_longest_prefix", ""),
        "sic4_bea_summary": _provenance(amb.assign(bea_any=amb["bea_any"].map(",".join),
                                                   bea_dominant=amb["bea_dominant"].map(",".join)),
                                        "derived_sic_naics97_bea", ""),
        "sic4_naics97_weights": _provenance(sic_naics, "ec97_bridge_else_census_equal", ""),
    }
    for name, df in tables.items():
        _write(df.reset_index(drop=True), out / f"{name}.parquet")
    return {"rows": {k: int(len(v)) for k, v in tables.items()}, "ambiguity": amb,
            "bea_naics_sheet": sheet}


def ambiguity_summary(amb: pd.DataFrame,
                      weights: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Share of SICs mapping to exactly one BEA summary industry (any NAICS of the SIC)
    vs many vs none, and the dominant-NAICS (share >= 0.5) rule; ``weights``: optional
    SIC -> number of filers, for filer-weighted shares."""
    def shares(w: pd.Series) -> dict[str, float]:
        tot = float(w.sum()) or 1.0
        return {"one": round(float(w[amb["n_bea_any"] == 1].sum()) / tot, 4),
                "many": round(float(w[amb["n_bea_any"] > 1].sum()) / tot, 4),
                "none": round(float(w[amb["n_bea_any"] == 0].sum()) / tot, 4),
                "dominant_ok": round(float(w[amb["dominant_ok"]].sum()) / tot, 4)}

    out = {"n_sics": int(len(amb)), "by_sic_count": shares(pd.Series(1.0, index=amb.index)),
           "match_level_counts": {str(k): int(v) for k, v in
                                  amb["match_level"].value_counts().items()}}
    if weights is not None:
        out["by_filer_count"] = shares(amb["sic4"].map(weights).fillna(0).astype(float))
    return out


def build_all(sec_sics: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Build use tables + crosswalks, write stats.json + schema.md."""
    t0 = time.time()
    long, meta = build_use_tables()
    det, dmeta = build_detail_tables()
    if sec_sics is None:
        sec_sics = sec_sic_counts()
    cw = build_crosswalks(sec_sics=list(sec_sics))
    amb = cw.pop("ambiguity")
    years = sorted(int(y) for y in long["data_year"].unique())
    diag = long[long["is_diagonal"]].groupby("data_year")["share_of_buyer_inputs"].mean()
    stats = {
        "use_rows": int(len(long)), "years": years, "vintage": meta["docprops"],
        "use_member": meta["member"], "partitions_by_ts_year": meta["partitions_by_ts_year"],
        "n_industries_by_year": {str(k): int(v) for k, v in
                                 long.groupby("data_year")["buy_code"].nunique().items()},
        "n_commodities_by_year": {str(k): int(v) for k, v in
                                  long.groupby("data_year")["sell_code"].nunique().items()},
        "total_intermediate_musd_by_year": {
            str(k): round(float(v), 1) for k, v in
            long.drop_duplicates(["data_year", "buy_code"]).groupby("data_year")[
                "buyer_total_intermediate_musd"].sum().items()},
        "mean_diagonal_share_by_year": {str(k): round(float(v), 4) for k, v in diag.items()},
        "share_sum_check_max_abs_dev": round(float(
            (long.groupby(["data_year", "buy_code"])["share_of_buyer_inputs"].sum() - 1)
            .abs().max()), 6),
        "detail": {"rows": int(len(det)), "vintage": dmeta["docprops"], "member": dmeta["member"],
                   "data_years": sorted(int(y) for y in det["data_year"].unique()),
                   "n_industries_by_year": {str(k): int(v) for k, v in
                                            det.groupby("data_year")["buy_code"].nunique().items()},
                   "partitions_by_ts_year": dmeta["partitions_by_ts_year"]},
        "crosswalk_rows": cw["rows"], "bea_naics_sheet": cw["bea_naics_sheet"],
        "sic_bea_ambiguity": ambiguity_summary(amb, sec_sics),
        "valid_from_rule": f"1 January of data_year + {VALID_FROM_LAG_YEARS} (UTC); "
                           "vintage_modified_utc stored per row",
        "elapsed_s": round(time.time() - t0, 1),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (BEA_ROOT / "stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True, default=str))
    (BEA_ROOT / "schema.md").write_text(SCHEMA_MD)
    return stats


def sec_sic_counts() -> dict[str, int]:
    """SIC4 -> number of SEC filers (``filer_profiles``, current SIC as served)."""
    p = REPO / "data_cache" / "equities_xs" / "filer_profiles.parquet"
    fp = safe_load_parquet(p, zone_config=EQUITIES_ZONES_PATH,
                           caller_intent="bea_io.sec_sic_counts")
    s = fp["sic"].astype(str).str.strip()
    s = s[s.str.fullmatch(r"\d{3,4}")].str.zfill(4)
    s = s[s != "0000"]
    return {k: int(v) for k, v in s.value_counts().items()}


SCHEMA_MD: Final[str] = """# BEA links schema

## use_summary/year=YYYY/part.parquet  (YYYY = year of ts)
Long BEA summary use table, before redefinitions, producers' prices, $ millions,
`IOUse_Before_Redefinitions_PRO_1997-2023_Summary.xlsx` from AllTablesIO.zip.

| column | meaning |
|---|---|
| data_year | table year (1997-2023) |
| sell_code / sell_name | commodity row (BEA summary commodity code, e.g. `331`) |
| buy_code / buy_name | industry column (BEA summary industry code) |
| value_musd | flow, $ millions (NaN = suppressed) |
| buyer_total_intermediate_musd | the buyer column's T005 (total intermediate inputs) |
| share_of_buyer_inputs | value / T005 |
| share_of_buyer_inputs_sum | value / column sum of commodity rows |
| is_diagonal | sell_code == buy_code (KEPT; zero it in the edge builder) |
| vintage_modified_utc | workbook docProps modified (single vintage, 2024-09-06) |
| ts = valid_from | 1 Jan of data_year + 2, UTC |
| valid_to | 1 Jan of data_year + 3 |
| vintage_after_valid_from | stored vintage post-dates valid_from (revision look-ahead) |

Load: `eqxs.bea_io.load_use_summary(start=..., end=...)`.

## use_detail/year=YYYY/part.parquet
Same schema, benchmark DETAIL use tables 2007 / 2012 / 2017 (~400 industries) from
`IOUse_Before_Redefinitions_PRO_2017_Detail.xlsx` (vintage docProps modified 2024-05-20);
blank detail cells are NaN (BEA leaves zero cells blank). Load:
`eqxs.bea_io.load_use_detail(start=..., end=...)`.

## static/*.parquet (no time axis)
sic87_naics97_census (official concordance, part flags), sic87_naics97_ec97_bridge (1997
Economic Census SIC x NAICS: estabs, shipments, payroll, employment, ship_share, emp_share),
sic87_naics97_efsy (EFSY 2021 weights), naics97_naics02 / naics02_naics07 / naics07_naics12
(official concordances), bea_detail_naics17 (BEA detail -> related 2017 NAICS strings),
bea_naics17_prefix (expanded prefixes), naics97_bea_summary (chained), sic4_bea_summary
(per SEC SIC4: BEA codes reachable, dominant NAICS and share, dominant_ok = share >= 0.5 and one
BEA code). Every table carries `source` and `source_member`.
Load: `eqxs.bea_io.load_static_link(name)`.
"""
