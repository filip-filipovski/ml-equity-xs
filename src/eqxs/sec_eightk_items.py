"""8-K item codes for every filer, from the already-fetched EDGAR submissions pages.

Input: ``data_cache/equities_xs/raw/sec/submissions/CIK##########.json.gz`` (main page,
``filings.recent``) and ``CIK##########-submissions-NNN.json.gz`` (older pages; the
same column arrays at top level). Fetched 2026-10-10 by ``ingest_equities_xs.py
sec-fetch`` with the declared-contact User-Agent; nothing is downloaded here.

Output (``data_cache/equities_xs/events/eightk_items/quarter=YYYYqN/part.parquet``,
the calendar quarter of ``ts``): one row per (cik, accession) for forms ``8-K`` and
``8-K/A``.

* ``ts`` (UTC): EDGAR ``acceptanceDateTime``. Verified UTC on 2026-10-11 against the
  filing-date rule (800 random filers, 474k filings: acceptances at 18-20 h stamped
  "Z" keep their filing date, those at 21-23 h "Z" in summer roll to the next day, which
  is the 17:30 US/Eastern cut-off in UTC; if the stamp were Eastern wall clock the
  18-20 h bucket would roll). When the page has no acceptance time the fallback is the
  filing date at 22:00 US/Eastern (EDGAR's last acceptance time, never earlier than the
  truth), ``ts_source = 'filed_2200_et'``.
* ``items``: the item tokens as served (list of strings), ``items_raw``, and one
  boolean per item of interest (:data:`ITEMS_OF_INTEREST`).
* Pre-2004 numbering: the 8-K items were renumbered on 2004-08-23 (SEC Rel. 33-8400).
  EDGAR serves the old one-integer codes for older filings ("5", "7", "12", ...).
  :data:`LEGACY_MAP` maps them to the modern item with the same content; the item
  booleans include mapped legacy codes and ``numbering`` says which scheme the filing
  used (``modern`` / ``legacy`` / ``mixed`` / ``none``). Items 1.02, 2.03-2.06, 3.01,
  4.02 and 5.02 officer appointments have NO pre-2004 equivalent (they were created in
  2004), so those booleans are structurally False before 2004-08-23. Legacy item 6
  (director resignation) is mapped to 5.02, which is broader after 2004.

Loaders read only through :func:`eqxs.data_zones.safe_load_parquet` with
``zone_config=EQUITIES_ZONES_PATH``; partitions that end after the equities sealed
start are never opened without an unlock token.
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from eqxs.data_zones import EQUITIES_ZONES_PATH, safe_load_parquet, sealed_start

REPO: Final[Path] = Path(__file__).resolve().parents[2]
SUBMISSIONS_DIR: Final[Path] = REPO / "data_cache" / "equities_xs" / "raw" / "sec" / "submissions"
EIGHTK_ROOT: Final[Path] = REPO / "data_cache" / "equities_xs" / "events" / "eightk_items"

FORMS: Final[frozenset[str]] = frozenset({"8-K", "8-K/A"})
ITEMS_OF_INTEREST: Final[tuple[str, ...]] = (
    "1.01", "1.02", "1.03", "2.01", "2.02", "2.03", "2.04", "2.05", "2.06", "3.01",
    "4.01", "4.02", "5.01", "5.02", "5.03", "7.01", "8.01", "9.01",
)
# Pre-2004-08-23 8-K item -> modern item with the same content.
LEGACY_MAP: Final[dict[str, str]] = {
    "1": "5.01",   # changes in control of registrant
    "2": "2.01",   # acquisition or disposition of assets
    "3": "1.03",   # bankruptcy or receivership
    "4": "4.01",   # changes in registrant's certifying accountant
    "5": "8.01",   # other events
    "6": "5.02",   # resignations of registrant's directors
    "7": "9.01",   # financial statements and exhibits
    "8": "5.03",   # change in fiscal year
    "9": "7.01",   # regulation FD disclosure
    "10": "5.05",  # amendments to / waivers of the code of ethics
    "11": "5.04",  # temporary suspension of trading under employee benefit plans
    "12": "2.02",  # results of operations and financial condition
}
RENUMBER_DATE: Final[pd.Timestamp] = pd.Timestamp("2004-08-23")
_FILE_CIK: Final[re.Pattern[str]] = re.compile(r"CIK(\d{10})")
_MODERN: Final[re.Pattern[str]] = re.compile(r"^\d{1,2}\.\d{2}$")
_LEGACY: Final[re.Pattern[str]] = re.compile(r"^\d{1,2}$")


def item_col(item: str) -> str:
    """``'4.02'`` -> ``'item_4_02'``."""
    return "item_" + item.replace(".", "_")


COLUMNS: Final[tuple[str, ...]] = (
    "cik", "accession", "form", "filed", "accepted", "ts", "ts_source", "period", "items",
    "items_raw", "numbering", "n_items", *(item_col(i) for i in ITEMS_OF_INTEREST),
)


def split_items(raw: str | None) -> list[str]:
    """``'2.02,9.01'`` -> ``['2.02', '9.01']`` (whitespace stripped, order kept,
    duplicates removed). Tokens like ``'Item 2.02'`` are reduced to the code."""
    out: list[str] = []
    for t in str(raw or "").split(","):
        t = re.sub(r"(?i)^item\s*", "", t.strip())
        if t and t not in out:
            out.append(t)
    return out


def numbering_of(tokens: Sequence[str]) -> str:
    m = any(_MODERN.match(t) for t in tokens)
    leg = any(_LEGACY.match(t) for t in tokens)
    return "mixed" if (m and leg) else "modern" if m else "legacy" if leg else "none"


def mapped_items(tokens: Sequence[str]) -> set[str]:
    """Modern item codes of a filing: modern tokens as is, legacy tokens via
    :data:`LEGACY_MAP` (unknown legacy integers are dropped)."""
    out = set()
    for t in tokens:
        if _MODERN.match(t):
            out.add(t)
        elif _LEGACY.match(t) and t in LEGACY_MAP:
            out.add(LEGACY_MAP[t])
    return out


def cik_from_path(path: Path | str) -> int:
    m = _FILE_CIK.search(Path(path).name)
    if not m:
        raise ValueError(f"no CIK in {path}")
    return int(m.group(1))


def _block(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if "filings" in payload:
        return payload["filings"].get("recent", {})
    return payload


def edgar_last_acceptance_utc(filed: pd.Series) -> pd.Series:
    """Filing date at 22:00 US/Eastern, as UTC (conservative fallback for ``ts``)."""
    f = pd.to_datetime(filed, errors="coerce") + pd.Timedelta(hours=22)
    return f.dt.tz_localize("America/New_York", ambiguous="NaT",
                            nonexistent="shift_forward").dt.tz_convert("UTC")


def parse_page(payload: Mapping[str, Any], cik: int) -> pd.DataFrame:
    """One submissions page -> 8-K / 8-K/A rows (all of :data:`COLUMNS`)."""
    b = _block(payload)
    acc = list(b.get("accessionNumber") or [])
    n = len(acc)
    if n == 0:
        return empty_frame()

    def col(k: str, default: Any = None) -> list[Any]:
        v = b.get(k)
        return list(v) if v is not None and len(v) == n else [default] * n

    df = pd.DataFrame({"accession": acc, "form": col("form", ""), "filed": col("filingDate"),
                       "accepted": col("acceptanceDateTime"), "period": col("reportDate"),
                       "items_raw": col("items", "")})
    df = df[df["form"].astype(str).str.strip().isin(FORMS)]
    if df.empty:
        return empty_frame()
    return _finish(df, cik)


def empty_frame() -> pd.DataFrame:
    return _finish(pd.DataFrame({c: pd.Series(dtype="object") for c in
                                 ("accession", "form", "filed", "accepted", "period",
                                  "items_raw")}), 0)


def _finish(df: pd.DataFrame, cik: int) -> pd.DataFrame:
    df = df.copy()
    df["cik"] = np.int64(cik)
    df["form"] = df["form"].astype(str).str.strip().astype("string")
    df["accession"] = df["accession"].astype(str).astype("string")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    acc = df["accepted"].astype("string").str.strip().replace("", pd.NA)
    df["accepted"] = pd.to_datetime(acc, errors="coerce", utc=True, format="ISO8601")
    df["period"] = pd.to_datetime(df["period"].astype("string").replace("", pd.NA),
                                  errors="coerce")
    df["items_raw"] = df["items_raw"].fillna("").astype(str).astype("string")
    toks = [split_items(x) for x in df["items_raw"].astype(str)]
    df["items"] = pd.Series(toks, index=df.index, dtype="object")
    df["numbering"] = pd.Series([numbering_of(t) for t in toks], index=df.index, dtype="string")
    df["n_items"] = pd.Series([len(t) for t in toks], index=df.index, dtype="int16")
    mapped = [mapped_items(t) for t in toks]
    for it in ITEMS_OF_INTEREST:
        df[item_col(it)] = np.fromiter((it in m for m in mapped), dtype=bool, count=len(mapped))
    fb = edgar_last_acceptance_utc(df["filed"]) if len(df) else pd.Series(
        dtype="datetime64[ns, UTC]")
    has = df["accepted"].notna()
    df["ts"] = df["accepted"].where(has, fb)
    df["ts_source"] = pd.Series(np.where(has, "acceptance", "filed_2200_et"), index=df.index,
                                dtype="string")
    return df.loc[:, list(COLUMNS)].reset_index(drop=True)


def parse_file(path: Path | str) -> pd.DataFrame:
    """One ``.json.gz`` submissions page (main or paginated)."""
    p = Path(path)
    with gzip.open(p, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    return parse_page(payload, cik_from_path(p))


def parse_files(paths: Iterable[Path | str]) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Parse many pages; returns (rows, errors). Errors never abort the batch."""
    frames, errs = [], []
    for p in paths:
        try:
            d = parse_file(p)
        except Exception as e:  # noqa: BLE001 - recorded, reported in stats
            errs.append({"file": Path(p).name, "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        if len(d):
            frames.append(d)
    if not frames:
        return empty_frame(), errs
    return pd.concat(frames, ignore_index=True), errs


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (cik, accession): main and paginated pages can overlap; keep the
    row with an acceptance time if any."""
    d = df.assign(_has=df["accepted"].notna()).sort_values(["cik", "accession", "_has"],
                                                           ascending=[True, True, False])
    return d.drop_duplicates(["cik", "accession"]).drop(columns="_has").reset_index(drop=True)


def build_stats(df: pd.DataFrame, *, excluded_forms: Mapping[str, int] | None = None,
                errors: Sequence[Mapping[str, str]] = ()) -> dict[str, Any]:
    """Counts only (no row content): filings by year, per-item counts by year, share with
    an acceptance timestamp, numbering scheme by year, co-registrant accessions."""
    y = pd.to_datetime(df["ts"], utc=True).dt.year
    by_year = df.groupby(y).size()
    items = {it: {str(k): int(v) for k, v in df.groupby(y)[item_col(it)].sum().items()}
             for it in ITEMS_OF_INTEREST}
    acc_share = df.groupby(y)["accepted"].apply(lambda s: float(s.notna().mean()))
    numbering = df.groupby([y, "numbering"]).size().unstack(fill_value=0)
    shared = df.groupby("accession")["cik"].nunique()
    legacy_tokens: dict[str, int] = {}
    for toks in df.loc[df["numbering"].isin(["legacy", "mixed"]), "items"]:
        for t in toks:
            if _LEGACY.match(t):
                legacy_tokens[t] = legacy_tokens.get(t, 0) + 1
    other_tokens: dict[str, int] = {}
    known = set(ITEMS_OF_INTEREST)
    for toks in df["items"]:
        for t in toks:
            if _MODERN.match(t) and t not in known:
                other_tokens[t] = other_tokens.get(t, 0) + 1
    return {
        "rows": int(len(df)), "ciks": int(df["cik"].nunique()),
        "accessions": int(df["accession"].nunique()),
        "accessions_on_multiple_ciks": int((shared > 1).sum()),
        "forms": {str(k): int(v) for k, v in df["form"].value_counts().items()},
        "excluded_8k_variant_forms": dict(excluded_forms or {}),
        "filings_by_year": {str(k): int(v) for k, v in by_year.items()},
        "items_by_year": items,
        "share_with_acceptance_by_year": {str(k): round(v, 4) for k, v in acc_share.items()},
        "share_with_acceptance_all": round(float(df["accepted"].notna().mean()), 4) if len(df)
        else None,
        "numbering_by_year": {str(k): {str(c): int(v) for c, v in r.items()}
                              for k, r in numbering.iterrows()},
        "legacy_token_counts": dict(sorted(legacy_tokens.items(), key=lambda kv: int(kv[0]))),
        "legacy_map": LEGACY_MAP,
        "other_modern_item_counts": dict(sorted(other_tokens.items())),
        "empty_items_rows": int((df["n_items"] == 0).sum()),
        "parse_errors": len(errors),
    }


def quarter_label(ts: pd.Series) -> pd.Series:
    """``YYYYqN`` = calendar quarter of ``ts`` (UTC)."""
    t = pd.to_datetime(ts, utc=True)
    return t.dt.year.astype(str) + "q" + t.dt.quarter.astype(str)


def write_partitions(df: pd.DataFrame, root: Path = EIGHTK_ROOT) -> dict[str, int]:
    """Write ``quarter=YYYYqN/part.parquet`` by calendar quarter of ``ts`` (replaces
    each partition). Quarters, not years, so a sealed start on a quarter boundary
    (2022-01-01, 2024-07-01) never hides the pre-sealed part of a year."""
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for q, g in df.groupby(quarter_label(df["ts"])):
        d = root / f"quarter={q}"
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "part.parquet.tmp"
        g.sort_values(["ts", "cik", "accession"]).reset_index(drop=True).to_parquet(
            tmp, index=False)
        tmp.replace(d / "part.parquet")
        out[str(q)] = int(len(g))
    return out


# ---------------------------------------------------------------------------
# Zone-aware loader
# ---------------------------------------------------------------------------
def _utc(x: pd.Timestamp | str | None) -> pd.Timestamp | None:
    if x is None:
        return None
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def partition_bounds(name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """``year=2010`` -> [2010-01-01, 2011-01-01); ``quarter=2010q3`` -> [2010-07-01,
    2010-10-01) (UTC)."""
    key, val = name.split("=", 1)
    if key == "year":
        y = int(val)
        return pd.Timestamp(f"{y}-01-01", tz="UTC"), pd.Timestamp(f"{y + 1}-01-01", tz="UTC")
    if key == "quarter":
        y, q = val.split("q")
        p0 = pd.Timestamp(f"{int(y)}-{3 * (int(q) - 1) + 1:02d}-01", tz="UTC")
        return p0, p0 + pd.DateOffset(months=3)
    raise ValueError(name)


def year_partitions(root: Path, *, start: pd.Timestamp | None, end: pd.Timestamp | None,
                    sealed: pd.Timestamp | None) -> list[Path]:
    """``year=YYYY`` / ``quarter=YYYYqN`` partition files overlapping [start, end); with
    ``sealed`` given, only partitions that END at or before it (no partition holding a
    sealed date is ever opened)."""
    files = []
    for f in sorted(root.glob("*=*/*.parquet")):
        p0, p1 = partition_bounds(f.parent.name)
        if sealed is not None and p1 > sealed:
            continue
        if start is not None and p1 <= start:
            continue
        if end is not None and p0 >= end:
            continue
        files.append(f)
    return files


def load_eightk_items(*, start: pd.Timestamp | str | None = None,
                      end: pd.Timestamp | str | None = None,
                      items: Iterable[str] | None = None, ciks: Iterable[int] | None = None,
                      columns: Sequence[str] | None = None, allow_sealed: bool = False,
                      unlock_token: str | None = None, root: Path = EIGHTK_ROOT,
                      zone_config: Path | str = EQUITIES_ZONES_PATH) -> pd.DataFrame:
    """8-K rows with ``start <= ts < end`` (end exclusive). ``items``: keep filings with
    ANY of these item codes. Without ``allow_sealed`` + token, no partition ending after
    the equities sealed start is opened."""
    if not root.exists():
        raise FileNotFoundError(f"{root} missing - run scripts/build_eightk_items.py")
    s, e = _utc(start), _utc(end)
    sealed = None if (allow_sealed and unlock_token) else sealed_start(zone_config)
    frames = []
    for f in year_partitions(root, start=s, end=e, sealed=sealed):
        df = safe_load_parquet(f, zone_config=zone_config, allow_sealed=allow_sealed,
                               unlock_token=unlock_token,
                               caller_intent="sec_eightk_items.load_eightk_items")
        t = pd.to_datetime(df["ts"], utc=True)
        m = np.ones(len(df), dtype=bool)
        if s is not None:
            m &= np.asarray(t >= s)
        if e is not None:
            m &= np.asarray(t < e)
        df = df.loc[m]
        if ciks is not None:
            df = df[df["cik"].isin([int(c) for c in ciks])]
        if items is not None:
            cols = [item_col(i) for i in items if item_col(i) in df.columns]
            df = df[df[cols].any(axis=1)] if cols else df.iloc[0:0]
        if columns is not None:
            df = df.loc[:, [c for c in columns if c in df.columns]]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(columns) if columns else list(COLUMNS))
    return pd.concat(frames, ignore_index=True)
