"""Earnings calendar for the forward option-snapshot universe (pure helpers, no network).

History comes from SEC EDGAR: every 8-K whose items include **2.02** (Results of
Operations), its ``acceptanceDateTime`` mapped to ET and classified BMO / AMC /
INTRADAY, then one event per (ticker, calendar quarter of the announcement) — the
stage-1 scheduled-event study's rule, reused verbatim from
the 8-K item 2.02 parsing functions below (``parse_edgar_8k_202``,
``acceptance_to_et``, ``classify_timing``, ``dedupe_earnings``). The calendar quarter
of the announcement date is the stage-1 proxy for "first per fiscal quarter".
Guidance / deliveries 8-Ks are skipped through the same documented mechanism as
stage 1: explicit per-accession overrides plus the Tesla rule. Stage 1 (2015-2020)
identified Tesla's production & deliveries releases by item 7.01; from 2023 they
are filed as plain 2.02 + 9.01 like the earnings 8-Ks, so the rule here also drops
any TSLA 2.02 8-K accepted on days 1-7 of a quarter's first month (deliveries come
out ~2 days into the quarter; Tesla has reported earnings no earlier than the
15th since 2012).

Nothing is silently resolved beyond that: :func:`flag_ambiguous` marks rows a human
should eyeball (several 2.02 8-Ks in one quarter, INTRADAY acceptance, weekend
announcement, no 9.01 exhibit, implausible spacing between consecutive kept events).
Two of those flags (``multi_202_in_quarter``, ``cadence_prefers``) are LOOK-AHEAD for a
kept row: they are set from 2.02 8-Ks filed later in the same quarter. Studies select
"clean" events with :func:`clean_event_mask`, which by default uses only the flags known
at formation (:data:`PIT_FLAGS`); the stored ``ambiguous`` string is the record and is
never rewritten.

Upcoming dates: :func:`next_expected_dates` reads yfinance ``earnings_dates`` — a
best-effort, UNVERIFIED expectation, stored in a separately-labelled column and never
merged into the 8-K history.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd


# 8-K item 2.02 parsing


def _edgar_block(sub: Mapping) -> Mapping:
    """``filings.recent`` of the main submissions JSON, or the top-level arrays of an
    older paginated ``CIK##########-submissions-NNN.json`` file."""
    if "filings" in sub:
        return sub["filings"]["recent"]
    return sub


def parse_edgar_8k_202(sub: Mapping) -> pd.DataFrame:
    """8-K filings whose ``items`` include 2.02 (Results of Operations) from an EDGAR
    submissions JSON (main or paginated). Columns ``accession, filing_date,
    acceptance_raw, items``."""
    b = _edgar_block(sub)
    n = len(b.get("form", []))
    rows = []
    for i in range(n):
        if b["form"][i] != "8-K":
            continue
        items = str(b["items"][i] or "")
        if "2.02" not in [x.strip() for x in items.split(",")]:
            continue
        rows.append(
            {
                "accession": b["accessionNumber"][i],
                "filing_date": pd.Timestamp(b["filingDate"][i]),
                "acceptance_raw": b["acceptanceDateTime"][i],
                "items": items,
            }
        )
    return pd.DataFrame(rows, columns=["accession", "filing_date", "acceptance_raw", "items"])


def acceptance_to_et(acceptance_raw: str) -> pd.Timestamp:
    """EDGAR ``acceptanceDateTime`` (ISO, suffixed Z) -> America/New_York timestamp.

    The study treats the stamp as UTC (checked in the report against BMO filers JPM/XOM,
    whose stamps convert to 06:xx-08:xx ET inside EDGAR's 06:00-22:00 ET filing window).
    """
    ts = pd.Timestamp(acceptance_raw)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.tz_convert("America/New_York")


def classify_timing(acc_et: pd.Timestamp) -> str:
    """``BMO`` (before 09:30 ET), ``AMC`` (at/after 16:00 ET) or ``INTRADAY``."""
    t = acc_et.hour * 60 + acc_et.minute
    if t < 9 * 60 + 30:
        return "BMO"
    if t >= 16 * 60:
        return "AMC"
    return "INTRADAY"


def dedupe_earnings(df: pd.DataFrame, overrides: Mapping[str, str] | None = None) -> pd.DataFrame:
    """One earnings event per (ticker, calendar quarter of the announcement date).

    Keeps the FIRST 2.02 8-K per quarter (spec) unless ``overrides`` maps an accession to a
    reason for dropping it (documented pre-announcements, e.g. a guidance cut), in which
    case the next filing in that quarter is kept. Adds ``keep`` and ``drop_reason``.
    ``df`` needs ``ticker, accession, announce_date``.
    """
    overrides = dict(overrides or {})
    out = df.sort_values(["ticker", "announce_date", "accession"]).copy()
    out["quarter"] = out["announce_date"].dt.to_period("Q").astype(str)
    out["keep"] = False
    out["drop_reason"] = ""
    for _, g in out.groupby(["ticker", "quarter"], sort=False):
        kept = False
        for idx in g.index:
            acc = out.at[idx, "accession"]
            if acc in overrides:
                out.at[idx, "drop_reason"] = f"override: {overrides[acc]}"
                continue
            if not kept:
                out.at[idx, "keep"] = True
                kept = True
            else:
                out.at[idx, "drop_reason"] = "duplicate_in_quarter"
    return out


__all__ = [
    "CSV_COLUMNS",
    "DELIVERIES_RULE_TICKERS",
    "GAP_LONG_DAYS",
    "GAP_SHORT_DAYS",
    "build_ticker_events",
    "coverage_per_year",
    "flag_ambiguous",
    "is_deliveries_release",
    "cik_from_entity_hits",
    "clean_event_mask",
    "ambiguity_flag_names",
    "LOOKAHEAD_FLAGS",
    "PIT_FLAGS",
    "ALL_FLAGS",
    "next_expected_dates",
    "sec_ticker_variants",
    "submissions_match_ticker",
]

GAP_SHORT_DAYS = 50
GAP_LONG_DAYS = 120

#: Flags set from information that exists only AFTER a kept event: the other 2.02 8-Ks of
#: the same calendar quarter. ``dedupe_earnings`` keeps the FIRST candidate of the quarter,
#: so for a kept row these always look forward (red team 2026-10-10, finding F1: 12,088 of
#: 222,759 kept TRAIN events, 5.4 %, are flagged only by these).
LOOKAHEAD_FLAGS: tuple[str, ...] = ("multi_202_in_quarter", "cadence_prefers")
#: Flags known at the event's own acceptance (``intraday_acceptance``, ``weekend_announce``,
#: ``no_9.01_exhibit`` describe the filing itself; ``gap_short`` / ``gap_long`` look back to
#: the previous kept event).
PIT_FLAGS: tuple[str, ...] = ("intraday_acceptance", "weekend_announce", "no_9.01_exhibit",
                              "gap_short", "gap_long")
ALL_FLAGS: tuple[str, ...] = LOOKAHEAD_FLAGS + PIT_FLAGS
DELIVERIES_RULE_TICKERS = frozenset({"TSLA"})
DELIVERIES_REASON = (
    "production & deliveries release (2.02 8-K carrying item 7.01, or accepted on days 1-7 "
    "of a quarter's first month), not the earnings report"
)


def is_deliveries_release(items: object, announce_date: pd.Timestamp) -> bool:
    """Tesla production & deliveries 8-K heuristic (see module docstring)."""
    d = pd.Timestamp(announce_date)
    return "7.01" in _items(items) or (d.month in (1, 4, 7, 10) and d.day <= 7)
CSV_COLUMNS = (
    "ticker", "cik", "accession", "filing_date", "acceptance_raw", "acceptance_et", "timing",
    "announce_date", "quarter", "items", "keep", "drop_reason", "ambiguous", "filer_cik",
    "source_url",
    "yf_expected_next_unverified",
)


def sec_ticker_variants(ticker: str) -> list[str]:
    """Spellings EDGAR may use for a share class (``BRK.B`` -> ``BRK-B``, ``BRKB``)."""
    t = ticker.upper().strip()
    out = [t]
    for sep in (".", "/"):
        if sep in t:
            out += [t.replace(sep, "-"), t.replace(sep, "")]
    return list(dict.fromkeys(out))


def cik_from_entity_hits(payload: Mapping[str, Any], ticker: str) -> int | None:
    """CIK whose ``tickers`` list contains ``ticker`` exactly, from an EDGAR
    ``efts.sec.gov/LATEST/search-index?keysTyped=<T>`` response (the keyless lookup;
    ``www.sec.gov/files/company_tickers.json`` refuses clients without an e-mail in
    the User-Agent). Highest ``rank`` wins when several entities match."""
    want = set(sec_ticker_variants(ticker))
    best: tuple[float, int] | None = None
    for h in (payload.get("hits") or {}).get("hits") or []:
        src = h.get("_source") or {}
        tks = {x.strip().upper() for x in str(src.get("tickers") or "").split(",") if x.strip()}
        if tks & want:
            try:
                cand = (float(src.get("rank") or 0.0), int(h["_id"]))
            except (KeyError, TypeError, ValueError):
                continue
            if best is None or cand > best:
                best = cand
    return best[1] if best else None


def submissions_match_ticker(main: Mapping[str, Any], ticker: str) -> bool:
    """True when the EDGAR submissions JSON lists ``ticker`` (any share-class spelling)."""
    tks = {str(x).upper() for x in main.get("tickers") or []}
    return bool(tks & set(sec_ticker_variants(ticker)))


def _items(s: object) -> set[str]:
    return {x.strip() for x in str(s or "").split(",") if x.strip()}


def build_ticker_events(
    subs: Iterable[Mapping[str, Any]],
    *,
    ticker: str,
    cik: int,
    since: str | pd.Timestamp,
    until: str | pd.Timestamp | None = None,
    overrides: Mapping[str, str] | None = None,
    predecessor_subs: Mapping[int, Iterable[Mapping[str, Any]]] | None = None,
) -> pd.DataFrame:
    """2.02 8-Ks for one issuer from its EDGAR submissions JSONs (main + pages).

    ``predecessor_subs`` maps a predecessor registrant's CIK (e.g. Google Inc. for
    GOOGL, EXXON MOBIL CORP for XOM after the 2026 holding-company reorganisation)
    to its submissions; their 2.02 8-Ks join the same ticker history. Co-filed 8-Ks
    (same accession under both CIKs) appear once; ``filer_cik`` records whose
    listing it came from (the successor's when co-filed).

    Returns every 2.02 8-K in ``[since, until]`` with ``acceptance_et``, ``timing``,
    ``announce_date`` (ET date of acceptance), ``quarter``, ``keep`` and
    ``drop_reason`` (stage-1 :func:`dedupe_earnings`).
    """
    cols = ["accession", "filing_date", "acceptance_raw", "items"]
    groups: list[tuple[int, Iterable[Mapping[str, Any]]]] = [(int(cik), subs)]
    groups += [(int(k), v) for k, v in (predecessor_subs or {}).items()]
    parts = []
    for filer, ss in groups:
        for sub in ss:
            p = parse_edgar_8k_202(sub)
            if len(p):
                parts.append(p.assign(filer_cik=filer))
    df = (pd.concat(parts, ignore_index=True) if parts
          else pd.DataFrame(columns=[*cols, "filer_cik"]))
    df = df.drop_duplicates("accession").copy()
    df["ticker"] = ticker
    df["cik"] = int(cik)
    if df.empty:
        for c in ("acceptance_et", "timing", "announce_date", "quarter", "keep", "drop_reason"):
            df[c] = pd.Series(dtype="object")
        return df
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    lo = pd.Timestamp(since)
    df = df[df["filing_date"] >= lo]
    if until is not None:
        df = df[df["filing_date"] <= pd.Timestamp(until)]
    df = df.copy()
    df["acceptance_et"] = df["acceptance_raw"].map(acceptance_to_et)
    df["timing"] = df["acceptance_et"].map(classify_timing)
    df["announce_date"] = df["acceptance_et"].map(lambda t: pd.Timestamp(t.date()))
    ov = dict(overrides or {})
    if ticker.upper() in DELIVERIES_RULE_TICKERS:
        for acc, it, d in zip(df["accession"], df["items"], df["announce_date"], strict=True):
            if is_deliveries_release(it, d):
                ov.setdefault(acc, f"{ticker} {DELIVERIES_REASON}")
    if df.empty:
        df["quarter"] = pd.Series(dtype="object")
        df["keep"] = pd.Series(dtype=bool)
        df["drop_reason"] = pd.Series(dtype="object")
        return df
    return dedupe_earnings(df, ov)


def flag_ambiguous(
    df: pd.DataFrame, *, gap_short: int = GAP_SHORT_DAYS, gap_long: int = GAP_LONG_DAYS
) -> pd.DataFrame:
    """Add ``ambiguous`` (``;``-joined reasons, ``""`` when clean). Never changes ``keep``.

    * ``multi_202_in_quarter(n)`` — n > 1 candidate 2.02 8-Ks in the quarter (all
      candidate rows; filings already dropped by an explicit override are resolved
      and not counted): a pre-announcement / guidance 8-K may have been kept
      instead of the report. When the candidate closest to the issuer's quarterly
      cadence (previous kept event + 91 days) is not the kept one, the kept row
      also gets ``cadence_prefers(YYYY-MM-DD)`` — a suggestion, not applied.
      **Both are LOOK-AHEAD flags** (:data:`LOOKAHEAD_FLAGS`): the kept row is the first
      candidate of its quarter, so ``n`` and the cadence comparison use 8-Ks filed AFTER
      it. They are resolved only once the calendar quarter has ended. Do not use them to
      select events for a signal formed before then; use :func:`clean_event_mask`
      (default ``flags="pit"``).
    * ``intraday_acceptance`` — kept event accepted 09:30-16:00 ET (release time
      unknown; e.g. JPM releases ~07:00 but its 8-K lands ~10:45).
    * ``weekend_announce`` — kept event accepted on a Saturday/Sunday.
    * ``no_9.01_exhibit`` — kept event without item 9.01 (results releases normally
      attach the press release as exhibit 99.1).
    * ``gap_short(Nd)`` / ``gap_long(Nd)`` — kept event < ``gap_short`` or >
      ``gap_long`` days after the previous kept event of the same ticker (quarter
      boundary drift, a missed report, or a non-report kept).

    The last four bullets are point-in-time (:data:`PIT_FLAGS`): known at the event's own
    acceptance.
    """
    out = df.copy()
    reasons: dict[Any, list[str]] = {i: [] for i in out.index}
    if out.empty:
        out["ambiguous"] = pd.Series(dtype="object")
        return out
    overridden = out["drop_reason"].astype(str).str.startswith("override:")
    cand = out[~overridden]
    n_q = cand.groupby(["ticker", "quarter"])["accession"].transform("size")
    for i in cand.index[n_q > 1]:
        reasons[i].append(f"multi_202_in_quarter({int(n_q[i])})")
    k = out[out["keep"].astype(bool)].sort_values(["ticker", "announce_date"])
    prev_kept: dict[Any, pd.Timestamp] = {}
    for _, g in k.groupby("ticker"):
        last = None
        for i in g.index:
            if last is not None:
                prev_kept[i] = last
            last = pd.Timestamp(g.at[i, "announce_date"])
    for (_tk, _q), g in cand.groupby(["ticker", "quarter"]):
        kept_i = [i for i in g.index if bool(out.at[i, "keep"])]
        if len(g) < 2 or not kept_i or kept_i[0] not in prev_kept:
            continue
        target = prev_kept[kept_i[0]] + pd.Timedelta(days=91)
        dist = (pd.to_datetime(g["announce_date"]) - target).abs()
        best = dist.idxmin()
        if best != kept_i[0]:
            d = pd.Timestamp(out.at[best, "announce_date"]).date().isoformat()
            reasons[kept_i[0]].append(f"cadence_prefers({d})")
    for i, r in k.iterrows():
        if r["timing"] == "INTRADAY":
            reasons[i].append("intraday_acceptance")
        if pd.Timestamp(r["announce_date"]).weekday() >= 5:
            reasons[i].append("weekend_announce")
        if "9.01" not in _items(r["items"]):
            reasons[i].append("no_9.01_exhibit")
    for _, g in k.groupby("ticker"):
        prev = None
        for i, r in g.iterrows():
            d = pd.Timestamp(r["announce_date"])
            if prev is not None:
                gap = (d - prev).days
                if gap < gap_short:
                    reasons[i].append(f"gap_short({gap}d)")
                elif gap > gap_long:
                    reasons[i].append(f"gap_long({gap}d)")
            prev = d
    out["ambiguous"] = [";".join(reasons[i]) for i in out.index]
    return out


def ambiguity_flag_names(ambiguous: pd.Series) -> pd.Series:
    """Set of flag NAMES per row of an ``ambiguous`` column (``"gap_long(155d)"`` ->
    ``"gap_long"``; NaN / ``""`` -> empty set)."""
    a = ambiguous.fillna("").astype(str)
    return a.map(lambda r: frozenset(x.split("(", 1)[0].strip() for x in r.split(";") if x.strip()))


def clean_event_mask(ambiguous: pd.Series, *, flags: str | Iterable[str] = "pit") -> pd.Series:
    """Boolean mask, True where an event counts as "clean" (unambiguous).

    ``flags``:

    * ``"pit"`` (default) — clean unless the row carries a :data:`PIT_FLAGS` flag. The
      :data:`LOOKAHEAD_FLAGS` (``multi_202_in_quarter``, ``cadence_prefers``) are ignored,
      because they are set from 2.02 8-Ks filed after the kept event (red team
      2026-10-10, F1).
    * ``"legacy_any"`` — clean only when ``ambiguous`` is empty (any flag, look-ahead ones
      included). This is the rule studies 11 and 12 ran with; keep it only to reproduce
      them.
    * an iterable of flag names — clean unless the row carries one of those names.

    Raises ``ValueError`` on a flag name not in :data:`ALL_FLAGS` (in the data or in
    ``flags``), so a new flag cannot silently pass or fail events.
    """
    if isinstance(flags, str):
        if flags == "legacy_any":
            return ~(ambiguous.fillna("").astype(str).str.len() > 0)
        if flags != "pit":
            raise ValueError(f"flags={flags!r}: use 'pit', 'legacy_any' or an iterable of names")
        bad = frozenset(PIT_FLAGS)
    else:
        bad = frozenset(flags)
        unknown = bad - set(ALL_FLAGS)
        if unknown:
            raise ValueError(f"unknown ambiguity flags {sorted(unknown)}")
    names = ambiguity_flag_names(ambiguous)
    seen = frozenset().union(*names.tolist()) if len(names) else frozenset()
    unknown = seen - set(ALL_FLAGS)
    if unknown:
        raise ValueError(f"ambiguous column carries unknown flags {sorted(unknown)}")
    return ~names.map(lambda f: bool(f & bad)).astype(bool)


def coverage_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """Kept events per ticker per calendar year (rows: ticker, columns: year)."""
    k = df[df["keep"].astype(bool)]
    if k.empty:
        return pd.DataFrame()
    y = pd.to_datetime(k["announce_date"]).dt.year
    return k.groupby([k["ticker"], y]).size().unstack(fill_value=0)


def next_expected_dates(
    earnings_dates: pd.DataFrame | pd.Series | None, now: pd.Timestamp
) -> list[pd.Timestamp]:
    """Future (``>= now``) timestamps from a yfinance ``earnings_dates`` frame.

    yfinance returns a frame indexed by tz-aware datetimes ("Earnings Date"); past
    rows carry reported EPS. Only the index is used. UNVERIFIED by construction —
    callers must label the result as such.
    """
    if earnings_dates is None or len(earnings_dates) == 0:
        return []
    idx = pd.DatetimeIndex(pd.to_datetime(earnings_dates.index, errors="coerce", utc=True))
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    fut = sorted(t for t in idx.dropna() if t >= now)
    return [pd.Timestamp(t) for t in fut]
