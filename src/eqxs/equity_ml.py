"""Machine-learning cross-sectional return model on the free US equity panel (study 13).

Pure functions (feature construction, point-in-time joins, target, expanding-window splitter
with embargo, model wrappers M0-M6 behind one interface, decile portfolios, nulls, metrics)
used by ``scripts/validate/equity_ml_xs_train.py``
(``reports/diagnostic/equity_ml_xs_train.md``). Nothing here reads data: callers pass
matrices / tables built from the ``eqxs.equities_xs`` loaders.

Conventions
-----------
* ``sessions``: sorted tz-naive trading dates; row ``t`` of a daily return matrix ``R`` holds the
  close(t-1) -> close(t) return (NaN = no bar / guarded). Matrices are (sessions x firms).
* Formation dates are month-end sessions (rows ``me_rows``). A feature at formation row ``t`` uses
  prices through row ``t`` (the close of T) and SEC facts with ``ts`` <= T 16:00 New York.
* A target formed at month-end row ``t`` with the next month-end row ``u`` is the compounded return
  over rows ``t+2 .. u+1`` (entry at the close of row ``t+1``, exit at the close of row ``u+1``).
* Month index ``mi`` = year * 12 + month - 1 (an integer per calendar month).
"""

# ruff: noqa: N803, N806, N815  (X / R / Xf: matrix names follow the maths)
from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------
PRICE_FEATURES: tuple[str, ...] = (
    "mom_12_1",
    "mom_6_1",
    "rev_1m",
    "rev_1w",
    "vol_63",
    "ivol_63",
    "maxret_21",
    "amihud_63",
    "log_dv_63",
    "beta_252",
    "skew_63",
    "hi_252",
    "log_price",
)
EVENT_FEATURES: tuple[str, ...] = ("days_since_ea", "days_to_exp_ea", "ann_ret_last", "ea_timing")
FUND_FEATURES: tuple[str, ...] = (
    "log_mcap",
    "bm",
    "ep",
    "sp",
    "gp",
    "asset_growth",
    "accruals",
    "nsi",
    "leverage",
    "cash_assets",
    "roe",
    "sue",
)
NUMERIC_FEATURES: tuple[str, ...] = PRICE_FEATURES + EVENT_FEATURES + FUND_FEATURES
SIC_DIVISIONS: tuple[tuple[str, int, int], ...] = (
    ("A", 1, 9),
    ("B", 10, 14),
    ("C", 15, 17),
    ("D", 20, 39),
    ("E", 40, 49),
    ("F", 50, 51),
    ("G", 52, 59),
    ("H", 60, 67),
    ("I", 70, 89),
    ("J", 91, 99),
)
SIC_COLS: tuple[str, ...] = tuple(f"sic_{d[0]}" for d in SIC_DIVISIONS) + ("sic_unknown",)
M0_FEATURES: tuple[str, ...] = ("mom_12_1", "sue", "log_mcap")
ABLATION_DROP: tuple[str, ...] = ("rev_1m", "rev_1w", "ivol_63", "maxret_21")
SURVIVORSHIP_FAMILY: tuple[str, ...] = ("rev_1m", "rev_1w", "ivol_63", "maxret_21")

# Which SEC tag each fundamental feature depends on (for the 30 % coverage rule).
FEATURE_TAGS: dict[str, tuple[str, ...]] = {
    "log_mcap": ("shares",),
    "bm": ("shares", "equity"),
    "ep": ("shares", "net_income"),
    "sp": ("shares", "revenue"),
    "gp": ("revenue", "cost_of_revenue", "assets"),
    "asset_growth": ("assets",),
    "accruals": ("net_income", "ocf", "assets"),
    "nsi": ("shares",),
    "leverage": ("liabilities", "assets"),
    "cash_assets": ("cash", "assets"),
    "roe": ("net_income", "equity"),
    "sue": ("eps",),
}
TAG_GROUPS: dict[str, tuple[str, ...]] = {
    "shares": ("EntityCommonStockSharesOutstanding",),
    "equity": ("StockholdersEquity",),
    "net_income": ("NetIncomeLoss",),
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ),
    "cost_of_revenue": ("CostOfRevenue",),
    "assets": ("Assets",),
    "ocf": ("NetCashProvidedByUsedInOperatingActivities",),
    "liabilities": ("Liabilities",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "eps": ("EarningsPerShareDiluted", "EarningsPerShareBasic"),
}
TAG_UOM: dict[str, str] = {"shares": "shares", "eps": "USD/shares"}  # every other group: USD
INSTANT_GROUPS: frozenset[str] = frozenset({"shares", "equity", "assets", "liabilities", "cash"})
FLOW_GROUPS: frozenset[str] = frozenset({"net_income", "revenue", "cost_of_revenue", "ocf"})
ALL_SEC_TAGS: tuple[str, ...] = tuple(t for g in TAG_GROUPS.values() for t in g)
SEC_FORMS: tuple[str, ...] = ("10-K", "10-Q", "10-K/A", "10-Q/A")
STALE_MONTHS: int = 18


def input_columns(numeric: Sequence[str], sic: bool = True) -> list[str]:
    """Model input columns: normalised numeric features, SIC one-hots, missing indicators."""
    cols = list(numeric)
    if sic:
        cols += list(SIC_COLS)
    cols += [f"miss_{c}" for c in numeric]
    return cols


def feature_groups(numeric: Sequence[str], sic: bool = True) -> dict[str, list[str]]:
    """Permutation-importance groups: each numeric feature with its missing indicator;
    the SIC one-hots form one group."""
    g = {c: [c, f"miss_{c}"] for c in numeric}
    if sic:
        g["sic"] = list(SIC_COLS)
    return g


# ---------------------------------------------------------------------------
# Calendars
# ---------------------------------------------------------------------------
def month_index(dates: pd.Series | pd.DatetimeIndex | np.ndarray) -> np.ndarray:
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    return (d.year * 12 + d.month - 1).to_numpy().astype(np.int64)


def month_end_rows(sessions: pd.DatetimeIndex) -> np.ndarray:
    """Row of the last session of each calendar month present in ``sessions``."""
    mi = month_index(sessions)
    last = np.flatnonzero(np.r_[mi[1:] != mi[:-1], True])
    return last.astype(np.int64)


def nearest_month_key(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Month index of the nearest month END (study 11's quarter key: (d + 15 days) -> month - 1)."""
    d = pd.DatetimeIndex(pd.to_datetime(dates)) + pd.Timedelta(days=15)
    return (d.year * 12 + d.month - 1).to_numpy().astype(np.int64) - 1


def close_ts_utc_ns(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """int64 ns UTC of 16:00 America/New_York on each date (the close of T)."""
    d = pd.DatetimeIndex(pd.to_datetime(dates)).normalize() + pd.Timedelta(hours=16)
    loc = d.tz_localize("America/New_York")
    return loc.tz_convert("UTC").asi8.astype(np.int64)


def utc_ns(ts: pd.Series) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).asi8.astype(np.int64)


# ---------------------------------------------------------------------------
# Price matrices, universe, target
# ---------------------------------------------------------------------------
@dataclass
class PriceMatrices:
    """Wide daily matrices on the union calendar. ``R`` float32 with NaN for no bar / guarded."""

    sessions: pd.DatetimeIndex
    symbols: np.ndarray
    R: np.ndarray  # float32 daily returns (NaN = missing)
    adj: np.ndarray  # float32 adj_close (NaN = no bar)
    raw_close: np.ndarray  # float32 traded price (splits after t within the panel undone)
    dv: np.ndarray  # float32 dollar volume close*volume (split-invariant)
    log_split_cum: np.ndarray  # float64 cumulative log split ratio (row t includes splits on t)
    first_bar: np.ndarray  # int64 first row with a bar per firm (n_rows if none)
    n_guarded: int = 0

    @property
    def has_bar(self) -> np.ndarray:
        return np.isfinite(self.adj)

    @property
    def n_rows(self) -> int:
        return len(self.sessions)

    @property
    def n_cols(self) -> int:
        return len(self.symbols)


def build_price_matrices(
    sessions: pd.DatetimeIndex,
    symbols: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    close: np.ndarray,
    adj_close: np.ndarray,
    volume: np.ndarray,
    splits: np.ndarray,
    *,
    r_max: float = 3.0,
    r_min: float = -0.95,
) -> PriceMatrices:
    """Long bar arrays (row/col indices) -> :class:`PriceMatrices`.

    Return = adj_close ratio to the firm's previous bar (a gap return lands on the next bar).
    Guard (study 11): missing if adj <= 0, prev <= 0, r > ``r_max`` or r < ``r_min``.
    """
    n, k = len(sessions), len(symbols)

    def wide(v: np.ndarray, fill: float = np.nan, dtype: Any = np.float32) -> np.ndarray:
        m = np.full((n, k), fill, dtype=dtype)
        m[row, col] = v
        return m

    adj = wide(adj_close)
    prev = pd.DataFrame(adj).ffill().shift(1).to_numpy(dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = adj / prev - 1.0
    bad = ~np.isfinite(r) | (adj <= 0) | (prev <= 0) | (r > r_max) | (r < r_min)
    n_guarded = int((bad & np.isfinite(adj) & np.isfinite(prev)).sum())
    r[bad] = np.nan
    spl = wide(splits, 0.0, np.float64)
    spl = np.where((spl > 0) & np.isfinite(spl), spl, 1.0)
    lsc = np.cumsum(np.log(spl), axis=0)
    after = np.exp(lsc[-1][None, :] - lsc).astype(np.float32)  # splits strictly after t
    raw = wide(close) * after
    dv = wide(close * volume)
    has = np.isfinite(adj)
    first = np.where(has.any(axis=0), has.argmax(axis=0), n).astype(np.int64)
    return PriceMatrices(
        sessions, np.asarray(symbols), r.astype(np.float32), adj, raw, dv, lsc, first, n_guarded
    )


def rolling_median_at(x: np.ndarray, rows: np.ndarray, win: int, minp: int) -> np.ndarray:
    """nanmedian of ``x[t-win+1 .. t]`` at each row in ``rows`` (NaN if < ``minp`` finite)."""
    out = np.full((len(rows), x.shape[1]), np.nan, dtype=np.float32)
    for i, t in enumerate(rows):
        w = x[max(0, t - win + 1) : t + 1]
        cnt = np.isfinite(w).sum(axis=0)
        with np.errstate(all="ignore"), _Quiet():
            med = np.nanmedian(w, axis=0)
        out[i] = np.where(cnt >= minp, med, np.nan)
    return out


class _Quiet:
    def __enter__(self) -> None:
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, *a: Any) -> None:
        self._w.__exit__(*a)


@dataclass
class Universe:
    me_rows: np.ndarray  # month-end rows
    med_dv: np.ndarray  # (n_me x cols) 63-day median dollar volume
    all_: np.ndarray  # (n_me x cols) bool
    liq: np.ndarray  # (n_me x cols) bool


def build_universe(
    pm: PriceMatrices,
    me_rows: np.ndarray,
    *,
    min_price: float = 5.0,
    min_dv: float = 1e6,
    win: int = 63,
    minp: int = 50,
    eligible_cols: np.ndarray | None = None,
) -> Universe:
    """ALL = bar on T, raw close >= ``min_price``, 63-day median dollar volume >= ``min_dv``;
    LIQ = ALL with percentile rank of the median dollar volume > 0.5 within the date."""
    med = rolling_median_at(pm.dv, me_rows, win, minp)
    px = pm.raw_close[me_rows]
    has = np.isfinite(pm.adj[me_rows])
    with np.errstate(invalid="ignore"):
        a = has & (px >= min_price) & (med >= min_dv)
    if eligible_cols is not None:
        a &= eligible_cols[None, :]
    liq = np.zeros_like(a)
    for i in range(len(me_rows)):
        idx = np.flatnonzero(a[i])
        if len(idx) == 0:
            continue
        pct = pd.Series(med[i, idx]).rank(pct=True).to_numpy()
        liq[i, idx[pct > 0.5]] = True
    return Universe(np.asarray(me_rows), med, a, liq)


def ew_market_return(R: np.ndarray, me_rows: np.ndarray, uni_all: np.ndarray) -> np.ndarray:
    """Daily EW mean of valid returns over the ALL universe of the latest month-end strictly
    before each day (NaN before the first month-end universe)."""
    n = R.shape[0]
    out = np.full(n, np.nan)
    for i, t in enumerate(me_rows):
        a = t + 1
        b = me_rows[i + 1] + 1 if i + 1 < len(me_rows) else n
        cols = np.flatnonzero(uni_all[i])
        if len(cols) == 0 or a >= n:
            continue
        block = R[a:b][:, cols]
        with _Quiet():
            out[a:b] = np.nanmean(block, axis=1)
    return out


def holding_returns(
    R: np.ndarray, me_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compounded return from the close of row t+1 to the close of row u+1 for each month-end t
    (u = next month-end). Returns (ret (n_me x cols) float32 with NaN where infeasible,
    entry_rows, exit_rows) (-1 where infeasible). Missing daily returns count as 0."""
    n = R.shape[0]
    lc = np.vstack(
        [
            np.zeros((1, R.shape[1])),
            np.cumsum(np.log1p(np.nan_to_num(R, nan=0.0).astype(np.float64)), axis=0),
        ]
    )
    out = np.full((len(me_rows), R.shape[1]), np.nan, dtype=np.float32)
    entry = np.full(len(me_rows), -1, dtype=np.int64)
    exit_ = np.full(len(me_rows), -1, dtype=np.int64)
    for i, t in enumerate(me_rows):
        if i + 1 >= len(me_rows):
            continue
        e, x = t + 1, me_rows[i + 1] + 1
        if x >= n:
            continue
        entry[i], exit_[i] = e, x
        out[i] = np.expm1(lc[x + 1] - lc[e + 1]).astype(np.float32)
    return out, entry, exit_


def compound_series(x: np.ndarray, entry: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    """Compounded value of a daily series over rows entry+1 .. exit (NaN where infeasible)."""
    lc = np.r_[0.0, np.cumsum(np.log1p(np.nan_to_num(np.asarray(x, dtype=np.float64))))]
    out = np.full(len(entry), np.nan)
    ok = (entry >= 0) & (exit_ > entry)
    out[ok] = np.expm1(lc[exit_[ok] + 1] - lc[entry[ok] + 1])
    return out


# ---------------------------------------------------------------------------
# Price-based features (computed only at month-end rows; window slices, vectorised across firms)
# ---------------------------------------------------------------------------
def _nan_skew(w: np.ndarray, minp: int) -> np.ndarray:
    cnt = np.isfinite(w).sum(axis=0)
    with _Quiet():
        mu = np.nanmean(w, axis=0)
        d = w - mu
        m2 = np.nanmean(d**2, axis=0)
        m3 = np.nanmean(d**3, axis=0)
        s = m3 / m2**1.5
    return np.where((cnt >= minp) & (m2 > 0), s, np.nan)


def _pair_moments(
    r: np.ndarray, m: np.ndarray, minp: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each firm column: (n, var_r, cov_rm, var_m) over rows where both r and m are finite."""
    mm = np.broadcast_to(m[:, None], r.shape)
    ok = np.isfinite(r) & np.isfinite(mm)
    n = ok.sum(axis=0).astype(np.float64)
    rr = np.where(ok, r, 0.0).astype(np.float64)
    m2 = np.where(ok, mm, 0.0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mr, mmu = rr.sum(0) / n, m2.sum(0) / n
        vr = ((rr - mr) ** 2 * ok).sum(0) / (n - 1)
        vm = ((m2 - mmu) ** 2 * ok).sum(0) / (n - 1)
        cv = ((rr - mr) * (m2 - mmu) * ok).sum(0) / (n - 1)
    bad = n < minp
    for a in (vr, vm, cv):
        a[bad] = np.nan
    return n, vr, cv, vm


def price_features(
    pm: PriceMatrices,
    me_rows: np.ndarray,
    mkt_ew: np.ndarray,
    rows_wanted: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Price-based features at month-end rows (positions into ``me_rows``; default all).

    Uses only rows <= the formation row. Output arrays are (n_wanted x cols) float32.
    """
    pos = np.arange(len(me_rows)) if rows_wanted is None else np.asarray(rows_wanted)
    R = pm.R
    lc = np.vstack(
        [
            np.zeros((1, pm.n_cols)),
            np.cumsum(np.log1p(np.nan_to_num(R, nan=0.0).astype(np.float64)), axis=0),
        ]
    )
    first = pm.first_bar
    out = {f: np.full((len(pos), pm.n_cols), np.nan, dtype=np.float32) for f in PRICE_FEATURES}

    def cum(a: int, b: int) -> np.ndarray:
        """Return from the close of row a to the close of row b; NaN if a < first bar."""
        v = np.expm1(lc[b + 1] - lc[a + 1])
        return np.where(first <= a, v, np.nan)

    for k, p in enumerate(pos):
        t = int(me_rows[p])
        if p >= 12:
            out["mom_12_1"][k] = cum(int(me_rows[p - 12]), int(me_rows[p - 1]))
        if p >= 6:
            out["mom_6_1"][k] = cum(int(me_rows[p - 6]), int(me_rows[p - 1]))
        if p >= 1:
            out["rev_1m"][k] = cum(int(me_rows[p - 1]), t)
        if t >= 5:
            out["rev_1w"][k] = cum(t - 5, t)
        w63 = R[max(0, t - 62) : t + 1]
        cnt63 = np.isfinite(w63).sum(axis=0)
        with _Quiet():
            sd = np.nanstd(w63, axis=0, ddof=1)
        out["vol_63"][k] = np.where(cnt63 >= 50, sd, np.nan)
        n, vr, cv, vm = _pair_moments(w63, mkt_ew[max(0, t - 62) : t + 1], 50)
        with np.errstate(invalid="ignore", divide="ignore"):
            resid_var = vr - cv**2 / vm
        out["ivol_63"][k] = np.sqrt(np.clip(resid_var, 0.0, None))
        w21 = R[max(0, t - 20) : t + 1]
        with _Quiet():
            mx = np.nanmax(w21, axis=0)
        out["maxret_21"][k] = np.where(np.isfinite(w21).sum(axis=0) >= 15, mx, np.nan)
        dv63 = pm.dv[max(0, t - 62) : t + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            ill = np.abs(w63) / (dv63 / 1e6)
        ill = np.where(np.isfinite(ill) & (dv63 > 0), ill, np.nan)
        cnt = np.isfinite(ill).sum(axis=0)
        with _Quiet():
            out["amihud_63"][k] = np.where(cnt >= 50, np.nanmean(ill, axis=0), np.nan)
        cntd = np.isfinite(dv63).sum(axis=0)
        with _Quiet():
            med = np.nanmedian(dv63, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["log_dv_63"][k] = np.where((cntd >= 50) & (med > 0), np.log(med), np.nan)
        w252 = R[max(0, t - 251) : t + 1]
        n, vr, cv, vm = _pair_moments(w252, mkt_ew[max(0, t - 251) : t + 1], 200)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["beta_252"][k] = cv / vm
        out["skew_63"][k] = _nan_skew(w63, 50)
        a252 = pm.adj[max(0, t - 251) : t + 1]
        with _Quiet():
            hi = np.nanmax(a252, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["hi_252"][k] = np.where(
                np.isfinite(a252).sum(axis=0) >= 200, pm.adj[t] / hi, np.nan
            )
            out["log_price"][k] = np.where(pm.raw_close[t] > 0, np.log(pm.raw_close[t]), np.nan)
    return out


# ---------------------------------------------------------------------------
# Earnings-calendar features
# ---------------------------------------------------------------------------
def announcement_signal(
    day0_row: np.ndarray, col: np.ndarray, R: np.ndarray, mkt: np.ndarray
) -> np.ndarray:
    """Study-11 B signal: sum over sessions d-1, d, d+1 of (r - mkt); NaN unless all 3 valid."""
    n = R.shape[0]
    sig = np.zeros(len(day0_row))
    good = (day0_row >= 1) & (day0_row + 1 < n) & (col >= 0)
    for k in (-1, 0, 1):
        rr = np.full(len(day0_row), np.nan)
        rr[good] = R[day0_row[good] + k, col[good]]
        good &= np.isfinite(rr)
        sig += np.nan_to_num(rr) - np.where(good, mkt[np.clip(day0_row + k, 0, n - 1)], 0.0)
    return np.where(good, sig, np.nan)


def event_features(
    events: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    me_rows: np.ndarray,
    n_cols: int,
    rows_wanted: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Earnings-calendar features at month-end rows.

    ``events``: one row per kept 8-K 2.02 event with columns ``col`` (firm column), ``day0``
    (date), ``day0_row`` (session row), ``clean`` (bool, no ambiguity flag) and ``signal``
    (announcement return, NaN allowed). Only events with day 0 <= T enter a feature at T.
    """
    pos = np.arange(len(me_rows)) if rows_wanted is None else np.asarray(rows_wanted)
    t_rows = me_rows[pos]
    t_days = pd.DatetimeIndex(sessions[t_rows]).values.astype("datetime64[D]").astype(np.int64)
    out = {f: np.full((len(pos), n_cols), np.nan, dtype=np.float32) for f in EVENT_FEATURES}
    sess_days = pd.DatetimeIndex(sessions).values.astype("datetime64[D]").astype(np.int64)
    ev = events[(events["col"] >= 0) & (events["day0_row"] >= 0)].copy()
    ev["d"] = pd.DatetimeIndex(ev["day0"]).values.astype("datetime64[D]").astype(np.int64)
    ev = ev.sort_values(["col", "d"], kind="stable")
    exp_days = ev["d"].to_numpy() + 364
    # nearest session to the expected date (ties -> earlier), clamped to the calendar
    hi = np.clip(np.searchsorted(sess_days, exp_days, side="left"), 0, len(sess_days) - 1)
    lo = np.clip(hi - 1, 0, len(sess_days) - 1)
    pick_lo = np.abs(exp_days - sess_days[lo]) <= np.abs(sess_days[hi] - exp_days)
    ev["e_row"] = np.where(pick_lo, lo, hi)
    ev["e_day"] = exp_days
    beyond = exp_days > sess_days[-1]
    ev.loc[beyond, "e_row"] = len(sess_days) + 10  # never resolvable inside the calendar
    for c, g in ev.groupby("col", sort=False):
        d = g["d"].to_numpy()
        drow = g["day0_row"].to_numpy()
        e_row = g["e_row"].to_numpy()
        e_day = g["e_day"].to_numpy()
        clean = g["clean"].to_numpy(bool)
        sig = g["signal"].to_numpy(float)
        # timing status of each event's expected date
        status = np.full(len(g), np.nan)
        for k in range(len(g)):
            later = np.arange(k + 1, len(g))
            m = later[(d[later] >= e_day[k] - 45) & (drow[later] <= e_row[k] + 5)]
            if len(m) == 0:
                status[k] = 1.0
            elif drow[m[0]] <= e_row[k] - 5:
                status[k] = -1.0
            else:
                status[k] = 0.0
        resolve_row = e_row + 5
        cl_idx = np.flatnonzero(clean & np.isfinite(sig))
        for i, (tr, td) in enumerate(zip(t_rows, t_days, strict=True)):
            n_known = np.searchsorted(drow, tr, side="right")  # day0 row <= T row
            if n_known == 0:
                continue
            last = d[n_known - 1]
            if td - last <= 400:
                out["days_since_ea"][i, c] = td - last
            j = np.searchsorted(d[:n_known], td - 364, side="right")
            if j < n_known:
                out["days_to_exp_ea"][i, c] = d[j] + 364 - td
            # latest clean event with d+1 <= T, within 365 days
            ok = cl_idx[(drow[cl_idx] + 1 <= tr) & (d[cl_idx] >= td - 365)]
            if len(ok):
                out["ann_ret_last"][i, c] = sig[ok[-1]]
            # timing flag: latest expected date resolvable at T (E + 5 sessions <= T),
            # with E >= T - 120 d
            res = np.flatnonzero((resolve_row <= tr) & (np.arange(len(g)) < n_known))
            if len(res):
                k = res[np.argmax(e_day[res])]
                if e_day[k] >= td - 120:
                    out["ea_timing"][i, c] = status[k]
    return out


def pit_clean(ambiguous: pd.Series) -> np.ndarray:
    """True where a kept 8-K event carries no point-in-time ambiguity flag.

    Delegates to the shared definition :func:`eqxs.earnings_calendar.clean_event_mask`
    (``flags="pit"``): ``multi_202_in_quarter`` / ``cadence_prefers`` are set from LATER 8-Ks of the
    quarter, so selecting on them is look-ahead (red team 2026-10-10, F1) and they are ignored."""
    from eqxs.earnings_calendar import clean_event_mask

    return clean_event_mask(pd.Series(ambiguous)).to_numpy(bool)


def sic_division(sic: pd.Series) -> pd.Series:
    """4-digit SIC string -> division letter (A-J) or 'unknown'."""
    num = pd.to_numeric(pd.Series(sic).astype(str).str[:2], errors="coerce")
    out = pd.Series("unknown", index=num.index, dtype=object)
    for letter, lo, hi in SIC_DIVISIONS:
        out[(num >= lo) & (num <= hi)] = letter
    return out


def cik_symbol_map(
    cik_ticker_map: pd.DataFrame,
    yf_of: Mapping[str, str],
    col_of: Mapping[str, int],
    bars: np.ndarray,
) -> tuple[dict[int, str], dict[str, int]]:
    """Study-11 CIK -> price symbol map: common listing rows only, symbols claimed by two CIKs
    dropped, a CIK with several share classes keeps the symbol with most bars
    (tie: alphabetical)."""
    cm = cik_ticker_map[
        (cik_ticker_map["security_type"] == "common") & cik_ticker_map["cik"].notna()
    ].copy()
    cm["yf"] = cm["symbol"].map(yf_of)
    cm = cm[cm["yf"].isin(col_of.keys())]
    multi = cm.groupby("yf")["cik"].nunique()
    dup = set(multi[multi > 1].index)
    cm = cm[~cm["yf"].isin(dup)].drop_duplicates(["cik", "yf"])
    cm["bars"] = [int(bars[col_of[s]]) for s in cm["yf"]]
    cm = cm.sort_values(["cik", "bars", "yf"], ascending=[True, False, True])
    n_multi = int((cm.groupby("cik").size() > 1).sum())
    first = cm.drop_duplicates("cik")
    mp = {int(c): s for c, s in zip(first["cik"], first["yf"], strict=True)}
    return mp, {
        "symbols_on_two_ciks_dropped": len(dup),
        "ciks_with_several_symbols": n_multi,
        "ciks_mapped": len(mp),
    }


# ---------------------------------------------------------------------------
# Fundamentals: point-in-time joins
# ---------------------------------------------------------------------------
@dataclass
class Facts:
    """Minimal fact table for one tag (or tag group): cik, period month key, ts, value."""

    cik: np.ndarray  # int64
    mk: np.ndarray  # int64 month key of ddate (nearest month end)
    ts: np.ndarray  # int64 ns UTC availability
    value: np.ndarray  # float64

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> Facts:
        return cls(
            df["cik"].to_numpy(np.int64),
            df["mk"].to_numpy(np.int64),
            df["ts"].to_numpy(np.int64),
            df["value"].to_numpy(np.float64),
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"cik": self.cik, "mk": self.mk, "ts": self.ts, "value": self.value})

    def __len__(self) -> int:
        return len(self.cik)


def prepare_facts(num: pd.DataFrame) -> pd.DataFrame:
    """companyfacts rows -> (cik, tag, qtrs, mk, ts, value), one row per (cik, tag, qtrs, period,
    filing ts) (duplicates within a filing: last). ``num`` needs cik, tag, uom, value, ddate, qtrs,
    form, ts."""
    d = num[num["form"].isin(SEC_FORMS) & num["value"].notna() & num["ddate"].notna()]
    out = pd.DataFrame(
        {
            "cik": d["cik"].to_numpy(np.int64),
            "tag": d["tag"].astype(str).to_numpy(),
            "uom": d["uom"].astype(str).to_numpy(),
            "qtrs": d["qtrs"].to_numpy(np.int64),
            "mk": nearest_month_key(d["ddate"]),
            "dday": pd.DatetimeIndex(pd.to_datetime(d["ddate"]))
            .values.astype("datetime64[D]")
            .astype(np.int64),
            "ts": utc_ns(d["ts"]),
            "value": d["value"].to_numpy(np.float64),
            "adsh": d["adsh"].astype(str).to_numpy(),
            "form": d["form"].astype(str).to_numpy(),
        }
    )
    return out


def pit_latest(f: Facts, q_cik: np.ndarray, q_ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each query (cik, ts): the value of the latest period among facts with ts <= query ts,
    the latest filing for that period. Returns (value, mk); NaN / -1 when none."""
    if len(f) == 0:
        return np.full(len(q_cik), np.nan), np.full(len(q_cik), -1, dtype=np.int64)
    df = f.frame().sort_values(["cik", "ts", "mk"], kind="stable")
    cm = df.groupby("cik")["mk"].cummax()
    df = df[df["mk"].to_numpy() >= cm.to_numpy()]
    df = df.sort_values(["ts", "mk"], kind="stable")
    q = pd.DataFrame(
        {"cik": q_cik.astype(np.int64), "ts": q_ts.astype(np.int64), "qi": np.arange(len(q_cik))}
    ).sort_values("ts", kind="stable")
    m = pd.merge_asof(q, df, on="ts", by="cik", direction="backward")
    m = m.sort_values("qi")
    return m["value"].to_numpy(np.float64), m["mk"].fillna(-1).to_numpy(np.int64)


def pit_for_period(f: Facts, q_cik: np.ndarray, q_mk: np.ndarray, q_ts: np.ndarray) -> np.ndarray:
    """Value for period ``q_mk`` of firm ``q_cik`` as known at ``q_ts`` (latest filing <= ts)."""
    if len(f) == 0:
        return np.full(len(q_cik), np.nan)
    df = f.frame().sort_values("ts", kind="stable")
    q = pd.DataFrame(
        {
            "cik": q_cik.astype(np.int64),
            "mk": q_mk.astype(np.int64),
            "ts": q_ts.astype(np.int64),
            "qi": np.arange(len(q_cik)),
        }
    ).sort_values("ts", kind="stable")
    m = pd.merge_asof(q, df, on="ts", by=["cik", "mk"], direction="backward")
    return m.sort_values("qi")["value"].to_numpy(np.float64)


def pit_ttm(
    fq: Facts, fa: Facts, q_cik: np.ndarray, q_ts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """TTM flow as known at ``q_ts``: sum of the 4 quarterly values ending at the latest quarterly
    period (all four known), else the latest annual; both -> the later period end
    (tie: quarterly)."""
    vq, mq = pit_latest(fq, q_cik, q_ts)
    tot = np.where(mq >= 0, vq, np.nan)
    for lag in (3, 6, 9):
        tot = tot + pit_for_period(fq, q_cik, mq - lag, q_ts)
    va, ma = pit_latest(fa, q_cik, q_ts)
    use_q = np.isfinite(tot) & (~np.isfinite(va) | (mq >= ma))
    val = np.where(use_q, tot, va)
    mk = np.where(use_q, mq, np.where(np.isfinite(va), ma, -1))
    return val, mk


def sue_events(eps: pd.DataFrame) -> pd.DataFrame:
    """Study 11's SUE events from EPS facts (prepared by :func:`prepare_facts`): USD/shares,
    qtrs == 1; diluted where present for (filing, period), else basic. Event = first original
    10-Q/10-K (by ts) whose own current quarter (latest qkey it reports) is q; EPS for q and every
    lag as known at the event ts. SUE = (EPS_q - EPS_q-4) / sd(8 previous seasonal differences),
    >= 6 of them and sd > 0. Returns (cik, ev_ts, qkey, sue)."""
    num = eps[(eps["uom"] == "USD/shares") & (eps["qtrs"] == 1)].copy()
    num["qkey"] = num["mk"]
    num["is_dil"] = num["tag"] == "EarningsPerShareDiluted"
    num = num.sort_values(["adsh", "qkey", "is_dil"]).drop_duplicates(["adsh", "qkey"], keep="last")
    facts = num.loc[:, ["cik", "adsh", "form", "qkey", "ts", "value"]]
    orig = facts[facts["form"].isin(["10-Q", "10-K"])]
    if orig.empty:
        return pd.DataFrame(columns=["cik", "ev_ts", "qkey", "sue"])
    own = orig.loc[orig.groupby("adsh")["qkey"].idxmax()]
    own = own.sort_values("ts", kind="stable").drop_duplicates(["cik", "qkey"], keep="first")
    evs = own.loc[:, ["cik", "qkey", "ts"]].rename(columns={"ts": "ev_ts"}).reset_index(drop=True)
    evs["eid"] = np.arange(len(evs))
    f = facts.loc[:, ["cik", "qkey", "ts", "value"]].sort_values("ts", kind="stable")
    vals = []
    for lag in range(13):
        q = evs.loc[:, ["eid", "cik", "qkey", "ev_ts"]].copy()
        q["qkey"] = q["qkey"] - 3 * lag
        q = q.sort_values("ev_ts", kind="stable")
        m = pd.merge_asof(
            q, f, left_on="ev_ts", right_on="ts", by=["cik", "qkey"], direction="backward"
        )
        vals.append(m.set_index("eid")["value"].reindex(evs["eid"]).to_numpy())
    e = np.column_stack(vals)
    diffs = e[:, 0:9] - e[:, 4:13]
    prior = diffs[:, 1:9]
    n_prior = np.isfinite(prior).sum(axis=1)
    with _Quiet():
        sd = np.nanstd(prior, axis=1, ddof=1)
    ok = np.isfinite(diffs[:, 0]) & (n_prior >= 6) & np.isfinite(sd) & (sd > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        evs["sue"] = np.where(ok, diffs[:, 0] / sd, np.nan)
    return evs.loc[ok, ["cik", "ev_ts", "qkey", "sue"]].reset_index(drop=True)


def latest_event_value(
    ev_cik: np.ndarray,
    ev_ts: np.ndarray,
    ev_val: np.ndarray,
    q_cik: np.ndarray,
    q_ts: np.ndarray,
    max_age_ns: int,
) -> np.ndarray:
    """Value of the latest event with ts <= query ts and age <= ``max_age_ns``."""
    right = pd.DataFrame(
        {
            "cik": ev_cik.astype(np.int64),
            "ts": ev_ts.astype(np.int64),
            "value": ev_val,
            "ev_ts": ev_ts.astype(np.int64),
        }
    ).sort_values("ts", kind="stable")
    q = pd.DataFrame(
        {"cik": q_cik.astype(np.int64), "ts": q_ts.astype(np.int64), "qi": np.arange(len(q_cik))}
    ).sort_values("ts", kind="stable")
    m = pd.merge_asof(q, right, on="ts", by="cik", direction="backward").sort_values("qi")
    v = m["value"].to_numpy(np.float64)
    age = q_ts - m["ev_ts"].fillna(np.iinfo(np.int64).min // 2).to_numpy(np.int64)
    return np.where(age <= max_age_ns, v, np.nan)


@dataclass
class FundQuery:
    """Firm-month queries for fundamentals: one row per panel row with a mapped CIK.

    ``split_to_T(dday)`` returns, per query, the product of the split ratios recorded on sessions
    after day ``dday`` (days since epoch) up to and including T (puts a share count dated ``dday``
    on T's share basis)."""

    cik: np.ndarray  # int64
    ts: np.ndarray  # int64 ns UTC (close of T)
    mk: np.ndarray  # int64 month index of T
    day: np.ndarray  # int64 days since epoch of T
    ts_12m: np.ndarray  # int64 ns UTC (close of the month-end 12 months earlier)
    raw_close: np.ndarray  # float64 traded price at T
    split_to_T: Callable[[np.ndarray], np.ndarray]


def group_facts(prepared: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Prepared facts (:func:`prepare_facts`) -> {tag group: rows} (see :data:`TAG_GROUPS`)."""
    group_of = {t: g for g, tags in TAG_GROUPS.items() for t in tags}
    f = prepared.assign(group=prepared["tag"].map(group_of))
    f = f[f["group"].notna()]
    return {str(g): d.drop(columns=["group"]).reset_index(drop=True) for g, d in f.groupby("group")}


def make_fund_query(
    pm: PriceMatrices, me_rows: np.ndarray, pos: np.ndarray, firm: np.ndarray, cik: np.ndarray
) -> FundQuery:
    """Queries for firm-months (month-end position ``pos`` into ``me_rows``, firm column, CIK):
    T close timestamp, month index, the close 12 month-ends earlier, the traded price and the
    split factor that puts a share count dated ``dday`` on T's share basis."""
    pos = np.asarray(pos, dtype=np.int64)
    firm = np.asarray(firm, dtype=np.int64)
    t_rows = me_rows[pos]
    dates = pm.sessions[t_rows]
    sess_days = pm.sessions.values.astype("datetime64[D]").astype(np.int64)
    lsc = pm.log_split_cum

    def split_to_t(dday: np.ndarray) -> np.ndarray:
        r0 = np.searchsorted(sess_days, np.asarray(dday, dtype=np.int64), side="right") - 1
        base = np.where(r0 >= 0, lsc[np.clip(r0, 0, None), firm], 0.0)
        return np.exp(lsc[t_rows, firm] - base)

    prev = np.clip(pos - 12, 0, None)
    return FundQuery(
        cik=np.asarray(cik, dtype=np.int64),
        ts=close_ts_utc_ns(dates),
        mk=month_index(dates),
        day=dates.values.astype("datetime64[D]").astype(np.int64),
        ts_12m=close_ts_utc_ns(pm.sessions[me_rows[prev]]),
        raw_close=pm.raw_close[t_rows, firm].astype(np.float64),
        split_to_T=split_to_t,
    )


def fundamental_features(
    facts: Mapping[str, pd.DataFrame],
    q: FundQuery,
    *,
    stale_months: int = STALE_MONTHS,
    sue_ev: pd.DataFrame | None = None,
    sue_max_age_days: int = 200,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fundamental features for the queries. ``facts[group]`` = prepared facts of the group's tags
    (see :data:`TAG_GROUPS`). Returns (features, tag_present) where ``tag_present[group]`` marks
    queries with a non-stale PIT value of the group (for the coverage rule)."""
    n = len(q.cik)
    nanv = np.full(n, np.nan)

    def fresh(mk: np.ndarray) -> np.ndarray:
        return (mk >= 0) & (q.mk - mk <= stale_months)

    def sub(group: str, tag: str | None = None, qtrs: int | None = None) -> Facts:
        df = facts.get(group)
        if df is None or df.empty:
            return Facts(
                np.array([], np.int64),
                np.array([], np.int64),
                np.array([], np.int64),
                np.array([], np.float64),
            )
        uom = TAG_UOM.get(group, "USD")
        m = df["uom"] == uom
        if tag is not None:
            m &= df["tag"] == tag
        if qtrs is not None:
            m &= df["qtrs"] == qtrs
        d = df.loc[m, ["cik", "mk", "dday", "ts", "value", "tag"]]
        if group == "shares":  # cover-page date: order and age by the exact day
            d = d.assign(mk=d["dday"])
        d = d.drop_duplicates(["cik", "mk", "ts"], keep="last")
        return Facts.from_frame(d)

    present: dict[str, np.ndarray] = {}
    inst: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for g in ("equity", "assets", "liabilities", "cash"):
        v, mk = pit_latest(sub(g, qtrs=0), q.cik, q.ts)
        ok = fresh(mk)
        inst[g] = (np.where(ok, v, np.nan), mk)
        present[g] = ok & np.isfinite(v)
    # shares (dei cover page), on T's split basis
    fs = sub("shares", qtrs=0)
    s_v, s_day = pit_latest(fs, q.cik, q.ts)
    s_ok = (s_day >= 0) & (q.day - s_day <= int(stale_months * 30.44)) & (s_v > 0)
    shares_T = np.where(s_ok, s_v * q.split_to_T(np.where(s_ok, s_day, q.day)), np.nan)
    present["shares"] = s_ok
    s_old, s_old_day = pit_latest(fs, q.cik, q.ts_12m)
    old_ok = (s_old_day >= 0) & (s_old > 0) & (s_day - s_old_day >= 274) & s_ok
    shares_old = np.where(old_ok, s_old * q.split_to_T(np.where(old_ok, s_old_day, q.day)), np.nan)
    # flows TTM
    flows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for g in ("net_income", "cost_of_revenue", "ocf"):
        tags = TAG_GROUPS[g]
        v, mk = pit_ttm(sub(g, tags[0], 1), sub(g, tags[0], 4), q.cik, q.ts)
        ok = fresh(mk)
        flows[g] = (np.where(ok, v, np.nan), mk)
        present[g] = ok & np.isfinite(v)
    best_v, best_mk = nanv.copy(), np.full(n, -1, dtype=np.int64)
    for tag in TAG_GROUPS["revenue"]:
        v, mk = pit_ttm(sub("revenue", tag, 1), sub("revenue", tag, 4), q.cik, q.ts)
        ok = fresh(mk) & np.isfinite(v)
        take = ok & (mk > best_mk)
        best_v = np.where(take, v, best_v)
        best_mk = np.where(take, mk, best_mk)
    flows["revenue"] = (best_v, best_mk)
    present["revenue"] = np.isfinite(best_v)
    mcap = q.raw_close * shares_T
    with np.errstate(invalid="ignore", divide="ignore"):
        be, be_mk = inst["equity"]
        at, at_mk = inst["assets"]
        li, li_mk = inst["liabilities"]
        ca, ca_mk = inst["cash"]
        ni = flows["net_income"][0]
        rev, rev_mk = flows["revenue"]
        cogs, cogs_mk = flows["cost_of_revenue"]
        ocf = flows["ocf"][0]
        at_pos = np.where(at > 0, at, np.nan)
        feats = {
            "log_mcap": np.where(mcap > 0, np.log(mcap), np.nan),
            "bm": be / mcap,
            "ep": ni / mcap,
            "sp": rev / mcap,
            "gp": np.where(rev_mk == cogs_mk, (rev - cogs) / at_pos, np.nan),
            "accruals": (ni - ocf) / at_pos,
            "nsi": np.log(shares_T / shares_old),
            "leverage": np.where(li_mk == at_mk, li / at_pos, np.nan),
            "cash_assets": np.where(ca_mk == at_mk, ca / at_pos, np.nan),
            "roe": ni / np.where(be != 0, be, np.nan),
        }
        at_old = pit_for_period(sub("assets", qtrs=0), q.cik, at_mk - 12, q.ts)
        feats["asset_growth"] = np.where(at_old > 0, at / at_old - 1.0, np.nan)
    eps_v, eps_mk = pit_latest(sub("eps", qtrs=1), q.cik, q.ts)
    present["eps"] = fresh(eps_mk) & np.isfinite(eps_v)
    if sue_ev is not None and len(sue_ev):
        feats["sue"] = latest_event_value(
            sue_ev["cik"].to_numpy(np.int64),
            utc_ns(sue_ev["ev_ts"])
            if not np.issubdtype(sue_ev["ev_ts"].dtype, np.integer)
            else sue_ev["ev_ts"].to_numpy(np.int64),
            sue_ev["sue"].to_numpy(np.float64),
            q.cik,
            q.ts,
            sue_max_age_days * 86_400 * 10**9,
        )
    else:
        feats["sue"] = nanv.copy()
    for k, v in feats.items():
        feats[k] = np.where(np.isfinite(v), v, np.nan)
    return feats, present


# ---------------------------------------------------------------------------
# Within-date preprocessing
# ---------------------------------------------------------------------------
def date_bounds(date_ids: np.ndarray) -> np.ndarray:
    """Start offsets of each date block (rows sorted by date) plus the total length."""
    d = np.asarray(date_ids)
    if len(d) == 0:
        return np.array([0], dtype=np.int64)
    if np.any(d[1:] < d[:-1]):
        raise ValueError("rows must be sorted by date")
    cuts = np.flatnonzero(d[1:] != d[:-1]) + 1
    return np.r_[0, cuts, len(d)].astype(np.int64)


def rank_unit(v: np.ndarray) -> np.ndarray:
    """Average ranks of the finite values mapped to [-0.5, +0.5]; NaN stays NaN; one value -> 0."""
    out = np.full(len(v), np.nan, dtype=np.float64)
    ok = np.isfinite(v)
    n = int(ok.sum())
    if n == 0:
        return out
    if n == 1:
        out[ok] = 0.0
        return out
    r = pd.Series(v[ok]).rank(method="average").to_numpy()
    out[ok] = (r - 1.0) / (n - 1.0) - 0.5
    return out


def normalize_within_date(
    values: np.ndarray, bounds: np.ndarray, *, lo_q: float = 0.01, hi_q: float = 0.99
) -> tuple[np.ndarray, np.ndarray]:
    """Winsorise at the 1st/99th percentile, rank, map to [-0.5, +0.5] within each date; missing ->
    0 (the within-date median rank). Returns (normalised float32, missing indicator float32)."""
    v = np.asarray(values, dtype=np.float64)
    out = np.zeros(len(v), dtype=np.float32)
    miss = (~np.isfinite(v)).astype(np.float32)
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        x = v[a:b]
        ok = np.isfinite(x)
        if ok.sum() >= 2:
            lo, hi = np.quantile(x[ok], [lo_q, hi_q])
            x = np.where(ok, np.clip(x, lo, hi), np.nan)
        r = rank_unit(x)
        out[a:b] = np.where(np.isfinite(r), r, 0.0)
    return out, miss


def rank_within_dates(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Within-date rank of ``x`` mapped to [-0.5, +0.5] (no winsorising)."""
    out = np.zeros(len(x), dtype=np.float64)
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        r = rank_unit(np.asarray(x[a:b], dtype=np.float64))
        out[a:b] = np.where(np.isfinite(r), r, 0.0)
    return out


# ---------------------------------------------------------------------------
# Expanding-window splitter with embargo and validation slice
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Split:
    """Inclusive target-month-index ranges for one refit (OOS year ``year``)."""

    year: int
    fit: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]
    embargoed: tuple[int, ...]


def expanding_splits(
    oos_years: Sequence[int],
    first_target_mi: int,
    *,
    embargo: int = 1,
    val_len: int = 12,
    fit_val_gap: int = 1,
) -> list[Split]:
    """OOS year Y = target months Y-01..Y-12. Training window = ``first_target_mi`` .. (first test
    month - 1 - ``embargo``); validation = its last ``val_len`` months; fit = everything before the
    validation slice minus ``fit_val_gap`` months."""
    out = []
    for y in oos_years:
        t0 = int(y) * 12
        win_end = t0 - 1 - embargo
        val = (win_end - val_len + 1, win_end)
        fit_end = val[0] - 1 - fit_val_gap
        if fit_end < first_target_mi:
            raise ValueError(f"no fit months for OOS year {y}")
        emb = tuple(range(win_end + 1, t0)) + tuple(range(fit_end + 1, val[0]))
        out.append(Split(int(y), (int(first_target_mi), fit_end), val, (t0, t0 + 11), emb))
    return out


def mask_months(mi: np.ndarray, rng_: tuple[int, int]) -> np.ndarray:
    return (mi >= rng_[0]) & (mi <= rng_[1])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def ic_by_date(pred: np.ndarray, y: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Spearman correlation per date block (ordinal ranks of pred; average ranks of y)."""
    out = np.full(len(bounds) - 1, np.nan)
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        if b - a < 3:
            continue
        p = pd.Series(pred[a:b]).rank().to_numpy()
        q = pd.Series(y[a:b]).rank().to_numpy()
        p -= p.mean()
        q -= q.mean()
        den = np.sqrt((p * p).sum() * (q * q).sum())
        out[i] = float((p * q).sum() / den) if den > 0 else np.nan
    return out


class FastIC:
    """Mean within-date Spearman IC against a fixed target (pre-ranked once); used for early
    stopping. ``__call__(pred)`` -> mean IC over date blocks."""

    def __init__(self, y: np.ndarray, bounds: np.ndarray) -> None:
        self.bounds = bounds
        self.yr = []
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            q = pd.Series(y[a:b]).rank().to_numpy()
            q = q - q.mean()
            self.yr.append(q / np.sqrt((q * q).sum()))

    def per_date(self, pred: np.ndarray) -> np.ndarray:
        out = np.empty(len(self.yr))
        for i, (a, b) in enumerate(zip(self.bounds[:-1], self.bounds[1:], strict=True)):
            p = np.argsort(np.argsort(pred[a:b], kind="stable"), kind="stable").astype(np.float64)
            p -= p.mean()
            s = np.sqrt((p * p).sum())
            out[i] = float((p * self.yr[i]).sum() / s) if s > 0 else 0.0
        return out

    def __call__(self, pred: np.ndarray) -> float:
        return float(np.mean(self.per_date(pred)))


def ic_summary(ics: np.ndarray) -> dict[str, float]:
    x = np.asarray(ics, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {
            "ic_mean": float("nan"),
            "ic_sd": float("nan"),
            "ic_t": float("nan"),
            "icir": float("nan"),
            "n_months": int(len(x)),
            "share_positive": float("nan"),
        }
    sd = x.std(ddof=1)
    return {
        "ic_mean": float(x.mean()),
        "ic_sd": float(sd),
        "ic_t": float(x.mean() / sd * np.sqrt(len(x))),
        "icir": float(x.mean() / sd),
        "n_months": int(len(x)),
        "share_positive": float((x > 0).mean()),
    }


def calibration_slope(pred: np.ndarray, r: np.ndarray) -> float:
    """OLS slope through the origin of r on pred (maps rank-unit predictions to return units)."""
    ok = np.isfinite(pred) & np.isfinite(r)
    den = float((pred[ok] ** 2).sum())
    return float((pred[ok] * r[ok]).sum() / den) if den > 0 else 0.0


def oos_r2(pred_ret: np.ndarray, r: np.ndarray) -> float:
    """GKX out-of-sample R^2 against a zero forecast: 1 - sum (r - f)^2 / sum r^2."""
    ok = np.isfinite(pred_ret) & np.isfinite(r)
    return float(1.0 - ((r[ok] - pred_ret[ok]) ** 2).sum() / (r[ok] ** 2).sum())


# ---------------------------------------------------------------------------
# Decile long-short portfolios
# ---------------------------------------------------------------------------
@dataclass
class LSResult:
    gross: np.ndarray
    net: np.ndarray
    long_ret: np.ndarray
    short_ret: np.ndarray
    to_long: np.ndarray
    to_short: np.ndarray
    n_long: np.ndarray
    n_short: np.ndarray


def decile_long_short(
    pred: np.ndarray,
    ret: np.ndarray,
    firm: np.ndarray,
    bounds: np.ndarray,
    cost_per_side: float,
    n_firms: int,
    *,
    n_bins: int = 10,
) -> LSResult:
    """Monthly EW long top-decile / short bottom-decile of ``pred`` within each date block,
    bought at the formation close + 1 session, held one month (``ret`` = holding return).
    Turnover per leg = sum |w_new - w_old drifted by the holding return|; cost = cost per side x
    (turnover long + turnover short). Consecutive blocks must be consecutive months."""
    nd = len(bounds) - 1
    res = {
        k: np.zeros(nd)
        for k in (
            "gross",
            "net",
            "long_ret",
            "short_ret",
            "to_long",
            "to_short",
            "n_long",
            "n_short",
        )
    }
    w_prev = {"L": np.zeros(n_firms), "S": np.zeros(n_firms)}
    for i in range(nd):
        a, b = int(bounds[i]), int(bounds[i + 1])
        p = pred[a:b]
        r = np.nan_to_num(ret[a:b])
        f = firm[a:b]
        n = b - a
        order = np.argsort(p, kind="stable")
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(1, n + 1)
        binv = np.ceil(rank * n_bins / n).astype(np.int64)
        for leg, sel in (("L", binv == n_bins), ("S", binv == 1)):
            k = int(sel.sum())
            w_new = np.zeros(n_firms)
            if k:
                w_new[f[sel]] = 1.0 / k
            lr = float(r[sel].mean()) if k else 0.0
            to = float(np.abs(w_new - w_prev[leg]).sum())
            drift = np.zeros(n_firms)
            if k:
                drift[f[sel]] = (1.0 / k) * (1.0 + r[sel]) / (1.0 + lr) if (1.0 + lr) != 0 else 0.0
            w_prev[leg] = drift
            if leg == "L":
                res["long_ret"][i], res["to_long"][i], res["n_long"][i] = lr, to, k
            else:
                res["short_ret"][i], res["to_short"][i], res["n_short"][i] = lr, to, k
    res["gross"] = res["long_ret"] - res["short_ret"]
    res["net"] = res["gross"] - cost_per_side * (res["to_long"] + res["to_short"])
    return LSResult(**res)


def permute_within_dates(x: np.ndarray, bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(x).copy()
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        out[a:b] = out[a:b][rng.permutation(b - a)]
    return out


def monthly_sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(12.0)) if sd > 0 else float("nan")


def monthly_stats(
    x: np.ndarray, months: np.ndarray, mktrf: np.ndarray, *, nw_lags: int = 3
) -> dict[str, Any]:
    """Sharpe (x sqrt 12), arithmetic annualised mean (x 12), vol, max drawdown of the compounded
    series, alpha / beta on Mkt-RF (Newey-West), per calendar year (compounded)."""
    from eqxs.equity_events import ols_alpha_beta

    x = np.asarray(x, dtype=float)
    cum = np.cumprod(1.0 + x)
    dd = float((cum / np.maximum.accumulate(cum) - 1.0).min())
    ab = ols_alpha_beta(x, np.asarray(mktrf, dtype=float), nw_lags=nw_lags)
    yrs = np.asarray(months) // 12
    per_year = []
    for y in np.unique(yrs):
        v = x[yrs == y]
        per_year.append(
            {
                "year": int(y),
                "ret_compounded": float(np.prod(1 + v) - 1),
                "months": int(len(v)),
                "sharpe": monthly_sharpe(v) if len(v) > 2 else float("nan"),
            }
        )
    pos = [r["ret_compounded"] > 0 for r in per_year]
    return {
        "sharpe": monthly_sharpe(x),
        "ann_return": float(x.mean() * 12.0),
        "ann_vol": float(x.std(ddof=1) * np.sqrt(12.0)),
        "max_dd": dd,
        "alpha_ann": float(ab["alpha_daily"] * 12.0),
        "alpha_t": ab["alpha_t"],
        "beta": ab["beta"],
        "beta_t": ab["beta_t"],
        "beta_flag": bool(abs(ab["beta"]) > 0.25),
        "per_year": per_year,
        "years_counted": len(per_year),
        "years_positive": int(sum(pos)),
        "pos_year_share": float(np.mean(pos)) if pos else float("nan"),
        "months": int(len(x)),
    }


def n2_block_bootstrap(
    x: np.ndarray, *, block: int = 3, n_draws: int = 1000, rng: np.random.Generator
) -> dict[str, float]:
    """Centred circular block bootstrap of a monthly series: p for H0 mean <= 0 and the MDE
    ((q95 - q20) of the centred resample means x 12)."""
    from eqxs.equity_events import circular_block_bootstrap_means

    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    boot = circular_block_bootstrap_means(x - m, block=block, n_draws=n_draws, rng=rng)
    return {
        "draws": n_draws,
        "block": block,
        "p": float((np.sum(boot >= m) + 1) / (n_draws + 1)),
        "se_ann_mean": float(boot.std(ddof=1) * 12.0),
        "mde_ann_return": float((np.quantile(boot, 0.95) - np.quantile(boot, 0.20)) * 12.0),
    }


def n1_permutation(
    pred: np.ndarray,
    ret: np.ndarray,
    firm: np.ndarray,
    bounds: np.ndarray,
    cost: float,
    n_firms: int,
    realised_sharpe: float,
    *,
    n_draws: int = 500,
    rng: np.random.Generator,
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Within-date permutation of the predictions through the same portfolio code path."""
    from eqxs.equity_events import mde_from_null, null_p

    sh, ar = [], []
    for i in range(n_draws):
        res = decile_long_short(
            permute_within_dates(pred, bounds, rng), ret, firm, bounds, cost, n_firms
        )
        sh.append(monthly_sharpe(res.net))
        ar.append(float(res.net.mean() * 12.0))
        if progress is not None and (i + 1) % 100 == 0:
            progress(i + 1)
    mde = mde_from_null(ar)
    return {
        "draws": n_draws,
        "p": null_p(sh, realised_sharpe),
        "null_sharpe_mean": float(np.mean(sh)),
        "null_sharpe_q95": float(np.quantile(sh, 0.95)),
        "null_ann_mean": float(np.mean(ar)),
        "null_ann_q95": float(np.quantile(ar, 0.95)),
        "mde_ann_return": mde,
        "mde_excess_over_null_mean": float(mde - np.mean(ar)),
        "mde_sharpe": mde_from_null(sh),
    }


# ---------------------------------------------------------------------------
# Models (M0-M6) behind one interface
# ---------------------------------------------------------------------------
@dataclass
class FitData:
    """Fit and validation slices (rows sorted by date; ``b*`` = date-block bounds)."""

    Xf: np.ndarray
    yf: np.ndarray
    bf: np.ndarray
    Xv: np.ndarray
    yv: np.ndarray
    bv: np.ndarray
    columns: list[str]


class Model:
    """Interface: ``fit(fd, seed=...)`` -> info dict (chosen hyperparameters, validation IC,
    seconds); ``predict(X, bounds)`` -> float array (rank units)."""

    name: str = "model"

    def fit(self, fd: FitData, *, seed: int) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, X: np.ndarray, bounds: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class RidgeModel(Model):
    """Ridge on a column subset (all columns when ``features`` is None); alpha on validation IC."""

    def __init__(
        self, name: str, alphas: Sequence[float], features: Sequence[str] | None = None
    ) -> None:
        self.name, self.alphas, self.features = name, list(alphas), features
        self.idx: np.ndarray | None = None
        self.model: Any = None

    def _cols(self, columns: Sequence[str]) -> np.ndarray:
        if self.features is None:
            return np.arange(len(columns))
        missing = [f for f in self.features if f not in columns]
        if missing:
            raise KeyError(f"{self.name}: missing features {missing}")
        return np.array([list(columns).index(f) for f in self.features])

    def fit(self, fd: FitData, *, seed: int) -> dict[str, Any]:
        from sklearn.linear_model import Ridge

        t0 = time.time()
        self.idx = self._cols(fd.columns)
        ic_v = FastIC(fd.yv, fd.bv)
        scores = {}
        best = None
        for a in self.alphas:
            m = Ridge(alpha=a).fit(fd.Xf[:, self.idx], fd.yf)
            s = ic_v(m.predict(fd.Xv[:, self.idx]))
            scores[str(a)] = s
            if best is None or s > best[0]:
                best = (s, a, m)
        assert best is not None
        self.model = best[2]
        return {
            "alpha": best[1],
            "val_ic": best[0],
            "grid_val_ic": scores,
            "seconds": time.time() - t0,
            "features": [fd.columns[i] for i in self.idx] if self.features is not None else "all",
        }

    def predict(self, X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        return self.model.predict(X[:, self.idx]).astype(np.float64)


XGB_GRID: tuple[dict[str, float], ...] = tuple(
    {
        "max_depth": d,
        "learning_rate": lr,
        "min_child_weight": mcw,
        "subsample": ss,
        "colsample_bytree": cs,
    }
    for d, lr, mcw, ss, cs in (
        (2, 0.05, 100, 0.8, 0.8),
        (3, 0.05, 500, 0.5, 0.5),
        (4, 0.03, 1000, 0.7, 0.6),
        (4, 0.10, 100, 0.5, 0.8),
        (6, 0.03, 500, 0.8, 0.5),
        (6, 0.10, 1000, 0.5, 0.7),
        (3, 0.10, 50, 0.8, 0.6),
        (5, 0.05, 200, 0.6, 0.7),
    )
)


class XGBModel(Model):
    """XGBoost regression (``rank=False``, M2) or ``rank:pairwise`` with date groups (M4) on the
    rank target; grid + early stopping on validation IC."""

    def __init__(
        self,
        name: str,
        *,
        rank: bool,
        grid: Sequence[Mapping[str, float]] = XGB_GRID,
        max_rounds: int = 2000,
        patience: int = 50,
        nthread: int = 8,
    ) -> None:
        self.name, self.rank, self.grid = name, rank, list(grid)
        self.max_rounds, self.patience, self.nthread = max_rounds, patience, nthread
        self.booster: Any = None
        self.best_iter = 0

    def _dmat(self, X: np.ndarray, y: np.ndarray | None, bounds: np.ndarray) -> Any:
        import xgboost as xgb

        kw: dict[str, Any] = {}
        if self.rank:
            kw["qid"] = np.repeat(np.arange(len(bounds) - 1), np.diff(bounds))
        lab = None if y is None else (y + 0.5 if self.rank else y)
        return xgb.DMatrix(X, label=lab, nthread=self.nthread, **kw)

    def fit(self, fd: FitData, *, seed: int) -> dict[str, Any]:
        import xgboost as xgb

        t0 = time.time()
        dfit = self._dmat(fd.Xf, fd.yf, fd.bf)
        dval = self._dmat(fd.Xv, fd.yv, fd.bv)
        ic_v = FastIC(fd.yv, fd.bv)
        nv = len(fd.yv)

        def metric(predt: np.ndarray, d: Any) -> tuple[str, float]:
            return "ic", ic_v(predt) if len(predt) == nv else float("nan")

        base: dict[str, Any] = {
            "tree_method": "hist",
            "max_bin": 64,
            "nthread": self.nthread,
            "seed": seed,
            "disable_default_eval_metric": 1,
            "verbosity": 0,
        }
        if self.rank:
            base.update(
                {
                    "objective": "rank:pairwise",
                    "lambdarank_pair_method": "mean",
                    "lambdarank_num_pair_per_sample": 8,
                }
            )
        else:
            base["objective"] = "reg:squarederror"
        results = []
        best = None
        for g in self.grid:
            params = {**base, **g}
            bst = xgb.train(
                params,
                dfit,
                num_boost_round=self.max_rounds,
                evals=[(dval, "val")],
                custom_metric=metric,
                maximize=True,
                early_stopping_rounds=self.patience,
                verbose_eval=False,
            )
            s = float(bst.best_score)
            results.append({**g, "val_ic": s, "best_iteration": int(bst.best_iteration)})
            if best is None or s > best[0]:
                best = (s, g, bst, int(bst.best_iteration))
        assert best is not None
        self.booster, self.best_iter = best[2], best[3]
        return {
            "params": dict(best[1]),
            "best_iteration": best[3],
            "val_ic": best[0],
            "grid": results,
            "seconds": time.time() - t0,
        }

    def predict(self, X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        import xgboost as xgb

        d = xgb.DMatrix(X, nthread=self.nthread)
        return self.booster.predict(d, iteration_range=(0, self.best_iter + 1)).astype(np.float64)


# --- torch models ---------------------------------------------------------------------------------
# torch's libomp and XGBoost's libomp cannot share a process on macOS (segfault / deadlock),
# so every
# torch call (training AND prediction) runs in loky worker processes that never import XGBoost.
# Arrays travel through .npy files (memory-mapped by the workers); states through torch.save files.
def torch_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def _executor(n_jobs: int) -> Any:
    from joblib.externals.loky import get_reusable_executor

    return get_reusable_executor(max_workers=n_jobs, timeout=900)


def _run_jobs(fn: Callable[..., Any], jobs: Sequence[Mapping[str, Any]], n_jobs: int) -> list[Any]:
    ex = _executor(n_jobs)
    futs = [ex.submit(fn, **j) for j in jobs]
    return [f.result() for f in futs]


class _Workdir:
    def __init__(self) -> None:
        import tempfile

        self.path = tempfile.mkdtemp(prefix="eqml_torch_")
        self.n = 0

    def save(self, arr: np.ndarray, tag: str) -> str:
        self.n += 1
        fp = os.path.join(self.path, f"{tag}_{self.n}.npy")
        np.save(fp, np.ascontiguousarray(arr))
        return fp

    def file(self, tag: str) -> str:
        self.n += 1
        return os.path.join(self.path, f"{tag}_{self.n}.pt")

    def clear(self) -> None:
        import shutil

        shutil.rmtree(self.path, ignore_errors=True)


def _load(fp: str) -> np.ndarray:
    return np.load(fp, mmap_mode="r")


def _nn3_module(p: int) -> Any:
    import torch.nn as nn

    layers: list[Any] = []
    width = p
    for h in (32, 16, 8):
        layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.ReLU()]
        width = h
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def _forward_chunks(model: Any, X: np.ndarray, chunk: int = 100_000) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(X), chunk):
            out.append(
                model(torch.from_numpy(np.ascontiguousarray(X[a : a + chunk], dtype=np.float32)))
                .squeeze(-1)
                .numpy()
            )
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def train_nn3_seed(
    Xf: np.ndarray,
    yf: np.ndarray,
    Xv: np.ndarray,
    yv: np.ndarray,
    bv: np.ndarray,
    lam: float,
    lr: float,
    seed: int,
    *,
    max_epochs: int = 100,
    patience: int = 5,
    batch: int = 10_000,
    threads: int = 1,
) -> dict[str, Any]:
    """One GKX NN3 fit: MSE + lam * sum |W| (Linear weights), Adam, early stopping on validation IC.
    Must run in a process without XGBoost (see :func:`_run_jobs`)."""
    import torch

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _nn3_module(Xf.shape[1])
    weights = [m.weight for m in model if isinstance(m, torch.nn.Linear)]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.from_numpy(np.ascontiguousarray(Xf, dtype=np.float32))
    Y = torch.from_numpy(np.ascontiguousarray(yf, dtype=np.float32))
    ic_v = FastIC(np.asarray(yv), np.asarray(bv))
    best_ic, best_state, best_ep, bad, finite = -np.inf, None, -1, 0, True
    ep = 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(X)))
        for a in range(0, len(X), batch):
            idx = perm[a : a + batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            out = model(X[idx]).squeeze(-1)
            loss = ((out - Y[idx]) ** 2).mean() + lam * sum(w.abs().sum() for w in weights)
            if not torch.isfinite(loss):
                finite = False
                break
            loss.backward()
            opt.step()
        if not finite:
            break
        sc = ic_v(_forward_chunks(model, Xv))
        if sc > best_ic:
            best_ic, best_ep, bad = sc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    return {
        "state": best_state,
        "val_ic": float(best_ic),
        "best_epoch": best_ep,
        "epochs": ep + 1,
        "finite": finite and best_state is not None,
        "lam": lam,
        "lr": lr,
        "seed": seed,
    }


def _job_nn3(paths: Mapping[str, str], state_out: str, **kw: Any) -> dict[str, Any]:
    import torch

    r = train_nn3_seed(
        _load(paths["Xf"]),
        _load(paths["yf"]),
        _load(paths["Xv"]),
        _load(paths["yv"]),
        _load(paths["bv"]),
        **kw,
    )
    if r["state"] is not None:
        torch.save(r["state"], state_out)
    r["state"] = state_out if r["state"] is not None else None
    return r


def _set_module(
    p: int, d_model: int = 64, heads: int = 4, ff: int = 128, layers: int = 2, dropout: float = 0.1
) -> Any:
    import torch.nn as nn

    class SetTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(p, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model, heads, ff, dropout, batch_first=True, norm_first=True
            )
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x: Any) -> Any:  # x: (1, n_firms, p)
            return self.head(self.norm(self.enc(self.inp(x)))).squeeze(-1)

    return SetTransformer()


def _set_predict(model: Any, X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    out = np.zeros(len(X))
    with torch.no_grad():
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            x = torch.from_numpy(np.ascontiguousarray(X[a:b], dtype=np.float32))[None]
            out[a:b] = model(x)[0].numpy()
    return out


def train_set_seed(
    Xf: np.ndarray,
    yf: np.ndarray,
    bf: np.ndarray,
    Xv: np.ndarray,
    yv: np.ndarray,
    bv: np.ndarray,
    seed: int,
    *,
    lr: float = 1e-3,
    max_epochs: int = 20,
    patience: int = 3,
    accum: int = 4,
    max_firms: int = 3000,
    threads: int = 1,
) -> dict[str, Any]:
    """One M5 fit: one date = one set; MSE per date; gradients accumulated over ``accum`` dates;
    dates shuffled each epoch; early stopping on validation IC (best epoch restored)."""
    import torch

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _set_module(Xf.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bv = np.asarray(bv)
    ic_v = FastIC(np.asarray(yv), bv)
    bf = np.asarray(bf)
    nd = len(bf) - 1
    best_ic, best_state, best_ep, bad, finite = -np.inf, None, -1, 0, True
    ep = 0
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        order = rng.permutation(nd)
        opt.zero_grad()
        for j, di in enumerate(order):
            a, b = int(bf[di]), int(bf[di + 1])
            idx = np.arange(a, b)
            if len(idx) > max_firms:
                idx = np.sort(rng.choice(idx, size=max_firms, replace=False))
            x = torch.from_numpy(np.ascontiguousarray(Xf[idx], dtype=np.float32))[None]
            y = torch.from_numpy(np.ascontiguousarray(yf[idx], dtype=np.float32))[None]
            loss = ((model(x) - y) ** 2).mean() / accum
            if not torch.isfinite(loss):
                finite = False
                break
            loss.backward()
            if (j + 1) % accum == 0 or j + 1 == nd:
                opt.step()
                opt.zero_grad()
        if not finite:
            break
        sc = ic_v(_set_predict(model, Xv, bv))
        if sc > best_ic:
            best_ic, best_ep, bad = sc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    return {
        "state": best_state,
        "val_ic": float(best_ic),
        "best_epoch": best_ep,
        "epochs": ep + 1,
        "finite": finite and best_state is not None,
        "seed": seed,
        "seconds": time.time() - t0,
    }


def _job_set(paths: Mapping[str, str], state_out: str, **kw: Any) -> dict[str, Any]:
    import torch

    r = train_set_seed(
        _load(paths["Xf"]),
        _load(paths["yf"]),
        _load(paths["bf"]),
        _load(paths["Xv"]),
        _load(paths["yv"]),
        _load(paths["bv"]),
        **kw,
    )
    if r["state"] is not None:
        torch.save(r["state"], state_out)
    r["state"] = state_out if r["state"] is not None else None
    return r


def _job_predict(
    kind: str, states: Sequence[str], x_path: str, b_path: str, threads: int = 1
) -> np.ndarray:
    """Mean prediction over the saved states (``kind`` 'nn3' or 'set')."""
    import torch

    torch.set_num_threads(threads)
    X, b = _load(x_path), np.asarray(_load(b_path))
    preds = []
    for sp in states:
        st = torch.load(sp, weights_only=True)
        if kind == "nn3":
            m = _nn3_module(X.shape[1])
            m.load_state_dict(st)
            preds.append(_forward_chunks(m, X))
        else:
            m = _set_module(X.shape[1])
            m.load_state_dict(st)
            preds.append(_set_predict(m, X, b))
    return np.mean(preds, axis=0)


class _TorchModel(Model):
    kind = "nn3"

    def __init__(self) -> None:
        self.work = _Workdir()
        self.states: list[str] = []
        self.n_jobs = 1

    def _fd_paths(self, fd: FitData) -> dict[str, str]:
        w = self.work
        return {
            "Xf": w.save(fd.Xf.astype(np.float32), "Xf"),
            "yf": w.save(fd.yf.astype(np.float32), "yf"),
            "bf": w.save(fd.bf, "bf"),
            "Xv": w.save(fd.Xv.astype(np.float32), "Xv"),
            "yv": w.save(fd.yv, "yv"),
            "bv": w.save(fd.bv, "bv"),
        }

    def predict(self, X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        xp, bp = self.work.save(X.astype(np.float32), "Xp"), self.work.save(bounds, "bp")
        out = _run_jobs(
            _job_predict,
            [{"kind": self.kind, "states": self.states, "x_path": xp, "b_path": bp, "threads": 4}],
            self.n_jobs,
        )[0]
        for fp in (xp, bp):
            os.remove(fp)
        return out

    def predict_many(self, Xs: Sequence[np.ndarray], bounds: np.ndarray) -> list[np.ndarray]:
        """Several inputs with the same bounds, spread over the workers."""
        bp = self.work.save(bounds, "bp")
        xps = [self.work.save(X.astype(np.float32), "Xp") for X in Xs]
        out = _run_jobs(
            _job_predict,
            [{"kind": self.kind, "states": self.states, "x_path": xp, "b_path": bp} for xp in xps],
            self.n_jobs,
        )
        for fp in [bp, *xps]:
            os.remove(fp)
        return out

    def close(self) -> None:
        self.work.clear()


class NN3Model(_TorchModel):
    """Gu-Kelly-Xiu NN3 (32-16-8, BN, ReLU, L1), grid on validation with 2-seed ensembles, the
    chosen point completed to ``n_seeds`` seeds; prediction = mean over seeds."""

    kind = "nn3"

    def __init__(
        self,
        name: str = "M3",
        *,
        lams: Sequence[float] = (1e-5, 1e-4, 1e-3),
        lrs: Sequence[float] = (1e-3, 1e-2),
        n_seeds: int = 10,
        select_seeds: int = 2,
        n_jobs: int = 4,
        max_epochs: int = 100,
        patience: int = 5,
        batch: int = 10_000,
    ) -> None:
        super().__init__()
        self.name, self.lams, self.lrs = name, list(lams), list(lrs)
        self.n_seeds, self.select_seeds, self.n_jobs = n_seeds, select_seeds, n_jobs
        self.kw = {"max_epochs": max_epochs, "patience": patience, "batch": batch}

    def fit(self, fd: FitData, *, seed: int) -> dict[str, Any]:
        t0 = time.time()
        paths = self._fd_paths(fd)
        grid = [(lam, lr) for lam in self.lams for lr in self.lrs]
        jobs = [
            {
                "paths": paths,
                "state_out": self.work.file("nn3"),
                "lam": lam,
                "lr": lr,
                "seed": seed + 1000 * gi + s,
                **self.kw,
            }
            for gi, (lam, lr) in enumerate(grid)
            for s in range(min(self.select_seeds, self.n_seeds))
        ]
        runs = _run_jobs(_job_nn3, jobs, self.n_jobs)
        ic_v = FastIC(fd.yv, fd.bv)
        vp, bp = paths["Xv"], paths["bv"]
        scores = []
        for lam, lr in grid:
            rs = [r for r in runs if r["lam"] == lam and r["lr"] == lr and r["finite"]]
            if not rs:
                scores.append({"lam": lam, "lr": lr, "val_ic": float("nan")})
                continue
            pv = _run_jobs(
                _job_predict,
                [{"kind": "nn3", "states": [r["state"] for r in rs], "x_path": vp, "b_path": bp}],
                self.n_jobs,
            )[0]
            scores.append({"lam": lam, "lr": lr, "val_ic": ic_v(pv)})
        ok = [sc for sc in scores if np.isfinite(sc["val_ic"])]
        if not ok:
            raise FloatingPointError("NN3: every grid run diverged")
        best = max(ok, key=lambda sc: sc["val_ic"])
        keep = [
            r for r in runs if r["lam"] == best["lam"] and r["lr"] == best["lr"] and r["finite"]
        ]
        gi = grid.index((best["lam"], best["lr"]))
        n_more = max(self.n_seeds - len(keep), 0)
        more = _run_jobs(
            _job_nn3,
            [
                {
                    "paths": paths,
                    "state_out": self.work.file("nn3"),
                    "lam": best["lam"],
                    "lr": best["lr"],
                    "seed": seed + 1000 * gi + 100 + s,
                    **self.kw,
                }
                for s in range(n_more)
            ],
            self.n_jobs,
        )
        keep += [r for r in more if r["finite"]]
        self.states = [r["state"] for r in keep]
        pv = _run_jobs(
            _job_predict,
            [{"kind": "nn3", "states": self.states, "x_path": vp, "b_path": bp}],
            self.n_jobs,
        )[0]
        return {
            "lam": best["lam"],
            "lr": best["lr"],
            "grid": scores,
            "n_seeds": len(self.states),
            "seed_epochs": [r["best_epoch"] for r in keep],
            "val_ic": ic_v(pv),
            "n_diverged": int(sum(not r["finite"] for r in [*runs, *more])),
            "seconds": time.time() - t0,
        }


class SetTransformerModel(_TorchModel):
    """M5: 2-layer transformer encoder over the firms of each date, ``n_seeds`` seeds averaged."""

    kind = "set"

    def __init__(
        self,
        name: str = "M5",
        *,
        n_seeds: int = 5,
        n_jobs: int = 4,
        max_epochs: int = 20,
        patience: int = 3,
        accum: int = 4,
        max_firms: int = 3000,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.name, self.n_seeds, self.n_jobs = name, n_seeds, n_jobs
        self.kw = {
            "max_epochs": max_epochs,
            "patience": patience,
            "accum": accum,
            "max_firms": max_firms,
            "lr": lr,
        }

    def fit(self, fd: FitData, *, seed: int) -> dict[str, Any]:
        t0 = time.time()
        paths = self._fd_paths(fd)
        runs = _run_jobs(
            _job_set,
            [
                {"paths": paths, "state_out": self.work.file("set"), "seed": seed + s, **self.kw}
                for s in range(self.n_seeds)
            ],
            self.n_jobs,
        )
        retried = 0
        final = []
        for s, r in enumerate(runs):
            if not r["finite"]:
                retried += 1
                r = _run_jobs(
                    _job_set,
                    [
                        {
                            "paths": paths,
                            "state_out": self.work.file("set"),
                            "seed": seed + 5000 + s,
                            **self.kw,
                        }
                    ],
                    self.n_jobs,
                )[0]
                if not r["finite"]:
                    raise FloatingPointError("M5: non-finite loss twice in a row")
            final.append(r)
        self.states = [r["state"] for r in final]
        ic_v = FastIC(fd.yv, fd.bv)
        pv = _run_jobs(
            _job_predict,
            [
                {
                    "kind": "set",
                    "states": self.states,
                    "x_path": paths["Xv"],
                    "b_path": paths["bv"],
                    "threads": 4,
                }
            ],
            self.n_jobs,
        )[0]
        return {
            "n_seeds": len(self.states),
            "seed_best_epochs": [r["best_epoch"] for r in final],
            "seed_epochs": [r["epochs"] for r in final],
            "seed_val_ic": [r["val_ic"] for r in final],
            "seed_seconds": [r["seconds"] for r in final],
            "val_ic": ic_v(pv),
            "retried": retried,
            "seconds": time.time() - t0,
            **self.kw,
        }


def make_model(name: str, *, quick: bool = False, n_jobs: int = 4) -> Model:
    """Pre-declared model factory. ``quick`` (development only) cuts seeds to 2."""
    if name == "M0":
        return RidgeModel("M0", (0.1, 10.0, 1e3, 1e5), features=M0_FEATURES)
    if name == "M1":
        return RidgeModel("M1", (1.0, 10.0, 100.0, 1e3, 1e4, 1e5))
    if name == "M2":
        return XGBModel("M2", rank=False)
    if name == "M3":
        return NN3Model(
            "M3", n_seeds=2 if quick else 10, select_seeds=1 if quick else 2, n_jobs=n_jobs
        )
    if name == "M4":
        return XGBModel("M4", rank=True)
    if name == "M5":
        return SetTransformerModel("M5", n_seeds=2 if quick else 5, n_jobs=n_jobs)
    raise ValueError(name)


def ensemble(preds: Sequence[np.ndarray], bounds: np.ndarray) -> np.ndarray:
    """M6: mean of the within-date rank-normalised component predictions."""
    return np.mean([rank_within_dates(p, bounds) for p in preds], axis=0)
