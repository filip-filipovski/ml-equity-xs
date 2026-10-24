"""Calendar-time equity event studies (earnings-announcement premium, PEAD, announcement delay).

Pure functions used by the earnings-announcement studies (study 11, EAP / PEAD, and
study 12, announcement delay). Nothing here reads data: callers
pass session calendars, return matrices and event tables built from the
``eqxs.equities_xs`` loaders.

Conventions
-----------
* ``sessions`` is a sorted, tz-naive ``DatetimeIndex`` of trading dates; row ``t`` of a
  return matrix ``rets`` holds the close(t-1) -> close(t) return.
* A position "entered at the close of row e and held through the close of row x" earns
  the returns of rows ``e+1 .. x`` (inclusive). :func:`interval_mask` takes those
  holding rows directly (``first_row = e+1``, ``last_row = x``).
* Legs are equal-weighted across open positions and rebalanced every session. Turnover
  includes the drift rebalance: ``sum_i |w_i(t) - w_i(t-1)(1+r_i(t-1))/(1+r_leg(t-1))|``.
  Costs are ``cost_per_side * turnover`` on every leg, charged on the row whose weights
  were set. :class:`SpreadResult` also carries the entry/exit-only (minimal-trading)
  turnover and net: the same gross, charged only for names entering (their new weight)
  and leaving (their drifted weight). An implementable book's cost lies between the two.

Red-team fixes (2026-10-10). Each
behaviour change has a new default and a named legacy option that reproduces studies 11
and 12 as run:

* F1 clean event set: :func:`clean_event_mask` ``flags="pit"`` (legacy ``"legacy_any"``).
* F2 INTRADAY day 0: :func:`day0_from_acceptance` ``intraday="acceptance_session"``
  (legacy ``"next_session"``).
* F3 cost: both turnover figures in :class:`SpreadResult` / :class:`HedgedShortResult`.
* F4 years rule: :func:`event_study_verdict` ``years_rule="ex_best_year_positive"``
  (legacy ``"share_ge_075"``).
* F5 next-event exit: :func:`next_event_exit_cap` / :func:`delay_labels`
  ``mode="expected_next"`` (legacy ``"next_event_lookahead"``).
* F6 delay null: :func:`shuffle_delay_offsets` ``stratify="eligibility"`` (legacy
  ``"none"``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

ANN: float = 252.0
OPEN_MINUTES_ET: int = 9 * 60 + 30
CLOSE_MINUTES_ET: int = 16 * 60
INTRADAY_MODES: tuple[str, ...] = ("acceptance_session", "next_session")
EXIT_MODES: tuple[str, ...] = ("expected_next", "hold_through", "next_event_lookahead")
YEARS_RULES: tuple[str, ...] = ("ex_best_year_positive", "share_ge_075")
STRATIFY_MODES: tuple[str, ...] = ("eligibility", "none")
NO_CAP: int = np.iinfo(np.int64).max // 4   # "no exit cap" sentinel row for next_event_exit_cap


def clean_event_mask(ambiguous: pd.Series, *, flags: str | Sequence[str] = "pit") -> pd.Series:
    """Clean-event mask from the stored ``ambiguous`` column; see
    :func:`eqxs.earnings_calendar.clean_event_mask`. Default ``flags="pit"`` uses
    only flags known at formation (F1); ``"legacy_any"`` reproduces studies 11/12."""
    from eqxs.earnings_calendar import clean_event_mask as _clean

    return _clean(ambiguous, flags=flags)


# ---------------------------------------------------------------------------
# Calendars
# ---------------------------------------------------------------------------
def _naive_days(x: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(pd.to_datetime(x))
    if d.tz is not None:
        d = d.tz_localize(None)
    return d.normalize()


def day0_from_acceptance(acceptance_et: pd.Series, sessions: pd.DatetimeIndex,
                         cutoff_minutes: int = OPEN_MINUTES_ET, *,
                         intraday: str = "acceptance_session") -> pd.Series:
    """Day 0 (the reaction session) of an 8-K from its acceptance instant.

    * Before 09:30 ET on a trading day (BMO): that session.
    * At or after 16:00 ET (AMC), or on a weekend / holiday: the next session.
    * 09:30-16:00 ET on a trading day (INTRADAY): depends on ``intraday``.
      ``"acceptance_session"`` (default) -> that same session. The press release usually
      comes out before or during the session and the 8-K is accepted later the same day;
      the red team (2026-10-10, F2, r02) found the acceptance session carries the largest
      |AR| in 42.8 % of INTRADAY events against 13.9 % for the next session (mean |AR|
      398 vs 208 bps). ``"next_session"`` -> the next session (the rule studies 11 and 12
      ran with; legacy, reproduction only).

    Returns NaT when no later session exists. ``cutoff_minutes`` (minutes after midnight
    ET) moves the 09:30 cutoff; with ``intraday="acceptance_session"`` any acceptance on a
    session before ``max(cutoff_minutes, 16:00)`` maps to that session. See
    :func:`known_session_from_acceptance`.
    """
    if intraday not in INTRADAY_MODES:
        raise ValueError(f"intraday={intraday!r}: one of {INTRADAY_MODES}")
    if intraday == "acceptance_session":
        cutoff_minutes = max(cutoff_minutes, CLOSE_MINUTES_ET)
    a = pd.to_datetime(acceptance_et, utc=True).dt.tz_convert("America/New_York")
    day = _naive_days(a.dt.tz_localize(None))
    minutes = (a.dt.hour * 60 + a.dt.minute).to_numpy()
    sess = pd.DatetimeIndex(sessions)
    idx = sess.searchsorted(day, side="left")
    is_session = np.zeros(len(day), dtype=bool)
    ok = idx < len(sess)
    is_session[ok] = sess[idx[ok]] == day[ok]
    pre_open = minutes < cutoff_minutes
    out_idx = np.where(is_session & ~pre_open, idx + 1, idx)
    res = np.full(len(day), np.datetime64("NaT"), dtype="datetime64[ns]")
    good = out_idx < len(sess)
    res[good] = sess.values[out_idx[good]]
    return pd.Series(res, index=acceptance_et.index)


def known_session_from_acceptance(acceptance_et: pd.Series,
                                  sessions: pd.DatetimeIndex) -> pd.Series:
    """Session at whose CLOSE an 8-K is public: acceptance before 16:00 ET on a session -> that
    session, else the next session. Equals the default day 0 (``intraday="acceptance_session"``);
    equals the legacy ``intraday="next_session"`` day 0 except for intraday filings (day 0 - 1)."""
    return day0_from_acceptance(acceptance_et, sessions, cutoff_minutes=CLOSE_MINUTES_ET,
                                intraday="next_session")


def session_after(dates: pd.Series | pd.DatetimeIndex, sessions: pd.DatetimeIndex) -> np.ndarray:
    """Row of the first session with date strictly after ``dates`` (calendar day); -1 if none."""
    d = _naive_days(dates)
    idx = pd.DatetimeIndex(sessions).searchsorted(d, side="right")
    return np.where(idx < len(sessions), idx, -1)


def nearest_session_row(dates: pd.Series | pd.DatetimeIndex,
                        sessions: pd.DatetimeIndex) -> np.ndarray:
    """Row of the session nearest to each date (ties -> the earlier session)."""
    d = _naive_days(dates).values.astype("datetime64[ns]").astype(np.int64)
    s = pd.DatetimeIndex(sessions).values.astype("datetime64[ns]").astype(np.int64)
    hi = np.clip(np.searchsorted(s, d, side="left"), 0, len(s) - 1)
    lo = np.clip(hi - 1, 0, len(s) - 1)
    pick_lo = np.abs(d - s[lo]) <= np.abs(s[hi] - d)
    return np.where(pick_lo, lo, hi)


def expected_day0_rows(day0: pd.Series, sessions: pd.DatetimeIndex,
                       lag_days: int = 364) -> np.ndarray:
    """PIT expected next-year announcement row: actual day 0 + ``lag_days`` calendar days,
    rounded to the nearest session (ties earlier). Known at the actual day 0."""
    return nearest_session_row(_naive_days(day0) + pd.Timedelta(days=lag_days), sessions)


def next_event_exit_cap(day0_row: np.ndarray, next_day0_row: np.ndarray | None = None,
                        sessions: pd.DatetimeIndex | None = None, *,
                        mode: str = "expected_next", lag_days: int = 91,
                        buffer: int = 2) -> np.ndarray:
    """Last holding row allowed by the "exit before the next announcement" rule.

    Callers hold through ``min(own exit, cap)``. ``NO_CAP`` means no cap.

    * ``"expected_next"`` (default, PIT): the session nearest to day 0 + ``lag_days``
      calendar days, minus ``buffer`` sessions. Known at day 0. No cap when that date lies
      past the end of ``sessions`` or ``day0_row < 0``. Needs ``sessions``.
    * ``"hold_through"``: no cap.
    * ``"next_event_lookahead"`` (legacy, studies 11 B and 12 early leg): the row before the
      firm's ACTUAL next day 0 (``next_day0_row - 1``; no cap when ``next_day0_row < 0``).
      The free data does not record when that date became known, so this is look-ahead
      at formation (red team 2026-10-10, F5: 31.8 % of study-11 B events cut by it).
    """
    if mode not in EXIT_MODES:
        raise ValueError(f"mode={mode!r}: one of {EXIT_MODES}")
    d = np.asarray(day0_row, dtype=np.int64)
    if mode == "hold_through":
        return np.full(len(d), NO_CAP, dtype=np.int64)
    if mode == "next_event_lookahead":
        if next_day0_row is None:
            raise ValueError("mode='next_event_lookahead' needs next_day0_row")
        nx = np.asarray(next_day0_row, dtype=np.int64)
        return np.where(nx >= 0, nx - 1, NO_CAP).astype(np.int64)
    if sessions is None:
        raise ValueError("mode='expected_next' needs sessions (calendar-day offset)")
    sess = pd.DatetimeIndex(sessions)
    ok = (d >= 0) & (d < len(sess))
    out = np.full(len(d), NO_CAP, dtype=np.int64)
    if ok.any():
        target = sess[d[ok]] + pd.Timedelta(days=lag_days)
        rows = nearest_session_row(target, sess) - buffer
        inside = np.asarray(target <= sess[-1])
        out[np.flatnonzero(ok)[inside]] = rows[inside]
    return out


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------
def trailing_deciles(values: np.ndarray, dates: pd.Series | pd.DatetimeIndex, *,
                     window_days: int = 365, min_obs: int = 100, n_bins: int = 10) -> np.ndarray:
    """Bin each event by the breakpoints of the TRAILING cross-section.

    Breakpoints for an event formed on day ``t`` are the ``1/n .. (n-1)/n`` quantiles of
    the values of all events formed in ``[t - window_days, t)`` (strictly earlier days, so
    same-day events never see each other). Returns bins ``1..n_bins``; ``0`` when fewer
    than ``min_obs`` trailing values exist or the value is NaN.
    """
    v = np.asarray(values, dtype=float)
    d = _naive_days(dates).values.astype("datetime64[ns]")
    order = np.argsort(d, kind="stable")
    ds, vs = d[order], v[order]
    out_sorted = np.zeros(len(v), dtype=np.int64)
    qs = np.arange(1, n_bins) / n_bins
    uniq, first = np.unique(ds, return_index=True)
    win = np.timedelta64(window_days, "D")
    lo_idx = np.searchsorted(ds, uniq - win, side="left")
    for k, (day_start, lo) in enumerate(zip(first, lo_idx, strict=True)):
        day_end = first[k + 1] if k + 1 < len(first) else len(ds)
        trail = vs[lo:day_start]
        trail = trail[np.isfinite(trail)]
        if len(trail) < min_obs:
            continue
        bps = np.quantile(trail, qs)
        cur = vs[day_start:day_end]
        b = np.searchsorted(bps, cur, side="left") + 1
        b[~np.isfinite(cur)] = 0
        out_sorted[day_start:day_end] = b
    out = np.zeros(len(v), dtype=np.int64)
    out[order] = out_sorted
    return out


def shuffle_within_groups(values: np.ndarray, groups: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    """Permute ``values`` within each group label (e.g. calendar month)."""
    v = np.asarray(values).copy()
    g = np.asarray(groups)
    order = np.argsort(g, kind="stable")
    gs = g[order]
    cuts = np.flatnonzero(gs[1:] != gs[:-1]) + 1
    for block in np.split(order, cuts):
        if len(block) > 1:
            v[block] = v[rng.permutation(block)]
    return v


# ---------------------------------------------------------------------------
# Calendar-time portfolios
# ---------------------------------------------------------------------------
def interval_mask(n_rows: int, n_cols: int, col: np.ndarray, first_row: np.ndarray,
                  last_row: np.ndarray) -> np.ndarray:
    """Boolean (rows x cols) mask, True on holding rows ``first_row..last_row`` (inclusive,
    clipped to the matrix) of column ``col``. Overlapping intervals are OR-ed."""
    col = np.asarray(col, dtype=np.int64)
    a = np.clip(np.asarray(first_row, dtype=np.int64), 0, n_rows)
    b = np.clip(np.asarray(last_row, dtype=np.int64) + 1, 0, n_rows)
    keep = (b > a) & (col >= 0)
    acc = np.zeros((n_rows + 1, n_cols), dtype=np.int32)
    np.add.at(acc, (a[keep], col[keep]), 1)
    np.add.at(acc, (b[keep], col[keep]), -1)
    return np.cumsum(acc[:-1], axis=0) > 0


@dataclass
class SpreadResult:
    """Daily long-minus-short series. ``net`` / ``to_*`` charge the equal-weight rebalance
    (every incumbent re-sized each session). ``net_entry_exit`` / ``to_*_entry_exit`` charge
    only entries (new weight) and exits (drifted weight): the minimal-trading cost of the
    same gross exposure (red team 2026-10-10, F3). Report both."""
    gross: np.ndarray
    net: np.ndarray
    long_ret: np.ndarray
    short_ret: np.ndarray
    to_long: np.ndarray
    to_short: np.ndarray
    n_long: np.ndarray
    n_short: np.ndarray
    to_long_entry_exit: np.ndarray | None = None
    to_short_entry_exit: np.ndarray | None = None
    net_entry_exit: np.ndarray | None = None


@dataclass
class LegTurnover:
    """Per-row turnover of an EW leg: ``total`` = ``entries + exits + incumbents``."""
    entries: np.ndarray
    exits: np.ndarray
    incumbents: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.entries + self.exits + self.incumbents

    @property
    def entry_exit(self) -> np.ndarray:
        return self.entries + self.exits


def _leg_full(mask: np.ndarray, rets: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, LegTurnover]:
    n = mask.sum(axis=1)
    w = np.where(mask, 1.0, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = w / np.where(n > 0, n, 1)[:, None]
    ret = (w * rets).sum(axis=1)
    # drift-adjusted previous weights
    drift = w[:-1] * (1.0 + rets[:-1])
    denom = 1.0 + ret[:-1]
    drift = drift / np.where(denom != 0, denom, 1.0)[:, None]
    to = np.empty(len(ret))
    to[0] = np.abs(w[0]).sum()
    to[1:] = np.abs(w[1:] - drift).sum(axis=1)
    # decomposition: entries (not held at t-1), exits (held at t-1, not at t), incumbents
    held_prev = mask[:-1]
    held_now = mask[1:]
    ent = np.empty(len(ret))
    ext = np.zeros(len(ret))
    ent[0] = w[0].sum()
    ent[1:] = np.where(held_now & ~held_prev, w[1:], 0.0).sum(axis=1)
    ext[1:] = np.where(held_prev & ~held_now, drift, 0.0).sum(axis=1)
    inc = np.zeros(len(ret))
    inc[1:] = np.where(held_prev & held_now, np.abs(w[1:] - drift), 0.0).sum(axis=1)
    return ret, to, n, LegTurnover(ent, ext, inc)


def _leg(mask: np.ndarray, rets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ret, to, n, _ = _leg_full(mask, rets)
    return ret, to, n


def leg_turnover(mask: np.ndarray, rets: np.ndarray) -> LegTurnover:
    """Turnover of the EW leg ``mask`` split into entries, exits and incumbent re-weighting
    (the drift rebalance equal weighting forces when names enter, leave or drift)."""
    return _leg_full(mask, rets)[3]


def calendar_time_spread(long_mask: np.ndarray, short_mask: np.ndarray, rets: np.ndarray,
                         cost_per_side: float) -> SpreadResult:
    """Equal-weight long-minus-short daily returns from holding masks.

    ``rets`` must be finite (missing returns filled with 0 by the caller: a held position
    with no bar that day earns 0 and the gap return lands on its next bar). ``net`` charges
    the EW rebalance turnover; ``net_entry_exit`` the entry/exit-only turnover."""
    rl, tl, nl, dl = _leg_full(long_mask, rets)
    rs, ts_, ns, ds = _leg_full(short_mask, rets)
    gross = rl - rs
    net = gross - cost_per_side * (tl + ts_)
    tle, tse = dl.entry_exit, ds.entry_exit
    net_ee = gross - cost_per_side * (tle + tse)
    return SpreadResult(gross, net, rl, rs, tl, ts_, nl, ns, tle, tse, net_ee)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def sharpe(x: np.ndarray, ann: float = ANN) -> float:
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(ann)) if sd > 0 else float("nan")


def ann_return(x: np.ndarray, ann: float = ANN) -> float:
    """Arithmetic annualised mean (mean daily x ``ann``)."""
    return float(np.mean(x) * ann)


def circular_block_bootstrap_means(x: np.ndarray, *, block: int = 63, n_draws: int = 1000,
                                   rng: np.random.Generator) -> np.ndarray:
    """Means of circular block-bootstrap resamples (same length as ``x``)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    nb = int(np.ceil(n / block))
    xx = np.concatenate([x, x[:block]])
    cs = np.concatenate([[0.0], np.cumsum(xx)])
    starts = rng.integers(0, n, size=(n_draws, nb))
    sums = cs[starts + block] - cs[starts]
    # trim the last block so every resample has exactly n observations
    last = n - (nb - 1) * block
    sums[:, -1] = cs[starts[:, -1] + last] - cs[starts[:, -1]]
    return sums.sum(axis=1) / n


def centred_block_bootstrap_p(x: np.ndarray, *, block: int = 63, n_draws: int = 1000,
                              rng: np.random.Generator) -> float:
    """p-value for H0: mean <= 0. Resample the CENTRED series (x - mean x) so the
    resampling distribution is the null; p = share of null means >= the realised mean."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    boot = circular_block_bootstrap_means(x - m, block=block, n_draws=n_draws, rng=rng)
    return float((np.sum(boot >= m) + 1) / (n_draws + 1))


def null_p(null_stats: Sequence[float], realised: float) -> float:
    """Share of null statistics >= the realised statistic (with the +1 correction)."""
    s = np.asarray(null_stats, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) == 0 or not np.isfinite(realised):
        return float("nan")
    return float((np.sum(s >= realised) + 1) / (len(s) + 1))


def mde_from_null(null_stats: Sequence[float], *, alpha: float = 0.05,
                  power: float = 0.80) -> float:
    """Minimum detectable effect, location-shift model on the null distribution.

    A one-sided test rejecting at ``stat >= q_{1-alpha}`` has power ``power`` against an
    alternative distributed as the null shifted by ``delta = q_{1-alpha} - q_{1-power}``;
    the MDE is that alternative's median, ``q_{0.5} + delta``.
    """
    s = np.asarray(null_stats, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 10:
        return float("nan")
    c = np.quantile(s, 1 - alpha)
    return float(np.median(s) + c - np.quantile(s, 1 - power))


def ols_alpha_beta(y: np.ndarray, x: np.ndarray, nw_lags: int = 5) -> dict[str, float]:
    """y = a + b x + e, Newey-West (Bartlett) standard errors."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    xm = np.column_stack([np.ones_like(x), x])
    inv = np.linalg.inv(xm.T @ xm)
    b = inv @ xm.T @ y
    e = y - xm @ b
    xe = xm * e[:, None]
    s = xe.T @ xe
    for lag in range(1, nw_lags + 1):
        g = xe[lag:].T @ xe[:-lag]
        s += (1 - lag / (nw_lags + 1)) * (g + g.T)
    se = np.sqrt(np.diag(inv @ s @ inv))
    return {"alpha_daily": float(b[0]), "beta": float(b[1]), "alpha_t": float(b[0] / se[0]),
            "beta_t": float(b[1] / se[1]), "alpha_ann": float(b[0] * ANN)}


def per_year(x: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar-year table: compounded return, arithmetic annualised mean, Sharpe, sessions."""
    s = pd.Series(np.asarray(x, dtype=float), index=pd.DatetimeIndex(dates))
    g = s.groupby(s.index.year)
    return pd.DataFrame({
        "ret_compounded": g.apply(lambda v: float(np.prod(1 + v) - 1)),
        "ann_mean": g.mean() * ANN,
        "sharpe": g.apply(lambda v: sharpe(v.to_numpy())),
        "sessions": g.size(),
    })


def years_gate(yearly: pd.DataFrame | Sequence[dict], *, rule: str = "ex_best_year_positive",
               min_sessions: int = 63) -> dict:
    """Yearly-consistency criterion on calendar-year compounded net returns.

    ``yearly`` is a :func:`per_year` table or its records (``ret_compounded``,
    ``sessions``); years with fewer than ``min_sessions`` sessions are not counted.

    * ``"ex_best_year_positive"`` (default): PASS when the mean yearly return with the best
      year removed is > 0 (the result is not carried by one year). Needs >= 2 counted years.
    * ``"share_ge_075"`` (legacy, studies 11 and 12): PASS when >= 75 % of counted years
      are positive.

    Why the default changed (red team 2026-10-10, F4, r07, iid-normal simulation):
    ``ceil(0.75 n)`` positive years is a much harsher bar at 14 years (11/14) than at 8
    (6/8). P(pass) at SR 0.3 / 0.5 / 0.75 / 1.0 is 0.13 / 0.28 / 0.55 / 0.79 with 14
    years against 0.30 / 0.48 / 0.70 / 0.85 with 8 years. For a strategy at the N2 MDE
    that N2 detects, the gate still fails 49 % of the time at 14 years against 12 % at 8.
    The 75 % rule therefore hides a requirement of SR ~ 1 in long cells. The positive-years
    share is reported as a diagnostic under both rules.

    Operating characteristics of the default (same iid simulation, vol 10 %, 5,000 paths,
    2026-10-10): P(pass) at SR 0 / 0.3 / 0.5 / 1.0 is 0.28 / 0.58 / 0.78 / 0.98 with 8
    years and 0.30 / 0.72 / 0.90 / 1.00 with 14. Given that the N2-like mean test passes it
    passes in 100 % of paths at both lengths. Under iid returns it is therefore implied by
    N2. What it adds is a check that one year does not carry the result. It is not a
    power-neutral replacement for N2 and must not be used alone.
    """
    if rule not in YEARS_RULES:
        raise ValueError(f"rule={rule!r}: one of {YEARS_RULES}")
    t = yearly if isinstance(yearly, pd.DataFrame) else pd.DataFrame(list(yearly))
    if len(t) == 0:
        r = np.array([], dtype=float)
    else:
        r = t.loc[t["sessions"] >= min_sessions, "ret_compounded"].to_numpy(dtype=float)
    share = float((r > 0).mean()) if len(r) else float("nan")
    ex_best = float(np.sort(r)[:-1].mean()) if len(r) >= 2 else float("nan")
    if rule == "share_ge_075":
        ok = bool(share >= 0.75)
        fail = "pos_years<75%"
    else:
        ok = bool(ex_best > 0)
        fail = "ex_best_year_mean<=0"
    return {"rule": rule, "pass": ok, "fail_label": None if ok else fail,
            "years_counted": int(len(r)), "years_positive": int((r > 0).sum()),
            "pos_year_share": share, "mean_ex_best_year": ex_best}


def event_study_verdict(net_sharpe: float, n1_p: float, n2_p: float,
                        yearly: pd.DataFrame | Sequence[dict], *,
                        years_rule: str = "ex_best_year_positive", alpha: float = 0.05,
                        min_sessions: int = 63) -> dict:
    """Pre-declared event-study verdict: PASS only if net Sharpe > 0, N1 p < ``alpha``,
    N2 p < ``alpha`` and the :func:`years_gate` passes under ``years_rule``.

    ``fails`` lists the failing criteria in that order with the labels studies 11/12 used
    (``net_sharpe<=0``, ``N1_p>=0.05``, ``N2_p>=0.05``, then ``pos_years<75%`` or
    ``ex_best_year_mean<=0``). Studies 11 and 12 ran with ``years_rule="share_ge_075"``.
    """
    fails = []
    if not net_sharpe > 0:
        fails.append("net_sharpe<=0")
    if not n1_p < alpha:
        fails.append(f"N1_p>={alpha:g}")
    if not n2_p < alpha:
        fails.append(f"N2_p>={alpha:g}")
    yg = years_gate(yearly, rule=years_rule, min_sessions=min_sessions)
    if not yg["pass"]:
        fails.append(yg["fail_label"])
    return {"fails": fails, "verdict": "PASS" if not fails else "REJECT", "years": yg}


# ---------------------------------------------------------------------------
# Decile-sorted event portfolios (PEAD-style) and the name-shuffle null
# ---------------------------------------------------------------------------
def decile_masks(col: np.ndarray, first_row: np.ndarray, last_row: np.ndarray, bins: np.ndarray,
                 n_rows: int, n_cols: int, *, top: int = 10, bottom: int = 1
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Long mask = events in bin ``top``; short mask = events in bin ``bottom``."""
    lo = bins == top
    sh = bins == bottom
    return (interval_mask(n_rows, n_cols, col[lo], first_row[lo], last_row[lo]),
            interval_mask(n_rows, n_cols, col[sh], first_row[sh], last_row[sh]))


def decile_spread(values: np.ndarray, form_dates: pd.Series | pd.DatetimeIndex, col: np.ndarray,
                  first_row: np.ndarray, last_row: np.ndarray, rets: np.ndarray,
                  cost_per_side: float,
                  *, window_days: int = 365, min_obs: int = 100) -> tuple[SpreadResult, np.ndarray]:
    """Trailing-decile sort of ``values`` -> long top / short bottom calendar-time spread."""
    bins = trailing_deciles(values, form_dates, window_days=window_days, min_obs=min_obs)
    lm, sm = decile_masks(col, first_row, last_row, bins, rets.shape[0], rets.shape[1])
    return calendar_time_spread(lm, sm, rets, cost_per_side), bins


def month_groups(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Integer calendar-month label (year*12 + month) for null shuffles."""
    d = _naive_days(dates)
    return (d.year * 12 + d.month).to_numpy()


def eligibility_at(row: np.ndarray, col: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """``universe[row, col]`` per record (False when ``col < 0`` or ``row`` is outside the
    matrix): e.g. "firm in the cell universe at its expected row E"."""
    r = np.asarray(row, dtype=np.int64)
    c = np.asarray(col, dtype=np.int64)
    ok = (c >= 0) & (r >= 0) & (r < universe.shape[0]) & (c < universe.shape[1])
    out = np.zeros(len(r), dtype=bool)
    out[ok] = universe[r[ok], c[ok]]
    return out


def stratified_groups(groups: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Integer label of each (group, stratum) pair, for :func:`shuffle_within_groups`."""
    g = np.asarray(groups)
    s = np.asarray(strata)
    _, lab = np.unique(np.column_stack([g.astype(np.int64), s.astype(np.int64)]), axis=0,
                       return_inverse=True)
    return lab.reshape(-1)


# ---------------------------------------------------------------------------
# Announcement delay (study 12): expected-date labels, offset shuffle, beta-hedged leg
# ---------------------------------------------------------------------------
ONTIME, LATE, EARLY = 0, 1, -1


@dataclass
class AnnouncementMatch:
    """Per expected-date record: the matched actual announcement (rows; -1 = none)."""
    day0: np.ndarray
    known: np.ndarray
    next_day0: np.ndarray


def match_announcements(rec_key: np.ndarray, rec_e_row: np.ndarray, ev_key: np.ndarray,
                        ev_day0_row: np.ndarray, ev_known_row: np.ndarray, *,
                        lookback: int = 30) -> AnnouncementMatch:
    """Match each record (firm ``rec_key``, expected row ``rec_e_row``) to the firm's first
    announcement with ``day0 >= E - lookback``; ``next_day0`` = that firm's first later day 0.

    Keys are non-negative integers (e.g. CIK); rows are session indices (>= 0)."""
    rk = np.asarray(rec_key, dtype=np.int64)
    re_ = np.asarray(rec_e_row, dtype=np.int64)
    ek = np.asarray(ev_key, dtype=np.int64)
    ed = np.asarray(ev_day0_row, dtype=np.int64)
    en = np.asarray(ev_known_row, dtype=np.int64)
    ok = (ed >= 0) & (ek >= 0)
    ek, ed, en = ek[ok], ed[ok], en[ok]
    order = np.lexsort((ed, ek))
    ek, ed, en = ek[order], ed[order], en[order]
    big = int(max(ed.max(initial=0), re_.max(initial=0))) + lookback + 10
    comp = ek * big + (ed + lookback)
    q = rk * big + np.maximum(re_, 0)
    pos = np.searchsorted(comp, q, side="left")
    have = pos < len(comp)
    have[have] &= ek[pos[have]] == rk[have]
    day0 = np.full(len(rk), -1, dtype=np.int64)
    known = np.full(len(rk), -1, dtype=np.int64)
    day0[have] = ed[pos[have]]
    known[have] = en[pos[have]]
    # first event of the same firm with a strictly later day 0
    npos = np.searchsorted(comp, rk * big + day0 + lookback + 1, side="left")
    hn = have & (npos < len(comp))
    hn[hn] &= ek[npos[hn]] == rk[hn]
    nxt = np.full(len(rk), -1, dtype=np.int64)
    nxt[hn] = ed[npos[hn]]
    return AnnouncementMatch(day0, known, nxt)


@dataclass
class DelayTrades:
    label: np.ndarray       # LATE / EARLY / ONTIME
    entry_row: np.ndarray   # close of entry (-1 when not traded)
    last_row: np.ndarray    # last holding row (inclusive); holding rows entry+1 .. last


def delay_labels(e_row: np.ndarray, m: AnnouncementMatch, n_rows: int, *, late_after: int = 5,
                 early_by: int = 5, hold: int = 40, early_exit: str = "expected_next",
                 sessions: pd.DatetimeIndex | None = None) -> DelayTrades:
    """Late / early / on-time labels and trade intervals.

    Late: the matched announcement is not known by the close of ``E + late_after`` (or
    none). Enter at that close; hold through day +1 or ``hold`` sessions, whichever first.
    Early: matched day 0 <= ``E - early_by``. Enter at the close of day +1; hold ``hold``
    sessions or until the :func:`next_event_exit_cap` (``mode=early_exit``), whichever
    first. ``early_exit="expected_next"`` (default) needs ``sessions``;
    ``"next_event_lookahead"`` (exit the session before the ACTUAL next day 0) is the rule
    study 12 ran with; ``"hold_through"`` never cuts.
    """
    e = np.asarray(e_row, dtype=np.int64)
    late = (m.known < 0) | (m.known > e + late_after)
    early = ~late & (m.day0 >= 0) & (m.day0 <= e - early_by)
    label = np.where(late, LATE, np.where(early, EARLY, ONTIME)).astype(np.int64)
    l_entry = e + late_after
    l_last = np.where(m.day0 >= 0, np.minimum(l_entry + hold, m.day0 + 1), l_entry + hold)
    e_entry = m.day0 + 1
    cap = next_event_exit_cap(m.day0, m.next_day0, sessions, mode=early_exit)
    e_last = np.minimum(e_entry + hold, cap)
    entry = np.where(label == LATE, l_entry, np.where(label == EARLY, e_entry, -1))
    last = np.where(label == LATE, l_last, np.where(label == EARLY, e_last, -1))
    last = np.minimum(last, n_rows - 1)
    bad = (entry < 0) | (entry >= n_rows - 1) | (last <= entry)
    entry = np.where(bad, -1, entry)
    last = np.where(bad, -1, last)
    return DelayTrades(label, entry, last)


def shuffle_delay_offsets(trades: DelayTrades, e_row: np.ndarray, groups: np.ndarray,
                          rng: np.random.Generator, n_rows: int, *,
                          eligible: np.ndarray | None = None,
                          stratify: str = "eligibility") -> DelayTrades:
    """Name-shuffle null for delay books: permute (label, entry - E, last - E) within groups,
    then apply each permuted triple at the record's OWN expected row. Holding lengths and
    the number of trades per group are preserved; the identity permutation is a no-op.

    ``stratify="eligibility"`` (default) permutes within (group x ``eligible``), where
    ``eligible`` is a per-record bool such as "firm in the cell universe at row E"
    (:func:`eligibility_at`). Then the shuffled legs keep the real legs' thickness after the
    universe filter. Unstratified, real late firms fail the filter more often than average,
    so the null late leg was ~50 % thicker (red team 2026-10-10, F6, r06: D1 real 7.42
    names, null 11.39, stratified 7.50). ``stratify="none"`` is the rule study 12 ran with.
    """
    if stratify not in STRATIFY_MODES:
        raise ValueError(f"stratify={stratify!r}: one of {STRATIFY_MODES}")
    e = np.asarray(e_row, dtype=np.int64)
    if stratify == "eligibility":
        if eligible is None:
            raise ValueError("stratify='eligibility' needs eligible "
                             "(e.g. eligibility_at(E, col, universe))")
        groups = stratified_groups(groups, np.asarray(eligible, dtype=bool))
    perm = shuffle_within_groups(np.arange(len(e)), groups, rng)
    traded = trades.entry_row >= 0
    lab = trades.label[perm]
    tr = traded[perm]
    entry = np.where(tr, e + (trades.entry_row - e)[perm], -1)
    last = np.where(tr, e + (trades.last_row - e)[perm], -1)
    bad = tr & ((entry < 0) | (entry >= n_rows - 1) | (last <= entry))
    last = np.where(tr & ~bad, np.minimum(last, n_rows - 1), -1)
    entry = np.where(tr & ~bad, entry, -1)
    return DelayTrades(lab, entry, last)


def delay_leg_masks(trades: DelayTrades, col: np.ndarray, universe: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(early mask, late mask): holding rows of traded records whose firm is in ``universe``
    (rows x cols bool) at the entry close."""
    n_rows, n_cols = universe.shape
    col = np.asarray(col, dtype=np.int64)
    ok = (trades.entry_row >= 0) & (col >= 0)
    ok[ok] &= universe[trades.entry_row[ok], col[ok]]
    out = []
    for lab in (EARLY, LATE):
        k = ok & (trades.label == lab)
        out.append(interval_mask(n_rows, n_cols, col[k], trades.entry_row[k] + 1,
                                 trades.last_row[k]))
    return out[0], out[1]


def both_or_none(long_mask: np.ndarray, short_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Close both legs on rows where either leg is empty (keeps the book dollar-neutral)."""
    both = long_mask.any(axis=1) & short_mask.any(axis=1)
    return long_mask & both[:, None], short_mask & both[:, None]


def trailing_beta(y: np.ndarray, x: np.ndarray, active: np.ndarray, *, window: int = 252,
                  min_obs: int = 126) -> np.ndarray:
    """PIT OLS beta of y on x: the value at row t uses rows t-window .. t-1 where ``active``.
    NaN with fewer than ``min_obs`` active rows or zero variance."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    a = np.asarray(active, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    xa, ya = np.where(a, x, 0.0), np.where(a, y, 0.0)

    def wsum(v: np.ndarray) -> np.ndarray:
        cs = np.concatenate([[0.0], np.cumsum(v)])
        t = np.arange(len(v))
        return cs[t] - cs[np.maximum(t - window, 0)]

    n = wsum(a.astype(float))
    sx, sy, sxx, sxy = wsum(xa), wsum(ya), wsum(xa * xa), wsum(xa * ya)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = sxx - sx * sx / n
        cov = sxy - sx * sy / n
        beta = cov / var
    return np.where((n >= min_obs) & (var > 1e-14), beta, np.nan)


@dataclass
class HedgedShortResult:
    gross: np.ndarray       # -r_leg + h r_mkt
    net: np.ndarray
    raw_gross: np.ndarray   # unhedged short leg
    raw_net: np.ndarray
    leg_ret: np.ndarray
    mkt_ret: np.ndarray
    hedge: np.ndarray
    to_leg: np.ndarray
    to_hedge: np.ndarray
    n_leg: np.ndarray
    active: np.ndarray
    to_leg_entry_exit: np.ndarray | None = None
    to_hedge_entry_exit: np.ndarray | None = None
    net_entry_exit: np.ndarray | None = None


def beta_hedged_short(leg_mask: np.ndarray, mkt_mask: np.ndarray, rets: np.ndarray,
                      cost_per_side: float, *, window: int = 252, min_obs: int = 126
                      ) -> HedgedShortResult:
    """Short an EW leg, hedged LONG the EW ``mkt_mask`` basket by the leg's trailing beta.

    The hedge ratio at row t is the PIT beta of the leg's return on the basket's return over
    rows t-window .. t-1 on which the leg held names; no position (leg and hedge) on rows
    without an estimate or with an empty leg. Hedge cost = cost x (|h_t - h_{t-1}| +
    |h_t| x the basket's own turnover). ``net_entry_exit`` uses the entry/exit-only turnover
    for both the leg and the basket (the |h_t - h_{t-1}| term is unchanged)."""
    rl, tl, nl = _leg(leg_mask, rets)
    rm, tm, _, dm = _leg_full(mkt_mask, rets)
    beta = trailing_beta(rl, rm, nl > 0, window=window, min_obs=min_obs)
    active = (nl > 0) & np.isfinite(beta)
    h = np.where(active, beta, 0.0)
    rl2, tl2, nl2, dl2 = _leg_full(leg_mask & active[:, None], rets)
    dh = np.abs(np.diff(np.concatenate([[0.0], h])))
    to_h = dh + np.abs(h) * tm
    gross = -rl2 + h * rm
    net = gross - cost_per_side * (tl2 + to_h)
    tle = dl2.entry_exit
    to_he = dh + np.abs(h) * dm.entry_exit
    net_ee = gross - cost_per_side * (tle + to_he)
    return HedgedShortResult(gross, net, -rl, -rl - cost_per_side * tl, rl2, rm, h, tl2, to_h,
                             nl2, active, tle, to_he, net_ee)
