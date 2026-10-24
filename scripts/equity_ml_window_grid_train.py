#!/usr/bin/env python3
"""Study 14: training-window / timeframe grid for the equity ML cross-section -- equities TRAIN
2004-01-01 -> 2019-01-01 only.

Pre-declared design (FINAL, v2, after the design review). Ten trial
cells T1-T10 (window {expanding, rolling 5 y, rolling 3 y} x horizon {weekly, monthly,
quarterly} at annual refit, plus rolling 3 y / quarterly refit / weekly), ENS5 = M0..M4 on the
liquid half at 5 bps per side; diagnostics D0-D6. Library: ``eqxs.equity_ml_grid``
(wraps study 13's ``eqxs.equity_ml``, which is not modified).

All data come through ``eqxs.equities_xs`` loaders with ``end="2019-01-01"``. Study 14's
panels, refit cache, predictions and portfolios live under
``data_cache/equities_xs/derived/eqml_grid/``; study 13's cache (``eqml_runs/``) is never read.

    python scripts/equity_ml_window_grid_train.py --rebuild-panels --build-only
    python scripts/equity_ml_window_grid_train.py --cells all
    python scripts/equity_ml_window_grid_train.py --report-only
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from eqxs.data_zones import (  # noqa: E402
    EQUITIES_ZONES_PATH,
    assert_sealed_clean,
    safe_load_parquet,
)
from eqxs import equities_xs as xs  # noqa: E402
from eqxs import equity_events as ee  # noqa: E402
from eqxs import equity_ml as ml  # noqa: E402
from eqxs import equity_ml_grid as g  # noqa: E402

TRAIN_START = pd.Timestamp("2004-01-01")
TRAIN_END = pd.Timestamp("2019-01-01")
WARMUP_YEAR = 2002
FUND_START = pd.Timestamp("2009-01-01")
DERIVED = xs.XS_ROOT / "derived"
GRID = DERIVED / "eqml_grid"
CELLS_DIR = GRID / "cells"
PRED_DIR = GRID / "predictions"
PORT_DIR = GRID / "portfolios"
STUDY13_PANEL = DERIVED / "ml_panel_train.parquet"
STUDY13_MONTHS = DERIVED / "ml_months_train.parquet"
STUDY13_META = DERIVED / "ml_panel_train.meta.json"
STUDY13_JSON = REPO / "results" / "equity_ml_xs_train.json"
OUT_JSON = REPO / "results" / "equity_ml_window_grid_train.json"
OUT_MD = REPO / "results" / "equity_ml_window_grid_train.md"
SEED = 20261013  # study 13's SEED; refit seed = SEED + month index of the first test month
N_N1 = 500
N_N2 = 1000
COVERAGE_MIN = 0.30
SURV_NOTE = ("survivor-only price panel (yfinance: stocks listed on 2026-10-09; no delisted firms, "
             "no delisting returns)")
SURV_DIRECTION = ("survivorship direction: long-leg returns biased UP, short-leg returns biased UP "
                  "(dead firms that would sit in the short decile are missing), so the long/short "
                  "spread is biased DOWN on the short side and UP on the long side; the bias scales "
                  "with block age (2009 ~50 % of listed firms missing, 2018 ~30 %)")
H_NAME = g.HORIZON_NAME
SUMMARY_AFTER_RUN = (
    "Written after the run (2026-10-11); numbers from the tables below. **All 10 trials REJECT; study REJECT.** "
    "Headline T5 (expanding, annual, monthly): net −1.66 %/yr (SR −0.09), gross −0.38 %/yr, N1-gross p 0.50, "
    "N2 p 0.62. Best trial T8 (rolling 5 y, quarterly): net +4.34 %/yr (SR +0.34), N2 p 0.14, DSR 0.59 against "
    "SR0 +0.27 (N = 10). No cell clears N2 or the gross N1 gate.",
    "**Horizon matters more than window, and in opposite directions for signal and cost.** Weekly cells carry the "
    "most information (annual-refit T1-T3 IC +0.019 to +0.020, t 3.2-3.4; T10 +0.013, t 1.9; gross +2.7 to +5.4 %/yr) but turn over 78-100× a year, so "
    "break-even is 2.8-6.9 bps per side against the 5 bps charged; net is −2.2 to +1.5 %/yr. Monthly cells have "
    "IC t ≤ 1.4 and sit near zero gross. Quarterly expanding / rolling-5 y cells are net positive (+3.3 / +4.3 %/yr, "
    "break-even 38-48 bps), but they have 39 periods and N2 p 0.14-0.22 against an N2 MDE of 9.5-10 %/yr.",
    "**No monotone window effect.** Rolling 3 y beats expanding at weekly (+1.48 vs −2.22 %/yr) and monthly (+0.24 vs "
    "−1.66) and loses at quarterly (T9 −2.35 %/yr; it fits on ~8 cross-sections). The window spread of every cell is "
    "well inside one N2 MDE.",
    "**2009 dominates every weekly and monthly cell (−39 % to −52 %).** It is a momentum-crash exposure, not a "
    "window effect: 2009 MOM loading +0.53 to +1.00 (t 6-12), MOM contribution −32 % to −53 %, ex-ante book beta "
    "−0.61 to −0.65 (T9 −0.80). The quarterly expanding / rolling-5 y books were short MOM in 2009 (−0.21) and made +9.7 %. D6 "
    "(2014-2018, same predictions): every trial is net positive (SR +0.22 to +1.11), and T5 alone passes the per-cell "
    "rule on that slice (+9.6 %/yr, N2 p 0.003). D6 is a diagnostic: it cannot PASS anything, and the slice was "
    "pre-declared for the XBRL confound, not chosen on returns. Anything built on it is a new hypothesis that needs "
    "its own pre-declaration and the WFA stage.",
    "Bridges: D0 (same v3 rules on the study-13 panel) is +1.20 %/yr vs T5 −1.66, so the study-14 panel changes "
    "(acceptance-session day 0, session lookbacks, R8 flag) cost about 2.9 pp/yr here, inside the noise (both "
    "REJECT). D3 (ENS3 = M2-M4) +0.06 %/yr: the two ridges cost about 1.7 pp/yr. D1 (quarterly refit) −1.55 %/yr ≈ T5. "
    "D2 (3-month embargo) −0.57 %/yr. D7 (beta-hedged T5) −0.45 %/yr, realised β −0.32 → −0.08. D5 (survivorship family "
    "removed) hurts the weekly and monthly ensembles (−3.80 / −2.48 %/yr) and helps the quarterly one (+4.11), the "
    "opposite of study 13's ablation in the short horizons. ENS5 − M0 is positive in 9 of 10 trials, and p < 0.05 in "
    "none (lowest 0.07).",
)
DISCLOSURES = (
    "Discarded fits (bug fixes, not result-driven). A first full run under the v2 machinery completed T5, D3, D0, "
    "T4, T6, T7, T8 and T9 and part of D1 before the author's v3 addendum arrived (study-13 red team, "
    "red-team items R1-R15). That run was stopped; its cell JSONs, "
    "predictions and refit cache were moved unread to data_cache/equities_xs/derived/eqml_grid/discarded_v2_unread/ "
    "(the run log showed only validation ICs and timings; no OOS metric was printed or opened). Every cell in this "
    "report was refitted from scratch under v3 on rebuilt panels (cache version eqgrid-v3).",
    "v3 R3: validation IC (ridge alpha choice, NN3 grid and early stopping, reported booster IC) is the average-rank "
    "Spearman per date (a constant prediction scores 0); decile membership ties are broken by a seeded uniform draw "
    "(seed 20261014), not by row (alphabetical) order.",
    "v3 R2/R4: M2 and M4 use FIXED settings, no grid and no early stopping: max_depth 3, learning_rate 0.05, 300 "
    "rounds, subsample 0.8, colsample_bytree 0.8, min_child_weight 100 (M2) and 1 (M4, hessian-scaled for "
    "rank:pairwise); fitted on the fit slice only (the v2 GKX protocol: no refit on fit + validation). The study-13 "
    "red team observed that fixed 300 rounds lifted study 13 by about 1 pp/yr (still REJECT); this setting is "
    "pre-declared with that knowledge. M3 (NN3) keeps study 13's grid and early stopping. Validation slices are "
    "v2's (12 months for annual refits, 6 for the quarterly refits, 24 / 24 / 12 at the quarterly horizon); the "
    "addendum did not change the splitter. Every fit asserts more than one distinct prediction per date on the "
    "validation and test rows; a failing cell would be reported NOT RUN.",
    "v3 R1: the N1 gate is the GROSS-statistic within-date permutation p; N1 on the net statistic is reported beside "
    "it as a persistence diagnostic. The 2×-cost rule uses the same gross N1 p (cost-invariant), the 2×-cost N2 and "
    "the 2×-cost years rule.",
    "v3 R5: ex-ante book beta = long-decile mean minus short-decile mean of each name's trailing 252-session OLS beta of "
    "(r − RF) on Mkt-RF (≥ 200 pairs, PIT at the formation close), reported per year for every trial. D7 (added by "
    "the addendum) = T5 index-hedged: r − h·MktRF − cost·|Δh| with the hedge turnover charged at the book's 5 bps; its "
    "N1 re-hedges every permuted book with its own beta. No beta-residual target.",
    "v3 R6: per-year daily FF3 + MOM attribution of each trial's gross L/S spread (daily equal-weight buy-and-hold leg "
    "returns from the same TRAIN price matrix; French daily factors loaded with end=2019-01-01).",
    "v3 R8: rows with implied daily turnover (63-day median dollar volume / market cap) > 10 % have every fundamental "
    "feature set missing (split-distorted or unit-error XBRL shares); counts per panel are in the Panels section. The "
    "30 % coverage rule was evaluated before the flag (the flag removes 0.74 % of rows with a market cap and cannot move "
    "any tag across 30 %). The study-13 panel used by D0 is not flagged. v3 R9: predictions stored as float64.",
    "D7 was added to the diagnostic list and the run order (after D3, it reuses the T5 fits) by the v3 addendum; it is "
    "never a trial. The cell RNG seeds are indexed on the v3 cell list.",
    "Cache location: the refit cache is data_cache/equities_xs/derived/eqml_grid/eqml14_runs/ (v2's eqml14_runs/ "
    "name, inside the coordinator's eqml_grid/ directory); study 13's eqml_runs/ is never read.",
    "XGBoost runs on 4 threads and NN3 on 4 worker processes (machine sharing); torch and XGBoost never share a "
    "process (study 13's loky worker split). Panel build peak RSS 2.50 GB.",
    "The equity_ml / weekly equality: the weekly builder reproduces the monthly panel's features exactly on the shared "
    "dates (match share recorded below). The EW market return used by ivol_63 / beta_252 is defined on month-end ALL "
    "universes at every horizon (study-13 definition), so it does not depend on the formation grid.",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


def assert_pre_2019(df: pd.DataFrame, what: str) -> None:
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            tz = getattr(s.dt, "tz", None)
            bound = TRAIN_END.tz_localize("UTC") if tz is not None else TRAIN_END
            mx = s.max()
            assert pd.isna(mx) or mx < bound, f"{what}: column {c} has a date >= 2019-01-01 ({mx})"


# ---------------------------------------------------------------------------
# Data loading (study 13's loaders, copied; day 0 = acceptance session)
# ---------------------------------------------------------------------------
def load_prices() -> tuple[ml.PriceMatrices, dict[str, str]]:
    uc = xs.load_static("universe_current")
    common = uc[uc["security_type"] == "common"]
    yf_of = dict(zip(common["symbol"], common["yf_symbol"], strict=False))
    keep = set(common["yf_symbol"])
    cat = pd.CategoricalDtype(sorted(keep))
    parts = []
    for y in range(WARMUP_YEAR, TRAIN_END.year):
        px = xs.load_yf_daily(columns=["symbol", "date", "close", "adj_close", "volume", "splits"],
                              start=f"{y}-01-01", end=f"{y + 1}-01-01")
        px = px[px["symbol"].isin(keep)]
        px = px.assign(date=pd.to_datetime(px["date"]), symbol=px["symbol"].astype(cat))
        for c in ("close", "adj_close", "splits"):
            px[c] = px[c].astype(np.float32)
        px["volume"] = px["volume"].astype(np.float64)
        parts.append(px)
    px = pd.concat(parts, ignore_index=True)
    del parts
    assert px["date"].max() < TRAIN_END
    codes = px["symbol"].cat.codes.to_numpy()
    present = np.unique(codes)
    symbols = np.asarray(cat.categories)[present]
    sessions = pd.DatetimeIndex(np.unique(px["date"].to_numpy()))
    row = sessions.get_indexer(px["date"])
    col = np.searchsorted(present, codes)
    pm = ml.build_price_matrices(sessions, symbols, row, col, px["close"].to_numpy(np.float64),
                                 px["adj_close"].to_numpy(np.float64),
                                 px["volume"].to_numpy(np.float64), px["splits"].to_numpy(np.float64))
    log(f"price matrices: {pm.n_rows} sessions x {pm.n_cols} symbols, {len(px):,} bars, "
        f"{pm.n_guarded} guarded returns")
    return pm, yf_of


def load_ff(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    ff = xs.load_ff_daily(end=TRAIN_END)
    idx = pd.DatetimeIndex(ff.index)
    ff.index = idx.tz_localize(None) if idx.tz is not None else idx
    ff = ff.loc[ff.index < TRAIN_END].reindex(sessions)
    assert ff["Mkt-RF"].notna().all(), "FF must cover the trading calendar"
    return ff


def load_events(pm: ml.PriceMatrices, cik_map: dict[int, str], col_of: dict[str, int],
                mkt: np.ndarray) -> tuple[pd.DataFrame, dict]:
    from eqxs.earnings_calendar import clean_event_mask

    ev = xs.load_earnings_all(end=TRAIN_END)
    assert pd.to_datetime(ev["ts"], utc=True).max() < TRAIN_END.tz_localize("UTC")
    kept = ev[ev["keep"]].copy()
    kept["clean"] = clean_event_mask(kept["ambiguous"], flags="pit").to_numpy(bool)
    kept["symbol"] = kept["cik"].astype(int).map(cik_map)
    kept["col"] = kept["symbol"].map(col_of).fillna(-1).astype(int)
    kept["day0"] = ee.day0_from_acceptance(kept["acceptance_et"], pm.sessions,
                                           intraday="acceptance_session")
    kept["day0_next"] = ee.day0_from_acceptance(kept["acceptance_et"], pm.sessions,
                                                intraday="next_session")
    n_changed = int((kept["day0"] != kept["day0_next"]).sum())
    kept = kept[kept["day0"].notna() & (kept["col"] >= 0)].copy()
    kept["day0_row"] = pm.sessions.get_indexer(pd.DatetimeIndex(kept["day0"]))
    kept = kept[kept["day0_row"] >= 0]
    kept = kept.sort_values(["col", "day0"]).drop_duplicates(["col", "day0"])
    kept["signal"] = ml.announcement_signal(kept["day0_row"].to_numpy(), kept["col"].to_numpy(),
                                            pm.R, mkt)
    diag = {"raw_rows": int(len(ev)), "kept_with_price_symbol": int(len(kept)),
            "day0_intraday_mode": "acceptance_session",
            "day0_rule": "equity_events.day0_from_acceptance(intraday='acceptance_session')",
            "clean_mask": "earnings_calendar.clean_event_mask(flags='pit')",
            "kept_events_day0_differs_from_next_session": n_changed,
            "clean_share": float(kept["clean"].mean())}
    return kept.loc[:, ["cik", "col", "day0", "day0_row", "clean", "signal"]].reset_index(drop=True), diag


def load_fundamentals(ciks: set[int]) -> tuple[dict[str, pd.DataFrame], dict]:
    parts = []
    n_raw = 0
    for y in range(2008, TRAIN_END.year):
        for qn in range(4):
            s = pd.Timestamp(year=y, month=3 * qn + 1, day=1)
            e = s + pd.DateOffset(months=3)
            num = xs.load_sec_num(tags=list(ml.ALL_SEC_TAGS), ciks=sorted(ciks), start=s, end=e)
            n_raw += len(num)
            if len(num):
                f = ml.prepare_facts(num)
                parts.append(f[f["uom"].isin(["USD", "shares", "USD/shares"])])
            del num
    f = pd.concat(parts, ignore_index=True)
    del parts
    assert f["ts"].max() < TRAIN_END.tz_localize("UTC").value
    out = ml.group_facts(f)
    log(f"fundamentals: {n_raw:,} raw rows, {len(f):,} prepared")
    return out, {"raw_rows": int(n_raw), "prepared_rows": int(len(f))}


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_paths(h: str) -> dict[str, Path]:
    return {"panel": GRID / f"panel_{h}.parquet", "months": GRID / f"months_{h}.parquet",
            "meta": GRID / f"panel_{h}.meta.json"}


class Shared:
    """Raw inputs shared by the three horizon panels (loaded once)."""

    def __init__(self) -> None:
        t0 = time.time()
        self.pm, yf_of = load_prices()
        pm = self.pm
        self.col_of = {s: i for i, s in enumerate(pm.symbols)}
        self.me_all = ml.month_end_rows(pm.sessions)
        uni_me = ml.build_universe(pm, self.me_all)
        # EW universe return used by ivol / beta: month-end ALL universes for every horizon
        # (study-13 definition), so the feature does not depend on the formation grid.
        self.mkt_ew = ml.ew_market_return(pm.R, self.me_all, uni_me.all_)
        self.ff = load_ff(pm.sessions)
        self.mkt_ff = (self.ff["Mkt-RF"] + self.ff["RF"]).to_numpy()
        bars = np.isfinite(pm.adj).sum(axis=0)
        self.cmap, self.map_diag = ml.cik_symbol_map(xs.load_static("cik_ticker_map"), yf_of,
                                                     self.col_of, bars)
        self.events, self.ev_diag = load_events(pm, self.cmap, self.col_of, self.mkt_ff)
        self.sym_cik = {s: c for c, s in self.cmap.items()}
        mapped = set(int(c) for c in self.cmap)
        self.facts, self.fund_diag = load_fundamentals(mapped)
        self.sue_ev = ml.sue_events(self.facts.get("eps", pd.DataFrame(columns=["cik"])))
        self.fund_diag["sue_events"] = int(len(self.sue_ev))
        prof = xs.load_static("filer_profiles")
        self.sic_of = dict(zip(prof["cik"].astype(int), prof["sic"].astype(str), strict=False))
        self.seconds = time.time() - t0
        log(f"shared inputs loaded in {self.seconds:.0f} s (peak RSS {rss_gb():.2f} GB)")


def build_one(sh: Shared, h: str) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    t0 = time.time()
    pm = sh.pm
    fr_all = g.formation_rows(pm.sessions, h)
    d = pm.sessions[fr_all]
    fr = fr_all[(d >= TRAIN_START) & (d < TRAIN_END)]
    fdates = pm.sessions[fr]
    uni = ml.build_universe(pm, fr)
    hret, entry, exit_ = ml.holding_returns(pm.R, fr)
    pf = g.price_features_sessions(pm, fr, sh.mkt_ew)
    efe = ml.event_features(sh.events, pm.sessions, fr, pm.n_cols)
    log(f"[{h}] {len(fr)} formation dates {fdates[0].date()} .. {fdates[-1].date()}: price + event "
        f"features ({time.time() - t0:.0f} s)")
    parts = [pd.DataFrame({"k": k, "firm": np.flatnonzero(uni.all_[k])}) for k in range(len(fr))]
    panel = pd.concat(parts, ignore_index=True)
    del parts
    k, f = panel["k"].to_numpy(), panel["firm"].to_numpy()
    panel["date"] = fdates[k]
    panel["ts"] = panel["date"].dt.tz_localize("UTC")
    panel["date_id"] = k
    panel["mi"] = ml.month_index(panel["date"])
    panel["symbol"] = pm.symbols[f]
    panel["cik"] = pd.array([sh.sym_cik.get(s) for s in panel["symbol"]], dtype="Int64")
    panel["in_liq"] = uni.liq[k, f]
    ok_t = entry[k] >= 0
    ent = np.where(ok_t, entry[k], 0)
    exi = np.where(ok_t, exit_[k], 0)
    panel["entry_date"] = pd.Series(pm.sessions[ent]).where(ok_t)
    panel["exit_date"] = pd.Series(pm.sessions[exi]).where(ok_t)
    panel["target_mi"] = np.where(ok_t, ml.month_index(pm.sessions[ent]), -1)
    panel["y_raw"] = np.where(ok_t, hret[k, f], np.nan)
    for name, arr in pf.items():
        panel[f"raw_{name}"] = arr[k, f]
    for name, arr in efe.items():
        panel[f"raw_{name}"] = arr[k, f]
    del pf, efe
    panel["size_log_dv"] = panel["raw_log_dv_63"]
    have = panel["cik"].notna().to_numpy()
    sub = panel.loc[have]
    q = g.make_fund_query_sessions(pm, fr, sub["k"].to_numpy(), sub["firm"].to_numpy(),
                                   sub["cik"].to_numpy(np.int64))
    ff_feats, present = ml.fundamental_features(sh.facts, q, sue_ev=sh.sue_ev)
    pre = (sub["date"] < FUND_START).to_numpy()
    for name in ml.FUND_FEATURES:
        v = np.full(len(panel), np.nan)
        v[have] = np.where(pre, np.nan, ff_feats[name])
        panel[f"raw_{name}"] = v
    # R8 (v3): implied daily turnover = 63-day median dollar volume / market cap > 10 %/day marks
    # a split-distorted or unit-error XBRL share count -> every fundamental feature missing
    with np.errstate(invalid="ignore", over="ignore"):
        imp_turn = np.exp(panel["raw_log_dv_63"].to_numpy(np.float64) - panel["raw_log_mcap"].to_numpy(np.float64))
    r8 = np.isfinite(imp_turn) & (imp_turn > 0.10)
    r8_diag = {"rows_flagged": int(r8.sum()), "rows_with_mcap": int(np.isfinite(imp_turn).sum()),
               "share_of_rows_with_mcap": float(r8.sum() / max(np.isfinite(imp_turn).sum(), 1)),
               "top_symbols": panel.loc[r8, "symbol"].value_counts().head(15).to_dict()}
    for name in ml.FUND_FEATURES:
        v = panel[f"raw_{name}"].to_numpy(np.float64).copy()
        v[r8] = np.nan
        panel[f"raw_{name}"] = v
    # ex-ante (PIT) beta per name for book-beta reporting and the D7 hedge (R5); not a model input
    B = g.name_betas(pm.R, sh.ff["Mkt-RF"].to_numpy(), sh.ff["RF"].to_numpy(), fr)
    panel["beta_exante"] = B[k, f]
    del B
    pres = {}
    for gname, arr in present.items():
        v = np.zeros(len(panel), dtype=bool)
        v[have] = arr & ~pre
        pres[gname] = v
    del ff_feats, present, sub, q
    # coverage rule inputs (ALL firm-months 2010-01..2018-12), aligned with the unsorted panel
    m10u = (panel["date"] >= pd.Timestamp("2010-01-01")).to_numpy()
    tag_cov = {gname: float(pres[gname][m10u].mean()) for gname in ml.TAG_GROUPS}
    del pres
    div = ml.sic_division(pd.Series([sh.sic_of.get(int(c), "") if pd.notna(c) else ""
                                     for c in panel["cik"]]))
    for c in ml.SIC_COLS:
        panel[c] = (div.to_numpy() == c.split("_", 1)[1]).astype(np.float32)
    panel = panel.sort_values(["date_id", "firm"]).reset_index(drop=True)
    b = ml.date_bounds(panel["date_id"].to_numpy())
    y = panel["y_raw"].to_numpy()
    liq = panel["in_liq"].to_numpy()
    y_ex_all = np.full(len(panel), np.nan)
    y_ex_liq = np.full(len(panel), np.nan)
    y_rank = np.full(len(panel), np.nan)
    for a, bb in zip(b[:-1], b[1:], strict=True):
        yy = y[a:bb]
        if not np.isfinite(yy).any():
            continue
        y_ex_all[a:bb] = yy - np.nanmean(yy)
        y_rank[a:bb] = ml.rank_unit(yy)
        lm = liq[a:bb]
        y_ex_liq[a:bb] = np.where(lm, yy - np.nanmean(yy[lm]), np.nan)
    panel["y_ex_all"], panel["y_ex_liq"], panel["y_rank"] = y_ex_all, y_ex_liq, y_rank
    for name in ml.NUMERIC_FEATURES:
        z, miss = ml.normalize_within_date(panel[f"raw_{name}"].to_numpy(), b)
        panel[name] = z
        panel[f"miss_{name}"] = miss
    mrows = []
    rf_all = sh.ff["RF"].to_numpy()
    for kk in range(len(fr)):
        if entry[kk] < 0:
            continue
        e, x = int(entry[kk]), int(exit_[kk])
        mk_tot = float(np.prod(1 + sh.mkt_ff[e + 1:x + 1]) - 1)
        rf = float(np.prod(1 + rf_all[e + 1:x + 1]) - 1)
        mrows.append({"date_id": kk, "date": fdates[kk], "entry_date": pm.sessions[e],
                      "exit_date": pm.sessions[x],
                      "target_mi": int(ml.month_index(pd.DatetimeIndex([pm.sessions[e]]))[0]),
                      "sessions_held": x - e, "mkt": mk_tot, "rf": rf, "mktrf": mk_tot - rf})
    months = pd.DataFrame(mrows)
    months["ts"] = pd.to_datetime(months["date"]).dt.tz_localize("UTC")
    feat_cov = panel.groupby(panel["date"].dt.year)[[f"raw_{n}" for n in ml.NUMERIC_FEATURES]].agg(
        lambda s: float(s.notna().mean()))
    feat_cov.columns = list(ml.NUMERIC_FEATURES)
    meta = {"version": g.GRID_VERSION, "horizon": h, "built_utc": pd.Timestamp.now("UTC").isoformat(),
            "n_rows": int(len(panel)), "n_dates": int(len(fr)),
            "first_formation": str(fdates[0].date()), "last_formation": str(fdates[-1].date()),
            "formation_rule": {"W": f"every {g.WEEKLY_STEP}th session from {g.WEEKLY_ANCHOR} (row 0)",
                               "M": "last session of each month",
                               "Q": "last session of March, June, September, December"}[h],
            "lookbacks": "session-based: mom_12_1 T-252->T-21, mom_6_1 T-126->T-21, rev_1m T-21->T, "
                         "rev_1w T-5->T; fundamentals '12 months earlier' = last session <= T-365 d; "
                         "EW market for ivol/beta = month-end ALL universes (study-13 definition)",
            "target": "close(T+1) -> close(next formation row + 1), rank within date over ALL",
            "day0_intraday_mode": "acceptance_session", "events": sh.ev_diag,
            "n_symbols_matrix": int(pm.n_cols), "n_sessions": int(pm.n_rows),
            "mapping": sh.map_diag, "fundamentals": sh.fund_diag,
            "feature_coverage_per_year": {int(yv): {kk: float(vv) for kk, vv in r.items()}
                                          for yv, r in feat_cov.iterrows()},
            "build_seconds": round(time.time() - t0, 1)}
    meta["tag_coverage_2010_own"] = tag_cov
    meta["r8_share_flag"] = r8_diag
    return panel, months, meta, {"tag_cov": tag_cov}


def coverage_rule(tag_cov: dict) -> tuple[list, list]:
    dropped_tags = sorted(gname for gname, c in tag_cov.items() if c < COVERAGE_MIN)
    dropped_feats = sorted(f for f, tags in ml.FEATURE_TAGS.items() if set(tags) & set(dropped_tags))
    return dropped_tags, dropped_feats


def equality_check(pw: pd.DataFrame, pmo: pd.DataFrame) -> dict:
    """Weekly panel vs monthly panel on dates that are both month-ends and weekly formation dates
    (pre-declaration: any mismatch other than float rounding blocks the run)."""
    shared = np.intersect1d(pw["date"].unique(), pmo["date"].unique())
    a = pw[pw["date"].isin(shared)].set_index(["date", "firm"]).sort_index()
    b = pmo[pmo["date"].isin(shared)].set_index(["date", "firm"]).sort_index()
    same_universe = bool(a.index.equals(b.index))
    cols = ([f"raw_{n}" for n in ml.NUMERIC_FEATURES] + list(ml.NUMERIC_FEATURES)
            + [f"miss_{n}" for n in ml.NUMERIC_FEATURES] + list(ml.SIC_COLS) + ["in_liq"])
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    n_cells = 0
    n_match = 0
    per_col = {}
    for c in cols:
        x = a[c].to_numpy(np.float64)
        y = b[c].to_numpy(np.float64)
        both_nan = np.isnan(x) & np.isnan(y)
        close = np.isclose(x, y, rtol=1e-6, atol=1e-9) | both_nan
        per_col[c] = float(close.mean())
        n_cells += len(x)
        n_match += int(close.sum())
    return {"shared_dates": int(len(shared)), "rows_compared": int(len(common)),
            "same_universe": same_universe, "match_share": n_match / max(n_cells, 1),
            "worst_columns": dict(sorted(per_col.items(), key=lambda kv: kv[1])[:5])}


def build_panels() -> dict:
    t0 = time.time()
    GRID.mkdir(parents=True, exist_ok=True)
    sh = Shared()
    built = {}
    for h in ("M", "Q", "W"):
        panel, months, meta, extra = build_one(sh, h)
        built[h] = (panel, months, meta, extra)
        log(f"[{h}] panel {len(panel):,} rows, {meta['n_dates']} dates ({meta['build_seconds']} s, "
            f"peak RSS {rss_gb():.2f} GB)")
    aux = GRID / "aux_daily.npz"
    assert sh.pm.sessions.max() < TRAIN_END
    np.savez(aux, R=sh.pm.R, sessions=sh.pm.sessions.values.astype("datetime64[ns]").astype(np.int64),
             symbols=sh.pm.symbols.astype(str))
    del sh
    # coverage rule: ALL firm-months 2010-01..2018-12 of the MONTHLY panel (study-13 rule), applied
    # to every horizon so all panels share one input set
    tag_cov = built["M"][3]["tag_cov"]
    dropped_tags, dropped_feats = coverage_rule(tag_cov)
    log(f"coverage 2010+ (monthly): {json.dumps({k: round(v, 3) for k, v in tag_cov.items()})}; "
        f"dropped {dropped_tags} -> {dropped_feats}")
    eq = equality_check(built["W"][0], built["M"][0])
    log(f"weekly vs monthly equality: {json.dumps(eq)}")
    if not (eq["same_universe"] and eq["match_share"] > 1 - 1e-9):
        raise RuntimeError(f"weekly/monthly feature mismatch blocks the run: {eq}")
    numeric = [f for f in ml.NUMERIC_FEATURES if f not in dropped_feats]
    columns = ml.input_columns(numeric)
    out = {}
    for h, (panel, months, meta, _) in built.items():
        meta.update({"tag_coverage_2010_monthly": tag_cov, "dropped_tags": dropped_tags,
                     "dropped_features": dropped_feats, "coverage_rule": "monthly panel, ALL 2010-2018"})
        X = panel[columns].to_numpy(np.float32)
        y = panel["y_rank"].to_numpy(np.float64)
        meta["sha256_input_matrix"] = hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()
        meta["sha256_target"] = hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()
        meta["input_columns"] = columns
        if h == "W":
            meta["weekly_vs_monthly_equality"] = eq
        panel = panel.drop(columns=["k"])
        assert_pre_2019(panel, f"panel_{h}")
        assert_pre_2019(months, f"months_{h}")
        num_cols = [c for c in panel.columns
                    if c.startswith(("raw_", "miss_", "sic_")) or c in ml.NUMERIC_FEATURES]
        panel[num_cols] = panel[num_cols].astype(np.float32)
        for c in ("y_raw", "y_ex_all", "y_ex_liq", "y_rank", "size_log_dv", "beta_exante"):
            panel[c] = panel[c].astype(np.float32)
        # recompute hashes on the stored dtypes (what the fits read)
        X = panel[columns].to_numpy(np.float32)
        y = panel["y_rank"].to_numpy(np.float64)
        meta["sha256_input_matrix"] = hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()
        meta["sha256_target"] = hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()
        pp = panel_paths(h)
        panel.to_parquet(pp["panel"], index=False)
        months.to_parquet(pp["months"], index=False)
        pp["meta"].write_text(json.dumps(meta, indent=1, default=str))
        out[h] = meta
        log(f"[{h}] wrote {pp['panel']} sha256(X) {meta['sha256_input_matrix'][:12]}")
    log(f"panels built in {time.time() - t0:.0f} s, peak RSS {rss_gb():.2f} GB")
    return out


KEEP_COLS = ("date", "ts", "date_id", "target_mi", "entry_date", "exit_date", "firm", "in_liq", "y_raw",
             "y_ex_all", "y_ex_liq", "y_rank", "beta_exante")


def slim(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep only what the fits and portfolios read (model inputs + keys + targets)."""
    inputs = set(ml.input_columns(list(ml.NUMERIC_FEATURES)))
    return panel.loc[:, [c for c in panel.columns if c in KEEP_COLS or c in inputs]].copy()


def load_grid_panel(h: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pp = panel_paths(h)
    panel = safe_load_parquet(pp["panel"], max_end_date=TRAIN_END, caller_intent=f"eqgrid.panel_{h}",
                              zone_config=EQUITIES_ZONES_PATH)
    months = safe_load_parquet(pp["months"], max_end_date=TRAIN_END,
                               caller_intent=f"eqgrid.months_{h}", zone_config=EQUITIES_ZONES_PATH)
    assert_pre_2019(panel, f"panel_{h}")
    assert_pre_2019(months, f"months_{h}")
    return slim(panel), months, json.loads(pp["meta"].read_text())


def load_study13_panel(month_sessions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Study 13's panel (read only; D0). Held sessions are taken from the study-14 monthly table
    (same calendar, same entry / exit dates)."""
    panel = safe_load_parquet(STUDY13_PANEL, max_end_date=TRAIN_END, caller_intent="eqgrid.d0_panel13",
                              zone_config=EQUITIES_ZONES_PATH)
    months = safe_load_parquet(STUDY13_MONTHS, max_end_date=TRAIN_END,
                               caller_intent="eqgrid.d0_months13", zone_config=EQUITIES_ZONES_PATH)
    assert_pre_2019(panel, "study13 panel")
    assert_pre_2019(months, "study13 months")
    held = month_sessions.set_index("entry_date")["sessions_held"]
    months["sessions_held"] = months["entry_date"].map(held)
    assert months["sessions_held"].notna().all(), "study-13 entry dates must match the calendar"
    panel = slim(panel)
    chk = month_sessions.set_index("entry_date")["exit_date"]
    assert (months["exit_date"].to_numpy() == months["entry_date"].map(chk).to_numpy()).all()
    return panel, months, json.loads(STUDY13_META.read_text())


# ---------------------------------------------------------------------------
# Designs and refits
# ---------------------------------------------------------------------------
class Design:
    def __init__(self, panel: pd.DataFrame, months: pd.DataFrame, numeric: list[str], horizon: str,
                 tag: str) -> None:
        self.panel, self.months, self.numeric, self.horizon, self.tag = panel, months, numeric, horizon, tag
        self.columns = ml.input_columns(numeric)
        self.X = panel[self.columns].to_numpy(np.float32)
        self.y = panel["y_rank"].to_numpy(np.float64)
        self.tmi = panel["target_mi"].to_numpy()
        self.date_id = panel["date_id"].to_numpy()
        ent = pd.to_datetime(panel["entry_date"])
        exi = pd.to_datetime(panel["exit_date"])
        self.entry_ns = np.where(ent.notna(), ent.values.astype("datetime64[ns]").astype(np.int64), -1)
        self.exit_ns = np.where(exi.notna(), exi.values.astype("datetime64[ns]").astype(np.int64), -1)
        self.panel_hash = g.array_hash(self.X, self.y)

    def rows(self, rng_: tuple[int, int]) -> np.ndarray:
        return np.flatnonzero(ml.mask_months(self.tmi, rng_) & (self.tmi >= 0))

    def bounds(self, idx: np.ndarray) -> np.ndarray:
        return ml.date_bounds(self.date_id[idx])


def runs_dir(quick: bool) -> Path:
    d = GRID / ("eqml14_runs_quick" if quick else "eqml14_runs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def fit_refit(design: Design, name: str, sp: g.GridSplit, quick: bool) -> dict:
    seed = g.refit_seed(SEED, sp)
    key = g.cache_key(model=name, panel_hash=design.panel_hash, columns=design.columns, split=sp,
                      seed=seed, quick=quick)
    path = runs_dir(quick) / g.cache_filename(name, sp, key)
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        info = json.loads(meta_path.read_text())
        if info.get("cache_key") == key:
            info["pred_test"] = np.load(path)["pred_test"]
            info["cached"] = True
            return info
    fi, vi, ti = design.rows(sp.fit), design.rows(sp.val), design.rows(sp.test)
    if len(fi) == 0 or len(vi) == 0:
        raise RuntimeError(f"{design.tag} {sp.refit_id}: empty fit / validation slice")
    fd = ml.FitData(design.X[fi], design.y[fi], design.bounds(fi), design.X[vi], design.y[vi],
                    design.bounds(vi), design.columns)
    model = g.make_model(name, quick=quick)
    t0 = time.time()
    info = model.fit(fd, seed=seed)
    pred_val = np.asarray(info.pop("pred_val"), dtype=np.float64)
    pred_test = (model.predict(design.X[ti], design.bounds(ti)) if len(ti)
                 else np.zeros(0, dtype=np.float64))
    if hasattr(model, "close"):
        model.close()
    # R2 (v3): every fit must give more than one distinct prediction on every date
    info["min_distinct_val"] = g.assert_nondegenerate(pred_val, fd.bv, f"{design.tag} {sp.refit_id} {name} val")
    info["min_distinct_test"] = (g.assert_nondegenerate(pred_test, design.bounds(ti),
                                                        f"{design.tag} {sp.refit_id} {name} test")
                                 if len(ti) else None)
    info.update({"cache_key": key, "model": name, "panel_tag": design.tag, "panel_hash": design.panel_hash,
                 "split": {**sp.__dict__}, "seed": seed, "wall_seconds": time.time() - t0,
                 "n_fit": int(len(fi)), "n_val": int(len(vi)), "n_test": int(len(ti)),
                 "n_fit_dates": int(len(np.unique(design.date_id[fi]))),
                 "n_val_dates": int(len(np.unique(design.date_id[vi])))})
    np.savez_compressed(path, pred_test=np.asarray(pred_test, dtype=np.float64))
    meta_path.write_text(json.dumps(info, default=float))
    info["pred_test"] = pred_test
    info["cached"] = False
    return info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def legs_block(ls: g.LSDual, mkt: np.ndarray, h: str) -> dict:
    P = g.PERIODS_PER_YEAR[h]
    return {"long_ann": float(ls.long_ret.mean() * P), "short_ann": float(ls.short_ret.mean() * P),
            "long_minus_mkt_ann": float((ls.long_ret - mkt).mean() * P),
            "short_minus_mkt_ann": float((ls.short_ret - mkt).mean() * P),
            "mkt_ann": float(mkt.mean() * P),
            "mean_n_long": float(ls.n_long.mean()), "mean_n_short": float(ls.n_short.mean())}


def book_metrics(p: np.ndarray, sub: pd.DataFrame, b: np.ndarray, mo: pd.DataFrame, universe: str,
                 h: str, n_firms: int, *, members: bool = False) -> tuple[dict, g.LSDual, dict[str, np.ndarray]]:
    c = g.COST[universe]
    P = g.PERIODS_PER_YEAR[h]
    yex = sub["y_ex_all" if universe == "ALL" else "y_ex_liq"].to_numpy(np.float64)
    ics = ml.ic_by_date(p, yex, b)
    beta = sub["beta_exante"].to_numpy(np.float64) if "beta_exante" in sub.columns else None
    ls = g.decile_long_short_dual(p, sub["y_raw"].to_numpy(np.float64), sub["firm"].to_numpy(), b,
                                  n_firms, attr=beta, keep_members=members)
    years = mo["target_mi"].to_numpy() // 12
    held = mo["sessions_held"].to_numpy()
    mktrf = mo["mktrf"].to_numpy()
    nets = {"gross": ls.gross, "net_ew": ls.net(c, "ew"), "net_ee": ls.net(c, "ee"),
            "net_ew_2x": ls.net(2 * c, "ew"), "net_ee_2x": ls.net(2 * c, "ee")}
    stats = {k: g.period_stats(v, years, held, mktrf, h) for k, v in nets.items()}
    out = {"ic": ml.ic_summary(ics), "cost_per_side_bps": c * 1e4, **stats,
           "turnover_per_year_ew": float(ls.two_sided_turnover("ew").mean() * P),
           "turnover_per_year_ee": float(ls.two_sided_turnover("ee").mean() * P),
           "incumbent_reweight_share_of_turnover": float(
               1 - ls.two_sided_turnover("ee").sum() / ls.two_sided_turnover("ew").sum()),
           "break_even_cost_bps_ew": g.break_even_cost(ls, "ew") * 1e4,
           "break_even_cost_bps_ee": g.break_even_cost(ls, "ee") * 1e4,
           "cost_drag_ann_ew": float((ls.gross - nets["net_ew"]).mean() * P),
           "legs": legs_block(ls, mo["mkt"].to_numpy(), h), "periods": int(len(ls.gross)),
           "first_entry": str(pd.Timestamp(mo["entry_date"].iloc[0]).date()),
           "last_entry": str(pd.Timestamp(mo["entry_date"].iloc[-1]).date())}
    if beta is not None:
        hb = ls.book_attr()
        out["ex_ante_book_beta_mean"] = float(np.nanmean(hb))
        out["ex_ante_book_beta_by_year"] = {int(y): float(np.nanmean(hb[years == y])) for y in np.unique(years)}
    return out, ls, nets


def attribution(ls: g.LSDual, sub: pd.DataFrame, mo: pd.DataFrame, aux: dict) -> list[dict]:
    """Per-year FF3 + MOM attribution of the daily gross spread (R6)."""
    sess = aux["sessions"]
    e_rows = sess.get_indexer(pd.DatetimeIndex(mo["entry_date"]))
    x_rows = sess.get_indexer(pd.DatetimeIndex(mo["exit_date"]))
    assert (e_rows >= 0).all() and (x_rows >= 0).all()
    spread, rows = g.daily_spread(aux["R"], ls, sub["firm"].to_numpy(), e_rows, x_rows)
    fac = aux["fac"].iloc[rows]
    return g.factor_attribution(spread, fac, np.asarray(sess[rows].year))


class OOS:
    def __init__(self, design: Design, splits: list[g.GridSplit], preds: dict[str, list[np.ndarray]],
                 ensemble: tuple[str, ...], year_range: tuple[int, int] | None = None) -> None:
        idx = np.concatenate([design.rows(sp.test) for sp in splits])
        pr = {m: np.concatenate(v) for m, v in preds.items()}
        if year_range is not None:
            keep = ((design.tmi[idx] // 12) >= year_range[0]) & ((design.tmi[idx] // 12) <= year_range[1])
            idx = idx[keep]
            pr = {m: v[keep] for m, v in pr.items()}
        assert (np.diff(design.date_id[idx]) >= 0).all(), "OOS rows must be sorted by date"
        self.design, self.idx = design, idx
        self.p = design.panel.iloc[idx].reset_index(drop=True)
        self.bounds = ml.date_bounds(self.p["date_id"].to_numpy())
        self.pred = pr
        self.pred["ENS"] = ml.ensemble([pr[c] for c in ensemble], self.bounds)
        self.n_firms = int(design.panel["firm"].max()) + 1

    def universe(self, u: str) -> np.ndarray:
        return np.ones(len(self.p), bool) if u == "ALL" else self.p["in_liq"].to_numpy(bool)


def _slice(oos: OOS, universe: str) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    m = oos.universe(universe)
    sub = oos.p.loc[m].reset_index(drop=True)
    b = ml.date_bounds(sub["date_id"].to_numpy())
    did = sub["date_id"].to_numpy()[b[:-1]]
    mo = oos.design.months.set_index("date_id").loc[did].reset_index()
    return m, sub, b, mo


def evaluate(oos: OOS, universe: str, h: str, rng: np.random.Generator, *, with_nulls: bool = True,
             components: tuple[str, ...] = (), aux: dict | None = None) -> dict:
    m, sub, b, mo = _slice(oos, universe)
    c = g.COST[universe]
    ens, ls, nets = book_metrics(oos.pred["ENS"][m], sub, b, mo, universe, h, oos.n_firms,
                                 members=aux is not None)
    base, _, nets0 = book_metrics(oos.pred["M0"][m], sub, b, mo, universe, h, oos.n_firms)
    comp = {}
    for cname in components:
        cm, _, _ = book_metrics(oos.pred[cname][m], sub, b, mo, universe, h, oos.n_firms)
        comp[cname] = {"ic": cm["ic"], "net_sharpe": cm["net_ew"]["sharpe"],
                       "net_ann": cm["net_ew"]["ann_return"], "gross_ann": cm["gross"]["ann_return"]}
    out: dict[str, Any] = {"universe": universe, "ENS": ens, "M0": base, "components": comp,
                           "survivorship_note": SURV_NOTE, "survivorship_direction": SURV_DIRECTION}
    if aux is not None:
        out["factor_attribution"] = attribution(ls, sub, mo, aux)
    d = nets["net_ew"] - nets0["net_ew"]
    out["ens_minus_m0"] = {"net_ann_diff": g.ann_mean(d, h), "sharpe_diff":
                           ens["net_ew"]["sharpe"] - base["net_ew"]["sharpe"],
                           "share_periods_better": float((d > 0).mean()),
                           "block_bootstrap_p_le0": g.n2_bootstrap(d, h, rng=rng, n_draws=N_N2)["p"]}
    if with_nulls:
        t0 = time.time()
        real = {"gross": ens["gross"]["sharpe"], "ew1": ens["net_ew"]["sharpe"],
                "ew2": ens["net_ew_2x"]["sharpe"], "ee1": ens["net_ee"]["sharpe"]}
        n1 = g.n1_permutation_dual(oos.pred["ENS"][m], sub["y_raw"].to_numpy(np.float64),
                                   sub["firm"].to_numpy(), b, oos.n_firms, h, real,
                                   {"gross": (0.0, "gross"), "ew1": (c, "ew"), "ew2": (2 * c, "ew"),
                                    "ee1": (c, "ee")}, n_draws=N_N1, rng=rng)
        n2 = {"ew1": g.n2_bootstrap(nets["net_ew"], h, rng=rng, n_draws=N_N2),
              "ew2": g.n2_bootstrap(nets["net_ew_2x"], h, rng=rng, n_draws=N_N2),
              "ee1": g.n2_bootstrap(nets["net_ee"], h, rng=rng, n_draws=N_N2),
              "ew1_12m_block": g.n2_bootstrap(nets["net_ew"], h, rng=rng, n_draws=N_N2,
                                              block=g.N2_BLOCK_12M[h])}
        # v3 (R1): the N1 gate is the GROSS-statistic permutation p; N1 on net is a diagnostic
        v1 = g.cell_verdict(ens["net_ew"]["sharpe"], n1["gross"]["p"], n2["ew1"]["p"],
                            ens["net_ew"]["per_year"])
        v2 = g.cell_verdict(ens["net_ew_2x"]["sharpe"], n1["gross"]["p"], n2["ew2"]["p"],
                            ens["net_ew_2x"]["per_year"])
        out.update({"N1": n1, "N2": n2, "verdict": v1["verdict"], "fails": v1["fails"],
                    "years_rule": v1["years"], "verdict_2x_cost": v2["verdict"], "fails_2x_cost": v2["fails"],
                    "null_seconds": round(time.time() - t0, 1)})
    return out


def evaluate_hedged(oos: OOS, universe: str, h: str, rng: np.random.Generator) -> dict:
    """D7: the ENS book index-hedged with its ex-ante book beta, hedge turnover charged."""
    m, sub, b, mo = _slice(oos, universe)
    c = g.COST[universe]
    beta = sub["beta_exante"].to_numpy(np.float64)
    ls = g.decile_long_short_dual(oos.pred["ENS"][m], sub["y_raw"].to_numpy(np.float64),
                                  sub["firm"].to_numpy(), b, oos.n_firms, attr=beta)
    mktrf = mo["mktrf"].to_numpy()
    years = mo["target_mi"].to_numpy() // 12
    held = mo["sessions_held"].to_numpy()
    ser = {"gross": g.hedged_series(ls, mktrf, c, gross=True), "net_ew": g.hedged_series(ls, mktrf, c),
           "net_ew_2x": g.hedged_series(ls, mktrf, 2 * c)}
    stats = {k: g.period_stats(v, years, held, mktrf, h) for k, v in ser.items()}
    real = {"gross": stats["gross"]["sharpe"], "ew1": stats["net_ew"]["sharpe"],
            "ew2": stats["net_ew_2x"]["sharpe"]}
    n1 = g.n1_permutation_dual(oos.pred["ENS"][m], sub["y_raw"].to_numpy(np.float64), sub["firm"].to_numpy(),
                               b, oos.n_firms, h, real, {"gross": (c, "gross"), "ew1": (c, "ew"),
                                                         "ew2": (2 * c, "ew")},
                               n_draws=N_N1, rng=rng, beta=beta, mktrf=mktrf)
    n2 = {"ew1": g.n2_bootstrap(ser["net_ew"], h, rng=rng, n_draws=N_N2),
          "ew2": g.n2_bootstrap(ser["net_ew_2x"], h, rng=rng, n_draws=N_N2)}
    v1 = g.cell_verdict(stats["net_ew"]["sharpe"], n1["gross"]["p"], n2["ew1"]["p"], stats["net_ew"]["per_year"])
    hb = ls.book_attr()
    return {"universe": universe, "ENS_hedged": {**stats, "periods": int(len(ls.gross)),
                                                 "hedge_turnover_per_year": float(np.abs(np.diff(np.r_[0.0, np.nan_to_num(hb)])).mean()
                                                                                  * g.PERIODS_PER_YEAR[h]),
                                                 "ex_ante_book_beta_mean": float(np.nanmean(hb))},
            "N1": n1, "N2": n2, "verdict_info": v1["verdict"], "fails": v1["fails"], "years_rule": v1["years"]}


def cell_rng(cell_id: str, universe: str) -> np.random.Generator:
    order = list(g.ALL_CELLS) + [f"D6_{t.cell_id}" for t in g.TRIALS]
    return np.random.default_rng([SEED, order.index(cell_id), 0 if universe == "LIQ" else 1])


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------
class Panels:
    def __init__(self) -> None:
        self.cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]] = {}
        self.designs: dict[tuple, Design] = {}

    def aux(self) -> dict:
        """Daily returns (TRAIN calendar) and French daily factors for the R6 attribution."""
        if not hasattr(self, "_aux"):
            z = np.load(GRID / "aux_daily.npz")
            sess = pd.DatetimeIndex(z["sessions"].astype("datetime64[ns]"))
            assert sess.max() < TRAIN_END
            ff = xs.load_ff_daily(end=TRAIN_END)
            idx = pd.DatetimeIndex(ff.index)
            ff.index = idx.tz_localize(None) if idx.tz is not None else idx
            fac = ff.loc[ff.index < TRAIN_END].reindex(sess)
            self._aux = {"R": z["R"], "sessions": sess, "fac": fac}
        return self._aux

    def get(self, which: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        if which not in self.cache:
            if which == "study13":
                _, mo_m, _ = self.get("M")
                self.cache[which] = load_study13_panel(mo_m)
            else:
                # keep at most the requested grid panel plus the monthly one in memory
                for k in [k for k in self.cache if k not in ("M", which, "study13")]:
                    self.cache.pop(k)
                    self.designs = {kk: v for kk, v in self.designs.items() if kk[0] != k}
                self.cache[which] = load_grid_panel(which)
            log(f"panel {which} loaded: {len(self.cache[which][0]):,} rows (RSS peak {rss_gb():.2f} GB)")
        return self.cache[which]

    def design(self, spec: g.CellSpec) -> Design:
        which = "study13" if spec.panel == "study13" else spec.horizon
        key = (which, spec.drop_features)
        if key not in self.designs:
            panel, months, meta = self.get(which)
            numeric = [f for f in ml.NUMERIC_FEATURES
                       if f not in meta["dropped_features"] and f not in spec.drop_features]
            self.designs[key] = Design(panel, months, numeric, spec.horizon, f"{which}{'-abl' if spec.drop_features else ''}")
        return self.designs[key]


def run_cell(spec: g.CellSpec, panels: Panels, quick: bool) -> dict:
    t0 = time.time()
    load0 = os.getloadavg()
    design = panels.design(spec)
    splits = g.grid_splits(spec.horizon, spec.window, spec.refit, spec.embargo)
    if quick:
        splits = splits[:2]
    viol = g.check_embargo(splits, design.tmi, design.entry_ns, design.exit_ns)
    if viol:
        raise RuntimeError(f"{spec.cell_id}: embargo violations {viol}")
    models = list(g.GRID_MODELS)
    preds: dict[str, list[np.ndarray]] = {m: [] for m in models}
    refits = []
    fit_seconds = 0.0
    for sp in splits:
        for name in models:
            if time.time() - t0 > g.CAP_SECONDS:
                return {"cell": spec.as_dict(), "status": "NOT RUN",
                        "reason": f"wall clock cap {g.CAP_SECONDS / 60:.0f} min exceeded at refit "
                                  f"{sp.refit_id} {name}", "elapsed_s": time.time() - t0,
                        "load_avg_start": load0, "load_avg_end": os.getloadavg()}
            try:
                info = fit_refit(design, name, sp, quick)
            except g.DegeneratePredictor as exc:
                return {"cell": spec.as_dict(), "status": "NOT RUN",
                        "reason": f"degenerate predictor assertion (R2): {exc}",
                        "elapsed_s": time.time() - t0, "load_avg_start": load0,
                        "load_avg_end": os.getloadavg()}
            preds[name].append(np.asarray(info["pred_test"]))
            fit_seconds += 0.0 if info["cached"] else float(info["wall_seconds"])
            keep = {k: info.get(k) for k in ("alpha", "params", "best_iteration", "lam", "lr", "n_seeds",
                                             "val_ic", "val_ic_se", "min_distinct_val", "min_distinct_test",
                                             "wall_seconds", "n_fit", "n_val", "n_test",
                                             "n_fit_dates", "n_val_dates", "cached")}
            refits.append({"refit_id": sp.refit_id, "model": name, **keep})
            log(f"[{spec.cell_id}] {sp.refit_id} {name}: val IC {info['val_ic']:+.4f} "
                f"(se {info.get('val_ic_se', float('nan')):.4f}), "
                f"{info['wall_seconds']:.0f} s{' (cache)' if info['cached'] else ''}, fit rows "
                f"{info['n_fit']:,} / {info['n_fit_dates']} dates, val {info['n_val_dates']} dates")
    res: dict[str, Any] = {"cell": spec.as_dict(), "status": "RUN", "quick": quick,
                           "splits": [sp.__dict__ for sp in splits], "refits": refits,
                           "panel_hash": design.panel_hash, "inputs": design.columns,
                           "fit_seconds": round(fit_seconds, 1)}
    if quick:
        res["elapsed_s"] = round(time.time() - t0, 1)
        return res
    oos = OOS(design, splits, preds, spec.ensemble)
    save_predictions(spec.cell_id, oos)
    t1 = time.time()
    if spec.hedged:
        res["LIQ"] = evaluate_hedged(oos, "LIQ", spec.horizon, cell_rng(spec.cell_id, "LIQ"))
    else:
        res["LIQ"] = evaluate(oos, "LIQ", spec.horizon, cell_rng(spec.cell_id, "LIQ"),
                              components=tuple(m for m in models if m != "M0"),
                              aux=panels.aux() if spec.role == "trial" else None)
        res["ALL"] = evaluate(oos, "ALL", spec.horizon, cell_rng(spec.cell_id, "ALL"))
    res["eval_seconds"] = round(time.time() - t1, 1)
    res["elapsed_s"] = round(time.time() - t0, 1)
    res["load_avg_start"], res["load_avg_end"] = load0, os.getloadavg()
    res["peak_rss_gb_main"] = round(rss_gb(), 2)
    log(f"[{spec.cell_id}] done in {res['elapsed_s']:.0f} s")
    return res


def save_predictions(cell_id: str, oos: OOS) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pr = oos.p.loc[:, ["date", "ts", "date_id", "target_mi", "entry_date", "exit_date", "firm",
                       "in_liq"]].copy()
    for k, v in oos.pred.items():
        pr[f"pred_{k}"] = np.asarray(v, dtype=np.float64)  # R9: float64
    assert_pre_2019(pr, f"predictions {cell_id}")
    pr.to_parquet(PRED_DIR / f"{cell_id}.parquet", index=False)


def run_d6(panels: Panels) -> dict:
    """Every trial re-sliced to OOS 2014-2018: the same predictions, no refit; books restart in
    2014-01 (first period full turnover)."""
    out = {}
    for spec in g.TRIALS:
        cj = CELLS_DIR / f"{spec.cell_id}.json"
        if not cj.exists():
            out[spec.cell_id] = {"status": "NOT RUN", "reason": "trial not run"}
            continue
        r = json.loads(cj.read_text())
        if r.get("status") != "RUN":
            out[spec.cell_id] = {"status": "NOT RUN", "reason": "trial NOT RUN"}
            continue
        design = panels.design(spec)
        splits = g.grid_splits(spec.horizon, spec.window, spec.refit, spec.embargo)
        preds = {m: [] for m in g.GRID_MODELS}
        for sp in splits:
            for name in g.GRID_MODELS:
                info = fit_refit(design, name, sp, False)
                assert info["cached"], f"D6 must not refit ({spec.cell_id} {sp.refit_id} {name})"
                preds[name].append(np.asarray(info["pred_test"]))
        oos = OOS(design, splits, preds, spec.ensemble, year_range=g.D6_YEARS)
        e = evaluate(oos, "LIQ", spec.horizon, cell_rng(f"D6_{spec.cell_id}", "LIQ"))
        out[spec.cell_id] = {"status": "RUN", "LIQ": e}
        log(f"[D6 {spec.cell_id}] net SR {e['ENS']['net_ew']['sharpe']:+.2f}")
    return out


# ---------------------------------------------------------------------------
# Study assembly
# ---------------------------------------------------------------------------
def study13_reference() -> dict:
    o = json.loads(STUDY13_JSON.read_text())
    ref = {}
    for u in ("LIQ", "ALL"):
        h = o["headline"][u]
        ref[u] = {"cell": f"eqml_ENS_{u}", "components": h.get("components"), "ic": h["ic"],
                  "gross_sharpe": h["gross"]["sharpe"], "gross_ann": h["gross"]["ann_return"],
                  "net_sharpe": h["net"]["sharpe"], "net_ann": h["net"]["ann_return"],
                  "max_dd": h["net"]["max_dd"], "beta": h["net"]["beta"], "N1_p": h["N1"]["p"],
                  "N2_p": h["N2"]["p"], "N2_mde": h["N2"]["mde_ann_return"],
                  "N1_mde": h["N1"]["mde_ann_return"], "years_positive": h["net"]["years_positive"],
                  "years_counted": h["net"]["years_counted"], "verdict": h["verdict"],
                  "fails": h["fails"], "rule": "study 13's own rule (8 of 10 positive years)"}
    return ref


def assemble() -> dict:
    cells = {}
    for cid in g.ALL_CELLS:
        p = CELLS_DIR / f"{cid}.json"
        cells[cid] = json.loads(p.read_text()) if p.exists() else {"status": "NOT RUN",
                                                                   "reason": "not executed"}
    d6p = CELLS_DIR / "D6.json"
    d6 = json.loads(d6p.read_text()) if d6p.exists() else {}
    trials_in = {}
    for t in g.TRIALS:
        c = cells[t.cell_id]
        if c.get("status") == "RUN":
            st = c["LIQ"]["ENS"]["net_ew"]
            trials_in[t.cell_id] = {"sr_ann": st["sharpe"], "horizon": t.horizon, "n": st["periods"],
                                    "skew": st["skew"], "kurt": st["kurtosis"]}
        else:
            trials_in[t.cell_id] = None
    dsr = g.deflated_sharpe(trials_in, n_declared=g.N_TRIALS)
    # study verdict
    t5 = cells["T5"]
    a_pass = t5.get("status") == "RUN" and t5["LIQ"]["verdict"] == "PASS"
    b_cells = []
    for t in g.TRIALS:
        c = cells[t.cell_id]
        if c.get("status") != "RUN":
            continue
        L = c["LIQ"]
        dd = dsr["per_cell"].get(t.cell_id, {}).get("dsr", float("nan"))
        if L["verdict"] == "PASS" and dd > 0.95 and L["verdict_2x_cost"] == "PASS":
            labels = ["GRID-SELECTED"]
            if L["ens_minus_m0"]["block_bootstrap_p_le0"] >= 0.05:
                labels.append("baseline-equivalent")
            b_cells.append({"cell": t.cell_id, "dsr": dd, "labels": labels})
    study_pass = a_pass or bool(b_cells)
    verdict = {"route_a_T5_pass": a_pass, "route_b_cells": b_cells,
               "study_verdict": "PASS" if study_pass else "REJECT",
               "per_trial": {t.cell_id: (cells[t.cell_id]["LIQ"]["verdict"]
                                         if cells[t.cell_id].get("status") == "RUN" else "NOT RUN")
                             for t in g.TRIALS}}
    metas = {h: json.loads(panel_paths(h)["meta"].read_text()) for h in ("M", "Q", "W")
             if panel_paths(h)["meta"].exists()}
    return {"study": "eqgrid (study 14)", "zone": "equities TRAIN 2004-01-01..2019-01-01",
            "predeclaration": "study 14 pre-declaration (v2)",
            "seed": SEED, "survivorship_note": SURV_NOTE, "survivorship_direction": SURV_DIRECTION,
            "periods_per_year": g.PERIODS_PER_YEAR, "n2_block": g.N2_BLOCK,
            "panel_meta": metas, "cells": cells, "D6": d6, "dsr": dsr, "verdict": verdict,
            "study13_reference": study13_reference(),
            "equity_ml_sha256": hashlib.sha256((REPO / "src/eqxs/equity_ml.py").read_bytes()).hexdigest()}


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _pc(x: Any, nd: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:+.{nd}f}%"


def _f(x: Any, nd: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def _p(x: Any) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.3f}"


def _sr(st: dict) -> str:
    return f"{_f(st['sharpe'])} ({_pc(st['ann_return'])})"


def cell_label(spec: dict) -> str:
    return (f"{spec['window']}, {spec['refit']} refit, {H_NAME[spec['horizon']]}, "
            f"embargo {spec['embargo']} m")


def write_markdown(out: dict) -> None:
    L: list[str] = []
    a = L.append
    cells, dsr, ver = out["cells"], out["dsr"], out["verdict"]
    a("# Study 14: training-window / timeframe grid for the equity ML cross-section, TRAIN only")
    a("")
    a("Study id `eqgrid`, equities zone, TRAIN 2004-01-01 → 2019-01-01 (exclusive); WFA and sealed untouched. "
      "Pre-declaration: study 14 (v2, after the design review). Library `eqxs.equity_ml_grid` "
      "(wraps study 13's `equity_ml`, unmodified), driver `scripts/equity_ml_window_grid_train.py`, "
      "cache `data_cache/equities_xs/derived/eqml_grid/`. Auto-generated by the driver.")
    a("")
    a(f"**Every table is on the {SURV_NOTE}.** Survivorship direction for every cell: {SURV_DIRECTION}. "
      "Each cell's long − market and short − market columns below show the measured legs.")
    a("")
    a("Annualisation per horizon: P = 50.4 (weekly, 252/5), 12 (monthly), 4 (quarterly); Sharpe = mean/sd × √P, "
      "annualised return = mean × P (in brackets beside every Sharpe); 252 sessions/year basis. Costs per side: "
      "5 bps liquid half (LIQ, the trial universe), 15 bps all firms (ALL, sensitivity only).")
    a("")
    a(f"## Verdict: **{ver['study_verdict']}**")
    a("")
    t5 = cells["T5"]
    a(f"* Route (a), headline T5 PASS: **{ver['route_a_T5_pass']}**"
      + (f" (T5 fails: {', '.join(t5['LIQ']['fails']) or '-'})" if t5.get("status") == "RUN" else " (T5 NOT RUN)"))
    if ver["route_b_cells"]:
        for bc in ver["route_b_cells"]:
            a(f"* Route (b): {bc['cell']} PASS, DSR {bc['dsr']:.3f}, passes at 2× cost — labels {', '.join(bc['labels'])}")
    else:
        a("* Route (b): no trial cell PASSes the per-cell rule with DSR > 0.95 and a 2×-cost pass.")
    a(f"* Per-trial verdicts: {', '.join(f'{k} {v}' for k, v in ver['per_trial'].items())}.")
    a("")
    a("## Summary")
    a("")
    for line in SUMMARY_AFTER_RUN:
        a(line)
        a("")
    a("## Disclosures (implementation; decided before any study-14 result was seen)")
    a("")
    for line in DISCLOSURES:
        a(f"* {line}")
    a("")
    # trial table
    a("## Trial table (LIQ, 5 bps/side, ENS5 = M0..M4; OOS target months 2009-01 → last feasible period)")
    a("")
    a("| cell | configuration | periods | IC (t) | ICIR | gross SR (ann.) | net SR EW-reb. (ann.) | net SR entry/exit (ann.) "
      "| net SR 2× cost (ann.) | turnover/yr EW (E/E) | break-even bps/side | max DD | β realised | ex-ante book β | "
      "N1 p gross (gate) | N1 p net (diag) | N2 p | N1-net MDE (excess) "
      "| N2 MDE | years ex-best mean (pos/n) | verdict | fails | 2× cost verdict |")
    a("|" + "---|" * 23)
    for t in g.TRIALS:
        c = cells[t.cell_id]
        if c.get("status") != "RUN":
            a(f"| {t.cell_id} | {cell_label(t.as_dict())} | NOT RUN: {c.get('reason')} |" + " |" * 20)
            continue
        Lq = c["LIQ"]
        E = Lq["ENS"]
        yr = Lq["years_rule"]
        a(f"| {t.cell_id}{' (headline)' if t.cell_id == 'T5' else ''} | {cell_label(t.as_dict())} | {E['periods']} | "
          f"{E['ic']['ic_mean']:+.4f} ({_f(E['ic']['ic_t'])}) | {_f(E['ic']['icir'], 3)} | {_sr(E['gross'])} | "
          f"{_sr(E['net_ew'])} | {_sr(E['net_ee'])} | {_sr(E['net_ew_2x'])} | {E['turnover_per_year_ew']:.1f} "
          f"({E['turnover_per_year_ee']:.1f}) | {E['break_even_cost_bps_ew']:+.1f} | {_pc(E['net_ew']['max_dd'], 1)} | "
          f"{_f(E['net_ew']['beta'])}{' ⚠' if E['net_ew']['beta_flag'] else ''} | "
          f"{_f(E.get('ex_ante_book_beta_mean'))} | {_p(Lq['N1']['gross']['p'])} | {_p(Lq['N1']['ew1']['p'])} | "
          f"{_p(Lq['N2']['ew1']['p'])} | {_pc(Lq['N1']['ew1']['mde_ann_return'])} "
          f"({_pc(Lq['N1']['ew1']['mde_excess_over_null_mean'])}) | {_pc(Lq['N2']['ew1']['mde_ann_return'])} | "
          f"{_pc(yr['mean_ex_best_year'])} ({yr['years_positive']}/{yr['years_counted']}) | **{Lq['verdict']}** | "
          f"{', '.join(Lq['fails']) or '-'} | {Lq['verdict_2x_cost']} |")
    a("")
    a("IC = mean per-period within-date Spearman of the ENS5 prediction vs the realised excess return (t = mean/sd·√n); "
      "ICIR = mean/sd per period (not annualised). Turnover = Σ|Δw| over both legs per year (full replacement of both "
      "legs = 4.0 per rebalance); EW-reb. = equal-weight rebalance accounting (study 13), E/E = entry/exit only (red team F3). "
      "Break-even = gross mean / two-sided turnover (cost per side at which net = 0). N1 = within-date permutation of "
      "the predictions (500 draws, same portfolio code); v3 gate = the GROSS-statistic p (does the sort beat a random "
      "sort of the same names), the net-statistic p is a persistence diagnostic (a random book turns over more and "
      "pays more cost); the MDE column is the net-statistic null's MDE with the excess over the null mean in brackets. N2 = "
      "centred circular block bootstrap of the net series, ~3-month blocks (13/3/1 periods), 1000 draws, H0 mean ≤ 0; "
      "MDE = (q95 − q20) × P. Years rule `ex_best_year_positive` over years with ≥ 63 held sessions. β vs Mkt-RF "
      "compounded over each holding window, NW ceil(P/4) lags; ⚠ when |β| > 0.25.")
    a("")
    # N2 diagnostics
    a("### N2 diagnostics and the entry/exit and 2× cost nulls (LIQ)")
    a("")
    a("| cell | N2 p 3-m block | N2 p 12-m block | NW-t mean (12-m bandwidth) | N1 p E/E | N2 p E/E | N1 p 2× | N2 p 2× | null N1 mean net SR (ann. ret.) |")
    a("|---|---|---|---|---|---|---|---|---|")
    for t in g.TRIALS:
        c = cells[t.cell_id]
        if c.get("status") != "RUN":
            continue
        Lq = c["LIQ"]
        a(f"| {t.cell_id} | {_p(Lq['N2']['ew1']['p'])} | {_p(Lq['N2']['ew1_12m_block']['p'])} | "
          f"{_f(Lq['ENS']['net_ew']['nw_t_mean_12m'])} | {_p(Lq['N1']['ee1']['p'])} | {_p(Lq['N2']['ee1']['p'])} | "
          f"{_p(Lq['N1']['ew2']['p'])} | {_p(Lq['N2']['ew2']['p'])} | {_f(Lq['N1']['ew1']['null_sharpe_mean'])} "
          f"({_pc(Lq['N1']['ew1']['null_ann_mean'])}) |")
    a("")
    # DSR
    a("## Deflated Sharpe (Bailey & López de Prado 2014, pinned in v2)")
    a("")
    a(f"N = {dsr['N']} declared trials ({dsr['n_ran']} ran; NOT RUN: {', '.join(dsr['not_run']) or 'none'}). "
      f"Cross-trial variance V of the annualised LIQ net SRs = {dsr['V']:.4f} (sd {dsr['sd_cross_trial']:.3f}); "
      f"SR0 = √V·[(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne))] = **{dsr['sr0_ann']:+.3f}** annualised. DSR_i = PSR of cell i "
      "against SR0/√P_i in its own period units (own n, skewness, kurtosis).")
    a("")
    a("| cell | net SR (ann.) | net ann. return | periods | skew | kurtosis | DSR (PSR vs SR0) | > 0.95 |")
    a("|---|---|---|---|---|---|---|---|")
    for t in g.TRIALS:
        pc = dsr["per_cell"].get(t.cell_id)
        if pc is None:
            a(f"| {t.cell_id} | NOT RUN | | | | | | |")
            continue
        st = cells[t.cell_id]["LIQ"]["ENS"]["net_ew"]
        a(f"| {t.cell_id}{' **(best)**' if t.cell_id == dsr['best_cell'] else ''} | {_f(pc['sr_ann'])} | "
          f"{_pc(st['ann_return'])} | {st['periods']} | {_f(st['skew'])} | {st['kurtosis']:.2f} | {pc['dsr']:.3f} | "
          f"{'yes' if pc['dsr'] > 0.95 else 'no'} |")
    a("")
    a(f"Best cell {dsr['best_cell']}: DSR {dsr['best_dsr']:.3f} → {'PASS' if dsr['best_dsr'] > 0.95 else 'FAIL'} "
      "(the DSR bar alone; route (b) also needs the per-cell PASS and the 2×-cost pass).")
    a("")
    # study-13 reference
    ref = out["study13_reference"]["LIQ"]
    a("## Study-13 reference (M6) beside T5 ENS5 and D0 (LIQ, 5 bps)")
    a("")
    a("| row | model set / panel | IC (t) | gross SR (ann.) | net SR (ann.) | max DD | β | N1 p | N2 p | years | verdict (rule) |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    a(f"| study 13 M6 (reference, not re-graded) | {', '.join(ref['components'] or [])}; study-13 panel | "
      f"{ref['ic']['ic_mean']:+.4f} ({_f(ref['ic']['ic_t'])}) | {_f(ref['gross_sharpe'])} ({_pc(ref['gross_ann'])}) | "
      f"{_f(ref['net_sharpe'])} ({_pc(ref['net_ann'])}) | {_pc(ref['max_dd'], 1)} | {_f(ref['beta'])} | "
      f"{_p(ref['N1_p'])} | {_p(ref['N2_p'])} | {ref['years_positive']}/{ref['years_counted']} pos | "
      f"{ref['verdict']} (study 13: 8/10 years) |")
    for cid, lab in (("T5", "ENS5 M0-M4; study-14 monthly panel (acceptance-session day 0, session lookbacks)"),
                     ("D3", "ENS3 M2-M4 on the T5 fits; study-14 monthly panel"),
                     ("D0", "ENS5 M0-M4; study-13 panel (next-session day 0, positional lookbacks)")):
        c = cells[cid]
        if c.get("status") != "RUN":
            a(f"| {cid} | {lab} | NOT RUN | | | | | | | | |")
            continue
        Lq = c["LIQ"]
        E = Lq["ENS"]
        yr = Lq["years_rule"]
        a(f"| {cid} | {lab} | {E['ic']['ic_mean']:+.4f} ({_f(E['ic']['ic_t'])}) | {_sr(E['gross'])} | {_sr(E['net_ew'])} | "
          f"{_pc(E['net_ew']['max_dd'], 1)} | {_f(E['net_ew']['beta'])} | {_p(Lq['N1']['gross']['p'])} (gross; net "
          f"{_p(Lq['N1']['ew1']['p'])}) | "
          f"{_p(Lq['N2']['ew1']['p'])} | {yr['years_positive']}/{yr['years_counted']} pos, ex-best {_pc(yr['mean_ex_best_year'])} | "
          f"{Lq['verdict']} (study 14; {'trial' if cid == 'T5' else 'diagnostic, never PASS'}) |")
    a("")
    a("The study-13 row's N1 p is study 13's net-statistic N1 (its gate). D0 differs from T5 in the PANEL only (same "
      "v3 model rules, same splits, same seeds): the study-13 panel dates INTRADAY 8-Ks to the next session, uses "
      "positional (month-end-row) lookbacks for `mom_12_1`, `mom_6_1`, `rev_1m` and the 12-month-earlier fundamentals "
      "instant, and has no R8 share flag; the study-14 panel uses the acceptance session, session-based lookbacks "
      "and the R8 flag. T5 − D0 is the effect of those panel changes together. D3 − T5 isolates the ensemble "
      "composition (dropping M0/M1). The difference between study-13 M6 and T5 is panel dating + model rules "
      "(v3 fixed boosters, tie-safe IC) + ensemble composition (no M5, adds M0/M1); it is not attributable to any one.")
    a("")
    # M0 at every cell
    a("## M0 (ridge-3 baseline) at every cell and ENS5 − M0 (LIQ, net EW)")
    a("")
    a("| cell | M0 IC (t) | M0 net SR (ann.) | ENS5 net SR (ann.) | ENS5 − M0 net/yr | Δ SR | periods ENS5 better | block-bootstrap p (diff ≤ 0) |")
    a("|---|---|---|---|---|---|---|---|")
    for cid in [t.cell_id for t in g.TRIALS] + ["D0", "D1", "D2", "D5w", "D5m", "D5q"]:
        c = cells[cid]
        if c.get("status") != "RUN":
            a(f"| {cid} | NOT RUN | | | | | | |")
            continue
        Lq = c["LIQ"]
        M0, E, dm = Lq["M0"], Lq["ENS"], Lq["ens_minus_m0"]
        a(f"| {cid} | {M0['ic']['ic_mean']:+.4f} ({_f(M0['ic']['ic_t'])}) | {_sr(M0['net_ew'])} | {_sr(E['net_ew'])} | "
          f"{_pc(dm['net_ann_diff'])} | {_f(dm['sharpe_diff'])} | {dm['share_periods_better']:.0%} | "
          f"{_p(dm['block_bootstrap_p_le0'])} |")
    a("")
    # legs and survivorship
    a("## Legs and survivorship direction (LIQ, ENS5)")
    a("")
    a("| cell | long ann. | short ann. | long − mkt | short − mkt | mkt ann. | names L, S | survivorship read |")
    a("|---|---|---|---|---|---|---|---|")
    for cid in [t.cell_id for t in g.TRIALS]:
        c = cells[cid]
        if c.get("status") != "RUN":
            continue
        lg = c["LIQ"]["ENS"]["legs"]
        read = ("short leg earns above the market: the missing dead firms would have pulled it down, so the measured "
                "spread understates a short-side edge and overstates nothing on that side; the long leg is survivor-inflated"
                if lg["short_minus_mkt_ann"] > 0 else
                "short leg below the market even without the dead firms (whose absence biases it up); long leg survivor-inflated")
        a(f"| {cid} | {_pc(lg['long_ann'])} | {_pc(lg['short_ann'])} | {_pc(lg['long_minus_mkt_ann'])} | "
          f"{_pc(lg['short_minus_mkt_ann'])} | {_pc(lg['mkt_ann'])} | {lg['mean_n_long']:.0f}, {lg['mean_n_short']:.0f} | {read} |")
    a("")
    # per-year
    a("## Per-year net returns, LIQ ENS5 (compounded by year of entry; held sessions in brackets)")
    a("")
    ran = [t.cell_id for t in g.TRIALS if cells[t.cell_id].get("status") == "RUN"]
    a("| year | " + " | ".join(ran) + " |")
    a("|---|" + "---|" * len(ran))
    yrs = sorted({r["year"] for cid in ran for r in cells[cid]["LIQ"]["ENS"]["net_ew"]["per_year"]})
    for y in yrs:
        row = [str(y)]
        for cid in ran:
            r = next((r for r in cells[cid]["LIQ"]["ENS"]["net_ew"]["per_year"] if r["year"] == y), None)
            row.append("" if r is None else f"{_pc(r['ret_compounded'], 1)} ({r['sessions']})")
        a("| " + " | ".join(row) + " |")
    a("| ex-best-year mean | " + " | ".join(_pc(cells[cid]["LIQ"]["years_rule"]["mean_ex_best_year"], 1) for cid in ran) + " |")
    a("| positive years | " + " | ".join(f"{cells[cid]['LIQ']['years_rule']['years_positive']}/"
                                        f"{cells[cid]['LIQ']['years_rule']['years_counted']}" for cid in ran) + " |")
    a("")
    # ALL sensitivity
    a("## All-firm sensitivity cells (15 bps/side; never trials, never PASS)")
    a("")
    a("| cell | IC (t) | gross SR (ann.) | net SR EW (ann.) | net SR E/E (ann.) | turnover/yr | break-even bps | β | N1 p gross | N1 p net | N2 p | years ex-best (pos/n) | per-cell rule (info) |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in g.TRIALS:
        c = cells[t.cell_id]
        if c.get("status") != "RUN":
            continue
        A = c["ALL"]
        E = A["ENS"]
        yr = A["years_rule"]
        a(f"| {t.cell_id} | {E['ic']['ic_mean']:+.4f} ({_f(E['ic']['ic_t'])}) | {_sr(E['gross'])} | {_sr(E['net_ew'])} | "
          f"{_sr(E['net_ee'])} | {E['turnover_per_year_ew']:.1f} | {E['break_even_cost_bps_ew']:+.1f} | {_f(E['net_ew']['beta'])} | "
          f"{_p(A['N1']['gross']['p'])} | {_p(A['N1']['ew1']['p'])} | {_p(A['N2']['ew1']['p'])} | {_pc(yr['mean_ex_best_year'])} "
          f"({yr['years_positive']}/{yr['years_counted']}) | {A['verdict']} (sensitivity) |")
    a("")
    # diagnostics
    def diag_row(cid: str, c: dict) -> str:
        if c.get("status") != "RUN":
            return f"| {cid} | NOT RUN: {c.get('reason')} | | | | | | | |"
        Lq = c["LIQ"]
        E = Lq["ENS"]
        yr = Lq["years_rule"]
        return (f"| {cid} | {E['ic']['ic_mean']:+.4f} ({_f(E['ic']['ic_t'])}) | {_sr(E['gross'])} | {_sr(E['net_ew'])} | "
                f"{E['turnover_per_year_ew']:.1f} | {_f(E['net_ew']['beta'])} | {_p(Lq['N1']['gross']['p'])} | "
                f"{_p(Lq['N2']['ew1']['p'])} | {_pc(yr['mean_ex_best_year'])} ({yr['years_positive']}/{yr['years_counted']}) |")

    hdr = ("| cell | IC (t) | gross SR (ann.) | net SR (ann.) | turnover/yr | β | N1 p gross | N2 p | years ex-best (pos/n) |\n"
           "|---|---|---|---|---|---|---|---|---|")
    for title, rows, ref_ids, txt in (
        ("D0: T5 on the study-13 panel", ["D0"], ["T5"], "Bridge for the panel change (day-0 rule + lookback definition)."),
        ("D1: T5 with quarterly refit (40 refits, 6-month validation)", ["D1"], ["T5"], "Refit cadence at the headline."),
        ("D2: T5 with a 3-month embargo", ["D2"], ["T5"], "Embargo at the headline."),
        ("D3: ENS3 = M2 + M3 + M4 on the T5 fits", ["D3"], ["T5"], "Model-set effect (drop M0, M1)."),
        ("D5: survivorship-family ablation (rev_1m, rev_1w, ivol_63, maxret_21 removed; all five models refit)",
         ["D5w", "D5m", "D5q"], ["T1", "T5", "T7"], "D5w vs T1, D5m vs T5, D5q vs T7."),
    ):
        a(f"## {title} (diagnostic, LIQ)")
        a("")
        a(txt)
        a("")
        a(hdr)
        for cid in [*ref_ids, *rows]:
            a(diag_row(cid, cells[cid]))
        a("")
    a("## D6: every trial re-sliced to OOS 2014-2018 (same predictions, no refit; books restart 2014-01)")
    a("")
    a("| cell | periods | IC (t) | gross SR (ann.) | net SR (ann.) | N1 p gross | N2 p | years ex-best (pos/n) | rule (info) |")
    a("|---|---|---|---|---|---|---|---|---|")
    for t in g.TRIALS:
        r = out["D6"].get(t.cell_id)
        if not r or r.get("status") != "RUN":
            a(f"| {t.cell_id} | NOT RUN | | | | | | | |")
            continue
        Lq = r["LIQ"]
        E = Lq["ENS"]
        yr = Lq["years_rule"]
        a(f"| {t.cell_id} | {E['periods']} | {E['ic']['ic_mean']:+.4f} ({_f(E['ic']['ic_t'])}) | {_sr(E['gross'])} | "
          f"{_sr(E['net_ew'])} | {_p(Lq['N1']['gross']['p'])} | {_p(Lq['N2']['ew1']['p'])} | {_pc(yr['mean_ex_best_year'])} "
          f"({yr['years_positive']}/{yr['years_counted']}) | {Lq['verdict']} (diagnostic) |")
    a("")
    # ex-ante book beta per year
    a("## Ex-ante book beta per year (LIQ ENS5; long-decile mean minus short-decile mean of the trailing 252-session "
      "beta vs Mkt-RF, PIT)")
    a("")
    ran = [t.cell_id for t in g.TRIALS if cells[t.cell_id].get("status") == "RUN"]
    a("| year | " + " | ".join(ran) + " |")
    a("|---|" + "---|" * len(ran))
    yrs = sorted({int(y) for cid in ran for y in cells[cid]["LIQ"]["ENS"].get("ex_ante_book_beta_by_year", {})})
    for y in yrs:
        a(f"| {y} | " + " | ".join(_f(cells[cid]["LIQ"]["ENS"]["ex_ante_book_beta_by_year"].get(str(y),
                                     cells[cid]["LIQ"]["ENS"]["ex_ante_book_beta_by_year"].get(y))) for cid in ran) + " |")
    a("| mean | " + " | ".join(_f(cells[cid]["LIQ"]["ENS"].get("ex_ante_book_beta_mean")) for cid in ran) + " |")
    a("")
    # factor attribution
    a("## Per-year FF3 + MOM attribution of the daily gross L/S spread (LIQ ENS5, French daily factors, end 2019-01-01)")
    a("")
    a("Daily OLS of the gross long-minus-short spread on Mkt-RF, SMB, HML and MOM per calendar year (NW 5 lags). "
      "MOM contribution = MOM loading × the year's summed MOM return; compare with the year's summed spread. A large "
      "negative MOM contribution in 2009 is the momentum crash, not a window effect.")
    a("")
    for key, title, fmt in (("b_MOM", "MOM loading (t)", "bt"), ("contrib_MOM", "MOM contribution vs summed spread", "cs"),
                            ("alpha_ann", "FF4 alpha, annualised (t)", "at")):
        a(f"**{title}**")
        a("")
        a("| year | " + " | ".join(ran) + " |")
        a("|---|" + "---|" * len(ran))
        att = {cid: {r["year"]: r for r in cells[cid]["LIQ"].get("factor_attribution", [])} for cid in ran}
        yrs2 = sorted({y for cid in ran for y in att[cid]})
        for y in yrs2:
            row = [str(y)]
            for cid in ran:
                r = att[cid].get(y)
                if r is None:
                    row.append("")
                elif fmt == "bt":
                    row.append(f"{r['b_MOM']:+.2f} ({r['t_MOM']:+.1f})")
                elif fmt == "cs":
                    row.append(f"{_pc(r['contrib_MOM'], 1)} / {_pc(r['sum_spread'], 1)}")
                else:
                    row.append(f"{_pc(r['alpha_ann'], 1)} ({r['alpha_t']:+.1f})")
            a("| " + " | ".join(row) + " |")
        a("")
    # D7
    d7 = cells.get("D7", {})
    a("## D7: T5 index-hedged with its ex-ante book beta (diagnostic; hedge turnover charged at 5 bps)")
    a("")
    if d7.get("status") == "RUN":
        H = d7["LIQ"]["ENS_hedged"]
        yr = d7["LIQ"]["years_rule"]
        a("| book | gross SR (ann.) | net SR (ann.) | net SR 2× cost (ann.) | realised β | mean ex-ante book β | hedge "
          "turnover/yr | N1 p gross | N1 p net | N2 p | years ex-best (pos/n) | rule (info) |")
        a("|---|---|---|---|---|---|---|---|---|---|---|---|")
        a(f"| D7 hedged | {_sr(H['gross'])} | {_sr(H['net_ew'])} | {_sr(H['net_ew_2x'])} | {_f(H['net_ew']['beta'])} | "
          f"{_f(H['ex_ante_book_beta_mean'])} | {H['hedge_turnover_per_year']:.2f} | {_p(d7['LIQ']['N1']['gross']['p'])} | "
          f"{_p(d7['LIQ']['N1']['ew1']['p'])} | {_p(d7['LIQ']['N2']['ew1']['p'])} | {_pc(yr['mean_ex_best_year'])} "
          f"({yr['years_positive']}/{yr['years_counted']}) | {d7['LIQ']['verdict_info']} (diagnostic) |")
        if cells["T5"].get("status") == "RUN":
            E = cells["T5"]["LIQ"]["ENS"]
            a(f"| T5 unhedged | {_sr(E['gross'])} | {_sr(E['net_ew'])} | {_sr(E['net_ew_2x'])} | {_f(E['net_ew']['beta'])} | "
              f"{_f(E.get('ex_ante_book_beta_mean'))} | 0 | {_p(cells['T5']['LIQ']['N1']['gross']['p'])} | "
              f"{_p(cells['T5']['LIQ']['N1']['ew1']['p'])} | {_p(cells['T5']['LIQ']['N2']['ew1']['p'])} | | {cells['T5']['LIQ']['verdict']} |")
    else:
        a(f"NOT RUN: {d7.get('reason')}")
    a("")
    # validation-IC SE
    a("## Validation IC and its standard error per refit (tie-safe IC; SE = sd of the per-date validation ICs / √dates)")
    a("")
    a("NN3 (M3) is the only early-stopped model; its grid choice and stopping epoch are made on this IC. For the fixed "
      "boosters and the ridges the figure is reported only.")
    a("")
    a("| cell | model | per refit: validation IC (SE) |")
    a("|---|---|---|")
    for cid in g.RUN_ORDER:
        c = cells[cid]
        if c.get("status") != "RUN" or c["cell"].get("reuse_of"):
            continue
        for mname in ("M3", "M2", "M4"):
            rr = [r for r in c["refits"] if r["model"] == mname]
            a(f"| {cid} | {mname} | " + "; ".join(f"{r['refit_id']} {_f(r['val_ic'], 3)} ({(r.get('val_ic_se') or float('nan')):.3f})"
                                                for r in rr) + " |")
    a("")
    mins = [r.get("min_distinct_test") for cid in g.RUN_ORDER if cells[cid].get("status") == "RUN"
            for r in cells[cid]["refits"] if r.get("min_distinct_test") is not None]
    a(f"Non-degeneracy assertion (R2): every fit passed; the smallest number of distinct test predictions on any date "
      f"over all fits is {min(mins) if mins else 'n/a'}.")
    a("")
    # wall clock
    a("## Wall clock per cell")
    a("")
    a("| cell | status | elapsed s | fit s (new fits) | eval + nulls s | refits | load avg start → end | peak RSS main GB |")
    a("|---|---|---|---|---|---|---|---|")
    for cid in g.RUN_ORDER:
        c = cells[cid]
        if c.get("status") != "RUN":
            a(f"| {cid} | {c.get('status')} ({c.get('reason')}) | {c.get('elapsed_s', '')} | | | | | |")
            continue
        n_ref = len({r['refit_id'] for r in c["refits"]})
        la0 = c.get("load_avg_start", [float("nan")])[0]
        la1 = c.get("load_avg_end", [float("nan")])[0]
        a(f"| {cid} | RUN | {c['elapsed_s']:.0f} | {c['fit_seconds']:.0f} | {c.get('eval_seconds', 0):.0f} | {n_ref} | "
          f"{la0:.1f} → {la1:.1f} | {c.get('peak_rss_gb_main', float('nan')):.2f} |")
    a("")
    # model timings per refit class
    a("Mean new-fit seconds per refit by model and cell:")
    a("")
    a("| cell | " + " | ".join(g.GRID_MODELS) + " |")
    a("|---|" + "---|" * len(g.GRID_MODELS))
    for cid in g.RUN_ORDER:
        c = cells[cid]
        if c.get("status") != "RUN":
            continue
        row = [cid]
        for m in g.GRID_MODELS:
            v = [r["wall_seconds"] for r in c["refits"] if r["model"] == m and not r.get("cached")]
            row.append(f"{np.mean(v):.1f}" if v else "cache")
        a("| " + " | ".join(row) + " |")
    a("")
    # panels
    a("## Panels")
    a("")
    a("| horizon | rows | dates | first → last formation | day-0 rule | sha256(X) | sha256(y) |")
    a("|---|---|---|---|---|---|---|")
    for h, m in out["panel_meta"].items():
        a(f"| {H_NAME[h]} | {m['n_rows']:,} | {m['n_dates']} | {m['first_formation']} → {m['last_formation']} | "
          f"{m['day0_intraday_mode']} | `{m['sha256_input_matrix'][:16]}` | `{m['sha256_target'][:16]}` |")
    a("")
    mw = out["panel_meta"].get("W", {})
    if "weekly_vs_monthly_equality" in mw:
        eq = mw["weekly_vs_monthly_equality"]
        a(f"Weekly vs monthly feature equality on shared dates: {eq['shared_dates']} dates, {eq['rows_compared']:,} "
          f"firm-dates, same universe {eq['same_universe']}, match share {eq['match_share']:.6f}.")
        a("")
    for h, m in out["panel_meta"].items():
        r8 = m.get("r8_share_flag")
        if r8:
            a(f"R8 flag, {H_NAME[h]} panel: {r8['rows_flagged']:,} of {r8['rows_with_mcap']:,} rows with a market cap "
              f"({r8['share_of_rows_with_mcap']:.2%}); most-flagged symbols "
              + ", ".join(f"{k} {v}" for k, v in list(r8["top_symbols"].items())[:8]) + ".")
    a("")
    mm = out["panel_meta"].get("M", {})
    if mm:
        a(f"Coverage rule (monthly ALL 2010-2018): dropped tags {mm.get('dropped_tags')} → features "
          f"{mm.get('dropped_features')}. Inputs {len(mm.get('input_columns', []))}. Kept 8-K events with a price "
          f"symbol {mm['events']['kept_with_price_symbol']:,}; day 0 differs from the next-session rule for "
          f"{mm['events']['kept_events_day0_differs_from_next_session']:,} kept events. Sealed tripwire: "
          f"assert_sealed_clean() = {out.get('sealed_clean')}.")
        a("")
    OUT_MD.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def smoke_eval(panels: Panels) -> None:
    """Exercise the evaluation code on RANDOM predictions (no model output involved); prints only
    timings."""
    for h in ("M", "W", "Q"):
        spec = g.CellSpec(f"smoke{h}", h, "expanding", "annual", 1, "diagnostic")
        design = panels.design(spec)
        splits = g.grid_splits(h, "expanding", "annual", 1)[:2]
        rng = np.random.default_rng(0)
        preds = {m: [rng.normal(size=len(design.rows(sp.test))) for sp in splits] for m in g.GRID_MODELS}
        oos = OOS(design, splits, preds, g.ENS5)
        t0 = time.time()
        global N_N1, N_N2
        n1, n2 = N_N1, N_N2
        N_N1, N_N2 = 20, 50
        try:
            r = evaluate(oos, "LIQ", h, rng, aux=panels.aux())
            r2 = evaluate_hedged(oos, "LIQ", h, rng)
            assert len(r["factor_attribution"]) == 2 and "ENS_hedged" in r2
        finally:
            N_N1, N_N2 = n1, n2
        log(f"smoke {h}: evaluate() ok in {time.time() - t0:.1f} s, keys {sorted(r)[:5]}...")


def main() -> None:
    global N_N1, N_N2
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="", help="comma list of cell ids, 'trials', or 'all' (run order)")
    ap.add_argument("--diagnostics", action="store_true", help="also run D6 (needs the trials)")
    ap.add_argument("--quick", action="store_true", help="development: 2 refits, 2 seeds, no OOS metrics")
    ap.add_argument("--rebuild-panels", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--smoke-eval", action="store_true", help="evaluation code on random predictions")
    ap.add_argument("--force", action="store_true", help="re-run cells that already have results")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    if args.rebuild_panels or not all(panel_paths(h)["panel"].exists() for h in ("M", "Q", "W")):
        build_panels()
    if args.build_only:
        assert_sealed_clean()
        return
    panels = Panels()
    if args.smoke_eval:
        smoke_eval(panels)
        return
    if args.cells:
        if args.cells == "all":
            todo = list(g.RUN_ORDER)
        elif args.cells == "trials":
            todo = [c for c in g.RUN_ORDER if c.startswith("T")]
        else:
            todo = args.cells.split(",")
        for cid in todo:
            spec = g.ALL_CELLS[cid]
            path = CELLS_DIR / f"{cid}.json"
            if path.exists() and not args.force and not args.quick:
                log(f"[{cid}] already done, skipped")
                continue
            log(f"[{cid}] start: {spec.note or cell_label(spec.as_dict())}; load avg {os.getloadavg()[0]:.1f}")
            res = run_cell(spec, panels, args.quick)
            if args.quick:
                log(f"[{cid}] quick done in {res['elapsed_s']} s (no OOS metrics)")
                continue
            path.write_text(json.dumps(res, default=float))
    if args.diagnostics and not args.quick:
        d6 = run_d6(panels)
        (CELLS_DIR / "D6.json").write_text(json.dumps(d6, default=float))
    if args.report_only or (args.cells and not args.quick) or args.diagnostics:
        out = assemble()
        assert_sealed_clean()
        out["sealed_clean"] = True
        OUT_JSON.write_text(json.dumps(out, indent=1, default=float))
        write_markdown(out)
        log(f"wrote {OUT_JSON} + {OUT_MD}; study verdict {out['verdict']['study_verdict']}")


if __name__ == "__main__":
    main()
