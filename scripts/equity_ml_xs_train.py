#!/usr/bin/env python3
"""Study 13: machine-learning cross-sectional return model on the free US equity panel --
equities TRAIN 2004-01-01 -> 2019-01-01 only.

Pre-declared design (study 13). Monthly panel of common
stocks (ALL = close >= $5 and 63-day median dollar volume >= $1M; LIQ = its top half by dollar
volume), 29 PIT features + SIC, rank target of the next-month excess return; models M0 ridge-3,
M1 ridge-all, M2 XGBoost, M3 GKX NN3, M4 XGBoost rank:pairwise, M5 cross-sectional transformer,
M6 ensemble of M2-M5; OOS 2009-2018 with yearly expanding refits (1-month embargo, 12-month
validation slice). Headline cells ENS_LIQ / ENS_ALL with N1 (prediction permutation) and N2
(3-month block bootstrap) nulls.

All data come through ``eqxs.equities_xs`` loaders with ``end="2019-01-01"``. The derived
cache lives under ``data_cache/equities_xs/derived/`` and is checked to hold nothing dated 2019+.

    python scripts/equity_ml_xs_train.py --quick      # development: 2 OOS years, 2 seeds
    python scripts/equity_ml_xs_train.py              # full study
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from eqxs.data_zones import assert_sealed_clean, safe_load_parquet  # noqa: E402
from eqxs import equities_xs as xs  # noqa: E402
from eqxs import equity_events as ee  # noqa: E402
from eqxs import equity_ml as ml  # noqa: E402

TRAIN_START = pd.Timestamp("2004-01-01")
TRAIN_END = pd.Timestamp("2019-01-01")
WARMUP_YEAR = 2002
FUND_START = pd.Timestamp("2009-01-01")
DERIVED = xs.XS_ROOT / "derived"
PANEL_PATH = DERIVED / "ml_panel_train.parquet"
MONTHS_PATH = DERIVED / "ml_months_train.parquet"
META_PATH = DERIVED / "ml_panel_train.meta.json"
SCHEMA_PATH = DERIVED / "ml_panel_train.schema.md"
PRED_PATH = DERIVED / "eqml_oos_predictions_train.parquet"
PORT_PATH = DERIVED / "eqml_oos_portfolios_train.parquet"
OUT_JSON = REPO / "results" / "equity_ml_xs_train.json"
OUT_MD = REPO / "results" / "equity_ml_xs_train.md"
QUICK_JSON = REPO / "results" / "equity_ml_xs_train_quick.json"
MARKER = "<!-- AUTO-RESULTS-BELOW: everything after this marker is (re)written by the script -->"
SEED = 20261013
COST = {"ALL": 0.0015, "LIQ": 0.0005}
OOS_YEARS = list(range(2009, 2019))
BASE_MODELS = ["M0", "M1", "M2", "M3", "M4", "M5"]
ENS_COMPONENTS = ["M2", "M3", "M4", "M5"]
M5_BUDGET_S = 7200.0
N_N1 = 500
N_N2 = 1000
COVERAGE_MIN = 0.30
SURV_NOTE = ("survivor-only price panel (yfinance: stocks listed on 2026-10-09; no delisted firms, "
             "no delisting returns)")
RUN_VERSION = "eqml-v2"  # v2: PIT-legal clean event set (red team F1)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def assert_pre_2019(df: pd.DataFrame, what: str) -> None:
    """Every datetime column (and ``ts``) must be strictly before 2019-01-01."""
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            t = pd.to_datetime(s, utc=True) if getattr(s.dt, "tz", None) is not None else pd.to_datetime(s)
            bound = TRAIN_END.tz_localize("UTC") if getattr(t.dt, "tz", None) is not None else TRAIN_END
            mx = t.max()
            assert pd.isna(mx) or mx < bound, f"{what}: column {c} has a date >= 2019-01-01 ({mx})"


# ---------------------------------------------------------------------------
# Panel construction
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
        for c in ("close", "adj_close", "volume", "splits"):
            px[c] = px[c].astype(np.float32) if c != "volume" else px[c].astype(np.float64)
        parts.append(px)
        log(f"prices {y}: {len(px):,} bars")
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
                                 px["adj_close"].to_numpy(np.float64), px["volume"].to_numpy(np.float64),
                                 px["splits"].to_numpy(np.float64))
    log(f"price matrices: {pm.n_rows} sessions x {pm.n_cols} symbols, {len(px):,} bars, "
        f"{pm.n_guarded} guarded returns")
    return pm, yf_of


def load_ff(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    try:
        ff = xs.load_ff_daily(end=TRAIN_END)   # loader with start/end (red team F13)
    except TypeError:
        ff = xs.load_ff_daily()                # older loader: single pre-2022 file, cut below
    idx = pd.DatetimeIndex(ff.index)
    ff.index = idx.tz_localize(None) if idx.tz is not None else idx
    ff = ff.loc[ff.index < TRAIN_END].reindex(sessions)   # 2019-2021 rows dropped at once
    assert ff["Mkt-RF"].notna().all(), "FF must cover the trading calendar"
    return ff


def load_events(pm: ml.PriceMatrices, cik_map: dict[int, str], col_of: dict[str, int],
                mkt: np.ndarray) -> tuple[pd.DataFrame, dict]:
    ev = xs.load_earnings_all(end=TRAIN_END)
    assert pd.to_datetime(ev["ts"], utc=True).max() < TRAIN_END.tz_localize("UTC")
    kept = ev[ev["keep"]].copy()
    # PIT clean set: forward-looking flags (multi_202_in_quarter, cadence_prefers) are ignored
    from eqxs.earnings_calendar import clean_event_mask

    kept["clean"] = clean_event_mask(kept["ambiguous"]).to_numpy(bool)   # shared PIT helper (== ml.pit_clean)
    kept["clean_study11"] = ~(kept["ambiguous"].fillna("").astype(str).str.len() > 0)
    kept["symbol"] = kept["cik"].astype(int).map(cik_map)
    kept["col"] = kept["symbol"].map(col_of).fillna(-1).astype(int)
    # pre-declared day-0 rule = study 11's committed 09:30 rule (INTRADAY -> next session); pinned
    # explicitly because later versions of equity_events default INTRADAY to the acceptance session
    try:
        kept["day0"] = ee.day0_from_acceptance(kept["acceptance_et"], pm.sessions, intraday="next_session")
    except TypeError:
        kept["day0"] = ee.day0_from_acceptance(kept["acceptance_et"], pm.sessions)
    kept = kept[kept["day0"].notna() & (kept["col"] >= 0)].copy()
    kept["day0_row"] = pm.sessions.get_indexer(pd.DatetimeIndex(kept["day0"]))
    kept = kept[kept["day0_row"] >= 0]
    kept = kept.sort_values(["col", "day0"]).drop_duplicates(["col", "day0"])
    kept["signal"] = ml.announcement_signal(kept["day0_row"].to_numpy(), kept["col"].to_numpy(), pm.R, mkt)
    diag = {"raw_rows": int(len(ev)), "kept_with_price_symbol": int(len(kept)),
            "day0_intraday_mode": "next_session (study-11 09:30 rule, pinned)",
            "clean_share": float(kept["clean"].mean()),
            "clean_share_study11_rule": float(kept["clean_study11"].mean())}
    return kept.loc[:, ["cik", "col", "day0", "day0_row", "clean", "signal"]].reset_index(drop=True), diag


def load_fundamentals(ciks: set[int]) -> tuple[dict[str, pd.DataFrame], dict]:
    """XBRL facts of the study tags, one availability quarter at a time (2008Q1 .. 2018Q4)."""
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
                f = f[f["uom"].isin(["USD", "shares", "USD/shares"])]
                parts.append(f)
            del num
        log(f"fundamentals: availability year {y} read ({n_raw:,} raw rows so far)")
    f = pd.concat(parts, ignore_index=True)
    del parts
    assert f["ts"].max() < TRAIN_END.tz_localize("UTC").value
    out = ml.group_facts(f)
    diag = {"raw_rows": int(n_raw), "prepared_rows": int(len(f)),
            "rows_by_group": {g: int(len(d)) for g, d in out.items()}}
    return out, diag


def filer_counts() -> pd.DataFrame:
    sub = xs._load_partitioned(xs.XS_ROOT / "sec_fsds" / "sub", "year", intent="eqml.filer_counts",
                               columns=["cik", "form", "ts"], end=TRAIN_END)
    sub = sub[sub["form"].isin(["10-K", "10-Q", "10-K/A", "10-Q/A"])]
    sub["year"] = pd.to_datetime(sub["ts"], utc=True).dt.year
    return sub.groupby("year")["cik"].nunique().rename("sec_10k_10q_filers").to_frame()


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    t0 = time.time()
    pm, yf_of = load_prices()
    col_of = {s: i for i, s in enumerate(pm.symbols)}
    me = ml.month_end_rows(pm.sessions)
    uni = ml.build_universe(pm, me)
    mkt_ew = ml.ew_market_return(pm.R, me, uni.all_)
    ff = load_ff(pm.sessions)
    mkt_ff = (ff["Mkt-RF"] + ff["RF"]).to_numpy()
    hret, entry, exit_ = ml.holding_returns(pm.R, me)
    me_dates = pm.sessions[me]
    pos = np.flatnonzero((me_dates >= TRAIN_START) & (me_dates < TRAIN_END))
    log(f"month-ends: {len(me)} total, {len(pos)} formation dates {me_dates[pos[0]].date()} .. "
        f"{me_dates[pos[-1]].date()}")
    pf = ml.price_features(pm, me, mkt_ew, pos)
    log("price features done")
    bars = np.isfinite(pm.adj).sum(axis=0)
    cmap, map_diag = ml.cik_symbol_map(xs.load_static("cik_ticker_map"), yf_of, col_of, bars)
    events, ev_diag = load_events(pm, cmap, col_of, mkt_ff)
    efe = ml.event_features(events, pm.sessions, me, pm.n_cols, pos)
    log(f"event features done ({len(events):,} kept events with a price symbol)")
    # long panel of ALL-universe rows
    rows = []
    sym_cik = {s: c for c, s in cmap.items()}
    for k, p in enumerate(pos):
        cols = np.flatnonzero(uni.all_[p])
        d = {"pos": p, "k": k, "firm": cols}
        rows.append(pd.DataFrame(d))
    panel = pd.concat(rows, ignore_index=True)
    k, f = panel["k"].to_numpy(), panel["firm"].to_numpy()
    p = panel["pos"].to_numpy()
    panel["date"] = me_dates[p]
    panel["ts"] = panel["date"].dt.tz_localize("UTC")
    panel["date_id"] = k
    panel["mi"] = ml.month_index(panel["date"])
    panel["symbol"] = pm.symbols[f]
    panel["cik"] = pd.array([sym_cik.get(s) for s in panel["symbol"]], dtype="Int64")
    panel["in_liq"] = uni.liq[p, f]
    ok_t = entry[p] >= 0
    panel["entry_date"] = pd.NaT
    panel["exit_date"] = pd.NaT
    panel.loc[ok_t, "entry_date"] = pm.sessions[entry[p[ok_t]]]
    panel.loc[ok_t, "exit_date"] = pm.sessions[exit_[p[ok_t]]]
    panel["target_mi"] = np.where(ok_t, ml.month_index(panel["entry_date"].fillna(TRAIN_START)), -1)
    panel["y_raw"] = np.where(ok_t, hret[p, f], np.nan)
    for name, arr in pf.items():
        panel[f"raw_{name}"] = arr[k, f]
    for name, arr in efe.items():
        panel[f"raw_{name}"] = arr[k, f]
    panel["size_log_dv"] = panel["raw_log_dv_63"]
    # fundamentals
    have = panel["cik"].notna().to_numpy()
    fund_ciks = set(int(c) for c in panel.loc[have, "cik"].unique())
    facts, fund_diag = load_fundamentals(fund_ciks)
    sue_ev = ml.sue_events(facts.get("eps", pd.DataFrame(columns=["cik"])))
    fund_diag["sue_events"] = int(len(sue_ev))
    sub = panel.loc[have]
    q = ml.make_fund_query(pm, me, sub["pos"].to_numpy(), sub["firm"].to_numpy(), sub["cik"].to_numpy(np.int64))
    ff_feats, present = ml.fundamental_features(facts, q, sue_ev=sue_ev)
    del facts
    pre = (sub["date"] < FUND_START).to_numpy()
    for name in ml.FUND_FEATURES:
        v = np.full(len(panel), np.nan)
        v[have] = np.where(pre, np.nan, ff_feats[name])
        panel[f"raw_{name}"] = v
    pres = {}
    for g, arr in present.items():
        v = np.zeros(len(panel), dtype=bool)
        v[have] = arr & ~pre
        pres[g] = v
    log("fundamental features done")
    # SIC
    prof = xs.load_static("filer_profiles")
    sic_of = dict(zip(prof["cik"].astype(int), prof["sic"].astype(str), strict=False))
    div = ml.sic_division(pd.Series([sic_of.get(int(c), "") if pd.notna(c) else "" for c in panel["cik"]]))
    for c in ml.SIC_COLS:
        panel[c] = (div.to_numpy() == c.split("_", 1)[1]).astype(np.float32)
    # targets
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
    # coverage rule (ALL firm-months 2010-01 .. 2018-12)
    m10 = (panel["date"] >= pd.Timestamp("2010-01-01")).to_numpy()
    tag_cov = {g: float(pres[g][m10].mean()) for g in ml.TAG_GROUPS}
    dropped_tags = sorted(g for g, c in tag_cov.items() if c < COVERAGE_MIN)
    dropped_feats = sorted(f for f, tags in ml.FEATURE_TAGS.items() if set(tags) & set(dropped_tags))
    log(f"tag coverage 2010+: {json.dumps({g: round(c, 3) for g, c in tag_cov.items()})}; dropped tags "
        f"{dropped_tags} -> features {dropped_feats}")
    # normalisation within date over ALL
    for name in ml.NUMERIC_FEATURES:
        z, miss = ml.normalize_within_date(panel[f"raw_{name}"].to_numpy(), b)
        panel[name] = z
        panel[f"miss_{name}"] = miss
    # month table (market over each holding window)
    mrows = []
    for kk, pp in enumerate(pos):
        if entry[pp] < 0:
            continue
        e, x = entry[pp], exit_[pp]
        mk_tot = float(np.prod(1 + mkt_ff[e + 1:x + 1]) - 1)
        rf = float(np.prod(1 + ff["RF"].to_numpy()[e + 1:x + 1]) - 1)
        mrows.append({"date_id": kk, "date": me_dates[pp], "entry_date": pm.sessions[e],
                      "exit_date": pm.sessions[x], "target_mi": int(ml.month_index([pm.sessions[e]])[0]),
                      "mkt": mk_tot, "rf": rf, "mktrf": mk_tot - rf})
    months = pd.DataFrame(mrows)
    months["ts"] = pd.to_datetime(months["date"]).dt.tz_localize("UTC")
    # coverage per year vs SEC filers
    fc = filer_counts()
    per_year = panel.groupby(panel["date"].dt.year).agg(
        mean_all=("firm", lambda s: len(s) / 12.0), mean_liq=("in_liq", lambda s: s.sum() / 12.0),
        share_with_cik=("cik", lambda s: float(s.notna().mean())))
    per_year = per_year.join(fc, how="left")
    feat_cov = panel.groupby(panel["date"].dt.year)[[f"raw_{n}" for n in ml.NUMERIC_FEATURES]].agg(
        lambda s: float(s.notna().mean()))
    feat_cov.columns = list(ml.NUMERIC_FEATURES)
    meta = {"version": RUN_VERSION, "built_utc": pd.Timestamp.now("UTC").isoformat(),
            "n_rows": int(len(panel)), "n_dates": int(panel["date_id"].nunique()),
            "n_symbols_matrix": int(pm.n_cols), "n_sessions": int(pm.n_rows), "guarded_returns": pm.n_guarded,
            "mapping": map_diag, "events": ev_diag, "fundamentals": fund_diag,
            "tag_coverage_2010": tag_cov, "dropped_tags": dropped_tags, "dropped_features": dropped_feats,
            "coverage_per_year": {int(yv): {kk: (None if pd.isna(vv) else float(vv)) for kk, vv in r.items()}
                                  for yv, r in per_year.iterrows()},
            "feature_coverage_per_year": {int(yv): {kk: float(vv) for kk, vv in r.items()}
                                          for yv, r in feat_cov.iterrows()},
            "build_seconds": round(time.time() - t0, 1)}
    panel = panel.drop(columns=["pos", "k"])
    for c in ("entry_date", "exit_date"):
        panel[c] = pd.to_datetime(panel[c])
    assert_pre_2019(panel, "ml_panel_train")
    assert_pre_2019(months, "ml_months_train")
    num_cols = [c for c in panel.columns if c.startswith(("raw_", "miss_", "sic_")) or c in ml.NUMERIC_FEATURES]
    panel[num_cols] = panel[num_cols].astype(np.float32)
    for c in ("y_raw", "y_ex_all", "y_ex_liq", "y_rank", "size_log_dv"):
        panel[c] = panel[c].astype(np.float32)
    DERIVED.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    months.to_parquet(MONTHS_PATH, index=False)
    META_PATH.write_text(json.dumps(meta, indent=1, default=str))
    write_schema(panel)
    log(f"panel: {len(panel):,} firm-months, {meta['n_dates']} dates -> {PANEL_PATH} "
        f"({meta['build_seconds']} s)")
    return panel, months, meta


def write_schema(panel: pd.DataFrame) -> None:
    lines = ["# ml_panel_train.parquet (study 13, eqml) -- schema note", "",
             "One row per (month-end formation date T, firm) in the ALL universe, TRAIN formation dates",
             "2004-01 .. 2018-12; nothing dated 2019-01-01 or later (asserted on write and on read).",
             f"Producer: scripts/equity_ml_xs_train.py ({RUN_VERSION}). {SURV_NOTE}.", "",
             "* `date` T (last session of the month), `ts` = T as UTC (for the data-zone tripwire), `date_id`,",
             "  `mi` month index (year*12+month-1) of T, `target_mi` month index of the entry session (-1 = no target).",
             "* `symbol` (yfinance), `firm` (column id), `cik` (Int64, study-11 map), `in_liq` (top half by 63-day",
             "  median dollar volume within ALL).",
             "* `entry_date` / `exit_date`: first session after T / first session after the next month-end.",
             "* `y_raw` holding return (adj_close, missing daily returns = 0), `y_ex_all` / `y_ex_liq` minus the",
             "  universe EW mean, `y_rank` within-date rank of `y_raw` over ALL in [-0.5, 0.5] (training target).",
             "* `raw_<feature>` the PIT feature value; `<feature>` winsorised (1/99) + ranked within date to",
             "  [-0.5, 0.5] with missing -> 0; `miss_<feature>` 0/1; `sic_*` one-hot SIC division (A-J, unknown).",
             "* `size_log_dv` = raw log 63-day median dollar volume (size-tercile diagnostic).", "",
             "| column | dtype |", "|---|---|"]
    lines += [f"| {c} | {t} |" for c, t in panel.dtypes.astype(str).items()]
    SCHEMA_PATH.write_text("\n".join(lines) + "\n")


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    panel = safe_load_parquet(PANEL_PATH, max_end_date=TRAIN_END, caller_intent="eqml.load_panel")
    months = safe_load_parquet(MONTHS_PATH, max_end_date=TRAIN_END, caller_intent="eqml.load_months")
    assert_pre_2019(panel, "ml_panel_train")
    assert_pre_2019(months, "ml_months_train")
    meta = json.loads(META_PATH.read_text())
    return panel, months, meta


# ---------------------------------------------------------------------------
# Model runs (one refit per OOS year; each result cached so a crash resumes)
# ---------------------------------------------------------------------------
class Design:
    """Arrays shared by every model: inputs, target, splits, OOS test rows."""

    def __init__(self, panel: pd.DataFrame, numeric: list[str], years: list[int]) -> None:
        self.numeric = numeric
        self.columns = ml.input_columns(numeric)
        self.X = panel[self.columns].to_numpy(np.float32)
        self.y = panel["y_rank"].to_numpy(np.float64)
        self.tmi = panel["target_mi"].to_numpy()
        self.date_id = panel["date_id"].to_numpy()
        valid = self.tmi >= 0
        self.splits = ml.expanding_splits(years, int(self.tmi[valid].min()))
        self.groups = ml.feature_groups(numeric)
        self.data_hash = hashlib.sha256(self.X.tobytes() + self.y.tobytes()).hexdigest()

    def rows(self, rng_: tuple[int, int]) -> np.ndarray:
        return np.flatnonzero(ml.mask_months(self.tmi, rng_) & (self.tmi >= 0))

    def bounds(self, idx: np.ndarray) -> np.ndarray:
        return ml.date_bounds(self.date_id[idx])


def run_fingerprint(variant: str, name: str, design: Design, quick: bool) -> str:
    h = hashlib.sha256(json.dumps([RUN_VERSION, variant, name, design.columns, quick]).encode())
    h.update(design.data_hash.encode())
    return h.hexdigest()[:16]


def runs_dir(quick: bool) -> Path:
    d = DERIVED / ("eqml_runs_quick" if quick else "eqml_runs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def permuted_inputs(design: Design, test_idx: np.ndarray, year: int) -> dict[str, np.ndarray]:
    """For each feature group: test inputs with the group permuted within each date (seeded by
    year and group index, identical for every model)."""
    Xt = design.X[test_idx]
    bt = design.bounds(test_idx)
    out = {}
    for gi, (g, cols) in enumerate(design.groups.items()):
        ci = [design.columns.index(c) for c in cols]
        rng = np.random.default_rng([SEED, year, gi])
        Xp = Xt.copy()
        for a, b in zip(bt[:-1], bt[1:], strict=True):
            perm = rng.permutation(b - a)
            Xp[a:b][:, ci] = Xt[a:b][perm][:, ci]
        out[g] = Xp
    return out


def fit_one(design: Design, variant: str, name: str, split: ml.Split, quick: bool,
            with_perm: bool) -> dict:
    path = runs_dir(quick) / f"{variant}_{name}_{split.year}.npz"
    meta_path = path.with_suffix(".json")
    fp = run_fingerprint(variant, name, design, quick)
    if path.exists() and meta_path.exists():
        info = json.loads(meta_path.read_text())
        if info.get("fingerprint") == fp:
            z = np.load(path)
            info["pred_test"], info["pred_val"] = z["pred_test"], z["pred_val"]
            info["perm"] = {k[5:]: z[k] for k in z.files if k.startswith("perm_")}
            return info
    fi, vi, ti = design.rows(split.fit), design.rows(split.val), design.rows(split.test)
    fd = ml.FitData(design.X[fi], design.y[fi], design.bounds(fi), design.X[vi], design.y[vi],
                    design.bounds(vi), design.columns)
    model = ml.make_model(name, quick=quick)
    t0 = time.time()
    info = model.fit(fd, seed=SEED + split.year)
    bt = design.bounds(ti)
    pred_test = model.predict(design.X[ti], bt)
    pred_val = model.predict(fd.Xv, fd.bv)
    perm = {}
    if with_perm:
        pin = permuted_inputs(design, ti, split.year)
        if hasattr(model, "predict_many"):
            outs = model.predict_many(list(pin.values()), bt)
            perm = {g: v.astype(np.float32) for g, v in zip(pin, outs, strict=True)}
        else:
            perm = {g: model.predict(Xp, bt).astype(np.float32) for g, Xp in pin.items()}
        del pin
    if hasattr(model, "close"):
        model.close()
    info.update({"fingerprint": fp, "model": name, "variant": variant, "year": split.year,
                 "wall_seconds": time.time() - t0, "n_fit": int(len(fi)), "n_val": int(len(vi)),
                 "n_test": int(len(ti)), "fit_months": list(split.fit), "val_months": list(split.val),
                 "test_months": list(split.test), "embargoed_months": list(split.embargoed)})
    np.savez_compressed(path, pred_test=pred_test.astype(np.float64), pred_val=pred_val.astype(np.float64),
                        **{f"perm_{g}": v for g, v in perm.items()})
    meta_path.write_text(json.dumps(info, default=float))
    info["pred_test"], info["pred_val"], info["perm"] = pred_test, pred_val, perm
    return info


def run_models(design: Design, variant: str, models: list[str], quick: bool) -> dict[str, dict]:
    """Model-major loop. Returns {model: {"years": {year: info}, "dropped": reason|None,
    "seconds": total}}."""
    out: dict[str, dict] = {}
    for name in models:
        if name == "M5" and not ml.torch_available():
            out[name] = {"years": {}, "dropped": "torch not installed", "seconds": 0.0}
            continue
        res: dict[int, dict] = {}
        total = 0.0
        dropped = None
        for sp in design.splits:
            if name == "M5" and total > M5_BUDGET_S:
                dropped = f"wall clock {total:.0f} s exceeded the {M5_BUDGET_S:.0f} s budget before year {sp.year}"
                break
            try:
                info = fit_one(design, variant, name, sp, quick, with_perm=(variant == "main"
                                                                            and name in ENS_COMPONENTS))
            except FloatingPointError as exc:
                dropped = f"unstable: {exc}"
                break
            res[sp.year] = info
            total += float(info.get("wall_seconds", info.get("seconds", 0.0)))
            msg = {k: info[k] for k in ("alpha", "params", "lam", "lr", "best_iteration", "n_seeds")
                   if k in info}
            log(f"[{variant}] {name} {sp.year}: val IC {info['val_ic']:+.4f}, {info['wall_seconds']:.0f} s, "
                f"{json.dumps(msg, default=float)}")
        if name == "M5" and dropped is None and total > M5_BUDGET_S:
            dropped = f"wall clock {total:.0f} s across refits exceeded the {M5_BUDGET_S:.0f} s budget"
        out[name] = {"years": res, "dropped": dropped, "seconds": total}
        if dropped:
            log(f"[{variant}] {name} DROPPED: {dropped}")
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
class OOS:
    """OOS rows (concatenated years, sorted by date) with every model's predictions."""

    def __init__(self, panel: pd.DataFrame, design: Design, runs: dict[str, dict],
                 components: list[str]) -> None:
        self.design = design
        years = sorted({y for r in runs.values() for y in r["years"]})
        idx = np.concatenate([design.rows(sp.test) for sp in design.splits if sp.year in years])
        self.idx = idx
        self.p = panel.iloc[idx].reset_index(drop=True)
        self.bounds = ml.date_bounds(self.p["date_id"].to_numpy())
        self.pred: dict[str, np.ndarray] = {}
        self.val: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        val_rows = {sp.year: design.rows(sp.val) for sp in design.splits}
        self.val_rows = val_rows
        for name, r in runs.items():
            if r["dropped"] or len(r["years"]) != len(years):
                continue
            self.pred[name] = np.concatenate([r["years"][y]["pred_test"] for y in years])
            self.val[name] = {y: (val_rows[y], r["years"][y]["pred_val"]) for y in years}
        self.components = [c for c in components if c in self.pred]
        if self.components:
            self.pred["M6"] = ml.ensemble([self.pred[c] for c in self.components], self.bounds)
            self.val["M6"] = {}
            for y in years:
                vr = val_rows[y]
                bv = design.bounds(vr)
                self.val["M6"][y] = (vr, ml.ensemble([self.val[c][y][1] for c in self.components], bv))
        self.years = years
        self.y_ex_all_full = panel["y_ex_all"].to_numpy(np.float64)
        self.year_of_row = self.p["target_mi"].to_numpy() // 12

    def subset(self, universe: str) -> np.ndarray:
        return np.ones(len(self.p), bool) if universe == "ALL" else self.p["in_liq"].to_numpy(bool)


def legs_block(res: ml.LSResult, months: pd.DataFrame) -> dict:
    mk = months["mkt"].to_numpy()
    return {"long_ann": float(res.long_ret.mean() * 12), "short_ann": float(res.short_ret.mean() * 12),
            "long_minus_mkt_ann": float((res.long_ret - mk).mean() * 12),
            "short_minus_mkt_ann": float((res.short_ret - mk).mean() * 12),
            "mkt_ann": float(mk.mean() * 12),
            "turnover_long_month": float(res.to_long[1:].mean()), "turnover_short_month": float(res.to_short[1:].mean()),
            "cost_drag_ann": float((res.gross - res.net).mean() * 12),
            "mean_n_long": float(res.n_long.mean()), "mean_n_short": float(res.n_short.mean())}


def evaluate_cell(oos: OOS, months: pd.DataFrame, name: str, universe: str, pred: np.ndarray | None = None,
                  mask: np.ndarray | None = None) -> tuple[dict, ml.LSResult, np.ndarray]:
    """Metrics of one model x universe (optionally an extra row mask, e.g. a size tercile)."""
    m = oos.subset(universe) if mask is None else mask
    p = (oos.pred[name] if pred is None else pred)[m]
    sub = oos.p.loc[m]
    b = ml.date_bounds(sub["date_id"].to_numpy())
    yex = sub["y_ex_all" if universe == "ALL" else "y_ex_liq"].to_numpy(np.float64)
    if mask is not None:  # tercile: excess vs the tercile mean
        yex = sub["y_raw"].to_numpy(np.float64).copy()
        for a, bb in zip(b[:-1], b[1:], strict=True):
            yex[a:bb] -= yex[a:bb].mean()
    ics = ml.ic_by_date(p, yex, b)
    res = ml.decile_long_short(p, sub["y_raw"].to_numpy(np.float64), sub["firm"].to_numpy(), b,
                               COST[universe], int(oos.p["firm"].max()) + 1)
    did = sub["date_id"].to_numpy()[b[:-1]]
    mo = months.set_index("date_id").loc[did].reset_index()
    tmi = mo["target_mi"].to_numpy()
    out = {"model": name, "universe": universe, "cost_per_side_bps": COST[universe] * 1e4,
           "survivorship_note": SURV_NOTE, "ic": ml.ic_summary(ics),
           "net": ml.monthly_stats(res.net, tmi, mo["mktrf"].to_numpy()),
           "gross": ml.monthly_stats(res.gross, tmi, mo["mktrf"].to_numpy()),
           "legs": legs_block(res, mo), "months": int(len(res.net)),
           "first_month": str(pd.Timestamp(mo["entry_date"].iloc[0]).date()),
           "last_month": str(pd.Timestamp(mo["entry_date"].iloc[-1]).date())}
    if pred is None and mask is None and name in oos.val:
        # GKX OOS R^2: per-year calibration slope on the validation slice
        yr = oos.year_of_row[m]
        f = np.full(len(p), np.nan)
        for y in oos.years:
            vr, pv = oos.val[name][y]
            r_val = oos.y_ex_all_full[vr]
            slope = ml.calibration_slope(pv, r_val)
            f[yr == y] = slope * p[yr == y]
        out["oos_r2"] = ml.oos_r2(f, yex)
    return out, res, ics


def corr_matrix(series: dict[str, np.ndarray]) -> dict:
    df = pd.DataFrame(series)
    return df.corr().round(4).to_dict()


def permutation_importance(oos: OOS, runs: dict[str, dict]) -> list[dict]:
    """Ensemble permutation importance on the OOS years (ALL universe IC)."""
    comps = oos.components
    if not comps:
        return []
    yex = oos.p["y_ex_all"].to_numpy(np.float64)
    base = ml.ic_by_date(oos.pred["M6"], yex, oos.bounds)
    groups = list(oos.design.groups)
    out = []
    for g in groups:
        perm_pred = np.concatenate([
            ml.ensemble([runs[c]["years"][y]["perm"][g] for c in comps],
                        oos.design.bounds(oos.design.rows(next(sp.test for sp in oos.design.splits
                                                               if sp.year == y))))
            for y in oos.years])
        ics = ml.ic_by_date(perm_pred, yex, oos.bounds)
        d = base - ics
        out.append({"group": g, "importance_ic": float(np.nanmean(d)),
                    "t": float(np.nanmean(d) / (np.nanstd(d, ddof=1) / np.sqrt(np.isfinite(d).sum())))
                    if np.nanstd(d) > 0 else float("nan"),
                    "share_months_positive": float(np.nanmean(d > 0))})
    return sorted(out, key=lambda r: -r["importance_ic"])


def single_feature_ic(oos: OOS) -> list[dict]:
    out = []
    for f in oos.design.numeric:
        row = {"feature": f}
        for u in ("ALL", "LIQ"):
            m = oos.subset(u)
            sub = oos.p.loc[m]
            b = ml.date_bounds(sub["date_id"].to_numpy())
            ics = ml.ic_by_date(sub[f].to_numpy(np.float64),
                                sub["y_ex_all" if u == "ALL" else "y_ex_liq"].to_numpy(np.float64), b)
            s = ml.ic_summary(ics)
            row[f"{u}_ic"], row[f"{u}_t"] = s["ic_mean"], s["ic_t"]
        out.append(row)
    return sorted(out, key=lambda r: -abs(r["ALL_ic"]))


def size_terciles(oos: OOS, months: pd.DataFrame, name: str = "M6") -> list[dict]:
    out = []
    size = oos.p["size_log_dv"].to_numpy(np.float64)
    for u in ("ALL", "LIQ"):
        um = oos.subset(u)
        terc = np.zeros(len(oos.p), dtype=int)
        for a, b in zip(oos.bounds[:-1], oos.bounds[1:], strict=True):
            idx = np.arange(a, b)[um[a:b]]
            r = pd.Series(size[idx]).rank(pct=True).to_numpy()
            terc[idx] = np.ceil(r * 3).astype(int)
        for t in (1, 2, 3):
            res, _, _ = evaluate_cell(oos, months, name, u, mask=um & (terc == t))
            out.append({"universe": u, "tercile": t, "label": {1: "small", 2: "mid", 3: "large"}[t],
                        "ic_mean": res["ic"]["ic_mean"], "ic_t": res["ic"]["ic_t"],
                        "gross_sharpe": res["gross"]["sharpe"], "gross_ann": res["gross"]["ann_return"],
                        "net_sharpe": res["net"]["sharpe"], "net_ann": res["net"]["ann_return"],
                        "max_dd": res["net"]["max_dd"], "turnover_long": res["legs"]["turnover_long_month"],
                        "mean_n_long": res["legs"]["mean_n_long"],
                        "years_positive": res["net"]["years_positive"]})
    return out


def headline(oos: OOS, months: pd.DataFrame, universe: str, rng: np.random.Generator) -> dict:
    cell, res, _ = evaluate_cell(oos, months, "M6", universe)
    m = oos.subset(universe)
    sub = oos.p.loc[m]
    b = ml.date_bounds(sub["date_id"].to_numpy())
    t0 = time.time()
    n1 = ml.n1_permutation(oos.pred["M6"][m], sub["y_raw"].to_numpy(np.float64), sub["firm"].to_numpy(), b,
                           COST[universe], int(oos.p["firm"].max()) + 1, cell["net"]["sharpe"],
                           n_draws=N_N1, rng=rng, progress=lambda i: log(f"  ENS_{universe} N1 {i}/{N_N1}"))
    n2 = ml.n2_block_bootstrap(res.net, block=3, n_draws=N_N2, rng=rng)
    fails = []
    if not cell["net"]["sharpe"] > 0:
        fails.append("net_sharpe<=0")
    if not n1["p"] < 0.05:
        fails.append("N1_p>=0.05")
    if not n2["p"] < 0.05:
        fails.append("N2_p>=0.05")
    if not cell["net"]["pos_year_share"] >= 0.75:
        fails.append("pos_years<75%")
    cell.update({"N1": n1, "N2": n2, "fails": fails, "verdict": "PASS" if not fails else "REJECT",
                 "null_seconds": round(time.time() - t0, 1), "components": oos.components})
    log(f"ENS_{universe}: net SR {cell['net']['sharpe']:+.2f} ({cell['net']['ann_return'] * 100:+.2f}%/yr), "
        f"N1 p {n1['p']:.3f}, N2 p {n2['p']:.3f}, pos {cell['net']['years_positive']}/"
        f"{cell['net']['years_counted']} -> {cell['verdict']}")
    return cell


def hyperparams_table(runs: dict[str, dict]) -> dict:
    out = {}
    for name, r in runs.items():
        rows = {}
        for y, info in r["years"].items():
            keep = {k: v for k, v in info.items() if k not in ("pred_test", "pred_val", "perm", "grid",
                                                                "grid_val_ic", "fingerprint")}
            rows[int(y)] = keep
        out[name] = {"dropped": r["dropped"], "seconds": r["seconds"], "per_year": rows}
    return out


def save_oos(oos: OOS, abl: OOS | None, port: pd.DataFrame) -> None:
    pr = oos.p.loc[:, ["date", "ts", "date_id", "target_mi", "entry_date", "exit_date", "symbol", "firm",
                       "cik", "in_liq", "y_raw", "y_ex_all", "y_ex_liq"]].copy()
    for k, v in oos.pred.items():
        pr[f"pred_{k}"] = v.astype(np.float32)
    if abl is not None and "M6" in abl.pred:
        assert len(abl.p) == len(oos.p) and (abl.p["firm"].to_numpy() == oos.p["firm"].to_numpy()).all()
        pr["pred_M6_ablation"] = abl.pred["M6"].astype(np.float32)
    assert_pre_2019(pr, "eqml_oos_predictions")
    pr.to_parquet(PRED_PATH, index=False)
    assert_pre_2019(port, "eqml_oos_portfolios")
    port.to_parquet(PORT_PATH, index=False)


def port_frame(name: str, universe: str, res: ml.LSResult, months: pd.DataFrame,
               date_ids: np.ndarray) -> pd.DataFrame:
    mo = months.set_index("date_id").loc[date_ids].reset_index()
    return pd.DataFrame({"model": name, "universe": universe, "date_id": date_ids, "date": mo["date"],
                         "ts": mo["ts"], "entry_date": mo["entry_date"], "exit_date": mo["exit_date"],
                         "target_mi": mo["target_mi"], "gross": res.gross, "net": res.net,
                         "long_ret": res.long_ret, "short_ret": res.short_ret, "to_long": res.to_long,
                         "to_short": res.to_short, "n_long": res.n_long, "n_short": res.n_short,
                         "mkt": mo["mkt"], "mktrf": mo["mktrf"]})


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def parse_years(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def study(args: argparse.Namespace) -> dict:
    t0 = time.time()
    if PANEL_PATH.exists() and not args.rebuild:
        panel, months, meta = load_panel()
        log(f"panel loaded from cache: {len(panel):,} rows")
    else:
        panel, months, meta = build_panel()
    numeric = [f for f in ml.NUMERIC_FEATURES if f not in meta["dropped_features"]]
    years = parse_years(args.oos_years)
    if args.quick:
        years = years[:2]
    models = [m for m in BASE_MODELS if m in args.models.split(",")]
    universes = ["LIQ", "ALL"] if args.universe == "both" else [args.universe]
    design = Design(panel, numeric, years)
    log(f"design: {design.X.shape[0]:,} rows x {design.X.shape[1]} inputs; years {years}; models {models}")
    for sp in design.splits:
        log(f"  split {sp.year}: fit {sp.fit}, val {sp.val}, test {sp.test}, embargoed {sp.embargoed}")
    runs = run_models(design, "main", models, args.quick)
    if args.quick:
        q = {name: {"dropped": r["dropped"], "seconds": r["seconds"],
                    "val_ic": {y: i["val_ic"] for y, i in r["years"].items()},
                    "n_test_preds_finite": {y: int(np.isfinite(i["pred_test"]).sum()) for y, i in r["years"].items()}}
             for name, r in runs.items()}
        QUICK_JSON.write_text(json.dumps({"years": years, "models": q}, indent=1, default=float))
        log(f"quick run done (no OOS metrics computed): {QUICK_JSON}")
        if not args.smoke_eval:
            return {}
    comps = [c for c in ENS_COMPONENTS if c in runs and not runs[c]["dropped"]]
    abl_models = [c for c in args.ablation_models.split(",") if c in comps]
    abl_numeric = [f for f in numeric if f not in ml.ABLATION_DROP]
    design_abl = Design(panel, abl_numeric, years)
    runs_abl = run_models(design_abl, "abl", abl_models, args.quick)
    oos = OOS(panel, design, runs, ENS_COMPONENTS)
    oos_abl = OOS(panel, design_abl, runs_abl, ENS_COMPONENTS)
    cells: dict[str, dict] = {}
    ports = []
    nets: dict[str, dict[str, np.ndarray]] = {u: {} for u in universes}
    for u in universes:
        for name in [m for m in [*BASE_MODELS, "M6"] if m in oos.pred]:
            cell, res, _ = evaluate_cell(oos, months, name, u)
            cell["role"] = "HEADLINE" if name == "M6" else "DIAGNOSTIC"
            cells[f"{name}_{u}"] = cell
            sub = oos.p.loc[oos.subset(u)]
            b = ml.date_bounds(sub["date_id"].to_numpy())
            ports.append(port_frame(name, u, res, months, sub["date_id"].to_numpy()[b[:-1]]))
            nets[u][name] = res.net
            log(f"{name}_{u}: IC {cell['ic']['ic_mean']:+.4f} (t {cell['ic']['ic_t']:+.2f}), net SR "
                f"{cell['net']['sharpe']:+.2f} ({cell['net']['ann_return'] * 100:+.2f}%/yr), gross SR "
                f"{cell['gross']['sharpe']:+.2f}")
    rng = np.random.default_rng(SEED)
    head = {u: headline(oos, months, u, rng) for u in universes if "M6" in oos.pred}
    base_cmp = {}
    for u in universes:
        if "M6" in nets[u] and "M0" in nets[u]:
            d = nets[u]["M6"] - nets[u]["M0"]
            bt = ml.n2_block_bootstrap(d, block=3, n_draws=N_N2, rng=rng)
            base_cmp[u] = {"ens_minus_m0_net_ann": float(d.mean() * 12), "ens_minus_m0_sharpe_diff":
                           cells[f"M6_{u}"]["net"]["sharpe"] - cells[f"M0_{u}"]["net"]["sharpe"],
                           "months_ens_better": float((d > 0).mean()), "block_bootstrap_p_le0": bt["p"],
                           "diff_series_sharpe": ml.monthly_sharpe(d)}
    corr = {u: corr_matrix(nets[u]) for u in universes}
    log("attribution ...")
    imp = permutation_importance(oos, runs)
    sfi = single_feature_ic(oos)
    terc = size_terciles(oos, months) if "M6" in oos.pred else []
    abl_cells = {}
    for u in universes:
        if "M6" in oos_abl.pred:
            c, res, _ = evaluate_cell(oos_abl, months, "M6", u)
            c["role"] = "DIAGNOSTIC (pre-declared ablation)"
            abl_cells[u] = c
            sub = oos_abl.p.loc[oos_abl.subset(u)]
            b = ml.date_bounds(sub["date_id"].to_numpy())
            ports.append(port_frame("M6_ablation", u, res, months, sub["date_id"].to_numpy()[b[:-1]]))
    # like-for-like reference: full-feature ensemble of the SAME components as the ablation
    abl_ref = {}
    if oos_abl.components and set(oos_abl.components) != set(oos.components):
        ref = ml.ensemble([oos.pred[c] for c in oos_abl.components], oos.bounds)
        for u in universes:
            c, _, _ = evaluate_cell(oos, months, "M6", u, pred=ref)
            c["role"] = "DIAGNOSTIC (full features, ablation components " + ", ".join(oos_abl.components) + ")"
            abl_ref[u] = c
    top5 = [r["group"] for r in imp[:5]]
    surv_top = [g for g in top5 if g in ml.SURVIVORSHIP_FAMILY]
    small_beats_large = {u: (next(r for r in terc if r["universe"] == u and r["tercile"] == 1)["net_sharpe"]
                             > next(r for r in terc if r["universe"] == u and r["tercile"] == 3)["net_sharpe"])
                         for u in universes} if terc else {}
    out = {
        "study": "eqml", "zone": "equities TRAIN 2004-01-01..2019-01-01 (OOS target months 2009-01..2018-11)",
        "seed": SEED, "survivorship_note": SURV_NOTE, "panel_meta": meta, "numeric_features": numeric,
        "inputs": design.columns, "ablation_inputs": design_abl.columns, "oos_years": years,
        "splits": [sp.__dict__ for sp in design.splits], "ensemble_components": oos.components,
        "ablation_components": oos_abl.components, "headline": head, "cells": cells,
        "baseline_comparison": base_cmp, "correlations": corr, "permutation_importance": imp,
        "single_feature_ic": sfi, "size_terciles": terc, "ablation": abl_cells, "ablation_reference": abl_ref,
        "survivorship_flags": {"top5_importance": top5, "survivorship_family_in_top5": surv_top,
                               "small_tercile_beats_large_net_sharpe": small_beats_large},
        "hyperparameters": hyperparams_table(runs), "hyperparameters_ablation": hyperparams_table(runs_abl),
        "wall_clock_s": {**{f"{k}": round(v["seconds"], 1) for k, v in runs.items()},
                         **{f"{k}_ablation": round(v["seconds"], 1) for k, v in runs_abl.items()}},
        "dropped_models": {k: v["dropped"] for k, v in {**runs}.items() if v["dropped"]},
    }
    save_oos(oos, oos_abl, pd.concat(ports, ignore_index=True))
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _pc(x: float | None, nd: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:+.{nd}f}%"


def _f(x: float | None, nd: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def _p(x: float | None) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.3f}"


def write_markdown(out: dict) -> None:
    L: list[str] = []
    a = L.append
    meta = out["panel_meta"]
    a("")
    a(f"## §1. Results (auto-generated; runtime {out['runtime_s']:.0f} s, seed {out['seed']})")
    a("")
    a(f"**Every table: {SURV_NOTE}.** Monthly series, OOS target months 2009-01 … 2018-11 (119 months; "
      "Dec 2018 infeasible). Sharpe = mean/sd × √12, annualised return = mean × 12 (in brackets beside each "
      "Sharpe); per-year rows are compounded calendar-year returns (2018 = 11 months). Costs per side on "
      "turnover: 5 bps LIQ, 15 bps ALL.")
    a("")
    a("### Headline verdict cells (M6 ensemble of " + ", ".join(out["ensemble_components"]) + ")")
    a("")
    a("| cell | gross SR (ann.) | net SR (ann.) | α ann (t) | β | max DD | N1 p | N1 MDE level (excess over null) "
      "| N2 p | N2 MDE | positive years | verdict | fails |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for u, c in out["headline"].items():
        g, n = c["gross"], c["net"]
        a(f"| eqml_ENS_{u} | {_f(g['sharpe'])} ({_pc(g['ann_return'])}) | {_f(n['sharpe'])} ({_pc(n['ann_return'])}) "
          f"| {_pc(n['alpha_ann'])} ({_f(n['alpha_t'])}) | {_f(n['beta'])}{' ⚠' if n['beta_flag'] else ''} | "
          f"{_pc(n['max_dd'], 1)} | {_p(c['N1']['p'])} | {_pc(c['N1']['mde_ann_return'])} "
          f"({_pc(c['N1']['mde_excess_over_null_mean'])}) | {_p(c['N2']['p'])} | {_pc(c['N2']['mde_ann_return'])} | "
          f"{n['years_positive']}/{n['years_counted']} | **{c['verdict']}** | {', '.join(c['fails']) or '-'} |")
    a("")
    a("N1 = within-date permutation of the ensemble predictions (500 draws, same portfolio code, statistic net "
      "Sharpe). The N1 null carries the same cost drag as the real book, so its MDE level can be below zero; the "
      "bracket is the MDE above the null mean. N2 = centred circular block bootstrap of the monthly net series "
      "(3-month blocks, 1000 draws), H0 mean ≤ 0; N2 MDE = (q95 − q20 of the centred means) × 12, the "
      "profitability bar. β flagged ⚠ when |β| > 0.25.")
    a("")
    a("### Model × universe (M6 rows are the headline cells; every other row is a DIAGNOSTIC, not a verdict)")
    a("")
    a("| model | universe | role | IC mean | IC t | ICIR | OOS R² | gross SR (ann.) | net SR (ann.) | max DD "
      "| turnover/month L, S | cost drag/yr | β | pos. years |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, c in out["cells"].items():
        g, n, ic, lg = c["gross"], c["net"], c["ic"], c["legs"]
        a(f"| {c['model']} | {c['universe']} | {c['role']} | {ic['ic_mean']:+.4f} | {_f(ic['ic_t'])} | "
          f"{_f(ic['icir'], 3)} | {_pc(c.get('oos_r2'), 3)} | {_f(g['sharpe'])} ({_pc(g['ann_return'])}) | "
          f"{_f(n['sharpe'])} ({_pc(n['ann_return'])}) | {_pc(n['max_dd'], 1)} | "
          f"{lg['turnover_long_month']:.2f}, {lg['turnover_short_month']:.2f} | {_pc(-lg['cost_drag_ann'])} | "
          f"{_f(n['beta'])}{' ⚠' if n['beta_flag'] else ''} | {n['years_positive']}/{n['years_counted']} |")
    a("")
    a("Turnover = Σ|Δw| per leg per month (full replacement = 2.0). OOS R² (GKX) is against a zero forecast on "
      "raw excess returns, with the rank-unit prediction mapped to return units by a slope fitted on each "
      "refit's validation slice; M0-M5 predict the RANK target, so R² is a calibration-dependent diagnostic.")
    a("")
    a("### Legs (short leg understated: dead firms, who would sit in it, are missing)")
    a("")
    a("| model | universe | long ann. | short ann. | long − mkt | short − mkt | mkt ann. | names L, S |")
    a("|---|---|---|---|---|---|---|---|")
    for k, c in out["cells"].items():
        lg = c["legs"]
        a(f"| {c['model']} | {c['universe']} | {_pc(lg['long_ann'])} | {_pc(lg['short_ann'])} | "
          f"{_pc(lg['long_minus_mkt_ann'])} | {_pc(lg['short_minus_mkt_ann'])} | {_pc(lg['mkt_ann'])} | "
          f"{lg['mean_n_long']:.0f}, {lg['mean_n_short']:.0f} |")
    a("")
    for u in ("LIQ", "ALL"):
        ks = [k for k, c in out["cells"].items() if c["universe"] == u]
        if not ks:
            continue
        a(f"### Per-year net returns, {u} (compounded; survivor-only panel)")
        a("")
        a("| year | " + " | ".join(out["cells"][k]["model"] for k in ks) + " |")
        a("|---|" + "---|" * len(ks))
        yrs = sorted({r["year"] for k in ks for r in out["cells"][k]["net"]["per_year"]})
        for y in yrs:
            row = [str(y)]
            for k in ks:
                r = next((r for r in out["cells"][k]["net"]["per_year"] if r["year"] == y), None)
                row.append("" if r is None else f"{_pc(r['ret_compounded'])}")
            a("| " + " | ".join(row) + " |")
        a("| positive | " + " | ".join(f"{out['cells'][k]['net']['years_positive']}/"
                                      f"{out['cells'][k]['net']['years_counted']}" for k in ks) + " |")
        a("")
    a("### Ensemble vs the ridge-3 baseline (M6 − M0, monthly net)")
    a("")
    a("| universe | net ann. difference | net Sharpe difference | months M6 better | block-bootstrap p (diff ≤ 0) |")
    a("|---|---|---|---|---|")
    for u, d in out["baseline_comparison"].items():
        a(f"| {u} | {_pc(d['ens_minus_m0_net_ann'])} | {_f(d['ens_minus_m0_sharpe_diff'])} | "
          f"{d['months_ens_better']:.1%} | {_p(d['block_bootstrap_p_le0'])} |")
    a("")
    for u, cm in out["correlations"].items():
        a(f"### Correlation of monthly net long-short returns, {u}")
        a("")
        names = list(cm.keys())
        a("| | " + " | ".join(names) + " |")
        a("|---|" + "---|" * len(names))
        for r in names:
            a(f"| {r} | " + " | ".join(f"{cm[r][c2]:+.2f}" for c2 in names) + " |")
        a("")
    a("### Feature attribution (ensemble, OOS)")
    a("")
    a("Permutation importance = mean monthly ALL IC of the ensemble minus the IC with the group permuted within "
      "date (one draw per group and year, identical across components).")
    a("")
    a("| rank | group | Δ IC | t | months Δ > 0 |")
    a("|---|---|---|---|---|")
    for i, r in enumerate(out["permutation_importance"][:15], 1):
        fam = " (survivorship-flattered family)" if r["group"] in ml.SURVIVORSHIP_FAMILY else ""
        a(f"| {i} | {r['group']}{fam} | {r['importance_ic']:+.4f} | {_f(r['t'])} | {r['share_months_positive']:.0%} |")
    a("")
    a("Single-feature IC (normalised model input vs realised excess return, OOS months):")
    a("")
    a("| feature | ALL IC | ALL t | LIQ IC | LIQ t |")
    a("|---|---|---|---|---|")
    for r in out["single_feature_ic"]:
        a(f"| {r['feature']} | {r['ALL_ic']:+.4f} | {_f(r['ALL_t'])} | {r['LIQ_ic']:+.4f} | {_f(r['LIQ_t'])} |")
    a("")
    sf = out["survivorship_flags"]
    a(f"Top-5 importance groups: {', '.join(sf['top5_importance'])}. Survivorship-flattered family among them: "
      f"{', '.join(sf['survivorship_family_in_top5']) or 'none'}. Small-tercile net Sharpe above large-tercile: "
      f"{sf['small_tercile_beats_large_net_sharpe']}.")
    a("")
    a("### Survivorship diagnostics")
    a("")
    a("**Size terciles** (within-universe terciles of 63-day median dollar volume; ensemble re-ranked and "
      "decile-sorted within the tercile; universe cost):")
    a("")
    a("| universe | tercile | IC (t) | gross SR (ann.) | net SR (ann.) | max DD | turnover L | names L | pos. years |")
    a("|---|---|---|---|---|---|---|---|---|")
    for r in out["size_terciles"]:
        a(f"| {r['universe']} | {r['label']} | {r['ic_mean']:+.4f} ({_f(r['ic_t'])}) | {_f(r['gross_sharpe'])} "
          f"({_pc(r['gross_ann'])}) | {_f(r['net_sharpe'])} ({_pc(r['net_ann'])}) | {_pc(r['max_dd'], 1)} | "
          f"{r['turnover_long']:.2f} | {r['mean_n_long']:.0f} | {r['years_positive']}/10 |")
    a("")
    a("**Pre-declared ablation** (" + ", ".join(out["ablation_components"]) + " retrained without "
      + ", ".join(ml.ABLATION_DROP) + "; DIAGNOSTIC). Deviation from the brief: M5 is not retrained for the "
      "ablation (author's resource decision taken before any result was seen, saves ~1.5 h); the like-for-like "
      "comparison is the full-feature ensemble of the same components (rows marked *same components*):")
    a("")
    a("| cell | IC (t) | gross SR (ann.) | net SR (ann.) | max DD | β | pos. years |")
    a("|---|---|---|---|---|---|---|")
    for u, c in out["ablation"].items():
        a(f"| ENS_{u} ablation | {c['ic']['ic_mean']:+.4f} ({_f(c['ic']['ic_t'])}) | {_f(c['gross']['sharpe'])} "
          f"({_pc(c['gross']['ann_return'])}) | {_f(c['net']['sharpe'])} ({_pc(c['net']['ann_return'])}) | "
          f"{_pc(c['net']['max_dd'], 1)} | {_f(c['net']['beta'])} | {c['net']['years_positive']}/{c['net']['years_counted']} |")
    for u, c in out.get("ablation_reference", {}).items():
        a(f"| ENS_{u} full features, same components | {c['ic']['ic_mean']:+.4f} ({_f(c['ic']['ic_t'])}) | "
          f"{_f(c['gross']['sharpe'])} ({_pc(c['gross']['ann_return'])}) | {_f(c['net']['sharpe'])} "
          f"({_pc(c['net']['ann_return'])}) | {_pc(c['net']['max_dd'], 1)} | {_f(c['net']['beta'])} | "
          f"{c['net']['years_positive']}/{c['net']['years_counted']} |")
    for u, c in out["headline"].items():
        a(f"| ENS_{u} full (headline, all components) | {c['ic']['ic_mean']:+.4f} ({_f(c['ic']['ic_t'])}) | "
          f"{_f(c['gross']['sharpe'])} ({_pc(c['gross']['ann_return'])}) | {_f(c['net']['sharpe'])} "
          f"({_pc(c['net']['ann_return'])}) | {_pc(c['net']['max_dd'], 1)} | {_f(c['net']['beta'])} | "
          f"{c['net']['years_positive']}/{c['net']['years_counted']} |")
    a("")
    a("**Universe coverage per year** (mean names per month; SEC = distinct CIKs filing a 10-K/10-Q that year):")
    a("")
    a("| year | ALL | LIQ | share of ALL with a CIK | SEC 10-K/10-Q filers | ALL / SEC |")
    a("|---|---|---|---|---|---|")
    for y, r in sorted(meta["coverage_per_year"].items(), key=lambda kv: int(kv[0])):
        sec = r.get("sec_10k_10q_filers")
        sec_s = "n/a" if sec is None else f"{sec:,.0f}"
        ratio = "n/a" if not sec else f"{r['mean_all'] / sec:.1%}"
        a(f"| {y} | {r['mean_all']:.0f} | {r['mean_liq']:.0f} | {r['share_with_cik']:.1%} | {sec_s} | {ratio} |")
    a("")
    a("**Feature coverage per year** (share of ALL firm-months with a non-missing raw value):")
    a("")
    fc = meta["feature_coverage_per_year"]
    yrs = sorted(fc, key=int)
    a("| feature | " + " | ".join(str(y) for y in yrs) + " |")
    a("|---|" + "---|" * len(yrs))
    for f in ml.NUMERIC_FEATURES:
        a(f"| {f} | " + " | ".join(f"{fc[y][f]:.0%}" for y in yrs) + " |")
    a("")
    a("Tag coverage 2010-01 … 2018-12 (share of ALL firm-months with a non-stale PIT value; < 30 % → dropped): "
      + ", ".join(f"{g} {c:.1%}" for g, c in meta["tag_coverage_2010"].items())
      + f". Dropped tags: {meta['dropped_tags'] or 'none'}; dropped features: {meta['dropped_features'] or 'none'}.")
    a("")
    a("### Hyperparameters chosen per refit and wall clock")
    a("")
    a("| model | wall clock (main) | wall clock (ablation) | dropped |")
    a("|---|---|---|---|")
    for m, v in out["hyperparameters"].items():
        ab = out["wall_clock_s"].get(f"{m}_ablation")
        a(f"| {m} | {v['seconds'] / 60:.1f} min | {'-' if ab is None else f'{ab / 60:.1f} min'} | {v['dropped'] or '-'} |")
    a("")
    a("| model | year | chosen | val IC | best iteration / epochs |")
    a("|---|---|---|---|---|")
    for m, v in out["hyperparameters"].items():
        for y, r in sorted(v["per_year"].items(), key=lambda kv: int(kv[0])):
            if "alpha" in r:
                ch, it = f"alpha {r['alpha']:g}", "-"
            elif "params" in r:
                pp = r["params"]
                ch = (f"depth {pp['max_depth']}, lr {pp['learning_rate']}, mcw {pp['min_child_weight']}, "
                      f"ss {pp['subsample']}, cs {pp['colsample_bytree']}")
                it = str(r["best_iteration"])
            elif "lam" in r:
                ch, it = f"L1 {r['lam']:g}, lr {r['lr']:g}, {r['n_seeds']} seeds", str(r.get("seed_epochs"))
            else:
                ch, it = f"{r.get('n_seeds')} seeds, lr {r.get('lr')}", str(r.get("seed_best_epochs"))
            a(f"| {m} | {y} | {ch} | {r['val_ic']:+.4f} | {it} |")
    a("")
    a(f"Panel: {meta['n_rows']:,} ALL firm-months over {meta['n_dates']} formation dates; price matrix "
      f"{meta['n_symbols_matrix']:,} symbols × {meta['n_sessions']:,} sessions; {meta['guarded_returns']} returns "
      f"set missing by the data-error guard. CIK map: {meta['mapping']['ciks_mapped']:,} CIKs. Kept 8-K events "
      f"with a price symbol {meta['events']['kept_with_price_symbol']:,}; XBRL rows read "
      f"{meta['fundamentals']['raw_rows']:,} ({meta['fundamentals']['prepared_rows']:,} used); SUE events "
      f"{meta['fundamentals']['sue_events']:,}. Inputs: {len(out['inputs'])} (ablation {len(out['ablation_inputs'])}).")
    a("")
    text = OUT_MD.read_text() if OUT_MD.exists() else ""
    head = text.split(MARKER)[0]
    OUT_MD.write_text(head.rstrip("\n") + "\n\n" + MARKER + "\n" + "\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["both", "LIQ", "ALL"], default="both")
    ap.add_argument("--models", default=",".join(BASE_MODELS))
    ap.add_argument("--oos-years", default="2009-2018")
    ap.add_argument("--quick", action="store_true", help="development: 2 OOS years, 2 seeds, no OOS metrics")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the derived feature panel")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--ablation-models", default="M2,M3,M4",
                    help="ensemble components retrained for the ablation (M5 skipped)")
    ap.add_argument("--smoke-eval", action="store_true",
                    help="with --quick: exercise the evaluation code, outputs to /tmp, stdout discarded")
    args = ap.parse_args()
    if args.smoke_eval:
        global OUT_JSON, OUT_MD, PRED_PATH, PORT_PATH, N_N1, N_N2
        tmp = Path("/tmp/eqml_smoke")
        tmp.mkdir(exist_ok=True)
        (tmp / "r.md").write_text(OUT_MD.read_text() if OUT_MD.exists() else "")
        OUT_JSON, OUT_MD = tmp / "r.json", tmp / "r.md"
        PRED_PATH, PORT_PATH = tmp / "pred.parquet", tmp / "port.parquet"
        N_N1, N_N2 = 5, 20
    if args.build_only:
        build_panel()
        assert_sealed_clean()
        return
    out = study(args)
    assert_sealed_clean()
    if not out:
        return
    out["sealed_clean"] = True
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float))
    write_markdown(out)
    log(f"wrote {OUT_JSON} + {OUT_MD} ({out['runtime_s']} s)")


if __name__ == "__main__":
    main()
