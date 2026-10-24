#!/usr/bin/env python3
"""P9: ML capacity, learning curves, leakage controls on the study-13/14 TRAIN panel.

Design (binding): docs/P9_PREDECLARATION.md (v2) and the independent red team.
Machinery reused, unmodified: eqxs.equity_ml (splitter, models, nulls) and
eqxs.equity_ml_grid (tie-safe IC, fixed boosters, assert_nondegenerate, decile L/S with per-side
costs).  TRAIN only: the panel is read through experiments/_train_loader.py; sealed_load_count 0.

Subcommands (each independently resumable, each writes results/p9_<e>.json):
    e1 [--models M1,M2,M3] [--parts a,b,c,d,e]   leakage / integrity controls
    e2                                            learning curve
    e3 [--parts ridge,xgb,mlp]                    capacity sweep (+ oracle ceiling + SPA)
    e4                                            IC detectability (planted signal on real y_rank)
    figs                                          the four figures
Add --smoke for the 1-refit, tiny-grid, 2-shuffle smoke run (writes p9_cache/smoke/, never the real checkpoints).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"; RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import _train_loader as TL  # noqa: E402  (puts src/ on sys.path)

ROOT = TL.ROOT
from eqxs import equity_ml as ml  # noqa: E402
from eqxs import equity_ml_grid as g  # noqa: E402

# ------------------------------------------------------------------------------------------------
# Constants (pre-declared)
# ------------------------------------------------------------------------------------------------
SEED = 20261013  # study 13 / 14 base seed; refit seed = SEED + month index of the refit's first test month
PANEL_PATH = TL.CACHE / "equities_xs" / "derived" / "ml_panel_train.parquet"
META_PATH = TL.CACHE / "equities_xs" / "derived" / "ml_panel_train.meta.json"
SEALED_START = pd.Timestamp("2024-07-01", tz="UTC")
OOS_YEARS = list(range(2009, 2019))
FIRST_TARGET_MI = g.FIRST_TARGET_MI
DELTA = 0.010  # equivalence margin (pre-declared)
N_BOOT = 2000
COST = {"LIQ": 0.0005, "ALL": 0.0015}
LEAK_LEVELS = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10)  # plus level -1 = label-permuted null (size control)
PLANT_RHOS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08)
XGB_SIZES = (25, 50, 100, 200, 400, 800)
XGB_DEPTHS = (2, 3, 4, 6)
XGB_LRS = (0.03, 0.1)
XGB_MCWS = (1, 20, 100)
MLP_WIDTHS = (16, 64, 256)
MLP_DEPTHS = (1, 2, 3)
MLP_DROPS = (0.0, 0.3)
MLP_EPOCHS = (5, 10, 20, 40)
RIDGE_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)
RIDGE_EXT = (1e4, 1e5)  # study-13 grid points above the declared 8, so ridge is not handicapped
M1_ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
KNOWN_LEAK_NUMERIC = ("log_price", "log_mcap", "bm", "ep", "sp")

SMOKE = False
CACHE = RESULTS / "p9_cache"
FIG = RESULTS / "figures"


def cache_dir() -> Path:
    d = CACHE / ("smoke" if SMOKE else "")
    d.mkdir(parents=True, exist_ok=True)
    return d


def cells_dir() -> Path:
    d = cache_dir() / "cells"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ck_path(e: str) -> Path:
    return (cache_dir() / f"p9_{e}.json") if SMOKE else (RESULTS / f"p9_{e}.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_ck(e: str) -> dict:
    p = ck_path(e)
    if p.exists():
        return json.loads(p.read_text())
    return {"cells": {}}


def save_ck(e: str, ck: dict) -> None:
    p = ck_path(e)
    if p.exists():  # absorb cells written meanwhile by a concurrent process (never drop anything)
        try:
            for k, v in json.loads(p.read_text())["cells"].items():
                ck["cells"].setdefault(k, v)
        except (json.JSONDecodeError, KeyError):
            pass
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(ck, default=float))
    tmp.replace(p)


def append_table(e: str, title: str, text: str) -> None:
    f = cache_dir() / f"p9_{e}_tables.md"
    with open(f, "a") as fh:
        fh.write(f"\n## {title}  ({time.strftime('%Y-%m-%d %H:%M:%S')})\n\n{text}\n")
    print(f"\n## {title}\n{text}\n", flush=True)


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def f3(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.{nd}f}"


# ------------------------------------------------------------------------------------------------
# Cluster statistics (cluster = OOS refit-year, 10 clusters; block bootstrap block=12 aligned to the year)
# ------------------------------------------------------------------------------------------------
_BOOT: dict[int, np.ndarray] = {}


def boot_idx(G: int, B: int = N_BOOT) -> np.ndarray:
    if (G, B) not in _BOOT:
        _BOOT[(G, B)] = np.random.default_rng(20261017).integers(0, G, size=(B, G))
    return _BOOT[(G, B)]


def cstats(x, cl) -> dict:
    """Mean, cluster-robust SE / t (G-1 df), percentile cluster-bootstrap CI (2,000 draws), plain per-month t
    (too-small SE: iid months), monthly ICIR and the sqrt(12) 'annualised' ICIR."""
    x = np.asarray(x, float)
    cl = np.asarray(cl)
    ok = np.isfinite(x)
    x, cl = x[ok], cl[ok]
    u, inv = np.unique(cl, return_inverse=True)
    G = len(u)
    N = len(x)
    sums = np.bincount(inv, weights=x, minlength=G)
    cnt = np.bincount(inv, minlength=G).astype(float)
    m = float(x.mean())
    S = sums - cnt * m
    se = float(math.sqrt(G / (G - 1) * np.sum(S**2) / N**2)) if G > 1 else float("nan")
    t = m / se if se > 0 else float("nan")
    tcrit = float(sps.t.ppf(0.975, G - 1))
    idx = boot_idx(G)
    bm = sums[idx].sum(1) / cnt[idx].sum(1)
    sd = float(x.std(ddof=1)) if N > 1 else float("nan")
    return {
        "mean": m,
        "se_cluster": se,
        "t_cluster": t,
        "p_t": float(2 * (1 - sps.t.cdf(abs(t), G - 1))) if np.isfinite(t) else float("nan"),
        "ci_t": [m - tcrit * se, m + tcrit * se],
        "ci_boot": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
        "boot_se": float(bm.std(ddof=1)),
        "t_crit": tcrit,
        "sd_month": sd,
        "icir_monthly": m / sd if sd > 0 else float("nan"),
        "icir_annualised": m / sd * math.sqrt(12) if sd > 0 else float("nan"),
        "t_month_iid": m / (sd / math.sqrt(N)) if sd > 0 else float("nan"),
        "n_months": N,
        "n_clusters": G,
    }


def fmt_ci(st: dict, key="ci_boot") -> str:
    lo, hi = st[key]
    return f"[{lo:+.4f}, {hi:+.4f}]"


def delta_verdict(st: dict, delta: float = DELTA, kind: str = "capacity") -> str:
    """Pre-declared rule on a paired delta-IC.  Bootstrap CI is the declared one; flag if the wider t-CI changes it."""
    lo, hi = st["ci_boot"]
    tlo, thi = st["ci_t"]
    if kind == "capacity":
        if lo > 0:
            v = "REVISE: flexible model beats ridge (lower 95% bound > 0)"
            if hi < delta:
                v += "; gain below the delta margin"
            robust = tlo > 0
        elif hi < delta:
            v = f"DOES NOT HELP (upper 95% bound {hi:+.4f} < delta {delta:+.3f})"
            robust = thi < delta
        else:
            v = f"INCONCLUSIVE (upper 95% bound {hi:+.4f} >= delta {delta:+.3f}; NOT 'capacity absent')"
            robust = not (thi < delta) and not (tlo > 0)
    else:  # data quantity: delta = IC(more data) - IC(less data)
        if lo > 0:
            v = "DATA QUANTITY BINDS (lower 95% bound > 0)"
            robust = tlo > 0
        elif hi < delta:
            v = f"NOT BINDING (upper 95% bound {hi:+.4f} < delta {delta:+.3f})"
            robust = thi < delta
        else:
            v = f"INCONCLUSIVE (upper 95% bound {hi:+.4f} >= delta {delta:+.3f})"
            robust = not (thi < delta) and not (tlo > 0)
    return v + ("" if robust else " [verdict differs under the wider t-based CI]")


# ------------------------------------------------------------------------------------------------
# Panel
# ------------------------------------------------------------------------------------------------
@dataclass
class RF:
    year: int
    start: int
    fit: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]
    win_end: int


def make_refits(years, embargo: int = 1, window_years: int | None = None, gap: int = 1, val_len: int = 12) -> list[RF]:
    out = []
    for y in years:
        t0 = int(y) * 12
        win_end = t0 - 1 - embargo
        start = FIRST_TARGET_MI if window_years is None else max(win_end - 12 * window_years + 1, FIRST_TARGET_MI)
        val = (win_end - val_len + 1, win_end)
        fit_end = val[0] - 1 - gap
        assert fit_end >= start, (y, start, fit_end)
        out.append(RF(int(y), start, (start, fit_end), val, (t0, t0 + 11), win_end))
    return out


class YearScorer:
    def __init__(self, panel: "Panel", rows: np.ndarray):
        self.rows = rows
        self.b_all = ml.date_bounds(panel.date_id[rows])
        self.ts_all = g.TieSafeIC(panel.y_ex_all[rows], self.b_all)
        self.liq = panel.in_liq[rows]
        rl = rows[self.liq]
        self.b_liq = ml.date_bounds(panel.date_id[rl])
        self.ts_liq = g.TieSafeIC(panel.y_ex_liq[rl], self.b_liq)

    def ic_all(self, pred: np.ndarray) -> np.ndarray:
        return self.ts_all.per_date(pred)

    def ic_liq(self, pred: np.ndarray) -> np.ndarray:
        return self.ts_liq.per_date(pred[self.liq])


class Panel:
    def __init__(self):
        df = TL.train_load(PANEL_PATH, "p9 panel (TRAIN-only, row-filtered loader)")
        assert pd.to_datetime(df["ts"]).max() < SEALED_START, "panel reaches the sealed zone"
        meta = json.loads(META_PATH.read_text())
        self.numeric = [f for f in ml.NUMERIC_FEATURES if f not in meta["dropped_features"]]
        self.columns = ml.input_columns(self.numeric)
        assert (np.diff(df["date_id"].to_numpy()) >= 0).all()
        self.X = df[self.columns].to_numpy(np.float32)
        self.y = df["y_rank"].to_numpy(np.float64)
        self.y_raw = df["y_raw"].to_numpy(np.float64)
        self.y_ex_all = df["y_ex_all"].to_numpy(np.float64)
        self.y_ex_liq = df["y_ex_liq"].to_numpy(np.float64)
        self.tmi = df["target_mi"].to_numpy()
        self.date_id = df["date_id"].to_numpy()
        self.firm = df["firm"].to_numpy().astype(np.int64)
        self.in_liq = df["in_liq"].to_numpy(bool)
        self.n_firms = int(self.firm.max()) + 1
        self.max_ts = str(df["ts"].max())
        self.n = len(df)
        self.valid = self.tmi >= 0
        self.col_ix = {c: i for i, c in enumerate(self.columns)}
        self._ys: dict[int, YearScorer] = {}
        self.sealed_load_count = 0

    def rows(self, rng_: tuple[int, int]) -> np.ndarray:
        return np.flatnonzero((self.tmi >= rng_[0]) & (self.tmi <= rng_[1]) & self.valid)

    def bounds(self, idx: np.ndarray) -> np.ndarray:
        return ml.date_bounds(self.date_id[idx])

    def year_scorer(self, year: int) -> YearScorer:
        if year not in self._ys:
            self._ys[year] = YearScorer(self, self.rows((year * 12, year * 12 + 11)))
        return self._ys[year]

    # ---- design variants ----
    def lagged_X(self, k: int) -> tuple[np.ndarray, float]:
        """Features of firm f at formation date d replaced by its features at d-k (k=-1: next month = look-ahead).
        Firms with no row at d-k get the 'missing' encoding (numeric 0, miss 1); SIC kept (static)."""
        key = self.firm * 1000 + self.date_id
        idx = pd.Index(key).get_indexer(key - k)
        X2 = np.empty_like(self.X)
        have = idx >= 0
        X2[have] = self.X[idx[have]]
        num_ix = [self.col_ix[c] for c in self.numeric]
        miss_ix = [self.col_ix[f"miss_{c}"] for c in self.numeric]
        sic_ix = [self.col_ix[c] for c in ml.SIC_COLS]
        nh = ~have
        X2[np.ix_(nh, num_ix)] = 0.0
        X2[np.ix_(nh, miss_ix)] = 1.0
        X2[np.ix_(nh, sic_ix)] = self.X[np.ix_(nh, sic_ix)]
        return X2, float(nh[self.valid].mean())

    def ablate_cols(self) -> list[int]:
        drop = set(KNOWN_LEAK_NUMERIC) | {f"miss_{c}" for c in KNOWN_LEAK_NUMERIC} | set(ml.SIC_COLS)
        return [i for i, c in enumerate(self.columns) if c not in drop]


# ------------------------------------------------------------------------------------------------
# One pipeline cell: fit M? on every refit, score, optional portfolio
# ------------------------------------------------------------------------------------------------
def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", s)


def cell_file(cell_id: str, model: str) -> Path:
    return cells_dir() / f"{safe_name(cell_id)}__{model}.npz"


def make_model(name: str, nn_quick: bool):
    return g.make_model(name, quick=nn_quick, n_jobs=4, nthread=4)


def fit_predict_cell(panel: Panel, X: np.ndarray, cols: list[str], refits: list[RF], model_name: str, *,
                     y: np.ndarray | None = None, label_fn=None, fit_filter=None, refit_all: bool = False,
                     nn_quick: bool = False) -> dict:
    """Returns {"pred": {year: array}, "slope": {year: float}, "n_fit": {year: int}, "info": {year: dict}}."""
    y = panel.y if y is None else y
    out = {"pred": {}, "slope": {}, "n_fit": {}, "info": {}}
    for rf in refits:
        fi, vi, ti = panel.rows(rf.fit), panel.rows(rf.val), panel.rows(rf.test)
        if fit_filter is not None:
            fi = fit_filter(rf, fi)
        yf, yv = y[fi], y[vi]
        if label_fn is not None:
            fi, yf, vi, yv = label_fn(rf, fi, vi)
        fd = ml.FitData(X[fi], yf, panel.bounds(fi), X[vi], yv, panel.bounds(vi), cols)
        seed = SEED + rf.test[0]
        t0 = time.time()
        model = make_model(model_name, nn_quick)
        info = model.fit(fd, seed=seed)
        pv = np.asarray(info.pop("pred_val"), dtype=np.float64)
        n_fit = len(fi)
        if refit_all:
            ra = np.flatnonzero((panel.tmi >= rf.start) & (panel.tmi <= rf.win_end) & panel.valid)
            fd2 = ml.FitData(X[ra], y[ra], panel.bounds(ra), fd.Xv, fd.yv, fd.bv, cols)
            if model_name == "M1":
                model2 = g.RidgeTS("M1", (info["alpha"],))
            else:
                model2 = make_model(model_name, nn_quick)
            model2.fit(fd2, seed=seed)
            pt = model2.predict(X[ti], panel.bounds(ti))
            n_fit = len(ra)
        else:
            pt = model.predict(X[ti], panel.bounds(ti))
        if hasattr(model, "close"):
            model.close()
        g.assert_nondegenerate(pv, fd.bv, f"{model_name} {rf.year} val")
        g.assert_nondegenerate(pt, panel.bounds(ti), f"{model_name} {rf.year} test")
        out["pred"][rf.year] = np.asarray(pt, dtype=np.float64)
        out["slope"][rf.year] = ml.calibration_slope(pv, panel.y_ex_all[vi]) if len(vi) == len(pv) else float("nan")
        out["n_fit"][rf.year] = int(n_fit)
        out["info"][rf.year] = {k: v for k, v in info.items() if k in ("alpha", "val_ic", "val_ic_se", "lam", "lr",
                                                                        "n_seeds", "seconds")}
        out["info"][rf.year]["wall"] = time.time() - t0
        out["info"][rf.year]["n_val"] = int(len(vi))
    return out


def cell_cached(cell_id: str, model: str, compute) -> dict:
    f = cell_file(cell_id, model)
    if f.exists():
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return {"pred": {int(k[1:]): z[k] for k in z.files if k.startswith("p")},
                "slope": {int(k): v for k, v in meta["slope"].items()},
                "n_fit": {int(k): v for k, v in meta["n_fit"].items()},
                "info": {int(k): v for k, v in meta["info"].items()}, "cached": True}
    res = compute()
    arrs = {f"p{y}": v for y, v in res["pred"].items()}
    meta = {"slope": res["slope"], "n_fit": res["n_fit"], "info": res["info"]}
    np.savez_compressed(f, meta=np.array(json.dumps(meta, default=float)), **arrs)
    res["cached"] = False
    return res


def score_preds(panel: Panel, res: dict, portfolio: bool = True) -> dict:
    years = sorted(res["pred"])
    ia, il, cl = [], [], []
    for y in years:
        ys = panel.year_scorer(y)
        p = res["pred"][y]
        ia.append(ys.ic_all(p))
        il.append(ys.ic_liq(p))
        cl += [y] * len(ia[-1])
    out = {"years": years, "ic_all": np.concatenate(ia).tolist(), "ic_liq": np.concatenate(il).tolist(),
           "cluster": cl,
           "val_ic": [res["info"][y].get("val_ic") for y in years],
           "alpha": [res["info"][y].get("alpha") for y in years],
           "n_fit_mean": float(np.mean([res["n_fit"][y] for y in years])),
           "wall_seconds": float(sum(res["info"][y].get("wall", 0) for y in years))}
    rows = np.concatenate([panel.year_scorer(y).rows for y in years])
    pred = np.concatenate([res["pred"][y] for y in years])
    slopes = np.concatenate([np.full(len(panel.year_scorer(y).rows), res["slope"][y]) for y in years])
    out["oos_r2"] = float(ml.oos_r2(slopes * pred, panel.y_ex_all[rows]))
    r2y = {}
    for y in years:
        r = panel.year_scorer(y).rows
        r2y[y] = float(ml.oos_r2(res["slope"][y] * res["pred"][y], panel.y_ex_all[r]))
    out["oos_r2_per_year"] = r2y
    if portfolio:
        out["portfolio"] = {u: portfolio_stats(panel, rows, pred, u) for u in ("LIQ", "ALL")}
    return out


def portfolio_stats(panel: Panel, rows: np.ndarray, pred: np.ndarray, universe: str) -> dict:
    m = np.ones(len(rows), bool) if universe == "ALL" else panel.in_liq[rows]
    r = rows[m]
    p = pred[m]
    b = ml.date_bounds(panel.date_id[r])
    ls = g.decile_long_short_dual(p, panel.y_raw[r], panel.firm[r], b, panel.n_firms)
    c = COST[universe]
    net = ls.net(c, "ew")
    out = {"cost_per_side_bps": c * 1e4}
    for lab, x in (("gross", ls.gross), ("net", net)):
        out[lab] = {"ann_return": float(x.mean() * 12), "sharpe": g.ann_sharpe(x, "M"),
                    "ann_vol": float(x.std(ddof=1) * math.sqrt(12))}
    out["turnover_per_yr_two_sided"] = float(ls.two_sided_turnover("ew").mean() * 12)
    out["break_even_bps_per_side"] = float(g.break_even_cost(ls) * 1e4)
    out["months"] = int(len(ls.gross))
    yr = panel.tmi[r][b[:-1]] // 12
    out["per_year"] = [{"year": int(y_), "gross_ret": float(np.prod(1 + ls.gross[yr == y_]) - 1),
                        "net_ret": float(np.prod(1 + net[yr == y_]) - 1), "months": int((yr == y_).sum())}
                       for y_ in np.unique(yr)]
    return out


def run_cell(panel: Panel, ck: dict, cell_id: str, model: str, X: np.ndarray, cols: list[str], refits: list[RF],
             *, portfolio: bool = True, extra: dict | None = None, ck_name: str = "", **kw) -> dict | None:
    key = f"{cell_id}|{model}"
    if key in ck["cells"] and "ic_all" in ck["cells"][key]:
        return ck["cells"][key]
    t0 = time.time()
    try:
        res = cell_cached(cell_id, model, lambda: fit_predict_cell(panel, X, cols, refits, model, **kw))
    except Exception as exc:  # noqa: BLE001
        log(f"  CELL FAILED {key}: {type(exc).__name__}: {exc}")
        ck["cells"][key] = {"failed": f"{type(exc).__name__}: {exc}"}
        save_ck(ck_name, ck)
        return None
    sc = score_preds(panel, res, portfolio=portfolio)
    sc["model"] = model
    sc["cell"] = cell_id
    sc.update(extra or {})
    ck["cells"][key] = sc
    save_ck(ck_name, ck)
    log(f"  {key:<34} IC_all {np.mean(sc['ic_all']):+.4f}  IC_liq {np.mean(sc['ic_liq']):+.4f}  "
        f"({'cache' if res['cached'] else f'{time.time() - t0:.0f}s'})")
    return sc


def get_ic(ck: dict, cell: str, model: str, u: str = "all") -> np.ndarray | None:
    c = ck["cells"].get(f"{cell}|{model}")
    if not c or "ic_all" not in c:
        return None
    return np.asarray(c[f"ic_{u}"], float)


def clusters_of(ck: dict, cell: str, model: str) -> np.ndarray:
    return np.asarray(ck["cells"][f"{cell}|{model}"]["cluster"])


# ------------------------------------------------------------------------------------------------
# E1: leakage / integrity controls
# ------------------------------------------------------------------------------------------------
def shuffled_y(panel: Panel, i: int) -> np.ndarray:
    rng = np.random.default_rng([SEED, 1, i])
    y = panel.y.copy()
    b = ml.date_bounds(panel.date_id)
    for a, bb in zip(b[:-1], b[1:], strict=True):
        v = np.flatnonzero(panel.valid[a:bb]) + a
        y[v] = y[v][rng.permutation(len(v))]
    return y


def circ_label_fn(panel: Panel, draw: int):
    """Firm-wise circular time-shift of the training labels (random lag >= 12 months, within the refit's own
    training window so no OOS label can enter fitting).  Rows whose shifted-in label does not exist are dropped."""

    def fn(rf: RF, fi: np.ndarray, vi: np.ndarray):
        rows = np.concatenate([fi, vi])
        span = rf.win_end - rf.start + 1
        rng = np.random.default_rng([SEED, 2, draw, rf.year])
        lag_of_firm = rng.integers(12, span - 11, size=panel.n_firms)
        f = panel.firm[rows]
        m = panel.tmi[rows] - rf.start
        src_m = (m - lag_of_firm[f]) % span
        keys = f * 1000 + m
        order = np.argsort(keys)
        ks = keys[order]
        q = f * 1000 + src_m
        pos = np.searchsorted(ks, q)
        pos = np.clip(pos, 0, len(ks) - 1)
        found = ks[pos] == q
        src_row = rows[order[pos]]
        newy = panel.y[src_row]
        is_fit = np.arange(len(rows)) < len(fi)
        kf = found & is_fit
        kv = found & ~is_fit
        return rows[kf], newy[kf], rows[kv], newy[kv]

    return fn


class LeakCtx:
    """Fast within-date Spearman of a candidate leaky column with y_rank on every 4th formation date (calibration)."""

    def __init__(self, panel: Panel):
        v = panel.valid
        dates = np.unique(panel.date_id[v])[::4]
        self.rows = np.flatnonzero(v & np.isin(panel.date_id, dates))
        self.b = ml.date_bounds(panel.date_id[self.rows])
        self.yq = []
        y = panel.y[self.rows]
        for a_, b_ in zip(self.b[:-1], self.b[1:], strict=True):
            r = rankdata(y[a_:b_])
            r = r - r.mean()
            self.yq.append(r / math.sqrt(float((r * r).sum())))
        self.y = y
        self.sy = float(panel.y[v].std())
        self.low_miss = [c for c in panel.numeric if panel.X[v, panel.col_ix[f"miss_{c}"]].mean() < 0.10]

    def ic(self, col: np.ndarray) -> float:
        out = []
        for (a_, b_), q in zip(zip(self.b[:-1], self.b[1:], strict=True), self.yq, strict=True):
            r = rankdata(col[a_:b_])
            r = r - r.mean()
            n = math.sqrt(float((r * r).sum()))
            out.append(float(r @ q) / n if n > 0 else 0.0)
        return float(np.mean(out))


_LEAK_CTX: dict = {}


def leak_columns(panel: Panel, iota: float, draw: int, obs_only: bool = False) -> tuple[np.ndarray, dict]:
    """3 real, mostly-observed numeric features CONTAMINATED IN PLACE (f + lambda * z(y_rank); the clean original is NOT kept
    alongside -- a leaky copy next to its own clean original lets any linear model subtract the original and recover y
    exactly, IC ~ 1);
    lambda is calibrated by bisection on the realised within-date SPEARMAN so that each leaky column's IC with the
    target exceeds its own un-leaked IC by exactly ~iota (Pearson calibration is wrong for tie-heavy columns).  Injected
    on EVERY row, including the OOS rows (a look-ahead leak of the label into the feature)."""
    if "ctx" not in _LEAK_CTX:
        _LEAK_CTX["ctx"] = LeakCtx(panel)
    ctx: LeakCtx = _LEAK_CTX["ctx"]
    rng = np.random.default_rng([SEED, 3, draw, int(round(abs(iota) * 1e4)) + (99999 if iota < 0 else 0)])
    null_control = iota < 0  # label-permuted leak: same lambda as iota=0.02, but the injected term carries NO information
    iota = 0.02 if null_control else iota
    y_inj = panel.y
    if null_control:
        y_inj = shuffled_y(panel, 1000 + draw)
    pick = rng.choice(ctx.low_miss, size=3, replace=False)
    v = panel.valid
    cols, lams, own, realised = [], [], [], []
    for name in pick:
        j = panel.col_ix[name]
        f = panel.X[:, j].astype(np.float64)
        sf = float(f[v].std())
        fs = f[ctx.rows] / sf
        obs_all = (1.0 - panel.X[:, panel.col_ix[f"miss_{name}"]].astype(np.float64)) if obs_only else np.ones(panel.n)
        ys = ctx.y / ctx.sy * obs_all[ctx.rows]
        ic0 = ctx.ic(fs)
        lam = 0.0
        if iota > 0:
            lo, hi = 0.0, 2.0
            for _ in range(28):
                mid = (lo + hi) / 2
                if ctx.ic(fs + mid * ys) - ic0 < iota:
                    lo = mid
                else:
                    hi = mid
            lam = (lo + hi) / 2
        col = (f / sf) + lam * np.where(v, y_inj, 0.0) / ctx.sy * obs_all
        cols.append((col * sf).astype(np.float32))
        lams.append(lam)
        own.append(ic0)
        realised.append(0.0 if null_control else ctx.ic(fs + lam * ys) - ic0)
    return np.stack(cols, axis=1), {"features": [str(n) for n in pick], "col_index": [int(panel.col_ix[n]) for n in pick],
                                    "lambda": lams, "own_ic_of_feature": own, "realised_injected_ic_per_col": realised}


def e1_cells(panel: Panel, models: list[str], parts: set[str], smoke: bool) -> list[tuple]:
    """(cell_id, model, builder) in run order."""
    n_shuf = 2 if smoke else 20
    n_circ = 2 if smoke else 10
    leak_levels = (0.0, 0.05) if smoke else LEAK_LEVELS
    n_leak = {"M1": 2 if smoke else 20, "M2": 2 if smoke else int(os.environ.get("P9_LEAK_DRAWS_M2", 20))}
    nn_shuf = 1 if smoke else int(os.environ.get("P9_NN_SHUF", 10))
    nn_circ = 1 if smoke else int(os.environ.get("P9_NN_CIRC", 5))
    cells = [("base", "-")]
    if "b" in parts:
        cells += [(f"lag{k}", "-") for k in (-1, 1, 2, 3)]
    if "c" in parts:
        cells += [("abl", "-")] + [(f"circ{i:02d}", "-") for i in range(n_circ)]
    if "e" in parts:
        cells += [(f"emb{e}", "-") for e in (0, 3, 6)]
    if "a" in parts:
        cells += [(f"shuf{i:02d}", "-") for i in range(n_shuf)]
    if "d" in parts:
        cells += [(f"leak{lv:g}_{d:02d}", "-") for lv in (*leak_levels, -1.0) for d in range(max(n_leak.values()))]
    if "f" in parts:  # diagnostic: leak injected on OBSERVED rows only (no tie/missing atoms to break)
        cells += [(f"leakobs{lv:g}_{d:02d}", "-") for lv in (0.002, 0.01) for d in range(5 if smoke else 10)]
    plan = []
    for cid, _ in cells:
        for m in models:
            if cid.startswith("shuf") and m == "M3" and int(cid[4:]) >= nn_shuf:
                continue
            if cid.startswith("circ") and m == "M3" and int(cid[4:]) >= nn_circ:
                continue
            if cid.startswith("leakobs"):
                if m == "M3":
                    continue
            elif cid.startswith("leak"):
                if m == "M3":
                    continue
                if int(cid.split("_")[1]) >= n_leak[m]:
                    continue
            plan.append((cid, m))
    return plan


def run_e1(args) -> None:
    panel = Panel()
    years = [2012] if SMOKE else OOS_YEARS
    models = args.models.split(",")
    parts = set(args.parts.split(","))
    ck = load_ck("e1")
    ck["meta"] = {"years": years, "models": models, "parts": sorted(parts), "smoke": SMOKE,
                  "panel_max_ts": panel.max_ts, "sealed_load_count": panel.sealed_load_count}
    refits1 = make_refits(years, 1)
    plan = e1_cells(panel, models, parts, SMOKE)
    log(f"E1: {len(plan)} (cell, model) fits planned; years {years}")
    lag_cache: dict[int, tuple] = {}
    for cid, m in plan:
        nn_quick = SMOKE and m == "M3"
        kw = dict(ck_name="e1", nn_quick=nn_quick)
        if cid == "base":
            run_cell(panel, ck, cid, m, panel.X, panel.columns, refits1, **kw)
        elif cid.startswith("lag"):
            k = int(cid[3:])
            if k not in lag_cache:
                lag_cache.clear()
                lag_cache[k] = panel.lagged_X(k)
                log(f"  built lag {k}: share of valid rows filled as missing {lag_cache[k][1]:.3f}")
            X2, fill = lag_cache[k]
            run_cell(panel, ck, cid, m, X2, panel.columns, refits1, extra={"fill_share": fill}, **kw)
        elif cid == "abl":
            keep = panel.ablate_cols()
            run_cell(panel, ck, cid, m, panel.X[:, keep], [panel.columns[i] for i in keep], refits1,
                     extra={"dropped": [c for i, c in enumerate(panel.columns) if i not in set(keep)]}, **kw)
        elif cid.startswith("emb"):
            e = int(cid[3:])
            run_cell(panel, ck, cid, m, panel.X, panel.columns, make_refits(years, e), **kw)
        elif cid.startswith("shuf"):
            i = int(cid[4:])
            run_cell(panel, ck, cid, m, panel.X, panel.columns, refits1, y=shuffled_y(panel, i), portfolio=False, **kw)
        elif cid.startswith("circ"):
            i = int(cid[4:])
            run_cell(panel, ck, cid, m, panel.X, panel.columns, refits1, label_fn=circ_label_fn(panel, i),
                     portfolio=False, **kw)
        elif cid.startswith("leak"):
            obs = cid.startswith("leakobs")
            lv, d = cid[7:].split("_") if obs else cid[4:].split("_")
            lv, d = float(lv), int(d)
            if f"{cid}|{m}" in ck["cells"] and "ic_all" in ck["cells"][f"{cid}|{m}"]:
                continue
            L, linfo = leak_columns(panel, lv, d, obs_only=obs)
            X2 = panel.X.copy()
            X2[:, linfo["col_index"]] = L
            run_cell(panel, ck, cid, m, X2, panel.columns, refits1, portfolio=False, extra={"leak": linfo, "iota": lv}, **kw)
            del X2, L
    analyse_e1(ck, models, smoke=SMOKE)
    save_ck("e1", ck)


def paired(ck: dict, a: tuple[str, str], b: tuple[str, str], u: str = "all"):
    xa, xb = get_ic(ck, *a, u=u), get_ic(ck, *b, u=u)
    if xa is None or xb is None or len(xa) != len(xb):
        return None
    return cstats(xa - xb, clusters_of(ck, *a))


def analyse_e1(ck: dict, models: list[str], smoke: bool = False) -> None:
    out: dict = {}
    allm = sorted({k.split("|")[1] for k in ck["cells"] if "|" in k})
    # ---- baseline ----
    rows = []
    for m in allm:
        c = ck["cells"].get(f"base|{m}")
        if not c or "ic_all" not in c:
            continue
        for u in ("all", "liq"):
            st = cstats(c[f"ic_{u}"], c["cluster"])
            out.setdefault("baseline", {}).setdefault(m, {})[u] = st
            rows.append([m, u.upper(), f3(st["mean"]), fmt_ci(st), fmt_ci(st, "ci_t"), f3(st["t_cluster"], 2),
                         f3(st["t_month_iid"], 2) + " (iid, too-small SE)", f3(st["icir_monthly"], 3),
                         f3(st["icir_annualised"], 3) + " (annualised)",
                         f3(c["oos_r2"], 4) if u == "all" else ""])
    if rows:
        append_table("e1", "E1 baseline (expanding, embargo 1, annual refit, OOS 2009-2018)",
                     md_table(["model", "univ", "IC mean", "95% CI (cluster bootstrap)", "95% CI (cluster t)",
                               "cluster t", "plain per-month t", "ICIR monthly", "ICIR annualised", "OOS R2 (ALL)"], rows))
        prow = []
        for m in allm:
            c = ck["cells"].get(f"base|{m}")
            if not c or "portfolio" not in c:
                continue
            for u in ("LIQ", "ALL"):
                p = c["portfolio"][u]
                prow.append([m, u, f"{p['cost_per_side_bps']:.0f}", f"{p['gross']['ann_return'] * 100:+.2f}%",
                             f3(p["gross"]["sharpe"], 2), f"{p['net']['ann_return'] * 100:+.2f}%", f3(p["net"]["sharpe"], 2),
                             f"{p['turnover_per_yr_two_sided']:.1f}x", f"{p['break_even_bps_per_side']:.1f} bps"])
        append_table("e1", "E1 baseline decile L/S portfolios (secondary to IC)",
                     md_table(["model", "univ", "bps/side", "gross ann ret", "gross SR", "net ann ret", "net SR",
                               "turnover/yr (2-sided)", "break-even/side"], prow))
    # ---- model comparison on the baseline (capacity at a pre-fixed configuration) ----
    cmp_rows = []
    for m in allm:
        if m == "M1":
            continue
        pr = paired(ck, ("base", m), ("base", "M1"))
        if pr is not None:
            out.setdefault("baseline_vs_ridge", {})[m] = {"paired": pr, "verdict": delta_verdict(pr)}
            cmp_rows.append([f"{m} - M1", f3(pr["mean"]), fmt_ci(pr), delta_verdict(pr)])
    if cmp_rows:
        append_table("e1", "E1 baseline: paired dIC of the pre-fixed flexible models vs ridge-all (delta margin +0.010)",
                     md_table(["pair", "dIC", "95% CI (cluster bootstrap)", "pre-declared verdict"], cmp_rows))
    # ---- E1a ----
    for m in allm:
        shuf = [get_ic(ck, f"shuf{i:02d}", m) for i in range(100)]
        shuf = [s for s in shuf if s is not None]
        if len(shuf) < 2:
            continue
        means = np.array([s.mean() for s in shuf])
        avg_series = np.mean(shuf, axis=0)
        st = cstats(avg_series, clusters_of(ck, "shuf00", m))
        se_across = float(means.std(ddof=1) / math.sqrt(len(means)))
        within2 = abs(means.mean()) < 2 * se_across
        ci_has0 = st["ci_boot"][0] <= 0 <= st["ci_boot"][1]
        verdict = ("INTEGRITY-CLEAN (evaluation/selection-bug check only; NOT a leakage test)"
                   if (within2 and ci_has0) else "INTEGRITY CHECK FAILED: shuffled-label IC differs from 0")
        out.setdefault("e1a", {})[m] = {"n_shuffles": len(shuf), "mean_of_means": float(means.mean()),
                                        "sd_across_shuffles": float(means.std(ddof=1)), "se_across": se_across,
                                        "min": float(means.min()), "max": float(means.max()),
                                        "cluster_ci_boot_avg_series": st["ci_boot"], "verdict": verdict}
    if "e1a" in out:
        append_table("e1", "E1a label-shuffle integrity check (E[IC]=0; NOT a leakage test)",
                     md_table(["model", "n shuffles", "mean OOS IC", "SD across shuffles", "SE (SD/sqrt n)", "min..max",
                               "cluster CI of shuffle-avg series", "verdict"],
                              [[m, v["n_shuffles"], f3(v["mean_of_means"]), f3(v["sd_across_shuffles"]), f3(v["se_across"]),
                                f"{v['min']:+.4f}..{v['max']:+.4f}", f"[{v['cluster_ci_boot_avg_series'][0]:+.4f}, "
                                f"{v['cluster_ci_boot_avg_series'][1]:+.4f}]", v["verdict"]] for m, v in out["e1a"].items()]))
    # ---- E1b ----
    for m in allm:
        rows, d = [], {}
        for k in (-1, 0, 1, 2, 3):
            cid = "base" if k == 0 else f"lag{k}"
            x = get_ic(ck, cid, m)
            if x is None:
                continue
            st = cstats(x, clusters_of(ck, cid, m))
            pr = None if k == 0 else paired(ck, (cid, m), ("base", m))
            d[str(k)] = {"ic": st, "d_vs_k0": pr}
            fill = ck["cells"][f"{cid}|{m}"].get("fill_share")
            rows.append([k, f3(st["mean"]), fmt_ci(st), "" if pr is None else f3(pr["mean"]),
                         "" if pr is None else fmt_ci(pr), "" if fill is None else f"{fill:.3f}"])
        if d:
            v = ""
            if "-1" in d and d["-1"]["d_vs_k0"] is not None:
                lo = d["-1"]["d_vs_k0"]["ci_boot"][0]
                v = ("LOOK-AHEAD-SENSITIVE: IC(k=-1) >> IC(k=0) (gross sanity row: the pipeline can see a catastrophic leak)"
                     if lo > 0 else "gross sanity row NOT visible (pipeline blind to the +1-month leak)")
            ks = [(k, d[str(k)]["ic"]["mean"]) for k in (0, 1, 2, 3) if str(k) in d]
            mono = all(ks[i][1] >= ks[i + 1][1] for i in range(len(ks) - 1)) if len(ks) > 1 else None
            d["verdict"] = v
            d["monotone_decay_k0_to_k3"] = mono
            out.setdefault("e1b", {})[m] = d
            append_table("e1", f"E1b feature-lag ladder, model {m}  [{v}; IC(k) non-increasing for k>=0: {mono}]",
                         md_table(["k (features from T-k)", "IC (ALL)", "95% CI", "dIC vs k=0", "95% CI of dIC",
                                   "share rows filled missing"], rows))
    # ---- E1c ----
    for m in allm:
        r = {}
        pr = paired(ck, ("base", m), ("abl", m))
        if pr is not None:
            lo, hi = pr["ci_boot"]
            r["ablation_full_minus_ablated"] = pr
            r["ablation_verdict"] = ("FLATTERED by the split/survivor-tainted block (lower bound > 0)" if lo > 0 else
                                     f"no detectable contribution of the tainted block (95% CI {lo:+.4f}..{hi:+.4f}); "
                                     + ("equivalent within delta" if (hi < DELTA and lo > -DELTA) else "NOT excluded >= delta"))
        circ = [get_ic(ck, f"circ{i:02d}", m) for i in range(100)]
        circ = [c_ for c_ in circ if c_ is not None]
        if circ:
            means = np.array([c_.mean() for c_ in circ])
            st = cstats(np.mean(circ, axis=0), clusters_of(ck, "circ00", m))
            r["circshift"] = {"n_draws": len(circ), "mean_ic": float(means.mean()), "sd_across": float(means.std(ddof=1)) if len(means) > 1 else None,
                              "cluster": st}
            r["circshift_verdict"] = ("FIRM-FINGERPRINT MEMORISATION DETECTED (shifted-label IC > 0, lower bound > 0)"
                                      if st["ci_boot"][0] > 0 else "no firm-fingerprint memorisation detected "
                                      f"(shifted-label IC {st['mean']:+.4f}, 95% CI {fmt_ci(st)})")
        if r:
            out.setdefault("e1c", {})[m] = r
            lines = []
            if "ablation_full_minus_ablated" in r:
                a_ = r["ablation_full_minus_ablated"]
                lines.append(f"- ablation (drop log_price, log_mcap, bm, ep, sp, miss_*, sic_*): full minus ablated "
                             f"dIC {a_['mean']:+.4f} CI {fmt_ci(a_)}; **{r['ablation_verdict']}**")
            if "circshift" in r:
                c_ = r["circshift"]
                lines.append(f"- firm-wise circular label shift (>=12m), {c_['n_draws']} draws: mean OOS IC "
                             f"{c_['mean_ic']:+.4f}, cluster CI {fmt_ci(c_['cluster'])}; **{r['circshift_verdict']}**")
            append_table("e1", f"E1c known-leak ablation + circular label shift, model {m}", "\n".join(lines))
    # ---- E1e ----
    for m in allm:
        rows, d = [], {}
        for e in (0, 1, 3, 6):
            cid = "base" if e == 1 else f"emb{e}"
            x = get_ic(ck, cid, m)
            if x is None:
                continue
            st = cstats(x, clusters_of(ck, cid, m))
            pr = None if e == 1 else paired(ck, (cid, m), ("base", m))
            d[str(e)] = {"ic": st, "d_vs_e1": pr}
            rows.append([e, f3(st["mean"]), fmt_ci(st), "" if pr is None else f3(pr["mean"]), "" if pr is None else fmt_ci(pr)])
        if len(d) > 1:
            out.setdefault("e1e", {})[m] = d
            append_table("e1", f"E1e embargo ladder (weak probe), model {m}",
                         md_table(["embargo (months)", "IC (ALL)", "95% CI", "dIC vs embargo 1", "95% CI of dIC"], rows))
    # ---- E1d ----
    for m in ("M1", "M2"):
        det = {}
        tcrit = float(sps.t.ppf(0.975, 9))
        for lv in (*LEAK_LEVELS, -1.0):
            ds = []
            for d_ in range(100):
                cid = f"leak{lv:g}_{d_:02d}"
                pr = paired(ck, (cid, m), ("base", m))
                if pr is None:
                    continue
                ds.append(pr)
            if not ds:
                continue
            rej = np.mean([(p["t_cluster"] > tcrit) for p in ds])  # detected = ic increased, two-sided 5% t
            rej2 = np.mean([(abs(p["t_cluster"]) > tcrit) for p in ds])
            # unpaired: leaky IC vs the clean-IC cluster SE (treats the two series as independent, conservative)
            base = cstats(get_ic(ck, "base", m), clusters_of(ck, "base", m))
            unp = []
            for d_ in range(100):
                x = get_ic(ck, f"leak{lv:g}_{d_:02d}", m)
                if x is None:
                    continue
                s2 = cstats(x, clusters_of(ck, "base", m))
                z = (s2["mean"] - base["mean"]) / math.sqrt(s2["se_cluster"] ** 2 + base["se_cluster"] ** 2)
                unp.append(z > 1.96)
            det[f"{lv:g}"] = {"n_draws": len(ds), "mean_dIC": float(np.mean([p["mean"] for p in ds])),
                              "power_paired_up": float(rej), "rej_two_sided": float(rej2),
                              "power_unpaired": float(np.mean(unp)) if unp else None}
        if det:
            pos = [(float(k), v["power_paired_up"]) for k, v in det.items() if float(k) > 0]  # level -1 (null) excluded
            mde = next((k for k, p in sorted(pos) if p >= 0.8), None)
            pos_u = [(float(k), v["power_unpaired"]) for k, v in det.items() if float(k) > 0 and v["power_unpaired"] is not None]
            mde_u = next((k for k, p in sorted(pos_u) if p >= 0.8), None)
            size = det.get("-1", {}).get("rej_two_sided")
            res = {"levels": det, "smallest_detected_paired": mde, "smallest_detected_unpaired": mde_u, "size_at_zero_two_sided": size}
            res["verdict"] = (f"smallest injected single-feature IC detected at >=80% power (paired dIC vs clean, cluster t): "
                              f"{mde if mde is not None else '> 0.10 (not reached)'}; vs unpaired clean-IC SE: "
                              f"{mde_u if mde_u is not None else '> 0.10 (not reached)'}")
            out.setdefault("e1d", {})[m] = res
            append_table("e1", f"E1d graded leak injection, model {m}: {res['verdict']}",
                         md_table(["injected IC per leaky col (3 cols)", "draws", "mean dIC (leak - clean)", "power (paired, 5%)",
                                   "power (unpaired SE)", "two-sided rejection"],
                                  [[k, v["n_draws"], f3(v["mean_dIC"]), f"{v['power_paired_up']:.2f}",
                                    "" if v["power_unpaired"] is None else f"{v['power_unpaired']:.2f}", f"{v['rej_two_sided']:.2f}"]
                                   for k, v in det.items()]))
    # ---- diagnostic: leak on observed rows only vs all rows ----
    tcrit = float(sps.t.ppf(0.975, 9))
    rows = []
    for m in ("M1", "M2"):
        for lv in (0.002, 0.01):
            for tag, pref in (("all rows", "leak"), ("observed rows only", "leakobs")):
                ds = [paired(ck, (f"{pref}{lv:g}_{d_:02d}", m), ("base", m)) for d_ in range(100)]
                ds = [p for p in ds if p is not None]
                if ds:
                    rows.append([m, f"{lv:g}", tag, len(ds), f3(float(np.mean([p["mean"] for p in ds]))),
                                 f"{np.mean([p['t_cluster'] > tcrit for p in ds]):.2f}"])
                    out.setdefault("e1d_obs_diag", {}).setdefault(m, {})[f"{lv:g}|{tag}"] = {"mean_dIC": float(np.mean([p["mean"] for p in ds])),
                                                                                           "power": float(np.mean([p["t_cluster"] > tcrit for p in ds])), "n": len(ds)}
    if rows and any(r_[2] == "observed rows only" for r_ in rows):
        append_table("e1", "E1d diagnostic: does the leak need tied / missing rows to be exploited? (leak on all rows vs observed rows only)",
                     md_table(["model", "injected IC per col", "rows contaminated", "draws", "mean dIC vs clean", "power (paired)"], rows))
    ck["analysis"] = out


# ------------------------------------------------------------------------------------------------
# E2: learning curve
# ------------------------------------------------------------------------------------------------
def firm_filter(panel: Panel, frac: float, draw: int):
    rng = np.random.default_rng([SEED, 5, int(round(frac * 100)), draw])
    keep = rng.random(panel.n_firms) < frac

    def fn(rf: RF, fi: np.ndarray) -> np.ndarray:
        return fi[keep[panel.firm[fi]]]

    return fn


def run_e2(args) -> None:
    panel = Panel()
    years = [2012] if SMOKE else OOS_YEARS
    ck = load_ck("e2")
    ck["meta"] = {"years": years, "smoke": SMOKE, "delta": DELTA, "sealed_load_count": panel.sealed_load_count}
    models = ["M1", "M2"]
    n_fs = {"M1": 3 if SMOKE else 10, "M2": 2 if SMOKE else 10}
    arms = [("w2", 2), ("w3", 3), ("w5", 5), ("base", None)]
    log("E2 learning curve")
    for m in models:
        for cid, L in arms:
            refits = make_refits(years, 1, window_years=L)
            run_cell(panel, ck, cid, m, panel.X, panel.columns, refits, ck_name="e2", extra={"window_years": L})
        run_cell(panel, ck, "rv", m, panel.X, panel.columns, make_refits(years, 1), ck_name="e2", refit_all=True,
                 extra={"window_years": None})
        for frac in ((0.25,) if SMOKE else (0.10, 0.25, 0.50)):
            for d in range(n_fs[m]):
                run_cell(panel, ck, f"fs{int(frac * 100)}_{d}", m, panel.X, panel.columns, make_refits(years, 1),
                         ck_name="e2", fit_filter=firm_filter(panel, frac, d), extra={"firm_frac": frac}, portfolio=False)
    analyse_e2(ck)
    save_ck("e2", ck)


def analyse_e2(ck: dict) -> None:
    out: dict = {}
    for m in ("M1", "M2"):
        base = ck["cells"].get(f"base|{m}")
        if not base or "ic_all" not in base:
            continue
        cl = np.asarray(base["cluster"])
        xb = np.asarray(base["ic_all"])
        rows, d = [], {}

        def one(label, series_all, series_liq, nfit, delta_series, kind, sign=1):
            st = cstats(series_all, cl)
            stl = cstats(series_liq, cl)
            dd = None if delta_series is None else cstats(delta_series, cl)
            v = None if dd is None else delta_verdict(dd, kind="data")
            d[label] = {"ic_all": st, "ic_liq": stl, "n_fit_mean": nfit, "delta_vs_expanding": dd, "verdict": v}
            rows.append([label, f"{nfit:,.0f}", f3(st["mean"]), fmt_ci(st), f3(stl["mean"]),
                         "" if dd is None else f3(dd["mean"]), "" if dd is None else fmt_ci(dd), "" if v is None else v])

        for cid, lab in (("w2", "2y window"), ("w3", "3y window"), ("w5", "5y window")):
            c = ck["cells"].get(f"{cid}|{m}")
            if c and "ic_all" in c:
                one(lab, c["ic_all"], c["ic_liq"], c["n_fit_mean"], xb - np.asarray(c["ic_all"]), "data")
        one("expanding (study 13)", base["ic_all"], base["ic_liq"], base["n_fit_mean"], None, "data")
        c = ck["cells"].get(f"rv|{m}")
        if c and "ic_all" in c:
            # gain from refitting on fit+val: delta = rv - expanding (positive = last 12 months matter)
            one("expanding, refit on fit+val", c["ic_all"], c["ic_liq"], c["n_fit_mean"], np.asarray(c["ic_all"]) - xb, "data")
        for frac in (0.10, 0.25, 0.50):
            cs = [ck["cells"].get(f"fs{int(frac * 100)}_{i}|{m}") for i in range(50)]
            cs = [c_ for c_ in cs if c_ and "ic_all" in c_]
            if not cs:
                continue
            ia = np.mean([c_["ic_all"] for c_ in cs], axis=0)
            il = np.mean([c_["ic_liq"] for c_ in cs], axis=0)
            spread = float(np.std([np.mean(c_["ic_all"]) for c_ in cs], ddof=1)) if len(cs) > 1 else None
            one(f"{int(frac * 100)}% of firms ({len(cs)} draws avg)", ia, il, float(np.mean([c_["n_fit_mean"] for c_ in cs])),
                xb - ia, "data")
            d[f"{int(frac * 100)}% of firms ({len(cs)} draws avg)"]["sd_across_draws_of_mean_ic"] = spread
        out[m] = d
        append_table("e2", f"E2 learning curve, model {m} (delta = IC(expanding) - IC(arm); refit arm: IC(arm) - IC(expanding); "
                           f"delta margin {DELTA:+.3f})",
                     md_table(["arm", "actual fit rows (mean over refits)", "IC ALL", "95% CI", "IC LIQ", "paired dIC", "95% CI of dIC",
                               "pre-declared verdict"], rows))
    ck["analysis"] = out


# ------------------------------------------------------------------------------------------------
# E3: capacity sweep
# ------------------------------------------------------------------------------------------------
class RidgeEig:
    """Ridge with intercept at many alphas via one eigendecomposition per fit slice (== sklearn Ridge)."""

    def __init__(self, Xf: np.ndarray):
        self.mu = Xf.mean(0, dtype=np.float64)
        Xc = Xf.astype(np.float64) - self.mu
        self.s, self.V = np.linalg.eigh(Xc.T @ Xc)
        self.Xc_T = None

    def weights(self, q: np.ndarray, alphas) -> np.ndarray:
        """q = Xc' (y - ybar) (p,) or (p, D); returns (p, A[, D])."""
        a = np.asarray(alphas, float)
        z = self.V.T @ q
        if q.ndim == 1:
            return self.V @ (z[:, None] / (self.s[:, None] + a[None, :]))
        return np.einsum("pk,kd,ka->pad", self.V, z, 1.0 / (self.s[:, None] + a[None, :]))


def ridge_fit_all_alphas(panel: Panel, rf: RF, alphas, y: np.ndarray | None = None):
    y = panel.y if y is None else y
    fi, vi, ti = panel.rows(rf.fit), panel.rows(rf.val), panel.rows(rf.test)
    Xf = panel.X[fi]
    eig = RidgeEig(Xf)
    yc = y[fi] - y[fi].mean()
    q = (Xf.astype(np.float64) - eig.mu).T @ yc
    W = eig.weights(q, alphas)
    b0 = y[fi].mean()
    Pv = (panel.X[vi].astype(np.float64) - eig.mu) @ W + b0
    Pt = (panel.X[ti].astype(np.float64) - eig.mu) @ W + b0
    return fi, vi, ti, Pv, Pt


def e3_ridge(panel: Panel, ck: dict, years: list[int]) -> None:
    alphas = list(RIDGE_LAMBDAS) + list(RIDGE_EXT)
    if "ridge" in ck["cells"]:
        return
    res = {"alphas": alphas, "ic_all": [[] for _ in alphas], "ic_liq": [[] for _ in alphas], "val_ic": [], "cluster": [],
           "selected_alpha_8pt": [], "selected_alpha_10pt": []}
    for rf in make_refits(years, 1):
        fi, vi, ti, Pv, Pt = ridge_fit_all_alphas(panel, rf, alphas)
        tsv = g.TieSafeIC(panel.y[vi], panel.bounds(vi))
        vic = [tsv(Pv[:, k]) for k in range(len(alphas))]
        ys = panel.year_scorer(rf.year)
        for k in range(len(alphas)):
            res["ic_all"][k] += ys.ic_all(Pt[:, k]).tolist()
            res["ic_liq"][k] += ys.ic_liq(Pt[:, k]).tolist()
        res["val_ic"].append(vic)
        res["selected_alpha_8pt"].append(alphas[int(np.argmax(vic[:8]))])
        res["selected_alpha_10pt"].append(alphas[int(np.argmax(vic))])
        res["cluster"] += [rf.year] * len(ys.ic_all(Pt[:, 0]))
    ck["cells"]["ridge"] = res
    save_ck("e3", ck)
    log("  E3 ridge lambda sweep done: " + ", ".join(f"{a:g}:{np.mean(r):+.4f}" for a, r in zip(alphas, res["ic_all"])))


def xgb_cfgs(smoke: bool) -> list[tuple[int, float, int]]:
    if smoke:
        return [(3, 0.1, 100), (2, 0.03, 20)]
    return [(d, lr, mcw) for d in XGB_DEPTHS for lr in XGB_LRS for mcw in XGB_MCWS]


def e3_xgb(panel: Panel, ck: dict, years: list[int], smoke: bool) -> None:
    import xgboost as xgb

    sizes = (25, 50, 100) if smoke else XGB_SIZES
    nmax = max(sizes)
    store = ck["cells"].setdefault("xgb", {})
    for rf in make_refits(years, 1):
        todo = [c for c in xgb_cfgs(smoke) if f"{rf.year}|{c[0]}_{c[1]}_{c[2]}" not in store]
        if not todo:
            continue
        fi, vi, ti = panel.rows(rf.fit), panel.rows(rf.val), panel.rows(rf.test)
        dfit = xgb.DMatrix(panel.X[fi], label=panel.y[fi], nthread=4)
        dval, dte = xgb.DMatrix(panel.X[vi], nthread=4), xgb.DMatrix(panel.X[ti], nthread=4)
        tsv = g.TieSafeIC(panel.y[vi], panel.bounds(vi))
        ys = panel.year_scorer(rf.year)
        for (d, lr, mcw) in todo:
            t0 = time.time()
            params = {"tree_method": "hist", "max_bin": 64, "nthread": 4, "seed": SEED + rf.test[0], "verbosity": 0,
                      "objective": "reg:squarederror", "max_depth": d, "learning_rate": lr, "min_child_weight": mcw,
                      "subsample": 0.8, "colsample_bytree": 0.8}
            bst = xgb.train(params, dfit, num_boost_round=nmax)
            ent = {}
            for n in sizes:
                pv = bst.predict(dval, iteration_range=(0, n)).astype(np.float64)
                pt = bst.predict(dte, iteration_range=(0, n)).astype(np.float64)
                try:
                    g.assert_nondegenerate(pt, panel.bounds(ti), f"xgb {d}/{lr}/{mcw}/{n}")
                    g.assert_nondegenerate(pv, panel.bounds(vi), f"xgb val {d}/{lr}/{mcw}/{n}")
                    ent[str(n)] = {"val_ic": tsv(pv), "ic_all": ys.ic_all(pt).tolist(), "ic_liq": ys.ic_liq(pt).tolist()}
                except g.DegeneratePredictor as exc:
                    ent[str(n)] = {"degenerate": str(exc)}
            store[f"{rf.year}|{d}_{lr}_{mcw}"] = ent
            save_ck("e3", ck)
            log(f"  xgb {rf.year} d{d} lr{lr} mcw{mcw}: {time.time() - t0:.0f}s  "
                + " ".join(f"{n}:{np.mean(ent[str(n)]['ic_all']):+.4f}" for n in sizes if 'ic_all' in ent[str(n)]))
        del dfit, dval, dte


def mlp_cfgs(smoke: bool):
    if smoke:
        return [(16, 1, 0.0), (64, 2, 0.3)]
    return [(w, dep, dr) for w in MLP_WIDTHS for dep in MLP_DEPTHS for dr in MLP_DROPS]


def _mlp_job(paths: dict, width: int, depth: int, dropout: float, snaps: tuple, seed: int, lr: float = 1e-3,
             batch: int = 2048, threads: int = 1) -> dict:
    """Plain MLP (Linear-ReLU-Dropout x depth), MSE on y_rank, Adam, FIXED epochs; predictions at snapshot epochs.
    Runs in a loky worker (torch never shares a process with XGBoost)."""
    import numpy as _np
    import torch
    import torch.nn as nn

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = _np.random.default_rng(seed)
    Xf = _np.load(paths["Xf"], mmap_mode="r")
    yf = _np.load(paths["yf"], mmap_mode="r")
    Xv = _np.load(paths["Xv"], mmap_mode="r")
    Xt = _np.load(paths["Xt"], mmap_mode="r")
    p = Xf.shape[1]
    layers, w_in = [], p
    for _ in range(depth):
        layers += [nn.Linear(w_in, width), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        w_in = width
    layers.append(nn.Linear(w_in, 1))
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    X = torch.from_numpy(_np.ascontiguousarray(Xf, dtype=_np.float32))
    Y = torch.from_numpy(_np.ascontiguousarray(yf, dtype=_np.float32))
    TV = torch.from_numpy(_np.ascontiguousarray(Xv, dtype=_np.float32))
    TT = torch.from_numpy(_np.ascontiguousarray(Xt, dtype=_np.float32))
    out = {}
    for ep in range(1, max(snaps) + 1):
        net.train()
        perm = torch.from_numpy(rng.permutation(len(X)))
        for a in range(0, len(X), batch):
            idx = perm[a:a + batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = ((net(X[idx]).squeeze(-1) - Y[idx]) ** 2).mean()
            loss.backward()
            opt.step()
        if ep in snaps:
            net.eval()
            with torch.no_grad():
                out[ep] = (net(TV).squeeze(-1).numpy().astype(_np.float64), net(TT).squeeze(-1).numpy().astype(_np.float64))
    return out


def e3_mlp(panel: Panel, ck: dict, years: list[int], smoke: bool, seeds: tuple = (0,), tag: str = "mlp",
           cfgs=None, n_jobs: int = 4) -> None:
    snaps = (5, 10) if smoke else MLP_EPOCHS
    store = ck["cells"].setdefault(tag, {})
    cfgs = cfgs if cfgs is not None else mlp_cfgs(smoke)
    wd = ml._Workdir()
    try:
        for rf in make_refits(years, 1):
            jobs, keys = [], []
            fi, vi, ti = panel.rows(rf.fit), panel.rows(rf.val), panel.rows(rf.test)
            paths = None
            for (w, dep, dr) in cfgs:
                for s in seeds:
                    key = f"{rf.year}|{w}_{dep}_{dr}_s{s}"
                    if key in store:
                        continue
                    if paths is None:
                        paths = {"Xf": wd.save(panel.X[fi], "Xf"), "yf": wd.save(panel.y[fi].astype(np.float32), "yf"),
                                 "Xv": wd.save(panel.X[vi], "Xv"), "Xt": wd.save(panel.X[ti], "Xt")}
                    jobs.append({"paths": paths, "width": w, "depth": dep, "dropout": dr, "snaps": snaps,
                                 "seed": SEED + rf.test[0] + 17 * s + w + dep})
                    keys.append(key)
            if not jobs:
                continue
            t0 = time.time()
            res = ml._run_jobs(_mlp_job, jobs, n_jobs)
            tsv = g.TieSafeIC(panel.y[vi], panel.bounds(vi))
            ys = panel.year_scorer(rf.year)
            for key, r in zip(keys, res, strict=True):
                ent = {}
                for ep, (pv, pt) in r.items():
                    try:
                        g.assert_nondegenerate(pt, panel.bounds(ti), f"mlp {key} {ep}")
                        g.assert_nondegenerate(pv, panel.bounds(vi), f"mlp val {key} {ep}")
                        ent[str(ep)] = {"val_ic": tsv(pv), "ic_all": ys.ic_all(pt).tolist(), "ic_liq": ys.ic_liq(pt).tolist()}
                    except g.DegeneratePredictor as exc:
                        ent[str(ep)] = {"degenerate": str(exc)}
                store[key] = ent
            save_ck("e3", ck)
            log(f"  {tag} {rf.year}: {len(jobs)} trainings in {time.time() - t0:.0f}s")
    finally:
        wd.clear()


def xgb_seed_runs(panel: Panel, ck: dict, years: list[int], cfg: tuple, n: int, seeds: int) -> None:
    import xgboost as xgb

    store = ck["cells"].setdefault("xgb_seeds", {})
    d, lr, mcw = cfg
    for rf in make_refits(years, 1):
        fi, vi, ti = panel.rows(rf.fit), panel.rows(rf.val), panel.rows(rf.test)
        dfit = xgb.DMatrix(panel.X[fi], label=panel.y[fi], nthread=4)
        dte = xgb.DMatrix(panel.X[ti], nthread=4)
        ys = panel.year_scorer(rf.year)
        for s in range(seeds):
            key = f"{rf.year}|{d}_{lr}_{mcw}_{n}_s{s}"
            if key in store:
                continue
            params = {"tree_method": "hist", "max_bin": 64, "nthread": 4, "seed": SEED + rf.test[0] + 1000 * (s + 1), "verbosity": 0,
                      "objective": "reg:squarederror", "max_depth": d, "learning_rate": lr, "min_child_weight": mcw,
                      "subsample": 0.8, "colsample_bytree": 0.8}
            bst = xgb.train(params, dfit, num_boost_round=n)
            store[key] = {"ic_all": ys.ic_all(bst.predict(dte).astype(np.float64)).tolist()}
        save_ck("e3", ck)
    log(f"  xgb seed runs for {cfg} n={n} done")


def spa_pvalues(D: np.ndarray, cl: np.ndarray, B: int = N_BOOT) -> dict:
    """Hansen (2005) SPA (consistent recentring) and studentised max-t (White RC) p-values for H0: no model beats the
    benchmark, on D = (models x months) paired IC differences (model - benchmark), cluster bootstrap by refit-year."""
    K, N = D.shape
    u, inv = np.unique(cl, return_inverse=True)
    G = len(u)
    cnt = np.bincount(inv, minlength=G).astype(float)
    sums = np.stack([np.bincount(inv, weights=D[k], minlength=G) for k in range(K)])  # (K, G)
    dbar = sums.sum(1) / N
    idx = boot_idx(G, B)
    bm = np.einsum("kbg->bk", sums[:, idx]) / cnt[idx].sum(1)[:, None]  # (B, K)
    se = bm.std(axis=0, ddof=1)
    se = np.where(se > 0, se, np.nan)
    t = dbar / se
    T = max(float(np.nanmax(t)), 0.0)
    thr = -math.sqrt(2 * math.log(math.log(N)))
    mu_c = np.where(t <= thr, dbar, 0.0)
    Zc = (bm - dbar + mu_c) / se
    Zr = (bm - dbar) / se
    p_spa = float(np.mean(np.maximum(np.nanmax(Zc, axis=1), 0.0) >= T))
    p_max = float(np.mean(np.maximum(np.nanmax(Zr, axis=1), 0.0) >= T))
    return {"K": K, "T_stat": T, "p_spa_c": p_spa, "p_maxt": p_max, "best_k": int(np.nanargmax(t))}


def analyse_e3(ck: dict, smoke: bool = False) -> None:
    out: dict = {}
    base_ck = None
    p1 = RESULTS / "p9_e1.json"
    if SMOKE:
        p1 = cache_dir() / "p9_e1.json"
    if p1.exists():
        base_ck = json.loads(p1.read_text())
    ref = base_ck["cells"].get("base|M1") if base_ck else None
    if ref is None or "ic_all" not in ref:
        log("E3 analysis needs the E1 base|M1 cell (the study-13 ridge reference); run e1 first.")
        return
    cl = np.asarray(ref["cluster"])
    ridge_ref = np.asarray(ref["ic_all"])
    ridge_ref_liq = np.asarray(ref["ic_liq"])
    out["reference_ridge_M1_study13_grid"] = cstats(ridge_ref, cl)
    # ---- ridge sweep ----
    r = ck["cells"].get("ridge")
    if r:
        rows = []
        for k, a in enumerate(r["alphas"]):
            st = cstats(r["ic_all"][k], r["cluster"])
            rows.append([f"{a:g}" + (" (ext)" if a in RIDGE_EXT else ""), f3(st["mean"]), fmt_ci(st)])
        sel = {}
        for nm, cols_ in (("sel8", 8), ("sel10", 10)):
            ser, serl = [], []
            for yi, y in enumerate(sorted(set(r["cluster"]))):
                vi_ = int(np.argmax(r["val_ic"][yi][:cols_]))
                msk = np.asarray(r["cluster"]) == y
                ser += np.asarray(r["ic_all"][vi_])[msk].tolist()
                serl += np.asarray(r["ic_liq"][vi_])[msk].tolist()
            sel[nm] = (np.asarray(ser), np.asarray(serl))
        out["ridge"] = {"curve": {f"{a:g}": cstats(r["ic_all"][k], r["cluster"]) for k, a in enumerate(r["alphas"])},
                        "selected_causal_8pt": cstats(sel["sel8"][0], r["cluster"]),
                        "selected_causal_10pt": cstats(sel["sel10"][0], r["cluster"]),
                        "d_sel8_vs_ref": cstats(sel["sel8"][0] - ridge_ref, cl)}
        append_table("e3", "E3 ridge lambda sweep (no selection; OOS IC ALL)", md_table(["lambda", "IC", "95% CI"], rows))
    fam_series: dict[str, dict] = {}
    # ---- xgb ----
    xs = ck["cells"].get("xgb")
    years_done = sorted({int(k.split("|")[0]) for k in xs}) if xs else []
    if xs and years_done:
        cfgs = sorted({tuple(k.split("|")[1].split("_")) for k in xs})
        sizes = sorted({int(n) for e in xs.values() for n in e})
        cells = {}
        for cfg in cfgs:
            for n in sizes:
                ia, il, vic = [], [], []
                okc = True
                for y in years_done:
                    e = xs.get(f"{y}|{'_'.join(cfg)}", {}).get(str(n))
                    if not e or "ic_all" not in e:
                        okc = False
                        break
                    ia += e["ic_all"]
                    il += e["ic_liq"]
                    vic.append(e["val_ic"])
                if okc:
                    cells[(cfg, n)] = {"ic_all": np.asarray(ia), "ic_liq": np.asarray(il), "val_ic": vic}
        fam_series["xgb"] = {"cells": cells, "years": years_done}
    ml_ = ck["cells"].get("mlp")
    if ml_:
        years_done_m = sorted({int(k.split("|")[0]) for k in ml_})
        cfgs = sorted({k.split("|")[1] for k in ml_ if k.endswith("_s0")})
        cells = {}
        for cfgk in cfgs:
            for ep in (5, 10, 20, 40):
                ia, il, vic = [], [], []
                okc = True
                for y in years_done_m:
                    e = ml_.get(f"{y}|{cfgk}", {}).get(str(ep))
                    if not e or "ic_all" not in e:
                        okc = False
                        break
                    ia += e["ic_all"]
                    il += e["ic_liq"]
                    vic.append(e["val_ic"])
                if okc:
                    cells[(cfgk, ep)] = {"ic_all": np.asarray(ia), "ic_liq": np.asarray(il), "val_ic": vic}
        fam_series["mlp"] = {"cells": cells, "years": years_done_m}
    summary_rows = []
    for fam, d in fam_series.items():
        cells = d["cells"]
        yrs = d["years"]
        if not cells or len(yrs) < 2:
            continue
        L = len(next(iter(cells.values()))["ic_all"])
        cl_f = np.concatenate([[y] * (12 if y != 2018 else 11) for y in yrs])[:L]
        ref_f = ridge_ref[:L]
        # causal selection per refit among all (config, size) cells
        keys = list(cells)
        sel_all, sel_liq, chosen = [], [], []
        pos = 0
        for yi, y in enumerate(yrs):
            nm = 12 if y != 2018 else 11
            kbest = max(keys, key=lambda k: cells[k]["val_ic"][yi])
            sel_all += cells[kbest]["ic_all"][pos:pos + nm].tolist()
            sel_liq += cells[kbest]["ic_liq"][pos:pos + nm].tolist()
            chosen.append({"year": y, "cell": str(kbest), "val_ic": cells[kbest]["val_ic"][yi]})
            pos += nm
        sel_all, sel_liq = np.asarray(sel_all), np.asarray(sel_liq)
        st_sel = cstats(sel_all, cl_f)
        dd = cstats(sel_all - ref_f, cl_f)
        v = delta_verdict(dd)
        # oracle ceiling
        means = {k: float(np.mean(cells[k]["ic_all"])) for k in keys}
        kb = max(means, key=means.get)
        d_or = cstats(cells[kb]["ic_all"] - ref_f, cl_f)
        Dmat = np.stack([cells[k]["ic_all"] - ref_f for k in keys])
        spa = spa_pvalues(Dmat, cl_f)
        v_or = delta_verdict(d_or)
        # conservative second reference: the best-OOS ridge lambda (selection-biased in ridge's favour)
        d_vs_best_ridge = v_vs_best_ridge = None
        if r:
            best_r = int(np.argmax([np.mean(x) for x in r["ic_all"]]))
            rb = np.asarray(r["ic_all"][best_r])[:L]
            d_vs_best_ridge = cstats(sel_all - rb, cl_f)
            v_vs_best_ridge = delta_verdict(d_vs_best_ridge)
        out[fam] = {"causal_selected": st_sel, "d_vs_ridge": dd, "verdict": v, "chosen_per_refit": chosen,
                    "oracle_best_cell": str(kb), "oracle_ic": cstats(cells[kb]["ic_all"], cl_f), "oracle_d_vs_ridge": d_or,
                    "oracle_verdict_selection_biased": v_or, "d_vs_best_oos_ridge": d_vs_best_ridge,
                    "verdict_vs_best_oos_ridge": v_vs_best_ridge, "spa": spa, "n_cells": len(keys),
                    "mean_ic_all_cells": {str(k): v_ for k, v_ in means.items()}}
        summary_rows.append([fam, "causal per-refit selection (val IC)", f3(st_sel["mean"]), fmt_ci(st_sel), f3(dd["mean"]), fmt_ci(dd), v])
        if d_vs_best_ridge is not None:
            summary_rows.append([fam, "causal selection vs BEST-OOS ridge lambda (ridge-favouring reference)", f3(st_sel["mean"]), fmt_ci(st_sel),
                                 f3(d_vs_best_ridge["mean"]), fmt_ci(d_vs_best_ridge), v_vs_best_ridge])
        summary_rows.append([fam, f"ORACLE best OOS cell {kb} (selection-biased in its own favour; n_cells={len(keys)})",
                             f3(out[fam]['oracle_ic']['mean']), fmt_ci(out[fam]['oracle_ic']), f3(d_or["mean"]), fmt_ci(d_or),
                             f"SPA_c p={spa['p_spa_c']:.3f}, max-t p={spa['p_maxt']:.3f}; {v_or}"])
        # capacity curves (mean IC by size / epochs)
        if fam == "xgb":
            curve = {}
            for cfg in sorted({k[0] for k in keys}):
                curve["_".join(cfg)] = {str(n): float(np.mean(cells[(cfg, n)]["ic_all"])) for n in sorted({k[1] for k in keys}) if (cfg, n) in cells}
            out[fam]["capacity_curve_mean_ic"] = curve
        else:
            curve = {}
            for cfgk in sorted({k[0] for k in keys}):
                curve[cfgk] = {str(e): float(np.mean(cells[(cfgk, e)]["ic_all"])) for e in (5, 10, 20, 40) if (cfgk, e) in cells}
            out[fam]["capacity_curve_mean_ic"] = curve
        # per-family final-cell seeds
    # pooled SPA over every non-ridge cell
    allD, allk = [], []
    for fam, d in fam_series.items():
        if len(d["years"]) < 10 and not smoke:
            continue
        for k, v_ in d["cells"].items():
            if len(v_["ic_all"]) == len(ridge_ref):
                allD.append(v_["ic_all"] - ridge_ref)
                allk.append((fam, k))
    if allD and len(allD) > 1:
        out["pooled_spa_all_flexible_cells"] = {**spa_pvalues(np.stack(allD), cl), "n_cells": len(allD)}
    # final-cell seed spread
    for tag, fam in (("xgb_seeds", "xgb"), ("mlp_seeds", "mlp")):
        s = ck["cells"].get(tag)
        if not s:
            continue
        by_seed: dict[str, dict] = {}
        for k, v_ in s.items():
            y_, rest = k.split("|")
            sd_ = rest.split("_s")[-1]
            ent = v_["ic_all"] if "ic_all" in v_ else None
            if ent is None:
                continue
            by_seed.setdefault(sd_, {})[int(y_)] = ent
        seeds_ic = {sd_: np.concatenate([by_seed[sd_][y] for y in sorted(by_seed[sd_])]) for sd_ in by_seed if len(by_seed[sd_]) >= 10 or smoke}
        if seeds_ic:
            m_ = np.array([v_.mean() for v_ in seeds_ic.values()])
            out.setdefault(f"{fam}_final_cell_seeds", {"n_seeds": len(m_), "mean_ic_per_seed": m_.tolist(),
                                                       "sd_across_seeds": float(m_.std(ddof=1)) if len(m_) > 1 else None})
    if summary_rows:
        append_table("e3", f"E3 capacity verdicts (paired dIC_t = IC_flex - IC_ridge(study-13 M1 grid); delta {DELTA:+.3f})",
                     md_table(["family", "arm", "IC ALL", "95% CI", "paired dIC", "95% CI of dIC", "pre-declared verdict"], summary_rows))
    if "pooled_spa_all_flexible_cells" in out:
        append_table("e3", "E3 pooled SPA over every flexible cell vs ridge", json.dumps(out["pooled_spa_all_flexible_cells"], indent=1))
    ck["analysis"] = out


def run_e3(args) -> None:
    panel = Panel()
    years = [2012] if SMOKE else OOS_YEARS
    parts = set(args.parts.split(","))
    ck = load_ck("e3")
    ck["meta"] = {"years": years, "smoke": SMOKE, "delta": DELTA, "sealed_load_count": panel.sealed_load_count}
    if "ridge" in parts:
        e3_ridge(panel, ck, years)
    if "xgb" in parts:
        e3_xgb(panel, ck, years, SMOKE)
    if "mlp" in parts:
        e3_mlp(panel, ck, years, SMOKE)
    if "seeds" in parts:
        for fam in ("xgb", "mlp"):
            fc = final_cell(ck, fam)
            if fc is None:
                continue
            if fam == "xgb":
                xgb_seed_runs(panel, ck, years, fc[0], fc[1], 2 if SMOKE else 5)
            else:
                w, dep, dr = fc[0].split("_")
                e3_mlp_final(panel, ck, years, (int(w), int(dep), float(dr)), fc[1])
    analyse_e3(ck, SMOKE)
    save_ck("e3", ck)


def final_cell(ck: dict, fam: str):
    """Global 'final cell' per family: highest mean validation IC across the refits (a causal-ish pick, one cell)."""
    if fam == "xgb":
        xs = ck["cells"].get("xgb")
        if not xs:
            return None
        agg: dict = {}
        for k, e in xs.items():
            y, cfg = k.split("|")
            for n, v in e.items():
                if "val_ic" in v:
                    agg.setdefault((tuple(cfg.split("_")), int(n)), []).append(v["val_ic"])
        agg = {k: np.mean(v) for k, v in agg.items() if len(v) >= (1 if SMOKE else 10)}
        if not agg:
            return None
        cfg, n = max(agg, key=agg.get)
        return ((int(cfg[0]), float(cfg[1]), int(cfg[2])), n)
    ml_ = ck["cells"].get("mlp")
    if not ml_:
        return None
    agg = {}
    for k, e in ml_.items():
        y, cfg = k.split("|")
        if not cfg.endswith("_s0"):
            continue
        for ep, v in e.items():
            if "val_ic" in v:
                agg.setdefault((cfg[:-3], int(ep)), []).append(v["val_ic"])
    agg = {k: np.mean(v) for k, v in agg.items() if len(v) >= (1 if SMOKE else 10)}
    if not agg:
        return None
    return max(agg, key=agg.get)


def e3_mlp_final(panel: Panel, ck: dict, years: list[int], cfg: tuple, epoch: int) -> None:
    """>=5 seeds of the single 'final' MLP cell; per-seed OOS IC series at the chosen epoch."""
    n_seeds = 2 if SMOKE else 5
    e3_mlp(panel, ck, years, SMOKE, seeds=tuple(range(n_seeds)), tag="mlp_seed_raw", cfgs=[cfg])
    store = ck["cells"].setdefault("mlp_seeds", {})
    for k, e in ck["cells"]["mlp_seed_raw"].items():
        if str(epoch) in e and "ic_all" in e[str(epoch)]:
            store[k] = {"ic_all": e[str(epoch)]["ic_all"]}
    save_ck("e3", ck)


# ------------------------------------------------------------------------------------------------
# E4: detectability (planted signal on the REAL y_rank)
# ------------------------------------------------------------------------------------------------
class PlantEngine:
    def __init__(self, panel: Panel, years: list[int]):
        self.panel = panel
        self.years = years
        v = panel.valid
        self.vi = np.flatnonzero(v)
        self.Xv = panel.X[self.vi]
        self.y = panel.y[self.vi]
        self.date = panel.date_id[self.vi]
        self.tmi = panel.tmi[self.vi]
        self.b = ml.date_bounds(self.date)
        self.nd = len(self.b) - 1
        self.blk = np.repeat(np.arange(self.nd), np.diff(self.b))
        self.pos_in = np.arange(len(self.y)) - self.b[self.blk]
        self.n_in = np.diff(self.b)[self.blk]
        self.refits = make_refits(years, 1)
        self.p = self.Xv.shape[1]
        # per refit slices (indices into the valid-only arrays)
        remap = -np.ones(panel.n, np.int64)
        remap[self.vi] = np.arange(len(self.vi))
        self.slices = []
        for rf in self.refits:
            fi, vi_, ti = (remap[panel.rows(rf.fit)], remap[panel.rows(rf.val)], remap[panel.rows(rf.test)])
            self.slices.append((rf, fi, vi_, ti))
        self.ridge_eigs = {}
        for rf, fi, vi_, ti in self.slices:
            Xf = self.Xv[fi]
            eig = RidgeEig(Xf)
            self.ridge_eigs[rf.year] = (eig, (Xf.astype(np.float64) - eig.mu).astype(np.float32))
        self.month_cluster = np.array([self.tmi[self.b[i]] // 12 for i in range(self.nd)])
        self.month_tmi = np.array([self.tmi[self.b[i]] for i in range(self.nd)])

    def rank_unit(self, z: np.ndarray) -> np.ndarray:
        order = np.lexsort((z, self.date))
        r = np.empty(len(z))
        r[order] = self.pos_in  # sorted order keeps date blocks aligned with self.b
        return r / np.maximum(self.n_in - 1, 1) - 0.5

    def norm_ranks(self, z: np.ndarray) -> np.ndarray:
        """Within-date centred, unit-norm AVERAGE ranks (Spearman as a dot product; tie-safe; zero-norm date -> 0)."""
        out = np.zeros(len(z))
        for a_, b_ in zip(self.b[:-1], self.b[1:], strict=True):
            r = rankdata(z[a_:b_])
            r = r - r.mean()
            n = math.sqrt(float((r * r).sum()))
            if n > 0:
                out[a_:b_] = r / n
        return out

    def calibrate(self, s: np.ndarray, rho: float, active: np.ndarray | None) -> float:
        """delta so that the mean over (active) months of the within-month Pearson(s, y + delta s) EXCEEDS its value at
        delta = 0 (a random direction already has a small chance IC with the real y) by rho_eff, where rho_eff = rho when
        planting in every month and rho * n_months / n_active when planting in a subset (so rho is always the average
        planted IC over ALL months)."""
        cov = np.add.reduceat(self.y * s, self.b[:-1]) / np.diff(self.b)
        vy = np.add.reduceat(self.y**2, self.b[:-1]) / np.diff(self.b)
        vs = np.add.reduceat(s**2, self.b[:-1]) / np.diff(self.b)
        act = np.ones(self.nd, bool) if active is None else active

        def f(dl):
            with np.errstate(all="ignore"):
                r = (cov + dl * vs) / (np.sqrt(vs) * np.sqrt(vy + 2 * dl * cov + dl * dl * vs))
            return float(np.nan_to_num(r[act]).mean())

        rho_eff = rho if active is None else rho * self.nd / max(int(act.sum()), 1)
        target = f(0.0) + rho_eff
        lo, hi = 0.0, 200.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if f(mid) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def realised(self, s: np.ndarray, yp: np.ndarray, active) -> float:
        """Mean over ALL months of the within-month Spearman(s, yp) (inactive months contribute 0)."""
        per = np.add.reduceat(self.norm_ranks(s) * self.norm_ranks(yp), self.b[:-1])
        if active is not None:
            per = np.where(active, per, 0.0)
        return float(per.sum() / self.nd)

    def plant(self, rho: float, rng: np.random.Generator, frac_present: float | None):
        w = rng.normal(size=self.p)
        w /= np.linalg.norm(w)
        z = self.Xv.astype(np.float64) @ w
        s = self.rank_unit(z)
        active = None
        if frac_present is not None:
            active = np.zeros(self.nd, bool)
            active[rng.choice(self.nd, size=int(round(frac_present * self.nd)), replace=False)] = True
            s = np.where(active[self.blk], s, 0.0)
        if rho <= 0:
            return self.y.copy(), 0.0, 0.0
        base = self.realised(s, self.y, active)
        dl = self.calibrate(s, rho, active)
        yp = self.y + dl * s
        rr = self.realised(s, yp, active) - base
        if rr > 0:  # one refinement so that the realised ADDED Spearman average hits rho
            dl *= rho / rr
            yp = self.y + dl * s
            rr = self.realised(s, yp, active) - base
        return yp, dl, rr

    def fit_score(self, YP: np.ndarray) -> dict:
        """YP: (n_valid, D) planted targets.  Ridge-all (M1 alphas) per refit, causal alpha by tie-safe-equivalent val IC,
        OOS IC against the planted ranks.  Returns per-draw monthly IC series (D, months) and chosen alphas."""
        D = YP.shape[1]
        YR = np.stack([self.norm_ranks(YP[:, d]) for d in range(D)], axis=1)  # (n, D)
        ic = np.zeros((D, self.nd))
        alphas = np.asarray(M1_ALPHAS)
        sel = np.zeros((D, len(self.refits)), int)
        for ri, (rf, fi, vi_, ti) in enumerate(self.slices):
            eig, Xc = self.ridge_eigs[rf.year]
            Yf = YP[fi]
            q = (Xc.T @ (Yf - Yf.mean(0)).astype(np.float32)).astype(np.float64)
            W = eig.weights(q, alphas)  # (p, A, D)
            b0 = Yf.mean(0)
            Xvc = self.Xv[vi_].astype(np.float64) - eig.mu
            Xtc = self.Xv[ti].astype(np.float64) - eig.mu
            bv = ml.date_bounds(self.date[vi_])
            bt = ml.date_bounds(self.date[ti])
            Pv = np.einsum("np,pad->nad", Xvc, W)
            Pt = np.einsum("np,pad->nad", Xtc, W)
            Yv_r = YR[vi_]
            for d in range(D):
                vic = np.zeros(len(alphas))
                for a_, bb in zip(bv[:-1], bv[1:], strict=True):
                    pr = rankdata(Pv[a_:bb, :, d], axis=0)
                    pr = pr - pr.mean(0)
                    pr /= np.sqrt((pr * pr).sum(0))
                    # validation labels: tie-safe average ranks of the planted target
                    vic += pr.T @ Yv_r[a_:bb, d]
                k = int(np.argmax(vic))
                sel[d, ri] = k
                for a_, bb in zip(bt[:-1], bt[1:], strict=True):
                    pr = rankdata(Pt[a_:bb, k, d])
                    pr = pr - pr.mean()
                    pr /= math.sqrt(float((pr * pr).sum()))
                    # month index of this block
                    mi_ = int(np.searchsorted(self.b, ti[a_], side="right") - 1)
                    ic[d, mi_] = float(pr @ YR[ti[a_:bb], d])
        return {"ic": ic, "sel": sel}


def run_e4(args) -> None:
    panel = Panel()
    years = [2012] if SMOKE else OOS_YEARS
    ck = load_ck("e4")
    ck["meta"] = {"years": years, "smoke": SMOKE, "sealed_load_count": panel.sealed_load_count,
                  "scoring": "IC vs the PLANTED target ranks; power = share of draws with |cluster t| > t_{0.975, 9}"}
    eng = PlantEngine(panel, years)
    oos_months = np.flatnonzero(np.isin(eng.month_cluster, years))
    cl = eng.month_cluster[oos_months]
    n_draw = 4 if SMOKE else 100
    rhos = (0.0, 0.03) if SMOKE else PLANT_RHOS
    kinds = (("stationary", None), ("nonstationary30", 0.30))
    batch = 10 if not SMOKE else 4
    for kind, frac in kinds:
        for ri, rho in enumerate(rhos):
            key = f"{kind}|{rho:g}"
            if key in ck["cells"] and ck["cells"][key].get("n_draws", 0) >= n_draw:
                continue
            if rho == 0 and kind != "stationary":
                continue
            t0 = time.time()
            rng = np.random.default_rng([SEED, 4, ri, 0 if frac is None else 1])
            nd = 1 if rho == 0 else n_draw
            ics, deltas, realised = [], [], []
            for b0 in range(0, nd, batch):
                nb = min(batch, nd - b0)
                cols = []
                for _ in range(nb):
                    yp, dl, rr = eng.plant(rho, rng, frac)
                    cols.append(yp)
                    deltas.append(dl)
                    realised.append(rr)
                res = eng.fit_score(np.stack(cols, axis=1))
                ics.append(res["ic"][:, oos_months])
            ic = np.concatenate(ics, axis=0)
            per_draw = [cstats(ic[d], cl) for d in range(len(ic))]
            tcrit = per_draw[0]["t_crit"]
            reject = np.array([abs(p["t_cluster"]) > tcrit and p["mean"] > 0 for p in per_draw])
            reject_boot = np.array([p["ci_boot"][0] > 0 for p in per_draw])
            ck["cells"][key] = {"kind": kind, "rho": rho, "n_draws": len(ic), "power_cluster_t": float(reject.mean()),
                                "power_boot_ci": float(reject_boot.mean()),
                                "mean_ic": float(np.mean([p["mean"] for p in per_draw])),
                                "sd_ic_across_draws": float(np.std([p["mean"] for p in per_draw], ddof=1)) if len(ic) > 1 else None,
                                "mean_t_cluster": float(np.mean([p["t_cluster"] for p in per_draw])),
                                "realised_added_ic_mean": float(np.mean(realised)), "delta_mean": float(np.mean(deltas)),
                                "recovery_ratio": float(np.mean([p["mean"] for p in per_draw]) / rho) if rho > 0 else None,
                                "ic_series_draw0": ic[0].tolist(), "months_cluster": cl.tolist()}
            save_ck("e4", ck)
            log(f"  E4 {key:<22} power(t) {reject.mean():.2f}  power(boot) {reject_boot.mean():.2f}  mean IC "
                f"{ck['cells'][key]['mean_ic']:+.4f}  realised rho {np.mean(realised):+.4f}  ({time.time() - t0:.0f}s)")
    analyse_e4(ck)
    save_ck("e4", ck)


def mde80(points: list[tuple[float, float]]) -> float | None:
    pts = sorted(points)
    for (r0, p0), (r1, p1) in zip(pts[:-1], pts[1:]):
        if p0 < 0.8 <= p1:
            return r0 + (0.8 - p0) / (p1 - p0) * (r1 - r0)
    return pts[0][0] if pts and pts[0][1] >= 0.8 else None


def analyse_e4(ck: dict) -> None:
    out: dict = {}
    for kind in ("stationary", "nonstationary30"):
        for key_p in ("power_cluster_t", "power_boot_ci"):
            pts = [(c["rho"], c[key_p]) for k, c in ck["cells"].items() if c.get("kind") == kind and c["rho"] > 0]
            out.setdefault(kind, {})[f"mde80_{key_p}"] = mde80(pts)
    rows = []
    for kind in ("stationary", "nonstationary30"):
        for rho in sorted({c["rho"] for c in ck["cells"].values() if c.get("kind") == kind}):
            c = ck["cells"].get(f"{kind}|{rho:g}")
            if c is None:
                continue
            rows.append([kind, f"{rho:g}", c["n_draws"], f"{c['realised_added_ic_mean']:+.4f}", f3(c["mean_ic"]),
                         "" if c["recovery_ratio"] is None else f"{c['recovery_ratio']:.2f}", f"{c['power_cluster_t']:.2f}",
                         f"{c['power_boot_ci']:.2f}"])
    if rows:
        s = out.get("stationary", {})
        n = out.get("nonstationary30", {})
        append_table("e4", "E4 detectability (ridge-all on real y_rank + planted signal)",
                     md_table(["plant", "target rho", "draws", "realised added planted IC", "mean OOS IC", "recovery (IC/rho)",
                               "power (cluster t, 5%)", "power (bootstrap CI>0)"], rows)
                     + f"\n\nMDE80 stationary (UPPER BOUND on detectability): cluster-t {s.get('mde80_power_cluster_t')}, "
                       f"bootstrap {s.get('mde80_power_boot_ci')}\nMDE80 non-stationary (30% of months): cluster-t "
                       f"{n.get('mde80_power_cluster_t')}, bootstrap {n.get('mde80_power_boot_ci')}")
    # equivalence statements from the E1 baselines
    p1 = (cache_dir() / "p9_e1.json") if SMOKE else (RESULTS / "p9_e1.json")
    if p1.exists():
        e1 = json.loads(p1.read_text())
        eq = {}
        series = {}
        for m in ("M1", "M2", "M3"):
            c = e1["cells"].get(f"base|{m}")
            if c and "ic_all" in c:
                series[m] = (np.asarray(c["ic_all"]), np.asarray(c["cluster"]))
        for m, (x, cl_) in series.items():
            st = cstats(x, cl_)
            eq[m] = {"mean_ic": st["mean"], "ci_boot": st["ci_boot"], "ci_t": st["ci_t"],
                     "statement": f"the OOS mean IC of {st['mean']:.3f} has a 95% CI {fmt_ci(st)} (bootstrap) / {fmt_ci(st, 'ci_t')} "
                                  f"(cluster t); an IC >= {st['ci_t'][1]:.3f} cannot be ruled out"}
        out["equivalence"] = eq
        if eq:
            append_table("e4", "E4 equivalence statements for the realised E1 baseline ICs (ALL universe)",
                         "\n".join(f"- {m}: {v['statement']}" for m, v in eq.items()))
    ck["analysis"] = out


def run_refresh_portfolios(args) -> None:
    """Recompute the decile-L/S portfolio blocks (with the per-year table) for every cached cell of the named experiment."""
    panel = Panel()
    ck = load_ck(args.which)
    for key, c in ck["cells"].items():
        if not isinstance(c, dict) or "ic_all" not in c or "portfolio" not in c:
            continue
        cid, m = key.split("|")
        if args.only and cid not in args.only.split(","):
            continue
        res = cell_cached(cid, m, lambda: (_ for _ in ()).throw(RuntimeError("no cached preds")))
        years = sorted(res["pred"])
        rows = np.concatenate([panel.year_scorer(y).rows for y in years])
        pred = np.concatenate([res["pred"][y] for y in years])
        c["portfolio"] = {u: portfolio_stats(panel, rows, pred, u) for u in ("LIQ", "ALL")}
    save_ck(args.which, ck)
    log(f"portfolios refreshed for {args.which}")


def run_e4b(args) -> None:
    """Supplement to E4 (NOT in the pre-declaration, flagged as such): the same planted-signal power curve through the
    fixed booster M2 (the pre-fixed model that carries the largest realised IC), stationary plant, fewer draws."""
    import xgboost as xgb

    panel = Panel()
    years = [2012] if SMOKE else OOS_YEARS
    ck = load_ck("e4b")
    ck["meta"] = {"years": years, "smoke": SMOKE, "supplement": "booster M2 through the E4 plant (not pre-declared)"}
    eng = PlantEngine(panel, years)
    oos_months = np.flatnonzero(np.isin(eng.month_cluster, years))
    cl = eng.month_cluster[oos_months]
    n_draw = 2 if SMOKE else int(os.environ.get("P9_E4B_DRAWS", 20))
    rhos = (0.03,) if SMOKE else (0.01, 0.02, 0.03, 0.05)
    base_params = {"tree_method": "hist", "max_bin": 64, "nthread": 4, "verbosity": 0, **g.BOOSTER_FIXED,
                   "min_child_weight": g.BOOSTER_MCW["M2"], "objective": "reg:squarederror"}
    for ri, rho in enumerate((0.0, *rhos)):
        key = f"stationary|{rho:g}"
        if key in ck["cells"] and ck["cells"][key]["n_draws"] >= (1 if rho == 0 else n_draw):
            continue
        t0 = time.time()
        rng = np.random.default_rng([SEED, 44, ri])
        ics = []
        for d in range(1 if rho == 0 else n_draw):
            yp, dl, rr = eng.plant(rho, rng, None)
            yr = eng.norm_ranks(yp)
            ic = np.zeros(eng.nd)
            for rf, fi, vi_, ti in eng.slices:
                bst = xgb.train({**base_params, "seed": SEED + rf.test[0]}, xgb.DMatrix(eng.Xv[fi], label=yp[fi], nthread=4),
                                num_boost_round=g.BOOSTER_ROUNDS)
                pt = bst.predict(xgb.DMatrix(eng.Xv[ti], nthread=4)).astype(np.float64)
                bt = ml.date_bounds(eng.date[ti])
                for a_, bb in zip(bt[:-1], bt[1:], strict=True):
                    pr = rankdata(pt[a_:bb])
                    pr = pr - pr.mean()
                    nn = math.sqrt(float((pr * pr).sum()))
                    mi_ = int(np.searchsorted(eng.b, ti[a_], side="right") - 1)
                    ic[mi_] = float(pr @ yr[ti[a_:bb]]) / nn if nn > 0 else 0.0
            ics.append(ic[oos_months])
        per = [cstats(x, cl) for x in ics]
        tcrit = per[0]["t_crit"]
        rej = np.array([abs(p["t_cluster"]) > tcrit and p["mean"] > 0 for p in per])
        ck["cells"][key] = {"kind": "stationary", "rho": rho, "n_draws": len(ics), "power_cluster_t": float(rej.mean()),
                            "mean_ic": float(np.mean([p["mean"] for p in per])),
                            "sd_ic_across_draws": float(np.std([p["mean"] for p in per], ddof=1)) if len(ics) > 1 else None,
                            "mean_t_cluster": float(np.mean([p["t_cluster"] for p in per])),
                            "mean_se_cluster": float(np.mean([p["se_cluster"] for p in per]))}
        save_ck("e4b", ck)
        log(f"  E4b booster {key}: power {rej.mean():.2f} mean IC {ck['cells'][key]['mean_ic']:+.4f} mean cluster SE "
            f"{ck['cells'][key]['mean_se_cluster']:.4f} ({time.time() - t0:.0f}s)")
    pts = [(c["rho"], c["power_cluster_t"]) for c in ck["cells"].values() if c["rho"] > 0]
    base_ic = ck["cells"]["stationary|0"]["mean_ic"]
    ck["analysis"] = {"mde80_power_cluster_t": None, "note": "power is ~1.00 at every rho because the booster's unplanted IC already rejects "
                      "IC=0, so an MDE80 is meaningless here; read the recovered added IC instead",
                      "recovered_added_ic": {k: v["mean_ic"] - base_ic for k, v in ck["cells"].items() if v["rho"] > 0}}
    rows = [[c["rho"], c["n_draws"], f3(c["mean_ic"]), f3(c["mean_se_cluster"]), f"{c['power_cluster_t']:.2f}"]
            for c in sorted(ck["cells"].values(), key=lambda c: c["rho"])]
    append_table("e4b", "E4b supplement: fixed booster M2 through the planted-signal power curve (stationary plant)",
                 md_table(["added planted IC", "draws", "mean OOS IC", "mean cluster SE", "power (cluster t, 5%)"], rows)
                 + "\n\nrecovered added IC (mean OOS IC minus the unplanted 0.0314): "
                 + ", ".join(f"rho {k.split('|')[1]}: {v:+.4f}" for k, v in ck["analysis"]["recovered_added_ic"].items()))
    save_ck("e4b", ck)


# ------------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------------
def run_figs(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(exist_ok=True)
    e1 = json.loads((RESULTS / "p9_e1.json").read_text()) if (RESULTS / "p9_e1.json").exists() else None
    e2 = json.loads((RESULTS / "p9_e2.json").read_text()) if (RESULTS / "p9_e2.json").exists() else None
    e3 = json.loads((RESULTS / "p9_e3.json").read_text()) if (RESULTS / "p9_e3.json").exists() else None
    e4 = json.loads((RESULTS / "p9_e4.json").read_text()) if (RESULTS / "p9_e4.json").exists() else None
    col = {"M1": "#1f77b4", "M2": "#d62728", "M3": "#2ca02c"}
    if e1:
        fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
        a = e1.get("analysis", {})
        for m in ("M1", "M2", "M3"):
            sh = [np.mean(c["ic_all"]) for k, c in e1["cells"].items() if k.startswith("shuf") and k.endswith(f"|{m}") and "ic_all" in c]
            if sh:
                ax[0].hist(sh, bins=10, alpha=0.5, color=col[m], label=f"{m} (n={len(sh)})")
        ax[0].axvline(0, color="k", lw=1)
        ax[0].set_title("E1a label-shuffle OOS IC\n(integrity check, not a leakage test)")
        ax[0].set_xlabel("mean OOS IC")
        ax[0].legend()
        for m in ("M1", "M2", "M3"):
            d = a.get("e1b", {}).get(m)
            if not d:
                continue
            ks = [k for k in (-1, 0, 1, 2, 3) if str(k) in d]
            ax[1].errorbar(ks, [d[str(k)]["ic"]["mean"] for k in ks],
                           yerr=[[d[str(k)]["ic"]["mean"] - d[str(k)]["ic"]["ci_boot"][0] for k in ks],
                                 [d[str(k)]["ic"]["ci_boot"][1] - d[str(k)]["ic"]["mean"] for k in ks]],
                           color=col[m], marker="o", label=m, capsize=3)
        ax[1].set_yscale("symlog", linthresh=0.02)
        ax[1].set_title("E1b feature-lag ladder (features from T-k)\nk=-1 is the +1-month look-ahead sanity row")
        ax[1].set_xlabel("k")
        ax[1].set_ylabel("OOS IC (ALL)")
        ax[1].legend()
        for m in ("M1", "M2"):
            d = a.get("e1d", {}).get(m)
            if not d:
                continue
            lv = [k for k in d["levels"] if float(k) > 0]
            ax[2].plot([float(k) for k in lv], [d["levels"][k]["power_paired_up"] for k in lv], marker="o", color=col[m],
                       label=f"{m} paired dIC vs clean")
            ax[2].plot([float(k) for k in lv], [d["levels"][k]["power_unpaired"] or 0 for k in lv], marker="x", ls="--",
                       color=col[m], label=f"{m} vs unpaired SE")
        ax[2].axhline(0.8, color="k", ls=":", lw=1)
        ax[2].set_xscale("log")
        ax[2].set_title("E1d detection power of an injected leak\n(per-column injected IC; 3 columns)")
        ax[2].set_xlabel("injected single-feature IC")
        ax[2].set_ylabel("power")
        ax[2].legend(fontsize=7)
        labs, vals, lo, hi = [], [], [], []
        for m in ("M1", "M2", "M3"):
            c = a.get("e1c", {}).get(m, {})
            if "ablation_full_minus_ablated" in c:
                s = c["ablation_full_minus_ablated"]
                labs.append(f"{m}\nfull-ablated")
                vals.append(s["mean"])
                lo.append(s["mean"] - s["ci_boot"][0])
                hi.append(s["ci_boot"][1] - s["mean"])
            if "circshift" in c:
                s = c["circshift"]["cluster"]
                labs.append(f"{m}\ncirc-shift IC")
                vals.append(s["mean"])
                lo.append(s["mean"] - s["ci_boot"][0])
                hi.append(s["ci_boot"][1] - s["mean"])
        if labs:
            ax[3].errorbar(range(len(vals)), vals, yerr=[lo, hi], fmt="o", capsize=3)
            ax[3].set_xticks(range(len(vals)))
            ax[3].set_xticklabels(labs, fontsize=7)
            ax[3].axhline(0, color="k", lw=1)
        ax[3].set_title("E1c known-leak ablation / firm-fingerprint shift")
        fig.tight_layout()
        fig.savefig(FIG / "p9_leakage.png", dpi=130)
        plt.close(fig)
    if e2 and e2.get("analysis"):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for i, m in enumerate(("M1", "M2")):
            d = e2["analysis"].get(m, {})
            for lab, v in d.items():
                if not isinstance(v, dict) or "ic_all" not in v:
                    continue
                mk = "s" if "firms" in lab else ("^" if "refit" in lab else "o")
                ax[i].errorbar(v["n_fit_mean"], v["ic_all"]["mean"],
                               yerr=[[v["ic_all"]["mean"] - v["ic_all"]["ci_boot"][0]], [v["ic_all"]["ci_boot"][1] - v["ic_all"]["mean"]]],
                               marker=mk, capsize=3, label=lab)
            ax[i].set_xscale("log")
            ax[i].set_title(f"E2 learning curve, {m} (OOS IC vs ACTUAL fit rows)")
            ax[i].set_xlabel("mean fit rows per refit")
            ax[i].set_ylabel("OOS IC (ALL), 95% cluster CI")
            ax[i].axhline(0, color="k", lw=0.5)
            ax[i].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(FIG / "p9_learning_curve.png", dpi=130)
        plt.close(fig)
    if e3 and e3.get("analysis"):
        a = e3["analysis"]
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
        if "ridge" in a:
            ks = list(a["ridge"]["curve"])
            ax[0].errorbar(range(len(ks)), [a["ridge"]["curve"][k]["mean"] for k in ks],
                           yerr=[[a["ridge"]["curve"][k]["mean"] - a["ridge"]["curve"][k]["ci_boot"][0] for k in ks],
                                 [a["ridge"]["curve"][k]["ci_boot"][1] - a["ridge"]["curve"][k]["mean"] for k in ks]], marker="o", capsize=3)
            ax[0].set_xticks(range(len(ks)))
            ax[0].set_xticklabels(ks, rotation=45, fontsize=7)
            ax[0].set_title("E3 ridge: OOS IC vs lambda (no selection)")
            ax[0].set_ylabel("OOS IC (ALL)")
        if "xgb" in a:
            for k, cv in a["xgb"]["capacity_curve_mean_ic"].items():
                d_ = int(k.split("_")[0])
                ax[1].plot([int(n) for n in cv], list(cv.values()), alpha=0.6, color=plt.cm.viridis(d_ / 7), lw=1)
            ax[1].set_xscale("log")
            ax[1].set_title("E3 XGB no-selection capacity curve\n(24 configs; colour = depth)")
            ax[1].set_xlabel("trees (iteration_range)")
        if "mlp" in a:
            for k, cv in a["mlp"]["capacity_curve_mean_ic"].items():
                w_ = int(k.split("_")[0])
                ax[2].plot([int(n) for n in cv], list(cv.values()), alpha=0.6, color=plt.cm.plasma(math.log2(w_) / 9), lw=1)
            ax[2].set_title("E3 MLP OOS IC at fixed epochs\n(18 configs; colour = width)")
            ax[2].set_xlabel("epochs")
        ref = a.get("reference_ridge_M1_study13_grid")
        for x in ax[1:]:
            if ref:
                x.axhline(ref["mean"], color="k", ls="--", lw=1, label="ridge M1 (study-13 grid)")
                x.axhline(ref["mean"] + DELTA, color="r", ls=":", lw=1, label="ridge + delta (0.010)")
            x.legend(fontsize=7)
            x.set_ylabel("mean OOS IC (ALL)")
        fig.tight_layout()
        fig.savefig(FIG / "p9_capacity_sweep.png", dpi=130)
        plt.close(fig)
    if e4:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for kind, c_ in (("stationary", "#1f77b4"), ("nonstationary30", "#d62728")):
            pts = sorted((c["rho"], c["power_cluster_t"], c["power_boot_ci"]) for c in e4["cells"].values() if c.get("kind") == kind)
            if pts:
                ax[0].plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=c_, label=f"{kind} (cluster t)")
                ax[0].plot([p[0] for p in pts], [p[2] for p in pts], marker="x", ls="--", color=c_, label=f"{kind} (bootstrap CI)")
        ax[0].axhline(0.8, color="k", ls=":", lw=1)
        ax[0].set_title("E4 power to detect a planted IC (ridge-all, real y_rank)")
        ax[0].set_xlabel("planted average within-month IC")
        ax[0].set_ylabel("power (reject IC = 0 at 5%)")
        ax[0].legend(fontsize=7)
        if e1:
            for m, c_ in (("M1", "#1f77b4"), ("M2", "#d62728"), ("M3", "#2ca02c")):
                c = e1["cells"].get(f"base|{m}")
                if c and "ic_all" in c:
                    st = cstats(c["ic_all"], c["cluster"])
                    ax[1].errorbar([m], [st["mean"]], yerr=[[st["mean"] - st["ci_t"][0]], [st["ci_t"][1] - st["mean"]]], marker="o",
                                   capsize=4, color=c_)
            m80 = e4.get("analysis", {}).get("stationary", {}).get("mde80_power_cluster_t")
            if m80:
                ax[1].axhline(m80, color="k", ls="--", lw=1, label=f"MDE80 stationary {m80:.3f} (upper bound)")
            m80n = e4.get("analysis", {}).get("nonstationary30", {}).get("mde80_power_cluster_t")
            if m80n:
                ax[1].axhline(m80n, color="gray", ls=":", lw=1, label=f"MDE80 non-stationary {m80n:.3f}")
            ax[1].axhline(0, color="k", lw=0.5)
            ax[1].set_title("Realised OOS IC with 95% cluster-t CI vs MDE80")
            ax[1].set_ylabel("OOS IC (ALL)")
            ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "p9_ic_mde.png", dpi=130)
        plt.close(fig)
    log("figures written")


# ------------------------------------------------------------------------------------------------
def main() -> None:
    global SMOKE
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("e1", "e2", "e3", "e4", "e4b", "figs", "analyse", "portfolios"):
        sp = sub.add_parser(name)
        sp.add_argument("--smoke", action="store_true")
        sp.add_argument("--models", default="M1,M2,M3")
        sp.add_argument("--parts", default={"e1": "a,b,c,d,e", "e3": "ridge,xgb,mlp,seeds"}.get(name, ""))
        sp.add_argument("--which", default="e1")
        sp.add_argument("--only", default="")
    args = ap.parse_args()
    SMOKE = args.smoke
    t0 = time.time()
    if args.cmd == "e1":
        run_e1(args)
    elif args.cmd == "e2":
        run_e2(args)
    elif args.cmd == "e3":
        run_e3(args)
    elif args.cmd == "e4":
        run_e4(args)
    elif args.cmd == "e4b":
        run_e4b(args)
    elif args.cmd == "figs":
        run_figs(args)
    elif args.cmd == "portfolios":
        run_refresh_portfolios(args)
    elif args.cmd == "analyse":
        ck = load_ck(args.which)
        {"e1": lambda: analyse_e1(ck, args.models.split(","), SMOKE), "e2": lambda: analyse_e2(ck),
         "e3": lambda: analyse_e3(ck, SMOKE), "e4": lambda: analyse_e4(ck)}[args.which]()
        save_ck(args.which, ck)
    log(f"done in {(time.time() - t0) / 60:.1f} min; panel max ts asserted < 2024-07-01; sealed_load_count 0")


if __name__ == "__main__":
    main()
