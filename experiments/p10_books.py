#!/usr/bin/env python
"""P10 book constructions B0-B4 and metrics (incl. maximum drawdown). Pure numpy/pandas over a monthly cross-section
frame with columns: tmi (target month index), firm, pred, y, beta, in_liq and a SIC one-hot matrix. Pre-declared in
P10_PREDECLARATION.md. No borrow cost; cost = cost_per_side x sum|dw| (traded notional)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

LADDER = (0.0005, 0.0015, 0.0025, 0.005)


def _centred_rank(x: np.ndarray) -> np.ndarray:
    r = pd.Series(x).rank(method="average").to_numpy()
    n = len(r)
    return (r - 0.5) / n - 0.5


def w_decile(pred: np.ndarray) -> np.ndarray:
    q = pd.Series(pred).rank(pct=True, method="first").to_numpy()
    w = np.zeros(len(pred))
    L, S = q > 0.9, q <= 0.1
    if L.sum() == 0 or S.sum() == 0:
        return w
    w[L] = 1.0 / L.sum()
    w[S] = -1.0 / S.sum()
    return w


def w_rank(pred: np.ndarray) -> np.ndarray:
    w = _centred_rank(pred)
    w = w - w.mean()
    s = np.abs(w).sum()
    return 2.0 * w / s if s > 0 else w


def w_neutral(pred: np.ndarray, beta: np.ndarray, sic: np.ndarray) -> np.ndarray:
    """Rank weights made dollar-, beta- and industry-neutral: residual of w on [1, beta, SIC dummies], rescaled."""
    w = w_rank(pred)
    b = np.asarray(beta, float).copy()
    if np.isfinite(b).any():
        b[~np.isfinite(b)] = np.nanmedian(b)
    else:
        b[:] = 0.0
    X = np.column_stack([np.ones(len(w)), b, np.asarray(sic, float)])
    keep = [j for j in range(X.shape[1]) if j < 2 or X[:, j].sum() > 0]
    X = X[:, keep]
    coef, *_ = np.linalg.lstsq(X, w, rcond=None)
    r = w - X @ coef
    s = np.abs(r).sum()
    return 2.0 * r / s if s > 0 else r


def maxdd(net: np.ndarray) -> float:
    cum = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(cum)
    return float((cum / peak - 1.0).min()) if len(cum) else float("nan")


def series_metrics(gross: np.ndarray, to: np.ndarray, cost: float, tmi: np.ndarray, seed: int = 11) -> dict:
    net = gross - cost * to
    n = len(net)
    sd = net.std(ddof=1) if n > 1 else float("nan")
    rng = np.random.default_rng(seed)
    B, L = 20_000, 3
    starts = rng.integers(0, n, size=(B, math.ceil(n / L)))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n] % n
    ann_b = net[idx].mean(1) * 12
    sr_b = np.array([m / s * math.sqrt(12) if s > 0 else np.nan for m, s in zip(net[idx].mean(1), net[idx].std(axis=1, ddof=1))])
    yrs = (tmi // 12).astype(int)
    per_year = {int(y): float(np.prod(1 + net[yrs == y]) - 1) for y in np.unique(yrs)}
    return {"n_months": int(n), "gross_ann": float(gross.mean() * 12), "net_ann": float(net.mean() * 12),
            "ann_vol": float(sd * math.sqrt(12)), "net_sharpe": float(net.mean() / sd * math.sqrt(12)) if sd > 0 else float("nan"),
            "maxdd": maxdd(net), "turnover_per_month": float(to.mean()),
            "net_ann_ci95": [float(np.percentile(ann_b, 2.5)), float(np.percentile(ann_b, 97.5))],
            "net_sharpe_ci95": [float(np.nanpercentile(sr_b, 2.5)), float(np.nanpercentile(sr_b, 97.5))],
            "cost_ladder_net_ann": {str(int(c * 1e4)): float((gross - c * to).mean() * 12) for c in LADDER},
            "per_year_net": per_year, "share_months_pos": float((net > 0).mean())}


def run_book(df: pd.DataFrame, sic_cols: list[str], kind: str, hold: int = 1, cost: float = 0.0015,
             liq_only: bool = False) -> dict:
    """kind in {decile, rank, neutral}. Returns {'tmi','gross','turnover'} arrays over all months in df."""
    d = df[df["in_liq"]] if liq_only else df
    months = np.array(sorted(d["tmi"].unique()))
    cohorts: dict[int, dict[int, float]] = {}
    for t in months:
        m = d[d["tmi"] == t]
        if len(m) < 50:
            continue
        p = m["pred"].to_numpy(float)
        if kind == "decile":
            w = w_decile(p)
        elif kind == "rank":
            w = w_rank(p)
        else:
            w = w_neutral(p, m["beta"].to_numpy(float), m[sic_cols].to_numpy(float))
        f = m["firm"].to_numpy()
        cohorts[int(t)] = {int(ff): float(ww) for ff, ww in zip(f, w) if ww != 0.0}
    ret = d.set_index(["tmi", "firm"])["y"]
    prev: dict[int, float] = {}
    out_t, out_g, out_to = [], [], []
    for t in months:
        active = [cohorts[int(t) - k] for k in range(hold) if (int(t) - k) in cohorts]
        if not active:
            continue
        w: dict[int, float] = {}
        for c in active:
            for ff, v in c.items():
                w[ff] = w.get(ff, 0.0) + v / hold
        rr = ret.loc[int(t)] if int(t) in ret.index.get_level_values(0) else None
        g = 0.0
        if rr is not None:
            rd = rr.to_dict()
            for ff, v in w.items():
                yv = rd.get(ff, np.nan)
                g += v * (0.0 if not np.isfinite(yv) else float(yv))
        keys = set(w) | set(prev)
        to = sum(abs(w.get(ff, 0.0) - prev.get(ff, 0.0)) for ff in keys)
        out_t.append(int(t)); out_g.append(g); out_to.append(to)
        prev = w
    return {"tmi": np.array(out_t), "gross": np.array(out_g), "turnover": np.array(out_to), "cost": cost}


BOOKS = {  # id -> (kind, hold, cost, liq_only)
    "B0_decile": ("decile", 1, 0.0015, False),
    "B1_rank": ("rank", 1, 0.0015, False),
    "B2_neutral": ("neutral", 1, 0.0015, False),
    "B3_neutral_cohort3": ("neutral", 3, 0.0015, False),
    "B4_neutral_cohort3_LIQ": ("neutral", 3, 0.0005, True),
}


def all_books(df: pd.DataFrame, sic_cols: list[str]) -> dict[str, dict]:
    return {bid: run_book(df, sic_cols, *spec) for bid, spec in BOOKS.items()}


def window_metrics(bk: dict, lo_tmi: int, hi_tmi: int) -> dict:
    m = (bk["tmi"] >= lo_tmi) & (bk["tmi"] <= hi_tmi)
    if m.sum() < 6:
        return {"n_months": int(m.sum())}
    return series_metrics(bk["gross"][m], bk["turnover"][m], bk["cost"], bk["tmi"][m])
