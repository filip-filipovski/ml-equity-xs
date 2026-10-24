#!/usr/bin/env python
"""P12 shared constructions (same formulas as p12_stage0.py, generalised to any prediction frame) and metrics.
Frame columns: tmi, firm, y (raw next-month return), in_liq, beta, pred, raw_vol_63, sic_* one-hots. Weights per month
sum |w| = 2 and are dollar-neutral. Constructions: rank | vs (vol-scaled rank) | vsn (vol-scaled + beta/SIC-division
neutral) | band_vs (hysteresis enter 20 % / exit 35 %, vol-scaled). Prediction smoothing k = mean of the last k monthly
centred ranks per firm. Cost = cost_per_side x traded notional; no borrow. TRAIN months only by construction of the frame."""
from __future__ import annotations
import math
import numpy as np, pandas as pd
import p10_books as BK

DEV = (2009 * 12, 2018 * 12 + 11); CHK = (2019 * 12, 2022 * 12 + 11); LAST = 2022 * 12 + 11

class Cube:
    def __init__(self, frame: pd.DataFrame):
        frame = frame[frame["tmi"] <= LAST].reset_index(drop=True); assert frame["tmi"].max() <= LAST
        self.sic_cols = [c for c in frame.columns if c.startswith("sic_")]
        self.months = np.array(sorted(frame["tmi"].unique())); self.T = len(self.months); mpos = {int(t): i for i, t in enumerate(self.months)}
        self.firms = np.array(sorted(frame["firm"].unique())); self.F = len(self.firms); fpos = {int(f): j for j, f in enumerate(self.firms)}
        self.ti = frame["tmi"].map(mpos).to_numpy(); self.fj = frame["firm"].map(fpos).to_numpy(); self.frame = frame
        self.Y, self.V, self.B = self.mat("y"), self.mat("raw_vol_63"), self.mat("beta"); self.LIQ = self.mat("in_liq", 0.0) > 0.5
        self.SIC = np.stack([self.mat(c, 0.0) for c in self.sic_cols], -1)
        vol_floor = np.nanpercentile(self.V, 5); self.V = np.where(np.isnan(self.V), np.nanmedian(self.V), self.V); self.V = np.maximum(self.V, vol_floor)
    def mat(self, col, fill=np.nan):
        M = np.full((self.T, self.F), fill); M[self.ti, self.fj] = self.frame[col].to_numpy(float); return M
    def pred_matrix(self, pred: np.ndarray) -> np.ndarray:
        M = np.full((self.T, self.F), np.nan); M[self.ti, self.fj] = np.asarray(pred, float); return M

def centred_rank_row(x, m):
    r = np.full(len(x), np.nan); xs = x[m]; rk = pd.Series(xs).rank(method="average").to_numpy(); r[m] = (rk - 0.5) / len(xs) - 0.5; return r
def normalise(w):
    w = np.where(np.isnan(w), 0.0, w); nz = w != 0
    if nz.any(): w = np.where(nz, w - w[nz].mean(), 0.0)
    s = np.abs(w).sum(); return 2.0 * w / s if s > 0 else w
def neutralise(w, beta, sic, m):
    b = np.where(np.isfinite(beta), beta, np.nanmedian(beta[m]) if np.isfinite(beta[m]).any() else 0.0)
    X = np.column_stack([np.ones(m.sum()), b[m], sic[m]]); X = X[:, [j for j in range(X.shape[1]) if j < 2 or X[:, j].sum() > 0]]
    coef, *_ = np.linalg.lstsq(X, w[m], rcond=None); r = np.zeros_like(w); r[m] = w[m] - X @ coef; return normalise(r)

def smooth(cube: Cube, P: np.ndarray, k: int) -> np.ndarray:
    present = ~np.isnan(P); Z = np.full_like(P, np.nan)
    for i in range(cube.T): Z[i] = centred_rank_row(P[i], present[i])
    if k == 1: return Z
    S = np.full_like(Z, np.nan)
    for i in range(cube.T):
        blk = Z[max(0, i - k + 1): i + 1]
        with np.errstate(all="ignore"): S[i] = np.where(present[i], np.nanmean(blk, 0), np.nan)
    return S

def run_book(cube: Cube, P: np.ndarray, kind: str, k_smooth: int = 1, liq: bool = False, cost: float = 0.0015,
             voltarget: float | None = None, band=(0.2, 0.35)) -> dict:
    S = smooth(cube, P, k_smooth); present = ~np.isnan(P); st = np.zeros(cube.F, int); prev = np.zeros(cube.F); G, TO, TT = [], [], []
    for i in range(cube.T):
        m = present[i] & ~np.isnan(S[i]) & (cube.LIQ[i] if liq else True)
        if m.sum() < 50: continue
        r = centred_rank_row(S[i], m); inv = np.where(m, 1.0 / cube.V[i], 0.0)
        if kind == "rank": w = normalise(np.where(m, r, 0.0))
        elif kind in ("vs", "vsn"):
            w = normalise(np.where(m, r * inv, 0.0))
            if kind == "vsn": w = neutralise(w, cube.B[i], cube.SIC[i], m)
        elif kind == "band_vs":
            q = np.full(cube.F, np.nan); q[m] = pd.Series(S[i][m]).rank(pct=True).to_numpy(); enter, exit_ = band; new = st.copy()
            new[(st == 0) & (q >= 1 - enter)] = 1; new[(st == 0) & (q <= enter)] = -1
            new[(st == 1) & ~(q >= 1 - exit_)] = 0; new[(st == -1) & ~(q <= exit_)] = 0; new[~m] = 0
            w = normalise(np.where(new != 0, new * inv, 0.0)); st[:] = new
        else: raise ValueError(kind)
        if voltarget is not None and len(G) >= 12:
            sd = np.std(G[-12:], ddof=1) * math.sqrt(12); w = w * (min(3.0, voltarget / sd) if sd > 0 else 1.0)
        y = np.where(np.isnan(cube.Y[i]), 0.0, cube.Y[i]); G.append(float(w @ y)); TO.append(float(np.abs(w - prev).sum())); TT.append(int(cube.months[i])); prev = w
    return {"tmi": np.array(TT), "gross": np.array(G), "turnover": np.array(TO), "cost": cost}

def metrics(bk: dict) -> dict:
    out = {}
    for name, (lo, hi) in (("dev", DEV), ("chk", CHK), ("full", (DEV[0], CHK[1]))):
        mm = BK.window_metrics(bk, lo, hi); g = bk["gross"][(bk["tmi"] >= lo) & (bk["tmi"] <= hi)]
        mm["gross_sharpe"] = float(g.mean() / g.std(ddof=1) * math.sqrt(12)) if len(g) > 2 and g.std(ddof=1) > 0 else float("nan")
        out[name] = {k: mm[k] for k in ("n_months", "gross_ann", "gross_sharpe", "net_ann", "net_sharpe", "ann_vol", "maxdd", "turnover_per_month", "cost_ladder_net_ann", "net_sharpe_ci95") if k in mm}
    return out

def ic_by_month(cube: Cube, P: np.ndarray, target: np.ndarray | None = None, method: str = "spearman") -> np.ndarray:
    """Monthly correlation of pred with the realised return (default) or with a target matrix."""
    Tm = cube.Y if target is None else target; out = np.full(cube.T, np.nan)
    for i in range(cube.T):
        m = ~np.isnan(P[i]) & ~np.isnan(Tm[i])
        if m.sum() >= 50: out[i] = pd.Series(P[i][m]).corr(pd.Series(Tm[i][m]), method=method)
    return out

def window_ic(cube: Cube, ic: np.ndarray) -> dict:
    out = {}
    for name, (lo, hi) in (("dev", DEV), ("chk", CHK), ("full", (DEV[0], CHK[1]))):
        s = ic[(cube.months >= lo) & (cube.months <= hi)]; s = s[~np.isnan(s)]
        out[name] = {"mean": float(s.mean()), "sd": float(s.std(ddof=1)), "t": float(s.mean() / s.std(ddof=1) * math.sqrt(len(s))), "icir_ann": float(s.mean() / s.std(ddof=1) * math.sqrt(12)), "n": int(len(s))}
    return out
