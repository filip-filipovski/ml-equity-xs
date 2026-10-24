"""vsn4 construction (addendum 1): vol-scaled rank weights demeaned within SIC4 (fallback SIC2, division), beta-neutral."""
from __future__ import annotations
import numpy as np, pandas as pd
import p12_books as PB

def group_demean(w, g, m, min_n=3):
    out = w.copy(); done = np.zeros(len(w), bool)
    s = pd.DataFrame({"w": w[m], "g": g[m]}); gg = s.groupby("g")["w"]; mu = gg.transform("mean").to_numpy(); n = gg.transform("size").to_numpy()
    idx = np.flatnonzero(m); okg = (n >= min_n) & (s["g"].to_numpy() >= 0)
    out[idx[okg]] = w[idx[okg]] - mu[okg]; done[idx[okg]] = True
    return out, done

def w_vsn4(cube, G4, G2, GD, sig, i, m, beta):
    r = PB.centred_rank_row(sig, m); w = PB.normalise(np.where(m, r / cube.V[i], 0.0))
    w1, d1 = group_demean(w, G4[i], m); w2, d2 = group_demean(w, G2[i], m & ~d1); w3, _ = group_demean(w, GD[i], m & ~d1 & ~d2)
    w = PB.normalise(np.where(d1, w1, np.where(d2, w2, w3)))
    b = np.where(np.isfinite(beta), beta, 1.0); X = np.column_stack([np.ones(m.sum()), b[m]]); coef, *_ = np.linalg.lstsq(X, w[m], rcond=None)
    out = np.zeros_like(w); out[m] = w[m] - X @ coef; return PB.normalise(out)

def run_vsn4(cube, G4, G2, GD, BETA, P, k=1, liq=False, cost=0.0015, band=False):
    S = PB.smooth(cube, P, k); present = ~np.isnan(P); st = np.zeros(cube.F, int); prev = np.zeros(cube.F); G, TO, TT = [], [], []
    for i in range(cube.T):
        m = present[i] & ~np.isnan(S[i]) & (cube.LIQ[i] if liq else True)
        if m.sum() < 50: continue
        sig = S[i].copy()
        if band:
            q = np.full(cube.F, np.nan); q[m] = pd.Series(S[i][m]).rank(pct=True).to_numpy(); new = st.copy()
            new[(st == 0) & (q >= 0.8)] = 1; new[(st == 0) & (q <= 0.2)] = -1; new[(st == 1) & ~(q >= 0.65)] = 0; new[(st == -1) & ~(q <= 0.35)] = 0; new[~m] = 0
            st[:] = new; m = m & (new != 0); sig = np.where(new != 0, new.astype(float), np.nan)
            if m.sum() < 50: continue
        w = w_vsn4(cube, G4, G2, GD, sig, i, m, BETA[i])
        y = np.where(np.isnan(cube.Y[i]), 0.0, cube.Y[i]); G.append(float(w @ y)); TO.append(float(np.abs(w - prev).sum())); TT.append(int(cube.months[i])); prev = w
    return {"tmi": np.array(TT), "gross": np.array(G), "turnover": np.array(TO), "cost": cost}
