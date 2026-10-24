#!/usr/bin/env python
"""P12 Stage 0 — transfer and bound on the EXISTING M2 signal (cached predictions; no refit), per P12_PREDECLARATION.md.
TRAIN formation months only (tmi <= 2022-12 = 24275); development 2009-2018, check 2019-2022. Writes p12_stage0.json."""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import p9_ml_capacity as P9  # noqa: E402
import p10_books as BK  # noqa: E402
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
D = P9.TL.CACHE / "equities_xs" / "derived"; P9.PANEL_PATH = D / "ml_panel_wfa2024.parquet"; P9.META_PATH = D / "ml_panel_wfa2024.meta.json"
DEV = (2009 * 12, 2018 * 12 + 11); CHK = (2019 * 12, 2022 * 12 + 11); LAST = 2022 * 12 + 11
frame = pd.read_parquet(HERE / "p9_cache" / "p9_b_c1_frame_2009_2023.parquet")
frame = frame[frame["tmi"] <= LAST].reset_index(drop=True); assert frame["tmi"].max() <= LAST
sic_cols = [c for c in frame.columns if c.startswith("sic_")]
pan = P9.TL.train_load(P9.PANEL_PATH, "p12 stage0 vol columns")[["target_mi", "firm", "raw_vol_63", "raw_log_dv_63", "date"]]
pan = pan[(pan["target_mi"] >= 0) & (pan["target_mi"] <= LAST)]  # TRAIN formation months only (target_mi -1 = rows without a target); the loader itself is bounded by the sealed start
assert str(pd.to_datetime(pan["date"]).max())[:10] < "2023-01-01", pan["date"].max()  # formation dates inside TRAIN
pan = pan.rename(columns={"target_mi": "tmi"}).drop(columns="date"); pan["firm"] = pan["firm"].astype(np.int64)
frame = frame.merge(pan, on=["tmi", "firm"], how="left"); log("frame", frame.shape, "vol missing", float(frame["raw_vol_63"].isna().mean()))
assert P9.TL.sealed_load_count() == 0 if hasattr(P9.TL, "sealed_load_count") else True

# ---- matrices (tmi x firm) for smoothing / bands
months = np.array(sorted(frame["tmi"].unique())); T = len(months); mpos = {int(t): i for i, t in enumerate(months)}
firms = np.array(sorted(frame["firm"].unique())); fpos = {int(f): j for j, f in enumerate(firms)}; F = len(firms)
ti = frame["tmi"].map(mpos).to_numpy(); fj = frame["firm"].map(fpos).to_numpy()
def mat(col, fill=np.nan):
    M = np.full((T, F), fill); M[ti, fj] = frame[col].to_numpy(float); return M
P, Y, V = mat("pred"), mat("y"), mat("raw_vol_63"); B = mat("beta"); LIQ = mat("in_liq", 0.0) > 0.5
SIC = np.stack([mat(c, 0.0) for c in sic_cols], -1); present = ~np.isnan(P)
vol_floor = np.nanpercentile(V, 5); V = np.where(np.isnan(V), np.nanmedian(V), V); V = np.maximum(V, vol_floor)
log("months", T, "firms", F, "vol floor %.4f" % vol_floor)

def centred_rank_row(x, m):
    r = np.full(len(x), np.nan); xs = x[m]; rk = pd.Series(xs).rank(method="average").to_numpy(); r[m] = (rk - 0.5) / len(xs) - 0.5; return r
def normalise(w):
    w = np.where(np.isnan(w), 0.0, w); w = w - w[w != 0].mean() if (w != 0).any() else w; s = np.abs(w).sum(); return 2.0 * w / s if s > 0 else w
def neutralise(w, beta, sic, m):
    b = np.where(np.isfinite(beta), beta, np.nanmedian(beta[m]) if np.isfinite(beta[m]).any() else 0.0)
    X = np.column_stack([np.ones(m.sum()), b[m], sic[m]]); X = X[:, [j for j in range(X.shape[1]) if j < 2 or X[:, j].sum() > 0]]
    coef, *_ = np.linalg.lstsq(X, w[m], rcond=None); r = np.zeros_like(w); r[m] = w[m] - X @ coef; return normalise(r)

def smooth_pred(k):
    """Per-firm mean of the last k available monthly predictions (rank-standardised per month first)."""
    Z = np.full_like(P, np.nan)
    for i in range(T): Z[i] = centred_rank_row(P[i], present[i])
    if k == 1: return Z
    S = np.full_like(Z, np.nan)
    for i in range(T):
        blk = Z[max(0, i - k + 1): i + 1]; S[i] = np.where(present[i], np.nanmean(np.where(np.isnan(blk), np.nan, blk), 0), np.nan)
    return S

def weights_for(kind, sig_row, i, m, band_state=None, band=(0.2, 0.35)):
    """kind: rank | vs (vol-scaled rank) | vsn (vol-scaled + neutral) | band_vs (hysteresis on rank, vol-scaled)."""
    r = centred_rank_row(sig_row, m)
    if kind == "rank": return normalise(np.where(m, r, 0.0))
    inv = np.where(m, 1.0 / V[i], 0.0)
    if kind in ("vs", "vsn"):
        w = np.where(m, r * inv, 0.0); w = normalise(w)
        return neutralise(w, B[i], SIC[i], m) if kind == "vsn" else w
    if kind == "band_vs":
        q = np.full(F, np.nan); q[m] = pd.Series(sig_row[m]).rank(pct=True).to_numpy()
        st = band_state; enter, exit_ = band
        new = st.copy()
        new[(st == 0) & (q >= 1 - enter)] = 1; new[(st == 0) & (q <= enter)] = -1
        new[(st == 1) & ~(q >= 1 - exit_)] = 0; new[(st == -1) & ~(q <= exit_)] = 0; new[~m] = 0
        w = np.where(new != 0, new * inv, 0.0); w = normalise(w); st[:] = new; return w
    raise ValueError(kind)

def run(kind, k_smooth=1, liq=False, cost=0.0015, voltarget=None):
    S = smooth_pred(k_smooth); st = np.zeros(F, int); prev = np.zeros(F); G, TO, TT = [], [], []
    for i in range(T):
        m = present[i] & (LIQ[i] if liq else True); m = m & ~np.isnan(S[i])
        if m.sum() < 50: continue
        w = weights_for(kind, S[i], i, m, st)
        if voltarget is not None and len(G) >= 12:
            sd = np.std(G[-12:], ddof=1) * math.sqrt(12); lev = min(3.0, voltarget / sd) if sd > 0 else 1.0; w = w * lev
        y = np.where(np.isnan(Y[i]), 0.0, Y[i]); G.append(float(w @ y)); TO.append(float(np.abs(w - prev).sum())); TT.append(int(months[i])); prev = w
    return {"tmi": np.array(TT), "gross": np.array(G), "turnover": np.array(TO), "cost": cost}

def metrics(bk):
    out = {}
    for name, (lo, hi) in (("dev", DEV), ("chk", CHK), ("full", (DEV[0], CHK[1]))):
        mm = BK.window_metrics(bk, lo, hi); g = bk["gross"][(bk["tmi"] >= lo) & (bk["tmi"] <= hi)]
        mm["gross_sharpe"] = float(g.mean() / g.std(ddof=1) * math.sqrt(12)) if len(g) > 2 and g.std(ddof=1) > 0 else float("nan")
        out[name] = {k: mm[k] for k in ("n_months", "gross_ann", "gross_sharpe", "net_ann", "net_sharpe", "ann_vol", "maxdd", "turnover_per_month", "cost_ladder_net_ann", "net_sharpe_ci95") if k in mm}
    return out

# ---- ICIR bound and Pearson transfer from the monthly series
def ic_series(kind):
    out = []
    for i in range(T):
        m = present[i] & ~np.isnan(Y[i])
        if m.sum() < 50: out.append(np.nan); continue
        a, b = P[i][m], Y[i][m]
        out.append(pd.Series(a).corr(pd.Series(b), method=kind))
    return np.array(out)
res = {"design": __doc__, "bounds": {}}
for lab, (lo, hi) in (("dev", DEV), ("chk", CHK), ("full", (DEV[0], CHK[1]))):
    sel = (months >= lo) & (months <= hi); rk, pe = ic_series("spearman")[sel], ic_series("pearson")[sel]
    res["bounds"][lab] = {"rank_ic_mean": float(np.nanmean(rk)), "rank_ic_sd": float(np.nanstd(rk, ddof=1)), "icir_ann_bound": float(np.nanmean(rk) / np.nanstd(rk, ddof=1) * math.sqrt(12)),
                          "pearson_ic_mean": float(np.nanmean(pe)), "pearson_icir_ann": float(np.nanmean(pe) / np.nanstd(pe, ddof=1) * math.sqrt(12)), "n_months": int(sel.sum())}
    log(f"bound {lab}: rank IC {np.nanmean(rk):+.4f} sd {np.nanstd(rk, ddof=1):.4f} ICIR-ann {res['bounds'][lab]['icir_ann_bound']:.2f} | pearson IC {np.nanmean(pe):+.4f} ICIR-ann {res['bounds'][lab]['pearson_icir_ann']:.2f}")

GRID = []
for kind in ("rank", "vs", "vsn", "band_vs"):
    for k in (1, 3, 6):
        for liq, cost in ((False, 0.0015), (True, 0.0005)):
            GRID.append((kind, k, liq, cost))
res["cells"] = {}
log(f"{'cell':<26}{'dev gSR':>8}{'dev nSR':>8}{'dev net%':>9}{'DD':>7}{'to/mo':>7} | {'chk gSR':>8}{'chk nSR':>8}{'chk net%':>9}{'DD':>7}")
for kind, k, liq, cost in GRID:
    cid = f"{kind}_k{k}_{'LIQ5' if liq else 'ALL15'}"
    bk = run(kind, k, liq, cost); mt = metrics(bk); res["cells"][cid] = mt
    d, c = mt["dev"], mt["chk"]
    log(f"{cid:<26}{d['gross_sharpe']:>8.2f}{d['net_sharpe']:>8.2f}{d['net_ann']*100:>9.1f}{d['maxdd']*100:>7.0f}{d['turnover_per_month']:>7.2f} | {c['gross_sharpe']:>8.2f}{c['net_sharpe']:>8.2f}{c['net_ann']*100:>9.1f}{c['maxdd']*100:>7.0f}")
# vol-targeted versions of the two most promising kinds (declared: vol target 10 %)
res["cells_voltarget10"] = {}
for kind, k, liq, cost in (("vsn", 3, False, 0.0015), ("vsn", 3, True, 0.0005), ("band_vs", 3, False, 0.0015), ("band_vs", 3, True, 0.0005)):
    cid = f"{kind}_k{k}_{'LIQ5' if liq else 'ALL15'}_vt10"; bk = run(kind, k, liq, cost, voltarget=0.10); mt = metrics(bk); res["cells_voltarget10"][cid] = mt
    d, c = mt["dev"], mt["chk"]; log(f"{cid:<26}{d['gross_sharpe']:>8.2f}{d['net_sharpe']:>8.2f}{d['net_ann']*100:>9.1f}{d['maxdd']*100:>7.0f}{d['turnover_per_month']:>7.2f} | {c['gross_sharpe']:>8.2f}{c['net_sharpe']:>8.2f}{c['net_ann']*100:>9.1f}{c['maxdd']*100:>7.0f}")
best = max(((cid, m["dev"]["gross_sharpe"]) for cid, m in res["cells"].items()), key=lambda x: x[1])
res["best_dev_gross"] = {"cell": best[0], "gross_sharpe": best[1]}
(HERE / "p12_stage0.json").write_text(json.dumps(res, indent=1, default=float)); log("best development GROSS Sharpe", best, "| wrote p12_stage0.json")
