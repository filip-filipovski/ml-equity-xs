#!/usr/bin/env python
"""P12 Stage 1b — matched construction for the residual targets (addendum 1 of P12_PREDECLARATION.md): vol-scaled rank
weights demeaned within SIC4 (fallback SIC2, then division) and beta-neutralised ('vsn4'), with and without the 20/35 %
hysteresis band, k in {1, 3}, ALL 15 bps / LIQ 5 bps, on the cached Stage 1 predictions (no new fit). Reads the realised
raw-return P&L (what a book earns), plus the gross decomposition into residual and factor parts. Writes p12_stage1b.json."""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import p9_ml_capacity as P9  # noqa: E402
import p12_books as PB  # noqa: E402
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
D = P9.TL.CACHE / "equities_xs" / "derived"; P9.PANEL_PATH = D / "ml_panel_wfa2024.parquet"; P9.META_PATH = D / "ml_panel_wfa2024.meta.json"
CACHE = HERE / "p12_cache"; LAST = PB.LAST
panel = P9.Panel(); assert panel.sealed_load_count == 0; panel.valid = panel.valid & (panel.tmi <= LAST)
df = P9.TL.train_load(P9.PANEL_PATH, "p12 stage1b frame"); assert len(df) == panel.n
assert str(pd.to_datetime(df["date"])[panel.valid].max())[:10] < "2023-01-01"
# SIC4 at the formation date (same join as Stage 1)
mem = pd.read_parquet(P9.TL.CACHE / "links" / "edges" / "e1_sic_membership.parquet", columns=["cik", "sic4", "sic2", "valid_from", "valid_to"])
for c in ("valid_from", "valid_to"): mem[c] = pd.to_datetime(mem[c], utc=True).dt.tz_localize(None)
mem = mem.dropna(subset=["cik"]); mem["cik"] = mem["cik"].astype("int64"); mem = mem.sort_values("valid_from")
key = pd.DataFrame({"cik": df["cik"].astype("float").fillna(-1).astype("int64"), "date": pd.to_datetime(df["date"]).dt.tz_localize(None), "_i": np.arange(len(df))}).sort_values("date")
j = pd.merge_asof(key, mem, left_on="date", right_on="valid_from", by="cik", direction="backward")
ok = j["valid_to"].isna() | (j["valid_to"] >= j["date"]); j.loc[~ok, ["sic4", "sic2"]] = np.nan; j = j.sort_values("_i")
sic4 = pd.Series(j["sic4"].to_numpy()).fillna(-1).astype(int).to_numpy(); sic2 = pd.Series(j["sic2"].to_numpy()).fillna(-1).astype(int).to_numpy()
div = np.argmax(df[[c for c in df.columns if c.startswith("sic_")]].to_numpy(), 1)
years = list(range(2009, 2023)); rows = np.concatenate([panel.year_scorer(y).rows for y in years]); sic_cols = [c for c in df.columns if c.startswith("sic_")]
skel = pd.DataFrame({"tmi": panel.tmi[rows], "firm": panel.firm[rows], "y": panel.y_raw[rows], "in_liq": panel.in_liq[rows], "beta": df["raw_beta_252"].to_numpy(float)[rows], "raw_vol_63": df["raw_vol_63"].to_numpy(float)[rows]})
for c in sic_cols: skel[c] = df[c].to_numpy(float)[rows]
skel["g4"] = sic4[rows]; skel["g2"] = sic2[rows]; skel["gd"] = div[rows]
cube = PB.Cube(skel.assign(pred=0.0)); G4, G2, GD = cube.mat("g4", -1), cube.mat("g2", -1), cube.mat("gd", -1)
m_t = (panel.y_raw - panel.y_ex_all); MKT = cube.pred_matrix(m_t[rows]); BETA = cube.B
log("cube", cube.T, "x", cube.F, "| SIC4 known for %.1f %% of rows" % (100 * np.mean(skel["g4"] >= 0)))

def group_demean(w, g, m, min_n=3):
    """Demean w within groups g (over mask m); groups smaller than min_n are left for the caller's fallback (returned mask)."""
    out = w.copy(); done = np.zeros(len(w), bool)
    s = pd.DataFrame({"w": w[m], "g": g[m]}); gg = s.groupby("g")["w"]; mu = gg.transform("mean").to_numpy(); n = gg.transform("size").to_numpy()
    idx = np.flatnonzero(m); okg = (n >= min_n) & (s["g"].to_numpy() >= 0)
    out[idx[okg]] = w[idx[okg]] - mu[okg]; done[idx[okg]] = True
    return out, done
def w_vsn4(sig, i, m, beta):
    r = PB.centred_rank_row(sig, m); w = np.where(m, r / cube.V[i], 0.0); w = PB.normalise(w)
    w1, d1 = group_demean(w, G4[i], m); w2, d2 = group_demean(w, G2[i], m & ~d1); w3, _ = group_demean(w, GD[i], m & ~d1 & ~d2)
    w = np.where(d1, w1, np.where(d2, w2, w3)); w = PB.normalise(w)
    # beta-neutral: remove the beta tilt (regress on [1, beta]) — keeps industry neutrality approximately
    b = np.where(np.isfinite(beta), beta, 1.0); X = np.column_stack([np.ones(m.sum()), b[m]]); coef, *_ = np.linalg.lstsq(X, w[m], rcond=None)
    out = np.zeros_like(w); out[m] = w[m] - X @ coef; return PB.normalise(out)
def run_vsn4(P, k, liq, cost, band=False):
    S = PB.smooth(cube, P, k); present = ~np.isnan(P); st = np.zeros(cube.F, int); prev = np.zeros(cube.F); G, GR, TO, TT = [], [], [], []
    for i in range(cube.T):
        m = present[i] & ~np.isnan(S[i]) & (cube.LIQ[i] if liq else True)
        if m.sum() < 50: continue
        sig = S[i].copy()
        if band:
            q = np.full(cube.F, np.nan); q[m] = pd.Series(S[i][m]).rank(pct=True).to_numpy(); new = st.copy()
            new[(st == 0) & (q >= 0.8)] = 1; new[(st == 0) & (q <= 0.2)] = -1; new[(st == 1) & ~(q >= 0.65)] = 0; new[(st == -1) & ~(q <= 0.35)] = 0; new[~m] = 0
            st[:] = new; m = m & (new != 0); sig = np.where(new != 0, new.astype(float), np.nan)
            if m.sum() < 50: continue
        w = w_vsn4(sig, i, m, BETA[i])
        y = np.where(np.isnan(cube.Y[i]), 0.0, cube.Y[i]); resid = y - np.where(np.isfinite(BETA[i]), BETA[i], 1.0) * np.where(np.isnan(MKT[i]), 0.0, MKT[i])
        G.append(float(w @ y)); GR.append(float(w @ resid)); TO.append(float(np.abs(w - prev).sum())); TT.append(int(cube.months[i])); prev = w
    return {"tmi": np.array(TT), "gross": np.array(G), "gross_resid": np.array(GR), "turnover": np.array(TO), "cost": cost}

out = {"design": __doc__, "cells": {}}
cells = sorted(p.stem.replace("stage1_", "") for p in CACHE.glob("stage1_*.npz") if "quick" not in p.name)
log(f"{'cell':<38}{'dev gSR':>8}{'dev nSR':>8}{'net%/yr':>8}{'DD':>6}{'to/mo':>6}{'resSR':>7} | {'chk gSR':>8}{'chk nSR':>8}{'DD':>6}")
for cid in cells:
    pred = np.load(CACHE / f"stage1_{cid}.npz")["pred"]; P = cube.pred_matrix(pred); out["cells"][cid] = {}
    for band in (False, True):
        for k in (1, 3):
            for liq, cost in ((False, 0.0015), (True, 0.0005)):
                bid = f"{'band_' if band else ''}vsn4_k{k}_{'LIQ5' if liq else 'ALL15'}"; bk = run_vsn4(P, k, liq, cost, band); mt = PB.metrics(bk)
                for w, (lo, hi) in (("dev", PB.DEV), ("chk", PB.CHK)):
                    mm = (bk["tmi"] >= lo) & (bk["tmi"] <= hi); gr = bk["gross_resid"][mm]
                    mt[w]["resid_gross_sharpe"] = float(gr.mean() / gr.std(ddof=1) * math.sqrt(12)) if mm.sum() > 2 else float("nan")
                out["cells"][cid][bid] = mt; d, c = mt["dev"], mt["chk"]
                log(f"{cid + ' ' + bid:<38}{d['gross_sharpe']:>8.2f}{d['net_sharpe']:>8.2f}{d['net_ann']*100:>8.1f}{d['maxdd']*100:>6.0f}{d['turnover_per_month']:>6.2f}{d['resid_gross_sharpe']:>7.2f} | {c['gross_sharpe']:>8.2f}{c['net_sharpe']:>8.2f}{c['maxdd']*100:>6.0f}")
best = max(((cid, bid, m["dev"]["gross_sharpe"], m["dev"]["net_sharpe"]) for cid, bb in out["cells"].items() for bid, m in bb.items()), key=lambda x: x[2])
out["best_dev_gross"] = best; out["max_formation_date_used"] = str(pd.to_datetime(df["date"])[panel.valid].max())[:10]; out["sealed_load_count"] = int(panel.sealed_load_count)
(HERE / "p12_stage1b.json").write_text(json.dumps(out, indent=1, default=float)); log("best development gross:", best, "| wrote p12_stage1b.json")
