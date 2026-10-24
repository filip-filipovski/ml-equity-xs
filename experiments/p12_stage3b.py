#!/usr/bin/env python
"""P12 Stage 3b — partial adjustment toward the weekly aim portfolio (addendum 2): w_t = w_{t-1} + theta (w*_t - w_{t-1}),
theta in {0.05, 0.1, 0.2, 0.35, 0.5, 1.0}; aims 'vsn' and 'vs' from Stage 3; ALL 15 bps / LIQ 5 bps; cost ladder.
Development window (OOS 2009-2018 of study 14's cached weekly M2 predictions). Writes p12_stage3b.json."""
from __future__ import annotations
import glob, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import p12_books as PB  # noqa: E402
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
G = Path("data_cache/equities_xs/derived/eqml_grid"); RUNS = G / "eqml14_runs"
pw = pd.read_parquet(G / "panel_W.parquet"); sic_cols = [c for c in pw.columns if c.startswith("sic_")]
pw = pw[["firm", "date", "target_mi", "y_raw", "in_liq", "raw_vol_63", "raw_beta_252"] + sic_cols]; assert str(pd.to_datetime(pw["date"]).max())[:10] < "2019-01-01"
pw["date"] = pd.to_datetime(pw["date"]).dt.tz_localize(None); tm = pw["target_mi"].to_numpy(); pred = np.full(len(pw), np.nan)
for Y in range(2009, 2019):
    rows = np.flatnonzero((tm >= Y * 12) & (tm <= Y * 12 + 11))
    for jf in sorted(glob.glob(str(RUNS / f"M2_W_expanding_annual_e1_{Y}_*.json"))):
        if json.load(open(jf)).get("panel_tag") == "W": pred[rows] = np.load(jf.replace(".json", ".npz"))["pred_test"]
pw["pred"] = pred; pw = pw[(tm >= 0) & ~np.isnan(pred)].reset_index(drop=True)
dates = np.array(sorted(pw["date"].unique())); dpos = {d: i for i, d in enumerate(dates)}; T = len(dates)
firms = np.array(sorted(pw["firm"].unique())); fpos = {int(f): j for j, f in enumerate(firms)}; F = len(firms)
ti = pw["date"].map(dpos).to_numpy(); fj = pw["firm"].map(fpos).to_numpy()
def mat(col, fill=np.nan):
    M = np.full((T, F), fill); M[ti, fj] = pw[col].to_numpy(float); return M
P, Y, V, B = mat("pred"), mat("y_raw"), mat("raw_vol_63"), mat("raw_beta_252"); LIQ = mat("in_liq", 0.0) > 0.5
SIC = np.stack([mat(c, 0.0) for c in sic_cols], -1); present = ~np.isnan(P)
vfloor = np.nanpercentile(V, 5); V = np.where(np.isnan(V), np.nanmedian(V), V); V = np.maximum(V, vfloor); PER = 52
Z = np.full_like(P, np.nan)
for i in range(T): Z[i] = PB.centred_rank_row(P[i], present[i])
def aim(kind, i, m):
    inv = np.where(m, 1.0 / V[i], 0.0); w = PB.normalise(np.where(m, Z[i] * inv, 0.0))
    return PB.neutralise(w, B[i], SIC[i], m) if kind == "vsn" else w
def run(kind, theta, liq, cost):
    prev = np.zeros(F); G_, TO = [], []
    for i in range(T):
        m = present[i] & (LIQ[i] if liq else True)
        if m.sum() < 50: continue
        a = aim(kind, i, m); w = prev + theta * (a - prev); w[~m & (prev == 0)] = 0.0
        # positions in names that left the universe are closed (counted as turnover)
        w = np.where(m | (prev != 0), w, 0.0); w = np.where(m, w, 0.0)
        y = np.where(np.isnan(Y[i]), 0.0, Y[i]); G_.append(float(w @ y)); TO.append(float(np.abs(w - prev).sum())); prev = w
    g, to = np.array(G_), np.array(TO); out = {"n_weeks": int(len(g)), "gross_ann": float(g.mean() * PER), "gross_sharpe": float(g.mean() / g.std(ddof=1) * math.sqrt(PER)), "turnover_per_week": float(to.mean()), "gross_exposure_mean": None}
    for c in (0.0002, 0.0005, 0.0015, 0.0025):
        net = g - c * to; sd = net.std(ddof=1); cum = np.cumprod(1 + net)
        out[f"net_{int(c*1e4)}bps"] = {"sharpe": float(net.mean() / sd * math.sqrt(PER)) if sd > 0 else float("nan"), "ann": float(net.mean() * PER), "maxdd": float((cum / np.maximum.accumulate(cum) - 1).min())}
    out["declared_cost_bps"] = int(cost * 1e4); return out
res = {"design": __doc__, "cells": {}}
log(f"{'cell':<22}{'gSR':>6}{'gross%':>8}{'to/wk':>7} | net SR @2/5/15/25 bps | net %/yr @declared | DD @declared")
for kind in ("vsn", "vs"):
    for theta in (0.05, 0.1, 0.2, 0.35, 0.5, 1.0):
        for liq, cost in ((False, 0.0015), (True, 0.0005)):
            cid = f"{kind}_theta{theta}_{'LIQ5' if liq else 'ALL15'}"; m = run(kind, theta, liq, cost); res["cells"][cid] = m; dc = m[f"net_{int(cost*1e4)}bps"]
            log(f"{cid:<22}{m['gross_sharpe']:>6.2f}{m['gross_ann']*100:>8.1f}{m['turnover_per_week']:>7.2f} | {m['net_2bps']['sharpe']:.2f} / {m['net_5bps']['sharpe']:.2f} / {m['net_15bps']['sharpe']:.2f} / {m['net_25bps']['sharpe']:.2f} | {dc['ann']*100:+.1f} | {dc['maxdd']*100:.0f} %")
best = max(res["cells"].items(), key=lambda kv: kv[1][f"net_{kv[1]['declared_cost_bps']}bps"]["sharpe"])
res["best_dev_net_declared_cost"] = {"cell": best[0], **best[1][f"net_{best[1]['declared_cost_bps']}bps"]}
(HERE / "p12_stage3b.json").write_text(json.dumps(res, indent=1, default=float)); log("best dev net at declared cost:", res["best_dev_net_declared_cost"], "| wrote p12_stage3b.json")
