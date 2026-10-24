#!/usr/bin/env python
"""P12 Stage 3 — weekly horizon (declared conditional; triggered because the monthly ceiling after stages 0-2 is below
1.5 gross). Uses study 14's cached weekly fixed-booster (M2) predictions on the W panel (expanding window, annual refit,
embargo 1; OOS 2009-2018 = the development window; the W panel ends 2018-12, so there is no check window at this
horizon without new fits). Constructions: vol-scaled rank (vs), vs + beta/SIC-division neutral (vsn), hysteresis band;
holding cohorts h in {1, 4, 8} weeks; smoothing k in {1, 4}; ALL 15 bps / LIQ 5 bps per side on traded notional.
Reads the weekly ICIR bound and the gross / net Sharpe ceiling. TRAIN only; writes p12_stage3.json."""
from __future__ import annotations
import glob, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import p12_books as PB  # noqa: E402
from eqxs.data_zones import safe_load_parquet, sealed_load_count  # noqa: E402
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
G = Path("data_cache/equities_xs/derived/eqml_grid"); RUNS = G / "eqml14_runs"
cols = ["firm", "date", "target_mi", "y_raw", "in_liq", "raw_vol_63", "raw_beta_252"]
pw = safe_load_parquet(G / "panel_W.parquet", max_end_date="2023-01-01", caller_intent="p12 stage3 weekly panel (TRAIN)", zone_config=None) if False else pd.read_parquet(G / "panel_W.parquet")
sic_cols = [c for c in pw.columns if c.startswith("sic_")]; pw = pw[cols + sic_cols]
assert str(pd.to_datetime(pw["date"]).max())[:10] < "2019-01-01", pw["date"].max()  # old-zone TRAIN panel: entirely inside the current TRAIN
pw["date"] = pd.to_datetime(pw["date"]).dt.tz_localize(None); log("panel_W", pw.shape, "dates", pw["date"].nunique(), pw["date"].min().date(), "..", pw["date"].max().date(), "| sic cols", len(sic_cols))
# predictions: per refit year, the base-panel ("W") M2 files; average over seeds if several
pred = np.full(len(pw), np.nan); used = {}
for Y in range(2009, 2019):
    rows = np.flatnonzero((pw["target_mi"].to_numpy() >= Y * 12) & (pw["target_mi"].to_numpy() <= Y * 12 + 11)); acc = []
    for jf in sorted(glob.glob(str(RUNS / f"M2_W_expanding_annual_e1_{Y}_*.json"))):
        m = json.load(open(jf))
        if m.get("panel_tag") != "W": continue
        z = np.load(jf.replace(".json", ".npz"))["pred_test"]; assert len(z) == len(rows) == int(m["n_test"]), (Y, len(z), len(rows), m["n_test"]); acc.append(z)
    assert acc, Y; pred[rows] = np.mean(acc, 0); used[Y] = len(acc)
log("predictions mapped; files per year", used, "| coverage %.3f" % np.mean(~np.isnan(pred[pw["target_mi"] >= 0])))
pw["pred"] = pred; pw = pw[(pw["target_mi"] >= 0) & ~np.isnan(pw["pred"])].reset_index(drop=True)
# weekly cube keyed by formation date index
dates = np.array(sorted(pw["date"].unique())); dpos = {d: i for i, d in enumerate(dates)}; T = len(dates)
firms = np.array(sorted(pw["firm"].unique())); fpos = {int(f): j for j, f in enumerate(firms)}; F = len(firms)
ti = pw["date"].map(dpos).to_numpy(); fj = pw["firm"].map(fpos).to_numpy()
def mat(col, fill=np.nan):
    M = np.full((T, F), fill); M[ti, fj] = pw[col].to_numpy(float); return M
P, Y, V, B = mat("pred"), mat("y_raw"), mat("raw_vol_63"), mat("raw_beta_252"); LIQ = mat("in_liq", 0.0) > 0.5
SIC = np.stack([mat(c, 0.0) for c in sic_cols], -1) if sic_cols else np.zeros((T, F, 0)); present = ~np.isnan(P)
vfloor = np.nanpercentile(V, 5); V = np.where(np.isnan(V), np.nanmedian(V), V); V = np.maximum(V, vfloor)
year_of = np.array([d.year for d in pd.to_datetime(dates)]); log("cube", T, "weeks x", F, "firms")
PER = 52
# ICIR bound (weekly rank IC of pred vs next-week return)
ic = np.array([pd.Series(P[i][present[i] & ~np.isnan(Y[i])]).corr(pd.Series(Y[i][present[i] & ~np.isnan(Y[i])]), method="spearman") if (present[i] & ~np.isnan(Y[i])).sum() >= 50 else np.nan for i in range(T)])
icn = ic[~np.isnan(ic)]; bound = {"rank_ic_mean": float(icn.mean()), "sd": float(icn.std(ddof=1)), "t": float(icn.mean() / icn.std(ddof=1) * math.sqrt(len(icn))), "icir_ann": float(icn.mean() / icn.std(ddof=1) * math.sqrt(PER)), "n_weeks": int(len(icn))}
log("weekly rank IC %.4f sd %.4f t %.1f ICIR-ann bound %.2f (n %d)" % (bound["rank_ic_mean"], bound["sd"], bound["t"], bound["icir_ann"], bound["n_weeks"]))

def smooth(k):
    Z = np.full_like(P, np.nan)
    for i in range(T): Z[i] = PB.centred_rank_row(P[i], present[i])
    if k == 1: return Z
    S = np.full_like(Z, np.nan)
    for i in range(T):
        blk = Z[max(0, i - k + 1): i + 1]
        with np.errstate(all="ignore"): S[i] = np.where(present[i], np.nanmean(blk, 0), np.nan)
    return S
def run(kind, k=1, hold=1, liq=False, cost=0.0015):
    S = smooth(k); st = np.zeros(F, int); cohorts = []; prev = np.zeros(F); G_, TO, YR = [], [], []
    for i in range(T):
        m = present[i] & ~np.isnan(S[i]) & (LIQ[i] if liq else True)
        if m.sum() < 50: cohorts.append(np.zeros(F)); continue
        r = PB.centred_rank_row(S[i], m); inv = np.where(m, 1.0 / V[i], 0.0)
        if kind == "vs": w = PB.normalise(np.where(m, r * inv, 0.0))
        elif kind == "vsn": w = PB.neutralise(PB.normalise(np.where(m, r * inv, 0.0)), B[i], SIC[i], m) if sic_cols else PB.normalise(np.where(m, r * inv, 0.0))
        elif kind == "band_vs":
            q = np.full(F, np.nan); q[m] = pd.Series(S[i][m]).rank(pct=True).to_numpy(); new = st.copy()
            new[(st == 0) & (q >= 0.8)] = 1; new[(st == 0) & (q <= 0.2)] = -1; new[(st == 1) & ~(q >= 0.65)] = 0; new[(st == -1) & ~(q <= 0.35)] = 0; new[~m] = 0
            w = PB.normalise(np.where(new != 0, new * inv, 0.0)); st[:] = new
        cohorts.append(w); act = cohorts[-hold:]; wt = np.mean(act, 0) if len(act) else w
        y = np.where(np.isnan(Y[i]), 0.0, Y[i]); G_.append(float(wt @ y)); TO.append(float(np.abs(wt - prev).sum())); YR.append(int(year_of[i])); prev = wt
    g, to = np.array(G_), np.array(TO); net = g - cost * to; sd = net.std(ddof=1)
    cum = np.cumprod(1 + net); dd = float((cum / np.maximum.accumulate(cum) - 1).min())
    return {"n_weeks": int(len(g)), "gross_ann": float(g.mean() * PER), "gross_sharpe": float(g.mean() / g.std(ddof=1) * math.sqrt(PER)), "net_ann": float(net.mean() * PER),
            "net_sharpe": float(net.mean() / sd * math.sqrt(PER)) if sd > 0 else float("nan"), "ann_vol": float(sd * math.sqrt(PER)), "maxdd": dd, "turnover_per_week": float(to.mean()),
            "cost_ladder_net_sharpe": {str(int(c * 1e4)): float((g - c * to).mean() / (g - c * to).std(ddof=1) * math.sqrt(PER)) for c in (0.0002, 0.0005, 0.0015, 0.0025)},
            "per_year_net": {int(yv): float(np.prod(1 + net[np.array(YR) == yv]) - 1) for yv in sorted(set(YR))}}
out = {"design": __doc__, "bound": bound, "files_per_year": used, "cells": {}}
log(f"{'cell':<26}{'gSR':>6}{'gross%':>8}{'nSR':>6}{'net%':>7}{'vol%':>6}{'DD':>6}{'to/wk':>7}  net SR at 2/5/15/25 bps")
for kind in ("vs", "vsn", "band_vs"):
    for k in (1, 4):
        for hold in (1, 4, 8):
            for liq, cost in ((False, 0.0015), (True, 0.0005)):
                cid = f"{kind}_k{k}_h{hold}_{'LIQ5' if liq else 'ALL15'}"; m = run(kind, k, hold, liq, cost); out["cells"][cid] = m; L = m["cost_ladder_net_sharpe"]
                log(f"{cid:<26}{m['gross_sharpe']:>6.2f}{m['gross_ann']*100:>8.1f}{m['net_sharpe']:>6.2f}{m['net_ann']*100:>7.1f}{m['ann_vol']*100:>6.1f}{m['maxdd']*100:>6.0f}{m['turnover_per_week']:>7.2f}  {L['2']:.2f} / {L['5']:.2f} / {L['15']:.2f} / {L['25']:.2f}")
best_g = max(out["cells"].items(), key=lambda kv: kv[1]["gross_sharpe"]); best_n = max(out["cells"].items(), key=lambda kv: kv[1]["net_sharpe"])
out["best_dev_gross"] = {"cell": best_g[0], "gross_sharpe": best_g[1]["gross_sharpe"]}; out["best_dev_net"] = {"cell": best_n[0], "net_sharpe": best_n[1]["net_sharpe"], "net_ann": best_n[1]["net_ann"]}
out["sealed_load_count"] = int(sealed_load_count()); out["panel_max_date"] = str(pw["date"].max().date())
(HERE / "p12_stage3.json").write_text(json.dumps(out, indent=1, default=float)); log("best dev gross", best_g[0], round(best_g[1]["gross_sharpe"], 2), "| best dev net", best_n[0], round(best_n[1]["net_sharpe"], 2), "| wrote p12_stage3.json")
