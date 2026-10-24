#!/usr/bin/env python
"""P12c step 3 — the P12b routes on the rebuilt panels (eqml_grid_2023) with the P12c predictions: development 2009-2018 and
check 2019-2022. Selection by the pre-declared rule on development (highest net at the declared cost, MaxDD <= 25 %); the
check window is read as a result for that cell only (all cells' check numbers are stored). Writes p12c_eval.json."""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
RESULTS = HERE.parent / "results"; RESULTS.mkdir(exist_ok=True)
import p12_books as PB  # noqa: E402
import p12_stage1b_lib as L1  # noqa: E402
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
G = (HERE.parent / "data_cache/equities_xs/derived/eqml_grid_2023"); E = (HERE.parent / "data_cache/links/edges"); CACHE = RESULTS / "p12_cache"
PER = 52; THETAS = (0.05, 0.1, 0.2, 0.5, 1.0); COSTS = (0.0001, 0.0002, 0.0005, 0.0015); DECL = {"ALL": "15", "LIQ": "5", "TOP500": "2"}; LAST = 2022 * 12 + 11
def load(h, cols):
    pf = pd.read_parquet(G / f"panel_{h}.parquet"); pf = pf[[c for c in cols if c in pf.columns]]
    z = np.load(CACHE / f"p12c_{h}.npz"); pred = np.full(len(pf), np.nan); pred[z["rows"]] = z["pred"]; pf["pred"] = pred
    pf["date"] = pd.to_datetime(pf["date"]).dt.tz_localize(None); pf = pf[(pf["target_mi"] >= 0) & (pf["target_mi"] <= LAST) & ~np.isnan(pred)].reset_index(drop=True)
    assert str(pf["date"].max())[:10] < "2023-01-01"; log(f"{h}: {len(pf)} rows with predictions, {pf['date'].min().date()} .. {pf['date'].max().date()}"); return pf
base = ["firm", "date", "cik", "target_mi", "y_raw", "in_liq", "raw_vol_63", "raw_beta_252", "raw_log_dv_63"]
W = load("W", base + [f"sic_{c}" for c in "ABCDEFGHIJ"] + ["sic_unknown"]); sic_cols = [c for c in W.columns if c.startswith("sic_")]
M = load("M", ["firm", "date", "target_mi"]); Q = load("Q", ["firm", "date", "target_mi"])
mem = pd.read_parquet(E / "e1_sic_membership.parquet", columns=["cik", "sic4", "sic2", "valid_from", "valid_to"])
for c in ("valid_from", "valid_to"): mem[c] = pd.to_datetime(mem[c], utc=True).dt.tz_localize(None)
mem = mem.dropna(subset=["cik"]); mem["cik"] = mem["cik"].astype("int64"); mem = mem.sort_values("valid_from")
key = pd.DataFrame({"cik": W["cik"].astype("float").fillna(-1).astype("int64"), "date": W["date"], "_i": np.arange(len(W))}).sort_values("date")
j = pd.merge_asof(key, mem, left_on="date", right_on="valid_from", by="cik", direction="backward"); ok = j["valid_to"].isna() | (j["valid_to"] >= j["date"])
j.loc[~ok, ["sic4", "sic2"]] = np.nan; j = j.sort_values("_i"); W["g4"] = pd.Series(j["sic4"].to_numpy()).fillna(-1).astype(int).to_numpy(); W["g2"] = pd.Series(j["sic2"].to_numpy()).fillna(-1).astype(int).to_numpy(); W["gd"] = np.argmax(W[sic_cols].to_numpy(), 1)
dates = np.array(sorted(W["date"].unique())); dpos = {d: i for i, d in enumerate(dates)}; T = len(dates)
firms = np.array(sorted(W["firm"].unique())); fpos = {int(f): k for k, f in enumerate(firms)}; F = len(firms); ti = W["date"].map(dpos).to_numpy(); fj = W["firm"].map(fpos).to_numpy()
def mat(col, fill=np.nan):
    Mx = np.full((T, F), fill); Mx[ti, fj] = W[col].to_numpy(float); return Mx
class C: pass
cube = C(); cube.T, cube.F = T, F; cube.Y, cube.V, cube.B = mat("y_raw"), mat("raw_vol_63"), mat("raw_beta_252"); cube.LIQ = mat("in_liq", 0.0) > 0.5; cube.SIC = np.stack([mat(c, 0.0) for c in sic_cols], -1)
PW = mat("pred"); DV = mat("raw_log_dv_63"); G4, G2, GD = mat("g4", -1), mat("g2", -1), mat("gd", -1); present = ~np.isnan(PW)
vfloor = np.nanpercentile(cube.V, 5); cube.V = np.where(np.isnan(cube.V), np.nanmedian(cube.V), cube.V); cube.V = np.maximum(cube.V, vfloor)
TOP = np.zeros((T, F), bool)
for i in range(T):
    m = present[i] & ~np.isnan(DV[i]); idx = np.flatnonzero(m); TOP[i, idx[np.argsort(-DV[i][idx])][:500]] = True
def carried(src):
    out = np.full((T, F), np.nan); fd = np.array(sorted(src["date"].unique())); snaps = {d: dict(zip(g["firm"].astype(int), g["pred"])) for d, g in src.groupby("date")}
    for i, d in enumerate(dates):
        k = np.searchsorted(fd, d, side="right") - 1
        if k < 0: continue
        for f, v in snaps[fd[k]].items():
            jx = fpos.get(f)
            if jx is not None and present[i, jx]: out[i, jx] = v
    return out
PM, PQ = carried(M), carried(Q)
def ranks(Px):
    Z = np.full_like(Px, np.nan)
    for i in range(T):
        m = ~np.isnan(Px[i]); Z[i] = PB.centred_rank_row(Px[i], m) if m.sum() else np.nan
    return Z
ZW, ZM, ZQ = ranks(PW), ranks(PM), ranks(PQ)
with np.errstate(all="ignore"): AIMS = {"R1_weekly": ZW, "R2_quarterly_slow": ZQ, "R4_blend_WMQ": np.nanmean(np.stack([ZW, ZM, ZQ]), 0), "R4b_blend_WM": np.nanmean(np.stack([ZW, ZM]), 0)}
year = np.array([pd.Timestamp(d).year for d in dates]); DEVM = year <= 2018; CHKM = year >= 2019
def stats(g, to, msk):
    gg, tt = g[msk], to[msk]; out = {"n_weeks": int(msk.sum()), "gross_ann": float(gg.mean() * PER), "gross_sharpe": float(gg.mean() / gg.std(ddof=1) * math.sqrt(PER)), "turnover_per_week": float(tt.mean())}
    for c in COSTS:
        net = gg - c * tt; sd = net.std(ddof=1); cum = np.cumprod(1 + net); out[f"net_{c*1e4:g}bps"] = {"sharpe": float(net.mean() / sd * math.sqrt(PER)) if sd > 0 else float("nan"), "ann": float(net.mean() * PER), "maxdd": float((cum / np.maximum.accumulate(cum) - 1).min())}
    return out
def run(Z, theta, uni):
    prev = np.zeros(F); G_, TO, used = [], [], []
    for i in range(T):
        m = present[i] & ~np.isnan(Z[i]) & (cube.LIQ[i] if uni == "LIQ" else TOP[i] if uni == "TOP500" else True)
        if m.sum() < 50: G_.append(0.0); TO.append(0.0); used.append(False); continue
        a = L1.w_vsn4(cube, G4, G2, GD, Z[i], i, m, cube.B[i]); w = prev + theta * (a - prev); w = np.where(m, w, 0.0)
        y = np.where(np.isnan(cube.Y[i]), 0.0, cube.Y[i]); G_.append(float(w @ y)); TO.append(float(np.abs(w - prev).sum())); used.append(True); prev = w
    g, to, u = np.array(G_), np.array(TO), np.array(used); return {"dev": stats(g, to, DEVM & u), "chk": stats(g, to, CHKM & u)}
res = {"design": __doc__, "cells": {}}
log(f"{'cell':<34}{'dev gSR':>8}{'dev net@decl':>13}{'DD':>6} | {'chk gSR':>8}{'chk net@decl':>13}{'DD':>6}")
for aname, Z in AIMS.items():
    for uni in ("ALL", "LIQ", "TOP500"):
        if aname.startswith("R2") and uni == "TOP500": continue
        for theta in THETAS:
            cid = f"{aname}_{uni}_theta{theta}"; r = run(Z, theta, uni); res["cells"][cid] = r; k = f"net_{DECL[uni]}bps"; d, c = r["dev"], r["chk"]
            log(f"{cid:<34}{d['gross_sharpe']:>8.2f}{d[k]['sharpe']:>13.2f}{d[k]['maxdd']*100:>6.0f} | {c['gross_sharpe']:>8.2f}{c[k]['sharpe']:>13.2f}{c[k]['maxdd']*100:>6.0f}")
elig = [(cid, r) for cid, r in res["cells"].items() if r["dev"][f"net_{DECL[cid.split('_')[-2]]}bps"]["maxdd"] >= -0.25]
sel = max(elig, key=lambda kv: kv[1]["dev"][f"net_{DECL[kv[0].split('_')[-2]]}bps"]["sharpe"]); k = f"net_{DECL[sel[0].split('_')[-2]]}bps"
res["selected"] = {"cell": sel[0], "dev": sel[1]["dev"][k], "dev_gross_sharpe": sel[1]["dev"]["gross_sharpe"], "chk": sel[1]["chk"][k], "chk_gross_sharpe": sel[1]["chk"]["gross_sharpe"], "declared_cost_bps": k}
res["pass"] = bool(sel[1]["dev"][k]["sharpe"] >= 1.5 and sel[1]["chk"][k]["sharpe"] >= 1.5 and sel[1]["dev"][k]["maxdd"] >= -0.25 and sel[1]["chk"][k]["maxdd"] >= -0.25)
(RESULTS / "p12c_eval.json").write_text(json.dumps(res, indent=1, default=float)); log("SELECTED:", json.dumps(res["selected"], default=float), "| PASS:", res["pass"], "| wrote p12c_eval.json")
