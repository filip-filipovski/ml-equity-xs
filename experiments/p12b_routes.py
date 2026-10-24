#!/usr/bin/env python
"""P12b — the red team's routes (P12_PREDECLARATION.md addendum 3), development window 2009-2018 only, on study 14's cached
M2 predictions (W / M / Q panels, expanding window, annual refit, embargo 1). R1 weekly SIC4-matched aim (vsn4) with
partial adjustment; R2 quarterly slow aim carried forward to weekly dates; R4 multi-horizon blend (W + M + Q centred ranks);
R5 top-500-by-dollar-volume subset at 1-2 bps with the R1 and R4 aims. Universes ALL 15 bps / LIQ 5 bps / TOP500 2 bps;
cost ladder 1 / 2 / 5 / 15 bps for every cell. Writes p12b_routes.json."""
from __future__ import annotations
import glob, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
RESULTS = HERE.parent / "results"; RESULTS.mkdir(exist_ok=True)
import p12_books as PB  # noqa: E402
import p12_stage1b_lib as L1  # noqa: E402
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
G = (HERE.parent / "data_cache/equities_xs/derived/eqml_grid"); RUNS = G / "eqml14_runs"; E = (HERE.parent / "data_cache/links/edges")
PER = 52; THETAS = (0.05, 0.1, 0.2, 0.5, 1.0); COSTS = (0.0001, 0.0002, 0.0005, 0.0015)

def load_cached(h: str, cols: list[str]) -> pd.DataFrame:
    """Panel rows (formation date, firm, target_mi, ...) with the base-panel M2 prediction for OOS 2009-2018."""
    pf = pd.read_parquet(G / f"panel_{h}.parquet", columns=[c for c in cols if c in pd.read_parquet(G / f"panel_{h}.parquet", columns=None).columns[:0].tolist() or True])
    pf = pf[[c for c in cols if c in pf.columns]]; tm = pf["target_mi"].to_numpy(); pred = np.full(len(pf), np.nan); n_files = 0
    for Y in range(2009, 2019):
        rows = np.flatnonzero((tm >= Y * 12) & (tm <= Y * 12 + 11))
        for jf in sorted(glob.glob(str(RUNS / f"M2_{h}_expanding_annual_e1_{Y}_*.json"))):
            m = json.load(open(jf))
            if m.get("panel_tag") != h: continue
            z = np.load(jf.replace(".json", ".npz"))["pred_test"]
            if len(z) != len(rows): log(f"  {h} {Y}: n_test {len(z)} != rows {len(rows)} — skipped"); continue
            pred[rows] = z; n_files += 1
    pf["pred"] = pred; pf["date"] = pd.to_datetime(pf["date"]).dt.tz_localize(None); pf = pf[(tm >= 0) & ~np.isnan(pred)].reset_index(drop=True)
    assert str(pf["date"].max())[:10] < "2019-01-01"; log(f"{h}: {len(pf)} rows with predictions from {n_files} refit files; dates {pf['date'].min().date()} .. {pf['date'].max().date()}"); return pf

base_cols = ["firm", "date", "cik", "target_mi", "y_raw", "in_liq", "raw_vol_63", "raw_beta_252", "raw_log_dv_63"]
W = load_cached("W", base_cols + [f"sic_{c}" for c in "ABCDEFGHIJ"] + ["sic_unknown"]); sic_cols = [c for c in W.columns if c.startswith("sic_")]
M = load_cached("M", ["firm", "date", "target_mi"]); Q = load_cached("Q", ["firm", "date", "target_mi"])
# SIC4 at the weekly formation date
mem = pd.read_parquet(E / "e1_sic_membership.parquet", columns=["cik", "sic4", "sic2", "valid_from", "valid_to"])
for c in ("valid_from", "valid_to"): mem[c] = pd.to_datetime(mem[c], utc=True).dt.tz_localize(None)
mem = mem.dropna(subset=["cik"]); mem["cik"] = mem["cik"].astype("int64"); mem = mem.sort_values("valid_from")
key = pd.DataFrame({"cik": W["cik"].astype("float").fillna(-1).astype("int64"), "date": W["date"], "_i": np.arange(len(W))}).sort_values("date")
j = pd.merge_asof(key, mem, left_on="date", right_on="valid_from", by="cik", direction="backward"); ok = j["valid_to"].isna() | (j["valid_to"] >= j["date"])
j.loc[~ok, ["sic4", "sic2"]] = np.nan; j = j.sort_values("_i"); W["g4"] = pd.Series(j["sic4"].to_numpy()).fillna(-1).astype(int).to_numpy(); W["g2"] = pd.Series(j["sic2"].to_numpy()).fillna(-1).astype(int).to_numpy()
W["gd"] = np.argmax(W[sic_cols].to_numpy(), 1); log("SIC4 coverage %.1f %%" % (100 * np.mean(W["g4"] >= 0)))
# weekly cube
dates = np.array(sorted(W["date"].unique())); dpos = {d: i for i, d in enumerate(dates)}; T = len(dates)
firms = np.array(sorted(W["firm"].unique())); fpos = {int(f): k for k, f in enumerate(firms)}; F = len(firms)
ti = W["date"].map(dpos).to_numpy(); fj = W["firm"].map(fpos).to_numpy()
def mat(col, fill=np.nan):
    Mx = np.full((T, F), fill); Mx[ti, fj] = W[col].to_numpy(float); return Mx
class C: pass
cube = C(); cube.T, cube.F, cube.months = T, F, np.arange(T); cube.Y, cube.V, cube.B = mat("y_raw"), mat("raw_vol_63"), mat("raw_beta_252"); cube.LIQ = mat("in_liq", 0.0) > 0.5
cube.SIC = np.stack([mat(c, 0.0) for c in sic_cols], -1); PW = mat("pred"); DV = mat("raw_log_dv_63"); G4, G2, GD = mat("g4", -1), mat("g2", -1), mat("gd", -1)
vfloor = np.nanpercentile(cube.V, 5); cube.V = np.where(np.isnan(cube.V), np.nanmedian(cube.V), cube.V); cube.V = np.maximum(cube.V, vfloor); present = ~np.isnan(PW)
TOP = np.zeros((T, F), bool)
for i in range(T):
    m = present[i] & ~np.isnan(DV[i]); idx = np.flatnonzero(m); order = idx[np.argsort(-DV[i][idx])][:500]; TOP[i, order] = True
# carried-forward monthly / quarterly aims at weekly dates (latest formation date <= weekly date; no look-ahead)
def carried(src: pd.DataFrame) -> np.ndarray:
    out = np.full((T, F), np.nan); fdates = np.array(sorted(src["date"].unique())); snaps = {d: dict(zip(g["firm"].astype(int), g["pred"])) for d, g in src.groupby("date")}
    for i, d in enumerate(dates):
        k = np.searchsorted(fdates, d, side="right") - 1
        if k < 0: continue
        snap = snaps[fdates[k]]
        for f, v in snap.items():
            jx = fpos.get(f)
            if jx is not None and present[i, jx]: out[i, jx] = v
    return out
PM, PQ = carried(M), carried(Q); log("carried coverage of weekly rows: M %.2f, Q %.2f" % (np.mean(~np.isnan(PM[present])), np.mean(~np.isnan(PQ[present]))))
def ranks(Px):
    Z = np.full_like(Px, np.nan)
    for i in range(T):
        m = ~np.isnan(Px[i]); Z[i] = PB.centred_rank_row(Px[i], m) if m.sum() else np.nan
    return Z
ZW, ZM, ZQ = ranks(PW), ranks(PM), ranks(PQ)
with np.errstate(all="ignore"): ZB = np.nanmean(np.stack([ZW, ZM, ZQ]), 0)
AIMS = {"R1_weekly": ZW, "R2_quarterly_slow": ZQ, "R4_blend_WMQ": ZB, "R4b_blend_WM": np.nanmean(np.stack([ZW, ZM]), 0)}
def run(Z, theta, uni):
    prev = np.zeros(F); G_, TO = [], []
    for i in range(T):
        m = present[i] & ~np.isnan(Z[i]) & (cube.LIQ[i] if uni == "LIQ" else TOP[i] if uni == "TOP500" else True)
        if m.sum() < 50: continue
        a = L1.w_vsn4(cube, G4, G2, GD, Z[i], i, m, cube.B[i]); w = prev + theta * (a - prev); w = np.where(m, w, 0.0)
        y = np.where(np.isnan(cube.Y[i]), 0.0, cube.Y[i]); G_.append(float(w @ y)); TO.append(float(np.abs(w - prev).sum())); prev = w
    g, to = np.array(G_), np.array(TO); out = {"n_weeks": int(len(g)), "gross_ann": float(g.mean() * PER), "gross_sharpe": float(g.mean() / g.std(ddof=1) * math.sqrt(PER)), "turnover_per_week": float(to.mean())}
    for c in COSTS:
        net = g - c * to; sd = net.std(ddof=1); cum = np.cumprod(1 + net); out[f"net_{c*1e4:g}bps"] = {"sharpe": float(net.mean() / sd * math.sqrt(PER)) if sd > 0 else float("nan"), "ann": float(net.mean() * PER), "maxdd": float((cum / np.maximum.accumulate(cum) - 1).min())}
    return out
DECL = {"ALL": "15", "LIQ": "5", "TOP500": "2"}
res = {"design": __doc__, "cells": {}}
log(f"{'cell':<34}{'gSR':>6}{'gross%':>7}{'to/wk':>7} | net SR @1/2/5/15 bps | declared: net SR, %/yr, DD")
for aname, Z in AIMS.items():
    for uni in ("ALL", "LIQ", "TOP500"):
        if aname.startswith("R2") and uni == "TOP500": continue
        for theta in THETAS:
            cid = f"{aname}_{uni}_theta{theta}"; m = run(Z, theta, uni); res["cells"][cid] = m; d = m[f"net_{DECL[uni]}bps"]
            log(f"{cid:<34}{m['gross_sharpe']:>6.2f}{m['gross_ann']*100:>7.1f}{m['turnover_per_week']:>7.2f} | {m['net_1bps']['sharpe']:.2f} / {m['net_2bps']['sharpe']:.2f} / {m['net_5bps']['sharpe']:.2f} / {m['net_15bps']['sharpe']:.2f} | {d['sharpe']:+.2f}, {d['ann']*100:+.1f}, {d['maxdd']*100:.0f} %")
def best_at(key):
    k, v = max(res["cells"].items(), key=lambda kv: kv[1][key]["sharpe"]); return {"cell": k, **v[key]}
res["best"] = {"declared_cost": max(((k, v[f"net_{DECL[k.split('_')[-2]]}bps"]) for k, v in res["cells"].items()), key=lambda kv: kv[1]["sharpe"]), "net_2bps": best_at("net_2bps"), "net_5bps": best_at("net_5bps"), "net_15bps": best_at("net_15bps"), "gross": max(((k, v["gross_sharpe"]) for k, v in res["cells"].items()), key=lambda kv: kv[1])}
(RESULTS / "p12b_routes.json").write_text(json.dumps(res, indent=1, default=float)); log("BEST:", json.dumps(res["best"], default=float)[:600], "| wrote p12b_routes.json")
