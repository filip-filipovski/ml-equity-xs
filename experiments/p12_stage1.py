#!/usr/bin/env python
"""P12 Stage 1 — target engineering on the TRAIN panel (per P12_PREDECLARATION.md): targets T0 y_rank, T1 beta-residual,
T2 SIC-division residual, T3 SIC4 residual (graph residual, membership valid at the formation date), T4 beta + SIC4
residual, T5 vol-scaled return; models M2 (fixed booster) and M1 (ridge); expanding refits 2009-2022 (make_refits).
Reads: monthly rank IC vs realised return and vs own target; Stage-0 constructions on each (development / check).
Predictions cached under p12_cache/. Writes p12_stage1.json."""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
RESULTS = HERE.parent / "results"; RESULTS.mkdir(exist_ok=True)
import p9_ml_capacity as P9  # noqa: E402
import p12_books as PB  # noqa: E402
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
D = P9.TL.CACHE / "equities_xs" / "derived"; P9.PANEL_PATH = D / "ml_panel_wfa2024.parquet"; P9.META_PATH = D / "ml_panel_wfa2024.meta.json"
CACHE = RESULTS / "p12_cache"; CACHE.mkdir(exist_ok=True)
QUICK = "--quick" in sys.argv
LAST = PB.LAST
panel = P9.Panel(); assert panel.sealed_load_count == 0
panel.valid = panel.valid & (panel.tmi <= LAST)  # TRAIN formation months only (the loader is bounded by the sealed start, so 2023 WFA rows exist in the file and are excluded here)
df = P9.TL.train_load(P9.PANEL_PATH, "p12 stage1 frame"); assert len(df) == panel.n and (df["date_id"].to_numpy() == panel.date_id).all()
assert str(pd.to_datetime(df["date"])[panel.valid].max())[:10] < "2023-01-01", df["date"][panel.valid].max()  # formation dates inside TRAIN
log("panel", int(panel.valid.sum()), "TRAIN rows used of", panel.n, "| features", len(panel.columns))
tmi = panel.tmi; yraw = panel.y_raw; valid = panel.valid

# ---- SIC4 membership at the formation date (graph: industry peers)
mem = pd.read_parquet(P9.TL.CACHE / "links" / "edges" / "e1_sic_membership.parquet", columns=["cik", "sic4", "sic2", "valid_from", "valid_to"])
for c in ("valid_from", "valid_to"): mem[c] = pd.to_datetime(mem[c], utc=True).dt.tz_localize(None)
mem = mem.dropna(subset=["cik"]); mem["cik"] = mem["cik"].astype("int64"); mem = mem.sort_values("valid_from")
key = pd.DataFrame({"cik": df["cik"].astype("float").fillna(-1).astype("int64"), "date": pd.to_datetime(df["date"]).dt.tz_localize(None), "_i": np.arange(len(df))}).sort_values("date")
j = pd.merge_asof(key, mem, left_on="date", right_on="valid_from", by="cik", direction="backward")
ok = j["valid_to"].isna() | (j["valid_to"] >= j["date"]); j.loc[~ok, ["sic4", "sic2"]] = np.nan; j = j.sort_values("_i")
sic4 = j["sic4"].to_numpy(); sic2 = j["sic2"].to_numpy(); log("SIC4 coverage of valid rows: %.1f %%" % (100 * np.mean(~pd.isna(sic4[valid]))))
div = np.argmax(df[[c for c in df.columns if c.startswith("sic_")]].to_numpy(), 1)

def grp_demean(x, keys, min_n=5):
    """x minus its mean within each key (tuple of arrays); groups smaller than min_n fall back to NaN (caller cascades)."""
    s = pd.DataFrame({"x": x}); k = pd.MultiIndex.from_arrays(keys); g = s.groupby(k)["x"]; mu = g.transform("mean").to_numpy(); n = g.transform("size").to_numpy()
    return np.where(n >= min_n, x - mu, np.nan)
def cascade(x, levels):
    out = np.full(len(x), np.nan)
    for keys in levels:
        r = grp_demean(x, keys); fill = np.isnan(out); out[fill] = r[fill]
    return out
def to_rank(x):
    """Within target-month centred percentile rank, on the same scale as panel.y (detected)."""
    r = pd.Series(x).groupby(tmi).rank(pct=True).to_numpy(); r = r - 0.5 if np.nanmin(panel.y) < 0 else r
    return np.where(np.isnan(x), np.nan, r)
m_t = yraw - panel.y_ex_all                    # cross-sectional mean return of the month (ALL universe)
beta = df["raw_beta_252"].to_numpy(float); beta = np.where(np.isfinite(beta), beta, 1.0)
vol = df["raw_vol_63"].to_numpy(float); vol = np.where(np.isfinite(vol), vol, np.nanmedian(vol[valid])); vol = np.maximum(vol, np.nanpercentile(vol[valid], 5))
sic4k = pd.Series(sic4).fillna(-1).astype(int).to_numpy(); sic2k = pd.Series(sic2).fillna(-1).astype(int).to_numpy()
e_beta = yraw - beta * m_t
TARGETS = {
    "T0_rank": panel.y,
    "T1_beta_resid": to_rank(e_beta),
    "T2_div_resid": to_rank(cascade(yraw, [(tmi, div)])),
    "T3_sic4_resid": to_rank(cascade(yraw, [(tmi, sic4k), (tmi, sic2k), (tmi, div)])),
    "T4_beta_sic4_resid": to_rank(cascade(e_beta, [(tmi, sic4k), (tmi, sic2k), (tmi, div)])),
    "T5_vol_scaled": to_rank(yraw / vol),
}
for k, v in TARGETS.items(): log(f"target {k}: nan share {np.mean(np.isnan(v[valid])):.3f} corr with y_rank {pd.Series(v[valid]).corr(pd.Series(panel.y[valid])):.3f}")
years = list(range(2009, 2023)) if not QUICK else [2017, 2018]
refits = P9.make_refits(years)
# frame skeleton for books (rows in the order of the yearly test rows)
rows = np.concatenate([panel.year_scorer(y).rows for y in years]); sic_cols = [c for c in df.columns if c.startswith("sic_")]
skel = pd.DataFrame({"tmi": panel.tmi[rows], "firm": panel.firm[rows], "y": panel.y_raw[rows], "in_liq": panel.in_liq[rows], "beta": df["raw_beta_252"].to_numpy(float)[rows], "raw_vol_63": df["raw_vol_63"].to_numpy(float)[rows]})
for c in sic_cols: skel[c] = df[c].to_numpy(float)[rows]
cube = PB.Cube(skel.assign(pred=0.0)); log("cube", cube.T, "months x", cube.F, "firms")
MODELS = ["M2", "M1"]; CONSTR = [("vs", 1, False, 0.0015), ("vsn", 3, False, 0.0015), ("band_vs", 3, False, 0.0015), ("vsn", 3, True, 0.0005), ("band_vs", 3, True, 0.0005)]
out = {"design": __doc__, "years": years, "cells": {}}
for tname, tgt in TARGETS.items():
    yfit = np.where(np.isnan(tgt), np.nanmedian(tgt[valid]), tgt)
    for mname in MODELS:
        cid = f"{tname}__{mname}"; cf = CACHE / f"stage1_{cid}{'_quick' if QUICK else ''}.npz"; t0 = time.time()
        if cf.exists():
            z = np.load(cf); pred = z["pred"]; log(f"{cid}: cached")
        else:
            res = P9.fit_predict_cell(panel, panel.X, list(panel.columns), refits, mname, y=yfit)
            pred = np.concatenate([res["pred"][y] for y in years]); np.savez_compressed(cf, pred=pred)
        P = cube.pred_matrix(pred)
        ic_y = PB.window_ic(cube, PB.ic_by_month(cube, P)); ic_own = PB.window_ic(cube, PB.ic_by_month(cube, P, cube.pred_matrix(tgt[rows])))
        cell = {"ic_vs_return": ic_y, "ic_vs_own_target": ic_own, "books": {}}
        for kind, k, liq, cost in CONSTR:
            bid = f"{kind}_k{k}_{'LIQ5' if liq else 'ALL15'}"; cell["books"][bid] = PB.metrics(PB.run_book(cube, P, kind, k, liq, cost))
        out["cells"][cid] = cell
        b = cell["books"]["vsn_k3_ALL15"]; bl = cell["books"]["vsn_k3_LIQ5"]; gv = lambda d, w, k: d.get(w, {}).get(k, float("nan"))
        log(f"{cid:<26} IC(ret) dev {ic_y['dev']['mean']:+.4f} (t {ic_y['dev']['t']:.1f}) chk {ic_y['chk']['mean']:+.4f} | ICIR-ann dev {ic_y['dev']['icir_ann']:.2f} chk {ic_y['chk']['icir_ann']:.2f} | "
            f"vsn_k3 ALL dev gSR {gv(b,'dev','gross_sharpe'):.2f} nSR {gv(b,'dev','net_sharpe'):.2f} ({gv(b,'dev','net_ann')*100:+.1f}%/yr, DD {gv(b,'dev','maxdd')*100:.0f}%) chk nSR {gv(b,'chk','net_sharpe'):.2f} | LIQ dev nSR {gv(bl,'dev','net_sharpe'):.2f} chk {gv(bl,'chk','net_sharpe'):.2f} | {time.time()-t0:.0f}s")
out["sealed_load_count"] = int(panel.sealed_load_count); out["max_formation_date_used"] = str(pd.to_datetime(df["date"])[panel.valid].max())[:10]
(RESULTS / ("p12_stage1_quick.json" if QUICK else "p12_stage1.json")).write_text(json.dumps(out, indent=1, default=float)); log("wrote p12_stage1.json")
