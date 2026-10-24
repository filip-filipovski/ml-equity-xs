#!/usr/bin/env python
"""P9 E6 — fresh-window replication of the fixed booster M2 on 2019-2022 (TRAIN by zone; never fit/selected/scored
on by any ML study). Pre-declaration: P9_E6_PREDECLARATION.md (committed before this is run). Composes P9 pieces
unchanged: Panel (pointed at the 2023 panel), make_refits, fit_predict_cell, score_preds, portfolio_stats, cstats.
--check2018: plumbing check on the OLD window only (M2 test year 2018 vs P9 E1); reads no post-2018 IC."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"; RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import p9_ml_capacity as P9  # noqa: E402

D = P9.TL.CACHE / "equities_xs" / "derived"
P9.PANEL_PATH = D / "ml_panel_train2023.parquet"
P9.META_PATH = D / "ml_panel_train2023.meta.json"
YEARS = [2019, 2020, 2021, 2022]
OUT = RESULTS / "p9_e6.json"
PX_FEATS = ["log_price", "log_mcap", "bm", "ep", "sp"]  # price-level features (addendum 1: non-PIT split channel)
P9_REF_IC = 0.0314  # P9 E1 M2 ALL, 2009-2018 (recomputed below from p9_e1.json)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def concat_rows_preds(panel, res):
    years = sorted(res["pred"])
    rows = np.concatenate([panel.year_scorer(y).rows for y in years])
    pred = np.concatenate([res["pred"][y] for y in years])
    return rows, pred


def read_cell(panel, res, name: str) -> dict:
    sc = P9.score_preds(panel, res, portfolio=False)
    ic, cl = np.asarray(sc["ic_all"], float), np.asarray(sc["cluster"])
    st_all = P9.cstats(ic, cl)
    st_liq = P9.cstats(np.asarray(sc["ic_liq"], float), cl)
    rows, pred = concat_rows_preds(panel, res)
    port = {u: P9.portfolio_stats(panel, rows, pred, u) for u in ("ALL", "LIQ")}
    per_year = {}
    for y in sc["years"]:
        sel = cl == y
        per_year[int(y)] = {"ic_all_mean": float(ic[sel].mean()), "n_months": int(sel.sum()),
                            "val_ic": sc["val_ic"][sc["years"].index(y)], "alpha": sc["alpha"][sc["years"].index(y)]}
    log(f"{name}: IC_all {st_all['mean']:+.4f} cluster-t {st_all['t_cluster']:+.2f} boot CI "
        f"[{st_all['ci_boot'][0]:+.4f},{st_all['ci_boot'][1]:+.4f}] iid-t {st_all['t_month_iid']:+.2f} "
        f"n={st_all['n_months']} | OOS R2 {sc['oos_r2']:+.4f} | L/S ALL net {port['ALL']['net']['ann_return']*100:+.2f}%/yr "
        f"SR {port['ALL']['net']['sharpe']:+.2f} (gross {port['ALL']['gross']['ann_return']*100:+.2f}%/yr "
        f"SR {port['ALL']['gross']['sharpe']:+.2f}; be {port['ALL']['break_even_bps_per_side']:.1f} bps)")
    return {"stats_all": st_all, "stats_liq": st_liq, "oos_r2": sc["oos_r2"], "ic_all": ic.tolist(),
            "ic_liq": sc["ic_liq"], "cluster": cl.tolist(), "years": sc["years"], "per_year": per_year,
            "portfolio": port, "wall_seconds": sc["wall_seconds"], "n_fit_mean": sc["n_fit_mean"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check2018", action="store_true", help="plumbing: M2 on test year 2018 (old window) only")
    a = ap.parse_args()
    panel = P9.Panel()
    mx = pd.Timestamp(panel.max_ts)
    log(f"panel n={panel.n} max_ts={panel.max_ts} sealed_load_count={panel.sealed_load_count} "
        f"features={len(panel.columns)} FIRST_TARGET_MI={P9.FIRST_TARGET_MI}")
    assert panel.sealed_load_count == 0
    assert mx < pd.Timestamp("2023-01-01", tz="UTC") and mx >= pd.Timestamp("2022-10-01", tz="UTC")
    meta = json.loads(P9.META_PATH.read_text())
    assert meta["dropped_features"] == ["gp"] and meta.get("feature_set_pinned_to_study13")

    if a.check2018:
        refits = P9.make_refits([2018], 1)
        res = P9.fit_predict_cell(panel, panel.X, panel.columns, refits, "M2")
        sc = P9.score_preds(panel, res, portfolio=False)
        ic = np.asarray(sc["ic_all"], float)
        ref = json.load(open(RESULTS / "p9_e1.json"))["cells"]["base|M2"]
        ric = np.asarray(ref["ic_all"], float)[np.asarray(ref["cluster"]) == 2018]
        # the extended window completes the 2018-12 holding window (12 months vs P9's 11): compare the common 11
        n = min(len(ic), len(ric))
        d = ic[:n] - ric[:n]
        log(f"check2018: new-panel M2 2018 IC mean {ic[:n].mean():+.4f} (n={len(ic)}) vs P9 E1 {ric.mean():+.4f} "
            f"(n={len(ric)}); common {n} months: per-month diff mean {d.mean():+.4f} sd {d.std(ddof=1):.4f}; "
            f"month-SD of the IC itself {ric.std(ddof=1):.4f}; corr {np.corrcoef(ic[:n], ric[:n])[0, 1]:+.3f}")
        (RESULTS / "p9_e6_check2018.json").write_text(json.dumps(
            {"new_ic": ic.tolist(), "p9_ic": ric.tolist(), "n_common": n, "diff_mean": float(d.mean()),
             "diff_sd": float(d.std(ddof=1)), "ic_month_sd_p9": float(ric.std(ddof=1)),
             "corr": float(np.corrcoef(ic[:n], ric[:n])[0, 1])}, indent=1))
        return

    refits = P9.make_refits(YEARS, 1)
    # the 2019 refit must train on target months <= 2018-11 only (P9's data, nothing newer)
    assert refits[0].win_end <= 2018 * 12 + 10, refits[0]
    for rf in refits:
        log(f"refit {rf.year}: fit tmi {rf.fit} val {rf.val} test {rf.test} win_end {rf.win_end}")
    out = {"meta": {"years": YEARS, "panel_max_ts": panel.max_ts, "sealed_load_count": panel.sealed_load_count,
                    "n_features": len(panel.columns), "seed": P9.SEED, "cost": P9.COST, "delta": P9.DELTA,
                    "refits": [rf.__dict__ for rf in refits]}}
    t0 = time.time()
    resM1 = P9.fit_predict_cell(panel, panel.X, panel.columns, refits, "M1")
    out["M1"] = read_cell(panel, resM1, "C0 ridge-all (control)")
    resM2 = P9.fit_predict_cell(panel, panel.X, panel.columns, refits, "M2")
    out["M2"] = read_cell(panel, resM2, "C1 fixed booster (primary)")
    out["meta"]["wall_seconds"] = time.time() - t0

    # --- C1b / C0b: PIT-safe variant (addendum 1): no-price-filter universe panel, price-level features dropped ---
    P9.PANEL_PATH = D / "ml_panel_train2023_nopx.parquet"
    P9.META_PATH = D / "ml_panel_train2023_nopx.meta.json"
    panelB = P9.Panel()
    mxB = pd.Timestamp(panelB.max_ts)
    assert panelB.sealed_load_count == 0
    assert mxB < pd.Timestamp("2023-01-01", tz="UTC") and mxB >= pd.Timestamp("2022-10-01", tz="UTC")
    metaB = json.loads(P9.META_PATH.read_text())
    assert metaB["dropped_features"] == ["gp"] and metaB.get("no_price_filter") is True
    dropB = set(PX_FEATS) | {f"miss_{c}" for c in PX_FEATS}
    keepB = [i for i, c in enumerate(panelB.columns) if c not in dropB]
    colsB = [panelB.columns[i] for i in keepB]
    assert len(colsB) == len(panelB.columns) - 2 * len(PX_FEATS), (len(colsB), len(panelB.columns))
    XB = np.ascontiguousarray(panelB.X[:, keepB])
    log(f"panel B (PIT-safe) n={panelB.n} max_ts={panelB.max_ts} design {len(colsB)} cols (dropped {sorted(dropB)})")
    refitsB = P9.make_refits(YEARS, 1)
    assert refitsB[0].win_end <= 2018 * 12 + 10
    out["meta"]["panelB"] = {"n": panelB.n, "max_ts": panelB.max_ts, "n_design_cols": len(colsB),
                             "dropped": sorted(dropB), "sealed_load_count": panelB.sealed_load_count}
    resM1b = P9.fit_predict_cell(panelB, XB, colsB, refitsB, "M1")
    out["M1b"] = read_cell(panelB, resM1b, "C0b ridge-all (PIT-safe control)")
    resM2b = P9.fit_predict_cell(panelB, XB, colsB, refitsB, "M2")
    out["M2b"] = read_cell(panelB, resM2b, "C1b fixed booster (PIT-safe)")
    out["meta"]["wall_seconds_total"] = time.time() - t0

    ref = json.load(open(RESULTS / "p9_e1.json"))["cells"]["base|M2"]
    sr = P9.cstats(np.asarray(ref["ic_all"], float), np.asarray(ref["cluster"]))

    def reads(key_m2: str, key_m1: str) -> dict:
        # paired dIC vs ridge (same months, same order)
        d = np.asarray(out[key_m2]["ic_all"]) - np.asarray(out[key_m1]["ic_all"])
        sd = P9.cstats(d, np.asarray(out[key_m2]["cluster"]))
        sf = out[key_m2]["stats_all"]
        # degradation vs P9 2009-2018 (two-sample, independent cluster SEs; descriptive)
        diff = sf["mean"] - sr["mean"]
        se = math.sqrt(sf["se_cluster"] ** 2 + sr["se_cluster"] ** 2)
        # pre-declared replication read
        m, tiid, lo = sf["mean"], sf["t_month_iid"], sf["ci_boot"][0]
        if m > 0 and tiid >= 2.0 and lo > 0:
            v = "REPLICATES (mean>0, per-month t>=2, cluster-bootstrap lower bound>0)"
        elif m > 0 and tiid >= 2.0:
            v = "REPLICATES AT IID POWER ONLY (per-month t>=2; cluster-bootstrap lower bound<=0, 4 clusters)"
        else:
            v = "DOES NOT REPLICATE"
        net = out[key_m2]["portfolio"]["ALL"]["net"]["ann_return"]
        return {"primary": v, "replicates": v.startswith("REPLICATES ("), "replicates_iid": v.startswith("REPLICATES"),
                "economic_secondary": f"net L/S ALL {net*100:+.2f}%/yr -> {'>0' if net > 0 else '<=0'}",
                "capacity_tertiary": P9.delta_verdict(sd, P9.DELTA, "capacity"), "dIC_stats": sd,
                "degradation_vs_p9": {"p9_2009_2018": sr["mean"], "fresh_2019_2022": sf["mean"], "diff": diff,
                                      "se": se, "z": diff / se if se > 0 else float("nan"),
                                      "ci95": [diff - 1.96 * se, diff + 1.96 * se]}}

    out["read_C1"] = reads("M2", "M1")
    out["read_C1b"] = reads("M2b", "M1b")
    a, b = out["read_C1"], out["read_C1b"]
    if a["replicates"] and b["replicates"]:
        combined = "REPLICATES, ROBUST TO THE SPLIT CHANNEL (C1 and C1b)"
    elif a["replicates"] and not b["replicates"]:
        combined = "REPLICATION NOT CLEAN - carried in part by the non-PIT split channel (C1 yes, C1b no); headline is C1b"
    elif not a["replicates"]:
        combined = "DOES NOT REPLICATE on C1" + (" (C1b replicates, reported, cannot rescue)" if b["replicates"] else "")
    if a["replicates_iid"] and not a["replicates"]:
        combined += " | C1 at iid power only"
    out["verdict"] = {"combined": combined, "C1": a["primary"], "C1b": b["primary"]}
    OUT.write_text(json.dumps(out, indent=1, default=float))
    for k, r in (("C1", a), ("C1b", b)):
        log(f"{k}: {r['primary']} | {r['economic_secondary']} | dIC vs ridge: {r['capacity_tertiary']} | "
            f"degradation z {r['degradation_vs_p9']['z']:+.2f}")
    log(f"COMBINED: {combined}")
    log(f"wrote {OUT.name}; sealed_load_count 0; panel max ts {panel.max_ts} / {panelB.max_ts}")


if __name__ == "__main__":
    main()
