#!/usr/bin/env python
"""P9 Phase B — the equities WFA one-shot (addendum B1, committed before this runs). Frozen cells C1 / C1b (+ ridge
controls) on target months 2023-01 .. 2024-05 from panels built to 2024-07-01. Frozen pass rule per cell:
IC > 0 and per-month t >= 2 and net decile L/S (ALL, 15 bps) > 0. Co-declared secondary: 3-month-cohort book.
WFA rows are loaded through the equities-zone-gated loader with an explicit intent; nothing sealed is readable."""
from __future__ import annotations

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
import p9_e6_fresh_window as E6  # noqa: E402  (PX_FEATS, read_cell)
from eqxs.data_zones import safe_load_parquet  # noqa: E402
from eqxs.equities_xs import EQUITIES_ZONES_PATH  # noqa: E402

D = P9.TL.CACHE / "equities_xs" / "derived"
YEARS_WFA = [2023]  # B3: the straddling 2024 partitions are refused whole by the zone rule -> 2023 only (11 months)
YEARS_E6 = [2019, 2020, 2021, 2022]
OUT = RESULTS / "p9_b_wfa.json"
SEALED = pd.Timestamp("2024-07-01", tz="UTC")
COST = 0.0015


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def zoned_load(path, intent):
    df = safe_load_parquet(path, zone_config=EQUITIES_ZONES_PATH, caller_intent=intent)
    assert pd.to_datetime(df["ts"]).max() < SEALED, "panel reaches the sealed zone"
    return df


P9.TL.train_load = zoned_load  # Panel() reads through the equities zone gate with the WFA intent


def month_tests(ic: np.ndarray, n_flip: int = 200_000, seed: int = 7) -> dict:
    ic = np.asarray(ic, float); n = len(ic); m = ic.mean(); sd = ic.std(ddof=1)
    t_iid = m / (sd / math.sqrt(n))
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_flip, n)) * ic
    p_flip = float(((flips.mean(1) >= m).sum() + 1) / (n_flip + 1))
    # circular block bootstrap, L=3
    L, B = 3, 20_000
    starts = rng.integers(0, n, size=(B, math.ceil(n / L)))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n] % n
    bm = ic[idx].mean(1)
    # Newey-West t, lag 6
    def nw_t(x, lag):
        x = x - x.mean(); g0 = (x @ x) / n; s = g0
        for l in range(1, lag + 1):
            s += 2 * (1 - l / (lag + 1)) * (x[l:] @ x[:-l]) / n
        return m / math.sqrt(s / n)
    return {"mean": m, "sd": sd, "n": n, "t_iid": t_iid, "p_signflip": p_flip,
            "block_ci": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
            "nw_t_lag6": nw_t(ic, 6) if n > 7 else float("nan"),
            "share_pos": float((ic > 0).mean())}


def cohort_book(panel, years, preds_by_year, hold: int = 3, cost: float = COST, universe: str = "ALL") -> dict:
    """3-month-cohort EW decile L/S from monthly predictions (B1 co-declared secondary). Weights: each cohort EW within
    leg, 1/hold of the book; a name missing in a later month earns 0; turnover = sum|dw| per month; cost on traded
    notional. Also returns the hold=1 book from the same code for an apples-to-apples monthly comparison."""
    rows = np.concatenate([panel.year_scorer(y).rows for y in years])
    pred = np.concatenate([preds_by_year[y] for y in years])
    df = pd.DataFrame({"tmi": panel.tmi[rows], "firm": panel.firm[rows], "pred": pred, "y": panel.y_raw[rows],
                       "liq": panel.in_liq[rows]})
    if universe == "LIQ":
        df = df[df["liq"]]
    months = sorted(df["tmi"].unique())
    # cohort weights formed at each month (target month index = tmi)
    cohorts = {}
    for t in months:
        d = df[df["tmi"] == t]
        if len(d) < 50:
            continue
        q = d["pred"].rank(pct=True, method="first")
        longs, shorts = d["firm"][q > 0.9].to_numpy(), d["firm"][q <= 0.1].to_numpy()
        w = {}
        for f in longs: w[int(f)] = 1.0 / len(longs)
        for f in shorts: w[int(f)] = w.get(int(f), 0.0) - 1.0 / len(shorts)
        cohorts[t] = w
    ret_by = df.set_index(["tmi", "firm"])["y"]
    out = {}
    for H in (1, hold):
        prev = {}
        gross, net, to = [], [], []
        for t in months:
            active = [cohorts[t - k] for k in range(H) if (t - k) in cohorts]
            if not active:
                continue
            w = {}
            for c in active:
                for f, v in c.items():
                    w[f] = w.get(f, 0.0) + v / H
            r = 0.0
            for f, v in w.items():
                yv = ret_by.get((t, f), np.nan)
                r += v * (0.0 if (yv is None or not np.isfinite(yv)) else float(yv))
            keys = set(w) | set(prev)
            tov = sum(abs(w.get(f, 0.0) - prev.get(f, 0.0)) for f in keys)
            gross.append(r); to.append(tov); net.append(r - cost * tov)
            prev = w
        g, nn, tt = np.array(gross), np.array(net), np.array(to)
        rng = np.random.default_rng(11)
        B = 20_000; L = 3; n = len(nn)
        starts = rng.integers(0, n, size=(B, math.ceil(n / L)))
        idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n] % n
        ann = nn[idx].mean(1) * 12
        out[f"hold{H}"] = {"n_months": int(n), "gross_ann": float(g.mean() * 12), "net_ann": float(nn.mean() * 12),
                           "net_sharpe": float(nn.mean() / nn.std(ddof=1) * math.sqrt(12)) if nn.std(ddof=1) > 0 else float("nan"),
                           "ann_vol": float(nn.std(ddof=1) * math.sqrt(12)), "turnover_per_month": float(tt.mean()),
                           "net_ann_block_ci": [float(np.percentile(ann, 2.5)), float(np.percentile(ann, 97.5))],
                           "cost_ladder_net_ann": {str(int(c * 1e4)): float((g - c * tt).mean() * 12) for c in (0.0005, 0.0015, 0.0025, 0.005)}}
    return out


def run_panel(stem: str, design: str, years: list[int]) -> tuple[dict, dict]:
    P9.PANEL_PATH = D / f"{stem}.parquet"; P9.META_PATH = D / f"{stem}.meta.json"
    panel = P9.Panel()
    meta = json.loads(P9.META_PATH.read_text())
    assert meta["dropped_features"] == ["gp"] and meta.get("wfa_build") is True
    assert pd.Timestamp(panel.max_ts) < SEALED and pd.Timestamp(panel.max_ts) >= pd.Timestamp("2023-10-01", tz="UTC")
    if design == "f23":
        drop = set(E6.PX_FEATS) | {f"miss_{c}" for c in E6.PX_FEATS}
        keep = [i for i, c in enumerate(panel.columns) if c not in drop]
        X, cols = np.ascontiguousarray(panel.X[:, keep]), [panel.columns[i] for i in keep]
    else:
        X, cols = panel.X, panel.columns
    refits = P9.make_refits(years, 1)
    for rf in refits:
        log(f"  refit {rf.year}: fit tmi {rf.fit} val {rf.val} test {rf.test} win_end {rf.win_end}")
    assert refits[0].win_end <= 2022 * 12 + 10  # 2023 refit trains on <= 2022-11 targets (TRAIN only)
    res = {}
    for m in ("M1", "M2"):
        r = P9.fit_predict_cell(panel, X, cols, refits, m)
        cell = E6.read_cell(panel, r, f"{stem}|{design}|{m}")
        cell["month_tests"] = month_tests(cell["ic_all"])
        cell["month_tests_liq"] = month_tests(cell["ic_liq"])
        cell["cohort_book_ALL"] = cohort_book(panel, years, r["pred"], 3, COST, "ALL")
        cell["cohort_book_LIQ"] = cohort_book(panel, years, r["pred"], 3, 0.0005, "LIQ")
        cell["preds"] = {int(y): r["pred"][y] for y in r["pred"]}
        res[m] = cell
    return res, {"n": panel.n, "max_ts": panel.max_ts, "n_design": len(cols), "sealed_load_count": panel.sealed_load_count,
                 "panel": panel}


def verdict(cell: dict) -> dict:
    mt = cell["month_tests"]; net = cell["portfolio"]["ALL"]["net"]["ann_return"]
    ok = (mt["mean"] > 0) and (mt["t_iid"] >= 2.0) and (net > 0)
    return {"ic": mt["mean"], "t_iid": mt["t_iid"], "p_signflip": mt["p_signflip"], "block_ci": mt["block_ci"],
            "net_ann_15bps": net, "pass": bool(ok),
            "text": ("SURVIVED the WFA one-shot (reduced power)" if ok else "WFA FAIL (zone spent)")
                    + f" - IC {mt['mean']:+.4f}, per-month t {mt['t_iid']:+.2f}, net {net*100:+.2f}%/yr"}


def main() -> None:
    t0 = time.time()
    out = {"meta": {"years_wfa": YEARS_WFA, "cost": COST, "rule": "IC>0 & per-month t>=2 & net L/S(ALL,15bps)>0",
                    "predeclaration": "P9_E6_PREDECLARATION.md addendum B1"}}
    # ---- the one read: WFA months ----
    resA, infoA = run_panel("ml_panel_wfa2024", "f28", YEARS_WFA)
    resB, infoB = run_panel("ml_panel_wfa2024_nopx", "f23", YEARS_WFA)
    out["meta"]["panelA"] = {k: v for k, v in infoA.items() if k != "panel"}
    out["meta"]["panelB"] = {k: v for k, v in infoB.items() if k != "panel"}
    for key, cell in (("C1", resA["M2"]), ("C0", resA["M1"]), ("C1b", resB["M2"]), ("C0b", resB["M1"])):
        cell = {k: v for k, v in cell.items() if k != "preds"}
        out[key] = cell
    out["verdict"] = {"C1": verdict(resA["M2"]), "C1b": verdict(resB["M2"]),
                      "controls": {"C0": verdict(resA["M1"]), "C0b": verdict(resB["M1"])}}
    # labelled diagnostic: C1b predictions on the >= $5 rows (red-team 'if one number' book) -- needs C1 universe rows
    pB = infoB["panel"]; pA = infoA["panel"]
    rowsB = np.concatenate([pB.year_scorer(y).rows for y in YEARS_WFA]); predB = np.concatenate([resB["M2"]["preds"][y] for y in YEARS_WFA])
    keyB = pd.DataFrame({"tmi": pB.tmi[rowsB], "sym": pB.firm[rowsB], "pred": predB})
    # map via (date, symbol) is not available from firm ids across panels; use the nopx panel's own price proxy instead:
    # rows whose raw_log_price >= log(5) -> approximate >= $5 universe within panel B
    out["diag_note"] = "C1b-on->=$5 diagnostic computed within panel B using raw_log_price >= log(5)"
    dfB = zoned_load(D / "ml_panel_wfa2024_nopx.parquet", "P9 B diag: price proxy")
    dfB = dfB.iloc[rowsB]
    mask = (dfB["raw_log_price"].to_numpy() >= math.log(5.0))
    sub_rows = rowsB[mask]; sub_pred = predB[mask]
    out["diag_C1b_on_ge5"] = P9.portfolio_stats(pB, sub_rows, sub_pred, "ALL")
    # ---- E6 window cohort books (secondary, same code) from the TRAIN-2023 panels, for comparison ----
    P9.TL.train_load = lambda path, intent: zoned_load(path, intent)
    e6 = {}
    for stem, design, key in (("ml_panel_train2023", "f28", "C1"), ("ml_panel_train2023_nopx", "f23", "C1b")):
        P9.PANEL_PATH = D / f"{stem}.parquet"; P9.META_PATH = D / f"{stem}.meta.json"
        panel = P9.Panel()
        if design == "f23":
            drop = set(E6.PX_FEATS) | {f"miss_{c}" for c in E6.PX_FEATS}
            keep = [i for i, c in enumerate(panel.columns) if c not in drop]
            X, cols = np.ascontiguousarray(panel.X[:, keep]), [panel.columns[i] for i in keep]
        else:
            X, cols = panel.X, panel.columns
        r = P9.fit_predict_cell(panel, X, cols, P9.make_refits(YEARS_E6, 1), "M2")
        e6[key] = {"cohort_book_ALL": cohort_book(panel, YEARS_E6, r["pred"], 3, COST, "ALL"),
                   "cohort_book_LIQ": cohort_book(panel, YEARS_E6, r["pred"], 3, 0.0005, "LIQ")}
    out["e6_window_cohort_books"] = e6
    out["meta"]["wall_seconds"] = time.time() - t0
    OUT.write_text(json.dumps(out, indent=1, default=float))
    for k in ("C1", "C1b"):
        v = out["verdict"][k]; log(f"{k}: {v['text']} | sign-flip p {v['p_signflip']:.4f} block CI [{v['block_ci'][0]:+.4f},{v['block_ci'][1]:+.4f}]")
    for k in ("C0", "C0b"):
        v = out["verdict"]["controls"][k]; log(f"{k} (control): {v['text']}")
    for k in ("C1", "C1b"):
        cb = out[k]["cohort_book_ALL"]
        log(f"{k} books ALL: monthly(hold1) net {cb['hold1']['net_ann']*100:+.2f}%/yr SR {cb['hold1']['net_sharpe']:+.2f} to/mo {cb['hold1']['turnover_per_month']:.2f} | "
            f"3-cohort net {cb['hold3']['net_ann']*100:+.2f}%/yr SR {cb['hold3']['net_sharpe']:+.2f} to/mo {cb['hold3']['turnover_per_month']:.2f} CI [{cb['hold3']['net_ann_block_ci'][0]*100:+.1f},{cb['hold3']['net_ann_block_ci'][1]*100:+.1f}]")
    log(f"diag C1b-on->=$5 (ALL cost): net {out['diag_C1b_on_ge5']['net']['ann_return']*100:+.2f}%/yr SR {out['diag_C1b_on_ge5']['net']['sharpe']:+.2f}")
    for k in ("C1", "C1b"):
        cb = e6[k]["cohort_book_ALL"]
        log(f"E6-window {k} books ALL: hold1 net {cb['hold1']['net_ann']*100:+.2f}%/yr SR {cb['hold1']['net_sharpe']:+.2f} | hold3 net {cb['hold3']['net_ann']*100:+.2f}%/yr SR {cb['hold3']['net_sharpe']:+.2f} to/mo {cb['hold3']['turnover_per_month']:.2f}")
    log(f"wrote {OUT.name}; sealed_load_count 0; panel max ts {infoA['max_ts']} / {infoB['max_ts']}")


if __name__ == "__main__":
    main()
