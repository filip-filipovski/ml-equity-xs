#!/usr/bin/env python
"""Dumbest-composite benchmark for the P9 booster (labelled diagnostic, no fitting; the interrogation's "dumbest-composite
benchmark before I look again"). Signs are PRE-SPECIFIED from the literature, written here before any number was read:
mom_12_1 +, mom_6_1 +, rev_1m -, rev_1w -, vol_63 -, ivol_63 -, maxret_21 -, amihud_63 + (illiquidity premium),
log_dv_63 -, beta_252 - (betting against beta), skew_63 -, hi_252 + (52-week high), log_price - (low-price effect),
ann_ret_last + (post-announcement drift), log_mcap - (size), bm +, ep +, sp +, asset_growth -, accruals -, nsi -, roe +,
sue +. Excluded for want of a literature sign: days_since_ea, days_to_exp_ea, ea_timing, leverage, cash_assets (gp is not
in the panel). The composite is the equal-weight mean of sign x within-date-ranked feature (missing -> 0), i.e. no
parameter is estimated anywhere. Compared with the C1 M2 booster predictions cached by p9_b_followups.py on the same
rows (WFA panel, 2009-2023; 2009-2022 TRAIN, 2023 the Phase B read): rank IC by year, paired monthly IC difference,
decile 15 bps and neutral-cohort books. Addendum (added before its number was read): a classic five-characteristic
composite (mom_12_1 +, bm +, log_mcap -, roe +, nsi -) as the fairer "dumb" benchmark, because 23 equal-weighted noisy
features are expected to lose to fitted weights. Writes p9_b_dumb_composite.json."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
RESULTS = HERE.parent / "results"; RESULTS.mkdir(exist_ok=True)
import p9_ml_capacity as P9, p9_b_wfa_oneshot as PB, p10_books as BK  # noqa (PB patches the panel loader)
from eqxs import equity_ml as ml
D = P9.TL.CACHE / "equities_xs" / "derived"
P9.PANEL_PATH = D / "ml_panel_wfa2024.parquet"; P9.META_PATH = D / "ml_panel_wfa2024.meta.json"
SIGNS = {"mom_12_1": 1, "mom_6_1": 1, "rev_1m": -1, "rev_1w": -1, "vol_63": -1, "ivol_63": -1, "maxret_21": -1, "amihud_63": 1,
         "log_dv_63": -1, "beta_252": -1, "skew_63": -1, "hi_252": 1, "log_price": -1, "ann_ret_last": 1, "log_mcap": -1,
         "bm": 1, "ep": 1, "sp": 1, "asset_growth": -1, "accruals": -1, "nsi": -1, "roe": 1, "sue": 1}
YEARS = list(range(2009, 2024)); COST = 0.0015
def log(m): print(m, flush=True)
panel = P9.Panel(); rows = np.concatenate([panel.year_scorer(y).rows for y in YEARS])
frame = pd.read_parquet(RESULTS / "p9_cache" / "p9_b_c1_frame_2009_2023.parquet")
assert len(frame) == len(rows) and (frame["tmi"].to_numpy() == panel.tmi[rows]).all() and (frame["firm"].to_numpy() == panel.firm[rows]).all()
feats = [f for f in SIGNS if f in panel.col_ix]; missing = [f for f in SIGNS if f not in panel.col_ix]
log(f"composite features {len(feats)} (absent: {missing}); rows {len(rows):,}")
def composite(fs): return sum(SIGNS[f] * panel.X[rows, panel.col_ix[f]].astype(np.float64) for f in fs) / len(fs)
price_f = [f for f in feats if f in ml.PRICE_FEATURES]; fund_f = [f for f in feats if f in ml.FUND_FEATURES]
CLASSIC5 = ["mom_12_1", "bm", "log_mcap", "roe", "nsi"]
frame["comp"] = composite(feats); frame["comp_price"] = composite(price_f); frame["comp_fund"] = composite(fund_f); frame["comp_classic5"] = composite(CLASSIC5)
log(f"price-side features {price_f}\nfundamental-side features {fund_f}")
def rank_ic(g, col):
    g = g.dropna(subset=[col, "y"])
    return float(np.corrcoef(g[col].rank().to_numpy(), g["y"].rank().to_numpy())[0, 1]) if len(g) > 2 else np.nan
ics = {}
for col in ("pred", "comp", "comp_price", "comp_fund", "comp_classic5"):
    ics[col] = frame.groupby("tmi").apply(lambda g: rank_ic(g, col)).astype(float)
ic = pd.DataFrame(ics); ic["year"] = ic.index // 12
def by_year(s): return {int(y): float(s[ic["year"] == y].mean()) for y in YEARS}
def nw_t(x, L=3):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m; S = (e @ e)
    for l in range(1, L + 1): S += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l])
    return float(m / math.sqrt(S) * math.sqrt(n) / math.sqrt(n)) if S > 0 else float("nan")  # = m / sqrt(S/n^2)... see below
def nw_t2(x, L=3):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m; S = (e @ e) / n
    for l in range(1, L + 1): S += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    return float(m / math.sqrt(S / n))
WIN = {"2009_2018": (2009, 2018), "2019_2022": (2019, 2022), "2023": (2023, 2023), "2009_2022": (2009, 2022), "2009_2023": (2009, 2023)}
out = {"signs": SIGNS, "features_used": feats, "ic_by_year": {c: by_year(ic[c]) for c in ("pred", "comp", "comp_price", "comp_fund", "comp_classic5")}, "windows": {}}
for w, (a, b) in WIN.items():
    m = (ic["year"] >= a) & (ic["year"] <= b); sub = ic[m]
    d = (sub["pred"] - sub["comp"]).to_numpy(); d5 = (sub["pred"] - sub["comp_classic5"]).to_numpy()
    out["windows"][w] = {"n_months": int(m.sum()), **{c: {"mean": float(sub[c].mean()), "month_t": float(sub[c].mean() / sub[c].std(ddof=1) * math.sqrt(m.sum()))} for c in ("pred", "comp", "comp_price", "comp_fund", "comp_classic5")},
                         "booster_minus_composite": {"mean": float(d.mean()), "month_t": float(d.mean() / d.std(ddof=1) * math.sqrt(len(d))), "nw3_t": nw_t2(d), "frac_months_booster_higher": float((d > 0).mean())},
                         "booster_minus_classic5": {"mean": float(d5.mean()), "month_t": float(d5.mean() / d5.std(ddof=1) * math.sqrt(len(d5))), "nw3_t": nw_t2(d5), "frac_months_booster_higher": float((d5 > 0).mean())}}
    r = out["windows"][w]
    log(f"IC {w:9s} n={r['n_months']:3d} booster {r['pred']['mean']:+.4f} (t {r['pred']['month_t']:+.1f}) | dumb-{len(feats)} {r['comp']['mean']:+.4f} (t {r['comp']['month_t']:+.1f}) | price-only {r['comp_price']['mean']:+.4f} | fund-only {r['comp_fund']['mean']:+.4f} | booster-dumb {r['booster_minus_composite']['mean']:+.4f} (t {r['booster_minus_composite']['month_t']:+.1f}, NW3 {r['booster_minus_composite']['nw3_t']:+.1f}, booster higher {r['booster_minus_composite']['frac_months_booster_higher']:.0%} of months) | classic-5 {r['comp_classic5']['mean']:+.4f} (t {r['comp_classic5']['month_t']:+.1f}); booster-classic5 {r['booster_minus_classic5']['mean']:+.4f} (t {r['booster_minus_classic5']['month_t']:+.1f})")
log("IC by year booster: " + str({y: round(v, 3) for y, v in out['ic_by_year']['pred'].items()}))
log("IC by year dumb:    " + str({y: round(v, 3) for y, v in out['ic_by_year']['comp'].items()}))
log("IC by year classic5:" + str({y: round(v, 3) for y, v in out['ic_by_year']['comp_classic5'].items()}))
sic_cols = list(ml.SIC_COLS); out["books"] = {}
for name, col in (("booster", "pred"), ("dumb", "comp"), ("classic5", "comp_classic5")):
    fr = frame.copy(); fr["pred"] = fr[col]
    for bid, kind, hold, cost, liq in (("decile_15bps", "decile", 1, COST, False), ("neutral_cohort3_15bps", "neutral", 3, COST, False), ("neutral_cohort3_LIQ_5bps", "neutral", 3, 0.0005, True)):
        bk = BK.run_book(fr, sic_cols, kind, hold, cost, liq); out["books"].setdefault(name, {})[bid] = {}
        for w, (a, b) in WIN.items():
            mm = (bk["tmi"] // 12 >= a) & (bk["tmi"] // 12 <= b)
            if mm.sum() < 6: continue
            met = BK.series_metrics(bk["gross"][mm], bk["turnover"][mm], bk["cost"], bk["tmi"][mm]); out["books"][name][bid][w] = met
            log(f"BOOK {name:8s} {bid:26s} {w:9s} net {met['net_ann']*100:+6.2f}%/yr SR {met['net_sharpe']:+.2f} DD {met['maxdd']*100:+.1f}% turnover/mo {met['turnover_per_month']:.2f}")
(RESULTS / "p9_b_dumb_composite.json").write_text(json.dumps(out, indent=1, default=float)); log("wrote p9_b_dumb_composite.json")
