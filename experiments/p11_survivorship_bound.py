#!/usr/bin/env python
"""P11 — a measured survivorship bound for the E6 / Phase B decile legs (labelled diagnostic, assumptions fixed here
BEFORE any number is read).

Problem: the price panel is the 2026-10 listing (survivors). Firms that existed at a formation date T but delisted
before 2026-10 are absent from every formation, including their final year, when their returns are extreme.
Data (all free, already cached): SEC 10-K/10-Q filing dates per CIK (`sec_fsds/sub`, read to 2023-12-31 through the
zone-gated loader); `filer_membership.last_periodic_filed` (a 2026-vintage survival label - used ONLY to label
'stopped filing within 12 months of T', never for returns); XBRL Assets / NetIncomeLoss (PIT by acceptance `ts` <= T,
`sec_fsds/num`); 8-K item flags (`events/eightk_items`, quarters <= 2023q4) to classify the cause of disappearance.

Population at T: operating filers (entity_type 'operating') that filed a 10-K or 10-Q in the 365 days before T with PIT
total assets >= $100M (a 'listed-size' proxy; filers are a superset of listed stocks, so shares are upper bounds for
the listed universe). Death12(T): last periodic filing <= T + 365 d. Cause, from 8-K items in [last filing - 365 d,
last filing + 90 d] (deaths <= 2023-12 only; later deaths 'unclassified'): 'failure' if item 1.03 (bankruptcy) or 3.01
without 5.01/2.01; 'm&a' if 5.01 or 2.01 and no 1.03; else 'dark'.
Bins: PIT-assets terciles within the population at T x profitable (NetIncome > 0) / loss-making.
Legs: the Phase-B/E6 model's top and bottom deciles at T (M2, C1 design), mapped to the same bins via CIK.
Bound: for each leg and year, missing_share = sum_bins w_leg(bin) * death12_rate(bin), split by cause; the bias on the
leg's annual return = sum_cause share_cause * ASSUMED abnormal final-year return, with ASSUMPTIONS FIXED NOW:
failure -40 %, m&a +20 %, dark -20 %, unclassified = population-weighted mix of the classified causes. Reported with the
assumptions and as a range (x0.5 .. x2 on the assumed magnitudes). Formation dates: every third month 2014-01 .. 2023-10.
"""
from __future__ import annotations
import json, sys, glob
EX_SPAC = "--ex-spac" in sys.argv  # red-team F3: exclude blank-check filers (SIC 6770, current SIC from filer_membership)
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import p9_ml_capacity as P9, p9_b_wfa_oneshot as PB  # noqa (PB patches the panel loader)
from eqxs import equities_xs as xs
ROOT = P9.TL.ROOT; X = xs.XS_ROOT; D = X / "derived"
ASSUME = {"failure": -0.40, "m&a": 0.20, "dark": -0.20}
END = pd.Timestamp("2024-01-01")  # last reachable filing/8-K date (2024 partitions straddle the sealed start)
T_DATES = pd.date_range("2014-01-31", "2023-10-31", freq="3ME")

def log(m): print(m, flush=True)

# --- filings (10-K/10-Q) 2013-2023 ---
sub = xs._load_partitioned(X / "sec_fsds" / "sub", "year", intent="p11 survivorship: filings", columns=["cik", "form", "filed", "ts"], start="2013-01-01", end=END)
sub = sub[sub["form"].isin(["10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "10-QT"])].copy()
sub["filed"] = pd.to_datetime(sub["filed"]); log(f"filings 2013-2023: {len(sub):,} rows, {sub['cik'].nunique():,} CIKs")
fm = pd.read_parquet(X / "filer_membership.parquet"); fm["last"] = pd.to_datetime(fm["last_periodic_filed"]); fm = fm.set_index("cik")
log(f"filer_membership: {len(fm):,} CIKs; last_periodic_filed max {fm['last'].max().date()} (2026-vintage survival label)")
# --- 8-K items <= 2023q4 (direct read of pre-sealed partitions; ts < 2024-01-01 asserted) ---
parts = [f for f in sorted(glob.glob(str(X / "events" / "eightk_items" / "quarter=*" / "*.parquet"))) if "quarter=20" in f and f.split("quarter=")[1][:6] <= "2023q4" and f.split("quarter=")[1][:4] >= "2012"]
ek = pd.concat([pq.read_table(f, columns=["cik", "ts", "item_1_03", "item_3_01", "item_5_01", "item_2_01"]).to_pandas() for f in parts], ignore_index=True)
ek["ts"] = pd.to_datetime(ek["ts"], utc=True).dt.tz_localize(None); assert ek["ts"].max() < END; log(f"8-K items 2012-2023q4: {len(ek):,} rows")
# --- cause of disappearance per CIK, computed once (vectorised): 8-K items in [last-365d, last+90d] ---
_fm = fm[["last"]].reset_index(); _fm = _fm[_fm["last"] < END]
_w = ek.merge(_fm, on="cik", how="inner"); _w = _w[(_w["ts"] >= _w["last"] - pd.Timedelta(days=365)) & (_w["ts"] <= _w["last"] + pd.Timedelta(days=90))]
_agg = _w.groupby("cik")[["item_1_03", "item_3_01", "item_5_01", "item_2_01"]].sum().astype(float)
CAUSE_BY_CIK = {}
for cik_, r_ in _agg.iterrows():
    if r_["item_1_03"] > 0: CAUSE_BY_CIK[int(cik_)] = "failure"
    elif (r_["item_5_01"] + r_["item_2_01"]) > 0: CAUSE_BY_CIK[int(cik_)] = "m&a"
    elif r_["item_3_01"] > 0: CAUSE_BY_CIK[int(cik_)] = "failure"
    else: CAUSE_BY_CIK[int(cik_)] = "dark"
for cik_ in fm.index[fm["last"] >= END]: CAUSE_BY_CIK[int(cik_)] = "unclassified"
log(f"cause table: {len(CAUSE_BY_CIK):,} CIKs ({sum(v == 'failure' for v in CAUSE_BY_CIK.values()):,} failure, {sum(v == 'm&a' for v in CAUSE_BY_CIK.values()):,} m&a, {sum(v == 'dark' for v in CAUSE_BY_CIK.values()):,} dark, {sum(v == 'unclassified' for v in CAUSE_BY_CIK.values()):,} unclassified/post-2023)")
# --- PIT assets / net income from XBRL (Assets instant; NetIncomeLoss ~annual duration), by availability year ---
tags = ["Assets", "NetIncomeLoss"]; facts = []
for y in range(2012, 2024):
    num = xs.load_sec_num(tags=tags, start=f"{y}-01-01", end=f"{y + 1}-01-01")
    if len(num):
        num = num[num["uom"] == "USD"][["cik", "tag", "ts", "value", "ddate", "qtrs"]] if "qtrs" in num.columns else num[num["uom"] == "USD"][["cik", "tag", "ts", "value", "ddate"]]
        facts.append(num)
    log(f"  num {y}: {len(num):,}")
num = pd.concat(facts, ignore_index=True); num["ts"] = pd.to_datetime(num["ts"], utc=True).dt.tz_localize(None); num["ddate"] = pd.to_datetime(num["ddate"])
assets = num[num["tag"] == "Assets"].sort_values("ts"); ni = num[num["tag"] == "NetIncomeLoss"]
if "qtrs" in ni.columns: ni = ni[ni["qtrs"].isin([4])]  # annual durations only
ni = ni.sort_values("ts")
def pit_latest(df, T):
    d = df[df["ts"] <= T]; return d.groupby("cik")["value"].last()
# --- the model's legs (C1 design, M2) on the WFA panel for 2019-2023; earlier years for context ---
P9.PANEL_PATH = D / "ml_panel_wfa2024.parquet"; P9.META_PATH = D / "ml_panel_wfa2024.meta.json"
panel = P9.Panel(); pdf = P9.TL.train_load(P9.PANEL_PATH, "p11 legs frame")
years = list(range(2014, 2024)); res = P9.fit_predict_cell(panel, panel.X, panel.columns, P9.make_refits(years, 1), "M2")
rows = np.concatenate([panel.year_scorer(y).rows for y in years]); pred = np.concatenate([res["pred"][y] for y in years])
legs = pd.DataFrame({"tmi": panel.tmi[rows], "cik": pdf["cik"].to_numpy()[rows], "pred": pred})
legs["date"] = pd.to_datetime(pdf["date"].to_numpy()[rows])
out_rows = []
for T in T_DATES:
    active = sub[(sub["filed"] <= T) & (sub["filed"] > T - pd.Timedelta(days=365))]["cik"].unique()
    A = pit_latest(assets, T); N = pit_latest(ni, T)
    pop = pd.DataFrame({"cik": active}).merge(A.rename("assets"), left_on="cik", right_index=True, how="left").merge(N.rename("ni"), left_on="cik", right_index=True, how="left")
    pop = pop.merge(fm[["last", "entity_type", "sic"]], left_on="cik", right_index=True, how="left")
    pop = pop[(pop["entity_type"] == "operating") & (pop["assets"] >= 1e8)].copy()
    if EX_SPAC: pop = pop[pd.to_numeric(pop["sic"], errors="coerce") != 6770].copy()
    if len(pop) < 200: continue
    pop["death12"] = pop["last"] <= T + pd.Timedelta(days=365)
    pop["size_bin"] = pd.qcut(pop["assets"].rank(method="first"), 3, labels=["small", "mid", "large"])
    pop["profit"] = np.where(pop["ni"] > 0, "profit", "loss")
    pop["cause"] = np.where(pop["death12"], pop["cik"].map(CAUSE_BY_CIK).fillna("dark"), None)
    # bin death rates by cause
    g = pop.groupby(["size_bin", "profit"], observed=True)
    rate = {k: {"n": int(len(v)), "death12": float(v["death12"].mean()), **{c: float((v["cause"] == c).mean()) for c in ("failure", "m&a", "dark", "unclassified")}} for k, v in g}
    # legs at the formation month (target month = T's month + 1 -> tmi of the holding month)
    tm = (T.year * 12 + T.month - 1) + 1
    m = legs[legs["tmi"] == tm]
    if len(m) < 100: continue
    q = m["pred"].rank(pct=True, method="first"); L = m[q > 0.9]["cik"].dropna().astype(int); S = m[q <= 0.1]["cik"].dropna().astype(int)
    popi = pop.set_index("cik")
    def leg_stats(ciks):
        sub_ = popi.reindex(ciks.to_numpy()); sub_ = sub_[sub_["size_bin"].notna()]
        if len(sub_) == 0: return None
        comp = sub_.groupby(["size_bin", "profit"], observed=True).size() / len(sub_)
        miss = {c: 0.0 for c in ("failure", "m&a", "dark", "unclassified")}; d12 = 0.0
        for k, w in comp.items():
            r = rate.get(k); 
            if not r: continue
            d12 += w * r["death12"]
            for c in miss: miss[c] += w * r[c]
        # unclassified -> distribute like the classified mix of the population at T
        cls = {c: float((pop["cause"] == c).sum()) for c in ("failure", "m&a", "dark")}; tot = sum(cls.values()) or 1.0
        unc = miss.pop("unclassified")
        for c in cls: miss[c] += unc * cls[c] / tot
        bias = sum(miss[c] * ASSUME[c] for c in miss)  # expected abnormal return of the missing population share
        return {"n_matched": int(len(sub_)), "share_small_loss": float(comp.get(("small", "loss"), 0.0)), "death12": d12, **{f"miss_{c}": v for c, v in miss.items()}, "bias_on_leg_return": bias}
    out_rows.append({"T": str(T.date()), "pop_n": int(len(pop)), "pop_death12": float(pop["death12"].mean()),
                     "pop_cause": {c: float((pop["cause"] == c).mean()) for c in ("failure", "m&a", "dark", "unclassified")},
                     "long": leg_stats(L), "short": leg_stats(S), "rates": {f"{a}|{b}": v for (a, b), v in rate.items()}})
    r = out_rows[-1]; log(f"{r['T']}: pop {r['pop_n']:,} death12 {r['pop_death12']:.1%} causes {{f:{r['pop_cause']['failure']:.1%}, m&a:{r['pop_cause']['m&a']:.1%}, dark:{r['pop_cause']['dark']:.1%}, unc:{r['pop_cause']['unclassified']:.1%}}} | "
        f"long death12 {r['long']['death12'] if r['long'] else float('nan'):.1%} bias {r['long']['bias_on_leg_return'] if r['long'] else float('nan'):+.2%} | short death12 {r['short']['death12'] if r['short'] else float('nan'):.1%} bias {r['short']['bias_on_leg_return'] if r['short'] else float('nan'):+.2%}")
res_df = pd.DataFrame([{"T": r["T"], "year": int(r["T"][:4]), "pop_death12": r["pop_death12"], "long_d12": r["long"]["death12"] if r["long"] else np.nan, "short_d12": r["short"]["death12"] if r["short"] else np.nan,
                        "long_bias": r["long"]["bias_on_leg_return"] if r["long"] else np.nan, "short_bias": r["short"]["bias_on_leg_return"] if r["short"] else np.nan} for r in out_rows])
summary = {}
for lab, yrs in (("2014_2018", range(2014, 2019)), ("2019_2022", range(2019, 2023)), ("2023", [2023])):
    s = res_df[res_df["year"].isin(list(yrs))]
    if len(s) == 0: continue
    # spread bias = long bias - short bias (measured spread = true - (long_bias - short_bias))  [missing firms' abnormal returns absent]
    summary[lab] = {"n_T": int(len(s)), "pop_death12": float(s["pop_death12"].mean()), "long_death12": float(s["long_d12"].mean()), "short_death12": float(s["short_d12"].mean()),
                    "long_bias_pct": float(s["long_bias"].mean() * 100), "short_bias_pct": float(s["short_bias"].mean() * 100),
                    "measured_minus_true_spread_pct": float((-(s["long_bias"].mean()) + s["short_bias"].mean()) * 100)}
    v = summary[lab]; log(f"SUMMARY {lab}: pop death12 {v['pop_death12']:.1%} | long leg missing/yr {v['long_death12']:.1%} (abnormal {v['long_bias_pct']:+.2f} pp) | short leg missing/yr {v['short_death12']:.1%} (abnormal {v['short_bias_pct']:+.2f} pp) | measured spread minus true ≈ {v['measured_minus_true_spread_pct']:+.2f} pp/yr under the fixed assumptions (range x0.5..x2)")
(HERE / ("p11_survivorship_bound_exspac.json" if EX_SPAC else "p11_survivorship_bound.json")).write_text(json.dumps({"assumptions": ASSUME, "rows": out_rows, "summary": summary}, indent=1, default=float))
log("wrote " + ("p11_survivorship_bound_exspac.json" if EX_SPAC else "p11_survivorship_bound.json"))
