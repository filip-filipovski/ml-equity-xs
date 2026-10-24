#!/usr/bin/env python
"""P9 Phase B follow-ups (labelled diagnostics; C1 design, M2 fixed booster, WFA panel, refits 2009-2023 from the fit
cache). Nothing here is a new read: 2009-2022 is TRAIN, 2023 is the Phase B read already taken.
(1) Factor attribution of the decile / neutral-cohort / liquid books: French FF3+MOM and FF5+MOM, monthly, Newey-West t,
    windows 2009-18, 2019-22 (the E6 window on this panel), 2023, 2009-22, 2009-23. Book month t (formation month-end t,
    return earned over calendar month t+1) is matched to factor month t+1 (daily factors compounded within the month).
(2) Pass probability of the frozen Phase-B rule (IC > 0 and per-month t >= 2 and net decile L/S > 0, 11 months) under a
    true monthly IC of 0 .. 0.087: contiguous circular 11-month windows drawn from the model's own demeaned 2009-2022
    monthly (IC, net return) series, shifted by mu (return shifted by the fitted slope of return on IC); block-3 variant.
(3) Prior update for the decile net Sharpe: pooled 2009-2023; normal-normal update of the 2009-22 estimate with the 2023
    observation under the Lo (1+SR^2/2)/T and the naive 1/T standard errors.
(4) IC trend within the fixed cohort of firms present at the first 2009 formation (composition control), with the
    liquid-half trend (p9_b_liq_trend.json) copied in.
Writes p9_b_followups.json and caches the scored frame under p9_cache/ (not committed)."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
RESULTS = HERE.parent / "results"; RESULTS.mkdir(exist_ok=True)
import p9_ml_capacity as P9, p9_b_wfa_oneshot as PB, p10_books as BK  # noqa (PB patches the panel loader)
from eqxs import equity_ml as ml
from eqxs import equities_xs as xs

D = P9.TL.CACHE / "equities_xs" / "derived"
P9.PANEL_PATH = D / "ml_panel_wfa2024.parquet"; P9.META_PATH = D / "ml_panel_wfa2024.meta.json"
YEARS = list(range(2009, 2024)); COST = 0.0015; RULE_T = 2.0; N_WFA = 11; IC_2023 = 0.0869
def log(m): print(m, flush=True)

panel = P9.Panel(); df = P9.TL.train_load(P9.PANEL_PATH, "followups frame")
res = P9.fit_predict_cell(panel, panel.X, list(panel.columns), P9.make_refits(YEARS, 1), "M2")
log(f"fit cached: {res.get('cached')}")
rows = np.concatenate([panel.year_scorer(y).rows for y in YEARS]); pred = np.concatenate([res["pred"][y] for y in YEARS])
frame = pd.DataFrame({"tmi": panel.tmi[rows], "firm": panel.firm[rows], "y": panel.y_raw[rows], "in_liq": panel.in_liq[rows],
                      "beta": df["raw_beta_252"].to_numpy(np.float64)[rows], "pred": pred})
sic_cols = list(ml.SIC_COLS)
for c in sic_cols: frame[c] = panel.X[rows, panel.col_ix[c]]
(RESULTS / "p9_cache").mkdir(exist_ok=True); frame.to_parquet(RESULTS / "p9_cache" / "p9_b_c1_frame_2009_2023.parquet")
log(f"frame {len(frame):,} rows, tmi {frame.tmi.min()}..{frame.tmi.max()} ({frame.tmi.nunique()} months)")

# ---- monthly rank IC (all names) -------------------------------------------------------------------------------------
def rank_ic(g):
    g = g.dropna(subset=["pred", "y"])
    if len(g) < 3: return np.nan
    return float(np.corrcoef(g["pred"].rank().to_numpy(), g["y"].rank().to_numpy())[0, 1])
icm = frame.groupby("tmi").apply(rank_ic).astype(float)

# ---- books -------------------------------------------------------------------------------------------------------------
books = {"decile_15bps": BK.run_book(frame, sic_cols, "decile", 1, COST, False),
         "neutral_cohort3_15bps": BK.run_book(frame, sic_cols, "neutral", 3, COST, False),
         "neutral_cohort3_LIQ_5bps": BK.run_book(frame, sic_cols, "neutral", 3, 0.0005, True)}
def net_series(bk): return pd.Series(bk["gross"] - bk["cost"] * bk["turnover"], index=bk["tmi"].astype(int))

# ---- factors (pre-sealed rows only; end exclusive at the sealed start) -------------------------------------------------
ff = xs.load_ff_daily(end="2024-07-01").copy()
ix = pd.DatetimeIndex(ff.index); ix = ix.tz_convert(None) if ix.tz is not None else ix
ff.index = ix; ff["tmi"] = ff.index.year * 12 + ff.index.month - 1
FAC = ["Mkt-RF", "SMB", "HML", "MOM", "RMW", "CMA"]
fm = ff.groupby("tmi")[FAC].apply(lambda g: (1 + g).prod() - 1)
log(f"factors monthly {fm.index.min()}..{fm.index.max()} ({len(fm)} months); last daily {ff.index.max().date()}")

def ols_nw(y, X, L):
    n, k = X.shape; XtX_inv = np.linalg.inv(X.T @ X); b = XtX_inv @ X.T @ y; e = y - X @ b
    Xe = X * e[:, None]; S = Xe.T @ Xe
    for l in range(1, L + 1):
        G = Xe[l:].T @ Xe[:-l]; S += (1 - l / (L + 1)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv; se = np.sqrt(np.diag(V))
    return b, b / se, float(1 - e.var(ddof=0) / y.var(ddof=0)), int(n)

WINDOWS = {"2009_2018": (2009, 2018), "2019_2022": (2019, 2022), "2023": (2023, 2023), "2009_2022": (2009, 2022), "2009_2023": (2009, 2023)}
MODELS = {"FF3+MOM": ["Mkt-RF", "SMB", "HML", "MOM"], "FF5+MOM": FAC, "CAPM": ["Mkt-RF"]}
attrib = {}
for bname, bk in books.items():
    r = net_series(bk); attrib[bname] = {}
    for wname, (y0, y1) in WINDOWS.items():
        rw = r[(r.index >= y0 * 12) & (r.index <= y1 * 12 + 11)]
        f = fm.reindex(rw.index + 1)  # return earned over calendar month t+1
        ok = np.isfinite(f.to_numpy()).all(1) & np.isfinite(rw.to_numpy())
        attrib[bname][wname] = {"n_months": int(ok.sum()), "raw_net_ann": float(rw[ok].mean() * 12),
                                "raw_sharpe": float(rw[ok].mean() / rw[ok].std(ddof=1) * math.sqrt(12))}
        for mname, cols in MODELS.items():
            if ok.sum() < len(cols) + 6: attrib[bname][wname][mname] = None; continue
            X = np.column_stack([np.ones(ok.sum()), f.loc[ok, cols].to_numpy()]); b, t, r2, n = ols_nw(rw[ok].to_numpy(), X, 3)
            attrib[bname][wname][mname] = {"alpha_ann": float(b[0] * 12), "alpha_t_nw3": float(t[0]), "r2": r2,
                                           "loadings": {c: float(v) for c, v in zip(cols, b[1:])}, "loading_t": {c: float(v) for c, v in zip(cols, t[1:])}}
    for wname in WINDOWS:
        a = attrib[bname][wname]; m4 = a.get("FF3+MOM"); m6 = a.get("FF5+MOM")
        log(f"ATTRIB {bname:26s} {wname:9s} n={a['n_months']:3d} raw {a['raw_net_ann']*100:+6.2f}%/yr SR {a['raw_sharpe']:+.2f} | FF3+MOM alpha "
            + (f"{m4['alpha_ann']*100:+6.2f}%/yr t {m4['alpha_t_nw3']:+.2f} R2 {m4['r2']:.2f} load mkt {m4['loadings']['Mkt-RF']:+.2f} smb {m4['loadings']['SMB']:+.2f} hml {m4['loadings']['HML']:+.2f} mom {m4['loadings']['MOM']:+.2f}" if m4 else "n/a")
            + (f" | FF5+MOM alpha {m6['alpha_ann']*100:+6.2f}%/yr t {m6['alpha_t_nw3']:+.2f} rmw {m6['loadings']['RMW']:+.2f} cma {m6['loadings']['CMA']:+.2f}" if m6 else " | FF5+MOM n/a"))

# ---- pass probability of the frozen rule under a true IC mu ------------------------------------------------------------
rd = net_series(books["decile_15bps"]); common = icm.index.intersection(rd.index)
pre = common[common < 2023 * 12]; IC = icm.loc[pre].to_numpy(); R = rd.loc[pre].to_numpy()
ok = np.isfinite(IC) & np.isfinite(R); IC, R = IC[ok], R[ok]; n_pre = len(IC)
mu_ic, mu_r = IC.mean(), R.mean(); slope = float(np.cov(IC, R)[0, 1] / IC.var(ddof=1))
ICc, Rc = IC - mu_ic, R - mu_r
rng = np.random.default_rng(20261017); B = 50_000
starts = rng.integers(0, n_pre, B); idx_c = (starts[:, None] + np.arange(N_WFA)[None, :]) % n_pre
st3 = rng.integers(0, n_pre, (B, math.ceil(N_WFA / 3))); idx_3 = ((st3[:, :, None] + np.arange(3)[None, None, :]).reshape(B, -1)[:, :N_WFA]) % n_pre
def pass_prob(idx, mu):
    ic_s = ICc[idx] + mu; r_s = Rc[idx] + slope * mu
    m = ic_s.mean(1); t = m / ic_s.std(1, ddof=1) * math.sqrt(N_WFA); npos = r_s.mean(1) > 0
    return {"P_pass": float(((m > 0) & (t >= RULE_T) & npos).mean()), "P_t_ge_2": float((t >= RULE_T).mean()),
            "P_ic_ge_2023": float((m >= IC_2023).mean()), "P_net_pos": float(npos.mean()), "mean_ic": float(m.mean())}
MUS = [0.0, 0.01, 0.02, 0.03, 0.036, 0.04, 0.05, 0.06, IC_2023]
passp = {"n_months_2009_2022": n_pre, "ic_mean_2009_2022": float(mu_ic), "ic_sd_monthly": float(IC.std(ddof=1)),
         "ic_lag1_autocorr": float(np.corrcoef(ICc[1:], ICc[:-1])[0, 1]), "slope_net_ret_on_ic": slope, "B": B,
         "contiguous_11m": {str(m): pass_prob(idx_c, m) for m in MUS}, "block3": {str(m): pass_prob(idx_3, m) for m in MUS}}
for m in MUS:
    c, b3 = passp["contiguous_11m"][str(m)], passp["block3"][str(m)]
    log(f"PASSPROB mu={m:.3f}: contiguous P(pass) {c['P_pass']:.3f} P(t>=2) {c['P_t_ge_2']:.3f} P(IC>=0.087) {c['P_ic_ge_2023']:.3f} | block3 P(pass) {b3['P_pass']:.3f} P(IC>=0.087) {b3['P_ic_ge_2023']:.3f}")
log(f"PASSPROB inputs: n={n_pre} monthly IC sd {IC.std(ddof=1):.4f} lag1 ac {passp['ic_lag1_autocorr']:+.2f} slope ret/IC {slope:.3f}/month")

# ---- prior update for the decile Sharpe ------------------------------------------------------------------------------
r14 = rd[(rd.index >= 2009 * 12) & (rd.index < 2023 * 12)]; r23 = rd[rd.index >= 2023 * 12]; rall = rd[rd.index >= 2009 * 12]
def sr(x): return float(x.mean() / x.std(ddof=1) * math.sqrt(12))
sr14, sr23, srall = sr(r14), sr(r23), sr(rall); T14, T23 = len(r14) / 12, len(r23) / 12
se14 = math.sqrt((1 + sr14 ** 2 / 2) / T14); se23_lo = math.sqrt((1 + sr23 ** 2 / 2) / T23); se23_naive = math.sqrt(1 / T23)
def post(se23): w1, w2 = 1 / se14 ** 2, 1 / se23 ** 2; return float((sr14 * w1 + sr23 * w2) / (w1 + w2)), float(math.sqrt(1 / (w1 + w2)))
prior = {"sr_2009_2022": sr14, "se_lo": se14, "n_months_2009_2022": len(r14), "sr_2023": sr23, "n_months_2023": len(r23),
         "se_2023_lo": se23_lo, "se_2023_naive": se23_naive, "posterior_lo_se": post(se23_lo), "posterior_naive_se": post(se23_naive),
         "pooled_sr_2009_2023": srall, "pooled_net_ann": float(rall.mean() * 12), "pooled_se_lo": math.sqrt((1 + srall ** 2 / 2) / (len(rall) / 12))}
log(f"PRIOR decile net SR 2009-22 {sr14:+.2f} (SE {se14:.2f}, {len(r14)} m) | 2023 {sr23:+.2f} ({len(r23)} m; SE Lo {se23_lo:.2f}, naive {se23_naive:.2f}) | "
    f"posterior {post(se23_lo)[0]:+.2f} (Lo SE) / {post(se23_naive)[0]:+.2f} (naive SE) | pooled 2009-23 {srall:+.2f} (+{rall.mean()*12*100:.1f}%/yr, SE {prior['pooled_se_lo']:.2f})")

# ---- fixed-cohort IC trend (composition control) -----------------------------------------------------------------------
t0 = int(frame.tmi.min()); cohort = set(frame.loc[frame.tmi == t0, "firm"].unique())
fc = frame[frame.firm.isin(cohort)]; fe = frame[~frame.firm.isin(cohort)]
ic_c = fc.groupby("tmi").apply(rank_ic).astype(float); ic_e = fe.groupby("tmi").apply(rank_ic).astype(float)
def trend(s):
    s = s.dropna(); x = (s.index.to_numpy() - 2009 * 12) / 12.0; X = np.column_stack([np.ones(len(s)), x]); b, t, _, n = ols_nw(s.to_numpy(), X, 12)
    return {"slope_per_year": float(b[1]), "nw12_t": float(t[1]), "n": n}
def by_year(s): return {int(y): float(s[(s.index // 12) == y].mean()) for y in YEARS}
def names_by_year(d): return {int(y): int(len(d[(d.tmi // 12) == y]) / max(1, d[(d.tmi // 12) == y].tmi.nunique())) for y in YEARS}
cohort_out = {"cohort_first_tmi": t0, "cohort_n_firms": len(cohort), "cohort_ic_by_year": by_year(ic_c), "entrants_ic_by_year": by_year(ic_e),
              "cohort_names_per_month": names_by_year(fc), "entrants_names_per_month": names_by_year(fe),
              "cohort_trend": trend(ic_c), "entrants_trend": trend(ic_e[ic_e.index >= 2010 * 12]), "all_trend": trend(icm),
              "eras_cohort": [float(np.mean([by_year(ic_c)[y] for y in r])) for r in (range(2009, 2014), range(2014, 2019), range(2019, 2024))],
              "eras_all": [float(np.mean([by_year(icm)[y] for y in r])) for r in (range(2009, 2014), range(2014, 2019), range(2019, 2024))]}
try: cohort_out["liquid_half"] = json.load(open(RESULTS / "p9_b_liq_trend.json"))
except Exception: pass
log(f"COHORT firms present at first 2009 formation: {len(cohort):,}; cohort IC by year {dict((y, round(v, 3)) for y, v in cohort_out['cohort_ic_by_year'].items())}")
log(f"COHORT trend slope {cohort_out['cohort_trend']['slope_per_year']:+.4f}/yr NW12 t {cohort_out['cohort_trend']['nw12_t']:+.2f} | eras cohort {[round(v, 3) for v in cohort_out['eras_cohort']]} vs all {[round(v, 3) for v in cohort_out['eras_all']]} | all-names trend t {cohort_out['all_trend']['nw12_t']:+.2f} | entrants trend t {cohort_out['entrants_trend']['nw12_t']:+.2f}")
log(f"COHORT names/month cohort {cohort_out['cohort_names_per_month']} | entrants {cohort_out['entrants_names_per_month']}")

out = {"design": "C1 (28 features, price-filtered), M2 fixed booster, WFA panel ml_panel_wfa2024, refits 2009-2023 (fit cache)",
       "attribution": attrib, "pass_probability": passp, "prior_update": prior, "composition": cohort_out,
       "ic_by_year_all": by_year(icm), "ic_monthly_2009_2023": {str(int(k)): float(v) for k, v in icm.items()},
       "decile_net_monthly": {str(int(k)): float(v) for k, v in rd.items()}}
(RESULTS / "p9_b_followups.json").write_text(json.dumps(out, indent=1, default=float)); log("wrote p9_b_followups.json")
