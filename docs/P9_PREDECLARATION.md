# P9 pre-declaration v2 — ML capacity, learning curves, and leakage controls

Written 2026-10-17, branch `loop/ml-meeting`. **v2 incorporates the design red-team** (`REDTEAM_P9.md`, CRITICAL C1–C4
and MAJOR M1–M6) and is committed before any P9 return is read. Purpose: turn the briefing's weakest ML claim ("we did
not rule out capacity; the networks were not properly tuned") into measured curves, and preempt the two objections a
senior ML researcher is most likely to raise about studies 13/14 — **leakage** and **capacity**. This is a methodology
study, not a strategy. No tradeable claim, no PASS/verdict; it produces diagnostics and controls with goalposts fixed
here.

## Data and discipline

* Panel: `data_cache/equities_xs/derived/ml_panel_train.parquet` only — TRAIN zone, 277,722 firm-months, 114 PIT
  features, date_id 0..179 (2004-01..2018-12). OOS evaluation window 2009–2018 (date_id 59..177), the study-13 window.
  Names per OOS month 1,007–2,153 (median 1,465). Survivor-only; every conclusion is conditional on that.
* Sealed never read; panel ends 2018-12 (sealed start 2024-07), so trivially clean. Build still loads through
  `experiments/_train_loader.py`, asserts `max(ts) < 2024-07-01`, and prints `sealed_load_count 0`. No WFA, no
  sealed, no new data, no purchases.
* Protocol (fixed): **annual refit**, 10 refits, OOS years 2009–2018. Study-13 split: validation = last 12 months of the
  training block, 1-month fit/validation gap, embargo 1 month between train end and the OOS entry. Reuse
  `eqxs.equity_ml` splitter/fit/predict (no new PIT path) and the **tie-safe IC** from
  `eqxs.equity_ml_grid` throughout (the unmodified `equity_ml.FastIC` uses ordinal ranks; shallow boosters
  tie heavily and ordinal ranks resolve ties by row order, inflating IC). Call `assert_nondegenerate` after every fit.
* E1 "pipeline" = ridge-all (M1) + a fixed booster (M2, depth 3, lr 0.05, 300 rounds, min_child_weight 100) + NN3 (M3),
  **not** the ensemble and **not** the transformer M5. IC primary on the ALL universe, secondary on LIQ (re-ranked).
  Portfolio rows (secondary to IC): decile L/S equal-weight monthly, LIQ at 5 bps/side and ALL at 15 bps/side, gross and
  net, annualised return **and** Sharpe side by side, turnover/yr and break-even. OOS R² uses `y_ex_all` and the
  validation-fitted calibration slope (models predict rank units).

## Metrics (fixed)

* **Rank IC**: per-OOS-month tie-safe Spearman between prediction and realised forward return.
* **t-stat and CIs cluster by OOS refit-year** (10 clusters): block bootstrap, block = 12 months aligned to the refit
  year, 2,000 draws. Report the plain per-month t alongside, flagged as having too small an SE (lag-1 IC autocorrelation
  is −0.11; the binding dependence is positive at the refit-year scale because all 12 months of a year share one model).
  `SD(IC)/√n_months` is correct only for iid months and is labelled as such.
* **ICIR** reported monthly (mean/SD, matching studies 13/14); any √12 version is labelled "annualised".
* **OOS R²**: 1 − SSE/SST of the calibrated raw-return prediction, pooled and per year.
* Benchmark SE for interpretation: SE of the OOS mean IC ≈ 0.011; the real monthly IC SD is ≈ 0.10–0.14 (style-payoff
  swings), ~4× the 1/√n = 0.026 sampling floor.

## Experiments and pre-declared conclusion rules

**E1 — Leakage / integrity controls (primary).**
* **E1a Pipeline-integrity (label shuffle).** Permute the training AND validation targets within each formation month;
  keep true OOS labels for scoring; refit the pipeline; 20 shuffles. By exchangeability E[OOS IC] = 0, so this detects
  only evaluation/selection bugs (test labels reaching fitting, normalisation or early stopping). *It is NOT evidence
  against feature look-ahead or survivorship and is reported as such.* Rule: integrity-clean iff the mean shuffled IC is
  within 2 SE of 0 and the cluster CI contains 0.
* **E1b Feature-lag ladder.** Replace features at formation T with features from T−k, k∈{−1,0,1,2,3}; refit and score per
  k; report IC(k) and paired ΔIC vs k=0. A legitimate signal decays smoothly in k; a look-ahead leak peaks at k<0 and
  drops at k=0.
* **E1c Known-leak + survivorship ablation.** (i) Refit with the split/survivor-tainted block removed (`log_price`,
  `log_mcap`, `bm`, `ep`, `sp`, `sic_*`); report ΔIC vs the full set. (ii) Firm-wise circular time-shift of the training
  labels by a random lag ≥ 12 months, score on true OOS labels; a positive IC there is firm-fingerprint memorisation (a
  survivorship channel the within-month shuffle cannot see).
* **E1d Graded leak injection (sensitivity).** Add to 3 feature copies a component λ·z(rank(TRAIN y)) calibrated so the
  single leaky feature carries injected rank-IC ∈ {0.005, 0.01, 0.02, 0.05, 0.10}; refit ridge-all and the fixed
  booster; 20 draws/level. Report the smallest injected leak detected at ≥80 % power / 5 % size (cluster SE, M1). The
  +1-month feature shift is kept only as a gross sanity row (it carries Spearman ≈ 0.93 to the target, so it proves only
  that a catastrophic leak is visible).
* **E1e Embargo ladder {0,1,3,6} months**, paired ΔIC vs embargo 1. Expected ~no change; stated as a weak probe.

**E2 — Learning curve (IC vs training data).** Models: ridge-all and the fixed booster (same config as E1). Windows: 2y,
3y, 5y, expanding, and a **refit-on-fit+val** arm (hyperparameters fixed, refit including the validation slice — study
13 never does this, so its most recent 12 months are never fitted). Plus a firm-subsample axis at fixed recency
{10,25,50,100 %}. *Conclusion rule*: paired ΔIC(expanding − shorter) on the same OOS months, cluster CI, reported against
the **actual fit-row count** not nominal years. "Data quantity is not binding" iff the upper 95 % bound of that ΔIC <
δ = +0.010; a lower bound > 0 says data quantity binds. Flagged confounds: fundamentals coverage rises from ~0 (pre-2010)
to ~60–70 % (2012), so window length changes feature availability, regime and sample size together — hence the
fixed-recency firm-subsample axis as the clean sample-size test.

**E3 — Capacity sweep (IC vs capacity).** Expanding window, embargo 1, per-refit causal selection.
* Ridge: λ ∈ {1e-4 … 1e3} (8 points).
* XGBoost **no-selection capacity curve**: fit once at the largest size per refit and read OOS IC at `iteration_range` ∈
  {25,50,100,200,400,800}; across a grid depth ∈ {2,3,4,6} × lr ∈ {0.03,0.1} × min_child_weight ∈ {1,20,100} (squared-
  error objective, never rank:pairwise).
* MLP: width ∈ {16,64,256} × depth ∈ {1,2,3} × dropout ∈ {0,0.3}; OOS IC at fixed epochs {5,10,20,40} (no early-stop
  selection); ≥5 seeds for the final cell per family, report seed SD.
* Selection: **per refit from that refit's own validation slice only** (causal). In addition report, labelled "oracle
  upper bound, selection-biased in its own favour", the best OOS cell per family with a Hansen SPA / max-t bootstrap
  p-value vs ridge over the whole grid.
* *Conclusion rule*: compare on the **paired** monthly ΔIC_t = IC_flex,t − IC_ridge,t (same OOS months/refits), 95 %
  cluster-by-refit-year bootstrap CI, equivalence margin **δ = +0.010 IC** (set before looking; ≈ the gross L/S study 14
  needs to break even). "Capacity does not help" iff the upper 95 % bound of ΔIC < δ. If the upper bound ≥ δ → the result
  is **inconclusive**, not "capacity absent". If the lower bound > 0 → the study-13 read is revised. Even the oracle
  ceiling is checked against δ with its SPA p-value; a positive capacity result on a survivor panel may be survivorship
  exploitation, not signal, and is flagged.

**E4 — IC detectability (MDE), the ML-native P2.** Plant on top of the **real** y_rank (keeps the real cross-sectional
factor structure): target = real y_rank + δ·rank(X·w), w a random unit direction independent of y, δ calibrated so the
average planted within-month IC equals ρ ∈ {0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08}; 100 draws/ρ; pushed through the
ridge-all fit. Also a **non-stationary** plant (signal present in 30 % of months, or sign-flipping every 3 years). Power
= fraction rejecting IC = 0 at 5 % on the cluster-by-refit-year t. Report MDE80 for both plant types (stationary is
labelled an upper bound) and the equivalence statement: "the OOS mean IC of 0.017 has a 95 % CI [x, y]; an IC ≥ z cannot
be ruled out." Do **not** place a detectable-IC number beside the realised IC without that CI.

## What P9 will NOT claim

* No sealed evaluation; no statement that any model is tradeable. A positive E3 result would be reported as "capacity
  helped within TRAIN, still no sealed test, still below the study-14 cost break-even", not as an edge.
* Not "leakage-clean": E1a is an integrity check only; the known split/survivor leak is addressed (or not) by E1b/E1c.
* Multiplicity: E1–E4 produce ~10² cells; no single cell is claimed, and no max over a grid is quoted without its SPA
  p-value.
* The transformer M5 is **not** re-tuned here (compute); P9 says so explicitly, so the "transformer not tuned" caveat is
  only half-retired. An optional small width-{32,64,128} fixed-epoch check on 2–3 refits may be added if time permits.

## Outputs

`results/p9_*.json` + figures `figures/p9_leakage.png`, `p9_learning_curve.png`,
`p9_capacity_sweep.png`, `p9_ic_mde.png`. Build script `experiments/p9_ml_capacity.py` (driven off
`equity_ml` + `equity_ml_grid`; torch in loky workers as in study 13). Compute ~8–12 h; drop-if-time order and minimal
credible subset per `REDTEAM_P9.md`.
