# P9 E6 pre-declaration — fresh-window replication of the fixed booster (2019–2022)

Written 2026-10-17, branch `loop/ml-meeting`, **before any post-2018 IC is computed or read**. Addendum to
`P9_PREDECLARATION.md`. Mandate: search for an edge; this is the highest-prior legitimate candidate in the repo,
tested the only honest way — on data the model has never been fit, selected or evaluated on.

## Why this candidate

P9 E1 found the pre-fixed booster M2 (depth 3, lr 0.05, 300 rounds, mcw 100, subsample/colsample 0.8, **no
per-refit selection**) has OOS rank IC **+0.0314** on 2009–2018 (cluster t 5.4, positive in all 10 refit-years), and that
study 13's selector — not the model class — destroyed it. Every P9 read used the study-13 panel, which ends 2018-12.
Under the 2026-10-11 equities zones, TRAIN runs to **2023-01-01**: the four years **2019-01 → 2022-12 are TRAIN data
that no ML study (11–14, P9) has ever fit, tuned, selected or scored on.** They are also the *least* survivor-flattered
part of the panel (only 2023→2026 delistings are missing). This is a replication on fresh data, not a hold-out: it is
TRAIN by zone. It is pre-registered so that it cannot be tuned.

## Data and discipline

* Panel rebuilt by the unmodified study-13 builder (`scripts/validate/equity_ml_xs_train.py::build_panel`) with its
  module constants overridden: `TRAIN_END = 2023-01-01`, outputs to **new** derived files
  `ml_panel_train2023.{parquet,meta.json}` (the study-13 cache is never overwritten). Every loader is zone-gated
  (`safe_load_parquet(zone_config=EQUITIES_ZONES_PATH)`); nothing at or after 2024-07-01 is loadable; nothing at or
  after 2023-01-01 is loaded. `sealed_load_count 0` printed. No WFA, no sealed, no new data, no purchases.
* **Feature set pinned** to study 13 / P9: the same 28 features; the coverage-rule drop list is forced to the
  study-13 result (`gp` dropped) regardless of what the extended coverage says, and any difference is reported.
* Refits: `p9_ml_capacity.make_refits` unchanged — expanding window from `FIRST_TARGET_MI`, embargo 1, validation =
  the 12 target months before the test year, fit/val gap 1. Test years **2019, 2020, 2021, 2022** (2022 has 11 target
  months, 2022-01..2022-11, because the 2022-12 holding window needs a January 2023 price). **The 2019 refit trains on
  target months ≤ 2018-11 only — i.e. exactly P9's data — so its predictions are P9's protocol rolled one year
  forward with zero exposure to the new years.** Asserted in code.
* Seeds: `SEED + 12·year` as in P9. Torch not involved (booster + ridge only).

## Integrity gates (must all pass before any 2019+ IC is read)

1. **Overlap identity:** on formation dates 2004-01 … 2018-12, every (date, symbol) row of the new panel must match the
   study-13 panel on all normalised features, `miss_*`, `sic_*`, `y_raw`, `y_rank`, `y_ex_all`, `y_ex_liq`, `in_liq`,
   `target_mi`. Tolerance: ≥ 99.9 % of rows bit-identical on every column; any residual must be attributable to the
   CIK↔symbol map (which uses whole-matrix bar counts) and is listed. Failure → stop, investigate, no result.
2. Panel `max(ts) < 2023-01-01` and `≥ 2022-10-01`; `sealed_load_count 0`.
3. Plumbing check on the **old** window: M2 on test year 2018 from the new panel must land within the P9 E1 2018
   month-IC noise (not bit-identical — firm column ids shift, so XGBoost's 0.8 subsample draws differ).
4. `assert_nondegenerate` on every refit's validation and test predictions.

## Cells (fixed; nothing else is run on the fresh window)

* **C1 — primary:** M2 fixed booster, ALL universe (LIQ secondary), expanding refits 2019–2022.
* **C0 — control:** M1 ridge-all, alpha chosen on validation tie-safe IC exactly as P9 E1 (the P9-protocol ridge).

## Metrics

Tie-safe rank IC per target month; `cstats` (cluster-by-refit-year percentile bootstrap, 2 000 draws, **4 clusters =
3 dof, wide**) plus the plain per-month t (47 months, flagged iid-optimistic), monthly ICIR, OOS R². Decile long/short
EW book, gross and net at **15 bps/side ALL, 5 bps/side LIQ**, with **annualised return beside every Sharpe**,
turnover/yr, break-even bps/side, per-year table. Paired ΔIC (M2 − M1) with the P9 margin δ = +0.010.

## Pre-declared reads

* **Replicates** iff M2 fresh-window mean IC > 0 **and** per-month t ≥ 2.0 **and** cluster-bootstrap 95 % lower bound
  > 0. If only the per-month t clears: "replicates at iid power only" (stated as such). Otherwise **does not replicate**.
* **Degradation:** fresh IC vs the P9 2009–2018 M2 IC (+0.0314), two-sample with each window's cluster SE; reported
  as a z and a difference CI, no threshold — descriptive.
* **Economic read (secondary):** net decile L/S annualised return > 0 at 15 bps ALL. Noted but never promoted on its
  own; 47 months cannot clear an N2-style bootstrap at this book's volatility (study-13 MDE ≈ 10–13 %/yr).
* **Capacity read (tertiary):** paired ΔIC vs ridge, P9 rule. Only to say whether the booster's advantage over ridge
  survives out of its window.

## What E6 will NOT claim

* Not a hold-out pass: TRAIN by zone. The honest sentence is "the pre-fixed booster's rank IC replicated / did not
  replicate on four years it never saw, inside TRAIN". Not sealed, not WFA.
* Not tradeable: survivor-only (less so here), ordinal-only, 47 months, no N1/N2 null battery in this read.
* No cell selection, no max over anything, no re-run with other settings. One read.

## Phase B — frozen now, NOT run in E6 (decision, irreversible)

If E6 **replicates**, the proposal is a **one-shot** on the equities WFA zone **2023-01-01 → 2024-07-01** with the
identical frozen C1 config: panel extended to 2024-07-01, expanding refits for test years 2023 (12 target months) and
2024 (target months 2024-01 … 2024-05, 5 months; the June window needs July prices), **~17 months**. Pre-declared
WFA read: IC > 0 and per-month t ≥ 2.0 **and** net L/S annualised return > 0 at 15 bps → "survived the WFA one-shot
(reduced power)". Anything else → WFA FAIL, zone spent. This is written here so Phase B cannot be shaped by E6's
numbers. **It runs only on an explicit go-ahead.**

## Compute and outputs

Build ≈ 2 min (study-13 build was 81 s), cells ≈ 2 min (M2 ≈ 12–25 s/refit in P9). Scripts `p9_e6_build_panel.py`
(build + overlap identity), `p9_e6_fresh_window.py` (cells + reads). Outputs `p9_e6_build.json`, `p9_e6.json`,
`P9_E6_FRESH_WINDOW.md`. Red-teamed after the run, before any user-facing claim.

---

## Addendum 1 (2026-10-17, after the build + integrity gate, BEFORE any post-2018 IC is read)

**Gate 1 failed as literally written, and the failure is fully explained.** On the 2004–2018 overlap the raw price,
event and fundamental features are bit-identical on matched (date, symbol) rows (mom/rev/vol/amihud/skew/hi/EA
features 1.0000; accruals/nsi/leverage/roe/sue ≥ 0.998), `y_raw` differs only on the two last-2018 formation dates
whose holding window the old build could not complete (expected), and `cik` is unchanged. What differs:

* **Universe membership**, ~6.7 firms per date (1,271 old-only rows = 0.46 %, 852 new-only = 0.31 %): **100 % of the
  old-only rows are symbols with a 2019–2022 reverse split (77/77), 100 % of the new-only rows symbols with a
  2019–2022 forward split (35/35).** Cause (`equity_ml.build_price_matrices`): `raw_close = close × Π(splits strictly
  after t *within the loaded panel*)`; the ALL filter is `raw_close ≥ $5`. The reconstructed "traded price" therefore
  depends on splits that occur after T up to the panel's end — future information, bounded by the window.
* **Price-level features** `log_price, log_mcap, bm, ep, sp` differ for 7.7 % of rows — all 271 affected symbols have a
  2019–2022 split; the old/new ratios equal the cumulative split (AMZN 1/20, MUX ×10, AAPL 1/4, TSLA 1/15).
* Every rank-normalised column, `y_rank`, `y_ex_all`, `in_liq` and the EW-ALL-market-dependent `beta_252`/`ivol_63`
  then differ slightly on every date, because the within-date universe differs by a few names.

**Finding F1 (red team of the study-13/14/P9 machinery, new):** the universe price filter and the price-level
features are not point-in-time: they use split events after T inside the loaded window. Study 13/14/P9 (window to
2018-12) had the same channel from 2004–2018 splits. Direction: forward-splitters (future winners) tend to be *added*
at low prices, reverse-splitters (future losers) *dropped* — a potential flattering of a long/short rank IC. Magnitude on
this panel ≈ 0.4 % of rows plus the price-level features. To be quantified; recorded for `LESSONS_LEARNED`.

**Amended integrity criterion (replaces gate 1):** (i) raw features identical on matched rows except the two
EW-market-dependent and five price-level features, whose differences must be 100 % split-attributable; (ii) membership
differences ≤ 1 % of rows and 100 % split-attributable; (iii) `y_raw` differences only on the end-of-window dates;
(iv) `cik` unchanged. **All four hold → the extended panel is a faithful run of the unmodified builder; gate 1 PASS
under the amended criterion.** Gates 2–4 unchanged.

**Added robustness cell (pre-declared now, same refits/reads/rules):**
* **C1b — PIT-safe variant of C1:** panel rebuilt with the universe price filter removed (`build_universe(...,
  min_price=0)` via a declared one-line override; the $1 M 63-day median dollar-volume filter — split-invariant — is
  kept) to `ml_panel_train2023_nopx.*`, and the five price-level features dropped from the design (23 features +
  their `miss_*` + SIC). Same M2 config, seeds, refits, metrics. **C0b** = M1 ridge on the same design (paired ΔIC).
* **Interpretation rule:** C1 replicates **and** C1b replicates → "replicates, robust to the split channel". C1
  replicates but C1b does not → "replication not clean — carried in part by the non-PIT split channel"; the headline
  is then C1b's, not C1's. C1 does not replicate → C1b is reported but cannot rescue (no cell selection).
* The 2018 plumbing check (gate 3) stays on the C1 panel; expected close to P9 E1, not identical (membership).

---

## Addendum B1 — Phase B, the equities WFA one-shot (written 2026-10-17 ~15:15, BEFORE any WFA-zone IC is computed)

Decision on 2026-10-17: Phase B is run now, exactly as frozen above and
with the red-team-corrected reads; nothing in the pass rule is changed.

* **Data:** panels rebuilt with the same builder to the WFA end **2024-07-01** → `ml_panel_wfa2024.*` (price-filtered,
  C1 design) and `ml_panel_wfa2024_nopx.*` (no price filter, C1b design). Two loop bounds in the builder
  (`load_prices`: `range(WARMUP_YEAR, TRAIN_END.year)`; `load_fundamentals`: `range(2008, TRAIN_END.year)`) stop at the
  year *before* the end date and would silently drop the partial year 2024, so they are extended by one year in a
  declared override; every loader call still receives `end = 2024-07-01` and is zone-gated (`EQUITIES_ZONES_PATH`,
  sealed from 2024-07-01). Panel `max ts` asserted `< 2024-07-01`; nothing sealed is readable. Feature drop list pinned
  to study 13 as before. Known consequence of the truncation defect (F1, corrected form): pre-2023 prices in these
  panels are re-reconstructed with 2023-24 splits, so the 2023/2024 refits train on slightly different pre-2023 rows than
  E6's (membership differences expected to be 100 % split-attributable; reported, not re-diagnosed).
* **Refits:** `make_refits([2023, 2024], 1)` — expanding, embargo 1, 12-month validation, fit/val gap 1. The 2023 refit
  trains on target months ≤ 2022-11 (TRAIN only); the 2024 refit on ≤ 2023-11 (TRAIN + 2023 WFA labels — a
  walk-forward refit, not tuning; M2 is fixed, ridge alpha is chosen on validation as in P9). Target months 2023-01 …
  2023-12 and 2024-01 … 2024-05 (the June window needs the July close): **17 months**.
* **Cells (frozen):** C1 = M2 fixed booster, 28 features, price-filtered panel; C1b = M2, 23 features, no-filter panel;
  C0 / C0b ridge controls. Seeds `SEED + 12·year` as before. No other model, window, feature set or cost is run.
* **Frozen pass rule (unchanged from the Phase B section):** per cell, **IC > 0 and per-month t ≥ 2.0 and net decile
  L/S annualised return > 0 at 15 bps/side (ALL)** → "survived the WFA one-shot (reduced power)"; anything else →
  "WFA FAIL; zone spent". Headline = C1's verdict with C1b's beside it. Red-team-corrected descriptive reads are added,
  not substituted: month-level sign-flip p (one-sided, 200k flips), circular block-bootstrap CI (L = 3), Newey–West t;
  no cluster statistics (2 clusters are meaningless). Book reported with its block-bootstrap interval, the cost ladder
  (5 / 15 / 25 / 50 bps) and "no borrow cost".
* **Co-declared secondary economic read (new, declared here before the run):** a **3-month-cohort book** from the same
  monthly predictions — each formation month opens an EW top-decile long / bottom-decile short cohort held for three
  target months (three overlapping cohorts, weights 1/3 each; a name missing in a later month earns 0 for that month);
  turnover = Σ|Δw| per month; cost 15 bps/side on traded notional. Study 14 found quarterly books survive costs where
  monthly ones do not; this reads that on the same WFA months. Reported for 2019-22 (E6 window) and for the WFA months,
  C1 and C1b. It is **not** part of the pass rule and cannot rescue a FAIL.
* **Labelled diagnostics after the read (no headline from them):** the C1b predictions evaluated on the ≥ $5 rows (the
  red team's "if one number" book); per-year table; LIQ IC and book.
* **What Phase B will not claim:** not sealed; survivor-only universe (listed as of 2026-10); 17 months (power is low:
  with month-IC SD ≈ 0.07 the per-month t ≥ 2 leg needs mean IC ≳ 0.034); the economic leg is undetermined at this power
  whatever its sign. A pass is "survived one further hold-out", never "an edge".
* **Outputs:** `p9_b_wfa_oneshot.py`, `p9_b_wfa.json`, `P9_B_WFA_ONESHOT.md`, figure `figures/p9_b_wfa.png`. The
  equities WFA zone is spent by this read for this family (frozen config, one look).

### B2 note (2026-10-17 ~15:25, before any WFA read)

New goal set the same day: a model with net Sharpe above 1.5 and maximum drawdown under 25 % using graph methods. Decision: **Phase B is deferred, not cancelled.** The equities WFA zone is the one remaining hold-out (sealed
stays reserved), so the single WFA read will co-declare **C1 (baseline, exactly as frozen above)** and the
goal-directed **graph-augmented candidate** developed on TRAIN (2004–2022) under a new pre-declaration (P10). The
Phase B kit (`p9_e6_build_panel.py --wfa`, `p9_b_wfa_oneshot.py`) is committed ready-to-run; nothing in B1's rule
changes for C1. No WFA-zone IC has been computed.

### B3 note (2026-10-17 ~15:50, before any WFA read) — the reachable WFA zone is 2023 only

The equities loaders list, without a token, only partitions that **end** at or before the sealed start
(`equities_xs._zoned_files`: "nothing sealed is ever opened"). The `year=2024` price and event partitions straddle
2024-07-01 and are therefore refused whole; 2024-01 … 2024-06 is unreachable for this read without an unlock token,
which is a human-only step that is not taken. Phase B is therefore read on **test year 2023 only: target months
2023-01 … 2023-11 (11 months; the 2023-12 holding window needs the January-2024 close).** The 2024 refit is dropped;
nothing else in B1 changes (cells, seeds, cost, the frozen pass rule applied to the 11 months, the co-declared cohort
book, the labelled diagnostics). Panels are the B1 builds as written to disk (formation dates to 2023-12; the stem
`ml_panel_wfa2024` is kept, `max ts` recorded). **Power caveat, stated before the read:** per-month t ≥ 2 on 11 months
needs mean IC ≳ 0.042 (month-IC SD ≈ 0.07); a FAIL on the frozen rule is therefore reported together with the IC, its
block-bootstrap CI and the sign-flip p, so that "FAIL on power" is distinguishable from "IC not positive". The 2023
window is spent for this family by this read either way.
