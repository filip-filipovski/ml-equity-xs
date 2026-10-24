# P12 — pre-declaration: Sharpe > 1.5, MaxDD < 25 % through ML + graph methods (owner directive 2026-10-17 ~20:00)

Owner, verbatim: "set a /goal of 1.5 sharpe whtough ml graphs one, /loop untill you are done. dont ask me anything, no stops,
full autonomy, no human in the loop." Written before any P12 return is read; committed at HEAD before `p12_stage0.py` runs.
Two things fixed: no sealed-zone read; no reported pass that is not real (a pass is relayed only after an independent red
team). The meeting kit (freeze 18:00 on the 18th) is not changed by P12 except by a red-teamed pass.

## Zones, data, costs
* Equities zones (2026-10-11): TRAIN formation months 2009-01 … 2022-12 only: the panel loader is bounded by the sealed start (the file holds
  the 2023 WFA rows), so every P12 script restricts target months to ≤ 2022-12 itself and asserts max ts < 2023-01-01 on
  the rows it uses; `sealed_load_count` asserted 0. No WFA (2023 spent; 2024-H1 token-gated), no sealed. **Development window 2009–2018,
  check window 2019–2022** (the check window has been read twice before as a check: E6 and P10; P12 is its third use —
  its evidential value is correspondingly reduced and is stated as such).
* Panel `ml_panel_wfa2024` (TRAIN rows), SIC4 membership `links/edges/e1_sic_membership`, BEA use edges `e5_bea_use_edges`
  + `e5_sic4_bea_map`, E4 customer→supplier edges; the cached C1/M2 predictions (`p9_cache/p9_b_c1_frame_2009_2023`,
  rows > 2022-12 dropped) for Stage 0. Costs: 15 bps/side ALL, 5 bps/side LIQ (study-13 convention); borrow ladder as a
  sensitivity; survivor-only prices (stated on every result). No purchase, no sign-up.

## Stages (cheap first; every cell logged in `p12.json`; annualised return beside every Sharpe)
0. **Transfer and bound, cached predictions (no refit).** Books on the existing M2 signal: rank (P10 B1 reproduction),
   vol-scaled rank (w ∝ centred rank / σ_i, σ = trailing 63-day vol, floored), vol-scaled + beta/SIC-division neutral,
   prediction smoothing k ∈ {1, 3, 6} months, hysteresis bands (enter top/bottom 20 %, exit at 35 %), book vol target 10 %.
   Also the ICIR-implied bound and the realised Pearson transfer. Purpose: the ceiling of the *existing* signal.
1. **Target engineering (new refits, M2 fixed booster + ridge M1, expanding refits 2009–2022, same `make_refits`):** T0
   y_rank; T1 rank of beta-residual (y_raw − β_252·m_t); T2 rank of SIC-division residual; T3 rank of SIC4 residual
   (graph residual, membership valid at the formation date); T4 rank of beta + SIC4 residual; T5 rank of y_raw / σ_i.
   Read: monthly rank IC against y_rank (comparable) and against the own target; Stage-0 constructions on each.
2. **Graph-ML arm:** G1 prediction propagation over SIC4 peers and BEA hop-1 neighbours, p′ = (1−α)p + α·mean_peers(p)
   and p − α·mean_peers(p), α ∈ {0.25, 0.5}; G2 M2 refit with SIC4-peer means of the 8 strongest characteristics as
   extra columns (P10 used only LOO industry momentum). T3/T4 are the graph-residual targets.
3. **Weekly horizon (conditional):** only if the monthly ceiling after stages 0–2 is below 1.5 gross; study-14 W panel,
   fixed booster, cohorts 4–8 weeks and bands. Declared now so that running it is not a post-hoc choice.
4. **Construction grid and selection:** signals from 1–2 × constructions {vol-scaled rank, + neutral, bands, smoothing} ×
   universe {ALL 15 bps, LIQ 5 bps} × vol target 10 %. Selection = highest net Sharpe on development subject to
   MaxDD ≤ 25 %; the check window is read once for the selected cell and reported for all. Trial count ≈ 6 targets × 2
   models × 8 constructions × 2 universes + 10 graph cells ≈ 200; DSR over that count with an effective-N estimate.

## Success rule (literal, pre-declared)
PASS only if the selected cell has **net Sharpe ≥ 1.5 and MaxDD ≤ 25 % on BOTH the development window and the check
window** at the declared cost (ALL 15 bps or LIQ 5 bps), with the check-window block-bootstrap 90 % lower bound of net
Sharpe > 0.5, and survives an independent red team. Outcome classes: (i) PASS; (ii) gross-feasible, net-fail (gross
≥ 1.5 somewhere, net < 1.5): report the cost gap; (iii) EXHAUSTION: best zero-cost, ideal-transfer development Sharpe
< 1.5 after stages 0–3 — the target is unreachable with these methods on this data; report that, not a pass.

## Honest prior (stated before running)
P10: the same signal's rank book is gross SR 0.37 on development (0.70 on the check window); the ICIR-implied ceiling is
≈ 1.2 (2009–22) / ≈ 1.8 (2019–22) gross at ideal transfer and zero cost. Residual targets and vol-scaling typically lift
realised transfer by tens of percent, not by the 4× needed; a net 1.5 on development is therefore unlikely. The program
is run anyway, in full, because the owner asked for it and because exhaustion has to be shown, not asserted.

## Addendum 1 (2026-10-17 ~21:05, written before the Stage 1b read)
Stage 0 (`p12_stage0.json`): the existing M2 signal's ceiling on development is gross Sharpe 0.66 (vol-scaled rank, ALL;
net 0.12 at 15 bps; drawdown −9 %), ICIR-implied bound 0.95. Stage 1 (`p12_stage1.json`): no target raises the IC
against the raw return (T3 SIC4-residual ICIR 1.09 vs 0.95), but the models predict their **own residual targets** far
better (own-target development ICIR: T1 beta-residual 1.72, T3 SIC4-residual 1.33, T4 beta + SIC4-residual 2.43). A
book realises the residual return only if its weights are neutral at the same resolution as the target. The declared
"neutral" construction was beta + SIC-division; **Stage 1b adds the matched construction `vsn4`: vol-scaled rank weights
demeaned within SIC4 (fallback SIC2, then division), then beta-neutralised, with and without the 20/35 % hysteresis
band, k ∈ {1, 3}, ALL 15 bps and LIQ 5 bps**, applied to the cached Stage 1 predictions of all six targets × two models
(no new fit). This is the construction the T4 own-target ICIR of 2.43 would need; the ceiling it reaches on development
is read as the Stage 1b result. Selection rule, success rule and the check-window discipline are unchanged.

## Addendum 2 (2026-10-17 ~21:55, written before the Stage 3b read)
Stage 3 realised gross 1.18 against a linear bound of 1.28 at weekly frequency, with turnover 1.8/week; holding cohorts
and smoothing cut gross faster than cost. The one turnover-control not yet run is the standard transaction-cost-aware
rule: **partial adjustment toward the aim portfolio, w_t = w_{t−1} + θ (w*_t − w_{t−1}), θ ∈ {0.05, 0.1, 0.2, 0.35, 0.5,
1.0}**, with w*_t the Stage 3 `vsn` aim (vol-scaled rank, beta/SIC-division neutral) and, separately, the `vs` aim;
ALL 15 bps and LIQ 5 bps; cost ladder 2 / 5 / 15 / 25 bps reported. Development window only (as Stage 3). Read as the
Stage 3b result; no change to the success rule.

## Addendum 3 (2026-10-17 ~21:00, written before the P12b read) — the red team's routes, same rule
`REDTEAM_P12.md`: "not met" stands; "exhausted" stands only for the grid run; the ICIR "ceiling" argument is withdrawn
(benchmarks, not bounds). Declared now, in the red team's rank order, all on the cached development-window predictions
(weekly / quarterly panels end 2018-12; no check window at those horizons without new fits): **R1** weekly SIC4-matched
aim (`vsn4` on the W panel, SIC4 membership at the formation date) with partial adjustment θ ∈ {0.05, 0.1, 0.2, 0.5, 1};
**R2** slow-horizon aim from study 14's cached quarterly M2 predictions carried forward to weekly dates, traded with the
same θ ladder; **R4** multi-horizon blend: mean of the per-date centred ranks of the weekly, monthly (carried forward)
and quarterly (carried forward) aims, `vsn4`, θ ladder; **R5** top-liquid subset (top 500 names by trailing dollar
volume at each date) at 1 and 2 bps with the best R1/R4 aim; **R3** (risk-model optimiser) is approximated by `vsn4`
+ partial adjustment and is not separately run; **R6** (learned / lead-lag / news graphs) is not run (prior ≤ +0.05).
Universes ALL 15 bps, LIQ 5 bps, TOP500 1–2 bps; cost ladder 1 / 2 / 5 / 15 bps reported for every cell. (Note added
after the run: the script also ran R4b, the weekly + monthly blend without the quarterly aim; it was not listed here,
is reported as an added variant, and is the rule-selected cell in P12c.) Success rule
unchanged (net ≥ 1.5 and MaxDD ≤ 25 % on both windows after a red team) — any development-only pass is a trigger for
check-window fits, not a pass.

## Addendum 4 (2026-10-17 21:07 commit clock, written before any P12c read) — P12c: corrected-build panels through 2022 and the check window
P12b's best cells are development-only (study 14's W / M / Q panels end 2018-12) and on the older panel build. P12c:
(1) rebuild the weekly / monthly / quarterly panels with study 14's own builder, patched only in its bounds (TRAIN_END
2023-01-01, OOS_LAST_YEAR 2022) and its output directory (`eqml_grid_2023/`, nothing in `eqml_grid/` is overwritten);
the zone loader bounds every input at the equities TRAIN end; (2) refit the fixed booster M2 (expanding window, annual
refits 2009–2022, embargo 1 month, P9 machinery) on each horizon; (3) rerun the P12b routes on the new panels for the
development window 2009–2018 **and**, read once for the cell the pre-declared rule selects on development (highest net at
the declared cost, MaxDD ≤ 25 %), the check window 2019–2022; every cell's check-window numbers are stored in the JSON
but only the selected cell's is read as a result. The old-vs-new development comparison is reported for the blend.
Success rule unchanged (net ≥ 1.5, MaxDD ≤ 25 %, both windows, red team). Expected: the development net of ≈ 0.8 already
fails the rule; P12c is run to put the best book found on the same footing as the monthly results and to learn whether
its development Sharpe is a property or a window.
