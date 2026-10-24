# P13 — graph neural networks on the free US equity panel (pre-registration v0, 2026-10-18 03:55)

Decision (2026-10-18 ~03:10): train graph models end to end on the panel, under this plan written before any result is read.
Goal: **a model with net Sharpe > 1.5 and MaxDD < 25 % using graph methods.**

Status of the question before P13: P12 (349 pre-registered cells on the same panels) topped out at net Sharpe 0.63 ± 0.32 on
development and 0.28 ± 0.51 on the check window for the selected cell; graph *features* (study 15, P12) added no IC; the gap is
information, not cost (≈ 2.5× the IC of the least demanding cell is needed). **What P12 did not try: a graph model trained
end-to-end (message passing over PIT firm graphs).** That is P13. Honest prior: small or no ΔIC over the non-graph baseline.
A pass is relayed only after an independent review; the goal is never fabricated.

## 1. Data (all TRAIN; sealed never read; no new zone)
- Panels `data_cache/equities_xs/derived/eqml_grid_2023/panel_{W,M,Q}.parquet` (rebuilt 2026-10-17 for P12c; formations through
  2022-12; 28 rank-normalised PIT features `mom_12_1 … sue` with `miss_*` flags; `raw_*` columns; targets `y_raw`, `y_rank`,
  `y_ex_all`, `y_ex_liq`; keys `firm`, `cik`, `date`, `target_mi`, `in_liq`). Weekly 1,664,463 rows; monthly 397,417; quarterly 132,961.
- Graph layers (edges with validity intervals, `data_cache/links/edges/`): E1 SIC4 membership (industry co-membership, fallback
  SIC2), E4 customer → supplier (PIT resolver, precision 0.96), E5 BEA use table (industry-level, mapped to firms via `e5_sic4_bea_map`).
- New PIT layer: return-correlation kNN (k = 10) from daily returns with session date ≤ formation date, 252-session window,
  residual to the market; built from `aux_daily.npz` if it carries daily returns, otherwise from the yfinance daily parquet through
  `safe_load_parquet(zone_config=EQUITIES_ZONES_PATH)` only (never a raw `pd.read_parquet`).
- Optional later: feature-space kNN; GDELT co-mention (2016+ only; no stored edge table — skipped in phase 1).

## 2. Windows and the rule (identical to P12c so results are comparable)
- Yearly refits 2009 … 2022, expanding window, first fit on 2004–2008; per-horizon embargo (no training snapshot whose target period
  ends after refit date − horizon); OOS predictions for every formation 2009-01 … 2022-12.
- **Development = 2009–2018** (selection), **check = 2019–2022** (read for the selected cell; all cells stored). The check window has
  been read before (P9 E6 once, P12c for 349 cells); its reuse is disclosed and counted; no fresh window exists without the
  2024-H1 unlock token. A P13 pass therefore means "dev ≥ 1.5 and check ≥ 1.5", not a sealed result.
- Books: the P12c grid unchanged (`experiments/p12c_eval.py`): aims R1_weekly, R2_quarterly_slow, R4_blend_WMQ, R4b_blend_WM;
  universes ALL (15 bps/side), LIQ (5), TOP500 (2); partial adjustment θ ∈ {0.05, 0.1, 0.2, 0.5, 1.0}; w_vsn4 industry-matched,
  vol-scaled, beta-neutral weights; 55 book cells per prediction set. Selection: highest development net Sharpe at the declared cost
  with MaxDD ≥ −25 %. **Pass: selected cell net Sharpe ≥ 1.5 on development AND on check, MaxDD ≥ −25 % on both.**
- Every Sharpe is reported with annualised return and MaxDD beside it; precision (iid SE, 95 % CI) as in `p12c_precision.py`;
  trial count N = P12 349 + P13 cells × 55, with the expected maximum of N null trials beside the best.

## 3. Cells (phase 1; hyperparameters fixed before any return is read — the P9 lesson: selection on noisy validation IC collapses it)
Common: node features = 28 rank features + 28 miss flags (+ 10 SIC one-hots); target `y_rank` (centred cross-sectional rank);
loss MSE; hidden 64; 2 message-passing layers; dropout 0.1; Adam lr 1e-3, weight decay 1e-5; **15 epochs fixed, no early stopping,
no validation-based selection**; one snapshot (formation date) per step; seeds {0, 1, 2}, predictions averaged across seeds.
- **C0 MLP** — no graph, same features / loop / epochs: the baseline every ΔIC is measured against.
- **C1 R-SAGE(E1, E4, E5)** — GraphSAGE-mean with relation-specific linear maps (sum over relations), self + neighbour.
- **C2 R-SAGE(E1, E4, E5, CORR)** — C1 plus the correlation kNN layer.
- **C3 R-GAT(E1, E4, E5, CORR)** — attention over neighbours, 4 heads, same relations as C2.
- **C4 C2 on rewired graphs** — degree-preserving edge shuffle within each date and relation: the graph-contribution null.
- **C5 (phase 2) temporal C2** — GRU over the last 12 feature snapshots per firm before message passing.
Each cell × horizon {W, M, Q} × seed is one job; a refit is one trained model (phase 1 ≈ 5 cells × 3 horizons × 3 seeds × 14 refits
≈ 630 model fits). Horizon W is the one P12 found most IC in (weekly IC t 5.6); M and Q feed the blend aims.

## 4. Diagnostics and nulls (scripted; run by the runner, read by the refuter)
- Rank IC per formation; development / check means with month-t and cluster-robust t; **paired ΔIC vs C0** per formation.
- C4 rewiring null for the graph contribution; label-shuffle sanity (IC ≈ 0); feature-lag ladder (features lagged +1 period must not
  raise IC; a deliberate −1 look-ahead row must — the P9 E1 battery, scripted).
- PIT assertions in code and tests: every edge used at formation d satisfies valid_from ≤ d < valid_to; correlation edges use
  returns with session ≤ d only; no target or future column enters the feature matrix; neighbours' targets are never aggregated.

## 5. Workflow (persists across sessions)
- Code `experiments/gnn/`: `common.py` (paths, formats, firm/date index — the interface), `build_graphs.py`,
  `gnn.py`, `train.py`, `books.py`, `status.py`; tests `tests/test_p13_*.py`.
- Data `data_cache/studies/p13_gnn/` (on the data volume): `graphs/`, `preds/`, `results/`, `queue/`, `progress.log`.
- Reports `results/gnn/`: `STATE.json`, `RESULTS.md`.
- Persistence: a file queue of jobs (JSON per job; pending → running → done/failed by atomic rename) executed by a runner process
  started on a timer (lock file; mounts the data volume); `status.py` summarises the queue.
- Gates: no P13 book is evaluated before the design review's blocking items are closed (the training jobs may start — they do
  not read books); any pass → independent red team with reproductions before it is relayed.

## 6. Stop rule
P13 ends when (a) a cell passes the rule and survives the red team (relayed as a pass with its reuse caveat), or (b) phase 1 + phase 2
are exhausted with every cell's dev and check net Sharpe and ΔIC reported, in which case the finding is "graph models trained
end-to-end add ΔIC of x ± y over the same features without a graph" and the gap to 1.5 is restated in information units.

## v1 amendments (2026-10-18 04:1x) — all ten blocking items of the design review adopted before any P13 IC or book is read
- **A1 embargo pinned** (`common.test_mask` / `train_mask`): test rows for refit Y are target_mi ∈ [12Y, 12Y+11]; training rows
  target_mi ≤ 12Y−2 AND exit_date ≤ the first test formation date (known at the decision; drops the Q Sep-30 label). Both dates per
  refit go into the prediction meta. P13 trains through Nov of Y−1; P12c trained through Oct of Y−2 — results are comparable, not identical.
- **A2 inputs pinned**: the model input matrix is exactly the panel builder's `input_columns` (28 features + 11 SIC one-hots + 28 miss
  flags = 67; `gp` was dropped by the builder's coverage rule), identical across W / M / Q, read only through `common.node_features`;
  a forbidden-name regex rejects every target / raw / key column.
- **A3 predictions keyed**: `firm`, `date_id` and the panel's sha256 travel with every prediction; `load_pred` asserts them and that the
  row set is exactly P12c's (W 1,347,743 / M 318,475 / Q 103,945) so every cell trades the same universe; isolated nodes are still predicted.
- **A4 books as a library**: `books.py` is the extraction of `p12c_eval.py` with the prediction source as the only parameter; the regression
  test must reproduce every number of `p12c_eval.json` (55 cells × dev/chk × gross + 4 cost levels) to 1e-9; the cross-cell selection lives
  in a committed `selection.py`.
- **A5 control C0g**: the same MLP / loop / epochs / seeds on [X, one-hop mean_r(X) per relation, log1p(deg_r)]. C2 − C0g isolates learned
  message passing; C0g − C0 is the graph-feature effect. Every paired ΔIC is reported on two bases: raw `y_ex_all`, and with prediction and
  target demeaned within the book's SIC4 / SIC2 / division groups (what `w_vsn4` can monetise).
- **A6 null redesigned**: C4 trains on edge tables rewired ONCE per seed, persistent across dates, relation-specific and stratified: E1 SIC4
  labels permuted among firms within SIC2; E4 `rewire_stratified` by customer SIC4 on table rows (validity intervals kept); E5
  `permute_within_columns` per data year; CORR neighbours redrawn within SIC2 × beta tercile, fixed per (firm, year). Three draws = three seeds;
  swap share reported (≥ 50 %).
- **A7 selection**: only C1, C2, C3, C5 are eligible; C0, C0g, C4 are evaluated and counted, never selected. A pass additionally needs the
  selected cell's development paired ΔIC (demeaned basis) > 0 with month-level t ≥ 2 against C0 AND against C4; otherwise the outcome is
  "book reached X, not attributable to the graph". MaxDD filter on development only. N = 349 + 55 × every P13 prediction set evaluated
  (controls and phase 2 included); beside the best: BLdP deflated Sharpe and the EXACT expected maximum of N null Sharpes. Hyperparameters
  are frozen by hash (`common.hp_hash` = `872821b168bcbcc3`, `registry.json`); any change after a P13 number is read is a new registered cell.
  Books use `*_mean.npz` only (no seed selection). C5 runs whatever phase 1 shows.
- **A8 pass conditions declared now**: a pass is relayed only if it also (i) survives retraining the selected cell and C0 without the five
  split-distorted price-level features (`log_price`, `log_mcap`, `bm`, `ep`, `sp` + miss flags) on both windows; (ii) holds with drift-charged
  turnover (weights drifted by the realised period return before |w − prev|); (iii) is reported with HAC / 13-week block-bootstrap SE beside iid.
- **A9 survivorship guards**: no per-node embeddings or firm-count-shaped parameters; no degree / neighbour-count / "has relation" inputs in
  C1–C5 (mean aggregation with explicit zero for isolated nodes; GAT softmax-normalised; no sum aggregation); degree enters C0g only; P9's
  firm-wise 12-month circular label-shift test runs on C2 and C3.
- **A10 gate scope**: no IC, ΔIC or book of any P13 prediction set is computed before A1–A9 are implemented and `selection.py` is committed;
  label-shuffle and the deliberate look-ahead row (integrity checks) are exempt. Smoke runs before this point are void and unread.

## v1.1 notes (2026-10-18 04:5x, after the second closure check)
- C4 null, as built: E1 SIC4 labels permuted among firms with the same SIC2 history (fixed over time, so per-date group sizes are
  preserved only approximately); E4 `rewire_stratified` by customer SIC4 — swap share 0.53–0.55 at table level (study 15's ≥ 50 %
  rule), 0.30–0.39 measured on snapshot edges (strata are small; E4 is ≈ 64 of ≈ 63,900 edges per monthly snapshot); E5 0.95; CORR 0.92
  with every firm refilled to its real neighbour count from coarser strata (SIC2 × beta tercile → SIC2 → division → all present).
- C2 − C4 is reported as the seed-mean ΔIC and as the spread across the three draws (each draw also paired seed-by-seed: C2 s_i vs C4 s_i).
- Label-shift diagnostic (A9): each firm's TRAINING labels are shifted circularly by one calendar year inside each refit's training
  window (calendar months, not rows); no test-year label can enter training.
- Reproducibility statement (2026-10-18 05:5x): C0 / C0g predictions are bit-identical across reruns; graph cells (C1–C5) at the
  trainer's 4 CPU threads reproduce to ≈ 1e-7 (thread races in torch's index_add message passing), bit-identical with 1 thread.
  Seeds are therefore "statistical" seeds for the graph cells; nothing in the rule depends on bit identity.

## v1.2 — as built (2026-10-18 06:1x, after the fourth closure check; supersedes the v1.1 wording where they differ)
- E1 null: SIC4 labels are permuted only among firms with the IDENTICAL SIC2 timeline (ordered runs of SIC2 with their dates), fixed
  over time; zero SIC2 crossings on every firm-date (asserted against the real table); ≈ 12 % of firms keep their label (unique timeline
  or fixed point) — conservative for C2 − C4. CORR null: each firm's rewired in-degree equals its real in-degree at every snapshot.
- Label-shift diagnostic (A9), as built: for each refit, every TRAINING row takes the label of the same firm's training row K formations
  earlier (K = 54 / 13 / 5 for W / M / Q, longer than the 252-session lookback; fallback one formation before, then after); never the row
  itself; only sources whose target window ended ≥ 366 days before the row's formation (so the source return never overlaps the row's
  feature lookback); rows without a valid source are dropped (share in meta; ≈ 17-21 % on M / Q); test rows untouched. Nothing is wrapped.
- How the label-shift result is read (declared before any read): for C2 and C3, seed 0, each horizon, the shifted-label model's rank IC
  on development and check is reported beside the real model's. If the shifted model's development IC is > 0 with month-level t ≥ 2 AND
  at least half of the real model's development IC, the cell's signal persists under a one-year label shift: identity / survivorship or
  persistent-characteristic channel. A pass by rule from such a cell is NOT relayed as a graph result; it is relabelled "not attributable to
  current information" and the red team is told. Otherwise the diagnostic is reported and does not change the verdict.
