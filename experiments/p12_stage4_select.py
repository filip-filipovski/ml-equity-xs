#!/usr/bin/env python
"""P12 Stage 4 — pre-declared selection over the recorded cells (no new return read): highest development net Sharpe
subject to MaxDD <= 25 %, with the check-window read of the selected cell (and of all cells, as recorded in the JSONs);
cells without a check window (weekly stages 3, 3b) listed separately; exact expected maxima of N zero-Sharpe trials over
10 years by Monte Carlo (red-team finding 9); weekly Pearson / rank benchmarks copied from p12_stage3.json. Writes p12_stage4.json."""
import json, math, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"; RESULTS.mkdir(exist_ok=True)
s0, s1, s1b, s2, s3, s3b = (json.load(open(RESULTS / f"p12_stage{k}.json")) for k in ("0", "1", "1b", "2", "3", "3b"))
cells = []
for cid, m in s0["cells"].items(): cells.append(("S0 M2(cached)", cid, m))
for cid, m in s0["cells_voltarget10"].items(): cells.append(("S0 M2(cached)", cid, m))
for cid, c in s1["cells"].items():
    for bid, m in c["books"].items(): cells.append((f"S1 {cid}", bid, m))
for cid, bb in s1b["cells"].items():
    for bid, m in bb.items(): cells.append((f"S1b {cid}", bid, m))
for grp in ("G1", "G2"):
    for cid, c in s2[grp].items():
        for bid, m in c["books"].items(): cells.append((f"S2 {cid}", bid, m))
weekly = [("S3", cid, m) for cid, m in s3["cells"].items()] + [("S3b", cid, m) for cid, m in s3b["cells"].items()]
elig = [x for x in cells if x[2]["dev"]["maxdd"] >= -0.25]; sel = sorted(elig, key=lambda x: -x[2]["dev"]["net_sharpe"])
top = [{"signal": s, "book": b, "dev_net_sharpe": m["dev"]["net_sharpe"], "dev_net_ann": m["dev"]["net_ann"], "dev_gross_sharpe": m["dev"]["gross_sharpe"], "dev_maxdd": m["dev"]["maxdd"],
        "dev_net_sharpe_ci95": m["dev"].get("net_sharpe_ci95"), "chk_net_sharpe": m["chk"]["net_sharpe"], "chk_net_ann": m["chk"]["net_ann"], "chk_gross_sharpe": m["chk"]["gross_sharpe"], "chk_maxdd": m["chk"]["maxdd"]} for s, b, m in sel[:8]]
both = [(s, b, m["dev"]["net_sharpe"], m["chk"]["net_sharpe"]) for s, b, m in cells if m["dev"]["net_sharpe"] >= 0.5 and m["chk"]["net_sharpe"] >= 0.5]
w_best_decl = max(((s, c, m) for s, c, m in weekly), key=lambda x: (x[2].get("net_sharpe") if "net_sharpe" in x[2] else x[2][f"net_{x[2]['declared_cost_bps']}bps"]["sharpe"]))
rng = np.random.default_rng(1); se = 1 / math.sqrt(10); emax = {N: float((rng.standard_normal((40000, N)) * se).max(1).mean()) for N in (30, 100, len(cells), len(cells) + len(weekly))}
out = {"design": __doc__, "n_cells_with_check": len(cells), "n_weekly_cells": len(weekly), "n_total": len(cells) + len(weekly), "selection_top8": top,
       "cells_with_net_ge_0_5_on_both_windows": both, "expected_max_of_N_noise_sharpes_10y_exact_mc": emax, "asymptotic_sqrt2lnN_over_sqrt10": {N: float(se * math.sqrt(2 * math.log(N))) for N in emax},
       "weekly_benchmarks": s3.get("bound_detail"), "monthly_benchmarks": s0["bounds"]}
(RESULTS / "p12_stage4.json").write_text(json.dumps(out, indent=1, default=float))
print("cells", len(cells), "+", len(weekly), "weekly; selected:", top[0]["signal"], top[0]["book"], round(top[0]["dev_net_sharpe"], 3), "chk", round(top[0]["chk_net_sharpe"], 3), "CI", top[0]["dev_net_sharpe_ci95"]); print("both >= 0.5:", both); print("E[max] exact:", {k: round(v, 2) for k, v in emax.items()})
