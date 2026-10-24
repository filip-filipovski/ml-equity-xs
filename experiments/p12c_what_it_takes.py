#!/usr/bin/env python
"""P12c follow-up (arithmetic only; no new return read) — what net Sharpe 1.5 would take on the corrected-build books.

Reads the stored summaries in p12c_eval.json (gross annual return, gross Sharpe, mean weekly turnover, and the net ladder
at 1 / 2 / 5 / 15 bps per side) and, per cell and window, computes
  * annual vol            sigma = gross_ann / gross_sharpe
  * cost drag per year    drag(c) = turnover_per_week * 52 * c
  * net Sharpe at cost c  (gross_ann - drag(c)) / sigma         [checked against the stored ladder]
  * break-even cost       c_be  = gross_ann / (turnover * 52)
  * cost for net 1.5      c_15  = (gross_ann - 1.5 sigma) / (turnover * 52)   (negative = unreachable at any cost)
  * gross Sharpe needed for net 1.5 at the declared cost, and the first-order IC multiple = needed / realised
    (first order: the gross return scales with the IC at fixed construction, vol and turnover).
Writes p12c_what_it_takes.json and prints a table."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PER = 52; DECL = {"ALL": 15, "LIQ": 5, "TOP500": 2}; LADDER = (1, 2, 5, 15); TARGET = 1.5
ev = json.loads((HERE / "p12c_eval.json").read_text()); cells = ev["cells"]; selected = ev["selected"]["cell"]
def arith(s: dict, decl_bps: int) -> dict:
    gs, ga, to = s["gross_sharpe"], s["gross_ann"], s["turnover_per_week"]
    if abs(gs) < 1e-9 or to <= 0: return {"skip": True}
    sigma = ga / gs; turns_yr = to * PER
    approx = {b: (ga - turns_yr * b * 1e-4) / sigma for b in LADDER}
    err = max(abs(approx[b] - s[f"net_{b}bps"]["sharpe"]) for b in LADDER)
    drag_decl = turns_yr * decl_bps * 1e-4
    need_gross = TARGET + drag_decl / sigma
    return {"skip": False, "sigma_ann": sigma, "turns_per_year": turns_yr, "ladder_max_abs_err": err,
            "breakeven_bps": ga / turns_yr * 1e4, "cost_for_net15_bps": (ga - TARGET * sigma) / turns_yr * 1e4,
            "declared_bps": decl_bps, "drag_declared_ann": drag_decl, "net_declared_stored": s[f"net_{decl_bps}bps"]["sharpe"],
            "gross_needed_for_net15_at_declared": need_gross, "ic_multiple_needed": need_gross / gs if gs > 0 else float("nan"),
            "gross_sharpe": gs, "gross_ann": ga}
out = {"design": __doc__, "cells": {}}
for cid, r in cells.items():
    uni = cid.split("_")[-2]; out["cells"][cid] = {w: arith(r[w], DECL[uni]) for w in ("dev", "chk")}
maxerr = max(v[w]["ladder_max_abs_err"] for v in out["cells"].values() for w in ("dev", "chk") if not v[w]["skip"])
out["ladder_check_max_abs_err"] = maxerr
def row(cid, w):
    a = out["cells"][cid][w]
    if a["skip"]: return f"{cid:<34}{w:>4}  (skipped: zero gross or turnover)"
    c15 = a["cost_for_net15_bps"]; c15s = f"{c15:6.2f}" if c15 > 0 else "  <0 "
    return (f"{cid:<34}{w:>4}{a['gross_sharpe']:>7.2f}{a['gross_ann']*100:>7.1f}{a['sigma_ann']*100:>6.1f}{a['turns_per_year']:>7.1f}"
            f"{a['breakeven_bps']:>8.1f}{c15s:>8}{a['declared_bps']:>5}{a['net_declared_stored']:>7.2f}{a['gross_needed_for_net15_at_declared']:>8.2f}{a['ic_multiple_needed']:>7.1f}")
hdr = f"{'cell':<34}{'win':>4}{'gSR':>7}{'g%/yr':>7}{'vol%':>6}{'turn/y':>7}{'be bps':>8}{'c@1.5':>8}{'decl':>5}{'netSR':>7}{'gNeed':>8}{'ICx':>7}"
lines = [f"ladder check: max |approx - stored| over all cells and ladder points = {maxerr:.3f} Sharpe", "", hdr]
lines += ["-- selected cell (pre-declared rule)"] + [row(selected, w) for w in ("dev", "chk")]
dec = lambda cid: DECL[cid.split("_")[-2]]
top = sorted(cells, key=lambda c: -cells[c]["dev"][f"net_{dec(c)}bps"]["sharpe"])[:6]
lines += ["-- six best development cells by net at the declared cost"] + [row(c, w) for c in top for w in ("dev", "chk")]
for w in ("dev", "chk"):
    bg = max(cells, key=lambda c: cells[c][w]["gross_sharpe"]); lines += [f"-- highest gross Sharpe on {w}"] + [row(bg, w)]
# least demanding cell: smallest IC multiple on each window, and the smallest of the max over both windows
ok = [c for c in cells if not out["cells"][c]["dev"]["skip"] and not out["cells"][c]["chk"]["skip"]]
least = {w: min(ok, key=lambda c: out["cells"][c][w]["ic_multiple_needed"]) for w in ("dev", "chk")}
both = min(ok, key=lambda c: max(out["cells"][c]["dev"]["ic_multiple_needed"], out["cells"][c]["chk"]["ic_multiple_needed"]))
for w in ("dev", "chk"): lines += [f"-- smallest IC multiple on {w}"] + [row(least[w], w)]
lines += ["-- smallest of the larger-of-both-windows IC multiple"] + [row(both, w) for w in ("dev", "chk")]
# summary numbers
summ = {"selected": selected,
        "selected_dev": out["cells"][selected]["dev"], "selected_chk": out["cells"][selected]["chk"],
        "least_demanding_dev": least["dev"], "least_demanding_chk": least["chk"], "least_demanding_both": both,
        "ic_multiple_least_dev": out["cells"][least["dev"]]["dev"]["ic_multiple_needed"],
        "ic_multiple_least_chk": out["cells"][least["chk"]]["chk"]["ic_multiple_needed"],
        "ic_multiple_least_both": max(out["cells"][both]["dev"]["ic_multiple_needed"], out["cells"][both]["chk"]["ic_multiple_needed"]),
        "cells_with_positive_cost_for_net15": {w: sorted([c for c in ok if out["cells"][c][w]["cost_for_net15_bps"] > 0]) for w in ("dev", "chk")},
        "max_breakeven_bps": {w: max((out["cells"][c][w]["breakeven_bps"], c) for c in ok) for w in ("dev", "chk")}}
out["summary"] = summ
lines += ["", f"cells where some positive cost gives net 1.5: dev {len(summ['cells_with_positive_cost_for_net15']['dev'])}, chk {len(summ['cells_with_positive_cost_for_net15']['chk'])} (of {len(ok)})",
          f"largest break-even cost: dev {summ['max_breakeven_bps']['dev'][0]:.1f} bps ({summ['max_breakeven_bps']['dev'][1]}), chk {summ['max_breakeven_bps']['chk'][0]:.1f} bps ({summ['max_breakeven_bps']['chk'][1]})",
          f"smallest IC multiple for net 1.5 at the declared cost: dev {summ['ic_multiple_least_dev']:.1f}x ({least['dev']}), chk {summ['ic_multiple_least_chk']:.1f}x ({least['chk']}), both windows {summ['ic_multiple_least_both']:.1f}x ({both})"]
txt = "\n".join(lines); print(txt); (HERE / "p12c_what_it_takes.json").write_text(json.dumps(out, indent=1, default=float)); (HERE / "p12c_what_it_takes.log").write_text(txt + "\n")
