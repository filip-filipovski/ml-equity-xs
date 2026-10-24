#!/usr/bin/env python
"""P12c precision — standard errors of the stored P12c Sharpes (arithmetic on p12c_eval.json only; no return is read).

Per cell and window (development 2009-18, check 2019-22) and for the two windows pooled (2009-22, exact for the concatenated
weekly series: mean and variance recovered from the stored n_weeks / ann / sharpe and combined by the total-variance identity):
  SR_w = SR_ann / sqrt(52);  SE(SR_ann) = sqrt((1 + SR_w^2 / 2) / n_weeks) * sqrt(52)   (Lo 2002, iid weeks)
  95 % interval = SR +- 1.96 SE;  z_to_1.5 = (1.5 - SR) / SE;  t_to_0 = SR / SE.
Caveats printed with the table: iid SE (no autocorrelation / fat-tail allowance); the development figure of the selected cell is
the maximum of 55 (selection-biased upward); the check figure is one pre-declared read; the pooled figure inherits the selection.
Writes p12c_precision.json."""
from __future__ import annotations
import json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"; RESULTS.mkdir(exist_ok=True)
PER = 52; TARGET = 1.5; Z95 = 1.96
DECL = {"ALL": 15, "LIQ": 5, "TOP500": 2}
ev = json.loads((RESULTS / "p12c_eval.json").read_text())
def uni_of(cid: str) -> str: return cid.split("_")[-2]
def moments(sr_ann: float, ann: float, n: int) -> tuple[float, float]:
    """weekly mean and sd (ddof=1) recovered from the stored annualised Sharpe and annual mean."""
    m = ann / PER; sr_w = sr_ann / math.sqrt(PER)
    return m, (m / sr_w if abs(sr_w) > 1e-12 else float("nan"))
def prec(sr_ann: float, n: int) -> dict:
    sr_w = sr_ann / math.sqrt(PER); se = math.sqrt((1 + sr_w * sr_w / 2) / n) * math.sqrt(PER)
    return {"sharpe": sr_ann, "n_weeks": n, "se_iid": se, "ci95": [sr_ann - Z95 * se, sr_ann + Z95 * se],
            "z_to_target": (TARGET - sr_ann) / se, "t_to_zero": sr_ann / se}
def pooled(parts: list[tuple[float, float, int]]) -> tuple[float, float, int]:
    """exact mean / sd (ddof=1) of the concatenation of series with known (mean, sd ddof=1, n)."""
    N = sum(n for _, _, n in parts); m = sum(mu * n for mu, _, n in parts) / N
    ss = sum((n - 1) * sd * sd + n * (mu - m) ** 2 for mu, sd, n in parts)
    return m, math.sqrt(ss / (N - 1)), N
out = {"design": __doc__, "cells": {}, "summary": {}}
for cid, r in ev["cells"].items():
    k = f"net_{DECL[uni_of(cid)]}bps"; rec = {"declared_bps": DECL[uni_of(cid)]}
    parts_net, parts_gross = [], []
    for w in ("dev", "chk"):
        s = r[w]; n = s["n_weeks"]
        rec[w] = {"net_declared": {**prec(s[k]["sharpe"], n), "ann": s[k]["ann"], "maxdd": s[k]["maxdd"]},
                  "gross": {**prec(s["gross_sharpe"], n), "ann": s["gross_ann"]}}
        parts_net.append((*moments(s[k]["sharpe"], s[k]["ann"], n), n)); parts_gross.append((*moments(s["gross_sharpe"], s["gross_ann"], n), n))
    for nm, parts in (("net_declared", parts_net), ("gross", parts_gross)):
        m, sd, N = pooled(parts); sr = m / sd * math.sqrt(PER)
        rec.setdefault("pooled", {})[nm] = {**prec(sr, N), "ann": m * PER}
    out["cells"][cid] = rec
sel = ev["selected"]["cell"]; S = out["cells"][sel]
best_dev = sorted(out["cells"], key=lambda c: -out["cells"][c]["dev"]["net_declared"]["sharpe"])[:6]
hdr = f"{'cell':<34}{'win':>7}{'netSR':>7}{'%/yr':>6}{'SE':>6}{'95% lo':>8}{'95% hi':>8}{'z→1.5':>7}{'t→0':>6} | {'gross':>6}{'SE':>6}{'z→1.5':>7}"
lines = ["P12c precision (iid SE from the stored summaries; no return read)", hdr]
def row(cid, w, d):
    a, g = d["net_declared"], d["gross"]
    return f"{cid:<34}{w:>7}{a['sharpe']:>7.2f}{a['ann']*100:>6.1f}{a['se_iid']:>6.2f}{a['ci95'][0]:>8.2f}{a['ci95'][1]:>8.2f}{a['z_to_target']:>7.1f}{a['t_to_zero']:>6.1f} | {g['sharpe']:>6.2f}{g['se_iid']:>6.2f}{g['z_to_target']:>7.1f}"
for cid in [sel] + [c for c in best_dev if c != sel]:
    for w in ("dev", "chk", "pooled"): lines.append(row(cid + (" (selected)" if cid == sel and w == "dev" else ""), w, out["cells"][cid][w]))
    lines.append("")
# summary over all 55 cells
up15 = [c for c in out["cells"] if out["cells"][c]["pooled"]["net_declared"]["ci95"][1] >= TARGET]
up10 = [c for c in out["cells"] if out["cells"][c]["pooled"]["net_declared"]["ci95"][1] >= 1.0]
gross_up15 = [c for c in out["cells"] if out["cells"][c]["pooled"]["gross"]["ci95"][1] >= TARGET]
maxp = max(out["cells"], key=lambda c: out["cells"][c]["pooled"]["net_declared"]["sharpe"]); mp = out["cells"][maxp]["pooled"]["net_declared"]
minz = min(out["cells"], key=lambda c: out["cells"][c]["pooled"]["net_declared"]["z_to_target"]); mz = out["cells"][minz]["pooled"]["net_declared"]
out["summary"] = {"selected": sel,
    "selected_dev_net": S["dev"]["net_declared"], "selected_chk_net": S["chk"]["net_declared"], "selected_pooled_net": S["pooled"]["net_declared"],
    "selected_dev_gross": S["dev"]["gross"], "selected_chk_gross": S["chk"]["gross"], "selected_pooled_gross": S["pooled"]["gross"],
    "n_cells": len(out["cells"]), "cells_pooled_net_upper95_ge_1.5": up15, "cells_pooled_net_upper95_ge_1.0": up10,
    "cells_pooled_gross_upper95_ge_1.5": gross_up15,
    "max_pooled_net": {"cell": maxp, **mp}, "min_z_to_target_pooled_net": {"cell": minz, **mz},
    "caveats": ["iid weekly SE (Lo 2002); no autocorrelation or fat-tail allowance",
                "the development figure of the selected cell is the maximum over 55 cells (selection-biased upward); the check figure is one pre-declared read; pooled inherits the selection",
                "the P12 block-bootstrap interval on a 10-year development book had half-width 0.55 (stored in P12_RESULT.md), about the iid value for that length"]}
lines += [f"selected {sel}: dev net {S['dev']['net_declared']['sharpe']:.2f} ± {S['dev']['net_declared']['se_iid']:.2f} (+{S['dev']['net_declared']['ann']*100:.1f} %/yr), "
          f"chk {S['chk']['net_declared']['sharpe']:.2f} ± {S['chk']['net_declared']['se_iid']:.2f} (+{S['chk']['net_declared']['ann']*100:.1f} %/yr), "
          f"pooled 2009-22 {S['pooled']['net_declared']['sharpe']:.2f} ± {S['pooled']['net_declared']['se_iid']:.2f} (+{S['pooled']['net_declared']['ann']*100:.1f} %/yr; 95 % [{S['pooled']['net_declared']['ci95'][0]:.2f}, {S['pooled']['net_declared']['ci95'][1]:.2f}])",
          f"1.5 is {S['dev']['net_declared']['z_to_target']:.1f} / {S['chk']['net_declared']['z_to_target']:.1f} / {S['pooled']['net_declared']['z_to_target']:.1f} SE above the dev / chk / pooled net; gross pooled {S['pooled']['gross']['sharpe']:.2f} ± {S['pooled']['gross']['se_iid']:.2f}, z {S['pooled']['gross']['z_to_target']:.1f}",
          f"cells (of {len(out['cells'])}) whose pooled net 95 % upper limit reaches 1.5: {len(up15)}; reaches 1.0: {len(up10)} {up10}; pooled GROSS upper limit reaches 1.5: {len(gross_up15)} {gross_up15}",
          f"highest pooled net: {maxp} {mp['sharpe']:.2f} ± {mp['se_iid']:.2f} (+{mp['ann']*100:.1f} %/yr), z to 1.5 {mp['z_to_target']:.1f}; smallest z to 1.5 (pooled net): {minz} z {mz['z_to_target']:.1f} (SR {mz['sharpe']:.2f} ± {mz['se_iid']:.2f})"]
txt = "\n".join(lines); print(txt)
(RESULTS / "p12c_precision.json").write_text(json.dumps(out, indent=1)); (RESULTS / "p12c_precision.log").write_text(txt + "\n"); print("wrote p12c_precision.json / .log")
