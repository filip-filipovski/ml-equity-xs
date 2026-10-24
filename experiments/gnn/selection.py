#!/usr/bin/env python
"""P13 cross-cell selection and goal rule (design review A7). Named selection.py, not select.py: a script called select.py in this directory
would shadow the stdlib `select` module for every script run from it (subprocess -> selectors -> select).

Reads RESULTS/*_books.json (books.py output; `quick` smoke runs and the key-less P12c regression are ignored).
  * only cells with common.CELLS[cell]["eligible"] (C1, C2, C3, C5) and the seed-mean sets (pred_set == "mean") can be selected; C0 / C0g / C4 and
    single-seed sets are evaluated and counted in N, never selected;
  * selected = highest development net Sharpe at the declared cost of the book cell's universe with development MaxDD >= -25 % (dev only);
  * PASS label needs: dev net >= 1.5 AND chk net >= 1.5, both MaxDD >= -25 %, AND for every horizon used by the selected book's aim the development
    paired dIC (demeaned basis, diag.py) > 0 with month-level t >= 2 against C0 AND against C4;
  * otherwise NOT MET; when the book alone clears 1.5 the wording is "book reached X, not attributable to the graph";
  * N = 349 + 55 x (prediction sets evaluated); beside the best: BLdP deflated Sharpe (equity_ml_grid.deflated_sharpe over every book cell of every
    evaluated set, V = cross-trial Sharpe variance, N declared) and the EXACT expected maximum of N iid null Sharpes at the development SE
    (20,000 seeded draws).  A8 (i) (retrain without the split-distorted price-level features) is NOT checked here; it is a relay condition.
A10: refuses (rc 3) unless ROOT/GATE_OPEN exists. For vs C4 the dIC spread over the three C4 draws (diag.py dic_vs_seeds) is printed beside the condition;
the distinct code hashes of the evaluated prediction sets are listed (warning if a cell has more than one).
Writes REPORTS/SELECTION.json and SELECTION.md. Last printed line: `GOAL: NOT MET` or `GOAL: PASS BY RULE — RED TEAM REQUIRED BEFORE RELAY`."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

class GateClosed(RuntimeError):
    pass


def gate_open() -> bool:
    return (common.ROOT / "GATE_OPEN").exists()


def require_gate() -> None:
    if not gate_open(): raise GateClosed(f"A10 gate closed: selection.py refuses without {common.ROOT / 'GATE_OPEN'}")


DECL = {"ALL": "15", "LIQ": "5", "TOP500": "2"}
AIM_HORIZONS = {"R1_weekly": ("W",), "R2_quarterly_slow": ("Q",), "R4_blend_WMQ": ("W", "M", "Q"), "R4b_blend_WM": ("W", "M")}
N_P12 = 349; BOOK_CELLS = 55; MAXDD_FILTER = -0.25; BAR = 1.5
GOAL_NO = "GOAL: NOT MET"; GOAL_PASS = "GOAL: PASS BY RULE — RED TEAM REQUIRED BEFORE RELAY"


def load_sets(results: Path | None = None) -> list[dict]:
    out = []
    for p in sorted((results or common.RESULTS).glob("*_books.json")):
        try: r = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError): continue
        if r.get("quick") or r.get("pred_kind") == "p12c" or not r.get("cell") or r.get("cell") not in common.CELLS: continue
        r["_file"] = p.name; out.append(r)
    return out


def uni_of(cid: str) -> str: return cid.split("_")[-2]
def aim_of(cid: str) -> str: return cid.rsplit("_", 2)[0]
def net_of(r: dict, side: str, cid: str) -> dict: return r["cells"][cid][side][f"net_{DECL[uni_of(cid)]}bps"]


def expected_max_null(N: int, se: float, draws: int = 20000, seed: int = 0) -> dict:
    """Exact (simulated) expected maximum of N iid N(0, se^2) Sharpes: mean over `draws` repetitions of the max of N normals."""
    rng = np.random.default_rng(seed); mx = np.empty(draws); step = max(1, 4_000_000 // max(N, 1))
    for a in range(0, draws, step):
        b = min(draws, a + step); mx[a:b] = rng.normal(0.0, se, size=(b - a, N)).max(axis=1)
    return {"N": int(N), "se": float(se), "draws": draws, "expected_max": float(mx.mean()), "p95_max": float(np.quantile(mx, 0.95))}


def dsr_block(sets: list[dict], best_key: str, N: int) -> dict:
    from eqxs.equity_ml_grid import deflated_sharpe, PERIODS_PER_YEAR
    P = PERIODS_PER_YEAR["W"]; adj = math.sqrt(P / 52.0)    # books annualise with 52; deflated_sharpe divides by sqrt(50.4): rescale so the per-period Sharpe is identical
    trials = {}
    for r in sets:
        for cid in r["cells"]:
            d = net_of(r, "dev", cid)
            if not np.isfinite(d["sharpe"]): continue
            trials[f"{r['_file']}:{cid}"] = {"sr_ann": d["sharpe"] * adj, "horizon": "W", "n": r["cells"][cid]["dev"]["n_weeks"], "skew": d.get("skew", 0.0), "kurt": d.get("kurt", 3.0)}
    out = deflated_sharpe(trials, n_declared=N); b = out["per_cell"].get(best_key)
    return {"N": out["N"], "n_cells_ran": out["n_ran"], "cross_trial_sd_sharpe": out["sd_cross_trial"] / adj, "sr0_bldp_sharpe": out["sr0_ann"] / adj, "dsr_best": None if b is None else b["dsr"]}


def code_hash_report(sets: list[dict]) -> dict:
    by_cell: dict = {}
    for r in sets:
        for v in (r.get("code_hash") or {}).values():
            if v: by_cell.setdefault(r["cell"], set()).add(v)
    allh = sorted({v for vs in by_cell.values() for v in vs})
    return {"distinct": allh, "by_cell": {c: sorted(v) for c, v in by_cell.items()}, "warnings": [f"{c}: {len(v)} distinct code hashes {sorted(v)}" for c, v in by_cell.items() if len(v) > 1]}


def evaluate_selection(results: Path | None = None) -> dict:
    require_gate()
    results = results or common.RESULTS; sets = load_sets(results); N = N_P12 + BOOK_CELLS * len(sets)
    res = {"n_sets_evaluated": len(sets), "N": N, "sets": [r["_file"] for r in sets], "pass": False, "label": "NOT MET", "reason": [], "best": None, "code_hashes": code_hash_report(sets)}
    cands = []
    for r in sets:
        if r.get("pred_set") != "mean" or not common.CELLS[r["cell"]].get("eligible"): continue
        for cid in r["cells"]:
            dv = net_of(r, "dev", cid)
            if dv["maxdd"] >= MAXDD_FILTER and np.isfinite(dv["sharpe"]): cands.append((dv["sharpe"], r, cid))
    if not cands:
        res["reason"].append("no evaluated seed-mean book set of an eligible cell (C1, C2, C3, C5) with development MaxDD >= -25 %"); return res
    _, r, cid = max(cands, key=lambda t: t[0]); dv, ck = net_of(r, "dev", cid), net_of(r, "chk", cid)
    sel = r.get("selected") or {}; se = (sel.get("se") if sel.get("cell") == cid else None) or {}
    best = {"cell": r["cell"], "book_cell": cid, "set": r["_file"], "declared_cost_bps": DECL[uni_of(cid)], "hp_hash": r.get("hp_hash"), "git_head": r.get("git_head"),
            "dev": dv, "chk": ck, "dev_gross_sharpe": r["cells"][cid]["dev"]["gross_sharpe"], "chk_gross_sharpe": r["cells"][cid]["chk"]["gross_sharpe"], "se": se}
    if f"net_{DECL[uni_of(cid)]}bps_drift" in r["cells"][cid]["dev"]:
        k = f"net_{DECL[uni_of(cid)]}bps_drift"; best["dev_drift"], best["chk_drift"] = r["cells"][cid]["dev"][k], r["cells"][cid]["chk"][k]
    res["best"] = best
    seiid = (se.get("dev") or {}).get("se_iid"); res["null_max"] = expected_max_null(N, seiid) if seiid else None
    try: res["dsr"] = dsr_block(sets, f"{r['_file']}:{cid}", N)
    except Exception as e:  # noqa: BLE001
        res["dsr"] = {"error": f"{type(e).__name__}: {e}"}
    book_ok = dv["sharpe"] >= BAR and ck["sharpe"] >= BAR and dv["maxdd"] >= MAXDD_FILTER and ck["maxdd"] >= MAXDD_FILTER
    dic = {}; c4_spread = {}; c4_paired = {}; dic_ok = True; hs = AIM_HORIZONS[aim_of(cid)]
    for vs in ("C0", "C4"):
        for h in hs:
            p = results / f"dic_{r['cell']}_vs_{vs}_{h}.json"; key = f"{vs}_{h}"
            if not p.exists(): dic[key] = None; dic_ok = False; continue
            dj = json.loads(p.read_text()); d = dj["dev"]["demeaned"]["delta"]; dic[key] = d
            if vs == "C4":
                c4_spread[h] = ((dj.get("dic_vs_seeds") or {}).get("dev") or {}).get("demeaned"); c4_paired[h] = ((dj.get("dic_vs_seeds_paired") or {}).get("dev") or {}).get("demeaned")
            if not (d["mean"] is not None and d["mean"] > 0 and d["t"] is not None and d["t"] >= 2): dic_ok = False
    res["dic_dev_demeaned"] = dic; res["dic_c4_spread_dev_demeaned"] = c4_spread; res["dic_c4_paired_dev_demeaned"] = c4_paired; res["dic_ok"] = dic_ok; res["book_clears_bar"] = book_ok
    f = lambda d: f"net Sharpe {d['sharpe']:+.2f}, ann {d['ann']*100:+.1f} %, MaxDD {d['maxdd']*100:.1f} %"
    if book_ok and dic_ok: res["pass"] = True; res["label"] = "PASS BY RULE"
    elif book_ok: res["reason"].append(f"book reached dev {dv['sharpe']:+.2f} / chk {ck['sharpe']:+.2f}, not attributable to the graph (dIC vs C0 and C4 not > 0 with month-level t >= 2 on every horizon of the aim {aim_of(cid)}"
                                       + (": missing diag files" if any(v is None for v in dic.values()) else "") + ")")
    else: res["reason"].append(f"best eligible cell does not clear {BAR} on both windows with MaxDD >= -25 %: dev {f(dv)}; chk {f(ck)}")
    return res


def to_md(res: dict) -> str:
    L = ["# P13 SELECTION (auto-generated by selection.py)", "", f"Prediction sets evaluated: {res['n_sets_evaluated']}; trial count N = {N_P12} + {BOOK_CELLS} x {res['n_sets_evaluated']} = {res['N']}.", ""]
    b = res["best"]; fs = lambda v: "n/a" if not v or v.get("min") is None else "{:+.4f} / {:+.4f} / {}".format(v["min"], v["max"], "n/a" if v["sd"] is None else "{:.4f}".format(v["sd"]))
    fp = lambda v: "n/a" if not v or v.get("mean") is None else "{:+.4f} / {} / {}".format(v["mean"], "n/a" if v["sd"] is None else "{:.4f}".format(v["sd"]), [round(x, 4) for x in v["per_seed"]])
    fd = lambda v: "n/a" if v is None else "{:+.4f} (t {})".format(v["mean"], v["t"])
    if b is None: L += ["No selectable set. " + "; ".join(res["reason"])]
    else:
        f = lambda d: f"net Sharpe {d['sharpe']:+.2f}, ann {d['ann']*100:+.1f} %, MaxDD {d['maxdd']*100:.1f} %"
        L += [f"Best eligible: **{b['cell']}** / `{b['book_cell']}` (declared cost {b['declared_cost_bps']} bps/side; set `{b['set']}`, hp_hash `{b['hp_hash']}`, git `{b['git_head']}`)",
              f"- development: {f(b['dev'])}; gross {b['dev_gross_sharpe']:+.2f}", f"- check: {f(b['chk'])}; gross {b['chk_gross_sharpe']:+.2f}"]
        if b["se"]: L.append("- Sharpe SE: " + ", ".join(f"{s} iid {b['se'][s]['se_iid']:.2f} / block-13 {b['se'][s]['se_block13']:.2f}" for s in ("dev", "chk") if s in b["se"]))
        if "dev_drift" in b: L += [f"- drift-charged: dev {f(b['dev_drift'])}; chk {f(b['chk_drift'])}"]
        if res.get("null_max"): L.append(f"- exact expected maximum of {res['null_max']['N']} iid null Sharpes at the development SE {res['null_max']['se']:.2f}: {res['null_max']['expected_max']:.2f} (95th pct of the max {res['null_max']['p95_max']:.2f})")
        if res.get("dsr") and "error" not in res["dsr"]: L.append(f"- BLdP: cross-trial sd of Sharpe {res['dsr']['cross_trial_sd_sharpe']:.2f}, SR0 {res['dsr']['sr0_bldp_sharpe']:.2f}, deflated Sharpe of the best {res['dsr']['dsr_best']:.3f} (N = {res['dsr']['N']}, {res['dsr']['n_cells_ran']} cells ran)")
        L += ["- paired dev dIC (demeaned basis, month-level t): " + (", ".join(f"{k} {fd(v)}" for k, v in res["dic_dev_demeaned"].items()) or "n/a"),
              "- spread of the dev demeaned dIC over the three C4 draws (min / max / sd): " + (", ".join(f"{h} {fs(v)}" for h, v in res.get("dic_c4_spread_dev_demeaned", {}).items()) or "n/a"),
              "- seed-paired dev demeaned dIC vs C4 (C2 s_i vs C4 s_i; mean / sd / per-seed): " + (", ".join(f"{h} {fp(v)}" for h, v in res.get("dic_c4_paired_dev_demeaned", {}).items()) or "n/a"), "",
              f"Label: **{res['label']}**. " + " ".join(res["reason"]), "", "A8 (i) (retrain without log_price, log_mcap, bm, ep, sp + miss flags) is not checked by this script; it is a relay condition for any pass."]
    ch = res.get("code_hashes") or {}; L += ["", "Code hashes of the evaluated prediction sets: " + (", ".join(ch.get("distinct", [])) or "none recorded") + "".join(f"  WARNING {w}" for w in ch.get("warnings", []))]
    L += ["", res["goal"]]; return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--results", default=None); a = ap.parse_args(argv)
    try: res = evaluate_selection(Path(a.results) if a.results else None)
    except GateClosed as e: print(str(e), file=sys.stderr); return 3
    res["goal"] = GOAL_PASS if res["pass"] else GOAL_NO
    common.REPORTS.mkdir(parents=True, exist_ok=True); (common.REPORTS / "SELECTION.json").write_text(json.dumps(res, indent=1, default=float)); (common.REPORTS / "SELECTION.md").write_text(to_md(res))
    print(to_md(res).rstrip()); print(res["goal"]); return 0


if __name__ == "__main__":
    sys.exit(main())
