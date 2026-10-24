#!/usr/bin/env python
"""P13 compact status (< 40 lines) and results/gnn/STATE.json. `status.py --json` prints the JSON only.

Reads: QUEUE/*/ (job counts, running jobs), LOGS/*.log (last line, `MODEL .* trained` count), PREDS (seed / mean files),
RESULTS/{cell}_{h}_mean_ic.json (else mean of {cell}_{h}_s*_ic.json) with dev / check IC (flat keys dev_ic/chk_ic|check_ic, or nested
{"dev": {"ic": x}, "chk"|"check": {"ic": y}}), RESULTS/{cell}_books.json (books.py output), REPORTS/SELECTION.json (selection.py; the PASS line needs it). The final line is exactly one of the two goal strings."""
from __future__ import annotations
import argparse, json, re, sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

GOAL_NO = "GOAL: NOT MET"
GOAL_PASS = "GOAL: PASS BY RULE (dev ≥ 1.5 and chk ≥ 1.5, MaxDD ≥ −25 %) — RED TEAM REQUIRED BEFORE RELAY"
MODEL_RE = re.compile(r"MODEL .* trained")


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _ic(d: dict, side: str):
    keys = {"dev": ("dev_ic", "dev_mean_ic", "ic_dev"), "chk": ("chk_ic", "check_ic", "ic_chk", "ic_check")}[side]
    for k in keys:
        if _num(d.get(k)) is not None: return float(d[k])
    for k in ({"dev": ("dev", "development_2009_2018"), "chk": ("chk", "check", "check_2019_2022")}[side]):   # trainer writes the long keys
        v = d.get(k)
        if _num(v) is not None: return float(v)
        if isinstance(v, dict):
            for kk in ("ic", "mean_ic", "mean", "rank_ic"):
                if _num(v.get(kk)) is not None: return float(v[kk])
    return None


def read_ic(cell: str, h: str):
    def load(p):
        try: return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError): return None
    m = common.RESULTS / f"{cell}_{h}_mean_ic.json"
    if m.exists() and (d := load(m)): return {"src": "mean", "dev": _ic(d, "dev"), "chk": _ic(d, "chk")}
    per = [d for p in sorted(common.RESULTS.glob(f"{cell}_{h}_s*_ic.json")) if (d := load(p))]
    if not per: return None
    out = {"src": f"seeds({len(per)})"}
    for side in ("dev", "chk"):
        v = [x for x in (_ic(d, side) for d in per) if x is not None]; out[side] = sum(v) / len(v) if v else None
    return out


def tail(path: Path, n=110) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); f.seek(max(0, f.tell() - 4096)); t = f.read().decode("utf-8", "replace")
        lines = [l.strip() for l in t.splitlines() if l.strip()]; return lines[-1][:n] if lines else ""
    except OSError: return ""


def jobs(status: str) -> list[dict]:
    out = []
    for p in sorted(common.job_dirs()[status].glob("*.json")):
        try: out.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError): pass
    return out


def collect() -> dict:
    now = datetime.now(timezone.utc)
    st = {"time": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "queue": {s: len(jobs(s)) for s in ("pending", "running", "done", "failed")}, "running": [], "cells": {}, "books": {}}
    for j in jobs("running"):
        try: el = (now - datetime.strptime(j["started"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)).total_seconds() / 60
        except Exception: el = None
        st["running"].append({"id": j["id"], "elapsed_min": None if el is None else round(el, 1), "last": tail(Path(j.get("log", "")))})
    # trainer logs only ({cell}_{h}_s{seed}.log); the runner's job logs (train_*.log, diag_*.log, books_*.log) duplicate every line
    st["models_trained"] = sum(len(MODEL_RE.findall(p.read_text(errors="replace"))) for p in common.LOGS.glob("*.log")
                               if not p.name.startswith(("train_", "diag_", "books_", "smoke_", "ls"))) if common.LOGS.exists() else 0
    for cell in common.CELLS:
        row = {}
        for h in common.HORIZONS:
            seeds = len(list(common.PREDS.glob(f"{cell}_{h}_s*.npz"))) if common.PREDS.exists() else 0
            mean = common.pred_path(cell, h, "mean").exists(); ic = read_ic(cell, h)
            if seeds or mean or ic: row[h] = {"seeds_done": seeds, "mean": mean, "ic": ic}
        if row: st["cells"][cell] = row
        bp = common.RESULTS / f"{cell}_books.json"
        if bp.exists():
            try: b = json.loads(bp.read_text())
            except (OSError, json.JSONDecodeError): continue
            s = b.get("selected")
            if s: st["books"][cell] = {"book": s["cell"], "dev": s["dev"], "chk": s["chk"], "dev_gross_sharpe": s["dev_gross_sharpe"], "chk_gross_sharpe": s["chk_gross_sharpe"],
                                       "trial_count": b.get("trial_count"), "git_head": b.get("git_head"), "pass": bool(b.get("pass"))}
    best = max(st["books"].items(), key=lambda kv: kv[1]["dev"]["sharpe"], default=None)
    st["best"] = None if best is None else {"cell": best[0], **best[1]}
    ok = lambda s: s["dev"]["sharpe"] >= 1.5 and s["chk"]["sharpe"] >= 1.5 and s["dev"]["maxdd"] >= -0.25 and s["chk"]["maxdd"] >= -0.25
    st["books_clear_bar"] = sorted(c for c, b in st["books"].items() if ok(b))
    sp = common.REPORTS / "SELECTION.json"; st["selection"] = None
    if sp.exists():
        try:
            sj = json.loads(sp.read_text()); b = sj.get("best") or {}
            st["selection"] = {"pass": bool(sj.get("pass")), "N": sj.get("N"), "n_sets": sj.get("n_sets_evaluated"), "best_cell": b.get("cell"), "book_cell": b.get("book_cell"),
                               "dev": b.get("dev"), "chk": b.get("chk"), "dic_ok": sj.get("dic_ok"), "reason": sj.get("reason"), "null_max": (sj.get("null_max") or {}).get("expected_max"), "dsr": (sj.get("dsr") or {}).get("dsr_best")}
        except (OSError, json.JSONDecodeError): pass
    # PASS only by selection.py (eligibility, dIC vs C0 and C4); books alone never produce the PASS line
    st["goal"] = GOAL_PASS if st["selection"] and st["selection"]["pass"] else GOAL_NO
    return st


def fmt(st: dict) -> str:
    f = lambda x, p=3: "  n/a" if x is None else f"{x:+.{p}f}"
    q = st["queue"]; L = [f"P13 {st['time']}  queue: pending {q['pending']} running {q['running']} done {q['done']} failed {q['failed']}  models trained: {st['models_trained']}"]
    for r in st["running"][:4]: L.append(f"  RUN {r['id']} {r['elapsed_min']}min | {r['last']}")
    for cell, row in st["cells"].items():
        seg = []
        for h in common.HORIZONS:
            if h in row: c = row[h]; ic = c["ic"] or {}; seg.append(f"{h} s{c['seeds_done']}{'+m' if c['mean'] else ''} IC dev {f(ic.get('dev'))} chk {f(ic.get('chk'))}")
        L.append(f"  {cell}: " + " | ".join(seg))
    for cell, b in st["books"].items():
        d, c = b["dev"], b["chk"]
        L.append(f"  BOOK {cell}: {b['book']} dev net SR {d['sharpe']:+.2f} ann {d['ann']*100:+.1f}% DD {d['maxdd']*100:.0f}% gross {b['dev_gross_sharpe']:+.2f} | chk SR {c['sharpe']:+.2f} ann {c['ann']*100:+.1f}% DD {c['maxdd']*100:.0f}%")
    b = st["best"]
    sel = st.get("selection")
    if sel and sel.get("best_cell"):
        d, c = sel["dev"], sel["chk"]; L.append(f"  SELECTION (N={sel['N']}, {sel['n_sets']} sets): {sel['best_cell']}/{sel['book_cell']} dev {d['sharpe']:+.2f} (ann {d['ann']*100:+.1f}% DD {d['maxdd']*100:.0f}%) chk {c['sharpe']:+.2f} (ann {c['ann']*100:+.1f}% DD {c['maxdd']*100:.0f}%) dIC ok={sel['dic_ok']} null max {sel['null_max'] and round(sel['null_max'], 2)} DSR {sel['dsr'] and round(sel['dsr'], 3)}")
    elif sel: L.append("  SELECTION: no selectable set - " + "; ".join(sel.get("reason") or []))
    if st.get("books_clear_bar") and not (sel and sel["pass"]): L.append(f"  books >= 1.5 on both windows for {st['books_clear_bar']} but selection.py has not passed (run it; dIC needed)")
    L.append("  best so far: none" if b is None else f"  best so far: {b['cell']} / {b['book']} dev {b['dev']['sharpe']:+.2f} chk {b['chk']['sharpe']:+.2f}")
    L.append(st["goal"]); return "\n".join(L if len(L) <= 39 else L[:38] + [f"  ... {len(L) - 39} more lines in STATE.json", L[-1]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--json", action="store_true"); a = ap.parse_args(argv)
    st = collect(); common.REPORTS.mkdir(parents=True, exist_ok=True); (common.REPORTS / "STATE.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1) if a.json else fmt(st)); return 0


if __name__ == "__main__":
    sys.exit(main())
