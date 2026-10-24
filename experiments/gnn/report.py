#!/usr/bin/env python
"""P13 RESULTS.md renderer. JSON reads only (ic / books / dic / ls / SELECTION jsons, trainer log line counts); never opens a panel or a prediction file.
Regenerates results/gnn/RESULTS.md and figures/p13_books.png from scratch on every run. Every number is read from a JSON; a missing file renders as "pending".
CLI: report.py [--out PATH]."""
from __future__ import annotations
import argparse, datetime as dt, json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

PENDING = "pending"
BAR = 1.5
DECL = {"ALL": "15", "LIQ": "5", "TOP500": "2"}
SEED_NAMES = ("s0", "s1", "s2")
MODEL_RE = re.compile(r"MODEL .* trained")


# ---- tiny readers / formatters -----------------------------------------------------------------------------------------
def _j(p: Path):
    try: return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError): return None


def sh(v) -> str: return PENDING if v is None else f"{v:.2f}"
def pc(v) -> str: return PENDING if v is None else f"{v * 100:.1f} %"
def ic(v) -> str: return PENDING if v is None else f"{v:+.4f}"
def tt(v) -> str: return "n/a" if v is None else f"{v:+.1f}"


def sr_full(d: dict | None) -> str:
    """Sharpe with annualised return and MaxDD beside it."""
    if not d or d.get("sharpe") is None: return PENDING
    return f"{d['sharpe']:+.2f} ({d['ann'] * 100:+.1f} %, DD {d['maxdd'] * 100:.1f} %)"


def ic_t(d: dict | None) -> str:
    return PENDING if not d or d.get("mean") is None else f"{ic(d['mean'])} (t {tt(d.get('t'))})"


def git_head() -> str:
    try: return subprocess.run(["git", "rev-parse", "HEAD"], cwd=common.REPO, capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception: return "unknown"


def count_models() -> int:
    n = 0
    for cell in common.CELLS:
        for h in common.HORIZONS:
            for f in Path(common.LOGS).glob(f"{cell}_{h}_s[0-9].log"):
                try: n += sum(1 for line in f.read_text(errors="replace").splitlines() if MODEL_RE.search(line))
                except OSError: pass
    return n


def table(head: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"] + ["| " + " | ".join(r) + " |" for r in rows]


# ---- data gathering -------------------------------------------------------------------------------------------------------
def ic_summary(cell: str, h: str) -> tuple[dict | None, dict | None, str]:
    """(dev, chk, mark): the seed-mean file, else the mean over the seed files present (marked)."""
    R = Path(common.RESULTS); m = _j(R / f"{cell}_{h}_mean_ic.json")
    if m: return m["development_2009_2018"], m["check_2019_2022"], ""
    ss = [j for j in (_j(R / f"{cell}_{h}_{s}_ic.json") for s in SEED_NAMES) if j]
    if not ss: return None, None, ""
    def avg(k):
        return {"mean": sum(j[k]["mean"] for j in ss) / len(ss), "t": sum(j[k]["t"] for j in ss) / len(ss)}
    return avg("development_2009_2018"), avg("check_2019_2022"), f" †{len(ss)}s"


def books_rows() -> list[dict]:
    rows = []
    for cell, spec in common.CELLS.items():
        b = _j(Path(common.RESULTS) / f"{cell}_books.json")
        if not b or b.get("quick") or not b.get("selected"): rows.append({"cell": cell, "eligible": bool(spec.get("eligible")), "books": None}); continue
        s = b["selected"]; uni = s.get("uni") or s["cell"].split("_")[-2]; se = s.get("se") or {}
        rows.append({"cell": cell, "eligible": bool(spec.get("eligible")), "books": b, "book_cell": s["cell"], "uni": uni, "dev": s["dev"], "chk": s["chk"],
                     "dev_gross": s.get("dev_gross_sharpe"), "chk_gross": s.get("chk_gross_sharpe"), "dev_drift": s.get("dev_drift"), "chk_drift": s.get("chk_drift"),
                     "se_dev": se.get("dev") or {}, "se_chk": se.get("chk") or {}, "cost": s.get("declared_cost_bps"), "pred_set": b.get("pred_set", "mean")})
    return rows


def best_eligible(rows: list[dict]) -> str | None:
    c = [r for r in rows if r.get("books") and r["eligible"] and r["pred_set"] == "mean" and r["dev"]["maxdd"] >= -0.25 and r["dev"].get("sharpe") is not None]
    return max(c, key=lambda r: r["dev"]["sharpe"])["cell"] if c else None


def title_claim(rows: list[dict]) -> str:
    """Computed from the data: either 'none eligible reaches 1.5' or the list of eligible cells that do (on dev and/or check)."""
    base = "net Sharpe at declared cost, selected book per cell (P13)"
    el = [r for r in rows if r.get("books") and r["eligible"]]
    if not el: return base + "; no eligible cell evaluated yet"
    hit = [f"{r['cell']} (dev {r['dev']['sharpe']:.2f} / chk {r['chk']['sharpe']:.2f})" for r in el if r["dev"]["sharpe"] >= BAR or r["chk"]["sharpe"] >= BAR]
    return base + ("; none eligible reaches 1.5" if not hit else "; eligible at or above 1.5 on dev or chk: " + ", ".join(hit))


def p12c_ref() -> dict | None:
    j = _j(Path(common.ML_RESULTS) / "p12c_eval.json")
    if not j or not j.get("selected"): return None
    s = j["selected"]; return {"cell": s["cell"], "dev": s["dev"], "chk": s["chk"], "dev_gross": s.get("dev_gross_sharpe"), "chk_gross": s.get("chk_gross_sharpe")}


# ---- sections ---------------------------------------------------------------------------------------------------------------
def sec_header(sel: dict | None) -> list[str]:
    goal = sel.get("goal") if sel and sel.get("pass") and sel.get("goal") else "GOAL: NOT MET"
    return ["# P13 RESULTS (auto-generated by report.py; do not edit)", "",
            f"- generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}", f"- git HEAD: `{git_head()}`", f"- hp_hash: `{common.hp_hash()}`",
            f"- refits trained (trainer log lines `MODEL .. trained`): {count_models()}", f"- {goal}", "",
            "Goal: net Sharpe > 1.5 and MaxDD < 25 % using graph methods. Every Sharpe below is followed by (annualised return, MaxDD).", ""]


def sec_ic() -> list[str]:
    out = ["## 1. Rank IC (seed mean; development 2009-2018 / check 2019-2022)", "", "Cell values: `dev IC (t) / chk IC (t)`. t is the naive per-formation t (no overlap correction). † = no seed-mean file yet; mean over the seed files present.", ""]
    rows = []
    for cell in common.CELLS:
        r = [cell]
        for h in common.HORIZONS:
            d, c, mk = ic_summary(cell, h); r.append(PENDING if d is None else f"{ic_t(d)} / {ic_t(c)}{mk}")
        rows.append(r)
    return out + table(["cell", "W", "M", "Q"], rows) + [""]


def sec_books(rows: list[dict]) -> list[str]:
    best = best_eligible(rows)
    out = ["## 2. Books (selected book cell per prediction set, declared cost of its universe)", "",
           "Format: `net Sharpe (ann %, MaxDD %)`. Drift = turnover charged against drifted weights. SE = Sharpe standard error (iid / 13-week block bootstrap). **bold** = best eligible cell by the rule (highest dev net Sharpe with dev MaxDD >= -25 %).", ""]
    tr = []
    for r in rows:
        nm = f"**{r['cell']}**" if r["cell"] == best else r["cell"]
        if not r.get("books"): tr.append([nm, "yes" if r["eligible"] else "no", PENDING] + [PENDING] * 8); continue
        se = lambda s: PENDING if s.get("se_iid") is None else f"{s['se_iid']:.2f} / {sh(s.get('se_block13'))}"
        tr.append([nm, "yes" if r["eligible"] else "no", f"`{r['book_cell']}` ({r['cost']})", sr_full(r["dev"]), sr_full(r["chk"]), sh(r["dev_gross"]), sh(r["chk_gross"]),
                   sr_full(r["dev_drift"]), sr_full(r["chk_drift"]), se(r["se_dev"]), se(r["se_chk"])])
    ref = p12c_ref()
    tr.append(["P12c (reference)", "n/a", f"`{ref['cell']}` (net_2bps)" if ref else PENDING, sr_full(ref["dev"]) if ref else PENDING, sr_full(ref["chk"]) if ref else PENDING,
               sh(ref["dev_gross"]) if ref else PENDING, sh(ref["chk_gross"]) if ref else PENDING] + ["n/a"] * 4)
    out += table(["cell", "eligible", "selected book (cost bps)", "dev net", "chk net", "dev gross", "chk gross", "dev drift-charged", "chk drift-charged", "SE dev", "SE chk"], tr)
    out += ["", f"Target: net Sharpe >= {BAR:.1f} on development AND check with MaxDD >= -25 % on both (selected eligible cell: C1, C2, C3, C5).", ""]
    return out


def dic_cell(d: dict | None, win: str, basis: str) -> str:
    try: x = d[win][basis]["delta"]; return f"{ic(x['mean'])} (t {tt(x['t'])})"
    except (TypeError, KeyError): return PENDING


def sec_dic() -> list[str]:
    out = ["## 3. Paired dIC (cell minus control, per formation, month-level t; demeaned basis primary, raw secondary)", "",
           "`*` marks development demeaned t >= 2. Demeaned = prediction and target demeaned within the book's SIC groups.", ""]
    rows = []; c4 = []
    for cell, spec in common.CELLS.items():
        if not spec.get("eligible"): continue
        for vs in ("C0", "C4", "C0g"):
            for h in common.HORIZONS:
                d = _j(Path(common.RESULTS) / f"dic_{cell}_vs_{vs}_{h}.json")
                if d is None: rows.append([cell, vs, h] + [PENDING] * 4); continue
                try: star = " *" if d["dev"]["demeaned"]["delta"]["t"] >= 2 else ""
                except (TypeError, KeyError): star = ""
                rows.append([cell, vs, h, dic_cell(d, "dev", "demeaned") + star, dic_cell(d, "chk", "demeaned"), dic_cell(d, "dev", "raw"), dic_cell(d, "chk", "raw")])
                if vs == "C4" and d.get("dic_vs_seeds"):
                    sp = (d["dic_vs_seeds"].get("dev") or {}).get("demeaned") or {}; pr = (d.get("dic_vs_seeds_paired", {}).get("dev") or {}).get("demeaned") or {}
                    f4 = lambda v: "n/a" if v is None else f"{v:+.4f}"
                    c4.append([cell, h, f"{f4(sp.get('min'))} / {f4(sp.get('max'))} / {f4(sp.get('sd'))}", f"{f4(pr.get('mean'))} / {f4(pr.get('sd'))}"])
    out += table(["cell", "vs", "h", "dev dIC demeaned", "chk dIC demeaned", "dev dIC raw", "chk dIC raw"], rows) + [""]
    out += ["C4 spread (development, demeaned): cell mean vs each of the three C4 draws (min / max / sd), and the seed-paired dIC (C_i vs C4_i; mean / sd).", ""]
    out += table(["cell", "h", "vs-draws min / max / sd", "seed-paired mean / sd"], c4) if c4 else ["pending (no dIC vs C4 files yet)"]
    return out + [""]


def sec_ls() -> list[str]:
    out = ["## 4. Label-shift diagnostic (seed 0; shifted-label model vs real model)", "",
           "Declared rule: signal persists under a one-year label shift if shifted dev IC > 0 with t >= 2 AND >= 0.5 x real dev IC; then a pass is relabelled \"not attributable to current information\".", ""]
    rows = []; any_drop = False; raw = []
    for cell in common.CELLS:
        for h in common.HORIZONS:
            K = common.LABEL_SHIFT_K[h]; s = _j(Path(common.RESULTS) / f"ls{K}_{cell}_{h}_s0_ic.json")
            if s is None: continue
            r = _j(Path(common.RESULTS) / f"{cell}_{h}_s0_ic.json"); sd, sc = s["development_2009_2018"], s["check_2019_2022"]
            rd, rc = (r["development_2009_2018"], r["check_2019_2022"]) if r else (None, None)
            drop = (s.get("meta") or {}).get("label_shift_dropped_share"); any_drop |= drop is not None
            ratio = None if not rd or not rd["mean"] else sd["mean"] / rd["mean"]
            flag = None if not rd else bool(sd["mean"] > 0 and sd["t"] >= 2 and rd["mean"] > 0 and sd["mean"] >= 0.5 * rd["mean"])
            raw.append([cell, f"{h} (K={K})", ic_t(sd), ic_t(sc), ic_t(rd), ic_t(rc), PENDING if ratio is None else f"{ratio:.2f}", PENDING if drop is None else pc(drop), PENDING if flag is None else str(flag)])
    head = ["cell", "h", "shifted dev IC (t)", "shifted chk IC (t)", "real dev IC (t)", "real chk IC (t)", "shifted/real (dev)", "dropped share", "persists under shift"]
    if raw and not any_drop: head.pop(7); raw = [r[:7] + r[8:] for r in raw]
    return out + (table(head, raw) if raw else ["pending (no label-shift files yet)"]) + [""]


def sec_selection(rows: list[dict]) -> list[str]:
    out = ["## 5. Selection (selection.py)", ""]; s = _j(Path(common.REPORTS) / "SELECTION.json")
    if s is None:
        pend = [r["cell"] for r in rows if r["eligible"] and not r.get("books")]
        return out + [f"selection.py not yet run (books pending: {', '.join(pend) if pend else 'none'})", ""]
    out.append(f"- prediction sets evaluated: {s.get('n_sets_evaluated')}; trial count N = {s.get('N')}")
    b = s.get("best")
    if b:
        out.append(f"- best eligible: {b['cell']} / `{b['book_cell']}` (declared cost {b['declared_cost_bps']} bps/side): dev {sr_full(b['dev'])}, chk {sr_full(b['chk'])}")
    nm = s.get("null_max")
    if nm: out.append(f"- exact expected maximum of {nm['N']} iid null Sharpes at the development SE {nm['se']:.2f}: {nm['expected_max']:.2f} (95th percentile of the max {nm['p95_max']:.2f})")
    d = s.get("dsr")
    if d and "error" not in d: out.append(f"- BLdP: cross-trial sd of Sharpe {d['cross_trial_sd_sharpe']:.2f}, SR0 {d['sr0_bldp_sharpe']:.2f}, deflated Sharpe of the best {d['dsr_best']:.3f}")
    elif d: out.append(f"- BLdP: {d['error']}")
    out.append(f"- verdict: **{s.get('label')}**. {' '.join(s.get('reason') or [])}")
    out.append(f"- {s.get('goal', 'GOAL: NOT MET')}")
    return out + [""]


def sec_figure(rows: list[dict]) -> list[str]:
    return ["## 6. Figure", "", "![P13 books](figures/p13_books.png)", "", f"Title (computed from the data): {title_claim(rows)}.", ""]


def make_figure(rows: list[dict], path: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, numpy as np, textwrap
    items = [(r["cell"] + ("" if r["eligible"] else " (control)"), r["dev"]["sharpe"], r["chk"]["sharpe"], r["se_dev"], r["se_chk"]) for r in rows if r.get("books")]
    ref = p12c_ref()
    if ref: items.append(("P12c (ref)", ref["dev"]["sharpe"], ref["chk"]["sharpe"], {}, {}))
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(items)); w = 0.38
    def err(items_, k):
        e = [(s.get("se_block13") if s.get("se_block13") is not None else s.get("se_iid")) for s in (i[k] for i in items_)]
        return np.array([np.nan if v is None else v for v in e])
    if items:
        ax.bar(x - w / 2, [i[1] for i in items], w, yerr=err(items, 3), capsize=3, label="development 2009-2018", color="#4C72B0")
        ax.bar(x + w / 2, [i[2] for i in items], w, yerr=err(items, 4), capsize=3, label="check 2019-2022", color="#DD8452")
        for xi, i in zip(x, items):
            ax.text(xi - w / 2, i[1], f"{i[1]:.2f}", ha="center", va="bottom" if i[1] >= 0 else "top", fontsize=7)
            ax.text(xi + w / 2, i[2], f"{i[2]:.2f}", ha="center", va="bottom" if i[2] >= 0 else "top", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels([i[0] for i in items], rotation=20, ha="right", fontsize=8)
    ax.axhline(BAR, color="red", ls="--", lw=1); ax.text(ax.get_xlim()[1], BAR, " goal", color="red", va="bottom", ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("net Sharpe"); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("\n".join(textwrap.wrap(title_claim(rows), 100)), fontsize=8)
    fig.text(0.01, 0.005, "error bars: 13-week block-bootstrap SE (iid SE if absent); P12c has none", fontsize=6)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=130); plt.close(fig)


def render(out: Path | None = None) -> Path:
    out = Path(out) if out else Path(common.REPORTS) / "RESULTS.md"; out.parent.mkdir(parents=True, exist_ok=True)
    rows = books_rows(); sel = _j(Path(common.REPORTS) / "SELECTION.json")
    make_figure(rows, out.parent / "figures" / "p13_books.png")
    md = sec_header(sel) + sec_ic() + sec_books(rows) + sec_dic() + sec_ls() + sec_selection(rows) + sec_figure(rows)
    out.write_text("\n".join(md).rstrip() + "\n"); return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--out", default=None); a = ap.parse_args(argv)
    print(render(a.out)); return 0


if __name__ == "__main__":
    sys.exit(main())
