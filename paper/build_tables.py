#!/usr/bin/env python
"""Build the LaTeX table fragments and number macros for the course paper.

Run:  <venv python> -I paper/build_tables.py [--selftest]

Every number is read from the stored JSON / markdown artefacts listed in
INPUTS below (nothing under data_cache/ is touched).  Missing values are
emitted as the string ``n/a`` and listed in the final report.

Outputs: paper/tables/*.tex  (booktabs ``tabular`` fragments and
numbers.tex with \\newcommand macros).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ML = ROOT / "results"
P13 = ROOT / "results" / "gnn"
OUT = HERE / "tables"

INPUTS = {
    "eval": ML / "p12c_eval.json",
    "wit": ML / "p12c_what_it_takes.json",
    "prec": ML / "p12c_precision.json",
    "st4": ML / "p12_stage4.json",
    "pow": ML / "p9_b_forward_power.json",
    "e6": ML / "p9_e6.json",
    "wfa": ML / "p9_b_wfa.json",
    "full": ML / "p9_b_fullhistory.json",
    "refit": ML / "p12c_refit.json",
    "fu": ML / "p9_b_followups.json",
    "state": P13 / "STATE.json",
}
RESULTS_MD = P13 / "RESULTS.md"

DECL_KEY = {"ALL": "net_15bps", "LIQ": "net_5bps", "TOP500": "net_2bps"}
DECL_BPS = {"ALL": 15, "LIQ": 5, "TOP500": 2}
COST_KEYS = [("net_1bps", 1), ("net_2bps", 2), ("net_5bps", 5), ("net_15bps", 15)]
SIM_NS = [294, 349, 404, 459, 514, 569]
SIM_DRAWS = 20000
SIM_SEED = 20261024

NA = "n/a"
NA_LOG: list[str] = []
WRITTEN: list[Path] = []
SOURCE_NOTES: list[str] = []


# ----------------------------------------------------------------- helpers
def na(where: str) -> str:
    NA_LOG.append(where)
    return NA


def load_json(key: str):
    with open(INPUTS[key]) as fh:
        return json.load(fh)


def esc(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace(">", r"$>$")
        .replace("<", r"$<$")
    )


def _fmt(x, dp: int, sign: bool) -> str:
    s = f"{x:.{dp}f}"
    if float(s) == 0.0:  # rounds to zero: unsigned zero
        return f"{0.0:.{dp}f}"
    if sign and x > 0:
        s = "+" + s
    return s


def tex_num(x, dp=2, sign=False, where="") -> str:
    """Number for table cells; negative sign rendered as math minus."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return na(where or "tex_num")
    return _fmt(x, dp, sign).replace("-", "$-$")


def plain_num(x, dp=2, sign=False, where="") -> str:
    """Number for macros: plain text, hyphen-minus."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return na(where or "plain_num")
    return _fmt(x, dp, sign)


def pct(x, dp=1, sign=False, tex=True, where="") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return na(where or "pct")
    f = tex_num if tex else plain_num
    return f(100.0 * x, dp, sign, where)


_CELL_RE = re.compile(r"^(R\d+b?)_(.+)_(ALL|LIQ|TOP500)_theta([0-9.]+)$")
_MID = {
    "weekly": "weekly",
    "quarterly_slow": "quarterly slow",
    "blend_WM": "W+M",
    "blend_WMQ": "W+M+Q",
}


def parse_cell(name: str):
    m = _CELL_RE.match(name)
    if not m:
        raise ValueError(f"unparsable cell name {name!r}")
    fam, mid, uni, theta = m.groups()
    return fam, mid, uni, theta


def universe(name: str) -> str:
    return parse_cell(name)[2]


def pretty_cell(name: str) -> str:
    fam, mid, uni, theta = parse_cell(name)
    return f"{fam} {_MID[mid]} / {uni} / $\\theta$={theta}"


def render(colspec: str, header: list[str], rows: list, ncols: int) -> str:
    """rows: list of cell-lists, or the string 'MID', or ('MULTI', text).
    MULTI rows are emitted as wrapped table notes under the tabular (threeparttable),
    so a long note never widens the table."""
    body = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    for h in header:
        body.append(h if h.startswith("\\cmidrule") else h + " \\\\")
    body.append("\\midrule")
    notes = []
    for r in rows:
        if r == "MID":
            body.append("\\midrule")
        elif isinstance(r, tuple) and r[0] == "MULTI":
            notes.append(r[1])
        else:
            assert len(r) == ncols, (len(r), ncols, r)
            body.append(" & ".join(r) + " \\\\")
    if body[-1] == "\\midrule":
        body.pop()
    body.append("\\bottomrule")
    body.append("\\end{tabular}")
    if not notes:
        return "\n".join(body) + "\n"
    out = ["\\begin{threeparttable}"] + body + ["\\begin{tablenotes}[flushleft]\\footnotesize"]
    out += [f"\\item[] {n}" for n in notes]
    out += ["\\end{tablenotes}", "\\end{threeparttable}"]
    return "\n".join(out) + "\n"


def write(name: str, text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(text)
    WRITTEN.append(p)


# ------------------------------------------------- P12c ladder arithmetic
def ladder_errors(ev: dict) -> float:
    """Max |stored net Sharpe - reconstructed| over cells x windows x costs."""
    worst = 0.0
    for cell, rec in ev["cells"].items():
        for w in ("dev", "chk"):
            r = rec[w]
            sigma = r["gross_ann"] / r["gross_sharpe"]
            for key, bps in COST_KEYS:
                drag = r["turnover_per_week"] * 52 * bps / 1e4
                net = (r["gross_ann"] - drag) / sigma
                worst = max(worst, abs(net - r[key]["sharpe"]))
    return worst


# ------------------------------------------------- expected max simulation
def simulate_expected_max(ns, draws=SIM_DRAWS, seed=SIM_SEED, years=10):
    """E[max of N iid null annualised Sharpes], each ~ N(0, 1/sqrt(years)).

    One draws x max(N) matrix; the N-trial maximum uses the first N columns
    (nested, so the estimates are monotone in N)."""
    rng = np.random.default_rng(seed)
    nmax = max(ns)
    z = rng.standard_normal((draws, nmax)) / np.sqrt(years)
    cm = np.maximum.accumulate(z, axis=1)
    return {n: float(cm[:, n - 1].mean()) for n in ns}


def asymptotic(n, years=10):
    return float(np.sqrt(2 * np.log(n)) / np.sqrt(years))


# ----------------------------------------------------------- P13 markdown
def parse_md_tables(text: str) -> dict[str, list[list[str]]]:
    out = {}
    for sec in re.split(r"^## ", text, flags=re.M)[1:]:
        title, _, body = sec.partition("\n")
        m = re.match(r"(\d+)\.", title.strip())
        if not m:
            continue
        rows = []
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue
            rows.append(cells)
        out[m.group(1)] = rows[1:]  # drop the header row
    return out


_IC_RE = re.compile(r"([+-]?\d+\.\d+) \(t ([+-]?\d+\.\d+)\)")
_NET_RE = re.compile(r"([+-]?\d+\.\d+) \(([+-]?\d+\.\d+) %, DD ([+-]?\d+\.\d+) %\)")
_SE_RE = re.compile(r"(\d+\.\d+) / (\d+\.\d+)")
_BOOK_RE = re.compile(r"`([^`]+)` \((net_\d+bps)\)")


def ic_pairs(s: str):
    """'a (t b) / c (t d)' -> [(a,b),(c,d)]; 'pending' -> None."""
    f = [(float(a), float(b)) for a, b in _IC_RE.findall(s)]
    if not f:
        if "pending" in s:
            return None
        raise ValueError(f"cannot parse IC entry {s!r}")
    return f


def parse_results(path: Path) -> dict:
    text = path.read_text()
    tabs = parse_md_tables(text)
    res = {"refits": None, "ic": [], "books": [], "dic": [], "shift": []}
    m = re.search(r"refits trained[^:]*:\s*(\d+)", text)
    if m:
        res["refits"] = int(m.group(1))
    for r in tabs["1"]:
        res["ic"].append({"cell": r[0].strip("*"), "h": dict(zip("WMQ", map(ic_pairs, r[1:4])))})
    for r in tabs["2"]:
        cell = r[0]
        bold = cell.startswith("**")
        rec = {"cell": cell.strip("*"), "bold": bold, "eligible": r[1]}
        mb = _BOOK_RE.search(r[2])
        rec["book"] = mb.group(1) if mb else None
        rec["cost"] = mb.group(2) if mb else None

        def net(s):
            mm = _NET_RE.search(s)
            return None if not mm else tuple(float(v) for v in mm.groups())

        rec["dev_net"], rec["chk_net"] = net(r[3]), net(r[4])
        rec["dev_gross"] = None if "pending" in r[5] or r[5] == "n/a" else float(r[5])
        rec["chk_gross"] = None if "pending" in r[6] or r[6] == "n/a" else float(r[6])
        for key, s in (("se_dev", r[9]), ("se_chk", r[10])):
            ms = _SE_RE.search(s)
            rec[key] = "n/a" if s.strip() == "n/a" else (None if not ms else (float(ms.group(1)), float(ms.group(2))))
        res["books"].append(rec)
    for r in tabs["3"]:
        res["dic"].append(
            {"cell": r[0], "vs": r[1], "h": r[2],
             "vals": [ic_pairs(x) for x in r[3:7]],  # each None or [(d,t)]
             "star": "*" in r[3]}
        )
    for r in tabs["4"]:
        res["shift"].append(
            {"cell": r[0], "h": r[1],
             "vals": [ic_pairs(x)[0] for x in r[2:6]],
             "ratio": float(r[6]), "persists": r[7].strip() == "True"}
        )
    return res


def cross_check_state(res: dict, state: dict) -> list[str]:
    msgs = []
    for rec in res["books"]:
        c = rec["cell"]
        if c not in state["books"]:
            continue
        sb = state["books"][c]
        if rec["book"] != sb["book"]:
            msgs.append(f"{c}: book md={rec['book']} json={sb['book']}")
        pairs = [
            ("dev net SR", rec["dev_net"][0], sb["dev"]["sharpe"], 0.01),
            ("chk net SR", rec["chk_net"][0], sb["chk"]["sharpe"], 0.01),
            ("dev gross SR", rec["dev_gross"], sb["dev_gross_sharpe"], 0.01),
            ("chk gross SR", rec["chk_gross"], sb["chk_gross_sharpe"], 0.01),
            ("dev ann %", rec["dev_net"][1], 100 * sb["dev"]["ann"], 0.06),
            ("chk ann %", rec["chk_net"][1], 100 * sb["chk"]["ann"], 0.06),
            ("dev DD %", rec["dev_net"][2], 100 * sb["dev"]["maxdd"], 0.06),
            ("chk DD %", rec["chk_net"][2], 100 * sb["chk"]["maxdd"], 0.06),
        ]
        for lab, a, b, tol in pairs:
            if abs(a - b) > tol:
                msgs.append(f"{c} {lab}: md={a} json={b}")
    return msgs


# ================================================================== tables
def build_all(selftest_only=False):
    ev = load_json("eval")
    wit = load_json("wit")
    prec = load_json("prec")
    st4 = load_json("st4")
    pw = load_json("pow")
    e6 = load_json("e6")
    wfa = load_json("wfa")
    full = load_json("full")
    refit = load_json("refit")
    state = load_json("state")
    md = parse_results(RESULTS_MD)

    cells = ev["cells"]
    sel = ev["selected"]
    sel_name = sel["cell"]
    sel_uni = universe(sel_name)
    sel_key = DECL_KEY[sel_uni]
    assert sel["declared_cost_bps"] == sel_key, (sel["declared_cost_bps"], sel_key)

    def dnet(c, w):
        return cells[c][w][DECL_KEY[universe(c)]]

    # ---- 1. IC windows ----------------------------------------------------
    rows = []
    hz = {"W": "weekly", "M": "monthly", "Q": "quarterly"}
    for h in "WMQ":
        r = refit[h]
        rows.append([r"dev 2009--2018", hz[h], "fixed booster (P12c)",
                     tex_num(r["ic_dev_mean"], 3, True), tex_num(r["ic_dev_t"], 1), str(r["n_dev_periods"])])
        rows.append([r"check 2019--2022", hz[h], "fixed booster (P12c)",
                     tex_num(r["ic_chk_mean"], 3, True), tex_num(r["ic_chk_t"], 1), str(r["n_chk_periods"])])
    rows.append("MID")
    for lab, key, nm in (("booster", "M2", "fixed booster (E6)"), ("ridge control", "M1", "ridge-all control (E6)")):
        s = e6[key]["stats_all"]
        rows.append([r"E6 2019--2022", "monthly", nm, tex_num(s["mean"], 3, True),
                     tex_num(s["t_month_iid"], 1), str(s["n_months"])])
    rows.append("MID")
    for key, nm in (("C1", "booster (hold-out)"), ("C0", "ridge-all control (hold-out)")):
        s = wfa[key]["stats_all"]
        rows.append([r"hold-out 2023", "monthly", nm, tex_num(s["mean"], 3, True),
                     tex_num(s["t_month_iid"], 1), str(s["n_months"])])
    write("tab_ic_windows.tex",
          render("llllrr", ["Window & Horizon & Model & Mean rank IC & $t$ & $n$ periods"], rows, 6))

    # ---- 2. P12c cells ----------------------------------------------------
    elig = []
    for c in cells:
        d = cells[c]["dev"][DECL_KEY[universe(c)]]
        if d["maxdd"] >= -0.25:
            elig.append((d["sharpe"], c))
    elig.sort(reverse=True)
    assert elig[0][1] == sel_name, ("selected cell is not the top-ranked eligible cell", elig[0], sel_name)
    top = [c for _, c in elig[:8]]
    if sel_name not in top:
        top.append(sel_name)
    rows = []
    for c in top:
        u = DECL_KEY[universe(c)]
        d, k = cells[c]["dev"], cells[c]["chk"]
        lab = pretty_cell(c) + (r"\,$\dagger$" if c == sel_name else "")
        rows.append([
            lab,
            tex_num(d["gross_sharpe"], 2), tex_num(d[u]["sharpe"], 2), pct(d[u]["ann"], 1, True), pct(d[u]["maxdd"], 1),
            tex_num(k["gross_sharpe"], 2), tex_num(k[u]["sharpe"], 2), pct(k[u]["ann"], 1, True), pct(k[u]["maxdd"], 1),
            tex_num(d["turnover_per_week"], 3),
        ])
    n_half = sum(1 for c in cells if dnet(c, "dev")["sharpe"] >= 0.5 and dnet(c, "chk")["sharpe"] >= 0.5)
    n_15 = sum(1 for c in cells if dnet(c, "dev")["sharpe"] >= 1.5 and dnet(c, "chk")["sharpe"] >= 1.5)
    mg_dev_c = max(cells, key=lambda c: cells[c]["dev"]["gross_sharpe"])
    mg_chk_c = max(cells, key=lambda c: cells[c]["chk"]["gross_sharpe"])
    mg_dev = cells[mg_dev_c]["dev"]["gross_sharpe"]
    mg_chk = cells[mg_chk_c]["chk"]["gross_sharpe"]
    rows.append("MID")
    rows.append(("MULTI",
                 f"Of {len(cells)} cells at the declared cost: net SR $\\ge 0.5$ on both windows: {n_half}; "
                 f"net SR $\\ge 1.5$ on both windows: {n_15}; "
                 f"max gross SR dev {tex_num(mg_dev, 2)} ({pretty_cell(mg_dev_c)}), "
                 f"chk {tex_num(mg_chk, 2)} ({pretty_cell(mg_chk_c)}). "
                 f"$\\dagger$ = selected cell."))
    write("tab_p12c_cells.tex", render(
        "lrrrrrrrrr",
        [r" & \multicolumn{4}{c}{Development 2009--2018} & \multicolumn{4}{c}{Check 2019--2022} &",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Cell & gross SR & net SR & ann.\ \% & MaxDD \% & gross SR & net SR & ann.\ \% & MaxDD \% & turnover/wk (dev)"],
        rows, 10))

    # ---- 3. cost ladder ---------------------------------------------------
    rows = []
    for w, lab in (("dev", "dev 2009--2018"), ("chk", "check 2019--2022")):
        r = cells[sel_name][w]
        row = [lab, f"{tex_num(r['gross_sharpe'], 2)} ({pct(r['gross_ann'], 1, True)})"]
        for key, _ in COST_KEYS:
            row.append(f"{tex_num(r[key]['sharpe'], 2)} ({pct(r[key]['ann'], 1, True)})")
        rows.append(row)
    rows.append(("MULTI", f"Selected cell {pretty_cell(sel_name)}; entries are Sharpe (annualised return \\%); "
                          f"declared cost {DECL_BPS[sel_uni]} bps per side."))
    write("tab_cost_ladder.tex", render(
        "lccccc", ["Window & gross & 1 bp & 2 bps & 5 bps & 15 bps"], rows, 6))

    # ---- 4. what it takes -------------------------------------------------
    S = wit["summary"]

    def wrow(case, cell, w, rec):
        return [case, pretty_cell(cell), w, tex_num(rec["gross_sharpe"], 2), pct(rec["sigma_ann"], 1),
                tex_num(rec["turns_per_year"], 1), str(rec["declared_bps"]), tex_num(rec["breakeven_bps"], 1),
                tex_num(rec["gross_needed_for_net15_at_declared"], 2), tex_num(rec["ic_multiple_needed"], 2)]

    wc = wit["cells"]
    rows = [
        wrow("selected", S["selected"], "dev", S["selected_dev"]),
        wrow("selected", S["selected"], "chk", S["selected_chk"]),
        wrow("least demanding (dev)", S["least_demanding_dev"], "dev", wc[S["least_demanding_dev"]]["dev"]),
        wrow("least demanding (chk)", S["least_demanding_chk"], "chk", wc[S["least_demanding_chk"]]["chk"]),
        wrow("least demanding (both)", S["least_demanding_both"], "dev", wc[S["least_demanding_both"]]["dev"]),
        wrow("least demanding (both)", S["least_demanding_both"], "chk", wc[S["least_demanding_both"]]["chk"]),
    ]
    nd = len(S["cells_with_positive_cost_for_net15"]["dev"])
    nc = len(S["cells_with_positive_cost_for_net15"]["chk"])
    rows.append("MID")
    rows.append(("MULTI",
                 f"Cells where a positive cost would give net 1.5: dev {nd}, chk {nc}. "
                 f"Max ladder reproduction error: {wit['ladder_check_max_abs_err']:.4f} Sharpe. "
                 f"$\\sigma$ = annualised book volatility; $\\kappa$ = first-order IC multiple needed."))
    write("tab_what_it_takes.tex", render(
        "llllrrrrrr",
        ["Case & Cell & Window & gross SR & $\\sigma$ ann.\\ \\% & turns/yr & declared bps & break-even bps & gross SR needed & $\\kappa$"],
        rows, 10))

    # ---- 5. precision -----------------------------------------------------
    PS = prec["summary"]
    rows = []
    for w, lab in (("dev", "dev"), ("chk", "chk"), ("pooled", "pooled")):
        for q in ("net", "gross"):
            r = PS[f"selected_{w}_{q}"]
            rows.append([lab, q, tex_num(r["sharpe"], 2), str(r["n_weeks"]), tex_num(r["se_iid"], 2),
                         f"[{tex_num(r['ci95'][0], 2)}, {tex_num(r['ci95'][1], 2)}]",
                         tex_num(r["z_to_target"], 2), tex_num(r["t_to_zero"], 2)])
    rows.append("MID")
    ge15 = PS["cells_pooled_net_upper95_ge_1.5"]
    ge10 = PS["cells_pooled_net_upper95_ge_1.0"]
    gg15 = PS["cells_pooled_gross_upper95_ge_1.5"]
    mp = PS["max_pooled_net"]
    nlist = lambda L: ", ".join(pretty_cell(c) for c in L) if L else "none"
    rows.append(("MULTI", f"Of {PS['n_cells']} cells, pooled net upper 95\\% bound $\\ge 1.5$: {len(ge15)} ({nlist(ge15)}); "
                          f"$\\ge 1.0$: {len(ge10)} ({nlist(ge10)}); pooled gross upper bound $\\ge 1.5$: {len(gg15)} ({nlist(gg15)})."))
    rows.append(("MULTI", f"Maximum pooled net SR: {tex_num(mp['sharpe'], 2)} ({pretty_cell(mp['cell'])}), "
                          f"95\\% CI [{tex_num(mp['ci95'][0], 2)}, {tex_num(mp['ci95'][1], 2)}]. "
                          f"SE is iid (Lo 2002); the selected cell is the maximum over {PS['n_cells']} cells."))
    write("tab_precision.tex", render(
        "llrrrcrr", ["Window & & SR & weeks & iid SE & 95\\% CI & $z$ to 1.5 & $t$ to 0"], rows, 8))

    # ---- 6. benchmarks ----------------------------------------------------
    wb, mb = st4["weekly_benchmarks"], st4["monthly_benchmarks"]
    w_all_icir = wb["ALL"]["spearman"]["icir_ann"]
    rows = []
    for u in ("ALL", "LIQ"):
        b = wb[u]
        c = lambda k: f"{tex_num(b[k]['icir_ann'], 2)} ({tex_num(b[k]['mean'], 3, True)})"
        rows.append([u, c("spearman"), c("pearson"), c("pearson_volscaled"), str(b["spearman"]["n_weeks"])])
    rows.append("MID")
    rows.append([r"\textit{Monthly}", "rank IC mean", "rank ICIR", "Pearson ICIR", "months"])
    for k, lab in (("dev", "dev"), ("chk", "check"), ("full", "full")):
        b = mb[k]
        rows.append([lab, tex_num(b["rank_ic_mean"], 3, True), tex_num(b["icir_ann_bound"], 2),
                     tex_num(b["pearson_icir_ann"], 2), str(b["n_months"])])
    rows.append("MID")
    rows.append([r"\textit{Transfer}", "max gross SR (55 cells)", "weekly ALL Spearman ICIR",
                 "transfer coef.", ""])
    tc = {}
    for w, lab in (("dev", "dev"), ("chk", "check")):
        mg = max(cells[c][w]["gross_sharpe"] for c in cells)
        tc[w] = mg / w_all_icir
        rows.append([lab, tex_num(mg, 2), tex_num(w_all_icir, 2), tex_num(tc[w], 2), ""])
    rows.append(("MULTI", "Weekly: annualised ICIR (mean IC in parentheses). Transfer coefficient = max realised gross SR over the "
                          "55 cells divided by the weekly ALL Spearman ICIR (measured, both windows against the same weekly benchmark)."))
    write("tab_benchmarks.tex", render(
        "lcccc",
        [r"Weekly & Spearman ICIR (IC) & Pearson ICIR (IC) & Pearson vol-scaled ICIR (IC) & weeks"], rows, 5))

    # ---- 7. expected max --------------------------------------------------
    stored = st4["expected_max_of_N_noise_sharpes_10y_exact_mc"]
    asy = st4["asymptotic_sqrt2lnN_over_sqrt10"]
    all_ns = sorted({30, 100, 234, *SIM_NS})
    sim = simulate_expected_max(all_ns)
    chk294 = abs(sim[294] - stored["294"])
    print(f"[expected-max check] simulated N=294: {sim[294]:.4f}  stored: {stored['294']:.4f}  |diff| = {chk294:.4f}")
    for n in (30, 100, 234):
        print(f"[expected-max check] N={n}: simulated {sim[n]:.4f} stored {stored[str(n)]:.4f}")
    rows = []
    for n in all_ns:
        s = stored.get(str(n))
        a = asy.get(str(n))
        simulated_only = s is None
        rows.append([str(n) + (r"$^{\mathrm{sim}}$" if simulated_only else ""),
                     tex_num(s, 2, where=f"tab_expected_max stored N={n}") if s is not None else na(f"tab_expected_max stored N={n}"),
                     tex_num(sim[n], 2),
                     tex_num(a, 2) if a is not None else tex_num(asymptotic(n), 2)])
    rows.append(("MULTI", f"Null annualised Sharpe of a 10-year series $\\sim N(0, 1/\\sqrt{{10}})$. "
                          f"Rows marked sim.\\ are not stored: simulated only ({SIM_DRAWS:,} draws, seed {SIM_SEED}); "
                          f"asymptotic column for them is $\\sqrt{{2\\ln N}}/\\sqrt{{10}}$. "
                          f"349 = P12 total; 569 adds the P13 prediction sets C1, C2, C3, C5."))
    write("tab_expected_max.tex", render(
        "rccc", ["$N$ trials & exact MC (stored) & simulated & asymptotic $\\sqrt{2\\ln N}/\\sqrt{10}$"], rows, 4))

    # ---- 8. power ---------------------------------------------------------
    def bp(rec, m):
        v = rec["boot_power_by_n"].get(str(m))
        return pct(v, 0, where=f"tab_power boot {m}") if v is not None else na(f"tab_power boot {m}")

    pms = (6, 12, 24, 60, 120)
    rows = [("MULTI", r"\textit{A. Forward IC claims: $H_0$ IC $\le 0$ (one-sided 5\%); months needed and bootstrap power (\%)}")]
    for x in pw["ic_months_needed"]:
        rows.append([esc(x["claim"]), tex_num(x["delta"], 3), str(x["n80_analytic"]), str(x["n90_analytic"])]
                    + [bp(x, m) for m in pms])
    rows.append("MID")
    rows.append(("MULTI", r"\textit{B. Book Sharpe $>0$: months (years) needed and bootstrap power (\%)}"))
    for x in pw["sharpe_months_needed"]:
        def my(m, y):
            return f"{m} ({tex_num(y, 1)})" if m is not None and y is not None else na("tab_power sharpe months")
        rows.append([f"SR {tex_num(x['sharpe'], 1)}", "", my(x["n80_analytic_months"], x["n80_years"]),
                     my(x["n90_analytic_months"], x["n90_years"])] + [bp(x, m) for m in pms])
    write("tab_power.tex", render(
        "llrrrrrrr",
        [r"Claim / level & $\delta$ & months (80\%) & months (90\%) & 6 m & 12 m & 24 m & 60 m & 120 m"], rows, 9))

    # ---- 9. P13 IC --------------------------------------------------------
    rows = []
    for r in md["ic"]:
        row = [r["cell"]]
        for h in "WMQ":
            v = r["h"][h]
            if v is None:
                row.append("pending")
            else:
                (dm, dt), (cm_, ct) = v
                row.append(f"{tex_num(dm, 3, True)} ({tex_num(dt, 1, True)}) / {tex_num(cm_, 3, True)} ({tex_num(ct, 1, True)})")
        rows.append(row)
    rows.append(("MULTI", "Cell values: dev 2009--2018 IC ($t$) / check 2019--2022 IC ($t$); seed means; naive per-formation $t$."))
    write("tab_p13_ic.tex", render("llll", ["Cell & W & M & Q"], rows, 4))

    # ---- 10. P13 books ----------------------------------------------------
    def netcell(t):
        if t is None:
            return "pending"
        sr, ann, dd = t
        return f"{tex_num(sr, 2)} ({tex_num(ann, 1, True)}, {tex_num(dd, 1)})"

    def se(t):
        if t == "n/a":
            return "n/a"  # literal n/a in the source markdown (P12c reference row)
        return "pending" if t is None else f"{t[0]:.2f} / {t[1]:.2f}"

    rows = []
    for r in md["books"]:
        nm = r["cell"]
        if r["bold"]:
            nm = f"\\textbf{{{esc(nm)}}}"
        elif r["cell"].startswith("P12c"):
            nm = "P12c (ref.)"
        else:
            nm = esc(nm)
        rows.append([nm, r["eligible"], pretty_cell(r["book"]) if r["book"] else "pending",
                     netcell(r["dev_net"]), netcell(r["chk_net"]),
                     "pending" if r["dev_gross"] is None else tex_num(r["dev_gross"], 2),
                     "pending" if r["chk_gross"] is None else tex_num(r["chk_gross"], 2),
                     se(r["se_dev"]), se(r["se_chk"])])
    rows.append(("MULTI", "Net SR (annualised return \\%, MaxDD \\%) at the declared cost of the book's universe; "
                          "SE = Sharpe standard error, iid / 13-week block bootstrap; bold = best eligible cell."))
    write("tab_p13_books.tex", render(
        "llllcrrcc",
        ["Cell & eligible & selected book & dev net & chk net & dev gross & chk gross & SE dev & SE chk"], rows, 9))

    # ---- 11. P13 dIC ------------------------------------------------------
    rows = []
    for r in md["dic"]:
        if r["vals"][0] is None:
            continue
        cellsv = []
        for i, v in enumerate(r["vals"]):
            if v is None:
                cellsv.append("pending")
                continue
            d, t = v[0]
            s = f"{tex_num(d, 4, True)} ($t$ {tex_num(t, 1, True)})"
            if i == 0 and abs(t) >= 2:
                s += "\\,*"
            cellsv.append(s)
        rows.append([r["cell"], r["vs"], r["h"]] + cellsv)
    rows.append(("MULTI", "Paired $\\Delta$IC (cell minus control) per formation, month-level $t$. "
                          "* = |$t$| $\\ge 2$ on the development demeaned basis. Rows with pending C4/C5 results omitted."))
    write("tab_p13_dic.tex", render(
        "lllllll",
        ["Cell & control & $h$ & dev demeaned & chk demeaned & dev raw & chk raw"], rows, 7))

    # ---- 12. label shift --------------------------------------------------
    rows = []
    for r in md["shift"]:
        v = r["vals"]
        f = lambda p: f"{tex_num(p[0], 3, True)} ($t$ {tex_num(p[1], 1, True)})"
        rows.append([r["cell"], esc(r["h"]).replace("(", "(").replace(")", ")"), f(v[0]), f(v[1]), f(v[2]), f(v[3]),
                     tex_num(r["ratio"], 2), "yes" if r["persists"] else "no"])
    rows.append(("MULTI", "Persists = shifted dev IC $>0$ with $t\\ge 2$ and $\\ge 0.5\\times$ the real dev IC (seed 0)."))
    write("tab_label_shift.tex", render(
        "llccccrl",
        ["Cell & $h$ & shifted dev & shifted chk & real dev & real chk & shifted/real (dev) & persists"], rows, 8))

    # ---- 13. IC by year ---------------------------------------------------
    c1, c1b = full["C1"], full["C1b"]
    rows = []
    for y in range(2009, 2024):
        ys = str(y)
        win = ("dev 2009--2018" if y == 2009 else "") if y <= 2018 else (
            "check 2019--2022" if y == 2019 else ("hold-out 2023" if y == 2023 else ""))
        if y == 2019 or y == 2023:
            rows.append("MID")
        rows.append([win, ys, tex_num(c1["ic_by_year"][ys], 3, True), tex_num(c1b["ic_by_year"][ys], 3, True)])
    rows.append("MID")
    rows.append(["full period", "mean", tex_num(c1["ic_full_mean"], 3, True), tex_num(c1b["ic_full_mean"], 3, True)])
    rows.append(["", "monthly $t$", tex_num(c1["ic_full_month_t"], 1), tex_num(c1b["ic_full_month_t"], 1)])
    write("tab_ic_by_year.tex", render("llrr", ["Window & Year & C1 (booster) & C1b (split-safe)"], rows, 4))

    # ---- 13b. crisis years in the check window (post-hoc diagnostic) ------
    # Per-year series stored in p9_b_fullhistory.json: yearly mean monthly rank IC (C1 booster),
    # calendar-year net return of the monthly decile book (ALL, 15 bps) and of the neutral 3-month-cohort book.
    icy = c1["ic_by_year"]
    dec_py = c1["books"]["B0_decile"]["pre_2009_2022"]["per_year_net"]
    neu_py = c1["books"]["B3_neutral_cohort3"]["pre_2009_2022"]["per_year_net"]
    dev_ys = [str(y) for y in range(2009, 2019)]
    chk_ys = [str(y) for y in range(2019, 2023)]
    ex20 = [y for y in chk_ys if y != "2020"]

    def ymean(d, ys):
        return float(np.mean([d[y] for y in ys]))

    worst_dec = min(dec_py, key=lambda y: dec_py[y])
    rows = []
    for y in chk_ys:
        rows.append([y, tex_num(icy[y], 3, True), pct(dec_py[y], 1, True), pct(neu_py[y], 1, True)])
    rows.append("MID")
    rows.append(["mean, development 2009--2018", tex_num(ymean(icy, dev_ys), 3, True), pct(ymean(dec_py, dev_ys), 1, True), pct(ymean(neu_py, dev_ys), 1, True)])
    rows.append(["mean, check 2019--2022", tex_num(ymean(icy, chk_ys), 3, True), pct(ymean(dec_py, chk_ys), 1, True), pct(ymean(neu_py, chk_ys), 1, True)])
    rows.append(["mean, check without 2020", tex_num(ymean(icy, ex20), 3, True), pct(ymean(dec_py, ex20), 1, True), pct(ymean(neu_py, ex20), 1, True)])
    rows.append([f"worst decile-book year ({worst_dec})", tex_num(icy[worst_dec], 3, True), pct(dec_py[worst_dec], 1, True), pct(neu_py[worst_dec], 1, True)])
    rows.append(("MULTI", "Yearly mean of the monthly rank IC of the fixed booster; calendar-year net return of the monthly decile "
                          "long--short book (ALL, 15 bps per side) and of a full-breadth rank-weighted book made dollar-, beta- and "
                          "SIC-division-neutral and held in three overlapping monthly cohorts (ALL, 15 bps). Post-hoc split of a "
                          "pre-declared window; means are of yearly values."))
    write("tab_crisis.tex", render("lrrr", ["Year & rank IC & decile book net \\%/yr & neutral cohort book net \\%/yr"], rows, 4))

    # ---- 14. numbers ------------------------------------------------------
    sd, sc = sel["dev"], sel["chk"]
    pd_, pc, pp = PS["selected_dev_net"], PS["selected_chk_net"], PS["selected_pooled_net"]
    mpw = wb["ALL"]
    icn = {x["claim"]: x for x in pw["ic_months_needed"]}
    icp = {x["claim"]: x for x in pw["ic_power_at_n"]}

    def claim(prefix):
        ks = [k for k in icn if k.startswith(prefix)]
        assert len(ks) == 1, (prefix, ks)
        return ks[0]

    k3, k6 = claim("IC > 0 at true 0.03 "), claim("IC > 0 at true 0.06 ")
    shn = {x["sharpe"]: x for x in pw["sharpe_months_needed"]}
    bk = {r["cell"]: r for r in md["books"]}
    dic_c0 = [r for r in md["dic"] if r["vs"] == "C0" and r["vals"][0] is not None]
    max_dic_t = max(abs(r["vals"][0][0][1]) for r in dic_c0)
    e6b, e6r = e6["M2"]["stats_all"], e6["M1"]["stats_all"]
    w1, w0 = wfa["C1"]["stats_all"], wfa["C0"]["stats_all"]
    N = {}
    N["SelCell"] = esc(sel_name)
    N["SelDevNet"] = plain_num(sd["sharpe"], 2)
    N["SelDevAnn"] = pct(sd["ann"], 1, tex=False)
    N["SelDevDD"] = pct(sd["maxdd"], 1, tex=False)
    N["SelDevGross"] = plain_num(sel["dev_gross_sharpe"], 2)
    N["SelChkNet"] = plain_num(sc["sharpe"], 2)
    N["SelChkAnn"] = pct(sc["ann"], 1, tex=False)
    N["SelChkDD"] = pct(sc["maxdd"], 1, tex=False)
    N["SelChkGross"] = plain_num(sel["chk_gross_sharpe"], 2)
    N["SelDevSE"] = plain_num(pd_["se_iid"], 2)
    N["SelChkSE"] = plain_num(pc["se_iid"], 2)
    N["PooledNet"] = plain_num(pp["sharpe"], 2)
    N["PooledSE"] = plain_num(pp["se_iid"], 2)
    N["PooledUpper"] = plain_num(pp["ci95"][1], 2)
    N["SelDevZ"] = plain_num(pd_["z_to_target"], 2)
    N["SelChkZ"] = plain_num(pc["z_to_target"], 2)
    N["PooledZ"] = plain_num(pp["z_to_target"], 2)
    N["KappaSelDev"] = plain_num(S["selected_dev"]["ic_multiple_needed"], 2)
    N["KappaSelChk"] = plain_num(S["selected_chk"]["ic_multiple_needed"], 2)
    N["KappaLeastBoth"] = plain_num(S["ic_multiple_least_both"], 2)
    N["KappaLeastDev"] = plain_num(S["ic_multiple_least_dev"], 2)
    N["KappaLeastChk"] = plain_num(S["ic_multiple_least_chk"], 2)
    N["MaxGrossDev"] = plain_num(mg_dev, 2)
    N["MaxGrossChk"] = plain_num(mg_chk, 2)
    N["CellsHalfBoth"] = str(n_half)
    N["BreakevenSelDev"] = plain_num(S["selected_dev"]["breakeven_bps"], 1)
    N["BreakevenSelChk"] = plain_num(S["selected_chk"]["breakeven_bps"], 1)
    N["ExpMaxTwoNineFour"] = plain_num(stored["294"], 2)
    N["ExpMaxThreeFourNine"] = plain_num(sim[349], 2)
    N["ExpMaxFiveSixNine"] = plain_num(sim[569], 2)
    N["ICWeeklyDev"] = plain_num(refit["W"]["ic_dev_mean"], 3, True)
    N["ICWeeklyDevT"] = plain_num(refit["W"]["ic_dev_t"], 1)
    N["ICWeeklyChk"] = plain_num(refit["W"]["ic_chk_mean"], 3, True)
    N["ICWeeklyChkT"] = plain_num(refit["W"]["ic_chk_t"], 1)
    N["ICMonthlyDev"] = plain_num(refit["M"]["ic_dev_mean"], 3, True)
    N["ICMonthlyChk"] = plain_num(refit["M"]["ic_chk_mean"], 3, True)
    N["ICQuarterlyDev"] = plain_num(refit["Q"]["ic_dev_mean"], 3, True)
    N["ICQuarterlyChk"] = plain_num(refit["Q"]["ic_chk_mean"], 3, True)
    N["ICEsixBooster"] = plain_num(e6b["mean"], 3, True)
    N["ICEsixBoosterT"] = plain_num(e6b["t_month_iid"], 1)
    N["ICEsixRidge"] = plain_num(e6r["mean"], 3, True)
    N["ICTwentyThree"] = plain_num(w1["mean"], 3, True)
    N["ICTwentyThreeT"] = plain_num(w1["t_month_iid"], 1)
    N["ICTwentyThreeRidge"] = plain_num(w0["mean"], 3, True)
    N["ICFullMean"] = plain_num(c1["ic_full_mean"], 3, True)
    N["ICFullT"] = plain_num(c1["ic_full_month_t"], 1)
    N["WeeklyRankICIR"] = plain_num(mpw["spearman"]["icir_ann"], 2)
    N["WeeklyPearsonICIR"] = plain_num(mpw["pearson"]["icir_ann"], 2)
    N["TransferCoefDev"] = plain_num(tc["dev"], 2)
    # Like-for-like per-window ICIR of the P12c weekly refit itself: ICIR_ann = t * sqrt(52 / n)
    for lab, key in (("Dev", "dev"), ("Chk", "chk")):
        t_ = refit["W"][f"ic_{key}_t"]; n_ = refit["W"][f"n_{key}_periods"]
        icir = t_ * math.sqrt(52.0 / n_)
        N[f"WeeklyRankICIRRefit{lab}"] = plain_num(icir, 2)
        N[f"TransferCoefRefit{lab}"] = plain_num((mg_dev if key == "dev" else mg_chk) / icir, 2)
    # Numbers the fact-check derived by hand, now generated (FACTCHECK_PAPER.md W2, W3, W18, O7, O8, O11)
    N["ICYearTwentyOne"] = plain_num(c1["ic_by_year"]["2021"], 3, True)
    _ic22 = pw["ic_2009_2022"]
    N["ICMeanTwentyTwo"] = plain_num(_ic22["mean"], 3, True)
    N["ICSdTwentyTwo"] = plain_num(_ic22["sd"], 3)
    N["ICLagTwentyTwo"] = plain_num(_ic22["lag1"], 2, True)
    _icir_chk = refit["W"]["ic_chk_t"] * math.sqrt(52.0 / refit["W"]["n_chk_periods"])
    N["TransferCoefChkSE"] = plain_num(pc["se_iid"] / _icir_chk, 2)
    _wc = load_json("wit")["cells"][S["least_demanding_both"]]
    N["LeastBothDevGross"] = plain_num(_wc["dev"]["gross_sharpe"], 2)
    N["LeastBothChkGross"] = plain_num(_wc["chk"]["gross_sharpe"], 2)
    _dec = c1["books"]["B0_decile"]["pre_2009_2022"]
    N["DecileSharpe"] = plain_num(_dec["net_sharpe"], 2)
    N["DecileAnn"] = pct(_dec["net_ann"], 1, tex=False)
    N["DecileDD"] = pct(_dec["maxdd"], 1, tex=False)
    _ev = load_json("eval")["cells"]
    def _rank(v):
        v = np.asarray(v, float); r = np.empty(len(v)); r[np.argsort(v)] = np.arange(len(v)); return r
    _names = sorted(_ev)
    _dn = [_ev[c]["dev"][DECL_KEY[universe(c)]]["sharpe"] for c in _names]
    _cn = [_ev[c]["chk"][DECL_KEY[universe(c)]]["sharpe"] for c in _names]
    _dg = [_ev[c]["dev"]["gross_sharpe"] for c in _names]
    _cg = [_ev[c]["chk"]["gross_sharpe"] for c in _names]
    N["XwinRhoNet"] = plain_num(float(np.corrcoef(_rank(_dn), _rank(_cn))[0, 1]), 2)
    N["XwinRhoGross"] = plain_num(float(np.corrcoef(_rank(_dg), _rank(_cg))[0, 1]), 2)
    N["MonthsICThree"] = str(icn[k3]["n80_analytic"])
    N["MonthsICSix"] = str(icn[k6]["n80_analytic"])
    N["YearsSharpeThree"] = plain_num(shn[0.3]["n80_years"], 1)
    N["YearsSharpeOne"] = plain_num(shn[1.0]["n80_years"], 1)
    N["PowerSixMonthsICThree"] = pct(icp[k3]["m6"], 1, tex=False)
    N["PowerSixMonthsICSix"] = pct(icp[k6]["m6"], 1, tex=False)
    for cn, nm in (("C0", "Czero"), ("C1", "Cone"), ("C2", "Ctwo"), ("C3", "Cthree")):
        N[f"Pthirteen{nm}DevNet"] = plain_num(bk[cn]["dev_net"][0], 2)
        N[f"Pthirteen{nm}ChkNet"] = plain_num(bk[cn]["chk_net"][0], 2)
    # Crisis-year diagnostic (Table tab_crisis)
    N["ICYearNineteen"] = plain_num(icy["2019"], 3, True)
    N["ICYearTwenty"] = plain_num(icy["2020"], 3, True)
    N["ICYearTwentyTwo"] = plain_num(icy["2022"], 3, True)
    N["ICDevYearMean"] = plain_num(ymean(icy, dev_ys), 3, True)
    N["ICChkYearMean"] = plain_num(ymean(icy, chk_ys), 3, True)
    N["ICChkYearMeanExTwenty"] = plain_num(ymean(icy, ex20), 3, True)
    N["DecileYearTwenty"] = pct(dec_py["2020"], 1, True, tex=False)
    N["DecileYearTwentyTwo"] = pct(dec_py["2022"], 1, True, tex=False)
    N["DecileDevYearMean"] = pct(ymean(dec_py, dev_ys), 1, True, tex=False)
    N["DecileChkYearMean"] = pct(ymean(dec_py, chk_ys), 1, True, tex=False)
    N["DecileChkYearMeanExTwenty"] = pct(ymean(dec_py, ex20), 1, True, tex=False)
    N["DecileWorstYear"] = worst_dec
    N["DecileWorstYearNet"] = pct(dec_py[worst_dec], 1, True, tex=False)
    N["NeutralYearTwenty"] = pct(neu_py["2020"], 1, True, tex=False)
    N["NeutralYearTwentyTwo"] = pct(neu_py["2022"], 1, True, tex=False)
    # Crisis months and window volatility (FACTCHECK_PAPER_2 A3, A5): generated, not hand-derived
    fu = load_json("fu")
    icm = {int(k): v for k, v in fu["ic_monthly_2009_2023"].items()}
    dnm = {int(k): v for k, v in fu["decile_net_monthly"].items()}
    tmi = lambda y, m: 12 * y + m - 1
    m168 = [t for t in sorted(icm) if 2009 <= t // 12 <= 2022]
    assert len(m168) == 168, len(m168)
    feb, mar, apr = tmi(2020, 2), tmi(2020, 3), tmi(2020, 4)
    N["ICFebTwenty"] = plain_num(icm[feb], 3, True)
    N["ICMarTwenty"] = plain_num(icm[mar], 3, True)
    N["ICAprTwenty"] = plain_num(icm[apr], 3, True)
    N["ICFebTwentyRank"] = str(1 + sum(1 for t in m168 if icm[t] < icm[feb]))
    y20 = np.array([icm[tmi(2020, m)] for m in range(1, 13)])
    assert abs(y20.mean() - icy["2020"]) < 1e-9
    N["ICTwentySE"] = plain_num(float(y20.std(ddof=1) / math.sqrt(12)), 3)
    N["DecileFebTwenty"] = pct(dnm[feb], 1, True, tex=False)
    N["DecileMarTwenty"] = pct(dnm[mar], 1, True, tex=False)
    N["DecileAprTwenty"] = pct(dnm[apr], 1, True, tex=False)
    _dev = np.array([dnm[t] for t in m168 if t // 12 <= 2018]); _chk = np.array([dnm[t] for t in m168 if t // 12 >= 2019])
    N["DecileVolDev"] = pct(float(_dev.std(ddof=1) * math.sqrt(12)), 1, tex=False)
    N["DecileVolChk"] = pct(float(_chk.std(ddof=1) * math.sqrt(12)), 1, tex=False)
    N["SelDevGrossAnn"] = pct(sd["gross_ann"] if "gross_ann" in sd else cells[sel_name]["dev"]["gross_ann"], 1, tex=False)
    N["SelChkGrossAnn"] = pct(cells[sel_name]["chk"]["gross_ann"], 1, tex=False)
    N["SelDevVol"] = pct(wc[sel_name]["dev"]["sigma_ann"], 1, tex=False)
    N["SelChkVol"] = pct(wc[sel_name]["chk"]["sigma_ann"], 1, tex=False)
    _vr = np.array([wc[c]["chk"]["sigma_ann"] / wc[c]["dev"]["sigma_ann"] for c in cells])
    N["VolRatioMin"] = plain_num(float(_vr.min()), 1)
    N["VolRatioMax"] = plain_num(float(_vr.max()), 1)
    N["VolRatioMedian"] = plain_num(float(np.median(_vr)), 1)
    N["CellsDeeperChkDD"] = str(sum(1 for c in cells if dnet(c, "chk")["maxdd"] < dnet(c, "dev")["maxdd"]))
    N["CellsHigherChkGrossAnn"] = str(sum(1 for c in cells if cells[c]["chk"]["gross_ann"] > cells[c]["dev"]["gross_ann"]))
    N["PthirteenRefits"] = str(md["refits"]) if md["refits"] is not None else na("PthirteenRefits")
    N["PthirteenMaxAbsDicT"] = plain_num(max_dic_t, 1)

    lines = ["% Auto-generated by reports/paper/build_tables.py -- do not edit.\n"]
    for k, v in N.items():
        assert re.fullmatch(r"[A-Za-z]+", k), k
        lines.append(f"\\newcommand{{\\num{k}}}{{{v}}}\n")
    write("numbers.tex", "".join(lines))

    # ---- cross-checks -----------------------------------------------------
    mism = cross_check_state(md, state)
    print("[STATE.json vs RESULTS.md]", "no mismatches" if not mism else "MISMATCHES:")
    for m in mism:
        print("   ", m)
    print(f"[ladder] max |stored - reconstructed| net Sharpe = {ladder_errors(ev):.5f}")
    print(f"[cells] net>=0.5 both windows: {n_half}; net>=1.5 both: {n_15}; max gross dev {mg_dev:.3f} ({mg_dev_c}); chk {mg_chk:.3f} ({mg_chk_c})")
    return N


def selftest() -> None:
    ev = load_json("eval")
    st4 = load_json("st4")
    err = ladder_errors(ev)
    stored = st4["expected_max_of_N_noise_sharpes_10y_exact_mc"]
    sim = simulate_expected_max([294])
    e2 = abs(sim[294] - stored["294"])
    print(f"selftest: ladder max abs error = {err:.6f} (limit 0.03)")
    print(f"selftest: expected-max N=294 simulated {sim[294]:.4f} vs stored {stored['294']:.4f}, abs error = {e2:.6f} (limit 0.015)")
    assert err < 0.03, err
    assert e2 < 0.015, e2
    print("selftest: OK")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    build_all()
    print("\nFiles written:")
    for p in WRITTEN:
        print("  ", p)
    if NA_LOG:
        print("\nn/a emitted for:")
        for x in NA_LOG:
            print("  ", x)
    else:
        print("\nNo n/a emitted.")


if __name__ == "__main__":
    main()
