#!/usr/bin/env python
"""Build the paper figures (PDF, matplotlib Agg) from the stored artefacts.

Run:  <venv python> -I paper/build_figures.py

Reads only the JSON / markdown files named in build_tables.py (nothing under
data_cache/).  Writes paper/figures/*.pdf and copies p13_books.png.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # -I drops the script dir from sys.path

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

import build_tables as bt  # noqa: E402

OUT = HERE / "figures"
WRITTEN: list[Path] = []
MM = 1 / 25.4
W_IN = 160 * MM

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "font.family": "DejaVu Sans",
    "axes.linewidth": 0.6, "pdf.fonttype": 42,
})

UNI_COL = {"ALL": "#0072B2", "LIQ": "#E69F00", "TOP500": "#009E73"}
FAM_MARK = {"R1": "o", "R2": "s", "R4": "^", "R4b": "D"}
FAM_LAB = {"R1": "R1 weekly", "R2": "R2 quarterly slow", "R4": "R4 W+M+Q", "R4b": "R4b W+M"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    WRITTEN.append(p)


def plain_cell(name):
    return bt.pretty_cell(name).replace("$\\theta$", "θ")


# ---------------------------------------------------------------- figure 0 (schematic)
from matplotlib.patches import FancyBboxPatch, Circle  # noqa: E402

REL_COL = {"E1 SIC4": "#0072B2", "E4 customer–supplier": "#D55E00", "E5 BEA input–output": "#009E73", "CORR return correlation": "#CC79A7"}


def _box(ax, x, y, w, h, text, fc="#F4F4F4", ec="0.3", fs=6.1, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.7, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def _varrow(ax, x, y0, y1):
    ax.annotate("", (x, y1), (x, y0), arrowprops=dict(arrowstyle="-|>", color="0.25", lw=0.8, shrinkA=0, shrinkB=0), zorder=1)


def _check_text_in_boxes(fig, name, tol_px=0.5):
    """Gate: every text whose centre lies in a rounded box must fit inside that box (presenter-surface rule)."""
    from matplotlib.patches import FancyBboxPatch
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for ax in fig.axes:
        boxes = [p for p in ax.patches if isinstance(p, FancyBboxPatch)]
        for t in ax.texts:
            if not t.get_text().strip():
                continue
            tb = t.get_window_extent(r)
            cx, cy = tb.get_points().mean(0)
            for b in boxes:
                bb = b.get_window_extent(r)
                if not bb.contains(cx, cy):
                    continue
                over = max(bb.x0 - tb.x0, tb.x1 - bb.x1, bb.y0 - tb.y0, tb.y1 - bb.y1)
                if over > tol_px:
                    bad.append(f"{over:.1f}px: {t.get_text()[:50]!r}")
    if bad:
        raise RuntimeError(f"{name}: text overflows its box:\n  " + "\n  ".join(bad))


def fig_pipeline():
    fig = plt.figure(figsize=(W_IN, 3.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.0], wspace=0.04, left=0.01, right=0.99, top=0.97, bottom=0.03)
    axL, axG, axR = [fig.add_subplot(gs[0, k]) for k in range(3)]
    for ax in (axL, axG, axR):
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # ---- (a) panel -> features -> target
    axL.text(0.0, 0.985, "(a) panel at one date $t$", fontsize=7.2, fontweight="bold", va="top")
    _box(axL, 0.02, 0.73, 0.96, 0.15,
         "universe $U_t$: US common stocks,\nclose $\\geq$ \\$5, median \\$-volume $\\geq$ \\$1M")
    _varrow(axL, 0.5, 0.73, 0.655)
    _box(axL, 0.02, 0.45, 0.96, 0.205,
         "28 characteristics (prices, 8-K events,\nSEC XBRL fundamentals), each ranked\nwithin the date to $[-\\frac{1}{2},\\frac{1}{2}]$;\n"
         "missing flags; 11 SIC one-hots\n$\\Rightarrow\\ x_{i,t}\\in\\mathbb{R}^{67}$ (every firm $i$ is a node)")
    _varrow(axL, 0.5, 0.45, 0.39)
    _box(axL, 0.02, 0.20, 0.96, 0.19,
         "label $y^{rank}_{i,t}$: within-date rank of\nthe return from close $t{+}1$ to close $u{+}1$\n($u$ = next formation date);\nnever sent along an edge")
    _box(axL, 0.02, 0.02, 0.96, 0.15, "yearly expanding refit with embargo:\nno training label shares a daily\nreturn with a test label", fc="#FFFFFF", ec="0.5", fs=6.0)
    axL.annotate("", (1.03, 0.56), (0.99, 0.56), arrowprops=dict(arrowstyle="-|>", color="0.25", lw=0.9), annotation_clip=False)

    # ---- (b) graph: 2-hop receptive field
    axG.set_xlim(-3.05, 3.05)
    axG.set_ylim(-3.55, 3.75)
    axG.set_aspect("equal")
    axG.text(0.0, 3.72, "(b) firm graph: two layers = two hops", fontsize=7.2, fontweight="bold", va="top", ha="center")
    rels = list(REL_COL)
    r1, r2 = 1.05, 2.15
    ang1 = np.deg2rad([95, 167, 239, 311, 23])
    p1 = [(r1 * np.cos(a), r1 * np.sin(a) - 0.25) for a in ang1]
    ang2 = np.deg2rad([80, 118, 152, 190, 228, 262, 300, 335, 10, 45])
    p2 = [(r2 * np.cos(a), r2 * np.sin(a) - 0.25) for a in ang2]
    c0 = (0.0, -0.25)
    for rr, lab in ((r1 + 0.5, "hop 1: $N_r(i)$, read by layer 1"), (r2 + 0.5, "hop 2: read by layer 2 through $h^{(1)}_j$")):
        axG.add_patch(Circle(c0, rr, fill=False, ls=(0, (3, 2)), lw=0.6, ec="0.55", zorder=0))
    for rr, lab in ((r1 + 0.5, "hop 1"), (r2 + 0.5, "hop 2")):
        axG.text(c0[0] + rr * np.cos(np.deg2rad(135)), c0[1] + rr * np.sin(np.deg2rad(135)), lab, ha="center", va="center", fontsize=6.0, color="0.4", bbox=dict(fc="white", ec="none", pad=0.8), zorder=4)
    # ring-2 -> ring-1 edges (k -> j), ring-1 -> centre (j -> i)
    e2 = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (6, 3), (7, 3), (8, 4), (9, 4), (9, 0)]
    col2 = [rels[k % 4] for k in (0, 3, 1, 2, 0, 3, 2, 1, 0, 3, 2)]
    for (k, j), rl in zip(e2, col2):
        axG.annotate("", p1[j], p2[k], arrowprops=dict(arrowstyle="-|>", color=REL_COL[rl], lw=0.8, shrinkA=3.5, shrinkB=3.5, alpha=0.85), zorder=1)
    col1 = [rels[k] for k in (0, 1, 2, 3, 0)]
    for j, rl in enumerate(col1):
        axG.annotate("", c0, p1[j], arrowprops=dict(arrowstyle="-|>", color=REL_COL[rl], lw=1.1, shrinkA=4, shrinkB=4), zorder=1)
    for (x, y) in p2:
        axG.add_patch(Circle((x, y), 0.14, fc="white", ec="0.45", lw=0.7, zorder=3))
    for (x, y) in p1:
        axG.add_patch(Circle((x, y), 0.17, fc="#DDDDDD", ec="0.3", lw=0.8, zorder=3))
    axG.add_patch(Circle(c0, 0.22, fc="black", ec="black", zorder=3))
    axG.text(c0[0] + 0.32, c0[1] + 0.05, "$i$", fontsize=8, va="center", zorder=4)
    axG.text(p1[0][0] + 0.26, p1[0][1], "$j$", fontsize=7.5, va="center", zorder=4)
    axG.text(p2[0][0] + 0.24, p2[0][1], "$k$", fontsize=7.5, va="center", zorder=4)
    axG.text(0.0, 3.42, "layer 1: $h^{(1)}_i\\leftarrow x_i,\\ \\{x_j\\}_{j\\in N_r(i)}$\nlayer 2: $h^{(2)}_i\\leftarrow h^{(1)}_i,\\ \\{h^{(1)}_j\\}$, and $h^{(1)}_j$ holds $x_k$\n$\\hat y_i = w^\\top h^{(2)}_i + b$: firms two edges away influence $i$",
             ha="center", va="top", fontsize=6.0, linespacing=1.3, bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)
    handles = [Line2D([], [], c=c, lw=1.2, label=l) for l, c in REL_COL.items()]
    axG.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, fontsize=5.6, columnspacing=0.8, handlelength=1.4, handletextpad=0.4)
    axG.annotate("", (3.1, -0.25), (3.0, -0.25), arrowprops=dict(arrowstyle="-|>", color="0.25", lw=0.9), annotation_clip=False)

    # ---- (c) prediction -> IC and book
    axR.text(0.0, 0.985, "(c) score $\\to$ IC $\\to$ book", fontsize=7.2, fontweight="bold", va="top")
    _box(axR, 0.02, 0.775, 0.96, 0.115, "score $\\hat y_{i,t}$ for every firm and date\n(mean of three seeds)", fc="#E8E8E8", bold=True)
    _varrow(axR, 0.5, 0.775, 0.745)
    _box(axR, 0.02, 0.545, 0.96, 0.20,
         "rank IC per date against $y^{ex}$ (raw and\nindustry-demeaned bases); paired\n$\\Delta\\mathrm{IC}_d=\\mathrm{IC}_d(\\mathrm{graph})-\\mathrm{IC}_d(\\mathrm{C0})$\nC0 = same network, no edges;\nC4 = C2 on rewired edges (null)")
    _varrow(axR, 0.5, 0.545, 0.525)
    _box(axR, 0.02, 0.275, 0.96, 0.25,
         "aim $a_t$: centred rank $\\to$ divide by $\\sigma_i$\n$\\to\\ \\Sigma|w|=2$ $\\to$ industry demean\n$\\to$ project out $[\\mathbf{1},\\beta]$;\n"
         "partial adjustment\n$w_t=w_{t-1}+\\theta(a_t-w_{t-1})$;\n55 cells: aim $\\times$ universe $\\times\\ \\theta$")
    _varrow(axR, 0.5, 0.275, 0.25)
    _box(axR, 0.02, 0.01, 0.96, 0.24,
         "net$_t=w_t^\\top y^{raw}_t-c\\,\\|w_t-w_{t-1}\\|_1$\nSharpe, annualised return, MaxDD\npass: net SR $\\geq1.5$ and MaxDD $\\geq-25\\%$\non both windows; a graph cell also needs\ndevelopment $\\Delta$IC$>0$ (demeaned), $t\\geq2$,\nvs C0 and C4 on every horizon used", fc="#FFFFFF", ec="0.5")
    _check_text_in_boxes(fig, "fig_pipeline")
    save(fig, "fig_pipeline.pdf")


# ---------------------------------------------------------------- figure 1
def fig_cost_frontier():
    ev = bt.load_json("eval")
    cells, sel = ev["cells"], ev["selected"]["cell"]
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 3.2), sharey=False)
    for ax, w, ttl in ((axes[0], "dev", "Development 2009–2018"), (axes[1], "chk", "Check 2019–2022")):
        xs, ys = [], []
        for c, rec in cells.items():
            fam, _, uni, _ = bt.parse_cell(c)
            g = rec[w]["gross_sharpe"]
            n = rec[w][bt.DECL_KEY[uni]]["sharpe"]
            xs.append(g)
            ys.append(n)
            big = c == sel
            ax.scatter(g, n, s=70 if big else 22, marker=FAM_MARK[fam], c=UNI_COL[uni],
                       edgecolors="black" if big else "white", linewidths=1.3 if big else 0.4,
                       zorder=4 if big else 3)
            if big:
                ax.annotate(plain_cell(c), (g, n), xytext=(0.04, 0.88), textcoords="axes fraction",
                            ha="left", fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.5), zorder=5)
        xhi = max(max(xs), 1.5) + 0.15
        yhi = max(max(ys), 1.5) + 0.15
        xlo = min(min(xs), 0) - 0.15
        ylo = min(min(ys), 0) - 0.15
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        dlo, dhi = max(xlo, ylo), min(xhi, yhi)
        ax.plot([dlo, dhi], [dlo, dhi], ls=":", c="0.4", lw=0.8, zorder=1)
        ax.axhline(1.5, ls="--", c="0.25", lw=0.8, zorder=1)
        ax.axvline(1.5, ls="--", c="0.25", lw=0.8, zorder=1)
        ax.axhline(0, c="0.8", lw=0.5, zorder=0)
        ax.axvline(0, c="0.8", lw=0.5, zorder=0)
        ax.set_xlabel("gross Sharpe ratio")
        ax.set_ylabel("net Sharpe ratio at declared cost")
        ax.set_title(ttl, fontsize=8, loc="left")
        ax.grid(alpha=0.2, lw=0.4)
    handles = [Line2D([], [], marker="o", ls="", c=UNI_COL[u], markersize=5,
                      label=f"{u} ({bt.DECL_BPS[u]} bps)") for u in UNI_COL]
    handles += [Line2D([], [], marker=FAM_MARK[f], ls="", c="0.35", markersize=5, label=FAM_LAB[f]) for f in FAM_MARK]
    handles += [Line2D([], [], ls="--", c="0.25", lw=0.8, label="Sharpe 1.5"),
                Line2D([], [], ls=":", c="0.4", lw=0.8, label="net = gross")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.5, -0.12), columnspacing=1.2, handletextpad=0.3)
    fig.tight_layout()
    save(fig, "fig_cost_frontier.pdf")


# ---------------------------------------------------------------- figure 2
def fig_ic_by_year():
    ic = bt.load_json("full")["C1"]["ic_by_year"]
    years = list(range(2009, 2024))
    vals = np.array([ic[str(y)] for y in years])
    wins = [("development", 2009, 2018, "#DCE9F5"), ("check", 2019, 2022, "#FBE8C8"), ("hold-out", 2023, 2023, "#D5EBDD")]
    fig, ax = plt.subplots(figsize=(W_IN, 2.6))
    for lab, a, b, col in wins:
        ax.axvspan(a - 0.5, b + 0.5, color=col, zorder=0, lw=0)
    ax.bar(years, vals, width=0.7, color="0.35", zorder=3)
    ymax = vals.max() * 1.28
    ax.set_ylim(min(0, vals.min() * 1.2), ymax)
    for lab, a, b, col in wins:
        m = float(np.mean([ic[str(y)] for y in range(a, b + 1)]))
        ax.hlines(m, a - 0.45, b + 0.45, colors="#B22222", lw=1.4, zorder=4)
        ax.text((a + b) / 2, ymax * 0.97, (f"{lab}\nmean {m:+.3f}" if b > a else f"{lab}\n{m:+.3f}"), ha="center", va="top", fontsize=6.5 if b == a else 7)
    ax.axhline(0, c="k", lw=0.5, zorder=2)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.set_xlim(2008.4, 2023.6)
    ax.set_ylabel("rank IC (yearly mean)")
    ax.grid(axis="y", alpha=0.25, lw=0.4, zorder=1)
    ax.legend(handles=[Line2D([], [], c="#B22222", lw=1.4, label="window mean of yearly values")],
              loc="upper left", bbox_to_anchor=(0.0, 0.80), frameon=False, fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig_ic_by_year.pdf")


# ---------------------------------------------------------------- figure 3
def fig_p13_dic():
    md = bt.parse_results(bt.RESULTS_MD)
    rows = {}
    for r in md["dic"]:
        if r["vs"] != "C0" or r["cell"] not in ("C1", "C2", "C3"):
            continue
        if r["vals"][0] is None:
            continue
        rows[(r["cell"], r["h"])] = r
    order = [(c, h) for c in ("C1", "C2", "C3") for h in "WMQ" if (c, h) in rows]
    fig, ax = plt.subplots(figsize=(W_IN, 3.4))
    xs_all = []
    for i, key in enumerate(order):
        r = rows[key]
        for j, (kind, off, filled) in enumerate((("dev", -0.13, True), ("chk", +0.13, False))):
            idx = 0 if kind == "dev" else 1
            v = r["vals"][idx]
            if v is None:
                continue
            d, t = v[0]
            if t == 0:
                continue
            se = abs(d / t)
            lo, hi = d - 1.96 * se, d + 1.96 * se
            xs_all += [lo, hi]
            col = "#0072B2" if filled else "#D55E00"
            ax.hlines(i + off, lo, hi, colors=col, lw=1.1 if filled else 0.9, zorder=3)
            ax.plot(d, i + off, "o", ms=4.5, mfc=col if filled else "white", mec=col, mew=1.0, zorder=4)
            ax.text(hi + 0.0004, i + off, f"t={t:+.1f}", va="center", fontsize=6.2, color=col)
    ax.axvline(0, c="k", lw=0.7, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{c} {h}" for c, h in order])
    ax.invert_yaxis()
    ax.set_ylim(len(order) - 0.4, -0.6)
    span = max(xs_all) - min(xs_all)
    ax.set_xlim(min(xs_all) - 0.05 * span, max(xs_all) + 0.2 * span)
    for k in range(3):  # separators between C1 / C2 / C3 blocks
        if k:
            ax.axhline(3 * k - 0.5, c="0.8", lw=0.5)
    ax.set_xlabel("Δ rank IC vs C0 (demeaned basis), bars = ±1.96 SE recovered from t")
    ax.set_ylabel("cell and horizon")
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    ax.legend(handles=[Line2D([], [], marker="o", ls="-", c="#0072B2", ms=4.5, label="development 2009–2018"),
                       Line2D([], [], marker="o", ls="-", c="#D55E00", mfc="white", ms=4.5, label="check 2019–2022")],
              loc="upper right", frameon=False, fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig_p13_dic.pdf")


# ---------------------------------------------------------------- figure 4
def fig_power():
    pw = bt.load_json("pow")
    ticks = [6, 12, 24, 60, 120, 240]
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 3.0), sharey=True)

    def curve(ax, rec, label, color, ls="-"):
        ms = sorted(int(k) for k in rec["boot_power_by_n"])
        ax.plot(ms, [rec["boot_power_by_n"][str(m)] for m in ms], color=color, ls=ls, lw=1.2, label=label)

    cols = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, len(pw["ic_months_needed"])))
    for i, (rec, c) in enumerate(zip(pw["ic_months_needed"], cols)):
        curve(axes[0], rec, rec["claim"], c, ls="--" if rec["claim"].startswith("IC > 0.03") else "-")
    sh = [r for r in pw["sharpe_months_needed"] if r["sharpe"] in (0.3, 0.5, 1.0, 1.5)]
    cols = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, len(sh)))
    for rec, c in zip(sh, cols):
        curve(axes[1], rec, f"Sharpe {rec['sharpe']:g}", c)
    for ax, lab in zip(axes, ("forward months (IC claims)", "forward months (book Sharpe)")):
        ax.axhline(0.8, ls="--", c="0.3", lw=0.8)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xlim(3, 240)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(lab)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("power (bootstrap, one-sided 5 %)")
    axes[0].text(0.03, 0.8, "80 %", transform=axes[0].get_yaxis_transform(), fontsize=6.5, va="bottom", ha="left")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=5.8)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=6.5)
    axes[0].text(0.0, 1.02, "(a)", transform=axes[0].transAxes, fontsize=8, fontweight="bold")
    axes[1].text(0.0, 1.02, "(b)", transform=axes[1].transAxes, fontsize=8, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_power.pdf")


# ---------------------------------------------------------------- figure 5
def copy_books_png():
    src = bt.P13 / "figures" / "p13_books.png"
    dst = OUT / "p13_books.png"
    OUT.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print(f"WARNING: {src} missing; not copied")
        return
    shutil.copy(src, dst)
    WRITTEN.append(dst)


def main():
    fig_pipeline()
    fig_cost_frontier()
    fig_ic_by_year()
    fig_p13_dic()
    fig_power()
    copy_books_png()
    print("Files written:")
    for p in WRITTEN:
        print("  ", p)


if __name__ == "__main__":
    main()
