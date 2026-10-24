#!/usr/bin/env python
"""P13 book evaluation: a faithful port of reports/ml_meeting/p12c_eval.py (the P12c 55-cell book grid) into a function that takes
arbitrary prediction files in the common format (rows, pred) for the three horizons W / M / Q.

Aims, universes, cost tiers, thetas, DECL, w_vsn4 construction, partial adjustment, the >= 50-names rule, the selection rule and the pass rule are
byte-identical in logic to P12c. Development = formations 2009-2018, check = 2019-2022. Selection: highest development net Sharpe at the declared
cost of the cell's universe with development MaxDD >= -25 %. Pass: selected cell net Sharpe >= 1.5 on development AND check with MaxDD >= -25 % on both.
Every Sharpe is stored with its annualised return and MaxDD (stats dict). CLI: books.py --cell C2 [--pred mean|s0] [--drift] [--quick AIM ...]
or books.py --p12c (the P12c caches, key-less, regression only).
v1 (design review A3/A4/A7/A8; A10: refuses P13 sets unless ROOT/GATE_OPEN exists, rc 3): P13 predictions load through common.load_pred(verify=True), their hp_hash must be registered and their row set must equal
the P12c row set exactly; --drift adds net_*bps_drift (turnover charged against the weights drifted by the realised period return, gross unchanged);
the selected cell gets iid and 13-week block-bootstrap Sharpe SEs (2,000 seeded draws)."""
from __future__ import annotations
import argparse, json, math, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

sys.path.insert(0, str(common.ML_MEETING))
import p12_books as PB  # noqa: E402
import p12_stage1b_lib as L1  # noqa: E402

PER = 52; THETAS = (0.05, 0.1, 0.2, 0.5, 1.0); COSTS = (0.0001, 0.0002, 0.0005, 0.0015); DECL = {"ALL": "15", "LIQ": "5", "TOP500": "2"}; LAST = 2022 * 12 + 11
AIM_NAMES = ("R1_weekly", "R2_quarterly_slow", "R4_blend_WMQ", "R4b_blend_WM")
DESIGN = __doc__
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=common.REPO, capture_output=True, text=True, timeout=20).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class GateClosed(RuntimeError):
    pass


def gate_open() -> bool:
    return (common.ROOT / "GATE_OPEN").exists()


def require_gate() -> None:
    if not gate_open(): raise GateClosed(f"A10 gate closed: refusing to evaluate P13 predictions without {common.ROOT / 'GATE_OPEN'}")


def read_pred(path: Path, h: str, p12c: bool = False):
    """Returns (rows, pred, hp_hash, code_hash). P13 files: verified keys / fingerprint / row set + registered hash. p12c=True: the key-less P12c cache (regression only)."""
    if p12c:
        allowed = (common.ML_MEETING / "p12_cache").resolve(); rp = Path(path).resolve()
        assert rp.is_relative_to(allowed), f"--p12c may only read files under {allowed}, not {rp} (a P13 file would bypass the A10 gate)"
        z = np.load(path); return z["rows"].astype(np.int64), z["pred"], "p12c", None
    d = common.load_pred(Path(path), verify=True); hh = d["meta"].get("hp_hash"); assert hh, f"{path}: no hp_hash in meta"
    common.assert_registered(hh)
    assert np.array_equal(np.sort(d["rows"]), np.sort(common.p12c_rows(h))), f"{path}: row set differs from the P12c row set (A3)"
    return d["rows"], d["pred"], hh, d["meta"].get("code_hash")


def _load(h: str, cols: list[str], pred_path: Path, p12c: bool = False):
    pf = pd.read_parquet(common.panel_path(h)); pf = pf[[c for c in cols if c in pf.columns]]
    rows, p, hh, ch = read_pred(pred_path, h, p12c); pred = np.full(len(pf), np.nan); pred[rows] = p; pf["pred"] = pred
    pf["date"] = pd.to_datetime(pf["date"]).dt.tz_localize(None)
    pf = pf[(pf["target_mi"] >= 0) & (pf["target_mi"] <= LAST) & ~np.isnan(pred)].reset_index(drop=True)
    assert str(pf["date"].max())[:10] < "2023-01-01"
    log(f"{h}: {len(pf)} rows with predictions, {pf['date'].min().date()} .. {pf['date'].max().date()}"); return pf, hh, ch


def block_se(x: np.ndarray, block: int = 13, draws: int = 2000, seed: int = 0) -> float:
    """Circular block-bootstrap standard error of the annualised Sharpe of a weekly net-return series (A8 iii)."""
    x = np.asarray(x, float); n = len(x); rng = np.random.default_rng(seed); nb = -(-n // block)
    starts = rng.integers(0, n, size=(draws, nb)); idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(draws, -1)[:, :n]
    b = x[idx]; sd = b.std(axis=1, ddof=1); sr = np.where(sd > 0, b.mean(axis=1) / np.where(sd > 0, sd, 1) * math.sqrt(PER), np.nan)
    return float(np.nanstd(sr, ddof=1))


def iid_se(sr_ann: float, n: int) -> float:
    return float(math.sqrt((1 + 0.5 * (sr_ann / math.sqrt(PER)) ** 2) / n) * math.sqrt(PER))


def evaluate(pred_paths: dict[str, Path], tag: str, out: Path | None = None, aims: list[str] | None = None, drift: bool = False, p12c: bool = False, cell: str | None = None) -> dict:
    """pred_paths: {'W': npz, 'M': npz, 'Q': npz} in the common prediction format. aims: optional subset of AIM_NAMES (smoke runs only). p12c=True reads key-less P12c caches (regression only)."""
    if p12c:
        for q_ in pred_paths.values(): assert Path(q_).resolve().is_relative_to((common.ML_MEETING / "p12_cache").resolve()), f"--p12c may only read files under reports/ml_meeting/p12_cache, not {q_}"
    else: require_gate()   # A10: no P13 book before the orchestrator opens the gate (the key-less P12c regression is exempt)
    for h in common.HORIZONS: assert h in pred_paths, f"missing horizon {h}"
    base = ["firm", "date", "cik", "target_mi", "y_raw", "in_liq", "raw_vol_63", "raw_beta_252", "raw_log_dv_63"]
    W, hw, cw = _load("W", base + [f"sic_{c}" for c in "ABCDEFGHIJ"] + ["sic_unknown"], Path(pred_paths["W"]), p12c); sic_cols = [c for c in W.columns if c.startswith("sic_")]
    M, hm, cm = _load("M", ["firm", "date", "target_mi"], Path(pred_paths["M"]), p12c); Q, hq, cq = _load("Q", ["firm", "date", "target_mi"], Path(pred_paths["Q"]), p12c)
    assert hw == hm == hq, f"hp_hash differs across horizons: {hw} {hm} {hq}"
    mem = pd.read_parquet(common.EDGES / "e1_sic_membership.parquet", columns=["cik", "sic4", "sic2", "valid_from", "valid_to"])
    for c in ("valid_from", "valid_to"): mem[c] = pd.to_datetime(mem[c], utc=True).dt.tz_localize(None)
    mem = mem.dropna(subset=["cik"]); mem["cik"] = mem["cik"].astype("int64"); mem = mem.sort_values("valid_from")
    key = pd.DataFrame({"cik": W["cik"].astype("float").fillna(-1).astype("int64"), "date": W["date"], "_i": np.arange(len(W))}).sort_values("date")
    j = pd.merge_asof(key, mem, left_on="date", right_on="valid_from", by="cik", direction="backward"); ok = j["valid_to"].isna() | (j["valid_to"] >= j["date"])
    j.loc[~ok, ["sic4", "sic2"]] = np.nan; j = j.sort_values("_i")
    W["g4"] = pd.Series(j["sic4"].to_numpy()).fillna(-1).astype(int).to_numpy(); W["g2"] = pd.Series(j["sic2"].to_numpy()).fillna(-1).astype(int).to_numpy(); W["gd"] = np.argmax(W[sic_cols].to_numpy(), 1)
    dates = np.array(sorted(W["date"].unique())); dpos = {d: i for i, d in enumerate(dates)}; T = len(dates)
    firms = np.array(sorted(W["firm"].unique())); fpos = {int(f): k for k, f in enumerate(firms)}; F = len(firms); ti = W["date"].map(dpos).to_numpy(); fj = W["firm"].map(fpos).to_numpy()

    def mat(col, fill=np.nan):
        Mx = np.full((T, F), fill); Mx[ti, fj] = W[col].to_numpy(float); return Mx

    class C: pass
    cube = C(); cube.T, cube.F = T, F; cube.Y, cube.V, cube.B = mat("y_raw"), mat("raw_vol_63"), mat("raw_beta_252"); cube.LIQ = mat("in_liq", 0.0) > 0.5; cube.SIC = np.stack([mat(c, 0.0) for c in sic_cols], -1)
    PW = mat("pred"); DV = mat("raw_log_dv_63"); G4, G2, GD = mat("g4", -1), mat("g2", -1), mat("gd", -1); present = ~np.isnan(PW)
    vfloor = np.nanpercentile(cube.V, 5); cube.V = np.where(np.isnan(cube.V), np.nanmedian(cube.V), cube.V); cube.V = np.maximum(cube.V, vfloor)
    TOP = np.zeros((T, F), bool)
    for i in range(T):
        m = present[i] & ~np.isnan(DV[i]); idx = np.flatnonzero(m); TOP[i, idx[np.argsort(-DV[i][idx])][:500]] = True

    def carried(src):
        o = np.full((T, F), np.nan); fd = np.array(sorted(src["date"].unique())); snaps = {d: dict(zip(g["firm"].astype(int), g["pred"])) for d, g in src.groupby("date")}
        for i, d in enumerate(dates):
            k = np.searchsorted(fd, d, side="right") - 1
            if k < 0: continue
            for f, v in snaps[fd[k]].items():
                jx = fpos.get(f)
                if jx is not None and present[i, jx]: o[i, jx] = v
        return o

    PM, PQ = carried(M), carried(Q)

    def ranks(Px):
        Z = np.full_like(Px, np.nan)
        for i in range(T):
            m = ~np.isnan(Px[i]); Z[i] = PB.centred_rank_row(Px[i], m) if m.sum() else np.nan
        return Z

    ZW, ZM, ZQ = ranks(PW), ranks(PM), ranks(PQ)
    with np.errstate(all="ignore"):
        AIMS = {"R1_weekly": ZW, "R2_quarterly_slow": ZQ, "R4_blend_WMQ": np.nanmean(np.stack([ZW, ZM, ZQ]), 0), "R4b_blend_WM": np.nanmean(np.stack([ZW, ZM]), 0)}
    if aims is not None:
        unknown = set(aims) - set(AIMS); assert not unknown, f"unknown aims {unknown}"; AIMS = {k: v for k, v in AIMS.items() if k in aims}
    year = np.array([pd.Timestamp(d).year for d in dates]); DEVM = year <= 2018; CHKM = year >= 2019

    def stats(g, to, msk, tdr=None):
        gg, tt = g[msk], to[msk]; o = {"n_weeks": int(msk.sum()), "gross_ann": float(gg.mean() * PER), "gross_sharpe": float(gg.mean() / gg.std(ddof=1) * math.sqrt(PER)), "turnover_per_week": float(tt.mean())}
        for c in COSTS:
            net = gg - c * tt; sd = net.std(ddof=1); cum = np.cumprod(1 + net)
            o[f"net_{c*1e4:g}bps"] = {"sharpe": float(net.mean() / sd * math.sqrt(PER)) if sd > 0 else float("nan"), "ann": float(net.mean() * PER), "maxdd": float((cum / np.maximum.accumulate(cum) - 1).min())}
            zs = (net - net.mean()) / net.std(ddof=0) if net.std(ddof=0) > 0 else np.zeros_like(net)
            o[f"net_{c*1e4:g}bps"].update({"skew": float((zs ** 3).mean()), "kurt": float((zs ** 4).mean())})
            if tdr is not None:
                nd = gg - c * tdr[msk]; sdd = nd.std(ddof=1); cd = np.cumprod(1 + nd)
                o[f"net_{c*1e4:g}bps_drift"] = {"sharpe": float(nd.mean() / sdd * math.sqrt(PER)) if sdd > 0 else float("nan"), "ann": float(nd.mean() * PER), "maxdd": float((cd / np.maximum.accumulate(cd) - 1).min())}
        if tdr is not None: o["turnover_per_week_drift"] = float(tdr[msk].mean())
        return o

    series: dict = {}

    def run(Z, theta, uni, cid=None):
        prev = np.zeros(F); prev_d = np.zeros(F); G_, TO, TOD, used = [], [], [], []
        for i in range(T):
            m = present[i] & ~np.isnan(Z[i]) & (cube.LIQ[i] if uni == "LIQ" else TOP[i] if uni == "TOP500" else True)
            if m.sum() < 50: G_.append(0.0); TO.append(0.0); TOD.append(0.0); used.append(False); continue
            a = L1.w_vsn4(cube, G4, G2, GD, Z[i], i, m, cube.B[i]); w = prev + theta * (a - prev); w = np.where(m, w, 0.0)
            y = np.where(np.isnan(cube.Y[i]), 0.0, cube.Y[i]); G_.append(float(w @ y)); TO.append(float(np.abs(w - prev).sum())); used.append(True)
            TOD.append(float(np.abs(w - prev_d).sum()))   # A8 ii: charge against the weights drifted by the realised return of the last held period
            prev = w; wd = w * (1.0 + y); gd_ = np.abs(wd).sum(); prev_d = wd * (np.abs(w).sum() / gd_) if gd_ > 0 else wd
        g, to, tod, u = np.array(G_), np.array(TO), np.array(TOD), np.array(used)
        if cid is not None: series[cid] = (g, to, tod, u)
        td = tod if drift else None
        return {"dev": stats(g, to, DEVM & u, td), "chk": stats(g, to, CHKM & u, td)}

    res: dict = {"design": DESIGN, "tag": tag, "git_head": git_head(), "cell": cell, "pred_kind": "p12c" if p12c else "p13", "hp_hash": hw, "code_hash": {"W": cw, "M": cm, "Q": cq}, "drift": drift, "pred_paths": {h: str(pred_paths[h]) for h in common.HORIZONS}, "aims": list(AIMS), "quick": aims is not None, "cells": {}}
    log(f"{'cell':<34}{'dev gSR':>8}{'dev net@decl':>13}{'DD':>6} | {'chk gSR':>8}{'chk net@decl':>13}{'DD':>6}")
    for aname, Z in AIMS.items():
        for uni in ("ALL", "LIQ", "TOP500"):
            if aname.startswith("R2") and uni == "TOP500": continue
            for theta in THETAS:
                cid = f"{aname}_{uni}_theta{theta}"; r = run(Z, theta, uni, cid); res["cells"][cid] = r; k = f"net_{DECL[uni]}bps"; d, c = r["dev"], r["chk"]
                log(f"{cid:<34}{d['gross_sharpe']:>8.2f}{d[k]['sharpe']:>13.2f}{d[k]['maxdd']*100:>6.0f} | {c['gross_sharpe']:>8.2f}{c[k]['sharpe']:>13.2f}{c[k]['maxdd']*100:>6.0f}")
    res["trial_count"] = len(res["cells"])
    elig = [(cid, r) for cid, r in res["cells"].items() if r["dev"][f"net_{DECL[cid.split('_')[-2]]}bps"]["maxdd"] >= -0.25]
    if not elig:
        res["selected"] = None; res["pass"] = False
    else:
        sel = max(elig, key=lambda kv: kv[1]["dev"][f"net_{DECL[kv[0].split('_')[-2]]}bps"]["sharpe"]); k = f"net_{DECL[sel[0].split('_')[-2]]}bps"
        res["selected"] = {"cell": sel[0], "dev": sel[1]["dev"][k], "dev_gross_sharpe": sel[1]["dev"]["gross_sharpe"], "chk": sel[1]["chk"][k], "chk_gross_sharpe": sel[1]["chk"]["gross_sharpe"], "declared_cost_bps": k}
        res["pass"] = bool(sel[1]["dev"][k]["sharpe"] >= 1.5 and sel[1]["chk"][k]["sharpe"] >= 1.5 and sel[1]["dev"][k]["maxdd"] >= -0.25 and sel[1]["chk"][k]["maxdd"] >= -0.25)
        g, to, tod, u = series[sel[0]]; cc = float(k.split("_")[1][:-3]) * 1e-4; se = {}
        for side, msk in (("dev", DEVM & u), ("chk", CHKM & u)):
            net = (g - cc * to)[msk]; sr = sel[1][side][k]["sharpe"]
            se[side] = {"n_weeks": int(msk.sum()), "sharpe": sr, "se_iid": iid_se(sr, int(msk.sum())), "se_block13": block_se(net, 13, 2000, 0)}
        if drift: res["selected"]["dev_drift"] = sel[1]["dev"][k + "_drift"]; res["selected"]["chk_drift"] = sel[1]["chk"][k + "_drift"]
        res["selected"]["se"] = se; res["selected"]["uni"] = sel[0].split("_")[-2]
    if out is not None:
        out = Path(out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1, default=float))
    log("SELECTED:", json.dumps(res["selected"], default=float), "| PASS:", res["pass"], "|", f"wrote {out}" if out else "not written")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell"); ap.add_argument("--pred", default="mean", help="mean | s0 | s1 | ... (seed file)")
    ap.add_argument("--p12c", action="store_true", help="evaluate the key-less P12c caches (regression only; tag p12c)")
    ap.add_argument("--drift", action="store_true", help="also charge turnover against drifted weights (net_*bps_drift)")
    ap.add_argument("--quick", nargs="+", metavar="AIM", choices=AIM_NAMES, help="smoke run: evaluate only these aims (never a result)")
    ap.add_argument("--tag", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.p12c:
        paths = {h: common.ML_MEETING / "p12_cache" / f"p12c_{h}.npz" for h in common.HORIZONS}; a.cell = a.cell or "P12C"
    else:
        if not a.cell: ap.error("--cell is required unless --p12c")
        seed = "mean" if a.pred == "mean" else (int(a.pred.lstrip("s")) if a.pred.lstrip("s").isdigit() else a.pred)
        paths = {h: common.pred_path(a.cell, h, seed) for h in common.HORIZONS}
    if not a.p12c and not gate_open():
        print(f"A10 gate closed: books.py refuses P13 predictions without {common.ROOT / 'GATE_OPEN'}", file=sys.stderr); return 3
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing: print("missing prediction files:", *missing, sep="\n  ", file=sys.stderr); return 2
    pk = "p12c" if a.p12c else a.pred
    tag = a.tag or (a.cell if pk == "mean" else f"{a.cell}_{pk}") + ("_quick" if a.quick else "")
    common.ensure_dirs(); out = Path(a.out) if a.out else common.RESULTS / f"{tag}_books.json"
    res = evaluate(paths, tag, out, aims=a.quick, drift=a.drift, p12c=a.p12c, cell=a.cell)
    res["pred_set"] = pk; out.write_text(json.dumps(res, indent=1, default=float)); return 0


if __name__ == "__main__":
    sys.exit(main())
