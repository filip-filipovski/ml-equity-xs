#!/usr/bin/env python
"""P13 paired rank-IC diagnostics (design review A5 ii / A7): diag.py --cell C2 --vs C0.

For each horizon, the per-formation Spearman rank IC of the two cells' seed-mean predictions against y_ex_all, on two bases:
  raw       Spearman(pred, y_ex_all)
  demeaned  both pred and y_ex_all demeaned within SIC4 (fallback SIC2, then division) with p12_stage1b_lib.group_demean, the group mapping being
            the E1 membership-at-date merge_asof block of p12c_eval.py (what w_vsn4 can monetise)
and the paired difference (cell - vs). Dev = target year <= 2018, chk = 2019-2022: mean dIC and a month-level t (the per-formation series is
averaged within calendar months, then t = mean / (sd / sqrt(n_months)); for W this clusters the ~4 weekly formations of a month).
Writes RESULTS/dic_{cell}_vs_{vs}_{h}.json.  A10 gate: every entry that touches y_* refuses unless common.ROOT/GATE_OPEN exists (the file is the only switch).
A5: dev / chk are split by target year (target_mi // 12: <= 2018 dev, >= 2019 chk), as the trainer's test_mask does.
A6: for --vs C4 the json also holds dic_vs_seeds: dIC of the cell's mean against each C4 seed file (min / max / sd per basis and window), and dic_vs_seeds_paired: cell s_i vs C4 s_i for i = 0, 1, 2 (per-seed values, mean, sd)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

sys.path.insert(0, str(common.EXPERIMENTS))
import p12_stage1b_lib as L1  # noqa: E402


class GateClosed(RuntimeError):
    pass


def gate_open() -> bool:
    return (common.ROOT / "GATE_OPEN").exists()


def require_gate() -> None:
    if not gate_open(): raise GateClosed(f"A10 gate closed: refusing to compute IC / dIC on P13 predictions without {common.ROOT / 'GATE_OPEN'}")


def demean_cascade(w: np.ndarray, g4: np.ndarray, g2: np.ndarray, gd: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Demean w within SIC4, else SIC2, else division over the rows with m (same cascade as w_vsn4; groups < 3 names fall to the next level)."""
    w1, d1 = L1.group_demean(w, g4, m); w2, d2 = L1.group_demean(w, g2, m & ~d1); w3, _ = L1.group_demean(w, gd, m & ~d1 & ~d2)
    return np.where(d1, w1, np.where(d2, w2, w3))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3: return float("nan")
    ra, rb = rankdata(a), rankdata(b); ra -= ra.mean(); rb -= rb.mean(); den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def paired_ic(date_id, pred_a, pred_b, y, g4, g2, gd) -> pd.DataFrame:
    """Row-aligned arrays -> per-formation IC of both predictions on both bases (index = date_id). Gated (touches y)."""
    require_gate()
    date_id = np.asarray(date_id); order = np.argsort(date_id, kind="stable"); d = date_id[order]; cut = np.flatnonzero(np.diff(d)) + 1; starts = np.r_[0, cut]; ends = np.r_[cut, len(d)]
    out = []
    for s, e in zip(starts, ends):
        ix = order[s:e]; a, b, yy = np.asarray(pred_a, float)[ix], np.asarray(pred_b, float)[ix], np.asarray(y, float)[ix]
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(yy)
        if ok.sum() < 20: continue
        a, b, yy = a[ok], b[ok], yy[ok]; ones = np.ones(len(a), bool); G = [np.asarray(g, int)[ix][ok] for g in (g4, g2, gd)]
        yd = demean_cascade(yy, *G, ones); ad = demean_cascade(a, *G, ones); bd = demean_cascade(b, *G, ones)
        out.append({"date_id": int(d[s]), "n": int(ok.sum()), "raw_a": spearman(a, yy), "raw_b": spearman(b, yy), "dm_a": spearman(ad, yd), "dm_b": spearman(bd, yd)})
    return pd.DataFrame(out).set_index("date_id")


def month_t(values: np.ndarray, dates: np.ndarray) -> dict:
    """Mean and t of a per-formation series clustered by calendar month (average within month first)."""
    v = pd.Series(values, index=pd.to_datetime(dates)).dropna()
    if len(v) == 0: return {"mean": None, "t": None, "n_formations": 0, "n_months": 0}
    mv = v.groupby([v.index.year, v.index.month]).mean(); n = len(mv); sd = mv.std(ddof=1) if n > 1 else float("nan")
    return {"mean": float(v.mean()), "t": float(mv.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else None, "n_formations": int(len(v)), "n_months": int(n)}


def summarise(df: pd.DataFrame, dates: np.ndarray, target_mi: np.ndarray) -> dict:
    """df from paired_ic, dates = formation date per df row, target_mi = target period start month index per df row (same order).
    Window split by target year (target_mi // 12): <= 2018 dev, >= 2019 chk (A5). Gated (derived from y)."""
    require_gate()
    ty = np.asarray(target_mi) // 12; out = {}
    for name, msk in (("dev", ty <= common.DEV_LAST_YEAR), ("chk", ty >= common.DEV_LAST_YEAR + 1)):
        out[name] = {}
        for basis, (ca, cb) in (("raw", ("raw_a", "raw_b")), ("demeaned", ("dm_a", "dm_b"))):
            d = (df[ca] - df[cb]).to_numpy()[msk]; dt = np.asarray(dates)[msk]
            out[name][basis] = {"delta": month_t(d, dt), "ic_cell": month_t(df[ca].to_numpy()[msk], dt)["mean"], "ic_vs": month_t(df[cb].to_numpy()[msk], dt)["mean"]}
    return out


def sic_groups(pf: pd.DataFrame):
    """E1 membership-at-date merge_asof block copied from p12c_eval.py -> (g4, g2, gd) per panel row."""
    sic_cols = [c for c in pf.columns if c.startswith("sic_")]
    mem = pd.read_parquet(common.EDGES / "e1_sic_membership.parquet", columns=["cik", "sic4", "sic2", "valid_from", "valid_to"])
    for c in ("valid_from", "valid_to"): mem[c] = pd.to_datetime(mem[c], utc=True).dt.tz_localize(None)
    mem = mem.dropna(subset=["cik"]); mem["cik"] = mem["cik"].astype("int64"); mem = mem.sort_values("valid_from")
    key = pd.DataFrame({"cik": pf["cik"].astype("float").fillna(-1).astype("int64"), "date": pf["date"], "_i": np.arange(len(pf))}).sort_values("date")
    j = pd.merge_asof(key, mem, left_on="date", right_on="valid_from", by="cik", direction="backward"); ok = j["valid_to"].isna() | (j["valid_to"] >= j["date"])
    j.loc[~ok, ["sic4", "sic2"]] = np.nan; j = j.sort_values("_i")
    return (pd.Series(j["sic4"].to_numpy()).fillna(-1).astype(int).to_numpy(), pd.Series(j["sic2"].to_numpy()).fillna(-1).astype(int).to_numpy(), np.argmax(pf[sic_cols].to_numpy(), 1))


def _load_checked(cell: str, h: str, seed) -> dict:
    p = common.load_pred(common.pred_path(cell, h, seed), verify=True); common.assert_registered(p["meta"].get("hp_hash")); return p


def run_horizon(cell: str, vs: str, h: str) -> dict:
    require_gate()
    pa, pb = _load_checked(cell, h, "mean"), _load_checked(vs, h, "mean")
    oa, ob = np.argsort(pa["rows"]), np.argsort(pb["rows"]); assert np.array_equal(pa["rows"][oa], pb["rows"][ob]), "the two cells predict different row sets"
    rows = pa["rows"][oa]
    cols = ["firm", "date", "date_id", "cik", "target_mi", "y_ex_all"] + [f"sic_{c}" for c in "ABCDEFGHIJ"] + ["sic_unknown"]
    pf = pd.read_parquet(common.panel_path(h), columns=cols); pf["date"] = pd.to_datetime(pf["date"]).dt.tz_localize(None)
    g4, g2, gd = sic_groups(pf); sub = pf.iloc[rows]; y = sub["y_ex_all"].to_numpy(); G = (g4[rows], g2[rows], gd[rows]); did = sub["date_id"].to_numpy()

    def summary(pred_other, pred_cell=None):
        df = paired_ic(did, pa["pred"][oa] if pred_cell is None else pred_cell, pred_other, y, *G)
        first = sub.groupby("date_id")[["date", "target_mi"]].first().loc[df.index]
        return summarise(df, first["date"].to_numpy(), first["target_mi"].to_numpy()), len(df)

    main_sum, nform = summary(pb["pred"][ob])
    res = {"cell": cell, "vs": vs, "horizon": h, "hp_hash": pa["meta"].get("hp_hash"), "n_formations": int(nform), "split": "target_mi // 12 (dev <= 2018, chk >= 2019)", **main_sum}
    if vs == "C4":   # A6: the null is a distribution of three persistent rewirings; report the spread of the paired dIC over the draws
        per = []
        for sd_ in common.SEEDS:
            pc = _load_checked("C4", h, sd_); oc = np.argsort(pc["rows"]); assert np.array_equal(pc["rows"][oc], rows), "C4 seed row set differs"
            per.append(summary(pc["pred"][oc])[0])
        paired = []   # seed-paired: cell s_i vs C4 s_i, so both sides carry the same (single-seed) averaging
        for sd_ in common.SEEDS:
            pcell, pc = _load_checked(cell, h, sd_), _load_checked("C4", h, sd_); o1, o2 = np.argsort(pcell["rows"]), np.argsort(pc["rows"])
            assert np.array_equal(pcell["rows"][o1], rows) and np.array_equal(pc["rows"][o2], rows), "seed file row set differs"
            paired.append(summary(pc["pred"][o2], pcell["pred"][o1])[0])
        res["dic_vs_seeds_paired"] = {w: {b: spread([q[w][b]["delta"]["mean"] for q in paired]) | {"mean": _mean([q[w][b]["delta"]["mean"] for q in paired])} for b in ("raw", "demeaned")} for w in ("dev", "chk")}
        res["dic_vs_seeds"] = {w: {b: spread([q[w][b]["delta"]["mean"] for q in per]) for b in ("raw", "demeaned")} for w in ("dev", "chk")}
    common.ensure_dirs(); (common.RESULTS / f"dic_{cell}_vs_{vs}_{h}.json").write_text(json.dumps(res, indent=1, default=float)); return res


def _mean(v: list):
    v = [x for x in v if x is not None]; return float(np.mean(v)) if v else None


def spread(v: list) -> dict:
    v = np.array([x for x in v if x is not None], float)
    return {"per_seed": [float(x) for x in v], "min": float(v.min()), "max": float(v.max()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None} if len(v) else {"per_seed": [], "min": None, "max": None, "sd": None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--cell", required=True); ap.add_argument("--vs", required=True)
    ap.add_argument("--horizons", nargs="+", default=list(common.HORIZONS)); a = ap.parse_args(argv)
    try: require_gate()
    except GateClosed as e: print(str(e), file=sys.stderr); return 3
    for h in a.horizons:
        r = run_horizon(a.cell, a.vs, h); d = r["dev"]["demeaned"]["delta"]; print(f"{a.cell} vs {a.vs} {h}: dev demeaned dIC {d['mean']} t {d['t']} | raw {r['dev']['raw']['delta']['mean']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
