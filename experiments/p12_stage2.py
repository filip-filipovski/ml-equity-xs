#!/usr/bin/env python
"""P12 Stage 2 — graph-ML arm (P12_PREDECLARATION.md): G1 prediction propagation over SIC4 peers and BEA hop-1 industry
neighbours on the cached T0/M2 predictions, p' = (1-a) p + a peer and p - a peer, a in {0.25, 0.5}; G2 M2 refit with
leave-one-out SIC4-peer means of eight characteristics as extra columns. TRAIN months only; sealed_load_count 0.
Books: Stage 0 'vs' / 'vsn' and the Stage 1b 'vsn4' construction. Writes p12_stage2.json."""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
RESULTS = HERE.parent / "results"; RESULTS.mkdir(exist_ok=True)
import p9_ml_capacity as P9  # noqa: E402
import p12_books as PB  # noqa: E402
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
D = P9.TL.CACHE / "equities_xs" / "derived"; P9.PANEL_PATH = D / "ml_panel_wfa2024.parquet"; P9.META_PATH = D / "ml_panel_wfa2024.meta.json"
CACHE = RESULTS / "p12_cache"; LAST = PB.LAST; E = P9.TL.CACHE / "links" / "edges"
panel = P9.Panel(); assert panel.sealed_load_count == 0; panel.valid = panel.valid & (panel.tmi <= LAST)
df = P9.TL.train_load(P9.PANEL_PATH, "p12 stage2 frame"); assert len(df) == panel.n
assert str(pd.to_datetime(df["date"])[panel.valid].max())[:10] < "2023-01-01"
tmi = panel.tmi; valid = panel.valid
# SIC4 at formation
mem = pd.read_parquet(E / "e1_sic_membership.parquet", columns=["cik", "sic4", "sic2", "valid_from", "valid_to"])
for c in ("valid_from", "valid_to"): mem[c] = pd.to_datetime(mem[c], utc=True).dt.tz_localize(None)
mem = mem.dropna(subset=["cik"]); mem["cik"] = mem["cik"].astype("int64"); mem = mem.sort_values("valid_from")
dates = pd.to_datetime(df["date"]).dt.tz_localize(None)
key = pd.DataFrame({"cik": df["cik"].astype("float").fillna(-1).astype("int64"), "date": dates, "_i": np.arange(len(df))}).sort_values("date")
j = pd.merge_asof(key, mem, left_on="date", right_on="valid_from", by="cik", direction="backward")
ok = j["valid_to"].isna() | (j["valid_to"] >= j["date"]); j.loc[~ok, ["sic4", "sic2"]] = np.nan; j = j.sort_values("_i")
sic4 = pd.Series(j["sic4"].to_numpy()).fillna(-1).astype(int).to_numpy(); sic2 = pd.Series(j["sic2"].to_numpy()).fillna(-1).astype(int).to_numpy()
div = np.argmax(df[[c for c in df.columns if c.startswith("sic_")]].to_numpy(), 1)
# BEA industry of each firm (sic4 -> bea) and the hop-1 adjacency (edges valid at the formation date; self-links excluded)
bmap = pd.read_parquet(E / "e5_sic4_bea_map.parquet"); bmap = bmap[bmap["kept"] == True] if "kept" in bmap else bmap
s2b = dict(zip(bmap["sic4"].astype(int), bmap["bea"].astype(str))); bea = np.array([s2b.get(int(s), "") for s in sic4], dtype=object)
edges = pd.read_parquet(E / "e5_bea_use_edges.parquet", columns=["sell_code", "buy_code", "weight", "is_diagonal", "valid_from", "valid_to"])
edges = edges[~edges["is_diagonal"].astype(bool)]
for c in ("valid_from", "valid_to"): edges[c] = pd.to_datetime(edges[c], utc=True).dt.tz_localize(None)
log("bea coverage of valid rows %.1f %%; edges %d" % (100 * np.mean(bea[valid] != ""), len(edges)))
years = list(range(2009, 2023)); refits = P9.make_refits(years); rows = np.concatenate([panel.year_scorer(y).rows for y in years]); sic_cols = [c for c in df.columns if c.startswith("sic_")]
skel = pd.DataFrame({"tmi": panel.tmi[rows], "firm": panel.firm[rows], "y": panel.y_raw[rows], "in_liq": panel.in_liq[rows], "beta": df["raw_beta_252"].to_numpy(float)[rows], "raw_vol_63": df["raw_vol_63"].to_numpy(float)[rows]})
for c in sic_cols: skel[c] = df[c].to_numpy(float)[rows]
skel["g4"] = sic4[rows]; skel["g2"] = sic2[rows]; skel["gd"] = div[rows]; skel["bea"] = bea[rows]; skel["date"] = dates.to_numpy()[rows]
cube = PB.Cube(skel.assign(pred=0.0)); G4, G2, GD = cube.mat("g4", -1), cube.mat("g2", -1), cube.mat("gd", -1); BETA = cube.B
import p12_stage1b_lib as L1  # noqa: E402  (vsn4 weights factored out below)

def peer_mean_matrix(P: np.ndarray, G: np.ndarray, min_n: int = 3) -> np.ndarray:
    """Leave-one-out mean of P within groups G per month (NaN where the group is too small or unknown)."""
    out = np.full_like(P, np.nan)
    for i in range(cube.T):
        m = ~np.isnan(P[i]) & (G[i] >= 0)
        if m.sum() == 0: continue
        s = pd.DataFrame({"p": P[i][m], "g": G[i][m]}); gg = s.groupby("g")["p"]; tot = gg.transform("sum").to_numpy(); n = gg.transform("size").to_numpy()
        loo = np.where(n >= min_n, (tot - s["p"].to_numpy()) / np.maximum(n - 1, 1), np.nan); idx = np.flatnonzero(m); out[i, idx] = loo
    return out

def bea_neighbour_matrix(P: np.ndarray) -> np.ndarray:
    """Weighted mean of the industry-mean prediction over BEA hop-1 neighbours (suppliers and customers), per month."""
    BEA = np.full((cube.T, cube.F), "", dtype=object); BEA[cube.ti, cube.fj] = skel["bea"].to_numpy(); out = np.full_like(P, np.nan)
    month_date = {int(t): d for t, d in zip(skel["tmi"], skel["date"])}
    for i in range(cube.T):
        d0 = pd.Timestamp(month_date[int(cube.months[i])]); m = ~np.isnan(P[i]) & (BEA[i] != "")
        if m.sum() == 0: continue
        ind = pd.DataFrame({"p": P[i][m], "b": BEA[i][m]}).groupby("b")["p"].mean()
        ev = edges[(edges["valid_from"] <= d0) & (edges["valid_to"].isna() | (edges["valid_to"] > d0))]
        if len(ev) == 0: ev = edges[edges["valid_from"] == edges["valid_from"].min()]
        nb = {}
        for a, b, w in zip(ev["sell_code"].astype(str), ev["buy_code"].astype(str), ev["weight"].astype(float)):
            if a in ind.index and b in ind.index:
                nb.setdefault(b, []).append((w, ind[a])); nb.setdefault(a, []).append((w, ind[b]))
        nbm = {k: sum(w * v for w, v in lst) / sum(w for w, _ in lst) for k, lst in nb.items() if sum(w for w, _ in lst) > 0}
        vals = np.array([nbm.get(b, np.nan) for b in BEA[i][m]]); idx = np.flatnonzero(m); out[i, idx] = vals
    return out

def books(P: np.ndarray) -> dict:
    res = {}
    for kind, k, liq, cost in (("vs", 1, False, 0.0015), ("vsn", 1, False, 0.0015), ("vs", 1, True, 0.0005)):
        res[f"{kind}_k{k}_{'LIQ5' if liq else 'ALL15'}"] = PB.metrics(PB.run_book(cube, P, kind, k, liq, cost))
    for liq, cost in ((False, 0.0015), (True, 0.0005)):
        res[f"vsn4_k1_{'LIQ5' if liq else 'ALL15'}"] = PB.metrics(L1.run_vsn4(cube, G4, G2, GD, BETA, P, 1, liq, cost))
    return res
def report(name, P):
    ic = PB.window_ic(cube, PB.ic_by_month(cube, P)); bk = books(P); best = max(bk.items(), key=lambda kv: kv[1]["dev"]["gross_sharpe"])
    log(f"{name:<34} IC dev {ic['dev']['mean']:+.4f} (t {ic['dev']['t']:.1f}, ICIR {ic['dev']['icir_ann']:.2f}) chk {ic['chk']['mean']:+.4f} (ICIR {ic['chk']['icir_ann']:.2f}) | best dev gross {best[1]['dev']['gross_sharpe']:.2f} net {best[1]['dev']['net_sharpe']:.2f} ({best[0]}) chk gross {best[1]['chk']['gross_sharpe']:.2f} net {best[1]['chk']['net_sharpe']:.2f}")
    return {"ic": ic, "books": bk}

out = {"design": __doc__, "G1": {}, "G2": {}}
P0 = cube.pred_matrix(np.load(CACHE / "stage1_T0_rank__M2.npz")["pred"]); out["G1"]["base_T0_M2"] = report("base T0 M2", P0)
Z0 = PB.smooth(cube, P0, 1)  # per-month centred ranks of the base prediction
peer4 = peer_mean_matrix(Z0, G4); peerB = bea_neighbour_matrix(Z0)
log("peer coverage: SIC4 %.1f %%, BEA %.1f %%" % (100 * np.mean(~np.isnan(peer4[~np.isnan(Z0)])), 100 * np.mean(~np.isnan(peerB[~np.isnan(Z0)]))))
for gname, PEER in (("sic4", peer4), ("bea1", peerB)):
    pe = np.where(np.isnan(PEER), 0.0, PEER)
    for a in (0.25, 0.5):
        out["G1"][f"{gname}_plus_a{a}"] = report(f"G1 {gname} +a{a} (propagate)", np.where(np.isnan(Z0), np.nan, (1 - a) * Z0 + a * pe))
        out["G1"][f"{gname}_minus_a{a}"] = report(f"G1 {gname} -a{a} (relative)", np.where(np.isnan(Z0), np.nan, Z0 - a * pe))
# G2: refit M2 with LOO SIC4-peer means of eight characteristics as extra columns
feats = [c for c in ("mom_12_1", "mom_6_1", "rev_1m", "vol_63", "ivol_63", "log_mcap", "bm", "sue") if c in panel.col_ix]
Xp = np.zeros((panel.n, len(feats)), np.float32)
g_all = pd.Series(sic4); key_all = pd.MultiIndex.from_arrays([tmi, sic4])
for jx, f in enumerate(feats):
    x = panel.X[:, panel.col_ix[f]].astype(float); s = pd.DataFrame({"x": x, "k1": tmi, "k2": sic4}); gg = s.groupby(["k1", "k2"])["x"]
    tot = gg.transform("sum").to_numpy(); n = gg.transform("size").to_numpy(); loo = np.where((n >= 3) & (sic4 >= 0), (tot - x) / np.maximum(n - 1, 1), 0.0); Xp[:, jx] = loo
X2 = np.ascontiguousarray(np.hstack([panel.X, Xp])); cols2 = list(panel.columns) + [f"peer4_{f}" for f in feats]
cf = CACHE / "stage2_G2_peer4_M2.npz"
if cf.exists(): pred2 = np.load(cf)["pred"]
else:
    t0 = time.time(); res = P9.fit_predict_cell(panel, X2, cols2, refits, "M2"); pred2 = np.concatenate([res["pred"][y] for y in years]); np.savez_compressed(cf, pred=pred2); log("G2 refit %.0f s" % (time.time() - t0))
out["G2"]["peer4_M2"] = report("G2 M2 + SIC4-peer means (8)", cube.pred_matrix(pred2))
out["sealed_load_count"] = int(panel.sealed_load_count); out["max_formation_date_used"] = str(pd.to_datetime(df["date"])[panel.valid].max())[:10]
(RESULTS / "p12_stage2.json").write_text(json.dumps(out, indent=1, default=float)); log("wrote p12_stage2.json")
