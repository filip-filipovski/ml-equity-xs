"""P13 graph builder: one PIT graph-snapshot npz per (horizon, relation) via common.save_graph.

CLI: build_graphs.py --horizon {W,M,Q,all} --relation {E1_SIC4,E4_CUSTSUPP,E5_BEA,CORR_KNN,all} [--check] [--no-build]

Conventions (also written to each npz meta):
  * Snapshot i = common.date_index(h)[i]; a firm is present at i iff it has a panel row at that date; both edge endpoints must be present.
  * Interval relations (E1/E4/E5): an edge/membership is used at d iff valid_from <= d < valid_to (NaT valid_to = open); d is the formation
    date at 00:00 (conservative: filings accepted during day d are not used at d).
  * Edge arrays hold positions into common.firm_index(h). E1/E4/E5 store both directions. E1 and E5 are neighbour-sampled (CAP per firm).
    E4: w=+1 on customer->supplier, w=-1 on supplier->customer.  E5: w=+share on supplier->customer, w=-share on customer->supplier.
    CORR_KNN: src=neighbour, dst=firm, w=corr (messages flow src->dst; in-degree <= 10 per firm; one-directional).
  * CORR_KNN returns come from aux_daily.npz (R[session, firm_id]; firm id == column), sessions <= d only, window 252.
"""
from __future__ import annotations
import argparse, fcntl, json, sys, time, zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

CAP = common.HP.get("cap", 20)
BEA_TOP = 10
MIN_OVERLAP = 189
I64MAX = np.iinfo(np.int64).max
BUILDER = "p13_gnn/build_graphs.py v1"
INTERVAL_RELS = ("E1_SIC4", "E4_CUSTSUPP", "E5_BEA")


# ---------------------------------------------------------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------------------------------------------------------
def _naive_ns(s: pd.Series) -> np.ndarray:
    """datetime series (maybe tz-aware, maybe NaT) -> int64 ns UTC-naive; NaT -> I64MAX (open interval)."""
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    out = s.astype("datetime64[ns]").to_numpy().astype("int64")
    out[s.isna().to_numpy()] = I64MAX
    return out


@dataclass
class Sources:
    e1: pd.DataFrame | None = None        # cik, sic4, valid_from, valid_to
    e4: pd.DataFrame | None = None        # supplier_cik, customer_cik, valid_from, valid_to
    e5: pd.DataFrame | None = None        # data_year, sell_code, buy_code, weight, valid_from, valid_to
    sicmap: pd.DataFrame | None = None    # sic4, bea, kept
    R: np.ndarray | None = None           # (sessions, firm ids) daily returns, float32
    sessions: np.ndarray | None = None    # datetime64[ns] (S,)
    meta: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict)

    # ---- prepared arrays ----
    def e1_arrays(self):
        if "e1" not in self._cache:
            e = self.e1
            codes, uniq = pd.factorize(e["sic4"].astype(str), sort=True)
            self._cache["e1"] = (e["cik"].to_numpy(np.int64), _naive_ns(e["valid_from"]), _naive_ns(e["valid_to"]),
                                 codes.astype(np.int32), np.asarray(uniq, dtype=object))
        return self._cache["e1"]

    def e4_arrays(self):
        if "e4" not in self._cache:
            e = self.e4
            self._cache["e4"] = (e["customer_cik"].to_numpy(np.int64), e["supplier_cik"].to_numpy(np.int64),
                                 _naive_ns(e["valid_from"]), _naive_ns(e["valid_to"]))
        return self._cache["e4"]

    def bea_codes(self) -> np.ndarray:
        if "codes" not in self._cache:
            self._cache["codes"] = np.array(sorted(set(self.e5["sell_code"]) | set(self.e5["buy_code"])), dtype=object)
        return self._cache["codes"]

    def sic_to_bea(self) -> np.ndarray:
        """array over E1 sic codes (factor index) -> BEA industry index, -1 if the SIC4 is not in E5 (map `kept`)."""
        if "s2b" not in self._cache:
            _, _, _, _, uniq = self.e1_arrays()
            ix = {c: k for k, c in enumerate(self.bea_codes())}
            kept = {s: b for s, b, k in zip(self.sicmap["sic4"].astype(str), self.sicmap["bea"], self.sicmap["kept"]) if k and b in ix}
            self._cache["s2b"] = np.array([ix[kept[s]] if s in kept else -1 for s in uniq], dtype=np.int32)
        return self._cache["s2b"]

    def e5_years(self):
        """list of (data_year, valid_from_ns, valid_to_ns, W[K,K] zero diagonal, nbr per industry)."""
        if "e5y" not in self._cache:
            e = self.e5; codes = self.bea_codes(); ix = {c: k for k, c in enumerate(codes)}; K = len(codes)
            out = []
            for yr, g in e.groupby("data_year"):
                W = np.zeros((K, K))
                W[g["sell_code"].map(ix).to_numpy(), g["buy_code"].map(ix).to_numpy()] = g["weight"].to_numpy(np.float64)
                np.fill_diagonal(W, 0.0)
                vf, vt = _naive_ns(g["valid_from"])[0], _naive_ns(g["valid_to"])[0]
                out.append((int(yr), int(vf), int(vt), W, industry_neighbours(W, BEA_TOP)))
            self._cache["e5y"] = out
        return self._cache["e5y"]


def industry_neighbours(W: np.ndarray, top: int = BEA_TOP) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """W[s,c] = use share of seller s in buyer c. Q = {(s,c): c in top-`top` buyers of s} U {(s,c): s in top-`top` suppliers of c}, w>0.
    One direction per industry pair (larger share; tie -> smaller index is the seller). Per industry I: (J, share, role) with
    role +1 if I is the supplier of J, -1 if I is the customer of J."""
    K = W.shape[0]; M = np.zeros_like(W)
    for s in range(K):
        r = W[s]; t = np.argsort(-r, kind="stable")[:top]; t = t[r[t] > 0]; M[s, t] = r[t]
    for c in range(K):
        r = W[:, c]; t = np.argsort(-r, kind="stable")[:top]; t = t[r[t] > 0]; M[t, c] = r[t]
    idx = np.arange(K)
    keep = (M > 0) & ((M > M.T) | ((M == M.T) & (idx[:, None] < idx[None, :])))
    Q = np.where(keep, M, 0.0)
    out = []
    for I in range(K):
        sup = np.flatnonzero(Q[I] > 0); cus = np.flatnonzero(Q[:, I] > 0)
        J = np.concatenate([sup, cus]); sh = np.concatenate([Q[I, sup], Q[cus, I]]); role = np.concatenate([np.ones(len(sup), np.int8), -np.ones(len(cus), np.int8)])
        out.append((J.astype(np.int64), sh, role))
    return out


def load_sources(relations) -> Sources:
    s = Sources(); E = common.EDGES
    if any(r in relations for r in ("E1_SIC4", "E5_BEA")):
        s.e1 = pd.read_parquet(E / "e1_sic_membership.parquet", columns=["cik", "sic4", "valid_from", "valid_to", "source"])
        s.meta["e1_source_rows"] = {k: int(v) for k, v in s.e1["source"].value_counts().items()}
    if "E4_CUSTSUPP" in relations:
        s.e4 = pd.read_parquet(E / "e4_customer_supplier_edges.parquet", columns=["customer_cik", "supplier_cik", "valid_from", "valid_to", "customer_sic4"])
    if "E5_BEA" in relations:
        s.e5 = pd.read_parquet(E / "e5_bea_use_edges.parquet", columns=["data_year", "sell_code", "buy_code", "weight", "valid_from", "valid_to", "vintage_modified_utc"])
        s.meta["bea_vintage_modified_utc"] = str(s.e5["vintage_modified_utc"].iloc[0])
        s.meta["bea_data_years"] = [int(s.e5["data_year"].min()), int(s.e5["data_year"].max())]
        s.sicmap = pd.read_parquet(E / "e5_sic4_bea_map.parquet", columns=["sic4", "bea", "kept"])
    if "CORR_KNN" in relations:
        z = np.load(common.PANELS / "aux_daily.npz", allow_pickle=False)
        s.R = z["R"]; s.sessions = z["sessions"].astype("datetime64[ns]")
        assert s.sessions[-1] < np.datetime64("2023-01-01"), "aux_daily must end in TRAIN"
    return s


# ---------------------------------------------------------------------------------------------------------------------------
# panel context
# ---------------------------------------------------------------------------------------------------------------------------
@dataclass
class Ctx:
    h: str
    dates: np.ndarray        # datetime64[ns] (T,)
    firms: np.ndarray        # int64 (F,)   firm ids, sorted
    cik: np.ndarray          # int64 (F,)   cik per firm position, -1 if none
    pptr: np.ndarray         # (T+1,) offsets into ppos
    ppos: np.ndarray         # positions (into firms) of present firms, sorted by snapshot then firm
    pbeta: np.ndarray | None = None   # raw_beta_252 aligned with ppos (only when make_ctx(with_beta=True))

    def present(self, i: int) -> np.ndarray:
        return self.ppos[self.pptr[i]:self.pptr[i + 1]]

    def beta(self, i: int) -> np.ndarray:
        return self.pbeta[self.pptr[i]:self.pptr[i + 1]]


def make_ctx(h: str, with_beta: bool = False) -> Ctx:
    pf = common.load_panel(h, ["firm", "date", "cik"] + (["raw_beta_252"] if with_beta else []))
    dates, firms = common.date_index(h), common.firm_index(h)
    di = np.searchsorted(dates, pf["date"].to_numpy("datetime64[ns]")); assert (dates[di] == pf["date"].to_numpy("datetime64[ns]")).all()
    fp = np.searchsorted(firms, pf["firm"].to_numpy(np.int64)); assert (firms[fp] == pf["firm"].to_numpy(np.int64)).all()
    cik = np.full(len(firms), -1, np.int64)
    ck = pf["cik"].astype("Int64").fillna(-1).to_numpy(np.int64)
    ok = ck >= 0
    cik[fp[ok]] = ck[ok]
    chk = pd.DataFrame({"f": fp[ok], "c": ck[ok]}).drop_duplicates()
    assert chk["f"].is_unique, "a firm has several ciks"
    order = np.lexsort((fp, di))
    ptr = np.zeros(len(dates) + 1, np.int64); np.add.at(ptr, di + 1, 1); ptr = np.cumsum(ptr)
    pbeta = pf["raw_beta_252"].to_numpy(np.float64)[order] if with_beta else None
    return Ctx(h, dates, firms, cik, ptr, fp[order].astype(np.int64), pbeta)


def rng_for(h: str, relation: str, i: int) -> np.random.Generator:
    return np.random.default_rng([zlib.crc32(h.encode()), zlib.crc32(relation.encode()), int(i)])


def _ns(d) -> int:
    return int(np.datetime64(d, "ns").astype("int64"))


# ---------------------------------------------------------------------------------------------------------------------------
# per-snapshot edge builders (return src, dst, w as positions into ctx.firms + stats)
# ---------------------------------------------------------------------------------------------------------------------------
def sic_codes_at(ctx: Ctx, srcs: Sources, i: int) -> np.ndarray:
    """E1 SIC4 code (factor index) per present firm at snapshot i, -1 if no membership interval covers dates[i]."""
    ck, vf, vt, codes, _ = srcs.e1_arrays()
    d = _ns(ctx.dates[i]); m = (vf <= d) & (d < vt)
    cm, sm = ck[m], codes[m]
    pos = ctx.present(i); c = ctx.cik[pos]
    if len(cm) == 0:
        return np.full(len(pos), -1, np.int32)
    o = np.argsort(cm, kind="stable"); cm, sm = cm[o], sm[o]
    j = np.minimum(np.searchsorted(cm, c), len(cm) - 1)
    hit = (c >= 0) & (cm[j] == c)
    return np.where(hit, sm[j], -1).astype(np.int32)


def _uniq_edges(s, d, w, F):
    code = s.astype(np.int64) * F + d
    u, ix = np.unique(code, return_index=True)
    return (u // F).astype(np.int32), (u % F).astype(np.int32), w[ix].astype(np.float32)


def e1_snapshot(ctx: Ctx, srcs: Sources, i: int, cap: int = CAP):
    pos = ctx.present(i); F = len(ctx.firms); rng = rng_for(ctx.h, "E1_SIC4", i)
    sic = sic_codes_at(ctx, srcs, i)
    idx = np.flatnonzero(sic >= 0); order = idx[np.argsort(sic[idx], kind="stable")]
    groups = np.split(order, np.flatnonzero(np.diff(sic[order])) + 1) if len(order) else []
    S, D = [], []; max_sampled = 0
    for g in groups:
        m = len(g)
        if m < 2:
            continue
        if m <= cap + 1:
            a, b = np.nonzero(~np.eye(m, dtype=bool)); max_sampled = max(max_sampled, m - 1)
        else:
            keys = rng.random((m, m)); np.fill_diagonal(keys, -1.0)
            sel = np.argpartition(-keys, cap - 1, axis=1)[:, :cap]
            a, b = np.repeat(np.arange(m), cap), sel.ravel(); max_sampled = max(max_sampled, cap)
        S.append(pos[g[a]]); D.append(pos[g[b]])
    if not S:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32), {"max_sampled": 0}
    s, d = np.concatenate(S), np.concatenate(D)
    s2, d2, w = _uniq_edges(np.concatenate([s, d]), np.concatenate([d, s]), np.ones(2 * len(s), np.float32), F)
    return s2, d2, w, {"max_sampled": max_sampled}


def e4_snapshot(ctx: Ctx, srcs: Sources, i: int, cap: int = CAP):
    cc, sc, vf, vt = srcs.e4_arrays(); d = _ns(ctx.dates[i]); F = len(ctx.firms)
    m = (vf <= d) & (d < vt); cc, sc = cc[m], sc[m]
    pos = ctx.present(i); c = ctx.cik[pos]; ok = c >= 0
    pos, c = pos[ok], c[ok]; o = np.argsort(c, kind="stable"); pos, c = pos[o], c[o]
    if len(c) == 0 or len(cc) == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32), {}
    def look(x):
        j = np.minimum(np.searchsorted(c, x), len(c) - 1); return np.where(c[j] == x, pos[j], -1)
    pc, ps = look(cc), look(sc); k = (pc >= 0) & (ps >= 0) & (pc != ps)
    pc, ps = pc[k], ps[k]
    s, dd, w = _uniq_edges(np.concatenate([pc, ps]), np.concatenate([ps, pc]),
                           np.concatenate([np.ones(len(pc), np.float32), -np.ones(len(pc), np.float32)]), F)
    return s, dd, w, {}


def e5_snapshot(ctx: Ctx, srcs: Sources, i: int, cap: int = CAP):
    d = _ns(ctx.dates[i]); F = len(ctx.firms)
    yr = [y for y in srcs.e5_years() if y[1] <= d < y[2]]
    assert len(yr) <= 1
    empty = (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32), {"max_sampled": 0})
    if not yr:
        return empty
    nbr = yr[0][4]; pos = ctx.present(i); sic = sic_codes_at(ctx, srcs, i)
    bea = np.where(sic >= 0, srcs.sic_to_bea()[np.maximum(sic, 0)], -1)
    members = {int(I): np.flatnonzero(bea == I) for I in np.unique(bea[bea >= 0])}
    rng = rng_for(ctx.h, "E5_BEA", i); S, D, Wt = [], [], []; max_sampled = 0
    for I in sorted(members):
        mem = members[I]; J, sh, role = nbr[I]
        cands, cw, cr = [], [], []
        for j, s_, r_ in zip(J, sh, role):
            mj = members.get(int(j))
            if mj is not None:
                cands.append(mj); cw.append(np.full(len(mj), s_)); cr.append(np.full(len(mj), r_))
        if not cands:
            continue
        cand, cw, cr = np.concatenate(cands), np.concatenate(cw), np.concatenate(cr)
        n = len(cand)
        if n <= cap:
            a, b = np.repeat(np.arange(len(mem)), n), np.tile(np.arange(n), len(mem)); max_sampled = max(max_sampled, n)
        else:
            keys = np.log(rng.random((len(mem), n))) / cw[None, :]          # Efraimidis-Spirakis weighted sampling w/o replacement
            sel = np.argpartition(-keys, cap - 1, axis=1)[:, :cap]
            a, b = np.repeat(np.arange(len(mem)), cap), sel.ravel(); max_sampled = max(max_sampled, cap)
        fa, fb = pos[mem[a]], pos[cand[b]]; sh_, rl = cw[b], cr[b]
        s1 = np.where(rl > 0, fa, fb); d1 = np.where(rl > 0, fb, fa)
        S += [s1, d1]; D += [d1, s1]; Wt += [sh_, -sh_]
    if not S:
        return empty
    s, dd, w = _uniq_edges(np.concatenate(S), np.concatenate(D), np.concatenate(Wt), F)
    return s, dd, w, {"max_sampled": max_sampled}


def corr_snapshot(ctx: Ctx, srcs: Sources, i: int, k: int | None = None, window: int | None = None, min_overlap: int = MIN_OVERLAP,
                  R: np.ndarray | None = None):
    """kNN on market-residualised daily returns of the last `window` sessions <= dates[i]. `R` may override srcs.R (used by the check
    to prove that rows after dates[i] are never touched)."""
    k = common.HP["knn_k"] if k is None else k; window = common.HP["corr_window"] if window is None else window
    R = srcs.R if R is None else R
    pos = ctx.present(i); F = len(ctx.firms); fid = ctx.firms[pos]
    end = int(np.searchsorted(srcs.sessions, ctx.dates[i], side="right")); lo = max(0, end - window)
    last = srcs.sessions[end - 1] if end > 0 else None
    X = np.asarray(R[lo:end][:, fid], dtype=np.float32)
    V = np.isfinite(X); n = len(pos)
    empty = (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32), {"last_session": last})
    if X.shape[0] < min_overlap or n < 2:
        return empty
    X = np.where(V, X, 0.0)
    cnt = V.sum(1); mu = np.where(cnt > 0, X.sum(1) / np.maximum(cnt, 1), 0.0)       # cross-sectional mean of present firms per session
    Z = np.where(V, X - mu[:, None], 0.0).astype(np.float32)
    cm = Z.sum(0) / np.maximum(V.sum(0), 1); Z = np.where(V, Z - cm[None, :], 0.0).astype(np.float32)   # centre per firm (stability only)
    Vf = V.astype(np.float32)
    N = Vf.T @ Vf; SX = Z.T @ Vf; SXX = (Z * Z).T @ Vf; SXY = Z.T @ Z
    with np.errstate(divide="ignore", invalid="ignore"):
        Nn = np.maximum(N, 1.0)
        cov = SXY - SX * SX.T / Nn
        vx = SXX - SX * SX / Nn
        C = cov / np.sqrt(vx * vx.T)
    bad = ~np.isfinite(C) | (N < min_overlap) | (vx <= 1e-12) | (vx.T <= 1e-12)
    C[bad] = -np.inf; np.fill_diagonal(C, -np.inf)
    kk = min(k, n - 1)
    nb = np.argpartition(-C, kk - 1, axis=1)[:, :kk]
    cv = np.take_along_axis(C, nb, 1)
    o = np.argsort(-cv, axis=1, kind="stable"); nb, cv = np.take_along_axis(nb, o, 1), np.take_along_axis(cv, o, 1)
    keep = np.isfinite(cv) & (cv > 0)
    rows = np.repeat(np.arange(n), kk).reshape(n, kk)
    return pos[nb[keep]].astype(np.int32), pos[rows[keep]].astype(np.int32), cv[keep].astype(np.float32), {"last_session": last}   # src = neighbour, dst = firm


SNAP_FN = {"E1_SIC4": e1_snapshot, "E4_CUSTSUPP": e4_snapshot, "E5_BEA": e5_snapshot, "CORR_KNN": corr_snapshot}
PIT_RULE = {
    "E1_SIC4": "firm pair linked iff both firms' CIKs have an e1_sic_membership row with valid_from <= date < valid_to (NaT open) and the same SIC4; "
               "strictly SIC4 (no SIC2 fallback); both endpoints present in the panel at date",
    "E4_CUSTSUPP": "e4_customer_supplier_edges row used iff valid_from <= date < valid_to (NaT open); customer and supplier firms (by CIK) both present at date",
    "E5_BEA": "BEA use-table data year Y with valid_from <= date < valid_to (Y = year(date)-2); firm SIC4 at date (E1 interval rule) -> e5_sic4_bea_map (kept) "
              "-> BEA industry; industry pairs = top-%d buyers of each seller U top-%d suppliers of each buyer (one direction per pair, larger share); "
              "firms sampled from neighbouring industries with prob ~ share" % (BEA_TOP, BEA_TOP),
    "CORR_KNN": "daily returns R[session<=date], last %d sessions, minus the daily cross-sectional mean of the present firms, pairwise-complete correlation "
                "with >=%d overlapping sessions, top-%d positive-correlation neighbours per firm",
}


def build_relation(h: str, relation: str, ctx: Ctx, srcs: Sources, *, cap: int = CAP, progress: bool = False, **kw) -> dict:
    fn = SNAP_FN[relation]; T = len(ctx.dates); t0 = time.time()
    S, D, W = [], [], []; counts = np.zeros(T, np.int64); max_sampled = 0
    for i in range(T):
        out = fn(ctx, srcs, i, **({"cap": cap} if relation in ("E1_SIC4", "E4_CUSTSUPP", "E5_BEA") else kw))
        s, d, w, st = out; S.append(s); D.append(d); W.append(w); counts[i] = len(s)
        max_sampled = max(max_sampled, st.get("max_sampled", 0))
        if progress and (i + 1) % 100 == 0:
            print(f"  {h} {relation}: {i + 1}/{T} snapshots, {counts.sum()} edges, {time.time() - t0:.0f}s", flush=True)
    return _pack(h, relation, ctx, srcs, S, D, W, counts, max_sampled, t0, cap, kw)


def _pack(h, relation, ctx, srcs, S, D, W, counts, max_sampled, t0, cap, kw) -> dict:
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    src = np.concatenate(S) if S else np.zeros(0, np.int32); dst = np.concatenate(D) if D else np.zeros(0, np.int32); w = np.concatenate(W) if W else np.zeros(0, np.float32)
    yrs = pd.DatetimeIndex(ctx.dates).year
    per_year = {int(y): int(counts[yrs == y].sum()) for y in sorted(set(yrs))}
    npres = np.diff(ctx.pptr)
    params = {"CAP": cap, "knn_k": kw.get("k") or common.HP["knn_k"], "corr_window": kw.get("window") or common.HP["corr_window"],
              "min_overlap": kw.get("min_overlap", MIN_OVERLAP), "bea_top_industries": BEA_TOP}
    rule = PIT_RULE[relation]
    if relation == "CORR_KNN":
        rule = rule % (params["corr_window"], params["min_overlap"], params["knn_k"])
    sign = {"E4_CUSTSUPP": "w=+1: src=customer,dst=supplier; w=-1: src=supplier,dst=customer",
            "E5_BEA": "w=+share: src=supplier,dst=customer; w=-share: src=customer,dst=supplier (|w| = BEA use share)",
            "E1_SIC4": "w=1.0, symmetric (both directions stored; each firm samples <=CAP neighbours, so degree can exceed CAP)",
            "CORR_KNN": "w=correlation of residual returns; messages flow src->dst: src=neighbour, dst=firm, so each firm aggregates ITS OWN top-k (in-degree <= 10), not symmetrised"}[relation]
    meta = {"builder": BUILDER, "relation": relation, "horizon": h, "parameters": params, "pit_rule": rule, "direction_and_sign": sign,
            "edges_per_year": per_year, "n_edges": int(len(src)),
            "firms_present_per_snapshot": {"min": int(npres.min()), "median": float(np.median(npres)), "max": int(npres.max())},
            "edges_per_snapshot": {"min": int(counts.min()), "median": float(np.median(counts)), "max": int(counts.max())},
            "max_sampled_per_firm": int(max_sampled), "wall_seconds": round(time.time() - t0, 1), **srcs.meta}
    return {"dates": ctx.dates, "firms": ctx.firms, "ptr": ptr, "src": src, "dst": dst, "w": w, "meta": meta}


def write_graph(h: str, relation: str, g: dict, rewire_seed: int | None = None) -> Path:
    return common.save_graph(h, relation, g["dates"], g["firms"], g["ptr"], g["src"], g["dst"], g["w"], g["meta"], rewire_seed=rewire_seed)


# ---------------------------------------------------------------------------------------------------------------------------
# persistent rewiring nulls (A6): the edge TABLES are rewired once per seed and the snapshots built from them as for the real graph
# ---------------------------------------------------------------------------------------------------------------------------
def _rng(seed: int, *tags) -> np.random.Generator:
    return np.random.default_rng([int(seed)] + [zlib.crc32(str(t).encode()) for t in tags])


E1_WIN = (_ns("2004-01-01"), _ns("2023-01-01"))      # panel window; timelines are compared inside it only


def sic2_timeline_keys(e1: pd.DataFrame, ciks) -> dict:
    """cik -> exact SIC2 timeline key: the ordered runs (SIC2, start, end) of its membership rows (consecutive rows with the same SIC2 and no gap are
    merged, so a SIC4 change inside a SIC2 is not a break), clipped to the panel window. Two ciks with equal keys have the same SIC2 at every date."""
    lo, hi = E1_WIN
    e = e1[e1["cik"].isin(np.asarray(list(ciks), np.int64))].assign(_s2=lambda x: x["sic4"].astype(str).str[:2])
    e = e.assign(_vf=_naive_ns(e["valid_from"]), _vt=_naive_ns(e["valid_to"])).sort_values(["cik", "_vf"], kind="stable")
    out = {}
    for c, g in e.groupby("cik", sort=True):
        runs = []
        for s2, vf, vt in zip(g["_s2"], g["_vf"], g["_vt"]):
            if runs and runs[-1][0] == s2 and runs[-1][2] == vf:
                runs[-1][2] = vt
            else:
                runs.append([s2, vf, vt])
        out[int(c)] = tuple((r[0], max(r[1], lo), min(r[2], hi)) for r in runs if r[2] > lo and r[1] < hi)
    return out


def rewire_e1(e1: pd.DataFrame, ciks, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Permute whole firm SIC4 timelines among ciks with the IDENTICAL SIC2 timeline (ordered runs of SIC2 with their dates, clipped to the panel window);
    one fixed permutation per timeline group, applied to every row of the firm, so the label is persistent in time and every firm-date keeps its real SIC2.
    A firm with a unique SIC2 timeline keeps its label. Cliques stay cliques; only ciks in `ciks` (the panel firms) take part."""
    keys = sic2_timeline_keys(e1, ciks); groups: dict = {}
    for c, k_ in keys.items():
        groups.setdefault(k_, []).append(c)
    donors, owners = [], []; n_fixed = n_unique = 0
    for k_ in sorted(groups, key=repr):
        mem = np.sort(np.asarray(groups[k_], np.int64)); p = rng.permutation(len(mem))
        donors.append(mem[p]); owners.append(mem); n_fixed += int((mem[p] == mem).sum()); n_unique += int(len(mem) == 1)   # cik mem[j] takes the timeline of mem[p[j]]
    donors, owners = (np.concatenate(donors), np.concatenate(owners)) if donors else (np.zeros(0, np.int64),) * 2
    o = np.argsort(donors); donors, owners = donors[o], owners[o]
    ck = e1["cik"].to_numpy(np.int64); j = np.minimum(np.searchsorted(donors, ck), max(len(donors) - 1, 0))
    hit = (donors[j] == ck) if len(donors) else np.zeros(len(ck), bool)
    out = e1.copy(); out["cik"] = np.where(hit, owners[j] if len(donors) else ck, ck).astype(np.int64)
    n = int(len(donors))
    return out, {"method": "firm SIC4 timelines permuted among ciks with the identical SIC2 timeline (ordered runs of SIC2 with start/end, clipped to 2004-2023), "
                           "one permutation per timeline group, all rows of a firm move together; unique-timeline firms keep their label",
                 "n_permuted_ciks": n, "n_fixed_points": n_fixed, "n_unique_timeline": n_unique, "n_timeline_groups": len(groups),
                 "share_label_kept": round(n_fixed / max(n, 1), 4), "share_unique_timeline": round(n_unique / max(n, 1), 4)}


def e1_sic2_crossings(ctx: Ctx, real: Sources, rw: Sources, snaps=None) -> tuple[int, int]:
    """(# firm-dates whose rewired SIC2 differs from the real SIC2 (missing counts as a difference), # firm-dates) over the given snapshots (default all),
    measured on the real and rewired E1 tables themselves, not on the grouping key."""
    _, _, _, _, ur = real.e1_arrays(); _, _, _, _, uw = rw.e1_arrays(); tot = cross = 0
    s2r_tab = np.array([str(x)[:2] for x in ur] + [""], dtype=object); s2w_tab = np.array([str(x)[:2] for x in uw] + [""], dtype=object)
    for i in (range(len(ctx.dates)) if snaps is None else snaps):
        sr, sw = sic_codes_at(ctx, real, int(i)), sic_codes_at(ctx, rw, int(i))
        tot += len(sr); cross += int((s2r_tab[sr] != s2w_tab[sw]).sum())            # code -1 -> last entry ""
    return cross, tot


def _strata_labels(sic4: np.ndarray) -> np.ndarray:
    """study-15 fallback: SIC4 stratum if it has >= 2 edges, else SIC3, else SIC2."""
    sic4 = np.asarray(sic4, dtype=object); out = np.empty(len(sic4), dtype=object)
    cnt = [pd.Series([x[:n] for x in sic4]).value_counts().to_dict() for n in (4, 3, 2)]
    for i, x in enumerate(sic4):
        out[i] = next((x[:n] for n, c in zip((4, 3), cnt) if c[x[:n]] >= 2), x[:2])
    return out


def rewire_e4(e4: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Study-15 N3 on the unique (customer, supplier) pairs: equity_graph.rewire_stratified (degree-preserving double-edge swaps within
    customer-SIC4 strata, SIC3 then SIC2 fallback). The swapped customer is applied to EVERY table row of the pair, so rows keep their
    validity intervals and the swap persists in time."""
    from eqxs.equity_graph import rewire_stratified
    pr = e4.assign(_s=e4["customer_sic4"].astype(object).where(e4["customer_sic4"].notna(), "????").astype(str))
    u = pr.drop_duplicates(["customer_cik", "supplier_cik"]).sort_values(["supplier_cik", "customer_cik"], kind="stable")
    cust, supp = u["customer_cik"].to_numpy(np.int64), u["supplier_cik"].to_numpy(np.int64)
    new, share = rewire_stratified(cust, supp, _strata_labels(u["_s"].to_numpy()), rng)
    m = dict(zip(zip(cust.tolist(), supp.tolist()), new.tolist()))
    out = e4.copy()
    out["customer_cik"] = [m[(c, s_)] for c, s_ in zip(e4["customer_cik"].to_numpy(np.int64).tolist(), e4["supplier_cik"].to_numpy(np.int64).tolist())]
    return out, {"method": "equity_graph.rewire_stratified on unique (customer, supplier) pairs by customer_sic4 (SIC3/SIC2 fallback), applied to all rows of a pair",
                 "table_pair_swap_share": round(float(share), 4), "table_row_swap_share": round(float((out["customer_cik"].to_numpy() != e4["customer_cik"].to_numpy()).mean()), 4),
                 "n_pairs": int(len(u))}


def rewire_e5(e5: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    """Study-15 N3: equity_graph.permute_within_columns once per data year on the use table (each buyer keeps its share distribution)."""
    from eqxs.equity_graph import permute_within_columns
    codes = sorted(set(e5["sell_code"]) | set(e5["buy_code"])); ix = {c: k for k, c in enumerate(codes)}; K = len(codes); rows = []
    for yr, g in e5.groupby("data_year"):
        W = np.zeros((K, K)); W[g["sell_code"].map(ix).to_numpy(), g["buy_code"].map(ix).to_numpy()] = g["weight"].to_numpy(np.float64)
        np.fill_diagonal(W, 0.0)
        Wp = permute_within_columns(W, _rng(seed, "E5_BEA", int(yr))); a, b = np.nonzero(Wp > 0)
        rows.append(pd.DataFrame({"data_year": int(yr), "sell_code": np.array(codes, dtype=object)[a], "buy_code": np.array(codes, dtype=object)[b],
                                  "weight": Wp[a, b], "valid_from": g["valid_from"].iloc[0], "valid_to": g["valid_to"].iloc[0]}))
    return pd.concat(rows, ignore_index=True), {"method": "equity_graph.permute_within_columns once per data year (use-table columns = buyers)"}


def cik_universe() -> set:
    out: set = set()
    for h in common.HORIZONS:
        if common.panel_path(h).exists():
            out |= set(pd.read_parquet(common.panel_path(h), columns=["cik"])["cik"].dropna().astype(np.int64).tolist())
    return out


def rewired_sources(srcs: Sources, relation: str, seed: int, universe) -> Sources:
    """Sources whose edge table for `relation` is the persistent rewired one (CORR_KNN: unchanged, rewired in rewire_corr)."""
    if relation == "E1_SIC4":
        t, info = rewire_e1(srcs.e1, universe, _rng(seed, "E1_SIC4")); return replace(srcs, e1=t, _cache={}, meta={**srcs.meta, "rewire_table": info})
    if relation == "E4_CUSTSUPP":
        t, info = rewire_e4(srcs.e4, _rng(seed, "E4_CUSTSUPP")); return replace(srcs, e4=t, _cache={}, meta={**srcs.meta, "rewire_table": info})
    if relation == "E5_BEA":
        t, info = rewire_e5(srcs.e5, seed); return replace(srcs, e5=t, _cache={}, meta={**srcs.meta, "rewire_table": info})
    return srcs


def sic2_division(s2: str) -> int:
    try:
        n = int(s2)
    except ValueError:
        return 10
    for k_, (lo, hi) in enumerate(((0, 9), (10, 14), (15, 17), (20, 39), (40, 49), (50, 51), (52, 59), (60, 67), (70, 89), (91, 99))):
        if lo <= n <= hi:
            return k_
    return 10


def corr_strata(ctx: Ctx, srcs: Sources, year_snaps: np.ndarray):
    """Per firm position, fixed for a calendar year from the firm's first present snapshot of the year: stratum key (SIC2 x beta tercile;
    tercile cut points from the cross-section at the year's first snapshot), SIC2 id and SIC division (-1 = unknown SIC).
    Returns (key[F] int64, -1 = absent all year; first_i[F]; sid[F]; div[F])."""
    F = len(ctx.firms); key = np.full(F, -1, np.int64); first_i = np.full(F, -1, np.int64); sid_f = np.full(F, -1, np.int64); div_f = np.full(F, -1, np.int64)
    _, _, _, _, uniq = srcs.e1_arrays(); s2str = np.array([str(x)[:2] for x in uniq], dtype=object)
    s2 = pd.factorize(s2str)[0].astype(np.int64); dv = np.array([sic2_division(x) for x in s2str], dtype=np.int64)
    b0 = ctx.beta(int(year_snaps[0])); b0 = b0[np.isfinite(b0)]
    cuts = np.quantile(b0, [1 / 3, 2 / 3]) if len(b0) else np.array([np.inf, np.inf])
    for i in year_snaps[::-1]:
        pos = ctx.present(int(i)); first_i[pos] = i
        sic = sic_codes_at(ctx, srcs, int(i)); b = ctx.beta(int(i))
        terc = np.where(np.isfinite(b), (b > cuts[0]).astype(np.int64) + (b > cuts[1]), -1)
        sid = np.where(sic >= 0, s2[np.maximum(sic, 0)], -1)
        key[pos] = (sid + 1) * 4 + (terc + 1); sid_f[pos] = sid; div_f[pos] = np.where(sic >= 0, dv[np.maximum(sic, 0)], -1)
    return key, first_i, sid_f, div_f


def rewire_corr(h: str, ctx: Ctx, srcs: Sources, real: dict, seed: int, k: int | None = None, progress: bool = False) -> dict:
    """Each firm-year has ONE persistent random ranking of the other active firms, ordered by stratum level (SIC2 x beta tercile, then SIC2, then
    SIC division, then everyone) and, inside a level, by a random key. At every snapshot the firm's in-neighbours are the best-ranked PRESENT firms,
    as many as its REAL in-degree there (so the refill from coarser strata is itself persistent); the firm's real sorted weights are reassigned in
    order. Asserts rewired in-degree == real in-degree for every firm at every snapshot."""
    assert ctx.pbeta is not None, "make_ctx(with_beta=True)"
    k = common.HP["knn_k"] if k is None else k; t0 = time.time(); T, F = len(ctx.dates), len(ctx.firms)
    yrs = pd.DatetimeIndex(ctx.dates).year.to_numpy(); S, D, W = [], [], []; counts = np.zeros(T, np.int64)
    lvl_edges = np.zeros(5, np.int64)
    for y in sorted(set(yrs.tolist())):
        idx = np.flatnonzero(yrs == y); key, first_i, sid, dvs = corr_strata(ctx, srcs, idx)
        rng = _rng(seed, "CORR_KNN", h, y); act = np.flatnonzero(key >= 0); A = len(act)
        RK = np.full((F, max(A - 1, 1)), -1, np.int32)
        ka, sa, da = key[act], sid[act], dvs[act]
        for a_, f in enumerate(act):
            lvl = np.full(A, 4, np.int8); lvl[da == da[a_]] = 3; lvl[sa == sa[a_]] = 2; lvl[ka == ka[a_]] = 1; lvl[a_] = 9
            RK[f, :A - 1] = act[np.lexsort((rng.random(A), lvl))[:A - 1]]
        P = np.zeros(F, bool)
        for i in idx:
            a, b = int(real["ptr"][i]), int(real["ptr"][i + 1])
            if a == b:
                S.append(np.zeros(0, np.int32)); D.append(np.zeros(0, np.int32)); W.append(np.zeros(0, np.float32)); continue
            d_r, w_r = real["dst"][a:b], real["w"][a:b]               # the firm is dst
            us, st, cn = np.unique(d_r, return_index=True, return_counts=True)
            kk = int(cn.max()); jj = np.arange(kk)[None, :]; Wm = np.zeros((len(us), kk), np.float32)
            Wm[jj < cn[:, None]] = w_r[(st[:, None] + jj)[jj < cn[:, None]]]
            P[:] = False; P[ctx.present(int(i))] = True
            Rk = RK[us]; valid = (Rk >= 0) & P[np.maximum(Rk, 0)]
            cs = np.cumsum(valid, axis=1, dtype=np.int32); keep = valid & (cs <= cn[:, None])
            assert (keep.sum(1) == cn).all(), f"rewired in-degree != real in-degree at snapshot {i}"
            r, c = np.nonzero(keep)
            src_ = Rk[r, c]; dst_ = us[r]
            lv = np.where(key[src_] == key[dst_], 1, np.where(sid[src_] == sid[dst_], 2, np.where(dvs[src_] == dvs[dst_], 3, 4)))
            lvl_edges += np.bincount(lv, minlength=5)
            S.append(src_.astype(np.int32)); D.append(dst_.astype(np.int32)); W.append(Wm[r, cs[r, c] - 1])   # src = replacement neighbour, dst = firm
            counts[i] = len(src_)
        if progress:
            print(f"  {h} CORR_KNN rw{seed}: year {y} done, {counts.sum()} edges, {time.time() - t0:.0f}s", flush=True)
    g = _pack(h, "CORR_KNN", ctx, srcs, S, D, W, counts, 0, t0, CAP, {"k": k, "window": real["meta"]["parameters"]["corr_window"],
                                                                      "min_overlap": real["meta"]["parameters"]["min_overlap"]})
    tot = max(int(lvl_edges.sum()), 1)
    g["meta"]["edge_share_by_stratum_level"] = {n_: round(float(lvl_edges[j] / tot), 4) for j, n_ in ((1, "sic2_x_beta_tercile"), (2, "sic2"), (3, "division"), (4, "any"))}
    g["meta"]["in_degree_equals_real"] = True
    return g


def swap_share(real: dict, rw: dict, F: int) -> float:
    """Mean over snapshots with real edges of the fraction of real (src, dst) pairs absent from the rewired snapshot."""
    out = []
    for i in range(len(real["ptr"]) - 1):
        a, b = int(real["ptr"][i]), int(real["ptr"][i + 1])
        if a == b:
            continue
        c, d = int(rw["ptr"][i]), int(rw["ptr"][i + 1])
        rc = real["src"][a:b].astype(np.int64) * F + real["dst"][a:b]; wc = rw["src"][c:d].astype(np.int64) * F + rw["dst"][c:d]
        out.append(1.0 - float(np.isin(rc, wc).mean()))
    return float(np.mean(out)) if out else float("nan")


def build_rewired(h: str, relation: str, seed: int, ctx: Ctx, srcs: Sources, universe, real: dict | None = None, progress: bool = False):
    """Returns (graph dict with rewire meta, the Sources used for the snapshots)."""
    t0 = time.time(); real = common.load_graph(h, relation) if real is None else real
    srcs_rw = rewired_sources(srcs, relation, seed, universe)
    if relation == "E1_SIC4":
        cross, tot = e1_sic2_crossings(ctx, srcs, srcs_rw); assert cross == 0, f"E1 rewiring crosses SIC2 on {cross}/{tot} firm-dates"
        srcs_rw.meta["rewire_table"]["sic2_crossings_firm_dates"] = cross; srcs_rw.meta["rewire_table"]["firm_dates_checked"] = tot
    g = rewire_corr(h, ctx, srcs, real, seed, progress=progress) if relation == "CORR_KNN" else build_relation(h, relation, ctx, srcs_rw, progress=progress)
    method_txt = "per firm-year persistent ranking of other firms by stratum level (SIC2 x raw_beta_252 tercile at the first present snapshot of the year, then SIC2, " \
                 "division, all) and a random key within level; at each snapshot the best-ranked present firms, as many as the firm's real in-degree; the firm's " \
                 "real sorted weights reassigned in order; rewired in-degree == real in-degree asserted" if relation == "CORR_KNN" else None
    g["meta"]["rewire"] = {"seed": seed, "method": method_txt or srcs_rw.meta["rewire_table"]["method"], "table_level": None if relation == "CORR_KNN" else srcs_rw.meta["rewire_table"], "swap_share": round(swap_share(real, g, len(ctx.firms)), 4),
                           "swap_share_level": "snapshot (mean over snapshots of the fraction of real (src,dst) pairs absent from the rewired snapshot)", "source_graph_edges": int(real["meta"]["n_edges"]),
                           "persistent": "tables/neighbour sets drawn once per seed (CORR: once per firm-year), never per snapshot", "rewire_wall_seconds": round(time.time() - t0, 1)}
    return g, srcs_rw


def check_rewired(h: str, relation: str, seed: int, ctx: Ctx, srcs_rw: Sources, g: dict | None = None, srcs_real: Sources | None = None) -> dict:
    g = common.load_graph(h, relation, seed) if g is None else g
    F, T = len(ctx.firms), len(ctx.dates)
    if relation != "CORR_KNN":
        res = check_graph(h, relation, ctx, srcs_rw, g=g)
    else:
        res = {"relation": relation, "horizon": h}
        _check_basic(g, ctx); res["alignment"] = "PASS"; res["endpoints_present"] = "PASS"
        real = common.load_graph(h, "CORR_KNN"); snap = np.repeat(np.arange(T, dtype=np.int64), np.diff(g["ptr"])); rsnap = np.repeat(np.arange(T, dtype=np.int64), np.diff(real["ptr"]))
        dg, dr = np.bincount(snap * F + g["dst"], minlength=T * F), np.bincount(rsnap * F + real["dst"], minlength=T * F)
        assert (dg == dr).all(), "rewired in-degree != real in-degree"; assert dg.max() <= common.HP["knn_k"]
        res["in_degree_equals_real"] = f"PASS (every firm at every snapshot; max {int(dg.max())} <= {common.HP['knn_k']}; {int((dg > 0).sum())} firm-snapshots)"
        # refill rule on 4 spread snapshots: edges inside the SIC2 x tercile stratum (SIC2) == min(in-degree, #present members)
        yrs_all = pd.DatetimeIndex(ctx.dates).year.to_numpy(); nz = np.flatnonzero(np.diff(g["ptr"]) > 0); rs = nz[np.linspace(0, len(nz) - 1, min(4, len(nz))).astype(int)]
        for i in rs:
            i = int(i); key, _, sid, _ = corr_strata(ctx, srcs_real or srcs_rw, np.flatnonzero(yrs_all == yrs_all[i]))
            a, b = int(g["ptr"][i]), int(g["ptr"][i + 1]); s_, d_ = g["src"][a:b].astype(np.int64), g["dst"][a:b].astype(np.int64)
            pres = ctx.present(i); n_in = np.bincount(d_, minlength=F)
            for lab, arr in (("stratum", key), ("sic2", sid)):
                c_pool = np.bincount(arr[pres] + 1, minlength=int(arr.max()) + 3)       # present members per label
                same = np.bincount(d_[arr[s_] == arr[d_]], minlength=F)
                for f in np.unique(d_):
                    expect = min(int(n_in[f]), int(c_pool[arr[f] + 1]) - 1)
                    assert same[f] == expect, f"refill rule violated ({lab}) firm {f} snapshot {i}: {same[f]} != {expect}"
        res["refill_rule"] = "PASS (in-stratum and in-SIC2 edge counts == min(in-degree, present members) on 4 snapshots)"
        res["edge_share_by_stratum_level"] = g["meta"].get("edge_share_by_stratum_level")
        yr = np.repeat(yrs_all, np.diff(g["ptr"])).astype(np.int64)
        dist = lambda gg, yy, ss: np.bincount(np.unique(np.unique((yy * F + gg["dst"]) * F + gg["src"]) // F, return_inverse=True)[1]).mean() if len(ss) else float("nan")
        ryr = np.repeat(yrs_all, np.diff(real["ptr"])).astype(np.int64)
        res["distinct_neighbours_per_firm_year_mean"] = {"rewired": round(float(dist(g, yr, g["src"])), 2), "real": round(float(dist(real, ryr, real["src"])), 2)}
    if relation == "E1_SIC4" and srcs_real is not None:
        cross, tot = e1_sic2_crossings(ctx, srcs_real, srcs_rw); assert cross == 0, f"rewired E1 SIC2 != real E1 SIC2 on {cross}/{tot} firm-dates"
        res["sic2_crossing"] = f"PASS (rewired SIC2 == real SIC2 on all {tot} present firm-dates, all snapshots, measured against the real E1 table)"
        res["cross_sic2_firm_date_share"] = 0.0
    pair = g["src"].astype(np.int64) * F + g["dst"]; snapi = np.repeat(np.arange(T, dtype=np.int64), np.diff(g["ptr"]))
    assert (g["src"] != g["dst"]).all(), "self loop"
    assert len(np.unique(snapi * F * F + pair)) == len(pair), "duplicate edge in a snapshot"
    res["no_selfloops_no_duplicates"] = "PASS"
    return res


# ---------------------------------------------------------------------------------------------------------------------------
# checks on real output
# ---------------------------------------------------------------------------------------------------------------------------
def _edge_snapshots(g, idx):
    return np.searchsorted(g["ptr"], idx, side="right") - 1


def _check_basic(g: dict, ctx: Ctx) -> None:
    T, F = len(ctx.dates), len(ctx.firms)
    assert (g["dates"] == ctx.dates).all() and (g["firms"] == ctx.firms).all() and len(g["ptr"]) == T + 1 and g["ptr"][-1] == len(g["src"])
    assert (np.diff(g["ptr"]) >= 0).all() and (len(g["src"]) == 0 or (g["src"].min() >= 0 and g["dst"].max() < F))
    present = np.zeros(F, bool); bad = 0
    for i in range(T):                    # endpoints present at every snapshot (all edges)
        a, b = int(g["ptr"][i]), int(g["ptr"][i + 1])
        if a == b:
            continue
        present[:] = False; present[ctx.present(i)] = True
        bad += int((~present[g["src"][a:b]]).sum() + (~present[g["dst"][a:b]]).sum())
    assert bad == 0, f"{bad} edge endpoints not present"


def check_graph(h: str, relation: str, ctx: Ctx, srcs: Sources, g: dict | None = None, n_sample: int = 200, seed: int = 0,
                cap: int = CAP) -> dict:
    g = common.load_graph(h, relation) if g is None else g
    T, F = len(ctx.dates), len(ctx.firms); res: dict = {"relation": relation, "horizon": h}
    _check_basic(g, ctx); res["alignment"] = "PASS"
    res["endpoints_present"] = "PASS (all %d edges)" % len(g["src"])
    E = len(g["src"]); rng = np.random.default_rng(seed)
    if E == 0:
        res["note"] = "no edges"; return res
    pick = rng.choice(E, size=min(n_sample, E), replace=False); sn = _edge_snapshots(g, pick)
    # recompute 4 spread snapshots with edges: must reproduce exactly
    nz = np.flatnonzero(np.diff(g["ptr"]) > 0); rs = nz[np.linspace(0, len(nz) - 1, min(4, len(nz))).astype(int)]
    fn = SNAP_FN[relation]
    if relation in INTERVAL_RELS:
        for i in rs:
            s, d, w, st = fn(ctx, srcs, int(i), cap=cap)
            a, b = int(g["ptr"][i]), int(g["ptr"][i + 1])
            assert np.array_equal(s, g["src"][a:b]) and np.array_equal(d, g["dst"][a:b]) and np.array_equal(w, g["w"][a:b]), f"recompute mismatch at {i}"
            if relation != "E4_CUSTSUPP":
                assert st["max_sampled"] <= cap, "cap violated"
        res["recompute_reproduces"] = "PASS (snapshots %s)" % [str(ctx.dates[i])[:10] for i in rs]
        res["cap"] = f"PASS (max sampled per firm {g['meta'].get('max_sampled_per_firm')} <= {cap})" if relation != "E4_CUSTSUPP" else "n/a (E4 uncapped)"
        if relation != "E4_CUSTSUPP":
            assert g["meta"]["max_sampled_per_firm"] <= cap
        _check_intervals(relation, ctx, srcs, g, pick, sn); res["interval_rule"] = "PASS (%d sampled snapshot-edges verified against the source table)" % len(pick)
    else:
        par = g["meta"]["parameters"]
        for i in rs[: min(3, len(rs))]:
            i = int(i); a, b = int(g["ptr"][i]), int(g["ptr"][i + 1]); d0 = ctx.dates[i]
            end = int(np.searchsorted(srcs.sessions, d0, side="right"))
            for label, fill in (("zeroed", lambda x: np.zeros_like(x)), ("garbage", lambda x: rng.normal(5.0, 50.0, x.shape).astype(x.dtype))):
                R2 = srcs.R.copy(); R2[end:] = fill(R2[end:])
                s, d, w, st = corr_snapshot(ctx, srcs, i, R=R2, k=par["knn_k"], window=par["corr_window"], min_overlap=par["min_overlap"])
                assert st["last_session"] <= d0, "session after the formation date used"
                assert np.array_equal(s, g["src"][a:b]) and np.array_equal(d, g["dst"][a:b]) and np.allclose(w, g["w"][a:b], atol=0), \
                    f"future-{label} changes/does not reproduce edges at {i}"
        res["no_future"] = "PASS (edges identical with sessions > date zeroed / randomised; last session used <= date) on %d snapshots" % min(3, len(rs))
        k = g["meta"]["parameters"]["knn_k"]
        snap = np.repeat(np.arange(T, dtype=np.int64), np.diff(g["ptr"]))
        deg = np.bincount(snap * F + g["dst"], minlength=1); assert deg.max() <= k, "kNN in-degree > k"
        res["cap"] = f"PASS (max in-degree per firm per snapshot {int(deg.max())} <= {k}; src = neighbour, dst = firm)"
    return res


def _members(srcs, cik, d):
    ck, vf, vt, codes, uniq = srcs.e1_arrays()
    m = (ck == cik) & (vf <= d) & (d < vt)
    return set(uniq[codes[m]])


def _check_intervals(relation, ctx, srcs, g, pick, sn):
    for e, i in zip(pick, sn):
        d = _ns(ctx.dates[i]); x, y = int(g["src"][e]), int(g["dst"][e]); w = float(g["w"][e]); cx, cy = int(ctx.cik[x]), int(ctx.cik[y])
        assert cx >= 0 and cy >= 0 and cx != cy
        if relation == "E1_SIC4":
            sx, sy = _members(srcs, cx, d), _members(srcs, cy, d)
            assert sx and sy and (sx & sy), f"E1 edge {x}->{y} at {i}: sic {sx} vs {sy}"
        elif relation == "E4_CUSTSUPP":
            cust, supp = (cx, cy) if w > 0 else (cy, cx); assert abs(abs(w) - 1) < 1e-6
            e4 = srcs.e4
            m = (e4["customer_cik"] == cust) & (e4["supplier_cik"] == supp) & (_naive_ns(e4["valid_from"]) <= d) & (d < _naive_ns(e4["valid_to"]))
            assert m.any(), f"E4 edge {cust}->{supp} not valid at {ctx.dates[i]}"
        else:
            sx, sy = _members(srcs, cx, d), _members(srcs, cy, d)
            if "sm" not in srcs._cache:
                srcs._cache["sm"] = {str(a): b for a, b, k in zip(srcs.sicmap["sic4"], srcs.sicmap["bea"], srcs.sicmap["kept"]) if k}
            sm = srcs._cache["sm"]
            bx = {sm[s] for s in sx if s in sm}; by = {sm[s] for s in sy if s in sm}
            assert bx and by, "E5 endpoint without BEA industry"
            sell, buy = (bx, by) if w > 0 else (by, bx)
            e5 = srcs.e5
            m = e5["sell_code"].isin(sell) & e5["buy_code"].isin(buy) & (_naive_ns(e5["valid_from"]) <= d) & (d < _naive_ns(e5["valid_to"])) \
                & np.isclose(e5["weight"], abs(w), atol=1e-6, rtol=1e-5)
            assert m.any(), f"E5 edge {x}->{y} w={w} not in the use table at {ctx.dates[i]}"


# ---------------------------------------------------------------------------------------------------------------------------
# report + CLI
# ---------------------------------------------------------------------------------------------------------------------------
def update_report(entry: dict, h: str, relation: str) -> None:
    common.GRAPHS.mkdir(parents=True, exist_ok=True)
    p = common.GRAPHS / "build_report.json"
    with open(common.GRAPHS / ".build_report.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        rep = json.loads(p.read_text()) if p.exists() else {}
        cur = rep.setdefault(h, {}).setdefault(relation, {})
        for k_, v in entry.items():
            if k_ == "rewired" and isinstance(v, dict):
                cur.setdefault("rewired", {}).update(v)
            else:
                cur[k_] = v
        p.write_text(json.dumps(rep, indent=1, default=str))


def main_rewired(a, hs, rels) -> int:
    seed = a.rewire_seed; universe = cik_universe()
    srcs = load_sources(set(rels) | {"E1_SIC4"}); fail = 0
    for h in hs:
        ctx = make_ctx(h, with_beta=True)
        for r in rels:
            t0 = time.time(); entry: dict = {}
            if not a.no_build:
                g, srcs_rw = build_rewired(h, r, seed, ctx, srcs, universe, progress=True); write_graph(h, r, g, rewire_seed=seed)
                entry = {"n_edges": g["meta"]["n_edges"], "edges_per_year": g["meta"]["edges_per_year"], "swap_share": g["meta"]["rewire"]["swap_share"],
                         "method": g["meta"]["rewire"]["method"], "table_level": g["meta"]["rewire"]["table_level"], "build_seconds": round(time.time() - t0, 1)}
                print(f"[{h} {r} rw{seed}] edges={g['meta']['n_edges']} swap_share={entry['swap_share']} build={entry['build_seconds']}s", flush=True)
            else:
                srcs_rw = rewired_sources(srcs, r, seed, universe)
            if a.check:
                t1 = time.time()
                try:
                    chk = check_rewired(h, r, seed, ctx, srcs_rw, srcs_real=srcs); chk["passed"] = True
                except AssertionError as ex:
                    chk = {"passed": False, "error": str(ex)}; fail += 1
                chk["check_seconds"] = round(time.time() - t1, 1); entry["check"] = chk
                print(f"[{h} {r} rw{seed}] CHECK {'PASS' if chk['passed'] else 'FAIL'}: {json.dumps(chk, default=str)[:500]}", flush=True)
            if not a.no_build:
                del g
            update_report({"rewired": {str(seed): entry}}, h, r)
    return 1 if fail else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="all", choices=["W", "M", "Q", "all"])
    ap.add_argument("--relation", default="all", choices=list(common.RELATIONS) + ["all"])
    ap.add_argument("--check", action="store_true", help="run assertions on the output after building")
    ap.add_argument("--no-build", action="store_true", help="with --check: check the existing files, do not rebuild")
    ap.add_argument("--rewire-seed", type=int, choices=[0, 1, 2], default=None, help="build the persistent rewired null (A6) for this seed instead of the real graphs")
    a = ap.parse_args(argv)
    hs = common.HORIZONS if a.horizon == "all" else (a.horizon,)
    rels = common.RELATIONS if a.relation == "all" else (a.relation,)
    if a.rewire_seed is not None:
        return main_rewired(a, hs, rels)
    srcs = load_sources(rels); fail = 0
    for h in hs:
        ctx = make_ctx(h)
        for r in rels:
            entry: dict = {}
            if not a.no_build:
                t0 = time.time(); g = build_relation(h, r, ctx, srcs, progress=True); write_graph(h, r, g)
                entry |= {"n_edges": g["meta"]["n_edges"], "edges_per_year": g["meta"]["edges_per_year"], "edges_per_snapshot": g["meta"]["edges_per_snapshot"],
                          "firms_present": g["meta"]["firms_present_per_snapshot"], "build_seconds": round(time.time() - t0, 1),
                          "built_utc": pd.Timestamp.utcnow().isoformat()}
                print(f"[{h} {r}] edges={g['meta']['n_edges']} per-snapshot median={g['meta']['edges_per_snapshot']['median']} build={entry['build_seconds']}s", flush=True)
                del g
            if a.check:
                t0 = time.time()
                try:
                    chk = check_graph(h, r, ctx, srcs); chk["passed"] = True
                except AssertionError as ex:
                    chk = {"passed": False, "error": str(ex)}; fail += 1
                chk["check_seconds"] = round(time.time() - t0, 1); entry["check"] = chk
                print(f"[{h} {r}] CHECK {'PASS' if chk['passed'] else 'FAIL'}: {json.dumps(chk, default=str)[:600]}", flush=True)
            update_report(entry, h, r)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
