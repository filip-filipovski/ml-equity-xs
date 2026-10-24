"""Economic-link graph machinery: edge tables, point-in-time resolver, propagation operator and nulls.

* edge tables with ``valid_from`` / ``valid_to`` (E1 SIC co-membership, E4 customer ->
  supplier, E5 BEA input-output) and the point-in-time E4 resolver (exact or unique >= 6-char
  prefix match against filer names valid at the filing date, exclusion rules);
* the exact simple-path propagation operator for hops 1..3;
* the market-model S-return (PIT 252-session beta, ``equity_ml_grid.name_betas``);
* the path-weight-weighted MEAN hop signal and its residualisation over the REACHED set only;
* the N3 nulls: stratified double-edge rewiring (E4) and within-column permutation (E5);
* the coverage gate and dip rule for the sorting units.

``scripts/build_edges.py`` builds the edge tables and the coverage counts from this module;
``experiments/gnn/build_graphs.py`` reuses the two null generators.
"""

# ruff: noqa: N802, N803, N806, E501  (matrix names follow the maths; report strings)
from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TRAIN_END = pd.Timestamp("2023-01-01")  # study-15 computation bound (equities TRAIN end)
EDGES_DIR = REPO / "data_cache" / "links" / "edges"
ART_DIR = REPO / "data_cache" / "studies" / "eqgraph"
OPERATOR_VERSION = "exact-simple-paths-v1"
RESOLVER_VERSION = "e4-pit-resolver-v1"
EDGE_VERSION = "eqgraph-edges-v1"
FLAT_BELOW = {"E4": 30, "E5": 20}
GATE_MEDIAN = {"E4": 60, "E5": 40}
SRET_WIN = 21
SRET_MIN_OBS = 15  # finite daily returns required in the 21-session S-return window
BETA_WIN, BETA_MINP = 252, 200
REWIRE_TRIES_PER_EDGE = 10
LESSOR_SIC = (set(range(6500, 6600)) | {6798})
LESSOR_NAME = re.compile(r"income\s*&\s*growth fund|income and growth fund|equipment fund|leasing fund", re.I)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# 1. Propagation operator (exact simple paths)
# ===========================================================================
def _offdiag(A: np.ndarray) -> np.ndarray:
    B = np.array(A, dtype=np.float64, copy=True)
    np.fill_diagonal(B, 0.0)
    return B


def path_matrices(W: np.ndarray, hmax: int = 3) -> list[np.ndarray]:
    """Exact simple-path matrices P1..P_hmax (hmax <= 3) of a weighted directed graph without
    self-loops: P1 = W, P2 = offdiag(W W),
    P3 = offdiag( offdiag(W W) W - W diag(W W) + W o W^T o W ), where (W diag(W W))[i, j] =
    W[i, j] (W W)[j, j] (removes i -> j -> l -> j) and the last term adds back i -> j -> i -> j,
    which offdiag(W W) W never contained but the correction removed (pre-declaration,
    "Graph machinery"). Dense float64."""
    W = _offdiag(np.asarray(W, dtype=np.float64))
    out = [W]
    if hmax >= 2:
        W2 = W @ W
        out.append(_offdiag(W2))
    if hmax >= 3:
        d2 = np.diag(W2).copy()
        P3 = out[1] @ W - W * d2[None, :] + W * W.T * W
        P3 = _offdiag(P3)
        P3[np.abs(P3) < 1e-300] = 0.0
        out.append(P3)
    if hmax > 3:
        raise ValueError("hmax <= 3")
    return out


def path_matrices_sparse(W: Any, hmax: int = 3) -> list[Any]:
    """Same exact simple-path formula on a scipy.sparse matrix (E4 graphs of a few thousand
    filers); returns CSR matrices."""
    import scipy.sparse as sps

    W = sps.csr_matrix(W, dtype=np.float64)
    W.setdiag(0)
    W.eliminate_zeros()
    out = [W]
    if hmax >= 2:
        W2 = (W @ W).tocsr()
        P2 = W2.copy()
        P2.setdiag(0)
        P2.eliminate_zeros()
        out.append(P2)
    if hmax >= 3:
        d2 = np.asarray(W2.diagonal()).ravel()
        P3 = (P2 @ W - W @ sps.diags(d2) + W.multiply(W.T).multiply(W)).tocsr()
        P3.setdiag(0)
        P3.data[np.abs(P3.data) < 1e-300] = 0.0
        P3.eliminate_zeros()
        out.append(P3)
    return out


def brute_simple_paths(Wd: np.ndarray, h: int) -> np.ndarray:
    """Sum of weight products over simple paths with exactly ``h`` edges (brute force)."""
    Wd = np.asarray(Wd, dtype=np.float64)
    n = Wd.shape[0]
    out = np.zeros_like(Wd)
    adj = [np.flatnonzero(Wd[i]) for i in range(n)]

    def rec(path: list[int], prod: float) -> None:
        if len(path) == h + 1:
            out[path[0], path[-1]] += prod
            return
        for nx in adj[path[-1]]:
            if nx in path:
                continue
            rec([*path, int(nx)], prod * Wd[path[-1], nx])

    for i in range(n):
        rec([i], 1.0)
    return out


def hop_signal(P: np.ndarray, s: np.ndarray, origin_ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Path-weight-weighted MEAN of origin shocks at every target j:
    sum_i P[i, j] s_i / sum_i P[i, j] over origins with ``origin_ok``; returns (signal, total
    path weight); signal NaN where the total weight is 0 (target not reached)."""
    m = np.asarray(origin_ok, bool) & np.isfinite(s)
    Pm = P[m] if not hasattr(P, "tocsr") else P.tocsr()[np.flatnonzero(m)]
    tw = np.asarray(Pm.sum(axis=0)).ravel()
    num = np.asarray(Pm.T @ np.asarray(s, dtype=np.float64)[m]).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        sig = np.where(tw > 0, num / np.where(tw > 0, tw, 1.0), np.nan)
    return sig, tw


# ===========================================================================
# 2. Residualisation on the reached set only
# ===========================================================================
def residualise_reached(y: np.ndarray, X: Sequence[np.ndarray] | np.ndarray, reached: np.ndarray,
                        *, min_extra: int = 2) -> np.ndarray:
    """Within-date OLS residual of ``y`` on [1, X] fitted over the REACHED units only (units
    with a structurally nonzero signal); NaN outside the reached set, and NaN everywhere when
    the reached set has fewer than ncol + ``min_extra`` units. Missing regressor values are
    filled with 0 (the cross-sectional centre of an abnormal-return regressor)."""
    y = np.asarray(y, dtype=np.float64)
    reached = np.asarray(reached, bool) & np.isfinite(y)
    if isinstance(X, np.ndarray) and X.ndim == 2:
        Xs = [np.asarray(X[:, k], dtype=np.float64) for k in range(X.shape[1])]
    else:
        Xs = [np.asarray(x, dtype=np.float64) for x in X]
    out = np.full(len(y), np.nan)
    k = int(reached.sum())
    if k < len(Xs) + 1 + min_extra:
        return out
    A = np.column_stack([np.ones(k)] + [np.nan_to_num(x[reached], nan=0.0) for x in Xs])
    b, *_ = np.linalg.lstsq(A, y[reached], rcond=None)
    out[reached] = y[reached] - A @ b
    return out


# ===========================================================================
# 3. N3 nulls
# ===========================================================================
def edge_strata(cust_sic4: np.ndarray) -> np.ndarray:
    """Stratum label per edge from the CUSTOMER's SIC4 (string array, '' = unknown): SIC4 when
    that SIC4 stratum has >= 2 edges, else SIC3, then SIC2; edges whose customer SIC is unknown
    or that are alone at SIC2 get a unique singleton label (never swapped)."""
    s4 = np.asarray([str(x) if x is not None else "" for x in cust_sic4], dtype=object)
    lab = np.empty(len(s4), dtype=object)
    known = np.array([len(x) == 4 and x.isdigit() and x != "0000" for x in s4])
    c4 = pd.Series(s4[known]).value_counts()
    c3 = pd.Series([x[:3] for x in s4[known]]).value_counts()
    c2 = pd.Series([x[:2] for x in s4[known]]).value_counts()
    for e, x in enumerate(s4):
        if not known[e]:
            lab[e] = f"_single_{e}"
        elif c4.get(x, 0) >= 2:
            lab[e] = "4:" + x
        elif c3.get(x[:3], 0) >= 2:
            lab[e] = "3:" + x[:3]
        elif c2.get(x[:2], 0) >= 2:
            lab[e] = "2:" + x[:2]
        else:
            lab[e] = f"_single_{e}"
    return lab


def rewire_stratified(cust: np.ndarray, supp: np.ndarray, strata: np.ndarray, rng: np.random.Generator,
                      *, tries_per_edge: int = REWIRE_TRIES_PER_EDGE) -> tuple[np.ndarray, float]:
    """Degree-preserving double-edge swaps restricted to pairs of edges in the same customer
    stratum: (i1 -> j1, i2 -> j2) -> (i2 -> j1, i1 -> j2), i.e. the two suppliers exchange
    customers. Every supplier keeps its in-degree, its weights (the weight stays with the
    supplier's slot) and the stratum composition of its customers; every customer keeps its
    out-degree. Self-loops and duplicate edges are refused. Returns (new customer array aligned
    with ``supp``, share of edges whose customer changed)."""
    c = np.asarray(cust).copy()
    s = np.asarray(supp)
    m = len(c)
    present = set(zip(c.tolist(), s.tolist(), strict=True))
    groups: dict[Any, list[int]] = {}
    for e, g in enumerate(strata):
        groups.setdefault(g, []).append(e)
    for g, idx in groups.items():
        if len(idx) < 2:
            continue
        ia = np.asarray(idx)
        tries = tries_per_edge * len(ia)
        a = rng.choice(ia, tries)
        b = rng.choice(ia, tries)
        for e1, e2 in zip(a.tolist(), b.tolist(), strict=True):
            if e1 == e2:
                continue
            i1, j1, i2, j2 = c[e1], s[e1], c[e2], s[e2]
            if i1 == i2 or j1 == j2 or i2 == j1 or i1 == j2:
                continue
            if (i2, j1) in present or (i1, j2) in present:
                continue
            present.discard((i1, j1))
            present.discard((i2, j2))
            present.add((i2, j1))
            present.add((i1, j2))
            c[e1], c[e2] = i2, i1
    share = float((c != np.asarray(cust)).mean()) if m else float("nan")
    return c, share


def permute_within_columns(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """E5 N3: independent permutation of the off-diagonal entries within each column (each
    buyer keeps its input-share distribution and its zero diagonal)."""
    W = np.asarray(W, dtype=np.float64)
    n = W.shape[0]
    out = np.zeros_like(W)
    for j in range(n):
        rows = np.r_[0:j, j + 1:n]
        out[rows, j] = W[rows, j][rng.permutation(n - 1)]
    return out


# ===========================================================================
# 4. Shocks
# ===========================================================================
def s_returns(R: np.ndarray, mktrf: np.ndarray, rf: np.ndarray, rows: np.ndarray, betas: np.ndarray, *,
              win: int = SRET_WIN, min_obs: int = SRET_MIN_OBS) -> np.ndarray:
    """Market-model S-return at each formation row t: sum over the ``win`` sessions t-win+1..t
    of (r - rf - beta_t MktRF), beta_t = the PIT 252-session OLS beta through the close of t
    (``betas`` row-aligned with ``rows``; NaN = origin dropped). NaN with fewer than ``min_obs``
    finite daily returns in the window. Uses no row after t."""
    out = np.full((len(rows), R.shape[1]), np.nan, dtype=np.float64)
    mk = np.asarray(mktrf, dtype=np.float64)
    rfv = np.asarray(rf, dtype=np.float64)
    for i, t in enumerate(np.asarray(rows, dtype=np.int64)):
        a = max(0, int(t) - win + 1)
        r = R[a:t + 1].astype(np.float64)
        b = np.asarray(betas[i], dtype=np.float64)
        e = r - rfv[a:t + 1, None] - b[None, :] * mk[a:t + 1, None]
        ok = np.isfinite(e)
        n = ok.sum(0)
        out[i] = np.where((n >= min_obs) & np.isfinite(b), np.where(ok, e, 0.0).sum(0), np.nan)
    return out


def loo_mean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Leave-one-out mean of ``values`` within ``groups`` (NaN values and '' groups ignored);
    NaN when no other member has a value."""
    v = np.asarray(values, dtype=np.float64)
    g = pd.Series(np.asarray(groups, dtype=object))
    ok = np.isfinite(v) & g.notna().to_numpy() & (g.astype(str) != "").to_numpy()
    has_g = g.notna().to_numpy() & (g.astype(str) != "").to_numpy()
    df = pd.DataFrame({"g": g.where(has_g, "__none__").astype(str), "v": np.where(ok, v, 0.0), "n": ok.astype(float)})
    tot = df.groupby("g")[["v", "n"]].transform("sum")
    num = tot["v"].to_numpy() - np.where(ok, v, 0.0)
    den = tot["n"].to_numpy() - ok.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    out[~has_g] = np.nan
    return out


# ===========================================================================
# 5. E4 resolver (point-in-time names, binary match, exclusions)
# ===========================================================================
SUFFIXES = frozenset("member inc incorporated corp corporation co company ltd limited llc lp plc "
                     "holdings holding group the sa ag nv se de new".split())
_SPLIT_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")
# Frozen anonymity regex (study 15, applied in addition to the ingest's ``anonymous`` flag; the
# union is stricter than either). Built from the design review's s03 rules.
ANON_START = re.compile(
    r"^(the)?(major|significant|largest|top|single|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"first|second|third|fourth|fifth|other|others|unrelated|unnamed|unaffiliated|individual|primary|key|"
    r"principal|no|twenty|customer|customers|client|clients|distributor|distributors|wholesaler|retailer|"
    r"oem|licensee|partner|reseller|tenant|lessee|payer|payor|purchaser|buyer|account|vendor|vendors|"
    r"supplier|suppliers|all)", re.I)
ANON_END = re.compile(
    r"(customer|customers|client|clients|distributor|distributors|wholesaler|purchaser|licensee|tenant|"
    r"tenants|account|accounts|payor|payer|reseller|resellers|retailer|retailers|partner|partners|"
    r"counterparty|vendor|vendors|supplier|suppliers)([a-z]|\d+|one|two|three|four|five|six|seven|eight|"
    r"nine|ten)?$", re.I)
GOV_TOKENS = frozenset("government governments governmental federal department departments dod army navy "
                       "medicare medicaid agency agencies ministry nasa va usps tricare municipal "
                       "municipality municipalities".split())
GOV_PHRASES = ("air force", "state of", "city of", "county of", "airforce", "stateof", "cityof", "countyof")


def name_tokens(name: str) -> list[str]:
    """Lower-case alphanumeric tokens of a name; CamelCase member names are split first."""
    s = str(name or "")
    s = _SPLIT_CAMEL.sub(" ", s)
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def normalise_name(name: str) -> str:
    """Pinned normalisation: lower case, alphanumerics only, trailing corporate suffixes removed
    (member, inc, incorporated, corp, corporation, co, company, ltd, limited, llc, lp, plc,
    holdings, holding, group, the, sa, ag, nv, se, de, new), applied token-wise so that a name
    ending in the letters 'co' is not truncated."""
    tok = name_tokens(name)
    while tok and tok[-1] in SUFFIXES:
        tok.pop()
    return "".join(tok)


def is_anonymous_member(member: str) -> bool:
    k = "".join(name_tokens(member))
    if not k:
        return True
    return bool(ANON_START.match(k) or ANON_END.search(k))


def is_government_member(member: str) -> bool:
    tok = name_tokens(member)
    if any(t in GOV_TOKENS for t in tok):
        return True
    sp_ = " ".join(tok)
    compact = "".join(tok)
    return any(p in sp_ or p in compact for p in GOV_PHRASES)


def is_lessor(sic: Any, name: str) -> bool:
    try:
        s = int(str(sic)[:4])
    except (TypeError, ValueError):
        s = -1
    return s in LESSOR_SIC or bool(LESSOR_NAME.search(str(name or "")))


def parse_former_names(former: Any) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """filer_profiles ``former_names`` ("NAME|from|to;...") -> dated entries; undated parts are
    dropped (never used)."""
    out = []
    if not isinstance(former, str) or not former:
        return out
    for part in former.split(";"):
        bits = part.split("|")
        if len(bits) != 3 or not bits[0]:
            continue
        a, b = pd.to_datetime(bits[1], errors="coerce"), pd.to_datetime(bits[2], errors="coerce")
        if pd.isna(a) or pd.isna(b):
            continue
        out.append((bits[0], a, b))
    return out


@dataclass
class PitNameIndex:
    """Normalised filer name -> [(cik, valid_from, valid_to)], names VALID AT A DATE only:
    the current name from the end of its latest dated former name (from the beginning when the
    CIK has no former names), each dated former name over its own [from, to]."""

    entries: dict[str, list[tuple[int, pd.Timestamp, pd.Timestamp]]]
    keys_sorted: list[str]
    display: dict[tuple[int, str], str]
    undated_parts: int = 0

    @classmethod
    def build(cls, profiles: pd.DataFrame) -> PitNameIndex:
        ent: dict[str, list] = {}
        disp: dict[tuple[int, str], str] = {}
        lo, hi = pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")
        und = 0
        for cik, nm, former in profiles.loc[:, ["cik", "name", "former_names"]].itertuples(index=False):
            cik = int(cik)
            parts = parse_former_names(former)
            und += (len([p for p in str(former or "").split(";") if p]) - len(parts))
            last_to = lo
            for fn, a, b in parts:
                k = normalise_name(fn)
                if k:
                    ent.setdefault(k, []).append((cik, a, b))
                    disp[(cik, k)] = fn
                last_to = max(last_to, b)
            k = normalise_name(nm)
            if k:
                ent.setdefault(k, []).append((cik, last_to, hi))
                disp[(cik, k)] = str(nm)
        return cls(ent, sorted(ent), disp, und)

    def _valid(self, key: str, d: pd.Timestamp) -> set[int]:
        return {c for c, a, b in self.entries.get(key, []) if a <= d <= b}

    def resolve(self, member: str, d: pd.Timestamp) -> tuple[int | None, str, str]:
        """(cik or None, method, matched name). Exact normalised match against names valid at
        ``d``; else a unique prefix match (member key >= 6 characters is a prefix of the valid
        names of exactly one CIK). Binary: no fuzzy matching."""
        import bisect

        n = normalise_name(member)
        if not n:
            return None, "empty", ""
        ex = self._valid(n, d)
        if len(ex) == 1:
            c = next(iter(ex))
            return c, "exact", self.display.get((c, n), n)
        if len(ex) > 1:
            return None, "ambiguous_exact", ""
        if len(n) < 6:
            return None, "none", ""
        i = bisect.bisect_left(self.keys_sorted, n)
        hits: dict[int, str] = {}
        while i < len(self.keys_sorted) and self.keys_sorted[i].startswith(n):
            k = self.keys_sorted[i]
            if k != n:
                for c in self._valid(k, d):
                    hits.setdefault(c, k)
            i += 1
        if len(hits) == 1:
            c, k = next(iter(hits.items()))
            return c, "prefix", self.display.get((c, k), k)
        return None, ("ambiguous_prefix" if hits else "none"), ""


def first_token_counts(profiles: pd.DataFrame) -> dict[str, int]:
    """How many filers (current or former names) have each first name token (stem rule)."""
    cnt: dict[str, set[int]] = {}
    for cik, nm, former in profiles.loc[:, ["cik", "name", "former_names"]].itertuples(index=False):
        names = [str(nm)] + [p[0] for p in parse_former_names(former)]
        for x in names:
            t = name_tokens(x)
            if t:
                cnt.setdefault(t[0], set()).add(int(cik))
    return {k: len(v) for k, v in cnt.items()}


STEM_MAX_FILERS = 20  # a first token is "distinctive" when <= 20 filers start with it


def shared_distinctive_stem(a: str, b: str, ftc: Mapping[str, int]) -> bool:
    """Parent / subsidiary by name: the two names share their first token, the token has >= 3
    characters and at most ``STEM_MAX_FILERS`` SEC filers start with it (e.g. QEP Resources ->
    QEP Midstream; 'american' or 'first' are not distinctive)."""
    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb or ta[0] != tb[0] or len(ta[0]) < 3:
        return False
    return ftc.get(ta[0], 0) <= STEM_MAX_FILERS


# ===========================================================================
# 6. Coverage gate and dip rule
# ===========================================================================
def coverage_gate(counts: Sequence[int], *, gate_median: int, flat_below: int) -> dict[str, Any]:
    """Pinned: NOT RUN when the median count over the OOS span is below ``gate_median``; for a
    trial that runs, months with fewer than ``flat_below`` units are held flat."""
    c = np.asarray(counts, dtype=float)
    med = float(np.median(c)) if len(c) else float("nan")
    run = bool(len(c) and med >= gate_median)
    flat = (c < flat_below) if run else np.zeros(len(c), bool)
    return {"median": med, "min": float(c.min()) if len(c) else float("nan"),
            "max": float(c.max()) if len(c) else float("nan"), "gate_median": gate_median,
            "flat_below": flat_below, "run": run, "n_months": int(len(c)),
            "flat_months": int(flat.sum()), "flat_mask": flat.tolist()}


# ===========================================================================
# 7. Frame hash
# ===========================================================================


def frame_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    h.update(",".join(map(str, df.columns)).encode())
    return h.hexdigest()


# ===========================================================================
# 8. French daily factors bounded at the TRAIN end
# ===========================================================================
FF_TRAIN_PATH = ART_DIR / "ff_daily_train.parquet"


def build_ff_train(end: pd.Timestamp = TRAIN_END, out: Path = FF_TRAIN_PATH) -> dict[str, Any]:
    """The stored ``ff_daily.parquet`` is split at the OLD equities sealed start (2022-01-01),
    so 2022 (TRAIN since the 2026-10-11 zone change) is only in ``ff_daily_sealed.parquet``,
    which the loader opens only with a token. This rebuilds the factor table offline from the
    SAME raw French zips already on disk (no download; the ingest's own join), drops every row
    at or after ``end`` before anything else touches it, writes a TRAIN-bounded parquet and
    checks that its pre-2022 rows equal ``ff_daily.parquet`` exactly."""
    import io
    import zipfile

    from eqxs.data_zones import EQUITIES_ZONES_PATH, safe_load_parquet
    from eqxs import equities_xs as xs

    raw = REPO / "data_cache" / "equities_xs" / "raw" / "french"
    names = {"ff3": "F-F_Research_Data_Factors_daily_CSV.zip", "mom": "F-F_Momentum_Factor_daily_CSV.zip",
             "ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"}
    fr = {}
    bound = end.tz_localize("UTC") if end.tzinfo is None else end
    for k, n in names.items():
        z = zipfile.ZipFile(io.BytesIO((raw / n).read_bytes()))
        d = xs.parse_french_daily_csv(z.read(z.namelist()[0]).decode("latin-1"))
        fr[k] = d.loc[d.index < end.tz_localize(None)]  # bound applied first
    ff = fr["ff3"].join(fr["mom"].rename(columns=lambda c: "MOM" if "Mom" in c else c), how="outer")
    ff = ff.join(fr["ff5"][["RMW", "CMA"]], how="outer")
    ff.index.name = "date"
    ff = ff.sort_index()
    ff.index = ff.index.tz_localize("UTC")
    ff = ff.loc[ff.index < bound]
    ref = safe_load_parquet(REPO / "data_cache" / "equities_xs" / "ff_daily.parquet", zone_config=EQUITIES_ZONES_PATH,
                            caller_intent="eqgraph.build_ff_train.check")
    pre = ff.loc[ff.index < ref.index.max() + pd.Timedelta(days=1)]
    same = bool(pre.shape == ref.shape and np.allclose(pre.to_numpy(), ref[pre.columns].to_numpy(), equal_nan=True, rtol=0, atol=0))
    out.parent.mkdir(parents=True, exist_ok=True)
    ff.to_parquet(out)
    return {"rows": int(len(ff)), "first": str(ff.index.min().date()), "last": str(ff.index.max().date()),
            "pre2022_identical_to_ff_daily_parquet": same, "path": str(out.relative_to(REPO))}


def load_ff_train(end: pd.Timestamp = TRAIN_END) -> pd.DataFrame:
    from eqxs.data_zones import EQUITIES_ZONES_PATH, safe_load_parquet

    if not FF_TRAIN_PATH.exists():
        build_ff_train(end)
    ff = safe_load_parquet(FF_TRAIN_PATH, max_end_date=end, zone_config=EQUITIES_ZONES_PATH,
                           caller_intent="eqgraph.load_ff_train")
    idx = pd.DatetimeIndex(ff.index)
    ff.index = idx.tz_localize(None) if idx.tz is not None else idx
    return ff.loc[ff.index < end]


# ===========================================================================
# 9. Data: price base (prices, universe, holding returns, betas, S-returns)
# ===========================================================================
PRICE_START_YEAR = 2003  # 252-session beta warm-up for the first formation (2004-12-31)


@dataclass
class PriceBase:
    sessions: pd.DatetimeIndex
    symbols: np.ndarray
    R: np.ndarray  # float32 daily returns
    mktrf: np.ndarray
    rf: np.ndarray
    col_cik: np.ndarray  # int64, -1 = no CIK
    me_rows: np.ndarray  # month-end rows (formation dates)
    all_: np.ndarray  # (n_me x cols) ALL universe
    liq: np.ndarray
    hret: np.ndarray  # (n_me x cols) holding return entry -> exit (float64)
    entry: np.ndarray
    exit_: np.ndarray
    beta: np.ndarray  # (n_me x cols) PIT beta through the formation close
    sret: np.ndarray  # (n_me x cols) S-return
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def me_dates(self) -> pd.DatetimeIndex:
        return self.sessions[self.me_rows]

    def col_of_cik(self) -> dict[int, int]:
        return {int(c): i for i, c in enumerate(self.col_cik) if c >= 0}


def holding_rows(n: int, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Entry = close of t+1; exit = close of the next formation + 1, truncated at the last
    session before the computation bound (only the final period, Dec 2022, is affected)."""
    entry = np.full(len(rows), -1, np.int64)
    exit_ = np.full(len(rows), -1, np.int64)
    for i in range(len(rows) - 1):
        e, x = int(rows[i]) + 1, min(int(rows[i + 1]) + 1, n - 1)
        if e < x:
            entry[i], exit_[i] = e, x
    return entry, exit_


def compound_between(R: np.ndarray, entry: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    """Compounded return over rows entry+1 .. exit (missing daily returns = 0), per column."""
    lc = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(np.log1p(np.nan_to_num(R, nan=0.0).astype(np.float64)), axis=0)])
    out = np.full((len(entry), R.shape[1]), np.nan)
    for i, (e, x) in enumerate(zip(entry, exit_, strict=True)):
        if e >= 0:
            out[i] = np.expm1(lc[x + 1] - lc[e + 1])
    return out


def build_price_base(end: pd.Timestamp = TRAIN_END, *, cache: Path | None = ART_DIR / "price_base.npz",
                     refresh: bool = False) -> PriceBase:
    """Survivor-only yfinance panel 2003 .. ``end`` (exclusive) through the zone-aware loaders,
    the study-13 universe (close >= $5, 63-session median dollar volume >= $1M = ALL; LIQ = top
    half of ALL), holding returns, PIT betas and S-returns at every month-end. Cached (float64
    / float32 arrays, every date < ``end``)."""
    from eqxs import equities_xs as xs
    from eqxs import equity_ml as ml
    from eqxs import equity_ml_grid as g

    if cache is not None and cache.exists() and not refresh:
        z = np.load(cache, allow_pickle=False)
        sess = pd.DatetimeIndex(z["sessions"].astype("datetime64[ns]"))
        assert sess.max() < end
        meta = json.loads(str(z["meta"]))
        return PriceBase(sess, z["symbols"].astype(str), z["R"], z["mktrf"], z["rf"], z["col_cik"], z["me_rows"],
                         z["all_"], z["liq"], z["hret"], z["entry"], z["exit_"], z["beta"], z["sret"], meta)
    t0 = time.time()
    uc = xs.load_static("universe_current")
    common = uc[uc["security_type"] == "common"]
    yf_of = dict(zip(common["symbol"], common["yf_symbol"], strict=False))
    keep = set(common["yf_symbol"])
    cat = pd.CategoricalDtype(sorted(keep))
    parts = []
    for y in range(PRICE_START_YEAR, end.year):
        px = xs.load_yf_daily(columns=["symbol", "date", "close", "adj_close", "volume", "splits"],
                              start=f"{y}-01-01", end=min(pd.Timestamp(f"{y + 1}-01-01"), end))
        px = px[px["symbol"].isin(keep)]
        px = px.assign(date=pd.to_datetime(px["date"]), symbol=px["symbol"].astype(cat))
        for c in ("close", "adj_close", "splits"):
            px[c] = px[c].astype(np.float32)
        px["volume"] = px["volume"].astype(np.float64)
        parts.append(px)
    px = pd.concat(parts, ignore_index=True)
    del parts
    assert px["date"].max() < end
    codes = px["symbol"].cat.codes.to_numpy()
    present = np.unique(codes)
    symbols = np.asarray(cat.categories)[present]
    sessions = pd.DatetimeIndex(np.unique(px["date"].to_numpy()))
    row = sessions.get_indexer(px["date"])
    col = np.searchsorted(present, codes)
    pm = ml.build_price_matrices(sessions, symbols, row, col, px["close"].to_numpy(np.float64),
                                 px["adj_close"].to_numpy(np.float64), px["volume"].to_numpy(np.float64),
                                 px["splits"].to_numpy(np.float64))
    n_bars = int(len(px))
    del px
    _log(f"price matrices {pm.n_rows} x {pm.n_cols}, {n_bars:,} bars ({time.time() - t0:.0f} s)")
    ff = load_ff_train(end).reindex(sessions)
    assert ff["Mkt-RF"].notna().all(), "French daily must cover the calendar"
    mktrf, rf = ff["Mkt-RF"].to_numpy(np.float64), ff["RF"].to_numpy(np.float64)
    me = ml.month_end_rows(sessions)
    uni = ml.build_universe(pm, me)
    bars = np.isfinite(pm.adj).sum(axis=0)
    col_of = {s: i for i, s in enumerate(pm.symbols)}
    cmap, map_diag = ml.cik_symbol_map(xs.load_static("cik_ticker_map"), yf_of, col_of, bars)
    col_cik = np.full(pm.n_cols, -1, np.int64)
    for c, s in cmap.items():
        col_cik[col_of[s]] = int(c)
    R = pm.R
    del pm
    entry, exit_ = holding_rows(len(sessions), me)
    hret = compound_between(R, entry, exit_)
    beta = g.name_betas(R, mktrf, rf, me, win=BETA_WIN, minp=BETA_MINP)
    sret = s_returns(R, mktrf, rf, me, beta)
    meta = {"sessions": [str(sessions[0].date()), str(sessions[-1].date())], "n_symbols": int(len(symbols)),
            "n_bars": n_bars, "cik_map": map_diag, "end": str(end.date()), "built_s": round(time.time() - t0, 1),
            "universe": "close >= $5, 63-session median dollar volume >= $1M (ALL); LIQ = top half of ALL",
            "sret": f"sum over {SRET_WIN} sessions of r - rf - beta MktRF, beta PIT {BETA_WIN}-session (>= {BETA_MINP} obs), >= {SRET_MIN_OBS} finite returns"}
    pb = PriceBase(sessions, np.asarray(symbols).astype(str), R, mktrf, rf, col_cik, me, uni.all_, uni.liq, hret,
                   entry, exit_, beta.astype(np.float32), sret, meta)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, sessions=sessions.values.astype("datetime64[ns]").astype(np.int64), symbols=pb.symbols,
                 R=pb.R, mktrf=mktrf, rf=rf, col_cik=col_cik, me_rows=me, all_=pb.all_, liq=pb.liq, hret=hret,
                 entry=entry, exit_=exit_, beta=pb.beta, sret=sret, meta=json.dumps(meta))
    _log(f"price base built ({time.time() - t0:.0f} s)")
    return pb


# ===========================================================================
# 10. Data: FSN sub (PIT SIC, 10-K list, co-registrant families)
# ===========================================================================
FSN_DIR = REPO / "data_cache" / "equities_xs" / "raw" / "sec_bulk" / "fsds_notes"


def fsn_quarters(last: str = "2022q4") -> list[str]:
    out = []
    for y in range(2009, int(last[:4]) + 1):
        for q in range(1, 5):
            lab = f"{y}q{q}"
            if (y, q) < (2009, 2) or lab > last:
                continue
            out.append(lab)
    return out


def read_fsn_sub(quarters: Sequence[str] | None = None, end: pd.Timestamp = TRAIN_END) -> pd.DataFrame:
    """``sub.tsv`` of the Financial Statement and Notes quarterly zips 2009q2 .. 2022q4 only (no
    zip of a quarter at or after the bound is opened; rows accepted at/after ``end`` dropped):
    adsh, cik, name, sic (as of the filing), form, accepted (UTC, from FSN's US/Eastern stamp),
    aciks (co-registrant CIKs)."""
    import zipfile

    from eqxs import equities_xs as xs

    frames = []
    for q in quarters or fsn_quarters():
        zp = FSN_DIR / f"{q}_notes.zip"
        if not zp.exists():
            alt = sorted(FSN_DIR.glob(f"{q}_notes_*.zip"))
            if not alt:
                raise FileNotFoundError(zp)
            zp = alt[-1]
        with zipfile.ZipFile(zp) as z, z.open("sub.tsv") as f:
            df = pd.read_csv(f, sep="\t", dtype=str, usecols=["adsh", "cik", "name", "sic", "form", "accepted", "aciks", "filed"],
                             keep_default_na=False, quoting=3, encoding="utf-8", encoding_errors="replace")
        df["quarter"] = q
        frames.append(df)
    sub = pd.concat(frames, ignore_index=True)
    sub["cik"] = pd.to_numeric(sub["cik"], errors="coerce").astype("Int64")
    sub["accepted_utc"] = xs.fsn_accepted_utc(sub["accepted"])
    sub = sub[sub["accepted_utc"].notna() & (sub["accepted_utc"] < end.tz_localize("UTC"))].copy()
    sub["ts"] = sub["accepted_utc"]
    sub["sic"] = sub["sic"].str.strip().str.zfill(4).where(sub["sic"].str.strip() != "", "")
    return sub.drop(columns=["accepted"]).drop_duplicates("adsh").reset_index(drop=True)


# ===========================================================================
# 11. Edge tables
# ===========================================================================
E4_FORMS = ("10-K", "10-K/A", "10-KT")
E4_TAGS = ("ConcentrationRiskPercentage1", "ConcentrationRiskPercentage")
ORIGINAL_10K = ("10-K", "10-KT")


def session_of(instants_utc: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """Session at whose close an instant is public (the day-0 rule, acceptance-session mode:
    09:30-16:00 ET -> that session, before 09:30 -> that session, after 16:00 / weekend /
    holiday -> the next session); NaT past the calendar."""
    from eqxs import equity_events as ee

    return ee.day0_from_acceptance(pd.Series(pd.to_datetime(instants_utc, utc=True)).reset_index(drop=True), sessions,
                                   intraday="acceptance_session")


def build_e1(sub: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """E1 membership intervals: per CIK, each FSN filing's SIC valid from its acceptance to the
    next filing's acceptance; before the first FSN filing the first FSN SIC (``backfill``);
    CIKs never in FSN: the current profile SIC over all dates (``profile``)."""
    s = sub[sub["sic"].str.len() == 4].loc[:, ["cik", "accepted_utc", "sic", "adsh"]].dropna()
    s = s.sort_values(["cik", "accepted_utc"]).reset_index(drop=True)
    s["valid_to"] = s.groupby("cik")["accepted_utc"].shift(-1)  # NaT = open at the bound
    rows = [s.rename(columns={"accepted_utc": "valid_from"}).assign(source="fsn_as_of_filing")]
    first = s.groupby("cik").first().reset_index()
    rows.append(pd.DataFrame({"cik": first["cik"], "valid_from": pd.Timestamp("1990-01-01", tz="UTC"),
                              "valid_to": first["accepted_utc"], "sic": first["sic"], "adsh": "",
                              "source": "backfill_first_fsn"}))
    seen = set(s["cik"].astype(int))
    pr = profiles[~profiles["cik"].astype(int).isin(seen)]
    pr = pr[pr["sic"].astype(str).str.fullmatch(r"\d{4}") & (pr["sic"].astype(str) != "0000")]
    rows.append(pd.DataFrame({"cik": pr["cik"].astype("Int64"), "valid_from": pd.Timestamp("1990-01-01", tz="UTC"),
                              "valid_to": pd.NaT, "sic": pr["sic"].astype(str), "adsh": "",
                              "source": "profile_current"}))
    e1 = pd.concat(rows, ignore_index=True)
    e1["sic2"] = e1["sic"].str[:2]
    e1["ts"] = e1["valid_from"]
    return e1.loc[:, ["cik", "sic", "sic2", "valid_from", "valid_to", "source", "adsh", "ts"]].rename(columns={"sic": "sic4"})


def sic_at(e1: pd.DataFrame) -> Any:
    """Closure cik, instant -> SIC4 ('' unknown) from the E1 table (latest interval starting at
    or before the instant; naive instants are read as UTC)."""
    e = e1.assign(vf=pd.DatetimeIndex(pd.to_datetime(e1["valid_from"], utc=True)).asi8).sort_values(["cik", "vf"])
    by = {int(c): (g_["vf"].to_numpy(), g_["sic4"].to_numpy()) for c, g_ in e.groupby("cik")}

    def f(cik: int, t: pd.Timestamp) -> str:
        v = by.get(int(cik))
        if v is None:
            return ""
        t = pd.Timestamp(t)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        i = int(np.searchsorted(v[0], t.value, side="right")) - 1
        return str(v[1][max(i, 0)])

    return f


def build_sic_bea_map(sics: Sequence[str]) -> pd.DataFrame:
    """Firm -> BEA summary industry (pinned): SEC SIC4 -> the 1997 NAICS with the largest share
    of that SIC's 1997 Economic Census receipts (Census 1987-SIC / 1997-NAICS bridge,
    ``ship_share``), then NAICS -> BEA summary by BEA's NAICS prefixes. Kept only when the SIC is
    a 1987 SIC4 of the bridge (``match_level == 'sic4'``; SEC group codes such as 7370 have no
    receipts of their own and are dropped), the dominant share is >= 0.5 and it maps to exactly
    one BEA code. No equal splits."""
    from eqxs import bea_io

    bridge = bea_io.load_static_link("sic87_naics97_ec97_bridge")
    nb = bea_io.load_static_link("naics97_bea_summary")
    amb = bea_io.sic_bea_ambiguity(sorted(set(sics)), bridge.loc[:, ["sic4", "naics97", "ship_share", "shipments_kusd"]],
                                   nb.loc[:, ["naics97", "bea_summary"]], share_col="ship_share", dominant_min=0.5)
    amb["bea"] = [b[0] if len(b) == 1 else "" for b in amb["bea_dominant"]]
    amb["kept"] = (amb["match_level"] == "sic4") & amb["dominant_ok"]
    amb["drop_reason"] = np.where(amb["kept"], "", np.where(amb["match_level"] != "sic4", "not_a_bridge_sic4_" + amb["match_level"],
                                                             "dominant_share_lt_0.5_or_not_unique_bea"))
    return amb.loc[:, ["sic4", "match_level", "dominant_naics97", "dominant_share", "bea", "dominant_ok", "kept",
                       "drop_reason", "n_bea_any"]]


def build_e5(end: pd.Timestamp = TRAIN_END) -> tuple[pd.DataFrame, dict]:
    """E5 edges: selling industry -> buying industry, weight = commodity i's share of industry
    j's total intermediate inputs (T005), DIAGONAL SET TO ZERO (the diagonal share is stored),
    valid_from = 1 Jan data year + 2, valid_to = 1 Jan data year + 3. Only data years whose
    valid_from is before ``end`` are read (<= 2020)."""
    from eqxs import bea_io

    u = bea_io.load_use_summary(end=end)
    assert pd.to_datetime(u["ts"], utc=True).max() < end.tz_localize("UTC")
    codes = sorted(set(u["buy_code"]))
    u = u[u["sell_code"].isin(codes)].copy()
    u["share_raw"] = u["share_of_buyer_inputs"].fillna(0.0).astype(np.float64)
    u["diag_share_of_buyer"] = u["buy_code"].map(dict.fromkeys(codes, 0.0))
    dg = u[u["is_diagonal"]].set_index(["data_year", "buy_code"])["share_raw"]
    u["diag_share_of_buyer"] = [float(dg.get((y, b), 0.0)) for y, b in zip(u["data_year"], u["buy_code"], strict=True)]
    u["weight"] = np.where(u["is_diagonal"], 0.0, u["share_raw"])
    e5 = u.loc[:, ["data_year", "sell_code", "sell_name", "buy_code", "buy_name", "weight", "share_raw", "is_diagonal",
                   "diag_share_of_buyer", "value_musd", "buyer_total_intermediate_musd", "vintage_modified_utc", "ts",
                   "valid_to"]].rename(columns={"ts": "valid_from"})
    vt = pd.to_datetime(e5["valid_to"], utc=True)
    e5["valid_to"] = vt.where(vt < end.tz_localize("UTC"))  # NaT = open at the bound
    e5["ts"] = e5["valid_from"]
    stats = {"industries": len(codes), "data_years": [int(e5["data_year"].min()), int(e5["data_year"].max())],
             "mean_diag_share": float(e5[e5["is_diagonal"]]["share_raw"].mean()),
             "offdiag_density_gt_0.1pct": float((e5[~e5["is_diagonal"]]["weight"] > 0.001).mean()),
             "vintage_modified_utc": str(e5["vintage_modified_utc"].iloc[0])}
    return e5.reset_index(drop=True), stats


def w_matrix(e5: pd.DataFrame, data_year: int, codes: Sequence[str]) -> np.ndarray:
    d = e5[e5["data_year"] == data_year]
    ix = {c: k for k, c in enumerate(codes)}
    W = np.zeros((len(codes), len(codes)))
    W[d["sell_code"].map(ix).to_numpy(), d["buy_code"].map(ix).to_numpy()] = d["weight"].to_numpy(np.float64)
    np.fill_diagonal(W, 0.0)
    return W


def build_e4(links: pd.DataFrame, sub: pd.DataFrame, profiles: pd.DataFrame, sessions: pd.DatetimeIndex,
             e1: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """E4 customer -> supplier edges with the pinned resolver and exclusions (see the module
    docstring and the pre-declaration). One row per (supplier filing, resolved customer):
    weight = max ``ConcentrationRiskPercentage1`` / 100 over the filing's revenue-share facts
    for that member; valid_from = the acceptance session of the filing (Notes ``accepted``);
    valid_to = the acceptance session of the supplier's next original 10-K / 10-KT, or 15
    months after acceptance, whichever is first."""
    st: dict[str, Any] = {"rows_in": int(len(links))}
    L = links[links["form"].isin(E4_FORMS)]
    st["rows_10k_forms"] = int(len(L))
    L = L[L["tag"].isin(E4_TAGS)]
    st["rows_concentration_pct"] = int(len(L))
    L = L.assign(pct=pd.to_numeric(L["pct"], errors="coerce"))
    bench = L["benchmark"].fillna("").astype(str)
    L = L[(L["is_revenue_benchmark"].fillna(False).astype(bool)) | (bench == "")]
    st["rows_revenue_or_untagged_benchmark"] = int(len(L))
    L = L[(L["pct"] > 0) & (L["pct"] <= 100)]
    st["rows_pct_in_0_100"] = int(len(L))
    mem = L["customer_member"].fillna("").astype(str)
    anon = L["anonymous"].fillna(False).astype(bool).to_numpy() | np.array([is_anonymous_member(m) for m in mem])
    gov = L["government"].fillna(False).astype(bool).to_numpy() | np.array([is_government_member(m) for m in mem])
    vend = L["vendor_hint"].fillna(False).astype(bool).to_numpy()
    less = np.array([is_lessor(s, n) for s, n in zip(L["supplier_sic"], L["supplier_name"], strict=True)])
    L = L.assign(_anon=anon, _gov=gov, _vend=vend, _lessor=less)
    g_ = L.groupby(["adsh", "customer_member"], sort=False)
    F = g_.agg(supplier_cik=("supplier_cik", "first"), supplier_name=("supplier_name", "first"),
               supplier_sic=("supplier_sic", "first"), customer_name=("customer_name", "first"),
               pct=("pct", "max"), form=("form", "first"), fy=("fy", "first"), filed=("filed", "first"),
               accepted_fsn=("accepted_fsn", "first"), anon=("_anon", "max"), gov=("_gov", "max"),
               vend=("_vend", "max"), lessor=("_lessor", "max"), n_facts=("pct", "size")).reset_index()
    F["filed_date"] = pd.to_datetime(F["filed"].astype(str).str[:10], errors="coerce")
    bad_f = F["filed_date"].isna()
    F.loc[bad_f, "filed_date"] = pd.to_datetime(F.loc[bad_f, "filed"].astype(str), format="%Y%m%d", errors="coerce")
    F["year"] = pd.to_datetime(F["accepted_fsn"], utc=True).dt.year
    st["member_filing_pairs"] = int(len(F))
    reason = np.full(len(F), "", dtype=object)
    reason[F["lessor"].to_numpy()] = "supplier_lessor"
    m = reason == ""
    reason[m & F["anon"].to_numpy()] = "anonymous"
    m = reason == ""
    reason[m & F["gov"].to_numpy()] = "government"
    m = reason == ""
    reason[m & F["vend"].to_numpy()] = "vendor"
    idx = PitNameIndex.build(profiles)
    ftc = first_token_counts(profiles)
    fam: dict[int, set[int]] = {}
    for c, ac in zip(sub["cik"], sub["aciks"].fillna(""), strict=True):
        if pd.isna(c) or not ac:
            continue
        grp = {int(c)} | {int(x) for x in re.findall(r"\d+", str(ac))}
        for x in grp:
            fam.setdefault(x, set()).update(grp)
    cur_name = dict(zip(profiles["cik"].astype(int), profiles["name"].astype(str), strict=False))
    cache: dict[tuple[str, pd.Timestamp], tuple] = {}
    ccik = np.full(len(F), -1, np.int64)
    meth = np.full(len(F), "", dtype=object)
    mname = np.full(len(F), "", dtype=object)
    for k, (mm, d) in enumerate(zip(F["customer_member"], F["filed_date"], strict=True)):
        if reason[k]:
            continue
        key = (normalise_name(mm), d)
        if key not in cache:
            cache[key] = idx.resolve(mm, d)
        c, how, nm = cache[key]
        meth[k] = how
        if c is None:
            reason[k] = f"unresolved_{how}"
            continue
        ccik[k], mname[k] = c, nm
        s = int(F["supplier_cik"].iloc[k])
        if c == s:
            reason[k] = "self"
        elif c in fam.get(s, set()) or s in fam.get(c, set()):
            reason[k] = "parent_subsidiary_aciks"
        elif shared_distinctive_stem(nm, str(F["supplier_name"].iloc[k]), ftc) or shared_distinctive_stem(
                cur_name.get(c, nm), str(F["supplier_name"].iloc[k]), ftc):
            reason[k] = "parent_subsidiary_name_stem"
    F["customer_cik"] = ccik
    F["match_method"] = meth
    F["matched_name"] = mname
    F["customer_current_name"] = [cur_name.get(int(c), "") if c >= 0 else "" for c in ccik]
    F["drop_reason"] = reason
    st["drop_reasons"] = {str(k): int(v) for k, v in pd.Series(reason[reason != ""]).value_counts().items()}
    st["drop_reasons_by_year"] = {int(y): {str(k): int(v) for k, v in grp["drop_reason"].value_counts().items()}
                                  for y, grp in F.groupby("year")}
    E = F[F["drop_reason"] == ""].copy()
    # validity
    acc = pd.to_datetime(E["accepted_fsn"], utc=True)
    E["valid_from"] = session_of(acc, sessions).to_numpy()
    tk = sub[sub["form"].isin(ORIGINAL_10K)].loc[:, ["cik", "accepted_utc"]].dropna().sort_values(["cik", "accepted_utc"])
    nxt = {int(c): pd.DatetimeIndex(grp["accepted_utc"]).asi8 for c, grp in tk.groupby("cik")}
    nv = []
    for c, a in zip(E["supplier_cik"].astype(int), acc, strict=True):
        arr = nxt.get(c)
        cap = a + pd.DateOffset(months=15)
        t = None
        if arr is not None:
            j = int(np.searchsorted(arr, a.value, side="right"))
            t = pd.Timestamp(int(arr[j]), tz="UTC") if j < len(arr) else None
        nv.append(min(t, cap) if t is not None else cap)
    vt = session_of(pd.Series(nv), sessions)
    E["valid_to"] = vt.to_numpy()  # NaT = open at the computation bound (next 10-K not observable)
    E["valid_to_rule"] = ["next_10k" if (x is not None and x < a + pd.DateOffset(months=15)) else "15_months"
                          for x, a in zip(nv, acc, strict=True)]
    E = E[E["valid_from"].notna()].copy()
    E["weight"] = E["pct"].astype(np.float64) / 100.0
    sa = sic_at(e1)
    E["customer_sic4"] = [sa(c, a) for c, a in zip(E["customer_cik"], pd.to_datetime(E["accepted_fsn"], utc=True), strict=True)]
    E["resolver"] = RESOLVER_VERSION
    E["ts"] = pd.to_datetime(E["valid_from"]).dt.tz_localize("UTC")
    E = E.loc[:, ["supplier_cik", "supplier_name", "supplier_sic", "customer_cik", "customer_member", "customer_name",
                  "matched_name", "customer_current_name", "match_method", "customer_sic4", "weight", "pct", "n_facts",
                  "form", "fy", "adsh", "filed", "accepted_fsn", "valid_from", "valid_to", "valid_to_rule", "year",
                  "resolver", "ts"]].reset_index(drop=True)
    E["supplier_cik"] = E["supplier_cik"].astype(np.int64)
    st["edges"] = int(len(E))
    st["edges_by_year"] = {int(k): int(v) for k, v in E.groupby("year").size().items()}
    st["distinct_pairs_by_year"] = {int(y): int(grp[["supplier_cik", "customer_cik"]].drop_duplicates().shape[0])
                                    for y, grp in E.groupby("year")}
    st["suppliers_by_year"] = {int(y): int(grp["supplier_cik"].nunique()) for y, grp in E.groupby("year")}
    st["match_methods"] = {str(k): int(v) for k, v in E["match_method"].value_counts().items()}
    st["undated_former_name_parts_ignored"] = int(idx.undated_parts)
    return E, {**st, "all_pairs_table": F}


# ===========================================================================
# 12. Study data, clocks, signals and unit panels
# ===========================================================================
def edges_read(name: str, end: pd.Timestamp = TRAIN_END) -> pd.DataFrame:
    from eqxs.data_zones import EQUITIES_ZONES_PATH, safe_load_parquet

    df = safe_load_parquet(EDGES_DIR / name, max_end_date=end, zone_config=EQUITIES_ZONES_PATH,
                           caller_intent=f"eqgraph.edges.{name}")
    return df


@dataclass
class Clock:
    """Formation clock (monthly from the price base, or weekly): per formation row the ALL /
    LIQ universes, holding returns, PIT betas, S-returns, SIC4 and BEA industry index."""

    name: str
    rows: np.ndarray
    dates: pd.DatetimeIndex
    all_: np.ndarray
    liq: np.ndarray
    hret: np.ndarray
    entry: np.ndarray
    exit_: np.ndarray
    beta: np.ndarray
    sret: np.ndarray
    sic4: np.ndarray
    ind: np.ndarray
    loo4: np.ndarray
    loo2: np.ndarray
    horizon: str

    def target_mi(self, sessions: pd.DatetimeIndex) -> np.ndarray:
        from eqxs import equity_ml as ml

        out = np.full(len(self.rows), -1, np.int64)
        ok = self.entry >= 0
        out[ok] = ml.month_index(sessions[self.entry[ok]])
        return out


@dataclass
class StudyData:
    pb: PriceBase
    e1: pd.DataFrame
    e4: pd.DataFrame | None
    e5: pd.DataFrame
    sicmap: pd.DataFrame
    codes: list[str]
    monthly: Clock
    meta: dict[str, Any] = field(default_factory=dict)
    _weekly: Clock | None = None
    _wcache: dict = field(default_factory=dict)
    _e4_active: dict = field(default_factory=dict)

    def W(self, data_year: int) -> np.ndarray:
        if data_year not in self._wcache:
            self._wcache[data_year] = w_matrix(self.e5, data_year, self.codes)
        return self._wcache[data_year]

    def weekly(self) -> Clock:
        if self._weekly is None:
            self._weekly = make_weekly_clock(self)
        return self._weekly

    def e4_active(self, clock: Clock, i: int) -> pd.DataFrame:
        key = (clock.name, i)
        if key not in self._e4_active:
            T = clock.dates[i]
            e = self.e4
            a = e[(e["valid_from"] <= T) & (e["valid_to"].isna() | (T < e["valid_to"]))]
            a = a.groupby(["customer_cik", "supplier_cik"], as_index=False).agg(
                w=("weight", "max"), customer_sic4=("customer_sic4", "first"))
            self._e4_active[key] = a
        return self._e4_active[key]


def sic_matrix_from_e1(pb: PriceBase, e1: pd.DataFrame, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(dates x cols) SIC4 as of the formation close from the E1 intervals; source code 0 =
    FSN as of the filing, 1 = backfill (first FSN SIC), 2 = profile, -1 = none."""
    from eqxs import equity_ml as ml

    close = pd.DatetimeIndex(pd.to_datetime(ml.close_ts_utc_ns(dates), utc=True)).tz_localize(None).values
    code = {"fsn_as_of_filing": 0, "backfill_first_fsn": 1, "profile_current": 2}
    e = e1.assign(vf=pd.to_datetime(e1["valid_from"], utc=True).dt.tz_localize(None),
                  src=e1["source"].map(code)).sort_values(["cik", "vf"])
    by = {int(c): (g_["vf"].to_numpy(), g_["sic4"].to_numpy(), g_["src"].to_numpy()) for c, g_ in e.groupby("cik")}
    out = np.full((len(dates), len(pb.col_cik)), "", dtype=object)
    src = np.full((len(dates), len(pb.col_cik)), -1, np.int8)
    for j, c in enumerate(pb.col_cik):
        v = by.get(int(c)) if c >= 0 else None
        if v is None:
            continue
        pos = np.searchsorted(v[0], close, side="right") - 1
        ok = pos >= 0
        out[ok, j] = v[1][pos[ok]]
        src[ok, j] = v[2][pos[ok]]
    return out, src


def loo_matrices(sret: np.ndarray, sic4: np.ndarray, all_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out mean S-return of the OTHER ALL firms of the same SIC4 / SIC2 (D1 and the
    T7 regressors), per formation row."""
    n, k = sret.shape
    l4 = np.full((n, k), np.nan)
    l2 = np.full((n, k), np.nan)
    for i in range(n):
        v = np.where(all_[i], sret[i], np.nan)
        s4 = np.where(all_[i], sic4[i], "")
        s2 = np.array([x[:2] if x else "" for x in s4], dtype=object)
        l4[i] = loo_mean(v, s4)
        l2[i] = loo_mean(v, s2)
    return l4, l2


def industry_index(sic4: np.ndarray, sicmap: pd.DataFrame, codes: Sequence[str]) -> np.ndarray:
    kept = sicmap[sicmap["kept"]]
    ix = {c: k for k, c in enumerate(codes)}
    m = {s: ix[b] for s, b in zip(kept["sic4"], kept["bea"], strict=True) if b in ix}
    f = np.vectorize(lambda s: m.get(s, -1), otypes=[np.int16])
    return f(sic4) if sic4.size else np.zeros(sic4.shape, np.int16)


def load_study_data(end: pd.Timestamp = TRAIN_END, *, with_e4: bool = True) -> StudyData:
    pb = build_price_base(end)
    e1 = edges_read("e1_sic_membership.parquet", end)
    e5 = edges_read("e5_bea_use_edges.parquet", end)
    sicmap = edges_read("e5_sic4_bea_map.parquet", end)
    e4 = None
    if with_e4 and (EDGES_DIR / "e4_customer_supplier_edges.parquet").exists():
        e4 = edges_read("e4_customer_supplier_edges.parquet", end)
        e4["valid_from"] = pd.to_datetime(e4["valid_from"])
        e4["valid_to"] = pd.to_datetime(e4["valid_to"])
    codes = sorted(set(e5["buy_code"]))
    dates = pb.me_dates
    sic4, src = sic_matrix_from_e1(pb, e1, dates)
    ind = industry_index(sic4, sicmap, codes)
    l4, l2 = loo_matrices(pb.sret, sic4, pb.all_)
    mon = Clock("M", pb.me_rows, dates, pb.all_, pb.liq, pb.hret, pb.entry, pb.exit_, pb.beta, pb.sret, sic4, ind, l4, l2, "M")
    meta = {"sic_source_share_ALL": {lab: float((src[pb.all_] == c).mean()) for lab, c in
                                     (("fsn_as_of_filing", 0), ("backfill_first_fsn", 1), ("profile_current", 2), ("none", -1))},
            "ALL_firm_months_mapped_to_BEA": float((ind[pb.all_] >= 0).mean())}
    return StudyData(pb, e1, e4, e5, sicmap, codes, mon, meta)


def make_weekly_clock(sd: StudyData) -> Clock:
    """D5 clock: every 5th session from 2004-01-02 (``equity_ml_grid.weekly_rows``), hold
    5 sessions; betas and S-returns recomputed at each weekly row; ALL / LIQ, SIC and BEA
    industry from the latest month-end at or before the weekly row (documented
    simplification: the universe is refreshed monthly)."""
    from eqxs import equity_ml_grid as g

    pb = sd.pb
    rows = g.weekly_rows(pb.sessions)
    rows = rows[(pb.sessions[rows] >= pd.Timestamp("2004-12-01"))]
    entry, exit_ = holding_rows(len(pb.sessions), rows)
    hret = compound_between(pb.R, entry, exit_)
    beta = g.name_betas(pb.R, pb.mktrf, pb.rf, rows, win=BETA_WIN, minp=BETA_MINP)
    sret = s_returns(pb.R, pb.mktrf, pb.rf, rows, beta)
    mpos = np.searchsorted(pb.me_rows, rows, side="right") - 1
    m = sd.monthly
    all_ = m.all_[mpos]
    sic4 = m.sic4[mpos]
    l4, l2 = loo_matrices(sret, sic4, all_)
    return Clock("W", rows, pb.sessions[rows], all_, m.liq[mpos], hret, entry, exit_, beta.astype(np.float32), sret, sic4,
                 m.ind[mpos], l4, l2, "W")


@dataclass
class CellDef:
    cell_id: str
    edge: str  # E4 | E5 | E1
    hop: int
    direction: str = "fwd"  # fwd | rev (D6, transposed)
    signal: str = "incremental"  # incremental | raw
    clock: str = "M"
    first_target: str = "2005-01"
    last_target: str = "2022-12"
    role: str = "trial"


def _span_rows(clock: Clock, sessions: pd.DatetimeIndex, first: str, last: str) -> np.ndarray:
    tmi = clock.target_mi(sessions)
    a = int(first[:4]) * 12 + int(first[5:7]) - 1
    b = int(last[:4]) * 12 + int(last[5:7]) - 1
    return np.flatnonzero((tmi >= a) & (tmi <= b) & (clock.entry >= 0))


def e4_signals_at(sd: StudyData, clock: Clock, i: int, *, direction: str = "fwd", hop: int = 1,
                  edges: pd.DataFrame | None = None, col_of: Mapping[int, int] | None = None) -> dict[str, Any]:
    """E4 at formation row ``i``: hop-``hop`` raw signal at every target column in ALL with a
    structurally nonzero signal (priced origins only; the graph is built on all filers, so
    intermediate nodes of a hop-2/3 path need no price), its total path weight and the
    incremental signal (residual over the reached set on own SIC4 / SIC2 leave-one-out
    momentum, own S-return and, for hop > 1, the lower-hop signals)."""
    col_of = col_of if col_of is not None else sd.pb.col_of_cik()
    a = sd.e4_active(clock, i) if edges is None else edges
    if direction == "fwd":
        o_cik, t_cik = a["customer_cik"].to_numpy(np.int64), a["supplier_cik"].to_numpy(np.int64)
    else:
        o_cik, t_cik = a["supplier_cik"].to_numpy(np.int64), a["customer_cik"].to_numpy(np.int64)
    w = a["w"].to_numpy(np.float64)
    nodes = np.unique(np.r_[o_cik, t_cik])
    nix = {c: k for k, c in enumerate(nodes.tolist())}
    n = len(nodes)
    s = np.full(n, np.nan)
    ncol = np.array([col_of.get(int(c), -1) for c in nodes])
    okc = ncol >= 0
    s[okc] = clock.sret[i, ncol[okc]]
    bok = np.zeros(n, bool)
    bok[okc] = np.isfinite(clock.beta[i, ncol[okc]])
    origin_ok = np.isfinite(s) & bok
    tgt_ok = np.zeros(n, bool)
    tgt_ok[okc] = clock.all_[i, ncol[okc]]
    import scipy.sparse as sps

    ri = np.array([nix[c] for c in o_cik], np.int64)
    ci = np.array([nix[c] for c in t_cik], np.int64)
    keep_e = ri != ci
    W = sps.csr_matrix((w[keep_e], (ri[keep_e], ci[keep_e])), shape=(n, n))
    P = path_matrices_sparse(W, hop)
    raws, tws = [], []
    for h in range(1, hop + 1):
        sg, tw = hop_signal(P[h - 1], s, origin_ok)
        raws.append(sg)
        tws.append(tw)
    reached = tgt_ok & (tws[-1] > 0)
    cols = ncol
    X = [clock.loo4[i, np.maximum(cols, 0)], clock.loo2[i, np.maximum(cols, 0)], np.where(okc, s, np.nan)] + raws[:-1]
    inc = residualise_reached(raws[-1], X, reached)
    sel = np.flatnonzero(reached)
    return {"cols": cols[sel], "raw": raws[-1][sel], "inc": inc[sel], "tw": tws[-1][sel], "n_reached": int(len(sel)),
            "n_edges": int(len(w)), "n_nodes": int(n)}


def e5_signals_at(sd: StudyData, clock: Clock, i: int, *, direction: str = "fwd", hop: int = 1,
                  W: np.ndarray | None = None, universe: str = "ALL") -> dict[str, Any]:
    """E5 at formation row ``i``: industry shocks (EW S-return of the industry's priced ALL
    firms), hop signals on the zero-diagonal BEA graph of data year T.year - 2, units =
    industries with >= 3 ALL firms, the incremental signal = residual over the reached set on
    the lower-hop signals and the own-industry shock. ``universe='LIQ'`` keeps the ALL-based
    signal and re-forms units / constituents from liquid firms (D4)."""
    yr = int(clock.dates[i].year) - 2
    W = sd.W(yr) if W is None else W
    if direction == "rev":
        W = W.T.copy()
    K = len(sd.codes)
    ind = clock.ind[i]
    allm = clock.all_[i] & (ind >= 0)
    sr = clock.sret[i]
    shock = np.full(K, np.nan)
    n_all = np.bincount(ind[allm], minlength=K)
    okf = allm & np.isfinite(sr)
    n_ok = np.bincount(ind[okf], minlength=K)
    tot = np.bincount(ind[okf], weights=sr[okf], minlength=K)
    shock[n_ok > 0] = tot[n_ok > 0] / n_ok[n_ok > 0]
    origin_ok = n_ok > 0
    P = path_matrices(W, hop)
    raws, tws = [], []
    for h in range(1, hop + 1):
        sg, tw = hop_signal(P[h - 1], shock, origin_ok)
        raws.append(sg)
        tws.append(tw)
    units = n_all >= 3
    reached = units & (tws[-1] > 0)
    inc = residualise_reached(raws[-1], [*raws[:-1], shock], reached)
    if universe == "LIQ":
        lm = allm & clock.liq[i]
        units_l = np.bincount(ind[lm], minlength=K) >= 3
        sel = np.flatnonzero(reached & units_l)
        cons = {k: np.flatnonzero(lm & (ind == k)) for k in sel}
    else:
        sel = np.flatnonzero(reached)
        cons = {k: np.flatnonzero(allm & (ind == k)) for k in sel}
    return {"units": sel, "raw": raws[-1][sel], "inc": inc[sel], "tw": tws[-1][sel], "cons": cons,
            "n_reached": int(len(sel)), "shock": shock}


def d1_signals_at(sd: StudyData, clock: Clock, i: int) -> dict[str, Any]:
    cols = np.flatnonzero(clock.all_[i] & np.isfinite(clock.loo4[i]))
    return {"cols": cols, "raw": clock.loo4[i, cols], "inc": clock.loo4[i, cols], "n_reached": int(len(cols))}


def cell_signals(sd: StudyData, cd: CellDef, *, universe: str = "ALL", rng: np.random.Generator | None = None,
                 swap_shares: list[float] | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Per-date signal dicts of a cell over its span. With ``rng`` the graph is the N3 null:
    E4 = stratified double-edge swaps of each date's active graph (customer SIC4 strata, SIC3 /
    SIC2 fall-back), E5 = within-column permutation of each data year's W (one permutation
    per data year per draw); the signal is rebuilt and re-residualised."""
    clock = sd.monthly if cd.clock == "M" else sd.weekly()
    rows = _span_rows(clock, sd.pb.sessions, cd.first_target, cd.last_target)
    col_of = sd.pb.col_of_cik()
    out = []
    wperm: dict[int, np.ndarray] = {}
    for i in rows:
        if cd.edge == "E4":
            a = sd.e4_active(clock, i)
            if rng is not None and len(a):
                strata = edge_strata(a["customer_sic4"].fillna("").to_numpy())
                newc, share = rewire_stratified(a["customer_cik"].to_numpy(np.int64), a["supplier_cik"].to_numpy(np.int64),
                                                strata, rng)
                if swap_shares is not None:
                    swap_shares.append(share)
                a = a.assign(customer_cik=newc)
            out.append(e4_signals_at(sd, clock, i, direction=cd.direction, hop=cd.hop, edges=a, col_of=col_of))
        elif cd.edge == "E5":
            W = None
            if rng is not None:
                yr = int(clock.dates[i].year) - 2
                if yr not in wperm:
                    wperm[yr] = permute_within_columns(sd.W(yr), rng)
                W = wperm[yr]
            out.append(e5_signals_at(sd, clock, i, direction=cd.direction, hop=cd.hop, W=W, universe=universe))
        else:
            out.append(d1_signals_at(sd, clock, i))
    return rows, out


def cell_counts(sd: StudyData, cd: CellDef, universe: str = "ALL") -> tuple[np.ndarray, np.ndarray]:
    """Coverage counts per formation date: sorting units in the universe with a structurally
    nonzero signal (edge tables + price availability only; no return is read)."""
    rows, per = cell_signals(sd, cd, universe=universe)
    if cd.edge == "E5":
        c = np.array([int(len(p["units"])) for p in per])
    else:
        clock = sd.monthly if cd.clock == "M" else sd.weekly()
        c = np.array([int(len(p["cols"])) if universe == "ALL" else int(clock.liq[i, p["cols"]].sum())
                      for i, p in zip(rows, per, strict=True)])
    return rows, c


def trial_defs() -> dict[str, CellDef]:
    """The four pre-declared trials (N = 4)."""
    return {"T7": CellDef("T7", "E4", 1, first_target="2013-01", last_target="2022-12"),
            "T10": CellDef("T10", "E5", 1), "T11": CellDef("T11", "E5", 2), "T12": CellDef("T12", "E5", 3)}
