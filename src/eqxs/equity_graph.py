"""Study 15: economic-link graph, shock propagation and the hop axis (pre-declared v2).

Pre-declaration: ``reports/research/STUDY_15_PREDECLARATION_2026-10-11.md`` (v2, final).
Design review: ``reports/redteam/STUDY_15_DESIGN_REVIEW_2026-10-11.md`` (reference code under
``scripts/audit/study15_design_review_2026-10-11/``).

This module holds every piece of study-15 machinery that the generic study runner does not
have, and composes the rest from :mod:`eqxs.study_runner`,
:mod:`eqxs.equity_ml_grid`, :mod:`eqxs.equity_ml` and
:mod:`eqxs.equity_events`:

* edge tables with ``valid_from`` / ``valid_to`` (E1 SIC co-membership, E4 customer ->
  supplier, E5 BEA input-output) and the point-in-time E4 resolver (exact or unique >= 6-char
  prefix match against filer names valid at the filing date, exclusion rules);
* the exact simple-path propagation operator for hops 1..3 and its brute-force check;
* the market-model S-return (PIT 252-session beta, ``equity_ml_grid.name_betas``);
* the path-weight-weighted MEAN hop signal and its residualisation over the REACHED set only;
* the N3 nulls: stratified double-edge rewiring (E4) and within-column permutation (E5);
* quintile long/short books with per-name liquidity-tier costs (5 / 15 bps per side) at the
  firm level, both cost accountings, flat months (coverage dip rule);
* pooled 1-session impulse-response curves;
* the study driver (``python -m eqxs.equity_graph run``) and a thin audit-battery
  adapter (``python -m eqxs.equity_graph battery``).

Why a driver of its own (runner hooks that do not exist, implemented here at the minimum):
the runner's evaluator hard-codes deciles (``n_bins = 10``), one scalar cost per universe, no
flat months and one firm panel per horizon; study 15 needs quintiles, per-name tiered costs,
the dip rule and industry sorting units. The runner reaches this module through its plugin
hook (``study_runner.PLUGIN_MODULES["eqgraph"]``; the pre-declared YAML is frozen, so it has no
``plugin:`` key): ``scripts/validate/run_study.py config/studies/eqgraph.yaml`` calls
:func:`run_study_hook` and ``scripts/audit/standard_battery.py eqgraph`` :func:`battery_adapter`.
The driver registers the eqgraph signal,
panel and N3 hook, validates the YAML with ``study_runner.load_spec``, evaluates the cells
here and writes the report with ``study_runner.assemble`` / ``write_markdown`` /
``log_verdicts`` (plus a study-15 section).
"""

# ruff: noqa: N802, N803, N806, E501  (matrix names follow the maths; report strings)
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
TRAIN_END = pd.Timestamp("2023-01-01")  # study-15 computation bound (equities TRAIN end)
EDGES_DIR = REPO / "data_cache" / "links" / "edges"
ART_DIR = REPO / "data_cache" / "studies" / "eqgraph"
OPERATOR_VERSION = "exact-simple-paths-v1"
RESOLVER_VERSION = "e4-pit-resolver-v1"
EDGE_VERSION = "eqgraph-edges-v1"
N_BINS = 5
COST_LIQ = 0.0005
COST_REST = 0.0015
FLAT_BELOW = {"E4": 30, "E5": 20}
GATE_MEDIAN = {"E4": 60, "E5": 40}
SRET_WIN = 21
SRET_MIN_OBS = 15  # finite daily returns required in the 21-session S-return window
BETA_WIN, BETA_MINP = 252, 200
REWIRE_TRIES_PER_EDGE = 10
LESSOR_SIC = (set(range(6500, 6600)) | {6798})
LESSOR_NAME = re.compile(r"income\s*&\s*growth fund|income and growth fund|equipment fund|leasing fund", re.I)
D7_SICS = ("3721", "3728", "3761", "3812", "7373", "8711")


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
# 7. Quintile book with per-name tiered costs (firm level), flat months
# ===========================================================================
@dataclass
class UnitPanel:
    """Sorting units of one cell, sorted by date. Unit rows k = 0..n-1 (``bounds`` = date
    blocks); constituents of row k are ``cons_firm[cptr[k]:cptr[k+1]]`` (firm column ids) with
    their holding returns ``cons_ret``, cost per side ``cons_cost`` and ex-ante beta
    ``cons_beta``. ``y`` = unit holding return (EW of constituents). ``flat`` per date."""

    date_id: np.ndarray
    unit: np.ndarray
    y: np.ndarray
    cptr: np.ndarray
    cons_firm: np.ndarray
    cons_ret: np.ndarray
    cons_cost: np.ndarray
    cons_beta: np.ndarray
    bounds: np.ndarray
    flat: np.ndarray
    n_firms: int
    signals: dict[str, np.ndarray] = field(default_factory=dict)
    in_liq: np.ndarray | None = None

    @property
    def n_dates(self) -> int:
        return len(self.bounds) - 1

    def unit_beta(self) -> np.ndarray:
        out = np.full(len(self.y), np.nan)
        for k in range(len(self.y)):
            v = self.cons_beta[self.cptr[k]:self.cptr[k + 1]]
            v = v[np.isfinite(v)]
            out[k] = float(v.mean()) if len(v) else np.nan
        return out


@dataclass
class Book:
    gross: np.ndarray
    long_ret: np.ndarray
    short_ret: np.ndarray
    to_ew: np.ndarray  # two-sided turnover, EW rebalance accounting (unit-free)
    to_ee: np.ndarray  # entry/exit-only
    cost_ew: np.ndarray  # cost at 1x schedule, EW accounting (return units)
    cost_ee: np.ndarray
    n_long: np.ndarray
    n_short: np.ndarray
    beta_long: np.ndarray
    beta_short: np.ndarray
    flat: np.ndarray
    members: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None
    row_members: list[tuple[np.ndarray, np.ndarray]] | None = None

    def net(self, mult: float = 1.0, accounting: str = "ew") -> np.ndarray:
        c = self.cost_ew if accounting == "ew" else self.cost_ee
        return self.gross - mult * c

    def book_beta(self) -> np.ndarray:
        return self.beta_long - self.beta_short


def quintile_book(up: UnitPanel, pred: np.ndarray, *, n_bins: int = N_BINS, tie_seed: int | None = None,
                  keep_members: bool = False, cost_mult_vec: np.ndarray | None = None) -> Book:
    """Top-minus-bottom ``n_bins``-tile book over units, equal-weight units, equal-weight
    constituents inside a unit, drift by the holding return, seeded random tie-breaks
    (``equity_ml_grid.TIE_SEED`` by default, R3), costs per NAME by its tier. EW accounting
    charges sum_f c_f |w_new - w_drift|; entry/exit-only charges new names' weights and exited
    names' drifted weights. A flat date earns 0 and costs 0, and the next date enters from
    an empty book."""
    from eqxs import equity_ml_grid as g

    nd = up.n_dates
    keys = ("gross", "long_ret", "short_ret", "to_ew", "to_ee", "cost_ew", "cost_ee", "n_long", "n_short",
            "beta_long", "beta_short")
    res = {k: np.zeros(nd) for k in keys}
    res["beta_long"][:] = np.nan
    res["beta_short"][:] = np.nan
    trng = np.random.default_rng(g.TIE_SEED if tie_seed is None else tie_seed)
    w_s = np.zeros(up.n_firms)
    c_s = np.zeros(up.n_firms)
    prev = {"L": (np.zeros(0, np.int64), np.zeros(0), np.zeros(0)),
            "S": (np.zeros(0, np.int64), np.zeros(0), np.zeros(0))}
    members = [] if keep_members else None
    rmem = [] if keep_members else None
    cp = up.cptr
    for i in range(nd):
        a, b = int(up.bounds[i]), int(up.bounds[i + 1])
        n = b - a
        if up.flat[i] or n < n_bins:
            prev = {"L": (np.zeros(0, np.int64), np.zeros(0), np.zeros(0)),
                    "S": (np.zeros(0, np.int64), np.zeros(0), np.zeros(0))}
            if keep_members:
                members.append((np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64), np.zeros(0)))
                rmem.append((np.zeros(0, np.int64), np.zeros(0, np.int64)))
            continue
        p = pred[a:b]
        order = np.lexsort((trng.random(n), p))
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(1, n + 1)
        binv = np.ceil(rank * n_bins / n).astype(np.int64)
        mem = []
        for leg, sel in (("L", np.flatnonzero(binv == n_bins)), ("S", np.flatnonzero(binv == 1))):
            k = len(sel)
            rows = a + sel
            st_, en_ = cp[rows], cp[rows + 1]
            ln = en_ - st_
            nz = ln > 0
            tot = int(ln.sum())
            if tot:
                off = np.repeat(st_[nz] - np.r_[0, np.cumsum(ln[nz])[:-1]], ln[nz]) + np.arange(tot)
                f = up.cons_firm[off].astype(np.int64)
                w = np.repeat(1.0 / (k * ln[nz]), ln[nz])
                w = w / w.sum()
                r = np.nan_to_num(up.cons_ret[off])
                c = up.cons_cost[off]
                bb = up.cons_beta[off]
            else:
                f, w, r, c, bb = (np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
            if cost_mult_vec is not None:
                c = c * cost_mult_vec[i]
            pidx, pw, pc = prev[leg]
            w_s[pidx] = pw
            c_s[pidx] = pc
            w_old = w_s[f]
            d_new = np.abs(w - w_old)
            to_new = float(d_new.sum())
            cost_new = float((c * d_new).sum())
            entry = w_old == 0
            to_entry = float(w[entry].sum())
            cost_entry = float((c[entry] * w[entry]).sum())
            w_s[f] = 0.0
            to_exit = float(w_s[pidx].sum())
            cost_exit = float((w_s[pidx] * c_s[pidx]).sum())
            w_s[pidx] = 0.0
            c_s[pidx] = 0.0
            lr = float((w * r).sum()) if len(f) else 0.0
            if len(f) and (1.0 + lr) != 0:
                dw = w * (1.0 + r) / (1.0 + lr)
            else:
                dw = np.zeros(len(f))
            prev[leg] = (f, dw, c)
            fin = np.isfinite(bb)
            bv = float((w[fin] * bb[fin]).sum() / w[fin].sum()) if fin.any() else np.nan
            if leg == "L":
                res["long_ret"][i], res["n_long"][i], res["beta_long"][i] = lr, k, bv
            else:
                res["short_ret"][i], res["n_short"][i], res["beta_short"][i] = lr, k, bv
            res["to_ew"][i] += to_new + to_exit
            res["to_ee"][i] += to_entry + to_exit
            res["cost_ew"][i] += cost_new + cost_exit
            res["cost_ee"][i] += cost_entry + cost_exit
            mem += [f, w]
        if keep_members:
            members.append(tuple(mem))
            rmem.append((a + np.flatnonzero(binv == n_bins), a + np.flatnonzero(binv == 1)))
    res["gross"] = res["long_ret"] - res["short_ret"]
    flat = np.array([bool(up.flat[i]) or (up.bounds[i + 1] - up.bounds[i]) < n_bins for i in range(nd)])
    return Book(**res, flat=flat, members=members, row_members=rmem)


# ===========================================================================
# 8. Statistics wrappers (runner-shaped metric blocks)
# ===========================================================================
@dataclass
class Months:
    """One row per formation date of a cell: year of entry, held sessions, MktRF over the
    holding window, entry / exit dates and rows."""

    target_mi: np.ndarray
    sessions_held: np.ndarray
    mktrf: np.ndarray
    entry_date: pd.DatetimeIndex
    exit_date: pd.DatetimeIndex
    entry_row: np.ndarray
    exit_row: np.ndarray
    form_date: pd.DatetimeIndex

    @property
    def years(self) -> np.ndarray:
        return self.target_mi // 12


def unit_ic(pred: np.ndarray, y: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Within-date Spearman IC with AVERAGE ranks of both sides (tie-safe; E5 / E1 signals tie
    within industry by construction)."""
    out = np.full(len(bounds) - 1, np.nan)
    for i in range(len(bounds) - 1):
        a, b = int(bounds[i]), int(bounds[i + 1])
        if b - a < 3:
            continue
        p = pd.Series(pred[a:b]).rank().to_numpy()
        q = pd.Series(y[a:b]).rank().to_numpy()
        p -= p.mean()
        q -= q.mean()
        den = np.sqrt((p * p).sum() * (q * q).sum())
        out[i] = float((p * q).sum() / den) if den > 0 else np.nan
    return out


def book_metrics(bk: Book, up: UnitPanel, pred: np.ndarray, mo: Months, h: str, ref_cost: float) -> dict:
    from eqxs import equity_ml as ml
    from eqxs import equity_ml_grid as g

    P = g.PERIODS_PER_YEAR[h]
    yex = np.full(len(up.y), np.nan)
    for i in range(up.n_dates):
        a, b = int(up.bounds[i]), int(up.bounds[i + 1])
        yy = up.y[a:b]
        yex[a:b] = yy - np.nanmean(yy) if b > a else yy
    ics = unit_ic(pred, yex, up.bounds)
    ics[bk.flat] = np.nan
    nets = {"gross": bk.gross, "net_ew": bk.net(1, "ew"), "net_ee": bk.net(1, "ee"),
            "net_ew_2x": bk.net(2, "ew"), "net_ee_2x": bk.net(2, "ee")}
    st = {k: g.period_stats(v, mo.years, mo.sessions_held, mo.mktrf, h) for k, v in nets.items()}
    act = ~bk.flat
    out = {"ic": ml.ic_summary(ics), **st,
           "turnover_per_year_ew": float(bk.to_ew.mean() * P),
           "turnover_per_year_ee": float(bk.to_ee.mean() * P),
           "cost_weighted_turnover_per_year_ew": float(bk.cost_ew.mean() / ref_cost * P),
           "cost_weighted_turnover_per_year_ee": float(bk.cost_ee.mean() / ref_cost * P),
           "cost_ref_bps": ref_cost * 1e4,
           "mean_cost_per_side_bps_of_traded": float(bk.cost_ew.sum() / bk.to_ew.sum() * 1e4) if bk.to_ew.sum() > 0 else float("nan"),
           "incumbent_reweight_share_of_turnover": float(1 - bk.to_ee.sum() / bk.to_ew.sum()) if bk.to_ew.sum() > 0 else float("nan"),
           "break_even_cost_bps_ew": float(bk.gross.mean() / bk.to_ew.mean() * 1e4) if bk.to_ew.mean() > 0 else float("nan"),
           "break_even_cost_bps_ee": float(bk.gross.mean() / bk.to_ee.mean() * 1e4) if bk.to_ee.mean() > 0 else float("nan"),
           "break_even_multiple_of_tier_schedule_ew": float(bk.gross.mean() / bk.cost_ew.mean()) if bk.cost_ew.mean() > 0 else float("nan"),
           "cost_drag_ann_ew": float(bk.cost_ew.mean() * P),
           "legs": {"long_ann": float(bk.long_ret.mean() * P), "short_ann": float(bk.short_ret.mean() * P),
                    "mean_n_long": float(bk.n_long[act].mean()) if act.any() else 0.0,
                    "mean_n_short": float(bk.n_short[act].mean()) if act.any() else 0.0},
           "periods": int(len(bk.gross)), "flat_periods": int(bk.flat.sum()),
           "first_entry": str(mo.entry_date[0].date()), "last_entry": str(mo.entry_date[-1].date())}
    hb = bk.book_beta()
    with np.errstate(all="ignore"):
        out["ex_ante_book_beta_mean"] = float(np.nanmean(hb)) if np.isfinite(hb).any() else float("nan")
        out["ex_ante_book_beta_by_year"] = {int(y): (float(np.nanmean(hb[mo.years == y])) if np.isfinite(hb[mo.years == y]).any()
                                                    else float("nan")) for y in np.unique(mo.years)}
    return out


def n1_dual(up: UnitPanel, pred: np.ndarray, h: str, real: Mapping[str, float], *, n_draws: int,
            rng: np.random.Generator) -> dict[str, dict[str, Any]]:
    """N1: within-date permutation of the signal across the sorting units through the same
    book code; statistics gross (the gate), net EW 1x / 2x and net E/E 1x from the SAME books."""
    from eqxs import equity_events as ee
    from eqxs import equity_ml_grid as g

    labels = {"gross": (0.0, "ew"), "ew1": (1.0, "ew"), "ew2": (2.0, "ew"), "ee1": (1.0, "ee")}
    sh = {k: [] for k in labels}
    ar = {k: [] for k in labels}
    for _ in range(n_draws):
        bk = quintile_book(up, g.permute_within_dates(pred, up.bounds, rng))
        for k, (mlt, acc) in labels.items():
            x = bk.gross if k == "gross" else bk.net(mlt, acc)
            sh[k].append(g.ann_sharpe(x, h))
            ar[k].append(g.ann_mean(x, h))
    out = {}
    for k in labels:
        mde = ee.mde_from_null(ar[k])
        out[k] = {"draws": n_draws, "p": ee.null_p(sh[k], real[k]), "null_sharpe_mean": float(np.nanmean(sh[k])),
                  "null_sharpe_q95": float(np.nanquantile(sh[k], 0.95)), "null_ann_mean": float(np.nanmean(ar[k])),
                  "mde_ann_return": mde, "mde_excess_over_null_mean": float(mde - np.nanmean(ar[k]))}
    return out


# ===========================================================================
# 9. Cache key
# ===========================================================================
def cache_key(**parts: Any) -> str:
    """sha256 over the study-15 cache-key fields (edge-table content hash, resolver version,
    crosswalk hash, BEA vintage hash, edge type, hop, operator version, formation range,
    rewiring draw index, seed)."""
    s = json.dumps({k: (str(v) if not isinstance(v, (int, float, str, list, dict, type(None))) else v)
                    for k, v in sorted(parts.items())}, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()


def frame_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    h.update(",".join(map(str, df.columns)).encode())
    return h.hexdigest()


def ann_sr_months(x: np.ndarray) -> float:
    """Annualised Sharpe of a monthly series (x sqrt 12), equal to equity_ml_grid.ann_sharpe(x, 'M')."""
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / sd * math.sqrt(12.0)) if sd > 0 else float("nan")


# ===========================================================================
# 9b. French daily factors bounded at the TRAIN end
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
# 10. Data: price base (prices, universe, holding returns, betas, S-returns)
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


def _assert_before(df: pd.DataFrame, what: str, end: pd.Timestamp = TRAIN_END) -> None:
    from eqxs.study_runner import assert_before

    assert_before(df, end, what)


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
# 11. Data: FSN sub (PIT SIC, 10-K list, co-registrant families)
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


def sic_matrix(pb: PriceBase, sub: pd.DataFrame, profiles: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """(n_me x cols) SIC4 string codes as of each formation close: the SIC of the latest FSN
    filing accepted before the close; before the CIK's first FSN filing the first FSN SIC
    (``backfill``), else the current profile SIC (``profile``). Source codes: 0 = FSN as of
    the filing, 1 = backfill, 2 = profile, -1 = none."""
    from eqxs import equity_ml as ml

    close_ns = ml.close_ts_utc_ns(pb.me_dates)
    s = sub[sub["sic"].str.len() == 4].loc[:, ["cik", "accepted_utc", "sic"]].dropna()
    s = s.assign(t=pd.DatetimeIndex(s["accepted_utc"]).asi8).sort_values(["cik", "t"])
    by = {int(c): (grp["t"].to_numpy(), grp["sic"].to_numpy()) for c, grp in s.groupby("cik")}
    prof = dict(zip(profiles["cik"].astype(int), profiles["sic"].astype(str).str.zfill(4), strict=False))
    n_me, k = len(pb.me_rows), len(pb.col_cik)
    out = np.full((n_me, k), "", dtype=object)
    src = np.full((n_me, k), -1, np.int8)
    for j, c in enumerate(pb.col_cik):
        if c < 0:
            continue
        if int(c) in by:
            t, v = by[int(c)]
            pos = np.searchsorted(t, close_ns, side="right") - 1
            out[:, j] = np.where(pos >= 0, v[np.maximum(pos, 0)], v[0])
            src[:, j] = np.where(pos >= 0, 0, 1)
        elif prof.get(int(c), "") not in ("", "0000", "None"):
            out[:, j] = prof[int(c)]
            src[:, j] = 2
    diag = {"share_firm_months_in_ALL_by_source": {
        lab: float((src[pb.all_] == code).mean()) for lab, code in (("fsn_as_of_filing", 0), ("backfill_first_fsn", 1),
                                                                    ("profile_current", 2), ("none", -1))}}
    return out, src, diag


# ===========================================================================
# 12. Edge tables
# ===========================================================================
E4_FORMS = ("10-K", "10-K/A", "10-KT")
E4_TAGS = ("ConcentrationRiskPercentage1", "ConcentrationRiskPercentage")
ORIGINAL_10K = ("10-K", "10-KT")
FAR = pd.Timestamp("2100-01-01")


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
# 13. Study data, clocks, signals and unit panels
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


def unit_panel(sd: StudyData, clock: Clock, rows: np.ndarray, per_date: Sequence[dict[str, Any]], *, kind: str,
               universe: str, flat_below: int, signal_keys: Sequence[str] = ("inc", "raw")) -> UnitPanel:
    """Assemble a :class:`UnitPanel` from per-date signal dicts (firm units: ``cols``;
    industry units: ``units`` + ``cons``). Units with a NaN incremental signal are dropped
    (never assigned a residual). Cost per side by tier: 5 bps liquid half, 15 bps the rest of
    ALL; the LIQ universe (D4) is 5 bps throughout."""
    did, unit, y, cptr, cf, cr, cc, cb = [], [], [], [0], [], [], [], []
    sig = {k: [] for k in signal_keys}
    flat = np.zeros(len(rows), bool)
    liq_flag = []
    for d, (i, pdict) in enumerate(zip(rows, per_date, strict=True)):
        if kind == "firm":
            cols = pdict["cols"]
            keep = np.isfinite(pdict["inc"])
            if universe == "LIQ":
                keep &= clock.liq[i, cols]
            sel = np.flatnonzero(keep)
            groups = [np.array([cols[k]]) for k in sel]
            uid = cols[sel]
        else:
            keep = np.isfinite(pdict["inc"])
            sel = np.flatnonzero(keep)
            groups = [pdict["cons"][pdict["units"][k]] for k in sel]
            uid = pdict["units"][sel]
        if len(sel) < flat_below:
            flat[d] = True
        for k, gcols, u in zip(sel, groups, uid, strict=True):
            r = np.nan_to_num(clock.hret[i, gcols])
            did.append(d)
            unit.append(int(u))
            y.append(float(r.mean()) if len(r) else 0.0)
            cf.append(gcols)
            cr.append(r)
            cc.append(np.full(len(gcols), COST_LIQ) if universe == "LIQ" else np.where(clock.liq[i, gcols], COST_LIQ, COST_REST))
            cb.append(clock.beta[i, gcols].astype(np.float64))
            cptr.append(cptr[-1] + len(gcols))
            liq_flag.append(bool(clock.liq[i, gcols].all()))
            for kk in signal_keys:
                sig[kk].append(float(pdict[kk][k]))
    did_a = np.asarray(did, np.int64)
    b = np.searchsorted(did_a, np.arange(len(rows) + 1), side="left")
    def cat(L: list, dt: Any) -> np.ndarray:
        return np.concatenate(L).astype(dt) if L else np.zeros(0, dt)

    return UnitPanel(did_a, np.asarray(unit, np.int64), np.asarray(y), np.asarray(cptr, np.int64),
                     cat(cf, np.int64), cat(cr, np.float64), cat(cc, np.float64), cat(cb, np.float64), b, flat,
                     int(len(sd.pb.col_cik)), {k: np.asarray(v) for k, v in sig.items()}, np.asarray(liq_flag))


def months_of(sd: StudyData, clock: Clock, rows: np.ndarray) -> Months:
    from eqxs import equity_ml as ml

    pb = sd.pb
    e, x = clock.entry[rows], clock.exit_[rows]
    mk = compound_rows(pb.mktrf + pb.rf, e, x) - compound_rows(pb.rf, e, x)
    return Months(ml.month_index(pb.sessions[e]), (x - e).astype(np.int64), mk, pb.sessions[e], pb.sessions[x], e, x,
                  clock.dates[rows])


def compound_rows(v: np.ndarray, entry: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    lc = np.r_[0.0, np.cumsum(np.log1p(np.asarray(v, dtype=np.float64)))]
    return np.expm1(lc[exit_ + 1] - lc[entry + 1])


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


def diagnostic_defs() -> dict[str, CellDef]:
    """Sort-book diagnostics (never trials): D1 control; D2 raw signal; D5 weekly gross;
    D6 transposed direction; D3 E4 hops 2 / 3 (book only if the median count reaches 30).
    D4 = the LIQ universe block of every trial; D7 / D8 event studies and D9 (2x cost,
    break-even) are computed separately."""
    T = trial_defs()
    out: dict[str, CellDef] = {"D1": CellDef("D1", "E1", 0, role="diagnostic")}
    for k, t in T.items():
        out[f"D2_{k}"] = CellDef(f"D2_{k}", t.edge, t.hop, "fwd", "raw", "M", t.first_target, t.last_target, "diagnostic")
    for h in (2, 3):
        out[f"D3_H{h}"] = CellDef(f"D3_H{h}", "E4", h, first_target="2013-01", last_target="2022-12", role="diagnostic")
    for k, t in T.items():
        out[f"D5_{k}"] = CellDef(f"D5_{k}", t.edge, t.hop, "fwd", "incremental", "W", t.first_target, t.last_target,
                                 "diagnostic")
    for k, t in T.items():
        out[f"D6_{k}"] = CellDef(f"D6_{k}", t.edge, t.hop, "rev", "incremental", "M", t.first_target, t.last_target,
                                 "diagnostic")
    return out


# ===========================================================================
# 14. Cell evaluation (runner-shaped universe blocks)
# ===========================================================================
FAC_COLS = ("Mkt-RF", "SMB", "HML", "MOM")


def daily_book_spread(R: np.ndarray, bk: Book, mo: Months) -> tuple[np.ndarray, np.ndarray]:
    """Daily long-minus-short spread of a firm-level buy-and-hold book (members with weights)
    over each holding window, and the session rows."""
    rets, rows = [], []
    for mem, e, x in zip(bk.members, mo.entry_row, mo.exit_row, strict=True):
        n = int(x - e)
        if len(mem) < 4 or len(mem[0]) == 0:
            rets.append(np.zeros(n))
            rows.append(np.arange(e + 1, x + 1))
            continue
        legs = []
        for f, w in ((mem[0], mem[1]), (mem[2], mem[3])):
            r = np.nan_to_num(R[e + 1:x + 1][:, f].astype(np.float64))
            v = np.vstack([w[None, :], w[None, :] * np.cumprod(1.0 + r, axis=0)])
            s_ = v.sum(1)
            legs.append(s_[1:] / s_[:-1] - 1.0)
        rets.append(legs[0] - legs[1])
        rows.append(np.arange(e + 1, x + 1))
    return np.concatenate(rets), np.concatenate(rows)


def factor_attribution_x(spread: np.ndarray, F: pd.DataFrame, years: np.ndarray, *, nw_lags: int = 5) -> list[dict]:
    """Per-year OLS of the daily spread on the columns of ``F`` (FF3 + MOM + the D1 industry-
    momentum spread 'IMOM'), NW standard errors, alpha x 252, contributions = loading x summed
    factor return."""
    out = []
    y = np.asarray(spread, dtype=np.float64)
    X0 = F.to_numpy(np.float64)
    cols = list(F.columns)
    for yr in np.unique(years):
        m = (years == yr) & np.isfinite(y) & np.isfinite(X0).all(1)
        if m.sum() < 30:
            continue
        X = np.column_stack([np.ones(m.sum()), X0[m]])
        inv = np.linalg.pinv(X.T @ X)
        b = inv @ X.T @ y[m]
        e = y[m] - X @ b
        xe = X * e[:, None]
        S = xe.T @ xe
        for lag in range(1, nw_lags + 1):
            gm = xe[lag:].T @ xe[:-lag]
            S += (1 - lag / (nw_lags + 1)) * (gm + gm.T)
        se = np.sqrt(np.maximum(np.diag(inv @ S @ inv), 0))
        ss = float(((y[m] - y[m].mean()) ** 2).sum())
        row = {"year": int(yr), "days": int(m.sum()), "sum_spread": float(y[m].sum()), "alpha_ann": float(b[0] * 252),
               "alpha_t": float(b[0] / se[0]) if se[0] > 0 else float("nan"), "r2": 1 - float(e @ e) / ss if ss > 0 else float("nan")}
        for j, f in enumerate(cols, 1):
            row[f"b_{f}"] = float(b[j])
            row[f"t_{f}"] = float(b[j] / se[j]) if se[j] > 0 else float("nan")
            row[f"contrib_{f}"] = float(b[j] * X0[m, j - 1].sum())
        out.append(row)
    return out


def cell_verdict_full(net_sr: float, n1_p: float, n2_p: float, per_year: list, n3_p: float | None) -> dict:
    from eqxs import equity_events as ee

    v = ee.event_study_verdict(net_sr, n1_p, n2_p, per_year, years_rule="ex_best_year_positive", min_sessions=63)
    fails = list(v["fails"])
    if n3_p is not None and not n3_p < 0.05:
        fails.append("N3_p>=0.05")
    return {"verdict": "PASS" if not fails else "REJECT", "fails": fails, "years": v["years"]}


def evaluate_book(sd: StudyData, cd: CellDef, up: UnitPanel, pred: np.ndarray, mo: Months, *, universe: str,
                  rng: np.random.Generator, n1_draws: int, n2_draws: int, n3_draws: int, gross_only: bool = False,
                  imom: pd.Series | None = None, attribution: bool = False, deadline: float | None = None) -> dict:
    """One universe block: metrics, N1 (gross gate + net diagnostics), N2 (net EW 1x / 2x,
    E/E, 12-month block), N3 (trials), verdict at 1x and 2x cost, ex-ante beta, the
    index-hedged diagnostic and (trial universe) per-year FF3 + MOM + IMOM attribution. RNG
    consumption order: N1, N2 x 4, N3."""
    from eqxs import equity_ml_grid as g

    h = "W" if cd.clock == "W" else "M"
    ref = COST_LIQ if universe == "LIQ" else COST_REST
    keep = attribution and sd is not None
    bk = quintile_book(up, pred, keep_members=keep)
    ens = book_metrics(bk, up, pred, mo, h, ref)
    real = {"gross": ens["gross"]["sharpe"], "ew1": ens["net_ew"]["sharpe"], "ew2": ens["net_ew_2x"]["sharpe"],
            "ee1": ens["net_ee"]["sharpe"]}
    t0 = time.time()
    n1 = n1_dual(up, pred, h, real, n_draws=n1_draws, rng=rng)
    base = bk.gross if gross_only else bk.net(1, "ew")
    n2 = {"ew1": g.n2_bootstrap(base, h, rng=rng, n_draws=n2_draws),
          "ew2": g.n2_bootstrap(bk.gross if gross_only else bk.net(2, "ew"), h, rng=rng, n_draws=n2_draws),
          "ee1": g.n2_bootstrap(bk.gross if gross_only else bk.net(1, "ee"), h, rng=rng, n_draws=n2_draws),
          "ew1_12m_block": g.n2_bootstrap(base, h, rng=rng, n_draws=n2_draws, block=g.N2_BLOCK_12M[h])}
    out: dict[str, Any] = {"universe": universe, "role": "trial" if universe == "ALL" else "sensitivity", "ENS": ens,
                           "N1": n1, "N2": n2, "gross_only": gross_only}
    n3p = None
    if n3_draws:
        sh, swaps, nreach = [], [], []
        for _ in range(n3_draws):
            if deadline is not None and time.time() > deadline:
                raise TimeoutError("cell wall-clock cap exceeded during N3")
            sw: list[float] = []
            rows, per = cell_signals(sd, cd, universe=universe, rng=rng, swap_shares=sw)
            up3 = unit_panel(sd, sd.monthly if cd.clock == "M" else sd.weekly(), rows, per,
                             kind="industry" if cd.edge == "E5" else "firm", universe=universe,
                             flat_below=FLAT_BELOW["E5" if cd.edge == "E5" else "E4"])
            b3 = quintile_book(up3, up3.signals["inc" if cd.signal == "incremental" else "raw"])
            sh.append(g.ann_sharpe(b3.gross, h))
            swaps.append(float(np.mean(sw)) if sw else float("nan"))
            nreach.append(float(np.mean(np.diff(up3.bounds))))
        from eqxs import equity_events as ee

        n3p = ee.null_p(sh, real["gross"])
        out["N3"] = {"draws": n3_draws, "p": n3p, "statistic": "gross annualised Sharpe",
                     "kind": "stratified double-edge swaps (customer SIC4 -> SIC3 -> SIC2)" if cd.edge == "E4"
                     else "within-column permutation of W (per data year)",
                     "null_sharpe_mean": float(np.nanmean(sh)), "null_sharpe_q95": float(np.nanquantile(sh, 0.95)),
                     "swap_share_mean": float(np.nanmean(swaps)) if cd.edge == "E4" else None,
                     "swap_share_min": float(np.nanmin(swaps)) if cd.edge == "E4" else None,
                     "swap_share_per_draw": swaps if cd.edge == "E4" else None,
                     "mean_units_per_date_null": float(np.mean(nreach))}
    out["null_seconds"] = round(time.time() - t0, 1)
    v1 = cell_verdict_full(ens["gross" if gross_only else "net_ew"]["sharpe"], n1["gross"]["p"], n2["ew1"]["p"],
                           ens["gross" if gross_only else "net_ew"]["per_year"], n3p)
    v2 = cell_verdict_full(ens["gross" if gross_only else "net_ew_2x"]["sharpe"], n1["gross"]["p"], n2["ew2"]["p"],
                           ens["gross" if gross_only else "net_ew_2x"]["per_year"], n3p)
    def lab(x: str) -> str:
        return x if cd.role == "trial" and universe == "ALL" else f"info:{x}"

    out.update({"verdict": lab(v1["verdict"]), "fails": v1["fails"], "years_rule": v1["years"],
                "verdict_2x_cost": lab(v2["verdict"]), "fails_2x_cost": v2["fails"],
                "verdict_predeclared": lab("PASS" if v1["verdict"] == "PASS" and v2["verdict"] == "PASS" else "REJECT"),
                "beta_flag": bool(abs(ens["net_ew"]["beta"]) > 0.25),
                "ex_ante_beta_flag_years": [y for y, b in ens["ex_ante_book_beta_by_year"].items() if np.isfinite(b) and abs(b) > 0.25]})
    hb = np.nan_to_num(bk.book_beta())
    hedged = bk.net(1, "ew") - hb * mo.mktrf - COST_LIQ * np.abs(np.diff(np.r_[0.0, hb]))
    out["beta_hedged_net_ew"] = {"sharpe": g.ann_sharpe(hedged, h), "ann_return": g.ann_mean(hedged, h),
                                 "hedge_cost_bps_per_side": COST_LIQ * 1e4}
    if keep:
        spread, srow = daily_book_spread(sd.pb.R, bk, mo)
        fac = load_ff_train().reindex(sd.pb.sessions).iloc[srow].loc[:, list(FAC_COLS)].reset_index(drop=True)
        yrs = np.asarray(sd.pb.sessions[srow].year)
        out["factor_attribution"] = g.factor_attribution(spread, fac.loc[:, list(FAC_COLS)], yrs)
        if imom is not None:
            fac["IMOM"] = imom.reindex(srow).to_numpy()
            out["factor_attribution_with_imom"] = factor_attribution_x(spread, fac, yrs)
        out["_daily_spread"] = pd.Series(spread, index=srow)
    return out


def run_sort_cell(sd: StudyData, cd: CellDef, spec: Mapping[str, Any], *, gate: Mapping[str, Any] | None,
                  imom: pd.Series | None, cap_minutes: float, artefacts: Path, universes: Sequence[str]) -> dict:
    """Signals, unit panels, the coverage gate / dip rule, predictions + constituents artefacts and
    the universe blocks of one sort cell."""
    from eqxs import study_runner as sr

    t0 = time.time()
    deadline = t0 + cap_minutes * 60
    clock = sd.monthly if cd.clock == "M" else sd.weekly()
    kind = "industry" if cd.edge == "E5" else "firm"
    fb = FLAT_BELOW["E5" if cd.edge == "E5" else "E4"]
    res: dict[str, Any] = {"horizon": "W" if cd.clock == "W" else "M", "status": "RUN",
                           "signal_info": {"edge": cd.edge, "hop": cd.hop, "direction": cd.direction, "signal": cd.signal,
                                           "clock": cd.clock, "span": [cd.first_target, cd.last_target],
                                           "operator": OPERATOR_VERSION, "flat_below": fb}}
    rows, per = cell_signals(sd, cd, universe="ALL")
    mo = months_of(sd, clock, rows)
    nulls = spec["nulls"]
    n1d, n2d = int(nulls["N1"]["draws"]), int(nulls["N2"]["draws"])
    n3d = int((nulls.get("N3") or {}).get("draws", 0)) if cd.role == "trial" else 0
    for u in universes:
        per_u = per
        if cd.edge == "E5" and u == "LIQ":
            _, per_u = cell_signals(sd, cd, universe="LIQ")
        up = unit_panel(sd, clock, rows, per_u, kind=kind, universe=u, flat_below=fb)
        if gate is not None and u == "ALL" and cd.role == "trial":
            gm = np.asarray(gate["flat_mask"], bool)
            if len(gm) == up.n_dates and not np.array_equal(gm, up.flat):
                res.setdefault("notes", []).append("flat mask recomputed from finite incremental signals differs from the committed gate on "
                                                   f"{int((gm != up.flat).sum())} months; the committed gate mask is used")
                up.flat = gm
        pred = up.signals["inc" if cd.signal == "incremental" else "raw"]
        rng = sr.cell_rng(spec, cd.cell_id, u)
        blk = evaluate_book(sd, cd, up, pred, mo, universe=u, rng=rng, n1_draws=n1d, n2_draws=n2d,
                            n3_draws=n3d if u == "ALL" else 0, gross_only=cd.clock == "W", imom=imom,
                            attribution=(cd.role == "trial" and u == "ALL") or (cd.cell_id == "D1" and u == "ALL"),
                            deadline=deadline)
        blk["coverage"] = {"units_per_date_median": float(np.median(np.diff(up.bounds))), "units_per_date_min": int(np.diff(up.bounds).min()),
                           "flat_months": int(up.flat.sum()), "flat_below": fb,
                           "mean_constituents_per_unit": float(np.mean(np.diff(up.cptr))) if len(up.y) else 0.0}
        res[u] = blk
        save_cell_artefacts(sd, clock, cd, rows, up, pred, artefacts, u)
        if time.time() > deadline:
            raise TimeoutError(f"cell wall-clock cap {cap_minutes} min exceeded")
    res["predictions"] = str((artefacts / "predictions" / f"{cd.cell_id}.parquet").relative_to(REPO))
    res["elapsed_s"] = round(time.time() - t0, 1)
    return res


def save_cell_artefacts(sd: StudyData, clock: Clock, cd: CellDef, rows: np.ndarray, up: UnitPanel, pred: np.ndarray,
                        artefacts: Path, universe: str) -> None:
    from eqxs import equity_ml as ml

    d = artefacts / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    form = clock.dates[rows][up.date_id]
    ent = sd.pb.sessions[clock.entry[rows]][up.date_id]
    ext = sd.pb.sessions[clock.exit_[rows]][up.date_id]
    fr = pd.DataFrame({"date": form, "ts": pd.DatetimeIndex(form).tz_localize("UTC"), "date_id": up.date_id,
                       "target_mi": ml.month_index(ent), "entry_date": ent, "exit_date": ext, "firm": up.unit,
                       "in_liq": up.in_liq if up.in_liq is not None else False, "y_raw": up.y, "pred_ENS": pred,
                       "pred_RAW": up.signals.get("raw", pred), "pred_INC": up.signals.get("inc", pred),
                       "n_cons": np.diff(up.cptr), "beta_exante": up.unit_beta()})
    yex = np.full(len(up.y), np.nan)
    for i in range(up.n_dates):
        a, b = int(up.bounds[i]), int(up.bounds[i + 1])
        if b > a:
            yex[a:b] = up.y[a:b] - up.y[a:b].mean()
    fr["y_ex_all"] = yex
    fr["y_ex_liq"] = yex
    _assert_before(fr, f"predictions {cd.cell_id}")
    suf = "" if universe == "ALL" else f"_{universe}"
    fr.to_parquet(d / f"{cd.cell_id}{suf}.parquet", index=False)
    cons = pd.DataFrame({"row": np.repeat(np.arange(len(up.y)), np.diff(up.cptr)), "firm_col": up.cons_firm,
                         "ret": up.cons_ret, "cost": up.cons_cost, "beta": up.cons_beta})
    cons.to_parquet(d / f"{cd.cell_id}{suf}_constituents.parquet", index=False)
    pd.DataFrame({"date_id": np.arange(up.n_dates), "flat": up.flat, "form_date": clock.dates[rows],
                  "ts": pd.DatetimeIndex(clock.dates[rows]).tz_localize("UTC"), "entry_row": clock.entry[rows],
                  "exit_row": clock.exit_[rows]}).to_parquet(d / f"{cd.cell_id}{suf}_dates.parquet", index=False)


def load_cell_panel(artefacts: Path, cell_id: str, universe: str = "ALL") -> tuple[UnitPanel, pd.DataFrame, pd.DataFrame]:
    """Rebuild a cell's UnitPanel from its artefacts through the tripwire (battery adapter)."""
    from eqxs.data_zones import EQUITIES_ZONES_PATH, safe_load_parquet

    d = artefacts / "predictions"
    suf = "" if universe == "ALL" else f"_{universe}"
    def rd(n: str) -> pd.DataFrame:
        return safe_load_parquet(d / n, max_end_date=TRAIN_END, zone_config=EQUITIES_ZONES_PATH,
                                 caller_intent=f"eqgraph.load_cell_panel.{cell_id}")

    fr = rd(f"{cell_id}{suf}.parquet")
    dates = rd(f"{cell_id}{suf}_dates.parquet")
    import pyarrow.parquet as pq

    cons = pq.read_table(d / f"{cell_id}{suf}_constituents.parquet").to_pandas()  # no dates in this table
    nd = len(dates)
    did = fr["date_id"].to_numpy(np.int64)
    b = np.searchsorted(did, np.arange(nd + 1), side="left")
    cptr = np.r_[0, np.cumsum(fr["n_cons"].to_numpy(np.int64))]
    up = UnitPanel(did, fr["firm"].to_numpy(np.int64), fr["y_raw"].to_numpy(np.float64), cptr,
                   cons["firm_col"].to_numpy(np.int64), cons["ret"].to_numpy(np.float64), cons["cost"].to_numpy(np.float64),
                   cons["beta"].to_numpy(np.float64), b, dates["flat"].to_numpy(bool),
                   int(cons["firm_col"].max()) + 1 if len(cons) else 1,
                   {"inc": fr["pred_INC"].to_numpy(np.float64), "raw": fr["pred_RAW"].to_numpy(np.float64)},
                   fr["in_liq"].to_numpy(bool))
    return up, fr, dates


# ===========================================================================
# 15. Impulse responses (1-session, lags 1..30, pooled; never choose anything)
# ===========================================================================
IR_LAGS = 30


def impulse_curve(sd: StudyData, cd: CellDef, rows: np.ndarray, per: Sequence[dict[str, Any]], *,
                  signal: str = "inc") -> tuple[np.ndarray, np.ndarray]:
    """Per formation date d and lag k (sessions t+1..t+30 after the formation close t): the
    cross-sectional slope of the downstream units' abnormal return (market-model residual with
    the formation beta; a unit = EW of its constituents) on the z-scored upstream signal.
    Returns (dates x lags) slopes and the pooled curve (mean over dates)."""
    pb = sd.pb
    clock = sd.monthly
    kind = "industry" if cd.edge == "E5" else "firm"
    B = np.full((len(rows), IR_LAGS), np.nan)
    n = len(pb.sessions)
    for d, (i, p) in enumerate(zip(rows, per, strict=True)):
        x = np.asarray(p[signal], dtype=np.float64)
        ok = np.isfinite(x)
        if ok.sum() < 6:
            continue
        t = int(clock.rows[i])
        hi = min(t + IR_LAGS, n - 1)
        if hi <= t:
            continue
        rr = pb.R[t + 1:hi + 1].astype(np.float64)
        ar = rr - pb.rf[t + 1:hi + 1, None] - clock.beta[i].astype(np.float64)[None, :] * pb.mktrf[t + 1:hi + 1, None]
        if kind == "firm":
            Y = ar[:, p["cols"][ok]]
        else:
            Y = np.column_stack([np.nanmean(ar[:, p["cons"][u]], axis=1) if len(p["cons"][u]) else np.full(len(ar), np.nan)
                                 for u in p["units"][ok]])
        z = x[ok]
        z = (z - z.mean()) / (z.std() if z.std() > 0 else 1.0)
        for k in range(Y.shape[0]):
            y = Y[k]
            m = np.isfinite(y)
            if m.sum() < 6:
                continue
            zz = z[m] - z[m].mean()
            yy = y[m] - y[m].mean()
            den = float(zz @ zz)
            B[d, k] = float(zz @ yy / den) if den > 0 else np.nan
    return B, np.nanmean(B, axis=0)


def curve_nulls(sd: StudyData, cd: CellDef, B: np.ndarray, *, n3_draws: int, bb_draws: int, seed: int,
                signal: str = "inc", block: int = 3) -> dict[str, Any]:
    """(a) the cell's N3 null (rewired graph, signal rebuilt and re-residualised): per-lag
    2.5 / 97.5 % band of the pooled curve; (b) centred circular block bootstrap of the per-date
    slopes (3 formation months ~ 63 sessions per block): per-lag band and two-sided p."""
    from eqxs import equity_events as ee

    rng = np.random.default_rng([seed, 15, 1])
    curve = np.nanmean(B, axis=0)
    out: dict[str, Any] = {"curve": curve.tolist(), "cum_curve": np.nancumsum(curve).tolist()}
    if n3_draws and cd.edge in ("E4", "E5"):
        nc = []
        for _ in range(n3_draws):
            rows, per = cell_signals(sd, cd, rng=rng)
            _, c = impulse_curve(sd, cd, rows, per, signal=signal)
            nc.append(c)
        nc = np.asarray(nc)
        out["n3_lo"], out["n3_hi"] = np.nanquantile(nc, 0.025, axis=0).tolist(), np.nanquantile(nc, 0.975, axis=0).tolist()
        out["n3_p_two_sided"] = [float((np.sum(np.abs(nc[:, k] - np.nanmean(nc[:, k])) >= abs(curve[k] - np.nanmean(nc[:, k]))) + 1)
                                       / (len(nc) + 1)) for k in range(B.shape[1])]
        cum = np.nancumsum(nc, axis=1)
        out["n3_cum21_lo_hi"] = [float(np.nanquantile(cum[:, 20], 0.025)), float(np.nanquantile(cum[:, 20], 0.975))]
        out["n3_draws"] = n3_draws
    lo, hi, pv = [], [], []
    for k in range(B.shape[1]):
        x = B[:, k]
        x = x[np.isfinite(x)]
        if len(x) < 10:
            lo.append(np.nan)
            hi.append(np.nan)
            pv.append(np.nan)
            continue
        m = x.mean()
        bm = ee.circular_block_bootstrap_means(x - m, block=block, n_draws=bb_draws, rng=rng)
        lo.append(float(np.quantile(bm, 0.025)))
        hi.append(float(np.quantile(bm, 0.975)))
        pv.append(float((np.sum(np.abs(bm) >= abs(m)) + 1) / (bb_draws + 1)))
    out.update({"bb_lo": lo, "bb_hi": hi, "bb_p_two_sided": pv, "bb_draws": bb_draws, "bb_block_months": block})
    c21 = np.nansum(B[:, :21], axis=1)
    c21 = c21[np.isfinite(c21)]
    bm = ee.circular_block_bootstrap_means(c21 - c21.mean(), block=block, n_draws=bb_draws, rng=rng)
    out["cum21"] = float(c21.mean())
    out["cum21_bb_p_two_sided"] = float((np.sum(np.abs(bm) >= abs(c21.mean())) + 1) / (bb_draws + 1))
    k = int(np.nanargmax(np.abs(curve)))
    out["peak_lag"] = k + 1
    out["peak_value"] = float(curve[k])
    out["peak_outside_bb_band"] = bool(not (lo[k] <= curve[k] <= hi[k])) if np.isfinite(lo[k]) else None
    if "n3_lo" in out:
        out["peak_outside_n3_band"] = bool(not (out["n3_lo"][k] <= curve[k] <= out["n3_hi"][k]))
        out["lags_outside_n3_band"] = [j + 1 for j in range(B.shape[1]) if not (out["n3_lo"][j] <= curve[j] <= out["n3_hi"][j])]
    out["lags_outside_bb_band"] = [j + 1 for j in range(B.shape[1]) if np.isfinite(lo[j]) and not (lo[j] <= curve[j] <= hi[j])]
    out["n_dates"] = int(np.isfinite(B).any(axis=1).sum())
    return out


# ===========================================================================
# 16. Event diagnostics D7 (budget calendar) and D8 (S-dated 8-K shocks)
# ===========================================================================
D7_EVENTS = (
    {"event": "2011-08-02", "what": "debt-ceiling resolution (Budget Control Act; also created the sequester)", "sign": 0},
    {"event": "2013-03-01", "what": "sequestration takes effect", "sign": -1},
    {"event": "2013-10-01", "what": "shutdown begins (window truncated at the 2013-10-16 session)", "sign": -1,
     "truncate_at": "2013-10-16"},
    {"event": "2018-12-22", "what": "partial shutdown begins (day 0 = 2018-12-24; DoD funded)", "sign": 0},
)
D7_DROPPED = ({"event": "2013-10-17", "why": "the end of the 2013-10-01 shutdown"},
              {"event": "2018-01-20", "why": "the shutdown ended on its own day 0 (2018-01-22)"})


def _mde_t(sd_: float, n: int) -> float:
    from scipy.stats import t as tdist

    if n < 2 or not np.isfinite(sd_):
        return float("nan")
    return float((tdist.ppf(0.95, n - 1) + tdist.ppf(0.80, n - 1)) * sd_ / math.sqrt(n))


def _t_test(x: np.ndarray) -> dict[str, float]:
    from scipy.stats import t as tdist

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"), "t": float("nan"), "p_one_sided": float("nan"),
                "sd": float("nan")}
    sd_ = float(x.std(ddof=1))
    t = float(x.mean() / (sd_ / math.sqrt(n))) if sd_ > 0 else float("nan")
    return {"n": n, "mean": float(x.mean()), "sd": sd_, "t": t, "p_one_sided": float(1 - tdist.cdf(t, n - 1)) if np.isfinite(t) else float("nan")}


def d7_budget(sd: StudyData) -> dict[str, Any]:
    """Event study on SIC 3721 / 3728 / 3761 / 3812 / 7373 / 8711 firms (PIT SIC, ALL at the
    month-end before day 0): per event the 21-session cumulative abnormal spread (EW exposed
    firms minus EW other ALL firms) from day 0; signed events tested with an event-level
    t-test, MDE from the sd of non-overlapping 21-session spread windows."""
    from eqxs import equity_events as ee

    pb, m = sd.pb, sd.monthly
    sess = pb.sessions
    R = pb.R

    def spread_rows(i_me: int, rows: np.ndarray) -> np.ndarray:
        exp = m.all_[i_me] & np.isin(m.sic4[i_me], D7_SICS)
        oth = m.all_[i_me] & ~exp
        rr = R[rows].astype(np.float64)
        return np.nanmean(np.where(exp[None, :], rr, np.nan), axis=1) - np.nanmean(np.where(oth[None, :], rr, np.nan), axis=1), int(exp.sum())

    ev = []
    for e in D7_EVENTS:
        d0 = ee.day0_from_acceptance(pd.Series([pd.Timestamp(e["event"] + " 12:00", tz="America/New_York")]), sess)[0]
        r0 = int(sess.get_loc(d0))
        r1 = r0 + 20
        if e.get("truncate_at"):
            r1 = min(r1, int(sess.get_loc(pd.Timestamp(e["truncate_at"]))))
        i_me = int(np.searchsorted(m.rows, r0, side="left") - 1)
        x, n_exp = spread_rows(i_me, np.arange(r0, r1 + 1))
        ev.append({**e, "day0": str(d0.date()), "window_end": str(sess[r1].date()), "sessions": int(r1 - r0 + 1),
                   "n_exposed_firms": n_exp, "car": float(np.nansum(x)),
                   "signed_car": float(np.nansum(x) * e["sign"]) if e["sign"] else None})
    # MDE: non-overlapping 21-session windows 2005-2022 of the same spread (exposure set refreshed monthly)
    w = []
    for i in range(len(m.rows) - 1):
        a = int(m.rows[i]) + 1
        if sess[a].year < 2005:
            continue
        rows = np.arange(a, min(a + 21, len(sess)))
        if len(rows) < 21:
            continue
        x, _ = spread_rows(i, rows)
        w.append(float(np.nansum(x)))
    sdw = float(np.std(w, ddof=1))
    signed = np.array([e["signed_car"] for e in ev if e["signed_car"] is not None])
    tt = _t_test(signed)
    return {"events": ev, "dropped": list(D7_DROPPED), "signed_test": tt, "window_sd_21": sdw,
            "mde_signed_n": _mde_t(sdw, len(signed)), "mde_if_6_events": _mde_t(sdw, 6),
            "note": "no N2, no years rule (pinned); 2 signed events -> df = 1"}


def d8_eightk(sd: StudyData, *, end: pd.Timestamp = TRAIN_END) -> dict[str, Any]:
    """S-dated 8-K shocks (4.02 restatement, 2.04 default / acceleration, 1.03 bankruptcy;
    sign -1; 5.02 officer change unsigned) at the origin firm (no price needed), propagated
    one hop: E4 (the origin as customer -> its suppliers with an edge valid at day 0) and E1
    (the other ALL firms of the origin's SIC4). Day 0 = acceptance session. Per event the EW
    mean of the targets' 21-session market-model CAR from day 0 (beta = PIT, month-end before
    day 0); event-level t-test on the signed means, MDE stated."""
    from eqxs.sec_eightk_items import load_eightk_items
    from eqxs import equity_events as ee

    pb, m = sd.pb, sd.monthly
    sess = pb.sessions
    ev = load_eightk_items(start="2004-01-01", end=end, items=["4.02", "2.04", "1.03", "5.02"],
                           columns=["cik", "accession", "ts", "item_4_02", "item_2_04", "item_1_03", "item_5_02"])
    assert pd.to_datetime(ev["ts"], utc=True).max() < end.tz_localize("UTC")
    ev = ev.drop_duplicates(["cik", "accession"]).reset_index(drop=True)
    ev["day0"] = ee.day0_from_acceptance(pd.Series(pd.to_datetime(ev["ts"], utc=True)), sess, intraday="acceptance_session").to_numpy()
    ev = ev[ev["day0"].notna()].copy()
    ev["r0"] = sess.get_indexer(pd.DatetimeIndex(ev["day0"]))
    ev = ev[(ev["r0"] >= 0) & (ev["r0"] + 20 < len(sess))]
    ev["neg"] = ev["item_4_02"] | ev["item_2_04"] | ev["item_1_03"]
    col_of = pb.col_of_cik()
    sa = sic_at(sd.e1)
    res: dict[str, Any] = {"events_loaded": int(len(ev)), "clean_event_mask": "n/a (8-K item table has no ambiguity flags)",
                           "acceptance_stamp_note": "8-K ts = EDGAR submissions acceptance; the SEC agent found it late by 4-5 h "
                                                    "on ~17 % of XBRL filings (never early): day 0 can be one session late, never early"}
    e4 = sd.e4

    def car(i_me: int, cols: np.ndarray, r0: int) -> np.ndarray:
        rr = pb.R[r0:r0 + 21][:, cols].astype(np.float64)
        ar = rr - pb.rf[r0:r0 + 21, None] - m.beta[i_me, cols].astype(np.float64)[None, :] * pb.mktrf[r0:r0 + 21, None]
        ok = np.isfinite(ar).sum(0) >= 15
        return np.where(ok, np.nansum(ar, axis=0), np.nan)

    for graph in ("E4", "E1"):
        if graph == "E4" and e4 is None:
            res[graph] = {"status": "NOT RUN", "reason": "links table absent"}
            continue
        recs = []
        sub_ev = ev[(ev["day0"] >= pd.Timestamp("2011-01-01")) & ev["cik"].astype(int).isin(set(e4["customer_cik"].astype(int)))] \
            if graph == "E4" else ev
        for cik, d0, r0, neg, i502 in zip(sub_ev["cik"].astype(int), sub_ev["day0"], sub_ev["r0"], sub_ev["neg"],
                                          sub_ev["item_5_02"], strict=True):
            i_me = int(np.searchsorted(m.rows, r0, side="left") - 1)
            if i_me < 0:
                continue
            if graph == "E4":
                a = e4[(e4["customer_cik"] == cik) & (e4["valid_from"] <= d0) & (e4["valid_to"].isna() | (d0 < e4["valid_to"]))]
                cols = np.array([col_of.get(int(s), -1) for s in a["supplier_cik"].unique()], np.int64)
            else:
                s4 = sa(cik, pd.Timestamp(d0).tz_localize("America/New_York"))
                if not s4:
                    continue
                cols = np.flatnonzero(m.all_[i_me] & (m.sic4[i_me] == s4))
            cols = cols[(cols >= 0)]
            cols = cols[m.all_[i_me, cols] & (pb.col_cik[cols] != cik)]
            if len(cols) == 0:
                continue
            c = car(i_me, cols, int(r0))
            if not np.isfinite(c).any():
                continue
            recs.append({"cik": cik, "day0": d0, "neg": bool(neg), "i502": bool(i502), "n_targets": int(np.isfinite(c).sum()),
                         "mean_car": float(np.nanmean(c))})
        df = pd.DataFrame(recs)
        out = {"events_with_targets": int(len(df))}
        if len(df):
            neg = df[df["neg"]]
            out["negative_items_signed"] = {**_t_test(-neg["mean_car"].to_numpy()),
                                            "mde": _mde_t(float(neg["mean_car"].std(ddof=1)), len(neg)),
                                            "mean_targets_per_event": float(neg["n_targets"].mean()) if len(neg) else float("nan"),
                                            "events_by_year": {int(k): int(v) for k, v in pd.DatetimeIndex(neg["day0"]).year.value_counts().sort_index().items()}}
            o502 = df[df["i502"] & ~df["neg"]]
            out["item_5_02_unsigned"] = {**_t_test(o502["mean_car"].to_numpy()), "note": "unsigned, not traded"}
        res[graph] = out
    return res


# ===========================================================================
# 17. Runner registration, study driver
# ===========================================================================
def register() -> None:
    """Register the eqgraph signal / panel / N3 hook with the generic runner (names only: the
    evaluation is :func:`run_eqgraph`, which composes the runner's assembly, markdown, DSR,
    verdict and verdict-log functions)."""
    from eqxs import study_runner as sr

    if "eqgraph" not in sr.SIGNALS:
        @sr.register_signal("eqgraph")
        def _sig(ctx: Any, cell: Mapping, params: Mapping) -> Any:  # pragma: no cover - driver only
            raise NotImplementedError("eqgraph cells are evaluated by eqxs.equity_graph.run_eqgraph")

    if "eqgraph" not in sr.PANELS:
        @sr.register_panel("eqgraph")
        def _panel(ctx: Any, horizon: str, params: Mapping) -> Any:  # pragma: no cover - driver only
            raise NotImplementedError("eqgraph builds per-cell unit panels (equity_graph.unit_panel)")

    if "eqgraph_rewire" not in sr.NULLS:
        @sr.register_null("eqgraph_rewire")
        def _n3(*a: Any, **k: Any) -> dict:  # pragma: no cover - driver only
            raise NotImplementedError("eqgraph N3 is computed in equity_graph.evaluate_book")


def precision_gate(path: Path = EDGES_DIR / "e4_precision_sample.csv") -> dict[str, Any]:
    """Hand-check result of the 200-edge sample (seed 15): precision = correct / labelled
    (ambiguous counts as not correct; the gate is >= 0.9). Unlabelled -> gate not passed."""
    if not path.exists():
        return {"status": "absent", "pass": False}
    d = pd.read_csv(path)
    lab = d["label"].fillna("").astype(str).str.strip().str.lower()
    n = int(len(d))
    nl = int((lab != "").sum())
    c, w, a = int((lab == "correct").sum()), int((lab == "wrong").sum()), int((lab == "ambiguous").sum())
    prec = c / n if n else float("nan")
    return {"n": n, "labelled": nl, "correct": c, "wrong": w, "ambiguous": a, "precision": prec,
            "precision_excl_ambiguous": c / (c + w) if c + w else float("nan"), "gate": 0.9,
            "pass": bool(nl == n and n == 200 and prec >= 0.9),
            "rule": "precision = correct / 200, ambiguous counted as not correct; gate >= 0.9"}


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def run_eqgraph(spec_path: str | Path, *, cells: Sequence[str] | None = None, log_db: bool = True,
                write: bool = True) -> dict[str, Any]:
    from eqxs.data_zones import assert_sealed_clean
    from eqxs import study_runner as sr

    register()
    spec = sr.load_spec(spec_path)
    t_start = time.time()
    sd = load_study_data()
    art = REPO / spec["output"]["artefacts"]
    gate = json.loads((EDGES_DIR / "coverage_gate.json").read_text())
    prec = precision_gate()
    cap = float(spec["caps"].get("cell_minutes", 90))
    by_id = {c["id"]: c for c in spec["cells"]}
    defs = {**trial_defs(), **diagnostic_defs()}
    results: dict[str, dict] = {}
    # the D1 daily spread is the IMOM attribution factor (built first; it chooses nothing)
    _log("D1 industry-momentum factor for the attribution")
    d1 = defs["D1"]
    rows, per = cell_signals(sd, d1)
    up = unit_panel(sd, sd.monthly, rows, per, kind="firm", universe="ALL", flat_below=FLAT_BELOW["E4"])
    mo = months_of(sd, sd.monthly, rows)
    bk = quintile_book(up, up.signals["inc"], keep_members=True)
    sp, srow = daily_book_spread(sd.pb.R, bk, mo)
    imom = pd.Series(sp, index=srow)
    for cid in spec["run_order"]:
        cell = by_id[cid]
        base = {"cell": dict(cell), "horizon": "W" if cell["cadence"] == "weekly" else "M"}
        if cells is not None and cid not in cells:
            results[cid] = {**base, "status": "NOT RUN", "reason": "not executed in this invocation"}
            continue
        cd = defs[cid]
        reason = None
        if cd.edge == "E4" and sd.e4 is None:
            reason = "links table absent"
        elif cid == "T7" and not prec.get("pass"):
            reason = f"precision gate failed (precision {prec.get('precision', float('nan')):.3f} < 0.9)" if prec.get("labelled") == 200 \
                else "precision gate not completed"
        elif cd.role == "trial" and not gate["cells"][cid]["run"]:
            reason = (f"coverage gate: median {gate['cells'][cid]['median']:.0f} units < {gate['cells'][cid]['gate_median']}")
        elif cid.startswith("D3_") and not gate["cells"].get(cid, {}).get("book", False):
            reason = f"D3 count rule: median {gate['cells'].get(cid, {}).get('median', float('nan')):.0f} targets < 30 (counts reported)"
        elif cid.startswith(("D2_T7", "D5_T7", "D6_T7")) and not gate["cells"]["T7"]["run"]:
            reason = "coverage gate of T7 failed"
        elif cid.startswith(("D2_T7", "D5_T7", "D6_T7")) and not prec.get("pass"):
            reason = "precision gate of E4 failed / not completed"
        if reason:
            results[cid] = {**base, "status": "NOT RUN", "reason": reason}
            _log(f"[eqgraph {cid}] NOT RUN: {reason}")
            continue
        _log(f"[eqgraph {cid}] start (load avg {__import__('os').getloadavg()[0]:.1f})")
        unis = ["ALL", "LIQ"] if cd.role == "trial" else ["ALL"]
        try:
            r = run_sort_cell(sd, cd, spec, gate=gate["cells"].get(cid) if cd.role == "trial" else None, imom=imom,
                              cap_minutes=cap, artefacts=art, universes=unis)
        except TimeoutError as exc:
            results[cid] = {**base, "status": "NOT RUN", "reason": str(exc)}
            continue
        results[cid] = {**base, **r}
        L = r["ALL"]
        _log(f"[eqgraph {cid}] gross {L['ENS']['gross']['ann_return'] * 100:+.2f}%/yr SR {L['ENS']['gross']['sharpe']:+.2f}; "
             f"net {L['ENS']['net_ew']['ann_return'] * 100:+.2f}%/yr SR {L['ENS']['net_ew']['sharpe']:+.2f} -> {L['verdict']} "
             f"({r['elapsed_s']:.0f} s)")
    ctx = sr.StudyContext(spec, REPO)
    out = sr.assemble(ctx, _clean(results))
    ver = out["verdict"]
    ver["route_a_note"] = ("pre-declaration: every PASS needs all per-cell gates AND DSR_i > 0.95 (route (a) without the DSR "
                           "is not a PASS route in study 15); study verdict = PASS only via route (b)")
    ver["study_verdict"] = "PASS" if ver["route_b_cells"] else "REJECT"
    curves = json.loads((art / "curves.json").read_text()) if (art / "curves.json").exists() else None
    d7 = d7_budget(sd)
    d8 = d8_eightk(sd)
    estats = json.loads((EDGES_DIR / "stats.json").read_text())
    d9 = {}
    for cid in ("T7", "T10", "T11", "T12"):
        c = out["cells"].get(cid, {})
        if c.get("status") == "RUN":
            E = c["ALL"]["ENS"]
            d9[cid] = {"net_ew_2x_sr": E["net_ew_2x"]["sharpe"], "net_ew_2x_ann": E["net_ew_2x"]["ann_return"],
                       "verdict_2x_cost": c["ALL"]["verdict_2x_cost"], "fails_2x_cost": c["ALL"]["fails_2x_cost"],
                       "break_even_cost_bps_ew": E["break_even_cost_bps_ew"],
                       "break_even_multiple_of_tier_schedule_ew": E["break_even_multiple_of_tier_schedule_ew"]}
    out["study15"] = {
        "coverage_gate": {k: {kk: v for kk, v in g_.items() if kk not in ("counts", "flat_mask", "target_months")}
                          for k, g_ in gate["cells"].items()},
        "coverage_gate_computed_utc": gate.get("computed_utc"), "precision_gate": prec,
        "edge_stats": {k: v for k, v in estats.items() if k in ("e1", "e5", "e5_map", "e4", "bound", "edge_version",
                                                                 "resolver", "operator")},
        "study_data_meta": sd.meta, "price_base_meta": sd.pb.meta, "D7": d7, "D8": d8, "D9": d9,
        "impulse_responses": curves,
        "costs": "per name by liquidity tier at formation: 5 bps / side liquid half, 15 bps rest of ALL; LIQ (D4) 5 bps",
        "survivorship": "survivor-only prices; graphs on all filers, prices at origin and target only; spread biased towards "
                        "zero (false-REJECT direction); a REJECT is uninformative about the full universe",
        "ff_note": "French daily factors rebuilt TRAIN-bounded from the raw zips (stored ff_daily.parquet ends 2021-12-31 at "
                   "the old sealed start); pre-2022 rows identical",
        "dec2022_note": "the last holding period (entry 2022-12-01) exits at the 2022-12-30 close (20 sessions): no 2023 row",
        "wall_clock_s": round(time.time() - t_start, 1)}
    assert_sealed_clean()
    out["sealed_clean"] = True
    if write:
        jp, mp = REPO / spec["output"]["json"], REPO / spec["output"]["md"]
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(out, indent=1, default=sr._json_default))
        sr.write_markdown(json.loads(jp.read_text()), mp)
        with mp.open("a") as f:
            f.write(study15_markdown(json.loads(jp.read_text())))
    if log_db and spec["output"].get("log_verdicts", True):
        out["logged"] = sr.log_verdicts(out, spec)
    return out


# ===========================================================================
# 18. Study-15 report section
# ===========================================================================
def _pc(x: Any, nd: int = 2) -> str:
    return "n/a" if x is None or not isinstance(x, (int, float)) or not np.isfinite(x) else f"{x * 100:+.{nd}f}%"


def _fx(x: Any, nd: int = 2) -> str:
    return "n/a" if x is None or not isinstance(x, (int, float)) or not np.isfinite(x) else f"{x:+.{nd}f}"


def _pv(x: Any) -> str:
    return "n/a" if x is None or not isinstance(x, (int, float)) or not np.isfinite(x) else f"{x:.3f}"


def study15_markdown(out: Mapping[str, Any]) -> str:
    s = out["study15"]
    L: list[str] = ["", "# Study 15 specifics", ""]
    a = L.append
    a("## Trial table (ALL, EW-rebalance net = gate; tiered cost 5 / 15 bps per side)")
    a("")
    a("| trial | status | IC (t) | gross %/yr (SR) | net EW %/yr (SR) | net E/E %/yr (SR) | net 2x %/yr (SR) | turnover/yr EW | "
      "break-even bps (x tier) | N1 p gross | N3 p (swap share) | N2 p | N2 MDE %/yr | years ex-best (pos/n) | flat months | verdict | fails (1x / 2x) |")
    a("|" + "---|" * 17)
    for cid in ("T7", "T10", "T11", "T12"):
        c = out["cells"].get(cid, {})
        if c.get("status") != "RUN":
            a(f"| {cid} | NOT RUN: {c.get('reason')} |" + " |" * 15)
            continue
        Lq = c["ALL"]
        E = Lq["ENS"]
        n3 = Lq.get("N3", {})
        sw = f" ({n3['swap_share_mean']:.2f})" if n3.get("swap_share_mean") is not None else ""
        yr = Lq["years_rule"]
        a(f"| {cid} | RUN | {E['ic']['ic_mean']:+.4f} ({_fx(E['ic']['ic_t'])}) | {_pc(E['gross']['ann_return'])} ({_fx(E['gross']['sharpe'])}) | "
          f"{_pc(E['net_ew']['ann_return'])} ({_fx(E['net_ew']['sharpe'])}) | {_pc(E['net_ee']['ann_return'])} ({_fx(E['net_ee']['sharpe'])}) | "
          f"{_pc(E['net_ew_2x']['ann_return'])} ({_fx(E['net_ew_2x']['sharpe'])}) | {E['turnover_per_year_ew']:.1f} | "
          f"{_fx(E['break_even_cost_bps_ew'], 1)} ({_fx(E['break_even_multiple_of_tier_schedule_ew'], 2)}) | {_pv(Lq['N1']['gross']['p'])} | "
          f"{_pv(n3.get('p'))}{sw} | {_pv(Lq['N2']['ew1']['p'])} | {_pc(Lq['N2']['ew1']['mde_ann_return'])} | "
          f"{_pc(yr['mean_ex_best_year'])} ({yr['years_positive']}/{yr['years_counted']}) | {E['flat_periods']} | **{Lq['verdict_predeclared']}** | "
          f"{', '.join(Lq['fails']) or '-'} / {', '.join(Lq['fails_2x_cost']) or '-'} |")
    a("")
    a("Per-cell PASS = net SR > 0, N1 gross p < 0.05, N3 p < 0.05, N2 p < 0.05, years rule, all again at 2x cost; then DSR > 0.95 "
      "(route (a) without the DSR is not a PASS route here). N1 MDE (gross, %/yr): " + ", ".join(
          f"{cid} {_pc(out['cells'][cid]['ALL']['N1']['gross']['mde_ann_return'])}" for cid in ("T7", "T10", "T11", "T12")
          if out["cells"].get(cid, {}).get("status") == "RUN") + ".")
    a("")
    a("### Per-year net EW returns of the trials (compounded, by entry year)")
    a("")
    ran = [c for c in ("T7", "T10", "T11", "T12") if out["cells"].get(c, {}).get("status") == "RUN"]
    if ran:
        a("| year | " + " | ".join(ran) + " |")
        a("|---|" + "---|" * len(ran))
        yrs = sorted({r["year"] for c in ran for r in out["cells"][c]["ALL"]["ENS"]["net_ew"]["per_year"]})
        for y in yrs:
            row = [str(y)]
            for c in ran:
                r = next((r for r in out["cells"][c]["ALL"]["ENS"]["net_ew"]["per_year"] if r["year"] == y), None)
                row.append("" if r is None else f"{_pc(r['ret_compounded'], 1)}")
            a("| " + " | ".join(row) + " |")
    a("")
    a("### Ex-ante book beta per year, beta-hedged diagnostic, attribution (reported, never gates)")
    a("")
    for c in ran:
        Lq = out["cells"][c]["ALL"]
        bb = Lq["ENS"]["ex_ante_book_beta_by_year"]
        a(f"* {c}: mean ex-ante book beta {_fx(Lq['ENS']['ex_ante_book_beta_mean'])}; years |beta| > 0.25: "
          f"{Lq['ex_ante_beta_flag_years'] or 'none'}; realised beta of net {_fx(Lq['ENS']['net_ew']['beta'])}; index-hedged net "
          f"{_pc(Lq['beta_hedged_net_ew']['ann_return'])} (SR {_fx(Lq['beta_hedged_net_ew']['sharpe'])}); per-year beta "
          + ", ".join(f"{y} {_fx(v)}" for y, v in bb.items()))
        fa = Lq.get("factor_attribution_with_imom") or []
        if fa:
            a("  * FF3 + MOM + IMOM (D1) per year: " + "; ".join(
                f"{r['year']}: spread {_pc(r['sum_spread'], 1)}, alpha {_pc(r['alpha_ann'], 1)} (t {_fx(r['alpha_t'], 1)}), "
                f"b_IMOM {_fx(r.get('b_IMOM'))}" for r in fa))
    a("")
    a("## D4: trials on the liquid half only (5 bps)")
    a("")
    a("| trial | gross %/yr (SR) | net %/yr (SR) | N1 p gross | N2 p | units/date median |")
    a("|---|---|---|---|---|---|")
    for c in ran:
        Lq = out["cells"][c].get("LIQ")
        if Lq:
            E = Lq["ENS"]
            a(f"| {c} | {_pc(E['gross']['ann_return'])} ({_fx(E['gross']['sharpe'])}) | {_pc(E['net_ew']['ann_return'])} "
              f"({_fx(E['net_ew']['sharpe'])}) | {_pv(Lq['N1']['gross']['p'])} | {_pv(Lq['N2']['ew1']['p'])} | "
              f"{Lq['coverage']['units_per_date_median']:.0f} |")
    a("")
    a("## Diagnostics D1, D2, D3, D5, D6 (never trials, never PASS, never promoted)")
    a("")
    a("| cell | status | gross %/yr (SR) | net EW %/yr (SR) | N1 p gross | N2 p | IC | note |")
    a("|---|---|---|---|---|---|---|---|")
    for cid, c in out["cells"].items():
        if cid.startswith("T"):
            continue
        if c.get("status") != "RUN":
            a(f"| {cid} | NOT RUN: {c.get('reason')} | | | | | | |")
            continue
        Lq = c["ALL"]
        E = Lq["ENS"]
        note = "GROSS only (weekly, P = 50.4)" if Lq.get("gross_only") else ""
        a(f"| {cid} | RUN | {_pc(E['gross']['ann_return'])} ({_fx(E['gross']['sharpe'])}) | "
          f"{'-' if Lq.get('gross_only') else _pc(E['net_ew']['ann_return']) + ' (' + _fx(E['net_ew']['sharpe']) + ')'} | "
          f"{_pv(Lq['N1']['gross']['p'])} | {_pv(Lq['N2']['ew1']['p'])} | {E['ic']['ic_mean']:+.4f} | {note} |")
    a("")
    cg = s["coverage_gate"]
    a("## Coverage gate (committed before any return was read) and D3 counts")
    a("")
    for k, v in cg.items():
        if "run" in v:
            a(f"* {k}: median {v.get('median')} units (min {v.get('min')}, max {v.get('max')}), gate {v.get('gate_median')} -> "
              f"{'RUN' if v.get('run') else 'NOT RUN'}; flat months (< {v.get('flat_below')}): {v.get('flat_months')}; median by year "
              f"{v.get('median_by_year')}")
        else:
            a(f"* {k}: median {v.get('median')} targets (max {v.get('max')}); book: {v.get('book')}; mean by year {v.get('mean_by_year')}")
    a(f"* computed {s.get('coverage_gate_computed_utc')}")
    a("")
    pg = s["precision_gate"]
    a("## E4 resolver precision gate (200 edges, seed 15, hand-checked)")
    a("")
    a(f"correct {pg.get('correct')}, wrong {pg.get('wrong')}, ambiguous {pg.get('ambiguous')}: precision {_pv(pg.get('precision'))} "
      f"(excluding ambiguous {_pv(pg.get('precision_excl_ambiguous'))}); gate >= 0.9 -> {'PASS' if pg.get('pass') else 'FAIL'}. "
      "Labels and notes: `data_cache/links/edges/e4_precision_sample.csv`.")
    a("")
    e4 = s["edge_stats"].get("e4", {})
    if e4:
        a(f"E4 edges by acceptance year: {e4.get('edges_by_year')}; suppliers by year: {e4.get('suppliers_by_year')}; "
          f"match methods {e4.get('match_methods')}; drop reasons {e4.get('drop_reasons')}.")
        a("")
    a("## Impulse responses (pooled, 1-session lags 1-30; they choose nothing)")
    a("")
    ir = s.get("impulse_responses")
    if ir:
        a("Slope of the downstream unit's market-model abnormal return on the z-scored upstream signal (return per 1 sd), mean over "
          "formation months. Bands: N3 (rewired graph, 200 draws, 2.5-97.5 %) and centred circular block bootstrap of the per-date "
          "slopes (3-month blocks ~ 63 sessions, 1000 draws).")
        a("")
        a("| curve | dates | peak lag | peak value (bps) | outside N3 band? | outside BB band? | lags outside N3 band | lags outside BB band | "
          "cum lags 1-21 (bps) | cum BB p | N3 cum21 band (bps) |")
        a("|---|---|---|---|---|---|---|---|---|---|---|")
        for k, v in ir.get("curves", {}).items():
            n3b = v.get("n3_cum21_lo_hi")
            a(f"| {k} | {v.get('n_dates')} | {v.get('peak_lag')} | {v.get('peak_value', float('nan')) * 1e4:+.2f} | "
              f"{v.get('peak_outside_n3_band', 'n/a')} | {v.get('peak_outside_bb_band')} | {v.get('lags_outside_n3_band', 'n/a')} | "
              f"{v.get('lags_outside_bb_band')} | {v.get('cum21', float('nan')) * 1e4:+.1f} | {_pv(v.get('cum21_bb_p_two_sided'))} | "
              f"{'n/a' if not n3b else f'[{n3b[0] * 1e4:+.1f}, {n3b[1] * 1e4:+.1f}]'} |")
        a("")
    else:
        a("not computed (run scripts/validate/equity_graph_curves.py first)")
        a("")
    d7 = s["D7"]
    a("## D7 government budget calendar (event study; no N2, no years rule)")
    a("")
    a("| event | sign | day 0 | window end | sessions | exposed firms | CAR (exposed - other ALL) |")
    a("|---|---|---|---|---|---|---|")
    for e in d7["events"]:
        a(f"| {e['event']} {e['what']} | {e['sign'] if e['sign'] else 'unsigned'} | {e['day0']} | {e['window_end']} | {e['sessions']} | "
          f"{e['n_exposed_firms']} | {_pc(e['car'])} |")
    t = d7["signed_test"]
    a("")
    a(f"Signed events: n = {t['n']}, mean signed CAR {_pc(t['mean'])}, t {_fx(t.get('t'))}, one-sided p {_pv(t.get('p_one_sided'))}; "
      f"21-session window sd {_pc(d7['window_sd_21'])}; MDE at n = {t['n']} (df {t['n'] - 1}) {_pc(d7['mde_signed_n'])} "
      f"(at 6 events it would be {_pc(d7['mde_if_6_events'])}). Dropped: {', '.join(x['event'] for x in d7['dropped'])}.")
    a("")
    d8 = s["D8"]
    a("## D8 S-dated 8-K shocks propagated one hop (event level)")
    a("")
    a(f"{d8.get('events_loaded')} 8-K events (items 4.02 / 2.04 / 1.03 / 5.02) loaded. {d8.get('acceptance_stamp_note')}.")
    a("")
    for gname in ("E4", "E1"):
        v = d8.get(gname, {})
        if v.get("status") == "NOT RUN":
            a(f"* {gname}: NOT RUN ({v.get('reason')})")
            continue
        ng = v.get("negative_items_signed", {})
        u5 = v.get("item_5_02_unsigned", {})
        a(f"* {gname}: events with priced targets {v.get('events_with_targets')}; negative items (signed x -1): n {ng.get('n')}, "
          f"mean {_pc(ng.get('mean'))}, t {_fx(ng.get('t'))}, p {_pv(ng.get('p_one_sided'))}, MDE {_pc(ng.get('mde'))}, "
          f"targets/event {_fx(ng.get('mean_targets_per_event'), 1)}; item 5.02 unsigned: n {u5.get('n')}, mean {_pc(u5.get('mean'))}, "
          f"t {_fx(u5.get('t'))}")
    a("")
    a("## D9: 2x cost and break-even")
    a("")
    for k, v in s["D9"].items():
        a(f"* {k}: net 2x {_pc(v['net_ew_2x_ann'])} (SR {_fx(v['net_ew_2x_sr'])}), 2x verdict {v['verdict_2x_cost']} "
          f"({', '.join(v['fails_2x_cost']) or '-'}); break-even {_fx(v['break_even_cost_bps_ew'], 1)} bps / side "
          f"({_fx(v['break_even_multiple_of_tier_schedule_ew'])} x the tier schedule)")
    a("")
    a("## Data facts and limitations")
    a("")
    for k in ("costs", "survivorship", "ff_note", "dec2022_note"):
        a(f"* {s[k]}")
    a(f"* SIC source share of ALL firm-months: {s['study_data_meta']['sic_source_share_ALL']}; ALL firm-months mapped to a BEA "
      f"industry: {s['study_data_meta']['ALL_firm_months_mapped_to_BEA']:.3f}")
    a(f"* E5 map: {s['edge_stats'].get('e5_map')}")
    a(f"* E5: {s['edge_stats'].get('e5')}")
    a(f"* wall clock {s['wall_clock_s']:.0f} s; assert_sealed_clean() ok: {out.get('sealed_clean')}")
    a("")
    return "\n".join(L) + "\n"


# ===========================================================================
# 19. Thin audit-battery adapter
# ===========================================================================
def make_battery_adapter(study_id: str = "eqgraph") -> Any:
    """``audit_battery.Runner`` with the study-15 book (quintiles, per-name tiered costs, flat
    months; turnover exposed in units of the reference cost so that the cost identity of B7
    holds exactly), its own price matrix for B6 and its own N1 replay. Headline = the
    declared headline if it ran, else the first declared trial that ran."""
    from eqxs import audit_battery as ab
    from eqxs import equity_ml_grid as g

    register()

    class _LS:
        def __init__(self, bk: Book, ref: float) -> None:
            self.bk, self.ref = bk, ref
            self.gross = bk.gross
            self.to_long, self.to_short = bk.cost_ew / ref, np.zeros_like(bk.gross)
            self.to_long_ee, self.to_short_ee = bk.cost_ee / ref, np.zeros_like(bk.gross)
            self.members = bk.row_members

        def net(self, c: float, accounting: str = "ew") -> np.ndarray:
            return self.bk.net(c / self.ref, accounting)

        def two_sided_turnover(self, accounting: str = "ew") -> np.ndarray:
            return (self.bk.cost_ew if accounting == "ew" else self.bk.cost_ee) / self.ref

        def book_attr(self) -> np.ndarray:
            return self.bk.book_beta()

    class EqGraphAdapter(ab.Runner):
        kind = "runner+eqgraph"

        def __init__(self) -> None:
            super().__init__(study_id)
            self.art = REPO / self.spec["output"]["artefacts"]
            ran = [c for c in self.trial_cells if self.cells.get(c, {}).get("status") == "RUN"]
            head = self.json["verdict"]["headline"]
            self.headline_cells = [(head if head in ran else ran[0], "ALL")] if ran else []
            self._up: dict[str, UnitPanel] = {}

        def cost(self, universe: str) -> float:
            return COST_LIQ if universe == "LIQ" else COST_REST

        def oos(self, cell: str) -> Any:
            if cell not in self._oos:
                up, fr, dates = load_cell_panel(self.art, cell)
                self._up[cell] = up
                sd_rows = dates
                pb = build_price_base()
                e, x = sd_rows["entry_row"].to_numpy(), sd_rows["exit_row"].to_numpy()
                from eqxs import equity_ml as ml

                mo = pd.DataFrame({"date_id": np.arange(len(dates)), "target_mi": ml.month_index(pb.sessions[e]),
                                   "sessions_held": x - e,
                                   "mktrf": compound_rows(pb.mktrf + pb.rf, e, x) - compound_rows(pb.rf, e, x),
                                   "entry_date": pb.sessions[e], "exit_date": pb.sessions[x]})
                self._oos[cell] = ab.OOSData(fr.drop(columns=[c for c in fr.columns if c.startswith("pred_")]),
                                             {"ENS": fr["pred_ENS"].to_numpy(np.float64)}, mo, up.n_firms, (),
                                             stored_ens=fr["pred_ENS"].to_numpy(np.float64))
                self._key_cell = getattr(self, "_key_cell", {})
                self._key_cell[self._key(fr)] = cell
            return self._oos[cell]

        @staticmethod
        def _key(fr: pd.DataFrame) -> str:
            return hashlib.sha256(np.ascontiguousarray(fr["date_id"].to_numpy(np.int64)).tobytes()
                                  + np.ascontiguousarray(fr["firm"].to_numpy(np.int64)).tobytes()
                                  + np.ascontiguousarray(fr["y_raw"].to_numpy(np.float64)).tobytes()).hexdigest()

        def book(self, pred: np.ndarray, sub: pd.DataFrame, b: np.ndarray, n_firms: int, *, attr: np.ndarray | None = None,
                 members: bool = False) -> Any:
            cell = self._key_cell[self._key(sub)]
            return _LS(quintile_book(self._up[cell], pred, keep_members=True), COST_REST)

        def stats(self, x: np.ndarray, mo: pd.DataFrame, h: str) -> dict[str, Any]:
            return g.period_stats(x, mo["target_mi"].to_numpy() // 12, mo["sessions_held"].to_numpy(), mo["mktrf"].to_numpy(), h)

        def reported(self, cell: str, universe: str) -> dict[str, Any]:
            r = super().reported(cell, universe)
            E = self.cells[cell][universe]["ENS"]
            r["turnover_per_year_ew"] = E["cost_weighted_turnover_per_year_ew"]
            r["turnover_per_year_ee"] = E["cost_weighted_turnover_per_year_ee"]
            return r

        def n1_replay(self, cell: str, universe: str, o: Any) -> dict[str, Any]:
            from eqxs import study_runner as sr

            h = self.horizon(cell)
            up = self._up[cell]
            pred = o.preds["ENS"]
            bk = quintile_book(up, pred)
            real = {"gross": g.ann_sharpe(bk.gross, h), "ew1": g.ann_sharpe(bk.net(1, "ew"), h),
                    "ew2": g.ann_sharpe(bk.net(2, "ew"), h), "ee1": g.ann_sharpe(bk.net(1, "ee"), h)}
            n1 = n1_dual(up, pred, h, real, n_draws=int(self.spec["nulls"]["N1"]["draws"]), rng=sr.cell_rng(self.spec, cell, universe))
            return {"net_p": n1["ew1"]["p"], "gross_p": n1["gross"]["p"], "exact_replay": True}

        def panel_metas(self) -> dict[str, dict]:
            return {"eqgraph M": {"source": "eqgraph", "day0_intraday_mode": "acceptance_session (E4 valid_from, D8 day 0)",
                                  "events": {"clean_mask": "clean_event_mask(flags='pit') not applicable: the 8-K item table "
                                                           "and the E4 filings carry no ambiguity flags"}}}

        def acceptance_tables(self) -> tuple[str, ...]:
            return ("events/eightk_items",)  # D8 8-K shocks; E4 dates come from FSN sub.accepted

        def aux(self) -> dict[str, Any]:
            if self._aux is None:
                pb = build_price_base()
                fac = load_ff_train().reindex(pb.sessions)
                self._aux = {"R": pb.R, "sessions": pb.sessions, "fac": fac, "symbols": pb.symbols}
            return self._aux

        def artefacts(self) -> list[Path]:
            out = []
            for c in self.cells.values():
                if c.get("predictions"):
                    p = self.p(c["predictions"])
                    out += [p, p.with_name(p.stem + "_dates.parquet")]
            return out

    return EqGraphAdapter()


# study_runner plugin hooks (resolved by ``study_runner.plugin_hook``)
def run_study_hook(spec_path: str | Path, *, cells: Sequence[str] | None = None, log_db: bool = True) -> dict[str, Any]:
    return run_eqgraph(spec_path, cells=cells, log_db=log_db)


battery_adapter = make_battery_adapter


def run_battery_eqgraph(out_dir: Path | None = None) -> tuple[list, str]:
    from eqxs import audit_battery as ab

    a = make_battery_adapter("eqgraph")
    _log(f"battery eqgraph ({a.kind}): headline {a.headline_cells}")
    checks = ab.run_checks(a)
    ab.write_outputs(a, checks, out_dir or REPO / "reports" / "audit")
    return checks, ab.final_line(checks)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="python -m eqxs.equity_graph")
    sp_ = ap.add_subparsers(dest="cmd", required=True)
    r = sp_.add_parser("run")
    r.add_argument("spec", nargs="?", default="config/studies/eqgraph.yaml")
    r.add_argument("--cells", default="")
    r.add_argument("--no-log-db", action="store_true")
    v = sp_.add_parser("validate")
    v.add_argument("spec", nargs="?", default="config/studies/eqgraph.yaml")
    sp_.add_parser("battery")
    args = ap.parse_args(argv)
    if args.cmd == "validate":
        from eqxs import study_runner as sr

        register()
        spec = sr.load_spec(args.spec)
        print(f"{spec['id']}: valid ({len(spec['cells'])} cells)")
        return 0
    if args.cmd == "run":
        cells = [c for c in args.cells.split(",") if c] or None
        out = run_eqgraph(args.spec, cells=cells, log_db=not args.no_log_db)
        print(f"eqgraph: study verdict {out['verdict']['study_verdict']}; per trial {json.dumps(out['verdict']['per_trial'])}")
        return 0
    checks, line = run_battery_eqgraph()
    for c in checks:
        print(f"{c.id:4s} {c.status:5s} {c.reason}")
    print(line)
    sys.stdout.flush()
    return 0 if line == "BATTERY CLEAN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
