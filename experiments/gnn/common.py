"""P13 shared interface (v1, after the independent design review, items A1-A3, A6, A7, A9): paths, panel/firm/date
indices, pinned input columns, the embargo rule, graph-snapshot / prediction / queue formats, hyperparameter hash + registry.

Every P13 script imports this module and nothing else defines a path, a rule or a file format.
Zones: the panels end 2022-12 (TRAIN); nothing here opens a sealed partition. Never pd.read_parquet a price file.
"""
from __future__ import annotations
import hashlib, json, re
from functools import lru_cache
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data_cache"
PANELS = DATA / "equities_xs" / "derived" / "eqml_grid_2023"          # panel_{W,M,Q}.parquet + .meta.json, aux_daily.npz
EDGES = DATA / "links" / "edges"                                        # e1_sic_membership, e4_customer_supplier_edges, e5_bea_use_edges, e5_sic4_bea_map
ROOT = DATA / "studies" / "p13_gnn"
GRAPHS, PREDS, RESULTS, QUEUE, LOGS = ROOT / "graphs", ROOT / "preds", ROOT / "results", ROOT / "queue", ROOT / "logs"
REPORTS = REPO / "results" / "gnn"                                      # RESULTS.md, STATE.json, SELECTION.json, figures/
EXPERIMENTS = REPO / "experiments"                                      # p12_books.py, p12_stage1b_lib.py, p12c_eval.py (importable modules)
ML_RESULTS = REPO / "results"                                           # p12c_eval.json, p12_cache/ (written by the P12c scripts)
REGISTRY = Path(__file__).resolve().parent / "registry.json"           # committed list of allowed hyperparameter hashes (A7 v)

HORIZONS = ("W", "M", "Q")
TARGET = "y_rank"                     # centred cross-sectional rank of the forward return (study-13 convention); loss target only
KEYS = ["firm", "cik", "date", "date_id", "target_mi", "exit_date", "y_raw", "y_ex_all", "in_liq", "raw_vol_63", "raw_beta_252", "raw_log_dv_63"]
FORBIDDEN_INPUT = re.compile(r"^(y_|raw_|beta_exante|size_|entry|exit|target|ts$|mi$|date)")   # A2: never a model input
RELATIONS = ("E1_SIC4", "E4_CUSTSUPP", "E5_BEA", "CORR_KNN")
FIRST_OOS_YEAR, LAST_OOS_YEAR, DEV_LAST_YEAR = 2009, 2022, 2018
P12C_ROWSETS = {"W": 1_347_743, "M": 318_475, "Q": 103_945}           # rows predicted by P12c per horizon (A3): every P13 cell must predict exactly these
CELLS = {  # fixed before any return is read (P13_PLAN.md §3 v1). Eligible for selection: C1, C2, C3, C5 only (A7 i).
    "C0":  {"model": "mlp", "relations": (), "eligible": False, "note": "no graph; the baseline every dIC is measured against"},
    "C0g": {"model": "mlp_agg", "relations": RELATIONS, "eligible": False, "note": "MLP on [X, one-hop mean_r(X) per relation, log1p(deg_r)]: graph aggregates as features (A5)"},
    "C1":  {"model": "rsage", "relations": ("E1_SIC4", "E4_CUSTSUPP", "E5_BEA"), "eligible": True},
    "C2":  {"model": "rsage", "relations": RELATIONS, "eligible": True},
    "C3":  {"model": "rgat", "relations": RELATIONS, "eligible": True},
    "C4":  {"model": "rsage", "relations": RELATIONS, "eligible": False, "rewired": True, "note": "C2 on persistent stratified rewired edge tables, one draw per seed (A6)"},
    "C5":  {"model": "rsage_temporal", "relations": RELATIONS, "eligible": True, "seq_len": 12, "note": "phase 2; runs whatever phase 1 shows (A7 v)"},
}
HP = dict(hidden=64, layers=2, dropout=0.1, lr=1e-3, weight_decay=1e-5, epochs=15, heads=4, knn_k=10, corr_window=252, cap=20)
SEEDS = (0, 1, 2)


LABEL_SHIFT_K = {"W": 54, "M": 13, "Q": 5}   # A9: firm-wise label shift LONGER than the 252-session feature lookback (closure check 3), in formations


def code_hash() -> str:
    """sha256 (16 hex) of the P13 source files that define what a model is; recorded in every prediction meta (A7 hardening)."""
    h = hashlib.sha256()
    for f in ("common.py", "gnn.py", "train.py"):
        p = Path(__file__).resolve().parent / f
        if p.exists(): h.update(p.read_bytes())
    return h.hexdigest()[:16]


def hp_hash() -> str:
    """Hash of everything that defines a trained cell (A7 v). Written into every prediction meta; books.py refuses unregistered hashes."""
    return hashlib.sha256(json.dumps({"HP": HP, "CELLS": CELLS, "SEEDS": SEEDS}, sort_keys=True, default=str).encode()).hexdigest()[:16]


def assert_registered(h: str | None = None) -> None:
    h = h or hp_hash(); reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    assert h in reg, f"hyperparameter hash {h} is not in {REGISTRY}; register it by a committed addendum before training or evaluating"


def ensure_dirs() -> None:
    for d in (GRAPHS, PREDS, RESULTS, QUEUE, LOGS, REPORTS):
        d.mkdir(parents=True, exist_ok=True)


def panel_path(h: str) -> Path:
    assert h in HORIZONS; return PANELS / f"panel_{h}.parquet"


@lru_cache(maxsize=None)
def panel_meta(h: str) -> dict:
    return json.loads((PANELS / f"panel_{h}.meta.json").read_text())


@lru_cache(maxsize=None)
def input_columns() -> tuple[str, ...]:
    """A2: the model input matrix is exactly the panel builder's input_columns (28 features + 11 SIC one-hots + 28 miss flags),
    identical across W / M / Q, and contains no target / raw / key name."""
    cols = tuple(panel_meta("M")["input_columns"])
    for h in HORIZONS: assert tuple(panel_meta(h)["input_columns"]) == cols, f"input_columns differ for {h}"
    bad = [c for c in cols if FORBIDDEN_INPUT.match(c)]; assert not bad, bad; assert TARGET not in cols; return cols


def node_features(df: pd.DataFrame) -> np.ndarray:
    """The ONLY way a P13 model reads features (A2). NaN -> 0 (the miss flags carry the information)."""
    cols = list(input_columns()); X = df[cols].to_numpy(np.float32, copy=True); return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


@lru_cache(maxsize=None)
def panel_fingerprint(h: str) -> dict:
    p = panel_path(h); st = p.stat(); c = ROOT / "panel_fingerprints.json"; key = f"{h}:{st.st_size}:{int(st.st_mtime)}"
    cache = json.loads(c.read_text()) if c.exists() else {}
    if key not in cache:
        hsh = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""): hsh.update(chunk)
        cache[key] = hsh.hexdigest(); c.parent.mkdir(parents=True, exist_ok=True); c.write_text(json.dumps(cache, indent=1))
    return {"horizon": h, "sha256": cache[key], "size": st.st_size, "n_rows": int(panel_meta(h).get("n_rows", -1))}


def load_panel(h: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Panel rows in file order (row position is the prediction key). Never filters, never reorders. Asserts the order (A3)."""
    cols = None if columns is None else sorted(set(columns) | {"date_id", "firm"})
    pf = pd.read_parquet(panel_path(h), columns=cols)
    if "date" in pf.columns:
        pf["date"] = pd.to_datetime(pf["date"]).dt.tz_localize(None); assert str(pf["date"].max())[:10] < "2023-01-01", "panel must end in TRAIN"
    if "exit_date" in pf.columns: pf["exit_date"] = pd.to_datetime(pf["exit_date"]).dt.tz_localize(None)
    key = pf["date_id"].to_numpy().astype(np.int64) * (1 << 32) + pf["firm"].to_numpy().astype(np.int64)
    assert np.all(np.diff(key) > 0), "panel is not sorted by (date_id, firm) or has duplicates"
    n = panel_meta(h).get("n_rows");  assert n is None or len(pf) == n, (len(pf), n)
    return pf


def firm_index(h: str) -> np.ndarray:
    """Sorted unique firm ids of the FULL panel for horizon h (int64). Graph snapshots index firms by position in this array."""
    return np.array(sorted(pd.read_parquet(panel_path(h), columns=["firm"])["firm"].unique()), dtype=np.int64)


def date_index(h: str) -> np.ndarray:
    """Sorted unique formation dates (datetime64[ns], naive) of the FULL panel. Snapshot i = dates[i]."""
    d = pd.to_datetime(pd.read_parquet(panel_path(h), columns=["date"])["date"]).dt.tz_localize(None)
    return np.array(sorted(d.unique()), dtype="datetime64[ns]")


# ---- the embargo / refit rule (A1) -------------------------------------------------------------------------------------
# target_mi = absolute month index (year*12 + month-1) of the START of the target period. Refit for OOS year Y:
#   test rows      : 12Y <= target_mi <= 12Y+11                      (P12c year_scorer)
#   training rows  : 0 <= target_mi <= 12Y-2  AND  exit_date <= min(date of the test rows)   (known at the decision; drops the Q Sep-30 label)
# Both dates are written into the prediction meta per refit. P13 trains through Nov of Y-1 (P12c: through Oct of Y-2; not identical).
def test_mask(target_mi: np.ndarray, Y: int) -> np.ndarray:
    return (target_mi >= 12 * Y) & (target_mi <= 12 * Y + 11)


def train_mask(target_mi: np.ndarray, exit_date: np.ndarray, Y: int, min_test_date: np.datetime64) -> np.ndarray:
    return (target_mi >= 0) & (target_mi <= 12 * Y - 2) & (np.asarray(exit_date, "datetime64[ns]") <= np.datetime64(min_test_date, "ns"))


# ---- graph snapshot format -------------------------------------------------------------------------------------------
# One npz per (horizon, relation[, rewire seed]): GRAPHS / f"{h}_{relation}.npz" or f"{h}_{relation}_rw{seed}.npz" (A6) with
#   dates  : datetime64[ns] (T,)          == date_index(h)
#   firms  : int64 (F,)                   == firm_index(h)
#   ptr    : int64 (T+1,)                 CSR offsets by snapshot: edges of snapshot i are [ptr[i], ptr[i+1])
#   src, dst : int32 (E,)                 positions into firms; directed (both directions stored for symmetric relations)
#   w      : float32 (E,)                 edge weight (sign encodes direction for E4 / E5; corr for CORR_KNN; see meta)
#   meta   : json string (0-d)            builder, parameters, PIT rule, counts, rewiring method + swap share for rw files
# Edges of snapshot i satisfy valid_from <= dates[i] < valid_to for interval relations, and use sessions <= dates[i] only for CORR_KNN.
# Both endpoints are present in the panel at dates[i]. Rewired files are drawn ONCE per seed from the edge tables (persistent across dates).
def graph_path(h: str, relation: str, rewire_seed: int | None = None) -> Path:
    assert relation in RELATIONS; return GRAPHS / (f"{h}_{relation}.npz" if rewire_seed is None else f"{h}_{relation}_rw{rewire_seed}.npz")


def save_graph(h: str, relation: str, dates, firms, ptr, src, dst, w, meta: dict, rewire_seed: int | None = None) -> Path:
    p = graph_path(h, relation, rewire_seed); p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, dates=np.asarray(dates, "datetime64[ns]"), firms=np.asarray(firms, np.int64), ptr=np.asarray(ptr, np.int64),
                        src=np.asarray(src, np.int32), dst=np.asarray(dst, np.int32), w=np.asarray(w, np.float32), meta=json.dumps(meta))
    return p


def load_graph(h: str, relation: str, rewire_seed: int | None = None) -> dict:
    z = np.load(graph_path(h, relation, rewire_seed), allow_pickle=False)
    return {k: z[k] for k in ("dates", "firms", "ptr", "src", "dst", "w")} | {"meta": json.loads(str(z["meta"]))}


def snapshot_edges(g: dict, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = int(g["ptr"][i]), int(g["ptr"][i + 1]); return g["src"][a:b], g["dst"][a:b], g["w"][a:b]


# ---- prediction format (P12c-compatible rows/pred, plus keys and fingerprint; A3) -------------------------------------
# PREDS / f"{cell}_{h}_s{seed}.npz" and the seed mean PREDS / f"{cell}_{h}_mean.npz" with
#   rows    : int64 (n,)   positions into panel_{h}.parquet file order; exactly the P12c row set (P12C_ROWSETS[h] rows)
#   pred    : float32 (n,) model output (higher = better); never NaN
#   firm    : int64 (n,)   panel.firm[rows]        (asserted on load)
#   date_id : int64 (n,)   panel.date_id[rows]     (asserted on load)
#   meta    : json string  cell, horizon, seed(s), hp_hash, panel fingerprint, per-refit {Y, n_train_rows, max_train_exit_date, min_test_date},
#                          epochs, n_models, device, torch version, wall time, git HEAD, rewiring method (C4)
def pred_path(cell: str, h: str, seed: int | str) -> Path:
    return PREDS / (f"{cell}_{h}_s{seed}.npz" if isinstance(seed, int) else f"{cell}_{h}_{seed}.npz")


def save_pred(path: Path, rows, pred, firm, date_id, meta: dict) -> Path:
    rows = np.asarray(rows, np.int64); pred = np.asarray(pred, np.float32); firm = np.asarray(firm, np.int64); date_id = np.asarray(date_id, np.int64)
    assert rows.ndim == 1 and len(rows) == len(pred) == len(firm) == len(date_id) and np.isfinite(pred).all() and len(np.unique(rows)) == len(rows)
    meta = dict(meta); meta.setdefault("hp_hash", hp_hash()); assert "panel_fingerprint" in meta and "horizon" in meta, "meta needs horizon + panel_fingerprint"
    path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, rows=rows, pred=pred, firm=firm, date_id=date_id, meta=json.dumps(meta, default=str)); return path


def load_pred(path: Path, verify: bool = True) -> dict:
    z = np.load(path, allow_pickle=False); d = {k: z[k] for k in ("rows", "pred", "firm", "date_id")} | {"meta": json.loads(str(z["meta"]))}
    if verify:
        h = d["meta"]["horizon"]; fp = panel_fingerprint(h); assert d["meta"]["panel_fingerprint"]["sha256"] == fp["sha256"], "panel changed since prediction"
        pf = pd.read_parquet(panel_path(h), columns=["firm", "date_id"]); r = d["rows"]
        assert np.array_equal(pf["firm"].to_numpy()[r], d["firm"]) and np.array_equal(pf["date_id"].to_numpy()[r], d["date_id"]), "prediction keys do not match the panel"
        assert len(r) == P12C_ROWSETS[h], f"{path.name}: {len(r)} rows predicted, P12c predicted {P12C_ROWSETS[h]} (A3: identical universe required)"
    return d


def p12c_rows(h: str) -> np.ndarray:
    """The P12c row set per horizon (results/p12_cache/p12c_{h}.npz) — the universe every P13 cell must predict exactly."""
    return np.load(ML_RESULTS / "p12_cache" / f"p12c_{h}.npz", allow_pickle=False)["rows"].astype(np.int64)


# ---- job queue format ---------------------------------------------------------------------------------------------------
# QUEUE / f"{status}" / f"{job_id}.json", status in {pending, running, done, failed}; job = {"id", "kind": "train"|"books"|"diag"|"graphs",
# "cell", "horizon", "seed", "priority", "cmd": [argv], "requires": [paths], "created", "started", "finished", "exit", "log"}.
# Transitions are atomic renames between the status directories.
def job_dirs() -> dict[str, Path]:
    return {s: QUEUE / s for s in ("pending", "running", "done", "failed")}

