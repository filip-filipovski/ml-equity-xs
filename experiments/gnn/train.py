"""P13 trainer (v1: design review items A1-A3, A5, A6, A9, A10): one (cell, horizon, seed) job = 14 yearly expanding-window refits.

    train.py --cell C2 --horizon W --seed 0 [--epochs 15] [--device cpu|mps] [--smoke] [--years 2009-2022]
             [--rewire-seed S] [--label-shift-year]

* A1  test rows of refit Y = common.test_mask(target_mi, Y); training rows = common.train_mask(target_mi, exit_date, Y, min_test_date),
      min_test_date = earliest formation date among the refit's test rows. Asserted per refit; recorded in the prediction meta.
      The predicted row set (union of the test masks) must equal common.p12c_rows(h) (A3); a difference is reported, nothing is adjusted.
* A2  node features only through common.node_features (67 columns). A fixed x sqrt(12) scale is applied to the rank-feature columns
      (not the sic_/miss_ columns); it is data-independent and recorded in the meta.
* A5  C0g (model mlp_agg) = MLP on [X, mean_r(X), log1p(deg_r)].  A6  C4 loads persistent rewired tables (--rewire-seed, default seed).
* A9  no per-node parameters, no degree inputs in C1-C5; --label-shift-year = firm-wise circular label shift by one year of formations (common.LABEL_SHIFT_K[h]), separate files.
* A10 no IC is computed unless common.ROOT/GATE_OPEN exists.
No xgboost import here (libomp clash with torch on macOS).
"""
from __future__ import annotations

import argparse, json, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.stats import rankdata

import common
import gnn

FEATURE_SCALE = float(np.sqrt(12.0))          # rank features are centred in [-.5,.5] (sd ~.289): fixed constant, data-independent
RULE = ("A1: test rows = common.test_mask(target_mi, Y); train rows = common.train_mask(target_mi, exit_date, Y, min_test_date) "
        "(0 <= target_mi <= 12Y-2 and exit_date <= min formation date of the test rows)")
FORBIDDEN_LABELS = ("y_raw", "y_rank", "y_ex_all", "y_ex_liq", "target_mi")


def gate_open() -> bool:
    return (common.ROOT / "GATE_OPEN").exists()


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True); self.f = open(path, "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a); line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True); self.f.write(line + "\n"); self.f.flush()


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(common.REPO), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def pick_device(want: str, log) -> torch.device:
    if want == "mps":
        try:
            d = torch.device("mps")
            z = torch.zeros(4, 3, device=d); z.index_add_(0, torch.tensor([0, 1, 1, 3], device=d), torch.ones(4, 3, device=d))
            e = torch.zeros(4, 2, device=d); e.scatter_reduce_(0, torch.tensor([[0, 0], [1, 1], [1, 1], [3, 3]], device=d), torch.ones(4, 2, device=d), reduce="amax", include_self=True)
            torch.mps.synchronize(); assert abs(float(z.sum()) - 12) < 1e-5
            return d
        except Exception as ex:
            log(f"WARNING: mps unusable for index_add_/scatter_reduce_ ({ex!r}); falling back to cpu")
    return torch.device("cpu")


def _legacy_global_shift(y: np.ndarray, fpos: np.ndarray, snap: np.ndarray, K: int) -> np.ndarray:
    """LEGACY (leaky; kept for the regression test only, never called by the trainer): shifts a firm's WHOLE series by K of its own
    rows before the refit split, so training rows carry test-year labels and gaps shift the calendar."""
    p = np.lexsort((snap, fpos)); f = fpos[p]
    new = np.r_[True, f[1:] != f[:-1]]; gid = np.cumsum(new) - 1
    gs = np.flatnonzero(new); gn = np.diff(np.r_[gs, len(f)])
    j = np.arange(len(f)) - gs[gid]; srcj = (j + K) % gn[gid]
    out = np.empty_like(y); out[p] = y[p[gs[gid] + srcj]]
    return out


YEAR = np.timedelta64(366, "D")
LABEL_SHIFT_RULE = ("per refit, training rows only: a row takes the label of the same firm's training row K formations earlier "
                    "(K = common.LABEL_SHIFT_K[h] = 54/13/5, longer than the 252-session lookback), else the formation just before, else just after; "
                    "a candidate is valid only if it is not the row itself and its target window ended >= 366 days before the row's formation; "
                    "rows with no valid source are DROPPED (no label), never wrapped")


def refit_shifted_labels(y, snap, fpos, trm, K: int, T: int, F: int, date, exit_):
    """A9 label-shift diagnostic, per refit, inside that refit's TRAINING rows only (arrays in snapshot-sorted row order; `date` = formation
    date, `exit_` = end of the target window per row). Row r (firm f, formation i) takes the label of the firm's training row at i-K,
    else i-K-1, else i-K+1, accepting a candidate s only if s != r and exit_[s] <= date[r] - 366 days (so the source label's return window
    ends before the start of r's 252-session feature lookback; this also implies the source formation is >= one year away). Rows without a
    valid source are dropped (NaN label, excluded from the loss); nothing wraps. Returns (y_train [n]; src [n_train], -1 = dropped,
    aligned to flatnonzero(trm); n_dropped)."""
    tr = np.flatnonzero(trm); si, sf = snap[tr], fpos[tr]
    G = np.full((T, F), -1, np.int64); G[si, sf] = tr
    src = np.full(len(tr), -1, np.int64); rdate = date[tr]
    for off in (0, -1, 1):
        t = np.flatnonzero(src < 0)
        if not len(t):
            break
        j = si[t] - K + off; inb = (j >= 0) & (j < T)
        g = np.where(inb, G[np.where(inb, j, 0), sf[t]], -1)
        ok = (g >= 0) & (g != tr[t])
        ok &= np.where(g >= 0, exit_[np.where(g >= 0, g, 0)] <= rdate[t] - YEAR, False)
        src[t[ok]] = g[ok]
    out = np.full(len(y), np.nan, np.float32); has = src >= 0; out[tr[has]] = y[src[has]]
    return out, src, int((~has).sum())


def check_label_sources(src, rows, trm, tmi, Y, date, exit_):
    """Per-refit assertions for the label-shift diagnostic (valid pairs only): source is a training row of this refit (so target_mi <= 12Y-2,
    never from the test year or later), is never the row itself, and its target window ended >= 366 days before the row's formation."""
    ok = src >= 0; s, r = src[ok], rows[ok]
    assert trm[s].all() and (tmi[s] <= 12 * Y - 2).all() and (tmi[s] < 12 * Y).all(), "shifted label from the test year or later"
    assert (s != r).all(), "a row took its own label"
    assert (exit_[s] <= date[r] - YEAR).all(), "source label's target window overlaps the row's feature lookback (< 366 days)"
    assert (date[r] - date[s] >= YEAR).all()


# ------------------------------------------------------------------------------------------------------------------ data
def load_arrays(h: str, incols: list[str], log=lambda *a: None) -> dict:
    """Low-memory panel loader (same numbers as common.node_features(common.load_panel(...)), same assertions as load_panel).
    Reads the key/label columns once and the input columns in chunks of 8 straight into one float32 matrix (cast float32 first, then
    NaN/inf -> 0, exactly node_features' order). No DataFrame is ever built."""
    path = common.panel_path(h)
    n = pq.ParquetFile(path).metadata.num_rows
    meta_n = common.panel_meta(h).get("n_rows"); assert meta_n is None or n == meta_n, (n, meta_n)
    kt = pq.read_table(path, columns=["firm", "date", "date_id", common.TARGET, "target_mi", "exit_date", "y_raw"])
    col = lambda t, c: t[c].to_numpy()
    date = np.asarray(col(kt, "date")).astype("datetime64[ns]"); assert str(date.max())[:10] < "2023-01-01", "panel must end in TRAIN"
    firm = np.asarray(col(kt, "firm")).astype(np.int64); date_id = np.asarray(col(kt, "date_id")).astype(np.int64)
    assert np.all(np.diff(date_id * (1 << 32) + firm) > 0), "panel is not sorted by (date_id, firm) or has duplicates"
    out = {"n": n, "firm": firm, "date_id": date_id, "date": date, "y": np.asarray(col(kt, common.TARGET)).astype(np.float32),
           "y_raw": np.asarray(col(kt, "y_raw")).astype(np.float64), "tmi": np.asarray(col(kt, "target_mi")).astype(np.int64),
           "exit": np.asarray(col(kt, "exit_date")).astype("datetime64[ns]")}
    del kt
    X = np.empty((n, len(incols)), np.float32); nan_total = 0
    for a in range(0, len(incols), 8):
        t = pq.read_table(path, columns=incols[a:a + 8])
        for j, c in enumerate(incols[a:a + 8], start=a):
            v = np.asarray(t[c].to_numpy()).astype(np.float32)
            nan_total += int(np.isnan(v).sum()); X[:, j] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        del t
    out["X"] = X; out["nan_share"] = nan_total / max(X.size, 1)
    return out


class Data:
    """Panel arranged snapshot-contiguously. `order[start[i]:start[i+1]]` = file positions of snapshot i's rows
    (the panel is asserted sorted by (date_id, firm), so snapshots are contiguous in file order and `order` is the identity)."""

    def __init__(self, h: str, cell: str, seed: int, log, rewire_seed: int | None = None, label_shift: int | None = None):
        spec = common.CELLS[cell]; self.h, self.cell, self.spec = h, cell, spec
        incols = list(common.input_columns())
        assert not [c for c in incols if common.FORBIDDEN_INPUT.match(c)] and common.TARGET not in incols     # A2
        A = load_arrays(h, incols, log); self.n = n = A["n"]
        self.dates = common.date_index(h); self.firms = common.firm_index(h); T, F = len(self.dates), len(self.firms); self.T, self.F = T, F
        d64 = A["date"]
        snap = np.searchsorted(self.dates, d64); assert (self.dates[snap] == d64).all()
        fpos = np.searchsorted(self.firms, A["firm"]); assert (self.firms[fpos] == A["firm"]).all()
        assert (np.diff(snap) >= 0).all()
        self.order = np.arange(n)
        cnt = np.bincount(snap, minlength=T); self.start = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64); assert (cnt > 0).all()
        loc = np.arange(n) - self.start[snap]
        self.rowpos = np.full((T, F), -1, np.int32); self.rowpos[snap, fpos] = loc
        assert (self.rowpos >= 0).sum() == n, "duplicate (date, firm) rows"
        self.fpos_sorted, self.snap_sorted = fpos, snap
        # features: only through the pinned input columns (A2; identical numbers to common.node_features)
        X = A["X"]; assert X.shape[1] == len(incols)
        scaled = np.array([not (c.startswith("miss_") or c.startswith("sic_")) for c in incols]); self.scaled_cols = int(scaled.sum())
        log(f"inputs: {X.shape[1]} columns via the low-memory loader == common.node_features ({self.scaled_cols} rank features x{FEATURE_SCALE:.4f}); "
            f"NaN share before fill {A['nan_share']:.4f}")
        for j in np.flatnonzero(scaled):
            X[:, j] *= FEATURE_SCALE
        self.X = X
        self.label_shift = label_shift
        if label_shift:
            log(f"LABEL SHIFT diagnostic: labels re-assigned PER REFIT inside the training rows (K = {label_shift} formations = one year)")
        self.y = A["y"]                                           # NaN excluded from the loss, still predicted
        self.y_raw, self.tmi, self.exit = A["y_raw"], A["tmi"], A["exit"]
        self.date_row = d64; self.firm_row = A["firm"]; self.date_id_row = A["date_id"]
        del A
        # graphs, stored per relation in CSR form (positions as uint16, weights float32): row-local positions of the present set
        self.rels = list(spec["relations"]); dropped = 0
        rs = None
        if spec.get("rewired"):
            rs = seed if rewire_seed is None else rewire_seed
        self.rewire_seed = rs
        self.g_ptr, self.g_src, self.g_dst, self.g_w, gmeta = [], [], [], [], {}
        n_edges = np.zeros(len(self.rels), np.int64)
        for k, r in enumerate(self.rels):
            ptr_k, src_k, dst_k, w_k, meta_k, drop_k = self._load_relation(h, r, rs, cell, T, log)
            gmeta[r] = meta_k; dropped += drop_k
            self.g_ptr.append(ptr_k); self.g_src.append(src_k); self.g_dst.append(dst_k); self.g_w.append(w_k); n_edges[k] = int(ptr_k[-1])
        self.dropped_edges = dropped
        self.graph_meta = gmeta if rs is not None else None
        log(f"graphs {self.rels or '-'}{' REWIRED draw ' + str(rs) if rs is not None else ''}: {n_edges.tolist()} edges over {T} snapshots; "
            f"{dropped} dropped (endpoint absent at date)")

    def _load_relation(self, h, r, rs, cell, T, log):
        """One relation -> CSR over local row positions, array by array (peak ~ final size + one raw array): src and dst become uint16
        local positions (65535 = endpoint absent at that date), w stays float32; edge order is preserved."""
        path = common.graph_path(h, r, rs)
        try:
            z = np.load(path, allow_pickle=False)
        except FileNotFoundError as ex:
            raise FileNotFoundError(f"graph table missing for {cell}/{h}/{r} (rewire_seed={rs}): {ex}") from ex
        assert (z["dates"] == self.dates).all() and (z["firms"] == self.firms).all(), f"{r}: graph index != panel index"
        ptr = z["ptr"].astype(np.int64); E = int(ptr[-1]); meta = json.loads(str(z["meta"]))
        assert len(ptr) == T + 1 and ptr[0] == 0 and E == len(z["w"]) and E == len(z["src"]) == len(z["dst"])
        SENT = 65535
        def local(name):
            raw = z[name]; out = np.empty(E, np.uint16)
            for c0 in range(0, T, 64):
                c1 = min(c0 + 64, T); e0, e1 = int(ptr[c0]), int(ptr[c1])
                sn = np.repeat(np.arange(c0, c1, dtype=np.int64), np.diff(ptr[c0:c1 + 1]))
                v = self.rowpos[sn, raw[e0:e1]]; assert (v < SENT).all()
                out[e0:e1] = np.where(v >= 0, v, SENT).astype(np.uint16)
            return out
        ls = local("src"); ld = local("dst"); w = np.asarray(z["w"], np.float32); z.close()
        keep = (ls != SENT) & (ld != SENT); dropped = int((~keep).sum())
        if dropped:
            cs = np.concatenate([[0], np.cumsum(keep)]).astype(np.int64); ptr = cs[ptr]
            ls, ld, w = ls[keep], ld[keep], w[keep]
        return ptr, ls, ld, w, meta, dropped

    def edges_of(self, i: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """per relation (src, dst, w) of snapshot i as local row positions (uint16 views) and float32 weights."""
        return [(self.g_src[k][self.g_ptr[k][i]:self.g_ptr[k][i + 1]], self.g_dst[k][self.g_ptr[k][i]:self.g_ptr[k][i + 1]],
                 self.g_w[k][self.g_ptr[k][i]:self.g_ptr[k][i + 1]]) for k in range(len(self.rels))]

    def snapshot(self, i: int, dev):
        a, b = int(self.start[i]), int(self.start[i + 1])
        x = torch.from_numpy(self.X[a:b]).to(dev)
        edges = [(torch.from_numpy(s.astype(np.int64)).to(dev), torch.from_numpy(d.astype(np.int64)).to(dev), torch.from_numpy(w).to(dev))
                 for s, d, w in self.edges_of(i)]
        return x, edges

    def sequence(self, i: int, seq_len: int, dev):
        """[N, seq_len, 67] features of the firms present at i over snapshots i-seq_len+1 .. i (absent -> 0) and mask [N, seq_len]."""
        a, b = int(self.start[i]), int(self.start[i + 1]); fp = self.fpos_sorted[a:b]; xs, ms = [], []
        for j in range(i - seq_len + 1, i + 1):
            if j < 0:
                xs.append(np.zeros((b - a, self.X.shape[1]), np.float32)); ms.append(np.zeros(b - a, np.float32)); continue
            loc = self.rowpos[j][fp]; ok = loc >= 0
            xs.append(np.where(ok[:, None], self.X[np.where(ok, self.start[j] + loc, 0)], 0.0).astype(np.float32)); ms.append(ok.astype(np.float32))
        return torch.from_numpy(np.stack(xs, 1)).to(dev), torch.from_numpy(np.stack(ms, 1)).to(dev)


def forward_snapshot(model, data: Data, i: int, dev):
    x, edges = data.snapshot(i, dev)
    if data.spec["model"] == "rsage_temporal":
        xs, ms = data.sequence(i, data.spec["seq_len"], dev); return model(xs, edges, ms)
    return model(x, edges)


# --------------------------------------------------------------------------------------------------------------------- IC
def ic_summary(data: Data, rows: np.ndarray, pred: np.ndarray) -> dict:
    """rank IC per formation (Spearman, pred vs y_raw); dev (OOS year <= 2018) / check (2019-2022) means + naive t. GATED (A10)."""
    inv = np.empty(data.n, np.int64); inv[data.order] = np.arange(data.n); sp = inv[rows]; snap = data.snap_sorted[sp]; out = []
    for i in np.unique(snap):
        m = snap == i; yy = data.y_raw[sp[m]]; pp = pred[m]; ok = np.isfinite(yy) & np.isfinite(pp)
        if ok.sum() < 20:
            continue
        ic = float(np.corrcoef(rankdata(pp[ok]), rankdata(yy[ok]))[0, 1])
        out.append((str(pd.Timestamp(data.dates[i]).date()), int(data.tmi[sp[m]][0] // 12), ic, int(ok.sum())))

    def blk(lo, hi):
        v = np.array([o[2] for o in out if lo <= o[1] <= hi])
        if len(v) < 3:
            return {"n": int(len(v)), "mean": None, "sd": None, "t": None}
        return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "t": float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))}
    return {"development_2009_2018": blk(0, common.DEV_LAST_YEAR), "check_2019_2022": blk(common.DEV_LAST_YEAR + 1, 9999), "all": blk(0, 9999),
            "by_year": {str(y): float(np.mean([o[2] for o in out if o[1] == y])) for y in sorted({o[1] for o in out})},
            "t_note": "naive t on the per-formation series (no overlap correction)",
            "per_formation": [{"date": o[0], "year": o[1], "ic": o[2], "n": o[3]} for o in out]}


class IcView:
    """The few arrays ic_summary needs, read straight from the panel with pyarrow (no features, no graphs): ~70 MB on W."""

    def __init__(self, h: str):
        t = pq.read_table(common.panel_path(h), columns=["firm", "date", "date_id", "y_raw", "y_ex_all", "target_mi"])
        self.n = t.num_rows; date = np.asarray(t["date"].to_numpy()).astype("datetime64[ns]")
        self.dates = common.date_index(h); self.snap_sorted = np.searchsorted(self.dates, date); assert (self.dates[self.snap_sorted] == date).all()
        self.order = np.arange(self.n)
        firm = np.asarray(t["firm"].to_numpy()).astype(np.int64); did = np.asarray(t["date_id"].to_numpy()).astype(np.int64)
        assert np.all(np.diff(did * (1 << 32) + firm) > 0), "panel is not sorted by (date_id, firm)"
        self.y_raw = np.asarray(t["y_raw"].to_numpy()).astype(np.float64); self.y_ex_all = np.asarray(t["y_ex_all"].to_numpy()).astype(np.float64)
        self.tmi = np.asarray(t["target_mi"].to_numpy()).astype(np.int64)


def write_ic(tag: str, ic: dict) -> Path:
    p = common.RESULTS / f"{tag}_ic.json"; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(ic, indent=1)); return p


def ic_only(cell: str, h: str, seed) -> dict:
    """Write the RESULTS ic json of an EXISTING prediction file (seed int or 'mean') with the same function and keys as the training path.
    Refuses unless common.ROOT/GATE_OPEN exists. No training, no graphs."""
    if not gate_open():
        raise SystemExit("IC summary refused: gate closed (GATE_OPEN absent)")
    d = common.load_pred(common.pred_path(cell, h, seed), verify=True); m = d["meta"]; assert m["horizon"] == h and m["cell"] == cell
    view = IcView(h); ic = ic_summary(view, d["rows"], d["pred"].astype(np.float64))
    if seed == "mean":
        tag = f"{cell}_{h}_mean"
    else:
        tag = f"{cell}_{h}_s{seed}"; ic["meta"] = {k: m[k] for k in ("cell", "horizon", "seed", "epochs", "refit_years")}
    p = write_ic(tag, ic); print(_fmt_ic(tag, ic), flush=True); print(f"wrote {p}", flush=True)
    return ic


def _fmt_ic(tag: str, s: dict) -> str:
    f = lambda b: "n/a" if b["mean"] is None else f"{b['mean']:+.4f} (t {b['t']:+.2f}, n {b['n']})"
    return f"IC[{tag}] dev {f(s['development_2009_2018'])} | check {f(s['check_2019_2022'])} | all {f(s['all'])}"


# ------------------------------------------------------------------------------------------------------------------ main
def expected_rowset(data: Data, yrs: list[int], log) -> np.ndarray:
    """Union of the test masks over `yrs` (file positions), checked against common.p12c_rows(h) (A3). Differences are reported, not adjusted."""
    union = np.zeros(data.n, bool)
    for Y in yrs:
        union |= common.test_mask(data.tmi, Y)
    mine = np.sort(data.order[np.flatnonzero(union)]); p12 = np.sort(common.p12c_rows(data.h))
    if len(yrs) < common.LAST_OOS_YEAR - common.FIRST_OOS_YEAR + 1:       # partial run: compare with P12c restricted to the same target years
        fp_tmi = np.empty(data.n, np.int64); fp_tmi[data.order] = data.tmi
        yr = fp_tmi[p12] // 12; p12 = p12[(yr >= min(yrs)) & (yr <= max(yrs))]
    if not np.array_equal(mine, p12):
        only_m, only_p = np.setdiff1d(mine, p12), np.setdiff1d(p12, mine)
        log(f"ROWSET MISMATCH vs p12c_rows({data.h}): mine {len(mine)}, p12c {len(p12)}, only-mine {len(only_m)} (e.g. {only_m[:5].tolist()}), "
            f"only-p12c {len(only_p)} (e.g. {only_p[:5].tolist()})")
        raise RuntimeError(f"A3: predicted row set != p12c_rows({data.h}); nothing adjusted")
    log(f"row set check: union of test masks == p12c_rows({data.h}) ({len(mine):,} rows)")
    return mine


def run(cell: str, h: str, seed: int, epochs: int | None = None, device: str = "cpu", smoke: bool = False,
        years: tuple[int, int] = (common.FIRST_OOS_YEAR, common.LAST_OOS_YEAR), rewire_seed: int | None = None,
        label_shift_year: bool = False) -> dict:
    common.ensure_dirs()
    git0, code0 = git_head(), common.code_hash()        # captured at job start, before any training
    label_shift = common.LABEL_SHIFT_K[h] if label_shift_year else None
    prefix = "smoke_" if smoke else (f"ls{label_shift}_" if label_shift else "")
    if smoke and label_shift:
        prefix = f"smoke_ls{label_shift}_"
    tag = f"{prefix}{cell}_{h}_s{seed}"
    log = Log(common.LOGS / f"{tag}.log")
    if smoke:
        if h != "M":
            log(f"WARNING: --smoke forces horizon M (got {h})"); h = "M"; tag = f"{prefix}{cell}_{h}_s{seed}"
            if label_shift_year:
                label_shift = common.LABEL_SHIFT_K[h]; prefix = f"smoke_ls{label_shift}_"; tag = f"{prefix}{cell}_{h}_s{seed}"
        years = (2009, 2010); epochs = 2 if epochs is None else epochs
    epochs = common.HP["epochs"] if epochs is None else epochs
    torch.set_num_threads(4); dev = pick_device(device, log)
    t_start = time.time(); spec = common.CELLS[cell]
    log(f"P13 train: cell {cell} {spec['model']} rel {list(spec['relations'])} horizon {h} seed {seed} epochs {epochs} years {years} device {dev} "
        f"torch {torch.__version__} smoke {smoke} label_shift {label_shift} hp_hash {common.hp_hash()}")
    data = Data(h, cell, seed, log, rewire_seed, label_shift)
    log(f"panel {h}: {data.n:,} rows, {data.T} snapshots, {data.F} firms, in_dim {data.X.shape[1]}")
    HP = common.HP; yrs = list(range(years[0], years[1] + 1))
    want_rows = expected_rowset(data, yrs, log)
    in_dim = data.X.shape[1]; rows_all, pred_all, refits = [], [], []
    for k, Y in enumerate(yrs, 1):
        t0 = time.time()
        test = common.test_mask(data.tmi, Y); assert test.any(), Y
        min_test = data.date_row[test].min()
        trm = common.train_mask(data.tmi, data.exit, Y, min_test)
        assert trm.any(), Y
        # A1 assertions (per refit): no training row violates the rule
        assert (data.tmi[trm] >= 0).all() and (data.tmi[trm] <= 12 * Y - 2).all()
        assert (data.exit[trm] <= min_test).all() and not (trm & test).any()
        max_exit = data.exit[trm].max()
        tr_snaps = np.unique(data.snap_sorted[trm])
        assert data.date_row[trm].max() < min_test
        torch.manual_seed(seed * 1000 + Y); rng = np.random.RandomState(seed * 1000 + Y)
        model = gnn.build_model(spec["model"], in_dim, len(spec["relations"]), HP["hidden"], HP["layers"], HP["dropout"], HP["heads"]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
        ytr, ls_drop = data.y, None
        if label_shift:
            ytr, lsrc, nd = refit_shifted_labels(data.y, data.snap_sorted, data.fpos_sorted, trm, label_shift, data.T, data.F, data.date_row, data.exit)
            check_label_sources(lsrc, np.flatnonzero(trm), trm, data.tmi, Y, data.date_row, data.exit)
            assert not np.isfinite(ytr[~trm]).any() and not test[lsrc[lsrc >= 0]].any()
            ls_drop = nd / max(int(trm.sum()), 1)
            log(f"  label shift K={label_shift}: {nd:,} of {int(trm.sum()):,} training rows dropped (no valid source >= 1 year away): {100 * ls_drop:.2f} %")
        losses = []; n_loss_rows = int((trm & np.isfinite(ytr)).sum())
        for ep in range(epochs):
            model.train(); tot, cnt = 0.0, 0
            for i in rng.permutation(tr_snaps):
                a, b = int(data.start[i]), int(data.start[i + 1]); y = ytr[a:b]; ok = np.isfinite(y) & trm[a:b]
                if ok.sum() < 2:
                    continue
                pr = forward_snapshot(model, data, int(i), dev)
                yt = torch.from_numpy(np.nan_to_num(y, nan=0.0)).to(dev); okt = torch.from_numpy(ok).to(dev)
                loss = ((pr - yt)[okt] ** 2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += float(loss.detach()); cnt += 1
            losses.append(tot / max(cnt, 1))
        model.eval(); rr, pp = [], []
        with torch.no_grad():
            for i in np.unique(data.snap_sorted[test]):
                a, b = int(data.start[i]), int(data.start[i + 1]); p = forward_snapshot(model, data, int(i), dev).cpu().numpy().astype(np.float64)
                m = test[a:b]; assert np.isfinite(p[m]).all(), f"non-finite prediction (refit {Y}, snapshot {i})"
                rr.append(data.order[a:b][m]); pp.append(p[m])
        rows_all.append(np.concatenate(rr)); pred_all.append(np.concatenate(pp))
        assert len(rows_all[-1]) == int(test.sum())
        refits.append({"Y": Y, "n_train_rows": int(trm.sum()), "n_train_loss_rows": n_loss_rows, "n_train_snapshots": int(len(tr_snaps)),
                       "max_train_exit_date": str(pd.Timestamp(max_exit).date()), "min_test_date": str(pd.Timestamp(min_test).date()),
                       "n_test_rows": int(test.sum()), "label_shift_dropped_share": ls_drop, "epoch_loss": [round(x, 6) for x in losses], "seconds": round(time.time() - t0, 1)})
        log(f"refit {Y}: train rows {int(trm.sum()):,} ({len(tr_snaps)} snapshots; max exit {refits[-1]['max_train_exit_date']} <= min test date "
            f"{refits[-1]['min_test_date']}); test rows {int(test.sum()):,}; epoch loss {[round(x, 4) for x in losses]}; {time.time() - t0:.1f}s")
        log(f"MODEL {k}/{len(yrs)} trained (cell {cell}, horizon {h}, seed {seed}, year {Y})")
    rows = np.concatenate(rows_all).astype(np.int64); pred = np.concatenate(pred_all); o = np.argsort(rows); rows, pred = rows[o], pred[o]
    assert np.array_equal(rows, want_rows)
    inv = np.empty(data.n, np.int64); inv[data.order] = np.arange(data.n); sp = inv[rows]
    wall = time.time() - t_start
    meta = {"cell": cell, "horizon": h, "seed": seed, "model": spec["model"], "relations": list(spec["relations"]), "rule": RULE,
            "panel_fingerprint": common.panel_fingerprint(h), "refits": refits, "refit_years": yrs, "epochs": epochs, "n_models": len(yrs), "hp": HP,
            "in_dim": in_dim, "model_in_dim": getattr(getattr(model, "mlp", None), "lins", [None])[0].in_features if spec["model"] == "mlp_agg" else in_dim,
            "feature_scale": {"value": FEATURE_SCALE, "applied_to": "rank-feature columns only (not sic_/miss_)", "n_scaled_cols": data.scaled_cols,
                              "data_independent": True},
            "device": str(dev), "torch": torch.__version__, "wall_seconds": round(wall, 1), "git_head": git0, "code_hash": code0, "FEATURE_SCALE": FEATURE_SCALE, "smoke": smoke,
            "label_shift": label_shift, "label_shift_rule": LABEL_SHIFT_RULE if label_shift else None, "rewire_seed": data.rewire_seed, "rewire_meta": data.graph_meta, "dropped_edges": data.dropped_edges,
            "rowset_equals_p12c": True, "loss": "MSE(y_rank), finite training rows, one snapshot per step"}
    path = common.PREDS / f"{prefix}{cell}_{h}_s{seed}.npz"
    common.save_pred(path, rows, pred, data.firm_row[sp], data.date_id_row[sp], meta); log(f"saved {path} ({len(rows):,} rows)")
    out = {"rows": rows, "pred": pred, "path": path, "meta": meta, "data": data}
    if gate_open():
        ic = ic_summary(data, rows, pred.astype(np.float32).astype(np.float64)); ic["meta"] = {k: meta[k] for k in ("cell", "horizon", "seed", "epochs", "refit_years")}
        write_ic(tag, ic); log(_fmt_ic(tag, ic)); out["ic"] = ic
    else:
        log("IC summary skipped: gate closed")
    if not smoke and not label_shift:
        files = [common.pred_path(cell, h, s) for s in common.SEEDS]
        if all(f.exists() for f in files):
            ps = [common.load_pred(f) for f in files]
            assert all((p["rows"] == ps[0]["rows"]).all() for p in ps), "seed files have different rows"
            mean = np.mean([p["pred"].astype(np.float64) for p in ps], axis=0)
            mm = {"cell": cell, "horizon": h, "seeds": list(common.SEEDS), "derived": "mean of seed files", "epochs": epochs, "n_models": len(yrs),
                  "refit_years": yrs, "rule": RULE, "git_head": git0, "code_hash": code0, "FEATURE_SCALE": FEATURE_SCALE, "torch": torch.__version__,
                  "panel_fingerprint": common.panel_fingerprint(h), "refits": refits, "rowset_equals_p12c": True,
                  "rewire_seeds": list(common.SEEDS) if spec.get("rewired") else None}
            mp = common.save_pred(common.pred_path(cell, h, "mean"), ps[0]["rows"], mean, ps[0]["firm"], ps[0]["date_id"], mm); log(f"saved seed-mean {mp}")
            if gate_open():
                mic = ic_summary(data, ps[0]["rows"], mean.astype(np.float32).astype(np.float64)); write_ic(f"{cell}_{h}_mean", mic)
                log(_fmt_ic(f"{cell}_{h}_mean", mic)); out["mean_ic"] = mic
    log(f"done in {time.time() - t_start:.0f}s")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=sorted(common.CELLS)); ap.add_argument("--horizon", required=True, choices=common.HORIZONS)
    ap.add_argument("--seed", default="0", help="int; with --ic-only also 'mean'"); ap.add_argument("--ic-only", action="store_true"); ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"]); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--years", default="2009-2022"); ap.add_argument("--rewire-seed", type=int, default=None)
    ap.add_argument("--label-shift-year", action="store_true")
    a = ap.parse_args(argv)
    common.assert_registered()  # A7 v: refuse unregistered hyperparameters
    if a.ic_only:
        ic_only(a.cell, a.horizon, a.seed if a.seed == "mean" else int(a.seed)); return
    a.seed = int(a.seed)
    y0, y1 = (int(v) for v in a.years.split("-"))
    run(a.cell, a.horizon, a.seed, a.epochs, a.device, a.smoke, (y0, y1), a.rewire_seed, a.label_shift_year)


if __name__ == "__main__":
    main()
