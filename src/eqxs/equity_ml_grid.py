"""Study 14: training-window / timeframe grid for the equity ML cross-section.

Wraps :mod:`eqxs.equity_ml` (study 13) without modifying it. Everything that
study 14's pre-declaration (``reports/research/STUDY_14_PREDECLARATION_2026-10-10.md``, v2)
changes relative to study 13 lives here:

* formation rows per horizon (weekly every 5th session from 2004-01-02, month-ends,
  quarter-ends) and SESSION-based price lookbacks (design review M2), so the weekly, monthly
  and quarterly panels share one feature definition;
* fundamentals "12 months earlier" = last session on or before T - 365 calendar days;
* a generic splitter (expanding / rolling 5 y / rolling 3 y windows, annual / quarterly refits,
  per-horizon validation length) with an embargo checker (M3);
* a refit cache key that includes the split definition and the horizon (C1);
* per-horizon annualisation with fixed periods per year (C3) for Sharpe, mean, MDEs,
  turnover and Newey-West lags;
* decile long/short with BOTH cost accountings (red team F3: equal-weight rebalance and
  entry/exit only), break-even cost and the 2x-cost check (M6);
* the deflated Sharpe with the Bailey-Lopez de Prado cross-trial variance (M1);
* the years rule ``ex_best_year_positive`` via the shared
  :func:`eqxs.equity_events.event_study_verdict`;
* v3 addendum (coordinator, after the study-13 red team R1-R15, before any study-14 result):
  tie-safe average-rank validation IC and random (seeded) decile tie-breaks (R3); fixed booster
  settings without grid or early stopping, M4 with a hessian-scaled min_child_weight, and a
  non-degenerate-prediction assertion after every fit (R2, R4); the N1 gate on the GROSS
  statistic (R1); ex-ante book beta and a beta-hedged book (R5); daily FF3 + MOM attribution of
  the spread (R6); split-distorted XBRL share rows flagged (R8).

Nothing here reads data; the driver is ``scripts/validate/equity_ml_window_grid_train.py``.
"""

# ruff: noqa: N803, N806  (X / R: matrix names follow the maths)
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from eqxs import equity_events as ee
from eqxs import equity_ml as ml

# ---------------------------------------------------------------------------
# Pinned constants (pre-declaration v2)
# ---------------------------------------------------------------------------
HORIZONS: tuple[str, ...] = ("W", "M", "Q")
HORIZON_NAME: dict[str, str] = {"W": "weekly", "M": "monthly", "Q": "quarterly"}
SESSIONS_PER_YEAR: float = 252.0
WEEKLY_STEP: int = 5
WEEKLY_ANCHOR: str = "2004-01-02"  # row 0 of the weekly formation grid, never moved
PERIODS_PER_YEAR: dict[str, float] = {"W": SESSIONS_PER_YEAR / WEEKLY_STEP, "M": 12.0, "Q": 4.0}
N2_BLOCK: dict[str, int] = {"W": 13, "M": 3, "Q": 1}  # ~ 3 calendar months (the gate)
N2_BLOCK_12M: dict[str, int] = {"W": 50, "M": 12, "Q": 4}  # diagnostic
NW_LAGS_12M: dict[str, int] = {"W": 50, "M": 12, "Q": 4}  # NW-t of the mean, 12-month bandwidth
MIN_EMBARGO_MONTHS: dict[str, int] = {"W": 1, "M": 0, "Q": 0}
FIRST_TARGET_MI: int = 2004 * 12 + 1  # 2004-02
OOS_FIRST_YEAR: int = 2009
OOS_LAST_YEAR: int = 2018
FIT_VAL_GAP: int = 1
MIN_YEAR_SESSIONS: int = 63
EULER_GAMMA: float = 0.5772156649015329
GRID_VERSION: str = "eqgrid-v3"  # v3 addendum: R1-R9 machinery fixes
# session offsets of the price lookbacks (pre-declaration "Targets, features, panels")
MOM_12_1: tuple[int, int] = (252, 21)
MOM_6_1: tuple[int, int] = (126, 21)
REV_1M: int = 21
REV_1W: int = 5
FUND_LOOKBACK_DAYS: int = 365


def nw_lags_alpha(horizon: str) -> int:
    """Newey-West lags for the alpha / beta regression: ceil(P / 4) (13 / 3 / 1)."""
    return int(math.ceil(PERIODS_PER_YEAR[horizon] / 4.0))


# ---------------------------------------------------------------------------
# Formation rows
# ---------------------------------------------------------------------------
def weekly_rows(
    sessions: pd.DatetimeIndex, *, anchor: str = WEEKLY_ANCHOR, step: int = WEEKLY_STEP
) -> np.ndarray:
    """Every ``step``-th session counting from ``anchor`` (which must be a session) = row 0."""
    sess = pd.DatetimeIndex(sessions)
    a = sess.get_loc(pd.Timestamp(anchor))
    if not isinstance(a, (int, np.integer)):
        raise ValueError(f"anchor {anchor} is not a unique session")
    return np.arange(int(a), len(sess), step, dtype=np.int64)


def quarter_end_rows(sessions: pd.DatetimeIndex) -> np.ndarray:
    """Last session of March, June, September and December."""
    me = ml.month_end_rows(sessions)
    months = pd.DatetimeIndex(sessions[me]).month
    return me[(months % 3) == 0]


def formation_rows(sessions: pd.DatetimeIndex, horizon: str) -> np.ndarray:
    """All formation rows of a horizon on the calendar (callers restrict to TRAIN dates)."""
    if horizon == "M":
        return ml.month_end_rows(sessions)
    if horizon == "Q":
        return quarter_end_rows(sessions)
    if horizon == "W":
        return weekly_rows(sessions)
    raise ValueError(horizon)


# ---------------------------------------------------------------------------
# Session-based features
# ---------------------------------------------------------------------------
def _cum_log(R: np.ndarray) -> np.ndarray:
    return np.vstack(
        [
            np.zeros((1, R.shape[1])),
            np.cumsum(np.log1p(np.nan_to_num(R, nan=0.0).astype(np.float64)), axis=0),
        ]
    )


def price_features_sessions(
    pm: ml.PriceMatrices, form_rows: np.ndarray, mkt_ew: np.ndarray
) -> dict[str, np.ndarray]:
    """Study-13 price features at arbitrary formation rows with SESSION-based lookbacks.

    ``vol_63`` .. ``log_price`` and ``rev_1w`` are already session windows in study 13 and are
    taken from :func:`equity_ml.price_features`; ``mom_12_1`` = close(T-252) -> close(T-21),
    ``mom_6_1`` = close(T-126) -> close(T-21), ``rev_1m`` = close(T-21) -> close(T) replace the
    positional (month-end row) versions. A window whose start row precedes the firm's first bar
    (or row 0) is missing, as in study 13."""
    rows = np.asarray(form_rows, dtype=np.int64)
    out = ml.price_features(pm, rows, mkt_ew)
    lc = _cum_log(pm.R)
    first = pm.first_bar

    def cum(a: int, b: int) -> np.ndarray:
        v = np.expm1(lc[b + 1] - lc[a + 1])
        return np.where(first <= a, v, np.nan).astype(np.float32)

    for name in ("mom_12_1", "mom_6_1", "rev_1m", "rev_1w"):
        out[name] = np.full((len(rows), pm.n_cols), np.nan, dtype=np.float32)
    for k, t in enumerate(rows):
        t = int(t)
        if t - MOM_12_1[0] >= 0:
            out["mom_12_1"][k] = cum(t - MOM_12_1[0], t - MOM_12_1[1])
        if t - MOM_6_1[0] >= 0:
            out["mom_6_1"][k] = cum(t - MOM_6_1[0], t - MOM_6_1[1])
        if t - REV_1M >= 0:
            out["rev_1m"][k] = cum(t - REV_1M, t)
        if t - REV_1W >= 0:
            out["rev_1w"][k] = cum(t - REV_1W, t)
    return out


def make_fund_query_sessions(
    pm: ml.PriceMatrices, form_rows: np.ndarray, pos: np.ndarray, firm: np.ndarray, cik: np.ndarray
) -> ml.FundQuery:
    """:func:`equity_ml.make_fund_query` with the "12 months earlier" instant replaced by the
    close of the last session on or before T - 365 calendar days (not 12 rows back)."""
    q = ml.make_fund_query(pm, form_rows, pos, firm, cik)
    t_rows = np.asarray(form_rows)[np.asarray(pos, dtype=np.int64)]
    target = pm.sessions[t_rows] - pd.Timedelta(days=FUND_LOOKBACK_DAYS)
    prev = np.searchsorted(pm.sessions.values, target.values, side="right") - 1
    prev = np.clip(prev, 0, None)
    q.ts_12m = ml.close_ts_utc_ns(pm.sessions[prev])
    return q


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------
WINDOWS: dict[str, int | None] = {"expanding": None, "roll5": 5, "roll3": 3}
REFITS: tuple[str, ...] = ("annual", "quarterly")


@dataclass(frozen=True)
class GridSplit:
    """Inclusive target-month-index ranges of one refit."""

    refit_id: str
    fit: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]
    embargo: int
    window: str
    refit: str
    horizon: str
    val_len: int

    @property
    def first_test_mi(self) -> int:
        return self.test[0]


def validation_months(horizon: str, window: str, refit: str) -> int:
    """12 (annual refit) or 6 (quarterly refit); at the quarterly horizon 24 (expanding /
    rolling 5 y) or 12 (rolling 3 y), annual refit only."""
    if horizon == "Q":
        if refit != "annual":
            raise NotImplementedError("quarterly horizon is declared with annual refits only")
        return 12 if window == "roll3" else 24
    return 12 if refit == "annual" else 6


def oos_blocks(
    refit: str, first_year: int = OOS_FIRST_YEAR, last_year: int = OOS_LAST_YEAR
) -> list[tuple[str, int, int]]:
    out = []
    for y in range(first_year, last_year + 1):
        if refit == "annual":
            out.append((str(y), y * 12, y * 12 + 11))
        elif refit == "quarterly":
            for q in range(4):
                out.append((f"{y}Q{q + 1}", y * 12 + 3 * q, y * 12 + 3 * q + 2))
        else:
            raise ValueError(refit)
    return out


def grid_splits(
    horizon: str,
    window: str,
    refit: str,
    embargo: int,
    *,
    first_target_mi: int = FIRST_TARGET_MI,
    first_year: int = OOS_FIRST_YEAR,
    last_year: int = OOS_LAST_YEAR,
    gap: int = FIT_VAL_GAP,
) -> list[GridSplit]:
    """win_end = first test month - 1 - embargo; rolling L years = [win_end - 12L + 1, win_end]
    truncated at ``first_target_mi``; validation = last ``val_len`` months of the window;
    fit = the rest minus a ``gap``-month fit/validation gap."""
    if window not in WINDOWS:
        raise ValueError(window)
    if embargo < MIN_EMBARGO_MONTHS[horizon]:
        raise ValueError(f"embargo {embargo} below the {horizon} minimum")
    val_len = validation_months(horizon, window, refit)
    out = []
    for rid, t0, t1 in oos_blocks(refit, first_year, last_year):
        win_end = t0 - 1 - embargo
        L = WINDOWS[window]
        start = first_target_mi if L is None else max(win_end - 12 * L + 1, first_target_mi)
        val = (win_end - val_len + 1, win_end)
        fit_end = val[0] - 1 - gap
        if fit_end < start:
            raise ValueError(f"no fit months for refit {rid}")
        out.append(
            GridSplit(
                rid, (start, fit_end), val, (t0, t1), embargo, window, refit, horizon, val_len
            )
        )
    return out


def check_embargo(
    splits: Sequence[GridSplit], target_mi: np.ndarray, entry_row: np.ndarray, exit_row: np.ndarray
) -> list[str]:
    """Violations of: max(training exit) <= min(test entry) and max(fit exit) <= min(validation
    entry), per refit (row indices into the session calendar; labels share no daily return
    when the earlier exit is at or before the later entry). Empty list = clean."""
    tmi = np.asarray(target_mi)
    bad = []
    for sp in splits:
        f = ml.mask_months(tmi, sp.fit) & (tmi >= 0)
        v = ml.mask_months(tmi, sp.val) & (tmi >= 0)
        t = ml.mask_months(tmi, sp.test) & (tmi >= 0)
        if not f.any() or not v.any():
            bad.append(f"{sp.refit_id}: empty fit or validation slice")
            continue
        if t.any() and exit_row[f | v].max() > entry_row[t].min():
            bad.append(f"{sp.refit_id}: training exit after test entry")
        if exit_row[f].max() > entry_row[v].min():
            bad.append(f"{sp.refit_id}: fit exit after validation entry")
    return bad


# ---------------------------------------------------------------------------
# Refit cache key (design review C1)
# ---------------------------------------------------------------------------
def refit_seed(base_seed: int, split: GridSplit) -> int:
    """Seed of a refit = study 13's SEED + the month index of the refit's first test month."""
    return int(base_seed) + int(split.first_test_mi)


def cache_key(
    *,
    model: str,
    panel_hash: str,
    columns: Sequence[str],
    split: GridSplit,
    seed: int,
    quick: bool = False,
    version: str = GRID_VERSION,
) -> str:
    payload = {
        "version": version,
        "model": model,
        "panel_hash": panel_hash,
        "columns": list(columns),
        "horizon": split.horizon,
        "window": split.window,
        "window_years": WINDOWS[split.window],
        "refit": split.refit,
        "refit_id": split.refit_id,
        "fit": list(split.fit),
        "val": list(split.val),
        "test": list(split.test),
        "embargo": split.embargo,
        "val_len": split.val_len,
        "seed": int(seed),
        "quick": bool(quick),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_filename(model: str, split: GridSplit, key: str) -> str:
    return (
        f"{model}_{split.horizon}_{split.window}_{split.refit}_e{split.embargo}_"
        f"{split.refit_id}_{key[:16]}.npz"
    )


def array_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Tie-safe validation IC (red team R3)
# ---------------------------------------------------------------------------
class TieSafeIC:
    """Mean within-date Spearman IC with AVERAGE ranks on both sides (as
    :func:`equity_ml.ic_by_date`). A date on which the prediction is constant contributes 0
    (never the IC of row / alphabetical order); a constant target date contributes 0."""

    def __init__(self, y: np.ndarray, bounds: np.ndarray) -> None:
        from scipy.stats import rankdata

        self.bounds = np.asarray(bounds)
        self.yr = []
        for a, b in zip(self.bounds[:-1], self.bounds[1:], strict=True):
            q = rankdata(np.asarray(y[a:b], dtype=np.float64))
            q = q - q.mean()
            nq = np.sqrt((q * q).sum())
            self.yr.append(q / nq if nq > 0 else np.zeros_like(q))

    def per_date(self, pred: np.ndarray) -> np.ndarray:
        from scipy.stats import rankdata

        out = np.empty(len(self.yr))
        for i, (a, b) in enumerate(zip(self.bounds[:-1], self.bounds[1:], strict=True)):
            p = rankdata(np.asarray(pred[a:b], dtype=np.float64))
            p -= p.mean()
            s = np.sqrt((p * p).sum())
            out[i] = float((p * self.yr[i]).sum() / s) if s > 0 else 0.0
        return out

    def __call__(self, pred: np.ndarray) -> float:
        return float(np.mean(self.per_date(pred)))

    def se(self, pred: np.ndarray) -> float:
        """Standard error of the validation IC: sd of the per-date ICs / sqrt(dates)."""
        d = self.per_date(pred)
        return float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")


class DegeneratePredictor(RuntimeError):
    """A fitted model gives a single distinct prediction on some date (red team R2)."""


def assert_nondegenerate(pred: np.ndarray, bounds: np.ndarray, what: str) -> int:
    """Raise unless every date block has more than one distinct prediction; returns the minimum
    number of distinct values per date."""
    mins = []
    for a, b in zip(np.asarray(bounds)[:-1], np.asarray(bounds)[1:], strict=True):
        if b - a < 2:
            continue
        mins.append(len(np.unique(np.asarray(pred[a:b]))))
    m = int(min(mins)) if mins else 0
    if mins and m <= 1:
        raise DegeneratePredictor(f"{what}: a date has a single distinct prediction")
    return m


# ---------------------------------------------------------------------------
# Models (study-13 wrappers, v3 rules)
# ---------------------------------------------------------------------------
GRID_MODELS: tuple[str, ...] = ("M0", "M1", "M2", "M3", "M4")
ENS5: tuple[str, ...] = ("M0", "M1", "M2", "M3", "M4")
ENS3: tuple[str, ...] = ("M2", "M3", "M4")
BOOSTER_FIXED: dict[str, float] = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
BOOSTER_ROUNDS: int = 300
BOOSTER_MCW: dict[str, float] = {"M2": 100.0, "M4": 1.0}  # M4: hessian-scaled (rank:pairwise)


class RidgeTS(ml.RidgeModel):
    """Study-13 ridge with the alpha chosen on the tie-safe validation IC."""

    def fit(self, fd: ml.FitData, *, seed: int) -> dict[str, Any]:
        import time

        from sklearn.linear_model import Ridge

        t0 = time.time()
        self.idx = self._cols(fd.columns)
        ic_v = TieSafeIC(fd.yv, fd.bv)
        scores, best = {}, None
        for a in self.alphas:
            m = Ridge(alpha=a).fit(fd.Xf[:, self.idx], fd.yf)
            sc = ic_v(m.predict(fd.Xv[:, self.idx]))
            scores[str(a)] = sc
            if best is None or sc > best[0]:
                best = (sc, a, m)
        assert best is not None
        self.model = best[2]
        pv = self.model.predict(fd.Xv[:, self.idx])
        return {
            "alpha": best[1],
            "val_ic": best[0],
            "val_ic_se": ic_v.se(pv),
            "grid_val_ic": scores,
            "seconds": time.time() - t0,
            "pred_val": pv,
        }


class FixedBooster(ml.Model):
    """XGBoost with FIXED settings (v3, R4): depth 3, lr 0.05, 300 rounds, subsample and colsample
    0.8, min_child_weight 100 (M2, reg:squarederror) or 1 (M4, rank:pairwise with date groups,
    study-13 pair settings). No grid, no early stopping; fitted on the fit slice."""

    def __init__(self, name: str, *, nthread: int = 4) -> None:
        self.name, self.rank, self.nthread = name, name == "M4", nthread
        self.booster: Any = None

    def _dmat(self, X: np.ndarray, y: np.ndarray | None, bounds: np.ndarray) -> Any:
        import xgboost as xgb

        kw: dict[str, Any] = {}
        if self.rank:
            kw["qid"] = np.repeat(np.arange(len(bounds) - 1), np.diff(bounds))
        lab = None if y is None else (y + 0.5 if self.rank else y)
        return xgb.DMatrix(X, label=lab, nthread=self.nthread, **kw)

    def params(self, seed: int) -> dict[str, Any]:
        p: dict[str, Any] = {
            "tree_method": "hist",
            "max_bin": 64,
            "nthread": self.nthread,
            "seed": seed,
            "verbosity": 0,
            **BOOSTER_FIXED,
            "min_child_weight": BOOSTER_MCW[self.name],
        }
        if self.rank:
            p.update(
                {
                    "objective": "rank:pairwise",
                    "lambdarank_pair_method": "mean",
                    "lambdarank_num_pair_per_sample": 8,
                }
            )
        else:
            p["objective"] = "reg:squarederror"
        return p

    def fit(self, fd: ml.FitData, *, seed: int) -> dict[str, Any]:
        import time

        import xgboost as xgb

        t0 = time.time()
        params = self.params(seed)
        self.booster = xgb.train(
            params, self._dmat(fd.Xf, fd.yf, fd.bf), num_boost_round=BOOSTER_ROUNDS
        )
        pv = self.predict(fd.Xv, fd.bv)
        ic_v = TieSafeIC(fd.yv, fd.bv)
        return {
            "params": {k: params[k] for k in (*BOOSTER_FIXED, "min_child_weight", "objective")},
            "rounds": BOOSTER_ROUNDS,
            "val_ic": ic_v(pv),
            "val_ic_se": ic_v.se(pv),
            "seconds": time.time() - t0,
            "pred_val": pv,
        }

    def predict(self, X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        import xgboost as xgb

        return self.booster.predict(xgb.DMatrix(X, nthread=self.nthread)).astype(np.float64)


def train_nn3_seed_ts(
    Xf: np.ndarray,
    yf: np.ndarray,
    Xv: np.ndarray,
    yv: np.ndarray,
    bv: np.ndarray,
    lam: float,
    lr: float,
    seed: int,
    *,
    max_epochs: int = 100,
    patience: int = 5,
    batch: int = 10_000,
    threads: int = 1,
) -> dict[str, Any]:
    """:func:`equity_ml.train_nn3_seed` with the tie-safe validation IC (identical otherwise).
    Runs in a worker process without XGBoost."""
    import torch

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ml._nn3_module(Xf.shape[1])
    weights = [m.weight for m in model if isinstance(m, torch.nn.Linear)]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.from_numpy(np.ascontiguousarray(Xf, dtype=np.float32))
    Y = torch.from_numpy(np.ascontiguousarray(yf, dtype=np.float32))
    ic_v = TieSafeIC(np.asarray(yv), np.asarray(bv))
    best_ic, best_state, best_ep, bad, finite = -np.inf, None, -1, 0, True
    ep = 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(X)))
        for a in range(0, len(X), batch):
            idx = perm[a : a + batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            out = model(X[idx]).squeeze(-1)
            loss = ((out - Y[idx]) ** 2).mean() + lam * sum(w.abs().sum() for w in weights)
            if not torch.isfinite(loss):
                finite = False
                break
            loss.backward()
            opt.step()
        if not finite:
            break
        sc = ic_v(ml._forward_chunks(model, Xv))
        if sc > best_ic:
            best_ic, best_ep, bad = sc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    return {
        "state": best_state,
        "val_ic": float(best_ic),
        "best_epoch": best_ep,
        "epochs": ep + 1,
        "finite": finite and best_state is not None,
        "lam": lam,
        "lr": lr,
        "seed": seed,
    }


def _job_nn3_ts(paths: Mapping[str, str], state_out: str, **kw: Any) -> dict[str, Any]:
    import torch

    r = train_nn3_seed_ts(
        ml._load(paths["Xf"]),
        ml._load(paths["yf"]),
        ml._load(paths["Xv"]),
        ml._load(paths["yv"]),
        ml._load(paths["bv"]),
        **kw,
    )
    if r["state"] is not None:
        torch.save(r["state"], state_out)
    r["state"] = state_out if r["state"] is not None else None
    return r


class NN3TS(ml.NN3Model):
    """Study-13 NN3 (grid, 2-seed selection, completion to 10 seeds, early stopping on the
    validation slice) with the tie-safe validation IC."""

    def fit(self, fd: ml.FitData, *, seed: int) -> dict[str, Any]:
        import time

        t0 = time.time()
        paths = self._fd_paths(fd)
        grid = [(lam, lr) for lam in self.lams for lr in self.lrs]
        jobs = [
            {
                "paths": paths,
                "state_out": self.work.file("nn3"),
                "lam": lam,
                "lr": lr,
                "seed": seed + 1000 * gi + s,
                **self.kw,
            }
            for gi, (lam, lr) in enumerate(grid)
            for s in range(min(self.select_seeds, self.n_seeds))
        ]
        runs = ml._run_jobs(_job_nn3_ts, jobs, self.n_jobs)
        ic_v = TieSafeIC(fd.yv, fd.bv)
        vp, bp = paths["Xv"], paths["bv"]
        scores = []
        for lam, lr in grid:
            rs = [r for r in runs if r["lam"] == lam and r["lr"] == lr and r["finite"]]
            if not rs:
                scores.append({"lam": lam, "lr": lr, "val_ic": float("nan")})
                continue
            pv = ml._run_jobs(
                ml._job_predict,
                [{"kind": "nn3", "states": [r["state"] for r in rs], "x_path": vp, "b_path": bp}],
                self.n_jobs,
            )[0]
            scores.append({"lam": lam, "lr": lr, "val_ic": ic_v(pv)})
        ok = [sc for sc in scores if np.isfinite(sc["val_ic"])]
        if not ok:
            raise FloatingPointError("NN3: every grid run diverged")
        best = max(ok, key=lambda sc: sc["val_ic"])
        keep = [
            r for r in runs if r["lam"] == best["lam"] and r["lr"] == best["lr"] and r["finite"]
        ]
        gi = grid.index((best["lam"], best["lr"]))
        more = ml._run_jobs(
            _job_nn3_ts,
            [
                {
                    "paths": paths,
                    "state_out": self.work.file("nn3"),
                    "lam": best["lam"],
                    "lr": best["lr"],
                    "seed": seed + 1000 * gi + 100 + s,
                    **self.kw,
                }
                for s in range(max(self.n_seeds - len(keep), 0))
            ],
            self.n_jobs,
        )
        keep += [r for r in more if r["finite"]]
        self.states = [r["state"] for r in keep]
        pv = ml._run_jobs(
            ml._job_predict,
            [{"kind": "nn3", "states": self.states, "x_path": vp, "b_path": bp}],
            self.n_jobs,
        )[0]
        return {
            "lam": best["lam"],
            "lr": best["lr"],
            "grid": scores,
            "n_seeds": len(self.states),
            "seed_epochs": [r["best_epoch"] for r in keep],
            "val_ic": ic_v(pv),
            "val_ic_se": ic_v.se(pv),
            "n_diverged": int(sum(not r["finite"] for r in [*runs, *more])),
            "seconds": time.time() - t0,
            "pred_val": pv,
        }


def make_model(name: str, *, quick: bool = False, n_jobs: int = 4, nthread: int = 4) -> ml.Model:
    """v3 model factory: M0 / M1 study-13 ridges (tie-safe IC), M2 / M4 fixed boosters, M3 NN3
    (tie-safe IC). M5 is excluded from study 14."""
    if name == "M0":
        return RidgeTS("M0", (0.1, 10.0, 1e3, 1e5), features=ml.M0_FEATURES)
    if name == "M1":
        return RidgeTS("M1", (1.0, 10.0, 100.0, 1e3, 1e4, 1e5))
    if name in ("M2", "M4"):
        return FixedBooster(name, nthread=nthread)
    if name == "M3":
        return NN3TS(
            "M3", n_seeds=2 if quick else 10, select_seeds=1 if quick else 2, n_jobs=n_jobs
        )
    raise ValueError(f"model {name} is not part of study 14")


# ---------------------------------------------------------------------------
# Decile long/short with both cost accountings
# ---------------------------------------------------------------------------
TIE_SEED: int = 20261014  # fixed seed of the random decile tie-breaks (R3)


@dataclass
class LSDual:
    gross: np.ndarray
    long_ret: np.ndarray
    short_ret: np.ndarray
    to_long: np.ndarray  # equal-weight rebalance turnover (study 13)
    to_short: np.ndarray
    to_long_ee: np.ndarray  # entry/exit-only turnover (incumbent re-weighting excluded)
    to_short_ee: np.ndarray
    n_long: np.ndarray
    n_short: np.ndarray
    attr_long: np.ndarray | None = None  # mean of ``attr`` over each leg (e.g. ex-ante beta)
    attr_short: np.ndarray | None = None
    members: list[tuple[np.ndarray, np.ndarray]] | None = None  # (long rows, short rows) per date

    def net(self, cost_per_side: float, accounting: str = "ew") -> np.ndarray:
        if accounting == "ew":
            to = self.to_long + self.to_short
        elif accounting == "ee":
            to = self.to_long_ee + self.to_short_ee
        else:
            raise ValueError(accounting)
        return self.gross - cost_per_side * to

    def two_sided_turnover(self, accounting: str = "ew") -> np.ndarray:
        return (
            (self.to_long + self.to_short)
            if accounting == "ew"
            else (self.to_long_ee + self.to_short_ee)
        )

    def book_attr(self) -> np.ndarray:
        assert self.attr_long is not None and self.attr_short is not None
        return self.attr_long - self.attr_short


def decile_long_short_dual(
    pred: np.ndarray,
    ret: np.ndarray,
    firm: np.ndarray,
    bounds: np.ndarray,
    n_firms: int,
    *,
    n_bins: int = 10,
    tie_seed: int | None = TIE_SEED,
    attr: np.ndarray | None = None,
    keep_members: bool = False,
) -> LSDual:
    """Same portfolios as :func:`equity_ml.decile_long_short` (identical binning, EW legs,
    drift by the holding return), with turnover split into the equal-weight rebalance figure
    (sum |w_new - w_drift|) and the entry/exit-only figure (new names' weights + exited names'
    drifted weights; names held across the rebalance cost nothing).

    Ties (v3, R3): rows with equal predictions are ordered by a seeded uniform draw, not by row
    (= alphabetical) order; with ``tie_seed=None`` the study-13 stable order is used. Continuous
    predictions give exactly the study-13 books either way."""
    nd = len(bounds) - 1
    keys = (
        "gross",
        "long_ret",
        "short_ret",
        "to_long",
        "to_short",
        "to_long_ee",
        "to_short_ee",
        "n_long",
        "n_short",
        "attr_long",
        "attr_short",
    )
    res = {k: np.zeros(nd) for k in keys}
    members: list[tuple[np.ndarray, np.ndarray]] = []
    trng = np.random.default_rng(tie_seed) if tie_seed is not None else None
    scratch = np.zeros(n_firms)
    prev: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "L": (np.zeros(0, np.int64), np.zeros(0)),
        "S": (np.zeros(0, np.int64), np.zeros(0)),
    }
    for i in range(nd):
        a, b = int(bounds[i]), int(bounds[i + 1])
        p = pred[a:b]
        r = np.nan_to_num(ret[a:b])
        f = firm[a:b]
        n = b - a
        if trng is not None:
            order = np.lexsort((trng.random(n), p))
        else:
            order = np.argsort(p, kind="stable")
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(1, n + 1)
        binv = np.ceil(rank * n_bins / n).astype(np.int64)
        legs = (("L", binv == n_bins), ("S", binv == 1))
        if keep_members:
            members.append((a + np.flatnonzero(legs[0][1]), a + np.flatnonzero(legs[1][1])))
        for leg, sel in legs:
            k = int(sel.sum())
            pidx, pw = prev[leg]
            idx = f[sel].astype(np.int64)
            lr = float(r[sel].mean()) if k else 0.0
            scratch[pidx] = pw
            w_old_on_new = scratch[idx]
            wn = 1.0 / k if k else 0.0
            to_new = float(np.abs(wn - w_old_on_new).sum())
            to_entry = float(wn * (w_old_on_new == 0).sum())
            scratch[idx] = 0.0
            to_exit = float(scratch[pidx].sum())  # previous names not in the new book
            scratch[pidx] = 0.0
            to = to_new + to_exit
            to_ee = to_entry + to_exit
            if k and (1.0 + lr) != 0:
                dw = wn * (1.0 + r[sel]) / (1.0 + lr)
            else:
                dw = np.zeros(k)
            prev[leg] = (idx, dw)
            av = float("nan")
            if attr is not None and k:
                with np.errstate(all="ignore"):
                    v = np.asarray(attr[a:b])[sel]
                    av = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            if leg == "L":
                res["long_ret"][i], res["to_long"][i], res["to_long_ee"][i] = lr, to, to_ee
                res["n_long"][i], res["attr_long"][i] = k, av
            else:
                res["short_ret"][i], res["to_short"][i], res["to_short_ee"][i] = lr, to, to_ee
                res["n_short"][i], res["attr_short"][i] = k, av
    res["gross"] = res["long_ret"] - res["short_ret"]
    if attr is None:
        res["attr_long"] = res["attr_short"] = None
    return LSDual(**res, members=members if keep_members else None)


def hedged_series(
    ls: LSDual,
    mktrf: np.ndarray,
    cost_per_side: float,
    *,
    accounting: str = "ew",
    gross: bool = False,
) -> np.ndarray:
    """Index-hedged book: r - h_t * MktRF_t - cost * |h_t - h_{t-1}| (h = ex-ante book beta,
    long-leg mean beta minus short-leg mean beta; hedge turnover charged at the book's cost per
    side; the gross version charges neither)."""
    h = np.nan_to_num(ls.book_attr())
    base = ls.gross if gross else ls.net(cost_per_side, accounting)
    out = base - h * np.asarray(mktrf)
    if not gross:
        out = out - cost_per_side * np.abs(np.diff(np.r_[0.0, h]))
    return out


def break_even_cost(ls: LSDual, accounting: str = "ew") -> float:
    """Cost per side at which the mean net return is zero: mean gross / mean two-sided turnover
    (the per-year ratio is identical)."""
    to = float(ls.two_sided_turnover(accounting).mean())
    return float(ls.gross.mean() / to) if to > 0 else float("nan")


# ---------------------------------------------------------------------------
# Per-horizon statistics
# ---------------------------------------------------------------------------
def ann_sharpe(x: np.ndarray, horizon: str) -> float:
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / sd * np.sqrt(PERIODS_PER_YEAR[horizon])) if sd > 0 else float("nan")


def ann_mean(x: np.ndarray, horizon: str) -> float:
    return float(np.mean(x) * PERIODS_PER_YEAR[horizon])


def nw_t_mean(x: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett) t of the mean."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    e = x - x.mean()
    s = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        s += 2.0 * (1 - lag / (lags + 1)) * float(e[lag:] @ e[:-lag]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else float("nan")


def per_year_table(
    x: np.ndarray, years: np.ndarray, sessions_held: np.ndarray, horizon: str
) -> list[dict[str, Any]]:
    """Calendar-year rows (year of the entry session): compounded return, held sessions,
    periods, annualised Sharpe."""
    x = np.asarray(x, dtype=float)
    out = []
    for y in np.unique(years):
        m = years == y
        v = x[m]
        out.append(
            {
                "year": int(y),
                "ret_compounded": float(np.prod(1 + v) - 1),
                "sessions": int(np.asarray(sessions_held)[m].sum()),
                "periods": int(m.sum()),
                "sharpe": ann_sharpe(v, horizon) if len(v) > 2 else float("nan"),
            }
        )
    return out


def period_stats(
    x: np.ndarray, years: np.ndarray, sessions_held: np.ndarray, mktrf: np.ndarray, horizon: str
) -> dict[str, Any]:
    """Sharpe (x sqrt P), annualised mean (x P), vol, max drawdown of the compounded series,
    alpha / beta on Mkt-RF (NW ceil(P/4) lags), NW-t of the mean (12-month bandwidth),
    calendar-year table and both years rules."""
    P = PERIODS_PER_YEAR[horizon]
    x = np.asarray(x, dtype=float)
    cum = np.cumprod(1.0 + x)
    dd = float((cum / np.maximum.accumulate(cum) - 1.0).min())
    ab = ee.ols_alpha_beta(x, np.asarray(mktrf, dtype=float), nw_lags=nw_lags_alpha(horizon))
    py = per_year_table(x, years, sessions_held, horizon)
    yg = ee.years_gate(py, rule="ex_best_year_positive", min_sessions=MIN_YEAR_SESSIONS)
    yg75 = ee.years_gate(py, rule="share_ge_075", min_sessions=MIN_YEAR_SESSIONS)
    m3 = float(pd.Series(x).skew()) if len(x) > 2 else float("nan")
    m4 = float(pd.Series(x).kurt() + 3.0) if len(x) > 3 else float("nan")
    return {
        "sharpe": ann_sharpe(x, horizon),
        "ann_return": ann_mean(x, horizon),
        "ann_vol": float(x.std(ddof=1) * np.sqrt(P)),
        "max_dd": dd,
        "alpha_ann": float(ab["alpha_daily"] * P),
        "alpha_t": ab["alpha_t"],
        "beta": ab["beta"],
        "beta_t": ab["beta_t"],
        "beta_flag": bool(abs(ab["beta"]) > 0.25),
        "nw_t_mean_12m": nw_t_mean(x, NW_LAGS_12M[horizon]),
        "per_year": py,
        "years_rule": yg,
        "years_rule_share_ge_075": yg75,
        "years_counted": yg["years_counted"],
        "years_positive": yg["years_positive"],
        "periods": int(len(x)),
        "skew": m3,
        "kurtosis": m4,
        "periods_per_year": P,
    }


def n2_bootstrap(
    x: np.ndarray,
    horizon: str,
    *,
    rng: np.random.Generator,
    n_draws: int = 1000,
    block: int | None = None,
) -> dict[str, float]:
    """Centred circular block bootstrap of the net period series: one-sided p for mean <= 0 and
    the MDE ((q95 - q20) of the centred resample means x P). Default block = ~3 months."""
    blk = N2_BLOCK[horizon] if block is None else int(block)
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    boot = ee.circular_block_bootstrap_means(x - m, block=blk, n_draws=n_draws, rng=rng)
    P = PERIODS_PER_YEAR[horizon]
    return {
        "draws": n_draws,
        "block": blk,
        "p": float((np.sum(boot >= m) + 1) / (n_draws + 1)),
        "se_ann_mean": float(boot.std(ddof=1) * P),
        "mde_ann_return": float((np.quantile(boot, 0.95) - np.quantile(boot, 0.20)) * P),
    }


def permute_within_dates(x: np.ndarray, bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return ml.permute_within_dates(x, bounds, rng)


def n1_permutation_dual(
    pred: np.ndarray,
    ret: np.ndarray,
    firm: np.ndarray,
    bounds: np.ndarray,
    n_firms: int,
    horizon: str,
    realised: Mapping[str, float],
    costs: Mapping[str, tuple[float, str]],
    *,
    n_draws: int = 500,
    rng: np.random.Generator,
    beta: np.ndarray | None = None,
    mktrf: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Within-date permutation of the predictions through the same portfolio code; the null
    statistic (annualised Sharpe) is computed for every label in ``costs`` (label -> (cost per
    side, accounting); accounting ``"gross"`` = the gross spread, the v3 N1 gate) from the SAME
    permuted books. With ``beta`` and ``mktrf`` every permuted book is index-hedged with its own
    ex-ante book beta (:func:`hedged_series`)."""
    sh: dict[str, list[float]] = {k: [] for k in costs}
    ar: dict[str, list[float]] = {k: [] for k in costs}
    for _ in range(n_draws):
        ls = decile_long_short_dual(
            permute_within_dates(pred, bounds, rng), ret, firm, bounds, n_firms, attr=beta
        )
        for k, (c, acc) in costs.items():
            if beta is not None:
                x = hedged_series(
                    ls, mktrf, c, accounting="ew" if acc == "gross" else acc, gross=acc == "gross"
                )
            else:
                x = ls.gross if acc == "gross" else ls.net(c, acc)
            sh[k].append(ann_sharpe(x, horizon))
            ar[k].append(ann_mean(x, horizon))
    out = {}
    for k in costs:
        mde = ee.mde_from_null(ar[k])
        out[k] = {
            "draws": n_draws,
            "p": ee.null_p(sh[k], realised[k]),
            "null_sharpe_mean": float(np.mean(sh[k])),
            "null_sharpe_q95": float(np.quantile(sh[k], 0.95)),
            "null_ann_mean": float(np.mean(ar[k])),
            "mde_ann_return": mde,
            "mde_excess_over_null_mean": float(mde - np.mean(ar[k])),
        }
    return out


def cell_verdict(
    net_sharpe: float, n1_p: float, n2_p: float, per_year: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Pre-declared per-cell rule: net SR > 0, N1 p < 0.05 (v3: the GROSS-statistic permutation
    p), N2 p < 0.05 (3-month block), ``ex_best_year_positive`` over years with >= 63 held
    sessions."""
    return ee.event_study_verdict(
        net_sharpe,
        n1_p,
        n2_p,
        list(per_year),
        years_rule="ex_best_year_positive",
        min_sessions=MIN_YEAR_SESSIONS,
    )


# ---------------------------------------------------------------------------
# Ex-ante betas and daily factor attribution (red team R5, R6)
# ---------------------------------------------------------------------------
def name_betas(
    R: np.ndarray,
    mktrf: np.ndarray,
    rf: np.ndarray,
    rows: np.ndarray,
    *,
    win: int = 252,
    minp: int = 200,
) -> np.ndarray:
    """(len(rows) x cols) trailing ``win``-session OLS beta of (r - RF) on Mkt-RF through the
    close of each formation row (point in time), NaN with fewer than ``minp`` pairs."""
    out = np.full((len(rows), R.shape[1]), np.nan, dtype=np.float32)
    mk = np.asarray(mktrf, dtype=np.float64)
    rfv = np.asarray(rf, dtype=np.float64)
    for i, t in enumerate(np.asarray(rows, dtype=np.int64)):
        a = max(0, int(t) - win + 1)
        r = R[a : t + 1].astype(np.float64) - rfv[a : t + 1, None]
        m = mk[a : t + 1]
        ok = np.isfinite(r) & np.isfinite(m)[:, None]
        n = ok.sum(0)
        mm = np.where(ok, m[:, None], 0.0)
        rr = np.where(ok, r, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx, my = mm.sum(0) / n, rr.sum(0) / n
            cov = ((rr - my) * (mm - mx) * ok).sum(0)
            var = (((mm - mx) ** 2) * ok).sum(0)
            out[i] = np.where((n >= minp) & (var > 0), cov / var, np.nan)
    return out


def daily_leg_returns(R: np.ndarray, cols: np.ndarray, entry_row: int, exit_row: int) -> np.ndarray:
    """Daily returns (rows entry+1 .. exit) of an equal-weight buy-and-hold book bought at the
    close of ``entry_row`` (missing daily returns = 0, as in the holding returns); compounding
    them gives the book's holding return exactly."""
    if len(cols) == 0:
        return np.zeros(exit_row - entry_row)
    r = np.nan_to_num(R[entry_row + 1 : exit_row + 1][:, cols].astype(np.float64))
    v = np.vstack([np.ones((1, len(cols))), np.cumprod(1.0 + r, axis=0)])
    return (v[1:].sum(1) / v[:-1].sum(1)) - 1.0


def daily_spread(
    R: np.ndarray, ls: LSDual, firm: np.ndarray, entry_rows: np.ndarray, exit_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Daily long-minus-short spread of a decile book (``ls`` built with ``keep_members``) and the
    session row of each day."""
    assert ls.members is not None
    rets, rows = [], []
    for (lm, sm), e, x in zip(ls.members, entry_rows, exit_rows, strict=True):
        lr = daily_leg_returns(R, firm[lm], int(e), int(x))
        sr = daily_leg_returns(R, firm[sm], int(e), int(x))
        rets.append(lr - sr)
        rows.append(np.arange(int(e) + 1, int(x) + 1))
    return np.concatenate(rets), np.concatenate(rows)


FACTORS: tuple[str, ...] = ("Mkt-RF", "SMB", "HML", "MOM")


def factor_attribution(
    spread: np.ndarray, fac: pd.DataFrame, years: np.ndarray, *, nw_lags: int = 5
) -> list[dict[str, Any]]:
    """Per calendar year: OLS of the daily spread on Mkt-RF, SMB, HML, MOM (French daily) with
    NW standard errors; alpha annualised x 252; each factor's contribution = loading x the sum of
    the factor's daily returns in the year; the summed spread for comparison."""
    out = []
    y = np.asarray(spread, dtype=np.float64)
    F = fac.loc[:, list(FACTORS)].to_numpy(np.float64)
    for yr in np.unique(years):
        m = (years == yr) & np.isfinite(y) & np.isfinite(F).all(1)
        if m.sum() < 30:
            continue
        X = np.column_stack([np.ones(m.sum()), F[m]])
        inv = np.linalg.inv(X.T @ X)
        b = inv @ X.T @ y[m]
        e = y[m] - X @ b
        xe = X * e[:, None]
        S = xe.T @ xe
        for lag in range(1, nw_lags + 1):
            gm = xe[lag:].T @ xe[:-lag]
            S += (1 - lag / (nw_lags + 1)) * (gm + gm.T)
        se = np.sqrt(np.diag(inv @ S @ inv))
        r2 = 1 - float(e @ e) / float(((y[m] - y[m].mean()) ** 2).sum())
        row: dict[str, Any] = {
            "year": int(yr),
            "days": int(m.sum()),
            "sum_spread": float(y[m].sum()),
            "alpha_ann": float(b[0] * SESSIONS_PER_YEAR),
            "alpha_t": float(b[0] / se[0]),
            "r2": r2,
        }
        for j, f in enumerate(FACTORS, 1):
            row[f"b_{f}"] = float(b[j])
            row[f"t_{f}"] = float(b[j] / se[j])
            row[f"contrib_{f}"] = float(b[j] * F[m, j - 1].sum())
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Deflated Sharpe (Bailey & Lopez de Prado 2014), pinned per v2 / design review M1
# ---------------------------------------------------------------------------
def sr0_bldp(V: float, N: int) -> float:
    """Expected max Sharpe under the null: sqrt(V) [(1-g) Phi^-1(1-1/N) + g Phi^-1(1-1/(N e))]."""
    from scipy.stats import norm

    if N <= 1 or not np.isfinite(V):
        return 0.0
    return float(
        np.sqrt(V)
        * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / N) + EULER_GAMMA * norm.ppf(1 - 1 / (N * np.e)))
    )


def psr(sr_hat: float, sr_star: float, n: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe ratio in per-period units (``kurt`` non-excess)."""
    from scipy.stats import norm

    den = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2
    if not den > 0 or n < 2:
        return float("nan")
    return float(norm.cdf((sr_hat - sr_star) * np.sqrt(n - 1) / np.sqrt(den)))


def deflated_sharpe(trials: Mapping[str, Mapping[str, Any] | None], *, n_declared: int) -> dict:
    """``trials``: id -> {"sr_ann", "horizon", "n", "skew", "kurt"} or None (NOT RUN).

    SR_i = annualised net SR; V = sample variance (ddof 1) over the cells that ran; N =
    ``n_declared`` (NOT RUN cells included); DSR_i = PSR of cell i against SR0 / sqrt(P_i) in
    its own period units."""
    ran = {k: v for k, v in trials.items() if v is not None and np.isfinite(v["sr_ann"])}
    srs = np.array([v["sr_ann"] for v in ran.values()], dtype=float)
    V = float(srs.var(ddof=1)) if len(srs) > 1 else float("nan")
    sr0 = sr0_bldp(V, n_declared)
    per = {}
    for k, v in ran.items():
        P = PERIODS_PER_YEAR[v["horizon"]]
        per[k] = {
            "sr_ann": float(v["sr_ann"]),
            "sr0_ann": sr0,
            "dsr": psr(
                v["sr_ann"] / np.sqrt(P),
                sr0 / np.sqrt(P),
                int(v["n"]),
                float(v["skew"]),
                float(v["kurt"]),
            ),
        }
    best = max(per, key=lambda k: per[k]["sr_ann"]) if per else None
    return {
        "N": int(n_declared),
        "n_ran": len(ran),
        "V": V,
        "sd_cross_trial": float(np.sqrt(V)),
        "sr0_ann": sr0,
        "per_cell": per,
        "best_cell": best,
        "best_dsr": per[best]["dsr"] if best else float("nan"),
        "not_run": sorted(k for k in trials if k not in ran),
    }


# ---------------------------------------------------------------------------
# Cell specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    horizon: str
    window: str
    refit: str
    embargo: int
    role: str  # "trial" | "diagnostic"
    panel: str = "grid"  # "grid" (study-14 panel of the horizon) | "study13"
    drop_features: tuple[str, ...] = ()
    ensemble: tuple[str, ...] = ENS5
    reuse_of: str | None = None  # D3 / D7 reuse another cell's fits
    hedged: bool = False  # D7: beta-hedged book
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


TRIALS: tuple[CellSpec, ...] = (
    CellSpec("T1", "W", "expanding", "annual", 1, "trial"),
    CellSpec("T2", "W", "roll5", "annual", 1, "trial"),
    CellSpec("T3", "W", "roll3", "annual", 1, "trial"),
    CellSpec("T4", "M", "roll5", "annual", 1, "trial"),
    CellSpec("T5", "M", "expanding", "annual", 1, "trial", note="study-14 headline"),
    CellSpec("T6", "M", "roll3", "annual", 1, "trial"),
    CellSpec("T7", "Q", "expanding", "annual", 1, "trial"),
    CellSpec("T8", "Q", "roll5", "annual", 1, "trial"),
    CellSpec("T9", "Q", "roll3", "annual", 1, "trial"),
    CellSpec("T10", "W", "roll3", "quarterly", 1, "trial", note="fast corner"),
)
DIAGNOSTICS: tuple[CellSpec, ...] = (
    CellSpec(
        "D0",
        "M",
        "expanding",
        "annual",
        1,
        "diagnostic",
        panel="study13",
        note="T5 (v3 model rules) on the study-13 panel: next-session intraday day 0, "
        "positional lookbacks, no R8 share flag",
    ),
    CellSpec("D1", "M", "expanding", "quarterly", 1, "diagnostic", note="T5 with quarterly refit"),
    CellSpec("D2", "M", "expanding", "annual", 3, "diagnostic", note="T5 with a 3-month embargo"),
    CellSpec(
        "D3",
        "M",
        "expanding",
        "annual",
        1,
        "diagnostic",
        ensemble=ENS3,
        reuse_of="T5",
        note="ENS3 = M2 + M3 + M4 on the T5 fits",
    ),
    CellSpec(
        "D7",
        "M",
        "expanding",
        "annual",
        1,
        "diagnostic",
        reuse_of="T5",
        hedged=True,
        note="T5 index-hedged with its ex-ante book beta (hedge turnover charged)",
    ),
    CellSpec(
        "D5w",
        "W",
        "expanding",
        "annual",
        1,
        "diagnostic",
        drop_features=ml.SURVIVORSHIP_FAMILY,
        note="T1 without the survivorship family",
    ),
    CellSpec(
        "D5m",
        "M",
        "expanding",
        "annual",
        1,
        "diagnostic",
        drop_features=ml.SURVIVORSHIP_FAMILY,
        note="T5 without the survivorship family",
    ),
    CellSpec(
        "D5q",
        "Q",
        "expanding",
        "annual",
        1,
        "diagnostic",
        drop_features=ml.SURVIVORSHIP_FAMILY,
        note="T7 without the survivorship family",
    ),
)
RUN_ORDER: tuple[str, ...] = (
    "T5",
    "D3",
    "D7",
    "D0",
    "T4",
    "T6",
    "T7",
    "T8",
    "T9",
    "D1",
    "D2",
    "T1",
    "T2",
    "T3",
    "T10",
    "D5w",
    "D5m",
    "D5q",
)
ALL_CELLS: dict[str, CellSpec] = {c.cell_id: c for c in (*TRIALS, *DIAGNOSTICS)}
N_TRIALS: int = len(TRIALS)
D6_YEARS: tuple[int, int] = (2014, 2018)
CAP_SECONDS: float = 90 * 60.0
COST: dict[str, float] = {"LIQ": 0.0005, "ALL": 0.0015}
