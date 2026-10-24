"""Tests for eqxs.equity_ml_grid (study 14, window / timeframe grid).

The study-14 pre-declaration (v2) requires, before any fit: the cache key differs across splits and
horizons; annualisation per horizon is right on series with a known Sharpe; the per-horizon embargo
leaves no training label overlapping a test block; the weekly panel's features equal the monthly
panel's on shared dates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eqxs import equity_ml as ml
from eqxs import equity_ml_grid as g


# ---------------------------------------------------------------------------
# Calendars and formation rows
# ---------------------------------------------------------------------------
def _calendar(start: str = "2002-01-01", end: str = "2018-12-31") -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.bdate_range(start, end))


def test_formation_rows() -> None:
    s = _calendar()
    w = g.weekly_rows(s)
    assert s[w[0]] == pd.Timestamp("2004-01-02")
    assert (np.diff(w) == 5).all()
    q = g.quarter_end_rows(s)
    assert set(pd.DatetimeIndex(s[q]).month) == {3, 6, 9, 12}
    me = ml.month_end_rows(s)
    assert set(q) <= set(me) and len(q) * 3 == len(me)
    assert g.PERIODS_PER_YEAR == {"W": 50.4, "M": 12.0, "Q": 4.0}
    assert [g.nw_lags_alpha(h) for h in ("W", "M", "Q")] == [13, 3, 1]
    assert g.N2_BLOCK == {"W": 13, "M": 3, "Q": 1}
    assert g.N2_BLOCK_12M == {"W": 50, "M": 12, "Q": 4}


# ---------------------------------------------------------------------------
# Cache key (design review C1)
# ---------------------------------------------------------------------------
def test_cache_key_differs_across_splits_and_horizons() -> None:
    cols = ["a", "b"]

    def key(sp: g.GridSplit, model: str = "M2", seed: int | None = None) -> tuple[str, str]:
        sd = g.refit_seed(20261013, sp) if seed is None else seed
        k = g.cache_key(model=model, panel_hash="h", columns=cols, split=sp, seed=sd)
        return k, g.cache_filename(model, sp, k)

    exp = g.grid_splits("M", "expanding", "annual", 1)
    r5 = g.grid_splits("M", "roll5", "annual", 1)
    r3 = g.grid_splits("M", "roll3", "annual", 1)
    # same OOS year, only the window differs (roll5 == expanding in 2009 by truncation, but
    # the window label still differs, so the key differs)
    for i in range(len(exp)):
        ks = {key(exp[i])[0], key(r5[i])[0], key(r3[i])[0]}
        fs = {key(exp[i])[1], key(r5[i])[1], key(r3[i])[1]}
        assert len(ks) == 3 and len(fs) == 3
    # only the horizon differs
    w = g.grid_splits("W", "expanding", "annual", 1)
    q = g.grid_splits("Q", "expanding", "annual", 1)
    for i in range(len(exp)):
        assert len({key(exp[i])[0], key(w[i])[0], key(q[i])[0]}) == 3
        assert len({key(exp[i])[1], key(w[i])[1], key(q[i])[1]}) == 3
    # quarterly refits of the same year
    qr = g.grid_splits("M", "expanding", "quarterly", 1)
    y09 = [sp for sp in qr if sp.refit_id.startswith("2009")]
    assert len(y09) == 4
    assert len({key(sp)[0] for sp in y09}) == 4 and len({key(sp)[1] for sp in y09}) == 4
    # embargo, model, seed, panel hash and columns all enter the key
    e3 = g.grid_splits("M", "expanding", "annual", 3)
    assert key(exp[0])[0] != key(e3[0])[0]
    assert key(exp[0], "M3")[0] != key(exp[0], "M4")[0]
    assert key(exp[0], seed=1)[0] != key(exp[0], seed=2)[0]
    k1 = g.cache_key(model="M2", panel_hash="h1", columns=cols, split=exp[0], seed=1)
    k2 = g.cache_key(model="M2", panel_hash="h2", columns=cols, split=exp[0], seed=1)
    k3 = g.cache_key(model="M2", panel_hash="h1", columns=["a"], split=exp[0], seed=1)
    assert len({k1, k2, k3}) == 3


def test_refit_seed_is_first_test_month_index() -> None:
    sp = g.grid_splits("M", "expanding", "annual", 1)[0]
    assert g.refit_seed(20261013, sp) == 20261013 + 2009 * 12
    qr = g.grid_splits("W", "roll3", "quarterly", 1)
    assert [g.refit_seed(0, s) for s in qr[:4]] == [2009 * 12 + 3 * i for i in range(4)]


# ---------------------------------------------------------------------------
# Splitter and embargo
# ---------------------------------------------------------------------------
def test_split_windows_and_validation_lengths() -> None:
    exp = g.grid_splits("M", "expanding", "annual", 1)
    r5 = g.grid_splits("M", "roll5", "annual", 1)
    r3 = g.grid_splits("M", "roll3", "annual", 1)
    assert len(exp) == 10 and exp[0].test == (2009 * 12, 2009 * 12 + 11)
    assert exp[0].fit[0] == g.FIRST_TARGET_MI
    assert r5[0].fit == exp[0].fit and r5[0].val == exp[0].val  # truncated: roll5 == exp 2009
    for sp in r5[1:]:
        win_end = sp.test[0] - 2
        assert sp.fit[0] == win_end - 59 and sp.val == (win_end - 11, win_end)
        assert sp.fit[1] == sp.val[0] - 2
    for sp in r3:
        assert sp.fit[0] == sp.test[0] - 2 - 35
    assert all(sp.val_len == 12 for sp in exp)
    assert all(sp.val_len == 6 for sp in g.grid_splits("W", "roll3", "quarterly", 1))
    assert all(sp.val_len == 24 for sp in g.grid_splits("Q", "expanding", "annual", 1))
    assert all(sp.val_len == 24 for sp in g.grid_splits("Q", "roll5", "annual", 1))
    assert all(sp.val_len == 12 for sp in g.grid_splits("Q", "roll3", "annual", 1))
    assert len(g.grid_splits("M", "expanding", "quarterly", 1)) == 40
    with pytest.raises(ValueError):
        g.grid_splits("W", "expanding", "annual", 0)  # below the weekly minimum
    with pytest.raises(NotImplementedError):
        g.grid_splits("Q", "expanding", "quarterly", 1)


def _labels(sessions: pd.DatetimeIndex, horizon: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = g.formation_rows(sessions, horizon)
    d = sessions[rows]
    rows = rows[(d >= pd.Timestamp("2004-01-01")) & (d < pd.Timestamp("2019-01-01"))]
    R = np.zeros((len(sessions), 1), dtype=np.float32)
    _, entry, exit_ = ml.holding_returns(R, rows)
    ok = entry >= 0
    tmi = np.full(len(rows), -1)
    tmi[ok] = ml.month_index(sessions[entry[ok]])
    return tmi, entry, exit_


@pytest.mark.parametrize("cell", [c.cell_id for c in (*g.TRIALS, *g.DIAGNOSTICS)])
def test_embargo_leaves_no_overlap(cell: str) -> None:
    spec = g.ALL_CELLS[cell]
    s = _calendar()
    tmi, entry, exit_ = _labels(s, spec.horizon)
    splits = g.grid_splits(spec.horizon, spec.window, spec.refit, spec.embargo)
    assert g.check_embargo(splits, tmi, entry, exit_) == []
    # every refit has fit rows and test rows except possibly an infeasible tail
    for sp in splits:
        assert (ml.mask_months(tmi, sp.fit) & (tmi >= 0)).any()


def test_embargo_checker_catches_overlapping_labels() -> None:
    """Tiled labels (exit of one = entry of the next) never overlap; the checker must catch
    the design-review M3 case: 3-month labels on MONTHLY rows with a 1-month embargo."""
    s = _calendar()
    rows = ml.month_end_rows(s)
    d = s[rows]
    rows = rows[(d >= pd.Timestamp("2004-01-01")) & (d < pd.Timestamp("2019-01-01"))]
    entry = rows + 1
    exit_ = np.r_[rows[3:] + 1, np.full(3, -1)]  # 3-month label on every month-end
    ok = exit_ >= 0
    tmi = np.where(ok, ml.month_index(s[np.clip(entry, 0, len(s) - 1)]), -1)
    splits = g.grid_splits("M", "expanding", "annual", 1)
    assert len(g.check_embargo(splits, tmi, entry, np.where(ok, exit_, 0))) > 0
    tmi_w, en_w, ex_w = _labels(s, "W")
    assert g.check_embargo(g.grid_splits("W", "expanding", "annual", 1), tmi_w, en_w, ex_w) == []


# ---------------------------------------------------------------------------
# Annualisation per horizon (design review C3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("horizon", ["W", "M", "Q"])
def test_annualisation_known_sharpe(horizon: str) -> None:
    P = g.PERIODS_PER_YEAR[horizon]
    mu_ann, sr_ann = 0.10, 0.9
    sd_ann = mu_ann / sr_ann
    years = 400
    n = int(years * P)
    rng = np.random.default_rng(11)
    x = rng.normal(mu_ann / P, sd_ann / np.sqrt(P), n)
    se_sr = np.sqrt((1 + sr_ann**2 / (2 * P)) / years)
    assert abs(g.ann_sharpe(x, horizon) - sr_ann) < 4 * se_sr
    assert abs(g.ann_mean(x, horizon) - mu_ann) < 4 * sd_ann / np.sqrt(years)
    mkt = rng.normal(0.0, 0.05 / np.sqrt(P), n)
    yrs = np.repeat(np.arange(years), int(np.ceil(P)))[:n] + 2000
    st = g.period_stats(x, yrs, np.full(n, 252 / P), mkt, horizon)
    assert abs(st["sharpe"] - sr_ann) < 4 * se_sr
    assert abs(st["ann_vol"] - sd_ann) < 0.05 * sd_ann
    assert abs(st["alpha_ann"] - mu_ann) < 6 * sd_ann / np.sqrt(years)
    # bootstrap MDE in annual units: ~ (z95 + z80) * sd_ann / sqrt(years) for iid data
    n2 = g.n2_bootstrap(x, horizon, rng=rng, n_draws=400)
    expected = (1.645 + 0.8416) * sd_ann / np.sqrt(years)
    assert 0.8 * expected < n2["mde_ann_return"] < 1.2 * expected
    # study-13 monthly helper would be wrong off the monthly horizon
    if horizon != "M":
        assert abs(ml.monthly_sharpe(x) - sr_ann) > 4 * se_sr


def test_nw_t_mean_matches_iid_t() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(0.01, 0.05, 5000)
    t_iid = x.mean() / x.std(ddof=1) * np.sqrt(len(x))
    assert abs(g.nw_t_mean(x, 0) - t_iid) < 0.05
    assert abs(g.nw_t_mean(x, 12) - t_iid) < 0.15 * abs(t_iid)


# ---------------------------------------------------------------------------
# Decile portfolios with both cost accountings
# ---------------------------------------------------------------------------
def test_dual_decile_matches_study13_and_splits_turnover() -> None:
    rng = np.random.default_rng(5)
    n_firms, nd = 300, 30
    rows = []
    for d in range(nd):
        f = np.sort(rng.choice(n_firms, size=int(rng.integers(150, 250)), replace=False))
        rows.append(
            pd.DataFrame(
                {"d": d, "f": f, "p": rng.normal(size=len(f)), "r": rng.normal(0.01, 0.1, len(f))}
            )
        )
    df = pd.concat(rows, ignore_index=True)
    df.loc[df.sample(frac=0.02, random_state=1).index, "r"] = np.nan
    b = ml.date_bounds(df["d"].to_numpy())
    p, r, f = df["p"].to_numpy(), df["r"].to_numpy(), df["f"].to_numpy()
    ref = ml.decile_long_short(p, r, f, b, 0.0005, n_firms)
    new = g.decile_long_short_dual(p, r, f, b, n_firms)
    np.testing.assert_allclose(new.gross, ref.gross, atol=1e-12)
    np.testing.assert_allclose(new.to_long, ref.to_long, atol=1e-12)
    np.testing.assert_allclose(new.to_short, ref.to_short, atol=1e-12)
    np.testing.assert_allclose(new.net(0.0005, "ew"), ref.net, atol=1e-12)
    assert (new.to_long_ee <= new.to_long + 1e-12).all()
    assert new.to_long_ee[0] == pytest.approx(1.0) and new.to_long[0] == pytest.approx(1.0)
    # a persistent signal: incumbents are re-weighted, entries/exits are not all of turnover
    p2 = np.concatenate([np.linspace(-1, 1, b[i + 1] - b[i]) for i in range(nd)])
    pers = g.decile_long_short_dual(p2, r, f, b, n_firms)
    assert (pers.to_long_ee <= pers.to_long + 1e-12).all()
    be = g.break_even_cost(new)
    assert np.isclose(new.net(be).mean(), 0.0, atol=1e-12)


def test_n1_dual_costs_share_draws() -> None:
    rng = np.random.default_rng(9)
    n_firms, nd = 200, 24
    d = np.repeat(np.arange(nd), 150)
    f = np.tile(np.arange(150), nd)
    p = rng.normal(size=len(d))
    r = 0.05 * p + rng.normal(0, 0.1, len(d))
    b = ml.date_bounds(d)
    ls = g.decile_long_short_dual(p, r, f, b, n_firms)
    real = {"ew1": g.ann_sharpe(ls.net(0.0005), "M"), "ew2": g.ann_sharpe(ls.net(0.001), "M")}
    out = g.n1_permutation_dual(
        p,
        r,
        f,
        b,
        n_firms,
        "M",
        real,
        {"ew1": (0.0005, "ew"), "ew2": (0.001, "ew")},
        n_draws=40,
        rng=np.random.default_rng(1),
    )
    assert out["ew1"]["p"] < 0.05 and out["ew2"]["p"] < 0.05
    assert out["ew2"]["null_ann_mean"] < out["ew1"]["null_ann_mean"]


def test_cell_verdict_uses_ex_best_year_rule() -> None:
    py = [
        {"year": 2009 + i, "ret_compounded": v, "sessions": 250}
        for i, v in enumerate([0.5, -0.01, 0.02, 0.03, -0.02])
    ]
    v = g.cell_verdict(0.3, 0.01, 0.01, py)
    assert v["verdict"] == "PASS" and v["years"]["rule"] == "ex_best_year_positive"
    py2 = [
        {**r, "ret_compounded": -abs(r["ret_compounded"])} if r["year"] != 2009 else r for r in py
    ]
    assert g.cell_verdict(0.3, 0.01, 0.01, py2)["fails"] == ["ex_best_year_mean<=0"]
    short = [{"year": 2018, "ret_compounded": -0.5, "sessions": 40}, *py]
    assert g.cell_verdict(0.3, 0.01, 0.01, short)["years"]["years_counted"] == 5


# ---------------------------------------------------------------------------
# Deflated Sharpe
# ---------------------------------------------------------------------------
def test_deflated_sharpe_native_units_matches_annual_approx() -> None:
    from scipy.stats import norm

    trials = {
        f"T{i}": {
            "sr_ann": s,
            "horizon": h,
            "n": int(10 * g.PERIODS_PER_YEAR[h]),
            "skew": 0.0,
            "kurt": 3.0,
        }
        for i, (s, h) in enumerate([(0.2, "W"), (-0.3, "M"), (0.9, "Q"), (0.1, "M")], 1)
    }
    trials["T5"] = None  # NOT RUN: counts in N, not in V
    out = g.deflated_sharpe(trials, n_declared=10)
    srs = np.array([0.2, -0.3, 0.9, 0.1])
    assert out["V"] == pytest.approx(srs.var(ddof=1))
    assert out["sr0_ann"] == pytest.approx(np.sqrt(srs.var(ddof=1)) * g.sr0_bldp(1.0, 10))
    assert out["not_run"] == ["T5"] and out["best_cell"] == "T3"
    approx = norm.cdf((0.9 - out["sr0_ann"]) * np.sqrt(10))
    assert out["per_cell"]["T3"]["dsr"] == pytest.approx(approx, abs=0.03)


# ---------------------------------------------------------------------------
# Weekly vs monthly features on shared dates (design review M2)
# ---------------------------------------------------------------------------
def _price_world(seed: int = 4) -> tuple[ml.PriceMatrices, np.ndarray]:
    rng = np.random.default_rng(seed)
    sessions = pd.DatetimeIndex(pd.bdate_range("2002-06-03", "2005-12-30"))
    n, k = len(sessions), 12
    r = rng.normal(0.0004, 0.02, (n, k))
    r[rng.random((n, k)) < 0.01] = np.nan
    r[:150, 3] = np.nan  # a late listing
    adj = 20.0 * np.cumprod(1.0 + np.nan_to_num(r), axis=0)
    vol = rng.uniform(5e4, 5e5, (n, k))
    splits = np.zeros((n, k))
    splits[400, 1] = 2.0
    rows, cols = np.nonzero(np.isfinite(r))
    pm = ml.build_price_matrices(
        sessions,
        np.array([f"S{i}" for i in range(k)]),
        rows,
        cols,
        adj[rows, cols],
        adj[rows, cols],
        vol[rows, cols],
        splits[rows, cols],
    )
    me = ml.month_end_rows(sessions)
    uni = ml.build_universe(pm, me, min_price=0.0, min_dv=0.0)
    mkt = ml.ew_market_return(pm.R, me, uni.all_)
    return pm, mkt


def test_weekly_equals_monthly_features_on_shared_dates() -> None:
    pm, mkt = _price_world()
    s = pm.sessions
    me = ml.month_end_rows(s)
    wk = g.weekly_rows(s)
    shared = np.intersect1d(me, wk)
    assert len(shared) >= 5
    fm = g.price_features_sessions(pm, me, mkt)
    fw = g.price_features_sessions(pm, wk, mkt)
    im = np.searchsorted(me, shared)
    iw = np.searchsorted(wk, shared)
    for name in ml.PRICE_FEATURES:
        a, b = fm[name][im], fw[name][iw]
        assert np.array_equal(np.isnan(a), np.isnan(b)), name
        np.testing.assert_allclose(a[np.isfinite(a)], b[np.isfinite(b)], rtol=1e-6, err_msg=name)
    # session definitions
    t = int(shared[-1])
    lc = np.r_[0.0, np.cumsum(np.log1p(np.nan_to_num(pm.R[:, 0].astype(np.float64))))]
    mom = np.expm1(lc[t - 21 + 1] - lc[t - 252 + 1])
    assert fw["mom_12_1"][iw[-1], 0] == pytest.approx(mom, rel=1e-5)
    # event features depend only on the formation date
    ev = pd.DataFrame(
        {
            "col": [0, 0, 1, 2],
            "day0_row": [60, 330, 200, 500],
            "clean": [True, True, False, True],
            "signal": [0.01, -0.02, 0.03, np.nan],
        }
    )
    ev["day0"] = s[ev["day0_row"].to_numpy()]
    em = ml.event_features(ev, s, me, pm.n_cols)
    ew = ml.event_features(ev, s, wk, pm.n_cols)
    for name in ml.EVENT_FEATURES:
        a, b = em[name][im], ew[name][iw]
        assert np.array_equal(np.isnan(a), np.isnan(b), equal_nan=True), name
        np.testing.assert_array_equal(a[np.isfinite(a)], b[np.isfinite(b)])
    # fundamental query "12 months earlier" = last session on or before T - 365 days
    firm = np.zeros(len(shared), np.int64)
    qm = g.make_fund_query_sessions(pm, me, im, firm, np.ones(len(shared), np.int64))
    qw = g.make_fund_query_sessions(pm, wk, iw, firm, np.ones(len(shared), np.int64))
    np.testing.assert_array_equal(qm.ts_12m, qw.ts_12m)
    np.testing.assert_array_equal(qm.ts, qw.ts)
    d0 = s[shared[-1]] - pd.Timedelta(days=365)
    prev = s[s <= d0][-1]
    assert qw.ts_12m[-1] == ml.close_ts_utc_ns(pd.DatetimeIndex([prev]))[0]


# ---------------------------------------------------------------------------
# v3 addendum (study-13 red team R1-R9)
# ---------------------------------------------------------------------------
def _blocks(n_dates: int = 6, n: int = 120, seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n_dates * n)
    return y, ml.date_bounds(np.repeat(np.arange(n_dates), n))


def test_tie_safe_ic_constant_predictor_is_zero() -> None:
    y, b = _blocks()
    ic = g.TieSafeIC(y, b)
    assert ic(np.zeros(len(y))) == 0.0
    # study 13's FastIC gives the IC of row order for a constant predictor
    assert ml.FastIC(y, b)(np.zeros(len(y))) != 0.0
    # a constant predictor's IC does not depend on row order
    perm = np.random.default_rng(0).permutation(len(y) // 6)
    y2 = np.concatenate([y[b[i] : b[i + 1]][perm] for i in range(6)])
    assert g.TieSafeIC(y2, b)(np.zeros(len(y))) == 0.0


def test_tie_safe_ic_matches_average_rank_spearman_with_66pct_ties() -> None:
    from scipy.stats import spearmanr

    y, b = _blocks()
    rng = np.random.default_rng(4)
    p = rng.normal(size=len(y))
    tied = rng.random(len(y)) < 0.66
    p[tied] = 0.0  # 66 % of rows share one value
    per = g.TieSafeIC(y, b).per_date(p)
    ref = [spearmanr(p[b[i] : b[i + 1]], y[b[i] : b[i + 1]])[0] for i in range(len(b) - 1)]
    np.testing.assert_allclose(per, ref, atol=1e-12)
    np.testing.assert_allclose(per, ml.ic_by_date(p, y, b), atol=1e-12)


def test_decile_ties_broken_at_random_not_by_row_order() -> None:
    n_firms, n = 400, 400
    d = np.zeros(n, dtype=int)
    b = ml.date_bounds(d)
    f = np.arange(n)
    r = np.zeros(n)
    const = np.zeros(n)
    ls = g.decile_long_short_dual(const, r, f, b, n_firms, keep_members=True)
    lm, sm = ls.members[0]
    assert len(lm) == 40 and len(sm) == 40
    assert not np.array_equal(lm, np.arange(360, 400))  # stable order would pick the last rows
    assert not np.array_equal(sm, np.arange(40))
    ls2 = g.decile_long_short_dual(const, r, f, b, n_firms, keep_members=True)
    assert np.array_equal(ls2.members[0][0], lm)  # seeded: reproducible
    old = g.decile_long_short_dual(const, r, f, b, n_firms, keep_members=True, tie_seed=None)
    assert np.array_equal(old.members[0][0], np.arange(360, 400))


def test_assert_nondegenerate() -> None:
    _, b = _blocks()
    p = np.random.default_rng(1).normal(size=b[-1])
    assert g.assert_nondegenerate(p, b, "x") > 1
    p[b[2] : b[3]] = 1.0
    with pytest.raises(g.DegeneratePredictor):
        g.assert_nondegenerate(p, b, "x")


def test_fixed_boosters_are_not_constant() -> None:
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    nd, n, k = 20, 200, 6
    X = rng.uniform(-0.5, 0.5, (nd * n, k)).astype(np.float32)
    did = np.repeat(np.arange(nd), n)
    yraw = 0.3 * X[:, 0] + rng.normal(0, 1, len(did))
    b = ml.date_bounds(did)
    y = ml.rank_within_dates(yraw, b)
    fi, vi = np.arange(15 * n), np.arange(15 * n, nd * n)
    fd = ml.FitData(
        X[fi],
        y[fi],
        ml.date_bounds(did[fi]),
        X[vi],
        y[vi],
        ml.date_bounds(did[vi]),
        [f"c{i}" for i in range(k)],
    )
    for name in ("M2", "M4"):
        m = g.make_model(name, nthread=2)
        info = m.fit(fd, seed=1)
        assert info["rounds"] == 300 and info["params"]["min_child_weight"] == g.BOOSTER_MCW[name]
        pv = info["pred_val"]
        assert g.assert_nondegenerate(pv, fd.bv, name) > 10
        assert info["val_ic"] > 0 and np.isfinite(info["val_ic_se"])
    assert g.BOOSTER_MCW == {"M2": 100.0, "M4": 1.0}


def test_name_betas_and_hedge() -> None:
    rng = np.random.default_rng(3)
    n = 400
    mk = rng.normal(0, 0.01, n)
    rf = np.full(n, 0.0001)
    true = np.array([0.5, 1.0, 1.5])
    R = (rf[:, None] + mk[:, None] * true[None, :] + rng.normal(0, 0.002, (n, 3))).astype(
        np.float32
    )
    B = g.name_betas(R, mk, rf, np.array([150, 300, 399]))
    assert np.isnan(B[0]).all()  # < 200 pairs
    np.testing.assert_allclose(B[1:], np.tile(true, (2, 1)), atol=0.05)
    ls = g.LSDual(
        *(np.zeros(3) for _ in range(9)),
        attr_long=np.array([1.2, 1.0, 1.4]),
        attr_short=np.array([1.0, 1.0, 1.0]),
    )
    ls.gross = np.array([0.01, 0.02, -0.01])
    mkt = np.array([0.05, -0.02, 0.01])
    hs = g.hedged_series(ls, mkt, 0.0005)
    h = np.array([0.2, 0.0, 0.4])
    np.testing.assert_allclose(hs, ls.gross - h * mkt - 0.0005 * np.abs(np.diff(np.r_[0.0, h])))


def test_daily_leg_returns_compound_to_holding_return_and_attribution() -> None:
    rng = np.random.default_rng(8)
    n, k = 300, 30
    R = rng.normal(0.0005, 0.02, (n, k)).astype(np.float32)
    R[rng.random((n, k)) < 0.02] = np.nan
    cols = np.array([1, 4, 7, 9])
    d = g.daily_leg_returns(R, cols, 10, 31)
    hold = np.mean(np.prod(1 + np.nan_to_num(R[11:32][:, cols].astype(np.float64)), axis=0) - 1)
    assert np.prod(1 + d) - 1 == pytest.approx(hold, rel=1e-10)
    # attribution recovers known loadings
    days = 2 * 252
    fac = pd.DataFrame(rng.normal(0, 0.01, (days, 4)), columns=list(g.FACTORS))
    spread = 0.0002 + fac.to_numpy() @ np.array([0.3, -0.2, 0.5, 0.8]) + rng.normal(0, 0.001, days)
    years = np.repeat([2009, 2010], 252)
    att = g.factor_attribution(spread, fac, years)
    assert [a["year"] for a in att] == [2009, 2010]
    assert att[0]["b_MOM"] == pytest.approx(0.8, abs=0.05) and att[1]["b_HML"] == pytest.approx(
        0.5, abs=0.05
    )
    assert att[0]["alpha_ann"] == pytest.approx(0.0002 * 252, abs=0.05)  # 3 se


def test_n1_gross_label_ignores_cost() -> None:
    rng = np.random.default_rng(9)
    n_firms, nd = 200, 24
    d = np.repeat(np.arange(nd), 150)
    f = np.tile(np.arange(150), nd)
    p = rng.normal(size=len(d))
    r = 0.05 * p + rng.normal(0, 0.1, len(d))
    b = ml.date_bounds(d)
    ls = g.decile_long_short_dual(p, r, f, b, n_firms)
    real = {"gross": g.ann_sharpe(ls.gross, "M"), "ew1": g.ann_sharpe(ls.net(0.0015), "M")}
    out = g.n1_permutation_dual(
        p,
        r,
        f,
        b,
        n_firms,
        "M",
        real,
        {"gross": (0.0, "gross"), "ew1": (0.0015, "ew")},
        n_draws=30,
        rng=np.random.default_rng(2),
    )
    assert abs(out["gross"]["null_ann_mean"]) < abs(out["ew1"]["null_ann_mean"])
    assert out["gross"]["p"] < 0.05
