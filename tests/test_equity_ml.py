"""Tests for eqxs.equity_ml (study 13, ML cross-sectional return model)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eqxs import equity_ml as ml

# ---------------------------------------------------------------------------
# Synthetic world: prices, events, XBRL facts (all dates shift together with ``offset``)
# ---------------------------------------------------------------------------
N_FIRMS = 10
BASE_START = pd.Timestamp("2009-01-05")   # Monday; +364 d keeps weekdays, no Feb 29 in range
BASE_END = pd.Timestamp("2010-12-31")


def _world(offset_days: int = 0, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    off = pd.Timedelta(days=offset_days)
    sessions = pd.bdate_range(BASE_START, BASE_END) + off
    n, k = len(sessions), N_FIRMS
    r = rng.normal(0.0004, 0.02, (n, k))
    r[rng.random((n, k)) < 0.01] = np.nan                       # occasional missing bars
    adj = 20.0 * np.cumprod(1.0 + np.nan_to_num(r), axis=0)
    vol = rng.uniform(5e4, 5e5, (n, k))
    splits = np.zeros((n, k))
    splits[300, 0] = 2.0                                          # a 2:1 split for firm 0
    rows, cols = np.nonzero(np.isfinite(r))
    pm = ml.build_price_matrices(sessions, np.array([f"S{i}" for i in range(k)]), rows, cols,
                                 adj[rows, cols], adj[rows, cols], vol[rows, cols], splits[rows, cols])
    # events: quarterly day 0 per firm with jitter, ~ every 63 sessions
    ev = []
    for c in range(k):
        r0 = 5 + c
        while r0 < n - 2:
            ev.append({"cik": 1000 + c, "col": c, "day0_row": r0, "clean": bool(rng.random() > 0.3)})
            r0 += 63 + int(rng.integers(-6, 7))
    ev = pd.DataFrame(ev)
    ev["day0"] = sessions[ev["day0_row"].to_numpy()]
    with np.errstate(all="ignore"):
        mkt = np.nan_to_num(np.nanmean(np.where(np.isfinite(pm.R), pm.R, 0.0), axis=1))
    ev["signal"] = ml.announcement_signal(ev["day0_row"].to_numpy(), ev["col"].to_numpy(), pm.R, mkt)
    # XBRL-like facts: each quarterly filing (40 days after quarter end, 12:00 UTC) reports the
    # quarter and its prior-year comparative; 10-K for Q4 also reports the annual value.
    facts = []
    qends = pd.date_range("2006-03-31", "2010-09-30", freq="QE")
    for c in range(k):
        cik = 1000 + c
        base = rng.uniform(1e8, 1e9)
        for qi, qe in enumerate(qends):
            ts = (qe + pd.Timedelta(days=40)).normalize() + pd.Timedelta(hours=12)
            form = "10-K" if qe.month == 12 else "10-Q"
            adsh = f"{cik}-{qi}"

            def add(tag: str, uom: str, value: float, ddate: pd.Timestamp, qtrs: int,
                    _ts: pd.Timestamp = ts, _form: str = form, _adsh: str = adsh, _cik: int = cik) -> None:
                facts.append({"adsh": _adsh, "cik": _cik, "tag": tag, "uom": uom, "value": value,
                              "ddate": ddate, "qtrs": qtrs, "form": _form, "ts": _ts})

            g = 1.0 + 0.01 * qi
            add("Assets", "USD", base * 3 * g, qe, 0)
            add("Liabilities", "USD", base * 1.5 * g, qe, 0)
            add("StockholdersEquity", "USD", base * 1.5 * g, qe, 0)
            add("CashAndCashEquivalentsAtCarryingValue", "USD", base * 0.3 * g, qe, 0)
            add("EntityCommonStockSharesOutstanding", "shares", 1e7 * (1 + 0.002 * qi),
                qe + pd.Timedelta(days=30), 0)
            for tag, mult in (("NetIncomeLoss", 0.02), ("Revenues", 0.3), ("CostOfRevenue", 0.18),
                              ("NetCashProvidedByUsedInOperatingActivities", 0.03)):
                if qe.month != 12:
                    add(tag, "USD", base * mult * g * rng.uniform(0.8, 1.2), qe, 1)
                else:
                    add(tag, "USD", base * mult * 4 * g, qe, 4)
            eps = 1.0 + 0.1 * np.sin(qi) + 0.05 * c + rng.normal(0, 0.05)
            add("EarningsPerShareDiluted", "USD/shares", eps, qe, 1)
            if qi >= 4:
                add("EarningsPerShareDiluted", "USD/shares", eps - 0.1, qends[qi - 4], 1)
    num = pd.DataFrame(facts)
    num["ddate"] = pd.to_datetime(num["ddate"]) + off
    num["ts"] = pd.to_datetime(num["ts"]).dt.tz_localize("UTC") + off
    return {"pm": pm, "events": ev, "num": num, "mkt": mkt}


def _features(w: dict, me_rows: np.ndarray, pos: np.ndarray) -> pd.DataFrame:
    pm = w["pm"]
    uni = ml.build_universe(pm, me_rows, min_price=0.0, min_dv=0.0)
    mkt_ew = ml.ew_market_return(pm.R, me_rows, uni.all_)
    pf = ml.price_features(pm, me_rows, mkt_ew, pos)
    ef = ml.event_features(w["events"], pm.sessions, me_rows, pm.n_cols, pos)
    kk, ff = np.meshgrid(np.arange(len(pos)), np.arange(pm.n_cols), indexing="ij")
    kk, ff = kk.ravel(), ff.ravel()
    out = {f"{n}": a[kk, ff] for n, a in {**pf, **ef}.items()}
    facts = ml.group_facts(ml.prepare_facts(w["num"]))
    sue = ml.sue_events(facts["eps"])
    q = ml.make_fund_query(pm, me_rows, pos[kk], ff, 1000 + ff)
    fund, _ = ml.fundamental_features(facts, q, sue_ev=sue)
    out.update(fund)
    df = pd.DataFrame(out)
    df.insert(0, "firm", ff)
    df.insert(0, "k", kk)
    return df


def _me_rows(pm: ml.PriceMatrices) -> np.ndarray:
    return ml.month_end_rows(pm.sessions)


def test_pit_shift_one_year_every_feature_moves_with_the_data() -> None:
    w0, w1 = _world(0), _world(364)
    # formation rows = the base calendar's month-ends; in the shifted world the same rows are the
    # same sessions moved by 364 days (52 weeks: weekdays align, same calendar month)
    me = _me_rows(w0["pm"])
    d0, d1 = w0["pm"].sessions[me], w1["pm"].sessions[me]
    assert ((d1 - d0) == pd.Timedelta(days=364)).all() and (d1.month == d0.month).all()
    pos = np.arange(13, len(me))
    f0, f1 = _features(w0, me, pos), _features(w1, me, pos)
    for name in ml.NUMERIC_FEATURES:
        a, b = f0[name].to_numpy(), f1[name].to_numpy()
        assert np.isfinite(a).sum() > 0, f"{name} never computed in the synthetic world"
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True, err_msg=name)


def test_pit_no_leakage_from_data_after_formation() -> None:
    w = _world(0)
    me = _me_rows(w["pm"])
    p = len(me) - 6
    t = int(me[p])
    t_close = pd.Timestamp(w["pm"].sessions[t]).tz_localize("America/New_York") + pd.Timedelta(hours=16)
    base = _features(w, me, np.array([p]))
    w2 = _world(0)
    pm = w2["pm"]
    rng = np.random.default_rng(1)
    pm.R[t + 1:] = rng.normal(0, 0.1, pm.R[t + 1:].shape).astype(np.float32)
    pm.adj[t + 1:] *= 3.0
    pm.dv[t + 1:] *= 0.1
    pm.raw_close[t + 1:] *= 5.0
    ev = w2["events"]
    late = ev["day0_row"] > t
    ev.loc[late, "signal"] = 99.0
    ev.loc[late, "clean"] = True
    num = w2["num"]
    after = num["ts"] > t_close.tz_convert("UTC")
    num.loc[after, "value"] = num.loc[after, "value"] * 7.0 + 1.0
    w2["num"] = num
    pert = _features(w2, me, np.array([p]))
    for name in ml.NUMERIC_FEATURES:
        np.testing.assert_allclose(base[name], pert[name], rtol=1e-6, equal_nan=True, err_msg=name)


def test_pit_latest_respects_amendments_and_periods() -> None:
    ns = 10**9 * 86_400
    f = ml.Facts(cik=np.array([1, 1, 1, 1]), mk=np.array([10, 13, 10, 13]),
                 ts=np.array([100, 200, 300, 400]) * ns, value=np.array([1.0, 2.0, 1.5, 2.5]))
    v, mk = ml.pit_latest(f, np.array([1, 1, 1, 1, 1]), np.array([50, 150, 250, 350, 450]) * ns)
    # before anything: NaN; then period 10; then period 13; an amendment of the OLD period (ts 300)
    # does not replace the newer period; the amendment of period 13 (ts 400) does.
    assert np.isnan(v[0]) and list(v[1:]) == [1.0, 2.0, 2.0, 2.5] and list(mk[1:]) == [10, 13, 13, 13]
    old = ml.pit_for_period(f, np.array([1, 1]), np.array([10, 10]), np.array([250, 350]) * ns)
    assert list(old) == [1.0, 1.5]


def test_pit_ttm_quarterly_vs_annual() -> None:
    ns = 10**9 * 86_400
    fq = ml.Facts(cik=np.ones(4, int), mk=np.array([3, 6, 9, 12]), ts=np.array([10, 20, 30, 40]) * ns,
                  value=np.array([1.0, 2.0, 3.0, 4.0]))
    fa = ml.Facts(cik=np.ones(1, int), mk=np.array([9]), ts=np.array([35]) * ns, value=np.array([7.0]))
    v, mk = ml.pit_ttm(fq, fa, np.ones(3, int), np.array([31, 36, 41]) * ns)
    # t=31: only 3 quarters, no annual -> NaN; t=36: annual (period 9); t=41: 4 quarters to period 12
    assert np.isnan(v[0]) and v[1] == 7.0 and mk[1] == 9 and v[2] == 10.0 and mk[2] == 12


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------
def test_splitter_embargo_and_validation_slice() -> None:
    first = 2004 * 12 + 1
    for sp in ml.expanding_splits(range(2009, 2019), first):
        t0 = sp.year * 12
        assert sp.test == (t0, t0 + 11)
        # no training target month at or after the embargo month (t0 - 1)
        assert sp.fit[1] < t0 - 1 and sp.val[1] < t0 - 1 and sp.val[1] == t0 - 2
        assert sp.val[1] - sp.val[0] + 1 == 12
        assert sp.fit[1] == sp.val[0] - 2 and sp.fit[0] == first
        assert t0 - 1 in sp.embargoed and sp.val[0] - 1 in sp.embargoed
    mi = np.arange(first, 2019 * 12)
    for sp in ml.expanding_splits([2012], first):
        train = mi[ml.mask_months(mi, sp.fit) | ml.mask_months(mi, sp.val)]
        assert train.max() <= sp.test[0] - 2


# ---------------------------------------------------------------------------
# Preprocessing, portfolios, nulls
# ---------------------------------------------------------------------------
def test_normalize_within_date() -> None:
    v = np.array([1.0, np.nan, 3.0, 2.0, 10.0, -5.0])
    z, miss = ml.normalize_within_date(v, np.array([0, 3, 6]))
    assert list(miss) == [0, 1, 0, 0, 0, 0]
    assert z[0] == -0.5 and z[2] == 0.5 and z[1] == 0.0
    assert z[3] == 0.0 and z[4] == 0.5 and z[5] == -0.5


def test_decile_long_short_turnover_and_costs() -> None:
    n = 20
    firm = np.tile(np.arange(n), 2)
    pred = np.r_[np.arange(n), np.arange(n)].astype(float)    # same ranking both months
    ret = np.zeros(2 * n)
    res = ml.decile_long_short(pred, ret, firm, np.array([0, n, 2 * n]), 0.001, n)
    assert list(res.n_long) == [2, 2] and list(res.to_long) == [1.0, 0.0] and list(res.to_short) == [1.0, 0.0]
    assert res.net[0] == pytest.approx(-0.002) and res.net[1] == 0.0
    pred2 = np.r_[np.arange(n), -np.arange(n)].astype(float)   # flip: full replacement
    res2 = ml.decile_long_short(pred2, ret, firm, np.array([0, n, 2 * n]), 0.0, n)
    assert res2.to_long[1] == pytest.approx(2.0)


def test_holding_returns_window() -> None:
    R = np.full((10, 1), 0.01, dtype=np.float32)
    R[5, 0] = np.nan
    me = np.array([2, 6, 8])
    h, e, x = ml.holding_returns(R, me)
    # month-end 2 -> entry row 3, exit row 7: rows 4..7 earn (row 5 missing = 0)
    assert e[0] == 3 and x[0] == 7 and h[0, 0] == pytest.approx(1.01 ** 3 - 1, rel=1e-5)
    assert e[1] == 7 and x[1] == 9 and e[2] == -1


def test_nulls_basic() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.02, 0.01, 120)
    assert ml.n2_block_bootstrap(x, rng=rng)["p"] < 0.01
    assert ml.n2_block_bootstrap(-x, rng=rng)["p"] > 0.9


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _synthetic_panel(n_dates: int = 40, n_firms: int = 250, seed: int = 3) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    cols = ml.input_columns(["mom_12_1", "sue", "log_mcap", "x4"], sic=False)
    X = np.zeros((n_dates * n_firms, len(cols)), dtype=np.float32)
    y = np.zeros(n_dates * n_firms)
    did = np.repeat(np.arange(n_dates), n_firms)
    for d in range(n_dates):
        sl = slice(d * n_firms, (d + 1) * n_firms)
        raw = rng.uniform(-0.5, 0.5, (n_firms, 4))
        X[sl, :4] = raw
        sig = raw[:, 0] ** 2 + 0.5 * np.sign(raw[:, 3]) * raw[:, 1]   # U-shape in mom + interaction
        y[sl] = ml.rank_unit(sig + rng.normal(0, 0.02, n_firms))
    return X, y, did, cols


def _fd(x: np.ndarray, y: np.ndarray, did: np.ndarray, cols: list[str], cut: int, vcut: int) -> tuple:
    f, v, t = did < cut, (did >= cut) & (did < vcut), did >= vcut
    b = ml.date_bounds
    fd = ml.FitData(x[f], y[f], b(did[f]), x[v], y[v], b(did[v]), cols)
    return fd, x[t], y[t], b(did[t])


def test_planted_nonlinear_signal_recovered_by_m2_and_m3_not_m0() -> None:
    X, y, did, cols = _synthetic_panel()
    fd, Xt, yt, bt = _fd(X, y, did, cols, 28, 34)
    m0 = ml.RidgeModel("M0", (1.0,), features=ml.M0_FEATURES)
    m0.fit(fd, seed=0)
    ic0 = np.mean(ml.ic_by_date(m0.predict(Xt, bt), yt, bt))
    m2 = ml.XGBModel("M2", rank=False, grid=[ml.XGB_GRID[1] | {"min_child_weight": 10}], max_rounds=300,
                     patience=30, nthread=2)
    m2.fit(fd, seed=0)
    ic2 = np.mean(ml.ic_by_date(m2.predict(Xt, bt), yt, bt))
    assert abs(ic0) < 0.1 and ic2 > 0.5
    if not ml.torch_available():
        pytest.skip("torch not installed")
    m3 = ml.NN3Model("M3", lams=(1e-5,), lrs=(1e-2,), n_seeds=2, select_seeds=1, n_jobs=1,
                     max_epochs=60, patience=8, batch=512)
    m3.fit(fd, seed=0)
    ic3 = np.mean(ml.ic_by_date(m3.predict(Xt, bt), yt, bt))
    assert ic3 > 0.4


def test_m4_rank_objective_runs() -> None:
    X, y, did, cols = _synthetic_panel(n_dates=20, n_firms=100)
    fd, Xt, yt, bt = _fd(X, y, did, cols, 12, 16)
    m4 = ml.XGBModel("M4", rank=True, grid=[ml.XGB_GRID[0]], max_rounds=50, patience=10, nthread=2)
    info = m4.fit(fd, seed=0)
    assert np.isfinite(info["val_ic"]) and np.isfinite(m4.predict(Xt, bt)).all()


@pytest.mark.skipif(not ml.torch_available(), reason="torch not installed")
def test_m5_smoke_tiny_panel() -> None:
    X, y, did, cols = _synthetic_panel(n_dates=10, n_firms=40)
    fd, Xt, yt, bt = _fd(X, y, did, cols, 6, 8)
    m5 = ml.SetTransformerModel("M5", n_seeds=1, n_jobs=1, max_epochs=2, patience=1)
    info = m5.fit(fd, seed=0)
    p = m5.predict(Xt, bt)
    assert p.shape == (len(Xt),) and np.isfinite(p).all() and info["n_seeds"] == 1


def test_ensemble_rank_normalised_mean() -> None:
    b = np.array([0, 3])
    e = ml.ensemble([np.array([1.0, 2.0, 3.0]), np.array([30.0, 20.0, 10.0])], b)
    np.testing.assert_allclose(e, [0.0, 0.0, 0.0])


def test_pit_clean_ignores_forward_looking_flags() -> None:
    amb = pd.Series(["", "multi_202_in_quarter(2)", "cadence_prefers(2010-01-05)", "intraday_acceptance",
                     "gap_long(130d);multi_202_in_quarter(2)", "no_9.01_exhibit", None, "weekend_announce"])
    assert list(ml.pit_clean(amb)) == [True, True, True, False, False, False, True, False]
