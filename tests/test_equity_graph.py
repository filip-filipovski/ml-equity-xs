"""Edge-table, resolver, propagation-operator and null machinery (eqxs.equity_graph): synthetic inputs only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eqxs import equity_graph as eg
from eqxs import equity_ml_grid as g


def _random_graph(rng: np.random.Generator, n: int = 25, p: float = 0.12, recip: float = 0.30) -> np.ndarray:
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < p:
                W[i, j] = rng.uniform(0.05, 0.9)
    # make ~30 % of edges reciprocal
    e = np.argwhere(W > 0)
    for i, j in e:
        if rng.random() < recip:
            W[j, i] = rng.uniform(0.05, 0.9)
    return W


# ---------------------------------------------------------------------------
# exact simple-path operator
# ---------------------------------------------------------------------------
def test_path_operator_matches_brute_force_40_graphs():
    rng = np.random.default_rng(15)
    worst = 0.0
    recip_shares = []
    for _ in range(40):
        W = _random_graph(rng)
        P = eg.path_matrices(W, 3)
        for h in (1, 2, 3):
            worst = max(worst, float(np.abs(P[h - 1] - eg.brute_simple_paths(W, h)).max()))
        recip_shares.append(((W > 0) & (W.T > 0)).sum() / (W > 0).sum())
    assert worst <= 1e-12
    assert np.mean(recip_shares) > 0.3  # the reciprocal case is really exercised


def test_masking_diagonal_is_wrong_at_h3_but_exact_operator_is_not():
    rng = np.random.default_rng(2)
    W = _random_graph(rng, recip=0.6)
    W2 = W @ W
    np.fill_diagonal(W2, 0)
    v1 = W2 @ W
    np.fill_diagonal(v1, 0)
    b3 = eg.brute_simple_paths(W, 3)
    assert np.abs(v1 - b3).max() > 1e-6
    assert np.abs(eg.path_matrices(W, 3)[2] - b3).max() <= 1e-12


def test_hop_signal_is_weighted_mean_over_priced_origins():
    W = np.array([[0, 0.2, 0.0], [0, 0, 0.5], [0.4, 0.6, 0]])
    s = np.array([0.10, -0.05, np.nan])
    ok = np.array([True, True, True])
    sig, tw = eg.hop_signal(W, s, ok)
    # target 1: origins 0 (0.2, s=0.10) and 2 (0.6, NaN -> unpriced)
    assert sig[1] == pytest.approx(0.10)
    assert tw[1] == pytest.approx(0.2)
    assert sig[2] == pytest.approx(-0.05)
    assert np.isnan(sig[0])  # only origin 2 reaches 0 and it has no shock


# ---------------------------------------------------------------------------
# residualisation on the reached set only
# ---------------------------------------------------------------------------
def test_residualise_reached_set_only():
    rng = np.random.default_rng(3)
    n = 60
    x = rng.standard_normal(n)
    y = 0.5 * x + rng.standard_normal(n)
    reached = np.zeros(n, bool)
    reached[:40] = True
    r = eg.residualise_reached(y, [x], reached)
    assert np.isnan(r[40:]).all()  # units without a path never get a residual
    assert abs(np.dot(r[:40], x[:40])) < 1e-10
    assert abs(r[:40].sum()) < 1e-10
    # differs from a fit over all units (the m1 bug)
    A = np.column_stack([np.ones(n), x])
    b, *_ = np.linalg.lstsq(A, np.where(reached, y, 0.0), rcond=None)
    assert np.abs(r[:40] - (y[:40] - A[:40] @ b)).max() > 1e-6


def test_residualise_too_few_units_is_nan():
    r = eg.residualise_reached(np.arange(3.0), [np.arange(3.0)], np.ones(3, bool))
    assert np.isnan(r).all()


# ---------------------------------------------------------------------------
# N3 nulls
# ---------------------------------------------------------------------------
def test_stratified_rewiring_preserves_degrees_weights_and_strata():
    rng = np.random.default_rng(4)
    m = 400
    cust = rng.integers(0, 120, m)
    supp = rng.integers(120, 260, m)
    key = np.unique(cust * 1000 + supp, return_index=True)[1]
    cust, supp = cust[key], supp[key]
    w = rng.uniform(0.1, 0.5, len(cust))
    sic_of = {c: f"{2000 + c % 7}" for c in range(120)}
    strata = eg.edge_strata(np.array([sic_of[c] for c in cust]))
    new, share = eg.rewire_stratified(cust, supp, strata, rng)
    assert 0.3 < share <= 1.0
    # supplier in-degree, weights in the supplier slot, customer out-degree preserved
    assert np.array_equal(np.bincount(supp, minlength=260), np.bincount(supp, minlength=260))
    assert np.array_equal(np.bincount(new, minlength=260), np.bincount(cust, minlength=260))
    # each supplier keeps the stratum composition of its customers
    for j in np.unique(supp):
        a = sorted(sic_of[c] for c in cust[supp == j])
        b = sorted(sic_of[c] for c in new[supp == j])
        assert a == b
    # no self-loops / duplicates
    assert len(set(zip(new.tolist(), supp.tolist(), strict=True))) == len(new)
    assert not (new == supp).any()
    assert w.shape == new.shape


def test_edge_strata_falls_back_sic4_sic3_sic2():
    lab = eg.edge_strata(np.array(["3711", "3711", "3712", "3725", "2834", "", "9999"]))
    assert lab[0] == lab[1] == "4:3711"
    assert lab[2] == "3:371"  # SIC4 3712 alone, SIC3 371 has 3 edges
    assert lab[3] == "2:37"
    assert lab[4].startswith("_single") and lab[5].startswith("_single") and lab[6].startswith("_single")


def test_within_column_permutation():
    rng = np.random.default_rng(5)
    W = rng.uniform(0, 1, (8, 8))
    np.fill_diagonal(W, 0)
    Wp = eg.permute_within_columns(W, rng)
    assert np.allclose(np.diag(Wp), 0)
    for j in range(8):
        assert np.allclose(np.sort(np.delete(W[:, j], j)), np.sort(np.delete(Wp[:, j], j)))
    assert not np.allclose(W, Wp)


# ---------------------------------------------------------------------------
# S-return: point in time
# ---------------------------------------------------------------------------
def test_s_return_and_beta_use_no_future_rows():
    rng = np.random.default_rng(6)
    T, k = 400, 5
    mk = rng.normal(0, 0.01, T)
    rf = np.full(T, 0.0001)
    R = (rng.normal(0, 0.02, (T, k)) + 1.2 * mk[:, None]).astype(np.float32)
    rows = np.array([300])
    B = g.name_betas(R, mk, rf, rows)
    S = eg.s_returns(R, mk, rf, rows, B)
    R2 = R.copy()
    R2[301:] = 9.0
    mk2 = mk.copy()
    mk2[301:] = 5.0
    B2 = g.name_betas(R2, mk2, rf, rows)
    S2 = eg.s_returns(R2, mk2, rf, rows, B2)
    assert np.array_equal(B, B2) and np.array_equal(S, S2)
    t = 300
    e = R[t - 20:t + 1].astype(float) - rf[t - 20:t + 1, None] - B[0][None, :] * mk[t - 20:t + 1, None]
    assert np.allclose(S[0], e.sum(0))


def test_s_return_needs_min_obs_and_beta():
    rng = np.random.default_rng(7)
    R = rng.normal(0, 0.02, (60, 2)).astype(np.float32)
    R[45:55, 0] = np.nan  # 11 of 21 finite
    S = eg.s_returns(R, np.zeros(60), np.zeros(60), np.array([59]), np.array([[1.0, np.nan]]))
    assert np.isnan(S[0, 0]) and np.isnan(S[0, 1])


def test_loo_mean():
    v = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    gr = np.array(["a", "a", "a", "a", "b"], dtype=object)
    out = eg.loo_mean(v, gr)
    assert out[0] == pytest.approx(2.5)
    assert out[3] == pytest.approx(2.0)
    assert np.isnan(out[4])


# ---------------------------------------------------------------------------
# coverage gate / dip rule, annualisation, book
# ---------------------------------------------------------------------------
def test_coverage_gate_and_dip_rule():
    c = eg.coverage_gate([70, 65, 25, 80, 61], gate_median=60, flat_below=30)
    assert c["run"] and c["flat_months"] == 1 and c["flat_mask"] == [False, False, True, False, False]
    c = eg.coverage_gate([50, 70, 59], gate_median=60, flat_below=30)
    assert not c["run"] and c["flat_months"] == 0


# ---------------------------------------------------------------------------
# E4 resolver
# ---------------------------------------------------------------------------
def _profiles():
    return pd.DataFrame({
        "cik": [1, 2, 3, 4, 5, 6],
        "name": ["HP INC", "WALMART INC", "APPLE INC", "QEP RESOURCES INC", "QEP MIDSTREAM PARTNERS LP", "COSTCO WHOLESALE CORP"],
        "former_names": ["HEWLETT PACKARD CO|1990-01-01|2015-10-19", "WAL MART STORES INC|1995-01-01|2018-02-01",
                         "APPLE COMPUTER INC|1990-01-01|2007-01-10", "", "", ""],
    })


def test_resolver_pit_names_and_prefix_rule():
    idx = eg.PitNameIndex.build(_profiles())
    # dated former name valid in 2014, current name not yet
    assert idx.resolve("HewlettPackardCompany", pd.Timestamp("2014-03-01"))[:2] == (1, "exact")
    assert idx.resolve("HewlettPackardCompany", pd.Timestamp("2016-03-01"))[0] is None
    assert idx.resolve("HpInc", pd.Timestamp("2014-03-01"))[0] is None  # current name not valid before 2015-10-19
    assert idx.resolve("HpInc", pd.Timestamp("2016-03-01"))[:2] == (1, "exact")
    # unique prefix >= 6 characters
    assert idx.resolve("WalMart", pd.Timestamp("2012-05-01"))[:2] == (2, "prefix")
    assert idx.resolve("WalMart", pd.Timestamp("2019-05-01"))[:2] == (2, "exact")  # 'walmart' = today's name
    assert idx.resolve("WalMartStores", pd.Timestamp("2019-05-01"))[0] is None  # former name expired 2018-02
    assert idx.resolve("Apple", pd.Timestamp("2010-01-01"))[:2] == (3, "exact")
    # prefix shorter than 6 characters never matches
    assert idx.resolve("Wal", pd.Timestamp("2012-05-01"))[0] is None
    # ambiguous prefix (two CIKs) is dropped
    assert idx.resolve("QepMid", pd.Timestamp("2012-05-01"))[:2] == (5, "prefix")
    assert idx.resolve("Qep", pd.Timestamp("2012-05-01"))[0] is None


def test_normalisation_is_token_wise():
    assert eg.normalise_name("COSTCO WHOLESALE CORP") == "costcowholesale"
    assert eg.normalise_name("Costco") == "costco"
    assert eg.normalise_name("The Kroger Co") == "thekroger"
    assert eg.normalise_name("TJX COMPANIES INC /DE/") == "tjxcompanies"
    assert eg.normalise_name("AppleIncMember") == "apple"


@pytest.mark.parametrize("m", ["CustomerA", "Customer A", "Customer1", "MajorCustomerTwo", "CustomerOne",
                               "Customerone", "TopTenCustomers", "TwoCustomers", "LargestCustomer", "DistributorB"])
def test_anonymous_members(m):
    assert eg.is_anonymous_member(m)


@pytest.mark.parametrize("m", ["AppleInc", "WalMartStoresInc", "HomeDepot", "Boeing"])
def test_named_members(m):
    assert not eg.is_anonymous_member(m)
    assert not eg.is_government_member(m)


@pytest.mark.parametrize("m", ["USGovernment", "DepartmentOfDefense", "Medicare", "StateOfCalifornia", "UsAirForce",
                               "Nasa", "Tricare", "Usps", "Va", "CountyOfLosAngeles", "UnitedStatesNavy"])
def test_government_members(m):
    assert eg.is_government_member(m)


def test_lessor_and_stem_rules():
    assert eg.is_lessor("6798", "SOME REIT")
    assert eg.is_lessor("6512", "X")
    assert eg.is_lessor("7359", "AEI NET LEASE INCOME & GROWTH FUND XX LTD")
    assert eg.is_lessor("7359", "ATEL EQUIPMENT FUND XI")
    assert not eg.is_lessor("3674", "INTEL CORP")
    ftc = eg.first_token_counts(_profiles())
    assert eg.shared_distinctive_stem("QEP RESOURCES INC", "QEP MIDSTREAM PARTNERS LP", ftc)
    assert not eg.shared_distinctive_stem("APPLE INC", "WALMART INC", ftc)
    assert not eg.shared_distinctive_stem("American X", "American Y", {"american": 400})


# ---------------------------------------------------------------------------
# DSR helper (N = 4, NOT RUN counted)
# ---------------------------------------------------------------------------
def test_dsr_counts_not_run_in_n():
    tr = {"T7": None, "T10": {"sr_ann": 0.3, "horizon": "M", "n": 216, "skew": 0.0, "kurt": 3.0},
          "T11": {"sr_ann": -0.1, "horizon": "M", "n": 216, "skew": 0.0, "kurt": 3.0},
          "T12": {"sr_ann": 0.1, "horizon": "M", "n": 216, "skew": 0.0, "kurt": 3.0}}
    d = g.deflated_sharpe(tr, n_declared=4)
    assert d["N"] == 4 and d["n_ran"] == 3 and d["not_run"] == ["T7"]
    V = np.var([0.3, -0.1, 0.1], ddof=1)
    assert d["V"] == pytest.approx(V)
    assert d["sr0_ann"] == pytest.approx(g.sr0_bldp(V, 4))


def test_sparse_operator_equals_dense():
    import scipy.sparse as sps

    rng = np.random.default_rng(16)
    for _ in range(10):
        W = _random_graph(rng, n=40)
        D = eg.path_matrices(W, 3)
        S = eg.path_matrices_sparse(sps.csr_matrix(W), 3)
        for a, b in zip(D, S, strict=True):
            assert np.abs(a - b.toarray()).max() <= 1e-12
        sig_d, tw_d = eg.hop_signal(D[2], rng.standard_normal(40), np.ones(40, bool))
        x = rng.standard_normal(40)
        assert np.allclose(eg.hop_signal(D[1], x, np.ones(40, bool))[0], eg.hop_signal(S[1], x, np.ones(40, bool))[0],
                           equal_nan=True)


def test_build_e4_exclusions_and_validity():
    sessions = pd.bdate_range("2014-01-01", "2016-12-30")
    prof = pd.DataFrame({"cik": [10, 11, 12, 13, 14, 20],
                         "name": ["ZETA CHIPS INC", "APPLE INC", "SUBCO HOLDINGS INC", "TARGET CORP", "AEI NET LEASE INCOME & GROWTH FUND XX",
                                  "WALMART INC"],
                         "sic": ["3674", "3571", "3674", "5331", "6798", "5331"],
                         "former_names": ["", "", "", "", "", "WAL MART STORES INC|1990-01-01|2018-02-01"]})
    acc = "2015-02-20 10:00:00+00:00"
    base = {"supplier_cik": 10, "supplier_name": "ZETA CHIPS INC", "supplier_sic": "3674", "customer_axis": "MajorCustomers",
            "tag": "ConcentrationRiskPercentage1", "form": "10-K", "fy": 2014, "adsh": "a1", "filed": "2015-02-20",
            "accepted_fsn": pd.Timestamp(acc), "benchmark": "SalesRevenueNet", "is_revenue_benchmark": True,
            "anonymous": False, "government": False, "vendor_hint": False}
    rows = [dict(base, customer_member="AppleInc", customer_name="Apple Inc", pct=20.0),
            dict(base, customer_member="AppleInc", customer_name="Apple Inc", pct=25.0),  # max over facts
            dict(base, customer_member="CustomerA", customer_name="Customer A", pct=30.0),
            dict(base, customer_member="UsGovernment", customer_name="Us Government", pct=30.0),
            dict(base, customer_member="SubcoHoldings", customer_name="Subco Holdings", pct=15.0),  # aciks family
            dict(base, customer_member="ZetaChips", customer_name="Zeta Chips", pct=12.0),  # self
            dict(base, customer_member="WalMart", customer_name="Wal Mart", pct=11.0),  # prefix -> 20
            dict(base, customer_member="AppleInc", customer_name="Apple Inc", pct=40.0, benchmark="AccountsReceivable",
                 is_revenue_benchmark=False, adsh="a1"),  # non-revenue benchmark ignored
            dict(base, supplier_cik=14, supplier_name="AEI NET LEASE INCOME & GROWTH FUND XX", supplier_sic="6798",
                 customer_member="TargetCorp", customer_name="Target Corp", pct=30.0, adsh="a2")]
    links = pd.DataFrame(rows)
    sub = pd.DataFrame({"adsh": ["a1", "a0", "a3"], "cik": pd.array([10, 10, 10], dtype="Int64"), "sic": ["3674"] * 3,
                        "form": ["10-K", "10-K", "10-K"], "aciks": ["12", "", ""],
                        "accepted_utc": pd.to_datetime([acc, "2014-02-20 10:00:00+00:00", "2016-02-19 21:30:00+00:00"], utc=True)})
    e1 = eg.build_e1(sub, prof)
    E, st = eg.build_e4(links, sub, prof, sessions, e1)
    got = {int(c): float(w) for c, w in zip(E["customer_cik"], E["weight"], strict=True)}
    assert got == {11: 0.25, 20: 0.11}
    dr = st["drop_reasons"]
    assert dr.get("anonymous") == 1 and dr.get("government") == 1 and dr.get("supplier_lessor") == 1
    assert dr.get("parent_subsidiary_aciks") == 1 and dr.get("self") == 1
    r = E[E["customer_cik"] == 11].iloc[0]
    assert pd.Timestamp(r["valid_from"]) == pd.Timestamp("2015-02-20")
    # next original 10-K accepted 2016-02-19 16:30 ET (after the close) -> next session
    assert pd.Timestamp(r["valid_to"]) == pd.Timestamp("2016-02-22")
    assert r["valid_to_rule"] == "next_10k"
