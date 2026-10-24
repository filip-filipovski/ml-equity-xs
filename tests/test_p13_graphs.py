"""Unit tests for experiments/gnn/build_graphs.py on a tiny synthetic world (never touches the real data)."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "gnn"))
import common  # noqa: E402
import build_graphs as bg  # noqa: E402

D = [pd.Timestamp("2010-01-29"), pd.Timestamp("2010-02-26"), pd.Timestamp("2010-03-31")]
NF = 12


def _world(tmp_path, monkeypatch):
    panels, edges, graphs = tmp_path / "panels", tmp_path / "edges", tmp_path / "graphs"
    for p in (panels, edges, graphs):
        p.mkdir()
    monkeypatch.setattr(common, "PANELS", panels); monkeypatch.setattr(common, "EDGES", edges); monkeypatch.setattr(common, "GRAPHS", graphs)
    rows = []
    for i, d in enumerate(D):
        for f in range(NF):
            if f == 11 and i == 0:
                continue                      # firm 11 absent at snapshot 0
            rows.append((i, f, 1000 + f, d, 0.5 + 0.1 * f))
    pd.DataFrame(rows, columns=["date_id", "firm", "cik", "date", "raw_beta_252"]).astype({"cik": "Int64"}).to_parquet(panels / "panel_M.parquet")
    (panels / "panel_M.meta.json").write_text(json.dumps({"n_rows": len(rows)}))
    common.panel_meta.cache_clear()
    monkeypatch.setattr(common, "HORIZONS", common.HORIZONS)

    def ts(s):
        return pd.Timestamp(s, tz="UTC") if s else pd.NaT

    e1 = []
    for f in range(NF):
        grp = "1111" if f < 6 else "2222"
        if f == 5:                            # firm 5 moves from industry 1111 to 2222 on 2010-03-10 (between snapshots 1 and 2)
            e1 += [(1000 + f, "1111", ts("1990-01-01"), ts("2010-03-10")), (1000 + f, "2222", ts("2010-03-10"), None)]
        else:
            e1.append((1000 + f, grp, ts("1990-01-01"), None))
    e1 = pd.DataFrame(e1, columns=["cik", "sic4", "valid_from", "valid_to"])
    e4 = pd.DataFrame({"customer_cik": [1001, 1003], "supplier_cik": [1002, 1008],
                       "valid_from": [pd.Timestamp("2010-03-01"), pd.Timestamp("2009-01-01")], "valid_to": [pd.NaT, pd.Timestamp("2011-01-01")]})
    e5 = pd.DataFrame({"data_year": [2008] * 4, "sell_code": ["A", "B", "A", "B"], "buy_code": ["B", "A", "A", "B"], "weight": [0.3, 0.1, 0.0, 0.0],
                       "valid_from": [pd.Timestamp("2010-01-01", tz="UTC")] * 4, "valid_to": [pd.Timestamp("2011-01-01", tz="UTC")] * 4})
    sicmap = pd.DataFrame({"sic4": ["1111", "2222"], "bea": ["A", "B"], "kept": [True, True]})
    # daily returns: sessions up to 2010-03-31 plus 10 sessions after it; firms (2k, 2k+1) share a factor; planted spike after snapshot 2
    sess = pd.bdate_range("2009-06-01", "2010-04-14").to_numpy("datetime64[ns]")
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.01, (len(sess), NF)).astype(np.float32)
    for k in range(NF // 2):
        base[:, 2 * k + 1] += 0.9 * base[:, 2 * k]
    R = base.copy(); end2 = int(np.searchsorted(sess, np.datetime64(D[2]), side="right"))
    R_spike = R.copy(); R_spike[end2:, 0] = 50.0; R_spike[end2:, 4] = -50.0; R_spike[end2:, 7] += 30.0
    srcs = bg.Sources(e1=e1, e4=e4, e5=e5, sicmap=sicmap, R=R, sessions=sess)
    return bg.make_ctx("M", with_beta=True), srcs, R_spike


@pytest.fixture(autouse=True)
def _clear_meta_cache():
    yield
    common.panel_meta.cache_clear()


def _edge_set(g, i):
    s, d, w = common.snapshot_edges(g, i)
    return {(int(a), int(b)): float(x) for a, b, x in zip(s, d, w)}


def test_e4_edge_absent_then_present(tmp_path, monkeypatch):
    ctx, srcs, _ = _world(tmp_path, monkeypatch)
    g = bg.build_relation("M", "E4_CUSTSUPP", ctx, srcs)
    # customer 1001 (firm 1) -> supplier 1002 (firm 2): valid_from 2010-03-01 is after snapshot 1, <= snapshot 2
    assert (1, 2) not in _edge_set(g, 1) and (2, 1) not in _edge_set(g, 1)
    e2 = _edge_set(g, 2)
    assert e2[(1, 2)] == 1.0 and e2[(2, 1)] == -1.0                     # sign: +1 customer->supplier, -1 supplier->customer
    assert _edge_set(g, 0).get((3, 8)) == 1.0 and _edge_set(g, 1).get((3, 8)) == 1.0   # the long-lived edge is always there


def test_industry_change_moves_e1_neighbours(tmp_path, monkeypatch):
    ctx, srcs, _ = _world(tmp_path, monkeypatch)
    g = bg.build_relation("M", "E1_SIC4", ctx, srcs)
    for i in (0, 1):
        e = _edge_set(g, i)
        assert all((5, j) in e for j in (0, 1, 2, 3, 4)) and not any((5, j) in e for j in range(6, 12))
    e = _edge_set(g, 2)
    assert all((5, j) in e for j in range(6, 12)) and not any((5, j) in e for j in range(0, 5))
    assert all((j, 5) in e for j in range(6, 12))                        # both directions
    assert not any((a, b) in e for a in range(5) for b in range(6, 12))   # no cross-industry links
    assert set(common.snapshot_edges(g, 0)[0]) <= set(range(11)) - {11}   # absent firm 11 never an endpoint at snapshot 0


def test_corr_knn_ignores_future_spike(tmp_path, monkeypatch):
    ctx, srcs, R_spike = _world(tmp_path, monkeypatch)
    kw = dict(k=3, window=120, min_overlap=60)
    g = bg.build_relation("M", "CORR_KNN", ctx, srcs, **kw)
    s, d, w, st = bg.corr_snapshot(ctx, srcs, 2, R=R_spike, **kw)
    a, b = int(g["ptr"][2]), int(g["ptr"][3])
    assert np.array_equal(s, g["src"][a:b]) and np.array_equal(d, g["dst"][a:b]) and np.array_equal(w, g["w"][a:b])
    assert st["last_session"] <= np.datetime64(D[2])
    # the planted structure is found: partner of firm 0 is firm 1 and is ranked first
    e = [(int(x), int(y)) for x, y in zip(s, d)]
    assert (0, 1) in e and (1, 0) in e and (4, 5) in e
    # orientation: messages flow src -> dst, so a firm's own <= k neighbours are its IN-edges (only positive correlations are kept)
    assert np.bincount(d, minlength=NF).max() <= 3
    assert bg.MIN_OVERLAP == 189


def test_corr_matches_numpy_pairwise(tmp_path, monkeypatch):
    ctx, srcs, _ = _world(tmp_path, monkeypatch)
    R = srcs.R.copy(); R[:40, 3] = np.nan                                  # a firm with missing history
    s, d, w, _ = bg.corr_snapshot(ctx, srcs, 2, k=2, window=100, min_overlap=50, R=R)
    end = int(np.searchsorted(srcs.sessions, np.datetime64(D[2]), side="right")); X = R[end - 100:end].astype(np.float64)
    X = X - np.nanmean(X, axis=1, keepdims=True)
    i = [j for j, (a, b) in enumerate(zip(s, d)) if a == 0 and b == 1][0]
    c = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
    assert abs(w[i] - c) < 1e-4


def test_endpoints_present_cap_and_e5(tmp_path, monkeypatch):
    ctx, srcs, _ = _world(tmp_path, monkeypatch)
    for rel in bg.INTERVAL_RELS:
        g = bg.build_relation("M", rel, ctx, srcs, cap=3)
        for i in range(3):
            pres = set(ctx.present(i).tolist()); e = _edge_set(g, i)
            assert all(a in pres and b in pres for a, b in e)
        if rel != "E4_CUSTSUPP":
            assert g["meta"]["max_sampled_per_firm"] <= 3
    # E1 with cap 3 on a group of 6: every firm samples exactly 3 (6 > cap + 1)
    s, d, w, st = bg.e1_snapshot(ctx, srcs, 0, cap=3)
    assert st["max_sampled"] == 3
    # cap is independent of how many candidates exist; with cap 20 a group of <= 21 is a full clique
    g = bg.build_relation("M", "E1_SIC4", ctx, srcs, cap=20)
    assert len(_edge_set(g, 1)) == 6 * 5 + 6 * 5
    # E5: industries A (firms 0-5) -> B (6-11), A is the supplier: +share on supplier->customer, -share on customer->supplier
    g5 = bg.build_relation("M", "E5_BEA", ctx, srcs, cap=20)
    e = _edge_set(g5, 1)
    assert e[(0, 6)] == pytest.approx(0.3) and e[(6, 0)] == pytest.approx(-0.3)
    assert not any(a in range(6) and b in range(6) for a, b in e)
    pairs = {(a, b) for a, b in e}
    assert all((b, a) in pairs for a, b in pairs)                       # both directions stored
    assert len(e) == 2 * 6 * 6                                          # 6 x 6 pairs, both directions, snapshot 1
    # the move of firm 5 into B at snapshot 2 removes it from the supplier side
    e2 = _edge_set(g5, 2)
    assert (5, 6) not in e2 and (0, 5) in e2 and e2[(0, 5)] == pytest.approx(0.3)


def test_roundtrip_and_check(tmp_path, monkeypatch):
    ctx, srcs, _ = _world(tmp_path, monkeypatch)
    g = bg.build_relation("M", "E1_SIC4", ctx, srcs)
    p = bg.write_graph("M", "E1_SIC4", g)
    assert p.exists() and p.parent == common.GRAPHS
    g2 = common.load_graph("M", "E1_SIC4")
    for k in ("dates", "firms", "ptr", "src", "dst", "w"):
        assert np.array_equal(g[k], g2[k]), k
    assert g2["meta"]["relation"] == "E1_SIC4" and g2["meta"]["parameters"]["CAP"] == 20
    assert g2["meta"]["edges_per_year"] == {"2010": int(len(g["src"]))}
    for i in range(3):
        s, d, w = common.snapshot_edges(g2, i)
        assert len(s) == len(d) == len(w) == int(g2["ptr"][i + 1] - g2["ptr"][i])
    assert (g2["firms"] == ctx.firms).all() and (g2["dates"] == ctx.dates).all()
    # the real-output checker passes on the synthetic graphs for all interval relations and for CORR_KNN
    for rel in bg.INTERVAL_RELS:
        bg.write_graph("M", rel, bg.build_relation("M", rel, ctx, srcs))
        assert bg.check_graph("M", rel, ctx, srcs, n_sample=50)["interval_rule"].startswith("PASS")
    kw = dict(k=3, window=120, min_overlap=60)
    gc = bg.build_relation("M", "CORR_KNN", ctx, srcs, **kw); bg.write_graph("M", "CORR_KNN", gc)
    r = bg.check_graph("M", "CORR_KNN", ctx, srcs)
    assert r["no_future"].startswith("PASS")


def test_check_catches_a_violation(tmp_path, monkeypatch):
    ctx, srcs, _ = _world(tmp_path, monkeypatch)
    g = bg.build_relation("M", "E4_CUSTSUPP", ctx, srcs)
    # inject an edge for a pair that is not in the table
    ptr = g["ptr"].copy(); ptr[3] += 1
    g["src"] = np.append(g["src"], np.int32(9)); g["dst"] = np.append(g["dst"], np.int32(10)); g["w"] = np.append(g["w"], np.float32(1.0)); g["ptr"] = ptr
    with pytest.raises(AssertionError):
        bg.check_graph("M", "E4_CUSTSUPP", ctx, srcs, g=g, n_sample=len(g["src"]))


# ---- A6: persistent rewiring nulls ----------------------------------------------------------------------------------------------
def _e1_two_sic2():
    """12 ciks: SIC2 '11' has SIC4 1111 (ciks 0-3) and 1112 (4-7); SIC2 '22' has SIC4 2222 (8-11). cik 3 changes 1111 -> 1112 in 2010-02."""
    rows = []
    for c in range(12):
        s4 = "1111" if c < 4 else "1112" if c < 8 else "2222"
        if c == 3:
            rows += [(1000 + c, "1111", pd.Timestamp("1990-01-01", tz="UTC"), pd.Timestamp("2010-02-10", tz="UTC")),
                     (1000 + c, "1112", pd.Timestamp("2010-02-10", tz="UTC"), pd.NaT)]
        else:
            rows.append((1000 + c, s4, pd.Timestamp("1990-01-01", tz="UTC"), pd.NaT))
    rows += [(1012, "1111", pd.Timestamp("1990-01-01", tz="UTC"), pd.Timestamp("2010-02-10", tz="UTC")),      # unique SIC2 history ('11', '22')
             (1012, "2222", pd.Timestamp("2010-02-10", tz="UTC"), pd.NaT)]
    return pd.DataFrame(rows, columns=["cik", "sic4", "valid_from", "valid_to"])


def _labels_at(e1, cik, d):
    m = (e1["cik"] == cik) & (e1["valid_from"] <= pd.Timestamp(d, tz="UTC")) & (e1["valid_to"].isna() | (e1["valid_to"] > pd.Timestamp(d, tz="UTC")))
    return list(e1.loc[m, "sic4"])


def test_rewire_e1_persistent_and_group_sizes_preserved():
    e1 = _e1_two_sic2(); ciks = [1000 + c for c in range(13)]
    moved = False
    for seed in range(6):
        rw, info = bg.rewire_e1(e1, ciks, bg._rng(seed, "E1_SIC4"))
        assert len(rw) == len(e1) and (rw["valid_from"].to_numpy() == e1["valid_from"].to_numpy()).all()   # intervals untouched
        assert info["n_unique_timeline"] == 1 and info["share_label_kept"] > 0
        assert rw[rw["cik"] == 1012]["sic4"].tolist() == ["1111", "2222"]                                  # unique SIC2 history -> label kept
        for d in D:                                                                                  # label multiset per date and SIC2 is preserved
            for g2 in ("11", "22"):
                real = sorted(x for c in ciks for x in _labels_at(e1, c, d) if x.startswith(g2))
                new = sorted(x for c in ciks for x in _labels_at(rw, c, d) if x.startswith(g2))
                assert real == new
        # persistence: every firm's rewired label path over the three snapshots is exactly the real path of one donor firm (one fixed mapping)
        real_paths = {tuple(tuple(_labels_at(e1, c, d)) for d in D) for c in ciks}
        for c in ciks:
            assert tuple(tuple(_labels_at(rw, c, d)) for d in D) in real_paths
        assert sorted(tuple(tuple(_labels_at(rw, c, d)) for d in D) for c in ciks) == sorted(tuple(tuple(_labels_at(e1, c, d)) for d in D) for c in ciks)
        # labels never cross SIC2
        assert all(_labels_at(rw, 1000 + c, D[0])[0].startswith("22") for c in range(8, 12))
        moved |= any(_labels_at(rw, c, D[0]) != _labels_at(e1, c, D[0]) for c in ciks)
    assert moved


def test_rewired_e1_graph_is_still_cliques_with_same_sizes(tmp_path, monkeypatch):
    ctx, srcs, _ = _world(tmp_path, monkeypatch)
    srcs2 = bg.replace(srcs, e1=_e1_two_sic2(), _cache={})
    real = bg.build_relation("M", "E1_SIC4", ctx, srcs2)
    srcs_rw = bg.rewired_sources(srcs2, "E1_SIC4", 1, set(ctx.cik.tolist()))
    rw = bg.build_relation("M", "E1_SIC4", ctx, srcs_rw)
    for i in (0, 1):
        for g in (real, rw):
            e = _edge_set(g, i); deg = {}
            for a, b in e:
                deg[a] = deg.get(a, 0) + 1
            comps = {}                                           # connected groups must be cliques
            for a, b in e:
                comps.setdefault(a, set()).add(b)
            for a, nb in comps.items():
                assert all(comps[b] - {a} == nb - {b} for b in nb)
    # the null differs from the real graph on at least one seed
    assert any(bg.swap_share(real, bg.build_relation("M", "E1_SIC4", ctx, bg.rewired_sources(srcs2, "E1_SIC4", sd, set(ctx.cik.tolist()))), NF) > 0 for sd in range(4))


def test_rewire_e4_keeps_intervals_and_swaps():
    rows = []
    for k in range(12):                                          # 12 pairs, 4 suppliers x 3 customer-SIC4 strata
        rows.append((2000 + k, 3000 + k // 3, "5331" if k % 2 == 0 else "5734", pd.Timestamp("2011-01-01") + pd.Timedelta(days=k),
                     pd.Timestamp("2012-01-01") + pd.Timedelta(days=k)))
    e4 = pd.DataFrame(rows, columns=["customer_cik", "supplier_cik", "customer_sic4", "valid_from", "valid_to"])
    e4 = pd.concat([e4, e4.assign(valid_from=e4["valid_from"] + pd.Timedelta(days=366), valid_to=e4["valid_to"] + pd.Timedelta(days=366))], ignore_index=True)  # 2nd year rows
    best = 0
    for seed in range(5):
        rw, info = bg.rewire_e4(e4, bg._rng(seed, "E4_CUSTSUPP"))
        assert (rw["valid_from"] == e4["valid_from"]).all() and (rw["valid_to"] == e4["valid_to"]).all()
        assert (rw["supplier_cik"] == e4["supplier_cik"]).all()
        assert sorted(rw["customer_cik"]) == sorted(e4["customer_cik"])                     # customer out-degree preserved
        by = e4.assign(new=rw["customer_cik"]).groupby(["customer_cik", "supplier_cik"])["new"].nunique()
        assert (by == 1).all()                                                               # the swap is persistent across a pair's rows
        # swapped customers stay in their customer-SIC4 stratum
        sic_of = dict(zip(e4["customer_cik"], e4["customer_sic4"]))
        assert all(sic_of[o] == sic_of[n] for o, n in zip(e4["customer_cik"], rw["customer_cik"]))
        assert not (rw["customer_cik"].to_numpy() == rw["supplier_cik"].to_numpy()).any()
        best = max(best, float((rw["customer_cik"] != e4["customer_cik"]).mean()))
    assert best > 0


def test_rewire_e5_permutes_within_columns():
    e5 = pd.DataFrame({"data_year": [2008] * 9, "sell_code": list("ABCABCABC"), "buy_code": list("AAABBBCCC"),
                       "weight": [0, .1, .2, .3, 0, .4, .5, .6, 0.0], "valid_from": [pd.Timestamp("2010-01-01", tz="UTC")] * 9,
                       "valid_to": [pd.Timestamp("2011-01-01", tz="UTC")] * 9})
    rw, _ = bg.rewire_e5(e5, 0)
    for col in "ABC":                                           # each buyer keeps its share multiset, zero diagonal stays zero
        a = sorted(e5.loc[(e5.buy_code == col) & (e5.sell_code != col), "weight"]); b = sorted(rw.loc[rw.buy_code == col, "weight"])
        assert [x for x in a if x > 0] == pytest.approx(b)
    assert not ((rw.sell_code == rw.buy_code)).any()


def test_rewired_corr_indegree_equals_real_refill_and_persistence(tmp_path, monkeypatch):
    ctx, srcs, _ = _world(tmp_path, monkeypatch)
    kw = dict(k=3, window=120, min_overlap=60)
    real = bg.build_relation("M", "CORR_KNN", ctx, srcs, **kw)
    bg.write_graph("M", "CORR_KNN", real)
    rw = bg.rewire_corr("M", ctx, srcs, real, 0, k=3)
    beta0 = ctx.beta(0); cuts = np.quantile(beta0, [1 / 3, 2 / 3])

    def stratum(f):                                              # independent of corr_strata: firm 5 is still in SIC2 11 at the first snapshot
        b = 0.5 + 0.1 * int(ctx.firms[f]); return ("11" if int(ctx.firms[f]) < 6 else "22", int(b > cuts[0]) + int(b > cuts[1]))
    nbrs = {}
    for i in range(3):
        s, d, w = common.snapshot_edges(rw, i); s_r, d_r, w_r = common.snapshot_edges(real, i)
        pres = set(ctx.present(i).tolist())
        assert len(s) > 0 and all(a in pres and b in pres and a != b for a, b in zip(s, d))
        assert np.array_equal(np.bincount(d, minlength=NF), np.bincount(d_r, minlength=NF))      # in-degree == real in-degree for every firm
        for f in set(d.tolist()):
            n_f = int((d == f).sum()); same = [a for a in s[d == f].tolist() if stratum(a) == stratum(f)]
            pool = [g for g in pres if g != f and stratum(g) == stratum(f)]
            assert len(same) == min(n_f, len(pool))                                               # coarser levels only when the stratum is exhausted
            assert list(w[d == f]) == list(w_r[d_r == f])                                         # real sorted weights reassigned in order
            nbrs.setdefault(i, {})[f] = s[d == f].tolist()
    for f in nbrs[1]:                                            # persistence: snapshots 1 and 2 have the same presence -> the shorter list is a prefix of the longer
        if f in nbrs[2]:
            a, b = nbrs[1][f], nbrs[2][f]; n = min(len(a), len(b)); assert a[:n] == b[:n]
    assert 0.0 <= bg.swap_share(real, rw, NF) <= 1.0
    r = bg.check_rewired("M", "CORR_KNN", 0, ctx, srcs, g=rw)
    assert r["in_degree_equals_real"].startswith("PASS")


def test_corr_knn_asymmetric_pair_edge_direction(tmp_path, monkeypatch):
    """A's top-1 neighbour is B, B's top-1 is C: the edge B -> A (src = B, dst = A) exists and A -> B does not."""
    ctx, srcs, _ = _world(tmp_path, monkeypatch)
    rng = np.random.default_rng(5); n = len(srcs.sessions)
    R = rng.normal(0, 0.01, (n, NF)).astype(np.float32)
    f1, f2 = rng.normal(0, 0.02, n), rng.normal(0, 0.02, n)
    R[:, 2] = f1 + 0.1 * R[:, 2]; R[:, 1] = 0.8 * f1 + 0.6 * f2; R[:, 0] = f2 + 0.1 * R[:, 0]       # C = 2, B = 1, A = 0
    srcs2 = bg.replace(srcs, R=R, _cache={})
    s, d, w, _ = bg.corr_snapshot(ctx, srcs2, 2, k=1, window=120, min_overlap=60)
    e = set(zip(s.tolist(), d.tolist()))
    assert (1, 0) in e and (0, 1) not in e                       # B -> A yes, A -> B no
    assert (2, 1) in e                                           # C -> B
    assert np.bincount(d, minlength=NF).max() <= 1               # in-degree <= k


def test_e1_rewire_never_swaps_firms_with_same_sic2_set_in_different_order(tmp_path, monkeypatch):
    UTC = lambda x: pd.Timestamp(x, tz="UTC")
    rows = []
    for c in (1000, 1001):                                       # X firms: SIC2 28 -> 35 on 2010-02-10
        rows += [(c, "2811" if c == 1000 else "2812", UTC("1990-01-01"), UTC("2010-02-10")), (c, "3511", UTC("2010-02-10"), pd.NaT)]
    for c in (1002, 1003):                                       # Y firms: SIC2 35 -> 28 on 2010-02-10 (same SIC2 set, other order)
        rows += [(c, "3521" if c == 1002 else "3522", UTC("1990-01-01"), UTC("2010-02-10")), (c, "2821", UTC("2010-02-10"), pd.NaT)]
    e1 = pd.DataFrame(rows, columns=["cik", "sic4", "valid_from", "valid_to"]); ciks = [1000, 1001, 1002, 1003]
    keys = bg.sic2_timeline_keys(e1, ciks)
    assert keys[1000] == keys[1001] and keys[1002] == keys[1003] and keys[1000] != keys[1002]
    for seed in range(8):
        rw, info = bg.rewire_e1(e1, ciks, bg._rng(seed, "E1_SIC4"))
        assert info["n_timeline_groups"] == 2
        for c in ciks:
            for d in (D[0], D[2]):
                assert _labels_at(rw, c, d)[0][:2] == _labels_at(e1, c, d)[0][:2], (seed, c, d)       # SIC2 of every firm-date is its real SIC2
    # and the on-table crossing counter agrees (0 crossings; a deliberate cross-swap is counted)
    ctx, srcs, _ = _world(tmp_path, monkeypatch)
    real_s = bg.replace(srcs, e1=_e1_two_sic2(), _cache={}); rw_s = bg.rewired_sources(real_s, "E1_SIC4", 3, set(ctx.cik.tolist()))
    assert bg.e1_sic2_crossings(ctx, real_s, rw_s)[0] == 0
    bad = real_s.e1.copy(); bad["cik"] = bad["cik"].map({1000: 1008, 1008: 1000}).fillna(bad["cik"]).astype(np.int64)    # SIC2 11 <-> 22 swap
    assert bg.e1_sic2_crossings(ctx, real_s, bg.replace(real_s, e1=bad, _cache={}))[0] > 0
