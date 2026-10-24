"""Tests for eqxs.equity_events (study 11, EAP / PEAD)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eqxs import equity_events as ee

SESS = pd.bdate_range("2010-01-04", periods=40)  # Mon 2010-01-04 ..


def test_day0_from_acceptance_rule_legacy_next_session() -> None:
    """Legacy rule (studies 11/12 as run): INTRADAY -> next session."""
    acc = pd.Series([
        "2010-01-05 08:00:00-05:00",   # pre-open on a session -> same day
        "2010-01-05 09:30:00-05:00",   # at the open -> next session
        "2010-01-05 16:05:00-05:00",   # after close -> next session
        "2010-01-09 10:00:00-05:00",   # Saturday -> Monday
        "2010-01-08 21:30:00-05:00",   # Friday after close (Sat 02:30 UTC) -> Monday
    ])
    d0 = ee.day0_from_acceptance(acc, SESS, intraday="next_session")
    assert list(d0) == [pd.Timestamp("2010-01-05"), pd.Timestamp("2010-01-06"),
                        pd.Timestamp("2010-01-06"), pd.Timestamp("2010-01-11"),
                        pd.Timestamp("2010-01-11")]


def test_nearest_session_and_expected() -> None:
    # Saturday 2010-01-09 is equidistant between Fri 8 and Mon 11 -> earlier (Fri)
    rows = ee.nearest_session_row(pd.Series([pd.Timestamp("2010-01-09"), pd.Timestamp("2010-01-10")]), SESS)
    assert SESS[rows[0]] == pd.Timestamp("2010-01-08") and SESS[rows[1]] == pd.Timestamp("2010-01-11")
    sess = pd.bdate_range("2010-01-01", "2011-03-01")
    r = ee.expected_day0_rows(pd.Series([pd.Timestamp("2010-01-05")]), sess)
    assert sess[r[0]] == pd.Timestamp("2011-01-04")  # +364 days, same weekday
    assert ee.session_after(pd.Series([pd.Timestamp("2010-01-08")]), SESS)[0] == 5


def test_trailing_deciles_uses_only_strictly_earlier_days() -> None:
    dates = pd.Series(list(pd.date_range("2010-01-01", periods=100)) + [pd.Timestamp("2010-04-11")] * 2)
    vals = np.r_[np.arange(100, dtype=float), 1e9, -1e9]
    b = ee.trailing_deciles(vals, dates, window_days=365, min_obs=50)
    assert (b[:50] == 0).all()                  # not enough history
    assert b[100] == 10 and b[101] == 1          # extremes vs trailing 0..99
    # same-day extremes must not move each other's breakpoints
    b2 = ee.trailing_deciles(vals[:101], dates[:101], window_days=365, min_obs=50)
    assert b2[100] == 10


def test_interval_mask_and_turnover() -> None:
    m = ee.interval_mask(6, 2, np.array([0, 0, 1]), np.array([1, 2, 3]), np.array([2, 4, 3]))
    assert m[:, 0].tolist() == [False, True, True, True, True, False]
    assert m[:, 1].tolist() == [False, False, False, True, False, False]
    rets = np.zeros((6, 2))
    res = ee.calendar_time_spread(m, np.zeros_like(m), rets, cost_per_side=0.001)
    # long leg: enter col0 at row1 (TO 1), add col1 at row3 (TO 1: 0.5 out of col0, 0.5 into col1),
    # leave col1 at row4 (TO 1), exit col0 at row5 (TO 1)
    assert res.to_long.tolist() == pytest.approx([0, 1, 0, 1, 1, 1])
    assert res.net.sum() == pytest.approx(-0.004)


def test_drift_turnover_zero_for_buy_and_hold_single_name() -> None:
    m = np.ones((5, 1), dtype=bool)
    rets = np.full((5, 1), 0.05)
    res = ee.calendar_time_spread(m, np.zeros_like(m), rets, 0.0)
    assert res.to_long[1:].tolist() == pytest.approx([0, 0, 0, 0])


def test_block_bootstrap_and_null_p() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.002, 0.01, 2000)
    assert ee.centred_block_bootstrap_p(x, rng=rng) < 0.01
    y = rng.normal(0.0, 0.01, 2000)
    assert ee.centred_block_bootstrap_p(y - y.mean() - 0.0005, rng=rng) > 0.5
    means = ee.circular_block_bootstrap_means(np.arange(10.0), block=3, n_draws=5, rng=rng)
    assert means.shape == (5,)
    assert ee.null_p([0.1, 0.2, 0.3], 0.25) == pytest.approx(2 / 4)
    s = rng.normal(0, 1, 100000)
    assert ee.mde_from_null(s) == pytest.approx(1.645 + 0.842, abs=0.03)


def test_ols_alpha_beta_recovers() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, 5000)
    y = 0.0004 + 0.3 * x + rng.normal(0, 0.002, 5000)
    out = ee.ols_alpha_beta(y, x)
    assert out["beta"] == pytest.approx(0.3, abs=0.02)
    assert out["alpha_daily"] == pytest.approx(0.0004, abs=0.0001)


def _synthetic_pead(drift: float, seed: int, n_rows: int = 500, n_cols: int = 200):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_rows)
    rets = rng.normal(0.0, 0.02, (n_rows, n_cols))
    col, form, val = [], [], []
    for c in range(n_cols):
        r0 = int(rng.integers(0, 60))
        while r0 < n_rows - 5:
            col.append(c)
            form.append(r0)
            val.append(rng.normal())
            r0 += int(rng.integers(55, 70))
    col, form, val = np.array(col), np.array(form), np.array(val)
    first, last = form + 1, np.minimum(form + 40, n_rows - 1)
    if drift:
        for c, f, la, v in zip(col, first, last, val, strict=True):
            rets[f:la + 1, c] += drift * v
    return rets, dates, col, dates[form], first, last, val


def _run(rets, fdates, col, first, last, val, rng, n_null):
    res, _ = ee.decile_spread(val, fdates, col, first, last, rets, 0.0, min_obs=100)
    live = slice(150, None)
    real = ee.sharpe(res.net[live])
    groups = ee.month_groups(fdates)
    null = []
    for _ in range(n_null):
        v = ee.shuffle_within_groups(val, groups, rng)
        r, _ = ee.decile_spread(v, fdates, col, first, last, rets, 0.0, min_obs=100)
        null.append(ee.sharpe(r.net[live]))
    return real, ee.null_p(null, real)


def test_injected_drift_is_recovered_and_shuffle_null_rejects() -> None:
    rets, dates, col, fdates, first, last, val = _synthetic_pead(drift=0.002, seed=3)
    real, p = _run(rets, fdates, col, first, last, val, np.random.default_rng(4), n_null=60)
    assert real > 2.0
    assert p < 0.05


def test_no_drift_shuffle_null_p_roughly_uniform() -> None:
    ps = []
    for seed in range(12):
        rets, dates, col, fdates, first, last, val = _synthetic_pead(drift=0.0, seed=100 + seed)
        _, p = _run(rets, fdates, col, first, last, val, np.random.default_rng(seed), n_null=40)
        ps.append(p)
    ps = np.array(ps)
    assert 0.25 < ps.mean() < 0.75
    assert (ps < 0.05).sum() <= 2


def test_shuffle_within_groups_preserves_group_multisets() -> None:
    rng = np.random.default_rng(0)
    v = np.arange(20)
    g = np.repeat([1, 2, 3, 4], 5)
    s = ee.shuffle_within_groups(v, g, rng)
    for k in (1, 2, 3, 4):
        assert sorted(s[g == k]) == sorted(v[g == k])
    assert not np.array_equal(s, v)


# ---------------------------------------------------------------------------
# Study 12: announcement delay
# ---------------------------------------------------------------------------
def test_known_session_rule() -> None:
    acc = pd.Series([
        "2010-01-05 08:00:00-05:00",   # pre-open -> known at the close of the same day (= day 0)
        "2010-01-05 11:00:00-05:00",   # intraday -> known at the close of the same day (day 0 - 1)
        "2010-01-05 16:05:00-05:00",   # after close -> next session (= day 0)
        "2010-01-09 10:00:00-05:00",   # Saturday -> Monday
    ])
    kn = ee.known_session_from_acceptance(acc, SESS)
    d0 = ee.day0_from_acceptance(acc, SESS, intraday="next_session")
    assert list(kn) == [pd.Timestamp("2010-01-05"), pd.Timestamp("2010-01-05"),
                        pd.Timestamp("2010-01-06"), pd.Timestamp("2010-01-11")]
    assert d0.iloc[1] == pd.Timestamp("2010-01-06")
    # the default (F2) day 0 is the known session
    assert ee.day0_from_acceptance(acc, SESS).tolist() == kn.tolist()


def test_match_and_delay_labels_hand_case() -> None:
    # firm 7 announces at rows 40, 100, 200 (known 40, 99 intraday, 200); firm 9 never again
    m = ee.match_announcements(np.array([7, 7, 7, 9]), np.array([100, 92, 120, 100]),
                               np.array([7, 7, 7]), np.array([40, 100, 200]), np.array([40, 99, 200]),
                               lookback=30)
    # E=100: first day0 >= 70 is 100; E=92 -> 100; E=120: first day0 >= 90 is 100; firm 9 none
    assert m.day0.tolist() == [100, 100, 100, -1]
    assert m.known.tolist() == [99, 99, 99, -1]
    assert m.next_day0.tolist() == [200, 200, 200, -1]
    t = ee.delay_labels(np.array([100, 92, 120, 100]), m, n_rows=400, early_exit="next_event_lookahead")
    # E=100: known 99 <= 105 and day0 100 > 95 -> on time
    # E=92: known 99 <= 97? no -> late, entry 97, last min(137, 101) = 101
    # E=120: day0 100 <= 115 -> early, entry 101, last min(141, 199) = 141
    # firm 9: no match -> late, entry 105, last 145
    assert t.label.tolist() == [ee.ONTIME, ee.LATE, ee.EARLY, ee.LATE]
    assert t.entry_row.tolist() == [-1, 97, 101, 105]
    assert t.last_row.tolist() == [-1, 101, 141, 145]
    # early exit truncated by the next announcement
    m2 = ee.AnnouncementMatch(np.array([50]), np.array([50]), np.array([70]))
    t2 = ee.delay_labels(np.array([60]), m2, n_rows=400, early_exit="next_event_lookahead")
    assert t2.label[0] == ee.EARLY and t2.entry_row[0] == 51 and t2.last_row[0] == 69


def test_shuffle_delay_offsets_identity_and_lengths() -> None:
    rng = np.random.default_rng(0)
    e = np.array([100, 110, 120, 300, 310])
    t = ee.DelayTrades(np.array([ee.LATE, ee.ONTIME, ee.EARLY, ee.LATE, ee.ONTIME]),
                       np.array([105, -1, 112, 305, -1]), np.array([120, -1, 150, 345, -1]))
    g = np.array([1, 1, 1, 2, 2])
    lengths = []
    for _ in range(20):
        s = ee.shuffle_delay_offsets(t, e, g, rng, n_rows=1000, stratify="none")
        tr = s.entry_row >= 0
        assert tr[:3].sum() == 2 and tr[3:].sum() == 1
        lengths.append(sorted((s.last_row - s.entry_row)[tr][:2].tolist()))
        # offsets are applied at each record's own expected row
        for i in np.flatnonzero(tr):
            assert (s.entry_row[i] - e[i]) in {5, -8}
    assert all(lg == [15, 38] for lg in lengths)
    same = ee.shuffle_delay_offsets(t, e, np.arange(5), rng, n_rows=1000, stratify="none")  # singletons
    assert same.entry_row.tolist() == t.entry_row.tolist()


def test_trailing_beta_and_hedged_short_is_beta_neutral() -> None:
    rng = np.random.default_rng(2)
    n, k = 1500, 60
    mkt = rng.normal(0.0004, 0.01, n)
    rets = 1.4 * mkt[:, None] + rng.normal(0, 0.01, (n, k))
    rets_all = np.column_stack([rets, mkt[:, None] + rng.normal(0, 1e-4, (n, 200))])
    leg = np.zeros((n, k + 200), dtype=bool)
    leg[:, :k] = True
    mk = np.zeros_like(leg)
    mk[:, k:] = True
    b = ee.trailing_beta(rets.mean(axis=1), mkt, np.ones(n, dtype=bool))
    assert np.isnan(b[:126]).all()
    assert b[300:] == pytest.approx(1.4, abs=0.05)
    res = ee.beta_hedged_short(leg, mk, rets_all, 0.0)
    live = res.active
    assert not live[:126].any() and live[200:].all()
    raw = ee.ols_alpha_beta(res.raw_gross[live], mkt[live])
    hed = ee.ols_alpha_beta(res.gross[live], mkt[live])
    assert raw["beta"] == pytest.approx(-1.4, abs=0.05)
    assert abs(hed["beta"]) < 0.05
    resc = ee.beta_hedged_short(leg, mk, rets_all, 0.001)
    assert (resc.net[live] <= resc.gross[live] + 1e-15).all()


def test_both_or_none() -> None:
    lm = np.array([[True, False], [False, False], [True, True]])
    sm = np.array([[False, True], [False, True], [False, False]])
    a, b = ee.both_or_none(lm, sm)
    assert a.tolist() == [[True, False], [False, False], [False, False]]
    assert b.tolist() == [[False, True], [False, False], [False, False]]


def _synthetic_delay(effect: float, seed: int, n_rows: int = 1300, n_cols: int = 160):
    """Quarterly announcers (63-row spacing); 10 % of announcements late by 9-20 rows, 10 %
    early by 7-15 rows. ``effect`` = daily drift: late firms lose it from E+6 to day +1,
    early firms gain it over day +2 .. +41."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.02, (n_rows, n_cols))
    keys, day0 = [], []
    base = {}
    for c in range(n_cols):
        ph = int(rng.integers(0, 63))
        q = 0
        while ph + 63 * q < n_rows - 10:
            b = ph + 63 * q
            u = rng.random()
            dev = int(rng.integers(9, 21)) if u < 0.1 else (-int(rng.integers(7, 16)) if u < 0.2 else 0)
            keys.append(c)
            day0.append(b + dev)
            base[(c, q)] = (b, dev)
            q += 1
    keys, day0 = np.array(keys), np.array(day0)
    ok = (day0 >= 0) & (day0 < n_rows)
    keys, day0 = keys[ok], day0[ok]
    # expected row of quarter q = (actual day 0 of quarter q-4) + 252
    rec_c, rec_e = [], []
    for (c, q), _ in base.items():
        if (c, q - 4) in base:
            b4, d4 = base[(c, q - 4)]
            rec_c.append(c)
            rec_e.append(b4 + d4 + 252)
    rec_c, rec_e = np.array(rec_c), np.array(rec_e)
    keep = (rec_e > 40) & (rec_e < n_rows - 6)
    rec_c, rec_e = rec_c[keep], rec_e[keep]
    m = ee.match_announcements(rec_c, rec_e, keys, day0, day0)
    t = ee.delay_labels(rec_e, m, n_rows, early_exit="next_event_lookahead")
    if effect:
        for c, lab, a, b in zip(rec_c, t.label, t.entry_row, t.last_row, strict=True):
            if a >= 0:
                rets[a + 1:b + 1, c] += -effect if lab == ee.LATE else effect
    return rets, rec_c, rec_e, t


def _delay_run(rets, rec_c, rec_e, t, rng, n_null):
    n_rows, n_cols = rets.shape
    uni = np.ones((n_rows, n_cols), dtype=bool)
    groups = rec_e // 21

    def book(tr):
        lm, sm = ee.delay_leg_masks(tr, rec_c, uni)
        lm, sm = ee.both_or_none(lm, sm)
        return ee.calendar_time_spread(lm, sm, rets, 0.0).net[300:]

    real = ee.sharpe(book(t))
    null = [ee.sharpe(book(ee.shuffle_delay_offsets(t, rec_e, groups, rng, n_rows, stratify="none")))
            for _ in range(n_null)]
    return real, ee.null_p(null, real)


def test_injected_late_underperformance_recovered_and_shuffle_rejects() -> None:
    rets, rec_c, rec_e, t = _synthetic_delay(effect=0.004, seed=11)
    # a deviated year-ago announcement shifts E too, so each share is ~10-20 %
    assert 0.08 < (t.label == ee.LATE).mean() < 0.3
    assert 0.08 < (t.label == ee.EARLY).mean() < 0.3
    real, p = _delay_run(rets, rec_c, rec_e, t, np.random.default_rng(12), n_null=60)
    assert real > 1.5
    assert p < 0.05


def test_no_delay_effect_shuffle_p_roughly_uniform() -> None:
    ps = []
    for seed in range(12):
        rets, rec_c, rec_e, t = _synthetic_delay(effect=0.0, seed=200 + seed)
        _, p = _delay_run(rets, rec_c, rec_e, t, np.random.default_rng(seed), n_null=40)
        ps.append(p)
    ps = np.array(ps)
    assert 0.25 < ps.mean() < 0.75
    assert (ps < 0.05).sum() <= 2


# ---------------------------------------------------------------------------
# Red-team machinery fixes 2026-10-10
# ---------------------------------------------------------------------------
def test_day0_intraday_maps_to_acceptance_session_by_default() -> None:
    """F2: INTRADAY -> the acceptance session (the reaction day); BMO same; AMC next;
    Friday evening -> Monday; a holiday is skipped."""
    sess = pd.bdate_range("2010-01-11", "2010-01-22").drop(pd.Timestamp("2010-01-18"))  # MLK day
    acc = pd.Series([
        "2010-01-12 11:00:00-05:00",   # INTRADAY on a session -> that session
        "2010-01-12 16:05:00-05:00",   # AMC -> next session
        "2010-01-12 07:30:00-05:00",   # BMO -> same session
        "2010-01-15 18:00:00-05:00",   # Friday evening -> Monday (here Mon 18 is a holiday -> Tue 19)
        "2010-01-18 10:00:00-05:00",   # on the holiday itself -> next session
        "2010-01-14 18:00:00-05:00",   # Thursday AMC -> Friday
        "2010-01-12 09:30:00-05:00",   # exactly at the open -> INTRADAY -> same session
        "2010-01-12 15:59:00-05:00",   # last intraday minute -> same session
    ])
    d0 = ee.day0_from_acceptance(acc, sess)
    assert d0.tolist() == [pd.Timestamp(x) for x in (
        "2010-01-12", "2010-01-13", "2010-01-12", "2010-01-19", "2010-01-19", "2010-01-15",
        "2010-01-12", "2010-01-12")]
    legacy = ee.day0_from_acceptance(acc, sess, intraday="next_session")
    assert legacy.iloc[0] == pd.Timestamp("2010-01-13") and legacy.iloc[6] == pd.Timestamp("2010-01-13")
    assert legacy.iloc[[1, 2, 3, 4, 5]].tolist() == d0.iloc[[1, 2, 3, 4, 5]].tolist()  # only INTRADAY differs
    # Friday evening with a normal Monday
    fri = ee.day0_from_acceptance(pd.Series(["2010-01-08 19:00:00-05:00"]), SESS)
    assert fri.iloc[0] == pd.Timestamp("2010-01-11")
    with pytest.raises(ValueError):
        ee.day0_from_acceptance(acc, sess, intraday="both")


def test_next_event_exit_cap_modes() -> None:
    """F5: expected_next = nearest session to day0 + 91 calendar days, minus 2 sessions."""
    sess = pd.bdate_range("2010-01-04", periods=200)
    day0 = np.array([0, 10, -1, 190])
    nxt = np.array([50, -1, -1, 195])
    exp = ee.next_event_exit_cap(day0, nxt, sess)
    # 2010-01-04 + 91 d = 2010-04-05 (Mon): row 65 -> cap 63
    assert sess[exp[0] + 2] == pd.Timestamp("2010-04-05") and exp[0] == 63
    assert exp[1] == ee.nearest_session_row(pd.Series([sess[10] + pd.Timedelta(days=91)]), sess)[0] - 2
    assert exp[2] == ee.NO_CAP          # no day 0
    assert exp[3] == ee.NO_CAP          # expected date past the calendar end
    legacy = ee.next_event_exit_cap(day0, nxt, sess, mode="next_event_lookahead")
    assert legacy.tolist() == [49, ee.NO_CAP, ee.NO_CAP, 194]
    hold = ee.next_event_exit_cap(day0, nxt, mode="hold_through")
    assert (hold == ee.NO_CAP).all()
    with pytest.raises(ValueError):
        ee.next_event_exit_cap(day0, nxt, mode="expected_next")  # needs sessions
    with pytest.raises(ValueError):
        ee.next_event_exit_cap(day0, None, sess, mode="next_event_lookahead")


def test_delay_labels_early_exit_modes() -> None:
    """F5 in the study-12 early leg: an early announcement at row 50, next actual day 0 at
    row 70, hold 40 (to row 91)."""
    sess = pd.bdate_range("2010-01-04", periods=400)
    m = ee.AnnouncementMatch(np.array([50]), np.array([50]), np.array([70]))
    e = np.array([60])
    look = ee.delay_labels(e, m, 400, early_exit="next_event_lookahead")
    assert look.label[0] == ee.EARLY and look.entry_row[0] == 51 and look.last_row[0] == 69
    hold = ee.delay_labels(e, m, 400, early_exit="hold_through")
    assert hold.last_row[0] == 91
    expn = ee.delay_labels(e, m, 400, sessions=sess)   # default: expected_next
    cap = ee.nearest_session_row(pd.Series([sess[50] + pd.Timedelta(days=91)]), sess)[0] - 2
    assert expn.last_row[0] == min(91, cap) == 91        # ~63 sessions later: does not bind at hold 40
    short = ee.delay_labels(e, m, 400, sessions=sess, hold=80)
    assert short.last_row[0] == cap < 131
    with pytest.raises(ValueError):
        ee.delay_labels(e, m, 400)                       # expected_next needs sessions


def test_entry_exit_turnover_on_three_name_book() -> None:
    """F3: names A, B held; C enters at row 2; A exits at row 3. Zero returns: EW
    re-weighting of incumbents is exactly the gap between the two turnover figures."""
    m = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)
    rets = np.zeros((4, 3))
    t = ee.leg_turnover(m, rets)
    assert t.entries.tolist() == pytest.approx([1.0, 0.0, 1 / 3, 0.0])
    assert t.exits.tolist() == pytest.approx([0.0, 0.0, 0.0, 1 / 3])
    assert t.incumbents.tolist() == pytest.approx([0.0, 0.0, 1 / 3, 1 / 3])  # A,B 1/2->1/3; B,C 1/3->1/2
    res = ee.calendar_time_spread(m, np.zeros_like(m), rets, cost_per_side=0.001)
    assert res.to_long.tolist() == pytest.approx([1.0, 0.0, 2 / 3, 2 / 3])
    assert res.to_long_entry_exit.tolist() == pytest.approx([1.0, 0.0, 1 / 3, 1 / 3])
    assert (res.net_entry_exit - res.net).tolist() == pytest.approx((0.001 * t.incumbents).tolist())
    # with returns the identity total = entries + exits + incumbents still holds exactly
    rng = np.random.default_rng(5)
    r2 = rng.normal(0, 0.03, (4, 3))
    t2 = ee.leg_turnover(m, r2)
    res2 = ee.calendar_time_spread(m, np.zeros_like(m), r2, 0.0)
    assert t2.total.tolist() == pytest.approx(res2.to_long.tolist())
    # the exiting name is charged its DRIFTED weight
    w_a = 1 / 3 * (1 + r2[2, 0]) / (1 + r2[2].mean())
    assert t2.exits[3] == pytest.approx(w_a)


def test_hedged_short_entry_exit_cost_not_above_rebalance_cost() -> None:
    rng = np.random.default_rng(3)
    n, k = 600, 40
    rets = rng.normal(0, 0.01, (n, k))
    leg = rng.random((n, k)) < 0.2
    mk = np.zeros_like(leg)
    mk[:, 20:] = True
    res = ee.beta_hedged_short(leg, mk, rets, 0.001)
    assert (res.to_leg_entry_exit <= res.to_leg + 1e-12).all()
    assert (res.net_entry_exit >= res.net - 1e-15).all()
    assert res.net_entry_exit[res.active].mean() > res.net[res.active].mean()


def _yearly(rets: list[float], sessions: int = 252) -> pd.DataFrame:
    return pd.DataFrame({"ret_compounded": rets, "sessions": [sessions] * len(rets)},
                        index=range(2004, 2004 + len(rets)))


def test_years_gate_rules() -> None:
    """F4: legacy share >= 75 % vs default mean-ex-best-year > 0."""
    # 8/14 positive, small losses: legacy FAIL, ex-best-year mean > 0 -> PASS
    a = _yearly([0.05] * 8 + [-0.01] * 6)
    leg, new = ee.years_gate(a, rule="share_ge_075"), ee.years_gate(a)
    assert not leg["pass"] and leg["fail_label"] == "pos_years<75%"
    assert new["pass"] and new["rule"] == "ex_best_year_positive"
    assert new["pos_year_share"] == pytest.approx(8 / 14)          # diagnostic under both
    # 11/14 positive but three large losses: legacy PASS, new FAIL
    b = _yearly([0.01] * 11 + [-0.10] * 3)
    assert ee.years_gate(b, rule="share_ge_075")["pass"]
    nb = ee.years_gate(b)
    assert not nb["pass"] and nb["fail_label"] == "ex_best_year_mean<=0"
    # one year carries the result: mean > 0 overall, ex-best < 0 -> FAIL
    c = _yearly([0.60] + [-0.02] * 7)
    assert np.mean(c["ret_compounded"]) > 0 and not ee.years_gate(c)["pass"]
    # short years (< 63 sessions) are not counted
    d = pd.concat([_yearly([0.05, 0.05]), _yearly([-0.50], sessions=20)])
    assert ee.years_gate(d)["years_counted"] == 2 and ee.years_gate(d)["pass"]
    assert not ee.years_gate(_yearly([0.05]))["pass"]                # one year: undecidable
    with pytest.raises(ValueError):
        ee.years_gate(a, rule="majority")


def test_event_study_verdict_legacy_reproduces_as_run_fails() -> None:
    recs = [{"year": 2004 + i, "ret_compounded": r, "ann_mean": r, "sharpe": 0.0, "sessions": 252.0}
            for i, r in enumerate([0.05] * 8 + [-0.01] * 6)]
    v = ee.event_study_verdict(0.4, 0.02, 0.048, recs, years_rule="share_ge_075")
    assert v["verdict"] == "REJECT" and v["fails"] == ["pos_years<75%"]
    v2 = ee.event_study_verdict(0.4, 0.02, 0.048, recs)
    assert v2["verdict"] == "PASS" and v2["fails"] == []
    v3 = ee.event_study_verdict(-0.1, 0.3, 0.9, recs, years_rule="share_ge_075")
    assert v3["fails"] == ["net_sharpe<=0", "N1_p>=0.05", "N2_p>=0.05", "pos_years<75%"]
    assert ee.event_study_verdict(float("nan"), 0.01, 0.01, recs)["fails"] == ["net_sharpe<=0"]


def test_stratified_delay_null_keeps_real_leg_thickness() -> None:
    """F6: late labels go mostly to firms outside the universe. The month-only shuffle hands
    late offsets to eligible firms and thickens the null late leg; stratifying by
    eligibility at E keeps the real counts exactly."""
    rng = np.random.default_rng(7)
    n_rows, n_cols = 1500, 300
    elig_col = rng.random(n_cols) < 0.5
    universe = np.repeat(elig_col[None, :], n_rows, axis=0)   # constant over time
    col = np.repeat(np.arange(n_cols), 4)
    e = (rng.integers(100, 1300, size=len(col))).astype(np.int64)
    p_late = np.where(elig_col[col], 0.05, 0.40)
    late = rng.random(len(col)) < p_late
    early = ~late & (rng.random(len(col)) < 0.2)
    label = np.where(late, ee.LATE, np.where(early, ee.EARLY, ee.ONTIME))
    entry = np.where(late, e + 5, np.where(early, e - 3, -1))
    last = np.where(late, e + 25, np.where(early, e + 37, -1))
    trades = ee.DelayTrades(label.astype(np.int64), entry.astype(np.int64), last.astype(np.int64))
    groups = e // 21
    elig = ee.eligibility_at(e, col, universe)
    assert elig.tolist() == elig_col[col].tolist()

    def leg_counts(t):
        ok = (t.entry_row >= 0) & ee.eligibility_at(np.maximum(t.entry_row, 0), col, universe)
        return int((ok & (t.label == ee.LATE)).sum()), int((ok & (t.label == ee.EARLY)).sum())

    def late_name_days(t):
        return int(ee.delay_leg_masks(t, col, universe)[1].sum())

    real = leg_counts(trades)
    real_nd = late_name_days(trades)
    strat, plain = [], []
    for _ in range(20):
        strat.append(leg_counts(ee.shuffle_delay_offsets(trades, e, groups, rng, n_rows, eligible=elig)))
        plain.append(leg_counts(ee.shuffle_delay_offsets(trades, e, groups, rng, n_rows, stratify="none")))
    assert all(c == real for c in strat)
    assert np.mean([c[0] for c in plain]) > 1.5 * real[0]
    t_s = ee.shuffle_delay_offsets(trades, e, groups, rng, n_rows, eligible=elig)
    assert abs(late_name_days(t_s) - real_nd) <= 0.02 * real_nd      # only same-firm overlaps differ
    with pytest.raises(ValueError):
        ee.shuffle_delay_offsets(trades, e, groups, rng, n_rows)     # default needs eligible
