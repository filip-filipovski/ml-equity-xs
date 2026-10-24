"""Unit tests for the free US equity cross-section (``eqxs.equities_xs``).

Synthetic inputs only; no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eqxs.data_zones import SealedWindowLeakError
from eqxs import equities_xs as xs

NASDAQ_TXT = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
    "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
    "QQQ|Invesco QQQ Trust, Series 1|G|N|N|100|Y|N\n"
    "ZXZZT|NASDAQ TEST STOCK|G|Y|N|100|N|N\n"
    "ABCDW|Abcd Acquisition Corp - Warrants|S|N|N|100|N|N\n"
    "PDD|PDD Holdings Inc. - American Depositary Shares|Q|N|N|100|N|N\n"
    "File Creation Time: 1010202608:00|||||||\n"
)
OTHER_TXT = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
    "BRK.B|Berkshire Hathaway Inc. Class B|N|BRK.B|N|100|N|BRK=B\n"
    "WHR$A|Whirlpool Corp Depositary Shares, each representing a 1/20th interest in a share of"
    " 8.50% Series A Mandatory Convertible Preferred Stock|N|WHR.PRA|N|100|N|WHR-A\n"
    "SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY\n"
    "File Creation Time: 1010202608:00|||||||\n"
)


def test_universe_current_parse_and_classify() -> None:
    u = xs.build_universe_current(NASDAQ_TXT, OTHER_TXT)
    t = dict(zip(u["symbol"], u["security_type"], strict=True))
    assert t == {"AAPL": "common", "QQQ": "etf", "ZXZZT": "test", "ABCDW": "other",
                 "PDD": "adr", "BRK.B": "common", "WHR$A": "other", "SPY": "etf"}
    row = u.set_index("symbol").loc["BRK.B"]
    assert row["exchange"] == "NYSE" and row["yf_symbol"] == "BRK-B"
    assert u.set_index("symbol").loc["AAPL", "market_category"] == "Nasdaq Global Select"
    assert (u["file_creation_time"] == "1010202608:00").all()


def _yf_frame() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    cols = pd.MultiIndex.from_product(
        [["AAA", "BBB"], ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends",
                          "Stock Splits"]])
    df = pd.DataFrame(np.ones((4, 16)), index=idx, columns=cols)
    df[("BBB", "Close")] = np.nan  # BBB returned nothing
    df[("AAA", "Close")] = [10.0, 11.0, 12.0, 13.0]
    return df


def test_normalize_yf_download_long_format() -> None:
    out = xs.normalize_yf_download(_yf_frame(), ["AAA", "BBB", "CCC"])
    assert list(out.columns) == list(xs.PRICE_COLUMNS)
    assert set(out["symbol"]) == {"AAA"} and len(out) == 4
    assert str(out["ts"].dt.tz) == "UTC"
    assert (out["source"] == "yfinance").all()


def test_dividend_adjustment_factor_reproduces_adj_close() -> None:
    d = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=4),
                      "close": [100.0, 100.0, 98.0, 99.0], "dividends": [0, 0, 2.0, 0]})
    f = xs.dividend_adjustment_factor(d)
    # rows before the 2.0 ex-date are scaled by (1 - 2/100) = 0.98, later rows by 1
    np.testing.assert_allclose(f.to_numpy(), [0.98, 0.98, 1.0, 1.0])


def test_membership_delisted_flag() -> None:
    cal = pd.bdate_range("2020-01-01", periods=200)
    rows = [("LIVE", d) for d in cal] + [("GONE", d) for d in cal[:100]] + \
           [("RECENT", d) for d in cal[:180]]
    px = pd.DataFrame(rows, columns=["symbol", "date"])
    m = xs.build_membership(px, file_date=cal[-1], current_symbols=["LIVE"], gap_days=60)
    m = m.set_index("symbol")
    assert bool(m.loc["GONE", "delisted_flag"]) is True       # 100 days stale, not listed
    assert bool(m.loc["RECENT", "delisted_flag"]) is False    # only 20 days stale
    assert bool(m.loc["LIVE", "delisted_flag"]) is False
    assert m.loc["GONE", "n_days"] == 100


FACTS = {
    "cik": 123,
    "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"start": "2015-01-01", "end": "2015-03-31", "val": 100, "accn": "A-1", "fy": 2015,
             "fp": "Q1", "form": "10-Q", "filed": "2015-05-01", "frame": "CY2015Q1"},
            {"start": "2015-01-01", "end": "2015-03-31", "val": 90, "accn": "A-2", "fy": 2015,
             "fp": "Q1", "form": "10-Q/A", "filed": "2015-08-01"},
        ]}},
        "Assets": {"units": {"USD": [
            {"end": "2015-03-31", "val": 1000, "accn": "A-1", "fy": 2015, "fp": "Q1",
             "form": "10-Q", "filed": "2015-05-01"}]}},
    }},
}
SUBS = {"cik": 123, "name": "TEST CO", "sic": "1234", "tickers": ["TST"], "exchanges": ["NYSE"],
        "filings": {"recent": {
            "accessionNumber": ["A-1", "A-2", "B-1"],
            "form": ["10-Q", "10-Q/A", "8-K"],
            "filingDate": ["2015-05-01", "2015-08-01", "2015-04-28"],
            "acceptanceDateTime": ["2015-05-01T21:00:00.000Z", "2015-08-01T12:00:00.000Z",
                                   "2015-04-28T20:30:00.000Z"],
            "reportDate": ["2015-03-31", "2015-03-31", ""],
            "items": ["", "", "2.02,9.01"],
            "isXBRL": [1, 1, 0]},
            "files": []}}


def test_parse_companyfacts_shape_and_qtrs() -> None:
    num = xs.parse_companyfacts(FACTS)
    assert list(num.columns) == list(xs.NUM_COLUMNS)
    assert len(num) == 3 and (num["cik"] == 123).all()
    q = dict(zip(num["tag"] + ":" + num["adsh"], num["qtrs"], strict=True))
    assert q["Revenues:A-1"] == 1 and q["Assets:A-1"] == 0


def test_sub_rows_and_pit_filter_by_filed() -> None:
    fil = xs.parse_submission_filings([SUBS], 123)
    num = xs.attach_num_ts(xs.parse_companyfacts(FACTS), fil)
    sub = xs.build_sub_rows(fil, xs.filer_profile(SUBS), xs.fyfp_by_adsh(num))
    assert set(sub["form"]) == {"10-Q", "10-Q/A"} and (sub["sic"] == "1234").all()
    assert sub.set_index("adsh").loc["A-1", "fp"] == "Q1"
    # ts = acceptance instant, not ddate
    assert num.set_index("adsh")["ts"].min() == pd.Timestamp("2015-05-01T21:00:00Z")
    rev = num[num["tag"] == "Revenues"]
    # before the original filing nothing is knowable (period ended 2015-03-31)
    assert xs.pit_filter(rev, "2015-04-30").empty
    # after the original, before the amendment -> 100
    snap = xs.latest_as_of(rev, "2015-06-01")
    assert snap["value"].tolist() == [100]
    # after the amendment -> 90
    assert xs.latest_as_of(rev, "2015-09-01")["value"].tolist() == [90]
    # same day as the original filing but before its 21:00 UTC acceptance -> nothing
    assert xs.pit_filter(rev, pd.Timestamp("2015-05-01T20:00:00Z")).empty


def test_fy_quarter_label() -> None:
    lab = xs.fy_quarter_label(pd.Series(pd.to_datetime(["2015-05-01", "2021-12-31"], utc=True)))
    assert lab.tolist() == ["2015q2", "2021q4"]


def test_cik_ticker_map_exact_and_fuzzy() -> None:
    prof = pd.DataFrame([
        {"cik": 1, "name": "ALPHA CORP", "tickers": "ALP", "exchanges": "NYSE"},
        {"cik": 2, "name": "Beta Widgets, Inc.", "tickers": "", "exchanges": ""},
    ])
    uni = pd.DataFrame({"symbol": ["ALP", "BWID", "NOPE"],
                        "security_name": ["Alpha Corp Common", "Beta Widgets Inc. Common Stock",
                                          "Nobody Ltd"],
                        "exchange": ["NYSE", "Nasdaq", "NYSE"], "security_type": "common"})
    m = xs.build_cik_ticker_map(prof, uni).set_index("symbol")
    assert m.loc["ALP", "cik"] == 1 and m.loc["ALP", "match"] == "sec_tickers"
    assert m.loc["BWID", "cik"] == 2 and m.loc["BWID", "match"] == "name_fuzzy"
    assert m.loc["NOPE", "match"] == "none" and pd.isna(m.loc["NOPE", "cik"])


def test_parse_french_daily_csv() -> None:
    txt = ("This file was created by CMPT_ME_BEME_RETS_DAILY\n\n"
           ",Mkt-RF,SMB,HML,RF\n19260701,    0.09,   -0.25,   -0.27,    0.01\n"
           "19260702,    0.45,   -0.33,   -0.06,    0.01\n\nCopyright 2026 Kenneth R. French\n")
    df = xs.parse_french_daily_csv(txt)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert df.index[0] == pd.Timestamp("1926-07-01")
    assert df.iloc[0]["Mkt-RF"] == pytest.approx(0.0009)


def _write_year(root, year: int, dates) -> None:
    d = root / "yf_daily" / f"year={year}"
    d.mkdir(parents=True, exist_ok=True)
    dt = pd.to_datetime(dates)
    pd.DataFrame({"symbol": "AAA", "date": dt, "ts": dt.tz_localize("UTC"),
                  "close": 1.0}).to_parquet(d / "part-0000.parquet", index=False)


SEALED_2024 = pd.Timestamp("2024-07-01", tz="UTC")  # equities zone change 2026-10-11 (eed5ffb)


def test_equities_sealed_start_is_2024_07_01() -> None:
    assert xs.EQUITY_SEALED_START == SEALED_2024


def test_loader_never_opens_sealed_partitions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(xs, "XS_ROOT", tmp_path)
    # 2023-06-01 is TRAIN/WFA-visible; the 2024 year partition straddles the sealed start
    # (2024-07-01) and is therefore never listed without a token
    _write_year(tmp_path, 2023, ["2023-06-01", "2023-12-29"])
    _write_year(tmp_path, 2024, ["2024-08-01"])
    opened: list[str] = []
    real = xs.safe_load_parquet

    def spy(path, **kw):
        opened.append(str(path))
        return real(path, **kw)

    monkeypatch.setattr(xs, "safe_load_parquet", spy)
    df = xs.load_yf_daily()
    assert len(df) == 2 and df["date"].min() == pd.Timestamp("2023-06-01")
    assert df["date"].max() < pd.Timestamp("2024-07-01")
    assert all("year=2024" not in p for p in opened)
    # without a token, asking for sealed still never lists the 2024 partition
    assert all("year=2024" not in str(p)
               for p in xs._zoned_files(tmp_path / "yf_daily", "year", unlock_token=None))


def test_loader_tripwire_raises_on_misfiled_sealed_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(xs, "XS_ROOT", tmp_path)
    # a sealed-dated row planted in a pre-sealed partition must trip the wire
    _write_year(tmp_path, 2023, ["2023-06-01", "2024-08-01"])
    with pytest.raises(SealedWindowLeakError):
        xs.load_yf_daily()


def test_sec_num_loader_partition_bounds(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(xs, "XS_ROOT", tmp_path)
    for lab, ts in (("2023q2", "2023-06-01"), ("2024q2", "2024-05-01"), ("2024q3", "2024-08-01")):
        d = tmp_path / "sec_fsds" / "num" / f"fy_quarter={lab}"
        d.mkdir(parents=True)
        pd.DataFrame({"cik": [1], "tag": ["Assets"], "value": [1.0],
                      "ts": [pd.Timestamp(ts, tz="UTC")]}).to_parquet(d / "part-0000.parquet")
    df = xs.load_sec_num()
    assert len(df) == 2 and df["ts"].max() < SEALED_2024


def test_membership_loader_masks_post_sealed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(xs, "XS_ROOT", tmp_path)
    pd.DataFrame({"symbol": ["A", "B"],
                  "first_date": pd.to_datetime(["2010-01-04", "2024-08-01"]),
                  "last_date": pd.to_datetime(["2023-06-01", "2026-10-09"]),
                  "n_days": [100, 900], "delisted_flag": [True, False],
                  "in_current_list": [False, True]}).to_parquet(
        tmp_path / "universe_membership.parquet", index=False)
    m = xs.load_membership()
    assert "delisted_flag" not in m.columns and "in_current_list" not in m.columns
    assert m["last_date"].iloc[0] == pd.Timestamp("2023-06-01")  # visible under new bounds
    assert pd.isna(m["last_date"].iloc[1]) and pd.isna(m["first_date"].iloc[1])
    with pytest.raises(Exception, match="LOCKED_CONFIG"):
        xs.load_membership(allow_sealed=True, unlock_token="nope")


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


def test_token_bucket_caps_rate() -> None:
    clk = _FakeClock()
    b = xs.TokenBucket(rate=8.0, capacity=8.0, clock=clk, sleep=clk.sleep)
    for _ in range(8):  # initial burst = capacity
        assert b.try_acquire()
    assert not b.try_acquire()
    for _ in range(80):
        b.acquire()
    # 80 more tokens at 8/s need ~10 s of (fake) time, never less
    assert clk.t == pytest.approx(10.0, abs=0.1)


def test_token_bucket_thread_safe_count() -> None:
    import threading

    b = xs.TokenBucket(rate=1e9, capacity=1000.0)
    got: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        n = sum(1 for _ in range(400) if b.try_acquire())
        with lock:
            got.append(n)

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sum(got) <= 1600  # never more tokens than were available + refilled


def test_locked_appender_whole_lines_under_threads(tmp_path) -> None:
    import json as _json
    import threading

    app = xs.LockedAppender()
    p = tmp_path / "done.jsonl"
    payload = "x" * 5000

    def worker(k: int) -> None:
        for i in range(200):
            app.append(p, _json.dumps({"cik": k * 1000 + i, "pad": payload}))

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    lines = p.read_text().splitlines()
    assert len(lines) == 1200
    assert len({_json.loads(ln)["cik"] for ln in lines}) == 1200  # every line parses whole


def test_polite_client_403_triggers_cooldown_and_retries(monkeypatch) -> None:
    class R:
        def __init__(self, code: int) -> None:
            self.status_code, self.content = code, b"ok"

    seq = iter([R(403), R(403), R(200)])

    class H:
        def get(self, url, **kw):
            return next(seq)

    cd = xs.GlobalCooldown(seconds=0.0)
    c = xs.PoliteClient(min_interval=0.0, cooldown=cd, throttle_403=True)
    c._client = H()
    assert c.get("https://data.sec.gov/x.json") == b"ok"
    assert cd.n_triggers == 2 and c.n_requests == 3
    # without the throttle flag a 403 is a hard failure
    c2 = xs.PoliteClient(min_interval=0.0)
    c2._client = type("H2", (), {"get": lambda self, u, **kw: R(403)})()
    with pytest.raises(xs.FetchError):
        c2.get("https://data.sec.gov/x.json")


def test_loader_start_end_skip_partitions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(xs, "XS_ROOT", tmp_path)
    _write_year(tmp_path, 2018, ["2018-12-28", "2018-12-31"])
    _write_year(tmp_path, 2019, ["2019-01-02"])
    _write_year(tmp_path, 2020, ["2020-01-02"])
    opened: list[str] = []
    real = xs.safe_load_parquet

    def spy(path, **kw):
        opened.append(str(path))
        return real(path, **kw)

    monkeypatch.setattr(xs, "safe_load_parquet", spy)
    df = xs.load_yf_daily(end="2019-01-01")
    assert len(df) == 2 and df["date"].max() < pd.Timestamp("2019-01-01")
    assert all("year=2019" not in p and "year=2020" not in p for p in opened)
    df2 = xs.load_yf_daily(start="2018-12-31", end="2019-01-01")
    assert list(df2["date"]) == [pd.Timestamp("2018-12-31")]


def test_load_ff_daily_start_end_never_returns_outside_rows(tmp_path, monkeypatch) -> None:
    """F13 (red team 2026-10-10): rows outside [start, end) are never returned."""
    monkeypatch.setattr(xs, "XS_ROOT", tmp_path)
    idx = pd.DatetimeIndex(["2018-12-28", "2018-12-31", "2019-01-02", "2021-12-31"]).tz_localize("UTC")
    pd.DataFrame({"Mkt-RF": [0.01, 0.02, 0.03, 0.04], "RF": 0.0}, index=idx).to_parquet(
        tmp_path / "ff_daily.parquet")
    full = xs.load_ff_daily()
    assert len(full) == 4  # default: no bound (same contract as load_yf_daily)
    tr = xs.load_ff_daily(end="2019-01-01")
    assert len(tr) == 2 and (tr.index < pd.Timestamp("2019-01-01", tz="UTC")).all()
    one = xs.load_ff_daily(start="2018-12-31", end=pd.Timestamp("2019-01-01"))
    assert list(one["Mkt-RF"]) == [0.02]
    tz_naive_end = xs.load_ff_daily(end=pd.Timestamp("2019-01-02"))
    assert tz_naive_end.index.max() == pd.Timestamp("2018-12-31", tz="UTC")
