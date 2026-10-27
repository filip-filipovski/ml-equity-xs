"""Unit tests for the www.sec.gov bulk data-set additions to ``equities_xs``
(contact User-Agent, streaming download, listings, company_tickers, master index,
Notes customer-supplier edges, name resolution, insider transactions, loaders).

Synthetic inputs only; no network."""

from __future__ import annotations

import zipfile
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from eqxs import equities_xs as xs


# --------------------------------------------------------------------- HTTP / UA
def test_sec_user_agent_constant_and_host_routing() -> None:
    assert xs.SEC_USER_AGENT.startswith("eqxs-research/1.0 (")
    for u in ("https://www.sec.gov/files/company_tickers.json",
              "https://data.sec.gov/submissions/CIK0000320193.json",
              "https://efts.sec.gov/x", "https://sec.gov/"):
        assert xs.user_agent_for(u) == xs.SEC_USER_AGENT
    for u in ("https://www.nasdaqtrader.com/x.txt", "https://notsec.gov.example.com/",
              "https://mba.tuck.dartmouth.edu/f.zip"):
        assert xs.user_agent_for(u) == xs.USER_AGENT
        assert "@" not in xs.user_agent_for(u)


def test_polite_client_sends_contact_ua_only_to_sec(monkeypatch) -> None:
    seen: list[tuple[str, str]] = []

    class R:
        status_code, content = 200, b"ok"

    class H:
        def get(self, url, headers=None, **kw):
            seen.append((url, headers["User-Agent"]))
            return R()

    c = xs.PoliteClient(min_interval=0.0)
    c._client = H()
    c.get("https://www.sec.gov/a")
    c.get("https://www.nasdaqtrader.com/b")
    assert seen == [("https://www.sec.gov/a", xs.SEC_USER_AGENT),
                    ("https://www.nasdaqtrader.com/b", xs.USER_AGENT)]


def test_write_raw_records_per_host_user_agent(tmp_path) -> None:
    m1 = xs.write_raw(tmp_path / "a.json", b"{}", "https://www.sec.gov/files/a.json")
    m2 = xs.write_raw(tmp_path / "b.txt", b"x", "https://www.nasdaqtrader.com/b.txt")
    assert m1["user_agent"] == xs.SEC_USER_AGENT and m2["user_agent"] == xs.USER_AGENT


class _Stream:
    def __init__(self, code: int, body: bytes, headers: dict[str, str], fail_after: int | None):
        self.status_code, self._body, self.headers, self._fail = code, body, headers, fail_after

    def iter_raw(self, chunk: int):
        for i in range(0, len(self._body), 2):
            if self._fail is not None and i >= self._fail:
                raise OSError("connection reset")
            yield self._body[i:i + 2]


def test_download_streams_and_resumes_with_range(tmp_path, monkeypatch) -> None:
    body = b"0123456789ABCDEF"
    calls: list[dict[str, str]] = []

    class H:
        def __init__(self) -> None:
            self.n = 0

        @contextmanager
        def stream(self, method, url, headers=None):
            calls.append(dict(headers))
            self.n += 1
            if self.n == 1:  # first attempt dies after 6 bytes
                yield _Stream(200, body, {"content-length": str(len(body))}, fail_after=6)
            else:
                start = int(headers["Range"].split("=")[1].rstrip("-"))
                yield _Stream(206, body[start:], {"content-range": f"bytes {start}-15/16"}, None)

    monkeypatch.setattr(xs.time, "sleep", lambda s: None)
    c = xs.PoliteClient(min_interval=0.0)
    c._client = H()
    dest = tmp_path / "f.zip"
    r = c.download("https://www.sec.gov/files/f.zip", dest)
    assert dest.read_bytes() == body and not (tmp_path / "f.zip.part").exists()
    assert r["size"] == 16 and r["sha256"] == xs.sha256_bytes(body)
    assert "Range" not in calls[0] and calls[1]["Range"] == "bytes=6-"
    assert all(h["User-Agent"] == xs.SEC_USER_AGENT for h in calls)
    assert all(h["Accept-Encoding"] == "identity" for h in calls)


def test_download_404_and_hard_403(tmp_path) -> None:
    class H:
        def __init__(self, code):
            self.code = code

        @contextmanager
        def stream(self, method, url, headers=None):
            yield _Stream(self.code, b"", {}, None)

    c = xs.PoliteClient(min_interval=0.0)
    c._client = H(404)
    assert c.download("https://www.sec.gov/x.zip", tmp_path / "x.zip") is None
    c._client = H(403)
    with pytest.raises(xs.FetchError):
        c.download("https://www.sec.gov/x.zip", tmp_path / "x.zip")


# ------------------------------------------------------------------ listings
LISTING = """<table>
<tr><td>2026 09<br>Monthly</td><td>ZIP</td><td>53.48 MB</td>
  <td><a href="/files/dera/data/financial-statement-notes-data-sets/2026_09_notes.zip">x</a></td></tr>
<tr><td><a href="/files/dera/data/financial-statement-notes-data-sets/2010q1_notes_1.zip">2010 Q1</a></td>
  <td>ZIP</td><td>521.41 KB</td></tr>
<tr><td><a href="https://www.sec.gov/files/x/2006q1_form345.zip">2006 Q1</a></td><td>ZIP</td><td>1.5 GB</td></tr>
<tr><td><a href="/files/readme.pdf">readme</a></td></tr>
</table>"""


def test_parse_dera_listing_and_periods() -> None:
    df = xs.parse_dera_listing(LISTING)
    assert list(df["period"]) == ["2006q1", "2010q1", "2026m09"]
    assert list(df["period_kind"]) == ["quarter", "quarter", "month"]
    assert df.iloc[1]["url"].startswith("https://www.sec.gov/files/dera/")
    assert list(df["size_listed"]) == [1_500_000_000, 521_410, 53_480_000]
    assert xs.parse_size_text("27.41 KB") == 27_410 and xs.parse_size_text("n/a") is None
    s, e = xs.period_bounds("2025m12")
    assert s == pd.Timestamp("2025-12-01") and e == pd.Timestamp("2026-01-01")
    s, e = xs.period_bounds("2009q2")
    assert s == pd.Timestamp("2009-04-01") and e == pd.Timestamp("2009-07-01")


def test_full_index_quarters_and_url() -> None:
    q = xs.full_index_quarters("2004q3", "2005-04-02")
    assert q == ["2004q3", "2004q4", "2005q1", "2005q2"]
    assert xs.full_index_url("2004q1") == \
        "https://www.sec.gov/Archives/edgar/full-index/2004/QTR1/master.gz"


def test_parse_master_index() -> None:
    txt = ("Description: Master Index\n\nCIK|Company Name|Form Type|Date Filed|Filename\n"
           "--------------------------------------------------------------------------------\n"
           "1000045|NICHOLAS FINANCIAL INC|10-Q|2004-02-13|edgar/data/1000045/"
           "0001193125-04-021379.txt\n"
           "320193|APPLE COMPUTER INC|4|2004-01-05|edgar/data/320193/0001181431-04-000123.txt\n")
    df = xs.parse_master_index(txt)
    assert len(df) == 2 and df.iloc[1]["cik"] == 320193
    assert df.iloc[0]["adsh"] == "0001193125-04-021379"
    assert df.iloc[1]["filed"] == pd.Timestamp("2004-01-05") and df.iloc[1]["form"] == "4"


# ----------------------------------------------------------- company_tickers
def test_company_tickers_parse_and_merge() -> None:
    t1 = xs.parse_company_tickers({"0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple Inc."},
                                   "1": {"cik_str": 1, "ticker": "ZZZ", "title": "Zed"}})
    t2 = xs.parse_company_tickers_exchange({"fields": ["cik", "name", "ticker", "exchange"],
                                            "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"],
                                                     [2, "Other", "OTH", None]]})
    ct = xs.build_company_tickers(t1, t2, "20261011").set_index("ticker")
    assert ct.loc["AAPL", "exchange"] == "Nasdaq" and ct.loc["AAPL", "title"] == "Apple Inc."
    assert ct.loc["ZZZ", "exchange"] == "" and ct.loc["OTH", "title"] == "Other"
    assert ct.loc["OTH", "cik"] == 2 and (ct["fetched_utc"] == "20261011").all()


# -------------------------------------------------- customer-supplier edges
def test_segments_axis_and_names() -> None:
    seg = ("ConcentrationRiskByBenchmark=SalesRevenueNet;ConcentrationRiskByType="
           "CustomerConcentrationRisk;MajorCustomers=WalMartStoresInc;")
    assert xs.parse_segments(seg)[2] == ("MajorCustomers", "WalMartStoresInc")
    assert xs.customer_axis_member(seg) == ("MajorCustomers", "WalMartStoresInc")
    assert xs.customer_axis_member("CustomerType=Government;") is None
    assert xs.customer_axis_member("LocationOfCustomer=US;") is None
    assert xs.customer_axis_member(
        "FiniteLivedIntangibleAssetsByMajorClass=CustomerRelationships;") is None
    assert xs.customer_axis_member("CustomerName=AppleInc;") == ("CustomerName", "AppleInc")
    assert xs.split_camel("WalMartStoresInc") == "Wal Mart Stores Inc"
    assert xs.split_camel("USNavy") == "US Navy"
    assert xs.split_camel("Customer4Member") == "Customer 4"
    for n in ("Customer A", "Two Customers", "Largest Customer", "Top Ten Customers",
              "Customer C Concentration Risk", "Significant Distributor 3", "Customera",
              "Vendor One", "Commercial", "International Customers", "Related Parties"):
        assert xs.is_anonymous_customer(n), n
    for n in ("Apple Inc", "General Electric", "US Navy", "National Grid", "Target"):
        assert not xs.is_anonymous_customer(n), n
    assert xs.is_government_customer("US Government") and xs.is_government_customer("US Navy")
    assert not xs.is_government_customer("General Electric")


def _fsn_tables():
    dim = pd.DataFrame({
        "dimhash": ["0x1", "0x2", "0x3", "0x4"],
        "segments": ["ConcentrationRiskByBenchmark=SalesRevenueNet;MajorCustomers=AppleInc;",
                     "ConcentrationRiskByBenchmark=AccountsReceivable;MajorCustomers=CustomerA;",
                     "BusinessSegments=Retail;", "MajorCustomers=AppleInc;"]})
    num = pd.DataFrame({
        "adsh": ["0000000001-12-000001"] * 5,
        "tag": ["ConcentrationRiskPercentage", "ConcentrationRiskPercentage1", "Revenues",
                "Revenues", "Assets"],
        "ddate": ["20111231", "20111231", "20111231", "20111231", "20111231"],
        "qtrs": ["4", "0", "4", "4", "0"], "uom": ["pure", "pure", "USD", "USD", "USD"],
        "dimh": ["0x1", "0x2", "0x4", "0x3", "0x1"],
        "value": ["0.153", "22", "5000000", "9", "1"], "coreg": [None, None, None, None, None]})
    sub = pd.DataFrame({"adsh": ["0000000001-12-000001"], "cik": ["111"], "name": ["SUPPLIER INC"],
                        "sic": ["3674"], "form": ["10-K"], "fy": ["2011"], "fp": ["FY"],
                        "filed": ["20120215"], "accepted": ["2012-02-15 16:44:00.0"]})
    return dim, num, sub


def test_customer_edges_from_tables() -> None:
    dim, num, sub = _fsn_tables()
    dims = xs.customer_dims(dim)
    assert set(dims["dimh"]) == {"0x1", "0x2", "0x4"}
    e = xs.customer_edges_from_tables(num, dims, sub, "2012q1")
    assert len(e) == 3  # Assets tag and the non-customer dimension are dropped
    pct = e[e["tag"] == "ConcentrationRiskPercentage"].iloc[0]
    assert pct["pct"] == pytest.approx(15.3) and not pct["pct_scale_suspect"]
    assert pct["customer_name"] == "Apple Inc" and pct["is_revenue_benchmark"]
    assert pct["supplier_cik"] == 111 and pct["ddate"] == pd.Timestamp("2011-12-31")
    # FSN accepted is US/Eastern wall clock: 16:44 EST = 21:44 UTC
    assert pct["accepted_fsn"] == pd.Timestamp("2012-02-15 21:44", tz="UTC")
    typed = e[e["tag"] == "ConcentrationRiskPercentage1"].iloc[0]
    assert typed["pct"] == 22 and typed["pct_scale_suspect"] and typed["anonymous"]
    assert not typed["is_revenue_benchmark"]
    amt = e[e["tag"] == "Revenues"].iloc[0]
    assert np.isnan(amt["pct"]) and amt["value"] == 5_000_000
    assert not e["vendor_hint"].any()
    assert list(e.columns) == list(xs.EDGE_COLUMNS)


def _tsv(df: pd.DataFrame) -> bytes:
    return df.to_csv(sep="\t", index=False, na_rep="").encode()


def test_extract_customer_edges_from_notes_zip(tmp_path) -> None:
    dim, num, sub = _fsn_tables()
    p = tmp_path / "2012q1_notes.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("dim.tsv", _tsv(dim))
        zf.writestr("num.tsv", _tsv(num.assign(version="us-gaap/2011", iprx="0")))
        zf.writestr("sub.tsv", _tsv(sub.assign(countryba="US")))
        zf.writestr("tag.tsv", "tag\tversion\n")
    e, meta = xs.extract_customer_edges_from_notes_zip(p)
    assert len(e) == 3 and meta["n_customer_dimhash"] == 3 and meta["skipped_num"] == 0
    assert (e["dataset"] == "2012q1").all()
    assert set(e["customer_member"]) == {"AppleInc", "CustomerA"}


def _index() -> xs.NameIndex:
    names = pd.DataFrame({"cik": [1, 2, 3, 4, 5, 6, 7],
                          "name": ["Apple Inc.", "WAL-MART STORES INC", "Apple Hospitality REIT",
                                   "General Electric Co", "Acme Holdings Corp",
                                   "Acme Holdings Ltd", "Precision Castparts Corp"]})
    return xs.NameIndex.build(names, has_ticker=[1, 2, 4, 5])


def test_name_index_resolution_is_conservative() -> None:
    ix = _index()
    assert ix.resolve("Apple Inc") == (1, "exact", 1.0)
    assert ix.resolve("Wal Mart Stores Inc")[:2] == (2, "exact")
    assert ix.resolve("WalMart Stores")[:2] == (2, "exact_compact")
    assert ix.resolve("General Electric Company")[:2] == (4, "exact")
    # two CIKs share the key; only one has a ticker -> picked
    assert ix.resolve("Acme Holdings")[:2] == (5, "exact")
    # normalised-equal names match exactly; partial names never fall through to a guess
    assert ix.resolve("Apple")[:2] == (1, "exact")
    assert ix.resolve("Apple Hospitality")[0] is None  # missing "REIT": no guess
    assert ix.resolve("Hospitality Partners")[0] is None
    assert ix.resolve("Precision Castparts Corp Aerostructures")[0] is None
    ix2 = xs.NameIndex.build(pd.DataFrame({"cik": [10, 11], "name": ["Foo Bar", "Foo Bar Inc"]}))
    assert ix2.resolve("Foo Bar") == (None, "ambiguous", 1.0)
    assert ix2.candidates("Foo Bar") == (10, 11)


def test_name_index_prefix_listed() -> None:
    names = pd.DataFrame({"cik": [1, 2, 3, 4, 5, 6],
                          "name": ["AMAZON COM INC", "Ford Motor Co", "Ford Motor Credit Co LLC",
                                   "Delta Air Lines Inc", "Delta Apparel Inc", "Amazonica Corp"]})
    ix = xs.NameIndex.build(names, has_ticker=[1, 2, 4, 5])
    c, meth, score = ix.resolve("Amazon")
    assert (c, meth) == (1, "prefix_listed") and 0 < score < 1
    assert ix.resolve("Ford")[:2] == (2, "prefix_listed")  # the credit arm has no ticker
    assert ix.resolve("Delta")[0] is None  # two listed names start with "delta"
    assert ix.resolve("Motor")[0] is None  # not a leading token
    # a listed firm's FORMER name never feeds the prefix rule
    allnames = pd.concat([names, pd.DataFrame({"cik": [7, 7], "name": ["TAP Real Estate Inc",
                                                                       "Tesoro Enterprises"]})])
    ix2 = xs.NameIndex.build(allnames, has_ticker=[1, 2, 4, 5, 7],
                             current_names=allnames[allnames["name"] != "Tesoro Enterprises"])
    assert ix2.resolve("Tesoro")[0] is None
    assert ix2.resolve("Tesoro Enterprises")[:2] == (7, "exact")  # exact still sees it


def test_name_index_token_match() -> None:
    # "international" is near-ubiquitous in this name universe -> tiny IDF weight
    names = pd.DataFrame({"cik": list(range(1, 30)),
                          "name": ["Zyx Quantum Devices"]
                          + [f"Filler{i} Widgets International" for i in range(28)]})
    ix = xs.NameIndex.build(names)
    c, meth, score = ix.resolve("Zyx Quantum Devices International")
    assert (c, meth) == (1, "token") and score >= 0.9
    # a distinctive extra token is never absorbed
    assert ix.resolve("Zyx Quantum Devices Holdco Partners")[0] is None
    assert ix.resolve("Zyx Quantum Semiconductor")[0] is None


def test_resolve_customers_methods() -> None:
    dim, num, sub = _fsn_tables()
    e = xs.customer_edges_from_tables(num, xs.customer_dims(dim), sub, "2012q1")
    extra = e.iloc[[0, 0]].copy()
    extra["customer_name"] = ["US Government", "Supplier"]
    extra["government"] = [True, False]
    e = pd.concat([e, extra], ignore_index=True)
    ix = xs.NameIndex.build(pd.DataFrame({"cik": [1, 111], "name": ["Apple Inc.", "Supplier Inc"]}))
    r = xs.resolve_customers(e, ix)
    m = dict(zip(r["customer_name"], r["match_method"], strict=False))
    assert m["Apple Inc"] == "exact" and m["Customer A"] == "anonymous"
    assert m["US Government"] == "government" and m["Supplier"] == "self"
    assert r.loc[r["customer_name"] == "Apple Inc", "customer_cik"].iloc[0] == 1
    assert r.loc[r["customer_name"] == "Supplier", "customer_cik"].isna().all()
    assert r.loc[r["customer_name"] == "Apple Inc", "match_name"].iloc[0] == "Apple Inc."
    # prefix hits are suggestions only
    pe = e.iloc[[0]].copy()
    pe["customer_name"] = "Amazon"
    ixp = xs.NameIndex.build(pd.DataFrame({"cik": [9], "name": ["AMAZON COM INC"]}), has_ticker=[9])
    rp = xs.resolve_customers(pe, ixp).iloc[0]
    assert rp["match_method"] == "prefix_listed_candidate" and pd.isna(rp["customer_cik"])
    assert rp["match_candidates"] == "9" and rp["match_name"] == "AMAZON COM INC"


# ------------------------------------------------------------------- insider
def _insider_tables():
    sub = pd.DataFrame({"ACCESSION_NUMBER": ["A1", "A2", "A3"],
                        "FILING_DATE": ["31-JAN-2015", "02-FEB-2015", "27-FEB-2015"],
                        "PERIOD_OF_REPORT": ["29-JAN-2015", "30-JAN-2015", "25-FEB-2015"],
                        "DOCUMENT_TYPE": ["4", "4", "4/A"], "ISSUERCIK": ["0000000100"] * 3,
                        "ISSUERNAME": ["ISSUER CO"] * 3, "ISSUERTRADINGSYMBOL": ["ISS"] * 3})
    owner = pd.DataFrame({"ACCESSION_NUMBER": ["A1", "A1", "A2", "A3"],
                          "RPTOWNERCIK": ["0000000900", "0000000901", "0000000902", "0000000900"],
                          "RPTOWNERNAME": ["Fund LP", "Fund GP", "Doe Jane", "Fund LP"],
                          "RPTOWNER_RELATIONSHIP": ["TenPercentOwner", "TenPercentOwner",
                                                    "Director,Officer", "TenPercentOwner"],
                          "RPTOWNER_TITLE": [None, None, "CFO", None]})
    tx = pd.DataFrame({"ACCESSION_NUMBER": ["A1", "A1", "A2", "A2", "A3"],
                       "NONDERIV_TRANS_SK": ["1", "2", "3", "4", "5"],
                       "SECURITY_TITLE": ["Common Stock"] * 5,
                       "TRANS_DATE": ["29-JAN-2015", "29-JAN-2015", "30-JAN-2015", "30-JAN-2015",
                                      "25-FEB-2015"],
                       "TRANS_FORM_TYPE": ["4"] * 5, "TRANS_CODE": ["P", "P", "S", "M", "P"],
                       "EQUITY_SWAP_INVOLVED": ["0"] * 5, "TRANS_TIMELINESS": [None, None, "L", None, None],
                       "TRANS_SHARES": ["100", "50", "10", "999", "20"],
                       "TRANS_PRICEPERSHARE": ["10.0", "11.0", "12.5", "0", None],
                       "TRANS_ACQUIRED_DISP_CD": ["A", "A", "D", "A", "A"],
                       "SHRS_OWND_FOLWNG_TRANS": ["1000", "1050", "5", "1004", "1070"],
                       "DIRECT_INDIRECT_OWNERSHIP": ["D", "D", "D", "D", "I"]})
    return sub, owner, tx


def test_build_insider_transactions_and_ts() -> None:
    sub, owner, tx = _insider_tables()
    acc = pd.DataFrame({"adsh": ["A2"], "accepted": [pd.Timestamp("2015-02-02 21:05", tz="UTC")]})
    t = xs.build_insider_transactions(sub, owner, tx, acc, dataset="2015q1")
    assert len(t) == 4 and set(t["trans_code"]) == {"P", "S"}  # the M line is dropped
    a1 = t[t["adsh"] == "A1"]
    assert (a1["owner_id"] == 900).all() and (a1["n_owners"] == 2).all()
    assert a1["is_ten_pct"].all() and not a1["is_director"].any()
    assert a1["dollars"].tolist() == [1000.0, 550.0]
    # no acceptance -> 22:00 ET of the filing date (31 Jan 2015, EST) = 03:00 UTC next day
    assert (a1["ts_source"] == "filed_2200ET_bound").all()
    assert a1["ts"].iloc[0] == pd.Timestamp("2015-02-01 03:00", tz="UTC")
    a2 = t[t["adsh"] == "A2"].iloc[0]
    assert a2["ts_source"] == "accepted" and a2["ts"] == pd.Timestamp("2015-02-02 21:05", tz="UTC")
    # a submissions stamp later than 22:00 ET of the filing date (the +4/5 h shift) loses
    late = pd.DataFrame({"adsh": ["A2"], "accepted": [pd.Timestamp("2015-02-03 05:00", tz="UTC")]})
    t2 = xs.build_insider_transactions(sub, owner, tx, late)
    b = t2[t2["adsh"] == "A2"].iloc[0]
    assert b["ts_source"] == "filed_2200ET_bound" and b["ts"] == pd.Timestamp("2015-02-03 03:00",
                                                                             tz="UTC")
    assert a2["is_director"] and a2["is_officer"] and a2["late"]
    a3 = t[t["adsh"] == "A3"].iloc[0]
    assert np.isnan(a3["dollars"]) and a3["doc_type"] == "4/A"
    assert a2["trans_date"] == pd.Timestamp("2015-01-30")
    assert list(t.columns) == list(xs.INSIDER_TX_COLUMNS)


def test_insider_monthly_keyed_by_filing_availability() -> None:
    sub, owner, tx = _insider_tables()
    t = xs.build_insider_transactions(sub, owner, tx, None)
    m = xs.insider_monthly(t).set_index("month")
    # A1 traded 29 Jan but became available 1 Feb (UTC) -> counted in February; A3 filed
    # 27 Feb -> bound 22:00 EST = 28 Feb 03:00 UTC, still February
    assert list(m.index) == [pd.Timestamp("2015-02-01")]
    feb = m.loc[pd.Timestamp("2015-02-01")]
    assert feb["buy_shares"] == 170 and feb["sell_shares"] == 10
    assert feb["net_shares"] == 160 and feb["buy_dollars"] == 1550.0
    assert feb["net_dollars"] == pytest.approx(1550.0 - 125.0)
    assert feb["n_buy_tx"] == 3 and feb["n_buy_filings"] == 2 and feb["n_sell_tx"] == 1
    assert feb["n_insiders_buying"] == 1 and feb["n_insiders_selling"] == 1  # joint filer once
    assert feb["n_missing_price"] == 1 and feb["n_dollars_suspect"] == 0
    assert feb["ts"] == pd.Timestamp("2015-02-28 23:59:59.999999", tz="UTC")
    # a filing accepted on the last evening of the month (21:30 EST 28 Feb, before the
    # 22:00 ET bound) lands in the next UTC month
    sub.loc[2, "FILING_DATE"] = "28-FEB-2015"
    acc = pd.DataFrame({"adsh": ["A3"], "accepted": [pd.Timestamp("2015-03-01 02:30", tz="UTC")]})
    m2 = xs.insider_monthly(xs.build_insider_transactions(sub, owner, tx, acc))
    assert list(m2["month"]) == [pd.Timestamp("2015-02-01"), pd.Timestamp("2015-03-01")]


def test_insider_dollar_typo_guard() -> None:
    sub, owner, tx = _insider_tables()
    tx.loc[0, "TRANS_PRICEPERSHARE"] = "250000000"  # mis-keyed price
    t = xs.build_insider_transactions(sub, owner, tx, None)
    assert t["dollars_suspect"].sum() == 1
    feb = xs.insider_monthly(t).iloc[0]
    assert feb["buy_dollars"] == 550.0 and feb["n_dollars_suspect"] == 1
    assert feb["buy_shares"] == 170


def test_read_insider_zip_roundtrip(tmp_path) -> None:
    sub, owner, tx = _insider_tables()
    p = tmp_path / "2015q1_form345.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("SUBMISSION.tsv", _tsv(sub.assign(REMARKS="")))
        zf.writestr("REPORTINGOWNER.tsv", _tsv(owner.assign(RPTOWNER_TXT="")))
        zf.writestr("NONDERIV_TRANS.tsv", _tsv(tx))
    t, meta = xs.insider_transactions_from_zip(p)
    assert meta["n_ps_rows"] == 4 and meta["dataset"] == "2015q1" and meta["skipped_tx"] == 0
    assert (t["ts_source"] == "filed_2200ET_bound").all()


def test_form345_acceptance_from_submissions() -> None:
    subs = [{"filings": {"recent": {
        "accessionNumber": ["X1", "X2", "X3"], "form": ["4", "10-K", "4/A"],
        "acceptanceDateTime": ["2015-02-02T21:05:00.000Z", "2015-02-03T12:00:00.000Z",
                               "2015-03-01T01:00:00.000Z"]}}}]
    a = xs.form345_acceptance(subs)
    assert list(a["adsh"]) == ["X1", "X3"]
    assert a.iloc[0]["accepted"] == pd.Timestamp("2015-02-02 21:05", tz="UTC")


def test_after_hours_bound_dst() -> None:
    b = xs.edgar_accept_after_hours_bound(pd.Series(pd.to_datetime(["2015-07-01", "2015-01-02"])))
    assert b.iloc[0] == pd.Timestamp("2015-07-02 02:00", tz="UTC")  # EDT
    assert b.iloc[1] == pd.Timestamp("2015-01-03 03:00", tz="UTC")  # EST


# ------------------------------------------------------------------- loaders
def _write(root, sub: str, label: str, df: pd.DataFrame) -> None:
    d = root / sub / f"quarter={label}"
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "part-0000.parquet", index=False)


def _qlabel(t: pd.Timestamp) -> str:
    return f"{t.year}q{t.quarter}"


def test_new_loaders_are_zone_aware(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(xs, "XS_ROOT", tmp_path)
    sealed = xs.EQUITY_SEALED_START
    stamps = [pd.Timestamp("2010-05-01", tz="UTC"), sealed - pd.Timedelta(days=60),
              sealed + pd.Timedelta(days=1)]
    for t in stamps:
        lab = _qlabel(t)
        _write(tmp_path, "links/customer_supplier", lab, pd.DataFrame(
            {"supplier_cik": [1, 1], "customer_cik": pd.array([5, None], dtype="Int64"),
             "ts": [t, t]}))
        _write(tmp_path, "insider/monthly", lab, pd.DataFrame({"cik": [1], "month": [t], "ts": [t]}))
        _write(tmp_path, "insider/transactions", lab, pd.DataFrame(
            {"issuer_cik": pd.array([1, 2], dtype="Int64"), "ts": [t, t]}))
    opened: list[str] = []
    real = xs.safe_load_parquet

    def spy(path, **kw):
        opened.append(str(path))
        assert kw.get("zone_config") == xs.EQUITIES_ZONES_PATH
        return real(path, **kw)

    monkeypatch.setattr(xs, "safe_load_parquet", spy)
    lk = xs.load_customer_links()
    assert len(lk) == 4 and pd.to_datetime(lk["ts"]).max() < sealed
    assert len(xs.load_customer_links(resolved_only=True)) == 2
    assert len(xs.load_customer_links(end="2011-01-01")) == 2
    assert len(xs.load_insider_monthly(start="2011-01-01")) == 1
    assert len(xs.load_insider_transactions(issuers=[2])) == 2
    sealed_lab = f"quarter={_qlabel(stamps[2])}"
    assert all(sealed_lab not in p for p in opened)
    n_before = len(opened)
    xs.load_customer_links(end="2011-01-01")
    assert all(_qlabel(stamps[1]) not in p for p in opened[n_before:])


def test_load_static_company_tickers(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(xs, "XS_ROOT", tmp_path)
    pd.DataFrame({"cik": [1], "ticker": ["A"], "title": ["A"], "exchange": [""],
                  "fetched_utc": ["20261011"]}).to_parquet(tmp_path / "company_tickers.parquet")
    assert len(xs.load_static("company_tickers")) == 1


def test_sec_bulk_root_layout() -> None:
    assert xs.SEC_BULK_ROOT == xs.RAW_ROOT / "sec_bulk"
