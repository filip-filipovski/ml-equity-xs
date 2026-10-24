"""US equity cross-section from free, keyless sources (adapters, parsers, loaders).

Everything lives under ``data_cache/equities_xs/`` (``data_cache`` is a symlink into
the external-SSD image). Producer: ``scripts/ingest/ingest_equities_xs.py``.
Report: ``reports/research/US_EQUITY_XS_DATASET_2026-10-10.md``.

Sources (what actually worked on 2026-10-10, $0, no accounts, no e-mail anywhere)
--------------------------------------------------------------------------------
* **Nasdaq Trader symbol directories** (``nasdaqlisted.txt`` + ``otherlisted.txt``)
  -> ``universe_current.parquet`` (today's listed universe; no history).
* **yfinance** daily bars for the current common-stock list (1999+): ``Close``
  (split-adjusted, NOT dividend-adjusted), ``Adj Close`` (split + dividend adjusted),
  ``Dividends``, ``Stock Splits`` -> ``yf_daily/year=YYYY/``. Survivor-only by
  construction (Yahoo does not serve most delisted tickers).
* **SEC EDGAR, data.sec.gov** (on 2026-10-10 ``www.sec.gov`` answered 403 to a
  User-Agent without an e-mail address, so the keyless data.sec.gov APIs were used;
  see the 2026-10-11 bulk item below for what changed). The data.sec.gov tables:

  - XBRL ``frames`` API -> enumerate every XBRL filer (``cik_universe.parquet``);
  - XBRL ``companyfacts`` per CIK -> ``sec_fsds/num/fy_quarter=YYYYqN/`` (same XBRL
    facts the FSDS ``num.txt`` is built from: one row per (fact, filing), with the
    filing's accession ``adsh``, ``form``, ``fy``/``fp`` and ``filed`` date, so a
    restated value appears once per filing that reported it -> point-in-time);
  - ``submissions`` JSON per CIK -> ``sec_fsds/sub/`` (one row per 10-K/10-Q/20-F/
    40-F family filing: ``adsh, cik, name, sic, form, period, fy, fp, filed,
    accepted``), ``cik_ticker_map.parquet`` and ``filer_membership.parquet``, and the
    8-K item-2.02 earnings calendar ``events/earnings_8k_all/``.

* **Kenneth French data library** daily factors -> ``ff_daily.parquet``.
* **FRED DGS3MO** reused from ``data_cache/macro/fred/DGS3MO.parquet``.

* **SEC www.sec.gov bulk data sets (added 2026-10-11).** The author authorised a
  declared-contact User-Agent (:data:`SEC_USER_AGENT`, the only place the contact
  e-mail appears; every request to any ``*.sec.gov`` host now carries it, including the
  data.sec.gov calls above). With it www.sec.gov answers 200, unlocking
  ``company_tickers.json`` / ``company_tickers_exchange.json``
  (``company_tickers.parquet``), the Financial Statement **and Notes** data sets
  (``raw/sec_bulk/fsds_notes/``; dimensional facts -> customer-supplier edges in
  ``links/customer_supplier/``), the Insider Transactions data sets
  (``raw/sec_bulk/insider/`` -> ``insider/transactions/`` + ``insider/monthly/``) and
  the EDGAR full-index ``master.gz`` per quarter (``raw/sec_bulk/full_index/``).
  Report: ``reports/research/SEC_BULK_UNLOCK_2026-10-11.md``.

Not obtained: **Stooq** (``robots.txt`` is ``User-agent: * / Disallow: /`` and
``/db/h/`` is behind a JavaScript proof-of-work bot wall -> skipped, not circumvented),
so there is **no free delisted-inclusive price panel** here. Delisted *filers* are
present in the SEC tables (fundamentals + filing history), not in the price table.

Point-in-time and the sealed tripwire
-------------------------------------
Every canonical time-varying table carries a ``ts`` column (UTC) = the instant the row
became knowable, and is physically split by that instant (``year=`` / ``fy_quarter=``
partitions), so study loaders never even open a partition at or after the equities
sealed boundary (``config/data_zones_equities.yaml``, 2022-01-01) and every file they
do open goes through :func:`eqxs.data_zones.safe_load_parquet` with
``zone_config=EQUITIES_ZONES_PATH`` (which raises if a file contains sealed rows).

* prices: ``ts`` = trading date 00:00 UTC (the bar is known after that day's close).
* SEC facts / filings: ``ts`` = EDGAR ``acceptanceDateTime`` when known, else the
  ``filed`` date 00:00 UTC. Use :func:`pit_filter` (``ts <= as_of``) for PIT
  fundamentals; ``ddate`` (period end) must never be used as the availability date.
* earnings events: ``ts`` = 8-K acceptance instant.

Ingest code reads its own outputs through :func:`_ingest_only_read` (plain
``pd.read_parquet``) and must not print anything about rows dated >= 2022-01-01
beyond counts and schema.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from eqxs.data_zones import EQUITIES_ZONES_PATH, safe_load_parquet, sealed_start

logger = logging.getLogger(__name__)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
XS_ROOT: Final[Path] = _PROJECT_ROOT / "data_cache" / "equities_xs"
RAW_ROOT: Final[Path] = XS_ROOT / "raw"

USER_AGENT: Final[str] = "EquityXS/1.0 (non-commercial academic research)"
# Declared-contact User-Agent for EVERY request to any sec.gov host (www.sec.gov,
# data.sec.gov, ...). Authorised by the author 2026-10-11: the e-mail is used only in
# this header (SEC fair-access policy), never sent anywhere else. Single source of truth.
SEC_USER_AGENT: Final[str] = "eqxs-research/1.0 (Filip F; fikas007@gmail.com)"


def is_sec_host(url_or_host: str) -> bool:
    host = urlparse(url_or_host).netloc if "://" in url_or_host else url_or_host
    host = host.split(":")[0].lower()
    return host == "sec.gov" or host.endswith(".sec.gov")


def user_agent_for(url: str, default: str = USER_AGENT) -> str:
    """:data:`SEC_USER_AGENT` for any sec.gov host, else ``default``."""
    return SEC_USER_AGENT if is_sec_host(url) else default

NASDAQ_LISTED_URL: Final[str] = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL: Final[str] = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
SEC_FRAMES_URL: Final[str] = "https://data.sec.gov/api/xbrl/frames/{tax}/{tag}/{uom}/{period}.json"
SEC_FACTS_URL: Final[str] = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_SUB_URL: Final[str] = "https://data.sec.gov/submissions/{name}"
FF3_DAILY_URL: Final[str] = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)
FF_MOM_DAILY_URL: Final[str] = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
)
FF5_DAILY_URL: Final[str] = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)

# Periodic-report forms kept in ``sub`` (FSDS covers 10-K/10-Q/20-F/40-F and amendments).
PERIODIC_FORMS: Final[frozenset[str]] = frozenset(
    {"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-KT/A", "10-QT", "10-QT/A",
     "20-F", "20-F/A", "40-F", "40-F/A"}
)
EQUITY_SEALED_START: Final[pd.Timestamp] = sealed_start(EQUITIES_ZONES_PATH)
# The canonical price tables are split-adjusted (Close) but NOT dividend-adjusted;
# ``adj_close`` (yfinance) carries split + dividend adjustment.
PRICE_ADJUSTMENT_NOTE: Final[str] = (
    "close = split-adjusted, NOT dividend-adjusted; adj_close = split+dividend adjusted (yfinance)"
)


# ---------------------------------------------------------------------------
# Polite HTTP + raw storage
# ---------------------------------------------------------------------------
class FetchError(RuntimeError):
    """Persistent HTTP failure (after retries) or a non-retryable status."""


class TokenBucket:
    """Thread-safe process-wide rate cap: at most ``rate`` acquisitions per second
    (burst ``capacity``). ``acquire`` blocks until a token is available."""

    def __init__(self, rate: float, capacity: float | None = None,
                 clock: Any = time.monotonic, sleep: Any = time.sleep) -> None:
        self.rate = float(rate)
        self.capacity = float(capacity if capacity is not None else rate)
        self._tokens = self.capacity
        self._clock, self._sleep = clock, sleep
        self._t = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._t) * self.rate)
        self._t = now

    def try_acquire(self) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def acquire(self) -> None:
        while not self.try_acquire():
            with self._lock:
                need = (1.0 - self._tokens) / self.rate
            self._sleep(max(need, 0.005))


class GlobalCooldown:
    """Shared pause: any thread that sees a throttle response calls ``trigger``; every
    thread calls ``wait`` before each request and sleeps until the cooldown ends."""

    def __init__(self, seconds: float = 60.0) -> None:
        self.seconds = seconds
        self._until = 0.0
        self._lock = threading.Lock()
        self.n_triggers = 0

    def trigger(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._until <= now:  # do not stack cooldowns from simultaneous 403s
                self.n_triggers += 1
            self._until = max(self._until, now + self.seconds)

    def wait(self) -> None:
        while True:
            with self._lock:
                rem = self._until - time.monotonic()
            if rem <= 0:
                return
            time.sleep(min(rem, 5.0))


class LockedAppender:
    """Append whole lines to files under one lock (safe across worker threads)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def append(self, path: Path, line: str) -> None:
        with self._lock, path.open("a") as fh:
            fh.write(line.rstrip("\n") + "\n")


@dataclass
class PoliteClient:
    """Keyless HTTP client: >= ``min_interval`` s between requests to any one host
    from this client, optional process-wide :class:`TokenBucket` (``bucket``) shared
    across worker threads, exponential backoff on 429 / 5xx / network errors,
    descriptive User-Agent, no e-mail address. ``get`` returns ``None`` on 404.

    ``throttle_403``: data.sec.gov answers 403 when its request-rate threshold is
    exceeded; with a ``cooldown`` set, a 403 triggers a shared pause of all threads
    and the request is retried (up to ``retries_403`` times) instead of failing."""

    user_agent: str = USER_AGENT
    min_interval: float = 1.05
    retries: int = 6
    timeout: float = 120.0
    bucket: TokenBucket | None = None
    cooldown: GlobalCooldown | None = None
    throttle_403: bool = False
    retries_403: int = 10
    _last: dict[str, float] = field(default_factory=dict)
    n_requests: int = 0
    _client: Any = None

    def _http(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(
                headers={"Accept-Encoding": "gzip, deflate"},
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    def ua(self, url: str) -> str:
        """User-Agent for ``url``: :data:`SEC_USER_AGENT` on every sec.gov host."""
        return user_agent_for(url, self.user_agent)

    def _space(self, host: str) -> None:
        if self.cooldown is not None:
            self.cooldown.wait()
        if self.bucket is not None:
            self.bucket.acquire()
        wait = self.min_interval - (time.monotonic() - self._last.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.monotonic()

    def get(self, url: str) -> bytes | None:
        host = urlparse(url).netloc
        last_err = ""
        attempt, n403 = 0, 0
        while attempt < self.retries:
            attempt += 1
            self._space(host)
            self.n_requests += 1
            try:
                r = self._http().get(url, headers={"User-Agent": self.ua(url)})
                code = r.status_code
            except Exception as exc:  # noqa: BLE001 - network layer
                code, r, last_err = -1, None, f"{type(exc).__name__}: {exc}"
            if code == 200 and r is not None:
                return r.content
            if code == 404:
                return None
            if code == 403 and self.throttle_403 and self.cooldown is not None:
                n403 += 1
                if n403 > self.retries_403:
                    raise FetchError(f"{url} -> HTTP 403 x{n403} (throttle never cleared)")
                logger.warning("%s -> HTTP 403 (rate threshold); global cooldown %.0fs (%d/%d)",
                               url, self.cooldown.seconds, n403, self.retries_403)
                self.cooldown.trigger()
                attempt -= 1  # throttle pauses do not consume the ordinary retry budget
                continue
            if code not in (-1, 429) and code < 500:
                raise FetchError(f"{url} -> HTTP {code}")
            back = min(2.0 * 2 ** (attempt - 1), 120.0)
            logger.warning("%s -> HTTP %s %s; retry %d/%d in %.0fs", url, code, last_err,
                           attempt, self.retries, back)
            time.sleep(back)
        raise FetchError(f"{url}: gave up after {self.retries} attempts ({last_err})")

    def head(self, url: str) -> dict[str, Any] | None:
        """HEAD ``url`` (same spacing / bucket / cooldown / 403 handling as :meth:`get`).
        Returns ``{"status", "size", "last_modified", "etag"}`` or ``None`` on 404."""
        host = urlparse(url).netloc
        attempt, n403 = 0, 0
        while attempt < self.retries:
            attempt += 1
            self._space(host)
            self.n_requests += 1
            try:
                r = self._http().head(url, headers={"User-Agent": self.ua(url),
                                                    "Accept-Encoding": "identity"})
                code = r.status_code
            except Exception:  # noqa: BLE001 - network layer
                code, r = -1, None
            if code == 200 and r is not None:
                cl = r.headers.get("content-length")
                return {"status": 200, "size": int(cl) if cl and cl.isdigit() else None,
                        "last_modified": r.headers.get("last-modified"),
                        "etag": r.headers.get("etag")}
            if code == 404:
                return None
            if code == 403 and self.throttle_403 and self.cooldown is not None:
                n403 += 1
                if n403 > self.retries_403:
                    raise FetchError(f"HEAD {url} -> HTTP 403 x{n403}")
                self.cooldown.trigger()
                attempt -= 1
                continue
            if code not in (-1, 429) and code < 500:
                raise FetchError(f"HEAD {url} -> HTTP {code}")
            time.sleep(min(2.0 * 2 ** (attempt - 1), 120.0))
        raise FetchError(f"HEAD {url}: gave up after {self.retries} attempts")

    def download(self, url: str, dest: Path, *, chunk: int = 1 << 20) -> dict[str, Any] | None:
        """Stream ``url`` to ``dest`` byte-for-byte (``Accept-Encoding: identity``), with
        HTTP Range resume from ``dest.part`` across retries and process restarts.
        Memory stays O(``chunk``). Returns ``{"size", "sha256"}`` (of the file as served)
        or ``None`` on 404. Same spacing / bucket / cooldown / 403 rules as :meth:`get`."""
        host = urlparse(url).netloc
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        last_err = ""
        attempt, n403 = 0, 0
        while attempt < self.retries:
            attempt += 1
            self._space(host)
            self.n_requests += 1
            have = part.stat().st_size if part.exists() else 0
            hdr = {"User-Agent": self.ua(url), "Accept-Encoding": "identity"}
            if have:
                hdr["Range"] = f"bytes={have}-"
            code = -1
            try:
                with self._http().stream("GET", url, headers=hdr) as r:
                    code = r.status_code
                    if code in (200, 206):
                        mode = "ab" if code == 206 else "wb"
                        with part.open(mode) as fh:
                            for b in r.iter_raw(chunk):
                                fh.write(b)
                        total = r.headers.get("content-range", "").rsplit("/", 1)[-1]
                        size = part.stat().st_size
                        if code == 206 and total.isdigit() and size != int(total):
                            raise OSError(f"short read {size}/{total}")
                        if code == 200:
                            cl = r.headers.get("content-length")
                            if cl and cl.isdigit() and size != int(cl):
                                raise OSError(f"short read {size}/{cl}")
            except Exception as exc:  # noqa: BLE001 - network layer; .part is kept
                last_err = f"{type(exc).__name__}: {exc}"
                code = -1 if code in (200, 206) else code
            if code in (200, 206):
                h = hashlib.sha256()
                with part.open("rb") as fh:
                    for b in iter(lambda: fh.read(chunk), b""):
                        h.update(b)
                size = part.stat().st_size
                part.replace(dest)
                return {"size": size, "sha256": h.hexdigest()}
            if code == 404:
                return None
            if code == 416 and part.exists():  # range past end: restart cleanly
                part.unlink()
                continue
            if code == 403 and self.throttle_403 and self.cooldown is not None:
                n403 += 1
                if n403 > self.retries_403:
                    raise FetchError(f"{url} -> HTTP 403 x{n403} (throttle never cleared)")
                logger.warning("%s -> HTTP 403; global cooldown %.0fs (%d/%d)", url,
                               self.cooldown.seconds, n403, self.retries_403)
                self.cooldown.trigger()
                attempt -= 1
                continue
            if code not in (-1, 429) and code < 500:
                raise FetchError(f"{url} -> HTTP {code}")
            back = min(2.0 * 2 ** (attempt - 1), 120.0)
            logger.warning("%s -> HTTP %s %s; retry %d/%d in %.0fs", url, code, last_err,
                           attempt, self.retries, back)
            time.sleep(back)
        raise FetchError(f"{url}: gave up after {self.retries} attempts ({last_err})")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_raw(dest: Path, payload: bytes, url: str, *, gzip_it: bool = False,
              sidecar: bool = True, manifest: Path | None = None,
              appender: LockedAppender | None = None) -> dict[str, Any]:
    """Store a downloaded payload exactly as served (optionally gzip-wrapped) plus its
    provenance: a ``<file>.meta.json`` sidecar and/or one line in ``manifest`` (used for
    the tens of thousands of per-CIK SEC files). ``sha256``/``size`` are of the payload
    as served (before the optional gzip wrapper)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(gzip.compress(payload, 6) if gzip_it else payload)
    tmp.replace(dest)
    meta = {
        "url": url,
        "fetched_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "size": len(payload),
        "sha256": sha256_bytes(payload),
        "stored_as": dest.name,
        "gzip_wrapped": gzip_it,
        "user_agent": user_agent_for(url),
    }
    if sidecar:
        Path(str(dest) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if manifest is not None:
        line = json.dumps(meta, separators=(",", ":"))
        if appender is not None:
            appender.append(manifest, line)
        else:
            with manifest.open("a") as fh:
                fh.write(line + "\n")
    return meta


def read_raw(path: Path) -> bytes:
    b = path.read_bytes()
    return gzip.decompress(b) if path.suffix == ".gz" else b


# ---------------------------------------------------------------------------
# 1. Nasdaq Trader symbol directories
# ---------------------------------------------------------------------------
_OTHER_EXCHANGE = {"A": "NYSE American", "N": "NYSE", "P": "NYSE Arca", "Z": "Cboe BZX",
                   "V": "IEX"}
_NASDAQ_TIER = {"Q": "Nasdaq Global Select", "G": "Nasdaq Global Market",
                "S": "Nasdaq Capital Market"}
_NON_COMMON_NAME = re.compile(
    r"\b(warrants?|units?|rights?|preferred|preference|depositary shares? (?:each )?representing"
    r" (?:a )?(?:1/|one-|\d+/)|notes?|debentures?|due \d{4}|% |percent|subordinated|trust preferred"
    r"|when issued|contingent value)\b",
    re.IGNORECASE,
)
_ADR_NAME = re.compile(r"american deposit[ao]ry|\bADS\b|\bADR\b|deposit[ao]ry shares?",
                       re.IGNORECASE)


def _split_pipe(text: str) -> pd.DataFrame:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    footer = [ln for ln in lines if ln.startswith("File Creation Time")]
    body = [ln for ln in lines if not ln.startswith("File Creation Time")]
    header = body[0].split("|")
    rows = [ln.split("|") for ln in body[1:]]
    rows = [r for r in rows if len(r) == len(header)]
    df = pd.DataFrame(rows, columns=header)
    fct = footer[0].split(":", 1)[1].strip("| ").strip() if footer else ""
    df.attrs["file_creation_time"] = fct
    return df


def parse_nasdaq_listed(text: str) -> pd.DataFrame:
    """``nasdaqlisted.txt`` -> canonical rows (exchange 'Nasdaq', tier in ``market_category``)."""
    d = _split_pipe(text)
    out = pd.DataFrame({
        "symbol": d["Symbol"].str.strip(),
        "security_name": d["Security Name"].str.strip(),
        "exchange": "Nasdaq",
        "market_category": d["Market Category"].map(_NASDAQ_TIER).fillna(d["Market Category"]),
        "etf": d["ETF"].eq("Y"),
        "test_issue": d["Test Issue"].eq("Y"),
        "financial_status": d["Financial Status"].str.strip(),
        "round_lot": pd.to_numeric(d["Round Lot Size"], errors="coerce"),
        "cqs_symbol": d["Symbol"].str.strip(),
        "source_file": "nasdaqlisted.txt",
    })
    out.attrs["file_creation_time"] = d.attrs["file_creation_time"]
    return out


def parse_other_listed(text: str) -> pd.DataFrame:
    """``otherlisted.txt`` (NYSE / NYSE American / Arca / BZX / IEX) -> canonical rows."""
    d = _split_pipe(text)
    out = pd.DataFrame({
        "symbol": d["ACT Symbol"].str.strip(),
        "security_name": d["Security Name"].str.strip(),
        "exchange": d["Exchange"].map(_OTHER_EXCHANGE).fillna(d["Exchange"]),
        "market_category": "",
        "etf": d["ETF"].eq("Y"),
        "test_issue": d["Test Issue"].eq("Y"),
        "financial_status": "",
        "round_lot": pd.to_numeric(d["Round Lot Size"], errors="coerce"),
        "cqs_symbol": d["CQS Symbol"].str.strip(),
        "source_file": "otherlisted.txt",
    })
    out.attrs["file_creation_time"] = d.attrs["file_creation_time"]
    return out


def classify_security(symbol: str, name: str, *, etf: bool, test_issue: bool) -> str:
    """``common`` / ``adr`` / ``etf`` / ``test`` / ``other`` (warrant, unit, right,
    preferred, note, when-issued...). Symbol suffix conventions: ACT ``$`` = preferred,
    ``.W``/``.WS`` warrants, ``.U`` units, ``.R`` rights; Nasdaq 5-letter ``W/U/R``."""
    if test_issue:
        return "test"
    if etf:
        return "etf"
    s = symbol.upper()
    if "$" in s or re.search(r"\.(W|WS|WI|U|R|RT)$|\.W[A-Z]$", s):
        return "other"
    if len(s) == 5 and s[-1] in "WUR" and "." not in s and _NON_COMMON_NAME.search(name or ""):
        return "other"
    if _NON_COMMON_NAME.search(name or ""):
        return "other"
    if _ADR_NAME.search(name or ""):
        return "adr"
    return "common"


def build_universe_current(nasdaq_text: str, other_text: str) -> pd.DataFrame:
    a, b = parse_nasdaq_listed(nasdaq_text), parse_other_listed(other_text)
    df = pd.concat([a, b], ignore_index=True)
    df["security_type"] = [classify_security(s, n, etf=e, test_issue=t) for s, n, e, t in
                           zip(df["symbol"], df["security_name"], df["etf"], df["test_issue"],
                               strict=True)]
    df["yf_symbol"] = df["symbol"].map(to_yahoo_symbol)
    df["file_creation_time"] = a.attrs["file_creation_time"] or b.attrs["file_creation_time"]
    return df.drop_duplicates("symbol").reset_index(drop=True)


def to_yahoo_symbol(symbol: str) -> str:
    """ACT/Nasdaq symbol -> Yahoo spelling (``BRK.B`` -> ``BRK-B``)."""
    return symbol.strip().upper().replace(".", "-").replace("/", "-")


# ---------------------------------------------------------------------------
# 2. Price panels (yfinance) + membership
# ---------------------------------------------------------------------------
PRICE_COLUMNS: Final[tuple[str, ...]] = (
    "symbol", "date", "ts", "open", "high", "low", "close", "adj_close", "volume",
    "dividends", "splits", "source",
)


def normalize_yf_download(raw: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    """``yf.download(..., group_by='ticker', auto_adjust=False, actions=True)`` output
    (column MultiIndex ``(ticker, field)``, or flat for a single ticker) -> long frame
    with :data:`PRICE_COLUMNS`. Rows with no close are dropped."""
    fields = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
              "Adj Close": "adj_close", "Volume": "volume", "Dividends": "dividends",
              "Stock Splits": "splits"}
    parts = []
    if raw is None or raw.empty:
        return pd.DataFrame(columns=list(PRICE_COLUMNS))
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        for s in symbols:
            if s not in lvl0:
                continue
            sub = raw[s].copy()
            parts.append(_one_symbol(sub, s, fields))
    else:
        parts.append(_one_symbol(raw.copy(), symbols[0], fields))
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=list(PRICE_COLUMNS))
    return pd.concat(parts, ignore_index=True)


def _one_symbol(sub: pd.DataFrame, sym: str, fields: Mapping[str, str]) -> pd.DataFrame:
    sub = sub.rename(columns=fields)
    for c in fields.values():
        if c not in sub.columns:
            sub[c] = np.nan
    sub = sub[sub["close"].notna()]
    idx = pd.DatetimeIndex(sub.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out = pd.DataFrame({
        "symbol": sym,
        "date": idx.normalize(),
        "open": sub["open"].to_numpy(float), "high": sub["high"].to_numpy(float),
        "low": sub["low"].to_numpy(float), "close": sub["close"].to_numpy(float),
        "adj_close": sub["adj_close"].to_numpy(float),
        "volume": sub["volume"].to_numpy(float),
        "dividends": sub["dividends"].fillna(0.0).to_numpy(float),
        "splits": sub["splits"].fillna(0.0).to_numpy(float),
        "source": "yfinance",
    })
    out["ts"] = out["date"].dt.tz_localize("UTC")
    return out.loc[:, list(PRICE_COLUMNS)]


def dividend_adjustment_factor(df: pd.DataFrame) -> pd.Series:
    """CRSP-style cumulative dividend factor from ``dividends`` and split-adjusted
    ``close`` for ONE symbol sorted by date: on each ex-date the prior close is
    multiplied by ``1 - div / prev_close``; factor is the backward cumulative product
    (1.0 on the last row). ``close * factor`` reproduces Yahoo's ``adj_close``."""
    d = df.sort_values("date")
    prev = d["close"].shift(1)
    m = (1.0 - d["dividends"] / prev).where(d["dividends"] > 0, 1.0).fillna(1.0)
    # factor applies to rows strictly before each ex-date
    f = m[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    return pd.Series(f.to_numpy(), index=d.index)


def build_membership(prices: pd.DataFrame, *, file_date: pd.Timestamp,
                     current_symbols: Iterable[str], exchange_of: Mapping[str, str] | None = None,
                     gap_days: int = 60, source: str = "yfinance") -> pd.DataFrame:
    """Per symbol: ``first_date, last_date, n_days, delisted_flag, exchange``.

    ``delisted_flag`` = last bar more than ``gap_days`` *trading* days before
    ``file_date`` (trading calendar = union of dates in ``prices``) AND the symbol is
    not in ``current_symbols``. Pure counts / dates — nothing about values."""
    cur = set(current_symbols)
    g = prices.groupby("symbol")["date"]
    m = pd.DataFrame({"first_date": g.min(), "last_date": g.max(), "n_days": g.size()})
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    cal = cal[cal <= pd.Timestamp(file_date)]
    pos_file = len(cal)
    pos_last = cal.searchsorted(m["last_date"].to_numpy(), side="right")
    m["trading_days_since_last"] = pos_file - pos_last
    m["in_current_list"] = m.index.isin(cur)
    m["delisted_flag"] = (m["trading_days_since_last"] > gap_days) & ~m["in_current_list"]
    m["exchange"] = m.index.map(lambda s: (exchange_of or {}).get(s, ""))
    m["source"] = source
    return m.reset_index().rename(columns={"index": "symbol"})


# ---------------------------------------------------------------------------
# 3. SEC: frames universe, companyfacts (num), submissions (sub, 8-K, tickers)
# ---------------------------------------------------------------------------
def frame_periods(first_year: int, last_year: int) -> list[str]:
    return [f"CY{y}Q{q}I" for y in range(first_year, last_year + 1) for q in range(1, 5)]


def parse_frame(payload: Mapping[str, Any], period: str) -> pd.DataFrame:
    rows = payload.get("data") or []
    return pd.DataFrame({
        "cik": [int(r["cik"]) for r in rows],
        "entity_name": [r.get("entityName", "") for r in rows],
        "period": period,
    })


def _sane_dates(s: pd.Series) -> pd.Series:
    """Parse ISO dates; anything unparseable or outside 1900-2100 (filer typos such as
    year 0201 / 9999 exist in companyfacts) becomes NaT."""
    d = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d")
    return d.where((d >= pd.Timestamp("1900-01-01")) & (d < pd.Timestamp("2100-01-01")))


def _qtrs(start: pd.Series, end: pd.Series) -> pd.Series:
    """FSDS ``qtrs``: 0 for instants, else duration in quarters (rounded)."""
    days = (end - start).dt.days
    q = (days / 91.31).round()
    return q.where(start.notna(), 0).fillna(0).astype("int32")


NUM_COLUMNS: Final[tuple[str, ...]] = (
    "adsh", "cik", "taxonomy", "tag", "uom", "value", "ddate", "start", "qtrs", "fy", "fp",
    "form", "filed", "frame",
)


def parse_companyfacts(payload: Mapping[str, Any]) -> pd.DataFrame:
    """XBRL ``companyfacts`` JSON -> FSDS-``num``-like long frame (:data:`NUM_COLUMNS`).

    One row per (fact, unit, reporting filing). ``ddate`` = period end, ``start`` =
    period start (NaT for instants), ``filed`` = filing date of ``adsh``; ``frame`` is
    SEC's calendar-frame label (set on one row per period only). ``coreg`` and
    dimensional facts are not in companyfacts (FSDS ``num`` has them)."""
    cik = int(payload.get("cik") or 0)
    cols: dict[str, list[Any]] = {c: [] for c in NUM_COLUMNS if c != "cik" and c != "qtrs"}
    for tax, tags in (payload.get("facts") or {}).items():
        for tag, body in tags.items():
            for uom, vals in (body.get("units") or {}).items():
                for v in vals:
                    cols["adsh"].append(v.get("accn"))
                    cols["taxonomy"].append(tax)
                    cols["tag"].append(tag)
                    cols["uom"].append(uom)
                    cols["value"].append(v.get("val"))
                    cols["ddate"].append(v.get("end"))
                    cols["start"].append(v.get("start"))
                    cols["fy"].append(v.get("fy"))
                    cols["fp"].append(v.get("fp"))
                    cols["form"].append(v.get("form"))
                    cols["filed"].append(v.get("filed"))
                    cols["frame"].append(v.get("frame"))
    df = pd.DataFrame(cols)
    if df.empty:
        out = pd.DataFrame(columns=list(NUM_COLUMNS))
        return out
    df["cik"] = cik
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for c in ("ddate", "start", "filed"):
        df[c] = _sane_dates(df[c])
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int32")
    df["qtrs"] = _qtrs(df["start"], df["ddate"])
    for c in ("adsh", "taxonomy", "tag", "uom", "fp", "form", "frame"):
        df[c] = df[c].astype("string")
    return df.loc[:, list(NUM_COLUMNS)]


def _edgar_block(sub: Mapping[str, Any]) -> Mapping[str, Any]:
    return sub["filings"]["recent"] if "filings" in sub else sub


FILING_COLUMNS: Final[tuple[str, ...]] = (
    "adsh", "cik", "form", "filed", "accepted", "period", "items", "is_xbrl",
)


def parse_submission_filings(subs: Iterable[Mapping[str, Any]], cik: int) -> pd.DataFrame:
    """All filings from a CIK's submissions JSON (main + paginated) -> one row per
    accession (deduplicated). ``accepted`` = ``acceptanceDateTime`` read as UTC."""
    frames = []
    for s in subs:
        b = _edgar_block(s)
        n = len(b.get("accessionNumber", []))
        if n == 0:
            continue
        frames.append(pd.DataFrame({
            "adsh": b["accessionNumber"],
            "form": b.get("form", [""] * n),
            "filed": b.get("filingDate", [None] * n),
            "accepted": b.get("acceptanceDateTime", [None] * n),
            "period": b.get("reportDate", [None] * n),
            "items": b.get("items", [""] * n),
            "is_xbrl": b.get("isXBRL", [0] * n),
        }))
    if not frames:
        return pd.DataFrame(columns=list(FILING_COLUMNS))
    df = pd.concat(frames, ignore_index=True).drop_duplicates("adsh")
    df["cik"] = int(cik)
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["accepted"] = pd.to_datetime(df["accepted"], errors="coerce", utc=True, format="ISO8601")
    df["period"] = pd.to_datetime(df["period"].replace("", None), errors="coerce")
    df["items"] = df["items"].fillna("").astype(str)
    df["is_xbrl"] = pd.to_numeric(df["is_xbrl"], errors="coerce").fillna(0).astype("int8")
    return df.loc[:, list(FILING_COLUMNS)].reset_index(drop=True)


def filer_profile(main: Mapping[str, Any]) -> dict[str, Any]:
    """Static fields from a submissions main JSON (current state as served)."""
    return {
        "cik": int(main.get("cik") or 0),
        "name": main.get("name") or "",
        "sic": str(main.get("sic") or ""),
        "sic_description": main.get("sicDescription") or "",
        "entity_type": main.get("entityType") or "",
        "category": main.get("category") or "",
        "state_of_incorporation": main.get("stateOfIncorporation") or "",
        "fiscal_year_end": main.get("fiscalYearEnd") or "",
        "tickers": ",".join(str(t) for t in (main.get("tickers") or [])),
        "exchanges": ",".join(str(e or "") for e in (main.get("exchanges") or [])),
        "former_names": ";".join(f"{f.get('name')}|{f.get('from', '')[:10]}|{f.get('to', '')[:10]}"
                                 for f in (main.get("formerNames") or [])),
    }


def availability_ts(accepted: pd.Series, filed: pd.Series) -> pd.Series:
    """PIT availability: acceptance instant (UTC) if known, else filing date 00:00 UTC."""
    f = pd.to_datetime(filed).dt.tz_localize("UTC") if pd.to_datetime(filed).dt.tz is None \
        else pd.to_datetime(filed)
    a = pd.to_datetime(accepted, utc=True)
    return a.where(a.notna(), f)


def build_sub_rows(filings: pd.DataFrame, profile: Mapping[str, Any],
                   fyfp: pd.DataFrame | None = None) -> pd.DataFrame:
    """FSDS-``sub``-like rows: periodic-report filings of one CIK, with name/sic from the
    profile and ``fy``/``fp`` taken from the filing's own XBRL facts (``fyfp``: adsh ->
    fy, fp; mode over the filing's facts)."""
    s = filings[filings["form"].isin(PERIODIC_FORMS)].copy()
    s["name"] = profile.get("name", "")
    s["sic"] = profile.get("sic", "")
    if fyfp is not None and len(fyfp):
        s = s.merge(fyfp, on="adsh", how="left")
    else:
        s["fy"] = pd.array([pd.NA] * len(s), dtype="Int32")
        s["fp"] = pd.array([pd.NA] * len(s), dtype="string")
    s["ts"] = availability_ts(s["accepted"], s["filed"])
    return s.loc[:, ["adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed",
                     "accepted", "is_xbrl", "ts"]].reset_index(drop=True)


def fyfp_by_adsh(num: pd.DataFrame) -> pd.DataFrame:
    """Most common (fy, fp) reported in each accession's facts."""
    if num.empty:
        return pd.DataFrame({"adsh": pd.Series(dtype="string"), "fy": pd.Series(dtype="Int32"),
                             "fp": pd.Series(dtype="string")})
    g = num.dropna(subset=["fy"]).groupby(["adsh", "fy", "fp"], dropna=False).size()
    g = g.reset_index(name="n").sort_values(["adsh", "n"], ascending=[True, False])
    return g.drop_duplicates("adsh").loc[:, ["adsh", "fy", "fp"]].reset_index(drop=True)


def attach_num_ts(num: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """Add ``accepted`` and ``ts`` (PIT availability) to companyfacts rows by accession."""
    acc = filings.loc[:, ["adsh", "accepted"]].drop_duplicates("adsh")
    out = num.merge(acc, on="adsh", how="left")
    out["ts"] = availability_ts(out["accepted"], out["filed"])
    return out


def fy_quarter_label(ts: pd.Series) -> pd.Series:
    """FSDS data-set quarter label ``YYYYqN`` = calendar quarter of availability."""
    t = pd.to_datetime(ts, utc=True)
    return t.dt.year.astype(str) + "q" + t.dt.quarter.astype(str)


def pit_filter(df: pd.DataFrame, as_of: pd.Timestamp | str) -> pd.DataFrame:
    """Rows knowable at ``as_of`` (``ts <= as_of``). Use for fundamentals: the most
    recent value of a (cik, tag, ddate, qtrs) as of a date is the last row by ``ts``."""
    a = pd.Timestamp(as_of)
    a = a.tz_localize("UTC") if a.tzinfo is None else a.tz_convert("UTC")
    return df.loc[pd.to_datetime(df["ts"], utc=True) <= a]


def latest_as_of(num: pd.DataFrame, as_of: pd.Timestamp | str,
                 keys: Sequence[str] = ("cik", "tag", "uom", "ddate", "qtrs")) -> pd.DataFrame:
    """Point-in-time snapshot: for each key the value from the latest filing with
    ``ts <= as_of`` (later amendments override earlier ones only once filed)."""
    p = pit_filter(num, as_of)
    return p.sort_values("ts").drop_duplicates(list(keys), keep="last").reset_index(drop=True)


def name_key(name: str) -> str:
    """Normalised company name for fuzzy CIK<->listing matching."""
    s = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())
    s = re.sub(r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|lp|holdings?"
               r"|group|the|common stock|class [a-z]|ordinary shares|shares|n v|s a|ag|se)\b",
               " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_cik_ticker_map(profiles: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Ticker <-> CIK from the submissions JSON ``tickers`` lists (exact, keyless),
    joined to today's Nasdaq-Trader listing; listing symbols that no CIK claims get a
    fuzzy name match (``match = 'name_fuzzy'``) when the normalised names agree
    exactly and uniquely."""
    rows = []
    for _, p in profiles.iterrows():
        tks = [t for t in str(p["tickers"]).split(",") if t]
        exs = str(p["exchanges"]).split(",")
        for i, t in enumerate(tks):
            rows.append({"ticker": t.upper(), "cik": int(p["cik"]), "sec_name": p["name"],
                         "sec_exchange": exs[i] if i < len(exs) else "", "match": "sec_tickers"})
    m = pd.DataFrame(rows, columns=["ticker", "cik", "sec_name", "sec_exchange", "match"])
    m = m.drop_duplicates(["ticker", "cik"])
    m["norm"] = m["ticker"].str.replace(r"[.\-/]", "", regex=True)
    u = universe.loc[:, ["symbol", "security_name", "exchange", "security_type"]].copy()
    u["norm"] = u["symbol"].str.upper().str.replace(r"[.\-/$]", "", regex=True)
    out = u.merge(m.drop(columns=["ticker"]), on="norm", how="left")
    miss = out["cik"].isna()
    if miss.any():
        nk = profiles.assign(k=profiles["name"].map(name_key))
        nk = nk[nk["k"] != ""]
        uniq = nk.groupby("k")["cik"].agg(lambda x: list(set(x)))
        uniq = uniq[uniq.map(len) == 1].map(lambda x: x[0])
        names = nk.drop_duplicates("cik").set_index("cik")["name"]
        keys = out.loc[miss, "security_name"].map(name_key)
        hit = keys.map(uniq)
        out.loc[miss, "cik"] = hit
        out.loc[miss & hit.notna(), "match"] = "name_fuzzy"
        out.loc[miss & hit.notna(), "sec_name"] = hit.dropna().map(names)
    out["match"] = out["match"].fillna("none")
    out["cik"] = out["cik"].astype("Int64")
    return out.drop(columns=["norm"]).reset_index(drop=True)


def build_filer_membership(filings: pd.DataFrame, profiles: pd.DataFrame,
                           as_of: pd.Timestamp, stale_days: int = 550) -> pd.DataFrame:
    """Per CIK: first/last periodic-report filing date, counts, and ``inactive_flag`` =
    no 10-K/10-Q/20-F/40-F filed in the ``stale_days`` before ``as_of`` (a free proxy
    for delisting / deregistration / acquisition; not a delisting date)."""
    p = filings[filings["form"].isin(PERIODIC_FORMS)]
    g = p.groupby("cik")
    m = pd.DataFrame({
        "first_periodic_filed": g["filed"].min(),
        "last_periodic_filed": g["filed"].max(),
        "n_10k": g["form"].agg(lambda s: int(s.str.startswith("10-K").sum())),
        "n_10q": g["form"].agg(lambda s: int(s.str.startswith("10-Q").sum())),
        "n_foreign_annual": g["form"].agg(lambda s: int(s.str.match(r"^(20|40)-F").sum())),
    })
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=stale_days)
    m["inactive_flag"] = m["last_periodic_filed"] < cutoff
    prof = profiles.set_index("cik")[["name", "sic", "tickers", "exchanges", "entity_type"]]
    return m.join(prof, how="left").reset_index()


# ---------------------------------------------------------------------------
# 3b. SEC bulk data sets on www.sec.gov (declared-contact User-Agent, 2026-10-11)
# ---------------------------------------------------------------------------
SEC_WWW: Final[str] = "https://www.sec.gov"
SEC_COMPANY_TICKERS_URL: Final[str] = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_TICKERS_EXCHANGE_URL: Final[str] = (
    "https://www.sec.gov/files/company_tickers_exchange.json")
SEC_NOTES_LISTING_URL: Final[str] = (
    "https://www.sec.gov/dera/data/financial-statement-and-notes-data-set")
SEC_INSIDER_LISTING_URL: Final[str] = "https://www.sec.gov/dera/data/form-345"
SEC_FULL_INDEX_URL: Final[str] = (
    "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.gz")
SEC_BULK_ROOT: Final[Path] = RAW_ROOT / "sec_bulk"

_SIZE_UNITS: Final[dict[str, float]] = {"B": 1.0, "KB": 1e3, "MB": 1e6, "GB": 1e9}


def parse_size_text(text: str) -> int | None:
    """``'53.48 MB'`` -> 53_480_000 (SEC listing pages use decimal units)."""
    m = re.fullmatch(r"\s*([\d.,]+)\s*([KMG]?B)\s*", text or "", re.I)
    if not m:
        return None
    return int(round(float(m.group(1).replace(",", "")) * _SIZE_UNITS[m.group(2).upper()]))


def dera_period(file_name: str) -> tuple[str, str] | None:
    """``2010q1_notes_1.zip`` -> (``'2010q1'``, ``'quarter'``); ``2025_10_notes.zip`` ->
    (``'2025m10'``, ``'month'``); ``2006q1_form345.zip`` -> (``'2006q1'``, ``'quarter'``)."""
    m = re.match(r"(\d{4})q([1-4])_", file_name)
    if m:
        return f"{m.group(1)}q{m.group(2)}", "quarter"
    m = re.match(r"(\d{4})_(\d{2})_", file_name)
    if m:
        return f"{m.group(1)}m{m.group(2)}", "month"
    return None


def period_bounds(period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Calendar span ``[start, end)`` of a data-set period label (``YYYYqN`` / ``YYYYmMM``)."""
    if "q" in period:
        y, q = period.split("q")
        s = pd.Timestamp(f"{int(y)}-{3 * (int(q) - 1) + 1:02d}-01")
        return s, s + pd.DateOffset(months=3)
    y, mo = period.split("m")
    s = pd.Timestamp(f"{int(y)}-{int(mo):02d}-01")
    return s, s + pd.DateOffset(months=1)


def parse_dera_listing(html: str, base: str = SEC_WWW) -> pd.DataFrame:
    """A DERA data-set landing page (Financial Statement and Notes, Insider Transactions)
    -> one row per zip: ``url, file, period, period_kind, size_listed`` (bytes, from the
    page's size column; exact size comes from a HEAD), sorted by period start."""
    rows = []
    for tr in re.findall(r"<tr.*?</tr>", html, re.S | re.I):
        m = re.search(r'href="([^"]+\.zip)"', tr, re.I)
        if not m:
            continue
        href = m.group(1)
        cells = [re.sub(r"<[^>]+>", "", c).strip()
                 for c in re.findall(r"<td.*?</td>", tr, re.S | re.I)]
        fname = href.rsplit("/", 1)[-1]
        per = dera_period(fname)
        if per is None:
            continue
        rows.append({"url": href if href.startswith("http") else base + href, "file": fname,
                     "period": per[0], "period_kind": per[1],
                     "size_listed": next((v for v in map(parse_size_text, cells)
                                          if v is not None), None)})
    df = pd.DataFrame(rows, columns=["url", "file", "period", "period_kind", "size_listed"])
    df = df.drop_duplicates("url")
    df["size_listed"] = df["size_listed"].astype("Int64")
    df["_s"] = df["period"].map(lambda p: period_bounds(p)[0])
    return df.sort_values(["_s", "file"]).drop(columns="_s").reset_index(drop=True)


def full_index_quarters(first: str, last: pd.Timestamp | str) -> list[str]:
    """Quarter labels ``YYYYqN`` from ``first`` through the quarter containing ``last``."""
    y, q = (int(v) for v in first.split("q"))
    t = pd.Timestamp(last)
    out = []
    while (y, q) <= (t.year, t.quarter):
        out.append(f"{y}q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def full_index_url(quarter: str) -> str:
    y, q = quarter.split("q")
    return SEC_FULL_INDEX_URL.format(year=int(y), q=int(q))


# --- company_tickers --------------------------------------------------------
def parse_company_tickers(payload: Mapping[str, Any]) -> pd.DataFrame:
    """``company_tickers.json`` (``{"0": {"cik_str", "ticker", "title"}, ...}``) ->
    ``cik, ticker, title``. Current-only: SEC lists today's tickers of today's filers."""
    rows = [{"cik": int(v["cik_str"]), "ticker": str(v["ticker"]).upper(),
             "title": str(v.get("title") or "")} for v in payload.values()]
    return pd.DataFrame(rows, columns=["cik", "ticker", "title"])


def parse_company_tickers_exchange(payload: Mapping[str, Any]) -> pd.DataFrame:
    """``company_tickers_exchange.json`` (``{"fields": [...], "data": [[...]]}``) ->
    ``cik, name, ticker, exchange``."""
    f = [str(c) for c in payload["fields"]]
    df = pd.DataFrame(payload["data"], columns=f)
    df = df.rename(columns={"name": "name", "exchange": "exchange"})
    df["cik"] = df["cik"].astype(int)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["exchange"] = df["exchange"].fillna("").astype(str)
    return df.loc[:, ["cik", "name", "ticker", "exchange"]]


def build_company_tickers(tick: pd.DataFrame, tick_ex: pd.DataFrame,
                          fetched_utc: str) -> pd.DataFrame:
    """One row per (cik, ticker): title from ``company_tickers``, exchange from the
    exchange file (outer join; either file may list a ticker the other lacks)."""
    out = tick.merge(tick_ex, on=["cik", "ticker"], how="outer")
    out["title"] = out["title"].fillna(out["name"])
    out["exchange"] = out["exchange"].fillna("")
    out["fetched_utc"] = fetched_utc
    return out.loc[:, ["cik", "ticker", "title", "exchange", "fetched_utc"]] \
        .drop_duplicates(["cik", "ticker"]).sort_values(["cik", "ticker"]).reset_index(drop=True)


# --- EDGAR full-index -------------------------------------------------------
def parse_master_index(text: str) -> pd.DataFrame:
    """EDGAR ``master.idx`` (``CIK|Company Name|Form Type|Date Filed|Filename`` after a
    dashed header) -> ``cik, company, form, filed, filename, adsh``. ``filed`` is a date
    only (no acceptance time)."""
    lines = text.splitlines()
    start = next((i + 1 for i, ln in enumerate(lines) if ln.startswith("-----")), 0)
    rows = [ln.split("|") for ln in lines[start:] if ln.count("|") == 4]
    df = pd.DataFrame(rows, columns=["cik", "company", "form", "filed", "filename"])
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce", format="%Y-%m-%d")
    df["adsh"] = df["filename"].str.extract(r"(\d{10}-\d{2}-\d{6})", expand=False)
    return df


# --- Financial Statement and Notes: customer concentration links ------------
CONCENTRATION_TAGS: Final[frozenset[str]] = frozenset(
    {"ConcentrationRiskPercentage1", "ConcentrationRiskPercentage"})
CUSTOMER_AMOUNT_TAGS: Final[frozenset[str]] = frozenset({
    "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet", "SalesRevenueServicesNet",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax", "RevenueNet",
    "AccountsReceivableNetCurrent", "AccountsReceivableGrossCurrent",
})
# Axis local names arrive with the "Axis" suffix stripped (FSN ``dim.segments``).
_CUSTOMER_AXIS: Final[re.Pattern[str]] = re.compile(r"customer|client", re.I)
_NON_ENTITY_AXIS: Final[re.Pattern[str]] = re.compile(
    r"type|class|categor|group|industry|sector|channel|geograph|region|market|size|"
    r"relationship|segment|program|contract|tier|concentration|location|country|domicile|"
    r"state|payor|payer|benchmark", re.I)
_GENERIC_TOKENS: Final[frozenset[str]] = frozenset(
    "customer customers client clients one two three four five six seven eight nine ten "
    "eleven twelve first second third fourth fifth a b c d e f g h i j k l m n o p q r s t "
    "u v w x y z 1 2 3 4 5 6 7 8 9 10 11 12 no number top largest large larger major "
    "significant principal other others all remaining single individual unnamed unrelated "
    "key primary and of the member domain group concentration risk percent percentage "
    "entity entities company companies distributor distributors wholesaler wholesalers "
    "retailer retailers end user users vendor vendors supplier suppliers another external "
    "foreign international domestic commercial residential industrial consumer consumers "
    "wholesale retail intersegment related party parties affiliated affiliate affiliates "
    "third purchaser purchasers buyer buyers partner partners counterparty counterparties "
    "total combined aggregate various multiple several each any account accounts "
    "national regional local institutional private public non us usa sales revenue "
    "revenues receivable receivables oem oems contract contracts".split())
_LETTER_SUFFIX: Final[re.Pattern[str]] = re.compile(
    r"^(customer|client|vendor|supplier|distributor|company)s? ?[a-z0-9]{1,2}$", re.I)
_VENDOR: Final[re.Pattern[str]] = re.compile(r"vendor|supplier", re.I)
_GOVT: Final[re.Pattern[str]] = re.compile(
    r"\b(government|federal|navy|army|air force|department|dod|nasa|medicare|medicaid|"
    r"state of|county|city of|municipal|ministry|agency|u ?s ?g|usgovernment|"
    r"veterans|military|defense)\b", re.I)


def parse_segments(seg: str) -> list[tuple[str, str]]:
    """``'A=X;B=Y;'`` -> ``[('A','X'), ('B','Y')]`` (FSN ``dim.segments``)."""
    out = []
    for part in (seg or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out.append((k.strip(), v.strip()))
    return out


def customer_axis_member(seg: str) -> tuple[str, str] | None:
    """First (axis, member) whose axis names an individual customer (``MajorCustomers``,
    ``Customer``, ``CustomerName``, ``SignificantCustomers`` ...), excluding axes that
    classify customers by type / class / industry / region / segment."""
    for ax, mem in parse_segments(seg):
        if _CUSTOMER_AXIS.search(ax) and not _NON_ENTITY_AXIS.search(ax):
            return ax, mem
    return None


def split_camel(member: str) -> str:
    """XBRL member local name -> words: ``'WalMartStoresInc'`` -> ``'Wal Mart Stores
    Inc'``, ``'USNavy'`` -> ``'US Navy'``, ``'ATAndTInc'`` -> ``'AT And T Inc'``."""
    s = re.sub(r"(Member|Domain)$", "", member or "")
    s = s.replace("_", " ")
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    return re.sub(r"\s+", " ", s).strip()


def is_anonymous_customer(clean: str) -> bool:
    """``'Customer A'``, ``'Two Customers'``, ``'Largest Customer'`` -> True."""
    toks = [t for t in re.split(r"[^a-z0-9]+", (clean or "").lower()) if t]
    if _LETTER_SUFFIX.match((clean or "").strip()):
        return True  # 'Customera', 'Vendor B', 'Customer 12'
    return not toks or all(t in _GENERIC_TOKENS for t in toks)


def is_government_customer(clean: str) -> bool:
    return bool(_GOVT.search(clean or ""))


_BENCH_REVENUE: Final[re.Pattern[str]] = re.compile(r"revenue|sales", re.I)


def customer_dims(dim: pd.DataFrame) -> pd.DataFrame:
    """FSN ``dim`` (``dimhash, segments``) -> the dimension hashes carrying a customer
    axis: ``dimh, segments, n_axes, customer_axis, customer_member, benchmark,
    risk_type``."""
    rows = []
    for h, seg in zip(dim["dimhash"], dim["segments"].fillna(""), strict=True):
        cm = customer_axis_member(seg)
        if cm is None:
            continue
        kv = dict(parse_segments(seg))
        rows.append({"dimh": h, "segments": seg, "n_axes": len(kv), "customer_axis": cm[0],
                     "customer_member": cm[1],
                     "benchmark": kv.get("ConcentrationRiskByBenchmark", ""),
                     "risk_type": kv.get("ConcentrationRiskByType", "")})
    return pd.DataFrame(rows, columns=["dimh", "segments", "n_axes", "customer_axis",
                                       "customer_member", "benchmark", "risk_type"])


def _fsn_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype("string").str.slice(0, 8), errors="coerce", format="%Y%m%d")


def fsn_accepted_utc(accepted: pd.Series) -> pd.Series:
    """FSN ``sub.accepted`` (``'2009-04-15 16:44:00.0'``, EDGAR wall clock = US/Eastern)
    -> UTC. Checked against the submissions-JSON acceptance (true UTC) in the build."""
    t = pd.to_datetime(accepted.astype("string").str.slice(0, 19), errors="coerce",
                       format="%Y-%m-%d %H:%M:%S")
    return t.dt.tz_localize("America/New_York", ambiguous="NaT",
                            nonexistent="shift_forward").dt.tz_convert("UTC")


EDGE_COLUMNS: Final[tuple[str, ...]] = (
    "supplier_cik", "supplier_name", "supplier_sic", "coreg", "customer_axis",
    "customer_member", "customer_name", "anonymous", "government", "vendor_hint", "benchmark",
    "is_revenue_benchmark", "risk_type", "tag", "uom", "value", "pct", "pct_scale_suspect",
    "ddate", "qtrs", "fy", "fp", "form", "adsh", "filed", "accepted_fsn", "segments",
    "n_axes", "dataset",
)


def customer_edges_from_tables(num: pd.DataFrame, dims: pd.DataFrame, sub: pd.DataFrame,
                               dataset: str) -> pd.DataFrame:
    """Join filtered FSN ``num`` rows (already restricted to customer-axis ``dimh``) to
    their dimension and filing -> one edge row per (fact, filing) (:data:`EDGE_COLUMNS`).

    ``value`` is as reported; ``pct`` = percent of the benchmark (``value * 100`` for a
    ``pure`` ratio <= 1.5; above that the filer probably typed a percent, kept as is and
    flagged ``pct_scale_suspect``). Amount tags (revenue / receivables by customer) carry
    ``pct`` NaN. ``vendor_hint``: the member names a vendor / supplier (filers sometimes
    hang supplier concentration on a customer axis; the counterparty is then upstream)."""
    keep = num[num["tag"].isin(CONCENTRATION_TAGS | CUSTOMER_AMOUNT_TAGS)]
    e = keep.merge(dims, on="dimh", how="inner")
    s = sub.loc[:, ["adsh", "cik", "name", "sic", "form", "fy", "fp", "filed", "accepted"]] \
        .rename(columns={"cik": "supplier_cik", "name": "supplier_name", "sic": "supplier_sic",
                         "accepted": "accepted_fsn"})
    e = e.merge(s, on="adsh", how="left")
    e["value"] = pd.to_numeric(e["value"], errors="coerce")
    is_pct = e["tag"].isin(CONCENTRATION_TAGS)
    big = is_pct & (e["value"].abs() > 1.5)
    e["pct"] = np.where(is_pct, np.where(big, e["value"], e["value"] * 100.0), np.nan)
    e["pct_scale_suspect"] = big
    e["customer_name"] = e["customer_member"].map(split_camel)
    e["anonymous"] = e["customer_name"].map(is_anonymous_customer)
    e["government"] = e["customer_name"].map(is_government_customer)
    e["vendor_hint"] = e["customer_member"].fillna("").str.contains(_VENDOR)
    e["is_revenue_benchmark"] = e["benchmark"].fillna("").str.contains(_BENCH_REVENUE)
    e["ddate"] = _fsn_date(e["ddate"])
    e["filed"] = _fsn_date(e["filed"])
    e["accepted_fsn"] = fsn_accepted_utc(e["accepted_fsn"])
    e["qtrs"] = pd.to_numeric(e["qtrs"], errors="coerce").astype("Int16")
    e["fy"] = pd.to_numeric(e["fy"], errors="coerce").astype("Int32")
    e["supplier_cik"] = pd.to_numeric(e["supplier_cik"], errors="coerce").astype("Int64")
    e["n_axes"] = e["n_axes"].astype("Int16")
    e["coreg"] = e["coreg"].fillna("")
    e["dataset"] = dataset
    for c in ("supplier_name", "supplier_sic", "customer_axis", "customer_member",
              "customer_name", "benchmark", "risk_type", "tag", "uom", "fp", "form", "adsh",
              "segments", "coreg", "dataset"):
        e[c] = e[c].astype("string")
    return e.loc[:, list(EDGE_COLUMNS)].reset_index(drop=True)


_FSN_NUM_COLS: Final[tuple[str, ...]] = ("adsh", "tag", "ddate", "qtrs", "uom", "dimh", "value",
                                         "coreg")
_FSN_SUB_COLS: Final[tuple[str, ...]] = ("adsh", "cik", "name", "sic", "form", "fy", "fp", "filed",
                                         "accepted")


def _read_fsn_member(zf: Any, name: str, columns: Sequence[str],
                     dimh_keep: set[str] | None = None,
                     regex: tuple[str, str] | None = None) -> pd.DataFrame:
    """Stream one tab-separated FSN member with pyarrow (string columns only), optionally
    keeping only rows whose ``dimh`` is in ``dimh_keep`` and/or whose column
    ``regex[0]`` matches pattern ``regex[1]``. Memory ~ one block at a time."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv

    member = next((n for n in zf.namelist() if n.rsplit("/", 1)[-1].lower() == name), None)
    if member is None:
        return pd.DataFrame(columns=list(columns))
    keep_arr = pa.array(sorted(dimh_keep)) if dimh_keep is not None else None
    out: list[Any] = []
    skipped: list[int] = []

    def _skip(_row: Any) -> str:
        skipped.append(1)
        return "skip"

    for enc in ("utf8", "latin-1"):
        out, skipped = [], []
        try:
            with zf.open(member) as fh:
                rd = pacsv.open_csv(
                    fh,
                    read_options=pacsv.ReadOptions(block_size=16 << 20, encoding=enc,
                                                   use_threads=False),
                    parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False,
                                                     invalid_row_handler=_skip),
                    convert_options=pacsv.ConvertOptions(
                        include_columns=list(columns), include_missing_columns=True,
                        column_types={c: pa.string() for c in columns}, strings_can_be_null=True))
                for batch in rd:
                    if keep_arr is not None:
                        batch = batch.filter(pc.is_in(batch.column("dimh"), value_set=keep_arr))
                    if regex is not None:
                        hit = pc.fill_null(pc.match_substring_regex(
                            batch.column(regex[0]), regex[1], ignore_case=True), False)
                        batch = batch.filter(hit)
                    if batch.num_rows:
                        out.append(batch)
            break
        except (pa.ArrowInvalid, UnicodeDecodeError):
            if enc == "latin-1":
                raise
    df = pa.Table.from_batches(out).to_pandas() if out else pd.DataFrame(columns=list(columns))
    df.attrs["skipped_rows"] = len(skipped)
    return df


def extract_customer_edges_from_notes_zip(
        path: Path | str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One Financial Statement and Notes zip -> (customer-axis edge rows
    (:func:`customer_edges_from_tables`), parse meta). Reads ``dim`` and ``sub`` whole
    (small) and streams ``num`` keeping only customer-axis dimension hashes. Meta counts
    the dimension hashes, customer-axis hashes, kept ``num`` rows and malformed (skipped)
    rows per member."""
    import zipfile

    path = Path(path)
    per = dera_period(path.name)
    label = per[0] if per else path.stem
    with zipfile.ZipFile(path) as zf:
        dim = _read_fsn_member(zf, "dim.tsv", ("dimhash", "segments"),
                               regex=("segments", "customer|client"))
        dims = customer_dims(dim)
        meta: dict[str, Any] = {"dataset": label, "file": path.name,
                                "n_dimhash_mentioning_customer": len(dim),
                                "n_customer_dimhash": len(dims),
                                "skipped_dim": dim.attrs.get("skipped_rows", 0)}
        sub = _read_fsn_member(zf, "sub.tsv", _FSN_SUB_COLS)
        if dims.empty:
            meta["_sub_accepted"] = pd.DataFrame({
                "adsh": sub["adsh"].astype("string"),
                "accepted_fsn": fsn_accepted_utc(sub["accepted"]),
                "form": sub["form"].astype("string")})
            return pd.DataFrame(columns=list(EDGE_COLUMNS)), meta
        num = _read_fsn_member(zf, "num.tsv", _FSN_NUM_COLS, dimh_keep=set(dims["dimh"]))
    meta.update(n_num_customer_rows=len(num), n_sub=len(sub),
                skipped_num=num.attrs.get("skipped_rows", 0),
                skipped_sub=sub.attrs.get("skipped_rows", 0))
    e = customer_edges_from_tables(num, dims, sub, label)
    meta["n_edges"] = len(e)
    # every filing's FSN acceptance (for the submissions-JSON stamp audit); popped by the caller
    meta["_sub_accepted"] = pd.DataFrame({"adsh": sub["adsh"].astype("string"),
                                          "accepted_fsn": fsn_accepted_utc(sub["accepted"]),
                                          "form": sub["form"].astype("string")})
    return e, meta


# --- conservative name -> CIK resolution ------------------------------------
_EXTRA_SUFFIX: Final[re.Pattern[str]] = re.compile(r"\b(the|and|of|de|du|del)\b")


def resolve_key(name: str) -> str:
    """:func:`name_key` plus ``&``/``and`` and ``.com`` folding (used for matching)."""
    s = (name or "").replace("&", " and ")
    s = re.sub(r"\.com\b", "com", s, flags=re.I)
    s = re.sub(r"\bincorporation\b", " ", s, flags=re.I)
    s = name_key(s)
    return re.sub(r"\s+", " ", _EXTRA_SUFFIX.sub(" ", s)).strip()


@dataclass
class NameIndex:
    """Lookup of normalised company names -> CIKs, built from
    ``(cik, name, source)`` rows (company_tickers titles, filer_profiles names and
    former names). ``has_ticker``: CIKs listed in company_tickers (tie-breaker only)."""

    names: pd.DataFrame
    has_ticker: frozenset[int] = frozenset()
    exact: dict[str, frozenset[int]] = field(default_factory=dict)
    compact: dict[str, frozenset[int]] = field(default_factory=dict)
    tokens: dict[int, frozenset[str]] = field(default_factory=dict)
    by_token: dict[str, set[int]] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)
    display: dict[int, str] = field(default_factory=dict)
    seqs: dict[int, set[tuple[str, ...]]] = field(default_factory=dict)

    @classmethod
    def build(cls, names: pd.DataFrame, has_ticker: Iterable[int] = (),
              current_names: pd.DataFrame | None = None) -> NameIndex:
        """``names``: every (cik, name) to match on (current + former). The
        ``prefix_listed`` rule uses only ``current_names`` (default: ``names``), so a
        listed firm's long-abandoned former name cannot capture a short customer name."""
        idx = cls(names=names, has_ticker=frozenset(int(c) for c in has_ticker))
        ex: dict[str, set[int]] = {}
        co: dict[str, set[int]] = {}
        toks: dict[int, set[str]] = {}
        for cik, nm in zip(names["cik"].astype(int), names["name"].astype(str), strict=True):
            k = resolve_key(nm)
            if not k:
                continue
            ex.setdefault(k, set()).add(cik)
            co.setdefault(k.replace(" ", ""), set()).add(cik)
            toks.setdefault(cik, set()).update(k.split())
            idx.display.setdefault(cik, nm)
        cur = names if current_names is None else current_names
        for cik, nm in zip(cur["cik"].astype(int), cur["name"].astype(str), strict=True):
            k = resolve_key(nm)
            if k:
                idx.seqs.setdefault(cik, set()).add(tuple(k.split()))
        idx.exact = {k: frozenset(v) for k, v in ex.items()}
        idx.compact = {k: frozenset(v) for k, v in co.items()}
        idx.tokens = {c: frozenset(t) for c, t in toks.items()}
        for c, ts in idx.tokens.items():
            for t in ts:
                idx.by_token.setdefault(t, set()).add(c)
        n = max(len(idx.tokens), 1)
        idx.idf = {t: float(np.log((n + 1) / (len(cs) + 0.5))) for t, cs in idx.by_token.items()}
        return idx

    def _pick(self, ciks: frozenset[int]) -> int | None:
        if len(ciks) == 1:
            return next(iter(ciks))
        listed = [c for c in ciks if c in self.has_ticker]
        return listed[0] if len(listed) == 1 else None

    def candidates(self, name: str) -> tuple[int, ...]:
        """CIKs sharing the normalised (or compact) key of ``name`` (ambiguity audit)."""
        k = resolve_key(name)
        hit = self.exact.get(k) or self.compact.get(k.replace(" ", "")) or frozenset()
        return tuple(sorted(hit))

    def resolve(self, name: str, threshold: float = 0.92,
                margin: float = 0.1, prefix_max_names: int = 20) -> tuple[int | None, str, float]:
        """-> (cik or None, method, score). Methods, tried in this order:

        * ``exact`` / ``exact_compact``: normalised names equal (spaces ignored for the
          compact key); unique CIK, or unique among CIKs with a current ticker;
        * ``prefix_listed``: the customer's tokens are the leading tokens of the CURRENT
          name of exactly one CIK with a current ticker (at most 2 extra tokens; first
          token in <= ``prefix_max_names`` filer names), e.g. ``Amazon`` -> ``AMAZON
          COM INC``; score = IDF share of the matched name covered;
        * ``token``: IDF-weighted Jaccard >= ``threshold`` and the runner-up at least
          ``margin`` lower; both sides need >= 2 tokens;
        * otherwise ``ambiguous`` (a tie at an accepting score) or ``none``."""
        k = resolve_key(name)
        if not k:
            return None, "none", 0.0
        for meth, table, key in (("exact", self.exact, k),
                                 ("exact_compact", self.compact, k.replace(" ", ""))):
            if key in table:
                c = self._pick(table[key])
                return (c, meth, 1.0) if c is not None else (None, "ambiguous", 1.0)
        seq = tuple(k.split())
        first = self.by_token.get(seq[0], set())
        if self.has_ticker and 0 < len(first) <= prefix_max_names:
            hits = {c for c in first if c in self.has_ticker
                    and any(s[:len(seq)] == seq and len(s) - len(seq) <= 2
                            for s in self.seqs.get(c, ()))}
            if len(hits) == 1:
                c = next(iter(hits))
                full = min((s for s in self.seqs[c] if s[:len(seq)] == seq), key=len)
                tot = sum(self.idf.get(t, 1.0) for t in full)
                part = sum(self.idf.get(t, 1.0) for t in seq)
                return c, "prefix_listed", round(part / tot if tot else 0.0, 4)
        a = frozenset(seq)
        if len(a) < 2:
            return None, "none", 0.0
        rare = sorted(a, key=lambda t: -self.idf.get(t, 99.0))[:2]
        cands: set[int] = set()
        for t in rare:
            cands |= self.by_token.get(t, set())
        scored = []
        for c in cands:
            b = self.tokens[c]
            if len(b) < 2:
                continue
            inter = sum(self.idf.get(t, 1.0) for t in a & b)
            union = sum(self.idf.get(t, 1.0) for t in a | b)
            scored.append((inter / union if union else 0.0, c))
        if not scored:
            return None, "none", 0.0
        scored.sort(reverse=True)
        best, c = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        if best >= threshold and best - second >= margin:
            return c, "token", round(best, 4)
        return None, ("ambiguous" if best >= threshold else "none"), round(best, 4)


def resolve_customers(edges: pd.DataFrame, index: NameIndex,
                      threshold: float = 0.92) -> pd.DataFrame:
    """Add ``customer_cik, match_method, match_score, match_name, match_candidates``
    (CIKs tied on the exact key, for ``ambiguous`` rows) to edge rows (resolved
    once per distinct cleaned name). ``prefix_listed`` hits are NOT written to
    ``customer_cik`` (method ``prefix_listed_candidate``; the CIK goes to
    ``match_candidates``). Anonymous members (``Customer A``) and
    government customers are not matched (methods ``anonymous`` / ``government``);
    a match to the supplier's own CIK is dropped (method ``self``)."""
    names = edges.loc[~edges["anonymous"] & ~edges["government"], "customer_name"].dropna().unique()
    res = {n: index.resolve(str(n), threshold=threshold) for n in names}
    out = edges.copy()
    cik = out["customer_name"].map(lambda n: res.get(n, (None, "", 0.0))[0])
    meth = out["customer_name"].map(lambda n: res.get(n, (None, "none", 0.0))[1])
    score = out["customer_name"].map(lambda n: res.get(n, (None, "", 0.0))[2])
    meth = meth.where(~out["government"], "government").where(~out["anonymous"], "anonymous")
    cik = cik.where(~out["government"] & ~out["anonymous"])
    self_hit = cik.notna() & (cik == out["supplier_cik"])
    meth = meth.where(~self_hit, "self")
    cik = cik.where(~self_hit)
    # prefix matches are suggestions only (audited precision ~75 %): kept out of
    # customer_cik, offered in match_candidates
    pref = meth == "prefix_listed"
    pref_cik = cik.where(pref)
    meth = meth.where(~pref, "prefix_listed_candidate")
    cik = cik.where(~pref)
    out["customer_cik"] = pd.array(cik, dtype="Int64")
    out["match_method"] = meth.astype("string")
    out["match_score"] = pd.to_numeric(score, errors="coerce").astype("float32")
    shown = out["customer_cik"].astype("Float64").fillna(pd.Series(pref_cik, index=out.index)
                                                         .astype("Float64"))
    out["match_name"] = shown.map(
        lambda c: index.display.get(int(c), "") if pd.notna(c) else "").astype("string")
    amb = {n: ",".join(str(c) for c in index.candidates(str(n)))
           for n in out.loc[out["match_method"] == "ambiguous", "customer_name"].dropna().unique()}
    cand = out["customer_name"].map(lambda n: amb.get(n, "")).where(
        out["match_method"] == "ambiguous", "")
    pc = pd.Series(pref_cik, index=out.index)
    cand = cand.where(~pref, pc.map(lambda c: str(int(c)) if pd.notna(c) else ""))
    out["match_candidates"] = cand.astype("string")
    return out


# --- Insider Transactions (Forms 3/4/5) -------------------------------------
SECTION16_FORMS: Final[frozenset[str]] = frozenset({"3", "4", "5", "3/A", "4/A", "5/A"})
_INS_SUB_COLS: Final[tuple[str, ...]] = ("ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT",
                                         "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERNAME",
                                         "ISSUERTRADINGSYMBOL")
_INS_OWN_COLS: Final[tuple[str, ...]] = ("ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
                                         "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE")
_INS_TX_COLS: Final[tuple[str, ...]] = ("ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "SECURITY_TITLE",
                                        "TRANS_DATE", "TRANS_FORM_TYPE", "TRANS_CODE",
                                        "EQUITY_SWAP_INVOLVED", "TRANS_TIMELINESS",
                                        "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                                        "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS",
                                        "DIRECT_INDIRECT_OWNERSHIP")


# Typo guards: a single open-market line above these is almost surely a mis-keyed price
# or share count (BRK-A trades near 1e6 per share at most). The 99.9th percentile of
# line dollars 2014-2016 was $8 bn, made of shell companies reporting 10 M shares at
# "$1,000" and similar test-like filings, so lines above $1 bn are flagged; the rare
# genuine block above $1 bn is lost from the dollar sums (still in share sums).
INSIDER_MAX_PRICE: Final[float] = 1.0e6
INSIDER_MAX_DOLLARS: Final[float] = 1.0e9


def parse_sec_date(s: pd.Series) -> pd.Series:
    """Insider data-set dates (``'31-JUL-2015'``; ISO also accepted) -> datetime64."""
    s = s.astype("string").str.strip()
    d = pd.to_datetime(s, errors="coerce", format="%d-%b-%Y")
    iso = d.isna() & s.notna()
    if iso.any():
        d[iso] = pd.to_datetime(s[iso], errors="coerce", format="%Y-%m-%d")
    return d.where((d >= pd.Timestamp("1990-01-01")) & (d < pd.Timestamp("2100-01-01")))


def read_insider_zip(path: Path | str) -> dict[str, pd.DataFrame]:
    """The three members needed from an Insider Transactions zip (string columns)."""
    import zipfile

    out = {}
    with zipfile.ZipFile(path) as zf:
        for key, name, cols in (("sub", "submission.tsv", _INS_SUB_COLS),
                                ("owner", "reportingowner.tsv", _INS_OWN_COLS),
                                ("tx", "nonderiv_trans.tsv", _INS_TX_COLS)):
            out[key] = _read_fsn_member(zf, name, cols)
    return out


def edgar_accept_after_hours_bound(filed: pd.Series) -> pd.Series:
    """Upper bound on acceptance for a filing known only by its filing DATE: 22:00
    US/Eastern of that date (EDGAR's last acceptance time), as UTC. Never earlier than
    the true acceptance, so a PIT key built on it cannot look ahead."""
    f = pd.to_datetime(filed) + pd.Timedelta(hours=22)
    return f.dt.tz_localize("America/New_York", ambiguous="NaT",
                            nonexistent="shift_forward").dt.tz_convert("UTC")


INSIDER_TX_COLUMNS: Final[tuple[str, ...]] = (
    "adsh", "issuer_cik", "issuer_name", "issuer_symbol", "doc_type", "period_of_report",
    "owner_id", "owner_ciks", "owner_names", "n_owners", "is_director", "is_officer",
    "is_ten_pct", "is_other", "officer_title", "security_title", "trans_date", "trans_code",
    "acq_disp", "shares", "price", "dollars", "dollars_suspect", "shares_after",
    "direct_indirect",
    "equity_swap", "late", "filed", "accepted", "ts_source", "ts", "dataset",
)


def build_insider_transactions(sub: pd.DataFrame, owner: pd.DataFrame, tx: pd.DataFrame,
                               accepted: pd.DataFrame | None = None,
                               dataset: str = "") -> pd.DataFrame:
    """Non-derivative open-market purchases (``P``) and sales (``S``) per filing
    (:data:`INSIDER_TX_COLUMNS`).

    ``ts`` (PIT availability) = EDGAR acceptance instant (UTC) when ``accepted``
    (``adsh, accepted``) knows the accession and it is not after 22:00 ET of the filing
    date, else that bound (:func:`edgar_accept_after_hours_bound`); ``ts_source`` says
    which (:func:`attach_insider_acceptance`). Owners of a
    joint filing are collapsed onto the filing (``owner_id`` = smallest owner CIK) so
    joint filers reporting one transaction are not double-counted. ``dollars`` =
    shares x price (NaN when the price is missing); ``dollars_suspect`` flags lines with
    price > :data:`INSIDER_MAX_PRICE` or dollars > :data:`INSIDER_MAX_DOLLARS`."""
    t = tx[tx["TRANS_CODE"].isin(["P", "S"])].copy()
    s = sub.rename(columns={"ACCESSION_NUMBER": "adsh", "FILING_DATE": "filed",
                            "PERIOD_OF_REPORT": "period_of_report", "DOCUMENT_TYPE": "doc_type",
                            "ISSUERCIK": "issuer_cik", "ISSUERNAME": "issuer_name",
                            "ISSUERTRADINGSYMBOL": "issuer_symbol"})
    o = owner.rename(columns={"ACCESSION_NUMBER": "adsh"}).copy()
    rel = o["RPTOWNER_RELATIONSHIP"].fillna("").str.lower()
    o["is_director"] = rel.str.contains("director")
    o["is_officer"] = rel.str.contains("officer")
    o["is_ten_pct"] = rel.str.contains("tenpercent|10%|ten percent", regex=True)
    o["is_other"] = rel.str.contains("other")
    o["ocik"] = pd.to_numeric(o["RPTOWNERCIK"], errors="coerce")
    g = o.sort_values("ocik").groupby("adsh", sort=False)
    own = pd.DataFrame({
        "owner_id": g["ocik"].min(),
        "owner_ciks": g["ocik"].agg(lambda x: ",".join(str(int(v)) for v in x.dropna())),
        "owner_names": g["RPTOWNERNAME"].agg(lambda x: "; ".join(str(v) for v in x.dropna())),
        "n_owners": g.size(),
        "is_director": g["is_director"].any(), "is_officer": g["is_officer"].any(),
        "is_ten_pct": g["is_ten_pct"].any(), "is_other": g["is_other"].any(),
        "officer_title": g["RPTOWNER_TITLE"].agg(lambda x: "; ".join(str(v) for v in x.dropna())),
    }).reset_index()
    t = t.rename(columns={"ACCESSION_NUMBER": "adsh", "SECURITY_TITLE": "security_title",
                          "TRANS_DATE": "trans_date", "TRANS_CODE": "trans_code",
                          "TRANS_ACQUIRED_DISP_CD": "acq_disp", "TRANS_SHARES": "shares",
                          "TRANS_PRICEPERSHARE": "price", "SHRS_OWND_FOLWNG_TRANS": "shares_after",
                          "DIRECT_INDIRECT_OWNERSHIP": "direct_indirect",
                          "EQUITY_SWAP_INVOLVED": "equity_swap", "TRANS_TIMELINESS": "late"})
    t = t.merge(s, on="adsh", how="left").merge(own, on="adsh", how="left")
    for c in ("shares", "price", "shares_after"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["dollars"] = t["shares"] * t["price"]
    t["dollars_suspect"] = ((t["price"] > INSIDER_MAX_PRICE)
                            | (t["dollars"] > INSIDER_MAX_DOLLARS)).fillna(False).astype(bool)
    t["trans_date"] = parse_sec_date(t["trans_date"])
    t["period_of_report"] = parse_sec_date(t["period_of_report"])
    t["filed"] = parse_sec_date(t["filed"])
    t["issuer_cik"] = pd.to_numeric(t["issuer_cik"], errors="coerce").astype("Int64")
    t["owner_id"] = pd.to_numeric(t["owner_id"], errors="coerce").astype("Int64")
    t["n_owners"] = pd.to_numeric(t["n_owners"], errors="coerce").fillna(0).astype("int16")
    for c in ("is_director", "is_officer", "is_ten_pct", "is_other"):
        t[c] = t[c].fillna(False).astype(bool)
    t["equity_swap"] = t["equity_swap"].fillna("").isin(["1", "true", "True"])
    t["late"] = t["late"].fillna("").str.upper().eq("L")
    t["dataset"] = dataset
    for c in ("adsh", "issuer_name", "issuer_symbol", "doc_type", "owner_ciks", "owner_names",
              "officer_title", "security_title", "trans_code", "acq_disp", "direct_indirect",
              "dataset"):
        t[c] = t[c].astype("string")
    t = attach_insider_acceptance(t, accepted)
    return t.loc[:, list(INSIDER_TX_COLUMNS)].reset_index(drop=True)


def attach_insider_acceptance(t: pd.DataFrame, accepted: pd.DataFrame | None) -> pd.DataFrame:
    """Set ``accepted`` / ``ts_source`` / ``ts`` on insider rows from an ``adsh,
    accepted`` (UTC) map. ``ts`` = the earlier of two upper bounds on the true
    acceptance: the submissions-JSON stamp (never early, but late by the ET UTC offset on
    a minority of filings -- see ``reports/research/SEC_BULK_UNLOCK_2026-10-11.md``) and
    22:00 ET of the filing date; unmatched rows get the bound."""
    t = t.drop(columns=[c for c in ("accepted", "ts_source", "ts") if c in t.columns])
    if accepted is not None and len(accepted):
        acc = accepted.loc[:, ["adsh", "accepted"]].drop_duplicates("adsh")
        acc = acc.assign(adsh=acc["adsh"].astype("string"))
        t = t.merge(acc, on="adsh", how="left")
        t["accepted"] = pd.to_datetime(t["accepted"], utc=True)
    else:
        t["accepted"] = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns, UTC]")
    bound = edgar_accept_after_hours_bound(t["filed"])
    use_acc = t["accepted"].notna() & ~(t["accepted"] > bound)
    t["ts_source"] = pd.array(np.where(use_acc, "accepted", "filed_2200ET_bound"),
                              dtype="string")
    t["ts"] = t["accepted"].where(use_acc, bound)
    return t.loc[:, [c for c in INSIDER_TX_COLUMNS if c in t.columns]]


def insider_monthly(tx: pd.DataFrame) -> pd.DataFrame:
    """PIT monthly table per issuer CIK keyed by FILING availability (``ts``), not by
    transaction date: ``month`` = calendar month of ``ts`` (UTC); ``ts`` = last instant
    of that month (the row is complete, hence usable, only once the month has ended).
    Columns: buy/sell shares and dollars, net shares and dollars (P minus S), counts of
    buy / sell transactions and filings, distinct insiders buying / selling. Dollar sums
    skip ``dollars_suspect`` lines (counted in ``n_dollars_suspect``) and missing prices
    (``n_missing_price``); share sums keep every line."""
    if tx.empty:
        return pd.DataFrame(columns=["cik", "month", "ts"])
    t = tx.dropna(subset=["issuer_cik", "ts"]).copy()
    t["month"] = pd.to_datetime(t["ts"], utc=True).dt.tz_localize(None).dt.to_period("M") \
        .dt.to_timestamp()
    buy = t["trans_code"] == "P"
    sus = t["dollars_suspect"].astype(bool) if "dollars_suspect" in t.columns \
        else pd.Series(False, index=t.index)
    dol = t["dollars"].where(~sus)
    t["buy_shares"] = t["shares"].where(buy, 0.0)
    t["sell_shares"] = t["shares"].where(~buy, 0.0)
    t["buy_dollars"] = dol.where(buy, 0.0)
    t["sell_dollars"] = dol.where(~buy, 0.0)
    t["n_dollars_suspect"] = sus.astype("int32")
    t["is_buy"] = buy.astype("int32")
    t["is_sell"] = (~buy).astype("int32")
    t["no_price"] = t["price"].isna().astype("int32")
    keys = ["issuer_cik", "month"]
    m = t.groupby(keys)[["buy_shares", "sell_shares", "buy_dollars", "sell_dollars", "is_buy",
                         "is_sell", "no_price", "n_dollars_suspect"]].sum(min_count=1).fillna(0.0)
    m = m.rename(columns={"is_buy": "n_buy_tx", "is_sell": "n_sell_tx",
                          "no_price": "n_missing_price"})
    tb, tsell = t[buy], t[~buy]
    m["n_buy_filings"] = tb.groupby(keys)["adsh"].nunique()
    m["n_sell_filings"] = tsell.groupby(keys)["adsh"].nunique()
    m["n_insiders_buying"] = tb.groupby(keys)["owner_id"].nunique()
    m["n_insiders_selling"] = tsell.groupby(keys)["owner_id"].nunique()
    m = m.fillna(0)
    for c in ("n_buy_tx", "n_sell_tx", "n_buy_filings", "n_sell_filings", "n_insiders_buying",
              "n_insiders_selling", "n_missing_price", "n_dollars_suspect"):
        m[c] = m[c].astype("int32")
    m["net_shares"] = m["buy_shares"] - m["sell_shares"]
    m["net_dollars"] = m["buy_dollars"] - m["sell_dollars"]
    m = m.reset_index().rename(columns={"issuer_cik": "cik"})
    m["ts"] = (m["month"] + pd.offsets.MonthBegin(1) - pd.Timedelta(microseconds=1)) \
        .dt.tz_localize("UTC")
    return m


def form345_acceptance(subs: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """From a CIK's submissions JSONs (main + pages): ``adsh, accepted`` (UTC) of every
    Form 3/4/5 (and amendments) listed there, i.e. those where the CIK is the issuer
    or the reporting owner."""
    frames = []
    for s in subs:
        b = _edgar_block(s)
        n = len(b.get("accessionNumber", []))
        if not n:
            continue
        frames.append(pd.DataFrame({"adsh": b["accessionNumber"], "form": b.get("form", [""] * n),
                                    "accepted": b.get("acceptanceDateTime", [None] * n)}))
    if not frames:
        return pd.DataFrame({"adsh": pd.Series(dtype="string"),
                             "accepted": pd.Series(dtype="datetime64[ns, UTC]")})
    df = pd.concat(frames, ignore_index=True)
    df = df[df["form"].isin(SECTION16_FORMS)]
    df["accepted"] = pd.to_datetime(df["accepted"], errors="coerce", utc=True, format="ISO8601")
    return df.loc[:, ["adsh", "accepted"]].drop_duplicates("adsh").reset_index(drop=True)


def insider_transactions_from_zip(path: Path | str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Worker: one Insider Transactions zip -> (P/S transactions without acceptance,
    meta with member row counts and malformed-row counts)."""
    path = Path(path)
    per = dera_period(path.name)
    parts = read_insider_zip(path)
    tx = build_insider_transactions(parts["sub"], parts["owner"], parts["tx"], None,
                                    dataset=per[0] if per else path.stem)
    meta = {"dataset": per[0] if per else path.stem, "file": path.name,
            "n_submissions": len(parts["sub"]), "n_owner_rows": len(parts["owner"]),
            "n_nonderiv_rows": len(parts["tx"]), "n_ps_rows": len(tx),
            **{f"skipped_{k}": int(v.attrs.get("skipped_rows", 0)) for k, v in parts.items()}}
    return tx, meta


def form345_acceptance_from_files(paths: Sequence[Path | str]) -> pd.DataFrame:
    """Worker: :func:`form345_acceptance` over a batch of raw submissions JSON files
    (``.json.gz`` as stored by the per-CIK fetch)."""
    frames = []
    for p in paths:
        try:
            frames.append(form345_acceptance([json.loads(read_raw(Path(p)))]))
        except (OSError, ValueError, KeyError):
            continue
    if not frames:
        return form345_acceptance([])
    return pd.concat(frames, ignore_index=True).drop_duplicates("adsh")


# ---------------------------------------------------------------------------
# 4. Kenneth French daily factors
# ---------------------------------------------------------------------------
def parse_french_daily_csv(text: str) -> pd.DataFrame:
    """French daily CSV (header lines, then ``YYYYMMDD,f1,f2,...`` in percent) -> frame
    indexed by date with factors in DECIMAL (percent / 100). Stops at the first
    non-data block (annual tables / copyright footer)."""
    lines = text.splitlines()
    hdr_i = next(i for i, ln in enumerate(lines)
                 if ln.strip().startswith(",") and re.search(r"[A-Za-z]", ln))
    cols = [c.strip() for c in lines[hdr_i].split(",")[1:]]
    rows = []
    for ln in lines[hdr_i + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if not parts or not re.fullmatch(r"\d{8}", parts[0]):
            if rows:
                break
            continue
        rows.append([parts[0]] + [float(x) for x in parts[1:1 + len(cols)]])
    df = pd.DataFrame(rows, columns=["date", *cols])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date")
    return df / 100.0


# ---------------------------------------------------------------------------
# Loaders (study use: tripwire-routed only)
# ---------------------------------------------------------------------------
def _partition_value(p: Path, key: str) -> str | None:
    for part in p.parts:
        if part.startswith(key + "="):
            return part.split("=", 1)[1]
    return None


def _partition_start(key: str, val: str) -> pd.Timestamp:
    if key == "year":
        return pd.Timestamp(f"{int(val)}-01-01", tz="UTC")
    if key in ("fy_quarter", "quarter"):
        y, q = val.split("q")
        return pd.Timestamp(f"{int(y)}-{3 * (int(q) - 1) + 1:02d}-01", tz="UTC")
    raise ValueError(key)


def _partition_end(key: str, val: str) -> pd.Timestamp:
    if key == "year":
        return pd.Timestamp(f"{int(val) + 1}-01-01", tz="UTC")
    s = _partition_start(key, val)
    return s + pd.DateOffset(months=3)


def _utc(x: pd.Timestamp | str | None) -> pd.Timestamp | None:
    if x is None:
        return None
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _zoned_files(root: Path, key: str, *, unlock_token: str | None,
                 start: pd.Timestamp | str | None = None,
                 end: pd.Timestamp | str | None = None) -> list[Path]:
    """Partition files under ``root``. Without a token only partitions that END at or
    before the equities sealed start are listed (nothing sealed is ever opened).
    ``start`` / ``end`` (optional, ``ts`` bounds, end exclusive) additionally skip every
    partition lying wholly outside ``[start, end)`` -- e.g. a TRAIN-only study passes
    ``end="2019-01-01"`` so the WFA-stage partitions are never opened either."""
    files = sorted(root.glob(f"{key}=*/*.parquet"))
    if unlock_token is None:
        files = [f for f in files
                 if _partition_end(key, _partition_value(f, key) or "0") <= EQUITY_SEALED_START]
    s, e = _utc(start), _utc(end)
    if s is not None:
        files = [f for f in files if _partition_end(key, _partition_value(f, key) or "0") > s]
    if e is not None:
        files = [f for f in files if _partition_start(key, _partition_value(f, key) or "0") < e]
    return files


def _load_partitioned(root: Path, key: str, *, intent: str, columns: Sequence[str] | None = None,
                      allow_sealed: bool = False, unlock_token: str | None = None,
                      filters: Mapping[str, Iterable[Any]] | None = None,
                      start: pd.Timestamp | str | None = None,
                      end: pd.Timestamp | str | None = None) -> pd.DataFrame:
    if not root.exists():
        raise FileNotFoundError(f"{root} missing - run scripts/ingest/ingest_equities_xs.py")
    frames = []
    s, e = _utc(start), _utc(end)
    for f in _zoned_files(root, key, unlock_token=unlock_token if allow_sealed else None,
                          start=s, end=e):
        df = safe_load_parquet(f, zone_config=EQUITIES_ZONES_PATH, allow_sealed=allow_sealed,
                               unlock_token=unlock_token, caller_intent=intent)
        if (s is not None or e is not None) and "ts" in df.columns:
            t = pd.to_datetime(df["ts"], utc=True)
            m = pd.Series(True, index=df.index)
            if s is not None:
                m &= t >= s
            if e is not None:
                m &= t < e
            df = df.loc[m]
        if filters:
            for c, vals in filters.items():
                df = df[df[c].isin(list(vals))]
        if columns is not None:
            df = df.loc[:, [c for c in columns if c in df.columns]]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(columns) if columns else [])
    return pd.concat(frames, ignore_index=True)


def load_yf_daily(*, symbols: Iterable[str] | None = None, columns: Sequence[str] | None = None,
                  allow_sealed: bool = False, unlock_token: str | None = None,
                  start: pd.Timestamp | str | None = None,
                  end: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """yfinance daily panel (< 2022-01-01 unless unlocked). Survivor-only universe.
    ``start`` / ``end`` (``ts`` bounds, end exclusive) skip partitions outside the range."""
    return _load_partitioned(XS_ROOT / "yf_daily", "year", intent="equities_xs.load_yf_daily",
                             columns=columns, allow_sealed=allow_sealed, unlock_token=unlock_token,
                             filters={"symbol": symbols} if symbols is not None else None,
                             start=start, end=end)


def load_sec_num(*, ciks: Iterable[int] | None = None, tags: Iterable[str] | None = None,
                 allow_sealed: bool = False, unlock_token: str | None = None,
                 start: pd.Timestamp | str | None = None,
                 end: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """XBRL facts (FSDS-``num``-like), partitioned by availability quarter; point-in-time
    by ``ts`` (acceptance instant, else filed date). Rows are one per (fact, filing):
    a later amendment's value is a separate row with a later ``ts``. Use
    :func:`latest_as_of` / :func:`pit_filter`; never key availability on ``ddate``."""
    filt: dict[str, Iterable[Any]] = {}
    if ciks is not None:
        filt["cik"] = [int(c) for c in ciks]
    if tags is not None:
        filt["tag"] = list(tags)
    return _load_partitioned(XS_ROOT / "sec_fsds" / "num", "fy_quarter",
                             intent="equities_xs.load_sec_num", allow_sealed=allow_sealed,
                             unlock_token=unlock_token, filters=filt or None,
                             start=start, end=end)


def load_sec_sub(*, allow_sealed: bool = False,
                 unlock_token: str | None = None) -> pd.DataFrame:
    """Periodic-report filings (FSDS-``sub``-like), PIT by ``ts`` = acceptance instant."""
    return _load_partitioned(XS_ROOT / "sec_fsds" / "sub", "year",
                             intent="equities_xs.load_sec_sub",
                             allow_sealed=allow_sealed, unlock_token=unlock_token)


def load_earnings_all(*, allow_sealed: bool = False,
                      unlock_token: str | None = None,
                      start: pd.Timestamp | str | None = None,
                      end: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """8-K item-2.02 earnings events for every 10-K/10-Q filer (``ts`` = acceptance)."""
    return _load_partitioned(XS_ROOT / "events" / "earnings_8k_all", "year",
                             intent="equities_xs.load_earnings_all", allow_sealed=allow_sealed,
                             unlock_token=unlock_token, start=start, end=end)


def _load_single(path: Path, intent: str, allow_sealed: bool = False,
                 unlock_token: str | None = None) -> pd.DataFrame:
    return safe_load_parquet(path, zone_config=EQUITIES_ZONES_PATH, allow_sealed=allow_sealed,
                             unlock_token=unlock_token, caller_intent=intent)


def load_ff_daily(*, allow_sealed: bool = False, unlock_token: str | None = None,
                  start: pd.Timestamp | str | None = None,
                  end: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """French daily factors (decimal), split into pre-sealed / sealed files.

    ``start`` / ``end`` (date bounds on the index, end exclusive) follow the partitioned
    loaders' contract: rows outside ``[start, end)`` are never returned. The pre-sealed
    file is one parquet holding every row up to 2021, so it is read whole and filtered
    here; a TRAIN-only study passes ``end="2019-01-01"`` so the WFA-stage rows never reach
    the caller (red team 2026-10-10, F13). Defaults ``None`` (no bound), as in
    :func:`load_yf_daily`."""
    name = "ff_daily_sealed.parquet" if allow_sealed else "ff_daily.parquet"
    df = _load_single(XS_ROOT / name, "equities_xs.load_ff_daily", allow_sealed, unlock_token)
    if allow_sealed:
        pre = _load_single(XS_ROOT / "ff_daily.parquet", "equities_xs.load_ff_daily")
        df = pd.concat([pre, df]).sort_index()
    if start is not None or end is not None:
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
        m = np.ones(len(df), dtype=bool)
        s, e = _utc(start), _utc(end)
        if s is not None:
            m &= np.asarray(idx >= s)
        if e is not None:
            m &= np.asarray(idx < e)
        df = df.loc[m]
    return df


def load_static(name: str) -> pd.DataFrame:
    """Static (non-time-series) tables: ``universe_current``, ``cik_universe``,
    ``cik_ticker_map``, ``filer_profiles``, ``company_tickers`` (SEC's current
    ticker file). They describe the CURRENT state (e.g. today's tickers) and carry no
    ``ts``: do not use them to form historical portfolios without joining to a PIT
    table."""
    allowed = {"universe_current", "cik_universe", "cik_ticker_map", "filer_profiles",
               "company_tickers"}
    if name not in allowed:
        raise ValueError(f"{name!r} not a static table ({sorted(allowed)})")
    return _load_single(XS_ROOT / f"{name}.parquet", f"equities_xs.load_static({name})")


LINKS_ROOT: Final[Path] = XS_ROOT / "links" / "customer_supplier"
INSIDER_ROOT: Final[Path] = XS_ROOT / "insider"


def load_customer_links(*, suppliers: Iterable[int] | None = None,
                        customers: Iterable[int] | None = None, resolved_only: bool = False,
                        columns: Sequence[str] | None = None, allow_sealed: bool = False,
                        unlock_token: str | None = None,
                        start: pd.Timestamp | str | None = None,
                        end: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """Customer-supplier edges from the Financial Statement and Notes data sets, one row
    per (concentration / by-customer fact, filing), PIT by ``ts`` (filing acceptance;
    partitioned by its calendar quarter ``quarter=YYYYqN``, so a partition never
    straddles the quarter-aligned equities sealed start; sealed partitions are never
    opened unless unlocked). ``resolved_only`` keeps
    rows whose customer was matched to a CIK. ``start`` / ``end`` are ``ts`` bounds (end
    exclusive). See ``data_cache/equities_xs/links/customer_supplier/schema.md``."""
    filt: dict[str, Iterable[Any]] = {}
    if suppliers is not None:
        filt["supplier_cik"] = [int(c) for c in suppliers]
    if customers is not None:
        filt["customer_cik"] = [int(c) for c in customers]
    df = _load_partitioned(XS_ROOT / "links" / "customer_supplier", "quarter",
                           intent="equities_xs.load_customer_links",
                           columns=None, allow_sealed=allow_sealed, unlock_token=unlock_token,
                           filters=filt or None, start=start, end=end)
    if resolved_only and "customer_cik" in df.columns:
        df = df[df["customer_cik"].notna()]
    if columns is not None:
        df = df.loc[:, [c for c in columns if c in df.columns]]
    return df.reset_index(drop=True)


def load_insider_monthly(*, ciks: Iterable[int] | None = None,
                         allow_sealed: bool = False, unlock_token: str | None = None,
                         start: pd.Timestamp | str | None = None,
                         end: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """Monthly open-market insider buying / selling per issuer CIK, keyed by FILING
    availability (``month`` = month of acceptance; ``ts`` = last instant of that month,
    the earliest time the full-month row is knowable). Partitioned ``quarter=YYYYqN``
    by ``ts``; sealed partitions are never opened unless unlocked."""
    return _load_partitioned(XS_ROOT / "insider" / "monthly", "quarter",
                             intent="equities_xs.load_insider_monthly",
                             allow_sealed=allow_sealed, unlock_token=unlock_token,
                             filters={"cik": [int(c) for c in ciks]} if ciks is not None else None,
                             start=start, end=end)


def load_insider_transactions(*, issuers: Iterable[int] | None = None,
                              columns: Sequence[str] | None = None, allow_sealed: bool = False,
                              unlock_token: str | None = None,
                              start: pd.Timestamp | str | None = None,
                              end: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """Non-derivative open-market insider purchases (P) and sales (S), one row per
    transaction line, PIT by ``ts`` (acceptance, else 22:00 ET of the filing date)."""
    return _load_partitioned(XS_ROOT / "insider" / "transactions", "quarter",
                             intent="equities_xs.load_insider_transactions", columns=columns,
                             allow_sealed=allow_sealed, unlock_token=unlock_token,
                             filters={"issuer_cik": [int(c) for c in issuers]}
                             if issuers is not None else None, start=start, end=end)


_MEMBERSHIP_FORWARD_COLS: Final[tuple[str, ...]] = (
    "delisted_flag", "in_current_list", "trading_days_since_last", "inactive_flag", "tickers",
    "exchanges",
)


def load_membership(name: str = "universe_membership", *, allow_sealed: bool = False,
                    unlock_token: str | None = None) -> pd.DataFrame:
    """``universe_membership`` (per price symbol) or ``filer_membership`` (per CIK).

    The stored tables are survivorship BOOKKEEPING computed on the whole file (their
    ``last_date`` and ``delisted_flag`` / ``inactive_flag`` use post-2022 information).
    The study view (default) masks every date >= the equities sealed start to NaT and
    drops the forward-looking flag columns; pass ``allow_sealed`` + a valid
    ``LOCKED_CONFIG_<hex>`` token for the full table. Never form portfolios from this
    table - derive membership as of each date from the PIT price / filing panels."""
    if name not in {"universe_membership", "filer_membership"}:
        raise ValueError(name)
    df = _load_single(XS_ROOT / f"{name}.parquet", f"equities_xs.load_membership({name})")
    if allow_sealed:
        from eqxs.data_zones import SealedZoneAccessError, _is_valid_unlock_token

        if not _is_valid_unlock_token(unlock_token):
            raise SealedZoneAccessError(
                "load_membership(allow_sealed=True) needs a LOCKED_CONFIG token")
        return df
    df = df.drop(columns=[c for c in _MEMBERSHIP_FORWARD_COLS if c in df.columns])
    bound = EQUITY_SEALED_START.tz_localize(None)
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            col = df[c].dt.tz_localize(None) if getattr(df[c].dt, "tz", None) is not None else df[c]
            df[c] = col.where(col < bound)
    return df


def _ingest_only_read(path: Path | str, **kw: Any) -> pd.DataFrame:
    """INGEST ONLY: plain read of an equities_xs parquet (all dates). Study code must use
    the ``load_*`` functions; ingest code must not print anything about rows dated >=
    2022-01-01 beyond counts and schema."""
    return pd.read_parquet(path, **kw)
