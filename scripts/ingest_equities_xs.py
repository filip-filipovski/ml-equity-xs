#!/usr/bin/env python3
"""Ingest the free US equity cross-section into ``data_cache/equities_xs/``.

Free, keyless, no accounts. The only e-mail address is the author-authorised SEC
contact in the User-Agent of sec.gov requests (``xs.SEC_USER_AGENT``, since
2026-10-11); it is sent nowhere else. >= 1 s between requests on
any one connection, exponential backoff on 429 / 5xx, descriptive User-Agent. The SEC
per-CIK fetch runs ``--workers`` (default 6) such connections in parallel under a
process-wide token bucket of ``--max-rps`` (default 8 req/s; SEC's published
fair-access limit is 10 req/s) and pauses all threads ~60 s on a data.sec.gov 403
(rate-threshold page) - measured 5.7 req/s, zero 403s, on 2026-10-10. See
``src/eqxs/equities_xs.py`` for what each source yields and why some planned
sources are not used (Stooq: robots.txt disallows all agents; SEC www.sec.gov answered
403 without a contact e-mail until the 2026-10-11 contact header -> data.sec.gov APIs).

Subcommands (run in this order; each is idempotent, ``--resume`` skips finished work):

    python scripts/ingest_equities_xs.py nasdaq
    python scripts/ingest_equities_xs.py stooq          # records the skip
    python scripts/ingest_equities_xs.py yfinance --resume
    python scripts/ingest_equities_xs.py sec-universe
    python scripts/ingest_equities_xs.py sec-fetch --resume   # hours
    python scripts/ingest_equities_xs.py sec-build      # offline parse
    python scripts/ingest_equities_xs.py ff
    python scripts/ingest_equities_xs.py fred
    python scripts/ingest_equities_xs.py stats          # pre-2022 stats JSON

SEC www.sec.gov bulk data sets (2026-10-11; every sec.gov request carries the declared
contact User-Agent ``xs.SEC_USER_AGENT``, authorised by the author):

    python scripts/ingest_equities_xs.py sec-bulk-fetch --inventory-only
    python scripts/ingest_equities_xs.py sec-bulk-fetch --workers 2
    python scripts/ingest_equities_xs.py sec-links-build --workers 6
    python scripts/ingest_equities_xs.py sec-insider-build --workers 6

``--dry-run`` prints what would be fetched without any network call.

Sealed discipline: ingestion stores all dates; nothing here prints or summarises a row
dated >= 2022-01-01 beyond counts and schema (the ``stats`` subcommand computes every
coverage number on dates < 2022-01-01 or as pure counts).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from eqxs import equities_xs as xs  # noqa: E402
from eqxs.earnings_calendar import (  # noqa: E402
    CSV_COLUMNS,
    build_ticker_events,
    flag_ambiguous,
)

log = logging.getLogger("ingest_equities_xs")
ROOT = xs.XS_ROOT
RAW = xs.RAW_ROOT
STATUS = ROOT / "ingest_status.json"
SEALED = xs.EQUITY_SEALED_START  # 2022-01-01 UTC
SEALED_NAIVE = SEALED.tz_localize(None)
EARNINGS_SINCE = "2004-01-01"   # equities TRAIN start; 8-K items exist from 2004-08
PAGES_SINCE = "2004-01-01"

# Pre-announcements that are not the scheduled quarterly report (public record, decided
# from the filing list before any return was computed; see the sidecar).
STAGE1_OVERRIDES = {
    "0000320193-19-000002": "AAPL 2019-01-02 letter to investors: revenue-guidance cut (unscheduled "
                            "pre-announcement; the scheduled Q1 FY19 report is the 2019-01-29 8-K)",
    "0001045810-19-000004": "NVDA 2019-01-28 revenue-guidance cut (unscheduled pre-announcement; the "
                            "scheduled Q4 FY19 report is the 2019-02-14 8-K)",
}
# Public-record pre-announcements decided from the filing list (no returns computed),
# in addition to the stage-1 overrides.
UNIVERSE_OVERRIDES = {
    "NVDA:2022-08-08": "NVDA 2022-08-08 preliminary Q2 FY23 revenue release (unscheduled "
                       "pre-announcement; the scheduled report is the 2022-08-24 8-K)",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _guard_volume() -> None:
    """Make sure the cache directory exists before anything is written (``data_cache`` may be a symlink)."""
    (REPO / "data_cache").resolve().mkdir(parents=True, exist_ok=True)


def _status_update(key: str, **kw: Any) -> None:
    st = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    st.setdefault(key, {}).update(kw)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(st, indent=2, default=str) + "\n")


def _write_partitioned(df: pd.DataFrame, root: Path, key: str, label: pd.Series,
                       part_name: str = "part-0000.parquet") -> int:
    """Write ``df`` split by ``label`` into ``root/<key>=<label>/<part_name>``."""
    n = 0
    for val, g in df.groupby(label.to_numpy(), sort=True):
        d = root / f"{key}={val}"
        d.mkdir(parents=True, exist_ok=True)
        g.reset_index(drop=True).to_parquet(d / part_name, index=False, compression="zstd")
        n += 1
    return n


def _du(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# 1. Nasdaq Trader
# ---------------------------------------------------------------------------
def cmd_nasdaq(a: argparse.Namespace) -> None:
    t0 = time.monotonic()
    if a.dry_run:
        print("would GET", xs.NASDAQ_LISTED_URL, xs.OTHER_LISTED_URL)
        return
    cli = xs.PoliteClient()
    stamp = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%MZ")
    texts = {}
    for url in (xs.NASDAQ_LISTED_URL, xs.OTHER_LISTED_URL):
        b = cli.get(url)
        if b is None:
            raise SystemExit(f"{url}: 404")
        name = url.rsplit("/", 1)[1]
        xs.write_raw(RAW / "nasdaqtrader" / f"{stamp}_{name}", b, url)
        texts[name] = b.decode("utf-8", "replace")
    uni = xs.build_universe_current(texts["nasdaqlisted.txt"], texts["otherlisted.txt"])
    uni["fetched_utc"] = stamp
    uni.to_parquet(ROOT / "universe_current.parquet", index=False)
    vc = uni["security_type"].value_counts().to_dict()
    print(f"universe_current: {len(uni)} symbols; by type {vc}; "
          f"by exchange {uni['exchange'].value_counts().to_dict()}")
    _status_update("nasdaq", rows=len(uni), by_type=vc, elapsed_s=round(time.monotonic() - t0, 1),
                   file_creation_time=str(uni["file_creation_time"].iloc[0]),
                   raw_bytes=_du(RAW / "nasdaqtrader"),
                   parquet_bytes=_du(ROOT / "universe_current.parquet"))


# ---------------------------------------------------------------------------
# 2. Stooq (policy check only)
# ---------------------------------------------------------------------------
def cmd_stooq(a: argparse.Namespace) -> None:
    """Fetch robots.txt and the /db/h/ page; if robots disallows all agents or the page
    is a bot wall, record the skip and stop. Never circumvents either."""
    if a.dry_run:
        print("would GET https://stooq.com/robots.txt and https://stooq.com/db/h/")
        return
    cli = xs.PoliteClient()
    stamp = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%MZ")
    robots = cli.get("https://stooq.com/robots.txt") or b""
    xs.write_raw(RAW / "stooq" / f"{stamp}_robots.txt", robots, "https://stooq.com/robots.txt")
    page = cli.get("https://stooq.com/db/h/") or b""
    xs.write_raw(RAW / "stooq" / f"{stamp}_db_h.html", page, "https://stooq.com/db/h/")
    txt = robots.decode("utf-8", "replace")
    disallow_all = _robots_disallows_all(txt)
    bot_wall = b"requires JavaScript to verify your browser" in page or b"/__verify" in page
    verdict = ("SKIPPED: robots.txt disallows all user agents" if disallow_all else "") + (
        "; /db/h/ is a JavaScript proof-of-work bot wall" if bot_wall else "")
    print(f"stooq: disallow_all={disallow_all} bot_wall={bot_wall} -> {verdict or 'allowed'}")
    _status_update("stooq", disallow_all=disallow_all, bot_wall=bot_wall,
                   verdict=verdict or "allowed (not implemented)", robots_txt=txt)


def _robots_disallows_all(txt: str) -> bool:
    agent, rules = None, {}
    for ln in txt.splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ":" not in ln:
            continue
        k, v = (x.strip() for x in ln.split(":", 1))
        if k.lower() == "user-agent":
            agent = v
        elif k.lower() == "disallow" and agent is not None:
            rules.setdefault(agent, []).append(v)
    return "/" in rules.get("*", [])


# ---------------------------------------------------------------------------
# 3. yfinance
# ---------------------------------------------------------------------------
def _yf_symbols() -> pd.DataFrame:
    uni = pd.read_parquet(ROOT / "universe_current.parquet")
    return uni[uni["security_type"] == "common"].sort_values("yf_symbol").reset_index(drop=True)


def cmd_yfinance(a: argparse.Namespace) -> None:
    import yfinance as yf

    t0 = time.monotonic()
    syms = _yf_symbols()
    tick = syms["yf_symbol"].tolist()
    batches = [tick[i:i + a.batch] for i in range(0, len(tick), a.batch)]
    rawd = RAW / "yfinance"
    rawd.mkdir(parents=True, exist_ok=True)
    print(f"yfinance: {len(tick)} common stocks in {len(batches)} batches of <= {a.batch}")
    if a.dry_run:
        return
    skips: dict[str, str] = {}
    for bi, batch in enumerate(batches):
        dest = rawd / f"batch_{bi:04d}.parquet"
        if a.resume and dest.exists():
            continue
        got = None
        for attempt in range(1, 5):
            time.sleep(1.5)
            try:
                raw = yf.download(batch, start="1999-01-01", auto_adjust=False, actions=True,
                                  group_by="ticker", threads=False, progress=False, timeout=60)
                got = xs.normalize_yf_download(raw, batch)
                break
            except Exception as exc:  # noqa: BLE001 - unofficial endpoint
                log.warning("batch %d attempt %d: %s", bi, attempt, exc)
                time.sleep(10 * attempt)
        if got is None:
            for s in batch:
                skips[s] = "batch failed persistently"
            continue
        missing = sorted(set(batch) - set(got["symbol"]))
        # retry the empties once, singly (yfinance drops a ticker on transient errors)
        for s in missing:
            time.sleep(1.2)
            try:
                r1 = yf.download(s, start="1999-01-01", auto_adjust=False, actions=True,
                                 threads=False, progress=False, timeout=60)
                g1 = xs.normalize_yf_download(r1, [s])
            except Exception as exc:  # noqa: BLE001
                g1, skips[s] = None, f"{type(exc).__name__}: {exc}"[:160]
            if g1 is not None and len(g1):
                got = pd.concat([got, g1], ignore_index=True)
            else:
                skips.setdefault(s, "no data returned")
        got.to_parquet(dest, index=False, compression="zstd")
        meta = {"call": "yfinance.download", "yfinance_version": yf.__version__,
                "params": {"start": "1999-01-01", "auto_adjust": False, "actions": True,
                           "group_by": "ticker", "threads": False},
                "tickers": batch, "fetched_utc": pd.Timestamp.now("UTC").isoformat(),
                "rows": int(len(got)), "symbols_returned": int(got["symbol"].nunique()),
                "sha256": xs.sha256_bytes(dest.read_bytes()), "size": dest.stat().st_size}
        Path(str(dest) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"  batch {bi + 1}/{len(batches)}: {got['symbol'].nunique()}/{len(batch)} symbols, "
              f"{len(got)} rows", flush=True)
        (rawd / "skips.json").write_text(json.dumps(skips, indent=2) + "\n")
    # merge previously recorded skips (resume) minus anything now present
    _build_yf_canonical(syms, t0, skips)


def _build_yf_canonical(syms: pd.DataFrame, t0: float, skips: dict[str, str]) -> None:
    rawd = RAW / "yfinance"
    parts = [pd.read_parquet(p) for p in sorted(rawd.glob("batch_*.parquet"))]
    px = pd.concat(parts, ignore_index=True).drop_duplicates(["symbol", "date"])
    px = px.sort_values(["symbol", "date"]).reset_index(drop=True)
    out = ROOT / "yf_daily"
    if out.exists():
        shutil.rmtree(out)
    _write_partitioned(px, out, "year", px["date"].dt.year)
    present = set(px["symbol"])
    allskips = {s: r for s, r in skips.items() if s not in present}
    for s in set(syms["yf_symbol"]) - present:
        allskips.setdefault(s, "no data returned")
    (rawd / "skips.json").write_text(json.dumps(dict(sorted(allskips.items())), indent=2) + "\n")
    # membership (pure dates / counts)
    uni = pd.read_parquet(ROOT / "universe_current.parquet")
    exch = dict(zip(uni["yf_symbol"], uni["exchange"], strict=False))
    file_date = pd.Timestamp(px["date"].max())
    mem = xs.build_membership(px, file_date=file_date, current_symbols=uni["yf_symbol"],
                              exchange_of=exch, source="yfinance")
    mem.to_parquet(ROOT / "universe_membership.parquet", index=False)
    print(f"yf_daily: {px['symbol'].nunique()} symbols, {len(px)} rows; skipped {len(allskips)}; "
          f"membership delisted_flag={int(mem['delisted_flag'].sum())} (survivor-only source)")
    _status_update("yfinance", symbols=int(px["symbol"].nunique()), rows=int(len(px)),
                   requested=int(len(syms)), skipped=len(allskips),
                   elapsed_s=round(time.monotonic() - t0, 1), raw_bytes=_du(rawd),
                   parquet_bytes=_du(out), adjustment=xs.PRICE_ADJUSTMENT_NOTE)


# ---------------------------------------------------------------------------
# 4. SEC: universe via XBRL frames
# ---------------------------------------------------------------------------
FRAME_SPECS = (("us-gaap", "Assets", "USD"), ("dei", "EntityCommonStockSharesOutstanding", "shares"))


def cmd_sec_universe(a: argparse.Namespace) -> None:
    t0 = time.monotonic()
    now = pd.Timestamp.now("UTC")
    periods = xs.frame_periods(2009, now.year)
    rawd = RAW / "sec" / "frames"
    rawd.mkdir(parents=True, exist_ok=True)
    manifest = RAW / "sec" / "manifest_frames.jsonl"
    if a.dry_run:
        print(f"would GET {len(periods) * len(FRAME_SPECS)} frames")
        return
    cli = xs.PoliteClient()
    frames = []
    for tax, tag, uom in FRAME_SPECS:
        for per in periods:
            dest = rawd / f"{tax}_{tag}_{uom}_{per}.json.gz"
            if not dest.exists():
                url = xs.SEC_FRAMES_URL.format(tax=tax, tag=tag, uom=uom, period=per)
                b = cli.get(url)
                if b is None:
                    continue
                xs.write_raw(dest, b, url, gzip_it=True, sidecar=False, manifest=manifest)
            d = json.loads(xs.read_raw(dest))
            f = xs.parse_frame(d, per)
            f["spec"] = f"{tax}:{tag}"
            frames.append(f)
    fr = pd.concat(frames, ignore_index=True)
    fr["year"] = fr["period"].str[2:6].astype(int)
    pre = fr[fr["year"] < SEALED.year]
    g = fr.groupby("cik")
    uni = pd.DataFrame({
        "entity_name": g["entity_name"].last(),
        "n_frames": g["period"].nunique(),
        "n_frames_presealed": pre.groupby("cik")["period"].nunique(),
        "first_frame_year": g["year"].min(),
    }).fillna({"n_frames_presealed": 0}).reset_index()
    uni["n_frames_presealed"] = uni["n_frames_presealed"].astype(int)
    uni = uni.sort_values(["n_frames_presealed", "n_frames", "cik"], ascending=[False, False, True])
    uni["priority"] = np.arange(len(uni))
    uni.to_parquet(ROOT / "cik_universe.parquet", index=False)
    print(f"cik_universe: {len(uni)} CIKs from {fr['period'].nunique()} frame periods; "
          f"{int((uni['n_frames_presealed'] > 0).sum())} with a pre-2022 frame; "
          f"{cli.n_requests} requests")
    _status_update("sec_universe", ciks=len(uni),
                   ciks_presealed=int((uni["n_frames_presealed"] > 0).sum()),
                   elapsed_s=round(time.monotonic() - t0, 1), requests=cli.n_requests)


# ---------------------------------------------------------------------------
# 5. SEC: per-CIK submissions (+pages) and companyfacts
# ---------------------------------------------------------------------------
SUBD = RAW / "sec" / "submissions"
FACTD = RAW / "sec" / "companyfacts"
DONE = RAW / "sec" / "fetch_done.jsonl"


def _done_set() -> set[int]:
    if not DONE.exists():
        return set()
    out = set()
    for ln in DONE.read_text().splitlines():
        try:
            out.add(int(json.loads(ln)["cik"]))
        except (ValueError, KeyError):
            continue
    return out


def _fetch_one_cik(cik: int, cli: xs.PoliteClient, app: xs.LockedAppender,
                   man: Path) -> dict[str, Any]:
    """Per-CIK work: submissions main + pages overlapping PAGES_SINCE + companyfacts."""
    rec: dict[str, Any] = {"cik": cik}
    try:
        name = f"CIK{cik:010d}.json"
        url = xs.SEC_SUB_URL.format(name=name)
        b = cli.get(url)
        if b is None:
            rec["submissions"] = "404"
        else:
            xs.write_raw(SUBD / (name + ".gz"), b, url, gzip_it=True, sidecar=False,
                         manifest=man, appender=app)
            main = json.loads(b)
            pages = [f for f in main.get("filings", {}).get("files", [])
                     if f.get("filingTo", "") >= PAGES_SINCE]
            rec["pages"] = len(pages)
            for f in pages:
                pdst = SUBD / (f["name"] + ".gz")
                if pdst.exists():
                    continue
                purl = xs.SEC_SUB_URL.format(name=f["name"])
                pb = cli.get(purl)
                if pb is not None:
                    xs.write_raw(pdst, pb, purl, gzip_it=True, sidecar=False, manifest=man,
                                 appender=app)
        furl = xs.SEC_FACTS_URL.format(cik=cik)
        fb = cli.get(furl)
        if fb is None:
            rec["companyfacts"] = "404"
        else:
            xs.write_raw(FACTD / f"CIK{cik:010d}.json.gz", fb, furl, gzip_it=True,
                         sidecar=False, manifest=man, appender=app)
            rec["companyfacts_bytes"] = len(fb)
        rec["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - record and continue
        rec["status"] = f"error: {type(exc).__name__}: {exc}"[:300]
        log.warning("CIK %d: %s", cik, rec["status"])
    rec["fetched_utc"] = pd.Timestamp.now("UTC").isoformat(timespec="seconds")
    app.append(DONE if rec["status"] == "ok" else ERRS, json.dumps(rec))
    return rec


ERRS = RAW / "sec" / "fetch_errors.jsonl"


def cmd_sec_fetch(a: argparse.Namespace) -> None:
    """Parallel per-CIK fetch: ``--workers`` threads, one PoliteClient each (>= 1.05 s
    between that thread's requests), a process-wide token bucket capping the total at
    ``--max-rps`` (default 8, under SEC's published 10 req/s fair-access limit), and a
    global ~60 s cooldown of all threads on a data.sec.gov 403 (rate threshold)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    t0 = time.monotonic()
    uni = pd.read_parquet(ROOT / "cik_universe.parquet").sort_values("priority")
    ciks = uni["cik"].astype(int).tolist()
    done = _done_set() if a.resume else set()
    if a.retry_errors:
        errs = set()
        if ERRS.exists():
            for ln in ERRS.read_text().splitlines():
                try:
                    errs.add(int(json.loads(ln)["cik"]))
                except (ValueError, KeyError):
                    continue
        todo = [c for c in ciks if c in errs and c not in done]
    else:
        todo = [c for c in ciks if c not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"sec-fetch: {len(ciks)} CIKs, {len(done)} done, {len(todo)} to fetch, "
          f"{a.workers} workers, <= {a.max_rps} req/s total", flush=True)
    if a.dry_run:
        return
    SUBD.mkdir(parents=True, exist_ok=True)
    FACTD.mkdir(parents=True, exist_ok=True)
    man = RAW / "sec" / "manifest_cik.jsonl"
    bucket = xs.TokenBucket(rate=a.max_rps, capacity=a.max_rps)
    cooldown = xs.GlobalCooldown(seconds=60.0)
    app = xs.LockedAppender()
    local = threading.local()
    clients: list[xs.PoliteClient] = []
    clients_lock = threading.Lock()

    def client() -> xs.PoliteClient:
        c = getattr(local, "cli", None)
        if c is None:
            c = xs.PoliteClient(bucket=bucket, cooldown=cooldown, throttle_403=True)
            local.cli = c
            with clients_lock:
                clients.append(c)
        return c

    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(lambda c=c: _fetch_one_cik(c, client(), app, man)) for c in todo]
        for i, fut in enumerate(as_completed(futs)):
            if fut.result()["status"] == "ok":
                n_ok += 1
            else:
                n_err += 1
            if (i + 1) % 200 == 0:
                el = time.monotonic() - t0
                nreq = sum(c.n_requests for c in clients)
                print(f"  {i + 1}/{len(todo)} CIKs, {nreq} requests ({nreq / el:.1f}/s), "
                      f"{el / 60:.0f} min, errors {n_err}, 403-cooldowns {cooldown.n_triggers}, "
                      f"eta {el / (i + 1) * (len(todo) - i - 1) / 3600:.1f} h", flush=True)
    nreq = sum(c.n_requests for c in clients)
    print(f"sec-fetch done: ok {n_ok}, errors {n_err}, requests {nreq}, "
          f"403-cooldowns {cooldown.n_triggers}, {(time.monotonic() - t0) / 60:.0f} min", flush=True)
    _status_update("sec_fetch", ciks_done=len(_done_set()), ciks_total=len(ciks),
                   last_run_requests=nreq, last_run_errors=n_err, workers=a.workers,
                   max_rps=a.max_rps, cooldowns_403=cooldown.n_triggers,
                   last_run_elapsed_s=round(time.monotonic() - t0, 1),
                   raw_bytes=_du(RAW / "sec"))


# ---------------------------------------------------------------------------
# 6. SEC build: sub, num, profiles, ticker map, filer membership, earnings
# ---------------------------------------------------------------------------
def _load_subs(cik: int) -> list[dict]:
    main_p = SUBD / f"CIK{cik:010d}.json.gz"
    if not main_p.exists():
        return []
    main = json.loads(xs.read_raw(main_p))
    subs = [main]
    for f in main.get("filings", {}).get("files", []):
        if f.get("filingTo", "") < PAGES_SINCE:
            continue
        p = SUBD / (f["name"] + ".gz")
        if p.exists():
            subs.append(json.loads(xs.read_raw(p)))
    return subs


def _earnings_for(subs: list[dict], cik: int, profile: dict) -> pd.DataFrame:
    tks = [t for t in profile["tickers"].split(",") if t]
    label = tks[0].upper() if tks else f"CIK{cik}"
    ov = {**STAGE1_OVERRIDES}
    df = build_ticker_events(subs, ticker=label, cik=cik, since=EARNINGS_SINCE, overrides=ov)
    if df.empty:
        return df
    extra = {acc: UNIVERSE_OVERRIDES[f"{label}:{d.date()}"]
             for acc, d in zip(df["accession"], df["announce_date"], strict=True)
             if f"{label}:{d.date()}" in UNIVERSE_OVERRIDES}
    if extra:
        df = build_ticker_events(subs, ticker=label, cik=cik, since=EARNINGS_SINCE,
                                 overrides={**ov, **extra})
    df = flag_ambiguous(df)
    df["source_url"] = xs.SEC_SUB_URL.format(name=f"CIK{cik:010d}.json")
    df["yf_expected_next_unverified"] = ""
    df["tickers_current"] = profile["tickers"]
    df["sic"] = profile["sic"]
    return df


def cmd_sec_build(a: argparse.Namespace) -> None:
    t0 = time.monotonic()
    done = sorted(_done_set())
    if a.limit:
        done = done[: a.limit]
    print(f"sec-build: {len(done)} fetched CIKs", flush=True)
    if a.dry_run:
        return
    out_sub, out_num = ROOT / "sec_fsds" / "sub", ROOT / "sec_fsds" / "num"
    out_ev = ROOT / "events" / "earnings_8k_all"
    for d in (out_sub, out_num, out_ev):
        if d.exists():
            shutil.rmtree(d)
    profiles, all_filings_light, sub_parts, ev_parts = [], [], [], []
    num_buf: list[pd.DataFrame] = []
    n_num_rows, batch_id = 0, 0

    def flush_num() -> None:
        nonlocal num_buf, batch_id, n_num_rows
        if not num_buf:
            return
        nb = pd.concat(num_buf, ignore_index=True)
        n_num_rows += len(nb)
        _write_partitioned(nb, out_num, "fy_quarter", xs.fy_quarter_label(nb["ts"]),
                           part_name=f"part-{batch_id:04d}.parquet")
        batch_id += 1
        num_buf = []

    buf_rows = 0
    build_errors: dict[int, str] = {}
    for i, cik in enumerate(done):
        try:
            subs = _load_subs(cik)
            if not subs:
                build_errors[cik] = "no submissions file"
                continue
            prof = xs.filer_profile(subs[0])
            prof["cik"] = cik
            fil = xs.parse_submission_filings(subs, cik)
            fpath = FACTD / f"CIK{cik:010d}.json.gz"
            num = xs.parse_companyfacts(json.loads(xs.read_raw(fpath))) if fpath.exists() \
                else pd.DataFrame(columns=list(xs.NUM_COLUMNS))
            if len(num):
                num["cik"] = cik
                num = xs.attach_num_ts(num, fil)
            sub_rows = xs.build_sub_rows(fil, prof, xs.fyfp_by_adsh(num))
        except Exception as exc:  # noqa: BLE001 - one bad filer must not kill the build
            build_errors[cik] = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("build CIK %d: %s", cik, build_errors[cik])
            continue
        profiles.append(prof)
        all_filings_light.append(fil.loc[fil["form"].isin(xs.PERIODIC_FORMS),
                                         ["adsh", "cik", "form", "filed"]])
        if len(num):
            num_buf.append(num)
            buf_rows += len(num)
        sub_parts.append(sub_rows)
        try:
            ev = _earnings_for(subs, cik, prof)
            if len(ev):
                ev_parts.append(ev)
        except Exception as exc:  # noqa: BLE001
            log.warning("earnings CIK %d: %s", cik, exc)
        if buf_rows > 4_000_000:
            flush_num()
            buf_rows = 0
        if (i + 1) % 1000 == 0:
            print(f"  parsed {i + 1}/{len(done)} CIKs ({(time.monotonic() - t0) / 60:.0f} min)",
                  flush=True)
    flush_num()
    prof_df = pd.DataFrame(profiles)
    prof_df.to_parquet(ROOT / "filer_profiles.parquet", index=False)
    sub = pd.concat(sub_parts, ignore_index=True)
    sub["fy"] = pd.to_numeric(sub["fy"], errors="coerce").astype("Int32")
    sub["fp"] = sub["fp"].astype("string")
    _write_partitioned(sub, out_sub, "year", pd.to_datetime(sub["ts"], utc=True).dt.year)
    filings = pd.concat(all_filings_light, ignore_index=True)
    now = pd.Timestamp.now("UTC").tz_localize(None)
    xs.build_filer_membership(filings, prof_df, as_of=now).to_parquet(
        ROOT / "filer_membership.parquet", index=False)
    uni = pd.read_parquet(ROOT / "universe_current.parquet")
    tmap = xs.build_cik_ticker_map(prof_df, uni)
    tmap.to_parquet(ROOT / "cik_ticker_map.parquet", index=False)
    # SEC-side ticker list (every ticker any fetched CIK claims, incl. non-Nasdaq-listed)
    ev = pd.concat(ev_parts, ignore_index=True) if ev_parts else pd.DataFrame(columns=CSV_COLUMNS)
    ev["acceptance_et"] = ev["acceptance_et"].astype(str)
    ev["ts"] = pd.to_datetime(ev["acceptance_raw"], utc=True)
    keep_cols = [*CSV_COLUMNS, "tickers_current", "sic", "ts"]
    ev = ev.loc[:, keep_cols]
    ev["keep"] = ev["keep"].astype(bool)
    _write_partitioned(ev, out_ev, "year", ev["ts"].dt.year)
    print(f"sub rows {len(sub)}; num rows {n_num_rows}; profiles {len(prof_df)}; "
          f"ticker map matched {int(tmap['cik'].notna().sum())}/{len(tmap)}; "
          f"earnings 2.02 8-Ks {len(ev)} (kept {int(ev['keep'].sum())})", flush=True)
    (ROOT / "sec_build_errors.json").write_text(json.dumps(build_errors, indent=2) + "\n")
    _status_update("sec_build", ciks=len(prof_df), build_errors=len(build_errors), sub_rows=int(len(sub)), num_rows=int(n_num_rows),
                   earnings_rows=int(len(ev)), earnings_kept=int(ev["keep"].sum()),
                   elapsed_s=round(time.monotonic() - t0, 1),
                   parquet_bytes={"sub": _du(out_sub), "num": _du(out_num), "earnings": _du(out_ev)})


# ---------------------------------------------------------------------------
# 7. French factors, FRED
# ---------------------------------------------------------------------------
def cmd_ff(a: argparse.Namespace) -> None:
    import io
    import zipfile

    t0 = time.monotonic()
    urls = {"ff3": xs.FF3_DAILY_URL, "mom": xs.FF_MOM_DAILY_URL, "ff5": xs.FF5_DAILY_URL}
    if a.dry_run:
        print("would GET", list(urls.values()))
        return
    cli = xs.PoliteClient()
    frames = {}
    for k, url in urls.items():
        b = cli.get(url)
        if b is None:
            print(f"{url}: 404")
            continue
        xs.write_raw(RAW / "french" / url.rsplit("/", 1)[1], b, url)
        z = zipfile.ZipFile(io.BytesIO(b))
        txt = z.read(z.namelist()[0]).decode("latin-1")
        frames[k] = xs.parse_french_daily_csv(txt)
    ff = frames["ff3"].join(frames["mom"].rename(columns=lambda c: "MOM" if "Mom" in c else c),
                            how="outer")
    if "ff5" in frames:
        f5 = frames["ff5"][["RMW", "CMA"]]
        ff = ff.join(f5, how="outer")
    ff.index.name = "date"
    ff = ff.sort_index()
    ff.index = ff.index.tz_localize("UTC")
    ff.loc[ff.index < SEALED].to_parquet(ROOT / "ff_daily.parquet")
    ff.loc[ff.index >= SEALED].to_parquet(ROOT / "ff_daily_sealed.parquet")
    pre = ff.loc[ff.index < SEALED]
    print(f"ff_daily: cols {list(ff.columns)}; {len(ff)} rows total; pre-2022 span "
          f"{pre.index.min().date()}..{pre.index.max().date()}")
    _status_update("ff", rows=int(len(ff)), rows_presealed=int(len(pre)), columns=list(ff.columns),
                   span_presealed=[str(pre.index.min().date()), str(pre.index.max().date())],
                   elapsed_s=round(time.monotonic() - t0, 1), raw_bytes=_du(RAW / "french"))


def cmd_fred(a: argparse.Namespace) -> None:
    src = REPO / "data_cache" / "macro" / "fred" / "DGS3MO.parquet"
    if a.dry_run:
        print("would reuse", src)
        return
    df = pd.read_parquet(src)
    if "ts" in df.columns:
        df = df.set_index("ts")
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    s = pd.Series(df["value"].to_numpy(float), index=idx, name="dgs3mo_pct").dropna()
    out = pd.DataFrame({"dgs3mo_pct": s, "rf_daily": (s / 100.0) / 252.0})
    out.index.name = "date"
    out.loc[out.index < SEALED].to_parquet(ROOT / "rf_dgs3mo.parquet")
    out.loc[out.index >= SEALED].to_parquet(ROOT / "rf_dgs3mo_sealed.parquet")
    pre = out.loc[out.index < SEALED]
    print(f"rf_dgs3mo: reused {src.relative_to(REPO)}; pre-2022 {len(pre)} rows "
          f"{pre.index.min().date()}..{pre.index.max().date()}; total rows {len(out)}")
    _status_update("fred", source=str(src.relative_to(REPO)), rows=int(len(out)),
                   rows_presealed=int(len(pre)),
                   span_presealed=[str(pre.index.min().date()), str(pre.index.max().date())])


# ---------------------------------------------------------------------------
# 8. Stats (pre-2022 only, or pure counts)
# ---------------------------------------------------------------------------
def cmd_stats(a: argparse.Namespace) -> None:
    rng = np.random.default_rng(20261010)
    st: dict[str, Any] = {}
    # yfinance panel, pre-sealed partitions only for spans/stats
    yd = ROOT / "yf_daily"
    if yd.exists():
        files = sorted(yd.glob("year=*/*.parquet"))
        pre_files = [f for f in files if int(f.parent.name.split("=")[1]) < SEALED.year]
        rows_total = sum(pd.read_parquet(f, columns=["symbol"]).shape[0] for f in files)
        px = pd.concat([pd.read_parquet(f) for f in pre_files], ignore_index=True)
        per_year = px.groupby(px["date"].dt.year)["symbol"].nunique()
        st["yf"] = {"rows_total": int(rows_total), "rows_presealed": int(len(px)),
                    "symbols_presealed": int(px["symbol"].nunique()),
                    "span_presealed": [str(px["date"].min().date()), str(px["date"].max().date())],
                    "symbols_per_year_presealed": {int(k): int(v) for k, v in per_year.items()}}
        # dividend-adjustment self-consistency (adj_close vs close*factor) on 200 symbols
        syms = rng.choice(px["symbol"].unique(), size=min(200, px["symbol"].nunique()),
                          replace=False)
        errs = []
        for s in syms:
            g = px[px["symbol"] == s].sort_values("date")
            if len(g) < 50:
                continue
            f = xs.dividend_adjustment_factor(g)
            mine = g["close"] * f
            # rescale both to the last pre-sealed date (Yahoo's adj is anchored at today)
            r_mine = mine / mine.iloc[-1]
            r_yf = g["adj_close"] / g["adj_close"].iloc[-1]
            errs.append(float((np.abs(r_mine / r_yf - 1) < 0.005).mean()))
        st["yf"]["div_factor_agreement_share_days_within_0.5pct_200sym"] = float(np.mean(errs))
        # independent check vs French market: equal-weight mean daily return, 2004-2021
        px = px.sort_values(["symbol", "date"])
        px["ret"] = px.groupby("symbol")["adj_close"].pct_change()
        px = px[(px["ret"].abs() < 1.0) & (px["date"] >= "2004-01-01")]
        ew = px.groupby("date")["ret"].mean()
        ffp = ROOT / "ff_daily.parquet"
        if ffp.exists():
            ff = pd.read_parquet(ffp)
            ff.index = pd.DatetimeIndex(ff.index).tz_localize(None)
            mkt = (ff["Mkt-RF"] + ff["RF"]).reindex(ew.index).dropna()
            st["yf"]["corr_ew_vs_ff_mkt_2004_2021"] = float(ew.reindex(mkt.index).corr(mkt))
    # SEC sub / num / earnings
    sd = ROOT / "sec_fsds" / "sub"
    if sd.exists():
        files = sorted(sd.glob("year=*/*.parquet"))
        n_total = sum(pd.read_parquet(f, columns=["adsh"]).shape[0] for f in files)
        sub = pd.concat([pd.read_parquet(f) for f in files
                         if int(f.parent.name.split("=")[1]) < SEALED.year], ignore_index=True)
        sub["y"] = pd.to_datetime(sub["ts"], utc=True).dt.year
        by = sub.groupby("y")
        st["sec_sub"] = {
            "rows_total": int(n_total), "rows_presealed": int(len(sub)),
            "filers_per_year": {int(k): int(v) for k, v in by["cik"].nunique().items()},
            "n_10k_per_year": {int(k): int(v) for k, v in
                               sub[sub["form"].str.startswith("10-K")].groupby("y").size().items()},
            "n_10q_per_year": {int(k): int(v) for k, v in
                               sub[sub["form"].str.startswith("10-Q")].groupby("y").size().items()},
            "xbrl_share_per_year": {int(k): round(float(v), 3) for k, v in
                                    by["is_xbrl"].mean().items()},
        }
    nd = ROOT / "sec_fsds" / "num"
    if nd.exists():
        import pyarrow.parquet as pq

        files = sorted(nd.glob("fy_quarter=*/*.parquet"))
        tot = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        pre = [f for f in files if xs._partition_end("fy_quarter", f.parent.name.split("=")[1])
               <= SEALED]
        pre_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in pre)
        ciks_pre: set[int] = set()
        tags_pre: set[str] = set()
        for f in pre:
            t = pq.read_table(f, columns=["cik", "tag"])
            ciks_pre.update(t.column("cik").to_pylist())
            tags_pre.update(set(t.column("tag").to_pylist()))
        st["sec_num"] = {"rows_total": int(tot), "rows_presealed": int(pre_rows),
                         "partitions": len(files), "ciks_presealed": len(ciks_pre),
                         "distinct_tags_presealed": len(tags_pre)}
    ed = ROOT / "events" / "earnings_8k_all"
    if ed.exists():
        files = sorted(ed.glob("year=*/*.parquet"))
        n_total = sum(pd.read_parquet(f, columns=["accession"]).shape[0] for f in files)
        ev = pd.concat([pd.read_parquet(f) for f in files
                        if int(f.parent.name.split("=")[1]) < SEALED.year], ignore_index=True)
        k = ev[ev["keep"]]
        k = k.assign(y=pd.to_datetime(k["ts"], utc=True).dt.year)
        cy = k.groupby(["cik", "y"]).size()
        full = cy[(cy.index.get_level_values("y") >= 2005)]
        st["earnings"] = {
            "rows_total": int(n_total), "rows_presealed": int(len(ev)),
            "kept_presealed": int(len(k)), "ciks_presealed": int(k["cik"].nunique()),
            "ciks_per_year": {int(a_): int(b) for a_, b in k.groupby("y")["cik"].nunique().items()},
            "mean_events_per_cik_year_2005_2021": float(full.mean()),
            "share_cik_years_with_4": float((full == 4).mean()),
            "timing_share": {str(a_): round(float(b), 3) for a_, b in
                             k["timing"].value_counts(normalize=True).items()},
            "ambiguous_share_kept": float((k["ambiguous"] != "").mean()),
        }
    fm = ROOT / "filer_membership.parquet"
    if fm.exists():
        m = pd.read_parquet(fm)
        st["filer_membership"] = {"ciks": int(len(m)), "inactive_flag": int(m["inactive_flag"].sum()),
                                  "with_current_ticker": int((m["tickers"].fillna("") != "").sum())}
    tm = ROOT / "cik_ticker_map.parquet"
    if tm.exists():
        t = pd.read_parquet(tm)
        st["cik_ticker_map"] = {"listing_rows": int(len(t)),
                                "by_match": t["match"].value_counts().to_dict(),
                                "common_matched": int(t[(t["security_type"] == "common")]["cik"]
                                                      .notna().sum()),
                                "common_total": int((t["security_type"] == "common").sum())}
    st["disk_bytes"] = {"equities_xs_total": _du(ROOT), "raw": _du(RAW)}
    (ROOT / "stats.json").write_text(json.dumps(st, indent=2, default=str) + "\n")
    print(json.dumps(st, indent=2, default=str)[:6000])


# ---------------------------------------------------------------------------
# 7. SEC bulk data sets on www.sec.gov (declared-contact User-Agent)
# ---------------------------------------------------------------------------
BULK = xs.SEC_BULK_ROOT
# The 2026-10-10 report's "9,830 pre-2022 filers without a ticker" baseline is fixed at
# 2022-01-01 (the then sealed start) so the company_tickers delta is comparable.
PRE2022 = pd.Timestamp("2022-01-01")
BULK_MAN = BULK / "manifest.jsonl"
BULK_ITEMS = ("company_tickers", "fsds_notes", "insider", "full_index")


def _bulk_records(kind: str) -> dict[str, dict[str, Any]]:
    """Latest manifest record of ``kind`` per URL."""
    out: dict[str, dict[str, Any]] = {}
    if not BULK_MAN.exists():
        return out
    for ln in BULK_MAN.read_text().splitlines():
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("kind") == kind:
            out[r["url"]] = r
    return out


def _bulk_targets(cli: xs.PoliteClient, items: set[str], a: argparse.Namespace,
                  app: xs.LockedAppender) -> list[dict[str, Any]]:
    """Enumerate every file to fetch (listing pages are fetched and stored raw)."""
    now = pd.Timestamp.now("UTC")
    stamp = now.strftime("%Y%m%d")
    tg: list[dict[str, Any]] = []
    if "company_tickers" in items:
        for url in (xs.SEC_COMPANY_TICKERS_URL, xs.SEC_COMPANY_TICKERS_EXCHANGE_URL):
            name = url.rsplit("/", 1)[1].replace(".json", f"_{stamp}.json")
            tg.append({"item": "company_tickers", "url": url, "period": stamp,
                       "dest": f"company_tickers/{name}", "size_listed": None})
    for item, lurl, since in (("fsds_notes", xs.SEC_NOTES_LISTING_URL, a.notes_since),
                              ("insider", xs.SEC_INSIDER_LISTING_URL, a.insider_since)):
        if item not in items:
            continue
        html = cli.get(lurl)
        if html is None:
            raise SystemExit(f"{lurl}: 404")
        xs.write_raw(BULK / "listings" / f"{now:%Y%m%dT%H%MZ}_{item}_listing.html", html, lurl)
        lst = xs.parse_dera_listing(html.decode("utf-8", "replace"))
        lo = xs.period_bounds(since)[0]
        lst = lst[lst["period"].map(lambda p: xs.period_bounds(p)[0]) >= lo]
        for r in lst.itertuples(index=False):
            tg.append({"item": item, "url": r.url, "period": r.period,
                       "dest": f"{item}/{r.file}", "size_listed": r.size_listed})
    if "full_index" in items:
        for qtr in xs.full_index_quarters(a.index_since, now.tz_localize(None)):
            tg.append({"item": "full_index", "url": xs.full_index_url(qtr), "period": qtr,
                       "dest": f"full_index/master_{qtr}.gz", "size_listed": None})
    return tg


def cmd_sec_bulk_fetch(a: argparse.Namespace) -> None:
    """Inventory + resumable download of the www.sec.gov bulk data sets into
    ``raw/sec_bulk/{company_tickers,fsds_notes,insider,full_index}/``.

    Every request carries :data:`xs.SEC_USER_AGENT`. ``--workers`` concurrent
    connections (default 2), a process-wide token bucket of ``--max-rps``, a global 60 s
    cooldown on 403/429; an item whose files keep answering 403 is stopped and reported
    (no evasion). Each target first gets an ``inventory`` manifest line (listed size,
    HEAD size, Last-Modified); each finished file a ``file`` line (size, sha256, fetch
    time). Files already in the manifest with a matching on-disk size are skipped (the
    current quarter's master index and the current month's notes are re-fetched only
    with ``--refresh-current``). Partial downloads resume from ``*.part`` via HTTP Range."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    t0 = time.monotonic()
    items = set(a.items.split(",")) if a.items else set(BULK_ITEMS)
    bad = items - set(BULK_ITEMS)
    if bad:
        raise SystemExit(f"unknown items {bad}; choose from {BULK_ITEMS}")
    BULK.mkdir(parents=True, exist_ok=True)
    bucket = xs.TokenBucket(rate=a.max_rps, capacity=a.max_rps)
    cooldown = xs.GlobalCooldown(seconds=60.0)
    app = xs.LockedAppender()
    mk = lambda: xs.PoliteClient(min_interval=0.25, bucket=bucket, cooldown=cooldown,  # noqa: E731
                                 throttle_403=True, retries_403=3, timeout=600.0)
    cli = mk()
    targets = _bulk_targets(cli, items, a, app)
    inv = _bulk_records("inventory")
    need_head = [t for t in targets if t["url"] not in inv or a.reinventory]
    print(f"sec-bulk-fetch: {len(targets)} targets ({', '.join(sorted(items))}); "
          f"HEAD {len(need_head)}", flush=True)
    if a.dry_run:
        for t in targets[:10]:
            print("  would GET", t["url"], "->", t["dest"])
        return
    for i, t in enumerate(need_head):
        try:
            h = cli.head(t["url"])
        except xs.FetchError as exc:
            h = {"status": f"error: {exc}"[:200]}
        rec = {"kind": "inventory", "item": t["item"], "url": t["url"], "period": t["period"],
               "size_listed": t["size_listed"], "size_head": (h or {}).get("size"),
               "last_modified": (h or {}).get("last_modified"),
               "status": "404" if h is None else h.get("status"),
               "checked_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds")}
        app.append(BULK_MAN, json.dumps(rec, separators=(",", ":")))
        inv[t["url"]] = rec
        if (i + 1) % 50 == 0:
            print(f"  HEAD {i + 1}/{len(need_head)}", flush=True)
    by_item: dict[str, list[int]] = {}
    for t in targets:
        r = inv.get(t["url"], {})
        by_item.setdefault(t["item"], [0, 0])
        by_item[t["item"]][0] += 1
        by_item[t["item"]][1] += int(r.get("size_head") or t["size_listed"] or 0)
    for k, (n, b) in sorted(by_item.items()):
        print(f"  inventory {k}: {n} files, {b / 1e9:.2f} GB", flush=True)
    if a.inventory_only:
        return
    done = _bulk_records("file")
    now = pd.Timestamp.now("UTC").tz_localize(None)
    cur_q, cur_m = f"{now.year}q{now.quarter}", f"{now.year}m{now.month:02d}"
    todo = []
    for t in targets:
        p = BULK / t["dest"]
        r = done.get(t["url"])
        fresh = p.exists() and r is not None and r.get("size") == p.stat().st_size
        if fresh and not (a.refresh_current and t["period"] in (cur_q, cur_m)):
            continue
        if inv.get(t["url"], {}).get("status") == "404":
            continue
        todo.append(t)
    tot = sum(int(inv.get(t["url"], {}).get("size_head") or t["size_listed"] or 0) for t in todo)
    print(f"  to download: {len(todo)} files, ~{tot / 1e9:.2f} GB, {a.workers} connections",
          flush=True)
    blocked: dict[str, str] = {}
    lock = threading.Lock()
    local = threading.local()
    stats = {"bytes": 0, "files": 0, "errors": 0}

    def one(t: dict[str, Any]) -> None:
        if t["item"] in blocked:
            return
        c = getattr(local, "cli", None)
        if c is None:
            c = local.cli = mk()
        dest = BULK / t["dest"]
        t1 = time.monotonic()
        try:
            r = c.download(t["url"], dest)
        except xs.FetchError as exc:
            msg = str(exc)
            with lock:
                stats["errors"] += 1
                if "403" in msg:
                    blocked[t["item"]] = msg[:200]
            app.append(BULK_MAN, json.dumps({"kind": "error", "item": t["item"], "url": t["url"],
                                             "error": msg[:300]}, separators=(",", ":")))
            log.warning("%s: %s", t["url"], msg)
            return
        if r is None:
            return
        rec = {"kind": "file", "item": t["item"], "url": t["url"], "period": t["period"],
               "stored_as": t["dest"], "size": r["size"], "sha256": r["sha256"],
               "fetched_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
               "elapsed_s": round(time.monotonic() - t1, 1),
               "last_modified": inv.get(t["url"], {}).get("last_modified"),
               "user_agent": xs.user_agent_for(t["url"])}
        app.append(BULK_MAN, json.dumps(rec, separators=(",", ":")))
        with lock:
            stats["bytes"] += r["size"]
            stats["files"] += 1
            n, b = stats["files"], stats["bytes"]
        if n % 10 == 0 or r["size"] > 200e6:
            el = time.monotonic() - t0
            print(f"  {n}/{len(todo)} files, {b / 1e9:.2f} GB, {b / 1e6 / max(el, 1):.1f} MB/s, "
                  f"{el / 60:.0f} min (last {t['dest']})", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(one, t) for t in todo]):
            f.result()
    el = time.monotonic() - t0
    print(f"sec-bulk-fetch done: {stats['files']} files, {stats['bytes'] / 1e9:.2f} GB, "
          f"errors {stats['errors']}, 403-cooldowns {cooldown.n_triggers}, {el / 60:.1f} min; "
          f"blocked items {blocked or 'none'}", flush=True)
    if "company_tickers" in items:
        ctst = _build_company_tickers()
        print("company_tickers:", json.dumps(ctst, default=str), flush=True)
        _status_update("company_tickers", **ctst)
    _status_update("sec_bulk_fetch", files=stats["files"], bytes=stats["bytes"],
                   errors=stats["errors"], blocked=blocked, cooldowns_403=cooldown.n_triggers,
                   elapsed_s=round(el, 1), workers=a.workers, max_rps=a.max_rps,
                   raw_bytes=_du(BULK), user_agent=xs.SEC_USER_AGENT)


def _latest_raw(item_dir: Path, prefix: str) -> Path | None:
    c = sorted(f for f in item_dir.glob(f"{prefix}_*.json")
               if re.fullmatch(rf"{prefix}_\d{{8}}\.json", f.name))
    return c[-1] if c else None


def _build_company_tickers() -> dict[str, Any]:
    """``company_tickers.parquet`` from the newest raw pair + CIK->ticker coverage delta
    versus ``cik_ticker_map`` / ``filer_membership`` (counts only)."""
    d = BULK / "company_tickers"
    p1, p2 = _latest_raw(d, "company_tickers"), _latest_raw(d, "company_tickers_exchange")
    if p1 is None or p2 is None:
        return {"status": "missing raw files"}
    t1 = xs.parse_company_tickers(json.loads(p1.read_bytes()))
    t2 = xs.parse_company_tickers_exchange(json.loads(p2.read_bytes()))
    ct = xs.build_company_tickers(t1, t2, fetched_utc=p1.stem.rsplit("_", 1)[-1])
    ct.to_parquet(ROOT / "company_tickers.parquet", index=False)
    st: dict[str, Any] = {"rows": len(ct), "ciks": int(ct["cik"].nunique()),
                          "rows_company_tickers": len(t1), "rows_exchange_file": len(t2),
                          "by_exchange": ct["exchange"].replace("", "(none)").value_counts()
                          .to_dict()}
    fm = pd.read_parquet(ROOT / "filer_membership.parquet")
    pre = fm[pd.to_datetime(fm["first_periodic_filed"]) < PRE2022]
    no_tk = pre[pre["tickers"].fillna("") == ""]
    ct_ciks = set(ct["cik"])
    st["pre2022_filers"] = len(pre)
    st["pre2022_filers_without_profile_ticker"] = len(no_tk)
    st["of_which_in_company_tickers"] = int(no_tk["cik"].isin(ct_ciks).sum())
    st["pre2022_filers_in_company_tickers"] = int(pre["cik"].isin(ct_ciks).sum())
    st["pre2022_filers_with_profile_ticker"] = int((pre["tickers"].fillna("") != "").sum())
    prof_ciks = set(pd.read_parquet(ROOT / "filer_profiles.parquet")["cik"].astype(int))
    st["company_tickers_ciks_not_in_our_filer_universe"] = len(ct_ciks - prof_ciks)
    tm = pd.read_parquet(ROOT / "cik_ticker_map.parquet")
    com = tm[tm["security_type"] == "common"]
    miss = com[com["cik"].isna()]
    norm = lambda s: s.astype(str).str.upper().str.replace(r"[.\-/$]", "", regex=True)  # noqa: E731
    ctn = set(norm(ct["ticker"]))
    st["listing_common_rows"] = len(com)
    st["listing_common_unmapped_before"] = len(miss)
    st["listing_common_unmapped_now_in_company_tickers"] = int(norm(miss["symbol"]).isin(ctn).sum())
    mapped = com[com["cik"].notna()]
    jj = mapped.assign(n=norm(mapped["symbol"])).merge(
        ct.assign(n=norm(ct["ticker"])), on="n", how="inner", suffixes=("", "_ct"))
    st["listing_common_mapped_agree_cik"] = int((jj["cik"].astype(int) == jj["cik_ct"]).sum())
    st["listing_common_mapped_disagree_cik"] = int((jj["cik"].astype(int) != jj["cik_ct"]).sum())
    return st


def _sub_acceptance() -> pd.DataFrame:
    """INGEST ONLY: ``adsh, accepted`` (true UTC, submissions JSON) for every periodic
    filing in our ``sec_fsds/sub`` (all partitions)."""
    parts = [xs._ingest_only_read(p, columns=["adsh", "accepted"])
             for p in sorted((ROOT / "sec_fsds" / "sub").glob("year=*/*.parquet"))]
    s = pd.concat(parts, ignore_index=True).drop_duplicates("adsh")
    s["adsh"] = s["adsh"].astype("string")
    s["accepted"] = pd.to_datetime(s["accepted"], utc=True)
    return s


LINKS_SCHEMA = """# customer_supplier links (Financial Statement and Notes data sets)

Producer: `scripts/ingest_equities_xs.py sec-links-build` from
`raw/sec_bulk/fsds_notes/*.zip` (SEC DERA Financial Statement and Notes data sets,
quarterly 2009q2-2025q3, monthly 2025-10+). Partitioned `quarter=YYYYqN` by the calendar
quarter of `ts` (quarters align with the equities sealed start, 2024-07-01 since the
2026-10-11 zone change; a year partition would straddle it).
Loader: `eqxs.equities_xs.load_customer_links(start=, end=)` (tripwire-routed,
< 2022 unless unlocked).

One row per (XBRL fact, filing) where the fact's dimension set (`dim.segments`) has a
customer axis (axis local name contains `customer` / `client` and is not a
type/class/industry/region/segment classification axis) and the tag is a concentration
percentage (`ConcentrationRiskPercentage1`, `ConcentrationRiskPercentage`) or a
revenue / receivable amount (`Revenues`, `SalesRevenueNet`, `SalesRevenueGoodsNet`,
`SalesRevenueServicesNet`, `RevenueFromContractWithCustomer{Ex,In}cludingAssessedTax`,
`RevenueNet`, `AccountsReceivable{Net,Gross}Current`). A filing reports prior periods
too, so the same (supplier, customer, ddate) appears once per filing that reported it
(restatements included) -- point-in-time by `ts`, never by `ddate`.

| column | meaning |
|---|---|
| supplier_cik, supplier_name, supplier_sic | the filer (FSN `sub`) |
| coreg | co-registrant the fact belongs to ('' = the filer itself) |
| customer_axis | axis local name (FSN strips the `Axis` suffix), e.g. `MajorCustomers` |
| customer_member | raw member local name (`Member` suffix stripped by FSN), e.g. `AppleInc` |
| customer_name | `customer_member` split into words, e.g. `Apple Inc` |
| anonymous | generic member (`Customer A`, `Customera`, `Two Customers`, `Largest Customer`, `Vendor One`, `Commercial`, `International Customers`) |
| vendor_hint | member names a vendor / supplier (supplier concentration hung on a customer axis: counterparty is upstream) |
| government | government / agency / military / Medicare-type customer |
| benchmark | `ConcentrationRiskByBenchmark` member (`SalesRevenueNet`, `AccountsReceivable`, ...; '' = not tagged) |
| is_revenue_benchmark | benchmark mentions revenue / sales (`pct` is a % of revenue) |
| risk_type | `ConcentrationRiskByType` member (e.g. `CustomerConcentrationRisk`) |
| tag, uom, value | XBRL tag, unit, value as reported |
| pct | percent of benchmark (concentration tags; `value*100` for ratios <= 1.5) |
| pct_scale_suspect | `pure` value > 1.5 (filer typed a percent); `pct` = value as typed |
| ddate, qtrs | period end, duration in quarters (0 = instant) |
| fy, fp, form | fiscal year / period / form of the filing |
| adsh, filed | accession, filing date |
| accepted_fsn | FSN `sub.accepted` read as US/Eastern -> UTC |
| accepted | best estimate of the EDGAR acceptance instant (UTC): the submissions-JSON stamp when it agrees with `accepted_fsn` within 2 min, else `accepted_fsn` (the JSON stamp is late by exactly the ET UTC offset on ~15-28 % of filings; verified against EDGAR index pages) |
| accepted_source | `submissions_json=fsn`, `fsn_sub(json_shifted)`, `fsn_sub` (no JSON), `submissions_json_unverified`, `none` |
| ts | PIT availability (UTC) = the earliest of three upper bounds on the true acceptance: submissions-JSON stamp, `accepted_fsn` + 30 s (FSN rounds to the minute), 22:00 ET of `filed` |
| segments, n_axes | full dimension string and number of axes (> 1: the % is within a sub-segment) |
| dataset | data-set period the row came from (`2012q2`, `2025m10`) |
| customer_cik | resolved customer CIK (NULL if unresolved) |
| match_method | `exact`, `exact_compact` (normalised names equal), `prefix_listed` (customer tokens = leading tokens of the CURRENT name of exactly one currently-listed filer, <= 2 extra tokens, first token in <= 20 filer names, e.g. `Amazon` -> `AMAZON COM INC`; a SUGGESTION only: written as `prefix_listed_candidate` with the CIK in `match_candidates`, never in `customer_cik`, because a hand audit of 40 such matches found ~25 % wrong, e.g. `Ring LLC` -> Ring Energy), `token` (IDF-weighted token Jaccard >= 0.92 and >= 0.10 above the runner-up), `ambiguous`, `none`, `anonymous`, `government`, `self` (matched the supplier itself; dropped) |
| match_score | 1.0 for exact, the Jaccard for token |
| match_name | the matched name (company_tickers title / EDGAR name / former name) |
| match_candidates | `ambiguous`: comma list of the CIKs tied on the exact key; `prefix_listed_candidate`: the suggested CIK |

Name universe for resolution: `company_tickers.parquet` titles (current listed
filers) + `filer_profiles` current and former EDGAR names (every XBRL filer, live or
dead). Exact matches tied between several CIKs are broken only if exactly one of them
is in company_tickers; otherwise `ambiguous`.
"""

INSIDER_SCHEMA = """# insider (SEC Insider Transactions data sets, Forms 3/4/5)

Producer: `scripts/ingest_equities_xs.py sec-insider-build` from
`raw/sec_bulk/insider/*_form345.zip` (quarterly, 2006q1+; each zip = filings FILED in
that quarter). Two tables, both partitioned `quarter=YYYYqN` by the calendar quarter of
`ts` (aligned with the quarter-aligned equities sealed start).

## transactions/ (`load_insider_transactions`)
Non-derivative table lines (`NONDERIV_TRANS`) with transaction code `P` (open-market
or private purchase) or `S` (open-market or private sale); all other codes (grants
`A`, option exercises `M`, tax withholding `F`, gifts `G`, ...) are excluded.

| column | meaning |
|---|---|
| adsh | accession of the Form 4/5 (or amendment) |
| issuer_cik, issuer_name, issuer_symbol | issuer as reported on the form (`ISSUERTRADINGSYMBOL` is as typed by the filer) |
| doc_type | `4`, `4/A`, `5`, `5/A` (3 has no transactions) |
| period_of_report | form's period of report |
| owner_id | smallest reporting-owner CIK of the filing (joint filers collapsed) |
| owner_ciks, owner_names, n_owners | all reporting owners of the filing |
| is_director, is_officer, is_ten_pct, is_other | any owner with that relationship |
| officer_title | owner title(s) |
| security_title | as reported (common stock, class A, units, ...) |
| trans_date | transaction date (NOT the availability date) |
| trans_code, acq_disp | `P`/`S`; `A` acquired / `D` disposed |
| shares, price, dollars | shares, price per share, shares x price (NaN if price missing) |
| dollars_suspect | price > 1e6 or dollars > 1e9 on one line (mostly mis-keyed shell-company filings; excluded from monthly dollar sums) |
| shares_after | shares owned following the transaction |
| direct_indirect | `D` / `I` |
| equity_swap, late | equity-swap flag; `TRANS_TIMELINESS = L` (late filing) |
| filed | filing date (date only) |
| accepted | EDGAR acceptance stamp (UTC) from the issuer's / owner's submissions JSON, when the accession is listed there (late by the ET UTC offset on a minority of filings, never early) |
| ts_source | `accepted` or `filed_2200ET_bound` |
| ts | PIT availability (UTC): the earlier of `accepted` and 22:00 US/Eastern of `filed` (EDGAR's last acceptance time); both are upper bounds on the true acceptance |
| dataset | source zip quarter |

## monthly/ (`load_insider_monthly`)
Per issuer `cik` and `month` (calendar month of `ts`, i.e. of filing availability, NOT
of the transaction): `buy_shares, sell_shares, buy_dollars, sell_dollars,
net_shares, net_dollars` (P minus S), `n_buy_tx, n_sell_tx` (transaction lines),
`n_buy_filings, n_sell_filings`, `n_insiders_buying, n_insiders_selling` (distinct
`owner_id`), `n_missing_price`, `n_dollars_suspect` (dollar sums skip missing-price
and suspect lines; share sums keep every line). `ts` = last instant of the month (UTC): the full-month
row is knowable only then. Months without any P/S filing are absent (not zero rows).
"""


def _write_q_parts(df: pd.DataFrame, root: Path, part_name: str) -> None:
    """Partition by the calendar quarter of ``ts`` (``quarter=YYYYqN``): quarter
    boundaries align with the (quarter-aligned) equities sealed start, so no partition
    straddles it after the 2026-10-11 zone change (sealed from 2024-07-01)."""
    t = pd.to_datetime(df["ts"], utc=True)
    lab = t.dt.year.astype(str) + "q" + t.dt.quarter.astype(str)
    _write_partitioned(df, root, "quarter", lab, part_name=part_name)


def _year_key(t: pd.Timestamp) -> str:
    """Stats label: the year, or ``YYYY<boundary`` for the sealed-boundary year."""
    s = SEALED.tz_localize(None)
    return f"{t.year}<{s.date()}" if t.year == s.year and s != pd.Timestamp(f"{s.year}-01-01") \
        else str(t.year)


def cmd_sec_links_build(a: argparse.Namespace) -> None:
    """Customer-supplier edges from every downloaded Notes zip (``--workers`` processes),
    acceptance from our submissions-based ``sub`` (fallback FSN ``sub.accepted``),
    conservative name -> CIK resolution, written to ``links/customer_supplier/``."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    t0 = time.monotonic()
    done = {r["stored_as"] for r in _bulk_records("file").values() if r["item"] == "fsds_notes"}
    files = [BULK / d for d in sorted(done) if (BULK / d).exists()]
    if a.limit:
        files = files[: a.limit]
    print(f"sec-links-build: {len(files)} notes zips, {a.workers} workers", flush=True)
    if a.dry_run:
        return
    edges, metas, fsn_acc = [], [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(xs.extract_customer_edges_from_notes_zip, f): f for f in files}
        for i, fut in enumerate(as_completed(futs)):
            e, meta = fut.result()
            fsn_acc.append(meta.pop("_sub_accepted"))
            edges.append(e)
            metas.append(meta)
            print(f"  {i + 1}/{len(files)} {meta['file']}: {meta.get('n_edges', 0)} edges "
                  f"({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
    e = pd.concat([x for x in edges if len(x)], ignore_index=True)
    n_raw = len(e)
    e = e.drop_duplicates([c for c in xs.EDGE_COLUMNS if c != "dataset"])
    sub = _sub_acceptance()
    e = e.merge(sub.rename(columns={"accepted": "accepted_sub"}), on="adsh", how="left")
    both = e[e["accepted_sub"].notna() & e["accepted_fsn"].notna()
             & (e["accepted_sub"] < SEALED)].drop_duplicates("adsh")
    dmin = (both["accepted_fsn"] - both["accepted_sub"]).dt.total_seconds() / 60.0
    off_y = (dmin.abs() > 2).groupby(pd.to_datetime(both["accepted_sub"], utc=True).dt.year).mean()
    offset = {"filings_compared_presealed": len(both),
              "share_json_shifted_by_year": {int(k): round(float(v), 3) for k, v in off_y.items()},
              "shift_hours_counts": (dmin[dmin.abs() > 2] / 60).round().value_counts().to_dict(),
              "share_abs_diff_le_1min": round(float((dmin.abs() <= 1).mean()), 4) if len(both) else None,
              "share_abs_diff_le_60min": round(float((dmin.abs() <= 60).mean()), 4) if len(both) else None,
              "median_diff_min": float(dmin.median()) if len(both) else None}
    # Three upper bounds on the true acceptance instant, each never earlier than it:
    # the submissions-JSON stamp (late by the ET UTC offset on a minority of filings),
    # FSN sub.accepted (ET wall clock rounded to the minute) + 30 s, and 22:00 ET of the
    # filing date. ts = the earliest available; `accepted` = best point estimate.
    fsn_up = e["accepted_fsn"] + pd.Timedelta(seconds=30)
    bound = xs.edgar_accept_after_hours_bound(e["filed"])
    agree = (e["accepted_sub"] - e["accepted_fsn"]).abs() <= pd.Timedelta(minutes=2)
    # audit of the submissions-JSON acceptance stamp over EVERY FSN filing (pre-sealed)
    fa = pd.concat(fsn_acc, ignore_index=True).drop_duplicates("adsh").merge(sub, on="adsh")
    fa = fa[fa["accepted"].notna() & fa["accepted_fsn"].notna() & (fa["accepted"] < SEALED)]
    dh = (fa["accepted"] - fa["accepted_fsn"]).dt.total_seconds() / 3600.0
    sh = dh.abs() > 2 / 60
    fy = pd.to_datetime(fa["accepted"], utc=True).dt.tz_localize(None).map(_year_key)
    stamp_audit = {
        "filings_compared": len(fa), "share_json_shifted": round(float(sh.mean()), 4),
        "shift_hours_counts": dh[sh].round().value_counts().to_dict(),
        "share_shifted_by_year": {k: round(float(v), 3) for k, v in sh.groupby(fy).mean().items()},
        "share_shifted_by_form": {k: round(float(v), 3) for k, v in
                                  sh.groupby(fa["form"]).mean().items()
                                  if (fa["form"] == k).sum() >= 500},
        "json_earlier_than_fsn_by_gt_2min": int((dh < -2 / 60).sum()),
    }
    e["accepted"] = e["accepted_sub"].where(agree, e["accepted_fsn"])
    e["accepted"] = e["accepted"].where(e["accepted"].notna(), e["accepted_sub"])
    src = np.select([agree, e["accepted_fsn"].notna() & e["accepted_sub"].notna(),
                     e["accepted_fsn"].notna(), e["accepted_sub"].notna()],
                    ["submissions_json=fsn", "fsn_sub(json_shifted)", "fsn_sub",
                     "submissions_json_unverified"], default="none")
    e["accepted_source"] = pd.array(src, dtype="string")
    ups = pd.concat([e["accepted_sub"], fsn_up, bound], axis=1)
    e["ts"] = ups.min(axis=1)
    e = e.drop(columns=["accepted_sub"])
    n_no_ts = int(e["ts"].isna().sum())
    e = e[e["ts"].notna()].reset_index(drop=True)
    ct = pd.read_parquet(ROOT / "company_tickers.parquet")
    prof = pd.read_parquet(ROOT / "filer_profiles.parquet")
    former = []
    for cik, fn in zip(prof["cik"], prof["former_names"].fillna(""), strict=True):
        for part in str(fn).split(";"):
            nm = part.split("|", 1)[0].strip()
            if nm:
                former.append((int(cik), nm))
    names = pd.concat([ct.loc[:, ["cik", "title"]].rename(columns={"title": "name"}),
                       prof.loc[:, ["cik", "name"]],
                       pd.DataFrame(former, columns=["cik", "name"])], ignore_index=True)
    names = names[names["name"].fillna("").str.strip() != ""].drop_duplicates()
    current = pd.concat([ct.loc[:, ["cik", "title"]].rename(columns={"title": "name"}),
                         prof.loc[:, ["cik", "name"]]], ignore_index=True)
    current = current[current["name"].fillna("").str.strip() != ""].drop_duplicates()
    index = xs.NameIndex.build(names, has_ticker=ct["cik"], current_names=current)
    e = xs.resolve_customers(e, index)
    out = xs.LINKS_ROOT
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    _write_q_parts(e, out, "part-0000.parquet")
    (out / "schema.md").write_text(LINKS_SCHEMA)
    # ---- stats: pre-2022 numbers; >= 2022 as a total count only
    ts = pd.to_datetime(e["ts"], utc=True)
    pre = e[ts < SEALED].copy()
    pre["y"] = pd.to_datetime(pre["ts"], utc=True).dt.tz_localize(None).map(_year_key)
    named = ~pre["anonymous"] & ~pre["government"]
    conc = pre["tag"].isin(xs.CONCENTRATION_TAGS)
    by_year = {}
    for y, g in pre.groupby("y"):
        nm = ~g["anonymous"] & ~g["government"]
        by_year[str(y)] = {
            "edges": len(g), "concentration_pct_edges": int(g["tag"].isin(xs.CONCENTRATION_TAGS).sum()),
            "revenue_benchmark_pct_edges": int((g["tag"].isin(xs.CONCENTRATION_TAGS)
                                                & g["is_revenue_benchmark"]).sum()),
            "suppliers": int(g["supplier_cik"].nunique()), "named_edges": int(nm.sum()),
            "resolved_edges": int(g["customer_cik"].notna().sum()),
            "resolved_share_of_named": round(float(g.loc[nm, "customer_cik"].notna().mean()), 4)
            if nm.any() else None,
            "distinct_resolved_pairs": int(g.dropna(subset=["customer_cik"])
                                           .drop_duplicates(["supplier_cik", "customer_cik"]).shape[0]),
        }
    res = pre.dropna(subset=["customer_cik"])
    top = (res.groupby(["customer_cik", "match_name"])["supplier_cik"].nunique()
           .sort_values(ascending=False).head(25))
    dn = pre.loc[named].drop_duplicates("customer_name")
    st = {
        "built_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        "zips": len(files), "rows_extracted": n_raw, "rows_after_dedup": len(e) + n_no_ts,
        "rows_dropped_no_ts": n_no_ts,
        "rows_presealed": len(pre), "rows_sealed_count_only": int((ts >= SEALED).sum()),
        "presealed": {
            "concentration_pct_rows": int(conc.sum()),
            "amount_rows": int((~conc).sum()),
            "anonymous_share": round(float(pre["anonymous"].mean()), 4),
            "government_share": round(float(pre["government"].mean()), 4),
            "named_rows": int(named.sum()),
            "resolved_rows": int(pre["customer_cik"].notna().sum()),
            "resolved_share_of_named_rows": round(float(pre.loc[named, "customer_cik"].notna().mean()), 4),
            "distinct_named_customer_strings": len(dn),
            "distinct_named_strings_resolved": int(dn["customer_cik"].notna().sum()),
            "match_methods": pre["match_method"].value_counts().to_dict(),
            "accepted_source": pre["accepted_source"].value_counts().to_dict(),
            "vendor_hint_rows": int(pre["vendor_hint"].sum()),
            "pct_scale_suspect_rows": int(pre["pct_scale_suspect"].sum()),
            "coreg_rows": int((pre["coreg"].fillna("") != "").sum()),
            "multi_axis_rows": int((pre["n_axes"] > 1).sum()),
            "customer_axes_top": pre["customer_axis"].value_counts().head(15).to_dict(),
            "by_year": by_year,
            "top_customers_by_distinct_suppliers": [
                {"customer_cik": int(c), "name": n, "suppliers": int(v)} for (c, n), v in top.items()],
        },
        "fsn_accepted_vs_submissions_json": offset,
        "submissions_json_stamp_audit_all_fsn_filings": stamp_audit,
        "per_zip": sorted(metas, key=lambda m: xs.period_bounds(m["dataset"])[0]
                          if re.match(r"^\d{4}[qm]", m["dataset"]) else pd.Timestamp(0)),
        "elapsed_min": round((time.monotonic() - t0) / 60, 1),
    }
    (out / "stats.json").write_text(json.dumps(st, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in st.items() if k != "per_zip"}, indent=1, default=str)[:5000],
          flush=True)
    _status_update("sec_links_build", rows=len(e), rows_presealed=len(pre),
                   elapsed_s=round(time.monotonic() - t0, 1), parquet_bytes=_du(out))


def cmd_sec_insider_build(a: argparse.Namespace) -> None:
    """Insider P/S transactions + PIT monthly table from every downloaded Form 3/4/5 zip
    (``--workers`` processes). Acceptance instants come from the raw submissions JSONs of
    the per-CIK fetch (Form 3/4/5 listed under the issuer or owner CIK)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    t0 = time.monotonic()
    done = {r["stored_as"] for r in _bulk_records("file").values() if r["item"] == "insider"}
    files = [BULK / d for d in sorted(done) if (BULK / d).exists()]
    if a.limit:
        files = files[: a.limit]
    subs = sorted(SUBD.glob("*.json.gz"))
    print(f"sec-insider-build: {len(files)} insider zips, {len(subs)} submissions files, "
          f"{a.workers} workers", flush=True)
    if a.dry_run:
        return
    batches = [subs[i:i + 400] for i in range(0, len(subs), 400)]
    accs = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(xs.form345_acceptance_from_files, b) for b in batches]):
            accs.append(f.result())
    acc = pd.concat(accs, ignore_index=True).drop_duplicates("adsh")
    acc["adsh"] = acc["adsh"].astype("string[pyarrow]")
    print(f"  form 3/4/5 acceptance map: {len(acc)} accessions "
          f"({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
    out = xs.INSIDER_ROOT
    if out.exists():
        shutil.rmtree(out)
    (out / "transactions").mkdir(parents=True)
    metas, chk = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(xs.insider_transactions_from_zip, f): f for f in files}
        for i, fut in enumerate(as_completed(futs)):
            tx, meta = fut.result()
            tx = xs.attach_insider_acceptance(tx, acc)
            meta["n_no_ts"] = int(tx["ts"].isna().sum())
            tx = tx[tx["ts"].notna()].reset_index(drop=True)
            meta["n_accepted"] = int((tx["ts_source"] == "accepted").sum())
            if len(tx):
                _write_q_parts(tx, out / "transactions", f"part-{meta['dataset']}.parquet")
                a_et = tx["accepted"].dt.tz_convert("America/New_York")
                ok = tx["accepted"].notna() & (tx["accepted"] < SEALED)
                chk.append(pd.DataFrame({
                    "same_day": (a_et.dt.normalize().dt.tz_localize(None) == tx["filed"])[ok],
                    "after_bound": (tx["accepted"] > xs.edgar_accept_after_hours_bound(tx["filed"]))[ok]}))
            metas.append(meta)
            print(f"  {i + 1}/{len(files)} {meta['file']}: {meta['n_ps_rows']} P/S rows, "
                  f"{meta['n_accepted']} with acceptance ({(time.monotonic() - t0) / 60:.1f} min)",
                  flush=True)
    del acc
    # monthly, per ts-quarter partition (a month never spans two ts quarters)
    (out / "monthly").mkdir(parents=True)
    cols = ["adsh", "issuer_cik", "owner_id", "trans_code", "shares", "price", "dollars",
            "dollars_suspect", "ts"]
    qdirs = sorted((out / "transactions").glob("quarter=*"),
                   key=lambda p: xs.period_bounds(p.name.split("=")[1])[0])
    st_years: dict[str, dict[str, Any]] = {}
    n_tx_total = n_m_total = 0
    pre_tx = []
    for qd in qdirs:
        lab = qd.name.split("=")[1]
        qs, qe = xs.period_bounds(lab)
        tx = pd.concat([xs._ingest_only_read(p, columns=cols) for p in sorted(qd.glob("*.parquet"))],
                       ignore_index=True)
        m = xs.insider_monthly(tx)
        _write_q_parts(m, out / "monthly", "part-0000.parquet")
        n_tx_total += len(tx)
        n_m_total += len(m)
        if qe.tz_localize("UTC") <= SEALED:
            y = _year_key(qs)
            d = st_years.setdefault(y, {"ps_rows": 0, "purchases": 0, "sales": 0, "filings": 0,
                                        "issuer_quarters": 0, "monthly_rows": 0,
                                        "issuer_months_with_buying": 0, "buy_dollars_bn": 0.0,
                                        "sell_dollars_bn": 0.0})
            d["ps_rows"] += len(tx)
            d["purchases"] += int((tx["trans_code"] == "P").sum())
            d["sales"] += int((tx["trans_code"] == "S").sum())
            d["filings"] += int(tx["adsh"].nunique())
            d["issuer_quarters"] += int(tx["issuer_cik"].nunique())
            d["monthly_rows"] += len(m)
            d["issuer_months_with_buying"] += int((m["n_buy_tx"] > 0).sum())
            d["buy_dollars_bn"] = round(d["buy_dollars_bn"] + float(m["buy_dollars"].sum()) / 1e9, 2)
            d["sell_dollars_bn"] = round(d["sell_dollars_bn"]
                                         + float(m["sell_dollars"].sum()) / 1e9, 2)
            pre_tx.extend(xs._ingest_only_read(p, columns=[
                "ts_source", "is_director", "is_officer", "is_ten_pct", "price", "late",
                "doc_type", "dollars_suspect", "ts"]) for p in sorted(qd.glob("*.parquet")))
    pt = pd.concat(pre_tx, ignore_index=True) if pre_tx else pd.DataFrame()
    c = pd.concat(chk, ignore_index=True) if chk else pd.DataFrame({"same_day": [], "after_bound": []})
    (out / "schema.md").write_text(INSIDER_SCHEMA)
    st = {
        "built_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        "zips": len(files), "ps_rows_total": n_tx_total, "monthly_rows_total": n_m_total,
        "presealed": {
            "ps_rows": len(pt),
            "ts_source": pt["ts_source"].value_counts().to_dict() if len(pt) else {},
            "share_director": round(float(pt["is_director"].mean()), 4) if len(pt) else None,
            "share_officer": round(float(pt["is_officer"].mean()), 4) if len(pt) else None,
            "share_ten_pct": round(float(pt["is_ten_pct"].mean()), 4) if len(pt) else None,
            "share_missing_price": round(float(pt["price"].isna().mean()), 4) if len(pt) else None,
            "share_late": round(float(pt["late"].mean()), 4) if len(pt) else None,
            "dollars_suspect_rows": int(pt["dollars_suspect"].sum()) if len(pt) else 0,
            "doc_type": pt["doc_type"].value_counts().to_dict() if len(pt) else {},
            "acceptance_et_date_equals_filed": round(float(c["same_day"].mean()), 4) if len(c) else None,
            "acceptance_after_2200ET_bound": int(c["after_bound"].sum()),
            "by_year": st_years,
        },
        "per_zip": sorted(metas, key=lambda m: m["dataset"]),
        "elapsed_min": round((time.monotonic() - t0) / 60, 1),
    }
    (out / "stats.json").write_text(json.dumps(st, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in st.items() if k != "per_zip"}, indent=1, default=str)[:5000],
          flush=True)
    _status_update("sec_insider_build", ps_rows=n_tx_total, monthly_rows=n_m_total,
                   elapsed_s=round(time.monotonic() - t0, 1), parquet_bytes=_du(out))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["nasdaq", "stooq", "yfinance", "sec-universe", "sec-fetch",
                                    "sec-build", "ff", "fred", "stats", "sec-bulk-fetch",
                                    "sec-links-build", "sec-insider-build"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6, help="sec-fetch worker threads")
    ap.add_argument("--max-rps", type=float, default=8.0,
                    help="sec-fetch process-wide request cap (SEC fair access: 10/s)")
    ap.add_argument("--retry-errors", action="store_true",
                    help="sec-fetch: only re-fetch CIKs listed in fetch_errors.jsonl")
    ap.add_argument("--items", default="",
                    help="sec-bulk-fetch: comma list of " + ",".join(BULK_ITEMS))
    ap.add_argument("--inventory-only", action="store_true")
    ap.add_argument("--reinventory", action="store_true", help="re-HEAD every target")
    ap.add_argument("--refresh-current", action="store_true",
                    help="re-fetch the current quarter's index / current month's notes")
    ap.add_argument("--notes-since", default="2009q2")
    ap.add_argument("--insider-since", default="2006q1")
    ap.add_argument("--index-since", default="2004q1")
    ap.add_argument("--max-mem-gb", type=float, default=3.0,
                    help="sec-links-build / sec-insider-build: soft memory budget")
    a = ap.parse_args()
    _guard_volume()
    ROOT.mkdir(parents=True, exist_ok=True)
    {"nasdaq": cmd_nasdaq, "stooq": cmd_stooq, "yfinance": cmd_yfinance,
     "sec-universe": cmd_sec_universe, "sec-fetch": cmd_sec_fetch, "sec-build": cmd_sec_build,
     "ff": cmd_ff, "fred": cmd_fred, "stats": cmd_stats,
     "sec-bulk-fetch": cmd_sec_bulk_fetch, "sec-links-build": cmd_sec_links_build,
     "sec-insider-build": cmd_sec_insider_build}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
