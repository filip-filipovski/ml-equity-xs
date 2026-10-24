#!/usr/bin/env python3
"""BEA input-output use tables + SIC -> NAICS -> BEA crosswalks (keyless).

    python scripts/ingest_bea_io.py fetch --max-gb 1 --resume
    python scripts/ingest_bea_io.py build

``fetch`` downloads the files in ``bea_io.SOURCES`` (manifest-driven, md5 + size,
one connection, byte cap). ``build`` unpacks nothing to disk: it reads the zips in
place and writes

    data_cache/links/bea/use_summary/year=YYYY/part.parquet   long use table (ts = valid_from)
    data_cache/links/bea/static/*.parquet                     crosswalks (no ts, static)
    data_cache/links/bea/stats.json, schema.md

Every request carries the declared-contact User-Agent. www.census.gov / www2.census.gov
answer 403 (Cloudflare challenge) to that User-Agent (2026-10-11), so the Census
concordance files are not fetched directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import httpx  # noqa: E402

from eqxs import bea_io as bea  # noqa: E402
from eqxs import equities_xs as xs  # noqa: E402

USER_AGENT = xs.SEC_USER_AGENT


# ---------------------------------------------------------------------------
# Manifest-driven, byte-capped, resumable downloader (one connection)
# ---------------------------------------------------------------------------
class ByteCapReached(RuntimeError):
    """Raised before a download that would exceed the byte cap."""


@dataclass
class Manifest:
    """Append-only jsonl manifest. ``kind='download'`` rows carry ``bytes`` / ``md5``."""

    path: Path
    rows: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def open(cls, path: Path) -> Manifest:
        m = cls(path=path)
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    m.rows.append(json.loads(line))
        return m

    def append(self, row: Mapping[str, Any]) -> None:
        r = {"logged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row}
        self.rows.append(r)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    def done(self, url: str) -> dict[str, Any] | None:
        hits = [r for r in self.rows if r.get("kind") == "download" and r.get("url") == url
                and r.get("status") == "ok"]
        return hits[-1] if hits else None

    def bytes_downloaded(self) -> int:
        """Bytes actually transferred (sum over download rows, deletions keep them)."""
        return int(sum(int(r.get("bytes_transferred", 0)) for r in self.rows
                       if r.get("kind") == "download"))


def total_bytes(manifests: Iterable[Manifest]) -> int:
    return int(sum(m.bytes_downloaded() for m in manifests))


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()  # noqa: S324 - integrity fingerprint, not security
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download(client: Any, url: str, dest: Path, manifest: Manifest, *, cap_bytes: int,
             used_bytes: int, expected_size: int | None = None, resume: bool = True,
             extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Stream ``url`` to ``dest`` (via ``dest.part``; HTTP Range resume when the part
    exists). Refuses to start if ``used_bytes + remaining > cap_bytes``. Records a
    manifest row with size, md5, bytes transferred."""
    prev = manifest.done(url)
    if resume and prev and dest.exists() and dest.stat().st_size == prev.get("size"):
        return prev
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if (resume and part.exists()) else 0
    if expected_size is not None and used_bytes + (expected_size - have) > cap_bytes:
        raise ByteCapReached(f"{url}: needs {expected_size - have} B, cap left "
                             f"{cap_bytes - used_bytes} B")
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if have:
        headers["Range"] = f"bytes={have}-"
    transferred = 0
    with client.stream("GET", url, headers=headers) as r:
        if r.status_code == 416:
            pass
        else:
            r.raise_for_status()
            mode = "ab" if (have and r.status_code == 206) else "wb"
            if mode == "wb":
                have = 0
            with part.open(mode) as f:
                for b in r.iter_bytes(1 << 20):
                    if used_bytes + transferred + len(b) > cap_bytes:
                        manifest.append({"kind": "download", "url": url, "dest": str(dest),
                                         "status": "cap_stop", "bytes_transferred": transferred,
                                         **(extra or {})})
                        raise ByteCapReached(f"cap hit during {url}")
                    f.write(b)
                    transferred += len(b)
    part.replace(dest)
    size = dest.stat().st_size
    if expected_size is not None and size != expected_size:
        manifest.append({"kind": "download", "url": url, "dest": str(dest),
                         "status": "size_mismatch",
                         "size": size, "expected_size": expected_size,
                         "bytes_transferred": transferred, **(extra or {})})
        raise OSError(f"{url}: size {size} != expected {expected_size}")
    row = {"kind": "download", "url": url, "dest": str(dest), "status": "ok", "size": size,
           "md5": md5_file(dest), "bytes_transferred": transferred, **(extra or {})}
    manifest.append(row)
    return row


def cmd_fetch(a: argparse.Namespace) -> None:
    m = Manifest.open(bea.MANIFEST)
    others: list[Manifest] = []
    cap = int(a.max_gb * 1e9)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120.0,
                      follow_redirects=True, limits=httpx.Limits(max_connections=1)) as c:
        for name, src in bea.SOURCES.items():
            used = total_bytes([m, *others])
            dest = bea.RAW_DIR / src["file"]
            try:
                head = c.head(src["url"])
                size = int(head.headers.get("content-length", 0)) or None
                lm = head.headers.get("last-modified")
            except httpx.HTTPError:
                size, lm = None, None
            try:
                row = download(c, src["url"], dest, m, cap_bytes=cap, used_bytes=used,
                                  expected_size=size, resume=a.resume,
                                  extra={"source": name, "http_last_modified": lm,
                                         "provenance": src["provenance"]})
                print(name, row.get("size"), row.get("md5"))
            except ByteCapReached as e:
                print("CAP", e)
                break
            except (httpx.HTTPError, OSError) as e:
                m.append({"kind": "download", "url": src["url"], "status": "error",
                          "error": str(e)[:300], "bytes_transferred": 0, "source": name})
                print("ERROR", name, e)


def cmd_build(a: argparse.Namespace) -> None:
    stats = bea.build_all()
    print(json.dumps({k: stats[k] for k in ("use_rows", "years", "vintage") if k in stats},
                     default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--max-gb", type=float, default=80.0)
    f.add_argument("--resume", action="store_true")
    f.set_defaults(fn=cmd_fetch)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
