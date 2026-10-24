#!/usr/bin/env python3
"""Build the 8-K item-code table from the already-fetched EDGAR submissions pages.

Offline: reads ``data_cache/equities_xs/raw/sec/submissions/*.json.gz`` (20,637 files,
fetched 2026-10-10), writes ``data_cache/equities_xs/events/eightk_items/quarter=YYYYqN/
part.parquet`` + ``stats.json`` + ``schema.md``. No network.

    python scripts/build_eightk_items.py --workers 3

Sealed discipline: the stats are counts by year (no row content); the loader
(``sec_eightk_items.load_eightk_items``) never opens a partition past the equities
sealed start without an unlock token.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from eqxs import sec_eightk_items as ek  # noqa: E402

SCHEMA = """# eightk_items schema

One row per (cik, accession) for forms 8-K and 8-K/A, from EDGAR submissions pages
(main ``filings.recent`` + paginated pages). Partition ``quarter=YYYYqN`` = calendar quarter of
``ts``.

| column | type | meaning |
|---|---|---|
| cik | int64 | filer CIK (from the file name; co-registrant filings appear once per CIK) |
| accession | string | EDGAR accession number |
| form | string | ``8-K`` or ``8-K/A`` |
| filed | datetime64 (naive date) | EDGAR filing date |
| accepted | datetime64[UTC] | ``acceptanceDateTime`` (verified UTC), null if not served |
| ts | datetime64[UTC] | PIT availability: ``accepted``, else filing date 22:00 US/Eastern |
| ts_source | string | ``acceptance`` / ``filed_2200_et`` |
| period | datetime64 | ``reportDate`` (date of the earliest event reported) |
| items | list<string> | item tokens as served |
| items_raw | string | the served ``items`` string |
| numbering | string | ``modern`` (N.NN codes), ``legacy`` (pre-2004-08-23 integers), ``mixed``, ``none`` |
| n_items | int16 | number of tokens |
| item_X_YY | bool | filing reports item X.YY (modern code, or a legacy code mapped by ``LEGACY_MAP``) |

Legacy map (pre-2004-08-23 -> modern): 1->5.01, 2->2.01, 3->1.03, 4->4.01, 5->8.01, 6->5.02,
7->9.01, 8->5.03, 9->7.01, 10->5.05, 11->5.04, 12->2.02. Items 1.02, 2.03-2.06, 3.01, 4.02 have
no legacy equivalent (created 2004), so they are False before 2004-08-23 by construction.
Load with ``eqxs.sec_eightk_items.load_eightk_items(start=..., end=...)``.
"""


def _work(paths: list[str]) -> tuple[pd.DataFrame, list[dict], dict[str, int]]:
    df, errs = ek.parse_files(paths)
    excl: Counter = Counter()
    for p in paths:  # count 8-K variant forms that are excluded (cheap second pass)
        try:
            with gzip.open(p, "rt") as f:
                d = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        b = d["filings"]["recent"] if "filings" in d else d
        for fo in b.get("form") or []:
            fo = str(fo).strip()
            if fo.startswith("8-K") and fo not in ek.FORMS:
                excl[fo] += 1
    return df, errs, dict(excl)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--src", type=Path, default=ek.SUBMISSIONS_DIR)
    ap.add_argument("--out", type=Path, default=ek.EIGHTK_ROOT)
    ap.add_argument("--chunk", type=int, default=400)
    a = ap.parse_args()
    t0 = time.time()
    files = sorted(str(p) for p in a.src.glob("CIK*.json.gz"))
    chunks = [files[i:i + a.chunk] for i in range(0, len(files), a.chunk)]
    frames, errs, excl = [], [], Counter()
    with ProcessPoolExecutor(max_workers=min(a.workers, 3)) as ex:
        for i, (df, er, ec) in enumerate(ex.map(_work, chunks)):
            frames.append(df)
            errs += er
            excl.update(ec)
            if i % 10 == 0:
                print(f"chunk {i + 1}/{len(chunks)} rows so far {sum(len(f) for f in frames)}",
                      flush=True)
    df = ek.dedupe(pd.concat(frames, ignore_index=True))
    parts = ek.write_partitions(df, a.out)
    stats = ek.build_stats(df, excluded_forms=dict(excl), errors=errs)
    stats.update({"source_files": len(files), "partitions": parts,
                  "elapsed_s": round(time.time() - t0, 1), "errors_sample": errs[:20],
                  "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True))
    (a.out / "schema.md").write_text(SCHEMA)
    print(json.dumps({k: stats[k] for k in ("rows", "ciks", "share_with_acceptance_all",
                                            "parse_errors", "elapsed_s")}))


if __name__ == "__main__":
    main()
