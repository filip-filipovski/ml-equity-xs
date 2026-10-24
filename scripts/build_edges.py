#!/usr/bin/env python3
"""Study 15: build the E1 / E4 / E5 edge tables, the 200-edge E4 precision sample and the
coverage-gate counts (edge tables + price availability only; no return is read here).

    OMP_NUM_THREADS=3 python scripts/build_edges.py [--skip-e4]

Writes under data_cache/links/edges/: e1_sic_membership.parquet, e4_customer_supplier_edges.parquet,
e4_member_filing_pairs.parquet (every candidate with its drop reason), e4_precision_sample.csv,
e5_bea_use_edges.parquet, e5_sic4_bea_map.parquet, fsn_sub_2009q2_2022q4.parquet, schema.md,
stats.json, coverage_gate.json. Nothing dated 2023-01-01 or later is read.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from eqxs import equity_graph as eg  # noqa: E402

OUT = eg.EDGES_DIR

SCHEMA = """# Study 15 edge tables (`data_cache/links/edges/`)

Built by `scripts/build_edges.py` (module `eqxs.equity_graph`,
edge version `{edge_version}`, resolver `{resolver}`, operator `{operator}`). Pre-declared as study 15 (v2). Every table carries `ts`
(= valid_from, UTC) for the data-zone tripwire; nothing dated 2023-01-01 or later is read or written.
A validity end that falls at or after the computation bound is stored as NaT ("open").
An edge is used at formation date T iff valid_from <= T < valid_to (NaT = open).

## e1_sic_membership.parquet (E1, SIC4 / SIC2 co-membership, undirected)
| column | meaning |
|---|---|
| cik | SEC filer |
| sic4, sic2 | SIC as of the filing (FSN `sub.tsv`, 2009q2-2022q4 Notes data sets) |
| valid_from, valid_to | filing acceptance (FSN `accepted`, US/Eastern -> UTC) to the next filing's acceptance |
| source | `fsn_as_of_filing`; `backfill_first_fsn` (before a CIK's first FSN filing: its first FSN SIC, look-ahead in the SIC label only); `profile_current` (CIKs never in FSN: today's SIC) |
| adsh | the filing |

## e4_customer_supplier_edges.parquet (E4, customer -> supplier)
One row per (supplier 10-K / 10-K/A / 10-KT, resolved customer). `weight` = max `ConcentrationRiskPercentage1`
(or `ConcentrationRiskPercentage`) / 100 over the filing's facts for that member on a customer axis whose
benchmark is revenue (or untagged); explicit non-revenue benchmarks (accounts receivable, payables, ...) are
not a share of revenue and are dropped. Resolver (pinned, binary): member names normalised (lower case,
alphanumerics only, trailing corporate suffixes removed token-wise), matched to a CIK only by an exact normalised
match, or a unique prefix match of >= 6 characters, against SEC filer names VALID AT THE FILING DATE
(`filer_profiles`: the current name from the end of the latest dated former name; dated former names over their
own range; undated names never used). Dropped (see `e4_member_filing_pairs.parquet`, `drop_reason`): anonymous
(ingest flag OR the frozen regex `equity_graph.ANON_START` / `ANON_END`), government / agency / public payer
(ingest flag OR the pinned term list), vendor / supplier (ingest `vendor_hint`), unresolved or resolved to more
than one CIK, the supplier itself, parent / subsidiary (same filer family in FSN `sub.tsv` `aciks`, either
direction, or a shared distinctive first name token: <= 20 filers start with it), and suppliers that are lessors
(SIC 6500-6599, 6798, or a name with "income & growth fund" / "equipment fund" / "leasing fund").
`valid_from` = the acceptance session of the filing from the Notes data set's `accepted` (coordinator
2026-10-11: the submissions-JSON stamp is late by 4-5 h on ~17 % of filings); `valid_to` = the acceptance session
of the supplier's next original 10-K / 10-KT, or 15 months after acceptance, whichever is first
(`valid_to_rule`). `customer_sic4` = the customer's SIC4 at the supplier's filing (E1), used for the N3 strata.

## e5_bea_use_edges.parquet (E5, selling industry -> buying industry)
BEA `IOUse_Before_Redefinitions_PRO_1997-2023_Summary.xlsx` (single vintage, docProps 2024-09-06), 71 summary
industries, data years 1997-2020 only (sheets 2021-2023 never opened). `weight` = commodity i's share of industry
j's total intermediate inputs (T005), DIAGONAL SET TO ZERO; `share_raw` keeps the published share,
`diag_share_of_buyer` the buyer's own-industry share. valid_from = 1 Jan of data year + 2, valid_to = 1 Jan data
year + 3. Rows `Used` / `Other` (scrap, noncomparable imports) are not industries and are dropped.

## e5_sic4_bea_map.parquet (firm -> BEA industry)
Per SIC4: 1997 NAICS with the largest share of the SIC's 1997 Economic Census receipts (bridge `ship_share`),
its BEA summary code by NAICS prefix, `match_level` (only `sic4` = a 1987 SIC of the bridge is kept), `kept` =
level sic4, dominant share >= 0.5 and a unique BEA code. Firms of a dropped SIC are not in E5.

## e4_precision_sample.csv
200 resolved E4 edges drawn at random (numpy seed 15) for the hand check (precision gate >= 0.9), with the
reviewer's labels (`label`: correct / wrong / ambiguous) and notes.

## coverage_gate.json
Per trial: units per formation date over the OOS span (structural reach + availability only), median / min,
the gate (median >= 60 suppliers / 40 industries) and the flat months (< 30 / < 20).
"""


def rss() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main() -> int:
    from eqxs import equities_xs as xs

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-e4", action="store_true")
    ap.add_argument("--gate-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    stats: dict = json.loads((OUT / "stats.json").read_text()) if (OUT / "stats.json").exists() else {}
    if not args.gate_only:
        sub = eg.read_fsn_sub()
        eg._log(f"FSN sub: {len(sub):,} filings, {sub['cik'].nunique():,} CIKs ({time.time() - t0:.0f} s, RSS {rss():.2f} GB)")
        sub.to_parquet(OUT / "fsn_sub_2009q2_2022q4.parquet", index=False)
        profiles = xs.load_static("filer_profiles")
        e1 = eg.build_e1(sub, profiles)
        e1.to_parquet(OUT / "e1_sic_membership.parquet", index=False)
        stats["e1"] = {"rows": int(len(e1)), "ciks": int(e1["cik"].nunique()),
                       "by_source": {k: int(v) for k, v in e1["source"].value_counts().items()},
                       "sha256": eg.frame_hash(e1)}
        e5, st5 = eg.build_e5()
        e5.to_parquet(OUT / "e5_bea_use_edges.parquet", index=False)
        stats["e5"] = {**st5, "rows": int(len(e5)), "sha256": eg.frame_hash(e5)}
        sics = sorted(set(e1["sic4"].dropna().astype(str)) | set(profiles["sic"].astype(str).str.zfill(4)))
        sics = [s for s in sics if len(s) == 4 and s.isdigit()]
        smap = eg.build_sic_bea_map(sics)
        smap["ts"] = pd.Timestamp("1997-01-01", tz="UTC")
        smap.to_parquet(OUT / "e5_sic4_bea_map.parquet", index=False)
        stats["e5_map"] = {"sic_codes": int(len(smap)), "kept": int(smap["kept"].sum()),
                           "drop_reasons": {k: int(v) for k, v in smap.loc[~smap["kept"], "drop_reason"].value_counts().items()},
                           "bea_industries_reached": int(smap.loc[smap["kept"], "bea"].nunique()), "sha256": eg.frame_hash(smap)}
        eg._log(f"E1 {len(e1):,} rows, E5 {len(e5):,} rows, map kept {smap['kept'].sum()}/{len(smap)} SIC codes")
        if not args.skip_e4:
            pb = eg.build_price_base()
            links = xs.load_customer_links(end=eg.TRAIN_END)
            assert pd.to_datetime(links["ts"], utc=True).max() < eg.TRAIN_END.tz_localize("UTC")
            eg._log(f"links: {len(links):,} rows (RSS {rss():.2f} GB)")
            E, st4 = eg.build_e4(links, sub, profiles, pb.sessions, e1)
            F = st4.pop("all_pairs_table")
            F["ts"] = pd.to_datetime(F["accepted_fsn"], utc=True)
            F.to_parquet(OUT / "e4_member_filing_pairs.parquet", index=False)
            E.to_parquet(OUT / "e4_customer_supplier_edges.parquet", index=False)
            del links
            stats["e4"] = {**st4, "sha256": eg.frame_hash(E)}
            rng = np.random.default_rng(15)
            pick = np.sort(rng.choice(len(E), size=min(200, len(E)), replace=False))
            ps = E.iloc[pick].loc[:, ["supplier_cik", "supplier_name", "supplier_sic", "year", "form", "adsh",
                                      "customer_member", "customer_name", "match_method", "matched_name",
                                      "customer_cik", "customer_current_name", "pct"]].reset_index()
            ps = ps.rename(columns={"index": "edge_row"})
            ps["label"] = ""
            ps["note"] = ""
            dst = OUT / "e4_precision_sample.csv"
            if dst.exists():
                old = pd.read_csv(dst)
                if "label" in old and old["label"].notna().any() and list(old["edge_row"]) == list(ps["edge_row"]):
                    ps["label"], ps["note"] = old["label"], old["note"]
            ps.to_csv(dst, index=False)
            eg._log(f"E4: {len(E):,} edges; precision sample {len(ps)} rows -> {dst.relative_to(REPO)}")
        stats.update({"built_utc": pd.Timestamp.now("UTC").isoformat(), "edge_version": eg.EDGE_VERSION,
                      "resolver": eg.RESOLVER_VERSION, "operator": eg.OPERATOR_VERSION, "bound": str(eg.TRAIN_END.date())})
        (OUT / "stats.json").write_text(json.dumps(stats, indent=1, default=str))
        (OUT / "schema.md").write_text(SCHEMA.format(edge_version=eg.EDGE_VERSION, resolver=eg.RESOLVER_VERSION,
                                                     operator=eg.OPERATOR_VERSION))
    # coverage gate (structural reach + price availability; no return is read)
    sd = eg.load_study_data()
    gate: dict = {"computed_utc": pd.Timestamp.now("UTC").isoformat(), "rule": {
        "T7": "median suppliers in ALL with a structurally nonzero hop-1 signal over 2013-01..2022-12 >= 60, else NOT RUN; months < 30 held flat",
        "T10-T12": "median BEA industries with >= 3 priced ALL firms and a structurally nonzero hop-h signal over 2005-01..2022-12 >= 40, else NOT RUN; months < 20 held flat"},
        "cells": {}, "study_data_meta": sd.meta}
    defs = eg.trial_defs()
    for cid, cd in defs.items():
        if cd.edge == "E4" and sd.e4 is None:
            gate["cells"][cid] = {"run": False, "reason": "links table absent"}
            continue
        rows, c = eg.cell_counts(sd, cd)
        rule = eg.coverage_gate(c, gate_median=eg.GATE_MEDIAN[cd.edge], flat_below=eg.FLAT_BELOW[cd.edge])
        tmi = sd.monthly.target_mi(sd.pb.sessions)[rows]
        by_year = pd.Series(c).groupby(tmi // 12).median()
        gate["cells"][cid] = {**rule, "counts": c.tolist(),
                              "target_months": [f"{m // 12}-{m % 12 + 1:02d}" for m in tmi],
                              "median_by_year": {int(k): float(v) for k, v in by_year.items()}}
        eg._log(f"coverage {cid}: median {rule['median']:.0f} (min {rule['min']:.0f}), run {rule['run']}, "
                f"flat months {rule['flat_months']}")
    if sd.e4 is not None:
        for h in (2, 3):
            cd = eg.CellDef(f"D3_H{h}", "E4", h, first_target="2013-01", last_target="2022-12", role="diagnostic")
            rows, c = eg.cell_counts(sd, cd)
            tmi = sd.monthly.target_mi(sd.pb.sessions)[rows]
            gate["cells"][cd.cell_id] = {"median": float(np.median(c)), "min": float(c.min()), "max": float(c.max()),
                                         "book_rule": "quintile book only if the median count >= 30",
                                         "book": bool(np.median(c) >= 30), "counts": c.tolist(),
                                         "median_by_year": {int(k): float(v) for k, v in
                                                            pd.Series(c).groupby(tmi // 12).median().items()},
                                         "mean_by_year": {int(k): float(v) for k, v in
                                                          pd.Series(c).groupby(tmi // 12).mean().items()}}
            eg._log(f"D3 hop {h}: median {np.median(c):.0f}, max {c.max()}")
    (OUT / "coverage_gate.json").write_text(json.dumps(gate, indent=1, default=str))
    eg._log(f"done in {time.time() - t0:.0f} s, peak RSS {rss():.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
