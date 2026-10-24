#!/usr/bin/env python
"""P9 E6 — build the equity ML panel through the 2026-10-11 equities TRAIN end (2023-01-01) with the UNMODIFIED
study-13 builder (module constants overridden; outputs to NEW derived files), then prove the 2004-2018 overlap is
identical to the study-13 panel. Pre-declaration: P9_E6_PREDECLARATION.md. No IC is computed here."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"; RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import _train_loader as TL  # noqa: E402  (src/ on sys.path)

from eqxs import equity_ml as ml  # noqa: E402

REPO = TL.ROOT
TRAIN_END_2023 = pd.Timestamp("2023-01-01")
STUDY13_DROPPED_TAGS = ["cost_of_revenue"]
STUDY13_DROPPED_FEATURES = ["gp"]
OUT = RESULTS / "p9_e6_build.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_builder():
    spec = importlib.util.spec_from_file_location("eqml_builder", REPO / "scripts" / "equity_ml_xs_train.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eqml_builder"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    import argparse
    import functools
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-price-filter", action="store_true",
                    help="addendum 1 / C1b: universe without the $5 raw-close filter (min_price=0; the split-invariant "
                         "$1M dollar-volume filter is kept) -> ml_panel_train2023_nopx.*; no overlap-identity check")
    ap.add_argument("--wfa", action="store_true",
                    help="addendum B1 / Phase B: build to the equities WFA end 2024-07-01 -> ml_panel_wfa2024{,_nopx}.*; "
                         "the builder's two year loops are extended by one year so the partial 2024 loads (loaders "
                         "still receive end=2024-07-01 and are zone-gated); no overlap check")
    a = ap.parse_args()
    B = load_builder()
    D = B.DERIVED
    old_panel, old_meta = D / "ml_panel_train.parquet", D / "ml_panel_train.meta.json"
    suffix = "_nopx" if a.no_price_filter else ""
    if a.no_price_filter:  # declared one-line override; every other line of the builder unchanged
        B.ml.build_universe = functools.partial(B.ml.build_universe, min_price=0.0)
        log("build_universe overridden: min_price=0.0 (no raw-close price filter)")
    end = pd.Timestamp("2024-07-01") if a.wfa else TRAIN_END_2023
    stem = "ml_panel_wfa2024" if a.wfa else "ml_panel_train2023"
    if a.wfa:
        # declared override (B1): the builder's loops stop at TRAIN_END.year and would drop the partial year 2024.
        xs_ = B.xs; ml_ = B.ml; np_ = B.np; pd_ = B.pd; log_ = B.log

        def load_prices_wfa():
            uc = xs_.load_static("universe_current")
            common = uc[uc["security_type"] == "common"]
            yf_of = dict(zip(common["symbol"], common["yf_symbol"], strict=False))
            keep = set(common["yf_symbol"])
            cat = pd_.CategoricalDtype(sorted(keep))
            parts = []
            for y in range(B.WARMUP_YEAR, B.TRAIN_END.year + 1):  # +1: partial final year
                e_ = min(pd_.Timestamp(f"{y + 1}-01-01"), B.TRAIN_END)
                px = xs_.load_yf_daily(columns=["symbol", "date", "close", "adj_close", "volume", "splits"],
                                       start=f"{y}-01-01", end=e_)
                px = px[px["symbol"].isin(keep)]
                px = px.assign(date=pd_.to_datetime(px["date"]), symbol=px["symbol"].astype(cat))
                for c in ("close", "adj_close", "volume", "splits"):
                    px[c] = px[c].astype(np_.float32) if c != "volume" else px[c].astype(np_.float64)
                parts.append(px)
                log_(f"prices {y}: {len(px):,} bars")
            px = pd_.concat(parts, ignore_index=True)
            del parts
            assert px["date"].max() < B.TRAIN_END
            codes = px["symbol"].cat.codes.to_numpy()
            present = np_.unique(codes)
            symbols = np_.asarray(cat.categories)[present]
            sessions = pd_.DatetimeIndex(np_.unique(px["date"].to_numpy()))
            row = sessions.get_indexer(px["date"])
            col = np_.searchsorted(present, codes)
            pm = ml_.build_price_matrices(sessions, symbols, row, col, px["close"].to_numpy(np_.float64),
                                          px["adj_close"].to_numpy(np_.float64), px["volume"].to_numpy(np_.float64),
                                          px["splits"].to_numpy(np_.float64))
            log_(f"price matrices: {pm.n_rows} sessions x {pm.n_cols} symbols, {len(px):,} bars, "
                 f"{pm.n_guarded} guarded returns")
            return pm, yf_of

        def load_fundamentals_wfa(ciks):
            parts = []
            n_raw = 0
            for y in range(2008, B.TRAIN_END.year + 1):  # +1: partial final year
                for qn in range(4):
                    s = pd_.Timestamp(year=y, month=3 * qn + 1, day=1)
                    if s >= B.TRAIN_END:
                        continue
                    e = min(s + pd_.DateOffset(months=3), B.TRAIN_END)
                    num = xs_.load_sec_num(tags=list(ml_.ALL_SEC_TAGS), ciks=sorted(ciks), start=s, end=e)
                    n_raw += len(num)
                    if len(num):
                        f = ml_.prepare_facts(num)
                        f = f[f["uom"].isin(["USD", "shares", "USD/shares"])]
                        parts.append(f)
                    del num
                log_(f"fundamentals: availability year {y} read ({n_raw:,} raw rows so far)")
            f = pd_.concat(parts, ignore_index=True)
            del parts
            assert f["ts"].max() < B.TRAIN_END.tz_localize("UTC").value
            out = ml_.group_facts(f)
            diag = {"raw_rows": int(n_raw), "prepared_rows": int(len(f)),
                    "rows_by_group": {g: int(len(d)) for g, d in out.items()}}
            return out, diag

        B.load_prices = load_prices_wfa
        B.load_fundamentals = load_fundamentals_wfa
        log("B1 override: load_prices / load_fundamentals year loops extended to include the partial final year")
    # --- override constants: end date + NEW output paths (study-13 cache untouched) ---
    B.TRAIN_END = end
    B.PANEL_PATH = D / f"{stem}{suffix}.parquet"
    B.MONTHS_PATH = D / f"{stem.replace('panel', 'months')}{suffix}.parquet"
    B.META_PATH = D / f"{stem}{suffix}.meta.json"
    B.SCHEMA_PATH = D / f"{stem}{suffix}.schema.md"
    assert B.PANEL_PATH != old_panel
    log(f"builder {B.RUN_VERSION}; TRAIN_END overridden -> {B.TRAIN_END.date()}; out -> {B.PANEL_PATH.name}")
    t0 = time.time()
    panel, months, meta = B.build_panel()
    log(f"built {len(panel):,} rows, {meta['n_dates']} dates, {time.time() - t0:.0f}s")
    assert pd.to_datetime(panel["ts"]).max() < end.tz_localize("UTC")
    # B3: a year partition that straddles the sealed start is refused whole by the loaders (nothing sealed is
    # ever opened), so a build to 2024-07-01 legitimately ends at the last 2023 formation date.
    assert pd.to_datetime(panel["ts"]).max() >= (end - pd.DateOffset(months=9)).tz_localize("UTC")
    # --- pin the feature set to study 13 (report any coverage-rule difference) ---
    built_drop = {"tags": meta["dropped_tags"], "features": meta["dropped_features"]}
    pinned = built_drop["features"] == STUDY13_DROPPED_FEATURES
    meta["dropped_tags_as_built"], meta["dropped_features_as_built"] = built_drop["tags"], built_drop["features"]
    meta["dropped_tags"], meta["dropped_features"] = STUDY13_DROPPED_TAGS, STUDY13_DROPPED_FEATURES
    meta["feature_set_pinned_to_study13"] = True
    meta["train_end"] = str(end.date())
    meta["no_price_filter"] = bool(a.no_price_filter)
    meta["wfa_build"] = bool(a.wfa)
    B.META_PATH.write_text(json.dumps(meta, indent=1, default=str))
    s = B.SCHEMA_PATH.read_text().replace("2004-01 .. 2018-12; nothing dated 2019-01-01 or later",
                                          f"2004-01 .. {(end - pd.DateOffset(months=1)).strftime('%Y-%m')} (P9 E6/B extension); "
                                          f"nothing dated {end.date()} or later")
    B.SCHEMA_PATH.write_text(s)
    log(f"dropped as built {built_drop}; pinned to study 13 {STUDY13_DROPPED_FEATURES} (same={pinned})")

    if a.wfa:
        new = TL.train_load(B.PANEL_PATH, "p9 B wfa build check") if False else pd.read_parquet(B.PANEL_PATH, columns=["date", "ts", "symbol", "date_id"])
        # (plain read of our own just-written file; max ts asserted above < 2024-07-01)
        ref = pd.read_parquet(D / f"ml_panel_train2023{suffix}.parquet", columns=["date", "symbol"])
        pre = new[new["date"] < pd.Timestamp("2023-01-01")]
        k = ["date", "symbol"]
        m = ref.merge(pre[k], on=k, how="outer", indicator=True)["_merge"].value_counts().to_dict()
        outp = {"rows": int(len(new)), "n_dates": int(new["date_id"].nunique()), "max_ts": str(new["ts"].max()),
                "wfa_rows_2023_2024": int((new["date"] >= pd.Timestamp("2023-01-01")).sum()),
                "pre2023_vs_train2023_panel": {kk: int(v) for kk, v in m.items()},
                "build_seconds": meta["build_seconds"], "feature_drop_as_built": built_drop}
        (RESULTS / f"p9_b_build{suffix}.json").write_text(json.dumps(outp, indent=1, default=str))
        log(f"WFA panel: {len(new):,} rows, max ts {new['ts'].max()}, WFA rows {outp['wfa_rows_2023_2024']:,}; "
            f"pre-2023 membership vs train2023 panel: {m}")
        return

    if a.no_price_filter:
        # sanity vs the price-filtered 2023 panel: its ALL universe must be a subset of this one
        px = TL.train_load(D / "ml_panel_train2023.parquet", "p9 e6 nopx check: price-filtered panel")
        new = TL.train_load(B.PANEL_PATH, "p9 e6 nopx check: nopx panel")
        k = ["date", "symbol"]
        missing = px[k].merge(new[k], on=k, how="left", indicator=True)["_merge"].eq("left_only").sum()
        outp = {"px_rows": int(len(px)), "nopx_rows": int(len(new)), "px_rows_missing_from_nopx": int(missing),
                "extra_rows_nopx": int(len(new) - len(px)), "new_panel_max_ts": str(new["ts"].max()),
                "build_seconds": meta["build_seconds"], "feature_drop_as_built": built_drop,
                "loader_log": TL.LOADER_LOG[-2:]}
        (RESULTS / "p9_e6_build_nopx.json").write_text(json.dumps(outp, indent=1, default=str))
        log(f"nopx: {len(new):,} rows vs px {len(px):,}; px rows missing from nopx {int(missing)} (must be 0); "
            f"extra {len(new) - len(px):,}")
        assert missing == 0
        return

    # --- overlap identity vs the study-13 panel (both via the TRAIN loader) ---
    old = TL.train_load(old_panel, "p9 e6 overlap check: study-13 panel")
    new = TL.train_load(B.PANEL_PATH, "p9 e6 overlap check: 2023 panel")
    new_ov = new[new["date"] < pd.Timestamp("2019-01-01")]
    key = ["date", "symbol"]
    m = old.merge(new_ov, on=key, how="outer", suffixes=("_o", "_n"), indicator=True)
    counts = m["_merge"].value_counts().to_dict()
    both = m[m["_merge"] == "both"]
    feats = [f for f in ml.NUMERIC_FEATURES if f not in STUDY13_DROPPED_FEATURES]
    cols = feats + [f"miss_{f}" for f in feats] + list(ml.SIC_COLS) + \
        ["y_raw", "y_rank", "y_ex_all", "y_ex_liq", "in_liq", "target_mi", "cik"]
    report, worst = {}, 1.0
    for c in cols:
        a, b = both[f"{c}_o"], both[f"{c}_n"]
        if c in ("in_liq",):
            eq = (a.astype(bool).to_numpy() == b.astype(bool).to_numpy())
        elif c in ("target_mi", "cik"):
            eq = ((a.isna() & b.isna()) | (a == b)).to_numpy()
        else:
            av, bv = a.to_numpy(np.float64), b.to_numpy(np.float64)
            eq = (np.isnan(av) & np.isnan(bv)) | np.isclose(av, bv, rtol=0, atol=1e-6, equal_nan=False)
        share = float(np.mean(eq))
        report[c] = {"share_equal": share, "n_diff": int((~eq).sum())}
        worst = min(worst, share)
    pass_ = (counts.get("left_only", 0) == 0) and worst >= 0.999
    # attribute residual diffs: rows whose cik differs explain feature diffs?
    cik_diff = ~(((both["cik_o"].isna() & both["cik_n"].isna()) | (both["cik_o"] == both["cik_n"])).to_numpy())
    out = {"old_rows": int(len(old)), "new_rows_total": int(len(new)), "new_rows_overlap": int(len(new_ov)),
           "merge_counts": {k: int(v) for k, v in counts.items()}, "per_column": report,
           "worst_share_equal": worst, "rows_with_cik_change": int(cik_diff.sum()),
           "feature_drop_as_built": built_drop, "pinned_same_as_study13": pinned,
           "new_panel_max_ts": str(new["ts"].max()), "new_n_dates": int(new["date_id"].nunique()),
           "fresh_rows_2019_2022": int((new["date"] >= pd.Timestamp("2019-01-01")).sum()),
           "build_seconds": meta["build_seconds"], "overlap_identity_pass": bool(pass_),
           "loader_log": TL.LOADER_LOG[-2:]}
    OUT.write_text(json.dumps(out, indent=1, default=str))
    log(f"overlap: merge {counts}; worst column share_equal {worst:.5f}; cik changes {int(cik_diff.sum())}; "
        f"fresh rows {out['fresh_rows_2019_2022']:,}; PASS={pass_}")
    if not pass_:
        bad = {c: r for c, r in report.items() if r["share_equal"] < 0.999}
        log(f"NOT identical on: {bad}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
