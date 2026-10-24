#!/usr/bin/env python
"""Stitch the model's full history 2009-2023 (14 TRAIN refit-years + the 2023 WFA year) for the decile and the
neutral-cohort books, C1 (A28) and C1b (B23): monthly series -> 15-year SR, MaxDD, per-year. Labelled context, not a
new read (the 2023 part is the Phase B read; 2009-2022 is TRAIN)."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import p9_ml_capacity as P9, p9_e6_fresh_window as E6, p9_b_wfa_oneshot as PB, p10_books as BK  # noqa (PB patches loader)
from eqxs import equity_ml as ml
D = P9.TL.CACHE / "equities_xs" / "derived"
out = {}
for stem, design, key in (("ml_panel_wfa2024", "f28", "C1"), ("ml_panel_wfa2024_nopx", "f23", "C1b")):
    P9.PANEL_PATH = D / f"{stem}.parquet"; P9.META_PATH = D / f"{stem}.meta.json"
    panel = P9.Panel(); df = P9.TL.train_load(P9.PANEL_PATH, "full-history frame")
    cols = list(panel.columns)
    if design == "f23":
        drop = set(E6.PX_FEATS) | {f"miss_{c}" for c in E6.PX_FEATS}
        keep = [i for i, c in enumerate(cols) if c not in drop]; X = np.ascontiguousarray(panel.X[:, keep]); cols = [cols[i] for i in keep]
    else:
        X = panel.X
    years = list(range(2009, 2024))
    res = P9.fit_predict_cell(panel, X, cols, P9.make_refits(years, 1), "M2")
    rows = np.concatenate([panel.year_scorer(y).rows for y in years]); pred = np.concatenate([res["pred"][y] for y in years])
    frame = pd.DataFrame({"tmi": panel.tmi[rows], "firm": panel.firm[rows], "y": panel.y_raw[rows], "in_liq": panel.in_liq[rows],
                          "beta": df["raw_beta_252"].to_numpy(np.float64)[rows], "pred": pred})
    sic_cols = list(ml.SIC_COLS)
    for c in sic_cols:
        frame[c] = panel.X[rows, panel.col_ix[c]]
    sc = P9.score_preds(panel, res, portfolio=False); ic = np.asarray(sc["ic_all"]); cl = np.asarray(sc["cluster"])
    books = {"B0_decile": BK.run_book(frame, sic_cols, "decile", 1, 0.0015, False),
             "B3_neutral_cohort3": BK.run_book(frame, sic_cols, "neutral", 3, 0.0015, False),
             "B4_neutral_cohort3_LIQ": BK.run_book(frame, sic_cols, "neutral", 3, 0.0005, True)}
    out[key] = {"ic_by_year": {int(y): float(ic[cl == y].mean()) for y in years}, "ic_full_mean": float(ic.mean()),
                "ic_full_month_t": float(ic.mean() / ic.std(ddof=1) * math.sqrt(len(ic))), "books": {}}
    for bid, bk in books.items():
        full = BK.series_metrics(bk["gross"], bk["turnover"], bk["cost"], bk["tmi"])
        ex23 = (bk["tmi"] < 2023 * 12)
        pre = BK.series_metrics(bk["gross"][ex23], bk["turnover"][ex23], bk["cost"], bk["tmi"][ex23])
        y23 = ~ex23
        out[key]["books"][bid] = {"full_2009_2023": full, "pre_2009_2022": pre,
                                  "y2023": BK.series_metrics(bk["gross"][y23], bk["turnover"][y23], bk["cost"], bk["tmi"][y23]) if y23.sum() > 6 else None}
        print(f"{key} {bid:22s} 2009-23: net {full['net_ann']*100:+.2f}%/yr SR {full['net_sharpe']:+.2f} DD {full['maxdd']*100:+.1f}% vol {full['ann_vol']*100:.1f}% | "
              f"2009-22: net {pre['net_ann']*100:+.2f}% SR {pre['net_sharpe']:+.2f} DD {pre['maxdd']*100:+.1f}% | 2023: net {out[key]['books'][bid]['y2023']['net_ann']*100:+.1f}% SR {out[key]['books'][bid]['y2023']['net_sharpe']:+.2f}")
    print(f"{key} IC by year:", {y: round(v, 3) for y, v in out[key]['ic_by_year'].items()}, "| full mean", round(out[key]['ic_full_mean'], 4), "month t", round(out[key]['ic_full_month_t'], 2))
(HERE / "p9_b_fullhistory.json").write_text(json.dumps(out, indent=1, default=float))
