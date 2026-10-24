#!/usr/bin/env python
"""P12c step 2 — fixed booster M2 (P9 machinery: expanding window, annual refits 2009-2022, embargo 1 month) on the rebuilt
W / M / Q panels in eqml_grid_2023/. Predictions cached to p12_cache/p12c_{h}.npz (panel row indices + predictions).
TRAIN only: target months <= 2022-12 enforced; formation dates asserted < 2023-01-01."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
RESULTS = HERE.parent / "results"; RESULTS.mkdir(exist_ok=True)
import p9_ml_capacity as P9  # noqa: E402
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
G = P9.TL.CACHE / "equities_xs" / "derived" / "eqml_grid_2023"; CACHE = RESULTS / "p12_cache"; CACHE.mkdir(exist_ok=True)
years = list(range(2009, 2023)); LAST = 2022 * 12 + 11; out = {}
for h in ("M", "Q", "W"):
    t0 = time.time(); P9.PANEL_PATH = G / f"panel_{h}.parquet"; P9.META_PATH = G / f"panel_{h}.meta.json"; P9._PANEL_CACHE = None
    panel = P9.Panel(); panel.valid = panel.valid & (panel.tmi <= LAST)
    dates = pd.to_datetime(P9.TL.train_load(P9.PANEL_PATH, f"p12c {h} dates")["date"]); assert str(dates[panel.valid].max())[:10] < "2023-01-01", dates[panel.valid].max()
    refits = P9.make_refits(years, embargo=1)
    res = P9.fit_predict_cell(panel, panel.X, list(panel.columns), refits, "M2")
    rows = np.concatenate([panel.year_scorer(y).rows for y in years]); pred = np.concatenate([res["pred"][y] for y in years])
    np.savez_compressed(CACHE / f"p12c_{h}.npz", rows=rows, pred=pred)
    sc = P9.score_preds(panel, res, portfolio=False); ic = np.array(sc["ic_all"]); cl = np.array(sc["cluster"])
    dev, chk = ic[cl <= 2018], ic[cl >= 2019]
    out[h] = {"n_rows_fit_panel": int(panel.valid.sum()), "n_pred": int(len(pred)), "ic_dev_mean": float(dev.mean()), "ic_dev_t": float(dev.mean() / dev.std(ddof=1) * np.sqrt(len(dev))), "ic_chk_mean": float(chk.mean()), "ic_chk_t": float(chk.mean() / chk.std(ddof=1) * np.sqrt(len(chk))), "n_dev_periods": int(len(dev)), "n_chk_periods": int(len(chk)), "max_formation_date": str(dates[panel.valid].max())[:10], "wall_s": time.time() - t0}
    log(f"{h}: {len(pred)} predictions; IC dev {dev.mean():+.4f} (t {out[h]['ic_dev_t']:.1f}, n {len(dev)}) chk {chk.mean():+.4f} (t {out[h]['ic_chk_t']:.1f}, n {len(chk)}); {time.time()-t0:.0f} s")
(RESULTS / "p12c_refit.json").write_text(json.dumps(out, indent=1, default=float)); log("wrote p12c_refit.json")
