#!/usr/bin/env python
"""P12c step 1 — rebuild study 14's W / M / Q panels through 2022-12 with the study-14 builder itself, patched only in its
bounds and output directory (P12_PREDECLARATION.md addendum 4). Writes data_cache/equities_xs/derived/eqml_grid_2023/."""
from __future__ import annotations
import importlib.util, sys, time
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO / "src"))
spec = importlib.util.spec_from_file_location("eqgrid_driver", REPO / "scripts/validate/equity_ml_window_grid_train.py")
drv = importlib.util.module_from_spec(spec); sys.modules["eqgrid_driver"] = drv; spec.loader.exec_module(drv)
drv.TRAIN_END = pd.Timestamp("2023-01-01"); drv.GRID = drv.DERIVED / "eqml_grid_2023"
drv.CELLS_DIR = drv.GRID / "cells"; drv.PRED_DIR = drv.GRID / "predictions"; drv.PORT_DIR = drv.GRID / "portfolios"
drv.g.OOS_LAST_YEAR = 2022
print(f"[{time.strftime('%H:%M:%S')}] P12c build: TRAIN_END {drv.TRAIN_END.date()} -> {drv.GRID}", flush=True)
assert not (drv.GRID / "panel_W.parquet").exists(), "eqml_grid_2023 already has panels; refusing to overwrite"
meta = drv.build_panels()
print(f"[{time.strftime('%H:%M:%S')}] done:", {h: (m.get("n_rows"), m.get("n_dates"), str(m.get("last_formation"))[:10]) for h, m in meta.items()}, flush=True)
