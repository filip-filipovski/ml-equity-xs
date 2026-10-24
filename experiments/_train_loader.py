"""TRAIN-bounded loader shared by the ml_meeting scripts (copied from p3_vrp_battery.py).

Rows at or after TRAIN_END never reach memory: daily spot goes through load_zoned_returns(TRAIN); other files go
through safe_load_parquet(max_end=TRAIN_END), and when the tripwire refuses a file (perp / funding) a pyarrow row
filter ts < TRAIN_END is applied at read time and asserted.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "validate"))

from eqxs.data_zones import (  # noqa: E402
    TRAIN_END_DEFAULT,
    DataZone,
    SealedWindowLeakError,
    load_zoned_returns,
    safe_load_parquet,
)

CACHE = ROOT / "data_cache"
TRAIN_END = pd.Timestamp(TRAIN_END_DEFAULT)
LOADER_LOG: list[dict] = []


def _io_filtered_train(path: Path) -> pd.DataFrame:
    schema = pq.read_schema(path)
    col = next((c for c in ("ts", "timestamp") if c in schema.names), None)
    if col is None:
        raise RuntimeError(f"{path}: no ts/timestamp column; refusing")
    typ = schema.field(col).type
    if pa.types.is_timestamp(typ):
        bound = pa.scalar(TRAIN_END if typ.tz else TRAIN_END.tz_localize(None), type=typ)
    elif pa.types.is_integer(typ):
        bound = pa.scalar(int(TRAIN_END.value // 1_000_000), type=typ)  # epoch-ms
    else:
        raise RuntimeError(f"{path}: unsupported ts type {typ}")
    tbl = pq.read_table(path, filters=[(col, "<", bound)])
    assert (pc.sum(pc.greater_equal(tbl[col], bound)).as_py() or 0) == 0, f"row filter leak in {path}"
    return tbl.to_pandas()


def train_load(path: Path, intent: str) -> pd.DataFrame:
    path = Path(path)
    name = path.name
    if path.parent == CACHE and name.startswith("binance_") and name.endswith("_1d.parquet"):
        sym = name[len("binance_"):-len("_1d.parquet")]
        df = load_zoned_returns(DataZone.TRAIN, sym, "1d", caller_intent=intent)
        LOADER_LOG.append({"path": name, "via": "load_zoned_returns(TRAIN)", "max_ts": str(df.index.max())})
        return df.reset_index()
    try:
        df = safe_load_parquet(path, max_end_date=TRAIN_END_DEFAULT, caller_intent=intent)
        via = "safe_load_parquet(max_end=TRAIN_END)"
    except SealedWindowLeakError:
        df = _io_filtered_train(path)
        via = "pyarrow row filter ts<TRAIN_END"
    col = "timestamp" if "timestamp" in df.columns else "ts"
    LOADER_LOG.append({"path": name, "via": via, "max_ts": str(df[col].max())})
    return df


def import_builder(filename: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, ROOT / "scripts" / "validate" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    mod._safe_load_truncated = train_load
    return mod
