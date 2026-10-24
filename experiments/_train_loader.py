"""TRAIN-bounded loader shared by the experiment scripts.

Rows at or after TRAIN_END never reach memory: files go through safe_load_parquet(max_end=TRAIN_END), and when the
tripwire refuses a file a pyarrow row filter ts < TRAIN_END is applied at read time and asserted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eqxs.data_zones import SealedWindowLeakError, safe_load_parquet  # noqa: E402

CACHE = ROOT / "data_cache"
RESULTS = ROOT / "results"
TRAIN_END_DEFAULT = pd.Timestamp("2024-10-01", tz="UTC")
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
    try:
        df = safe_load_parquet(path, max_end_date=TRAIN_END_DEFAULT, caller_intent=intent)
        via = "safe_load_parquet(max_end=TRAIN_END)"
    except SealedWindowLeakError:
        df = _io_filtered_train(path)
        via = "pyarrow row filter ts<TRAIN_END"
    col = "timestamp" if "timestamp" in df.columns else "ts"
    LOADER_LOG.append({"path": name, "via": via, "max_ts": str(df[col].max())})
    return df
