"""Pins the equities zone change of 2026-10-11 (author's decision): TRAIN 2004-01-01 -> 2023-01-01,
WFA 2023-01-01 -> 2024-07-01, SEALED 2024-07-01 -> 2026-06-01, and that the tripwire
with the equities zone file refuses rows dated 2024-07-01 or later."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import pandas as pd
import pytest

from eqxs.data_zones import (
    EQUITIES_ZONES_PATH,
    SealedWindowLeakError,
    load_zone_spec,
    reset_sealed_load_count,
    resolve_sealed_bound,
    safe_load_parquet,
    sealed_start,
)

NEW_SEALED = pd.Timestamp("2024-07-01", tz=UTC)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_sealed_load_count()
    yield
    reset_sealed_load_count()


def test_equities_zone_bounds_2026_10_11() -> None:
    spec = load_zone_spec(EQUITIES_ZONES_PATH)
    assert spec.train.start == pd.Timestamp("2004-01-01", tz=UTC)
    assert spec.train.end == spec.wfa_stage.start == pd.Timestamp("2023-01-01", tz=UTC)
    assert spec.wfa_stage.end == spec.sealed_test.start == NEW_SEALED
    assert spec.sealed_test.end == pd.Timestamp("2026-06-01", tz=UTC)
    assert sealed_start(EQUITIES_ZONES_PATH) == NEW_SEALED
    assert resolve_sealed_bound(zone_config=EQUITIES_ZONES_PATH) == NEW_SEALED


def _write(tmp_path: Path, dates: list[str], name: str) -> Path:
    p = tmp_path / name
    pd.DataFrame({"ts": pd.DatetimeIndex(dates).tz_localize("UTC"), "v": range(len(dates))}).to_parquet(p)
    return p


def test_safe_load_refuses_rows_on_or_after_2024_07_01(tmp_path: Path) -> None:
    p = _write(tmp_path, ["2023-06-01", "2024-06-28", "2024-07-01"], "edge.parquet")
    with pytest.raises(SealedWindowLeakError):
        safe_load_parquet(p, zone_config=EQUITIES_ZONES_PATH)
    p2 = _write(tmp_path, ["2023-06-01", "2025-03-03"], "later.parquet")
    with pytest.raises(SealedWindowLeakError):
        safe_load_parquet(p2, zone_config=EQUITIES_ZONES_PATH)


def test_safe_load_accepts_wfa_and_train_rows(tmp_path: Path) -> None:
    p = _write(tmp_path, ["2018-12-31", "2021-06-01", "2023-06-01", "2024-06-28"], "ok.parquet")
    df = safe_load_parquet(p, zone_config=EQUITIES_ZONES_PATH)
    assert len(df) == 4
    assert pd.Timestamp("2023-06-01", tz=UTC) in set(df["ts"])


def test_explicit_old_train_end_still_stricter(tmp_path: Path) -> None:
    """Studies 11-14 keep their own explicit end (2019-01-01): the zone file cannot widen it."""
    assert resolve_sealed_bound("2019-01-01", EQUITIES_ZONES_PATH) == pd.Timestamp("2019-01-01", tz=UTC)
    p = _write(tmp_path, ["2018-12-31", "2019-01-02"], "old.parquet")
    with pytest.raises(SealedWindowLeakError):
        safe_load_parquet(p, max_end_date="2019-01-01", zone_config=EQUITIES_ZONES_PATH)
