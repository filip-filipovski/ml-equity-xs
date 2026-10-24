"""Three-zone data discipline for strategy discovery, WFA, and sealed test.

Codifies the data split spec in ``config/data_zones.yaml`` so the rest
of the codebase has a single place to ask: *what window am I allowed to
look at right now?*

The three zones:

* :class:`DataZone.TRAIN` (2018-09-01 -> 2024-10-01, ~6.1y) — all
  strategy discovery, parameter tuning, panel composition, allocator
  hyperparameter choice happens HERE ONLY.
* :class:`DataZone.WFA_STAGE` (2024-10-01 -> 2025-10-01, 12m) — used
  as a SECOND-STAGE gate.  Configs that PASS train-only validation are
  tested on this window once.  Not for parameter selection.
* :class:`DataZone.SEALED_TEST` (2025-10-01 -> 2026-04-25, ~6m, latest
  data) — NEVER LOADED during discovery.  Opens only when a config is
  fully locked AND the WFA-stage PASS has been recorded.

The sealed zone is enforced by a tripwire: :func:`load_zoned_returns`
raises :class:`SealedZoneAccessError` when called with
``zone=DataZone.SEALED_TEST`` unless an explicit ``unlock_token``
argument matching ``LOCKED_CONFIG_<hash>`` is supplied.  Crossing zones
without explicit unlock is logged via :func:`validate_zone_access`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Project root resolves up from src/eqxs/data_zones.py.
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG: Final[Path] = _PROJECT_ROOT / "config" / "data_zones.yaml"
# Asset-class zone files.  Options-only work uses the options split; any
# cross-asset work (options joined with the equity parquet) must use the
# STRICTER equities split (sealed from 2022-01-01).
EQUITIES_ZONES_PATH: Final[Path] = _PROJECT_ROOT / "config" / "data_zones_equities.yaml"
OPTIONS_ZONES_PATH: Final[Path] = _PROJECT_ROOT / "config" / "data_zones_options.yaml"
# News-track split (GDELT / 8-K text / pageviews; sealed from 2024-07-01).  News
# loaders REQUIRE this file (the legacy default bound would let 2024-07..2025-10
# sealed news through).
NEWS_ZONES_PATH: Final[Path] = _PROJECT_ROOT / "config" / "data_zones_news.yaml"
_DEFAULT_CACHE_DIR: Final[Path] = _PROJECT_ROOT / "data_cache"

# Pattern for sealed-zone unlock tokens.  Format: LOCKED_CONFIG_<hex>.
_UNLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^LOCKED_CONFIG_[A-Fa-f0-9]{4,}$"
)


class DataZone(str, Enum):
    """The three discovery zones."""

    TRAIN = "train"
    WFA_STAGE = "wfa_stage"
    SEALED_TEST = "sealed_test"


class SealedZoneAccessError(RuntimeError):
    """Raised when the sealed test zone is accessed without an unlock token."""


@dataclass(frozen=True)
class ZoneWindow:
    """Inclusive-start, exclusive-end half-open window for a zone."""

    zone: DataZone
    start: pd.Timestamp
    end: pd.Timestamp

    def contains(self, ts: pd.Timestamp) -> bool:
        """Half-open membership check: ``start <= ts < end``."""
        return self.start <= ts < self.end


@dataclass(frozen=True)
class ZoneSpec:
    """Full three-zone specification loaded from YAML."""

    train: ZoneWindow
    wfa_stage: ZoneWindow
    sealed_test: ZoneWindow
    sealed_until: str
    wfa_until: str

    def window(self, zone: DataZone) -> ZoneWindow:
        if zone is DataZone.TRAIN:
            return self.train
        if zone is DataZone.WFA_STAGE:
            return self.wfa_stage
        if zone is DataZone.SEALED_TEST:
            return self.sealed_test
        raise ValueError(f"Unknown zone: {zone}")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def _parse_ts(s: str) -> pd.Timestamp:
    """Parse an ISO date as a UTC pandas timestamp."""
    return pd.Timestamp(s, tz=timezone.utc)


def load_zone_spec(config_path: Path | str | None = None) -> ZoneSpec:
    """Load the three-zone spec from YAML.

    Parameters
    ----------
    config_path
        Path to the ``data_zones.yaml`` file.  Defaults to
        ``<project_root>/config/data_zones.yaml``.
    """
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    zones = raw["zones"]
    spec = ZoneSpec(
        train=ZoneWindow(
            DataZone.TRAIN,
            _parse_ts(zones["train"]["start"]),
            _parse_ts(zones["train"]["end"]),
        ),
        wfa_stage=ZoneWindow(
            DataZone.WFA_STAGE,
            _parse_ts(zones["wfa_stage"]["start"]),
            _parse_ts(zones["wfa_stage"]["end"]),
        ),
        sealed_test=ZoneWindow(
            DataZone.SEALED_TEST,
            _parse_ts(zones["sealed_test"]["start"]),
            _parse_ts(zones["sealed_test"]["end"]),
        ),
        sealed_until=raw["discipline"]["sealed_until"],
        wfa_until=raw["discipline"]["wfa_until"],
    )

    # Sanity: zones must be contiguous and non-overlapping.
    if spec.train.end != spec.wfa_stage.start:
        raise ValueError(
            f"train.end ({spec.train.end}) != wfa_stage.start "
            f"({spec.wfa_stage.start})"
        )
    if spec.wfa_stage.end != spec.sealed_test.start:
        raise ValueError(
            f"wfa_stage.end ({spec.wfa_stage.end}) != sealed_test.start "
            f"({spec.sealed_test.start})"
        )
    return spec


# Module-level cache so callers don't re-read YAML on every call.
_SPEC_CACHE: ZoneSpec | None = None


def _spec() -> ZoneSpec:
    global _SPEC_CACHE
    if _SPEC_CACHE is None:
        _SPEC_CACHE = load_zone_spec()
    return _SPEC_CACHE


def reset_spec_cache() -> None:
    """Clear the cached spec.  Useful in tests that override ``config_path``."""
    global _SPEC_CACHE
    _SPEC_CACHE = None


# ---------------------------------------------------------------------------
# Tripwire / access validation
# ---------------------------------------------------------------------------
def _is_valid_unlock_token(token: str | None) -> bool:
    if token is None:
        return False
    return bool(_UNLOCK_PATTERN.match(token))


def validate_zone_access(
    zone: DataZone,
    caller_intent: str,
    *,
    unlock_token: str | None = None,
) -> None:
    """Log/raise on access patterns that violate the three-zone discipline.

    The contract:

    * Access to :attr:`DataZone.TRAIN` and :attr:`DataZone.WFA_STAGE` is
      always permitted; this function only logs the access for audit.
    * Access to :attr:`DataZone.SEALED_TEST` is **forbidden** unless an
      ``unlock_token`` matching ``LOCKED_CONFIG_<hex>`` is supplied AND
      ``caller_intent`` explicitly references a locked-config evaluation.

    Parameters
    ----------
    zone
        The zone being requested.
    caller_intent
        Free-text description of why the caller wants this data
        (e.g. ``"variant_b parameter sweep"``,
        ``"locked-config sealed-test eval"``).  Logged.
    unlock_token
        Mandatory iff ``zone == DataZone.SEALED_TEST``.

    Raises
    ------
    SealedZoneAccessError
        If the sealed zone is requested without a valid unlock token.
    """
    if zone is DataZone.SEALED_TEST:
        if not _is_valid_unlock_token(unlock_token):
            logger.error(
                "SEALED_TEST access DENIED (intent=%r, token=%r)",
                caller_intent,
                unlock_token,
            )
            raise SealedZoneAccessError(
                f"Sealed-test zone is locked.  Caller intent: {caller_intent!r}.  "
                f"Supply unlock_token='LOCKED_CONFIG_<hash>' to bypass."
            )
        logger.warning(
            "SEALED_TEST access GRANTED via unlock token=%s (intent=%r)",
            unlock_token,
            caller_intent,
        )
        return

    # TRAIN / WFA_STAGE — log for audit but always permit.
    logger.info("Zone %s access (intent=%r)", zone.value, caller_intent)


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------
def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a UTC DatetimeIndex.

    Accepts either:
    - already-indexed DataFrames with a DatetimeIndex (tz-aware or naive), OR
    - DataFrames with a ``ts`` column that can be promoted to the index.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        out = df
        if out.index.tz is None:
            out = out.tz_localize(timezone.utc)
        elif out.index.tz != timezone.utc:
            out = out.tz_convert(timezone.utc)
        return out

    if "ts" in df.columns:
        out = df.copy()
        out["ts"] = pd.to_datetime(out["ts"], utc=True)
        out = out.set_index("ts")
        return out

    raise ValueError(
        "DataFrame must have a DatetimeIndex or a 'ts' column"
    )


def as_of_split(df: pd.DataFrame, zone: DataZone) -> pd.DataFrame:
    """Filter ``df`` to the rows that fall inside ``zone``.

    Helper for the common case: caller already has a DataFrame and just
    wants the zone-restricted slice.  Does NOT validate access — pair
    with :func:`validate_zone_access` at the call site.
    """
    spec = _spec()
    window = spec.window(zone)
    indexed = _ensure_utc_index(df)
    mask = (indexed.index >= window.start) & (indexed.index < window.end)
    return indexed.loc[mask]


def load_zoned_returns(
    zone: DataZone,
    symbol: str,
    freq: str = "1d",
    *,
    unlock_token: str | None = None,
    caller_intent: str = "load_zoned_returns",
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Load a price/return parquet, restricted to ``zone``.

    Raises :class:`SealedZoneAccessError` if ``zone == DataZone.SEALED_TEST``
    and ``unlock_token`` is not a valid ``LOCKED_CONFIG_<hex>`` string.

    Parameters
    ----------
    zone
        Which zone to slice to.
    symbol
        Binance-format symbol, e.g. ``"BTCUSDT"``.
    freq
        Candle frequency suffix, e.g. ``"1d"``, ``"1h"``, ``"4h"``.
    unlock_token
        Required iff ``zone == DataZone.SEALED_TEST``.
    caller_intent
        Free-text description of the call site for the audit log.
    cache_dir
        Override the default ``<project_root>/data_cache`` location.
    """
    validate_zone_access(zone, caller_intent, unlock_token=unlock_token)

    root = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
    path = root / f"binance_{symbol}_{freq}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Parquet cache missing for {symbol}@{freq}: {path}"
        )

    df = pd.read_parquet(path)
    return as_of_split(df, zone)


# ---------------------------------------------------------------------------
# Hardening — process-level guard against accidental sealed-window reads
# from direct ``pd.read_parquet`` callers that bypass ``load_zoned_returns``.
# ---------------------------------------------------------------------------
#
# Background: commit 1ed3b7d documented a framework violation in
# ``scripts/test_hurst_vr_eval.py`` where the realized-PnL panel was read
# via ``pd.read_parquet`` directly and then sliced to the sealed window
# without ever passing through :func:`load_zoned_returns`.  The original
# tripwire only fires at the named entry point, so a direct parquet read
# is invisible to the audit log.
#
# :func:`safe_load_parquet` is the hardened replacement.  It (a) reads the
# parquet, (b) records how many rows fall in the sealed window, (c)
# unconditionally drops those rows unless an explicit unlock token is
# supplied, and (d) bumps a process-level counter so scripts can call
# :func:`assert_sealed_clean` at the end of a run to assert the discipline
# was held.

# Default upper bound for safe loads: end of the WFA stage (``2025-10-01``).
# A script that wants to read into the WFA stage should pass an explicit
# ``max_end_date`` instead of relying on the default.
WFA_END_DEFAULT: Final[pd.Timestamp] = pd.Timestamp("2025-10-01", tz=timezone.utc)
TRAIN_END_DEFAULT: Final[pd.Timestamp] = pd.Timestamp("2024-10-01", tz=timezone.utc)
SEALED_END_DEFAULT: Final[pd.Timestamp] = pd.Timestamp("2026-04-25", tz=timezone.utc)

# Process-level audit counters.  Each successful sealed-window peek
# (whether granted by ``allow_sealed=True`` or denied via raise) is
# recorded.  ``assert_sealed_clean`` raises if any peeks were granted
# during the process.
_SEALED_LOAD_COUNT: int = 0
_SEALED_LOAD_LOG: list[dict[str, object]] = []


class SealedWindowLeakError(RuntimeError):
    """Raised when ``safe_load_parquet`` sees rows past ``max_end_date``
    and the caller did not opt in via ``allow_sealed=True`` + a valid
    unlock token."""


# Date-column names we recognise, in priority order.  A parquet may key its
# timestamps on a ``DatetimeIndex`` OR on one of these columns.  Funding
# parquets use ``timestamp``; perp/spot OHLCV parquets use ``ts``; option-chain
# parquets (the options store) key every contract row on
# ``quote_date`` (date / datetime / ISO string, or a categorical when read
# back from a hive-partitioned dataset).
_DATE_COLUMN_CANDIDATES: Final[tuple[str, ...]] = ("ts", "timestamp", "quote_date")

# Heuristic threshold for distinguishing epoch-MILLISECOND integers from
# epoch-second / epoch-nanosecond integers.  2018-09-01 in ms is ~1.535e12
# and the sealed boundary (2025-10-01) is ~1.759e12, so any plausible ms
# timestamp for this project sits well above 1e12.  Epoch SECONDS for the
# same dates are ~1.5e9 (below 1e12); epoch NANOSECONDS are ~1.5e18 (above).
# We only need to disambiguate ms from the default ns interpretation that
# ``pd.to_datetime`` applies to bare integers, which silently maps ms values
# into 1970 — the exact blind spot that let perp parquets leak.
_EPOCH_MS_LOWER: Final[float] = 1e12
_EPOCH_MS_UPPER: Final[float] = 1e16


def _to_utc_datetime_for_count(values: pd.Series | pd.Index) -> pd.Series | pd.Index:
    """Coerce a date-like Series/Index to a UTC-aware datetime for comparison.

    Handles three encodings seen in the cache:

    * already-datetime (tz-aware or naive) — localised/converted to UTC;
    * integer **epoch-millisecond** columns (``int64`` ``ts`` in perp/spot
      parquets) — detected when the integer magnitudes sit in the ms band
      (``> 1e12``) and converted with ``unit="ms"``.  Without this branch
      ``pd.to_datetime`` treats the integers as nanoseconds and maps every
      value into 1970, so a ``>= 2025-10-01`` comparison is uniformly
      ``False`` and sealed rows leak;
    * anything else (e.g. ISO strings) — parsed via ``pd.to_datetime`` with
      ``utc=True``.

    Counts/comparisons only — never used to compute a statistic on values.
    """
    # Categorical (e.g. a hive partition column ``quote_date=2024-01-02`` read
    # back by pyarrow) — compare on the underlying category values.
    if isinstance(values.dtype, pd.CategoricalDtype):
        if isinstance(values, pd.Index):
            values = pd.Index(np.asarray(values, dtype=object))
        else:
            values = pd.Series(np.asarray(values, dtype=object), index=values.index)

    # Integer epoch handling (covers int64-ms ``ts``/``timestamp`` columns).
    if pd.api.types.is_integer_dtype(values):
        # Use the max magnitude to classify the encoding.  Empty -> ns is fine.
        try:
            mx = float(abs(values).max()) if len(values) else 0.0
        except (TypeError, ValueError):
            mx = 0.0
        if _EPOCH_MS_LOWER < mx < _EPOCH_MS_UPPER:
            return pd.to_datetime(values, unit="ms", utc=True)
        # Below the ms band: most likely epoch seconds.
        if 0 < mx <= _EPOCH_MS_LOWER:
            return pd.to_datetime(values, unit="s", utc=True)
        # Fall through (e.g. all-zero or implausibly large) — default parse.
        return pd.to_datetime(values, utc=True)

    # Already a datetime Index.
    if isinstance(values, pd.DatetimeIndex):
        idx = values
        if idx.tz is None:
            return idx.tz_localize(timezone.utc)
        if idx.tz != timezone.utc:
            return idx.tz_convert(timezone.utc)
        return idx

    # Datetime Series (tz-aware or naive) or string-like — let pandas parse.
    return pd.to_datetime(values, utc=True)


def _date_column(df: pd.DataFrame) -> str | None:
    """Return the name of the first recognised date column, or ``None``.

    The DataFrame index is checked by the caller; this only inspects columns.
    """
    for name in _DATE_COLUMN_CANDIDATES:
        if name in df.columns:
            return name
    return None


def _sealed_mask(df: pd.DataFrame, max_end_date: pd.Timestamp) -> pd.Series | None:
    """Return a boolean mask ``ts >= max_end_date`` aligned to ``df``.

    Recognises a UTC ``DatetimeIndex`` first, then a recognised date column
    (``ts`` or ``timestamp``), correctly handling int64 epoch-millisecond
    encodings.  Returns ``None`` when no date key is visible (nothing to
    leak from).
    """
    if isinstance(df.index, pd.DatetimeIndex):
        idx = _to_utc_datetime_for_count(df.index)
        return pd.Series(idx >= max_end_date, index=df.index)
    col = _date_column(df)
    if col is not None:
        ts = _to_utc_datetime_for_count(df[col])
        return pd.Series(pd.Index(ts) >= max_end_date, index=df.index)
    return None


def _count_sealed_rows(df: pd.DataFrame, max_end_date: pd.Timestamp) -> int:
    """Return how many rows in ``df`` have a timestamp >= ``max_end_date``.

    Counts only — does NOT compute any statistic on the values.  Used to
    report leak size in the error message and the audit log.

    Recognises a UTC ``DatetimeIndex``, a datetime/int-epoch-ms ``ts``
    column, AND a datetime/int-epoch-ms ``timestamp`` column.  The
    ``timestamp``-column and int64-ms cases were previously invisible,
    which let funding (``timestamp`` col) and perp (``int64`` ms ``ts``
    col) parquets silently return sealed rows.
    """
    mask = _sealed_mask(df, max_end_date)
    if mask is None:
        # No date column visible — treat as zero (nothing to leak from).
        return 0
    return int(mask.to_numpy().sum())


def sealed_start(zone_config: Path | str) -> pd.Timestamp:
    """Return the sealed-test start (UTC) declared by a zone YAML file."""
    return load_zone_spec(zone_config).sealed_test.start


def resolve_sealed_bound(
    max_end_date: pd.Timestamp | str | None = None,
    zone_config: Path | str | None = None,
) -> pd.Timestamp:
    """Resolve the tripwire bound for :func:`safe_load_parquet`.

    * neither given -> :data:`WFA_END_DEFAULT` (unchanged legacy default);
    * only ``max_end_date`` -> that date;
    * only ``zone_config`` -> the sealed-test start of that YAML;
    * both -> the EARLIER (stricter) of the two.  A zone file can never be
      used to widen an explicit bound.
    """
    bounds: list[pd.Timestamp] = []
    if max_end_date is not None:
        if isinstance(max_end_date, str):
            b = pd.Timestamp(max_end_date, tz=timezone.utc)
        else:
            b = pd.Timestamp(max_end_date)
            if b.tzinfo is None:
                b = b.tz_localize(timezone.utc)
            elif b.tz != timezone.utc:
                b = b.tz_convert(timezone.utc)
        bounds.append(b)
    if zone_config is not None:
        bounds.append(sealed_start(zone_config))
    if not bounds:
        return WFA_END_DEFAULT
    return min(bounds)


def safe_load_parquet(
    path: Path | str,
    *,
    max_end_date: pd.Timestamp | str | None = None,
    allow_sealed: bool = False,
    unlock_token: str | None = None,
    caller_intent: str = "safe_load_parquet",
    zone_config: Path | str | None = None,
) -> pd.DataFrame:
    """Read a parquet and refuse to return sealed-window rows by default.

    The default ``max_end_date`` is :data:`WFA_END_DEFAULT` (i.e. the start
    of the sealed test zone).  Rows with timestamp >= ``max_end_date`` are:

    * **dropped** when ``allow_sealed`` is ``False`` (the default) AND
      :class:`SealedWindowLeakError` is raised if any such rows existed,
      so the caller can never "accidentally" read them; OR
    * **retained** when ``allow_sealed`` is ``True`` AND ``unlock_token``
      is a valid ``LOCKED_CONFIG_<hex>`` string.  In that path the
      process-level :data:`_SEALED_LOAD_COUNT` is incremented and a log
      entry is appended.

    This is the hardened replacement for ad-hoc
    ``df = pd.read_parquet(p); df = df[df.index < WFA_END]`` patterns
    sprinkled through ``scripts/``.  The pattern is correct only when the
    truncation is actually applied — commit 1ed3b7d caught the case where
    it was not.

    Parameters
    ----------
    path
        Parquet file path.
    max_end_date
        Upper bound for permissible timestamps; rows with ``ts >= bound``
        are sealed-window for the purposes of this function.  Defaults to
        :data:`WFA_END_DEFAULT`.  Pass :data:`TRAIN_END_DEFAULT` to confine
        to the train zone.
    allow_sealed
        If ``True`` AND ``unlock_token`` is a valid
        ``LOCKED_CONFIG_<hex>`` string, sealed-window rows are returned
        in the DataFrame.  Otherwise sealed-window rows are dropped and
        :class:`SealedWindowLeakError` is raised if any existed in the
        source parquet.
    unlock_token
        Required iff ``allow_sealed`` is ``True``.  Must match the
        ``LOCKED_CONFIG_<hex>`` pattern.
    caller_intent
        Free-text audit-log description of why the caller wants this
        parquet.  Logged at INFO normally, WARNING when sealed access is
        granted, ERROR when a leak is refused.
    zone_config
        Optional asset-class zone YAML (e.g. :data:`OPTIONS_ZONES_PATH`,
        :data:`EQUITIES_ZONES_PATH`).  Its sealed-test start becomes the
        bound; combined with ``max_end_date`` the stricter (earlier) bound
        wins — see :func:`resolve_sealed_bound`.
    """
    global _SEALED_LOAD_COUNT, _SEALED_LOAD_LOG

    bound = resolve_sealed_bound(max_end_date, zone_config)

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet not found: {p}")

    df = pd.read_parquet(p)
    leak_count = _count_sealed_rows(df, bound)

    if leak_count == 0:
        # Nothing past the bound — return as-is.
        logger.info(
            "safe_load_parquet(%s) intent=%r: %d rows, no sealed rows",
            p.name,
            caller_intent,
            len(df),
        )
        return df

    if allow_sealed:
        if not _is_valid_unlock_token(unlock_token):
            logger.error(
                "safe_load_parquet SEALED ACCESS DENIED (path=%s, "
                "intent=%r, token=%r, leak_count=%d)",
                p,
                caller_intent,
                unlock_token,
                leak_count,
            )
            raise SealedZoneAccessError(
                f"safe_load_parquet refused sealed access to {p.name}: "
                f"allow_sealed=True requires unlock_token='LOCKED_CONFIG_<hex>' "
                f"(got {unlock_token!r}, {leak_count} rows past {bound.date()})"
            )
        _SEALED_LOAD_COUNT += 1
        _SEALED_LOAD_LOG.append(
            {
                "path": str(p),
                "bound": bound.isoformat(),
                "leak_count": leak_count,
                "unlock_token": unlock_token,
                "caller_intent": caller_intent,
            }
        )
        logger.warning(
            "safe_load_parquet SEALED ACCESS GRANTED via token=%s "
            "(path=%s, intent=%r, leak_count=%d)",
            unlock_token,
            p,
            caller_intent,
            leak_count,
        )
        return df

    # Default discovery path: drop sealed rows AND raise so the caller
    # knows the parquet contained sealed material.  The error carries the
    # truncated frame so the caller can recover gracefully if they meant
    # to discard sealed rows all along.  Uses the same date-key detection
    # as ``_count_sealed_rows`` so ``timestamp`` columns and int64-ms
    # encodings are truncated, not silently passed through.
    mask = _sealed_mask(df, bound)
    if mask is not None:
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = _to_utc_datetime_for_count(df.index)
        clean = df.loc[~mask.to_numpy()].copy()
    else:
        clean = df

    logger.error(
        "safe_load_parquet SEALED LEAK REFUSED (path=%s, intent=%r, "
        "leak_count=%d, bound=%s).  Pass allow_sealed=True + "
        "unlock_token to opt in.",
        p,
        caller_intent,
        leak_count,
        bound.isoformat(),
    )
    raise SealedWindowLeakError(
        f"Parquet {p.name} contained {leak_count} rows >= {bound.date()} "
        f"(sealed window).  Discovery code must NOT read these.  "
        f"If this is a locked-config eval, call with "
        f"allow_sealed=True and unlock_token='LOCKED_CONFIG_<hex>'."
    )


def assert_sealed_clean() -> None:
    """Raise if any sealed-window read was granted in this process.

    Scripts that claim "SEALED UNTOUCHED" should call this near the end
    of ``main()`` as a structural check.  Pairs with
    :func:`safe_load_parquet`: every ``allow_sealed=True`` call bumps
    :data:`_SEALED_LOAD_COUNT`, so a process that swears it didn't peek
    can prove the claim by asserting the counter is zero.
    """
    if _SEALED_LOAD_COUNT != 0:
        raise SealedZoneAccessError(
            f"assert_sealed_clean: {_SEALED_LOAD_COUNT} sealed-window "
            f"read(s) were granted in this process.  "
            f"Log: {_SEALED_LOAD_LOG!r}"
        )


def sealed_load_count() -> int:
    """Return the process-level count of sealed reads granted so far."""
    return _SEALED_LOAD_COUNT


def reset_sealed_load_count() -> None:
    """Test-only: reset the process-level sealed-read counter."""
    global _SEALED_LOAD_COUNT, _SEALED_LOAD_LOG
    _SEALED_LOAD_COUNT = 0
    _SEALED_LOAD_LOG = []


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------
def zone_window(zone: DataZone) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return ``(start, end)`` UTC timestamps for the requested zone."""
    w = _spec().window(zone)
    return w.start, w.end


def zones_overview() -> dict[str, dict[str, str]]:
    """Diagnostic dict of all three windows.  Safe to log."""
    spec = _spec()
    out: dict[str, dict[str, str]] = {}
    for z in DataZone:
        w = spec.window(z)
        out[z.value] = {"start": w.start.isoformat(), "end": w.end.isoformat()}
    return out


def parse_iso_timestamp(s: str) -> datetime:
    """Public helper to parse an ISO date as UTC, mirrored from `_parse_ts`."""
    return _parse_ts(s).to_pydatetime()
