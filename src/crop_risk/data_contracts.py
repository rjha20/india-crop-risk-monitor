"""Input contracts and validation for district/crop/season observations.

Canonical inputs are CSV exports prepared from licensed source snapshots.
The package never downloads restricted material automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

KEY_COLUMNS = ["district_id", "state", "district", "crop", "season", "season_year"]
IDENTITY_COLUMNS = ["district_id", "crop", "season", "season_year"]
YIELD_REQUIRED = KEY_COLUMNS + ["area_ha", "yield_t_ha"]
EXPOSURE_COLUMNS = [
    "rainfall_mm",
    "rainfall_anomaly_pct",
    "dry_spell_days",
    "heat_days",
    "excess_rain_days",
    "ndvi_mean",
    "ndvi_trend",
    "ndvi_valid_obs",
    "coverage_score",
    "issue_day",
    "feature_end_day",
]
DEFAULT_DATA_STATUS = "observed"


@dataclass(frozen=True)
class InputBundle:
    yields: pd.DataFrame
    exposures: pd.DataFrame
    source_dir: Path


class DataContractError(ValueError):
    """Raised when an input cannot safely enter the forecasting workflow."""


def _normalize_keys(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise DataContractError(f"{table_name} is missing key columns: {', '.join(missing)}")
    result = frame.copy()
    for column in ("district_id", "state", "district", "crop", "season"):
        result[column] = result[column].astype("string").str.strip()
    result["crop"] = result["crop"].str.lower()
    result["season"] = result["season"].str.lower()
    result["season_year"] = pd.to_numeric(result["season_year"], errors="coerce")
    if result[KEY_COLUMNS].isna().any().any():
        bad = int(result[KEY_COLUMNS].isna().any(axis=1).sum())
        raise DataContractError(f"{table_name} contains {bad} rows with missing identity keys")
    if not result["season_year"].mod(1).eq(0).all():
        raise DataContractError(f"{table_name}.season_year must contain whole years")
    result["season_year"] = result["season_year"].astype("int64")
    duplicate_count = int(result.duplicated(IDENTITY_COLUMNS).sum())
    if duplicate_count:
        raise DataContractError(
            f"{table_name} has {duplicate_count} duplicate stable district/crop/season/year keys"
        )
    return result


def load_input_bundle(source_dir: str | Path) -> InputBundle:
    """Read `yields.csv` and `exposures.csv` and enforce their canonical units.

    `yield_t_ha` is tonnes per hectare. If it is absent, it is derived only
    when both `production_t` and positive `area_ha` are supplied.
    """

    source_path = Path(source_dir)
    yield_path = source_path / "yields.csv"
    exposure_path = source_path / "exposures.csv"
    if not yield_path.is_file() or not exposure_path.is_file():
        raise DataContractError(
            f"Expected yields.csv and exposures.csv in {source_path}. "
            "See data/raw/README.md for the canonical schema."
        )

    yields = pd.read_csv(yield_path)
    exposures = pd.read_csv(exposure_path)
    yields = _normalize_keys(yields, "yields.csv")
    exposures = _normalize_keys(exposures, "exposures.csv")

    if "area_ha" not in yields:
        raise DataContractError("yields.csv must contain area_ha")
    yields["area_ha"] = pd.to_numeric(yields["area_ha"], errors="coerce")
    if "yield_t_ha" not in yields:
        if "production_t" not in yields:
            raise DataContractError(
                "yields.csv must contain yield_t_ha or both production_t and area_ha"
            )
        yields["production_t"] = pd.to_numeric(yields["production_t"], errors="coerce")
        yields["yield_t_ha"] = yields["production_t"] / yields["area_ha"].where(
            yields["area_ha"] > 0
        )
    yields["yield_t_ha"] = pd.to_numeric(yields["yield_t_ha"], errors="coerce")
    if (yields["area_ha"].dropna() <= 0).any():
        raise DataContractError("area_ha must be positive when present")
    if (yields["yield_t_ha"].dropna() < 0).any():
        raise DataContractError("yield_t_ha cannot be negative")
    if yields["yield_t_ha"].notna().sum() == 0:
        raise DataContractError("yields.csv has no usable yield_t_ha values")

    if "data_status" not in yields:
        yields["data_status"] = DEFAULT_DATA_STATUS
    if "data_status" not in exposures:
        exposures["data_status"] = DEFAULT_DATA_STATUS

    for column in EXPOSURE_COLUMNS:
        if column not in exposures:
            if column in ("issue_day", "feature_end_day"):
                exposures[column] = 90
            else:
                exposures[column] = pd.NA
        exposures[column] = pd.to_numeric(exposures[column], errors="coerce")

    if exposures["issue_day"].isna().any():
        raise DataContractError("exposures.csv contains a row without issue_day")
    if (exposures["issue_day"] < 0).any() or (exposures["issue_day"] > 365).any():
        raise DataContractError("issue_day must be between 0 and 365")
    if (exposures["feature_end_day"] > exposures["issue_day"]).any():
        raise DataContractError(
            "Point-in-time check failed: a feature window ends after its issue_day"
        )

    # Exposure-only rows are allowed for an in-season issue year whose final
    # harvested yield is not available yet. They become forecast candidates.
    return InputBundle(yields=yields, exposures=exposures, source_dir=source_path)
