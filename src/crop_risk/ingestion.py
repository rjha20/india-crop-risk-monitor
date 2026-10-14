"""Explicitly mapped import for official APY CSV exports.

This module avoids guessing source columns, unit scales, or district IDs. The
caller supplies a reviewed mapping and boundary crosswalk when needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from crop_risk.data_contracts import IDENTITY_COLUMNS, KEY_COLUMNS

AREA_FACTORS = {"ha": 1.0, "hectare": 1.0, "hectares": 1.0, "acre": 0.4046856422, "acres": 0.4046856422}
PRODUCTION_FACTORS = {
    "t": 1.0,
    "tonne": 1.0,
    "tonnes": 1.0,
    "metric_tonne": 1.0,
    "kg": 0.001,
    "thousand_tonnes": 1000.0,
    "lakh_tonnes": 100000.0,
}
MAPPING_FIELDS = ("state", "district", "crop", "season", "season_year", "area", "production")


class IngestionError(ValueError):
    """Raised when an APY source cannot be normalized without guesswork."""


def load_mapping(path: str | Path) -> dict[str, Any]:
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise IngestionError("Column mapping must be a JSON object")
    missing = [field for field in MAPPING_FIELDS if not mapping.get(field)]
    if missing:
        raise IngestionError(f"Mapping is missing fields: {', '.join(missing)}")
    return mapping


def normalize_apy_csv(
    input_path: str | Path,
    output_path: str | Path,
    mapping: dict[str, Any],
    *,
    area_unit: str,
    production_unit: str,
    boundary_crosswalk: str | Path | None = None,
    source_id: str = "DES_APY_LOCAL_SNAPSHOT",
    outcome_vintage: str | None = None,
) -> pd.DataFrame:
    """Normalize a mapped APY CSV to the project's canonical yields schema.

    Column names must come from the user's inspected source export. Area and
    production units are explicit arguments; duplicate keys fail closed.
    """

    area_unit_key = area_unit.strip().lower()
    production_unit_key = production_unit.strip().lower()
    if area_unit_key not in AREA_FACTORS:
        raise IngestionError(f"Unsupported area unit {area_unit!r}; supported units: {sorted(AREA_FACTORS)}")
    if production_unit_key not in PRODUCTION_FACTORS:
        raise IngestionError(
            f"Unsupported production unit {production_unit!r}; supported units: {sorted(PRODUCTION_FACTORS)}"
        )

    source = pd.read_csv(input_path)
    column_map = {field: mapping[field] for field in MAPPING_FIELDS}
    district_id_source = mapping.get("district_id")
    required_source_columns = list(column_map.values())
    if district_id_source:
        required_source_columns.append(district_id_source)
    missing_columns = [name for name in required_source_columns if name not in source.columns]
    if missing_columns:
        raise IngestionError("Source CSV is missing mapped columns: " + ", ".join(missing_columns))

    result = pd.DataFrame()
    for field in MAPPING_FIELDS:
        result[field] = source[column_map[field]]
    result = result.rename(
        columns={"area": "area_raw", "production": "production_raw"}
    )
    result["state"] = result["state"].astype("string").str.strip()
    result["district"] = result["district"].astype("string").str.strip()
    result["crop"] = result["crop"].astype("string").str.strip().str.lower()
    result["season"] = result["season"].astype("string").str.strip().str.lower()
    result["season_year"] = pd.to_numeric(result["season_year"], errors="coerce")
    result["area_ha"] = pd.to_numeric(result["area_raw"], errors="coerce") * AREA_FACTORS[area_unit_key]
    result["production_t"] = (
        pd.to_numeric(result["production_raw"], errors="coerce") * PRODUCTION_FACTORS[production_unit_key]
    )
    if district_id_source:
        result["district_id"] = source[district_id_source].astype("string").str.strip()
    elif boundary_crosswalk:
        crosswalk = pd.read_csv(boundary_crosswalk)
        needed = {"state", "district", "district_id"}
        if not needed.issubset(crosswalk.columns):
            raise IngestionError("Crosswalk must have state,district,district_id columns")
        crosswalk = crosswalk[list(needed)].copy()
        crosswalk["state"] = crosswalk["state"].astype("string").str.strip()
        crosswalk["district"] = crosswalk["district"].astype("string").str.strip()
        if crosswalk.duplicated(["state", "district"]).any():
            raise IngestionError("Crosswalk has duplicate state/district labels")
        result = result.merge(crosswalk, on=["state", "district"], how="left", validate="many_to_one")
    else:
        raise IngestionError(
            "A stable district_id column or reviewed state/district boundary crosswalk is required"
        )

    if result[KEY_COLUMNS].isna().any().any():
        raise IngestionError("Mapped rows have missing keys; inspect source codes and the district crosswalk")
    if result[["area_ha", "production_t"]].isna().any().any():
        raise IngestionError("Mapped rows have non-numeric or missing area/production values")
    if (result["area_ha"] <= 0).any() or (result["production_t"] < 0).any():
        raise IngestionError("Area must be positive and production cannot be negative")
    result["yield_t_ha"] = result["production_t"] / result["area_ha"]
    result["source_id"] = source_id
    result["official_vintage"] = outcome_vintage or "unknown—record the source release vintage"
    result["data_status"] = "observed"
    if result.duplicated(IDENTITY_COLUMNS).any():
        duplicate_rows = int(result.duplicated(IDENTITY_COLUMNS, keep=False).sum())
        raise IngestionError(
            f"Found {duplicate_rows} repeated district/crop/season/year source rows. "
            "Review source codes and aggregation rules; no automatic sum is applied."
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.drop(columns=["area_raw", "production_raw"]).to_csv(output, index=False)
    return result
