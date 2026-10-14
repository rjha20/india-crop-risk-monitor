"""Descriptive statistics for the district/crop/season feature panel.

These summaries describe the supplied panel; they are not causal estimates and
do not account for dependence between districts or years unless stated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _json_number(value: Any) -> float | None:
    """Return a finite Python float or None, suitable for JSON serialization."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _numeric_profile(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {
            "count": 0,
            "missing": int(len(series)),
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    quantiles = values.quantile([0.05, 0.50, 0.95])
    return {
        "count": int(values.size),
        "missing": int(len(series) - values.size),
        "mean": _json_number(values.mean()),
        "std": _json_number(values.std(ddof=1)) if values.size > 1 else None,
        "min": _json_number(values.min()),
        "p05": _json_number(quantiles.loc[0.05]),
        "median": _json_number(quantiles.loc[0.50]),
        "p95": _json_number(quantiles.loc[0.95]),
        "max": _json_number(values.max()),
    }


def _safe_value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.astype("string").fillna("<missing>").value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def summarize_statistics(features: pd.DataFrame | None) -> dict[str, Any]:
    """Return a JSON-safe profile of the crop panel with small-data guards.

    The summary intentionally reports sample counts and missingness alongside
    distribution statistics. The naive confidence interval for the overall
    mean is omitted because panel rows are not independent observations.
    """
    if features is None:
        frame = pd.DataFrame()
    else:
        frame = features.copy()

    key_columns = [column for column in ("district_id", "crop", "season", "season_year") if column in frame]
    duplicate_keys = int(frame.duplicated(key_columns, keep=False).sum()) if key_columns else None

    result: dict[str, Any] = {
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "unique_districts": int(frame["district_id"].nunique(dropna=True)) if "district_id" in frame else None,
        "unique_crops": int(frame["crop"].nunique(dropna=True)) if "crop" in frame else None,
        "unique_season_years": int(frame["season_year"].nunique(dropna=True)) if "season_year" in frame else None,
        "duplicate_panel_key_rows": duplicate_keys,
        "missing_fraction_by_column": {
            str(column): _json_number(frame[column].isna().mean()) for column in frame.columns
        },
        "numeric_profiles": {},
        "categorical_counts": {},
        "annual_yield_anomaly_pct": [],
        "pairwise_correlations": {},
        "interpretation_note": (
            "Descriptive panel summaries only; no causal interpretation. "
            "Rows within a district and across years are dependent."
        ),
    }

    if frame.empty:
        return result

    numeric_candidates = (
        "yield_t_ha",
        "expected_yield_t_ha",
        "yield_anomaly_pct",
        "area_ha",
        "rainfall_mm",
        "rainfall_anomaly_pct",
        "dry_spell_days",
        "heat_days",
        "excess_rain_days",
        "ndvi_mean",
        "ndvi_trend",
        "ndvi_valid_obs",
        "yield_lag_1",
        "yield_lag_2",
    )
    for column in numeric_candidates:
        if column in frame:
            result["numeric_profiles"][column] = _numeric_profile(frame[column])

    for column in ("state", "district", "crop", "risk_label"):
        if column in frame:
            result["categorical_counts"][column] = _safe_value_counts(frame[column])

    if "season_year" in frame and "yield_anomaly_pct" in frame:
        annual = frame[["season_year", "yield_anomaly_pct"]].copy()
        annual["season_year"] = pd.to_numeric(annual["season_year"], errors="coerce")
        annual["yield_anomaly_pct"] = pd.to_numeric(annual["yield_anomaly_pct"], errors="coerce")
        annual = annual.replace([np.inf, -np.inf], np.nan).dropna()
        if not annual.empty:
            grouped = annual.groupby("season_year", sort=True)["yield_anomaly_pct"]
            result["annual_yield_anomaly_pct"] = [
                {
                    "season_year": int(year),
                    "count": int(values.count()),
                    "mean": _json_number(values.mean()),
                    "median": _json_number(values.median()),
                    "std": _json_number(values.std(ddof=1)) if values.count() > 1 else None,
                }
                for year, values in grouped
            ]

    correlation_columns = [
        column
        for column in (
            "yield_anomaly_pct",
            "rainfall_anomaly_pct",
            "dry_spell_days",
            "heat_days",
            "excess_rain_days",
            "ndvi_mean",
            "ndvi_trend",
        )
        if column in frame
    ]
    if len(correlation_columns) > 1:
        numeric = frame[correlation_columns].apply(pd.to_numeric, errors="coerce")
        correlation = numeric.corr(method="spearman", min_periods=3)
        for left in correlation.index:
            for right in correlation.columns:
                if left < right:
                    result["pairwise_correlations"][f"{left}__{right}"] = _json_number(
                        correlation.loc[left, right]
                    )
        result["correlation_method"] = "Spearman rank correlation; pairwise complete cases, min n=3"

    return result
