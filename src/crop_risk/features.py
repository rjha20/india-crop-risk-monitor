"""Point-in-time feature construction and conservative yield labels."""

from __future__ import annotations

import pandas as pd

from crop_risk.data_contracts import EXPOSURE_COLUMNS, IDENTITY_COLUMNS, InputBundle

RISK_THRESHOLD_PCT = -15.0
MIN_TREND_HISTORY = 3
FEATURE_COLUMNS = [
    "rainfall_mm",
    "rainfall_anomaly_pct",
    "dry_spell_days",
    "heat_days",
    "excess_rain_days",
    "ndvi_mean",
    "ndvi_trend",
    "ndvi_valid_obs",
    "coverage_score",
    "yield_lag_1",
    "yield_lag_2",
    "issue_day",
]


def build_feature_panel(bundle: InputBundle, risk_threshold_pct: float = RISK_THRESHOLD_PCT) -> pd.DataFrame:
    """Join outcomes to issue-time exposures and derive prior-only labels.

    Expected yield is a trailing median over up to five *prior* seasons, with
    three prior outcomes required. The current yield only creates the target;
    it is excluded from `FEATURE_COLUMNS` used by estimators.
    """

    panel = bundle.yields.merge(
        bundle.exposures[IDENTITY_COLUMNS + ["state", "district", *EXPOSURE_COLUMNS, "data_status"]],
        on=IDENTITY_COLUMNS,
        how="outer",
        validate="one_to_one",
        suffixes=("_yield", "_exposure"),
    )
    for label in ("state", "district"):
        left = panel[f"{label}_yield"]
        right = panel[f"{label}_exposure"]
        mismatch = left.notna() & right.notna() & left.astype("string").ne(right.astype("string"))
        if mismatch.any():
            raise ValueError(
                f"The source and exposure {label} labels differ for {int(mismatch.sum())} stable keys; "
                "review the district crosswalk before feature construction."
            )
        panel[label] = left.combine_first(right)
    panel = panel.sort_values(["district_id", "crop", "season", "season_year"]).reset_index(
        drop=True
    )
    group_keys = ["district_id", "crop", "season"]
    grouped = panel.groupby(group_keys, sort=False)["yield_t_ha"]
    panel["yield_lag_1"] = grouped.shift(1)
    panel["yield_lag_2"] = grouped.shift(2)
    panel["expected_yield_t_ha"] = grouped.transform(
        lambda values: values.shift(1).rolling(window=5, min_periods=MIN_TREND_HISTORY).median()
    )
    panel["yield_anomaly_pct"] = (
        100.0 * (panel["yield_t_ha"] - panel["expected_yield_t_ha"])
        / panel["expected_yield_t_ha"].where(panel["expected_yield_t_ha"] > 0)
    )
    panel["risk_label"] = (panel["yield_anomaly_pct"] <= risk_threshold_pct).astype("Int64")
    panel.loc[panel["yield_anomaly_pct"].isna(), "risk_label"] = pd.NA

    for column in FEATURE_COLUMNS:
        if column not in panel:
            panel[column] = pd.NA
    panel["issue_day"] = panel["issue_day"].fillna(90)

    exposure_status = panel.get("data_status_exposure")
    yield_status = panel.get("data_status_yield")
    if exposure_status is not None and yield_status is not None:
        statuses = pd.concat([yield_status, exposure_status], axis=1)

        def combine_status(row: pd.Series) -> str:
            values = {str(value).strip().lower() for value in row.dropna() if str(value).strip()}
            if not values:
                return "observed"
            if len(values) > 1:
                return "mixed_or_unknown"
            value = values.pop()
            return value

        panel["data_status"] = statuses.apply(combine_status, axis=1)
    elif exposure_status is not None:
        panel["data_status"] = exposure_status.fillna("observed")
    else:
        panel["data_status"] = panel.get("data_status", "observed")

    support_columns = [
        "rainfall_mm",
        "rainfall_anomaly_pct",
        "dry_spell_days",
        "heat_days",
        "excess_rain_days",
        "ndvi_mean",
        "ndvi_trend",
    ]
    panel["coverage_score"] = panel[support_columns].notna().mean(axis=1)
    panel["coverage_score"] = panel["coverage_score"].astype(float)
    return panel
