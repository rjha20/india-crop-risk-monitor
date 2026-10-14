"""Historical yield-risk features and strictly chronological model evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


GROUP_COLUMNS = ["state_key", "district_key", "crop_key", "season"]
KEY_COLUMNS = ["year", *GROUP_COLUMNS]
RISK_THRESHOLD_PCT = -15.0
MODEL_FEATURES = [
    "yield_lag_1",
    "yield_lag_2",
    "yield_lag_3",
    "prior_median",
    "prior_std",
    "prior_trend_pct",
    "area_lag_1",
]


def build_historical_panel(source: pd.DataFrame) -> pd.DataFrame:
    """Create prior-only yield anomalies; no current-year target enters features."""
    required = {*KEY_COLUMNS, "year", "season_year", "state_name", "district_name", "crop_name", "area", "yield"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError("Crop source is missing analysis columns: " + ", ".join(missing))

    frame = source.copy()
    for column in ("season_year", "area", "yield"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["season_year", *GROUP_COLUMNS, "yield"]).copy()
    frame["season_year"] = frame["season_year"].astype(int)
    if frame.duplicated(KEY_COLUMNS, keep=False).any():
        duplicate_count = int(frame.duplicated(KEY_COLUMNS, keep=False).sum())
        raise ValueError(
            f"The source contains {duplicate_count:,} duplicate year/district/crop/season rows. "
            "Resolve the source keys before analysis; rows were not aggregated."
        )
    frame = frame.sort_values([*GROUP_COLUMNS, "season_year"], kind="stable").reset_index(drop=True)
    grouped_yield = frame.groupby(GROUP_COLUMNS, sort=False, dropna=False)["yield"]
    grouped_area = frame.groupby(GROUP_COLUMNS, sort=False, dropna=False)["area"]
    grouped_year = frame.groupby(GROUP_COLUMNS, sort=False, dropna=False)["season_year"]

    frame["yield_lag_1"] = grouped_yield.shift(1).where(grouped_year.shift(1).eq(frame["season_year"] - 1))
    frame["yield_lag_2"] = grouped_yield.shift(2).where(grouped_year.shift(2).eq(frame["season_year"] - 2))
    frame["yield_lag_3"] = grouped_yield.shift(3).where(grouped_year.shift(3).eq(frame["season_year"] - 3))
    frame["area_lag_1"] = grouped_area.shift(1).where(grouped_year.shift(1).eq(frame["season_year"] - 1))

    # Rolling by row would silently stretch a five-season window across gaps.
    # Use only observations whose source year is within the previous five years.
    prior_median = np.full(len(frame), np.nan, dtype=float)
    prior_std = np.full(len(frame), np.nan, dtype=float)
    years_all = frame["season_year"].to_numpy(dtype=int)
    yields_all = frame["yield"].to_numpy(dtype=float)
    for positions in frame.groupby(GROUP_COLUMNS, sort=False, dropna=False).indices.values():
        years = years_all[positions]
        values = yields_all[positions]
        for local_index, row_position in enumerate(positions):
            current_year = years[local_index]
            in_window = (years >= current_year - 5) & (years < current_year)
            prior_values = values[in_window]
            prior_values = prior_values[np.isfinite(prior_values)]
            if prior_values.size >= 3:
                prior_median[row_position] = float(np.median(prior_values))
                prior_std[row_position] = float(np.std(prior_values, ddof=1))
    frame["prior_median"] = prior_median
    frame["prior_std"] = prior_std
    frame["prior_trend_pct"] = (
        100.0 * (frame["yield_lag_1"] - frame["yield_lag_3"])
        / frame["yield_lag_3"].where(frame["yield_lag_3"] > 0)
    )
    frame["yield_anomaly_pct"] = (
        100.0 * (frame["yield"] - frame["prior_median"])
        / frame["prior_median"].where(frame["prior_median"] > 0)
    )
    observed_low_yield = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    eligible = frame["yield_anomaly_pct"].notna()
    observed_low_yield.loc[eligible] = frame.loc[eligible, "yield_anomaly_pct"].le(RISK_THRESHOLD_PCT).astype(int)
    frame["observed_low_yield"] = observed_low_yield
    return frame


def summarize_source(frame: pd.DataFrame) -> dict[str, Any]:
    """Return compact, non-inferential source quality and coverage statistics."""
    numeric_yield = pd.to_numeric(frame["yield"], errors="coerce")
    duplicate_keys = int(frame.duplicated(KEY_COLUMNS, keep=False).sum())
    return {
        "rows": int(len(frame)),
        "missing_yield_rows": int(numeric_yield.isna().sum()),
        "negative_yield_rows": int(numeric_yield.lt(0).sum()),
        "missing_district_code_rows": int(frame["district_code"].isna().sum()),
        "missing_crop_code_rows": int(frame["crop_code"].isna().sum()),
        "duplicate_key_rows": duplicate_keys,
        "states": int(frame["state_name"].nunique(dropna=True)),
        "districts": int(frame["district_key"].nunique(dropna=True)),
        "crops": int(frame["crop_key"].nunique(dropna=True)),
        "seasons": int(frame["season"].nunique(dropna=True)),
        "year_start": int(frame["season_year"].min()),
        "year_end": int(frame["season_year"].max()),
        "risk_threshold_pct": RISK_THRESHOLD_PCT,
    }


def walk_forward_evaluation(panel: pd.DataFrame, *, holdout_years: int = 3) -> dict[str, Any]:
    """Evaluate a history-only random forest on later year blocks.

    Each fold trains exclusively on years before its holdout year. Features
    use lagged yield/area values and prior-only summaries. Scores are uncalibrated.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    usable = panel.dropna(subset=[*MODEL_FEATURES, "observed_low_yield"]).copy()
    if usable.empty:
        return {
            "status": "insufficient_history",
            "message": "No rows have enough prior-year history for a walk-forward model.",
            "folds": [],
            "overall": None,
        }
    usable["observed_low_yield"] = usable["observed_low_yield"].astype(int)
    years = sorted(int(value) for value in usable["season_year"].unique())
    validation_years = years[-holdout_years:]
    predictions: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    folds: list[dict[str, Any]] = []

    for year in validation_years:
        train = usable[usable["season_year"] < year]
        valid = usable[usable["season_year"] == year]
        if len(train) < 500 or train["observed_low_yield"].nunique() < 2 or valid.empty:
            continue
        model = RandomForestClassifier(
            n_estimators=80,
            max_depth=8,
            min_samples_leaf=30,
            max_features=0.9,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(train[MODEL_FEATURES], train["observed_low_yield"])
        positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
        probability = model.predict_proba(valid[MODEL_FEATURES])[:, positive_index]
        actual = valid["observed_low_yield"].to_numpy(dtype=int)
        predictions.append(probability)
        actuals.append(actual)
        folds.append(
            {
                "holdout_year": year,
                "train_years": int(train["season_year"].nunique()),
                "train_rows": int(len(train)),
                "holdout_rows": int(len(valid)),
                "observed_low_yield_rate": float(actual.mean()),
                "average_precision": (
                    float(average_precision_score(actual, probability))
                    if np.unique(actual).size == 2
                    else None
                ),
                "roc_auc": (
                    float(roc_auc_score(actual, probability)) if np.unique(actual).size == 2 else None
                ),
                "brier_score": float(brier_score_loss(actual, probability)),
            }
        )

    if not folds:
        return {
            "status": "insufficient_history",
            "message": "No holdout year had enough prior training rows and both target classes.",
            "folds": [],
            "overall": None,
        }

    actual = np.concatenate(actuals)
    probability = np.concatenate(predictions)
    overall = {
        "holdout_rows": int(len(actual)),
        "holdout_years": [fold["holdout_year"] for fold in folds],
        "observed_low_yield_rate": float(actual.mean()),
        "baseline_brier_score": float(actual.mean() * (1.0 - actual.mean())),
        "average_precision": (
            float(average_precision_score(actual, probability))
            if np.unique(actual).size == 2
            else None
        ),
        "roc_auc": float(roc_auc_score(actual, probability)) if np.unique(actual).size == 2 else None,
        "brier_score": float(brier_score_loss(actual, probability)),
        "baseline_average_precision": float(actual.mean()),
        "model": "Random forest on prior yield/area history",
        "target": f"Observed yield at least {abs(RISK_THRESHOLD_PCT):.0f}% below its prior 5-year median",
        "probability_status": "uncalibrated historical score",
    }
    return {"status": "complete", "folds": folds, "overall": overall}
