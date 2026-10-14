"""Leakage-aware yield anomaly forecasting for a district/crop panel.

This is a screening prototype. Historical forecasts are evaluated with year
blocks: every prediction for year *t* is trained only on labeled years < *t*.
The classifier's probabilities are deliberately marked uncalibrated.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .statistics import summarize_statistics


FORECAST_COLUMNS = [
    "district_id",
    "state",
    "district",
    "crop",
    "season",
    "season_year",
    "expected_yield_t_ha",
    "predicted_yield_t_ha",
    "lower_90_t_ha",
    "upper_90_t_ha",
    "risk_probability",
    "risk_label",
    "probability_status",
    "issue_day",
    "coverage_score",
    "ood_flag",
    "data_status",
    "forecast_type",
]

NUMERIC_FEATURES = [
    "season_year",
    "expected_yield_t_ha",
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
]
CATEGORICAL_FEATURES = ["state", "district", "district_id", "crop"]
COVERAGE_FEATURES = [
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
]

MIN_ML_ROWS = 20
MIN_ML_YEARS = 3
MIN_RISK_ROWS = 12
MIN_INTERVAL_RESIDUALS = 5
RANDOM_STATE = 42

_POSITIVE_RISK_LABELS = {
    "1",
    "true",
    "yes",
    "y",
    "high",
    "high risk",
    "high-risk",
    "high_risk",
    "very high",
    "very high risk",
    "severe",
    "critical",
    "elevated",
    "elevated risk",
    "at risk",
    "at-risk",
    "risk",
    "risk flagged",
    "alert",
}
_NEGATIVE_RISK_LABELS = {
    "0",
    "false",
    "no",
    "n",
    "low",
    "low risk",
    "low-risk",
    "low_risk",
    "normal",
    "safe",
    "not at risk",
    "not_at_risk",
    "not elevated",
    "not_elevated",
    "no risk",
    "none",
}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _json_clean(value: Any) -> Any:
    """Convert nested pandas/numpy values to strict JSON-safe values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): _json_clean(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(child) for child in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        try:
            return pd.Timestamp(value).isoformat()
        except (TypeError, ValueError):
            return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value if isinstance(value, str) else str(value)


def _data_status(frame: pd.DataFrame, original: pd.DataFrame) -> str:
    """Preserve explicit provenance; callers should tag synthetic fixtures."""
    for column in ("data_status", "source_status", "dataset_status"):
        if column in frame:
            values = frame[column].dropna().astype(str).str.strip().str.lower()
            if not values.empty:
                unique = set(values)
                if len(unique) > 1:
                    return "mixed_or_unknown"
                source = next(iter(unique))
                normalized = re.sub(r"[_\-]+", " ", source)
                if any(token in normalized for token in ("synthetic", "simulated", "demo fixture")):
                    return "synthetic_demo"
                return source
    source: Any = original.attrs.get("data_status")
    if source is None:
        # A caller that passes an untagged real-input panel gets the documented
        # real-input default. Demo generation should set the column or DataFrame
        # attribute to "synthetic_demo" rather than relying on inference.
        return "observed"
    normalized = re.sub(r"[_\-]+", " ", str(source).strip().lower())
    if any(token in normalized for token in ("synthetic", "simulated", "demo fixture")):
        return "synthetic_demo"
    return "observed"


def _normalized_risk_label(value: Any) -> float:
    """Map supported high-risk labels to 1, known negatives to 0, else NaN."""
    if value is None or value is pd.NA:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    numeric = _finite(value)
    if numeric is not None:
        if numeric in (0.0, 1.0):
            return numeric
        return float(numeric > 0)
    normalized = re.sub(r"[_\-]+", " ", str(value).strip().lower())
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in _POSITIVE_RISK_LABELS or any(
        phrase in normalized
        for phrase in ("high risk", "elevated risk", "at risk", "severe risk", "critical risk")
    ):
        return 1.0
    if normalized in _NEGATIVE_RISK_LABELS or any(
        phrase in normalized for phrase in ("not at risk", "no risk", "low risk", "not elevated")
    ):
        return 0.0
    return np.nan


def _model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select only point-in-time covariates; never include observed outcomes."""
    columns = [column for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES if column in frame]
    output = frame[columns].copy()
    for column in NUMERIC_FEATURES:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")
            output[column] = output[column].replace([np.inf, -np.inf], np.nan)
    for column in CATEGORICAL_FEATURES:
        if column in output:
            output[column] = output[column].astype("string").fillna("<missing>").astype(str)
    return output


def _make_pipeline(columns: list[str], classifier: bool = False):
    """Create a fold-local imputation/encoding and random-forest pipeline."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric = [column for column in columns if column in NUMERIC_FEATURES]
    categorical = [column for column in columns if column in CATEGORICAL_FEATURES]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric))
    if categorical:
        categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))
    if not transformers:
        raise ValueError("No usable model features are present.")
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    if classifier:
        estimator = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    else:
        estimator = RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def _trailing_median(history: pd.DataFrame, row: pd.Series) -> float:
    """Use only earlier-year yield anomalies, with broader fallbacks."""
    if history.empty or "yield_anomaly_pct" not in history:
        return 0.0
    hist = history.copy()
    hist["yield_anomaly_pct"] = pd.to_numeric(hist["yield_anomaly_pct"], errors="coerce")
    hist = hist.replace([np.inf, -np.inf], np.nan).dropna(subset=["yield_anomaly_pct"])
    if hist.empty:
        return 0.0

    def matching(columns: list[str]) -> pd.DataFrame:
        selected = hist
        for column in columns:
            value = row.get(column)
            if column not in hist or value is None or pd.isna(value):
                return hist.iloc[0:0]
            selected = selected[selected[column].astype(str) == str(value)]
        return selected

    for keys in (["district_id", "crop"], ["state", "crop"], ["crop"]):
        selected = matching(list(keys))
        if not selected.empty:
            if "season_year" in selected:
                selected = selected.sort_values("season_year", kind="stable")
            values = selected["yield_anomaly_pct"].tail(5)
            if not values.empty:
                return float(values.median())
    return float(hist["yield_anomaly_pct"].tail(5).median())


def _yield_from_anomaly(expected: Any, anomaly: Any) -> float | None:
    expected_value = _finite(expected)
    anomaly_value = _finite(anomaly)
    if expected_value is None or anomaly_value is None or expected_value < 0:
        return None
    # Negative production is physically invalid; the anomaly itself remains
    # unchanged for evaluation, while yield-unit outputs are bounded at zero.
    return max(0.0, expected_value * (1.0 + anomaly_value / 100.0))


def _coverage_score(row: pd.Series) -> float:
    available = [column for column in COVERAGE_FEATURES if column in row.index]
    if not available:
        return 0.0
    present = 0
    for column in available:
        value = row[column]
        if value is None:
            continue
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if not missing and (not isinstance(value, (float, np.floating)) or math.isfinite(float(value))):
            present += 1
    return float(present / len(available))


def _ood_flag(row: pd.Series, train: pd.DataFrame) -> bool:
    """Simple support flag, not a multivariate or calibrated OOD detector."""
    if train.empty:
        return True
    for column in ("state", "crop", "district_id"):
        if column not in row.index or column not in train:
            continue
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        known = set(train[column].dropna().astype(str))
        if known and str(value) not in known:
            return True
    numeric_train = train.reindex(columns=[c for c in NUMERIC_FEATURES if c in train])
    for column in numeric_train.columns:
        value = _finite(row.get(column))
        series = pd.to_numeric(numeric_train[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if value is None or len(series) < 5:
            continue
        q01, q25, q75, q99 = series.quantile([0.01, 0.25, 0.75, 0.99]).tolist()
        spread = max(float(q75 - q25), abs(float(q25)) * 0.05, 1e-6)
        if value < q01 - 3.0 * spread or value > q99 + 3.0 * spread:
            return True
    return False


def _regression_metrics(actual: list[Any], predicted: list[Any]) -> dict[str, Any]:
    pairs = [(_finite(a), _finite(p)) for a, p in zip(actual, predicted)]
    pairs = [(a, p) for a, p in pairs if a is not None and p is not None]
    if not pairs:
        return {"n": 0, "mae": None, "rmse": None, "r2": None}
    y = np.asarray([pair[0] for pair in pairs], dtype=float)
    p = np.asarray([pair[1] for pair in pairs], dtype=float)
    residual = y - p
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if len(y) > 1 and denominator > 0 else None
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": r2,
    }


def _average_precision(actual: np.ndarray, probability: np.ndarray) -> float | None:
    positive_count = int(np.sum(actual == 1))
    if len(actual) == 0 or positive_count == 0 or positive_count == len(actual):
        return None
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(actual, probability))
    except ImportError:
        order = np.argsort(-probability, kind="stable")
        ordered = actual[order]
        precision_at_k = np.cumsum(ordered == 1) / np.arange(1, len(ordered) + 1)
        return float(np.sum(precision_at_k * (ordered == 1)) / positive_count)


def _classification_metrics(actual: list[Any], probability: list[Any]) -> dict[str, Any]:
    pairs = [(_finite(a), _finite(p)) for a, p in zip(actual, probability)]
    pairs = [(int(a), min(1.0, max(0.0, p))) for a, p in pairs if a in (0.0, 1.0) and p is not None]
    if not pairs:
        return {
            "n": 0,
            "positive_count": 0,
            "pr_auc_average_precision": None,
            "brier_score": None,
            "threshold": 0.5,
            "confusion": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            "precision": None,
            "recall": None,
            "specificity": None,
            "f1": None,
        }
    y = np.asarray([pair[0] for pair in pairs], dtype=int)
    p = np.asarray([pair[1] for pair in pairs], dtype=float)
    predicted = p >= 0.5
    actual_positive = y == 1
    tp = int(np.sum(predicted & actual_positive))
    fp = int(np.sum(predicted & ~actual_positive))
    fn = int(np.sum(~predicted & actual_positive))
    tn = int(np.sum(~predicted & ~actual_positive))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "n": int(len(y)),
        "positive_count": int(np.sum(y)),
        "pr_auc_average_precision": _average_precision(y, p),
        "brier_score": float(np.mean((p - y) ** 2)),
        "threshold": 0.5,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def _interval_width(abs_errors: list[float]) -> float | None:
    """Finite-sample absolute-residual conformal quantile from prior years."""
    values = np.asarray([value for value in abs_errors if np.isfinite(value)], dtype=float)
    if len(values) < MIN_INTERVAL_RESIDUALS:
        return None
    # Split-conformal order statistic for a nominal 90% two-sided interval.
    rank = min(len(values), max(1, math.ceil((len(values) + 1) * 0.90)))
    return float(np.sort(values)[rank - 1])


def _empty_metrics(status: str, data_status: str, profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": status,
        "data_status": data_status,
        "target": "yield_anomaly_pct",
        "risk_target": "risk_label (binary: elevated/high vs other known labels)",
        "probability_status": "unavailable",
        "temporal_validation": {
            "scheme": "expanding year-block rolling origin",
            "rule": "training season_year < validation season_year; no random split",
            "folds": [],
        },
        "yield_anomaly_metrics": {"trailing_median": _regression_metrics([], []), "random_forest": _regression_metrics([], []), "selected_forecast": _regression_metrics([], [])},
        "yield_t_ha_metrics": {"trailing_median": _regression_metrics([], []), "random_forest": _regression_metrics([], []), "selected_forecast": _regression_metrics([], [])},
        "risk_metrics": _classification_metrics([], []),
        "interval_metrics": {"n": 0, "nominal_coverage": 0.9, "empirical_coverage": None, "mean_width_t_ha": None},
        "data_profile": profile,
        "limitations": ["No forecast is released when there is no earlier labeled season to train or form a historical baseline."],
    }


def run_modeling(features: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit and evaluate year-block forecasts; write `forecasts.csv` and metrics.

    Parameters
    ----------
    features:
        One row per district/crop/season year. For synthetic demonstration data,
        set ``features.attrs['data_status'] = 'synthetic_demo'`` or provide a
        ``data_status`` column. Untagged input is treated as observed.
    output_dir:
        Directory receiving the forecast CSV and JSON metrics artifact.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Forecasts with the stable public artifact schema and its JSON-safe
        evaluation/profile document. Class probabilities are not calibrated.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    original = features
    frame = features.copy().reset_index(drop=True)
    status = _data_status(frame, original)
    profile = summarize_statistics(frame)
    forecasts = pd.DataFrame(columns=FORECAST_COLUMNS)

    def persist(result: pd.DataFrame, metrics: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        result.reindex(columns=FORECAST_COLUMNS).to_csv(destination / "forecasts.csv", index=False)
        clean = _json_clean(metrics)
        (destination / "metrics.json").write_text(
            json.dumps(clean, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return result.reindex(columns=FORECAST_COLUMNS), clean

    if frame.empty:
        metrics = _empty_metrics("empty_input", status, profile)
        metrics["limitations"].append("Input feature panel is empty.")
        return persist(forecasts, metrics)

    for column in (
        "district_id",
        "state",
        "district",
        "crop",
        "season",
        "season_year",
        "expected_yield_t_ha",
        "yield_anomaly_pct",
        "yield_t_ha",
        "risk_label",
        "issue_day",
    ):
        if column not in frame:
            frame[column] = np.nan
    frame["season_year"] = pd.to_numeric(frame["season_year"], errors="coerce")
    frame["yield_anomaly_pct"] = pd.to_numeric(frame["yield_anomaly_pct"], errors="coerce")
    frame["yield_t_ha"] = pd.to_numeric(frame["yield_t_ha"], errors="coerce")
    frame["expected_yield_t_ha"] = pd.to_numeric(frame["expected_yield_t_ha"], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame["_risk_binary"] = frame["risk_label"].map(_normalized_risk_label)
    model_x = _model_features(frame)
    feature_columns = model_x.columns.tolist()

    if frame["season_year"].notna().sum() == 0:
        metrics = _empty_metrics("insufficient_data", status, profile)
        metrics["limitations"].append("No valid season_year values; chronological validation is not possible.")
        return persist(forecasts, metrics)

    years = sorted(int(value) for value in frame["season_year"].dropna().unique())
    folds: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    anomaly_oof: dict[str, list[Any]] = {"trailing_median": [], "random_forest": [], "selected_forecast": []}
    yield_oof: dict[str, list[Any]] = {"trailing_median": [], "random_forest": [], "selected_forecast": []}
    risk_actual: list[Any] = []
    risk_probability: list[Any] = []
    interval_actual: list[float] = []
    interval_lower: list[float] = []
    interval_upper: list[float] = []
    prior_absolute_yield_errors: list[float] = []
    estimator_errors: list[str] = []

    for year in years:
        test_mask = frame["season_year"].eq(year)
        test_index = frame.index[test_mask].tolist()
        train_mask = frame["season_year"].lt(year) & frame["yield_anomaly_pct"].notna()
        train_index = frame.index[train_mask].tolist()
        if not train_index or not test_index:
            continue

        train = frame.loc[train_index]
        test = frame.loc[test_index]
        train_x = model_x.loc[train_index]
        test_x = model_x.loc[test_index]
        baseline_predictions = [_trailing_median(train, frame.loc[index]) for index in test_index]
        ml_predictions: list[float | None] = [None] * len(test_index)
        ml_status = "baseline_fallback"
        training_years = int(train["season_year"].nunique())

        if len(train) >= MIN_ML_ROWS and training_years >= MIN_ML_YEARS and feature_columns:
            try:
                regressor = _make_pipeline(feature_columns)
                regressor.fit(train_x, train["yield_anomaly_pct"].astype(float))
                candidate = regressor.predict(test_x)
                ml_predictions = [_finite(value) for value in candidate]
                ml_status = "random_forest"
            except Exception as exc:  # preserve the statistical baseline if a model fold cannot fit
                estimator_errors.append(f"regressor year {year}: {type(exc).__name__}: {exc}")
                ml_status = "model_fit_failed_baseline_used"
        else:
            ml_status = "insufficient_history_baseline_used"

        selected_predictions = [
            ml if ml is not None else baseline
            for ml, baseline in zip(ml_predictions, baseline_predictions)
        ]
        interval_width = _interval_width(prior_absolute_yield_errors)

        # Risk classifier uses only earlier-year risk labels. A constant prior
        # is allowed for one-class history but is still explicitly uncalibrated.
        classifier_train = train[train["_risk_binary"].notna()]
        classifier_x = model_x.loc[classifier_train.index]
        classifier_predictions: list[float | None] = [None] * len(test_index)
        classifier_status = "unavailable"
        if len(classifier_train) >= MIN_RISK_ROWS:
            classes = sorted(classifier_train["_risk_binary"].astype(int).unique().tolist())
            if len(classes) >= 2 and feature_columns:
                try:
                    classifier = _make_pipeline(feature_columns, classifier=True)
                    classifier.fit(classifier_x, classifier_train["_risk_binary"].astype(int))
                    probabilities = classifier.predict_proba(test_x)
                    positive_index = list(classifier.named_steps["model"].classes_).index(1)
                    classifier_predictions = [_finite(row[positive_index]) for row in probabilities]
                    classifier_status = "uncalibrated_random_forest"
                except Exception as exc:
                    estimator_errors.append(f"classifier year {year}: {type(exc).__name__}: {exc}")
                    classifier_status = "model_fit_failed"
            else:
                prior = float(classifier_train["_risk_binary"].mean())
                classifier_predictions = [prior] * len(test_index)
                classifier_status = "uncalibrated_historical_prevalence_fallback"

        fold_labeled_actual: list[float] = []
        fold_labeled_selected: list[float] = []
        for offset, index in enumerate(test_index):
            row = frame.loc[index]
            baseline = baseline_predictions[offset]
            ml_prediction = ml_predictions[offset]
            selected = selected_predictions[offset]
            expected = row.get("expected_yield_t_ha")
            expected_number = _finite(expected)
            expected_yield = expected_number if expected_number is not None and expected_number >= 0 else None
            predicted_yield = _yield_from_anomaly(expected, selected)
            lower = max(0.0, predicted_yield - interval_width) if predicted_yield is not None and interval_width is not None else None
            upper = predicted_yield + interval_width if predicted_yield is not None and interval_width is not None else None
            probability = classifier_predictions[offset]
            predicted_risk = "elevated_risk" if probability is not None and probability >= 0.5 else (
                "not_elevated" if probability is not None else None
            )

            forecast_rows.append(
                {
                    "district_id": _json_clean(row.get("district_id")),
                    "state": _json_clean(row.get("state")),
                    "district": _json_clean(row.get("district")),
                    "crop": _json_clean(row.get("crop")),
                    "season": _json_clean(row.get("season")),
                    "season_year": int(year),
                    "expected_yield_t_ha": expected_yield,
                    "predicted_yield_t_ha": predicted_yield,
                    "lower_90_t_ha": lower,
                    "upper_90_t_ha": upper,
                    "risk_probability": probability,
                    "risk_label": predicted_risk,
                    "probability_status": "uncalibrated" if probability is not None else "unavailable",
                    "issue_day": _json_clean(row.get("issue_day")),
                    "coverage_score": _coverage_score(row),
                    "ood_flag": _ood_flag(row, train),
                    "data_status": status,
                    "forecast_type": (
                        "out_of_time_backtest"
                        if _finite(row.get("yield_anomaly_pct")) is not None
                        else "forward_issue"
                    ),
                }
            )

            actual_anomaly = _finite(row.get("yield_anomaly_pct"))
            actual_yield = _finite(row.get("yield_t_ha"))
            if actual_anomaly is not None:
                anomaly_oof["trailing_median"].append(actual_anomaly)
                anomaly_oof["trailing_median"].append(baseline)
                anomaly_oof["selected_forecast"].append(actual_anomaly)
                anomaly_oof["selected_forecast"].append(selected)
                fold_labeled_actual.append(actual_anomaly)
                fold_labeled_selected.append(selected)
                if ml_prediction is not None:
                    anomaly_oof["random_forest"].append(actual_anomaly)
                    anomaly_oof["random_forest"].append(ml_prediction)
            if actual_yield is None and actual_anomaly is not None and expected_yield is not None:
                actual_yield = _yield_from_anomaly(expected_yield, actual_anomaly)
            if actual_yield is not None:
                if expected_yield is not None:
                    baseline_yield = _yield_from_anomaly(expected_yield, baseline)
                    selected_yield = predicted_yield
                    if baseline_yield is not None and selected_yield is not None:
                        yield_oof["trailing_median"].extend([actual_yield, baseline_yield])
                        yield_oof["selected_forecast"].extend([actual_yield, selected_yield])
                    if ml_prediction is not None:
                        ml_yield = _yield_from_anomaly(expected_yield, ml_prediction)
                        if ml_yield is not None:
                            yield_oof["random_forest"].extend([actual_yield, ml_yield])
                if lower is not None and upper is not None:
                    interval_actual.append(actual_yield)
                    interval_lower.append(lower)
                    interval_upper.append(upper)
            risk_target = _finite(row.get("_risk_binary"))
            if risk_target in (0.0, 1.0) and probability is not None:
                risk_actual.append(risk_target)
                risk_probability.append(probability)

        # Historical error from this completed year can calibrate later years.
        for offset, index in enumerate(test_index):
            row = frame.loc[index]
            actual_yield = _finite(row.get("yield_t_ha"))
            if actual_yield is None:
                actual_anomaly = _finite(row.get("yield_anomaly_pct"))
                expected_yield = _finite(row.get("expected_yield_t_ha"))
                actual_yield = _yield_from_anomaly(expected_yield, actual_anomaly)
            if actual_yield is None:
                continue
            expected_yield = _finite(row.get("expected_yield_t_ha"))
            prediction_yield = _yield_from_anomaly(expected_yield, selected_predictions[offset])
            if prediction_yield is not None:
                prior_absolute_yield_errors.append(abs(actual_yield - prediction_yield))

        folds.append(
            {
                "season_year": year,
                "training_rows": int(len(train)),
                "training_years": training_years,
                "validation_rows": int(len(test)),
                "validation_labeled_yield_rows": int(len(fold_labeled_actual)),
                "yield_model": ml_status,
                "risk_model": classifier_status,
            }
        )

    # The lists above store alternating actual/predicted values for compactness.
    def paired_metrics(values: list[Any]) -> dict[str, Any]:
        return _regression_metrics(values[0::2], values[1::2])

    yield_anomaly_metrics = {name: paired_metrics(values) for name, values in anomaly_oof.items()}
    yield_metrics = {name: paired_metrics(values) for name, values in yield_oof.items()}
    interval_n = len(interval_actual)
    interval_metrics = {
        "n": interval_n,
        "nominal_coverage": 0.9,
        "empirical_coverage": (
            float(np.mean((np.asarray(interval_actual) >= np.asarray(interval_lower)) & (np.asarray(interval_actual) <= np.asarray(interval_upper))))
            if interval_n
            else None
        ),
        "mean_width_t_ha": float(np.mean(np.asarray(interval_upper) - np.asarray(interval_lower))) if interval_n else None,
        "interval_note": "Symmetric split-conformal absolute-error bands use only earlier rolling-fold residuals; unavailable until at least five prior residuals.",
    }
    risk_metrics = _classification_metrics(risk_actual, risk_probability)

    if forecast_rows:
        forecast_frame = pd.DataFrame(forecast_rows).reindex(columns=FORECAST_COLUMNS)
    else:
        forecast_frame = forecasts
    probability_status = "uncalibrated" if forecast_frame["risk_probability"].notna().any() else "unavailable"
    metrics: dict[str, Any] = {
        "status": "ok" if folds else "insufficient_data",
        "data_status": status,
        "target": "yield_anomaly_pct",
        "risk_target": "risk_label (binary: elevated/high vs other known labels)",
        "probability_status": probability_status,
        "temporal_validation": {
            "scheme": "expanding year-block rolling origin",
            "rule": "training season_year < validation season_year; no random split",
            "folds": folds,
            "validation_row_count": int(sum(fold["validation_rows"] for fold in folds)),
        },
        "yield_anomaly_metrics": yield_anomaly_metrics,
        "yield_t_ha_metrics": yield_metrics,
        "risk_metrics": risk_metrics,
        "interval_metrics": interval_metrics,
        "model_configuration": {
            "yield_regressor": "scikit-learn RandomForestRegressor with fold-local imputation and categorical encoding",
            "risk_classifier": "scikit-learn RandomForestClassifier with class-weight balancing, when history supports it",
            "minimum_random_forest_history": {"rows": MIN_ML_ROWS, "distinct_years": MIN_ML_YEARS},
            "risk_probability_threshold": 0.5,
            "probability_status": "uncalibrated; no calibrator is fitted",
        },
        "data_profile": profile,
        "limitations": [
            "Any supplied synthetic demonstration target is a software fixture, not measured crop evidence or model validation.",
            "The trailing median and random forest estimate association/prediction, not causal effects or agronomic recommendations.",
            "Risk labels outside the documented binary aliases are excluded from classifier training and scoring.",
            "Coverage is a feature-presence fraction; OOD is a simple marginal-range and unseen-category heuristic.",
            "Yield intervals are omitted until earlier out-of-time residual history is sufficient; intervals are not guaranteed calibrated under drift.",
            "Point-in-time correctness of lag/weather/satellite fields must be enforced by feature generation; this module cannot verify source timestamps.",
        ],
    }
    if estimator_errors:
        metrics["model_fit_warnings"] = estimator_errors
    if not folds:
        metrics["limitations"].append("No season with a prior labeled year was available; no forecasts were emitted.")
    return persist(forecast_frame, metrics)
