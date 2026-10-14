"""Optional read-only API for the simulated crop-risk demo artifacts.

Run with ``uvicorn crop_risk.api:create_app --factory`` after installing the API extra.
FastAPI is imported only when ``create_app`` is called. If demo outputs are missing,
the demo pipeline is loaded lazily on the first metadata/forecast request.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
FORECAST_PATH = PROCESSED / "forecasts.csv"
METRICS_PATH = PROCESSED / "metrics.json"
DEFAULT_MIN_COVERAGE = 0.70
FILTER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _./()'\-]{0,99}$")
PREDICTION_COLUMNS = (
    "expected_yield_t_ha",
    "predicted_yield_t_ha",
    "lower_90_t_ha",
    "upper_90_t_ha",
    "risk_probability",
    "risk_label",
)
REQUIRED_COLUMNS = {
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
}


class ArtifactUnavailable(RuntimeError):
    """Raised when a valid demo artifact set cannot be read or created."""


def _coverage_threshold(metrics: dict[str, Any]) -> float:
    keys = ("min_coverage_score", "minimum_coverage_score", "coverage_threshold", "min_coverage")

    def search(value: Any) -> float | None:
        if isinstance(value, dict):
            for key in keys:
                if key in value:
                    try:
                        result = float(value[key])
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(result):
                        return result / 100 if 1 < result <= 100 else result
            for child in value.values():
                result = search(child)
                if result is not None:
                    return result
        elif isinstance(value, list):
            for child in value:
                result = search(child)
                if result is not None:
                    return result
        return None

    threshold = search(metrics)
    if threshold is None or not 0 <= threshold <= 1:
        return DEFAULT_MIN_COVERAGE
    return threshold


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "ood", "out_of_distribution"}


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_value(value: Any) -> Any:
    """Convert pandas/numpy scalar values into JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            return str(value)
    if pd.isna(value):
        return None
    return str(value)


def _safe_row(row: pd.Series, minimum_coverage: float) -> dict[str, Any]:
    """Return full predictions only after all publication gates pass."""
    coverage = _finite_number(row.get("coverage_score"))
    reasons: list[str] = []
    if coverage is None:
        reasons.append("coverage score is missing")
    elif coverage < minimum_coverage:
        reasons.append(f"coverage score is below {minimum_coverage:.0%}")
    if _is_true(row.get("ood_flag")):
        reasons.append("out-of-distribution flag is set")

    identifiers = {
        name: _json_value(row.get(name))
        for name in (
            "district_id",
            "state",
            "district",
            "crop",
            "season",
            "season_year",
            "issue_day",
            "forecast_type",
        )
    }
    status = {
        "data_status": "SIMULATED_DEMO",
        "operational_status": "NON_OPERATIONAL",
        "forecast_status": "SUPPRESSED" if reasons else "SIMULATED_DEMO",
        "coverage_score": coverage,
        "ood_flag": _is_true(row.get("ood_flag")),
    }
    if reasons:
        status["suppression_reasons"] = reasons
        return {**identifiers, **status, **{column: None for column in PREDICTION_COLUMNS}}

    result = {name: _json_value(row.get(name)) for name in REQUIRED_COLUMNS if name in row.index}
    result.update(identifiers)
    result.update(status)

    probability_status = str(row.get("probability_status", "")).strip().lower()
    if probability_status in {"", "unavailable", "not_available", "suppressed", "uncalibrated", "not_calibrated"}:
        result["risk_probability"] = None
        result["risk_label"] = None
    return result


def create_app():
    """Build the FastAPI app, importing FastAPI only when the optional API is used."""
    try:
        from fastapi import FastAPI, HTTPException, Query
    except ImportError as exc:  # pragma: no cover - depends on optional runtime installation
        raise RuntimeError("Install FastAPI to enable the optional crop-risk API.") from exc

    app = FastAPI(
        title="India Crop Risk Demo API",
        version="0.1.0",
        description=(
            "Read-only API for synthetic crop-risk demonstration artifacts. "
            "Every response is explicitly non-operational."
        ),
    )

    def ensure_artifacts() -> tuple[pd.DataFrame, dict[str, Any]]:
        if not FORECAST_PATH.is_file() or not METRICS_PATH.is_file():
            # Import the demo pipeline only when its outputs are absent.
            try:
                from crop_risk.pipeline import run_demo_pipeline

                run_demo_pipeline()
            except Exception as exc:
                raise ArtifactUnavailable("Demo artifacts are missing and could not be generated.") from exc
        try:
            forecasts = pd.read_csv(FORECAST_PATH)
            metrics_value = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise ArtifactUnavailable("Demo artifacts are missing or malformed.") from exc
        if not isinstance(metrics_value, dict):
            raise ArtifactUnavailable("The metrics artifact must contain a JSON object.")
        missing = REQUIRED_COLUMNS - set(forecasts.columns)
        if missing:
            raise ArtifactUnavailable("The forecast artifact does not match the documented schema.")
        return forecasts, metrics_value

    def report_unavailable(exc: ArtifactUnavailable) -> None:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    def validate_text_filter(name: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or len(normalized) > 100 or not FILTER_PATTERN.fullmatch(normalized):
            raise HTTPException(status_code=422, detail=f"Invalid {name} filter.")
        return normalized

    @app.get("/health")
    def health() -> dict[str, Any]:
        # Health stays a read-only probe and does not trigger artifact generation.
        ready = FORECAST_PATH.is_file() and METRICS_PATH.is_file()
        return {
            "status": "ok" if ready else "degraded",
            "artifacts_ready": ready,
            "data_status": "SIMULATED_DEMO",
            "operational_status": "NON_OPERATIONAL",
        }

    @app.get("/metadata")
    def metadata() -> dict[str, Any]:
        try:
            forecasts, metrics = ensure_artifacts()
        except ArtifactUnavailable as exc:
            report_unavailable(exc)
        return {
            "data_status": "SIMULATED_DEMO",
            "operational_status": "NON_OPERATIONAL",
            "forecast_rows": int(len(forecasts)),
            "fields": sorted(forecasts.columns.tolist()),
            "coverage_suppression_threshold": _coverage_threshold(metrics),
            "metrics": metrics,
        }

    @app.get("/forecasts")
    def get_forecasts(
        state: str | None = Query(default=None, max_length=100),
        district_id: str | None = Query(default=None, max_length=100),
        district: str | None = Query(default=None, max_length=100),
        crop: str | None = Query(default=None, max_length=100),
        season: str | None = Query(default=None, max_length=100),
        season_year: int | None = Query(default=None, ge=1900, le=2100),
    ) -> dict[str, Any]:
        state_value = validate_text_filter("state", state)
        district_id_value = validate_text_filter("district_id", district_id)
        district_value = validate_text_filter("district", district)
        crop_value = validate_text_filter("crop", crop)
        season_value = validate_text_filter("season", season)
        try:
            forecasts, metrics = ensure_artifacts()
        except ArtifactUnavailable as exc:
            report_unavailable(exc)
        filters = {
            "state": state_value,
            "district_id": district_id_value,
            "district": district_value,
            "crop": crop_value,
            "season": season_value,
            "season_year": season_year,
        }
        filtered = forecasts.copy()
        for column, value in filters.items():
            if value is None:
                continue
            if column == "season_year":
                values = pd.to_numeric(filtered[column], errors="coerce")
                filtered = filtered[values == value]
            else:
                filtered = filtered[filtered[column].astype(str).str.casefold() == value.casefold()]

        threshold = _coverage_threshold(metrics)
        items = [_safe_row(row, threshold) for _, row in filtered.iterrows()]
        suppressed = sum(item["forecast_status"] == "SUPPRESSED" for item in items)
        return {
            "data_status": "SIMULATED_DEMO",
            "operational_status": "NON_OPERATIONAL",
            "coverage_suppression_threshold": threshold,
            "count": len(items),
            "suppressed_count": suppressed,
            "forecasts": items,
        }

    return app
