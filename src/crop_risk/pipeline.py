"""Reproducible batch pipeline for synthetic demos or canonical source snapshots."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from crop_risk.data_contracts import load_input_bundle
from crop_risk.features import build_feature_panel


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_pipeline(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    bulletins_path: str | Path | None = None,
) -> dict:
    """Validate canonical data, build labels/features, train/evaluate and report.

    Production forecasts remain screening outputs; no field action is automated.
    """

    source_dir, out = Path(data_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = load_input_bundle(source_dir)
    features = build_feature_panel(bundle)
    feature_path = out / "features.csv"
    features.to_csv(feature_path, index=False)

    from crop_risk.modeling import run_modeling
    from crop_risk.statistics import summarize_statistics

    forecasts, model_metrics = run_modeling(features, out)
    statistics_summary = summarize_statistics(features)
    (out / "statistics.json").write_text(
        json.dumps(statistics_summary, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )

    nlp_metrics: dict = {"status": "not_run", "reason": "No advisory corpus supplied."}
    bulletins = Path(bulletins_path) if bulletins_path else source_dir / "bulletins.jsonl"
    if bulletins.is_file():
        from crop_risk.nlp import process_bulletins

        nlp_metrics = process_bulletins(bulletins, out)

    statuses = sorted(set(features["data_status"].dropna().astype(str)))
    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir.resolve()),
        "input_hashes": {
            "yields.csv": _file_hash(source_dir / "yields.csv"),
            "exposures.csv": _file_hash(source_dir / "exposures.csv"),
        },
        "data_status": statuses[0] if len(statuses) == 1 else "mixed_or_unknown",
        "input_rows": int(len(features)),
        "labeled_rows": int(features["risk_label"].notna().sum()),
        "risk_threshold_pct": -15.0,
        "forecast_issue_day": 90,
        "forecast_rows": int(len(forecasts)),
        "model_metrics": model_metrics,
        "statistics": statistics_summary,
        "nlp": nlp_metrics,
        "decision_boundary": (
            "District-season screening only. Not an official forecast, diagnosis, "
            "insurance decision, or farmer-level recommendation."
        ),
    }
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, default=str), encoding="utf-8")
    return run_manifest


def run_demo_pipeline(project_root: str | Path | None = None) -> dict:
    """Generate and run the included synthetic demo from the project root."""

    from crop_risk.demo_data import generate_demo_data

    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    demo_dir = root / "data" / "demo"
    # Keep checked-in/generated fixtures aligned with the current generator.
    generate_demo_data(demo_dir)
    return run_pipeline(demo_dir, root / "data" / "processed", bulletins_path=demo_dir / "bulletins.jsonl")
