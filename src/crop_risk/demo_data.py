"""Generate a small, deterministic synthetic dataset for offline demonstration."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

DISTRICTS = [
    ("BR-PAT", "Bihar", "Patna"),
    ("BR-GAY", "Bihar", "Gaya"),
    ("BR-PUR", "Bihar", "Purnia"),
    ("BR-MUZ", "Bihar", "Muzaffarpur"),
    ("OD-CUT", "Odisha", "Cuttack"),
    ("OD-BAL", "Odisha", "Balasore"),
    ("OD-KOR", "Odisha", "Koraput"),
    ("UP-LKO", "Uttar Pradesh", "Lucknow"),
    ("UP-GOR", "Uttar Pradesh", "Gorakhpur"),
    ("UP-VAR", "Uttar Pradesh", "Varanasi"),
]
CROPS = {
    "rice": {"base_yield": 3.2, "season": "kharif", "rainfall": 900.0},
    "wheat": {"base_yield": 3.0, "season": "rabi", "rainfall": 420.0},
    "maize": {"base_yield": 2.7, "season": "kharif", "rainfall": 650.0},
}
YEARS = range(2008, 2026)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_demo_data(output_dir: str | Path, seed: int = 2417) -> dict:
    """Create synthetic yields, issue-time exposures and synthetic bulletins.

    No row represents a real district observation. IDs and values are fixtures
    for exercising pipeline behavior only.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    yields: list[dict] = []
    exposures: list[dict] = []
    bulletins: list[dict] = []

    for district_index, (district_id, state, district) in enumerate(DISTRICTS):
        crops = list(CROPS)
        for crop_index, crop in enumerate(crops):
            spec = CROPS[crop]
            district_baseline = spec["base_yield"] * (0.88 + district_index * 0.025)
            for year in YEARS:
                drought_year = year % 7 == 2
                wet_year = year % 9 == 4
                rainfall_anomaly = rng.gauss(0.0, 15.0)
                if drought_year:
                    rainfall_anomaly -= rng.uniform(20.0, 38.0)
                if wet_year:
                    rainfall_anomaly += rng.uniform(25.0, 45.0)
                rainfall_anomaly = round(max(-65.0, min(80.0, rainfall_anomaly)), 2)
                rainfall_mm = max(30.0, spec["rainfall"] * (1 + rainfall_anomaly / 100))
                dry_spell = max(0, int(rng.gauss(9 + max(0, -rainfall_anomaly) / 3, 4)))
                heat_days = max(0, int(rng.gauss(5 + (year % 5 == 0) * 5, 3)))
                excess_rain_days = max(0, int(rng.gauss(2 + max(0, rainfall_anomaly - 25) / 8, 2)))
                ndvi_mean = max(
                    0.15,
                    min(
                        0.9,
                        0.68
                        + rainfall_anomaly * 0.0011
                        - max(0, heat_days - 6) * 0.009
                        + rng.gauss(0, 0.035),
                    ),
                )
                ndvi_trend = round(rng.gauss(0.002, 0.018) - max(0, -rainfall_anomaly) * 0.00025, 4)
                yield_penalty = (
                    max(0.0, -rainfall_anomaly - 8.0) * 0.004
                    + max(0.0, rainfall_anomaly - 35.0) * 0.0023
                    + max(0, heat_days - 6) * 0.012
                    + max(0, dry_spell - 12) * 0.003
                )
                trend = 1.0 + (year - 2008) * 0.006
                yield_value = district_baseline * trend * (1 - min(0.48, yield_penalty))
                yield_value += rng.gauss(0, district_baseline * 0.055)
                yield_value = round(max(0.35, yield_value), 3)
                area_ha = round(900 + district_index * 113 + crop_index * 67 + rng.uniform(-100, 100), 1)
                key = {
                    "district_id": district_id,
                    "state": state,
                    "district": district,
                    "crop": crop,
                    "season": spec["season"],
                    "season_year": year,
                }
                yields.append(
                    {
                        **key,
                        "area_ha": area_ha,
                        "production_t": round(area_ha * yield_value, 3),
                        "yield_t_ha": yield_value,
                        "data_status": "synthetic_demo",
                        "source_id": "SYNTHETIC_FIXTURE",
                    }
                )
                exposures.append(
                    {
                        **key,
                        "rainfall_mm": round(rainfall_mm, 2),
                        "rainfall_anomaly_pct": rainfall_anomaly,
                        "dry_spell_days": dry_spell,
                        "heat_days": heat_days,
                        "excess_rain_days": excess_rain_days,
                        "ndvi_mean": round(ndvi_mean, 4),
                        "ndvi_trend": ndvi_trend,
                        "ndvi_valid_obs": rng.randint(5, 10),
                        "coverage_score": round(rng.uniform(0.78, 1.0), 3),
                        "issue_day": 90,
                        "feature_end_day": 90,
                        "data_status": "synthetic_demo",
                        "source_id": "SYNTHETIC_FIXTURE",
                    }
                )
                if year >= 2021 and (year + district_index + crop_index) % 3 == 0:
                    hazard = (
                        "rainfall deficit"
                        if rainfall_anomaly < -15
                        else "excess rainfall"
                        if rainfall_anomaly > 30
                        else "heat stress"
                        if heat_days > 8
                        else "crop monitoring"
                    )
                    text = (
                        "SYNTHETIC DEMONSTRATION TEXT — NOT AN OFFICIAL ADVISORY. "
                        f"For {crop} during {spec['season']} in {district}, review {hazard} indicators. "
                        "This fixture contains no field instruction or real bulletin recommendation."
                    )
                    bulletins.append(
                        {
                            "bulletin_id": f"SYN-{district_id}-{crop}-{year}",
                            "issue_date": f"{year}-07-15" if spec["season"] == "kharif" else f"{year}-01-15",
                            "state": state,
                            "district": district,
                            "crop": crop,
                            "language": "en",
                            "text": text,
                            "source_url": "",
                            "data_status": "synthetic_demo",
                        }
                    )

            # A separate exposure-only issue row exercises forward forecasting.
            # It intentionally has no harvested outcome and remains synthetic.
            issue_year = max(YEARS) + 1
            issue_rainfall_anomaly = round(rng.gauss(-3.0, 17.0), 2)
            issue_rainfall_anomaly = max(-65.0, min(80.0, issue_rainfall_anomaly))
            issue_heat_days = max(0, int(rng.gauss(8.0, 3.0)))
            issue_dry_spell = max(0, int(rng.gauss(10 + max(0, -issue_rainfall_anomaly) / 3, 4)))
            issue_ndvi = max(
                0.15,
                min(
                    0.9,
                    0.68
                    + issue_rainfall_anomaly * 0.0011
                    - max(0, issue_heat_days - 6) * 0.009
                    + rng.gauss(0, 0.035),
                ),
            )
            issue_key = {
                "district_id": district_id,
                "state": state,
                "district": district,
                "crop": crop,
                "season": spec["season"],
                "season_year": issue_year,
            }
            exposures.append(
                {
                    **issue_key,
                    "rainfall_mm": round(spec["rainfall"] * (1 + issue_rainfall_anomaly / 100), 2),
                    "rainfall_anomaly_pct": issue_rainfall_anomaly,
                    "dry_spell_days": issue_dry_spell,
                    "heat_days": issue_heat_days,
                    "excess_rain_days": max(0, int(rng.gauss(3.0, 2.0))),
                    "ndvi_mean": round(issue_ndvi, 4),
                    "ndvi_trend": round(rng.gauss(0.0, 0.018), 4),
                    "ndvi_valid_obs": rng.randint(5, 10),
                    "coverage_score": round(rng.uniform(0.78, 1.0), 3),
                    "issue_day": 90,
                    "feature_end_day": 90,
                    "data_status": "synthetic_demo",
                    "source_id": "SYNTHETIC_FIXTURE",
                }
            )

    _write_csv(out / "yields.csv", yields)
    _write_csv(out / "exposures.csv", exposures)
    with (out / "bulletins.jsonl").open("w", encoding="utf-8") as handle:
        for row in bulletins:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "data_status": "synthetic_demo",
        "synthetic": True,
        "seed": seed,
        "historical_yield_years": [min(YEARS), max(YEARS)],
        "exposure_years": [min(YEARS), max(YEARS) + 1],
        "forward_issue_year": max(YEARS) + 1,
        "yield_rows": len(yields),
        "exposure_rows": len(exposures),
        "bulletin_rows": len(bulletins),
        "warning": "Generated fixtures are not real measurements and must not support field or policy decisions.",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
