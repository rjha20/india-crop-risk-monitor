# CropLens India: observed-data crop production and yield-risk analysis

## Executive summary

The released Streamlit dashboard analyzes real annual Ministry of Agriculture & Farmers Welfare APY/UPAG records obtained through the India Data Portal CKAN API. It combines source validation, historical anomaly screening and chronological random-forest evaluation. This report replaces the earlier synthetic-demo case-study template; synthetic fixtures belong to a separate offline software workflow.

## Dataset and release

The bundled snapshot was retrieved on 14 October 2026 and contains **156,189 rows**, covering **2013-14 through 2024-25**, **34 states/UTs**, **766 district identifiers**, **30 crop identifiers** and **3 seasons**. It contains 1,126 missing yields (0.72%), zero duplicate analysis keys and zero negative reported yields. District identifiers are source identifiers, not a validated geographic boundary crosswalk.

The app uses a six-hour cache and manual refresh, checks schema and source license metadata, and falls back to a SHA-256-verified snapshot when the API is unavailable. Source attribution and manifest remain in `data/public/`. See [snapshot analysis](production_snapshot_analysis.md) for provenance and detailed quality counts.

## Features and target

The unit is district/crop/season/reporting year. A low-yield event is yield at least 15% below the median of available observations in the five preceding calendar years, requiring at least three observations. The panel includes 98,907 rows with sufficient history, of which 17,245 meet this historical flag.

Features include yield lags 1-3, prior area, prior median and standard deviation, and prior yield trend. Lag checks enforce expected year offsets; the baseline window uses actual years rather than assuming adjacent rows are consecutive years. Missing years therefore do not silently extend the five-year reference window. No current-year outcome is used as a predictor.

## Evaluation and results

Each expanding-year fold trains only on earlier reporting years. The default Rice benchmark pools 2,780 held-out rows from the final three years.

| Reporting year | Rows | Event rate | AP | ROC-AUC | Brier score |
|---|---:|---:|---:|---:|---:|
| 2022-23 | 1,012 | 0.095 | 0.136 | 0.597 | 0.200 |
| 2023-24 | 891 | 0.158 | 0.292 | 0.645 | 0.192 |
| 2024-25 | 877 | 0.139 | 0.476 | 0.792 | 0.171 |
| Pooled | 2,780 | 0.129 | 0.285 | 0.681 | 0.188 |

The pooled AP is approximately **2.2 times the prevalence baseline**. This is an AP-to-prevalence ratio, not top-decile lift or intervention impact. The model Brier score (0.188) is worse than the constant-prevalence baseline (0.112); probabilities remain uncalibrated. Ranking performance varies by year. These results establish a retrospective benchmark, not validated prospective crop forecasting.

## Released app versus offline research

| Component | Status |
|---|---|
| `dashboard/app.py`, `live_data.py`, `production_analysis.py` | Released observed-data dashboard and evaluation |
| `data/public/` | Attributed APY/UPAG snapshot and manifest |
| `data/demo/` and `crop-risk demo` | Synthetic fixtures for offline software-path exercises; not the released source data |
| Weather, satellite, advisory NLP and leaf-disease vision | Separate research/demo components; not linked to or fused into the released annual-data model |

The demo fixtures are retained for reproducibility of the offline workflow. Their presence does not mean the public dashboard uses synthetic crop outcomes. Future modules in the project blueprint are not completed production capabilities.

## Reproduction and interpretation

From the repository root, run `uv sync --locked` then `uv run streamlit run dashboard/app.py`. The dashboard exposes the source vintage, fallback status and chronological benchmark. `uv run python scripts/refresh_live_snapshot.py` refreshes the attributed source snapshot; a later snapshot may produce different counts and metrics.

Use this app to explore historical yield patterns and inspect model behavior. Annual data do not describe current field conditions. Weather/satellite fusion, causal loss attribution, independent future validation and reliable calibrated probabilities remain future work. The app should not determine agronomic advice, insurance, credit or relief eligibility.

See [model card](../docs/model_card.md), [runbook](../docs/runbook.md), [source and results](production_snapshot_analysis.md), and [implementation](../src/crop_risk/production_analysis.py).
