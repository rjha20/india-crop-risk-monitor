# CropLens India — crop production and harvest-risk monitor

CropLens India is a public-facing data product for exploring district-level crop production, yield history,
and observed low-yield anomalies. The dashboard reads real annual APY/UPAG records, preserves source
attribution, and includes a time-ordered machine-learning backtest.

> **Scope:** this application is a retrospective analyst tool. It is not an operational crop forecast, an
> agronomic advisory service, or a source of farmer, insurance, credit, relief, or policy decisions.

## Current release

- Public APY/UPAG source integration through the India Data Portal's CKAN DataStore API, with a manual
  refresh control and a six-hour in-app cache.
- A verified compressed fallback snapshot in `data/public/`, with source/license attribution and a SHA-256
  manifest. The snapshot contains 156,189 rows for 34 states/UTs and 766 district identifiers, from 2013-14 through
  2024-25, retrieved on 14 October 2026.
- Historical yield anomaly screening: compare each row with a prior-only median from up to five years;
  require at least three prior observations; flag values at least 15% below that median.
- Random-forest evaluation on later year blocks using only lagged yield and area history. The app reports
  out-of-time metrics against the observed event prevalence and labels model scores uncalibrated.
- Source schema/duplicate/missingness indicators, filtered record export, deployment instructions, and a
  model card with explicit use limits.

The source data is annual and the newest outcome vintage in this snapshot is 2024-25. The dashboard does
not claim to know current 2026 field conditions. Public deployment is prepared for Streamlit Community Cloud;
the repository and app still need to be published under the requested `rjha20` account.

## Run locally

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --locked
uv run streamlit run dashboard/app.py
```

To fetch a fresh validated source snapshot and replace the bundled fallback:

```powershell
uv run python scripts/refresh_live_snapshot.py
```

The dashboard refreshes the public API at startup when its six-hour cache expires. If the source is
unavailable, it displays the checksum-verified bundled snapshot and shows that fallback status in the UI.
No API key or secret is required by the current public CKAN DataStore endpoint.

## Data and attribution

The dashboard uses the Ministry of Agriculture & Farmers Welfare APY/UPAG crop-wise production dataset,
catalogued by the [India Data Portal](https://indiadataportal.com/p/area-production-yield-apy). The source
metadata reports the [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/index.html).
The snapshot attribution and retrieval manifest are in `data/public/`. The live loader checks that the source
still reports an ODC Attribution license before refreshing; a license metadata change stops refresh and
leaves the attributed snapshot available for review.

The current API host is `ckandev.indiadataportal.com`; the source page and CKAN resource can change. The
six-hour cache, schema checks, and snapshot fallback protect the dashboard from short outages, but this
public source has no availability guarantee from this project. Recheck the endpoint and license before
releasing a new data vintage.

## Project contents

```text
dashboard/app.py                     Public read-only dashboard
src/crop_risk/live_data.py            CKAN source discovery, paging, and snapshot validation
src/crop_risk/production_analysis.py  Prior-only anomalies and chronological ML evaluation
scripts/refresh_live_snapshot.py     Attributed snapshot refresh command
data/public/                         Licensed APY/UPAG fallback snapshot and manifest
data/demo/                           Separate synthetic fixtures for offline software-path demos
docs/model_card.md                   Model target, evaluation, limits, and intended use
docs/runbook.md                      Local operation and Streamlit deployment path
PROJECT_PLAN.md                      Extended future forecasting project blueprint
```

The offline `crop-risk demo` workflow remains useful for exercising older pipeline components. Its synthetic
results are separate from the observed APY/UPAG dashboard and are never shown as crop data in the app.

## Deployment

1. Publish this project as a **public** GitHub repository owned by `rjha20`.
2. Connect Streamlit Community Cloud using the same GitHub account.
3. Select the repository's `main` branch and `dashboard/app.py` as the app entrypoint; select Python 3.12.
4. Review the app URL, source status, attribution, and deployment logs after the first launch.

Detailed steps and current publication status are in `docs/runbook.md`. No GitHub or hosting secrets belong
in this repository; the current data endpoint does not require one.

## Analytical boundaries

- Yield declines are retrospective screening indicators; they do not establish a cause.
- The 15% threshold is a project threshold for exploratory analysis, not an agronomist-approved standard.
- The model uses historical yield and area only. It cannot react to a current weather event, crop condition,
  irrigation, soil, market, or pest outbreak.
- Out-of-time performance may vary by crop, state, reporting coverage, district boundary, and year. Scores
  are not calibrated probabilities or evidence of field performance.
- NLP bulletin extraction and leaf-image computer vision remain separate research modules. The online
  district dashboard does not fuse those outputs because the dataset lacks defensible point-in-time joins.

See `docs/data_sources.md`, `docs/data_policy.md`, and `docs/model_card.md` before interpreting or sharing
analysis from the project.
