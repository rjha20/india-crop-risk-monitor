# Runbook — CropLens India

## Local dashboard

Use Python 3.12 and uv from the project root:

```powershell
uv sync --locked
uv run streamlit run dashboard/app.py
```

The dashboard attempts to read current APY/UPAG rows through the public India Data Portal CKAN API. It
caches a successful response for six hours. If the API is unavailable, the app verifies and reads
`data/public/upag_crop_production.csv.gz` against `data/public/manifest.json` and labels the page as a
snapshot fallback.

## Refresh the committed fallback snapshot

```powershell
uv run python scripts/refresh_live_snapshot.py
```

The refresh resolves the resource from the dataset slug, verifies that source metadata still reports the
Open Data Commons Attribution License, pages the full table, validates the row count and schema, and writes
a compressed CSV and manifest. Review the updated `data/public/manifest.json` and `ATTRIBUTION.md`, then
review the data source's current terms before committing a new vintage. A changed license, schema, or
record count stops refresh rather than silently publishing partial data.

## Container

```powershell
docker compose up --build
```

The container binds to `127.0.0.1:8501` for local access. Put it behind a managed HTTPS reverse proxy if
running it on a server. The Streamlit process does not expose an action API or a write endpoint.

## Streamlit Community Cloud publication

The project is prepared for Community Cloud, but it has not been published yet.

1. Publish the reviewed code and attributed fallback snapshot to a public repository owned by `rjha20`.
2. Sign in to Streamlit Community Cloud with the same `rjha20` GitHub account and grant access to that repo.
3. Create an app using the repository's default branch, entrypoint `dashboard/app.py`, and Python 3.12.
4. Wait for build completion, open the resulting public URL, and inspect the first-render source status and
   source attribution.
5. Confirm the app can fall back to the committed snapshot when the data source is unreachable.

The current dashboard requires no API key or Streamlit secret. Never add a GitHub token, account password,
or other secret to this repository. Streamlit Community Cloud uses GitHub authentication and requires
repository permissions for app administration.

## Monitoring and stop conditions

- Watch the latest source vintage, retrieval time, row count, year range, missing-yield rows, duplicate
  keys, and schema/license checks.
- If a source field, license, unit, district key, or reporting convention changes, stop refresh and review
  the source before accepting new output.
- If source access fails, the dashboard identifies snapshot mode and shows its retrieval date; do not
  present that mode as a current source refresh.
- Review model performance by crop/state/year and compare average precision with event prevalence. Treat
  deteriorating or unstable results as a reason to withhold model interpretation.
- Keep the displayed 15% anomaly threshold identified as an exploratory project threshold.
- No forecasts or action recommendations are released. Stop if someone proposes using these retrospective
  rows for a farm, insurance, credit, public-benefit, or policy decision without separate data and review.

## Separate offline modules

The synthetic fixture pipeline, read-only FastAPI prototype, advisory NLP extraction, and leaf-image CV
benchmark remain separate from the online APY/UPAG dashboard. Their outputs are not loaded into the public
app. See `PROJECT_PLAN.md` for the broader future forecasting blueprint.
