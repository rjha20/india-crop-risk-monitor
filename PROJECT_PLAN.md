# India Crop-Loss and Harvest-Risk Forecasting

> **Implementation status (2026-10-14):** The current dashboard release candidate loads observed Ministry APY/UPAG crop-production records through the India Data Portal CKAN API, carries a licensed fallback snapshot with attribution/checksum, screens historical yield declines, and supports chronological random-forest evaluation. Its public source repository is [rjha20/india-crop-risk-monitor](https://github.com/rjha20/india-crop-risk-monitor). Streamlit hosting still requires an app to be created from a Community Cloud session authorized as `rjha20`. It does **not** forecast current crop loss or include point-in-time weather/satellite features, an operational crop calendar, agronomist review, NLP bulletin integration, or field-validated computer vision. The wider phases below remain a future blueprint; see `README.md`, `docs/model_card.md`, and `docs/runbook.md` for the delivered scope and release boundary.

**End-to-end implementation blueprint**

**Audience:** data science, geospatial, agronomy, and product teams

**Type:** portfolio-grade research and decision-support prototype

**Planning estimate:** 12–16 calendar weeks; 300–425 focused person-hours for one experienced practitioner

**Data status:** sources below are verified entry points to investigate, not a claim that all data have been downloaded, harmonized, or cleared for every reuse.

---

## 1. Executive problem and decision framing

Crop losses from rainfall deficit or excess, heat, flooding, pests, and disease may become apparent only after crops have suffered substantial damage. District yield statistics support planning but generally arrive after harvest. Satellite and weather observations may provide earlier signals; agricultural advisories can add context; crop photographs support a separate computer-vision benchmark.

Build a reproducible system that estimates, before harvest, which **crop–district–season** combinations are at elevated risk of finishing below expected yield. Combine official crop outcomes, weather/rainfall exposures, satellite crop-condition proxies, statistical and ML forecasts with uncertainty, NLP over agricultural advisories, a separately evaluated leaf-disease classifier, and an analyst-facing map/dashboard.

### Decision and user

The primary user is an agricultural planning analyst deciding **which districts and crop seasons merit closer field verification or additional analyst attention**. The forecast is a screening signal. It is not an instruction to distribute aid, determine insurance eligibility, prescribe chemicals, or advise an individual farmer.

Each forecast should answer: What yield and interval are predicted? What is the probability of a materially low yield? Which observations influenced the estimate? How complete and current is the evidence? Which source records and advisory passages support the context?

India’s FASAL program develops crop forecasts at national, state, and district levels using econometric, agrometeorological, and remote-sensing methods. This demonstrates the real operational need; it does **not** establish that FASAL inputs, crop masks, or forecasts are open downloads. DES publishes Area, Production and Yield reports that can ground retrospective evaluation. See [FASAL program description](https://desagri.gov.in/divisions-cell/agricultural-statistics-division/) and [DES APY portal](https://data.desagri.gov.in/website/crops-apy-report-web).

### Candidate pilot

- Crops: rice/paddy and wheat.
- Geography: selected districts in Bihar and Odisha for rice, and Uttar Pradesh for wheat.
- Forecast unit: district × crop × agricultural season × crop year.
- History target: at least 10 consistent seasons per district/crop; aim for 15–20 where available.
- Issue points: 30, 60, and 90 days after a documented local sowing-window start; optionally 4–8 weeks pre-harvest.

These are candidates, not verified coverage. Prove that crop labels, seasons, yield series, boundaries, and satellite coverage align. If readiness fails, narrow to one crop/state rather than fabricate data.

### Stakeholders

| Stakeholder | Responsibility |
|---|---|
| Agricultural planning analyst | Primary user; interprets risk, uncertainty, and evidence |
| Agronomist/crop scientist | Validates crop calendars, target threshold, and failure cases |
| Statistician/data steward | Preserves sampling, outcome vintages, geography, and definitions |
| Remote-sensing analyst | Builds crop masks and satellite features; reviews image coverage |
| Data scientist/ML engineer | Owns point-in-time features, validation, calibration, and model versioning |
| NLP/CV reviewer | Audits multilingual extraction and image-model generalization |
| Product/operations owner | Reviews releases, feedback, monitoring, and suppression gates |

Require a second reviewer to approve model promotion and changes to the risk threshold. Outputs never trigger automated real-world actions.

---

## 2. Outcome, risk definition, and horizon

For district d, crop c, season s, and year t:

- Observed yield = official production divided by official harvested area, standardized to tonnes per hectare.
- Expected yield = robust district/crop trend estimated only from years available before the forecast year. Start with a Theil–Sen trend and compare it with a trailing median; consider hierarchical partial pooling when histories are short.
- Yield anomaly = 100 × (observed yield − expected yield) / expected yield.
- High-risk label = final anomaly ≤ **−15%** versus the training-only expected-yield baseline. This is a provisional, pre-registered threshold; publish sensitivity at −10% and −20%, then seek agronomist review.

Use published final yield for retrospective labels, preserving publication date and estimate vintage. Never pretend a revised current estimate was known at an earlier issue date.

At each issue point predict final yield, 50% and 90% intervals, probability of the high-risk label, lead time, model version, data cutoff, and an input-coverage/out-of-distribution flag. Only features available by that timestamp are allowed.

The [MOSPI NSS 77th Round Report 587](https://www.mospi.gov.in/sites/default/files/publication_reports/Report_587m_0.pdf) records household crop-loss experience and reported reasons, including inadequate rainfall/drought and disease/insects. It is contextual survey evidence, not a district-season yield panel or a substitute target.

Out of scope: household/farm predictions; insurance/compensation eligibility; autonomous agronomic prescriptions; claims that leaf benchmarks represent disease prevalence; or presenting this as a FASAL forecast.

---

## 3. Data inventory, access, and limitations

Record retrieval date, URL, terms, version, coverage, unit, checksum, and redistribution decision for each source. Re-check current terms before distributing data or derived products.

| Data | Candidate source and purpose | Access and representativeness limitations |
|---|---|---|
| Crop outcomes | [DES APY reports](https://data.desagri.gov.in/website/crops-apy-report-web); [OGD district-wise season-wise production catalog](https://www.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0). District/crop/year area and production; derive yield. | DES offers crop, season, state, district, and year filters. OGD metadata describes annual records and NDSAP release. Capture the actual API/download resource, vintage, terms, and revisions. Coverage and definitions may vary. |
| FASAL context | [DES FASAL description](https://desagri.gov.in/divisions-cell/agricultural-statistics-division/). | Program description and methodology context, not a guaranteed open archive of model inputs, crop masks, or predictions. |
| Crop-loss survey | [MOSPI NSS report](https://www.mospi.gov.in/sites/default/files/publication_reports/Report_587m_0.pdf). | Household survey context only. Respect sample design/weights; do not create district labels without compatible geography and valid inference. |
| Satellite | [Copernicus Sentinel-2](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2). Multispectral 10/20/60 m observations for vegetation/crop-stage proxies. | Copernicus describes data as free and open, subject to exceptional restrictions. Read the current legal notice and attribute. Clouds, haze, revisit gaps, and fragmented plots limit coverage. Pixels are not yield labels. |
| Vegetation indices | [NASA MODIS VI products](https://modis.gsfc.nasa.gov/data/dataprod/mod13.php). NDVI/EVI seasonal trajectories and broad cross-check. | Regular 16-day composites, but coarse pixels mix crops and land cover in smallholder landscapes. Use quality flags and respect Earthdata terms. |
| Weather | [IMD gridded data](https://rcc.imdpune.gov.in/download.php); [IMD Data Service Portal](https://dsp.imdpune.gov.in/); optional [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download). | IMD documents 0.25° daily merged satellite-gauge rainfall and IMDAA reanalysis for 1979–2018. Historical station data can require registration/fees and have reproduction terms. ERA5-Land lists CC-BY, requiring attribution. Grids are not field observations. |
| Weather forecast | [IMD API reference](https://api.imd.gov.in/public/api_reference.html); [IMD Agromet service](https://mausam.imd.gov.in/responsive/agromet_adv_ser_block_current_en.php). | Verify API access, authentication, rate limits, retention, and terms. Realized historical weather cannot substitute for an issue-time forecast. |
| Advisory NLP | [IMD Agromet portal](https://imdagrimet.gov.in/); [district bulletins](https://mausam.imd.gov.in/responsive/agromet_adv_ser_block_current_en.php). | Bulletins may be current/past and multilingual. Archive completeness, bulk retrieval, and reuse rights vary. Begin with a bounded date-stamped corpus; preserve provenance and do not crawl beyond terms. |
| Disease CV | [PlantVillage repository](https://github.com/spMohanty/PlantVillage-Dataset); [original paper](https://doi.org/10.3389/fpls.2016.01419). | Repository describes 54,306 images, 14 crop species, 26 diseases. Check dataset-specific rights; public availability is not automatic permission for commercial reuse. Controlled backgrounds create a major Indian field-domain gap. |

### Data-readiness gate by week 2

Confirm consistent crop/season codes; ten or more usable labels per selected district/crop; boundary crosswalks; validated crop masks or an explicitly coarser fallback; lawful/repeatable access for every retained modality; and a source inventory distinguishing verified local data from benchmark-only and synthetic components.

If access fails, narrow or defer. Never substitute invented local records for missing observations.

---

## 4. Canonical data model and alignment

Use stable geography, crop, season, source, and forecast-issue dimensions. Keep source-specific labels and original units beside standardized values.

Core fact tables:

- Yield: district/crop/year/season, area, production, published and derived yield, vintage, estimate stage, unit, and quality flags.
- Weather: grid/station/date, rainfall, min/max temperature, source and quality flags.
- Satellite: pixel or district aggregate, acquisition window, cloud coverage, crop-weighted indices/quantiles, product version.
- Advisory: document ID, issue/valid dates, language, geography, extracted crop/stage/hazard/action, confidence and source span.
- Disease image: image ID, dataset, class, grouped split ID, rights note, and quality metadata.
- Forecast: issue timestamp/cutoff, model version, yield quantiles, risk probability, explanation, evidence coverage, OOD flag.

Alignment rules:

1. Version district boundaries and crosswalks; names alone fail across spelling changes, splits, and mergers.
2. Unit conversions and yield derivation require documentation and tests.
3. Use area-weighted gridded aggregation and crop weights only where a defensible crop mask exists.
4. Preserve satellite cloud/valid observation coverage; missing imagery is not normal crop condition.
5. Align features to documented local calendars and keep calendar uncertainty explicit.
6. Reconstruct every historical forecast using an issue-time snapshot.
7. Preserve original advisory language, translation version, and page/character offsets.
8. Keep leaf-image labels independent from district yield unless a scientifically defensible paired dataset is obtained.

---

## 5. Phased zero-to-end workflow

### Phase 0: scope and contract (Week 1)

Agree on decision, threshold, crops, pilot districts, outcome vintage, issue horizons, and cutoffs with agronomy/statistics reviewers. Create charter, data dictionary, source register, source-to-field map, and readiness gate.

### Phase 1: reproducible setup (Week 1)

Use Python, locked dependencies, Git, deterministic seeds, lint/format rules, and environment-variable names only in example configuration. Keep raw data out of version control when rights are unclear. Layer the pipeline as acquisition → validation → harmonization/features → training → evaluation → batch inference → read-only serving and monitoring. Each run records code/config/data/package/model versions.

### Phase 2: acquisition (Weeks 1–3)

Download DES/OGD outcomes; check climate terms and access; sample Sentinel-2 and MODIS coverage; assemble a bounded multilingual IMD bulletin corpus; acquire disease images only after terms review. Store source files read-only with checksum, retrieval date, URL, and manifest. Build coverage by source, district, crop, and year.

### Phase 3: QA and harmonization (Weeks 2–4)

Check schemas, duplicates, invalid codes, units, negative values, zero-area denominators, missing labels, totals, yield arithmetic, district changes, satellite masks/cloud gaps, weather missingness, bulletin date/language/OCR quality, duplicate images, and class balance. Quarantine suspect rows instead of silently repairing them; document and test every repair. Publish machine-readable checks plus a human caveat report.

### Phase 4: EDA (Weeks 3–5)

Plot area, production, yield/anomaly; rainfall totals, dry spells, excess-rain and heat; seasonal vegetation curves and cloud gaps; spatial coverage/missingness; and label sensitivity to boundaries/vintage. Review bilingual advisory extractions. Use MOSPI at the household survey level only. Treat patterns as predictive association, not cause.

### Phase 5: statistical baselines (Weeks 4–6)

Implement trailing median, robust district/crop trend, hierarchical regression with partial pooling where justified, and a transparent weather-only regularized/GAM model. Backtest with expanding years, not random rows; reserve latest contiguous years.

Report MAE/RMSE in yield units, anomaly MAE, secondary R², district/year block-bootstrap uncertainty, residual trend/heteroskedasticity/spatial dependence, crop bias, and stability under estimate revisions. Compare labels at −10/−15/−20%. Advanced models must beat an appropriate baseline out of time and show usable calibration; otherwise present the baseline as the best model.

### Phase 6: tabular and time-series ML (Weeks 5–8)

Build point-in-time features: lagged yields/area/production and trend residuals; crop-stage rainfall anomaly, dry-spell length, heavy-rain and heat days; satellite level/slope/integral/anomaly/variability and valid-observation count; crop-calendar phase and coverage. Use weather forecasts only when issue-time archives exist.

Compare regularized linear, random forest/ExtraTrees, gradient boosting, and a modest temporal CNN/GRU only if independent seasons justify it. Fit preprocessing/tuning within training folds. Produce quantile forecasts and calibrated risk probabilities.

### Phase 7: DL and computer vision (Weeks 6–9)

**Satellite track:** begin with engineered VI time series. If history/coverage justify it, compare a small temporal CNN or pretrained geospatial representation. Use season/geography splits and point-in-time-safe compositing. Without a validated crop mask, call features mixed-land-cover proxies. Retain DL only if it adds robust value over weather/tabular models.

**Leaf track:** fine-tune a CNN/vision transformer on PlantVillage with grouped leaf/source splits; report macro metrics, per-class recall, confusion matrix and calibration. Add lawful field-like external images if available, report domain gap, and implement abstention. Keep the classifier independent from district yield and explicitly benchmark-only; it is not farmer diagnosis.

### Phase 8: advisory NLP (Weeks 6–9)

Ingest bulletins with source URL, publisher, issue/valid dates, geography, and language. Extract crop, stage, hazard, action, and source span. Compare rules/dictionaries with a multilingual transformer against human-reviewed examples stratified by language/crop. Retrieval should display the exact bulletin date and passage. Have bilingual agronomy reviewers assess ambiguity and translation.

Summarize only what the bulletin says; do not invent recommendations or alter safety-critical chemical instructions. Defer if archive access or rights are unclear.

### Phase 9: multimodal fusion (Weeks 8–10)

Compare yield-history baseline; weather model; weather+satellite; weather+satellite+advisory context. Keep disease CV separate absent paired data. Use regularized/late fusion from out-of-fold predictions. Test missing-modality conditions, geographic/temporal holdouts, and one-modality-at-a-time ablations. Drop modules that do not generalize.

### Phase 10: validation, calibration, robustness, explanation (Weeks 9–11)

Primary evaluation is rolling-origin temporal validation; secondary is held-out districts; leave-one-state-out only when sample counts permit. Keep latest years untouched. Fit preprocessing, tuning and calibration within training workflow.

| Task | Primary metrics | Further checks |
|---|---|---|
| Yield | MAE in units and anomaly % | RMSE, baseline skill, interval coverage |
| Risk | Brier score, PR-AUC | ROC-AUC and precision/recall at review capacity |
| Decision | High-risk recall at fixed review capacity | False-alert burden, lead time |
| Intervals | 50%/90% coverage and width | Crop/state/horizon/coverage slices |
| NLP | Entity precision/recall/F1 | Per-language performance, source-span accuracy |
| Disease CV | Macro-F1, balanced accuracy | Per-class recall, calibration, abstention, external-set gap |

Do not rely on accuracy for imbalanced risk labels. Show sample counts and uncertainty. Slice by crop, state, district, year, horizon, data vintage, satellite coverage, and advisory language. Stress-test missing imagery, stale weather, boundary changes, and extreme unseen seasons. Add OOD/data-support flags. SHAP-like explanations are predictive attribution, not causal effects.

### Phase 11: dashboard and API (Weeks 10–12)

Dashboard: national/state overview; district risk map; detail view with yield history, interval, rainfall/vegetation curves, source evidence; advisory passage view; model performance/calibration; and data quality/coverage page.

Show target, threshold, issue date, data cutoff, model version, interval, coverage, and “screening signal” together. Return “no reliable forecast” when support is insufficient. Visibly distinguish verified observations, benchmarks, and synthetic demos. Export only aggregate predictions and source references.

A read-only API can provide health, metadata, forecast filters, district history, and source-record lookup. Validate identifiers/dates. Do not add operational-action endpoints.

### Phase 12: deployment and monitoring (Weeks 12–14)

Batch flow: versioned source snapshot → data QA gate → point-in-time features → approved model → evaluation/calibration report → analyst review → read-only publication. Suppress release when freshness, calibration, coverage, or OOD gates fail.

Monitor source freshness/schema, yield revisions, weather/satellite coverage, cloud gaps, drift, OOD frequency, interval width, risk prevalence, delayed outcome scores, NLP language quality, and CV domain gap. Reproduce every release from source manifests, cutoff, code/config, and model artifact.

### Phase 13: handoff (Weeks 14–16)

Deliver executive case study, technical report, dashboard demo, model card, data cards, QA/validation report, rights register, reproducibility guide, and runbook. Explain baseline performance, split design, uncertainty, failure cases, and provenance of all real, benchmark-only, and synthetic elements.

---

## 6. Production architecture, repository, and stack

### Architecture

Source registry/raw snapshots → validation and geography normalization → versioned Parquet/DuckDB panel and point-in-time features → model training/evaluation registry → batch forecast artifact → read-only dashboard/API → monitoring and analyst release review.

Start batch-first; avoid streaming before a genuine real-time source and user need exist. Store large/restricted data outside Git; version manifests and repeatable retrieval instructions.

### Repository structure

- Root: README, project metadata, locked environment, example configuration.
- Config: project, crop calendars, geography, source registry.
- Data: raw, interim, processed, manifests, and isolated synthetic-demo area.
- Docs: decision brief, data dictionary, rights, data cards, model card, validation, limitations, runbook.
- Source modules: ingest, QA, geography, features, labels, statistical/tabular models, satellite CV, disease CV, NLP, evaluation, inference, API.
- Notebooks: bounded audit, EDA, error analysis; production logic remains in modules.
- Dashboard, reports, tests, release manifests, and figures.

### Stack

Python 3.11+, uv; pandas/Polars, NumPy, SciPy, statsmodels, scikit-learn; GeoPandas, Shapely, Rasterio, Xarray; PyTorch; spaCy/rules and Hugging Face Transformers; Parquet + DuckDB, optionally PostGIS; FastAPI + Streamlit; MLflow; pytest/Pandera; Docker for reproducible demo. Review optional library licenses and operational overhead.

---

## 7. Acceptance criteria

- Every displayed number maps to a source snapshot, transformation, unit, and issue cutoff.
- Source versions, retrieval dates, rights, and checksums are recorded; malformed inputs fail closed or are quarantined.
- Boundary/crop/season keys and quality rules are audited.
- Target, thresholds, horizons, and splits are fixed before final test.
- Advanced models are compared out of time and by geography with transparent baselines.
- Probabilities/intervals are calibrated or labeled uncalibrated; sample counts and subgroup results are shown.
- Insufficient or OOD cases suppress a forecast.
- Dashboard distinguishes real local observations, public benchmarks, and synthetic fixtures.
- No output claims official/agronomic/insurance authority or triggers an action.
- README, source/data cards, model card, validation, limitations, runbook, dashboard demo, and reproducible workflow are complete.

---

## 8. Risks and mitigation

| Risk | Mitigation |
|---|---|
| Yield labels revised/inconsistent | Preserve vintages, test definition sensitivity, and report uncertainty |
| Short district histories | Robust baselines/partial pooling; narrow scope; avoid data-hungry DL |
| Boundary changes | Version geometries/crosswalks; aggregate only to defensible units |
| Weak crop masks | Validate crop footprint or label mixed-cover proxy |
| Clouds/coarse grids | Show coverage/resolution; compare sources; withhold when inadequate |
| Future leakage | Issue-time snapshots and timestamp invariants |
| PlantVillage domain gap | Group splits, lawful field test set, abstention, independent benchmark module |
| Uneven languages/archives | Report coverage and per-language scores; bilingual review or defer |
| Unclear reuse rights | Review source terms; keep raw local; publish manifests, not restricted files |
| Correlation mistaken for cause | Predictive language, no prescriptions, document confounding |
| False alerts/misses harm decisions | Analyst-only screening, human review, release gate and kill switch |
| Synthetic data look real | Separate storage/schema, labels, IDs, and audit checks |

---

## 9. Timeline and effort

Assume one experienced data scientist, agronomy/statistics review for 2–4 hours/week, and selected engineering support.

| Weeks | Work | Hours |
|---|---|---:|
| 1 | Scope, source inventory, target, environment | 20–28 |
| 1–3 | Acquisition, rights review, catalog | 28–40 |
| 2–4 | QA, geography/calendar harmonization | 35–48 |
| 3–5 | EDA, labels, baselines | 28–38 |
| 5–8 | Tabular/time-series ML | 40–55 |
| 6–9 | Satellite and separate disease CV | 32–48 |
| 6–9 | Advisory NLP and review | 20–32 |
| 8–10 | Fusion, calibration, robustness | 24–34 |
| 10–12 | Dashboard/API | 28–40 |
| 12–14 | Monitoring simulation, packaging, QA/docs | 24–34 |
| 14–16 | Access contingency and handoff | 20–30 |
| **Total** | **Single-person equivalent** | **299–427** |

A focused 260–360-hour version uses one crop/state, engineered satellite summaries instead of custom satellite DL, and independent NLP/CV benchmarks. Two people can parallelize acquisition, NLP annotation, and CV after the data contract; source-access delays can extend calendar time.

Milestones: week 2 data gate; week 5 panel/baseline; week 8 model/modality prototypes; week 10 validation/fusion; week 12 dashboard; week 14–16 docs and handoff.

---

## 10. Test strategy

This is planned verification, not implementation or test execution.

- Unit: unit conversion, yield arithmetic, risk labels, trend, crop-calendar windows, feature aggregation.
- Data contracts: schema, unique keys, value domains, units, vintage, missingness, CRS.
- Geography: crosswalk coverage, splits/mergers, stable IDs, area conservation.
- Leakage: feature timestamp ≤ issue; no label in features; preprocessing within training folds.
- Satellite: quality mask, cloud gate, composite windows, crop weights, no observations.
- NLP: date/geography extraction, source span, language holdout, no unsupported summary.
- CV: duplicate detection, grouped splits, label mapping, external test isolation, abstention, rights manifest.
- Model: baseline reproducibility, metric arithmetic, calibration, validation-only threshold choice, output schema.
- Integration: miniature fixture source-to-forecast run with fixed expected output.
- UI/API: provenance present, missing values not silently zero, suppressed forecasts have no confidence score, downloads stay aggregate.
- Release: pinned environment, reproducible run ID, model/data cards, rights and review sign-off.

Use synthetic fixtures where licenses prevent checked-in examples. Isolate them and never use them to claim real-world model performance.

---

## 11. Final caveat

This is a **portfolio decision-support prototype**, not an agronomic authority, official forecast, insurance assessment, compensation decision, or farm-level recommendation system. A valid outcome may be that open labels, crop masks, timely forecasts, or field-representative images are insufficient for the selected geography. Say so plainly. Mark every benchmark-only and synthetic component. Never invent local observations to make the work appear complete.
