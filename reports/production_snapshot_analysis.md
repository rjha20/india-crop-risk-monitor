# APY/UPAG snapshot analysis

This is the observed-data baseline for the CropLens India public portfolio release candidate. It is a
reproducible summary of the bundled APY/UPAG snapshot, not a forecast or agronomic finding.

## Data vintage and quality

- Source: Ministry of Agriculture & Farmers Welfare APY/UPAG, through the India Data Portal.
- Source resource last modified: 12 October 2026.
- Snapshot retrieved: 14 October 2026.
- Source rows: 156,189; coverage: 2013-14 through 2024-25.
- Coverage: 34 states/UTs, 766 district identifiers, 30 crop identifiers, and 3 seasons.
- Missing yield: 1,126 rows (0.72%). Duplicate analysis keys: 0. Negative reported yields: 0.
- Missing district codes: 54; missing crop codes: 9,344. Fallback identifiers use district/state or crop
  labels for analysis; missing source codes remain missing in the displayed source columns.
- Snapshot checksum: `3500ea719355e2cb0dc096221507c48136c724cd954b2951d7ce006152983dc3`.

The source is annual administrative APY reporting. District identifiers and reporting coverage change over
time. The fallback labels do not constitute a reviewed official boundary crosswalk.

## Exploratory anomaly counts

The panel contains 155,063 rows with numeric reported yield after excluding missing-yield rows. 98,907
rows have enough prior observations in the five preceding years for the prior-only baseline. 17,245 of
those rows are at least 15% below their own prior median. These are historical flags; they do not identify
the cause of a yield change, the amount of crop loss, or a present-day hazard.

## Rice classifier walk-forward benchmark

Default benchmark scope: India, Rice, all seasons. Target: reported yield at least 15% below the median of
available observations from the five preceding years. Features: exact-year lags of yield/area and prior-only
summary statistics. Each validation fold trains on years strictly before that holdout year.

| Holdout reporting year | Holdout rows | Event rate | Average precision | ROC AUC | Brier score |
|---|---:|---:|---:|---:|---:|
| 2022-23 | 1,012 | 9.5% | 0.136 | 0.597 | 0.200 |
| 2023-24 | 891 | 15.8% | 0.292 | 0.645 | 0.192 |
| 2024-25 | 877 | 13.9% | 0.476 | 0.792 | 0.171 |
| Pooled | 2,780 | 12.9% | 0.285 | 0.681 | 0.188 |

The pooled average-precision prevalence baseline is 0.129. The constant-prevalence Brier baseline is 0.112,
so this random forest's pooled Brier score is worse. Its ranking metrics vary substantially by year, and its
probabilities are uncalibrated. This benchmark is not evidence of reliable prospective forecasting.

## Release interpretation

The public app should be presented as a data-backed retrospective monitoring and model-evaluation project.
The source vintage is the newest APY/UPAG vintage available in this snapshot, not real-time crop status.
Weather, satellite, crop-calendar, NLP, and computer-vision signals are not paired to these annual records
and are not part of the released dashboard model.
