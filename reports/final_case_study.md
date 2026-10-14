# India Crop-Loss and Harvest-Risk Forecasting — case study template

## Executive summary

This project provides a reproducible workflow for district/crop/season yield anomaly screening. It joins official outcome snapshots to issue-time-safe weather and vegetation summaries, compares chronological baselines with ML models, and presents evidence coverage for analyst review. Advisory NLP and leaf-disease vision remain separate evidence tracks.

## Current result status

**The included run is synthetic software demonstration only.** No source-backed model skill, actual district risk ranking, crop loss estimate, early-warning lead time, or field diagnosis is claimed. Replace this section only after the source bundle and validation evidence described in `docs/validation.md` have been reviewed.

| Evidence | Current status |
|---|---|
| Official district/crop yield panel | Not included; user must retrieve a permitted source snapshot |
| Weather/satellite features | Synthetic fixture only; no operational feed or crop mask |
| Advisory text | Synthetic placeholder bulletins only |
| Leaf disease images | Not included; external benchmark required |
| Temporal generalization | Not established on observed outcomes |
| Operational decision | Not authorized; analyst demo only |

## Analysis design

The forecast unit is district × crop × season × year. Expected yield is a trailing median of prior outcomes. Yield anomaly is measured against that prior-only baseline; a provisional −15% threshold defines the risk label, with −10%/−20% sensitivity planned. Every training and evaluation split must respect year order.

Weather/satellite features are cutoff-aware. Advisory extracts retain source evidence. Disease-image evaluation is separate because its images are not paired to district yields. The system suppresses predictions when evidence support is low or out of distribution.

## Interpretation

On a real source panel, the report should compare MAE/RMSE and interval coverage against the trailing-median baseline, report class balance and risk calibration, and show state/crop/time/coverage slices. Associations are predictive, not causal. A high anomaly-risk probability is a request for analyst attention, not proof of crop loss or a trigger for aid or insurance.

## Reproducibility

Use the run manifest to record source checksums, code/config versions, cutoff, and data status. Keep source records and row-level outputs out of public Git unless redistribution and privacy review explicitly allow them. The `SIMULATED_DEMO` label must remain visible whenever synthetic data are displayed.
