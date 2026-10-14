# Validation and release evidence

## Current observed-data evidence

- The APY/UPAG snapshot covers 156,189 annual district/crop/season records from 2013-14 through 2024-25.
- Snapshot checks report 34 states/UTs, 766 district identifiers, 30 crop identifiers, 1,126 missing-yield
  rows, no duplicate analysis keys, and no negative reported yields.
- The history-only Rice model was evaluated on 2022-23, 2023-24, and 2024-25 holdout years. Pooled average
  precision was 0.285 against a 0.129 prevalence baseline; pooled ROC AUC was 0.681. Its pooled Brier score
  was 0.188, worse than the 0.112 constant-prevalence baseline. Risk scores remain uncalibrated.
- Detailed counts, per-year metrics, source vintage, retrieval date, and snapshot checksum are in
  `reports/production_snapshot_analysis.md` and `data/public/manifest.json`.
- These observations are annual administrative production history. They do not establish current crop
  conditions, explain causes, or validate farm-level predictions.
- NLP bulletin extraction and the plant-leaf CV benchmark remain independent research modules; neither is
  paired with this district-year dataset.

## Release checks still required

1. **Source continuity:** verify the current India Data Portal endpoint and license on every snapshot
   refresh; review any changes in units, crop/district codes, source vintage, or schema.
2. **Public deployment:** publish only under the requested `rjha20` account, deploy the `dashboard/app.py`
   entrypoint, and verify live-source and snapshot-fallback states in the hosted app.
3. **Temporal robustness:** rerun rolling-origin evaluation when the source updates; compare against
   prevalence and constant-risk baselines; review performance by crop/state/year before interpreting scores.
4. **Source quality:** inspect missingness, row counts, district splits/mergers, estimate revisions, and
   coverage changes. Maintain a reviewed crosswalk before making district-level longitudinal comparisons.
5. **Future forecasting scope:** before adding weather/satellite features, prove issue-time availability,
   crop/calendar alignment, and source rights; keep all preprocessing inside temporal training folds.
6. **NLP:** perform bilingual human-labeled evidence-span evaluation, source/date checks, and language/source
   slices before displaying any extracted text.
7. **CV:** use group-disjoint splits, an external field-image evaluation, per-class metrics, and source/license
   review. Do not transfer controlled-image scores to field diagnosis claims.

The app is prepared for public portfolio hosting but has not been published. It is not approved for
operational agricultural, insurance, credit, or government decisions.
