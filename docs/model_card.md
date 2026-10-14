# Model card — CropLens India historical low-yield screening

## Release status

**Public portfolio release candidate; not an operational crop-forecasting service.** The dashboard currently
uses observed Ministry APY/UPAG records and calculates retrospective yield anomalies. It does not generate a
current-season forecast or provide farm-level advice. The project has not been field-validated.

## Intended use

Help data-science and agricultural-planning analysts explore annual district/crop/season production,
identify reported yields that fall below their own prior history, and inspect a time-ordered model benchmark.
Outputs are descriptive review signals only.

## Excluded uses

- Farmer or field-level decisions, yield guarantees, planting/input/chemical recommendations, or pest diagnosis.
- Crop insurance, credit, compensation, procurement, public-aid, or eligibility decisions.
- Causal claims about weather, climate, policy, or agronomic interventions.
- Any claim that historical APY values establish current local crop conditions.

## Data and unit

- Source: Ministry of Agriculture & Farmers Welfare crop-wise Area, Production and Yield (APY), published
  through UPAg and catalogued at the [India Data Portal](https://indiadataportal.com/p/area-production-yield-apy).
- Unit: district × crop × season × annual reporting year.
- Dashboard yield unit: kilograms per hectare, as supplied by UPAG.
- Snapshot coverage at retrieval: 2013-14 through 2024-25; source retrieval date and checksum are in
  `data/public/manifest.json`.
- Source license metadata reports the [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/index.html).

The source can revise records, codes, district definitions, and vintage. The live API response is checked
against its metadata and the bundled fallback snapshot is attributed and checksummed. The newest outcome
vintage remains annual reporting history; it is not a current-season feed.

## Historical anomaly target

For each district/crop/season time series, compute a median of up to five **previous** reporting years;
require at least three prior observations. The anomaly is `(observed yield - prior median) / prior median`.
The exploratory low-yield flag is an anomaly of −15% or lower. The current yield is used only to compute
the historical label and never enters the risk model's features.

## Machine-learning benchmark

- Model: scikit-learn random-forest classifier using lagged yield, lagged area, prior rolling median,
  prior rolling standard deviation, and a prior-only yield trend.
- Target: the observed row meets the project low-yield threshold above.
- Validation: expanding year-block evaluation over the latest three available reporting years. Each fold
  trains only on rows earlier than the holdout year; random row splits are not used.
- Metrics shown: average precision, event-prevalence baseline, Brier score, and per-year fold counts.
- Probability status: **uncalibrated**. Scores are only historical benchmark outputs; the app does not issue
  a latest-year or future-year forecast from these scores.

The benchmark does not use weather, soil, satellite, irrigation, crop calendars, current prices, or
field-observed crop conditions. Results may be weak or unstable and should be compared with the prevalence
baseline by crop/state/year. They have not been independently reviewed or field-validated.

## Known limitations and monitoring

Annual records can be missing or revised; district boundaries, crop codes, and reporting procedures can
change; administrative yield is not a field-level observation. The 15% cutoff is exploratory, not an
agronomic standard. Monitor source age, schema/license, row count, missingness, duplicate keys, model
performance slices, and district/crop coverage. A source contract or license change stops refresh pending
review.

NLP bulletin extraction and leaf-image computer vision remain separate research modules. Their data are
not paired with these annual district outcomes, so they are not fused into the dashboard or the model.
