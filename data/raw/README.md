# Raw-input contract

Place licensed, versioned source snapshots here locally. Raw files are ignored by Git. Record the source URL, retrieval timestamp, license/terms review, source vintage, checksum, and transformation in a local source manifest before processing.

The pipeline expects two CSV files in a source snapshot directory.

## `yields.csv`

Required columns:

| Column | Type/unit | Rule |
|---|---|---|
| `district_id` | string | Stable district key from a documented boundary vintage; names alone are not a key |
| `state` | string | Source administrative state/UT label |
| `district` | string | Source district label, retained for review |
| `crop` | string | Normalized crop name; original source code should be retained separately |
| `season` | string | Source season code normalized to lowercase |
| `season_year` | integer | Harvest/outcome year; document its convention |
| `area_ha` | numeric, hectares | Positive when present |
| `yield_t_ha` | numeric, tonnes/hectare | Required or derived from `production_t / area_ha` |

Optional provenance columns such as `production_t`, `official_vintage`, `outcome_stage`, `source_id`, and `data_status` are preserved. Do not mix provisional and final estimates without retaining their vintage and release stage.

## `exposures.csv`

One row per district/crop/season/year key. An exposure row may have no matching yield row when it represents the current in-season forecast issue before final yield is known. All numeric exposure windows must be available on or before `issue_day`.

Required key columns are the same six identifiers above. The package accepts these optional numeric columns; unavailable values remain missing and reduce support:

`rainfall_mm`, `rainfall_anomaly_pct`, `dry_spell_days`, `heat_days`, `excess_rain_days`, `ndvi_mean`, `ndvi_trend`, `ndvi_valid_obs`, `coverage_score`, `issue_day`, `feature_end_day`.

If issue fields are absent, the contract defaults to day 90. It rejects any row where `feature_end_day > issue_day`. The current outcome may not appear in exposure features. Document grid resolution, crop-calendar mapping, cloud/quality flags, interpolation, units, and aggregation method in your source manifest.

## Run

```powershell
uv run crop-risk run --data-dir data/raw/my-versioned-snapshot --output-dir data/processed/my-run
```

For an inspected APY CSV, create a mapping from `configs/apy_column_map.example.json`, set the exact columns and units, and normalize with:

```powershell
uv run crop-risk normalize-apy --input data/raw/source-export.csv --mapping configs/my-apy-map.json --area-unit ha --production-unit tonnes --crosswalk data/raw/district-crosswalk.csv --vintage "DES final release YYYY" --output data/raw/my-versioned-snapshot/yields.csv
```

This step produces the yield table only. You must still prepare the matching `exposures.csv` using a documented weather/satellite aggregation and issue-time cutoff. The normalizer will not infer a district key, add observations, silently sum duplicate rows, or choose a unit scale.

The demo files under `data/demo/` are synthetic and use separate provenance labels. Never place them in a real source snapshot or describe their scores as observed performance.
