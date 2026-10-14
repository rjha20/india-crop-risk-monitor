# Data handling policy

## Default handling

- `data/raw/` is ignored by Git. Keep downloaded source snapshots read-only and record source URL, retrieval date, release/vintage, license/terms review, checksum, units, and transformation.
- `data/public/` is the explicit exception for the APY/UPAG fallback snapshot. It is included with the source metadata's Open Data Commons Attribution license, attribution, retrieval manifest, and checksum. Verify the current source license and terms before replacing it with a refreshed vintage.
- `data/processed/` is ignored because row-level district/crop outcomes, joined exposures, and individual forecast rows may carry source restrictions or become misleading out of context.
- `models/` and common checkpoint/image formats are ignored. Do not upload weights or images unless their terms and provenance have been reviewed.
- `.env*`, local secrets, and Streamlit secrets are ignored. Never put API tokens in config, notebooks, logs, docs, or Git history.
- `data/demo/` is the separate synthetic fixture. Its records must remain marked `synthetic_demo` in every derived artifact and must never be described as source observations.

## Source-specific rules

1. Do not copy other licensed government or satellite files into the public repository by default. `data/public/` is limited to the specifically attributed APY/UPAG source snapshot; share retrieval instructions, checksum, and citation with it.
2. Do not retain agricultural bulletin text or translations unless source terms permit it. The NLP output must keep provenance and exact evidence spans; reports should aggregate themes and avoid fabricated advice.
3. Treat household survey microdata as restricted unless the release explicitly allows the planned access and use. Apply survey weights for supported estimates.
4. Plant disease images remain local. Review license, attribution, and duplicate/leaf grouping before training; report controlled-to-field domain limits.
5. Do not fuse independent leaf-image predictions and district yield outcomes into one risk score without a paired, spatially and temporally valid dataset.

## Public release checklist

- Verify `git status` and the list of staged files; raw records, row-level extracts, private credentials, images, and weights must be absent.
- Inspect source manifest, reports, dashboard exports, and logs for names, user-entered text, credentials, and suppressed cells.
- Verify every public metric points to its source mode, source vintage/retrieval date, threshold, and validation period.
- Keep live source data and the synthetic offline demo visibly separate; never relabel synthetic data as collected or observed.
