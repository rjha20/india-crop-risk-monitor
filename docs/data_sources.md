# Data source register and retrieval notes

This register covers both the live dashboard source and future extensions. Re-check access, publication vintage, crop/season definitions, terms, and redistribution rights each time a snapshot is obtained.

| Component | Candidate source | What it can support | Main limits and required checks |
|---|---|---|---|
| **Current dashboard source** | [APY dataset at India Data Portal](https://indiadataportal.com/p/area-production-yield-apy), resource `Crop-wise Area, Production & Yield (UPAG)` | Observed annual district/crop/season area, production, and yield; public CKAN DataStore API currently includes 2013-14 through 2024-25 | The dashboard resolves the resource by dataset slug, checks ODC-BY license metadata, pages data, and falls back to a checksummed snapshot. API currently uses the `ckandev.indiadataportal.com` host, which may change and has no project SLA. Preserve source/update/retrieval metadata and recheck the license before redistributing a refreshed snapshot. |
| Yield outcomes | [OGD India district-wise, season-wise crop production catalog](https://www.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0); [DES APY query portal](https://www.data.desagri.gov.in/website/apy-index-report-web) | Annual district/crop/season area and production; derive tonnes/hectare with compatible units | Preserve retrieval date, final/provisional stage, year convention, crop codes, district boundary vintage, and source terms. OGD download metadata and DES query parameters can change. |
| Household crop loss context | [MOSPI NSS 77th round report 587](https://www.mospi.gov.in/sites/default/files/publication_reports/Report_587m_0.pdf) | Survey context on reported crop loss/reasons | Survey design and weights apply. This is not a district-season yield target; do not infer district facts from household data without valid design-based estimation. |
| Rainfall | [IMD climate-services overview](https://mausam.imd.gov.in/imd_latest/contents/pdf/pubbrochures/Climate%20Services.pdf); [IMD data service portal](https://dsp.imdpune.gov.in/) | Historical gridded or station weather features where the selected product is licensed and obtainable | Access can require registration or approval. Check the exact product, resolution, units, time coverage, missingness, and redistribution terms. Weather grids are not field measurements. |
| Alternative reanalysis | [Copernicus ERA5-Land catalogue](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) | Retrospective gridded weather covariates | Reanalysis is modeled/reanalyzed context, not local gauge truth. Preserve attribution and product version. |
| Satellite crop-condition proxies | [Copernicus Sentinel data collections](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2); [NASA MODIS vegetation indices](https://modis.gsfc.nasa.gov/data/dataprod/mod13.php) | Cloud/quality-filtered vegetation trajectories at product-specific resolutions | Needs validated crop masks/calendar and quality flags. Mixed pixels, clouds, irrigated/non-crop vegetation, and smallholder field size can make district means misleading. Do not call NDVI yield. |
| Advisory NLP | [IMD Agromet service](https://mausam.imd.gov.in/responsive/agromet_adv_ser_block_current_en.php); [IMD Agromet portal](https://imdagrimet.gov.in/) | Dated bulletin passages for evidence retrieval and bounded extraction | Check archive completeness, language, source span, reuse terms, geographies, valid dates, and OCR. Do not generate or paraphrase safety-critical input advice. |
| Leaf disease CV benchmark | [PlantVillage original source repository](https://github.com/spMohanty/PlantVillage-Dataset); [original paper](https://doi.org/10.3389/fpls.2016.01419) | A controlled-image benchmark for disease/healthy classification | Review the source's current data/license terms, group duplicate/leaf images, and maintain a field-domain test. Public availability is not evidence of field performance or permission to redistribute images. Keep separate from district yield forecasts. |
| District boundaries / geography | State or national administrative boundary release, selected and reviewed for a known vintage | Joins, choropleths, or spatial features when a stable district code and matching boundary vintage are available | No boundary layer is bundled. Check redistribution rights, district splits/mergers, CRS, and area conservation. Do not invent or infer map polygons from names. |

## Access gate

Before model fitting on observed records:

1. Download a small sample under the source's allowed access method.
2. Record the source URL, exact resource/API, date, version/vintage, terms, checksum, and any transformation.
3. Check whether at least ten consistent seasons exist for the selected district/crop combinations; otherwise narrow the scope and report the short history.
4. Build and review a district crosswalk and local crop calendar; do not rely on names or an outdated national calendar.
5. Confirm that each feature existed by its forecast issue day and that final outcome vintages are not backdated as if known earlier.
6. Obtain agronomy/statistics review before interpreting a threshold or releasing real-data forecasts.

The live dashboard snapshot is in `data/public/` and has an attribution file and retrieval manifest. The
separate `data/demo/` fixtures remain synthetic and are suitable only for exercising offline software paths.
