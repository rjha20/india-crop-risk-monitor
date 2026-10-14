# Public dataset attribution

The compressed snapshot in this directory contains crop-wise Area, Production and Yield (APY) records
published by the Ministry of Agriculture & Farmers Welfare through the Unified Portal for Agricultural
Statistics (UPAg) and catalogued by the India Data Portal.

- Dataset page: <https://indiadataportal.com/p/area-production-yield-apy>
- Source organization: Ministry of Agriculture & Farmers Welfare, Government of India
- Dataset license reported in source metadata: [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/index.html)
- Source resource name: `Crop-wise Area, Production & Yield (UPAG)`
- Retrieved at: see `manifest.json`
- Coverage in the bundled snapshot: 2013-14 through 2024-25
- Transformations: selected source columns, normalized string/numeric types, added a numeric season-year field, and created fallback state/district/crop keys where source codes are missing. Records are sorted; no rows were intentionally aggregated.

The source license and dataset metadata are recorded alongside the snapshot. Check the source page and
current license metadata again before distributing a later refresh.
