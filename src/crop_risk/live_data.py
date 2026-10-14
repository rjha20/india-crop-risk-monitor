"""Public APY source discovery, paging, and snapshot helpers.

The dashboard uses the India Data Portal CKAN DataStore API for the UPAG
crop-wise production table. A compressed, attributed snapshot ships with the
project so the public app can still start if the source API is unavailable.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


API_ROOT = "https://ckandev.indiadataportal.com/api/3/action"
PACKAGE_SLUG = "area-production-yield-apy"
UPAG_RESOURCE_NAME = "Crop-wise Area, Production & Yield (UPAG)"
DATA_PORTAL_DATASET_URL = "https://indiadataportal.com/p/area-production-yield-apy"
LICENSE_URL = "https://opendatacommons.org/licenses/by/1-0/index.html"
PAGE_SIZE = 32_000
FIELDS = [
    "year",
    "state_name",
    "state_code",
    "district_name",
    "district_code",
    "season",
    "crop_category",
    "crop_name",
    "crop_code",
    "area",
    "production",
    "yield",
]
REQUIRED_FIELDS = set(FIELDS)
SOURCE_KEY_FIELDS = ["year", "state_key", "district_key", "crop_key", "season"]


class LiveDataError(RuntimeError):
    """Raised when a source response cannot be trusted as a complete snapshot."""


@dataclass(frozen=True)
class SourceMetadata:
    source_name: str
    source_url: str
    api_url: str
    license_name: str
    license_url: str
    resource_id: str
    source_updated_at: str | None
    fetched_at_utc: str
    row_count: int
    year_start: int | None
    year_end: int | None


def _get_json(url: str, *, timeout: int = 45) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "india-crop-risk-monitor/0.2 (public portfolio analytics)",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise LiveDataError(f"Crop source API request failed: {exc}") from exc
    try:
        result = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveDataError("Crop source API returned invalid JSON.") from exc
    if not isinstance(result, dict) or result.get("success") is not True:
        raise LiveDataError("Crop source API returned an unsuccessful response.")
    return result


def discover_upag_resource() -> dict[str, Any]:
    """Resolve the current UPAG resource from its stable dataset slug."""
    query = urlencode({"id": PACKAGE_SLUG})
    result = _get_json(f"{API_ROOT}/package_show?{query}").get("result")
    if not isinstance(result, dict):
        raise LiveDataError("Crop dataset metadata was missing from the source response.")

    license_name = str(result.get("license_title") or result.get("license_id") or "").strip()
    if "open data commons attribution" not in license_name.lower():
        raise LiveDataError(
            "The source license metadata changed; the data refresh was stopped for review."
        )

    resources = result.get("resources")
    if not isinstance(resources, list):
        raise LiveDataError("Crop dataset resource metadata was missing.")
    resource = next(
        (
            item
            for item in resources
            if isinstance(item, dict)
            and str(item.get("name", "")).strip().casefold() == UPAG_RESOURCE_NAME.casefold()
            and item.get("datastore_active") is True
        ),
        None,
    )
    if resource is None or not resource.get("id"):
        raise LiveDataError("The current UPAG DataStore resource could not be resolved.")
    return {
        "resource_id": str(resource["id"]),
        "resource_name": str(resource.get("name", UPAG_RESOURCE_NAME)),
        "resource_updated_at": resource.get("last_modified") or resource.get("created"),
        "license_name": license_name,
    }


def _year_bounds(frame: pd.DataFrame) -> tuple[int | None, int | None]:
    if frame.empty or "year" not in frame:
        return None, None
    starts = frame["year"].astype("string").str.extract(r"((?:19|20)\d{2})", expand=False)
    years = pd.to_numeric(starts, errors="coerce").dropna()
    if years.empty:
        return None, None
    return int(years.min()), int(years.max())


def normalize_source_rows(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Keep source units and labels explicit while standardizing data types."""
    if not records:
        raise LiveDataError("The source returned no crop records.")
    frame = pd.DataFrame.from_records(records)
    missing = sorted(REQUIRED_FIELDS - set(frame.columns))
    if missing:
        raise LiveDataError("Crop data schema changed; missing fields: " + ", ".join(missing))
    frame = frame[FIELDS].copy()
    for column in ("year", "state_name", "state_code", "district_name", "district_code", "season", "crop_category", "crop_name", "crop_code"):
        frame[column] = frame[column].astype("string").str.strip()
    for column in ("area", "production", "yield"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    starts = frame["year"].str.extract(r"((?:19|20)\d{2})", expand=False)
    frame["season_year"] = pd.to_numeric(starts, errors="coerce").astype("Int64")
    frame = frame.dropna(
        subset=[
            "season_year",
            "state_name",
            "district_name",
            "crop_name",
            "season",
        ]
    )
    frame["season_year"] = frame["season_year"].astype(int)
    state_code = frame["state_code"].fillna("").astype(str).str.strip()
    state_name_key = frame["state_name"].astype("string").str.casefold()
    frame["state_key"] = ("CODE:" + state_code).where(state_code.ne(""), "NAME:" + state_name_key)
    district_code = frame["district_code"].fillna("").astype(str).str.strip()
    district_name_key = frame["district_name"].astype("string").str.casefold()
    district_fallback = "NAME:" + frame["state_key"].astype("string") + ":" + district_name_key
    frame["district_key"] = ("CODE:" + district_code).where(district_code.ne(""), district_fallback)
    crop_code = frame["crop_code"].fillna("").astype(str).str.strip()
    crop_name_key = frame["crop_name"].astype("string").str.casefold()
    frame["crop_key"] = ("CODE:" + crop_code).where(crop_code.ne(""), "NAME:" + crop_name_key)
    frame = frame.sort_values(
        ["season_year", "state_key", "district_key", "crop_key", "season"],
        kind="stable",
    ).reset_index(drop=True)
    return frame


def fetch_current_upag() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch all UPAG rows with bounded pagination and a stable source vintage."""
    resource = discover_upag_resource()
    resource_id = resource["resource_id"]
    fields = ",".join(FIELDS)
    first_query = urlencode(
        {"resource_id": resource_id, "limit": PAGE_SIZE, "offset": 0, "fields": fields}
    )
    first = _get_json(f"{API_ROOT}/datastore_search?{first_query}")
    first_result = first.get("result")
    if not isinstance(first_result, dict):
        raise LiveDataError("Crop data response did not contain a result table.")
    total = int(first_result.get("total") or 0)
    if total <= 0:
        raise LiveDataError("The current UPAG resource reported no rows.")

    records: list[dict[str, Any]] = list(first_result.get("records") or [])
    offset = len(records)
    while offset < total:
        query = urlencode(
            {"resource_id": resource_id, "limit": PAGE_SIZE, "offset": offset, "fields": fields}
        )
        page = _get_json(f"{API_ROOT}/datastore_search?{query}")
        page_result = page.get("result")
        if not isinstance(page_result, dict) or int(page_result.get("total") or -1) != total:
            raise LiveDataError("The source row count changed during refresh; no partial snapshot was used.")
        page_records = list(page_result.get("records") or [])
        if not page_records:
            raise LiveDataError("The source returned an incomplete page; no partial snapshot was used.")
        records.extend(page_records)
        offset += len(page_records)
        if offset > total:
            raise LiveDataError("The source returned more rows than its reported snapshot count.")

    frame = normalize_source_rows(records)
    if len(frame) != total:
        raise LiveDataError(
            f"Expected {total:,} source rows but {len(frame):,} passed schema validation."
        )
    duplicate_rows = int(frame.duplicated(SOURCE_KEY_FIELDS, keep=False).sum())
    if duplicate_rows:
        raise LiveDataError(
            f"The source contains {duplicate_rows:,} duplicate district/crop/season/year keys. "
            "The refresh was stopped without aggregating them."
        )
    year_start, year_end = _year_bounds(frame)
    metadata = SourceMetadata(
        source_name="Ministry of Agriculture & Farmers Welfare — APY (UPAG)",
        source_url=DATA_PORTAL_DATASET_URL,
        api_url=f"{API_ROOT}/datastore_search",
        license_name=resource["license_name"],
        license_url=LICENSE_URL,
        resource_id=resource_id,
        source_updated_at=resource.get("resource_updated_at"),
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),
        row_count=len(frame),
        year_start=year_start,
        year_end=year_end,
    )
    return frame, asdict(metadata)


def read_snapshot(snapshot_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a compressed, previously validated source snapshot."""
    path = Path(snapshot_path)
    manifest_path = path.with_name("manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise LiveDataError("The bundled crop-data snapshot or its manifest is missing.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != manifest.get("snapshot_sha256"):
            raise LiveDataError("The bundled crop-data snapshot checksum does not match its manifest.")
        frame = pd.read_csv(
            path,
            compression="gzip",
            dtype={
                "state_code": "string",
                "district_code": "string",
                "crop_code": "string",
                "state_key": "string",
                "district_key": "string",
                "crop_key": "string",
            },
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise LiveDataError(f"The bundled crop-data snapshot could not be read: {exc}") from exc
    if not isinstance(manifest, dict) or len(frame) != int(manifest.get("row_count", -1)):
        raise LiveDataError("The bundled crop-data snapshot does not match its manifest.")
    for column in ("state_name", "district_name", "season", "crop_category", "crop_name", "year"):
        if column in frame:
            frame[column] = frame[column].astype("string")
    for column in ("area", "production", "yield"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["season_year"] = pd.to_numeric(frame["season_year"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["season_year"]).copy()
    frame["season_year"] = frame["season_year"].astype(int)
    return frame, manifest


def write_snapshot(frame: pd.DataFrame, metadata: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Write a deterministic gzip CSV and its source/license manifest."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    snapshot_path = directory / "upag_crop_production.csv.gz"
    with snapshot_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            zipped.write(csv_bytes)
    manifest = dict(metadata)
    manifest["row_count"] = int(len(frame))
    manifest["snapshot_file"] = snapshot_path.name
    manifest["snapshot_sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    manifest["year_start"], manifest["year_end"] = _year_bounds(frame)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
