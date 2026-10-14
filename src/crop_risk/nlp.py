"""Provenance-preserving rule extraction from local agricultural bulletins.

This module extracts dictionary matches only. It does not summarize bulletins,
translate them, or generate agronomic recommendations. Every mention quotes an
exact source substring with zero-based, half-open character offsets.

Input: local JSONL/NDJSON or CSV. Each record needs a text/content/body field;
date, URL, language, and source metadata are retained when present.
CLI: python -m crop_risk.nlp INPUT_PATH OUTPUT_DIR
Integration: process_bulletins(input_path: Path, output_dir: Path) -> dict.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


VOCABULARY_VERSION = "en-hi-v1"

# Canonical label -> literal forms found in source text. Keep this deliberately
# transparent and conservative; add vocabulary only after language review.
_VOCABULARY: dict[str, dict[str, tuple[str, ...]]] = {
    "crop": {
        "rice": ("rice", "paddy", "धान", "चावल"),
        "wheat": ("wheat", "गेहूं", "गेंहूँ"),
        "maize": ("maize", "corn", "मक्का"),
        "cotton": ("cotton", "कपास"),
        "sugarcane": ("sugarcane", "sugar cane", "गन्ना"),
        "chickpea": ("chickpea", "chick pea", "gram", "चना"),
        "pigeon_pea": ("pigeon pea", "red gram", "अरहर", "तूर"),
        "lentil": ("lentil", "मसूर"),
        "mustard": ("mustard", "सरसों"),
        "soybean": ("soybean", "soya bean", "सोयाबीन"),
        "groundnut": ("groundnut", "peanut", "मूंगफली"),
        "potato": ("potato", "आलू"),
        "tomato": ("tomato", "टमाटर"),
        "onion": ("onion", "प्याज", "प्याज़"),
        "jute": ("jute", "जूट"),
        "pearl_millet": ("pearl millet", "bajra", "बाजरा"),
        "sorghum": ("sorghum", "jowar", "ज्वार"),
        "finger_millet": ("finger millet", "ragi", "रागी"),
    },
    "hazard": {
        "drought": ("drought", "dry spell", "rainfall deficit", "सूखा", "कम वर्षा"),
        "heat_stress": ("heat wave", "heatwave", "high temperature", "heat stress", "लू", "ताप लहर", "उच्च तापमान"),
        "flood": ("flood", "flooding", "बाढ़"),
        "waterlogging": ("waterlogging", "water logging", "जलभराव"),
        "heavy_rain": ("heavy rain", "excess rainfall", "excess rain", "अत्यधिक वर्षा", "भारी वर्षा"),
        "cyclone": ("cyclone", "चक्रवात"),
        "pest": ("pest", "pests", "कीट"),
        "aphid": ("aphid", "aphids", "माहू", "चेपा"),
        "fall_armyworm": ("fall armyworm", "army worm", "फॉल आर्मीवर्म"),
        "stem_borer": ("stem borer", "stem-borer", "तना छेदक"),
        "locust": ("locust", "locusts", "टिड्डी"),
        "disease": ("disease", "fungal disease", "रोग"),
        "blast": ("blast", "rice blast", "ब्लास्ट"),
        "rust": ("rust", "rust disease", "रतुआ"),
        "blight": ("blight", "leaf blight", "झुलसा"),
        "wilt": ("wilt", "उकठा"),
        "powdery_mildew": ("powdery mildew", "mildew", "चूर्णिल आसिता"),
    },
    "stage": {
        "nursery": ("nursery", "नर्सरी"),
        "sowing": ("sowing", "sown", "at sowing", "बुवाई", "बोआई"),
        "emergence": ("emergence", "seedling emergence", "अंकुरण"),
        "transplanting": ("transplanting", "transplant", "रोपाई"),
        "vegetative": ("vegetative stage", "vegetative growth", "वानस्पतिक अवस्था"),
        "tillering": ("tillering", "tiller stage", "कल्ले निकलना", "कल्ले बनने"),
        "flowering": ("flowering", "flower initiation", "फूल आने", "फूलना"),
        "reproductive": ("reproductive stage", "reproductive phase", "प्रजनन अवस्था"),
        "grain_filling": ("grain filling", "grain-fill", "दाना भरना"),
        "fruiting": ("fruiting", "fruit development", "फल लगना"),
        "maturity": ("maturity", "mature crop", "पकने की अवस्था", "परिपक्वता"),
        "harvest": ("harvest", "harvesting", "कटाई", "फसल कटाई"),
    },
    "action": {
        "irrigate": ("irrigate", "irrigation", "irrigate the crop", "सिंचाई", "सिंचाई करें"),
        "drain_water": ("drain water", "drain excess water", "remove standing water", "पानी निकालें", "जल निकासी"),
        "sow": ("sow", "sowing", "बुवाई करें", "बोआई करें"),
        "avoid_sowing": ("avoid sowing", "do not sow", "delay sowing", "बुवाई न करें", "बोआई न करें"),
        "spray": ("spray", "spraying", "foliar spray", "छिड़काव", "स्प्रे"),
        "apply": ("apply", "application", "apply fertilizer", "उर्वरक डालें"),
        "monitor": ("monitor", "scout", "inspect", "निगरानी", "निगरानी करें"),
        "remove_affected_plants": ("remove affected plants", "remove infected plants", "उखाड़ें", "प्रभावित पौधों को हटाएं"),
        "harvest": ("harvest", "harvesting", "कटाई करें"),
        "seed_treatment": ("seed treatment", "treat the seed", "बीज उपचार"),
        "use": ("use", "उपयोग करें"),
    },
}

_TEXT_KEYS = ("text", "content", "body", "bulletin_text", "advisory", "message")
_ID_KEYS = ("document_id", "bulletin_id", "id", "record_id")
_DATE_KEYS = ("issue_date", "issued_at", "published_at", "date", "bulletin_date")
_URL_KEYS = ("source_url", "url", "source_link")
_LANGUAGE_KEYS = ("language", "lang", "language_code")
_SOURCE_NAME_KEYS = ("source_name", "publisher", "source_title")
_SYNTHETIC_KEYS = ("is_synthetic", "synthetic", "demo", "data_status", "dataset_type")


def _first_value(record: Mapping[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return default


def _truthy_marker(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "synthetic", "demo"}


def _is_synthetic(record: Mapping[str, Any]) -> bool:
    for key in _SYNTHETIC_KEYS:
        if key not in record:
            continue
        value = record[key]
        if key in {"data_status", "dataset_type"}:
            if str(value).strip().casefold() in {"synthetic", "demo", "synthetic_demo"}:
                return True
        elif _truthy_marker(value):
            return True
    source = " ".join(str(record.get(k, "")) for k in ("source_name", "publisher", "source_url", "dataset"))
    return "synthetic" in source.casefold() or "demo bulletin" in source.casefold()


def _normalize_date(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        try:
            return date.fromisoformat(text[:10]).isoformat()
        except ValueError:
            return None


def _language(record: Mapping[str, Any], text: str) -> tuple[str, str, str]:
    raw = _first_value(record, _LANGUAGE_KEYS, "")
    if raw:
        normalized = str(raw).strip().replace("_", "-").casefold()
        primary = normalized.split("-", 1)[0]
        return primary or "und", str(raw), "metadata"
    if re.search(r"[\u0900-\u097f]", text):
        return "hi", "", "devanagari-script-heuristic"
    return "und", "", "unknown"


def _load_records(input_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    suffix = input_path.suffix.casefold()
    if suffix in {".jsonl", ".ndjson"}:
        with input_path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {input_path}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected a JSON object on line {line_number} of {input_path}")
                yield line_number, record
        return
    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row_number, record in enumerate(csv.DictReader(stream), start=2):
                yield row_number, dict(record)
        return
    raise ValueError(f"Unsupported input extension {suffix!r}; use .jsonl, .ndjson, or .csv")


def _build_patterns() -> list[tuple[str, str, re.Pattern[str]]]:
    patterns: list[tuple[str, str, re.Pattern[str]]] = []
    for entity_type, labels in _VOCABULARY.items():
        for canonical, aliases in labels.items():
            for alias in sorted(set(aliases), key=lambda item: (-len(item), item.casefold())):
                body = re.escape(alias).replace(r"\ ", r"\s+")
                patterns.append(
                    (entity_type, canonical, re.compile(r"(?<!\w)" + body + r"(?!\w)", re.IGNORECASE | re.UNICODE))
                )
    return patterns


_PATTERNS = _build_patterns()


def _extract_record(record: Mapping[str, Any], row_number: int) -> dict[str, Any]:
    raw_text = _first_value(record, _TEXT_KEYS, None)
    if raw_text is None:
        raise ValueError(f"Record at input row/line {row_number} has no text/content/body field")
    text = str(raw_text)
    document_id = str(_first_value(record, _ID_KEYS, f"row-{row_number}")).strip()
    issue_date = str(_first_value(record, _DATE_KEYS, "")).strip()
    source_url = str(_first_value(record, _URL_KEYS, "")).strip()
    source_name = str(_first_value(record, _SOURCE_NAME_KEYS, "")).strip()
    language, language_raw, language_detection = _language(record, text)
    synthetic = _is_synthetic(record)

    mentions: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for entity_type, canonical, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            key = (entity_type, canonical, match.start(), match.end())
            if key in seen:
                continue
            seen.add(key)
            mentions.append(
                {
                    "document_id": document_id,
                    "entity_type": entity_type,
                    "canonical": canonical,
                    "matched_phrase": text[match.start():match.end()],
                    "span_start": match.start(),
                    "span_end": match.end(),
                    "source_url": source_url,
                    "source_name": source_name,
                    "issue_date": issue_date,
                    "issue_date_normalized": _normalize_date(issue_date),
                    "language": language,
                    "language_raw": language_raw,
                    "language_detection": language_detection,
                    "data_status": "synthetic" if synthetic else "source_text",
                    "is_synthetic": synthetic,
                    "extraction_method": "dictionary_exact_span",
                    "generated_advice": False,
                }
            )
    mentions.sort(key=lambda item: (item["span_start"], item["span_end"], item["entity_type"], item["canonical"]))
    return {
        "document_id": document_id,
        "issue_date": issue_date,
        "issue_date_normalized": _normalize_date(issue_date),
        "source_url": source_url,
        "source_name": source_name,
        "language": language,
        "language_raw": language_raw,
        "language_detection": language_detection,
        "data_status": "synthetic" if synthetic else "source_text",
        "is_synthetic": synthetic,
        "mention_count": len(mentions),
        "mentions": mentions,
    }


def _metrics(documents: Sequence[Mapping[str, Any]], input_format: str | None = None) -> dict[str, Any]:
    mentions = [mention for document in documents for mention in document["mentions"]]
    metrics: dict[str, Any] = {
        "vocabulary_version": VOCABULARY_VERSION,
        "extraction_method": "dictionary_exact_span",
        "generated_advice": False,
        "document_count": len(documents),
        "document_count_with_mentions": sum(document["mention_count"] > 0 for document in documents),
        "document_count_without_mentions": sum(document["mention_count"] == 0 for document in documents),
        "documents_by_language": dict(sorted(Counter(document["language"] for document in documents).items())),
        "documents_by_data_status": dict(sorted(Counter(document["data_status"] for document in documents).items())),
        "mention_count": len(mentions),
        "synthetic_mention_count": sum(bool(mention["is_synthetic"]) for mention in mentions),
        "mentions_by_entity_type": dict(sorted(Counter(item["entity_type"] for item in mentions).items())),
        "mentions_by_language": dict(sorted(Counter(item["language"] for item in mentions).items())),
        "mentions_by_data_status": dict(sorted(Counter(item["data_status"] for item in mentions).items())),
        "synthetic_document_count": sum(bool(document["is_synthetic"]) for document in documents),
        "missing_source_url_document_count": sum(not document["source_url"] for document in documents),
        "missing_issue_date_document_count": sum(not document["issue_date"] for document in documents),
        "unparsed_issue_date_document_count": sum(
            bool(document["issue_date"]) and not document["issue_date_normalized"] for document in documents
        ),
        "unique_source_url_count": len({document["source_url"] for document in documents if document["source_url"]}),
    }
    if input_format is not None:
        metrics["input_format"] = input_format
    return metrics


def extract_bulletins(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Extract rule-based mentions from in-memory records without writing files."""
    documents = [_extract_record(record, row_number) for row_number, record in enumerate(records, start=1)]
    mentions = [mention for document in documents for mention in document["mentions"]]
    return {"documents": documents, "mentions": mentions, "metrics": _metrics(documents)}


MENTION_FIELDS = (
    "document_id", "entity_type", "canonical", "matched_phrase", "span_start", "span_end",
    "source_url", "source_name", "issue_date", "issue_date_normalized", "language",
    "language_raw", "language_detection", "data_status", "is_synthetic",
    "extraction_method", "generated_advice",
)


def process_bulletins(input_path: Path, output_dir: Path) -> dict[str, Any]:
    """Read local JSONL/CSV bulletins and write mention CSV plus JSON metrics.

    Synthetic/demo rows must carry an explicit true marker such as
    is_synthetic=true or data_status=synthetic; those rows remain visibly
    labeled in every emitted mention and in aggregate counts.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Bulletin input does not exist or is not a file: {input_path}")
    documents = [_extract_record(record, row_number) for row_number, record in _load_records(input_path)]
    metrics = _metrics(documents, input_path.suffix.casefold().lstrip("."))

    output_dir.mkdir(parents=True, exist_ok=True)
    mention_path = output_dir / "advisory_mentions.csv"
    metrics_path = output_dir / "nlp_metrics.json"
    mentions = [mention for document in documents for mention in document["mentions"]]
    with mention_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MENTION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(mentions)
    with metrics_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return {"metrics": metrics, "mentions_csv": str(mention_path), "metrics_json": str(metrics_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract provenance-preserving crop-risk entities from local bulletins.")
    parser.add_argument("input_path", type=Path, help="Local .jsonl/.ndjson or .csv bulletin file")
    parser.add_argument("output_dir", type=Path, help="Directory for advisory_mentions.csv and nlp_metrics.json")
    args = parser.parse_args(argv)
    result = process_bulletins(args.input_path, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
