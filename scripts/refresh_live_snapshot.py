"""Refresh the attributed UPAG data snapshot used by the public dashboard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crop_risk.live_data import fetch_current_upag, write_snapshot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "public",
        help="Directory for the compressed source snapshot and its manifest.",
    )
    args = parser.parse_args()

    frame, metadata = fetch_current_upag()
    manifest = write_snapshot(frame, metadata, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
