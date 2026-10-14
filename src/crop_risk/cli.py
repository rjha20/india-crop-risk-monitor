"""Command-line interface for the India Crop Risk project."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from crop_risk.demo_data import generate_demo_data
from crop_risk.pipeline import run_demo_pipeline, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crop-risk",
        description="Point-in-time district/crop/season yield risk analysis and review tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Generate and run the labeled synthetic demonstration")
    demo.add_argument("--root", type=Path, default=None, help="Project root; defaults to the installed project")

    generate = subparsers.add_parser("generate-demo", help="Write deterministic synthetic input fixtures")
    generate.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    generate.add_argument("--seed", type=int, default=2417)

    run = subparsers.add_parser("run", help="Run the workflow on canonical real/source inputs")
    run.add_argument("--data-dir", type=Path, required=True, help="Directory containing yields.csv and exposures.csv")
    run.add_argument("--output-dir", type=Path, default=Path("data/processed/real-run"))
    run.add_argument("--bulletins", type=Path, default=None, help="Optional local JSONL advisory corpus")

    normalize = subparsers.add_parser("normalize-apy", help="Normalize an inspected DES/OGD APY CSV")
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--output", type=Path, default=Path("data/raw/normalized/yields.csv"))
    normalize.add_argument("--mapping", type=Path, required=True, help="Reviewed JSON source-column mapping")
    normalize.add_argument("--area-unit", required=True, help="Source area unit, e.g. ha or acre")
    normalize.add_argument("--production-unit", required=True, help="Source production unit, e.g. tonnes")
    normalize.add_argument("--crosswalk", type=Path, default=None, help="Optional reviewed district crosswalk")
    normalize.add_argument("--source-id", default="DES_APY_LOCAL_SNAPSHOT")
    normalize.add_argument("--vintage", default=None, help="Source release/vintage as displayed by portal")

    dashboard = subparsers.add_parser("dashboard", help="Launch the local analyst review dashboard")
    dashboard.add_argument("--app", type=Path, default=Path("dashboard/app.py"))
    dashboard.add_argument("--port", type=int, default=8501)

    api = subparsers.add_parser("api", help="Launch the optional read-only forecast API")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)

    vision = subparsers.add_parser("vision-train", help="Train the optional separate leaf-image benchmark")
    vision.add_argument("--manifest", type=Path, required=True)
    vision.add_argument("--image-root", type=Path, required=True)
    vision.add_argument("--output-dir", type=Path, default=Path("models/vision"))
    vision.add_argument("--epochs", type=int, default=5)
    vision.add_argument("--max-images", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        manifest = run_demo_pipeline(args.root)
        print(json.dumps(manifest, indent=2, default=str))
        return 0
    if args.command == "generate-demo":
        result = generate_demo_data(args.output_dir, seed=args.seed)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "run":
        result = run_pipeline(args.data_dir, args.output_dir, bulletins_path=args.bulletins)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "normalize-apy":
        from crop_risk.ingestion import load_mapping, normalize_apy_csv

        result = normalize_apy_csv(
            args.input,
            args.output,
            load_mapping(args.mapping),
            area_unit=args.area_unit,
            production_unit=args.production_unit,
            boundary_crosswalk=args.crosswalk,
            source_id=args.source_id,
            outcome_vintage=args.vintage,
        )
        print(f"Normalized {len(result):,} rows to {args.output}")
        return 0
    if args.command == "dashboard":
        command = [sys.executable, "-m", "streamlit", "run", str(args.app), "--server.port", str(args.port)]
        return subprocess.call(command)
    if args.command == "api":
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "crop_risk.api:create_app",
                "--factory",
                "--host",
                args.host,
                "--port",
                str(args.port),
            ]
        )
    if args.command == "vision-train":
        from crop_risk.vision import train_and_evaluate

        result = train_and_evaluate(
            args.manifest, args.image_root, args.output_dir, epochs=args.epochs, max_images=args.max_images
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
