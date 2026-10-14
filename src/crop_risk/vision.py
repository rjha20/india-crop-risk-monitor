"""Optional grouped-split PyTorch leaf-image classifier.

This benchmark-only module is separate from district yield-risk forecasting.
It requires a CSV manifest with path,label,group_id,split columns and disjoint
groups across train/val/test. Paths are relative to an image root and follow a
class-folder layout: LABEL/image-file. It never downloads data or pretrained
weights. PyTorch, NumPy, and Pillow are imported only when a function is called.

CLI examples:
  python -m crop_risk.vision train --manifest data/vision_manifest.csv --image-root data/leaf_images --output-dir models/vision --epochs 5 --max-images 1000
  python -m crop_risk.vision predict --checkpoint models/vision/best_model.pt --image path/to/leaf.jpg

Programmatic training:
  train_and_evaluate(manifest_path, image_root, output_dir, epochs=5, max_images=None)

Outputs include macro-F1, per-class recall/support, and a confusion matrix for
validation and held-out test data. Scores are not calibrated; abstention is
disabled. This is not a farmer diagnosis or agronomic recommendation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


_REQUIRED_COLUMNS = {"path", "label", "group_id", "split"}
_ALLOWED_SPLITS = {"train", "val", "test"}
_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_IMAGE_SIZE = 224
_SEED = 2026
_ARCHITECTURE = "small_cnn_v1"
_MEAN = (0.5, 0.5, 0.5)
_STD = (0.5, 0.5, 0.5)


def _load_dependencies():
    """Load optional runtime dependencies only when CV is requested."""
    try:
        import importlib

        torch = importlib.import_module("torch")
        numpy = importlib.import_module("numpy")
        image_module = importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise ImportError(
            "Computer vision requires the optional dependencies PyTorch, NumPy, and Pillow. "
            "Install them in the project environment before using crop_risk.vision."
        ) from exc
    return torch, numpy, image_module


def _resolve_device(torch, requested: str = "auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use device='cpu' or 'auto'.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(requested)


def _safe_under_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_manifest(manifest_path: Path, image_root: Path) -> tuple[list[dict[str, str]], str]:
    manifest_path = Path(manifest_path)
    image_root = Path(image_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist or is not a file: {manifest_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist or is not a directory: {image_root}")

    image_root = image_root.resolve()
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = {str(column or "").strip() for column in (reader.fieldnames or [])}
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
        seen_paths: set[str] = set()
        group_splits: dict[str, set[str]] = {}
        for row_number, raw in enumerate(reader, start=2):
            raw_path = str(raw.get("path", "") or "").strip()
            label = str(raw.get("label", "") or "").strip()
            group_id = str(raw.get("group_id", "") or "").strip()
            split = str(raw.get("split", "") or "").strip().casefold()
            if split == "validation":
                split = "val"
            if not all((raw_path, label, group_id, split)):
                raise ValueError(f"Manifest row {row_number} has a blank path, label, group_id, or split.")
            if split not in _ALLOWED_SPLITS:
                raise ValueError(
                    f"Manifest row {row_number} has split {split!r}; use only train, val, or test."
                )

            normalized_input = raw_path.replace("\\", "/")
            candidate = Path(normalized_input)
            if not candidate.is_absolute():
                candidate = image_root / candidate
            try:
                resolved = candidate.resolve(strict=True)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Manifest row {row_number} image is missing: {raw_path}") from exc
            if not _safe_under_root(resolved, image_root):
                raise ValueError(f"Manifest row {row_number} points outside image_root: {raw_path}")
            if not resolved.is_file():
                raise ValueError(f"Manifest row {row_number} path is not a file: {raw_path}")
            if resolved.suffix.casefold() not in _ALLOWED_EXTENSIONS:
                raise ValueError(f"Manifest row {row_number} is not a supported image type: {raw_path}")
            relative = resolved.relative_to(image_root)
            if len(relative.parts) < 2 or relative.parts[0].casefold() != label.casefold():
                raise ValueError(
                    f"Manifest row {row_number} label {label!r} must match its class-folder "
                    f"under image_root; observed path {relative.as_posix()!r}."
                )

            path_key = str(resolved).casefold()
            if path_key in seen_paths:
                raise ValueError(f"Manifest contains duplicate image paths (row {row_number}): {raw_path}")
            seen_paths.add(path_key)
            group_splits.setdefault(group_id, set()).add(split)
            rows.append(
                {
                    "path": str(resolved),
                    "relative_path": relative.as_posix(),
                    "label": label,
                    "group_id": group_id,
                    "split": split,
                }
            )

    if not rows:
        raise ValueError("Manifest has no image rows.")
    leaking_groups = sorted(group_id for group_id, splits in group_splits.items() if len(splits) != 1)
    if leaking_groups:
        preview = ", ".join(leaking_groups[:5])
        raise ValueError(
            "Group leakage detected: each group_id must appear in exactly one split. "
            f"Conflicting groups include: {preview}"
        )
    missing_splits = sorted(_ALLOWED_SPLITS - {row["split"] for row in rows})
    if missing_splits:
        raise ValueError(f"Manifest must contain non-empty train, val, and test splits; missing {missing_splits}.")

    train_labels = {row["label"] for row in rows if row["split"] == "train"}
    if len(train_labels) < 2:
        raise ValueError("Training split must contain at least two distinct classes.")
    unseen = sorted({row["label"] for row in rows if row["split"] in {"val", "test"}} - train_labels)
    if unseen:
        raise ValueError(f"Validation/test labels absent from training split: {unseen}")

    return rows, digest


def _limit_rows(rows: list[dict[str, str]], max_images: int | None, seed: int) -> list[dict[str, str]]:
    """Apply a reproducible, approximately class-balanced per-split cap."""
    if max_images is None or len(rows) <= max_images:
        return list(rows)
    labels = sorted({row["label"] for row in rows})
    if max_images < len(labels):
        raise ValueError(
            f"max_images={max_images} is below the {len(labels)} classes in this split; "
            "increase the cap so each observed class can be represented."
        )

    buckets: dict[str, list[dict[str, str]]] = {label: [] for label in labels}
    for row in rows:
        buckets[row["label"]].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    chosen: list[dict[str, str]] = []
    positions = {label: 0 for label in labels}
    while len(chosen) < max_images:
        added = False
        for label in labels:
            position = positions[label]
            bucket = buckets[label]
            if position < len(bucket) and len(chosen) < max_images:
                chosen.append(bucket[position])
                positions[label] += 1
                added = True
        if not added:
            break
    return chosen


def _make_model(torch, class_count: int):
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(p=0.25),
        nn.Linear(128, class_count),
    )


def _dataset_class(torch, numpy, Image):
    class ManifestImageDataset:
        def __init__(self, rows: list[dict[str, str]], labels: Sequence[str]):
            self.rows = rows
            self.label_to_index = {label: index for index, label in enumerate(labels)}

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            try:
                with Image.open(row["path"]) as opened:
                    image = opened.convert("RGB").resize(( _IMAGE_SIZE, _IMAGE_SIZE), Image.Resampling.BILINEAR)
                    array = numpy.asarray(image, dtype=numpy.float32) / 255.0
            except Exception as exc:
                raise RuntimeError(f"Could not decode image {row['relative_path']}: {exc}") from exc
            array = (array - numpy.asarray(_MEAN, dtype=numpy.float32)) / numpy.asarray(_STD, dtype=numpy.float32)
            tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
            return tensor, self.label_to_index[row["label"]]

    return ManifestImageDataset


def _metric_report(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str]) -> dict[str, Any]:
    class_count = len(labels)
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1

    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    for index, label in enumerate(labels):
        support = sum(matrix[index])
        predicted_count = sum(matrix[row][index] for row in range(class_count))
        true_positive = matrix[index][index]
        recall = true_positive / support if support else None
        precision = true_positive / predicted_count if predicted_count else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if recall is not None and precision + recall else 0.0
        f1_values.append(float(f1))
        per_class[label] = {
            "support": support,
            "recall": float(recall) if recall is not None else None,
            "precision": float(precision),
            "f1": float(f1),
        }
    total = len(y_true)
    correct = sum(matrix[index][index] for index in range(class_count))
    return {
        "sample_count": total,
        "accuracy": correct / total if total else None,
        "macro_f1": sum(f1_values) / class_count if class_count else None,
        "per_class": per_class,
        "per_class_recall": {label: per_class[label]["recall"] for label in labels},
        "per_class_support": {label: per_class[label]["support"] for label in labels},
        "confusion_matrix": {
            "labels": list(labels),
            "rows_actual_columns_predicted": matrix,
        },
    }


def _evaluate_loader(torch, model, loader, device, class_names: Sequence[str]) -> tuple[dict[str, Any], float]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device)
            target = target.to(device)
            logits = model(images)
            loss = criterion(logits, target)
            total_loss += float(loss.item()) * int(target.shape[0])
            total += int(target.shape[0])
            targets.extend(int(value) for value in target.detach().cpu().tolist())
            predictions.extend(int(value) for value in logits.argmax(dim=1).detach().cpu().tolist())
    report = _metric_report(targets, predictions, class_names)
    report["loss"] = total_loss / total if total else None
    return report, (total_loss / total if total else float("inf"))


def _make_loader(torch, dataset, batch_size: int, shuffle: bool = False, generator=None):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        generator=generator,
        drop_last=False,
    )


def _training_loop(
    torch,
    numpy,
    Image,
    rows: list[dict[str, str]],
    labels: list[str],
    epochs: int,
    batch_size: int,
    device,
):
    torch.manual_seed(_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_SEED)
    try:
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    except RuntimeError:
        pass

    Dataset = _dataset_class(torch, numpy, Image)
    by_split = {split: [row for row in rows if row["split"] == split] for split in _ALLOWED_SPLITS}
    train_dataset = Dataset(by_split["train"], labels)
    val_dataset = Dataset(by_split["val"], labels)
    test_dataset = Dataset(by_split["test"], labels)
    generator = torch.Generator().manual_seed(_SEED)
    train_loader = _make_loader(torch, train_dataset, batch_size, shuffle=True, generator=generator)
    val_loader = _make_loader(torch, val_dataset, batch_size)
    test_loader = _make_loader(torch, test_dataset, batch_size)

    model = _make_model(torch, len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    history: list[dict[str, Any]] = []
    best_state = None
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_count = 0
        for images, target in train_loader:
            images = images.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += float(loss.item()) * int(target.shape[0])
            epoch_count += int(target.shape[0])

        val_report, val_loss = _evaluate_loader(torch, model, val_loader, device, labels)
        train_loss = epoch_loss_sum / epoch_count if epoch_count else None
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_macro_f1": val_report["macro_f1"],
            }
        )
        score = float(val_report["macro_f1"] or 0.0)
        if score > best_val_f1 or (score == best_val_f1 and val_loss < best_val_loss):
            best_val_f1 = score
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint state.")
    model.load_state_dict(best_state)
    val_report, _ = _evaluate_loader(torch, model, val_loader, device, labels)
    test_report, _ = _evaluate_loader(torch, model, test_loader, device, labels)
    return model, history, val_report, test_report


def _checkpoint_load(torch, checkpoint_path: Path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This PyTorch version lacks safe weights_only checkpoint loading. "
            "Upgrade PyTorch before loading a crop-risk vision checkpoint."
        ) from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != _ARCHITECTURE:
        raise ValueError("Unsupported or malformed crop-risk vision checkpoint.")
    labels = checkpoint.get("class_names")
    state = checkpoint.get("state_dict")
    if not isinstance(labels, list) or len(labels) < 2 or not isinstance(state, dict):
        raise ValueError("Checkpoint is missing class names or model weights.")
    return checkpoint, labels, state


def train_and_evaluate(
    manifest_path: Path,
    image_root: Path,
    output_dir: Path,
    epochs: int = 5,
    max_images: int | None = None,
) -> dict[str, Any]:
    """Train on a validated grouped manifest and report validation and test results.

    max_images caps the number of image rows in each split after deterministic,
    class-balanced sampling. Groups remain in their declared split. It is a
    compute control, not a replacement for sound group assignment.
    """
    if not isinstance(epochs, int) or epochs < 1 or epochs > 100:
        raise ValueError("epochs must be an integer from 1 through 100.")
    if max_images is not None and (not isinstance(max_images, int) or max_images < 1):
        raise ValueError("max_images must be None or a positive integer.")

    torch, numpy, Image = _load_dependencies()
    rows, manifest_sha256 = _read_manifest(Path(manifest_path), Path(image_root))
    labels = sorted({row["label"] for row in rows if row["split"] == "train"})
    selected_rows: list[dict[str, str]] = []
    selected_counts: dict[str, int] = {}
    for index, split in enumerate(("train", "val", "test")):
        split_rows = [row for row in rows if row["split"] == split]
        selected = _limit_rows(split_rows, max_images, _SEED + index)
        if not selected:
            raise ValueError(f"After applying max_images, the {split} split is empty.")
        selected_rows.extend(selected)
        selected_counts[split] = len(selected)

    batch_size = 16
    requested_device = __import__("os").environ.get("CROP_RISK_DEVICE", "auto").casefold()
    device = _resolve_device(torch, requested_device)
    model, history, val_report, test_report = _training_loop(
        torch, numpy, Image, selected_rows, labels, epochs, batch_size, device
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    model = model.to("cpu")
    torch.save(
        {
            "architecture": _ARCHITECTURE,
            "class_names": labels,
            "image_size": _IMAGE_SIZE,
            "mean": list(_MEAN),
            "std": list(_STD),
            "state_dict": model.state_dict(),
            "manifest_sha256": manifest_sha256,
            "group_split_validation": "passed",
            "split_group_counts": {
                split: len({row["group_id"] for row in rows if row["split"] == split})
                for split in ("train", "val", "test")
            },
            "created_with": "crop_risk.vision",
            "calibration": {"status": "not_calibrated"},
            "abstention": {"status": "disabled"},
        },
        checkpoint_path,
    )

    metrics = {
        "task": "leaf_image_classification_benchmark",
        "architecture": _ARCHITECTURE,
        "epochs_requested": epochs,
        "epochs_trained": len(history),
        "batch_size": batch_size,
        "device": str(device),
        "seed": _SEED,
        "manifest_sha256": manifest_sha256,
        "group_split_validation": "passed",
        "group_counts_by_split": {
            split: len({row["group_id"] for row in rows if row["split"] == split})
            for split in ("train", "val", "test")
        },
        "sample_counts_by_split": selected_counts,
        "class_names": labels,
        "validation": val_report,
        "test": test_report,
        "calibration": {
            "status": "not_calibrated",
            "reason": "No independent calibration procedure was applied; softmax scores are uncalibrated.",
        },
        "abstention": {
            "status": "disabled",
            "reason": "No validated selective-prediction threshold is available.",
        },
        "limitations": [
            "This classifier is an image benchmark, not a farmer diagnosis.",
            "Class-folder benchmarks may not represent field images or Indian field conditions.",
            "The declared group_id quality is the user's responsibility; this module verifies split disjointness.",
            "The test split is used once for reporting after validation-based checkpoint selection; do not tune on it.",
        ],
        "training_history": history,
        "checkpoint": str(checkpoint_path),
    }
    metrics_path = output_dir / "vision_metrics.json"
    with metrics_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return {"metrics": metrics, "metrics_json": str(metrics_path), "checkpoint": str(checkpoint_path)}


def predict_image(image_path: Path, checkpoint_path: Path, device: str = "auto") -> dict[str, Any]:
    """Predict a class for one image; scores are explicitly uncalibrated."""
    torch, numpy, Image = _load_dependencies()
    selected_device = _resolve_device(torch, device)
    checkpoint, labels, state = _checkpoint_load(torch, Path(checkpoint_path), selected_device)
    image_path = Path(image_path)
    if not image_path.is_file() or image_path.suffix.casefold() not in _ALLOWED_EXTENSIONS:
        raise FileNotFoundError(f"Missing or unsupported image file: {image_path}")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB").resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.Resampling.BILINEAR)
        array = numpy.asarray(image, dtype=numpy.float32) / 255.0
    array = (array - numpy.asarray(_MEAN, dtype=numpy.float32)) / numpy.asarray(_STD, dtype=numpy.float32)
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(selected_device)

    model = _make_model(torch, len(labels)).to(selected_device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        scores = torch.softmax(model(tensor), dim=1)[0].detach().cpu().tolist()
    predicted_index = max(range(len(scores)), key=scores.__getitem__)
    return {
        "predicted_class": labels[predicted_index],
        "raw_softmax_score_uncalibrated": float(scores[predicted_index]),
        "scores_uncalibrated": {label: float(score) for label, score in zip(labels, scores)},
        "calibration": checkpoint.get("calibration", {"status": "not_calibrated"}),
        "abstention": checkpoint.get("abstention", {"status": "disabled"}),
        "is_diagnosis": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optional group-aware crop-leaf image classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Validate manifest, train, and evaluate on held-out test groups.")
    train.add_argument("--manifest", required=True, type=Path)
    train.add_argument("--image-root", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--epochs", type=int, default=5, help="CPU-safe default is 5; maximum accepted is 100.")
    train.add_argument("--max-images", type=int, default=None, help="Optional maximum per split, class-balanced.")
    predict = subparsers.add_parser("predict", help="Predict one class from a saved checkpoint.")
    predict.add_argument("--checkpoint", required=True, type=Path)
    predict.add_argument("--image", required=True, type=Path)
    predict.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)

    if args.command == "train":
        result = train_and_evaluate(args.manifest, args.image_root, args.output_dir, args.epochs, args.max_images)
    else:
        result = predict_image(args.image, args.checkpoint, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
