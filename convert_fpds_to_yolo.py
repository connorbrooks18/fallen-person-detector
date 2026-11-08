#!/usr/bin/env python3
"""Convert the FPDS folder layout and annotations to an Ultralytics YOLO dataset.

Expected source layout::

    data/
      train/<split>/*.(png|jpg) and matching *.txt files
      valid/<split>/*.(png|jpg) and matching *.txt files
      test/<split>/*.(png|jpg) and matching *.txt files

FPDS annotation rows are ``class x1 x2 y1 y2`` in pixel coordinates. The
converter writes normalized YOLO ``class x_center y_center width height`` rows,
copies the images into ``images/{train,val,test}``, and writes ``dataset.yaml``.

Usage:
    python3 convert_fpds_to_yolo.py --source data --output yolo_dataset
    yolo detect train data=yolo_dataset/dataset.yaml model=yolo11n.pt
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:  # Pillow is also installed with Ultralytics.
    raise SystemExit("Pillow is required to read image dimensions. Install it with: pip install pillow") from exc


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = (("train", "train"), ("valid", "val"), ("val", "val"), ("test", "test"))
CLASS_IDS = {1: 0, -1: 1}
CLASS_NAMES = {0: "fallen or person_on_ground", 1: "not_fallen or person_upright"}


def parse_annotation(label_path: Path, image_width: int, image_height: int) -> tuple[list[str], Counter[str]]:
    """Return converted rows and counts of annotation rows skipped by reason."""
    rows: list[str] = []
    skipped: Counter[str] = Counter()

    for raw_line in label_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        fields = line.split()
        if len(fields) != 5:
            skipped["expected five fields"] += 1
            continue

        try:
            class_value = float(fields[0])
            coords = [float(value) for value in fields[1:]]
        except ValueError:
            skipped["non-numeric values"] += 1
            continue

        if not math.isfinite(class_value) or any(not math.isfinite(value) for value in coords):
            skipped["non-finite values"] += 1
            continue
        if not class_value.is_integer() or int(class_value) not in CLASS_IDS:
            skipped["unknown class"] += 1
            continue

        # FPDS stores the two x endpoints before the two y endpoints.
        x1, x2, y1, y2 = coords
        # Clip small annotation overshoots to the image. Very large values usually
        # indicate a corrupt row; skip them instead of making a full-image box.
        if max(abs(x1), abs(x2)) > image_width * 2 or max(abs(y1), abs(y2)) > image_height * 2:
            skipped["coordinates far outside image"] += 1
            continue

        left, right = sorted((max(0.0, min(x1, image_width)), max(0.0, min(x2, image_width))))
        top, bottom = sorted((max(0.0, min(y1, image_height)), max(0.0, min(y2, image_height))))
        box_width = right - left
        box_height = bottom - top
        if box_width <= 0 or box_height <= 0:
            skipped["empty box"] += 1
            continue

        center_x = ((left + right) / 2) / image_width
        center_y = ((top + bottom) / 2) / image_height
        width = box_width / image_width
        height = box_height / image_height
        rows.append(
            f"{CLASS_IDS[int(class_value)]} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
        )

    return rows, skipped


def find_split_dirs(source: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for source_name, output_name in SPLITS:
        candidate = source / source_name
        if candidate.is_dir() and output_name not in found:
            found[output_name] = candidate

    missing = {"train", "val", "test"} - found.keys()
    if missing:
        raise FileNotFoundError(
            f"Missing split folder(s) under {source}: {', '.join(sorted(missing))}. "
            "Expected train, valid (or val), and test."
        )
    return found


def write_dataset_yaml(output: Path) -> Path:
    yaml_path = output / "dataset.yaml"
    # JSON-quoted strings are valid YAML and keep paths with spaces portable.
    lines = [
        f"path: {json.dumps(output.resolve().as_posix())}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {class_id}: {json.dumps(name)}" for class_id, name in CLASS_NAMES.items())
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def convert(source: Path, output: Path) -> Path:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the source directory so generated images are not re-imported.")

    split_dirs = find_split_dirs(source)
    totals: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    for split_name, split_dir in split_dirs.items():
        images = sorted(
            path for path in split_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No images found in split directory: {split_dir}")

        for image_path in images:
            relative_path = image_path.relative_to(split_dir)
            output_image = output / "images" / split_name / relative_path
            output_label = output / "labels" / split_name / relative_path.with_suffix(".txt")
            output_image.parent.mkdir(parents=True, exist_ok=True)
            output_label.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(image_path, output_image)
            with Image.open(image_path) as image:
                width, height = image.size

            source_label = image_path.with_suffix(".txt")
            if source_label.is_file():
                rows, bad_rows = parse_annotation(source_label, width, height)
                skipped.update(bad_rows)
            else:
                rows = []
                totals["images without annotation file"] += 1

            output_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            totals[f"{split_name} images"] += 1
            totals[f"{split_name} boxes"] += len(rows)

        # Report annotation files that do not have a corresponding image.
        image_stems = {path.relative_to(split_dir).with_suffix("") for path in images}
        for label_path in split_dir.rglob("*.txt"):
            if label_path.relative_to(split_dir).with_suffix("") not in image_stems:
                totals["orphan annotation files"] += 1

    yaml_path = write_dataset_yaml(output)
    print(f"Converted dataset written to: {output}")
    print(f"YOLO dataset config: {yaml_path}")
    for name in ("train", "val", "test"):
        print(f"  {name}: {totals[f'{name} images']} images, {totals[f'{name} boxes']} boxes")
    for key in ("images without annotation file", "orphan annotation files"):
        if totals[key]:
            print(f"  {key}: {totals[key]}")
    if skipped:
        print("Skipped annotation rows:")
        for reason, count in sorted(skipped.items()):
            print(f"  {reason}: {count}")
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data"), help="FPDS root containing train/valid/test (default: data)")
    parser.add_argument("--output", type=Path, default=Path("yolo_dataset"), help="Output YOLO dataset directory (default: yolo_dataset)")
    args = parser.parse_args()

    try:
        convert(args.source, args.output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
