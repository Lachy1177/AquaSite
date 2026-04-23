"""Load images from local FFT data directories."""

import csv
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional


class ImageRecord(NamedTuple):
    path: str           # Absolute path to image file
    filename: str       # Just the filename (e.g. frame_0176...)
    label: Optional[str]  # Class label or None if unlabeled
    metadata: dict      # GPS, depth, temperature, etc.


def _parse_metadata_csv(csv_path: str) -> Dict[str, dict]:
    """Parse metadata.csv into {filename: {col: value}} dict."""
    records = {}
    path = Path(csv_path)
    if not path.exists():
        return records

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename", "")
            if fname:
                # Convert numeric fields
                meta = {}
                for k, v in row.items():
                    if k == "filename":
                        continue
                    try:
                        meta[k] = float(v)
                    except (ValueError, TypeError):
                        meta[k] = v
                records[fname] = meta
    return records


def _parse_labels_csv(csv_path: str) -> Dict[str, str]:
    """Parse optional labels CSV (filename,label columns) into {filename: label}."""
    labels = {}
    path = Path(csv_path)
    if not path.exists():
        return labels

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename", "")
            label = row.get("label", "")
            if fname and label:
                labels[fname] = label
    return labels


def load_local_images(
    image_dirs: List[str],
    metadata_csv: Optional[str] = None,
    labels_csv: Optional[str] = None,
) -> List[ImageRecord]:
    """Scan local directories for images and pair with metadata/labels.

    Args:
        image_dirs: List of directories to scan for .jpg files.
        metadata_csv: Path to metadata.csv (GPS, depth, etc.).
        labels_csv: Path to optional labels CSV with filename,label columns.

    Returns:
        List of ImageRecord tuples.
    """
    # Load metadata and labels
    metadata = {}
    if metadata_csv:
        metadata = _parse_metadata_csv(metadata_csv)

    labels = {}
    if labels_csv:
        labels = _parse_labels_csv(labels_csv)

    # Scan for images, deduplicating by filename (first dir wins)
    seen = set()
    records = []

    for dir_path in image_dirs:
        p = Path(dir_path)
        if not p.exists():
            print(f"Warning: image directory not found: {dir_path}")
            continue

        for img_path in sorted(p.glob("*.jpg")):
            fname = img_path.name
            if fname in seen:
                continue
            seen.add(fname)

            records.append(ImageRecord(
                path=str(img_path),
                filename=fname,
                label=labels.get(fname),
                metadata=metadata.get(fname, {}),
            ))

    print(f"Local loader: found {len(records)} images "
          f"({sum(1 for r in records if r.label is not None)} labeled)")
    return records
