"""Datasets for the AquaSite ML pipeline.

Two modes:
  - AquaSiteDataset:     one sample per AnalysisImage, label = majority of its
                         labeled boxes (lossy — minority labels are discarded).
  - AquaSiteBoxDataset:  one sample per labeled BoundingBox, a crop of the
                         parent image. Every label is preserved.

Which one `build_dataset` returns is controlled by config.training.sample_mode.
Both expose the same interface so the training loop and _TransformOverride
wrapper can treat them interchangeably.
"""

from typing import List, NamedTuple, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

from ml.data.local_loader import ImageRecord


# ─── Image-level dataset (majority label per frame) ───────────────────────

class AquaSiteDataset(Dataset):
    def __init__(self, records: List[ImageRecord], transform=None):
        self.records = records
        self.transform = transform

        labels = sorted(set(r.label for r in records if r.label is not None))
        self.class_names = labels
        self.class_to_idx = {name: i for i, name in enumerate(labels)}
        self.num_classes = len(labels)
        self.has_labels = self.num_classes > 0

    def __len__(self) -> int:
        return len(self.records)

    def load_raw(self, idx: int) -> Image.Image:
        rec = self.records[idx]
        return Image.open(rec.path).convert("RGB")

    def get_label_idx(self, idx: int) -> int:
        rec = self.records[idx]
        if rec.label is not None and rec.label in self.class_to_idx:
            return self.class_to_idx[rec.label]
        return -1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.load_raw(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.get_label_idx(idx)

    def split(self, train_frac: float = 0.8) -> Tuple[Subset, Subset]:
        from collections import defaultdict
        import random

        if self.has_labels:
            groups = defaultdict(list)
            unlabeled = []
            for i, rec in enumerate(self.records):
                if rec.label is not None and rec.label in self.class_to_idx:
                    groups[rec.label].append(i)
                else:
                    unlabeled.append(i)

            train_idx, val_idx = [], []
            for _, idxs in groups.items():
                random.shuffle(idxs)
                split = max(1, int(len(idxs) * train_frac))
                train_idx.extend(idxs[:split])
                val_idx.extend(idxs[split:])

            train_idx.extend(unlabeled)
        else:
            import random
            indices = list(range(len(self)))
            random.shuffle(indices)
            split = int(len(indices) * train_frac)
            train_idx = indices[:split]
            val_idx = indices[split:]

        return Subset(self, train_idx), Subset(self, val_idx)


# ─── Box-level dataset (one sample per labeled BoundingBox) ───────────────

class BoxRecord(NamedTuple):
    box_id: str
    image_path: str       # local cached path to the parent frame
    image_url: str        # original Supabase URL (for reference)
    original_filename: str
    crop_pct: Tuple[float, float, float, float]  # (x, y, w, h) in percent 0–100
    label: str            # TaxonomyLabel.name


class AquaSiteBoxDataset(Dataset):
    def __init__(self, records: List[BoxRecord], transform=None):
        self.records = records
        self.transform = transform

        labels = sorted(set(r.label for r in records))
        self.class_names = labels
        self.class_to_idx = {name: i for i, name in enumerate(labels)}
        self.num_classes = len(labels)
        self.has_labels = self.num_classes > 0

    def __len__(self) -> int:
        return len(self.records)

    def load_raw(self, idx: int) -> Image.Image:
        rec = self.records[idx]
        img = Image.open(rec.image_path).convert("RGB")
        W, H = img.size
        x, y, w, h = rec.crop_pct
        left = max(0, int(round(x / 100.0 * W)))
        top = max(0, int(round(y / 100.0 * H)))
        right = min(W, int(round((x + w) / 100.0 * W)))
        bottom = min(H, int(round((y + h) / 100.0 * H)))
        if right - left < 2 or bottom - top < 2:
            # Degenerate box; fall back to whole image rather than crash
            return img
        return img.crop((left, top, right, bottom))

    def get_label_idx(self, idx: int) -> int:
        rec = self.records[idx]
        return self.class_to_idx[rec.label]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.load_raw(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.get_label_idx(idx)

    def split(self, train_frac: float = 0.8) -> Tuple[Subset, Subset]:
        from collections import defaultdict
        import random

        groups = defaultdict(list)
        for i, rec in enumerate(self.records):
            groups[rec.label].append(i)

        train_idx, val_idx = [], []
        for _, idxs in groups.items():
            random.shuffle(idxs)
            # max(1, ...) — every class sees at least 1 training sample, even
            # if that means val has 0 of that class (val F1 for missing class
            # is undefined rather than always-wrong).
            split = max(1, int(len(idxs) * train_frac))
            train_idx.extend(idxs[:split])
            val_idx.extend(idxs[split:])

        return Subset(self, train_idx), Subset(self, val_idx)


def load_box_records(config) -> List[BoxRecord]:
    """Walk labeled BoundingBox rows, download parent frames, build records."""
    from annotator.models import BoundingBox
    from ml.data.supabase_loader import _cached_download

    cache_dir = config.supabase.download_to
    min_area_pct_sq = config.training.min_box_fraction * 100 * 100  # w% * h% threshold

    qs = (BoundingBox.objects
          .filter(taxonomy_label__isnull=False)
          .select_related("taxonomy_label", "image"))

    records: List[BoxRecord] = []
    downloads: dict = {}
    skipped_tiny = 0
    skipped_missing = 0

    for box in qs:
        area = box.width * box.height  # each is a percent, so area is %²
        if area < min_area_pct_sq:
            skipped_tiny += 1
            continue

        img = box.image
        if img.image_url not in downloads:
            downloads[img.image_url] = _cached_download(img.image_url, cache_dir)
        local_path = downloads[img.image_url]
        if local_path is None:
            skipped_missing += 1
            continue

        records.append(BoxRecord(
            box_id=str(box.id),
            image_path=local_path,
            image_url=img.image_url,
            original_filename=img.original_filename or "",
            crop_pct=(box.x, box.y, box.width, box.height),
            label=box.taxonomy_label.name,
        ))

    print(
        f"Box loader: {len(records)} usable boxes across "
        f"{sum(1 for v in downloads.values() if v)} frames "
        f"(skipped {skipped_tiny} tiny, {skipped_missing} missing-image)"
    )
    return records


# ─── Dispatch ─────────────────────────────────────────────────────────────

def build_dataset(config) -> Dataset:
    """Build the configured dataset. Dispatches on config.training.sample_mode."""
    from ml.data.transforms import get_train_transform
    transform = get_train_transform(config.model.input_size)

    mode = config.training.sample_mode
    if mode == "box":
        records = load_box_records(config)
        return AquaSiteBoxDataset(records, transform=transform)

    # mode == "image"
    if config.data_source == "local":
        from ml.data.local_loader import load_local_images
        records = load_local_images(
            image_dirs=config.local.image_dirs,
            metadata_csv=config.local.metadata_csv,
            labels_csv=config.local.labels_csv,
        )
    elif config.data_source == "supabase":
        from ml.data.supabase_loader import load_supabase_images
        records = load_supabase_images(
            use_django_media=config.supabase.use_django_media,
            download_to=config.supabase.download_to,
            supabase_url=config.supabase_url,
            supabase_key=config.supabase_key,
        )
    else:
        raise ValueError(f"Unknown data_source: {config.data_source}")

    return AquaSiteDataset(records, transform=transform)
