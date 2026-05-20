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

    def stratified_split(
        self,
        val_frac: float,
        test_frac: float,
    ) -> Tuple[Subset, Subset, Subset]:
        """Split records into train / val / test with per-class proportional ratios.

        Why "stratified"
        ----------------
        The dataset is severely class-imbalanced (e.g. Sand=108 boxes,
        Reef Fish=1). A plain random shuffle could land *zero* examples
        of a rare class in val or test, which makes that class's macro-F1
        either undefined or wildly noisy. Stratifying enforces the same
        train/val/test ratio *inside every class* so coverage is balanced.

        Rare-class policy
        -----------------
        For classes too small to split (n=1, n=2, ...) the val/test counts
        round to 0 and the samples land entirely in train. This is the
        intentional "rare classes go train-only" behaviour: macro-F1 on
        val/test will simply skip those classes rather than report
        misleading numbers from a single-sample evaluation.

        Args
        ----
        val_frac:  Fraction of each class's samples to assign to val.
        test_frac: Fraction of each class's samples to assign to test.

        Returns
        -------
        (train_subset, val_subset, test_subset) — three
        torch.utils.data.Subset views over `self`. No data is copied; each
        subset only stores the indices it owns.
        """
        from collections import defaultdict
        import random

        # Sanity-check the fractions before doing any work. If they're bad,
        # fail loudly here rather than silently produce an empty train set.
        if not (0 <= val_frac < 1) or not (0 <= test_frac < 1):
            raise ValueError(
                f"val_frac and test_frac must be in [0, 1). "
                f"Got val_frac={val_frac}, test_frac={test_frac}."
            )
        if val_frac + test_frac >= 1:
            raise ValueError(
                f"val_frac + test_frac must leave room for train. "
                f"Got val_frac+test_frac={val_frac + test_frac}."
            )

        # ── Step 1: bucket each record's index under its class label ──
        # After this loop:
        #   indices_by_class["Sand"]       == [3, 7, 12, 19, ...]
        #   indices_by_class["Reef Fish"]  == [88]
        # These ints index into self.records.
        indices_by_class: dict = defaultdict(list)
        for record_idx, record in enumerate(self.records):
            indices_by_class[record.label].append(record_idx)

        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []

        # ── Step 2: split each class independently ────────────────────
        # The per-class loop is the heart of stratification — ratios are
        # enforced inside every class, not over the dataset as a whole.
        for label, class_indices in indices_by_class.items():
            # Copy so we don't mutate the dict's underlying list, then
            # shuffle in place so the assignment to train/val/test is
            # randomised within this class.
            class_indices = list(class_indices)
            random.shuffle(class_indices)
            n = len(class_indices)

            # round() gives val/test at least 1 sample once a class has
            # enough total samples (e.g. n=7 with frac=0.15 -> round(1.05)=1)
            # while still rounding down to 0 for very small classes
            # (e.g. n=3 with frac=0.15 -> round(0.45)=0). That's how the
            # rare-class-goes-to-train-only behaviour falls out naturally.
            n_test = int(round(n * test_frac))
            n_val = int(round(n * val_frac))
            n_train = n - n_val - n_test

            # Safety net: under extreme fraction settings (e.g. val=0.45,
            # test=0.45) rounding could leave n_train < 1 for small
            # classes. Claw one sample back from whichever larger split
            # so train never starves on a class it should be learning.
            if n_train < 1 and n >= 1:
                if n_test >= n_val:
                    n_test -= 1
                else:
                    n_val -= 1
                n_train = 1

            # Slice the (now shuffled) class indices into 3 contiguous
            # chunks: first n_train -> train, next n_val -> val,
            # remainder -> test.
            train_indices.extend(class_indices[:n_train])
            val_indices.extend(class_indices[n_train : n_train + n_val])
            test_indices.extend(class_indices[n_train + n_val :])

        # ── Step 3: wrap each index list in a Subset ──────────────────
        # Subset is a lazy view over self — it stores only the indices,
        # not the data itself, so no copying happens here.
        return (
            Subset(self, train_indices),
            Subset(self, val_indices),
            Subset(self, test_indices),
        )


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
