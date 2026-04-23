"""Load images from Supabase / Django ORM."""

import os
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
from urllib.request import urlopen

from ml.data.local_loader import ImageRecord


def load_supabase_images(
    use_django_media: bool = True,
    download_to: str = "./ml/outputs/supabase_cache",
    supabase_url: str = "",
    supabase_key: str = "",
) -> List[ImageRecord]:
    """Load images and labels from the Django database / Supabase.

    Mode A (use_django_media=True): Walk AnalysisImage rows via the
    Django ORM and fetch each row's image_url into a local cache. Labels
    come from BoundingBox.taxonomy_label.

    Mode B (use_django_media=False): List the Supabase storage bucket
    directly (no labels).
    """
    if use_django_media:
        return _load_from_django_orm(download_to)
    else:
        return _load_from_supabase_storage(download_to, supabase_url, supabase_key)


def _cached_download(url: str, cache_dir: str) -> Optional[str]:
    """Download url into cache_dir if not already present. Returns local path."""
    if not url:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    basename = os.path.basename(urlparse(url).path) or "image"
    local_path = os.path.join(cache_dir, basename)
    if os.path.exists(local_path):
        return local_path
    try:
        with urlopen(url, timeout=30) as resp:
            data = resp.read()
        with open(local_path, "wb") as f:
            f.write(data)
        return local_path
    except Exception as e:
        print(f"Warning: failed to download {url}: {e}")
        return None


def _load_from_django_orm(download_to: str) -> List[ImageRecord]:
    """Walk AnalysisImage rows, fetch image_url into a local cache, pull labels from the ORM."""
    from annotator.models import AnalysisImage

    records = []
    for img in AnalysisImage.objects.all():
        local_path = _cached_download(img.image_url, download_to)
        if local_path is None:
            continue

        label = _get_majority_label(img)

        records.append(ImageRecord(
            path=local_path,
            filename=img.original_filename or os.path.basename(local_path),
            label=label,
            metadata={
                "status": img.status,
                "uploaded_at": str(img.uploaded_at),
                "image_id": str(img.id),
            },
        ))

    print(f"Supabase/Django loader: found {len(records)} images "
          f"({sum(1 for r in records if r.label is not None)} labeled)")
    return records


def _get_majority_label(image) -> Optional[str]:
    """Get the most common taxonomy label from an image's bounding boxes."""
    boxes = image.boxes.filter(
        taxonomy_label__isnull=False,
    ).select_related("taxonomy_label")

    if not boxes.exists():
        return None

    # Count labels
    label_counts = {}
    for box in boxes:
        name = box.taxonomy_label.name
        label_counts[name] = label_counts.get(name, 0) + 1

    # Return the most common
    return max(label_counts, key=label_counts.get)


def _load_from_supabase_storage(
    download_to: str,
    supabase_url: str,
    supabase_key: str,
) -> List[ImageRecord]:
    """Download images from Supabase storage bucket."""
    from supabase import create_client

    os.makedirs(download_to, exist_ok=True)
    client = create_client(supabase_url, supabase_key)

    # List files in the analysis_images bucket
    bucket = "analysis-images"
    try:
        files = client.storage.from_(bucket).list()
    except Exception as e:
        print(f"Warning: could not list Supabase storage bucket '{bucket}': {e}")
        print("Falling back to Django ORM mode.")
        return _load_from_django_orm()

    records = []
    for file_info in files:
        fname = file_info.get("name", "")
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        local_path = os.path.join(download_to, fname)

        # Download if not cached
        if not os.path.exists(local_path):
            try:
                data = client.storage.from_(bucket).download(fname)
                with open(local_path, "wb") as f:
                    f.write(data)
            except Exception as e:
                print(f"Warning: failed to download {fname}: {e}")
                continue

        records.append(ImageRecord(
            path=local_path,
            filename=fname,
            label=None,  # Labels require ORM access
            metadata={},
        ))

    print(f"Supabase storage loader: downloaded {len(records)} images")
    return records
