#!/usr/bin/env python3
"""
dino_pipeline.py — DINOv3 frame curation

Reads the JPGs saved by extract_frames.py into extracted_frames/,
computes DINOv3 embeddings (768-dim) via the frozen backbone shared
with the rest of the ml/ pipeline, K-means clusters similar frames,
picks the most representative frame per cluster, and depending on
--mode:
  --mode db        → upload selected frames to Supabase Storage only
  --mode annotator → create AnalysisImage rows only (assumes files are
                     already uploaded — useful for re-stamping clusters)
  --mode both      → upload AND create DB rows (default)

Uses the same DINOv3 backbone, eval transform, and device config as
`python -m ml.run embed`, so the embeddings here share a feature space
with the rest of the pipeline. Selected frames are stamped with their
cluster_id on the AnalysisImage row, so no separate `assign_clusters`
pass is required for curated frames.

Run extract_frames.py FIRST, then this script.

Usage:
  python scripts/dino_pipeline.py                    (default: both)
  python scripts/dino_pipeline.py --mode db
  python scripts/dino_pipeline.py --mode annotator
  python scripts/dino_pipeline.py --n-clusters 100
  python scripts/dino_pipeline.py --keep-unchosen
"""

import argparse
import mimetypes
import os
import sys
import uuid
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AquaSightWeb.settings")
import django
django.setup()

from django.conf import settings

parser = argparse.ArgumentParser(
    description="DINOv3 embed + cluster + curate frames from extracted_frames/",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--mode",
    choices=["db", "annotator", "both"],
    default="both",
    help="db=upload only, annotator=DB rows only, both=upload and DB rows",
)
parser.add_argument(
    "--n-clusters",
    type=int,
    default=200,
    help="Number of K-means clusters / selected frames (default: 200)",
)
parser.add_argument(
    "--frames-dir",
    type=str,
    default=str(PROJECT_ROOT / "extracted_frames"),
    help="Directory where extract_frames.py saved the JPGs",
)
parser.add_argument(
    "--keep-unchosen",
    action="store_true",
    help="Keep non-selected frames in extracted_frames/ (default: delete)",
)
args = parser.parse_args()

print("Loading dependencies...")
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from annotator.models import AnalysisImage
from ml.data.config import load_config
from ml.data.transforms import get_eval_transform
from ml.models.backbone import DINOv3Backbone

config = load_config()
device = config.training.device

print(f"  PyTorch  : {torch.__version__}")
print(f"  Device   : {device}")
if device == "cuda" and torch.cuda.is_available():
    print(f"  GPU      : {torch.cuda.get_device_name(0)}")
print(f"  Backbone : {config.model.name}")
print(f"  Mode     : {args.mode}")
print(f"  Clusters : {args.n_clusters}")


def resolve_bucket() -> str:
    bucket = settings.SUPABASE_IMAGES_BUCKET
    if not bucket:
        raise RuntimeError(
            "SUPABASE_IMAGES_BUCKET is not set in .env. The annotator UI "
            "reads images from this bucket via AnalysisImage.image_url, so "
            "curated frames must land in the same place."
        )
    return bucket


def supabase_public_url(bucket: str, storage_path: str) -> str:
    return f"{settings.SUPABASE_URL}/storage/v1/object/public/{bucket}/{storage_path}"


def embed_frame(backbone: DINOv3Backbone, transform, img_path: str):
    """Return a (768,) numpy vector, or None if the image can't be opened."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"  WARNING: could not open {img_path}: {e}")
        return None
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = backbone.extract(tensor)
    return emb[0].cpu().numpy()


def upload_frame(sb, bucket: str, img_path: str, storage_path: str) -> None:
    with open(img_path, "rb") as f:
        data = f.read()
    content_type, _ = mimetypes.guess_type(img_path)
    content_type = content_type or "image/jpeg"
    sb.storage.from_(bucket).upload(
        storage_path,
        data,
        file_options={"content-type": content_type},
    )


def main():
    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        print(f"ERROR: frames directory not found: {frames_dir}")
        print("Run extract_frames.py first.")
        sys.exit(1)

    frame_paths = sorted(str(p) for p in frames_dir.glob("*.jpg"))
    print(f"\nFound {len(frame_paths)} frames in {frames_dir}")
    if not frame_paths:
        print("No frames to process.")
        sys.exit(1)

    print(f"\nLoading DINOv3 backbone...")
    backbone = DINOv3Backbone(
        model_name=config.model.name,
        device=device,
        hf_token=config.hf_token,
    )
    transform = get_eval_transform(config.model.input_size)

    print(f"\nStep 1: Embedding {len(frame_paths)} frames...")
    embeddings = []
    valid_paths = []
    for i, path in enumerate(frame_paths):
        emb = embed_frame(backbone, transform, path)
        if emb is not None:
            embeddings.append(emb)
            valid_paths.append(path)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(frame_paths)}")
    print(f"  Done — {len(valid_paths)} valid frames")

    if not valid_paths:
        print("ERROR: no frames embedded successfully.")
        sys.exit(1)

    matrix = np.stack(embeddings)
    matrix_norm = normalize(matrix, norm="l2")

    n_clusters = min(args.n_clusters, len(valid_paths))
    print(f"\nStep 2: K-means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(matrix_norm)
    print(f"  Done")

    print(f"\nStep 3: Picking representative frame per cluster...")
    selected = []
    for cid in range(n_clusters):
        member_idx = np.where(labels == cid)[0]
        if len(member_idx) == 0:
            continue
        centre = kmeans.cluster_centers_[cid]
        distances = np.linalg.norm(matrix_norm[member_idx] - centre, axis=1)
        best = member_idx[int(np.argmin(distances))]
        selected.append((valid_paths[best], cid))
    print(f"  Selected {len(selected)} frames")

    print(f"\nStep 4: Mode '{args.mode}'...")
    sb = None
    bucket = None
    if args.mode in ("db", "both"):
        from supabase import create_client
        bucket = resolve_bucket()
        sb = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    elif args.mode == "annotator":
        bucket = resolve_bucket()

    created = 0
    for i, (frame_path, cid) in enumerate(selected):
        filename = f"frame_{uuid.uuid4().hex}.jpg"
        storage_path = f"dino-images/{filename}"

        if args.mode in ("db", "both"):
            try:
                upload_frame(sb, bucket, frame_path, storage_path)
            except Exception as e:
                print(f"  WARNING: upload failed for {filename}: {e}")
                continue

        if args.mode in ("annotator", "both"):
            image_url = supabase_public_url(bucket, storage_path)
            AnalysisImage.objects.create(
                image_url=image_url,
                original_filename=os.path.basename(frame_path),
                status="bound",
                cluster_id=cid,
            )

        created += 1
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(selected)}")

    if not args.keep_unchosen:
        print(f"\nStep 5: Cleaning up unchosen frames...")
        selected_set = {p for p, _ in selected}
        deleted = 0
        for p in valid_paths:
            if p not in selected_set:
                try:
                    os.remove(p)
                    deleted += 1
                except Exception as e:
                    print(f"  WARNING: could not delete {p}: {e}")
        print(f"  Deleted {deleted}, kept {len(selected)}")

    print(f"\nPipeline complete — {created} frames processed (mode={args.mode})")
    if args.mode in ("db", "both"):
        print(f"  Uploaded to : {bucket}/analysis_images/")
    if args.mode in ("annotator", "both"):
        print(f"  DB rows     : annotator_analysisimage (cluster_id stamped)")


if __name__ == "__main__":
    main()
