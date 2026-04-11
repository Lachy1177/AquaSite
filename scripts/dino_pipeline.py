#!/usr/bin/env python3
"""
dino_pipeline.py — Script 2 of 2

What this script does:
  Reads the JPG frames saved by extract_frames.py, runs each one
  through DINOv3 on your GPU to get an embedding (fingerprint),
  clusters similar frames together using K-Means, picks the most
  representative frame from each cluster, then depending on the
  mode selected:
    --mode db        → upload to Supabase Storage only
    --mode annotator → save to Django database only (appears in workspace)
    --mode both      → upload to Supabase AND save to database (default)

  Run extract_frames.py FIRST, then run this script second.
S
Usage:
  python scripts/dino_pipeline.py                         (default: both)
  python scripts/dino_pipeline.py --mode db
  python scripts/dino_pipeline.py --mode annotatorS
"""

import os
import sys
import uuid
import glob
import argparse

# ── Locate project root and load .env ─────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def load_env(path):
    if not os.path.exists(path):
        print(f"ERROR: .env file not found at {path}")
        sys.exit(1)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

load_env(os.path.join(PROJECT_ROOT, ".env"))

# ── Configuration ──────────────────────────────────────────────────────────────

SUPABASE_URL   = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY   = os.environ.get("SUPABASE_KEY", "")
STORAGE_BUCKET = os.environ.get("SUPABASE_STORAGE_BUCKET", "dive-videos")

# Folder where extract_frames.py saved the JPGs
FRAMES_DIR = os.path.join(PROJECT_ROOT, "extracted_frames")

# Default number of diverse frames to select
DEFAULT_CLUSTERS = 200

# ── Parse arguments ────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="DINO classify and upload frames"
)
parser.add_argument(
    "--mode",
    choices=["db", "annotator", "both"],
    default="both",
    help=(
        "db        = upload to Supabase Storage only\n"
        "annotator = save to Django database only (appears in annotation workspace)\n"
        "both      = upload to Supabase AND save to database (default)"
    )
)


# ── Load dependencies ──────────────────────────────────────────────────────────

print("Loading dependencies...")
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.cluster import KMeans

print(f"  PyTorch  : {torch.__version__}")
print(f"  CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU      : {torch.cuda.get_device_name(0)}")
print(f"  Mode     : {args.mode}")
print(f"  Clusters : {args.clusters}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load DINOv3 ────────────────────────────────────────────────────────────────

print("\nLoading DINOv3 model...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
model.eval()
model.to(DEVICE)
print(f"  Model ready on {DEVICE}")

# Preprocessing pipeline for DINO
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── Helper functions ───────────────────────────────────────────────────────────

def get_embedding(img_path):
    """
    Load a JPG from disk, run it through DINO, return its
    embedding - a list of 384 numbers describing the image.
    Returns None if the image cannot be read.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor  = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(tensor)
    return embedding.cpu().numpy().flatten()


def upload_frame(sb, img_path, filename):
    """
    Read a JPG from disk and upload it to Supabase Storage
    under the analysis_images/ folder.
    Returns the storage path.
    """
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    storage_path = f"analysis_images/{filename}"
    sb.storage.from_(STORAGE_BUCKET).upload(
        storage_path,
        img_bytes,
        file_options={"content-type": "image/jpeg"},
    )
    return storage_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)

    if not os.path.exists(FRAMES_DIR):
        print(f"ERROR: Frames folder not found at {FRAMES_DIR}")
        print("Run extract_frames.py first!")
        sys.exit(1)

    # ── Step 1: Find all extracted frames ─────────────────────────────────────
    print(f"\n── Step 1: Finding extracted frames in {FRAMES_DIR}...")
    frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    print(f"  Found {len(frame_paths)} frames")

    if len(frame_paths) == 0:
        print("No frames found! Run extract_frames.py first.")
        sys.exit(1)

    # ── Step 2: Run DINO on every frame ───────────────────────────────────────
    print(f"\n── Step 2: Computing DINO embeddings on {DEVICE}...")
    print(f"  This may take a few minutes...\n")

    all_embeddings = []
    valid_paths    = []

    for i, path in enumerate(frame_paths):
        emb = get_embedding(path)
        if emb is not None:
            all_embeddings.append(emb)
            valid_paths.append(path)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(frame_paths)} frames...")

    print(f"  Done - {len(valid_paths)} valid frames embedded")

    # ── Step 3: Cluster similar frames together ────────────────────────────────
    n_clusters = min(DEFAULT_CLUSTERS, len(valid_paths))
    print(f"\n── Step 3: Clustering {len(valid_paths)} frames into {n_clusters} groups...")

    embeddings_matrix = np.stack(all_embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings_matrix)
    print(f"  Clustering complete")

    # ── Step 4: Pick best frame from each cluster ──────────────────────────────
    print(f"\n── Step 4: Selecting most representative frame per group...")
    selected_paths = []

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        centre          = kmeans.cluster_centers_[cluster_id]
        distances       = [
            np.linalg.norm(all_embeddings[j] - centre)
            for j in cluster_indices
        ]
        best_global_idx = cluster_indices[np.argmin(distances)]
        selected_paths.append(valid_paths[best_global_idx])

    print(f"  Selected {len(selected_paths)} diverse frames")

    # ── Step 5: Process frames based on mode ──────────────────────────────────
    print(f"\n── Step 5: Processing frames in mode='{args.mode}'...")

    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Set up Django ORM only if we need the database
    if args.mode in ("annotator", "both"):
        sys.path.insert(0, PROJECT_ROOT)
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AquaSightWeb.settings")
        import django
        django.setup()
        from annotator.models import AnalysisImage

    created_count = 0

    for i, frame_path in enumerate(selected_paths):
        filename     = f"frame_{uuid.uuid4().hex}.jpg"
        storage_path = f"analysis_images/{filename}"

        # Upload to Supabase Storage
        if args.mode in ("db", "both"):
            try:
                upload_frame(sb, frame_path, filename)
            except Exception as e:
                print(f"  WARNING: Upload failed for {filename} - {e}")
                continue

        # Save record to Django database (appears in annotation workspace)
        if args.mode in ("annotator", "both"):
            AnalysisImage.objects.create(
                image=storage_path,
                original_filename=os.path.basename(frame_path),
                status="bound",
            )

        created_count += 1

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(selected_paths)} frames...")

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n✓ Pipeline complete!")
    print(f"  Mode             : {args.mode}")
    print(f"  Frames processed : {created_count}")

    if args.mode in ("db", "both"):
        print(f"  Uploaded to      : Supabase Storage / {STORAGE_BUCKET} / analysis_images/")
    if args.mode in ("annotator", "both"):
        print(f"  Saved to         : Django database (annotator_analysisimage table)")
        print(f"  Open             : http://127.0.0.1:8000/annotator/ to start annotating")


if __name__ == "__main__":
    main()