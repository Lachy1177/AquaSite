"""
One-time script to compute approximate DINOv2 cluster centroids from existing
AnalysisImage records and save them to ml/cluster_centroids.npz.

Run from the project root after activating the virtual environment:
    python scripts/save_cluster_centroids.py

The output file is used by annotator/cluster_utils.py to assign clusters to
newly uploaded images in a background thread.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AquaSightWeb.settings")

import django
django.setup()

import numpy as np
import httpx
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoImageProcessor, AutoModel
from annotator.models import AnalysisImage

CENTROID_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ml", "cluster_centroids.npz"
)
MODEL_NAME = "facebook/dinov2-small"


def get_embedding(processor, model, image_url):
    resp = httpx.get(image_url, timeout=30)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0].squeeze().numpy()


def main():
    os.makedirs(os.path.dirname(CENTROID_PATH), exist_ok=True)

    print(f"Loading {MODEL_NAME}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.")

    cluster_ids = list(
        AnalysisImage.objects
        .exclude(cluster_id__isnull=True)
        .values_list("cluster_id", flat=True)
        .distinct()
        .order_by("cluster_id")
    )
    print(f"Found {len(cluster_ids)} clusters.")

    centroids, ids = [], []
    for cid in cluster_ids:
        rep = AnalysisImage.objects.filter(cluster_id=cid).exclude(image_url="").first()
        if not rep:
            print(f"  cluster {cid}: no image, skipping")
            continue
        try:
            emb = get_embedding(processor, model, rep.image_url)
            centroids.append(emb)
            ids.append(cid)
            print(f"  cluster {cid}: OK")
        except Exception as e:
            print(f"  cluster {cid}: SKIP ({e})")

    np.savez(CENTROID_PATH, centroids=np.array(centroids), cluster_ids=np.array(ids))
    print(f"\nSaved {len(ids)} centroids to {CENTROID_PATH}")


if __name__ == "__main__":
    main()
