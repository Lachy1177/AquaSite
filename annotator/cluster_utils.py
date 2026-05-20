import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

_processor = None
_model = None
_centroids = None
_cluster_ids = None

CENTROID_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ml", "cluster_centroids.npz"
)
MODEL_NAME = "facebook/dinov2-small"


def _load_centroids():
    global _centroids, _cluster_ids
    if _centroids is None:
        if not os.path.exists(CENTROID_PATH):
            return False
        data = np.load(CENTROID_PATH)
        _centroids = data["centroids"]
        _cluster_ids = data["cluster_ids"]
    return True


def _load_model():
    global _processor, _model
    if _model is None:
        from transformers import AutoImageProcessor, AutoModel
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()


def assign_cluster_for_image(image_id, image_url):
    """Compute DINOv2 embedding for image_url and assign nearest cluster.
    Intended to run in a daemon thread after upload."""
    try:
        import torch
        import httpx
        from PIL import Image
        from io import BytesIO

        if not _load_centroids():
            logger.warning(
                "No centroid file at %s — run scripts/save_cluster_centroids.py first",
                CENTROID_PATH,
            )
            return

        _load_model()

        resp = httpx.get(image_url, timeout=30)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        inputs = _processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = _model(**inputs)
        emb = outputs.last_hidden_state[:, 0].squeeze().numpy()

        dists = np.linalg.norm(_centroids - emb, axis=1)
        nearest_cluster = int(_cluster_ids[np.argmin(dists)])

        from annotator.models import AnalysisImage
        AnalysisImage.objects.filter(id=image_id).update(cluster_id=nearest_cluster)
        logger.info("Assigned cluster %d to image %s", nearest_cluster, image_id)

    except Exception:
        logger.exception("Background cluster assignment failed for image %s", image_id)
