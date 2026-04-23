"""Bridge between the ML pipeline and Django views.

Provides functions callable from annotator/views.py to run
DINOv3 inference on uploaded images.
"""

import io
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image

from ml.data.config import load_config
from ml.data.transforms import get_eval_transform
from ml.models.backbone import DINOv3Backbone
from ml.models.classifier import LinearClassifier, load_checkpoint

# (x, y, width, height) in percent 0-100 — the AnalysisImage/BoundingBox frame coord system.
CropPct = Tuple[float, float, float, float]

# Module-level cache to avoid reloading models on every request
_cached_backbone: Optional[DINOv3Backbone] = None
_cached_classifier: Optional[LinearClassifier] = None
_cached_class_names: Optional[List[str]] = None
_cached_config = None


def _ensure_models_loaded():
    """Load backbone and classifier once, cache for reuse."""
    global _cached_backbone, _cached_classifier, _cached_class_names, _cached_config

    if _cached_backbone is not None and _cached_classifier is not None:
        return

    config = load_config()
    _cached_config = config

    _cached_backbone = DINOv3Backbone(
        model_name=config.model.name,
        device=config.training.device,
        hf_token=config.hf_token,
    )

    # Load the best checkpoint if it exists
    best_ckpt = os.path.join(config.training.checkpoint_dir, "classifier_best.pt")
    if os.path.exists(best_ckpt):
        _cached_classifier, meta = load_checkpoint(
            best_ckpt, device=config.training.device,
        )
        _cached_class_names = meta.get("class_names", [])
        print(f"Loaded classifier: {len(_cached_class_names)} classes, "
              f"best F1={meta.get('best_val_f1', 0):.4f}")
    else:
        _cached_classifier = None
        _cached_class_names = None
        print("No trained classifier found. Only embeddings available.")


def _open_image(source: Union[str, bytes]) -> Image.Image:
    """Accept either a filesystem path or raw image bytes."""
    if isinstance(source, (bytes, bytearray)):
        return Image.open(io.BytesIO(source)).convert("RGB")
    return Image.open(source).convert("RGB")


def run_dino_inference(source: Union[str, bytes]) -> List[Dict]:
    """Run DINOv3 classification on a single image.

    Args:
        source: Either an absolute path to a local image file, or raw
            image bytes (e.g. downloaded from Supabase Storage).

    Returns:
        List of dicts: [{"label": str, "confidence": float}, ...]
        sorted by confidence descending. Returns empty list if no
        classifier is trained yet.
    """
    _ensure_models_loaded()

    if _cached_classifier is None:
        return []

    config = _cached_config
    transform = get_eval_transform(config.model.input_size)

    img = _open_image(source)
    tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

    embedding = _cached_backbone.extract(tensor)
    logits = _cached_classifier(embedding.to(config.training.device))
    probs = torch.softmax(logits, dim=1)[0]

    results = []
    for idx, prob in enumerate(probs):
        if idx < len(_cached_class_names):
            results.append({
                "label": _cached_class_names[idx],
                "confidence": round(prob.item(), 4),
            })

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


def _crop_by_pct(img: Image.Image, crop_pct: CropPct) -> Image.Image:
    """Crop a PIL image by percent-based (x, y, w, h). Matches AquaSiteBoxDataset.load_raw."""
    W, H = img.size
    x, y, w, h = crop_pct
    left = max(0, int(round(x / 100.0 * W)))
    top = max(0, int(round(y / 100.0 * H)))
    right = min(W, int(round((x + w) / 100.0 * W)))
    bottom = min(H, int(round((y + h) / 100.0 * H)))
    if right - left < 2 or bottom - top < 2:
        return img
    return img.crop((left, top, right, bottom))


def _format_top_k(probs: torch.Tensor, top_k: int) -> Dict:
    """Shape a probability vector into {label, confidence, top_k} for the UI."""
    names = _cached_class_names or []
    pairs = [(names[i], round(probs[i].item(), 4)) for i in range(len(names))]
    pairs.sort(key=lambda p: p[1], reverse=True)
    top = pairs[:top_k]
    return {
        "label": top[0][0] if top else None,
        "confidence": top[0][1] if top else 0.0,
        "top_k": [{"label": l, "confidence": c} for l, c in top],
    }


def run_dino_inference_on_crop(
    source: Union[str, bytes],
    crop_pct: CropPct,
    top_k: int = 3,
) -> Optional[Dict]:
    """Classify a single bounding-box crop of a frame.

    Args:
        source: frame image as a path or raw bytes.
        crop_pct: (x, y, w, h) in percent 0-100.
        top_k: how many top predictions to return.

    Returns:
        {"label": str, "confidence": float, "top_k": [{label, confidence}, ...]}
        or None if no classifier is trained yet.
    """
    _ensure_models_loaded()
    if _cached_classifier is None:
        return None

    config = _cached_config
    transform = get_eval_transform(config.model.input_size)

    img = _open_image(source)
    crop = _crop_by_pct(img, crop_pct)
    tensor = transform(crop).unsqueeze(0)

    embedding = _cached_backbone.extract(tensor)
    logits = _cached_classifier(embedding.to(config.training.device))
    probs = torch.softmax(logits, dim=1)[0]
    return _format_top_k(probs, top_k)


def run_dino_inference_on_crops(
    source: Union[str, bytes],
    crops: Sequence[Tuple[str, CropPct]],
    top_k: int = 3,
) -> List[Tuple[str, Dict]]:
    """Batched classification for many crops of the same frame.

    Downloads / opens the frame once, crops N times, runs a single
    forward pass.

    Args:
        source: frame image as a path or raw bytes.
        crops: iterable of (id, (x, y, w, h)) pairs. `id` is passed through.
        top_k: predictions per crop.

    Returns list of (id, result_dict). Empty list if no classifier.
    """
    _ensure_models_loaded()
    if _cached_classifier is None:
        return []

    crops = list(crops)
    if not crops:
        return []

    config = _cached_config
    transform = get_eval_transform(config.model.input_size)

    img = _open_image(source)
    ids: List[str] = []
    tensors: List[torch.Tensor] = []
    for cid, pct in crops:
        ids.append(cid)
        tensors.append(transform(_crop_by_pct(img, pct)))

    batch = torch.stack(tensors)
    embeddings = _cached_backbone.extract(batch)
    logits = _cached_classifier(embeddings.to(config.training.device))
    probs = torch.softmax(logits, dim=1)

    return [(ids[i], _format_top_k(probs[i], top_k)) for i in range(len(ids))]


def get_embedding(source: Union[str, bytes]):
    """Get the raw 768-dim DINOv3 embedding for an image.

    Useful for similarity search from the Django UI. Accepts either a
    filesystem path or raw image bytes. Returns a numpy array of shape
    (768,).
    """
    _ensure_models_loaded()

    config = _cached_config
    transform = get_eval_transform(config.model.input_size)

    img = _open_image(source)
    tensor = transform(img).unsqueeze(0)

    embedding = _cached_backbone.extract(tensor)
    return embedding[0].cpu().numpy()
