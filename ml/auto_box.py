"""Uniform tiling box proposer for AquaSite.

Strategy
--------
Earlier versions used PCA on DINOv3 patch tokens to separate foreground
from background. On underwater frames PC1 typically tracks illumination
or depth-from-camera rather than figure-ground, and the resulting boxes
were unreliable — usually one giant half-of-image rectangle rather than
useful per-object proposals.

We now cover every image with a uniform NxN grid of tiles. The reviewer
sees predictable, full-frame coverage; the classifier produces a label
per tile; the review queue surfaces the uncertain / rare-class ones for
human attention.

Pros: zero failure modes, no hidden assumptions, fast (no DINOv3 forward
needed inside this module — classification happens downstream in
ml.bridge).
Cons: tiles are grid-aligned rectangles, not object-shaped. For real
instance segmentation, SAM/FastSAM is the upgrade path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

# (x, y, w, h) in percent 0-100 — the AnalysisImage / BoundingBox frame coord system.
CropPct = Tuple[float, float, float, float]


@dataclass
class AutoBoxConfig:
    grid: int = 5             # NxN grid covering the full frame (5 → 25 tiles)
    padding: float = 0.0      # optional context padding per tile (fraction of tile side)
    max_boxes: int = 50       # safety cap; ignored unless grid*grid exceeds it

    # Retained for backwards-compatibility with older configs. Currently unused.
    input_size: int = 512
    min_patches: int = 2
    morph_close: int = 1
    dense_threshold: float = 0.40
    subdivision_grid: int = 4


def propose_boxes(
    source=None,
    backbone=None,
    cfg: AutoBoxConfig | None = None,
) -> List[CropPct]:
    """Return an NxN grid of bounding boxes covering the full frame.

    Args:
        source: unused (kept for API compatibility with the old proposer).
        backbone: unused (no DINOv3 forward needed for tiling).
        cfg: tuning knobs. Defaults to AutoBoxConfig().

    Returns:
        List of (x, y, w, h) in percent 0-100, raster order (row-major
        top-to-bottom, left-to-right).
    """
    cfg = cfg or AutoBoxConfig()
    n = max(1, int(cfg.grid))
    cell_w = 100.0 / n
    cell_h = 100.0 / n
    pad = max(0.0, float(cfg.padding))

    boxes: List[CropPct] = []
    for i in range(n):
        for j in range(n):
            x = j * cell_w
            y = i * cell_h
            w = cell_w
            h = cell_h
            if pad > 0:
                pad_x = w * pad
                pad_y = h * pad
                x = max(0.0, x - pad_x)
                y = max(0.0, y - pad_y)
                w = min(100.0 - x, w + 2 * pad_x)
                h = min(100.0 - y, h + 2 * pad_y)
            boxes.append((x, y, w, h))

    return boxes[: cfg.max_boxes]
