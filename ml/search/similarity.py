"""
Cosine-similarity index over the DINOv3 embeddings.

This is the substrate for:
  - "find more like this" in the annotator UI
  - cross-frame grouping / near-duplicate detection
  - the kNN classifier head (once labels exist)
  - the active-learning review queue
  - text/concept search later (text embedding -> same query path)

Backed by a single numpy matrix loaded from ml/outputs/embeddings/embeddings.npy
plus filenames.json. At 22k x 768 a brute-force cosine search is ~30 ms; we can
swap in FAISS later if the dataset grows past a few hundred thousand vectors.

Usage:
    from ml.search import get_index
    idx = get_index()
    hits = idx.find_similar_by_filename("frame_0176...jpg", k=20)
    # -> [(filename, score), ...]  sorted by score desc, query excluded
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDINGS = REPO_ROOT / "ml" / "outputs" / "embeddings" / "embeddings.npy"
DEFAULT_FILENAMES = REPO_ROOT / "ml" / "outputs" / "embeddings" / "filenames.json"


@dataclass
class Hit:
    filename: str
    score: float
    index: int


class SimilarityIndex:
    """Cosine-similarity search over a fixed embedding matrix."""

    def __init__(
        self,
        embeddings_path: Path = DEFAULT_EMBEDDINGS,
        filenames_path: Path = DEFAULT_FILENAMES,
    ):
        embeddings = np.load(embeddings_path).astype(np.float32)
        filenames: list[str] = json.loads(Path(filenames_path).read_text())

        if len(filenames) != embeddings.shape[0]:
            raise ValueError(
                f"embeddings ({embeddings.shape[0]}) and filenames "
                f"({len(filenames)}) are misaligned"
            )

        # L2-normalise once so cosine similarity = dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = embeddings / norms  # (N, D), unit vectors
        self.filenames = filenames
        self._index_of = {fn: i for i, fn in enumerate(filenames)}

    # ---- public API --------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self._matrix.shape

    def get_embedding(self, filename: str) -> np.ndarray:
        """Return the unit-normalised embedding for a known filename."""
        i = self._index_of.get(filename)
        if i is None:
            raise KeyError(f"filename not in index: {filename}")
        return self._matrix[i]

    def find_similar(
        self,
        query: np.ndarray,
        k: int = 20,
        exclude_indices: Iterable[int] = (),
    ) -> list[Hit]:
        """Top-k nearest neighbours to a query vector (any 768-d embedding)."""
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(q)
        if n == 0:
            return []
        q = q / n

        scores = self._matrix @ q  # (N,)
        if exclude_indices:
            scores = scores.copy()
            for i in exclude_indices:
                scores[i] = -np.inf

        # argpartition is O(N), then sort just the top-k
        if k >= len(scores):
            top = np.argsort(-scores)
        else:
            top_unsorted = np.argpartition(-scores, k)[:k]
            top = top_unsorted[np.argsort(-scores[top_unsorted])]

        return [
            Hit(filename=self.filenames[i], score=float(scores[i]), index=int(i))
            for i in top
            if scores[i] != -np.inf
        ]

    def find_similar_by_filename(
        self,
        filename: str,
        k: int = 20,
        include_self: bool = False,
    ) -> list[Hit]:
        """Top-k frames most similar to a known frame in the index."""
        i = self._index_of.get(filename)
        if i is None:
            raise KeyError(f"filename not in index: {filename}")
        exclude = () if include_self else (i,)
        return self.find_similar(self._matrix[i], k=k, exclude_indices=exclude)


# ---- module-level singleton (cheap lazy load) -----------------------------

_lock = threading.Lock()
_singleton: SimilarityIndex | None = None


def get_index() -> SimilarityIndex:
    """Lazy-loaded process-wide similarity index."""
    global _singleton
    if _singleton is None:
        with _lock:
            if _singleton is None:
                _singleton = SimilarityIndex()
    return _singleton
