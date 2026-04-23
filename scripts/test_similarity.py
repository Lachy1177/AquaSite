"""
Quick smoke test for the similarity index.

Run from repo root:
  python scripts/test_similarity.py
  python scripts/test_similarity.py frame_01762662287260768800.jpg
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.search import get_index


def main() -> int:
    print("Loading index...")
    t0 = time.time()
    idx = get_index()
    print(f"  loaded {idx.shape[0]} embeddings of dim {idx.shape[1]} in {time.time()-t0:.2f}s")

    query = sys.argv[1] if len(sys.argv) > 1 else idx.filenames[0]
    print(f"\nQuery: {query}")

    t0 = time.time()
    hits = idx.find_similar_by_filename(query, k=10)
    print(f"  search took {(time.time()-t0)*1000:.1f} ms\n")

    print("Top 10 most similar frames:")
    for rank, h in enumerate(hits, 1):
        print(f"  {rank:>2}. {h.score:.4f}  {h.filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
